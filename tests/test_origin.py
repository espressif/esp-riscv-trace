# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Register origin walks over made-up native events.

The walks cover calls, loops, traps, untraced code and memory, and what the native event types change: boundaries,
warnings, lost and untraced gaps, unreadable code, image errors, the scheduler name and bus access indexes.
"""

from __future__ import annotations

from collections.abc import Callable, Collection

import pytest
from helpers import APPS, TESTS

from esp_riscv_trace import FaultWindow, FaultWindowBuilder, HistoryBuilder, open_capture
from esp_riscv_trace.analysis.origin import (
    MAX_STEPS,
    MAX_VALUE_DEPTH,
    WINDOW_EVENTS,
    BranchOutcome,
    BusAccess,
    Crossing,
    RegisterStep,
    Relation,
    StackCall,
    StepKind,
    StopReason,
)
from esp_riscv_trace.decoder.reconstruct import GapReason
from esp_riscv_trace.elf import CodeReadError
from esp_riscv_trace.ir.events import (
    InsnClass,
    LinkKind,
    TraceBoundary,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
    TraceWarning,
)

RA, SP, T0, S0, A0, A1, A3, A4, A5, A6 = 1, 2, 5, 8, 10, 11, 13, 14, 15, 16
MAIN, FOO, OTHER, HANDLER, SCHEDULER = 0x1000, 0x2000, 0x3000, 0x4000, 0x6000
NAMES = {MAIN: "main", FOO: "foo", OTHER: "task_b", HANDLER: "isr", SCHEDULER: "vTaskSwitchContext"}
LOAD_FAULT, STORE_FAULT, FETCH_FAULT, TIMER, M_ECALL = 5, 7, 1, 7, 11
STACK = 0x2F019BC0
MRET = 0x30200073
# Bits 6..0 of 0b0111111 mark an encoding wider than 32 bits, which does not decode
WIDE = 0x0000003F


def _i(opcode: int, rd: int, funct3: int, rs1: int, imm: int) -> int:
    return (imm & 0xFFF) << 20 | rs1 << 15 | funct3 << 12 | rd << 7 | opcode


def addi(rd: int, rs1: int, imm: int) -> int:
    return _i(0x13, rd, 0, rs1, imm)


def li(rd: int, imm: int) -> int:
    return addi(rd, 0, imm)


def mv(rd: int, rs: int) -> int:
    return addi(rd, rs, 0)


def add(rd: int, rs1: int, rs2: int) -> int:
    return rs2 << 20 | rs1 << 15 | rd << 7 | 0x33


def lw(rd: int, rs1: int, imm: int) -> int:
    return _i(0x03, rd, 2, rs1, imm)


def jalr(rd: int, rs1: int, imm: int = 0) -> int:
    return _i(0x67, rd, 0, rs1, imm)


def sw(rs2: int, rs1: int, imm: int) -> int:
    return (imm >> 5 & 0x7F) << 25 | rs2 << 20 | rs1 << 15 | 2 << 12 | (imm & 0x1F) << 7 | 0x23


def jal(rd: int, offset: int) -> int:
    imm = offset & 0x1FFFFF
    return (
        (imm >> 20 & 1) << 31
        | (imm >> 1 & 0x3FF) << 21
        | (imm >> 11 & 1) << 20
        | (imm >> 12 & 0xFF) << 12
        | rd << 7
        | 0x6F
    )


def branch(funct3: int, rs1: int, rs2: int, offset: int) -> int:
    imm = offset & 0x1FFF
    return (
        (imm >> 12 & 1) << 31
        | (imm >> 5 & 0x3F) << 25
        | rs2 << 20
        | rs1 << 15
        | funct3 << 12
        | (imm >> 1 & 0xF) << 8
        | (imm >> 11 & 1) << 7
        | 0x63
    )


def cm_stack(stack_op: int) -> int:
    """A Zcmp stack instruction of {ra} with no extra stack adjustment."""
    return 0b101 << 13 | stack_op << 8 | 4 << 4 | 0b10


CM_PUSH = cm_stack(0b11000)
CM_POPRET = cm_stack(0b11110)


class Code:
    """The code the made-up events ran, by pc, with the function of each 4 KiB range named."""

    def __init__(self) -> None:
        self.words: dict[int, int] = {}
        self.names = dict(NAMES)

    def read(self, vaddr: int, size: int) -> bytes:
        word = self.words.get(vaddr)
        if word is None:
            raise CodeReadError(f"0x{vaddr:08x} has no code")
        return (word & ((1 << 8 * size) - 1)).to_bytes(size, "little")

    def symbol_at(self, vaddr: int) -> str | None:
        return self.names.get(vaddr & ~0xFFF)


def window_of(events: list[TraceEvent], image: Code | None, *, end_interrupts: Collection[int] = ()) -> FaultWindow:
    builder = FaultWindowBuilder(image, end_interrupts=end_interrupts)
    for event in events:
        builder.push(event)
    return builder.close()


class Run:
    """A made-up trace of the crashed core, with the code it ran."""

    def __init__(self, start: int = MAIN) -> None:
        self.code = Code()
        self.events: list[TraceEvent] = [TraceSync(start, 3, None)]

    def block(self, pc: int, *words: int) -> Run:
        """Instructions that retire one after the other from pc on."""
        for word in words:
            size = 4 if word & 3 == 3 else 2
            self.jump(pc, word, LinkKind.NONE, pc + size)
            pc += size
        return self

    def jump(self, pc: int, word: int | None, link: LinkKind, to: int | None) -> Run:
        """An instruction that goes to ``to``. ``None`` as ``word`` leaves its code out of the image."""
        if word is not None:
            self.code.words[pc] = word
        size = 2 if word is not None and word & 3 != 3 else 4
        self.events.append(TraceInstruction(pc, size, InsnClass.NONE, None, to, None, link=link))
        return self

    def trap(self, cause: int, epc: int, *, interrupt: bool = False) -> Run:
        self.events.append(TraceTrap(cause, interrupt, None, None, None, None, epc=epc))
        return self

    def fault(self, cause: int, pc: int, word: int | None = None) -> Run:
        if word is not None:
            self.code.words[pc] = word
        return self.trap(cause, pc)

    def add(self, event: TraceEvent) -> Run:
        self.events.append(event)
        return self

    def origin(self, registers: dict[int, int]) -> RegisterStep:
        root = window_of(self.events, self.code).origin(registers)
        assert root is not None
        return root


def gap(reason: str) -> TraceGap:
    return TraceGap(reason, "", None)


def sync(pc: int) -> TraceSync:
    return TraceSync(pc, 3, None)


def chain(root: RegisterStep) -> list[tuple[int, StepKind, int | None, int | None]]:
    return [(s.reg, s.kind, s.pc, s.value) for s in root.walk()]


# Made-up runs, each with the registers the trap handler saved


def callee_return() -> tuple[Run, dict[int, int]]:
    """The value came back from a function that has returned."""
    run = (
        Run()
        .jump(MAIN, jal(RA, FOO - MAIN), LinkKind.CALL, FOO)
        .block(FOO, li(A0, 0))
        .jump(FOO + 4, jalr(0, RA), LinkKind.RETURN, MAIN + 4)
        .block(MAIN + 4, mv(A5, A0))
        .fault(LOAD_FAULT, MAIN + 8, lw(A4, A5, 0))
    )
    return run, {A0: 0, A5: 0}


def interrupted(resume: int = MAIN + 4, *, nested: bool = False) -> tuple[Run, dict[int, int]]:
    """a5 is set, then a timer interrupt saves, changes and restores a5, and returns to resume."""
    run = Run().block(MAIN, li(A5, 0)).trap(TIMER, MAIN + 4, interrupt=True).block(HANDLER, sw(A5, SP, 0), li(A5, 1))
    if nested:
        run.trap(TIMER, HANDLER + 8, interrupt=True).block(HANDLER + 0x100, li(A5, 2))
        run.jump(HANDLER + 0x104, MRET, LinkKind.TRAP_RETURN, HANDLER + 8)
    run.block(HANDLER + 8, lw(A5, SP, 0)).jump(HANDLER + 12, MRET, LinkKind.TRAP_RETURN, resume)
    return run.fault(LOAD_FAULT, resume, lw(A4, A5, 0)), {A5: 0, SP: STACK}


def scheduled() -> tuple[Run, dict[int, int]]:
    """A timer interrupt calls the scheduler and returns to where it was taken."""
    run = Run().block(MAIN, li(A5, 0)).trap(TIMER, MAIN + 4, interrupt=True)
    run.jump(HANDLER, jal(RA, SCHEDULER - HANDLER), LinkKind.CALL, SCHEDULER)
    run.jump(SCHEDULER, jalr(0, RA), LinkKind.RETURN, HANDLER + 4)
    run.jump(HANDLER + 4, MRET, LinkKind.TRAP_RETURN, MAIN + 4)
    return run.fault(LOAD_FAULT, MAIN + 4, lw(A4, A5, 0)), {A5: 0}


def untraced_call(base: int = S0) -> tuple[Run, dict[int, int]]:
    """Code calls through t0 into untraced code, like a ROM function, which returns."""
    run = Run().block(MAIN, li(S0, 8), li(A5, 9)).jump(MAIN + 8, jalr(RA, T0), LinkKind.CALL, None)
    run.add(gap(GapReason.UNTRACED)).add(sync(MAIN + 12)).block(MAIN + 12, li(A6, 0))
    return run.fault(LOAD_FAULT, MAIN + 16, lw(A4, base, 0)), {S0: 8, A5: 9}


def popret() -> tuple[Run, dict[int, int]]:
    """A store to 12(sp) after cm.push {ra},-16 overwrites the saved ra, and cm.popret returns to it."""
    run = Run().jump(MAIN, jal(RA, FOO - MAIN), LinkKind.CALL, FOO).block(FOO, CM_PUSH, li(A5, 4), sw(A5, SP, 12))
    run.jump(FOO + 10, CM_POPRET, LinkKind.RETURN, 0x4)
    return run.fault(FETCH_FAULT, 0x4), {RA: 0x4, A5: 4, SP: STACK}


def stack() -> tuple[Run, dict[int, int]]:
    """main lowers sp by 32 and calls foo, which lowers it by 16. A callee and an interrupt in between leave sp."""
    run = (
        Run()
        .block(MAIN, addi(SP, SP, -32))
        .jump(MAIN + 4, jal(RA, FOO - MAIN - 4), LinkKind.CALL, FOO)
        .block(FOO, addi(SP, SP, -16))
        .jump(FOO + 4, jal(RA, OTHER - FOO - 4), LinkKind.CALL, OTHER)
        .block(OTHER, addi(SP, SP, -8), addi(SP, SP, 8))
        .jump(OTHER + 8, jalr(0, RA), LinkKind.RETURN, FOO + 8)
        .trap(TIMER, FOO + 8, interrupt=True)
        .block(HANDLER, addi(SP, SP, -64))
        .jump(HANDLER + 4, MRET, LinkKind.TRAP_RETURN, FOO + 8)
        .fault(STORE_FAULT, FOO + 8, sw(A5, A4, 0))
    )
    return run, {SP: STACK}


def branches() -> tuple[Run, dict[int, int]]:
    """foo takes the branch at its start and not the next one, then faults."""
    run = Run().jump(MAIN, jal(RA, FOO - MAIN), LinkKind.CALL, FOO)
    run.jump(FOO, branch(0, A0, A1, 8), LinkKind.NONE, FOO + 8)
    run.jump(FOO + 8, branch(0, A0, A1, 8), LinkKind.NONE, FOO + 12).block(FOO + 12, addi(A5, A5, 1))
    return run.fault(STORE_FAULT, FOO + 16, sw(A5, A4, 0)), {A5: 1}


def bus_error() -> tuple[Run, dict[int, int]]:
    """A load from a5 retires, and the bus error it caused is taken two instructions later."""
    run = Run().block(MAIN, li(A5, 0x40), lw(A4, A5, 0), li(A6, 0), li(A1, 0))
    return run.trap(LOAD_FAULT, MAIN + 16), {A5: 0x40}


def loop(passes: int) -> Run:
    """a5 grows by 4 on each pass of a loop and is used after it."""
    run = Run().block(MAIN, li(A5, 0), li(A6, 4 * passes))
    for n in range(passes):
        run.block(MAIN + 8, addi(A5, A5, 4))
        run.jump(MAIN + 12, branch(1, A5, A6, -4), LinkKind.NONE, MAIN + 8 if n < passes - 1 else MAIN + 16)
    return run.fault(LOAD_FAULT, MAIN + 16, lw(A4, A5, 0))


def slot(*between: int, value: int = 4) -> Run:
    """p is a volatile local, so 4 goes through 12(sp) before the store through it faults."""
    run = Run(FOO).block(FOO, addi(SP, SP, -16), li(A5, value), sw(A5, SP, 12))
    pc = FOO + 12
    for word in between:
        run.block(pc, word)
        pc += 4
    return run.block(pc, lw(A5, SP, 12)).fault(STORE_FAULT, pc + 4, sw(0, A5, 0))


def in_foo(pc: int) -> bool:
    return pc & ~0xFFF == FOO


# Walks across calls, traps and memory


def test_callee_return_value() -> None:
    run, registers = callee_return()
    root = run.origin(registers)
    assert chain(root) == [(A5, StepKind.COPIED, MAIN + 4, 0), (A0, StepKind.SET, FOO, 0)]
    assert (root.relation, root.sources[0].relation, root.sources[0].instruction) == (None, Relation.CALLEE, "li a0,0")
    assert root.use_pc == MAIN + 8


def test_caller_argument() -> None:
    run = Run().block(MAIN, li(A0, 4)).jump(MAIN + 4, jal(RA, FOO - MAIN - 4), LinkKind.CALL, FOO)
    root = run.block(FOO, li(A1, 1)).fault(LOAD_FAULT, FOO + 4, lw(A4, A0, 0)).origin({A0: 4})
    assert chain(root) == [(A0, StepKind.SET, MAIN, 4)]
    assert root.relation == Relation.CALLER


def test_walk_lists_each_step_once_before_its_sources() -> None:
    run = Run().block(MAIN, li(A3, 1), li(A4, 2), add(A5, A3, A4)).fault(LOAD_FAULT, MAIN + 12, lw(A6, A5, 0))
    root = run.origin({A5: 3})
    assert chain(root) == [
        (A5, StepKind.COMPUTED, MAIN + 8, 3),
        (A3, StepKind.SET, MAIN, 1),
        (A4, StepKind.SET, MAIN + 4, 2),
    ]
    assert len({id(step) for step in root.walk()}) == 3


def test_a_long_chain_ends_at_the_step_limit() -> None:
    run = Run().block(MAIN, li(A5, 0))
    for n in range(MAX_STEPS + 2):
        run.block(MAIN + 4 + 4 * n, addi(A5, A5, 4))
    root = run.fault(LOAD_FAULT, MAIN + 4 * (MAX_STEPS + 3), lw(A4, A5, 0)).origin({})
    steps = list(root.walk())
    assert len(steps) == MAX_STEPS + 1
    assert (steps[-1].kind, steps[-1].stop) == (StepKind.UNKNOWN, StopReason.STEPS)


def test_last_loop_pass_and_the_passes_before_it() -> None:
    root = loop(3).origin({A5: 12, A6: 12})
    assert chain(root) == [
        (A5, StepKind.COMPUTED, MAIN + 8, 12),
        (A5, StepKind.COMPUTED, MAIN + 8, 8),
        (A5, StepKind.COMPUTED, MAIN + 8, 4),
        (A5, StepKind.SET, MAIN, 0),
    ]


def test_popret_loads_the_bad_return_address_cm_push_saved() -> None:
    """A store to 12(sp) after cm.push {ra},-16 overwrites the saved ra, and cm.popret returns to it."""
    run, registers = popret()
    root = run.origin(registers)
    assert chain(root) == [(RA, StepKind.LOADED, FOO + 10, 0x4), (A5, StepKind.SET, FOO + 2, 4)]
    assert (root.address, root.store_pc, root.store_instruction) == (STACK - 4, FOO + 6, "sw a5,12(sp)")


def test_load_from_a_stack_slot_goes_on_to_its_store() -> None:
    root = slot().origin({A5: 4, SP: STACK})
    assert chain(root) == [(A5, StepKind.LOADED, FOO + 12, 4), (A5, StepKind.SET, FOO + 4, 4)]
    assert (root.address, root.store_pc, root.store_instruction, root.crossed) == (
        STACK + 12,
        FOO + 8,
        "sw a5,12(sp)",
        (),
    )
    assert root.sources[0].use_pc == FOO + 8


@pytest.mark.parametrize(
    ("run", "store_pc"),
    [
        (slot(sw(A6, SP, 8)), FOO + 8),
        (slot(li(A6, 4), sw(A6, SP, 12)), FOO + 16),
        (slot(sw(A6, A4, 0)), None),
        (slot(value=5), None),
    ],
    ids=["other_slot", "same_slot_later", "unknown_address", "other_value"],
)
def test_stores_between_a_load_and_its_slot(run: Run, store_pc: int | None) -> None:
    """A store to an address the trace does not show could have written the slot, and so could one of another value."""
    root = run.origin({A5: 4, SP: STACK})
    assert (root.kind, root.address, root.store_pc) == (StepKind.LOADED, STACK + 12, store_pc)
    assert len(root.sources) == (store_pc is not None)


def test_constants_are_found_without_registers() -> None:
    """Values the code computes from constants are known, and a load through an unknown sp is not followed."""
    run, _ = callee_return()
    assert chain(run.origin({})) == [(A5, StepKind.COPIED, MAIN + 4, 0), (A0, StepKind.SET, FOO, 0)]
    root = slot().origin({})
    assert chain(root) == [(A5, StepKind.LOADED, FOO + 12, None)]
    assert (root.address, root.store_pc) == (None, None)


@pytest.mark.parametrize("written_later", [False, True], ids=["not_saved", "written_later"])
def test_stack_adjustments_are_worked_back_to_a_constant(written_later: bool) -> None:
    """sp is set, moved down twice and back up, then copied. The frame does not give sp at the copy."""
    run = Run().block(MAIN, li(SP, 0x100), addi(SP, SP, -16), addi(SP, SP, -16), addi(SP, SP, 32), mv(A5, SP))
    if written_later:
        run.block(MAIN + 20, mv(SP, A0))
    pc = MAIN + 24 if written_later else MAIN + 20
    root = run.fault(LOAD_FAULT, pc, lw(A4, A5, 0)).origin({})
    assert chain(root)[0] == (A5, StepKind.COPIED, MAIN + 16, 0x100)


@pytest.mark.parametrize("length", range(MAX_VALUE_DEPTH - 8, MAX_VALUE_DEPTH + 2))
def test_value_beyond_the_depth_of_one_source_is_found_for_another(length: int) -> None:
    """a1 adds a long chain to a0 and a2 copies a0. The walk of a1 reaches a0 too deep, a2 reaches it near the top."""
    a2 = 12
    run = Run().block(MAIN, li(A0, 0), addi(A0, A0, 1), addi(A0, A0, 1), addi(A0, A0, 1), mv(A1, A0))
    pc = MAIN + 20
    for _ in range(length):
        run.block(pc, addi(A1, A1, 1))
        pc += 4
    run.block(pc, mv(a2, A0), add(A5, A1, a2))
    root = run.fault(LOAD_FAULT, pc + 8, lw(A4, A5, 0)).origin({})
    assert next(step for step in root.walk() if step.reg == a2).value == 3


@pytest.mark.parametrize("nested", [False, True], ids=["interrupt", "nested"])
def test_interrupt_that_returns_to_where_it_was_taken_is_crossed(nested: bool) -> None:
    run, registers = interrupted(nested=nested)
    root = run.origin(registers)
    assert chain(root) == [(A5, StepKind.SET, MAIN, 0)]
    assert root.crossed == (Crossing.INTERRUPT,)


def test_trap_return_to_another_task_stops() -> None:
    run, registers = interrupted(OTHER)
    root = run.origin(registers)
    assert chain(root) == [(A5, StepKind.UNKNOWN, None, 0)]
    assert (root.stop, root.stop_pc) == (StopReason.CONTEXT_SWITCH, OTHER)


@pytest.mark.parametrize("named", [True, False], ids=["scheduler_name", "no_name"])
def test_scheduler_the_image_names_stops_the_walk(named: bool) -> None:
    """Tasks that yield through the same code are preempted at the same instruction, so another can resume there."""
    run, registers = scheduled()
    if not named:
        del run.code.names[SCHEDULER]
    root = run.origin(registers)
    if named:
        assert (chain(root), root.stop, root.stop_pc) == (
            [(A5, StepKind.UNKNOWN, None, 0)],
            StopReason.SCHEDULER,
            MAIN + 4,
        )
    else:
        assert (chain(root), root.crossed) == ([(A5, StepKind.SET, MAIN, 0)], (Crossing.INTERRUPT,))


def test_syscall_result_is_not_carried_back_past_the_ecall() -> None:
    """a5 copies a0 before an ecall whose handler returns 99 in a0, which the code then adds 1 to."""
    run = Run().block(MAIN, mv(A5, A0)).trap(M_ECALL, MAIN + 4).block(HANDLER, li(A0, 99))
    run.jump(HANDLER + 4, MRET, LinkKind.TRAP_RETURN, MAIN + 8).block(MAIN + 8, addi(A0, A0, 1))
    root = run.fault(LOAD_FAULT, MAIN + 12, lw(A4, A5, 0)).origin({A0: 100})
    assert chain(root) == [(A5, StepKind.COPIED, MAIN, None), (A0, StepKind.UNKNOWN, None, None)]


# Native events


def test_callee_saved_register_survives_an_untraced_call() -> None:
    run, registers = untraced_call()
    root = run.origin(registers)
    assert (chain(root), root.crossed) == ([(S0, StepKind.SET, MAIN, 8)], (Crossing.UNTRACED_CALL,))
    run, registers = untraced_call(A5)
    root = run.origin(registers)
    assert (root.kind, root.stop) == (StepKind.UNKNOWN, StopReason.UNTRACED)


@pytest.mark.parametrize("reason", [GapReason.TRACE_LOST, GapReason.PACKET_ERROR])
def test_lost_trace_stops_with_the_gap_reason(reason: GapReason) -> None:
    run = Run().block(MAIN, li(S0, 8)).add(gap(reason)).add(sync(MAIN + 4)).fault(LOAD_FAULT, MAIN + 4, lw(A4, S0, 0))
    root = run.origin({S0: 8})
    assert (root.kind, root.stop, root.detail) == (StepKind.UNKNOWN, StopReason.TRACE_LOST, str(reason))


def test_trace_warning_is_crossed() -> None:
    run = Run().block(MAIN, li(A5, 0)).add(TraceWarning("note", "made up", None)).block(MAIN + 4, li(A6, 0))
    root = run.fault(LOAD_FAULT, MAIN + 8, lw(A4, A5, 0)).origin({A5: 0})
    assert (chain(root), root.crossed) == ([(A5, StepKind.SET, MAIN, 0)], (Crossing.TRACE_WARNING,))


@pytest.mark.parametrize("word", [None, WIDE], ids=["missing", "wider_than_32_bits"])
def test_code_that_cannot_be_read_or_decoded_is_unreadable(word: int | None) -> None:
    run = Run().block(MAIN, li(A5, 0)).jump(MAIN + 4, word, LinkKind.NONE, MAIN + 8)
    root = run.fault(LOAD_FAULT, MAIN + 8, lw(A4, A5, 0)).origin({A5: 0})
    assert (root.kind, root.stop, root.stop_pc) == (StepKind.UNKNOWN, StopReason.UNREADABLE, MAIN + 4)
    assert window_of(run.events, run.code).unsure_writes(MAIN + 4, MAIN + 8) is None


class Failing(Code):
    """Code whose reads raise an error other than ``CodeReadError`` while ``failing``."""

    def __init__(self, code: Code, *, failing: bool = True) -> None:
        super().__init__()
        self.words = code.words
        self.failing = failing

    def read(self, vaddr: int, size: int) -> bytes:
        if self.failing:
            raise RuntimeError("image read failed")
        return super().read(vaddr, size)


def test_image_errors_other_than_code_read_error_end_the_query() -> None:
    run, registers = callee_return()
    window = window_of(run.events, Failing(run.code))
    with pytest.raises(RuntimeError, match="image read failed"):
        window.origin(registers)
    with pytest.raises(RuntimeError, match="image read failed"):
        window.unsure_writes(MAIN + 8, MAIN + 12)


Query = Callable[[list[TraceEvent], Code, dict[int, int]], object]

QUERIES: dict[str, Query] = {
    "origin": lambda events, code, registers: window_of(events, code).origin(registers),
    "stack_calls": lambda events, code, registers: window_of(events, code).stack_calls(registers),
    "branches": lambda events, code, registers: window_of(events, code).branches(in_foo),
}


@pytest.mark.parametrize(
    "build",
    [callee_return, interrupted, lambda: interrupted(nested=True), scheduled, untraced_call, popret, stack, branches],
    ids=["callee_return", "interrupt", "nested", "scheduler", "untraced_call", "popret", "stack", "branches"],
)
def test_boundaries_do_not_change_a_walk(build: Callable[[], tuple[Run, dict[int, int]]]) -> None:
    run, registers = build()
    expected = {name: query(run.events, run.code, registers) for name, query in QUERIES.items()}
    assert any(expected.values())
    for at in range(len(run.events) + 1):
        events = [*run.events[:at], TraceBoundary("wrap", "", None), *run.events[at:]]
        assert {name: query(events, run.code, registers) for name, query in QUERIES.items()} == expected, at


def test_stack_calls_work_the_sp_of_each_call_back_from_the_registers() -> None:
    run, registers = stack()
    window = window_of(run.events, run.code)
    assert window.stack_calls(registers) == (StackCall(MAIN + 4, STACK + 16),)
    assert window.stack_calls({}) == ()


def test_branches_of_the_function_the_fault_is_in() -> None:
    run, _ = branches()
    assert window_of(run.events, run.code).branches(in_foo) == (
        BranchOutcome(FOO, True),
        BranchOutcome(FOO + 8, False),
    )


def test_branch_to_its_fall_through_has_the_outcome_the_trace_reports() -> None:
    """beq a5,a5,+4 goes on to the next instruction whether it jumps or not."""
    run = Run().jump(MAIN, jal(RA, FOO - MAIN), LinkKind.CALL, FOO)
    run.code.words[FOO] = branch(0, A5, A5, 4)
    run.add(TraceInstruction(FOO, 4, InsnClass.BRANCH, True, FOO + 4, None))
    run.block(FOO + 4, addi(A5, A5, 1)).fault(STORE_FAULT, FOO + 8, sw(A5, A4, 0))
    assert window_of(run.events, run.code).branches(in_foo) == (BranchOutcome(FOO, True),)


def test_bus_access_index_is_its_position_in_the_events_pushed() -> None:
    run, registers = bus_error()
    boundary = TraceBoundary("wrap", "", None)
    events = [run.events[0], boundary, run.events[1], boundary, *run.events[2:]]
    window = window_of(events, run.code)
    access = window.bus_access(registers, address=0x40, store=False)
    assert access == BusAccess(MAIN + 4, "lw a4,0(a5)", 4)
    assert events[access.index] is run.events[2]
    root = window.origin(registers, access=access)
    assert root is not None and chain(root) == [(A5, StepKind.SET, MAIN, 0x40)]
    assert window.bus_access(registers, address=0x40, store=True) is None


@pytest.mark.parametrize(
    "access",
    [BusAccess(MAIN + 4, "lw a4,0(a5)", 3), BusAccess(MAIN + 8, "lw a4,0(a5)", 4), BusAccess(MAIN + 4, "", 99)],
    ids=["boundary", "other_pc", "out_of_range"],
)
def test_access_that_is_not_in_the_events_is_rejected(access: BusAccess) -> None:
    run, registers = bus_error()
    boundary = TraceBoundary("wrap", "", None)
    events = [run.events[0], boundary, run.events[1], boundary, *run.events[2:]]
    with pytest.raises(ValueError, match="is not the access"):
        window_of(events, run.code).origin(registers, access=access)


def test_unsure_writes_are_those_of_the_instructions_up_to_saved_mepc() -> None:
    run = Run().block(MAIN, lw(A4, A5, 0), li(A6, 0)).fault(LOAD_FAULT, MAIN + 8)
    window = window_of(run.events, run.code)
    assert window.unsure_writes(MAIN, MAIN + 4) == {A4}
    assert window.unsure_writes(MAIN, MAIN + 8) == {A4, A6}
    assert window.unsure_writes(MAIN, MAIN + 2) is None
    assert window.unsure_writes(MAIN, MAIN + 12) is None


# The window


def ends_late() -> Run:
    """An ecall and an interrupt that both return, a load fault, then events after it."""
    run = Run().block(MAIN, li(A5, 0)).trap(8, MAIN + 4).block(HANDLER, li(A6, 0))
    run.jump(HANDLER + 4, MRET, LinkKind.TRAP_RETURN, MAIN + 8).trap(TIMER, MAIN + 8, interrupt=True)
    run.jump(HANDLER + 8, MRET, LinkKind.TRAP_RETURN, MAIN + 8).fault(LOAD_FAULT, MAIN + 8, lw(A4, A5, 0))
    return run.block(HANDLER + 12, li(A5, 1)).fault(STORE_FAULT, HANDLER + 16, sw(A5, A5, 0))


@pytest.mark.parametrize(("end_interrupts", "end"), [((), 7), ((TIMER,), 5)], ids=["fault", "panic_interrupt"])
def test_window_stops_where_history_stops(end_interrupts: tuple[int, ...], end: int) -> None:
    run = ends_late()
    history = HistoryBuilder(run.code, end_interrupts=end_interrupts)
    builder = FaultWindowBuilder(run.code, end_interrupts=end_interrupts)
    for event in run.events:
        history.push(event)
        builder.push(event)
        assert builder.done == history.done
    window = builder.close()
    assert window.fault is run.events[end]
    assert window.complete
    assert (window.origin({A5: 0}) is None) == (end == 5)


def filled(fill: int, *, boundaries: bool = False, delayed: bool = False) -> Run:
    """a5 is set, then ``fill`` events of a loop, then a load through a5 faults.

    The loop events are instructions, every tenth a boundary when ``boundaries``. When ``delayed``, the load retires
    and its bus error is taken two instructions later.
    """
    run = Run().block(MAIN, li(A5, 0x40))
    run.code.words[MAIN + 0x100] = addi(A6, A6, 1)
    run.code.words[MAIN + 0x104] = branch(1, A6, A1, -4)
    loop_body = [
        TraceInstruction(MAIN + 0x100, 4, InsnClass.NONE, None, MAIN + 0x104, None),
        TraceInstruction(MAIN + 0x104, 4, InsnClass.NONE, None, MAIN + 0x100, None),
    ]
    for n in range(fill):
        run.add(TraceBoundary("wrap", "", None) if boundaries and n % 10 == 9 else loop_body[n % 2])
    if delayed:
        return run.block(MAIN + 4, lw(A4, A5, 0), li(A6, 0), li(A1, 0)).trap(LOAD_FAULT, MAIN + 16)
    return run.fault(LOAD_FAULT, MAIN + 4, lw(A4, A5, 0))


@pytest.mark.parametrize(
    ("extra", "complete", "found"),
    [(0, True, True), (1, False, True), (2, False, False)],
    ids=["all", "sync_dropped", "writer_dropped"],
)
@pytest.mark.parametrize("boundaries", [False, True], ids=["instructions", "boundaries"])
def test_window_keeps_the_last_window_events(extra: int, complete: bool, found: bool, boundaries: bool) -> None:
    """The sync and the write of a5 are the first two events, so the window drops them one by one."""
    run = filled(WINDOW_EVENTS - 3 + extra, boundaries=boundaries)
    assert len(run.events) == WINDOW_EVENTS + extra
    window = window_of(run.events, run.code)
    assert window.complete == complete
    root = window.origin({})
    assert root is not None
    if found:
        assert chain(root) == [(A5, StepKind.SET, MAIN, 0x40)]
    else:
        assert (root.kind, root.stop) == (StepKind.UNKNOWN, StopReason.WINDOW)


def test_bus_access_index_is_its_position_in_the_events_kept() -> None:
    run = filled(WINDOW_EVENTS + 100, boundaries=True, delayed=True)
    window = window_of(run.events, run.code)
    kept = run.events[-WINDOW_EVENTS:]
    registers = {A5: 0x40, SP: STACK}
    access = window.bus_access(registers, address=0x40, store=False)
    assert access is not None
    assert kept[access.index] is run.events[-4]
    root = window.origin(registers, access=access)
    assert root is not None and (root.reg, root.kind, root.stop) == (A5, StepKind.UNKNOWN, StopReason.WINDOW)
    boundary = next(i for i, event in enumerate(kept) if isinstance(event, TraceBoundary))
    with pytest.raises(ValueError, match="is not the access"):
        window.origin(registers, access=BusAccess(MAIN + 4, "", boundary))


def test_window_without_an_end_finds_nothing() -> None:
    empty = FaultWindowBuilder(Code()).close()
    assert (empty.fault, empty.complete) == (None, True)
    run, registers = callee_return()
    builder = FaultWindowBuilder(run.code)
    for event in run.events[:-1]:
        builder.push(event)
    assert not builder.done
    assert window_of(run.events, run.code).unsure_writes(MAIN + 4, MAIN + 8) == {A5}
    for window in (empty, builder.close()):
        assert window.origin(registers) is None
        assert window.bus_access(registers, address=0, store=False) is None
        assert window.stack_calls({SP: STACK}) == ()
        assert window.branches(in_foo) == ()
        assert window.unsure_writes(MAIN + 4, MAIN + 8) is None
    unended = filled(WINDOW_EVENTS)
    assert not window_of(unended.events[:-1], unended.code).complete
    # the trace stops at an interrupt that does not end the window, after calls that have not returned
    run, registers = stack()
    events = run.events[: run.events.index(next(e for e in run.events if isinstance(e, TraceTrap))) + 1]
    assert window_of(events, run.code, end_interrupts=(TIMER,)).stack_calls(registers) != ()
    window = window_of(events, run.code)
    assert window.fault is None
    assert window.stack_calls(registers) == ()


def test_without_an_image_no_code_is_read() -> None:
    run, registers = stack()
    window = window_of(run.events, None)
    assert window.fault is run.events[-1]
    assert window.origin(registers) is None
    assert window.stack_calls(registers) == ()
    assert window.branches(in_foo) == ()
    assert window.unsure_writes(FOO + 8, FOO + 12) is None


def test_query_that_raises_leaves_the_window_usable() -> None:
    run, registers = stack()
    image = Failing(run.code, failing=False)
    window = window_of(run.events, image)
    expected = (window.origin(registers), window.stack_calls(registers))
    image.failing = True
    with pytest.raises(RuntimeError):
        window.origin(registers)
    image.failing = False

    def inside(pc: int) -> bool:
        raise RuntimeError("inside failed")

    with pytest.raises(RuntimeError, match="inside failed"):
        window.branches(inside)
    assert (window.origin(registers), window.stack_calls(registers)) == expected


@pytest.mark.parametrize("target", ["esp32c5", "esp32c6", "esp32s31"])
def test_window_of_a_capture_ends_with_its_history(target: str) -> None:
    """Core 0 of each capture takes an illegal instruction exception. The esp32c6 one comes after 43,375 events.

    The exception is in panic_abort, which has no conditional branch, and it depends on no register.
    """
    faults = 0
    with open_capture(TESTS / target / "coredump.elf", elf=APPS / f"{target}.elf") as capture:
        image = capture.image
        assert image is not None
        for core in capture.cores:
            history = HistoryBuilder(image)
            builder = FaultWindowBuilder(image)
            pushed: list[TraceEvent] = []
            for event in core.events():
                history.push(event)
                builder.push(event)
                pushed.append(event)
                assert builder.done == history.done
                if history.done:
                    break
            window = builder.close()
            closed = history.close()
            fault = closed.fault
            assert (window.fault is None) == (fault is None)
            if window.fault is None or fault is None:
                continue
            faults += 1
            assert (window.fault.epc, window.fault.cause) == (fault.pc, fault.cause)
            assert window.complete == (len(pushed) <= WINDOW_EVENTS)
            assert window.origin({}) is None
            calls = window.stack_calls({SP: STACK})
            assert [call.pc for call in calls] == [frame.pc for frame in closed.stack]
            assert [call.sp for call in calls] == sorted(call.sp for call in calls)

            def inside(pc: int, name: str | None = image.symbol_at(window.fault.epc or 0)) -> bool:
                return image.symbol_at(pc) == name

            assert window.branches(inside) == ()
    assert faults == 1
