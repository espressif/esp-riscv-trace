# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Trap packet reconstruction for ET 2.0 and PT 1.0."""

from __future__ import annotations

import pytest
from helpers import BASE, JALR_X0_RA, NOP, FakeImage, make_frame, make_sync

from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.packets import Format2Packet, TrapPacket
from esp_riscv_trace.decoder.params import DecoderProfile, default_profile
from esp_riscv_trace.ir.events import InsnClass, TraceGap, TraceInstruction, TraceSync, TraceTrap
from esp_riscv_trace.snapshot import abi

ET = default_profile("esp32c5")
PT = default_profile("esp32c6")

HANDLER = 0x40801000
ECALL = bytes.fromhex("73000000")
EBREAK = bytes.fromhex("73001000")
C_EBREAK = bytes.fromhex("0290")

# RISC-V exception codes
_ILLEGAL_INSTRUCTION = 2
_BREAKPOINT = 3
_ECALL_M = 11


def _trap(address: int, *, ecause: int, interrupt: int = 0, thaddr: int | None = 1, tval: int | None = 0) -> TrapPacket:
    return TrapPacket(
        frame=make_frame(1),
        branch=1,
        privilege=0,
        ecause=ecause,
        interrupt=interrupt,
        thaddr=thaddr,
        raw_address=address >> 1,
        address_pc=address,
        tval=tval,
    )


def _with_handler(code: bytes, handler: bytes = JALR_X0_RA) -> FakeImage:
    """Image with ``code`` at BASE and ``handler`` at HANDLER."""
    return FakeImage(BASE, code.ljust(HANDLER - BASE, b"\0") + handler)


def _run(packets, image: FakeImage, profile: DecoderProfile = ET, mode: abi.AddressMode = abi.AddressMode.FULL):
    return list(reconstruct.reconstruct(packets, image, profile, mode))


def _instructions(events) -> list[TraceInstruction]:
    return [e for e in events if isinstance(e, TraceInstruction)]


@pytest.mark.parametrize(
    ("insn", "ecause"),
    [(ECALL, _ECALL_M), (EBREAK, _BREAKPOINT), (C_EBREAK, _BREAKPOINT)],
    ids=["ecall", "ebreak", "c_ebreak"],
)
@pytest.mark.parametrize(("thaddr", "address"), [(1, HANDLER), (0, BASE + 4)], ids=["handler", "interrupted"])
def test_et20_synchronous_trap_retires_trap_instruction(insn: bytes, ecause: int, thaddr: int, address: int) -> None:
    image = _with_handler(NOP + insn + NOP)
    events = _run([make_sync(BASE, branch=1), _trap(address, ecause=ecause, thaddr=thaddr, tval=0x1234)], image)

    assert [type(e) for e in events] == [TraceSync, TraceInstruction, TraceInstruction, TraceTrap]
    nop, trap_insn = _instructions(events)
    assert (nop.pc, nop.cls) == (BASE, InsnClass.NONE)
    assert trap_insn.pc == BASE + 4
    assert trap_insn.cls == InsnClass.TRAP
    assert trap_insn.length == len(insn)
    assert trap_insn.branch_taken is None
    assert trap_insn.next_pc == address
    trap = events[-1]
    assert isinstance(trap, TraceTrap)
    assert trap.cause == ecause
    assert trap.interrupt is False
    assert trap.tval == 0x1234
    assert trap.epc == BASE + 4
    if thaddr == 1:
        assert (trap.handler_pc, trap.interrupted_pc) == (HANDLER, None)
    else:
        assert (trap.handler_pc, trap.interrupted_pc) == (None, BASE + 4)


def test_et20_interrupt_does_not_retire_trap_instruction() -> None:
    image = _with_handler(NOP + ECALL)
    events = _run([make_sync(BASE, branch=1), _trap(HANDLER, ecause=7, interrupt=1, tval=None)], image)

    assert [e.pc for e in _instructions(events)] == [BASE]
    trap = events[-1]
    assert isinstance(trap, TraceTrap)
    assert trap.interrupt is True
    assert (trap.cause, trap.handler_pc, trap.interrupted_pc, trap.tval) == (7, HANDLER, None, None)


def test_trap_at_non_trap_instruction_has_no_trap_instruction() -> None:
    image = _with_handler(NOP + JALR_X0_RA)
    events = _run([make_sync(BASE, branch=1), _trap(HANDLER, ecause=_ILLEGAL_INSTRUCTION)], image)

    assert [(e.pc, e.cls) for e in _instructions(events)] == [(BASE, InsnClass.NONE)]
    assert isinstance(events[-1], TraceTrap)


@pytest.mark.parametrize(
    ("ecause", "interrupt", "tval", "interrupted_pc"),
    [
        (_ILLEGAL_INSTRUCTION, 0, BASE + 4, BASE + 4),
        (_ILLEGAL_INSTRUCTION, 0, (1 << 32) | (BASE + 4), BASE + 4),  # masked to the address width
        (_ECALL_M, 0, BASE + 4, None),
        (_ILLEGAL_INSTRUCTION, 1, BASE + 4, None),
        (_ILLEGAL_INSTRUCTION, 0, None, None),
    ],
    ids=["illegal", "illegal_masked", "other_cause", "interrupt", "no_tval"],
)
def test_pt10_trap_interrupted_pc_from_tval(
    ecause: int, interrupt: int, tval: int | None, interrupted_pc: int | None
) -> None:
    image = _with_handler(NOP + JALR_X0_RA)
    trap = _trap(HANDLER, ecause=ecause, interrupt=interrupt, thaddr=None, tval=tval)
    events = _run([make_sync(BASE, branch=1), trap], image, PT)

    last = events[-1]
    assert isinstance(last, TraceTrap)
    assert last.handler_pc == HANDLER
    assert last.interrupted_pc == interrupted_pc
    assert last.tval == tval


def test_pt10_ecall_retires_to_handler() -> None:
    image = _with_handler(ECALL)
    events = _run([make_sync(BASE, branch=1), _trap(HANDLER, ecause=_ECALL_M, thaddr=None)], image, PT)

    (ecall,) = _instructions(events)
    assert (ecall.pc, ecall.cls, ecall.next_pc) == (BASE, InsnClass.TRAP, HANDLER)


def test_trap_after_pc_out_of_image_still_yields_trap() -> None:
    image = _with_handler(NOP)
    events = _run([make_sync(BASE - 4, branch=1), _trap(HANDLER, ecause=_BREAKPOINT)], image)

    assert [type(e) for e in events] == [TraceSync, TraceGap, TraceTrap]
    gap = events[1]
    assert isinstance(gap, TraceGap)
    assert gap.reason == reconstruct.GapReason.PC_OUT_OF_IMAGE
    assert events[-1].epc is None


@pytest.mark.parametrize("tail", [b"", b"\x1f\x00\x00\x00\x00\x00"], ids=["out_of_image", "decode_error"])
def test_trap_instruction_read_failure_still_yields_trap(tail: bytes) -> None:
    # The return jumps to the updiscon target BASE+12, which the trap names as its epc
    image = FakeImage(BASE, (NOP + JALR_X0_RA + NOP + tail).ljust(HANDLER - BASE, b"\xff") + JALR_X0_RA)
    updiscon = Format2Packet(
        frame=make_frame(1), notify=0, updiscon=1, raw_address=(BASE + 12) >> 1, address_width_bits=32
    )
    events = _run([make_sync(BASE, branch=1), updiscon, _trap(HANDLER, ecause=_BREAKPOINT)], image)

    assert [(e.pc, e.next_pc) for e in _instructions(events)] == [(BASE, BASE + 4), (BASE + 4, BASE + 12)]
    assert not any(isinstance(e, TraceGap) for e in events)
    trap = events[-1]
    assert isinstance(trap, TraceTrap)
    assert (trap.handler_pc, trap.epc) == (HANDLER, BASE + 12)


def test_walker_resumes_at_handler_after_trap() -> None:
    image = _with_handler(NOP + ECALL + JALR_X0_RA, NOP + JALR_X0_RA)
    events = _run([make_sync(BASE, branch=1), _trap(HANDLER, ecause=_ECALL_M), make_sync(BASE + 8, branch=1)], image)

    trap_at = next(i for i, e in enumerate(events) if isinstance(e, TraceTrap))
    after = events[trap_at + 1 :]
    assert [type(e) for e in after] == [TraceInstruction, TraceInstruction, TraceSync]
    nop, ret = _instructions(after)
    assert (nop.pc, nop.cls) == (HANDLER, InsnClass.NONE)
    # The sync supplies the target of the return the walker waited at
    assert (ret.pc, ret.cls, ret.next_pc) == (HANDLER + 4, InsnClass.UNINFERABLE, BASE + 8)
    assert not any(isinstance(e, TraceGap) for e in events)


def test_handler_branch_takes_the_trap_branch_bit() -> None:
    # beq x0, x0, +8 at the handler, the trap branch bit 1 means not taken
    image = _with_handler(ECALL, bytes.fromhex("63040000") + NOP + JALR_X0_RA)
    events = _run([make_sync(BASE, branch=1), _trap(HANDLER, ecause=_ECALL_M)], image)

    branch = next(e for e in _instructions(events) if e.cls == InsnClass.BRANCH)
    assert (branch.pc, branch.branch_taken, branch.next_pc) == (HANDLER, False, HANDLER + 4)


def test_walker_waits_for_sync_after_trap_without_handler() -> None:
    image = FakeImage(BASE, NOP + ECALL + NOP + JALR_X0_RA)
    # Delta of +2 halfwords from the interrupted address
    delta = Format2Packet(frame=make_frame(2), notify=0, updiscon=0, raw_address=2, address_width_bits=8)
    events = _run(
        [make_sync(BASE, branch=1), _trap(BASE + 4, ecause=_ECALL_M, thaddr=0), delta, make_sync(BASE + 8, branch=1)],
        image,
        mode=abi.AddressMode.DELTA,
    )

    trap_at = next(i for i, e in enumerate(events) if isinstance(e, TraceTrap))
    after = events[trap_at + 1 :]
    assert [type(e) for e in after] == [TraceSync, TraceInstruction]
    assert not any(isinstance(e, TraceGap) for e in events)
    assert [e.pc for e in _instructions(after)] == [BASE + 8]
