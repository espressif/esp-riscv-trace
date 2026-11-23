# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Follow a register back through the trace from the trap a fault ends at.

Walking the events of one core backward from the fault meets the writer of a register whatever function it ran in. A
load is followed on to the store that wrote its bytes, and the stored value is checked against the loaded one. Untraced
code, lost trace, a task switch and a trap handler that ran the scheduler end a walk. A trap handler that returned to
where it was taken and a call to untraced code that returned are crossed by the calling convention.
"""

from __future__ import annotations

import bisect
from collections import deque
from collections.abc import Callable, Collection, Iterator, Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum

from ..decoder.encoding import A0, A1, ECALL_LENGTH, GP, MASK32, RA, S_REGS, SP, TP, XLEN_BYTES
from ..decoder.reconstruct import GapReason
from ..decoder.semantics import LINKS, InsnInfo, decode_at, evaluate, insn_text, sources
from ..elf import CodeImage
from ..ir.events import (
    LinkKind,
    TraceBoundary,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
    TraceWarning,
)
from .history import _ECALL_CAUSES, _SCHEDULER_FUNCTIONS, ends_history

# Events a window keeps, the fault trap included, which bounds the time and memory of a walk
WINDOW_EVENTS = 1 << 15
# Steps one origin tree records, so that a long chain still reads as a finding
MAX_STEPS = 8
# Writers a value is computed back through
MAX_VALUE_DEPTH = 32
# Events a load is followed back through to the store that wrote its bytes
MAX_STORE_EVENTS = 4096
# Instructions between a bus error and the trap that reports it, as far as the walk looks
BUS_ERROR_INSTRUCTIONS = 8
# Branches of the function the fault is in that the walk returns
MAX_BRANCHES = 4

# Registers a call keeps by the calling convention, gp and tp because nothing allocates them
_CALLEE_SAVED = frozenset({SP, GP, TP, *S_REGS})
# Pointers the walk does not follow when a value is computed from them
_POINTERS = frozenset({SP, GP, TP})
# Instruction address misaligned, instruction access fault, instruction page fault
_FETCH_CAUSES = frozenset({0, 1, 12})
# Load and store/AMO address misaligned, access fault and page fault
_ACCESS_CAUSES = frozenset({4, 5, 6, 7, 13, 15})
_COPIES = frozenset({"c.mv", "cm.mva01s", "cm.mvsa01"})
_INDIRECT_JUMPS = frozenset({"jalr", "c.jr", "c.jalr"})
_POP_RETURNS = frozenset({"cm.popret", "cm.popretz"})
_POPS = frozenset({"cm.pop"}) | _POP_RETURNS
_BRANCHES = frozenset({"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"})
# Instructions that set a register to itself plus a constant
_ADDS = frozenset({"addi", "c.addi", "c.addi16sp", "cm.push", "cm.pop", "cm.popret", "cm.popretz"})
# Stores of an integer register, and loads into one, that a value can be followed through
_STORES = frozenset({"sw", "sh", "sb", "c.sw", "c.swsp", "c.sh", "c.sb"})
_LOADS = frozenset({"lw", "lh", "lhu", "lb", "lbu", "c.lw", "c.lwsp", "c.lh", "c.lhu", "c.lbu"})
_SIGNED_LOADS = frozenset({"lh", "lb", "c.lh"})


class StepKind(StrEnum):
    """How a register got the value a ``RegisterStep`` describes."""

    SET = "set"  # a constant, from li, lui or auipc
    COPIED = "copied"  # by mv or a Zcmp move
    COMPUTED = "computed"  # from other registers
    LOADED = "loaded"  # from memory, by a load or a cm.pop form
    LINKED = "linked"  # the return address a call wrote
    READ = "read"  # from a CSR
    WRITTEN = "written"  # by another instruction with no register sources, such as fmv.x.w or sc.w
    POINTER = "pointer"  # sp, gp or tp, which a value was computed from and which is not followed
    UNKNOWN = "unknown"  # no writer found, the stop says why


class StopReason(StrEnum):
    """Why a walk found no writer."""

    TRACE_START = "trace_start"  # the trace starts before the writer
    WINDOW = "window"  # the events the walk has start before the writer
    TRACE_LOST = "trace_lost"
    UNTRACED = "untraced"  # untraced code ran, and the register is not one a call keeps
    CONTEXT_SWITCH = "context_switch"  # a trap handler returned to another place than where the trap was taken
    # A trap handler ran the scheduler, which can resume another task where the trap was taken
    SCHEDULER = "scheduler"
    SYSCALL = "syscall"  # the handler of an ecall returns its result in a0 and a1
    UNSUPPORTED = "unsupported"  # an instruction whose written registers are not known
    UNREADABLE = "unreadable"  # an instruction whose code cannot be read or decoded
    STEPS = "steps"  # the tree has MAX_STEPS steps


class Crossing(StrEnum):
    """An event between a write and its use that the walk went on across."""

    INTERRUPT = "interrupt"  # an interrupt handler that returned to where it was taken
    EXCEPTION = "exception"  # an exception handler that returned to where it was taken
    SYSCALL = "syscall"  # an ecall handler that returned after the ecall
    UNTRACED_CALL = "untraced_call"  # a call to untraced code that returned to its call site
    TRAP_ENTRY = "trap_entry"  # the value was used in a trap handler and comes from the code it interrupted
    TRACE_WARNING = "trace_warning"  # a TraceWarning
    # The value went through memory, and the stored value is not known to compare with the loaded one
    UNCHECKED_VALUE = "unchecked_value"


class Relation(StrEnum):
    """Where the writer ran, relative to the code that used the value."""

    CALLEE = "callee"  # in a function that returned before the value was used
    CALLER = "caller"  # before a call to the function that used the value


@dataclass(frozen=True)
class RegisterStep:
    """Where one register got the value an instruction used. Each node of a tree is a distinct object."""

    reg: int
    kind: StepKind
    value: int | None = None  # the value the using instruction read
    pc: int | None = None  # the instruction that wrote the value
    instruction: str | None = None  # the writer as objdump prints it, or its mnemonic
    use_pc: int | None = None  # the instruction that used the value, the faulting one for the root
    address: int | None = None  # where a load read the value
    store_pc: int | None = None  # the store that wrote the bytes a load read. The stored register is the source
    store_instruction: str | None = None
    relation: Relation | None = None
    sources: tuple[RegisterStep, ...] = ()  # steps of the registers the value came from
    crossed: tuple[Crossing, ...] = ()  # in the order the walk met them, from the use back to the writer
    stop: StopReason | None = None  # why an UNKNOWN step has no writer
    stop_pc: int | None = None  # the event the walk stopped at
    detail: str | None = None  # the mnemonic of an unsupported instruction, or the reason of a gap

    def walk(self) -> Iterator[RegisterStep]:
        """This step and every step under it, each before its sources."""
        yield self
        for source in self.sources:
            yield from source.walk()


@dataclass(frozen=True)
class BusAccess:
    """A load or store the trace shows retiring before a bus error, and its position in the events of the window."""

    pc: int
    instruction: str
    index: int


@dataclass(frozen=True)
class StackCall:
    """A call not returned from at the fault, and sp at the call."""

    pc: int
    sp: int


@dataclass(frozen=True)
class BranchOutcome:
    """The last time a conditional branch ran before the fault, and whether it jumped."""

    pc: int
    taken: bool


class FaultWindowBuilder:
    """Keeps the last ``WINDOW_EVENTS`` events of one core for the walks, up to where ``HistoryBuilder`` stops."""

    def __init__(self, image: CodeImage | None, *, end_interrupts: Collection[int] = ()) -> None:
        self._image = image
        self._end_interrupts = frozenset(end_interrupts)
        self._events: deque[TraceEvent] = deque(maxlen=WINDOW_EVENTS)
        self._pushed = 0
        self._fault: TraceTrap | None = None

    @property
    def done(self) -> bool:
        """True once the end is pushed. Later events are ignored."""
        return self._fault is not None

    def push(self, event: TraceEvent) -> None:
        """Keep ``event``. Every event counts against ``WINDOW_EVENTS``, a ``TraceBoundary`` too."""
        if self._fault is not None:
            return
        self._events.append(event)
        self._pushed += 1
        if isinstance(event, TraceTrap) and ends_history(event, self._end_interrupts):
            self._fault = event

    def close(self) -> FaultWindow:
        """The window over the events kept. Without the end, it keeps none and its queries find nothing."""
        events = tuple(self._events) if self._fault is not None else ()
        return FaultWindow(self._image, events, self._fault, self._pushed <= WINDOW_EVENTS)


class FaultWindow:
    """The events of one core up to and including the fault trap, and the walks over them.

    ``fault`` is that trap, or ``None`` when the trace ended first. ``complete`` is ``True`` when the window starts at
    the first event of the trace. Each query keeps its caches to itself, so a query that raises leaves the window
    usable. A ``BusAccess`` from ``bus_access()`` is valid only with the window that returned it.
    """

    def __init__(
        self,
        image: CodeImage | None,
        events: Sequence[TraceEvent],
        fault: TraceTrap | None,
        complete: bool,
    ) -> None:
        self.fault = fault
        self.complete = complete
        self._image = image
        self._kept = _Kept.of(events)

    def origin(self, registers: Mapping[int, int], *, access: BusAccess | None = None) -> RegisterStep | None:
        """Where the register the fault depends on got its value, ``None`` when there is no register to follow."""
        walk = self._walk(registers)
        if walk is None:
            return None
        root = walk.run(None if access is None else walk.index_of(access))
        return None if root is None else root.freeze()

    def bus_access(self, registers: Mapping[int, int], *, address: int, store: bool) -> BusAccess | None:
        """The last load, or store when ``store``, to ``address`` in the instructions that ran straight to the trap."""
        walk = self._walk(registers)
        return None if walk is None else walk.bus_access(address, store)

    def stack_calls(self, registers: Mapping[int, int]) -> tuple[StackCall, ...]:
        """The calls not returned from at the fault, newest first, with sp worked back from ``registers``."""
        walk = self._walk(registers)
        return () if walk is None else tuple(StackCall(pc, sp) for pc, sp in walk.stack_calls())

    def branches(self, inside: Callable[[int], bool]) -> tuple[BranchOutcome, ...]:
        """The latest outcome of each of the most recent branches of the function the fault is in, oldest first.

        The walk stops at the call that entered the function, so the branches of its callers are not included.
        """
        walk = self._walk({})
        return () if walk is None else tuple(BranchOutcome(pc, taken) for pc, taken in walk.branches(inside))

    def unsure_writes(self, trapped: int, saved: int) -> frozenset[int] | None:
        """The registers written from ``trapped`` up to ``saved``, the mepc a bus error trap at ``trapped`` saved.

        ``None`` when ``saved`` is not on an instruction boundary within ``BUS_ERROR_INSTRUCTIONS`` instructions of
        ``trapped``, or when that code is unreadable or unsupported.
        """
        if self.fault is None:
            return None
        written: set[int] = set()
        pc = trapped
        for _ in range(BUS_ERROR_INSTRUCTIONS):
            info = decode_at(self._image, pc)
            if info is None or not info.supported:
                return None
            written |= info.writes
            pc += info.size
            if saved < pc:
                return frozenset(written) if saved <= trapped else None
            if saved == pc:
                return frozenset(written)
        return None

    def _walk(self, registers: Mapping[int, int]) -> _Walk | None:
        return _Walk.over(self._kept, self._image, registers, self.complete)


@dataclass
class _Step:
    """A ``RegisterStep`` while the walk builds it."""

    reg: int
    kind: StepKind
    value: int | None = None
    pc: int | None = None
    use_pc: int | None = None
    text: str | None = None
    address: int | None = None
    store_pc: int | None = None
    store_text: str | None = None
    relation: Relation | None = None
    sources: list[_Step] = field(default_factory=list)
    crossed: list[Crossing] = field(default_factory=list)
    stop: StopReason | None = None
    stop_pc: int | None = None
    detail: str | None = None

    def freeze(self) -> RegisterStep:
        return RegisterStep(
            reg=self.reg,
            kind=self.kind,
            value=self.value,
            pc=self.pc,
            instruction=self.text,
            use_pc=self.use_pc,
            address=self.address,
            store_pc=self.store_pc,
            store_instruction=self.store_text,
            relation=self.relation,
            sources=tuple(source.freeze() for source in self.sources),
            crossed=tuple(self.crossed),
            stop=self.stop,
            stop_pc=self.stop_pc,
            detail=self.detail,
        )


@dataclass(frozen=True)
class _Scan:
    """What a scan back from one event found for one register."""

    writer: int | None  # index of the instruction that wrote the register
    end: int  # index the scan ended at, which is the writer, the event it stopped at, or -1
    depth: int = 0  # call depth of the writer relative to the use
    crossed: tuple[Crossing, ...] = ()
    stop: StopReason | None = None
    stop_pc: int | None = None
    detail: str | None = None


@dataclass(frozen=True)
class _Kept:
    """The events a walk reads, which are those given without boundaries."""

    events: tuple[TraceEvent, ...]
    positions: tuple[int, ...]  # positions[i] is where events[i] is in the events given

    @classmethod
    def of(cls, events: Sequence[TraceEvent]) -> _Kept:
        positions = tuple(i for i, event in enumerate(events) if not isinstance(event, TraceBoundary))
        return cls(tuple(events[i] for i in positions), positions)


class _Walk:
    """The events up to the fault trap without boundaries, with caches for one query."""

    def __init__(
        self,
        kept: _Kept,
        image: CodeImage | None,
        registers: Mapping[int, int],
        complete: bool,
    ) -> None:
        self.positions = kept.positions
        self.events = kept.events
        self.image = image
        self.frame = dict(registers)
        self.complete = complete
        self.symbols: dict[int, str | None] = {}
        self.trap_spans: dict[int, tuple[int, Crossing] | StopReason] = {}
        self.fault = len(self.events) - 1
        self.steps = 0
        self.infos: dict[int, InsnInfo | None] = {}
        self.scans: dict[tuple[int, int], _Scan] = {}
        # The value of each (register, index), and the depth it was looked up at
        self.values: dict[tuple[int, int], tuple[int | None, int]] = {}
        self.spans: list[tuple[int, int]] | None = None
        self.ends: dict[int, int] | None = None

    @classmethod
    def over(
        cls,
        kept: _Kept,
        image: CodeImage | None,
        registers: Mapping[int, int],
        complete: bool,
    ) -> _Walk | None:
        """A walk of the events kept, or ``None`` when they do not end with a trap."""
        if not kept.events or not isinstance(kept.events[-1], TraceTrap):
            return None
        return cls(kept, image, registers, complete)

    def index_of(self, access: BusAccess) -> int:
        i = bisect.bisect_left(self.positions, access.index)
        event = self.events[i] if i < len(self.positions) and self.positions[i] == access.index else None
        if not isinstance(event, TraceInstruction) or event.pc != access.pc:
            raise ValueError(f"event {access.index} of these events is not the access at 0x{access.pc:08x}")
        return i

    def run(self, access: int | None) -> _Step | None:
        target = self.target() if access is None else self.access_target(access)
        if target is None:
            return None
        reg, use, use_pc = target
        return self.step(reg, use, use_pc)

    def access_target(self, access: int) -> tuple[int, int, int] | None:
        """The base register of the load or store at index access."""
        insn = self.events[access]
        assert isinstance(insn, TraceInstruction)
        info = self.info(insn)
        if info is None or info.mem is None or not info.mem.base:
            return None
        return info.mem.base, access, insn.pc

    def bus_access(self, address: int, store: bool) -> BusAccess | None:
        seen = 0
        for i in range(self.fault - 1, -1, -1):
            event = self.events[i]
            if isinstance(event, TraceSync):
                continue
            if not isinstance(event, TraceInstruction) or seen == BUS_ERROR_INSTRUCTIONS:
                return None
            seen += 1
            info = self.info(event)
            if info is None or not info.supported:
                return None
            mem = info.mem
            if mem is None or mem.is_store != store:
                continue
            base = self.value(mem.base, i)
            if base is not None and (base + mem.offset) & MASK32 == address:
                return BusAccess(event.pc, insn_text(info, event.pc) or info.mnemonic, self.positions[i])
        return None

    def target(self) -> tuple[int, int, int] | None:
        """The register the fault depends on, the event that used it, and its pc."""
        fault = self.events[self.fault]
        assert isinstance(fault, TraceTrap)
        if fault.interrupt or fault.epc is None:
            return None
        if fault.cause in _ACCESS_CAUSES:
            info = decode_at(self.image, fault.epc)
            # x0 needs no walk because it reads as zero
            if info is None or info.mem is None or not info.mem.base:
                return None
            return info.mem.base, self.fault, fault.epc
        if fault.cause in _FETCH_CAUSES and self.fault > 0:
            jump = self.events[self.fault - 1]
            if not isinstance(jump, TraceInstruction) or jump.next_pc != fault.epc:
                return None
            info = self.info(jump)
            if info is None or not info.supported:
                return None
            if info.mnemonic in _INDIRECT_JUMPS and info.rs1:
                return info.rs1, self.fault - 1, jump.pc
            # cm.popret loads ra and jumps through it, so it is the writer
            if info.mnemonic in _POP_RETURNS:
                return RA, self.fault, jump.pc
        return None

    def back(self) -> Iterator[tuple[int, TraceInstruction]]:
        """The instructions of the code the fault is in, newest first, over handlers and calls that returned."""
        i = self.fault - 1
        while i >= 0:
            event = self.events[i]
            if isinstance(event, TraceInstruction):
                if event.link == LinkKind.TRAP_RETURN:
                    span = self.trap_span(i)
                    if isinstance(span, StopReason):
                        return
                    i = span[0] - 1
                    continue
                yield i, event
            elif isinstance(event, TraceTrap):
                # The fault is in a trap handler, which can run on another stack
                return
            elif isinstance(event, TraceGap):
                if _lost(event) or self.untraced_call(i) is None:
                    return
                # Skip the untraced code and the call into it
                i -= 2
                continue
            i -= 1

    def stack_calls(self) -> list[tuple[int, int]]:
        sp = self.frame.get(SP)
        calls: list[tuple[int, int]] = []
        if sp is None:
            return calls
        # Returns seen walking back whose call is not reached yet
        returned = 0
        for _, insn in self.back():
            info = self.info(insn)
            if info is None or not info.supported:
                break
            if SP in info.writes:
                adds = info.mnemonic in _ADDS and sources(info, SP) == frozenset({SP})
                delta = evaluate(info, SP, insn.pc, {SP: 0}) if adds else None
                if delta is None:
                    break
                sp = (sp - delta) & MASK32
            if insn.link == LinkKind.RETURN:
                returned += 1
            elif insn.link == LinkKind.CALL:
                if returned:
                    returned -= 1
                else:
                    calls.append((insn.pc, sp))
        return calls

    def branches(self, inside: Callable[[int], bool]) -> list[tuple[int, bool]]:
        # The last outcome by pc, newest first
        found: dict[int, bool] = {}
        returned = 0
        for i, insn in self.back():
            if len(found) == MAX_BRANCHES:
                break
            if insn.link == LinkKind.RETURN:
                returned += 1
            elif returned:
                if insn.link == LinkKind.CALL:
                    returned -= 1
            elif not inside(insn.pc) or insn.link == LinkKind.CALL:
                break
            elif insn.pc not in found:
                taken = self.taken(i)
                if taken is not None:
                    found[insn.pc] = taken
        return list(found.items())[::-1]

    def taken(self, index: int) -> bool | None:
        """Whether the conditional branch at index jumped. None for another instruction."""
        insn = self.events[index]
        assert isinstance(insn, TraceInstruction)
        info = self.info(insn)
        if info is None or info.mnemonic not in _BRANCHES:
            return None
        # A branch to its own fall-through moves on to the same pc either way
        if insn.branch_taken is not None:
            return insn.branch_taken
        next_pc, after = insn.next_pc, self.events[index + 1]
        if next_pc is None and isinstance(after, TraceInstruction):
            next_pc = after.pc
        return None if next_pc is None else next_pc != (insn.pc + insn.length) & MASK32

    def info(self, insn: TraceInstruction) -> InsnInfo | None:
        if insn.pc not in self.infos:
            self.infos[insn.pc] = decode_at(self.image, insn.pc)
        return self.infos[insn.pc]

    def runs_scheduler(self, pc: int | None) -> bool:
        """Whether a call to pc switches tasks."""
        if pc is None:
            return False
        if pc not in self.symbols:
            self.symbols[pc] = None if self.image is None else self.image.symbol_at(pc)
        return self.symbols[pc] in _SCHEDULER_FUNCTIONS

    def step(self, reg: int, use: int, use_pc: int | None) -> _Step:
        """Where reg, as the event at index use read it, got its value."""
        value = self.value(reg, use)
        if self.steps >= MAX_STEPS:
            return _Step(reg, StepKind.UNKNOWN, value, use_pc=use_pc, stop=StopReason.STEPS)
        self.steps += 1
        scan = self.scan(reg, use)
        if scan.writer is None:
            return _Step(
                reg,
                StepKind.UNKNOWN,
                value,
                use_pc=use_pc,
                crossed=list(scan.crossed),
                stop=scan.stop,
                stop_pc=scan.stop_pc,
                detail=scan.detail,
            )
        writer = self.events[scan.writer]
        assert isinstance(writer, TraceInstruction)
        info = self.info(writer)
        assert info is not None
        step = _Step(
            reg,
            StepKind.UNKNOWN,
            value,
            writer.pc,
            use_pc,
            insn_text(info, writer.pc) or info.mnemonic,
            relation=Relation.CALLEE if scan.depth > 0 else Relation.CALLER if scan.depth < 0 else None,
            crossed=list(scan.crossed),
        )
        needed = sources(info, reg)
        if info.mnemonic in LINKS:
            step.kind = StepKind.LINKED
        elif needed is None:
            if reg in info.loaded:
                step.kind = StepKind.LOADED
            else:
                step.kind = StepKind.READ if info.csr is not None else StepKind.WRITTEN
            if step.kind == StepKind.LOADED:
                slot = self.loaded_from(info, reg, scan.writer)
                if slot is not None:
                    step.address, width = slot
                    if info.mnemonic in _LOADS or info.mnemonic in _POPS:
                        self.follow_store(step, width, info.mnemonic in _SIGNED_LOADS, scan.writer)
        elif not needed:
            step.kind = StepKind.SET
        else:
            step.kind = StepKind.COPIED if _is_copy(info) else StepKind.COMPUTED
            for source in sorted(needed):
                if source in _POINTERS:
                    step.sources.append(
                        _Step(source, StepKind.POINTER, self.value(source, scan.writer), use_pc=writer.pc)
                    )
                else:
                    step.sources.append(self.step(source, scan.writer, writer.pc))
        return step

    def loaded_from(self, info: InsnInfo, reg: int, index: int) -> tuple[int, int] | None:
        """The address and width the instruction at index loaded reg from, when the trace shows them."""
        if info.mnemonic in _POPS:
            sp = self.value(SP, index)
            if sp is None or info.imm is None:
                return None
            # cm.push left the highest register at the new sp - 4 and ra lowest
            saved = sorted(info.loaded)
            return (sp + info.imm - XLEN_BYTES * (len(saved) - saved.index(reg))) & MASK32, XLEN_BYTES
        if info.mem is None:
            return None
        base = self.value(info.mem.base, index)
        if base is None:
            return None
        return (base + info.mem.offset) & MASK32, info.mem.width

    def follow_store(self, step: _Step, width: int, signed: bool, index: int) -> None:
        """Go on from the load at index to the store that wrote its bytes, when it stored the value loaded."""
        assert step.address is not None
        found = self.find_store(step.address, width, index)
        if found is None:
            return
        store_index, reg, crossed = found
        stored = self.value(reg, store_index)
        if stored is None or step.value is None:
            crossed.append(Crossing.UNCHECKED_VALUE)
        elif _as_loaded(stored, width, signed) != step.value:
            return
        store = self.events[store_index]
        assert isinstance(store, TraceInstruction)
        info = self.info(store)
        assert info is not None
        step.store_pc = store.pc
        step.store_text = insn_text(info, store.pc) or info.mnemonic
        for item in crossed:
            _add(step.crossed, item)
        # sw zero stores a constant, which the store's text already says
        if reg in _POINTERS:
            step.sources.append(_Step(reg, StepKind.POINTER, stored, use_pc=store.pc))
        elif reg:
            step.sources.append(self.step(reg, store_index, store.pc))

    def find_store(self, address: int, width: int, load: int) -> tuple[int, int, list[Crossing]] | None:
        """The last store before the load at index load to these bytes, its register and the crossings in between."""
        crossed: list[Crossing] = []
        end = address + width
        i = load - 1
        while i >= 0 and load - i <= MAX_STORE_EVENTS:
            event = self.events[i]
            if isinstance(event, TraceInstruction):
                info = self.info(event)
                if info is None or not info.supported:
                    return None
                for start, size, reg in self.stores(info, i):
                    if start is None:
                        return None
                    if start < end and address < start + size:
                        if start == address and size == width and reg is not None:
                            return i, reg, crossed
                        return None
            elif isinstance(event, TraceTrap):
                if event.interrupt:
                    _add(crossed, Crossing.INTERRUPT)
                elif event.cause in _ECALL_CAUSES:
                    _add(crossed, Crossing.SYSCALL)
                else:
                    _add(crossed, Crossing.EXCEPTION)
            elif isinstance(event, TraceGap):
                return None
            elif isinstance(event, TraceWarning):
                _add(crossed, Crossing.TRACE_WARNING)
            i -= 1
        return None

    def stores(self, info: InsnInfo, index: int) -> list[tuple[int | None, int, int | None]]:
        """The memory the instruction at index writes as (address, width, register stored), with None where unknown."""
        if info.mnemonic == "cm.push":
            sp = self.value(SP, index)
            if sp is None:
                return [(None, 0, None)]
            # The highest register goes to sp - 4 and ra lowest, as the Zcmp spec lists it
            saved = sorted(r for r in info.reads if r != SP)
            return [((sp - XLEN_BYTES * (len(saved) - n)) & MASK32, XLEN_BYTES, reg) for n, reg in enumerate(saved)]
        mem = info.mem
        if mem is None or not mem.is_store:
            return []
        base = self.value(mem.base, index)
        if base is None:
            return [(None, 0, None)]
        return [((base + mem.offset) & MASK32, mem.width, info.rs2 if info.mnemonic in _STORES else None)]

    def scan(self, reg: int, use: int) -> _Scan:
        key = (reg, use)
        if key not in self.scans:
            self.scans[key] = self._scan(reg, use)
        return self.scans[key]

    def _scan(self, reg: int, use: int) -> _Scan:
        """The last instruction before the event at index use that wrote reg."""
        depth = 0
        crossed: list[Crossing] = []

        def stop(i: int, why: StopReason, pc: int | None = None, detail: str | None = None) -> _Scan:
            return _Scan(None, i, depth, tuple(crossed), why, pc, detail)

        i = use - 1
        while i >= 0:
            event = self.events[i]
            if isinstance(event, TraceInstruction):
                if event.link == LinkKind.TRAP_RETURN:
                    span = self.trap_span(i)
                    if isinstance(span, StopReason):
                        switched = span in (StopReason.CONTEXT_SWITCH, StopReason.SCHEDULER)
                        return stop(i, span, event.next_pc if switched else None)
                    entry, kind = span
                    if kind == Crossing.SYSCALL and reg in (A0, A1):
                        trap = self.events[entry]
                        assert isinstance(trap, TraceTrap)
                        return stop(i, StopReason.SYSCALL, trap.epc)
                    _add(crossed, kind)
                    i = entry - 1
                    continue
                info = self.info(event)
                if info is None:
                    return stop(i, StopReason.UNREADABLE, event.pc)
                if not info.supported:
                    name = info.mnemonic if info.mnemonic != "unknown" else f"0x{info.insn:0{2 * info.size}x}"
                    return stop(i, StopReason.UNSUPPORTED, event.pc, name)
                # a return that writes the register, like cm.popretz, wrote it in the callee
                if event.link == LinkKind.RETURN:
                    depth += 1
                elif event.link == LinkKind.CALL:
                    depth -= 1
                if reg in info.writes:
                    return _Scan(i, i, depth, tuple(crossed))
            elif isinstance(event, TraceTrap):
                # the use is in a trap handler, which found the register as the interrupted code left it
                _add(crossed, Crossing.TRAP_ENTRY)
            elif isinstance(event, TraceGap):
                if _lost(event):
                    return stop(i, StopReason.TRACE_LOST, detail=str(event.reason))
                call = self.untraced_call(i)
                if call is None or reg not in _CALLEE_SAVED:
                    left = self.events[i - 1] if i else None
                    return stop(i, StopReason.UNTRACED, left.pc if isinstance(left, TraceInstruction) else None)
                _add(crossed, Crossing.UNTRACED_CALL)
                # the call wrote only ra, which is not callee-saved
                i -= 2
                continue
            elif isinstance(event, TraceWarning):
                _add(crossed, Crossing.TRACE_WARNING)
            i -= 1
        return stop(-1, StopReason.TRACE_START if self.complete else StopReason.WINDOW)

    def trap_span(self, trap_return: int) -> tuple[int, Crossing] | StopReason:
        """The trap the trap return at index trap_return ends and its crossing, or why the handler did not resume."""
        if trap_return not in self.trap_spans:
            self.trap_spans[trap_return] = self._trap_span(trap_return)
        return self.trap_spans[trap_return]

    def _trap_span(self, trap_return: int) -> tuple[int, Crossing] | StopReason:
        entry = self.trap_entry(trap_return)
        if isinstance(entry, StopReason):
            return entry
        trap = self.events[entry]
        mret = self.events[trap_return]
        assert isinstance(trap, TraceTrap) and isinstance(mret, TraceInstruction)
        if mret.next_pc == trap.epc:
            kind = Crossing.INTERRUPT if trap.interrupt else Crossing.EXCEPTION
        elif (
            not trap.interrupt
            and trap.cause in _ECALL_CAUSES
            and trap.epc is not None
            and mret.next_pc == trap.epc + ECALL_LENGTH
        ):
            kind = Crossing.SYSCALL
        else:
            return StopReason.CONTEXT_SWITCH
        # Tasks that yield through the same code are preempted at the same instruction, so returning there shows nothing
        if any(
            isinstance(e, TraceInstruction) and e.link == LinkKind.CALL and self.runs_scheduler(e.next_pc)
            for e in self.events[entry + 1 : trap_return]
        ):
            return StopReason.SCHEDULER
        return entry, kind

    def skipped(self) -> list[tuple[int, int]]:
        """The (trap, trap return) indexes of the handlers a walk back from the fault steps over."""
        if self.spans is None:
            self.spans = []
            i = self.fault - 1
            while i >= 0:
                event = self.events[i]
                if isinstance(event, TraceInstruction) and event.link == LinkKind.TRAP_RETURN:
                    span = self.trap_span(i)
                    if isinstance(span, StopReason):
                        break
                    self.spans.append((span[0], i))
                    i = span[0] - 1
                    continue
                if isinstance(event, TraceGap) and _lost(event):
                    break
                i -= 1
        return self.spans

    def on_path(self, index: int) -> bool:
        """Whether the event at index ran outside the handlers a walk back from the fault steps over."""
        return not any(start < index < end for start, end in self.skipped())

    def trap_entry(self, trap_return: int) -> int | StopReason:
        """The index of the trap the trap return at index trap_return returns from, or why it is not known."""
        level = 1
        i = trap_return - 1
        while i >= 0:
            event = self.events[i]
            if isinstance(event, TraceTrap):
                level -= 1
                if level == 0:
                    return i
            elif isinstance(event, TraceInstruction) and event.link == LinkKind.TRAP_RETURN:
                level += 1
            elif isinstance(event, TraceGap) and _lost(event):
                return StopReason.TRACE_LOST
            i -= 1
        return StopReason.TRACE_START if self.complete else StopReason.WINDOW

    def untraced_call(self, gap: int) -> TraceInstruction | None:
        """The call before the untraced gap at index gap, when the trace goes on at its return address."""
        if gap == 0 or gap + 1 >= len(self.events):
            return None
        call = self.events[gap - 1]
        resume = self.events[gap + 1]
        if not isinstance(call, TraceInstruction) or call.link != LinkKind.CALL or call.next_pc is not None:
            return None
        if not isinstance(resume, TraceSync) or resume.pc != call.pc + call.length:
            return None
        return call

    def value(self, reg: int, use: int, depth: int = 0) -> int | None:
        """The value of reg as the event at index use read it, when the frame or the trace shows it."""
        if reg == 0:
            return 0
        key = (reg, use)
        cached = self.values.get(key)
        # A value not found deeper down can still be found with more of the depth left
        if cached is not None and (cached[0] is not None or cached[1] <= depth):
            return cached[0]
        if depth >= MAX_VALUE_DEPTH:
            return None
        result = self._forward(reg, use, depth)
        if result is None:
            result = self._computed(reg, use, depth)
        self.values[key] = (result, depth)
        return result

    def _forward(self, reg: int, use: int, depth: int) -> int | None:
        """The value of reg at use from the frame, back through the constants the writers up to the fault add."""
        if depth >= MAX_VALUE_DEPTH or not self.on_path(use):
            return None
        if self.scan(reg, self.fault).end < use:
            # Nothing between use and the fault wrote reg, as far as the trace shows
            return self.frame.get(reg)
        writer = self.next_writer(reg, use)
        if writer is None:
            return None
        insn = self.events[writer]
        assert isinstance(insn, TraceInstruction)
        info = self.info(insn)
        if info is None or info.mnemonic not in _ADDS or sources(info, reg) != frozenset({reg}):
            return None
        delta = evaluate(info, reg, insn.pc, {reg: 0})
        # Only forward, since the writers before use would ask for the value at use again
        after = self._forward(reg, writer + 1, depth + 1)
        if delta is None or after is None:
            return None
        return (after - delta) & MASK32

    def next_writer(self, reg: int, use: int) -> int | None:
        """The first instruction from index use on to the fault that writes reg, None when the way is not clear."""
        if self.ends is None:
            self.ends = dict(self.skipped())
        ends = self.ends
        i = use
        while i < self.fault:
            event = self.events[i]
            if i in ends:
                # The handler of an ecall returns its result in a0 and a1
                if reg in (A0, A1) and self.trap_span(ends[i]) == (i, Crossing.SYSCALL):
                    return None
                i = ends[i] + 1
                continue
            if isinstance(event, TraceInstruction):
                info = self.info(event)
                if info is None or not info.supported:
                    return None
                if reg in info.writes:
                    return i
            elif isinstance(event, TraceGap):
                if _lost(event) or reg not in _CALLEE_SAVED or self.untraced_call(i) is None:
                    return None
            i += 1
        return None

    def _computed(self, reg: int, use: int, depth: int) -> int | None:
        scan = self.scan(reg, use)
        if scan.writer is None:
            return None
        if scan.writer == self.scan(reg, self.fault).writer and reg in self.frame:
            # the last write before the fault, which the frame saved
            return self.frame[reg]
        writer = self.events[scan.writer]
        assert isinstance(writer, TraceInstruction)
        info = self.info(writer)
        needed = sources(info, reg) if info is not None else None
        if info is None or needed is None:
            return None
        regs = {}
        for source in needed:
            value = self.value(source, scan.writer, depth + 1)
            if value is None:
                return None
            regs[source] = value
        return evaluate(info, reg, writer.pc, regs)


def _lost(gap: TraceGap) -> bool:
    """Whether the gap lost trace. Untraced code loses none."""
    return gap.reason != GapReason.UNTRACED


def _as_loaded(stored: int, width: int, signed: bool) -> int:
    """What a load of width bytes reads back from the bytes a store of the value stored wrote."""
    bits = 8 * width
    value = stored & ((1 << bits) - 1)
    if signed and value >> (bits - 1):
        value |= MASK32 & ~((1 << bits) - 1)
    return value


def _is_copy(info: InsnInfo) -> bool:
    return info.mnemonic in _COPIES or (info.mnemonic == "addi" and info.imm == 0)


def _add(items: list[Crossing], item: Crossing) -> None:
    if item not in items:
        items.append(item)


__all__ = [
    "BUS_ERROR_INSTRUCTIONS",
    "MAX_BRANCHES",
    "MAX_STEPS",
    "MAX_STORE_EVENTS",
    "MAX_VALUE_DEPTH",
    "WINDOW_EVENTS",
    "BranchOutcome",
    "BusAccess",
    "Crossing",
    "FaultWindow",
    "FaultWindowBuilder",
    "RegisterStep",
    "Relation",
    "StackCall",
    "StepKind",
    "StopReason",
]
