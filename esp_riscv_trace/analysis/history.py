# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Function-level outline of a trace, up to the first fault.

``HistoryBuilder`` follows the call stack effect of every jump (``TraceInstruction.link``) on a shadow stack
and keeps one entry per call, return, tail call, trap, trap return, context switch and gap instead of one per
instruction. Runs of entries that repeat, such as a loop that calls the same functions on each pass, fold
into one ``HistoryLoop`` with a pass count. ``History.loop`` is the loop of instructions the code ran in where the
history ends.

The fault is the first exception other than an environment call. The caller can name interrupts that end the
history too, such as the watchdog interrupt that runs a panic handler.
"""

from __future__ import annotations

from collections import deque
from collections.abc import Collection, Iterable, Iterator, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum

from ..decoder.reconstruct import GapReason
from ..elf import CodeImage
from ..ir.events import InsnClass, LinkKind, TraceEvent, TraceGap, TraceInstruction, TraceSync, TraceTrap

# mcause exception codes of the RISC-V privileged ISA
EXCEPTION_NAMES: dict[int, str] = {
    0: "instruction address misaligned",
    1: "instruction access fault",
    2: "illegal instruction",
    3: "breakpoint",
    4: "load address misaligned",
    5: "load access fault",
    6: "store/AMO address misaligned",
    7: "store/AMO access fault",
    8: "environment call from U-mode",
    9: "environment call from S-mode",
    11: "environment call from M-mode",
    12: "instruction page fault",
    13: "load page fault",
    15: "store/AMO page fault",
}

# Environment calls are requests the handler serves and returns from, not faults
_ECALL_CAUSES = frozenset({8, 9, 11})

# ecall has no compressed form
_ECALL_LENGTH = 4

# Longest run of items that folds into a loop
MAX_LOOP_PERIOD = 16

# Oldest frames are dropped beyond this, so runaway recursion cannot grow the shadow stack without bound
_MAX_FRAMES = 1024

# Instructions a new loop looks back over for where its first pass began
_LOOP_LOOKBACK = 4096

# The FreeRTOS function that picks the next task. A trap handler that calls it can resume another task, which
# tasks that yield through the same code do at the very instruction the trap interrupted.
SCHEDULER_FUNCTIONS = frozenset({"vTaskSwitchContext"})

_SCHEDULER_SWITCH = "the trap handler ran the scheduler and resumed another task at the interrupted instruction"


def is_fault(trap: TraceTrap) -> bool:
    """Tell whether ``trap`` is an exception other than an environment call, where a crash analysis ends."""
    return not trap.interrupt and trap.cause not in _ECALL_CAUSES


def cause_name(cause: int, interrupt: bool) -> str:
    """Return a readable name for an mcause value."""
    if interrupt:
        return f"interrupt {cause}"
    return EXCEPTION_NAMES.get(cause, f"exception {cause}")


class HistoryKind(StrEnum):
    """What one history entry records."""

    START = "start"  # the trace starts, or restarts after a gap
    CALL = "call"
    RETURN = "return"
    TAIL_CALL = "tail_call"  # a jump into another function, which keeps the call depth
    INTERRUPT = "interrupt"
    ECALL = "ecall"  # an environment call, which the handler returns from
    TRAP_RETURN = "trap_return"  # mret back to the instruction the trap was taken at
    # mret to other code than the trapped one, usually another task, or to the trapped instruction where a later
    # return shows another task
    CONTEXT_SWITCH = "context_switch"
    # Where the history ends: the first exception other than an environment call, or an interrupt that ends it
    FAULT = "fault"
    GAP = "gap"


# Entries that move control by a jump, whose target is None when it leaves the traced range
_JUMP_KINDS = frozenset({HistoryKind.CALL, HistoryKind.RETURN, HistoryKind.TAIL_CALL, HistoryKind.TRAP_RETURN})


@dataclass(frozen=True)
class HistoryEntry:
    """One control transfer the call outline shows.

    ``depth`` is the outline depth. A call and its return both sit at the depth of the caller, every
    other entry at the depth of the code it happens in. Depths compare only within one ``segment``.
    """

    kind: HistoryKind
    pc: int | None  # instruction that moved control: the jump, or the epc of a trap; None when not known
    target: int | None  # where execution went on: callee, return address, handler or start address
    depth: int
    # Increments at every START, at a CONTEXT_SWITCH to other code than the trapped one, and when a trap return loses
    # the stack
    segment: int
    instructions: int  # retired after this entry, up to the next one
    cause: int | None = None  # mcause of INTERRUPT, ECALL and FAULT
    tval: int | None = None  # mtval of ECALL and FAULT
    detail: str = ""  # gap reason, or why a return does not match the shadow stack
    interrupt: bool = False  # FAULT: the trap is an interrupt that ends the history


@dataclass(frozen=True)
class HistoryLoop:
    """A run of items that repeated ``count`` times in a row."""

    count: int
    body: tuple[HistoryItem, ...]  # the last pass
    instructions: int  # retired over every pass


HistoryItem = HistoryEntry | HistoryLoop


@dataclass(frozen=True)
class StackFrame:
    """A call, or a trap, that had not returned where the history ends: one frame of the call stack."""

    kind: HistoryKind  # CALL, INTERRUPT or ECALL
    pc: int | None  # the call instruction or the epc of the trap; None for a call from code outside the traced range
    returns_to: int | None  # where the frame returns to; None when not known
    cause: int | None = None  # mcause of a trap frame
    # A trap handler that ran the scheduler resumed this frame, so it can belong to another task
    unsure: bool = False


@dataclass(frozen=True)
class CodeLoop:
    """The loop the code ran in, without leaving it, where the history ends.

    The loop is the code from ``start``, where a jump or branch back goes, to ``end``, that jump, and the calls
    it makes. Code outside it, a trap or a loss of trace ends it, so its instructions count back to the last of these.
    """

    start: int
    end: int
    instructions: int  # retired in the loop, counting at most _LOOP_LOOKBACK of its first pass
    passes: int  # jumps back to start
    # The loop goes back to the first instruction of the trace, or the first after it lost trace, so it may have run
    # longer
    whole: bool


@dataclass(frozen=True)
class History:
    """The call outline of one core, oldest item first."""

    items: tuple[HistoryItem, ...]
    omitted: int  # older items left out of ``items``
    instructions: int  # retired up to the fault, or to the end of the trace
    fault: HistoryEntry | None  # the exception, or the interrupt that ends it, the history ends at
    # Where the history ends, the frames whose call or trap the trace shows, newest first. Older frames are not in
    # the trace: it starts, loses trace or switches tasks below them.
    stack: tuple[StackFrame, ...] = ()
    loop: CodeLoop | None = None

    def entries(self) -> Iterator[HistoryEntry]:
        """Yield every entry in order, with the last pass of each loop."""
        return iter_entries(self.items)


@dataclass(slots=True)
class _Frame:
    """One shadow stack frame."""

    depth: int  # outline depth of the code that entered the frame
    resume: tuple[int, ...]  # addresses that return into that code; empty when not known, which accepts any
    trap: bool
    # Under a trap that returned to its epc. Another task preempted at the same instruction resumes there too.
    resumed: bool = False
    # Trap frame: the handler ran the scheduler. Resumed frame: a trap that resumed it did, so returns that match
    # it do not show which task runs, as tasks that yield through the same functions return the same way.
    scheduled: bool = False
    # Entered from code outside the traced range, so it returns into untraced code
    untraced_caller: bool = False
    pc: int | None = None  # the call instruction or the epc of the trap
    kind: HistoryKind = HistoryKind.CALL
    cause: int | None = None  # mcause of a trap frame

    def accepts(self, target: int | None) -> bool:
        """Tell whether a return to ``target`` returns into this frame. ``None`` is untraced code."""
        if target is None:
            return True
        if self.untraced_caller:
            return False
        return not self.resume or target in self.resume


@dataclass(slots=True)
class _Loop:
    """The loop the code runs in, while the builder counts it."""

    start: int
    end: int
    depth: int
    segment: int
    instructions: int = 0
    passes: int = 1
    whole: bool = False

    def holds(self, pc: int, depth: int, segment: int) -> bool:
        """Tell whether the instruction at ``pc`` and ``depth`` can be part of the loop: in it, or in a call."""
        if segment != self.segment or depth < self.depth:
            return False
        return depth > self.depth or self.start <= pc <= self.end


class HistoryBuilder:
    """Streams trace events of one core into a ``History``.

    ``image`` names the function of a jump and its target, which tells a tail call from a jump inside a
    function. Without it, jumps that do not call or return are not recorded. An interrupt whose mcause
    exception code is in ``end_interrupts`` ends the history like a fault.
    """

    def __init__(self, image: CodeImage | None = None, *, end_interrupts: Collection[int] = ()) -> None:
        self._image = image
        self._end_interrupts = frozenset(end_interrupts)
        self._symbols: dict[int, str | None] = {}
        self._items: list[HistoryItem] = []
        self._keys: list[tuple[object, ...]] = []
        self._pending: HistoryEntry | None = None
        self._pending_instructions = 0
        self._stack: list[_Frame] = []
        self._depth = 0
        self._segment = -1
        self._running = False
        # Code outside the traced range runs, the stack stays valid until the trace enters the range again
        self._untraced = False
        self._fault: HistoryEntry | None = None
        self._instructions = 0
        # The trap return of a handler that ran the scheduler, where a later return can show that another task resumed.
        # None when not known: another such trap return resumed the same frames before it.
        self._switch: HistoryEntry | None = None
        self._switch_pending = False
        self._loop: _Loop | None = None
        # pc, depth and segment of the newest instructions; None where a trap or a loss of trace came between them
        self._recent: deque[tuple[int, int, int] | None] = deque(maxlen=_LOOP_LOOKBACK)
        self._since_loss = 0  # instructions since the trace started or last lost trace

    @property
    def done(self) -> bool:
        """True once the fault is recorded. Later events are ignored."""
        return self._fault is not None

    def push(self, event: TraceEvent) -> None:
        if self.done:
            return
        if isinstance(event, TraceInstruction):
            if self._untraced:
                self._reenter(event.pc)
            elif not self._running:
                self._start(event.pc)
            self._instructions += 1
            self._pending_instructions += 1
            self._follow_loop(event)
            self._on_instruction(event)
        elif isinstance(event, TraceSync):
            if self._untraced:
                self._reenter(event.pc)
            elif not self._running:
                self._start(event.pc)
        elif isinstance(event, TraceTrap):
            self._untraced = False
            self._on_trap(event)
        elif isinstance(event, TraceGap):
            if event.reason == GapReason.UNTRACED and self._running:
                self._on_untraced()
            else:
                self._on_gap(event)

    def close(self, last: int = 0) -> History:
        """Return the history, with only the ``last`` newest items when it is not zero."""
        self._finish_pending()
        items = _rebase(self._items[-last:] if last else self._items)
        fault = items[-1] if self._fault is not None else None
        assert fault is None or isinstance(fault, HistoryEntry)
        loop = self._loop
        return History(
            items=items,
            omitted=len(self._items) - len(items),
            instructions=self._instructions,
            fault=fault,
            stack=tuple(
                StackFrame(
                    kind=f.kind,
                    pc=f.pc,
                    returns_to=f.resume[0] if f.resume else None,
                    cause=f.cause,
                    unsure=f.resumed and f.scheduled,
                )
                for f in reversed(self._stack)
            ),
            loop=None if loop is None else CodeLoop(loop.start, loop.end, loop.instructions, loop.passes, loop.whole),
        )

    # ------------------------------------------------------------------
    # Events
    # ------------------------------------------------------------------

    def _on_instruction(self, insn: TraceInstruction) -> None:
        link = insn.link
        if link == LinkKind.CALL:
            if self._runs_scheduler(insn.next_pc):
                handler = next((f for f in reversed(self._stack) if f.trap), None)
                if handler is not None:
                    handler.scheduled = True
            self._open(HistoryKind.CALL, insn.pc, insn.next_pc, self._depth)
            self._push_frame(_Frame(self._depth, (insn.pc + insn.length,), trap=False, pc=insn.pc))
            self._depth += 1
        elif link == LinkKind.RETURN:
            self._on_return(insn)
        elif link == LinkKind.TRAP_RETURN:
            self._on_trap_return(insn)
        elif link in (LinkKind.JUMP, LinkKind.COROUTINE) and self._leaves_function(insn):
            self._open(HistoryKind.TAIL_CALL, insn.pc, insn.next_pc, self._depth)

    def _on_return(self, insn: TraceInstruction) -> None:
        target = insn.next_pc
        detail = ""
        frame: _Frame | None = None
        if self._stack and not self._stack[-1].trap:
            # A return into untraced code cannot be checked against the call
            frame = self._stack.pop() if target is None else self._pop_to_call(target)
            if frame is None and self._stack[-1].resumed:
                scheduled = self._stack[-1].scheduled
                if not self._lose_resumed_stack(scheduled):
                    detail = (
                        f"task switch: {_SCHEDULER_SWITCH}"
                        if scheduled
                        else "returns into other code than the interrupted one, likely another task"
                    )
            elif frame is None:
                frame = self._stack.pop()
                expected = f"0x{frame.resume[0]:08x}" if frame.resume else "untraced code"
                detail = f"return address does not match the call, which returns to {expected}"
        if frame is not None:
            self._depth = frame.depth
            if frame.resumed and not frame.scheduled and not detail and target is not None:
                # The return proves that the trap resumed the code it interrupted
                for below in self._stack:
                    below.resumed = False
        else:
            # The caller is older than the trace, or the trace entered this code without a call
            self._depth -= 1
        self._open(HistoryKind.RETURN, insn.pc, target, self._depth, detail=detail)

    def _lose_resumed_stack(self, scheduled: bool) -> bool:
        """Drop a stack that a trap return resumed but a return shows to belong to other code.

        The other code entered its frames through the same instruction, so the depths go on. Return whether the trap
        return that resumed it is known, which then becomes the context switch: the last entry, or the trap return of
        the handler that ran the scheduler.
        """
        switch = self._switch
        self._stack.clear()
        self._switch = None
        self._switch_pending = False
        detail = _SCHEDULER_SWITCH if scheduled else ""
        pending = self._pending
        if pending is not None and pending.kind == HistoryKind.TRAP_RETURN and not pending.detail:
            self._pending = replace(pending, kind=HistoryKind.CONTEXT_SWITCH, detail=detail)
            return True
        if not scheduled or switch is None:
            return False
        # A loop that took the trap return in leaves only its last pass, so the switch is not an entry then
        at = next((i for i in range(len(self._items) - 1, -1, -1) if self._items[i] is switch), None)
        if at is None:
            return False
        entry = replace(switch, kind=HistoryKind.CONTEXT_SWITCH, detail=detail)
        self._items[at] = entry
        self._keys[at] = _entry_key(entry)
        return True

    def _pop_to_call(self, target: int) -> _Frame | None:
        """Pop the newest call frame that ``target`` returns into, and the frames above it."""
        for i in range(len(self._stack) - 1, -1, -1):
            frame = self._stack[i]
            if frame.trap:
                return None
            if frame.accepts(target):
                del self._stack[i:]
                return frame
        return None

    def _on_trap_return(self, insn: TraceInstruction) -> None:
        target = insn.next_pc
        trap_at = next((i for i in range(len(self._stack) - 1, -1, -1) if self._stack[i].trap), None)
        if trap_at is None:
            # The trap entry is older than the trace, so the call stack of the code it returns to is not known
            self._new_segment()
            self._open(HistoryKind.TRAP_RETURN, insn.pc, target, 0, detail="trap entry not in the trace")
            return
        frame = self._stack[trap_at]
        if frame.accepts(target):
            del self._stack[trap_at:]
            # When an earlier trap return after a scheduler run resumed these frames too, either one can be the switch
            switch_known = not any(f.resumed and f.scheduled for f in self._stack)
            for resumed in self._stack:
                resumed.resumed = True
                resumed.scheduled = resumed.scheduled or frame.scheduled
            self._depth = frame.depth
            self._open(HistoryKind.TRAP_RETURN, insn.pc, target, self._depth)
            if frame.scheduled:
                self._switch = None
                self._switch_pending = switch_known
            return
        self._new_segment()
        self._open(HistoryKind.CONTEXT_SWITCH, insn.pc, target, 0)

    def _on_trap(self, trap: TraceTrap) -> None:
        if not self._running:
            self._new_segment()
            self._running = True
        epc = trap.epc
        ends = trap.interrupt and trap.cause in self._end_interrupts
        if ends or is_fault(trap):
            tval = None if trap.interrupt else trap.tval
            self._open(
                HistoryKind.FAULT, epc, trap.handler_pc, self._depth, cause=trap.cause, tval=tval, interrupt=ends
            )
            self._fault = self._pending
            self._finish_pending()
            return
        self._break_loop()
        if trap.interrupt:
            kind, tval = HistoryKind.INTERRUPT, None
            resume: tuple[int, ...] = () if epc is None else (epc,)
        else:
            # The handler returns past the ecall, or retries it
            kind, tval = HistoryKind.ECALL, trap.tval
            resume = () if epc is None else (epc + _ECALL_LENGTH, epc)
        self._open(kind, epc, trap.handler_pc, self._depth, cause=trap.cause, tval=tval)
        self._push_frame(_Frame(self._depth, resume, trap=True, pc=epc, kind=kind, cause=trap.cause))
        self._depth += 1

    def _on_gap(self, gap: TraceGap) -> None:
        pending = self._pending
        if (
            pending is not None
            and pending.kind == HistoryKind.GAP
            and pending.detail == gap.reason
            and self._pending_instructions == 0
        ):
            return
        self._open(HistoryKind.GAP, None, None, self._depth, detail=gap.reason)
        self._running = False
        self._untraced = False
        self._stack.clear()
        self._break_loop()
        self._since_loss = 0

    def _on_untraced(self) -> None:
        """Keep the stack while code outside the traced range runs, which returns or calls back into it."""
        pending = self._pending
        left_by_jump = (
            pending is not None
            and pending.target is None
            and pending.kind in _JUMP_KINDS
            and self._pending_instructions == 0
        )
        if not left_by_jump:
            self._open(HistoryKind.GAP, None, None, self._depth, detail=GapReason.UNTRACED)
        self._untraced = True

    def _reenter(self, pc: int) -> None:
        """Continue at ``pc``, where the trace enters the traced range again."""
        self._untraced = False
        frame = self._pop_to_call(pc)
        if frame is not None:
            self._depth = frame.depth
            self._open(HistoryKind.RETURN, None, pc, self._depth)
            return
        # Untraced code calls into the traced range, as ROM code calls back into the application
        self._open(HistoryKind.CALL, None, pc, self._depth)
        self._push_frame(_Frame(self._depth, (), trap=False, untraced_caller=True))
        self._depth += 1

    # ------------------------------------------------------------------
    # State
    # ------------------------------------------------------------------

    def _start(self, pc: int) -> None:
        self._new_segment()
        self._running = True
        self._open(HistoryKind.START, None, pc, 0)

    def _new_segment(self) -> None:
        self._segment += 1
        self._stack.clear()
        self._depth = 0
        self._untraced = False
        self._switch = None
        self._switch_pending = False

    def _push_frame(self, frame: _Frame) -> None:
        self._stack.append(frame)
        if len(self._stack) > _MAX_FRAMES:
            del self._stack[0]

    def _follow_loop(self, insn: TraceInstruction) -> None:
        """Count ``insn`` into the loop the code runs in, or start a loop at a jump back."""
        depth, segment = self._depth, self._segment
        loop = self._loop
        if loop is not None and not loop.holds(insn.pc, depth, segment):
            self._loop = loop = None
        if loop is not None:
            loop.instructions += 1
        self._recent.append((insn.pc, depth, segment))
        self._since_loss += 1
        target = insn.next_pc
        if target is None or not self._jumps_back(insn):
            return
        if loop is None or (depth == loop.depth and target < loop.start):
            self._loop = self._new_loop(_Loop(target, insn.pc, depth, segment))
        elif (target, insn.pc, depth) == (loop.start, loop.end, loop.depth):
            loop.passes += 1

    def _new_loop(self, loop: _Loop) -> _Loop:
        """Count the first pass of ``loop`` back from its jump back, the newest instruction."""
        in_call = 0  # instructions in a call, part of the loop only when the loop made the call
        for item in reversed(self._recent):
            if item is None or not loop.holds(*item):
                break
            if item[1] > loop.depth:
                in_call += 1
            else:
                loop.instructions += 1 + in_call
                in_call = 0
        loop.whole = loop.instructions == self._since_loss
        return loop

    def _jumps_back(self, insn: TraceInstruction) -> bool:
        """Tell whether ``insn`` jumps or branches back inside its function, as the end of a loop does."""
        if insn.next_pc is None or insn.next_pc > insn.pc:
            return False
        if insn.cls == InsnClass.BRANCH:
            return bool(insn.branch_taken)
        return insn.cls == InsnClass.JUMP and insn.link == LinkKind.JUMP and not self._leaves_function(insn)

    def _break_loop(self) -> None:
        self._loop = None
        self._recent.append(None)

    def _runs_scheduler(self, target: int | None) -> bool:
        return self._image is not None and target is not None and self._symbol(target) in SCHEDULER_FUNCTIONS

    def _leaves_function(self, insn: TraceInstruction) -> bool:
        if insn.next_pc is None:
            return True
        if self._image is None:
            return False
        return self._symbol(insn.next_pc) != self._symbol(insn.pc)

    def _symbol(self, pc: int) -> str | None:
        if pc not in self._symbols:
            assert self._image is not None
            self._symbols[pc] = self._image.symbol_at(pc)
        return self._symbols[pc]

    def _open(
        self,
        kind: HistoryKind,
        pc: int | None,
        target: int | None,
        depth: int,
        *,
        cause: int | None = None,
        tval: int | None = None,
        detail: str = "",
        interrupt: bool = False,
    ) -> None:
        """Finish the pending entry and start counting instructions for a new one."""
        self._finish_pending()
        self._pending = HistoryEntry(
            kind=kind,
            pc=pc,
            target=target,
            depth=depth,
            segment=self._segment,
            instructions=0,
            cause=cause,
            tval=tval,
            detail=detail,
            interrupt=interrupt,
        )
        self._pending_instructions = 0

    def _finish_pending(self) -> None:
        if self._pending is None:
            return
        entry = replace(self._pending, instructions=self._pending_instructions)
        self._pending = None
        if self._switch_pending:
            self._switch = entry
            self._switch_pending = False
        self._items.append(entry)
        self._keys.append(_entry_key(entry))
        while self._fold():
            pass

    def _fold(self) -> bool:
        """Fold the newest items into a loop when they repeat the items before them."""
        items, keys = self._items, self._keys
        n = len(items)
        # Another pass of the loop just before
        for period in range(1, min(MAX_LOOP_PERIOD, n - 1) + 1):
            start = n - period - 1
            prev = items[start]
            if (
                isinstance(prev, HistoryLoop)
                and len(prev.body) == period
                and keys[start][1] == tuple(keys[start + 1 :])
            ):
                body = tuple(items[start + 1 :])
                self._replace_tail(start, HistoryLoop(prev.count + 1, body, prev.instructions + _instructions(body)))
                return True
        # Two passes in a row
        for period in range(1, min(MAX_LOOP_PERIOD, n // 2) + 1):
            start = n - 2 * period
            if keys[start : n - period] == keys[n - period :]:
                body = tuple(items[n - period :])
                self._replace_tail(start, HistoryLoop(2, body, _instructions(items[start:])))
                return True
        return False

    def _replace_tail(self, start: int, loop: HistoryLoop) -> None:
        body_keys = tuple(self._keys[len(self._keys) - len(loop.body) :])
        del self._items[start:]
        del self._keys[start:]
        self._items.append(loop)
        self._keys.append(("loop", body_keys))


def build_history(
    events: Iterable[TraceEvent],
    image: CodeImage | None = None,
    *,
    last: int = 0,
    end_interrupts: Collection[int] = (),
) -> History:
    """Build the history of one core's events, reading them only up to the fault.

    An interrupt whose mcause exception code is in ``end_interrupts`` ends the history like a fault.
    """
    builder = HistoryBuilder(image, end_interrupts=end_interrupts)
    for event in events:
        builder.push(event)
        if builder.done:
            break
    return builder.close(last)


def _entry_key(entry: HistoryEntry) -> tuple[object, ...]:
    """What makes two entries the same step of a loop. Instruction counts and segments may differ."""
    return (entry.kind, entry.pc, entry.target, entry.depth, entry.cause, entry.detail)


def _instructions(items: Iterable[HistoryItem]) -> int:
    return sum(item.instructions for item in items)


def iter_entries(items: Iterable[HistoryItem]) -> Iterator[HistoryEntry]:
    """Yield the entries of ``items`` in order, with the last pass of each loop."""
    for item in items:
        if isinstance(item, HistoryLoop):
            yield from iter_entries(item.body)
        else:
            yield item


def _rebase(items: Sequence[HistoryItem]) -> tuple[HistoryItem, ...]:
    """Shift depths so the shallowest entry of each segment in ``items`` is at depth 0."""
    lowest: dict[int, int] = {}
    for entry in iter_entries(items):
        lowest[entry.segment] = min(lowest.get(entry.segment, entry.depth), entry.depth)

    def shift(item: HistoryItem) -> HistoryItem:
        if isinstance(item, HistoryLoop):
            return replace(item, body=tuple(shift(i) for i in item.body))
        return replace(item, depth=item.depth - lowest[item.segment])

    return tuple(shift(item) for item in items)


__all__ = [
    "EXCEPTION_NAMES",
    "MAX_LOOP_PERIOD",
    "SCHEDULER_FUNCTIONS",
    "CodeLoop",
    "History",
    "HistoryBuilder",
    "HistoryEntry",
    "HistoryItem",
    "HistoryKind",
    "HistoryLoop",
    "StackFrame",
    "build_history",
    "cause_name",
    "is_fault",
    "iter_entries",
]
