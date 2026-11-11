# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Call stacks of a trace, for a profile in retired instructions.

``ProfileAccumulator`` follows the call stack effect of every jump (``TraceInstruction.link``) on a shadow stack and
counts each instruction under the functions it ran in, outermost first. A trap adds a frame named after its cause
above the interrupted function, so handler code shows apart from the code it interrupted.

The trace starts inside running code, so the outermost frames are only the ones whose call the trace shows. A
return to a caller older than the trace goes on in that caller with no frame below it. A loss of trace, and a trap
return into other code than the trapped one, usually another task, drop the stack the same way.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..decoder.reconstruct import GapReason
from ..elf import CodeImage
from ..ir.events import LinkKind, TraceEvent, TraceGap, TraceInstruction, TraceSync, TraceTrap
from .history import cause_name, is_fault
from .stats import unresolved_function_label

# Frame of code outside the traced range that called back into it, as ROM code calls the application
UNTRACED_FRAME = "[untraced]"

# ecall has no compressed form
_ECALL_LENGTH = 4

# Oldest frames are dropped beyond this, so runaway recursion cannot grow the shadow stack without bound
_MAX_FRAMES = 1024

# Call stack, outermost function first
Stack = tuple[str, ...]


def trap_frame(trap: TraceTrap) -> str:
    """Return the name of the frame a trap adds, such as ``[interrupt 25]`` or ``[load access fault]``."""
    return f"[{cause_name(trap.cause, trap.interrupt)}]"


@dataclass(frozen=True)
class Profile:
    """The instructions of one core by the call stack they ran under."""

    # (stack, instructions) for every run of instructions under one stack, in trace order
    runs: tuple[tuple[Stack, int], ...]
    instructions: int

    @classmethod
    def merge(cls, profiles: Iterable[Profile]) -> Profile:
        """Join the runs of ``profiles`` in order."""
        runs: list[tuple[Stack, int]] = []
        for profile in profiles:
            runs.extend(profile.runs)
        return cls(runs=tuple(runs), instructions=sum(count for _stack, count in runs))

    def stacks(self) -> Counter[Stack]:
        """Return the instructions of every stack, summed over its runs."""
        counts: Counter[Stack] = Counter()
        for stack, count in self.runs:
            counts[stack] += count
        return counts


@dataclass(slots=True)
class _Frame:
    """One shadow stack frame."""

    names: Stack  # what the frame adds to the stack: the caller, or the interrupted function and the trap
    resume: tuple[int, ...]  # addresses that return into the frame; empty when not known, which accepts any
    trap: bool
    # Entered from code outside the traced range, so it returns into untraced code
    untraced_caller: bool = False

    def accepts(self, target: int | None) -> bool:
        """Tell whether a return to ``target`` returns into this frame. ``None`` is untraced code."""
        if target is None:
            return True
        if self.untraced_caller:
            return False
        return not self.resume or target in self.resume


class ProfileAccumulator:
    """Streams trace events of one core into a ``Profile``. ``image`` names the functions."""

    def __init__(self, image: CodeImage) -> None:
        self._image = image
        self._symbols: dict[int, str] = {}
        self._frames: list[_Frame] = []
        self._outer: Stack | None = ()  # names of every frame; None when a frame changed since
        self._runs: list[tuple[Stack, int]] = []
        self._stack: Stack | None = None  # stack of the newest run
        self._count = 0  # instructions of the newest run
        # Outer frames and function of the newest instruction, so the next one can skip building its stack
        self._run_outer: Stack | None = None
        self._run_function: str | None = None
        self._function: str | None = None  # function of the newest instruction
        self._running = False
        # Code outside the traced range runs, the stack stays valid until the trace enters the range again
        self._untraced = False

    def push(self, event: TraceEvent) -> None:
        if isinstance(event, TraceInstruction):
            if self._untraced:
                self._reenter(event.pc)
            self._running = True
            self._count_instruction(event.pc)
            self._follow(event)
        elif isinstance(event, TraceSync):
            if self._untraced:
                self._reenter(event.pc)
        elif isinstance(event, TraceTrap):
            self._untraced = False
            self._on_trap(event)
        elif isinstance(event, TraceGap):
            if event.reason == GapReason.UNTRACED and self._running:
                self._untraced = True
            else:
                self._clear()
                self._running = False
                self._untraced = False
                self._function = None

    def close(self) -> Profile:
        runs = list(self._runs)
        if self._stack is not None:
            runs.append((self._stack, self._count))
        return Profile(runs=tuple(runs), instructions=sum(count for _stack, count in runs))

    def _count_instruction(self, pc: int) -> None:
        function = self._symbol(pc)
        self._function = function
        outer = self._outer
        if outer is None:
            outer = self._outer = tuple(name for frame in self._frames for name in frame.names)
        if outer is self._run_outer and function == self._run_function:
            self._count += 1
            return
        self._run_outer, self._run_function = outer, function
        stack = (*outer, function)
        if stack == self._stack:
            self._count += 1
            return
        if self._stack is not None:
            self._runs.append((self._stack, self._count))
        self._stack = stack
        self._count = 1

    def _follow(self, insn: TraceInstruction) -> None:
        link = insn.link
        if link == LinkKind.CALL:
            assert self._function is not None
            self._push(_Frame((self._function,), (insn.pc + insn.length,), trap=False))
        elif link == LinkKind.RETURN:
            self._on_return(insn.next_pc)
        elif link == LinkKind.TRAP_RETURN:
            trap_at = next((i for i in range(len(self._frames) - 1, -1, -1) if self._frames[i].trap), None)
            if trap_at is not None and self._frames[trap_at].accepts(insn.next_pc):
                self._pop(trap_at)
            else:
                # Back into code whose stack the trace does not show, usually another task
                self._clear()

    def _on_return(self, target: int | None) -> None:
        frames = self._frames
        if not frames or frames[-1].trap:
            # The caller is older than the trace, or the trace entered this code without a call
            return
        if target is None:
            self._pop(len(frames) - 1)
            return
        for i in range(len(frames) - 1, -1, -1):
            if frames[i].trap:
                break
            if frames[i].accepts(target):
                self._pop(i)
                return
        # The return does not match any call, so the stack below the newest trap is not this code's
        while frames and not frames[-1].trap:
            frames.pop()
        self._outer = None

    def _on_trap(self, trap: TraceTrap) -> None:
        epc = trap.epc
        interrupted = self._symbol(epc) if epc is not None else self._function
        names = (trap_frame(trap),) if interrupted is None else (interrupted, trap_frame(trap))
        resume: tuple[int, ...] = ()
        if epc is not None:
            # The handler returns past an ecall, or retries it
            ecall = not trap.interrupt and not is_fault(trap)
            resume = (epc + _ECALL_LENGTH, epc) if ecall else (epc,)
        self._push(_Frame(names, resume, trap=True))
        self._running = True

    def _reenter(self, pc: int) -> None:
        """Continue at ``pc``, where the trace enters the traced range again."""
        self._untraced = False
        frames = self._frames
        for i in range(len(frames) - 1, -1, -1):
            if frames[i].trap:
                break
            if frames[i].accepts(pc):
                self._pop(i)
                return
        self._push(_Frame((UNTRACED_FRAME,), (), trap=False, untraced_caller=True))

    def _push(self, frame: _Frame) -> None:
        self._frames.append(frame)
        if len(self._frames) > _MAX_FRAMES:
            del self._frames[0]
        self._outer = None

    def _pop(self, index: int) -> None:
        """Drop the frame at ``index`` and every frame above it."""
        del self._frames[index:]
        self._outer = None

    def _clear(self) -> None:
        self._frames.clear()
        self._outer = None

    def _symbol(self, pc: int) -> str:
        name = self._symbols.get(pc)
        if name is None:
            name = self._symbols[pc] = self._image.symbol_at(pc) or unresolved_function_label(pc)
        return name


def build_profile(events: Iterable[TraceEvent], image: CodeImage) -> Profile:
    """Build the profile of one core's events."""
    acc = ProfileAccumulator(image)
    for event in events:
        acc.push(event)
    return acc.close()


__all__ = [
    "UNTRACED_FRAME",
    "Profile",
    "ProfileAccumulator",
    "Stack",
    "build_profile",
    "trap_frame",
]
