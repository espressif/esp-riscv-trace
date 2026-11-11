# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Streaming statistics accumulator for trace events.

Counts instruction hits by source line and function block entry points.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
from dataclasses import dataclass

from ..elf import CodeImage, SourceResolver
from ..ir.events import (
    InsnClass,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
)

UNRESOLVED_FILE = "<unresolved>"
UNRESOLVED_LINE = 0


def unresolved_function_label(pc: int) -> str:
    """Return the placeholder label used when no symbol covers pc."""
    return f"<0x{pc:08x}:unresolved>"


_TERMINATOR_CLASSES = frozenset(
    {
        InsnClass.BRANCH,
        InsnClass.JUMP,
        InsnClass.UNINFERABLE,
        InsnClass.TRAP,
    }
)


@dataclass(frozen=True)
class StatsSnapshot:
    """Snapshot of accumulated statistics for one core."""

    sample_hits: Counter[tuple[str, str, int]]
    block_hits: Counter[tuple[str, int]]
    unresolved_pcs: Counter[str]
    total_instructions: int

    @classmethod
    def merge(cls, snapshots: Iterable[StatsSnapshot]) -> StatsSnapshot:
        """Sum every counter across ``snapshots``."""
        sample: Counter[tuple[str, str, int]] = Counter()
        block: Counter[tuple[str, int]] = Counter()
        unresolved: Counter[str] = Counter()
        total = 0
        for snap in snapshots:
            sample.update(snap.sample_hits)
            block.update(snap.block_hits)
            unresolved.update(snap.unresolved_pcs)
            total += snap.total_instructions
        return cls(sample_hits=sample, block_hits=block, unresolved_pcs=unresolved, total_instructions=total)


class StatsAccumulator:
    """Accumulates instruction counts and block entry hits from trace events."""

    __slots__ = (
        "_block_hits",
        "_image",
        "_in_block",
        "_resolver",
        "_sample_hits",
        "_total_instructions",
        "_unresolved",
    )

    def __init__(self, image: CodeImage, resolver: SourceResolver) -> None:
        self._image = image
        self._resolver = resolver
        self._sample_hits: Counter = Counter()
        self._block_hits: Counter = Counter()
        self._unresolved: Counter = Counter()
        self._total_instructions: int = 0
        self._in_block: bool = False

    def push(self, event: TraceEvent) -> None:
        if isinstance(event, TraceInstruction):
            self._push_insn(event)
        elif isinstance(event, (TraceSync, TraceTrap, TraceGap)):
            self._in_block = False

    def close(self) -> StatsSnapshot:
        return StatsSnapshot(
            sample_hits=self._sample_hits.copy(),
            block_hits=self._block_hits.copy(),
            unresolved_pcs=self._unresolved.copy(),
            total_instructions=self._total_instructions,
        )

    def _push_insn(self, event: TraceInstruction) -> None:
        pc = event.pc
        function = self._image.symbol_at(pc)
        if function is None:
            function = unresolved_function_label(pc)
            self._unresolved["function"] += 1

        loc = self._resolver.resolve(pc)
        if loc is None:
            file_name = UNRESOLVED_FILE
            line = UNRESOLVED_LINE
            self._unresolved["line"] += 1
        else:
            file_name = loc.file
            line = loc.line

        self._sample_hits[(function, file_name, line)] += 1
        self._total_instructions += 1

        if not self._in_block:
            self._block_hits[(function, pc)] += 1

        if event.cls in _TERMINATOR_CLASSES:
            self._in_block = False
        else:
            self._in_block = True


__all__ = [
    "UNRESOLVED_FILE",
    "UNRESOLVED_LINE",
    "StatsAccumulator",
    "StatsSnapshot",
    "unresolved_function_label",
]
