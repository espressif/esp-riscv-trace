# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Per-core diagnostic accumulator.

Tracks event counts, gap reasons, repeated gaps, and possible root
causes across a trace event stream for summary output.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

from ..decoder.reconstruct import GapReason
from ..ir.capture_note import TraceCaptureNote
from ..ir.events import (
    TraceBoundary,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
    TraceWarning,
)
from ..snapshot import abi

DIAG_ROOT_NO_PCS_RECOVERED = "no_pcs_recovered"

_HINT_RESYNC = "enable packet resync in the encoder config"
_HINT_NO_SYNC_POINT = "no synchronization point in this window. Capture a longer trace"

_TOP_GAP_REASONS = 3

_TOP_REPEATED_GAPS = 5

_REPEATED_GAP_MIN_COUNT = 2


def _top_counts(counts: Counter[str]) -> tuple[tuple[str, int], ...]:
    return tuple(sorted(counts.items(), key=lambda kv: (-kv[1], kv[0])))


@dataclass(frozen=True)
class TraceDiagnostic:
    """Per-core diagnostic summary."""

    instructions: int
    syncs: int
    traps: int
    gaps: int
    gap_reasons: Counter[str]
    top_gap_reasons: tuple[tuple[str, int], ...]
    root_cause: str | None = None
    hint: str | None = None
    repeated_gaps: tuple[tuple[str, str, int], ...] = ()
    warnings: int = 0
    warning_reasons: tuple[tuple[str, int], ...] = ()
    boundaries: int = 0
    boundary_kinds: tuple[tuple[str, int], ...] = ()


class DiagnosticAccumulator:
    """Collects diagnostic counters from a stream of trace events."""

    __slots__ = (
        "_boundary_kinds",
        "_gap_reasons",
        "_isolated_gap_details",
        "_last_gap_reason",
        "_warning_reasons",
        "boundaries",
        "gaps",
        "instructions",
        "syncs",
        "traps",
        "warnings",
    )

    def __init__(self) -> None:
        self.instructions: int = 0
        self.syncs: int = 0
        self.traps: int = 0
        self.gaps: int = 0
        self.warnings: int = 0
        self.boundaries: int = 0
        self._gap_reasons: Counter[str] = Counter()
        self._warning_reasons: Counter[str] = Counter()
        self._boundary_kinds: Counter[str] = Counter()
        self._last_gap_reason: str | None = None
        self._isolated_gap_details: Counter[tuple[str, str]] = Counter()

    def push(self, event: TraceEvent) -> None:
        if isinstance(event, TraceInstruction):
            self.instructions += 1
            self._last_gap_reason = None
        elif isinstance(event, TraceSync):
            self.syncs += 1
            self._last_gap_reason = None
        elif isinstance(event, TraceTrap):
            self.traps += 1
            self._last_gap_reason = None
        elif isinstance(event, TraceGap):
            self.gaps += 1
            self._gap_reasons[event.reason] += 1
            # Code outside the traced range runs often, as every call into ROM, and is no decoding problem
            if event.reason != self._last_gap_reason and event.reason != GapReason.UNTRACED:
                self._isolated_gap_details[(event.reason, event.detail)] += 1
            self._last_gap_reason = event.reason
        elif isinstance(event, TraceWarning):
            self.warnings += 1
            self._warning_reasons[event.kind] += 1
            self._last_gap_reason = None
        elif isinstance(event, TraceBoundary):
            self.boundaries += 1
            self._boundary_kinds[event.kind] += 1
            self._last_gap_reason = None

    def close(
        self,
        note: TraceCaptureNote,
    ) -> TraceDiagnostic:
        gap_reasons = self._gap_reasons.copy()
        top = _top_counts(gap_reasons)[:_TOP_GAP_REASONS]
        repeated = self._repeated_gaps_top()
        root_cause = self._pick_root_cause()
        hint = self._pick_hint(root_cause, note)
        return TraceDiagnostic(
            instructions=self.instructions,
            syncs=self.syncs,
            traps=self.traps,
            gaps=self.gaps,
            gap_reasons=gap_reasons,
            top_gap_reasons=top,
            root_cause=root_cause,
            hint=hint,
            repeated_gaps=repeated,
            warnings=self.warnings,
            warning_reasons=_top_counts(self._warning_reasons),
            boundaries=self.boundaries,
            boundary_kinds=_top_counts(self._boundary_kinds),
        )

    def _repeated_gaps_top(self) -> tuple[tuple[str, str, int], ...]:
        items = [
            (reason, detail, count)
            for (reason, detail), count in self._isolated_gap_details.items()
            if count >= _REPEATED_GAP_MIN_COUNT
        ]
        items.sort(key=lambda x: (-x[2], x[0], x[1]))
        return tuple(items[:_TOP_REPEATED_GAPS])

    def _pick_root_cause(self) -> str | None:
        if self.instructions == 0 and self.syncs == 0 and self.gaps > 0:
            return DIAG_ROOT_NO_PCS_RECOVERED
        return None

    def _pick_hint(
        self,
        root_cause: str | None,
        note: TraceCaptureNote,
    ) -> str | None:
        if root_cause != DIAG_ROOT_NO_PCS_RECOVERED:
            return None
        disabled = abi.enum_name(abi.ResyncMode, int(abi.ResyncMode.DISABLED))
        unknown = abi.enum_name(abi.ResyncMode, int(abi.ResyncMode.UNKNOWN))
        # Resync already on. A longer capture is the likely fix.
        if note.resync_mode not in (disabled, unknown):
            return _HINT_NO_SYNC_POINT
        return _HINT_RESYNC


__all__ = [
    "DIAG_ROOT_NO_PCS_RECOVERED",
    "DiagnosticAccumulator",
    "TraceDiagnostic",
]
