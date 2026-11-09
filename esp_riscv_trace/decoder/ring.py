# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Linearize ring buffers into chronological byte order."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..containers.base import TraceBuffer
from ..errors import EspRiscvTraceError
from ..snapshot import abi


class RingError(EspRiscvTraceError):
    """Raised when the buffer cannot be linearized."""


class SegmentKind(StrEnum):
    """Role of a physical region in the linearized stream."""

    LINEAR = "linear"
    LOOP_NO_WRAP = "loop_no_wrap"
    LOOP_WRAP_ALIGNED = "loop_wrap_aligned"
    LOOP_PRE_WRAP = "loop_pre_wrap"
    LOOP_POST_WRAP = "loop_post_wrap"


class WrapInference(StrEnum):
    """How the wrap state was decided."""

    LINEAR = "linear_mode"
    EXPLICIT = "explicit"
    NONZERO_TAIL = "nonzero_tail"
    CLEAN_TAIL = "zero_tail_clean"
    HEAD_AT_END = "head_at_capacity"


@dataclass(frozen=True)
class Segment:
    """One chronological byte span within a linearized trace.

    ``start`` and ``length`` locate the segment in the linearized stream.
    ``source_start`` locates it in the original physical buffer. The
    segments are chronological, but packet framing at their physical seam
    must be verified rather than assumed.
    """

    start: int
    length: int
    source_start: int
    kind: SegmentKind

    @property
    def end(self) -> int:
        return self.start + self.length


@dataclass(frozen=True)
class LinearizedBuffer:
    """Chronological bytes with segment provenance for one core."""

    core_id: int
    data: bytes
    segments: tuple[Segment, ...]
    seam: int | None
    wrap_state: int
    wrap_state_inferred: bool
    inference_reason: str
    original: TraceBuffer

    def __post_init__(self) -> None:
        if self.wrap_state == int(abi.WrapState.UNKNOWN):
            raise RingError("linearized wrap_state must be resolved, not UNKNOWN")
        if not self.segments:
            raise RingError("linearized buffer must have at least one segment")
        total = sum(s.length for s in self.segments)
        if total != len(self.data):
            raise RingError(f"segment lengths {total} do not match data length {len(self.data)}")
        cursor = 0
        capacity = self.original.capacity
        for i, s in enumerate(self.segments):
            if s.length < 0:
                raise RingError(f"segment {i} has negative length {s.length}")
            if s.start != cursor:
                raise RingError(f"segment {i} start {s.start} does not follow the previous end {cursor}")
            if s.source_start < 0 or s.source_start + s.length > capacity:
                raise RingError(
                    f"segment {i} source range [{s.source_start}, "
                    f"{s.source_start + s.length}) outside physical buffer [0, {capacity})"
                )
            source_bytes = self.original.raw[s.source_start : s.source_start + s.length]
            if self.data[s.start : s.end] != source_bytes:
                raise RingError(
                    f"segment {i} data does not match physical bytes at [{s.source_start}, {s.source_start + s.length})"
                )
            cursor = s.end
        if len(self.segments) == 2:
            if self.seam is None:
                raise RingError("two-segment layout requires a seam offset")
            if self.seam != self.segments[0].end:
                raise RingError(f"seam {self.seam} must equal first segment end {self.segments[0].end}")
        elif len(self.segments) == 1:
            if self.seam is not None:
                raise RingError(f"single-segment layout must not have a seam (got {self.seam})")
        else:
            raise RingError(f"unexpected segment count {len(self.segments)}, expected 1 or 2")

    def slice(self, segment: Segment) -> bytes:
        """Return the linearized bytes for one segment."""
        return self.data[segment.start : segment.end]


def infer_wrap_state(raw: bytes, head_offset: int, capacity: int) -> tuple[int, str]:
    """Mirror ``buffer_has_wrapped``. Return ``(wrap_state, reason)``.

    The tail after ``head_offset`` is examined. Any non-zero byte marks the
    ring as wrapped. When ``head_offset == capacity`` the tail is empty and
    the buffer is treated as not wrapped, matching the target-side check.
    """
    if head_offset < 0 or head_offset > capacity:
        raise RingError(f"head_offset {head_offset} outside [0, {capacity}]")
    if len(raw) != capacity:
        raise RingError(f"buffer length {len(raw)} does not match capacity {capacity}")
    if head_offset >= capacity:
        return int(abi.WrapState.NOT_WRAPPED), WrapInference.HEAD_AT_END
    if any(raw[head_offset:capacity]):
        return int(abi.WrapState.WRAPPED), WrapInference.NONZERO_TAIL
    return int(abi.WrapState.NOT_WRAPPED), WrapInference.CLEAN_TAIL


def linearize(buf: TraceBuffer) -> LinearizedBuffer:
    """Return the chronological bytes and segment layout for one buffer."""

    if not buf.head_valid:
        raise RingError(f"core {buf.core_id} has head_valid clear, head_offset is a placeholder")

    memory_mode = buf.memory_mode
    if memory_mode not in (int(abi.MemoryMode.LINEAR), int(abi.MemoryMode.LOOP)):
        raise RingError(f"core {buf.core_id} has unsupported memory_mode {memory_mode}")

    wrap_state, reason, inferred = _resolve_wrap_state(buf)

    def make(data: bytes, segments: tuple[Segment, ...], seam: int | None = None) -> LinearizedBuffer:
        return LinearizedBuffer(
            core_id=buf.core_id,
            data=data,
            segments=segments,
            seam=seam,
            wrap_state=wrap_state,
            wrap_state_inferred=inferred,
            inference_reason=reason,
            original=buf,
        )

    head = buf.head_offset
    capacity = buf.capacity
    if memory_mode == int(abi.MemoryMode.LINEAR):
        return make(buf.raw[:head], (Segment(start=0, length=head, source_start=0, kind=SegmentKind.LINEAR),))
    if wrap_state == int(abi.WrapState.NOT_WRAPPED):
        return make(buf.raw[:head], (Segment(start=0, length=head, source_start=0, kind=SegmentKind.LOOP_NO_WRAP),))
    if head == 0 or head == capacity:
        return make(buf.raw, (Segment(start=0, length=capacity, source_start=0, kind=SegmentKind.LOOP_WRAP_ALIGNED),))

    pre_len = capacity - head
    pre = Segment(start=0, length=pre_len, source_start=head, kind=SegmentKind.LOOP_PRE_WRAP)
    post = Segment(start=pre_len, length=head, source_start=0, kind=SegmentKind.LOOP_POST_WRAP)
    return make(buf.raw[head:] + buf.raw[:head], (pre, post), pre_len)


_VALID_WRAP_STATES = (
    int(abi.WrapState.UNKNOWN),
    int(abi.WrapState.NOT_WRAPPED),
    int(abi.WrapState.WRAPPED),
)


def _resolve_wrap_state(buf: TraceBuffer) -> tuple[int, str, bool]:
    if buf.wrap_state not in _VALID_WRAP_STATES:
        raise RingError(f"core {buf.core_id} has unrecognized wrap_state {buf.wrap_state}")
    if buf.memory_mode == int(abi.MemoryMode.LINEAR):
        return int(abi.WrapState.NOT_WRAPPED), WrapInference.LINEAR, False
    if buf.wrap_state == int(abi.WrapState.WRAPPED):
        return int(abi.WrapState.WRAPPED), WrapInference.EXPLICIT, False
    if buf.wrap_state == int(abi.WrapState.NOT_WRAPPED):
        return int(abi.WrapState.NOT_WRAPPED), WrapInference.EXPLICIT, False
    ws, reason = infer_wrap_state(buf.raw, buf.head_offset, buf.capacity)
    return ws, reason, True


__all__ = [
    "LinearizedBuffer",
    "RingError",
    "Segment",
    "SegmentKind",
    "WrapInference",
    "infer_wrap_state",
    "linearize",
]
