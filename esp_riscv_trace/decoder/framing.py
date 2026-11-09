# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Espressif trace memory framing and packet boundary parser."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from .ring import LinearizedBuffer, Segment

HEADER_SIZE = 1
INDEX_SIZE = 2
PREFIX_SIZE = HEADER_SIZE + INDEX_SIZE
PACKET_MIN_LEN = 4
PACKET_MAX_LEN = 13
PAYLOAD_MAX_LEN = PACKET_MAX_LEN - PREFIX_SIZE
ANCHOR_ZERO_RUN_BYTES = 14
INDEX_MASK = 0xFFFF
HEADER_LEN_MASK = 0x1F
HEADER_RESERVED_MASK = 0xE0


class FramingEventKind(StrEnum):
    """Kind of a structural event found while framing the byte stream."""

    ANCHOR_FOUND = "anchor_found"
    NO_ANCHOR = "no_anchor"
    PRE_ANCHOR_SKIP = "pre_anchor_skip"
    ALIGNMENT_LOST = "alignment_lost"
    INDEX_JUMP = "index_jump"
    FRAMING_ERROR = "framing_error"
    PARTIAL_FRAME = "partial_frame"
    PACKET_ERROR = "packet_error"
    WRAP_BOUNDARY = "wrap_boundary"
    TRACE_LOST = "trace_lost"


_EVENT_ORDER = {
    FramingEventKind.TRACE_LOST: 0,
    FramingEventKind.ANCHOR_FOUND: 2,
    FramingEventKind.WRAP_BOUNDARY: 3,
    FramingEventKind.NO_ANCHOR: 4,
    FramingEventKind.ALIGNMENT_LOST: 5,
    FramingEventKind.PRE_ANCHOR_SKIP: 5,
    FramingEventKind.INDEX_JUMP: 6,
    FramingEventKind.PARTIAL_FRAME: 7,
    FramingEventKind.PACKET_ERROR: 7,
    FramingEventKind.FRAMING_ERROR: 7,
}


@dataclass(frozen=True)
class Frame:
    """One framed packet within a trace segment."""

    segment_index: int
    stream_offset: int
    length: int
    packet_index: int
    payload: bytes


@dataclass(frozen=True)
class FramingEvent:
    """A structural event produced while walking trace memory."""

    kind: FramingEventKind
    segment_index: int
    stream_offset: int
    detail: str
    span: int = 0
    packet_index: int = -1
    payload: bytes | None = None


@dataclass(frozen=True)
class FramingResult:
    frames: tuple[Frame, ...]
    events: tuple[FramingEvent, ...]
    ring: LinearizedBuffer


def frame(ring: LinearizedBuffer) -> FramingResult:
    """Walk the linearized stream and return frames plus structural events.

    Physical wrap is reported separately from an execution gap. Each
    physical region is framed independently because ESP trace memory does
    not guarantee that a packet continues across the physical wrap. Packet
    and decoder continuity are kept only when both regions begin and
    end on verified frame boundaries and their packet indexes connect.
    """
    frames: list[Frame] = []
    events: list[FramingEvent] = []

    if ring.original.fifo_overflow:
        events.append(
            FramingEvent(
                kind=FramingEventKind.TRACE_LOST,
                segment_index=0,
                stream_offset=0,
                detail="hardware reported encoder FIFO overflow",
            )
        )

    if ring.seam is not None:
        events.append(
            FramingEvent(
                kind=FramingEventKind.WRAP_BOUNDARY,
                segment_index=1,
                stream_offset=ring.seam,
                detail="circular trace memory wraps here",
            )
        )

    previous_index: int | None = None
    previous_ended_cleanly = False
    for seg_idx, seg in enumerate(ring.segments):
        previous_index, previous_ended_cleanly = _walk_segment(
            seg_idx,
            seg,
            ring.slice(seg),
            frames,
            events,
            previous_index if previous_ended_cleanly else None,
            recovering_from_gap=seg_idx > 0 and not previous_ended_cleanly,
        )
    events.sort(key=lambda e: (e.stream_offset, _EVENT_ORDER.get(e.kind, 9)))
    return FramingResult(frames=tuple(frames), events=tuple(events), ring=ring)


def _walk_segment(
    seg_idx: int,
    seg: Segment,
    data: bytes,
    frames_out: list[Frame],
    events_out: list[FramingEvent],
    previous_index: int | None,
    *,
    recovering_from_gap: bool,
) -> tuple[int | None, bool]:
    """Frame one physical region.

    Return the last packet index and whether the region ended on a verified
    frame boundary. A caller may carry that index into the next region only
    when the returned clean-boundary flag is true.
    """
    found = _find_anchor(data)
    if found is None:
        events_out.append(
            FramingEvent(
                kind=FramingEventKind.NO_ANCHOR,
                segment_index=seg_idx,
                stream_offset=seg.start,
                detail="no anchor found in this physical region",
                span=seg.length,
            )
        )

    # Prefer a fully validated chain starting at the physical region boundary.
    # This is what permits a clean wrap to remain continuous even when the new
    # region contains no anchor. A single plausible header is not sufficient.
    if _is_complete_frame_chain(data):
        off = 0
    elif found is not None:
        off, tag_start = found
        if tag_start and not recovering_from_gap:
            events_out.append(
                FramingEvent(
                    kind=FramingEventKind.ALIGNMENT_LOST,
                    segment_index=seg_idx,
                    stream_offset=seg.start,
                    detail=(f"cannot reconstruct {tag_start} bytes before the first verified packet boundary"),
                    span=tag_start,
                )
            )
            previous_index = None
    elif not recovering_from_gap:
        events_out.append(
            FramingEvent(
                kind=FramingEventKind.FRAMING_ERROR,
                segment_index=seg_idx,
                stream_offset=seg.start,
                detail="cannot establish packet alignment in physical region",
                span=seg.length,
            )
        )
        return None, False
    else:
        # The preceding physical region already reported the discontinuity.
        # Do not count the other fragment of that same boundary break as a
        # second execution gap.
        return None, False

    if found is not None:
        anchor, tag_start = found
        events_out.append(
            FramingEvent(
                kind=FramingEventKind.ANCHOR_FOUND,
                segment_index=seg_idx,
                stream_offset=seg.start + anchor,
                detail=f"clean packet boundary after {anchor - tag_start}-byte anchor tag",
                span=anchor - tag_start,
            )
        )

    end = len(data)

    while off < end:
        if data[off] == 0:
            off = _skip_zeros(data, off)
            continue

        header = data[off]
        plen = header & HEADER_LEN_MASK
        reserved = header & HEADER_RESERVED_MASK

        if plen < PACKET_MIN_LEN or plen > PACKET_MAX_LEN:
            events_out.append(
                FramingEvent(
                    kind=FramingEventKind.FRAMING_ERROR,
                    segment_index=seg_idx,
                    stream_offset=seg.start + off,
                    detail=(f"invalid packet length {plen} (expected {PACKET_MIN_LEN}..{PACKET_MAX_LEN})"),
                    span=end - off,
                )
            )
            return None, False
        if off + plen > end:
            events_out.append(
                FramingEvent(
                    kind=FramingEventKind.PARTIAL_FRAME,
                    segment_index=seg_idx,
                    stream_offset=seg.start + off,
                    detail=(f"frame cannot be reconstructed ({end - off} of {plen} bytes at physical-region end)"),
                    span=end - off,
                )
            )
            return None, False
        if reserved != 0:
            events_out.append(
                FramingEvent(
                    kind=FramingEventKind.FRAMING_ERROR,
                    segment_index=seg_idx,
                    stream_offset=seg.start + off,
                    detail=f"reserved bits set in header 0x{header:02x}",
                    span=end - off,
                )
            )
            return None, False

        packet_index = data[off + 1] | (data[off + 2] << 8)
        payload = bytes(data[off + PREFIX_SIZE : off + plen])

        if previous_index is not None:
            expected = (previous_index + 1) & INDEX_MASK
            if packet_index != expected:
                forward_gap = (packet_index - expected) & INDEX_MASK
                events_out.append(
                    FramingEvent(
                        kind=FramingEventKind.INDEX_JUMP,
                        segment_index=seg_idx,
                        stream_offset=seg.start + off,
                        detail=(
                            f"packet index jump. prev={previous_index} got={packet_index} "
                            f"expected={expected} missing_packets={forward_gap}"
                        ),
                        span=0,
                    )
                )
        previous_index = packet_index

        frames_out.append(
            Frame(
                segment_index=seg_idx,
                stream_offset=seg.start + off,
                length=plen,
                packet_index=packet_index,
                payload=payload,
            )
        )
        off += plen
    return previous_index, True


def _skip_zeros(data: bytes, off: int) -> int:
    """Return the offset of the first non-zero byte at or after ``off``, or ``len(data)``."""
    end = len(data)
    while off < end and data[off] == 0:
        off += 1
    return off


def _is_complete_frame_chain(data: bytes) -> bool:
    """Return whether ``data`` is structurally framed from start to end.

    Packet-index discontinuities do not invalidate alignment; the framing
    walk reports those separately as execution gaps.
    """
    end = len(data)
    off = 0
    while off < end:
        if data[off] == 0:
            off = _skip_zeros(data, off)
            continue
        plen = data[off] & HEADER_LEN_MASK
        reserved = data[off] & HEADER_RESERVED_MASK
        if reserved != 0 or plen < PACKET_MIN_LEN or plen > PACKET_MAX_LEN or off + plen > end:
            return False
        off += plen
    return True


def _find_anchor(data: bytes) -> tuple[int, int] | None:
    """Return ``(anchor_offset, tag_start)`` for the first anchor tag.

    ``anchor_offset`` is the offset of the first non-zero byte following
    a zero-run of at least ``ANCHOR_ZERO_RUN_BYTES``. ``tag_start`` is the
    offset where that zero-run began. Returns ``None`` when no anchor tag
    is followed by a packet start within the data.
    """
    off = 0
    end = len(data)
    while off < end:
        if data[off] != 0:
            off += 1
            continue
        tag_start = off
        off = _skip_zeros(data, off)
        zero_run = off - tag_start
        if zero_run >= ANCHOR_ZERO_RUN_BYTES:
            if off >= end:
                return None
            return off, tag_start
    return None


__all__ = [
    "ANCHOR_ZERO_RUN_BYTES",
    "HEADER_LEN_MASK",
    "HEADER_RESERVED_MASK",
    "HEADER_SIZE",
    "INDEX_MASK",
    "INDEX_SIZE",
    "PACKET_MAX_LEN",
    "PACKET_MIN_LEN",
    "PAYLOAD_MAX_LEN",
    "PREFIX_SIZE",
    "Frame",
    "FramingEvent",
    "FramingEventKind",
    "FramingResult",
    "frame",
]
