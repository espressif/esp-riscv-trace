# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Ring buffer linearization and wrap state inference."""

from __future__ import annotations

import dataclasses

import pytest
from helpers import make_buffer

from esp_riscv_trace.decoder.ring import (
    LinearizedBuffer,
    RingError,
    Segment,
    SegmentKind,
    WrapInference,
    infer_wrap_state,
    linearize,
)
from esp_riscv_trace.snapshot import abi

_RAW = bytes(range(1, 17))
_LINEAR = int(abi.MemoryMode.LINEAR)
_LOOP = int(abi.MemoryMode.LOOP)
_UNKNOWN = int(abi.WrapState.UNKNOWN)
_NOT_WRAPPED = int(abi.WrapState.NOT_WRAPPED)
_WRAPPED = int(abi.WrapState.WRAPPED)


def _kinds(lin: LinearizedBuffer) -> list[SegmentKind]:
    return [s.kind for s in lin.segments]


def test_linear_mode_keeps_bytes_before_head() -> None:
    lin = linearize(make_buffer(_RAW, head=6, memory_mode=_LINEAR, wrap_state=_UNKNOWN))
    assert lin.data == _RAW[:6]
    assert _kinds(lin) == [SegmentKind.LINEAR]
    assert lin.seam is None
    assert lin.wrap_state == _NOT_WRAPPED
    assert lin.inference_reason == WrapInference.LINEAR
    assert not lin.wrap_state_inferred


def test_loop_not_wrapped_keeps_bytes_before_head() -> None:
    lin = linearize(make_buffer(_RAW, head=6, wrap_state=_NOT_WRAPPED))
    assert lin.data == _RAW[:6]
    assert _kinds(lin) == [SegmentKind.LOOP_NO_WRAP]
    assert lin.seam is None
    assert lin.inference_reason == WrapInference.EXPLICIT


@pytest.mark.parametrize("head", [0, len(_RAW)])
def test_loop_wrapped_at_buffer_edge_is_one_segment(head: int) -> None:
    lin = linearize(make_buffer(_RAW, head=head, wrap_state=_WRAPPED))
    assert lin.data == _RAW
    assert _kinds(lin) == [SegmentKind.LOOP_WRAP_ALIGNED]
    assert lin.seam is None
    assert lin.wrap_state == _WRAPPED


def test_loop_wrapped_mid_buffer_has_seam() -> None:
    lin = linearize(make_buffer(_RAW, head=6, wrap_state=_WRAPPED))
    assert lin.data == _RAW[6:] + _RAW[:6]
    assert _kinds(lin) == [SegmentKind.LOOP_PRE_WRAP, SegmentKind.LOOP_POST_WRAP]
    pre, post = lin.segments
    assert (pre.start, pre.length, pre.source_start) == (0, 10, 6)
    assert (post.start, post.length, post.source_start) == (10, 6, 0)
    assert lin.seam == 10
    assert lin.slice(pre) == _RAW[6:]
    assert lin.slice(post) == _RAW[:6]
    assert lin.inference_reason == WrapInference.EXPLICIT
    assert not lin.wrap_state_inferred


@pytest.mark.parametrize(
    ("raw", "head", "wrap_state", "reason"),
    [
        (_RAW[:6] + bytes(10), 6, _NOT_WRAPPED, WrapInference.CLEAN_TAIL),
        (_RAW[:6] + bytes(9) + b"\x01", 6, _WRAPPED, WrapInference.NONZERO_TAIL),
        (_RAW, len(_RAW), _NOT_WRAPPED, WrapInference.HEAD_AT_END),
    ],
    ids=["clean_tail", "nonzero_tail", "head_at_end"],
)
def test_loop_unknown_wrap_state_is_inferred(raw: bytes, head: int, wrap_state: int, reason: str) -> None:
    lin = linearize(make_buffer(raw, head=head, wrap_state=_UNKNOWN))
    assert lin.wrap_state == wrap_state
    assert lin.inference_reason == reason
    assert lin.wrap_state_inferred


@pytest.mark.parametrize(
    ("buf_kwargs", "fragment"),
    [
        (dict(head_valid=False), "head_valid clear"),
        (dict(memory_mode=int(abi.MemoryMode.UNKNOWN)), "unsupported memory_mode 0"),
        (dict(memory_mode=7), "unsupported memory_mode 7"),
        (dict(wrap_state=5), "unrecognized wrap_state 5"),
    ],
)
def test_linearize_rejects(buf_kwargs: dict, fragment: str) -> None:
    with pytest.raises(RingError, match=fragment):
        linearize(make_buffer(_RAW, head=6, **buf_kwargs))


@pytest.mark.parametrize(
    ("raw", "head", "capacity", "fragment"),
    [
        (_RAW, -1, 16, "head_offset -1 outside"),
        (_RAW, 17, 16, "head_offset 17 outside"),
        (_RAW, 4, 8, "buffer length 16 does not match capacity 8"),
    ],
)
def test_infer_wrap_state_rejects(raw: bytes, head: int, capacity: int, fragment: str) -> None:
    with pytest.raises(RingError, match=fragment):
        infer_wrap_state(raw, head, capacity)


def _two_segment() -> LinearizedBuffer:
    return linearize(make_buffer(_RAW, head=6, wrap_state=_WRAPPED))


_SEG_A = Segment(start=0, length=10, source_start=6, kind=SegmentKind.LOOP_PRE_WRAP)
_SEG_B = Segment(start=10, length=6, source_start=0, kind=SegmentKind.LOOP_POST_WRAP)


@pytest.mark.parametrize(
    ("changes", "fragment"),
    [
        (dict(wrap_state=_UNKNOWN), "must be resolved"),
        (dict(segments=()), "at least one segment"),
        (dict(data=_RAW[:8]), "segment lengths 16 do not match data length 8"),
        (dict(segments=(_SEG_A, dataclasses.replace(_SEG_B, start=11))), "does not follow the previous end"),
        (dict(segments=(_SEG_A, dataclasses.replace(_SEG_B, source_start=12))), "outside physical buffer"),
        (dict(segments=(_SEG_A, dataclasses.replace(_SEG_B, source_start=1))), "data does not match physical bytes"),
        (dict(seam=None), "two-segment layout requires a seam"),
        (dict(seam=9), "seam 9 must equal first segment end 10"),
        (dict(segments=(Segment(0, 16, 0, SegmentKind.LOOP_WRAP_ALIGNED),), data=_RAW), "must not have a seam"),
        (
            dict(
                segments=(
                    Segment(0, 6, 10, SegmentKind.LOOP_PRE_WRAP),
                    Segment(6, 4, 0, SegmentKind.LOOP_POST_WRAP),
                    Segment(10, 6, 4, SegmentKind.LOOP_POST_WRAP),
                ),
                data=_RAW[10:] + _RAW[:4] + _RAW[4:10],
            ),
            "unexpected segment count 3",
        ),
    ],
)
def test_linearized_buffer_validates_layout(changes: dict, fragment: str) -> None:
    with pytest.raises(RingError, match=fragment):
        dataclasses.replace(_two_segment(), **changes)


def test_linearized_buffer_rejects_negative_length() -> None:
    lin = linearize(make_buffer(_RAW, head=0, wrap_state=_NOT_WRAPPED))
    with pytest.raises(RingError, match="negative length"):
        dataclasses.replace(
            lin,
            segments=(Segment(0, -1, 0, SegmentKind.LOOP_NO_WRAP), Segment(-1, 1, 0, SegmentKind.LOOP_NO_WRAP)),
            data=b"",
        )
