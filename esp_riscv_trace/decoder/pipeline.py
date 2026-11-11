# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Decode one core's linearized trace buffer into trace events."""

from __future__ import annotations

from collections.abc import Callable, Iterator

from ..elf import CodeImage
from ..ir.events import TraceEvent
from ..snapshot import abi
from . import framing
from .framing import Frame, FramingEvent, FramingResult
from .packets import Packet, PacketParseError
from .packets_et20 import parse_et20
from .packets_pt10 import parse_pt10
from .params import DecoderProfile
from .reconstruct import InputEvent, reconstruct
from .ring import LinearizedBuffer

PacketParser = Callable[[Frame, DecoderProfile], Packet]


def decode_events(
    linear: LinearizedBuffer,
    profile: DecoderProfile,
    image: CodeImage,
    address_mode: abi.AddressMode,
) -> Iterator[TraceEvent]:
    """Frame, parse and reconstruct ``linear`` with the parser for ``profile.packet_format``."""
    parser = parse_et20 if profile.packet_format == abi.PacketFormat.EFFICIENT_TRACE_2_0 else parse_pt10
    stream = interleave(framing.frame(linear), parser, profile)
    return reconstruct(stream, image, profile, address_mode)


def interleave(result: FramingResult, parser: PacketParser, profile: DecoderProfile) -> Iterator[InputEvent]:
    """Yield framing events and parsed packets in stream order.

    A framed packet whose payload cannot be decoded becomes a framing event, so reconstruction state is reset.
    """
    frame_iter = iter(result.frames)
    fut = next(frame_iter, None)
    for evt in result.events:
        while fut is not None and (
            fut.segment_index < evt.segment_index
            or (fut.segment_index == evt.segment_index and fut.stream_offset < evt.stream_offset)
        ):
            yield _parse_or_error(fut, parser, profile)
            fut = next(frame_iter, None)
        yield evt
    while fut is not None:
        yield _parse_or_error(fut, parser, profile)
        fut = next(frame_iter, None)


def _parse_or_error(frame: Frame, parser: PacketParser, profile: DecoderProfile) -> InputEvent:
    try:
        return parser(frame, profile)
    except PacketParseError as exc:
        return FramingEvent(
            kind=framing.FramingEventKind.PACKET_ERROR,
            segment_index=frame.segment_index,
            stream_offset=frame.stream_offset,
            detail=f"cannot decode packet payload: {exc}",
            span=frame.length,
            packet_index=frame.packet_index,
            payload=frame.payload,
        )


__all__ = ["PacketParser", "decode_events", "interleave"]
