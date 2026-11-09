# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Gap vs boundary vs warning semantics of framing circular buffers."""

from __future__ import annotations

from helpers import BASE, JALR_X0_RA, NOP, FakeImage, make_buffer, make_frame, make_note, make_sync, pack_bits

from esp_riscv_trace.analysis.diagnostics import DiagnosticAccumulator
from esp_riscv_trace.containers.base import TraceBuffer
from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.framing import (
    FramingEvent,
    FramingEventKind,
    frame,
)
from esp_riscv_trace.decoder.packets import Format2Packet
from esp_riscv_trace.decoder.packets_et20 import parse_et20
from esp_riscv_trace.decoder.params import default_profile
from esp_riscv_trace.decoder.pipeline import interleave
from esp_riscv_trace.decoder.ring import linearize
from esp_riscv_trace.exporters import text as text_exporter
from esp_riscv_trace.ir.events import (
    InsnClass,
    TraceBoundary,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceWarning,
)
from esp_riscv_trace.snapshot import abi


def _pkt(index: int, payload: bytes = b"\xaa\xbb") -> bytes:
    plen = 3 + len(payload)
    assert 4 <= plen <= 13
    return bytes([plen, index & 0xFF, (index >> 8) & 0xFF]) + payload


def _sync_payload(pc: int, profile) -> bytes:
    return pack_bits(
        (3, 2),  # Format 3
        (0, 2),  # Sync
        (0, 1),  # branch
        (0, profile.privilege_width),
        (pc >> profile.iaddress_lsb, profile.address_bits),
    )


def _format2_payload(pc: int, profile) -> bytes:
    return pack_bits(
        (2, 2),  # Format 2
        (0, 1),  # notify matches address MSB
        (0, 1),  # updiscon matches notify
        (0, 4),  # zero_extend
        (pc >> profile.iaddress_lsb, 32),
    )


def _support_payload(*, enable: int = 1) -> bytes:
    return pack_bits(
        (3, 2),  # Format 3
        (3, 2),  # Support
        (enable, 1),
        (0, 1),  # encoder_mode
        (0, 2),  # qual_status
        (0b000100, 6),  # full address mode
    )


def _wrapped_buffer(linear: bytes, seam: int, *, fifo_overflow: bool = False) -> TraceBuffer:
    assert 0 < seam < len(linear)
    raw = linear[seam:] + linear[:seam]
    return make_buffer(raw, head=len(linear) - seam, fifo_overflow=fifo_overflow)


def _decode_buffer(buf: TraceBuffer, image: FakeImage):
    profile = default_profile("esp32c5")
    result = frame(linearize(buf))
    stream = interleave(result, parse_et20, profile)
    events = list(reconstruct.reconstruct(stream, image, profile, abi.AddressMode.FULL))
    return result, events


def _format2(pc: int, index: int) -> Format2Packet:
    return Format2Packet(
        frame=make_frame(index),
        notify=0,
        updiscon=0,
        raw_address=pc >> 1,
        address_width_bits=32,
    )


def _kinds(events) -> list[str]:
    return [e.kind for e in events]


def test_linear_trace_has_no_spurious_gap_or_boundary() -> None:
    profile = default_profile("esp32c5")
    used = bytes(14) + _pkt(1, _sync_payload(BASE, profile)) + _pkt(2, _support_payload())
    raw = used + bytes(8)
    buf = make_buffer(
        raw,
        head=len(used),
        memory_mode=int(abi.MemoryMode.LINEAR),
        wrap_state=int(abi.WrapState.NOT_WRAPPED),
    )
    result, events = _decode_buffer(buf, FakeImage(BASE, JALR_X0_RA))

    assert len(result.frames) == 2
    assert FramingEventKind.WRAP_BOUNDARY not in _kinds(result.events)
    assert FramingEventKind.PARTIAL_FRAME not in _kinds(result.events)
    assert FramingEventKind.TRACE_LOST not in _kinds(result.events)
    assert FramingEventKind.NO_ANCHOR not in _kinds(result.events)
    assert any(isinstance(e, TraceSync) for e in events)
    assert not any(isinstance(e, TraceGap) for e in events)


def _continuous_wrapped_buffer(*, target_index: int = 2) -> TraceBuffer:
    profile = default_profile("esp32c5")
    sync = _pkt(1, _sync_payload(BASE, profile))
    target = _pkt(target_index, _format2_payload(BASE + 8, profile))
    linear = bytes(14) + sync + target
    return _wrapped_buffer(linear, seam=14 + len(sync))


def test_circular_wrap_with_verified_packet_continuity_is_not_gap() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    result, events = _decode_buffer(_continuous_wrapped_buffer(), image)

    assert [f.packet_index for f in result.frames] == [1, 2]
    assert FramingEventKind.PARTIAL_FRAME not in _kinds(result.events)
    assert FramingEventKind.WRAP_BOUNDARY in _kinds(result.events)
    assert any(isinstance(e, TraceBoundary) and e.kind == "wrap" for e in events)
    assert any(isinstance(e, TraceWarning) and e.kind == "no_anchor" for e in events)
    assert not any(isinstance(e, TraceGap) for e in events)
    assert [e.pc for e in events if isinstance(e, TraceInstruction)] == [
        BASE,
        BASE + 4,
        BASE + 8,
    ]


def test_packet_fragments_on_opposite_sides_of_wrap_are_not_stitched() -> None:
    pkt0 = _pkt(1, b"\xaa\xbb\xcc")
    pkt1 = _pkt(2, b"\x11\x12\x13\x14\x15")
    pkt2 = _pkt(3, b"\x21\x22\x23")
    linear = bytes(14) + pkt0 + pkt1 + pkt2
    linear += bytes(40 - len(linear))
    result = frame(linearize(_wrapped_buffer(linear, seam=24)))

    assert [f.packet_index for f in result.frames] == [1]
    assert FramingEventKind.PARTIAL_FRAME in _kinds(result.events)
    assert FramingEventKind.WRAP_BOUNDARY in _kinds(result.events)
    profile = default_profile("esp32c5")
    events = list(
        reconstruct.reconstruct(
            result.events,
            FakeImage(BASE, JALR_X0_RA),
            profile,
            abi.AddressMode.FULL,
        )
    )
    assert [e.reason for e in events if isinstance(e, TraceGap)] == [reconstruct.GapReason.PARTIAL_FRAME]


def test_circular_wrap_without_anchor_is_warning_not_gap() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    _result, events = _decode_buffer(_continuous_wrapped_buffer(), image)

    assert any(isinstance(e, TraceBoundary) and e.kind == "wrap" for e in events)
    assert any(isinstance(e, TraceWarning) and e.kind == "no_anchor" for e in events)
    assert not any(isinstance(e, TraceGap) for e in events)

    wrap_line = text_exporter.format_event(next(e for e in events if isinstance(e, TraceBoundary)))
    warn_line = text_exporter.format_event(next(e for e in events if isinstance(e, TraceWarning)))
    assert wrap_line.startswith('--- boundary wrap "circular trace memory wraps here"')
    assert warn_line.startswith('--- warning no_anchor "no anchor found in this physical region"')


def test_packet_index_discontinuity_at_wrap_is_real_gap() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    _result, events = _decode_buffer(_continuous_wrapped_buffer(target_index=9), image)

    assert any(isinstance(e, TraceBoundary) for e in events)
    assert any(isinstance(e, TraceWarning) and e.kind == "no_anchor" for e in events)
    assert [e.reason for e in events if isinstance(e, TraceGap)] == [reconstruct.GapReason.INDEX_JUMP]


def test_circular_wrap_keeps_walker_state() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    profile = default_profile("esp32c5")
    wrap = FramingEvent(
        kind=FramingEventKind.WRAP_BOUNDARY,
        segment_index=1,
        stream_offset=8,
        detail="circular trace memory wraps here",
    )
    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=0), wrap, _format2(BASE + 8, 2)],
            image,
            profile,
            abi.AddressMode.FULL,
        )
    )
    assert not any(isinstance(e, TraceGap) for e in events)
    assert any(isinstance(e, TraceBoundary) for e in events)
    jump = next(e for e in events if isinstance(e, TraceInstruction) and e.cls == InsnClass.UNINFERABLE)
    assert jump.next_pc == BASE + 8


def test_valid_packets_before_later_anchor_are_not_discarded() -> None:
    profile = default_profile("esp32c5")
    first = _pkt(1, _sync_payload(BASE, profile))
    second = _pkt(2, _format2_payload(BASE + 8, profile))
    third = _pkt(3, _sync_payload(BASE + 8, profile))
    linear = first + second + bytes(14) + third
    buf = _wrapped_buffer(linear, seam=len(first) + len(second))
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)

    result, events = _decode_buffer(buf, image)

    assert [f.packet_index for f in result.frames] == [1, 2, 3]
    assert any(isinstance(e, TraceWarning) and e.kind == "no_anchor" for e in events)
    assert not any(isinstance(e, TraceGap) for e in events)


def test_unaligned_prefix_before_anchor_is_real_gap() -> None:
    profile = default_profile("esp32c5")
    valid = _pkt(7, _sync_payload(BASE, profile))
    used = b"\xff\x01\x02" + bytes(14) + valid
    buf = make_buffer(
        used,
        head=len(used),
        memory_mode=int(abi.MemoryMode.LINEAR),
        wrap_state=int(abi.WrapState.NOT_WRAPPED),
    )

    result, events = _decode_buffer(buf, FakeImage(BASE, JALR_X0_RA))

    assert FramingEventKind.ALIGNMENT_LOST in _kinds(result.events)
    assert any(isinstance(e, TraceGap) and e.reason == reconstruct.GapReason.ALIGNMENT_LOST for e in events)


def test_malformed_packet_payload_is_real_gap_and_resets_walker() -> None:
    profile = default_profile("esp32c5")
    sync1 = _pkt(1, _sync_payload(BASE, profile))
    malformed = _pkt(2, b"\x02")  # Format 2 without an address field
    sync2 = _pkt(3, _sync_payload(BASE + 8, profile))
    used = bytes(14) + sync1 + malformed + sync2
    buf = make_buffer(
        used,
        head=len(used),
        memory_mode=int(abi.MemoryMode.LINEAR),
        wrap_state=int(abi.WrapState.NOT_WRAPPED),
    )

    result, events = _decode_buffer(buf, FakeImage(BASE, JALR_X0_RA + NOP + JALR_X0_RA))

    gaps = [e for e in events if isinstance(e, TraceGap)]
    assert [g.reason for g in gaps] == [reconstruct.GapReason.PACKET_ERROR]
    assert gaps[0].provenance is not None
    assert gaps[0].provenance.packet_index == 2
    assert gaps[0].provenance.payload == b"\x02"
    assert FramingEventKind.PACKET_ERROR not in _kinds(result.events)  # Generated while parsing, not framing.
    assert [e.pc for e in events if isinstance(e, TraceSync)] == [BASE, BASE + 8]


def test_execution_gap_clears_differential_address_anchor() -> None:
    profile = default_profile("esp32c5")
    index_gap = FramingEvent(
        kind=FramingEventKind.INDEX_JUMP,
        segment_index=0,
        stream_offset=1,
        detail="packet missing",
    )
    delta_address = Format2Packet(
        frame=make_frame(2),
        notify=0,
        updiscon=0,
        raw_address=1,
        address_width_bits=8,
    )

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=0), index_gap, delta_address],
            FakeImage(BASE, JALR_X0_RA),
            profile,
            abi.AddressMode.DELTA,
        )
    )

    assert [e.reason for e in events if isinstance(e, TraceGap)] == [
        reconstruct.GapReason.INDEX_JUMP,
        reconstruct.GapReason.NO_PC,
    ]


def test_incomplete_frame_at_stream_end_is_gap() -> None:
    profile = default_profile("esp32c5")
    used = bytes(14) + _pkt(1, _sync_payload(BASE, profile)) + bytes([8, 2, 0, 0x11])
    raw = used + bytes(4)
    buf = make_buffer(
        raw,
        head=len(used),
        memory_mode=int(abi.MemoryMode.LINEAR),
        wrap_state=int(abi.WrapState.NOT_WRAPPED),
    )
    result, events = _decode_buffer(buf, FakeImage(BASE, JALR_X0_RA))
    assert FramingEventKind.PARTIAL_FRAME in _kinds(result.events)
    assert len(result.frames) == 1

    gaps = [e for e in events if isinstance(e, TraceGap)]
    assert len(gaps) == 1
    assert gaps[0].reason == reconstruct.GapReason.PARTIAL_FRAME
    line = text_exporter.format_event(gaps[0])
    assert line.startswith("--- gap partial_frame")


def test_fifo_overflow_is_trace_lost_gap() -> None:
    profile = default_profile("esp32c5")
    used = bytes(14) + _pkt(1, _sync_payload(BASE, profile))
    raw = used + bytes(8)
    buf = make_buffer(
        raw,
        head=len(used),
        memory_mode=int(abi.MemoryMode.LINEAR),
        wrap_state=int(abi.WrapState.NOT_WRAPPED),
        fifo_overflow=True,
    )
    result, events = _decode_buffer(buf, FakeImage(BASE, JALR_X0_RA))
    assert FramingEventKind.TRACE_LOST in _kinds(result.events)

    gaps = [e for e in events if isinstance(e, TraceGap)]
    assert [g.reason for g in gaps] == [reconstruct.GapReason.TRACE_LOST]
    assert text_exporter.format_event(gaps[0]).startswith("--- gap trace_lost")


def test_trace_disabled_is_execution_gap() -> None:
    profile = default_profile("esp32c5")
    used = bytes(14) + _pkt(1, _sync_payload(BASE, profile)) + _pkt(2, _support_payload(enable=0))
    buf = make_buffer(
        used,
        head=len(used),
        memory_mode=int(abi.MemoryMode.LINEAR),
        wrap_state=int(abi.WrapState.NOT_WRAPPED),
    )
    _result, events = _decode_buffer(buf, FakeImage(BASE, JALR_X0_RA))

    gaps = [e for e in events if isinstance(e, TraceGap)]
    assert [g.reason for g in gaps] == [reconstruct.GapReason.TRACE_DISABLED]
    assert any(isinstance(e, TraceSync) for e in events)


def test_summary_counts_only_execution_gaps() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    _result, events = _decode_buffer(_continuous_wrapped_buffer(), image)
    acc = DiagnosticAccumulator()
    for event in events:
        acc.push(event)
    diag = acc.close(make_note())
    assert diag.gaps == 0
    assert diag.warnings == 1
    assert diag.boundaries == 1
    assert diag.warning_reasons == (("no_anchor", 1),)
    assert diag.boundary_kinds == (("wrap", 1),)

    lines = text_exporter.format_diagnostic(diag)
    assert "gaps=0" in lines[0]
    assert any("warnings=no_anchor=1" in line for line in lines)
    assert any("boundaries=wrap=1" in line for line in lines)
