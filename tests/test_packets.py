# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Bit layout of ET 2.0 and PT 1.0 packet payloads."""

from __future__ import annotations

from collections.abc import Callable

import pytest
from helpers import BASE, make_frame, pack_bits

from esp_riscv_trace.decoder.framing import Frame
from esp_riscv_trace.decoder.packets import (
    F3_CONTEXT,
    F3_SUPPORT,
    F3_SYNC,
    F3_TRAP,
    FORMAT_F0,
    FORMAT_F1,
    FORMAT_F2,
    FORMAT_F3,
    BitReader,
    ContextPacket,
    Format0Packet,
    Format1NoAddrPacket,
    Format1WithAddrPacket,
    Format2Packet,
    Packet,
    PacketParseError,
    SupportPacket,
    SyncPacket,
    TrapPacket,
    branch_map_width,
)
from esp_riscv_trace.decoder.packets_et20 import parse_et20
from esp_riscv_trace.decoder.packets_pt10 import parse_pt10
from esp_riscv_trace.decoder.params import DecoderProfile, default_profile

Parser = Callable[[Frame, DecoderProfile], Packet]

ET = default_profile("esp32c5")
PT = default_profile("esp32c6")

_FORMATS = pytest.mark.parametrize(
    ("parse", "profile"),
    [(parse_et20, ET), (parse_pt10, PT)],
    ids=["et20", "pt10"],
)

_RAW_ADDR = (BASE + 0x1234) >> 1
_MAP_31 = 0x5A5A5A5A & ((1 << 31) - 1)


def _parse(parse: Parser, profile: DecoderProfile, *fields: tuple[int, int]) -> Packet:
    return parse(make_frame(3, pack_bits(*fields)), profile)


@_FORMATS
def test_sync(parse: Parser, profile: DecoderProfile) -> None:
    pkt = _parse(
        parse,
        profile,
        (FORMAT_F3, 2),
        (F3_SYNC, 2),
        (1, 1),  # branch
        (1, profile.privilege_width),
        (_RAW_ADDR, profile.address_bits),
    )
    assert isinstance(pkt, SyncPacket)
    assert (pkt.branch, pkt.privilege, pkt.raw_address, pkt.pc) == (1, 1, _RAW_ADDR, BASE + 0x1234)
    assert pkt.frame.packet_index == 3


def test_sync_truncated_raises() -> None:
    with pytest.raises(PacketParseError, match="past end"):
        _parse(parse_et20, ET, (FORMAT_F3, 2), (F3_SYNC, 2), (0, 4))


@pytest.mark.parametrize("thaddr", [0, 1])
def test_et20_exception_trap(thaddr: int) -> None:
    pkt = _parse(
        parse_et20,
        ET,
        (FORMAT_F3, 2),
        (F3_TRAP, 2),
        (0, 1),  # branch
        (1, ET.privilege_width),
        (0x2A, ET.ecause_width),
        (0, 1),  # interrupt
        (thaddr, 1),
        (_RAW_ADDR, ET.address_bits),
        (0xDEADBEEF, ET.iaddress_width),
    )
    assert isinstance(pkt, TrapPacket)
    assert (pkt.ecause, pkt.interrupt, pkt.thaddr, pkt.privilege) == (0x2A, 0, thaddr, 1)
    assert pkt.address_pc == BASE + 0x1234
    assert pkt.tval == 0xDEADBEEF


def test_et20_interrupt_trap_has_no_tval() -> None:
    pkt = _parse(
        parse_et20,
        ET,
        (FORMAT_F3, 2),
        (F3_TRAP, 2),
        (1, 1),
        (0, ET.privilege_width),
        (7, ET.ecause_width),
        (1, 1),  # interrupt
        (1, 1),  # thaddr
        (_RAW_ADDR, ET.address_bits),
    )
    assert isinstance(pkt, TrapPacket)
    assert (pkt.branch, pkt.ecause, pkt.interrupt, pkt.thaddr) == (1, 7, 1, 1)
    assert pkt.tval is None


@pytest.mark.parametrize("interrupt", [0, 1])
def test_pt10_trap_always_has_tval(interrupt: int) -> None:
    pkt = _parse(
        parse_pt10,
        PT,
        (FORMAT_F3, 2),
        (F3_TRAP, 2),
        (0, 1),
        (1, PT.privilege_width),
        (0x1F, PT.ecause_width),
        (interrupt, 1),
        (_RAW_ADDR, PT.address_bits),
        (0x40801000, PT.iaddress_width),
    )
    assert isinstance(pkt, TrapPacket)
    assert (pkt.ecause, pkt.interrupt, pkt.thaddr) == (0x1F, interrupt, None)
    assert pkt.address_pc == BASE + 0x1234
    assert pkt.tval == 0x40801000


@_FORMATS
def test_context(parse: Parser, profile: DecoderProfile) -> None:
    assert isinstance(_parse(parse, profile, (FORMAT_F3, 2), (F3_CONTEXT, 2)), ContextPacket)


@pytest.mark.parametrize(
    ("ioptions", "full_address_mode"),
    [(0b000100, True), (0b111011, False), (0, False)],
)
def test_et20_support(ioptions: int, full_address_mode: bool) -> None:
    pkt = _parse(
        parse_et20,
        ET,
        (FORMAT_F3, 2),
        (F3_SUPPORT, 2),
        (1, 1),  # enable
        (1, 1),  # encoder_mode
        (2, 2),  # qual_status
        (ioptions, 6),
    )
    assert isinstance(pkt, SupportPacket)
    assert (pkt.enable, pkt.encoder_mode, pkt.qual_status, pkt.ioptions) == (1, 1, 2, ioptions)
    assert pkt.full_address_mode is full_address_mode


def test_pt10_support_has_enable_and_qual_status_only() -> None:
    pkt = _parse(parse_pt10, PT, (FORMAT_F3, 2), (F3_SUPPORT, 2), (0, 1), (3, 2))
    assert isinstance(pkt, SupportPacket)
    assert (pkt.enable, pkt.qual_status) == (0, 3)
    assert (pkt.encoder_mode, pkt.ioptions, pkt.full_address_mode) == (None, None, None)


@pytest.mark.parametrize("width", [8, 16, 24, 32])
def test_et20_format2_address_width_follows_payload(width: int) -> None:
    raw = (0xA5A5A5A5 >> (32 - width)) | 1
    pkt = _parse(parse_et20, ET, (FORMAT_F2, 2), (1, 1), (0, 1), (0, 4), (raw, width))
    assert isinstance(pkt, Format2Packet)
    assert (pkt.notify, pkt.updiscon, pkt.raw_address, pkt.address_width_bits) == (1, 0, raw, width)


@pytest.mark.parametrize("payload_bytes", [1, 6])
def test_et20_format2_rejects_address_width(payload_bytes: int) -> None:
    payload = bytes([FORMAT_F2]) + bytes(payload_bytes - 1)
    with pytest.raises(PacketParseError, match=f"address field is {8 * (payload_bytes - 1)} bits"):
        parse_et20(make_frame(0, payload), ET)


def test_pt10_format2_has_fixed_width() -> None:
    pkt = _parse(parse_pt10, PT, (FORMAT_F2, 2), (_RAW_ADDR, PT.address_bits), (0, 1), (1, 1))
    assert isinstance(pkt, Format2Packet)
    assert (pkt.notify, pkt.updiscon, pkt.raw_address) == (0, 1, _RAW_ADDR)
    assert pkt.address_width_bits == PT.address_bits


@_FORMATS
def test_format1_without_address(parse: Parser, profile: DecoderProfile) -> None:
    pkt = _parse(parse, profile, (FORMAT_F1, 2), (0, 5), (_MAP_31, 31))
    assert isinstance(pkt, Format1NoAddrPacket)
    assert (pkt.branches, pkt.branch_map) == (0, _MAP_31)


# Branch count, branch map width and zero_ext width before an ET 2.0 address
_ET_F1_LAYOUTS = [(1, 1, 6), (2, 3, 4), (3, 3, 4), (5, 7, 0), (9, 15, 0), (31, 31, 0)]


@pytest.mark.parametrize(("branches", "width", "zero_ext"), _ET_F1_LAYOUTS)
@pytest.mark.parametrize("address_bits", [8, 32])
def test_et20_format1_with_address(branches: int, width: int, zero_ext: int, address_bits: int) -> None:
    branch_map = (1 << width) - 2
    raw = (1 << (address_bits - 1)) | 3
    pkt = _parse(
        parse_et20,
        ET,
        (FORMAT_F1, 2),
        (branches, 5),
        (branch_map, width),
        (0, 1),  # notify
        (1, 1),  # updiscon
        (0, zero_ext),
        (raw, address_bits),
    )
    assert isinstance(pkt, Format1WithAddrPacket)
    assert (pkt.branches, pkt.branch_map, pkt.branch_map_width_bits) == (branches, branch_map, width)
    assert (pkt.notify, pkt.updiscon, pkt.raw_address, pkt.address_width_bits) == (0, 1, raw, address_bits)


def test_et20_format1_rejects_missing_address() -> None:
    # Seven branches fill exactly two bytes before the address
    payload = pack_bits((FORMAT_F1, 2), (7, 5), (0, 7), (0, 1), (0, 1))
    with pytest.raises(PacketParseError, match="has no room for zero_ext and address"):
        parse_et20(make_frame(0, payload), ET)


def test_et20_format1_rejects_address_width() -> None:
    payload = pack_bits((FORMAT_F1, 2), (1, 5), (0, 1), (0, 1), (0, 1), (0, 6))
    with pytest.raises(PacketParseError, match="Format 1 address field is 0 bits"):
        parse_et20(make_frame(0, payload), ET)


@pytest.mark.parametrize(("branches", "width"), [(1, 1), (3, 3), (6, 7), (12, 15), (17, 31)])
def test_pt10_format1_with_address(branches: int, width: int) -> None:
    branch_map = 1 << (width - 1)
    pkt = _parse(
        parse_pt10,
        PT,
        (FORMAT_F1, 2),
        (branches, 5),
        (branch_map, width),
        (_RAW_ADDR, PT.address_bits),
        (1, 1),  # notify
        (0, 1),  # updiscon
    )
    assert isinstance(pkt, Format1WithAddrPacket)
    assert (pkt.branches, pkt.branch_map, pkt.branch_map_width_bits) == (branches, branch_map, width)
    assert (pkt.notify, pkt.updiscon, pkt.raw_address, pkt.address_width_bits) == (1, 0, _RAW_ADDR, PT.address_bits)


@_FORMATS
def test_format0(parse: Parser, profile: DecoderProfile) -> None:
    assert isinstance(_parse(parse, profile, (FORMAT_F0, 2), (0x3F, 6)), Format0Packet)


@pytest.mark.parametrize(
    ("branches", "width"),
    [(1, 1), (2, 3), (3, 3), (4, 7), (7, 7), (8, 15), (15, 15), (16, 31), (31, 31)],
)
def test_branch_map_width(branches: int, width: int) -> None:
    assert branch_map_width(branches) == width


@pytest.mark.parametrize("branches", [0, 32, -1])
def test_branch_map_width_rejects(branches: int) -> None:
    with pytest.raises(PacketParseError, match=f"branches {branches} out of range 1..31"):
        branch_map_width(branches)


def test_bit_reader_reads_lsb_first() -> None:
    reader = BitReader(bytes([0b10110100, 0xFF]))
    assert reader.total_bits == 16
    assert reader.read(2) == 0b00
    assert reader.read(3) == 0b101
    reader.skip(3)
    assert reader.remaining() == 8
    assert reader.read(8) == 0xFF
    assert reader.read(0) == 0
    assert reader.remaining() == 0


@pytest.mark.parametrize(
    ("op", "n", "fragment"),
    [
        ("read", 9, "read 9 bits past end"),
        ("read", -1, "read -1 bits past end"),
        ("skip", 9, "skip 9 bits past end"),
        ("skip", -1, "skip -1 bits past end"),
    ],
)
def test_bit_reader_rejects_past_end(op: str, n: int, fragment: str) -> None:
    reader = BitReader(b"\x01")
    with pytest.raises(PacketParseError, match=fragment):
        getattr(reader, op)(n)


def test_empty_payload_raises() -> None:
    with pytest.raises(PacketParseError, match="past end"):
        parse_et20(make_frame(0, b""), ET)
