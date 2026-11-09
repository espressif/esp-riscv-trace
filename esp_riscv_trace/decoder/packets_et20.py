# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RISC-V Efficient Trace v2.0 packet field layout."""

from __future__ import annotations

from .framing import Frame
from .packets import (
    BRANCHES_BITS,
    FORMAT_BITS,
    BitReader,
    Format1WithAddrPacket,
    Format2Packet,
    FormatReaders,
    Packet,
    PacketParseError,
    SupportPacket,
    TrapPacket,
    branch_map_width,
    parse_packet,
)
from .params import DecoderProfile

# Address field widths a Format 1 or Format 2 payload can end with
_ADDRESS_WIDTHS = (8, 16, 24, 32)
_F2_ZERO_EXTEND_BITS = 4
# TRM 3.6.3.3 zero_ext width per branch_map width
_F1_ZERO_EXT_BITS = {1: 6, 3: 4, 7: 0, 15: 0, 31: 0}
_IOPTIONS_BITS = 6
_IOPTIONS_FULL_ADDRESS_BIT = 2


def parse_et20(frame: Frame, profile: DecoderProfile) -> Packet:
    """Parse one ET 2.0 payload, raising ``PacketParseError`` when it is malformed."""
    return parse_packet(frame, profile, _READERS)


def _parse_trap(frame: Frame, reader: BitReader, profile: DecoderProfile) -> TrapPacket:
    branch = reader.read(1)
    privilege = reader.read(profile.privilege_width)
    ecause = reader.read(profile.ecause_width)
    interrupt = reader.read(1)
    thaddr = reader.read(1)
    raw_address = reader.read(profile.address_bits)
    tval: int | None = reader.read(profile.iaddress_width) if interrupt == 0 else None
    return TrapPacket(
        frame=frame,
        branch=branch,
        privilege=privilege,
        ecause=ecause,
        interrupt=interrupt,
        thaddr=thaddr,
        raw_address=raw_address,
        address_pc=profile.pc_from_raw(raw_address),
        tval=tval,
    )


def _parse_support(frame: Frame, reader: BitReader, profile: DecoderProfile) -> SupportPacket:
    enable = reader.read(1)
    encoder_mode = reader.read(1)
    qual_status = reader.read(2)
    ioptions = reader.read(_IOPTIONS_BITS)
    return SupportPacket(
        frame=frame,
        enable=enable,
        qual_status=qual_status,
        encoder_mode=encoder_mode,
        ioptions=ioptions,
        full_address_mode=bool((ioptions >> _IOPTIONS_FULL_ADDRESS_BIT) & 1),
    )


def _parse_f2(frame: Frame, reader: BitReader, profile: DecoderProfile) -> Format2Packet:
    notify = reader.read(1)
    updiscon = reader.read(1)
    reader.skip(_F2_ZERO_EXTEND_BITS)
    address_bits = reader.remaining()
    if address_bits not in _ADDRESS_WIDTHS:
        raise PacketParseError(f"Format 2 address field is {address_bits} bits, expected 8/16/24/32")
    return Format2Packet(
        frame=frame,
        notify=notify,
        updiscon=updiscon,
        raw_address=reader.read(address_bits),
        address_width_bits=address_bits,
    )


def _parse_f1_with_addr(
    frame: Frame, reader: BitReader, profile: DecoderProfile, branches: int
) -> Format1WithAddrPacket:
    width = branch_map_width(branches)
    branch_map = reader.read(width)
    notify = reader.read(1)
    updiscon = reader.read(1)
    fixed_bits = FORMAT_BITS + BRANCHES_BITS + width + 1 + 1
    if reader.total_bits - fixed_bits <= 0:
        raise PacketParseError(
            f"Format 1 with address has no room for zero_ext and address "
            f"(payload {reader.total_bits} bits, fixed {fixed_bits})"
        )
    zero_ext = _F1_ZERO_EXT_BITS[width]
    reader.skip(zero_ext)
    address_bits = reader.remaining()
    if address_bits not in _ADDRESS_WIDTHS:
        raise PacketParseError(
            f"Format 1 address field is {address_bits} bits (branches {branches}, "
            f"branch_map_width {width}, zero_ext {zero_ext}), expected 8/16/24/32"
        )
    return Format1WithAddrPacket(
        frame=frame,
        branches=branches,
        branch_map=branch_map,
        branch_map_width_bits=width,
        notify=notify,
        updiscon=updiscon,
        raw_address=reader.read(address_bits),
        address_width_bits=address_bits,
    )


_READERS = FormatReaders(trap=_parse_trap, support=_parse_support, f2=_parse_f2, f1_with_addr=_parse_f1_with_addr)

__all__ = ["parse_et20"]
