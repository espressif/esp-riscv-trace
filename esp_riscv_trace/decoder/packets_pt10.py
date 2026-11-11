# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RISC-V Processor Trace v1.0 packet field layout."""

from __future__ import annotations

from .framing import Frame
from .packets import (
    BitReader,
    Format1WithAddrPacket,
    Format2Packet,
    FormatReaders,
    Packet,
    SupportPacket,
    TrapPacket,
    branch_map_width,
    parse_packet,
)
from .params import DecoderProfile


def parse_pt10(frame: Frame, profile: DecoderProfile) -> Packet:
    """Parse one PT 1.0 payload, raising ``PacketParseError`` when it is malformed."""
    return parse_packet(frame, profile, _READERS)


def _parse_trap(frame: Frame, reader: BitReader, profile: DecoderProfile) -> TrapPacket:
    branch = reader.read(1)
    privilege = reader.read(profile.privilege_width)
    ecause = reader.read(profile.ecause_width)
    interrupt = reader.read(1)
    raw_address = reader.read(profile.address_bits)
    tval = reader.read(profile.iaddress_width)
    return TrapPacket(
        frame=frame,
        branch=branch,
        privilege=privilege,
        ecause=ecause,
        interrupt=interrupt,
        thaddr=None,
        raw_address=raw_address,
        address_pc=profile.pc_from_raw(raw_address),
        tval=tval,
    )


def _parse_support(frame: Frame, reader: BitReader, profile: DecoderProfile) -> SupportPacket:
    enable = reader.read(1)
    qual_status = reader.read(2)
    return SupportPacket(
        frame=frame,
        enable=enable,
        qual_status=qual_status,
        encoder_mode=None,
        ioptions=None,
        full_address_mode=None,
    )


def _parse_f2(frame: Frame, reader: BitReader, profile: DecoderProfile) -> Format2Packet:
    raw_address = reader.read(profile.address_bits)
    notify = reader.read(1)
    updiscon = reader.read(1)
    return Format2Packet(
        frame=frame,
        notify=notify,
        updiscon=updiscon,
        raw_address=raw_address,
        address_width_bits=profile.address_bits,
    )


def _parse_f1_with_addr(
    frame: Frame, reader: BitReader, profile: DecoderProfile, branches: int
) -> Format1WithAddrPacket:
    width = branch_map_width(branches)
    branch_map = reader.read(width)
    raw_address = reader.read(profile.address_bits)
    notify = reader.read(1)
    updiscon = reader.read(1)
    return Format1WithAddrPacket(
        frame=frame,
        branches=branches,
        branch_map=branch_map,
        branch_map_width_bits=width,
        notify=notify,
        updiscon=updiscon,
        raw_address=raw_address,
        address_width_bits=profile.address_bits,
    )


_READERS = FormatReaders(trap=_parse_trap, support=_parse_support, f2=_parse_f2, f1_with_addr=_parse_f1_with_addr)

__all__ = ["parse_pt10"]
