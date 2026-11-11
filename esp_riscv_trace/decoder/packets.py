# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Packet types and the dispatch shared by the ET 2.0 and PT 1.0 parsers."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from .framing import Frame
from .params import DecoderProfile

# TRM 3.6.3 payload format prefix (bits 0..1 of payload byte 0).
FORMAT_F0 = 0
FORMAT_F1 = 1
FORMAT_F2 = 2
FORMAT_F3 = 3

# TRM 3.6.3.1 Format 3 subformat encoding (bits 2..3 of payload byte 0).
F3_SYNC = 0
F3_TRAP = 1
F3_CONTEXT = 2
F3_SUPPORT = 3

# TRM support packet qual_status value for an ended trace whose preceding packet names the last traced instruction
QUAL_STATUS_ENDED_REP = 1
# TRM support packet qual_status value for an ended trace with a pending updiscon
QUAL_STATUS_ENDED_NTR = 3

FORMAT_BITS = 2
SUBFORMAT_BITS = 2
BRANCHES_BITS = 5
BRANCHES_MAX = (1 << BRANCHES_BITS) - 1
NO_ADDR_BRANCH_MAP_BITS = 31


class PacketParseError(ValueError):
    """Raised when a packet payload cannot be parsed against the profile."""


class BitReader:
    """LSB-first bit reader over a byte payload."""

    __slots__ = ("_buf", "_nbits", "_pos")

    def __init__(self, data: bytes) -> None:
        self._buf = int.from_bytes(data, byteorder="little")
        self._nbits = len(data) * 8
        self._pos = 0

    @property
    def total_bits(self) -> int:
        return self._nbits

    def remaining(self) -> int:
        return self._nbits - self._pos

    def read(self, n: int) -> int:
        if n < 0 or self._pos + n > self._nbits:
            raise PacketParseError(f"read {n} bits past end (pos {self._pos}, total {self._nbits})")
        value = (self._buf >> self._pos) & ((1 << n) - 1)
        self._pos += n
        return value

    def skip(self, n: int) -> None:
        if n < 0 or self._pos + n > self._nbits:
            raise PacketParseError(f"skip {n} bits past end (pos {self._pos}, total {self._nbits})")
        self._pos += n


@dataclass(frozen=True)
class _PacketBase:
    frame: Frame


@dataclass(frozen=True)
class SyncPacket(_PacketBase):
    """Format 3 subformat 0. Start of trace or resync anchor."""

    branch: int
    privilege: int
    raw_address: int
    pc: int


@dataclass(frozen=True)
class TrapPacket(_PacketBase):
    """Format 3 subformat 1. Exception or interrupt entry."""

    branch: int
    privilege: int
    ecause: int
    interrupt: int
    thaddr: int | None
    raw_address: int
    address_pc: int
    tval: int | None


@dataclass(frozen=True)
class ContextPacket(_PacketBase):
    """Format 3 subformat 2. Context change packet."""


@dataclass(frozen=True)
class SupportPacket(_PacketBase):
    """Format 3 subformat 3. Trace support and status packet."""

    enable: int
    qual_status: int
    encoder_mode: int | None
    ioptions: int | None
    full_address_mode: bool | None


@dataclass(frozen=True)
class Format2Packet(_PacketBase):
    """Format 2. Address-only packet."""

    notify: int
    updiscon: int
    raw_address: int
    address_width_bits: int


@dataclass(frozen=True)
class Format1WithAddrPacket(_PacketBase):
    """Format 1 with address. Branch map with target address."""

    branches: int
    branch_map: int
    branch_map_width_bits: int
    notify: int
    updiscon: int
    raw_address: int
    address_width_bits: int


@dataclass(frozen=True)
class Format1NoAddrPacket(_PacketBase):
    """Format 1 without address. Branch map only."""

    branches: int
    branch_map: int


@dataclass(frozen=True)
class Format0Packet(_PacketBase):
    """Format 0 packet."""


Packet = (
    SyncPacket
    | TrapPacket
    | ContextPacket
    | SupportPacket
    | Format2Packet
    | Format1WithAddrPacket
    | Format1NoAddrPacket
    | Format0Packet
)

FieldReader = Callable[[Frame, BitReader, DecoderProfile], Packet]


@dataclass(frozen=True)
class FormatReaders:
    """The packet field readers that differ between ET 2.0 and PT 1.0."""

    trap: FieldReader
    support: FieldReader
    f2: FieldReader
    f1_with_addr: Callable[[Frame, BitReader, DecoderProfile, int], Packet]


def branch_map_width(branches: int) -> int:
    """Return the branch_map width in bits for a Format 1 packet with address (TRM 3.6.3.3)."""
    if not 1 <= branches <= BRANCHES_MAX:
        raise PacketParseError(f"branches {branches} out of range 1..{BRANCHES_MAX} for Format 1 with address")
    return (1 << branches.bit_length()) - 1


def parse_packet(frame: Frame, profile: DecoderProfile, readers: FormatReaders) -> Packet:
    """Parse one payload, using ``readers`` for the format-specific fields."""
    reader = BitReader(frame.payload)
    fmt = reader.read(FORMAT_BITS)
    if fmt == FORMAT_F3:
        subformat = reader.read(SUBFORMAT_BITS)
        if subformat == F3_SYNC:
            return _parse_sync(frame, reader, profile)
        if subformat == F3_TRAP:
            return readers.trap(frame, reader, profile)
        if subformat == F3_CONTEXT:
            return ContextPacket(frame=frame)
        return readers.support(frame, reader, profile)
    if fmt == FORMAT_F2:
        return readers.f2(frame, reader, profile)
    if fmt == FORMAT_F1:
        branches = reader.read(BRANCHES_BITS)
        if branches == 0:
            return Format1NoAddrPacket(frame=frame, branches=0, branch_map=reader.read(NO_ADDR_BRANCH_MAP_BITS))
        return readers.f1_with_addr(frame, reader, profile, branches)
    return Format0Packet(frame=frame)


def _parse_sync(frame: Frame, reader: BitReader, profile: DecoderProfile) -> SyncPacket:
    branch = reader.read(1)
    privilege = reader.read(profile.privilege_width)
    raw_address = reader.read(profile.address_bits)
    return SyncPacket(
        frame=frame,
        branch=branch,
        privilege=privilege,
        raw_address=raw_address,
        pc=profile.pc_from_raw(raw_address),
    )


__all__ = [
    "F3_CONTEXT",
    "F3_SUPPORT",
    "F3_SYNC",
    "F3_TRAP",
    "FORMAT_F0",
    "FORMAT_F1",
    "FORMAT_F2",
    "FORMAT_F3",
    "QUAL_STATUS_ENDED_NTR",
    "QUAL_STATUS_ENDED_REP",
    "BitReader",
    "ContextPacket",
    "Format0Packet",
    "Format1NoAddrPacket",
    "Format1WithAddrPacket",
    "Format2Packet",
    "FormatReaders",
    "Packet",
    "PacketParseError",
    "SupportPacket",
    "SyncPacket",
    "TrapPacket",
    "branch_map_width",
    "parse_packet",
]
