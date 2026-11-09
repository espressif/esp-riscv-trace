# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Describe the trace note of a core dump in the lines esp-coredump ``info_corefile`` prints."""

from __future__ import annotations

import enum
from collections.abc import Sequence

from ..snapshot.abi import AddressMode, CaptureReason, MemoryMode, NoteHeader, NoteRecord, PacketFormat, ResyncMode

_FORMAT_NAMES: dict[int, str] = {
    PacketFormat.PROCESSOR_TRACE_1_0: "PT10",
    PacketFormat.EFFICIENT_TRACE_2_0: "ET20",
}


def format_note(header: NoteHeader, records: Sequence[NoteRecord]) -> list[str]:
    """Return the lines that describe the note header and the record of each core."""
    lines = [
        f"ABI {header.abi_major}.{header.abi_minor}, reason {_name(CaptureReason, header.capture_reason)}, "
        f"cores {header.core_count}",
        f"Target 0x{header.target_id:04x}, chip revision v{header.chip_revision // 100}.{header.chip_revision % 100}",
    ]
    if any(header.app_elf_sha256):
        lines.append(f"App ELF SHA256: {header.app_elf_sha256.hex()}")
    for record in records:
        lines += _format_record(record)
    return lines


def _format_record(record: NoteRecord) -> list[str]:
    packet_format = _FORMAT_NAMES.get(record.packet_format) or _name(PacketFormat, record.packet_format)
    lines = [
        f"Core {record.core_id}: memory {_name(MemoryMode, record.memory_mode)}, format {packet_format}, "
        f"address {_name(AddressMode, record.address_mode)}",
        f"  resync {_name(ResyncMode, record.resync_mode)}, threshold {record.resync_threshold}",
    ]
    if record.has_segment():
        head = f"0x{record.head_offset:x}" if record.head_valid else "invalid"
        lines.append(
            f"  buffer 0x{record.segment_vaddr:x}, capacity 0x{record.capacity:x}, head {head}, "
            f"segment {record.segment_index}"
        )
    else:
        lines.append("  no buffer")
    lines.append(
        f"  hw fifo 0x{record.fifo_status_raw:08x}, intr 0x{record.intr_status_raw:08x}, "
        f"empty {_yes_no(record.fifo_empty)}, full {_yes_no(record.memory_full)}, "
        f"overflow {_yes_no(record.fifo_overflow)}"
    )
    flags = _quality_flags(record)
    if flags:
        lines.append(f"  quality: {', '.join(flags)}")
    return lines


def _name(cls: type[enum.IntEnum], value: int) -> str:
    try:
        return cls(value).name
    except ValueError:
        return str(value)


def _yes_no(flag: int) -> str:
    return "yes" if flag else "no"


def _quality_flags(record: NoteRecord) -> list[str]:
    flags = []
    if not record.head_valid:
        flags.append("HEAD_INVALID")
    if record.memory_full:
        flags.append("MEMORY_FULL")
    if record.fifo_overflow:
        flags.append("FIFO_OVERFLOW")
    if record.has_segment() and not record.fifo_empty:
        flags.append("FIFO_NOT_EMPTY")
    return flags


__all__ = ["format_note"]
