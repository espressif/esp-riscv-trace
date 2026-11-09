# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Per-core capture header for the trace intermediate representation."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from ..snapshot import abi


class InferredWrapState(StrEnum):
    """Wrap state shown when the decoder inferred it."""

    INFERRED_WRAPPED = "inferred_wrapped"
    INFERRED_NOT_WRAPPED = "inferred_not_wrapped"


class ElfShaStatus(StrEnum):
    """Result of comparing the ELF SHA-256 with the capture."""

    MATCH = "match"
    MISMATCH = "mismatch"
    NO_SNAPSHOT_HASH = "no_snapshot_hash"
    NO_ELF = "no_elf"


@dataclass(frozen=True)
class TraceCaptureNote:
    """One core's decoder header."""

    source: str
    chip: str | None
    core_id: int
    packet_format: str
    address_mode: str
    memory_mode: str
    resync_mode: str
    resync_threshold: int
    wrap_state: str
    capacity: int
    head_offset: int
    fifo_status_raw: int
    intr_status_raw: int
    fifo_empty: bool
    memory_full: bool
    fifo_overflow: bool
    capture_reason: str
    owner: str
    target_id: int | None
    chip_revision: int | None
    snapshot_app_elf_sha256: bytes | None
    elf_path: str | None
    elf_sha256: bytes | None
    elf_sha_status: ElfShaStatus
    wrap_reason: str | None = None
    rom_elf_path: str | None = None
    rom_elf_status: str | None = None
    fifo_emptied: bool | None = None
    abi_major: int = abi.ABI_MAJOR
    abi_minor: int = abi.ABI_MINOR


__all__ = [
    "ElfShaStatus",
    "InferredWrapState",
    "TraceCaptureNote",
]
