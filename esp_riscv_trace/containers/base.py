# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Common types shared by every container loader."""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

from ..errors import EspRiscvTraceError
from ..snapshot import abi


class ContainerError(EspRiscvTraceError):
    """Raised when a container file cannot be loaded or is malformed."""


@dataclass(frozen=True)
class TraceBuffer:
    """Captured trace buffer data and metadata for one core."""

    core_id: int
    raw: bytes
    capacity: int
    head_offset: int
    memory_mode: int
    wrap_state: int
    packet_format: int
    address_mode: int
    fifo_status_raw: int
    intr_status_raw: int
    owner: int
    state: int
    head_valid: bool
    encoder_params: abi.EncoderParams | None
    segment_index: int | None
    resync_mode: int = int(abi.ResyncMode.UNKNOWN)
    resync_threshold: int = 0
    fifo_empty: bool = False
    memory_full: bool = False
    fifo_overflow: bool = False

    @classmethod
    def from_record(
        cls,
        rec: abi.CoreDesc | abi.NoteRecord,
        raw: bytes,
        params: abi.EncoderParams | None,
        segment_index: int | None,
    ) -> TraceBuffer:
        """Build a buffer from an ESPT core descriptor or a coredump note record."""
        return cls(
            core_id=rec.core_id,
            raw=raw,
            capacity=rec.capacity,
            head_offset=rec.head_offset,
            memory_mode=rec.memory_mode,
            wrap_state=int(abi.WrapState.UNKNOWN),
            packet_format=rec.packet_format,
            address_mode=rec.address_mode,
            fifo_status_raw=rec.fifo_status_raw,
            intr_status_raw=rec.intr_status_raw,
            owner=rec.owner,
            state=rec.state,
            head_valid=bool(rec.head_valid),
            fifo_empty=bool(rec.fifo_empty),
            memory_full=bool(rec.memory_full),
            fifo_overflow=bool(rec.fifo_overflow),
            encoder_params=params,
            segment_index=segment_index,
            resync_mode=rec.resync_mode,
            resync_threshold=rec.resync_threshold,
        )

    def __post_init__(self) -> None:
        if self.capacity <= 0:
            raise ContainerError(f"core {self.core_id} capacity {self.capacity} must be positive")
        if len(self.raw) != self.capacity:
            raise ContainerError(
                f"core {self.core_id} buffer length {len(self.raw)} does not match capacity {self.capacity}"
            )
        if self.head_offset < 0 or self.head_offset > self.capacity:
            raise ContainerError(f"core {self.core_id} head_offset {self.head_offset} outside [0, {self.capacity}]")
        if self.memory_mode == int(abi.MemoryMode.LINEAR) and self.wrap_state == int(abi.WrapState.WRAPPED):
            raise ContainerError(
                f"core {self.core_id} has memory_mode=LINEAR but wrap_state=WRAPPED, which is impossible"
            )


@dataclass(frozen=True)
class Capture:
    """A loaded container. One or more trace buffers plus common metadata."""

    source: str
    source_path: Path
    abi_major: int
    abi_minor: int
    capture_reason: int
    write_seq: int
    target_id: int | None
    chip_revision: int | None
    app_elf_sha256: bytes | None
    cores: tuple[TraceBuffer, ...] = field(default_factory=tuple)
    fifo_emptied: bool | None = None  # None when the container has no END status

    def __post_init__(self) -> None:
        object.__setattr__(self, "cores", tuple(self.cores))
        seen: set[int] = set()
        for c in self.cores:
            if c.core_id in seen:
                raise ContainerError(f"duplicate core_id {c.core_id} in capture")
            seen.add(c.core_id)


__all__ = [
    "Capture",
    "ContainerError",
    "TraceBuffer",
]
