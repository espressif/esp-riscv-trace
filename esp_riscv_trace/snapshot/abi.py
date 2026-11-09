# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RISC-V trace snapshot ABI and coredump note parsers."""

from __future__ import annotations

import dataclasses
import enum
import struct
from dataclasses import dataclass
from typing import Any, TypeVar

from ..errors import EspRiscvTraceError

MAGIC = 0x53545652
ABI_MAJOR = 1
ABI_MINOR = 1
SNAPSHOT_DESC_SIZE = 32
CORE_DESC_SIZE = 36
ENCODER_PARAMS_SIZE = 28
ENCODER_PARAMS_VERSION = 1

# Eight exceeds current Espressif hart counts and rejects corrupt data.
CORE_COUNT_MAX = 8

NOTE_NAME = "ESP_RISCV_TRACE"
NOTE_TYPE = 680
NOTE_HEADER_SIZE = 80
NOTE_RECORD_SIZE = 44
NOTE_NO_SEGMENT = 0xFFFFFFFF


class CaptureReason(enum.IntEnum):
    UNKNOWN = 0
    EXPLICIT_STOP = 1
    PANIC = 2
    MEMORY_FULL = 3


class PacketFormat(enum.IntEnum):
    UNKNOWN = 0
    PROCESSOR_TRACE_1_0 = 100
    EFFICIENT_TRACE_2_0 = 200


class State(enum.IntEnum):
    UNAVAILABLE = 0
    READY = 1
    CAPTURING = 2
    STOPPED = 3
    FROZEN = 4


class WrapState(enum.IntEnum):
    """Decoder-side wrap state. Not stored in the on-target ABI."""

    UNKNOWN = 0
    NOT_WRAPPED = 1
    WRAPPED = 2


class MemoryMode(enum.IntEnum):
    UNKNOWN = 0
    LINEAR = 1
    LOOP = 2


class AddressMode(enum.IntEnum):
    UNKNOWN = 0
    DELTA = 1
    FULL = 2


class Owner(enum.IntEnum):
    FW = 0
    OPENOCD = 1


class ResyncMode(enum.IntEnum):
    UNKNOWN = 0
    DISABLED = 1
    PACKET = 2
    CYCLE = 3


def enum_name(cls: type[enum.IntEnum], value: int) -> str:
    """Return the lower-case enum member name or ``unknown(<value>)``."""
    try:
        return cls(value).name.lower()
    except ValueError:
        return f"unknown({value})"


class AbiError(EspRiscvTraceError):
    """Raised when a byte range fails snapshot ABI structural validation."""


@dataclass(frozen=True)
class SnapshotDesc:
    """Snapshot descriptor (32 bytes, ABI v1.x)."""

    magic: int
    abi_major: int
    abi_minor: int
    snapshot_desc_size: int
    core_desc_size: int
    core_count: int
    capture_reason: int
    write_seq: int
    cores_addr: int
    app_elf_sha256_addr: int
    app_elf_sha256_size: int
    target_id: int
    chip_revision: int
    encoder_params_addr: int
    encoder_params_size: int

    @classmethod
    def parse(cls, data: bytes) -> SnapshotDesc:
        return _parse_snapshot_desc(data)


@dataclass(frozen=True)
class CoreDesc:
    """Per-core descriptor (36 bytes, ABI v1.x)."""

    core_id: int
    state: int
    buffer_addr: int
    capacity: int
    head_offset: int
    memory_mode: int
    packet_format: int
    address_mode: int
    resync_mode: int
    owner: int
    head_valid: int
    fifo_empty: int
    memory_full: int
    fifo_overflow: int
    resync_threshold: int
    fifo_status_raw: int
    intr_status_raw: int

    @classmethod
    def parse(cls, data: bytes) -> CoreDesc:
        return _parse_core_desc(data)


@dataclass(frozen=True)
class EncoderParams:
    """Version 1 encoder parameters (28 bytes)."""

    params_version: int
    params_size: int
    arch_p: int
    bpred_size_p: int
    cache_size_p: int
    call_counter_size_p: int
    ctype_width_p: int
    context_width_p: int
    ecause_width_p: int
    ecause_choice_p: int
    f0s_width_p: int
    filter_context_p: int
    filter_excint_p: int
    filter_privilege_p: int
    filter_tval_p: int
    iaddress_lsb_p: int
    iaddress_width_p: int
    iretire_width_p: int
    ilastsize_width_p: int
    itype_width_p: int
    nocontext_p: int
    notime_p: int
    privilege_width_p: int
    retires_p: int
    return_stack_size_p: int
    sijump_p: int
    taken_branches_p: int
    impdef_width_p: int

    @classmethod
    def parse(cls, data: bytes) -> EncoderParams:
        return _parse_encoder_params(data)


@dataclass(frozen=True)
class NoteHeader:
    """ESP_RISCV_TRACE coredump note header (80 bytes)."""

    magic: int
    abi_major: int
    abi_minor: int
    note_header_size: int
    note_record_size: int
    core_count: int
    capture_reason: int
    write_seq: int
    target_id: int
    chip_revision: int
    app_elf_sha256_size: int
    app_elf_sha256: bytes
    encoder_params: EncoderParams | None

    @classmethod
    def parse(cls, data: bytes) -> NoteHeader:
        return _parse_note_header(data)


@dataclass(frozen=True)
class NoteRecord:
    """Per-core coredump note record (44 bytes)."""

    core_id: int
    state: int
    segment_index: int
    segment_vaddr: int
    segment_size: int
    capacity: int
    head_offset: int
    memory_mode: int
    packet_format: int
    address_mode: int
    resync_mode: int
    owner: int
    head_valid: int
    fifo_empty: int
    memory_full: int
    fifo_overflow: int
    resync_threshold: int
    fifo_status_raw: int
    intr_status_raw: int

    def has_segment(self) -> bool:
        return self.segment_index != NOTE_NO_SEGMENT

    @classmethod
    def parse(cls, data: bytes) -> NoteRecord:
        return _parse_note_record(data)


_SNAPSHOT_DESC_FMT = "<IIIIIHHBBBBBBBB"
_CORE_DESC_FMT = "<6I12B"
_ENCODER_PARAMS_FMT = "<BB26B"
_NOTE_HEADER_FMT = "<IIHHBBBBBBBB32s28s"
_NOTE_RECORD_FMT = "<12B8I"

# Field names in wire order, one per struct format item
_SNAPSHOT_DESC_FIELDS = (
    "magic",
    "write_seq",
    "cores_addr",
    "app_elf_sha256_addr",
    "encoder_params_addr",
    "target_id",
    "chip_revision",
    "abi_major",
    "abi_minor",
    "snapshot_desc_size",
    "core_desc_size",
    "core_count",
    "capture_reason",
    "app_elf_sha256_size",
    "encoder_params_size",
)
_CORE_DESC_FIELDS = (
    "buffer_addr",
    "capacity",
    "head_offset",
    "resync_threshold",
    "fifo_status_raw",
    "intr_status_raw",
    "core_id",
    "state",
    "memory_mode",
    "packet_format",
    "address_mode",
    "resync_mode",
    "owner",
    "head_valid",
    "fifo_empty",
    "memory_full",
    "fifo_overflow",
    "reserved",
)
_ENCODER_PARAMS_FIELDS = tuple(f.name for f in dataclasses.fields(EncoderParams))
_NOTE_HEADER_FIELDS = (
    "magic",
    "write_seq",
    "target_id",
    "chip_revision",
    "abi_major",
    "abi_minor",
    "note_header_size",
    "note_record_size",
    "core_count",
    "capture_reason",
    "app_elf_sha256_size",
    "encoder_params_size",
    "app_elf_sha256",
    "encoder_params_blob",
)
_NOTE_RECORD_FIELDS = (
    "core_id",
    "state",
    "memory_mode",
    "packet_format",
    "address_mode",
    "resync_mode",
    "owner",
    "head_valid",
    "fifo_empty",
    "memory_full",
    "fifo_overflow",
    "reserved",
    "segment_index",
    "segment_vaddr",
    "segment_size",
    "capacity",
    "head_offset",
    "resync_threshold",
    "fifo_status_raw",
    "intr_status_raw",
)
_FLAG_FIELDS = ("head_valid", "fifo_empty", "memory_full", "fifo_overflow")

_T = TypeVar("_T")


def _unpack(fmt: str, names: tuple[str, ...], data: bytes, size: int, what: str) -> dict[str, Any]:
    if len(data) < size:
        raise AbiError(f"{what} needs {size} bytes, got {len(data)}")
    return dict(zip(names, struct.unpack_from(fmt, data, 0), strict=True))


def _build(cls: type[_T], values: dict[str, Any]) -> _T:
    return cls(**{f.name: values[f.name] for f in dataclasses.fields(cls)})  # type: ignore[arg-type]


def _check_common_header(v: dict[str, Any]) -> None:
    if v["magic"] != MAGIC:
        raise AbiError(f"bad magic 0x{v['magic']:08x}, want 0x{MAGIC:08x}")
    if v["abi_major"] != ABI_MAJOR:
        raise AbiError(f"unsupported abi_major {v['abi_major']}, want {ABI_MAJOR}")
    if v["write_seq"] & 1:
        raise AbiError(f"write_seq {v['write_seq']} is odd, snapshot was mid-update")


def _check_core_count(v: dict[str, Any]) -> None:
    if v["core_count"] == 0 or v["core_count"] > CORE_COUNT_MAX:
        raise AbiError(f"core_count {v['core_count']} out of range 1..{CORE_COUNT_MAX}")


def _check_sha_size(v: dict[str, Any]) -> None:
    if v["app_elf_sha256_size"] not in (0, 32):
        raise AbiError(f"app_elf_sha256_size {v['app_elf_sha256_size']}, want 0 or 32")


def _check_flags(v: dict[str, Any], what: str) -> None:
    for name in _FLAG_FIELDS:
        if v[name] not in (0, 1):
            raise AbiError(f"{name} {v[name]}, want 0 or 1")
    if v["reserved"] != 0:
        raise AbiError(f"{what} reserved byte {v['reserved']}, want 0")


def _check_head(v: dict[str, Any]) -> None:
    if not v["head_valid"] and v["head_offset"] != 0:
        raise AbiError(f"head_valid is 0 but head_offset is {v['head_offset']}")


def _parse_snapshot_desc(data: bytes) -> SnapshotDesc:
    v = _unpack(_SNAPSHOT_DESC_FMT, _SNAPSHOT_DESC_FIELDS, data, SNAPSHOT_DESC_SIZE, "snapshot descriptor")
    _check_common_header(v)
    if v["snapshot_desc_size"] != SNAPSHOT_DESC_SIZE:
        raise AbiError(f"snapshot_desc_size {v['snapshot_desc_size']}, want {SNAPSHOT_DESC_SIZE}")
    if v["core_desc_size"] != CORE_DESC_SIZE:
        raise AbiError(f"core_desc_size {v['core_desc_size']}, want {CORE_DESC_SIZE}")
    _check_core_count(v)
    if v["cores_addr"] == 0:
        raise AbiError(f"core_count {v['core_count']} but cores_addr is 0")
    _check_sha_size(v)
    if v["app_elf_sha256_size"] == 32 and v["app_elf_sha256_addr"] == 0:
        raise AbiError("app_elf_sha256_size is 32 but app_elf_sha256_addr is 0")
    params_addr, params_size = v["encoder_params_addr"], v["encoder_params_size"]
    if bool(params_addr) != bool(params_size):
        raise AbiError(
            f"encoder_params_addr 0x{params_addr:x} and "
            f"encoder_params_size {params_size} must both be zero or both nonzero"
        )
    if params_addr and params_size < ENCODER_PARAMS_SIZE:
        raise AbiError(f"encoder_params_size {params_size} < {ENCODER_PARAMS_SIZE}")
    return _build(SnapshotDesc, v)


def _parse_core_desc(data: bytes) -> CoreDesc:
    v = _unpack(_CORE_DESC_FMT, _CORE_DESC_FIELDS, data, CORE_DESC_SIZE, "core descriptor")
    buffer_addr, capacity, head_offset = v["buffer_addr"], v["capacity"], v["head_offset"]
    _check_flags(v, "core descriptor")
    if buffer_addr != 0 and capacity != 0:
        if head_offset > capacity:
            raise AbiError(f"head_offset {head_offset} exceeds capacity {capacity}")
    elif buffer_addr != 0 or capacity != 0 or head_offset != 0:
        raise AbiError("buffer_addr, capacity and head_offset must all be zero when the buffer is absent")
    elif v["head_valid"]:
        raise AbiError("head_valid set but buffer_addr or capacity is 0")
    _check_head(v)
    return _build(CoreDesc, v)


def _parse_encoder_params(data: bytes) -> EncoderParams:
    v = _unpack(_ENCODER_PARAMS_FMT, _ENCODER_PARAMS_FIELDS, data, ENCODER_PARAMS_SIZE, "encoder params")
    if v["params_version"] != ENCODER_PARAMS_VERSION:
        raise AbiError(f"params_version {v['params_version']}, want {ENCODER_PARAMS_VERSION}")
    if v["params_size"] != ENCODER_PARAMS_SIZE:
        raise AbiError(f"params_size {v['params_size']}, want {ENCODER_PARAMS_SIZE}")
    return _build(EncoderParams, v)


def _parse_note_header(data: bytes) -> NoteHeader:
    v = _unpack(_NOTE_HEADER_FMT, _NOTE_HEADER_FIELDS, data, NOTE_HEADER_SIZE, "note header")
    _check_common_header(v)
    if v["note_header_size"] != NOTE_HEADER_SIZE:
        raise AbiError(f"note_header_size {v['note_header_size']}, want {NOTE_HEADER_SIZE}")
    if v["note_record_size"] != NOTE_RECORD_SIZE:
        raise AbiError(f"note_record_size {v['note_record_size']}, want {NOTE_RECORD_SIZE}")
    _check_core_count(v)
    _check_sha_size(v)
    if v["app_elf_sha256_size"] == 0 and any(v["app_elf_sha256"]):
        raise AbiError("app_elf_sha256 area is non-zero while size is 0")
    params_size = v["encoder_params_size"]
    if params_size not in (0, ENCODER_PARAMS_SIZE):
        raise AbiError(f"encoder_params_size {params_size}, want 0 or {ENCODER_PARAMS_SIZE}")
    params: EncoderParams | None = None
    if params_size == ENCODER_PARAMS_SIZE:
        params = _parse_encoder_params(v["encoder_params_blob"])
    elif any(v["encoder_params_blob"]):
        raise AbiError("encoder params area is non-zero while encoder_params_size is 0")
    v["app_elf_sha256"] = bytes(v["app_elf_sha256"])
    v["encoder_params"] = params
    return _build(NoteHeader, v)


def _parse_note_record(data: bytes) -> NoteRecord:
    v = _unpack(_NOTE_RECORD_FMT, _NOTE_RECORD_FIELDS, data, NOTE_RECORD_SIZE, "note record")
    _check_flags(v, "note record")
    capacity, head_offset = v["capacity"], v["head_offset"]
    if v["segment_index"] != NOTE_NO_SEGMENT:
        if v["segment_vaddr"] == 0 or v["segment_size"] == 0:
            raise AbiError("segment_index is set but segment_vaddr or segment_size is 0")
        if capacity != v["segment_size"]:
            raise AbiError(f"capacity {capacity} does not match segment_size {v['segment_size']}")
        if head_offset > capacity:
            raise AbiError(f"head_offset {head_offset} exceeds capacity {capacity}")
    else:
        if v["segment_vaddr"] != 0 or v["segment_size"] != 0:
            raise AbiError("segment_index is NO_SEGMENT but segment_vaddr or segment_size is set")
        if capacity != 0:
            raise AbiError("buffer is absent but capacity is set")
    _check_head(v)
    return _build(NoteRecord, v)


def parse_note_records(data: bytes, core_count: int) -> list[NoteRecord]:
    """Parse ``core_count`` consecutive note records from the note body."""
    if core_count <= 0 or core_count > CORE_COUNT_MAX:
        raise AbiError(f"core_count {core_count} out of range 1..{CORE_COUNT_MAX}")
    need = core_count * NOTE_RECORD_SIZE
    if len(data) < need:
        raise AbiError(f"note record array needs {need} bytes, got {len(data)}")
    records = []
    for i in range(core_count):
        off = i * NOTE_RECORD_SIZE
        records.append(_parse_note_record(data[off : off + NOTE_RECORD_SIZE]))
    return records


def parse_note(body: bytes) -> tuple[NoteHeader, list[NoteRecord]]:
    """Parse the ESP_RISCV_TRACE note body (header + all records).

    Requires ``len(body) == NOTE_HEADER_SIZE + core_count * NOTE_RECORD_SIZE``.
    Trailing bytes are rejected so container loaders cannot silently accept
    truncated or over-long notes.
    """
    header = _parse_note_header(body[:NOTE_HEADER_SIZE])
    expected = NOTE_HEADER_SIZE + header.core_count * NOTE_RECORD_SIZE
    if len(body) != expected:
        raise AbiError(
            f"note body length {len(body)} does not match expected {expected} for core_count {header.core_count}"
        )
    records = parse_note_records(body[NOTE_HEADER_SIZE:], header.core_count)
    return header, records


__all__ = [
    "ABI_MAJOR",
    "ABI_MINOR",
    "CORE_COUNT_MAX",
    "CORE_DESC_SIZE",
    "ENCODER_PARAMS_SIZE",
    "ENCODER_PARAMS_VERSION",
    "MAGIC",
    "NOTE_HEADER_SIZE",
    "NOTE_NAME",
    "NOTE_NO_SEGMENT",
    "NOTE_RECORD_SIZE",
    "NOTE_TYPE",
    "SNAPSHOT_DESC_SIZE",
    "AbiError",
    "AddressMode",
    "CaptureReason",
    "CoreDesc",
    "EncoderParams",
    "MemoryMode",
    "NoteHeader",
    "NoteRecord",
    "Owner",
    "PacketFormat",
    "ResyncMode",
    "SnapshotDesc",
    "State",
    "WrapState",
    "enum_name",
    "parse_note",
    "parse_note_records",
]
