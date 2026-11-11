# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Snapshot descriptor and coredump note struct parsing."""

from __future__ import annotations

import struct

import pytest
from helpers import (
    C5_STOP,
    CORE_DESC_DEFAULTS,
    ENCODER_PARAMS_DEFAULTS,
    SNAPSHOT_DESC_DEFAULTS,
    pack_core_desc,
    pack_encoder_params,
    pack_snapshot_desc,
    pack_struct,
)

from esp_riscv_trace.snapshot import abi

_SHA = bytes(range(32))

_NOTE_HEADER_DEFAULTS: dict[str, object] = dict(
    magic=abi.MAGIC,
    write_seq=2,
    target_id=0x0017,
    chip_revision=0x0065,
    abi_major=abi.ABI_MAJOR,
    abi_minor=abi.ABI_MINOR,
    note_header_size=abi.NOTE_HEADER_SIZE,
    note_record_size=abi.NOTE_RECORD_SIZE,
    core_count=1,
    capture_reason=int(abi.CaptureReason.PANIC),
    app_elf_sha256_size=32,
    encoder_params_size=abi.ENCODER_PARAMS_SIZE,
    app_elf_sha256=_SHA,
    encoder_params_blob=pack_encoder_params(),
)

_NOTE_RECORD_DEFAULTS: dict[str, object] = dict(
    core_id=0,
    state=int(abi.State.FROZEN),
    memory_mode=int(abi.MemoryMode.LOOP),
    packet_format=int(abi.PacketFormat.EFFICIENT_TRACE_2_0),
    address_mode=int(abi.AddressMode.FULL),
    resync_mode=int(abi.ResyncMode.PACKET),
    owner=int(abi.Owner.FW),
    head_valid=1,
    fifo_empty=1,
    memory_full=0,
    fifo_overflow=0,
    reserved=0,
    segment_index=3,
    segment_vaddr=0x4080D714,
    segment_size=64,
    capacity=64,
    head_offset=16,
    resync_threshold=128,
    fifo_status_raw=1,
    intr_status_raw=0,
)

# Field values that leave the buffer absent in a note record
_NO_SEGMENT = dict(segment_index=abi.NOTE_NO_SEGMENT, segment_vaddr=0, segment_size=0, capacity=0)


def _note_header(**overrides) -> bytes:
    return pack_struct(abi._NOTE_HEADER_FMT, _NOTE_HEADER_DEFAULTS, overrides)


def _note_record(**overrides) -> bytes:
    return pack_struct(abi._NOTE_RECORD_FMT, _NOTE_RECORD_DEFAULTS, overrides)


@pytest.mark.parametrize(
    ("fmt", "size"),
    [
        (abi._SNAPSHOT_DESC_FMT, abi.SNAPSHOT_DESC_SIZE),
        (abi._CORE_DESC_FMT, abi.CORE_DESC_SIZE),
        (abi._ENCODER_PARAMS_FMT, abi.ENCODER_PARAMS_SIZE),
        (abi._NOTE_HEADER_FMT, abi.NOTE_HEADER_SIZE),
        (abi._NOTE_RECORD_FMT, abi.NOTE_RECORD_SIZE),
    ],
)
def test_struct_format_matches_size(fmt: str, size: int) -> None:
    assert struct.calcsize(fmt) == size


def test_snapshot_desc_round_trip() -> None:
    desc = abi.SnapshotDesc.parse(pack_snapshot_desc())
    assert desc == abi.SnapshotDesc(**SNAPSHOT_DESC_DEFAULTS)  # type: ignore[arg-type]


def test_snapshot_desc_without_sha_or_params() -> None:
    data = pack_snapshot_desc(
        app_elf_sha256_addr=0, app_elf_sha256_size=0, encoder_params_addr=0, encoder_params_size=0
    )
    desc = abi.SnapshotDesc.parse(data)
    assert desc.app_elf_sha256_size == 0
    assert desc.encoder_params_size == 0


def test_snapshot_desc_accepts_larger_params_block() -> None:
    assert abi.SnapshotDesc.parse(pack_snapshot_desc(encoder_params_size=64)).encoder_params_size == 64


def test_real_capture_header_parses() -> None:
    data = C5_STOP.read_bytes()
    desc = abi.SnapshotDesc.parse(data[8:40])
    assert desc.magic == abi.MAGIC
    assert desc.write_seq == 8
    assert desc.target_id == 0x0017
    assert desc.chip_revision == 0x0065
    assert desc.core_count == 1
    assert desc.capture_reason == abi.CaptureReason.EXPLICIT_STOP
    params = abi.EncoderParams.parse(data[40:68])
    assert params.params_version == abi.ENCODER_PARAMS_VERSION
    assert params.params_size == abi.ENCODER_PARAMS_SIZE


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        (dict(magic=0x12345678), "bad magic 0x12345678"),
        (dict(abi_major=2), "unsupported abi_major 2"),
        (dict(write_seq=9), "write_seq 9 is odd"),
        (dict(snapshot_desc_size=31), "snapshot_desc_size 31"),
        (dict(core_desc_size=0), "core_desc_size 0"),
        (dict(core_count=0), "core_count 0 out of range"),
        (dict(core_count=abi.CORE_COUNT_MAX + 1), f"core_count {abi.CORE_COUNT_MAX + 1} out of range"),
        (dict(cores_addr=0), "cores_addr is 0"),
        (dict(app_elf_sha256_size=16), "app_elf_sha256_size 16"),
        (dict(app_elf_sha256_addr=0), "app_elf_sha256_addr is 0"),
        (dict(encoder_params_addr=0), "must both be zero or both nonzero"),
        (dict(encoder_params_size=0), "must both be zero or both nonzero"),
        (dict(encoder_params_size=abi.ENCODER_PARAMS_SIZE - 1), f"< {abi.ENCODER_PARAMS_SIZE}"),
    ],
)
def test_snapshot_desc_rejects(overrides: dict, fragment: str) -> None:
    with pytest.raises(abi.AbiError, match=fragment):
        abi.SnapshotDesc.parse(pack_snapshot_desc(**overrides))


def test_snapshot_desc_rejects_short_input() -> None:
    with pytest.raises(abi.AbiError, match="snapshot descriptor needs 32 bytes, got 31"):
        abi.SnapshotDesc.parse(pack_snapshot_desc()[:-1])


def test_core_desc_round_trip() -> None:
    desc = abi.CoreDesc.parse(pack_core_desc())
    expected = {k: v for k, v in CORE_DESC_DEFAULTS.items() if k != "reserved"}
    assert desc == abi.CoreDesc(**expected)  # type: ignore[arg-type]


def test_core_desc_absent_buffer() -> None:
    desc = abi.CoreDesc.parse(pack_core_desc(buffer_addr=0, capacity=0, head_offset=0, head_valid=0))
    assert desc.capacity == 0
    assert desc.head_valid == 0


def test_core_desc_head_at_capacity() -> None:
    assert abi.CoreDesc.parse(pack_core_desc(head_offset=16)).head_offset == 16


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        (dict(head_valid=2), "head_valid 2, want 0 or 1"),
        (dict(fifo_empty=2), "fifo_empty 2, want 0 or 1"),
        (dict(memory_full=3), "memory_full 3, want 0 or 1"),
        (dict(fifo_overflow=255), "fifo_overflow 255, want 0 or 1"),
        (dict(reserved=1), "core descriptor reserved byte 1"),
        (dict(head_offset=17), "head_offset 17 exceeds capacity 16"),
        (dict(buffer_addr=0), "must all be zero when the buffer is absent"),
        (dict(capacity=0), "must all be zero when the buffer is absent"),
        (dict(buffer_addr=0, capacity=0), "must all be zero when the buffer is absent"),
        (dict(buffer_addr=0, capacity=0, head_offset=0), "head_valid set but buffer_addr or capacity is 0"),
        (dict(head_valid=0), "head_valid is 0 but head_offset is 4"),
    ],
)
def test_core_desc_rejects(overrides: dict, fragment: str) -> None:
    with pytest.raises(abi.AbiError, match=fragment):
        abi.CoreDesc.parse(pack_core_desc(**overrides))


def test_encoder_params_round_trip() -> None:
    params = abi.EncoderParams.parse(pack_encoder_params(ecause_width_p=6, privilege_width_p=2, impdef_width_p=9))
    expected = {**ENCODER_PARAMS_DEFAULTS, "ecause_width_p": 6, "privilege_width_p": 2, "impdef_width_p": 9}
    assert params == abi.EncoderParams(**expected)  # type: ignore[arg-type]


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        (pack_encoder_params(params_version=2), "params_version 2, want 1"),
        (pack_encoder_params(params_size=27), "params_size 27, want 28"),
        (pack_encoder_params()[:-1], "encoder params needs 28 bytes, got 27"),
    ],
)
def test_encoder_params_rejects(data: bytes, fragment: str) -> None:
    with pytest.raises(abi.AbiError, match=fragment):
        abi.EncoderParams.parse(data)


def test_note_header_round_trip() -> None:
    header = abi.NoteHeader.parse(_note_header())
    assert header.magic == abi.MAGIC
    assert header.write_seq == 2
    assert header.target_id == 0x0017
    assert header.chip_revision == 0x0065
    assert header.core_count == 1
    assert header.capture_reason == abi.CaptureReason.PANIC
    assert header.app_elf_sha256_size == 32
    assert header.app_elf_sha256 == _SHA
    assert header.encoder_params == abi.EncoderParams.parse(pack_encoder_params())


def test_note_header_without_sha_or_params() -> None:
    header = abi.NoteHeader.parse(
        _note_header(app_elf_sha256_size=0, app_elf_sha256=bytes(32), encoder_params_size=0, encoder_params_blob=b"")
    )
    assert header.app_elf_sha256 == bytes(32)
    assert header.encoder_params is None


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        (dict(magic=0), "bad magic"),
        (dict(abi_major=0), "unsupported abi_major 0"),
        (dict(write_seq=3), "write_seq 3 is odd"),
        (dict(note_header_size=79), "note_header_size 79"),
        (dict(note_record_size=40), "note_record_size 40"),
        (dict(core_count=0), "core_count 0 out of range"),
        (dict(core_count=abi.CORE_COUNT_MAX + 1), "out of range"),
        (dict(app_elf_sha256_size=1), "app_elf_sha256_size 1"),
        (dict(app_elf_sha256_size=0), "app_elf_sha256 area is non-zero while size is 0"),
        (dict(encoder_params_size=12), "encoder_params_size 12, want 0 or 28"),
        (dict(encoder_params_size=0), "encoder params area is non-zero while encoder_params_size is 0"),
        (dict(encoder_params_blob=pack_encoder_params(params_version=0)), "params_version 0"),
    ],
)
def test_note_header_rejects(overrides: dict, fragment: str) -> None:
    with pytest.raises(abi.AbiError, match=fragment):
        abi.NoteHeader.parse(_note_header(**overrides))


def test_note_record_round_trip() -> None:
    record = abi.NoteRecord.parse(_note_record())
    expected = {k: v for k, v in _NOTE_RECORD_DEFAULTS.items() if k != "reserved"}
    assert record == abi.NoteRecord(**expected)  # type: ignore[arg-type]
    assert record.has_segment()


def test_note_record_without_segment() -> None:
    record = abi.NoteRecord.parse(_note_record(**_NO_SEGMENT, head_valid=0, head_offset=0))
    assert not record.has_segment()


@pytest.mark.parametrize(
    ("overrides", "fragment"),
    [
        (dict(fifo_empty=2), "fifo_empty 2, want 0 or 1"),
        (dict(reserved=7), "note record reserved byte 7"),
        (dict(segment_vaddr=0), "segment_index is set but segment_vaddr or segment_size is 0"),
        (dict(segment_size=0), "segment_index is set but segment_vaddr or segment_size is 0"),
        (dict(capacity=32), "capacity 32 does not match segment_size 64"),
        (dict(head_offset=65), "head_offset 65 exceeds capacity 64"),
        ({**_NO_SEGMENT, "segment_vaddr": 0x4080D714}, "is NO_SEGMENT but segment_vaddr or segment_size is set"),
        ({**_NO_SEGMENT, "segment_size": 64}, "is NO_SEGMENT but segment_vaddr or segment_size is set"),
        ({**_NO_SEGMENT, "capacity": 64}, "buffer is absent but capacity is set"),
        (dict(head_valid=0), "head_valid is 0 but head_offset is 16"),
    ],
)
def test_note_record_rejects(overrides: dict, fragment: str) -> None:
    with pytest.raises(abi.AbiError, match=fragment):
        abi.NoteRecord.parse(_note_record(**overrides))


def test_parse_note_two_cores() -> None:
    body = _note_header(core_count=2) + _note_record() + _note_record(core_id=1, segment_index=4)
    header, records = abi.parse_note(body)
    assert header.core_count == 2
    assert [r.core_id for r in records] == [0, 1]
    assert [r.segment_index for r in records] == [3, 4]


@pytest.mark.parametrize(
    "body",
    [
        _note_header() + _note_record() + b"\x00",
        _note_header() + _note_record()[:-1],
        _note_header(core_count=2) + _note_record(),
    ],
    ids=["trailing", "short", "missing_record"],
)
def test_parse_note_rejects_body_length(body: bytes) -> None:
    with pytest.raises(abi.AbiError, match="note body length"):
        abi.parse_note(body)


@pytest.mark.parametrize(
    ("data", "core_count", "fragment"),
    [
        (_note_record(), 0, "core_count 0 out of range"),
        (_note_record(), abi.CORE_COUNT_MAX + 1, "out of range"),
        (_note_record(), 2, "note record array needs 88 bytes, got 44"),
    ],
)
def test_parse_note_records_rejects(data: bytes, core_count: int, fragment: str) -> None:
    with pytest.raises(abi.AbiError, match=fragment):
        abi.parse_note_records(data, core_count)


@pytest.mark.parametrize(
    ("cls", "value", "name"),
    [
        (abi.CaptureReason, 2, "panic"),
        (abi.PacketFormat, 200, "efficient_trace_2_0"),
        (abi.MemoryMode, 9, "unknown(9)"),
    ],
)
def test_enum_name(cls, value: int, name: str) -> None:
    assert abi.enum_name(cls, value) == name
