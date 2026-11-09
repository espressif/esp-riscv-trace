# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""ESPT and coredump container loading and format detection."""

from __future__ import annotations

import struct
from collections.abc import Callable
from pathlib import Path

import pytest
from helpers import C5_ELF, C5_STOP, TESTS, pack_core_desc, pack_encoder_params, pack_snapshot_desc

from esp_riscv_trace import containers
from esp_riscv_trace.containers import ContainerError, coredump, espt
from esp_riscv_trace.snapshot import abi

C5_COREDUMP = TESTS / "esp32c5" / "coredump.elf"

_SHA = bytes(range(32, 64))
_CAPACITY = 16


def _record(rtype: int, core: int, payload: bytes, *, length: int | None = None) -> bytes:
    size = len(payload) if length is None else length
    return espt.RECORD_HEADER.pack(rtype, core, 0, size) + payload


def _core(core: int = 0, **overrides) -> bytes:
    return _record(espt.REC_CORE_DESC, core, pack_core_desc(core_id=core, **overrides))


def _data(core: int = 0, size: int = _CAPACITY) -> bytes:
    return _record(espt.REC_DATA, core, bytes(range(1, size + 1)))


def _end(status: int = espt.END_OK) -> bytes:
    return _record(espt.REC_END, espt.REC_NO_CORE, struct.pack("<I", status))


def _espt(
    *records: bytes,
    magic: bytes = espt.MAGIC,
    version: int = espt.VERSION,
    hdr_size: int = espt.HDR_SIZE,
    snapshot: bytes | None = None,
    params: bytes | None = None,
    extra_header: bytes = b"",
) -> bytes:
    header = espt.PREAMBLE.pack(magic, version, hdr_size)
    header += pack_snapshot_desc() if snapshot is None else snapshot
    header += pack_encoder_params() if params is None else params
    return header + _SHA + extra_header + b"".join(records)


def _write(tmp_path: Path, data: bytes, name: str = "capture.esptrace") -> Path:
    path = tmp_path / name
    path.write_bytes(data)
    return path


def test_espt_single_core(tmp_path: Path) -> None:
    capture = espt.load(_write(tmp_path, _espt(_core(), _data(), _end())))
    assert capture.source == "espt"
    assert capture.abi_major == abi.ABI_MAJOR
    assert capture.write_seq == 8
    assert capture.target_id == 0x0017
    assert capture.chip_revision == 0x0065
    assert capture.capture_reason == abi.CaptureReason.EXPLICIT_STOP
    assert capture.app_elf_sha256 == _SHA
    assert capture.fifo_emptied is True
    (buf,) = capture.cores
    assert buf.raw == bytes(range(1, _CAPACITY + 1))
    assert buf.capacity == _CAPACITY
    assert buf.head_offset == 4
    assert buf.wrap_state == abi.WrapState.UNKNOWN
    assert buf.segment_index is None
    assert buf.encoder_params == abi.EncoderParams.parse(pack_encoder_params())


def test_espt_two_cores_and_extended_header(tmp_path: Path) -> None:
    data = _espt(
        _core(0),
        _data(0),
        _core(1),
        _data(1),
        _end(),
        hdr_size=espt.HDR_SIZE + 4,
        extra_header=bytes(4),
    )
    capture = espt.load(_write(tmp_path, data))
    assert [c.core_id for c in capture.cores] == [0, 1]


def test_espt_without_sha(tmp_path: Path) -> None:
    snapshot = pack_snapshot_desc(app_elf_sha256_addr=0, app_elf_sha256_size=0)
    capture = espt.load(_write(tmp_path, _espt(_core(), _data(), _end(), snapshot=snapshot)))
    assert capture.app_elf_sha256 is None


def test_espt_fifo_not_empty_at_stop(tmp_path: Path) -> None:
    capture = espt.load(_write(tmp_path, _espt(_core(), _data(), _end(espt.END_FIFO_NOT_EMPTY))))
    assert capture.fifo_emptied is False


def test_espt_real_capture() -> None:
    capture = espt.load(C5_STOP)
    assert capture.fifo_emptied is True
    assert len(capture.cores) == 1


_ESPT_ERRORS: list[tuple[str, Callable[[], bytes], str]] = [
    ("short_file", lambda: b"ESPT", "need at least 100 for the ESPT header"),
    ("bad_magic", lambda: _espt(_end(), magic=b"ESPX"), "does not start with the ESPT magic"),
    ("version", lambda: _espt(_end(), version=2), "has ESPT version 2"),
    ("hdr_size_small", lambda: _espt(_end(), hdr_size=espt.HDR_SIZE - 1), "invalid ESPT header size 99"),
    ("hdr_size_large", lambda: _espt(_end(), hdr_size=4096), "invalid ESPT header size 4096"),
    ("bad_snapshot", lambda: _espt(_end(), snapshot=pack_snapshot_desc(magic=0)), "invalid ESPT header. bad magic"),
    ("bad_params", lambda: _espt(_end(), params=pack_encoder_params(params_version=9)), "params_version 9"),
    ("record_header", lambda: _espt(_core(), _data(), b"\x03\xff"), "truncated in a record header"),
    ("record_payload", lambda: _espt(_core(), _record(espt.REC_DATA, 0, bytes(4), length=16)), "is truncated"),
    ("unknown_type", lambda: _espt(_record(9, 0, b""), _end()), "unknown record type 9"),
    ("bad_core_desc", lambda: _espt(_core(reserved=1), _data(), _end()), "core 0 descriptor invalid"),
    (
        "core_mismatch",
        lambda: _espt(_record(1, 1, pack_core_desc()), _end()),
        "CORE_DESC record core 1 holds core_id 0",
    ),
    ("data_first", lambda: _espt(_data(), _end()), "DATA record for core 0 has no matching CORE_DESC"),
    ("data_other_core", lambda: _espt(_core(0), _data(1), _end()), "DATA record for core 1 has no matching"),
    ("data_twice", lambda: _espt(_core(), _data(), _data(), _end()), "has no matching CORE_DESC"),
    ("data_length", lambda: _espt(_core(), _data(size=8), _end()), "DATA is 8 bytes, descriptor capacity is 16"),
    ("no_end", lambda: _espt(_core(), _data()), "has no END record"),
    ("end_length", lambda: _espt(_core(), _data(), _record(3, 0xFF, b"\x00\x00")), "END record has 2 bytes"),
    ("end_status", lambda: _espt(_core(), _data(), _end(2)), "capture ended with status 2"),
    ("after_end", lambda: _espt(_core(), _data(), _end(), _end()), "data after the END record"),
    ("duplicate_core", lambda: _espt(_core(), _data(), _core(), _data(), _end()), "duplicate core_id 0"),
]


@pytest.mark.parametrize(("build", "fragment"), [e[1:] for e in _ESPT_ERRORS], ids=[e[0] for e in _ESPT_ERRORS])
def test_espt_rejects(tmp_path: Path, build: Callable[[], bytes], fragment: str) -> None:
    with pytest.raises(ContainerError, match=fragment):
        espt.load(_write(tmp_path, build()))


def test_coredump_real_capture() -> None:
    capture = coredump.load(C5_COREDUMP)
    assert capture.source == "coredump"
    assert capture.fifo_emptied is None
    (buf,) = capture.cores
    assert buf.segment_index == 9
    assert buf.capacity == 4096
    assert buf.memory_mode == abi.MemoryMode.LOOP
    assert buf.encoder_params is not None


def test_coredump_rejects_non_elf(tmp_path: Path) -> None:
    path = _write(tmp_path, b"\x7fELF" + bytes(60), "bad.elf")
    with pytest.raises(ContainerError, match="is not a valid ELF file"):
        coredump.load(path)


def test_coredump_rejects_executable() -> None:
    with pytest.raises(ContainerError, match="is not a core file"):
        coredump.load(C5_ELF)


_NOTE_NAME = abi.NOTE_NAME.encode() + b"\x00"


def _patched_coredump(tmp_path: Path, offset_from_desc: int, value: bytes) -> Path:
    data = bytearray(C5_COREDUMP.read_bytes())
    desc = data.index(_NOTE_NAME) + len(_NOTE_NAME)
    data[desc + offset_from_desc : desc + offset_from_desc + len(value)] = value
    return _write(tmp_path, bytes(data), "coredump.elf")


@pytest.mark.parametrize(
    ("offset", "value", "fragment"),
    [
        # n_type sits 4 bytes before the name
        (-len(_NOTE_NAME) - 4, struct.pack("<I", abi.NOTE_TYPE + 1), "has no ESP_RISCV_TRACE note"),
        (0, struct.pack("<I", 0), "has a malformed ESP_RISCV_TRACE note. bad magic"),
        # segment_vaddr of the first record
        (abi.NOTE_HEADER_SIZE + 16, struct.pack("<I", 0x4FF00000), "but no PT_LOAD matches"),
    ],
    ids=["no_note", "bad_note", "no_segment"],
)
def test_coredump_rejects_patched(tmp_path: Path, offset: int, value: bytes, fragment: str) -> None:
    with pytest.raises(ContainerError, match=fragment):
        coredump.load(_patched_coredump(tmp_path, offset, value))


@pytest.mark.parametrize(("path", "source"), [(C5_STOP, "espt"), (C5_COREDUMP, "coredump")])
def test_load_detects_format(path: Path, source: str) -> None:
    assert containers.load(path).source == source


def test_load_rejects_unknown_format(tmp_path: Path) -> None:
    with pytest.raises(ContainerError, match="is neither an ESPT trace file nor a coredump ELF"):
        containers.load(_write(tmp_path, b"not a trace file", "unknown.bin"))
