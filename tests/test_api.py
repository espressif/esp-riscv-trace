# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""The Python API that esp-coredump uses to read trace captures."""

from __future__ import annotations

import pytest
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from helpers import APPS, IDF_ENV_VARS, TESTS

from esp_riscv_trace import (
    NOTE_NAME,
    NOTE_TYPE,
    ElfShaStatus,
    HistoryBuilder,
    HistoryKind,
    NoteHeader,
    compare_elf_sha,
    format_note,
    open_capture,
    parse_note,
)
from esp_riscv_trace.elf.image import ElfImage

S31_COREDUMP = TESTS / "esp32s31" / "coredump.elf"
S31_ELF = APPS / "esp32s31.elf"
C5_ELF = APPS / "esp32c5.elf"

_ILLEGAL_INSTRUCTION = 2
# pxCurrentTCBs of the esp32s31 app, one pointer per core
_CURRENT_TCBS = 0x2F00EC64


def _note_desc() -> bytes:
    with S31_COREDUMP.open("rb") as f:
        return next(
            bytes(note["n_desc"])
            for segment in ELFFile(f).iter_segments()
            if isinstance(segment, NoteSegment)
            for note in segment.iter_notes()
            if note["n_name"] == NOTE_NAME and note["n_type"] == NOTE_TYPE
        )


def _note_header() -> NoteHeader:
    return parse_note(_note_desc())[0]


@pytest.fixture(autouse=True)
def _no_idf_env(monkeypatch: pytest.MonkeyPatch) -> None:
    for name in IDF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)


def test_coredump_history_ends_at_the_fault_with_its_source_line() -> None:
    with open_capture(S31_COREDUMP, elf=S31_ELF) as capture:
        assert capture.chip == "esp32s31"
        assert [core.core_id for core in capture.cores] == [0, 1]
        core = capture.core(0)
        assert core is not None
        assert core.note.elf_sha_status == ElfShaStatus.MATCH

        history = core.history(last=5)
        fault = history.fault
        assert fault is not None and fault.pc is not None
        assert (fault.kind, fault.cause, fault.tval) == (HistoryKind.FAULT, _ILLEGAL_INSTRUCTION, 0)
        assert len(history.items) == 5
        assert history.omitted > 0

        where = capture.resolve(fault.pc)
        assert where.function == "panic_abort"
        assert where.file is not None and where.file.endswith("panic.c")
        assert capture.image is not None
        assert len(capture.image.read(fault.pc, 2)) == 2


def test_capture_names_the_variable_at_an_address() -> None:
    with open_capture(S31_COREDUMP, elf=S31_ELF) as capture:
        assert capture.data_at(_CURRENT_TCBS) == "pxCurrentTCBs"
        assert capture.data_at(_CURRENT_TCBS + 4) == "pxCurrentTCBs+4"
        # code, which no data symbol covers
        assert capture.data_at(0x40000020) is None


def test_capture_without_an_elf_names_no_variable() -> None:
    with open_capture(S31_COREDUMP) as capture:
        assert capture.data_at(_CURRENT_TCBS) is None


@pytest.mark.parametrize(
    ("elf", "status"),
    [(S31_ELF, ElfShaStatus.MATCH), (C5_ELF, ElfShaStatus.MISMATCH), (None, ElfShaStatus.NO_ELF)],
    ids=["match", "mismatch", "no_elf"],
)
def test_capture_compares_the_whole_recorded_digest(elf, status) -> None:
    with open_capture(S31_COREDUMP, elf=elf) as capture:
        assert {core.note.elf_sha_status for core in capture.cores} == {status}


@pytest.mark.parametrize(
    ("size", "elf", "status"),
    [
        (32, S31_ELF, ElfShaStatus.MATCH),
        (8, S31_ELF, ElfShaStatus.MATCH),
        (1, S31_ELF, ElfShaStatus.MATCH),
        (32, C5_ELF, ElfShaStatus.MISMATCH),
        (8, C5_ELF, ElfShaStatus.MISMATCH),
        (0, S31_ELF, ElfShaStatus.NO_SNAPSHOT_HASH),
        (40, S31_ELF, ElfShaStatus.MATCH),  # longer than the digest the note holds
    ],
)
def test_recorded_hash_is_compared_up_to_its_declared_size(size, elf, status) -> None:
    header = _note_header()
    assert header.app_elf_sha256_size == 32
    with ElfImage(elf) as image:
        assert compare_elf_sha(header.app_elf_sha256, size, image.sha256()) == status


def test_no_elf_takes_precedence_over_a_missing_hash() -> None:
    assert compare_elf_sha(b"", 0, None) == ElfShaStatus.NO_ELF
    assert compare_elf_sha(bytes(32), 32, None) == ElfShaStatus.NO_ELF


def test_history_builder_fed_one_event_at_a_time_matches_core_history() -> None:
    with open_capture(S31_COREDUMP, elf=S31_ELF) as capture:
        core = capture.core(0)
        assert core is not None
        builder = HistoryBuilder(capture.image)
        for event in core.events():
            builder.push(event)
            if builder.done:
                break
        assert builder.close(last=20) == core.history(last=20)


def test_format_note_describes_the_note_without_decoding() -> None:
    assert format_note(*parse_note(_note_desc())) == [
        "ABI 1.0, reason PANIC, cores 2",
        "Target 0x0020, chip revision v0.0",
        "App ELF SHA256: 04a6a7ec10b0c0146d34f1a1a476e020203113f61bd46ed24360c03e70d9f5a9",
        "Core 0: memory LOOP, format ET20, address DELTA",
        "  resync PACKET, threshold 128",
        "  buffer 0x2f0116dc, capacity 0x1000, head 0xb30, segment 17",
        "  hw fifo 0x00000001, intr 0x00000000, empty yes, full no, overflow no",
        "Core 1: memory LOOP, format ET20, address DELTA",
        "  resync PACKET, threshold 128",
        "  buffer 0x2f0105bc, capacity 0x1000, head 0x850, segment 18",
        "  hw fifo 0x00000001, intr 0x00000000, empty yes, full no, overflow no",
    ]
