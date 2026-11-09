# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""The Python API that esp-coredump uses to read trace captures."""

from __future__ import annotations

import pytest
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment
from helpers import APPS, IDF_ENV_VARS, TESTS

from esp_riscv_trace import NOTE_NAME, NOTE_TYPE, ElfShaStatus, HistoryKind, format_note, open_capture, parse_note

S31_COREDUMP = TESTS / "esp32s31" / "coredump.elf"
S31_ELF = APPS / "esp32s31.elf"

_ILLEGAL_INSTRUCTION = 2


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


def test_format_note_describes_the_note_without_decoding() -> None:
    with S31_COREDUMP.open("rb") as f:
        desc = next(
            bytes(note["n_desc"])
            for segment in ELFFile(f).iter_segments()
            if isinstance(segment, NoteSegment)
            for note in segment.iter_notes()
            if note["n_name"] == NOTE_NAME and note["n_type"] == NOTE_TYPE
        )
    assert format_note(*parse_note(desc)) == [
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
