# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""The Python API that esp-coredump uses to read trace captures."""

from __future__ import annotations

import pytest
from helpers import APPS, IDF_ENV_VARS, TESTS

from esp_riscv_trace import ElfShaStatus, HistoryKind, open_capture

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
