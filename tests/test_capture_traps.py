# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""The faulting instruction of real crash captures."""

from __future__ import annotations

from pathlib import Path

import pytest
from helpers import APPS, IDF_ENV_VARS, TESTS

from esp_riscv_trace import session
from esp_riscv_trace.ir.events import TraceGap, TraceInstruction, TraceTrap

# (capture, app ELF, cause, epc, tval, last two retired PCs)
CRASHES = [
    # abort() ends in an illegal instruction
    (TESTS / "esp32s31" / "coredump.elf", APPS / "esp32s31.elf", 2, 0x2F0071DC, 0, [0x2F0071D4, 0x2F0071D8]),
    (TESTS / "esp32c5" / "coredump.elf", APPS / "esp32c5.elf", 2, 0x40806E94, 0, [0x40806E8C, 0x40806E90]),
]


@pytest.mark.parametrize(
    ("capture", "elf", "cause", "epc", "tval", "last"),
    CRASHES,
    ids=[f"{c[0].parent.name}-{c[0].stem}" for c in CRASHES],
)
def test_exception_is_taken_at_the_faulting_instruction(
    monkeypatch, capture: Path, elf: Path, cause: int, epc: int, tval: int, last: list[int]
) -> None:
    for name in IDF_ENV_VARS:
        monkeypatch.delenv(name, raising=False)
    with session.open_session(capture, elf=elf) as sess:
        (view,) = [v for v in session.iter_cores(sess, None) if v.core.core_id == 0]
        events = list(view.events())

    at = next(i for i, e in enumerate(events) if isinstance(e, TraceTrap) and not e.interrupt)
    trap = events[at]
    assert isinstance(trap, TraceTrap)
    assert (trap.cause, trap.epc, trap.tval) == (cause, epc, tval)
    retired = [e for e in events[:at] if isinstance(e, TraceInstruction)]
    assert [e.pc for e in retired[-2:]] == last
    assert retired[-1].next_pc == epc
    assert not any(isinstance(e, TraceGap) for e in events[events.index(retired[-1]) : at])
