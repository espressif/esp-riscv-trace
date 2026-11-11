# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""The history command and decode --last.

The exact output of every capture is checked in test_captures.py.
"""

from __future__ import annotations

import re

from helpers import APPS, TESTS, run_main

from esp_riscv_trace import cli

S31_COREDUMP = TESTS / "esp32s31" / "coredump.elf"
S31_ELF = APPS / "esp32s31.elf"

_FAULT_EPC = "0x2f0071dc"


def _s31_coredump(cmd: str, *extra: str):
    return run_main(cmd, S31_COREDUMP, "--elf", S31_ELF, "--core", "0", "--quiet", *extra, expect=cli.EXIT_OK)


def _outline_rows(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln and not ln.startswith("#")]


def _instruction_lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln.startswith("0x")]


def test_history_ends_at_the_fault() -> None:
    p = _s31_coredump("history")

    assert "# fault             illegal instruction in panic_abort" in p.stdout
    assert re.search(r"^# history +core=0  items=40  omitted=\d+", p.stdout, re.MULTILINE)
    rows = _outline_rows(p.stdout)
    assert "panic_abort() {" in rows[-2]
    assert "!! illegal instruction in panic_abort" in rows[-1]
    assert rows[-1].rstrip().endswith("panic.c:508")


def test_decode_last_ends_at_the_fault() -> None:
    p = _s31_coredump("decode", "--last", "5")

    window = next(ln for ln in p.stdout.splitlines() if ln.startswith("# window"))
    assert re.search(r"instructions=5  omitted=\d+  ends_at=fault  cause=2  epc=0x2f0071dc$", window)
    assert len(_instruction_lines(p.stdout)) == 5
    events = [ln for ln in p.stdout.splitlines() if ln.startswith(("0x", "---"))]
    assert events[-1].startswith("--- trap cause=2")
    assert f"epc={_FAULT_EPC}" in events[-1]
