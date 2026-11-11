# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Control-flow class, link kind and direct target of every instruction in the test ELFs, checked against objdump."""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from pathlib import Path

import pytest
from helpers import APPS

from esp_riscv_trace.decoder.instruction import decode
from esp_riscv_trace.ir.events import InsnClass, LinkKind

_OBJDUMP = "riscv32-esp-elf-objdump"


def _find_objdump() -> str | None:
    found = shutil.which(_OBJDUMP)
    if found is not None:
        return found
    tools = Path(os.environ.get("IDF_TOOLS_PATH", Path.home() / ".espressif")) / "tools" / "riscv32-esp-elf"
    candidates = sorted(tools.glob(f"*/riscv32-esp-elf/bin/{_OBJDUMP}"))
    return str(candidates[-1]) if candidates else None


OBJDUMP = _find_objdump()

pytestmark = pytest.mark.skipif(OBJDUMP is None, reason=f"{_OBJDUMP} is not on PATH or in IDF_TOOLS_PATH")

ELFS = sorted(APPS.glob("*.elf"))

# "  2f008016:	c33fa0ef          	jal	ra,2f002c48 <esp_crosscore_int_send_yield>"
_LINE = re.compile(r"^\s*([0-9a-f]+):\t([0-9a-f]+)\s*\t(\S+)\s*([^<#]*)")

_LINK_REGS = {"ra", "t0"}
_BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
_TRAPS = {"ecall", "ebreak", "c.ebreak"}
_ZCM_JUMPS = {
    "cm.popret": LinkKind.RETURN,
    "cm.popretz": LinkKind.RETURN,
    "cm.jt": LinkKind.JUMP,
    "cm.jalt": LinkKind.CALL,
}


def _ras_link(rd: str, rs1: str | None) -> LinkKind:
    """Return-address stack hint table of the RISC-V unprivileged ISA."""
    writes, reads = rd in _LINK_REGS, rs1 in _LINK_REGS
    if writes and reads:
        return LinkKind.CALL if rd == rs1 else LinkKind.COROUTINE
    if writes:
        return LinkKind.CALL
    return LinkKind.RETURN if reads else LinkKind.JUMP


def _expected(mnemonic: str, operands: list[str]) -> tuple[InsnClass, LinkKind, int | None]:
    if mnemonic in _BRANCHES:
        return InsnClass.BRANCH, LinkKind.NONE, int(operands[-1], 16)
    if mnemonic == "jal":
        return InsnClass.JUMP, _ras_link(operands[0], None), int(operands[-1], 16)
    if mnemonic in ("c.j", "c.jal"):
        return InsnClass.JUMP, LinkKind.CALL if mnemonic == "c.jal" else LinkKind.JUMP, int(operands[-1], 16)
    if mnemonic == "jalr":
        rs1 = re.fullmatch(r"-?\d*\((\w+)\)", operands[1])
        assert rs1 is not None, operands
        return InsnClass.UNINFERABLE, _ras_link(operands[0], rs1.group(1)), None
    if mnemonic in ("c.jr", "c.jalr"):
        return InsnClass.UNINFERABLE, _ras_link("ra" if mnemonic == "c.jalr" else "zero", operands[0]), None
    if mnemonic in ("mret", "sret"):
        return InsnClass.UNINFERABLE, LinkKind.TRAP_RETURN, None
    if mnemonic in _ZCM_JUMPS:
        return InsnClass.UNINFERABLE, _ZCM_JUMPS[mnemonic], None
    if mnemonic in _TRAPS:
        return InsnClass.TRAP, LinkKind.NONE, None
    return InsnClass.NONE, LinkKind.NONE, None


@pytest.mark.parametrize("elf", ELFS, ids=[e.stem for e in ELFS])
def test_decode_agrees_with_objdump(elf: Path) -> None:
    assert OBJDUMP is not None
    listing = subprocess.run(
        [OBJDUMP, "-d", "-M", "no-aliases", str(elf)], capture_output=True, text=True, check=True
    ).stdout
    checked = 0
    mismatches: list[str] = []
    for line in listing.splitlines():
        m = _LINE.match(line)
        if m is None:
            continue
        addr, raw, mnemonic, operand_text = m.groups()
        # Data that objdump cannot disassemble
        if mnemonic.startswith("."):
            continue
        pc = int(addr, 16)
        decoded = decode(pc, int(raw, 16).to_bytes(len(raw) // 2, "little"))
        operands = [op.strip() for op in operand_text.split(",") if op.strip()]
        got = (decoded.cls, decoded.link, decoded.direct_target)
        want = _expected(mnemonic, operands)
        checked += 1
        if got != want:
            mismatches.append(f"{line.strip()}: decoded {got}, objdump says {want}")
    assert checked > 10_000
    assert not mismatches, f"{len(mismatches)} of {checked} differ:\n" + "\n".join(mismatches[:20])
