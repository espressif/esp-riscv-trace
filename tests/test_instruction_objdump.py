# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Both instruction decoders against objdump, over every instruction in the test ELFs.

The control-flow decoder gives the class, link kind and direct target. The semantics decoder gives the registers an
instruction reads and writes, its memory access and its text. The expected values come from objdump's operand text,
so they do not share a decoding table with the code under test. objdump's ``.word`` and ``.insn`` lines are skipped,
since some are data in the text section and others encodings the ELF's architecture string does not name.
"""

from __future__ import annotations

import os
import re
import shutil
import subprocess
from collections.abc import Iterator
from pathlib import Path

import pytest
from helpers import APPS

from esp_riscv_trace.decoder.encoding import REG_NAMES
from esp_riscv_trace.decoder.instruction import decode
from esp_riscv_trace.decoder.semantics import decode_insn, insn_text
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

# Distinct instruction words of the ELFs, 27,012 with objdump of esp-16.1.0_20260609. Without any one of the ELFs
# there are at most 24,396, so a missing ELF fails the check. The margin leaves room for objdump versions that
# disassemble a few more or fewer words.
MIN_DISTINCT_WORDS = 26_000

# "  2f008016:	c33fa0ef          	jal	ra,2f002c48 <esp_crosscore_int_send_yield>"
_LINE = re.compile(r"^\s*([0-9a-f]+):\t([0-9a-f]+)\s*\t(\S+)\s*([^<#]*)")


def _listing(elf: Path, *options: str) -> Iterator[tuple[int, int, str, str]]:
    """Yield the address, instruction word, mnemonic and operand text of each instruction objdump prints for ``elf``."""
    assert OBJDUMP is not None
    out = subprocess.run([OBJDUMP, "-d", *options, str(elf)], capture_output=True, text=True, check=True).stdout
    for line in out.splitlines():
        m = _LINE.match(line)
        if m is None:
            continue
        addr, raw, mnemonic, operands = m.groups()
        # Data that objdump cannot disassemble
        if mnemonic.startswith("."):
            continue
        yield int(addr, 16), int(raw, 16), mnemonic, operands.strip()


# ---------------------------------------------------------------------------
# Control flow
# ---------------------------------------------------------------------------

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
    checked = 0
    mismatches: list[str] = []
    for pc, raw, mnemonic, operand_text in _listing(elf, "-M", "no-aliases"):
        size = 4 if raw & 0x3 == 0x3 else 2
        decoded = decode(pc, raw.to_bytes(size, "little"))
        operands = [op.strip() for op in operand_text.split(",") if op.strip()]
        got = (decoded.cls, decoded.link, decoded.direct_target)
        want = _expected(mnemonic, operands)
        checked += 1
        if got != want:
            mismatches.append(f"0x{pc:08x} 0x{raw:x} {mnemonic} {operand_text}: decoded {got}, objdump says {want}")
    assert checked > 10_000
    assert not mismatches, f"{len(mismatches)} of {checked} differ:\n" + "\n".join(mismatches[:20])


# ---------------------------------------------------------------------------
# Register effects
# ---------------------------------------------------------------------------

_REG = re.compile(r"(?<![\w.])(zero|ra|sp|gp|tp|t[0-6]|s1[01]|s[0-9]|a[0-7])(?!\w)")
_MEM_OPERAND = re.compile(r"(-?\d+)?\((\w+)\)")
_RLIST = re.compile(r"\{ra(?:,s0(?:-s(\d+))?)?\}")
_S_REGS = ["s0", "s1"] + [f"s{i}" for i in range(2, 12)]
_NO_DEST = {
    "sb", "sh", "sw", "fsw", "fsd", "c.sw", "c.swsp", "c.sb", "c.sh", "c.fsw", "c.fswsp", "c.fsd", "c.fsdsp",
    "beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez",
    "c.j", "c.jr", "fence", "fence.i", "ecall", "ebreak", "c.ebreak", "mret", "sret", "wfi", "cm.jt",
}  # fmt: skip
# Write ra without naming it
_LINK_RA = {"c.jal", "c.jalr", "cm.jalt"}
# Compressed two-address forms, whose first operand is read and written
_READS_DEST = {
    "c.addi", "c.addi16sp", "c.slli", "c.srli", "c.srai", "c.andi", "c.add", "c.sub", "c.xor", "c.or", "c.and",
    "c.mul", "c.not", "c.zext.b", "c.zext.h", "c.sext.b", "c.sext.h",
}  # fmt: skip
_ZCMP_POPS = {"cm.pop", "cm.popret", "cm.popretz"}


def _reg(name: str) -> int:
    return REG_NAMES.index(name)


def _regs(*names: str) -> frozenset[int]:
    return frozenset(_reg(n) for n in names)


def _zcmp_list(operands: str) -> frozenset[int]:
    match = _RLIST.search(operands)
    assert match, operands
    names = ["ra"]
    if ",s0" in operands:
        last = int(match.group(1) or 0)
        names += _S_REGS[: last + 1]
    return _regs(*names)


def _expected_effects(mnemonic: str, operands: str) -> tuple[frozenset[int], frozenset[int]]:
    """Return the registers read and written, from the operand text and the operand order of each form."""
    named = _regs(*_REG.findall(operands))
    if mnemonic == "cm.push":
        return _zcmp_list(operands) | _regs("sp"), _regs("sp")
    if mnemonic in _ZCMP_POPS:
        writes = _zcmp_list(operands) | _regs("sp")
        return _regs("sp"), writes | _regs("a0") if mnemonic == "cm.popretz" else writes
    if mnemonic == "cm.mvsa01":
        return _regs("a0", "a1"), named
    if mnemonic == "cm.mva01s":
        return named, _regs("a0", "a1")
    if mnemonic in _NO_DEST or mnemonic in _LINK_RA:
        return named - {0}, _regs("ra") if mnemonic in _LINK_RA else frozenset()
    first = operands.split(",")[0]
    if not _REG.fullmatch(first):
        # A floating point destination
        return named - {0}, frozenset()
    rest = _regs(*_REG.findall(operands[len(first) :]))
    reads = rest | {_reg(first)} if mnemonic in _READS_DEST else rest
    return reads - {0}, frozenset({_reg(first)}) - {0}


def _effect_mismatches(insn: int, mnemonic: str, operands: str) -> list[str]:
    info = decode_insn(insn)
    if mnemonic.startswith("esp.") or mnemonic == "c.unimp":
        return ["decoded as supported"] if info.supported else []
    if not info.supported:
        return ["unsupported"]
    if info.mnemonic != mnemonic:
        return [f"decoded as {info.mnemonic}"]
    problems = []
    reads, writes = _expected_effects(mnemonic, operands)
    if info.writes != writes:
        problems.append(f"writes {sorted(info.writes)}, expected {sorted(writes)}")
    if info.reads != reads:
        problems.append(f"reads {sorted(info.reads)}, expected {sorted(reads)}")
    operand = _MEM_OPERAND.search(operands)
    if operand and mnemonic != "jalr" and not mnemonic.startswith("cm."):
        base, offset = _reg(operand.group(2)), int(operand.group(1) or 0)
        if info.mem is None or (info.mem.base, info.mem.offset) != (base, offset):
            problems.append(f"memory access {info.mem}, expected base {base} offset {offset}")
    return problems


def test_semantics_agree_with_objdump() -> None:
    """Each distinct instruction word of the ELFs, once."""
    seen: dict[int, tuple[str, str]] = {}
    for elf in ELFS:
        for _, insn, mnemonic, operands in _listing(elf, "-M", "no-aliases"):
            seen.setdefault(insn, (mnemonic, operands))
    assert len(seen) >= MIN_DISTINCT_WORDS
    failures = [
        f"0x{insn:08x} {mnemonic} {operands}: {problem}"
        for insn, (mnemonic, operands) in sorted(seen.items())
        for problem in _effect_mismatches(insn, mnemonic, operands)
    ]
    assert not failures, "\n".join(failures[:20])


def _objdump_text(mnemonic: str, operands: str) -> str:
    """Return objdump's text with the jump target as GDB prints it, 0x and all."""
    if mnemonic in ("j", "jal"):
        *link, target = operands.split(",")
        operands = ",".join([*link, f"0x{target}"])
    return f"{mnemonic} {operands}".strip()


def test_insn_text_agrees_with_objdump() -> None:
    """A text ``insn_text`` gives must be the one objdump and GDB print, pseudo-instructions included.

    It may give none, but never a wrong one.
    """
    seen: dict[tuple[int, int], str] = {}
    for elf in ELFS:
        for pc, insn, mnemonic, operands in _listing(elf):
            # Only a jump target depends on the address
            seen.setdefault((insn, pc if mnemonic in ("j", "jal") else 0), _objdump_text(mnemonic, operands))
    failures = []
    covered = 0
    for (insn, pc), expected in sorted(seen.items()):
        text = insn_text(decode_insn(insn), pc)
        if text is not None:
            covered += 1
            if text != expected:
                failures.append(f"0x{insn:08x} at 0x{pc:x}: {text!r}, objdump {expected!r}")
    assert not failures, "\n".join(failures[:30])
    # Most of the instructions a value is written by, not a handful
    assert covered > len(seen) // 2
