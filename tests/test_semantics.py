# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Integer register effects of RV32 instructions, and their agreement with the control-flow decoder.

A walk back over the traced path names the instruction that wrote a register only if these are right, so a wrong
write set would put a wrong writer into a finding. ``test_instruction_objdump.py`` checks every instruction of the
app ELFs. These cases cover what the ELFs lack and the encodings the two decoders share.
"""

from __future__ import annotations

import itertools

import pytest
from helpers import BASE

from esp_riscv_trace.decoder.encoding import MASK32, REG_NAMES, InsnError, length
from esp_riscv_trace.decoder.instruction import decode
from esp_riscv_trace.decoder.semantics import InsnInfo, MemAccess, decode_insn, evaluate, sources
from esp_riscv_trace.ir.events import InsnClass, LinkKind


def reg(name: str) -> int:
    return REG_NAMES.index(name)


def regs(*names: str) -> frozenset[int]:
    return frozenset(reg(n) for n in names)


PC = 0x42000100


# Encodings from riscv32-esp-elf-as
@pytest.mark.parametrize(
    ("insn", "target", "known", "value"),
    [
        (0x408007B7, "a5", {}, 0x40800000),  # lui a5,0x40800
        (0xFFC78793, "a5", {"a5": 0x40800000}, 0x407FFFFC),  # addi a5,a5,-4
        (0x00001517, "a0", {}, PC + 0x1000),  # auipc a0,0x1
        (0x00500513, "a0", {}, 5),  # addi a0,zero,5 reads x0 as zero
        (0x57FD, "a5", {}, 0xFFFFFFFF),  # c.li a5,-1
        (0x87AA, "a5", {"a0": 0x1234}, 0x1234),  # c.mv a5,a0
        (0x00B50533, "a0", {"a0": 0xFFFFFFFF, "a1": 1}, 0),  # add a0,a0,a1 wraps at 32 bits
        (0x40455513, "a0", {"a0": 0x80000000}, 0xF8000000),  # srai keeps the sign
        (0x20F747B3, "a5", {"a4": 3, "a5": 0x1000}, 0x100C),  # sh2add a5,a4,a5
        (0x713D, "sp", {"sp": 0x3FC88000}, 0x3FC87FE0),  # c.addi16sp sp,-32
        (0x0808, "a0", {"sp": 0x3FC88000}, 0x3FC88010),  # c.addi4spn a0,sp,16
        (0x9FF5, "a5", {"a5": 0}, 0xFFFFFFFF),  # c.not a5
        (0x9FE1, "a5", {"a5": 0x1FF}, 0xFF),  # c.zext.b a5
        (0xFF7FF0EF, "ra", {}, PC + 4),  # jal ra writes the return address
        (0x9782, "ra", {}, PC + 2),  # c.jalr a5
        (0xB862, "sp", {"sp": 0x3FC88000}, 0x3FC87FF0),  # cm.push {ra,s0-s1},-16
        (0xBA62, "sp", {"sp": 0x3FC87FF0}, 0x3FC88000),  # cm.pop {ra,s0-s1},16
        (0xBEF2, "sp", {"sp": 0x3FC87FC0}, 0x3FC88000),  # cm.popret {ra,s0-s11},64
        (0xBC42, "a0", {}, 0),  # cm.popretz {ra},16
        (0xAD66, "a0", {"s2": 7, "s1": 9}, 7),  # cm.mva01s s2,s1
        (0xAD66, "a1", {"s2": 7, "s1": 9}, 9),
        (0xAC26, "s1", {"a0": 7, "a1": 9}, 9),  # cm.mvsa01 s0,s1
        (0xAC62, "a1", {"s0": 7}, 7),  # cm.mva01s s0,s0
    ],
)
def test_evaluate(insn: int, target: str, known: dict[str, int], value: int) -> None:
    known_regs = {reg(name): v for name, v in known.items()}
    assert evaluate(decode_insn(insn), reg(target), PC, known_regs) == value


# Encodings from riscv32-esp-elf-as -march=rv32ima. No app ELF has the .aq and .rl forms of the AMOs.
_ATOMICS = [
    (0x00B6252F, "amoadd.w", "a0", "a1", "a2"),
    (0x04B6252F, "amoadd.w.aq", "a0", "a1", "a2"),
    (0x02B6252F, "amoadd.w.rl", "a0", "a1", "a2"),
    (0x06B6252F, "amoadd.w.aqrl", "a0", "a1", "a2"),
    (0x0C9122AF, "amoswap.w.aq", "t0", "s1", "sp"),
    (0xE2B6202F, "amomaxu.w.rl", "zero", "a1", "a2"),
]


@pytest.mark.parametrize(("insn", "mnemonic", "rd", "rs2", "base"), _ATOMICS, ids=[a[1] for a in _ATOMICS])
def test_amo_ordering(insn: int, mnemonic: str, rd: str, rs2: str, base: str) -> None:
    info = decode_insn(insn)
    assert (info.mnemonic, info.size, info.supported) == (mnemonic, 4, True)
    assert info.reads == regs(rs2, base)
    assert info.writes == info.loaded == regs(rd) - {0}
    # An AMO counts as a store, as mcause reports its faults
    assert info.mem == MemAccess(reg(base), 0, 4, is_store=True)
    if rd != "zero":
        # The old value comes from memory
        assert sources(info, reg(rd)) is None


@pytest.mark.parametrize(
    ("insn", "mnemonic", "reads", "writes", "is_store"),
    [
        (0x1406252F, "lr.w.aq", ("a2",), ("a0",), False),
        (0x1206252F, "lr.w.rl", ("a2",), ("a0",), False),
        (0x1CB626AF, "sc.w.aq", ("a1", "a2"), ("a3",), True),
        (0x1EB626AF, "sc.w.aqrl", ("a1", "a2"), ("a3",), True),
    ],
)
def test_reservation_ordering(
    insn: int, mnemonic: str, reads: tuple[str, ...], writes: tuple[str, ...], is_store: bool
) -> None:
    info = decode_insn(insn)
    assert (info.mnemonic, info.reads, info.writes) == (mnemonic, regs(*reads), regs(*writes))
    assert info.mem == MemAccess(reg("a2"), 0, 4, is_store=is_store)
    # lr.w loads rd. sc.w writes its success status, which comes from neither memory nor registers.
    assert info.loaded == (regs(*writes) if not is_store else frozenset())
    assert sources(info, reg(writes[0])) is None


# Reserved encodings, and the Zcmp and Zcmt forms that share the c.fsdsp encoding
_FORMS = [
    ("cm_push", 0xB862, "cm.push"),  # cm.push {ra,s0-s1},-16
    ("cm_popret", 0xBEF2, "cm.popret"),  # cm.popret {ra,s0-s11},64
    ("cm_mva01s", 0xAD66, "cm.mva01s"),  # cm.mva01s s2,s1
    ("cm_jt_31", 0xA07E, "cm.jt"),
    ("cm_jalt_32", 0xA082, "cm.jalt"),
    ("cm_push_reserved_rlist", 0xB802, None),
    ("cm_popret_reserved_rlist", 0xBE02, None),
    ("cm_mva01s_same_registers", 0xAC62, "cm.mva01s"),  # cm.mva01s s0,s0
    ("cm_mvsa01_same_registers", 0xAC22, None),  # cm.mvsa01 s0,s0
    ("c_fsdsp_slot_reserved", 0xA402, None),  # bits 12..10 0b001
    ("c_addi4spn_zero", 0x0004, None),
    ("c_addi16sp_zero", 0x6101, None),
    ("c_lui_zero", 0x6781, None),
    ("c_srli_shamt_32", 0x9101, None),
    ("c_slli_shamt_32", 0x1502, None),
    ("c_jr_x0", 0x8002, None),
    ("c_lwsp_x0", 0x4002, None),
    ("jalr_funct3", 0x00009067, None),
    ("branch_funct3", 0x00B52063, None),
    ("slli_shamt_32", 0x02051513, None),
    ("ld", 0x00053503, None),  # RV64 only
    ("fcvt_w_s_rounding_101", 0xC0005553, None),
    ("fcvt_w_s_rounding_110", 0xC0006553, None),
    ("fcvt_w_s_dynamic_rounding", 0xC0007553, "fcvt.w.s"),
    ("fadd_s_rounding_101", 0x00005053, None),
    ("fmadd_s_rounding_110", 0x00006043, None),
    ("esp_vst_128_ip", 0x8311009F, None),  # PIE, in the 48-bit prefix space
]


@pytest.mark.parametrize(("insn", "mnemonic"), [f[1:] for f in _FORMS], ids=[f[0] for f in _FORMS])
def test_forms(insn: int, mnemonic: str | None) -> None:
    size = 4 if insn & 0x3 == 0x3 else 2
    info = decode_insn(insn)
    assert info.size == size
    if mnemonic is None:
        # Nothing is known, so a walk stops here instead of guessing
        assert (info.supported, info.mnemonic) == (False, "unknown")
        assert not (info.reads or info.writes or info.loaded or info.mem)
    else:
        assert (info.supported, info.mnemonic) == (True, mnemonic)
    # The control-flow decoder reads the same length and does not reject reserved encodings
    assert decode(BASE, insn.to_bytes(size, "little")).length == size


def test_unimp() -> None:
    assert decode_insn(0x0000) == InsnInfo(insn=0, size=2, mnemonic="c.unimp", supported=False)


@pytest.mark.parametrize("insn", [0x0000003F, 0x0000007F, 0x000000BF])
def test_wide_encoding_rejected(insn: int) -> None:
    with pytest.raises(InsnError, match="wider than 32 bits"):
        decode_insn(insn)


def _ras_link(rd: int, rs1: int | None) -> LinkKind:
    """Return-address stack hint table of the RISC-V unprivileged ISA, whose link registers are ra and t0."""
    writes, reads = rd in (1, 5), rs1 in (1, 5)
    if writes and reads:
        return LinkKind.CALL if rd == rs1 else LinkKind.COROUTINE
    if writes:
        return LinkKind.CALL
    return LinkKind.RETURN if reads else LinkKind.JUMP


_BRANCHES = {"beq", "bne", "blt", "bge", "bltu", "bgeu", "c.beqz", "c.bnez"}
_ZCM_LINKS = {
    "cm.jt": LinkKind.JUMP,
    "cm.jalt": LinkKind.CALL,
    "cm.popret": LinkKind.RETURN,
    "cm.popretz": LinkKind.RETURN,
}


def _control_flow(info: InsnInfo, pc: int) -> tuple[int, InsnClass, LinkKind, int | None]:
    """What the control-flow decoder must say about an instruction, from its semantics."""
    name = info.mnemonic if info.supported else None
    if name in _BRANCHES:
        assert info.imm is not None
        return info.size, InsnClass.BRANCH, LinkKind.NONE, (pc + info.imm) & MASK32
    if name in ("jal", "c.j", "c.jal"):
        assert info.imm is not None
        return info.size, InsnClass.JUMP, _ras_link(info.rd or 0, None), (pc + info.imm) & MASK32
    if name in ("jalr", "c.jr", "c.jalr"):
        return info.size, InsnClass.UNINFERABLE, _ras_link(info.rd or 0, info.rs1), None
    if name in ("mret", "sret"):
        return info.size, InsnClass.UNINFERABLE, LinkKind.TRAP_RETURN, None
    if name in _ZCM_LINKS:
        return info.size, InsnClass.UNINFERABLE, _ZCM_LINKS[name], None
    if name in ("ecall", "ebreak", "c.ebreak"):
        return info.size, InsnClass.TRAP, LinkKind.NONE, None
    return info.size, InsnClass.NONE, LinkKind.NONE, None


def _disagreements(words: list[int], pc: int) -> list[str]:
    problems = []
    for insn in words:
        size = length(insn & 0xFFFF)
        decoded = decode(pc, insn.to_bytes(size, "little"))
        got = (decoded.length, decoded.cls, decoded.link, decoded.direct_target)
        want = _control_flow(decode_insn(insn), pc)
        if got != want:
            problems.append(f"0x{insn:0{size * 2}x}: control flow {got}, semantics {want}")
    return problems


def test_views_agree_on_every_compressed_encoding() -> None:
    words = [hw for hw in range(0x10000) if hw & 0x3 != 0x3]
    problems = _disagreements(words, BASE)
    assert not problems, "\n".join(problems[:20])


def test_views_agree_on_32bit_encodings() -> None:
    """Every major opcode and funct3, with the register and immediate fields the control-flow decoder reads."""
    words = []
    for opcode in range(0x03, 0x80, 0x04):
        if opcode & 0x3F == 0x3F:
            continue
        for funct3, rd, rs1, upper in itertools.product(
            range(8), (0, 1, 5, 10), (0, 1, 5, 10), (0x000, 0x001, 0x102, 0x105, 0x302, 0x7FF, 0x800, 0xFFF)
        ):
            words.append((upper << 20) | (rs1 << 15) | (funct3 << 12) | (rd << 7) | opcode)
    problems = _disagreements(words, BASE)
    assert not problems, "\n".join(problems[:20])
