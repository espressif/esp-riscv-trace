# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Integer register effects of RV32 instructions, for following a value back through the trace.

Floating point registers are not tracked. x0 is never in ``reads`` or ``writes``.
"""

from __future__ import annotations

from collections.abc import Callable, Mapping
from dataclasses import dataclass

from ..elf import CodeImage, CodeReadError
from .encoding import (
    A0,
    A1,
    MASK32,
    OPCODE_AMO,
    OPCODE_AUIPC,
    OPCODE_BRANCH,
    OPCODE_JAL,
    OPCODE_JALR,
    OPCODE_LOAD,
    OPCODE_LOAD_FP,
    OPCODE_LUI,
    OPCODE_MADD,
    OPCODE_MASK,
    OPCODE_MISC_MEM,
    OPCODE_MSUB,
    OPCODE_NMADD,
    OPCODE_NMSUB,
    OPCODE_OP,
    OPCODE_OP_FP,
    OPCODE_OP_IMM,
    OPCODE_STORE,
    OPCODE_STORE_FP,
    OPCODE_SYSTEM,
    QUADRANT_32BIT,
    QUADRANT_MASK,
    RA,
    S_REGS,
    SP,
    XLEN_BYTES,
    ZCMP_RLIST_MIN,
    InsnError,
    b_imm,
    cb_imm,
    cj_imm,
    j_imm,
    length,
    reg_name,
    sign_extend,
    zcm_mnemonic,
)

# The first 16-bit parcel of an instruction
_PARCEL_MASK = 0xFFFF
_PARCEL_BYTES = 2
_XLEN_BITS = 32
# Shift amounts and bit indexes of a 32-bit register take five bits
_SHAMT_MASK = 31
# The 3-bit register fields of compressed instructions name x8 to x15
_C_REG_BASE = 8


@dataclass(frozen=True)
class MemAccess:
    """One memory access at base register + offset."""

    base: int
    offset: int
    width: int
    # AMOs load and store. They count as stores, as mcause reports their faults.
    is_store: bool


@dataclass(frozen=True)
class InsnInfo:
    """What one instruction does to the integer registers."""

    insn: int
    size: int
    mnemonic: str  # as riscv32-esp-elf-objdump -M no-aliases prints it
    supported: bool
    reads: frozenset[int] = frozenset()
    writes: frozenset[int] = frozenset()
    # Written registers whose new value comes from memory, which are loads, lr, AMOs and the cm.pop forms
    loaded: frozenset[int] = frozenset()
    # cm.push and cm.pop access several addresses and have no single MemAccess
    mem: MemAccess | None = None
    rd: int | None = None
    rs1: int | None = None
    rs2: int | None = None
    # Sign-extended immediate. lui and auipc hold the shifted value, cm.push a negative stack adjustment.
    imm: int | None = None
    csr: int | None = None


def decode_insn(insn: int) -> InsnInfo:
    """Decode the instruction in the low 16 or 32 bits of ``insn``. Raises ``InsnError`` as ``length()`` does."""
    if length(insn & _PARCEL_MASK) == _PARCEL_BYTES:
        insn &= _PARCEL_MASK
        info = _decode16(insn)
    else:
        insn &= MASK32
        info = _decode32(insn)
    return info if info is not None else _unsupported(insn)


def decode_at(image: CodeImage | None, pc: int) -> InsnInfo | None:
    """Decode the instruction of ``image`` at ``pc``, ``None`` when it cannot be read or decoded."""
    if image is None:
        return None
    try:
        low = int.from_bytes(image.read(pc, _PARCEL_BYTES), "little")
        size = length(low)
        return decode_insn(low if size == _PARCEL_BYTES else int.from_bytes(image.read(pc, size), "little"))
    except (CodeReadError, InsnError):
        return None


def sources(info: InsnInfo, reg: int) -> frozenset[int] | None:
    """Return the registers the value ``info`` writes to ``reg`` is computed from, ``None`` when not from registers.

    An empty set is a constant or an address relative to the instruction.
    """
    if reg not in info.writes or reg in info.loaded or info.csr is not None:
        return None
    mnemonic = info.mnemonic
    if mnemonic.startswith(("f", "sc.")):
        return None
    if mnemonic in _ZCMP_STACK:
        return frozenset() if reg == A0 else frozenset({SP})
    if mnemonic == "cm.mva01s":
        source = info.rs1 if reg == A0 else info.rs2
        assert source is not None
        return frozenset({source})
    if mnemonic == "cm.mvsa01":
        return frozenset({A0 if reg == info.rs1 else A1})
    if mnemonic in LINKS:
        return frozenset()
    return info.reads


def evaluate(info: InsnInfo, reg: int, pc: int, regs: Mapping[int, int]) -> int | None:
    """Return the value ``info`` at ``pc`` writes to ``reg``, ``None`` when ``regs`` or the operation lack it."""
    needed = sources(info, reg)
    if needed is None or any(r not in regs for r in needed):
        return None
    operation = _EVALUATE.get(info.mnemonic)
    if operation is None:
        return None

    def value(r: int | None) -> int:
        return 0 if not r else regs[r] & MASK32

    return operation(info, reg, pc, value) & MASK32


def _unsupported(insn: int, mnemonic: str = "unknown") -> InsnInfo:
    return InsnInfo(insn=insn, size=length(insn & _PARCEL_MASK), mnemonic=mnemonic, supported=False)


# ---------------------------------------------------------------------------
# Memory accesses
# ---------------------------------------------------------------------------

# Access width in bytes by opcode and funct3. The RV64 ld, lwu and sd are left out, the 8-byte ones are fld and fsd.
_LOAD_WIDTHS = {OPCODE_LOAD: {0: 1, 1: 2, 2: 4, 4: 1, 5: 2}, OPCODE_LOAD_FP: {2: 4, 3: 8}}
_STORE_WIDTHS = {OPCODE_STORE: {0: 1, 1: 2, 2: 4}, OPCODE_STORE_FP: {2: 4, 3: 8}}


def _memory_access(insn: int) -> tuple[int, int, int] | None:
    """Return the base register, signed byte offset and width in bytes of a load or store, ``None`` for others."""
    if insn & QUADRANT_MASK == QUADRANT_32BIT:
        opcode, funct3 = insn & OPCODE_MASK, (insn >> 12) & 0x7
        if opcode in _LOAD_WIDTHS:
            width = _LOAD_WIDTHS[opcode].get(funct3)
            imm = insn >> 20
        elif opcode in _STORE_WIDTHS:
            width = _STORE_WIDTHS[opcode].get(funct3)
            imm = ((insn >> 25) << 5) | ((insn >> 7) & 0x1F)
        else:
            return None
        if width is None:
            return None
        return (insn >> 15) & 0x1F, sign_extend(imm, 12), width

    insn &= _PARCEL_MASK
    quadrant, funct3 = insn & QUADRANT_MASK, insn >> 13
    if quadrant == 0:
        rs1 = ((insn >> 7) & 0x7) + _C_REG_BASE
        if funct3 == 4:
            # Zcb c.lbu, c.lhu, c.lh, c.sb, c.sh. Bits 12..10 pick the form.
            form = (insn >> 10) & 0x7
            if form in (0, 2):  # A byte access takes uimm[1] from bit 5 and uimm[0] from bit 6
                return rs1, ((insn >> 5) & 0x1) << 1 | ((insn >> 6) & 0x1), 1
            if form in (1, 3):  # A halfword access takes uimm[1] from bit 5
                return rs1, ((insn >> 5) & 0x1) << 1, 2
            return None
        if funct3 in (1, 5):  # c.fld, c.fsd
            return rs1, ((insn >> 10) & 0x7) << 3 | ((insn >> 5) & 0x3) << 6, 8
        if funct3 in (2, 3, 6, 7):  # c.lw, c.flw, c.sw, c.fsw
            return rs1, ((insn >> 10) & 0x7) << 3 | ((insn >> 6) & 0x1) << 2 | ((insn >> 5) & 0x1) << 6, 4
        return None
    if quadrant == 2:
        if funct3 == 5:
            # Zcmp owns this slot, see zcm_mnemonic()
            return None
        if funct3 == 1:  # c.fldsp
            return SP, ((insn >> 12) & 0x1) << 5 | ((insn >> 5) & 0x3) << 3 | ((insn >> 2) & 0x7) << 6, 8
        if funct3 in (2, 3):  # c.lwsp, c.flwsp
            return SP, ((insn >> 12) & 0x1) << 5 | ((insn >> 4) & 0x7) << 2 | ((insn >> 2) & 0x3) << 6, 4
        if funct3 in (6, 7):  # c.swsp, c.fswsp
            return SP, ((insn >> 9) & 0xF) << 2 | ((insn >> 7) & 0x3) << 6, 4
    return None


# ---------------------------------------------------------------------------
# 32-bit encodings
# ---------------------------------------------------------------------------


def _make(
    insn: int,
    mnemonic: str,
    *,
    reads: tuple[int, ...] = (),
    writes: tuple[int, ...] = (),
    loaded: tuple[int, ...] = (),
    mem: MemAccess | None = None,
    rd: int | None = None,
    rs1: int | None = None,
    rs2: int | None = None,
    imm: int | None = None,
    csr: int | None = None,
) -> InsnInfo:
    return InsnInfo(
        insn=insn,
        size=length(insn & _PARCEL_MASK),
        mnemonic=mnemonic,
        supported=True,
        reads=frozenset(r for r in reads if r),
        writes=frozenset(r for r in writes if r),
        loaded=frozenset(r for r in loaded if r),
        mem=mem,
        rd=rd,
        rs1=rs1,
        rs2=rs2,
        imm=imm,
        csr=csr,
    )


def _mem(insn: int, is_store: bool) -> MemAccess | None:
    access = _memory_access(insn)
    if access is None:
        return None
    return MemAccess(*access, is_store=is_store)


_BRANCHES = {0: "beq", 1: "bne", 4: "blt", 5: "bge", 6: "bltu", 7: "bgeu"}
_LOADS = {0: "lb", 1: "lh", 2: "lw", 4: "lbu", 5: "lhu"}
_STORES = {0: "sb", 1: "sh", 2: "sw"}
_OP_IMM = {0: "addi", 2: "slti", 3: "sltiu", 4: "xori", 6: "ori", 7: "andi"}
# (funct7, funct3) of OP for RV32I, M, Zba, Zbb, Zbs and Zicond
_OP = {
    (0x00, 0): "add", (0x20, 0): "sub", (0x00, 1): "sll", (0x00, 2): "slt",
    (0x00, 3): "sltu", (0x00, 4): "xor", (0x00, 5): "srl", (0x20, 5): "sra",
    (0x00, 6): "or", (0x00, 7): "and",
    (0x01, 0): "mul", (0x01, 1): "mulh", (0x01, 2): "mulhsu", (0x01, 3): "mulhu",
    (0x01, 4): "div", (0x01, 5): "divu", (0x01, 6): "rem", (0x01, 7): "remu",
    (0x10, 2): "sh1add", (0x10, 4): "sh2add", (0x10, 6): "sh3add",
    (0x20, 7): "andn", (0x20, 6): "orn", (0x20, 4): "xnor",
    (0x05, 4): "min", (0x05, 5): "minu", (0x05, 6): "max", (0x05, 7): "maxu",
    (0x05, 1): "clmul", (0x05, 2): "clmulr", (0x05, 3): "clmulh",
    (0x30, 1): "rol", (0x30, 5): "ror",
    (0x24, 1): "bclr", (0x24, 5): "bext", (0x14, 1): "bset", (0x34, 1): "binv",
    (0x07, 5): "czero.eqz", (0x07, 7): "czero.nez",
}  # fmt: skip
# OP-IMM funct3 1 and 5 by imm[11:5]
_SHIFT_IMM = {
    (1, 0x00): "slli", (1, 0x24): "bclri", (1, 0x14): "bseti", (1, 0x34): "binvi",
    (5, 0x00): "srli", (5, 0x20): "srai", (5, 0x24): "bexti", (5, 0x30): "rori",
}  # fmt: skip
# OP-IMM funct3 1 with imm[11:5] 0x30, by the rs2 field
_ZBB_UNARY = {0: "clz", 1: "ctz", 2: "cpop", 4: "sext.b", 5: "sext.h"}
# OP-IMM funct3 5 with the whole 12-bit immediate fixed
_ZBB_FIXED = {0x287: "orc.b", 0x698: "rev8"}
_SYSTEM = {0x00000073: "ecall", 0x00100073: "ebreak", 0x10200073: "sret", 0x30200073: "mret", 0x10500073: "wfi"}
_CSR = {1: "csrrw", 2: "csrrs", 3: "csrrc", 5: "csrrwi", 6: "csrrsi", 7: "csrrci"}
_AMO = {
    0x00: "amoadd.w", 0x01: "amoswap.w", 0x04: "amoxor.w", 0x08: "amoor.w", 0x0C: "amoand.w",
    0x10: "amomin.w", 0x14: "amomax.w", 0x18: "amominu.w", 0x1C: "amomaxu.w",
}  # fmt: skip
_AMO_LR = 0x02
_AMO_SC = 0x03
# By the aq and rl bits, bits 26..25
_AMO_ORDER = ("", ".rl", ".aq", ".aqrl")
_FP_FORMATS = {0: "s", 1: "d"}
# funct3 is the rounding mode of these OP-FP instructions and of the fused ones. 0b101 and 0b110 are reserved.
_FP_ROUNDED = frozenset({0x00, 0x01, 0x02, 0x03, 0x08, 0x0B, 0x18, 0x1A})
_RESERVED_ROUNDING_MODES = frozenset({0b101, 0b110})
_FP_FUSED = {OPCODE_MADD: "fmadd", OPCODE_MSUB: "fmsub", OPCODE_NMSUB: "fnmsub", OPCODE_NMADD: "fnmadd"}
_FP_MEMORY = {
    (OPCODE_LOAD_FP, 2): "flw", (OPCODE_LOAD_FP, 3): "fld", (OPCODE_STORE_FP, 2): "fsw", (OPCODE_STORE_FP, 3): "fsd",
}  # fmt: skip
# OP-FP by funct5 when the result and the operands are all floating point
_FP_ARITH = {0x00: "fadd", 0x01: "fsub", 0x02: "fmul", 0x03: "fdiv"}
_FP_BY_FUNCT3 = {
    0x04: ("fsgnj", "fsgnjn", "fsgnjx"),
    0x05: ("fmin", "fmax"),
    0x14: ("fle", "flt", "feq"),  # codespell:ignore fle
}


def _decode32(insn: int) -> InsnInfo | None:
    opcode = insn & OPCODE_MASK
    rd = (insn >> 7) & 0x1F
    funct3 = (insn >> 12) & 0x7
    rs1 = (insn >> 15) & 0x1F
    rs2 = (insn >> 20) & 0x1F
    funct7 = insn >> 25
    i_imm = sign_extend(insn >> 20, 12)

    if opcode == OPCODE_LUI:
        return _make(insn, "lui", writes=(rd,), rd=rd, imm=sign_extend(insn & 0xFFFFF000, 32))
    if opcode == OPCODE_AUIPC:
        return _make(insn, "auipc", writes=(rd,), rd=rd, imm=sign_extend(insn & 0xFFFFF000, 32))
    if opcode == OPCODE_JAL:
        return _make(insn, "jal", writes=(rd,), rd=rd, imm=j_imm(insn))
    if opcode == OPCODE_JALR:
        if funct3:
            return None
        return _make(insn, "jalr", reads=(rs1,), writes=(rd,), rd=rd, rs1=rs1, imm=i_imm)
    if opcode == OPCODE_BRANCH:
        if funct3 not in _BRANCHES:
            return None
        return _make(insn, _BRANCHES[funct3], reads=(rs1, rs2), rs1=rs1, rs2=rs2, imm=b_imm(insn))
    if opcode == OPCODE_LOAD:
        if funct3 not in _LOADS:
            return None
        return _make(
            insn,
            _LOADS[funct3],
            reads=(rs1,),
            writes=(rd,),
            loaded=(rd,),
            mem=_mem(insn, False),
            rd=rd,
            rs1=rs1,
            imm=i_imm,
        )
    if opcode == OPCODE_STORE:
        if funct3 not in _STORES:
            return None
        mem = _mem(insn, True)
        assert mem is not None
        return _make(insn, _STORES[funct3], reads=(rs1, rs2), mem=mem, rs1=rs1, rs2=rs2, imm=mem.offset)
    if opcode in (OPCODE_LOAD_FP, OPCODE_STORE_FP):
        # Other widths are vector loads and stores
        name = _FP_MEMORY.get((opcode, funct3))
        if name is None:
            return None
        mem = _mem(insn, opcode == OPCODE_STORE_FP)
        assert mem is not None
        return _make(insn, name, reads=(rs1,), mem=mem, rs1=rs1, imm=mem.offset)
    if opcode == OPCODE_OP_IMM:
        return _decode_op_imm(insn, rd, funct3, rs1, rs2, i_imm)
    if opcode == OPCODE_OP:
        name = _OP.get((funct7, funct3))
        if name is None and (funct7, funct3, rs2) == (0x04, 4, 0):
            return _make(insn, "zext.h", reads=(rs1,), writes=(rd,), rd=rd, rs1=rs1)
        if name is None:
            return None
        return _make(insn, name, reads=(rs1, rs2), writes=(rd,), rd=rd, rs1=rs1, rs2=rs2)
    if opcode == OPCODE_MISC_MEM:
        # The register fields of fence are reserved and ignored by the core
        name = {0: "fence", 1: "fence.i"}.get(funct3)
        return _make(insn, name) if name else None
    if opcode == OPCODE_SYSTEM:
        return _decode_system(insn, rd, funct3, rs1)
    if opcode == OPCODE_AMO:
        return _decode_amo(insn, rd, funct3, rs1, rs2, funct7)
    if opcode in _FP_FUSED:
        fmt = _FP_FORMATS.get(funct7 & 0x3)
        if fmt is None or funct3 in _RESERVED_ROUNDING_MODES:
            return None
        return _make(insn, f"{_FP_FUSED[opcode]}.{fmt}")
    if opcode == OPCODE_OP_FP:
        return _decode_op_fp(insn, rd, funct3, rs1, rs2, funct7)
    return None


def _decode_op_imm(insn: int, rd: int, funct3: int, rs1: int, rs2: int, i_imm: int) -> InsnInfo | None:
    if funct3 in _OP_IMM:
        return _make(insn, _OP_IMM[funct3], reads=(rs1,), writes=(rd,), rd=rd, rs1=rs1, imm=i_imm)
    imm12 = insn >> 20
    name = _ZBB_FIXED.get(imm12) if funct3 == 5 else None
    if name is None and funct3 == 1 and imm12 >> 5 == 0x30:
        name = _ZBB_UNARY.get(rs2)
    if name is not None:
        return _make(insn, name, reads=(rs1,), writes=(rd,), rd=rd, rs1=rs1)
    # On RV32 a shift amount is five bits, so imm[5] set is reserved
    name = _SHIFT_IMM.get((funct3, imm12 >> 5))
    if name is None:
        return None
    return _make(insn, name, reads=(rs1,), writes=(rd,), rd=rd, rs1=rs1, imm=rs2)


def _decode_system(insn: int, rd: int, funct3: int, rs1: int) -> InsnInfo | None:
    if funct3 == 0:
        name = _SYSTEM.get(insn)
        return _make(insn, name) if name else None
    name = _CSR.get(funct3)
    if name is None:
        return None
    csr = insn >> 20
    if funct3 >= 5:
        # rs1 holds an immediate
        return _make(insn, name, writes=(rd,), rd=rd, imm=rs1, csr=csr)
    return _make(insn, name, reads=(rs1,), writes=(rd,), rd=rd, rs1=rs1, csr=csr)


def _decode_amo(insn: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> InsnInfo | None:
    if funct3 != 2:
        return None
    funct5, order = funct7 >> 2, _AMO_ORDER[funct7 & 0x3]
    if funct5 == _AMO_LR:
        if rs2:
            return None
        mem = MemAccess(rs1, 0, 4, is_store=False)
        return _make(insn, "lr.w" + order, reads=(rs1,), writes=(rd,), loaded=(rd,), mem=mem, rd=rd, rs1=rs1)
    mem = MemAccess(rs1, 0, 4, is_store=True)
    if funct5 == _AMO_SC:
        # rd receives the success status, not a value from memory
        return _make(insn, "sc.w" + order, reads=(rs1, rs2), writes=(rd,), mem=mem, rd=rd, rs1=rs1, rs2=rs2)
    if funct5 not in _AMO:
        return None
    return _make(
        insn, _AMO[funct5] + order, reads=(rs1, rs2), writes=(rd,), loaded=(rd,), mem=mem, rd=rd, rs1=rs1, rs2=rs2
    )


def _decode_op_fp(insn: int, rd: int, funct3: int, rs1: int, rs2: int, funct7: int) -> InsnInfo | None:
    fmt = _FP_FORMATS.get(funct7 & 0x3)
    if fmt is None:
        return None
    funct5 = funct7 >> 2
    if funct5 in _FP_ROUNDED and funct3 in _RESERVED_ROUNDING_MODES:
        return None
    if funct5 in _FP_ARITH:
        return _make(insn, f"{_FP_ARITH[funct5]}.{fmt}")
    if funct5 == 0x0B and rs2 == 0:
        return _make(insn, f"fsqrt.{fmt}")
    if funct5 == 0x08 and rs2 == 1 - (funct7 & 0x3):
        return _make(insn, f"fcvt.{fmt}.{_FP_FORMATS[rs2]}")
    names = _FP_BY_FUNCT3.get(funct5)
    if names is not None:
        if funct3 >= len(names):
            return None
        if funct5 == 0x14:
            # Comparisons write an integer register
            return _make(insn, f"{names[funct3]}.{fmt}", writes=(rd,), rd=rd)
        return _make(insn, f"{names[funct3]}.{fmt}")
    if funct5 == 0x18 and rs2 < 2:
        return _make(insn, f"fcvt.{'wu' if rs2 else 'w'}.{fmt}", writes=(rd,), rd=rd)
    if funct5 == 0x1A and rs2 < 2:
        return _make(insn, f"fcvt.{fmt}.{'wu' if rs2 else 'w'}", reads=(rs1,), rs1=rs1)
    if funct5 == 0x1C and rs2 == 0:
        if funct3 == 1:
            return _make(insn, f"fclass.{fmt}", writes=(rd,), rd=rd)
        if funct3 == 0 and fmt == "s":
            return _make(insn, "fmv.x.w", writes=(rd,), rd=rd)
    if (funct5, funct3, rs2, fmt) == (0x1E, 0, 0, "s"):
        return _make(insn, "fmv.w.x", reads=(rs1,), rs1=rs1)
    return None


# ---------------------------------------------------------------------------
# 16-bit encodings
# ---------------------------------------------------------------------------

# Zcmp s-register numbers by the 3-bit sreg field of cm.mva01s and cm.mvsa01
_SREG = S_REGS[:8]
_ZCMP_STACK = frozenset({"cm.push", "cm.pop", "cm.popretz", "cm.popret"})
# The register list that names ra and s0 to s11
_ZCMP_RLIST_ALL = 15
# cm.push and cm.pop keep sp 16-byte aligned
_ZCMP_STACK_ALIGN = 16
# Instructions that write a return address
LINKS = frozenset({"jal", "jalr", "c.jal", "c.jalr", "cm.jalt"})
_C_ALU = ("c.sub", "c.xor", "c.or", "c.and")
_ZCB_UNARY = {0: "c.zext.b", 1: "c.sext.b", 2: "c.zext.h", 3: "c.sext.h", 5: "c.not"}


def _zcmp_registers(rlist: int) -> tuple[int, ...]:
    """Return ra and the s registers a Zcmp register list names."""
    count = rlist - ZCMP_RLIST_MIN if rlist < _ZCMP_RLIST_ALL else len(S_REGS)
    return (RA, *S_REGS[:count])


def _zcmp_stack_adjust(rlist: int, spimm: int) -> int:
    """Return the bytes cm.push and cm.pop move sp by on RV32."""
    saved_bytes = XLEN_BYTES * len(_zcmp_registers(rlist))
    aligned = -(-saved_bytes // _ZCMP_STACK_ALIGN) * _ZCMP_STACK_ALIGN
    return aligned + _ZCMP_STACK_ALIGN * spimm


def _decode16(insn: int) -> InsnInfo | None:
    if insn == 0:
        return _unsupported(insn, "c.unimp")
    quadrant, funct3 = insn & QUADRANT_MASK, insn >> 13
    rd = (insn >> 7) & 0x1F
    rs2 = (insn >> 2) & 0x1F
    rdp = ((insn >> 7) & 0x7) + _C_REG_BASE
    rs2p = ((insn >> 2) & 0x7) + _C_REG_BASE
    ci_imm = sign_extend(((insn >> 7) & 0x20) | rs2, 6)
    if quadrant == 0:
        return _decode16_q0(insn, funct3, rdp, rs2p)
    if quadrant == 1:
        return _decode16_q1(insn, funct3, rd, rdp, rs2p, ci_imm)
    return _decode16_q2(insn, funct3, rd, rs2, ci_imm)


def _decode16_q0(insn: int, funct3: int, rs1p: int, rp: int) -> InsnInfo | None:
    mem = _mem(insn, funct3 >= 5)
    if funct3 == 0:
        imm = ((insn >> 7) & 0x30) | ((insn >> 1) & 0x3C0) | ((insn >> 4) & 0x4) | ((insn >> 2) & 0x8)
        if not imm:
            return None
        return _make(insn, "c.addi4spn", reads=(SP,), writes=(rp,), rd=rp, rs1=SP, imm=imm)
    if funct3 == 4:
        form = (insn >> 10) & 0x7
        bit6 = (insn >> 6) & 0x1
        if form == 0 or form == 1:
            name = "c.lbu" if form == 0 else "c.lh" if bit6 else "c.lhu"
            return _make(insn, name, reads=(rs1p,), writes=(rp,), loaded=(rp,), mem=_mem(insn, False), rd=rp, rs1=rs1p)
        if form == 2 or (form == 3 and not bit6):
            name = "c.sb" if form == 2 else "c.sh"
            return _make(insn, name, reads=(rs1p, rp), mem=_mem(insn, True), rs1=rs1p, rs2=rp)
        return None
    assert mem is not None
    if funct3 == 2:
        return _make(insn, "c.lw", reads=(rs1p,), writes=(rp,), loaded=(rp,), mem=mem, rd=rp, rs1=rs1p, imm=mem.offset)
    if funct3 == 6:
        return _make(insn, "c.sw", reads=(rs1p, rp), mem=mem, rs1=rs1p, rs2=rp, imm=mem.offset)
    # c.fld, c.flw, c.fsd, c.fsw
    name = {1: "c.fld", 3: "c.flw", 5: "c.fsd", 7: "c.fsw"}[funct3]
    return _make(insn, name, reads=(rs1p,), mem=mem, rs1=rs1p, imm=mem.offset)


def _decode16_q1(insn: int, funct3: int, rd: int, rdp: int, rs2p: int, ci_imm: int) -> InsnInfo | None:
    if funct3 == 0:
        # rd 0 is c.nop, which objdump prints as c.addi zero,0
        return _make(insn, "c.addi", reads=(rd,), writes=(rd,), rd=rd, rs1=rd, imm=ci_imm)
    if funct3 == 1:
        return _make(insn, "c.jal", writes=(RA,), rd=RA, imm=cj_imm(insn))
    if funct3 == 5:
        return _make(insn, "c.j", imm=cj_imm(insn))
    if funct3 == 2:
        return _make(insn, "c.li", writes=(rd,), rd=rd, rs1=0, imm=ci_imm)
    if funct3 == 3:
        if rd == SP:
            imm = (
                ((insn >> 3) & 0x200)
                | ((insn >> 2) & 0x10)
                | ((insn << 1) & 0x40)
                | ((insn << 4) & 0x180)
                | ((insn << 3) & 0x20)
            )
            if not imm:
                return None
            return _make(insn, "c.addi16sp", reads=(SP,), writes=(SP,), rd=SP, rs1=SP, imm=sign_extend(imm, 10))
        if not ci_imm:
            return None
        return _make(insn, "c.lui", writes=(rd,), rd=rd, imm=ci_imm << 12)
    if funct3 == 4:
        return _decode16_q1_alu(insn, rdp, rs2p, ci_imm)
    return _make(insn, "c.beqz" if funct3 == 6 else "c.bnez", reads=(rdp,), rs1=rdp, imm=cb_imm(insn))


def _decode16_q1_alu(insn: int, rdp: int, rs2p: int, ci_imm: int) -> InsnInfo | None:
    funct2 = (insn >> 10) & 0x3
    if funct2 < 2:
        # c.srli and c.srai. RV32 shift amounts are five bits, so bit 12 set is reserved.
        if insn & 0x1000:
            return None
        return _make(
            insn, "c.srai" if funct2 else "c.srli", reads=(rdp,), writes=(rdp,), rd=rdp, rs1=rdp, imm=ci_imm & 0x1F
        )
    if funct2 == 2:
        return _make(insn, "c.andi", reads=(rdp,), writes=(rdp,), rd=rdp, rs1=rdp, imm=ci_imm)
    op = (insn >> 5) & 0x3
    if not insn & 0x1000:
        return _make(insn, _C_ALU[op], reads=(rdp, rs2p), writes=(rdp,), rd=rdp, rs1=rdp, rs2=rs2p)
    if op == 2:
        return _make(insn, "c.mul", reads=(rdp, rs2p), writes=(rdp,), rd=rdp, rs1=rdp, rs2=rs2p)
    name = _ZCB_UNARY.get((insn >> 2) & 0x7) if op == 3 else None
    if name is None:
        return None
    return _make(insn, name, reads=(rdp,), writes=(rdp,), rd=rdp, rs1=rdp)


def _decode16_q2(insn: int, funct3: int, rd: int, rs2: int, ci_imm: int) -> InsnInfo | None:
    if funct3 == 0:
        if insn & 0x1000:
            return None
        return _make(insn, "c.slli", reads=(rd,), writes=(rd,), rd=rd, rs1=rd, imm=ci_imm & 0x1F)
    if funct3 in (1, 3, 7):
        # c.fldsp, c.flwsp, c.fswsp
        name = {1: "c.fldsp", 3: "c.flwsp", 7: "c.fswsp"}[funct3]
        mem = _mem(insn, funct3 == 7)
        assert mem is not None
        return _make(insn, name, reads=(SP,), mem=mem, rs1=SP, imm=mem.offset)
    if funct3 == 2:
        if rd == 0:
            return None
        mem = _mem(insn, False)
        assert mem is not None
        return _make(insn, "c.lwsp", reads=(SP,), writes=(rd,), loaded=(rd,), mem=mem, rd=rd, rs1=SP, imm=mem.offset)
    if funct3 == 6:
        mem = _mem(insn, True)
        assert mem is not None
        return _make(insn, "c.swsp", reads=(SP, rs2), mem=mem, rs1=SP, rs2=rs2, imm=mem.offset)
    if funct3 == 4:
        if not insn & 0x1000:
            if rs2:
                return _make(insn, "c.mv", reads=(rs2,), writes=(rd,), rd=rd, rs1=0, rs2=rs2)
            return _make(insn, "c.jr", reads=(rd,), rs1=rd) if rd else None
        if rs2:
            return _make(insn, "c.add", reads=(rd, rs2), writes=(rd,), rd=rd, rs1=rd, rs2=rs2)
        if rd:
            return _make(insn, "c.jalr", reads=(rd,), writes=(RA,), rd=RA, rs1=rd)
        return _make(insn, "c.ebreak")
    return _decode_zcm(insn)


def _decode_zcm(insn: int) -> InsnInfo | None:
    """Zcmp and Zcmt, in the c.fsdsp slot (quadrant 2, funct3 5)."""
    name = zcm_mnemonic(insn)
    if name is None:
        return None
    if name in _ZCMP_STACK:
        rlist = (insn >> 4) & 0xF
        saved = _zcmp_registers(rlist)
        adjust = _zcmp_stack_adjust(rlist, (insn >> 2) & 0x3)
        if name == "cm.push":
            return _make(insn, name, reads=(*saved, SP), writes=(SP,), rs1=SP, imm=-adjust)
        writes = (*saved, SP, A0) if name == "cm.popretz" else (*saved, SP)
        return _make(insn, name, reads=(SP,), writes=writes, loaded=saved, rs1=SP, imm=adjust)
    if name in ("cm.mvsa01", "cm.mva01s"):
        r1s, r2s = _SREG[(insn >> 7) & 0x7], _SREG[(insn >> 2) & 0x7]
        if name == "cm.mvsa01":
            return _make(insn, name, reads=(A0, A1), writes=(r1s, r2s), rs1=r1s, rs2=r2s)
        return _make(insn, name, reads=(r1s, r2s), writes=(A0, A1), rs1=r1s, rs2=r2s)
    # cm.jt and cm.jalt
    index = (insn >> 2) & 0xFF
    if name == "cm.jt":
        return _make(insn, name, imm=index)
    return _make(insn, name, writes=(RA,), rd=RA, imm=index)


# ---------------------------------------------------------------------------
# Forward evaluation
# ---------------------------------------------------------------------------

_Value = Callable[[int | None], int]
_Operation = Callable[[InsnInfo, int, int, _Value], int]


def _signed(value: int) -> int:
    return sign_extend(value, _XLEN_BITS)


def _imm(info: InsnInfo) -> int:
    assert info.imm is not None
    return info.imm


def _rotate_right(value: int, amount: int) -> int:
    amount &= _SHAMT_MASK
    return ((value >> amount) | (value << (_XLEN_BITS - amount))) & MASK32


def _ones(value: int) -> int:
    return bin(value & MASK32).count("1")


def _leading_zeros(value: int) -> int:
    return _XLEN_BITS - (value & MASK32).bit_length()


def _trailing_zeros(value: int) -> int:
    value &= MASK32
    return _XLEN_BITS if not value else (value & -value).bit_length() - 1


def _orc_b(value: int) -> int:
    return sum(0xFF << shift for shift in range(0, 32, 8) if (value >> shift) & 0xFF)


def _rev8(value: int) -> int:
    return int.from_bytes((value & MASK32).to_bytes(XLEN_BYTES, "little"), "big")


def _rr(function: Callable[[int, int], int]) -> _Operation:
    """Return the operation on the values of rs1 and rs2."""
    return lambda info, reg, pc, value: function(value(info.rs1), value(info.rs2))


def _ri(function: Callable[[int, int], int]) -> _Operation:
    """Return the operation on the value of rs1 and the immediate."""
    return lambda info, reg, pc, value: function(value(info.rs1), _imm(info))


def _r(function: Callable[[int], int]) -> _Operation:
    """Return the operation on the value of rs1."""
    return lambda info, reg, pc, value: function(value(info.rs1))


def _link(info: InsnInfo, reg: int, pc: int, value: _Value) -> int:
    return pc + info.size


def _zcmp_stack(info: InsnInfo, reg: int, pc: int, value: _Value) -> int:
    return 0 if reg == A0 else value(SP) + _imm(info)


def _mva01s(info: InsnInfo, reg: int, pc: int, value: _Value) -> int:
    return value(info.rs1 if reg == A0 else info.rs2)


def _mvsa01(info: InsnInfo, reg: int, pc: int, value: _Value) -> int:
    return value(A0 if reg == info.rs1 else A1)


def _add(a: int, b: int) -> int:
    return a + b


def _shl(a: int, b: int) -> int:
    return a << (b & _SHAMT_MASK)


def _shr(a: int, b: int) -> int:
    return a >> (b & _SHAMT_MASK)


def _sar(a: int, b: int) -> int:
    return _signed(a) >> (b & _SHAMT_MASK)


def _sh_add(shift: int) -> Callable[[int, int], int]:
    return lambda a, b: (a << shift) + b


_EVALUATE: dict[str, _Operation] = {
    "lui": lambda info, reg, pc, value: _imm(info),
    "c.lui": lambda info, reg, pc, value: _imm(info),
    "auipc": lambda info, reg, pc, value: pc + _imm(info),
    "c.li": _ri(_add),
    "addi": _ri(_add),
    "c.addi": _ri(_add),
    "c.addi16sp": _ri(_add),
    "c.addi4spn": _ri(_add),
    "c.mv": _rr(_add),
    "add": _rr(_add),
    "c.add": _rr(_add),
    "sub": _rr(lambda a, b: a - b),
    "c.sub": _rr(lambda a, b: a - b),
    "sh1add": _rr(_sh_add(1)),
    "sh2add": _rr(_sh_add(2)),
    "sh3add": _rr(_sh_add(3)),
    "and": _rr(lambda a, b: a & b),
    "c.and": _rr(lambda a, b: a & b),
    "or": _rr(lambda a, b: a | b),
    "c.or": _rr(lambda a, b: a | b),
    "xor": _rr(lambda a, b: a ^ b),
    "c.xor": _rr(lambda a, b: a ^ b),
    "andn": _rr(lambda a, b: a & ~b),
    "orn": _rr(lambda a, b: a | ~b),
    "xnor": _rr(lambda a, b: ~(a ^ b)),
    "andi": _ri(lambda a, b: a & b),
    "c.andi": _ri(lambda a, b: a & b),
    "ori": _ri(lambda a, b: a | b),
    "xori": _ri(lambda a, b: a ^ b),
    "sll": _rr(_shl),
    "srl": _rr(_shr),
    "sra": _rr(_sar),
    "slli": _ri(_shl),
    "c.slli": _ri(_shl),
    "srli": _ri(_shr),
    "c.srli": _ri(_shr),
    "srai": _ri(_sar),
    "c.srai": _ri(_sar),
    "slt": _rr(lambda a, b: int(_signed(a) < _signed(b))),
    "sltu": _rr(lambda a, b: int(a < b)),
    "slti": _ri(lambda a, b: int(_signed(a) < b)),
    "sltiu": _ri(lambda a, b: int(a < (b & MASK32))),
    "min": _rr(lambda a, b: min(_signed(a), _signed(b))),
    "max": _rr(lambda a, b: max(_signed(a), _signed(b))),
    "minu": _rr(min),
    "maxu": _rr(max),
    "mul": _rr(lambda a, b: a * b),
    "c.mul": _rr(lambda a, b: a * b),
    "zext.h": _r(lambda a: a & 0xFFFF),
    "c.zext.h": _r(lambda a: a & 0xFFFF),
    "c.zext.b": _r(lambda a: a & 0xFF),
    "sext.b": _r(lambda a: sign_extend(a, 8)),
    "c.sext.b": _r(lambda a: sign_extend(a, 8)),
    "sext.h": _r(lambda a: sign_extend(a, 16)),
    "c.sext.h": _r(lambda a: sign_extend(a, 16)),
    "c.not": _r(lambda a: ~a),
    "clz": _r(_leading_zeros),
    "ctz": _r(_trailing_zeros),
    "cpop": _r(_ones),
    "orc.b": _r(_orc_b),
    "rev8": _r(_rev8),
    "rol": _rr(lambda a, b: _rotate_right(a, -b)),
    "ror": _rr(_rotate_right),
    "rori": _ri(_rotate_right),
    "bset": _rr(lambda a, b: a | (1 << (b & _SHAMT_MASK))),
    "bseti": _ri(lambda a, b: a | (1 << b)),
    "bclr": _rr(lambda a, b: a & ~(1 << (b & _SHAMT_MASK))),
    "bclri": _ri(lambda a, b: a & ~(1 << b)),
    "binv": _rr(lambda a, b: a ^ (1 << (b & _SHAMT_MASK))),
    "binvi": _ri(lambda a, b: a ^ (1 << b)),
    "bext": _rr(lambda a, b: (a >> (b & _SHAMT_MASK)) & 1),
    "bexti": _ri(lambda a, b: (a >> b) & 1),
    "czero.eqz": _rr(lambda a, b: 0 if b == 0 else a),
    "czero.nez": _rr(lambda a, b: a if b == 0 else 0),
    "jal": _link,
    "jalr": _link,
    "c.jal": _link,
    "c.jalr": _link,
    "cm.jalt": _link,
    "cm.push": _zcmp_stack,
    "cm.pop": _zcmp_stack,
    "cm.popret": _zcmp_stack,
    "cm.popretz": _zcmp_stack,
    "cm.mva01s": _mva01s,
    "cm.mvsa01": _mvsa01,
}


# ---------------------------------------------------------------------------
# Instruction text
# ---------------------------------------------------------------------------

# Compressed forms objdump and GDB print as the base instruction
_C_BASE = {
    "c.addi": "addi", "c.addi16sp": "addi", "c.addi4spn": "addi", "c.li": "addi", "c.lui": "lui",
    "c.add": "add", "c.sub": "sub", "c.xor": "xor", "c.or": "or", "c.and": "and", "c.mul": "mul",
    "c.slli": "slli", "c.srli": "srli", "c.srai": "srai", "c.andi": "andi",
    "c.lw": "lw", "c.lwsp": "lw", "c.lbu": "lbu", "c.lh": "lh", "c.lhu": "lhu",
    "c.sw": "sw", "c.swsp": "sw", "c.sb": "sb", "c.sh": "sh",
    "c.j": "jal", "c.jal": "jal", "c.jr": "jalr", "c.jalr": "jalr",
    "c.zext.b": "zext.b", "c.sext.b": "sext.b", "c.zext.h": "zext.h", "c.sext.h": "sext.h",
}  # fmt: skip
# The CSRs objdump names. It prints the others, such as the ESP custom ones, as numbers.
_CSR_NAMES = {
    0x300: "mstatus", 0x301: "misa", 0x304: "mie", 0x305: "mtvec", 0x306: "mcounteren", 0x310: "mstatush",
    0x340: "mscratch", 0x341: "mepc", 0x342: "mcause", 0x343: "mtval", 0x344: "mip",
    0xF11: "mvendorid", 0xF12: "marchid", 0xF13: "mimpid", 0xF14: "mhartid",
    0xB00: "mcycle", 0xB02: "minstret", 0xB80: "mcycleh", 0xB82: "minstreth",
    0x7A0: "tselect", 0x7A1: "tdata1", 0x7A2: "tdata2", 0x7A3: "tdata3", 0x7B0: "dcsr", 0x7B1: "dpc",
    0x7B2: "dscratch0", 0x7B3: "dscratch1",
}  # fmt: skip
# The custom CSR ranges of the privileged spec, which objdump never names
_CSR_CUSTOM = ((0x7C0, 0x800), (0xBC0, 0xC00), (0xFC0, 0x1000), (0x800, 0x900), (0xCC0, 0xD00))
_CSR_WRITE_ALIASES = {
    "csrrw": "csrw", "csrrs": "csrs", "csrrc": "csrc", "csrrwi": "csrwi", "csrrsi": "csrsi", "csrrci": "csrci",
}  # fmt: skip
_LOAD_NAMES = frozenset(_LOADS.values())
_STORE_NAMES = frozenset(_STORES.values())
_SHIFT_NAMES = frozenset({"slli", "srli", "srai"})
_UNARY_NAMES = frozenset({"zext.b", "sext.b", "zext.h", "sext.h"})
_OP_IMM_NAMES = frozenset({"slti", "sltiu", "xori", "ori", "andi"})
_OP_NAMES = frozenset(
    {"add", "sub", "sll", "slt", "sltu", "xor", "srl", "sra", "or", "and", "mul", "mulh", "mulhsu", "mulhu"}
)


def insn_text(info: InsnInfo, pc: int) -> str | None:
    """Return the instruction at ``pc`` as objdump prints it with aliases, ``None`` for the forms not covered."""
    if not info.supported:
        return None
    name = _C_BASE.get(info.mnemonic, info.mnemonic)
    rd = _reg(info.rd)
    rs1 = _reg(info.rs1)
    rs2 = _reg(info.rs2)
    imm = info.imm
    if name in _LOAD_NAMES and info.mem is not None:
        return f"{name} {rd},{info.mem.offset}({_reg(info.mem.base)})"
    if name in _STORE_NAMES and info.mem is not None:
        return f"{name} {rs2},{info.mem.offset}({_reg(info.mem.base)})"
    if name in ("lui", "auipc") and imm is not None:
        return f"{name} {rd},0x{(imm >> 12) & 0xFFFFF:x}"
    if name == "c.mv":
        return f"mv {rd},{rs2}"
    if name == "addi" and imm is not None:
        if info.rd == 0 and info.rs1 == 0 and imm == 0:
            return "nop"
        if info.rs1 == 0:
            return f"li {rd},{imm}"
        return f"mv {rd},{rs1}" if imm == 0 else f"addi {rd},{rs1},{imm}"
    if name in _OP_IMM_NAMES and imm is not None:
        alias = {("sltiu", 1): "seqz", ("xori", -1): "not", ("andi", 255): "zext.b"}.get((name, imm))
        return f"{alias} {rd},{rs1}" if alias else f"{name} {rd},{rs1},{imm}"
    if name in _SHIFT_NAMES and imm is not None:
        return f"{name} {rd},{rs1},0x{imm:x}"
    if name in _UNARY_NAMES or name == "c.not":
        return f"{'not' if name == 'c.not' else name} {rd},{rs1}"
    if name in _OP_NAMES:
        return _op_text(name, info)
    if name == "jal" and imm is not None:
        target = (pc + imm) & MASK32
        if info.rd in (None, 0):
            return f"j 0x{target:x}"
        return f"jal 0x{target:x}" if info.rd == RA else f"jal {rd},0x{target:x}"
    if name == "jalr":
        return _jalr_text(info)
    if info.csr is not None:
        return _csr_text(name, info)
    return None


def _reg(reg: int | None) -> str:
    return reg_name(reg or 0)


def _op_text(name: str, info: InsnInfo) -> str:
    rd, rs1, rs2 = _reg(info.rd), _reg(info.rs1), _reg(info.rs2)
    if name == "sub" and not info.rs1:
        return f"neg {rd},{rs2}"
    if name == "sltu" and not info.rs1:
        return f"snez {rd},{rs2}"
    if name == "slt" and not info.rs2:
        return f"sltz {rd},{rs1}"
    if name == "slt" and not info.rs1:
        return f"sgtz {rd},{rs2}"
    return f"{name} {rd},{rs1},{rs2}"


def _jalr_text(info: InsnInfo) -> str:
    rs1, imm = _reg(info.rs1), info.imm or 0
    if not info.rd:
        if info.rs1 == RA and not imm:
            return "ret"
        return f"jr {rs1}" if not imm else f"jr {imm}({rs1})"
    if info.rd == RA:
        return f"jalr {rs1}" if not imm else f"jalr {imm}({rs1})"
    return f"jalr {_reg(info.rd)},{imm}({rs1})"


def _csr_text(name: str, info: InsnInfo) -> str | None:
    csr = info.csr
    assert csr is not None
    if csr in _CSR_NAMES:
        csr_name = _CSR_NAMES[csr]
    elif any(start <= csr < end for start, end in _CSR_CUSTOM):
        csr_name = f"0x{csr:x}"
    else:
        # A standard CSR objdump has a name for that this table lacks, or an alias
        return None
    source = str(info.imm) if name.endswith("i") else _reg(info.rs1)
    if name == "csrrs" and not info.rs1:
        return f"csrr {_reg(info.rd)},{csr_name}"
    if not info.rd:
        return f"{_CSR_WRITE_ALIASES[name]} {csr_name},{source}"
    return f"{name} {_reg(info.rd)},{csr_name},{source}"


__all__ = ["LINKS", "InsnInfo", "MemAccess", "decode_at", "decode_insn", "evaluate", "insn_text", "sources"]
