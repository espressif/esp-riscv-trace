# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RV32 encoding rules of the control-flow decoder."""

from __future__ import annotations

RA = 1
T0 = 5

# Bits 1..0 of the first parcel, which are 0b11 for a 32-bit instruction
QUADRANT_MASK = 0x3
QUADRANT_32BIT = 0x3
OPCODE_MASK = 0x7F
OPCODE_BRANCH = 0x63
OPCODE_JALR = 0x67
OPCODE_JAL = 0x6F
OPCODE_SYSTEM = 0x73


class InsnError(ValueError):
    """Raised for a byte slice that is too short, or for an encoding wider than 32 bits."""


def length(first_halfword: int) -> int:
    """Return the encoded instruction length in bytes.

    The 48-bit prefix is a 32-bit instruction, since the PIE SIMD ``esp.*`` instructions use it. Encodings of 64 bits
    and more raise ``InsnError``, because no Espressif hart has them.
    """
    if first_halfword & QUADRANT_MASK != QUADRANT_32BIT:
        return 2
    if (first_halfword & 0x3F) == 0x3F:
        # bits 6..0 equal 0b0111111 or 0b1111111 flag a 64-bit or longer encoding
        raise InsnError(f"encoding wider than 32 bits (halfword 0x{first_halfword:04x}) is unsupported")
    return 4


def sign_extend(value: int, bits: int) -> int:
    """Return the low ``bits`` of ``value`` as a two's complement number."""
    sign = 1 << (bits - 1)
    return (value & ((1 << bits) - 1)) - ((value & sign) << 1)


def b_imm(insn: int) -> int:
    """B-type immediate (BEQ, BNE, BLT, BGE, BLTU, BGEU)."""
    imm = ((insn >> 31) & 1) << 12
    imm |= ((insn >> 25) & 0x3F) << 5
    imm |= ((insn >> 8) & 0xF) << 1
    imm |= ((insn >> 7) & 1) << 11
    return sign_extend(imm, 13)


def j_imm(insn: int) -> int:
    """J-type immediate (JAL)."""
    imm = ((insn >> 31) & 1) << 20
    imm |= ((insn >> 21) & 0x3FF) << 1
    imm |= ((insn >> 20) & 1) << 11
    imm |= ((insn >> 12) & 0xFF) << 12
    return sign_extend(imm, 21)


def cb_imm(insn: int) -> int:
    """CB-type immediate (C.BEQZ, C.BNEZ)."""
    imm = ((insn >> 12) & 1) << 8
    imm |= ((insn >> 10) & 0x3) << 3
    imm |= ((insn >> 5) & 0x3) << 6
    imm |= ((insn >> 3) & 0x3) << 1
    imm |= ((insn >> 2) & 1) << 5
    return sign_extend(imm, 9)


def cj_imm(insn: int) -> int:
    """CJ-type immediate (C.J, C.JAL)."""
    imm = ((insn >> 12) & 1) << 11
    imm |= ((insn >> 11) & 1) << 4
    imm |= ((insn >> 9) & 0x3) << 8
    imm |= ((insn >> 8) & 1) << 10
    imm |= ((insn >> 7) & 1) << 6
    imm |= ((insn >> 6) & 1) << 7
    imm |= ((insn >> 3) & 0x7) << 1
    imm |= ((insn >> 2) & 1) << 5
    return sign_extend(imm, 12)


# cm.push and the cm.pop forms by bits 12..8
_ZCMP_STACK = {0b11000: "cm.push", 0b11010: "cm.pop", 0b11100: "cm.popretz", 0b11110: "cm.popret"}
# Register lists in bits 7..4 below {ra} are reserved
ZCMP_RLIST_MIN = 4
# cm.mvsa01 and cm.mva01s by bits 6..5, when bits 12..10 are 0b011
_ZCMP_MOVES = {1: "cm.mvsa01", 3: "cm.mva01s"}
# cm.jt and cm.jalt share an index field. Indexes from this one up are cm.jalt.
ZCMT_JALT_MIN = 32


def zcm_mnemonic(insn: int) -> str | None:
    """Return the Zcmp or Zcmt mnemonic in the c.fsdsp slot of a 16-bit instruction, ``None`` when reserved."""
    stack = _ZCMP_STACK.get((insn >> 8) & 0x1F)
    if stack is not None:
        return stack if (insn >> 4) & 0xF >= ZCMP_RLIST_MIN else None
    form = (insn >> 10) & 0x7
    if form == 0b011:
        mnemonic = _ZCMP_MOVES.get((insn >> 5) & 0x3)
        # cm.mvsa01 cannot write one s register twice
        if mnemonic == "cm.mvsa01" and (insn >> 7) & 0x7 == (insn >> 2) & 0x7:
            return None
        return mnemonic
    if form == 0:
        return "cm.jalt" if (insn >> 2) & 0xFF >= ZCMT_JALT_MIN else "cm.jt"
    return None


__all__ = [
    "OPCODE_BRANCH",
    "OPCODE_JAL",
    "OPCODE_JALR",
    "OPCODE_MASK",
    "OPCODE_SYSTEM",
    "QUADRANT_32BIT",
    "QUADRANT_MASK",
    "RA",
    "T0",
    "ZCMP_RLIST_MIN",
    "ZCMT_JALT_MIN",
    "InsnError",
    "b_imm",
    "cb_imm",
    "cj_imm",
    "j_imm",
    "length",
    "sign_extend",
    "zcm_mnemonic",
]
