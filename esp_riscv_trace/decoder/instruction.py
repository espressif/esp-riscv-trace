# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RV32I, RVC, Zcmp and Zcmt instruction decoding for control-flow reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.events import InsnClass, LinkKind

# Program counters are 32 bits wide
PC_MASK = 0xFFFFFFFF


class InsnError(ValueError):
    """Raised when the input byte slice is too short, or when the first
    parcel encodes an instruction wider than 32 bits (64/80+), which
    is not part of RV32IMAFC and is not supported by this decoder.
    """


@dataclass(frozen=True)
class DecodedInsn:
    """One decoded instruction as needed by the trace reconstructor."""

    length: int  # 2 or 4 bytes
    cls: InsnClass
    direct_target: int | None  # absolute byte PC for BRANCH and JUMP
    link: LinkKind = LinkKind.NONE


def length(first_halfword: int) -> int:
    """Return the encoded instruction length in bytes.

    Low two bits of the first parcel decide the length. ``0b11`` is a
    32-bit instruction, anything else is a 16-bit compressed instruction.
    The 48-bit prefix (bits 5..0 equal 0b011111) is a 32-bit instruction
    too: Espressif harts implement no 48-bit encodings, and the PIE SIMD
    extension of the ESP32-S31 places its 32-bit ``esp.*`` instructions
    there. 64-bit and longer encodings are rejected because no Espressif
    hart generates them and the parser has no way to walk past one safely.
    """
    low = first_halfword & 0x3
    if low != 0x3:
        return 2
    if (first_halfword & 0x3F) == 0x3F:
        # bits 6..0 equal 0b0111111 or 0b1111111 flag a 64-bit or longer encoding
        raise InsnError(f"encoding wider than 32 bits (halfword 0x{first_halfword:04x}) is unsupported")
    return 4


def decode(pc: int, insn_bytes: bytes) -> DecodedInsn:
    """Decode the instruction starting at ``pc``.

    ``insn_bytes`` must be at least ``length(first_halfword)`` bytes.
    Bytes are little-endian, matching how RISC-V stores instructions in
    memory.
    """
    if len(insn_bytes) < 2:
        raise InsnError(f"instruction at 0x{pc:08x} needs at least 2 bytes, got {len(insn_bytes)}")
    hw0 = insn_bytes[0] | (insn_bytes[1] << 8)
    ilen = length(hw0)
    if len(insn_bytes) < ilen:
        raise InsnError(f"instruction at 0x{pc:08x} needs {ilen} bytes, got {len(insn_bytes)}")
    if ilen == 2:
        return _decode_rvc(pc, hw0)
    insn = hw0 | (insn_bytes[2] << 16) | (insn_bytes[3] << 24)
    return _decode_rv32i(pc, insn)


def _sign_extend(value: int, bits: int) -> int:
    sign = 1 << (bits - 1)
    return (value & ((1 << bits) - 1)) - ((value & sign) << 1)


def _b_imm(insn: int) -> int:
    """B-type immediate (BEQ, BNE, BLT, BGE, BLTU, BGEU)."""
    imm = ((insn >> 31) & 1) << 12
    imm |= ((insn >> 25) & 0x3F) << 5
    imm |= ((insn >> 8) & 0xF) << 1
    imm |= ((insn >> 7) & 1) << 11
    return _sign_extend(imm, 13)


def _j_imm(insn: int) -> int:
    """J-type immediate (JAL)."""
    imm = ((insn >> 31) & 1) << 20
    imm |= ((insn >> 21) & 0x3FF) << 1
    imm |= ((insn >> 20) & 1) << 11
    imm |= ((insn >> 12) & 0xFF) << 12
    return _sign_extend(imm, 21)


def _cb_imm(insn: int) -> int:
    """CB-type immediate (C.BEQZ, C.BNEZ)."""
    imm = ((insn >> 12) & 1) << 8
    imm |= ((insn >> 10) & 0x3) << 3
    imm |= ((insn >> 5) & 0x3) << 6
    imm |= ((insn >> 3) & 0x3) << 1
    imm |= ((insn >> 2) & 1) << 5
    return _sign_extend(imm, 9)


def _cj_imm(insn: int) -> int:
    """CJ-type immediate (C.J, C.JAL)."""
    imm = ((insn >> 12) & 1) << 11
    imm |= ((insn >> 11) & 1) << 4
    imm |= ((insn >> 9) & 0x3) << 8
    imm |= ((insn >> 8) & 1) << 10
    imm |= ((insn >> 7) & 1) << 6
    imm |= ((insn >> 6) & 1) << 7
    imm |= ((insn >> 3) & 0x7) << 1
    imm |= ((insn >> 2) & 1) << 5
    return _sign_extend(imm, 12)


_OPCODE_BRANCH = 0x63
_OPCODE_JAL = 0x6F
_OPCODE_JALR = 0x67
_OPCODE_SYSTEM = 0x73

# Valid branch funct3 values.
_BRANCH_FUNCT3 = frozenset({0b000, 0b001, 0b100, 0b101, 0b110, 0b111})

# ra and t0
_LINK_REGS = frozenset({1, 5})
_RA = 1

# Zcmp cm.popretz and cm.popret in bits 12..8
_CM_POPRETZ = 0b11100
_CM_POPRET = 0b11110
# Zcmp register lists below {ra} are reserved
_CM_RLIST_RA = 4
# Zcmt table indexes from this one are cm.jalt
_CM_JALT_FIRST_INDEX = 32


def _jump_link(rd: int, rs1: int | None) -> LinkKind:
    """Return-address stack hint of a jump that writes ``rd`` and, when indirect, jumps through ``rs1``."""
    writes_link = rd in _LINK_REGS
    reads_link = rs1 in _LINK_REGS
    if writes_link and reads_link:
        return LinkKind.CALL if rd == rs1 else LinkKind.COROUTINE
    if writes_link:
        return LinkKind.CALL
    if reads_link:
        return LinkKind.RETURN
    return LinkKind.JUMP


def _decode_rv32i(pc: int, insn: int) -> DecodedInsn:
    opcode = insn & 0x7F
    rd = (insn >> 7) & 0x1F
    if opcode == _OPCODE_BRANCH:
        funct3 = (insn >> 12) & 0x7
        if funct3 not in _BRANCH_FUNCT3:
            return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)
        target = (pc + _b_imm(insn)) & PC_MASK
        return DecodedInsn(length=4, cls=InsnClass.BRANCH, direct_target=target)
    if opcode == _OPCODE_JAL:
        target = (pc + _j_imm(insn)) & PC_MASK
        return DecodedInsn(length=4, cls=InsnClass.JUMP, direct_target=target, link=_jump_link(rd, None))
    if opcode == _OPCODE_JALR:
        # JALR requires funct3 0.
        if (insn >> 12) & 0x7 != 0:
            return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)
        rs1 = (insn >> 15) & 0x1F
        return DecodedInsn(length=4, cls=InsnClass.UNINFERABLE, direct_target=None, link=_jump_link(rd, rs1))
    if opcode == _OPCODE_SYSTEM:
        return _decode_system(insn)
    return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)


def _decode_system(insn: int) -> DecodedInsn:
    funct3 = (insn >> 12) & 0x7
    if funct3 != 0:
        # Only funct3 0 encodes control-flow system instructions.
        return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)
    if insn == 0x00000073:
        return DecodedInsn(length=4, cls=InsnClass.TRAP, direct_target=None)
    if insn == 0x00100073:
        return DecodedInsn(length=4, cls=InsnClass.TRAP, direct_target=None)
    if insn == 0x30200073:
        return DecodedInsn(length=4, cls=InsnClass.UNINFERABLE, direct_target=None, link=LinkKind.TRAP_RETURN)
    if insn == 0x10200073:
        return DecodedInsn(length=4, cls=InsnClass.UNINFERABLE, direct_target=None, link=LinkKind.TRAP_RETURN)
    return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)


def _decode_rvc(pc: int, insn: int) -> DecodedInsn:
    op = insn & 0x3
    if op == 0x1:
        return _decode_rvc_op01(pc, insn)
    if op == 0x2:
        return _decode_rvc_op10(insn)
    # Quadrant 0 does not contain control flow instructions.
    return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)


def _decode_rvc_op01(pc: int, insn: int) -> DecodedInsn:
    funct3 = (insn >> 13) & 0x7
    if funct3 == 0x1:
        # c.jal
        target = (pc + _cj_imm(insn)) & PC_MASK
        return DecodedInsn(length=2, cls=InsnClass.JUMP, direct_target=target, link=LinkKind.CALL)
    if funct3 == 0x5:
        # c.j
        target = (pc + _cj_imm(insn)) & PC_MASK
        return DecodedInsn(length=2, cls=InsnClass.JUMP, direct_target=target, link=LinkKind.JUMP)
    if funct3 == 0x6 or funct3 == 0x7:
        target = (pc + _cb_imm(insn)) & PC_MASK
        return DecodedInsn(length=2, cls=InsnClass.BRANCH, direct_target=target)
    return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)


def _decode_rvc_op10(insn: int) -> DecodedInsn:
    if (insn >> 13) & 0x7 == 0x5:
        return _decode_zcm(insn)
    funct4 = (insn >> 12) & 0xF
    rs1 = (insn >> 7) & 0x1F
    rs2 = (insn >> 2) & 0x1F
    # CR-format control flow instructions require rs2 to be zero.
    if rs2 != 0:
        return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)
    if funct4 == 0x8 and rs1 != 0:
        # c.jr
        return DecodedInsn(length=2, cls=InsnClass.UNINFERABLE, direct_target=None, link=_jump_link(0, rs1))
    if funct4 == 0x9:
        if rs1 == 0:
            return DecodedInsn(length=2, cls=InsnClass.TRAP, direct_target=None)
        # c.jalr
        return DecodedInsn(length=2, cls=InsnClass.UNINFERABLE, direct_target=None, link=_jump_link(_RA, rs1))
    return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)


def _decode_zcm(insn: int) -> DecodedInsn:
    """Decode Zcmp and Zcmt, which reuse the RV32DC c.fsdsp encoding that no supported chip implements."""
    if (insn >> 10) & 0x7 == 0:
        # cm.jt and cm.jalt jump through the table at the jvt CSR
        index = (insn >> 2) & 0xFF
        link = LinkKind.CALL if index >= _CM_JALT_FIRST_INDEX else LinkKind.JUMP
        return DecodedInsn(length=2, cls=InsnClass.UNINFERABLE, direct_target=None, link=link)
    if (insn >> 8) & 0x1F in (_CM_POPRETZ, _CM_POPRET) and (insn >> 4) & 0xF >= _CM_RLIST_RA:
        return DecodedInsn(length=2, cls=InsnClass.UNINFERABLE, direct_target=None, link=LinkKind.RETURN)
    # cm.push, cm.pop, cm.mva01s and cm.mvsa01
    return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)


__all__ = [
    "DecodedInsn",
    "InsnClass",
    "InsnError",
    "LinkKind",
    "decode",
    "length",
]
