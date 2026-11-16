# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RV32I, RVC, Zcmp and Zcmt instruction decoding for control-flow reconstruction."""

from __future__ import annotations

from dataclasses import dataclass

from ..ir.events import InsnClass, LinkKind
from .encoding import (
    OPCODE_BRANCH,
    OPCODE_JAL,
    OPCODE_JALR,
    OPCODE_MASK,
    OPCODE_SYSTEM,
    QUADRANT_MASK,
    RA,
    T0,
    InsnError,
    b_imm,
    cb_imm,
    cj_imm,
    j_imm,
    length,
    zcm_mnemonic,
)

# Program counters are 32 bits wide
PC_MASK = 0xFFFFFFFF


@dataclass(frozen=True)
class DecodedInsn:
    """One decoded instruction as needed by the trace reconstructor."""

    length: int  # 2 or 4 bytes
    cls: InsnClass
    direct_target: int | None  # absolute byte PC for BRANCH and JUMP
    link: LinkKind = LinkKind.NONE


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


# Valid branch funct3 values.
_BRANCH_FUNCT3 = frozenset({0b000, 0b001, 0b100, 0b101, 0b110, 0b111})

_LINK_REGS = frozenset({RA, T0})

# The Zcmp and Zcmt instructions that jump. cm.jt and cm.jalt jump through the table at the jvt CSR.
_ZCM_LINKS = {
    "cm.jt": LinkKind.JUMP,
    "cm.jalt": LinkKind.CALL,
    "cm.popret": LinkKind.RETURN,
    "cm.popretz": LinkKind.RETURN,
}


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
    opcode = insn & OPCODE_MASK
    rd = (insn >> 7) & 0x1F
    if opcode == OPCODE_BRANCH:
        funct3 = (insn >> 12) & 0x7
        if funct3 not in _BRANCH_FUNCT3:
            return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)
        target = (pc + b_imm(insn)) & PC_MASK
        return DecodedInsn(length=4, cls=InsnClass.BRANCH, direct_target=target)
    if opcode == OPCODE_JAL:
        target = (pc + j_imm(insn)) & PC_MASK
        return DecodedInsn(length=4, cls=InsnClass.JUMP, direct_target=target, link=_jump_link(rd, None))
    if opcode == OPCODE_JALR:
        # JALR requires funct3 0.
        if (insn >> 12) & 0x7 != 0:
            return DecodedInsn(length=4, cls=InsnClass.NONE, direct_target=None)
        rs1 = (insn >> 15) & 0x1F
        return DecodedInsn(length=4, cls=InsnClass.UNINFERABLE, direct_target=None, link=_jump_link(rd, rs1))
    if opcode == OPCODE_SYSTEM:
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
    op = insn & QUADRANT_MASK
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
        target = (pc + cj_imm(insn)) & PC_MASK
        return DecodedInsn(length=2, cls=InsnClass.JUMP, direct_target=target, link=LinkKind.CALL)
    if funct3 == 0x5:
        # c.j
        target = (pc + cj_imm(insn)) & PC_MASK
        return DecodedInsn(length=2, cls=InsnClass.JUMP, direct_target=target, link=LinkKind.JUMP)
    if funct3 == 0x6 or funct3 == 0x7:
        target = (pc + cb_imm(insn)) & PC_MASK
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
        return DecodedInsn(length=2, cls=InsnClass.UNINFERABLE, direct_target=None, link=_jump_link(RA, rs1))
    return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)


def _decode_zcm(insn: int) -> DecodedInsn:
    """Decode Zcmp and Zcmt, which reuse the RV32DC c.fsdsp encoding that no supported chip implements."""
    mnemonic = zcm_mnemonic(insn)
    link = _ZCM_LINKS.get(mnemonic) if mnemonic is not None else None
    if link is None:
        # cm.push, cm.pop, cm.mva01s, cm.mvsa01 and reserved forms
        return DecodedInsn(length=2, cls=InsnClass.NONE, direct_target=None)
    return DecodedInsn(length=2, cls=InsnClass.UNINFERABLE, direct_target=None, link=link)


__all__ = [
    "DecodedInsn",
    "InsnClass",
    "InsnError",
    "LinkKind",
    "decode",
    "length",
]
