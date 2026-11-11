# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""RV32I and RVC control-flow classification and direct targets."""

from __future__ import annotations

import pytest
from helpers import BASE

from esp_riscv_trace.decoder.instruction import PC_MASK, InsnError, decode, length
from esp_riscv_trace.ir.events import InsnClass, LinkKind

# Encodings from riscv32-esp-elf-as with offsets relative to the instruction
_CASES = [
    ("jal_forward", 0x1000006F, 4, InsnClass.JUMP, 0x100),  # jal zero, .+0x100
    ("jal_backward", 0x801FF0EF, 4, InsnClass.JUMP, -0x800),  # jal ra, .-0x800
    ("jalr_ret", 0x00008067, 4, InsnClass.UNINFERABLE, None),  # jalr zero, 0(ra)
    ("jalr_call", 0x008780E7, 4, InsnClass.UNINFERABLE, None),  # jalr ra, 8(a5)
    ("beq", 0x00B50863, 4, InsnClass.BRANCH, 16),  # beq a0, a1, .+16
    ("bne", 0xFEB518E3, 4, InsnClass.BRANCH, -16),  # bne a0, a1, .-16
    ("blt", 0x7EB54FE3, 4, InsnClass.BRANCH, 0xFFE),  # blt a0, a1, .+0xffe
    ("bge", 0x80B55063, 4, InsnClass.BRANCH, -0x1000),  # bge a0, a1, .-0x1000
    ("bltu", 0x00B56463, 4, InsnClass.BRANCH, 8),  # bltu a0, a1, .+8
    ("bgeu", 0xFEB57CE3, 4, InsnClass.BRANCH, -8),  # bgeu a0, a1, .-8
    ("mret", 0x30200073, 4, InsnClass.UNINFERABLE, None),
    ("sret", 0x10200073, 4, InsnClass.UNINFERABLE, None),
    ("ecall", 0x00000073, 4, InsnClass.TRAP, None),
    ("ebreak", 0x00100073, 4, InsnClass.TRAP, None),
    ("wfi", 0x10500073, 4, InsnClass.NONE, None),
    ("csrrw", 0x30001073, 4, InsnClass.NONE, None),  # csrrw zero, mstatus, zero
    ("addi", 0x00150513, 4, InsnClass.NONE, None),  # addi a0, a0, 1
    ("branch_funct3_reserved", 0x00B52063, 4, InsnClass.NONE, None),  # funct3 0b010
    ("jalr_funct3_reserved", 0x00009067, 4, InsnClass.NONE, None),
    ("c_j_forward", 0xAFFD, 2, InsnClass.JUMP, 0x7FE),  # c.j .+0x7fe
    ("c_j_backward", 0xB001, 2, InsnClass.JUMP, -0x800),  # c.j .-0x800
    ("c_jal", 0x2005, 2, InsnClass.JUMP, 0x20),  # c.jal .+0x20
    ("c_beqz", 0xCD7D, 2, InsnClass.BRANCH, 0xFE),  # c.beqz a0, .+0xfe
    ("c_bnez", 0xF381, 2, InsnClass.BRANCH, -0x100),  # c.bnez a5, .-0x100
    ("c_jr", 0x8082, 2, InsnClass.UNINFERABLE, None),  # c.jr ra
    ("c_jalr", 0x9782, 2, InsnClass.UNINFERABLE, None),  # c.jalr a5
    ("c_ebreak", 0x9002, 2, InsnClass.TRAP, None),
    ("c_addi", 0x0505, 2, InsnClass.NONE, None),  # c.addi a0, 1
    ("c_nop", 0x0001, 2, InsnClass.NONE, None),
    ("c_lw", 0x4188, 2, InsnClass.NONE, None),  # c.lw a0, 0(a1)
    ("c_mv", 0x852E, 2, InsnClass.NONE, None),  # c.mv a0, a1
    ("c_jr_x0_reserved", 0x8002, 2, InsnClass.NONE, None),
    ("c_add", 0x952E, 2, InsnClass.NONE, None),  # c.add a0, a1
]


@pytest.mark.parametrize(
    ("insn", "size", "cls", "offset"),
    [c[1:] for c in _CASES],
    ids=[c[0] for c in _CASES],
)
def test_decode(insn: int, size: int, cls: InsnClass, offset: int | None) -> None:
    decoded = decode(BASE, insn.to_bytes(size, "little"))
    assert decoded.cls == cls
    assert decoded.length == size
    if offset is None:
        assert decoded.direct_target is None
    else:
        assert decoded.direct_target == BASE + offset


@pytest.mark.parametrize(
    ("pc", "insn", "size", "target"),
    [
        (0xFFFFFF00, 0x1000006F, 4, 0x00000000),  # jal zero, .+0x100
        (0x00000010, 0x80B55063, 4, 0xFFFFF010),  # bge a0, a1, .-0x1000
        (0xFFFFFFF0, 0x2005, 2, 0x00000010),  # c.jal .+0x20
        (0x00000004, 0xF381, 2, 0xFFFFFF04),  # c.bnez a5, .-0x100
    ],
)
def test_direct_target_wraps_at_pc_mask(pc: int, insn: int, size: int, target: int) -> None:
    decoded = decode(pc, insn.to_bytes(size, "little"))
    assert decoded.direct_target == target
    assert decoded.direct_target is not None
    assert decoded.direct_target <= PC_MASK


# Encodings from riscv32-esp-elf-as -march=rv32imac_zicsr_zcmp_zcmt
_LINK_CASES = [
    ("jal_ra", 0x801FF0EF, 4, InsnClass.JUMP, LinkKind.CALL),  # jal ra, .-0x800
    ("jal_t0", 0xFE3FF2EF, 4, InsnClass.JUMP, LinkKind.CALL),  # jal t0, .-0x1e
    ("jal_zero", 0x1000006F, 4, InsnClass.JUMP, LinkKind.JUMP),  # jal zero, .+0x100
    ("jalr_zero_ra", 0x00008067, 4, InsnClass.UNINFERABLE, LinkKind.RETURN),  # ret
    ("jalr_zero_t0", 0x00028067, 4, InsnClass.UNINFERABLE, LinkKind.RETURN),  # jr t0
    ("jalr_zero_a5", 0x00078067, 4, InsnClass.UNINFERABLE, LinkKind.JUMP),  # jr a5
    ("jalr_ra_a5", 0x008780E7, 4, InsnClass.UNINFERABLE, LinkKind.CALL),  # jalr ra, 8(a5)
    ("jalr_ra_ra", 0x000080E7, 4, InsnClass.UNINFERABLE, LinkKind.CALL),  # jalr ra, 0(ra)
    ("jalr_ra_t0", 0x000280E7, 4, InsnClass.UNINFERABLE, LinkKind.COROUTINE),  # jalr ra, 0(t0)
    ("jalr_t0_ra", 0x000082E7, 4, InsnClass.UNINFERABLE, LinkKind.COROUTINE),  # jalr t0, 0(ra)
    ("jalr_a0_ra", 0x00008567, 4, InsnClass.UNINFERABLE, LinkKind.RETURN),  # jalr a0, 0(ra)
    ("mret", 0x30200073, 4, InsnClass.UNINFERABLE, LinkKind.TRAP_RETURN),
    ("sret", 0x10200073, 4, InsnClass.UNINFERABLE, LinkKind.TRAP_RETURN),
    ("c_j", 0xAFFD, 2, InsnClass.JUMP, LinkKind.JUMP),
    ("c_jal", 0x2005, 2, InsnClass.JUMP, LinkKind.CALL),
    ("c_jr_ra", 0x8082, 2, InsnClass.UNINFERABLE, LinkKind.RETURN),
    ("c_jr_t0", 0x8282, 2, InsnClass.UNINFERABLE, LinkKind.RETURN),
    ("c_jr_a5", 0x8782, 2, InsnClass.UNINFERABLE, LinkKind.JUMP),
    ("c_jalr_a5", 0x9782, 2, InsnClass.UNINFERABLE, LinkKind.CALL),
    ("c_jalr_t0", 0x9282, 2, InsnClass.UNINFERABLE, LinkKind.COROUTINE),
    ("cm_popret", 0xBE42, 2, InsnClass.UNINFERABLE, LinkKind.RETURN),  # cm.popret {ra}, 16
    ("cm_popret_s1", 0xBE66, 2, InsnClass.UNINFERABLE, LinkKind.RETURN),  # cm.popret {ra, s0-s1}, 32
    ("cm_popretz", 0xBC42, 2, InsnClass.UNINFERABLE, LinkKind.RETURN),  # cm.popretz {ra}, 16
    ("cm_popretz_s11", 0xBCFE, 2, InsnClass.UNINFERABLE, LinkKind.RETURN),  # cm.popretz {ra, s0-s11}, 112
    ("cm_popret_reserved_rlist", 0xBE02, 2, InsnClass.NONE, LinkKind.NONE),
    ("cm_jt_0", 0xA002, 2, InsnClass.UNINFERABLE, LinkKind.JUMP),
    ("cm_jt_31", 0xA07E, 2, InsnClass.UNINFERABLE, LinkKind.JUMP),
    ("cm_jalt_32", 0xA082, 2, InsnClass.UNINFERABLE, LinkKind.CALL),
    ("cm_jalt_255", 0xA3FE, 2, InsnClass.UNINFERABLE, LinkKind.CALL),
    ("cm_push", 0xB842, 2, InsnClass.NONE, LinkKind.NONE),  # cm.push {ra}, -16
    ("cm_push_s11", 0xB8FE, 2, InsnClass.NONE, LinkKind.NONE),  # cm.push {ra, s0-s11}, -112
    ("cm_pop", 0xBA42, 2, InsnClass.NONE, LinkKind.NONE),  # cm.pop {ra}, 16
    ("cm_mva01s", 0xAC66, 2, InsnClass.NONE, LinkKind.NONE),  # cm.mva01s s0, s1
    ("cm_mvsa01", 0xAC26, 2, InsnClass.NONE, LinkKind.NONE),  # cm.mvsa01 s0, s1
    ("beq", 0x00B50863, 4, InsnClass.BRANCH, LinkKind.NONE),
    ("ecall", 0x00000073, 4, InsnClass.TRAP, LinkKind.NONE),
    ("addi", 0x00150513, 4, InsnClass.NONE, LinkKind.NONE),
    ("esp_vst_128_ip", 0x8311009F, 4, InsnClass.NONE, LinkKind.NONE),  # esp.vst.128.ip q0, a0, 16
]


@pytest.mark.parametrize(
    ("insn", "size", "cls", "link"),
    [c[1:] for c in _LINK_CASES],
    ids=[c[0] for c in _LINK_CASES],
)
def test_decode_link(insn: int, size: int, cls: InsnClass, link: LinkKind) -> None:
    decoded = decode(BASE, insn.to_bytes(size, "little"))
    assert (decoded.length, decoded.cls, decoded.link) == (size, cls, link)


def test_decode_ignores_trailing_bytes() -> None:
    assert decode(BASE, bytes.fromhex("0290 1300 0000")).length == 2


@pytest.mark.parametrize("halfword", [0x0000, 0x0001, 0x0002, 0x8082, 0xFFFC, 0xFFFE])
def test_length_compressed(halfword: int) -> None:
    assert length(halfword) == 2


# 0x009F and 0x021F are esp.vst.128.ip and esp.st.qacc.l.l.128.ip, PIE instructions in the 48-bit prefix space
@pytest.mark.parametrize("halfword", [0x0003, 0x0013, 0x0073, 0x006F, 0xFFFB, 0x001F, 0x009F, 0x021F, 0x005F])
def test_length_32bit(halfword: int) -> None:
    assert length(halfword) == 4


@pytest.mark.parametrize("halfword", [0x003F, 0x007F, 0x00BF, 0xFFFF])
def test_length_rejects_wide_encodings(halfword: int) -> None:
    with pytest.raises(InsnError, match="wider than 32 bits"):
        length(halfword)


@pytest.mark.parametrize(
    ("data", "fragment"),
    [
        (b"", "needs at least 2 bytes, got 0"),
        (b"\x13", "needs at least 2 bytes, got 1"),
        (b"\x13\x00", "needs 4 bytes, got 2"),
        (b"\x13\x00\x00", "needs 4 bytes, got 3"),
        (b"\x3f\x00\x00\x00\x00\x00\x00\x00", "wider than 32 bits"),
    ],
)
def test_decode_rejects(data: bytes, fragment: str) -> None:
    with pytest.raises(InsnError, match=fragment):
        decode(BASE, data)
