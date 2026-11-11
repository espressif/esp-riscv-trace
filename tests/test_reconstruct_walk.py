# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Walker stop reasons when instruction bytes cannot be followed."""

from __future__ import annotations

import pytest
from helpers import BASE, JALR_X0_RA, NOP, FakeImage, make_frame, make_sync

from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.packets import (
    ContextPacket,
    Format0Packet,
    Format1WithAddrPacket,
    Format2Packet,
)
from esp_riscv_trace.decoder.params import default_profile
from esp_riscv_trace.ir.events import InsnClass, TraceGap, TraceInstruction, TraceSync
from esp_riscv_trace.snapshot import abi

GapReason = reconstruct.GapReason

ET = default_profile("esp32c5")

# First halfword of a 64-bit encoding
_WIDE = bytes.fromhex("3f00 0000 0000 0000")
# c.j to itself
_C_J_SELF = bytes.fromhex("01a0")


def _run(packets, image: FakeImage, mode: abi.AddressMode = abi.AddressMode.FULL):
    return list(reconstruct.reconstruct(packets, image, ET, mode))


def _walk(data: bytes, start: int = BASE):
    return _run([make_sync(start, branch=1)], FakeImage(BASE, data))


def _gap(events) -> TraceGap:
    gap = events[-1]
    assert isinstance(gap, TraceGap)
    return gap


@pytest.mark.parametrize(
    ("data", "start", "pcs", "gap_pc"),
    [
        (NOP, BASE - 4, [], BASE - 4),
        (NOP, BASE + 4, [], BASE + 4),
        (NOP + NOP, BASE, [BASE, BASE + 4], BASE + 8),
        (NOP + NOP[:2], BASE, [BASE], BASE + 4),  # second halfword is past the image end
        (NOP + b"\x13", BASE, [BASE], BASE + 4),  # first halfword is past the image end
    ],
    ids=["before_image", "after_image", "walk_off_end", "split_32bit", "split_halfword"],
)
def test_pc_out_of_image_is_gap(data: bytes, start: int, pcs: list[int], gap_pc: int) -> None:
    events = _walk(data, start)

    assert isinstance(events[0], TraceSync)
    assert [e.pc for e in events if isinstance(e, TraceInstruction)] == pcs
    gap = _gap(events)
    assert gap.reason == GapReason.PC_OUT_OF_IMAGE
    assert gap.reason == "pc_out_of_image"
    assert f"pc=0x{gap_pc:08x}" in gap.detail


def test_wide_encoding_is_decode_error() -> None:
    events = _walk(NOP + _WIDE)

    assert [e.pc for e in events if isinstance(e, TraceInstruction)] == [BASE]
    gap = _gap(events)
    assert gap.reason == GapReason.DECODE_ERROR
    assert gap.reason == "decode_error"
    assert f"pc=0x{BASE + 4:08x}" in gap.detail
    assert "wider than 32 bits" in gap.detail


def test_compressed_self_jump_spins() -> None:
    events = _walk(NOP + _C_J_SELF)

    jump = events[-1]
    assert isinstance(jump, TraceInstruction)
    assert (jump.pc, jump.length, jump.next_pc) == (BASE + 4, 2, BASE + 4)
    assert not any(isinstance(e, TraceGap) for e in events)


def test_gap_resets_walker_until_next_sync() -> None:
    image = FakeImage(BASE, NOP)
    target = Format2Packet(frame=make_frame(1), notify=0, updiscon=0, raw_address=BASE >> 1, address_width_bits=32)
    events = _run([make_sync(BASE + 4, branch=1), target], image)

    assert [type(e) for e in events] == [TraceSync, TraceGap]


@pytest.mark.parametrize(
    "packet",
    [
        Format2Packet(frame=make_frame(1), notify=0, updiscon=0, raw_address=4, address_width_bits=8),
        Format1WithAddrPacket(
            frame=make_frame(1),
            branches=1,
            branch_map=0,
            branch_map_width_bits=1,
            notify=0,
            updiscon=0,
            raw_address=4,
            address_width_bits=8,
        ),
    ],
    ids=["format2", "format1"],
)
def test_delta_address_before_sync_is_not_anchored(packet) -> None:
    events = _run([packet], FakeImage(BASE, NOP), abi.AddressMode.DELTA)

    (gap,) = events
    assert isinstance(gap, TraceGap)
    assert gap.reason == GapReason.NO_PC
    assert gap.reason == "walker_not_anchored"


@pytest.mark.parametrize(
    ("packet", "reason"),
    [
        (ContextPacket(frame=make_frame(1)), GapReason.UNEXPECTED_CONTEXT),
        (Format0Packet(frame=make_frame(1)), GapReason.UNEXPECTED_F0),
    ],
    ids=["context", "format0"],
)
def test_unused_packet_kinds_are_gaps(packet, reason: GapReason) -> None:
    events = _run([make_sync(BASE, branch=1), packet], FakeImage(BASE, NOP + JALR_X0_RA))

    gap = _gap(events)
    assert gap.reason == reason
    assert gap.provenance is not None
    assert gap.provenance.packet_index == 1


def test_sync_retires_the_uninferable_jump_it_targets() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + NOP + JALR_X0_RA)
    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=1), make_sync(BASE + 12, branch=1)], image, ET, abi.AddressMode.FULL
        )
    )

    assert [(type(e).__name__, getattr(e, "pc", None)) for e in events] == [
        ("TraceSync", BASE),
        ("TraceInstruction", BASE),
        ("TraceInstruction", BASE + 4),
        ("TraceSync", BASE + 12),
        ("TraceInstruction", BASE + 12),
    ]
    ret = events[2]
    assert isinstance(ret, TraceInstruction)
    assert (ret.cls, ret.next_pc) == (InsnClass.UNINFERABLE, BASE + 12)


def test_address_the_code_cannot_reach_is_a_gap_before_it() -> None:
    # beq x0, x0, +8 at BASE + 4 and BASE + 16, a debugger moves the core from BASE + 4 to BASE + 12
    beq = bytes.fromhex("63040000")
    image = FakeImage(BASE, NOP + beq + JALR_X0_RA + NOP + beq + JALR_X0_RA + JALR_X0_RA)
    moved = Format2Packet(
        frame=make_frame(1), notify=0, updiscon=0, raw_address=(BASE + 12) >> 1, address_width_bits=32
    )
    returned = Format1WithAddrPacket(
        frame=make_frame(2),
        branches=1,
        branch_map=0,
        branch_map_width_bits=1,
        notify=0,
        updiscon=0,
        raw_address=(BASE + 8) >> 1,
        address_width_bits=32,
    )
    events = _run([make_sync(BASE, branch=1), moved, returned], image)

    assert [(type(e).__name__, getattr(e, "pc", None)) for e in events] == [
        ("TraceSync", BASE),
        ("TraceInstruction", BASE),
        ("TraceGap", None),
        ("TraceInstruction", BASE + 12),
        ("TraceInstruction", BASE + 16),
        ("TraceInstruction", BASE + 24),
    ]
    assert events[2].reason == GapReason.UNEXPLAINED_JUMP
    assert (events[4].branch_taken, events[5].next_pc) == (True, BASE + 8)


@pytest.mark.parametrize(("branch", "taken"), [(0, True), (1, False)])
def test_sync_on_branch_uses_the_sync_branch_bit(branch: int, taken: bool) -> None:
    # beq x0, x0, +8 at the sync address
    image = FakeImage(BASE, bytes.fromhex("63040000") + NOP + JALR_X0_RA + JALR_X0_RA)
    events = list(reconstruct.reconstruct([make_sync(BASE, branch=branch)], image, ET, abi.AddressMode.FULL))

    first = events[1]
    assert isinstance(first, TraceInstruction)
    assert (first.cls, first.branch_taken, first.next_pc) == (InsnClass.BRANCH, taken, BASE + 8 if taken else BASE + 4)
