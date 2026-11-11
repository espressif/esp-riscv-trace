# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Reconstruction semantics for Format 1/2 notify and updiscon fields."""

from __future__ import annotations

from helpers import BASE, JALR_X0_RA, NOP, FakeImage, make_frame, make_sync

from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.packets import (
    Format1WithAddrPacket,
    Format2Packet,
    SupportPacket,
)
from esp_riscv_trace.decoder.params import default_profile
from esp_riscv_trace.ir.events import InsnClass, TraceInstruction
from esp_riscv_trace.snapshot import abi

# beq x0, x0, 0. The trace branch map, rather than register evaluation,
# determines whether the reconstructed execution takes it.
_BRANCH_SELF = bytes.fromhex("63000000")


def _format2(pc: int, *, notify: int, updiscon: int, index: int) -> Format2Packet:
    return Format2Packet(
        frame=make_frame(index),
        notify=notify,
        updiscon=updiscon,
        raw_address=pc >> 1,
        address_width_bits=32,
    )


def _instruction_events(events) -> list[TraceInstruction]:
    return [event for event in events if isinstance(event, TraceInstruction)]


def test_format2_notify_is_not_used_as_an_uninferable_target() -> None:
    # Sync walks through the first NOP and waits at JALR. The notification
    # names an already reconstructed instruction and must not redirect JALR.
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + NOP + JALR_X0_RA)
    profile = default_profile("esp32c5")
    notification = _format2(BASE, notify=1, updiscon=1, index=1)
    real_target = _format2(BASE + 8, notify=0, updiscon=0, index=2)

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=1), notification, real_target],
            image,
            profile,
            abi.AddressMode.FULL,
        )
    )

    instructions = _instruction_events(events)
    jump = next(event for event in instructions if event.cls == InsnClass.UNINFERABLE)
    assert jump.pc == BASE + 4
    assert jump.next_pc == BASE + 8


def test_notify_uses_transmitted_delta_msb_not_resolved_pc_msb() -> None:
    image = FakeImage(BASE, NOP + NOP + NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    profile = default_profile("esp32c5")
    # The first packet reports BASE relative to the Sync at BASE+8. An 8-byte
    # backward delta is -4 halfwords, encoded as 0xfc in the 8-bit address
    # field. Its transmitted MSB is therefore one, so notify=0 marks it as a
    # notification even though the resolved PC itself has MSB zero.
    notification = Format2Packet(
        frame=make_frame(1),
        notify=0,
        updiscon=0,
        raw_address=0xFC,
        address_width_bits=8,
    )
    # The notification changed the differential base to BASE. The real JALR
    # target at BASE+16 is therefore encoded as +8 halfwords.
    real_target = Format2Packet(
        frame=make_frame(2),
        notify=0,
        updiscon=0,
        raw_address=8,
        address_width_bits=8,
    )

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE + 8, branch=1), notification, real_target],
            image,
            profile,
            abi.AddressMode.DELTA,
        )
    )

    instructions = _instruction_events(events)
    jump = next(event for event in instructions if event.cls == InsnClass.UNINFERABLE)
    assert jump.pc == BASE + 12
    assert jump.next_pc == BASE + 16


def test_format2_updiscon_stops_at_instruction_before_format3() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + NOP + JALR_X0_RA)
    profile = default_profile("esp32c5")
    # The address MSB and notify are zero. updiscon differs from notify, so
    # the target at BASE+8 is also the final instruction before Format 3.
    boundary = _format2(BASE + 8, notify=0, updiscon=1, index=1)
    following_format3 = SupportPacket(
        frame=make_frame(2),
        enable=0,
        qual_status=1,
        encoder_mode=0,
        ioptions=0,
        full_address_mode=False,
    )

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=1), boundary, following_format3],
            image,
            profile,
            abi.AddressMode.FULL,
        )
    )

    assert [event.pc for event in _instruction_events(events)] == [BASE, BASE + 4, BASE + 8]


def test_format1_notify_stops_at_reported_instruction() -> None:
    image = FakeImage(BASE, NOP + _BRANCH_SELF + NOP + NOP + JALR_X0_RA)
    profile = default_profile("esp32c5")
    notification = Format1WithAddrPacket(
        frame=make_frame(1),
        branches=1,
        branch_map=1,  # branch at BASE+4 was not taken
        branch_map_width_bits=1,
        notify=1,  # differs from the zero MSB of the reported address
        updiscon=1,
        raw_address=(BASE + 8) >> 1,
        address_width_bits=32,
    )

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=1), notification],
            image,
            profile,
            abi.AddressMode.FULL,
        )
    )

    assert [event.pc for event in _instruction_events(events)] == [BASE, BASE + 4, BASE + 8]


def test_ordinary_address_pauses_when_first_reached_inferably() -> None:
    image = FakeImage(BASE, NOP + _BRANCH_SELF + NOP + NOP + JALR_X0_RA)
    profile = default_profile("esp32c5")
    packet = Format1WithAddrPacket(
        frame=make_frame(1),
        branches=1,
        branch_map=1,
        branch_map_width_bits=1,
        notify=0,
        updiscon=0,
        raw_address=(BASE + 8) >> 1,
        address_width_bits=32,
    )

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=1), packet],
            image,
            profile,
            abi.AddressMode.FULL,
        )
    )

    # With neither encoded flag asserted, BASE+8 might instead be the target
    # of a later updiscon. Pause at the first occurrence until the next packet
    # disambiguates it, without speculating into BASE+12 yet.
    assert [event.pc for event in _instruction_events(events)] == [BASE, BASE + 4, BASE + 8]


def test_following_address_packet_replays_ambiguous_reported_address() -> None:
    image = FakeImage(
        BASE,
        NOP + _BRANCH_SELF + NOP + NOP + JALR_X0_RA + NOP + NOP + JALR_X0_RA,
    )
    profile = default_profile("esp32c5")
    first_report = Format1WithAddrPacket(
        frame=make_frame(1),
        branches=1,
        branch_map=1,
        branch_map_width_bits=1,
        notify=0,
        updiscon=0,
        raw_address=(BASE + 8) >> 1,
        address_width_bits=32,
    )
    second_report = _format2(BASE + 24, notify=0, updiscon=0, index=2)

    events = list(
        reconstruct.reconstruct(
            [make_sync(BASE, branch=1), first_report, second_report],
            image,
            profile,
            abi.AddressMode.FULL,
        )
    )

    jumps = [event for event in _instruction_events(events) if event.cls == InsnClass.UNINFERABLE]
    assert [(event.pc, event.next_pc) for event in jumps] == [
        (BASE + 16, BASE + 8),
        (BASE + 16, BASE + 24),
    ]
