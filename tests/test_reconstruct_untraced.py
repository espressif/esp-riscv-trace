# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Runs the encoder filter leaves untraced, such as calls into ROM."""

from __future__ import annotations

import dataclasses

from helpers import BASE, JALR_X0_RA, NOP, FakeImage, make_frame, make_sync

from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.packets import (
    QUAL_STATUS_ENDED_REP,
    SupportPacket,
    SyncPacket,
)
from esp_riscv_trace.decoder.params import default_profile
from esp_riscv_trace.ir.events import InsnClass, LinkKind, TraceGap, TraceInstruction, TraceSync
from esp_riscv_trace.snapshot import abi

ET = default_profile("esp32c5")


def _sync(pc: int, index: int) -> SyncPacket:
    return dataclasses.replace(make_sync(pc, branch=1), frame=make_frame(index))


def _support(index: int, qual_status: int, *, enable: int = 1) -> SupportPacket:
    return SupportPacket(
        frame=make_frame(index),
        enable=enable,
        qual_status=qual_status,
        encoder_mode=0,
        ioptions=0,
        full_address_mode=None,
    )


def _run(packets, image: FakeImage):
    return list(reconstruct.reconstruct(packets, image, ET, abi.AddressMode.FULL))


def test_ended_rep_retires_the_jump_out_of_the_traced_range() -> None:
    image = FakeImage(BASE, NOP + JALR_X0_RA + NOP + JALR_X0_RA)
    events = _run([_sync(BASE, 0), _support(1, QUAL_STATUS_ENDED_REP), _sync(BASE + 8, 2)], image)

    assert [type(e) for e in events] == [
        TraceSync,
        TraceInstruction,
        TraceInstruction,
        TraceGap,
        TraceSync,
        TraceInstruction,
    ]
    jump = events[2]
    assert isinstance(jump, TraceInstruction)
    assert (jump.pc, jump.cls, jump.link) == (BASE + 4, InsnClass.UNINFERABLE, LinkKind.RETURN)
    assert jump.next_pc is None
    gap = events[3]
    assert isinstance(gap, TraceGap)
    assert gap.reason == reconstruct.GapReason.UNTRACED
    assert gap.detail == f"the jump at 0x{BASE + 4:08x} leaves the traced range (qual_status=1)"
    resumed = events[4]
    assert isinstance(resumed, TraceSync)
    assert resumed.pc == BASE + 8
