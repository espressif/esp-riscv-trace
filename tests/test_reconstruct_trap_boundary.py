# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""The instruction a trap is taken at, from the Format 1/2 address before the trap packet."""

from __future__ import annotations

import pytest
from helpers import BASE, JALR_X0_RA, NOP, FakeImage, make_frame, make_sync

from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.packets import (
    Format1NoAddrPacket,
    Format1WithAddrPacket,
    Format2Packet,
    SupportPacket,
    TrapPacket,
)
from esp_riscv_trace.decoder.params import DecoderProfile, default_profile
from esp_riscv_trace.ir.events import (
    TraceGap,
    TraceInstruction,
    TraceTrap,
    TraceWarning,
)
from esp_riscv_trace.snapshot import abi

ET = default_profile("esp32c5")
PT = default_profile("esp32c6")

HANDLER = 0x40801000
# beq x0, x0, 0, the branch map decides whether it is taken
BRANCH_SELF = bytes.fromhex("63000000")

_ILLEGAL_INSTRUCTION = 2
_STORE_ACCESS_FAULT = 7
_MACHINE_TIMER = 7


def _image(code: bytes) -> FakeImage:
    """Image with ``code`` at BASE and a return at HANDLER."""
    return FakeImage(BASE, code.ljust(HANDLER - BASE, b"\0") + JALR_X0_RA)


def _f2(pc: int, *, updiscon: int = 0, index: int = 1) -> Format2Packet:
    return Format2Packet(
        frame=make_frame(index), notify=0, updiscon=updiscon, raw_address=pc >> 1, address_width_bits=32
    )


def _f1(pc: int, *, branch_map: int, index: int = 1) -> Format1WithAddrPacket:
    return Format1WithAddrPacket(
        frame=make_frame(index),
        branches=1,
        branch_map=branch_map,
        branch_map_width_bits=1,
        notify=0,
        updiscon=0,
        raw_address=pc >> 1,
        address_width_bits=32,
    )


def _trap(*, ecause: int, interrupt: int = 0, thaddr: int | None = 1, tval: int | None = 0, index: int = 2):
    return TrapPacket(
        frame=make_frame(index),
        branch=1,
        privilege=0,
        ecause=ecause,
        interrupt=interrupt,
        thaddr=thaddr,
        raw_address=HANDLER >> 1,
        address_pc=HANDLER,
        tval=tval,
    )


def _run(packets, image: FakeImage, profile: DecoderProfile = ET) -> list:
    return list(reconstruct.reconstruct(packets, image, profile, abi.AddressMode.FULL))


def _before_trap(events) -> tuple[list[int], TraceTrap]:
    """PCs retired before the first trap, and that trap."""
    at = next(i for i, e in enumerate(events) if isinstance(e, TraceTrap))
    return [e.pc for e in events[:at] if isinstance(e, TraceInstruction)], events[at]


def _pcs(events) -> list[int]:
    return [e.pc for e in events if isinstance(e, TraceInstruction)]


# NOPs at BASE, +4, +8 and a return at +12 the walker waits at
_PASSED = NOP + NOP + NOP + JALR_X0_RA


def test_exception_at_an_address_the_walker_already_passed() -> None:
    # The sync walks to the return at BASE+12. The faulting store at BASE+8
    # did not retire, so the trap is taken there.
    events = _run([make_sync(BASE, branch=1), _f2(BASE + 8), _trap(ecause=_STORE_ACCESS_FAULT)], _image(_PASSED))

    retired, trap = _before_trap(events)
    assert retired == [BASE, BASE + 4]
    assert trap.epc == BASE + 8
    assert not any(isinstance(e, TraceGap) for e in events)


def test_plain_interrupt_retires_the_reported_instruction() -> None:
    events = _run(
        [make_sync(BASE, branch=1), _f2(BASE + 8), _trap(ecause=_MACHINE_TIMER, interrupt=1, tval=None)],
        _image(_PASSED),
    )

    retired, trap = _before_trap(events)
    assert retired == [BASE, BASE + 4, BASE + 8]
    assert trap.epc == BASE + 12


# A branch at BASE+4 the walker waits at, then NOPs and a return at +16
_AHEAD = NOP + BRANCH_SELF + NOP + NOP + JALR_X0_RA


def test_pt10_exception_at_an_address_ahead_matches_tval() -> None:
    trap = _trap(ecause=_ILLEGAL_INSTRUCTION, thaddr=None, tval=BASE + 8)
    events = _run([make_sync(BASE, branch=1), _f1(BASE + 8, branch_map=1), trap], _image(_AHEAD), PT)

    retired, trap_event = _before_trap(events)
    assert retired == [BASE, BASE + 4]
    assert (trap_event.epc, trap_event.interrupted_pc) == (BASE + 8, BASE + 8)
    assert not any(isinstance(e, TraceWarning) for e in events)


def test_updiscon_interrupt_does_not_retire_the_target() -> None:
    image = _image(NOP + JALR_X0_RA + NOP + NOP)
    events = _run(
        [make_sync(BASE, branch=1), _f2(BASE + 12, updiscon=1), _trap(ecause=_MACHINE_TIMER, interrupt=1, tval=None)],
        image,
    )

    retired, trap = _before_trap(events)
    assert retired == [BASE, BASE + 4]
    assert trap.epc == BASE + 12


@pytest.mark.parametrize(
    "following",
    [
        Format1NoAddrPacket(frame=make_frame(2), branches=0, branch_map=0),
        _f2(BASE + 4, index=2),
        SupportPacket(
            frame=make_frame(2), enable=1, qual_status=0, encoder_mode=0, ioptions=0, full_address_mode=False
        ),
    ],
    ids=["format1", "format2_elsewhere", "support"],
)
def test_speculative_walk_out_of_the_image_is_still_a_gap(following) -> None:
    image = FakeImage(BASE, NOP + NOP)
    events = _run([make_sync(BASE, branch=1), following], image)

    assert _pcs(events)[:2] == [BASE, BASE + 4]
    gaps = [e for e in events if isinstance(e, TraceGap)]
    assert gaps and gaps[0].reason == reconstruct.GapReason.PC_OUT_OF_IMAGE
    assert events.index(gaps[0]) == 3
