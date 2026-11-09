# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Walker termination on inferable-only infinite loops.

A capture frozen while the core spins in the panic handler ends inside a
direct jump-to-self. No packet ever breaks that loop, so the walker must
stop on its own instead of following the jump forever. The core keeps
running the jump, so a trap that ends the loop is taken at it.
"""

from __future__ import annotations

from helpers import FakeImage, encode_jal, make_frame, make_sync

import esp_riscv_trace.decoder.reconstruct as reconstruct_mod
from esp_riscv_trace.decoder import reconstruct
from esp_riscv_trace.decoder.packets import TrapPacket
from esp_riscv_trace.decoder.params import default_profile
from esp_riscv_trace.ir.events import InsnClass, TraceGap, TraceInstruction, TraceSync, TraceTrap
from esp_riscv_trace.snapshot import abi

_BASE = 0x40807030
_HANDLER = _BASE + 0x100


def test_self_jump_spins_without_gap() -> None:
    # `jal zero, 0` at _BASE jumps to itself.
    image = FakeImage(_BASE, encode_jal(0, 0))
    profile = default_profile("esp32c5")

    events = list(reconstruct.reconstruct([make_sync(_BASE, branch=0)], image, profile, abi.AddressMode.FULL))

    assert isinstance(events[0], TraceSync)
    jump = events[1]
    assert isinstance(jump, TraceInstruction)
    assert jump.cls == InsnClass.JUMP
    assert jump.next_pc == _BASE
    # The self-jump is produced exactly once, not repeatedly.
    assert len(events) == 2


def test_interrupt_in_self_jump_is_taken_at_it() -> None:
    image = FakeImage(_BASE, encode_jal(0, 0).ljust(_HANDLER - _BASE, b"\0") + encode_jal(0, 0))
    profile = default_profile("esp32c5")
    interrupt = TrapPacket(
        frame=make_frame(1),
        branch=1,
        privilege=0,
        ecause=40,
        interrupt=1,
        thaddr=1,
        raw_address=_HANDLER >> 1,
        address_pc=_HANDLER,
        tval=0,
    )

    events = list(
        reconstruct.reconstruct([make_sync(_BASE, branch=0), interrupt], image, profile, abi.AddressMode.FULL)
    )

    assert [type(e) for e in events] == [TraceSync, TraceInstruction, TraceTrap, TraceInstruction]
    trap = events[2]
    assert isinstance(trap, TraceTrap)
    assert (trap.cause, trap.interrupt, trap.epc) == (40, True, _BASE)
    # The handler spins at its own self-jump
    handler = events[3]
    assert isinstance(handler, TraceInstruction)
    assert handler.pc == _HANDLER


def test_jump_cycle_stops_at_budget(monkeypatch) -> None:
    # Two blocks that jump to each other. Neither is a self-jump, so the
    # per-packet instruction budget is what breaks the cycle.
    monkeypatch.setattr(reconstruct_mod, "_WALK_INSN_LIMIT", 32)
    data = encode_jal(0, 4) + encode_jal(0, -4)  # _BASE -> _BASE+4 -> _BASE
    image = FakeImage(_BASE, data)
    profile = default_profile("esp32c5")

    events = list(reconstruct.reconstruct([make_sync(_BASE, branch=0)], image, profile, abi.AddressMode.FULL))

    gap = events[-1]
    assert isinstance(gap, TraceGap)
    assert gap.reason == reconstruct.GapReason.INFINITE_LOOP
    instructions = [e for e in events if isinstance(e, TraceInstruction)]
    assert len(instructions) == 32
