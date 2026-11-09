# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Call stacks of a profile, built from synthetic trace events."""

from __future__ import annotations

from helpers import BAR, FOO, IMAGE, ISR, MAIN, Trace

from esp_riscv_trace.analysis.profile import build_profile
from esp_riscv_trace.decoder.reconstruct import GapReason
from esp_riscv_trace.exporters.profile import format_folded, profile_to_speedscope

_TIMER_IRQ = 7


def test_profile_counts_each_instruction_under_its_call_stack() -> None:
    trace = (
        Trace()
        .sync(MAIN)
        .run(MAIN, 2)
        .call(MAIN + 8, FOO)
        .run(FOO, 3)
        .trap(_TIMER_IRQ, epc=FOO + 12, interrupt=True)
        .run(ISR, 2)
        .mret(ISR + 8, FOO + 12)
        .ret(FOO + 12, MAIN + 12)
        .run(MAIN + 12)
        # A call into code outside the traced range, which returns after it
        .call(MAIN + 16, None)
        .gap(GapReason.UNTRACED)
        .sync(MAIN + 20)
        .run(MAIN + 20)
        # main returns to a caller older than the trace
        .ret(MAIN + 24, BAR)
        .run(BAR, 2)
        .gap()
        .sync(FOO)
        .run(FOO)
    )
    profile = build_profile(trace.events, IMAGE)

    assert profile.runs == (
        (("main",), 3),
        (("main", "foo"), 3),
        (("main", "foo", "[interrupt 7]", "isr"), 3),
        (("main", "foo"), 1),
        (("main",), 4),
        (("bar",), 2),
        (("foo",), 1),
    )
    assert profile.instructions == 17
    assert format_folded([("core 0", profile)]) == [
        "core 0;bar 2",
        "core 0;foo 1",
        "core 0;main 7",
        "core 0;main;foo 4",
        "core 0;main;foo;[interrupt 7];isr 3",
    ]
    speedscope = profile_to_speedscope([("core 0", profile)], "test")
    assert [frame["name"] for frame in speedscope["shared"]["frames"]] == ["main", "foo", "[interrupt 7]", "isr", "bar"]
    (sampled,) = speedscope["profiles"]
    assert sampled["samples"] == [[0], [0, 1], [0, 1, 2, 3], [0, 1], [0], [4], [1]]
    assert sampled["weights"] == [3, 3, 3, 1, 4, 2, 1]
    assert (sampled["unit"], sampled["endValue"]) == ("none", 17)
