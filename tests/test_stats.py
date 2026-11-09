# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""StatsAccumulator fragment boundaries, unresolved PC accounting and denominator."""

from __future__ import annotations

from collections import Counter

import pytest
from helpers import FakeImage, FakeResolver

from esp_riscv_trace.analysis.stats import (
    UNRESOLVED_FILE,
    UNRESOLVED_LINE,
    StatsAccumulator,
    StatsSnapshot,
    unresolved_function_label,
)
from esp_riscv_trace.elf.dwarf import SourceLocation
from esp_riscv_trace.ir.events import (
    InsnClass,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
)


def _acc(symbols=None, locs=None) -> StatsAccumulator:
    return StatsAccumulator(FakeImage(symbols=symbols), FakeResolver(locs))


def _inst(pc: int, cls: InsnClass = InsnClass.NONE, next_pc: int | None = None) -> TraceInstruction:
    return TraceInstruction(
        pc=pc,
        length=4,
        cls=cls,
        branch_taken=None if cls != InsnClass.BRANCH else False,
        next_pc=(pc + 4) if next_pc is None else next_pc,
        provenance=None,
    )


def _sync(pc: int = 0x2000) -> TraceSync:
    return TraceSync(pc=pc, privilege=1, provenance=None)


def _trap() -> TraceTrap:
    return TraceTrap(
        cause=11,
        interrupt=False,
        handler_pc=None,
        interrupted_pc=None,
        tval=None,
        provenance=None,
    )


def _gap() -> TraceGap:
    return TraceGap("walker_not_anchored", "d", None)


def test_close_on_empty_stream_returns_zero_snapshot() -> None:
    snap = _acc().close()
    assert isinstance(snap, StatsSnapshot)
    assert snap.total_instructions == 0
    assert snap.sample_hits == Counter()
    assert snap.block_hits == Counter()
    assert snap.unresolved_pcs == Counter()


def test_snapshot_counters_are_copies() -> None:
    a = _acc(symbols={0x1000: "main"}, locs={0x1000: SourceLocation("main.c", 1)})
    a.push(_inst(0x1000))
    snap = a.close()
    snap.sample_hits[("evil", "x", 1)] = 999
    a.push(_inst(0x1000))
    assert ("evil", "x", 1) not in a.close().sample_hits


def test_resolved_instruction_populates_sample_hit() -> None:
    a = _acc(symbols={0x1000: "main"}, locs={0x1000: SourceLocation("main.c", 42)})
    a.push(_inst(0x1000))
    snap = a.close()
    assert snap.sample_hits == Counter({("main", "main.c", 42): 1})
    assert snap.unresolved_pcs == Counter()
    assert snap.total_instructions == 1


def test_unresolved_function_counted_and_uses_per_pc_label() -> None:
    a = _acc(locs={0x1000: SourceLocation("main.c", 42)})
    a.push(_inst(0x1000))
    snap = a.close()
    label = unresolved_function_label(0x1000)
    assert label == "<0x00001000:unresolved>"
    assert (label, "main.c", 42) in snap.sample_hits
    assert snap.unresolved_pcs["function"] == 1
    assert snap.unresolved_pcs["line"] == 0


def test_distinct_unresolved_pcs_stay_on_distinct_rows() -> None:
    a = _acc(locs={0x1000: SourceLocation("main.c", 42), 0x2000: SourceLocation("main.c", 43)})
    a.push(_inst(0x1000))
    a.push(_inst(0x2000))
    snap = a.close()
    assert (unresolved_function_label(0x1000), "main.c", 42) in snap.sample_hits
    assert (unresolved_function_label(0x2000), "main.c", 43) in snap.sample_hits
    assert snap.unresolved_pcs["function"] == 2


def test_unresolved_line_counted_and_uses_sentinel() -> None:
    a = _acc(symbols={0x1000: "main"})
    a.push(_inst(0x1000))
    snap = a.close()
    assert ("main", UNRESOLVED_FILE, UNRESOLVED_LINE) in snap.sample_hits
    assert snap.unresolved_pcs["line"] == 1
    assert snap.unresolved_pcs["function"] == 0


def test_both_unresolved_counts_both_reasons_independently() -> None:
    a = _acc()
    a.push(_inst(0x1000))
    snap = a.close()
    assert snap.unresolved_pcs["function"] == 1
    assert snap.unresolved_pcs["line"] == 1


def test_pct_denominator_is_total_instructions_not_sample_hits_len() -> None:
    a = _acc(symbols={0x1000: "main"}, locs={0x1000: SourceLocation("m.c", 1)})
    for _ in range(5):
        a.push(_inst(0x1000))
    snap = a.close()
    assert snap.total_instructions == 5
    assert sum(snap.sample_hits.values()) == 5
    assert len(snap.sample_hits) == 1


def test_first_instruction_starts_first_fragment() -> None:
    a = _acc(symbols={0x1000: "main"})
    a.push(_inst(0x1000, InsnClass.NONE))
    assert a.close().block_hits == Counter({("main", 0x1000): 1})


def test_consecutive_non_terminators_are_one_fragment() -> None:
    a = _acc(symbols={0x1000: "main", 0x1004: "main", 0x1008: "main"})
    for pc in (0x1000, 0x1004, 0x1008):
        a.push(_inst(pc, InsnClass.NONE))
    assert a.close().block_hits == Counter({("main", 0x1000): 1})


@pytest.mark.parametrize(
    "cls,next_pc,next_name",
    [
        pytest.param(InsnClass.BRANCH, 0x1004, "main", id="branch"),
        pytest.param(InsnClass.JUMP, 0x2000, "target", id="jump"),
        pytest.param(InsnClass.UNINFERABLE, 0x2000, "callee", id="uninferable"),
        pytest.param(InsnClass.TRAP, 0x2000, "handler", id="trap_class"),
    ],
)
def test_terminator_ends_fragment(cls: InsnClass, next_pc: int, next_name: str) -> None:
    a = _acc(symbols={0x1000: "main", next_pc: next_name})
    a.push(_inst(0x1000, cls))
    a.push(_inst(next_pc, InsnClass.NONE))
    assert a.close().block_hits == Counter({("main", 0x1000): 1, (next_name, next_pc): 1})


@pytest.mark.parametrize(
    "event,next_name",
    [
        pytest.param(_sync, "main", id="sync"),
        pytest.param(_trap, "handler", id="trap"),
        pytest.param(_gap, "main", id="gap"),
    ],
)
def test_event_ends_fragment(event, next_name: str) -> None:
    a = _acc(symbols={0x1000: "main", 0x1004: next_name})
    a.push(_inst(0x1000, InsnClass.NONE))
    a.push(event())
    a.push(_inst(0x1004, InsnClass.NONE))
    assert a.close().block_hits == Counter({("main", 0x1000): 1, (next_name, 0x1004): 1})


def test_block_hits_key_is_function_and_entry_pc_not_just_pc() -> None:
    a = _acc(symbols={0x1000: "main"})
    a.push(_inst(0x1000, InsnClass.NONE))
    keys = list(a.close().block_hits)
    assert keys
    for key in keys:
        assert isinstance(key, tuple) and len(key) == 2
        assert isinstance(key[0], str) and isinstance(key[1], int)


def test_repeated_fragment_execution_counts_up() -> None:
    a = _acc(symbols={0x1000: "main", 0x1004: "main"})
    a.push(_inst(0x1000, InsnClass.JUMP))
    a.push(_inst(0x1000, InsnClass.JUMP))
    assert a.close().block_hits == Counter({("main", 0x1000): 2})


def test_unresolved_function_appears_in_block_hits() -> None:
    a = _acc()
    a.push(_inst(0x1000, InsnClass.NONE))
    assert a.close().block_hits == Counter({(unresolved_function_label(0x1000), 0x1000): 1})


@pytest.mark.parametrize("event", [_sync, _trap, _gap], ids=["sync", "trap", "gap"])
def test_non_instruction_event_alone_does_not_touch_counters(event) -> None:
    a = _acc()
    a.push(event())
    snap = a.close()
    assert snap.sample_hits == Counter()
    assert snap.unresolved_pcs == Counter()
    assert snap.total_instructions == 0


def test_two_accumulators_do_not_share_state() -> None:
    a = _acc(symbols={0x1000: "main"})
    b = _acc(symbols={0x2000: "other"})
    a.push(_inst(0x1000, InsnClass.NONE))
    b.push(_inst(0x2000, InsnClass.NONE))
    snap_a = a.close()
    snap_b = b.close()
    assert snap_a.total_instructions == 1
    assert snap_b.total_instructions == 1
    assert ("main", "<unresolved>", 0) in snap_a.sample_hits
    assert ("other", "<unresolved>", 0) in snap_b.sample_hits
