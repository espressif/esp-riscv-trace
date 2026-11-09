# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Address lookup in overlapping interval indexes."""

from __future__ import annotations

import pytest

from esp_riscv_trace.elf.intervals import IntervalIndex

_NESTED = IntervalIndex([(0x100, 0x200, "outer"), (0x120, 0x140, "inner"), (0x180, 0x181, "byte")])


@pytest.mark.parametrize(
    ("addr", "value"),
    [
        (0x0FF, None),  # before the first start
        (0x100, "outer"),
        (0x11F, "outer"),
        (0x120, "inner"),
        (0x13F, "inner"),
        (0x140, "outer"),  # back in the outer interval past the inner end
        (0x17F, "outer"),
        (0x180, "byte"),
        (0x181, "outer"),
        (0x1FF, "outer"),
        (0x200, None),  # end is exclusive
    ],
)
def test_nested_intervals(addr: int, value: str | None) -> None:
    assert _NESTED.lookup(addr) == value


def test_overlapping_intervals_prefer_latest_start() -> None:
    index = IntervalIndex([(0, 10, "a"), (5, 15, "b")])
    assert [index.lookup(a) for a in (4, 5, 9, 10, 14, 15)] == ["a", "b", "b", "b", "b", None]


def test_equal_starts_later_entry_wins() -> None:
    index = IntervalIndex([(0, 10, "a"), (0, 10, "b")])
    assert index.lookup(3) == "b"


def test_equal_starts_shorter_later_entry_falls_back() -> None:
    index = IntervalIndex([(0, 10, "long"), (0, 4, "short")])
    assert index.lookup(2) == "short"
    assert index.lookup(6) == "long"


def test_size_one_interval() -> None:
    index = IntervalIndex([(7, 8, "x")])
    assert [index.lookup(a) for a in (6, 7, 8)] == [None, "x", None]


def test_gap_between_intervals() -> None:
    index = IntervalIndex([(0, 4, "a"), (10, 20, "b"), (30, 40, "c")])
    assert [index.lookup(a) for a in (4, 9, 20, 29, 40)] == [None] * 5
    assert index.lookup(35) == "c"


def test_long_early_interval_covers_later_gaps() -> None:
    index = IntervalIndex([(0, 100, "a"), (10, 12, "b"), (20, 22, "c")])
    assert index.lookup(30) == "a"
    assert index.lookup(21) == "c"


def test_empty_index() -> None:
    index: IntervalIndex[str] = IntervalIndex([])
    assert len(index) == 0
    assert index.lookup(0) is None


def test_len_counts_entries() -> None:
    assert len(_NESTED) == 3
