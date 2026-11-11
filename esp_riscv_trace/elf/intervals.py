# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Lookup of possibly overlapping address intervals."""

from __future__ import annotations

import bisect
from collections.abc import Sequence
from typing import Generic, TypeVar

V = TypeVar("V")


class IntervalIndex(Generic[V]):
    """Find the value of the latest-starting ``[start, end)`` interval that contains an address.

    ``intervals`` must be sorted by start. For equal starts the later entry wins.
    """

    def __init__(self, intervals: Sequence[tuple[int, int, V]]) -> None:
        self._starts = [start for start, _end, _value in intervals]
        self._ends = [end for _start, end, _value in intervals]
        self._values = [value for _start, _end, value in intervals]
        self._max_end_prefix: list[int] = []
        max_end = 0
        for end in self._ends:
            max_end = max(max_end, end)
            self._max_end_prefix.append(max_end)

    def __len__(self) -> int:
        return len(self._starts)

    def lookup(self, addr: int) -> V | None:
        """Return the value for ``addr`` or ``None``."""
        i = bisect.bisect_right(self._starts, addr) - 1
        while i >= 0:
            if addr < self._ends[i]:
                return self._values[i]
            # Stop once no earlier interval reaches addr
            if i == 0 or self._max_end_prefix[i - 1] <= addr:
                return None
            i -= 1
        return None

    def start_before(self, addr: int) -> int | None:
        """Return the latest interval start at or before ``addr``, or ``None``."""
        i = bisect.bisect_right(self._starts, addr) - 1
        return self._starts[i] if i >= 0 else None


__all__ = ["IntervalIndex"]
