# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Flame graph exporters for profiles: folded stacks and the speedscope file format.

Counts are retired instructions. The trace has no time, so neither format claims a time unit.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

from .. import __version__
from ..analysis.profile import Profile, Stack
from ._format import one_line

SPEEDSCOPE_SCHEMA = "https://www.speedscope.app/file-format-schema.json"


def _frame_name(name: str) -> str:
    # The folded format separates frames with semicolons
    return one_line(name).replace(";", ":")


def format_folded(profiles: Sequence[tuple[str | None, Profile]]) -> list[str]:
    """Return one ``frame;frame;function count`` line per stack, as FlameGraph and speedscope read.

    Each profile comes with the name of an extra outermost frame, such as ``core 0``, or None for none.
    """
    lines: list[str] = []
    for root, profile in profiles:
        for stack, count in sorted(profile.stacks().items()):
            frames = stack if root is None else (root, *stack)
            lines.append(f"{';'.join(_frame_name(name) for name in frames)} {count}")
    return lines


def profile_to_speedscope(profiles: Sequence[tuple[str, Profile]], name: str) -> dict[str, Any]:
    """Return a speedscope file with one sampled profile per named profile.

    Every run of instructions under one stack is a sample weighted by its instructions, in trace order, so the time
    order view of speedscope shows the trace in order, in instructions.
    """
    frames: list[dict[str, str]] = []
    index: dict[str, int] = {}

    def frame_ids(stack: Stack) -> list[int]:
        ids = []
        for frame in stack:
            if frame not in index:
                index[frame] = len(frames)
                frames.append({"name": frame})
            ids.append(index[frame])
        return ids

    out = []
    for profile_name, profile in profiles:
        out.append(
            {
                "type": "sampled",
                "name": profile_name,
                "unit": "none",
                "startValue": 0,
                "endValue": profile.instructions,
                "samples": [frame_ids(stack) for stack, _count in profile.runs],
                "weights": [count for _stack, count in profile.runs],
            }
        )
    return {
        "$schema": SPEEDSCOPE_SCHEMA,
        "name": name,
        "exporter": f"esp-riscv-trace {__version__}",
        "activeProfileIndex": 0,
        "shared": {"frames": frames},
        "profiles": out,
    }


__all__ = ["SPEEDSCOPE_SCHEMA", "format_folded", "profile_to_speedscope"]
