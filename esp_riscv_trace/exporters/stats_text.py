# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Text exporter for trace statistics tables."""

from __future__ import annotations

from ..analysis.stats import (
    UNRESOLVED_FILE,
    UNRESOLVED_LINE,
    StatsSnapshot,
)
from ._format import FUNCTION_COLUMN_WIDTH, one_line

_PCT_FMT = "{:>6.2f}"
_COUNT_FMT = "{:>6}"

_HEADER_STATS = f"#    pct  count  {'function':<{FUNCTION_COLUMN_WIDTH}}  file:line"
_HEADER_BLOCKS_TABLE = "#          count  entry_pc"


def format_stats(
    snapshot: StatsSnapshot,
    *,
    top: int = 20,
    per_block: bool = False,
    label: str = "",
) -> list[str]:
    """Return the lines for one ``StatsSnapshot`` table.

    ``top`` limits the sample table and must not be negative. ``per_block`` adds one block table per function,
    functions from the top rows first, then the rest by name. ``label`` goes into the ``# stats`` line.
    """
    if top < 0:
        raise ValueError(f"top must be >= 0, got {top}")
    lines: list[str] = []
    lines.append(_stats_header_meta(snapshot, top=top, label=label))

    if snapshot.total_instructions == 0:
        lines.append("# no instructions retired, sample table is empty")
        return lines

    lines.append(_HEADER_STATS)
    ranked = _rank_sample_hits(snapshot)
    top_rows = ranked[:top]
    functions_seen: list[str] = []
    seen_set: set[str] = set()
    for (func, file_name, line), count in top_rows:
        pct = 100.0 * count / snapshot.total_instructions
        lines.append(_format_sample_row(pct, count, func, file_name, line))
        if func not in seen_set:
            seen_set.add(func)
            functions_seen.append(func)

    unresolved = snapshot.unresolved_pcs
    if unresolved:
        parts = [f"{k}={unresolved[k]}" for k in sorted(unresolved) if unresolved[k] > 0]
        if parts:
            lines.append(f"#                   unresolved_pcs={', '.join(parts)}")

    if per_block:
        lines.extend(_format_block_tables(snapshot, functions_seen))

    return lines


def _stats_header_meta(
    snapshot: StatsSnapshot,
    *,
    top: int,
    label: str,
) -> str:
    div = snapshot.total_instructions
    label_part = f"{one_line(label)}  " if label else ""
    return f"# stats             {label_part}instructions={snapshot.total_instructions}  denominator={div}  top={top}"


def _rank_sample_hits(snapshot: StatsSnapshot) -> list:
    def sort_key(item):
        (func, file_name, line), count = item
        return (-count, func, file_name, line)

    return sorted(snapshot.sample_hits.items(), key=sort_key)


def _format_sample_row(pct: float, count: int, func: str, file_name: str, line: int) -> str:
    pct_s = _PCT_FMT.format(pct)
    count_s = _COUNT_FMT.format(count)
    func_render = _render_function(func)
    file_render = _render_file_line(file_name, line)
    func_padded = f"{func_render:<{FUNCTION_COLUMN_WIDTH}}"
    return f"  {pct_s} {count_s}  {func_padded}  {file_render}"


def _render_function(func: str) -> str:
    # Do not append parentheses to unresolved label placeholders.
    if func.startswith("<") and func.endswith(">"):
        safe = one_line(func)
        if len(safe) <= FUNCTION_COLUMN_WIDTH:
            return safe
        return safe[: FUNCTION_COLUMN_WIDTH - 1] + "~"
    safe = one_line(func)
    display = f"{safe}()"
    if len(display) <= FUNCTION_COLUMN_WIDTH:
        return display
    return display[: FUNCTION_COLUMN_WIDTH - 1] + "~"


def _render_file_line(file_name: str, line: int) -> str:
    if file_name == UNRESOLVED_FILE and line == UNRESOLVED_LINE:
        return UNRESOLVED_FILE
    safe = one_line(file_name)
    if line == UNRESOLVED_LINE:
        return safe
    return f"{safe}:{line}"


def _format_block_tables(snapshot: StatsSnapshot, functions_seen: list) -> list[str]:
    lines: list[str] = []
    grouped: dict = {}
    for (func, entry_pc), count in snapshot.block_hits.items():
        grouped.setdefault(func, []).append((entry_pc, count))
    if not grouped:
        return lines

    lines.append(f"# blocks            distinct_entry_pcs={len(snapshot.block_hits)}")
    seen: set = set()
    ordered_functions = [f for f in functions_seen if f in grouped]
    for func in ordered_functions:
        seen.add(func)
        lines.extend(_format_one_function_blocks(func, grouped[func]))
    for func in sorted(grouped):
        if func in seen:
            continue
        lines.extend(_format_one_function_blocks(func, grouped[func]))
    return lines


def _format_one_function_blocks(func: str, entries: list) -> list[str]:
    lines: list[str] = []
    lines.append(f"# {_render_function(func)}")
    lines.append(_HEADER_BLOCKS_TABLE)
    for entry_pc, count in sorted(entries, key=lambda x: (-x[1], x[0])):
        count_s = _COUNT_FMT.format(count)
        lines.append(f"          {count_s}  0x{entry_pc:08x}")
    return lines


__all__ = ["format_stats"]
