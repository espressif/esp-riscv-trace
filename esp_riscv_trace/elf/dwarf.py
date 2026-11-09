# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""DWARF line-program resolver that maps program counters to source lines."""

from __future__ import annotations

import os
from dataclasses import dataclass
from functools import lru_cache

from .image import ElfImage
from .intervals import IntervalIndex

_DWARF_CACHE_SIZE = 4096


@dataclass(frozen=True)
class SourceLocation:
    """One row of DWARF line-program state pointing at source coordinates."""

    file: str
    line: int


class DwarfResolver:
    """Resolve program counters to source file and line using DWARF."""

    def __init__(self, image: ElfImage) -> None:
        self._info = image.dwarf_info
        self._index: IntervalIndex[SourceLocation] = IntervalIndex(
            self._build_index() if self._info is not None else []
        )
        self._lookup = lru_cache(maxsize=_DWARF_CACHE_SIZE)(self._index.lookup)

    def resolve(self, pc: int) -> SourceLocation | None:
        """Return the source coordinates for ``pc`` or ``None``."""
        if not self._index:
            return None
        return self._lookup(pc)

    def _build_index(self) -> list[tuple[int, int, SourceLocation]]:
        intervals: list[tuple[int, int, SourceLocation]] = []
        info = self._info
        for cu in info.iter_CUs():
            lineprog = info.line_program_for_CU(cu)
            if lineprog is None:
                continue
            version = lineprog.header["version"]
            file_entries = lineprog["file_entry"]
            include_dirs = lineprog["include_directory"]
            cu_dir = _cu_comp_dir(cu)
            paths: dict[int, str] = {}
            seq_rows: list = []
            for entry in lineprog.get_entries():
                state = entry.state
                if state is None:
                    continue
                seq_rows.append(state)
                if state.end_sequence:
                    _flatten_sequence(seq_rows, file_entries, include_dirs, version, cu_dir, paths, intervals)
                    seq_rows = []
        intervals.sort(key=lambda t: t[0])
        return intervals


def _flatten_sequence(
    rows,
    file_entries,
    include_dirs,
    version: int,
    cu_dir: str,
    paths: dict[int, str],
    out: list,
) -> None:
    """Turn a completed line-program sequence into non-empty PC intervals."""
    for i in range(len(rows) - 1):
        row = rows[i]
        nxt = rows[i + 1]
        if row.end_sequence:
            continue
        if row.address >= nxt.address:
            continue
        path = paths.get(row.file)
        if path is None:
            path = paths[row.file] = _resolve_file(row.file, file_entries, include_dirs, version, cu_dir)
        out.append((row.address, nxt.address, SourceLocation(file=path, line=row.line)))


def _cu_comp_dir(cu) -> str:
    """Return the compilation-unit compilation directory or empty string."""
    top = cu.get_top_DIE()
    attr = top.attributes.get("DW_AT_comp_dir")
    if attr is None:
        return ""
    return _as_str(attr.value)


def _resolve_file(
    state_file: int,
    file_entries,
    include_dirs,
    version: int,
    cu_dir: str,
) -> str:
    """Build a source path from a DWARF line-program file index."""
    idx = state_file if version >= 5 else state_file - 1
    if idx < 0 or idx >= len(file_entries):
        return "<unknown>"
    fe = file_entries[idx]
    name = _as_str(fe.name)
    dir_index = getattr(fe, "dir_index", None)
    directory = ""
    from_include_table = False
    if version >= 5:
        # DWARFv5 directory indexes are zero-based.
        if dir_index is None:
            directory = cu_dir
        elif 0 <= dir_index < len(include_dirs):
            directory = _as_str(include_dirs[dir_index])
            from_include_table = True
    else:
        # DWARFv4 uses 1-based directory indexes.
        if dir_index is None or dir_index == 0:
            directory = cu_dir
        elif 0 < dir_index <= len(include_dirs):
            directory = _as_str(include_dirs[dir_index - 1])
            from_include_table = True
    # Resolve relative include paths against the compilation directory.
    if from_include_table and cu_dir and not os.path.isabs(directory):
        directory = os.path.join(cu_dir, directory)
    if directory:
        return os.path.join(directory, name)
    return name


def _as_str(value) -> str:
    """Return a DWARF string attribute as text, empty for ``None``."""
    if value is None:
        return ""
    if isinstance(value, bytes):
        return value.decode("utf-8", "replace")
    return str(value)


__all__ = [
    "DwarfResolver",
    "SourceLocation",
]
