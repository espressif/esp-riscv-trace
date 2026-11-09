# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Text and JSON exporters for the call history."""

from __future__ import annotations

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

from ..analysis.history import (
    History,
    HistoryEntry,
    HistoryItem,
    HistoryKind,
    HistoryLoop,
    cause_name,
    iter_entries,
)
from ..analysis.location import locate
from ..elf import CodeImage, SourceResolver
from ._format import one_line

_INDENT = "  "
# Deeper lines are drawn at this depth, so a runaway recursion stays readable
_MAX_DRAWN_DEPTH = 32
_OUTLINE_WIDTH = 72
_UNTRACED = "<untraced>"
_UNKNOWN = "?"
_DATA = " (data)"


@dataclass(frozen=True)
class _Line:
    pc: int | None  # source line shown for this line
    instructions: int | None
    depth: int
    text: str


def format_history(
    history: History,
    image: CodeImage | None,
    resolver: SourceResolver | None,
    *,
    label: str,
) -> list[str]:
    """Return the call outline as text lines, oldest first.

    Each line shows the PC whose source line it names, the instructions retired until the next line,
    and the outline. A call opens a ``{`` that its return closes, a call that returns before anything
    else happens is one line, and repeated passes of the same items show once with their count.
    """
    lines = [
        f"# history           {label}  items={len(history.items)}  omitted={history.omitted}  "
        f"instructions={history.instructions}"
    ]
    fault = history.fault
    if fault is not None:
        lines.append(f"# fault             {_fault_summary(fault, image)}")
    lines.append(f"#         pc    insns  outline{' ' * (_OUTLINE_WIDTH - len('outline'))}  source")
    lines.extend(_render(line, resolver) for line in _outline(history.items, image, extra=0))
    return lines


def history_to_dict(history: History, image: CodeImage | None, resolver: SourceResolver | None) -> dict[str, Any]:
    """Return the history as JSON-ready data. Addresses are ``0x``-prefixed hex strings."""

    def location(pc: int | None) -> dict[str, Any] | None:
        if pc is None:
            return None
        loc = locate(pc, image, resolver)
        return {"function": loc.function, "file": loc.file, "line": loc.line}

    def entry(e: HistoryEntry) -> dict[str, Any]:
        return {
            "kind": e.kind.value,
            "depth": e.depth,
            "segment": e.segment,
            "instructions": e.instructions,
            "pc": _hex(e.pc),
            "target": _hex(e.target),
            "site": location(e.pc),
            "dest": location(e.target),
            "cause": e.cause,
            "cause_name": (
                cause_name(e.cause, e.kind == HistoryKind.INTERRUPT or e.interrupt) if e.cause is not None else None
            ),
            "tval": _hex(e.tval),
            "detail": e.detail or None,
        }

    def item(i: HistoryItem) -> dict[str, Any]:
        if isinstance(i, HistoryLoop):
            return {
                "kind": "loop",
                "count": i.count,
                "instructions": i.instructions,
                "body": [item(b) for b in i.body],
            }
        return entry(i)

    return {
        "omitted": history.omitted,
        "instructions": history.instructions,
        "fault": entry(history.fault) if history.fault is not None else None,
        "items": [item(i) for i in history.items],
    }


def _outline(items: Sequence[HistoryItem], image: CodeImage | None, *, extra: int) -> Iterator[_Line]:
    i = 0
    while i < len(items):
        it = items[i]
        if isinstance(it, HistoryLoop):
            depth = min(e.depth for e in iter_entries(it.body)) + extra
            yield _Line(None, it.instructions, depth, f"loop x{it.count} {{")
            yield from _outline(it.body, image, extra=extra + 1)
            yield _Line(None, None, depth, "}")
            i += 1
            continue
        after = items[i + 1] if i + 1 < len(items) else None
        if it.kind == HistoryKind.CALL and _returns_at_once(it, after):
            pc = it.pc if it.pc is not None else it.target
            yield _Line(pc, it.instructions, it.depth + extra, _callee(it.target, image))
            i += 2
            continue
        pc, text = _describe(it, image)
        yield _Line(pc, it.instructions, it.depth + extra, text)
        i += 1


def _returns_at_once(call: HistoryEntry, after: HistoryItem | None) -> bool:
    return (
        isinstance(after, HistoryEntry)
        and after.kind == HistoryKind.RETURN
        and after.depth == call.depth
        and after.segment == call.segment
        and not after.detail
    )


def _describe(e: HistoryEntry, image: CodeImage | None) -> tuple[int | None, str]:
    """Return the PC whose source line the entry shows, and its outline text."""
    kind = e.kind
    if kind == HistoryKind.START:
        return e.target, f"start in {_function(e.target, image)}"
    if kind == HistoryKind.CALL:
        return (e.pc if e.pc is not None else e.target), f"{_callee(e.target, image)} {{"
    if kind == HistoryKind.RETURN:
        text = f"}} {_function(e.pc, image)}"
        pc = e.target if e.target is not None else e.pc
        return pc, f"{text}  !! {e.detail}" if e.detail else text
    if kind == HistoryKind.TAIL_CALL:
        return e.pc, f"~> {_function(e.target, image)}"
    if kind == HistoryKind.INTERRUPT:
        return e.pc, f"interrupt {e.cause} in {_function(e.pc, image)} {{"
    if kind == HistoryKind.ECALL:
        return e.pc, f"ecall in {_function(e.pc, image)} {{"
    if kind == HistoryKind.TRAP_RETURN:
        if e.detail:
            return e.target, f"mret to {_function(e.target, image)}  ({e.detail})"
        return e.target, "} mret"
    if kind == HistoryKind.CONTEXT_SWITCH:
        text = f"<~ switch to {_function(e.target, image)}"
        return e.target, f"{text}  ({e.detail})" if e.detail else text
    if kind == HistoryKind.FAULT:
        return e.pc, f"!! {_fault_summary(e, image)}"
    return None, f"-- gap {e.detail}"


def _fault_summary(e: HistoryEntry, image: CodeImage | None) -> str:
    name = cause_name(e.cause, e.interrupt) if e.cause is not None else "fault"
    tval = f"  tval={_hex(e.tval)}" if e.tval is not None else ""
    epc = f"  epc={_hex(e.pc)}" if e.pc is not None else ""
    return f"{name} in {_function(e.pc, image)}{epc}{tval}"


def _render(line: _Line, resolver: SourceResolver | None) -> str:
    pc = f"0x{line.pc:08x}" if line.pc is not None else ""
    count = str(line.instructions) if line.instructions is not None else ""
    outline = _INDENT * min(line.depth, _MAX_DRAWN_DEPTH) + one_line(line.text)
    source = ""
    if line.pc is not None:
        loc = locate(line.pc, None, resolver)
        if loc.file is not None:
            source = one_line(f"{loc.file}:{loc.line}")
    row = f"{pc:>10}  {count:>7}  {outline}"
    if source:
        row = f"{row.ljust(10 + 2 + 7 + 2 + _OUTLINE_WIDTH)}  {source}"
    return row.rstrip()


def _function(pc: int | None, image: CodeImage | None) -> str:
    """The function at ``pc``, or the variable code runs in there, as ``s_buf (data)``."""
    if pc is None:
        return _UNTRACED
    function = locate(pc, image, None).function
    if function:
        return function
    data_at = getattr(image, "data_at", None)
    try:
        variable = data_at(pc) if data_at is not None else None
    except Exception:
        variable = None
    return variable + _DATA if variable else _UNKNOWN


def _callee(pc: int | None, image: CodeImage | None) -> str:
    """Where a call goes, as ``foo()``, or as the variable when it goes into data."""
    name = _function(pc, image)
    return name if name.endswith(_DATA) else f"{name}()"


def _hex(value: int | None) -> str | None:
    return f"0x{value:08x}" if value is not None else None


__all__ = ["format_history", "history_to_dict"]
