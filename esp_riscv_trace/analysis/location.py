# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Function and source line of a program counter."""

from __future__ import annotations

from dataclasses import dataclass

from ..elf import CodeImage, SourceResolver


@dataclass(frozen=True)
class CodeLocation:
    """Where a program counter is in the code. Fields the ELF files cannot resolve are ``None``."""

    pc: int
    function: str | None
    file: str | None
    line: int | None


def locate(pc: int, image: CodeImage | None, resolver: SourceResolver | None) -> CodeLocation:
    """Resolve ``pc`` to its function and source line, treating a failing lookup as unresolved."""
    function: str | None = None
    if image is not None:
        try:
            function = image.symbol_at(pc)
        except Exception:
            function = None
    source = None
    if resolver is not None:
        try:
            source = resolver.resolve(pc)
        except Exception:
            source = None
    if source is None:
        return CodeLocation(pc=pc, function=function, file=None, line=None)
    return CodeLocation(pc=pc, function=function, file=source.file, line=source.line)


__all__ = ["CodeLocation", "locate"]
