# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""ELF and DWARF helpers for symbol and line resolution."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol

if TYPE_CHECKING:
    from .dwarf import SourceLocation


class CodeReadError(ValueError):
    """Raised by ``CodeImage.read`` when the image does not cover the requested range.

    The decoder reports this as a gap and continues, so it is not an ``EspRiscvTraceError``.
    """


class CodeImage(Protocol):
    """Instruction bytes and symbol names for program counters."""

    def read(self, vaddr: int, size: int) -> bytes: ...

    def symbol_at(self, vaddr: int) -> str | None: ...


class SourceResolver(Protocol):
    """Source coordinates for program counters."""

    def resolve(self, pc: int) -> SourceLocation | None: ...


__all__ = ["CodeImage", "CodeReadError", "SourceResolver"]
