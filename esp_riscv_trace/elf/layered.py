# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Layered ELF view combining an application ELF with an optional ROM ELF."""

from __future__ import annotations

from . import CodeImage, CodeReadError, SourceResolver
from .dwarf import SourceLocation

_ROM_SYMBOL_PREFIX = "rom:"


class LayeredImage:
    """App-first, ROM-fallback executable + symbol view."""

    def __init__(self, app: CodeImage, rom: CodeImage | None = None) -> None:
        self._app = app
        self._rom = rom

    def read(self, vaddr: int, size: int) -> bytes:
        """Return bytes from the application ELF, falling back to ROM."""
        try:
            return self._app.read(vaddr, size)
        except CodeReadError as app_err:
            if self._rom is None:
                raise
            try:
                return self._rom.read(vaddr, size)
            except CodeReadError:
                raise app_err from None

    def symbol_at(self, vaddr: int) -> str | None:
        """Return symbol name at vaddr, prefixed with 'rom:' if from ROM."""
        name = self._app.symbol_at(vaddr)
        if name is not None:
            return name
        if self._rom is None:
            return None
        rom_name = self._rom.symbol_at(vaddr)
        if rom_name is None:
            return None
        return _ROM_SYMBOL_PREFIX + rom_name

    def data_at(self, vaddr: int) -> str | None:
        """Return the variable at vaddr in the application ELF, when it names variables."""
        data_at = getattr(self._app, "data_at", None)
        return data_at(vaddr) if data_at is not None else None


class LayeredResolver:
    """App-first, ROM-fallback DWARF line resolver."""

    def __init__(self, app: SourceResolver, rom: SourceResolver | None = None) -> None:
        self._app = app
        self._rom = rom

    def resolve(self, pc: int) -> SourceLocation | None:
        loc = self._app.resolve(pc)
        if loc is not None:
            return loc
        if self._rom is None:
            return None
        return self._rom.resolve(pc)


__all__ = [
    "LayeredImage",
    "LayeredResolver",
]
