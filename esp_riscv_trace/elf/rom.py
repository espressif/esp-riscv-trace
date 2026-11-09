# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""ROM ELF auto-detection."""

from __future__ import annotations

from enum import StrEnum
from pathlib import Path

from esp_pylib.rom import get_idf_path, get_rom_elf_dir, get_rom_elf_path


class RomStatus(StrEnum):
    """Result of the ROM ELF lookup."""

    PROVIDED = "provided"
    AUTO = "auto-detected"
    NOT_FOUND = "not-found"
    MISSING_ENV = "missing-env"
    INVALID = "invalid"


def resolve_rom_elf(chip: str, chip_revision: int) -> tuple[Path | None, RomStatus]:
    """Find the ROM ELF for ``chip`` and ``chip_revision`` in the ESP-IDF environment."""
    if not get_idf_path() or not get_rom_elf_dir():
        return None, RomStatus.MISSING_ENV
    resolved = get_rom_elf_path(chip, chip_revision)
    if resolved is None or not Path(resolved).is_file():
        return None, RomStatus.NOT_FOUND
    return Path(resolved), RomStatus.AUTO


__all__ = [
    "RomStatus",
    "resolve_rom_elf",
]
