# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""ELF helpers: app-first ROM-fallback layering, symbol extents and ROM ELF lookup."""

from __future__ import annotations

import struct
from pathlib import Path
from unittest import mock

import pytest
from helpers import FakeImage, FakeResolver

from esp_riscv_trace.elf import CodeReadError, rom
from esp_riscv_trace.elf.dwarf import SourceLocation
from esp_riscv_trace.elf.image import ElfImage, _fill_zero_sizes
from esp_riscv_trace.elf.layered import LayeredImage, LayeredResolver
from esp_riscv_trace.elf.rom import (
    RomStatus,
    resolve_rom_elf,
)

_ROM_ENV = {"IDF_PATH": "/opt/esp/idf", "ESP_ROM_ELF_DIR": "/opt/esp/roms"}

_STT_NOTYPE = 0
_STT_FUNC = 2
_TEXT = 1
_SHN_ABS = 0xFFF1
_CODE = 0x1000
_CODE_SIZE = 0x100


def _write_elf(path: Path, symbols: list[tuple[str, int, int, int, int]]) -> Path:
    """Write a RISC-V ELF with one code segment and section, ``.text`` at _CODE, and global
    symbols given as (name, value, size, type, section index)."""
    strtab = b"\0"
    entries = [struct.pack("<IIIBBH", 0, 0, 0, 0, 0, 0)]
    for name, value, size, kind, shndx in symbols:
        entries.append(struct.pack("<IIIBBH", len(strtab), value, size, (1 << 4) | kind, 0, shndx))
        strtab += name.encode() + b"\0"
    symtab = b"".join(entries)
    shstrtab = b"\0.text\0.symtab\0.strtab\0.shstrtab\0"
    text_off = 0x100
    symtab_off = text_off + _CODE_SIZE
    strtab_off = symtab_off + len(symtab)
    shstrtab_off = strtab_off + len(strtab)
    shoff = (shstrtab_off + len(shstrtab) + 3) & ~3
    ident = b"\x7fELF" + bytes([1, 1, 1, 0]) + bytes(8)
    header = struct.pack("<16sHHIIIIIHHHHHH", ident, 2, 243, 1, _CODE, 52, shoff, 0, 52, 32, 1, 40, 5, 4)
    segment = struct.pack("<IIIIIIII", 1, text_off, _CODE, _CODE, _CODE_SIZE, _CODE_SIZE, 5, 4)
    sections = [
        bytes(40),
        struct.pack("<IIIIIIIIII", 1, 1, 6, _CODE, text_off, _CODE_SIZE, 0, 0, 4, 0),
        struct.pack("<IIIIIIIIII", 7, 2, 0, 0, symtab_off, len(symtab), 3, 1, 4, 16),
        struct.pack("<IIIIIIIIII", 15, 3, 0, 0, strtab_off, len(strtab), 0, 0, 1, 0),
        struct.pack("<IIIIIIIIII", 23, 3, 0, 0, shstrtab_off, len(shstrtab), 0, 0, 1, 0),
    ]
    data = bytearray(shoff + 40 * len(sections))
    data[0 : len(header)] = header
    data[52 : 52 + len(segment)] = segment
    data[strtab_off - len(symtab) : strtab_off] = symtab
    data[strtab_off:shstrtab_off] = strtab
    data[shstrtab_off : shstrtab_off + len(shstrtab)] = shstrtab
    data[shoff:] = b"".join(sections)
    path.write_bytes(bytes(data))
    return path


def test_layered_read_app_hit_short_circuits_rom() -> None:
    app = FakeImage(0x100, b"\xaa\xbb", name="app")
    rom_image = FakeImage(0x100, b"\xcc\xdd", name="rom")
    assert LayeredImage(app, rom_image).read(0x100, 2) == b"\xaa\xbb"


def test_layered_read_app_miss_falls_through_to_rom() -> None:
    app = FakeImage(name="app")
    rom_image = FakeImage(0x400, b"\x11\x22", name="rom")
    assert LayeredImage(app, rom_image).read(0x400, 2) == b"\x11\x22"


def test_layered_read_both_miss_raises_app_error() -> None:
    layered = LayeredImage(FakeImage(name="app"), FakeImage(name="rom"))
    with pytest.raises(CodeReadError, match="app 0x0000dead out of range"):
        layered.read(0xDEAD, 2)


def test_layered_read_no_rom_raises_app_error() -> None:
    with pytest.raises(CodeReadError, match="app 0x00000200 out of range"):
        LayeredImage(FakeImage(name="app"), None).read(0x200, 2)


def test_layered_symbol_app_hit_returns_bare_name() -> None:
    app = FakeImage(symbols={0x100: "app_foo"})
    rom_image = FakeImage(symbols={0x100: "rom_foo"})
    assert LayeredImage(app, rom_image).symbol_at(0x100) == "app_foo"


def test_layered_symbol_app_miss_falls_through_to_rom_with_prefix() -> None:
    rom_image = FakeImage(symbols={0x40000000: "esp_rom_intr_disable"})
    assert LayeredImage(FakeImage(), rom_image).symbol_at(0x40000000) == "rom:esp_rom_intr_disable"


def test_layered_symbol_both_miss_returns_none() -> None:
    assert LayeredImage(FakeImage(), FakeImage()).symbol_at(0xDEAD) is None


def test_layered_symbol_no_rom_returns_none_on_app_miss() -> None:
    assert LayeredImage(FakeImage(), None).symbol_at(0xDEAD) is None


def test_layered_resolver_app_hit_short_circuits_rom() -> None:
    app = FakeResolver({0x100: SourceLocation(file="a.c", line=10)})
    rom_resolver = FakeResolver({0x100: SourceLocation(file="rom.c", line=99)})
    assert LayeredResolver(app, rom_resolver).resolve(0x100) == SourceLocation(file="a.c", line=10)


def test_layered_resolver_app_miss_falls_through_to_rom() -> None:
    rom_resolver = FakeResolver({0x400: SourceLocation(file="rom.c", line=5)})
    assert LayeredResolver(FakeResolver(), rom_resolver).resolve(0x400) == SourceLocation(file="rom.c", line=5)


def test_layered_resolver_both_miss_returns_none() -> None:
    assert LayeredResolver(FakeResolver(), FakeResolver()).resolve(0xDEAD) is None


def test_layered_resolver_no_rom_returns_none_on_app_miss() -> None:
    assert LayeredResolver(FakeResolver(), None).resolve(0xDEAD) is None


@pytest.mark.parametrize(
    "addrs,sizes,section_ends,expected",
    [
        pytest.param([0x100, 0x180], [0, 0x10], [0x200, 0x200], [0x80, 0x10], id="extends_to_next_symbol"),
        pytest.param([0x100, 0x180], [0x10, 0], [0x200, 0x200], [0x10, 0x80], id="last_symbol_to_section_end"),
        pytest.param([0x100, 0x300], [0, 0], [0x200, 0x400], [0x100, 0x100], id="next_symbol_beyond_section_end"),
        pytest.param([0x100, 0x100], [0, 0x20], [0x200, 0x200], [0, 0x20], id="alias_of_sized_symbol_kept"),
        pytest.param([0x100], [0], [None], [0], id="unknown_section_kept"),
    ],
)
def test_fill_zero_sizes(addrs, sizes, section_ends, expected) -> None:
    assert _fill_zero_sizes(addrs, sizes, section_ends) == expected


def test_code_outside_functions_takes_the_label_before_it(tmp_path) -> None:
    elf = _write_elf(
        tmp_path / "app.elf",
        [
            ("func_a", _CODE, 0x10, _STT_FUNC, _TEXT),
            (".local_label", _CODE + 0x10, 0, _STT_NOTYPE, _TEXT),
            ("global_label", _CODE + 0x10, 0, _STT_NOTYPE, _TEXT),
            ("asm_entry", _CODE + 0x20, 0, _STT_NOTYPE, _TEXT),
            ("func_b", _CODE + 0x40, 0x10, _STT_FUNC, _TEXT),
            ("$x", _CODE + 0x60, 0, _STT_NOTYPE, _TEXT),
        ],
    )
    with ElfImage(elf) as image:
        assert image.symbol_at(_CODE + 4) == "func_a"
        # A label that is not a local one names the code better
        assert image.symbol_at(_CODE + 0x14) == "global_label"
        assert image.symbol_at(_CODE + 0x24) == "asm_entry"
        # func_b starts between asm_entry and the address
        assert image.symbol_at(_CODE + 0x50) is None
        # $x only marks code for disassemblers
        assert image.symbol_at(_CODE + 0x64) is None


def test_absolute_symbols_name_only_their_address(tmp_path) -> None:
    elf = _write_elf(
        tmp_path / "app.elf",
        [
            # A linker script assignment keeps the size of the libc qsort it replaces
            ("qsort", 0x2000, 0x300, _STT_FUNC, _SHN_ABS),
            ("Cache_Suspend", 0x2120, 0, _STT_NOTYPE, _SHN_ABS),
            ("asm_entry", _CODE, 0, _STT_NOTYPE, _TEXT),
            ("func_b", _CODE + 0x40, 0x10, _STT_FUNC, _TEXT),
            ("_text_start", _CODE + 0x80, 0, _STT_NOTYPE, _SHN_ABS),
        ],
    )
    with ElfImage(elf) as image:
        assert image.symbol_at(0x2000) == "qsort"
        assert image.symbol_at(0x2004) is None
        assert image.symbol_at(0x2120) == "Cache_Suspend"
        # An absolute label in the code of the ELF marks a section bound, not a function
        assert image.symbol_at(_CODE + 0x80) is None


@pytest.mark.parametrize(
    "env",
    [{"IDF_PATH": "", "ESP_ROM_ELF_DIR": "/x"}, {"IDF_PATH": "/x", "ESP_ROM_ELF_DIR": ""}],
    ids=["no_idf_path", "no_rom_dir"],
)
def test_rom_lookup_missing_env_reports_missing_env(env) -> None:
    with mock.patch.dict("os.environ", env):
        assert resolve_rom_elf("esp32c5", 100) == (None, RomStatus.MISSING_ENV)


def test_rom_lookup_found_file_reports_auto() -> None:
    fake = "/opt/esp/roms/esp32c5_rev100_rom.elf"
    with (
        mock.patch.dict("os.environ", _ROM_ENV),
        mock.patch.object(rom, "get_rom_elf_path", return_value=fake) as get,
        mock.patch("pathlib.Path.is_file", return_value=True),
    ):
        assert resolve_rom_elf("esp32c5", 100) == (Path(fake), RomStatus.AUTO)
    get.assert_called_once_with("esp32c5", 100)


def test_rom_lookup_no_roms_json_entry_reports_not_found() -> None:
    with mock.patch.dict("os.environ", _ROM_ENV), mock.patch.object(rom, "get_rom_elf_path", return_value=None):
        assert resolve_rom_elf("esp32unknown", 0) == (None, RomStatus.NOT_FOUND)


def test_rom_lookup_missing_file_reports_not_found() -> None:
    with (
        mock.patch.dict("os.environ", _ROM_ENV),
        mock.patch.object(rom, "get_rom_elf_path", return_value="/nope/rom.elf"),
        mock.patch("pathlib.Path.is_file", return_value=False),
    ):
        assert resolve_rom_elf("esp32c5", 100) == (None, RomStatus.NOT_FOUND)
