# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""ELF binary reader for executable instructions and symbols."""

from __future__ import annotations

import bisect
import contextlib
import hashlib
import io
from dataclasses import dataclass
from functools import cached_property, lru_cache
from pathlib import Path

from elftools.elf.elffile import ELFFile
from elftools.elf.sections import SymbolTableSection

from ..errors import EspRiscvTraceError
from . import CodeReadError
from .intervals import IntervalIndex

_SYMBOL_CACHE_SIZE = 4096

# PT_LOAD segment flag values from ELF specification.
_PF_X = 0x1

# sh_flags bit of a section that holds code
_SHF_EXECINSTR = 0x4


class ElfImageError(EspRiscvTraceError):
    """Raised when an ELF cannot be opened or parsed."""


@dataclass(frozen=True)
class _ExecRegion:
    """One contiguous file-backed executable range from a PT_LOAD segment."""

    vaddr: int
    size: int
    data: bytes

    def contains(self, addr: int, n: int) -> bool:
        return self.vaddr <= addr and addr + n <= self.vaddr + self.size


class ElfImage:
    """Executable-code and symbol view of a single application ELF."""

    def __init__(self, path: str | Path) -> None:
        self._path = Path(path)
        # Load the ELF into memory to keep reads stable across file changes.
        try:
            self._bytes: bytes = self._path.read_bytes()
        except OSError as exc:
            raise ElfImageError(f"cannot open ELF {self._path}") from exc
        self._file = io.BytesIO(self._bytes)
        try:
            try:
                self._elf = ELFFile(self._file)
                self._regions: list[_ExecRegion] = self._collect_exec_regions()
                self._region_starts: list[int] = [r.vaddr for r in self._regions]
                self._symbols: IntervalIndex[str] = self._load_symbols()
                self._labels: list[tuple[int, int, str]] = self._load_labels()
                self._label_starts: list[int] = [addr for addr, _end, _name in self._labels]
            except ElfImageError:
                raise
            except Exception as exc:
                raise ElfImageError(f"cannot parse ELF {self._path}") from exc
        except BaseException:
            self._file.close()
            raise
        self._sha256: bytes | None = None
        self._symbol_lookup = lru_cache(maxsize=_SYMBOL_CACHE_SIZE)(self._lookup)

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dwarf_info(self):
        """pyelftools ``DWARFInfo`` for this ELF, or ``None`` if absent."""
        if self._elf.has_dwarf_info():
            return self._elf.get_dwarf_info()
        return None

    def sha256(self) -> bytes:
        """SHA-256 of the ELF bytes this instance is bound to."""
        if self._sha256 is None:
            self._sha256 = hashlib.sha256(self._bytes).digest()
        return self._sha256

    def read(self, vaddr: int, size: int) -> bytes:
        """Return ``size`` bytes from executable memory at ``vaddr``.

        Raises ``CodeReadError`` if the range is not fully covered by an
        executable PT_LOAD segment. The reconstructor treats such a raise
        as an explicit gap and consumes a trace address packet to resync.
        """
        if size <= 0:
            raise CodeReadError(f"non-positive size {size}")
        if not self._regions:
            raise CodeReadError(f"vaddr 0x{vaddr:08x} out of range, ELF has no executable segments")
        idx = bisect.bisect_right(self._region_starts, vaddr) - 1
        if idx < 0:
            raise CodeReadError(f"vaddr 0x{vaddr:08x}+{size} below first executable segment")
        region = self._regions[idx]
        if not region.contains(vaddr, size):
            raise CodeReadError(
                f"vaddr 0x{vaddr:08x}+{size} not fully inside executable segment "
                f"[0x{region.vaddr:08x}, 0x{region.vaddr + region.size:08x})"
            )
        off = vaddr - region.vaddr
        return region.data[off : off + size]

    def symbol_at(self, vaddr: int) -> str | None:
        """Return the name of the STT_FUNC that covers ``vaddr``, or ``None``.

        Walks back through sorted symbols so nested or shadowed function
        ranges (a common pattern for ROM aliases) are still resolved. The
        walk stops as soon as no earlier symbol can reach ``vaddr``.

        Code outside every function takes the name of the nearest label before it in
        the same section, as GDB names it, when no function starts in between. Assembly
        routines without a ``.type`` directive have only such a label.
        """
        return self._symbol_lookup(vaddr)

    def data_at(self, vaddr: int) -> str | None:
        """Return the variable at ``vaddr``, as ``name`` or ``name+offset``, or ``None``.

        Sized STT_OBJECT symbols name variables, which code runs in when a jump goes into a buffer.
        """
        starts, objects = self._objects
        i = bisect.bisect_right(starts, vaddr) - 1
        if i < 0:
            return None
        start, size, name = objects[i]
        if vaddr >= start + size:
            return None
        return name if vaddr == start else f"{name}+{vaddr - start}"

    def close(self) -> None:
        with contextlib.suppress(OSError):
            self._file.close()

    def __enter__(self) -> ElfImage:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _collect_exec_regions(self) -> list[_ExecRegion]:
        regions: list[_ExecRegion] = []
        for seg in self._elf.iter_segments():
            if seg.header["p_type"] != "PT_LOAD":
                continue
            if not (seg.header["p_flags"] & _PF_X):
                continue
            vaddr = seg.header["p_vaddr"]
            filesz = seg.header["p_filesz"]
            memsz = seg.header["p_memsz"]
            covered = min(filesz, memsz)
            if covered == 0:
                continue
            data = bytes(seg.data()[:covered])
            if len(data) != covered:
                raise ElfImageError(
                    f"executable segment at 0x{vaddr:08x} is truncated (got {len(data)} bytes, expected {covered})"
                )
            regions.append(_ExecRegion(vaddr=vaddr, size=covered, data=data))
        regions.sort(key=lambda r: r.vaddr)
        for i in range(len(regions) - 1):
            end = regions[i].vaddr + regions[i].size
            if end > regions[i + 1].vaddr:
                raise ElfImageError(f"overlapping executable segments at 0x{regions[i + 1].vaddr:08x}")
        return regions

    def _lookup(self, vaddr: int) -> str | None:
        name = self._symbols.lookup(vaddr)
        if name is not None:
            return name
        i = bisect.bisect_right(self._label_starts, vaddr) - 1
        if i < 0:
            return None
        start, end, label = self._labels[i]
        if vaddr >= end:
            return None
        function = self._symbols.start_before(vaddr)
        if function is not None and function > start:
            return None
        return label

    def _iter_symbols(self):
        for sec in self._elf.iter_sections():
            if isinstance(sec, SymbolTableSection):
                yield from sec.iter_symbols()

    def _load_symbols(self) -> IntervalIndex[str]:
        addrs: list[int] = []
        sizes: list[int] = []
        names: list[str] = []
        limits: list[int | None] = []
        for sym in self._iter_symbols():
            kind = sym.entry["st_info"]["type"]
            # The linker scripts of ESP-IDF name ROM functions with absolute symbols. A
            # script assignment keeps the type and size of the definition it replaces,
            # such as the 766 bytes of the qsort of libc, so they name only their address.
            absolute = sym.entry["st_shndx"] == "SHN_ABS"
            if kind != "STT_FUNC" and not (absolute and kind == "STT_NOTYPE"):
                continue
            name = sym.name
            if not name:
                continue
            addr = sym.entry["st_value"]
            size = 0 if absolute else sym.entry["st_size"]
            # Absolute labels in the code of the ELF mark section bounds, as _iram_text_start does
            if addr == 0 or (absolute and kind == "STT_NOTYPE" and self._in_code(addr)):
                continue
            addrs.append(addr)
            sizes.append(size)
            names.append(name)
            limits.append(self._section_end(sym.entry["st_shndx"]))
        # Sort sized symbols after size 0 aliases for the same address.
        order = sorted(
            range(len(addrs)),
            key=lambda i: (addrs[i], 1 if sizes[i] else 0),
        )
        sorted_addrs = [addrs[i] for i in order]
        sorted_sizes = _fill_zero_sizes(sorted_addrs, [sizes[i] for i in order], [limits[i] for i in order])
        # A size 0 symbol covers only its own address
        return IntervalIndex(
            [
                (addr, addr + (size or 1), names[i])
                for addr, size, i in zip(sorted_addrs, sorted_sizes, order, strict=True)
            ]
        )

    @cached_property
    def _objects(self) -> tuple[list[int], list[tuple[int, int, str]]]:
        """The variables as (address, size, name) sorted, and their addresses. Loaded on the first lookup."""
        objects = sorted(
            (sym.entry["st_value"], sym.entry["st_size"], sym.name)
            for sym in self._iter_symbols()
            # Absolute symbols keep the size of the definition a linker script assignment replaced
            if sym.entry["st_info"]["type"] == "STT_OBJECT"
            and sym.entry["st_size"]
            and sym.name
            and sym.entry["st_shndx"] != "SHN_ABS"
        )
        return [addr for addr, _size, _name in objects], objects

    def _load_labels(self) -> list[tuple[int, int, str]]:
        """Labels in code sections outside every function, as (address, section end, name), sorted."""
        labels: dict[int, tuple[int, str]] = {}
        for sym in self._iter_symbols():
            if sym.entry["st_info"]["type"] != "STT_NOTYPE":
                continue
            name = sym.name
            # $x and $d only mark code and data for disassemblers
            if not name or name.startswith("$"):
                continue
            addr = sym.entry["st_value"]
            shndx = sym.entry["st_shndx"]
            end = self._section_end(shndx)
            if addr == 0 or end is None or not self._holds_code(shndx) or self._symbols.lookup(addr) is not None:
                continue
            # A label that is not a local one, as ".skip_restore" is, names the code better
            if addr not in labels or (labels[addr][1].startswith(".") and not name.startswith(".")):
                labels[addr] = (end, name)
        return [(addr, end, name) for addr, (end, name) in sorted(labels.items())]

    def _in_code(self, vaddr: int) -> bool:
        idx = bisect.bisect_right(self._region_starts, vaddr) - 1
        return idx >= 0 and self._regions[idx].contains(vaddr, 1)

    def _holds_code(self, shndx) -> bool:
        return bool(self._elf.get_section(shndx).header["sh_flags"] & _SHF_EXECINSTR)

    def _section_end(self, shndx) -> int | None:
        if not isinstance(shndx, int) or not 0 < shndx < self._elf.num_sections():
            return None
        hdr = self._elf.get_section(shndx).header
        return int(hdr["sh_addr"] + hdr["sh_size"])


def _fill_zero_sizes(addrs: list[int], sizes: list[int], limits: list[int | None]) -> list[int]:
    """Extend size 0 symbols up to the next symbol start or their section end."""
    sized_addrs = {a for a, s in zip(addrs, sizes, strict=False) if s}
    out = list(sizes)
    for i, (addr, size, limit) in enumerate(zip(addrs, sizes, limits, strict=False)):
        if size or limit is None or addr in sized_addrs:
            continue
        j = bisect.bisect_right(addrs, addr)
        end = min(addrs[j], limit) if j < len(addrs) else limit
        if end > addr:
            out[i] = end - addr
    return out


__all__ = [
    "ElfImage",
    "ElfImageError",
]
