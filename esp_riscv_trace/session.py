# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Load a capture with its ELF files and prepare each core for decoding."""

from __future__ import annotations

import contextlib
from collections.abc import Callable, Iterator
from dataclasses import dataclass
from pathlib import Path

from esp_pylib.logger import log
from rich.markup import escape

from . import containers
from .containers.base import Capture, TraceBuffer
from .decoder import ring
from .decoder.params import DecoderProfile, resolve_profile
from .decoder.pipeline import decode_events
from .elf import CodeImage, SourceResolver
from .elf.dwarf import DwarfResolver
from .elf.image import ElfImage, ElfImageError
from .elf.layered import LayeredImage, LayeredResolver
from .elf.rom import (
    RomStatus,
    resolve_rom_elf,
)
from .errors import EspRiscvTraceError
from .exporters._format import short_sha
from .ir import (
    TraceCaptureNote,
    TraceEvent,
)
from .ir.capture_note import ElfShaStatus, InferredWrapState
from .snapshot import abi
from .snapshot.chip import chip_name_from_id, is_riscv_trace_chip

# Receives each warning as plain text
WarnSink = Callable[[str], None]


def _log_warning(message: str) -> None:
    log.warn(escape(message))


class UsageError(EspRiscvTraceError):
    """A missing or contradicting option, reported as a usage problem."""


class ShaMismatch(EspRiscvTraceError):
    """The ELF SHA-256 differs from the one the capture recorded for a core."""

    def __init__(self, core_id: int, snapshot_sha: bytes | None, elf_sha: bytes | None) -> None:
        self.core_id = core_id
        self.snapshot_sha = snapshot_sha
        self.elf_sha = elf_sha
        super().__init__(
            f"ELF SHA-256 mismatch for core {core_id}. snapshot={short_sha(snapshot_sha)} elf={short_sha(elf_sha)}"
        )


class _StallImage:
    """Code image used without ``--elf``.

    Every read returns ``C.JR ra``, an uninferable instruction, so the walker only advances on packet addresses.
    """

    _INSN = bytes([0x82, 0x80])

    def read(self, vaddr: int, size: int) -> bytes:
        return (self._INSN * ((size + 1) // 2))[:size]

    def symbol_at(self, vaddr: int) -> str | None:
        return None


@dataclass
class Session:
    """A loaded capture with its code image and source resolver."""

    cap: Capture
    chip: str
    elf: ElfImage | None
    image: CodeImage
    resolver: SourceResolver | None
    force_elf: bool
    rom_elf_path: str | None = None
    rom_elf_status: str | None = None


@dataclass
class CoreView:
    """One core of a session, prepared for decoding."""

    session: Session
    core: TraceBuffer
    profile: DecoderProfile
    note: TraceCaptureNote
    linear: ring.LinearizedBuffer

    def events(self) -> Iterator[TraceEvent]:
        """Yield the reconstructed events of this core."""
        address_mode = abi.AddressMode(self.core.address_mode)
        return decode_events(self.linear, self.profile, self.session.image, address_mode)


@contextlib.contextmanager
def open_session(
    capture: Path,
    *,
    elf: Path | None = None,
    rom_elf: Path | None = None,
    chip: str | None = None,
    force_elf: bool = False,
    require_elf: bool = False,
    warn: WarnSink = _log_warning,
) -> Iterator[Session]:
    """Check the options, load ``capture`` and its ELF files, and yield the session.

    Problems that do not stop decoding go to ``warn``, which logs them by default.
    """
    if not capture.exists():
        raise FileNotFoundError(f"capture not found: {capture}")
    if require_elf and elf is None:
        raise UsageError("--elf is required for this command")
    if elf is not None and not elf.exists():
        raise FileNotFoundError(f"ELF not found: {elf}")
    if rom_elf is not None and elf is None:
        raise UsageError("--rom-elf requires --elf")

    cap = containers.load(capture)
    if cap.fifo_emptied is False:
        warn("the encoder FIFO did not empty at stop, the end of the trace may be missing")
    if cap.abi_minor > abi.ABI_MINOR:
        warn(
            f"capture uses snapshot ABI {cap.abi_major}.{cap.abi_minor}, this decoder supports "
            f"{abi.ABI_MAJOR}.{abi.ABI_MINOR}. Newer fields are ignored, update esp-riscv-trace"
        )
    chip_name = _resolve_chip(chip, cap, warn)
    if chip_name is None:
        raise UsageError("chip is not known. Pass --chip <name> or use a container that records target_id.")

    with contextlib.ExitStack() as stack:
        app: ElfImage | None = None
        image: CodeImage = _StallImage()
        resolver: SourceResolver | None = None
        rom_path: str | None = None
        rom_status: str | None = None
        if elf is not None:
            app = stack.enter_context(ElfImage(elf))
            app_resolver = _open_dwarf(app)
            image, resolver = app, app_resolver
            rom, rom_resolver, rom_path, rom_status = _open_rom_elf(stack, rom_elf, chip_name, cap, warn)
            if rom is not None and rom_resolver is not None:
                image = LayeredImage(app, rom)
                resolver = LayeredResolver(app_resolver, rom_resolver)
        yield Session(
            cap=cap,
            chip=chip_name,
            elf=app,
            image=image,
            resolver=resolver,
            force_elf=force_elf,
            rom_elf_path=rom_path,
            rom_elf_status=rom_status,
        )


def iter_cores(session: Session, core_filter: int | None) -> Iterator[CoreView]:
    """Yield one view per selected core in container order, raising ``ShaMismatch`` unless forced."""
    for core in session.cap.cores:
        if core_filter is not None and core.core_id != core_filter:
            continue
        profile = resolve_profile(session.chip, core.encoder_params)
        linear = ring.linearize(core)
        note = build_capture_note(session, core, profile, linear)
        if not session.force_elf and note.elf_sha_status == ElfShaStatus.MISMATCH:
            raise ShaMismatch(core.core_id, note.snapshot_app_elf_sha256, note.elf_sha256)
        yield CoreView(session=session, core=core, profile=profile, note=note, linear=linear)


def build_capture_note(
    session: Session,
    core: TraceBuffer,
    profile: DecoderProfile,
    linear: ring.LinearizedBuffer | None = None,
) -> TraceCaptureNote:
    """Describe the capture, the core and the ELF check for the output header."""
    capture = session.cap
    snapshot_sha = capture.app_elf_sha256
    elf_sha: bytes | None = None
    elf_path: str | None = None
    if session.elf is None:
        status = ElfShaStatus.NO_ELF
    else:
        elf_sha = session.elf.sha256()
        elf_path = str(session.elf.path)
        if snapshot_sha is None:
            status = ElfShaStatus.NO_SNAPSHOT_HASH
        elif elf_sha == snapshot_sha:
            status = ElfShaStatus.MATCH
        else:
            status = ElfShaStatus.MISMATCH
    wrap_state, wrap_reason = _wrap_state_view(core.wrap_state, linear)
    return TraceCaptureNote(
        source=str(capture.source_path),
        chip=session.chip,
        core_id=core.core_id,
        packet_format=abi.enum_name(abi.PacketFormat, profile.packet_format),
        address_mode=abi.enum_name(abi.AddressMode, core.address_mode),
        memory_mode=abi.enum_name(abi.MemoryMode, core.memory_mode),
        resync_mode=abi.enum_name(abi.ResyncMode, core.resync_mode),
        resync_threshold=core.resync_threshold,
        wrap_state=wrap_state,
        capacity=core.capacity,
        head_offset=core.head_offset,
        fifo_status_raw=core.fifo_status_raw,
        intr_status_raw=core.intr_status_raw,
        fifo_empty=core.fifo_empty,
        memory_full=core.memory_full,
        fifo_overflow=core.fifo_overflow,
        capture_reason=abi.enum_name(abi.CaptureReason, capture.capture_reason),
        owner=abi.enum_name(abi.Owner, core.owner),
        target_id=capture.target_id,
        chip_revision=capture.chip_revision,
        snapshot_app_elf_sha256=snapshot_sha,
        elf_path=elf_path,
        elf_sha256=elf_sha,
        elf_sha_status=status,
        wrap_reason=wrap_reason,
        rom_elf_path=session.rom_elf_path,
        rom_elf_status=session.rom_elf_status,
        fifo_emptied=capture.fifo_emptied,
        abi_major=capture.abi_major,
        abi_minor=capture.abi_minor,
    )


def _resolve_chip(chip: str | None, cap: Capture, warn: WarnSink) -> str | None:
    recorded = chip_name_from_id(cap.target_id) if cap.target_id is not None else None
    if chip is None:
        return recorded
    name = chip.strip().lower()
    if not is_riscv_trace_chip(name):
        raise UsageError(f"chip {chip!r} has no supported trace encoder")
    if recorded is not None and name != recorded:
        warn(f"--chip {name} overrides the chip recorded in the capture ({recorded})")
    return name


def _open_rom_elf(
    stack: contextlib.ExitStack,
    rom_elf: Path | None,
    chip: str,
    cap: Capture,
    warn: WarnSink,
) -> tuple[ElfImage | None, DwarfResolver | None, str | None, RomStatus]:
    """Open the ROM ELF from ``--rom-elf`` or the ESP-IDF environment, warn when it is unavailable."""
    revision = cap.chip_revision or 0
    path: Path | None
    if rom_elf is not None:
        path = rom_elf
        status = RomStatus.PROVIDED if rom_elf.is_file() else RomStatus.NOT_FOUND
    else:
        path, status = resolve_rom_elf(chip, revision)
    path_str = str(path) if path is not None else None
    if status == RomStatus.MISSING_ENV:
        warn("ROM ELF not searched, IDF_PATH or ESP_ROM_ELF_DIR is not set. Pass --rom-elf to use one")
        return None, None, None, status
    if status == RomStatus.NOT_FOUND or path is None:
        where = path_str if path_str is not None else f"{chip} revision {revision}"
        warn(f"ROM ELF not found for {where}, continuing without it")
        return None, None, path_str, status
    try:
        rom = stack.enter_context(ElfImage(path))
        return rom, _open_dwarf(rom), path_str, status
    except ElfImageError as exc:
        warn(f"{exc}, continuing without ROM ELF")
        return None, None, path_str, RomStatus.INVALID


def _open_dwarf(image: ElfImage) -> DwarfResolver:
    try:
        return DwarfResolver(image)
    except Exception as exc:
        raise ElfImageError(f"cannot parse DWARF from {image.path}") from exc


def _wrap_state_view(raw_wrap_state: int, linear: ring.LinearizedBuffer | None) -> tuple[str, str | None]:
    """Return the wrap state shown in the header and the reason when it was inferred."""
    raw_name = abi.enum_name(abi.WrapState, raw_wrap_state)
    if raw_wrap_state != int(abi.WrapState.UNKNOWN) or linear is None:
        return raw_name, None
    if not linear.wrap_state_inferred:
        return abi.enum_name(abi.WrapState, linear.wrap_state), None
    if linear.wrap_state == int(abi.WrapState.WRAPPED):
        return InferredWrapState.INFERRED_WRAPPED, linear.inference_reason
    if linear.wrap_state == int(abi.WrapState.NOT_WRAPPED):
        return InferredWrapState.INFERRED_NOT_WRAPPED, linear.inference_reason
    return raw_name, None


__all__ = [
    "CoreView",
    "Session",
    "ShaMismatch",
    "UsageError",
    "WarnSink",
    "build_capture_note",
    "iter_cores",
    "open_session",
]
