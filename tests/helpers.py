# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Shared fakes, builders, paths and CLI runners for the tests."""

from __future__ import annotations

import contextlib
import dataclasses
import functools
import io
import os
import struct
import subprocess
import sys
from pathlib import Path
from unittest import mock

from esp_riscv_trace import cli
from esp_riscv_trace.containers.base import TraceBuffer
from esp_riscv_trace.decoder.framing import Frame
from esp_riscv_trace.decoder.packets import SyncPacket
from esp_riscv_trace.elf import CodeReadError
from esp_riscv_trace.elf.dwarf import SourceLocation
from esp_riscv_trace.ir.capture_note import ElfShaStatus, TraceCaptureNote
from esp_riscv_trace.snapshot import abi

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
APPS = TESTS / "built_apps"

C5_STOP = TESTS / "esp32c5" / "stop.esptrace"
C5_ELF = APPS / "esp32c5.elf"
S31_ELF = APPS / "esp32s31.elf"

BASE = 0x40800000
NOP = bytes.fromhex("13000000")
JALR_X0_RA = bytes.fromhex("67800000")

# Environment that turns on ROM ELF auto-detection
IDF_ENV_VARS = ("IDF_PATH", "ESP_ROM_ELF_DIR")


class FakeImage:
    """Code image backed by one byte range and an optional symbol table."""

    def __init__(
        self, base: int = 0, data: bytes = b"", symbols: dict[int, str] | None = None, *, name: str = "image"
    ) -> None:
        self._base = base
        self._data = data
        self._symbols = dict(symbols or {})
        self._name = name

    def read(self, vaddr: int, size: int) -> bytes:
        off = vaddr - self._base
        if off < 0 or off + size > len(self._data):
            raise CodeReadError(f"{self._name} 0x{vaddr:08x} out of range")
        return self._data[off : off + size]

    def symbol_at(self, vaddr: int) -> str | None:
        return self._symbols.get(vaddr)


class FakeResolver:
    """Source resolver backed by a PC to location dict."""

    def __init__(self, locs: dict[int, SourceLocation] | None = None) -> None:
        self._locs = dict(locs or {})

    def resolve(self, pc: int) -> SourceLocation | None:
        return self._locs.get(pc)


def make_frame(index: int = 0, payload: bytes = b"") -> Frame:
    """Frame whose stream offset and packet index are both ``index``."""
    return Frame(segment_index=0, stream_offset=index, length=len(payload), packet_index=index, payload=payload)


def make_sync(pc: int, *, branch: int) -> SyncPacket:
    """Sync packet at ``pc`` in frame 0."""
    return SyncPacket(frame=make_frame(0), branch=branch, privilege=0, raw_address=pc >> 1, pc=pc)


def pack_bits(*fields: tuple[int, int]) -> bytes:
    """Pack ``(value, width)`` fields LSB first into the fewest whole bytes."""
    value = 0
    position = 0
    for field, width in fields:
        value |= field << position
        position += width
    return value.to_bytes((position + 7) // 8, "little")


def encode_jal(rd: int, imm: int) -> bytes:
    """Encode a 32-bit JAL with a signed byte offset."""
    assert imm % 2 == 0
    u = imm & 0x1FFFFF
    inst = (
        (((u >> 20) & 1) << 31)
        | (((u >> 1) & 0x3FF) << 21)
        | (((u >> 11) & 1) << 20)
        | (((u >> 12) & 0xFF) << 12)
        | (rd << 7)
        | 0x6F
    )
    return inst.to_bytes(4, "little")


def make_buffer(
    raw: bytes,
    *,
    head: int,
    memory_mode: int = int(abi.MemoryMode.LOOP),
    wrap_state: int = int(abi.WrapState.WRAPPED),
    fifo_overflow: bool = False,
    head_valid: bool = True,
) -> TraceBuffer:
    """Frozen ET 2.0 core 0 buffer holding ``raw`` with its write head at ``head``."""
    return TraceBuffer(
        core_id=0,
        raw=raw,
        capacity=len(raw),
        head_offset=head,
        memory_mode=memory_mode,
        wrap_state=wrap_state,
        packet_format=int(abi.PacketFormat.EFFICIENT_TRACE_2_0),
        address_mode=int(abi.AddressMode.FULL),
        fifo_status_raw=0,
        intr_status_raw=0,
        owner=int(abi.Owner.FW),
        state=int(abi.State.FROZEN),
        head_valid=head_valid,
        encoder_params=None,
        segment_index=None,
        fifo_overflow=fifo_overflow,
    )


def pack_struct(fmt: str, defaults: dict[str, object], overrides: dict[str, object]) -> bytes:
    """Pack ``defaults`` with ``overrides`` in the key order of ``defaults``."""
    unknown = set(overrides) - set(defaults)
    assert not unknown, f"unknown fields {sorted(unknown)}"
    values = {**defaults, **overrides}
    return struct.pack(fmt, *values.values())


# Wire order of each ABI struct with valid default values
SNAPSHOT_DESC_DEFAULTS: dict[str, object] = dict(
    magic=abi.MAGIC,
    write_seq=8,
    cores_addr=0x4080C244,
    app_elf_sha256_addr=0x4080C208,
    encoder_params_addr=0x4080C228,
    target_id=0x0017,
    chip_revision=0x0065,
    abi_major=abi.ABI_MAJOR,
    abi_minor=abi.ABI_MINOR,
    snapshot_desc_size=abi.SNAPSHOT_DESC_SIZE,
    core_desc_size=abi.CORE_DESC_SIZE,
    core_count=1,
    capture_reason=int(abi.CaptureReason.EXPLICIT_STOP),
    app_elf_sha256_size=32,
    encoder_params_size=abi.ENCODER_PARAMS_SIZE,
)

CORE_DESC_DEFAULTS: dict[str, object] = dict(
    buffer_addr=0x4080D714,
    capacity=16,
    head_offset=4,
    resync_threshold=128,
    fifo_status_raw=1,
    intr_status_raw=0,
    core_id=0,
    state=int(abi.State.FROZEN),
    memory_mode=int(abi.MemoryMode.LINEAR),
    packet_format=int(abi.PacketFormat.EFFICIENT_TRACE_2_0),
    address_mode=int(abi.AddressMode.FULL),
    resync_mode=int(abi.ResyncMode.PACKET),
    owner=int(abi.Owner.FW),
    head_valid=1,
    fifo_empty=1,
    memory_full=0,
    fifo_overflow=0,
    reserved=0,
)

ENCODER_PARAMS_DEFAULTS: dict[str, object] = {
    **{f.name: 0 for f in dataclasses.fields(abi.EncoderParams)},
    "params_version": abi.ENCODER_PARAMS_VERSION,
    "params_size": abi.ENCODER_PARAMS_SIZE,
    "iaddress_lsb_p": 1,
    "iaddress_width_p": 32,
}


def pack_snapshot_desc(**overrides) -> bytes:
    """Snapshot descriptor bytes with ``overrides`` applied."""
    return pack_struct(abi._SNAPSHOT_DESC_FMT, SNAPSHOT_DESC_DEFAULTS, overrides)


def pack_core_desc(**overrides) -> bytes:
    """Core descriptor bytes with ``overrides`` applied."""
    return pack_struct(abi._CORE_DESC_FMT, CORE_DESC_DEFAULTS, overrides)


def pack_encoder_params(**overrides) -> bytes:
    """Encoder parameter bytes with ``overrides`` applied."""
    return pack_struct(abi._ENCODER_PARAMS_FMT, ENCODER_PARAMS_DEFAULTS, overrides)


_NOTE_DEFAULTS = dict(
    source="test",
    chip="esp32c5",
    target_id=0x0017,
    chip_revision=0x0064,
    core_id=0,
    owner="fw",
    packet_format="efficient_trace_2_0",
    address_mode="full",
    memory_mode="loop",
    resync_mode="packet",
    resync_threshold=32,
    wrap_state="wrapped",
    capacity=40,
    head_offset=16,
    fifo_status_raw=0,
    intr_status_raw=0,
    fifo_empty=True,
    memory_full=False,
    fifo_overflow=False,
    capture_reason="panic",
    elf_sha_status=ElfShaStatus.MATCH,
    elf_sha256=None,
    snapshot_app_elf_sha256=None,
    elf_path=None,
    wrap_reason=None,
)


def make_note(**overrides) -> TraceCaptureNote:
    """ESP32-C5 capture note with ``overrides`` applied."""
    return TraceCaptureNote(**{**_NOTE_DEFAULTS, **overrides})


@dataclasses.dataclass(frozen=True)
class CliResult:
    returncode: int
    stdout: str
    stderr: str


def _check_exit(result: CliResult, expect: int | None) -> CliResult:
    if expect is not None:
        assert result.returncode == expect, (
            f"expected exit {expect}, got {result.returncode}\nstdout=\n{result.stdout}\nstderr=\n{result.stderr}"
        )
    return result


@contextlib.contextmanager
def _cli_env(idf_env: bool):
    with mock.patch.dict(os.environ):
        if not idf_env:
            for name in IDF_ENV_VARS:
                os.environ.pop(name, None)
        yield


@functools.cache
def _run_main_cached(args: tuple[str, ...], idf_env: bool) -> CliResult:
    out, err = io.StringIO(), io.StringIO()
    with _cli_env(idf_env), contextlib.redirect_stdout(out), contextlib.redirect_stderr(err):
        try:
            rc = cli.main(list(args))
        except SystemExit as exc:
            rc = exc.code if isinstance(exc.code, int) else cli.EXIT_USAGE
    return CliResult(rc, out.getvalue(), err.getvalue())


def run_main(*args, idf_env: bool = False, expect: int | None = None) -> CliResult:
    """Run ``cli.main`` in process, cached by arguments, without the ESP-IDF environment unless ``idf_env``."""
    return _check_exit(_run_main_cached(tuple(str(a) for a in args), idf_env), expect)


@functools.cache
def _run_cli_cached(args: tuple[str, ...], cwd: Path, idf_env: bool) -> CliResult:
    with _cli_env(idf_env):
        env = dict(os.environ)
    p = subprocess.run(
        [sys.executable, "-m", "esp_riscv_trace", *args],
        capture_output=True,
        text=True,
        cwd=str(cwd),
        env=env,
        check=False,
    )
    return CliResult(p.returncode, p.stdout, p.stderr)


def run_cli(*args, cwd: Path = ROOT, idf_env: bool = False, expect: int | None = None) -> CliResult:
    """Run ``python -m esp_riscv_trace`` in a subprocess, cached by cwd, arguments and ``idf_env``."""
    return _check_exit(_run_cli_cached(tuple(str(a) for a in args), Path(cwd), idf_env), expect)
