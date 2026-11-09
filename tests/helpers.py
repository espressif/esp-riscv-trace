# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Shared builders and paths for the tests."""

from __future__ import annotations

import dataclasses
import struct
from pathlib import Path

from esp_riscv_trace.snapshot import abi

ROOT = Path(__file__).resolve().parent.parent
TESTS = ROOT / "tests"
APPS = TESTS / "built_apps"

C5_STOP = TESTS / "esp32c5" / "stop.esptrace"
C5_ELF = APPS / "esp32c5.elf"


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
