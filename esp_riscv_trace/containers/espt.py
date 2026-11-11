# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Loader for the OpenOCD ``esp riscv_trace`` ``.esptrace`` file (ESPT container).

File layout produced by ``src/target/espressif/esp_riscv_trace.c`` (little-endian)::

    header  (hdr_size bytes)
        u32 magic "ESPT", u16 version, u16 hdr_size
        RVTS snapshot descriptor   (32 bytes, raw target copy)
        encoder parameters         (28 bytes, raw target copy)
        app ELF SHA-256            (32 bytes, zero when not published)
    records, until END
        u8 type, u8 core, u16 flags, u32 len, then len payload bytes
        CORE_DESC  raw RVTS core descriptor (36 bytes)
        DATA       raw trace buffer of the core named by the last CORE_DESC
        END        u32 status: 0 OK, 1 FIFO not empty at stop
"""

from __future__ import annotations

import struct
from pathlib import Path

from ..snapshot import abi
from .base import Capture, ContainerError, TraceBuffer

MAGIC = b"ESPT"
VERSION = 1

PREAMBLE = struct.Struct("<4sHH")
RECORD_HEADER = struct.Struct("<BBHI")

OFF_SNAPSHOT = PREAMBLE.size
OFF_PARAMS = OFF_SNAPSHOT + abi.SNAPSHOT_DESC_SIZE
OFF_SHA256 = OFF_PARAMS + abi.ENCODER_PARAMS_SIZE
SHA256_SIZE = 32
HDR_SIZE = OFF_SHA256 + SHA256_SIZE

REC_CORE_DESC = 1
REC_DATA = 2
REC_END = 3
REC_NO_CORE = 0xFF

END_OK = 0
END_FIFO_NOT_EMPTY = 1


def sniff(head_bytes: bytes) -> bool:
    """Return True when ``head_bytes`` starts with the ESPT magic."""
    return head_bytes.startswith(MAGIC)


def load(path: str | Path) -> Capture:
    p = Path(path)
    data = p.read_bytes()
    if len(data) < HDR_SIZE:
        raise ContainerError(f"{p} is only {len(data)} bytes, need at least {HDR_SIZE} for the ESPT header")

    magic, version, hdr_size = PREAMBLE.unpack_from(data, 0)
    if magic != MAGIC:
        raise ContainerError(f"{p} does not start with the ESPT magic")
    if version != VERSION:
        raise ContainerError(f"{p} has ESPT version {version}, this decoder supports {VERSION}")
    if hdr_size < HDR_SIZE or hdr_size > len(data):
        raise ContainerError(f"{p} has invalid ESPT header size {hdr_size}")

    try:
        snap = abi.SnapshotDesc.parse(data[OFF_SNAPSHOT:OFF_PARAMS])
        params = abi.EncoderParams.parse(data[OFF_PARAMS:OFF_SHA256])
    except abi.AbiError as e:
        raise ContainerError(f"{p} has an invalid ESPT header. {e}") from e

    sha_bytes = None
    if snap.app_elf_sha256_size == SHA256_SIZE:
        sha_bytes = bytes(data[OFF_SHA256:HDR_SIZE])

    buffers: list[TraceBuffer] = []
    core_desc = None
    end_status = None
    off = hdr_size
    while off < len(data):
        if end_status is not None:
            raise ContainerError(f"{p} has data after the END record at offset {off}")
        if len(data) - off < RECORD_HEADER.size:
            raise ContainerError(f"{p} truncated in a record header at offset {off}")
        rtype, core, _flags, length = RECORD_HEADER.unpack_from(data, off)
        payload_off = off + RECORD_HEADER.size
        payload = data[payload_off : payload_off + length]
        if len(payload) != length:
            raise ContainerError(f"{p} record type {rtype} at offset {off} is truncated")
        off = payload_off + length

        if rtype == REC_CORE_DESC:
            try:
                core_desc = abi.CoreDesc.parse(payload)
            except abi.AbiError as e:
                raise ContainerError(f"{p} core {core} descriptor invalid. {e}") from e
            if core_desc.core_id != core:
                raise ContainerError(f"{p} CORE_DESC record core {core} holds core_id {core_desc.core_id}")
        elif rtype == REC_DATA:
            if core_desc is None or core_desc.core_id != core:
                raise ContainerError(f"{p} DATA record for core {core} has no matching CORE_DESC")
            if length != core_desc.capacity:
                raise ContainerError(
                    f"{p} core {core} DATA is {length} bytes, descriptor capacity is {core_desc.capacity}"
                )
            buffers.append(TraceBuffer.from_record(core_desc, bytes(payload), params, None))
            core_desc = None
        elif rtype == REC_END:
            if length != struct.calcsize("<I"):
                raise ContainerError(f"{p} END record has {length} bytes, expected 4")
            (end_status,) = struct.unpack("<I", payload)
        else:
            raise ContainerError(f"{p} has unknown record type {rtype} at offset {off - length}")

    if end_status is None:
        raise ContainerError(f"{p} has no END record, the file is truncated")
    if end_status not in (END_OK, END_FIFO_NOT_EMPTY):
        raise ContainerError(f"{p} capture ended with status {end_status}")

    return Capture(
        source="espt",
        source_path=p,
        abi_major=snap.abi_major,
        abi_minor=snap.abi_minor,
        capture_reason=snap.capture_reason,
        write_seq=snap.write_seq,
        target_id=snap.target_id,
        chip_revision=snap.chip_revision,
        app_elf_sha256=sha_bytes,
        cores=tuple(buffers),
        fifo_emptied=end_status == END_OK,
    )


__all__ = ["MAGIC", "VERSION", "load", "sniff"]
