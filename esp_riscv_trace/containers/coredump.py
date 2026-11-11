# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Loader for ESP-IDF coredump ELF files containing trace notes."""

from __future__ import annotations

from pathlib import Path

from elftools.common.exceptions import ELFError
from elftools.elf.elffile import ELFFile

from ..snapshot import abi
from .base import Capture, ContainerError, TraceBuffer

ELF_MAGIC = b"\x7fELF"


def sniff(head_bytes: bytes) -> bool:
    """Return True when ``head_bytes`` starts with the ELF magic."""
    return head_bytes.startswith(ELF_MAGIC)


def load(path: str | Path) -> Capture:
    p = Path(path)
    try:
        with open(p, "rb") as fp:
            elf = ELFFile(fp)
            if elf.header["e_type"] != "ET_CORE":
                raise ContainerError(f"{p} is not a core file (e_type={elf.header['e_type']})")

            note_desc = _find_riscv_trace_note(elf)
            if note_desc is None:
                raise ContainerError(f"{p} has no ESP_RISCV_TRACE note (type {abi.NOTE_TYPE})")

            try:
                header, records = abi.parse_note(note_desc)
            except abi.AbiError as e:
                raise ContainerError(f"{p} has a malformed ESP_RISCV_TRACE note. {e}") from e

            pt_loads = _index_pt_loads(elf)
            buffers: list[TraceBuffer] = []
            for i, rec in enumerate(records):
                if not rec.has_segment():
                    continue
                key = (rec.segment_vaddr, rec.segment_size)
                matches = pt_loads.get(key, [])
                if not matches:
                    raise ContainerError(
                        f"{p} core[{i}] references vaddr=0x{rec.segment_vaddr:x} "
                        f"size={rec.segment_size} but no PT_LOAD matches"
                    )
                if len(matches) > 1:
                    raise ContainerError(
                        f"{p} core[{i}] references vaddr=0x{rec.segment_vaddr:x} "
                        f"size={rec.segment_size} but {len(matches)} PT_LOADs match"
                    )
                seg = matches[0]
                raw = seg.data()
                if len(raw) != rec.segment_size:
                    raise ContainerError(
                        f"{p} core[{i}] segment payload is {len(raw)} bytes, expected {rec.segment_size}"
                    )
                buffers.append(TraceBuffer.from_record(rec, bytes(raw), header.encoder_params, rec.segment_index))
    except ELFError as e:
        raise ContainerError(f"{p} is not a valid ELF file. {e}") from e

    sha_bytes: bytes | None = None
    if header.app_elf_sha256_size == 32:
        sha_bytes = bytes(header.app_elf_sha256)

    return Capture(
        source="coredump",
        source_path=p,
        abi_major=header.abi_major,
        abi_minor=header.abi_minor,
        capture_reason=header.capture_reason,
        write_seq=header.write_seq,
        target_id=header.target_id,
        chip_revision=header.chip_revision,
        app_elf_sha256=sha_bytes,
        cores=tuple(buffers),
    )


def _find_riscv_trace_note(elf) -> bytes | None:
    for seg in elf.iter_segments():
        if seg.header["p_type"] != "PT_NOTE":
            continue
        for note in seg.iter_notes():
            if note["n_name"] == abi.NOTE_NAME and note["n_type"] == abi.NOTE_TYPE:
                return bytes(note["n_desc"])
    return None


def _index_pt_loads(elf) -> dict[tuple[int, int], list]:
    """Group ``PT_LOAD`` segments by ``(p_vaddr, p_filesz)``."""
    idx: dict[tuple[int, int], list] = {}
    for seg in elf.iter_segments():
        if seg.header["p_type"] != "PT_LOAD":
            continue
        key = (seg.header["p_vaddr"], seg.header["p_filesz"])
        idx.setdefault(key, []).append(seg)
    return idx


__all__ = ["ELF_MAGIC", "load", "sniff"]
