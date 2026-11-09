# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Chip ID definitions and name mappings for supported RISC-V targets."""

from __future__ import annotations

CHIP_ID_ESP32 = 0x0000
CHIP_ID_ESP32S2 = 0x0002
CHIP_ID_ESP32C3 = 0x0005
CHIP_ID_ESP32S3 = 0x0009
CHIP_ID_ESP32C2 = 0x000C
CHIP_ID_ESP32C6 = 0x000D
CHIP_ID_ESP32H2 = 0x0010
CHIP_ID_ESP32P4 = 0x0012
CHIP_ID_ESP32C61 = 0x0014
CHIP_ID_ESP32C5 = 0x0017
CHIP_ID_ESP32H21 = 0x0019
CHIP_ID_ESP32H4 = 0x001C
CHIP_ID_ESP32S31 = 0x0020


CHIP_NAME_TO_ID: dict[str, int] = {
    "esp32": CHIP_ID_ESP32,
    "esp32s2": CHIP_ID_ESP32S2,
    "esp32c3": CHIP_ID_ESP32C3,
    "esp32s3": CHIP_ID_ESP32S3,
    "esp32c2": CHIP_ID_ESP32C2,
    "esp32c6": CHIP_ID_ESP32C6,
    "esp32h2": CHIP_ID_ESP32H2,
    "esp32p4": CHIP_ID_ESP32P4,
    "esp32c61": CHIP_ID_ESP32C61,
    "esp32c5": CHIP_ID_ESP32C5,
    "esp32h21": CHIP_ID_ESP32H21,
    "esp32h4": CHIP_ID_ESP32H4,
    "esp32s31": CHIP_ID_ESP32S31,
}

CHIP_ID_TO_NAME: dict[int, str] = {v: k for k, v in CHIP_NAME_TO_ID.items()}


# Espressif RISC-V chips whose trace encoder we support decoding for.
RISCV_TRACE_CHIPS: frozenset[str] = frozenset(
    {
        "esp32c5",
        "esp32c6",
        "esp32c61",
        "esp32h2",
        "esp32h21",
        "esp32h4",
        "esp32p4",
        "esp32s31",
    }
)


def chip_name_from_id(target_id: int) -> str | None:
    """Return the chip name for a ``target_id``, or None when unmapped."""
    return CHIP_ID_TO_NAME.get(target_id)


def is_riscv_trace_chip(name_or_id: str | int) -> bool:
    """Return True when the chip has a supported RISC-V trace encoder."""
    if isinstance(name_or_id, int):
        name = CHIP_ID_TO_NAME.get(name_or_id)
        if name is None:
            return False
    else:
        name = name_or_id.strip().lower()
    return name in RISCV_TRACE_CHIPS


__all__ = [
    "CHIP_ID_ESP32",
    "CHIP_ID_ESP32C2",
    "CHIP_ID_ESP32C3",
    "CHIP_ID_ESP32C5",
    "CHIP_ID_ESP32C6",
    "CHIP_ID_ESP32C61",
    "CHIP_ID_ESP32H2",
    "CHIP_ID_ESP32H4",
    "CHIP_ID_ESP32H21",
    "CHIP_ID_ESP32P4",
    "CHIP_ID_ESP32S2",
    "CHIP_ID_ESP32S3",
    "CHIP_ID_ESP32S31",
    "CHIP_ID_TO_NAME",
    "CHIP_NAME_TO_ID",
    "RISCV_TRACE_CHIPS",
    "chip_name_from_id",
    "is_riscv_trace_chip",
]
