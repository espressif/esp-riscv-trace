# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Per-chip hardware decoder profiles and configuration defaults."""

from __future__ import annotations

from dataclasses import dataclass

from ..errors import EspRiscvTraceError
from ..snapshot import abi
from ..snapshot.chip import is_riscv_trace_chip


class ParamsError(EspRiscvTraceError):
    """Raised when a decoder profile cannot be resolved for a chip."""


@dataclass(frozen=True)
class DecoderProfile:
    """Hardware parameters that gate packet-format parsing for one core."""

    chip: str
    packet_format: abi.PacketFormat
    iaddress_width: int
    iaddress_lsb: int
    privilege_width: int
    ecause_width: int

    @property
    def address_bits(self) -> int:
        """Bits occupied by ``address`` in sync and trap packets."""
        return self.iaddress_width - self.iaddress_lsb

    def pc_from_raw(self, raw_address: int) -> int:
        """Recreate a byte PC from the address field value."""
        return raw_address << self.iaddress_lsb


_ET20 = abi.PacketFormat.EFFICIENT_TRACE_2_0
_PT10 = abi.PacketFormat.PROCESSOR_TRACE_1_0


def _make(chip: str, packet_format: abi.PacketFormat, privilege_width: int, ecause_width: int) -> DecoderProfile:
    return DecoderProfile(
        chip=chip,
        packet_format=packet_format,
        iaddress_width=32,
        iaddress_lsb=1,
        privilege_width=privilege_width,
        ecause_width=ecause_width,
    )


_CHIP_DEFAULTS: dict[str, DecoderProfile] = {
    "esp32c5": _make("esp32c5", _ET20, 1, 6),
    "esp32c61": _make("esp32c61", _ET20, 1, 6),
    "esp32h4": _make("esp32h4", _ET20, 1, 6),
    "esp32p4": _make("esp32p4", _ET20, 1, 6),
    "esp32s31": _make("esp32s31", _ET20, 2, 6),
    "esp32c6": _make("esp32c6", _PT10, 1, 5),
    "esp32h2": _make("esp32h2", _PT10, 1, 5),
    "esp32h21": _make("esp32h21", _PT10, 1, 5),
}


def default_profile(chip: str) -> DecoderProfile:
    """Return the fallback profile for ``chip``. Raises ``ParamsError``."""
    key = chip.strip().lower()
    if not is_riscv_trace_chip(key):
        raise ParamsError(f"chip {chip!r} has no supported trace encoder")
    if key not in _CHIP_DEFAULTS:
        raise ParamsError(f"no fallback profile for chip {chip!r}")
    return _CHIP_DEFAULTS[key]


def _check_snapshot_params(chip: str, params: abi.EncoderParams) -> None:
    if params.iaddress_width_p <= params.iaddress_lsb_p:
        raise ParamsError(
            f"chip {chip} snapshot iaddress_width_p {params.iaddress_width_p} "
            f"must exceed iaddress_lsb_p {params.iaddress_lsb_p}"
        )
    if params.iaddress_width_p > 64:
        raise ParamsError(f"chip {chip} snapshot iaddress_width_p {params.iaddress_width_p} exceeds 64")
    if params.privilege_width_p == 0 or params.privilege_width_p > 3:
        raise ParamsError(f"chip {chip} snapshot privilege_width_p {params.privilege_width_p} out of range 1..3")
    if params.ecause_width_p == 0 or params.ecause_width_p > 6:
        raise ParamsError(f"chip {chip} snapshot ecause_width_p {params.ecause_width_p} out of range 1..6")


def resolve_profile(chip: str, params: abi.EncoderParams | None = None) -> DecoderProfile:
    """Return a ``DecoderProfile`` for ``chip``.

    When ``params`` is given, its field values override the chip fallback for
    every field it covers. Packet format is always the chip default.
    """
    base = default_profile(chip)
    if params is None:
        return base
    _check_snapshot_params(base.chip, params)
    return DecoderProfile(
        chip=base.chip,
        packet_format=base.packet_format,
        iaddress_width=params.iaddress_width_p,
        iaddress_lsb=params.iaddress_lsb_p,
        privilege_width=params.privilege_width_p,
        ecause_width=params.ecause_width_p,
    )


__all__ = [
    "DecoderProfile",
    "ParamsError",
    "default_profile",
    "resolve_profile",
]
