# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Container loaders and format auto-detection."""

from __future__ import annotations

from pathlib import Path

from . import coredump, espt
from .base import Capture, ContainerError, TraceBuffer


def load(path: str | Path) -> Capture:
    """Load an ESPT trace file or an esp-coredump ELF."""
    p = Path(path)
    with open(p, "rb") as fp:
        head = fp.read(8)
    if espt.sniff(head):
        return espt.load(p)
    if coredump.sniff(head):
        return coredump.load(p)
    raise ContainerError(f"{p} is neither an ESPT trace file nor a coredump ELF")


__all__ = [
    "Capture",
    "ContainerError",
    "TraceBuffer",
    "coredump",
    "espt",
    "load",
]
