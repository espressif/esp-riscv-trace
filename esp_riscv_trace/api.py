# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Read trace captures from Python, as esp-coredump does.

``open_capture`` loads a capture with its ELF files and yields a ``TraceCapture``.
It never stops on an ELF that differs from the one the capture recorded,
``TraceCore.note.elf_sha_status`` tells the caller instead. Problems that do not stop decoding,
such as a missing ROM ELF, are collected in ``TraceCapture.warnings``.
"""

from __future__ import annotations

import contextlib
import os
from collections.abc import Collection, Iterator
from pathlib import Path

from .analysis.history import History, build_history
from .analysis.location import CodeLocation, locate
from .elf import CodeImage
from .ir.capture_note import TraceCaptureNote
from .ir.events import TraceEvent
from .session import CoreView, Session, iter_cores, open_session

StrPath = str | os.PathLike[str]


class TraceCore:
    """The trace of one core."""

    def __init__(self, view: CoreView) -> None:
        self._view = view

    @property
    def core_id(self) -> int:
        return self._view.core.core_id

    @property
    def note(self) -> TraceCaptureNote:
        """What the capture records about this core, and how the ELF compares with it."""
        return self._view.note

    def events(self) -> Iterator[TraceEvent]:
        """Decode the trace again and yield its events, oldest first."""
        return self._view.events()

    def history(self, last: int = 0, end_interrupts: Collection[int] = ()) -> History:
        """Return the call outline up to the fault, with only the ``last`` newest items when not zero.

        An interrupt whose mcause exception code is in ``end_interrupts`` ends it like a fault.
        """
        session = self._view.session
        image = session.image if session.elf is not None else None
        return build_history(self.events(), image, last=last, end_interrupts=end_interrupts)


class TraceCapture:
    """A loaded capture. Valid inside the ``open_capture`` block only."""

    def __init__(self, session: Session, warnings: list[str]) -> None:
        self._session = session
        self.warnings = warnings
        self.cores = [TraceCore(view) for view in iter_cores(session, None)]

    @property
    def chip(self) -> str:
        return self._session.chip

    @property
    def image(self) -> CodeImage | None:
        """Code bytes and function names of the application ELF, layered over the ROM ELF; None without ``elf``."""
        return self._session.image if self._session.elf is not None else None

    def core(self, core_id: int) -> TraceCore | None:
        return next((c for c in self.cores if c.core_id == core_id), None)

    def resolve(self, pc: int) -> CodeLocation:
        """Return the function and source line of ``pc``."""
        return locate(pc, self.image, self._session.resolver)


@contextlib.contextmanager
def open_capture(
    capture: StrPath,
    *,
    elf: StrPath | None = None,
    rom_elf: StrPath | None = None,
    chip: str | None = None,
) -> Iterator[TraceCapture]:
    """Load ``capture``, an ``.esptrace`` file or a core dump ELF with a trace note.

    ``elf`` is the application ELF. ``rom_elf`` defaults to the one ESP-IDF provides for the chip,
    found through ``IDF_PATH`` or ``ESP_ROM_ELF_DIR``. ``chip`` overrides the chip the capture records.
    """
    warnings: list[str] = []
    with open_session(
        Path(capture),
        elf=Path(elf) if elf is not None else None,
        rom_elf=Path(rom_elf) if rom_elf is not None else None,
        chip=chip,
        force_elf=True,
        warn=warnings.append,
    ) as session:
        yield TraceCapture(session, warnings)


__all__ = ["TraceCapture", "TraceCore", "open_capture"]
