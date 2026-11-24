# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""esp-riscv-trace top-level package.

Decodes RISC-V hardware instruction trace captured on Espressif chips.
``open_capture`` is the entry point for Python callers. ``parse_note`` and ``format_note`` read and describe the
trace note of a core dump alone. ``FaultWindowBuilder`` keeps the events before a fault for the walks of
``analysis.origin``.
"""

from __future__ import annotations

__version__ = "0.3.0"

from .analysis.history import (
    CodeLoop,
    History,
    HistoryBuilder,
    HistoryEntry,
    HistoryItem,
    HistoryKind,
    HistoryLoop,
    StackFrame,
    build_history,
    cause_name,
    is_fault,
)
from .analysis.location import CodeLocation
from .analysis.origin import FaultWindow, FaultWindowBuilder
from .api import TraceCapture, TraceCore, open_capture
from .decoder.reconstruct import GapReason
from .exporters.note import format_note
from .ir.capture_note import ElfShaStatus, TraceCaptureNote, compare_elf_sha
from .ir.events import (
    InsnClass,
    LinkKind,
    TraceBoundary,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
    TraceWarning,
)
from .snapshot.abi import (
    NOTE_NAME,
    NOTE_TYPE,
    AbiError,
    AddressMode,
    CaptureReason,
    MemoryMode,
    NoteHeader,
    NoteRecord,
    PacketFormat,
    ResyncMode,
    State,
    parse_note,
)

__all__ = [
    "NOTE_NAME",
    "NOTE_TYPE",
    "AbiError",
    "AddressMode",
    "CaptureReason",
    "CodeLocation",
    "CodeLoop",
    "ElfShaStatus",
    "FaultWindow",
    "FaultWindowBuilder",
    "GapReason",
    "History",
    "HistoryBuilder",
    "HistoryEntry",
    "HistoryItem",
    "HistoryKind",
    "HistoryLoop",
    "InsnClass",
    "LinkKind",
    "MemoryMode",
    "NoteHeader",
    "NoteRecord",
    "PacketFormat",
    "ResyncMode",
    "StackFrame",
    "State",
    "TraceBoundary",
    "TraceCapture",
    "TraceCaptureNote",
    "TraceCore",
    "TraceEvent",
    "TraceGap",
    "TraceInstruction",
    "TraceSync",
    "TraceTrap",
    "TraceWarning",
    "__version__",
    "build_history",
    "cause_name",
    "compare_elf_sha",
    "format_note",
    "is_fault",
    "open_capture",
    "parse_note",
]
