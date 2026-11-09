# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""esp-riscv-trace top-level package.

Decodes RISC-V hardware instruction trace captured on Espressif chips.
``open_capture`` is the entry point for Python callers. ``parse_note`` and ``format_note`` read and describe the
trace note of a core dump alone.
"""

from __future__ import annotations

__version__ = "0.1.0"

from .analysis.history import (
    CodeLoop,
    History,
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
from .api import TraceCapture, TraceCore, open_capture
from .decoder.reconstruct import GapReason
from .exporters.note import format_note
from .ir.capture_note import ElfShaStatus, TraceCaptureNote
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
    "GapReason",
    "History",
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
    "format_note",
    "is_fault",
    "open_capture",
    "parse_note",
]
