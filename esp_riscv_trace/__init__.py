# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""esp-riscv-trace top-level package.

Decodes RISC-V hardware instruction trace captured on Espressif chips.
``open_capture`` is the entry point for Python callers.
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

__all__ = [
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
    "StackFrame",
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
    "is_fault",
    "open_capture",
]
