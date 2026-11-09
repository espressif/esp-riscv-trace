# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Analysis helpers and streaming accumulators for trace events."""

from __future__ import annotations

from .diagnostics import (
    DIAG_ROOT_NO_PCS_RECOVERED,
    DiagnosticAccumulator,
    TraceDiagnostic,
)
from .history import (
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
    iter_entries,
)
from .location import CodeLocation, locate
from .stats import (
    UNRESOLVED_FILE,
    UNRESOLVED_LINE,
    StatsAccumulator,
    StatsSnapshot,
    unresolved_function_label,
)

__all__ = [
    "DIAG_ROOT_NO_PCS_RECOVERED",
    "UNRESOLVED_FILE",
    "UNRESOLVED_LINE",
    "CodeLocation",
    "CodeLoop",
    "DiagnosticAccumulator",
    "History",
    "HistoryBuilder",
    "HistoryEntry",
    "HistoryItem",
    "HistoryKind",
    "HistoryLoop",
    "StackFrame",
    "StatsAccumulator",
    "StatsSnapshot",
    "TraceDiagnostic",
    "build_history",
    "cause_name",
    "is_fault",
    "iter_entries",
    "locate",
    "unresolved_function_label",
]
