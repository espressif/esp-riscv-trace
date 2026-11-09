# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Analysis helpers and streaming accumulators for trace events."""

from __future__ import annotations

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

__all__ = [
    "CodeLocation",
    "CodeLoop",
    "History",
    "HistoryBuilder",
    "HistoryEntry",
    "HistoryItem",
    "HistoryKind",
    "HistoryLoop",
    "StackFrame",
    "build_history",
    "cause_name",
    "is_fault",
    "iter_entries",
    "locate",
]
