# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Trace intermediate representation.

Event dataclasses (``TraceSync`` / ``TraceInstruction`` / ``TraceTrap``
/ ``TraceGap`` / ``TraceBoundary`` / ``TraceWarning``) and the per-core
``TraceCaptureNote`` header. The walker in ``decoder/reconstruct.py``
yields these types. Exporters consume them.
"""

from __future__ import annotations

from .capture_note import (
    ElfShaStatus,
    InferredWrapState,
    TraceCaptureNote,
)
from .events import (
    InsnClass,
    LinkKind,
    Provenance,
    TraceBoundary,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
    TraceWarning,
)

__all__ = [
    "ElfShaStatus",
    "InferredWrapState",
    "InsnClass",
    "LinkKind",
    "Provenance",
    "TraceBoundary",
    "TraceCaptureNote",
    "TraceEvent",
    "TraceGap",
    "TraceInstruction",
    "TraceSync",
    "TraceTrap",
    "TraceWarning",
]
