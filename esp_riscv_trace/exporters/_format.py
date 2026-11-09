# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Formatting helpers shared by the text and stats exporters."""

from __future__ import annotations

FUNCTION_COLUMN_WIDTH = 32
_SHORT_SHA_CHARS = 16


def one_line(s: str) -> str:
    """Escape backslashes and line terminators so a value never breaks out of its row."""
    return s.replace("\\", "\\\\").replace("\r", "\\r").replace("\n", "\\n")


def one_line_quoted(s: str) -> str:
    """Return ``one_line(s)`` with double quotes escaped too, for values written inside quotes."""
    return one_line(s).replace('"', '\\"')


def short_sha(sha: bytes | None) -> str:
    """Return the first hex digits of ``sha``, or ``none``."""
    if sha is None:
        return "none"
    return sha.hex()[:_SHORT_SHA_CHARS]


def marker(body: str, prov: str) -> str:
    """Return a ``--- ... ---`` event marker line, with provenance when there is one."""
    return f"--- {body} {prov} ---" if prov else f"--- {body} ---"
