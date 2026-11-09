# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Base class of the errors that stop a command."""

from __future__ import annotations

from esp_pylib.errors import FatalError


class EspRiscvTraceError(FatalError):
    """An error that stops the command. The CLI reports it and exits with a non-zero code."""


__all__ = ["EspRiscvTraceError"]
