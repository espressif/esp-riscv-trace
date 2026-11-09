# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Trace intermediate representation event definitions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum, StrEnum


class InsnClass(IntEnum):
    """Control-flow class of a decoded instruction."""

    NONE = 0  # non-control-flow (advances PC by insn length)
    BRANCH = 1  # direct conditional branch (consumes one branch-map bit)
    JUMP = 2  # direct unconditional jump (target is PC + imm)
    UNINFERABLE = 3  # indirect jump or trap return (next PC from address packet)
    TRAP = 4  # synchronous trap cause (next PC from trap packet)


class LinkKind(StrEnum):
    """Call stack effect of a jump, from the return-address stack hints of the RISC-V ISA.

    ``ra`` and ``t0`` are the link registers.
    """

    NONE = "none"  # not a jump
    CALL = "call"  # writes a link register
    RETURN = "return"  # jumps through a link register without writing one
    JUMP = "jump"  # neither
    COROUTINE = "coroutine"  # jumps through one link register and writes the other
    TRAP_RETURN = "trap_return"  # mret or sret


@dataclass(frozen=True)
class Provenance:
    """Origin of an event in the framed byte stream."""

    segment_index: int
    stream_offset: int
    packet_index: int
    payload: bytes | None = None


@dataclass(frozen=True)
class TraceSync:
    """Synchronization event anchoring the decoder at pc."""

    pc: int
    privilege: int
    provenance: Provenance | None


@dataclass(frozen=True)
class TraceInstruction:
    """One executed instruction recovered from trace."""

    pc: int
    length: int
    cls: InsnClass
    branch_taken: bool | None
    # None when an uninferable jump leaves the traced code, which the encoder does not report
    next_pc: int | None
    provenance: Provenance | None
    inferred: bool = False
    link: LinkKind = LinkKind.NONE


@dataclass(frozen=True)
class TraceTrap:
    """Trap entry event (exception or interrupt)."""

    cause: int
    interrupt: bool
    handler_pc: int | None
    interrupted_pc: int | None
    tval: int | None
    provenance: Provenance | None
    # Instruction the trap was taken at, the value of mepc; None when the trace cannot tell
    epc: int | None = None


@dataclass(frozen=True)
class TraceGap:
    """A point where continuous instruction reconstruction cannot continue."""

    reason: str
    detail: str
    provenance: Provenance | None


@dataclass(frozen=True)
class TraceBoundary:
    """Physical trace-memory boundary that does not break reconstruction."""

    kind: str
    detail: str
    provenance: Provenance | None


@dataclass(frozen=True)
class TraceWarning:
    """Parser confidence note. Reconstruction may still be continuous."""

    kind: str
    detail: str
    provenance: Provenance | None


TraceEvent = TraceSync | TraceInstruction | TraceTrap | TraceGap | TraceBoundary | TraceWarning


__all__ = [
    "InsnClass",
    "LinkKind",
    "Provenance",
    "TraceBoundary",
    "TraceEvent",
    "TraceGap",
    "TraceInstruction",
    "TraceSync",
    "TraceTrap",
    "TraceWarning",
]
