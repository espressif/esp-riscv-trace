# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Reconstruct program execution flow from parsed trace packets and ELF code."""

from __future__ import annotations

import collections
from collections.abc import Generator, Iterable, Iterator
from dataclasses import dataclass
from enum import StrEnum

from ..elf import CodeImage, CodeReadError
from ..ir.events import (
    Provenance,
    TraceBoundary,
    TraceEvent,
    TraceGap,
    TraceInstruction,
    TraceSync,
    TraceTrap,
    TraceWarning,
)
from ..snapshot import abi
from .framing import (
    Frame,
    FramingEvent,
    FramingEventKind,
)
from .instruction import PC_MASK, DecodedInsn, InsnClass, InsnError, decode
from .instruction import length as _insn_length
from .packets import (
    QUAL_STATUS_ENDED_NTR,
    QUAL_STATUS_ENDED_REP,
    ContextPacket,
    Format0Packet,
    Format1NoAddrPacket,
    Format1WithAddrPacket,
    Format2Packet,
    Packet,
    SupportPacket,
    SyncPacket,
    TrapPacket,
)
from .params import DecoderProfile

InputEvent = Packet | FramingEvent


class GapReason(StrEnum):
    """Why instruction reconstruction stopped."""

    FRAMING_ERROR = "framing_error"
    ALIGNMENT_LOST = "alignment_lost"
    PACKET_ERROR = "packet_error"
    PARTIAL_FRAME = "partial_frame"
    INDEX_JUMP = "packet_index_jump"
    UNEXPECTED_F0 = "unexpected_format_0"
    UNEXPECTED_CONTEXT = "unexpected_context_packet"
    UNEXPECTED_PACKET = "unexpected_packet"
    TRACE_DISABLED = "trace_disabled"
    # The encoder filter stopped tracing, the core runs code outside the traced range until the next sync
    UNTRACED = "untraced"
    TRACE_LOST = "trace_lost"
    PC_OUT_OF_IMAGE = "pc_out_of_image"
    DECODE_ERROR = "decode_error"
    NO_PC = "walker_not_anchored"
    INFINITE_LOOP = "infinite_loop"
    # The core went to a reported address without a traced jump, as when a debugger sets the pc
    UNEXPLAINED_JUMP = "unexplained_jump"


_FRAMING_GAPS = {
    FramingEventKind.ALIGNMENT_LOST: GapReason.ALIGNMENT_LOST,
    FramingEventKind.PRE_ANCHOR_SKIP: GapReason.ALIGNMENT_LOST,
    FramingEventKind.INDEX_JUMP: GapReason.INDEX_JUMP,
    FramingEventKind.FRAMING_ERROR: GapReason.FRAMING_ERROR,
    FramingEventKind.PARTIAL_FRAME: GapReason.PARTIAL_FRAME,
    FramingEventKind.PACKET_ERROR: GapReason.PACKET_ERROR,
    FramingEventKind.TRACE_LOST: GapReason.TRACE_LOST,
}

# RISC-V mcause value for an illegal instruction exception
ECAUSE_ILLEGAL_INSTRUCTION = 2


def _prov_from_frame(frame: Frame) -> Provenance:
    return Provenance(
        segment_index=frame.segment_index,
        stream_offset=frame.stream_offset,
        packet_index=frame.packet_index,
        payload=frame.payload,
    )


def _prov_from_framing_event(event: FramingEvent) -> Provenance:
    """Convert structural provenance, keeping packet data when available."""
    return Provenance(
        segment_index=event.segment_index,
        stream_offset=event.stream_offset,
        packet_index=event.packet_index,
        payload=event.payload,
    )


def reconstruct(
    events: Iterable[InputEvent],
    image: CodeImage,
    profile: DecoderProfile,
    address_mode: abi.AddressMode,
) -> Iterator[TraceEvent]:
    """Walk ``events`` and yield the recovered trace event stream.

    ``image`` supplies instruction bytes. ``profile`` fixes the packet
    format and address widths. ``address_mode`` selects DELTA or FULL
    reconstruction and is ignored for PT 1.0 which is always full.
    """
    walker = _Walker(image, profile, address_mode)
    for event in events:
        yield from walker.feed(event)
    yield from walker.finish()


# Cap instructions followed from one packet to terminate on infinite loops.
_WALK_INSN_LIMIT = 1_000_000


@dataclass(frozen=True)
class _Report:
    """Format 1/2 address that a following trap may name as the instruction it was taken at."""

    pc: int
    # Position of that instruction in the held instructions, their count when it was not walked
    index: int
    # Reached without crossing an uninferable discontinuity, so a Format 3
    # packet confirms it as the report boundary
    first_occurrence: bool = True


@dataclass(frozen=True)
class _Parked:
    """Updiscon target the walker stopped at before fetching it."""

    pc: int
    provenance: Provenance | None


@dataclass(frozen=True)
class _DeferredGap:
    """Fetch or decode failure at a PC that no packet has shown to execute yet."""

    pc: int
    reason: str
    detail: str


class _Walker:
    """Trace state machine. Not part of the public API."""

    def __init__(
        self,
        image: CodeImage,
        profile: DecoderProfile,
        address_mode: abi.AddressMode,
    ) -> None:
        self._image = image
        self._profile = profile
        # PT 1.0 has no address-mode selector, only full-address encoding.
        self._configured_mode = (
            abi.AddressMode.FULL if profile.packet_format == abi.PacketFormat.PROCESSOR_TRACE_1_0 else address_mode
        )
        self._effective_mode = self._configured_mode
        self._addr_mask = (1 << profile.iaddress_width) - 1
        self._pc: int | None = None
        self._last_reported_pc: int | None = None
        self._branch_queue: collections.deque[int] = collections.deque()
        self._pending_address: int | None = None
        # A Format 1/2 address can be reached by ordinary inferred execution
        # before the decoder encounters the uninferable discontinuity whose
        # target it might describe.  Keep that ambiguity until the following
        # packet: another Format 1/2 means execution must continue until the
        # address is encountered again, while Format 3 confirms the first
        # occurrence was the reported instruction.
        self._inferred_reported_pc: int | None = None
        self._replay_address: int | None = None
        # Notifications and updiscon-before-Format-3 packets name the final
        # reported instruction.  Stop after reconstructing that instruction
        # instead of walking speculatively beyond it.
        self._stop_after_reported_pc: int | None = None
        self._ambiguous_reported_pc: int | None = None
        self._paused_at_reported_pc: bool = False
        self._trace_enabled: bool = True
        # Instructions walked past the information of the latest packet. The
        # next packet decides whether they retired: a trap can show that
        # execution stopped at a reported address inside them.
        self._held: list[TraceInstruction] = []
        self._report: _Report | None = None
        self._updiscon_target: int | None = None
        self._parked: _Parked | None = None
        self._deferred_gap: _DeferredGap | None = None
        # Direct jump to itself the core runs until a trap, which is taken at it, or a sync
        self._spin_pc: int | None = None
        # Boundaries and warnings seen while instructions are held
        self._held_markers: list[TraceEvent] = []

    def finish(self) -> Iterator[TraceEvent]:
        """Yield what still waits for a packet when the stream ends."""
        yield from self._settle()

    def feed(self, event: InputEvent) -> Iterator[TraceEvent]:
        if isinstance(event, FramingEvent):
            yield from self._feed_framing(event)
            return
        yield from self._feed_packet(event)

    # ------------------------------------------------------------------
    # Framing event dispatch
    # ------------------------------------------------------------------

    def _feed_framing(self, event: FramingEvent) -> Iterator[TraceEvent]:
        kind = event.kind
        if kind == FramingEventKind.ANCHOR_FOUND:
            return
        prov = _prov_from_framing_event(event)
        if kind == FramingEventKind.WRAP_BOUNDARY:
            yield from self._marker(TraceBoundary(kind="wrap", detail=event.detail, provenance=prov))
            return
        if kind == FramingEventKind.NO_ANCHOR:
            yield from self._marker(TraceWarning(kind="no_anchor", detail=event.detail, provenance=prov))
            return
        yield from self._settle()
        reason = _FRAMING_GAPS.get(kind)
        if reason is not None:
            yield self._gap(reason, event.detail, prov)
            return
        # Unknown framing kinds cannot advance the walker safely
        yield self._gap(GapReason.FRAMING_ERROR, f"unknown framing event kind {kind!r}", prov)

    # ------------------------------------------------------------------
    # Packet dispatch
    # ------------------------------------------------------------------

    def _feed_packet(self, pkt: Packet) -> Iterator[TraceEvent]:
        prov = _prov_from_frame(pkt.frame)
        if isinstance(pkt, SyncPacket):
            yield from self._on_sync(pkt, prov)
            return
        if isinstance(pkt, TrapPacket):
            yield from self._on_trap(pkt, prov)
            return
        if isinstance(pkt, SupportPacket):
            yield from self._on_support(pkt, prov)
            return
        if isinstance(pkt, Format1WithAddrPacket):
            yield from self._on_format1_with_addr(pkt, prov)
            return
        if isinstance(pkt, Format1NoAddrPacket):
            yield from self._on_format1_no_addr(pkt, prov)
            return
        if isinstance(pkt, Format2Packet):
            yield from self._on_format2(pkt, prov)
            return
        yield from self._settle()
        if isinstance(pkt, ContextPacket):
            # Context packets are not used on supported chips
            yield self._gap(GapReason.UNEXPECTED_CONTEXT, "context packet not enabled on this chip", prov)
            return
        if isinstance(pkt, Format0Packet):
            # Format 0 packets are not used on supported chips
            yield self._gap(GapReason.UNEXPECTED_F0, "format 0 packet not enabled on this chip", prov)
            return
        yield self._gap(GapReason.UNEXPECTED_PACKET, f"unknown packet type {type(pkt).__name__}", prov)

    # ------------------------------------------------------------------
    # Per-packet handlers
    # ------------------------------------------------------------------

    def _on_sync(self, pkt: SyncPacket, prov: Provenance) -> Iterator[TraceEvent]:
        yield from self._rewind_to_report()
        yield from self._settle()
        # The sync address is the target of an uninferable jump the walker is waiting at
        waiting = self._waiting_uninferable()
        if waiting is not None:
            yield TraceInstruction(
                pc=waiting[0],
                length=waiting[1].length,
                cls=InsnClass.UNINFERABLE,
                branch_taken=None,
                next_pc=pkt.pc,
                provenance=prov,
                link=waiting[1].link,
            )
        self._trace_enabled = True
        yield TraceSync(pc=pkt.pc, privilege=pkt.privilege, provenance=prov)
        yield from self._resume_at(pkt.pc, pkt.branch, prov)

    def _on_trap(self, pkt: TrapPacket, prov: Provenance) -> Iterator[TraceEvent]:
        # Walk as far as possible before handling the trap packet.
        # If the previous Format 1/2 address was reached without crossing an
        # uninferable discontinuity, this Format 3 packet resolves the
        # ambiguity: that first occurrence really was the reported boundary.
        if self._inferred_reported_pc is None and not self._paused_at_reported_pc:
            yield from self._walk(prov)
        else:
            self._inferred_reported_pc = None
            self._paused_at_reported_pc = False
        exception = pkt.interrupt == 0
        epc = yield from self._resolve_trap(exception)
        self._spin_pc = None
        handler_pc: int | None
        interrupted_pc: int | None
        is_pt10 = self._profile.packet_format == abi.PacketFormat.PROCESSOR_TRACE_1_0
        if is_pt10:
            handler_pc = pkt.address_pc
            # PT 1.0 tval carries the faulting PC on illegal instruction exceptions
            if exception and pkt.ecause == ECAUSE_ILLEGAL_INSTRUCTION and pkt.tval is not None:
                interrupted_pc = pkt.tval & self._addr_mask
            else:
                interrupted_pc = None
        elif pkt.thaddr == 1:
            handler_pc = pkt.address_pc
            interrupted_pc = None
        else:
            handler_pc = None
            interrupted_pc = pkt.address_pc
        if interrupted_pc is not None:
            if is_pt10 and epc is not None and epc != interrupted_pc:
                yield TraceWarning(
                    kind="epc_mismatch",
                    detail=f"reconstructed epc 0x{epc:08x}, tval reports 0x{interrupted_pc:08x}",
                    provenance=prov,
                )
            epc = interrupted_pc
        anchor = handler_pc if handler_pc is not None else interrupted_pc
        if anchor is not None:
            self._last_reported_pc = anchor
        # ecall and ebreak are reported as retired instructions that trap
        if exception and epc is not None:
            try:
                trap_insn = decode(epc, self._fetch(epc))
            except (CodeReadError, InsnError):
                # TraceTrap still records the event
                trap_insn = None
            if trap_insn is not None and trap_insn.cls == InsnClass.TRAP:
                yield TraceInstruction(
                    pc=epc,
                    length=trap_insn.length,
                    cls=trap_insn.cls,
                    branch_taken=None,
                    next_pc=anchor if anchor is not None else epc,
                    provenance=prov,
                )
        yield TraceTrap(
            cause=pkt.ecause,
            interrupt=bool(pkt.interrupt),
            handler_pc=handler_pc,
            interrupted_pc=interrupted_pc,
            tval=pkt.tval,
            provenance=prov,
            epc=epc,
        )
        if handler_pc is None:
            # Without a handler address the walker waits for a sync
            self._pc = None
            self._branch_queue.clear()
            self._clear_reported_address_state()
            return
        yield from self._resume_at(handler_pc, pkt.branch, prov)

    def _on_support(self, pkt: SupportPacket, prov: Provenance) -> Iterator[TraceEvent]:
        # A Format 3 packet normally confirms that an ambiguously reached
        # address was an inferred report boundary.  ended_ntr is the one
        # exception: the preceding address was going to be reported for an
        # updiscon anyway, so continue until its second occurrence.
        if pkt.qual_status != QUAL_STATUS_ENDED_NTR:
            yield from self._rewind_to_report()
        yield from self._settle()
        if self._paused_at_reported_pc:
            self._paused_at_reported_pc = False
        elif self._inferred_reported_pc is not None:
            if pkt.qual_status == QUAL_STATUS_ENDED_NTR:
                self._replay_address = self._inferred_reported_pc
                self._inferred_reported_pc = None
                yield from self._walk(prov)
            else:
                self._inferred_reported_pc = None
        else:
            yield from self._walk(prov)
        if pkt.enable == 0:
            yield from self._settle()
            self._trace_enabled = False
            yield self._gap(GapReason.TRACE_DISABLED, f"encoder reported disable (qual_status={pkt.qual_status})", prov)
            return
        if pkt.qual_status in (QUAL_STATUS_ENDED_REP, QUAL_STATUS_ENDED_NTR):
            yield from self._end_qualified_run(pkt.qual_status, prov)
        # ET 2.0 Support packets can update full address mode.
        self._trace_enabled = True
        if self._profile.packet_format == abi.PacketFormat.EFFICIENT_TRACE_2_0 and pkt.full_address_mode is not None:
            self._effective_mode = abi.AddressMode.FULL if pkt.full_address_mode else abi.AddressMode.DELTA

    def _end_qualified_run(self, qual_status: int, prov: Provenance) -> Iterator[TraceEvent]:
        """Stop at the last traced instruction, the encoder traces nothing until the next sync.

        That instruction is usually a jump out of the traced range, such as a call into ROM. It retired,
        but its target is not reported.
        """
        yield from self._settle()
        if self._pc is None:
            return
        waiting = self._waiting_uninferable()
        if waiting is None:
            detail = f"the encoder stopped tracing before 0x{self._pc:08x} (qual_status={qual_status})"
        else:
            pc, insn = waiting
            yield TraceInstruction(
                pc=pc,
                length=insn.length,
                cls=InsnClass.UNINFERABLE,
                branch_taken=None,
                next_pc=None,
                provenance=prov,
                link=insn.link,
            )
            detail = f"the jump at 0x{pc:08x} leaves the traced range (qual_status={qual_status})"
        yield self._gap(GapReason.UNTRACED, detail, prov)

    def _on_format1_with_addr(self, pkt: Format1WithAddrPacket, prov: Provenance) -> Iterator[TraceEvent]:
        if pkt.branches:
            yield from self._settle()
        self._enqueue_branches(pkt.branch_map, pkt.branches)
        accepted = yield from self._accept_reported_address(
            pkt.raw_address,
            pkt.address_width_bits,
            pkt.notify,
            pkt.updiscon,
            "format 1 with address",
            prov,
        )
        if not accepted:
            return
        yield from self._walk(prov)

    def _on_format1_no_addr(self, pkt: Format1NoAddrPacket, prov: Provenance) -> Iterator[TraceEvent]:
        yield from self._settle()
        self._paused_at_reported_pc = False
        self._resume_inferred_report()
        # branches == 0 indicates a full 31-bit branch map.
        self._enqueue_branches(pkt.branch_map, 31)
        yield from self._walk(prov)

    def _on_format2(self, pkt: Format2Packet, prov: Provenance) -> Iterator[TraceEvent]:
        accepted = yield from self._accept_reported_address(
            pkt.raw_address,
            pkt.address_width_bits,
            pkt.notify,
            pkt.updiscon,
            "format 2",
            prov,
        )
        if not accepted:
            return
        yield from self._walk(prov)

    # ------------------------------------------------------------------
    # Walker core
    # ------------------------------------------------------------------

    def _walk(self, prov: Provenance | None) -> Iterator[TraceEvent]:
        walked = 0
        while self._pc is not None and self._trace_enabled and self._deferred_gap is None:
            pc = self._pc
            if pc == self._spin_pc:
                return
            reached_ambiguous_report = self._ambiguous_reported_pc == pc and self._pending_address is not None
            speculative = self._speculating()
            walked += 1
            if walked > _WALK_INSN_LIMIT:
                yield from self._flush_gap(
                    GapReason.INFINITE_LOOP, f"walked {walked - 1} instructions from one packet at 0x{pc:08x}"
                )
                return
            try:
                insn = decode(pc, self._fetch(pc))
            except CodeReadError as exc:
                yield from self._fetch_failed(pc, GapReason.PC_OUT_OF_IMAGE, f"pc=0x{pc:08x}: {exc}", speculative)
                return
            except InsnError as exc:
                yield from self._fetch_failed(pc, GapReason.DECODE_ERROR, f"pc=0x{pc:08x}: {exc}", speculative)
                return

            cls = insn.cls
            if cls == InsnClass.NONE:
                next_pc = (pc + insn.length) & PC_MASK
                yield from self._output(
                    TraceInstruction(
                        pc=pc,
                        length=insn.length,
                        cls=cls,
                        branch_taken=None,
                        next_pc=next_pc,
                        provenance=prov,
                        inferred=True,
                    ),
                    speculative,
                    reached_ambiguous_report,
                )
                self._pc = next_pc
                if self._finish_reported_instruction(pc, reached_ambiguous_report):
                    return
                continue

            if cls == InsnClass.JUMP:
                target = insn.direct_target
                if target is None:
                    yield from self._flush_gap(GapReason.DECODE_ERROR, f"direct jump at 0x{pc:08x} has no target")
                    return
                yield from self._output(
                    TraceInstruction(
                        pc=pc,
                        length=insn.length,
                        cls=cls,
                        branch_taken=None,
                        next_pc=target,
                        provenance=prov,
                        inferred=True,
                        link=insn.link,
                    ),
                    speculative,
                    reached_ambiguous_report,
                )
                if self._finish_reported_instruction(pc, reached_ambiguous_report):
                    self._pc = target
                    return
                if target == pc:
                    # Direct jump to itself, such as a panic spin loop. No packet follows its passes.
                    self._spin_pc = pc
                    return
                self._pc = target
                continue

            if cls == InsnClass.BRANCH:
                if not self._branch_queue:
                    reported = self._pending_address
                    if reported is None or reported == pc or self._replay_address is not None:
                        # Wait for the next packet with branch decisions.
                        return
                    if (yield from self._jump_to_unreachable(pc, reported, prov)):
                        continue
                    return
                bit = self._branch_queue.popleft()
                taken = bit == 0
                if taken:
                    target = insn.direct_target
                    if target is None:
                        yield from self._flush_gap(GapReason.DECODE_ERROR, f"taken branch at 0x{pc:08x} has no target")
                        return
                    next_pc = target
                else:
                    next_pc = (pc + insn.length) & PC_MASK
                yield from self._output(
                    TraceInstruction(
                        pc=pc,
                        length=insn.length,
                        cls=cls,
                        branch_taken=taken,
                        next_pc=next_pc,
                        provenance=prov,
                    ),
                    speculative,
                    reached_ambiguous_report,
                )
                self._pc = next_pc
                if self._finish_reported_instruction(pc, reached_ambiguous_report):
                    return
                continue

            if cls == InsnClass.UNINFERABLE:
                if reached_ambiguous_report:
                    # This instruction is the reported boundary, not the
                    # source of the still-ambiguous target address.  Its real
                    # destination will be supplied by a later packet.
                    self._report = _Report(pc, len(self._held))
                    self._mark_inferred_report(pc)
                    return
                park = False
                reported = False
                if self._replay_address is not None:
                    target = self._replay_address
                    self._replay_address = None
                elif self._pending_address is not None:
                    target = self._pending_address
                    self._pending_address = None
                    # Reaching the reported address through an uninferable
                    # discontinuity removes the ordinary-first-occurrence
                    # ambiguity.
                    reported = self._ambiguous_reported_pc is not None
                    self._ambiguous_reported_pc = None
                    park = target == self._updiscon_target
                else:
                    # Wait for the next packet with target address.
                    return
                yield from self._output(
                    TraceInstruction(
                        pc=pc,
                        length=insn.length,
                        cls=cls,
                        branch_taken=None,
                        next_pc=target,
                        provenance=prov,
                        link=insn.link,
                    ),
                    speculative,
                    False,
                )
                self._pc = target
                if park:
                    # A trap can follow before the target retires, so it is
                    # not fetched until the next packet.
                    self._updiscon_target = None
                    self._stop_after_reported_pc = None
                    self._parked = _Parked(target, prov)
                    self._paused_at_reported_pc = True
                    return
                if reported:
                    self._report = _Report(target, len(self._held), first_occurrence=False)
                if self._finish_reported_instruction(pc, False):
                    return
                continue

            # Wait for the trap packet to resolve target PC.
            return

    # ------------------------------------------------------------------
    # Helpers
    # ------------------------------------------------------------------

    def _jump_to_unreachable(self, pc: int, target: int, prov: Provenance | None) -> Generator[TraceEvent, None, bool]:
        """Continue at a reported address the code cannot reach, return whether to keep walking.

        A packet decides every branch before the address it reports, so the branch at ``pc`` without
        a decision did not execute. The core went to ``target`` another way.
        """
        park = target == self._updiscon_target
        reported = self._ambiguous_reported_pc is not None
        yield from self._release_held()
        yield from self._release_markers()
        yield self._gap(
            GapReason.UNEXPLAINED_JUMP,
            f"the core went to the reported 0x{target:08x} before the branch at 0x{pc:08x}, without a traced jump",
            prov,
        )
        self._pc = target
        self._last_reported_pc = target
        if park:
            self._parked = _Parked(target, prov)
            self._paused_at_reported_pc = True
            return False
        if reported:
            self._report = _Report(target, len(self._held), first_occurrence=False)
        return True

    def _enqueue_branches(self, branch_map: int, count: int) -> None:
        for i in range(count):
            self._branch_queue.append((branch_map >> i) & 1)

    def _speculating(self) -> bool:
        """Return True when no packet information shows that the next instruction executes."""
        return (
            self._pending_address is None
            and self._replay_address is None
            and self._stop_after_reported_pc is None
            and not self._branch_queue
        )

    def _output(self, insn: TraceInstruction, speculative: bool, report: bool) -> Iterator[TraceEvent]:
        """Yield ``insn`` or hold it until the next packet shows whether it retired.

        ``report`` marks the first inferable occurrence of a reported address.
        """
        if report:
            self._held.append(insn)
            self._report = _Report(insn.pc, len(self._held) - 1)
        elif speculative or self._held:
            self._held.append(insn)
        else:
            yield insn

    def _marker(self, event: TraceEvent) -> Iterator[TraceEvent]:
        """Yield ``event`` after the instructions still held for earlier packets."""
        if self._held or self._held_markers or self._parked is not None or self._deferred_gap is not None:
            self._held_markers.append(event)
        else:
            yield event

    def _release_held(self) -> Iterator[TraceEvent]:
        held, self._held = self._held, []
        yield from held

    def _release_markers(self) -> Iterator[TraceEvent]:
        markers, self._held_markers = self._held_markers, []
        yield from markers

    def _fetch_failed(self, pc: int, reason: str, detail: str, speculative: bool) -> Iterator[TraceEvent]:
        """Report a fetch or decode failure, deferred while no packet shows that ``pc`` executes."""
        if speculative:
            self._deferred_gap = _DeferredGap(pc, reason, detail)
            return
        yield from self._flush_gap(reason, detail)

    def _flush_gap(self, reason: str, detail: str) -> Iterator[TraceEvent]:
        yield from self._release_held()
        yield from self._release_markers()
        yield self._gap(reason, detail, None)

    def _capture_report(self, pc: int) -> bool:
        """Record a reported address the walker already passed or failed to fetch.

        A following trap names that instruction as the one it was taken at;
        any other packet resolves it as an ordinary ambiguous report.
        """
        if (
            self._pc is None
            or self._paused_at_reported_pc
            or self._inferred_reported_pc is not None
            or not self._speculating()
        ):
            return False
        index = next((i for i, insn in enumerate(self._held) if insn.pc == pc), None)
        if index is None:
            deferred = self._deferred_gap
            if deferred is None or deferred.pc != pc:
                return False
            index = len(self._held)
        self._report = _Report(pc, index)
        self._inferred_reported_pc = pc
        return True

    def _settle(self) -> Iterator[TraceEvent]:
        """Retire what the walker held for the previous packet, which no trap interrupted."""
        self._report = None
        yield from self._release_held()
        parked = self._parked
        if parked is not None:
            self._parked = None
            self._stop_after_reported_pc = parked.pc
            self._paused_at_reported_pc = False
            yield from self._walk(parked.provenance)
        deferred = self._deferred_gap
        if deferred is not None:
            self._deferred_gap = None
            yield self._gap(deferred.reason, deferred.detail, None)
        yield from self._release_markers()

    def _rewind_to_report(self) -> Iterator[TraceEvent]:
        """Retire the held instructions up to the reported one and continue after it.

        A Format 3 packet confirms that the first occurrence of a reported
        address was the report boundary, as when the walker reaches it ahead.
        """
        report = self._report
        if report is None or not report.first_occurrence or report.index >= len(self._held):
            return
        kept = self._held[: report.index + 1]
        self._held = []
        self._report = None
        self._deferred_gap = None
        self._pc = kept[-1].next_pc
        yield from kept

    def _resolve_trap(self, exception: bool) -> Generator[TraceEvent, None, int | None]:
        """Yield the held instructions that retired before a trap and return its epc.

        The ESP encoder reports the instruction a trap was taken at in the
        Format 1/2 packet before the trap packet. An exception does not retire
        that instruction, so it is the epc. An interrupt retires it, and the
        epc is the instruction after it, unless it is an updiscon target the
        walker has not fetched, which is the epc itself.
        """
        held, report, parked, deferred = self._held, self._report, self._parked, self._deferred_gap
        self._held = []
        self._report = None
        self._parked = None
        self._deferred_gap = None
        epc: int | None
        reached_deferred = True
        if report is not None:
            if exception:
                keep, epc = report.index, report.pc
                reached_deferred = False
            elif report.index < len(held):
                keep, epc = report.index + 1, held[report.index].next_pc
                reached_deferred = False
            else:
                keep, epc = len(held), None
        elif parked is not None:
            keep, epc = len(held), parked.pc
        else:
            keep, epc = len(held), self._pc
        yield from held[:keep]
        if deferred is not None and reached_deferred:
            yield self._gap(deferred.reason, deferred.detail, None)
            epc = None
        yield from self._release_markers()
        return epc

    def _accept_reported_address(
        self,
        raw_address: int,
        width_bits: int,
        notify: int,
        updiscon: int,
        packet_name: str,
        prov: Provenance,
    ) -> Generator[TraceEvent, None, bool]:
        """Classify and install an address reported by Format 1 or 2.

        ``notify`` is encoded relative to the transmitted address MSB;
        ``updiscon`` is encoded relative to ``notify``.  Their values are
        therefore meaningful through comparison, not as standalone booleans.
        """
        resolved = self._resolve_address(raw_address, width_bits)
        if resolved is not None:
            address_msb = (raw_address >> (width_bits - 1)) & 1
            plain = notify == address_msb and updiscon == notify
            if plain and self._capture_report(resolved):
                self._last_reported_pc = resolved
                return True
        yield from self._settle()
        # Settling can end in a gap that drops the delta anchor
        resolved = self._resolve_address(raw_address, width_bits)
        if resolved is None:
            yield self._gap(GapReason.NO_PC, f"{packet_name} before first sync, cannot anchor", prov)
            return False

        # A following non-Format-3 packet resolves the previous ambiguous
        # occurrence in favor of an updiscon target: execution must encounter
        # that previous reported address once more.
        self._paused_at_reported_pc = False
        self._resume_inferred_report()

        address_msb = (raw_address >> (width_bits - 1)) & 1
        is_notification = notify != address_msb
        ends_before_format3 = updiscon != notify

        self._pending_address = None
        self._stop_after_reported_pc = None
        self._ambiguous_reported_pc = None
        self._updiscon_target = None

        if is_notification:
            # The address names an instruction reported because of a trigger
            # or filter match.  It is explicitly not an updiscon target.
            self._stop_after_reported_pc = resolved
        else:
            self._pending_address = resolved
            if ends_before_format3:
                # The target instruction is also the final instruction before
                # the next exception, privilege change, or resync packet.
                self._stop_after_reported_pc = resolved
                self._updiscon_target = resolved
            else:
                # It may be an updiscon target, or it may be an inferred
                # report boundary.  The next packet disambiguates the case.
                self._ambiguous_reported_pc = resolved

        self._last_reported_pc = resolved
        return True

    def _finish_reported_instruction(self, pc: int, reached_ambiguous_report: bool) -> bool:
        """Return True when walking must pause after the instruction at ``pc``."""
        if self._stop_after_reported_pc == pc:
            self._stop_after_reported_pc = None
            self._ambiguous_reported_pc = None
            self._updiscon_target = None
            self._paused_at_reported_pc = True
            return True
        if reached_ambiguous_report:
            self._mark_inferred_report(pc)
            return True
        return False

    def _mark_inferred_report(self, pc: int) -> None:
        self._pending_address = None
        self._ambiguous_reported_pc = None
        self._inferred_reported_pc = pc

    def _resume_inferred_report(self) -> None:
        if self._inferred_reported_pc is not None:
            self._replay_address = self._inferred_reported_pc
            self._inferred_reported_pc = None

    def _resolve_address(
        self,
        raw_address: int,
        width_bits: int,
    ) -> int | None:
        """Turn a Format 1/2 address field into an absolute PC.

        Returns ``None`` when the walker has no anchor to resolve a DELTA
        against. FULL-mode addresses never depend on prior state.
        """
        if self._effective_mode == abi.AddressMode.FULL:
            pc = (raw_address << self._profile.iaddress_lsb) & self._addr_mask
            return pc
        if self._last_reported_pc is None:
            return None
        if width_bits <= 0:
            return self._last_reported_pc
        sign_bit = 1 << (width_bits - 1)
        signed = raw_address - (1 << width_bits) if raw_address & sign_bit else raw_address
        delta_bytes = signed << self._profile.iaddress_lsb
        return (self._last_reported_pc + delta_bytes) & self._addr_mask

    def _resume_at(self, pc: int, branch: int, prov: Provenance) -> Iterator[TraceEvent]:
        """Anchor the walker at a packet address and walk from there.

        ``branch`` is the decision of the instruction at ``pc`` when that instruction is a branch.
        """
        self._pc = pc
        self._spin_pc = None
        self._last_reported_pc = pc
        self._branch_queue.clear()
        self._clear_reported_address_state()
        try:
            insn = decode(pc, self._fetch(pc))
        except (CodeReadError, InsnError):
            insn = None
        if insn is not None and insn.cls == InsnClass.BRANCH:
            self._branch_queue.append(branch)
        yield from self._walk(prov)

    def _waiting_uninferable(self) -> tuple[int, DecodedInsn] | None:
        """Return the pc and decoding of an uninferable instruction the walker stopped at for its target."""
        if self._pc is None or not self._trace_enabled or self._paused_at_reported_pc:
            return None
        try:
            insn = decode(self._pc, self._fetch(self._pc))
        except (CodeReadError, InsnError):
            return None
        if insn.cls != InsnClass.UNINFERABLE:
            return None
        return self._pc, insn

    def _fetch(self, pc: int) -> bytes:
        """Return the instruction bytes at ``pc``, raising ``CodeReadError`` or ``InsnError``."""
        first_hw = self._image.read(pc, 2)
        if _insn_length(first_hw[0] | (first_hw[1] << 8)) == 2:
            return first_hw
        return self._image.read(pc, 4)

    def _gap(self, reason: str, detail: str, prov: Provenance | None) -> TraceGap:
        """Reset the walker and return the gap that explains why."""
        self._reset_for_gap()
        return TraceGap(reason=reason, detail=detail, provenance=prov)

    def _reset_for_gap(self) -> None:
        self._pc = None
        self._spin_pc = None
        self._last_reported_pc = None
        self._branch_queue.clear()
        self._clear_reported_address_state()

    def _clear_reported_address_state(self) -> None:
        self._pending_address = None
        self._inferred_reported_pc = None
        self._replay_address = None
        self._stop_after_reported_pc = None
        self._ambiguous_reported_pc = None
        self._paused_at_reported_pc = False
        self._report = None
        self._parked = None
        self._updiscon_target = None
        self._deferred_gap = None


__all__ = [
    "GapReason",
    "InputEvent",
    "reconstruct",
]
