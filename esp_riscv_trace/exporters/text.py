# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Text exporter for decoded trace events."""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence

from ..analysis.diagnostics import TraceDiagnostic
from ..decoder.framing import PAYLOAD_MAX_LEN
from ..elf import CodeImage, SourceResolver
from ..ir.capture_note import (
    ElfShaStatus,
    TraceCaptureNote,
)
from ..ir.events import (
    InsnClass,
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
from ._format import FUNCTION_COLUMN_WIDTH, marker, one_line, one_line_quoted, short_sha

# Debug hex column fits the largest framed payload
_PAYLOAD_HEX_WIDTH = PAYLOAD_MAX_LEN * 3 - 1


def format_note(note: TraceCaptureNote, command: str = "decode") -> list[str]:
    """Return the header lines to write before the event stream.

    Every line starts with ``# `` so a reviewer or diff tool can tell the
    header apart from the event stream. This also mirrors how objdump
    tags its metadata block.
    """
    lines: list[str] = []
    lines.append(f"# esp-riscv-trace {command}")
    lines.append(f"# source            {one_line(note.source)}")
    lines.append(_format_chip_line(note))
    lines.append(f"# core              {note.core_id}  owner={note.owner}")
    lines.append(f"# packet_format     {note.packet_format}  address_mode={note.address_mode}")
    lines.append(f"# memory_mode       {note.memory_mode}  wrap_state={note.wrap_state}")
    lines.append(f"# resync            mode={note.resync_mode}  threshold={note.resync_threshold}")
    if note.wrap_reason is not None:
        lines.append(f"#                   wrap_reason={one_line(note.wrap_reason)}")
    lines.append(f"# buffer            capacity={note.capacity}  head_offset={note.head_offset}")
    lines.append(
        f"# hw_status         fifo=0x{note.fifo_status_raw:08x}  intr=0x{note.intr_status_raw:08x}  "
        f"empty={'yes' if note.fifo_empty else 'no'}  "
        f"full={'yes' if note.memory_full else 'no'}  "
        f"overflow={'yes' if note.fifo_overflow else 'no'}"
    )
    emptied = "  fifo_emptied=no" if note.fifo_emptied is False else ""
    lines.append(f"# capture           reason={note.capture_reason}{emptied}")
    if note.abi_minor > abi.ABI_MINOR:
        lines.append(
            f"# snapshot_abi      {note.abi_major}.{note.abi_minor}  supported={abi.ABI_MAJOR}.{abi.ABI_MINOR}"
        )
    lines.append(f"# app_elf_sha256    {_format_sha_status(note)}")
    rom_line = _format_rom_elf_line(note)
    if rom_line is not None:
        lines.append(rom_line)
    return lines


def format_capture_set(captures: Sequence[tuple[TraceCaptureNote, TraceDiagnostic]], command: str) -> list[str]:
    """Return the header for the cores of several captures, one ``(note, diagnostic)`` per core.

    The chip and ELF lines come from the first core. Every core gets one line with its summary counts, plus its ELF
    check when that differs from the first one.
    """
    first = captures[0][0]
    lines = [f"# esp-riscv-trace {command}", _format_chip_line(first)]
    lines.append(f"# app_elf_sha256    {_format_sha_status(first)}")
    rom_line = _format_rom_elf_line(first)
    if rom_line is not None:
        lines.append(rom_line)
    lines.append(f"# captures          {len({note.source for note, _diag in captures})}")
    for note, diag in captures:
        sha = f"  elf_sha256={note.elf_sha_status}" if note.elf_sha_status != first.elf_sha_status else ""
        lines.append(
            f"#                   {one_line(note.source)}  core={note.core_id}  instructions={diag.instructions}  "
            f"syncs={diag.syncs}  traps={diag.traps}  gaps={diag.gaps}{sha}"
        )
    return lines


def format_raw_buffer(
    raw: bytes,
    *,
    head_offset: int,
    head_valid: bool,
    core_id: int | None = None,
    width: int = 16,
) -> list[str]:
    """Return a hex dump of one core's physical encoder buffer.

    Lines stay ``# ``-prefixed so they sit with the snapshot header and
    stay distinct from the reconstructed event stream. ``head_offset``
    is marked on the line that contains it.
    """
    core = f"core={core_id}  " if core_id is not None else ""
    lines = [
        f"# raw_buffer       {core}bytes={len(raw)}  head_offset={head_offset}  "
        f"head_valid={'yes' if head_valid else 'no'}"
    ]
    if not raw:
        lines.append("#                  (empty)")
        return lines

    for off in range(0, len(raw), width):
        chunk = raw[off : off + width]
        hex_part = " ".join(f"{b:02x}" for b in chunk)
        hex_part = hex_part.ljust(width * 3 - 1)
        ascii_part = "".join(chr(b) if 32 <= b < 127 else "." for b in chunk)
        at_head = head_valid and (
            off <= head_offset < off + len(chunk) or (head_offset == len(raw) and off + len(chunk) == len(raw))
        )
        marker = "  <- head" if at_head else ""
        lines.append(f"# {off:08x}  {hex_part}  |{ascii_part}|{marker}")
    return lines


def _format_rom_elf_line(note: TraceCaptureNote) -> str | None:
    """Return the ``# rom_elf`` header line, or ``None`` when no ROM lookup ran because there was no ``--elf``."""
    status = note.rom_elf_status
    if status is None:
        return None
    path = one_line(note.rom_elf_path) if note.rom_elf_path else "none"
    return f"# rom_elf           {status}  path={path}"


def format_event(
    event: TraceEvent,
    resolver: SourceResolver | None = None,
    image: CodeImage | None = None,
    debug: bool = False,
) -> str:
    """Render one trace event as a single text line without a newline.

    ``resolver`` resolves an instruction PC to source ``file:line``. A
    missing or failing resolver leaves that column empty. ``image``
    resolves an instruction PC to a function name. A missing or failing
    image or an unresolved PC renders as ``?``. Any control characters
    in resolved names or in gap detail are escaped so one event stays
    on one physical line. ``debug`` prefixes the framed packet payload
    as hex on the event that consumed that packet. Walked instructions
    are tagged ``inferred`` instead so a packet is not shown as if it
    listed every reconstructed PC.
    """
    if isinstance(event, TraceInstruction):
        line = _format_instruction(event, resolver, image)
        inferred = event.inferred
    elif isinstance(event, TraceSync):
        line = _format_sync(event)
        inferred = False
    elif isinstance(event, TraceTrap):
        line = _format_trap(event)
        inferred = False
    elif isinstance(event, TraceGap):
        line = _format_gap(event)
        inferred = False
    elif isinstance(event, TraceBoundary):
        line = _format_boundary(event)
        inferred = False
    elif isinstance(event, TraceWarning):
        line = _format_warning(event)
        inferred = False
    else:
        line = f"# unknown event type {type(event).__name__}"
        inferred = False
    return _with_payload_prefix(line, getattr(event, "provenance", None), debug, inferred=inferred)


def format_diagnostic(diag: TraceDiagnostic) -> list[str]:
    """Return per-core footer lines summarising a ``TraceDiagnostic``.

    ``# summary`` always renders. The reason, warning and boundary lines follow it when they have
    counts, and ``# diagnostic`` only when a named root cause matched. Columns line up with ``format_note``.
    """
    lines: list[str] = []
    lines.append(
        f"# summary           instructions={diag.instructions}  "
        f"syncs={diag.syncs}  traps={diag.traps}  gaps={diag.gaps}"
    )
    if diag.top_gap_reasons:
        parts = [f"{one_line(r)}={c}" for r, c in diag.top_gap_reasons]
        lines.append(f"#                   top_gap_reasons={', '.join(parts)}")
    if diag.warning_reasons:
        parts = [f"{one_line(r)}={c}" for r, c in diag.warning_reasons]
        lines.append(f"#                   warnings={', '.join(parts)}")
    if diag.boundary_kinds:
        parts = [f"{one_line(r)}={c}" for r, c in diag.boundary_kinds]
        lines.append(f"#                   boundaries={', '.join(parts)}")
    if diag.repeated_gaps:
        parts = [f'{one_line(r)} "{one_line_quoted(d)}" x{c}' for r, d, c in diag.repeated_gaps]
        lines.append(f"#                   repeated_gaps={', '.join(parts)}")
    if diag.root_cause is not None:
        lines.append(f"# diagnostic        {one_line(diag.root_cause)}")
        if diag.hint is not None:
            lines.append(f"#                   hint={one_line(diag.hint)}")
    return lines


# ---------------------------------------------------------------------------
# Internal formatters
# ---------------------------------------------------------------------------


def _format_chip_line(note: TraceCaptureNote) -> str:
    chip = note.chip if note.chip is not None else "unknown"
    tid = f"0x{note.target_id:04x}" if note.target_id is not None else "none"
    rev = f"0x{note.chip_revision:04x}" if note.chip_revision is not None else "none"
    return f"# chip              {chip}  target_id={tid}  revision={rev}"


def _format_sha_status(note: TraceCaptureNote) -> str:
    status = note.elf_sha_status
    snap = short_sha(note.snapshot_app_elf_sha256)
    elf = short_sha(note.elf_sha256)
    elf_path = one_line(note.elf_path) if note.elf_path else "none"
    if status == ElfShaStatus.NO_ELF:
        return f"{status}  snapshot={snap}"
    if status == ElfShaStatus.NO_SNAPSHOT_HASH:
        return f"{status}  elf={elf}  path={elf_path}"
    if status == ElfShaStatus.MATCH:
        return f"{status}  sha={elf}  path={elf_path}"
    return f"{status}  snapshot={snap}  elf={elf}  path={elf_path}"


def _format_payload_hex(prov: Provenance | None, *, inferred: bool) -> str:
    if inferred:
        return "inferred".ljust(_PAYLOAD_HEX_WIDTH)
    raw = b"" if prov is None or not prov.payload else prov.payload
    hex_part = " ".join(f"{b:02x}" for b in raw)
    return hex_part.ljust(_PAYLOAD_HEX_WIDTH)


def _with_payload_prefix(
    line: str,
    provenance: Provenance | None,
    debug: bool,
    *,
    inferred: bool = False,
) -> str:
    if not debug:
        return line
    return f"{_format_payload_hex(provenance, inferred=inferred)}  {line}"


def _format_prov(prov: Provenance | None) -> str:
    if prov is None:
        return ""
    if prov.packet_index < 0:
        return f"[frame seg={prov.segment_index} off={prov.stream_offset}]"
    return f"[frame seg={prov.segment_index} off={prov.stream_offset} idx={prov.packet_index}]"


def _format_instruction(
    event: TraceInstruction,
    resolver: SourceResolver | None,
    image: CodeImage | None,
) -> str:
    pc = f"0x{event.pc:08x}"
    func = one_line(_lookup_function(event.pc, image))
    src = one_line(_lookup_source(event.pc, resolver))
    func_col = f"{func}()".ljust(FUNCTION_COLUMN_WIDTH)
    prov = _format_prov(event.provenance)
    tail = _instruction_tail(event)
    line = f"{pc}  {func_col}  {src}" if src else f"{pc}  {func_col}"
    if prov:
        line = f"{line}  {prov}"
    if tail:
        line = f"{line}  {tail}"
    return line


def _instruction_tail(event: TraceInstruction) -> str:
    if event.cls == InsnClass.BRANCH:
        return "# branch=taken" if event.branch_taken else "# branch=not_taken"
    next_pc = f"0x{event.next_pc:08x}" if event.next_pc is not None else "untraced"
    if event.cls == InsnClass.JUMP:
        return f"# jump next={next_pc}"
    if event.cls == InsnClass.UNINFERABLE:
        return f"# uninferable next={next_pc}"
    if event.cls == InsnClass.TRAP:
        return f"# trap next={next_pc}"
    return ""


def _format_sync(event: TraceSync) -> str:
    return marker(f"sync pc=0x{event.pc:08x} priv={event.privilege}", _format_prov(event.provenance))


def _format_trap(event: TraceTrap) -> str:
    async_flag = 1 if event.interrupt else 0
    handler = f"0x{event.handler_pc:08x}" if event.handler_pc is not None else "none"
    epc = f"0x{event.epc:08x}" if event.epc is not None else "none"
    tval = f"0x{event.tval:08x}" if event.tval is not None else "none"
    body = f"trap cause={event.cause} async={async_flag} handler={handler} epc={epc} tval={tval}"
    return marker(body, _format_prov(event.provenance))


def _format_gap(event: TraceGap) -> str:
    body = f'gap {one_line(event.reason)} "{one_line_quoted(event.detail)}"'
    return marker(body, _format_prov(event.provenance))


def _format_boundary(event: TraceBoundary) -> str:
    body = f'boundary {one_line(event.kind)} "{one_line_quoted(event.detail)}"'
    return marker(body, _format_prov(event.provenance))


def _format_warning(event: TraceWarning) -> str:
    body = f'warning {one_line(event.kind)} "{one_line_quoted(event.detail)}"'
    return marker(body, _format_prov(event.provenance))


class _GapRun:
    """One run of consecutive ``TraceGap`` events sharing a reason.

    ``reason`` is what qualifies events as belonging to the same run.
    ``detail`` strings within a run are tracked as a small histogram so
    the coalesced line preserves the per-detail counts. Any non-gap
    event or a gap with a different reason ends the run and a fresh one
    starts. The first-appearance order of details is preserved so the
    output is deterministic across runs.
    """

    __slots__ = (
        "_detail_counts",
        "_detail_order",
        "all_contiguous",
        "all_have_idx",
        "count",
        "first",
        "last",
        "reason",
    )

    def __init__(self, event: TraceGap) -> None:
        self.reason: str = event.reason
        self.first: TraceGap = event
        self.last: TraceGap = event
        self.count: int = 1
        self.all_have_idx: bool = event.provenance is not None and event.provenance.packet_index >= 0
        self.all_contiguous: bool = True
        self._detail_counts: dict[str, int] = {event.detail: 1}
        self._detail_order: list[str] = [event.detail]

    def matches(self, event: TraceGap) -> bool:
        return event.reason == self.reason

    def append(self, event: TraceGap) -> None:
        prev = self.last
        self.last = event
        self.count += 1
        cur_prov = event.provenance
        prev_prov = prev.provenance
        if cur_prov is None or cur_prov.packet_index < 0:
            self.all_have_idx = False
            self.all_contiguous = False
        elif prev_prov is None or prev_prov.packet_index < 0:
            self.all_contiguous = False
        elif cur_prov.packet_index != prev_prov.packet_index + 1:
            # Packet index is not contiguous.
            self.all_contiguous = False
        if event.detail in self._detail_counts:
            self._detail_counts[event.detail] += 1
        else:
            self._detail_counts[event.detail] = 1
            self._detail_order.append(event.detail)

    def details(self) -> list[tuple[str, int]]:
        return [(d, self._detail_counts[d]) for d in self._detail_order]


class GapCoalescer:
    """Collapses consecutive same-reason gap runs for cleaner output."""

    def __init__(
        self,
        format_non_gap: Callable[[TraceEvent], str],
        debug: bool = False,
    ) -> None:
        self._format = format_non_gap
        self._debug = debug
        self._pending: _GapRun | None = None

    def push(self, event: TraceEvent) -> Iterator[str]:
        # Update state before yielding output lines.
        if isinstance(event, TraceGap):
            if self._pending is not None and self._pending.matches(event):
                self._pending.append(event)
                return
            flushed: str | None = None
            if self._pending is not None:
                flushed = self._format_run(self._pending)
            self._pending = _GapRun(event)
            if flushed is not None:
                yield flushed
            return
        flushed = None
        if self._pending is not None:
            flushed = self._format_run(self._pending)
            self._pending = None
        if flushed is not None:
            yield flushed
        if isinstance(event, TraceGap):
            yield _with_payload_prefix(_format_gap(event), event.provenance, self._debug)
        else:
            yield self._format(event)

    def close(self) -> Iterator[str]:
        if self._pending is None:
            return
        run = self._pending
        self._pending = None
        yield self._format_run(run)

    def _format_run(self, run: _GapRun) -> str:
        return _with_payload_prefix(_format_gap_run(run), run.first.provenance, self._debug)


def _format_gap_run(run: _GapRun) -> str:
    """Render a coalesced gap run.

    A single-event run keeps the unchanged ``_format_gap`` layout so
    downstream greps stay stable. A multi-event run whose events all
    share one detail renders the detail inline; a run with multiple
    detail strings renders a bracketed ``["detail" xN, ...]``
    histogram in first-appearance order.
    """
    if run.count == 1:
        return _format_gap(run.first)
    reason = one_line(run.reason)
    details = run.details()
    if len(details) == 1:
        detail = one_line_quoted(details[0][0])
        head = f'--- gap {reason} x{run.count} "{detail}"'
    else:
        parts = [f'"{one_line_quoted(d)}" x{c}' for d, c in details]
        head = f"--- gap {reason} x{run.count} [{', '.join(parts)}]"
    span = _format_prov_span(
        run.first.provenance,
        run.last.provenance,
        run.all_have_idx,
        run.all_contiguous,
    )
    return f"{head} {span} ---" if span else f"{head} ---"


def _format_prov_span(
    first: Provenance | None,
    last: Provenance | None,
    all_have_idx: bool,
    all_contiguous: bool,
) -> str:
    """Render the ``[frames ...]`` suffix that covers a coalesced run.

    Returns ``""`` when either endpoint has no provenance so a run of
    walker-interior gaps stays clean instead of implying a range that
    was never observed. Runs live inside a single container segment in
    practice because a wrap is a boundary event rather than a gap, but
    a wider segment span is still rendered defensively. ``idx=A..B`` is
    printed only when every event in the run carries a real packet
    index, the run stays inside one segment, and every appended index
    equals the previous one plus exactly one. Any of those failing
    omits idx rather than printing a range the reader could take
    literally when the counter wrapped, an index was skipped, or the
    endpoints straddle a wrap boundary.
    """
    if first is None or last is None:
        return ""
    if first.segment_index == last.segment_index:
        seg = f"seg={first.segment_index}"
    else:
        seg = f"seg={first.segment_index}..{last.segment_index}"
    off = f"off={first.stream_offset}..{last.stream_offset}"
    same_segment = first.segment_index == last.segment_index
    idx = f" idx={first.packet_index}..{last.packet_index}" if all_have_idx and same_segment and all_contiguous else ""
    return f"[frames {seg} {off}{idx}]"


def _lookup_function(pc: int, image: CodeImage | None) -> str:
    if image is None:
        return "?"
    try:
        name = image.symbol_at(pc)
    except Exception:
        return "?"
    return name if name else "?"


def _lookup_source(pc: int, resolver: SourceResolver | None) -> str:
    if resolver is None:
        return ""
    try:
        loc = resolver.resolve(pc)
    except Exception:
        return ""
    if loc is None:
        return ""
    return f"{loc.file}:{loc.line}"


__all__ = [
    "GapCoalescer",
    "format_capture_set",
    "format_diagnostic",
    "format_event",
    "format_note",
]
