# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Command-line interface for esp-riscv-trace."""

from __future__ import annotations

import collections
import contextlib
import functools
import json
import sys
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, TextIO, TypeVar

import rich_click as click
from esp_pylib.errors import FatalError
from esp_pylib.excepthook import install_exception_reporting
from esp_pylib.logger import log
from rich.markup import escape

from . import __version__
from .analysis.diagnostics import DiagnosticAccumulator, TraceDiagnostic
from .analysis.history import History, HistoryBuilder, is_fault
from .analysis.profile import Profile, ProfileAccumulator
from .analysis.stats import StatsAccumulator, StatsSnapshot
from .containers.base import Capture
from .exporters import history as history_exporter
from .exporters import profile as profile_exporter
from .exporters import stats_text as stats_exporter
from .exporters import text as text_exporter
from .ir.events import TraceEvent, TraceInstruction, TraceTrap
from .session import CoreView, ShaMismatch, UsageError, iter_cores, open_sessions
from .snapshot.chip import RISCV_TRACE_CHIPS

EXIT_OK = 0
EXIT_USAGE = 1
EXIT_SHA_MISMATCH = 2
EXIT_CORE_NOT_FOUND = 3
EXIT_DECODE_FAILED = 4

_DEFAULT_TOP = 20
_DEFAULT_HISTORY_LAST = 40

# One core's view with its stats snapshot and diagnostic
_CoreStats = tuple[CoreView, StatsSnapshot, TraceDiagnostic]
# One core's view with its history and diagnostic
_CoreHistory = tuple[CoreView, History, TraceDiagnostic]

_F = TypeVar("_F", bound=Callable[..., Any])


@dataclass(frozen=True)
class _CaptureArgs:
    """The capture argument and the options shared by every subcommand."""

    capture: Path | tuple[Path, ...]
    elf: Path | None
    core: int | None
    chip: str | None
    rom_elf: Path | None
    out: Path | None
    quiet: bool

    @property
    def captures(self) -> tuple[Path, ...]:
        return self.capture if isinstance(self.capture, tuple) else (self.capture,)


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


# rich-click exits with 2 when the group runs without a command, which is EXIT_SHA_MISMATCH here,
# so a missing command is reported as a usage error instead
@click.group(
    context_settings={"help_option_names": ["-h", "--help"]},
    no_args_is_help=False,
)
@click.version_option(__version__, "--version", message="esp-riscv-trace v%(version)s")
def cli_group() -> None:
    """Decode RISC-V hardware trace captured on Espressif chips.

    Reads OpenOCD .esptrace files and esp-coredump ELFs with a trace note.
    """


_PATH = click.Path(path_type=Path)


def _capture_options(command: _F) -> _F:
    """Add the capture argument and the options shared by every subcommand."""
    return _add_options(command, click.argument("capture", type=_PATH, help="path to the trace capture file"))


def _captures_options(command: _F) -> _F:
    """Like ``_capture_options``, for a command that takes one or more captures."""
    argument = click.argument(
        "capture", type=_PATH, nargs=-1, required=True, help="paths to the trace capture files of one app build"
    )
    return _add_options(command, argument)


def _add_options(command: _F, argument: Callable[[_F], _F]) -> _F:
    options: list[Callable[[_F], _F]] = [
        argument,
        click.option("--elf", type=_PATH, default=None, help="application ELF for symbol and DWARF resolution"),
        click.option("--core", type=int, default=None, help="process only this core_id (default is every core)"),
        click.option(
            "--chip",
            type=click.Choice(sorted(RISCV_TRACE_CHIPS), case_sensitive=False),
            default=None,
            metavar="CHIP",
            help="override chip name (default is auto-detect from the container). "
            f"One of {', '.join(sorted(RISCV_TRACE_CHIPS))}",
        ),
        click.option(
            "--rom-elf",
            type=_PATH,
            default=None,
            help="ROM ELF for PCs outside the application ELF (requires --elf). "
            "When not given, search the ESP-IDF environment (IDF_PATH, ESP_ROM_ELF_DIR)",
        ),
        click.option("--out", type=_PATH, default=None, help="write the output to this file (default is stdout)"),
        click.option("--quiet", is_flag=True, help="suppress the one-line summary printed to stderr"),
    ]
    for option in reversed(options):
        command = option(command)
    return command


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point. Returns a process exit code."""
    install_exception_reporting()
    log.set_info_stream(sys.stderr)
    try:
        rc: int = cli_group.main(args=argv, prog_name="esp-riscv-trace", standalone_mode=False)
        return rc
    except click.Abort:
        # rich-click turns Ctrl+C into Abort, restore the default interrupt handling
        raise KeyboardInterrupt from None
    except click.ClickException as exc:
        # Click exits with 2 on usage errors, which is EXIT_SHA_MISMATCH here
        message = escape(exc.format_message())
        ctx = exc.ctx if isinstance(exc, click.UsageError) else None
        if ctx is None:
            log.die(message, exit_code=EXIT_USAGE)
        suggestion = escape(f"Try '{ctx.command_path} -h' for help.")
        log.die(f"{message} {suggestion}", exit_code=EXIT_USAGE, suggestion=suggestion)
    except (OSError, UsageError) as exc:
        log.die(escape(str(exc)), exit_code=EXIT_USAGE)
    except ShaMismatch as exc:
        suggestion = "Use the ELF built with the captured firmware."
        log.die(f"{escape(str(exc))}. {suggestion}", exit_code=EXIT_SHA_MISMATCH, suggestion=suggestion)
    except FatalError as exc:
        log.die(escape(str(exc)), exit_code=EXIT_DECODE_FAILED)


@contextlib.contextmanager
def _session_views(args: _CaptureArgs, *, require_elf: bool) -> Iterator[list[CoreView]]:
    """Yield the selected cores of every capture in order, or exit when no core of a capture matched."""
    with open_sessions(
        args.captures,
        elf=args.elf,
        rom_elf=args.rom_elf,
        chip=args.chip,
        require_elf=require_elf,
    ) as sessions:
        views: list[CoreView] = []
        for session in sessions:
            selected = list(iter_cores(session, args.core))
            if not selected:
                ids = ", ".join(str(c.core_id) for c in session.cap.cores) or "(none)"
                where = f" in {session.cap.source_path}" if len(sessions) > 1 else ""
                log.die(
                    escape(f"no core matched --core {args.core}{where}. Present cores {ids}."),
                    exit_code=EXIT_CORE_NOT_FOUND,
                )
            views.extend(selected)
        yield views


# ---------------------------------------------------------------------------
# decode command
# ---------------------------------------------------------------------------


@cli_group.command(
    "decode",
    short_help="decode a trace capture to a human-readable text dump",
    help=(
        "Auto-detect the container format, reconstruct the executed "
        "PC stream for every core, and write a text dump. Function "
        "and source-line resolution needs --elf."
    ),
)
@_capture_options
@click.option(
    "--debug",
    is_flag=True,
    help=(
        "dump each core's raw encoder buffer after the snapshot header "
        "and prefix event lines with the framed packet payload as hex"
    ),
)
@click.option(
    "--last",
    type=click.IntRange(min=1),
    default=None,
    help=(
        "write only the last N instructions up to the first fault (an exception other than ecall) "
        "and the fault itself, or the last N of the trace when there is no fault"
    ),
)
def _cmd_decode(debug: bool, last: int | None, **capture: Any) -> int:
    args = _CaptureArgs(**capture)
    with _session_views(args, require_elf=False) as views, _out_sink(args.out) as sink:
        diags = [_decode_core(sink, view, first=i == 0, debug=debug, last=last) for i, view in enumerate(views)]
    if not args.quiet:
        _print_summary(views, diags)
    return EXIT_OK


def _decode_core(sink: TextIO, view: CoreView, *, first: bool, debug: bool, last: int | None) -> TraceDiagnostic:
    """Write one core's header, events and diagnostic footer, and return the diagnostic."""
    _write_header(sink, view, "decode", first=first, debug=debug)
    session = view.session
    coalescer = text_exporter.GapCoalescer(
        format_non_gap=functools.partial(
            text_exporter.format_event, resolver=session.resolver, image=session.image, debug=debug
        ),
        debug=debug,
    )
    diag_acc = DiagnosticAccumulator()
    events = _tap(view.events(), diag_acc)
    if last is not None:
        events = _last_before_fault(sink, events, last)
    for ev in events:
        _write_lines(sink, coalescer.push(ev))
    _write_lines(sink, coalescer.close())
    diag = diag_acc.close(view.note)
    _write_lines(sink, text_exporter.format_diagnostic(diag))
    return diag


def _tap(events: Iterator[TraceEvent], diag_acc: DiagnosticAccumulator) -> Iterator[TraceEvent]:
    for ev in events:
        diag_acc.push(ev)
        yield ev


def _last_before_fault(sink: TextIO, events: Iterator[TraceEvent], last: int) -> Iterator[TraceEvent]:
    """Keep the last ``last`` instructions up to the first fault, write a header line, and yield the window.

    The rest of the stream is still consumed, so the diagnostic covers the whole trace.
    """
    window: collections.deque[TraceEvent] = collections.deque()
    instructions = 0
    dropped = 0
    fault: TraceTrap | None = None
    for ev in events:
        window.append(ev)
        if isinstance(ev, TraceInstruction):
            instructions += 1
            while instructions > last:
                if isinstance(window.popleft(), TraceInstruction):
                    instructions -= 1
                    dropped += 1
        if isinstance(ev, TraceTrap) and is_fault(ev):
            fault = ev
            break
    collections.deque(events, maxlen=0)
    if fault is None:
        ends = "end_of_trace"
    else:
        epc = f"0x{fault.epc:08x}" if fault.epc is not None else "none"
        ends = f"fault  cause={fault.cause}  epc={epc}"
    sink.write(f"# window            instructions={instructions}  omitted={dropped}  ends_at={ends}\n")
    yield from window


# ---------------------------------------------------------------------------
# stats command
# ---------------------------------------------------------------------------


@cli_group.command(
    "stats",
    short_help="produce per-source-line and per-block-fragment hit tables",
    help=(
        "Reconstruct the executed PC stream and aggregate hit counts "
        "by (function, file, line) plus per-observed-block-fragment "
        "entry PC. --elf is required because a stats table without "
        "symbols is meaningless. With several captures, for example "
        "a series of OpenOCD captures, the counts of each core are summed."
    ),
)
@_captures_options
@click.option(
    "--top",
    type=click.IntRange(min=0),
    default=_DEFAULT_TOP,
    help=f"show this many highest-count rows (default {_DEFAULT_TOP}). Must be zero or positive.",
)
@click.option("--per-block", is_flag=True, help="also print per-function block-fragment entry PC tables")
@click.option(
    "--merge-cores",
    is_flag=True,
    help="sum stats across every selected core and render one combined table instead of one per core",
)
@click.option("--debug", is_flag=True, help="dump each core's raw encoder buffer after the snapshot header")
def _cmd_stats(top: int, per_block: bool, merge_cores: bool, debug: bool, **capture: Any) -> int:
    args = _CaptureArgs(**capture)
    if debug and len(args.captures) > 1:
        raise UsageError("--debug takes one capture")
    with _session_views(args, require_elf=True) as views:
        per_core = [_collect_stats(view) for view in views]
        with _out_sink(args.out) as sink:
            if len(args.captures) > 1:
                _render_stats_of_captures(sink, per_core, top=top, per_block=per_block, merge_cores=merge_cores)
            else:
                _render_stats(sink, per_core, top=top, per_block=per_block, merge_cores=merge_cores, debug=debug)
    if not args.quiet:
        _print_summary(views, [diag for _view, _snap, diag in per_core])
    return EXIT_OK


def _collect_stats(view: CoreView) -> _CoreStats:
    """Run the statistics and diagnostic accumulators over one core."""
    session = view.session
    if session.resolver is None:
        raise UsageError("--elf is required for this command")
    stats_acc = StatsAccumulator(session.image, session.resolver)
    diag_acc = DiagnosticAccumulator()
    for ev in view.events():
        stats_acc.push(ev)
        diag_acc.push(ev)
    return view, stats_acc.close(), diag_acc.close(view.note)


def _render_stats(
    sink: TextIO,
    per_core: Sequence[_CoreStats],
    *,
    top: int,
    per_block: bool,
    merge_cores: bool,
    debug: bool,
) -> None:
    if merge_cores:
        for i, (view, _snap, _diag) in enumerate(per_core):
            _write_header(sink, view, "stats", first=i == 0, debug=debug)
        merged = StatsSnapshot.merge(snap for _view, snap, _diag in per_core)
        label = "cores=" + ",".join(str(view.core.core_id) for view, _snap, _diag in per_core)
        _write_lines(sink, stats_exporter.format_stats(merged, top=top, per_block=per_block, label=label))
        for view, _snap, diag in per_core:
            sink.write(f"\n# core-diagnostic   core={view.core.core_id}\n")
            _write_lines(sink, text_exporter.format_diagnostic(diag))
        return

    for i, (view, snapshot, diag) in enumerate(per_core):
        _write_header(sink, view, "stats", first=i == 0, debug=debug)
        label = f"core={view.core.core_id}"
        _write_lines(sink, stats_exporter.format_stats(snapshot, top=top, per_block=per_block, label=label))
        _write_lines(sink, text_exporter.format_diagnostic(diag))


def _render_stats_of_captures(
    sink: TextIO,
    per_core: Sequence[_CoreStats],
    *,
    top: int,
    per_block: bool,
    merge_cores: bool,
) -> None:
    """Write one header for all captures, then the tables summed per core, or over every core."""
    _write_lines(sink, text_exporter.format_capture_set([(view.note, diag) for view, _snap, diag in per_core], "stats"))
    by_core: dict[int, list[StatsSnapshot]] = {}
    for view, snap, _diag in per_core:
        by_core.setdefault(view.core.core_id, []).append(snap)
    captures = len({view.session.cap.source_path for view, _snap, _diag in per_core})
    if merge_cores:
        merged = StatsSnapshot.merge(snap for _view, snap, _diag in per_core)
        label = f"cores={','.join(str(core_id) for core_id in sorted(by_core))}  captures={captures}"
        _write_lines(sink, stats_exporter.format_stats(merged, top=top, per_block=per_block, label=label))
        return
    for core_id in sorted(by_core):
        snaps = by_core[core_id]
        label = f"core={core_id}  captures={len(snaps)}"
        _write_lines(
            sink, stats_exporter.format_stats(StatsSnapshot.merge(snaps), top=top, per_block=per_block, label=label)
        )


# ---------------------------------------------------------------------------
# profile command
# ---------------------------------------------------------------------------


@cli_group.command(
    "profile",
    short_help="write the call stacks the instructions ran under, for a flame graph",
    help=(
        "Reconstruct the executed PC stream, follow the call stack effect of each jump, and count "
        "every instruction under its call stack. Writes folded stacks for FlameGraph, or a "
        "speedscope file. Counts are retired instructions, not time. With several captures, the "
        "stacks of each core are summed. --elf is required to name the functions."
    ),
)
@_captures_options
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["folded", "speedscope"]),
    default="folded",
    help="folded stacks, one per line (default), or a speedscope JSON file",
)
@click.option("--merge-cores", is_flag=True, help="sum every core into one profile instead of one per core")
def _cmd_profile(fmt: str, merge_cores: bool, **capture: Any) -> int:
    args = _CaptureArgs(**capture)
    with _session_views(args, require_elf=True) as views:
        per_core = [_collect_profile(view) for view in views]
        by_core: dict[int, list[Profile]] = {}
        for view, profile, _diag in per_core:
            by_core.setdefault(view.core.core_id, []).append(profile)
        if merge_cores:
            name = "cores " + ",".join(str(core_id) for core_id in sorted(by_core))
            named = [(name, Profile.merge(p for core_id in sorted(by_core) for p in by_core[core_id]))]
        else:
            named = [(f"core {core_id}", Profile.merge(by_core[core_id])) for core_id in sorted(by_core)]
        with _out_sink(args.out) as sink:
            if fmt == "folded":
                _write_lines(sink, profile_exporter.format_folded([(None if merge_cores else n, p) for n, p in named]))
            else:
                captures = args.captures
                title = str(captures[0]) if len(captures) == 1 else f"{len(captures)} captures"
                json.dump(profile_exporter.profile_to_speedscope(named, title), sink)
                sink.write("\n")
    if not args.quiet:
        _print_summary(views, [diag for _view, _profile, diag in per_core])
    return EXIT_OK


def _collect_profile(view: CoreView) -> tuple[CoreView, Profile, TraceDiagnostic]:
    """Run the profile and diagnostic accumulators over one core."""
    profile_acc = ProfileAccumulator(view.session.image)
    diag_acc = DiagnosticAccumulator()
    for ev in view.events():
        profile_acc.push(ev)
        diag_acc.push(ev)
    return view, profile_acc.close(), diag_acc.close(view.note)


# ---------------------------------------------------------------------------
# history command
# ---------------------------------------------------------------------------


@cli_group.command(
    "history",
    short_help="outline the calls, returns, interrupts and context switches that led to the crash",
    help=(
        "Reconstruct the executed PC stream and follow the call stack effect of each jump to "
        "outline, per core, the calls and returns, interrupts, context switches and gaps up to "
        "the first fault (an exception other than ecall). Repeated passes of a loop show once "
        "with their count. --elf is required to name the functions."
    ),
)
@_capture_options
@click.option(
    "--last",
    type=click.IntRange(min=0),
    default=_DEFAULT_HISTORY_LAST,
    help=f"show this many newest outline items (default {_DEFAULT_HISTORY_LAST}). 0 shows every item",
)
@click.option("--json", "as_json", is_flag=True, help="write JSON instead of text")
def _cmd_history(last: int, as_json: bool, **capture: Any) -> int:
    args = _CaptureArgs(**capture)
    with _session_views(args, require_elf=True) as views:
        per_core = [_collect_history(view, last) for view in views]
        with _out_sink(args.out) as sink:
            if as_json:
                _write_history_json(sink, per_core)
            else:
                _write_history_text(sink, per_core)
    if not args.quiet:
        _print_summary(views, [diag for _view, _history, diag in per_core])
    return EXIT_OK


def _collect_history(view: CoreView, last: int) -> _CoreHistory:
    """Build one core's history and the diagnostic of its whole trace."""
    builder = HistoryBuilder(view.session.image)
    diag_acc = DiagnosticAccumulator()
    for ev in view.events():
        builder.push(ev)
        diag_acc.push(ev)
    return view, builder.close(last), diag_acc.close(view.note)


def _write_history_text(sink: TextIO, per_core: Sequence[_CoreHistory]) -> None:
    for i, (view, history, diag) in enumerate(per_core):
        _write_header(sink, view, "history", first=i == 0, debug=False)
        session = view.session
        label = f"core={view.core.core_id}"
        _write_lines(sink, history_exporter.format_history(history, session.image, session.resolver, label=label))
        _write_lines(sink, text_exporter.format_diagnostic(diag))


def _write_history_json(sink: TextIO, per_core: Sequence[_CoreHistory]) -> None:
    cores = []
    for view, history, diag in per_core:
        note = view.note
        session = view.session
        cores.append(
            {
                "core": view.core.core_id,
                "chip": note.chip,
                "source": note.source,
                "elf_sha256_status": str(note.elf_sha_status),
                "summary": {
                    "instructions": diag.instructions,
                    "syncs": diag.syncs,
                    "traps": diag.traps,
                    "gaps": diag.gaps,
                },
                "history": history_exporter.history_to_dict(history, session.image, session.resolver),
            }
        )
    json.dump({"command": "history", "cores": cores}, sink, indent=2)
    sink.write("\n")


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_header(sink: TextIO, view: CoreView, command: str, *, first: bool, debug: bool) -> None:
    """Write one core's capture header, with its raw buffer under ``--debug``."""
    if not first:
        sink.write("\n")
    _write_lines(sink, text_exporter.format_note(view.note, command))
    if debug:
        core = view.core
        _write_lines(
            sink,
            text_exporter.format_raw_buffer(
                core.raw, head_offset=core.head_offset, head_valid=core.head_valid, core_id=core.core_id
            ),
        )


def _write_lines(sink: TextIO, lines: Sequence[str] | Iterator[str]) -> None:
    for line in lines:
        sink.write(line + "\n")


@contextlib.contextmanager
def _out_sink(path: Path | None) -> Iterator[TextIO]:
    if path is None:
        yield sys.stdout
        return
    with path.open("w", encoding="utf-8") as f:
        yield f


def _print_summary(views: Sequence[CoreView], diags: Sequence[TraceDiagnostic]) -> None:
    session = views[0].session
    cap: Capture = session.cap
    captures = len({v.session.cap.source_path for v in views})
    source = cap.source_path if captures == 1 else f"{captures} captures"
    ids = ",".join(str(core_id) for core_id in dict.fromkeys(v.core.core_id for v in views))
    summary = (
        f"esp-riscv-trace: {source} chip={session.chip} cores=[{ids}] "
        f"instructions={sum(d.instructions for d in diags)} syncs={sum(d.syncs for d in diags)} "
        f"traps={sum(d.traps for d in diags)} gaps={sum(d.gaps for d in diags)}"
    )
    log.print(escape(summary), file=sys.stderr)


__all__ = [
    "EXIT_CORE_NOT_FOUND",
    "EXIT_DECODE_FAILED",
    "EXIT_OK",
    "EXIT_SHA_MISMATCH",
    "EXIT_USAGE",
    "cli_group",
    "main",
]
