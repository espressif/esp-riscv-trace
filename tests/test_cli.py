# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Exit codes, flags and output placement of the decode and stats subcommands.

Most checks run ``cli.main`` in process on the ESP32-C5 loop capture. The exact output of every capture is checked
in test_captures.py.
"""

from __future__ import annotations

import contextlib
import io
import re
from collections import Counter
from unittest import mock

import pytest
from helpers import (
    C5_ELF,
    C5_STOP,
    S31_ELF,
    S31_STOP,
    columns,
    make_note,
    make_snapshot,
    run_cli,
    run_main,
)

import esp_riscv_trace
from esp_riscv_trace import cli, session
from esp_riscv_trace.analysis.diagnostics import TraceDiagnostic
from esp_riscv_trace.analysis.stats import StatsSnapshot
from esp_riscv_trace.exporters import text as text_exporter
from esp_riscv_trace.ir.events import InsnClass, Provenance, TraceInstruction, TraceSync
from esp_riscv_trace.snapshot import abi

# Byte offsets in the ESPT file, see containers/espt.py
_ESPT_ABI_MINOR_OFFSET = 33
_ESPT_END_STATUS_OFFSET = -4

_COMMANDS = ["decode", "stats"]


def _run(cmd: str, *extra, expect=None):
    """Run ``cmd`` on the ESP32-C5 loop capture with its app ELF."""
    return run_main(cmd, C5_STOP, "--elf", C5_ELF, *extra, expect=expect)


def _stats_line(stdout: str) -> str:
    return next(ln for ln in stdout.splitlines() if ln.startswith("# stats"))


def _event_lines(stdout: str) -> list[str]:
    return [ln for ln in stdout.splitlines() if ln and not ln.startswith("#")]


def test_module_entry_point_runs_decode() -> None:
    p = run_cli("decode", C5_STOP, "--elf", C5_ELF, "--quiet", expect=cli.EXIT_OK)
    assert "# esp-riscv-trace decode" in p.stdout


def test_module_entry_point_exits_usage_on_bad_argument() -> None:
    p = run_cli("stats", C5_STOP, "--elf", C5_ELF, "--top", "-1", expect=cli.EXIT_USAGE)
    assert "--top" in p.stderr


@pytest.mark.parametrize("args", [["-h"], ["--help"], ["decode", "-h"], ["stats", "--help"]])
def test_help_exits_ok(args: list[str]) -> None:
    p = run_main(*args, expect=cli.EXIT_OK)
    assert "esp-riscv-trace" in p.stdout


def test_root_help_lists_commands() -> None:
    p = run_main("--help", expect=cli.EXIT_OK)
    for cmd in _COMMANDS:
        assert cmd in p.stdout


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_command_help_lists_shared_options(cmd: str) -> None:
    p = run_main(cmd, "--help", expect=cli.EXIT_OK)
    for option in ("--elf", "--core", "--chip", "--rom-elf", "--out", "--quiet", "--debug"):
        assert option in p.stdout


@pytest.mark.parametrize(
    "args",
    [
        [],
        ["bogus"],
        ["decode"],
        ["decode", C5_STOP, "--bogus"],
        ["stats", C5_STOP, "--elf", C5_ELF, "--top", "hello"],
        ["decode", C5_STOP, "--core", "x"],
    ],
)
def test_usage_errors_exit_usage_not_sha_mismatch(args: list) -> None:
    p = run_main(*args, expect=cli.EXIT_USAGE)
    assert "ERROR:" in p.stderr
    assert "-h' for help." in p.stderr


def test_usage_error_reaches_ide() -> None:
    sent = []
    with (
        mock.patch("esp_pylib.logger._ws_is_enabled", return_value=True),
        mock.patch("esp_pylib.logger.send_log_message", side_effect=lambda *a: sent.append(a)),
        contextlib.redirect_stderr(io.StringIO()),
        pytest.raises(SystemExit) as exc,
    ):
        cli.main(["decode"])
    assert exc.value.code == cli.EXIT_USAGE
    assert [(level, suggestion) for level, _, suggestion, *_ in sent] == [
        ("error", "Try 'esp-riscv-trace decode -h' for help.")
    ]


def test_version() -> None:
    p = run_main("--version", expect=cli.EXIT_OK)
    assert p.stdout.strip() == f"esp-riscv-trace v{esp_riscv_trace.__version__}"


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_command_ok(cmd: str) -> None:
    _run(cmd, expect=cli.EXIT_OK)


def test_stats_missing_elf_is_usage() -> None:
    p = run_main("stats", C5_STOP, expect=cli.EXIT_USAGE)
    assert "--elf is required" in p.stderr


def test_stats_missing_capture_is_usage() -> None:
    p = run_main("stats", "/nonexistent.bin", "--elf", C5_ELF, expect=cli.EXIT_USAGE)
    assert "not found" in p.stderr


def test_decode_missing_elf_still_runs() -> None:
    p = run_main("decode", C5_STOP, expect=cli.EXIT_OK)
    assert "chip=esp32c5" in p.stderr


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_core_filter_no_match_is_core_not_found(cmd: str) -> None:
    p = _run(cmd, "--core", "99", expect=cli.EXIT_CORE_NOT_FOUND)
    assert "no core matched" in p.stderr


def test_unknown_chip_is_usage() -> None:
    p = run_main("decode", C5_STOP, "--chip", "foo", expect=cli.EXIT_USAGE)
    assert "'foo'" in p.stderr
    assert "esp32c5" in p.stderr


def test_chip_is_case_insensitive() -> None:
    p = run_main("decode", C5_STOP, "--chip", "ESP32C5", expect=cli.EXIT_OK)
    assert "chip=esp32c5" in p.stderr
    assert "overrides" not in p.stderr


def test_open_session_unknown_chip_raises_usage_error() -> None:
    with (
        pytest.raises(session.UsageError, match="has no supported trace encoder"),
        session.open_session(C5_STOP, chip="foo"),
    ):
        pass


def test_chip_override_warns_on_mismatch() -> None:
    p = run_main("decode", C5_STOP, "--chip", "esp32c6", expect=cli.EXIT_OK)
    assert "overrides the chip recorded in the capture (esp32c5)" in p.stderr


def test_elf_sha_mismatch_exits_sha_mismatch() -> None:
    p = run_main("decode", C5_STOP, "--elf", S31_ELF, "--quiet", expect=cli.EXIT_SHA_MISMATCH)
    assert "Use the ELF built with the captured firmware" in p.stderr


def test_unknown_container_exits_decode_failed(tmp_path) -> None:
    capture = tmp_path / "garbage.bin"
    capture.write_bytes(b"garbage-not-a-capture")
    p = run_main("decode", capture, "--quiet", expect=cli.EXIT_DECODE_FAILED)
    assert "neither an ESPT trace file nor a coredump ELF" in p.stderr
    assert "Traceback" not in p.stderr


def test_capture_directory_is_usage() -> None:
    p = run_main("decode", C5_STOP.parent, expect=cli.EXIT_USAGE)
    assert "Traceback" not in p.stderr


def test_negative_top_rejected_by_parser() -> None:
    p = _run("stats", "--top", "-1", expect=cli.EXIT_USAGE)
    assert "--top" in p.stderr


def test_zero_top_accepted() -> None:
    p = _run("stats", "--top", "0", "--quiet", expect=cli.EXIT_OK)
    assert "# stats" in p.stdout


def test_decode_and_stats_print_the_same_summary() -> None:
    decode = _run("decode").stderr.splitlines()[-1]
    stats = _run("stats").stderr.splitlines()[-1]
    assert decode == stats
    assert re.search(r"instructions=\d+ syncs=\d+ traps=\d+ gaps=\d+", decode)


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_quiet_suppresses_summary(cmd: str) -> None:
    p = _run(cmd, "--quiet")
    assert "instructions=" not in p.stderr


def test_stats_prints_capture_header_before_table() -> None:
    p = _run("stats")
    assert p.stdout.index("# esp-riscv-trace stats") < p.stdout.index("# stats")


def test_stats_prints_sha_match() -> None:
    p = _run("stats")
    assert "# app_elf_sha256    match" in p.stdout


@pytest.mark.parametrize("cmd,table", [("decode", "# summary"), ("stats", "# stats")])
def test_debug_sits_between_header_and_body(cmd: str, table: str) -> None:
    p = _run(cmd, "--debug")
    idx_header = p.stdout.index(f"# esp-riscv-trace {cmd}")
    idx_raw = p.stdout.index("# raw_buffer")
    assert idx_header < idx_raw < p.stdout.index(table)
    assert "<- head" in p.stdout


def test_decode_debug_prefixes_event_lines_with_payload_hex() -> None:
    p = _run("decode", "--debug")
    sync = next(ln for ln in p.stdout.splitlines() if "--- sync" in ln)
    assert re.match(r"^[0-9a-f]{2}( [0-9a-f]{2})*  ", sync)


def test_format_event_debug_prefixes_known_payload() -> None:
    ev = TraceSync(pc=0x1000, privilege=3, provenance=Provenance(0, 8, 1, payload=bytes.fromhex("c01708")))
    line = text_exporter.format_event(ev, debug=True)
    assert line.startswith("c0 17 08")
    assert "sync pc=0x00001000" in line
    assert not text_exporter.format_event(ev).startswith("c0 17 08")


def test_format_event_debug_marks_inferred_instructions() -> None:
    prov = Provenance(0, 27, 2, payload=bytes.fromhex("0296540020"))
    reported = TraceInstruction(
        pc=0x4000D1B6, length=2, cls=InsnClass.UNINFERABLE, branch_taken=None, next_pc=0x4000A92C, provenance=prov
    )
    walked = TraceInstruction(
        pc=0x4000A92C,
        length=2,
        cls=InsnClass.NONE,
        branch_taken=None,
        next_pc=0x4000A92E,
        provenance=prov,
        inferred=True,
    )
    assert text_exporter.format_event(reported, debug=True).startswith("02 96 54 00 20")
    assert text_exporter.format_event(walked, debug=True).startswith("inferred")


def test_decode_debug_marks_inferred_walk_lines() -> None:
    lines = _event_lines(_run("decode", "--debug").stdout)
    assert any(ln.startswith("inferred") for ln in lines)
    assert any(ln[:2].isalnum() and ln[2:3] == " " for ln in lines)


def test_decode_omits_debug_output_by_default() -> None:
    p = _run("decode")
    assert "# raw_buffer" not in p.stdout
    lines = _event_lines(p.stdout)
    assert lines
    assert lines[0].startswith(("0x", "---"))


def test_stats_omits_raw_dump_by_default() -> None:
    assert "# raw_buffer" not in _run("stats").stdout


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_summary_footer_per_core(cmd: str) -> None:
    p = _run(cmd)
    assert "# summary           instructions=" in p.stdout
    assert "top_gap_reasons=" in p.stdout


def test_merge_cores_renders_diagnostic_per_source_core() -> None:
    p = _run("stats", "--merge-cores")
    assert "# core-diagnostic   core=0" in p.stdout
    assert "# summary           instructions=" in p.stdout


def test_stats_top_row_is_the_workload_loop() -> None:
    p = _run("stats")
    rows = [ln for ln in p.stdout.splitlines() if ln.startswith("  ")]
    assert rows, f"no data rows found in:\n{p.stdout}"
    assert columns(rows[0])[2].text == "branchy_work()"


def test_stats_per_block_lists_the_workload_loop() -> None:
    p = _run("stats", "--per-block", "--top", "1")
    assert "# branchy_work()" in p.stdout
    assert "distinct_entry_pcs=" in p.stdout


def test_stats_merge_cores_label_uses_cores_prefix() -> None:
    p = run_main("stats", S31_STOP, "--elf", S31_ELF, "--merge-cores", "--quiet", expect=cli.EXIT_OK)
    assert "cores=0,1" in [c.text for c in columns(_stats_line(p.stdout))]


def test_stats_per_core_label_uses_core_prefix() -> None:
    line = _stats_line(_run("stats").stdout)
    assert "core=0" in line
    assert "cores=" not in line


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_out_file_matches_stdout(tmp_path, cmd: str) -> None:
    out_path = tmp_path / "out.txt"
    p = _run(cmd, "--quiet", "--out", out_path, expect=cli.EXIT_OK)
    assert p.stdout == ""
    assert out_path.read_text() == _run(cmd).stdout


def test_iter_cores_yields_views_of_the_session() -> None:
    with session.open_session(C5_STOP, elf=C5_ELF) as sess:
        views = list(session.iter_cores(sess, None))
        assert len(views) == 1
        v = views[0]
        assert v.session is sess
        assert sess.resolver is not None
        assert v.core.core_id == 0
        assert v.note.chip == "esp32c5"


def test_iter_cores_core_filter_returns_empty_when_no_match() -> None:
    with session.open_session(C5_STOP, elf=C5_ELF) as sess:
        assert list(session.iter_cores(sess, core_filter=99)) == []


def test_open_session_require_elf_true_raises_when_elf_missing() -> None:
    with pytest.raises(session.UsageError), session.open_session(C5_STOP, require_elf=True):
        pass


def test_decode_from_tmp_cwd_succeeds(tmp_path) -> None:
    p = run_cli("decode", C5_STOP, "--elf", C5_ELF, "--quiet", cwd=tmp_path, expect=cli.EXIT_OK)
    assert "# esp-riscv-trace decode" in p.stdout


def test_stats_from_tmp_cwd_shows_elf_source_paths(tmp_path) -> None:
    p = run_cli("stats", C5_STOP, "--elf", C5_ELF, "--quiet", cwd=tmp_path, expect=cli.EXIT_OK)
    assert re.search(r"(?:^|\s)/IDF_PROJECT/main/test_app_main\.c:\d+", p.stdout)


def test_merge_sums_sample_hits() -> None:
    a = make_snapshot(sample_hits={("f", "/x.c", 1): 3, ("g", "/y.c", 2): 1})
    b = make_snapshot(sample_hits={("f", "/x.c", 1): 5})
    merged = StatsSnapshot.merge([a, b])
    assert merged.sample_hits[("f", "/x.c", 1)] == 8
    assert merged.sample_hits[("g", "/y.c", 2)] == 1


def test_merge_sums_block_hits() -> None:
    a = make_snapshot(block_hits={("f", 0x1000): 2})
    b = make_snapshot(block_hits={("f", 0x1000): 5, ("g", 0x2000): 1})
    merged = StatsSnapshot.merge([a, b])
    assert merged.block_hits[("f", 0x1000)] == 7
    assert merged.block_hits[("g", 0x2000)] == 1


def test_merge_sums_unresolved_pcs() -> None:
    a = make_snapshot(unresolved_pcs={"function": 3, "line": 4})
    b = make_snapshot(unresolved_pcs={"function": 1})
    merged = StatsSnapshot.merge([a, b])
    assert merged.unresolved_pcs["function"] == 4
    assert merged.unresolved_pcs["line"] == 4


def test_merge_sums_total_instructions() -> None:
    a = make_snapshot(sample_hits={("f", "/x.c", 1): 10}, total_instructions=10)
    b = make_snapshot(sample_hits={("f", "/x.c", 1): 7}, total_instructions=7)
    assert StatsSnapshot.merge([a, b]).total_instructions == 17


def test_merge_empty_iterable_produces_zero_snapshot() -> None:
    merged = StatsSnapshot.merge([])
    assert merged.total_instructions == 0
    assert len(merged.sample_hits) == 0
    assert len(merged.block_hits) == 0


def test_merge_preserves_counter_type() -> None:
    merged = StatsSnapshot.merge([make_snapshot(sample_hits={("f", "/x.c", 1): 1})])
    assert isinstance(merged.sample_hits, Counter)
    assert isinstance(merged.block_hits, Counter)
    assert isinstance(merged.unresolved_pcs, Counter)


class _FakeCore:
    def __init__(self, core_id: int) -> None:
        self.core_id = core_id


class _FakeView:
    def __init__(self, core_id: int) -> None:
        self.core = _FakeCore(core_id)
        self.note = make_note(
            source=f"fake-core-{core_id}",
            core_id=core_id,
            owner="openocd",
            wrap_state="not_wrapped",
            capacity=512,
            head_offset=142,
            capture_reason="explicit_stop",
        )


def _diag(total: int) -> TraceDiagnostic:
    return TraceDiagnostic(instructions=total, syncs=0, traps=0, gaps=0, gap_reasons=Counter(), top_gap_reasons=())


def _render_two_cores(merge_cores: bool) -> str:
    views = [_FakeView(0), _FakeView(1)]
    triples = [
        (views[0], make_snapshot(sample_hits={("f", "/x.c", 1): 3}), _diag(3)),
        (views[1], make_snapshot(sample_hits={("f", "/x.c", 1): 2, ("g", "/y.c", 1): 5}), _diag(7)),
    ]
    buf = io.StringIO()
    cli._render_stats(buf, triples, top=5, per_block=False, merge_cores=merge_cores, debug=False)  # type: ignore[arg-type]
    return buf.getvalue()


def test_merge_cores_label_lists_every_core() -> None:
    assert "cores=0,1" in _stats_line(_render_two_cores(merge_cores=True))


def test_merge_cores_uses_summed_counts_and_denominator() -> None:
    out = _render_two_cores(merge_cores=True)
    assert "denominator=10" in out
    rows = {c[2].text: c for c in (columns(ln) for ln in out.splitlines() if ln.startswith("  "))}
    assert [c.text for c in rows["f()"][:2]] == ["50.00", "5"]
    assert [c.text for c in rows["g()"][:2]] == ["50.00", "5"]


def test_merge_cores_writes_one_diagnostic_footer_per_source_core() -> None:
    out = _render_two_cores(merge_cores=True)
    assert out.count("# core-diagnostic   core=0") == 1
    assert out.count("# core-diagnostic   core=1") == 1
    assert out.count("# summary           instructions=") == 2


def test_per_core_mode_writes_two_stats_tables() -> None:
    out = _render_two_cores(merge_cores=False)
    assert out.count("# stats") == 2
    assert "core=0" in out
    assert "core=1" in out


@pytest.mark.parametrize("cmd", _COMMANDS)
def test_accepts_rom_elf_flag(cmd: str) -> None:
    # Any real ELF works as a ROM ELF, the app ELF is reused here
    p = _run(cmd, "--rom-elf", C5_ELF, "--quiet", expect=cli.EXIT_OK)
    assert "# rom_elf           provided" in p.stdout
    assert p.stderr == ""


def test_missing_env_warns_and_continues() -> None:
    p = _run("decode", expect=cli.EXIT_OK)
    assert "# rom_elf           missing-env" in p.stdout
    assert "ROM ELF not searched" in p.stderr


def test_missing_rom_elf_path_warns_and_continues() -> None:
    missing = "/definitely/not/a/real/rom.elf"
    p = _run("decode", "--rom-elf", missing, "--quiet", expect=cli.EXIT_OK)
    assert "# rom_elf           not-found" in p.stdout
    assert f"ROM ELF not found for {missing}" in p.stderr


def test_warning_shows_markup_like_path_verbatim() -> None:
    missing = "/tmp/[bold]rom[/bold].elf"
    p = _run("decode", "--rom-elf", missing, "--quiet", expect=cli.EXIT_OK)
    assert f"ROM ELF not found for {missing}" in p.stderr


def test_unreadable_rom_elf_warns_and_continues() -> None:
    p = _run("decode", "--rom-elf", C5_STOP, "--quiet", expect=cli.EXIT_OK)
    assert "# rom_elf           invalid" in p.stdout
    assert f"cannot parse ELF {C5_STOP}" in p.stderr


def test_no_app_elf_skips_rom_lookup() -> None:
    p = run_main("decode", C5_STOP, expect=cli.EXIT_OK)
    assert "# rom_elf" not in p.stdout
    assert "ROM ELF" not in p.stderr


def test_rom_elf_requires_app_elf() -> None:
    p = run_main("decode", C5_STOP, "--rom-elf", C5_ELF, "--quiet", expect=cli.EXIT_USAGE)
    assert "--rom-elf requires --elf" in p.stderr


def _decode_patched(tmp_path, offset: int, value: int):
    data = bytearray(C5_STOP.read_bytes())
    data[offset] = value
    capture = tmp_path / "patched.esptrace"
    capture.write_bytes(data)
    return run_main("decode", capture, "--quiet", expect=cli.EXIT_OK)


def test_fifo_not_empty_at_stop_is_reported(tmp_path) -> None:
    p = _decode_patched(tmp_path, _ESPT_END_STATUS_OFFSET, 1)
    assert "the encoder FIFO did not empty at stop" in p.stderr
    assert "# capture           reason=explicit_stop  fifo_emptied=no\n" in p.stdout


def test_newer_abi_minor_is_reported(tmp_path) -> None:
    newer = abi.ABI_MINOR + 1
    p = _decode_patched(tmp_path, _ESPT_ABI_MINOR_OFFSET, newer)
    assert f"capture uses snapshot ABI 1.{newer}, this decoder supports 1.{abi.ABI_MINOR}" in p.stderr
    assert f"# snapshot_abi      1.{newer}  supported=1.{abi.ABI_MINOR}\n" in p.stdout


def test_clean_capture_has_no_condition_lines() -> None:
    p = run_main("decode", C5_STOP, expect=cli.EXIT_OK)
    assert "fifo_emptied" not in p.stdout
    assert "# snapshot_abi" not in p.stdout
