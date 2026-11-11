# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Column layout of the stats sample table and the per-block tables."""

from __future__ import annotations

import pytest
from helpers import columns, make_snapshot

from esp_riscv_trace.analysis.stats import (
    UNRESOLVED_FILE,
    UNRESOLVED_LINE,
)
from esp_riscv_trace.exporters.stats_text import format_stats

# Sample rows are indented by two spaces, per-block rows by ten
_SAMPLE_ROW = "  "
_BLOCK_ROW = " " * 10


def _rows(lines: list[str]) -> list[str]:
    return [ln for ln in lines if ln.startswith(_SAMPLE_ROW)]


def _first_row(lines: list[str]) -> str:
    return _rows(lines)[0]


def _sample_header(lines: list[str]) -> str:
    return next(ln for ln in lines if ln.startswith("#") and "function" in ln)


def test_empty_snapshot_reports_empty_table() -> None:
    lines = format_stats(make_snapshot())
    assert len(lines) == 2
    assert lines[0].startswith("# stats")
    assert "instructions=0" in lines[0]
    assert "denominator=0" in lines[0]
    assert "top=20" in lines[0]
    assert lines[1] == "# no instructions retired, sample table is empty"


def test_label_appears_inside_stats_line_before_totals() -> None:
    lines = format_stats(make_snapshot(sample_hits={("main", "/x.c", 1): 5}), label="core=3")
    assert [c.text for c in columns(lines[0])][:3] == ["# stats", "core=3", "instructions=5"]


def test_no_label_omits_stray_double_space() -> None:
    snap = make_snapshot(sample_hits={("main", "/x.c", 1): 5})
    plain = columns(format_stats(snap)[0])
    labelled = columns(format_stats(snap, label="core=3")[0])
    assert not any(c.text.startswith("core=") for c in plain)
    assert plain[1].text.startswith("instructions=")
    assert plain[1].start == labelled[1].start


def test_top_limits_row_count() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 10, ("b", "/x.c", 2): 9, ("c", "/x.c", 3): 8})
    rows = _rows(format_stats(snap, top=2))
    assert [columns(r)[2].text for r in rows] == ["a()", "b()"]


def test_top_zero_shows_no_data_rows() -> None:
    assert _rows(format_stats(make_snapshot(sample_hits={("a", "/x.c", 1): 5}), top=0)) == []


def test_top_exceeds_row_count_returns_all() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 5, ("b", "/x.c", 2): 3})
    assert len(_rows(format_stats(snap, top=100))) == 2


def test_rows_sort_by_count_desc() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 3, ("b", "/x.c", 1): 10, ("c", "/x.c", 1): 5})
    assert [columns(r)[2].text for r in _rows(format_stats(snap))] == ["b()", "c()", "a()"]


def test_ties_break_on_function_then_file_then_line() -> None:
    snap = make_snapshot(
        sample_hits={("z", "/x.c", 1): 5, ("a", "/x.c", 2): 5, ("a", "/x.c", 1): 5, ("a", "/y.c", 1): 5}
    )
    rows = [[c.text for c in columns(r)[2:]] for r in _rows(format_stats(snap))]
    assert rows == [["a()", "/x.c:1"], ["a()", "/x.c:2"], ["a()", "/y.c:1"], ["z()", "/x.c:1"]]


def test_pct_uses_snapshot_total_by_default() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 50, ("b", "/x.c", 1): 50})
    assert [columns(r)[0].text for r in _rows(format_stats(snap))] == ["50.00", "50.00"]


def test_pct_formats_to_two_decimals() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 1}, total_instructions=3)
    assert columns(_first_row(format_stats(snap)))[0].text == "33.33"


def test_stats_header_and_data_columns_line_up() -> None:
    lines = format_stats(make_snapshot(sample_hits={("a", "/x.c", 1): 1}))
    pct, count, function, file_line = columns(_sample_header(lines))
    row = columns(_first_row(lines))
    assert [c.text for c in (pct, count, function, file_line)] == ["pct", "count", "function", "file:line"]
    assert [c.text for c in row] == ["100.00", "1", "a()", "/x.c:1"]
    assert row[0].end == pct.end
    assert row[1].end == count.end
    assert row[2].start == function.start
    assert row[3].start == file_line.start


def test_block_header_and_data_count_columns_line_up() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 3}, block_hits={("a", 0x1000): 3})
    lines = format_stats(snap, per_block=True)
    header = next(ln for ln in lines if ln.startswith("#") and "count" in ln and "entry_pc" in ln)
    count, entry_pc = columns(header)
    row = columns(next(ln for ln in lines if ln.startswith(_BLOCK_ROW)))
    assert [c.text for c in (count, entry_pc)] == ["count", "entry_pc"]
    assert [c.text for c in row] == ["3", "0x00001000"]
    assert row[0].end == count.end
    assert row[1].start == entry_pc.start


def test_short_function_gets_parens() -> None:
    lines = format_stats(make_snapshot(sample_hits={("main", "/x.c", 1): 1}))
    assert columns(_first_row(lines))[2].text == "main()"


def test_long_function_truncates_with_tilde() -> None:
    name = "an_extremely_long_function_name_over_thirty_two"
    lines = format_stats(make_snapshot(sample_hits={(name, "/x.c", 1): 1}))
    rendered = columns(_first_row(lines))[2].text
    assert len(rendered) <= 32
    assert rendered.endswith("~")
    assert name.startswith(rendered[:-1])


def test_unresolved_function_label_renders_without_parens() -> None:
    label = "<0x400003b0:unresolved>"
    lines = format_stats(make_snapshot(sample_hits={(label, "/x.c", 1): 1}))
    assert columns(_first_row(lines))[2].text == label
    assert not any(label + "()" in ln for ln in lines)


@pytest.mark.parametrize(
    "file_name,line,rendered",
    [
        pytest.param(UNRESOLVED_FILE, UNRESOLVED_LINE, UNRESOLVED_FILE, id="both_unresolved"),
        pytest.param("/known.c", UNRESOLVED_LINE, "/known.c", id="line_unresolved"),
        pytest.param("/known.c", 42, "/known.c:42", id="normal_pair"),
    ],
)
def test_file_line_rendering(file_name: str, line: int, rendered: str) -> None:
    row = _first_row(format_stats(make_snapshot(sample_hits={("f", file_name, line): 1})))
    assert columns(row)[-1].text == rendered
    assert row.rstrip().endswith(rendered)


@pytest.mark.parametrize(
    "unresolved,present,absent",
    [
        pytest.param({"function": 2, "line": 5}, ["function=2", "line=5"], [], id="both"),
        pytest.param({"function": 0, "line": 3}, ["line=3"], ["function=0"], id="zero_hidden"),
    ],
)
def test_unresolved_footer_lists_nonzero_counts(unresolved, present, absent) -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 1}, unresolved_pcs=unresolved)
    footer = [ln for ln in format_stats(snap) if "unresolved_pcs" in ln]
    assert len(footer) == 1
    for text in present:
        assert text in footer[0]
    for text in absent:
        assert text not in footer[0]


def test_all_unresolved_zero_hides_footer() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 1}, unresolved_pcs={"function": 0, "line": 0})
    assert not any("unresolved_pcs" in ln for ln in format_stats(snap))


@pytest.mark.parametrize(
    "block_hits,per_block",
    [
        pytest.param({("a", 0x1000): 1}, False, id="disabled_by_default"),
        pytest.param({}, True, id="no_block_hits"),
    ],
)
def test_per_block_tables_absent(block_hits, per_block: bool) -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 1}, block_hits=block_hits)
    assert not any("blocks" in ln for ln in format_stats(snap, per_block=per_block))


def test_per_block_groups_by_function() -> None:
    snap = make_snapshot(
        sample_hits={("a", "/x.c", 1): 2, ("b", "/x.c", 2): 2},
        block_hits={("a", 0x1000): 2, ("a", 0x1010): 1, ("b", 0x2000): 2},
    )
    text = "\n".join(format_stats(snap, per_block=True))
    assert text.index("# a()") < text.index("# b()")
    assert text.index("0x00001000") < text.index("0x00001010") < text.index("0x00002000")
    assert text.index("# a()") < text.index("0x00001000")
    assert text.index("0x00001010") < text.index("# b()") < text.index("0x00002000")


def test_per_block_orders_entries_by_count_desc() -> None:
    snap = make_snapshot(
        sample_hits={("a", "/x.c", 1): 10},
        block_hits={("a", 0x1000): 3, ("a", 0x1010): 7, ("a", 0x1020): 5},
    )
    rows = [columns(ln) for ln in format_stats(snap, per_block=True) if ln.startswith(_BLOCK_ROW)]
    assert [[c.text for c in r] for r in rows] == [["7", "0x00001010"], ["5", "0x00001020"], ["3", "0x00001000"]]


def test_per_block_functions_not_in_top_still_appear() -> None:
    snap = make_snapshot(sample_hits={("a", "/x.c", 1): 5}, block_hits={("a", 0x1000): 5, ("b", 0x2000): 1})
    text = "\n".join(format_stats(snap, per_block=True, top=1))
    assert "# a()" in text
    assert "# b()" in text
    assert text.index("# a()") < text.index("# b()")


def test_per_block_distinct_count_matches_block_hits_keys() -> None:
    snap = make_snapshot(
        sample_hits={("a", "/x.c", 1): 3},
        block_hits={("a", 0x1000): 1, ("a", 0x1010): 1, ("b", 0x2000): 1},
    )
    header = next(ln for ln in format_stats(snap, per_block=True) if "distinct_entry_pcs" in ln)
    assert "distinct_entry_pcs=3" in header


def test_two_runs_over_same_snapshot_produce_identical_output() -> None:
    snap = make_snapshot(
        sample_hits={("a", "/x.c", 1): 5, ("b", "/y.c", 2): 5, ("c", "/x.c", 1): 5},
        block_hits={("a", 0x1000): 5, ("b", 0x2000): 5, ("c", 0x3000): 5},
    )
    first = format_stats(snap, per_block=True, top=10, label="core=0")
    second = format_stats(snap, per_block=True, top=10, label="core=0")
    assert first == second


@pytest.mark.parametrize(
    "function,file_name,escaped",
    [
        pytest.param("evil\nname", "/x.c", "evil\\nname", id="newline_in_function"),
        pytest.param("evil\rname", "/x.c", "evil\\rname", id="carriage_return_in_function"),
        pytest.param("evil\\name", "/x.c", "evil\\\\name", id="backslash_in_function"),
        pytest.param("f", "/evil\npath.c", "/evil\\npath.c:1", id="newline_in_file"),
        pytest.param("f", "/evil\rpath.c", "/evil\\rpath.c:1", id="carriage_return_in_file"),
        pytest.param("evil\nname", "/evil\npath.c", "/evil\\npath.c:1", id="newline_in_both"),
    ],
)
def test_row_escapes_control_characters(function: str, file_name: str, escaped: str) -> None:
    row = _first_row(format_stats(make_snapshot(sample_hits={(function, file_name, 1): 1})))
    assert "\n" not in row
    assert "\r" not in row
    assert escaped in row


def test_newline_in_label_is_escaped() -> None:
    header = format_stats(make_snapshot(sample_hits={("a", "/x.c", 1): 1}), label="core=0\nspoof")[0]
    assert "\n" not in header
    assert "core=0\\nspoof" in header


def test_negative_top_raises_value_error() -> None:
    with pytest.raises(ValueError):
        format_stats(make_snapshot(sample_hits={("a", "/x.c", 1): 1}), top=-1)


def test_negative_top_message_names_the_argument() -> None:
    with pytest.raises(ValueError) as exc:
        format_stats(make_snapshot(), top=-5)
    assert "top" in str(exc.value)
    assert "-5" in str(exc.value)
