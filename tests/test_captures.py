# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Decode the real captures in tests/<target>/ and compare the output with the expected files."""

from __future__ import annotations

import hashlib
from pathlib import Path

import pytest
import rich_click as click
from esp_pylib.logger import log
from helpers import ROOT, run_cli

# Relative paths, the expected files hold them in the source line
TESTS = Path("tests")
APPS = TESTS / "built_apps"

# (target, capture, app ELF)
CAPTURES = [
    ("esp32s31", "stop.esptrace", "esp32s31.elf"),
    ("esp32s31", "panic_stop.esptrace", "esp32s31.elf"),
    ("esp32s31", "coredump.elf", "esp32s31.elf"),
    ("esp32c5", "stop.esptrace", "esp32c5.elf"),
    ("esp32c5", "panic_stop.esptrace", "esp32c5.elf"),
    ("esp32c5", "coredump.elf", "esp32c5.elf"),
    ("esp32c6", "stop.esptrace", "esp32c6.elf"),
    ("esp32c6", "panic_stop.esptrace", "esp32c6.elf"),
    ("esp32c6", "coredump.elf", "esp32c6.elf"),
]

# (command, target, capture, app ELF, extra args, expected file suffix)
CASES: list[tuple[str, str, str, str, list[str], str]] = (
    [("decode", *capture, [], "") for capture in CAPTURES]
    + [
        ("decode", "esp32s31", "stop.esptrace", "esp32s31.elf", ["--core", "1"], "_core1"),
        ("decode", "esp32c5", "coredump.elf", "esp32c5.elf", ["--last", "20"], "_last"),
        ("stats", "esp32s31", "stop.esptrace", "esp32s31.elf", [], ""),
        ("stats", "esp32s31", "stop.esptrace", "esp32s31.elf", ["--merge-cores"], "_merged"),
        (
            "stats",
            "esp32s31",
            "stop.esptrace",
            "esp32s31.elf",
            [str(TESTS / "esp32s31" / "panic_stop.esptrace")],
            "_and_panic_stop",
        ),
        ("history", "esp32s31", "coredump.elf", "esp32s31.elf", ["--last", "8", "--json"], "_json"),
        ("profile", "esp32c5", "coredump.elf", "esp32c5.elf", [], ""),
    ]
    + [("history", *capture, [], "") for capture in CAPTURES if capture[1] == "coredump.elf"]
)


def expected_path(cmd: str, target: str, capture: str, suffix: str) -> Path:
    return ROOT / TESTS / target / f"expected_{cmd}_{Path(capture).stem}{suffix}"


def run_case(cmd: str, target: str, capture: str, elf: str, args: list[str]) -> str:
    """Run the CLI without the ESP-IDF environment, so no ROM ELF changes the output."""
    p = run_cli(cmd, TESTS / target / capture, "--elf", APPS / elf, *args, "--quiet", cwd=ROOT)
    assert p.returncode == 0, f"exit {p.returncode}\nstderr:\n{p.stderr}"
    if cmd == "decode" and "--last" not in args:
        # Instruction lines make the full output too large, keep headers, markers and summaries plus a digest
        kept: list[str] = []
        digest = hashlib.sha256()
        for line in p.stdout.splitlines(keepends=True):
            if line.startswith(("#", "---")):
                kept.append(line)
            else:
                digest.update(line.encode())
        kept.append(f"# instruction_lines_sha256 {digest.hexdigest()}\n")
        return "".join(kept)
    return p.stdout


@pytest.mark.parametrize(
    "cmd,target,capture,elf,args,suffix",
    CASES,
    ids=[f"{c[1]}-{c[2]}-{c[0]}{c[5]}" for c in CASES],
)
def test_capture_output(cmd, target, capture, elf, args, suffix):
    expected = expected_path(cmd, target, capture, suffix).read_text(encoding="utf-8")
    assert run_case(cmd, target, capture, elf, args) == expected


@click.command(help=__doc__)
@click.option("--update", is_flag=True, help="rewrite the expected files")
def main(update: bool) -> None:
    if not update:
        raise click.UsageError("nothing to do, pass --update")
    for cmd, target, capture, elf, args, suffix in CASES:
        path = expected_path(cmd, target, capture, suffix)
        path.write_text(run_case(cmd, target, capture, elf, args), encoding="utf-8")
        log.print(str(path.relative_to(ROOT)))


if __name__ == "__main__":
    main()
