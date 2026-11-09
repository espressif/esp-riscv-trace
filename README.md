# esp-riscv-trace

Decoder for RISC-V hardware instruction trace captured on Espressif chips.

It reads OpenOCD `.esptrace` captures and `esp-coredump` core ELFs with a trace note. It reconstructs the executed
program flow, resolves each PC to a function and source line through the app ELF, and writes a text dump. Lost
instruction continuity is reported as a gap.

It is a command-line tool. Its modules are internal and have no stable API.

## Supported chips

| Chip | Packet format |
| --- | --- |
| ESP32-C5 | Efficient Trace 2.0 |
| ESP32-C6 | Processor Trace 1.0 |
| ESP32-C61 | Efficient Trace 2.0 |
| ESP32-H2 | Processor Trace 1.0 |
| ESP32-H21 | Processor Trace 1.0 |
| ESP32-H4 | Efficient Trace 2.0 |
| ESP32-P4 | Efficient Trace 2.0 |
| ESP32-S31 | Efficient Trace 2.0 |

## Installation

Requires Python 3.11 or newer.

```bash
pip install esp-riscv-trace
```

Development install:

```bash
pip install -e '.[dev,test]'
pre-commit install -t pre-commit -t commit-msg
```

## Capturing a trace

Enable `CONFIG_ESP_RISCV_TRACE_ENABLE` in the ESP-IDF application. The decoder reads loop-mode captures.

From OpenOCD (needs `CONFIG_ESP_RISCV_TRACE_OPENOCD`), in the telnet or GDB console:

```text
esp riscv_trace start loop trace.esptrace        # start tracing every traced core
esp riscv_trace stop                             # stop and write trace.esptrace
esp riscv_trace start loop trace.esptrace 500    # stop by itself after 500 ms
```

After a panic, the firmware stops the trace before the debugger halts, and `esp riscv_trace stop` writes it.
`esp riscv_trace dump <file>` writes buffers that are already stopped.

From a core dump: with core dump enabled, the trace starts at boot by default and the core dump stores the buffers.
Pass the core dump ELF (for example from `esp-coredump info_corefile -s coredump.elf ...`) to this tool.

## Quick start

```bash
esp-riscv-trace decode trace.esptrace --elf build/app.elf             # text dump of the executed instructions
esp-riscv-trace decode coredump.elf --elf build/app.elf --core 1      # trace from a core dump, one core
esp-riscv-trace decode coredump.elf --elf build/app.elf --last 50     # last 50 instructions before the crash
```

Without `--elf`, `decode` still runs but prints functions as `?()`. ROM addresses resolve through `--rom-elf` or the
ESP-IDF environment.

## Command reference

```
esp-riscv-trace decode  <capture> [options]
```

`-h` or `--help` after a command lists its options. The command output goes to stdout or `--out`. Warnings, errors
and the summary line go to stderr.

Options for every command:

| Option | Meaning |
| --- | --- |
| `--elf PATH` | Application ELF for function names and source lines |
| `--rom-elf PATH` | ROM ELF for ROM addresses (requires `--elf`). Without it, the ESP-IDF environment is searched |
| `--core N` | Process only this core |
| `--chip NAME` | Override the chip recorded in the capture. One of the [supported chips](#supported-chips), written like `esp32c5` |
| `--out PATH` | Write the output to a file instead of stdout |
| `--quiet` | Do not print the summary line on stderr |

`decode` adds:

| Option | Meaning |
| --- | --- |
| `--last N` | Write only the last N instructions up to the first fault and the fault itself, or the last N of the trace when there is none |
| `--debug` | Hex-dump each core's raw encoder buffer after its header, and prefix each event line with its packet payload |

A fault is an exception other than `ecall`. With `--last`, the summary still covers the whole trace.

### Exit codes

| Code | Meaning |
| --- | --- |
| `0` | Success |
| `1` | Usage error (invalid arguments or missing required files) |
| `2` | The ELF SHA-256 differs from the one recorded in the capture |
| `3` | Specified `--core N` not found in capture |
| `4` | Decode failure (corrupt container, unsupported hardware, or bad ELF) |

## Output format

`decode` prints a `#` header per core, then the events, then a summary. Shortened output of
`esp-riscv-trace decode tests/esp32c5/stop.esptrace --elf tests/built_apps/esp32c5.elf`:

```text
# chip              esp32c5  target_id=0x0017  revision=0x0065
# core              0  owner=fw
# packet_format     efficient_trace_2_0  address_mode=delta
# app_elf_sha256    match  sha=e09eeed40648e3d7  path=tests/built_apps/esp32c5.elf
...
--- sync pc=0x408023a6 priv=1 [frame seg=0 off=1051 idx=22461] ---
0x408023a6  branchy_work()                    /IDF_PROJECT/main/test_app_main.c:35  [frame seg=0 off=1051 idx=22461]
0x408023b8  branchy_work()                    /IDF_PROJECT/main/test_app_main.c:31  [frame seg=0 off=1059 idx=22462]  # branch=not_taken
...
--- boundary wrap "circular trace memory wraps here" [frame seg=1 off=3264] ---
...
# summary           instructions=48431  syncs=3  traps=0  gaps=21
#                   top_gap_reasons=walker_not_anchored=19, alignment_lost=1, partial_frame=1
```

Event lines are instructions, `sync` (the decoder locks on a known PC), `trap`, `gap` (instruction continuity lost, for
example FIFO overflow or a broken frame), `boundary` (the circular buffer wraps) and `warning`. Only gaps count as
lost instructions.

The encoder does not trace ROM code. A call into ROM shows as the jump with `next=untraced`, then a gap `untraced`,
then a `sync` where the core comes back into traced code:

```text
0x4200a842  _read_r_console()                 /IDF/components/esp_stdio/stdio_syscalls_simple.c:45  [frame seg=0 off=1724 idx=2043]  # uninferable next=untraced
--- gap untraced "the jump at 0x4200a842 leaves the traced range (qual_status=1)" [frame seg=0 off=1724 idx=2043] ---
--- sync pc=0x4200a846 priv=1 [frame seg=0 off=1729 idx=2044] ---
```

All commands share one pipeline: `containers` loads the per-core buffers, `decoder` orders, frames and parses the
packets and follows the control flow through the ELF, `analysis` builds the history, and `exporters`
write the result.

## Contributing

This project uses [pre-commit](https://pre-commit.com/) to enforce code formatting and commit style. Set up the
hooks with the development install above.

Run the tests from the repository root with `pytest`. [`tests/README.md`](./tests/README.md) explains the test
captures and how to regenerate them.

## License

Apache 2.0. See [`LICENSE`](./LICENSE).
