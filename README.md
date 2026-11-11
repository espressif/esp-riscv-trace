# esp-riscv-trace

Decoder for RISC-V hardware instruction trace captured on Espressif chips.

It reads OpenOCD `.esptrace` captures and `esp-coredump` core ELFs with a trace note. It reconstructs the executed
program flow, resolves each PC to a function and source line through the app ELF, and writes a text dump, a hot-spot
table, the call history that led to a crash, or call stacks for a flame graph. Lost instruction continuity is reported
as a gap.

It is a command-line tool. Other tools, such as `esp-coredump`, read captures through the [Python API][api]. The
other modules are internal and have no stable API.

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

One capture holds only the last few milliseconds before it stopped. [Profiling a running application][profiling]
sums many captures.

## Quick start

```bash
esp-riscv-trace decode trace.esptrace --elf build/app.elf             # text dump of the executed instructions
esp-riscv-trace decode coredump.elf --elf build/app.elf --core 1      # trace from a core dump, one core
esp-riscv-trace stats trace.esptrace --elf build/app.elf --top 20     # hottest source lines
esp-riscv-trace stats trace.esptrace --elf build/app.elf --merge-cores --per-block
esp-riscv-trace history coredump.elf --elf build/app.elf              # calls and returns that led to the crash
esp-riscv-trace decode coredump.elf --elf build/app.elf --last 50     # last 50 instructions before the crash
esp-riscv-trace profile trace.esptrace --elf build/app.elf > app.folded  # call stacks for a flame graph
```

Without `--elf`, `decode` still runs but prints functions as `?()`. ROM addresses resolve through `--rom-elf` or the
ESP-IDF environment.

`-h` after a command lists its options. [docs/commands.md][commands] describes the options and the output of every
command, and [docs/api.md][api] the Python API.

A core dump profile in the Left Heavy view of [speedscope](https://www.speedscope.app):

![speedscope Left Heavy view of a profile](./docs/images/profile-speedscope.png)

## Contributing

This project uses [pre-commit](https://pre-commit.com/) to enforce code formatting and commit style. Set up the
hooks with the development install above.

Run the tests from the repository root with `pytest`. [`tests/README.md`](./tests/README.md) explains the test
captures and how to regenerate them.

## License

Apache 2.0. See [`LICENSE`](./LICENSE).

[api]: https://github.com/espressif/esp-riscv-trace/blob/master/docs/api.md
[commands]: https://github.com/espressif/esp-riscv-trace/blob/master/docs/commands.md
[profiling]: https://github.com/espressif/esp-riscv-trace/blob/master/docs/commands.md#profiling-a-running-application
