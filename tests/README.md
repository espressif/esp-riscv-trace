# esp-riscv-trace Tests

The unit tests build their trace packets in Python. `test_captures.py` decodes real captures and compares the output
with the `expected_*` files. For `decode` only the header, marker and summary lines are compared, because the full
instruction listing is several MB per capture. A SHA-256 of the instruction lines pins them. `decode --last`,
`history` and `stats` outputs are compared in full.

`test_instruction_objdump.py` compares the instruction decoder with `riscv32-esp-elf-objdump` over every app ELF. It
finds objdump on `PATH` or in the ESP-IDF tools directory and is skipped without it.

## Test Data

Every target has the same captures, taken with the ESP-IDF OpenOCD trace test app
(`components/esp_riscv_trace/test_apps/openocd`) built from a clean ESP-IDF checkout of `c66245e0ae0`. One app build
per target serves all three captures:

| File | Capture |
| --- | --- |
| `<target>/stop.esptrace` | OpenOCD loop capture, stopped by `esp riscv_trace stop` |
| `<target>/panic_stop.esptrace` | OpenOCD loop capture, finalized by a panic |
| `<target>/coredump.elf` | core dump of a panic, saved as a core ELF |
| `built_apps/<target>.elf` | the app ELF of all three captures |

Every capture stores the SHA-256 of its app ELF. Replace the captures and the ELF together.

## Update the Captures

Build the app with OpenOCD trace control, core dump to UART and a reproducible build. The app uses a minimal build,
so `COMPONENTS` adds `espcoredump`. Autostart is off, because it would disable OpenOCD start and stop.

The build reproduces the committed ELF byte for byte when:

- ESP-IDF is a clean checkout of `c66245e0ae0` with its tags, so `IDF_VER` has no `-dirty` suffix, and its tools
  come from that checkout's `install.sh`.
- The ESP-IDF path contains no symlink. On macOS `/tmp` is a symlink, so do not build under it.
- The build directory is inside the app directory, as below.

```sh
cd $IDF_PATH/components/esp_riscv_trace/test_apps/openocd
cat > /tmp/sdkconfig.fixture <<'CFG'
# CONFIG_ESP_RISCV_TRACE_AUTOSTART is not set
CONFIG_ESP_RISCV_TRACE_OPENOCD=y
CONFIG_ESP_COREDUMP_ENABLE_TO_UART=y
CONFIG_ESP_COREDUMP_UART_DELAY=0
CONFIG_ESP_COREDUMP_DECODE_DISABLE=y
CONFIG_APP_REPRODUCIBLE_BUILD=y
CONFIG_APP_PROJECT_VER_FROM_CONFIG=y
CONFIG_APP_PROJECT_VER="1"
CFG
idf.py -B build_<target> -D SDKCONFIG=build_<target>/sdkconfig -D "COMPONENTS=main;espcoredump" \
    -D SDKCONFIG_DEFAULTS="sdkconfig.defaults;/tmp/sdkconfig.fixture" set-target <target> build flash
cp build_<target>/riscv_trace_openocd_test.elf <this repo>/tests/built_apps/<target>.elf
```

Run OpenOCD with the `esp riscv_trace` commands, for example `openocd -f board/<target>-builtin.cfg`, and use its
telnet console. The app console is on the external UART. On PT 1.0 targets (ESP32-C6) the OpenOCD build must stop the
encoder before the stop halt, otherwise `stop.esptrace` holds only Debug ROM code. openocd-esp32 MR 929 adds that stop.

1. `stop.esptrace`: `reset run`, wait for `riscv_trace_ocd: ready` on the console, then
   `esp riscv_trace start loop tests/<target>/stop.esptrace`, wait 1 s, `esp riscv_trace stop`.
2. `panic_stop.esptrace`: the same, but type `panic` on the console after 1 s, wait for the halt, then
   `esp riscv_trace stop`.
3. `coredump.elf`: `reset run`, wait for `riscv_trace_ocd: ready`, `esp riscv_trace start loop <any file>`, then
   `shutdown`. The encoder keeps running on every core. Type `panic` on the console, save the lines between
   `CORE DUMP START` and `CORE DUMP END` to `coredump.b64` and convert them to a core ELF:

   ```sh
   esp-coredump --chip <target> info_corefile -t b64 -c coredump.b64 -s tests/<target>/coredump.elf \
       tests/built_apps/<target>.elf
   ```

## Update the Expected Output

```sh
python tests/test_captures.py --update
```

The script runs the CLI from the repository root, because the expected files hold the relative capture path in the
`# source` line. It runs it without `IDF_PATH` and `ESP_ROM_ELF_DIR`, so ROM ELF lookup never changes the output.
Review the diff of the `expected_*` files before committing them.

`test_cli.py` also checks content of `esp32c5/stop.esptrace`, for example `branchy_work()` as the top row of `stats`.
Replacing that capture can need updates there too.
