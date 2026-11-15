# Python API

```python
from esp_riscv_trace import open_capture

with open_capture("coredump.elf", elf="build/app.elf") as capture:
    for core in capture.cores:
        history = core.history(last=20)  # the same outline as the history command
        if history.fault is not None:
            where = capture.resolve(history.fault.pc)
            print(core.core_id, where.function, where.file, where.line)
        for event in core.events():  # decoded events, oldest first
            ...
```

`open_capture` does not stop on an ELF that differs from the capture. `core.note.elf_sha_status` tells the caller.
Problems that do not stop decoding, such as a missing ROM ELF, are collected in `capture.warnings` instead of being
printed. `capture.image` reads code bytes and names functions. `capture.data_at(address)` names the variable of the
application ELF at an address, as `name` or `name+offset`, from its sized data symbols. It returns `None` without an
ELF.

`history.instructions` counts the instructions up to the fault, and `history.since_loss` those since the trace
started or last lost trace. Code outside the traced range loses no trace, so it does not reset the count.

`HistoryBuilder` builds the same history one event at a time, for a caller that passes the events to other code in
the same pass. Push events until `done` is `True`, then call `close()`. A trap handler that calls the FreeRTOS
scheduler, the function the image names `vTaskSwitchContext`, can resume another task. `StackFrame.unsure` marks
the frames such a handler resumed.

`parse_note` reads the `ESP_RISCV_TRACE` note of a core dump without decoding the buffers, for tools that find the
note themselves. It returns the note header and one record per core, and raises `AbiError` when the note is malformed.
`format_note` describes them in the lines esp-coredump `info_corefile` prints. `compare_elf_sha(recorded, size,
elf_sha256)` compares the first `size` bytes of the recorded hash, `header.app_elf_sha256` with
`header.app_elf_sha256_size`, with the SHA-256 of an ELF. It returns an `ElfShaStatus`: `NO_ELF` when `elf_sha256` is
`None`, `NO_SNAPSHOT_HASH` when `size` is 0, otherwise `MATCH` or `MISMATCH`.

```python
from elftools.elf.elffile import ELFFile
from elftools.elf.segments import NoteSegment

from esp_riscv_trace import NOTE_NAME, NOTE_TYPE, format_note, parse_note

with open("tests/esp32s31/coredump.elf", "rb") as f:
    desc = next(
        bytes(note["n_desc"])
        for segment in ELFFile(f).iter_segments()
        if isinstance(segment, NoteSegment)
        for note in segment.iter_notes()
        if note["n_name"] == NOTE_NAME and note["n_type"] == NOTE_TYPE
    )
header, records = parse_note(desc)
print("\n".join(format_note(header, records)))
```

```text
ABI 1.0, reason PANIC, cores 2
Target 0x0020, chip revision v0.0
App ELF SHA256: 04a6a7ec10b0c0146d34f1a1a476e020203113f61bd46ed24360c03e70d9f5a9
Core 0: memory LOOP, format ET20, address DELTA
  resync PACKET, threshold 128
  buffer 0x2f0116dc, capacity 0x1000, head 0xb30, segment 17
  hw fifo 0x00000001, intr 0x00000000, empty yes, full no, overflow no
...
```
