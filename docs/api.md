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

The trace does not report where a call into code outside the traced range goes. `HistoryEntry.target` of that call
is `None`, unless the instruction that retired right before the call is an `auipc` or `lui` at the address before it
that sets the base register of its `jalr`. The target is then computed from the two instructions, with bit 0 cleared
as `jalr` does, and `HistoryEntry.target_inferred` is `True`. A call the trace reached by a jump, or right after a
sync, a trap or a gap, keeps `None`, as does code the image cannot read or decode. `TraceInstruction.next_pc`
stays as the trace reported it.

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

## Crash analysis

`esp_riscv_trace.analysis.origin` walks back from the trap a fault ends at. It follows the register the fault depends
on to the instructions that wrote it, and finds the load or store of a bus error reported later, the sp at each call
on the stack and the branches of the function the fault is in. Nothing here depends on esp-coredump or its report.

`FaultWindowBuilder`, from the package top level, is the only entry point. It keeps the events while a capture is
decoded and returns a `FaultWindow` that runs the walks over them. The result types, the enums and the limits are
imported from `esp_riscv_trace.analysis.origin`.

### Fault window

`FaultWindowBuilder` is an accumulator like `HistoryBuilder`. It keeps the last `WINDOW_EVENTS` events of one core
and stops at the same event as `HistoryBuilder` with the same `end_interrupts`: the first exception other than an
environment call, or an interrupt whose mcause exception code is in `end_interrupts`. Push each event to both
builders in one pass.

```python
from esp_riscv_trace import FaultWindowBuilder, HistoryBuilder, open_capture

with open_capture("coredump.elf", elf="build/app.elf") as capture:
    core = capture.core(0)
    ends = ()  # mcause codes of the interrupts that run the panic handler
    history_builder = HistoryBuilder(capture.image, end_interrupts=ends)
    window_builder = FaultWindowBuilder(capture.image, end_interrupts=ends)
    for event in core.events():
        history_builder.push(event)
        window_builder.push(event)
        if history_builder.done:
            break
    history = history_builder.close(last=64)
    window = window_builder.close()
    registers = {2: 0x4080F3A0, 15: 0}  # saved x1 to x31 of the code that faulted, by register number
    root = window.origin(registers)
    for step in root.walk() if root is not None else ():
        print(step.reg, step.kind, step.pc, step.instruction, step.stop)
    for call in window.stack_calls(registers):
        print(call.pc, call.sp)
    if window.fault is not None and window.fault.epc is not None:
        function = capture.image.symbol_at(window.fault.epc)
        for branch in window.branches(lambda pc: capture.image.symbol_at(pc) == function):
            print(branch.pc, branch.taken)
```

`FaultWindowBuilder(image, *, end_interrupts=())`:

- `image` is a `CodeImage`, such as `capture.image`. `CodeImage` and `CodeReadError` are imported from
  `esp_riscv_trace.elf`. A read that raises `CodeReadError` is code the walks cannot read. Any other exception
  from `image` fails only the query that made the read. With `None`, as for `HistoryBuilder`, no code can be read
  and no function is named.
- A trap handler that calls the function the image names `vTaskSwitchContext` ends a walk, as `HistoryBuilder`
  finds the scheduler. So does a trap return to a pc other than the one its trap was taken at, or the next one after
  an environment call. The walks do not follow the task back past such a switch, whichever core it ran on before,
  because the trace names no task and the cores are not aligned in time. Values from before it are `unknown` with
  stop `scheduler` or `context_switch`.
- `push(event)` counts every event against `WINDOW_EVENTS`, `TraceBoundary` included, and ignores events after the
  end. `done` is `True` once the end is pushed.
- `close()` returns a `FaultWindow`.

`FaultWindow`:

- `fault` is the `TraceTrap` the window ends at, or `None` when the trace ended first. Without it every query returns
  `None` or an empty tuple.
- `complete` is `True` when the window starts at the first event of the trace, so no event was dropped.
- The walks read `TraceGap` with reason `UNTRACED` as code outside the traced range, and any other gap as lost
  trace. `origin()` records a `TraceWarning` it crosses as `trace_warning`. A `TraceBoundary` is ignored.

### Queries

`registers` maps register numbers 1 to 31 to the values the trap handler saved for the code that faulted. A missing
register is unknown. Each query takes its own `registers`, so a caller can drop the registers `unsure_writes` names.

| Query | Result | Empty result |
| --- | --- | --- |
| `origin(registers, *, access=None)` | `RegisterStep`, the root of a tree | `None`: no end trap, an interrupt, a trap without `epc`, a cause with no register to follow, an unreadable faulting instruction, base register x0, or a fetch fault not reached by an indirect jump or `cm.popret` |
| `bus_access(registers, *, address, store)` | `BusAccess(pc, instruction, index)` | `None`: no load, or store when `store`, to `address` in the `BUS_ERROR_INSTRUCTIONS` instructions that run straight to the trap |
| `stack_calls(registers)` | `tuple[StackCall(pc, sp), ...]`, newest first | Empty without sp. The tuple ends where sp is no longer known |
| `branches(inside)` | `tuple[BranchOutcome(pc, taken), ...]`, oldest first, at most `MAX_BRANCHES` | Empty when the function ran none. `inside(pc)` tells whether a pc is in the function the fault is in |
| `unsure_writes(trapped, saved)` | `frozenset[int]`, the registers written from `trapped` up to `saved`. A `saved` at or before `trapped` gives those the trapped instruction writes | `None` when `saved` is more than `BUS_ERROR_INSTRUCTIONS` instructions after `trapped` or not on an instruction boundary, or that code is unreadable or unsupported |

`origin(access=...)` takes a `BusAccess` from `bus_access()` of the same window and starts from its base register.
Its `index` is the position of the access in the events the window keeps, so a `BusAccess` is not valid with another
window. `origin()` raises `ValueError` when the event at `index` is not an instruction at `pc`.

`branches()` walks back through the function the fault is in. It stops at the call that entered it or at the first pc
`inside` rejects, whichever comes first, so the branches of its callers are not reported. Callees and trap handlers
that returned are skipped. Each branch appears once with its latest outcome, and once `MAX_BRANCHES` branches are
found, earlier ones are left out.

`RegisterStep` is frozen. Each node is a distinct object and `walk()` yields each node once, before its sources.

| Field | Meaning |
| --- | --- |
| `reg` | Register number |
| `kind` | `StepKind`: `set`, `copied`, `computed`, `loaded`, `linked`, `read`, `written`, `pointer` or `unknown` |
| `value` | The value the using instruction read, or `None` |
| `pc`, `instruction` | The instruction that wrote the value, and its disassembly |
| `use_pc` | The instruction that used the value, the faulting one for the root |
| `address` | Where a load read the value |
| `store_pc`, `store_instruction` | The store that wrote the bytes a load read. The stored register is the source |
| `relation` | `Relation`: `callee`, `caller` or `None` |
| `sources` | Steps of the registers the value came from |
| `crossed` | `Crossing` values, in the order the walk met them: `interrupt`, `exception`, `syscall`, `untraced_call`, `trap_entry`, `trace_warning`, `unchecked_value` |
| `stop`, `stop_pc` | `StopReason` of an `unknown` step and the event it stopped at: `trace_start`, `window`, `trace_lost`, `untraced`, `context_switch`, `scheduler`, `syscall`, `unsupported`, `unreadable`, `steps` |
| `detail` | The mnemonic of an unsupported instruction, or the reason of a gap |

`StepKind`, `Relation`, `Crossing` and `StopReason` are `StrEnum`s. A minor release can add members, so callers map
the members they know and treat any other as unknown.

### Failure, lifetime and limits

Queries are independent. A result does not depend on which queries ran before. A query that raises leaves the window
usable for the others, so a caller can keep the history and the other results.

A `FaultWindow` reads code through the image of its capture, so run the queries inside the `open_capture()` block.
The results are plain data and stay valid after it closes. Memory is bounded by `WINDOW_EVENTS` events plus the
caches of one query.

The limits are module constants, not parameters, so explanations that name them describe what ran:

| Constant | Value | Limits |
| --- | --- | --- |
| `WINDOW_EVENTS` | 32768 | Events the window keeps, the fault trap included |
| `MAX_STEPS` | 8 | Steps one `origin()` tree records |
| `MAX_VALUE_DEPTH` | 32 | Writers a value is computed back through |
| `MAX_STORE_EVENTS` | 4096 | Events a load is followed back to its store |
| `BUS_ERROR_INSTRUCTIONS` | 8 | Instructions between a bus error and its trap |
| `MAX_BRANCHES` | 4 | Branches `branches()` returns |

Without saved registers, `origin()` still finds each writer, its kind and its pc. Values and load addresses are
known only where code computes them from constants, so loads from the stack are not followed to their stores.
`stack_calls()` is empty, and `bus_access()` finds an access only through a base register set from constants.
`branches()` and `unsure_writes()` do not use registers.

The caller decides what the walks cannot know: which interrupts end the history, whether the trap matches the saved
exception, whether a bus error access is accepted, when to run each query, and how results are reported.
