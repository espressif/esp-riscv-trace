## v0.3.0 (2026-11-23)

### ✨ New Features

- **analysis**: Add the fault window for the register origin walks *(Erhan Kurubas - 15e2b34)*
- **history**: Name the callee of a call into untraced code *(Erhan Kurubas - f56f0fd)*
- **decoder**: Add the instruction semantics decoder *(Erhan Kurubas - bf35e03)*
- **history**: Count the instructions since the last lost trace *(Erhan Kurubas - 7ba225b)*
- **api**: Expose data symbols, the ELF hash check and HistoryBuilder *(Erhan Kurubas - be69c1c)*

### 🔧 Code Refactoring

- **decoder**: Move the shared RV32 encoding rules to encoding.py *(Erhan Kurubas - a0db23e)*


## v0.2.0 (2026-11-11)

### ✨ New Features

- **api**: Read and describe the trace note of a core dump *(Erhan Kurubas - b655dee)*
- **profile**: Add a profile command that writes flame graph stacks *(Erhan Kurubas - 45e094d)*
- **stats**: Sum several captures into one sampled profile *(Erhan Kurubas - 67b9c1b)*
- **stats**: Add a stats command for instructions per source line *(Erhan Kurubas - 834197c)*
- **history**: Add the history command and the open_capture API *(Erhan Kurubas - 6f5a736)*
- **cli**: Add a decode command that lists the executed instructions *(Erhan Kurubas - a349d34)*
- **history**: Outline the calls, returns and traps up to the fault *(Erhan Kurubas - bf7015f)*
- **decoder**: Reconstruct the executed instructions from the packets *(Erhan Kurubas - ac32951)*
- **decoder**: Unwrap the trace buffer and parse its packets *(Erhan Kurubas - 1cfea6a)*
- **elf**: Read code, symbols and source lines from the ELF files *(Erhan Kurubas - 43f78ec)*
- **containers**: Read the trace from core dumps and OpenOCD captures *(Erhan Kurubas - 7808134)*

### 📖 Documentation

- **readme**: Move the command and API details to docs *(Erhan Kurubas - 9af5407)*
