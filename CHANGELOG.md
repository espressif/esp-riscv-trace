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
