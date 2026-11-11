# SPDX-FileCopyrightText: 2026 Espressif Systems (Shanghai) CO LTD
# SPDX-License-Identifier: Apache-2.0
"""Call outline built from synthetic trace events."""

from __future__ import annotations

from helpers import BAR, FOO, IMAGE, ISR, MAIN, SCHEDULER, TASK_B, Trace

from esp_riscv_trace.analysis.history import (
    CodeLoop,
    History,
    HistoryEntry,
    HistoryKind,
    HistoryLoop,
)
from esp_riscv_trace.decoder.reconstruct import GapReason
from esp_riscv_trace.exporters.history import format_history

_STORE_ACCESS_FAULT = 7
_TIMER_IRQ = 7


def _outline(history: History) -> list[tuple[HistoryKind, int, int | None, int | None]]:
    """Kind, depth, pc and target of every entry."""
    return [(e.kind, e.depth, e.pc, e.target) for e in history.entries()]


def _entry(item: object) -> HistoryEntry:
    assert isinstance(item, HistoryEntry)
    return item


def _loop(item: object) -> HistoryLoop:
    assert isinstance(item, HistoryLoop)
    return item


def test_calls_and_returns_nest_at_the_caller_depth() -> None:
    trace = (
        Trace()
        .sync(MAIN)
        .run(MAIN, 2)
        .call(MAIN + 8, FOO)
        .run(FOO, 3)
        .call(FOO + 12, BAR)
        .run(BAR)
        .ret(BAR + 4, FOO + 16)
        .ret(FOO + 16, MAIN + 12)
        .run(MAIN + 12)
    )
    history = trace.history()

    assert _outline(history) == [
        (HistoryKind.START, 0, None, MAIN),
        (HistoryKind.CALL, 0, MAIN + 8, FOO),
        (HistoryKind.CALL, 1, FOO + 12, BAR),
        (HistoryKind.RETURN, 1, BAR + 4, FOO + 16),
        (HistoryKind.RETURN, 0, FOO + 16, MAIN + 12),
    ]
    # Each entry counts the instructions after it, up to and including the jump of the next one
    assert [e.instructions for e in history.entries()] == [3, 4, 2, 1, 1]
    assert history.instructions == 11
    assert history.fault is None
    assert history.omitted == 0
    assert all(not e.detail for e in history.entries())


def test_interrupt_returns_into_the_interrupted_call() -> None:
    trace = (
        Trace()
        .sync(MAIN)
        .call(MAIN, FOO)
        .run(FOO)
        .trap(_TIMER_IRQ, epc=FOO + 4, interrupt=True)
        .run(ISR)
        .mret(ISR + 4, FOO + 4)
        .ret(FOO + 4, MAIN + 4)
    )
    history = trace.history()

    assert _outline(history) == [
        (HistoryKind.START, 0, None, MAIN),
        (HistoryKind.CALL, 0, MAIN, FOO),
        (HistoryKind.INTERRUPT, 1, FOO + 4, ISR),
        (HistoryKind.TRAP_RETURN, 1, ISR + 4, FOO + 4),
        (HistoryKind.RETURN, 0, FOO + 4, MAIN + 4),
    ]
    assert {e.segment for e in history.entries()} == {0}
    assert all(not e.detail for e in history.entries())


def test_interrupt_the_caller_names_ends_the_history() -> None:
    watchdog = 40
    trace = (
        Trace()
        .sync(MAIN)
        .call(MAIN, FOO)
        .trap(_TIMER_IRQ, epc=FOO, interrupt=True)
        .mret(ISR, FOO)
        .jump(FOO, FOO)
        .trap(watchdog, epc=FOO, interrupt=True, tval=0x1234)
        .run(ISR, 5)
    )
    history = trace.history(end_interrupts=(watchdog,))

    assert _outline(history)[-1] == (HistoryKind.FAULT, 1, FOO, ISR)
    fault = history.fault
    assert fault is not None
    assert (fault.cause, fault.interrupt, fault.tval) == (watchdog, True, None)
    assert history.instructions == 3
    # The frame of the call the interrupt was taken in, and not the one of the interrupt
    assert [f.kind for f in history.stack] == [HistoryKind.CALL]
    assert (
        format_history(history, IMAGE, None, label="core 0")[1]
        == "# fault             interrupt 40 in foo  epc=0x00002000"
    )
    # Without it, the history goes on into the handler
    assert trace.history().fault is None


def test_loop_the_history_ends_in_counts_its_calls_and_passes() -> None:
    """while (!ready()) {} with interrupts masked, until the watchdog interrupt."""
    watchdog = 40
    trace = Trace().sync(MAIN).run(MAIN, 2)
    for _ in range(3):
        trace.call(MAIN + 8, FOO).run(FOO).ret(FOO + 4, MAIN + 12).branch(MAIN + 12, MAIN + 8)
    trace.call(MAIN + 8, FOO).trap(watchdog, epc=FOO, interrupt=True)
    history = trace.history(end_interrupts=(watchdog,))

    # The two instructions before the loop are not part of it, so it does not go back to the start of the trace
    assert history.loop == CodeLoop(start=MAIN + 8, end=MAIN + 12, instructions=13, passes=3, whole=False)


def test_trap_return_to_other_code_is_a_context_switch() -> None:
    trace = (
        Trace()
        .sync(MAIN)
        .call(MAIN, FOO)
        .trap(_TIMER_IRQ, epc=FOO, interrupt=True)
        .run(ISR)
        .mret(ISR + 4, TASK_B + 16)
        .ret(TASK_B + 16, BAR + 8)
        .run(BAR + 8)
    )
    history = trace.history()

    switch = _entry(history.items[3])
    assert (switch.kind, switch.target) == (HistoryKind.CONTEXT_SWITCH, TASK_B + 16)
    back = _entry(history.items[4])
    assert back.kind == HistoryKind.RETURN
    assert not back.detail
    # The task's caller is older than the trace, so its return goes one level up from the switch
    assert (switch.segment, switch.depth) == (back.segment, back.depth + 1) == (1, 1)


def test_switch_after_the_scheduler_ran_shows_at_the_trap_return() -> None:
    """Tasks that yield through the same functions return the same way, up to the task code itself."""
    trace = (
        Trace()
        .sync(TASK_B)
        .call(TASK_B, FOO)
        .call(FOO, BAR)
        .trap(_TIMER_IRQ, epc=BAR, interrupt=True)
        .call(ISR, SCHEDULER)
        .ret(SCHEDULER, ISR + 4)
        .mret(ISR + 4, BAR)
        .ret(BAR, FOO + 4)
        # main called foo, task_b did not
        .ret(FOO + 4, MAIN + 4)
        .run(MAIN + 4)
    )
    history = trace.history()

    switch = next(e for e in history.entries() if e.kind == HistoryKind.CONTEXT_SWITCH)
    assert (switch.target, switch.depth) == (BAR, 2)
    assert switch.detail == (
        "the trap handler ran the scheduler and resumed another task at the interrupted instruction"
    )
    last = _entry(history.items[-1])
    assert (last.kind, last.target, last.depth, last.detail) == (HistoryKind.RETURN, MAIN + 4, 0, "")
    assert {e.segment for e in history.entries()} == {0}


def test_return_to_other_code_after_a_trap_return_makes_it_a_context_switch() -> None:
    """The trap return resumed the interrupted instruction, but the return shows another task that runs it."""
    trace = (
        Trace()
        .sync(MAIN)
        .call(MAIN, FOO)
        .trap(_TIMER_IRQ, epc=FOO + 4, interrupt=True)
        .run(ISR)
        .mret(ISR + 4, FOO + 4)
        .ret(FOO + 4, BAR + 8)
        .run(BAR + 8)
    )
    history = trace.history()

    assert _outline(history)[3:] == [
        (HistoryKind.CONTEXT_SWITCH, 1, ISR + 4, FOO + 4),
        (HistoryKind.RETURN, 0, FOO + 4, BAR + 8),
    ]
    assert all(not e.detail for e in history.entries())
    assert "<~ switch to foo" in format_history(history, IMAGE, None, label="core 0")[-2]


def test_return_into_the_interrupted_code_proves_the_trap_resumed_it() -> None:
    trace = (
        Trace()
        .sync(MAIN)
        .call(MAIN, FOO)
        .call(FOO, BAR)
        .trap(_TIMER_IRQ, epc=BAR + 4, interrupt=True)
        .run(ISR)
        .mret(ISR + 4, BAR + 4)
        .ret(BAR + 4, FOO + 4)
        .ret(FOO + 4, BAR + 8)
    )
    history = trace.history()

    # foo was resumed too, so its wrong return is not a context switch
    last = _entry(history.items[-1])
    assert (last.kind, last.target, last.depth) == (HistoryKind.RETURN, BAR + 8, 0)
    assert last.detail == "return address does not match the call, which returns to 0x00001004"
    assert HistoryKind.CONTEXT_SWITCH not in {e.kind for e in history.entries()}


def test_repeated_calls_fold_into_a_loop() -> None:
    trace = Trace().sync(MAIN)
    for _ in range(5):
        trace.call(MAIN, FOO).run(FOO, 2).ret(FOO + 8, MAIN + 4).run(MAIN + 4)
    history = trace.history()

    assert len(history.items) == 2
    loop = _loop(history.items[1])
    assert loop.count == 5
    assert [(e.kind, e.target) for e in map(_entry, loop.body)] == [
        (HistoryKind.CALL, FOO),
        (HistoryKind.RETURN, MAIN + 4),
    ]
    # Every instruction after the first call belongs to the loop
    assert loop.instructions == history.instructions - 1 == 5 * 5 - 1


def test_call_into_untraced_code_returns_at_the_next_sync() -> None:
    trace = Trace().sync(MAIN).call(MAIN, None).gap(GapReason.UNTRACED).sync(MAIN + 4).run(MAIN + 4)
    history = trace.history()

    assert _outline(history) == [
        (HistoryKind.START, 0, None, MAIN),
        (HistoryKind.CALL, 0, MAIN, None),
        (HistoryKind.RETURN, 0, None, MAIN + 4),
    ]
    assert history.instructions == 2


def test_untraced_code_that_calls_into_the_trace_is_a_call() -> None:
    """ROM code calls back into the application."""
    trace = Trace().sync(MAIN).call(MAIN, None).gap(GapReason.UNTRACED).sync(FOO).run(FOO).ret(FOO + 4, None)
    history = trace.history()

    assert _outline(history) == [
        (HistoryKind.START, 0, None, MAIN),
        (HistoryKind.CALL, 0, MAIN, None),
        (HistoryKind.CALL, 1, None, FOO),
        (HistoryKind.RETURN, 1, FOO + 4, None),
    ]


def test_fault_ends_the_history() -> None:
    trace = (
        Trace()
        .sync(MAIN)
        .call(MAIN, FOO)
        .run(FOO)
        .trap(_STORE_ACCESS_FAULT, epc=FOO + 4, tval=0)
        .run(ISR, 10)
        .ret(ISR + 40, MAIN)
    )
    history = trace.history()

    fault = history.fault
    assert fault is not None
    assert fault is history.items[-1]
    assert (fault.kind, fault.depth, fault.pc, fault.target) == (HistoryKind.FAULT, 1, FOO + 4, ISR)
    assert (fault.cause, fault.tval) == (_STORE_ACCESS_FAULT, 0)
    # The handler after the fault is not part of the history
    assert history.instructions == 2
