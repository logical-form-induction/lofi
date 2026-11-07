"""Console-interrupt delivery at the native chart's span-width boundary."""

from __future__ import annotations

import signal
import subprocess
import sys
import threading
import time
from typing import TYPE_CHECKING, Final, override

import pytest
from tests.benchmarks.test_chart_cost import Unit, assembly
from tests.interrupts import INTERRUPT, NEW_GROUP, interrupt

from lofi.deduce import Counting, ParsePolicy, Reachability, Retain, parse

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.lexicon import Entry
    from lofi.terms import Term

_TOKENS: Final = (
    "the",
    "dog",
    "is",
    "seeing",
    "a",
    "big",
    "cat",
    "in",
    "the",
    "park",
    "near",
    "the",
    "man",
    "with",
    "a",
    "dog",
    "quickly",
)
_READY: Final = "chart-ready"
_GO: Final = "parse"
_INTERRUPTED: Final = "keyboard-interrupt"
_STARTUP_TIMEOUT: Final = 30.0
_WIDTH_TIMEOUT: Final = 5.0
_NATIVE_ENTRY_DELAY: Final = 0.1


class _ReadyUnit(Unit):
    """Pause after lexical enumeration, immediately before chart storage is filled."""

    @override
    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        """Publish the native-sweep handoff and await the parent process."""
        rows = tuple(super().axioms(tokens))
        sys.stdout.write(f"{_READY}\n")
        sys.stdout.flush()

        if sys.stdin.readline().strip() != _GO:
            message = "the interrupt test parent did not release the chart sweep"
            raise RuntimeError(message)

        return rows


def _child() -> int:
    """Run the current public parse and report how its signal surfaced."""
    signal.signal(INTERRUPT, signal.default_int_handler)
    lowered = assembly("hockenmaier-bisk")
    unit = _ReadyUnit(lowered.lexicon)

    try:
        parse(
            _TOKENS,
            rules=lowered.grammar.rules,
            source=unit,
            semiring=Counting(),
            scorer=unit,
            goal=lowered.grammar.goal,
            category_measure=lowered.category_measure,
            max_category_arity=lowered.grammar.budget.max_category_arity,
            max_category_size=lowered.grammar.budget.max_category_size,
            unary_depth=lowered.grammar.budget.unary_rounds,
            threads=lowered.grammar.budget.threads,
            policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
        )
    except KeyboardInterrupt:
        sys.stdout.write(f"{_INTERRUPTED}\n")
        sys.stdout.flush()
        return 0

    return 1


_CHILD: Final = (
    "from tests.integration.test_chart_interrupt import _child; "
    "raise SystemExit(_child())"
)


def _read_ready(process: subprocess.Popen[str]) -> str:
    """Read the child's sweep handoff without an unbounded pipe wait."""
    stdout = process.stdout

    if stdout is None:
        pytest.fail("the chart child has no stdout pipe")

    read: list[str] = []
    reader = threading.Thread(
        target=lambda: read.append(stdout.readline().strip()), daemon=True
    )
    reader.start()
    reader.join(_STARTUP_TIMEOUT)

    if not read:
        pytest.fail("the chart child did not reach its native sweep")

    return read[0]


@pytest.mark.integration
def test_the_console_interrupt_surfaces_at_the_next_native_span_width() -> None:
    """Interrupt a 17-token parse and observe ``KeyboardInterrupt`` promptly."""
    optimized = ("-O",) if sys.flags.optimize else ()
    process = subprocess.Popen(
        [sys.executable, *optimized, "-c", _CHILD],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        encoding="utf-8",
        creationflags=NEW_GROUP,
    )

    try:
        assert _read_ready(process) == _READY
        stdin = process.stdin

        if stdin is None:
            pytest.fail("the chart child has no stdin pipe")

        stdin.write(f"{_GO}\n")
        stdin.flush()
        time.sleep(_NATIVE_ENTRY_DELAY)
        interrupted_at = time.monotonic()
        interrupt(process)
        stdout, stderr = process.communicate(timeout=_WIDTH_TIMEOUT)
        elapsed = time.monotonic() - interrupted_at
    except subprocess.TimeoutExpired:
        pytest.fail(
            f"the chart did not check the interrupt within {_WIDTH_TIMEOUT} seconds"
        )
    finally:
        if process.poll() is None:
            process.kill()
            process.wait()

    assert process.returncode == 0, stderr
    assert stdout.splitlines() == [_INTERRUPTED]
    assert elapsed <= _WIDTH_TIMEOUT
