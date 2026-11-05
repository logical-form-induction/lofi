# Testing

```bash
uv run pytest
```

Tests mirror the source one to one under `tests/unit`. Suites that cannot mirror
a source module live in separate directories under `tests/`. This includes
conformance and integration suites.

## Test requirements

**A test must exercise behavior, not only import code.** Every public entry
point is called somewhere in the suite.

**`assert` is never validation.** Validation raises, and the suite passes under
`python -O`, which strips every `assert`.

A useful test fails when the behavior it covers is removed or changed.

## Markers and flags

Four markers classify tests. Integration and slow tests are collected but
skipped unless their flags are passed.

| marker | what it means | how to run it |
| --- | --- | --- |
| `integration` | real IO, optional dependencies or credentials | `--run-integration` |
| `slow` | long-running | `--run-slow` |
| `conformance` | a protocol conformance suite against one implementation | runs by default |
| `corpus` | needs a licensed corpus named by an environment variable | skips with a message |

Skipped families remain visible in the short summary. Their flags are declared
in `tests/conftest.py`.

Corpus tests skip with a message naming the required environment variable.
Licensed corpora are never committed or fetched.

## Property tests

Properties use [Hypothesis](https://hypothesis.readthedocs.io/) and set an
explicit `max_examples`.

Select a profile with `HYPOTHESIS_PROFILE`. `dev` is the fast default, `ci`
runs more cases without per-example deadlines, and `debug` prints detailed
search output. A test-level `max_examples` overrides its profile value.

## Continuous integration variants

LoFI supports Linux, macOS and Windows. Continuous integration runs the suite
in four configurations.

1. The default run with the wider Hypothesis profile, on Linux, macOS and
   Windows.
2. Under `python -O`, which removes validation written as an `assert`.
3. With `LOFI_CHECK=0`, which checks that tests do not require debug checks.
   See [The debug checks](../how-to/extending/checks.md).
4. In an interpreter without torch, numpy, quivers, lairs, or hypothesis,
   importing every symbolic-core module.

## Benchmarks

Measure timing with
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Benchmarks are
excluded from the default run because coverage instrumentation changes timing.
Coverage is a minimum threshold in the unit-test job, not a substitute for
behavioral assertions.

A benchmark that holds an operation to an absolute number of seconds asserts it
under the `wall_clock_budget` fixture:

```python
def test_hashing_is_within_budget(
    benchmark: BenchmarkFixture, wall_clock_budget: WallClockBudget
) -> None:
    benchmark(hash, form)

    if wall_clock_budget.enforced:
        assert benchmark.stats.stats.median <= 80e-9
```

The fixture serves any test under `tests/benchmarks`, timed by the benchmark
fixture or by its own clock. The `wall_clock_budgets` option in
`[tool.pytest.ini_options]` is on by default, and every hosted continuous
integration run of the suite turns it off with `-o wall_clock_budgets=false`.
Memory budgets, operation counts, ratios measured within one run and a
component's promise to honor its own declared timeout hold everywhere and take
no fixture.

Continuous integration splits the benchmarks by what they measure.

1. On GitHub-hosted Linux, macOS and Windows runners, every timed benchmark,
   with results saved as workflow artifacts. Absolute wall-clock budgets are
   disabled because shared runners vary in speed. Committed local measurements
   are not used as thresholds for different machines.
2. On hosted Linux, macOS and Windows runners, the commit against its parent.
   One job builds and times the parent in a worktree, then times the commit on
   the same machine with `-o wall_clock_budgets=false`, and fails when a median
   regresses past the `RELATIVE_REGRESSION` threshold declared in
   `.github/workflows/bench.yml`. A commit with no parent is timed against
   nothing: the comparison job records a notice and skips the comparison.
   The timing job still records every benchmark for that commit.
3. On Linux, the Rust core's instruction counts under Valgrind, which fail on a
   rise past the declared `INSTRUCTION_REGRESSION` over the base commit. They
   also run weekly, recording counts without a base to compare against.

To compare two commits the same way locally, save the parent's run to a scratch
storage and compare the commit against it:

```bash
uv run pytest tests/benchmarks --benchmark-only -o wall_clock_budgets=false \
  --benchmark-storage=/path/to/scratch --benchmark-save=parent
uv run pytest tests/benchmarks --benchmark-only -o wall_clock_budgets=false \
  --benchmark-storage=/path/to/scratch --benchmark-compare=0001 \
  --benchmark-compare-fail=median:20%
```
