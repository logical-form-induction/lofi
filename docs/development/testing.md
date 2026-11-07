# Testing

Run the tests from the repository root after the environment setup in
[CONTRIBUTING.md](https://github.com/logical-form-induction/lofi/blob/main/CONTRIBUTING.md).
`pytest` discovers and executes the checks:

```bash
uv run pytest
```

Tests mirror the source one to one under [`tests/unit`](https://github.com/logical-form-induction/lofi/tree/main/tests/unit). Suites that cannot mirror
a source module live in separate directories under `tests/`. This includes
conformance and integration suites.

While changing span indexing, for instance, run its unit tests directly:

```bash
uv run pytest tests/unit/deduce/test__spans.py
```

A unit test exercises a particular component; an integration test exercises
components working together. A failure reports behavior that disagrees with
the test's expectation. A skip reports an unmet prerequisite or an opt-in
condition; the skipped behavior remains untested. Run the
broader checks required by `CONTRIBUTING.md` before submitting the change.

## Test requirements

**A test must exercise behavior, not only import code.** Every public entry
point is called somewhere in the suite.

**`assert` is never input validation in implementation code.** Such validation
raises an explicit error because `python -O` removes ordinary assertions.
Tests use assertions to state their expected results. [Boundaries](../explanation/boundaries.md#assert-is-never-validation)
explains where validation belongs.

A useful test fails when the behavior it covers is removed or changed.

## Markers and flags

Four markers classify tests. Integration and slow tests are collected but
skipped unless their flags are passed. A **marker** labels a test; a flag
opts into running the corresponding family. For instance, append
`--run-integration` to `uv run pytest` to request integration tests.

| marker | what it means | how to run it |
| --- | --- | --- |
| `integration` | real IO, optional dependencies or credentials | `--run-integration` |
| `slow` | long-running | `--run-slow` |
| `conformance` | a protocol conformance suite against one implementation | runs by default |
| `corpus` | needs a licensed corpus named by an environment variable | skips with a message |

Skipped families remain visible in the short summary. Their flags are declared
in [`tests/conftest.py`](https://github.com/logical-form-induction/lofi/blob/main/tests/conftest.py).

Corpus tests skip with a message naming the required environment variable.
Licensed corpora are never committed or fetched.

## Property tests

Properties use [Hypothesis](https://hypothesis.readthedocs.io/) and set an
explicit `max_examples`.

Instead of checking only a few hand-written inputs, a property test generates
inputs and checks a stated relationship for each. A span index might be
required to recover a span after converting it to an identifier and back.
A [law suite](../reference/laws.md) applies shared requirements to an
implementation of a protocol. Generated cases can expose violations and
provide evidence about the inputs exercised. A proof of the relationship
must cover its full input domain.

Select a profile with [`HYPOTHESIS_PROFILE`](https://github.com/logical-form-induction/lofi/blob/main/tests/conftest.py). `dev` is the fast default, `ci`
runs more cases without per-example deadlines, and `debug` prints detailed
search output. A test-level `max_examples` overrides its profile value.

## Continuous integration variants

Continuous integration (CI) runs automated checks when changes reach GitHub.
The [Python workflow](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/ci.yml)
checks several distinct conditions:

1. The default run with the wider Hypothesis profile, on Linux, macOS and
   Windows.
2. Under `python -O`, which removes validation written as an `assert`.
3. With [`LOFI_CHECK=0`](../how-to/extending/checks.md#set-the-variable), which checks that tests do not require debug checks.
   See [The debug checks](../how-to/extending/checks.md).
4. In an interpreter without torch, numpy, quivers, lairs, or hypothesis,
   importing every symbolic core module.
5. A determinism check runs the unit and golden suites twice with a fixed
   model seed and independently randomized process hash seeds. It then
   compares the exact bytes recorded by the golden recorder. The comparison
   is implemented in
   [`scripts/ci/determinism.py`](https://github.com/logical-form-induction/lofi/blob/main/scripts/ci/determinism.py).

Static checks also inspect formatting, types, and allowed imports. The [Rust](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/rust.yml),
[documentation](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/docs.yml), and [benchmark workflows](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/bench.yml) add checks appropriate to those parts
of the project. All of these workflows contribute to release validation.

## Benchmarks

Measure timing with
[pytest-benchmark](https://pytest-benchmark.readthedocs.io/). Benchmarks are
excluded from the default run because coverage instrumentation changes timing.
The unit test job enforces a minimum coverage threshold alongside behavioral
assertions.

A benchmark that holds an operation to an absolute number of seconds asserts it
under the [`wall_clock_budget`](https://github.com/logical-form-induction/lofi/blob/main/tests/benchmarks/conftest.py) fixture. This excerpt shows the pattern inside
a benchmark test, where `form` and the imported fixture types are already
defined:

```python
def test_hashing_is_within_budget(
    benchmark: BenchmarkFixture, wall_clock_budget: WallClockBudget
) -> None:
    benchmark(hash, form)

    if wall_clock_budget.enforced:
        assert benchmark.stats.stats.median <= 80e-9
```

The fixture serves any test under [`tests/benchmarks`](https://github.com/logical-form-induction/lofi/tree/main/tests/benchmarks), timed by the benchmark
fixture or by its own clock. The `wall_clock_budgets` option in
[`[tool.pytest.ini_options]`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml) is on by default, and every hosted continuous
integration run of the suite turns it off with `-o wall_clock_budgets=false`.
Memory budgets, operation counts, ratios measured within one run and a
component's promise to honor its own declared timeout hold everywhere and take
no fixture.

An absolute time limit depends on the machine running the code. Comparing
two revisions on the same runner reduces that source of variation. An
instruction count instead measures executed machine instructions; it can
detect extra work without treating the elapsed time of a busy shared machine
as an intrinsic property of the implementation.

Continuous integration splits the benchmarks by what they measure.

1. On GitHub-hosted Linux, macOS and Windows runners, bounded benchmarks on
   pushes and pull requests, with results saved as workflow artifacts. The
   objective grid, posterior fit, training step grid, and longer normalization
   cases carry the `slow` mark; weekly and manual runs include them with
   `--run-slow`. The shortest normalization cases run on pushes. Their workloads,
   timing rounds, and assertions are the same in local and hosted runs.
   Absolute wall clock budgets are
   disabled because shared runners vary in speed. Committed local measurements
   are not used as thresholds for different machines.
2. On hosted Linux, macOS and Windows runners, the commit against its parent.
   One job builds and times the parent in a worktree, then times the commit on
   the same machine with `-o wall_clock_budgets=false`, and fails when a median
   regresses past the `RELATIVE_REGRESSION` threshold declared in
   [`.github/workflows/bench.yml`](https://github.com/logical-form-induction/lofi/blob/main/.github/workflows/bench.yml). A commit with no parent is timed against
   nothing: the comparison job records a notice and skips the comparison.
   The timing job still records the selected benchmarks for that commit.
3. On Linux, the Rust core's instruction counts under Valgrind, which fail on a
   rise past the declared `INSTRUCTION_REGRESSION` over the base commit. They
   also run weekly, recording counts without a base to compare against.

To compare two commits locally, use a checkout of the parent for the first
command and a checkout of the changed commit for the second. Both must run
on the same machine and use the same absolute scratch-storage path; their
project environments must correspond to their respective revisions. Saving
and comparing twice from the same revision would not measure the change.

In the parent checkout, save the baseline to an empty scratch directory:

```bash
uv run pytest tests/benchmarks --benchmark-only -o wall_clock_budgets=false \
  --benchmark-storage=/path/to/scratch --benchmark-save=parent
```

In the changed checkout, compare against that saved run. Replace
`/path/to/scratch` with the same directory; `0001` is the saved run number
when the storage was initially empty:

```bash
uv run pytest tests/benchmarks --benchmark-only -o wall_clock_budgets=false \
  --benchmark-storage=/path/to/scratch --benchmark-compare=0001 \
  --benchmark-compare-fail=median:20%
```

The median is the middle of the timing measurements. This comparison fails
when it increases by more than 20%, the relative threshold currently declared
in `.github/workflows/bench.yml`. A root commit has no parent, so this
particular comparison is unavailable until there are two revisions to compare.
