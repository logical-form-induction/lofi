# Turn on the debug checks

The environment variable `LOFI_CHECK` turns on LoFI's invariant checks, which
recheck records, batches, chart cells and scopes as a run passes them
between components. Set the variable when you need these checks during development or diagnosis.

## What is always checked

Boundary validation checks a value when it enters the system. An invariant
check examines a condition that should continue to hold as internal values
move between components. A broken invariant can indicate a component bug even
when the original document was valid. [Validation boundaries](../../explanation/boundaries.md#two-kinds-of-check)
explains the distinction; `LOFI_CHECK` controls the repeated internal checks.

Field validation is unconditional. A value is checked once, when it enters
the system: a negative length, a nonfinite score or an empty name is refused
at construction, with or without `LOFI_CHECK`. The variable controls only the
second layer, the invariant checks that would otherwise repeat on every batch.
## Set the variable

An environment variable is a named value made available to a process when it
starts. Set this one in the launching shell before Python imports LoFI. The
[check implementation](https://github.com/logical-form-induction/lofi/blob/main/lofi/_checks.py)
reads it during import, so changing a variable later in an already running
Python session does not update the flags bound by imported modules.

Set `LOFI_CHECK=1` in the environment of the process:

```bash
LOFI_CHECK=1 uv run python -c "from lofi._checks import CHECKS; print(CHECKS)"
```

```text
True
```

Only the value `1` turns the checks on. Any other value, or none, leaves them
off:

```bash
LOFI_CHECK=true uv run python -c "from lofi._checks import CHECKS; print(CHECKS)"
```

```text
False
```

`lofi._checks` reads the variable when it is imported, and each consuming
module binds the value when it is imported, so the variable must be set before
the process starts. Setting `os.environ` after LoFI is imported has no effect.

## When to use it

Turn the checks on while developing or debugging a component, where a broken
invariant should fail at the batch that broke it. Leave them off for
benchmarks and for runs whose timings are reported, since each check costs
time on every batch.

The test suite turns the checks on by default and off under
`--benchmark-only`. Continuous integration also runs the suite once with
`LOFI_CHECK=0`, so a test that passes only because a check caught something
is detected.
