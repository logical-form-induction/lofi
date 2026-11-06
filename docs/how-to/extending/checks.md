# Turn on the debug checks

The environment variable `LOFI_CHECK` turns on LoFI's invariant checks, which
recheck records, batches, chart cells and scopes as a run passes them
between components. This guide sets the variable, says what it does not change, and
says when to use it.

## What is always checked

Field validation is unconditional. A value is checked once, when it enters
the system: a negative length, a nonfinite score or an empty name is refused
at construction, with or without `LOFI_CHECK`. The variable controls only the
second layer, the invariant checks that would otherwise repeat on every batch.

## Set the variable

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
