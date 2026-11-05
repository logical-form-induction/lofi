# Choose a sweep design

A sweep's `design` selects which assignments of its axes become points: the
complete `grid`, a seeded `random` sample of it, a seeded `latin-hypercube`
sample, or a resolution-IV `fractional-factorial` fraction of its two-level
axes. An experiment's `evaluation` block then names the multiple-comparison
correction its preregistered comparisons are read under. This guide declares
each design, shows how many assignments each selects, and declares the
correction. [Run and price a sweep](sweeps.md) covers the axes themselves.

## Where the design sits

`design` is a field of the `sweep` block, beside `grid`, `yoked`, `exclude`,
`seeds` and `expected_points`. A sweep that declares no design takes the full
grid:

```yaml
sweep:
  design: {kind: grid}
  grid:
    - path: optimizer.learning_rate
      values: [0.001, 0.0005]
  seeds: [0, 1, 2]
  expected_points: 6
```

Each design is an installed plugin in the `lofi.sweep_design` entry-point
group, and the `kind` names the entry. The design runs first; the expansion
then drops every assignment an `exclude` clause matches, crosses each survivor
with every seed, and collapses any two points whose resolved configuration and
seed agree. `expected_points` is checked against the count that comes out, so
a design that selects fewer assignments than intended is refused with
`UnexpectedPointCountError` before anything is submitted.

| `kind` | Fields | Selects |
| --- | --- | --- |
| `grid` | none | Every combination of the grid axes and yoked groups. |
| `random` | `points`, `seed` | `points` combinations drawn without replacement from the full grid. |
| `latin-hypercube` | `points`, `seed` | `points` combinations in which each dimension's values are spread evenly across the draws. |
| `fractional-factorial` | `generators` | The fraction of the two-level grid axes the generators define, crossed with every other axis. |

## Compare the designs

The four designs can be compared without composing a run document. A
[`Sweep`](../../reference/run.md) validates a `sweep` block, and
`design_assignments` returns the assignments its design selects. Four
two-level axes of the documentation run, under each design:

```python
from lofi.run import Sweep, design_assignments

axes = [
    {"path": "optimizer.learning_rate", "values": [0.001, 0.0005]},
    {"path": "optimizer.weight_decay", "values": [0.0, 0.01]},
    {"path": "trainer.batch_size", "values": [2, 4]},
    {"path": "trainer.epochs", "values": [1, 2]},
]

designs = {
    "grid": {"kind": "grid"},
    "random": {"kind": "random", "points": 5, "seed": 3},
    "latin-hypercube": {"kind": "latin-hypercube", "points": 4, "seed": 3},
    "fractional-factorial": {
        "kind": "fractional-factorial",
        "generators": [
            {
                "factor": "trainer.epochs",
                "word": [
                    "optimizer.learning_rate",
                    "optimizer.weight_decay",
                    "trainer.batch_size",
                ],
            }
        ],
    },
}

for name, design in designs.items():
    sweep = Sweep.model_validate({"design": design, "grid": axes})
    print(name, len(design_assignments(sweep.design, sweep)))
```

```text
grid 16
random 5
latin-hypercube 4
fractional-factorial 8
```

The grid crosses the four axes into sixteen assignments. The two sampled
designs select the number of assignments they declare, and the fraction
selects half the grid.

## Sample at random

A `random` design draws `points` assignments without replacement from the full
grid, using a generator seeded by its own `seed`:

```yaml
design: {kind: random, points: 5, seed: 3}
```

The design `seed` fixes which assignments are drawn; it is separate from the
sweep's `seeds`, which every drawn assignment is crossed with. A design that
asks for more points than the grid holds is refused when the sweep is
expanded.

## Sample a Latin hypercube

A `latin-hypercube` design treats each grid axis and each yoked group as one
dimension. For each dimension it spreads the `points` draws as evenly as the
dimension's values allow, shuffles that column with a generator seeded by
`seed`, and reads one assignment from each row:

```yaml
design: {kind: latin-hypercube, points: 4, seed: 3}
```

With four draws over two-level axes, each value appears in exactly two draws.
Where `points` is smaller than an axis's value count, some values are never
drawn. Two draws that happen to agree on every axis produce one point, so set
`expected_points` to the count of distinct points the design yields.

## Declare a fractional factorial

A `fractional-factorial` design reduces two-level grid axes to the fraction
its `generators` define. Each generator names a generated `factor` and the
`word` of basic axes whose product sets it. The basic axes run through their
full two-level factorial, each generated axis takes the value the parity of
its word's levels gives, and every grid axis no generator names is crossed
with the fraction, as is every yoked group.

The fraction from the comparison above sets `trainer.epochs` from the other
three axes:

```python
sweep = Sweep.model_validate({"design": designs["fractional-factorial"], "grid": axes})
for assignment in design_assignments(sweep.design, sweep):
    print([value for _, value in assignment])
```

```text
[0.001, 0.0, 2, 1]
[0.001, 0.0, 4, 2]
[0.001, 0.01, 2, 2]
[0.001, 0.01, 4, 1]
[0.0005, 0.0, 2, 2]
[0.0005, 0.0, 4, 1]
[0.0005, 0.01, 2, 1]
[0.0005, 0.01, 4, 2]
```

The generators are checked when the sweep block is validated. They must name
distinct generated factors, no generated factor may appear in a word, and
every word of the defining relation they span must have at least four letters,
so that no main effect is aliased with a two-factor interaction. A generator
whose word names two basic axes spans a three-letter word and is refused:

```python
from didactic.api import ValidationError

try:
    Sweep.model_validate(
        {
            "design": {
                "kind": "fractional-factorial",
                "generators": [
                    {
                        "factor": "trainer.epochs",
                        "word": ["optimizer.learning_rate", "trainer.batch_size"],
                    }
                ],
            },
            "grid": axes,
        }
    )
except ValidationError as refused:
    print(str(refused).splitlines()[-1])
```

```text
  design.generators: the defining relation has the word ['optimizer.learning_rate', 'trainer.batch_size', 'trainer.epochs'], shorter than the 4 letters resolution 4 needs [validator_error]
```

When the sweep is expanded, a generator that names an axis the grid does not
declare, or a named axis without exactly two values, is refused. The same
generators can be held back as a `resolution-iv` rung and applied to a full
grid only if the experiment has to shrink; see
[Contract an experiment](contract-an-experiment.md).

## Declare a multiple-comparison correction

An experiment document declares how its preregistered comparisons are
corrected in its `evaluation` block. The documentation experiment,
[`docs/examples/experiments/synthetic.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/experiments/synthetic.yaml),
declares:

```yaml
evaluation:
  alpha: 0.05
  correction: {kind: benjamini-hochberg}
  dependence: prds
  minimum_seeds: 3
  seed_budget:
    metric: recovery
    target: mean
    precision: 0.05
    level: 0.95
```

`alpha` is the level each family of comparisons is controlled at, and
`dependence` is the regime the tests within a family were run under:
`independent`, `prds` (positive regression dependence on the true nulls) or
`arbitrary`. The families are the ones the preregistration assigns its
hypotheses to, and ingestion applies the correction within each family. The
correction is an installed plugin in the `lofi.correction` entry-point group:

| `kind` | Controls | Valid under |
| --- | --- | --- |
| `bonferroni` | Family-wise error, single step. | every regime |
| `holm` | Family-wise error, step down. | every regime |
| `benjamini-hochberg` | False discovery rate, step up. | `independent`, `prds` |
| `benjamini-yekutieli` | False discovery rate under any dependence. | every regime |

A correction declared under a regime its guarantee does not cover is refused
when the experiment is read. The checked-in
[`docs/examples/running/contracted.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/running/contracted.yaml)
declares `holm` under `arbitrary` dependence. Selecting
`benjamini-hochberg` in place of `holm` is refused:

```python
from pathlib import Path

from lofi.run import ExperimentError, read_experiment

try:
    read_experiment(
        Path("docs/examples/running/contracted.yaml"),
        overrides=("evaluation.correction.kind=benjamini-hochberg",),
    )
except ExperimentError as refused:
    print(str(refused).split(" is refused: ")[-1])
```

```text
evaluation was refused: the benjamini-hochberg correction is not valid under arbitrary dependence
```

`minimum_seeds` is the fewest seeds a reported cell may rest on, and it bounds
a `seeds` contraction rung's floor from below. The
[statistics reference](../../reference/stats.md) documents each correction
procedure, and [Experiments and
commitments](../../explanation/experiments-and-commitments.md) explains what a
corrected family licenses.
