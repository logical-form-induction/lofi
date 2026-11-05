# Comparing grammars

This chapter asks the question the earlier chapters prepare for: does the
grammar that generated a corpus fit it better than a rival grammar does? It
reads an experiment document that crosses one source grammar with two
candidate learner grammars and two validity controls, prices the experiment
from timed training steps, trains its ten points as a pilot, and reduces their
rows to a verdict. The whole experiment runs in about a minute on a laptop CPU.

## The experiment document

Chapters 1 to 5 each trained one point from a run document. An experiment
document names a run document and declares a sweep over it, the root seed its
points derive their seeds from, the rows it reports and the reducers that turn
those rows into a result. The one this chapter uses is the reduced freeze of
the S2 experiment,
[`experiments/s2/reduced-freeze.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s2/reduced-freeze.yaml),
over the run document `experiments/s2/s2-run.yaml`.

The run document trains a tagged chart parser on a corpus drawn from the
source grammar `slashes-application-k2-depth1`: two primitives, the two slash
constructors, forward and backward application, and a generated lexicon of
eight forms. Unlike chapters 2 to 5, the learner's grammar is not taken from
the data. It is declared separately under `model.language`, so the corpus and
the grammar the learner parses it with can vary independently:

```yaml
defaults:
  - data.datasets.train.dataset: s2-source-run
  - model.language: slashes-application-k2-depth1
  - model.spaces.categories.combinators: application
```

Its one evaluator, `held-out`, scores each point's held-out string
log-likelihood per sentence and its description length.

The experiment varies three groups together in a `yoked` sweep, so the
values at one position form one point:

```yaml
sweep:
  yoked:
    - - group: data.datasets.train.dataset
        values: [s2-source-run, s2-source-run, slashes-application-k2-depth1-reversed, slashes-application-k2-depth1-scrambled]
      - group: model.language
        values: [slashes-application-k2-depth1, arrow-composition-k4-depth2, slashes-application-k2-depth1, slashes-application-k2-depth1]
      - group: model.spaces.categories.combinators
        values: [application, arrow-composition, application, application]
      # ... and the comparison.source, comparison.condition and comparison.direction paths
  grid:
    - path: data.datasets.train.dataset.size
      values: [8]
  seeds: [0]
  expected_points: 10
signature_twins: {count: 3, seed: 109, primary_condition: primary, observed_twin: observed}
```

Reading the design:

| Part | What it says |
|---|---|
| First two positions | The source corpus under the source grammar itself, and under `arrow-composition-k4-depth2`, a grammar with one undirected arrow constructor, four primitives and composition. These are the `primary` cells. |
| Last two positions | Validity controls: the source grammar fitted to a corpus with its ground truth reversed, and to one with its lexicon scrambled. A sound comparison should not find the source grammar winning on either. |
| `comparison` | Each point's role in the comparison: the source grammar's name, the condition and the direction of the candidate. The reducer reads these paths. |
| `signature_twins` | Each primary candidate is also fitted under three signature twins, drawn with the declared seed, and the reducer checks each grammar's observed fit against its twins' fits. |
| `size: 8` | Eight sentences per corpus, which keeps each point to a few seconds. |

Two primary cells with three twins each, plus their observed fits and the two
controls, give the ten declared points.

## Expanding the points

[`read_experiment`](../reference/run.md) reads and validates the document, and
`expand_experiment` resolves one configuration per point, with each point's
identifier derived from its resolved configuration and seed:

```python
from pathlib import Path

from lofi.run import expand_experiment, read_experiment

document = Path("experiments/s2/reduced-freeze.yaml")
experiment = read_experiment(document)
points = expand_experiment(experiment, Path.cwd())

for point in points:
    comparison = point.config["comparison"]
    candidate = point.config["model"]["language"]["grammar"]["name"]
    print(point.point_id, candidate, comparison["condition"], comparison["twin"])
```

```text
2b7b746612b430f2 slashes-application-k2-depth1 primary observed
b08b43d61ccffc43 slashes-application-k2-depth1 primary twin-001
7255147eab1954df slashes-application-k2-depth1 primary twin-002
ea66da189418979f slashes-application-k2-depth1 primary twin-003
bd1872ebc81d1e44 arrow-composition-k4-depth2 primary observed
083aa253a3515c45 arrow-composition-k4-depth2 primary twin-001
d9b3ebaad78901bd arrow-composition-k4-depth2 primary twin-002
79e28a4d74995306 arrow-composition-k4-depth2 primary twin-003
2a13d03b6cb96f1a slashes-application-k2-depth1 reversed-ground-truth observed
0436780ffebb7a2e slashes-application-k2-depth1 lexicon-scrambled observed
```

## Pricing the experiment

`lofi price` builds every point's training stack, times a few of its steps on
the named device, and writes a price record under the root it is given:

```bash
uv run lofi price \
  --experiment experiments/s2/reduced-freeze.yaml \
  --device cpu \
  --warmup 1 \
  --repeats 2 \
  --root .
```

```text
stack 958799c221c8a9b8 points=1 length=1 epochs steps=3 mean=0.042716 s/step peak unmeasured
stack 282fd06417fdedcd points=1 length=1 epochs steps=3 mean=0.042996 s/step peak unmeasured
stack d0d3b1f408e4f29f points=1 length=1 epochs steps=3 mean=0.049888 s/step peak unmeasured
stack 87d4d445dff798d9 points=1 length=1 epochs steps=3 mean=0.038995 s/step peak unmeasured
stack cf3a65b91d87c406 points=1 length=1 epochs steps=3 mean=0.029208 s/step peak unmeasured
stack 2641e26893bdf427 points=1 length=1 epochs steps=3 mean=0.048681 s/step peak unmeasured
stack 339f87c4f6415aa4 points=1 length=1 epochs steps=3 mean=0.045131 s/step peak unmeasured
stack 9b76e12af47b92e1 points=1 length=1 epochs steps=3 mean=0.049686 s/step peak unmeasured
stack af62d94529de6c94 points=1 length=1 epochs steps=3 mean=0.051521 s/step peak unmeasured
stack 305499f28c694176 points=1 length=1 epochs steps=3 mean=0.054776 s/step peak unmeasured
device: cpu (cpu, torch 2.14.0, host arm)
points: 10
cost model: experiment-total-mean-bootstrap
resamples: 10000 (seed 109)
p50 hours: 0.000378
p90 hours: 0.000405
declared: unpriced
written: sweeps/3c75d013643c5c3a/price.json
```

Each line is one distinct training stack, with the mean wall time of its timed
steps. The p50 and p90 hours bootstrap the experiment's total from those
steps. `declared: unpriced` reports the document's own `compute` field; the
measured price is a separate record, written under the sweep's identifier, so
it is never confused with a declared estimate. The timings vary from machine
to machine, and on a loaded machine from run to run.

## Running the points

On a cluster, `lofi launch` launches an experiment's points against a freeze:
a tagged commit that every point records as its source revision. Before any
freeze exists, the same command runs the experiment as a
pilot. `--pilot` takes the place of `--freeze`, and the points train one after
another in this process on the serial launcher:

```bash
uv run lofi launch \
  --experiment experiments/s2/reduced-freeze.yaml \
  --pilot \
  --out pilot/launch.json
```

The command echoes the launch record it writes to `pilot/launch.json`: ten
points offered, ten launched and none served from the store. Each point's
seed is derived from the experiment's root seed, 109, and its identifier, and
each point writes the same records as in chapter 3, under
`pilot/runs/<point-id>/`. Launching the pilot again serves every point from
the pilot store instead of training it again.

A pilot reads no freeze record and requires no tag, so every row it reports
is marked `unfrozen` and its results directory,
`pilot/results/reduced-freeze/`, lies apart from every frozen one. Ingestion
and a freeze refuse these results with `PilotResultsError`; they
are for looking at a design before it is frozen.

## Reading the rows

Every point reports the rows the experiment declares, and the document
declares one table over the primary row, the held-out string log-likelihood
per sentence:

```yaml
reports:
  - name: held-out-fit
    rows: [held-out/string-log-likelihood-per-sentence]
    columns: [freeze_id, config_hash, split, n, value]
    labels: [Freeze, Configuration, Split, Sentences, Log-likelihood per sentence]
    precision: 3
```

`lofi results --pilot` renders the declared tables from the pilot's rows and
prints where each one was written:

```bash
uv run lofi results --experiment experiments/s2/reduced-freeze.yaml --pilot
```

```text
pilot/results/reduced-freeze/held-out-fit.md
```

The table, with each configuration digest shortened here to its first
sixteen characters:

| Freeze | Configuration | Split | Sentences | Log-likelihood per sentence |
| --- | --- | --- | --- | --- |
| unfrozen | 958799c221c8a9b8 | validation | 2 | -0.647 |
| unfrozen | 282fd06417fdedcd | validation | 2 | -0.612 |
| unfrozen | d0d3b1f408e4f29f | validation | 2 | -0.771 |
| unfrozen | 87d4d445dff798d9 | validation | 2 | -0.646 |
| unfrozen | cf3a65b91d87c406 | validation | 2 | -0.216 |
| unfrozen | 2641e26893bdf427 | validation | 2 | -0.212 |
| unfrozen | 339f87c4f6415aa4 | validation | 2 | -0.230 |
| unfrozen | 9b76e12af47b92e1 | validation | 2 | -0.303 |
| unfrozen | af62d94529de6c94 | validation | 2 | -0.749 |
| unfrozen | 305499f28c694176 | validation | 2 | -0.533 |

The rows follow the points in expansion order: the source grammar's observed
fit and its three twins, the arrow grammar's observed fit and its three
twins, then the reversed and the scrambled controls. Each configuration
digest is the stack identifier `lofi price` printed for the same point.
Higher is better. The reversed control fits worse than the source grammar's observed fit,
but the scrambled control fits better. The source grammar beats one of its
three twins, and the arrow grammar's fits are higher than the source grammar's. Each value counts two held-out sentences,
so none of these differences is yet evidence of anything.

## Reducing to a verdict

The `source-comparison` reducer the document declares reads these rows
together with each point's manifest and measured capacity, all of which the
pilot's results directory holds, and tests the source grammar against each
candidate, each twin and each control:

```python
from pathlib import Path

import pyarrow.parquet as pq

from lofi.eval import ReductionContext, reduce_experiment_rows
from lofi.run import comparison_points, expand_experiment, read_experiment
from lofi.run.results import pilot_directory

document = Path("experiments/s2/reduced-freeze.yaml")
experiment = read_experiment(document)
points = expand_experiment(experiment, Path.cwd())
results = pilot_directory(Path.cwd(), experiment)

rows = pq.read_table(results / "rows.parquet").to_pylist()
measured = pq.read_table(results / "measured.parquet").to_pylist()
manifests = [
    experiment.config.manifest_point(
        (results / "manifests" / f"{point.point_id}.json").read_bytes()
    )
    for point in points
]

reduction = reduce_experiment_rows(
    rows,
    experiment.reducers,
    ReductionContext(
        document.parent,
        seed=experiment.seed_policy.root,
        points=comparison_points(manifests, measured),
        preregistration=experiment.preregistration,
        alpha=experiment.evaluation.alpha,
        correction=experiment.evaluation.correction,
        dependence=experiment.evaluation.dependence,
    ),
)
for row in reduction.rows:
    value = row.get("value")
    shown = value if not isinstance(value, float) else round(value, 3)
    print(row["reducer"], row.get("candidate", row.get("winner")), shown)
```

```text
source-comparison-delta arrow-composition-k4-depth2 -0.432
source-comparison-test arrow-composition-k4-depth2 -0.432
source-comparison-test reversed-ground-truth 0.102
source-comparison-test lexicon-scrambled -0.114
source-comparison-twin arrow-composition-k4-depth2 0.667
source-comparison-twin slashes-application-k2-depth1 0.333
source-comparison-verdict arrow-composition-k4-depth2 None
source-by-winner arrow-composition-k4-depth2 1
source-win-threshold None None
```

The verdict row records which conditions the source grammar met:

```python
[verdict] = (row for row in reduction.rows if row["reducer"] == "source-comparison-verdict")
for key in ("winner", "source_rank", "beats_candidates", "beats_twins", "validity_controls_pass", "positive"):
    print(key, verdict[key])
```

```text
winner arrow-composition-k4-depth2
source_rank 2
beats_candidates False
beats_twins False
validity_controls_pass False
positive False
```

Reading the reduction:

| Row | What it shows |
|---|---|
| `source-comparison-delta` | The source grammar's held-out likelihood minus the candidate's: negative, so the candidate fits better at this size. |
| `source-comparison-test` | One test per rival, the candidate and each control, corrected across the family at the document's `alpha`. With one seed and two held-out sentences, none rejects. |
| `source-comparison-twin` | The twin check for each grammar. The source grammar's observed fit has value 0.333, and the candidate's has value 0.667; neither passes. |
| `source-comparison-verdict` | The source grammar ranks second, beats neither its twins nor the candidate, and the controls do not pass, so the verdict is not positive. |
| `source-by-winner` | One count in the cell for this source and this winner, the row a full experiment sums into a source-by-winner matrix. |

A positive verdict needs every condition at once: the source grammar beats each
candidate, beats its twins, and the controls behave. At eight sentences and one
seed the reduced freeze checks only that the pipeline produces each row; the
S2 production document, `experiments/s2/source-complexity.yaml`, crosses eight
source grammars with eight candidates at three corpus sizes and five seeds.

Once the experiment is frozen, `lofi launch --freeze` runs the
same points against the tag, `lofi ingest` accepts their results against the
freeze and applies the same reducer, and `lofi results` without `--pilot`
renders the same tables from the frozen results directory.

## Try this

- Add a seed to the sweep in a copy of the document (`seeds: [0, 1]` and
  `expected_points: 20`) and compare the tests' corrected p-values.
- Raise `data.datasets.train.dataset.size` in the grid to 32, price the
  experiment again, and compare the p50 hours with the first price.
- Expand `experiments/s2/source-complexity.yaml` with `expand_experiment` and
  count its points before pricing it.

## Next

[Chapter 7](07-charts-and-semirings.md) opens the chart every parser in these
chapters fills, and runs it under two semirings.
