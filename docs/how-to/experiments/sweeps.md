# Run and price a sweep

An experiment document's `sweep` block expands one run or judgment fit
document into deterministic `SweepPoint` values, each holding a fully resolved
configuration, its leaf provenance, a seed, and the stable identifier derived
from the configuration and seed. This guide declares the points, prices the
experiment, launches it from its freeze, polls a scheduled launch, and ingests
the results. A single development point is trained with `lofi run` instead, as
in the [first tutorial](../../tutorials/01-first-run.md).

## Declare the points

The checked-in
[`docs/examples/experiments/synthetic.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/experiments/synthetic.yaml)
crosses two learning rates with three seeds:

```yaml
sweep:
  grid:
    - path: optimizer.learning_rate
      values: [0.001, 0.0005]
  seeds: [0, 1, 2]
  expected_points: 6
seed_policy:
  root: 7
```

The fields of a sweep:

| Field | What it declares |
| --- | --- |
| `grid` | Axes crossed with one another. |
| `yoked` | Groups of at least two axes that advance together, one tuple of values per position. |
| `exclude` | Clauses, each naming axes by their key; an assignment that agrees with every pair of any clause is dropped. |
| `seeds` | Distinct seeds crossed with every surviving assignment. |
| `expected_points` | The point count the expansion must produce. An experiment document requires it. |
| `design` | Which assignments of the axes are kept; the full grid when omitted. See [Choose a sweep design](choose-a-sweep-design.md). |

An axis declares either a `path` or a `group`, each dotted. A `path` axis
assigns one leaf, so values on `optimizer.learning_rate` are numbers. A
`group` axis selects the fragment that fills a configuration slot, so values on
`model.encoders.terms` are fragment names. A group may sit at the root of the
document:

```yaml
grid:
  - group: device_policy
    values: [cpu, cuda-mixed]
  - path: trainer.early_stopping.patience
    values: [5, 10]
```

A leaf whose value is a tagged union, such as `loss`, takes whole mappings as
its values, each validated as a fragment would be:

```yaml
grid:
  - path: loss
    values:
      - terms:
          - objective: {kind: grammaticality, readout: sentences, noise: 0.0}
            weight: 1.0
      - terms:
          - objective: {kind: grammaticality, readout: sentences, noise: 0.1}
            weight: 1.0
```

A sweep is refused when it is validated if a grid axis repeats a value, a
yoked group repeats a tuple or mixes value counts, two axes share a key, an
exclusion clause names an undeclared axis, or one axis assigns a slot under
another's (a leaf under a selected group composes; a group under a group only
when the two are yoked).

The expansion also refuses an axis that shares a setting with a source that
would overrule it or that it would silently discard. If the run document, a
profile, an overlay or an override sets a leaf under a group an axis selects,
the expansion is refused naming the leaf and both sources, since that source
outranks the fragment the axis selected. A leaf axis varies the value the run
document gives its path, so that value is not a conflict; but a profile, an
overlay or an override at the path would be discarded unseen, and a
run document leaf under a mapping value would survive into it, so both are
refused in the same way. A point count other than `expected_points` is
refused with `UnexpectedPointCountError`.

Because a `generated` lexicon names no category, an axis may assign a CCG
signature's primitive list. Over
[`docs/examples/synthetic-generated.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-generated.yaml),
this sweep varies two, four, eight and sixteen primitives:

```yaml
sweep:
  grid:
    - path: data.datasets.train.dataset.grammar.signature.primitives
      values:
        - [S, P1]
        - [S, P1, P2, P3]
        - [S, P1, P2, P3, P4, P5, P6, P7]
        - [S, P1, P2, P3, P4, P5, P6, P7, P8, P9, P10, P11, P12, P13, P14, P15]
  seeds: [0]
  expected_points: 4
```

The inventory size stays at the declared `categories` for every point. To grow
it with the primitives, put
`data.datasets.train.dataset.grammar.lexicon.categories` in a `yoked` group
with the primitive axis.

## Seed the points

The experiment's `seed_policy.root` is the one place its root seed is
declared. A point's run seed is derived from that root and the point's
identifier, so two roots train one point under
different seeds and one root reproduces it exactly. Each run manifest records
the root, the point and the derived seed under `seed_derivation`.

`lofi run` and `lofi resume` train a single development point outside any
experiment and name its root with `--experiment-seed`. A run directory is
served from cache only under the root it was trained under; run again under
another root, the point is refused rather than served. A development point
receives the infrastructure attempts the failure policy declares by default
unless `--infrastructure-attempts` grants another number, and an attempt that
fails for want of infrastructure is retried in the same process from its
latest checkpoint.

## Declare what the experiment commits to

An experiment document binds its sweep to its preregistration and to the rows
its configuration document reports, and each binding is checked when the
document is read:

- `factors` names, for each factor of the preregistration's `grid`, the sweep
  axis that varies it and the axis value that realizes each preregistered
  level. A factor left out is spanned by the axis whose key is its name, each
  level being the value it takes. The document is refused until every
  preregistered factor is spanned and every level maps to a value its axis
  takes.
- `rows` names each report row as an evaluator role of the configuration
  document and a metric that evaluator reports, and `primary` names the row
  the claim is read from. A judgment fit reports its `divergences`, `max-rhat`
  and `min-ess` diagnostics under the role `fit`.
  [Reduce and report results](reduce-and-report-results.md) renders and
  reduces these rows.
- `stop_rules` and `decision_point.signal` are predicates. A `points`
  predicate holds once at least `at_least` points are in the listed statuses;
  a `metric` predicate holds once the `mean`, `min` or `max` of one row's
  values lies strictly `below` or `above` its `threshold`, read from at least
  `minimum_rows` values.
- `contraction` declares the rungs the experiment may shrink through after its
  freeze; [Contract an experiment](contract-an-experiment.md) applies them.
- `evaluation` declares the level, correction and dependence regime the
  preregistered comparisons are read under; see [Choose a sweep
  design](choose-a-sweep-design.md#declare-a-multiple-comparison-correction).
- `launch` names the launcher (`serial`, `array` with an optional
  `chunk_size` and `throttle`, or `remote`), and `failure` declares how a
  numerical failure and exhausted memory are handled (see
  [Failure outcomes](#failure-outcomes)).

The documentation experiment declares:

```yaml
rows:
  - {evaluator: held-out, metric: unlabeled-bracketing-f1}
primary: {evaluator: held-out, metric: unlabeled-bracketing-f1}
stop_rules:
  - name: numerical failure
    when: {kind: points, statuses: [numerical-failure], at_least: 1}
decision_point:
  date: "2026-12-01"
  signal:
    kind: metric
    row: {evaluator: held-out, metric: unlabeled-bracketing-f1}
    statistic: max
    direction: above
    threshold: 0.0
  fallback: The synthetic parser stack trains at one learning rate.
```

## Price the experiment

An experiment declares `compute: {kind: unpriced}` until it has been priced,
and then
`compute: {kind: estimate, p50_hours: ..., p90_hours: ..., cost_model: ...}`.
The field is required: a document that omits it is refused rather than read as
unpriced. Price the documentation experiment from the repository root:

```bash
LOFI_CHECK=0 uv run lofi price \
  --experiment docs/examples/experiments/synthetic.yaml \
  --device cpu \
  --warmup 1 \
  --repeats 5 \
  --root .
```

`--root` names the repository root. The experiment path is read relative to
it, the sweep is expanded under its configuration groups, and the price is
written beneath it as `sweeps/<sweep-id>/price.json`, which is where a freeze
of that root looks for the record. `--repeats` must be at least two.
`LOFI_CHECK=0` keeps the [debug checks](../extending/checks.md) out of the
timed step.

Points whose run documents agree differ only in their seed and share one
stack, so the command builds each distinct stack once, under the run seed its
first point derives from the experiment root, and times real training steps of
it on the named device. The warmup and timed steps run at positions drawn at
random, with replacement and seeded by that run seed, from across the stack's
epoch or horizon, and the record keeps the position each timed step read. A
point trains for the length its run document declares: a parser trainer's
epochs times the batches one epoch holds, or a distance trainer's steps. A
parser trainer that stops early does so at the latest after its declared
epochs, so it is priced at those epochs. A trainer that declares neither
epochs nor steps is refused before anything is built. Only an experiment over
run documents is priced; a judgment fit experiment is refused.

For each stack the command reports the mean seconds per step and the peak
allocated bytes. The experiment's total is the sum over its stacks of points
times declared steps times the mean step. The P50 and P90 accelerator hours are
quantiles of that total under a bootstrap in which each replicate resamples
every stack's timed seconds with replacement. The bootstrap draws from one
generator seeded by the experiment's root seed, and the `pricing` block's
`resamples` sets how many replicates it draws, 10000 when omitted:

```yaml
pricing:
  kind: stack-bootstrap
  resamples: 20000
```

`price.json` holds every raw timing, the resample count, the seed, the cost
model and a hardware descriptor (device name, driver, CUDA runtime, torch
version and host processor), so the figures can be recomputed from the record.

The `pricing` block names the cost model, an installed plugin in the
`lofi.pricing` entry point group, and every cost model declares `resamples`,
the bootstrap replicates its quantiles are read off, `10000` unless declared.
Without a `pricing` block every stack is priced off its own timed steps, the
`stack-bootstrap` model. A document declaring

```yaml
pricing:
  kind: length-batch
  lengths: [10, 20, 30, 40]
  batches: [8, 16, 32]
```

is priced by interpolation instead, and declares its own `resamples` the same
way: the training step of a reference run document is timed at every sentence
length and batch size of the grid, each stack records the mean sentence length
and batch size of its own epoch, and a point costs its declared steps at the
step seconds interpolated bilinearly at that shape. The lengths and batches
shown are the defaults when the block names the kind alone.

The reference is an ordinary run document, `reference`, by default the
packaged `lofi.resource:run/pricing-reference.yaml`, which declares the stack's
width, learning rate and grammar. Each cell composes it under `cell_overrides`,
`key=value` expressions in which `{length}` and `{batch}` stand for the cell's
sentence length and batch size; the defaults set the synthetic corpus's length
bounds and the trainer's batch size. Every cell is planned and prepared as a
run is, under the root seed `seed` (0 unless declared), on the device being
priced, and steps over the first batch of its epoch, which must hold exactly
the cell's length and size or the price is refused. A distance stack reads no sentences and is refused under this
cost model.

Each stack's peak allocation is read against the `activation_memory_bytes` its
own device policy declares. The packaged `cpu` and `mps` policies allow 8 GB
and `cuda-mixed` allows 40 GB. `--device` must be of the kind every point's
device policy declares, so pricing a stack declared for `mps` or `cuda` on
`cpu` is refused before that stack runs a step; a device index may differ, so
`cuda:1` prices a `cuda` declaration. A stack whose peak exceeds its budget is
priced, recorded and named on standard error, and an array launch leaves its
points out.

Declare the printed figures in the experiment's `compute` block. A freeze
admits only an experiment that declares an estimate, and for a run experiment
its `price.json` must be tracked, must name the experiment's sweep, the
experiment and its freeze, and must recompute to the declared figures at the
six decimals the command prints. A judgment fit experiment's estimate has no
record to recompute it from, so the freeze records it with the source
`supplied` and refuses one that names the measured cost model.

## Freeze the experiment

`lofi freeze` tags a clean `HEAD` as `freeze/<n>` and records what it froze in
`freezes/<n>/FREEZE.json` and one row of `freezes/ACTIVE.tsv`. It names every
experiment run against the freeze and a file of readiness notes, one line for
each of `FG-1` to `FG-9`. The following is a template command:

```text
uv run lofi freeze --number 1 \
  --experiment experiments/headline.yaml \
  --experiment experiments/analysis.yaml \
  --readiness ~/freeze-1-readiness.txt
```

Every experiment must lie in the repository, run against the freeze being
minted, be tracked with its configuration document and its preregistration,
and declare a compute estimate its price record supports, as
[Price the experiment](#price-the-experiment) describes. The record holds,
for each point, the identities its configuration kind reports, so a run
point's grammar, codec, corpus and component fingerprints equal what its
manifest records when it trains. A freeze number is minted once; the file and
row are written before the tag, and removed if the tag cannot be created, and
are left for the PI to commit.

Before freezing, price the seed count from a pilot's rows. `lofi seed-budget`
reads the metric, target, precision and level the experiment's evaluation
section budgets, reads their seed-to-seed spread off the pilot's results
directory, and writes the required count beside the experiment document as
`<experiment>.seed-budget.json`, next to the seed floor it declares:

```text
uv run lofi seed-budget --experiment experiments/headline.yaml \
  --pilot pilot/results/headline
```

## Run a pilot before the freeze

A pilot runs an experiment before its freeze exists, to look at the design and
to price its seed count. `--pilot` takes the place of `--freeze`:

```text
uv run lofi launch \
  --experiment experiments/headline.yaml --pilot --out pilot/launch.json
```

A pilot reads no freeze record and requires no tag. Every point trains in this
process on the serial launcher, whatever launcher the experiment declares, and
each manifest records the commit the checkout is at. A pilot submits nothing
to a site, so it takes no `--with` or `--dry-run`, and a `--set` override of
`launch` has no effect on it. Its store
lies under `pilot/` in the checkout unless `--root` names another, so its run
directories are `pilot/runs/<point_id>/`, and its results directory is
`pilot/results/<experiment>/`, apart from every frozen one. Its points are
bound and its rows reported as `unfrozen`, and its `STATUS.json` records the
freeze as `unfrozen`. The experiment's stop rules halt a pilot as they halt a
serial launch, and launching the pilot again serves every completed point from
the pilot store.

`lofi results --experiment experiments/headline.yaml --pilot` renders the
experiment's declared tables from the pilot's rows. Pilot results are never frozen results:
`lofi ingest` refuses a results directory whose status records the freeze as
`unfrozen` with `PilotResultsError`, wherever it lies, and
`lofi freeze` refuses a `HEAD` that tracks any file under
`pilot/` with the same error, so a pilot's results never ride inside a freeze's
tag. `lofi publish` refuses a point bound as `unfrozen` with the same error,
so a pilot's predictions are never published. The repository ignores `pilot/`.

## Launch from the freeze

An experiment's numbers are reported against a freeze, so it is executed only
by the code its freeze tags. After `lofi freeze` has minted
`freeze/<n>` and written `freezes/<n>/FREEZE.json`, check the tag out, install
the package from that checkout, and launch from the repository root. The
following is a template command:

```text
git checkout freeze/1
uv run lofi launch \
  --experiment experiments/headline.yaml \
  --freeze freezes/1/FREEZE.json --out launch.json
```

The experiment's `launch` block names the launcher, and `--set` overrides it
for one invocation, as `--set launch.kind=array --set launch.chunk_size=4`
does. Every point executes through the executor its experiment's configuration
kind builds, and a kind whose points execute only in the launching process,
such as a judgment fit, launches only on a launcher that submits nothing to a
site.
[Run on a cluster](run-on-a-cluster.md) covers the array and remote launchers
and the site profile they read.

Before anything is expanded, executed or written, the launcher checks the
checkout the `lofi` package is imported from and refuses the launch unless:

- that checkout has the tag the experiment's `freeze_id` names;
- its `HEAD` is the commit that tag names;
- no tracked file in it differs from that commit;
- the experiment document lies inside it;
- the freeze record names the experiment's freeze, records the commit the tag
  names, and lists the experiment with the sweep identifier its expansion
  reproduces now.

Untracked files, such as the run store, the freeze record and the results, are
not compared. Every manifest records the tagged commit as its source revision.
The check does not cover installed dependencies beyond the package itself;
`uv.lock` pins those, and the freeze records its digest.

### Points, evaluators and the cache

A run point is trained, and the held-out evaluators its run document declares
under `evaluators` are scored on its held-out split, each row reported against
the experiment's freeze. Outside an experiment, `lofi run` and `lofi resume`
score the same evaluators and write `unfrozen` in each row's `freeze_id`, so
development rows are visible in the run directory's `rows.json` and ingestion
against any freeze refuses them.

A judgment fit point is fitted by NUTS under the declared sampling plan, its
seed derived from the point's run seed, and records one row per posterior
diagnostic. A fit that declares `calibration` also simulates that many
replicates from the response model's priors over the dataset's design, fits
each by NUTS and by Quivers' structured and mean-field guides, and records one
row per record of the calibration report. A NUTS cutpoint recovery below the
preregistered floor is a numerical failure, and a calibrated fit whose
compiled response model does not vary its thresholds over participants is
refused before anything is fitted.

A point's directory is `<root>/<freeze_id>/runs/<point_id>/`, for instance
`<root>/freeze/1/runs/<point_id>/`, so two freezes never share a directory,
even for a point both offer. As the launcher submits a point, it writes
`point.json`, binding the directory to the freeze and the seed derivation. A
point is completed only when its directory is bound to this freeze under its
seed derivation, its manifest is closed as completed, and its rows were
produced under this freeze. Launching again serves every completed point from
its directory, and a directory bound under another experiment root is refused
rather than reused.

### Failure outcomes

Every outcome is recorded with the point's seed derivation in the point's
`failure.json`, and each attempt is recorded there before it starts, so an
attempt whose process is killed outright still counts.

- A completed point records the rows it emitted.
- A numerical failure, such as a nonfinite loss or a posterior sampler that
  reports a nonfinite position, draw or diagnostic, follows
  `failure.numerical`. Under `final` it is recorded at once with its
  diagnostic and the point is never executed again. Under `reseed` the point
  is attempted again under the next deterministic attempt seed, within the
  attempts described below.
- A configuration that cannot be prepared for training is recorded as
  `preparation-failed` and is final.
- An infrastructure failure is one the machine signals explicitly: an
  exhausted allocation, an interrupted call, a lost connection or broken pipe,
  a stall, or an operating system error whose errno reports a failing device
  (`EIO`), a stale shared filesystem handle (`ESTALE`), a timeout
  (`ETIMEDOUT`), a lost network (`ENETDOWN`, `ENETUNREACH`) or a reset
  connection (`ECONNRESET`). A run point resumes from its latest checkpoint;
  a judgment fit point is refitted at the same seed. Exhausted memory follows
  `failure.out_of_memory`: `retry` attempts again unchanged, and
  `halve-batch` halves the training batch first and records the failure once
  the batch cannot be halved.

Every launch gives each point the `failure.infrastructure_attempts` the
experiment declares, three unless declared, counted across every launch.
Once the attempts are spent the failure is recorded as `infra-failed` and is
final.

A process that receives `SIGTERM` while executing a point, as a scheduler
preempting a task sends it, records the interrupted attempt and stops; the
point resumes on a later launch within the attempts it has left. Any other
error, such as a missing corpus or a defect in the code, propagates and stops
the launch, and the attempt it ended is not counted.

### Results layout

The launcher writes the experiment's results directory beneath the
repository root:

```text
results/<n>/<experiment>/rows.parquet
results/<n>/<experiment>/measured.parquet
results/<n>/<experiment>/manifests/<point>.json
results/<n>/<experiment>/STATUS.json
```

`<n>` is the number of the experiment's `freeze/<n>`, so the `headline`
experiment of `freeze/1` writes `results/1/headline/`.

- `rows.parquet` holds every completed point's rows, each with its report
  identity, the budget and residual of the approximation it was scored under,
  and the `point_id` that leads back to its manifest. A held-out row states
  `budget_kind` as `enumeration` with its depth, beam and mass bounds. A
  judgment fit row states `budget_kind` as `none` with null bounds and
  residuals, and ingestion refuses a judgment fit row that claims an
  enumeration budget.
- `measured.parquet` holds one row per submitted point whose run directory
  records a `metrics.json`: its `point_id`, `config_hash` and status, and
  every non-sequence value of the record under its path, such as
  `point/parameter_count` or `corpus/noise/achieved_rate`.
- `manifests/` holds the manifest of every submitted point that has started,
  failed points included.
- `STATUS.json` records the sweep identifier, the number of points offered,
  the count and completed fraction overall and per cell (one configuration
  across its seeds), one entry per point, the stop rules that fire, and the
  decision point's state: `pending` before its date, then `signal-held` or
  `fallback`.

A point's status is `completed`; `running` for a manifest still open;
`killed` for a run that stopped without an outcome and awaits resubmission;
`awaiting-responses` for an active learning run whose batch awaits its
responses, which a later launch resumes; `numerical-failure`, `preparation-failed` or `infra-failed`, which are final
and never offered again; `submitted` for a point bound to this freeze that has
not started; or `not-submitted`. The launch record counts points as this store
records them, so its counts agree with `STATUS.json`.

## Poll a scheduled experiment

An array launch finishes its points after the launcher returns. Poll the store
its tasks write to, naming the same root the tasks were given. The following
is a template command:

```text
uv run lofi status \
  --experiment experiments/headline.yaml --root /absolute/path/to/scratch \
  --interval 300 --polls 288
```

A poll executes nothing. Each poll rewrites the results directory whole from
what the store holds and prints its status record on one line, so the results
converge as the array drains. The command polls up to `--polls` times, waiting
`--interval` seconds between polls, and stops early once no point is submitted
or running. It exits 1 as soon as a point's failure is final, 3 as soon as a
declared stop rule fires, 2 when its last poll still finds points submitted or
running, and 0 once none is. When a stop rule fires, the poll also cancels the
work the experiment's declared launcher left with the scheduler, as
[Run on a cluster](run-on-a-cluster.md) describes.

## Ingest the results

Once a results directory is complete, accept it against the freeze the
experiment ran under:

```bash
uv run lofi ingest --experiment experiments/headline.yaml \
  --freeze freezes/1/FREEZE.json --root .
```

Ingestion refuses a pilot's results, whose status records the freeze as
`unfrozen`, with `PilotResultsError`, a row scored outside a freeze or under
another freeze, a
row whose configuration digest is not that of a point in the frozen sweep, and
a manifest whose source revision is not the commit `FREEZE.json` records. An
amended sweep is accepted when its recorded amendments start from the sweep
the freeze recorded. The declared reducers run over the accepted rows, and the
declared correction is applied within each preregistered family. The rows are
written to `ingested.parquet` and the audit, which records the freeze, the
residual summary, the preregistration audit, each family's corrected
decisions, the completed fraction and, once the `fallback` rung has been
applied, the fallback claim, to `AUDIT.json`, both beside the results. A directory that does not conform exits 1 and writes nothing.

[Experiments and commitments](../../explanation/experiments-and-commitments.md)
explains what a freeze, a preregistration and a decision point commit an
experiment to.
