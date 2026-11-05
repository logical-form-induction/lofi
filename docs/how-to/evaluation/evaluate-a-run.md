# Evaluate a trained point

A run document's `evaluators` section declares the held-out evaluators a point
is scored by once it has trained. Each entry maps a role to a declaration whose
`kind` names a plugin in the `lofi.evaluator` entry-point group, with the
bootstrap interval its values carry and the conventions it scores under. This
guide declares an evaluator, chooses its interval and conventions, says which
split it scores, and reads the rows it writes.

## Declare an evaluator

The first tutorial's document,
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml),
declares one evaluator under the role `held-out`:

```yaml
evaluators:
  held-out:
    kind: bracketing
    interval: {replicates: 50, level: 0.95}
    punctuation: {source: none, forms: []}
    decoder: minimum-risk
    conventions:
      punctuation: keep
      unary_chains: keep
      trivial_spans: keep
      minimum_length: 1
      aggregation: corpus
      coverage: zero
      self_arcs: allow
      gold_derivation: canonical
```

The role is the name the evaluator's rows are reported under, and an
experiment's `rows` and `primary` name a row by that role and one of the
evaluator's metrics (see [Run and price a sweep](../experiments/sweeps.md)).
Two evaluators of one kind can be declared under two roles.

The remaining fields belong to the kind. `bracketing` reports unlabeled
bracketing F1 under a `minimum-risk` or `viterbi` tree decoder, and identifies
punctuation either nowhere (`source: none`) or by exact surface forms
(`source: surface` with a nonempty, distinct `forms` list).

Composing the document and reading the evaluator back:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
run = codec(document)

for role, evaluator in run.evaluators.items():
    print(role, evaluator.kind, evaluator.metrics())
    print(evaluator.interval.settings())
    print(evaluator.settings().conventions.aggregation)
```

```text
held-out bracketing ('unlabeled-bracketing-f1',)
IntervalSettings(replicates=50, level=0.95)
corpus
```

`metrics()` returns the names of the rows a declaration reports, in report
order, and is what an experiment's row bindings are checked against.

## Choose an evaluator kind

The packaged kinds, each the name of an entry in `lofi.evaluator`:

| Kind | What it scores |
| --- | --- |
| `bracketing` | Unlabeled bracketing F1 over the held-out split. |
| `held-out-likelihood` | Held-out string and normal-form derivational likelihood, and description length. |
| `grammaticality` | The grammaticality battery over a root-mass readout. |
| `category-side` | Category recovery, such as supertag accuracy, with primitive and constructor random-relabeling floors. |
| `logical-form` | Logical-form matching over decoded held-out forms, with its random-relabeling floor and fiber-relative recovery. |
| `inference` | Inference-label evaluation over held-out sentence pairs, with coverage, calibration and optional proof rows. |
| `expert-categories` | Agreement with expert ratings and a fixed error taxonomy over induced types. |
| `lexical-types` | Lexical-type F1 over the held-out split. |
| `decode` | Neural-term beams and optional supertag-factored A* parsing; see [Decode derivations](decode-derivations.md). |
| `equivalence` | Prover-decided equivalence of decoded and gold logical forms: equivalence, well-formed and timeout rates, mean proof time, and the prover calls and memo hits spent under a declared proof budget. |
| `lexical-diagnostics` | Supertag accuracy, ambiguity and confusion, clustering scores, collapse and decode refusal over decoded term posteriors. |
| `structural-diagnostics` | Model structure, derivation counts and three fixed bracketing baselines. |
| `reachability` | Step and derivation reachability of the held-out gold derivations under the run grammar's label and rule inventory, with the blocking histogram by label, rule and construction. |
| `constructor-diagnostics` | Beam, projection and reconstruction rows over constructor roles, and the predicted-minus-gold depth gap with its declared collapse warning. |
| `coherence-diagnostics` | Objective, entropy, collapse, usage and mass-agreement rows of the coherence objective. |
| `coherence-square` | Whether no, a sample of, or every decoded chart item is checked for coherence. |

`lofi/configs/evaluators/` holds a packaged fragment for most kinds, such as
`grammaticality/root-mass.yaml` and `held-out-likelihood.yaml`. The
[evaluation configuration reference](../../reference/configuration/evaluation.md)
lists every field of every kind. A plugin adds a kind by publishing a declaration class
and an entry in `lofi.evaluator`, as [Publish plugins](../extending/plugins.md)
describes. The declaration names the held-out data it scores through
`evaluation_data()` (gold derivations, the derivations of each inference pair
side, or labeled inference pairs) and its report rows through one `metrics`
method, which an evaluator of inference pairs reads the data's label inventory
from. The built evaluator reads its retained partition through `held_out`,
which asks the run's retained data for the shape it scores.

A packaged fragment is selected as a group whose slot is the role, so the
directory under `evaluators/` names the role the evaluator is added under.
Selecting the root-mass grammaticality fragment beside the declared
bracketing evaluator:

```python
run = codec(document, overrides=("evaluators/grammaticality=root-mass",))
print({role: evaluator.kind for role, evaluator in run.evaluators.items()})
```

```text
{'grammaticality': 'grammaticality', 'held-out': 'bracketing'}
```

In a document's `defaults` list the same selection is
`- evaluators.grammaticality: root-mass`. A kind no installed plugin registers
is refused, and the refusal lists the kinds that are registered.

## Set the interval

`interval.replicates` is the number of bootstrap resamples and must be at least
one; `interval.level` is the coverage the interval aims at and must lie
strictly between zero and one. Each value's interval resamples the units its
metric counts, such as sentences, and recomputes the metric on each resample,
drawn under the run seed. A value resting on fewer than two units has no
interval: a random-relabeling floor over a single held-out token, for instance,
reports its value with `n` of one and null interval columns. A level of one is
refused when the document is composed:

```python
try:
    codec(document, overrides=("evaluators.held-out.interval.level=1.0",))
except Exception as refused:
    print(type(refused).__name__, str(refused)[:200])
```

```text
ConfigValueError evaluators.held-out.interval.level was refused: an interval coverage is not a proportion: 1.0
```

## Declare the conventions

Every packaged evaluator requires all eight conventions, so no row is
scored under a convention its document did not state:

| Convention | Values | What it decides |
| --- | --- | --- |
| `punctuation` | `keep`, `strip` | Whether punctuation is scored or stripped first. |
| `unary_chains` | `keep`, `collapse` | Whether a gold unary chain stays a chain. |
| `trivial_spans` | `keep`, `drop` | Whether the whole-sentence span and single-word spans are scored. |
| `minimum_length` | a nonnegative integer | The shortest sentence still scored, after punctuation is removed. |
| `aggregation` | `sentence`, `corpus` | Whether a score averages sentences or pools units. |
| `coverage` | `ignore`, `zero` | The penalty for an item without an analysis. |
| `self_arcs` | `allow`, `forbid` | Whether a token may depend on itself. |
| `gold_derivation` | `canonical`, `oracle` | Which gold derivation a prediction is scored against. |

A value outside a convention's set is refused with the values it accepts:

```python
try:
    codec(document, overrides=("evaluators.held-out.conventions.aggregation=document",))
except Exception as refused:
    print(type(refused).__name__, str(refused)[:200])
```

```text
SettingParseError evaluators.held-out.conventions.aggregation expects one of ['sentence', 'corpus']; got 'document'
```

The conventions, extended by any setting the evaluator adds, are hashed into
each row's `convention_hash`, so rows scored under different conventions are
never pooled.

## Which split is scored

The trainer's evaluation dataset must declare named retained partitions. The
split an evaluator scores is chosen in this order:

1. the split the evaluators declare they score on, such as an alignment's
   `score_on`; evaluators that name two different splits are refused;
2. otherwise, the held-out split the dataset declares;
3. otherwise, the dataset's one retained split. A dataset retaining more than
   one split without either declaration is refused.

The split must be one the dataset retains. A trainer that declares no
evaluation dataset, such as distance pretraining, holds out no parse and is
refused.

## Read the rows

`lofi run` scores every declared evaluator after training and writes the rows
to `rows.json` in the point's directory, beside `metrics.json`. The file
records the freeze the rows were scored against, the seed derivation, and the
rows in emission order. A point trained outside an experiment records a null
freeze in the file and `unfrozen` in each row's `freeze_id`, so ingestion
against a freeze refuses its rows. Inside an experiment the freeze is the
experiment's. A document that declares no evaluator writes an empty `rows`
list.

Each row is one metric of one evaluator. Its columns are:

| Columns | What they hold |
| --- | --- |
| `freeze_id`, `config_hash`, `grammar_fingerprint`, `convention_hash` | The row's content identity: the freeze, the resolved document, the grammar and the conventions. |
| `suite`, `task`, `dataset`, `split`, `method`, `metric` | The suite `held-out`, the evaluator's role as the task, the dataset and split scored, the source system, and the metric name. |
| `value`, `n`, `resampling_unit`, `resampling_seed`, `seed` | The value, the number of units it was computed from, how and under which seed the interval resampled them, and the run seed. |
| `ci_low`, `ci_high`, `ci_level` | The bootstrap interval, or nulls for a value reported without one. |
| `budget_kind`, `budget_depth`, `budget_beam`, `budget_mass`, `residual_measured`, `residual_<source>` | The approximation the row was scored under. Every held-out row is scored under the grammar's enumeration budget, so `budget_kind` is `enumeration`, and its residual is unmeasured. |
| `normal_form`, `specification_hash`, `fold`, `rung`, `held_out_convention`, `required_control` | The normal form, the judgment-fit identity of a cross-fitted row, and the control method a metric requires beside it, if any. |
| `point_id` | The point whose directory and manifest the row came from. |

An evaluator that publishes artifacts writes them under
`evaluation-artifacts/<role>/` in the point's directory after its rows. An
experiment collects every point's rows into `rows.parquet`, as
[Run and price a sweep](../experiments/sweeps.md) describes.
