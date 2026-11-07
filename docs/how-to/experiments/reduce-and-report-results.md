# Reduce and report results

An experiment document declares two things that act on its rows after its
points have run: `reducers`, which derive further rows and comparisons from the
completed rows when the results are ingested, and `reports`, which
[`lofi results`](../../reference/cli.md#results) renders as tables through an installed report renderer. This guide names the rows
an experiment reports, declares a report layout, renders it, and declares a
reducer.
## Name the rows

Follow one measurement through the workflow: a point's evaluator emits a
held-out score; ingestion stores that score with the point and its conventions;
a reducer can compute a comparison across points; a renderer displays the
accepted values. Changing a table's precision changes its printed digits.
Changing a reducer can change the statistic being estimated. The
[report API](../../reference/run.md#report-columns-and-renderers) separates
these operations.

Every row a report or reducer reads is first declared in the experiment's
`rows`, as an evaluator role of the configuration document and a metric that
evaluator reports:

```yaml
rows:
  - {evaluator: held-out, metric: unlabeled-bracketing-f1}
primary: {evaluator: held-out, metric: unlabeled-bracketing-f1}
```

A row's label is `<evaluator>/<metric>`, here
`held-out/unlabeled-bracketing-f1`. Each stored row carries the evaluator role
as its `task` column and the metric as its `metric` column. A row that names
an evaluator role the configuration document does not declare, or a metric
that evaluator does not report, is refused when the experiment is expanded.

## Declare a report

A `reports` entry lays out one table. The checked-in
[`docs/examples/running/contracted.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/running/contracted.yaml)
declares one:

```yaml
reports:
  - name: bracketing
    rows: [held-out/unlabeled-bracketing-f1]
    columns: [config_hash, seed, value, ci_low, ci_high, nan_rate]
    labels: [Configuration, Seed, F1, Low, High, Missing]
    precision: 3
```

Reading the report line by line:

| Line | What it says |
| --- | --- |
| `name` | The output file's stem. It must be a bare filename, so a report cannot write outside the results directory. |
| `rows` | The labels of declared rows the table includes, in table order. |
| `columns` | The stored columns the table shows, in order. |
| `labels` | One heading per column. |
| `precision` | Decimal places for floating values; `3` when omitted. |
| `format` | The installed renderer that writes the table: `markdown` (the default) or `latex`, or any other entry of the `lofi.report_format` group. |

A column is one of the columns of the stored report rows, which
[`lofi.report.REPORT_FIELDS`](../../reference/run.md#lofi.report.REPORT_FIELDS) declares, or one of the derived columns
[`lofi.report.DERIVED_COLUMNS`](../../reference/run.md#lofi.report.DERIVED_COLUMNS) lists. The stored columns are the identity columns (`freeze_id`,
`config_hash`, `grammar_fingerprint`, `convention_hash`, `suite`, `task`,
`dataset`, `split`, `method`, `metric`, `seed` and the convention fields), the
value and its interval (`value`, `n`, `ci_low`, `ci_high`, `ci_level` and the
resampling fields), and the approximation fields (`budget_kind` and its
bounds, `residual_measured` and one `residual_<source>` per residual source).
`convention_hash` is null on a row that no parsing convention affects, such as
a [judgment model](../../explanation/judgment-models.md#from-responses-to-a-measurement-model) score.
`nan_rate` is the derived column: for each row it gives the fraction of rows sharing its
configuration, grammar, method and metric whose value is missing or NaN.

The layout is checked when the experiment is read. A report that reads a row
label the experiment does not declare, names an unknown column or a format no
installed renderer provides, gives a different number of labels than columns,
or repeats a report name is refused.

## Render the tables

[`lofi results`](../../reference/cli.md#results) renders every declared report from the experiment's
`rows.parquet`. Run this after the experiment's points have completed; the
checked-in declaration supplies the layout, while the run supplies the rows:

```bash
uv run lofi results --experiment docs/examples/running/contracted.yaml --root .
```

The command reads `results/<n>/<experiment>/rows.parquet` under `--root`,
where `<n>` is the number of the experiment's `freeze/<n>`, and writes each
report beside it as `<name>` followed by its renderer's suffix (`.md` for
`markdown`, `.tex` for `latex`), printing each path it writes.
For every row label in a layout it selects the stored rows whose `task` and
`metric` match, and renders them in that order. Rendering is deterministic, so
running it again over the same rows writes the same bytes. The rows are those
the launcher writes as points complete (see the results layout under
[Run and price a sweep](sweeps.md#results-layout)); rendering does not wait
for ingestion.

An experiment that declares no reports renders nothing and exits 0. A missing
`rows.parquet` exits 1 and names the file. Against the `contract-demo` copy
from [Contract an experiment](contract-an-experiment.md#apply-a-rung), where no
point has run:

```bash
uv run lofi results --experiment docs/examples/running/contracted.yaml --root contract-demo
```

```text
lofi results: experiment report rows cannot be read from contract-demo/results/1/contracted/rows.parquet: contract-demo/results/1/contracted/rows.parquet
```

A report whose rows select nothing from the stored rows, or a column absent
from the stored rows, is refused in the same way.

## Install a report format

A report format is an object with a `suffix` property, the output filename's
suffix with its dot, and a `render(labels, rows, /, *, precision)` method that
returns the table's bytes; [`lofi.report.ReportRenderer`](../../reference/run.md#lofi.report.ReportRenderer) is the protocol it
satisfies. [`lofi.report.format_cell`](../../reference/run.md#lofi.report.format_cell) writes one scalar cell at a precision,
which a renderer escapes for its format. A package installs a format by
naming the object in its `lofi.report_format` [entry point group](../../reference/entry-points.md#registered-groups-and-names), and a report
declares it by that entry's name:

```toml
[project.entry-points."lofi.report_format"]
html = "my_package.reports:HTML"
```

## Declare a reducer

For a paired comparison, match the two conditions within the units declared
by the design before computing their difference. With complete, equally
weighted pairs, the mean of the differences equals the difference of the
means. But uncertainty also depends on how the two measurements vary together
within a pair. The test or resampling procedure must retain that pairing to
represent this dependence. The [comparison definitions](../../explanation/experiments-and-commitments.md#paired-tests-and-equivalence)
state how the declared reducers use the accepted rows and which assumptions
the resulting comparisons require.

A reducer reads an experiment's completed rows when the results are ingested
([`lofi ingest`](../../reference/cli.md#ingest), or [`ingest_results`](../../reference/run.md#lofi.run.results.ingest_results) from Python) and adds the rows and
comparisons it derives. The derived rows are written with the accepted rows,
and the comparisons feed the [preregistration](../../explanation/experiments-and-commitments.md#preregistration) audit and the declared
[multiple comparison correction](../../explanation/experiments-and-commitments.md#corrections) (see
[Choose a sweep design](choose-a-sweep-design.md#declare-a-multiple-comparison-correction)).
Each reducer is an installed plugin in the `lofi.reducer` [entry point group](../../reference/entry-points.md#registered-groups-and-names),
and an experiment declares at most one reducer of each kind.

[`experiments/s2/source-complexity.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s2/source-complexity.yaml)
declares a `source-comparison` reducer:

```yaml
reducers:
  - kind: source-comparison
    metric: string-log-likelihood-per-sentence
    source_path: comparison.source
    candidate_path: model.language.grammar.name
    data_size_path: data.datasets.train.dataset.size
    condition_path: comparison.condition
    direction_path: comparison.direction
    twin_path: comparison.twin
    description_length_metric: description-length-nats
    primary_condition: primary
    reversed_ground_truth_condition: reversed-ground-truth
    lexicon_scrambled_condition: lexicon-scrambled
    observed_twin: observed
    permutation: {replicates: 999, seed: 109}
    equivalence: {replicates: 999, seed: 110}
    falsifier: A candidate grammar or validity control matches the planted source grammar.
    smoke_test: false
```

The `*_path` fields are dotted paths into each point's resolved document, and
the reducer reads the point's values there to say which source and candidate
grammar, data size, condition and twin a row belongs to. Its two tests run
under declared replicate counts and seeds. The declaration refuses two
conditions or two paths that coincide, and a fit metric that is also the
description length metric.

The packaged reducers:

| `kind` | Derives |
| --- | --- |
| `variance` | A method of moments decomposition of one metric over at least two named random factors, each read from a row column, with a [bootstrap interval](../evaluation/evaluate-a-run.md#set-the-interval). |
| `rescoring` | One metric rescored over released predictions across a grid of scoring conventions, read in a declared prediction format. |
| `rank-churn` | The change in method ranking between two named scoring conditions. |
| `source-comparison` | Source grammar selection from held-out fit rows and point values, with permutation and equivalence tests. |
| `transfer` | A zero-shot transfer report from a source language to declared gold or silver languages. |
| `cross-lingual` | Macro, floor, disparity and transfer gap summaries over languages. |
| `encoder-ablation` | The table crossing encoder [checkpoint](../models/choose-a-token-encoder.md#load-a-pretrained-checkpoint) and freeze state, over at least four seeds. |
| `crossed-recovery` | Recovery summaries over fully crossed factor cells and at least four seeds, against a sign-flip null. |
| `coherence-collapse` | A seed-paired comparison of collapse under KL and cross-entropy coherence, over at least four seeds. |
| `residual-structure` | A regression of a held-out response column on declared root entropy, non-target and derivation count columns. |
| `restitutive-coverage` | A restitutive scope query over one interpreted dataset [lexicon](../grammars/declare-a-lexicon.md#list-entries-inline). |
| `node-vocabulary` | A comparison of the four node vocabulary arms, each read from a persisted causal probe record. |

The `node-vocabulary` reducer declares `arms`, one entry for each of the four
representations, each naming the `source` probe records file (relative to the
experiment document's directory), the causal `probe` role and the `record`
(such as `pass-0`). It refuses arms whose evaluated pair sets, logical cells or
per-cell nulls differ, and emits one row per arm with its `pair_digest`,
`cell_count` and `mean_iia`.

```yaml
reducers:
  - kind: node-vocabulary
    arms:
      - {representation: derivation-variables, source: arms/derivation/probe-records.json, probe: causal, record: pass-0}
      - {representation: trained-transformer-sae-latents, source: arms/trained-sae/probe-records.json, probe: causal, record: pass-0}
      - {representation: dense-linear-probe, source: arms/dense/probe-records.json, probe: causal, record: pass-0}
      - {representation: random-transformer-sae-control, source: arms/random-sae/probe-records.json, probe: causal, record: pass-0}
```

The `rescoring` reducer's `metric` is resolved through the
`lofi.rescoring_metric` entry point group, whose packaged entry is
`recall-homogeneity`, and its `prediction_format` through
`lofi.prediction_format`, whose packaged entry is `lofi-json`. The
[configuration reference](../../reference/configuration/index.md) places
`reducers` among the experiment document's sections, and the
[specification reference](../../reference/spec.md) lists every field of each
[`ReducerSpec`](../../reference/spec.md#lofi.spec.ReducerSpec).
