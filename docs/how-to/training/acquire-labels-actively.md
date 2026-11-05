# Acquire labels actively

An active-learning document declares one bounded, round-by-round collection of
judgments. Each round refits a judgment model to every judgment
collected so far, scores the candidate items not yet judged, selects one batch
under a declared acquisition rule, and collects the responses to that batch.
The candidates are the bead items a judgment-item construction document
declares,
the model is the response model of a judgment-fit document refitted by MAP, and
its predictors are readouts of a trained LoFI point. This guide writes the
documents, runs them with `lofi active-learn`, fields the batches through bead
or answers them with a simulated collector, and runs the same document as an
experiment.

The documents below are checked in under
[`docs/examples/models-training/active-learning/`](https://github.com/logical-form-induction/lofi/tree/main/docs/examples/models-training/active-learning),
and the acceptance test `tests/acceptance/test_active_learning.py` runs them
over the packaged synthetic parser, with the bead collector and with each
simulated collector.

## Write the active-learning document

```yaml
acquisition:
  kind: rc2.1
  batch_size: 2
  allocate_by: "item['structure']"
  constraints:
    - {constraint_type: uniqueness, property_expression: "item['verb']"}
  share_tolerance: 0.5
stopping: {kind: asymptote, window: 3, maximum_gain: 0.0}
max_rounds: 2
bootstrap_samples: 200
pool: items.json
factors: {structure: "item['structure']", verb: "item['verb']"}
initial: {kind: bead, responses: initial.jsonl}
encoding: encoding.json
fit: fit.yaml
predictors:
  run: trained
  inputs: {words: sentence}
  tokenizer: {backend: whitespace, language: en, model_name: null}
  readouts:
    p_gram: {readout: sentences, measure: log-mass}
collector: {kind: bead}
cross_validation:
  held_out_unit: participant
  splitter: kfold
  folds: 3
  coverage_threshold: 0.0
  unseen_participant: plug_in
```

Every path is relative to the directory holding the document.

| Field | What it declares |
| --- | --- |
| `acquisition` | The rule that selects each batch, from the `lofi.acquisition` entry-point group. |
| `stopping` | The condition, checked after each round, that ends collection. |
| `max_rounds` | Positive bound on the number of rounds. |
| `confidence` | Level of the agreement and rank-correlation intervals, in (0, 1). Defaults to `0.95`. |
| `bootstrap_samples` | Positive number of resamples behind the agreement interval. Defaults to `1000`. |
| `pool` | The judgment-item construction document the candidate pool is built from. |
| `factors` | The bead expression each factor every judgment row records is read by over the item's attributes, keyed by the factor's name. The response model's factor terms and a factor `held_out_unit` read these; no other attribute becomes a factor, and an expression an item cannot answer as one level name is refused before the run starts. |
| `initial` | How the judgments the first round starts from are obtained: `kind: bead` reads already collected bead judgments, and `kind: collected` has the collector answer a seeded draw from every declared stratum as round 0. |
| `encoding` | The bead response encoding every bead response of the run answers. Required when `initial` or `collector` has `kind: bead`, and refused when neither does. |
| `fit` | The judgment-fit document whose response model each round refits. |
| `predictors` | The trained point, by its run directory; the rendered element each input role it publishes reads; the bead tokenizer splitting that element into words; and the published readout that fills each predictor column the response model reads. |
| `collector` | The collector that answers each batch, from the `lofi.active_collector` entry-point group. |
| `cross_validation` | The cross-validation each round's performance is measured by. |

The document declares no seed. Every random stream of a run derives from its
run seed: the MAP initializations, the initial draw, the cross-validation
split, the agreement bootstrap and the collector's own draws.

Every rule that groups or constrains candidates names the attributes it reads
with a [bead DSL](https://github.com/FACTSlab/bead) expression evaluated over
the item's `item_metadata`, as bead's list partitioner evaluates one: the
variable `item` is the item's attributes, so `item['structure']` reads the
attribute `structure`. When the run is checked, before anything runs, the pool
is constructed and every declared expression and factor is read from every
pool item; one naming an attribute an item does not carry is refused.

## Declare the candidate pool

`pool` names a [judgment-item construction](../judgments/construct-judgment-items.md)
document. Its resources are bead's, in bead's own serialized shape: bead
templates whose slots are filled under their own constraints, bead lexicons,
generators each proposing one more lexicon, the bead filling strategy of each
template, the bead item template whose elements name what each item renders
and whose constraints every item satisfies, and the attributes each item
records. The example reads its templates and its argument lexicon from bead
JSON Lines files beside it and proposes its verbs with a pinned masked
language model:

```json
--8<-- "docs/examples/models-training/active-learning/items.json"
```

Its attribute `verb` reads `slots.verb.lemma`, the lemma filling the `verb`
slot, and `structure` reads `template.name`. The acceptance test points the
generator's `checkpoint` at a small local model whose completions are words
the synthetic parser knows.

## Choose how the run starts

`initial.kind` selects one of two starts:

| `kind` | Fields | The judgments the first round starts from |
| --- | --- | --- |
| `bead` | `responses` | Bead judgments already collected: `responses` is a JSON Lines file of bead annotation records answering the document's `encoding`. |
| `collected` | `stratify_by`, `items_per_stratum` | The collector's answers to round 0, a draw of `items_per_stratum` candidates from every stratum `stratify_by` names. |

### Start from bead judgments

The initial judgments are read through the bead judgment source under the
document's top-level `encoding`, the one encoding every bead reader of the run
shares. It is a bead `ResponseEncoding`, whose labels are the response levels
in order:

```json
{"name": "acceptability", "n_levels": 3, "scale_type": "ordinal", "labels": ["low", "mid", "high"]}
```

`responses` holds one bead annotation record per line, each naming the
annotator, the item's UUID, the question and the response label:

```json
{"annotator_id": "annotator-a", "item_id": "0f5e3c9a-6a52-5f5c-9a2e-3a8f2c7b1d40", "question_name": "acceptability", "response_label": "high"}
```

A record answering another question is set aside and counted in the run's
manifest as `initial_skipped`. Every judged item must be a pool item, and a
judged item is not offered again.

### Start from a collected draw

```yaml
initial: {kind: collected, stratify_by: "item['structure']", items_per_stratum: 3}
```

`stratify_by` is an expression naming each candidate's stratum. Before round 1
the run draws `items_per_stratum` candidates from each stratum,
without replacement, under a seed derived from the run seed, so running the
document again with the same seed draws the same candidates. The draw is round
0's batch, and the run's collector answers it as it answers every later batch:
a simulated collector at once, and the bead collector by writing it to
`rounds/0/batch.jsonl` and stopping until its responses are placed beside it.
A stratum holding fewer than `items_per_stratum` candidates is refused. Since
each acquisition stratum's performance is a rank correlation over its judged
items, a draw of fewer than three items per stratum leaves the first round's
strata unrankable and is refused there.

## Declare the refitted model

`fit` names a judgment-fit document, as
[Fit a judgment model](../judgments/fit-a-judgment-model.md) describes. An
active-learning run reads three of its sections:

- `response`, the model refitted each round. Its levels must equal those of
  the document's `encoding`, which the run checks once before any round, and
  the readout columns it reads
  must be exactly the columns `predictors.readouts` fills.
- `inference`, which must be MAP. Each refit draws its initialization from the
  run seed.
- `dataset`, the identifier the collected judgments are reported under.

A fit declaring a `comparison`, a `dynamic_range` or an `ablation` is
refused, since an active-learning run performs none of them; its
cross-validation is declared in the active-learning document itself.

```yaml
dataset: collected-acceptability
response:
  ResponseDeclaration_responsespec:
    family: {family: cumulative}
    levels: 3
    objective: {objective: conditional}
    predictor:
      - {kind: link, readout: p_gram, linking: {family: linear}}
    identification: {constraint: fixed_discrimination, value: 1.0}
    priors: {thresholds: {}}
inference:
  inference: map
  optimizer: {kind: adam, learning_rate: 0.1}
  steps: 300
  initial_scale: 0.1
elicitation: {method: participants, version: "1"}
calibration: null
comparison: null
dynamic_range: null
ablation: []
interannotator: null
```

## Declare the cross-validation

Each round's performance is measured by the cross-validation the document
declares. Every fold is refitted by MAP on its retained judgments and predicts
its held-out judgments, and those cross-validated predictions measure the
model's performance and decide each stratum's share of the next batch.

| Field | What it declares |
| --- | --- |
| `held_out_unit` | The row column held out together: `response`, `participant`, `item`, or one of the document's `factors`. |
| `splitter` | The splitter, from the `lofi.splitter` entry-point group: `kfold` or `leave-one-group-out`. |
| `folds` | The number of folds, at least two, for `kfold`; `null` for `leave-one-group-out`, which holds out one level per fold. |
| `coverage_threshold` | The largest share, in [0, 1], of a fold's held-out participants its retained judgments may never show before the fold needs an unseen-participant rule. |
| `unseen_participant` | `plug_in`, `marginal`, or `null`: the rule predicting every participant a fold never retained. |

A fold whose unseen participants exceed `coverage_threshold` with no
`unseen_participant` rule is refused. Candidates are scored for a participant
the model never saw under the same rule, or at the population mean where it is
`null`. The split's seed derives from the run seed.

## Declare the trained point's predictors

`predictors.run` is the run directory of a completed point, which the run
restores through `lofi.run.hub.restore_trained_point`. A parser point publishes
one input role, `words`, and `predictors.inputs` maps it to the rendered element
of every pool item it reads, the same mapping a
[served point](../serving/serve-through-bead.md) declares for its roles; a
mapping naming another role, or an item rendering no such element, is refused.
`predictors.tokenizer` declares a bead tokenizer in the shape of bead's
`TokenizerConfig`: `backend`, one of `whitespace`, `spacy` or `stanza`, the
`language` it tokenizes and its `model_name`, or `null` for the one bead
resolves. It splits the element into the words the point's forward pass is
published over once.
Each entry of `predictors.readouts` fills one predictor column from a mass
readout the point publishes, at the span covering the whole element: `measure: log-mass` takes
its log mass and `measure: log-complement` the log of the remaining mass. A
readout the point does not publish, or one that is not a mass readout, is
refused.

## Choose the acquisition rule

One rule is registered, `rc2.1`. It ranks the remaining candidates by the
squared distance of their predicted response from the pool mean, highest
first, and fills a batch of `batch_size` items greedily under its quotas and
the declared bead list and batch constraints:

- each candidate is allocated to the stratum `allocate_by` names, each stratum
  receives a share of the batch proportional to the model's current
  performance on that stratum, and the realized share of every stratum must
  lie within `share_tolerance` of its target;
- each entry of `constraints` is a bead list constraint in bead's own
  serialized shape.
  A `uniqueness` constraint passes over a candidate whose
  `property_expression` value, a level name or a list of them, a selected item
  already has; a `conditional_uniqueness` constraint does so only for items
  its `condition_expression` holds of. Two constraints are independent, so
  `item['left']` and `item['right']` let a level fill `left` in one item and
  `right` in another. A `diversity` constraint is checked on the selected
  batch, which must hold at least `min_unique_values` distinct values, and a
  `size` constraint is checked against `batch_size` when the document is read.
  A `balance` constraint holds each level of a single-level expression to its
  `target_counts` entry, or with `target_counts: null` to an equal share of the
  batch over the levels the pool holds, within `tolerance` as a proportion of
  the target; a candidate whose level is already at its bound is passed over,
  and the completed batch is checked as bead checks it. A `quantile`
  constraint divides the pool into `n_quantiles` equal-frequency bins of a
  numeric expression and takes at most `items_per_quantile` candidates from
  each; a `grouped_quantile` constraint fits those bins within each level of
  its `group_by_expression`; and a `grid_stratification` constraint bins each
  of its `dimensions` by its own bead binning (`quantile`, `equal_width`,
  `threshold`, `stddev` or `categorical`) and takes at most `items_per_cell`
  candidates from each cell, the numeric bins fitted within each group when a
  `group_by_expression` is declared. Bins are assigned by bead's own grid
  stratification over the remaining pool. One `ordering` constraint may fix
  the order the batch is written in: practice items, where
  `practice_item_property` is declared, come first; items are then grouped
  into contiguous blocks of one `block_by_property` level, no block spanning
  more than `max_distance`; each `precedence_pairs` entry puts its `before`
  item ahead of its `after` item when both are selected; and items sharing a
  `no_adjacent_property` level have at least `min_distance` items (one, when
  not declared) between them. Within those rules items keep their rank order,
  and a batch no order satisfies is refused. Randomizing within blocks belongs
  to the presentation runtime, so `randomize_within_blocks: true` is refused.
  Any other `constraint_type` is refused when the document is read;
- each entry of `batch_constraints` is a bead batch constraint over all the
  batches the run acquires, each round's batch being one bead list. A
  `diversity` batch constraint passes over a candidate whose level already
  appears in `max_lists_per_value` earlier batches. A `balance` batch
  constraint passes over a candidate whose level would exceed its
  `target_distribution` share plus `tolerance` of all acquired items, takes
  first the candidates a level needs to reach its share minus `tolerance`,
  and checks the acquired shares after each batch. A `coverage` batch
  constraint takes first, in rank order, a candidate of each `target_values`
  level (every level the pool holds, when `null`) no acquired batch holds,
  until `min_coverage` of the levels are covered, and a `min_occurrence`
  batch constraint takes first the candidates of every level acquired fewer
  than `min_occurrences` times.

A candidate's predicted response is its expected response level for a
participant the model never saw, under the cross-validation's
`unseen_participant` rule, or at the population mean where it declares none. A
stratum's performance is the Spearman correlation, over the judged items
allocated to that stratum, between each item's mean response and its mean
cross-validated prediction. A stratum with fewer than three judged items, or
with a constant side, cannot be ranked and is refused. A stratum whose
performance is negative is floored at zero: it receives no proportional share
of the batch, only any rounding remainder left once every positive stratum has
its share. If no stratum's performance is positive, the batch is split evenly
over the strata. Each round's `stratum-share` rows record every stratum's
performance and whether it was floored.

## Choose the stopping condition

`stopping.kind` selects one of three conditions:

| `kind` | Stops when | Fields |
| --- | --- | --- |
| `agreement` | Krippendorff's alpha over the items judged more than once reaches `threshold`. | `threshold` in [0, 1]; `basis` |
| `rank-correlation` | The Spearman correlation between every judged item's mean response and its mean cross-validated prediction reaches `threshold`. | `threshold` in [-1, 1]; `basis` |
| `asymptote` | The rank correlations of the last `window` rounds lie within `maximum_gain` of one another. | `window` of at least 2; `maximum_gain` of at least 0 |

`basis` is `lower-bound` by default, which compares the lower end of the
interval at `confidence` with the threshold; `estimate` compares the point
estimate.

## Run the document and field each batch

Train the point first; `lofi run` prints its run directory:

```bash
uv run lofi run --config lofi/configs/run/synthetic-parser.yaml \
  --seed 3 --experiment-seed 7 --root runs
```

Point `predictors.run` at that directory (the example names a `trained` link
beside the document), then run the document:

```bash
uv run lofi active-learn docs/examples/models-training/active-learning/active-learning.yaml \
  --out collection --seed 0
```

The `bead` collector reads the responses to every batch through the document's
`encoding`, the same one a bead start reads through. It writes the first
round's batch to `collection/rounds/1/batch.jsonl`, one bead item per line, and
the run stops awaiting its responses:

```text
awaiting-responses 1 collection/rounds/1/batch.jsonl
```

Field the batch through bead, then place the annotation records that answer it
beside it as `collection/rounds/1/responses.jsonl` and run the same command
again. The run rebuilds the same rounds from the same inputs and seed, reads the
responses, refits, and continues until the next batch awaits, the stopping
condition holds, the remaining candidates cannot fill another batch, or
`max_rounds` is reached:

```text
completed 2 collection
```

A batch selected again that differs from the one written is refused rather
than overwritten, as are responses answering another question.

`collection/active-learning.json` records the run's manifest: the resolved
document, `completed` or `awaiting-responses` with the round it awaits (0 for
a collected start's draw), why a completed run stopped as `stop_reason`, the
rounds collected, the initial judgments read and set aside, the SHA-256 identity
of every document read, the run seed, the package versions and the source
revision. `collection/rows.json` holds the round rows.

A completed run's `stop_reason` is one of:

| `stop_reason` | Collection ended because |
| --- | --- |
| `agreement`, `rank-correlation`, `asymptote` | The declared stopping condition of that kind held. |
| `pool-exhausted` | The candidates not yet judged could not fill another batch under the acquisition rule, before the stopping condition held. |
| `max-rounds` | Round `max_rounds` was collected without the stopping condition holding. |

A stopping condition that holds is recorded even when the same round empties
the pool. A run whose pool is exhausted completes rather than failing: it
records every round it collected, and a pool that cannot fill even the first
batch completes with no round.

## Simulate the responses

A simulated collector answers every batch at once, so a run completes in one
command with no batch fielded. It draws or reads one rating design over the
whole candidate pool when the run starts, from the run's collector seed, and
answers candidate `i`, in pool order, with every response the design records
for its `i`-th item. Each participant of the design rates every candidate once.
Three simulated collectors are installed:

| `kind` | Fields | The design it answers from |
| --- | --- | --- |
| `predictor-truth` | `column`, `participants`, `scale`, `latent_range` | A known-effects design whose item scores are one predictor column of the trained point. `column` must be a column `predictors.readouts` fills. `scale` is required: `standardized` centers the column over the pool and divides it by its standard deviation, and `raw` takes it as it is. |
| `drawn-effects` | `participants`, `latent_range` | A known-effects design with one item per candidate and its own item scores, drawn from the collector seed. |
| `registered-design` | `design` | The design registered under `design` in the `lofi.judgment_design` entry-point group, such as `known-random-effects`. It must record ordinal responses on the fit's levels and rate at least as many items as the pool holds; its items beyond the pool are never asked. |

A known-effects design draws each participant's intercept and cutpoint offsets
and the responses through the generating model of the registered synthetic
designs, on the fit's response levels, with the declared number of
participants. Its population cutpoints divide `latent_range`, given as
`{lower, upper}` with `upper` above `lower`, into as many equal intervals as
the fit has levels: on `L` levels cutpoint `k` lies at
`lower + k * (upper - lower) / L`. A range wide enough for the item scores
places responses on every level whatever the fit's number of levels; on three
levels, `{lower: -3.0, upper: 3.0}` puts the cutpoints at -1 and 1. For
instance, a run whose true item effects are the trained point's root log mass:

```yaml
collector:
  kind: predictor-truth
  column: p_gram
  participants: 4
  scale: standardized
  latent_range: {lower: -3.0, upper: 3.0}
stopping: {kind: agreement, threshold: 0.5, basis: estimate}
```

```bash
uv run lofi active-learn docs/examples/models-training/active-learning/active-learning.yaml \
  --out simulated --seed 0
```

With `initial: {kind: collected, stratify_by: "item['structure']",
items_per_stratum: 3}` the same run reads no bead response file at all: the simulated collector answers the initial draw and every
batch, and the document declares no `responses` and no `encoding`.

## Read the round rows

Every collected round from round 1 contributes rows under the task
`active-learning`, each carrying its `round`, the run `seed`, the fit's
`dataset`, and the round's `stopped` flag and `stop_reason`. Only the last
round of a completed run is `stopped`, and its `stop_reason` is the manifest's;
every earlier round's is `null`. A collected start's round 0 is counted among
the initial judgments and contributes no row.

| `metric` | `value` | Further columns |
| --- | --- | --- |
| `agreement` | Krippendorff's ordinal alpha over every judgment so far | `lower`, `upper`; `n`, the judgments so far |
| `rank-correlation` | The cross-validated Spearman correlation | `lower`, `upper`; `n` |
| `batch-variance` | The variance of the batch's predicted responses | `n`, the batch size |
| `stratum-share` | One stratum's share of the batch | `stratum`; `performance`, the stratum's performance the batch was allocated by; `floored`, whether that performance was negative and floored at zero; `n`, the batch size |

## Run the document as an experiment

An experiment whose `config` declares `kind: active-learning` sweeps the
document: each sweep point is one run under the seed the experiment derives for
it, and its round rows are the point's report rows, named
`active-learning/<metric>`.

```yaml
config: {kind: active-learning, path: active-learning.yaml}
launch: {kind: serial}
sweep:
  grid: [{path: max_rounds, values: [2]}]
  seeds: [0]
  expected_points: 1
rows:
  - {evaluator: active-learning, metric: agreement}
  - {evaluator: active-learning, metric: rank-correlation}
primary: {evaluator: active-learning, metric: agreement}
```

A point runs in the launching process, so only the `serial` launcher is
admitted. A point whose batch awaits its responses is recorded with the status
`awaiting-responses` in the experiment's `STATUS.json`; once the responses are
placed in the point's `rounds/<r>/` directory, launching the experiment again
resumes it.

## Related

- [Fit a judgment model](../judgments/fit-a-judgment-model.md) declares the
  response model an active-learning run refits.
- [Serve through bead](../serving/serve-through-bead.md) serves the trained
  point that predicts the items a batch holds.
- [Run and price a sweep](../experiments/sweeps.md) launches an experiment and
  ingests its results.
