# Configure a trainer

A run document's `trainer` section declares how a point is fitted: which
trainer kind drives the optimizer, over which dataset, for how long, in what
batches, when it stops early, and how often it checkpoints. The `optimizer` and
`scheduler` sections declare the update rule and its learning-rate schedule,
and the root-level `type_sampler` and `pair_sampler` sections declare the
generated data that pretraining trainers draw. This guide selects each one,
composes the trainers that run several objectives over one model, and lists
how each declaration is refused.

The examples compose
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml)
and the documents under
[`docs/examples/models-training/`](https://github.com/logical-form-induction/lofi/tree/main/docs/examples/models-training)
with a [`RunSpecCodec`](../../reference/spec.md), as the
[first tutorial](../../tutorials/01-first-run.md) does:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
```

## Choose a trainer kind

Five trainer kinds are registered under the `lofi.trainer` entry-point group.
Each validates its own fields and declares which component roles it needs and
which it refuses as inert.

| `kind` | What it fits | Progress unit | Packaged fragment |
| --- | --- | --- | --- |
| `parser` | The chart parser and its front end on sentence batches, under every declared loss term. | Epochs | `trainer/parser.yaml` |
| `alternating` | Two or more named tasks over one parser graph, interleaved by a mixing policy. | Epochs of its `base` | none |
| `staged` | One parser graph through consecutive stages, each training its own parameter groups. | Epochs of its `base` | none |
| `terms` | Term encoders and decoders of named spaces on generated terms and combinator pairs. | Optimizer steps | `trainer/terms.yaml` |
| `distance` | One term encoder on generated type pairs, so that embedding distance tracks edit distance. | Optimizer steps | `trainer/distance.yaml` |

Select a packaged fragment from the `defaults` list (`- trainer: parser`) or
with `--set /trainer=parser` on the command line, then override single leaves.

## Declare a parser trainer

The synthetic example declares a parser trainer in full:

```python
run = codec(document)
trainer = run.trainer
print(trainer.kind, trainer.epochs, trainer.batch_size, trainer.accumulation_steps)
print(trainer.batch_sampler)
print(trainer.early_stopping.metric, trainer.early_stopping.patience)
```

```text
parser 1 2 2
LengthBucketBatchSamplerSpec(kind='length-bucket', bucket_factor=100, reshuffle_each_epoch=False)
loss/grammaticality 2
```

The fields of [`ParserTrainerSpec`](../../reference/configuration/training.md):

| Field | What it says |
| --- | --- |
| `dataset` | The dataset role under `data.datasets` whose sentence batches are trained and whose held-out split is evaluated. |
| `epochs` | Positive number of passes over the training partition. |
| `batch_size` | Positive number of items in one microbatch. |
| `accumulation_steps` | Positive number of microbatches whose gradients are summed into one optimizer step. Defaults to `1`. |
| `max_grad_norm` | Positive gradient-clipping norm, or `null` for no clipping. |
| `active_groups` | The parameter groups that take a gradient, or `null` for every declared group. A group whose component is declared frozen stays frozen even when named. |
| `batch_sampler` | How an epoch is partitioned into batches (see below). |
| `early_stopping` | The retained metric that selects the best checkpoint, and when to stop. |
| `checkpoints` | The resumable checkpoint cadence, in completed epochs, and how many to keep. |

A parser trainer needs a parser, a token encoder, a subword pooler and a loss
section. It refuses a `type_sampler` or `pair_sampler`, since it draws nothing
from either:

```python
for overrides in [("/type_sampler=recursive",), ("trainer.batch_size=1",)]:
    try:
        codec(document, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", refused)
```

```text
ConfigDocumentError | docs/examples/synthetic-parser.yaml cannot be composed: trainer: the 'parser' trainer does not consume component roles ['data.type_sampler']
ConfigDocumentError | docs/examples/synthetic-parser.yaml cannot be composed: trainer.accumulation_steps: 2 cannot divide an epoch of 3 training occurrences, taken 1 at a time, whose 3 microbatches close no whole accumulation windows
```

The second refusal is the epoch-shape check. Composition counts the
occurrences the dataset assigns to training, takes them `batch_size` at a time,
and refuses an `accumulation_steps` that does not divide the resulting
microbatches into whole optimizer steps. The synthetic example retains a
quarter of its four sentences for validation, so three occurrences remain.

### Batch samplers

`batch_sampler.kind` selects one of the `lofi.batch_sampler` entries. The
`length-bucket` and `random` samplers declare `reshuffle_each_epoch`; `false`
keeps the first epoch's partition for the whole run. The `sequential` sampler
always preserves item order and takes no shuffle setting.

| `kind` | Partition | Extra field |
| --- | --- | --- |
| `length-bucket` | Partitions shuffled items through buckets of similar length. | `bucket_factor` |
| `random` | Partitions a random permutation of the items. | none |
| `sequential` | Partitions the items in their declared order. | none |

### Early stopping

`early_stopping` monitors one retained metric, aggregated over each epoch's
validation pass:

| Field | What it says |
| --- | --- |
| `metric` | The telemetry key of the monitored metric, such as `loss/grammaticality`. A loss term logs under `loss/<name>`, where the name defaults to the objective's `kind`. |
| `mode` | `min` or `max`: whether smaller or larger values improve. |
| `patience` | Consecutive non-improving epochs after which fitting stops. |
| `minimum_delta` | Absolute change a value must exceed to count as an improvement. |
| `keep_best` | How many validation-best checkpoints to keep. |

Validation reports each loss term under `loss/<name>` and each of its
diagnostics under `loss/<name>/<diagnostic>`, so composition refuses a
`metric` that names no declared term. Take the key from a loss term's name, as
[Combine losses](combine-losses.md) describes:

```python
try:
    codec(document, overrides=("trainer.early_stopping.metric=loss/simplicity",))
except Exception as refused:
    print(type(refused).__name__, "|", refused)
```

```text
ConfigDocumentError | docs/examples/synthetic-parser.yaml cannot be composed: trainer.early_stopping.metric: validation reports no 'loss/simplicity'; it reports loss/{term} and loss/{term}/{diagnostic} for the declared loss terms ['coherence', 'grammaticality']
```

A diagnostic name is not checked, because objectives do not declare their
diagnostics; a `loss/<name>/<diagnostic>` key that the term never reports is
refused after the first validation pass.

Field-level refusals name the leaf:

```python
for overrides in [
    ("trainer.batch_size=0",),
    ("trainer.max_grad_norm=-1.0",),
    ('trainer.active_groups=["parser", "parser"]',),
    ("trainer.early_stopping.mode=lowest",),
]:
    try:
        codec(document, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", refused)
```

```text
ConfigValueError | trainer.batch_size was refused: batch_size is not positive: 0
ConfigValueError | trainer.max_grad_norm was refused: max_grad_norm is not positive: -1.0
ConfigValueError | trainer.active_groups was refused: active_groups names groups more than once: ['parser']
SettingParseError | trainer.early_stopping.mode expects one of ['min', 'max']; got 'lowest'
```

## Alternate tasks over one parser

An `alternating` trainer splits the declared loss terms into named tasks and
interleaves their batches under one mixing policy. Its `base` is a complete
parser trainer, which fixes the dataset, horizon, batches, stopping rule and
checkpoints. The checked-in
[`alternating-trainer.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/models-training/alternating-trainer.yaml)
trains the synthetic example's two loss terms as two tasks:

```yaml
trainer:
  kind: alternating
  base:
    kind: parser
    dataset: train
    epochs: 1
    batch_size: 2
    # batch sampler, active groups, early stopping and checkpoints
  tasks:
    - {kind: parser, name: grammaticality, dataset: train, objectives: [grammaticality]}
    - {kind: parser, name: coherence, dataset: train, objectives: [coherence]}
  mixing: {kind: round-robin}
```

```python
alternating = Path("docs/examples/models-training/alternating-trainer.yaml")
run = codec(alternating)
print(run.trainer.kind, [task.name for task in run.trainer.tasks], run.trainer.mixing.kind)
```

```text
alternating ['grammaticality', 'coherence'] round-robin
```

Each task is a step from the `lofi.step` group. A step names itself, its
dataset role and, in `objectives`, the loss-term names it trains; an empty
`objectives` trains every term. Composition refuses a name that no loss term
carries, at any depth of nesting, and lists the declared names:

```python
tasks = (
    'trainer.tasks=[{"kind": "parser", "name": "grammaticality", "dataset": "train",'
    ' "objectives": ["grammaticality"]}, {"kind": "parser", "name": "simplicity",'
    ' "dataset": "train", "objectives": ["simplicity"]}]'
)
try:
    codec(alternating, overrides=(tasks,))
except Exception as refused:
    print(type(refused).__name__, "|", refused)
```

```text
ConfigDocumentError | docs/examples/models-training/alternating-trainer.yaml cannot be composed: trainer: training task 'simplicity' selects absent objectives ['simplicity']; declared objectives are ['coherence', 'grammaticality']
```

| Step `kind` | What it trains | Extra fields |
| --- | --- | --- |
| `parser` | The parser objectives on sentence batches. | none |
| `supervised-warm-start` | Supervised objectives on sentence batches that carry gold forms. | none |
| `token-warm-up` | The token encoder on sentences whose tokens have lexical types in a space. | `space` |
| `terms` | A space's term encoder and identity decoder on generated terms and, with `combinators: true`, its combinator decoders on generated rule instances. | `space`, `combinators`, `generation`, `stop` |
| `controller` | The controller on generated rule instances. | `space`, `max_depth`, `stop` |
| `grounded` | Grounded-inference objectives on labeled inference pairs. | none |
| `judgment` | Judgment objectives on collected inference-judgment pairs. | none |
| `alternating` | Nested tasks under their own mixing policy. | `tasks`, `mixing` |

The `terms` and `controller` steps draw generated data, so a run that uses them
declares the samplers described below. A `terms` step's `generation` is the
same declaration a `terms` trainer makes, described under
[Pretrain terms or type distances](#pretrain-terms-or-type-distances). A `stop`
is an accuracy target, `{metric, target, max_steps}`, that ends the step when a
held-out accuracy reaches `target` or after `max_steps` optimizer steps. The
step must be the last stage of a staged trainer, and its metric must be one the
step evaluates: `loss/<space>/autoencoding/exact_match`, with
`loss/<space>/combinators/exact_match` when it trains combinators, for a
`terms` step, and `loss/<name>/accuracy` for a `controller` step.

`mixing.kind` selects one of the `lofi.mixing` entries:

| `kind` | How one epoch is shared among tasks | Extra field |
| --- | --- | --- |
| `round-robin` | Every task takes as many steps as the largest task needs. | none |
| `proportional` | Every task batch is visited exactly once. | none |
| `ratio` | Positive integer weights, one per task, are repeated until every task is covered. | `weights` |
| `temperature` | A fixed-size epoch is allocated by temperature-scaled dataset size. | `temperature` |

An alternating trainer is refused with fewer than two tasks or with two tasks
of one name. Its base must take one microbatch per optimizer step and keep one
fixed epoch partition:

```python
try:
    codec(alternating, overrides=("trainer.base.accumulation_steps=2",))
except Exception as refused:
    print(type(refused).__name__, "|", refused)
```

```text
ConfigValueError | trainer was refused: alternating training requires one microbatch per optimizer step
```

## Train in stages

A `staged` trainer runs consecutive stages over one parser graph. Each stage
names one step, the parameter groups it trains, and its number of epochs. The
checked-in
[`staged-trainer.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/models-training/staged-trainer.yaml)
warms the token encoder up on the gold lexicon before fitting the parser, as
[`lofi/configs/supertag-conditions/induction.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/supertag-conditions/induction.yaml)
does for the supertagging baselines:

```yaml
trainer:
  kind: staged
  validation_stage: parser
  base:
    kind: parser
    epochs: 2
    # the remaining parser-trainer fields
  stages:
    - name: lexical-warm-up
      step: {kind: token-warm-up, name: tokens, dataset: train, space: categories}
      active_groups: [token_encoder]
      epochs: 1
    - name: parser
      step: {kind: parser, name: parser, dataset: train}
      active_groups: [token_encoder, terms, identity, application, controller, parser]
      epochs: 1
```

```python
staged = Path("docs/examples/models-training/staged-trainer.yaml")
run = codec(staged)
for stage in run.trainer.stages:
    print(stage.name, stage.step.kind, stage.active_groups, stage.epochs)
```

```text
lexical-warm-up token-warm-up ('token_encoder',) 1
parser parser ('token_encoder', 'terms', 'identity', 'application', 'controller', 'parser') 1
```

A token warm-up step needs lexical types for the tokens, which the document
takes from a `gold-lexicon` constraint source; see
[Supervise and constrain a learner](supervise-and-constrain.md).
`validation_stage` names the stage whose epochs are validated, and must be the
last stage, or `null`. The stage epochs must sum to the base horizon, and
active groups are declared only within stages:

```python
for overrides in [
    ("trainer.base.epochs=3",),
    ("trainer.validation_stage=lexical-warm-up",),
    ('trainer.base.active_groups=["parser"]',),
]:
    try:
        codec(staged, overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", refused)
```

```text
ConfigValueError | trainer was refused: staged training declares 2 stage epochs, not its base horizon of 3
ConfigValueError | trainer was refused: validation stage 'lexical-warm-up' is not the final stage 'parser'
ConfigValueError | trainer was refused: staged training declares active groups only within its stages
```

## Pretrain terms or type distances

The `terms` and `distance` trainers fit term-space components on generated
data, counting progress in optimizer steps. Their checkpoint `interval` is
therefore a number of steps.

A `terms` trainer names the `spaces` it pretrains and the `generation` of its
terms, and sets exactly one of a fixed `steps` horizon and an accuracy `stop`.
`generation.max_depth` bounds the generated terms and rule instances, and
`child_depth`, `branching` and `constructor_weights` shape the distribution of
generated terms. `teacher_forcing_saturation` is the fraction of the run at
which the decoders' sampled share of each batch, rising linearly from zero,
reaches one; `null` keeps every step teacher-forced.

A `distance` trainer names one `encoder` and the `sort` of its types, and
declares `steps`, `inverse_distance_power` (the exponent of inverse-distance
pair weights) and `embedding_penalty`. An optional `ball_radius` bounds the
edit distance between the two types of a pair, and an optional
`depth_curriculum` grows the available depth from `start_depth` over `steps`
positions. The packaged
[`synthetic-distance.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-distance.yaml)
is a complete distance run:

```python
distance = codec(Path("lofi/configs/run/synthetic-distance.yaml"))
print(distance.trainer.kind, distance.trainer.steps, distance.trainer.checkpoints, distance.release.checkpoint)
print(distance.type_sampler)
```

```text
distance 4 CheckpointPolicySpec(interval=2, keep=2) resume-latest
ExhaustiveTypeSamplerSpec(kind='exhaustive', max_size=3, weighting=UniformExhaustiveWeightingSpec(kind='uniform'))
```

A distance trainer writes no validation-best checkpoint, so its run releases
`resume-latest`; see [Checkpoint and resume a run](resume-from-a-checkpoint.md).

## Declare the samplers

Generated-data trainers and steps read two root-level sections.

`type_sampler` draws types from the learner language. `kind: recursive` grows
types up to `max_size` nodes. `kind: exhaustive` enumerates every type up to
`max_size` and weights them `uniform`ly or by `size`, with an `exponent`.
Packaged fragments are `type_sampler/recursive.yaml` and
`type_sampler/exhaustive.yaml`.

`pair_sampler` draws labeled rule instances from the declared grammar rules.
Its one kind, `rule`, takes no fields; the packaged fragment is
`pair_sampler/rule.yaml`.

## Choose an optimizer and a schedule

`optimizer.kind` selects one of the `lofi.optimizer` entries, and
`scheduler.kind` one of the `lofi.scheduler` entries. Each packaged fragment
holds only its `kind`, so the class defaults apply until a document sets a
field:

| Optimizer | Fields |
| --- | --- |
| `adamw` | `learning_rate`, `betas`, `epsilon`, `weight_decay` (default `0.01`) |
| `adam` | `learning_rate`, `betas`, `epsilon`, `weight_decay` |
| `sgd` | `learning_rate`, `momentum`, `dampening`, `weight_decay`, `nesterov` |

| Scheduler | Fields |
| --- | --- |
| `constant` | `warmup_steps`, `warmup_start_factor` |
| `cosine` | `warmup_steps`, `warmup_start_factor` |
| `linear` | `warmup_steps`, `warmup_start_factor`, `final_factor` |

Select both by group and set one leaf. Leaves the document already sets, such
as `optimizer.weight_decay: 0.0`, carry over to the new selection:

```python
swapped = codec(
    document,
    overrides=("/optimizer=sgd", "/scheduler=linear", "optimizer.momentum=0.9"),
)
print(swapped.optimizer)
print(swapped.scheduler)
```

```text
SGDOptimizerSpec(kind='sgd', learning_rate=0.01, momentum=0.9, dampening=0.0, weight_decay=0.0, nesterov=False)
LinearSchedulerSpec(kind='linear', warmup_steps=0, warmup_start_factor=1e-08, final_factor=0.0)
```

The same selection on the command line is
`--set /optimizer=sgd --set /scheduler=linear --set optimizer.momentum=0.9`.
The scheduler section is optional; a run without one keeps a constant learning
rate.

## Related

- [Checkpoint and resume a run](resume-from-a-checkpoint.md) declares the
  checkpoint policy and the released checkpoint.
- [Choose a device policy](choose-a-device-policy.md) declares where and in
  what precision the trainer runs.
- [Training declarations](../../reference/configuration/training.md) renders
  every trainer, step, optimizer, scheduler and sampler class.
- [Training](../../reference/train.md) renders the training plans and
  plugins that execute these declarations.
