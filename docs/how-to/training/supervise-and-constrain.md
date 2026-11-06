# Supervise and constrain a learner

A training dataset's `constraint_source` declares where a learner's partial
supervision comes from, and its `constraint_policy` declares which part of the
corpus that source labels. Each selected constraint clamps a span: either to a
set of permitted categories or logical forms (the *target* route), or out of
the parser's chart altogether (the *negative bracket* route). A loss term,
`pinning` or `vector-anchor`, turns the clamps into a training signal, and the
parser trainer checks that the source and its consumers agree on the route
before anything is built.

## Pin lexical categories from the gold lexicon

[`lofi/configs/run/synthetic-pinning-lexical.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-pinning-lexical.yaml)
is the smallest complete example. It extends the packaged synthetic parser
with three selections:

```yaml
defaults:
  - synthetic-parser
  - data.datasets.train.constraint_source: gold-lexicon
  - data.datasets.train.constraint_policy.syntactic: all
  - loss: pinning
```

| Selection | What it says |
| --- | --- |
| `constraint_source: gold-lexicon` | `{kind: gold-lexicon, policy: syntactic}`: label forms with their gold lexical categories. |
| `constraint_policy.syntactic: all` | `{ambiguity: union, coverage: 1.0}`: emit a constraint for every selected form. |
| `loss: pinning` | Grammaticality, coherence at weight `0.1`, and `{kind: pinning, readout: types}`. |

Compose it and read the three parts back:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
run = codec(Path("lofi/configs/run/synthetic-pinning-lexical.yaml"))

bundle = run.data.datasets["train"]
print(bundle.constraint_source.kind, bundle.constraint_source.policy)
print(bundle.constraint_source.constraint_route())
print(bundle.constraint_policy.syntactic)
for term in run.loss.terms:
    print(term.key, term.weight)
```

```text
gold-lexicon syntactic
target
ConstraintPolicySpec(ambiguity=<AmbiguityMode.UNION: 'union'>, selector=RankSelectorSpec(kind='rank', first=0.0, last=1.0), coverage=1.0, seed=0)
grammaticality 1.0
coherence 0.1
pinning 1.0
```

The source serves the target route, and the `pinning` term reads its targets
through the `types` readout. Train it like any other point:

```bash
uv run lofi run \
  --config lofi/configs/run/synthetic-pinning-lexical.yaml \
  --root runs/pinning --seed 37 --experiment-seed 5
```

## Choose a constraint source

The `lofi.constraint_source` entry point group registers four sources. Each
declares the label domain it serves in `policy`, `syntactic` for categories or
`semantic` for logical forms, and the run selects its constraints under the
policy declared for that domain.

| `kind` | Labels | Domain | Packaged fragment |
| --- | --- | --- | --- |
| `gold-lexicon` | Each selected form with its gold lexical category. | `syntactic` only | `constraint_source/gold-lexicon.yaml` |
| `gold-derivation` | Constituents of each item's gold derivation with their categories. | `syntactic` only | `constraint_source/gold-derivation.yaml` |
| `gold-logical-form` | Each selected form with its gold lexical logical form, from the interpreted gold lexicon. | `semantic` only | `constraint_source/gold-logical-form.yaml` |
| `declared` | Spans located by declared `pins`, with the labels each permits. | either | `constraint_source/declared-pins.yaml`, `declared-inverted.yaml` |

A gold source declared under the other domain is refused at composition:

```python
def refusal(document, overrides):
    try:
        codec(Path(document), overrides=overrides)
    except Exception as refused:
        print(type(refused).__name__, "|", str(refused).splitlines()[-1])


lexical = "lofi/configs/run/synthetic-pinning-lexical.yaml"
refusal(lexical, ("data.datasets.train.constraint_source.policy=semantic",))
```

```text
ConfigValueError | data.datasets.train.constraint_source was refused: a gold lexicon labels forms with syntactic categories, so a gold-lexicon source cannot select under the semantic policy
```

A `declared` source lists its `pins`, at least one. The packaged
`declared-pins.yaml` uses all three kinds:

```yaml
kind: declared
policy: syntactic
pins:
  - {kind: item-feature, item: word-0, feature: NP}
  - {kind: span-pattern, pattern: [word-2, null], permitted: ["S\\NP"]}
  - {kind: sentence-top, permitted: [S]}
```

| Pin | Locates |
| --- | --- |
| `item-feature` | Every occurrence of one lexical `item`, pinned to one `feature`. |
| `span-pattern` | Every span matching `pattern`, a list of words in which `null` matches any word, pinned to the `permitted` labels. |
| `sentence-top` | The span covering the whole sentence, pinned to the `permitted` labels. |

A declared source may also carry `score_quantile: {quantile, scores}`, which
restricts its constraints to the items in the upper `quantile` of the declared
per-item `scores`;
[`synthetic-pinning-quantile.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-pinning-quantile.yaml)
declares one.

## Declare the policy that selects constraints

`constraint_policy` holds one `ConstraintPolicySpec` per domain, `syntactic`
and `semantic`, each with four fields:

| Field | What it says |
| --- | --- |
| `ambiguity` | How a form observed under several labels resolves: `union` (default), `majority`, `weighted` or `per-occurrence`. |
| `selector` | Where in the corpus selected forms sit. The default is a rank band over every form. |
| `coverage` | The share, in `[0, 1]`, of the selected forms or occurrences that receive a constraint. The default is `0.0`, so a source with no declared policy emits nothing. |
| `seed` | The seed the covered share is drawn under. |

The packaged fragments under `constraint_policy/syntactic/` are `all`
(coverage `1.0`), `none` (`0.0`), `partial` (`0.5`, seeded) and
`per-occurrence`; `constraint_policy/semantic/` holds `all` and `none`.

Six selectors are registered:

| `kind` | Selects | Fields |
| --- | --- | --- |
| `rank` | A band of frequency ranks, as shares of the ranked forms. | `first` (default `0.0`), `last` (default `1.0`). |
| `frequency` | A band of relative frequencies, as shares of the token mass. | `at_least`, `at_most`. |
| `count` | An inclusive band of absolute corpus counts. | `at_least` (default `1`), `at_most` (default `null`). |
| `quantile` | The most frequent forms covering a share of a population. | `share`, `over` (default `types`). |
| `category-class` | Forms observed under at least one named category. | `categories`. |
| `category-role` | Forms under the categories at the closed or open end of the class size order. | `role`, `share`. |

Every selector but `count` is read off the corpus as a share, so one
declaration selects the same part of the distribution whatever size of corpus
a grammar yields. The syntactic anchoring campaign in `experiments/s5/` crosses
rank, token mass quantile and closed- or open-class policies at three budgets,
for instance `rank-50.yaml`:

```yaml
ambiguity: union
selector: {kind: rank, first: 0.0, last: 0.5}
coverage: 1.0
seed: 0
```

and `experiments/l5/` does the same for logical forms with a
`gold-logical-form` source and `category-class` selectors such as
`{kind: category-class, categories: [constant, predicate]}`.

A policy value outside its range is refused at composition:

```python
refusal(lexical, ("data.datasets.train.constraint_policy.syntactic.coverage=1.5",))
```

```text
ConfigValueError | data.datasets.train.constraint_policy.syntactic.coverage was refused: a constraint policy's coverage is outside [0, 1]: 1.5
```

The parser trainer refuses `ambiguity: weighted` with "configured pinning does
not admit weighted labels".

## Declare how a clamp applies

Every source takes a `clamp` declaration. Its defaults pin a constituent hard
and stop the run on a clamp no analysis satisfies:

| Field | Default | What it says |
| --- | --- | --- |
| `mode` | `hard` | `hard` filters out analyses that violate the clamp; `soft` penalizes them, and the pinning term's gradient is scaled by `penalty`. |
| `penalty` | `null` | Required, positive and finite for a soft clamp; absent for a hard one. |
| `scope` | `constituent` | Whether the clamp constrains the span as a `constituent` or each `token` in it. |
| `matching` | `null` | The relation under which a label satisfies a target, or `null` for the grammar's own matching. |
| `meaning_matching` | `{kind: identity}` | The same, for logical forms. |
| `require_constituent` | `true` | `true` selects the target route; `false` selects the negative bracket route. |
| `negate` | `false` | Require the span *not* to carry a permitted label. |
| `on_unsatisfiable` | `{kind: error}` | `error`, `drop_clamp` or `skip_item`, from the `lofi.unsatisfiable_policy` group. |

[`synthetic-pinning-soft.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-pinning-soft.yaml)
declares `{mode: soft, penalty: 0.25}`, and
[`synthetic-pinning-inverted.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-pinning-inverted.yaml)
selects `declared-inverted`, whose clamp is `{negate: true}`: occurrences of
`word-0` are pushed away from `S`.

## Choose the target route

With `require_constituent: true`, each constraint names a span and a target
set, and the loss scores how much mass a readout gives that set. Two
objectives consume the target route:

- `{kind: pinning, readout: R}` takes the mean negative log of the mass that
  readout `R` publishes for each constraint's targets, or of its complement for
  a negated constraint. `R` must declare `targets`; the packaged `types`
  readout does, with `targets: grammar-goal`.
- `{kind: vector-anchor, readout: R}` pulls each constrained chart cell toward
  the term encoder's vectors for its targets in `R`'s space. `distance` is
  `cosine` or `squared-euclidean`, `aggregate` combines distances to several
  permitted targets as `nearest` or `mean`, and `margin` is the distance a
  negated constraint must keep.

[`experiments/l5/l5-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l5/l5-run.yaml)
anchors logical forms: its source is `gold-logical-form`, and its third loss
term is `{kind: vector-anchor, readout: forms}`, where `forms` is a term
readout over the `semantics` space of sort `Tm`.
[`experiments/s5/s5-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s5/s5-run.yaml)
pins categories with `{kind: pinning, readout: types}`.

The trainer refuses a constraint objective whose readout represents a
different sort from the one the source's domain labels, and a semantic target
source with no objective naming a logical form readout.

## Choose the negative bracket route

With `require_constituent: false`, each selected span becomes a negative
bracket: the parser is asked not to build it at all. Bracket supervision reads
parser marginals rather than a readout, so its consumer is a `pinning` term
with no `readout`. The example
[`docs/examples/models-training/negative-brackets.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/models-training/negative-brackets.yaml)
excludes every two-word span that starts with `word-2`:

```yaml
defaults:
  - run/synthetic-parser
  - data.datasets.train.constraint_policy.syntactic: per-occurrence
data:
  datasets:
    train:
      constraint_source:
        kind: declared
        policy: syntactic
        pins:
          - {kind: span-pattern, pattern: [word-2, null], permitted: ["S\\NP"]}
        clamp:
          require_constituent: false
          on_unsatisfiable: {kind: drop_clamp}
loss:
  terms:
    - objective: {kind: grammaticality, readout: sentences}
    - objective: {kind: coherence, readout: category-terms, space: categories,
        estimator: truncated, warmup: null}
      weight: 0.1
    - objective: {kind: pinning}
```

The source and the objective both report the negative bracket route, the
objective declares no readout dependency, and the parser trainer's plan
accepts the pair:

```python
from lofi.training_plugins import trainer_plugin

document = Path("docs/examples/models-training/negative-brackets.yaml")
brackets = codec(document)
source = brackets.data.datasets["train"].constraint_source
pinning = brackets.loss.terms[-1].objective
print(
    source.kind, source.constraint_route(), pinning.readout, pinning.constraint_route()
)
print(trainer_plugin(brackets.trainer).plan(brackets.trainer, brackets))
```

```text
declared negative-bracket None negative-bracket
('training:train:validation-split', 'training:train:training-batches', 'training:train:validation-batches')
```

The plan returns the sampler streams the trainer consumes. Mixing routes is
refused there. The packaged constituent pinning document with its clamp
switched to brackets still carries a `pinning` term that reads `types`:

```python
mismatched = codec(
    Path("lofi/configs/run/synthetic-pinning-constituent.yaml"),
    overrides=(
        "data.datasets.train.constraint_source.clamp.require_constituent=false",
    ),
)
try:
    trainer_plugin(mismatched.trainer).plan(mismatched.trainer, mismatched)
except ValueError as refused:
    print(refused)
```

```text
constraint source 'gold-derivation' serves the 'negative-bracket' route, but its consumers request ['target']
```

A `gold-derivation` source is refused on the negative bracket route even when
its consumers ask for that route. It selects every constituent of each item's
gold derivation, the whole sentence's span among them, and a negative bracket
on a span every derivation builds leaves no derivation. The bracket document
with its declared source replaced by a gold derivation one is refused when it
is planned, before any model is built:

```python
gold = codec(
    document,
    overrides=(
        'data.datasets.train.constraint_source={"kind": "gold-derivation", '
        '"policy": "syntactic", "clamp": {"require_constituent": false, '
        '"on_unsatisfiable": {"kind": "drop_clamp"}}}',
    ),
)
try:
    trainer_plugin(gold.trainer).plan(gold.trainer, gold)
except ValueError as refused:
    print(refused)
```

```text
constraint source 'gold-derivation' cannot serve the 'negative-bracket' route: a negative bracket excludes its span, and the source selects every gold constituent, including spans every derivation builds, so no derivation survives the clamp
```

`vector-anchor` serves only the target route, and a token-warm-up step cannot
consume negative brackets either. A negative bracket clamp is hard and is not
negated; both combinations are refused at composition:

```python
refusal(document, ("data.datasets.train.constraint_source.clamp.negate=true",))
refusal(
    document,
    (
        "data.datasets.train.constraint_source.clamp.mode=soft",
        "data.datasets.train.constraint_source.clamp.penalty=0.25",
    ),
)
```

```text
ConfigValueError | data.datasets.train.constraint_source.clamp was refused: a negative-bracket clamp already excludes its span and cannot also be negated
ConfigValueError | data.datasets.train.constraint_source.clamp was refused: a negative-bracket clamp is hard
```

A token-scoped clamp cannot exclude a constituent either.

### Brackets the parser always builds

Some spans are built by every derivation the parser can assign an item, so
excluding them leaves no mass. Before training, the run asks the learner's
category support which spans are mandatory for each sentence length and
applies `on_unsatisfiable` to a bracket on one of them: `error` stops the run,
`drop_clamp` drops that bracket, and `skip_item` emits no constraints for the
item.
The example declares `drop_clamp` because a two-word sentence that starts
with `word-2` would match the pattern, and a whole sentence's span is always
built.
Under the default `error` policy the run stops with an
`UnsatisfiableClampError` whose message reads "the negative bracket on the
span [*i*, *j*) excludes a span that every derivation of the input brackets".

A bracket that passes this check can still land on a span whose constituent
probability is exactly one for a given item. The pinning objective then raises
`ImpossibleNegativeBracketError` instead of returning an infinite loss. The error carries `objective`, `item` and `span`, and its
message reads "*objective* excludes the span [*i*, *j*) of item *n*, which
every derivation the parser assigns that item builds".

## Read what was selected

A point that trains under a constraint source records its selection in
`metrics.json`. Train the example:

```bash
uv run lofi run \
  --config docs/examples/models-training/negative-brackets.yaml \
  --root runs/brackets --seed 11 --experiment-seed 5
```

Then read the record back with [`read_metrics`](../../reference/run.md):

```python
from lofi.run import read_metrics

(directory,) = Path("runs/brackets").glob("runs/[0-9a-f]*")
metrics = read_metrics(directory)
clamp = metrics.selection.clamp
print(clamp.require_constituent, clamp.on_unsatisfiable, clamp.dropped_clamps)

steps = (step for epoch in metrics.fitting.epochs for step in epoch.steps)
first = next(step for step in steps if "loss/pinning" in step.metrics)
print(sorted(key for key in first.metrics if key.startswith("loss/pinning")))
```

```text
False drop_clamp 0
['loss/pinning', 'loss/pinning/gradient_scale', 'loss/pinning/satisfied']
```

`selection` also reports `types_pinned`, `tokens_covered`, `mass_share` (the
share of the corpus the selected constraints cover) and a per-category
breakdown; analyses of the anchoring campaigns read `mass_share` rather than
the nominal budget. `loss/pinning/satisfied` is the mean probability that a
constraint instance is satisfied, and `gradient_scale` is the soft clamp
penalty, or `1.0` for a hard clamp.

[Induction objectives](../../explanation/induction-objectives.md) explains
what partial supervision does and does not tell a learner. The declaration
classes are under [Training declarations](../../reference/configuration/training.md).
