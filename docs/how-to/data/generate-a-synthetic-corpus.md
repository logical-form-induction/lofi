# Generate a synthetic corpus

A `synthetic` dataset draws its items from a grammar the dataset declares: it
fixes how many strings to draw, the seed, the distribution of string lengths,
the mass each rule application contributes, and the share of the draw held
out in named splits. This guide declares those controls, draws a corpus from
Python to inspect it, and lists how each control is refused.

The packaged option `data.datasets.train.dataset: synthetic` supplies every
control with a default; a run document selects it and overrides the leaves it
needs. [`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml)
does so:

```yaml
data:
  datasets:
    train:
      dataset:
        retained: {validation: 0.25}
        grammar:
          # signature, rules, lexicon and goal
        size: 4
        seed: 5
        length:
          kind: uniform
          max_length: 3
```

The grammar is declared as the guides under
[Grammars and languages](../grammars/declare-a-signature.md) describe. The
rest of this page is about the fields beside it.

## Draw the corpus

[`plan_training`](../../reference/run.md) checks a composed run without
allocating a model, and the plan's `lower_dataset` draws one dataset role. The
generator certifies each derivation under the policy the run's `parse` section
declares:

```python
from pathlib import Path

from lofi.run import plan_training
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))

document = Path("docs/examples/synthetic-parser.yaml")
dataset = plan_training(codec(document)).lower_dataset("train")
corpus = dataset.corpus

example = corpus.examples[0]
print(len(dataset), corpus.seed, example.item.text, example.item.split)

derivation = example.supervision.derivation
for node_id in derivation.preorder:
    entry = derivation.node(node_id)
    print(entry.span, entry.label, entry.rule)
```

```text
4 5 word-7 word-4 word-1 validation
TokenSpan(0, 3) S application:\
TokenSpan(0, 2) NP application:/
TokenSpan(0, 1) NP/NP None
TokenSpan(1, 2) NP None
TokenSpan(2, 3) S\NP None
```

Each example pairs an item, with its tokens, text and split, with the gold
derivation it was drawn by. A leaf carries no rule; an inner node names the
rule that built it. Equal declarations, seed included, draw equal corpora.

## Set the size, seed and splits

| Field | What it says |
| --- | --- |
| `size` | How many items to draw, corrupted ones included. Zero is allowed. |
| `seed` | A nonnegative integer, or `derived` to take the seed each point of an experiment derives for this path, so that the sweep's seeds resample the data. |
| `retained` | Named splits and the fraction of the draw each holds. The remainder trains. |
| `training_split` | The name the training remainder is assigned, `train` by default. No retained split may share it. |

`retained` is exact: a fraction of `0.1` of 200 items holds 20 of them in that
split. The run document's trainer is checked against the training remainder,
so a size whose remainder does not fill the declared accumulation windows is
refused when the run document is composed.

## Choose a length distribution

`length` is the distribution a string's token count is drawn from. There are
three kinds:

| Kind | Fields | Draws |
| --- | --- | --- |
| `uniform` | `min_length` (default 1), `max_length` | every count between the bounds with equal probability |
| `negative-binomial` | `min_length`, `max_length`, `r`, `p`, `out_of_support` | a negative binomial with `r` successes and success probability `p`, shifted to `min_length` |
| `tabular` | `masses`, a list of `{length, mass}` | each listed count in proportion to its mass |

`out_of_support` says what happens to a negative-binomial draw above
`max_length`: `resample` (the default) draws again, `clip` takes the bound,
and `error` refuses the draw. The packaged `synthetic` option uses a negative
binomial with `r: 4.0` and `p: 0.6` between 2 and 24 tokens.

## Weight the rules

`sampler` says how much local mass each rule application contributes when a
derivation is drawn. `uniform`, the default, gives every application equal
mass. `weighted-rules` gives each compiled rule its own positive weight, keyed
by the rule's compiled name.

The draw below raises the size, changes the seed, draws lengths from a
negative binomial, weights forward application four times backward
application, and holds out two splits:

```python
from collections import Counter

run = codec(
    document,
    overrides=(
        "data.datasets.train.dataset.size=200",
        "data.datasets.train.dataset.seed=11",
        'data.datasets.train.dataset.length={"kind": "negative-binomial", "min_length": 2, "max_length": 8, "r": 2.0, "p": 0.5}',
        'data.datasets.train.dataset.sampler={"kind": "weighted-rules", "weights": {"application:/": 4.0, "application:\\\\": 1.0}}',
        'data.datasets.train.dataset.retained={"validation": 0.1, "test": 0.1}',
    ),
)
corpus = plan_training(run).lower_dataset("train").corpus

print(corpus.length.achieved, corpus.length.resampled)
print(Counter(example.item.split for example in corpus.examples))
print(
    Counter(
        node.rule
        for example in corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if node.rule is not None
    )
)
```

```text
Histogram({2: 44, 3: 60, 4: 32, 5: 30, 6: 19, 7: 8, 8: 7}) 5
Counter({'train': 160, 'test': 20, 'validation': 20})
Counter({'application:/': 372, 'application:\\': 200})
```

The length report gives the achieved histogram and how many draws fell outside
the support and were drawn again. The weights are local factors on rule
applications, not target proportions: the achieved rule counts also depend on
which derivations the lexicon admits at each length.

On the command line the same changes are `--set` overrides, e.g.
`--set data.datasets.train.dataset.size=200`.

## Aim at a structure

Three optional blocks reject drawn strings until a measurement falls inside a
band. Each band is a `{value, tolerance}` target, and each block declares
`attempts`, the number of draws one accepted string may take before the
target is declared unreachable.

| Block | Targets |
| --- | --- |
| `ambiguity` | `lexical`, the mean number of lexical categories per token, and `derivations`, the number of root derivations of a string. |
| `shape` | `depth`, `right_branching`, `slash_balance` and `arity` of each accepted derivation, as a `default` target or one target per split under `splits`. |
| `scope` | The logical-form readings per string, described in [Control semantic distributions](control-semantic-distributions.md). |

A `shape` schedule without a `default` needs a target for every declared
split, `train` included.

## Refusals

A weighting that does not name every compiled rule exactly once is refused
when the run is planned, before the corpus is drawn, with the rules it misses
or does not know:

```python
from lofi.spec import DeclarationError

partial = codec(
    document,
    overrides=(
        'data.datasets.train.dataset.sampler={"kind": "weighted-rules", "weights": {"application:/": 4.0}}',
    ),
)
try:
    plan_training(partial)
except DeclarationError as refused:
    print(refused)
```

```text
data.datasets.train.dataset was refused: rule weights do not match the compiled grammar: missing 'application:\\'
```

The other controls are refused when the document is validated:

- `min_length` must be positive and at most `max_length`; `r` must be
  positive and `p` strictly between zero and one.
- A tabular distribution may not repeat a length and needs some positive mass.
- A weight must be positive and finite.
- `seed` must be a nonnegative integer or `derived`.
- A `shape` schedule that names an undeclared split, or leaves a split without
  a target, is refused, as is a shape target that names no measurement.

[Source and learner declarations](../../explanation/synthetic-designs.md)
explains why the corpus carries its own grammar and how each control becomes
a factor of an experiment.
