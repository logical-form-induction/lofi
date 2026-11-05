# Corrupt strings and their labels

A synthetic corpus's `corruption` block replaces a declared share of its items
with strings the generating grammar does not derive, and its `noise` block
flips the grammaticality label an item is observed with. Together they spell
the three string-supervision regimes: positive-only data, labeled strings, and
certified negatives. This guide declares the corruption operators and
proportion, declares label noise, composes the packaged regime documents, and
reads the reports a draw records.

## Declare a corruption

A corruption names the share of the draw to corrupt, the operators that edit
a grammatical string into a candidate, and how many candidates one item may
take:

```yaml
corruption:
  proportion: {value: 0.25, draw: exact}
  operators:
    entries:
      - operator: {kind: swap, edits: 1}
        weight: 1.0
      - operator: {kind: deletion, edits: 1}
        weight: 1.0
  attempts: 64
```

| Field | What it says |
| --- | --- |
| `proportion.value` | The share of the `size` items to corrupt, between zero and one. |
| `proportion.draw` | `exact` corrupts the nearest whole count; `bernoulli` draws the count. |
| `operators.entries` | Installed operators and their relative draw weights. Each operator kind appears once. |
| `attempts` | How many candidates one corrupted item may take before the corruption is declared exhausted. Defaults to 64. |

Each operator is one entry in the `lofi.corruption` group and applies its edit
`edits` times:

| Operator | Edit |
| --- | --- |
| `swap` | Exchange two adjacent, distinct tokens. |
| `deletion` | Delete one token of a string of two or more. |
| `insertion` | Insert one lexicon form, up to the corpus's longest length. |
| `substitution` | Replace one token by a form that shares no category with it under the grammar's matching relation. |

A candidate is kept only when the parser certifies that the grammar does not
derive it, so every corrupted item is a certified negative. A candidate the
grammar still derives is counted as rejected, and an operator that cannot edit
a string, e.g. a swap over identical tokens, is counted as inapplicable.

The packaged fragments under `data.datasets.train.dataset.corruption` are
`operators: swap-deletion` and `operators: standard` (all four operators), and
`proportion: quarter` and `proportion: half`.

## Read the certified negatives

The packaged run document `lofi/configs/run/synthetic-certified-negatives.yaml`
extends the synthetic parser run with the `standard` operators at a quarter
proportion:

```yaml
defaults:
  - synthetic-parser
  - data.datasets.train.dataset.corruption.operators: standard
  - data.datasets.train.dataset.corruption.proportion: quarter
```

Draw it at 48 items and read the corruption report:

```python
from pathlib import Path

from lofi.run import plan_training
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("lofi/configs/run/synthetic-certified-negatives.yaml")
run = codec(document, overrides=("data.datasets.train.dataset.size=48",))
corpus = plan_training(run).lower_dataset("train").corpus

report = corpus.corruption
print(len(corpus.examples), len(corpus.corrupted), report.achieved_proportion)
print(dict(report.accepted))
print(dict(report.rejected))
print(dict(report.inapplicable))

first = corpus.corrupted[0]
print(first.kind, first.source, "->", first.item.text)
```

```text
36 12 0.25
{'swap': 6, 'deletion': 2, 'insertion': 0, 'substitution': 4}
{'swap': 0, 'deletion': 1, 'insertion': 0, 'substitution': 0}
{'swap': 0, 'deletion': 0, 'insertion': 1, 'substitution': 0}
swap ('word-4', 'word-5') -> word-5 word-4
```

`corpus.examples` holds the 36 grammatical items with their derivations and
`corpus.corrupted` the 12 negatives, each with the operator that made it, the
string it was made from, and a non-derivability certificate. The insertion
here was inapplicable once because the parser document's strings already have
the greatest length the corpus allows.

## Flip labels

`noise` flips the grammaticality label an item is observed with, at one rate
per true label:

```yaml
noise:
  grammatical: {rate: 0.1}
  ungrammatical: {rate: 0.1}
```

`grammatical.rate` is the probability that a derived item is observed as
ungrammatical, and `ungrammatical.rate` the probability that a corrupted item
is observed as grammatical. The packaged fragments are `noise.grammatical:
zero | tenth` and `noise.ungrammatical: zero | tenth`.
`lofi/configs/run/synthetic-labeled-strings.yaml` adds both `tenth` fragments
to the certified-negative document:

```python
labeled = codec(
    Path("lofi/configs/run/synthetic-labeled-strings.yaml"),
    overrides=("data.datasets.train.dataset.size=48",),
)
noise = plan_training(labeled).lower_dataset("train").corpus.noise
print(dict(noise.items), dict(noise.flipped))
print(dict(noise.achieved_rates))
```

```text
{'grammatical': 36, 'ungrammatical': 12} {'grammatical': 5, 'ungrammatical': 1}
{'grammatical': 0.1388888888888889, 'ungrammatical': 0.08333333333333333}
```

The report counts the items under each true label and how many were flipped,
so the achieved rate is read from the draw rather than assumed from the
declaration. `corpus.labels` holds one observed label per item, with its true
label and whether it was flipped.

## Choose a regime

The three regimes differ in what a learner is shown:

| Regime | Declaration | Packaged run document |
| --- | --- | --- |
| Positive-only | no corruption, or a corruption with `noise.ungrammatical.rate: 1.0` so every negative is observed as grammatical | `synthetic-positive-only.yaml` |
| Certified negatives | a corruption, no noise | `synthetic-certified-negatives.yaml` |
| Labeled strings | a corruption and symmetric noise | `synthetic-labeled-strings.yaml` |

An ungrammatical rate of one turns a corruption into contamination of
positive-only data by the corruption share of bad strings:

```python
contaminated = codec(
    document,
    overrides=(
        "data.datasets.train.dataset.size=48",
        "data.datasets.train.dataset.noise.ungrammatical.rate=1.0",
    ),
)
corpus = plan_training(contaminated).lower_dataset("train").corpus
print({label.observed_label for label in corpus.labels})
print(dict(corpus.noise.flipped))
```

```text
{'grammatical'}
{'grammatical': 0, 'ungrammatical': 12}
```

`experiments/s4` crosses each regime's ladder of corruption shares or flip
rates with lexical profiles; its `positive-only.yaml` yokes the corruption
share to the ungrammatical flip rate in exactly this way.

A corrupted string is always labeled ungrammatical, and the `noise` block alone
decides how often that label is observed as grammatical. The negative-data
`regime` belongs to ill-typed logical forms, as
[Generate ill-typed forms](generate-ill-typed-forms.md) shows.

## Refusals

- A proportion outside zero to one, a non-positive `attempts`, an empty
  operator list, or an operator named twice is refused when the document is
  validated.
- A flip rate outside zero to one is refused.
- A `regime` or `noise` field on a corruption is not a declared setting and
  is refused when the document is composed:

```python
from lofi.spec import UnknownSettingError

try:
    codec(
        document,
        overrides=("data.datasets.train.dataset.corruption.regime=label-flips",),
    )
except UnknownSettingError as refused:
    print(str(refused).split(";")[0])
```

```text
data.datasets.train.dataset.corruption.regime is not a declared setting
```

- A corruption whose operators exhaust `attempts` for one item without a
  certified negative stops the draw rather than returning fewer negatives than
  declared.

[Source and learner declarations](../../explanation/synthetic-designs.md)
explains how a corruption share becomes an experimental factor.
