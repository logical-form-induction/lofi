# Generate inference pairs

A `synthetic-inference` dataset draws premise and hypothesis strings from one
grammar, reads each string's logical form under a source interpretation, and
labels every pair by asking the interpretation's prover whether the premise
entails the hypothesis or its negation. This guide declares the pair draw,
aims it at a label proportion, adds declared label noise, and reads the
reports a draw records.

## Declare the draw

The packaged option `data.datasets.train.dataset: synthetic-inference` is a
complete declaration over a six-word grammar and a truth-table logic. Its
pair-drawing fields are:

```yaml
kind: synthetic-inference
labeling: three-way
retained: {validation: 0.3333333333333333}
size: 9
seed: 7
candidates: 12
attempts: 400
premise: {kind: uniform, min_length: 2, max_length: 2}
hypothesis: {kind: uniform, min_length: 2, max_length: 2}
```

| Field | What it says |
| --- | --- |
| `grammar` | The grammar both sides are drawn from. |
| `source.language`, `source.interpretation` | The language and interpretation used only to draw strings and label their relations. The learner declares its own. |
| `labeling` | The label vocabulary, a `lofi.inference_labeling` entry. |
| `size` | How many labeled pairs to draw. |
| `seed` | A nonnegative integer, or `derived`. |
| `candidates` | How many strings to draw on each side to pair among. |
| `attempts` | How many candidate pairs one accepted pair may take before a complexity target or a label proportion is declared unreachable. |
| `premise`, `hypothesis` | The length distribution of each side, as for a [synthetic corpus](generate-a-synthetic-corpus.md#choose-a-length-distribution). |
| `pairing` | How candidates are paired, a `lofi.pair_strategy` entry. |
| `relations` | The share of trivial pairs and the least complexity of an entailing pair. |

Three labelings are installed. `three-way` keeps entailment, contradiction
and neutral apart. `rte` merges contradiction and neutral into non-entailment.
`graded` uses the same two labels as an entailment degree on the RTE
probability scale.

Two pairing strategies are installed. `independent`, the default, draws the
two sides independently with replacement. `overlap-matched` draws among the
candidate pairs with the greatest predicate overlap; the packaged option
`synthetic-inference-overlap` selects it.

## Draw and read the pairs

`experiments/l4/l4-run.yaml` is a run over a relational source whose
sixteen words mean conjunctions and negations of three one-place predicates,
one two-place predicate and two individuals, drawn at 12 pairs from 128
candidates per side. Draw it under a label proportion and read the first
pair:

```python
from pathlib import Path

from lofi.data.synthetic.spec.inference import PLUGIN
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
policy = ParsePolicy(Retain.FOREST, Reachability.ALL)


def draw(run):
    declaration = run.data.datasets["train"].dataset
    return PLUGIN.lower(PLUGIN.check(declaration), policy).inference


document = Path("experiments/l4/l4-run.yaml")
run = codec(
    document,
    overrides=(
        'data.datasets.train.dataset.proportion={"entails": 0.5, "contradicts": 0.25, "neutral": 0.25, "tolerance": 0.1}',
    ),
)
declaration = run.data.datasets["train"].dataset
print(declaration.labeling, declaration.pairing.kind, declaration.size, declaration.candidates)

inference = draw(run)
pair = inference.pairs[0]
print(pair.premise.text, "|", pair.hypothesis.text, "|", pair.label.value, pair.split)
print({label.value: count for label, count in inference.inference.labels.items()}, inference.inference.rejected)
```

```text
three-way independent 12 128
sleep(mary) & (snore(mary) & walk(mary)) | sleep(mary) | entails validation
{'entails': 6, 'contradicts': 3, 'neutral': 3} 216
```

Each pair holds the two gold forms, the label, the proof certificate the
prover returned for the relation it found, and the split. The inference report
gives the achieved label counts and how many candidate pairs were rejected to
meet the proportion.

## Aim at a label proportion

`proportion` declares the share of the decided pairs each relation is aimed
at, with a tolerance on each achieved share:

```yaml
proportion: {entails: 0.5, contradicts: 0.25, neutral: 0.25, tolerance: 0.1}
```

The three shares sum to one. The packaged options
`synthetic-inference-balanced` (thirds, tolerance 0.01) and
`synthetic-inference-entailment` (entailment only) extend the base option
with a proportion. Without a proportion the labels fall as the drawn pairs
decide them.

`relations` excludes or admits trivial pairs, whose two forms are one term:
`trivial` is the share of trivial pairs, zero by default, and `entailment` is
the least number of subterms an entailing pair's forms must differ by.

## Declare label noise

`label_noise` flips a pair's relation to another at a per-relation rate and
routes each flip through a confusion structure:

```yaml
label_noise:
  rates:
    - {relation: entails, rate: 0.2}
  confusion:
    kind: tabular
    rows:
      - {oracle: entails, neutral: 1.0}
  tolerance: 0.5
```

| Field | What it says |
| --- | --- |
| `rates` | A sparse table of flip rates by oracle relation. A relation with no rate is never flipped. |
| `confusion` | `uniform` spreads a relation's flips evenly over the other two; `tabular` routes them by one row per relation, whose shares sum to one. |
| `tolerance` | How far an achieved confusion entry may fall from the declared one. |

The table above spells the entailment-to-neutral confusion of an annotator who
misses an entailment. Read the achieved confusion matrix, oracle relation by
observed relation:

```python
noisy = codec(
    document,
    overrides=(
        'data.datasets.train.dataset.proportion={"entails": 0.5, "contradicts": 0.25, "neutral": 0.25, "tolerance": 0.1}',
        'data.datasets.train.dataset.label_noise={"rates": [{"relation": "entails", "rate": 0.2}], "confusion": {"kind": "tabular", "rows": [{"oracle": "entails", "neutral": 1.0}]}, "tolerance": 0.5}',
    ),
)
confusion = draw(noisy).noise
for oracle, row in confusion.counts.items():
    print(oracle.value, {observed.value: count for observed, count in row.items()})
```

```text
entails {'entails': 5, 'contradicts': 0, 'neutral': 1}
contradicts {'entails': 0, 'contradicts': 3, 'neutral': 0}
neutral {'entails': 0, 'contradicts': 0, 'neutral': 3}
```

One of the six entailing pairs is observed as neutral, and no other relation
is touched. `experiments/l4/label-noise.yaml` crosses clean labels with two
uniform rates and an asymmetric tabular confusion in exactly this form.

## Refusals

A proportion whose shares do not sum to one is refused when the document is
validated:

```python
from lofi.spec import ConfigValueError

try:
    codec(
        document,
        overrides=(
            'data.datasets.train.dataset.proportion={"entails": 0.5, "contradicts": 0.5, "neutral": 0.5, "tolerance": 0.1}',
        ),
    )
except ConfigValueError as refused:
    print(refused)
```

```text
data.datasets.train.dataset.proportion was refused: a label proportion sums to 1.5 rather than one
```

The declaration is also refused for:

- a `labeling` that names no installed `lofi.inference_labeling` entry;
- a non-positive `candidates` or `attempts`, or a negative `size`;
- a noise rate or a confusion row that names a relation twice, a rate outside
  zero to one, or a confusion row whose shares sum to neither one nor zero;
- a positive `relations.trivial` share together with a positive
  `relations.entailment` bound, which no trivial pair can meet.

A draw that cannot meet a proportion or a complexity target within `attempts`
candidate pairs per accepted pair stops with a `ProportionUnreachableError` or
a sampling-exhausted error rather than returning a different mixture.

[Inference as evidence for logical forms](../../explanation/inference-as-evidence.md)
explains what a label tells a learner about its logical forms.
