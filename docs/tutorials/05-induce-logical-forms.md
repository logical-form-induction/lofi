# Inducing logical forms

This chapter learns lexical logical forms from inference judgments: it reads a
run document whose corpus is pairs of sentences labeled for entailment, trains
a learner whose only training signal is a theorem prover's verdict on the
forms it decodes, and reads how well the decoded forms match the meanings that
generated the labels. The grammar and the logic are known to the learner; the
word-to-meaning assignment is not.

## The document

The run is the one the L1 experiment sweeps,
[`experiments/l1/l1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/l1-run.yaml).
Its source grammar, `l1-slash`, is chapter 4's grammar restricted to four
words, and its interpretation is chapter 4's lexical interpretation with the
same meanings for those words:

```yaml
lexicon:
  kind: inline
  entries:
    - {form: john, category: NP, meaning: john}
    - {form: mary, category: NP, meaning: mary}
    - {form: sleeps, category: "S\\NP", meaning: "\\x. sleep(x)"}
    - {form: sees, category: "S\\NP/NP", meaning: "\\y. \\x. see(x)(y)"}
```

The document assembles the rest from fragments beside it under
`experiments/l1/data/`:

```yaml
defaults:
  - data.datasets.train.dataset: l1-pairs
  - data.datasets.train.dataset.grammar: l1-slash
  - data.datasets.train.dataset.source.language: l1-propositional
  - data.datasets.train.dataset.source.interpretation: l1-first-order
  - data.datasets.train.featurizer: pair
  - model.grounding: prover-enumerated
```

Reading the parts that differ from chapter 3:

| Part | What it says |
|---|---|
| `l1-pairs` | A `synthetic-inference` dataset: six premise-hypothesis pairs of two-token sentences, each given a three-way inference label by proving the pair's gold forms under the source logic, with a sixth of the pairs held out for each of `validation`, `dev` and `test`. |
| `source.language`, `source.interpretation` | The logic and the meanings the labels are computed from. They sit under `source`, so the learner never receives them. |
| `featurizer: pair` | The learner reads each pair's two token sequences and its label, not either sentence's form. |
| `model.grounding: prover-enumerated` | Decode candidate forms for the premise and the hypothesis, and score each label by proving the decoded pair under the declared truth-table prover. |
| `loss` | One `inference` objective with `signal: likelihood`: the observed label's probability under the grounded distribution. |
| `model.spaces.semantics` | A second representation space over the term sort `Tm`, with a type-directed decoder that only emits forms of the type the category erases to. |

The document also declares two evaluators. `category-side` scores the
categories as chapter 3 did. `logical-forms` decodes each test word's form
from the `forms` readout, fits a many-to-one map from the learner's constants
to the gold constants on `dev`, freezes it, and scores on `test`.

## Training the point

```bash
uv run lofi run \
  --config experiments/l1/l1-run.yaml \
  --root runs/forms \
  --seed 0 \
  --experiment-seed 108
```

```text
991e01b8d83d0ea2 completed runs/forms/runs/991e01b8d83d0ea2
```

`108` is the root seed the L1 reduced experiment declares in its
`seed_policy`. Each training
step calls the prover on the decoded pairs, which is why the corpus is six
pairs of two-word sentences: at this size the point trains in under a minute.

## Reading the rows

The rows are read exactly as in chapter 3:

```python
from pathlib import Path

from lofi.run import read_metrics, read_rows

[directory] = Path("runs/forms").glob("runs/[0-9a-f]*")
print(read_metrics(directory).stage)

for row in read_rows(directory).rows:
    value = row["value"]
    shown = None if value is None else round(value, 3)
    print(row["task"], row["split"], row["metric"], row["n"], shown)
```

```text
closed
category-side test supertag-accuracy 4 0.0
category-side test dependency:labeled-directed:zero:micro-f-score 2 0.0
category-side test lexical-type-f1 2 0.667
category-side test primitive-one-to-one-accuracy 2 1.0
category-side test primitive-random-relabeling-floor 2 1.0
category-side test constructor-one-to-one-accuracy 0 None
category-side test constructor-random-relabeling-floor 0 None
logical-forms test logical-form-normal-form 2 0.5
logical-forms test many-to-one-accuracy 4 0.75
logical-forms test random-relabeling-floor 4 0.594
logical-forms test fiber-relative-recovery 3 0.0
logical-forms test fixed-vocabulary-accuracy 3 0.0
logical-forms test fiber-floor 3 0.333
logical-forms test fiber-size 3 6.0
```

The logical-form rows separate three questions:

| Row | Question it answers |
|---|---|
| `logical-form-normal-form` | Does the decoded sentence form equal the gold form after normalization, with no alignment? Here, one of two test sentences. |
| `many-to-one-accuracy`, `random-relabeling-floor` | After the `dev`-fitted map from induced constants to gold constants, how many word forms match, and how many would a random relabeling match? Here 0.75 against a floor of 0.594. |
| `fiber-size`, `fiber-floor`, `fiber-relative-recovery` | Inference labels cannot tell apart lexical assignments that differ by a relabeling of the logical vocabulary that preserves every label. The fiber is the set of such assignments, six here, and recovery is scored relative to it and read against its own floor. |
| `fixed-vocabulary-accuracy` | How many word forms match with the logical vocabulary held fixed, with no relabeling allowed. |

The aligned accuracy sits above its floor while fiber-relative recovery is
zero, so in this point the matches are ones a relabeling could also produce.
This is why the L1 experiment declares `fiber-relative-recovery` its primary
row: it separates a failure to recover a meaning from a relabeling that
inference judgments cannot identify.

## Try this

- Train the experiment's other arm with `--set model/grounding=nli-baseline`,
  which predicts the label from sentence vectors without proving a decoded
  form, and compare its logical-form rows with this point's.
- Grow the pair set with `--set data.datasets.train.dataset.size=12` and
  compare `n` in each row and the time the point takes.
- Read [Reward with a theorem prover](../how-to/training/reward-with-a-theorem-prover.md)
  for the packaged `prover-reward` grounding, which trains the same decoder
  from proof outcomes as a reward.

## Next

[Chapter 6](06-compare-grammars.md) crosses a source grammar with candidate
learner grammars in one experiment and asks whether the source grammar wins.
