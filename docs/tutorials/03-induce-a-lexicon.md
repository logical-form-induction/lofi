# Inducing a lexicon

This chapter trains a chart parser on the strings of chapter 2's grammar and
measures how much of that grammar it recovers: it holds out development and
test sentences, declares two evaluators, trains the point, and reads the rows
the evaluators write. The learner sees only token strings. The lexical
categories and derivations the corpus holds are used only to score it.

## The document

[`docs/examples/tutorials/induce-lexicon.yaml`](../examples/tutorials/induce-lexicon.yaml)
names chapter 2's document in its `defaults` list and changes two things: the
split fractions and corpus size, and a new `evaluators` section.

```yaml
defaults:
  - declared-grammar
data:
  datasets:
    train:
      dataset:
        retained: {validation: 0.2, development: 0.2, test: 0.2}
        size: 40
evaluators:
  brackets:
    kind: bracketing
    interval: {replicates: 50, level: 0.95}
    punctuation: {source: none, forms: []}
    decoder: minimum-risk
    conventions: &conventions
      # ... scoring conventions, shown in full in the document
  types:
    kind: lexical-types
    interval: {replicates: 50, level: 0.95}
    readout: category-terms
    alignment:
      kind: one-to-one
      tie_break: least-gold-label
      fit_on: development
      score_on: test
      random_floor: {replicates: 50, level: 0.95}
      stability: {replicates: 50, level: 0.95}
    conventions: *conventions
```

A bare name in `defaults` is looked up beside the document that lists it, so
`declared-grammar` resolves to chapter 2's file, and this document's values are
composed over it. The grammar, model, loss and trainer are unchanged.

Reading the new sections:

| Line | What it says |
|---|---|
| `retained` | Hold out a fifth of the forty sentences for each of `validation`, `development` and `test`; the remaining two fifths train. Early stopping and release read `validation`. |
| `brackets` | Decode each test sentence's tree by minimum risk over the chart and score its unlabeled spans against the gold derivation's, with a 50-resample bootstrap interval. |
| `types` | Read each token's category off the `category-terms` readout, fit a one-to-one map from induced labels to gold labels on `development`, freeze it, and score on `test`. |
| `random_floor` | Score random relabelings under the same alignment, so an aligned accuracy can be read against what alignment alone achieves. |

The alignment is what makes the second evaluator meaningful. An induced
lexicon names its categories however training left them, and a learner that
found the right partition under other names would score zero against the gold
names. Fitting the map on one split and scoring it on another keeps the map
from being tuned to the sentences it is scored on.

## Training the point

The point trains in under a minute on a laptop CPU:

```bash
uv run lofi run \
  --config docs/examples/tutorials/induce-lexicon.yaml \
  --root runs/lexicon \
  --seed 0 \
  --experiment-seed 3
```

```text
fff495d80177bfea completed runs/lexicon/runs/fff495d80177bfea
```

Running the same command again prints the same line without training the
point a second time, since the directory already holds a completed point under
this experiment seed.

## Reading the rows

Beside the records chapter 1 read, a point with evaluators writes `rows.json`,
one row per evaluator metric. [`read_rows`](../reference/run.md) reads it back:

```python
from pathlib import Path

from lofi.run import read_metrics, read_rows

[directory] = Path("runs/lexicon").glob("runs/[0-9a-f]*")
print(read_metrics(directory).stage)

for row in read_rows(directory).rows:
    value = row["value"]
    shown = None if value is None else round(value, 3)
    print(row["task"], row["split"], row["metric"], row["n"], shown)
```

```text
closed
brackets test unlabeled-bracketing-f1 8 0.889
types test lexical-type-f1 8 0.375
types test primitive-one-to-one-accuracy 15 0.6
types test primitive-random-relabeling-floor 15 0.605
types test constructor-one-to-one-accuracy 0 None
types test constructor-random-relabeling-floor 0 None
```

Each row names the evaluator (`task`), the split it scored, the metric, the
number of units the metric counts (`n`) and the value. A row also carries its
bootstrap interval as `ci_low` and `ci_high`, and the hashes of the
configuration, grammar and scoring conventions it was computed under.

Reading the rows:

| Row | What it shows |
|---|---|
| `unlabeled-bracketing-f1` | Over the eight test sentences, the decoded trees share most of their spans with the gold trees. With at most five tokens and binary trees, many spans are forced, so this number is read against the length distribution, not on its own. |
| `lexical-type-f1` | The induced categories, after alignment, match a quarter of the gold token categories. |
| `primitive-one-to-one-accuracy` and its floor | Aligned accuracy over the primitive positions of the categories is slightly below the random-relabeling floor, so this point has not recovered the primitives beyond what the alignment alone supplies. |
| `constructor-…` | No constructor position was scored (`n` is 0), so these rows hold no value. |

One epoch on sixteen training sentences is not expected to recover a lexicon;
the point of this chapter is the shape of the measurement. Every value in the
table is computed from this run's own records, and the same command on another
machine reproduces it from the same seeds.

## Try this

- Train longer with `--set trainer.epochs=6` and compare the closed rows. The
  point identifier changes with the resolved document, so both points sit side
  by side under `runs/lexicon/runs/`.
- Grow the corpus with `--set data.datasets.train.dataset.size=160` and watch
  `n` grow in every row.
- Set `--set evaluators.brackets.decoder=viterbi` to score the single most
  probable tree instead of the minimum-risk tree, and compare the two
  bracketing rows.

## Next

[Chapter 4](04-declare-logical-forms.md) gives each lexical entry a meaning, so
that every derivation in the corpus also yields a typed logical form.
