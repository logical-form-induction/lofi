# Inducing a lexicon

This chapter trains a [chart parser](../how-to/models/configure-a-chart-parser.md#choose-the-chart-family) on the strings of chapter 2's grammar and
measures how much of that grammar it recovers: it holds out development and
test sentences, declares two evaluators, trains the point, and reads the rows
the evaluators write. The learner sees only token strings. The lexical
categories and derivations the corpus holds are used only to score it.

[Inducing a lexicon](../explanation/induction-objectives.md#grammaticality) means inferring the association between words and
categories from the evidence available to the learner. The source grammar
already knows that *mary* is an `NP` and *sleeps* is `S\NP`. The learner is
not given those assignments as training targets. It must find assignments
that support the observed combinations of words. Keeping the source
derivations lets us later ask whether it found the intended structure.

## The document

[`docs/examples/tutorials/induce-lexicon.yaml`](../examples/tutorials/induce-lexicon.yaml)
names chapter 2's document in its [`defaults`](../how-to/setup/configuration.md#select-fragments-with-defaults) list and changes two things: the
split fractions and corpus size, and a new [`evaluators`](../reference/spec.md#lofi.spec.EvaluatorSpec) section.

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

The YAML spelling `&conventions` gives a reusable name to the following
block; `*conventions` uses that same block elsewhere. The two evaluators thus
share [scoring conventions](../how-to/evaluation/evaluate-a-run.md#declare-the-conventions) without repeating their settings.

The four [data splits](../how-to/evaluation/evaluate-a-run.md#which-split-is-scored) serve different purposes. **Training** examples change
the model's parameters. **Validation** examples help select a [checkpoint](../reference/checkpoints.md#best-and-resumable-checkpoints).
**Development** examples establish how learned category names correspond to
reference names. **Test** examples measure recovery after those choices have
been made. This separation prevents the final score from benefiting from a
label mapping chosen specifically to improve that score.

Reading the new sections:

| Line | What it says |
|---|---|
| `retained` | Hold out a fifth of the forty sentences for each of `validation`, `development` and `test`; the remaining two fifths train. Early stopping and release read `validation`. |
| `brackets` | Decode each test sentence's tree by [minimum risk](../how-to/evaluation/decode-derivations.md#choose-the-decision-rule) over the chart and score its unlabeled spans against the gold derivation's, with a 50-resample bootstrap interval. |
| `types` | Read each token's category off the [`category-terms` readout](../how-to/models/configure-readouts.md#choose-a-route), fit a one-to-one map from induced labels to gold labels on `development`, freeze it, and score on `test`. |
| `random_floor` | Score random relabelings under the same alignment, so an aligned accuracy can be read against what alignment alone achieves. |

The [alignment](../how-to/grammars/map-between-signatures.md#score-a-learner-grammar-against-a-source-grammar) is what makes the second evaluator meaningful. An induced
[lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) names its categories however training left them, and a learner that
found the right partition under other names would score zero against the gold
names. Fitting the map on one split and scoring it on another keeps the map
from being tuned to the sentences it is scored on.

For instance, a learner might consistently call noun phrases `A` and nouns
`B`. A correspondence from `A` to `NP` and `B` to `N` recognizes that it found
the distinction despite different names. A one-to-one correspondence cannot
map both `A` and `B` to `NP`. LoFI fits correspondences for the primitive
symbols and constructors of category terms, then evaluates the resulting
category predictions under the declared conventions.

[Bracketing](../how-to/evaluation/decode-derivations.md#what-each-row-reports) asks a different question: which groups of words belong together?
For `[[the dog] [chases mary]]`, the spans for *the dog* and *chases mary*
are constituents even if we ignore their category labels. **Precision** asks
what fraction of predicted constituents are in the reference; **recall** asks
what fraction of reference constituents were predicted. F1 balances these
two quantities. In a hypothetical example, two shared constituents out
of three predicted and four reference constituents give precision $2/3$,
recall $1/2$, and F1 $2\times2/(3+4)=4/7$.

The [bootstrap interval](../how-to/evaluation/evaluate-a-run.md#set-the-interval) estimates sampling uncertainty by repeatedly
resampling the evaluated sentences and recomputing the metric. With eight
test sentences and fifty resamples, this configuration demonstrates the
procedure; it is too small for strong recovery claims.

## Training the point

Train the small example point on the CPU; its duration depends on the machine:

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

Beside the records chapter 1 read, a point with evaluators writes [`rows.json`](../how-to/evaluation/evaluate-a-run.md#read-the-rows),
one row per evaluator metric. [`read_rows`](../reference/run.md#lofi.run.read_rows) reads it back:

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
[number of units the metric counts](../how-to/evaluation/evaluate-a-run.md#metric-denominators) (`n`) and the value. A row also carries its
[bootstrap interval](../how-to/evaluation/evaluate-a-run.md#set-the-interval) as `ci_low` and `ci_high`, and the hashes of the
configuration, grammar and [scoring conventions](../how-to/evaluation/evaluate-a-run.md#declare-the-conventions) it was computed under.

The `[directory] = ...` line assumes that the output root contains exactly one
run, as it does after the first command above. After trying several settings,
select the particular directory printed by the run you want to inspect.
The `n` column counts the units for that metric: eight sentences for the
sentence-resampled rows, fifteen paired primitive positions for their
alignment, and no constructor positions here. `None` means there is no value
to report; it is different from an accuracy of zero on a nonempty sample.

Reading the rows:

| Row | What it shows |
|---|---|
| [`unlabeled-bracketing-f1`](../how-to/evaluation/decode-derivations.md#what-each-row-reports) | Over the eight test sentences, the decoded trees share most of their spans with the gold trees. With at most five tokens and binary trees, many spans are forced; interpret this score relative to the length distribution. |
| [`lexical-type-f1`](../reference/eval.md#lofi.eval.LexicalTypeRecovery) | The aligned lexical category predictions receive an F1 of 0.375 under the declared scoring conventions. This measure combines precision and recall for category recovery. |
| `primitive-one-to-one-accuracy` and its floor | Aligned accuracy over the primitive positions of the categories is slightly below the random relabeling floor, so this point has not recovered the primitives beyond what the alignment alone supplies. |
| `constructor-…` | No constructor position was scored (`n` is 0), so these rows hold no value. |

The table records [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) recovery after one epoch on sixteen training
sentences. Such a small run demonstrates the measurement procedure. The seeds
specify its randomized choices; floating-point results can also depend on the
machine and dependency versions. Compare measurements using your own run's records.

## Try this

- Train longer with `--set trainer.epochs=6` and compare the closed rows. The
  [point identifier](../explanation/documents-are-the-dsl.md#point-identity) changes with the resolved document, so both points sit side
  by side under `runs/lexicon/runs/`.
- Grow the corpus with `--set data.datasets.train.dataset.size=160` and watch
  `n` grow in every row.
- Set `--set evaluators.brackets.decoder=viterbi` to score the single most
  probable tree instead of the minimum risk tree, and compare the two
  bracketing rows.

The [minimum risk decoder](../how-to/evaluation/decode-derivations.md#choose-the-decision-rule) selects a tree to reduce expected evaluation loss
under the model's distribution; Viterbi selects the single most probable
tree. Those decisions need not coincide when several plausible trees share
many constituents.

## Next

[Chapter 4](04-declare-logical-forms.md) gives each lexical entry a meaning, so
that every derivation in the corpus also yields a typed logical form.
