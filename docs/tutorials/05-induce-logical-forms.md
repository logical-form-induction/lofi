# Inducing logical forms

This chapter learns lexical logical forms from inference judgments: it reads a
[run document](../how-to/setup/configuration.md#the-words-this-site-uses) whose corpus is pairs of sentences labeled for entailment, trains
a learner whose only training signal is a theorem prover's verdict on the
forms it decodes, and reads how well the decoded forms match the meanings that
generated the labels. The learner estimates word-to-meaning assignments under a known grammar and logic.

An [inference pair](../explanation/inference-as-evidence.md#a-label-is-a-pair-of-proof-questions) contains a **premise**, which we take as given, and a
**hypothesis**, whose status we ask about. For a consistent premise in this propositional setting, the label is
entailment when the hypothesis follows, contradiction when its negation
follows, and neutral when neither follows. These are questions about the
consequences licensed by the declared logic. Provers with time limits or
undecidable questions can also return an [undecided result](../explanation/inference-as-evidence.md#deciding-a-question), which must be
distinguished from a decided neutral label.

For illustration, `sleep(john)` entails itself and contradicts
`not(sleep(john))`, but alone does not determine `sleep(mary)`.
The learner proposes word meanings, combines them into sentence meanings,
and asks which inference labels those meanings predict. Training adjusts the
learner to give more support to the observed labels. The prover supplies the logical comparison used to score those proposals.

## The document

L1 is the name of the experiment that studies this lexical recovery problem.
The run is the one the L1 experiment sweeps,
[`experiments/l1/l1-run.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/l1-run.yaml).
Its source grammar, [`l1-slash`](https://github.com/logical-form-induction/lofi/blob/main/experiments/l1/data/datasets/train/dataset/grammar/l1-slash.yaml), is chapter 4's grammar restricted to four
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
[`experiments/l1/data/`](https://github.com/logical-form-induction/lofi/tree/main/experiments/l1/data):

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
| `l1-pairs` | A [`synthetic-inference`](../how-to/data/generate-inference-pairs.md#declare-the-draw) dataset: six pairs of premises and hypotheses of two-token sentences, each given a three-way inference label by proving the pair's gold forms under the source logic, with a sixth of the pairs held out for each of `validation`, `dev` and `test`. |
| [`source.language`, `source.interpretation`](../how-to/grammars/declare-an-interpretation.md#declare-the-datasets-interpretation) | The source declarations used to compute labels and reference forms. The learner trains on the pair tokens and labels. Its own language and grounding declarations specify the logic it uses. |
| [`featurizer: pair`](../how-to/data/featurize-a-dataset.md#match-the-featurizer-to-the-trainer) | The learner's inputs are each pair's two token sequences and its label. |
| [`model.grounding: prover-enumerated`](../reference/nn.md#grounding) | Decode candidate forms for the premise and the hypothesis, and score each label by proving the decoded pair under the declared truth table prover. |
| `loss` | One [`inference` objective](../explanation/induction-objectives.md#inference) with `signal: likelihood`: the observed label's probability under the grounded distribution. |
| [`model.spaces.semantics`](../how-to/models/configure-term-encoders-and-decoders.md#declare-a-space) | A second representation space over the term sort `Tm`, with a type-directed decoder that only emits forms of the type the category erases to. |

The [enumerated grounding](../explanation/inference-as-evidence.md#from-a-decision-to-a-gradient) considers weighted candidate forms for each sentence. Candidate pairs can lead to different prover
verdicts; their weights contribute to the predicted label probabilities. The
likelihood objective favors assigning probability to the observed label.
The [type-directed decoder](../how-to/models/configure-term-encoders-and-decoders.md#choose-decoders) restricts which candidates can be considered: a
meaning for a one-place predicate must have the appropriate function type,
as in chapter 4. `Tm` is the sort of logical form terms in the declaration.

The document also declares two evaluators. [`category-side`](../reference/configuration/evaluation.md#lofi.spec.evaluators.lexical_types.LexicalTypeEvaluatorSpec) scores the
categories as chapter 3 did. [`logical-forms`](../reference/configuration/evaluation.md#lofi.spec.evaluators.logical_form.LogicalFormEvaluatorSpec) decodes each test word's form
from the [`forms` readout](../how-to/models/configure-readouts.md#choose-a-route), fits a [many-to-one map](../how-to/grammars/map-between-signatures.md#score-a-learner-grammar-against-a-source-grammar) from the learner's constants
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
[`seed_policy`](../how-to/experiments/sweeps.md#seed-the-points). Each training
step calls the prover on the decoded pairs, which is why the corpus is six
pairs of two-word sentences. Training time depends on the machine and the
proof calls needed for the decoded pairs.

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

The directory-selection line again assumes that the output root contains one
run. If it contains several, select the directory printed for the desired
point. The different `n` values count [different evaluation units](../how-to/evaluation/evaluate-a-run.md#metric-denominators). In the
constructor rows, `n=0` means no constructor positions were scored.

Equality after
[normalization](../explanation/logical-forms-as-types.md#substitution-equality-and-matching) compares the expressions produced by the declared normalizer. Logically
equivalent expressions may retain different normal forms under that procedure.
Labels may also leave some lexical distinctions unidentifiable. If exchanging two otherwise unconstrained individual names
throughout the meanings preserves every observed inference label, labels
alone cannot tell which naming was intended.

The evaluation calls the set of assignments equivalent under its declared
relabelings a [fiber](../explanation/inference-as-evidence.md#what-inference-labels-identify). It asks about recovery relative to that set as well
as recovery with the vocabulary fixed. The allowed relabelings define the
comparison. The [many-to-one alignment](../how-to/grammars/map-between-signatures.md#score-a-learner-grammar-against-a-source-grammar), unlike
chapter 3's one-to-one mapping, may map several learned constants to one
reference constant. The rows expose these distinct comparison choices:

| Row | Question it answers |
|---|---|
| [`logical-form-normal-form`](../reference/eval.md#lofi.eval.NormalFormMetric) | Does the decoded sentence form equal the gold form after normalization, with no alignment? Here, one of two test sentences. |
| `many-to-one-accuracy`, `random-relabeling-floor` | After the `dev`-fitted map from induced constants to gold constants, how many word forms match, and how many would a random relabeling match? Here 0.75 against a floor of 0.594. |
| `fiber-size`, `fiber-floor`, `fiber-relative-recovery` | Inference labels cannot tell apart lexical assignments that differ by a relabeling of the logical vocabulary that preserves every label. The fiber is the set of such assignments, six here, and recovery is scored relative to it and read against its own floor. |
| `fixed-vocabulary-accuracy` | How many word forms match with the logical vocabulary held fixed, with no relabeling allowed. |

The aligned accuracy exceeds its estimated floor, while fiber-relative
recovery is zero. L1 uses `fiber-relative-recovery` as its primary row,
evaluating recovery under the relabelings that preserve the inference evidence.

## Try this

- Train the experiment's other arm with [`--set model/grounding=nli-baseline`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/model/grounding/nli-baseline.yaml),
  which predicts the label from sentence vectors without proving a decoded
  form, and compare its logical form rows with this point's.
- Grow the pair set with `--set data.datasets.train.dataset.size=12` and
  compare `n` in each row and the time the point takes.
- Read [Reward with a theorem prover](../how-to/training/reward-with-a-theorem-prover.md)
  for the packaged `prover-reward` grounding, which trains the same decoder
  from proof outcomes as a reward.

## Next

[Chapter 6](06-compare-grammars.md) crosses a source grammar with candidate
learner grammars in one experiment and asks whether the source grammar wins.
