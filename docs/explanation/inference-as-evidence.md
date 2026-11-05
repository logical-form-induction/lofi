# Inference as evidence for logical forms

This page explains how entailment labels constrain the logical forms a learner
assigns to words. It describes how a label is decided from proof questions,
what a prover, a countermodel search and fuzzy model checking each contribute
to that decision, how a decision becomes a training signal, and what inference
labels cannot determine.

## A label is a pair of proof questions

An inference pair has one premise and one hypothesis, each a string, and a
label from the declared labeling. The `three-way` labeling distinguishes
`entails`, `contradicts` and `neutral`; `rte` distinguishes `entails` from
`non-entails`; `graded` reads an entailment degree on the probability scale.

A learner never sees the logical forms of the pair. It assigns forms to the two
strings through its grammar and lexicon, and the grounding asks two questions
of the declared logic: whether the premise's form proves the hypothesis's (the
positive question) and whether it proves the hypothesis's negation (the
negative question). The two verdicts determine a cell:

| Premise proves hypothesis | Premise proves its negation | Cell |
| --- | --- | --- |
| yes | no | entails |
| no | yes | contradicts |
| no | no | neutral |
| yes | yes | degenerate |

A degenerate cell arises when the premise's form is inconsistent. Which label
each cell supports is the logical-form language's relation-cell semantics, a
label map that the labeling then coarsens into its own vocabulary. The
classical map counts a degenerate cell toward both entailment and
contradiction, as classical consequence does; the strict map lets it support
no label. The map matters, because it decides whether a learner may satisfy an
entailment label by assigning the premise a contradictory form. A
`joint-inconsistency` contradiction adds a second negative question, whether
the hypothesis proves the premise's negation.

During training under the observed-label query, only the questions the gold
label needs are asked: the positive question unless the label is
`contradicts`, the negative one unless it is `entails`.

## Deciding a question

A verdict is `Proved`, `Refuted`, `Graded` with a degree, or `Abstain` with a
cause (`timeout`, `budget-spent`, `out-of-fragment`, `gave-up` or
`search-bound`). An abstention leaves both truth values open. Three kinds of
decision procedure are available.

**Provers.** The `truth-table` prover decides the quantifier-free,
variable-free fragment by enumerating truth assignments, up to a declared atom
cap, and returns a proof or the first falsifying assignment as a countermodel;
equality, quantifiers, abstractions and free variables put a formula outside
its fragment. The `tptp` prover translates to first-order or higher-order TPTP
and calls E, Vampire or Leo-III, mapping the SZS status to a verdict. The `smt`
prover calls Z3 or cvc5 incrementally. A proof service applies per-step call,
wall-clock and CPU budgets, and abstains when one is spent. Only decided
verdicts are memoized.

**Countermodel search.** Before a prover is called, a bounded search over
complete finite structures with declared carrier sizes may find a structure in
which the theory and the premise hold and the hypothesis fails. Such a
countermodel is a certified refutation. The search is exhaustive only when its
`structures` bound reaches the number of structures over the declared carriers;
otherwise it is a partial search, and failing to find a countermodel is not a
proof.

**Fuzzy model checking.** A model-checking scorer evaluates the pair in a set
of structures under a declared [fuzzy logic](../reference/configuration/model.md):
a t-norm (`product`, `goedel` or `lukasiewicz`) with its t-conorm, an
implication (`material` or the t-norm's `residuum`), a quantifier reading
(`extreme`, the minimum and maximum, or folded through the t-norm) and an
equality (`crisp` or with a margin). Its score is the t-norm of premise and
hypothesis summed over structures in which the premise holds above a floor,
divided by the premise's summed value, an estimate of conditional
satisfaction. Structures are given by a knowledge base (an intended model),
sampled, or learned. A model-checking score is not certified: it reports how
well the forms fit the structures, not whether the entailment holds in every
model.

The two routes give evidence of different strength. A prover verdict is a claim
about every model of the theory. A model-checking score is a claim about the
structures consulted, and a learner can satisfy it with forms that happen to
agree on those structures.

## From a decision to a gradient

The grounding composes the verdicts into a distribution over labels for each
pair, and abstained mass goes to a residual rather than to a negative verdict.
The [`inference`](induction-objectives.md#inference) loss then scores that
distribution against the observed label, either as a likelihood or as an
expected payoff. Under the reward signal, a pair whose questions were proved
pays the declared `proved` payoff, a refuted one the `refuted` payoff, and an
abstention pays nothing and is excluded from the mean.

Prover and countermodel verdicts are constants: no gradient flows through a
proof. The gradient reaches the decoder that proposed the forms. When forms are
enumerated, it is pathwise through their probabilities. When forms are sampled
and the scorer is not differentiable, the grounding uses a score-function
surrogate, and a moving-average baseline is then required, as a control
variate that reduces the estimator's variance without changing its
expectation. When a sampled form meets a differentiable scorer, such as a
learned structure source, both paths contribute.

## What inference labels identify

Entailment labels constrain lexical forms only through the relations among the
sentences they are attached to. Two lexicons whose forms differ by a renaming
of non-logical constants that preserves every labeled relation are
indistinguishable to an inference objective. The L1 experiment therefore
reports, beside exact normal-form match, the random-relabeling floor, recovery
relative to the logical vocabulary's fiber, and the fiber's size, so that a
failure to recover a meaning can be told apart from a relabeling the labels
cannot resolve.

The labels also inherit the limits of the logic. A pair the prover abstains on
contributes nothing; a fragment the declared prover cannot decide is invisible
to the objective; and a degenerate cell under the classical map rewards an
inconsistent premise. What an inference-trained lexicon shows is that its
forms reproduce the labeled relations under the declared logic and label map.
It does not show that the forms are the intended meanings. To set up a
grounding, see [Reward with a theorem
prover](../how-to/training/reward-with-a-theorem-prover.md) and [Reward with
model checking](../how-to/training/reward-with-model-checking.md).
