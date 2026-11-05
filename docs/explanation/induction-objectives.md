# Induction objectives

This page explains what each family of loss terms assumes about the data it is
trained on. It distinguishes objectives that read labels from objectives that
read only strings, says which quantity of the chart each one scores, and states
what a combination of terms does and does not do.

## Objectives read published outputs

A loss term never calls the deduction engine or inspects a model's modules.
Each objective declares the outputs it requires, by name, and reads them from
what the chart and the readouts publish: the spans, cells, splits, constituent
marginals, rule actions, derivation entropy and outside cells of the chart; the
readouts' target masses and decoded terms; and the batch, the grounded
entailment batch and the gold forms where the data supply them. A term whose
required output no component publishes is refused when the model is built.

This matters for what an objective assumes. The grammaticality score, for
instance, is the mass a term readout assigns to the goal categories at the root
cell, decoded from that cell's vector. It is a model quantity read through a
readout, not the inside score of a symbolic chart, and its meaning depends on
the readout and mass strategy the model declares.

## Grammaticality

The `grammaticality` term is a binary cross-entropy over sentence labels: the
mean, over labeled items, of the negative log probability of the observed
label, where the probability of a positive label is the root target mass and
that of a negative label its complement. It reads both positive and negative
labels and skips items whose label is absent.

It assumes the labels say whether a string is a sentence of the language the
goal names. With `noise: 0` it assumes they are exact. A nonzero `noise` treats
each observed label as flipped with that probability, which models symmetric
label noise of a known rate; a rate of 0.5 or more is refused, since a label
flipped at least half the time carries no information about the sentence. The
term does not model asymmetric noise, and it cannot learn from positive-only
data alone: with no negative labels, it is minimized by a model that assigns
the root mass one to every string. Corpora that contain only positive strings
pair it with an objective that constrains the chart in another way.

## Coherence

The `coherence` term reads no labels. For each constituent it compares the
distribution over complete terms decoded at the parent cell with the
distribution obtained by merging the terms decoded at its daughters under the
representation space's declared combinator, and penalizes their divergence
(forward KL by default; cross-entropy and an α-entmax Fenchel–Young loss are
alternatives). Each span's divergence is weighted by its constituent marginal,
split probability and action probability, and a warmup moves the span weights
linearly from one to the marginals.

Coherence assumes that the model's representation of a constituent should
agree with the composition of its parts' representations under the declared
combinator. It is a self-consistency objective, not a statement about the
data, and a model can satisfy it with any internally consistent assignment of
terms to spans. It therefore shapes what grammaticality or reconstruction has
to explain; it does not identify a grammar by itself. The exact estimator
requires that decoding exhaust both paired frontiers, and is refused
otherwise; the truncated estimator requires a residual-mass bound and checks
the residual it leaves against the run's approximation bound (see [Charts and
approximation](charts-and-approximation.md)).

## Reconstruction

Two terms predict the words of the input from the chart.

- `span-reconstruction` decodes the words a span covers, followed by an end
  token, from the representation published for that span, and averages the
  negative log probability per span or per token. Spans are weighted uniformly
  or by their detached constituent marginals.
- `outside-reconstruction` is the leaf objective of DIORA (Drozdov et al.
  2019): it predicts each word from the outside representation of its own leaf
  span, that is, from the context outside the word.

Both read only strings. They assume that the words of a string are predictable
from the representations the chart builds, and they reward a bracketing to the
extent that its constituents compress their yields or their contexts.
Reconstruction rewards structure that is useful for predicting words; whether
that structure coincides with the grammar that generated the data is the
question a comparison has to test, not an assumption the objective makes true.

## Inference

The `inference` term trains the grounded distribution over entailment labels
against observed labels for premise–hypothesis pairs. Under `signal:
likelihood` its cost is the negative log probability of the observed label;
under `signal: reward` it is the negative expected payoff under declared proof
payoffs. A `judgment` response replaces the categorical label with a declared
ordinal instrument and marginalizes over labels. Only pairs the grounding
decided contribute to the mean.

It assumes that the labels are relations between the logical forms the grammar
assigns to the two strings, decided by the declared logic. It does not assume
that the labels identify those logical forms uniquely, and in general they do
not. [Inference as evidence for logical forms](inference-as-evidence.md)
describes what the labels constrain and what they leave open.

## Simplicity

The `simplicity` term is an expected description-length prior. It sums the
expected declared cost of the constructors decoded in every slot and averages
it over the cells in scope. The default scope is the lexicon, the width-one
cells, because an interior span's term follows from the derivation that builds
it from lexical ones, and pricing it would price the same lexical choices
twice. The symbol cost is `uniform`, `log-alphabet` or `per-constructor`.

Simplicity reads no data. It assumes that, among assignments the other terms
cannot distinguish, the one with shorter lexical terms is preferred, and it is
only as meaningful as the declared costs. It makes no claim that the source
grammar is the simplest grammar consistent with the data.

## Supervision and constraint

Several terms read stronger supervision where a corpus supplies it:
`form-supervision` scores the gold logical form at the root, `form-typing`
scores the observed well- or ill-typedness of root forms, `shaped-proof` gives
laddered credit from ill-formed to proved forms against gold forms, and
`pinning` scores target categories or negative brackets declared as
constraints. `vector-anchor`, `chart-scale` and `derivation-entropy` regularize
representations and the parser's entropy rather than encoding a claim about the
data. No term reads gold trees; tree structure is supervised only through
pinning constraints and negative brackets. An experiment that adds any of these
terms changes what the learner is told, and its result is a result about that
supervision regime.

## Combining terms

The `loss` section is a list of terms, each with a weight, an optional linear
weight schedule and a name. The objective is the weighted sum of the terms that
applied to the batch: a term with nothing to score in a batch contributes
nothing, rather than a zero that would dilute the mean. Weights are not
normalized, and a sum of terms is not a joint likelihood unless the terms
happen to be log probabilities of disjoint parts of the data. Two runs with
different weights therefore optimize different objectives, and a weight is a
modeling choice to be crossed or held fixed like any other. To declare terms,
see [Combine loss terms](../how-to/training/combine-losses.md).

## References

- Andrew Drozdov, Patrick Verga, Mohit Yadav, Mohit Iyyer, and Andrew
  McCallum. 2019. Unsupervised latent tree induction with deep inside-outside
  recursive auto-encoders. In *Proceedings of the 2019 Conference of the North
  American Chapter of the Association for Computational Linguistics: Human
  Language Technologies, Volume 1*, pages 1129–1141, Minneapolis.
  [N19-1116](https://aclanthology.org/N19-1116/).
