# Dataset declarations

A sentence can supply [tokens](../../how-to/models/choose-a-token-encoder.md) alone or tokens accompanied by labels and [span
constraints](../../how-to/training/supervise-and-constrain.md). The [dataset](#datasets) declares those observations; a
[featurizer](#featurizers) prepares their fields for a model; a
[constraint source](#constraint-sources) selects [supervision](../../explanation/induction-objectives.md#supervision-and-constraint), and a
[constraint policy](#constraint-policies) specifies how it is used.

For [synthetic data](../../how-to/data/generate-a-synthetic-corpus.md), the grammar and semantics that produce observations belong
to the [source declaration](#spec). The learner can make a different assumption
about the same corpus. [Source and learner declarations](../../explanation/synthetic-designs.md)
explains the comparison this separation permits.

This page renders the classes that validate a run document's `data` section: datasets, featurizers, the synthetic corpus a dataset may draw, and the constraint sources and policies that select partial supervision.

## Datasets

::: lofi.spec.datasets

## Featurizers

::: lofi.spec.featurizers.derivation

::: lofi.spec.featurizers.pair

::: lofi.spec.featurizers.scalar

::: lofi.spec.featurizers.span_constraint

::: lofi.spec.featurizers.unlabeled

## Spec

These declarations configure synthetic sources, including their grammar, pairing, and semantic interpretation. Source semantics used to generate labels need not be supplied to the learner as [supervision](../../explanation/induction-objectives.md#supervision-and-constraint).

::: lofi.data.synthetic.spec.ccgbank

::: lofi.data.synthetic.spec.corpus

::: lofi.data.synthetic.spec.inference

::: lofi.data.synthetic.spec.interpretation

::: lofi.data.synthetic.spec.operators

::: lofi.data.synthetic.spec.pairing

::: lofi.data.synthetic.spec.semantic_lexicon

## Constraint sources

::: lofi.spec.constraint_sources

## Constraint policies

::: lofi.spec.constraint_policy
