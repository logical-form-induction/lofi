# Judgment

Two participants may rate the same sentence differently even when given the
same grammatical prediction. A [response model](../explanation/judgment-models.md)
relates that prediction to their observations and can include [participant and
item effects](../explanation/judgment-models.md#a-judgment-is-a-noisy-reading-of-a-latent-scale). [`JudgmentFitSpec`](#lofi.judgment.JudgmentFitSpec) declares the
response model and fitting procedure.

For [ordered ratings](#the-cumulative-family), cumulative probabilities give
the chance of falling at or below each level; subtracting successive values
gives the probability of one level. [Fit a judgment model](../how-to/judgments/fit-a-judgment-model.md)
shows that declaration in use. Grammatical inference rules belong to the
separate [`lofi.judge`](judge.md) package.

This page renders [`lofi.judgment`](#lofi.judgment): the judgment fit declaration
([`JudgmentFitSpec`](#lofi.judgment.JudgmentFitSpec)) and its response, inference, calibration and comparison
blocks, judgment designs and their codecs, the response compiler and the [NUTS](../explanation/judgment-models.md#posterior-objectives-and-calibration),
[variational](../explanation/judgment-models.md#posterior-objectives-and-calibration) and [MAP](../explanation/judgment-models.md#posterior-objectives-and-calibration) fitters, [prior predictive calibration](../explanation/judgment-models.md#posterior-objectives-and-calibration)
([`CalibrationCampaign`](#lofi.judgment.CalibrationCampaign), [`ReplicateRecord`](#lofi.judgment.ReplicateRecord)), [held-out splitting](../explanation/judgment-models.md#what-a-fit-licenses) and prediction,
and the cumulative family and ordinal head.
[Fit a judgment model](../how-to/judgments/fit-a-judgment-model.md) shows them
in use.

[Judgment models](../explanation/judgment-models.md) defines the ordinal,
choice, and bounded response likelihoods, links, identification constraints,
marginalization, and posterior objectives used by these interfaces.

::: lofi.judgment

## The cumulative family

::: lofi.judgment.cumulative

## The cumulative link head

::: lofi.judgment.ordinal

## Synthetic designs

::: lofi.judgment.synthetic_designs
