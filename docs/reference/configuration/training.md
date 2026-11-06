# Training declarations

Doubling a [loss weight](../../explanation/induction-objectives.md#combining-terms)
doubles that term's contribution to the objective. Doubling an
[optimizer's learning rate](#optimizers) changes the scale of the parameter
update, with an effect that depends on the optimizer. A [scheduler](#schedulers) changes that
rate over training, while the [trainer](#trainers) organizes the steps.

[Induction objectives](../../explanation/induction-objectives.md) specifies
what each [cost](../../explanation/induction-objectives.md) measures and how it is averaged. [Acquisition](../../how-to/training/acquire-labels-actively.md)
and [judgment sources](#judgment-sources) separately determine how active
learning obtains further responses.

This page renders the classes that validate a run document's `loss`, `trainer`, `optimizer`, `scheduler`, `type_sampler` and `pair_sampler` sections, with the acquisition and judgment source declarations a trainer reads.

## Losses

::: lofi.spec.losses

The loss package exports the declarations a `loss` section names directly. The
modules below hold the remaining term declarations and their settings.

### Chart scale penalty

::: lofi.spec.losses.chart_scale

### Coherence and its divergences

::: lofi.spec.losses.coherence

### Grammaticality

::: lofi.spec.losses.grammaticality

### Outside reconstruction

::: lofi.spec.losses.outside_reconstruction

### Pinning

::: lofi.spec.losses.pinning

### Simplicity and symbol costs

::: lofi.spec.losses.simplicity

### Span reconstruction

::: lofi.spec.losses.span_reconstruction

### Vector anchor

::: lofi.spec.losses.vector_anchor

## Trainers

::: lofi.spec.trainers

## Optimizers

::: lofi.spec.optimizers.adam

::: lofi.spec.optimizers.adamw

::: lofi.spec.optimizers.sgd

## Schedulers

::: lofi.spec.schedulers.constant

::: lofi.spec.schedulers.cosine

::: lofi.spec.schedulers.linear

## Type samplers

A [recursive sampler](#lofi.spec.type_samplers.recursive.RecursiveTypeSamplerSpec) can vary a term's depth and branching before the term is
used for training. The distribution declarations below specify feasible child
depths and weights by [constructor arity](../../explanation/term-layout-capabilities.md#a-tree-and-its-representation); [exhaustive sampling](#lofi.spec.type_samplers.exhaustive.ExhaustiveTypeSamplerSpec) instead enumerates
its declared possibilities. [Term encoders and decoders](../../how-to/models/configure-term-encoders-and-decoders.md)
explains the representations learned for those terms. Here “type” refers to
the symbolic terms supplied to the learner.

::: lofi.spec.type_samplers.distribution

::: lofi.spec.type_samplers.exhaustive

::: lofi.spec.type_samplers.recursive

## Pair samplers

::: lofi.spec.pair_samplers.rule

## Acquisition

::: lofi.spec.acquisition

## Judgment sources

::: lofi.spec.judgment_sources
