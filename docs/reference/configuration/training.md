# Training declarations

This page renders the classes that validate a run document's `loss`, `trainer`, `optimizer`, `scheduler`, `type_sampler` and `pair_sampler` sections, with the acquisition and judgment-source declarations a trainer reads.

## Losses

::: lofi.spec.losses

The loss package exports the declarations a `loss` section names directly. The
modules below hold the remaining term declarations and their settings.

### Chart-scale penalty

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

The loss package exports the declarations a `loss` section names directly. The
modules below hold the remaining term declarations and their settings.

### Chart-scale penalty

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

::: lofi.spec.type_samplers.distribution

::: lofi.spec.type_samplers.exhaustive

::: lofi.spec.type_samplers.recursive

## Pair samplers

::: lofi.spec.pair_samplers.rule

## Acquisition

::: lofi.spec.acquisition

## Judgment sources

::: lofi.spec.judgment_sources
