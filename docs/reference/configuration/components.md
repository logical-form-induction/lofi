# Component roots

The [loss family](training.md#losses) contains alternatives such as grammaticality
and [coherence](../../explanation/induction-objectives.md#coherence). Its root class identifies the common configuration role; each
[variant](../../explanation/swappable-assumptions.md#every-choice-is-a-declared-value) specifies the fields of one objective. Adding a variant requires
conforming to that root, as the [plugin guide](../../how-to/extending/plugins.md)
explains.

To select an existing component, use [Model](model.md), [Training](training.md),
or [Datasets](datasets.md). The [API reading guide](../index.md#reading-a-python-api-entry)
explains the class and type notation used for these roots.

This page renders `lofi.spec.components`: the run document's root class and the family roots registered in the [`lofi.families`](../entry-points.md#family-roots) entry point group, against which each component family's variants are declared.

::: lofi.spec.components.base

::: lofi.spec.components.baseline

::: lofi.spec.components.data

::: lofi.spec.components.loss

::: lofi.spec.components.model

::: lofi.spec.components.run

::: lofi.spec.components.trainer
