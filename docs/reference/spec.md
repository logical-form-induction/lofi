# Spec

A [`RunSpec`](#lofi.spec.RunSpec) records the declared choices from which a run's
components are built. [Composition](../explanation/documents-are-the-dsl.md#composition)
resolves defaults and overrides before [validation](../explanation/boundaries.md#two-kinds-of-check). An unknown setting can thus
be reported while reading the document, before training begins.

[Configuration tables](configuration/index.md) map document keys to declaration
classes. [Codecs](#lofi.spec.RunSpecCodec) read or write those structured representations; the
[configuration guide](../how-to/setup/configuration.md) shows how to work with
the documents without constructing Python classes.

This page renders [`lofi.spec`](#lofi.spec): run specifications, component wiring and boundary codecs.

::: lofi.spec
