# Registry

Selecting `kind: grammaticality` in a [loss declaration](configuration/training.md#losses)
requires an installed [plugin](../how-to/extending/plugins.md) registered under that name. The registry resolves
the [tag](../explanation/swappable-assumptions.md#every-choice-is-a-declared-value) to the plugin that supplies its declaration and implementation.
An unknown tag causes resolution to fail.

[Entry point groups](entry-points.md) lists the registrations. The
[plugin guide](../how-to/extending/plugins.md) follows a new variant through
registration, while [family discovery](../explanation/swappable-assumptions.md#how-a-family-is-discovered)
explains how the declarations become available to a document.

This page renders [`lofi.registry`](#lofi.registry): the one resolver every component family
reaches its installed entry points through.

::: lofi.registry
