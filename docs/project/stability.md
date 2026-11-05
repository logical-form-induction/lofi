# Stability

LoFI is pre-1.0. Minor releases may change public signatures.

## Supported API

A package's `__init__` exports its supported surface, which is rendered in the
[API reference](../reference/index.md). Direct imports from underscore-prefixed
modules are private; names re-exported by a package `__init__` are public.

## Current implementation

The symbolic core includes terms and the term store, rules and rule sets,
lexical entries and sources, spans, and semirings. It requires only didactic and
panproto.

Above it, `lofi.interpret`, `lofi.formalism`, `lofi.spec`, `lofi.registry`,
`lofi.stats`, `lofi.nn`, `lofi.data`, `lofi.probe`, `lofi.judgment`,
`lofi.train`, `lofi.eval`, `lofi.run`, `lofi.typecheck`, and `lofi.lf` have
public APIs. `lofi.grammars` exports the built-in CCG declarations; grammar
subpackages expose additional declarations.

## Structural constraints

The following constraints are structural:

**The base install remains small.** `pip install lofi` installs only didactic
and panproto as runtime dependencies. Capabilities with additional dependencies
use named extras.

**The symbolic core remains independent.** Modules from `lofi.terms` through
`lofi.grammars` do not import tensor libraries, corpus readers, or the
logical-form packages.

**A probe reaches a model through published surfaces.** `lofi.nn.sites`,
`lofi.nn.ports` and `lofi.nn.policy` define that interface.

## The law suites

Implementations of a LoFI protocol should pass its law suite. Changes to those
laws are treated as compatibility changes.

## Versioning

Until 1.0, a minor version may break a signature. Check the
[law suites](../reference/laws.md) when updating an implementation.
