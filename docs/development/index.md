# Contributing

Environment setup, required checks, and coding conventions are in
[CONTRIBUTING.md](https://github.com/logical-form-induction/lofi/blob/main/CONTRIBUTING.md)
at the repository root. A source checkout is a local copy of the repository;
its root is the directory containing [`pyproject.toml`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml) and `CONTRIBUTING.md`.
Run this section's shell commands there, after setting up that environment.
Source builds require the Rust toolchain described in the contribution guide,
including when your own change is only to Python or documentation.

- [Documentation](documentation.md): edit a page or an API docstring, preview
  it locally, and check links and rendered references before sharing it.
- [Testing](testing.md): check a behavior change, understand test families,
  and interpret the automated checks on different platforms.
- [Public release preflight](public-release.md): the six read-only checks that
  maintainers run when preparing a repository visibility change.

## The tree

For a change to category syntax, start with [`terms/`](../reference/terms.md#lofi.terms); for a rule, [`judge/`](../reference/judge.md#lofi.judge);
for a run setting, [`spec/`](../reference/spec.md#lofi.spec). The
[layering explanation](../explanation/layering.md) describes their dependencies,
and the directory map locates the remaining roles:

```text
lofi/
├── _types.py      shared aliases and module-sharing declarations
├── _checks.py     field validation, the debug flag, LawViolationError
├── _alignment.py  token-to-row alignment shared by data and neural code
├── terms/         sorts, operators, signatures, terms, the term store
├── judge/         rules, rule sets, normal forms, reports
├── lexicon/       entries and lexical sources
├── deduce/        spans, semirings, charts, the engine, outside scores
├── interpret/     derivation readings and rendering
├── typecheck/     typing of logical forms
├── lf/            the logical-form language
├── formalism/     deduction profiles and region protocols
├── grammars/      cfg, ab, ccg, tlg
├── spec/          run specifications, component wiring, validated configuration
├── registry/      component discovery
├── stats/         p-values, null models, dependence, corrections
├── nn/            encoders, decoders, controllers, parsers, losses, sites
├── data/          corpora and featurizers
├── probe/         probes over a model's published surfaces
├── judgment/      the ordinal response model
├── train/         training steps, checkpoints, optimization
├── eval/          evaluation procedures and result values
└── run/           orchestration, manifests, telemetry sinks, reports
```

Tests mirror source modules one to one under [`tests/unit`](https://github.com/logical-form-induction/lofi/tree/main/tests/unit). For instance,
[`lofi/deduce/_spans.py`](https://github.com/logical-form-induction/lofi/blob/main/lofi/deduce/_spans.py) has [`tests/unit/deduce/test__spans.py`](https://github.com/logical-form-induction/lofi/blob/main/tests/unit/deduce/test__spans.py). [Integration tests](testing.md#markers-and-flags) can exercise several
packages together.

Inside a package, `_ports.py` defines protocols, `_values.py` defines value
types, `_errors.py` defines package errors, and `laws.py` exports conformance
properties. The package `__init__` reexports the [public API](../project/stability.md#supported-api).

A [protocol](../how-to/extending/plugins.md#register-a-component) states what a component must provide to be used in a role.
A value type records data; an error type identifies a kind of failure. A
reexport makes a name available through the supported package import, so
callers can use it without depending on the module where it happens to be
implemented. [Conformance laws](../reference/laws.md) specify behavior shared by implementations
of a protocol.

## Where the layer order lives

[`.importlinter`](https://github.com/logical-form-induction/lofi/blob/main/.importlinter) at the repository root defines the layer order. Run
`uv run lint-imports` to check it.

The [layer order](../explanation/layering.md#the-order) limits which packages may depend on which others. This keeps
changes to neural training, for instance, from making basic symbolic terms
depend on the neural stack. The command reports a violation when an import
crosses a forbidden dependency. Its scope is the package dependency structure;
the [test suites](testing.md) check the imported code's behavior.
