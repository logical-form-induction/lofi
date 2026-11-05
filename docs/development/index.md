# Development

Environment setup, required checks, and coding conventions are in
[CONTRIBUTING.md](https://github.com/logical-form-induction/lofi/blob/main/CONTRIBUTING.md)
at the repository root. This section describes the repository layout, tests,
and documentation build.

- [Testing](testing.md): the suite, its markers, the opt-in flags, and the four
  jobs that run the same tests under different conditions.
- [Documentation](documentation.md): building these pages, and what strict mode
  catches.
- [Public-release preflight](public-release.md): the six read-only checks that
  must pass before a repository visibility change.

## The tree

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
├── spec/          run specifications, wiring, codecs; the didactic boundary
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

Tests mirror source modules one to one under `tests/unit`. For instance,
`lofi/deduce/_spans.py` has `tests/unit/deduce/test__spans.py`.

Inside a package, `_ports.py` defines protocols, `_values.py` defines value
types, `_errors.py` defines package errors, and `laws.py` exports conformance
properties. The package `__init__` re-exports the public API.

## Where the layer order lives

`.importlinter` at the repository root defines the layer order. Run
`uv run lint-imports` to check it.
