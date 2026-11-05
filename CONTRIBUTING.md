# Contributing to LoFI

LoFI is a research package for inducing categorial grammars and typed logical
forms. Contributions should preserve the separation between symbolic values,
validated configuration, neural computation, and experiment execution.

## Environment

LoFI targets Python 3.14 and uses [uv](https://docs.astral.sh/uv/) for the
environment and lockfile. The interpreter version is pinned in
`.python-version`.

```bash
git clone https://github.com/logical-form-induction/lofi
cd lofi
uv sync
```

`uv.lock` is committed. Update it in the same change as any dependency edit.
Do not use vendored packages, sibling path dependencies, or editable sibling
checkouts.

LoFI has one compiled extension module, `lofi._native`, written in Rust in the
Cargo workspace under `crates/` and built by [maturin](https://www.maturin.rs/),
the project's build backend. LoFI supports Linux, macOS and Windows, and ships
platform wheels for manylinux_2_28 x86_64 and aarch64, macOS arm64 and Windows
x86_64. Every other install builds from source, and a source
build requires a Rust toolchain, on a laptop and on a cluster node alike.
Install [rustup](https://rustup.rs/); it reads `rust-toolchain.toml` and
installs the pinned compiler the first time `cargo` runs in the checkout. There
is no pure-Python mode: without a toolchain, or without `crates/`, `uv sync`
fails with a build error.

`uv sync` builds the extension in the release profile and places it in
`lofi/`. It rebuilds whenever a Rust source, a Cargo manifest, `Cargo.lock` or
`rust-toolchain.toml` changes, so `uv run` after an edit to a `.rs` file runs
the edited code. `Cargo.lock` and `rust-toolchain.toml` are committed. Update
`Cargo.lock` in the same change as any Cargo manifest edit, and bump
`rust-version` in `Cargo.toml` in the same change as the toolchain pin.

The type stub `lofi/_native.pyi` is generated from the compiled module. After
changing the module's Python surface, regenerate it in place with

```bash
uvx maturin@1.15.0 generate-stubs --locked --features pyo3/experimental-inspect --out .
```

and commit the result unedited. Continuous integration regenerates the stub the
same way and fails if it differs from the committed file.

## Checks

Run the standard checks from the repository root:

```bash
uv run ruff format --check lofi tests scripts
uv run ruff check lofi tests scripts
uv run ty check --error-on-warning
uv run lint-imports
uv run pytest
```

For changes to validation or checked execution, also run the optimized and
checked variants:

```bash
uv run python -O -m pytest tests/unit tests/independence
LOFI_CHECK=1 uv run pytest tests --ignore=tests/benchmarks
LOFI_CHECK=0 uv run pytest tests --ignore=tests/benchmarks
```

Changes under `crates/` also run the Rust checks:

```bash
cargo fmt --all --check
cargo clippy --workspace --all-targets --locked -- -D warnings
cargo nextest run --workspace --locked
cargo test --workspace --doc --locked
cargo deny check
cargo shear
```

Continuous integration runs the test suite on Linux, macOS and Windows, builds
and install-tests each platform wheel, and runs the core crate's tests under
Miri, its instruction-count benchmarks under Valgrind, and cargo-mutants on the
lines a pull request changes.

Integration tests are opt-in and must skip cleanly when their prerequisites are
absent. Benchmarks time operations and compare budgets. Ordinary tests assert
behavior outside timed benchmark bodies. An absolute wall-clock budget is
asserted only under `wall_clock_budget.enforced`, because only the fixed
benchmark machine is held to it; hosted runners compare each commit's timings
with its parent's instead. See [Testing](docs/development/testing.md).

## Conventions

These rules are part of the implementation contract:

- Keep development notes, task identifiers, decision history, and planning
  references out of code and tests. Put rationale in the pull request or a
  project issue.
- Use protocols for consumer capabilities. Do not use abstract base classes,
  `NotImplementedError` defaults, or optional methods on a shared base.
- Declare validated boundary data with didactic. Do not construct didactic
  models below their named boundaries or in an inner loop.
- Do not use dataclasses, pydantic, `typing.Any`, or untyped containers. Use a
  tagged union, a PEP 695 generic, a concrete union, a protocol, or the
  project's `JsonValue`.
- Make symbolic values and configuration values frozen. Do not store tensors,
  modules, devices, scores, or mutable collections on them.
- Validate each input once at its named boundary. `assert` is not validation.
  Interior code relies on the invariants established by that boundary.
- Keep the import graph acyclic. Do not use a function-local import to conceal
  a cycle and do not dispatch on component identity with `isinstance`,
  `hasattr`, or computed attribute names.
- Keep imports free of device probes, logging setup, filesystem writes,
  network calls, and other process-wide side effects.
- Derive dimensions, reserved indices, linearization order, and counts from
  their owner. Do not restate them in another component's configuration.
- Reach neural models from probes only through published sites. A probe must
  not walk module attributes, index a private tensor layout, or mutate the
  model it observes.
- Write NumPy-style docstrings. Functions and methods use imperative summary
  lines. Tests under `tests/unit` mirror source modules one to one and exercise
  the behavior they name.
- Use no suppression comments. Fix the diagnostic or document a narrowly
  justified tool configuration change in `pyproject.toml`.
- Never transcribe an empirical result as a numeric literal. Compute it from
  repository data or report that the required data is absent.

### Compiled code

Compiled code is Rust, in the `crates/` workspace, linked into the one
`lofi._native` module. The `lofi-chart` crate is the pure core: it has no Python
dependency and forbids unsafe code. The `lofi-native` crate is the binding: it
converts values and maps errors, contains no algorithm, and has no hand-written
unsafe code. Native term storage owns structural identity, hashing, lineage, and
substitution in `lofi-chart`; `lofi-native` converts validated Python boundary
values and maps typed refusals.

Where Rust implements a capability, Python reaches it through the binding and
keeps no parallel implementation. A Python implementation beside a Rust one
needs a compelling reason stated in the pull request. When a loop moves to
Rust, its Python version is deleted in the same change, with no fallback, flag
or pure-Python mode. Only `lofi.terms` and `lofi.deduce` import `lofi._native`;
all other Python packages reach native capabilities through those package entry
points.

Lint levels live in `Cargo.toml`. Rust source carries no `#[allow]` and no
`#[expect]`. Each Rust module has a `#[cfg(test)] mod tests`, and the core's
public interface has integration tests under `crates/lofi-chart/tests/`.

### Component registration

A swappable implementation declares its explicit tag in its own source module,
has one mirrored unit test, and adds one entry under the appropriate
`[project.entry-points."lofi.<family>"]` group. Do not add a central Python
registry, factory table, import-for-side-effect list, or `isinstance` chain.

## One current interface

LoFI has not been released. It therefore has no compatibility obligations to a
previous public interface. When a representation or API is replaced, remove
the superseded route and migrate every caller, fixture, configuration, entry
point, and export in the same change.

Do not retain aliases, adapters, deprecated spellings, migration paths,
fallback parsers, duplicate registry tags, version branches, compatibility
shims, or old configuration readers. Tests exercise only the single current
treatment. A scientifically required comparison may expose two current
mechanisms, but it must name and implement those mechanisms rather than preserve
one as a legacy route.

## Documentation

Documentation follows [Diataxis](https://diataxis.fr/). Each page is a tutorial,
how-to guide, reference page, or explanation page. Examples in tutorials and
how-to guides must run, and reference pages must describe only APIs that exist.

Build the documentation with:

```bash
uv run --with-requirements docs/requirements.txt mkdocs build --strict
```

## Pull requests and reports

Keep each pull request focused on one concern. Describe what changed, which
boundary or component it affects, and which behavior exposed the defect. Never
include model or assistant attribution in a commit message.

Use the bug, leak, or missing-capability issue template when reporting a
problem. A leak is an implementation detail exposed across a boundary or an
invariant enforced only by convention.
