# Install LoFI and its extras

LoFI installs from a source checkout as one base distribution plus an optional
extra for each capability that needs a heavier dependency. It runs on Linux,
macOS and Windows, and requires Python 3.14 or newer and, because the
distribution includes the compiled extension `lofi._native`, a Rust toolchain.

## Install the base distribution

Install [rustup](https://rustup.rs/) first. It reads `rust-toolchain.toml` in
the checkout and installs the pinned compiler the first time the build runs.
Then, from the checkout, install the base distribution into the active
environment:

```bash
uv pip install "lofi @ file://$PWD"
```

[maturin](https://www.maturin.rs/) builds `lofi._native` from the Cargo
workspace under `crates/` against the committed `Cargo.lock`. The base
distribution depends on three pure-Python packages, `didactic`,
`didactic-settings` and `panproto`, and nothing else. It holds the symbolic
core: terms, judges, lexicons, the deduction engine, interpretation, the
formalisms and the grammar packs.

## Add an extra

Each extra names one capability and declares only what that capability
imports directly:

| Extra | Installs | Capability |
| --- | --- | --- |
| `nn` | numpy, torch, transformers | The neural half: token encoders, chart parsers, term encoders and decoders, controllers, readouts, losses and trainers. |
| `corpora` | lairs (with its CoNLL-U reader) | Reading CCGbank, PMB, the NLI corpora, FraCaS and the judgment releases. |
| `judgments` | bead, quivers with its diagnostics extra, scipy, torchdiffeq | Fitting response models over elicited judgments, PSIS-LOO and ELPD model comparison, and the calibration reducer. |
| `reporting` | pyarrow | Writing report tables. |
| `publish` | huggingface-hub, lairs, numpy, torch, transformers | Bundling a trained point for the Hugging Face Hub. |
| `tracking` | wandb, mlflow, trackio | Sending a run's metrics to an experiment tracker. |
| `provers` | z3-solver | The SMT prover arm, which runs the `z3` executable this wheel installs. |
| `laws` | hypothesis | Running the exported law suites against an implementation. |
| `cross-engine` | quivers | The integration test that compares LoFI's deduction engine with quivers'. |

Name the extras in the requirement:

```bash
uv pip install "lofi[nn,corpora] @ file://$PWD"
```

Two extras may share a wheel. `judgments` installs torch because quivers
requires it, without `judgments` naming torch itself. A TPTP prover such as E
or Vampire is not packaged by any extra; the prover declaration names the
executable the site environment provides.

The installed metadata lists every extra the distribution declares:

```bash
uv run python -c "from importlib.metadata import metadata; print(metadata('lofi').get_all('Provides-Extra'))"
```

```text
['corpora', 'cross-engine', 'judgments', 'laws', 'nn', 'provers', 'publish', 'reporting', 'tracking']
```

## What the base install does not import

The symbolic core imports no tensor library, corpus reader or judgment
runtime. Importing it leaves torch, numpy, quivers, lairs and bead unloaded:

```bash
uv run python -c "
import sys
import lofi.deduce, lofi.grammars, lofi.terms, lofi.lexicon, lofi.interpret
print(sorted(m for m in ('torch', 'numpy', 'quivers', 'lairs', 'bead') if m in sys.modules))
"
```

```text
[]
```

Discovering the installed plugins needs no `corpora` extra. A dataset plugin
for CCGbank or a collected release declares its checked route in a light
module and loads its reader, and with it lairs, only when a run lowers that
release, so `lofi.spec.load_plugins()` leaves lairs, quivers and bead
unloaded.

The import contracts in `.importlinter` enforce this boundary, together with
two narrower ones: a corpus reader reaches no neural, deduction or judge
package, and only `lofi.run.hub` imports the Hugging Face Hub client. The law
suites under `lofi.*.laws` import Hypothesis, and no runtime module imports
them.

## Set up a development environment

Development uses [uv](https://docs.astral.sh/uv/) and the committed lockfile:

```bash
uv sync
```

The `dev` dependency group installs the test and lint tools together with the
`nn`, `corpora`, `cross-engine`, `judgments` and `tracking` extras.
[CONTRIBUTING.md](https://github.com/logical-form-induction/lofi/blob/main/CONTRIBUTING.md)
describes how the extension rebuilds after an edit to a Rust source.
