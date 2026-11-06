# Install LoFI and its extras

LoFI installs from a source checkout as one base distribution plus an optional
extra for each capability that needs a heavier dependency. It runs on Linux,
macOS and Windows, and requires Python 3.14 or newer and, because the
distribution includes the compiled extension `lofi._native`, a Rust toolchain.
## Install the base distribution

A [source checkout](https://github.com/logical-form-induction/lofi) is a local copy of the repository, including its files and
folders. Run the commands on this page in a terminal opened in that folder.
A [Python environment](https://docs.astral.sh/uv/pip/environments/) holds the packages used by one project; installing into
that environment makes LoFI available to its Python interpreter. Rust compiles
the part of LoFI that runs as native machine code. The build runs during installation even when later work uses only run
documents.

Install [rustup](https://rustup.rs/) first. It reads `rust-toolchain.toml` in
the checkout and installs the pinned compiler the first time the build runs.
Then, from the checkout, install the base distribution into the active
environment:

```bash
uv pip install "lofi @ file://$PWD"
```

[maturin](https://www.maturin.rs/) builds `lofi._native` from the Cargo
workspace under [`crates/`](https://github.com/logical-form-induction/lofi/tree/main/crates/) against the committed `Cargo.lock`. The base
distribution depends on three pure-Python packages, `didactic`,
`didactic-settings` and `panproto`, and nothing else. It holds the symbolic
core: terms, judges, [lexicons](../grammars/declare-a-lexicon.md#list-entries-inline), the deduction engine, interpretation, the
formalisms and the grammar packs.
## Add an extra

An [extra](https://packaging.python.org/en/latest/tutorials/installing-packages/#installing-extras) is a named set of optional dependencies. Choose it by the task:
symbolic terms and grammars use the base installation, whereas training a
neural model needs `nn`. Reading an external corpus may also need `corpora`.
The first tutorial requires `nn` to train its neural model. Installing an extra makes a capability
available; the run document still decides whether to use it.

Each extra names one capability and declares only what that capability
imports directly:

| Extra | Installs | Capability |
| --- | --- | --- |
| `nn` | numpy, torch, transformers | The neural half: [token encoders](../models/choose-a-token-encoder.md#the-four-encoders), chart parsers, [term encoders](../models/configure-term-encoders-and-decoders.md#choose-an-encoder) and decoders, [controllers](../models/configure-a-controller.md#scores-and-action-probabilities), readouts, losses and trainers. |
| `corpora` | lairs (with its CoNLL-U reader) | Reading CCGbank, PMB, the NLI corpora, FraCaS and the judgment releases. |
| `judgments` | bead, quivers with its diagnostics extra, scipy, torchdiffeq | Fitting response models over elicited judgments, PSIS-LOO and ELPD model comparison, and the calibration reducer. |
| `reporting` | pyarrow | Writing report tables. |
| `publish` | huggingface-hub, lairs, numpy, torch, transformers | Bundling a trained point for the Hugging Face Hub. |
| `tracking` | wandb, mlflow, trackio | Sending a run's metrics to an experiment tracker. |
| `provers` | z3-solver | The SMT prover arm, which runs the `z3` executable this wheel installs. |
| `laws` | hypothesis | Running the exported [law suites](../extending/laws.md#check-a-semiring) against an implementation. |
| `cross-engine` | quivers | The integration test that compares LoFI's deduction engine with quivers'. |

Name the extras in the requirement:

```bash
uv pip install "lofi[nn,corpora] @ file://$PWD"
```

Two extras may share a wheel. `judgments` installs torch because quivers
requires it, without `judgments` naming torch itself. Install TPTP provers
such as E or Vampire as external programs. The prover
declaration names the executable provided by the site environment.

The installed metadata lists every extra the distribution declares:

```bash
uv run python -c "from importlib.metadata import metadata; print(metadata('lofi').get_all('Provides-Extra'))"
```

```text
['corpora', 'cross-engine', 'judgments', 'laws', 'nn', 'provers', 'publish', 'reporting', 'tracking']
```
## What the base install does not import

Check the installation of any external program selected by a run. A TPTP
prover's executable must be available when the run requests a proof, even when
its declaration has already validated. The dependency table above lists the
Python packages each extra installs.

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
