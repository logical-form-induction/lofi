# Your first run

This chapter trains one LoFI point from start to finish. A [point](../explanation/documents-are-the-dsl.md#point-identity) is one
training and evaluation run with a particular set of settings and random seed.
The chapter installs the
package, reads a complete [run document](../how-to/setup/configuration.md#the-words-this-site-uses), composes that document into a validated
specification, trains the point on a small synthetic corpus, and reads the
records the run leaves behind. Every later chapter starts from the same
document and changes one section of it.

A [run document](../how-to/setup/configuration.md) is a settings file: it says which data to use, what to
learn, and how to check the result. It can reuse existing settings through a
[`defaults`](../how-to/setup/configuration.md#select-fragments-with-defaults) list and replace individual settings with [overrides](../how-to/setup/configuration.md#override-leaves-and-groups). Combining
these sources is called [composition](../explanation/documents-are-the-dsl.md#composition). LoFI then [validates the assembled document](../explanation/boundaries.md#where-declared-values-enter) before constructing the model. A misspelled setting or a value of the
wrong type is refused at this stage, when its source is still easy to locate.

## Prerequisites

Follow [Install LoFI and its extras](../how-to/setup/installing.md) to obtain a
source checkout and the required tools, including the Rust toolchain used to
build the compiled extension. A checkout is a local copy of the repository.
Open a terminal in its root directory, the directory containing
[`pyproject.toml`](https://github.com/logical-form-induction/lofi/blob/main/pyproject.toml), and build the project environment:

```bash
uv sync
```

`uv sync` installs the project's dependencies into its environment. In the
commands below, `uv run` runs a program in that environment, so it can find
LoFI and its dependencies. The environment includes the [`nn` extra](../how-to/setup/installing.md#add-an-extra), which installs PyTorch. A quick check
that the package and its configuration library are in place:

```bash
uv run python -c "from lofi.spec import load_plugins; print(len(load_plugins()) > 0)"
```

```text
True
```

[`load_plugins`](../reference/spec.md#lofi.spec.load_plugins) loads every installed component family and returns the names it
loaded. If that prints `True` without an error, you're set.

Shell blocks, like the two above, go in the terminal. For blocks marked
`python`, start an interactive session with `uv run python` and enter the code
there; do not type the displayed output. Keep that session open between Python
blocks in this chapter so that names such as `codec` remain available. To run
a shell command again, exit Python with `exit()` or use another terminal in the
same directory.

## The run document

The document this chapter trains is
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml).
Open it beside this page. It generates four short sentences, trains a small
model on their strings, checks held-out structure, and saves the result. These
settings use YAML: indentation groups settings, `name: value` assigns a value,
and square brackets enclose a list. Run the checked-in file; the excerpts below show selected entries.

The document has eight top-level parts. A [token encoder](../how-to/models/choose-a-token-encoder.md) turns words into
numbers a model can learn with; a [term decoder](../reference/nn.md#decoders)
predicts symbolic expressions from numerical representations. A
[loss](../explanation/induction-objectives.md#what-training-is-asked-to-improve)
measures the discrepancy that training tries to reduce:

| Section | What it says |
|---|---|
| [`defaults`](../how-to/setup/configuration.md#select-fragments-with-defaults) | Select one packaged fragment per slot: an [inside-outside chart parser](../how-to/models/configure-a-chart-parser.md), a [static token encoder](../how-to/models/choose-a-token-encoder.md#read-vectors-from-a-file), [recurrent neural network (RNN) term encoders and decoders](../how-to/models/configure-term-encoders-and-decoders.md#choose-an-encoder), a multilayer perceptron (MLP) [controller](../how-to/models/configure-a-controller.md), the synthetic dataset over the `directed` grammar, [AdamW](../how-to/training/configure-a-trainer.md#choose-an-optimizer-and-a-schedule), a cosine schedule, and the [CPU device policy](../how-to/training/choose-a-device-policy.md#select-a-packaged-policy). |
| `parse`, `approximation` | Retain the [packed parse forest](../explanation/charts-and-approximation.md#reusing-partial-analyses), construct every chart item, including those that never reach a root, and bound the [residual mass](../explanation/charts-and-approximation.md#the-residual-mass-budget) any truncating component may leave. |
| [`model`](../reference/spec.md#lofi.spec.ModelSection) | Take the learner's language from the data, shrink the packaged components to width 4, and declare one representation space, `categories`, over category terms. |
| [`loss`](../reference/configuration/training.md#losses) | Train on a grammaticality term and a coherence term over category terms, weighted `1.0` and `0.1`. |
| [`data`](../reference/spec.md#lofi.spec.DataSection) | Declare the training dataset: a grammar, a language, and how many sentences of which lengths to draw from it. |
| [`trainer`](../reference/configuration/training.md#trainers) | Train one epoch in batches of two, with [early stopping](../how-to/training/configure-a-trainer.md#early-stopping) on the grammaticality loss and one [checkpoint](../reference/checkpoints.md#best-and-resumable-checkpoints) per epoch. |
| `release`, [`optimizer`](../reference/configuration/training.md#optimizers) | Release the checkpoint that is best on validation, and turn off weight decay. |
| [`evaluators`](../reference/spec.md#lofi.spec.EvaluatorSpec) | Score held-out bracketing after training, with a [bootstrap interval](../how-to/evaluation/evaluate-a-run.md#set-the-interval). |

The two loss terms ask for different kinds of agreement.
[Grammaticality](../explanation/induction-objectives.md#grammaticality) favors
assigning probability to the observed grammaticality label.
[Coherence](../explanation/induction-objectives.md#coherence) compares the
category distribution decoded for a span with a distribution obtained by
combining its parts. Agreement between those routes can help organize the
model's representations. Recovery of the source grammar is measured by the
held-out evaluators.

Consider the two-word string `word-0 word-1`. The grammar gives `word-0` the
category `NP`, for noun phrase, and `word-1` the category `S\NP`, for an
expression that needs an `NP` on its left to make a sentence (`S`). Combining
them gives `S`. These artificial words let us study whether the learner finds
the combination pattern without relying on its knowledge of English.

The grammar is the part every later chapter changes. Here it is, as the
document declares it under `data.datasets.train.dataset`:

```yaml
grammar:
  formalism: declared
  name: synthetic-parser
  signature:
    primitives: [S, NP, conj]
    operators:
      - {name: 'NP[conj]', sort: Cat, erases_to: NP}
    conjunctions:
      markers: [conj]
      pairs:
        - unmarked: NP
          marked: NP[conj]
  rules:
    name: application
    rules:
      - rule: application
        view: /
      - rule: application
        view: "\\"
  lexicon:
    kind: inline
    entries:
      - {form: word-0, category: NP}
      - {form: word-1, category: "S\\NP"}
      - {form: word-2, category: "S\\NP/NP"}
      - {form: word-3, category: NP/NP}
      # ... four more entries over the same categories
  goal:
    categories: [S]
```

Reading the grammar line by line, with $X$ and $Y$ standing for arbitrary
categories in the application rules:

| Line | What it says |
|---|---|
| [`formalism: declared`](../how-to/grammars/choose-a-formalism.md#relate-the-declaration-to-the-mathematical-grammar) | Declare the grammar by specifying each of its parts. |
| `primitives: [S, NP, conj]` | The [atomic categories](../how-to/grammars/declare-a-signature.md#declare-primitives-and-constructors). Every other category is built from these by the signature's constructors. |
| `operators`, `conjunctions` | A [featured category](../how-to/grammars/declare-a-signature.md#declare-featured-categories-and-conjunctions) `NP[conj]` that erases to `NP`, and the marker that coordination would use. No coordination rule is declared, so neither takes part in a derivation here. |
| `rules` | [Forward and backward application](../how-to/grammars/choose-combinatory-rules.md#the-rule-families), the two views of one rule. $\frac{X/Y\quad Y}{X}$ and $\frac{Y\quad X\backslash Y}{X}$. |
| `lexicon` | Eight forms, each with one category. A form's category is what a learner has to recover. |
| [`goal`](../how-to/grammars/choose-normal-form-matching-goal.md#declare-a-goal) | A derivation of a whole sentence must end in `S`. |

The slash constructors `/` and `\` come from the `directed` fragment the
`defaults` list selects, and [composition](../explanation/documents-are-the-dsl.md#composition) merges the two.

## Composing the document

[`show-config`](../reference/cli.md#show-config) prints the composed document with the source of every leaf,
that is, every individual setting at the end of a nested path:

```bash
uv run lofi show-config --config docs/examples/synthetic-parser.yaml
```

The full listing runs to about four hundred lines. An
[epoch](../how-to/training/configure-a-trainer.md#optimization-units) is one pass
through the training examples; a [batch](../how-to/training/configure-a-trainer.md#batch-samplers) groups examples for processing
together. A [checkpoint](../reference/checkpoints.md#best-and-resumable-checkpoints) saves the model's state so that the selected state
can be used later. With those terms in mind, read the [`trainer`](../reference/configuration/training.md#trainers) section:

```text
trainer:
  kind: "parser"  # file docs/examples/synthetic-parser.yaml
  dataset: "train"  # file docs/examples/synthetic-parser.yaml
  epochs: 1  # file docs/examples/synthetic-parser.yaml
  batch_size: 2  # file docs/examples/synthetic-parser.yaml
  accumulation_steps: 2  # file docs/examples/synthetic-parser.yaml
  max_grad_norm: null  # file docs/examples/synthetic-parser.yaml
  active_groups[0]: "token_encoder"  # file docs/examples/synthetic-parser.yaml
  active_groups[1]: "terms"  # file docs/examples/synthetic-parser.yaml
  active_groups[2]: "identity"  # file docs/examples/synthetic-parser.yaml
  active_groups[3]: "application"  # file docs/examples/synthetic-parser.yaml
  active_groups[4]: "controller"  # file docs/examples/synthetic-parser.yaml
  active_groups[5]: "parser"  # file docs/examples/synthetic-parser.yaml
  batch_sampler:
    reshuffle_each_epoch: false  # file docs/examples/synthetic-parser.yaml
    kind: "length-bucket"  # file docs/examples/synthetic-parser.yaml
    bucket_factor: 100  # file docs/examples/synthetic-parser.yaml
    declared_output: null  # default
  early_stopping:
    metric: "loss/grammaticality"  # file docs/examples/synthetic-parser.yaml
    mode: "min"  # file docs/examples/synthetic-parser.yaml
    patience: 2  # file docs/examples/synthetic-parser.yaml
    minimum_delta: 0.0  # file docs/examples/synthetic-parser.yaml
    keep_best: 1  # file docs/examples/synthetic-parser.yaml
  checkpoints:
    interval: 1  # file docs/examples/synthetic-parser.yaml
    keep: 1  # file docs/examples/synthetic-parser.yaml
  declared_output: null  # default
```

A [leaf marked `file`](../explanation/documents-are-the-dsl.md#provenance) is spelled in the document, and one marked `default` is
the schema's own default. A leaf a selected fragment supplies is marked
`group` followed by the path of that fragment's file, so the `kind` of the
parser reads `# group <path>/lofi/configs/model/parser/inside-outside.yaml`.

The same [composition](../explanation/documents-are-the-dsl.md#composition) is available from Python. A
[`RunSpecCodec`](../reference/spec.md#lofi.spec.RunSpecCodec) composes a document against a search
path and returns the validated [`RunSpec`](../reference/spec.md#lofi.spec.RunSpec); [`config_search_path`](../reference/spec.md#lofi.spec.config_search_path) gives the path
the command line uses, which looks beside the document, then in the project's
`configs/` directory, then in the packaged library and in [installed plugins](../how-to/extending/plugins.md#how-families-are-discovered):

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
run = codec(Path("docs/examples/synthetic-parser.yaml"))

grammar = run.data.datasets["train"].dataset.grammar
print(grammar.signature.primitives)
print([constructor.name for constructor in grammar.signature.constructors])
print(len(grammar.lexicon.entries), grammar.goal.kind, grammar.goal.categories)
```

```text
('S', 'NP', 'conj')
['/', '\\']
8 root ('S',)
```

The primitives and the eight lexical entries are the document's own. The
constructors and the `root` goal kind come from the `directed` fragment, whose
own [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) and primitive `N` the document replaces. [`load_plugins`](../reference/spec.md#lofi.spec.load_plugins) registers
every installed component family before composition, so a plugin's variants
validate exactly as LoFI's own do.

An override is checked against the same schema. On the command line it is
`--set trainer.epochs=2`; from Python it is one more argument to the codec:

```python
document = Path("docs/examples/synthetic-parser.yaml")

longer = codec(document, overrides=("trainer.epochs=2",))
print(longer.trainer.epochs)
```

```text
2
```

A misspelled leaf is refused before anything is built, and the refusal lists
the settings the schema does declare:

```python
from lofi.spec import UnknownSettingError

try:
    codec(document, overrides=("trainer.epoch=2",))
except UnknownSettingError as refused:
    print(str(refused)[:60])
```

```text
trainer.epoch is not a declared setting; the schema declares
```

## Training the point

[`lofi run`](../reference/cli.md#run) composes the document again, draws the corpus, builds the model, and
trains it:

```bash
uv run lofi run \
  --config docs/examples/synthetic-parser.yaml \
  --root runs/first \
  --seed 11 \
  --experiment-seed 5
```

```text
79be832e39614eaa completed runs/first/runs/79be832e39614eaa
```

The command prints one line per point: its identifier, the status it ended
in, and its directory.

[`--seed`](../how-to/experiments/sweeps.md#seed-the-points) identifies a repetition of the point. `--experiment-seed` supplies
the experiment seed from which its actual training seed is derived. A seed
controls reproducible randomized choices, such as model initialization. The
point's directory records that
derivation, so running the same point under a different experiment seed is refused rather
than mixed into the same records. `--root` is the directory the records are
written under; it is a filesystem location, distinct from the experiment seed.

The point writes one directory, `runs/first/runs/<point-id>/`, named by its
sixteen-character [point identifier](../explanation/documents-are-the-dsl.md#point-identity), a digest used to locate the run. The
[checkpoint reference](../reference/checkpoints.md) describes each file in it.
Three matter here: `config.resolved.json` is the composed document the point
trained, `manifest.json` records what the point declared and how it ended, and
`metrics.json` records what it measured. Running the same command again finds the point recorded as
completed, prints the same
`79be832e39614eaa completed runs/first/runs/79be832e39614eaa` line and leaves
every file in the directory as it was.

## Reading the records

Open the directory printed by the command and open `metrics.json` in a text
editor. JSON stores named values using braces and quoted names; look for the
`stage`, `point`, `corpus`, and `phases` entries. A `closed` stage says the run
has finished recording its measurements. The evaluator rows report how well
the learned grammar recovered the reference structure.

[`read_metrics`](../reference/run.md#lofi.run.read_metrics) reads a point's metrics record back as a
validated [`MetricsRecord`](../reference/run.md#lofi.run.MetricsRecord), or returns `None` for a directory that holds none.
The record's `stage` is `prepared` once preparation ends and `closed` when the
point closes. Its `point` field holds the point's identifier and the measured
parameter count of each component role, `corpus` reports the lengths, [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline)
and ambiguity of the corpus that was drawn, and `phases` gives the wall time of
each phase. None of these numbers is declared anywhere: each is measured from
the built model or the drawn corpus.

## Try this

- Train two epochs with `--set trainer.epochs=2` and compare the closed
  metrics records of the two points. The resolved documents differ in one leaf,
  so the point identifiers differ too.
- Replace the RNN term encoder with the [packaged transformer](../how-to/models/configure-term-encoders-and-decoders.md#choose-an-encoder) by selecting a
  group from the command line:
  `--set model/encoders/terms=transformer --set model.encoders.terms.heads=2`.
  The document's term width of 4 still applies to the selected fragment, and
  the fragment's eight heads do not divide it, so the selection alone is
  refused with `model.encoders was refused: the heads divide the width they
  split between them`. Two heads do divide it.
- Add a ninth lexical entry with a category the grammar already uses, and check
  with [`show-config`](../reference/cli.md#show-config) that the composed [lexicon](../how-to/grammars/declare-a-lexicon.md#list-entries-inline) holds nine entries.

## Next

[Chapter 2](02-declare-a-grammar.md) writes a grammar from its signature, rules
and lexicon and draws a corpus from it, without training anything.
