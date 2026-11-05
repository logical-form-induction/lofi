# Your first run

This chapter trains one LoFI point from start to finish: it installs the
package, reads a complete run document, composes that document into a validated
specification, trains the point on a small synthetic corpus, and reads the
records the run leaves behind. Every later chapter starts from the same
document and changes one section of it.

If you have used a configuration system such as Hydra, the composition step
will look familiar: a `defaults` list selects packaged fragments, and the
command line overrides single leaves. The main difference is that LoFI
validates the composed document as a whole before it builds anything. A
misspelled key or a value of the wrong type is refused at composition time,
and the resolved document, not the files it was spelled in, identifies the run.

## Prerequisites

From a source checkout, build the project environment:

```bash
uv sync
```

The environment includes the `nn` extra, which installs PyTorch. A quick check
that the package and its configuration library are in place:

```bash
uv run python -c "from lofi.spec import load_plugins; print(len(load_plugins()) > 0)"
```

```text
True
```

`load_plugins` loads every installed component family and returns the names it
loaded. If that prints `True` without an error, you're set.

## The run document

The document this chapter trains is
[`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml).
Open it beside this page. It has eight top-level parts:

| Section | What it says |
|---|---|
| `defaults` | Select one packaged fragment per slot: an inside-outside chart parser, a static token encoder, RNN term encoders and decoders, an MLP controller, the synthetic dataset over the `directed` grammar, AdamW, a cosine schedule, and the CPU device policy. |
| `parse`, `approximation` | Retain the packed parse forest, construct every chart item rather than only those that reach a root, and bound the residual mass any truncating component may leave. |
| `model` | Take the learner's language from the data, shrink the packaged components to width 4, and declare one representation space, `categories`, over category terms. |
| `loss` | Train on a grammaticality term and a coherence term over category terms, weighted `1.0` and `0.1`. |
| `data` | Declare the training dataset: a grammar, a language, and how many sentences of which lengths to draw from it. |
| `trainer` | Train one epoch in batches of two, with early stopping on the grammaticality loss and one checkpoint per epoch. |
| `release`, `optimizer` | Release the checkpoint that is best on validation, and turn off weight decay. |
| `evaluators` | Score held-out bracketing after training, with a bootstrap interval. |

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

Reading the grammar line by line:

| Line | What it says |
|---|---|
| `formalism: declared` | The grammar is declared here, by its parts, rather than selected from a packaged formalism. |
| `primitives: [S, NP, conj]` | The atomic categories. Every other category is built from these by the signature's constructors. |
| `operators`, `conjunctions` | A featured category `NP[conj]` that erases to `NP`, and the marker that coordination would use. No coordination rule is declared, so neither takes part in a derivation here. |
| `rules` | Forward and backward application, the two views of one rule. `X/Y Y ⇒ X` and `Y X\Y ⇒ X`. |
| `lexicon` | Eight forms, each with one category. A form's category is what a learner has to recover. |
| `goal` | A derivation of a whole sentence must end in `S`. |

Notice what the grammar does not declare: the slash constructors `/` and `\`
themselves. Those come from the `directed` fragment the `defaults` list
selects, and composition merges the two.

## Composing the document

`show-config` prints the composed document with the source of every leaf:

```bash
uv run lofi show-config --config docs/examples/synthetic-parser.yaml
```

The full listing runs to about four hundred lines. Its `trainer` section reads:

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

A leaf marked `file` is spelled in the document, and one marked `default` is
the schema's own default. A leaf a selected fragment supplies is marked
`group` followed by the path of that fragment's file, so the `kind` of the
parser reads `# group <path>/lofi/configs/model/parser/inside-outside.yaml`.

The full listing runs to about four hundred lines. Its `trainer` section reads:

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

A leaf marked `file` is spelled in the document, and one marked `default` is
the schema's own default. A leaf a selected fragment supplies is marked
`group` followed by the path of that fragment's file, so the `kind` of the
parser reads `# group <path>/lofi/configs/model/parser/inside-outside.yaml`.

The same composition is available from Python. A
[`RunSpecCodec`](../reference/spec.md) composes a document against a search
path and returns the validated `RunSpec`; `config_search_path` gives the path
the command line uses, which looks beside the document, then in the project's
`configs/` directory, then in the packaged library and in installed plugins:

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
own lexicon and primitive `N` the document replaces. `load_plugins` registers
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

`lofi run` composes the document again, draws the corpus, builds the model, and
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

`--seed` is the point's seed and `--experiment-seed` is the root from which the
seed the point trains under is derived. The point's directory records that
derivation, so running the same point under a different root is refused rather
than mixed into the same records. `--root` is the directory the records are
written under.

The point writes one directory, `runs/first/runs/<point-id>/`, named by its
sixteen-character point identifier. The
[checkpoint reference](../reference/checkpoints.md) describes each file in it.
Three matter here: `config.resolved.json` is the composed document the point
trained, `manifest.json` records what the point declared and how it ended, and
`metrics.json` records what it measured. Running the same command again does
not train the point a second time. The directory already records the point as
completed, so the command prints the same
`79be832e39614eaa completed runs/first/runs/79be832e39614eaa` line and leaves
every file in the directory as it was.

## Reading the records

[`read_metrics`](../reference/run.md) reads a point's metrics record back as a
validated `MetricsRecord`, or returns `None` for a directory that holds none.
The record's `stage` is `prepared` once preparation ends and `closed` when the
point closes. Its `point` field holds the point's identifier and the measured
parameter count of each component role, `corpus` reports the lengths, lexicon
and ambiguity of the corpus that was drawn, and `phases` gives the wall time of
each phase. None of these numbers is declared anywhere: each is measured from
the built model or the drawn corpus.

## Try this

- Train two epochs with `--set trainer.epochs=2` and compare the closed
  metrics records of the two points. The resolved documents differ in one leaf,
  so the point identifiers differ too.
- Replace the RNN term encoder with the packaged transformer by selecting a
  group from the command line:
  `--set model/encoders/terms=transformer --set model.encoders.terms.heads=2`.
  The document's term width of 4 still applies to the selected fragment, and
  the fragment's eight heads do not divide it, so the selection alone is
  refused with `model.encoders was refused: the heads divide the width they
  split between them`. Two heads do divide it.
- Add a ninth lexical entry with a category the grammar already uses, and check
  with `show-config` that the composed lexicon holds nine entries.

## Next

[Chapter 2](02-declare-a-grammar.md) writes a grammar from its signature, rules
and lexicon and draws a corpus from it, without training anything.
