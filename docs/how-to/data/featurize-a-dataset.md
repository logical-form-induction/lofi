# Featurize a dataset

A featurizer turns a dataset's examples into training batches: it reads one
kind of supervision the dataset emits, hashes each item's tokens against the
dataset's vocabulary, and pads the batch with a reserved symbol. It is
declared beside the dataset, under `data.datasets.<role>.featurizer`. This
guide selects a featurizer, matches it to the supervision a dataset emits,
and sets the tokenizer and gold aggregation an NLI corpus reads with.

## Select a featurizer

Each featurizer is one entry in the `lofi.featurizer` group, and each has a
packaged option under `data.datasets.train.featurizer`:

| Option | Reads | Use it for |
| --- | --- | --- |
| `unlabeled` | unlabeled items | grammaticality and coherence training on strings alone |
| `derivation` | gold derivations | supervised parsing on CCGbank or a synthetic corpus |
| `span-constraint` | span constraints | training a parser under selected bracket constraints |
| `scalar` | a scalar response per item | graded judgments |
| `pair` | one pair-valued arm, chosen by `selection` | inference pairs |

Every featurizer takes a `padding` symbol, `<pad>` by default, which must be
one of the forms the dataset section reserves. The section declares those
forms in index order as `reserved_forms`, `["<pad>", "<unk>"]` by default, and
names the one an unseen token is read as in `unknown_form`, `<unk>` by
default:

```yaml
data:
  datasets:
    train:
      reserved_forms: ["<p>", "<bos>", "<u>"]
      unknown_form: <u>
      featurizer:
        kind: unlabeled
        padding: <p>
```

The model's token vocabulary and the featurizer's hashed lexicon both place
these forms ahead of the dataset's own forms. The pair
featurizer also takes `selection`, which names a `lofi.pair_selection` entry:
`forced-choice` (the default), `relational`, or `relational-scalar`. The
packaged `pair` option selects `relational`, which reads a labeled
relation between a premise and hypothesis.

Select a featurizer from the `defaults` list, or from the command line with a
group override. The checked-in parser document declares `unlabeled`; the
same document can be composed with another option:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")

print(codec(document).data.datasets["train"].featurizer)
for option in ("derivation", "pair"):
    run = codec(document, overrides=(f"data/datasets/train/featurizer={option}",))
    print(run.data.datasets["train"].featurizer)
```

```text
UnlabeledFeaturizerSpec(kind='unlabeled', padding='<pad>')
DerivationFeaturizerSpec(kind='derivation', padding='<pad>')
PairFeaturizerSpec(kind='pair', padding='<pad>', selection='relational')
```

The command line spelling of the same override is:

```bash
uv run lofi show-config \
  --config docs/examples/synthetic-parser.yaml \
  --set data/datasets/train/featurizer=derivation
```

An inference run selects the pair featurizer. The L4 run document does so in
its `defaults` list:

```python
inference = codec(Path("experiments/l4/l4-run.yaml"))
print(inference.data.datasets["train"].featurizer)
```

```text
PairFeaturizerSpec(kind='pair', padding='<pad>', selection='relational')
```

## Match the featurizer to the trainer

Composition validates the featurizer's own fields. Whether the featurizer
reads the examples the run feeds it is decided when the run is planned, before
any model is built. Each trainer's plugin declares the supervision arm it
presents to every featurized dataset, and each featurizer's plugin declares
the arms it accepts, read from its declaration without building it. Planning
refuses a featurizer that does not accept the arm its trainer presents with a
`FeaturizerMismatchError`, which names the dataset role, the trainer, the
presented arm and the arms the featurizer accepts. The parser document
composes with `pair` and the L4 run document with `derivation`, but neither
plans:

```python
from lofi.run import ExperimentSeed, plan_training

for path, option in [
    ("docs/examples/synthetic-parser.yaml", "pair"),
    ("experiments/l4/l4-run.yaml", "derivation"),
]:
    run = codec(Path(path), overrides=(f"data/datasets/train/featurizer={option}",))
    try:
        plan_training(run, point=ExperimentSeed(0).point("featurizer", 0))
    except ValueError as refused:
        print(type(refused).__name__, "|", refused)
```

```text
FeaturizerMismatchError | data.datasets.train.featurizer: the 'parser' trainer presents Unlabeled examples, but the 'pair' featurizer accepts Relational
FeaturizerMismatchError | data.datasets.train.featurizer: the 'parser' trainer presents Relational examples, but the 'derivation' featurizer accepts GoldDerivation
```

`lofi run` plans each point before it prepares one, so the same selection on
the command line exits with status `1` and the same message, and no point
directory is written:

```bash
uv run lofi run --config docs/examples/synthetic-parser.yaml \
  --set data/datasets/train/featurizer=pair \
  --root runs/featurize --experiment-seed 108
```

A synthetic corpus without an interpretation yields sentences and their gold
derivations, so `unlabeled`, `derivation` and `span-constraint` read it. A
`synthetic-inference` dataset yields labeled pairs, their sentences and their
certified forms, and is read with `pair`.

## Tokenize and aggregate an NLI corpus

A collected NLI or FraCaS dataset (`snli`, `mnli`, `chaosnli`, `unli`,
`fracas`) declares two more components inside the dataset, not beside it:

| Field | Default | What it says |
| --- | --- | --- |
| `tokenizer` | `{kind: regex, pattern: '\S+'}` | How a premise or hypothesis string becomes tokens. |
| `aggregation` | `{kind: majority}` | How repeated responses to one pair become one gold label. |

`tokenizer` is a `lofi.tokenizer` entry. The one installed tokenizer, `regex`,
splits a sentence into the matches of `pattern`, so the default splits on
whitespace. `aggregation` is a `lofi.gold_aggregation` entry. The one
installed aggregation, `majority`, reduces the repeated responses to each
pair of a premise and hypothesis to the unique most frequent relation, and refuses a
pair whose responses tie. Both are the defaults, so a declaration that omits
them reads whitespace tokens and majority labels.

[Read a corpus](read-a-corpus.md) lists the release fields an NLI dataset
declares.
