# Read a corpus

A collected corpus is a dataset whose `kind` names an installed
`lofi.dataset` entry other than `synthetic` or `synthetic-inference`: it
declares where the local copy lives, the grammar its derivations or strings
are read under, and, for an NLI corpus, how sentences are tokenized and how
repeated responses are reduced to one gold label. This guide selects a corpus,
points it at a local copy, reads it from Python to check that copy, and lists
how each kind is refused.

## List the installed readers

Each dataset kind is one entry in the `lofi.dataset` group, and each file
format a reader decodes is one entry in the `lairs.codecs` group:

```python
from importlib.metadata import entry_points

print(sorted(point.name for point in entry_points(group="lofi.dataset")))
print(sorted(point.name for point in entry_points(group="lairs.codecs")))
```

```text
['ccgbank', 'ccgbank-reskin', 'chaosnli', 'expert-ratings', 'fracas', 'mnli', 'pmb-ccg', 'pmb-drs', 'snli', 'synthetic', 'synthetic-inference', 'unli']
['bead', 'brat', 'ccgbank-auto', 'ccgbank-parg', 'chaosnli', 'conllu', 'fracas', 'layers-export', 'megaacceptability', 'megaintensionality', 'meganegraising', 'megaveridicality', 'mnli', 'pmb-ccg', 'pmb-drs', 'snli', 'unli']
```

The [entry-point reference](../../reference/entry-points.md) lists the module
each name loads. Codecs such as `bead`, `brat` and `conllu` come from the
installed `lairs` package rather than from LoFI. The dataset kinds other than
the two synthetic ones fall into five groups:

| Kind | Reads | Batches it yields |
| --- | --- | --- |
| `ccgbank` | CCGbank `AUTO` derivations, with `PARG` dependencies | sentences and gold derivations |
| `ccgbank-reskin` | CCGbank derivations whose frequent leaf categories are respelled as synthetic forms | sentences |
| `pmb-ccg`, `pmb-drs` | Parallel Meaning Bank CCG derivations, or its clausal DRS graphs as gold logical forms | sentences and gold forms |
| `snli`, `mnli`, `chaosnli`, `unli`, `fracas` | a normalized NLI or FraCaS release | inference pairs |
| `expert-ratings` | a persisted document of expert ratings of induced categories, bound to one annotation package by its digest | ratings |

## Point a dataset at CCGbank

The packaged `ccgbank` option selects the reader and reads the corpus root
from an environment variable:

```yaml
kind: ccgbank
source: ${oc.env:LOFI_CCGBANK}
splits: [train, validation, development, test]
unknown_words: {kind: skip}
```

Select it with the packaged CCGbank grammar, which declares the CCGbank
signature, its features and its combinators:

```yaml
defaults:
  - data.datasets.train.dataset: ccgbank
  - data.datasets.train.dataset.grammar: ccgbank
```

The fields are:

| Field | What it says |
| --- | --- |
| `source` | The corpus root, or one `AUTO` file. A single file is read as training data. |
| `splits` | The conventional sections to read: training 02 to 21, validation 00, development 22, test 23. Each split is named once. |
| `training_split` | The split whose gold derivations a parser trains on, `train` by default. It must be one of `splits`; every other split is retained. |
| `validation_split` | Which held-out split early stopping reads. |
| `unknown_words` | `skip` drops a sentence whose word the lexicon lacks; `assign` gives the word a declared `category`. |
| `grammar`, `language` | The grammar the derivations are read under and the learner's language. |

A source can also be written relative to the run's data root with
`${lofi.path:ccgbank}`. The command line takes the data root to be the
project directory, so this resolves to `ccgbank/` beside the project's
`configs/` directory. `experiments/s1/s1-run.yaml` declares its source that
way.

## Check the local copy

Reading the corpus once from Python checks that the copy is where the
document says and that its derivations compile under the declared grammar.
`experiments/s1/s1-reduced-run.yaml` reads the small CCGbank fixture checked
in under `experiments/s1/data/ccgbank`, one file per section:

```python
from collections import Counter
from pathlib import Path

from lofi.data.ccgbank import PLUGIN
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
run = codec(Path("experiments/s1/s1-reduced-run.yaml"))
declaration = run.data.datasets["train"].dataset
print(declaration.kind, declaration.source, declaration.splits, declaration.unknown_words.kind)

dataset = PLUGIN.lower(PLUGIN.check(declaration), ParsePolicy(Retain.FOREST, Reachability.ALL))
report = dataset.report
print(report.records_seen, report.emitted, dict(report.skips_by_reason))
print(Counter(example.item.split for example in dataset.corpus.examples))
```

```text
ccgbank experiments/s1/data/ccgbank ('train', 'validation', 'development', 'test') skip
8 8 {}
Counter({'train': 2, 'validation': 2, 'development': 2, 'test': 2})
```

`check` compiles the grammar and the language; `lower` reads the selected
splits and measures the corpus. Each AUTO and PARG file is read through the
registered `ccgbank-auto` and `ccgbank-parg` codecs' own `parse`, which returns
the derivations and dependencies directly; the codecs' `decode` into Layers
records translates that same parsed value, so the two routes cannot disagree.
The load report counts the records the reader
saw, the examples it emitted and the records it skipped by reason, so an
`unknown_words: {kind: skip}` policy that drops sentences shows up here rather
than as a smaller corpus with no explanation.

To check another copy, replace the source. Because `source` is a field of
several dataset kinds, with a different type in some of them, a bare
`data.datasets.train.dataset.source=<path>` override is refused until the
kind is fixed. Restate the kind in the same override; the rest of the dataset,
its grammar included, is kept:

```python
local = codec(
    Path("experiments/s1/s1-reduced-run.yaml"),
    overrides=('data.datasets.train.dataset={"kind": "ccgbank", "source": "experiments/s1/data/ccgbank/00"}',),
)
print(local.data.datasets["train"].dataset.source)
```

```text
experiments/s1/data/ccgbank/00
```

Lowering that declaration reads the one `AUTO` file under section 00 as
training data.

The same check from the command line composes the document and prints the
resolved source with the layer that set it:

```bash
uv run lofi show-config --config experiments/s1/s1-reduced-run.yaml | grep 'source: "'
```

```text
        source: "experiments/s1/data/ccgbank"  # file experiments/s1/s1-reduced-run.yaml
```

## Respell CCGbank derivations

`ccgbank-reskin` keeps each CCGbank derivation and replaces every frequent
feature-erased leaf category with a synthetic form, so a learner sees
CCGbank's derivation shapes over a vocabulary it cannot memorize from the
words:

```yaml
kind: ccgbank-reskin
source: ${lofi.path:ccgbank}
splits: [train]
mapping: {prefix: type, minimum_frequency: 10}
seed: derived
```

`mapping.prefix` spells the synthetic forms and `mapping.minimum_frequency`
is the least count a category needs to be mapped. A reskin also takes the
`corruption` and `noise` blocks of a synthetic corpus, described in
[Corrupt strings and their labels](corrupt-strings.md).

## Read PMB, NLI and FraCaS releases

A collected dataset reads one file per split. Each entry of `sources` names a
file beside the identity of the release it holds:

| Field | What it says |
| --- | --- |
| `sources[].source` | The local file. |
| `sources[].release.source_id`, `sources[].release.release_id` | The corpus and the release read. |
| `sources[].release.split` | The split the file holds. Each split is read from one file. |
| `sources[].release.language`, `sources[].release.source_method`, `sources[].release.spdx` | The language, how the data was made, and its license. None may be empty. |
| `sources[].release.eprint_refs` | Citations for the release. |
| `training_split` | `pmb-ccg`, `snli`, `mnli`, `chaosnli` and `fracas` only. The split a parser or a grounding trains on; every other split is retained, and the first retained split in declaration order is the held-out split. Unset, the dataset is read for evaluation and featurized prediction and refuses to train. |
| `labeling` | `snli`, `mnli`, `chaosnli` and `fracas` only. The installed inference labeling, `three-way` by default, that reads each pair's logical relation as an output label. |
| `tokenizer` | NLI and FraCaS only. Defaults to `regex` with pattern `\S+`. |
| `aggregation` | NLI and FraCaS only. Defaults to `majority`, which reduces each item pair's repeated relations to the unique most frequent one. |

Each collected kind has a packaged option under `data.datasets.train.dataset`:
`pmb-ccg`, `pmb-drs`, `snli`, `mnli`, `chaosnli`, `unli` and `fracas`. Each
names its files beneath the data root, e.g. `${lofi.path:snli/snli_1.0_train.jsonl}`,
and their release identity; a run selects a grammar and a language beside it and
overrides `sources` when it reads other files or splits. The packaged `snli`
option reads the training, development and test files as the `train`,
`validation` and `test` splits and trains on `train`. A release whose
license is not stated by its distribution declares `spdx: NOASSERTION`. Expert
ratings are read by the `expert-categories` evaluator, and its packaged option
is `evaluators.expert.ratings: expert-ratings`.

A `pmb-ccg` corpus trains a parser the way a CCGbank corpus does: each
derivation step is lowered to the compiled rule of the declared grammar that
licenses it, and a step no compiled rule licenses keeps no rule. An `snli`,
`mnli`, `chaosnli` or `fracas` corpus trains a grounding on the decided pairs of
its training split, each relation read through the declared labeling; a
relation other than `entails`, `contradicts` or `neutral` is refused. A
collected release has no gold logical forms, so a grounding trained on one
reports no inference rows that need them.

The PMB grammars are packaged per language under
`data.datasets.train.dataset.grammar`: `pmb/en`, `pmb/de`, `pmb/it` and
`pmb/nl` extend `pmb/base`, which declares the PMB signature with colon
features (`s:dcl`, `np:thr`), and each language adds its lexical coercions.
[Featurize a dataset](featurize-a-dataset.md) covers the tokenizer and the
aggregation.

## Refusals

- A CCGbank source or a collected `source` that is empty is refused when the
  document is validated, as is a split named twice and a `training_split` no
  declared file holds.
- `pmb-drs` and `unli` refuse to train: no training route reads gold graphs,
  or categorical and graded responses together.
- A bare `source` override is refused with an `AmbiguousSettingError`; name
  the `kind` in the same override, as above.
- A release field that is empty or only whitespace is refused.
- Majority aggregation refuses an item pair whose responses have no unique
  majority.
