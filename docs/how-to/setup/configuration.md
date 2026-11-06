# Author and compose a configuration

Composition turns one run document into one validated
[`RunSpec`](../../reference/spec.md): it fills each `defaults` slot with a
packaged or project fragment, applies a profile, the overlay documents and the
command line overrides in that order, interpolates the resolvers, and
validates the result as a whole. This guide shows each layer, how to read
where a leaf came from, and how a bad declaration is refused.

## The words this site uses

| Term | Meaning |
| --- | --- |
| run document | A YAML, TOML or JSON document whose composition is a `RunSpec`. [`docs/examples/synthetic-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/synthetic-parser.yaml) is one. |
| point | One resolved `RunSpec` under one seed. Its sixteen-character identifier is derived from the resolved document and the seed, so two spellings that resolve alike are one point. |
| sweep | The expansion of a run document over declared axes into points. |
| experiment | A document that declares a sweep together with the seed policy, freeze, stopping rules and reports it commits to. |

The [sweep guide](../experiments/sweeps.md) covers sweeps and experiments.
This page covers one point's document.

## Show the composed document

`show-config` composes a document and prints every resolved leaf with the
layer that set it:

```bash
uv run lofi show-config --config docs/examples/synthetic-parser.yaml
```

An excerpt of its output:

```text
trainer:
  kind: "parser"  # file docs/examples/synthetic-parser.yaml
  dataset: "train"  # file docs/examples/synthetic-parser.yaml
  epochs: 1  # file docs/examples/synthetic-parser.yaml
  batch_size: 2  # file docs/examples/synthetic-parser.yaml
```

A leaf the document leaves out is marked `# default`, and a leaf a fragment
supplies is marked `# group` with the fragment's path. Run `show-config`
before a long run to see whether an option came from the layer you intended.

From Python, a [`RunSpecCodec`](../../reference/spec.md) composes against a
search path and returns the `RunSpec`. `config_search_path` returns the path
the command line uses, and `load_plugins` registers every installed component
family first, so a plugin's variants validate as LoFI's own do:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")
base = codec(document)
```

## Select fragments with `defaults`

The `defaults` list fills one configuration slot per entry. A dotted slot maps
to nested directories, so this entry reads `model/encoders/terms/rnn.yaml`:

```yaml
defaults:
  - model.encoders.terms: rnn
  - optimizer: adamw
  - device_policy: cpu
```

`lofi configs` lists every slot and the options the search path offers for it:

```bash
uv run lofi configs
```

```text
baselines: cfg-collapse fully-supervised induction inventory-seed inventory-seeded inventory-seeded-fully-supervised inventory-seeded-prange inventory-seeded-tag-dictionary prange tag-dictionary
data.datasets.train.constraint_policy.semantic: all none
data.datasets.train.constraint_policy.syntactic: all none partial per-occurrence
data.datasets.train.dataset: ccgbank ccgbank-reskin synthetic synthetic-inference synthetic-inference-balanced synthetic-inference-entailment synthetic-inference-overlap
data.datasets.train.dataset.grammar: ab ccgbank cfg cfg-collapse cfg-toy directed nl-diamond undirected
data.datasets.train.dataset.grammar.rules: application application-identity composition type-raising
```

That is an excerpt; the full listing has one line per slot. Fragments are
looked for in this order, and the first option with a given name wins:

1. beside the document being composed;
2. under `configs/` in the current project;
3. in LoFI's packaged library, `lofi/configs/`;
4. in the directories installed plugins contribute through the
   `lofi.config_dirs` entry point group.

A project can thus replace a packaged option by placing one of the same name
under its own `configs/`, without editing LoFI or a plugin. Put the subtree
several runs share in a fragment and the values one run changes in the run
document. `synthetic-parser.yaml` selects the packaged RNN term encoder and
lowers its width in the document.

## Apply a profile, overlays and overrides

Composition applies its layers from least to most specific:

1. schema defaults and the selected fragments;
2. the run document;
3. a profile, `--profile NAME`;
4. each overlay document, `--extra PATH`, in command line order;
5. each override, `--set KEY=VALUE`, in command line order.

A profile is a document at `profiles/<NAME>` under any root of the search
path. An overlay is a document named on the command line, so it stays visible
in the invocation. An override sets one leaf.
[`docs/examples/running/patient.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/running/patient.yaml)
is an overlay that raises the early stopping patience. Apply it and one
override:

```bash
uv run lofi show-config \
  --config docs/examples/synthetic-parser.yaml \
  --extra docs/examples/running/patient.yaml \
  --set trainer.epochs=2
```

```text
  epochs: 2  # override trainer.epochs=2
  ...
    patience: 4  # extra
```

The same layers are arguments of the codec. Here the search path is extended
with `docs/examples/running`, whose `profiles/longer.yaml` sets two epochs, and
`compose_traced` returns every layer that set each leaf, lowest precedence
first:

```python
from lofi.spec import RunSpec, SpecLoader

search_path = (Path("docs/examples/running"), *config_search_path(Path.cwd()))

traced = SpecLoader(search_path=search_path).compose_traced(
    document,
    schema=RunSpec,
    profile="longer",
    extra={"trainer.early_stopping.patience": 4},
    overrides=("trainer.epochs=3",),
)
print(traced.spec.trainer.epochs, traced.spec.trainer.early_stopping.patience)
for leaf in ("trainer.epochs", "trainer.early_stopping.patience"):
    print(leaf, [(o.source.value, o.layer) for o in traced.sources[leaf]])
```

```text
3 4
trainer.epochs [('file', 'docs/examples/synthetic-parser.yaml'), ('profile', 'longer'), ('override', 'trainer.epochs=3')]
trainer.early_stopping.patience [('file', 'docs/examples/synthetic-parser.yaml'), ('extra', '')]
```

The document set one epoch, the profile two and the override three, and the
override wins. A point's run directory records the same provenance in
`config.sources.json`, beside the resolved `config.resolved.json`.

## Override leaves and groups

An override is `dotted.path=value`. The value is read against the declared
type of the field, so numbers, booleans, nulls and arrays keep their types.
Below a tagged union, the field is the one the selected variant declares: the
variant the document, its fragments and the earlier overrides have selected by
the time the override applies. `model/encoders/terms=rnn` followed by
`model.encoders.terms.width=6` sets the RNN's width, and a field only some
variants declare, or declare with different types, needs no restated `kind`.
An override whose key contains `/` selects a fragment instead of setting a
leaf: `model/encoders/terms=transformer` fills the `model.encoders.terms`
slot, and a slot at the root of the document takes a leading `/`:

```python
swapped = codec(document, overrides=("/optimizer=sgd",))
print(base.optimizer.kind, swapped.optimizer.kind)
```

```text
adamw sgd
```

A selected fragment still has to validate with the rest of the document.
Selecting the packaged transformer term encoder for `synthetic-parser.yaml`,
whose term width is 4, is refused with a `ConfigValueError`:
`model.encoders was refused: the heads divide the width they split between
them`.

## Extend a list without restating it

On the final path segment, `field+` appends to a list and `+field` prepends
to it. An unmarked field replaces the list whole. The marker works in
documents and in overrides:

```python
appended = codec(
    document,
    overrides=('loss.terms+=[{"objective":{"kind":"chart-scale"}}]',),
)
prepended = codec(
    document,
    overrides=('loss.+terms=[{"objective":{"kind":"chart-scale"}}]',),
)
for spec in (base, appended, prepended):
    print([term.objective.kind for term in spec.loss.terms])
```

```text
['grammaticality', 'coherence']
['grammaticality', 'coherence', 'chart-scale']
['chart-scale', 'grammaticality', 'coherence']
```

A fragment that refines another names it in its own `defaults` and extends
its list. The packaged `application-identity` rule inventory adds one rule to
`application` this way:

```yaml
defaults:
  - application
name: application-identity
rules+:
  - rule: pattern
    name: identity-combinator
    # remaining rule fields
```

A bare name in `defaults` is looked up beside the fragment that lists it. A
fragment in a project's `configs/` that refines a packaged option names it by
its path from the configuration root, as in
`- data/datasets/train/dataset/grammar/rules/application`. Each added element
keeps its own source, and extending a field that is not a list is refused.

## Interpolate paths, fragments and seeds

Three resolvers, registered through the `lofi.resolver` entry point group,
read values that are not leaves of the document:

| Expression | Value |
| --- | --- |
| `${lofi.path:rel}` | `rel` joined against the project's data root, for instance `source: ${lofi.path:ccgbank}` in `experiments/s1/s1-run.yaml`. |
| `${lofi.pkg:group/option}` | The absolute path of a packaged fragment. |
| `${lofi.seed:name}` | A subseed for the stream `name`, derived from the run seed. |

A path argument must be relative and POSIX-spelled: an absolute,
home-relative, drive-qualified or `..` path is refused. A string beginning
with `lofi.resource:` instead names an exact file below the packaged
configuration root, so a portable document never carries a developer's
absolute path:

```yaml
model:
  token_encoder:
    vectors: lofi.resource:run/synthetic-parser.vec
```

A subseed depends only on the run seed and the stream name, so a stream
draws the same seed in every process that composes the same run, whichever
other streams are asked for. [`derive_seed`](../../reference/spec.md)
computes it:

```python
from lofi.spec import derive_seed

print(derive_seed(5, "model-structures"), derive_seed(5, "model-structures"))
print(derive_seed(6, "model-structures"))
```

```text
693149464 693149464
2148614568
```

## How composition refuses

Every refusal comes before any model or data loader is built. Each names the
leaf, the layer or the options it expected:

```python
for override in ("trainer.epochs=two", "/optimizer=lion", "trainer.epochs"):
    try:
        codec(document, overrides=(override,))
    except Exception as refused:
        print(type(refused).__name__, "|", str(refused)[:68])
```

```text
SettingParseError | trainer.epochs expects int; got 'two'
UnresolvedFragmentError | optimizer selects 'lion', which is not an option; the search path of
MalformedOverrideError | override 'trainer.epochs' is malformed: Override 'trainer.epochs' is
```

The other refusals are `UnknownSettingError` for a path the schema does not
declare, with the settings it does declare; `ConfigValueError` for a value a
validator refuses, and for an override whose field the variants of a tagged
union declare with different types while no variant is yet selected; and
`ConfigDocumentError` for a document that does not parse or a composed
document refused as a whole, such as an accumulation window an epoch cannot
close.

## Where each section is declared

The sections of a run document have their own guides:

| Section | Guide |
| --- | --- |
| `data.datasets.<name>.dataset.grammar` | [Declare a category signature](../grammars/declare-a-signature.md), [Declare a lexicon](../grammars/declare-a-lexicon.md) |
| `data.datasets.<name>.dataset` | [Generate a synthetic corpus](../data/generate-a-synthetic-corpus.md), [Read a corpus](../data/read-a-corpus.md) |
| `model` | [Choose a token encoder](../models/choose-a-token-encoder.md), [Configure a chart parser](../models/configure-a-chart-parser.md) |
| `loss` | [Combine losses](../training/combine-losses.md) |
| `trainer`, `optimizer`, `scheduler` | [Configure a trainer](../training/configure-a-trainer.md) |
| `release` | [Checkpoint and resume a run](../training/resume-from-a-checkpoint.md) |
| `device_policy` | [Choose a device policy](../training/choose-a-device-policy.md) |
| `evaluators` | [Evaluate a trained point](../evaluation/evaluate-a-run.md) |

The [configuration reference](../../reference/configuration/index.md) lists
every field, and
[documents are the DSL](../../explanation/documents-are-the-dsl.md) explains
why a run is declared by a document rather than by code.
