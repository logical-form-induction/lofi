# Export a point to the Hugging Face Hub

`lofi export` writes the bundle of one completed point: the point's weights,
the resolved run document that rebuilds its model, and a Hugging Face model
card, in a local directory that can be uploaded to the Hub as it stands. The
command opens no network connection. This guide declares the release metadata,
runs the export, reads the bundle, and loads it back.

The model card is written with `huggingface-hub`, which the `publish` extra
installs ([Install LoFI and its extras](../setup/installing.md)).

## Declare the release

A release declaration supplies only what the model card displays and the
trained point cannot know about itself. The checked-in
[`docs/examples/running/release.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/running/release.yaml)
declares one for the point the [first tutorial](../../tutorials/01-first-run.md)
trains:

```yaml
model_name: lofi-synthetic-parser
datasets: [synthetic-parser]
languages: [en]
source_method: synthetic generation from a declared grammar
license_expression: MIT
licenses:
  - {identifier: MIT, applies_to: null}
eprint_refs: []
code_uri: https://github.com/logical-form-induction/lofi
command: uv run lofi run --config docs/examples/synthetic-parser.yaml --root runs/first --seed 11 --experiment-seed 5
environment: uv sync from the committed lockfile
```

| Field | What it says |
| --- | --- |
| `model_name` | The name the card displays. |
| `datasets`, `languages` | The corpora and language tags the card lists. |
| `source_method` | How the training annotation was arrived at. |
| `license_expression`, `licenses` | The license the release is distributed under, and each license the training data carries with the part it applies to. |
| `eprint_refs` | Publications describing the training data. Optional. |
| `code_uri`, `command`, `environment` | Where the training code is published, the command that trained the point, and the environment it ran in. |
| `pretrained_resources` | For each pretrained component role, the local directory holding its construction assets. Optional. |

The declaration names no model identity, corpus revision, source revision or
seed. Those are read from the point. An unknown field is refused as in every
LoFI document. The declaration composes like any other document:

```python
from pathlib import Path

from lofi.run import HubReleaseDeclaration
from lofi.spec import SpecLoader

release = SpecLoader().compose_traced(
    Path("docs/examples/running/release.yaml"), schema=HubReleaseDeclaration
).spec
print(release.model_name, release.datasets, release.license_expression)
```

```text
lofi-synthetic-parser ('synthetic-parser',) MIT
```

An empty required field is refused, naming the field:

```python
try:
    SpecLoader().compose_traced(
        Path("docs/examples/running/release.yaml"),
        schema=HubReleaseDeclaration,
        overrides=("model_name=",),
    ).spec
except Exception as refused:
    print(type(refused).__name__, "|", str(refused)[:200])
```

```text
ConfigValueError | model_name was refused: Hub release metadata is empty
```

## Run the export

Name the run directory of a completed point, the release declaration, and a
bundle directory that does not exist yet:

```bash
mkdir -p bundles
uv run lofi export \
  --run runs/first/runs/<point-id> \
  --release docs/examples/running/release.yaml \
  --out bundles/synthetic-parser
```

The command reads the point's manifest and refuses a directory that records no
completed run. It plans the recorded run document again under the point's seed
derivation, builds the model, and restores the newest checkpoint of the class
the document's `release.checkpoint` declares (`validation-best` or
`resume-latest`). A run that holds no checkpoint of that class, or whose
checkpoint was written for another run declaration, is refused. The bundle is written in a staging
directory beside `--out` and renamed into place only once every file is
written, so a failed export leaves no partial bundle. An existing `--out` is
refused rather than overwritten. On success the command prints the `--out`
path; a refusal is reported on standard error as `lofi export: <reason>` with
exit status `2`.

## Read the bundle

```text
bundles/synthetic-parser/
├── manifest.json
├── config.json
├── README.md
├── states/
│   └── <ordinal>.pt
└── resources/
    └── <ordinal>/...
```

| File | What it holds |
| --- | --- |
| `config.json` | The resolved run document the point trained. |
| `README.md` | The model card: the release metadata as Hugging Face card data, tagged `lofi`, with the grammar, codec and action-set identities and the corpus revision. |
| `states/<ordinal>.pt` | One state block per model component role, in sorted role order. |
| `resources/<ordinal>/` | The construction assets of each declared pretrained role, without weight files. Present only when `pretrained_resources` names a role. |
| `manifest.json` | The format `lofi-hub-model`, the point's seed derivation, the release record, and the SHA-256 digest of every other file, with each component's tag, declaration fingerprint and the shape and dtype of each state value. |

The model card's reproducibility block takes `code_uri`, `command` and
`environment` from the declaration, the source revision from the point's
manifest, and the random seed from the point's seed root. The grammar, codec
and action-set identities and the corpus revision come from the replanned run.

A pretrained role in `pretrained_resources` must be a component role the run
declares, and its declaration must pin a checkpoint and a revision. Files with
a weight suffix (`.bin`, `.h5`, `.msgpack`, `.onnx`, `.pt`, `.pth`,
`.safetensors`) are not copied from its directory, and a role whose directory
holds no other file is refused.

## Load a bundle

[`load_hub_bundle`](../../reference/run.md) rebuilds the model from a bundle
directory alone:

```python
from pathlib import Path

from lofi.run.hub import load_hub_bundle

loaded = load_hub_bundle(Path("bundles/synthetic-parser"))
print(loaded.release.model_name, loaded.release.grammar_identity)
```

Loading composes `config.json`, plans the run under the recorded seed
derivation, and checks the bundle before it constructs anything: the files
present must be exactly those the manifest lists, each must digest to its
recorded value, the configuration must be the one the plan rebuilds, the four
identities must match, and each component role must carry the recorded tag,
fingerprint, state keys, shapes and dtypes. A bundle that fails any check is
refused with the file or role at fault. Pretrained components are rebuilt from
the bundle's own `resources/` directories, without a network.

## Release a bundle

`lofi release` verifies, without a network, that a bundle reproduces the
checkpoint its run declares for release: it plans the run's recorded point,
restores the newest `resume-latest` checkpoint's validation-best partner and
the declared checkpoint, checks that both belong to one terminal parser run,
and compares the held-out item scores of the bundle and of the restored
checkpoint bitwise. Without `--publish` it prints that preflight and calls no
Hub:

```text
uv run lofi release \
  --bundle bundles/synthetic-parser \
  --repo-id organization/model \
  --run runs/synthetic-parser/runs/<point_id>
```

`--publish` makes the release public only once a reviewer has approved it. It
requires `--evidence`, the file the release evidence is recorded in,
`--reviewer`, `--authorized-license`, which must be the bundle's license
expression, both `--confirm-corpus-redistribution` and
`--confirm-artifact-redistribution`, and a Hub token in the environment
variable `--token-environment` names, `HF_TOKEN` unless given. A publication
reserves its evidence file before its first network call, uploads to a new
private repository, downloads the upload back with credentials and verifies
it, makes the repository public, downloads it again without credentials and
verifies that copy, and only then records the evidence. A failure once the
repository is public returns it to private visibility and records what a
recovery needs.

To serve a bundle's predictions, see
[Serve predictions through bead](serve-through-bead.md).
