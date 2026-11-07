# Publish records

A publish lens renders one in-memory result as the artifact an external target
validates. Lenses are listed in the `lofi.publisher` [entry point group](../../reference/entry-points.md#registered-groups-and-names), and one
is installed: `layers-records`, which renders a predicted constituency analysis
as a Layers annotation layer record. [`lofi publish`](../../reference/cli.md#publish) renders a trained point's
predictions through it, and the companion group `lofi.prediction_format` lists
readers for released prediction sets, of which one is installed, `lofi-json`.
This guide publishes a trained point's records, renders a record by hand, reads
what it carries, and reads a released prediction set in the declared format.

The analysis a lens renders, `PredictedAnalysis`, the lens itself and
`publish_trained_point` are published by `lofi.run.publishing`, which
`import lofi.run` does not import, since the Layers record models are an
optional dependency.

## Install the record models

The Layers record models come from `lairs`, which the `publish` extra
installs:

```bash
uv sync --extra publish
```
## Publish a trained point

Publishing records converts predictions into a shared annotation format.
The tokenization identifier anchors their offsets to the particular word
sequence used during decoding. The same numerical offsets in another
tokenization could refer to different words, so labels and offsets alone
are insufficient to identify an annotation. [Model export](../serving/export-to-the-hub.md)
instead packages the trained model needed to produce predictions.

[`lofi publish`](../../reference/cli.md#publish) restores a completed point from its declared release [checkpoint](../training/resume-from-a-checkpoint.md#declare-the-checkpoint-policy),
reruns the evaluator roles a selection document names over one retained split,
and labels each item's spans with the categories of the [A*](../../explanation/charts-and-approximation.md#a-widening) derivation the role's
`decode` evaluator selected. A span a unary chain covers takes the category of
the chain's highest node, and each category is written in the grammar's own
syntax.

Save the selection as `selection.yaml`. The document names the roles, the split and the Layers formalism
slug the labels are drawn from:

```yaml
evaluators: [decoded-types]
split: validation
formalism: ccg
```

Save the tokenization document as `tokenizations.yaml`. It joins each item, by its item id, to the Layers
tokenization record its token positions index into:

```yaml
tokenizations:
  - {item: "db61f72dc0e5:5:0", tokenization: 3f6c2a0e-8d7b-4f1e-9c55-2b8d1f0a7e41}
```

Train a point whose decode evaluator declares an A* search. The packaged
[`synthetic-decode-supertag.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-decode-supertag.yaml)
declares one, `decoded-types`; an overlay that gives it a wide beam of
lexical categories to search, `publish-overlay.yaml`, is

```yaml
model:
  readouts:
    lexical-beams:
      kind: terms
      space: categories
      reads: inside
      decoding: {strategy: beam, width: 100}
evaluators:
  decoded-types:
    readout: lexical-beams
```

Train the point and publish it, with every path absolute or relative to the
checkout:

```bash
uv run lofi run --config lofi/configs/run/synthetic-decode-supertag.yaml \
  --extra publish-overlay.yaml --experiment-seed 5 --root store
uv run lofi publish --run store/runs/2a613a14f383d7b3 \
  --tokenizations tokenizations.yaml --selection selection.yaml --out records
```

The command writes `records/<role>.jsonl`, one Layers record per line for each
labeled item of the split in retained order, and prints one line per role:

```text
decoded-types: 1 records, 0 unlabeled, records/decoded-types.jsonl
```

Reading the record back:

```python
import json

layer = json.loads(open("records/decoded-types.jsonl").readline())
print(layer["expression"], layer["tokenizationId"]["value"], layer["metadata"]["tool"])
for annotation in layer["annotations"]:
    print(annotation["uuid"]["value"], annotation["label"])
```

```text
db61f72dc0e5:5:0 3f6c2a0e-8d7b-4f1e-9c55-2b8d1f0a7e41 lofi 0.1.0
0-1 NP
0-2 NP
0-3 S
1-2 NP\NP
2-3 S\NP
```

The labels show the categories the trained point derived: a
point trained for two epochs on three sentences derives the category sequence
$\mathrm{NP},\mathrm{NP}\backslash\mathrm{NP},\mathrm S\backslash\mathrm{NP}$
where the gold derivation reads
$\mathrm{NP}/\mathrm{NP},\mathrm{NP},\mathrm S\backslash\mathrm{NP}$.

Each record carries the item id as its `expression`, the joined tokenization,
the declared formalism, `lofi <version>` as its tool and the terms the item's
corpus declared. An item whose A* search parsed nothing has no derivation and
is counted as unlabeled rather than written. The command refuses, with status
`2` and nothing written:

| Refusal | Raised as |
| --- | --- |
| The point is a pilot's: its binding records the freeze as `unfrozen`. | [`PilotResultsError`](../../reference/run.md#lofi.run.results.PilotResultsError) |
| A role's evaluator labels no derivation: it is not a `decode` evaluator, it declares no `astar` search, or its search parsed none of the split's items. | `UnlabeledDerivationError` |
| A role or split the run does not declare, an item of the split with no tokenization or no recorded provenance, or a run that did not complete. | `ValueError` |
| An existing destination. | `FileExistsError` |
| A selection naming no role or a role twice, or a mapping naming an item twice. | [`ConfigError`](../../reference/spec.md#lofi.spec.ConfigError) |

The same publication is reachable from Python:

```python
from pathlib import Path

from lofi.run.publishing import publish_trained_point

for published in publish_trained_point(
    Path("store/runs/2a613a14f383d7b3"),
    Path("tokenizations.yaml"),
    Path("selection.yaml"),
    Path("records-python"),
):
    print(published.role, published.records, published.unlabeled, published.path)
```

## Render a prediction

A `PredictedAnalysis` holds one item's identifier, the identifier of the
tokenization its spans index into, the predicted labeled spans, the terms the
item's corpus declared, and when the prediction was made. Load the installed
lens through its entry point, give it the producing tool's name and the
formalism its labels are drawn from, and render one analysis:

```python
from datetime import UTC, datetime
from importlib.metadata import entry_points

from lofi.data import LicenseRef, Licensing, Provenance
from lofi.eval import InducedBracketing
from lofi.run.publishing import PredictedAnalysis

publishers = {point.name: point for point in entry_points(group="lofi.publisher")}
lens = publishers["layers-records"].load()("lofi", "ccg")
print(lens.target)

analysis = PredictedAnalysis(
    "expression-1",
    "tokenization-1",
    InducedBracketing(4, {(0, 4): "S", (0, 1): "NP", (1, 4): "S\\NP", (2, 4): "NP"}),
    Provenance(
        "converted-corrected",
        Licensing("CC-BY-4.0", (LicenseRef("CC-BY-4.0", "annotations"),)),
        (),
        None,
    ),
    datetime(2026, 9, 22, 14, 30, tzinfo=UTC),
)
record = lens.render(analysis)
```

```text
PublishTarget('layers-records', 'application/json')
```

`render` returns the record as JSON text. The record is built through the
Layers models themselves, so a record that would not validate is refused
before any text is returned. Rendering opens no network connection, and
rendering the same analysis twice gives the same text.

## Read what the record carries

For a hypothetical four-word item `Ada may like Bea`, half-open span `[1,4)`
covers `may like Bea`: position one is included, position four is excluded.
The span `[2,4)` covers `like Bea` and lies inside `[1,4)`. Parent links record
this containment among the predicted spans. The [span convention](../../explanation/charts-and-approximation.md#spans-and-the-sweep)
uses token positions, as in the parser.

The record's top-level fields:

```python
import json

layer = json.loads(record)
print(
    {
        key: layer[key]
        for key in ("expression", "formalism", "kind", "subkind", "sourceMethod")
    }
)
print(layer["licensing"])
```

```text
{'expression': 'expression-1', 'formalism': 'ccg', 'kind': 'tree', 'subkind': 'constituency', 'sourceMethod': 'automatic'}
{'expression': 'CC-BY-4.0', 'licenses': [{'appliesTo': 'annotations', 'appliesToUri': None, 'attribution': None, 'name': None, 'notes': None, 'spdx': 'CC-BY-4.0', 'spdxUri': None, 'url': None}]}
```

| Record field | Where it comes from |
| --- | --- |
| `expression` | The analysis's item identifier, which joins the record to its source item. |
| `tokenizationId` | The analysis's tokenization identifier. |
| `formalism` | The formalism the lens was built with. |
| `kind`, `subkind` | Always `tree` and `constituency`. |
| `sourceMethod` | Always `automatic`, since a fitted system made the prediction, whatever source method the corpus declared. |
| `licensing`, `eprintRefs`, `reproducibility` | The corpus's declared terms, carried forward unchanged. |
| `createdAt`, `metadata` | The prediction time, and the tool name with the same time. |
| `annotations` | One annotation per predicted span. |

Each annotation is identified by its span as `<start>-<end>`, anchored to the
token indexes it covers, and linked to its parent, the shortest other
predicted span that contains it, and to the spans it immediately contains:

```python
for annotation in layer["annotations"]:
    print(
        annotation["uuid"]["value"],
        annotation["label"],
        annotation["parentId"],
        [child["value"] for child in annotation["childIds"]],
    )
```

```text
0-1 NP {'value': '0-4'} []
0-4 S None ['0-1', '1-4']
1-4 S\NP {'value': '0-4'} ['2-4']
2-4 NP {'value': '1-4'} []
```

The whole sentence, `0-4`, has no parent, and `2-4` hangs under `1-4`, the
shortest containing span between it and `0-4`.
## Read a released prediction set

A null prediction records that the system did not parse the item. An empty
collection of predicted spans can instead be an actual prediction under its
scoring conventions. Preserve that distinction when rescoring, because the
[evaluation conventions](../evaluation/evaluate-a-run.md#declare-the-conventions)
determine how failures and counted constituents affect the denominator.

The `lofi-json` format is one JSON object with a `system` name, the `budget`
the predictions were decoded under (`depth`, `beam` and a `mass` that may be
null), and `items`. Each item names its `item_id`, `split` and `length`, the
token positions of its `punctuation`, its `gold` constituents with their
depths, and its predicted `spans`, which are null for an item the system did
not parse. The checked-in
[`docs/examples/running/released-predictions.json`](../../examples/running/released-predictions.json)
holds one development and one test item. Read it through the installed
format:

```python
from pathlib import Path

formats = {point.name: point for point in entry_points(group="lofi.prediction_format")}
read = formats["lofi-json"].load()

released = Path("docs/examples/running/released-predictions.json").read_text()
predictions = read(released)
print(predictions.system, len(predictions.on("dev")), len(predictions.on("test")))
```

```text
released-parser 1 1
```

The set records the SHA-256 digest of the text it was read from. A set that
is not JSON, does not match the format, names an empty system or releases one
item twice is refused with [`PredictionError`](../../reference/eval.md#lofi.eval.PredictionError):

```python
from lofi.eval import PredictionError

try:
    read(released.replace('"released-parser"', '""'))
except PredictionError as refused:
    print(refused)
```

```text
a released prediction set was refused: a released prediction set's system is empty
```

## Declare the format in an experiment

An experiment reads a released set through its `rescoring` reducer, whose
`prediction_format` names the reader and whose `source` names the file,
resolved below the experiment document's directory:

```yaml
reducers:
  - kind: rescoring
    source: released-predictions.json
    prediction_format: lofi-json
    metric: recall-homogeneity
    # alignments, conventions and register as the reducer guide describes
```

[Reduce and report results](reduce-and-report-results.md) describes the
reducer's other fields. A format no installed plugin registers is refused when
the reducer runs.

## Add a target or a format

A publish lens is any class whose instances have a `target`, a
`PublishTarget` naming the target and the media type it takes, and a
`render` method from the result to the artifact's text. A prediction format is
a function from the released text to [`ReleasedPredictions`](../../reference/eval.md#lofi.eval.ReleasedPredictions). Either is added by
publishing it under the `lofi.publisher` or `lofi.prediction_format` group of
the plugin's own distribution, as the [plugin guide](../extending/plugins.md)
describes.
