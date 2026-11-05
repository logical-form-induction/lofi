# Serve predictions through bead

`lofi bead-predict` scores bead inference items with the trained point a run
directory holds and writes one prediction row per item. `LoFIPredictor` offers the same predictions through bead's
predictor protocol, from a [Hub bundle](export-to-the-hub.md). This guide
prepares the inputs, runs the command, reads the rows, and wraps a bundle for
bead.

bead is installed by the `judgments` extra
([Install LoFI and its extras](../setup/installing.md)).

## What a point must publish

Only a point whose training publishes an inference prediction capability can
be served: one trained on inference pairs with a grounding, whose restored
model predicts a distribution over the declared response levels and over
inference relations. Restoring any other point is refused with "the trained
point publishes no inference prediction capability". Grounding inference
pairs is declared as in
[Reward with a theorem prover](../training/reward-with-a-theorem-prover.md) and
[Reward with model checking](../training/reward-with-model-checking.md).

The point trained in [Inducing logical forms](../../tutorials/05-induce-logical-forms.md)
publishes this capability. The commands below use its run directory. Its label
response orders the levels as entailment, contradiction, and neutral; the serving
encoding must preserve that order.

## Prepare the inputs

The command reads two documents:

- a **serving document**, one JSON object with two fields:
  `response_encoding`, bead's `ResponseEncoding`, which names the response
  labels and so the width of each predicted distribution; and `inputs`, which
  maps each input role the trained point publishes to the rendered element of
  every item that fills it. An inference point publishes the roles `premise`
  and `hypothesis`; a mapping that names any other set of roles is refused,
  naming the roles the point reads. LoFI assumes no element name;
- an **items file**, JSON Lines with one bead `Item` per nonblank line. An
  item rendering a mapped element nowhere is refused, naming the item and the
  elements it renders.

Save the serving document as `serving.json`:

```json
{
  "response_encoding": {
    "name": "inference",
    "n_levels": 3,
    "scale_type": "nominal",
    "labels": ["entailment", "contradiction", "neutral"]
  },
  "inputs": {"premise": "context", "hypothesis": "claim"}
}
```

The same `serving` object is what a saved bead artifact records in its
manifest, so a predictor and the command read one declaration.

Create one item whose rendered elements match the declared mapping:

```python
from pathlib import Path
from uuid import uuid4

from bead.items import Item

item = Item(
    id=uuid4(),
    item_template_id=uuid4(),
    rendered_elements={"context": "john sleeps", "claim": "john sleeps"},
)
Path("items.jsonl").write_text(item.model_dump_json() + "\n")
```

The template identifier joins the item to its template in a bead collection.
This single-item command reads the rendered elements directly.

## Run the command

```bash
uv run lofi bead-predict \
  --run runs/forms/runs/991e01b8d83d0ea2 \
  --serving serving.json \
  --items items.jsonl \
  --out predictions.jsonl
```

The command restores the point from the checkpoint class its run document
declares under `release.checkpoint`, refuses a run directory that records no
completed run, and refuses an `--out` file that already exists.

Every predicted row is checked once, at this boundary, before anything is
written: there must be exactly one row per item, each as wide as the
encoding's level count, each value must be finite and in `[0, 1]`, and each
row must sum to one within `1e-6`. A refused row writes no file.

## Read the rows

Each line of `--out` is one prediction record:

| Field | What it holds |
| --- | --- |
| `item_id` | The bead item's identifier. |
| `probabilities` | One probability per encoding label, keyed by label. |
| `predicted_class` | The label with the highest probability. |
| `confidence` | That label's probability. |

On success the command prints the number of rows written and the `--out`
path. A refused run, input or row is reported on standard error as
`lofi bead-predict: <reason>` and exits with status `2`, as the
[command-line reference](../../reference/cli.md) lists.

## Serve a bundle through bead

Export the same inference point using its release declaration:

```bash
mkdir -p bundles
uv run lofi export \
  --run runs/forms/runs/991e01b8d83d0ea2 \
  --release docs/examples/serving/release.yaml \
  --out bundles/nli-point
```

`LoFIPredictor` adapts this bundle to bead's predictor protocol:

```python
from pathlib import Path

from lofi.run._boundary.bead import decode_serving
from lofi.run.bead import LoFIPredictor

serving = decode_serving(Path("serving.json").read_bytes())
predictor = LoFIPredictor.from_hub_bundle(Path("bundles/nli-point"), serving)
```

| Member | What it does |
| --- | --- |
| `model_name`, `model_version` | The release's model name and its grammar identity. A point whose response levels differ from the encoding's level count, or whose input roles differ from the mapped roles, is refused when the predictor is built. |
| `supported_task_types` | The one bead task type the encoding's scale type maps to. |
| `validate_item_compatibility(item, template)` | Refuses an item whose template has another task type, which fails bead's own item validation, or which renders a mapped element nowhere. |
| `predict_proba(items, participant_ids=None)` | Returns one checked response distribution per item as a NumPy array. Participant identifiers, when given, must be one named identifier per item. |
| `compute_nli(premise, hypothesis)` | Returns `entailment`, `neutral` and `contradiction` probabilities for one pair, and is refused unless the point's relation labels are exactly those three. |
| `save(path)`, `load(path)` | Write and read a bead artifact: a directory holding the bundle under `hub/` beside a `manifest.json` that records the format `lofi-bead-predictor` and the `serving` declaration. |

`save` refuses an existing destination and a destination inside the source
bundle, and writes through a staging directory. `load` refuses a directory
holding anything other than `manifest.json` and `hub/`, then loads the bundle
with every check [`load_hub_bundle`](../../reference/run.md) applies.
