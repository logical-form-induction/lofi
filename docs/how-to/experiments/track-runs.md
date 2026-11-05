# Track runs in an experiment tracker

A run document's `sinks` section declares the metric sinks a point reports to.
Each entry is keyed by a role name and names an installed sink by its `kind`,
one of the `lofi.sink` entry points: `mlflow`, `trackio` or `wandb`. This guide
installs the tracker clients, declares an MLflow sink in an overlay, lists each
kind's fields, and says what a sink receives.

## Install the tracker clients

The clients are in the `tracking` extra:

```bash
uv sync --extra tracking
```

A sink is built only when its point is executed, so a document that declares
one composes without the client installed. Executing the point then raises
`MissingTelemetryBackendError`, whose message names the missing package and
the extra to install.

## Declare a sink

A sink only observes a run, so it is natural to keep it out of the run
document and add it per invocation. The checked-in overlay
[`docs/examples/running/mlflow-sink.yaml`](../../examples/running/mlflow-sink.yaml)
declares one MLflow sink under the role `tracker`:

```yaml
sinks:
  tracker:
    kind: mlflow
    experiment_name: synthetic-parser
```

Apply it with `--extra` when the point is trained:

```bash
uv run lofi run \
  --config docs/examples/synthetic-parser.yaml \
  --extra docs/examples/running/mlflow-sink.yaml \
  --root runs/tracked \
  --seed 11 \
  --experiment-seed 5
```

The same declaration as two overrides, composed from Python:

```python
from pathlib import Path

from lofi.spec import RunSpecCodec, config_search_path, load_plugins

load_plugins()
codec = RunSpecCodec(search_path=config_search_path(Path.cwd()))
document = Path("docs/examples/synthetic-parser.yaml")

tracked = codec(
    document,
    overrides=(
        "sinks.tracker.kind=mlflow",
        "sinks.tracker.experiment_name=synthetic-parser",
    ),
)
print(tracked.sinks["tracker"].settings())
```

```text
MlflowSinkSettings(experiment_name='synthetic-parser', run_name=None, tracking_uri=None)
```

The run name and tracking store are left to their defaults, described below.

A document may declare several sinks under different roles, and each is built
and closed once per point. A document that declares none reports to a null
sink that discards everything.

## Choose a kind

| Kind | Required | Optional | Builds |
| --- | --- | --- | --- |
| `mlflow` | `experiment_name` | `run_name`, `tracking_uri` | An `MlflowClient` bound to `tracking_uri`, or to the client's configured store when it is left out. The experiment is created when the store holds none of that name, and one run is created in it. |
| `trackio` | `project` | `run_name`, `space_id`, `server_url` | One Trackio run in the project, written to the hosted Space `space_id`, to the self-hosted `server_url`, or to Trackio's own default when both are left out. |
| `wandb` | `project` | `entity`, `run_name` | One W&B run in the project, owned by `entity` or by the client's default. |

Every kind names its run after the point's sixteen-character identifier unless
`run_name` is given. A supplied name of any kind must be nonempty. Credentials
are not fields of any sink: each client reads its own login or environment, as
its documentation describes.

A Trackio sink takes one remote target. Declaring both is refused when the
document is composed:

```python
from lofi.spec import ConfigValueError

try:
    codec(
        document,
        overrides=(
            "sinks.tracker.kind=trackio",
            "sinks.tracker.project=synthetic-parser",
            "sinks.tracker.space_id=placeholder/space",
            "sinks.tracker.server_url=http://localhost:7860",
        ),
    )
except ConfigValueError as refused:
    print(refused)
```

```text
sinks.tracker was refused: a Trackio sink accepts either space_id or server_url, not both
```

A kind no installed plugin registers is refused in the same way, and the
refusal lists the kinds that are registered:

```python
try:
    codec(document, overrides=("sinks.tracker.kind=tensorboard",))
except ConfigValueError as refused:
    print(refused)
```

```text
sinks.tracker.kind was refused: Unknown variant 'sinks.tracker.kind' = 'tensorboard' (set by override:sinks.tracker.kind="tensorboard"); SinkSpec registers: ['mlflow', 'trackio', 'wandb']
```

A sink that leaves out its required name is refused with
`MissingSettingError`.

## What a sink receives

A sink receives scalars keyed by name at a global optimizer step, and named
tables of rows:

- At each optimizer step, the step's training metrics, such as `loss/<term>`
  for each loss term and the `optimizer/` values of the step itself.
- At each validation, the validation metrics, each prefixed `validation/`.
- When a fitting ends, `fitting/epochs`, `fitting/optimizer_steps`,
  `fitting/data_position` and `fitting/stopped_early`, and
  `fitting/achieved_accuracy` when the trainer reports one.
- After probes run, their rows as one table named `probes`.

MLflow logs a table as a JSON artifact named after the table, Trackio as a
`Table` value, and W&B as a table under its name. The held-out evaluators'
rows are not sent to sinks; they are written to the run directory's
`rows.json`, and an experiment reads them from there.

A point that is served from its run directory instead of being trained builds
no sink, so a tracker holds one run per executed point.

## A sink does not change the point

`sinks` is the run's one observation section: it says where metrics go and
nothing about what is computed, so a point's identifier is computed over every
other section. The document with and without the sink is one point:

```python
from lofi.run import point_id, resolved_document

plain = codec(document)
print(point_id(resolved_document(plain)) == point_id(resolved_document(tracked)))
```

```text
True
```

Thus a sink can be added to a point that has already been trained, and `lofi
probe` accepts an overlay that changes a trained run's sinks along with its
probes and its counterfactual population, while it refuses any other section
that differs from the recorded document.
