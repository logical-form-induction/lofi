# Checkpoint and resume a run

A trainer's `checkpoints` section declares how often a point writes a
resumable checkpoint and how many it keeps, and the run document's `release`
section declares which checkpoint represents the finished point. `lofi resume`
trains an interrupted point again from the newest resumable checkpoint in its
run directory. This guide declares both sections, reads the checkpoints a point
writes, and resumes a point stopped mid-run. The
[checkpoint reference](../../reference/checkpoints.md) specifies the on-disk
layout and the compatibility relation a resume checks.

## Declare the checkpoint policy

[`docs/examples/models-training/checkpointed-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/models-training/checkpointed-parser.yaml)
trains the synthetic parser for three epochs and checkpoints after each:

```yaml
trainer:
  epochs: 3
  early_stopping: {patience: 3}
  checkpoints:
    interval: 1
    keep: 2
release:
  checkpoint: resume-latest
```

| Field | What it says |
| --- | --- |
| `trainer.checkpoints.interval` | Positive cadence in the trainer's unit of progress: completed epochs for the `parser`, `alternating` and `staged` trainers, completed optimizer steps for `terms` and `distance`. The final position is always written. |
| `trainer.checkpoints.keep` | How many of the newest resumable checkpoints the point keeps. An older one is removed once a newer one is written. |
| `trainer.early_stopping.keep_best` | How many of the newest validation-best checkpoints the point keeps. |
| `release.checkpoint` | `validation-best` or `resume-latest`: the checkpoint kind that export and post-hoc probing restore. |

A parser trainer writes two kinds of checkpoint. A `validation-best`
checkpoint is written when the monitored metric reaches a new best. A
`resume-latest` checkpoint is written after every improving epoch, after the
epoch that stops the run, and every `interval` epochs; recovery reads only
this kind. The `terms` and `distance` trainers write no validation-best
checkpoint, so their runs release `resume-latest`, as the packaged
[`synthetic-distance.yaml`](https://github.com/logical-form-induction/lofi/blob/main/lofi/configs/run/synthetic-distance.yaml)
does.

## Read the checkpoints a point writes

Train the point:

```bash
uv run lofi run \
  --config docs/examples/models-training/checkpointed-parser.yaml \
  --root runs/checkpointed --seed 11 --experiment-seed 5
```

```text
0b721a54def66af5 completed runs/checkpointed/runs/0b721a54def66af5
```

Each checkpoint is a directory under `checkpoints/`, named by a six-digit
ordinal, the run identity and the step. Its `manifest.json` records the step
and, under `fitting`, the checkpoint's kind and the epochs completed:

```python
import json
from pathlib import Path

point = Path("runs/checkpointed/runs/0b721a54def66af5")
for checkpoint in sorted((point / "checkpoints").iterdir()):
    manifest = json.loads((checkpoint / "manifest.json").read_text())
    fitting = manifest["fitting"]
    print(checkpoint.name[:6], manifest["step"], fitting["kind"], fitting["completed_epochs"])
```

```text
000003 2 resume-latest 2
000004 3 validation-best 3
000005 3 resume-latest 3
```

The point kept three checkpoints: the two newest resumable ones,
as `keep: 2` declares, and the one validation-best checkpoint that the
packaged `keep_best: 1` retains. Ordinals keep counting across removals.

## Resume an interrupted point

`lofi resume` takes the same options as `lofi run` and composes the same point,
so pass the same document, overrides and seeds. It differs from `lofi run` in
one respect: it refuses a point that has no run directory under `--root`.

```bash
uv run lofi resume \
  --config docs/examples/models-training/checkpointed-parser.yaml \
  --root runs/none --seed 11 --experiment-seed 5
```

```text
lofi resume: nothing to resume, the point '0b721a54def66af5' has no run directory under runs/none
```

A point whose run completed is served from its directory, by either command,
without training again:

```bash
uv run lofi resume \
  --config docs/examples/models-training/checkpointed-parser.yaml \
  --root runs/checkpointed --seed 11 --experiment-seed 5
```

```text
0b721a54def66af5 completed runs/checkpointed/runs/0b721a54def66af5
```

### Give the point more than one attempt

Every launch of a point is an attempt, recorded in the point's `failure.json`
before training starts, so an attempt whose process is killed still counts.
`--infrastructure-attempts` sets how many attempts a point receives across
every launch, and defaults to the `3` the failure policy declares. A point that has spent its attempts is not
trained again, so declare more than one attempt from the first launch of a
point you expect to interrupt:

```bash
uv run lofi run \
  --config docs/examples/models-training/checkpointed-parser.yaml \
  --set trainer.epochs=40 --set trainer.early_stopping.patience=40 \
  --root runs/interrupted --seed 11 --experiment-seed 5 \
  --infrastructure-attempts 3
```

A process that receives `SIGTERM`, as a scheduler preempting it sends, records
the interrupted attempt and stops. Stopped after two epochs, this point's
`failure.json` reads:

```text
{
  "status": "interrupted",
  "diagnostic": "the process received SIGTERM while executing a point",
  "attempts": 1,
  "seed_derivation": {
    "experiment_root": 5,
    "point_id": "74494234295cd7d7",
    "seed": 11,
    "root": 1866311196
  }
}
```

Resume it with the same options:

```bash
uv run lofi resume \
  --config docs/examples/models-training/checkpointed-parser.yaml \
  --set trainer.epochs=40 --set trainer.early_stopping.patience=40 \
  --root runs/interrupted --seed 11 --experiment-seed 5 \
  --infrastructure-attempts 3
```

```text
74494234295cd7d7 completed runs/interrupted/runs/74494234295cd7d7
```

The resumed attempt restores the newest `resume-latest` checkpoint and trains
the remaining epochs. Its metrics record holds only the epochs it trained,
numbered from where the checkpoint left off, and a completed attempt removes
`failure.json`:

```python
point = Path("runs/interrupted/runs/74494234295cd7d7")
epochs = json.loads((point / "metrics.json").read_text())["fitting"]["epochs"]
print(epochs[0]["epoch"], epochs[-1]["epoch"], len(epochs))
print(json.loads((point / "manifest.json").read_text())["status"])
```

```text
2 39 38
completed
```

Had the point been launched with `--infrastructure-attempts 1`, the interrupted
launch would have spent it, and `lofi resume` would record the point as
`infra-failed` and report it without training it:

```text
74494234295cd7d7 failed runs/interrupted/runs/74494234295cd7d7
```

## How a resume is refused

The point identifier comes from the resolved document, so a resume that
changes any leaf composes a different point, and `lofi resume` reports that it
has no run directory. A resume under another `--experiment-seed` composes the
same point but derives a different training seed, and is refused rather than
mixed into the recorded run:

```bash
uv run lofi resume \
  --config docs/examples/models-training/checkpointed-parser.yaml \
  --root runs/checkpointed --seed 11 --experiment-seed 6
```

```text
lofi resume: the run of the point '0b721a54def66af5' records a seed_derivation other than the experiment root now given, so it is not served or resumed under it
```

Before any live state changes, the restore checks the checkpoint against the
model it is loaded into: its roles, component fingerprints, controller action
set, precision and grounding must agree, and every block's digest must match.
The [checkpoint reference](../../reference/checkpoints.md#compatibility-relation)
lists each key.

## Related

- [Configure a trainer](configure-a-trainer.md) declares the stopping rule
  that decides which checkpoints are validation-best.
- [Run on a cluster](../experiments/run-on-a-cluster.md) launches many points,
  each of which resumes the same way.
- [Command line](../../reference/cli.md) lists every `lofi run` and
  `lofi resume` option.
