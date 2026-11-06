# Checkpoints

A checkpoint saves training state at a completed
[optimizer step](train.md). Continuing training requires more than the learned
weights: an [optimizer](configuration/training.md#optimizers) can retain information about earlier updates, and random
streams must resume from their saved positions. The
[checkpoint selection rules](#best-and-resumable-checkpoints) distinguish the
save used for prediction from the save used for recovery.

The [manifest](#manifest-fields) describes the saved objects. Its digests are
content-derived identifiers used to check stored bytes. If rule order changes,
a saved action column could otherwise be interpreted as a different rule;
the [compatibility check](#compatibility-relation) refuses that restoration.

Configured parser training writes checkpoints beneath a [run directory](../how-to/training/resume-from-a-checkpoint.md):

```text
<root>/runs/<point-id>/
├── manifest.json
├── config.resolved.json
├── config.sources.json
├── schema.fingerprint
└── checkpoints/
    └── <ordinal>-<run-id>-step<step>/
        ├── manifest.json
        ├── blocks.json
        ├── roles/
        │   └── <role>.pt
        └── run_state/
            ├── optimizer.pt
            ├── scaler.pt
            └── scheduler.pt
```

`scaler.pt` and `scheduler.pt` are present only when the configured run has
those states. Checkpoint ordinals are six decimal digits and order saves from
oldest to newest. A save is staged and renamed into place only after every file
has been flushed.

## Manifest fields

[`CheckpointManifest`](train.md#lofi.train.CheckpointManifest) is a closed JSON record. Every field not marked optional
is required.

| field | meaning |
| --- | --- |
| `run_id` | Identity of the validated run declaration. |
| `step` | Number of completed [optimizer](configuration/training.md#optimizers) steps. |
| `data_position` | Global count of consumed training occurrences at that closed optimizer step. |
| `code_version` | LoFI distribution version that wrote the checkpoint. |
| `action_set` | Digest of the [controller action set](../how-to/models/configure-a-controller.md), in column order. |
| `components` | Component metadata keyed by structural role. |
| `rng` | Encoded state of each advancing random stream. |
| `precision` | Compute dtype, autocast device type, and [gradient scaling](../how-to/training/choose-a-device-policy.md) state. |
| `grounding` | Optional [grounding](../explanation/inference-as-evidence.md#from-a-decision-to-a-gradient) variant, scorer and source identities, symbolic label agreement, gradient path, proof budget, and step telemetry. |
| `stage` | Optional staged protocol name, active groups, and position. |
| `fitting` | Optional configured fitting progress and checkpoint role. |
| `optimizer` | Optional optimizer state reference. |
| `scaler` | Optional gradient scaler state reference. |
| `scheduler` | Optional [scheduler](configuration/training.md#schedulers) state reference. |

Each entry in `components` contains:

| field | meaning |
| --- | --- |
| `tag` | Registered implementation tag. |
| `spec_fingerprint` | Fingerprint of the role's component declaration. |
| `signature_fingerprint` | Fingerprint of the [symbol signature](../explanation/logical-forms-as-types.md#signatures-and-terms) used to build it. |
| `codec_identity` | Identity of the [codec](terms.md#lofi.terms.TermCodec) indexing saved tensors. |
| `state` | Block key and recorded digest. |
| `labels` | Optional output label to column mapping. |
| `pinned` | Optional [pinned pretrained checkpoint](../how-to/models/choose-a-token-encoder.md), `repository@revision`, a frozen role's parameters are loaded from. |

The checkpoint RNG payload contains the advancing Python and torch CPU streams,
each retained loader generator, and the declared accelerator stream when
present. The [seed policy](run.md#lofi.run.SeedPolicy) those streams were derived from is recorded once, in
the [run directory](../how-to/training/resume-from-a-checkpoint.md)'s `manifest.json`, with exactly `root`, `accelerator`,
`samplers`, `loaders`, and `workers`. Each named stream's seed is the
`blake2b` digest of the root and the stream's family and name, so sampler and
worker names remain in the policy, where their seeds are derived
deterministically. There is
no NumPy random stream or NumPy state in either record.

## Compatibility relation

A checkpoint can be loaded into a live run only when all of these keys agree:

1. The role sets are identical.
2. For every role, `tag`, `spec_fingerprint`, `signature_fingerprint`,
   `codec_identity`, `labels`, and `pinned` are identical.
3. `run_id`, `code_version`, `action_set`, `precision`, and `grounding` are
   identical. Reordering the declared rules reorders the [controller](../how-to/models/configure-a-controller.md)'s actions,
   so a resume after it is refused naming the action set digest.
4. The [optimizer](configuration/training.md#optimizers), scaler, and [scheduler](configuration/training.md#schedulers) have the same presence and the same
   block keys.

The comparison reports every disagreement. `step`, `data_position`, `rng`,
`stage`, `fitting`, and state digests are not compatibility keys. They describe
the saved position and payload after compatibility has established which live
objects may receive it. Integrity and restore preflight check those values
separately.

`blocks.json` must name exactly the manifest's component roles and run state
keys. Every block's bytes must match the digest in that index, and every block
is loaded on CPU with weights-only deserialization. Component state, optimizer
state, scaler state, scheduler state, and random state are all validated before
any live state is mutated. Restoration then applies the prepared operations as
one completed-step recovery.

## Best and resumable checkpoints

Best-validation and resumable checkpoints serve different purposes. As an
illustrative case, suppose epoch three has the best score and epoch four
has a worse score. Prediction can use the saved epoch-three
parameters, while recovery needs the later eligible state to continue from the
work already completed. The two checkpoint kinds record these distinct roles;
the retention rules below determine which saves remain available.

The configured fitter validates once per completed epoch. It aggregates metrics
by item occurrence, does not update the [optimizer](configuration/training.md#optimizers), and does not advance
`data_position`. Training increments `data_position` by the occurrences consumed
at each closed optimizer step.

The fitter writes two logically distinct checkpoint kinds:

- `validation-best` is written only when the monitored metric establishes a new
  global best. Its `fitting.best_checkpoint` is null because the store assigns
  the checkpoint's handle after writing its manifest.
- `resume-latest` is written after every improving epoch, after the epoch that
  stops the run, and every [`trainer.checkpoints.interval`](configuration/training.md#lofi.spec.trainers.CheckpointPolicySpec) completed epochs. It
  names the current global-best handle in `fitting.best_checkpoint` and
  contains the newest state eligible for recovery.

Every run declares which kind represents its released point:

```yaml
release:
  checkpoint: validation-best
```

[Export](cli.md#export) and [post hoc probing](../how-to/probes/run-probes.md) restore the newest checkpoint of that declared
kind. Parser runs may release either `validation-best` or `resume-latest`;
training recovery always uses `resume-latest`. A trainer that writes no
validation-best checkpoint, such as [distance pretraining](../how-to/training/configure-a-trainer.md#pretrain-terms-or-type-distances), must release
`resume-latest`.

After each epoch the store keeps the newest [`trainer.checkpoints.keep`](configuration/training.md#lofi.spec.trainers.CheckpointPolicySpec)
resumable checkpoints and the newest [`trainer.early_stopping.keep_best`](configuration/training.md#lofi.spec.trainers.EarlyStoppingSpec)
validation-best checkpoints, and removes the rest. Distance pretraining writes
a resumable checkpoint every [`trainer.checkpoints.interval`](configuration/training.md#lofi.spec.trainers.CheckpointPolicySpec) completed steps and
at its final step, and keeps the newest [`trainer.checkpoints.keep`](configuration/training.md#lofi.spec.trainers.CheckpointPolicySpec).

## Frozen pretrained weights

A role whose weights are loaded from a [pinned pretrained checkpoint](../how-to/models/choose-a-token-encoder.md), and none
of whose parameters take a gradient, is pinned: its block holds its buffers but
none of its parameters, and its manifest record names the pinned checkpoint in
`pinned`. A resumed run loads the same weights from the pinned checkpoint and
restores the rest of the role's state over them. A role takes no gradient when
it is declared frozen or when `trainer.active_groups` leaves it out, and a role
declared frozen stays frozen even where `trainer.active_groups` names it.

The first finite validation observation establishes the best. Later observations
reset [patience](../how-to/training/configure-a-trainer.md#early-stopping) only on a strict absolute improvement greater than
`minimum_delta` in the configured `min` or `max` direction. Equality at the
boundary is nonimproving. Fitting stops after exactly `patience` consecutive
nonimproving validations.

Production resume scans checkpoint handles from newest to oldest and selects
the first `resume-latest` manifest. It never resumes from `validation-best`.
The saved completed epoch count must imply the saved [optimizer](configuration/training.md#optimizers) step and
`data_position` under the current configured epoch shape.

[`lofi resume`](cli.md#resume) requires the point's [run directory](../how-to/training/resume-from-a-checkpoint.md) to exist. A completed run is
still a cache hit. Every attempt is recorded in the point's `failure.json`
before it starts, so a killed attempt, by `SIGTERM` or `SIGKILL`, spends one of
the point's infrastructure attempts, counted across launches. A killed or
unclosed run executes again only while attempts remain, and then uses the
newest eligible resume checkpoint found in that directory. Once the attempts
are spent, `lofi resume` trains nothing, records the point as `infra-failed`
and reports it `failed`.
