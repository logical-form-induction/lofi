# Command line

This page lists every `lofi` subcommand that `lofi.run.build_parser` declares,
with its usage line and each argument's help text.

The console script `lofi` and `python -m lofi.run` run the same entry point.
From a source checkout, write `uv run lofi …`. Every subcommand takes `-h` or
`--help`, which prints the usage and arguments shown here.

```text
usage: lofi [-h]
            {run,resume,task,launch,status,freeze,seed-budget,release,configs,show-config,price,contract,ingest,results,export,bead-predict,probe,publish,active-learn} ...
```

| Subcommand | Help |
| --- | --- |
| [`run`](#run) | train one composed point |
| [`resume`](#resume) | train one composed point again from its run directory |
| [`task`](#task) | train the listed points one scheduler task covers |
| [`launch`](#launch) | launch an experiment's points from its freeze, or as a pilot |
| [`status`](#status) | poll an experiment's store and rewrite its results |
| [`freeze`](#freeze) | tag a clean HEAD as a freeze and record what it froze |
| [`seed-budget`](#seed-budget) | price an experiment's seed count from a pilot's rows before freezing |
| [`release`](#release) | verify a model release locally, or publish it once authorized |
| [`configs`](#configs) | print the packaged configuration groups and their options |
| [`show-config`](#show-config) | print the resolved document with each leaf's source |
| [`price`](#price) | price an experiment from timed training steps of its own stacks |
| [`contract`](#contract) | apply one declared contraction rung to an experiment and record it |
| [`ingest`](#ingest) | accept an experiment's results against its freeze and write them |
| [`results`](#results) | render the tables an experiment declares from rows.parquet |
| [`export`](#export) | bundle a trained point for the Hugging Face Hub, without a network |
| [`bead-predict`](#bead-predict) | score bead inference items with the trained point a run holds |
| [`probe`](#probe) | execute declared probes against the trained point a run directory holds |
| [`publish`](#publish) | publish a trained point's labeled derivations as Layers records |
| [`active-learn`](#active-learn) | run the rounds of one active-learning document |

## Exit status

| Status | Meaning |
| --- | --- |
| `0` | The command succeeded. |
| `1` | A point failed, or a report was refused. |
| `2` | The input was refused. |

A refusal or failure is written to standard error as `lofi <subcommand>:
<reason>`.

## Composition options

`run`, `resume` and `show-config` compose a configuration document. They share
these options:

| Option | Required | Help |
| --- | --- | --- |
| `--config PATH` | yes | the configuration document to compose |
| `--profile NAME` | no | the named profile to apply over it |
| `--set KEY=VALUE` | no | an override applied last; repeatable |
| `--extra PATH` | no | an extra overlay document applied before the overrides; repeatable |
| `--seed N` | no, default `0` | the run seed |

`probe` takes `--set` and `--extra` without `--config` or `--profile`; it
composes the resolved document a run directory holds.

## run

```text
usage: lofi run [-h] --config PATH [--profile NAME] [--set KEY=VALUE] [--extra PATH]
                [--seed N] --experiment-seed N
                [--numerical {final,reseed}]
                [--out-of-memory {retry,halve-batch}]
                [--infrastructure-attempts N] [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| [composition options](#composition-options) | | |
| `--experiment-seed N` | yes | the root seed the point's run seed is derived from |
| `--numerical {final,reseed}` | no, default `final` | the failure policy's numerical handling |
| `--out-of-memory {retry,halve-batch}` | no, default `retry` | the failure policy's out of memory handling |
| `--infrastructure-attempts N` | no, default `3` | the attempts the point receives across launches before an infrastructure failure settles it |
| `--root DIR` | no | the directory run directories live under; the working directory when omitted |

The three failure options are the fields of an experiment's
[`failure`](configuration/index.md) policy, and an option left out takes the
default that policy declares. `--experiment-seed` must be a nonnegative
integer and `--infrastructure-attempts` a positive one.

## resume

```text
usage: lofi resume [-h] --config PATH [--profile NAME] [--set KEY=VALUE]
                   [--extra PATH] [--seed N] --experiment-seed N
                   [--numerical {final,reseed}]
                   [--out-of-memory {retry,halve-batch}]
                   [--infrastructure-attempts N] [--root DIR]
```

`resume` takes the same arguments as [`run`](#run). It exits with status `1`
when the composed point has no run directory under `--root`, and is refused
when the recorded run was seeded under a derivation other than the one
`--experiment-seed` derives for the point. [Checkpoints](checkpoints.md)
specifies what a resume restores.

## task

```text
usage: lofi task [-h] --experiment PATH [--set KEY=VALUE] [--root DIR]
                 points index chunk_size
```

| Argument | Required | Help |
| --- | --- | --- |
| `points` | yes | the points file a submission wrote |
| `index` | yes | the task index, from zero |
| `chunk_size` | yes | the points one task runs |
| `--experiment PATH` | yes | the experiment document the points belong to, relative to the executing checkout; it declares the freeze, the root seed and the failure policy |
| `--set KEY=VALUE` | no | an override of the experiment document, such as launch.kind=array; repeatable |
| `--root DIR` | no | the directory run directories live under; the working directory when omitted |

## launch

```text
usage: lofi launch [-h] --experiment PATH (--freeze PATH | --pilot) [--set KEY=VALUE]
                   [--root DIR] [--dry-run] [--with PATH] --out PATH
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document to launch |
| `--freeze PATH` | one of the two | the FREEZE.json of the freeze the experiment is launched from |
| `--pilot` | one of the two | run the experiment in this process outside any freeze, under pilot/ in the checkout, with its results marked unfrozen |
| `--set KEY=VALUE` | no | an override of the experiment document, such as launch.kind=array; repeatable |
| `--root DIR` | no | the store root an in-process launch runs under; the working directory when omitted, and pilot/ in the checkout for a pilot |
| `--dry-run` | no | write the launcher's submission and submit nothing |
| `--with PATH` | no | an experiment of the same freeze admitted together with this one; repeatable |
| `--out PATH` | yes | where to write the launch record |

A launch exits with status `3` when a declared stop rule fired. A pilot takes
no `--dry-run` or `--with`. A launch whose input is refused, such as an
experiment its freeze does not record, exits with status `2`, and one that
fails, such as a launch to a site over its quota, with status `1`.

## status

```text
usage: lofi status [-h] --experiment PATH [--set KEY=VALUE] [--root DIR]
                   --interval SECONDS --polls N
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document to poll |
| `--set KEY=VALUE` | no | an override of the experiment document, such as launch.kind=array; repeatable |
| `--root DIR` | no | the directory run directories live under; the working directory when omitted |
| `--interval SECONDS` | yes | seconds waited between polls |
| `--polls N` | yes | the most polls made before the loop ends |

A poll exits with status `0` once no point is submitted or running, `1` once a
point's failure is final, `2` when its last poll still finds points submitted
or running, and `3` once a declared stop rule fires.

## freeze

```text
usage: lofi freeze [-h] --number N [--experiment PATH] --readiness PATH [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| `--number N` | yes | the freeze number |
| `--experiment PATH` | no | an experiment document run against the freeze; repeatable |
| `--readiness PATH` | yes | a file with one line for each of FG-1 to FG-9 |
| `--root DIR` | no | the repository root; the working directory when omitted |

## seed-budget

```text
usage: lofi seed-budget [-h] --experiment PATH --pilot DIR [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document whose seed count is priced |
| `--pilot DIR` | yes | the results directory whose rows score the budgeted metric |
| `--root DIR` | no | the repository root the experiment and pilot are read under; the working directory when omitted |

## release

```text
usage: lofi release [-h] --bundle DIR --repo-id OWNER/NAME --run DIR [--publish]
                    [--evidence PATH] [--reviewer NAME] [--authorized-license SPDX]
                    [--confirm-corpus-redistribution]
                    [--confirm-artifact-redistribution] [--token-environment NAME]
```

| Option | Required | Help |
| --- | --- | --- |
| `--bundle DIR` | yes | the offline bundle lofi export wrote |
| `--repo-id OWNER/NAME` | yes | the Hugging Face repository the release is published to |
| `--run DIR` | yes | the run directory whose declared release checkpoint the bundle reproduces |
| `--publish` | no | publish the release; without it the command only verifies it |
| `--evidence PATH` | with `--publish` | where a publication records its evidence |
| `--reviewer NAME` | with `--publish` | the reviewer who authorized publication |
| `--authorized-license SPDX` | with `--publish` | the license expression the reviewer approved |
| `--confirm-corpus-redistribution` | no | confirm the corpus's redistribution terms were reviewed |
| `--confirm-artifact-redistribution` | no | confirm the artifact's redistribution terms were reviewed |
| `--token-environment NAME` | no, default `HF_TOKEN` | the environment variable the Hub token is read from |

## configs

```text
usage: lofi configs [-h]
```

`configs` takes no arguments.

## show-config

```text
usage: lofi show-config [-h] --config PATH [--profile NAME] [--set KEY=VALUE]
                        [--extra PATH] [--seed N]
```

`show-config` takes the [composition options](#composition-options) only.

## price

```text
usage: lofi price [-h] --experiment PATH --device DEVICE --warmup N --repeats N
                  [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document to price |
| `--device DEVICE` | yes | the device every stack is timed on, such as cpu, mps or cuda:0, of the kind each point's device policy declares |
| `--warmup N` | yes | untimed steps each stack runs before its timed ones |
| `--repeats N` | yes | timed steps each stack runs, at least two |
| `--root DIR` | no | the repository root the experiment is read and expanded under and the price is written beneath; the working directory when omitted |

## contract

```text
usage: lofi contract [-h] --experiment PATH --rung RUNG [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document to contract |
| `--rung RUNG` | yes | the declared contraction rung to apply |
| `--root DIR` | no | the repository root the experiment is read and expanded under; the working directory when omitted |

## ingest

```text
usage: lofi ingest [-h] --experiment PATH --freeze PATH [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document whose results are ingested |
| `--freeze PATH` | yes | the FREEZE.json of the freeze the experiment ran under |
| `--root DIR` | no | the repository root the experiment, freeze and results are read under; the working directory when omitted |

## results

```text
usage: lofi results [-h] --experiment PATH [--pilot] [--root DIR]
```

| Option | Required | Help |
| --- | --- | --- |
| `--experiment PATH` | yes | the experiment document whose reports are rendered |
| `--pilot` | no | render the experiment's pilot results, run outside any freeze, from pilot/results/<experiment>/ under the root |
| `--root DIR` | no | the directory run directories live under; the working directory when omitted |

## export

```text
usage: lofi export [-h] --run DIR --release PATH --out DIR
```

| Option | Required | Help |
| --- | --- | --- |
| `--run DIR` | yes | the run directory of the completed point |
| `--release PATH` | yes | the release declaration naming the model card's metadata |
| `--out DIR` | yes | the bundle directory to create, which must not exist |

## bead-predict

```text
usage: lofi bead-predict [-h] --run DIR --serving PATH --items PATH --out PATH
```

| Option | Required | Help |
| --- | --- | --- |
| `--run DIR` | yes | the run directory of the completed point |
| `--serving PATH` | yes | the serving document declaring the response encoding and the element each input role of the point reads |
| `--items PATH` | yes | the bead items JSONL document |
| `--out PATH` | yes | the prediction rows JSONL file to create, which must not exist |

## probe

```text
usage: lofi probe [-h] --run DIR --out PATH [--set KEY=VALUE] [--extra PATH]
```

| Option | Required | Help |
| --- | --- | --- |
| `--run DIR` | yes | the run directory of the completed point |
| `--out PATH` | yes | the probe rows file to create, which must not exist |
| `--set KEY=VALUE` | no | an override applied last; repeatable |
| `--extra PATH` | no | an extra overlay document applied before the overrides; repeatable |

## publish

```text
usage: lofi publish [-h] --run DIR --tokenizations PATH --selection PATH
                    --out DIR
```

| Option | Required | Help |
| --- | --- | --- |
| `--run DIR` | yes | the run directory of the completed point |
| `--tokenizations PATH` | yes | the document joining each item id to its Layers tokenization |
| `--selection PATH` | yes | the document naming the evaluator roles, the split and the formalism |
| `--out DIR` | yes | the records directory to create, which must not exist |

The command writes one `<role>.jsonl` file per selected evaluator role and
prints one line per role: `<role>: <n> records, <m> unlabeled, <path>`. A
pilot's point is refused with status `2`.
[Publish records](../how-to/experiments/publish-records.md) describes both
documents.

## active-learn

```text
usage: lofi active-learn [-h] --out DIR [--seed N] DOCUMENT
```

| Argument | Required | Help |
| --- | --- | --- |
| `DOCUMENT` | yes | the active-learning document |
| `--out DIR` | yes | the directory the run keeps its rounds, manifest and rows under; the same directory resumes a run awaiting responses |
| `--seed N` | no | the run seed |

A run that completes prints `completed <rounds> <DIR>`. A run whose collector
awaits the responses to a batch prints `awaiting-responses <round> <batch>`,
naming the batch file; running the command again once the responses are placed
beside it resumes the run. See
[Acquire labels actively](../how-to/training/acquire-labels-actively.md).
