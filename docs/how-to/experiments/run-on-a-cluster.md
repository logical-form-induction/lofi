# Run on a cluster

An experiment document's `launch` block names the installed launcher that runs
its points: `serial` trains them in the launching process, `array` submits them
as one scheduler array job, and `remote` runs them one at a time on a host
reached through a transport. The array and remote launchers read every value
they take from a compute site out of one site profile, a TOML file the
environment variable `LOFI_SITE_PROFILE` names. This guide declares a launcher,
writes a site profile from the checked-in templates, checks it, and launches
and polls an experiment with `lofi launch` and `lofi status`.

The steps assume an experiment that has been priced and frozen, as
[Run and price a sweep](sweeps.md) describes. A single development point needs
no launcher: `lofi run` trains it in place.

## Declare the launcher

The `launch` block is a tagged union over the `lofi.launcher` entry-point
group. The checked-in
[`docs/examples/experiments/synthetic.yaml`](../../examples/experiments/synthetic.yaml)
trains in the launching process:

```yaml
launch: {kind: serial}
```

The synthetic experiments under `experiments/s3/` and `experiments/s4/`
declare a scheduler array instead:

```yaml
launch: {kind: array, chunk_size: 1, throttle: 32}
```

| Kind | Fields | Reads a site profile | What it does |
| --- | --- | --- | --- |
| `serial` | none | no | Trains each pending point in the launching process, under the store root `--root` names. |
| `array` | `chunk_size` (default `1`), `throttle` (default none) | yes | Writes a points file and a submission script, then runs the profile's submit command once. Each task trains `chunk_size` consecutive points; `throttle` caps how many tasks run at once, and leaving it out lets every task run. |
| `remote` | none | yes | Writes the same points file and opens one transport session per pending point, in order. A session that exits nonzero marks its point failed and the next point still runs. |

A `chunk_size` or `throttle` below one is refused when the experiment document
is read. An experiment whose configuration kind executes its points only in the
launching process, such as a judgment fit, a baseline fit or an active-learning
run, is refused with `UnlaunchableExperimentError` on a launcher that submits
to a site, before anything is expanded.

`lofi launch` and `lofi status` read the experiment document with the
`--set KEY=VALUE` overrides they are given, the one override syntax every
command takes, so `--set launch.kind=array --set launch.chunk_size=4`
overrides the declaration for one invocation. A launch to a site passes the
same overrides to every task it submits, so a task reads the document the
launch read. Poll with the overrides the launch was given, so a stop rule
cancels the launcher the launch used.

## Write a site profile

A site profile is validated as a [`SiteProfile`](../../reference/run.md). Every
field is required, so a launcher reads no site value from anywhere else, and a
field the profile does not declare is refused. Three templates are checked in,
each holding placeholders only:

| File | What it describes |
| --- | --- |
| [`docs/examples/site-profile.toml`](../../examples/site-profile.toml) | A scheduler array whose directives follow the `--time`/`--mem`/`--gres` spelling, with a container wrapper around the interpreter. |
| `config/sites/array.toml` | A scheduler array whose every directive is spelled with a neutral `--walltime`/`--memory`/`--accelerator` template, with a comment on each field. |
| `config/sites/local-accelerator.toml` | One machine with one accelerator. Its submission and transport values are inert placeholders that keep the profile complete. |

Copy one outside the repository, replace every placeholder with the site's own
values, and keep credentials out of it: the profile names an account to charge
and a host to reach, never a password or a key. Then point the environment at
the copy:

```bash
export LOFI_SITE_PROFILE=/absolute/path/to/site-profile.toml
```

The profile has seven tables:

| Table | Fields | What they say |
| --- | --- | --- |
| `submission` | `command`, `directive_prefix`, `task_index_variable`, ten `*_syntax` templates, `job_prefix` and `cancel_command` | How the scheduler takes a script: the submit command, the text each directive line starts with, the variable the scheduler sets to a task's index, the spelling of each directive, the text every job name starts with, and the argument vector that cancels every job of one name. |
| `placement` | `account`, `partition`, `accelerator`, `accelerators_per_node`, `cpus_per_task` | Where a job runs and what each task takes. With `accelerators_per_node = 0` the accelerator directive is left out. |
| `storage` | `scratch_root`, `cache_root`, `quota`, `artifacts` | The absolute roots run outputs and caches live under, how the scratch root's holdings are measured, and which files of a settled point may be removed to stay under quota. |
| `environment` | `preamble` | Shell lines, such as module loads and exports, run before the interpreter. May be empty. |
| `interpreter` | `command` | The argument vector that runs Python, with any container wrapper spelled out. |
| `transport` | `host`, `command` | The host the remote launcher reaches and the argument vector that opens a session, to which the host and then the remote command are appended. |
| `allocation` | `accelerator_hours`, `share` | The accelerator-hours the site grants and the largest fraction, in $(0, 1]$, one paper's admitted experiments may hold. |

Each `*_syntax` field is a Python format template over a fixed set of names,
and a template that leaves one out or names another is refused:

| Template | Names |
| --- | --- |
| `array_syntax` | `first`, `last` |
| `throttle_syntax` | `throttle`, appended to the formatted array syntax |
| `account_syntax` | `account` |
| `partition_syntax` | `partition` |
| `accelerator_syntax` | `accelerator`, `accelerators_per_node` |
| `walltime_syntax` | `minutes` |
| `memory_syntax` | `megabytes` (mebibytes) |
| `cpus_syntax` | `cpus` |
| `log_syntax` | `directory`, the submission's directory |
| `name_syntax` | `name`, the job name every submission of one experiment under one freeze shares |

`cancel_command` is an argument vector whose words are templates over `name`
alone, at least one of them naming it.

`storage.quota` declares a `command` that prints the site's quota report, a
`used_pattern` regular expression with a group named `used`, the `unit_bytes`
one reported unit stands for, and the `limit_bytes` the scratch root is kept
under. Each `storage.artifacts` entry names a class of files by a glob relative
to a point's directory and the `regeneration_hours` remaking them costs. Files
no class matches are never removed.

## Check the profile

`resolve_site_profile` reads the file the environment names and validates it,
and the profile's `submission` table renders the directives a script would
carry. Render the template's directives for a twelve-task array that may run
four tasks at once:

```python
from lofi.run import resolve_site_profile

profile = resolve_site_profile({"LOFI_SITE_PROFILE": "docs/examples/site-profile.toml"})
submission = profile.submission

for directive in (
    *submission.placement_directives(profile.placement),
    *submission.resource_directives(
        profile.placement,
        minutes=90,
        megabytes=4096,
        directory="/placeholder/scratch/submission",
    ),
    submission.array_directive(first=0, last=11, throttle=4),
):
    print(submission.directive_prefix, directive)

print(profile.allocation.admitted_hours)
```

```text
#PLACEHOLDER --account=placeholder-account
#PLACEHOLDER --partition=placeholder-partition
#PLACEHOLDER --gres=placeholder-accelerator:1
#PLACEHOLDER --time=90
#PLACEHOLDER --mem=4096
#PLACEHOLDER --cpus-per-task=4
#PLACEHOLDER --output=/placeholder/scratch/submission/%A_%a.log
#PLACEHOLDER --array=0-11%4
6000.0
```

The last line is `accelerator_hours` times `share`, the hours one paper's
experiments may hold. A template that leaves out one of its names is refused
with the template and the missing name. Copy the template's submission table,
drop `{last}` from its array syntax, and validate it again:

```python
import didactic.api as dx

from lofi.run import SiteSubmission

fields = submission.model_dump()
fields["array_syntax"] = "--array={first}"

try:
    SiteSubmission.model_validate(fields)
except dx.ValidationError as refused:
    print(refused)
```

```text
1 validation error(s) for SiteSubmission
  array_syntax: the array syntax lacks last [validator_error]
```

An unset `LOFI_SITE_PROFILE` is refused with `MissingSettingError` naming the
variable, a relative `scratch_root` or `cache_root` is refused as not absolute,
and a `used_pattern` without a `used` group is refused before any quota is
read.

## Launch an experiment

`lofi launch` is the one command that launches an experiment. It runs from a
checkout of the freeze the experiment declares and refuses any other tree, as
[Run and price a sweep](sweeps.md) describes. A dry run of an array launch
writes the points file and the script and submits nothing. The following is a
template command:

```text
LOFI_SITE_PROFILE=/absolute/path/to/site-profile.toml \
  uv run lofi launch \
  --experiment experiments/s3/ambiguity.yaml \
  --freeze freezes/1/FREEZE.json \
  --set launch.chunk_size=4 \
  --out launch.json \
  --dry-run
```

The command's options:

| Option | What it sets |
| --- | --- |
| `--experiment` | The experiment document, which must lie inside the executing checkout. |
| `--freeze` | The freeze's `FREEZE.json`, which must list the experiment. |
| `--set KEY=VALUE` | An override of the experiment document, such as `launch.kind=array` or `launch.throttle=8`. Repeatable. |
| `--root` | The store root of a serial launch; the working directory when absent. The array and remote launchers write under the profile's `scratch_root` instead. |
| `--with` | An experiment of the same freeze admitted together with this one. Repeatable. |
| `--dry-run` | Describe what the launcher would submit and stop: an array launch writes its script and points file and prints the task count, the points file and the script, a remote launch prints the sessions it would open, and a serial launch the points it would execute. |
| `--out` | Where the launch record is written. |

Before an array or remote launch submits anything, it does three things with
the profile:

1. It admits the experiment against the allocation: the P90 hours the
   experiment and every `--with` experiment declare are summed and must not
   exceed `allocation.admitted_hours`. A headline experiment is admitted only
   together with its analysis experiment, and the reverse.
2. It runs the quota command, reads the held amount through `used_pattern` and
   `unit_bytes`, and, while it exceeds `limit_bytes`, removes the
   `storage.artifacts` of the experiment's settled points, the class with the
   smallest `regeneration_hours` first. If removal cannot bring the root under
   its limit, the launch is refused with `QuotaExceededError`.
3. For an array, it reads every task's resources off the experiment's tracked
   price record, `sweeps/<sweep-id>/price.json`, and leaves out every point
   whose stack the record predicts above its activation budget. An unpriced
   experiment is refused with a message naming the record it looked for.

A dry run admits the experiment but skips the quota step, so it removes no
artifact.

## What a submission writes

The array launcher writes `points.json` and the submission script under the
store's `sweeps/<id>/` directory beneath the scratch root. The script has, in
order:

- `#!/bin/sh`;
- one line per directive, each starting with `directive_prefix`: the account,
  partition and accelerator directives, then the wall-clock, memory, core and
  log directives, then the job name directive and the array directive;
- `set -eu` and the `environment.preamble` lines;
- one `exec` line that runs `interpreter.command` on
  `-m lofi.run task --root <scratch_root> --experiment <document>`, each of
  the launch's overrides as `--set KEY=VALUE`, then the points file, the
  scheduler's
  task-index variable and the chunk size.

Each task is granted the ninetieth percentile of the slowest pending point's
bootstrapped training seconds times the chunk size, rounded up to whole
minutes, and the largest peak allocation among the pending points' stacks,
rounded up to whole mebibytes. The array directive spans tasks `0` to the task
count less one, and its throttle is the declared one or, when none is
declared, the task count. Task $k$ trains the points with indices $kc\leq i<\min\{(k+1)c,n\}$,
where $c$ is `chunk_size` and $n$ is the number of pending points. Without `--dry-run`, the
launcher runs `submission.command` on the script and refuses a nonzero exit
with `SubmissionError`.

The remote launcher opens one session per point: `transport.command`, the
host, and one shell line that joins the preamble lines and an `exec` of the
same task program with `&&`, given the point's index and a chunk size of one.

`lofi task` is the command each task runs. It reads the experiment document
relative to the checkout its own process imports `lofi` from, refuses to run
unless that checkout is the tree the experiment's freeze tags, and refuses a
points file listed for another experiment or freeze:

```bash
uv run lofi task --help
```

```text
usage: lofi task [-h] --experiment PATH [--set KEY=VALUE] [--root DIR]
                 points index chunk_size

positional arguments:
  points                the points file a submission wrote
  index                 the task index, from zero
  chunk_size            the points one task runs

options:
  -h, --help            show this help message and exit
  --experiment PATH     the experiment document the points belong to, relative
                        to the executing checkout; it declares the freeze and
                        the root seed and the failure policy
  --set KEY=VALUE       an override of the experiment document, such as
                        launch.kind=array; repeatable
  --root DIR            the directory run directories live under; the working
                        directory when omitted
```

A task is never run by hand in normal use: the submission script supplies
every argument, and each point receives the infrastructure attempts the
experiment's `failure` policy declares.

## Poll a scheduled experiment

An array launch returns once the scheduler accepts the job, so its points
finish later. `lofi status` reads the store the tasks write to,
rewrites the experiment's results directory from it, and executes nothing.
Name the same root the tasks were given, the profile's `scratch_root`. The
following is a template command:

```text
uv run lofi status \
  --experiment experiments/s3/ambiguity.yaml \
  --root /absolute/path/to/scratch \
  --interval 300 \
  --polls 288
```

The command polls up to `--polls` times, `--interval` seconds apart, and echoes
each poll's status record on its own line. It exits `0` once no point is
submitted or running, `1` as soon as a point's failure is final, `2` when its
last poll still finds points submitted or running, and `3` as soon as a
declared stop rule fires. [Run and price a sweep](sweeps.md) describes the
results directory each poll rewrites and how to ingest it once it is complete.

When a stop rule fires, the poll also cancels what the experiment's declared
launcher left with the scheduler. Every array submission of one experiment
under one freeze carries the same job name, the profile's `job_prefix`
followed by sixteen hexadecimal digits of a digest over the experiment's name and freeze, through
the profile's `name_syntax` directive, so the poll runs the profile's
`cancel_command` over that name and no further task of the experiment starts.
A cancel command that exits nonzero raises `CancellationError`. The poll reads
the profile from `LOFI_SITE_PROFILE` only when it cancels, and a serial or
remote launch, which runs its points before returning, leaves nothing to
cancel. A serial launch reads the stop rules itself after every point and
executes nothing after the point a rule first fires on.
