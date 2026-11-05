# Run the packaged diagnostics

A `diagnostics` probe runs every diagnostic of the `lofi.diagnostic` catalog
whose required roles a held-out pass reaches, over the points of the roles it
declares. This guide declares the probe, restricts it to chosen roles, reads
what each diagnostic reports, and lists the catalog. Executing a probe and
reading its rows in general is covered in [Probe a trained point](run-probes.md).

## Declare the probe

With no `roles`, the probe reads every role the trained point publishes:

```yaml
probes:
  diag:
    kind: diagnostics
```

[`docs/examples/probes/probed-parser.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/probes/probed-parser.yaml)
declares it on the synthetic parser, and
[Probe a trained point](run-probes.md) trains that run and lists its rows.

`roles` restricts the probe to the points of the named `Role` members. Probe
the trained run again over two roles:

```bash
uv run lofi probe --run runs/probes/runs/d976a68761a68207 \
  --out runs/probes/marginals.json \
  --set 'probes.diag.roles=["CONSTITUENT_MARGINAL", "SPLIT_DISTRIBUTION"]'
```

```text
11 runs/probes/marginals.json
```

```python
from pathlib import Path

from lofi.run.probing import read_probe_rows

for row in read_probe_rows(Path("runs/probes/marginals.json")):
    print(row.measure, row.site, row.value, row.count)
```

```text
finite-readings/live parser/CONSTITUENT_MARGINAL/mu 6.0 1
finite-readings/live parser/SPLIT_DISTRIBUTION/alpha 4.0 1
finite-readings/log-zero parser/SPLIT_DISTRIBUTION/alpha 0.0 1
finite-readings/nonfinite parser/CONSTITUENT_MARGINAL/mu 0.0 1
finite-readings/nonfinite parser/SPLIT_DISTRIBUTION/alpha 0.0 1
finite-readings/nonfinite-share parser/CONSTITUENT_MARGINAL/mu 0.0 1
finite-readings/nonfinite-share parser/SPLIT_DISTRIBUTION/alpha 0.0 1
span-length-bias/normalized-span-length/uniform parser/CONSTITUENT_MARGINAL/mu 0.6666666865348816 1
span-length-bias/normalized-span-length/weighted parser/CONSTITUENT_MARGINAL/mu 0.6666666865348816 1
span-length-bias/spans parser/CONSTITUENT_MARGINAL/mu 2.0 1
cutpoint-entropy/refused None 1.0 1
```

Three diagnostics were selected: `finite-readings`, which requires no role,
`span-length-bias`, which requires `CONSTITUENT_MARGINAL`, and
`cutpoint-entropy`, which requires `SPLIT_DISTRIBUTION`. `split-posterior-kl`
also requires `INSIDE_SCORE` and `OUTSIDE_SCORE`, which the declaration leaves
out, so it is not run. A role name that is not a `Role` member is refused when
the document is validated, and a role no component publishes is refused when
the probe is composed.

## How the rows are formed

Each held-out pass is run once with every observable point of the selected
roles read. A measurement is reported as its mean over the passes that took it,
and `count` is the number of those passes. Where a pass reads exactly one
`CONSTITUENT_MARGINAL` point, its marginals weight the spans, and a quantity a
diagnostic aggregates over spans is reported twice: `…/weighted` under that
weighting and `…/uniform` under equal weights.

A diagnostic that refuses a pass's readings is not dropped. Its refusals are
counted under `<tag>/refused`, with the number of refused passes as the value;
`cutpoint-entropy` above refused the one pass it was selected for. Each
diagnostic's docstring in the [probe reference](../../reference/probe.md) lists
the conditions under which it refuses, such as `cutpoint-entropy` when no
eligible span remains.

## The catalog

Each diagnostic declares the roles it requires and the scores it reports:

| Tag | Requires | Measures |
| --- | --- | --- |
| `finite-readings` | none | The live positions of each reading that hold no finite number. |
| `span-length-bias` | `CONSTITUENT_MARGINAL` | Normalized proper-internal span length under the observed marginals. |
| `cutpoint-entropy` | `SPLIT_DISTRIBUTION` | Normalized entropy over non-forced, non-root split choices. |
| `split-posterior-kl` | `INSIDE_SCORE`, `OUTSIDE_SCORE`, `SPLIT_DISTRIBUTION` | How far DIORA attention lies from its exact split posterior. |
| `action-margin` | `COMBINATOR_LOGITS` | The gap between the two highest action weights, averaged over rows. |
| `controller-choice` | `COMBINATOR_LOGITS` | Confidence, normalized entropy and top-two margin of each controller choice. |
| `controller-relations` | `COMBINATOR_LOGITS` | Declared sequence and similar-input agreement at controller sites. |
| `atomic-complex-differentiation` | `NODE_ENCODING` | How far a term encoder holds atomic and complex nodes apart. |
| `constructor-embedding` | `NODE_ENCODING` | Whether slots carrying one constructor are encoded alike. |
| `depth-graded-encoding` | `NODE_ENCODING` | How a node's likeness to the root above it moves with its depth. |
| `recursion-spread` | `NODE_ENCODING` | How far a term holds its own nodes apart as it takes on more. |
| `superposition-interference` | `TERM_ENCODING` | Angular interference as type encodings are superposed in row order. |
| `gate-patterns` | `GATE` | The first two moments of every observed recurrent gate activation. |
| `hidden-state-evolution` | `RECURRENT_HIDDEN` | Normalized hidden-state displacement from each child to its parent. |
| `memory-cell-dynamics` | `GATE`, `RECURRENT_CELL`, `RECURRENT_HIDDEN` | Displacement and directional retention across memory-cell edges. |
| `recurrent-connection-analysis` | `RECURRENT_HIDDEN` | Directional continuity between observed child and parent states. |
| `attention-patterns` | `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS` | Exact observed attention distributions, without labeling heads. |
| `feed-forward-specialization` | `FFN_PREACTIVATION` | Feature sparsity and declared-group effects in feed-forward preactivations. |
| `residual-contributions` | `ATTENTION_LOGITS`, `ATTENTION_WEIGHTS`, `FFN_PREACTIVATION`, `RESIDUAL_STREAM` | Update and skip norm shares from paired residual stages. |

Which diagnostics a point can run depends on the roles its components publish
and its held-out passes reach. Each component kind has a declared role
inventory (`lofi/probe/ROLES.md`): a parser always publishes `CHART_CELL`, and
may publish the `INSIDE_SCORE`, `OUTSIDE_SCORE` and `SPLIT_DISTRIBUTION`
structure; a controller publishes `COMBINATOR_LOGITS`; a term encoder publishes
`TERM_ENCODING` and `NODE_ENCODING`, with optional attention, residual and
recurrence roles. On the synthetic parser above, the held-out passes reach only
the parser's sites. Print the catalog of the installed package:

```python
from lofi.probe import installed_diagnostics
from lofi.spec import load_plugins

load_plugins()
catalog = installed_diagnostics()
for tag in catalog.tags:
    kind = catalog.kind(tag)
    print(tag, sorted(role.name for role in kind.requires))
```

```text
action-margin ['COMBINATOR_LOGITS']
atomic-complex-differentiation ['NODE_ENCODING']
attention-patterns ['ATTENTION_LOGITS', 'ATTENTION_WEIGHTS']
constructor-embedding ['NODE_ENCODING']
controller-choice ['COMBINATOR_LOGITS']
controller-relations ['COMBINATOR_LOGITS']
cutpoint-entropy ['SPLIT_DISTRIBUTION']
depth-graded-encoding ['NODE_ENCODING']
feed-forward-specialization ['FFN_PREACTIVATION']
finite-readings []
gate-patterns ['GATE']
hidden-state-evolution ['RECURRENT_HIDDEN']
memory-cell-dynamics ['GATE', 'RECURRENT_CELL', 'RECURRENT_HIDDEN']
recurrent-connection-analysis ['RECURRENT_HIDDEN']
recursion-spread ['NODE_ENCODING']
residual-contributions ['ATTENTION_LOGITS', 'ATTENTION_WEIGHTS', 'FFN_PREACTIVATION', 'RESIDUAL_STREAM']
span-length-bias ['CONSTITUENT_MARGINAL']
split-posterior-kl ['INSIDE_SCORE', 'OUTSIDE_SCORE', 'SPLIT_DISTRIBUTION']
superposition-interference ['TERM_ENCODING']
```

## Add a diagnostic

A diagnostic is a class with a `tag`, a frozen set of required `Role`s, a frozen
set of `scores`, and a `run(trace, *, context)` method returning a
`DiagnosticResult`. It is installed by listing the class in the
`lofi.diagnostic` entry-point group under its tag; a class listed under another
tag than the one it declares is refused when the catalog is read.
[Publish plugins](../extending/plugins.md) covers entry points.
