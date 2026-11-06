# Contract an experiment

`lofi contract` applies one rung of an experiment's declared contraction
ladder to the sweep in force and records the amendment in a ledger beside the
experiment document. This guide declares a ladder, applies a rung, reads the
ledger, and lists how a rung is refused. It ends with the reduced-freeze
documents, which are a separate way to run a smaller version of a design.

## Declare the ladder

An experiment's `contraction` block lists the rungs the experiment may
contract through, in ladder order, and names the factorial structure the
ladder reduces. The checked-in
[`docs/examples/running/contracted.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/running/contracted.yaml)
crosses two learning rates with four seeds and declares two rungs:

```yaml
sweep:
  grid:
    - path: optimizer.learning_rate
      values: [0.001, 0.0005]
  seeds: [0, 1, 2, 3]
  expected_points: 8
contraction:
  rungs:
    - {kind: seeds, floor: 2}
    - {kind: fallback}
  factorial: two learning rates crossed with four seeds
```

A ladder declares at least one rung and each kind at most once. Each kind is
an installed plugin in the `lofi.contraction_rung` entry point group, which
lowers the declaration to the sweep transformation it states when the
experiment is read. Six kinds are built in:

| `kind` | Fields | What applying it does |
| --- | --- | --- |
| `unpreregistered-cells` | none | Drops every axis value that realizes no level of a preregistered factor. |
| `resolution-iv` | `generators` | Reduces the two-level grid axes the generators name to the resolution-IV fraction they define, as a [`fractional-factorial` design](choose-a-sweep-design.md#declare-a-fractional-factorial) does. |
| `languages` | `axis`, `order` | Drops the first value in `order` that `axis` still assigns. |
| `corpus-scale` | `axis`, `order` | Drops the first value in `order` that `axis` still assigns. |
| `seeds` | `floor` | Keeps every cell's first `floor` seeds. |
| `fallback` | none | Keeps the points and records that the experiment reports its decision point's fallback claim. |

A `seeds` floor must not exceed the seeds the sweep declares and must not fall
below the experiment's `evaluation.minimum_seeds`; either is refused when the
document is read. The `resolution-iv` generators are checked for resolution
four at the same point.

## Apply a rung

`lofi contract` reads the experiment relative to `--root`, applies the named
rung and writes the ledger. Because the ledger is written beside the
experiment document, try it on a copy:

```bash
mkdir -p contract-demo/docs
cp -R docs/examples contract-demo/docs/
uv run lofi contract \
  --experiment docs/examples/running/contracted.yaml \
  --rung seeds \
  --root contract-demo
```

```text
rung: seeds
dropped: [2, 3]
sweep: aa1be44be74c8b3b -> 9164f00f2f798485
points: 4
ledger: contract-demo/docs/examples/running/contracted.amendments.json
```

`--rung` names a rung by its `kind`. The output gives the values the rung
removed, the sweep identifier before and after, and the point count of the
contracted sweep: two learning rates times two seeds.

## Read the ledger

The ledger is `<name>.amendments.json` beside the experiment document:

```json
{
  "amendments": [
    {
      "rung": "seeds",
      "dropped": [
        2,
        3
      ],
      "previous_sweep_id": "aa1be44be74c8b3b",
      "sweep_id": "9164f00f2f798485",
      "points": 4,
      "date": "2026-11-04"
    }
  ]
}
```

Every later read of the experiment applies the recorded rungs again, in order,
to the declared sweep. A load is refused if the ledger names a rung the ladder
does not declare, applies rungs out of ladder order, or records dropped values
that applying the rung again does not remove. Each amendment must start from
the sweep the one before it produced. A freeze records the sweep identifier
of the declared experiment, and ingestion accepts an amended sweep only when
its first amendment starts from that identifier and its last produces the
sweep the experiment now expands to.

## Fall back

The `fallback` rung changes no point. It is the one rung that may leave the
sweep identifier unchanged, and it is recorded like any other:

```bash
uv run lofi contract \
  --experiment docs/examples/running/contracted.yaml \
  --rung fallback \
  --root contract-demo
```

```text
rung: fallback
dropped: []
sweep: 9164f00f2f798485 -> 9164f00f2f798485
points: 4
ledger: contract-demo/docs/examples/running/contracted.amendments.json
```

The loaded [`Experiment`](../../reference/run.md) reports an applied fallback
through its `falls_back` property, and the claim it falls back to is the
`decision_point.fallback` text.

## Refusals

A rung is refused, and nothing is written, when it cannot apply to the sweep
in force. Applying `seeds` a second time, once the sweep is at its floor:

```text
lofi contract: the experiment contract-demo/docs/examples/running/contracted.yaml is refused: the sweep's 2 seeds are already at the floor of 2
```

Naming a rung the ladder does not declare:

```text
lofi contract: the experiment contract-demo/docs/examples/running/contracted.yaml is refused: its ladder declares no rung 'resolution-iv'
```

Applying a rung that precedes one already applied, here `seeds` after
`fallback`:

```text
lofi contract: the experiment contract-demo/docs/examples/running/contracted.yaml is refused: the rung 'seeds' precedes the applied rung 'fallback'
```

Every rung but `fallback` is also refused when it changes no point of the
sweep in force. A `languages` or `corpus-scale` rung is refused when no value
left in its order is assigned by its axis, and a `resolution-iv` rung when it
names an axis that is not a two-valued grid axis. From Python,
`amend_experiment` applies a rung as the command does; a rung that cannot
apply raises `ContractionError`, which is reported as the `ExperimentError`
shown above.

## Reduced freezes

Several experiment directories hold a `reduced-freeze.yaml` beside their full
designs, for instance
[`experiments/s3/reduced-freeze.yaml`](https://github.com/logical-form-induction/lofi/blob/main/experiments/s3/reduced-freeze.yaml).
A reduced freeze is not produced by `lofi contract`. It is its own experiment
document, with its own preregistration (`reduced-freeze.json`), sweep,
`expected_points` and `freeze_id`, that runs a smaller treatment through the
same production path. It is priced, frozen, launched and ingested as any
experiment is; see [Run and price a sweep](sweeps.md). Contract the full
design when the declared experiment must shrink after its freeze; run the
reduced-freeze document when the smaller comparison is the one declared.

[Experiments and commitments](../../explanation/experiments-and-commitments.md)
explains why contraction is declared before launch.
