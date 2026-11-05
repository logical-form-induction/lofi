# Simulate a judgment design

A simulated judgment design is a crossed rating design whose responses are
drawn from a compiled response program at a known parameter vector, so a fit
to it can be read against the truth. This guide draws the two packaged
synthetic designs, resamples responses from a program, and runs
simulation-based calibration, which checks that a response model's posterior
recovers parameters drawn from its own priors, both through a judgment-fit
document and through the campaign script.

## The packaged designs

The `lofi.judgment_design` entry-point group registers two synthetic designs,
both instances of `KnownEffectsDesign` from
`lofi.judgment.synthetic_designs`:

| Tag | Participants | Items | Levels | Used for |
| --- | --- | --- | --- | --- |
| `known-random-effects` | 40 | 30 | 7 | Comparing likelihood objectives against known participant effects. |
| `prior-predictive-reference` | 20 | 25 | 7 | The rows simulation-based calibration replicates are drawn over. |

Each participant rates every item. Each item carries one `score` readout, and
each participant an intercept and a set of cutpoint offsets. The item scores and
participant offsets are drawn from the design's seed, the population values are
fixed in the module, and the responses are drawn through the program compiled
from `generating_spec(levels)`: a cumulative family over a linear link on
`score` plus a participant random intercept. The population cutpoints divide the
design's declared `latent_range` into as many equal intervals as it has levels,
one cutpoint at each interior boundary, so a design over `levels` levels on
`latent_range=(lower, upper)` places cutpoint `k` at
`lower + k * (upper - lower) / levels`. Both packaged designs declare
`(-2.5, 4.5)`, one unit between consecutive cutpoints.

Draw one and read what its provenance records:

```python
from lofi.judgment.synthetic_designs import KNOWN_RANDOM_EFFECTS, PRIOR_PREDICTIVE_REFERENCE
from lofi.spec import load_plugins

load_plugins()
for design in (KNOWN_RANDOM_EFFECTS, PRIOR_PREDICTIVE_REFERENCE):
    print(design)

read = KNOWN_RANDOM_EFFECTS.read()
print(read.provenance.tag, read.provenance.seed, read.provenance.population_digest[:12])
print(len(read.design.responses), read.design.response_levels)
print(read.design.responses.bincount(minlength=7).tolist())
```

```text
KnownEffectsDesign('known-random-effects', participants=40, items=30, levels=7, seed=20261012, latent_range=(-2.5, 4.5))
KnownEffectsDesign('prior-predictive-reference', participants=20, items=25, levels=7, seed=20261013, latent_range=(-2.5, 4.5))
known-random-effects 20261012 8c9a9146b606
1200 7
[280, 193, 228, 197, 135, 93, 74]
```

The same seed reproduces the same design, and the provenance records the tag,
seed, counts and the digest of the population values, the cutpoints among them, so a fit's manifest
identifies exactly what it was fitted to. A judgment-fit document names either
design by its tag in `dataset`, as
[Fit a judgment model](fit-a-judgment-model.md) does.

## Resample responses from a program

`program()` returns the generating program bound to the drawn design, and
`generating()` the parameter vector the responses were drawn at, laid out as the
program lays out its sites. `sample_responses` draws one level per row at any
vector, and `simulated_design` replaces a design's responses with the drawn
ones:

```python
import torch

from lofi.judgment import sample_responses, simulated_design

program = KNOWN_RANDOM_EFFECTS.program()
truth = KNOWN_RANDOM_EFFECTS.generating()
print(program.size, truth.shape)

generator = torch.Generator().manual_seed(1)
redrawn = simulated_design(
    KNOWN_RANDOM_EFFECTS(), sample_responses(program, truth, generator=generator)
)
print(redrawn.responses.bincount(minlength=7).tolist())
```

```text
295 torch.Size([295])
[291, 200, 228, 194, 126, 85, 76]
```

Sampling and likelihood evaluation share the bound program, its parameter
layout, its row locations and its cutpoints, so a design drawn here is scored
by exactly the model it was drawn from.

## Calibrate a fit

A judgment fit's `calibration` block declares simulation-based calibration.
Each replicate draws a parameter vector from the response model's priors,
samples one response per row of the fit's design at it, and fits the sampled
design three ways over the same compiled program: by NUTS under the fit's own
NUTS plan, and by the structured and mean-field guides.
[`docs/examples/judgments/calibrated.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/judgments/calibrated.yaml)
declares a deliberately small campaign over `prior-predictive-reference`:

```yaml
inference: {inference: nuts, warmup: 5, retained: 8, chains: 2}
calibration: {replicates: 2, guide_iterations: 20, guide_learning_rate: 0.01}
```

| Field | What it declares |
| --- | --- |
| `replicates` | Prior-predictive replicates, at least one. |
| `guide_iterations` | Steps each guide is fitted for, at least one. |
| `guide_learning_rate` | The Adam step size each guide is fitted at, finite and positive. |

Calibration is refused beside any inference but `nuts`, and beside a bounded
response. Before anything is fitted, a calibrated fit's compiled program must
draw one row of thresholds per participant, since cutpoint recovery is read per
participant; `ensure_participant_thresholds` checks it. Run the campaign in
process:

```python
from pathlib import Path

from lofi.judgment import (
    CalibrationSeeds,
    JudgmentFitSpec,
    NutsPlan,
    ResponseCompiler,
    ensure_participant_thresholds,
    prior_predictive_calibration,
)
from lofi.run.fits import RegisteredDesigns
from lofi.spec import SpecLoader

fit = SpecLoader(search_path=()).compose_traced(
    Path("docs/examples/judgments/calibrated.yaml"), schema=JudgmentFitSpec
).spec
compiled = ResponseCompiler.installed().compile(fit.response)
ensure_participant_thresholds(compiled)
design = RegisteredDesigns().read(fit.dataset).design

declared, plan = fit.calibration, fit.inference
campaign = prior_predictive_calibration(
    compiled,
    design,
    replicates=declared.replicates,
    guide_iterations=declared.guide_iterations,
    guide_learning_rate=declared.guide_learning_rate,
    plan=NutsPlan(
        warmup=plan.warmup, retained=plan.retained, chains=plan.chains, seed=1
    ),
    seeds=CalibrationSeeds(simulation=2, inference=3, rank=4),
)
for record in campaign.records:
    bounds = {method.value: round(bound, 1) for method, bound in record.evidence_bounds.items()}
    print(record.replicate, record.retained, dict(record.failures), bounds)
print(campaign.report)
print(campaign.report.clears_floor)
```

```text
0 True {} {'structured': -1382.4, 'mean_field': -1515.4}
1 True {} {'structured': -1356.6, 'mean_field': -1377.1}
CalibrationReport(replicates=2, blocks=7, coverage=21, cutpoints=18)
True
```

The call returns a `CalibrationCampaign`. Its `records` hold one
`ReplicateRecord` per replicate, in order: the replicate's seed, the NUTS
diagnostics or the message NUTS failed with, each guide's final evidence lower
bound or failure message, and whether every method fitted it (`retained`).
Only retained replicates are reduced. `report` is the `CalibrationReport` over
them, or `None` when no replicate was retained, and `failed` lists the records
some method could not fit. The report holds a rank-uniformity test per
parameter block for NUTS, the interval coverage of each method on each block,
and each method's Pearson recovery of every cutpoint across participants
against the preregistered floor; `clears_floor` says whether every NUTS
cutpoint clears it.

Launched as an experiment point, the same campaign takes its simulation,
inference and rank seeds from the point's run seed, records one row per record
of the report and one per replicate, and records a NUTS cutpoint recovery below
the floor as a numerical failure. A campaign that retains no replicate is a
numerical failure as well.

## Run the reference campaign

`scripts/judgment/simulation_calibration.py` runs the same calibration for the
reference specification over `prior-predictive-reference`, outside any
experiment. It takes every count, rate and seed on the command line; the NUTS
plan defaults to the pinned reference plan. A small campaign:

```bash
uv run python scripts/judgment/simulation_calibration.py \
  --replicates 2 --warmup 5 --retained 8 --chains 2 \
  --guide-iterations 20 --guide-learning-rate 0.01 \
  --simulation-seed 1 --inference-seed 2 --rank-seed 3 \
  --out runs/sbc
```

```text
CalibrationReport(replicates=2, blocks=7, coverage=21, cutpoints=18)
  replicates 2
  failed     none
  floor      missed
  elapsed    85.3 s
  written to runs/sbc
```

`runs/sbc/manifest.json` records the design's provenance, the specification
and the digest of the compiled program every replicate is fitted over, the NUTS
and guide settings, the root seeds, every replicate's record, the package
versions, the source revision, the wall time and the outcome.
`runs/sbc/rows.jsonl` holds the report's long rows, one JSON object per line,
each with a `summary` of `sbc`, `sbc_rank_bin`, `coverage` or
`cutpoint_recovery`. Both files are written before the script exits. It exits
with status `0` when every NUTS cutpoint clears the recovery floor, and with `1`
when some cutpoint misses it or no replicate is retained, so the campaign above,
whose floor is missed, exits with `1` after writing its rows and manifest. An
output directory that already holds a manifest is refused. At two replicates
and eight retained draws per chain the report is not evidence of calibration;
the script's own reference invocation runs 200 replicates under the pinned NUTS
plan.
