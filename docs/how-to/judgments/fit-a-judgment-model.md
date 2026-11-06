# Fit a judgment model

A judgment fit document declares one response model over a registered judgment
design, the inference its posterior is drawn by, and the protocol the
judgments were elicited under; [`JudgmentFitSpec`](../../reference/judgment.md)
validates it. This guide composes the checked-in document, fits it at small
scale through the public fitters, expands it as an experiment, and lists the
rows and files a launched fit records. The
[judgment models explanation](../../explanation/judgment-models.md) says what a
response model assumes.

## The document

[`docs/examples/judgments/known-effects.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/judgments/known-effects.yaml)
fits the synthetic `known-random-effects` design with the structured
variational guide:

```yaml
dataset: known-random-effects
response:
  ResponseDeclaration_responsespec:
    family: {family: cumulative}
    levels: 7
    objective: {objective: conditional}
    predictor:
      - kind: link
        readout: score
        linking: {family: linear}
      - {kind: random_intercept, group: participant}
    identification: {constraint: fixed_discrimination, value: 1.0}
    priors:
      thresholds: {}
inference:
  inference: svi
  guide: structured
  iterations: 1000
  draws: 100
  learning_rate: 0.01
elicitation: {method: participants, version: "1"}
calibration: null
comparison: null
dynamic_range: null
ablation: []
interannotator: null
```

Every field of a fit is required; the four that a fit may leave unused are
written `null` or `[]`.

| Field | What it declares |
| --- | --- |
| `dataset` | The tag of a design registered in the `lofi.judgment_design` entry point group. |
| `response` | The response model, under one constructor key: `ResponseDeclaration_responsespec` for an ordinal `ResponseSpec`, or `ResponseDeclaration_boundedresponsespec` for a `BoundedResponseSpec` over a closed scale. |
| `inference` | `nuts`, `svi` or `map`, below. |
| `elicitation` | `method` (`participants`, `model_prompting` or `model_probability`) and a nonempty protocol `version`. |
| `calibration` | Simulation-based calibration, or `null`. See [Simulate a judgment design](simulate-judgment-designs.md). |
| `comparison` | A held-out split and a fitted model comparison, or `null` for an in-sample fit. |
| `dynamic_range` | A dynamic range gate, read before fitting, or `null`: `readout` names the predictor column of a response link term whose root mass, as log odds against its complement, is read against the registered `criterion`. |
| `ablation` | Table ablation rungs, each with its `rows` (`attribute`, `levels`), its `columns` (`attribute`, `inventory`), width, lift and fit schedule; empty for none. Neither attribute has a default: each names the design factor its levels are recorded under. |
| `interannotator` | An interannotator correlation ceiling, or `null` for none: the `coefficients` reported (`spearman`, `pearson`, or both, in report order), the participant pair bootstrap's `replicates` and its interval `level`. Participants who rated exactly the same items form the pairs. |

The ordinal response model reads:

| Field | What it declares |
| --- | --- |
| `family` | The observation family: `cumulative`, `bounded` or `choice` compile to a program; `rank_consistent` and `metric` compile to none, and a fit refuses them. |
| `levels` | The number of ordered response levels, at least two. |
| `objective` | `conditional`, or `truncated_marginal` with the group it integrates out. |
| `predictor` | The terms whose sum is the latent response: `link`, `factor`, `interaction`, `random_intercept`, `random_slope`, `correlated_random_slope` and `offset`. A `link` term reads one readout column through a family of the `lofi.linking` group: `linear`, `logit`, `morcela` or `probability`. |
| `identification` | What fixes the latent scale: `fixed_discrimination`, `fixed_cutpoint_variance` or `identity_link_on_logit`. |
| `priors` | The priors of the family's drawn parameters; `thresholds: {}` takes the default base and increment priors. |

## Compose and validate it

A fit document is composed by [`SpecLoader`](../../reference/spec.md) under the
`JudgmentFitSpec` schema:

```python
from pathlib import Path

from lofi.judgment import JudgmentFitSpec
from lofi.spec import SpecLoader, load_plugins

load_plugins()
loader = SpecLoader(search_path=())
document = Path("docs/examples/judgments/known-effects.yaml")
fit = loader.compose_traced(document, schema=JudgmentFitSpec).spec
print(fit.dataset, fit.inference.method.value, fit.response.levels)
print(fit.response.readouts, fit.response.groups)
```

```text
known-random-effects structured 7
('score',) ('participant',)
```

The response reads one readout column, `score`, and groups by `participant`.
Validation refuses a fit whose parts disagree. Calibration fits every
replicate by NUTS, so it is refused beside a variational fit, and a guide's
learning rate must be positive:

```python
def refusal(*overrides):
    try:
        loader.compose_traced(
            document, schema=JudgmentFitSpec, overrides=overrides
        ).spec
    except Exception as refused:
        print(type(refused).__name__, "|", str(refused).splitlines()[-1])


refusal(
    'calibration={"replicates": 2, "guide_iterations": 20, "guide_learning_rate": 0.01}'
)
refusal("inference.learning_rate=0")
```

```text
ConfigDocumentError | docs/examples/judgments/known-effects.yaml cannot be composed: calibration fits every replicate by NUTS under the fit's NUTS plan, and this fit declares 'structured'
ConfigValueError | inference.learning_rate was refused: a guide's learning rate is positive, not 0.0
```

A link term naming a linking family that is not installed, a family that
compiles to no program, and a sampled fit under the truncated marginal
objective are refused the same way. The dataset tag is resolved only when the
fit is executed.

## Choose the inference

`inference` is a tagged union on its `inference` key:

| `inference` | Fields | Records |
| --- | --- | --- |
| `nuts` | `warmup` (at least 1), `retained` (at least 4 per chain), `chains` (at least 2). | `divergences`, `max-rhat` and `min-ess` rows, and the draws in `posterior.nc`. |
| `svi` | `guide` (`structured` or `mean_field`, from the `lofi.guide` group), `iterations`, `draws`, `learning_rate`. | An `evidence-bound` row, and the guide's draws in `posterior.nc`. |
| `map` | `optimizer` (an installed optimizer declaration), `steps`, `initial_scale`. | A `map-log-posterior` row, and the point in `point-estimate.safetensors`. |

No seed is declared. A fit run as an experiment point takes each method's seed
from its own sampler stream of the point's run seed (`nuts`, `guide`, `map`).
Only MAP may declare the truncated marginal objective, since NUTS and the
guides draw every site, the group-local block among them.

## Fit it at small scale

The fitters a launched point uses are public, so a fit can be checked before
an experiment is declared. Read the design through its registered reader,
compile the response over it, and fit the guide:

```python
from lofi.judgment import QuiversGuideFitter, ResponseCompiler
from lofi.judgment.synthetic_designs import KNOWN_RANDOM_EFFECTS
from lofi.run.fits import RegisteredDesigns

read = RegisteredDesigns().read(fit.dataset)
program = ResponseCompiler.installed().compile(fit.response).over(read.design)

inference = fit.inference
fitted = QuiversGuideFitter(
    inference.guide,
    iterations=inference.iterations,
    draws=inference.draws,
    learning_rate=inference.learning_rate,
    seed=0,
).fit(program)
print(fitted.draws.shape, round(fitted.evidence_bound, 1))

truth = program.sites(KNOWN_RANDOM_EFFECTS.generating())
estimate = program.sites(fitted.draws[0].mean(dim=0))
for site in ("score_slope", "participant_intercept_scale", "cutpoint_base"):
    print(site, round(float(truth[site][0]), 3), round(float(estimate[site][0]), 3))
```

```text
torch.Size([1, 100, 295]) -2016.0
score_slope 1.0 1.021
participant_intercept_scale 0.8 1.19
cutpoint_base -2.5 -1.367
```

The draws are one chain of 100 over the program's 295 parameters. Because the
design's responses were drawn at a known vector, each site's posterior mean is
read against its generating value: here the slope is close to its value and
the intercept scale and cutpoint base are not, at this budget. A fitting
failure, such as a nonfinite bound at any step, raises
`PosteriorFittingError`. `QuiversNutsFitter(NutsPlan(...))` and
`MapFitter(plan, seed=...)` fit the other two methods over the same program.

## Run it as an experiment

A fit enters an experiment through an experiment document whose `config` has
kind `judgment-fit`.
[`docs/examples/judgments/experiment.yaml`](https://github.com/logical-form-induction/lofi/blob/main/docs/examples/judgments/experiment.yaml)
crosses the two guides with three seeds and reports the evidence bound:

```yaml
config:
  kind: judgment-fit
  path: known-effects.yaml
sweep:
  grid:
    - path: inference.guide
      values: [structured, mean_field]
  seeds: [0, 1, 2]
  expected_points: 6
rows:
  - {evaluator: fit, metric: evidence-bound}
```

A fit's rows are reported under the evaluator `fit`, and the experiment may
name `divergences`, `max-rhat`, `min-ess` or `evidence-bound` among its rows.
Read and expand it:

```python
from lofi.run import expand_experiment, read_experiment

experiment = read_experiment(Path("docs/examples/judgments/experiment.yaml"))
points = expand_experiment(experiment, Path.cwd())
print(experiment.config.kind, len(points))
for point in points:
    print(point.point_id, point.seed, point.config["inference"]["guide"])
```

```text
judgment-fit 6
56eb3b8890a05099 0 structured
5724a27b40ff72cf 1 structured
ce1fb2b58480bd89 2 structured
b699565f1a2167c0 0 mean_field
040c44caf4867550 1 mean_field
96bce03fd782bff3 2 mean_field
```

Each point holds one fully resolved fit. A judgment fit experiment is launched
from its freeze like any other, with `lofi launch`; it is fitted in the
launching process, so only a launcher that submits nothing to a site runs it,
and `lofi price` refuses it. [Run and price a sweep](../experiments/sweeps.md) covers the
freeze, the launch and ingestion.

Before a point's directory is claimed, its fit is validated and compiled once
per launch. A program that draws no parameter is refused, a calibrated fit
whose thresholds do not vary over participants is refused, and a declared
`dynamic_range` gate is read against the design and refused if it does not
pass; a passing gate is recorded as the point's `dynamic-range` row, which
names the gated `readout`.

## What a fitted point records

A completed point's directory holds its rows, its draws or point, and a
judgment fit manifest that records the fit, the design's provenance (tag, seed,
counts and digests), the seed derivation, the artifact's digest, the package
versions and the source revision. Its rows are:

| Metric | One row per | Present when |
| --- | --- | --- |
| `divergences`, `max-rhat`, `min-ess` | fit | NUTS |
| `evidence-bound` | fit | a variational guide |
| `map-log-posterior` | fit, with its `objective` | MAP |
| `response-prediction` | design row, with its `participant`, `item`, `response`, predicted `distribution` and `soft_label` | the design is ordinal |
| `split-half-ceiling` | fit, with its interval | the design is ordinal |
| `interannotator-spearman`, `interannotator-pearson` | declared coefficient, with its interval and its participant pair count as `n`; undefined with `n` zero when fewer than two same-list pairs exist | `interannotator` is declared |
| `held-out-participant-fraction` | held-out fold, with its `fold` and `held_out_unit` | `comparison` is declared |
| calibration records | record of the calibration report and replicate | `calibration` is declared |

Every fit row states `budget_kind` as `none`, since a posterior fit enumerates
nothing. A sampler that reports a nonfinite position, draw or diagnostic, or a
calibration whose cutpoint recovery falls below the preregistered floor, is a
numerical failure and is recorded rather than refitted.

## Hold out participants

A `comparison` declares how the design is split and how fitted models are
compared:

| Field | What it declares |
| --- | --- |
| `held_out_unit` | `response`, `participant`, `item`, or a factor column; every row of one level stays in one fold. |
| `readout_units` | The grain of every readout the response reads, each named exactly once. |
| `splitter`, `folds`, `seed` | A splitter of the `lofi.splitter` group (`kfold` or `leave-one-group-out`), its fold count and seed. |
| `coverage_threshold` | The largest share of a fold's held-out participants that the retained rows may never show without a declared policy. It decides only whether a policy is required. |
| `unseen_participant` | `marginal`, `plug_in`, or `null`: how a participant the retained rows never show is predicted, in every fold. |
| `models` | The candidate set, this fit's `candidate`, the `psis_loo` criterion, the weighting and the Pareto gate. |

A fold above the coverage threshold with no `unseen_participant` policy is
refused. A declared policy predicts every unseen participant of every fold,
whether or not that fold's new-participant share is above the threshold. A MAP candidate is refitted on each fold's retained rows, and its
held-out rows are split into the `seen-participant` and `new-participant`
regimes: each regime records one `held-out-predictive-log-density` row per
held-out participant, the mean log density of that participant's held-out
responses, with its `fold`, `participant_regime`, `participant` and
`candidate`. When no policy is declared, the new participants of a fold within
the threshold are not predicted and are recorded as one
`held-out-unpredicted-participants` row instead. Posterior candidates are
compared through their declared PSIS criterion. Either comparison is assembled
when an experiment's results are reduced, by
`lofi.run.fits.declared_model_comparison_rows`, once every candidate in the set
has completed and its artifact matches its manifest's digest. A statistic
pooled over the folds and rungs of a cross-fitted comparison is keyed by a
[`PooledJudgment`](../../reference/eval.md) in place of one fold and rung.

## Campaign scripts

The analyses under `scripts/judgment/` read a judgment fit experiment's results
or run a declared campaign, and each writes a fresh output directory. Each
takes `--help`:

| Script | What it computes |
| --- | --- |
| `factor_effects.py` | Fits declared factor effect rungs and a shuffled-factor control by held-out NUTS, from one declaration. |
| `marginal_likelihood.py` | Compares the conditional and truncated marginal objectives on `known-random-effects` from an experiment's results and run store. |
| `monotonicity.py` | Tests whether one root scalar suffices for cross-fitted acceptability fits, from one analysis document. |
| `morcela_pilot.py` | Reports a MORCELA-linked acceptability candidate against the agreement ceiling. |
| `simulation_calibration.py` | Runs simulation-based calibration of the reference specification; see [Simulate a judgment design](simulate-judgment-designs.md). |

The corpus-backed designs read their releases from a data root, laid out as
[Lay out judgment corpora](judgment-corpora.md) describes.
