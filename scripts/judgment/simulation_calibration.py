"""Runs simulation-based calibration of the reference judgment specification.

Every replicate draws one parameter vector from the priors of the reference
response specification, samples one response per row of the prior-predictive
reference design at it, and fits the sampled responses by multi-chain NUTS and
by Quivers' structured and mean-field guides over the one compiled program the
responses were sampled from. The retained replicates are then reduced once by
the package's calibration reducer, which reports per-block rank tests, the
interval coverage of all three methods and per-participant cutpoint recovery
against its preregistered floor.

The script writes two files into a fresh output directory. ``manifest.json``
records the design's provenance, the specification and its digest, the digest
of the compiled program every replicate is fitted over, the NUTS and guide
settings, the root seeds, the package versions and source revision, the wall
time and the outcome. It also records each replicate: its simulation seed and
per-method inference seeds, the NUTS fit's per-chain divergences and
acceptance rates and per-site R-hat and effective sample size, each guide's
final evidence lower bound, and the message of each method that failed to
fit it. A replicate some method failed to fit is left out of the reduction.
``rows.jsonl`` holds the reducer's long rows, one JSON object per line, and
is written whenever at least one replicate is retained. Both files are
written before the script exits. It exits with status zero when every NUTS
cutpoint clears the recovery floor and with status one when some cutpoint
misses it. A campaign retaining no replicate writes no rows and exits with
status one.

Run it naming every count, rate and seed the campaign is declared with; the
NUTS warmup, retained draws and chains default to the pinned reference plan::

    python scripts/judgment/simulation_calibration.py --replicates 200
        --guide-iterations "$GUIDE_ITERATIONS" --guide-learning-rate "$RATE"
        --simulation-seed "$SIMULATION" --inference-seed "$INFERENCE"
        --rank-seed "$RANK" --out artifacts/judgment/simulation-calibration
"""

import json
import sys
from argparse import ArgumentParser
from pathlib import Path
from time import perf_counter
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import didactic.api as dx

from lofi.judgment import (
    GUIDE_METHODS,
    REFERENCE_NUTS_PLAN,
    CalibrationCampaign,
    CalibrationSeeds,
    InferenceMethod,
    JudgmentFitCalibration,
    NutsPlan,
    ResponseCompiler,
    ensure_participant_thresholds,
    prior_predictive_calibration,
    spec_digest,
)
from lofi.judgment.synthetic_designs import (
    PRIOR_PREDICTIVE_REFERENCE,
    KnownEffectsDesign,
)
from lofi.run import execution_environment

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.judgment import (
        JudgmentDesign,
        NutsDiagnostics,
        ReplicateRecord,
        ResponseProgram,
        ResponseSpec,
    )

MANIFEST_NAME: Final = "manifest.json"
"""The file recording the campaign's declaration, seeds, environment and outcome."""

ROWS_NAME: Final = "rows.jsonl"
"""The file holding the calibration reducer's long rows."""

METHODS: Final = (
    InferenceMethod.NUTS,
    InferenceMethod.STRUCTURED,
    InferenceMethod.MEAN_FIELD,
)
"""The inference methods every replicate is fitted by, in seed-recording order."""

COMPLETED: Final = "completed"
"""The outcome of a campaign that retained and reduced at least one replicate."""

FAILED: Final = "failed"
"""The outcome of a campaign in which every replicate had a failed fit."""


class ReferenceDesign(NamedTuple):
    """A compiled generating program and the design it is calibrated over."""

    #: The generating specification of the design.
    spec: ResponseSpec

    #: The program compiled from the design's generating specification.
    compiled: ResponseProgram

    #: The design every replicate is simulated over.
    design: JudgmentDesign

    #: The design's provenance as JSON.
    provenance: JsonValue


class CampaignSettings(NamedTuple):
    """The declared counts and seeds of one calibration campaign."""

    #: Prior-predictive replicates.
    replicates: int

    #: Warmup, retained draws and chains of every NUTS fit. Its own seed is
    #: not used, since every inference seed descends from ``seeds``.
    plan: NutsPlan

    #: Optimization steps each guide is fitted for.
    guide_iterations: int

    #: Adam step size each guide is fitted at.
    guide_learning_rate: float

    #: Streams every simulation, inference and rank seed descends from.
    seeds: CalibrationSeeds

    #: Root of the simulation stream, as declared.
    simulation_seed: int

    #: Root of the inference stream, as declared.
    inference_seed: int


def settings_of(
    *,
    replicates: int,
    warmup: int,
    retained: int,
    chains: int,
    guide_iterations: int,
    guide_learning_rate: float,
    simulation_seed: int,
    inference_seed: int,
    rank_seed: int,
) -> CampaignSettings:
    """Return validated campaign settings from declared counts and seeds.

    Parameters
    ----------
    replicates : int
        Prior-predictive replicates.
    warmup : int
        Discarded adaptation iterations per NUTS chain.
    retained : int
        Retained draws per NUTS chain.
    chains : int
        Independent NUTS chains.
    guide_iterations : int
        Optimization steps each guide is fitted for.
    guide_learning_rate : float
        Adam step size each guide is fitted at.
    simulation_seed : int
        Root of every replicate's simulation seed.
    inference_seed : int
        Root of every replicate's per-method inference seed.
    rank_seed : int
        Seed of the reducer's randomized-rank stream.

    Returns
    -------
    CampaignSettings
        The settings the campaign runs under.

    Raises
    ------
    didactic.api.ValidationError
        If fewer than one replicate or guide iteration is declared, or the
        guide learning rate is not positive and finite.
    ValueError
        If the NUTS plan is invalid or a seed is negative.
    """
    calibration = JudgmentFitCalibration(
        replicates=replicates,
        guide_iterations=guide_iterations,
        guide_learning_rate=guide_learning_rate,
    )

    return CampaignSettings(
        replicates=calibration.replicates,
        plan=NutsPlan(
            warmup=warmup, retained=retained, chains=chains, seed=inference_seed
        ),
        guide_iterations=calibration.guide_iterations,
        guide_learning_rate=calibration.guide_learning_rate,
        seeds=CalibrationSeeds(
            simulation=simulation_seed, inference=inference_seed, rank=rank_seed
        ),
        simulation_seed=simulation_seed,
        inference_seed=inference_seed,
    )


def declaration(settings: CampaignSettings) -> dict[str, JsonValue]:
    """Return a campaign's declared counts, settings and seeds as JSON fields.

    Parameters
    ----------
    settings : CampaignSettings
        The campaign's counts and seeds.
    """
    plan = settings.plan

    return {
        "replicates": settings.replicates,
        "nuts": {
            "warmup": plan.warmup,
            "retained": plan.retained,
            "chains": plan.chains,
        },
        "guides": {
            "methods": [method.value for method in METHODS if method in GUIDE_METHODS],
            "iterations": settings.guide_iterations,
            "learning_rate": settings.guide_learning_rate,
            "draws": plan.chains * plan.retained,
        },
        "seeds": {
            "simulation": settings.simulation_seed,
            "inference": settings.inference_seed,
            "rank": settings.seeds.rank,
        },
    }


def nuts_record(diagnostics: NutsDiagnostics | None) -> JsonValue:
    """Return a NUTS fit's diagnostics as JSON, or None if NUTS failed.

    Parameters
    ----------
    diagnostics : NutsDiagnostics | None
        The fit's diagnostics, or None if NUTS failed.
    """
    if diagnostics is None:
        return None

    return {
        "divergences": list(diagnostics.divergences),
        "acceptance_rates": list(diagnostics.acceptance_rates),
        "r_hat": dict(diagnostics.r_hat),
        "effective_sample_sizes": dict(diagnostics.effective_sample_sizes),
        "max_rhat": diagnostics.max_rhat,
        "min_ess": diagnostics.min_ess,
    }


def replicate_record(
    settings: CampaignSettings, record: ReplicateRecord
) -> dict[str, JsonValue]:
    """Return one replicate's seeds, diagnostics and failures as JSON fields.

    Parameters
    ----------
    settings : CampaignSettings
        The campaign's counts and seeds.
    record : ReplicateRecord
        What fitting the replicate by every method produced.
    """
    replicate = record.replicate

    return {
        "replicate": replicate,
        "simulation": record.seed,
        "inference": {
            method.value: settings.seeds.inference(method, replicate)
            for method in METHODS
        },
        "retained": record.retained,
        "nuts": nuts_record(record.nuts),
        "evidence_bounds": {
            method.value: bound for method, bound in record.evidence_bounds.items()
        },
        "failures": {
            method.value: message for method, message in record.failures.items()
        },
    }


def reference_design(source: KnownEffectsDesign, /) -> ReferenceDesign:
    """Return a design's compiled generating program, its rows and its provenance.

    Parameters
    ----------
    source : KnownEffectsDesign
        The synthetic design whose rows every replicate is simulated over.

    Raises
    ------
    ValueError
        If the program's thresholds do not vary over participants.
    """
    read = source.read()
    spec = source.spec
    compiled = ResponseCompiler.installed().compile(spec)
    ensure_participant_thresholds(compiled)

    return ReferenceDesign(
        spec,
        compiled,
        read.design,
        cast("JsonValue", json.loads(read.provenance.model_dump_json())),
    )


def manifest(
    settings: CampaignSettings,
    reference: ReferenceDesign,
    campaign: CalibrationCampaign,
    *,
    elapsed: float,
) -> dict[str, JsonValue]:
    """Return the manifest of one campaign run.

    Parameters
    ----------
    settings : CampaignSettings
        The campaign's counts and seeds.
    reference : ReferenceDesign
        The design the campaign was calibrated over.
    campaign : CalibrationCampaign
        Every replicate's record and the reduction of those retained.
    elapsed : float
        Wall time of the fitting and reduction, in seconds.

    Returns
    -------
    dict[str, JsonValue]
        The design, specification, program identity, settings, replicate
        records, environment and outcome.
    """
    environment = execution_environment()
    spec = reference.spec
    report = campaign.report

    return {
        "design": reference.provenance,
        "specification": cast("JsonValue", json.loads(spec.model_dump_json())),
        "specification_digest": spec_digest(spec),
        "program_digest": reference.compiled.digest,
        **declaration(settings),
        "replicate_records": [
            replicate_record(settings, record) for record in campaign.records
        ],
        "packages": dict(environment.packages),
        "git_revision": environment.git_revision,
        "accelerator": environment.accelerator,
        "outcome": FAILED if report is None else COMPLETED,
        "retained_replicates": 0 if report is None else report.replicates,
        "failed_replicates": [record.replicate for record in campaign.failed],
        "clears_floor": None if report is None else report.clears_floor,
        "elapsed_seconds": elapsed,
    }


def run(
    settings: CampaignSettings,
    out: Path,
    /,
    *,
    source: KnownEffectsDesign = PRIOR_PREDICTIVE_REFERENCE,
) -> int:
    """Run one campaign and write its manifest and, if it reduces, its rows.

    Parameters
    ----------
    settings : CampaignSettings
        The campaign's counts and seeds.
    out : Path
        Directory the manifest and rows are written into, which must not
        already hold a manifest.
    source : KnownEffectsDesign
        The synthetic design whose rows every replicate is simulated over,
        the prior-predictive reference design unless another is given.

    Returns
    -------
    int
        Zero when every NUTS cutpoint of the reduction clears the recovery
        floor, one when some cutpoint misses it or no replicate was retained.

    Raises
    ------
    FileExistsError
        If the directory already holds a manifest.
    """
    destination = out / MANIFEST_NAME

    if destination.exists():
        msg = f"{destination} already records a campaign"
        raise FileExistsError(msg)

    reference = reference_design(source)
    out.mkdir(parents=True, exist_ok=True)
    started = perf_counter()

    campaign = prior_predictive_calibration(
        reference.compiled,
        reference.design,
        replicates=settings.replicates,
        guide_iterations=settings.guide_iterations,
        guide_learning_rate=settings.guide_learning_rate,
        plan=settings.plan,
        seeds=settings.seeds,
    )
    elapsed = perf_counter() - started
    report = campaign.report

    if report is not None:
        lines = (
            json.dumps(row, sort_keys=True, allow_nan=False) for row in report.long_rows
        )
        (out / ROWS_NAME).write_text(
            "".join(f"{line}\n" for line in lines), "utf-8", newline="\n"
        )

    record = manifest(settings, reference, campaign, elapsed=elapsed)
    destination.write_text(
        json.dumps(record, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    if report is None:
        sys.stdout.write(
            f"calibration retained no replicate\n  recorded in {destination}\n"
        )

        return 1

    failed = ", ".join(str(each.replicate) for each in campaign.failed)
    sys.stdout.write(
        f"{report!r}\n"
        f"  replicates {report.replicates}\n"
        f"  failed     {failed or 'none'}\n"
        f"  floor      {'cleared' if report.clears_floor else 'missed'}\n"
        f"  elapsed    {elapsed:.1f} s\n"
        f"  written to {out}\n"
    )

    return 0 if report.clears_floor else 1


def main(argv: list[str] | None = None) -> int:
    """Run the calibration campaign the command line declares.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments without the program name, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Zero when the campaign's reduction clears the recovery floor, one when
        it misses the floor or every replicate had a failed fit.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--replicates",
        type=int,
        required=True,
        help="how many prior-predictive replicates to simulate and fit",
    )
    parser.add_argument(
        "--warmup",
        type=int,
        default=REFERENCE_NUTS_PLAN.warmup,
        help="discarded adaptation iterations per NUTS chain",
    )
    parser.add_argument(
        "--retained",
        type=int,
        default=REFERENCE_NUTS_PLAN.retained,
        help="retained draws per NUTS chain",
    )
    parser.add_argument(
        "--chains",
        type=int,
        default=REFERENCE_NUTS_PLAN.chains,
        help="independent NUTS chains",
    )
    parser.add_argument(
        "--guide-iterations",
        type=int,
        required=True,
        help="optimization steps each variational guide is fitted for",
    )
    parser.add_argument(
        "--guide-learning-rate",
        type=float,
        required=True,
        help="Adam step size each variational guide is fitted at",
    )
    parser.add_argument(
        "--simulation-seed",
        type=int,
        required=True,
        help="root of every replicate's simulation seed",
    )
    parser.add_argument(
        "--inference-seed",
        type=int,
        required=True,
        help="root of every replicate's per-method inference seed",
    )
    parser.add_argument(
        "--rank-seed",
        type=int,
        required=True,
        help="seed of the reducer's randomized-rank stream",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="a directory holding no manifest, where the results are written",
    )
    arguments = parser.parse_args(argv)
    out: Path = arguments.out

    try:
        settings = settings_of(
            replicates=arguments.replicates,
            warmup=arguments.warmup,
            retained=arguments.retained,
            chains=arguments.chains,
            guide_iterations=arguments.guide_iterations,
            guide_learning_rate=arguments.guide_learning_rate,
            simulation_seed=arguments.simulation_seed,
            inference_seed=arguments.inference_seed,
            rank_seed=arguments.rank_seed,
        )

    except (dx.ValidationError, ValueError) as refused:
        parser.error(str(refused))

    return run(settings, out)


if __name__ == "__main__":
    raise SystemExit(main())
