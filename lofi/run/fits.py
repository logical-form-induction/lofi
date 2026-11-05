"""Execution of the points of a judgment-fit experiment.

A judgment-fit point is one resolved judgment fit. Its dataset is resolved to
a design, its response model is compiled over that design, and its parameters
are inferred by the declared method: NUTS under the declared plan, a
variational guide fitted for the declared steps and then drawn from, or a
support-preserving maximum a posteriori point. Each method takes its seed from
its own sampler stream of the point's run seed. An
executor serves one launch: it reads each dataset once, on the first point
that needs it, and every later point of the launch shares that read, so the
design and the provenance of the bytes it was read from are both taken from
one read; each manifest records the provenance. A fit that declares
calibration is also calibrated over prior-predictive replicates simulated over
its design, each fitted by NUTS and by the structured and mean-field guides,
with every simulation, inference and rank seed derived from the point's run
seed through its named sampler streams. Each point is checked once per launch,
before its directory is claimed or created: its fit is validated and its
response model compiled, a program that draws no parameter is refused, and a
calibrated fit whose response model does not vary its thresholds over
participants is refused. The point's directory is claimed for the experiment's
freeze, then records the rows of the fit's diagnostics and calibration and
then its judgment-fit manifest. A fit gated by a declared dynamic range of one
readout also records the passing distribution report computed before the point was
claimed, so the result preserves the evidence that admitted the fit. A
completed fit is served from disk when it is executed again. A fit whose
sampler reports a numerical condition, or whose
calibration recovers a cutpoint below the preregistered floor, is a numerical
failure, recorded and not run again. A fit interrupted by an infrastructure
failure holds no checkpoint, so each further attempt, within the attempts a
point is given across launches, refits it from scratch at the same seed.

A fit row carries the experiment's freeze, the digest of the point's
configuration, the point's identifier, the run seed and the diagnostic's
value over the retained draws: the divergences, largest R-hat and smallest
effective sample size of a NUTS fit, or the evidence lower bound a guide
reached at its last step. A calibration row carries the same identifying
columns, the number of replicates reduced and one record of the calibration
report. A posterior fit and its calibration enumerate nothing, so every row
declares that no enumeration budget applies.
"""

from datetime import UTC, datetime
from hashlib import file_digest
from math import fsum
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, final

import torch
import xarray as xr
from quivers.diagnostics import to_datatree_from_svi
from safetensors.torch import save_file

from lofi._files import replace_path
from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    ParticipantResponse,
    config_digest,
    interannotator_ceiling,
)
from lofi.eval.elpd import compare_declared_models
from lofi.judgment import (
    COMPLEMENT,
    GUIDE_METHODS,
    PARTICIPANT,
    CalibrationSeeds,
    ConditionalPlan,
    ConditionalPointObjective,
    CorrelatedRandomSlope,
    CoverageGate,
    DesignCodec,
    DynamicRangeGate,
    DynamicRangeReport,
    FitArtifact,
    HeldOutPredictor,
    InferenceMethod,
    InteractionTerm,
    JudgmentFitManifest,
    JudgmentFitSpec,
    JudgmentRow,
    JudgmentTargets,
    LinkTerm,
    MapFitter,
    MapInference,
    MarginalLikelihoodError,
    MarginalPointObjective,
    MarginalPointPredictor,
    NutsInference,
    NutsPlan,
    OrdinalHead,
    PointFit,
    PointFittingError,
    PosteriorFittingError,
    QuiversGuideFitter,
    QuiversNutsFitter,
    RandomSlope,
    ResponsePrediction,
    RowColumns,
    TruncatedMarginalPlan,
    VariationalInference,
    decode_fit_manifest,
    encode_fit_manifest,
    ensure_participant_thresholds,
    fit_rung,
    ladder_of,
    log_likelihood,
    mean_log_density,
    predictive_log_density,
    prior_predictive_calibration,
    response_log_probabilities,
    split_half_ceiling,
    splitter_for,
)
from lofi.registry import Registry
from lofi.run._boundary.experiment import decode_judgment_fit
from lofi.run._boundary.manifest import (
    ManifestStatus,
    RecordedRows,
    encode_seed_derivation,
)
from lofi.run._executor import PointExecutionError, execute_to_outcome
from lofi.run._experiment import FIT_ROLE
from lofi.run._manifest import (
    FIT_MANIFEST_NAME,
    read_rows,
    replace_file,
    write_rows,
)
from lofi.run._store import PointStatus, RunOutcome
from lofi.spec import HeldOutParticipantPolicy, derive_seed
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path

    from xarray import DataTree

    from lofi._types import JsonValue
    from lofi.judgment import (
        BoundedDesign,
        BoundProgram,
        CalibrationCampaign,
        CalibrationReport,
        JudgmentDesign,
        PointObjective,
        PosteriorFit,
        Predictor,
        ReadCorpusDesign,
        ResponseCompiler,
        ResponseDesign,
        ResponseProgram,
        Split,
    )
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._environment import ExecutionEnvironment
    from lofi.run._seed import PointSeed
    from lofi.run._store import ExperimentStore
    from lofi.spec.evaluators.interannotator import InterannotatorSettings

__all__ = [
    "ABLATION_STREAM",
    "DESIGN_FAMILY",
    "GUIDE_STREAM",
    "HELD_OUT_PARTICIPANT_FRACTION_METRIC",
    "HELD_OUT_PREDICTIVE_METRIC",
    "HELD_OUT_UNPREDICTED_METRIC",
    "INFERENCE_STREAM",
    "MAP_STREAM",
    "NEW_PARTICIPANT_REGIME",
    "NUTS_STREAM",
    "POINT_ESTIMATE_NAME",
    "POSTERIOR_NAME",
    "RANK_STREAM",
    "SEEN_PARTICIPANT_REGIME",
    "SIMULATION_STREAM",
    "FitExecutor",
    "JudgmentDesigns",
    "RegisteredDesigns",
    "UnfittedCalibrationError",
    "declared_model_comparison_rows",
    "map_fold_probabilities",
    "map_point_objective",
]

NUTS_STREAM: Final = "nuts"
"""The sampler stream a fit's NUTS plan takes its seed from."""

GUIDE_STREAM: Final = "guide"
MAP_STREAM: Final = "map"
"""The sampler stream initializing a fit's maximum a posteriori point."""

"""The sampler stream a variational fit's guide takes its seed from."""

SIMULATION_STREAM: Final = "calibration-simulation"
"""The sampler stream every calibration replicate's simulation descends from."""

INFERENCE_STREAM: Final = "calibration-inference"
"""The sampler stream every calibration replicate's fits descend from."""

RANK_STREAM: Final = "calibration-rank"
"""The sampler stream the calibration reducer's randomized ranks take."""

ABLATION_STREAM: Final = "table-ablation"
"""The sampler stream initializing a point's declared table-ablation rungs."""

POSTERIOR_NAME: Final = "posterior.nc"
POINT_ESTIMATE_NAME: Final = "point-estimate.safetensors"
"""The safe tensor artifact holding a completed MAP point."""

"""The ArviZ-compatible posterior and pointwise likelihood of a completed fit."""

DESIGN_FAMILY: Final = "judgment_design"
"""The registry family a judgment dataset's design reader registers under."""

HELD_OUT_PREDICTIVE_METRIC: Final = "held-out-predictive-log-density"
"""The metric owned by a complete fitted-model candidate set."""

HELD_OUT_UNPREDICTED_METRIC: Final = "held-out-unpredicted-participants"
"""The metric counting a fold's new participants when no policy is declared."""

HELD_OUT_PARTICIPANT_FRACTION_METRIC: Final = "held-out-participant-fraction"
"""The metric giving the fraction of a fold's held-out participants never retained."""

SEEN_PARTICIPANT_REGIME: Final = "seen-participant"
"""The held-out regime of participants the fold's retained rows show."""

NEW_PARTICIPANT_REGIME: Final = "new-participant"
"""The held-out regime of participants the fold's retained rows never show."""


class UnfittedCalibrationError(RuntimeError):
    """No replicate of a fit's calibration was fitted by every method.

    A calibration reduces only the replicates NUTS and both guides fitted, so
    a campaign retaining none has no calibration to report.
    """


class DesignReader(Protocol):
    """Reads one registered judgment dataset as a design."""

    def read(self) -> ReadCorpusDesign:
        """Return the dataset's design and the provenance of what it was read from."""
        ...


class JudgmentDesigns(Protocol):
    """Resolves the design a judgment fit's dataset identifier names."""

    def read(self, dataset: str, /) -> ReadCorpusDesign:
        """Return the design of a judgment dataset and its provenance.

        Parameters
        ----------
        dataset : str
            The dataset identifier a fit declares.
        """
        ...


@final
class RegisteredDesigns:
    """Resolves a dataset through the reader registered under its identifier.

    A judgment dataset's reader registers under the ``lofi.judgment_design``
    entry-point group, named by the identifier a fit declares.
    """

    __slots__ = ()

    def read(self, dataset: str, /) -> ReadCorpusDesign:
        """Return what the reader registered under the identifier reads.

        Parameters
        ----------
        dataset : str
            The dataset identifier a fit declares.

        Raises
        ------
        UnknownTagError
            If no reader is registered under the identifier.
        """
        return Registry[DesignReader](DESIGN_FAMILY).get(dataset).read()


def _fit_rows(
    values: Mapping[str, JsonValue],
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
    draws: int,
) -> list[Mapping[str, JsonValue]]:
    """Return one row per diagnostic of a completed fit."""
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()

    return [
        {
            "freeze_id": freeze_id,
            "config_hash": config_digest(point.config),
            "point_id": point.point_id,
            "task": FIT_ROLE,
            "metric": metric,
            "value": value,
            "n": draws,
            "seed": seed,
            **approximation,
        }
        for metric, value in values.items()
    ]


def _dynamic_range_row(
    report: DynamicRangeReport,
    readout: str,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> Mapping[str, JsonValue]:
    """Return the report row preserving a fit's dynamic-range gate."""
    criterion = report.criterion

    return {
        "freeze_id": freeze_id,
        "config_hash": config_digest(point.config),
        "point_id": point.point_id,
        "task": FIT_ROLE,
        "metric": "dynamic-range",
        "readout": readout,
        "value": report.interquartile_range,
        "n": report.items,
        "seed": seed,
        "inventory": report.inventory,
        "observed_at": report.observed_at,
        "scale": "log-odds",
        "minimum": report.minimum,
        "lower_quartile": report.lower_quartile,
        "median": report.median,
        "upper_quartile": report.upper_quartile,
        "maximum": report.maximum,
        "ceiling_fraction": report.ceiling_fraction,
        "criterion_registered_at": criterion.registered_at,
        "criterion_minimum_interquartile_range": (
            criterion.minimum_interquartile_range
        ),
        "criterion_ceiling": criterion.ceiling,
        "criterion_maximum_ceiling_fraction": criterion.maximum_ceiling_fraction,
        "passed": report.passed,
        **Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec(),
    }


def _calibration_rows(
    report: CalibrationReport,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> list[Mapping[str, JsonValue]]:
    """Return one row per record of a calibration report."""
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()

    return [
        {
            "freeze_id": freeze_id,
            "config_hash": config_digest(point.config),
            "point_id": point.point_id,
            "n": report.replicates,
            "seed": seed,
            **record,
            **approximation,
        }
        for record in report.long_rows
    ]


def _replicate_rows(
    campaign: CalibrationCampaign,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> list[Mapping[str, JsonValue]]:
    """Return one row per method fitted or failed on each calibration replicate."""
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()
    rows: list[Mapping[str, JsonValue]] = []

    for record in campaign.records:
        nuts = record.nuts
        outcomes: list[tuple[InferenceMethod, Mapping[str, JsonValue]]] = [
            (
                InferenceMethod.NUTS,
                {
                    "divergences": None if nuts is None else sum(nuts.divergences),
                    "max_rhat": None if nuts is None else nuts.max_rhat,
                    "min_ess": None if nuts is None else nuts.min_ess,
                },
            ),
            *(
                (method, {"evidence_bound": record.evidence_bounds.get(method)})
                for method in sorted(GUIDE_METHODS)
            ),
        ]
        rows.extend(
            {
                "freeze_id": freeze_id,
                "config_hash": config_digest(point.config),
                "point_id": point.point_id,
                "n": len(campaign.records),
                "seed": seed,
                "summary": "replicate",
                "replicate": record.replicate,
                "simulation_seed": record.seed,
                "method": method.value,
                "retained": record.retained,
                "failure": record.failures.get(method),
                **diagnostics,
                **approximation,
            }
            for method, diagnostics in outcomes
        )

    return rows


def _unfitted_calibration(campaign: CalibrationCampaign, /) -> str:
    """Return the refusal naming every replicate's failed methods."""
    failures = "; ".join(
        f"replicate {record.replicate}: "
        + ", ".join(
            f"{method.value} ({message})" for method, message in record.failures.items()
        )
        for record in campaign.failed
    )

    return (
        f"no calibration replicate of {len(campaign.records)} was fitted by "
        f"every method: {failures}"
    )


def _flat_draws(program: BoundProgram, posterior: PosteriorFit, /) -> torch.Tensor:
    """Return NUTS draws in the bound program's flat parameter order."""
    sites = posterior.draws

    return torch.cat(
        [
            sites[slot.site].reshape(posterior.chains, posterior.retained, slot.size)
            for slot in program.layout
        ],
        dim=-1,
    )


def _posterior_tree(program: BoundProgram, draws: torch.Tensor, /) -> DataTree:
    """Return one persisted fit with pointwise conditional log likelihood."""
    flattened = draws.reshape(-1, program.size)
    likelihoods = torch.stack(
        [log_likelihood(program, parameters) for parameters in flattened]
    )

    return to_datatree_from_svi(
        {"parameters": flattened},
        observed_data={"response": program.host_data[program.observation.response]},
        log_likelihood={"response": likelihoods},
    )


def _write_posterior(directory: Path, tree: DataTree, /) -> None:
    """Replace a point's persisted posterior atomically."""
    staged = directory / f".{POSTERIOR_NAME}.staged"
    tree.to_netcdf(staged)
    replace_path(staged, directory / POSTERIOR_NAME)


def _artifact(
    directory: Path,
    name: Literal["posterior.nc", "point-estimate.safetensors"],
    /,
) -> FitArtifact:
    """Return the stored fit artifact's content identity."""
    path = directory / name

    with path.open("rb") as artifact:
        digest = file_digest(artifact, "sha256").hexdigest()

    return FitArtifact(name=name, sha256=digest)


def _write_point(directory: Path, fit: PointFit, *, objective: str) -> FitArtifact:
    """Persist one MAP point in the safe tensor representation and identify it."""
    staged = directory / f".{POINT_ESTIMATE_NAME}.staged"
    save_file(
        {"parameters": fit.parameters},
        staged,
        metadata={
            "objective": objective,
            "log_density": repr(fit.log_density),
            "steps": str(fit.steps),
        },
    )
    replace_path(staged, directory / POINT_ESTIMATE_NAME)

    return _artifact(directory, POINT_ESTIMATE_NAME)


def _active_participant_groups(program: BoundProgram, /) -> tuple[int, ...]:
    """Return observed participant indices from a bound program's owned column."""
    columns = tuple(
        name
        for name, factors in program.program.indices.items()
        if factors == (PARTICIPANT,)
    )

    if len(columns) != 1:
        msg = (
            "a marginal program has exactly one observed participant index; "
            f"found {columns!r}"
        )
        raise ValueError(msg)

    return tuple(sorted(set(program.host_data[columns[0]].tolist())))


def map_point_objective(program: BoundProgram, /) -> tuple[PointObjective, str]:
    """Construct the point objective a bound program's likelihood selects.

    Returns
    -------
    tuple[PointObjective, str]
        The objective a MAP fit optimizes and the objective's name.
    """
    match program.program.objective:
        case ConditionalPlan():
            return ConditionalPointObjective(program), "conditional"
        case TruncatedMarginalPlan():
            return (
                MarginalPointObjective(
                    program,
                    active_groups=_active_participant_groups(program),
                ),
                "truncated_marginal",
            )


def _design_codec(design: JudgmentDesign, /) -> DesignCodec:
    """Return the codec owned by an ordinal design's declared columns."""
    return DesignCodec(
        design.response_levels,
        factors=tuple(design.factors),
        readouts=tuple(design.readouts),
        alternatives=design.alternatives_per_row,
    )


def _prediction_rows(
    program: BoundProgram,
    design: JudgmentDesign,
    draws: torch.Tensor,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> list[Mapping[str, JsonValue]]:
    """Return posterior response distributions and their soft-label targets."""
    flattened = draws.reshape(-1, program.size)
    distributions = torch.stack(
        [
            response_log_probabilities(program, parameters).exp()
            for parameters in flattened
        ]
    ).mean(dim=0)
    targets = JudgmentTargets(design)
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()
    rows: list[Mapping[str, JsonValue]] = []

    for target, probabilities in zip(targets.rows, distributions, strict=True):
        prediction = ResponsePrediction(
            participant=target.participant,
            item=target.item,
            response=target.response,
            distribution=tuple(float(value) for value in probabilities),
            convention=None,
            agreement=target.stratum,
        )
        rows.append(
            {
                "freeze_id": freeze_id,
                "config_hash": config_digest(point.config),
                "point_id": point.point_id,
                "task": FIT_ROLE,
                "metric": "response-prediction",
                "value": prediction.log_density,
                "n": 1,
                "seed": seed,
                "participant": prediction.participant,
                "item": prediction.item,
                "response": prediction.response,
                "distribution": list(prediction.distribution),
                "soft_label": list(target.distribution),
                "agreement": prediction.agreement.value,
                "convention": None,
                **approximation,
            }
        )

    return rows


def _ceiling_row(
    design: JudgmentDesign,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> Mapping[str, JsonValue]:
    """Return the split-half agreement ceiling, or an undefined measurement."""
    by_item: list[list[float]] = [[] for _ in design.item_levels]

    for item, response in zip(
        design.items.tolist(), design.responses.tolist(), strict=True
    ):
        by_item[item].append(float(response))

    try:
        ceiling = split_half_ceiling(by_item)
        value: float | None = ceiling.value
        lower: float | None
        upper: float | None
        lower, upper = ceiling.interval
        n = ceiling.responses
    except ValueError:
        value = lower = upper = None
        n = 0

    return {
        "freeze_id": freeze_id,
        "config_hash": config_digest(point.config),
        "point_id": point.point_id,
        "task": FIT_ROLE,
        "metric": "split-half-ceiling",
        "value": value,
        "lower": lower,
        "upper": upper,
        "level": 0.95 if value is not None else None,
        "n": n,
        "seed": seed,
        **Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec(),
    }


def _interannotator_rows(
    design: JudgmentDesign | BoundedDesign,
    settings: InterannotatorSettings,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> list[Mapping[str, JsonValue]]:
    """Return the declared interannotator ceilings, or undefined measurements."""
    participants = design.participant_levels
    items = design.item_levels
    responses = tuple(
        ParticipantResponse(participants[participant], items[item], float(value))
        for participant, item, value in zip(
            design.participants.tolist(),
            design.items.tolist(),
            design.responses.tolist(),
            strict=True,
        )
    )
    bootstrap = Bootstrap(
        unit="participant-pair",
        replicates=settings.replicates,
        seed=derive_seed(seed, "interannotator"),
        level=settings.level,
    )
    common = {
        "freeze_id": freeze_id,
        "config_hash": config_digest(point.config),
        "point_id": point.point_id,
        "task": FIT_ROLE,
        "seed": bootstrap.seed,
        **Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec(),
    }

    try:
        ceilings = interannotator_ceiling(
            responses, coefficients=settings.coefficients, bootstrap=bootstrap
        )
    except ValueError:
        return [
            {
                **common,
                "metric": f"interannotator-{coefficient}",
                "value": None,
                "lower": None,
                "upper": None,
                "level": None,
                "n": 0,
            }
            for coefficient in settings.coefficients
        ]

    rows: list[Mapping[str, JsonValue]] = []

    for ceiling in ceilings:
        interval = ceiling.result.ci
        rows.append(
            {
                **common,
                "metric": f"interannotator-{ceiling.coefficient}",
                "value": ceiling.result.value,
                "lower": None if interval is None else interval.low,
                "upper": None if interval is None else interval.high,
                "level": None if interval is None else interval.level,
                "n": ceiling.pair_count,
            }
        )

    return rows


def _gated_readout(fit: JudgmentFitSpec, /) -> str:
    """Return the readout a fit's declared dynamic-range gate reads."""
    if fit.dynamic_range is None:
        msg = "a fit with a dynamic-range report declares its gate"
        raise RuntimeError(msg)
    return fit.dynamic_range.readout


def _readout_odds(
    readout: str, fit: JudgmentFitSpec, design: ResponseDesign, /
) -> torch.Tensor:
    """Return one root log-odds value of the declared readout per design item."""

    def links(term: Predictor) -> tuple[LinkTerm, ...]:
        match term:
            case LinkTerm():
                return (term,)
            case InteractionTerm(left=left, right=right):
                return (*links(left), *links(right))
            case RandomSlope(term=varying) | CorrelatedRandomSlope(term=varying):
                return links(varying)
            case _:
                return ()

    term = next(
        (
            nested
            for root in fit.response.terms
            for nested in links(root)
            if nested.readout == readout
        ),
        None,
    )

    if term is None:
        msg = (
            f"a dynamic-range gate reads {readout!r}, and no response link term "
            "reads it"
        )
        raise ValueError(msg)

    log_mass = design.readouts[term.readout]
    complement = term.linking.columns.get(COMPLEMENT)
    log_complement = (
        torch.log1p(-torch.exp(log_mass))
        if complement is None
        else design.readouts[complement]
    )
    odds = log_mass - log_complement
    first = tuple(
        next(
            position
            for position, item in enumerate(design.items.tolist())
            if item == index
        )
        for index in range(len(design.item_levels))
    )

    return odds[torch.tensor(first, device=odds.device)]


def _comparison_key(fit: JudgmentFitSpec, point: SweepPoint, /) -> tuple[object, ...]:
    """Return the declared comparison set a point belongs to, without its member."""
    comparison = fit.comparison

    if comparison is None:
        msg = "a model-comparison key needs a comparison declaration"
        raise ValueError(msg)

    models = comparison.models

    return (
        fit.dataset,
        point.seed,
        comparison.held_out_unit,
        tuple(comparison.readout_units.items()),
        comparison.splitter,
        comparison.folds,
        comparison.seed,
        comparison.coverage_threshold,
        models.candidates,
        models.criterion,
        models.weighting,
        models.pareto_fraction,
        models.pareto_threshold,
    )


def declared_model_comparison_rows(
    points: Sequence[SweepPoint], store: ExperimentStore, /
) -> list[Mapping[str, JsonValue]]:
    """Return every complete comparison over verified persisted fit artifacts.

    Each comparison is owned by the experiment results layer: it groups points
    with the same dataset, run seed and candidate inventory. Posterior candidates
    are lowered through their declared PSIS comparison. MAP candidates contribute
    their persisted fold-refit rows only after every candidate's point artifact is
    present and agrees with its manifest digest. An incomplete set reports no
    comparison yet.
    """
    groups: dict[
        tuple[object, ...],
        dict[
            str,
            tuple[
                SweepPoint,
                JudgmentFitSpec,
                Path,
                Literal["posterior.nc", "point-estimate.safetensors"],
            ],
        ],
    ] = {}

    for point in points:
        if store.entry(point)["status"] != PointStatus.COMPLETED.value:
            continue

        fit = decode_judgment_fit(point.config)
        comparison = fit.comparison

        if comparison is None:
            continue

        manifest_data = store.manifest(point)

        if manifest_data is None:
            continue

        manifest = decode_fit_manifest(manifest_data)

        if manifest.fit != fit:
            msg = (
                f"the fit manifest of point {point.point_id!r} does not describe "
                "that point's resolved fit"
            )
            raise ValueError(msg)

        artifact = manifest.artifact
        path = store.directory(point) / artifact.name

        if not path.is_file():
            continue

        with path.open("rb") as persisted:
            digest = file_digest(persisted, "sha256").hexdigest()

        if digest != artifact.sha256:
            msg = (
                f"the persisted fit artifact of point {point.point_id!r} has "
                f"SHA-256 {digest}, not its manifest digest {artifact.sha256}"
            )
            raise ValueError(msg)

        models = comparison.models
        members = groups.setdefault(_comparison_key(fit, point), {})

        if models.candidate in members:
            msg = (
                f"the comparison set {models.candidates!r} has two completed "
                f"points named {models.candidate!r} at seed {point.seed}"
            )
            raise ValueError(msg)

        members[models.candidate] = (point, fit, path, artifact.name)

    freeze_id = store.freeze_id
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()
    rows: list[Mapping[str, JsonValue]] = []

    for members in groups.values():
        representative = next(iter(members.values()))[1]
        comparison = representative.comparison

        if comparison is None:
            msg = "a grouped model comparison lost its declaration"
            raise RuntimeError(msg)

        candidates = comparison.models.candidates

        if frozenset(members) != frozenset(candidates):
            continue

        artifacts = frozenset(member[3] for member in members.values())

        if len(artifacts) != 1:
            msg = (
                f"the comparison set {candidates!r} mixes persisted fit "
                f"representations {tuple(sorted(artifacts))!r}"
            )
            raise ValueError(msg)

        [artifact_name] = artifacts

        if artifact_name == POINT_ESTIMATE_NAME:
            compared_rows: dict[str, tuple[Mapping[str, JsonValue], ...]] = {}

            for candidate in candidates:
                point = members[candidate][0]
                recorded = read_rows(store.directory(point))

                if recorded is None:
                    break

                selected = tuple(
                    row
                    for row in recorded.rows
                    if row.get("metric") == HELD_OUT_PREDICTIVE_METRIC
                )

                if not selected:
                    break

                if any(row.get("candidate") != candidate for row in selected):
                    msg = (
                        f"the held-out comparison rows of point {point.point_id!r} "
                        f"do not name candidate {candidate!r}"
                    )
                    raise ValueError(msg)

                compared_rows[candidate] = selected
            else:
                for candidate in candidates:
                    rows.extend(compared_rows[candidate])

            continue

        trees = {
            candidate: xr.open_datatree(members[candidate][2])
            for candidate in candidates
        }

        try:
            compared = compare_declared_models(trees, comparison.models)
        finally:
            for tree in trees.values():
                tree.close()

        observations = {row.name: row.observations for row in compared.models}

        for result in compared.rows():
            kind = result["kind"]
            candidate = cast(
                "str", result["model"] if kind == "model" else result["left"]
            )
            point = members[candidate][0]
            value = result["elpd"] if kind == "model" else result["delta"]
            rows.append(
                {
                    "freeze_id": freeze_id,
                    "config_hash": config_digest(point.config),
                    "point_id": point.point_id,
                    "task": FIT_ROLE,
                    "metric": (
                        "psis-loo-elpd" if kind == "model" else "psis-loo-elpd-delta"
                    ),
                    "value": value,
                    "n": observations[candidate],
                    "seed": point.seed,
                    "candidates": list(candidates),
                    **result,
                    **approximation,
                }
            )

    return rows


def _judgment_rows(design: ResponseDesign, /) -> tuple[JudgmentRow, ...]:
    """Reconstruct the row identities and factors needed by a split generator.

    A response-grain split uses source position, so the bounded design's
    continuous value is intentionally absent from this identity-only row.
    """
    if design.response_levels is not None:
        return _design_codec(design).decode(design)

    return tuple(
        JudgmentRow(
            design.participant_levels[int(design.participants[position])],
            design.item_levels[int(design.items[position])],
            0,
            RowColumns(
                factors={
                    name: design.factor_levels[name][int(column[position])]
                    for name, column in design.factors.items()
                },
                readouts={},
            ),
        )
        for position in range(design.n_rows)
    )


def _declared_split(
    fit: JudgmentFitSpec, design: ResponseDesign, /
) -> tuple[tuple[JudgmentRow, ...], Split] | None:
    """Return the rows and admitted split of a declared comparison."""
    comparison = fit.comparison

    if comparison is None:
        return None

    rows = _judgment_rows(design)
    gate = CoverageGate(
        comparison.coverage_threshold,
        policy=comparison.unseen_participant,
    )

    return rows, splitter_for(comparison).split(rows, gate)


def _comparison_rows(
    split: Split,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
) -> list[Mapping[str, JsonValue]]:
    """Return one participant-coverage row for every declared held-out fold."""
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()

    return [
        {
            "freeze_id": freeze_id,
            "config_hash": config_digest(point.config),
            "point_id": point.point_id,
            "task": FIT_ROLE,
            "metric": HELD_OUT_PARTICIPANT_FRACTION_METRIC,
            "value": fold.coverage.fraction,
            "n": len(fold.held_out),
            "seed": seed,
            "fold": fold.index,
            "held_out_unit": split.report.unit,
            **approximation,
        }
        for fold in split.folds
    ]


def _marginal_point_draw(
    program: BoundProgram,
    objective: PointObjective,
    fitted: PointFit,
    /,
) -> torch.Tensor:
    """Expand one marginal population point at every local prior mean."""
    population = {
        slot.site: values
        for slot, values in zip(
            objective.layout,
            torch.split(fitted.parameters, [slot.size for slot in objective.layout]),
            strict=True,
        )
    }

    return torch.cat(
        tuple(
            population.get(slot.site, fitted.parameters.new_zeros(slot.size))
            for slot in program.layout
        )
    )


def map_fold_probabilities(
    compiled: ResponseProgram,
    codec: DesignCodec,
    retained: JudgmentDesign,
    held_out: JudgmentDesign,
    objective: PointObjective,
    fitted: PointFit,
    policy: HeldOutParticipantPolicy | None,
    *,
    seed: int,
    unseen: bool,
) -> torch.Tensor:
    """Return response probabilities for held-out rows under a MAP point.

    The point was fitted to the retained design. A held-out participant the
    retained design never shows is predicted under the given policy: a
    conditional point admits only the plug-in policy for one, and a
    truncated-marginal point integrates it unless the policy is plug-in.

    Raises
    ------
    MarginalLikelihoodError
        If a conditional point is asked for an unseen participant under any
        policy other than plug-in.
    """
    generator = torch.Generator(device=held_out.responses.device).manual_seed(seed)

    def plug_in(draws: torch.Tensor) -> torch.Tensor:
        predictor = HeldOutPredictor(compiled, retained, draws, samples=1)
        parameters = predictor.parameters(
            held_out,
            policy=HeldOutParticipantPolicy.PLUG_IN,
            generator=generator,
        )
        target = compiled.over(held_out)

        return torch.stack(
            tuple(
                response_log_probabilities(target, vector).exp()
                for vector in parameters
            )
        ).mean(dim=0)

    match compiled.objective:
        case ConditionalPlan():
            if unseen and policy is not HeldOutParticipantPolicy.PLUG_IN:
                msg = (
                    "a conditional MAP point predicts an unseen participant only "
                    "under the declared plug-in policy"
                )
                raise MarginalLikelihoodError(msg)

            return plug_in(fitted.parameters.unsqueeze(0))
        case TruncatedMarginalPlan():
            if not unseen or policy is HeldOutParticipantPolicy.MARGINAL:
                return MarginalPointPredictor(
                    compiled,
                    codec,
                    retained,
                    fitted,
                ).probabilities(held_out)

            bound = compiled.over(retained)
            return plug_in(_marginal_point_draw(bound, objective, fitted).unsqueeze(0))


def _map_comparison_rows(
    compiled: ResponseProgram,
    design: JudgmentDesign,
    split: Split,
    plan: MapInference,
    *,
    map_seed: int,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
    candidate: str,
) -> list[Mapping[str, JsonValue]]:
    """Refit every MAP fold and return each held-out participant's density.

    Each fold's held-out rows are divided into the seen-participant and
    new-participant regimes, and each regime's rows by participant. One row
    is returned per fold, regime and participant: its value is the mean
    predictive log density of that participant's held-out responses and its
    count is the number of those responses, so a sum over any set of
    participants is recovered from the rows alone.

    A declared unseen-participant policy predicts the new participants of
    every fold. Without one, a fold whose coverage is within the threshold
    has no prediction for its new participants, and its new-participant
    regime is recorded as one row whose value is the number of those
    participants and whose count is the number of their held-out responses.
    """
    codec = _design_codec(design)
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()
    rows: list[Mapping[str, JsonValue]] = []

    for fold in split.folds:
        retained = codec.encode(fold.retained)
        bound = compiled.over(retained)
        objective, objective_name = map_point_objective(bound)
        fitted = MapFitter(
            plan,
            seed=derive_seed(map_seed, f"fold:{fold.index}"),
        ).fit(objective)
        seen = frozenset(row.participant for row in fold.retained)

        for regime, selected_rows in (
            (
                SEEN_PARTICIPANT_REGIME,
                tuple(row for row in fold.held_out if row.participant in seen),
            ),
            (
                NEW_PARTICIPANT_REGIME,
                tuple(row for row in fold.held_out if row.participant not in seen),
            ),
        ):
            if not selected_rows:
                continue

            unseen = regime == NEW_PARTICIPANT_REGIME
            policy = fold.policy if unseen else None

            if unseen and policy is None:
                rows.append(
                    {
                        "freeze_id": freeze_id,
                        "config_hash": config_digest(point.config),
                        "point_id": point.point_id,
                        "task": FIT_ROLE,
                        "metric": HELD_OUT_UNPREDICTED_METRIC,
                        "value": float(len({row.participant for row in selected_rows})),
                        "n": len(selected_rows),
                        "seed": seed,
                        "fold": fold.index,
                        "participant_regime": regime,
                        "objective": objective_name,
                        "convention": None,
                        "candidate": candidate,
                        **approximation,
                    }
                )

                continue

            held_out = codec.encode(selected_rows)
            fold_seed = derive_seed(map_seed, f"fold-prediction:{fold.index}:{regime}")
            probabilities = map_fold_probabilities(
                compiled,
                codec,
                retained,
                held_out,
                objective,
                fitted,
                policy,
                seed=fold_seed,
                unseen=unseen,
            )
            densities = predictive_log_density(probabilities, held_out.responses)

            if not bool(torch.isfinite(densities).all()):
                msg = (
                    f"MAP fold {fold.index} produced a nonfinite predictive log "
                    f"density for the {regime} regime"
                )
                raise MarginalLikelihoodError(msg)

            by_participant: dict[str, list[float]] = {}

            for row, density in zip(selected_rows, densities.tolist(), strict=True):
                by_participant.setdefault(row.participant, []).append(density)

            rows.extend(
                {
                    "freeze_id": freeze_id,
                    "config_hash": config_digest(point.config),
                    "point_id": point.point_id,
                    "task": FIT_ROLE,
                    "metric": HELD_OUT_PREDICTIVE_METRIC,
                    "value": fsum(participant_densities) / len(participant_densities),
                    "n": len(participant_densities),
                    "seed": seed,
                    "fold": fold.index,
                    "participant_regime": regime,
                    "participant": participant,
                    "objective": objective_name,
                    "convention": None if policy is None else policy.value,
                    "candidate": candidate,
                    **approximation,
                }
                for participant, participant_densities in by_participant.items()
            )

    return rows


def _ablation_rows(
    fit: JudgmentFitSpec,
    design: JudgmentDesign,
    split: Split,
    *,
    point: SweepPoint,
    freeze_id: str,
    seed: int,
    initialization_seed: int,
) -> list[Mapping[str, JsonValue]]:
    """Fit every declared table rung per fold and return held-out density rows."""
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None).to_spec()
    rows: list[Mapping[str, JsonValue]] = []
    generator = torch.Generator(device=design.responses.device).manual_seed(
        initialization_seed
    )
    comparison = fit.comparison

    if comparison is None:
        msg = "table-ablation execution needs its validated comparison"
        raise RuntimeError(msg)

    records = _judgment_rows(design)

    def level_of(record: JudgmentRow, position: int) -> str:
        unit = comparison.held_out_unit

        if unit == "response":
            return str(position)
        if unit == "participant":
            return record.participant
        if unit == "item":
            return record.item

        return record.factors[unit]

    for fold in split.folds:
        held_positions = tuple(
            position
            for position, record in enumerate(records)
            if level_of(record, position) in fold.units
        )
        retained_positions = tuple(
            position
            for position, record in enumerate(records)
            if level_of(record, position) not in fold.units
        )
        retained = design.select_rows(retained_positions)
        held_out = design.select_rows(held_positions)

        for rung in ladder_of(fit.ablation, generator=generator):
            head = OrdinalHead(design.response_levels)
            fit_rung(rung, head, retained)

            with torch.no_grad():
                density = float(mean_log_density(rung, head, held_out))

            rows.append(
                {
                    "freeze_id": freeze_id,
                    "config_hash": config_digest(point.config),
                    "point_id": point.point_id,
                    "task": FIT_ROLE,
                    "metric": "table-ablation-log-density",
                    "value": density,
                    "n": held_out.n_rows,
                    "seed": seed,
                    "fold": fold.index,
                    "rung": str(rung.spec.lift.lift),
                    **approximation,
                }
            )

    return rows


@final
class FitExecutor:
    """Execute the judgment-fit points of one launch through an experiment's store.

    The executor keeps what one launch shares: the checked fit and compiled
    response model of each point it has checked, and the design of each
    dataset it has read. A later launch constructs its own executor and so
    reads every dataset again.

    Parameters
    ----------
    store : ExperimentStore
        The store each point's directory lies under, read for the experiment
        whose root seeds every fit and whose freeze every row is reported
        against.
    designs : JudgmentDesigns
        Resolves each fit's dataset to its design.
    compiler : ResponseCompiler
        Compiles a fit's response model over its design.
    environment : ExecutionEnvironment
        The package versions and source revision every manifest records.
    infrastructure_attempts : int
        The attempts each point receives across launches before infrastructure
        failure settles it.
    """

    __slots__ = (
        "_checked",
        "_compiler",
        "_designs",
        "_environment",
        "_failure_policy",
        "_infrastructure_attempts",
        "_range_reports",
        "_read",
        "_store",
    )

    def __init__(
        self,
        store: ExperimentStore,
        *,
        designs: JudgmentDesigns,
        compiler: ResponseCompiler,
        environment: ExecutionEnvironment,
        infrastructure_attempts: int,
    ) -> None:
        self._store = store
        self._designs = designs
        self._compiler = compiler
        self._environment = environment
        self._failure_policy = store.experiment.failure
        self._infrastructure_attempts = infrastructure_attempts
        self._checked: dict[
            str,
            tuple[JudgmentFitSpec, ResponseProgram, PointObjective | None],
        ] = {}
        self._range_reports: dict[str, DynamicRangeReport | None] = {}
        self._read: dict[str, ReadCorpusDesign] = {}

    @property
    def store(self) -> ExperimentStore:
        """Return the store this executor is bound to."""
        return self._store

    def _fit(
        self,
        point: SweepPoint,
        fit: JudgmentFitSpec,
        compiled: ResponseProgram,
        read: ReadCorpusDesign,
        derived: PointSeed,
        range_report: DynamicRangeReport | None,
        point_objective: PointObjective | None,
        numerical_retry: int,
        /,
    ) -> RunOutcome:
        """Infer one fit and any calibration, then record rows and manifest."""
        directory = self._store.directory(point)
        freeze_id = self._store.freeze_id
        program = compiled.over(read.design)
        streams = derived.policy(
            accelerator=None,
            samplers=(
                NUTS_STREAM,
                GUIDE_STREAM,
                SIMULATION_STREAM,
                INFERENCE_STREAM,
                RANK_STREAM,
                ABLATION_STREAM,
                MAP_STREAM,
            ),
        )

        if numerical_retry:
            streams = streams.retried(numerical_retry)
        rows: list[Mapping[str, JsonValue]]
        posterior_draws: torch.Tensor | None = None
        point_fit: PointFit | None = None
        artifact: FitArtifact

        match fit.inference:
            case NutsInference(warmup=warmup, retained=retained, chains=chains):
                plan = NutsPlan(
                    warmup=warmup,
                    retained=retained,
                    chains=chains,
                    seed=streams.sampler_seed(NUTS_STREAM),
                )
                posterior = QuiversNutsFitter(plan).fit(program)
                posterior_draws = _flat_draws(program, posterior)
                diagnostics = posterior.diagnostics
                divergences = diagnostics.divergences
                rows = _fit_rows(
                    {
                        "divergences": (
                            None if divergences is None else float(divergences)
                        ),
                        "max-rhat": diagnostics.max_rhat,
                        "min-ess": diagnostics.min_ess,
                    },
                    point=point,
                    freeze_id=freeze_id,
                    seed=derived.root,
                    draws=posterior.chains * posterior.retained,
                )
                calibration = fit.calibration

                if calibration is not None:
                    if read.design.response_levels is None:
                        msg = "a calibrated fit has no ordinal judgment design"
                        raise RuntimeError(msg)

                    campaign = prior_predictive_calibration(
                        compiled,
                        read.design,
                        replicates=calibration.replicates,
                        guide_iterations=calibration.guide_iterations,
                        guide_learning_rate=calibration.guide_learning_rate,
                        plan=plan,
                        seeds=CalibrationSeeds(
                            simulation=streams.sampler_seed(SIMULATION_STREAM),
                            inference=streams.sampler_seed(INFERENCE_STREAM),
                            rank=streams.sampler_seed(RANK_STREAM),
                        ),
                    )
                    report = campaign.report

                    if report is None:
                        raise UnfittedCalibrationError(_unfitted_calibration(campaign))

                    rows += _calibration_rows(
                        report, point=point, freeze_id=freeze_id, seed=derived.root
                    )
                    rows += _replicate_rows(
                        campaign, point=point, freeze_id=freeze_id, seed=derived.root
                    )
            case VariationalInference(
                guide=guide,
                iterations=iterations,
                draws=draws,
                learning_rate=learning_rate,
            ):
                fitted = QuiversGuideFitter(
                    guide,
                    iterations=iterations,
                    draws=draws,
                    learning_rate=learning_rate,
                    seed=streams.sampler_seed(GUIDE_STREAM),
                ).fit(program)
                posterior_draws = fitted.draws
                rows = _fit_rows(
                    {"evidence-bound": fitted.evidence_bound},
                    point=point,
                    freeze_id=freeze_id,
                    seed=derived.root,
                    draws=draws,
                )
            case MapInference() as plan:
                if point_objective is None:
                    msg = "a checked MAP fit lost its point objective"
                    raise RuntimeError(msg)

                point_fit = MapFitter(
                    plan,
                    seed=streams.sampler_seed(MAP_STREAM),
                ).fit(point_objective)
                rows = [
                    {
                        **row,
                        "objective": program.program.objective.objective,
                    }
                    for row in _fit_rows(
                        {"map-log-posterior": point_fit.log_density},
                        point=point,
                        freeze_id=freeze_id,
                        seed=derived.root,
                        draws=point_fit.steps,
                    )
                ]
            case unsupported:
                msg = f"the inference {unsupported!r} has no fitter"
                raise TypeError(msg)

        if range_report is not None:
            rows.insert(
                0,
                _dynamic_range_row(
                    range_report,
                    _gated_readout(fit),
                    point=point,
                    freeze_id=freeze_id,
                    seed=derived.root,
                ),
            )

        if point_fit is None:
            if posterior_draws is None:
                msg = "a sampled fit produced no posterior draws"
                raise RuntimeError(msg)

            _write_posterior(directory, _posterior_tree(program, posterior_draws))
            artifact = _artifact(directory, POSTERIOR_NAME)
        else:
            artifact = _write_point(
                directory,
                point_fit,
                objective=program.program.objective.objective,
            )

        if read.design.response_levels is not None:
            ordinal_design = read.design

            if posterior_draws is not None:
                rows += _prediction_rows(
                    program,
                    ordinal_design,
                    posterior_draws,
                    point=point,
                    freeze_id=freeze_id,
                    seed=derived.root,
                )
            elif point_fit is not None:
                match program.program.objective:
                    case ConditionalPlan():
                        rows += _prediction_rows(
                            program,
                            ordinal_design,
                            point_fit.parameters.unsqueeze(0),
                            point=point,
                            freeze_id=freeze_id,
                            seed=derived.root,
                        )
                    case TruncatedMarginalPlan():
                        pass

            rows.append(
                _ceiling_row(
                    ordinal_design,
                    point=point,
                    freeze_id=freeze_id,
                    seed=derived.root,
                )
            )

        if fit.interannotator is not None:
            rows += _interannotator_rows(
                read.design,
                fit.interannotator.settings(),
                point=point,
                freeze_id=freeze_id,
                seed=derived.root,
            )

        declared = _declared_split(fit, read.design)

        if declared is not None:
            comparison = fit.comparison

            if comparison is None:
                msg = "a declared comparison split lost its declaration"
                raise RuntimeError(msg)

            _, split = declared
            rows += _comparison_rows(
                split,
                point=point,
                freeze_id=freeze_id,
                seed=derived.root,
            )

            if point_fit is not None:
                match fit.inference, program.program.objective:
                    case (
                        MapInference() as plan,
                        ConditionalPlan() | TruncatedMarginalPlan(),
                    ):
                        if read.design.response_levels is None:
                            msg = "a MAP comparison needs an ordinal design"
                            raise RuntimeError(msg)

                        rows += _map_comparison_rows(
                            compiled,
                            read.design,
                            split,
                            plan,
                            map_seed=streams.sampler_seed(MAP_STREAM),
                            point=point,
                            freeze_id=freeze_id,
                            seed=derived.root,
                            candidate=comparison.models.candidate,
                        )
                    case _:
                        pass

            if fit.ablation:
                if read.design.response_levels is None:
                    msg = "a validated table ablation has no ordinal response design"
                    raise RuntimeError(msg)

                rows += _ablation_rows(
                    fit,
                    read.design,
                    split,
                    point=point,
                    freeze_id=freeze_id,
                    seed=derived.root,
                    initialization_seed=streams.sampler_seed(ABLATION_STREAM),
                )

        write_rows(
            directory,
            RecordedRows(self._store.runs.freeze_id, derived, tuple(rows)),
        )
        manifest = JudgmentFitManifest(
            fit=fit,
            design=read.provenance,
            seed_derivation=encode_seed_derivation(derived),
            artifact=artifact,
            package_versions=dict(self._environment.packages),
            git_revision=self._environment.git_revision,
        )
        replace_file(directory / FIT_MANIFEST_NAME, encode_fit_manifest(manifest))

        return RunOutcome(directory, ManifestStatus.COMPLETED, cached=False)

    def _design(self, dataset: str, /) -> ReadCorpusDesign:
        """Return a dataset's design, read on the first request of the launch."""
        read = self._read.get(dataset)

        if read is None:
            read = self._designs.read(dataset)
            self._read[dataset] = read

        return read

    def check(
        self, point: SweepPoint, /
    ) -> tuple[JudgmentFitSpec, ResponseProgram, PointObjective | None]:
        """Return a point's checked fit, program and point objective.

        A point is checked once per executor: the first call validates the fit,
        compiles its response model, constructs a MAP objective when declared,
        requires the program to draw at least one parameter and, for a calibrated fit,
        requires its
        thresholds to vary over participants, and every later call returns
        that result. A launcher checks every point it is about to submit, so a
        fit that cannot be executed is refused before any point directory is
        claimed, and executing a point checks it before claiming its directory.

        Parameters
        ----------
        point : SweepPoint
            The point to check.

        Raises
        ------
        PointExecutionError
            If the fit's response program draws no parameter, or the fit
            declares calibration and its response model's thresholds do not
            vary over participants.
        """
        checked = self._checked.get(point.point_id)

        if checked is not None:
            return checked

        fit = decode_judgment_fit(point.config)
        compiled = self._compiler.compile(fit.response)
        point_objective = None

        if not compiled.draws:
            msg = (
                f"the fit of the point {point.point_id!r} is refused: its response "
                f"program draws no parameter, so it has no posterior"
            )
            raise PointExecutionError(msg)

        match fit.inference:
            case MapInference():
                point_objective, _ = map_point_objective(
                    compiled.over(self._design(fit.dataset).design)
                )
            case _:
                pass

        if fit.calibration is not None:
            try:
                ensure_participant_thresholds(compiled)

            except ValueError as refused:
                msg = f"the fit of the point {point.point_id!r} is refused: {refused}"
                raise PointExecutionError(msg) from refused

        range_report = None

        if fit.dynamic_range is not None:
            gate = fit.dynamic_range
            try:
                range_report = DynamicRangeGate(gate.criterion).admit(
                    _readout_odds(gate.readout, fit, self._design(fit.dataset).design),
                    inventory=fit.dataset,
                    observed_at=datetime.now(UTC),
                )
            except ValueError as refused:
                msg = f"the fit of the point {point.point_id!r} is refused: {refused}"
                raise PointExecutionError(msg) from refused

        self._checked[point.point_id] = (fit, compiled, point_objective)
        self._range_reports[point.point_id] = range_report

        return fit, compiled, point_objective

    def __call__(self, point: SweepPoint, /) -> RunOutcome:
        """Fit one point unless its directory already records its outcome.

        Raises
        ------
        PointExecutionError
            If the point is not executable, refused before its directory is
            claimed or created.
        ReseededRunError
            If the point's directory, or the failure it records, was seeded
            under another derivation.
        PointTerminatedError
            If the process receives ``SIGTERM`` while the point is fitted.
        """
        fit, compiled, point_objective = self.check(point)
        directory = self._store.directory(point)
        derived = self._store.experiment.seed_policy.point(point.point_id, point.seed)
        self._store.claim(point)

        if self._store.is_completed(point):
            return RunOutcome(directory, ManifestStatus.COMPLETED, cached=True)

        return execute_to_outcome(
            lambda numerical_retry: self._fit(
                point,
                fit,
                compiled,
                self._design(fit.dataset),
                derived,
                self._range_reports[point.point_id],
                point_objective,
                numerical_retry,
            ),
            directory,
            numerical=(
                MarginalLikelihoodError,
                PosteriorFittingError,
                PointFittingError,
                UnfittedCalibrationError,
            ),
            derivation=derived,
            infrastructure_attempts=self._infrastructure_attempts,
            failure_policy=self._failure_policy,
            halve_batch=lambda: False,
        )
