"""Compares the conditional and truncated-marginal objectives on known random effects.

The comparison is one judgment-fit experiment over a synthetic design whose
participant effects are known. Its points cross the two likelihood objectives
with two held-out units, each at every declared seed: holding out responses
leaves every held-out participant seen in training, and holding out
participants leaves every one of them new. Each point is one MAP candidate of
a two-member comparison set named by the objectives, and it predicts an
unseen participant under the convention its objective owns: the plug-in
convention for the conditional objective and the marginal convention for the
truncated-marginal objective.

This script reads that experiment's results directory and the point
directories of its run store, and writes one canonical Parquet report and a
manifest. Three families of rows are computed from the persisted fits.

The held-out predictive log density of a candidate is the mean over its
held-out responses, pooled over every fold, with an interval from a
bootstrap over participants. Each participant's held-out responses are
resampled together, from the per-participant densities and counts the fit
layer records.

The new-participant advantage is the truncated-marginal candidate's density
minus the conditional candidate's under the participant-held-out unit, and
the seen-participant agreement is the same difference under the
response-held-out unit, each with its participant-bootstrap interval. The
advantage holds when the difference exceeds the interval's half-width; the
agreement holds when the interval lies within the declared margin of zero.
When a seed's seen-participant candidates disagree, its new-participant
contrast is not read.

The random-effect recovery of a candidate is the mean absolute difference
between each declared population site of its fitted point and the value the
synthetic design's responses were drawn at.

Two stop rules refuse the whole report: a fit whose response model, with its
objective set to the conditional likelihood, is not the model the design's
responses were drawn from, or whose design provenance is not the synthetic
design's; and a response-held-out point that records a nonzero
new-participant fraction for any fold. A cell none of whose declared seeds
completed is reported in the manifest with the status of every point that
did not complete, and the contrasts it belongs to are not read.

Run it on an ingested experiment::

    python scripts/judgment/marginal_likelihood.py results/1/marginal-likelihood
        --runs "$STORE/runs" --declaration marginal-likelihood-analysis.json
        --out artifacts/judgment/marginal-likelihood
"""

import json
import math
import sys
from argparse import ArgumentParser
from hashlib import file_digest, sha256
from math import fsum
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, Self, final

import didactic.api as dx
import pyarrow.parquet as pq
import torch
from safetensors.torch import load_file

from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    Approximation,
    MetricResult,
    PooledJudgment,
    ReportIdentity,
    ReportRow,
    Resampling,
)
from lofi.judgment import (
    ConditionalLikelihood,
    ConditionalPlan,
    ConditionalPointObjective,
    MapInference,
    MarginalPointObjective,
    ResponseCompiler,
    ResponseSpec,
    TruncatedMarginalLikelihood,
    TruncatedMarginalPlan,
    decode_fit_manifest,
    spec_digest,
)
from lofi.registry import Registry
from lofi.run.fits import (
    DESIGN_FAMILY,
    HELD_OUT_PARTICIPANT_FRACTION_METRIC,
    HELD_OUT_PREDICTIVE_METRIC,
    NEW_PARTICIPANT_REGIME,
    POINT_ESTIMATE_NAME,
    SEEN_PARTICIPANT_REGIME,
)
from lofi.run.reporting import ParquetReportSink
from lofi.run.results import INGESTED_NAME, MANIFESTS_NAME, STATUS_NAME
from lofi.spec import HeldOutParticipantPolicy, derive_seed
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._types import JsonValue
    from lofi.judgment import (
        BoundProgram,
        JudgmentDesign,
        JudgmentFitManifest,
        PointObjective,
        ReadDesign,
        Slot,
    )
    from lofi.stats import BootstrapResult

CONDITIONAL: Final = "conditional"
"""The objective conditioning on fitted participant offsets."""

MARGINAL: Final = "truncated_marginal"
"""The objective integrating the participant-local block."""

OBJECTIVES: Final = (CONDITIONAL, MARGINAL)
"""The two candidates of every comparison set, in report order."""

CONVENTIONS: Final[Mapping[str, HeldOutParticipantPolicy]] = MappingProxyType(
    {
        CONDITIONAL: HeldOutParticipantPolicy.PLUG_IN,
        MARGINAL: HeldOutParticipantPolicy.MARGINAL,
    }
)
"""The unseen-participant convention each objective's candidate predicts under."""

SEEN_UNIT: Final = "response"
"""The held-out unit leaving every held-out participant seen in training."""

NEW_UNIT: Final = "participant"
"""The held-out unit leaving every held-out participant new."""

REGIMES: Final[Mapping[str, str]] = MappingProxyType(
    {SEEN_UNIT: SEEN_PARTICIPANT_REGIME, NEW_UNIT: NEW_PARTICIPANT_REGIME}
)
"""The participant regime each held-out unit is read under."""

DENSITY_METRIC: Final = "heldout-predictive-log-density"
"""Names a candidate's pooled held-out predictive log density."""

ADVANTAGE_METRIC: Final = "new-participant-advantage"
"""Names the marginal-minus-conditional density under new participants."""

AGREEMENT_METRIC: Final = "seen-participant-agreement"
"""Names the marginal-minus-conditional density under seen participants."""

RECOVERY_METRIC: Final = "random-effect-recovery"
"""Names the mean absolute error of one fitted population site."""

CONTRAST: Final = f"{MARGINAL}-minus-{CONDITIONAL}"
"""The method a contrast row is reported under."""

UNIT: Final = "participant"
"""The unit every interval resamples."""

ELEMENT: Final = "site-element"
"""The unit a recovery error is averaged over."""

REPORT_NAME: Final = "report.parquet"
"""Names the canonical report file within the output directory."""

MANIFEST_NAME: Final = "manifest.json"
"""Names the provenance manifest within the output directory."""


class BootstrapDeclaration(dx.Model, extra="forbid"):
    """Declares the participant bootstrap every interval is read off.

    Parameters
    ----------
    replicates : int
        Resamples of the participants drawn for each interval.
    seed : int
        Root every interval's resampling seed derives from.
    level : float
        Coverage of every interval.
    interval : Literal["percentile", "bca"]
        Bootstrap interval reported.
    """

    replicates: int
    seed: int
    level: float
    interval: Literal["percentile", "bca"]

    __axioms__ = (
        dx.axiom("replicates > 0", message="a participant bootstrap resamples"),
        dx.axiom("seed >= 0", message="a bootstrap seed is nonnegative"),
    )

    @dx.model_validator()
    def _level_is_a_proportion(self) -> Self:
        """Refuse a coverage outside the open unit interval."""
        if not math.isfinite(self.level) or not 0.0 < self.level < 1.0:
            msg = f"a bootstrap's coverage is not a proportion: {self.level!r}"
            raise ValueError(msg)

        return self


class RecoveryDeclaration(dx.Model, extra="forbid"):
    """Declares which fitted sites are compared with the known effects.

    Parameters
    ----------
    sites : tuple[str, ...]
        Population sites of the generating program whose fitted values are
        compared with the values the responses were drawn at.
    tolerance : float
        Largest mean absolute error a recovered site is admitted with.
    """

    sites: tuple[str, ...]
    tolerance: float

    __axioms__ = (
        dx.axiom("len(sites) > 0", message="a recovery compares at least one site"),
    )

    @dx.model_validator()
    def _sites_and_tolerance_are_declared(self) -> Self:
        """Refuse a repeated or blank site, or a tolerance that is not positive."""
        if any(not site for site in self.sites) or len(set(self.sites)) != len(
            self.sites
        ):
            msg = f"recovered sites are blank or repeated: {self.sites!r}"
            raise ValueError(msg)

        if not math.isfinite(self.tolerance) or self.tolerance <= 0.0:
            msg = f"a recovery tolerance is not positive: {self.tolerance!r}"
            raise ValueError(msg)

        return self


class AnalysisDeclaration(dx.Model, extra="forbid"):
    """Declares the identity, seeds, intervals and criteria of one comparison.

    Parameters
    ----------
    freeze_id : str
        Freeze the report is quoted against.
    grammar_fingerprint : str
        Fingerprint the report identity names; a judgment fit reads no grammar.
    suite : str
        Evaluation suite name.
    task : str
        Evaluation task name.
    split : str
        Judgment dataset split.
    seeds : tuple[int, ...]
        Sweep seeds every cell is declared at.
    bootstrap : BootstrapDeclaration
        The participant bootstrap.
    agreement_margin : float
        Half-width of the interval around zero the seen-participant
        difference must lie within.
    recovery : RecoveryDeclaration
        The random-effect recovery comparison.
    """

    freeze_id: str
    grammar_fingerprint: str
    suite: str
    task: str
    split: str
    seeds: tuple[int, ...]
    bootstrap: BootstrapDeclaration
    agreement_margin: float
    recovery: RecoveryDeclaration

    __axioms__ = (
        dx.axiom("len(freeze_id) > 0", message="an analysis names its freeze"),
        dx.axiom(
            "len(grammar_fingerprint) > 0",
            message="an analysis names its grammar fingerprint",
        ),
        dx.axiom("len(suite) > 0", message="an analysis names its suite"),
        dx.axiom("len(task) > 0", message="an analysis names its task"),
        dx.axiom("len(split) > 0", message="an analysis names its split"),
        dx.axiom("len(seeds) > 0", message="an analysis declares its seeds"),
    )

    @dx.model_validator()
    def _seeds_and_margin_are_declared(self) -> Self:
        """Refuse repeated or negative seeds and a margin that is not positive."""
        if len(set(self.seeds)) != len(self.seeds) or any(
            seed < 0 for seed in self.seeds
        ):
            msg = f"declared seeds are repeated or negative: {self.seeds!r}"
            raise ValueError(msg)

        if not math.isfinite(self.agreement_margin) or self.agreement_margin <= 0.0:
            msg = f"an agreement margin is not positive: {self.agreement_margin!r}"
            raise ValueError(msg)

        return self


class HeldOutRow(dx.Model, extra="ignore"):
    """One participant's held-out density in one fold, as the fit layer records it."""

    point_id: str
    candidate: str
    participant_regime: str
    participant: str
    fold: int
    value: float
    n: int


class FractionRow(dx.Model, extra="ignore"):
    """One fold's new-participant fraction, as the fit layer records it."""

    point_id: str
    fold: int
    value: float


class KnownEffects(Protocol):
    """A synthetic design drawn at a known parameter vector."""

    @property
    def spec(self) -> ResponseSpec:
        """Return the response model the responses were drawn from."""
        ...

    def read(self) -> ReadDesign:
        """Return the design and its provenance."""
        ...

    def program(self) -> BoundProgram:
        """Return the generating program bound to the design."""
        ...

    def generating(self) -> Tensor:
        """Return the parameter vector the responses were drawn at."""
        ...


class StopRuleError(ValueError):
    """A declared stop rule fires, so no cell of the comparison is reported."""


@final
class DeclaredGrammar:
    """The grammar fingerprint a report identity is quoted under.

    Parameters
    ----------
    fingerprint : str
        Stable grammar fingerprint.
    """

    __slots__ = ("_fingerprint",)

    def __init__(self, fingerprint: str) -> None:
        self._fingerprint = fingerprint

    @property
    def fingerprint(self) -> str:
        """Return the stable grammar fingerprint."""
        return self._fingerprint


class Cell(NamedTuple):
    """One objective under one held-out unit."""

    objective: str
    unit: str


class Candidate(NamedTuple):
    """One completed point of the comparison."""

    point_id: str
    seed: int
    cell: Cell
    manifest: JudgmentFitManifest


class Participant(NamedTuple):
    """One participant's held-out log density summed over every fold."""

    participant: str
    total: float
    count: int


class Contrast(NamedTuple):
    """One seed's marginal-minus-conditional difference and its verdict."""

    seed: int
    unit: str
    observed: float
    low: float
    high: float
    holds: bool


def read_declaration(path: Path) -> AnalysisDeclaration:
    """Return the validated analysis declaration a JSON file holds.

    Parameters
    ----------
    path : Path
        The declaration's JSON file.

    Raises
    ------
    ValueError
        If the file does not validate as a declaration.
    """
    try:
        return AnalysisDeclaration.model_validate_json(path.read_bytes())
    except dx.ValidationError as refused:
        msg = f"{path} is not a marginal-likelihood analysis declaration: {refused}"
        raise ValueError(msg) from refused


def _cell(manifest: JudgmentFitManifest) -> Cell:
    """Return a manifest's cell once its candidate and convention are coupled."""
    fit = manifest.fit
    comparison = fit.comparison
    point = manifest.seed_derivation.point_id

    match fit.response, fit.inference:
        case ResponseSpec(objective=ConditionalLikelihood()), MapInference():
            name = CONDITIONAL
        case ResponseSpec(objective=TruncatedMarginalLikelihood()), MapInference():
            name = MARGINAL
        case _:
            msg = f"the point {point!r} is not a MAP fit of an ordinal response"
            raise ValueError(msg)

    if comparison is None or comparison.held_out_unit not in REGIMES:
        msg = (
            f"the point {point!r} declares no comparison holding out "
            f"{SEEN_UNIT!r} or {NEW_UNIT!r}"
        )
        raise ValueError(msg)

    models = comparison.models

    if (
        models.candidate != name
        or frozenset(models.candidates) != frozenset(OBJECTIVES)
        or comparison.unseen_participant is not CONVENTIONS[name]
    ):
        msg = (
            f"the point {point!r} fits the {name!r} objective as candidate "
            f"{models.candidate!r} of {models.candidates!r} under the "
            f"{comparison.unseen_participant!r} convention, and each objective is "
            f"its own candidate under its own convention"
        )
        raise ValueError(msg)

    return Cell(name, comparison.held_out_unit)


def candidates(
    manifests: Mapping[str, JudgmentFitManifest], truth: KnownEffects
) -> dict[tuple[Cell, int], Candidate]:
    """Return every completed point by cell and seed, after the identity stop rule.

    Parameters
    ----------
    manifests : Mapping[str, JudgmentFitManifest]
        Every fit manifest of the results directory, by point.
    truth : KnownEffects
        The synthetic design the responses were drawn from.

    Raises
    ------
    StopRuleError
        If a fit's design provenance is not the synthetic design's, or its
        response model with the conditional objective is not the generating
        model.
    ValueError
        If a point is not a coupled MAP candidate, or two points share a cell
        and seed.
    """
    provenance = truth.read().provenance
    generating = spec_digest(truth.spec)
    found: dict[tuple[Cell, int], Candidate] = {}

    for point, manifest in manifests.items():
        cell = _cell(manifest)
        response = manifest.fit.response

        if manifest.design != provenance or (
            not isinstance(response, ResponseSpec)
            or spec_digest(response.with_(objective=ConditionalLikelihood()))
            != generating
        ):
            msg = (
                f"the point {point!r} fits another program or design than the "
                f"one the synthetic responses were drawn from"
            )
            raise StopRuleError(msg)

        seed = manifest.seed_derivation.seed
        key = (cell, seed)

        if key in found:
            msg = f"the cell {cell!r} has two completed points at seed {seed}"
            raise ValueError(msg)

        found[key] = Candidate(point, seed, cell, manifest)

    return found


def check_seen_coverage(
    rows: Sequence[Mapping[str, JsonValue]],
    found: Mapping[tuple[Cell, int], Candidate],
) -> None:
    """Refuse a response-held-out point whose folds hold out a new participant.

    Parameters
    ----------
    rows : Sequence[Mapping[str, JsonValue]]
        The experiment's report rows.
    found : Mapping[tuple[Cell, int], Candidate]
        Every completed point by cell and seed.

    Raises
    ------
    StopRuleError
        If a response-held-out point records a nonzero new-participant
        fraction for any fold.
    """
    seen = {
        candidate.point_id
        for (cell, _), candidate in found.items()
        if cell.unit == SEEN_UNIT
    }

    for row in rows:
        if row.get("metric") != HELD_OUT_PARTICIPANT_FRACTION_METRIC:
            continue

        fraction = FractionRow.model_validate(row)

        if fraction.point_id in seen and fraction.value > 0.0:
            msg = (
                f"the seen-participant point {fraction.point_id!r} holds out new "
                f"participants in fold {fraction.fold}, a fraction of "
                f"{fraction.value}"
            )
            raise StopRuleError(msg)


def participants(
    rows: Sequence[HeldOutRow], candidate: Candidate
) -> tuple[Participant, ...]:
    """Return one candidate's held-out density per participant over its folds.

    Parameters
    ----------
    rows : Sequence[HeldOutRow]
        Every held-out row of the experiment.
    candidate : Candidate
        The candidate whose rows are summed.

    Raises
    ------
    ValueError
        If the candidate records no held-out row, a row of another candidate
        or regime, or one participant twice in one fold.
    """
    regime = REGIMES[candidate.cell.unit]
    totals: dict[str, tuple[float, int]] = {}
    folds: set[tuple[int, str]] = set()

    for row in rows:
        if row.point_id != candidate.point_id:
            continue

        if row.candidate != candidate.cell.objective or row.participant_regime != (
            regime
        ):
            msg = (
                f"the point {candidate.point_id!r} records a "
                f"{row.participant_regime!r} row for candidate {row.candidate!r}, "
                f"and its cell is read under {regime!r} for "
                f"{candidate.cell.objective!r}"
            )
            raise ValueError(msg)

        if (row.fold, row.participant) in folds:
            msg = (
                f"the point {candidate.point_id!r} records participant "
                f"{row.participant!r} twice in fold {row.fold}"
            )
            raise ValueError(msg)

        folds.add((row.fold, row.participant))
        total, count = totals.get(row.participant, (0.0, 0))
        totals[row.participant] = (total + row.value * row.n, count + row.n)

    if not totals:
        msg = f"the point {candidate.point_id!r} records no held-out density"
        raise ValueError(msg)

    return tuple(
        Participant(participant, total, count)
        for participant, (total, count) in sorted(totals.items())
    )


def _mean(samples: Sequence[tuple[float, int]], /) -> float:
    """Return the density per response pooled over resampled participants."""
    return fsum(total for total, _ in samples) / sum(count for _, count in samples)


def interval(
    samples: Sequence[tuple[float, int]],
    declaration: BootstrapDeclaration,
    *,
    seed: int,
) -> tuple[BootstrapResult, Interval]:
    """Return the participant bootstrap of a pooled density and its interval.

    Parameters
    ----------
    samples : Sequence[tuple[float, int]]
        One summed log density and response count per participant.
    declaration : BootstrapDeclaration
        The declared bootstrap.
    seed : int
        The resampling seed.
    """
    result = Bootstrap(
        unit=UNIT,
        replicates=declaration.replicates,
        seed=seed,
        level=declaration.level,
    ).interval(samples, _mean)

    return result, result.bca if declaration.interval == "bca" else result.percentile


def paired(
    marginal: Sequence[Participant], conditional: Sequence[Participant]
) -> tuple[tuple[float, int], ...]:
    """Return each participant's marginal-minus-conditional summed density.

    Parameters
    ----------
    marginal : Sequence[Participant]
        The truncated-marginal candidate's participants.
    conditional : Sequence[Participant]
        The conditional candidate's participants.

    Raises
    ------
    ValueError
        If the candidates hold out different participants or response counts.
    """
    if [(row.participant, row.count) for row in marginal] != [
        (row.participant, row.count) for row in conditional
    ]:
        msg = "the two candidates hold out different participants or responses"
        raise ValueError(msg)

    return tuple(
        (left.total - right.total, left.count)
        for left, right in zip(marginal, conditional, strict=True)
    )


def _sites(layout: Sequence[Slot], vector: Tensor, /) -> dict[str, Tensor]:
    """Return a flat parameter vector divided into its named sites."""
    return {
        slot.site: values
        for slot, values in zip(
            layout,
            torch.split(vector, [slot.size for slot in layout]),
            strict=True,
        )
    }


def _layout(program: BoundProgram, design: JudgmentDesign, /) -> tuple[Slot, ...]:
    """Return the layout of the point a fit over every row of a design persists."""
    match program.program.objective:
        case ConditionalPlan():
            objective: PointObjective = ConditionalPointObjective(program)
        case TruncatedMarginalPlan():
            observed = design.participants.tolist()
            objective = MarginalPointObjective(
                program, active_groups=tuple(sorted(set(observed)))
            )

    return objective.layout


def recovery(
    candidate: Candidate,
    runs: Path,
    truth: KnownEffects,
    sites: Sequence[str],
) -> tuple[tuple[str, float, int], ...]:
    """Return each declared site's mean absolute error against the known effects.

    Parameters
    ----------
    candidate : Candidate
        The completed point whose persisted point is read.
    runs : Path
        The run store's directory holding one directory per point.
    truth : KnownEffects
        The synthetic design the responses were drawn from.
    sites : Sequence[str]
        The population sites compared.

    Returns
    -------
    tuple[tuple[str, float, int], ...]
        Each site's name, mean absolute error and element count.

    Raises
    ------
    ValueError
        If the persisted point is absent, is not the artifact its manifest
        names, or lacks a declared site.
    """
    artifact = candidate.manifest.artifact
    path = runs / candidate.point_id / artifact.name

    if artifact.name != POINT_ESTIMATE_NAME or not path.is_file():
        msg = f"the point {candidate.point_id!r} persists no MAP point at {path}"
        raise ValueError(msg)

    with path.open("rb") as persisted:
        digest = file_digest(persisted, "sha256").hexdigest()

    if digest != artifact.sha256:
        msg = (
            f"the persisted point of {candidate.point_id!r} has SHA-256 {digest}, "
            f"not its manifest digest {artifact.sha256}"
        )
        raise ValueError(msg)

    response = candidate.manifest.fit.response
    compiled = ResponseCompiler.installed().compile(response)
    design = truth.read().design
    fitted = _sites(
        _layout(compiled.over(design), design),
        load_file(path)["parameters"],
    )
    known = _sites(truth.program().layout, truth.generating())
    errors: list[tuple[str, float, int]] = []

    for site in sites:
        if site not in fitted or site not in known:
            msg = (
                f"the site {site!r} is not a population site of both the fitted "
                f"point of {candidate.point_id!r} and the generating program"
            )
            raise ValueError(msg)

        difference = fitted[site].to(known[site].dtype) - known[site]
        errors.append((site, float(difference.abs().mean()), difference.numel()))

    return tuple(errors)


class Lowering(NamedTuple):
    """The identity every report row of one comparison shares."""

    identity: ReportIdentity
    declaration: AnalysisDeclaration
    dataset: str


def _row(
    lowering: Lowering,
    *,
    method: str,
    metric: str,
    seed: int,
    result: MetricResult,
    pooled: PooledJudgment | None,
    convention: str | None,
) -> ReportRow:
    """Return one canonical report row of the comparison."""
    declaration = lowering.declaration

    return ReportRow(
        identity=lowering.identity,
        method=method,
        metric=metric,
        result=result,
        approximation=Approximation(NO_ENUMERATION_BUDGET, residual=None),
        suite=declaration.suite,
        task=declaration.task,
        dataset=lowering.dataset,
        split=declaration.split,
        seed=seed,
        held_out_convention=convention,
        pooled=pooled,
    )


def _bootstrapped(
    lowering: Lowering,
    samples: Sequence[tuple[float, int]],
    *,
    method: str,
    metric: str,
    seed: int,
    pooled: PooledJudgment,
    convention: str | None,
) -> tuple[ReportRow, BootstrapResult, Interval]:
    """Return a participant-bootstrapped density row pooled over its folds."""
    bootstrap = lowering.declaration.bootstrap
    resampling_seed = derive_seed(
        bootstrap.seed, f"{metric}:{method}:{pooled.rung}:{seed}"
    )
    result, bounds = interval(samples, bootstrap, seed=resampling_seed)
    row = _row(
        lowering,
        method=method,
        metric=metric,
        seed=seed,
        result=MetricResult(
            result.observed,
            result.units,
            Resampling(UNIT, seed=resampling_seed),
            ci=bounds,
        ),
        pooled=pooled,
        convention=convention,
    )

    return row, result, bounds


def rung_of(cell: Cell, /) -> str:
    """Return the rung a cell's rows are reported under."""
    return f"{cell.objective}/{cell.unit}"


def folds_of(rows: Sequence[HeldOutRow], candidate: Candidate) -> tuple[int, ...]:
    """Return the held-out folds a candidate's rows are pooled over.

    Parameters
    ----------
    rows : Sequence[HeldOutRow]
        Every held-out row of the experiment.
    candidate : Candidate
        The candidate whose folds are read.
    """
    return tuple(
        sorted({row.fold for row in rows if row.point_id == candidate.point_id})
    )


class Written(NamedTuple):
    """What one comparison wrote.

    Attributes
    ----------
    rows : int
        Report rows emitted.
    manifest : dict[str, JsonValue]
        The manifest written beside the report.
    """

    rows: int
    manifest: dict[str, JsonValue]


def read_manifests(directory: Path) -> dict[str, JudgmentFitManifest]:
    """Return every judgment-fit manifest of a results directory, by point.

    Parameters
    ----------
    directory : Path
        The experiment's results directory.
    """
    return {
        path.stem: decode_fit_manifest(path.read_bytes())
        for path in sorted((directory / MANIFESTS_NAME).glob("*.json"))
    }


def _statuses(directory: Path) -> dict[str, str]:
    """Return the status of every point the results directory records."""
    record = json.loads((directory / STATUS_NAME).read_text(encoding="utf-8"))
    points: dict[str, dict[str, JsonValue]] = record["points"]

    return {point: str(entry["status"]) for point, entry in points.items()}


def _contrast(
    lowering: Lowering,
    held: Sequence[HeldOutRow],
    found: Mapping[tuple[Cell, int], Candidate],
    *,
    unit: str,
    seed: int,
) -> tuple[ReportRow, Contrast] | None:
    """Return one seed's marginal-minus-conditional row under one unit."""
    marginal = found.get((Cell(MARGINAL, unit), seed))
    conditional = found.get((Cell(CONDITIONAL, unit), seed))

    if marginal is None or conditional is None:
        return None

    folds = folds_of(held, marginal)

    if folds != folds_of(held, conditional):
        msg = f"the two candidates under {unit!r} at seed {seed} hold out other folds"
        raise ValueError(msg)

    metric = AGREEMENT_METRIC if unit == SEEN_UNIT else ADVANTAGE_METRIC
    row, result, bounds = _bootstrapped(
        lowering,
        paired(participants(held, marginal), participants(held, conditional)),
        method=CONTRAST,
        metric=metric,
        seed=seed,
        pooled=PooledJudgment(
            rungs=(rung_of(marginal.cell), rung_of(conditional.cell)),
            specification_hashes=(
                spec_digest(marginal.manifest.fit),
                spec_digest(conditional.manifest.fit),
            ),
            folds=folds,
        ),
        convention=None,
    )
    margin = lowering.declaration.agreement_margin
    holds = (
        -margin <= bounds.low and bounds.high <= margin
        if unit == SEEN_UNIT
        else result.observed > (bounds.high - bounds.low) / 2.0
    )

    return row, Contrast(seed, unit, result.observed, bounds.low, bounds.high, holds)


def analyze(
    results: Path,
    source: Path,
    out: Path,
    /,
    *,
    runs: Path,
    truth: KnownEffects,
    table: str,
) -> Written:
    """Compute every row of the comparison and write the report and its manifest.

    Parameters
    ----------
    results : Path
        The experiment's results directory.
    source : Path
        The JSON analysis declaration.
    out : Path
        Directory receiving the report and the manifest.
    runs : Path
        The run store's directory holding one directory per point.
    truth : KnownEffects
        The synthetic design the responses were drawn from.
    table : str
        The name of the rows table within the results directory.

    Returns
    -------
    Written
        The row count and the manifest.

    Raises
    ------
    StopRuleError
        If a declared stop rule fires.
    ValueError
        If a point, a row or a persisted artifact is refused.
    """
    declaration = read_declaration(source)
    rows: tuple[dict[str, JsonValue], ...] = tuple(
        pq.read_table(results / table).to_pylist()
    )
    found = candidates(read_manifests(results), truth)
    check_seen_coverage(rows, found)
    undeclared = sorted({seed for _, seed in found} - frozenset(declaration.seeds))

    if undeclared:
        msg = f"completed points lie at undeclared seeds {undeclared!r}"
        raise ValueError(msg)

    held = tuple(
        HeldOutRow.model_validate(row)
        for row in rows
        if row.get("metric") == HELD_OUT_PREDICTIVE_METRIC
    )
    lowering = Lowering(
        ReportIdentity(
            declaration,
            DeclaredGrammar(declaration.grammar_fingerprint),
            None,
            freeze_id=declaration.freeze_id,
        ),
        declaration,
        truth.read().provenance.tag,
    )
    report: list[ReportRow] = []
    points: list[JsonValue] = []
    recovered: list[JsonValue] = []
    tolerance = declaration.recovery.tolerance

    for (cell, seed), candidate in sorted(found.items()):
        convention = (
            CONVENTIONS[cell.objective].value if cell.unit == NEW_UNIT else None
        )
        specification = spec_digest(candidate.manifest.fit)
        row, _, _ = _bootstrapped(
            lowering,
            tuple(
                (participant.total, participant.count)
                for participant in participants(held, candidate)
            ),
            method=cell.objective,
            metric=DENSITY_METRIC,
            seed=seed,
            pooled=PooledJudgment(
                rungs=(rung_of(cell),),
                specification_hashes=(specification,),
                folds=folds_of(held, candidate),
            ),
            convention=convention,
        )
        report.append(row)

        for site, error, elements in recovery(
            candidate, runs, truth, declaration.recovery.sites
        ):
            report.append(
                _row(
                    lowering,
                    method=f"{rung_of(cell)}:{site}",
                    metric=RECOVERY_METRIC,
                    seed=seed,
                    result=MetricResult(error, elements, Resampling(ELEMENT)),
                    pooled=None,
                    convention=None,
                )
            )
            recovered.append(
                {
                    "point_id": candidate.point_id,
                    "seed": seed,
                    "objective": cell.objective,
                    "held_out_unit": cell.unit,
                    "site": site,
                    "mean_absolute_error": error,
                    "recovered": error <= tolerance,
                }
            )

        manifest = candidate.manifest
        points.append(
            {
                "point_id": candidate.point_id,
                "seed": seed,
                "objective": cell.objective,
                "held_out_unit": cell.unit,
                "specification_hash": specification,
                "artifact": json.loads(manifest.artifact.model_dump_json()),
                "seed_derivation": json.loads(
                    manifest.seed_derivation.model_dump_json()
                ),
                "git_revision": manifest.git_revision,
            }
        )

    contrasts: list[JsonValue] = []

    for seed in declaration.seeds:
        seen = _contrast(lowering, held, found, unit=SEEN_UNIT, seed=seed)
        new = _contrast(lowering, held, found, unit=NEW_UNIT, seed=seed)

        report.extend(computed[0] for computed in (seen, new) if computed is not None)

        agreement = None if seen is None else seen[1]
        advantage = None if new is None else new[1]
        contrasts.append(
            {
                "seed": seed,
                AGREEMENT_METRIC: None if agreement is None else _verdict(agreement),
                ADVANTAGE_METRIC: (
                    None
                    if advantage is None
                    else {
                        **_verdict(advantage),
                        "read": agreement is not None and agreement.holds,
                    }
                ),
            }
        )

    statuses = _statuses(results)
    completed = {candidate.point_id for candidate in found.values()}
    cells: list[JsonValue] = []

    for objective in OBJECTIVES:
        for unit in (SEEN_UNIT, NEW_UNIT):
            completed_seeds: list[JsonValue] = []
            missing_seeds: list[JsonValue] = []

            for seed in declaration.seeds:
                if (Cell(objective, unit), seed) in found:
                    completed_seeds.append(seed)
                else:
                    missing_seeds.append(seed)

            cells.append(
                {
                    "objective": objective,
                    "held_out_unit": unit,
                    "completed_seeds": completed_seeds,
                    "missing_seeds": missing_seeds,
                }
            )

    out.mkdir(parents=True, exist_ok=True)
    ParquetReportSink(out / REPORT_NAME).emit(report)
    manifest: dict[str, JsonValue] = {
        "outcome": "completed",
        "declaration_path": source.name,
        "declaration_sha256": sha256(source.read_bytes()).hexdigest(),
        "results": str(results),
        "table": table,
        "design": json.loads(truth.read().provenance.model_dump_json()),
        "report": REPORT_NAME,
        "rows": len(report),
        "bootstrap": json.loads(declaration.bootstrap.model_dump_json()),
        "agreement_margin": declaration.agreement_margin,
        "points": points,
        "cells": cells,
        "unfinished": {
            point: status
            for point, status in sorted(statuses.items())
            if point not in completed
        },
        "contrasts": contrasts,
        "recovery": recovered,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )

    return Written(len(report), manifest)


def _verdict(contrast: Contrast, /) -> dict[str, JsonValue]:
    """Return one contrast's numbers and verdict as manifest JSON."""
    return {
        "difference": contrast.observed,
        "lower": contrast.low,
        "upper": contrast.high,
        "holds": contrast.holds,
    }


def _stopped(out: Path, source: Path, reason: str) -> None:
    """Write the manifest of a comparison a stop rule refused."""
    out.mkdir(parents=True, exist_ok=True)
    manifest: dict[str, JsonValue] = {
        "outcome": "stopped",
        "declaration_path": source.name,
        "declaration_sha256": sha256(source.read_bytes()).hexdigest(),
        "reason": reason,
    }
    (out / MANIFEST_NAME).write_text(
        json.dumps(manifest, indent=2) + "\n", encoding="utf-8", newline="\n"
    )


def main(argv: list[str] | None = None) -> int:
    """Write the comparison's report for one experiment's results.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments without the program name, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Zero after writing the report, one when a stop rule refused it.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "results", type=Path, help="the comparison experiment's results directory"
    )
    parser.add_argument(
        "--runs",
        type=Path,
        required=True,
        help="the run store's directory holding one directory per point",
    )
    parser.add_argument(
        "--declaration",
        type=Path,
        required=True,
        help="the JSON analysis declaration",
    )
    parser.add_argument(
        "--table",
        default=INGESTED_NAME,
        help="the rows table within the results directory",
    )
    parser.add_argument(
        "--out",
        type=Path,
        required=True,
        help="the directory receiving the report and the manifest",
    )
    arguments = parser.parse_args(argv)
    results: Path = arguments.results
    source: Path = arguments.declaration
    out: Path = arguments.out
    manifests = read_manifests(results)
    datasets = sorted({manifest.fit.dataset for manifest in manifests.values()})

    if len(datasets) != 1:
        msg = f"the comparison fits {datasets!r}, and it reads exactly one dataset"
        raise ValueError(msg)

    truth = Registry[KnownEffects](DESIGN_FAMILY).get(datasets[0])

    try:
        written = analyze(
            results,
            source,
            out,
            runs=arguments.runs,
            truth=truth,
            table=arguments.table,
        )
    except StopRuleError as stopped:
        _stopped(out, source, str(stopped))
        sys.stderr.write(f"stopped: {stopped}\n")

        return 1

    sys.stdout.write(
        f"{results}\n"
        f"  rows      {written.rows}\n"
        f"  report    {out / REPORT_NAME}\n"
        f"  manifest  {out / MANIFEST_NAME}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
