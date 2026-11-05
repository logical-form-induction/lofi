"""Reports the MORCELA-linked acceptability baseline against the agreement ceiling.

The pilot is one judgment-fit experiment whose declared comparison set holds a
candidate that enters a language model's sentence scores through the MORCELA
linking. This script reads that experiment's results directory and writes one
report holding four computed quantities and the decision they settle.

The correlation is the Spearman correlation, over items, between each item's
mean observed response and the mean of the candidate's predicted expected
response over the same rows, with a bias-corrected and accelerated bootstrap
interval over items. The predictive density is the candidate's held-out
predictive log density from the comparison, summed within each participant
regime, or its leave-one-out density where the comparison was posterior. The
ceiling is the record ``scripts/judgment/agreement_ceiling.py`` computed, which
must have been computed from files the candidate's design read. Every member
of the comparison set must share the candidate's ordinal head, differing from it
only in the terms of its linear predictor.

The outcome is the decision rule applied to the correlation's interval and the
ceiling. Reading a report applies the rule again and refuses a report whose
recorded outcome or rule disagrees with its own numbers.

Run it on an ingested pilot experiment::

    python scripts/judgment/morcela_pilot.py results/1/morcela-pilot
        --candidate morcela --ceiling ceiling.json --seed 1
        --out artifacts/judgment/morcela-pilot/report.json
"""

import json
import sys
from argparse import ArgumentParser
from enum import StrEnum
from pathlib import Path
from statistics import fmean
from typing import TYPE_CHECKING, Final, NamedTuple

import didactic.api as dx
import pyarrow.parquet as pq

from lofi.judgment import (
    LinkTerm,
    ResponseSpec,
    decode_fit_manifest,
    rank_correlation,
)
from lofi.judgment.morcela import FAMILY
from lofi.run.fits import HELD_OUT_PREDICTIVE_METRIC
from lofi.run.results import INGESTED_NAME, MANIFESTS_NAME
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.judgment import JudgmentFitManifest
    from lofi.stats import BootstrapResult

PREDICTION_METRIC: Final = "response-prediction"
"""The metric of a row holding one response's predicted distribution."""

LOO_METRIC: Final = "psis-loo-elpd"
"""The metric of a posterior candidate's leave-one-out density row."""

COEFFICIENT: Final = (
    "the Spearman rank correlation, over items, between each item's mean "
    "observed response and the mean over the same rows of the candidate's "
    "predicted expected response level"
)
"""Describes the reported correlation."""

UNIT: Final = "item"
"""The unit the correlation's bootstrap resamples."""

RULE: Final = (
    "If the bootstrap interval of the linked language model's correlation with "
    "the acceptability responses covers the verified ceiling, the judgment paper "
    "is written as a comparison paper with the language model as a member of the "
    "comparison set; otherwise it is written as a prediction paper."
)
"""The written decision rule the report applies."""


class Outcome(StrEnum):
    """The framing the decision rule settles."""

    COMPARISON = "comparison"
    PREDICTION = "prediction"


class CeilingRecord(dx.Model, extra="forbid"):
    """The agreement ceiling as its computing script records it."""

    value: float
    lower: float
    upper: float
    level: float
    splits: int
    items: int
    responses: int
    coefficient: str
    dataset: str
    group_by: tuple[str, ...]
    source_digests: dict[str, str]


class Correlation(dx.Model, extra="forbid"):
    """The linked model's correlation with the responses and its interval."""

    coefficient: str
    value: float
    lower: float
    upper: float
    level: float
    unit: str
    items: int
    replicates: int
    seed: int


class PredictiveDensity(dx.Model, extra="forbid"):
    """The candidate's predictive log density within one participant regime."""

    metric: str
    regime: str | None
    total: float
    observations: int


class PilotReport(dx.Model, extra="forbid"):
    """The pilot's numbers, the written rule and the outcome the rule gives."""

    freeze_id: str
    point_id: str
    candidate: str
    dataset: str
    checkpoint: str | None
    git_revision: str
    correlation: Correlation
    densities: tuple[PredictiveDensity, ...]
    ceiling: CeilingRecord
    rule: str
    outcome: Outcome

    @dx.model_validator()
    def _outcome_is_the_rule_applied(self) -> PilotReport:
        """Refuse a rule or outcome that the recorded numbers do not give."""
        if self.rule != RULE:
            msg = f"the report states another rule than the pilot's: {self.rule!r}"
            raise ValueError(msg)

        decided = decide(self.correlation, self.ceiling)

        if self.outcome != decided:
            msg = (
                f"the report records the outcome {self.outcome.value!r}, and its "
                f"interval [{self.correlation.lower}, {self.correlation.upper}] "
                f"against the ceiling {self.ceiling.value} gives {decided.value!r}"
            )
            raise ValueError(msg)

        return self


class PredictionRow(dx.Model, extra="ignore"):
    """One response's predicted distribution, as a fit point records it."""

    point_id: str
    item: str
    response: int
    distribution: tuple[float, ...]


class HeldOutRow(dx.Model, extra="ignore"):
    """One participant's mean held-out predictive log density within one fold.

    The value is the mean over that participant's held-out responses in the
    fold, under one candidate and participant regime, and ``n`` is the number
    of those responses.
    """

    freeze_id: str
    point_id: str
    candidate: str
    participant_regime: str
    value: float
    n: int


class LooRow(dx.Model, extra="ignore"):
    """One posterior candidate's leave-one-out summary."""

    freeze_id: str
    point_id: str
    model: str
    elpd: float
    observations: int


class Density(NamedTuple):
    """A predictive log density summed over one participant regime."""

    metric: str
    regime: str | None
    total: float
    observations: int


class Compared(NamedTuple):
    """The candidate's point, the points of its comparison set and its densities."""

    freeze_id: str
    point_id: str
    members: tuple[str, ...]
    densities: tuple[Density, ...]


def decide(correlation: Correlation, ceiling: CeilingRecord) -> Outcome:
    """Apply the written decision rule.

    Parameters
    ----------
    correlation : Correlation
        The linked model's correlation and its interval.
    ceiling : CeilingRecord
        The verified agreement ceiling.

    Returns
    -------
    Outcome
        A comparison paper when the interval covers the ceiling, and a
        prediction paper otherwise.
    """
    if correlation.lower <= ceiling.value <= correlation.upper:
        return Outcome.COMPARISON

    return Outcome.PREDICTION


def read_report(path: Path) -> PilotReport:
    """Return a written report after applying its decision rule again.

    Parameters
    ----------
    path : Path
        The report's JSON file.

    Returns
    -------
    PilotReport
        The validated report.

    Raises
    ------
    ValueError
        If the file does not validate as a report, or its rule or outcome
        disagrees with its numbers.
    """
    try:
        return PilotReport.model_validate_json(path.read_bytes())
    except dx.ValidationError as refused:
        msg = f"{path} is not a pilot report its own numbers support: {refused}"
        raise ValueError(msg) from refused


def read_ceiling(path: Path) -> CeilingRecord:
    """Return the agreement ceiling a ceiling record holds.

    Parameters
    ----------
    path : Path
        The JSON record the ceiling script wrote.

    Raises
    ------
    ValueError
        If the file does not validate as a ceiling record.
    """
    try:
        return CeilingRecord.model_validate_json(path.read_bytes())
    except dx.ValidationError as refused:
        msg = f"{path} is not an agreement-ceiling record: {refused}"
        raise ValueError(msg) from refused


def read_rows(path: Path) -> tuple[dict[str, JsonValue], ...]:
    """Return a results table's rows as serialized report columns.

    Parameters
    ----------
    path : Path
        The Parquet table of an experiment's report rows.
    """
    decoded: list[dict[str, JsonValue]] = pq.read_table(path).to_pylist()

    return tuple(decoded)


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


def _held_out(
    rows: Sequence[dict[str, JsonValue]], candidate: str, members: tuple[str, ...]
) -> Compared | None:
    """Return the candidate's held-out comparison, if the rows hold one."""
    held = tuple(
        HeldOutRow.model_validate(row)
        for row in rows
        if row.get("metric") == HELD_OUT_PREDICTIVE_METRIC
    )
    own = tuple(row for row in held if row.candidate == candidate)

    if not own:
        return None

    points = {row.point_id for row in own}

    if len(points) != 1:
        msg = (
            f"the candidate {candidate!r} has held-out rows at the points "
            f"{sorted(points)!r}, and the pilot reports one"
        )
        raise ValueError(msg)

    regimes: dict[str, tuple[float, int]] = {}

    for row in own:
        total, observations = regimes.get(row.participant_regime, (0.0, 0))
        regimes[row.participant_regime] = (
            total + row.value * row.n,
            observations + row.n,
        )

    return Compared(
        own[0].freeze_id,
        own[0].point_id,
        tuple(sorted({row.point_id for row in held if row.candidate in members})),
        tuple(
            Density(HELD_OUT_PREDICTIVE_METRIC, regime, total, observations)
            for regime, (total, observations) in sorted(regimes.items())
        ),
    )


def _leave_one_out(
    rows: Sequence[dict[str, JsonValue]], candidate: str, members: tuple[str, ...]
) -> Compared | None:
    """Return the candidate's leave-one-out comparison, if the rows hold one."""
    loo = tuple(
        LooRow.model_validate(row) for row in rows if row.get("metric") == LOO_METRIC
    )
    own = tuple(row for row in loo if row.model == candidate)

    if not own:
        return None

    if len(own) != 1:
        msg = (
            f"the candidate {candidate!r} has {len(own)} leave-one-out rows, and "
            f"the pilot reports one"
        )
        raise ValueError(msg)

    [summary] = own

    return Compared(
        summary.freeze_id,
        summary.point_id,
        tuple(sorted({row.point_id for row in loo if row.model in members})),
        (Density(LOO_METRIC, None, summary.elpd, summary.observations),),
    )


def compared(
    rows: Sequence[dict[str, JsonValue]], candidate: str, members: tuple[str, ...]
) -> Compared:
    """Return the candidate's point, its comparison members and its densities.

    Parameters
    ----------
    rows : Sequence[dict[str, JsonValue]]
        The experiment's report rows.
    candidate : str
        The candidate the pilot reports.
    members : tuple[str, ...]
        The candidates the candidate's comparison set declares.

    Raises
    ------
    ValueError
        If the rows hold no comparison of the candidate, or hold it at more
        than one point.
    """
    found = tuple(
        comparison
        for comparison in (
            _held_out(rows, candidate, members),
            _leave_one_out(rows, candidate, members),
        )
        if comparison is not None
    )

    if len(found) != 1:
        msg = (
            f"the rows hold {len(found)} completed comparisons of the candidate "
            f"{candidate!r}, and the pilot reports exactly one"
        )
        raise ValueError(msg)

    return found[0]


def _head(manifest: JudgmentFitManifest, point: str) -> ResponseSpec:
    """Return a point's ordinal response specification."""
    match manifest.fit.response:
        case ResponseSpec() as response:
            return response
        case _:
            msg = f"the point {point!r} does not fit an ordinal response head"
            raise ValueError(msg)


def _without_predictor(response: ResponseSpec) -> tuple[object, ...]:
    """Return every part of a response head except its linear predictor."""
    return (
        response.family,
        response.levels,
        response.identification,
        response.objective,
        response.priors,
        response.normalization,
        response.parameters,
    )


def check_head(
    manifests: dict[str, JudgmentFitManifest], point: str, members: tuple[str, ...]
) -> JudgmentFitManifest:
    """Return the candidate's manifest once its head and linking are checked.

    Parameters
    ----------
    manifests : dict[str, JudgmentFitManifest]
        Every manifest of the results directory, by point.
    point : str
        The candidate's point.
    members : tuple[str, ...]
        Every point of the candidate's comparison set, the candidate's included.

    Raises
    ------
    ValueError
        If a member has no manifest, a member's ordinal head differs from the
        candidate's other than in its linear predictor, or the candidate's
        predictor reads no score through the MORCELA linking.
    """
    missing = [member for member in members if member not in manifests]

    if missing:
        msg = f"the compared points {missing!r} have no manifest"
        raise ValueError(msg)

    manifest = manifests[point]
    head = _head(manifest, point)
    linked: list[str] = []

    for term in head.predictor:
        match term:
            case LinkTerm(readout=readout, linking=linking) if linking.family == FAMILY:
                linked.append(readout)
            case _:
                pass

    if not linked:
        msg = f"the point {point!r} reads no score through the {FAMILY!r} linking"
        raise ValueError(msg)

    for member in members:
        if _without_predictor(_head(manifests[member], member)) != _without_predictor(
            head
        ):
            msg = (
                f"the compared point {member!r} fits another ordinal head than "
                f"the candidate's point {point!r}"
            )
            raise ValueError(msg)

    return manifest


def check_ceiling(ceiling: CeilingRecord, manifest: JudgmentFitManifest) -> None:
    """Refuse a ceiling computed from a file the candidate's design did not read.

    Parameters
    ----------
    ceiling : CeilingRecord
        The agreement ceiling.
    manifest : JudgmentFitManifest
        The candidate's manifest, recording the digests of its design's files.

    Raises
    ------
    ValueError
        If the ceiling was computed from a file the design did not read.
    """
    read = frozenset(manifest.design.corpus_digests.values())
    unread = sorted(
        path for path, digest in ceiling.source_digests.items() if digest not in read
    )
    if not ceiling.source_digests or unread:
        msg = (
            f"the ceiling was computed from {unread or 'no file'!r}, which the "
            f"design {manifest.design.tag!r} did not read"
        )
        raise ValueError(msg)


def item_means(
    rows: Sequence[dict[str, JsonValue]], point: str
) -> tuple[tuple[float, float], ...]:
    """Return each item's mean predicted and mean observed response level.

    Parameters
    ----------
    rows : Sequence[dict[str, JsonValue]]
        The experiment's report rows.
    point : str
        The candidate's point.

    Returns
    -------
    tuple[tuple[float, float], ...]
        One pair per item in first-appearance order: the mean over the item's
        rows of the predicted expected level, and the mean observed level.

    Raises
    ------
    ValueError
        If the point records no predicted response distributions.
    """
    predicted: dict[str, list[tuple[float, float]]] = {}

    for row in rows:
        if row.get("metric") != PREDICTION_METRIC or row.get("point_id") != point:
            continue

        prediction = PredictionRow.model_validate(row)
        expected = sum(
            level * probability
            for level, probability in enumerate(prediction.distribution)
        )
        predicted.setdefault(prediction.item, []).append(
            (expected, float(prediction.response))
        )

    if not predicted:
        msg = f"the point {point!r} records no predicted response distributions"
        raise ValueError(msg)

    return tuple(
        (fmean(pair[0] for pair in pairs), fmean(pair[1] for pair in pairs))
        for pairs in predicted.values()
    )


def _spearman(pairs: Sequence[tuple[float, float]]) -> float:
    """Return the rank correlation between the two sides of item pairs."""
    return rank_correlation([pair[0] for pair in pairs], [pair[1] for pair in pairs])


def correlate(
    pairs: Sequence[tuple[float, float]], *, replicates: int, seed: int, level: float
) -> BootstrapResult:
    """Return the item-level correlation with its accelerated bootstrap interval.

    Parameters
    ----------
    pairs : Sequence[tuple[float, float]]
        One predicted and observed mean per item.
    replicates : int
        How many resamples of the items the interval is read off.
    seed : int
        The seed the resampling runs under.
    level : float
        The coverage the interval aims at.
    """
    return Bootstrap(unit=UNIT, replicates=replicates, seed=seed, level=level).interval(
        pairs, _spearman
    )


def pilot_report(
    directory: Path,
    *,
    table: str,
    candidate: str,
    ceiling: CeilingRecord,
    replicates: int,
    seed: int,
    level: float,
) -> PilotReport:
    """Return the pilot's report from an experiment's results directory.

    Parameters
    ----------
    directory : Path
        The experiment's results directory.
    table : str
        The name of the rows table within it.
    candidate : str
        The MORCELA-linked candidate of the comparison set.
    ceiling : CeilingRecord
        The verified agreement ceiling.
    replicates : int
        How many item resamples the correlation's interval is read off.
    seed : int
        The seed the resampling runs under.
    level : float
        The coverage the correlation's interval aims at.

    Raises
    ------
    ValueError
        If the candidate's comparison, manifest, head, linking or ceiling is
        refused.
    """
    rows = read_rows(directory / table)
    manifests = read_manifests(directory)
    declared = tuple(
        manifest.fit.comparison.models.candidates
        for manifest in manifests.values()
        if manifest.fit.comparison is not None
        and manifest.fit.comparison.models.candidate == candidate
    )

    if len(set(declared)) != 1:
        msg = (
            f"the candidate {candidate!r} is declared by {len(set(declared))} "
            f"comparison sets, and the pilot reports one"
        )
        raise ValueError(msg)

    comparison = compared(rows, candidate, declared[0])
    manifest = check_head(manifests, comparison.point_id, comparison.members)
    check_ceiling(ceiling, manifest)
    pairs = item_means(rows, comparison.point_id)
    drawn = correlate(pairs, replicates=replicates, seed=seed, level=level)
    correlation = Correlation(
        coefficient=COEFFICIENT,
        value=drawn.observed,
        lower=drawn.bca.low,
        upper=drawn.bca.high,
        level=level,
        unit=drawn.unit,
        items=drawn.units,
        replicates=drawn.replicates,
        seed=drawn.seed,
    )

    return PilotReport(
        freeze_id=comparison.freeze_id,
        point_id=comparison.point_id,
        candidate=candidate,
        dataset=manifest.fit.dataset,
        checkpoint=manifest.design.checkpoint,
        git_revision=manifest.git_revision,
        correlation=correlation,
        densities=tuple(
            PredictiveDensity(
                metric=density.metric,
                regime=density.regime,
                total=density.total,
                observations=density.observations,
            )
            for density in comparison.densities
        ),
        ceiling=ceiling,
        rule=RULE,
        outcome=decide(correlation, ceiling),
    )


def main(argv: list[str] | None = None) -> int:
    """Write the pilot's report for one experiment's results.

    Parameters
    ----------
    argv : list[str] | None
        Command-line arguments without the program name, or ``None`` for ``sys.argv``.

    Returns
    -------
    int
        Zero after writing the report.
    """
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "results", type=Path, help="the pilot experiment's results directory"
    )
    parser.add_argument(
        "--table",
        default=INGESTED_NAME,
        help="the rows table within the results directory",
    )
    parser.add_argument(
        "--candidate",
        required=True,
        help="the MORCELA-linked candidate of the declared comparison set",
    )
    parser.add_argument(
        "--ceiling",
        type=Path,
        required=True,
        help="the record the agreement-ceiling script wrote",
    )
    parser.add_argument(
        "--replicates",
        type=int,
        default=999,
        help="how many item resamples the correlation's interval is read off",
    )
    parser.add_argument(
        "--seed", type=int, required=True, help="the seed of the item resampling"
    )
    parser.add_argument(
        "--level",
        type=float,
        default=0.95,
        help="the coverage the correlation's interval aims at, as a proportion",
    )
    parser.add_argument(
        "--out", type=Path, required=True, help="where to write the JSON report"
    )
    arguments = parser.parse_args(argv)

    report = pilot_report(
        arguments.results,
        table=arguments.table,
        candidate=arguments.candidate,
        ceiling=read_ceiling(arguments.ceiling),
        replicates=arguments.replicates,
        seed=arguments.seed,
        level=arguments.level,
    )
    destination: Path = arguments.out
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(
        json.dumps(json.loads(report.model_dump_json()), indent=2) + "\n",
        encoding="utf-8",
        newline="\n",
    )
    correlation = report.correlation
    sys.stdout.write(
        f"{report.point_id} ({report.candidate}) at {report.git_revision}\n"
        f"  correlation {correlation.value:.4f} "
        f"[{correlation.lower:.4f}, {correlation.upper:.4f}] over "
        f"{correlation.items} items\n"
        f"  ceiling     {report.ceiling.value:.4f}\n"
        f"  outcome     {report.outcome.value}\n"
        f"  written to  {destination}\n"
    )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
