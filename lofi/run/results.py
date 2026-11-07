"""An experiment's results, written from its run store and ingested against its freeze.

An experiment's results arrive as one directory,
``<root>/results/<n>/<experiment>/``, where ``n`` is the number of
the freeze ``freeze/<n>`` the experiment runs against. The directory holds
the report rows as ``rows.parquet``, one manifest per point as
``manifests/<point>.json``, the launcher's ``STATUS.json``, and beside the
rows ``measured.parquet``: one row per submitted point whose directory records a
metrics record, with the point's measured parameter counts and every value its
record holds outside a sequence. A manifest is the one the experiment's
configuration kind writes: an experiment manifest for a run document, a
judgment fit manifest for a judgment fit document, a baseline fit manifest for a
baseline fit document, and an active learning manifest for an active learning
document, whose rows are its rounds.

Ingestion is anchored to the freeze's ``FREEZE.json``. The experiment's sweep is
expanded again and must reproduce the sweep identifier the freeze recorded for
it, or the sweep its recorded amendments reached from that one, so the frozen
points are the points of that expansion. A manifest's name
must be the point identifier recomputed from the configuration and seed it
records, that point must be frozen, its run seed must have been derived
from the experiment's root seed, and the source revision it records must be the
commit the freeze tags. A row must be reported against the experiment's freeze,
never ``UNFROZEN``, carry the configuration digest of a frozen point, and state
a budget and residual that rebuild as an approximation.

The directory is written from the run store the experiment executed under, read
through the experiment's store. Every point of the expanded sweep has one status
entry: completed, running, killed and awaiting resubmission, a recorded
numerical or infrastructure failure with its diagnostic, submitted and not yet
started, an active learning run awaiting the responses to a batch it fielded,
or not submitted. Submission is read from the binding each point's
directory records when it is submitted, so a point bound to another freeze is
not submitted for this one. Every submitted point whose directory records a
manifest contributes it, failed points included, and every completed point
contributes the rows it recorded, so a failed point is a reported outcome
rather than a missing row.

Ingestion also reads the experiment's evaluation section. The comparisons the
experiment reports are corrected within each family by the declared correction
at the declared level and under the declared dependence regime, and the
ingested results hold those decisions beside the fraction of points that
completed. Before an experiment is frozen, a pilot's results directory prices
its seed count: each pilot cell's spread in the budgeted metric gives the seeds
the declared precision needs, and the largest is written beside the seed floor
in ``<experiment>.seed-budget.json`` next to the experiment document.

A pilot runs an experiment outside any freeze, and its results directory lies
apart from every frozen one, at ``<root>/pilot/results/<experiment>/``. Its
``STATUS.json`` records the freeze as ``UNFROZEN`` and so do its rows. Pilot
results can be rendered, but ingestion refuses them with
``PilotResultsError``, whatever directory they are found in.
"""

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import pyarrow as pa
import pyarrow.parquet as pq

from lofi._files import replace_path
from lofi.eval import (
    NO_ENUMERATION_BUDGET,
    RESIDUAL_SOURCES,
    UNFROZEN,
    Approximation,
    DeclaredBudget,
    NoEnumerationBudget,
    PreregistrationAudit,
    Residual,
    ResidualSummary,
    check_preregistration,
    config_digest,
    freeze_number,
    residual_summary,
    seeds_required,
)
from lofi.eval._reducers import ReductionContext, reduce_experiment_rows
from lofi.run._boundary.metrics import metrics_row, read_metrics
from lofi.run._boundary.sweep import point_id
from lofi.run._commitments import assess
from lofi.run._contraction import SeedFloor
from lofi.run._experiment import expand_experiment, read_experiment
from lofi.run._launch import sweep_id
from lofi.run._manifest import read_rows, replace_file
from lofi.run._ports import Ingested
from lofi.run._result_points import ResultsLayoutError, comparison_points
from lofi.run._store import PointStatus

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from datetime import date
    from pathlib import Path

    from lofi._types import JsonValue
    from lofi.eval import Comparison, ResidualSource
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._experiment import (
        EvaluationPlan,
        Experiment,
        ExperimentConfiguration,
        ManifestPoint,
    )
    from lofi.run._seed import ExperimentSeed
    from lofi.run._store import ExperimentStore
    from lofi.stats import Decisions, PValue

__all__ = [
    "AUDIT_NAME",
    "INGESTED_NAME",
    "MANIFESTS_NAME",
    "MEASURED_NAME",
    "PILOT_NAME",
    "RESULTS_NAME",
    "ROWS_NAME",
    "SEED_BUDGET_SUFFIX",
    "STATUS_NAME",
    "ApproximationError",
    "FreezeMismatchError",
    "FreezeRecord",
    "FreezeRecordError",
    "IngestedResults",
    "PilotResultsError",
    "ResultsLayoutError",
    "SeedBudgetRecord",
    "UnfrozenConfigurationError",
    "frozen_experiment",
    "ingest_experiment",
    "ingest_results",
    "pilot_directory",
    "price_seed_budget",
    "require_frozen_results",
    "results_directory",
    "write_ingested",
    "write_results",
    "write_seed_budget",
]

ROWS_NAME: Final = "rows.parquet"
"""The file holding an experiment's report rows."""

MEASURED_NAME: Final = "measured.parquet"
"""The file holding one row of measured and achieved values per point."""

MANIFESTS_NAME: Final = "manifests"
"""The directory holding one experiment manifest per point."""

STATUS_NAME: Final = "STATUS.json"
"""The file holding the launcher's completion status."""

INGESTED_NAME: Final = "ingested.parquet"
"""The file holding an experiment's accepted rows."""

AUDIT_NAME: Final = "AUDIT.json"
"""The file holding the audit of an experiment's accepted rows."""

RESULTS_NAME: Final = "results"
"""The directory under a repository root that holds every experiment's results."""

PILOT_NAME: Final = "pilot"
"""The directory under a repository root that every pilot runs and reports under."""

SEED_BUDGET_SUFFIX: Final = ".seed-budget.json"
"""The suffix of the pre-flight record written beside an experiment document."""


class FreezeRecordError(ValueError):
    """Signals a freeze record that is unreadable or does not record the experiment."""


class FreezeMismatchError(ValueError):
    """Signals an experiment, sweep or row that disagrees with the freeze it names."""


class UnfrozenConfigurationError(ValueError):
    """Signals a row or a manifest whose configuration is not a frozen point."""


class ApproximationError(ValueError):
    """Signals a row whose budget and residual do not rebuild as an approximation."""


class PilotResultsError(ValueError):
    """Signals pilot results, run outside any freeze, offered as frozen results."""


class IngestedResults(NamedTuple):
    """An experiment's accepted results.

    Attributes
    ----------
    freeze_id : str
        The freeze the directory and every row are reported against.
    experiment : str
        The experiment name.
    rows : tuple[Mapping[str, JsonValue], ...]
        The report rows, as serialized report columns, in file order.
    points : tuple[Mapping[str, JsonValue], ...]
        One row per recorded point: its identifier, configuration digest and
        status, its measured parameter counts, and the values its metrics
        record holds outside a sequence, in file order.
    residuals : ResidualSummary
        How many rows measured every residual source, and the unmeasured
        count per source.
    audit : PreregistrationAudit
        The comparisons' coverage of the preregistered plan.
    decisions : Mapping[str, Decisions]
        The experiment's declared correction applied within each family of
        comparisons at its declared level, by family, each family's tests in
        the order the comparisons were given.
    completed_fraction : float
        The fraction of the experiment's points that completed.
    fallback : str | None
        The fallback claim of the experiment's decision point once its
        fallback rung has been applied, or None while its declared claim
        stands.
    """

    freeze_id: str
    experiment: str
    rows: tuple[Mapping[str, JsonValue], ...]
    points: tuple[Mapping[str, JsonValue], ...]
    residuals: ResidualSummary
    audit: PreregistrationAudit
    decisions: Mapping[str, Decisions]
    completed_fraction: float
    fallback: str | None


def results_directory(root: Path, experiment: Experiment, /) -> Path:
    """Return the directory an experiment's results are written to.

    Parameters
    ----------
    root : Path
        The repository root.
    experiment : Experiment
        The experiment whose results the directory holds.

    Returns
    -------
    Path
        ``<root>/results/<n>/<experiment>``, for the experiment's freeze
        ``freeze/<n>``.
    """
    return root.joinpath(RESULTS_NAME, *_results_parts(experiment))


def pilot_directory(root: Path, experiment: Experiment, /) -> Path:
    """Return the directory a pilot of an experiment writes its results to.

    Parameters
    ----------
    root : Path
        The repository root.
    experiment : Experiment
        The experiment the pilot runs.

    Returns
    -------
    Path
        ``<root>/pilot/results/<experiment>``, which names no freeze.
    """
    return root.joinpath(PILOT_NAME, RESULTS_NAME, experiment.name)


def require_frozen_results(directory: Path, /) -> None:
    """Refuse a results directory whose status marks it a pilot's.

    Parameters
    ----------
    directory : Path
        The results directory offered as an experiment's frozen results.

    Raises
    ------
    PilotResultsError
        If its ``STATUS.json`` records the freeze as ``UNFROZEN``.
    """
    path = directory / STATUS_NAME

    try:
        status: JsonValue = json.loads(path.read_bytes())

    except OSError, json.JSONDecodeError:
        return

    if isinstance(status, dict) and status.get("freeze_id") == UNFROZEN:
        msg = (
            f"the results at {directory} are a pilot's, run outside any freeze, "
            "and only results run against a freeze are accepted"
        )
        raise PilotResultsError(msg)


def _results_parts(experiment: Experiment, /) -> tuple[str, str]:
    """Return the freeze number and name an experiment's results are filed by.

    Parameters
    ----------
    experiment : Experiment
        The experiment whose results are filed.

    Returns
    -------
    tuple[str, str]
        The number of its freeze ``freeze/<n>``, and its name.
    """
    return (str(freeze_number(experiment.freeze_id)), experiment.name)


def write_results(
    directory: Path,
    *,
    points: Sequence[SweepPoint],
    store: ExperimentStore,
    today: date,
) -> Mapping[str, JsonValue]:
    """Write an experiment's results directory from the store it executed under.

    Every file is replaced whole, so the directory is rewritten on every
    poll and a reader sees either the previous or the current version of a
    file. The status names the freeze the store reports rows against,
    ``UNFROZEN`` for a pilot's store. The rows table holds the rows the
    experiment's configuration kind reports from its completed points' rows,
    with every column any row carries, null in a row that does not carry it,
    and so does the
    measured table, which holds one row
    per submitted point whose directory records a metrics record. A cell is
    one configuration across its seeds; the
    status records the count of points in each status and the completed
    fraction, for the experiment and for every cell, the names of the stop
    rules that fire on that status and the rows, and the state of the
    decision point on the given day.

    Parameters
    ----------
    directory : Path
        The experiment's results directory, created if absent.
    points : Sequence[SweepPoint]
        The points of the experiment's expanded sweep, in expansion order.
    store : ExperimentStore
        The store each point's directory lies under, read for the experiment.
    today : date
        The day the decision point is read on.

    Returns
    -------
    Mapping[str, JsonValue]
        The status record written.

    Raises
    ------
    ReseededRunError
        If a point's directory was seeded under another derivation.
    ValueError
        If a point's directory records malformed rows, bindings or failures.
    """
    experiment = store.experiment
    manifests = directory / MANIFESTS_NAME
    manifests.mkdir(parents=True, exist_ok=True)
    rows: list[Mapping[str, JsonValue]] = []
    measured: list[Mapping[str, JsonValue]] = []
    entries: dict[str, JsonValue] = {}
    statuses: dict[str, str] = {}
    cells: dict[str, list[str]] = {}

    for point in points:
        entry = store.entry(point)
        status = cast("str", entry["status"])
        manifest = store.manifest(point)

        if status != PointStatus.NOT_SUBMITTED.value and manifest is not None:
            replace_file(manifests / f"{point.point_id}.json", manifest)

        if status != PointStatus.NOT_SUBMITTED.value:
            record = read_metrics(store.directory(point))

            if record is not None:
                measured.append(
                    {
                        "point_id": point.point_id,
                        "config_hash": config_digest(point.config),
                        "status": status,
                        **metrics_row(record),
                    }
                )

        if status == PointStatus.COMPLETED.value:
            recorded = read_rows(store.directory(point))

            if recorded is not None:
                rows.extend(recorded.rows)

        entries[point.point_id] = entry
        statuses[point.point_id] = status
        cells.setdefault(config_digest(point.config), []).append(point.point_id)

    rows = list(experiment.config.result_rows(rows, points, store))
    _write_table(directory / ROWS_NAME, rows)
    _write_table(directory / MEASURED_NAME, measured)
    record: dict[str, JsonValue] = {
        "freeze_id": store.freeze_id,
        "experiment": experiment.name,
        "sweep_id": sweep_id(points),
        **_tally([statuses[point.point_id] for point in points]),
        "cells": {
            cell: {
                "points": cast("list[JsonValue]", members),
                **_tally([statuses[member] for member in members]),
            }
            for cell, members in cells.items()
        },
        "points": entries,
    }
    assessment = assess(
        experiment.stop_rules,
        experiment.decision_point,
        today=today,
        status=record,
        rows=rows,
    )
    record["stop_rules"] = list(assessment.fired)
    record["decision"] = assessment.decision.value
    status = json.dumps(record, indent=2)
    replace_file(directory / STATUS_NAME, f"{status}\n".encode())

    return record


def _write_table(path: Path, rows: Sequence[Mapping[str, JsonValue]], /) -> None:
    """Replace a parquet table with rows holding every column any row carries.

    Parameters
    ----------
    path : Path
        The table file, under a directory that exists.
    rows : Sequence[Mapping[str, JsonValue]]
        The rows, each null in a column it does not carry.
    """
    columns = dict.fromkeys(column for row in rows for column in row)
    staged = path.with_name(f".{path.name}.staged")
    pq.write_table(
        pa.Table.from_pylist(
            [{name: row.get(name) for name in columns} for row in rows]
        ),
        staged,
    )
    replace_path(staged, path)


def _tally(statuses: Sequence[str], /) -> dict[str, JsonValue]:
    """Return how many points are offered, in each status, and completed."""
    counts: dict[str, JsonValue] = {}

    for status in statuses:
        counts[status] = cast("int", counts.get(status, 0)) + 1

    completed = cast("int", counts.get(PointStatus.COMPLETED.value, 0))

    return {
        "offered": len(statuses),
        "counts": counts,
        "completed_fraction": completed / len(statuses) if statuses else 0.0,
    }


def ingest_results(
    directory: Path,
    *,
    experiment: Experiment,
    freeze: Path,
    root: Path,
) -> IngestedResults:
    """Return an experiment's results once they conform to the freeze.

    Parameters
    ----------
    directory : Path
        The experiment directory, ``<root>/results/<n>/<experiment>``.
    experiment : Experiment
        The loaded experiment whose results the directory holds.
    freeze : Path
        The ``FREEZE.json`` of the freeze the experiment ran under.
    root : Path
        The repository root whose configuration groups the sweep composes with.

    Returns
    -------
    IngestedResults
        The rows, the measured and achieved values of every recorded point,
        the rows' residual summary and the preregistration audit.

    Raises
    ------
    PilotResultsError
        If the directory's status marks it a pilot's results.
    FreezeRecordError
        If the freeze record is unreadable, malformed, or records no tagged
        commit or sweep identifier for the experiment.
    FreezeMismatchError
        If a manifest was run at another source revision than the commit the
        freeze tags, the experiment runs against another freeze, its expanded sweep's
        identifier differs from the frozen one, or a row is reported against
        another freeze.
    ResultsLayoutError
        If the directory's path does not name the experiment under its freeze,
        or a file it must hold is absent or malformed, or it holds no row.
    UnfrozenConfigurationError
        If a manifest's name is not the point its configuration and seed
        identify, that point is not frozen, a row's configuration digest is
        not that of a frozen point, or the measured table records a point outside
        the frozen sweep.
    ApproximationError
        If a row's budget or residual is absent, malformed, negative, not
        finite, or disagrees with its measured flag.
    UndeclaredFamilyError
        If a comparison is corrected within a family the preregistration does
        not declare.
    """
    require_frozen_results(directory)
    expected = results_directory(root, experiment)

    if directory.resolve() != expected.resolve():
        msg = (
            f"a results directory for {experiment.name!r} under "
            f"{experiment.freeze_id!r} lies at {expected}, not {directory}"
        )
        raise ResultsLayoutError(msg)

    points = expand_experiment(experiment, root)
    freeze_id, _, commit = frozen_experiment(freeze, experiment, points)
    manifests = _check_manifests(
        directory / MANIFESTS_NAME,
        frozenset(point.point_id for point in points),
        experiment.config,
        experiment.seed_policy,
        commit,
    )
    completed_fraction = _completed_fraction(directory / STATUS_NAME)
    configurations = frozenset(config_digest(point.config) for point in points)
    rows = _rows(directory / ROWS_NAME)
    approximations: list[Approximation] = []

    for index, row in enumerate(rows):
        where = f"row {index} of {experiment.name!r}"

        if row.get("freeze_id") == UNFROZEN:
            msg = (
                f"{where} was scored outside any freeze, and its experiment ran "
                f"under {freeze_id!r}"
            )
            raise FreezeMismatchError(msg)

        if row.get("freeze_id") != freeze_id:
            msg = (
                f"{where} is reported against {row.get('freeze_id')!r} where its "
                f"experiment ran under {freeze_id!r}"
            )
            raise FreezeMismatchError(msg)

        if row.get("config_hash") not in configurations:
            msg = (
                f"{where} carries the configuration digest "
                f"{row.get('config_hash')!r}, which is not that of a point in the "
                f"frozen sweep"
            )
            raise UnfrozenConfigurationError(msg)

        approximation = _approximation(where, row)

        if approximation.applies and not experiment.config.enumerates:
            msg = (
                f"{where} claims an enumeration budget, and no enumeration "
                f"budget applies to a {experiment.config.kind} row"
            )
            raise ApproximationError(msg)

        approximations.append(approximation)

    measured = _measured_points(
        directory / MEASURED_NAME, frozenset(point.point_id for point in points)
    )
    derived = reduce_experiment_rows(
        rows,
        experiment.reducers,
        ReductionContext(
            experiment.document.parent,
            seed=experiment.seed_policy.root,
            points=comparison_points(manifests, measured),
            preregistration=experiment.preregistration,
            alpha=experiment.evaluation.alpha,
            correction=experiment.evaluation.correction,
            dependence=experiment.evaluation.dependence,
        ),
    )
    reported = derived.comparisons
    audit = check_preregistration(experiment.preregistration, reported)

    return IngestedResults(
        freeze_id,
        experiment.name,
        (*rows, *derived.rows),
        measured,
        residual_summary(approximations),
        audit,
        _decisions(reported, experiment.evaluation),
        completed_fraction,
        experiment.decision_point.fallback if experiment.falls_back else None,
    )


def _decisions(
    comparisons: Sequence[Comparison], evaluation: EvaluationPlan, /
) -> Mapping[str, Decisions]:
    """Return the declared correction applied within each family of comparisons.

    Parameters
    ----------
    comparisons : Sequence[Comparison]
        The comparisons the experiment reports, in report order.
    evaluation : EvaluationPlan
        The level, correction and dependence regime the experiment declares.

    Returns
    -------
    Mapping[str, Decisions]
        One set of decisions per family, in first-appearance order.
    """
    families: dict[str, list[PValue]] = {}

    for comparison in comparisons:
        families.setdefault(comparison.family, []).append(comparison.p_value)

    return MappingProxyType(
        {
            family: evaluation.correction.apply(
                tuple(members), evaluation.alpha, evaluation.dependence
            )
            for family, members in families.items()
        }
    )


class SeedBudgetRecord(NamedTuple):
    """The seed count a pilot's rows price for an experiment, beside its seed floor.

    Attributes
    ----------
    metric : str
        The row metric whose seed-to-seed spread the pilot measured.
    seed_floor : int
        The fewest seeds the experiment's contraction may reduce a cell to:
        its seed rung's floor, or every swept seed when it declares no seed
        rung.
    seeds_required : int
        The most seeds any pilot cell needs for the declared precision.
    cells : Mapping[str, int]
        The seeds each pilot cell needs, keyed by configuration digest and,
        where the rows name them, task, dataset and split, joined by ``/``.
    """

    metric: str
    seed_floor: int
    seeds_required: int
    cells: Mapping[str, int]


def price_seed_budget(experiment: Experiment, pilot: Path, /) -> SeedBudgetRecord:
    """Return the seed count a pilot's rows price under an experiment's declaration.

    The pilot is a results directory whose rows hold the budgeted metric for
    one or more cells, each cell one configuration (and, where rows name them,
    one task, dataset and split) scored at several seeds. Each cell's seeds
    are priced by :func:`lofi.eval.seeds_required` under the declared target,
    precision and level, never below the declared minimum seed count, and the
    experiment needs the most any cell needs.

    Parameters
    ----------
    experiment : Experiment
        The experiment whose evaluation section declares the seed budget.
    pilot : Path
        The pilot's results directory.

    Returns
    -------
    SeedBudgetRecord
        The required count beside the experiment's seed floor.

    Raises
    ------
    ResultsLayoutError
        If the pilot holds no readable rows, no row of the budgeted metric, or
        two rows of it for one cell at one seed.
    ValueError
        If a cell has fewer seeds than the declared minimum, or its spread
        cannot be bounded.
    """
    budget = experiment.evaluation.seed_budget
    scored: dict[str, dict[int, float]] = {}

    for row in _rows(pilot / ROWS_NAME):
        value = row.get("value")

        if row.get("metric") != budget.metric or value is None:
            continue

        cell = "/".join(
            str(row[column])
            for column in ("config_hash", "task", "dataset", "split")
            if row.get(column) is not None
        )
        seed = cast("int", row["seed"])
        seeds = scored.setdefault(cell, {})

        if seed in seeds:
            msg = (
                f"the pilot at {pilot} scores the metric {budget.metric!r} twice "
                f"for the cell {cell!r} at the seed {seed}"
            )
            raise ResultsLayoutError(msg)

        seeds[seed] = cast("float", value)

    if not scored:
        msg = f"the pilot at {pilot} holds no row of the metric {budget.metric!r}"
        raise ResultsLayoutError(msg)

    cells = {
        cell: seeds_required(
            tuple(values.values()),
            target=budget.target,
            precision=budget.precision,
            level=budget.level,
            minimum=experiment.evaluation.minimum_seeds,
        )
        for cell, values in sorted(scored.items())
    }

    rung = experiment.contraction.rung("seeds")

    return SeedBudgetRecord(
        budget.metric,
        rung.floor if isinstance(rung, SeedFloor) else len(experiment.sweep.seeds),
        max(cells.values()),
        MappingProxyType(cells),
    )


def write_seed_budget(experiment: Experiment, pilot: Path, /) -> Path:
    """Write the seed count a pilot prices beside the experiment document.

    Parameters
    ----------
    experiment : Experiment
        The experiment whose seed count is priced.
    pilot : Path
        The pilot's results directory.

    Returns
    -------
    Path
        The record written, ``<experiment>.seed-budget.json`` beside the
        experiment document, holding the seed floor and the required count.

    Raises
    ------
    ResultsLayoutError
        If the pilot holds no readable rows, or no row of the budgeted metric.
    ValueError
        If a cell has too few seeds to price, or its spread cannot be bounded.
    """
    priced = price_seed_budget(experiment, pilot)
    budget = experiment.evaluation.seed_budget
    record: dict[str, JsonValue] = {
        "experiment": experiment.name,
        "metric": priced.metric,
        "target": budget.target,
        "precision": budget.precision,
        "level": budget.level,
        "seed_floor": priced.seed_floor,
        "seeds_required": priced.seeds_required,
        "cells": dict(priced.cells),
    }
    path = experiment.document.with_name(f"{experiment.name}{SEED_BUDGET_SUFFIX}")
    replace_file(path, f"{json.dumps(record, indent=2)}\n".encode())

    return path


def write_ingested(directory: Path, ingested: IngestedResults, /) -> Path:
    """Write an experiment's accepted rows and their audit beside its results.

    The rows table holds every column any accepted row carries, null where a
    row does not. The audit records the freeze, the experiment, the residual
    summary of the rows, the preregistration audit of the comparisons, the
    declared correction's decisions within each family, the completed
    fraction, and the fallback claim the results are reported under, null
    while the declared claim stands. Both files are replaced whole.

    Parameters
    ----------
    directory : Path
        The experiment's results directory the rows were ingested from.
    ingested : IngestedResults
        The accepted results.

    Returns
    -------
    Path
        The audit written.
    """
    rows = ingested.rows
    columns = dict.fromkeys(column for row in rows for column in row)
    staged = directory / f".{INGESTED_NAME}.staged"
    pq.write_table(
        pa.Table.from_pylist(
            [{name: row.get(name) for name in columns} for row in rows]
        ),
        staged,
    )
    replace_path(staged, directory / INGESTED_NAME)
    residuals = ingested.residuals
    audit: dict[str, JsonValue] = {
        "freeze_id": ingested.freeze_id,
        "experiment": ingested.experiment,
        "rows": residuals.rows,
        "residuals": {
            "measured": residuals.measured,
            "unmeasured": residuals.unmeasured,
            "not_applicable": residuals.not_applicable,
            "by_source": {
                str(source): count for source, count in residuals.by_source.items()
            },
        },
        "preregistration": ingested.audit.to_spec(),
        "decisions": {
            family: {
                "correction": decisions.correction,
                "alpha": decisions.alpha,
                "dependence": decisions.dependence.value,
                "raw_p": [pvalue.value for pvalue in decisions.raw],
                "adjusted_p": list(decisions.adjusted),
                "reject": list(decisions.reject),
            }
            for family, decisions in ingested.decisions.items()
        },
        "completed_fraction": ingested.completed_fraction,
        "fallback": ingested.fallback,
    }
    path = directory / AUDIT_NAME
    replace_file(path, f"{json.dumps(audit, indent=2)}\n".encode())

    return path


def ingest_experiment(document: Path, freeze: Path, root: Path, /) -> Ingested:
    """Ingest an experiment's results directory and write its accepted tables.

    The experiment is read from its document, its results directory under the
    root is ingested against the freeze with no caller comparison, and the
    accepted rows and their audit are written beside the results.

    Parameters
    ----------
    document : Path
        The experiment document.
    freeze : Path
        The ``FREEZE.json`` of the freeze the experiment ran under.
    root : Path
        The repository root the experiment's results lie under.

    Returns
    -------
    Ingested
        The accepted row count, the rows table and the audit.
    """
    experiment = read_experiment(document)
    directory = results_directory(root, experiment)
    ingested = ingest_results(
        directory, experiment=experiment, freeze=freeze, root=root
    )
    audit = write_ingested(directory, ingested)

    return Ingested(len(ingested.rows), directory / INGESTED_NAME, audit)


class FreezeRecord(NamedTuple):
    """What a freeze record states about one experiment."""

    freeze_id: str
    sweep_id: str
    commit: str


def frozen_experiment(
    freeze: Path, experiment: Experiment, points: Sequence[SweepPoint], /
) -> FreezeRecord:
    """Return what a freeze records of an experiment whose expansion it froze.

    Parameters
    ----------
    freeze : Path
        The freeze's ``FREEZE.json``.
    experiment : Experiment
        The loaded experiment.
    points : Sequence[SweepPoint]
        The experiment's sweep as expanded now, in expansion order.

    Returns
    -------
    FreezeRecord
        The freeze identifier, the experiment's frozen sweep identifier and the
        commit the freeze tags.

    Raises
    ------
    FreezeRecordError
        If the record is unreadable or malformed, or does not list the
        experiment with a sweep identifier.
    FreezeMismatchError
        If the experiment runs against another freeze than the record's, or its
        expansion's sweep identifier is neither the frozen one nor reached from
        it by the experiment's recorded amendments.
    """
    record = _freeze_record(freeze, experiment.name)

    if experiment.freeze_id != record.freeze_id:
        msg = (
            f"the experiment {experiment.name!r} runs against "
            f"{experiment.freeze_id!r}, not the freeze {record.freeze_id!r} "
            "its record names"
        )
        raise FreezeMismatchError(msg)

    expanded = sweep_id(points)
    amendments = experiment.amendments
    amended = (
        bool(amendments)
        and amendments[0].previous_sweep_id == record.sweep_id
        and amendments[-1].sweep_id == expanded
    )

    if expanded != record.sweep_id and not amended:
        msg = (
            f"the experiment {experiment.name!r} expands to the sweep {expanded!r}, "
            f"not the {record.sweep_id!r} {record.freeze_id!r} froze"
        )
        raise FreezeMismatchError(msg)

    return record


def _freeze_record(path: Path, experiment: str) -> FreezeRecord:
    """Return a freeze's identifier, tagged commit and an experiment's frozen sweep.

    Parameters
    ----------
    path : Path
        The freeze's ``FREEZE.json``.
    experiment : str
        The experiment whose frozen sweep identifier is read.

    Returns
    -------
    FreezeRecord
        The freeze identifier, the experiment's frozen sweep identifier and the
        commit the freeze tags.

    Raises
    ------
    FreezeRecordError
        If the file is unreadable, is not a JSON object, or does not record a
        freeze identifier, a tagged commit and a sweep identifier for the
        experiment.
    """
    try:
        record: JsonValue = json.loads(path.read_bytes())

    except (OSError, json.JSONDecodeError) as refused:
        msg = f"the freeze record {path} cannot be read: {refused}"
        raise FreezeRecordError(msg) from refused

    if not isinstance(record, dict):
        msg = f"the freeze record {path} is not a JSON object"
        raise FreezeRecordError(msg)

    freeze_id = record.get("freeze_id")
    sweeps = record.get("sweep_id")
    frozen = sweeps.get(experiment) if isinstance(sweeps, dict) else None
    tag = record.get("git_tag")
    commit = tag.get("commit") if isinstance(tag, dict) else None

    if (
        not isinstance(freeze_id, str)
        or not isinstance(frozen, str)
        or not isinstance(commit, str)
    ):
        msg = (
            f"the freeze record {path} records no freeze identifier, tagged "
            f"commit and sweep identifier for the experiment {experiment!r}"
        )
        raise FreezeRecordError(msg)

    return FreezeRecord(freeze_id, frozen, commit)


def _check_manifests(
    manifests: Path,
    frozen: frozenset[str],
    config: ExperimentConfiguration,
    seed: ExperimentSeed,
    commit: str,
) -> tuple[ManifestPoint, ...]:
    """Refuse a manifest not named by the frozen point it records, or not run at it.

    Parameters
    ----------
    manifests : Path
        The directory holding one manifest per point.
    frozen : frozenset[str]
        The point identifiers of the frozen sweep.
    config : ExperimentConfiguration
        The configuration kind whose manifests the directory holds.
    seed : ExperimentSeed
        The experiment root every point's run seed must derive from.
    commit : str
        The commit the freeze tags, which every manifest's source revision
        must be.

    Raises
    ------
    ResultsLayoutError
        If the directory is absent or a manifest does not decode as a
        manifest of the experiment's configuration kind.
    UnfrozenConfigurationError
        If a manifest's name differs from the point identifier of the
        configuration and seed it records, or that point is not frozen.
    FreezeMismatchError
        If a manifest's run seed was derived from another experiment root, or it
        records another source revision than the tagged commit.
    """
    if not manifests.is_dir():
        msg = f"a results directory holds no {MANIFESTS_NAME!r} directory: {manifests}"
        raise ResultsLayoutError(msg)

    recorded_manifests: list[ManifestPoint] = []
    for path in sorted(manifests.glob("*.json")):
        try:
            manifest = config.manifest_point(path.read_bytes())

        except ValueError as refused:
            msg = f"the manifest {path.name} is refused: {refused}"
            raise ResultsLayoutError(msg) from refused

        derived = manifest.seed
        recorded = point_id(manifest.config.document, seed=derived.seed)

        if path.stem != recorded or derived.point_id != recorded:
            msg = (
                f"the manifest {path.name} records the point {recorded!r}, "
                f"not the point its name gives"
            )
            raise UnfrozenConfigurationError(msg)

        if recorded not in frozen:
            msg = f"the manifest {path.name} records a point outside the frozen sweep"
            raise UnfrozenConfigurationError(msg)

        if derived.experiment != seed:
            msg = (
                f"the manifest {path.name} was seeded under the experiment root "
                f"{derived.experiment.root}, not the experiment's root {seed.root}"
            )
            raise FreezeMismatchError(msg)

        if manifest.git_revision != commit:
            msg = (
                f"the manifest {path.name} was run at the source revision "
                f"{manifest.git_revision}, not the commit {commit} the freeze tags"
            )
            raise FreezeMismatchError(msg)

        recorded_manifests.append(manifest)

    return tuple(recorded_manifests)


def _completed_fraction(path: Path) -> float:
    """Return the completed fraction a status file records.

    Parameters
    ----------
    path : Path
        The launcher's status file.

    Returns
    -------
    float
        The fraction of the experiment's points that completed.

    Raises
    ------
    ResultsLayoutError
        If the file is absent, unreadable, not a JSON object, or records no
        completed fraction between zero and one.
    """
    try:
        status: JsonValue = json.loads(path.read_bytes())

    except (OSError, json.JSONDecodeError) as refused:
        msg = f"a results directory holds no readable {STATUS_NAME}: {refused}"
        raise ResultsLayoutError(msg) from refused

    if not isinstance(status, dict):
        msg = f"the {STATUS_NAME} of a results directory is not a JSON object"
        raise ResultsLayoutError(msg)

    fraction = status.get("completed_fraction")

    if (
        isinstance(fraction, bool)
        or not isinstance(fraction, int | float)
        or not 0.0 <= fraction <= 1.0
    ):
        msg = (
            f"the {STATUS_NAME} of a results directory records no completed "
            f"fraction: {fraction!r}"
        )
        raise ResultsLayoutError(msg)

    return float(fraction)


def _rows(path: Path) -> tuple[Mapping[str, JsonValue], ...]:
    """Return the serialized report rows a results file holds.

    Parameters
    ----------
    path : Path
        The rows file.

    Returns
    -------
    tuple[Mapping[str, JsonValue], ...]
        One mapping of report columns per row, in file order.

    Raises
    ------
    ResultsLayoutError
        If the file is absent, is not a parquet table, or holds no row.
    """
    try:
        decoded: list[dict[str, JsonValue]] = pq.read_table(path).to_pylist()

    except (OSError, pa.ArrowException) as refused:
        msg = f"a results directory holds no readable {ROWS_NAME}: {refused}"
        raise ResultsLayoutError(msg) from refused

    if not decoded:
        msg = f"the {ROWS_NAME} of a results directory holds no row"
        raise ResultsLayoutError(msg)

    return tuple(decoded)


def _measured_points(
    path: Path, frozen: frozenset[str], /
) -> tuple[Mapping[str, JsonValue], ...]:
    """Return the measured and achieved values of every recorded point.

    Parameters
    ----------
    path : Path
        The points file.
    frozen : frozenset[str]
        The point identifiers of the frozen sweep.

    Returns
    -------
    tuple[Mapping[str, JsonValue], ...]
        One mapping of columns per point, in file order.

    Raises
    ------
    ResultsLayoutError
        If the file is absent or is not a parquet table.
    UnfrozenConfigurationError
        If a row records a point outside the frozen sweep.
    """
    try:
        decoded: list[dict[str, JsonValue]] = pq.read_table(path).to_pylist()

    except (OSError, pa.ArrowException) as refused:
        msg = f"a results directory holds no readable {MEASURED_NAME}: {refused}"
        raise ResultsLayoutError(msg) from refused

    for row in decoded:
        if row.get("point_id") not in frozen:
            msg = (
                f"the {MEASURED_NAME} of a results directory records the point "
                f"{row.get('point_id')!r}, which is outside the frozen sweep"
            )
            raise UnfrozenConfigurationError(msg)

    return tuple(decoded)


def _approximation(where: str, row: Mapping[str, JsonValue]) -> Approximation:
    """Rebuild a row's approximation from its budget and residual columns.

    The budget kind selects an enumeration budget, rebuilt from its bounds, or
    the declaration that no enumeration budget applies. The columns are passed
    to the budget, residual and approximation value types unchecked, so those
    types refuse a missing bound, a wrong type, a negative or nonfinite
    residual, or a residual where no budget applies, and the rebuilt columns
    must equal the row's, so a measured flag that disagrees with the residuals,
    or a bound stated where no budget applies, is refused.

    Parameters
    ----------
    where : str
        The row's position, for the message.
    row : Mapping[str, JsonValue]
        The serialized report columns.

    Returns
    -------
    Approximation
        The row's budget and residual.

    Raises
    ------
    ApproximationError
        If a column is absent, the budget kind is not declared, a value is
        refused by its value type, or the rebuilt columns differ from the row's.
    """
    try:
        values = {source: row[f"residual_{source}"] for source in RESIDUAL_SOURCES}
        measured: dict[ResidualSource, float] = {
            source: cast("float", value)
            for source, value in values.items()
            if value is not None
        }

        match row["budget_kind"]:
            case "enumeration":
                budget: DeclaredBudget | NoEnumerationBudget = DeclaredBudget(
                    cast("int", row["budget_depth"]),
                    cast("int", row["budget_beam"]),
                    mass=cast("float | None", row["budget_mass"]),
                )

            case "none":
                budget = NO_ENUMERATION_BUDGET

            case kind:
                msg = f"{where} states the budget kind {kind!r}, not one declared"
                raise ApproximationError(msg)

        approximation = Approximation(
            budget, residual=Residual(measured) if measured else None
        )

    except KeyError as missing:
        msg = f"{where} has no column {missing}"
        raise ApproximationError(msg) from missing

    except (TypeError, ValueError) as refused:
        msg = f"{where} states an approximation that is refused: {refused}"
        raise ApproximationError(msg) from refused

    rebuilt = approximation.to_spec()
    stated = {column: row.get(column) for column in rebuilt}

    if stated["residual_measured"] is not approximation.is_measured or any(
        stated[column] != value for column, value in rebuilt.items()
    ):
        msg = (
            f"{where} states the approximation columns {stated!r}, which "
            f"rebuild as {rebuilt!r}; an unmeasured residual is serialized as "
            f"null, never as a number"
        )
        raise ApproximationError(msg)

    return approximation
