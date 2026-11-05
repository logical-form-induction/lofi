"""Experiments, loaded once from their declared documents.

An experiment's sweep is expanded over one configuration document of the kind
the experiment declares. Each kind is an installed plugin of the
``lofi.experiment_config`` entry-point group, whose
:class:`ExperimentConfiguration` composes its points, reads its point
manifests, executes its points and reports their rows, so expansion, launching,
freezing, pricing and ingestion take the same route for every kind.
"""

from __future__ import annotations

import json
from datetime import date
from importlib import import_module
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    cast,
    final,
    override,
)

from lofi._frozen import FrozenValue
from lofi.eval import read_preregistration
from lofi.run._boundary.experiment import (
    ExperimentCompute,
    ExperimentDocument,
    FailurePolicySpec,
    MetricPredicate,
    PointCountPredicate,
    ReportSpec,
    SignatureTwinSettings,
    decode_amendments,
    encode_amendments,
)
from lofi.run._boundary.sweep import _as_point
from lofi.run._commitments import (
    DecisionPoint,
    MetricBound,
    PointCount,
    Predicate,
    ReportedRow,
    StopRule,
)
from lofi.run._contraction import (
    ContractionError,
    ExpandedAxis,
    ExpandedSweep,
    Rung,
    point_key,
)
from lofi.run._correction import correction
from lofi.run._cost_model import cost_model
from lofi.run._experiment_config import experiment_config
from lofi.run._launch import sweep_id
from lofi.run._manifest import replace_file
from lofi.run._rung import RungContext, contraction_rung
from lofi.run._signature_twins import signature_twin_augmenter
from lofi.run._sweep import GridSweepExpander, leading_overrides
from lofi.spec import (
    ConfigDocumentError,
    ConfigError,
    MalformedOverrideError,
    SpecLoader,
    UndeclaredLeafError,
    config_search_path,
    load_plugins,
)
from lofi.spec.judgment_sources import decode_predicted_judgment_source

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from pathlib import Path
    from types import ModuleType

    from lofi._types import JsonValue
    from lofi.eval import Preregistration, SeedTarget
    from lofi.judgment import (
        PredictedJudgmentSource,
        TrainedPointPredictor,
    )
    from lofi.run._boundary.experiment import LaunchSpec, PredicateSpec
    from lofi.run._boundary.manifest import (
        ManifestStatus,
    )
    from lofi.run._boundary.sweep import Axis, AxisValue, Sweep, SweepPoint
    from lofi.run._cost_model import CostModel
    from lofi.run._environment import ExecutionEnvironment
    from lofi.run._ports import TrainingOperations
    from lofi.run._seed import ExperimentSeed, PointSeed
    from lofi.run._serial import ExecutePoint
    from lofi.run._store import ExperimentStore
    from lofi.spec import (
        LeafOrigin,
        Provenance,
    )
    from lofi.spec.reducers import ReducerSpec
    from lofi.stats import Correction, Dependence

__all__ = [
    "FIT_METRICS",
    "FIT_ROLE",
    "UNPRICED",
    "Amendment",
    "ComputeEstimate",
    "ContractionLadder",
    "EvaluationPlan",
    "ExecutionContext",
    "Experiment",
    "ExperimentConfiguration",
    "ExperimentError",
    "SeedBudgetPlan",
    "Unpriced",
    "amend_experiment",
    "amendment_ledger",
    "execute_predicted_judgments",
    "expand_experiment",
    "read_experiment",
]


def execute_predicted_judgments(
    document: Path, predictor: TrainedPointPredictor, /
) -> PredictedJudgmentSource:
    """Execute one predicted-source document through the judgment runtime.

    The document's item set is relative to the directory holding it.
    """
    declaration = decode_predicted_judgment_source(document.read_bytes())
    return cast(
        "PredictedJudgmentSource",
        _judgment().predicted_judgment_source_for(
            declaration.settings(), document.parent, predictor
        ),
    )


FIT_ROLE: Final = "fit"
"""The role a judgment fit's diagnostic rows are reported under."""

FIT_METRICS: Final = ("divergences", "max-rhat", "min-ess", "evidence-bound")
"""The diagnostics a judgment fit may report, one row each: the sampler's
divergences, largest R-hat and smallest effective sample size, or a variational
guide's evidence bound."""


class ExperimentError(ValueError):
    """An experiment document refused by its loader.

    Attributes
    ----------
    document : Path
        The experiment document that was refused.
    reason : str
        What the document gets wrong.
    """

    def __init__(self, document: Path, reason: str) -> None:
        self.document = document
        self.reason = reason
        super().__init__(f"the experiment {document} is refused: {reason}")


_REFUSALS: Final = (
    ConfigDocumentError,
    ConfigError,
    MalformedOverrideError,
    UndeclaredLeafError,
)
"""The errors composition refuses a configuration document with."""


class TracedComposer(Protocol):
    """Composes one point's configuration and keeps where each leaf came from."""

    def __call__(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, JsonValue]:
        """Return the resolved configuration a point's overrides compose to.

        Parameters
        ----------
        overrides : tuple[str, ...]
            The point's ``key=value`` overrides.
        seed : int
            The run seed interpolation derives named streams from.
        """
        ...

    def provenance(self, overrides: tuple[str, ...], seed: int, /) -> Provenance:
        """Return the sources of the point composed under the overrides.

        Parameters
        ----------
        overrides : tuple[str, ...]
            The overrides a point was composed under.
        seed : int
            The run seed the point was composed under.
        """
        ...

    def sources(
        self, overrides: tuple[str, ...], seed: int, /
    ) -> Mapping[str, tuple[LeafOrigin, ...]]:
        """Return every source that set each path of the point composed under them.

        Parameters
        ----------
        overrides : tuple[str, ...]
            The overrides a point was composed under.
        seed : int
            The run seed the point was composed under.
        """
        ...


class ManifestPoint(NamedTuple):
    """What a point manifest records of its point, lifecycle and source.

    A point that stopped with a fielded batch awaiting its responses records a
    killed lifecycle and ``awaiting_responses``, and is resumed when submitted
    again.
    """

    config: ResolvedConfiguration
    seed: PointSeed
    status: ManifestStatus
    git_revision: str
    signature_fingerprint: str | None
    awaiting_responses: bool = False


@final
class ResolvedConfiguration(FrozenValue):
    """One boundary-validated point configuration retained as canonical JSON."""

    _noun = "resolved point configuration"

    __slots__ = ("_document",)

    def __init__(self, document: str) -> None:
        object.__setattr__(self, "_document", document)

    @property
    def document(self) -> dict[str, JsonValue]:
        """Return a fresh copy of the validated configuration document."""
        return cast("dict[str, JsonValue]", json.loads(self._document))


class ExecutionContext(NamedTuple):
    """What the points of one experiment execute against in this process.

    Attributes
    ----------
    store : ExperimentStore
        The experiment's store, whose root seeds every point and whose freeze,
        if any, every row is reported against.
    environment : ExecutionEnvironment
        The verified execution environment every manifest records.
    failure : FailurePolicySpec
        The experiment's declared handling of each failure kind.
    training : TrainingOperations
        What prepares, fits and scores one planned run.
    """

    store: ExperimentStore
    environment: ExecutionEnvironment
    failure: FailurePolicySpec
    training: TrainingOperations


class ExperimentConfiguration(Protocol):
    """The configuration document an experiment's sweep is expanded over.

    A configuration kind is an installed plugin, so everything a launch,
    freeze, price or results directory does differently for one kind is a
    capability of its configuration.
    """

    @property
    def kind(self) -> str:
        """Return the tag the experiment declares the document's kind under."""
        ...

    @property
    def in_process(self) -> bool:
        """Return whether this kind's points execute only in the launching process."""
        ...

    @property
    def timed(self) -> bool:
        """Return whether this kind's points train a step a step timer prices."""
        ...

    @property
    def path(self) -> Path:
        """Return the document."""
        ...

    @property
    def manifest_name(self) -> str:
        """Return the file a point's directory records its manifest in."""
        ...

    @property
    def enumerates(self) -> bool:
        """Return whether this kind's rows may claim an enumeration budget."""
        ...

    def composer(self, search_path: tuple[Path, ...], /) -> TracedComposer:
        """Return the composition a sweep expands its points through.

        Parameters
        ----------
        search_path : tuple[Path, ...]
            The roots configuration groups are resolved against.
        """
        ...

    def evaluators(
        self,
        search_path: tuple[Path, ...],
        overrides: tuple[str, ...],
        seed: int,
        /,
    ) -> Mapping[str, frozenset[str]]:
        """Return the metrics each evaluator the document declares reports, by role.

        Parameters
        ----------
        search_path : tuple[Path, ...]
            The roots configuration groups are resolved against.
        overrides : tuple[str, ...]
            The overrides of the point whose evaluators are read.
        seed : int
            The run seed used while composing that point.
        """
        ...

    def roles(self, config: Mapping[str, JsonValue], /) -> frozenset[str]:
        """Return the evaluator roles one resolved point configuration declares.

        Parameters
        ----------
        config : Mapping[str, JsonValue]
            The point's resolved configuration.
        """
        ...

    def manifest_point(self, data: bytes, /) -> ManifestPoint:
        """Return what a point manifest of this kind records.

        Parameters
        ----------
        data : bytes
            The manifest's bytes.

        Raises
        ------
        ValueError
            If the bytes are not a manifest of this kind, or its seed
            derivation is malformed.
        """
        ...

    def executor(
        self, context: ExecutionContext, points: Sequence[SweepPoint], /
    ) -> ExecutePoint:
        """Return what executes this kind's points, having checked each of them.

        Parameters
        ----------
        context : ExecutionContext
            The store, environment, failure policy and training operations the
            points execute against.
        points : Sequence[SweepPoint]
            The points the executor will be given, each checked before any is
            executed.

        Raises
        ------
        PointExecutionError
            If a point cannot be executed; nothing is executed.
        """
        ...

    def identities(self, points: Sequence[SweepPoint], /) -> Mapping[str, JsonValue]:
        """Return the identities a freeze records for each point, by identifier.

        Parameters
        ----------
        points : Sequence[SweepPoint]
            The experiment's expanded points.

        Raises
        ------
        ValueError
            If a point cannot be planned for what it executes.
        """
        ...

    def result_rows(
        self,
        rows: Sequence[Mapping[str, JsonValue]],
        points: Sequence[SweepPoint],
        store: ExperimentStore,
        /,
    ) -> tuple[Mapping[str, JsonValue], ...]:
        """Return the rows a results directory reports from its points' rows.

        Parameters
        ----------
        rows : Sequence[Mapping[str, JsonValue]]
            The rows the completed points recorded, in expansion order.
        points : Sequence[SweepPoint]
            The experiment's expanded points.
        store : ExperimentStore
            The store the points execute under.
        """
        ...


def _judgment() -> ModuleType:
    """Return the judgment package at the call site that needs it.

    Raises
    ------
    RuntimeError
        If the judgment capability is not installed.
    """
    try:
        return import_module("lofi.judgment")

    except ModuleNotFoundError as error:
        msg = "a judgment-fit experiment requires the lofi[judgments] capability"
        raise RuntimeError(msg) from error


class ComputeEstimate(NamedTuple):
    """An experiment's accelerator-hour estimate and the cost model behind it."""

    p50_hours: float
    p90_hours: float
    cost_model: str


class Unpriced:
    """The state of an experiment whose document declares itself unpriced.

    An experiment is priced from its own declarations, so its document may load
    declaring ``compute: {kind: unpriced}`` before it declares the estimate pricing
    produces. A freeze admits only an experiment that declares one.
    """

    __slots__ = ()

    @override
    def __repr__(self) -> str:
        """Return the absent-estimate spelling."""
        return "UNPRICED"


UNPRICED: Final = Unpriced()
"""The compute of an experiment whose document declares itself unpriced."""


class ContractionLadder(NamedTuple):
    """The rungs an experiment may contract through, in ladder order."""

    rungs: tuple[Rung, ...]
    factorial: str

    def rung(self, step: str, /) -> Rung | None:
        """Return the declared rung at a ladder position, or None.

        Parameters
        ----------
        step : str
            The ladder position.
        """
        return next((rung for rung in self.rungs if rung.step == step), None)

    def position(self, step: str, /) -> int:
        """Return the declared position of a rung."""
        for position, rung in enumerate(self.rungs):
            if rung.step == step:
                return position
        msg = f"the contraction ladder declares no rung {step!r}"
        raise ValueError(msg)


class SeedBudgetPlan(NamedTuple):
    """How an experiment's seed count is priced from a pilot's rows."""

    #: The row metric whose seed-to-seed spread the pilot measures.
    metric: str

    #: The quantity whose precision the seed count controls.
    target: SeedTarget

    #: The relative half-width the target is estimated to.
    precision: float

    #: The confidence the half-width holds at.
    level: float


class EvaluationPlan(NamedTuple):
    """The statistical settings an experiment's results are read under."""

    #: The level each family of preregistered comparisons is controlled at.
    alpha: float

    #: The correction applied within each family.
    correction: Correction

    #: The dependence regime the tests of a family were run under.
    dependence: Dependence

    #: The fewest seeds a reported cell may rest on.
    minimum_seeds: int

    #: How the seed count is priced from a pilot.
    seed_budget: SeedBudgetPlan


class Amendment(NamedTuple):
    """One contraction rung applied to an experiment's sweep, as recorded."""

    rung: str
    dropped: tuple[AxisValue, ...]
    previous_sweep_id: str
    sweep_id: str
    points: int
    amended_on: date


def _expanded_axis(axis: Axis, /) -> ExpandedAxis:
    """Lower one validated boundary axis to immutable interior values."""
    return ExpandedAxis(axis.key, axis.points, selects_group=axis.selects_group)


def _lowered_sweep(sweep: Sweep, /) -> ExpandedSweep:
    """Lower a validated boundary sweep to immutable interior values."""
    return ExpandedSweep(
        design=sweep.design,
        grid=tuple(_expanded_axis(axis) for axis in sweep.grid),
        yoked=tuple(
            tuple(_expanded_axis(axis) for axis in group) for group in sweep.yoked
        ),
        clauses=sweep.clauses,
        seeds=tuple(sweep.seeds),
        expected_points=sweep.expected_points,
    )


@final
class Experiment(FrozenValue):
    """One experiment, as its document and amendment ledger declare it.

    Parameters
    ----------
    name : str
        The experiment's name, which is its document's stem.
    document : Path
        The experiment document.
    role : str
        The part of the empirical track the experiment serves.
    freeze_id : str
        The freeze the experiment runs against.
    config : ExperimentConfiguration
        The configuration document the sweep's points are composed over.
    sweep : ExpandedSweep
        The sweep in force: the declared sweep with every recorded amendment
        applied in order.
    seed_policy : ExperimentSeed
        The root seed every point's run seed is derived from.
    rows : tuple[ReportedRow, ...]
        The report rows the experiment produces.
    primary : ReportedRow
        The row the claim is read from.
    preregistration : Preregistration
        The plan the factors and comparisons are declared in.
    preregistration_document : Path
        The document the preregistration was read from.
    factors : Mapping[str, str]
        The key of the axis spanning each preregistered factor.
    compute : ComputeEstimate | Unpriced
        The accelerator-hour estimate, or ``UNPRICED`` while none is declared.
    pricing : CostModel
        The cost model the experiment's training steps are priced under.
    contraction : ContractionLadder
        The contraction ladder.
    stop_rules : tuple[StopRule, ...]
        The conditions under which the experiment halts early.
    decision_point : DecisionPoint
        The decision point and its fallback claim.
    evaluation : EvaluationPlan
        The statistical settings the experiment's results are read under.
    amendments : tuple[Amendment, ...]
        The rungs applied to the declared sweep, in application order.
    """

    _noun = "experiment"

    __slots__ = (
        "_amendments",
        "_compute",
        "_config",
        "_contraction",
        "_decision_point",
        "_document",
        "_evaluation",
        "_factors",
        "_failure",
        "_freeze_id",
        "_launch",
        "_name",
        "_preregistration",
        "_preregistration_document",
        "_pricing",
        "_primary",
        "_reducers",
        "_reports",
        "_role",
        "_rows",
        "_seed_policy",
        "_signature_twins",
        "_stop_rules",
        "_sweep",
    )

    def __init__(
        self,
        *,
        name: str,
        document: Path,
        role: str,
        freeze_id: str,
        config: ExperimentConfiguration,
        launch: LaunchSpec,
        failure: FailurePolicySpec,
        sweep: ExpandedSweep,
        seed_policy: ExperimentSeed,
        signature_twins: SignatureTwinSettings | None,
        rows: tuple[ReportedRow, ...],
        reports: tuple[ReportSpec, ...],
        reducers: tuple[ReducerSpec, ...],
        primary: ReportedRow,
        preregistration: Preregistration,
        preregistration_document: Path,
        factors: Mapping[str, str],
        compute: ComputeEstimate | Unpriced,
        pricing: CostModel,
        contraction: ContractionLadder,
        stop_rules: tuple[StopRule, ...],
        decision_point: DecisionPoint,
        evaluation: EvaluationPlan,
        amendments: tuple[Amendment, ...],
    ) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_freeze_id", freeze_id)
        object.__setattr__(self, "_config", config)
        object.__setattr__(self, "_launch", launch)
        object.__setattr__(self, "_failure", failure)
        object.__setattr__(self, "_sweep", sweep)
        object.__setattr__(self, "_seed_policy", seed_policy)
        object.__setattr__(self, "_signature_twins", signature_twins)
        object.__setattr__(self, "_rows", rows)
        object.__setattr__(self, "_reports", reports)
        object.__setattr__(self, "_reducers", reducers)
        object.__setattr__(self, "_primary", primary)
        object.__setattr__(self, "_preregistration", preregistration)
        object.__setattr__(self, "_preregistration_document", preregistration_document)
        object.__setattr__(self, "_factors", MappingProxyType(dict(factors)))
        object.__setattr__(self, "_compute", compute)
        object.__setattr__(self, "_pricing", pricing)
        object.__setattr__(self, "_contraction", contraction)
        object.__setattr__(self, "_stop_rules", stop_rules)
        object.__setattr__(self, "_decision_point", decision_point)
        object.__setattr__(self, "_evaluation", evaluation)
        object.__setattr__(self, "_amendments", amendments)

    @property
    def name(self) -> str:
        """Return the experiment's name."""
        return self._name

    @property
    def document(self) -> Path:
        """Return the experiment document."""
        return self._document

    @property
    def role(self) -> str:
        """Return the part of the empirical track the experiment serves."""
        return self._role

    @property
    def freeze_id(self) -> str:
        """Return the freeze the experiment runs against."""
        return self._freeze_id

    @property
    def config(self) -> ExperimentConfiguration:
        """Return the configuration document the sweep's points are composed over."""
        return self._config

    @property
    def launch(self) -> LaunchSpec:
        """Return the declared launcher and its settings."""
        return self._launch

    @property
    def failure(self) -> FailurePolicySpec:
        """Return the declared numerical and out-of-memory policies."""
        return self._failure

    @property
    def sweep(self) -> ExpandedSweep:
        """Return the sweep in force, with every recorded amendment applied."""
        return self._sweep

    @property
    def seed_policy(self) -> ExperimentSeed:
        """Return the root seed every point's run seed is derived from."""
        return self._seed_policy

    @property
    def signature_twins(self) -> SignatureTwinSettings | None:
        """Return the deterministic twin expansion, when one is declared."""
        return self._signature_twins

    @property
    def rows(self) -> tuple[ReportedRow, ...]:
        """Return the report rows the experiment produces, in declared order."""
        return self._rows

    @property
    def reports(self) -> tuple[ReportSpec, ...]:
        """Return tables rendered from stored rows, in declaration order."""
        return self._reports

    @property
    def reducers(self) -> tuple[ReducerSpec, ...]:
        """Return experiment-row reducers in declaration order."""
        return self._reducers

    @property
    def primary(self) -> ReportedRow:
        """Return the row the claim is read from."""
        return self._primary

    @property
    def preregistration(self) -> Preregistration:
        """Return the plan the factors and comparisons are declared in."""
        return self._preregistration

    @property
    def preregistration_document(self) -> Path:
        """Return the document the preregistration was read from."""
        return self._preregistration_document

    @property
    def factors(self) -> Mapping[str, str]:
        """Return the key of the axis spanning each preregistered factor."""
        return self._factors

    @property
    def compute(self) -> ComputeEstimate | Unpriced:
        """Return the accelerator-hour estimate, or ``UNPRICED`` without one."""
        return self._compute

    @property
    def pricing(self) -> CostModel:
        """Return the cost model the experiment's training steps are priced under."""
        return self._pricing

    @property
    def contraction(self) -> ContractionLadder:
        """Return the contraction ladder."""
        return self._contraction

    @property
    def stop_rules(self) -> tuple[StopRule, ...]:
        """Return the conditions under which the experiment halts early."""
        return self._stop_rules

    @property
    def decision_point(self) -> DecisionPoint:
        """Return the decision point and its fallback claim."""
        return self._decision_point

    @property
    def evaluation(self) -> EvaluationPlan:
        """Return the statistical settings the results are read under."""
        return self._evaluation

    @property
    def amendments(self) -> tuple[Amendment, ...]:
        """Return the rungs applied to the declared sweep, in application order."""
        return self._amendments

    @property
    def falls_back(self) -> bool:
        """Return whether the fallback rung has been applied."""
        return any(amendment.rung == "fallback" for amendment in self._amendments)

    def amended(self, sweep: ExpandedSweep, amendment: Amendment, /) -> Experiment:
        """Return this experiment with one more amendment and the sweep it produced.

        Parameters
        ----------
        sweep : ExpandedSweep
            The sweep the amendment produced.
        amendment : Amendment
            The amendment, applied after every recorded one.
        """
        return Experiment(
            name=self._name,
            document=self._document,
            role=self._role,
            freeze_id=self._freeze_id,
            config=self._config,
            launch=self._launch,
            failure=self._failure,
            sweep=sweep,
            seed_policy=self._seed_policy,
            signature_twins=self._signature_twins,
            rows=self._rows,
            reports=self._reports,
            reducers=self._reducers,
            primary=self._primary,
            preregistration=self._preregistration,
            preregistration_document=self._preregistration_document,
            factors=self._factors,
            compute=self._compute,
            pricing=self._pricing,
            contraction=self._contraction,
            stop_rules=self._stop_rules,
            decision_point=self._decision_point,
            evaluation=self._evaluation,
            amendments=(*self._amendments, amendment),
        )

    @override
    def __repr__(self) -> str:
        """Return the experiment's identity."""
        return f"Experiment(name={self._name!r}, freeze_id={self._freeze_id!r})"


def amendment_ledger(document: Path, /) -> Path:
    """Return the file an experiment's amendments are recorded in.

    The ledger lies beside the experiment document as
    ``<name>.amendments.json``.

    Parameters
    ----------
    document : Path
        The experiment document.
    """
    return document.with_name(f"{document.stem}.amendments.json")


def _predicate(declared: PredicateSpec, /) -> Predicate:
    """Lower one declared predicate to the condition it states.

    Raises
    ------
    ValueError
        If a point-count predicate counts a status no point takes.
    """
    match declared:
        case PointCountPredicate(statuses=statuses, at_least=at_least):
            return PointCount(statuses, at_least)

        case MetricPredicate() as metric:
            return MetricBound(
                ReportedRow(metric.row.evaluator, metric.row.metric),
                statistic=metric.statistic,
                direction=metric.direction,
                threshold=metric.threshold,
                minimum_rows=metric.minimum_rows,
            )

    msg = f"no condition is stated by a {type(declared).__name__}"
    raise TypeError(msg)


def _spanned(
    document: ExperimentDocument, preregistration: Preregistration, /
) -> dict[str, tuple[str, frozenset[str]]]:
    """Return each preregistered factor's axis and the values realizing its levels.

    Raises
    ------
    ValueError
        If a binding names a factor the preregistration does not declare, a
        factor is spanned by no axis of the sweep, a level is left unmapped, or
        a level maps to a value its axis does not take.
    """
    sweep = document.sweep
    axes = {
        axis.key: axis
        for axis in (*sweep.grid, *(axis for group in sweep.yoked for axis in group))
    }
    unknown = sorted(set(document.factors) - set(preregistration.grid))

    if unknown:
        msg = f"it binds factors the preregistration does not declare: {unknown}"
        raise ValueError(msg)

    spanned: dict[str, tuple[str, frozenset[str]]] = {}
    unspanned: list[str] = []

    for factor, levels in preregistration.grid.items():
        binding = document.factors.get(factor)
        key = factor if binding is None else binding.axis
        axis = axes.get(key)

        if axis is None:
            unspanned.append(factor)
            continue

        mapped = (
            {
                point_key(_as_point(level)): point_key(_as_point(level))
                for level in levels
            }
            if binding is None or not binding.levels
            else {
                point_key(_as_point(entry.level)): point_key(_as_point(entry.value))
                for entry in binding.levels
            }
        )
        unmapped = [
            point_key(_as_point(level))
            for level in levels
            if point_key(_as_point(level)) not in mapped
        ]

        if unmapped:
            msg = f"the factor {factor!r} maps no axis value to its levels {unmapped}"
            raise ValueError(msg)

        taken = {point_key(point) for point in axis.points}
        realized = frozenset(mapped[point_key(_as_point(level))] for level in levels)
        missing = sorted(realized - taken)

        if missing:
            msg = (
                f"the axis {key!r} spanning the factor {factor!r} does not take "
                f"the values {missing} its preregistered levels map to"
            )
            raise ValueError(msg)

        spanned[factor] = (key, realized)

    if unspanned:
        msg = (
            f"its sweep spans no axis for the preregistered factors {unspanned}; "
            f"bind each under factors to the axis that varies it"
        )
        raise ValueError(msg)

    return spanned


def _check_rows(
    path: Path,
    rows: tuple[ReportedRow, ...],
    reported: Mapping[str, frozenset[str]],
    /,
) -> None:
    """Refuse a row that names no metric of an evaluator the configuration declares.

    Raises
    ------
    ExperimentError
        If a row names an evaluator role the configuration document does not
        declare, or a metric that evaluator does not report.
    """
    for row in rows:
        metrics = reported.get(row.evaluator)

        if metrics is None:
            msg = (
                f"its row {row.label!r} names the evaluator {row.evaluator!r}, "
                f"which its configuration document does not declare among "
                f"{sorted(reported)}"
            )
            raise ExperimentError(path, msg)

        if row.metric not in metrics:
            msg = (
                f"its row {row.label!r} names the metric {row.metric!r}, which "
                f"the evaluator {row.evaluator!r} does not report among "
                f"{sorted(metrics)}"
            )
            raise ExperimentError(path, msg)


def _amendments(
    path: Path, ladder: ContractionLadder, sweep: ExpandedSweep, /
) -> tuple[ExpandedSweep, tuple[Amendment, ...]]:
    """Return the sweep in force and the amendments the ledger records.

    Each recorded rung is applied again to the sweep the previous one
    produced, and must remove exactly the values the ledger records.

    Raises
    ------
    ExperimentError
        If the ledger does not parse, records a rung the ladder does not
        declare, or disagrees with what applying its rungs removes.
    """
    ledger = amendment_ledger(path)

    if not ledger.is_file():
        return sweep, ()

    try:
        recorded = decode_amendments(ledger.read_bytes())

    except ValueError as refused:
        raise ExperimentError(path, str(refused)) from refused

    amendments: list[Amendment] = []
    previous_position = -1

    for record in recorded.amendments:
        rung = ladder.rung(record.rung)

        if rung is None:
            msg = (
                f"its ledger applies the rung {record.rung!r}, which its "
                f"ladder does not declare"
            )
            raise ExperimentError(path, msg)

        position = ladder.position(record.rung)
        if position < previous_position:
            msg = (
                f"its ledger applies the rung {record.rung!r} after a later "
                "declared rung"
            )
            raise ExperimentError(path, msg)
        previous_position = position

        try:
            contracted = rung.contract(sweep)
            amended_on = date.fromisoformat(record.date)

        except (ContractionError, ValueError) as refused:
            raise ExperimentError(path, str(refused)) from refused

        dropped = tuple(_as_point(value) for value in record.dropped)

        if [point_key(point) for point in contracted.dropped] != [
            point_key(point) for point in dropped
        ]:
            msg = (
                f"its ledger records the rung {record.rung!r} removing {dropped}, "
                f"and applying it removes {contracted.dropped}"
            )
            raise ExperimentError(path, msg)

        sweep = contracted.sweep.replaced(expected_points=record.points)
        amendments.append(
            Amendment(
                record.rung,
                dropped,
                record.previous_sweep_id,
                record.sweep_id,
                record.points,
                amended_on,
            )
        )

    return sweep, tuple(amendments)


def read_experiment(path: Path, /, *, overrides: tuple[str, ...] = ()) -> Experiment:
    """Load and validate an experiment document.

    The document lies at ``experiments/<name>.yaml`` in a repository whose
    configuration groups its configuration document composes with, and the
    experiment is named by the document's stem. Its
    configuration document and preregistration are read relative to the
    document's directory, the configuration being a run document or a
    judgment-fit document as the experiment declares. Every preregistered
    factor must be spanned by an axis of the sweep taking a value for each of
    its levels, and every row must name a metric of an evaluator the
    configuration document declares. A document must declare ``compute``,
    either as an estimate or as ``{kind: unpriced}``, which loads as
    ``UNPRICED``. The amendments its ledger records are applied to the
    declared sweep in order.

    Parameters
    ----------
    path : Path
        The experiment document.
    overrides : tuple[str, ...]
        ``key=value`` overrides applied over the document before it is
        validated, such as ``launch.kind=array``.

    Returns
    -------
    Experiment
        The validated experiment.

    Raises
    ------
    ExperimentError
        If the document does not parse, omits or malforms a field, disagrees
        with its preregistration or its configuration document, or its ledger
        disagrees with its ladder.
    """
    load_plugins()

    try:
        document = (
            SpecLoader()
            .compose_traced(path, schema=ExperimentDocument, overrides=overrides)
            .spec
        )

    except (ConfigError, ConfigDocumentError, MalformedOverrideError) as refused:
        raise ExperimentError(path, str(refused)) from refused

    declared = path.parent / document.config.path

    if not declared.is_file():
        msg = f"its {document.config.kind} document {declared} is not a file"
        raise ExperimentError(path, msg)

    try:
        config = cast(
            "ExperimentConfiguration",
            experiment_config(document.config, declared),
        )

    except (TypeError, ValueError) as refused:
        raise ExperimentError(path, str(refused)) from refused

    preregistration_document = path.parent / document.preregistration

    try:
        decided = date.fromisoformat(document.decision_point.date)
        text = preregistration_document.read_text(encoding="utf-8")
        preregistration = read_preregistration(text)
        spanned = _spanned(document, preregistration)
        stop_rules = tuple(
            StopRule(rule.name, _predicate(rule.when)) for rule in document.stop_rules
        )
        signal = _predicate(document.decision_point.signal)

    except (OSError, ValueError) as refused:
        raise ExperimentError(path, str(refused)) from refused

    rows = tuple(ReportedRow(row.evaluator, row.metric) for row in document.rows)

    sweep = _lowered_sweep(document.sweep)

    try:
        reported = config.evaluators(
            config_search_path(path.parents[1]),
            leading_overrides(sweep),
            sweep.seeds[0],
        )

    except _REFUSALS as refused:
        msg = f"its configuration {declared} is not a {config.kind} document: {refused}"
        raise ExperimentError(path, msg) from refused

    _check_rows(path, rows, reported)

    match document.compute:
        case ExperimentCompute() as estimate:
            compute: ComputeEstimate | Unpriced = ComputeEstimate(
                estimate.p50_hours, estimate.p90_hours, estimate.cost_model
            )

        case _:
            compute = UNPRICED

    rungs = RungContext(
        spanned, len(document.sweep.seeds), document.evaluation.minimum_seeds
    )

    try:
        contraction = ContractionLadder(
            tuple(contraction_rung(rung, rungs) for rung in document.contraction.rungs),
            document.contraction.factorial,
        )
        pricing = cost_model(document.pricing)

    except (TypeError, ValueError) as refused:
        raise ExperimentError(path, str(refused)) from refused

    sweep, amendments = _amendments(path, contraction, _lowered_sweep(document.sweep))
    decision = document.decision_point
    evaluation = document.evaluation
    budget = evaluation.seed_budget

    procedure = correction(evaluation.correction)

    return Experiment(
        name=path.stem,
        document=path,
        role=document.role,
        freeze_id=document.freeze_id,
        config=config,
        launch=document.launch,
        failure=document.failure,
        sweep=sweep,
        seed_policy=document.seed_policy.seed(),
        signature_twins=(
            None
            if document.signature_twins is None
            else document.signature_twins.settings()
        ),
        rows=rows,
        reports=document.reports,
        reducers=document.reducers,
        primary=ReportedRow(document.primary.evaluator, document.primary.metric),
        preregistration=preregistration,
        preregistration_document=preregistration_document,
        factors={factor: key for factor, (key, _) in spanned.items()},
        compute=compute,
        pricing=pricing,
        contraction=contraction,
        stop_rules=stop_rules,
        decision_point=DecisionPoint(decided, signal, decision.fallback),
        evaluation=EvaluationPlan(
            evaluation.alpha,
            procedure,
            evaluation.dependence,
            evaluation.minimum_seeds,
            SeedBudgetPlan(
                budget.metric, budget.target, budget.precision, budget.level
            ),
        ),
        amendments=amendments,
    )


def amend_experiment(
    experiment: Experiment, step: str, root: Path, /, *, on: date
) -> Experiment:
    """Apply one declared contraction rung and record it in the experiment's ledger.

    The rung is applied to the sweep in force. Every rung but the fallback
    must change the experiment's points, and the ledger records the values the
    rung removed, the sweep identifiers before and after and the new point
    count, so a later load applies the same rung to the same sweep.

    Parameters
    ----------
    experiment : Experiment
        The experiment as loaded with every recorded amendment.
    step : str
        The ladder position of the rung to apply.
    root : Path
        The repository root whose configuration groups the points compose with.
    on : date
        The day the amendment is made.

    Returns
    -------
    Experiment
        The experiment with the amendment applied.

    Raises
    ------
    ExperimentError
        If the ladder declares no such rung, the rung precedes one already
        applied, it cannot reduce the sweep in force, or it changes no point.
    """
    rung = experiment.contraction.rung(step)

    if rung is None:
        msg = f"its ladder declares no rung {step!r}"
        raise ExperimentError(experiment.document, msg)

    applied = experiment.amendments

    if applied and experiment.contraction.position(
        step
    ) < experiment.contraction.position(applied[-1].rung):
        msg = f"the rung {step!r} precedes the applied rung {applied[-1].rung!r}"
        raise ExperimentError(experiment.document, msg)

    previous = sweep_id(expand_experiment(experiment, root))

    try:
        contracted = rung.contract(experiment.sweep)

    except ContractionError as refused:
        raise ExperimentError(experiment.document, str(refused)) from refused

    points = _expand(experiment, contracted.sweep, root)
    produced = sweep_id(points)

    if produced == previous and step != "fallback":
        msg = f"the rung {step!r} changes no point of the sweep in force"
        raise ExperimentError(experiment.document, msg)

    amendment = Amendment(step, contracted.dropped, previous, produced, len(points), on)
    amended = experiment.amended(
        contracted.sweep.replaced(expected_points=len(points)), amendment
    )
    replace_file(
        amendment_ledger(experiment.document), encode_amendments(amended.amendments)
    )

    return amended


def _expand(
    experiment: Experiment, sweep: ExpandedSweep, root: Path, /
) -> tuple[SweepPoint, ...]:
    """Return the points of one of the experiment's sweeps, checked against its rows.

    Raises
    ------
    ExperimentError
        If a point does not compose as a document of the declared kind, or a
        point's configuration declares no evaluator under a row's role.
    """
    config = experiment.config
    compose_point = config.composer(config_search_path(root))
    expander = GridSweepExpander(
        compose_point,
        augment=(
            None
            if experiment.signature_twins is None
            else signature_twin_augmenter(
                experiment.signature_twins, experiment.seed_policy
            )
        ),
        provenance=compose_point.provenance,
        sources=compose_point.sources,
    )

    try:
        points = expander.expand(sweep)

    except _REFUSALS as refused:
        msg = (
            f"its configuration {config.path} is not a {config.kind} document: "
            f"{refused}"
        )
        raise ExperimentError(experiment.document, msg) from refused

    needed = {row.evaluator for row in experiment.rows}

    for point in points:
        missing = sorted(needed - config.roles(point.config))

        if missing:
            msg = (
                f"its point {point.point_id} declares no evaluator under the "
                f"roles {missing} its rows name"
            )
            raise ExperimentError(experiment.document, msg)

    return points


def expand_experiment(experiment: Experiment, root: Path, /) -> tuple[SweepPoint, ...]:
    """Return the points of an experiment's fully expanded sweep, in sweep order.

    Each point is composed over the experiment's configuration document with no
    profile, overlay or override beyond the sweep's own, so the expansion
    depends only on the experiment and the configuration groups under the root.
    Every point carries the provenance of its composition, so it can be
    executed. An amended experiment must expand to the sweep its last
    amendment recorded.

    Parameters
    ----------
    experiment : Experiment
        The experiment whose sweep is expanded.
    root : Path
        The repository root whose configuration groups the points compose with.

    Returns
    -------
    tuple[SweepPoint, ...]
        The unique points in deterministic order.

    Raises
    ------
    ExperimentError
        If a point does not compose as a document of the declared kind, a
        point declares no evaluator a row names, or the expansion is not the
        sweep the last amendment recorded.
    """
    points = _expand(experiment, experiment.sweep, root)

    if experiment.amendments:
        last = experiment.amendments[-1]
        expanded = sweep_id(points)

        if expanded != last.sweep_id:
            msg = (
                f"it expands to the sweep {expanded!r}, not the sweep "
                f"{last.sweep_id!r} its last amendment recorded"
            )
            raise ExperimentError(experiment.document, msg)

    return points
