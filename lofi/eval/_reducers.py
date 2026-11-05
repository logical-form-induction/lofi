"""Registered reducers over completed experiment rows."""

from math import isfinite
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, cast, final

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._churn import RankedScore, rank_churn
from lofi.eval._comparison import (
    PointMetadata,
    Reduction,
    SourceComparisonReducer,
)
from lofi.eval._prereg import Preregistration
from lofi.eval._rescoring import (
    ConventionProvider,
    PredictionReader,
    RescoringMetric,
    declared_conventions,
    read_declared_predictions,
    rescore_predictions,
    rescoring_metric,
)
from lofi.eval._variance import FactorLevel, FactorScore, factor_variance_decomposition
from lofi.registry import ComponentPlugin, Registry, RegistryError
from lofi.spec.reducers import (
    RankChurnReducerSettings,
    RankChurnReducerSpec,
    ReducerSpec,
    RescoringReducerSettings,
    RescoringReducerSpec,
    SourceComparisonReducerSettings,
    SourceComparisonReducerSpec,
    VarianceReducerSettings,
    VarianceReducerSpec,
)
from lofi.stats import Correction, Dependence

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue

__all__ = [
    "RANK_CHURN_PLUGIN",
    "RESCORING_PLUGIN",
    "SOURCE_COMPARISON_PLUGIN",
    "VARIANCE_PLUGIN",
    "ExperimentReducer",
    "ReductionContext",
    "reduce_experiment_rows",
]


class ExperimentReducer(Protocol):
    """Reduces a completed set of serialized experiment rows."""

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return derived rows in deterministic report order."""
        ...


@final
class ReductionContext(FrozenValue):
    """Runtime resources shared by every reducer in one experiment."""

    _noun = "reduction context"
    __slots__ = (
        "_alpha",
        "_conventions",
        "_correction",
        "_dependence",
        "_points",
        "_prediction_formats",
        "_preregistration",
        "_rescoring_metrics",
        "_root",
        "_seed",
    )

    def __init__(
        self,
        root: Path,
        *,
        seed: int,
        points: Mapping[str, Sequence[PointMetadata]] | None = None,
        preregistration: Preregistration | None = None,
        alpha: float | None = None,
        correction: Correction | None = None,
        dependence: Dependence | None = None,
        prediction_formats: Registry[PredictionReader] | None = None,
        rescoring_metrics: Registry[RescoringMetric] | None = None,
        conventions: Registry[ConventionProvider] | None = None,
    ) -> None:
        if not isinstance(root, Path):
            msg = f"an experiment reduction root is not a path: {root!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_root", root.resolve())
        object.__setattr__(
            self, "_seed", ensure_non_negative("an experiment reduction seed", seed)
        )
        held_points: dict[str, tuple[PointMetadata, ...]] = {}
        for config_hash, members in () if points is None else points.items():
            name = config_hash.strip()
            if not name:
                msg = "an experiment reduction point group has an empty config hash"
                raise ValueError(msg)
            grouped = tuple(members)
            if any(not isinstance(member, PointMetadata) for member in grouped):
                msg = f"an experiment reduction point group is malformed: {grouped!r}"
                raise TypeError(msg)
            result_seeds = tuple(member.result_seed for member in grouped)
            if len(set(result_seeds)) != len(result_seeds):
                msg = (
                    f"config hash {name!r} repeats a point result seed: "
                    f"{result_seeds!r}"
                )
                raise ValueError(msg)
            held_points[name] = grouped
        comparison_resources = (preregistration, alpha, correction, dependence)
        if any(resource is not None for resource in comparison_resources) and any(
            resource is None for resource in comparison_resources
        ):
            msg = "source-comparison resources must be supplied together"
            raise ValueError(msg)
        if preregistration is not None and not isinstance(
            preregistration, Preregistration
        ):
            msg = f"an experiment preregistration is not one: {preregistration!r}"
            raise TypeError(msg)
        if correction is not None and not isinstance(correction, Correction):
            msg = f"an experiment correction is not one: {correction!r}"
            raise TypeError(msg)
        if dependence is not None and not isinstance(dependence, Dependence):
            msg = f"an experiment dependence regime is not one: {dependence!r}"
            raise TypeError(msg)
        checked_alpha = None if alpha is None else float(alpha)
        if checked_alpha is not None and (
            not isfinite(checked_alpha) or not 0.0 < checked_alpha < 0.5
        ):
            msg = f"an experiment evaluation level is not below one half: {alpha!r}"
            raise ValueError(msg)
        object.__setattr__(self, "_points", MappingProxyType(held_points))
        object.__setattr__(self, "_preregistration", preregistration)
        object.__setattr__(self, "_alpha", checked_alpha)
        object.__setattr__(self, "_correction", correction)
        object.__setattr__(self, "_dependence", dependence)
        object.__setattr__(
            self,
            "_prediction_formats",
            Registry[PredictionReader]("prediction_format")
            if prediction_formats is None
            else prediction_formats,
        )
        object.__setattr__(
            self,
            "_rescoring_metrics",
            Registry[RescoringMetric]("rescoring_metric")
            if rescoring_metrics is None
            else rescoring_metrics,
        )
        object.__setattr__(
            self,
            "_conventions",
            Registry[ConventionProvider]("convention")
            if conventions is None
            else conventions,
        )

    @property
    def root(self) -> Path:
        """Return the experiment document's root directory."""
        return self._root

    @property
    def seed(self) -> int:
        """Return the deterministic reducer seed."""
        return self._seed

    @property
    def preregistration(self) -> Preregistration:
        """Return the experiment's declared analysis plan."""
        if self._preregistration is None:
            msg = "a source-comparison reducer has no preregistration"
            raise RuntimeError(msg)
        return self._preregistration

    @property
    def alpha(self) -> float:
        """Return the declared statistical level."""
        if self._alpha is None:
            msg = "a source-comparison reducer has no evaluation level"
            raise RuntimeError(msg)
        return self._alpha

    @property
    def correction(self) -> Correction:
        """Return the declared multiple-comparison correction."""
        if self._correction is None:
            msg = "a source-comparison reducer has no correction"
            raise RuntimeError(msg)
        return self._correction

    @property
    def dependence(self) -> Dependence:
        """Return the declared dependence regime."""
        if self._dependence is None:
            msg = "a source-comparison reducer has no dependence regime"
            raise RuntimeError(msg)
        return self._dependence

    @property
    def points(self) -> Mapping[str, tuple[PointMetadata, ...]]:
        """Return expanded point metadata grouped by configuration digest."""
        return self._points

    def point(self, config_hash: str, result_seed: int, /) -> PointMetadata:
        """Return the unique expanded point for a row's digest and result seed."""
        members = self._points.get(config_hash)
        if members is None:
            msg = f"an experiment row names unknown config hash {config_hash!r}"
            raise ValueError(msg)
        matched = tuple(
            member for member in members if member.result_seed == result_seed
        )
        if len(matched) != 1:
            msg = (
                f"config hash {config_hash!r} has {len(matched)} points at result "
                f"seed {result_seed}, not one"
            )
            raise ValueError(msg)
        return matched[0]

    @property
    def prediction_formats(self) -> Registry[PredictionReader]:
        """Return the prediction-format registry."""
        return self._prediction_formats

    @property
    def rescoring_metrics(self) -> Registry[RescoringMetric]:
        """Return the rescoring-metric registry."""
        return self._rescoring_metrics

    @property
    def conventions(self) -> Registry[ConventionProvider]:
        """Return the convention-provider registry."""
        return self._conventions

    def source(self, name: str, /) -> Path:
        """Resolve a declared prediction source below the experiment root."""
        relative = Path(name)

        if relative.is_absolute():
            msg = (
                f"a rescoring source must be relative to the experiment root: {name!r}"
            )
            raise ValueError(msg)

        resolved = (self._root / relative).resolve()

        if resolved != self._root and self._root not in resolved.parents:
            msg = f"a rescoring source escapes the experiment root: {name!r}"
            raise ValueError(msg)

        return resolved


def _number(row: Mapping[str, JsonValue], column: str, /) -> float:
    """Return one finite numeric row cell."""
    value = row.get(column)

    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"experiment row column {column!r} is not numeric: {value!r}"
        raise TypeError(msg)

    held = float(value)

    if not isfinite(held):
        msg = f"experiment row column {column!r} is not finite: {value!r}"
        raise ValueError(msg)

    return held


def _text(row: Mapping[str, JsonValue], column: str, /) -> str:
    """Return one named text row cell."""
    value = row.get(column)

    if not isinstance(value, str) or not value:
        msg = f"experiment row column {column!r} is not named text: {value!r}"
        raise TypeError(msg)

    return value


def _factor(row: Mapping[str, JsonValue], column: str, /) -> FactorLevel:
    """Return one scalar factor level."""
    value = row.get(column)

    if isinstance(value, str | int | bool):
        return value

    msg = f"experiment row factor column {column!r} is not scalar: {value!r}"
    raise TypeError(msg)


@final
class _VarianceReducer:
    """Method-of-moments reducer over declared row factors."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self, settings: VarianceReducerSettings, context: ReductionContext
    ) -> None:
        self._settings = settings
        self._context = context

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return one row per declared factor or pooled remainder."""
        settings = self._settings
        selected = tuple(row for row in rows if row.get("metric") == settings.metric)

        if not selected:
            msg = f"a variance reducer found no rows for metric {settings.metric!r}"
            raise ValueError(msg)

        factor_names = tuple(factor.name for factor in settings.factors)
        decomposition = factor_variance_decomposition(
            (
                FactorScore(
                    method=_text(row, settings.method_column),
                    levels={
                        factor.name: _factor(row, factor.column)
                        for factor in settings.factors
                    },
                    score=_number(row, settings.value_column),
                )
                for row in selected
            ),
            factor_names,
            estimator=settings.estimator,
            seed=self._context.seed,
            replicates=settings.interval.replicates,
            level=settings.interval.level,
        )
        return Reduction(
            tuple(
                {
                    "reducer": "variance",
                    "metric": settings.metric,
                    "source": component.source,
                    "value": component.variance,
                    "ci_lo": component.ci.low,
                    "ci_hi": component.ci.high,
                    "ci_level": component.ci.level,
                    "n": decomposition.scores,
                    "methods": decomposition.methods,
                    "resampling_unit": decomposition.resampling.unit,
                    "resampling_seed": decomposition.resampling.seed,
                }
                for component in decomposition.components
            ),
            (),
        )


@final
class _RankChurnReducer:
    """Paired midrank-displacement reducer."""

    __slots__ = ("_settings",)

    def __init__(self, settings: RankChurnReducerSettings) -> None:
        self._settings = settings

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return one rank-churn row for the declared metric and conditions."""
        settings = self._settings
        conditions = {settings.constrained, settings.unconstrained}
        selected = tuple(
            row
            for row in rows
            if row.get("metric") == settings.metric
            and row.get(settings.condition_column) in conditions
        )
        result = rank_churn(
            (
                RankedScore(
                    _text(row, settings.method_column),
                    _text(row, settings.condition_column),
                    _number(row, settings.value_column),
                )
                for row in selected
            ),
            left=settings.constrained,
            right=settings.unconstrained,
        )
        return Reduction(
            ({"reducer": "rank-churn", "metric": settings.metric, **result.to_spec()},),
            (),
        )


@final
class _RescoringReducer:
    """Declared sweep over one released-prediction artifact."""

    __slots__ = ("_context", "_settings")

    def __init__(
        self, settings: RescoringReducerSettings, context: ReductionContext
    ) -> None:
        self._settings = settings
        self._context = context

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return rescored cells for the declared grid and register."""
        del rows
        settings = self._settings
        path = self._context.source(settings.source)

        try:
            text = path.read_text(encoding="utf-8")

        except OSError as refused:
            msg = f"the rescoring source {path} cannot be read: {refused}"
            raise ValueError(msg) from refused

        predictions = read_declared_predictions(
            text,
            settings.prediction_format,
            registry=self._context.prediction_formats,
        )
        metric = rescoring_metric(
            settings.metric, registry=self._context.rescoring_metrics
        )
        conventions = declared_conventions(
            settings.conventions,
            settings.register,
            registry=self._context.conventions,
        )
        return Reduction(
            tuple(
                {"reducer": "rescoring", **cell.to_spec()}
                for cell in rescore_predictions(
                    predictions,
                    alignments=settings.alignments,
                    metric=metric,
                    conventions=conventions,
                )
            ),
            (),
        )


def _variance(
    settings: VarianceReducerSettings, context: ReductionContext, /
) -> ExperimentReducer:
    """Build a variance reducer."""
    return _VarianceReducer(settings, context)


def _rank_churn(
    settings: RankChurnReducerSettings, context: ReductionContext, /
) -> ExperimentReducer:
    """Build a rank-churn reducer."""
    del context
    return _RankChurnReducer(settings)


def _rescoring(
    settings: RescoringReducerSettings, context: ReductionContext, /
) -> ExperimentReducer:
    """Build a rescoring reducer."""
    return _RescoringReducer(settings, context)


def _source_comparison(
    settings: SourceComparisonReducerSettings, context: ReductionContext, /
) -> ExperimentReducer:
    """Build a declared source-comparison reducer."""
    return SourceComparisonReducer(settings, context)


VARIANCE_PLUGIN: Final = ComponentPlugin(spec=VarianceReducerSpec, build=_variance)
RESCORING_PLUGIN: Final = ComponentPlugin(spec=RescoringReducerSpec, build=_rescoring)
RANK_CHURN_PLUGIN: Final = ComponentPlugin(spec=RankChurnReducerSpec, build=_rank_churn)
SOURCE_COMPARISON_PLUGIN: Final = ComponentPlugin(
    spec=SourceComparisonReducerSpec,
    build=_source_comparison,
)


class _ReducerDeclaration(Protocol):
    """A validated reducer declaration consumed by its plugin."""

    @property
    def kind(self) -> str:
        """Return the registered reducer tag."""
        ...

    def settings(self) -> object:
        """Return the reducer's interior settings."""
        ...


type _ReducerPlugin = ComponentPlugin[object, ReductionContext, ExperimentReducer]


def reduce_experiment_rows(
    rows: Sequence[Mapping[str, JsonValue]],
    declarations: Sequence[ReducerSpec],
    context: ReductionContext,
    /,
    *,
    registry: Registry[_ReducerPlugin] | None = None,
) -> Reduction:
    """Run declared reducers over the same completed experiment rows."""
    plugins = Registry[_ReducerPlugin]("reducer") if registry is None else registry
    derived: list[Mapping[str, JsonValue]] = []
    comparisons = []

    for declaration in declarations:
        configured = cast("_ReducerDeclaration", declaration)
        member = plugins.get(configured.kind)

        if not isinstance(member, ComponentPlugin):
            msg = (
                f"the reducer entry {configured.kind!r} names {member!r}, "
                "not a component plugin"
            )
            raise RegistryError(msg, family="reducer")

        plugin = member

        if plugin.spec is not type(declaration):
            msg = (
                f"the reducer entry {configured.kind!r} builds "
                f"{plugin.spec.__name__}, not {type(declaration).__name__}"
            )
            raise RegistryError(msg, family="reducer")

        reducer = plugin.build(configured.settings(), context)
        reduction = reducer.reduce(rows)
        derived.extend(reduction.rows)
        comparisons.extend(reduction.comparisons)

    return Reduction(tuple(derived), tuple(comparisons))
