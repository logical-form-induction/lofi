"""Registered reducers for experiment-level diagnostic report families."""

from itertools import combinations, product
from math import fsum, isfinite, sqrt
from statistics import fmean, stdev
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

from lofi.eval._comparison import Reduction
from lofi.registry import ComponentPlugin
from lofi.spec import derive_seed
from lofi.spec.reducers import (
    CoherenceCollapseReducerSettings,
    CoherenceCollapseReducerSpec,
    CrossedRecoveryReducerSettings,
    CrossedRecoveryReducerSpec,
    CrosslingualReducerSettings,
    CrosslingualReducerSpec,
    EncoderAblationReducerSettings,
    EncoderAblationReducerSpec,
    ResidualStructureReducerSettings,
    ResidualStructureReducerSpec,
    TransferReducerSettings,
    TransferReducerSpec,
)
from lofi.stats import paired_permutation

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.eval._reducers import ReductionContext

__all__ = [
    "COHERENCE_COLLAPSE_PLUGIN",
    "CROSSED_RECOVERY_PLUGIN",
    "CROSSLINGUAL_PLUGIN",
    "ENCODER_ABLATION_PLUGIN",
    "RESIDUAL_STRUCTURE_PLUGIN",
    "TRANSFER_PLUGIN",
]


def _number(row: Mapping[str, JsonValue], column: str, /) -> float:
    """Return one finite numeric cell from an experiment row."""
    value = row.get(column)
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"diagnostic row column {column!r} is not numeric: {value!r}"
        raise TypeError(msg)
    held = float(value)
    if not isfinite(held):
        msg = f"diagnostic row column {column!r} is not finite: {value!r}"
        raise ValueError(msg)
    return held


def _text(row: Mapping[str, JsonValue], column: str, /) -> str:
    """Return one nonempty text cell from an experiment row."""
    value = row.get(column)
    if not isinstance(value, str) or not value:
        msg = f"diagnostic row column {column!r} is not named text: {value!r}"
        raise TypeError(msg)
    return value


def _count(row: Mapping[str, JsonValue], column: str, /) -> int:
    """Return one nonnegative integer count cell."""
    value = row.get(column)
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        msg = f"diagnostic row column {column!r} is not a count: {value!r}"
        raise TypeError(msg)
    return value


def _factor(row: Mapping[str, JsonValue], column: str, /) -> Hashable:
    """Return one finite scalar factor level."""
    value = row.get(column)
    if isinstance(value, str | int | bool):
        return value
    if isinstance(value, float) and isfinite(value):
        return value
    msg = f"diagnostic row factor {column!r} is not scalar: {value!r}"
    raise TypeError(msg)


def _mean_rows(rows: Sequence[Mapping[str, JsonValue]], value_column: str, /) -> float:
    """Return the stable mean of one nonempty row collection."""
    if not rows:
        msg = "a diagnostic report cannot average an empty cell"
        raise ValueError(msg)
    values = tuple(_number(row, value_column) for row in rows)
    return fsum(values) / len(values)


@final
class _TransferReducer:
    """Reduce language-scored rows to scores, gaps, and class summaries."""

    __slots__ = ("_settings",)

    def __init__(self, settings: TransferReducerSettings) -> None:
        self._settings = settings

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return deterministic language and annotation-class transfer rows."""
        settings = self._settings
        languages = dict(settings.languages)
        emitted: list[Mapping[str, JsonValue]] = []
        for metric in settings.metrics:
            by_language = {
                language: tuple(
                    row
                    for row in rows
                    if row.get(settings.metric_column) == metric
                    and row.get(settings.language_column) == language
                )
                for language in languages
            }
            source_rows = by_language[settings.source_language]
            if not source_rows:
                msg = (
                    f"transfer metric {metric!r} has no source-language row for "
                    f"{settings.source_language!r}"
                )
                raise ValueError(msg)
            source_value = _mean_rows(source_rows, settings.value_column)
            values: dict[str, float | None] = {}
            for language, selected in by_language.items():
                if not selected and languages[language] == "gold":
                    msg = (
                        f"transfer metric {metric!r} has no row for gold language "
                        f"{language!r}"
                    )
                    raise ValueError(msg)
                value = (
                    None
                    if not selected
                    else _mean_rows(selected, settings.value_column)
                )
                values[language] = value
                count = (
                    None
                    if not selected
                    else sum(_count(row, settings.count_column) for row in selected)
                )
                common: dict[str, JsonValue] = {
                    "reducer": "transfer",
                    "metric": metric,
                    "language": language,
                    "source": languages[language],
                    "available": value is not None,
                    "reason": None if value is not None else "language-unavailable",
                    "n": count,
                }
                emitted.append({**common, "kind": "score", "value": value})
                emitted.append(
                    {
                        **common,
                        "kind": "gap",
                        "value": None if value is None else source_value - value,
                    }
                )

            for source_class in ("gold", "silver"):
                declared_targets = tuple(
                    language
                    for language in values
                    if language != settings.source_language
                    and languages[language] == source_class
                )
                if not declared_targets:
                    continue
                targets = tuple(values[language] for language in declared_targets)
                missing = tuple(
                    language
                    for language, value in zip(declared_targets, targets, strict=True)
                    if value is None
                )
                if missing:
                    common = {
                        "reducer": "transfer",
                        "metric": metric,
                        "language": f"{source_class}-targets",
                        "source": source_class,
                        "available": False,
                        "reason": "class-incomplete",
                        "value": None,
                        "n": None,
                    }
                    emitted.extend(
                        (
                            {**common, "kind": "score-summary"},
                            {**common, "kind": "gap-summary"},
                        )
                    )
                else:
                    available = tuple(cast("float", value) for value in targets)
                    emitted.extend(
                        (
                            {
                                "reducer": "transfer",
                                "metric": metric,
                                "language": f"{source_class}-targets",
                                "source": source_class,
                                "available": True,
                                "reason": None,
                                "kind": "score-summary",
                                "value": fmean(available),
                                "n": len(available),
                            },
                            {
                                "reducer": "transfer",
                                "metric": metric,
                                "language": f"{source_class}-targets",
                                "source": source_class,
                                "available": True,
                                "reason": None,
                                "kind": "gap-summary",
                                "value": fmean(
                                    source_value - value for value in available
                                ),
                                "n": len(available),
                            },
                        )
                    )
        return Reduction(tuple(emitted), ())


@final
class _CrosslingualReducer:
    """Reduce per-language rows to cross-lingual aggregate metrics."""

    __slots__ = ("_settings",)

    def __init__(self, settings: CrosslingualReducerSettings) -> None:
        self._settings = settings

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return macro, minimum, disparity, and source-gap rows per metric."""
        settings = self._settings
        emitted: list[Mapping[str, JsonValue]] = []
        for metric in settings.metrics:
            selected = tuple(
                row for row in rows if row.get(settings.metric_column) == metric
            )
            languages = sorted(
                {_text(row, settings.language_column) for row in selected}
            )
            values = {
                language: _mean_rows(
                    tuple(
                        row
                        for row in selected
                        if row.get(settings.language_column) == language
                    ),
                    settings.value_column,
                )
                for language in languages
            }
            if settings.source_language not in values:
                msg = (
                    f"cross-lingual metric {metric!r} has no source "
                    f"{settings.source_language!r}"
                )
                raise ValueError(msg)
            targets = tuple(
                value
                for language, value in values.items()
                if language != settings.source_language
            )
            if not targets:
                msg = f"cross-lingual metric {metric!r} has no target language"
                raise ValueError(msg)
            source = values[settings.source_language]
            summaries = (
                ("macro", fmean(values.values()), len(values)),
                ("minimum", min(values.values()), len(values)),
                ("disparity", max(values.values()) - min(values.values()), len(values)),
                (
                    "source-gap",
                    fmean(source - value for value in targets),
                    len(targets),
                ),
            )
            emitted.extend(
                {
                    "reducer": "cross-lingual",
                    "metric": metric,
                    "summary": summary,
                    "value": value,
                    "n": count,
                }
                for summary, value, count in summaries
            )
        return Reduction(tuple(emitted), ())


class _EncoderCell(NamedTuple):
    checkpoint: str
    freeze: str
    metric: str
    mean: float
    sd: float
    seeds: int
    weights: str


@final
class _EncoderAblationReducer:
    """Reduce seeded encoder cells to the declared ablation table."""

    __slots__ = ("_settings",)

    def __init__(self, settings: EncoderAblationReducerSettings) -> None:
        self._settings = settings

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return cell means and every within-freeze checkpoint contrast."""
        settings = self._settings
        cells: list[_EncoderCell] = []
        for checkpoint, freeze, metric in product(
            settings.checkpoints, settings.freeze_states, settings.metrics
        ):
            selected = tuple(
                row
                for row in rows
                if row.get(settings.checkpoint_column) == checkpoint
                and row.get(settings.freeze_column) == freeze
                and row.get(settings.metric_column) == metric
            )
            if not selected:
                msg = (
                    f"encoder ablation cell {(checkpoint, freeze, metric)!r} has no row"
                )
                raise ValueError(msg)
            seeds = tuple(_factor(row, settings.seed_column) for row in selected)
            if len(set(seeds)) != len(seeds):
                msg = (
                    f"encoder ablation cell {(checkpoint, freeze, metric)!r} "
                    "repeats a seed"
                )
                raise ValueError(msg)
            values = tuple(_number(row, settings.value_column) for row in selected)
            weights = tuple(_text(row, settings.weights_column) for row in selected)
            if any(weight not in {"pretrained", "stand-in"} for weight in weights):
                msg = (
                    f"encoder ablation cell {(checkpoint, freeze, metric)!r} "
                    f"has undeclared weights {weights!r}"
                )
                raise ValueError(msg)
            if len(set(weights)) != 1:
                msg = (
                    f"encoder ablation cell {(checkpoint, freeze, metric)!r} "
                    "changes its weight source"
                )
                raise ValueError(msg)
            if len(values) < settings.minimum_seeds:
                msg = (
                    f"encoder ablation cell {(checkpoint, freeze, metric)!r} has "
                    f"{len(values)} seeds, fewer than {settings.minimum_seeds}"
                )
                raise ValueError(msg)
            cells.append(
                _EncoderCell(
                    checkpoint,
                    freeze,
                    metric,
                    fmean(values),
                    0.0 if len(values) == 1 else stdev(values),
                    len(values),
                    weights[0],
                )
            )

        emitted: list[Mapping[str, JsonValue]] = [
            {
                "reducer": "encoder-ablation",
                "kind": "cell",
                "checkpoint": cell.checkpoint,
                "freeze_state": cell.freeze,
                "metric": cell.metric,
                "value": cell.mean,
                "seed_sd": cell.sd,
                "n": cell.seeds,
                "weights": cell.weights,
            }
            for cell in cells
        ]
        by_key = {(cell.checkpoint, cell.freeze, cell.metric): cell for cell in cells}
        for freeze, metric in product(settings.freeze_states, settings.metrics):
            for left, right in combinations(settings.checkpoints, 2):
                first = by_key[(left, freeze, metric)]
                second = by_key[(right, freeze, metric)]
                effect = first.mean - second.mean
                seed_sd = sqrt((first.sd**2 + second.sd**2) / 2.0)
                emitted.append(
                    {
                        "reducer": "encoder-ablation",
                        "kind": "difference",
                        "left": left,
                        "right": right,
                        "freeze_state": freeze,
                        "metric": metric,
                        "value": effect,
                        "effect": effect,
                        "seed_sd": seed_sd,
                        "standardized": (None if seed_sd == 0.0 else effect / seed_sd),
                        "n": min(first.seeds, second.seeds),
                        "left_weights": first.weights,
                        "right_weights": second.weights,
                        "involves_stand_in": "stand-in"
                        in (first.weights, second.weights),
                    }
                )
        return Reduction(tuple(emitted), ())


@final
class _CrossedRecoveryReducer:
    """Reduce fully crossed seeded recovery cells."""

    __slots__ = ("_seed", "_settings")

    def __init__(self, settings: CrossedRecoveryReducerSettings, *, seed: int) -> None:
        self._settings = settings
        self._seed = seed

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return a seed summary and floor-standardized recovery for each cell."""
        settings = self._settings
        levels = tuple(
            tuple(sorted({_factor(row, column) for row in rows}, key=repr))
            for column in settings.factor_columns
        )
        if any(not axis for axis in levels):
            msg = "crossed recovery has an empty factor axis"
            raise ValueError(msg)
        emitted: list[Mapping[str, JsonValue]] = []
        shared_seeds: frozenset[Hashable] | None = None
        for metric, cell in product(settings.metrics, product(*levels)):
            selected = tuple(
                row
                for row in rows
                if row.get(settings.metric_column) == metric
                and all(
                    _factor(row, column) == level
                    for column, level in zip(settings.factor_columns, cell, strict=True)
                )
            )
            if not selected:
                msg = f"crossed recovery cell {(metric, *cell)!r} has no row"
                raise ValueError(msg)
            seeds = tuple(_factor(row, settings.seed_column) for row in selected)
            seed_set = frozenset(seeds)
            if len(seed_set) != len(seeds):
                msg = f"crossed recovery cell {(metric, *cell)!r} repeats a seed"
                raise ValueError(msg)
            if len(seeds) < settings.minimum_seeds:
                msg = (
                    f"crossed recovery cell {(metric, *cell)!r} has {len(seeds)} "
                    f"seeds, fewer than {settings.minimum_seeds}"
                )
                raise ValueError(msg)
            if shared_seeds is None:
                shared_seeds = seed_set
            elif seed_set != shared_seeds:
                msg = f"crossed recovery cell {(metric, *cell)!r} uses another seed set"
                raise ValueError(msg)
            values = tuple(_number(row, settings.value_column) for row in selected)
            floors = tuple(_number(row, settings.floor_column) for row in selected)
            if len(set(floors)) != 1:
                msg = f"crossed recovery cell {(metric, *cell)!r} changes its floor"
                raise ValueError(msg)
            mean = fmean(values)
            sd = stdev(values)
            effect = mean - floors[0]
            identity = ":".join((metric, *(repr(level) for level in cell)))
            permutation = paired_permutation(
                values,
                floors,
                replicates=settings.null_replicates,
                seed=derive_seed(self._seed, f"crossed-recovery:{identity}"),
                tail="greater",
            )
            provenance = permutation.provenance
            emitted.append(
                {
                    "reducer": "crossed-recovery",
                    "metric": metric,
                    **{
                        column: cast("JsonValue", level)
                        for column, level in zip(
                            settings.factor_columns, cell, strict=True
                        )
                    },
                    "value": None if sd == 0.0 else effect / sd,
                    "observed": mean,
                    "floor": floors[0],
                    "effect": effect,
                    "seed_sd": sd,
                    "n": len(values),
                    "p_value": permutation.value,
                    "null_replicates": provenance.replicates,
                    "null_exact": provenance.exact,
                    "null_seed": provenance.seed,
                }
            )
        return Reduction(tuple(emitted), ())


@final
class _CoherenceCollapseReducer:
    """Reduce paired KL and cross-entropy collapse rows on the metric scale."""

    __slots__ = ("_settings",)

    def __init__(self, settings: CoherenceCollapseReducerSettings) -> None:
        self._settings = settings

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return both divergence cells and their seed-paired difference."""
        settings = self._settings
        levels = (settings.kl, settings.cross_entropy)
        by_level: dict[str, dict[Hashable, float]] = {}
        emitted: list[Mapping[str, JsonValue]] = []

        for level in levels:
            selected = tuple(
                row
                for row in rows
                if row.get(settings.metric_column) == settings.metric
                and row.get(settings.divergence_column) == level
            )
            seeds = tuple(_factor(row, settings.seed_column) for row in selected)
            if len(set(seeds)) != len(seeds):
                msg = f"coherence-collapse level {level!r} repeats a seed"
                raise ValueError(msg)
            if len(seeds) < settings.minimum_seeds:
                msg = (
                    f"coherence-collapse level {level!r} has {len(seeds)} seeds, "
                    f"fewer than {settings.minimum_seeds}"
                )
                raise ValueError(msg)
            values = tuple(_number(row, settings.value_column) for row in selected)
            by_level[level] = dict(zip(seeds, values, strict=True))
            emitted.append(
                {
                    "reducer": "coherence-collapse",
                    "kind": "cell",
                    "metric": settings.metric,
                    "divergence": level,
                    "value": fmean(values),
                    "seed_sd": stdev(values),
                    "n": len(values),
                }
            )

        left = by_level[settings.kl]
        right = by_level[settings.cross_entropy]
        if left.keys() != right.keys():
            msg = "coherence-collapse divergence levels use different seed sets"
            raise ValueError(msg)
        differences = tuple(left[seed] - right[seed] for seed in left)
        effect = fmean(differences)
        seed_sd = stdev(differences)
        emitted.append(
            {
                "reducer": "coherence-collapse",
                "kind": "difference",
                "metric": settings.metric,
                "left": settings.kl,
                "right": settings.cross_entropy,
                "value": effect,
                "effect": effect,
                "seed_sd": seed_sd,
                "standardized": None if seed_sd == 0.0 else effect / seed_sd,
                "n": len(differences),
            }
        )
        return Reduction(tuple(emitted), ())


def _linear_solve(
    matrix: list[list[float]], vector: list[float], /
) -> tuple[float, ...]:
    """Solve a nonsingular dense system by pivoted Gauss-Jordan elimination."""
    size = len(vector)
    augmented = [[*row, value] for row, value in zip(matrix, vector, strict=True)]
    for column in range(size):
        pivot = max(range(column, size), key=lambda row: abs(augmented[row][column]))
        if abs(augmented[pivot][column]) <= 1e-12:
            msg = "the residual-structure design is rank deficient"
            raise ValueError(msg)
        augmented[column], augmented[pivot] = augmented[pivot], augmented[column]
        scale = augmented[column][column]
        augmented[column] = [value / scale for value in augmented[column]]
        for row in range(size):
            if row == column:
                continue
            factor = augmented[row][column]
            augmented[row] = [
                value - factor * source
                for value, source in zip(augmented[row], augmented[column], strict=True)
            ]
    return tuple(augmented[row][-1] for row in range(size))


@final
class _ResidualStructureReducer:
    """Regress held-out residuals on the declared structural diagnostics."""

    __slots__ = ("_settings",)

    def __init__(self, settings: ResidualStructureReducerSettings) -> None:
        self._settings = settings

    def reduce(self, rows: Sequence[Mapping[str, JsonValue]], /) -> Reduction:
        """Return coefficients and explained variance for the structure test."""
        settings = self._settings
        categories = tuple(
            sorted({_text(row, settings.non_target_column) for row in rows})
        )
        if not categories:
            msg = "a residual-structure test has no held-out rows"
            raise ValueError(msg)
        terms = (
            "intercept",
            "root-entropy",
            "derivation-count",
            *(f"non-target:{category}" for category in categories[1:]),
        )
        design = tuple(
            (
                1.0,
                _number(row, settings.root_entropy_column),
                _number(row, settings.derivation_count_column),
                *(
                    float(_text(row, settings.non_target_column) == category)
                    for category in categories[1:]
                ),
            )
            for row in rows
        )
        response = tuple(_number(row, settings.response_column) for row in rows)
        if len(response) <= len(terms):
            msg = (
                f"a residual-structure test needs more than {len(terms)} rows, "
                f"not {len(response)}"
            )
            raise ValueError(msg)
        gram = [
            [
                fsum(row[left] * row[right] for row in design)
                for right in range(len(terms))
            ]
            for left in range(len(terms))
        ]
        target = [
            fsum(
                row[column] * value for row, value in zip(design, response, strict=True)
            )
            for column in range(len(terms))
        ]
        coefficients = _linear_solve(gram, target)
        fitted = tuple(
            fsum(
                weight * value for weight, value in zip(coefficients, row, strict=True)
            )
            for row in design
        )
        mean = fmean(response)
        rss = fsum(
            (observed - estimate) ** 2
            for observed, estimate in zip(response, fitted, strict=True)
        )
        tss = fsum((observed - mean) ** 2 for observed in response)
        r_squared = None if tss == 0.0 else 1.0 - rss / tss
        rows_out: list[Mapping[str, JsonValue]] = [
            {
                "reducer": "residual-structure",
                "metric": "coefficient",
                "term": term,
                "value": coefficient,
                "n": len(response),
                "baseline_non_target": categories[0],
            }
            for term, coefficient in zip(terms, coefficients, strict=True)
        ]
        rows_out.append(
            {
                "reducer": "residual-structure",
                "metric": "r-squared",
                "term": "model",
                "value": r_squared,
                "n": len(response),
                "rss": rss,
                "baseline_non_target": categories[0],
            }
        )
        return Reduction(tuple(rows_out), ())


def _transfer(
    settings: TransferReducerSettings, context: ReductionContext, /
) -> _TransferReducer:
    del context
    return _TransferReducer(settings)


def _crosslingual(
    settings: CrosslingualReducerSettings, context: ReductionContext, /
) -> _CrosslingualReducer:
    del context
    return _CrosslingualReducer(settings)


def _encoder_ablation(
    settings: EncoderAblationReducerSettings, context: ReductionContext, /
) -> _EncoderAblationReducer:
    del context
    return _EncoderAblationReducer(settings)


def _crossed_recovery(
    settings: CrossedRecoveryReducerSettings, context: ReductionContext, /
) -> _CrossedRecoveryReducer:
    return _CrossedRecoveryReducer(settings, seed=context.seed)


def _residual_structure(
    settings: ResidualStructureReducerSettings, context: ReductionContext, /
) -> _ResidualStructureReducer:
    del context
    return _ResidualStructureReducer(settings)


def _coherence_collapse(
    settings: CoherenceCollapseReducerSettings, context: ReductionContext, /
) -> _CoherenceCollapseReducer:
    del context
    return _CoherenceCollapseReducer(settings)


TRANSFER_PLUGIN: Final = ComponentPlugin(spec=TransferReducerSpec, build=_transfer)
CROSSLINGUAL_PLUGIN: Final = ComponentPlugin(
    spec=CrosslingualReducerSpec, build=_crosslingual
)
ENCODER_ABLATION_PLUGIN: Final = ComponentPlugin(
    spec=EncoderAblationReducerSpec, build=_encoder_ablation
)
CROSSED_RECOVERY_PLUGIN: Final = ComponentPlugin(
    spec=CrossedRecoveryReducerSpec, build=_crossed_recovery
)
RESIDUAL_STRUCTURE_PLUGIN: Final = ComponentPlugin(
    spec=ResidualStructureReducerSpec, build=_residual_structure
)
COHERENCE_COLLAPSE_PLUGIN: Final = ComponentPlugin(
    spec=CoherenceCollapseReducerSpec, build=_coherence_collapse
)
