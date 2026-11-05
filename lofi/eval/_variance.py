"""Score-variance decomposition across experimental factors."""

from itertools import product
from math import fsum
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._values import ConventionVector, Resampling
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.stats import Statistic

__all__ = [
    "VARIANCE_SOURCES",
    "Crossing",
    "FactorScore",
    "FactorVarianceComponent",
    "FactorVarianceDecomposition",
    "ReportedScore",
    "UnidentifiedDesignError",
    "VarianceComponent",
    "VarianceDecomposition",
    "VarianceSource",
    "factor_variance_decomposition",
    "variance_decomposition",
]

type FactorLevel = str | int | bool
"""A hashable scalar level read from one experiment-row column."""


type VarianceSource = Literal["convention", "seed", "interaction", "residual"]
"""Defines one of the four places the spread in a reported score comes from."""

VARIANCE_SOURCES: Final[tuple[VarianceSource, ...]] = (
    "convention",
    "seed",
    "interaction",
    "residual",
)
"""Variance sources in report order."""

_RESAMPLING_UNIT: Final = "convention-seed cell"
"""The resampled unit of a decomposition."""

_MINIMUM_LEVELS: Final = 2
"""The minimum factor levels for variance estimation."""

_SINGULAR: Final = 1e-9
"""The minimum pivot separating the moment equations."""


class UnidentifiedDesignError(ValueError):
    """Signals that a design does not separate the variances a decomposition names."""


class ReportedScore:
    """A published score with its method, convention, and seed.

    Parameters
    ----------
    method : str
        The method that produced the score.
    conventions : ConventionVector
        The scoring conventions.
    seed : int
        The run seed.
    score : float
        The reported value.

    Raises
    ------
    TypeError
        If an argument has an invalid type.
    ValueError
        If the method, seed, or score is invalid.
    """

    __slots__ = ("_conventions", "_method", "_score", "_seed")

    def __init__(
        self,
        *,
        method: str,
        conventions: ConventionVector,
        seed: int,
        score: float,
    ) -> None:
        if not isinstance(conventions, ConventionVector):
            msg = f"a reported score's conventions are not a vector: {conventions!r}"
            raise TypeError(msg)

        self._method = ensure_name("a reported score's method", method)
        self._conventions = conventions
        self._seed = ensure_non_negative("a reported score's seed", seed)
        self._score = ensure_finite("a reported score", score)

    @property
    def method(self) -> str:
        """Return the source method."""
        return self._method

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions the number was computed under."""
        return self._conventions

    @property
    def seed(self) -> int:
        """Return the seed the run producing the predictions was drawn under."""
        return self._seed

    @property
    def score(self) -> float:
        """Return the number itself."""
        return self._score

    @property
    def cell(self) -> tuple[ConventionVector, int]:
        """Return the cell of the crossing this number sits in."""
        return (self._conventions, self._seed)

    def _identity(self) -> tuple[str, ConventionVector, int, float]:
        """Return the four fields a reported score is equal on."""
        return (self._method, self._conventions, self._seed, self._score)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a score with the same four fields."""
        if type(other) is not ReportedScore:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields the score is equal on."""
        return hash((ReportedScore, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this score."""
        return (
            f"ReportedScore(method={self._method!r}, "
            f"conventions={self._conventions!r}, seed={self._seed!r}, "
            f"score={self._score!r})"
        )


class Crossing:
    """Stores decomposition design counts.

    Parameters
    ----------
    scores : int
        The number of reported scores.
    methods : int
        The number of methods.
    conventions : int
        The number of convention levels.
    seeds : int
        The number of seeds.
    cells : int
        The number of populated convention-seed cells.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or cells outnumber scores.
    """

    __slots__ = ("_cells", "_conventions", "_methods", "_scores", "_seeds")

    def __init__(
        self,
        *,
        scores: int,
        methods: int,
        conventions: int,
        seeds: int,
        cells: int,
    ) -> None:
        self._scores = ensure_non_negative("a crossing's score count", scores)
        self._methods = ensure_non_negative("a crossing's method count", methods)
        self._conventions = ensure_non_negative(
            "a crossing's convention count", conventions
        )
        self._seeds = ensure_non_negative("a crossing's seed count", seeds)
        self._cells = ensure_non_negative("a crossing's cell count", cells)

        if self._cells > self._scores:
            msg = (
                f"a crossing holds {cells} cells over {scores} scores, so a "
                f"cell holds no score"
            )
            raise ValueError(msg)

    @property
    def scores(self) -> int:
        """Return the score count."""
        return self._scores

    @property
    def methods(self) -> int:
        """Return the method count."""
        return self._methods

    @property
    def conventions(self) -> int:
        """Return how many distinct convention levels they were computed under."""
        return self._conventions

    @property
    def seeds(self) -> int:
        """Return how many distinct seeds."""
        return self._seeds

    @property
    def cells(self) -> int:
        """Return how many convention-seed cells hold at least one of them."""
        return self._cells

    @property
    def is_complete(self) -> bool:
        """Return whether every convention was scored under every seed."""
        return self._cells == self._conventions * self._seeds

    def _identity(self) -> tuple[int, int, int, int, int]:
        """Return the five counts a crossing is equal on."""
        return (
            self._scores,
            self._methods,
            self._conventions,
            self._seeds,
            self._cells,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a crossing with the same five counts."""
        if type(other) is not Crossing:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five counts."""
        return hash((Crossing, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this crossing."""
        return (
            f"Crossing(scores={self._scores!r}, methods={self._methods!r}, "
            f"conventions={self._conventions!r}, seeds={self._seeds!r}, "
            f"cells={self._cells!r})"
        )


class VarianceComponent:
    """A source's score variance.

    Parameters
    ----------
    source : VarianceSource
        The variance source.
    variance : float
        The nonnegative variance estimate.
    ci : Interval | None
        The optional resampled interval.

    Raises
    ------
    TypeError
        If the source, variance, or interval has an invalid type.
    ValueError
        If the variance is negative or nonfinite.
    """

    __slots__ = ("_ci", "_source", "_variance")

    def __init__(
        self, source: VarianceSource, variance: float, *, ci: Interval | None
    ) -> None:
        if source not in VARIANCE_SOURCES:
            msg = (
                f"a variance component names a source that is not a declared "
                f"one: {source!r}, which is not in {VARIANCE_SOURCES!r}"
            )
            raise TypeError(msg)

        if ci is not None and not isinstance(ci, Interval):
            msg = f"a variance component's interval is not one: {ci!r}"
            raise TypeError(msg)

        self._source: VarianceSource = source
        self._variance = ensure_finite(f"the variance of {source!r}", variance)
        self._ci = ci

        if self._variance < 0.0:
            msg = f"the variance of {source!r} is negative: {variance!r}"
            raise ValueError(msg)

    @property
    def source(self) -> VarianceSource:
        """Return the variance source."""
        return self._source

    @property
    def variance(self) -> float:
        """Return the variance estimate."""
        return self._variance

    @property
    def ci(self) -> Interval | None:
        """Return the resampled interval around it, if one was drawn."""
        return self._ci

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same source measured the same way."""
        if type(other) is not VarianceComponent:
            return NotImplemented

        return (
            self._source == other._source
            and self._variance == other._variance
            and self._ci == other._ci
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the source, the variance and the interval."""
        return hash((VarianceComponent, self._source, self._variance, self._ci))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this component."""
        return (
            f"VarianceComponent({self._source!r}, {self._variance!r}, ci={self._ci!r})"
        )


class VarianceDecomposition:
    """Stores score variance by source.

    Parameters
    ----------
    components : Iterable[VarianceComponent]
        One component for each variance source.
    crossing : Crossing
        The fitted design counts.
    resampling : Resampling
        The interval resampling scheme.

    Raises
    ------
    TypeError
        If a component, crossing, or resampling scheme has an invalid type.
    ValueError
        If a source is missing or repeated.
    """

    __slots__ = ("_components", "_crossing", "_resampling")

    def __init__(
        self,
        components: Iterable[VarianceComponent],
        *,
        crossing: Crossing,
        resampling: Resampling,
    ) -> None:
        if not isinstance(crossing, Crossing):
            msg = f"a decomposition's crossing is not one: {crossing!r}"
            raise TypeError(msg)

        if not isinstance(resampling, Resampling):
            msg = f"a decomposition's resampling scheme is not one: {resampling!r}"
            raise TypeError(msg)

        held: dict[VarianceSource, VarianceComponent] = {}

        for component in components:
            if not isinstance(component, VarianceComponent):
                msg = (
                    f"a decomposition holds something that is not a "
                    f"component: {component!r}"
                )
                raise TypeError(msg)

            if component.source in held:
                msg = f"a decomposition holds {component.source!r} twice"
                raise ValueError(msg)

            held[component.source] = component

        missing = tuple(source for source in VARIANCE_SOURCES if source not in held)

        if missing:
            msg = f"a decomposition names no variance for {missing!r}"
            raise ValueError(msg)

        self._components: Mapping[VarianceSource, VarianceComponent] = MappingProxyType(
            {source: held[source] for source in VARIANCE_SOURCES}
        )
        self._crossing = crossing
        self._resampling = resampling

    @property
    def components(self) -> tuple[VarianceComponent, ...]:
        """Return the four components, in the order a report lists them."""
        return tuple(self._components.values())

    @property
    def crossing(self) -> Crossing:
        """Return the decomposition design counts."""
        return self._crossing

    @property
    def resampling(self) -> Resampling:
        """Return the unit the intervals were drawn over, and the seed."""
        return self._resampling

    @property
    def total(self) -> float:
        """Return the variance of one reported score under the fitted model."""
        return fsum(component.variance for component in self._components.values())

    def component(self, source: VarianceSource) -> VarianceComponent:
        """Return one source's component.

        Parameters
        ----------
        source : VarianceSource
            The variance source.

        Returns
        -------
        VarianceComponent
            The source's estimate and interval.

        Raises
        ------
        TypeError
            If the source is not declared.
        """
        held = self._components.get(source)

        if held is None:
            msg = (
                f"a decomposition was asked for a source that is not a "
                f"declared one: {source!r}"
            )
            raise TypeError(msg)

        return held

    def share(self, source: VarianceSource) -> float | None:
        """Return one source's share of total variance.

        Parameters
        ----------
        source : VarianceSource
            The variance source.

        Returns
        -------
        float | None
            The share, or ``None`` when total variance is zero.

        Raises
        ------
        TypeError
            If the source is not declared.
        """
        total = self.total

        if total == 0.0:
            return None

        return self.component(source).variance / total

    @override
    def __repr__(self) -> str:
        """Return a summary naming each variance and the crossing behind them."""
        shares = ", ".join(
            f"{component.source}={component.variance!r}"
            for component in self._components.values()
        )

        return f"VarianceDecomposition({shares}, crossing={self._crossing!r})"


def _pooled[Key](
    pairs: Iterable[tuple[Key, float]],
) -> tuple[dict[Key, int], dict[Key, float]]:
    """Return counts and means by key.

    Parameters
    ----------
    pairs : Iterable[tuple[Key, float]]
        Keyed numeric values.

    Returns
    -------
    tuple[dict[Key, int], dict[Key, float]]
        The counts and means for each key.
    """
    counts: dict[Key, int] = {}
    totals: dict[Key, float] = {}

    for key, value in pairs:
        counts[key] = counts.get(key, 0) + 1
        totals[key] = totals.get(key, 0.0) + value

    return counts, {key: totals[key] / counts[key] for key in counts}


def _solved(
    rows: Sequence[Sequence[float]], targets: Sequence[float]
) -> tuple[float, ...]:
    """Return the solution of a square system by partial-pivot elimination.

    Parameters
    ----------
    rows : Sequence[Sequence[float]]
        The coefficient rows.
    targets : Sequence[float]
        The right-hand-side values.

    Returns
    -------
    tuple[float, ...]
        The solved column values.

    Raises
    ------
    UnidentifiedDesignError
        If the system is singular.
    """
    size = len(targets)
    table = [[*row, target] for row, target in zip(rows, targets, strict=True)]

    for column in range(size):
        pivot = column

        for candidate in range(column + 1, size):
            if abs(table[candidate][column]) > abs(table[pivot][column]):
                pivot = candidate

        if abs(table[pivot][column]) < _SINGULAR:
            msg = (
                f"the moment equations do not separate the components: "
                f"equation {column} leaves no pivot"
            )
            raise UnidentifiedDesignError(msg)

        table[column], table[pivot] = table[pivot], table[column]

        for index in range(size):
            if index == column:
                continue

            factor = table[index][column] / table[column][column]

            for entry in range(column, size + 1):
                table[index][entry] -= factor * table[column][entry]

    return tuple(table[index][size] / table[index][index] for index in range(size))


def _centered(
    scores: Sequence[ReportedScore],
) -> tuple[tuple[ReportedScore, float], ...]:
    """Return each score beside its value with its method's mean removed.

    Parameters
    ----------
    scores : Sequence[ReportedScore]
        The reported scores to center by method.

    Returns
    -------
    tuple[tuple[ReportedScore, float], ...]
        Each score paired with its method-centered value.
    """
    _, means = _pooled((score.method, score.score) for score in scores)

    return tuple((score, score.score - means[score.method]) for score in scores)


def _fit(scores: Sequence[ReportedScore]) -> Mapping[VarianceSource, float]:
    """Return variance estimates from the design's moment equations.

    Parameters
    ----------
    scores : Sequence[ReportedScore]
        The reported scores.

    Returns
    -------
    Mapping[VarianceSource, float]
        Nonnegative estimates keyed by variance source.

    Raises
    ------
    UnidentifiedDesignError
        If the design does not separate the sources.
    """
    centered = _centered(scores)
    total = len(centered)
    cell_counts, cell_means = _pooled((score.cell, value) for score, value in centered)
    convention_counts, convention_means = _pooled(
        (score.conventions, value) for score, value in centered
    )
    seed_counts, seed_means = _pooled((score.seed, value) for score, value in centered)
    methods = len({score.method for score in scores})
    grand = fsum(value for _, value in centered) / total

    within = fsum((value - cell_means[score.cell]) ** 2.0 for score, value in centered)
    between_conventions = fsum(
        count * (convention_means[key] - grand) ** 2.0
        for key, count in convention_counts.items()
    )
    between_seeds = fsum(
        count * (seed_means[key] - grand) ** 2.0 for key, count in seed_counts.items()
    )
    between_cells = fsum(
        count * (cell_means[key] - grand) ** 2.0 for key, count in cell_counts.items()
    )

    squared_by_convention: dict[ConventionVector, int] = {}
    squared_by_seed: dict[int, int] = {}

    for (conventions, seed), count in cell_counts.items():
        squared_by_convention[conventions] = (
            squared_by_convention.get(conventions, 0) + count * count
        )
        squared_by_seed[seed] = squared_by_seed.get(seed, 0) + count * count

    over_conventions = sum(count * count for count in convention_counts.values())
    over_seeds = sum(count * count for count in seed_counts.values())
    over_cells = sum(count * count for count in cell_counts.values())
    inside_conventions = fsum(
        squared_by_convention[key] / count for key, count in convention_counts.items()
    )
    inside_seeds = fsum(
        squared_by_seed[key] / count for key, count in seed_counts.items()
    )

    convention_row = [
        total - over_conventions / total,
        inside_conventions - over_seeds / total,
        inside_conventions - over_cells / total,
        float(len(convention_counts) - 1),
    ]
    seed_row = [
        inside_seeds - over_conventions / total,
        total - over_seeds / total,
        inside_seeds - over_cells / total,
        float(len(seed_counts) - 1),
    ]
    cell_row = [
        total - over_conventions / total,
        total - over_seeds / total,
        total - over_cells / total,
        float(len(cell_counts) - 1),
    ]
    interaction_row = [
        cell_row[index] - convention_row[index] - seed_row[index] for index in range(4)
    ]
    residual_row = [
        0.0,
        0.0,
        0.0,
        float(total - len(cell_counts) - methods + 1),
    ]
    estimated = _solved(
        [convention_row, seed_row, interaction_row, residual_row],
        [
            between_conventions,
            between_seeds,
            between_cells - between_conventions - between_seeds,
            within,
        ],
    )

    return {
        source: max(0.0, estimated[index])
        for index, source in enumerate(VARIANCE_SOURCES)
    }


def _statistic(source: VarianceSource) -> Statistic[Sequence[ReportedScore]]:
    """Return one source's variance statistic.

    Parameters
    ----------
    source : VarianceSource
        The variance source.

    Returns
    -------
    Statistic[Sequence[ReportedScore]]
        A statistic that refits the requested source.
    """

    def measured(sample: Sequence[ReportedScore]) -> float:
        return _fit(sample)[source]

    return measured


def _checked(scores: Iterable[ReportedScore]) -> tuple[ReportedScore, ...]:
    """Return validated reported scores.

    Parameters
    ----------
    scores : Iterable[ReportedScore]
        The reported scores.

    Returns
    -------
    tuple[ReportedScore, ...]
        The scores in input order.

    Raises
    ------
    TypeError
        If an entry is not a reported score.
    ValueError
        If a method, convention, and seed cell is repeated.
    """
    rows = tuple(scores)
    seen: set[tuple[str, ConventionVector, int]] = set()

    for row in rows:
        if not isinstance(row, ReportedScore):
            msg = f"a decomposition was handed {row!r}, which is not a reported score"
            raise TypeError(msg)

        key = (row.method, row.conventions, row.seed)

        if key in seen:
            msg = (
                f"the method {row.method!r} reports two numbers under one "
                f"convention and the seed {row.seed!r}"
            )
            raise ValueError(msg)

        seen.add(key)

    return rows


def _crossing(scores: Sequence[ReportedScore]) -> Crossing:
    """Return table design counts.

    Parameters
    ----------
    scores : Sequence[ReportedScore]
        The reported scores.
    """
    return Crossing(
        scores=len(scores),
        methods=len({score.method for score in scores}),
        conventions=len({score.conventions for score in scores}),
        seeds=len({score.seed for score in scores}),
        cells=len({score.cell for score in scores}),
    )


def variance_decomposition(
    scores: Iterable[ReportedScore],
    *,
    seed: int,
    replicates: int = 999,
    level: float = 0.95,
) -> VarianceDecomposition:
    """Decompose score variance by source.

    Parameters
    ----------
    scores : Iterable[ReportedScore]
        The reported scores in long form.
    seed : int
        The bootstrap seed.
    replicates : int
        The number of bootstrap replicates.
    level : float
        The interval coverage level.

    Returns
    -------
    VarianceDecomposition
        The fitted components and intervals.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If a cell is repeated or a bootstrap setting is invalid.
    UnidentifiedDesignError
        If the design does not identify every source.
    """
    rows = _checked(scores)
    crossing = _crossing(rows)

    if crossing.conventions < _MINIMUM_LEVELS or crossing.seeds < _MINIMUM_LEVELS:
        msg = (
            f"a decomposition over {crossing.conventions} conventions and "
            f"{crossing.seeds} seeds cannot separate the two, and needs at "
            f"least {_MINIMUM_LEVELS} of each"
        )
        raise UnidentifiedDesignError(msg)

    if crossing.scores - crossing.cells - crossing.methods + 1 <= 0:
        msg = (
            f"{crossing.scores} numbers over {crossing.cells} cells from "
            f"{crossing.methods} methods leave nothing inside a cell, so the "
            f"interaction and the residual are one quantity here"
        )
        raise UnidentifiedDesignError(msg)

    ordinals: dict[tuple[ConventionVector, int], str] = {}

    for row in rows:
        ordinals.setdefault(row.cell, f"cell-{len(ordinals)}")

    labels = [ordinals[row.cell] for row in rows]
    fitted = _fit(rows)
    components: list[VarianceComponent] = []

    for source in VARIANCE_SOURCES:
        bootstrap = Bootstrap(
            unit=_RESAMPLING_UNIT,
            replicates=replicates,
            seed=seed,
            level=level,
            groups=labels,
        )
        drawn = bootstrap.interval(rows, _statistic(source))
        components.append(
            VarianceComponent(
                source,
                fitted[source],
                ci=Interval(drawn.bca.low, drawn.bca.high, drawn.bca.level),
            )
        )

    return VarianceDecomposition(
        components,
        crossing=crossing,
        resampling=Resampling(_RESAMPLING_UNIT, seed),
    )


@final
class FactorScore(FrozenValue):
    """One method's score at a complete tuple of declared factor levels."""

    _noun = "factor score"
    __slots__ = ("_levels", "_method", "_score")

    def __init__(
        self,
        *,
        method: str,
        levels: Mapping[str, FactorLevel],
        score: float,
    ) -> None:
        held: dict[str, FactorLevel] = {}

        for name, level in levels.items():
            named = ensure_name("a variance factor", name)

            if isinstance(level, (bool, str)):
                checked: FactorLevel = level
            elif isinstance(level, int):
                checked = level
            else:
                msg = (
                    f"factor {named!r} has an unhashable or nonscalar level: {level!r}"
                )
                raise TypeError(msg)

            if isinstance(checked, str):
                checked = ensure_name(f"factor {named!r}'s level", checked)

            held[named] = checked

        if not held:
            msg = "a factor score declares no factor level"
            raise ValueError(msg)

        object.__setattr__(
            self, "_method", ensure_name("a factor score's method", method)
        )
        object.__setattr__(self, "_levels", MappingProxyType(held))
        object.__setattr__(self, "_score", ensure_finite("a factor score", score))

    @property
    def method(self) -> str:
        """Return the method that produced the score."""
        return self._method

    @property
    def levels(self) -> Mapping[str, FactorLevel]:
        """Return factor levels in declaration order."""
        return self._levels

    @property
    def score(self) -> float:
        """Return the reported score."""
        return self._score

    def cell(self, factors: Sequence[str], /) -> tuple[FactorLevel, ...]:
        """Return this score's levels in the requested factor order."""
        return tuple(self._levels[factor] for factor in factors)

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"FactorScore(method={self._method!r}, levels={dict(self._levels)!r}, "
            f"score={self._score!r})"
        )


@final
class FactorVarianceComponent(FrozenValue):
    """Variance assigned to one declared factor or pooled remainder."""

    _noun = "factor variance component"
    __slots__ = ("_ci", "_source", "_variance")

    def __init__(self, source: str, variance: float, *, ci: Interval) -> None:
        object.__setattr__(self, "_source", ensure_name("a variance source", source))
        held = ensure_finite(f"the variance of {source!r}", variance)

        if held < 0.0:
            msg = f"the variance of {source!r} is negative: {held!r}"
            raise ValueError(msg)

        if not isinstance(ci, Interval):
            msg = f"a factor variance interval is not one: {ci!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_variance", held)
        object.__setattr__(self, "_ci", ci)

    @property
    def source(self) -> str:
        """Return the factor or remainder name."""
        return self._source

    @property
    def variance(self) -> float:
        """Return the nonnegative variance estimate."""
        return self._variance

    @property
    def ci(self) -> Interval:
        """Return the method-resampled interval."""
        return self._ci


@final
class FactorVarianceDecomposition(FrozenValue):
    """A declared-factor method-of-moments decomposition."""

    _noun = "factor variance decomposition"
    __slots__ = ("_components", "_factors", "_methods", "_resampling", "_scores")

    def __init__(
        self,
        factors: Sequence[str],
        components: Sequence[FactorVarianceComponent],
        *,
        scores: int,
        methods: int,
        resampling: Resampling,
    ) -> None:
        object.__setattr__(self, "_factors", tuple(factors))
        object.__setattr__(self, "_components", tuple(components))
        object.__setattr__(self, "_scores", scores)
        object.__setattr__(self, "_methods", methods)
        object.__setattr__(self, "_resampling", resampling)

    @property
    def factors(self) -> tuple[str, ...]:
        """Return declared factors in model order."""
        return self._factors

    @property
    def components(self) -> tuple[FactorVarianceComponent, ...]:
        """Return factor, pooled-interaction, and residual components."""
        return self._components

    @property
    def scores(self) -> int:
        """Return the input score count."""
        return self._scores

    @property
    def methods(self) -> int:
        """Return the replicate-method count."""
        return self._methods

    @property
    def resampling(self) -> Resampling:
        """Return the interval resampling scheme."""
        return self._resampling

    def component(self, source: str, /) -> FactorVarianceComponent:
        """Return the named component."""
        for component in self._components:
            if component.source == source:
                return component

        msg = f"a decomposition declares no variance source {source!r}"
        raise KeyError(msg)

    @property
    def total(self) -> float:
        """Return total estimated variance."""
        return fsum(component.variance for component in self._components)


def _factor_fit(
    scores: Sequence[FactorScore], factors: tuple[str, ...]
) -> dict[str, float]:
    """Fit a complete balanced random-factor model by moments."""
    methods = tuple(sorted({row.method for row in scores}))
    levels = {
        factor: tuple(dict.fromkeys(row.levels[factor] for row in scores))
        for factor in factors
    }
    cells = tuple(product(*(levels[factor] for factor in factors)))
    by_key = {(row.method, row.cell(factors)): row.score for row in scores}
    expected = {(method, cell) for method in methods for cell in cells}

    if set(by_key) != expected or len(by_key) != len(scores):
        msg = (
            "a factor decomposition requires one score per method in every crossed cell"
        )
        raise UnidentifiedDesignError(msg)

    if any(len(levels[factor]) < _MINIMUM_LEVELS for factor in factors):
        msg = "every variance factor needs at least two observed levels"
        raise UnidentifiedDesignError(msg)

    if len(methods) < _MINIMUM_LEVELS:
        msg = "a factor decomposition needs at least two replicate methods"
        raise UnidentifiedDesignError(msg)

    method_means = {
        method: fsum(by_key[method, cell] for cell in cells) / len(cells)
        for method in methods
    }
    centered = {
        (method, cell): by_key[method, cell] - method_means[method]
        for method in methods
        for cell in cells
    }
    cell_means = {
        cell: fsum(centered[method, cell] for method in methods) / len(methods)
        for cell in cells
    }
    grand = fsum(cell_means.values()) / len(cells)
    between_cells = len(methods) * fsum(
        (mean - grand) ** 2.0 for mean in cell_means.values()
    )
    sums: dict[str, float] = {}
    factor_df = 0

    for position, factor in enumerate(factors):
        marginal = {
            level: fsum(cell_means[cell] for cell in cells if cell[position] == level)
            / (len(cells) / len(levels[factor]))
            for level in levels[factor]
        }
        multiplier = len(methods) * len(cells) / len(levels[factor])
        sums[factor] = multiplier * fsum(
            (mean - grand) ** 2.0 for mean in marginal.values()
        )
        factor_df += len(levels[factor]) - 1

    interaction_df = len(cells) - 1 - factor_df

    if interaction_df <= 0:
        msg = "the declared factors leave no interaction degree of freedom"
        raise UnidentifiedDesignError(msg)

    interaction_ss = between_cells - fsum(sums.values())
    within = fsum(
        (centered[method, cell] - cell_means[cell]) ** 2.0
        for method in methods
        for cell in cells
    )
    residual_df = (len(cells) - 1) * (len(methods) - 1)
    residual_ms = within / residual_df
    interaction_ms = interaction_ss / interaction_df
    fitted: dict[str, float] = {}

    for factor in factors:
        other_cells = len(cells) / len(levels[factor])
        mean_square = sums[factor] / (len(levels[factor]) - 1)
        fitted[factor] = max(
            0.0,
            (mean_square - interaction_ms) / (len(methods) * other_cells),
        )

    fitted["interaction"] = max(0.0, (interaction_ms - residual_ms) / len(methods))
    fitted["residual"] = max(0.0, residual_ms)
    return fitted


def _percentile(values: Sequence[float], probability: float) -> float:
    """Return a linearly interpolated empirical percentile."""
    ordered = sorted(values)
    position = probability * (len(ordered) - 1)
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return ordered[lower] * (1.0 - weight) + ordered[upper] * weight


def factor_variance_decomposition(
    scores: Iterable[FactorScore],
    factors: Sequence[str],
    *,
    estimator: Literal["moments"],
    seed: int,
    replicates: int,
    level: float,
) -> FactorVarianceDecomposition:
    """Decompose a balanced crossing over an arbitrary declared factor set."""
    rows = tuple(scores)
    named = tuple(ensure_name("a variance factor", factor) for factor in factors)

    if len(named) < 2 or len(set(named)) != len(named):
        msg = "a factor decomposition names at least two distinct factors"
        raise ValueError(msg)

    if estimator != "moments":
        msg = f"unknown variance estimator: {estimator!r}"
        raise ValueError(msg)

    if replicates < 1:
        msg = f"a variance interval draws at least one replicate, not {replicates}"
        raise ValueError(msg)

    if not 0.0 < level < 1.0:
        msg = f"a variance interval level is not a proportion: {level!r}"
        raise ValueError(msg)

    for row in rows:
        if not isinstance(row, FactorScore):
            msg = f"a factor decomposition was handed a non-score: {row!r}"
            raise TypeError(msg)

        if tuple(row.levels) != named:
            msg = (
                f"a factor score declares {tuple(row.levels)!r}, not the "
                f"decomposition's {named!r}"
            )
            raise ValueError(msg)

    fitted = _factor_fit(rows, named)
    methods = tuple(sorted({row.method for row in rows}))
    by_method = {
        method: tuple(row for row in rows if row.method == method) for method in methods
    }
    random = Random(ensure_non_negative("a variance bootstrap seed", seed))
    draws: dict[str, list[float]] = {
        source: [] for source in (*named, "interaction", "residual")
    }

    for _ in range(replicates):
        sampled: list[FactorScore] = []

        for index in range(len(methods)):
            chosen = methods[random.randrange(len(methods))]
            sampled.extend(
                FactorScore(
                    method=f"replicate-{index}", levels=row.levels, score=row.score
                )
                for row in by_method[chosen]
            )

        estimate = _factor_fit(sampled, named)

        for source, values in draws.items():
            values.append(estimate[source])

    tail = (1.0 - level) / 2.0
    components = tuple(
        FactorVarianceComponent(
            source,
            fitted[source],
            ci=Interval(
                _percentile(draws[source], tail),
                _percentile(draws[source], 1.0 - tail),
                level,
            ),
        )
        for source in draws
    )
    return FactorVarianceDecomposition(
        named,
        components,
        scores=len(rows),
        methods=len(methods),
        resampling=Resampling("method", seed),
    )
