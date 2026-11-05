"""Evaluation values and convention vectors."""

from statistics import fmean, stdev
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._types import ItemId
from lofi.stats import Interval, PValue

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.spec.evaluators.conventions import ConventionSettings

__all__ = [
    "CONVENTION_LEVELS",
    "MINIMUM_SEEDS",
    "Aggregation",
    "Comparison",
    "ConventionVector",
    "CoveragePenalty",
    "GoldDerivation",
    "MetricResult",
    "PunctuationPolicy",
    "Resampling",
    "Sample",
    "SelfArcPolicy",
    "TrivialSpanPolicy",
    "UnaryChainPolicy",
    "convention_crossing",
    "seed_summary",
]


class Resampling:
    """A metric uncertainty scheme.

    Parameters
    ----------
    unit : str
        Resampling-unit name.
    seed : int | None
        Random seed, if sampling occurred.

    Raises
    ------
    TypeError
        If ``unit`` is not text or ``seed`` is not an integer.
    ValueError
        If ``unit`` is empty or ``seed`` is negative.
    """

    __slots__ = ("_seed", "_unit")

    def __init__(self, unit: str, seed: int | None = None) -> None:
        self._unit = ensure_name("a resampling unit", unit)
        self._seed = (
            None if seed is None else ensure_non_negative("a resampling seed", seed)
        )

    @property
    def unit(self) -> str:
        """Return the resampling unit."""
        return self._unit

    @property
    def seed(self) -> int | None:
        """Return the seed the draw ran under, if a draw ran."""
        return self._seed

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same unit and seed."""
        if type(other) is not Resampling:
            return NotImplemented

        return self._unit == other._unit and self._seed == other._seed

    @override
    def __hash__(self) -> int:
        """Return a hash over the unit and the seed."""
        return hash((Resampling, self._unit, self._seed))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this scheme."""
        return f"Resampling({self._unit!r}, {self._seed!r})"


class Sample:
    """An item's contribution to a metric.

    Parameters
    ----------
    item_id : ItemId
        Item identifier.
    score : float
        Item score.
    weight : float
        Nonnegative aggregation weight.
    stratum : str | None
        Resampling stratum, if any.

    Raises
    ------
    TypeError
        If a field has an invalid type.
    ValueError
        If a name is empty, a scalar is not finite, or ``weight`` is negative.
    """

    __slots__ = ("_item_id", "_score", "_stratum", "_weight")

    def __init__(
        self,
        item_id: ItemId,
        score: float,
        *,
        weight: float = 1.0,
        stratum: str | None = None,
    ) -> None:
        self._item_id: ItemId = ItemId(ensure_name("a sample's item id", item_id))
        self._score = ensure_finite("a sample's score", score)
        self._weight = ensure_finite("a sample's weight", weight)
        self._stratum = (
            None if stratum is None else ensure_name("a sample's stratum", stratum)
        )

        if self._weight < 0.0:
            msg = f"a sample's weight is negative: {weight!r}"
            raise ValueError(msg)

    @property
    def item_id(self) -> ItemId:
        """Return the opaque handle the item is joined on."""
        return self._item_id

    @property
    def score(self) -> float:
        """Return the item's score."""
        return self._score

    @property
    def weight(self) -> float:
        """Return how much the item counts for in the aggregate."""
        return self._weight

    @property
    def stratum(self) -> str | None:
        """Return the stratum the item was drawn from, if it was stratified."""
        return self._stratum

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a sample with the same four fields."""
        if type(other) is not Sample:
            return NotImplemented

        return (
            self._item_id == other._item_id
            and self._score == other._score
            and self._weight == other._weight
            and self._stratum == other._stratum
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the id, the score, the weight and the stratum."""
        return hash((Sample, self._item_id, self._score, self._weight, self._stratum))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this sample."""
        return (
            f"Sample({self._item_id!r}, {self._score!r}, "
            f"weight={self._weight!r}, stratum={self._stratum!r})"
        )


class MetricResult:
    """Stores a metric value, counts, and uncertainty.

    Parameters
    ----------
    value : float | None
        Metric value, or ``None`` when undefined.
    n : int
        Number of resampling units.
    resampling : Resampling
        Resampling scheme.
    components : Mapping[str, float] | None
        Named metric components.
    ci : Interval | None
        Metric interval, if computed.

    Raises
    ------
    TypeError
        If a field has an invalid type.
    ValueError
        If a scalar is invalid or the value, count, and interval are inconsistent.
    """

    __slots__ = ("_ci", "_components", "_n", "_resampling", "_value")

    def __init__(
        self,
        value: float | None,
        n: int,
        resampling: Resampling,
        *,
        components: Mapping[str, float] | None = None,
        ci: Interval | None = None,
    ) -> None:
        if not isinstance(resampling, Resampling):
            msg = f"a metric's resampling scheme is not one: {resampling!r}"
            raise TypeError(msg)

        if ci is not None and not isinstance(ci, Interval):
            msg = f"a metric's interval is not one: {ci!r}"
            raise TypeError(msg)

        self._value = (
            None if value is None else ensure_finite("a metric's value", value)
        )
        self._n = ensure_non_negative("a metric's sample count", n)
        self._resampling = resampling
        self._components = MappingProxyType(
            {
                ensure_name("a component name", key): ensure_finite(
                    f"component {key!r}", part
                )
                for key, part in (components or {}).items()
            }
        )
        self._ci = ci

        if self._n == 0 and self._value is not None:
            msg = f"a metric over no data reports a value: {value!r}"
            raise ValueError(msg)

        if self._value is None and self._ci is not None:
            msg = f"a metric with no value reports an interval: {ci!r}"
            raise ValueError(msg)

    @property
    def value(self) -> float | None:
        """Return the number the metric computed, if it is defined."""
        return self._value

    @property
    def n(self) -> int:
        """Return how many resampling units stand behind the value."""
        return self._n

    @property
    def resampling(self) -> Resampling:
        """Return the count unit and sampling seed."""
        return self._resampling

    @property
    def components(self) -> Mapping[str, float]:
        """Return the metric's own decomposition, as a read-only view."""
        return self._components

    @property
    def ci(self) -> Interval | None:
        """Return the interval around the value, if one was computed."""
        return self._ci

    def _identity(
        self,
    ) -> tuple[
        float | None, int, Resampling, tuple[tuple[str, float], ...], Interval | None
    ]:
        """Return the five fields a result is equal on, with components ordered."""
        return (
            self._value,
            self._n,
            self._resampling,
            tuple(sorted(self._components.items())),
            self._ci,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a result reporting the same five fields."""
        if type(other) is not MetricResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the result is equal on."""
        return hash((MetricResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"MetricResult({self._value!r}, {self._n!r}, {self._resampling!r}, "
            f"components={dict(self._components)!r}, ci={self._ci!r})"
        )


def copy_validated_metric_result(
    value: float | None,
    n: int,
    unit: str,
    *,
    ci: Interval | None,
) -> MetricResult:
    """Copy metric-result fields already validated at a boundary."""
    resampling = object.__new__(Resampling)
    object.__setattr__(resampling, "_unit", unit)
    object.__setattr__(resampling, "_seed", None)
    result = object.__new__(MetricResult)
    object.__setattr__(result, "_value", value)
    object.__setattr__(result, "_n", n)
    object.__setattr__(result, "_resampling", resampling)
    object.__setattr__(result, "_components", MappingProxyType({}))
    object.__setattr__(result, "_ci", ci)
    return result


class Comparison(FrozenValue):
    """Stores two metric arms and their falsifier.

    Parameters
    ----------
    left : str
        Left comparison arm.
    right : str
        Right comparison arm.
    statistic : float
        Observed comparison statistic.
    p_value : PValue
        Null-test result.
    effect : float
        Effect size.
    falsifier : str
        Outcome that would falsify the comparison.
    smoke_test : bool
        Whether the setup entails the result.
    family : str
        Multiple-comparison family.
    ci : Interval | None
        Effect interval, if computed.

    Raises
    ------
    TypeError
        If a field has an invalid type.
    ValueError
        If metadata is empty, a scalar is not finite, or both arms are equal.
    """

    _noun = "comparison"

    __slots__ = (
        "_ci",
        "_effect",
        "_falsifier",
        "_family",
        "_left",
        "_p_value",
        "_right",
        "_smoke_test",
        "_statistic",
    )

    def __init__(
        self,
        left: str,
        right: str,
        *,
        statistic: float,
        p_value: PValue,
        effect: float,
        falsifier: str,
        smoke_test: bool,
        family: str,
        ci: Interval | None = None,
    ) -> None:
        if not isinstance(p_value, PValue):
            msg = (
                f"a comparison's p-value is not one: {p_value!r}. only a null "
                f"test produces one"
            )
            raise TypeError(msg)

        if ci is not None and not isinstance(ci, Interval):
            msg = f"a comparison's interval is not one: {ci!r}"
            raise TypeError(msg)

        if not isinstance(smoke_test, bool):
            msg = f"a comparison's smoke-test flag is not a bool: {smoke_test!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_left", ensure_name("a comparison's left arm", left))
        object.__setattr__(
            self, "_right", ensure_name("a comparison's right arm", right)
        )
        object.__setattr__(
            self,
            "_statistic",
            ensure_finite("a comparison's statistic", statistic),
        )
        object.__setattr__(self, "_p_value", p_value)
        object.__setattr__(
            self, "_effect", ensure_finite("a comparison's effect", effect)
        )
        object.__setattr__(
            self,
            "_falsifier",
            ensure_name("a comparison's falsifier", falsifier),
        )
        object.__setattr__(self, "_smoke_test", smoke_test)
        object.__setattr__(
            self, "_family", ensure_name("a comparison's family", family)
        )
        object.__setattr__(self, "_ci", ci)

        if self._left == self._right:
            msg = f"a comparison names one arm on both sides: {left!r}"
            raise ValueError(msg)

    @property
    def left(self) -> str:
        """Return the arm the statistic is signed from."""
        return self._left

    @property
    def right(self) -> str:
        """Return the arm it is signed against."""
        return self._right

    @property
    def statistic(self) -> float:
        """Return the observed statistic separating the two arms."""
        return self._statistic

    @property
    def p_value(self) -> PValue:
        """Return the null probability of the observed statistic."""
        return self._p_value

    @property
    def effect(self) -> float:
        """Return the effect size, on the scale the protocol declares."""
        return self._effect

    @property
    def ci(self) -> Interval | None:
        """Return the interval around the effect, if one was computed."""
        return self._ci

    @property
    def falsifier(self) -> str:
        """Return the outcome that would falsify this comparison protocol."""
        return self._falsifier

    @property
    def smoke_test(self) -> bool:
        """Return whether the result is entailed by the setup."""
        return self._smoke_test

    @property
    def family(self) -> str:
        """Return the declared family of comparisons this one was corrected within."""
        return self._family

    def _identity(
        self,
    ) -> tuple[str, str, float, PValue, float, Interval | None, str, bool, str]:
        """Return the nine fields a comparison is equal on."""
        return (
            self._left,
            self._right,
            self._statistic,
            self._p_value,
            self._effect,
            self._ci,
            self._falsifier,
            self._smoke_test,
            self._family,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a comparison with the same nine fields."""
        if type(other) is not Comparison:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the nine fields the comparison is equal on."""
        return hash((Comparison, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this comparison."""
        return (
            f"Comparison({self._left!r}, {self._right!r}, "
            f"statistic={self._statistic!r}, p_value={self._p_value!r}, "
            f"effect={self._effect!r}, falsifier={self._falsifier!r}, "
            f"smoke_test={self._smoke_test!r}, family={self._family!r}, "
            f"ci={self._ci!r})"
        )


MINIMUM_SEEDS: Final[int] = 4
"""The minimum seed count for a summary."""


def seed_summary(results: Sequence[MetricResult]) -> MetricResult:
    """Return a metric summary across seeds.

    Parameters
    ----------
    results : Sequence[MetricResult]
        Results from distinct seeds.

    Returns
    -------
    MetricResult
        Mean value with extrema and sample deviation as components.

    Raises
    ------
    TypeError
        If an entry is not a metric result.
    ValueError
        If results are insufficient, incompatible, unseeded, undefined, or repeated.
    """
    for entry in results:
        if not isinstance(entry, MetricResult):
            msg = f"a seed summary was handed {entry!r}, which is not a result"
            raise TypeError(msg)

    if len(results) < MINIMUM_SEEDS:
        msg = (
            f"a seed summary needs at least {MINIMUM_SEEDS} seeds and was "
            f"handed {len(results)}"
        )
        raise ValueError(msg)

    units = {entry.resampling.unit for entry in results}

    if len(units) > 1:
        msg = f"a seed summary pools results counted in different units: {units!r}"
        raise ValueError(msg)

    scored: dict[int, float] = {}

    for entry in results:
        seed = entry.resampling.seed

        if seed is None:
            msg = f"a seed summary was handed a result with no seed: {entry!r}"
            raise ValueError(msg)

        if entry.value is None:
            msg = f"a seed summary was handed a result with no value: {entry!r}"
            raise ValueError(msg)

        if seed in scored:
            msg = f"a seed summary was handed the seed {seed!r} twice"
            raise ValueError(msg)

        scored[seed] = entry.value

    values = [scored[seed] for seed in sorted(scored)]

    return MetricResult(
        fmean(values),
        len(values),
        Resampling("seed"),
        components={
            "maximum": max(values),
            "minimum": min(values),
            "mean": fmean(values),
            "sd": stdev(values),
            "seeds": float(len(values)),
        },
    )


type PunctuationPolicy = Literal["keep", "strip"]
"""Specifies whether scoring keeps or strips punctuation."""

type UnaryChainPolicy = Literal["keep", "collapse"]
"""Specifies whether a gold unary chain stays a chain, or collapses to one node."""

type TrivialSpanPolicy = Literal["keep", "drop"]
"""Specifies whether the whole-sentence span and single-word spans are scored."""

type Aggregation = Literal["sentence", "corpus"]
"""Specifies sentence-level or corpus-level score aggregation."""

type CoveragePenalty = Literal["ignore", "zero"]
"""Specifies what an item the system produced no analysis for costs."""

type SelfArcPolicy = Literal["allow", "forbid"]
"""Specifies whether a dependency from a token to itself is a legal arc."""

type GoldDerivation = Literal["canonical", "oracle"]
"""Specifies which gold derivation a prediction is scored against."""

_PUNCTUATION_LEVELS: Final[tuple[PunctuationPolicy, ...]] = ("keep", "strip")
_UNARY_CHAIN_LEVELS: Final[tuple[UnaryChainPolicy, ...]] = ("keep", "collapse")
_TRIVIAL_SPAN_LEVELS: Final[tuple[TrivialSpanPolicy, ...]] = ("keep", "drop")
_MINIMUM_LENGTH_LEVELS: Final[tuple[int, ...]] = (0, 3)
_AGGREGATION_LEVELS: Final[tuple[Aggregation, ...]] = ("sentence", "corpus")
_COVERAGE_LEVELS: Final[tuple[CoveragePenalty, ...]] = ("ignore", "zero")
_SELF_ARC_LEVELS: Final[tuple[SelfArcPolicy, ...]] = ("allow", "forbid")
_GOLD_DERIVATION_LEVELS: Final[tuple[GoldDerivation, ...]] = ("canonical", "oracle")


class ConventionVector:
    """The eight scoring conventions for a published value.

    Parameters
    ----------
    punctuation : PunctuationPolicy
        Punctuation policy.
    unary_chains : UnaryChainPolicy
        Unary-chain policy.
    trivial_spans : TrivialSpanPolicy
        Trivial-span policy.
    minimum_length : int
        Minimum scored sentence length.
    aggregation : Aggregation
        Score aggregation level.
    coverage : CoveragePenalty
        Missing-analysis policy.
    self_arcs : SelfArcPolicy
        Self-arc policy.
    gold_derivation : GoldDerivation
        Gold-derivation policy.
    """

    __slots__ = (
        "_aggregation",
        "_coverage",
        "_factors",
        "_gold_derivation",
        "_minimum_length",
        "_punctuation",
        "_self_arcs",
        "_trivial_spans",
        "_unary_chains",
    )

    def __init__(
        self,
        *,
        punctuation: PunctuationPolicy,
        unary_chains: UnaryChainPolicy,
        trivial_spans: TrivialSpanPolicy,
        minimum_length: int,
        aggregation: Aggregation,
        coverage: CoveragePenalty,
        self_arcs: SelfArcPolicy,
        gold_derivation: GoldDerivation,
    ) -> None:
        self._punctuation: PunctuationPolicy = punctuation
        self._unary_chains: UnaryChainPolicy = unary_chains
        self._trivial_spans: TrivialSpanPolicy = trivial_spans
        self._minimum_length = minimum_length
        self._aggregation: Aggregation = aggregation
        self._coverage: CoveragePenalty = coverage
        self._self_arcs: SelfArcPolicy = self_arcs
        self._gold_derivation: GoldDerivation = gold_derivation
        self._factors: Mapping[str, JsonValue] = MappingProxyType(
            {
                "punctuation": self._punctuation,
                "unary_chains": self._unary_chains,
                "trivial_spans": self._trivial_spans,
                "minimum_length": self._minimum_length,
                "aggregation": self._aggregation,
                "coverage": self._coverage,
                "self_arcs": self._self_arcs,
                "gold_derivation": self._gold_derivation,
            }
        )

    @property
    def punctuation(self) -> PunctuationPolicy:
        """Return whether punctuation is scored or stripped first."""
        return self._punctuation

    @property
    def unary_chains(self) -> UnaryChainPolicy:
        """Return whether a gold unary chain stays a chain or collapses."""
        return self._unary_chains

    @property
    def trivial_spans(self) -> TrivialSpanPolicy:
        """Return whether the whole-sentence span and single-word spans are scored."""
        return self._trivial_spans

    @property
    def minimum_length(self) -> int:
        """Return the shortest sentence still scored, after punctuation is removed."""
        return self._minimum_length

    @property
    def aggregation(self) -> Aggregation:
        """Return whether a corpus score averages sentences or pools spans."""
        return self._aggregation

    @property
    def coverage(self) -> CoveragePenalty:
        """Return the penalty for an item without an analysis."""
        return self._coverage

    @property
    def self_arcs(self) -> SelfArcPolicy:
        """Return whether a token may depend on itself."""
        return self._self_arcs

    @property
    def gold_derivation(self) -> GoldDerivation:
        """Return which gold derivation the prediction is scored against."""
        return self._gold_derivation

    @property
    def factors(self) -> Mapping[str, JsonValue]:
        """Return the eight conventions as a read-only mapping, in declared order."""
        return self._factors

    def extended(self, factors: Mapping[str, JsonValue], /) -> ConventionVector:
        """Return this vector with evaluator-specific conventions appended.

        Parameters
        ----------
        factors : Mapping[str, JsonValue]
            Additional named conventions in their declaration order. A name the
            shared vector already owns is refused rather than overwritten.

        Returns
        -------
        ConventionVector
            A new vector with the same shared choices and the added factors.

        Raises
        ------
        ValueError
            If an added name is empty or already belongs to the vector.
        """
        additions: dict[str, JsonValue] = {}

        for name, value in factors.items():
            key = ensure_name("an evaluator convention", name)

            if key in self._factors:
                msg = f"a convention vector already declares {key!r}"
                raise ValueError(msg)

            additions[key] = value

        extended = ConventionVector(
            punctuation=self._punctuation,
            unary_chains=self._unary_chains,
            trivial_spans=self._trivial_spans,
            minimum_length=self._minimum_length,
            aggregation=self._aggregation,
            coverage=self._coverage,
            self_arcs=self._self_arcs,
            gold_derivation=self._gold_derivation,
        )
        extended._factors = MappingProxyType({**self._factors, **additions})

        return extended

    @classmethod
    def from_settings(cls, settings: ConventionSettings, /) -> ConventionVector:
        """Return the vector a validated convention declaration sets.

        Parameters
        ----------
        settings : ConventionSettings
            The eight conventions a run document declared.
        """
        return cls(
            punctuation=settings.punctuation,
            unary_chains=settings.unary_chains,
            trivial_spans=settings.trivial_spans,
            minimum_length=settings.minimum_length,
            aggregation=settings.aggregation,
            coverage=settings.coverage,
            self_arcs=settings.self_arcs,
            gold_derivation=settings.gold_derivation,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value sets the same eight conventions."""
        if type(other) is not ConventionVector:
            return NotImplemented

        return tuple(self._factors.items()) == tuple(other._factors.items())

    @override
    def __hash__(self) -> int:
        """Return a hash over the eight conventions."""
        return hash((ConventionVector, tuple(self._factors.items())))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this vector."""
        shared = tuple(self._factors.items())[: len(CONVENTION_LEVELS)]
        arguments = ", ".join(f"{key}={value!r}" for key, value in shared)
        rebuilt = f"ConventionVector({arguments})"
        additions = dict(tuple(self._factors.items())[len(CONVENTION_LEVELS) :])

        return rebuilt if not additions else f"{rebuilt}.extended({additions!r})"


CONVENTION_LEVELS: Final[Mapping[str, tuple[JsonValue, ...]]] = MappingProxyType(
    {
        "punctuation": _PUNCTUATION_LEVELS,
        "unary_chains": _UNARY_CHAIN_LEVELS,
        "trivial_spans": _TRIVIAL_SPAN_LEVELS,
        "minimum_length": _MINIMUM_LENGTH_LEVELS,
        "aggregation": _AGGREGATION_LEVELS,
        "coverage": _COVERAGE_LEVELS,
        "self_arcs": _SELF_ARC_LEVELS,
        "gold_derivation": _GOLD_DERIVATION_LEVELS,
    }
)
"""Convention levels in crossing order."""


def convention_crossing() -> tuple[ConventionVector, ...]:
    """Return every convention vector in declared order.

    Returns
    -------
    tuple[ConventionVector, ...]
        Full crossing of declared convention levels.
    """
    return tuple(
        ConventionVector(
            punctuation=punctuation,
            unary_chains=unary_chains,
            trivial_spans=trivial_spans,
            minimum_length=minimum_length,
            aggregation=aggregation,
            coverage=coverage,
            self_arcs=self_arcs,
            gold_derivation=gold_derivation,
        )
        for punctuation in _PUNCTUATION_LEVELS
        for unary_chains in _UNARY_CHAIN_LEVELS
        for trivial_spans in _TRIVIAL_SPAN_LEVELS
        for minimum_length in _MINIMUM_LENGTH_LEVELS
        for aggregation in _AGGREGATION_LEVELS
        for coverage in _COVERAGE_LEVELS
        for self_arcs in _SELF_ARC_LEVELS
        for gold_derivation in _GOLD_DERIVATION_LEVELS
    )
