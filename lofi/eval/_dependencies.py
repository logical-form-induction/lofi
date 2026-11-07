"""Complete-corpus CCG dependency scoring."""

from collections import Counter
from statistics import fmean
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Protocol,
    final,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._types import ItemId
from lofi.data.deps import ProjectedDependency
from lofi.eval._values import (
    Aggregation,
    ConventionVector,
    CoveragePenalty,
    GoldDerivation,
    MetricResult,
    PunctuationPolicy,
    Resampling,
    SelfArcPolicy,
    TrivialSpanPolicy,
    UnaryChainPolicy,
)
from lofi.stats import Interval

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable

__all__ = [
    "DEPENDENCY_MATCH_MODES",
    "CompatibilityPattern",
    "DependencyArc",
    "DependencyCompatibilityFilter",
    "DependencyMatchMode",
    "DependencyPrediction",
    "DependencyScoreRow",
    "DependencyScorer",
    "DependencyToken",
    "parg_gold",
    "projected_prediction",
]

type DependencyMatchMode = Literal[
    "labeled-directed",
    "labeled-undirected",
    "unlabeled-directed",
    "unlabeled-undirected",
]
"""A dependency matching relation."""

DEPENDENCY_MATCH_MODES: Final[tuple[DependencyMatchMode, ...]] = (
    "labeled-directed",
    "labeled-undirected",
    "unlabeled-directed",
    "unlabeled-undirected",
)
"""Dependency matching relations in report order."""

_COVERAGE_POLICIES: Final[tuple[CoveragePenalty, ...]] = ("ignore", "zero")
_SENTENCE_UNITS: Final = Resampling("sentence")

type _MatchKey = tuple[str | int, ...]
type _ConventionState = tuple[
    PunctuationPolicy,
    UnaryChainPolicy,
    TrivialSpanPolicy,
    int,
    Aggregation,
    CoveragePenalty,
    SelfArcPolicy,
    GoldDerivation,
]
type _ResultState = tuple[
    float | None,
    int,
    Resampling,
    tuple[tuple[str, float], ...],
    Interval | None,
]


@runtime_checkable
class _PargDependency(Protocol):
    """A corpus dependency record consumable by PARG scoring."""

    @property
    def target_index(self) -> int:
        """Return the argument-token position."""

    @property
    def head_index(self) -> int:
        """Return the predicate-token position."""

    @property
    def head_category(self) -> str:
        """Return the lexical category of the predicate."""

    @property
    def argument_slot(self) -> int:
        """Return the one-based argument slot."""

    @property
    def target_word(self) -> str:
        """Return the argument-token surface form."""

    @property
    def head_word(self) -> str:
        """Return the predicate-token surface form."""


@runtime_checkable
class _PargSentence(Protocol):
    """A corpus sentence record consumable by PARG scoring."""

    @property
    def identifier(self) -> str:
        """Return the corpus sentence identifier."""

    @property
    def last_index(self) -> int:
        """Return the final token position."""

    @property
    def dependencies(self) -> tuple[_PargDependency, ...]:
        """Return the sentence's dependency records."""


type _RowState = tuple[str, DependencyMatchMode, _ConventionState, _ResultState]


@final
class DependencyToken(FrozenValue):
    """A positioned surface token."""

    _noun = "dependency token"

    __slots__ = ("_form", "_position")

    def __init__(self, position: int, form: str) -> None:
        object.__setattr__(
            self, "_position", ensure_non_negative("a dependency position", position)
        )
        object.__setattr__(self, "_form", ensure_name("a dependency form", form))

    @property
    def position(self) -> int:
        """Return the token position."""
        return self._position

    @property
    def form(self) -> str:
        """Return the surface form."""
        return self._form

    def _fields(self) -> tuple[int, str]:
        """Return fields in declaration order."""
        return self._position, self._form

    @override
    def __reduce__(self) -> tuple[type[DependencyToken], tuple[int, str]]:
        """Return the reconstruction call."""
        return DependencyToken, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another token has the same position and form."""
        if type(other) is not DependencyToken:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the position and form."""
        return hash((DependencyToken, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"DependencyToken{self._fields()!r}"


@final
class DependencyArc(FrozenValue):
    """A dependency arc with scoring and extraction metadata."""

    _noun = "dependency arc"

    __slots__ = ("_argument", "_category", "_predicate", "_rule_id", "_slot")

    def __init__(
        self,
        predicate: DependencyToken,
        argument: DependencyToken,
        category: str | None,
        slot: int | None,
        rule_id: str | None,
    ) -> None:
        if not isinstance(predicate, DependencyToken):
            msg = f"a dependency predicate is not a token: {predicate!r}"
            raise TypeError(msg)
        if not isinstance(argument, DependencyToken):
            msg = f"a dependency argument is not a token: {argument!r}"
            raise TypeError(msg)
        if category is not None:
            ensure_name("a dependency category", category)
        if slot is not None and (isinstance(slot, bool) or not isinstance(slot, int)):
            msg = f"a dependency slot is not an integer: {slot!r}"
            raise TypeError(msg)
        if rule_id is not None:
            ensure_name("a dependency rule id", rule_id)

        object.__setattr__(self, "_predicate", predicate)
        object.__setattr__(self, "_argument", argument)
        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_slot", slot)
        object.__setattr__(self, "_rule_id", rule_id)

    @property
    def predicate(self) -> DependencyToken:
        """Return the predicate token."""
        return self._predicate

    @property
    def argument(self) -> DependencyToken:
        """Return the argument token."""
        return self._argument

    @property
    def predicate_position(self) -> int:
        """Return the predicate position."""
        return self._predicate.position

    @property
    def predicate_form(self) -> str:
        """Return the predicate form."""
        return self._predicate.form

    @property
    def argument_position(self) -> int:
        """Return the argument position."""
        return self._argument.position

    @property
    def argument_form(self) -> str:
        """Return the argument form."""
        return self._argument.form

    @property
    def category(self) -> str | None:
        """Return the normalized predicate category, if present."""
        return self._category

    @property
    def slot(self) -> int | None:
        """Return the argument slot, if present."""
        return self._slot

    @property
    def rule_id(self) -> str | None:
        """Return the extraction rule identifier, if present."""
        return self._rule_id

    def _fields(
        self,
    ) -> tuple[DependencyToken, DependencyToken, str | None, int | None, str | None]:
        """Return fields in declaration order."""
        return (
            self._predicate,
            self._argument,
            self._category,
            self._slot,
            self._rule_id,
        )

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[DependencyArc],
        tuple[DependencyToken, DependencyToken, str | None, int | None, str | None],
    ]:
        """Return the reconstruction call."""
        return DependencyArc, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another arc has the same fields."""
        if type(other) is not DependencyArc:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((DependencyArc, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"DependencyArc{self._fields()!r}"


@final
class DependencyPrediction(FrozenValue):
    """A successful parse's dependency multiset."""

    _noun = "dependency prediction"

    __slots__ = ("_arcs",)

    def __init__(self, arcs: Iterable[DependencyArc]) -> None:
        held = tuple(arcs)
        for position, arc in enumerate(held):
            if not isinstance(arc, DependencyArc):
                msg = f"dependency arc {position} is not one: {arc!r}"
                raise TypeError(msg)
        object.__setattr__(self, "_arcs", held)

    @property
    def arcs(self) -> tuple[DependencyArc, ...]:
        """Return the arcs in source order."""
        return self._arcs

    @override
    def __reduce__(
        self,
    ) -> tuple[type[DependencyPrediction], tuple[tuple[DependencyArc, ...]]]:
        """Return the reconstruction call."""
        return DependencyPrediction, (self._arcs,)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another prediction has the same arc multiplicities."""
        if type(other) is not DependencyPrediction:
            return NotImplemented
        return Counter(self._arcs) == Counter(other._arcs)

    @override
    def __hash__(self) -> int:
        """Return a hash over arc multiplicities."""
        return hash((DependencyPrediction, frozenset(Counter(self._arcs).items())))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"DependencyPrediction({self._arcs!r})"


@final
class CompatibilityPattern[C: Hashable](FrozenValue):
    """A caller-declared raw dependency exclusion pattern."""

    _noun = "compatibility pattern"

    __slots__ = (
        "_argument_form",
        "_predicate_form",
        "_raw_category",
        "_rule_id",
        "_slot",
    )

    def __init__(
        self,
        predicate_form: str | None = None,
        argument_form: str | None = None,
        raw_category: C | None = None,
        slot: int | None = None,
        rule_id: str | None = None,
    ) -> None:
        if predicate_form is not None:
            ensure_name("a compatibility predicate form", predicate_form)
        if argument_form is not None:
            ensure_name("a compatibility argument form", argument_form)
        if slot is not None:
            if isinstance(slot, bool) or not isinstance(slot, int):
                msg = f"a compatibility slot is not an integer: {slot!r}"
                raise TypeError(msg)
            if slot < 1:
                msg = f"a compatibility slot is not positive: {slot}"
                raise ValueError(msg)
        if rule_id is not None:
            ensure_name("a compatibility rule id", rule_id)
        if all(
            part is None
            for part in (predicate_form, argument_form, raw_category, slot, rule_id)
        ):
            msg = "a compatibility pattern has no constraints"
            raise ValueError(msg)

        object.__setattr__(self, "_predicate_form", predicate_form)
        object.__setattr__(self, "_argument_form", argument_form)
        object.__setattr__(self, "_raw_category", raw_category)
        object.__setattr__(self, "_slot", slot)
        object.__setattr__(self, "_rule_id", rule_id)

    @property
    def predicate_form(self) -> str | None:
        """Return the predicate-form constraint."""
        return self._predicate_form

    @property
    def argument_form(self) -> str | None:
        """Return the argument-form constraint."""
        return self._argument_form

    @property
    def raw_category(self) -> C | None:
        """Return the raw-category constraint."""
        return self._raw_category

    @property
    def slot(self) -> int | None:
        """Return the slot constraint."""
        return self._slot

    @property
    def rule_id(self) -> str | None:
        """Return the rule-id constraint."""
        return self._rule_id

    def matches(self, dependency: ProjectedDependency[C]) -> bool:
        """Return whether the raw dependency satisfies every constraint."""
        if not isinstance(dependency, ProjectedDependency):
            msg = f"a projected dependency is not one: {dependency!r}"
            raise TypeError(msg)
        return (
            (
                self._predicate_form is None
                or self._predicate_form == dependency.predicate_form
            )
            and (
                self._argument_form is None
                or self._argument_form == dependency.argument_form
            )
            and (
                self._raw_category is None
                or self._raw_category == dependency.raw_category
            )
            and (self._slot is None or self._slot == dependency.slot)
            and (self._rule_id is None or self._rule_id == dependency.rule_id)
        )

    def _fields(
        self,
    ) -> tuple[str | None, str | None, C | None, int | None, str | None]:
        """Return fields in declaration order."""
        return (
            self._predicate_form,
            self._argument_form,
            self._raw_category,
            self._slot,
            self._rule_id,
        )

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[CompatibilityPattern[C]],
        tuple[str | None, str | None, C | None, int | None, str | None],
    ]:
        """Return the reconstruction call."""
        return CompatibilityPattern, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another pattern has the same constraints."""
        if type(other) is not CompatibilityPattern:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the constraints."""
        return hash((CompatibilityPattern, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"CompatibilityPattern{self._fields()!r}"


@final
class DependencyCompatibilityFilter[C: Hashable](FrozenValue):
    """Caller-owned dependency exclusions with source provenance."""

    _noun = "dependency compatibility filter"

    __slots__ = ("_patterns", "_provenance")

    def __init__(
        self,
        patterns: Iterable[CompatibilityPattern[C]],
        provenance: str,
    ) -> None:
        held = tuple(patterns)
        for position, pattern in enumerate(held):
            if not isinstance(pattern, CompatibilityPattern):
                msg = f"compatibility pattern {position} is not one: {pattern!r}"
                raise TypeError(msg)
        object.__setattr__(self, "_patterns", held)
        object.__setattr__(
            self,
            "_provenance",
            ensure_name("compatibility-filter provenance", provenance),
        )

    @property
    def patterns(self) -> tuple[CompatibilityPattern[C], ...]:
        """Return the exclusion patterns."""
        return self._patterns

    @property
    def provenance(self) -> str:
        """Return the source description."""
        return self._provenance

    def keeps(self, dependency: ProjectedDependency[C]) -> bool:
        """Return whether no exclusion pattern matches the dependency."""
        if not isinstance(dependency, ProjectedDependency):
            msg = f"a projected dependency is not one: {dependency!r}"
            raise TypeError(msg)
        return not any(pattern.matches(dependency) for pattern in self._patterns)

    def _fields(self) -> tuple[tuple[CompatibilityPattern[C], ...], str]:
        """Return fields in declaration order."""
        return self._patterns, self._provenance

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[DependencyCompatibilityFilter[C]],
        tuple[tuple[CompatibilityPattern[C], ...], str],
    ]:
        """Return the reconstruction call."""
        return DependencyCompatibilityFilter, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another filter has the same patterns and provenance."""
        if type(other) is not DependencyCompatibilityFilter:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the patterns and provenance."""
        return hash((DependencyCompatibilityFilter, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"DependencyCompatibilityFilter{self._fields()!r}"


def projected_prediction[C: Hashable](
    dependencies: Iterable[ProjectedDependency[C]],
    compatibility: DependencyCompatibilityFilter[C],
    category_text: Callable[[C], str],
) -> DependencyPrediction:
    """Adapt projected dependencies after applying raw exclusions.

    Raises
    ------
    TypeError
        If the filter, category renderer, or a dependency has the wrong type.
    ValueError
        If the category renderer returns an invalid name.
    """
    if not isinstance(compatibility, DependencyCompatibilityFilter):
        msg = f"a dependency compatibility filter is not one: {compatibility!r}"
        raise TypeError(msg)
    if not callable(category_text):
        msg = f"a dependency category renderer is not callable: {category_text!r}"
        raise TypeError(msg)

    arcs: list[DependencyArc] = []
    for position, dependency in enumerate(dependencies):
        if not isinstance(dependency, ProjectedDependency):
            msg = f"projected dependency {position} is not one: {dependency!r}"
            raise TypeError(msg)
        if dependency.predicate_position == dependency.argument_position:
            continue
        if not compatibility.keeps(dependency):
            continue
        category = ensure_name(
            "a normalized dependency category", category_text(dependency.raw_category)
        )
        arcs.append(
            DependencyArc(
                DependencyToken(
                    dependency.predicate_position, dependency.predicate_form
                ),
                DependencyToken(dependency.argument_position, dependency.argument_form),
                category,
                dependency.slot,
                dependency.rule_id,
            )
        )
    return DependencyPrediction(arcs)


def parg_gold(sentence: _PargSentence) -> DependencyPrediction:
    """Adapt one PARG sentence directly to scoring arcs.

    Raises
    ------
    TypeError
        If the sentence or one of its dependencies has the wrong type.
    ValueError
        If an identifier or index is invalid, an arc exceeds the sentence, or an
        argument slot is not positive.
    """
    if not isinstance(sentence, _PargSentence):
        msg = f"a PARG sentence is not one: {sentence!r}"
        raise TypeError(msg)
    ensure_name("a PARG sentence identifier", sentence.identifier)
    last_index = ensure_non_negative(
        "a PARG sentence's last index", sentence.last_index
    )

    arcs: list[DependencyArc] = []
    for position, dependency in enumerate(sentence.dependencies):
        if not isinstance(dependency, _PargDependency):
            msg = f"PARG dependency {position} is not one: {dependency!r}"
            raise TypeError(msg)
        if dependency.target_index > last_index or dependency.head_index > last_index:
            msg = f"PARG dependency {position} exceeds last index {last_index}"
            raise ValueError(msg)
        if dependency.argument_slot < 1:
            msg = (
                f"PARG dependency {position} has nonpositive slot "
                f"{dependency.argument_slot}"
            )
            raise ValueError(msg)
        if dependency.head_index == dependency.target_index:
            continue
        arcs.append(
            DependencyArc(
                DependencyToken(dependency.head_index, dependency.head_word),
                DependencyToken(dependency.target_index, dependency.target_word),
                ensure_name("a PARG head category", dependency.head_category),
                dependency.argument_slot,
                None,
            )
        )
    return DependencyPrediction(arcs)


@final
class DependencyScoreRow(FrozenValue):
    """A named dependency score and its conventions."""

    _noun = "dependency score row"

    __slots__ = ("_conventions", "_mode", "_name", "_result")

    def __init__(
        self,
        name: str,
        mode: DependencyMatchMode,
        conventions: ConventionVector,
        result: MetricResult,
    ) -> None:
        if mode not in DEPENDENCY_MATCH_MODES:
            msg = f"an unknown dependency match mode was requested: {mode!r}"
            raise ValueError(msg)
        if not isinstance(conventions, ConventionVector):
            msg = f"dependency conventions are not a vector: {conventions!r}"
            raise TypeError(msg)
        if not isinstance(result, MetricResult):
            msg = f"a dependency score result is not one: {result!r}"
            raise TypeError(msg)
        object.__setattr__(self, "_name", ensure_name("a dependency row name", name))
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_conventions", conventions)
        object.__setattr__(self, "_result", result)

    @property
    def name(self) -> str:
        """Return the row name."""
        return self._name

    @property
    def mode(self) -> DependencyMatchMode:
        """Return the dependency matching relation."""
        return self._mode

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def result(self) -> MetricResult:
        """Return the metric result."""
        return self._result

    def _fields(
        self,
    ) -> tuple[str, DependencyMatchMode, ConventionVector, MetricResult]:
        """Return fields in declaration order."""
        return self._name, self._mode, self._conventions, self._result

    @override
    def __reduce__(
        self,
    ) -> tuple[
        Callable[[_RowState], DependencyScoreRow],
        tuple[_RowState],
    ]:
        """Return the reconstruction call."""
        conventions: _ConventionState = (
            self._conventions.punctuation,
            self._conventions.unary_chains,
            self._conventions.trivial_spans,
            self._conventions.minimum_length,
            self._conventions.aggregation,
            self._conventions.coverage,
            self._conventions.self_arcs,
            self._conventions.gold_derivation,
        )
        result: _ResultState = (
            self._result.value,
            self._result.n,
            self._result.resampling,
            tuple(self._result.components.items()),
            self._result.ci,
        )
        return _restore_score_row, ((self._name, self._mode, conventions, result),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row has the same fields."""
        if type(other) is not DependencyScoreRow:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((DependencyScoreRow, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"DependencyScoreRow{self._fields()!r}"


def _restore_score_row(state: _RowState) -> DependencyScoreRow:
    """Rebuild a dependency score row from pickle-safe parts."""
    name, mode, conventions, result = state
    punctuation, unary, trivial, minimum, aggregation, coverage, self_arcs, gold = (
        conventions
    )
    value, n, resampling, components, ci = result
    return DependencyScoreRow(
        name,
        mode,
        ConventionVector(
            punctuation=punctuation,
            unary_chains=unary,
            trivial_spans=trivial,
            minimum_length=minimum,
            aggregation=aggregation,
            coverage=coverage,
            self_arcs=self_arcs,
            gold_derivation=gold,
        ),
        MetricResult(
            value,
            n,
            resampling,
            components=dict(components),
            ci=ci,
        ),
    )


class _SentenceCounts(NamedTuple):
    matched: int
    predicted: int
    gold: int
    exact: float


class _Item(NamedTuple):
    prediction: DependencyPrediction | None
    gold: DependencyPrediction


def _validated(prediction: DependencyPrediction, role: str) -> DependencyPrediction:
    """Validate labeled fields and remove self arcs."""
    arcs: list[DependencyArc] = []
    for position, arc in enumerate(prediction.arcs):
        if arc.predicate_position == arc.argument_position:
            continue
        if arc.category is None:
            msg = f"{role} dependency {position} has no category"
            raise ValueError(msg)
        if arc.slot is None:
            msg = f"{role} dependency {position} has no argument slot"
            raise ValueError(msg)
        if arc.slot < 1:
            msg = f"{role} dependency {position} has nonpositive slot {arc.slot}"
            raise ValueError(msg)
        arcs.append(arc)
    return DependencyPrediction(arcs)


def _token_key(token: DependencyToken) -> tuple[int, str]:
    """Build a token identity from position and form."""
    return token.position, token.form


def _match_key(arc: DependencyArc, mode: DependencyMatchMode) -> _MatchKey:
    """Build an arc key under one matching relation."""
    predicate = _token_key(arc.predicate)
    argument = _token_key(arc.argument)
    endpoints = (
        (predicate, argument)
        if mode.endswith("directed") and not mode.endswith("undirected")
        else tuple(sorted((predicate, argument)))
    )
    if mode.startswith("labeled"):
        return (*endpoints[0], *endpoints[1], arc.category or "", arc.slot or 0)
    return (*endpoints[0], *endpoints[1])


def _counts(
    prediction: DependencyPrediction | None,
    gold: DependencyPrediction,
    mode: DependencyMatchMode,
) -> _SentenceCounts:
    """Count multiset matches for one sentence."""
    predicted = Counter[_MatchKey](
        _match_key(arc, mode) for arc in (() if prediction is None else prediction.arcs)
    )
    expected = Counter[_MatchKey](_match_key(arc, mode) for arc in gold.arcs)
    matched = sum((predicted & expected).values())
    return _SentenceCounts(
        matched,
        predicted.total(),
        expected.total(),
        float(prediction is not None and predicted == expected),
    )


def _ratio(numerator: int, denominator: int) -> float:
    """Divide counts with an explicit zero-denominator value."""
    return numerator / denominator if denominator else 0.0


def _f_score(precision: float, recall: float) -> float:
    """Compute harmonic mean with an explicit zero value."""
    return (
        2.0 * precision * recall / (precision + recall) if precision + recall else 0.0
    )


def _conventions(
    aggregation: Literal["sentence", "corpus"],
    coverage: CoveragePenalty,
) -> ConventionVector:
    """Build dependency scoring conventions."""
    return ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation=aggregation,
        coverage=coverage,
        self_arcs="forbid",
        gold_derivation="canonical",
    )


def _result(
    value: float | None,
    n: int,
    *,
    matched: int,
    predicted: int,
    gold: int,
) -> MetricResult:
    """Build a dependency metric result."""
    return MetricResult(
        value,
        n,
        _SENTENCE_UNITS,
        components={
            "matched": float(matched),
            "predicted": float(predicted),
            "gold": float(gold),
        },
    )


def _sentence_result(exact: float | None, n: int, matches: int) -> MetricResult:
    """Build a sentence exact-match result."""
    return MetricResult(
        exact,
        n,
        _SENTENCE_UNITS,
        components={"exact": float(matches), "sentences": float(n)},
    )


def _coverage_result(value: float | None, parsed: int, total: int) -> MetricResult:
    """Build a parse coverage result."""
    return MetricResult(
        value,
        total,
        _SENTENCE_UNITS,
        components={"parsed": float(parsed), "sentences": float(total)},
    )


@final
class DependencyScorer:
    """A complete-corpus dependency scorer."""

    __slots__ = ("_items",)

    def __init__(self) -> None:
        self._items: dict[ItemId, _Item] = {}

    @property
    def name(self) -> str:
        """Return the scorer name."""
        return "dependency"

    @property
    def order_invariant(self) -> bool:
        """Return whether rows are invariant to update order."""
        return True

    def reset(self) -> None:
        """Clear all scored sentences."""
        self._items = {}

    def update(
        self,
        prediction: DependencyPrediction | None,
        gold: DependencyPrediction,
        *,
        key: ItemId,
    ) -> None:
        """Store one predicted and gold dependency analysis."""
        if prediction is not None and not isinstance(prediction, DependencyPrediction):
            msg = f"a dependency prediction is not one: {prediction!r}"
            raise TypeError(msg)
        if not isinstance(gold, DependencyPrediction):
            msg = f"a gold dependency analysis is not a prediction value: {gold!r}"
            raise TypeError(msg)

        identifier = ItemId(ensure_name("a dependency item id", key))
        held_prediction = (
            None if prediction is None else _validated(prediction, "predicted")
        )
        item = _Item(held_prediction, _validated(gold, "gold"))
        standing = self._items.get(identifier)
        if standing is not None:
            if standing != item:
                msg = f"item {identifier!r} was already scored against other analyses"
                raise ValueError(msg)
            return
        self._items[identifier] = item

    def rows(self) -> tuple[DependencyScoreRow, ...]:
        """Return all dependency scores in deterministic order."""
        items = tuple(self._items[key] for key in sorted(self._items))
        total = len(items)
        parsed = sum(item.prediction is not None for item in items)
        rows: list[DependencyScoreRow] = []

        for mode in DEPENDENCY_MATCH_MODES:
            for coverage in _COVERAGE_POLICIES:
                eligible = tuple(
                    _counts(item.prediction, item.gold, mode)
                    for item in items
                    if coverage == "zero" or item.prediction is not None
                )
                rows.extend(
                    self._score_rows(
                        mode, coverage, eligible, parsed=parsed, total=total
                    )
                )
        return tuple(rows)

    @staticmethod
    def _score_rows(
        mode: DependencyMatchMode,
        coverage: CoveragePenalty,
        counts: tuple[_SentenceCounts, ...],
        *,
        parsed: int,
        total: int,
    ) -> tuple[DependencyScoreRow, ...]:
        """Build the eight rows for one mode and coverage policy."""
        n = len(counts)
        matched = sum(entry.matched for entry in counts)
        predicted = sum(entry.predicted for entry in counts)
        gold = sum(entry.gold for entry in counts)
        micro_precision = None if not counts else _ratio(matched, predicted)
        micro_recall = None if not counts else _ratio(matched, gold)
        micro_f = (
            None
            if micro_precision is None or micro_recall is None
            else _f_score(micro_precision, micro_recall)
        )
        precisions = tuple(_ratio(entry.matched, entry.predicted) for entry in counts)
        recalls = tuple(_ratio(entry.matched, entry.gold) for entry in counts)
        f_scores = tuple(
            _f_score(precision, recall)
            for precision, recall in zip(precisions, recalls, strict=True)
        )
        macro_precision = None if not precisions else fmean(precisions)
        macro_recall = None if not recalls else fmean(recalls)
        macro_f = None if not f_scores else fmean(f_scores)
        exact = None if not counts else fmean(entry.exact for entry in counts)
        exact_matches = sum(int(entry.exact) for entry in counts)
        coverage_value = None if total == 0 else parsed / total
        corpus = _conventions("corpus", coverage)
        sentence = _conventions("sentence", coverage)

        def row(
            metric: str,
            conventions: ConventionVector,
            value: float | None,
        ) -> DependencyScoreRow:
            return DependencyScoreRow(
                f"dependency:{mode}:{coverage}:{metric}",
                mode,
                conventions,
                _result(
                    value,
                    n,
                    matched=matched,
                    predicted=predicted,
                    gold=gold,
                ),
            )

        return (
            row("micro-precision", corpus, micro_precision),
            row("micro-recall", corpus, micro_recall),
            row("micro-f-score", corpus, micro_f),
            row("macro-precision", sentence, macro_precision),
            row("macro-recall", sentence, macro_recall),
            row("macro-f-score", sentence, macro_f),
            DependencyScoreRow(
                f"dependency:{mode}:{coverage}:sentence-exact",
                mode,
                sentence,
                _sentence_result(exact, n, exact_matches),
            ),
            DependencyScoreRow(
                f"dependency:{mode}:{coverage}:coverage",
                mode,
                sentence,
                _coverage_result(coverage_value, parsed, total),
            ),
        )
