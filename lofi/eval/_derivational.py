"""Held-out derivational likelihood under a declared normal form and under none."""

from math import fsum, isclose
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, final, override

from lofi._checks import ensure_finite, ensure_name
from lofi.eval._values import MetricResult, Resampling
from lofi.judge import NoNormalForm

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from lofi._types import ItemId, JsonValue
    from lofi.data import Derivation
    from lofi.eval._values import ConventionVector
    from lofi.judge import NormalForm

__all__ = [
    "LIKELIHOOD_UNITS",
    "DerivationalLikelihood",
    "DerivationalLikelihoodReport",
    "DerivationalScoring",
    "LikelihoodRow",
    "LikelihoodUnit",
    "NormalFormScores",
    "RankChurn",
    "SentenceLikelihood",
    "derivational_likelihood_report",
]

type LikelihoodUnit = Literal["sentence", "token"]
"""The unit a held-out derivational log-likelihood is divided by."""

LIKELIHOOD_UNITS: Final[tuple[LikelihoodUnit, ...]] = ("sentence", "token")
"""The reporting units, in report order."""

_SCHEMES: Final[Mapping[LikelihoodUnit, Resampling]] = MappingProxyType(
    {unit: Resampling(unit) for unit in LIKELIHOOD_UNITS}
)
"""The resampling scheme each reporting unit counts in."""


@final
class SentenceLikelihood:
    """One sentence's gold derivation read against a derivation set.

    The score and the log partition are in the same log scale, so their
    difference is the log probability the model puts on the gold derivation.
    A score of None says the derivation set does not contain the gold
    derivation, which is what a normal form does to a gold derivation it
    rejects.

    Parameters
    ----------
    score : float | None
        Log score of the gold derivation, or None where the derivation set
        does not contain it.
    log_partition : float
        Log of the summed score over the whole derivation set.

    Raises
    ------
    TypeError
        If a number is not one.
    ValueError
        If a number is not finite, or the gold derivation outscores the whole
        derivation set.
    """

    __slots__ = ("_log_partition", "_score")

    def __init__(self, score: float | None, log_partition: float) -> None:
        held = None if score is None else ensure_finite("a derivation score", score)
        total = ensure_finite("a log partition", log_partition)

        if held is not None and held > total and not isclose(held, total):
            msg = (
                f"a gold derivation scores {score!r} against a derivation set "
                f"summing to {log_partition!r}"
            )
            raise ValueError(msg)

        self._score = held
        self._log_partition = total

    @property
    def score(self) -> float | None:
        """Return the gold derivation's log score, or None where it is absent."""
        return self._score

    @property
    def log_partition(self) -> float:
        """Return the log of the summed score over the derivation set."""
        return self._log_partition

    @property
    def log_likelihood(self) -> float | None:
        """Return the gold derivation's log probability, or None where it is absent."""
        return None if self._score is None else self._score - self._log_partition

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reports the same score and partition."""
        if type(other) is not SentenceLikelihood:
            return NotImplemented

        return (self._score, self._log_partition) == (
            other._score,
            other._log_partition,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the score and the partition."""
        return hash((SentenceLikelihood, self._score, self._log_partition))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this value."""
        return f"SentenceLikelihood({self._score!r}, {self._log_partition!r})"


class _Scored(NamedTuple):
    """One accumulated sentence.

    Attributes
    ----------
    likelihood : SentenceLikelihood
        The gold derivation read against the derivation set.
    tokens : int
        The number of leaves the gold derivation has.
    """

    likelihood: SentenceLikelihood
    tokens: int


class _Contained(NamedTuple):
    """One sentence whose gold derivation the derivation set contains.

    Attributes
    ----------
    log_likelihood : float
        The gold derivation's log probability.
    tokens : int
        The number of leaves the gold derivation has.
    """

    log_likelihood: float
    tokens: int


@final
class DerivationalLikelihood[L: Hashable]:
    """Held-out derivational log-likelihood over one derivation set.

    Each update reads one held-out sentence's gold derivation against the
    derivation set the model normalizes over, and takes the sentence length
    from the gold derivation rather than from the caller. The aggregate
    divides the summed log probability by the sentence count or by the token
    count, over the sentences whose gold derivation the set contains. A
    sentence the set does not contain has no log probability and is counted
    instead.

    Parameters
    ----------
    normal_form : str
        Name of the filter the derivation set was built under.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty.
    """

    __slots__ = ("_items", "_normal_form")

    def __init__(self, normal_form: str) -> None:
        self._normal_form = ensure_name("a normal form", normal_form)
        self._items: dict[ItemId, _Scored] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "derivational-log-likelihood"

    @property
    def normal_form(self) -> str:
        """Return the name of the filter the derivation set was built under."""
        return self._normal_form

    @property
    def conventions(self) -> ConventionVector | None:
        """Return the scoring conventions, of which this metric has none."""
        return None

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    def reset(self) -> None:
        """Forget every sentence accumulated so far."""
        self._items = {}

    def update(
        self, pred: SentenceLikelihood, gold: Derivation[L], *, key: ItemId
    ) -> None:
        """Accumulate one held-out sentence.

        Parameters
        ----------
        pred : SentenceLikelihood
            The gold derivation's score and the derivation set's log partition.
        gold : Derivation[L]
            The gold derivation, whose leaves give the sentence length.
        key : ItemId
            Item identifier.

        Raises
        ------
        TypeError
            If the prediction is not a sentence likelihood.
        ValueError
            If the key was already accumulated against a different sentence.
        """
        if not isinstance(pred, SentenceLikelihood):
            msg = f"a held-out sentence's likelihood is not one: {pred!r}"
            raise TypeError(msg)

        scored = _Scored(pred, len(gold.leaves()))
        standing = self._items.get(key)

        if standing is not None:
            if standing != scored:
                msg = (
                    f"the item {key!r} was already accumulated against a "
                    f"different sentence"
                )
                raise ValueError(msg)

            return

        self._items[key] = scored

    def compute(self) -> MetricResult:
        """Return the mean log probability per held-out sentence.

        Returns
        -------
        MetricResult
            The per-sentence aggregate, with the sentence and token counts and
            the summed log probability as components.
        """
        return self.result("sentence")

    def result(self, unit: LikelihoodUnit) -> MetricResult:
        """Return the held-out log probability divided by one unit's count.

        Parameters
        ----------
        unit : LikelihoodUnit
            Whether to divide by the sentence count or the token count.

        Returns
        -------
        MetricResult
            The aggregate over the sentences the derivation set contains, or an
            absent value where it contains none.

        Raises
        ------
        ValueError
            If the unit is not a declared one.
        """
        scheme = _SCHEMES.get(unit)

        if scheme is None:
            msg = f"a held-out likelihood was asked for an undeclared unit: {unit!r}"
            raise ValueError(msg)

        contained = self._contained()
        components = self._counts(contained)

        if not contained:
            return MetricResult(None, 0, scheme, components=components)

        total = fsum(one.log_likelihood for one in contained)
        divisor = (
            len(contained)
            if unit == "sentence"
            else sum(one.tokens for one in contained)
        )

        return MetricResult(total / divisor, divisor, scheme, components=components)

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each held-out sentence's log probability.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            One result per key, valued at the gold derivation's log probability
            and absent where the derivation set does not contain it.
        """
        results: dict[ItemId, MetricResult] = {}

        for key, item in self._items.items():
            value = item.likelihood.log_likelihood
            components = {
                "tokens": float(item.tokens),
                "log_partition": item.likelihood.log_partition,
            }

            if value is not None:
                components["per_token"] = value / item.tokens

            results[key] = MetricResult(
                value,
                0 if value is None else 1,
                _SCHEMES["sentence"],
                components=components,
            )

        return MappingProxyType(results)

    def _contained(self) -> tuple[_Contained, ...]:
        """Return the sentences whose gold derivation the derivation set contains."""
        found: list[_Contained] = []

        for item in self._items.values():
            likelihood = item.likelihood.log_likelihood

            if likelihood is not None:
                found.append(_Contained(likelihood, item.tokens))

        return tuple(found)

    def _counts(self, contained: Sequence[_Contained]) -> dict[str, float]:
        """Return the counts every aggregate reports beside its value.

        Parameters
        ----------
        contained : Sequence[_Contained]
            The sentences the derivation set contains.
        """
        return {
            "sentences": float(len(self._items)),
            "contained": float(len(contained)),
            "uncontained": float(len(self._items) - len(contained)),
            "tokens": float(sum(one.tokens for one in contained)),
            "total": fsum(one.log_likelihood for one in contained),
        }

    @override
    def __repr__(self) -> str:
        """Return the normal form and the number of sentences accumulated."""
        return (
            f"DerivationalLikelihood({self._normal_form!r}, items={len(self._items)!r})"
        )


class NormalFormScores(NamedTuple):
    """One held-out sentence scored against both derivation sets.

    Attributes
    ----------
    constrained : SentenceLikelihood
        The gold derivation read against the derivation set the declared normal
        form admits.
    unconstrained : SentenceLikelihood
        The gold derivation read against the whole derivation set.
    """

    constrained: SentenceLikelihood
    unconstrained: SentenceLikelihood


class LikelihoodRow(NamedTuple):
    """One published held-out derivational log-likelihood.

    Attributes
    ----------
    model : str
        Name of the scored model.
    normal_form : str
        Name of the filter the derivation set was built under.
    result : MetricResult
        The log probability divided by the row's unit count.
    """

    model: str
    normal_form: str
    result: MetricResult

    @property
    def unit(self) -> str:
        """Return the unit the value is divided by."""
        return self.result.resampling.unit

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as report columns.

        Returns
        -------
        dict[str, JsonValue]
            The model, the normal form, the unit, the value, the unit count and
            the counts behind them.
        """
        components: dict[str, JsonValue] = dict(self.result.components)

        return {
            "model": self.model,
            "normal_form": self.normal_form,
            "unit": self.unit,
            "value": self.result.value,
            "n": self.result.n,
            "components": components,
        }


@final
class DerivationalScoring[L: Hashable]:
    """One model's held-out likelihood under its normal form and under none.

    The two metrics are built together and updated together, so every held-out
    sentence reaches both and the two aggregates differ only in which
    derivations the model normalized over.

    Parameters
    ----------
    normal_form : NormalForm[T]
        The filter the constrained derivation set was built under.

    Raises
    ------
    ValueError
        If the declared filter is the unrestricted one, which would report the
        same derivation set twice.
    """

    __slots__ = ("_constrained", "_unconstrained")

    def __init__[T](self, normal_form: NormalForm[T]) -> None:
        unrestricted = NoNormalForm[T]().name

        if normal_form.name == unrestricted:
            msg = (
                f"a derivational scoring declares the unrestricted filter as "
                f"its normal form: {normal_form.name!r}"
            )
            raise ValueError(msg)

        self._constrained: DerivationalLikelihood[L] = DerivationalLikelihood(
            normal_form.name
        )
        self._unconstrained: DerivationalLikelihood[L] = DerivationalLikelihood(
            unrestricted
        )

    @property
    def constrained(self) -> DerivationalLikelihood[L]:
        """Return the metric over the derivation set the normal form admits."""
        return self._constrained

    @property
    def unconstrained(self) -> DerivationalLikelihood[L]:
        """Return the metric over the whole derivation set."""
        return self._unconstrained

    def reset(self) -> None:
        """Forget every sentence accumulated under either derivation set."""
        self._constrained.reset()
        self._unconstrained.reset()

    def update(
        self, pred: NormalFormScores, gold: Derivation[L], *, key: ItemId
    ) -> None:
        """Accumulate one held-out sentence under both derivation sets.

        Parameters
        ----------
        pred : NormalFormScores
            The gold derivation read against each derivation set.
        gold : Derivation[L]
            The gold derivation, whose leaves give the sentence length.
        key : ItemId
            Item identifier.
        """
        self._constrained.update(pred.constrained, gold, key=key)
        self._unconstrained.update(pred.unconstrained, gold, key=key)

    def rows(self, model: str) -> tuple[LikelihoodRow, ...]:
        """Return the model's rows, the two derivation sets crossed with the units.

        Parameters
        ----------
        model : str
            Name of the scored model.

        Returns
        -------
        tuple[LikelihoodRow, ...]
            Four rows, the constrained pair before the unconstrained pair.

        Raises
        ------
        TypeError
            If the model name is not a string.
        ValueError
            If the model name is empty.
        """
        name = ensure_name("a model name", model)

        return tuple(
            LikelihoodRow(name, metric.normal_form, metric.result(unit))
            for metric in (self._constrained, self._unconstrained)
            for unit in LIKELIHOOD_UNITS
        )

    @override
    def __repr__(self) -> str:
        """Return the two metrics."""
        return (
            f"DerivationalScoring(constrained={self._constrained!r}, "
            f"unconstrained={self._unconstrained!r})"
        )


def _ranking(values: Mapping[str, float]) -> tuple[str, ...]:
    """Return the model names in descending value order, ties broken by name.

    Parameters
    ----------
    values : Mapping[str, float]
        Each model's value.
    """
    return tuple(sorted(values, key=lambda name: (-values[name], name)))


def _inversions(one: Sequence[str], other: Sequence[str]) -> int:
    """Count the model pairs the two rankings order differently.

    Parameters
    ----------
    one : Sequence[str]
        One ranking, best first.
    other : Sequence[str]
        The other ranking over the same models, best first.
    """
    place = {name: at for at, name in enumerate(other)}

    return sum(
        1
        for first in range(len(one))
        for second in range(first + 1, len(one))
        if place[one[first]] > place[one[second]]
    )


@final
class RankChurn:
    """How far two rankings of the same models move apart.

    Parameters
    ----------
    unit : LikelihoodUnit
        The unit the ranked values are divided by.
    constrained : Mapping[str, float]
        Each model's value under the derivation set its normal form admits.
    unconstrained : Mapping[str, float]
        Each model's value over the whole derivation set.

    Raises
    ------
    ValueError
        If the unit is not a declared one, if the two mappings name different
        models, if they name none, or if a value is not finite.
    """

    __slots__ = ("_constrained", "_inversions", "_moved", "_unconstrained", "_unit")

    def __init__(
        self,
        unit: LikelihoodUnit,
        constrained: Mapping[str, float],
        unconstrained: Mapping[str, float],
    ) -> None:
        if unit not in LIKELIHOOD_UNITS:
            msg = f"a rank churn names an undeclared unit: {unit!r}"
            raise ValueError(msg)

        if set(constrained) != set(unconstrained):
            msg = (
                f"a rank churn ranks {sorted(constrained)!r} under one "
                f"derivation set and {sorted(unconstrained)!r} under the other"
            )
            raise ValueError(msg)

        if not constrained:
            msg = "a rank churn ranks no model"
            raise ValueError(msg)

        for name, value in (*constrained.items(), *unconstrained.items()):
            ensure_finite(f"the ranked value of {name!r}", value)

        self._unit: LikelihoodUnit = unit
        self._constrained = _ranking(constrained)
        self._unconstrained = _ranking(unconstrained)
        self._inversions = _inversions(self._constrained, self._unconstrained)
        self._moved = sum(
            1
            for one, other in zip(self._constrained, self._unconstrained, strict=True)
            if one != other
        )

    @property
    def unit(self) -> LikelihoodUnit:
        """Return the unit the ranked values are divided by."""
        return self._unit

    @property
    def constrained(self) -> tuple[str, ...]:
        """Return the ranking under the admitted derivation sets, best first."""
        return self._constrained

    @property
    def unconstrained(self) -> tuple[str, ...]:
        """Return the ranking over the whole derivation sets, best first."""
        return self._unconstrained

    @property
    def inversions(self) -> int:
        """Return how many model pairs the two rankings order differently."""
        return self._inversions

    @property
    def moved(self) -> int:
        """Return how many models hold a different position in the two rankings."""
        return self._moved

    @property
    def pairs(self) -> int:
        """Return how many model pairs the two rankings could have disagreed on."""
        ranked = len(self._constrained)

        return ranked * (ranked - 1) // 2

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the churn as report columns.

        Returns
        -------
        dict[str, JsonValue]
            The unit, both rankings, the inversion and moved counts, and how
            many pairs stood to disagree.
        """
        return {
            "unit": self._unit,
            "constrained": list(self._constrained),
            "unconstrained": list(self._unconstrained),
            "inversions": self._inversions,
            "moved": self._moved,
            "pairs": self.pairs,
        }

    def _fields(self) -> tuple[LikelihoodUnit, tuple[str, ...], tuple[str, ...]]:
        """Return the three fields a churn is equal on."""
        return (self._unit, self._constrained, self._unconstrained)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reports the same unit and rankings."""
        if type(other) is not RankChurn:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the unit and the two rankings."""
        return hash((RankChurn, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the unit, the two rankings and the inversion count."""
        return (
            f"RankChurn({self._unit!r}, constrained={list(self._constrained)!r}, "
            f"unconstrained={list(self._unconstrained)!r}, "
            f"inversions={self._inversions!r})"
        )


class DerivationalLikelihoodReport(NamedTuple):
    """Every model's two likelihoods and the churn between the two rankings.

    Attributes
    ----------
    rows : tuple[LikelihoodRow, ...]
        Four rows per model, the two derivation sets crossed with the units.
    churn : tuple[RankChurn, ...]
        One churn per reporting unit, present whether or not it is zero.
    """

    rows: tuple[LikelihoodRow, ...]
    churn: tuple[RankChurn, ...]

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the report as its two lists of columns.

        Returns
        -------
        dict[str, JsonValue]
            The likelihood rows and the churn rows.
        """
        rows: list[JsonValue] = [row.to_spec() for row in self.rows]
        churn: list[JsonValue] = [entry.to_spec() for entry in self.churn]

        return {"rows": rows, "churn": churn}


def _ranked_values[L: Hashable](
    metrics: Mapping[str, DerivationalLikelihood[L]], unit: LikelihoodUnit
) -> dict[str, float]:
    """Return each model's aggregate under one unit, refusing an absent value.

    Parameters
    ----------
    metrics : Mapping[str, DerivationalLikelihood[L]]
        One metric per model.
    unit : LikelihoodUnit
        The unit to read.

    Returns
    -------
    dict[str, float]
        Each model's aggregate.

    Raises
    ------
    ValueError
        If a model has no aggregate to rank.
    """
    values: dict[str, float] = {}

    for name, metric in metrics.items():
        value = metric.result(unit).value

        if value is None:
            msg = (
                f"the model {name!r} has no held-out derivational likelihood "
                f"per {unit} to rank"
            )
            raise ValueError(msg)

        values[name] = value

    return values


def derivational_likelihood_report[L: Hashable](
    models: Mapping[str, DerivationalScoring[L]], /
) -> DerivationalLikelihoodReport:
    """Return every model's two likelihoods and the churn between the rankings.

    Parameters
    ----------
    models : Mapping[str, DerivationalScoring[L]]
        One scoring per model, each already accumulated over the held-out set.

    Returns
    -------
    DerivationalLikelihoodReport
        The rows in model order, and one churn per reporting unit.

    Raises
    ------
    ValueError
        If the mapping names no model, or a model has no aggregate to rank.
    """
    if not models:
        msg = "a derivational likelihood report was asked for no model"
        raise ValueError(msg)

    rows = tuple(row for name in sorted(models) for row in models[name].rows(name))
    constrained = {name: scoring.constrained for name, scoring in models.items()}
    unconstrained = {name: scoring.unconstrained for name, scoring in models.items()}
    churn = tuple(
        RankChurn(
            unit,
            _ranked_values(constrained, unit),
            _ranked_values(unconstrained, unit),
        )
        for unit in LIKELIHOOD_UNITS
    )

    return DerivationalLikelihoodReport(rows, churn)
