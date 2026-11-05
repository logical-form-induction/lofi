"""Premise-hypothesis pairs labeled by the oracle, and what a draw aims at.

A pair's label is the relation the premise bears to the hypothesis, decided
by two questions to the entailment oracle: whether the premise entails the
hypothesis, and whether it entails its negation. A proof of the first is
entailment, a proof of the second contradiction, and two refutations leave
the pair neutral. An abstention on either question leaves it undecided, which
is not a relation but the absence of one.
"""

from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import (
    ensure_finite,
    ensure_fraction,
    ensure_name,
    ensure_non_negative,
)
from lofi.data._sampling import SamplingExhaustedError
from lofi.data._values import ByteSpan, Item, Provenance, Token
from lofi.lf import RELATION_LABELS, InferenceLabel

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import ItemId
    from lofi.data.synthetic._complexity import ComplexityReport
    from lofi.data.synthetic._entailment import EntailmentCertificate, GoldForm
    from lofi.data.synthetic._semantics import SemanticReport
    from lofi.data.synthetic._values import SyntheticCorpus
    from lofi.terms import Term

__all__ = [
    "ConfusionReport",
    "FragmentEscapeError",
    "InferenceNoise",
    "InferencePair",
    "InferenceReport",
    "LabelConfusion",
    "LabelProportion",
    "NoisedLabel",
    "PairRelationUnreachableError",
    "PairRelations",
    "ProportionUnreachableError",
    "RelationReport",
    "SyntheticInference",
    "relation_complexity",
    "synthetic_inference_item",
]


class FragmentEscapeError(ValueError):
    """Raised when a drawn logical form quantifies outside the declared fragment.

    Parameters
    ----------
    text : str
        The surface rendering of the form that escaped.
    escaped : Iterable[str]
        The quantifier names the form uses that the fragment does not declare.
    declared : Iterable[str]
        The quantifier names the fragment declares.
    """

    __slots__ = ("declared", "escaped", "text")

    def __init__(
        self, text: str, escaped: Iterable[str], declared: Iterable[str], /
    ) -> None:
        self.text = text
        self.escaped = tuple(sorted(escaped))
        self.declared = tuple(sorted(declared))
        super().__init__(
            f"the form {text!r} quantifies with {list(self.escaped)!r}, which the "
            f"declared fragment {list(self.declared)!r} does not admit"
        )


_LABELS: Final = RELATION_LABELS
"""The canonical inference labels in report order."""


def _shares(
    field: str, values: Mapping[InferenceLabel, float]
) -> Mapping[InferenceLabel, float]:
    """Return a share per decided label, refusing anything else.

    Parameters
    ----------
    field : str
        What the mapping is, named in any error.
    values : Mapping[InferenceLabel, float]
        The share aimed at for each decided label.

    Returns
    -------
    Mapping[InferenceLabel, float]
        The shares in report order, every decided label present.

    Raises
    ------
    ValueError
        If a label is missing, one is named that is not decided, or a share
        lies outside the unit interval.
    """
    unknown = [
        label
        for label in values
        if not isinstance(label, InferenceLabel) or label not in _LABELS
    ]

    if unknown:
        msg = f"{field} names what is not a decided label: {unknown!r}"
        raise ValueError(msg)

    missing = [label for label in _LABELS if label not in values]

    if missing:
        msg = f"{field} does not name every decided label: {missing!r}"
        raise ValueError(msg)

    held = {}

    for label in _LABELS:
        share = ensure_finite(f"the share of {label}", values[label])

        if not 0.0 <= share <= 1.0:
            msg = f"the share of {label} is not between zero and one: {share!r}"
            raise ValueError(msg)

        held[label] = share

    return MappingProxyType(held)


def _row(
    source: InferenceLabel, values: Mapping[InferenceLabel, float]
) -> Mapping[InferenceLabel, float]:
    """Return the share of one relation's flips that each other relation takes.

    Parameters
    ----------
    source : InferenceLabel
        The relation being flipped, named in any error.
    values : Mapping[InferenceLabel, float]
        The share each destination takes, a label left out taking none.

    Returns
    -------
    Mapping[InferenceLabel, float]
        The shares in report order, every decided label present.

    Raises
    ------
    ValueError
        If a destination is named that is not decided, or a share lies
        outside the unit interval.
    """
    unknown = [
        label
        for label in values
        if not isinstance(label, InferenceLabel) or label not in _LABELS
    ]

    if unknown:
        msg = (
            f"the confusion of {source} names what is not a decided label: {unknown!r}"
        )
        raise ValueError(msg)

    held = {}

    for label in _LABELS:
        share = ensure_finite(
            f"the share of {source} observed as {label}", values.get(label, 0.0)
        )

        if not 0.0 <= share <= 1.0:
            msg = (
                f"the share of {source} observed as {label} is not between "
                f"zero and one: {share!r}"
            )
            raise ValueError(msg)

        held[label] = share

    return MappingProxyType(held)


def _counted(
    source: InferenceLabel, values: Mapping[InferenceLabel, int]
) -> Mapping[InferenceLabel, int]:
    """Return how many pairs of one oracle relation each relation was observed under.

    Parameters
    ----------
    source : InferenceLabel
        The oracle relation the counts belong to, named in any error.
    values : Mapping[InferenceLabel, int]
        The count under each observed relation, a label left out counting
        nothing.

    Returns
    -------
    Mapping[InferenceLabel, int]
        The counts in report order, every decided label present.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is keyed by what is not a decided label, or is negative.
    """
    unknown = [
        label
        for label in values
        if not isinstance(label, InferenceLabel) or label not in _LABELS
    ]

    if unknown:
        msg = (
            f"the pairs of {source} are counted under {unknown!r}, which is not decided"
        )
        raise ValueError(msg)

    return MappingProxyType(
        {
            label: ensure_non_negative(
                f"the count of {source} observed as {label}", values.get(label, 0)
            )
            for label in _LABELS
        }
    )


def _elsewhere(source: InferenceLabel, row: Mapping[InferenceLabel, float]) -> float:
    """Return the share of a relation's flips that land on another relation.

    Parameters
    ----------
    source : InferenceLabel
        The relation being flipped.
    row : Mapping[InferenceLabel, float]
        The share each relation takes of its flips.
    """
    return sum(share for target, share in row.items() if target != source)


@final
class LabelProportion:
    """The share of the decided pairs each relation is aimed at.

    Parameters
    ----------
    shares : Mapping[InferenceLabel, float]
        The share of the decided pairs each of the three relations is aimed
        at, summing to one as the declaration that builds them requires.
    tolerance : float
        How far an achieved share may fall from the one aimed at.

    Raises
    ------
    ValueError
        If the shares do not cover the three relations or a share lies
        outside the unit interval, or the tolerance is negative or not
        finite.
    """

    __slots__ = ("_shares", "_tolerance")

    def __init__(
        self, shares: Mapping[InferenceLabel, float], *, tolerance: float
    ) -> None:
        self._shares = _shares("a label proportion", shares)
        self._tolerance = ensure_finite("a proportion tolerance", tolerance)

        if self._tolerance < 0:
            msg = f"a proportion tolerance is negative: {tolerance!r}"
            raise ValueError(msg)

    @property
    def shares(self) -> Mapping[InferenceLabel, float]:
        """Return the share of the decided pairs each relation is aimed at."""
        return self._shares

    @property
    def tolerance(self) -> float:
        """Return how far an achieved share may fall from the one aimed at."""
        return self._tolerance

    def quotas(self, count: int, /) -> Mapping[InferenceLabel, int]:
        """Return how many of a run's pairs each relation is to fill.

        The shares are apportioned by largest remainder, so the quotas sum to
        the requested count however the shares round.

        Parameters
        ----------
        count : int
            How many pairs the run draws.

        Returns
        -------
        Mapping[InferenceLabel, int]
            The quota of each decided label, in report order.

        Raises
        ------
        TypeError
            If the count is not an integer.
        ValueError
            If the count is negative.
        """
        held = ensure_non_negative("a pair count", count)
        exact = {label: self._shares[label] * held for label in _LABELS}
        quotas = {label: int(value) for label, value in exact.items()}
        order = sorted(
            _LABELS,
            key=lambda label: (quotas[label] - exact[label], label),
        )

        for label in order[: held - sum(quotas.values())]:
            quotas[label] += 1

        return MappingProxyType({label: quotas[label] for label in _LABELS})

    def admits(self, achieved: Mapping[InferenceLabel, float], /) -> bool:
        """Return whether every achieved share lies within tolerance of its target.

        Parameters
        ----------
        achieved : Mapping[InferenceLabel, float]
            The share of the decided pairs each relation reached.

        Returns
        -------
        bool
            Whether every decided label's share is within tolerance.
        """
        return all(
            abs(achieved.get(label, 0.0) - self._shares[label]) <= self._tolerance
            for label in _LABELS
        )

    def _fields(self) -> tuple[tuple[tuple[InferenceLabel, float], ...], float]:
        """Return the fields the proportion is equal on."""
        return (tuple(self._shares.items()), self._tolerance)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another proportion aims at the same shares."""
        if type(other) is not LabelProportion:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the shares and the tolerance."""
        return hash((LabelProportion, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this proportion."""
        return f"LabelProportion({dict(self._shares)!r}, tolerance={self._tolerance!r})"


class PairRelationUnreachableError(SamplingExhaustedError):
    """Raised when the relation policy refuses every candidate pair within budget."""


class ProportionUnreachableError(ValueError):
    """Raised when a declared label proportion cannot be filled from a candidate set.

    Parameters
    ----------
    requested : LabelProportion
        The proportion the draw aimed at.
    achieved : Mapping[InferenceLabel, int]
        How many pairs of each label the draw reached before its budget ran
        out.
    attempts : int
        The candidate budget each quota was given.
    reached : Mapping[InferenceLabel, int]
        How many candidates within the complexity bands bore each relation
        while the quota that ran out was being filled.
    """

    __slots__ = ("achieved", "attempts", "reached", "requested", "undecided")

    def __init__(
        self,
        requested: LabelProportion,
        achieved: Mapping[InferenceLabel, int],
        /,
        *,
        attempts: int,
        reached: Mapping[InferenceLabel, int],
        undecided: int,
    ) -> None:
        self.requested = requested
        self.achieved = MappingProxyType(dict(achieved))
        self.attempts = attempts
        self.reached = MappingProxyType(dict(reached))
        self.undecided = ensure_non_negative("an undecided candidate count", undecided)
        written = ", ".join(f"{label}={achieved.get(label, 0)}" for label in _LABELS)
        drawn = ", ".join(f"{label}={reached.get(label, 0)}" for label in _LABELS)
        super().__init__(
            f"the declared proportion {dict(requested.shares)!r} is unreachable "
            f"within {attempts} candidates per label; the draw achieved {written}, "
            f"and the last quota's candidates reached {drawn}, with "
            f"undecided={self.undecided}"
        )


@final
class LabelConfusion:
    """Where a flipped relation lands, as a share of its flips per destination.

    Parameters
    ----------
    rows : Mapping[InferenceLabel, Mapping[InferenceLabel, float]]
        The share of each decided label's flips that every decided label
        takes. A label with no row, or with a row summing to nothing, is
        never flipped; any other row sums to one. A row may keep a share on
        its own label, and a flip landing there leaves the relation as the
        oracle reached it, so the share a row keeps lowers that relation's
        flip rate below the noise rate. The declaration that builds the rows
        requires the sums.

    Raises
    ------
    ValueError
        If a row or a share is keyed by what is not a decided label, or a
        share lies outside the unit interval.
    """

    __slots__ = ("_rows",)

    def __init__(
        self, rows: Mapping[InferenceLabel, Mapping[InferenceLabel, float]]
    ) -> None:
        unknown = [
            label
            for label in rows
            if not isinstance(label, InferenceLabel) or label not in _LABELS
        ]

        if unknown:
            msg = f"a confusion has a row for what is not a decided label: {unknown!r}"
            raise ValueError(msg)

        self._rows = MappingProxyType(
            {label: _row(label, rows.get(label, {})) for label in _LABELS}
        )

    @classmethod
    def uniform(cls) -> LabelConfusion:
        """Return the confusion spreading a label's flips evenly over the others.

        Returns
        -------
        LabelConfusion
            The structure under which a flipped label is equally likely to be
            observed as either of the two it is not.
        """
        share = 1.0 / (len(_LABELS) - 1)

        return cls(
            {
                source: {target: share for target in _LABELS if target != source}
                for source in _LABELS
            }
        )

    @property
    def rows(self) -> Mapping[InferenceLabel, Mapping[InferenceLabel, float]]:
        """Return each decided label's destinations, in report order."""
        return self._rows

    def confuses(self, label: InferenceLabel, /) -> bool:
        """Return whether a relation is ever observed as another.

        Parameters
        ----------
        label : InferenceLabel
            The relation the oracle reached.

        Returns
        -------
        bool
            Whether the structure sends any share of the relation's flips to
            another relation, which an undecided label, a label with an empty
            row and a label whose row keeps all of its share do not.
        """
        if not isinstance(label, InferenceLabel) or label not in _LABELS:
            msg = f"an inference label is an InferenceLabel, not {label!r}"
            raise TypeError(msg)

        row = self._rows.get(label)

        return row is not None and _elsewhere(label, row) > 0.0

    def kept(self, label: InferenceLabel, /) -> float:
        """Return the share of a relation's flips that land on the relation itself.

        Parameters
        ----------
        label : InferenceLabel
            The relation the oracle reached.
        """
        if not isinstance(label, InferenceLabel) or label not in _LABELS:
            msg = f"an inference label is an InferenceLabel, not {label!r}"
            raise TypeError(msg)

        row = self._rows.get(label)

        return 0.0 if row is None else row[label]

    def lands(self, label: InferenceLabel, draw: float, /) -> InferenceLabel:
        """Return where a flip of a relation lands at a point of the unit interval.

        The destinations partition the interval in report order, each taking
        a stretch the width of its share.

        Parameters
        ----------
        label : InferenceLabel
            The relation the oracle reached.
        draw : float
            The point of the unit interval the flip is drawn at.

        Returns
        -------
        InferenceLabel
            The destination whose stretch holds the point, which is the
            relation itself where the row keeps a share there. A point beyond
            every stretch, which rounding in the shares can leave, lands on
            the last destination the relation has.

        Raises
        ------
        ValueError
            If the relation is never confused, or the point lies outside the
            unit interval.
        """
        if not self.confuses(label):
            msg = f"a flip of {label!r} lands nowhere, since it is never confused"
            raise ValueError(msg)

        if not 0.0 <= ensure_finite("a confusion draw", draw) < 1.0:
            msg = f"a confusion draw is not in the unit interval: {draw!r}"
            raise ValueError(msg)

        row = self._rows[label]
        running = 0.0

        for target in _LABELS:
            running += row[target]

            if draw < running:
                return target

        return next(target for target in reversed(_LABELS) if row[target] > 0.0)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another structure routes every flip the same way."""
        if type(other) is not LabelConfusion:
            return NotImplemented

        return self._fields() == other._fields()

    def _fields(self) -> tuple[tuple[InferenceLabel, tuple[float, ...]], ...]:
        """Return the fields the structure is equal on."""
        return tuple(
            (source, tuple(self._rows[source][target] for target in _LABELS))
            for source in _LABELS
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the rows."""
        return hash((LabelConfusion, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this structure."""
        written = {source: dict(row) for source, row in self._rows.items()}

        return f"LabelConfusion({written!r})"


@final
class InferenceNoise:
    """The rates oracle relations are flipped at and where the flips land.

    Parameters
    ----------
    rates : Mapping[InferenceLabel, float]
        The probability that a pair is flipped, by oracle relation. A relation
        left out has rate zero. A flip lands where the structure sends it,
        which may be the relation itself when its row keeps a share there.
    confusion : LabelConfusion
        Where a flip lands, as a share of each relation's flips.
    tolerance : float
        How far an achieved confusion matrix entry may fall from the declared
        one.

    Raises
    ------
    ValueError
        If a rate is keyed by what is not a decided relation, lies outside the
        unit interval, or the tolerance is negative or not finite.
    """

    __slots__ = ("_confusion", "_rates", "_tolerance")

    def __init__(
        self,
        rates: Mapping[InferenceLabel, float],
        *,
        confusion: LabelConfusion,
        tolerance: float,
    ) -> None:
        unknown = [
            label
            for label in rates
            if not isinstance(label, InferenceLabel) or label not in _LABELS
        ]
        if unknown:
            msg = f"inference-noise rates name undecided relations: {unknown!r}"
            raise ValueError(msg)
        held: dict[InferenceLabel, float] = {}
        for label in _LABELS:
            rate = ensure_finite(
                f"the inference-noise rate of {label}", rates.get(label, 0.0)
            )
            if not 0.0 <= rate <= 1.0:
                msg = (
                    f"the inference-noise rate of {label} is not between zero "
                    f"and one: {rate!r}"
                )
                raise ValueError(msg)
            held[label] = rate
        self._rates = MappingProxyType(held)

        self._tolerance = ensure_finite("a confusion tolerance", tolerance)

        if self._tolerance < 0:
            msg = f"a confusion tolerance is negative: {tolerance!r}"
            raise ValueError(msg)

        self._confusion = confusion

    @property
    def rates(self) -> Mapping[InferenceLabel, float]:
        """Return each decided relation's probability of being flipped."""
        return self._rates

    def rate(self, label: InferenceLabel, /) -> float:
        """Return the probability that one decided relation is flipped."""
        if not isinstance(label, InferenceLabel) or label not in _LABELS:
            msg = f"an inference label is an InferenceLabel, not {label!r}"
            raise TypeError(msg)
        return self._rates[label]

    @property
    def confusion(self) -> LabelConfusion:
        """Return where a flip lands."""
        return self._confusion

    @property
    def tolerance(self) -> float:
        """Return how far an achieved matrix entry may fall from the declared one."""
        return self._tolerance

    def matrix(self) -> Mapping[InferenceLabel, Mapping[InferenceLabel, float]]:
        """Return the declared chance of each observed relation given each oracle one.

        Returns
        -------
        Mapping[InferenceLabel, Mapping[InferenceLabel, float]]
            A row per decided label summing to one: the rate spread over the
            other relations the structure sends the label to, and what is
            left on the label itself. A label the structure never confuses
            keeps all of its mass whatever the rate.
        """
        rows = self._confusion.rows

        return MappingProxyType(
            {
                source: MappingProxyType(
                    {
                        target: (
                            1.0 - self._rates[source] * _elsewhere(source, rows[source])
                            if target == source
                            else self._rates[source] * rows[source][target]
                        )
                        for target in _LABELS
                    }
                )
                for source in _LABELS
            }
        )

    def admits(
        self, achieved: Mapping[InferenceLabel, Mapping[InferenceLabel, float]], /
    ) -> bool:
        """Return whether every achieved row lies within tolerance of the declared one.

        Parameters
        ----------
        achieved : Mapping[InferenceLabel, Mapping[InferenceLabel, float]]
            The share of each observed relation the pairs of each oracle
            relation reached. A relation the mapping leaves out was borne by
            no pair and measures nothing, so it is admitted.

        Returns
        -------
        bool
            Whether every entry of every measured row is within tolerance.
        """
        declared = self.matrix()

        return all(
            abs(row.get(target, 0.0) - declared[source][target]) <= self._tolerance
            for source, row in achieved.items()
            for target in _LABELS
        )

    def _fields(
        self,
    ) -> tuple[tuple[tuple[InferenceLabel, float], ...], LabelConfusion, float]:
        """Return the fields the request is equal on."""
        return (tuple(self._rates.items()), self._confusion, self._tolerance)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another request flips at the same rates the same way."""
        if type(other) is not InferenceNoise:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the rates, the structure and the tolerance."""
        return hash((InferenceNoise, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this request."""
        return (
            f"InferenceNoise({dict(self._rates)!r}, confusion={self._confusion!r}, "
            f"tolerance={self._tolerance!r})"
        )


@final
class NoisedLabel:
    """One pair's oracle relation beside the relation emitted for it.

    Parameters
    ----------
    oracle : InferenceLabel | None
        The relation the oracle reached, which the record keeps whatever is
        emitted.
    observed : InferenceLabel | None
        The relation a consumer is shown.

    Raises
    ------
    ValueError
        If either value is neither an inference label nor ``None``, or one is
        undecided while the other bears a relation. An undecided pair has no
        relation to flip.
    """

    __slots__ = ("_observed", "_oracle")

    def __init__(
        self,
        oracle: InferenceLabel | None,
        observed: InferenceLabel | None,
        /,
    ) -> None:
        for field, label in (("oracle", oracle), ("observed", observed)):
            if label is not None and (
                not isinstance(label, InferenceLabel) or label not in _LABELS
            ):
                msg = f"an {field} label is an InferenceLabel or None, not {label!r}"
                raise TypeError(msg)

        if (oracle is None) != (observed is None):
            msg = (
                f"a pair the oracle labeled {oracle!r} is observed as "
                f"{observed!r}, but absence of a decision is not confused"
            )
            raise ValueError(msg)

        self._oracle = oracle
        self._observed = observed

    @classmethod
    def of(cls, oracle: InferenceLabel | None, /) -> NoisedLabel:
        """Return the record of an oracle relation emitted as it stands.

        Parameters
        ----------
        oracle : InferenceLabel | None
            The relation the oracle reached.

        Returns
        -------
        NoisedLabel
            The record whose observed relation is the oracle's.
        """
        return cls(oracle, oracle)

    @property
    def oracle(self) -> InferenceLabel | None:
        """Return the relation the oracle reached."""
        return self._oracle

    @property
    def observed(self) -> InferenceLabel | None:
        """Return the relation a consumer is shown."""
        return self._observed

    @property
    def flipped(self) -> bool:
        """Return whether the emitted relation differs from the oracle's."""
        return self._oracle != self._observed

    def _fields(
        self,
    ) -> tuple[InferenceLabel | None, InferenceLabel | None]:
        """Return the fields the record is equal on."""
        return (self._oracle, self._observed)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another record holds the same two relations."""
        if type(other) is not NoisedLabel:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the two relations."""
        return hash((NoisedLabel, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this record."""
        return f"NoisedLabel({self._oracle!r}, {self._observed!r})"


@final
class ConfusionReport:
    """What flipping a run's relations requested and what it achieved.

    Parameters
    ----------
    requested : InferenceNoise
        The rate the relations were flipped at and where the flips were to
        land.
    counts : Mapping[InferenceLabel, Mapping[InferenceLabel, int]]
        How many pairs of each oracle relation are observed under each
        relation.
    undecided : int
        How many pairs the oracle reached no relation on, which no flip
        touches.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is keyed by what is not a decided label, or is negative.
    """

    __slots__ = ("_counts", "_requested", "_undecided")

    def __init__(
        self,
        requested: InferenceNoise,
        *,
        counts: Mapping[InferenceLabel, Mapping[InferenceLabel, int]],
        undecided: int,
    ) -> None:
        unknown = [
            label
            for label in counts
            if not isinstance(label, InferenceLabel) or label not in _LABELS
        ]

        if unknown:
            msg = f"a confusion count names what is not a decided label: {unknown!r}"
            raise ValueError(msg)

        self._requested = requested
        self._counts = MappingProxyType(
            {source: _counted(source, counts.get(source, {})) for source in _LABELS}
        )
        self._undecided = ensure_non_negative("undecided", undecided)

    @classmethod
    def of(
        cls, requested: InferenceNoise, labels: Iterable[NoisedLabel]
    ) -> ConfusionReport:
        """Return the report measured over the emitted label records.

        Parameters
        ----------
        requested : InferenceNoise
            The rate the relations were flipped at and where the flips were
            to land.
        labels : Iterable[NoisedLabel]
            The records, one per pair.

        Returns
        -------
        ConfusionReport
            The oracle relation against the observed one, counted pair by
            pair, with the undecided pairs counted apart.
        """
        counts: dict[InferenceLabel, Counter[InferenceLabel]] = {
            source: Counter() for source in _LABELS
        }
        undecided = 0

        for record in labels:
            oracle = record.oracle
            observed = record.observed

            if oracle is None:
                undecided += 1
            elif observed is None:
                msg = "a decided oracle relation has no observed relation"
                raise ValueError(msg)
            else:
                counts[oracle][observed] += 1

        return cls(requested, counts=counts, undecided=undecided)

    @property
    def requested(self) -> InferenceNoise:
        """Return the rate the relations were flipped at and where they land."""
        return self._requested

    @property
    def counts(self) -> Mapping[InferenceLabel, Mapping[InferenceLabel, int]]:
        """Return the oracle relation against the observed one, in report order."""
        return self._counts

    @property
    def undecided(self) -> int:
        """Return how many pairs bore no relation to flip."""
        return self._undecided

    @property
    def decided(self) -> int:
        """Return how many pairs the oracle reached a relation on."""
        return sum(sum(row.values()) for row in self._counts.values())

    @property
    def flip_count(self) -> int:
        """Return how many pairs are emitted under another relation."""
        return sum(
            count
            for source, row in self._counts.items()
            for target, count in row.items()
            if target != source
        )

    @property
    def achieved_rate(self) -> float:
        """Return the share of the decided pairs flipped, zero when none was."""
        decided = self.decided

        return self.flip_count / decided if decided else 0.0

    def matrix(self) -> Mapping[InferenceLabel, Mapping[InferenceLabel, float]]:
        """Return the achieved share of each observed relation given each oracle one.

        Returns
        -------
        Mapping[InferenceLabel, Mapping[InferenceLabel, float]]
            A row per decided relation some pair bears, each summing to one.
            A relation no pair bears is left out, since no pair measures it.
        """
        measured = {}

        for source in _LABELS:
            row = self._counts[source]
            borne = sum(row.values())

            if borne:
                measured[source] = MappingProxyType(
                    {target: row[target] / borne for target in _LABELS}
                )

        return MappingProxyType(measured)

    def satisfied(self) -> bool:
        """Return whether the achieved matrix lies within the declared tolerance.

        Returns
        -------
        bool
            Whether every entry of every measured row is within tolerance of
            the declared matrix.
        """
        return self._requested.admits(self.matrix())

    def _fields(
        self,
    ) -> tuple[InferenceNoise, tuple[tuple[InferenceLabel, tuple[int, ...]], ...], int]:
        """Return the fields the report is equal on."""
        return (
            self._requested,
            tuple(
                (
                    source,
                    tuple(self._counts[source][target] for target in _LABELS),
                )
                for source in _LABELS
            ),
            self._undecided,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report counts the same flips for one request."""
        if type(other) is not ConfusionReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the request and the counts."""
        return hash((ConfusionReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        written = {source: dict(row) for source, row in self._counts.items()}

        return (
            f"ConfusionReport({self._requested!r}, counts={written!r}, "
            f"undecided={self._undecided!r})"
        )


def _subterms(term: Term, /) -> frozenset[Term]:
    """Return every subterm of a term, the term included.

    Parameters
    ----------
    term : Term
        The term to take apart.
    """
    seen: set[Term] = set()
    pending = [term]

    while pending:
        current = pending.pop()

        if current not in seen:
            seen.add(current)
            pending.extend(current.args)

    return frozenset(seen)


def relation_complexity(premise: Term, hypothesis: Term, /) -> int:
    """Return how far apart the logical forms of a pair are.

    Parameters
    ----------
    premise : Term
        The premise's closed logical form.
    hypothesis : Term
        The hypothesis's closed logical form.

    Returns
    -------
    int
        How many subterms one form has that the other does not. Bound
        variables are nameless in a store, so two forms that differ only in
        the names of their bound variables are one term and lie zero apart,
        and every other pair lies at least one apart.
    """
    return len(_subterms(premise) ^ _subterms(hypothesis))


@final
class PairRelations:
    """Which premise-hypothesis relations a draw admits beside their labels.

    A pair is trivial when its two logical forms are one term, which covers a
    reflexive pair and a pair of alphabetic variants.

    Parameters
    ----------
    trivial : float
        The share of the drawn pairs that are trivial. Zero excludes them.
    entailment : int
        The least relation complexity an entailing pair may have.

    Raises
    ------
    TypeError
        If the share is not a number or the complexity is not an integer.
    ValueError
        If the share is outside the unit interval or the complexity is
        negative.
    """

    __slots__ = ("_entailment", "_trivial")

    def __init__(self, *, trivial: float, entailment: int) -> None:
        self._trivial = ensure_fraction("a trivial-pair share", trivial)

        self._entailment = ensure_non_negative(
            "a least entailment complexity", entailment
        )

    @property
    def trivial(self) -> float:
        """Return the share of the drawn pairs that are trivial."""
        return self._trivial

    @property
    def entailment(self) -> int:
        """Return the least relation complexity an entailing pair may have."""
        return self._entailment

    def quota(self, count: int, /) -> int:
        """Return how many of a run's pairs are trivial.

        Parameters
        ----------
        count : int
            How many pairs the run holds.

        Returns
        -------
        int
            The declared share of the count, taken to the nearest pair.
        """
        return round(self._trivial * ensure_non_negative("a pair count", count))

    def admits(self, label: InferenceLabel | None, complexity: int, /) -> bool:
        """Return whether a labeled pair is complex enough for its label.

        Parameters
        ----------
        label : InferenceLabel | None
            The relation the oracle reached on the pair.
        complexity : int
            The pair's relation complexity.
        """
        return label is not InferenceLabel.ENTAILS or complexity >= self._entailment

    def _fields(self) -> tuple[float, int]:
        """Return the fields the policy is equal on."""
        return (self._trivial, self._entailment)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another policy admits the same relations."""
        if type(other) is not PairRelations:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the share and the complexity."""
        return hash((PairRelations, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        return (
            f"PairRelations(trivial={self._trivial!r}, entailment={self._entailment!r})"
        )


@final
class RelationReport:
    """The trivial pairs a run holds and the candidates its relation policy refused.

    Parameters
    ----------
    requested : PairRelations
        The policy the draw was made under.
    pairs : int
        How many pairs the run holds.
    trivial : int
        How many of them are trivial.
    screened : int
        How many candidate pairs the policy refused and drew again: trivial
        pairs beyond the quota, other pairs where only trivial ones can still
        fill it, and entailing pairs below the least complexity.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative or more pairs are trivial than the run holds.
    """

    __slots__ = ("_pairs", "_requested", "_screened", "_trivial")

    def __init__(
        self, requested: PairRelations, *, pairs: int, trivial: int, screened: int
    ) -> None:
        self._requested = requested
        self._pairs = ensure_non_negative("a pair count", pairs)
        self._trivial = ensure_non_negative("a trivial-pair count", trivial)
        self._screened = ensure_non_negative("a screened-pair count", screened)

        if self._trivial > self._pairs:
            msg = f"{self._trivial} of a run of {self._pairs} pairs are trivial"
            raise ValueError(msg)

    @property
    def requested(self) -> PairRelations:
        """Return the policy the draw was made under."""
        return self._requested

    @property
    def pairs(self) -> int:
        """Return how many pairs the run holds."""
        return self._pairs

    @property
    def trivial(self) -> int:
        """Return how many pairs are trivial."""
        return self._trivial

    @property
    def screened(self) -> int:
        """Return how many candidates the policy refused and drew again."""
        return self._screened

    def satisfied(self) -> bool:
        """Return whether the run holds the declared share of trivial pairs."""
        return self._trivial == self._requested.quota(self._pairs)

    def _fields(self) -> tuple[PairRelations, int, int, int]:
        """Return the fields the report is equal on."""
        return (self._requested, self._pairs, self._trivial, self._screened)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report counted the same pairs the same way."""
        if type(other) is not RelationReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the policy and the counts."""
        return hash((RelationReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"RelationReport({self._requested!r}, pairs={self._pairs!r}, "
            f"trivial={self._trivial!r}, screened={self._screened!r})"
        )


@final
class InferencePair:
    """One premise-hypothesis pair, its label, and the forms it was decided on.

    Parameters
    ----------
    premise : GoldForm
        The premise item's identifier beside its logical form.
    hypothesis : GoldForm
        The hypothesis item's identifier beside its logical form.
    label : InferenceLabel | None
        The relation the oracle reached, or ``None`` when it reached none.
    entailment : EntailmentCertificate
        The decision of whether the premise entails the hypothesis.
    contradiction : EntailmentCertificate | None
        The decision of whether the premise entails the hypothesis's negation,
        or nothing when the first decision settled the pair.

    Raises
    ------
    ValueError
        If the label is not one of the declared labels.
    """

    __slots__ = (
        "_contradiction",
        "_entailment",
        "_hypothesis",
        "_label",
        "_premise",
        "_split",
    )

    def __init__(
        self,
        premise: GoldForm,
        hypothesis: GoldForm,
        /,
        *,
        label: InferenceLabel | None,
        entailment: EntailmentCertificate,
        contradiction: EntailmentCertificate | None = None,
        split: str | None = None,
    ) -> None:
        if label is not None and (
            not isinstance(label, InferenceLabel) or label not in _LABELS
        ):
            msg = f"an inference label is an InferenceLabel or None, not {label!r}"
            raise TypeError(msg)

        self._premise = premise
        self._hypothesis = hypothesis
        self._label = label
        self._entailment = entailment
        self._contradiction = contradiction
        self._split = (
            None if split is None else ensure_name("an inference split", split)
        )

    @property
    def premise(self) -> GoldForm:
        """Return the premise item beside its retained logical form."""
        return self._premise

    @property
    def hypothesis(self) -> GoldForm:
        """Return the hypothesis item beside its retained logical form."""
        return self._hypothesis

    @property
    def label(self) -> InferenceLabel | None:
        """Return the relation the oracle reached."""
        return self._label

    @property
    def entailment(self) -> EntailmentCertificate:
        """Return the decision of whether the premise entails the hypothesis."""
        return self._entailment

    @property
    def contradiction(self) -> EntailmentCertificate | None:
        """Return the decision on the negated hypothesis, when one was taken."""
        return self._contradiction

    @property
    def split(self) -> str | None:
        """Return the exact declared partition containing this pair."""
        return self._split

    def _fields(
        self,
    ) -> tuple[
        GoldForm,
        GoldForm,
        InferenceLabel | None,
        EntailmentCertificate,
        EntailmentCertificate | None,
        str | None,
    ]:
        """Return the fields the pair is equal on."""
        return (
            self._premise,
            self._hypothesis,
            self._label,
            self._entailment,
            self._contradiction,
            self._split,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another pair holds the same forms and decisions."""
        if type(other) is not InferencePair:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the forms, the label and the certificates."""
        return hash((InferencePair, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this pair."""
        return (
            f"InferencePair({self._premise!r}, {self._hypothesis!r}, "
            f"label={self._label!r}, entailment={self._entailment!r}, "
            f"contradiction={self._contradiction!r}, split={self._split!r})"
        )


@final
class InferenceReport:
    """The labels a run of pairs reached against the proportion it aimed at.

    Parameters
    ----------
    labels : Mapping[InferenceLabel, int]
        How many pairs each label holds.
    requested : LabelProportion | None
        The proportion the draw aimed at, or nothing when it aimed at none.
    undecided : int
        How many pairs the oracle reached no relation on.
    rejected : int
        How many candidate pairs bore a label the draw had no quota left for
        and were drawn again.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative, or a label is named that is not a label.
    """

    __slots__ = ("_labels", "_rejected", "_requested", "_undecided")

    def __init__(
        self,
        labels: Mapping[InferenceLabel, int],
        *,
        undecided: int,
        requested: LabelProportion | None,
        rejected: int,
    ) -> None:
        unknown = [
            label
            for label in labels
            if not isinstance(label, InferenceLabel) or label not in _LABELS
        ]

        if unknown:
            msg = f"a label count names what is not a label: {unknown!r}"
            raise ValueError(msg)

        self._labels = MappingProxyType(
            {
                label: ensure_non_negative(
                    f"the count of {label}", labels.get(label, 0)
                )
                for label in _LABELS
            }
        )
        self._undecided = ensure_non_negative("undecided", undecided)
        self._requested = requested
        self._rejected = ensure_non_negative("rejected", rejected)

    @property
    def labels(self) -> Mapping[InferenceLabel, int]:
        """Return how many pairs each label holds, in report order."""
        return self._labels

    @property
    def requested(self) -> LabelProportion | None:
        """Return the proportion the draw aimed at, when it aimed at one."""
        return self._requested

    @property
    def rejected(self) -> int:
        """Return how many candidates bore an unwanted label and were redrawn."""
        return self._rejected

    @property
    def undecided(self) -> int:
        """Return how many pairs bore no decided relation."""
        return self._undecided

    @property
    def total(self) -> int:
        """Return how many pairs the run holds."""
        return sum(self._labels.values()) + self._undecided

    @property
    def decided(self) -> int:
        """Return how many pairs the oracle reached a relation on."""
        return sum(self._labels[label] for label in _LABELS)

    def proportions(self) -> Mapping[InferenceLabel, float]:
        """Return each relation's share of the decided pairs.

        Returns
        -------
        Mapping[InferenceLabel, float]
            The share of the decided pairs each decided label holds, in report
            order, every share zero when no pair was decided.
        """
        decided = self.decided

        if decided == 0:
            return MappingProxyType(dict.fromkeys(_LABELS, 0.0))

        return MappingProxyType(
            {label: self._labels[label] / decided for label in _LABELS}
        )

    def satisfied(self) -> bool:
        """Return whether the achieved shares lie within the declared tolerance.

        Returns
        -------
        bool
            True when no proportion was declared, and otherwise whether every
            relation's achieved share is within tolerance of its target.
        """
        requested = self._requested

        return requested is None or requested.admits(self.proportions())

    def _fields(
        self,
    ) -> tuple[
        tuple[tuple[InferenceLabel, int], ...],
        int,
        LabelProportion | None,
        int,
    ]:
        """Return the fields the report is equal on."""
        return (
            tuple(self._labels.items()),
            self._undecided,
            self._requested,
            self._rejected,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report counted the same labels."""
        if type(other) is not InferenceReport:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts, the request and the rejections."""
        return hash((InferenceReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"InferenceReport({dict(self._labels)!r}, "
            f"undecided={self._undecided!r}, requested={self._requested!r}, "
            f"rejected={self._rejected!r})"
        )


@final
class SyntheticInference:
    """The pairs one inference draw produced, beside everything it achieved.

    Parameters
    ----------
    seed : int
        The seed the draw was made with. Equal seeds draw equal runs.
    pairs : tuple[InferencePair, ...]
        The labeled pairs, in draw order, each holding both logical forms.
    labels : tuple[NoisedLabel, ...]
        One record per pair, in draw order: the relation the oracle reached
        beside the relation emitted for it.
    noise : ConfusionReport | None
        What flipping the relations requested and achieved, or nothing when
        no noise was requested, in which case no relation is flipped.
    premises : SyntheticCorpus
        The sample the premises were drawn from, with the token counts, the
        readings and the refused forms its strings achieved.
    hypotheses : SyntheticCorpus
        The sample the hypotheses were drawn from, with the same reports.
    complexity : ComplexityReport
        Each complexity control's achieved distribution on each side, and the
        achieved predicate overlap.
    inference : InferenceReport
        The labels the pairs reached against the proportion aimed at.
    relations : RelationReport
        The trivial pairs the run holds and the candidates its relation
        policy refused.
    semantic : SemanticReport | None
        The achieved semantic inventory and token-frequency joint measured
        across both sides, or nothing for a fixed lexical interpretation.

    Raises
    ------
    TypeError
        If the seed is not an integer.
    ValueError
        If the two samples were drawn from different grammars, a pair names a
        premise or a hypothesis its sample holds no form for, the records do
        not cover the pairs one for one, a record's oracle relation disagrees
        with the pair it stands for, a relation is flipped without a
        confusion report, or the confusion report's counts are not those of
        the records.
    """

    __slots__ = (
        "_complexity",
        "_hypotheses",
        "_inference",
        "_labels",
        "_noise",
        "_pairs",
        "_premises",
        "_relations",
        "_seed",
        "_semantic",
    )

    def __init__(
        self,
        *,
        seed: int,
        pairs: tuple[InferencePair, ...],
        labels: tuple[NoisedLabel, ...],
        noise: ConfusionReport | None,
        premises: SyntheticCorpus,
        hypotheses: SyntheticCorpus,
        complexity: ComplexityReport,
        inference: InferenceReport,
        relations: RelationReport,
        semantic: SemanticReport | None,
    ) -> None:
        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = f"a seed is not an integer: {seed!r}"
            raise TypeError(msg)

        if premises.declaration != hypotheses.declaration:
            msg = "the premises and the hypotheses were drawn from different grammars"
            raise ValueError(msg)

        for pair in pairs:
            for side, sample in (
                (pair.premise, premises),
                (pair.hypothesis, hypotheses),
            ):
                if sample.forms.get(side.item_id) != side:
                    msg = f"pair item {side.item_id!r} bears no such form in its sample"
                    raise ValueError(msg)

        if len(labels) != len(pairs):
            msg = f"a run of {len(pairs)} pairs holds {len(labels)} label records"
            raise ValueError(msg)

        disagreeing = [
            index
            for index, (record, pair) in enumerate(zip(labels, pairs, strict=True))
            if record.oracle != pair.label
        ]

        if disagreeing:
            msg = (
                f"the records at {disagreeing!r} name another relation than the "
                f"oracle reached on the pairs they stand for"
            )
            raise ValueError(msg)

        if noise is None:
            flipped = sum(1 for record in labels if record.flipped)

            if flipped:
                msg = f"{flipped} relations are flipped without a confusion report"
                raise ValueError(msg)
        elif noise != ConfusionReport.of(noise.requested, labels):
            msg = (
                f"the confusion report holds {noise!r} but the records measure "
                f"{ConfusionReport.of(noise.requested, labels)!r}"
            )
            raise ValueError(msg)

        self._seed = seed
        self._pairs = pairs
        self._labels = labels
        self._noise = noise
        self._premises = premises
        self._hypotheses = hypotheses
        self._complexity = complexity
        self._inference = inference
        self._relations = relations
        self._semantic = semantic

    @property
    def declaration(self) -> bytes:
        """Return the canonical bytes of the grammar the two sides were drawn from."""
        return self._premises.declaration

    @property
    def seed(self) -> int:
        """Return the seed the draw was made with."""
        return self._seed

    @property
    def pairs(self) -> tuple[InferencePair, ...]:
        """Return the labeled pairs, in draw order."""
        return self._pairs

    @property
    def labels(self) -> tuple[NoisedLabel, ...]:
        """Return one record per pair, in draw order."""
        return self._labels

    @property
    def noise(self) -> ConfusionReport | None:
        """Return what flipping the relations achieved, when any was requested."""
        return self._noise

    @property
    def premises(self) -> SyntheticCorpus:
        """Return the sample the premises were drawn from."""
        return self._premises

    @property
    def hypotheses(self) -> SyntheticCorpus:
        """Return the sample the hypotheses were drawn from."""
        return self._hypotheses

    @property
    def complexity(self) -> ComplexityReport:
        """Return the complexity the pairs achieved against what was aimed at."""
        return self._complexity

    @property
    def inference(self) -> InferenceReport:
        """Return the labels the pairs reached against the proportion aimed at."""
        return self._inference

    @property
    def relations(self) -> RelationReport:
        """Return the trivial pairs held and the candidates the policy refused."""
        return self._relations

    @property
    def semantic(self) -> SemanticReport | None:
        """Return the joint achieved semantic report, when one was generated."""
        return self._semantic

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the seed, the pairs and the two samples."""
        return (
            f"SyntheticInference(seed={self._seed!r}, pairs={len(self._pairs)}, "
            f"premises={self._premises!r}, hypotheses={self._hypotheses!r}, "
            f"inference={self._inference!r})"
        )


def synthetic_inference_item(
    item_id: ItemId,
    text: str,
    /,
    *,
    split: str | None = None,
    provenance: Provenance | None = None,
) -> Item:
    """Tokenize one canonical synthetic-inference rendering.

    Synthetic inference strings contain nonempty surface forms separated by
    exactly one ASCII space. The returned tokens carry exact UTF-8 byte spans
    into the unchanged input text. Text outside that fragment is refused
    rather than normalized into another representation.
    """
    forms = text.split(" ")

    malformed = any(
        not form or any(character.isspace() for character in form) for form in forms
    )

    if not text or malformed:
        msg = (
            "synthetic inference text has nonempty surface forms separated "
            f"by one ASCII space: {text!r}"
        )
        raise ValueError(msg)

    tokens = []
    start = 0

    for position, form in enumerate(forms):
        end = start + len(form.encode())
        tokens.append(Token(position, form, ByteSpan(start, end)))
        start = end + 1

    return Item(
        item_id,
        tuple(tokens),
        text=text,
        split=split,
        provenance=provenance,
    )
