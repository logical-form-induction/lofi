"""Matched-recall and homogeneity metrics."""

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.eval._alignment import Contingency
from lofi.eval._bracketing import (
    GoldBracketing,
    _boundaries,
    _is_trivial,
    _reindexed,
    surviving_gold,
)
from lofi.eval._clustering import homogeneity, v_measure
from lofi.eval._values import ConventionVector, MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import ItemId
    from lofi.eval._alignment import Split

__all__ = ["MATCHED_ROWS", "InducedBracketing", "RecallHomogeneity"]


MATCHED_ROWS: tuple[str, ...] = (
    "unlabeled-recall",
    "homogeneity",
    "v-measure",
    "recall-homogeneity",
    "recall-v-measure",
)
"""Metric rows in report order."""


class InducedBracketing:
    """Spans labeled by an induced grammar.

    Parameters
    ----------
    length : int
        The sentence length in tokens.
    spans : Mapping[tuple[int, int], str]
        The induced label for each half-open span.

    Raises
    ------
    TypeError
        If a span is not a pair or a label is not a string.
    ValueError
        If the length, a span, or a label is invalid.
    """

    __slots__ = ("_length", "_spans")

    def __init__(self, length: int, spans: Mapping[tuple[int, int], str]) -> None:
        self._length = ensure_non_negative("an induced bracketing's length", length)
        self._spans: Mapping[tuple[int, int], str] = MappingProxyType(
            {
                self._checked(span): ensure_name("an induced label", label)
                for span, label in sorted(spans.items())
            }
        )

    def _checked(self, span: tuple[int, int]) -> tuple[int, int]:
        """Return a valid span.

        Parameters
        ----------
        span : tuple[int, int]
            The half-open span to validate.

        Returns
        -------
        tuple[int, int]
            The validated span.

        Raises
        ------
        TypeError
            If the span is not a pair of integers.
        ValueError
            If the span is empty, negative, or outside the sentence.
        """
        try:
            start, end = span

        except (TypeError, ValueError) as error:
            msg = f"an induced span is not a start and an end: {span!r}"
            raise TypeError(msg) from error

        low = ensure_non_negative("an induced span's start", start)
        high = ensure_non_negative("an induced span's end", end)

        if high <= low:
            msg = f"an induced span covers no token: [{start}, {end})"
            raise ValueError(msg)

        if high > self._length:
            msg = (
                f"an induced span runs past a sentence of {self._length} "
                f"tokens: [{start}, {end})"
            )
            raise ValueError(msg)

        return (low, high)

    @property
    def length(self) -> int:
        """Return how many tokens the sentence has."""
        return self._length

    @property
    def spans(self) -> Mapping[tuple[int, int], str]:
        """Return the induced label of each span, as a read-only view in span order."""
        return self._spans

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the unique induced labels in ascending order."""
        return tuple(sorted(set(self._spans.values())))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value labels the same spans the same way."""
        if type(other) is not InducedBracketing:
            return NotImplemented

        return self._length == other._length and dict(self._spans) == dict(other._spans)

    @override
    def __hash__(self) -> int:
        """Return a hash over the length and the labeled spans."""
        return hash((InducedBracketing, self._length, tuple(self._spans.items())))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this bracketing."""
        return f"InducedBracketing({self._length!r}, {dict(self._spans)!r})"


class _Matched(NamedTuple):
    """Stores a sentence's contributions to both products.

    Parameters
    ----------
    gold : int
        The number of surviving gold constituents.
    pairs : tuple[tuple[str, str], ...]
        The induced and gold labels of matched constituents.
    """

    gold: int
    pairs: tuple[tuple[str, str], ...]


class _Item(NamedTuple):
    """Stores one update's metric state.

    Parameters
    ----------
    pred : InducedBracketing | None
        The predicted bracketing, if available.
    gold : GoldBracketing
        The gold bracketing.
    scored : _Matched | None
        The retained contribution, if the sentence was scored.
    """

    pred: InducedBracketing | None
    gold: GoldBracketing
    scored: _Matched | None


def _labeled_spans(
    pred: InducedBracketing,
    boundaries: tuple[int, ...],
    length: int,
    conventions: ConventionVector,
) -> dict[tuple[int, int], str]:
    """Return labeled predicted spans after applying conventions.

    Parameters
    ----------
    pred : InducedBracketing
        The predicted labeled bracketing.
    boundaries : tuple[int, ...]
        Kept-token counts at the original token boundaries.
    length : int
        The number of retained tokens.
    conventions : ConventionVector
        The scoring conventions to apply.

    Returns
    -------
    dict[tuple[int, int], str]
        Labels for surviving spans; the wider original wins after a collision.
    """
    kept: dict[tuple[int, int], tuple[int, str]] = {}

    for (start, end), label in pred.spans.items():
        span = _reindexed(boundaries, start, end)

        if span is None:
            continue

        if conventions.trivial_spans == "drop" and _is_trivial(span, length):
            continue

        standing = kept.get(span)

        if standing is None or end - start > standing[0]:
            kept[span] = (end - start, label)

    return {span: label for span, (_, label) in kept.items()}


class RecallHomogeneity:
    """Computes matched recall and homogeneity with separate factors.

    Parameters
    ----------
    conventions : ConventionVector
        The scoring conventions.
    split : Split
        The data split recorded by the contingency table.
    seed : int | None
        The optional run seed.
    beta : float
        The completeness weight for V-measure.

    Raises
    ------
    TypeError
        If an argument has an invalid type.
    ValueError
        If the seed or beta value is invalid.
    """

    __slots__ = (
        "_beta",
        "_conventions",
        "_gold_units",
        "_items",
        "_matched_units",
        "_sentences",
        "_split",
    )

    def __init__(
        self,
        conventions: ConventionVector,
        *,
        split: Split,
        seed: int | None = None,
        beta: float = 1.0,
    ) -> None:
        if not isinstance(conventions, ConventionVector):
            msg = (
                f"a matched-set metric's conventions are not a vector: {conventions!r}"
            )
            raise TypeError(msg)

        weight = ensure_finite("a v-measure's beta", beta)

        if weight < 0.0:
            msg = f"a v-measure's beta is negative: {beta!r}"
            raise ValueError(msg)

        self._conventions = conventions
        self._split: Split = split
        self._beta = weight
        self._sentences = Resampling("sentence", seed)
        self._gold_units = Resampling("gold constituent", seed)
        self._matched_units = Resampling("matched constituent", seed)
        self._items: dict[ItemId, _Item] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return "recall-homogeneity"

    @property
    def conventions(self) -> ConventionVector:
        """Return the scoring conventions."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the aggregate."""
        return True

    @property
    def split(self) -> Split:
        """Return the split the matched-set contingency records."""
        return self._split

    @property
    def beta(self) -> float:
        """Return how much completeness weighs against homogeneity in the V-measure."""
        return self._beta

    def reset(self) -> None:
        """Forget every sentence scored so far."""
        self._items = {}

    def update(
        self, pred: InducedBracketing | None, gold: GoldBracketing, *, key: ItemId
    ) -> None:
        """Score one labeled bracketing against its gold analysis.

        Parameters
        ----------
        pred : InducedBracketing | None
            The predicted bracketing, or ``None`` for a failed analysis.
        gold : GoldBracketing
            The gold bracketing.
        key : ItemId
            The item identifier.

        Raises
        ------
        TypeError
            If either bracketing has an invalid type.
        ValueError
            If lengths differ or the key identifies a conflicting update.
        """
        if not isinstance(gold, GoldBracketing):
            msg = f"a gold bracketing is not one: {gold!r}"
            raise TypeError(msg)

        if pred is not None:
            if not isinstance(pred, InducedBracketing):
                msg = f"a predicted labeled bracketing is not one: {pred!r}"
                raise TypeError(msg)

            if pred.length != gold.length:
                msg = (
                    f"a prediction of {pred.length} tokens was scored against a "
                    f"gold analysis of {gold.length}"
                )
                raise ValueError(msg)

        standing = self._items.get(key)

        if standing is not None:
            if (standing.pred, standing.gold) != (pred, gold):
                msg = (
                    f"the item {key!r} was already scored against a different analysis"
                )
                raise ValueError(msg)

            return

        self._items[key] = self._scored(pred, gold)

    def compute(self) -> MetricResult:
        """Return the recall-homogeneity product over all scored items.

        Returns
        -------
        MetricResult
            The corpus-level recall-homogeneity result.
        """
        return self.rows()["recall-homogeneity"]

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return recall-homogeneity for each sentence.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            Results keyed by item identifier.
        """
        return MappingProxyType(
            {key: self._sentence_result(item) for key, item in self._items.items()}
        )

    def rows(self) -> Mapping[str, MetricResult]:
        """Return separate factors and products.

        Returns
        -------
        Mapping[str, MetricResult]
            Results in ``MATCHED_ROWS`` order.
        """
        scored = self._surviving()
        table = self._contingency()
        gold = sum(entry.gold for entry in scored)
        matched = table.total
        recall = None if gold == 0 else matched / gold
        pure = homogeneity(table, unit=self._matched_units.unit).result
        mixed = v_measure(table, unit=self._matched_units.unit, beta=self._beta).result
        factors = {
            "unlabeled-recall": MetricResult(
                recall,
                gold,
                self._gold_units,
                components={
                    "matched": float(matched),
                    "gold": float(gold),
                },
            ),
            "homogeneity": pure,
            "v-measure": mixed,
        }
        products = {
            "recall-homogeneity": self._product(recall, pure.value, scored, factors),
            "recall-v-measure": self._product(recall, mixed.value, scored, factors),
        }

        return MappingProxyType(
            {name: (factors | products)[name] for name in MATCHED_ROWS}
        )

    def contingency(self) -> Contingency:
        """Return induced-against-gold counts for matched constituents.

        Returns
        -------
        Contingency
            The matched-constituent contingency table.
        """
        return self._contingency()

    def _product(
        self,
        recall: float | None,
        factor: float | None,
        scored: tuple[_Matched, ...],
        factors: Mapping[str, MetricResult],
    ) -> MetricResult:
        """Return a product with its factors and counts.

        Parameters
        ----------
        recall : float | None
            The unlabeled recall, if defined.
        factor : float | None
            The clustering factor, if defined.
        scored : tuple[_Matched, ...]
            Contributions from scored sentences.
        factors : Mapping[str, MetricResult]
            Results for the component factors.

        Returns
        -------
        MetricResult
            The product and its component counts.
        """
        uncovered = sum(1 for item in self._items.values() if item.pred is None)
        recalled = factors["unlabeled-recall"]
        pure = factors["homogeneity"].value
        mixed = factors["v-measure"].value
        components = {
            "matched": recalled.components["matched"],
            "gold": recalled.components["gold"],
            "scored": float(len(scored)),
            "dropped": float(len(self._items) - len(scored)),
            "uncovered": float(uncovered),
        }

        if recall is not None:
            components["unlabeled_recall"] = recall

        if pure is not None:
            components["homogeneity"] = pure

            if recall is not None:
                components["recall_homogeneity"] = recall * pure

        if mixed is not None:
            components["v_measure"] = mixed

            if recall is not None:
                components["recall_v_measure"] = recall * mixed

        return MetricResult(
            None if recall is None or factor is None else recall * factor,
            len(scored),
            self._sentences,
            components=components,
        )

    def _sentence_result(self, item: _Item) -> MetricResult:
        """Return recall-homogeneity for one sentence.

        Parameters
        ----------
        item : _Item
            The stored sentence state.

        Returns
        -------
        MetricResult
            The sentence-level result.
        """
        scored = item.scored

        if scored is None or scored.gold == 0:
            return MetricResult(None, 0, self._gold_units)

        table = Contingency(self._counted((scored,)), split=self._split)
        factor = homogeneity(table, unit=self._matched_units.unit).result.value
        recall = len(scored.pairs) / scored.gold

        return MetricResult(
            None if factor is None else recall * factor,
            scored.gold,
            self._gold_units,
            components={
                "matched": float(len(scored.pairs)),
                "gold": float(scored.gold),
                "unlabeled_recall": recall,
            },
        )

    def _contingency(self) -> Contingency:
        """Return the matched constituent set of every scored sentence."""
        return Contingency(self._counted(self._surviving()), split=self._split)

    def _counted(self, scored: Iterable[_Matched]) -> dict[tuple[str, str], int]:
        """Return counts for matched induced and gold label pairs.

        Parameters
        ----------
        scored : Iterable[_Matched]
            The sentence contributions to count.

        Returns
        -------
        dict[tuple[str, str], int]
            Counts keyed by induced and gold label.
        """
        counts: dict[tuple[str, str], int] = {}

        for entry in scored:
            for pair in entry.pairs:
                counts[pair] = counts.get(pair, 0) + 1

        return counts

    def _surviving(self) -> tuple[_Matched, ...]:
        """Return sentence contributions in ascending key order."""
        return tuple(
            item.scored
            for _, item in sorted(self._items.items())
            if item.scored is not None
        )

    def _scored(self, pred: InducedBracketing | None, gold: GoldBracketing) -> _Item:
        """Return a sentence's contribution under the metric conventions.

        Parameters
        ----------
        pred : InducedBracketing | None
            The predicted bracketing, if available.
        gold : GoldBracketing
            The gold bracketing.

        Returns
        -------
        _Item
            The stored update state and retained contribution.
        """
        conventions = self._conventions
        removed = (
            gold.punctuation if conventions.punctuation == "strip" else frozenset()
        )
        boundaries = _boundaries(gold.length, removed)
        length = boundaries[gold.length]

        if length < conventions.minimum_length:
            return _Item(pred=pred, gold=gold, scored=None)

        surviving = surviving_gold(gold, boundaries, length, conventions)

        if not surviving:
            return _Item(pred=pred, gold=gold, scored=None)

        if pred is None and conventions.coverage == "ignore":
            return _Item(pred=pred, gold=gold, scored=None)

        proposed = (
            {}
            if pred is None
            else _labeled_spans(pred, boundaries, length, conventions)
        )
        taken: set[tuple[int, int]] = set()
        pairs: list[tuple[str, str]] = []

        for span, label, _ in surviving:
            induced = proposed.get(span)

            if induced is None or span in taken:
                continue

            taken.add(span)
            pairs.append((induced, label))

        return _Item(
            pred=pred,
            gold=gold,
            scored=_Matched(gold=len(surviving), pairs=tuple(pairs)),
        )
