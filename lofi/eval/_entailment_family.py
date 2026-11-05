"""The entailment-classifier baseline family, scored by label accuracy.

A pair the classifier abstained on counts as wrong under a zero coverage
penalty and leaves the score under an ignored one, and both sentences of a
scored pair reach the declared minimum length.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple

from lofi._checks import ensure_name, ensure_non_negative
from lofi._types import ItemId
from lofi.eval._baselines import (
    BaselineFamily,
)
from lofi.eval._inference import ENTAILMENT_ACCURACY_METRIC, UndeclaredLabelError
from lofi.eval._ports import HYPOTHESIS_ONLY
from lofi.eval._values import ConventionVector, MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping


__all__ = ["EntailmentAccuracy", "EntailmentPair", "entailment_family"]


class EntailmentPair(NamedTuple):
    """One sentence pair's gold label and the length of each member.

    Parameters
    ----------
    label : str
        The gold label, from the declared inventory.
    premise_length : int
        How many tokens the premise holds.
    hypothesis_length : int
        How many tokens the hypothesis holds.
    """

    label: str
    premise_length: int
    hypothesis_length: int


class _Labeled(NamedTuple):
    """One stored entailment item."""

    pred: str | None
    gold: EntailmentPair


class EntailmentAccuracy:
    """Accuracy of predicted entailment labels over a declared inventory.

    Under a coverage penalty of zero a pair the classifier abstained on counts
    as wrong and stays in the denominator; under a penalty of ignore it leaves
    the score. A pair is scored only when both of its sentences reach the
    minimum length, since a pair is as short as its shorter member.

    Attributes
    ----------
    name : str
        The metric name carried on the report row.
    conventions : ConventionVector
        The conventions accuracy is computed under.
    labels : frozenset[str]
        The declared label inventory.
    order_invariant : bool
        Whether the value is independent of update order, which it is.

    Parameters
    ----------
    conventions : ConventionVector
        The conventions accuracy is computed under.
    labels : frozenset[str]
        The declared label inventory.
    seed : int | None
        The run seed recorded in the result.

    Raises
    ------
    TypeError
        If the conventions are not a vector or a label is not text.
    ValueError
        If a label is empty, fewer than two labels are declared, or the seed
        is negative.
    """

    __slots__ = ("_conventions", "_items", "_labels", "_scheme")

    def __init__(
        self,
        conventions: ConventionVector,
        labels: frozenset[str],
        *,
        seed: int | None = None,
    ) -> None:
        if not isinstance(conventions, ConventionVector):
            msg = f"entailment conventions are not a vector: {conventions!r}"
            raise TypeError(msg)

        declared = frozenset(
            ensure_name("a declared entailment label", label) for label in labels
        )

        if len(declared) < 2:
            msg = (
                f"an entailment metric was declared over {len(declared)} "
                f"labels and classifies nothing"
            )
            raise ValueError(msg)

        self._conventions = conventions
        self._labels = declared
        self._scheme = Resampling("pair", seed)
        self._items: dict[ItemId, _Labeled] = {}

    @property
    def name(self) -> str:
        """Return the metric name."""
        return ENTAILMENT_ACCURACY_METRIC

    @property
    def required_control(self) -> str:
        """Return the artifact control required beside this accuracy."""
        return HYPOTHESIS_ONLY

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions accuracy is computed under."""
        return self._conventions

    @property
    def labels(self) -> frozenset[str]:
        """Return the declared label inventory."""
        return self._labels

    @property
    def order_invariant(self) -> bool:
        """Return whether rows are invariant to update order."""
        return True

    def reset(self) -> None:
        """Forget every pair scored so far."""
        self._items = {}

    def update(self, pred: str | None, gold: EntailmentPair, *, key: ItemId) -> None:
        """Store one predicted label against its gold label.

        Parameters
        ----------
        pred : str | None
            The label the classifier chose, or None when it abstained.
        gold : EntailmentPair
            The gold label and the two sentence lengths.
        key : ItemId
            The item identifier.

        Raises
        ------
        TypeError
            If the gold value is not a pair, or the prediction is neither text
            nor absent.
        UndeclaredLabelError
            If either label lies outside the declared inventory.
        ValueError
            If a length is negative or the key was already scored against
            another pair.
        """
        if not isinstance(gold, EntailmentPair):
            msg = f"a gold entailment pair is not one: {gold!r}"
            raise TypeError(msg)

        if pred is not None and not isinstance(pred, str):
            msg = f"a predicted entailment label is not text: {pred!r}"
            raise TypeError(msg)

        for label in (pred, gold.label):
            if label is not None and label not in self._labels:
                msg = (
                    f"the label {label!r} is not in the declared inventory "
                    f"{sorted(self._labels)}"
                )
                raise UndeclaredLabelError(msg)

        item = _Labeled(
            pred,
            EntailmentPair(
                gold.label,
                ensure_non_negative("a premise length", gold.premise_length),
                ensure_non_negative("a hypothesis length", gold.hypothesis_length),
            ),
        )
        identifier = ItemId(ensure_name("an entailment item id", key))
        standing = self._items.get(identifier)

        if standing is not None and standing != item:
            msg = f"item {identifier!r} was already scored against another pair"
            raise ValueError(msg)

        self._items[identifier] = item

    def compute(self) -> MetricResult:
        """Return accuracy over every admitted pair.

        Returns
        -------
        MetricResult
            Accuracy with its matched and labeled counts beside it, absent
            when no pair is admitted.
        """
        admitted = self._admitted()

        if not admitted:
            return MetricResult(None, 0, self._scheme)

        matched = sum(entry.pred == entry.gold.label for entry in admitted)
        labeled = sum(entry.pred is not None for entry in admitted)

        return MetricResult(
            matched / len(admitted),
            len(admitted),
            self._scheme,
            components={"matched": float(matched), "labeled": float(labeled)},
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return each pair's own accuracy, keyed by its item id.

        Returns
        -------
        Mapping[ItemId, MetricResult]
            One entry per update key. A pair the conventions do not admit has
            no value.
        """
        return MappingProxyType(
            {
                key: MetricResult(float(item.pred == item.gold.label), 1, self._scheme)
                if self._admits(item)
                else MetricResult(None, 0, self._scheme)
                for key, item in self._items.items()
            }
        )

    def _admits(self, item: _Labeled) -> bool:
        """Return whether the conventions score a stored pair.

        Parameters
        ----------
        item : _Labeled
            The stored pair.
        """
        conventions = self._conventions

        return (
            min(item.gold.premise_length, item.gold.hypothesis_length)
            >= conventions.minimum_length
            and item.gold.premise_length > 0
            and item.gold.hypothesis_length > 0
            and (conventions.coverage == "zero" or item.pred is not None)
        )

    def _admitted(self) -> tuple[_Labeled, ...]:
        """Return the stored pairs the conventions score, in key order."""
        return tuple(
            item for _, item in sorted(self._items.items()) if self._admits(item)
        )


def entailment_family(
    *,
    labels: frozenset[str],
    conventions: ConventionVector,
    seed: int | None = None,
) -> BaselineFamily[str | None, EntailmentPair]:
    """Return the entailment-classifier family, scored by label accuracy.

    The baseline reads a label off the two sentences' representations rather
    than off the distribution over logical forms, which is what makes it the
    comparison the inference tasks are measured against.

    Parameters
    ----------
    labels : frozenset[str]
        The declared label inventory.
    conventions : ConventionVector
        The conventions the family reports under.
    seed : int | None
        The run seed recorded in each result.

    Returns
    -------
    BaselineFamily[str | None, EntailmentPair]
        The family, whose items pair a predicted label with a gold pair and
        carry None where the classifier abstained.
    """
    return BaselineFamily(
        "nli-entailment", EntailmentAccuracy(conventions, labels, seed=seed)
    )
