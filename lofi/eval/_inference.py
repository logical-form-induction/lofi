"""Classification metrics over declared labels."""

from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

from lofi._checks import ensure_name
from lofi.data import Categorical
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.eval._values import ConventionVector

__all__ = [
    "ENTAILMENT_ACCURACY_METRIC",
    "InferenceLabelMetric",
    "LabelReduction",
    "UndeclaredLabelError",
    "declared_labels",
]

ENTAILMENT_ACCURACY_METRIC: Final = "entailment-label-accuracy"
"""The metric name a label accuracy over sentence pairs is reported under."""

_MINIMUM_LABELS: Final = 2
"""The minimum label count for classification."""


class UndeclaredLabelError(ValueError):
    """Signals a label outside the declared inventory."""


class LabelReduction:
    """A coarsening map over a label inventory.

    Parameters
    ----------
    mapping : Mapping[str, str]
        Fine labels mapped to coarse classes.

    Raises
    ------
    TypeError
        If a label on either side is not a string.
    ValueError
        If labels are empty or the mapping is empty, trivial, or leaves fewer
        than two classes.
    """

    __slots__ = ("_images", "_mapping")

    def __init__(self, mapping: Mapping[str, str]) -> None:
        reduced = {
            ensure_name("a reduced label's source", source): ensure_name(
                f"the class {source!r} reduces to", image
            )
            for source, image in mapping.items()
        }

        if not reduced:
            msg = "a label reduction maps nothing"
            raise ValueError(msg)

        images = frozenset(reduced.values())

        if len(images) < _MINIMUM_LABELS:
            msg = (
                f"a label reduction leaves {len(images)} class and needs at "
                f"least {_MINIMUM_LABELS}"
            )
            raise ValueError(msg)

        if len(images) >= len(reduced):
            msg = (
                f"a label reduction of {len(reduced)} labels leaves "
                f"{len(images)} classes, so it reduces nothing"
            )
            raise ValueError(msg)

        self._mapping: Mapping[str, str] = MappingProxyType(reduced)
        self._images: tuple[str, ...] = tuple(sorted(images))

    @property
    def mapping(self) -> Mapping[str, str]:
        """Return the label mapping as a read-only view."""
        return self._mapping

    @property
    def domain(self) -> frozenset[str]:
        """Return the source label inventory."""
        return frozenset(self._mapping)

    @property
    def images(self) -> tuple[str, ...]:
        """Return reduced classes in ascending order."""
        return self._images

    def apply(self, label: str) -> str:
        """Return the class a label reduces to.

        Parameters
        ----------
        label : str
            The label to reduce.

        Returns
        -------
        str
            Reduced class.

        Raises
        ------
        UndeclaredLabelError
            If the reduction does not map the label.
        """
        image = self._mapping.get(label)

        if image is None:
            msg = (
                f"the label {label!r} is not one this reduction maps: "
                f"{sorted(self._mapping)}"
            )
            raise UndeclaredLabelError(msg)

        return image

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value maps the same labels the same way."""
        if type(other) is not LabelReduction:
            return NotImplemented

        return dict(self._mapping) == dict(other._mapping)

    @override
    def __hash__(self) -> int:
        """Return a hash over the map."""
        return hash((LabelReduction, tuple(sorted(self._mapping.items()))))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this reduction."""
        return f"LabelReduction({dict(self._mapping)!r})"


class _Tally(NamedTuple):
    """Stores one class's confusion counts.

    Parameters
    ----------
    matched : int
        True-positive count.
    predicted : int
        Predicted count.
    gold : int
        Gold count.
    """

    matched: int
    predicted: int
    gold: int


type _ScoredPair = tuple[str, str | None]
"""Gold label and a predicted label or explicit abstention."""


def _tallied(pairs: Sequence[_ScoredPair], label: str) -> _Tally:
    """Return one class's counts over the scored pairs.

    Parameters
    ----------
    pairs : Sequence[_ScoredPair]
        Gold-prediction label pairs.
    label : str
        The class to count.

    Returns
    -------
    _Tally
        Confusion counts for the class.
    """
    matched = 0
    predicted = 0
    gold = 0

    for gold_label, predicted_label in pairs:
        if predicted_label == label:
            predicted += 1

        if gold_label == label:
            gold += 1

            if predicted_label == label:
                matched += 1

    return _Tally(matched, predicted, gold)


def _precision(tally: _Tally) -> float | None:
    """Return precision for a class.

    Parameters
    ----------
    tally : _Tally
        The class's counts.

    Returns
    -------
    float | None
        Precision, or ``None`` when the predicted count is zero.
    """
    if tally.predicted == 0:
        return None

    return tally.matched / tally.predicted


def _recall(tally: _Tally) -> float | None:
    """Return recall for a class.

    Parameters
    ----------
    tally : _Tally
        The class's counts.

    Returns
    -------
    float | None
        Recall, or ``None`` when the gold count is zero.
    """
    if tally.gold == 0:
        return None

    return tally.matched / tally.gold


def _f_measure(tally: _Tally) -> float | None:
    """Return the harmonic mean of a class's precision and recall.

    Parameters
    ----------
    tally : _Tally
        The class's counts.

    Returns
    -------
    float | None
        F1, or ``None`` when the class appears on neither side.
    """
    seen = tally.predicted + tally.gold

    if seen == 0:
        return None

    return 2.0 * tally.matched / seen


def _macro_f1(pairs: Sequence[_ScoredPair], classes: Sequence[str]) -> float | None:
    """Return macro-F1 over observed classes.

    Parameters
    ----------
    pairs : Sequence[_ScoredPair]
        Gold-prediction label pairs.
    classes : Sequence[str]
        Classes in report order.

    Returns
    -------
    float | None
        Macro-F1, or ``None`` when no class appears.
    """
    measured = [
        measure
        for label in classes
        if (measure := _f_measure(_tallied(pairs, label))) is not None
    ]

    if not measured:
        return None

    return fmean(measured)


type InferenceMeasure = Literal[
    "accuracy", "macro-f1", "precision", "recall", "f1", "abstention"
]
"""One report row computed from an inference-label confusion table."""


class InferenceLabelMetric:
    """One reportable classification measure over a declared label inventory.

    Each report row is its own accumulator, so the evaluator draws every row's
    interval under the bootstrap its declaration names.

    Parameters
    ----------
    name : str
        The report row's name.
    labels : frozenset[str]
        The declared label inventory.
    measure : InferenceMeasure
        What the row computes from the confusion table.
    target : str | None
        The scored class of a per-class measure, and ``None`` otherwise.
    reduction : LabelReduction | None
        Optional coarsening of the inventory into scored classes.
    conventions : ConventionVector
        The conventions the row is scored under.

    Raises
    ------
    TypeError
        If a label is not a string or the reduction is not one.
    ValueError
        If fewer than two labels are declared, the reduction is not total over
        the inventory, or the target does not fit the measure.
    """

    __slots__ = (
        "_classes",
        "_conventions",
        "_items",
        "_labels",
        "_measure",
        "_name",
        "_reduction",
        "_scheme",
        "_seen",
        "_target",
    )

    def __init__(
        self,
        name: str,
        labels: frozenset[str],
        *,
        measure: InferenceMeasure,
        target: str | None,
        reduction: LabelReduction | None,
        conventions: ConventionVector,
    ) -> None:
        self._name = ensure_name("an inference metric name", name)
        self._measure = measure
        self._target = target
        self._conventions = conventions
        self._scheme = Resampling("inference-pair")
        self._reduction = reduction
        self._items: dict[ItemId, tuple[str, str | None]] = {}
        self._seen: dict[ItemId, tuple[str, str | None]] = {}
        if reduction is not None and not isinstance(reduction, LabelReduction):
            msg = f"an inference metric's reduction is not one: {reduction!r}"
            raise TypeError(msg)

        declared = frozenset(ensure_name("a declared label", label) for label in labels)

        if len(declared) < _MINIMUM_LABELS:
            msg = (
                f"an inference metric was declared over {len(declared)} "
                f"labels and needs at least {_MINIMUM_LABELS}"
            )
            raise ValueError(msg)

        if reduction is not None and reduction.domain != declared:
            msg = (
                f"a reduction over {sorted(reduction.domain)} is not total over "
                f"the declared inventory {sorted(declared)}"
            )
            raise ValueError(msg)

        self._labels = declared
        self._classes: tuple[str, ...] = (
            tuple(sorted(declared)) if reduction is None else reduction.images
        )

        if measure in ("precision", "recall", "f1"):
            if target not in self._classes:
                msg = (
                    f"the target {target!r} is not among the scored classes "
                    f"{self._classes!r}"
                )
                raise ValueError(msg)
        elif target is not None:
            msg = f"the aggregate measure {measure!r} has a class target {target!r}"
            raise ValueError(msg)

    @property
    def name(self) -> str:
        """Return the unique name of this report row."""
        return self._name

    @property
    def conventions(self) -> ConventionVector:
        """Return the conventions the row is scored under."""
        return self._conventions

    @property
    def order_invariant(self) -> bool:
        """Return whether update order affects the confusion table."""
        return True

    @property
    def labels(self) -> frozenset[str]:
        """Return the inventory the evaluation set declared."""
        return self._labels

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the classes scored, in the order they are reported."""
        return self._classes

    def reset(self) -> None:
        """Forget every item scored so far."""
        self._items = {}
        self._seen = {}

    def update(self, pred: str | None, gold: Categorical, *, key: ItemId) -> None:
        """Add one declared prediction-gold pair to the confusion table."""
        if not isinstance(gold, Categorical):
            msg = f"a gold inference response is not a categorical one: {gold!r}"
            raise TypeError(msg)
        if gold.label is None:
            msg = f"the item {key!r} has a missing gold inference label"
            raise ValueError(msg)
        if gold.label_set != self._labels:
            msg = (
                f"the item {key!r} declares the inventory "
                f"{sorted(gold.label_set)}, and this metric scores over "
                f"{sorted(self._labels)}"
            )
            raise UndeclaredLabelError(msg)
        if pred is not None and pred not in self._labels:
            msg = (
                f"the predicted label {pred!r} is not in the declared "
                f"inventory {sorted(self._labels)}"
            )
            raise UndeclaredLabelError(msg)

        pair = (gold.label, pred)
        standing = self._seen.get(key)
        if standing is not None and standing != pair:
            msg = f"the item {key!r} was already scored against a different pair"
            raise ValueError(msg)
        self._seen[key] = pair

        if pred is not None or self._conventions.coverage == "zero":
            self._items[key] = (
                self._reduced(pair[0]),
                None if pred is None else self._reduced(pred),
            )

    def compute(self) -> MetricResult:
        """Return this view of the accumulated confusion table."""
        pairs = tuple(pair for _, pair in sorted(self._items.items()))
        count = len(pairs)

        if self._measure == "abstention":
            total = len(self._seen)
            abstained = sum(prediction is None for _, prediction in self._seen.values())
            return MetricResult(
                None if total == 0 else abstained / total,
                total,
                self._scheme,
                components={
                    "abstained": float(abstained),
                    "total": float(total),
                },
            )

        if not pairs:
            return MetricResult(None, 0, self._scheme)

        if self._measure == "accuracy":
            correct = sum(gold == predicted for gold, predicted in pairs)
            return MetricResult(
                correct / count,
                count,
                self._scheme,
                components={"correct": float(correct), "scored": float(count)},
            )

        if self._measure == "macro-f1":
            measured = _macro_f1(pairs, self._classes)
            return MetricResult(
                measured,
                count,
                self._scheme,
                components={"classes": float(len(self._classes))},
            )

        target = self._target
        if target is None:
            msg = f"the per-class measure {self._measure!r} has no target"
            raise ValueError(msg)

        tally = _tallied(pairs, target)
        cell = {
            "precision": _precision,
            "recall": _recall,
            "f1": _f_measure,
        }[self._measure]
        support = sum(self._supports_target(pair) for pair in pairs)

        return MetricResult(
            cell(tally),
            support,
            self._scheme,
            components={
                "matched": float(tally.matched),
                "predicted": float(tally.predicted),
                "gold": float(tally.gold),
            },
        )

    def per_item(self) -> Mapping[ItemId, MetricResult]:
        """Return one defined resampling unit for every eligible pair."""
        if self._measure == "abstention":
            return MappingProxyType(
                {
                    key: MetricResult(float(prediction is None), 1, self._scheme)
                    for key, (_, prediction) in self._seen.items()
                }
            )

        return MappingProxyType(
            {
                key: MetricResult(
                    (
                        float(gold == self._target and prediction == self._target)
                        if self._supports_target((gold, prediction))
                        else None
                    ),
                    1,
                    self._scheme,
                )
                for key, (gold, prediction) in self._items.items()
            }
        )

    def strata(self) -> Mapping[ItemId, str]:
        """Return strata that keep a target cell defined in every resample."""
        target = self._target
        if target is None:
            items = self._seen if self._measure == "abstention" else self._items
            return MappingProxyType(dict.fromkeys(items, "all"))

        def stratum(pair: tuple[str, str | None]) -> str:
            return "target-support" if self._supports_target(pair) else "other"

        return MappingProxyType(
            {key: stratum(pair) for key, pair in self._items.items()}
        )

    def _supports_target(self, pair: tuple[str, str | None], /) -> bool:
        """Return whether one pair supplies a target cell's denominator."""
        target = self._target
        if target is None:
            return True
        gold, prediction = pair
        if self._measure == "precision":
            return prediction == target
        if self._measure == "recall":
            return gold == target
        return target in (gold, prediction)

    def _reduced(self, label: str, /) -> str:
        """Return one declared label on this metric's scored scale."""
        if self._reduction is None:
            return label

        return self._reduction.apply(label)


def declared_labels(responses: Iterable[Categorical]) -> frozenset[str]:
    """Return the shared gold-label inventory.

    Parameters
    ----------
    responses : Iterable[Categorical]
        Gold categorical responses.

    Returns
    -------
    frozenset[str]
        Shared label inventory.

    Raises
    ------
    TypeError
        If a response is not a categorical one.
    UndeclaredLabelError
        If the responses declare more than one inventory.
    ValueError
        If there are no responses to read an inventory off.
    """
    declared: set[frozenset[str]] = set()

    for response in responses:
        if not isinstance(response, Categorical):
            msg = f"a gold inference response is not a categorical one: {response!r}"
            raise TypeError(msg)

        declared.add(response.label_set)

    if not declared:
        msg = "no responses were given, so no inventory was declared"
        raise ValueError(msg)

    if len(declared) > 1:
        msg = (
            f"the responses declare {len(declared)} inventories: "
            f"{sorted(sorted(one) for one in declared)}"
        )
        raise UndeclaredLabelError(msg)

    return next(iter(declared))
