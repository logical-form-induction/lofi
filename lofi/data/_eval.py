"""Evaluation-data joins by item identifier."""

from typing import TYPE_CHECKING, override

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi._types import ItemId
    from lofi.data._values import Item

__all__ = ["Annotation", "EvalDataset", "Join", "Pairing"]


class Annotation[G]:
    """A gold annotation.

    Parameters
    ----------
    item_id : ItemId
        Annotated item identifier.
    value : G
        Annotation value.
    annotator : str | None
        Annotator identifier, if recorded.

    Raises
    ------
    ValueError
        If the id is empty, or if the annotator is named by an empty string.
    """

    __slots__ = ("_annotator", "_item_id", "_value")

    def __init__(self, item_id: ItemId, value: G, annotator: str | None = None) -> None:
        if not item_id:
            msg = "an annotation has no item id"
            raise ValueError(msg)

        if annotator is not None and not annotator:
            msg = f"the annotation of item {item_id!r} names an empty annotator"
            raise ValueError(msg)

        self._item_id = item_id
        self._value = value
        self._annotator = annotator

    @property
    def item_id(self) -> ItemId:
        """Return the item the response was given to."""
        return self._item_id

    @property
    def value(self) -> G:
        """Return the response itself."""
        return self._value

    @property
    def annotator(self) -> str | None:
        """Return who gave the response, if the corpus names them."""
        return self._annotator

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is the same annotation."""
        if type(other) is not Annotation:
            return NotImplemented

        return (
            self._item_id == other._item_id
            and self._value == other._value
            and self._annotator == other._annotator
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the id, the response and the annotator."""
        return hash((Annotation, self._item_id, self._value, self._annotator))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this annotation."""
        return f"Annotation({self._item_id!r}, {self._value!r}, {self._annotator!r})"


class Pairing[G, P]:
    """A matched evaluation item.

    Parameters
    ----------
    item : Item
        Evaluated item.
    gold : Annotation[G]
        Gold annotation.
    prediction : P
        Predicted value.

    Raises
    ------
    ValueError
        If the annotation references a different item.
    """

    __slots__ = ("_gold", "_item", "_prediction")

    def __init__(self, item: Item, gold: Annotation[G], prediction: P) -> None:
        if gold.item_id != item.item_id:
            msg = (
                f"item {item.item_id!r} is paired with an annotation of "
                f"item {gold.item_id!r}"
            )
            raise ValueError(msg)

        self._item = item
        self._gold = gold
        self._prediction = prediction

    @property
    def item_id(self) -> ItemId:
        """Return the id both sides were joined on."""
        return self._item.item_id

    @property
    def item(self) -> Item:
        """Return the item both sides speak about."""
        return self._item

    @property
    def gold(self) -> Annotation[G]:
        """Return the response recorded against the item."""
        return self._gold

    @property
    def prediction(self) -> P:
        """Return what the run produced for the item."""
        return self._prediction

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value pairs the same item, gold and prediction."""
        if type(other) is not Pairing:
            return NotImplemented

        return (
            self._item == other._item
            and self._gold == other._gold
            and self._prediction == other._prediction
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the item, the gold response and the prediction."""
        return hash((Pairing, self._item, self._gold, self._prediction))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this pairing."""
        return f"Pairing({self._item!r}, {self._gold!r}, {self._prediction!r})"


class Join[G, P]:
    """Matched and unmatched evaluation records.

    Parameters
    ----------
    pairs : tuple[Pairing[G, P], ...]
        Matched records in dataset order.
    unpredicted : tuple[ItemId, ...]
        Dataset identifiers without predictions.
    unmatched : tuple[ItemId, ...]
        Prediction identifiers absent from the dataset.
    """

    __slots__ = ("_pairs", "_unmatched", "_unpredicted")

    def __init__(
        self,
        pairs: tuple[Pairing[G, P], ...],
        unpredicted: tuple[ItemId, ...],
        unmatched: tuple[ItemId, ...],
    ) -> None:
        self._pairs = pairs
        self._unpredicted = unpredicted
        self._unmatched = unmatched

    @property
    def pairs(self) -> tuple[Pairing[G, P], ...]:
        """Return the items present on both sides, in the set's order."""
        return self._pairs

    @property
    def unpredicted(self) -> tuple[ItemId, ...]:
        """Return the ids of items no prediction covered."""
        return self._unpredicted

    @property
    def unmatched(self) -> tuple[ItemId, ...]:
        """Return the ids in the predictions that the set does not have."""
        return self._unmatched

    @override
    def __repr__(self) -> str:
        """Return a summary naming how many pairs and how many of each remainder."""
        return (
            f"Join({len(self._pairs)} paired, {len(self._unpredicted)} unpredicted, "
            f"{len(self._unmatched)} unmatched)"
        )


class EvalDataset[G]:
    """Items and gold responses keyed by item identifier.

    Parameters
    ----------
    items : Iterable[Item]
        Items in evaluation order.
    annotations : Iterable[Annotation[G]]
        Gold annotations.

    Raises
    ------
    ValueError
        If item identifiers repeat or annotations are missing, duplicated, or
        reference an unknown item.
    """

    __slots__ = ("_annotations", "_items")

    def __init__(
        self, items: Iterable[Item], annotations: Iterable[Annotation[G]]
    ) -> None:
        by_id: dict[ItemId, Item] = {}

        for item in items:
            if item.item_id in by_id:
                msg = f"two items share the id {item.item_id!r}"
                raise ValueError(msg)

            by_id[item.item_id] = item

        gold: dict[ItemId, Annotation[G]] = {}

        for annotation in annotations:
            if annotation.item_id not in by_id:
                msg = (
                    f"an annotation names item {annotation.item_id!r}, "
                    f"which the set does not contain"
                )
                raise ValueError(msg)

            if annotation.item_id in gold:
                msg = f"item {annotation.item_id!r} has two annotations"
                raise ValueError(msg)

            gold[annotation.item_id] = annotation

        for item_id in by_id:
            if item_id not in gold:
                msg = f"item {item_id!r} has no annotation"
                raise ValueError(msg)

        self._items = by_id
        self._annotations = gold

    @property
    def ids(self) -> tuple[ItemId, ...]:
        """Return the ids in the set, in the order it was built in."""
        return tuple(self._items)

    @property
    def items(self) -> tuple[Item, ...]:
        """Return the items, in the order the set was built in."""
        return tuple(self._items.values())

    @property
    def annotations(self) -> tuple[Annotation[G], ...]:
        """Return the gold responses, in the order their items were built in."""
        return tuple(self._annotations[item_id] for item_id in self._items)

    def item(self, item_id: ItemId) -> Item:
        """Return an item by identifier.

        Parameters
        ----------
        item_id : ItemId
            The id to look up.

        Returns
        -------
        Item
            The requested item.

        Raises
        ------
        KeyError
            If the set has no such item.
        """
        return self._items[item_id]

    def annotation(self, item_id: ItemId) -> Annotation[G]:
        """Return a gold annotation by item identifier.

        Parameters
        ----------
        item_id : ItemId
            The id to look up.

        Returns
        -------
        Annotation[G]
            The requested annotation.

        Raises
        ------
        KeyError
            If the set has no such item.
        """
        return self._annotations[item_id]

    def join[P](self, predictions: Mapping[ItemId, P]) -> Join[G, P]:
        """Join gold annotations and predictions by item identifier.

        Parameters
        ----------
        predictions : Mapping[ItemId, P]
            Predictions keyed by item identifier.

        Returns
        -------
        Join[G, P]
            Matched pairs and unmatched identifiers from both sides.
        """
        pairs: list[Pairing[G, P]] = []
        unpredicted: list[ItemId] = []

        for item_id, item in self._items.items():
            if item_id in predictions:
                gold = self._annotations[item_id]
                pairs.append(Pairing(item, gold, predictions[item_id]))
            else:
                unpredicted.append(item_id)

        unmatched = tuple(
            item_id for item_id in predictions if item_id not in self._items
        )

        return Join(tuple(pairs), tuple(unpredicted), unmatched)

    def __len__(self) -> int:
        """Return how many items the set has."""
        return len(self._items)

    def __contains__(self, item_id: ItemId) -> bool:
        """Return whether the set has an item under an id.

        Parameters
        ----------
        item_id : ItemId
            The id to look for.

        Returns
        -------
        bool
            Whether the set has it.
        """
        return item_id in self._items

    @override
    def __repr__(self) -> str:
        """Return a summary naming how many items the set has."""
        return f"EvalDataset({len(self._items)} items)"
