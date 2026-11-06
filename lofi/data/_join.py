"""Item lookup and prejoining for supervised examples."""

from types import MappingProxyType
from typing import TYPE_CHECKING

from lofi.data._supervision import ForcedChoice, Relational, RelationalScalar

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from lofi._types import ItemId
    from lofi.data._supervision import Example, Supervision
    from lofi.data._values import Item

__all__ = [
    "DuplicateItemIdError",
    "ItemLookup",
    "ItemMismatchError",
    "JoinedExample",
    "MissingItemError",
    "PairAnchorError",
    "prejoin",
]


class DuplicateItemIdError(ValueError):
    """An error for two items carrying one identifier.

    Parameters
    ----------
    item_id : ItemId
        Repeated item identifier.
    """

    def __init__(self, item_id: ItemId) -> None:
        super().__init__(f"two items carry the id {item_id!r}")
        self.item_id = item_id


class MissingItemError(LookupError):
    """An error for supervision referring to an unavailable item.

    Parameters
    ----------
    item_id : ItemId
        Identifier that could not be resolved.
    """

    def __init__(self, item_id: ItemId) -> None:
        super().__init__(f"supervision refers to the unavailable item {item_id!r}")
        self.item_id = item_id


class ItemMismatchError(ValueError):
    """An error for one identifier attached to different item content.

    Parameters
    ----------
    item_id : ItemId
        Identifier shared by the conflicting items.
    """

    def __init__(self, item_id: ItemId) -> None:
        super().__init__(
            f"the example and lookup disagree about the item under {item_id!r}"
        )
        self.item_id = item_id


class PairAnchorError(ValueError):
    """An error for a choice example anchored on the wrong alternative.

    Parameters
    ----------
    anchor : ItemId
        Example anchor.
    expected : ItemId
        First presented alternative.
    """

    def __init__(self, anchor: ItemId, expected: ItemId) -> None:
        super().__init__(
            f"a choice example is anchored on {anchor!r}, not its first "
            f"alternative {expected!r}"
        )
        self.anchor = anchor
        self.expected = expected


class ItemLookup:
    """A frozen lookup from reader-assigned ids to items.

    Parameters
    ----------
    items : Sequence[Item]
        Available items.

    Raises
    ------
    DuplicateItemIdError
        If two items carry the same id.
    """

    __slots__ = ("_items",)

    def __init__(self, items: Sequence[Item]) -> None:
        held: dict[ItemId, Item] = {}

        for item in items:
            if item.item_id in held:
                raise DuplicateItemIdError(item.item_id)

            held[item.item_id] = item

        self._items = held

    @property
    def items(self) -> Mapping[ItemId, Item]:
        """Return the items keyed by id."""
        return MappingProxyType(self._items)

    def resolve(self, item_id: ItemId, /) -> Item:
        """Resolve an item id.

        Parameters
        ----------
        item_id : ItemId
            Reader-assigned item id.

        Returns
        -------
        Item
            The item carrying the id.

        Raises
        ------
        MissingItemError
            If the lookup has no item under the id.
        """
        try:
            return self._items[item_id]
        except KeyError as error:
            raise MissingItemError(item_id) from error

    def __len__(self) -> int:
        """Return the number of available items."""
        return len(self._items)


class JoinedExample[S]:
    """An example paired with the items its supervision references.

    Parameters
    ----------
    example : Example[S]
        Original example.
    items : tuple[Item, ...]
        Resolved items in semantic order.
    """

    __slots__ = ("_example", "_items")

    def __init__(self, example: Example[S], items: tuple[Item, ...]) -> None:
        self._example = example
        self._items = tuple(items)

    @property
    def example(self) -> Example[S]:
        """Return the original example."""
        return self._example

    @property
    def items(self) -> tuple[Item, ...]:
        """Return the resolved items in semantic order."""
        return self._items


def prejoin[L: Hashable](
    examples: Sequence[Example[Supervision[L]]],
    lookup: ItemLookup,
    /,
) -> tuple[JoinedExample[Supervision[L]], ...]:
    """Resolve every item reference in a batch of examples.

    Parameters
    ----------
    examples : Sequence[Example[Supervision[L]]]
        Examples to join.
    lookup : ItemLookup
        Available items by id.

    Returns
    -------
    tuple[JoinedExample[Supervision[L]], ...]
        Joined examples in input order.

    Raises
    ------
    MissingItemError
        If a referenced item is unavailable.
    ItemMismatchError
        If the lookup and example disagree under one id.
    PairAnchorError
        If a choice example is not anchored on its first alternative.
    """
    joined: list[JoinedExample[Supervision[L]]] = []

    for example in examples:
        supervision = example.supervision
        anchor = lookup.resolve(example.item.item_id)

        if anchor != example.item:
            raise ItemMismatchError(example.item.item_id)

        if isinstance(supervision, ForcedChoice):
            first = supervision.alternatives[0]

            if anchor.item_id != first:
                raise PairAnchorError(anchor.item_id, first)

            items = tuple(
                lookup.resolve(item_id) for item_id in supervision.alternatives
            )
        elif isinstance(supervision, (Relational, RelationalScalar)):
            items = (anchor, lookup.resolve(supervision.other))
        else:
            items = (anchor,)

        joined.append(JoinedExample(example, items))

    return tuple(joined)
