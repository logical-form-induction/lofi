"""Bead property expressions over the attributes a judgment item records.

A rule that groups, constrains or stratifies items names the attributes it
reads with a bead DSL expression, evaluated as bead's list partitioner
evaluates one: the variable ``item`` is bound to the item's ``item_metadata``,
so ``item['structure']`` reads the attribute ``structure``. Every expression a
run declares is checked against every item of its pool before the run starts,
so an expression naming an attribute the items do not carry is refused then.

Every source of judgments reads its factors the same way: a design's factors
are a mapping from factor name to expression, read from the item record each
row names. A bead item is such a record, and a corpus codec lowers what its
release says of an item into the same record.
"""

from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, final, override

from bead.dsl.errors import DSLError
from bead.dsl.evaluator import DSLEvaluator
from bead.dsl.parser import parse

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.judgment._design import DesignError

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping
    from uuid import UUID

    from bead.dsl.evaluator import DslValue
    from bead.items.item import MetadataValue

__all__ = [
    "NO_FACTORS",
    "AttributeExpression",
    "AttributedItem",
    "ItemFactors",
    "ItemRecord",
    "item_key",
    "refuse_uncarried",
]

_ITEM: Final = "item"
"""The DSL variable an expression reads an item's attributes through."""


class ItemRecord(Protocol):
    """An identified item and the attributes it records, as a bead item has them."""

    @property
    def id(self) -> UUID | str:
        """Return the item's identity."""
        ...

    @property
    def item_metadata(self) -> Mapping[str, MetadataValue]:
        """Return the item's attributes by name."""
        ...


@final
class AttributedItem(FrozenValue):
    """One corpus item lowered to an item record.

    Parameters
    ----------
    identity : str
        The identity the corpus gives the item.
    item_metadata : Mapping[str, MetadataValue]
        What the corpus records of the item, by attribute name.
    rendered_elements : Mapping[str, str]
        The item's text, by element name.
    """

    _noun = "attributed item"
    __slots__ = ("_elements", "_id", "_metadata")

    def __init__(
        self,
        identity: str,
        item_metadata: Mapping[str, MetadataValue],
        rendered_elements: Mapping[str, str],
        /,
    ) -> None:
        object.__setattr__(self, "_id", ensure_name("an item identity", identity))
        object.__setattr__(self, "_metadata", MappingProxyType(dict(item_metadata)))
        object.__setattr__(self, "_elements", MappingProxyType(dict(rendered_elements)))

    @property
    def id(self) -> str:
        """Return the item's identity."""
        return self._id

    @property
    def item_metadata(self) -> Mapping[str, MetadataValue]:
        """Return the item's attributes by name."""
        return self._metadata

    @property
    def rendered_elements(self) -> Mapping[str, str]:
        """Return the item's text by element name."""
        return self._elements


def item_key(item: ItemRecord, /) -> str:
    """Return an item's identity in judgment-row form."""
    return str(item.id)


@final
class AttributeExpression(FrozenValue):
    """One parsed bead DSL expression over an item's attributes.

    Parameters
    ----------
    text : str
        The declared expression.
    role : str
        What the expression declares, named in refusals.
    many : bool
        Whether the expression may yield a list of level names rather than
        one.
    truth : bool
        Whether the expression yields a truth value rather than level names.
    numeric : bool
        Whether the expression yields a number rather than level names.

    Raises
    ------
    ValueError
        If the expression does not parse.
    """

    _noun = "attribute expression"
    __slots__ = ("_evaluator", "_many", "_numeric", "_role", "_text", "_truth")

    def __init__(
        self,
        text: str,
        /,
        *,
        role: str,
        many: bool = False,
        truth: bool = False,
        numeric: bool = False,
    ) -> None:
        held = ensure_name(role, text)
        try:
            parse(held)
        except DSLError as refused:
            msg = f"{role} {held!r} is not a bead expression: {refused}"
            raise ValueError(msg) from refused
        object.__setattr__(self, "_text", held)
        object.__setattr__(self, "_role", role)
        object.__setattr__(self, "_many", many)
        object.__setattr__(self, "_truth", truth)
        object.__setattr__(self, "_numeric", numeric)
        object.__setattr__(self, "_evaluator", DSLEvaluator())

    @property
    def text(self) -> str:
        """Return the declared expression."""
        return self._text

    @property
    def role(self) -> str:
        """Return what the expression declares."""
        return self._role

    def value(self, item: ItemRecord, /) -> DslValue:
        """Return the expression's value over one item's attributes.

        Raises
        ------
        ValueError
            If the expression cannot be evaluated over the item's attributes.
        """
        context: Mapping[str, DslValue] = {_ITEM: dict(item.item_metadata)}
        try:
            return self._evaluator.evaluate(self._text, context)
        except DSLError as refused:
            carried = ", ".join(sorted(item.item_metadata)) or "nothing"
            msg = (
                f"{self._role} {self._text!r} cannot be read from item "
                f"{item_key(item)}, which carries {carried}: {refused}"
            )
            raise ValueError(msg) from refused

    def levels(self, item: ItemRecord, /) -> tuple[str, ...]:
        """Return the expression's value over one item as level names.

        A string is one level name; an expression declared to yield several
        may also yield a list of strings.

        Raises
        ------
        ValueError
            If the expression cannot be evaluated or yields neither.
        """
        value = self.value(item)
        if isinstance(value, str):
            return (value,)
        if (
            self._many
            and isinstance(value, list | tuple)
            and all(isinstance(member, str) for member in value)
        ):
            return tuple(str(member) for member in value)
        wanted = "a level name or a list of them" if self._many else "a level name"
        msg = (
            f"{self._role} {self._text!r} yields {value!r} for item "
            f"{item_key(item)}, not {wanted}"
        )
        raise ValueError(msg)

    def holds(self, item: ItemRecord, /) -> bool:
        """Return the expression's truth value over one item.

        Raises
        ------
        ValueError
            If the expression cannot be evaluated or yields no truth value.
        """
        match self.value(item):
            case bool() as held:
                return held
            case value:
                msg = (
                    f"{self._role} {self._text!r} yields {value!r} for item "
                    f"{item_key(item)}, not a truth value"
                )
                raise ValueError(msg)

    def number(self, item: ItemRecord, /) -> float:
        """Return the expression's finite numeric value over one item.

        Raises
        ------
        ValueError
            If the expression cannot be evaluated or yields no finite number.
        """
        value = self.value(item)
        if (
            isinstance(value, int | float)
            and not isinstance(value, bool)
            and isfinite(value)
        ):
            return float(value)
        msg = (
            f"{self._role} {self._text!r} yields {value!r} for item "
            f"{item_key(item)}, not a finite number"
        )
        raise ValueError(msg)

    def check(self, item: ItemRecord, /) -> None:
        """Refuse an item the expression cannot be read from as it is declared.

        Raises
        ------
        ValueError
            If the expression cannot be evaluated, or yields no truth value for
            a truth expression, no number for a numeric one, or no level names
            for another.
        """
        if self._truth:
            self.holds(item)
        elif self._numeric:
            self.number(item)
        else:
            self.levels(item)

    def level(self, item: ItemRecord, /) -> str:
        """Return the expression's single level name over one item.

        Raises
        ------
        ValueError
            If the expression cannot be evaluated or yields no single string.
        """
        match self.value(item):
            case str() as level:
                return level
            case value:
                msg = (
                    f"{self._role} {self._text!r} yields {value!r} for item "
                    f"{item_key(item)}, not a level name"
                )
                raise ValueError(msg)

    @override
    def __repr__(self) -> str:
        return f"AttributeExpression({self._text!r}, role={self._role!r})"


def refuse_uncarried(
    expressions: Iterable[AttributeExpression], items: Iterable[ItemRecord], /
) -> None:
    """Refuse an expression some item's attributes cannot answer.

    Every expression is evaluated over every item, so an expression naming an
    attribute an item does not carry, or yielding what its rule cannot read, is
    refused before any rule reads it.

    Raises
    ------
    ValueError
        If an expression cannot be read from an item.
    """
    held = tuple(items)
    for expression in expressions:
        for item in held:
            expression.check(item)


@final
class ItemFactors(FrozenValue):
    """A design's declared factors, each read from the items its rows name.

    Parameters
    ----------
    expressions : Mapping[str, str]
        The bead expression each factor's level is read by, keyed by the
        factor's name, in column order.
    items : Iterable[ItemRecord]
        Every item a row may name.

    Raises
    ------
    ValueError
        If a factor name is empty, an expression does not parse or cannot be
        read from an item as one level name, or two items share an identity.
    """

    _noun = "item factors"
    __slots__ = ("_expressions", "_items")

    def __init__(
        self, expressions: Mapping[str, str], items: Iterable[ItemRecord], /
    ) -> None:
        parsed = {
            ensure_name("a factor name", name): AttributeExpression(
                text, role=f"factor {name!r}"
            )
            for name, text in expressions.items()
        }
        held: dict[str, ItemRecord] = {}
        for item in items:
            key = item_key(item)
            if key in held:
                msg = f"two items share the identity {key!r}"
                raise ValueError(msg)
            held[key] = item
        refuse_uncarried(parsed.values(), held.values())
        object.__setattr__(self, "_expressions", MappingProxyType(parsed))
        object.__setattr__(self, "_items", MappingProxyType(held))

    @property
    def names(self) -> tuple[str, ...]:
        """Return the factor names in column order."""
        return tuple(self._expressions)

    def levels(self, key: str, /) -> dict[str, str]:
        """Return each factor's level for the item a row names.

        Raises
        ------
        DesignError
            If a factor is declared and no item has the identity.
        """
        if not self._expressions:
            return {}
        item = self._items.get(key)
        if item is None:
            msg = f"no item has the identity {key!r}, so it has no factor levels"
            raise DesignError(msg)
        return {name: read.level(item) for name, read in self._expressions.items()}

    @override
    def __repr__(self) -> str:
        return f"ItemFactors({self.names!r}, items={len(self._items)})"


NO_FACTORS: Final = ItemFactors({}, ())
"""The factors of a design that declares none."""
