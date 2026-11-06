"""Common derivation readings and dependency reductions."""

from typing import TYPE_CHECKING, NamedTuple, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue, Unpicklable

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.deduce import Derivation, Semiring
    from lofi.interpret._derive import StepContext
    from lofi.judge import Rule
    from lofi.lexicon import Entry

__all__ = [
    "CategoryReading",
    "HeadDependencies",
    "HeadLink",
    "StringReading",
    "WeightReading",
]


def _check_arity[Category, Weight, Value](
    context: StepContext[Category, Weight],
    children: tuple[Value, ...],
) -> None:
    """Reject a child-value count that disagrees with the rule.

    Parameters
    ----------
    context : StepContext[Category, Weight]
        Rule application being interpreted.
    children : tuple[Value, ...]
        Interpreted child values.

    Raises
    ------
    ValueError
        If the child count differs from the rule arity.
    """
    if len(children) != context.rule.arity:
        msg = (
            f"rule {context.rule.name!r} takes {context.rule.arity} child values, "
            f"not {len(children)}"
        )
        raise ValueError(msg)


@final
class CategoryReading[Category, Meaning = Category, Weight = float](
    Unpicklable, FrozenValue
):
    """Exact-category derivation reading."""

    _noun = "derivation reading"
    _because = "contains executable behavior"

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "category"

    @property
    def refines_items(self) -> bool:
        """Return whether exact categories refine packed items."""
        return True

    def supports(self, rule: Rule[Category], /) -> bool:
        """Return whether the reading defines the rule."""
        del rule
        return True

    def axiom(self, entry: Entry[Category, Meaning], /) -> Category:
        """Return the entry's exact category."""
        return entry.category

    def step(
        self,
        context: StepContext[Category, Weight],
        children: tuple[Category, ...],
        /,
    ) -> Category:
        """Return the step's exact conclusion."""
        _check_arity(context, children)
        return context.conclusion


@final
class StringReading[Category, Meaning = Category, Weight = float](
    Unpicklable, FrozenValue
):
    """Lexical-form derivation reading."""

    _noun = "derivation reading"
    _because = "contains executable behavior"

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "string"

    @property
    def refines_items(self) -> bool:
        """Return whether strings refine packed items."""
        return False

    def supports(self, rule: Rule[Category], /) -> bool:
        """Return whether the reading defines the rule."""
        del rule
        return True

    def axiom(self, entry: Entry[Category, Meaning], /) -> tuple[str, ...]:
        """Return the entry's lexical form."""
        return (entry.form,)

    def step(
        self,
        context: StepContext[Category, Weight],
        children: tuple[tuple[str, ...], ...],
        /,
    ) -> tuple[str, ...]:
        """Concatenate child forms in physical order."""
        _check_arity(context, children)
        return tuple(form for child in children for form in child)


@final
class WeightReading[Category, W, Meaning = Category](Unpicklable, FrozenValue):
    """Semiring-weight derivation reading.

    Parameters
    ----------
    semiring : Semiring[W]
        Algebra used to combine local and child weights.
    lift : Callable[[float], W]
        Function that lifts lexical weights into the semiring.
    """

    _noun = "derivation reading"
    _because = "contains executable behavior"

    __slots__ = ("_lift", "_semiring")

    def __init__(self, semiring: Semiring[W], lift: Callable[[float], W]) -> None:
        object.__setattr__(self, "_semiring", semiring)
        object.__setattr__(self, "_lift", lift)

    @property
    def name(self) -> str:
        """Return the interpretation name."""
        return "weight"

    @property
    def refines_items(self) -> bool:
        """Return whether weights refine packed items."""
        return True

    def supports(self, rule: Rule[Category], /) -> bool:
        """Return whether the reading defines the rule."""
        del rule
        return True

    def axiom(self, entry: Entry[Category, Meaning], /) -> W:
        """Lift the entry's lexical weight."""
        return self._lift(entry.weight)

    def step(
        self,
        context: StepContext[Category, W],
        children: tuple[W, ...],
        /,
    ) -> W:
        """Multiply the local weight by the child weights."""
        _check_arity(context, children)
        result = context.weight

        for child in children:
            result = self._semiring.times(result, child)

        return result


class HeadLink(NamedTuple):
    """One head-dependent relation.

    Attributes
    ----------
    head : int
        Head-token position.
    dependent : int
        Dependent-token position.
    label : str
        Name of the rule that created the relation.
    """

    head: int
    dependent: int
    label: str


class HeadDependencies[T]:
    """Reduces derivations to head-dependent relations.

    Parameters
    ----------
    heads : Mapping[str, int]
        Zero-based head-premise index keyed by rule name.

    Raises
    ------
    TypeError
        If a head-premise index is not an integer.
    ValueError
        If a head-premise index is negative.
    """

    __slots__ = ("_heads",)

    def __init__(self, heads: Mapping[str, int]) -> None:
        self._heads: dict[str, int] = {
            name: ensure_non_negative(f"heads[{name!r}]", at)
            for name, at in heads.items()
        }

    @property
    def name(self) -> str:
        """Return the reduction name."""
        return "head-dependencies"

    def reduce[W](self, derivation: Derivation[T, W], /) -> tuple[HeadLink, ...]:
        """Return a derivation's head-dependent links.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Tree to reduce.

        Returns
        -------
        tuple[HeadLink, ...]
            Links ordered by dependent and then head position.

        Raises
        ------
        ValueError
            If a nonunary rule lacks a valid head-premise index.
        """
        links: list[HeadLink] = []

        self._head_token(derivation, links)

        return tuple(sorted(links, key=lambda link: (link.dependent, link.head)))

    def _head_token[W](
        self, derivation: Derivation[T, W], links: list[HeadLink]
    ) -> int:
        """Return a subtree's head token and append its dependency links.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Subtree to reduce.
        links : list[HeadLink]
            Destination for relations licensed by the subtree.

        Returns
        -------
        int
            Head-token position.

        Raises
        ------
        ValueError
            If a nonunary rule lacks a valid head-premise index.
        """
        children = derivation.children

        if not children:
            return derivation.key.span[0]

        below = [self._head_token(child, links) for child in children]

        if len(below) == 1:
            return below[0]

        rule = derivation.edge.rule_name
        at = self._heads.get(rule)

        if at is None:
            msg = f"the head table names no premise of {rule!r}"
            raise ValueError(msg)

        if at >= len(below):
            msg = (
                f"the head table names premise {at} of {rule!r}, which has "
                f"{len(below)} premises here"
            )
            raise ValueError(msg)

        head = below[at]

        links.extend(
            HeadLink(head=head, dependent=token, label=rule)
            for premise, token in enumerate(below)
            if premise != at
        )

        return head

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the rules the head table covers."""
        named = ", ".join(sorted(self._heads))

        return f"HeadDependencies(heads=[{named}])"
