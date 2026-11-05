"""Validated dependency trees and projection conversion."""

from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.data._derivation import Dependency
from lofi.data.deps._projection import HeadProjection

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.data._values import Token

__all__ = ["AttachmentTree", "attachment_tree"]

ROOT: Final = 0
"""The head position of the token the artificial root governs."""


@final
class AttachmentTree(FrozenValue):
    """A single-rooted dependency tree over one sentence.

    Positions are one-based and the artificial root is position zero, so the
    head of token ``i`` is ``heads[i - 1]`` and a head of zero marks the
    sentence head.

    Parameters
    ----------
    forms : Sequence[str]
        Surface form of every token, in sentence order.
    heads : Sequence[int]
        One-based head position of every token, zero for the root arc.
    punctuation : Iterable[int]
        One-based positions of the punctuation tokens.

    Raises
    ------
    TypeError
        If a head or a punctuation position is not an integer.
    ValueError
        If the counts differ, a head or a punctuation position lies outside
        the sentence, a token heads itself, other than exactly one token
        attaches to the root, or a token does not reach the root.
    """

    _noun = "attachment tree"

    __slots__ = ("_forms", "_heads", "_punctuation")

    def __init__(
        self,
        forms: Sequence[str],
        heads: Sequence[int],
        punctuation: Iterable[int] = (),
    ) -> None:
        held_forms = tuple(
            ensure_name(f"the form at position {position}", form)
            for position, form in enumerate(forms, start=1)
        )
        held_heads = tuple(
            ensure_non_negative(f"the head at position {position}", head)
            for position, head in enumerate(heads, start=1)
        )
        length = len(held_forms)

        if len(held_heads) != length:
            msg = f"{length} forms are given {len(held_heads)} heads"
            raise ValueError(msg)

        for position, head in enumerate(held_heads, start=1):
            if head > length:
                msg = f"the head at position {position} lies outside {length} tokens"
                raise ValueError(msg)

            if head == position:
                msg = f"the token at position {position} heads itself"
                raise ValueError(msg)

        roots = held_heads.count(ROOT)

        if length and roots != 1:
            msg = f"{roots} tokens attach to the root"
            raise ValueError(msg)

        for position in range(1, length + 1):
            visited: set[int] = set()
            current = position

            while current != ROOT:
                if current in visited:
                    msg = f"the token at position {position} does not reach the root"
                    raise ValueError(msg)

                visited.add(current)
                current = held_heads[current - 1]

        held_punctuation = frozenset(
            ensure_non_negative("a punctuation position", position)
            for position in punctuation
        )

        for position in held_punctuation:
            if position == ROOT or position > length:
                msg = f"punctuation position {position} lies outside {length} tokens"
                raise ValueError(msg)

        object.__setattr__(self, "_forms", held_forms)
        object.__setattr__(self, "_heads", held_heads)
        object.__setattr__(self, "_punctuation", held_punctuation)

    @property
    def forms(self) -> tuple[str, ...]:
        """Return the surface forms in sentence order."""
        return self._forms

    @property
    def heads(self) -> tuple[int, ...]:
        """Return the head of every token, zero for the root arc."""
        return self._heads

    @property
    def punctuation(self) -> frozenset[int]:
        """Return the one-based positions of the punctuation tokens."""
        return self._punctuation

    @property
    def length(self) -> int:
        """Return the token count, including punctuation."""
        return len(self._forms)

    def head_of(self, position: int) -> int:
        """Return the head of the token at a one-based position.

        Parameters
        ----------
        position : int
            One-based token position.

        Raises
        ------
        ValueError
            If the position lies outside the sentence.
        """
        if position < 1 or position > len(self._heads):
            msg = f"position {position} lies outside {len(self._heads)} tokens"
            raise ValueError(msg)

        return self._heads[position - 1]

    def arcs(self) -> frozenset[tuple[int, int]]:
        """Return every head-to-dependent arc, the root arc included."""
        return frozenset(
            (head, position) for position, head in enumerate(self._heads, start=1)
        )

    def _fields(self) -> tuple[tuple[str, ...], tuple[int, ...], frozenset[int]]:
        """Return fields in declaration order."""
        return self._forms, self._heads, self._punctuation

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[AttachmentTree],
        tuple[tuple[str, ...], tuple[int, ...], frozenset[int]],
    ]:
        """Return the reconstruction call."""
        return AttachmentTree, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another tree has the same forms, heads and punctuation."""
        if type(other) is not AttachmentTree:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the forms, heads and punctuation."""
        return hash((AttachmentTree, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"AttachmentTree({self._forms!r}, {self._heads!r}, "
            f"{sorted(self._punctuation)!r})"
        )


def attachment_tree[S](
    projection: HeadProjection[Dependency, S],
    tokens: Sequence[Token],
    /,
    *,
    punctuation: Iterable[int] = (),
) -> AttachmentTree:
    """Read a completed dependency projection as an attachment tree.

    The projection's arcs are one-based with the root at position zero, as a
    completed induced-grammar projection emits them, and the tokens are
    indexed from zero in sentence order.

    Parameters
    ----------
    projection : HeadProjection[Dependency, S]
        Completed projection whose arcs give every token one head.
    tokens : Sequence[Token]
        Tokens of the projected sentence, indexed from zero.
    punctuation : Iterable[int]
        One-based positions of the punctuation tokens.

    Raises
    ------
    TypeError
        If the projection is not one.
    ValueError
        If the tokens are not consecutive from zero, an arc names a position
        outside the sentence, or a token has other than exactly one head.
    """
    if not isinstance(projection, HeadProjection):
        msg = f"a dependency projection is not one: {projection!r}"
        raise TypeError(msg)

    for expected, token in enumerate(tokens):
        if token.index != expected:
            msg = f"token {expected} has index {token.index}"
            raise ValueError(msg)

    length = len(tokens)
    heads: dict[int, int] = {}

    for dependency in projection.dependencies:
        if not isinstance(dependency, Dependency):
            msg = f"a projected dependency is not one: {dependency!r}"
            raise TypeError(msg)

        dependent = dependency.dependent

        if dependent < 1 or dependent > length or dependency.head > length:
            msg = f"arc {dependency!r} lies outside {length} tokens"
            raise ValueError(msg)

        if dependent in heads:
            msg = f"the token at position {dependent} has more than one head"
            raise ValueError(msg)

        heads[dependent] = dependency.head

    missing = [position for position in range(1, length + 1) if position not in heads]

    if missing:
        msg = f"the tokens at positions {missing} have no head"
        raise ValueError(msg)

    return AttachmentTree(
        tuple(token.surface for token in tokens),
        tuple(heads[position] for position in range(1, length + 1)),
        punctuation,
    )
