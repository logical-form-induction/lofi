"""Interior declarations supplied by a language's leaf lexicon."""

from typing import TYPE_CHECKING, final

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

__all__ = ["LeafConstant", "LeafLexicon"]


@final
class LeafConstant(FrozenValue):
    """One non-logical constant supplied by a leaf lexicon."""

    _noun = "leaf constant"

    __slots__ = ("_classes", "_name", "_type")

    def __init__(self, name: str, type_: str, classes: Sequence[str], /) -> None:
        """Validate and freeze one leaf-constant declaration."""
        if not name:
            msg = "a leaf constant needs a name"
            raise ValueError(msg)

        if not type_:
            msg = f"leaf constant {name!r} needs a type or 'infer'"
            raise ValueError(msg)

        held_classes = tuple(classes)

        if any(not held for held in held_classes):
            msg = f"leaf constant {name!r} class needs a name"
            raise ValueError(msg)

        if len(set(held_classes)) != len(held_classes):
            msg = f"leaf constant {name!r} declares a class twice"
            raise ValueError(msg)

        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_type", type_)
        object.__setattr__(self, "_classes", held_classes)

    @property
    def name(self) -> str:
        """Return the term-constant name."""
        return self._name

    @property
    def type(self) -> str:
        """Return the declared type text or ``infer``."""
        return self._type

    @property
    def classes(self) -> tuple[str, ...]:
        """Return the declared open constant classes."""
        return self._classes


@final
class LeafLexicon(FrozenValue):
    """An ordered, immutable collection of leaf constants."""

    _noun = "leaf lexicon"

    __slots__ = ("_entries",)

    def __init__(self, entries: Sequence[LeafConstant], /) -> None:
        """Copy entries and require each constant name once."""
        held = tuple(entries)
        names = tuple(entry.name for entry in held)

        if len(set(names)) != len(names):
            msg = "a leaf constant is declared twice"
            raise ValueError(msg)

        object.__setattr__(self, "_entries", held)

    def __getitem__(self, position: int, /) -> LeafConstant:
        """Return one declaration by position."""
        return self._entries[position]

    def __iter__(self) -> Iterator[LeafConstant]:
        """Iterate over declarations in source order."""
        return iter(self._entries)

    def __len__(self) -> int:
        """Return the number of declarations."""
        return len(self._entries)
