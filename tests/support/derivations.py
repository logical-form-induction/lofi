"""Unpacked chart derivations built by hand for interpretation tests."""

from typing import TYPE_CHECKING, final, overload

from lofi.deduce import LEXICAL_TAG_PREFIX, Derivation, Edge, ItemKey

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi.judge import Rule
    from lofi.lexicon import Entry

__all__ = ["FormLexicon", "axiom", "step"]


@overload
def axiom[C](entry: Entry[C, object], position: int = 0) -> Derivation[C, float]: ...


@overload
def axiom[C, W](
    entry: Entry[C, object], position: int = 0, *, weight: W
) -> Derivation[C, W]: ...


def axiom[C, W](
    entry: Entry[C, object], position: int = 0, *, weight: W | float = 0.0
) -> Derivation[C, W | float]:
    """Return the lexical axiom of one entry at a token position."""
    category = entry.category

    return Derivation(
        ItemKey((position, position + 1), category),
        Edge(f"{LEXICAL_TAG_PREFIX}{entry.form}", (), category, weight),
    )


def step[C, W](
    rule: Rule[C],
    children: Sequence[Derivation[C, W]],
    conclusion: C,
    weight: W,
    order: int = 0,
) -> Derivation[C, W]:
    """Return one rule application over its premises' derivations."""
    held = tuple(children)
    span = (held[0].key.span[0], held[-1].key.span[1]) if held else (0, 0)
    split = held[1].key.span[0] if len(held) == 2 else None

    return Derivation(
        ItemKey(span, conclusion),
        Edge(
            rule.name,
            tuple(child.key for child in held),
            conclusion,
            weight,
            split=split,
            order=order,
        ),
        held,
    )


@final
class FormLexicon[C, M = C]:
    """Rules by name and entries by form and category, wherever they stand."""

    __slots__ = ("_entries", "_rules")

    def __init__(
        self, rules: Iterable[Rule[C]], entries: Iterable[Entry[C, M]] = ()
    ) -> None:
        self._rules = {rule.name: rule for rule in rules}
        held: list[Entry[C, M]] = []

        for entry in entries:
            if entry not in held:
                held.append(entry)

        self._entries = tuple(held)

    def rule(self, name: str, /) -> Rule[C]:
        """Return the rule with a name."""
        return self._rules[name]

    def admitted(
        self, position: int, form: str, category: C, /
    ) -> tuple[Entry[C, M], ...]:
        """Return the entries with a form and category, ignoring the position."""
        del position

        return tuple(
            entry
            for entry in self._entries
            if entry.form == form and entry.category == category
        )
