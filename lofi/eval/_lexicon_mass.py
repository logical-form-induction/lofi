"""Lexicon posterior mass and supertag confusion tables."""

from collections import Counter
from math import fsum
from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import (
    ensure_fraction,
    ensure_name,
    ensure_non_negative,
)
from lofi.eval._supertagging import TokenSupertags
from lofi.terms import Term, TermKey, TermStore, TermSyntax

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence


__all__ = [
    "LexiconMassRow",
    "SupertagConfusionTable",
    "lexicon_mass_report",
    "supertag_confusion_table",
]

type _Observation = tuple[str, TokenSupertags, Term]
type _CategoryDetails = dict[TermKey, tuple[str, int]]


@final
class LexiconMassRow:
    """Posterior category mass for one caller-supplied word class."""

    __slots__ = ("_gold_attested_mass", "_top_categories", "_word_class")

    def __init__(
        self,
        word_class: str,
        top_categories: Sequence[tuple[str, float]],
        gold_attested_mass: float,
    ) -> None:
        held_categories = tuple(
            (
                ensure_name(f"top category {position}", category),
                ensure_fraction(f"top category {position} mass", mass),
            )
            for position, (category, mass) in enumerate(top_categories)
        )

        if len({category for category, _ in held_categories}) != len(held_categories):
            msg = "a lexicon mass row repeats a top category"
            raise ValueError(msg)

        object.__setattr__(
            self,
            "_word_class",
            ensure_name("a lexicon mass row's word class", word_class),
        )
        object.__setattr__(self, "_top_categories", held_categories)
        object.__setattr__(
            self,
            "_gold_attested_mass",
            ensure_fraction(
                "a lexicon mass row's gold-attested mass", gold_attested_mass
            ),
        )

    @property
    def word_class(self) -> str:
        """Return the caller-supplied word class."""
        return self._word_class

    @property
    def top_categories(self) -> tuple[tuple[str, float], ...]:
        """Return top categories with normalized mass in report order."""
        return self._top_categories

    @property
    def gold_attested_mass(self) -> float:
        """Return the posterior mass on gold-attested categories."""
        return self._gold_attested_mass

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a lexicon mass row is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a lexicon mass row is immutable"
        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another row has the same values."""
        if type(other) is not LexiconMassRow:
            return NotImplemented

        return (
            self._word_class == other._word_class
            and self._top_categories == other._top_categories
            and self._gold_attested_mass == other._gold_attested_mass
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the row values."""
        return hash(
            (
                LexiconMassRow,
                self._word_class,
                self._top_categories,
                self._gold_attested_mass,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"LexiconMassRow({self._word_class!r}, {self._top_categories!r}, "
            f"{self._gold_attested_mass!r})"
        )


@final
class SupertagConfusionTable:
    """A dense supertag confusion matrix over one ordered category axis."""

    __slots__ = ("_categories", "_counts")

    def __init__(
        self,
        categories: Sequence[str],
        counts: Sequence[Sequence[int]],
    ) -> None:
        held_categories = tuple(
            ensure_name(f"confusion category {position}", category)
            for position, category in enumerate(categories)
        )

        if len(set(held_categories)) != len(held_categories):
            msg = "a supertag confusion table repeats a category"
            raise ValueError(msg)

        held_counts = tuple(
            tuple(
                ensure_non_negative(f"confusion count [{row}][{column}]", count)
                for column, count in enumerate(values)
            )
            for row, values in enumerate(counts)
        )
        size = len(held_categories)

        if len(held_counts) != size or any(len(row) != size for row in held_counts):
            shape = tuple(len(row) for row in held_counts)
            msg = f"confusion counts are not a {size} by {size} matrix: {shape!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_categories", held_categories)
        object.__setattr__(self, "_counts", held_counts)

    @property
    def categories(self) -> tuple[str, ...]:
        """Return the shared gold-row and predicted-column category axis."""
        return self._categories

    @property
    def counts(self) -> tuple[tuple[int, ...], ...]:
        """Return counts with gold rows and predicted columns."""
        return self._counts

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a supertag confusion table is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a supertag confusion table is immutable"
        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another table has the same axis and counts."""
        if type(other) is not SupertagConfusionTable:
            return NotImplemented

        return self._categories == other._categories and self._counts == other._counts

    @override
    def __hash__(self) -> int:
        """Return a hash over the axis and counts."""
        return hash((SupertagConfusionTable, self._categories, self._counts))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"SupertagConfusionTable({self._categories!r}, {self._counts!r})"


def _term_depth(term: Term) -> int:
    """Compute one-based structural depth."""
    return 1 + max((_term_depth(child) for child in term.args), default=0)


def _require_context(store: TermStore, syntax: TermSyntax) -> None:
    """Validate the shared term context."""
    if not isinstance(store, TermStore):
        msg = f"a lexicon report store is not a TermStore: {store!r}"
        raise TypeError(msg)

    if not isinstance(syntax, TermSyntax):
        msg = f"a lexicon report syntax does not implement TermSyntax: {syntax!r}"
        raise TypeError(msg)

    if syntax.signature is not store.signature:
        msg = (
            f"lexicon syntax for {syntax.signature.name!r} cannot report terms from "
            f"{store.signature.name!r}"
        )
        raise ValueError(msg)


def _validated_observations(
    store: TermStore,
    syntax: TermSyntax,
    observations: Iterable[_Observation],
) -> tuple[tuple[_Observation, ...], _CategoryDetails]:
    """Validate observations and record canonical category details."""
    _require_context(store, syntax)
    held: list[_Observation] = []
    details: _CategoryDetails = {}

    for position, (word_class, posterior, gold) in enumerate(observations):
        name = ensure_name(f"observation {position}'s word class", word_class)

        if not isinstance(posterior, TokenSupertags):
            msg = (
                f"observation {position}'s posterior is not TokenSupertags: "
                f"{posterior!r}"
            )
            raise TypeError(msg)

        if not isinstance(gold, Term):
            msg = f"observation {position}'s gold category is not a term: {gold!r}"
            raise TypeError(msg)

        categories = (
            *(candidate.category for candidate in posterior.candidates),
            gold,
        )

        for category in categories:
            key = category.key()
            text = syntax.render(store, category)
            found = details.get(key)

            if found is not None and found[0] != text:
                msg = f"structural category {key!r} has conflicting canonical text"
                raise ValueError(msg)

            details[key] = (text, _term_depth(category))

        held.append((name, posterior, gold))

    return tuple(held), details


def lexicon_mass_report(
    store: TermStore,
    syntax: TermSyntax,
    observations: Iterable[_Observation],
    *,
    k: int,
) -> tuple[LexiconMassRow, ...]:
    """Compute normalized posterior mass by word class.

    Raises
    ------
    ValueError
        If ``k`` is zero or the observations contain inconsistent categories.
    """
    held_k = ensure_non_negative("k", k)

    if held_k == 0:
        msg = "k must be positive"
        raise ValueError(msg)

    held, details = _validated_observations(store, syntax, observations)
    masses: dict[str, dict[TermKey, list[float]]] = {}
    gold: dict[str, set[TermKey]] = {}

    for word_class, posterior, gold_category in held:
        by_category = masses.setdefault(word_class, {})
        gold.setdefault(word_class, set()).add(gold_category.key())

        for candidate in posterior.candidates:
            by_category.setdefault(candidate.category.key(), []).append(
                candidate.probability
            )

    rows: list[LexiconMassRow] = []

    for word_class in sorted(masses):
        raw = {
            key: fsum(contributions)
            for key, contributions in masses[word_class].items()
        }
        total = fsum(raw.values())
        normalized = {key: mass / total for key, mass in raw.items()}
        ranked = sorted(
            normalized.items(),
            key=lambda entry: (-entry[1], details[entry[0]][0]),
        )
        attested = gold[word_class]
        rows.append(
            LexiconMassRow(
                word_class,
                tuple((details[key][0], mass) for key, mass in ranked[:held_k]),
                fsum(mass for key, mass in raw.items() if key in attested) / total,
            )
        )

    return tuple(rows)


def supertag_confusion_table(
    store: TermStore,
    syntax: TermSyntax,
    observations: Iterable[_Observation],
) -> SupertagConfusionTable:
    """Compute first-max predictions ordered by one-based depth, then canonical text."""
    held, details = _validated_observations(store, syntax, observations)
    cells: Counter[tuple[TermKey, TermKey]] = Counter()

    for _, posterior, gold in held:
        predicted = max(
            posterior.candidates,
            key=lambda candidate: candidate.probability,
        ).category
        cells[(gold.key(), predicted.key())] += 1

    observed = {key for cell in cells for key in cell}
    keys = tuple(sorted(observed, key=lambda key: (details[key][1], details[key][0])))
    counts = tuple(
        tuple(cells[(gold, predicted)] for predicted in keys) for gold in keys
    )

    return SupertagConfusionTable(
        tuple(details[key][0] for key in keys),
        counts,
    )
