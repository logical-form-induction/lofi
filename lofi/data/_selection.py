"""Selection policies for constraint sources."""

from collections import Counter
from enum import Enum
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_finite, ensure_fraction, ensure_non_negative
from lofi._stats import spearman
from lofi._types import ClassRole, QuantilePopulation
from lofi.data._supervision import Weighted

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping

    from lofi.data._ports import AmbiguityRule

__all__ = [
    "CategoryBudget",
    "CategoryClass",
    "CategoryRole",
    "ClassRole",
    "CorpusProfile",
    "Form",
    "FormSelector",
    "FrequencyBand",
    "FrequencyCount",
    "FrequencyQuantile",
    "MajorityRule",
    "PerOccurrenceRule",
    "QuantilePopulation",
    "RankBand",
    "ScopeKind",
    "SelectionPolicy",
    "SelectionReport",
    "UnionRule",
    "WeightedRule",
    "agreement",
]

type Form = tuple[str, ...]
"""A token sequence, by surface."""


class ScopeKind(Enum):
    """The kind of scope a constraint source binds its constraints at."""

    #: The token level: one occurrence, at its own span.
    OCCURRENCE = "occurrence"

    #: The type level: every occurrence of a form.
    FORM = "form"


class CorpusProfile[C: Hashable]:
    """The token counts and observed categories of the forms of one corpus.

    The token mass is the total over the counted forms, so every share a
    selection reports is a share of the evidence the source could draw on
    rather than of the whole text.

    Parameters
    ----------
    counts : Mapping[Form, int]
        How often each form occurs.
    categories : Mapping[Form, frozenset[C]]
        The categories each form was observed under, empty where the evidence
        names none.

    Raises
    ------
    TypeError
        If a count is not an integer.
    ValueError
        If a count is negative, or the two mappings name different forms.
    """

    __slots__ = ("_categories", "_concordance", "_counts", "_ranked")

    def __init__(
        self, counts: Mapping[Form, int], categories: Mapping[Form, frozenset[C]]
    ) -> None:
        if set(counts) != set(categories):
            missing = sorted(set(counts) ^ set(categories))
            msg = (
                "a profile counts and categorizes the same forms, and these are "
                f"named on one side only: {missing}"
            )
            raise ValueError(msg)

        self._counts = MappingProxyType(
            {
                form: ensure_non_negative(f"the count of {list(form)}", count)
                for form, count in counts.items()
            }
        )
        self._categories = MappingProxyType(dict(categories))
        self._ranked = tuple(
            sorted(self._counts, key=lambda form: (-self._counts[form], form))
        )
        self._concordance = _concordance(self._counts, self._categories)

    @property
    def counts(self) -> Mapping[Form, int]:
        """Return how often each form occurs."""
        return self._counts

    @property
    def categories(self) -> Mapping[Form, frozenset[C]]:
        """Return the categories each form was observed under."""
        return self._categories

    @property
    def ranked(self) -> tuple[Form, ...]:
        """Return the forms from most to least frequent, ties in form order."""
        return self._ranked

    @property
    def types(self) -> int:
        """Return how many distinct forms the profile holds."""
        return len(self._counts)

    @property
    def tokens(self) -> int:
        """Return the token mass those forms hold."""
        return sum(self._counts.values())

    @property
    def class_sizes(self) -> Mapping[C, int]:
        """Return how many forms bear each observed category."""
        sizes: Counter[C] = Counter()

        for observed in self._categories.values():
            sizes.update(observed)

        return MappingProxyType(dict(sizes))

    @property
    def concordance(self) -> float | None:
        """Return the rank correlation of a form's count with its inverse class size.

        Every form-category pair is one unit. The correlation is near one when
        the most frequent forms belong to the smallest classes, near zero when
        frequency says nothing about class size, and nothing when the counts or
        the sizes do not vary.
        """
        return self._concordance

    def share(self, tokens: int, /) -> float:
        """Return a token count as a share of the profile's mass, zero when empty.

        Parameters
        ----------
        tokens : int
            The count to divide.
        """
        total = self.tokens

        return tokens / total if total else 0.0

    def _identity(
        self,
    ) -> tuple[tuple[tuple[Form, int], ...], frozenset[tuple[Form, frozenset[C]]]]:
        """Return the fields a profile is equal on."""
        return (
            tuple(sorted(self._counts.items())),
            frozenset(self._categories.items()),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value profiles the same forms the same way."""
        if type(other) is not CorpusProfile:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts and the observed categories."""
        return hash((CorpusProfile, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this profile."""
        return f"CorpusProfile({dict(self._counts)!r}, {dict(self._categories)!r})"


def _concordance[C: Hashable](
    counts: Mapping[Form, int], categories: Mapping[Form, frozenset[C]]
) -> float | None:
    """Return the rank correlation of form frequency with inverse class size.

    Parameters
    ----------
    counts : Mapping[Form, int]
        How often each form occurs.
    categories : Mapping[Form, frozenset[C]]
        The categories each form was observed under.
    """
    sizes: Counter[C] = Counter()

    for observed in categories.values():
        sizes.update(observed)

    frequencies: list[float] = []
    inverse_sizes: list[float] = []

    for form, observed in categories.items():
        for category in observed:
            frequencies.append(float(counts[form]))
            inverse_sizes.append(1.0 / sizes[category])

    return spearman(frequencies, inverse_sizes)


class RankBand:
    """The forms of a corpus whose frequency rank falls in a band of shares.

    The forms are ranked from most to least frequent and the band runs from
    the form after the first share of that ranking to the last form within
    the last share, each share taken to the nearest form. Because the bounds
    are shares of the ranking, one band names the same part of the
    distribution whatever the corpus holds.

    Parameters
    ----------
    first : float
        The share of the ranked forms above the band.
    last : float
        The share of the ranked forms at or above the band's lowest rank.

    Raises
    ------
    TypeError
        If a share is not a number.
    ValueError
        If a share is outside the unit interval or the last precedes the first.
    """

    __slots__ = ("_first", "_last")

    def __init__(self, first: float, last: float) -> None:
        self._first = ensure_fraction("a rank band's first share", first)
        self._last = ensure_fraction("a rank band's last share", last)

        if self._last < self._first:
            msg = (
                f"a rank band runs from share {self._first} to share "
                f"{self._last}, which admits none"
            )
            raise ValueError(msg)

    @property
    def first(self) -> float:
        """Return the share of the ranked forms above the band."""
        return self._first

    @property
    def last(self) -> float:
        """Return the share of the ranked forms at or above the band's floor."""
        return self._last

    @property
    def size_invariant(self) -> bool:
        """Return true: the band is a share of the ranking."""
        return True

    def admitted[C: Hashable](self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the corpus's forms whose rank falls inside the band.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the ranks are read off.
        """
        types = profile.types

        return frozenset(
            profile.ranked[round(self._first * types) : round(self._last * types)]
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value admits the same band of ranks."""
        if type(other) is not RankBand:
            return NotImplemented

        return (self._first, self._last) == (other._first, other._last)

    @override
    def __hash__(self) -> int:
        """Return a hash over the two shares."""
        return hash((RankBand, self._first, self._last))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this band."""
        return f"RankBand({self._first!r}, {self._last!r})"


class FrequencyBand:
    """The forms of a corpus whose relative frequency falls in a closed band.

    A form's relative frequency is its count as a share of the corpus's token
    mass, so one band names forms of the same relative frequency whatever the
    corpus holds.

    Parameters
    ----------
    at_least : float
        The lowest admitted share of the token mass.
    at_most : float
        The highest admitted share of the token mass.

    Raises
    ------
    TypeError
        If a bound is not a number.
    ValueError
        If a bound is outside the unit interval or the ceiling is below the
        floor.
    """

    __slots__ = ("_at_least", "_at_most")

    def __init__(self, at_least: float, at_most: float) -> None:
        self._at_least = ensure_fraction("a frequency band's floor", at_least)
        self._at_most = ensure_fraction("a frequency band's ceiling", at_most)

        if self._at_most < self._at_least:
            msg = (
                f"a frequency band runs from {self._at_least} to "
                f"{self._at_most}, which admits none"
            )
            raise ValueError(msg)

    @property
    def at_least(self) -> float:
        """Return the lowest admitted share of the token mass."""
        return self._at_least

    @property
    def at_most(self) -> float:
        """Return the highest admitted share of the token mass."""
        return self._at_most

    @property
    def size_invariant(self) -> bool:
        """Return true: the band is a share of the token mass."""
        return True

    def admitted[C: Hashable](self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the corpus's forms whose relative frequency is in the band.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the frequencies are read off.
        """
        return frozenset(
            form
            for form, count in profile.counts.items()
            if self._at_least <= profile.share(count) <= self._at_most
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value admits the same relative frequencies."""
        if type(other) is not FrequencyBand:
            return NotImplemented

        return (self._at_least, self._at_most) == (other._at_least, other._at_most)

    @override
    def __hash__(self) -> int:
        """Return a hash over the two bounds."""
        return hash((FrequencyBand, self._at_least, self._at_most))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this band."""
        return f"FrequencyBand({self._at_least!r}, {self._at_most!r})"


class FrequencyCount:
    """The forms of a corpus whose absolute count falls in an inclusive band.

    The band admits the same counts whatever the corpus holds, so the share of
    a corpus it selects grows or shrinks with the corpus.

    Parameters
    ----------
    at_least : int
        The lowest admitted count.
    at_most : int | None
        The highest admitted count, or None for no ceiling.

    Raises
    ------
    TypeError
        If a bound is not an integer.
    ValueError
        If a bound is negative or the ceiling is below the floor.
    """

    __slots__ = ("_at_least", "_at_most")

    def __init__(self, at_least: int, at_most: int | None) -> None:
        self._at_least = ensure_non_negative("a count band's floor", at_least)
        self._at_most = (
            None
            if at_most is None
            else ensure_non_negative("a count band's ceiling", at_most)
        )

        if self._at_most is not None and self._at_most < self._at_least:
            msg = (
                f"a count band admits counts from {self._at_least} up to "
                f"{self._at_most}, which admits none"
            )
            raise ValueError(msg)

    @property
    def at_least(self) -> int:
        """Return the lowest admitted count."""
        return self._at_least

    @property
    def at_most(self) -> int | None:
        """Return the highest admitted count, or None for no ceiling."""
        return self._at_most

    @property
    def size_invariant(self) -> bool:
        """Return false: the band is a range of absolute counts."""
        return False

    def admitted[C: Hashable](self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the corpus's forms whose count falls inside the band.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the counts are read off.
        """
        return frozenset(
            form
            for form, count in profile.counts.items()
            if self._at_least <= count
            and (self._at_most is None or count <= self._at_most)
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value admits the same counts."""
        if type(other) is not FrequencyCount:
            return NotImplemented

        return (self._at_least, self._at_most) == (other._at_least, other._at_most)

    @override
    def __hash__(self) -> int:
        """Return a hash over the two bounds."""
        return hash((FrequencyCount, self._at_least, self._at_most))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this band."""
        return f"FrequencyCount({self._at_least!r}, {self._at_most!r})"


class FrequencyQuantile:
    """The most frequent forms of a corpus, down to a declared share.

    On the type scale the share is a share of the distinct forms. On the token
    scale it is a share of the token mass, and the forms are taken from the
    most frequent down until their mass reaches it, so the form that crosses
    the share is included.

    Parameters
    ----------
    share : float
        The share of the scale to cover.
    over : QuantilePopulation
        The scale the share is measured on.

    Raises
    ------
    TypeError
        If the share is not a number.
    ValueError
        If the share is outside the unit interval.
    """

    __slots__ = ("_over", "_share")

    def __init__(self, share: float, *, over: QuantilePopulation) -> None:
        self._share = ensure_fraction("a quantile's share", share)
        self._over = over

    @property
    def share(self) -> float:
        """Return the share of the scale covered."""
        return self._share

    @property
    def over(self) -> QuantilePopulation:
        """Return the scale the share is measured on."""
        return self._over

    @property
    def size_invariant(self) -> bool:
        """Return true: the quantile is a share of its scale."""
        return True

    def admitted[C: Hashable](self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the most frequent forms covering the declared share.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the share is read off.
        """
        if self._over is QuantilePopulation.TYPES:
            return frozenset(profile.ranked[: round(self._share * profile.types)])

        wanted = self._share * profile.tokens
        taken: list[Form] = []
        reached = 0

        for form in profile.ranked:
            if reached >= wanted:
                break

            taken.append(form)
            reached += profile.counts[form]

        return frozenset(taken)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value covers the same share of the same scale."""
        if type(other) is not FrequencyQuantile:
            return NotImplemented

        return self._share == other._share and self._over is other._over

    @override
    def __hash__(self) -> int:
        """Return a hash over the share and the scale."""
        return hash((FrequencyQuantile, self._share, self._over))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this quantile."""
        return f"FrequencyQuantile({self._share!r}, over={self._over!r})"


class CategoryClass[C: Hashable]:
    """Every form the evidence observed under one of a declared set of categories.

    Parameters
    ----------
    categories : Iterable[C]
        The categories naming the class.

    Raises
    ------
    ValueError
        If no category is named.
    """

    __slots__ = ("_categories",)

    def __init__(self, categories: Iterable[C]) -> None:
        self._categories = frozenset(categories)

        if not self._categories:
            msg = "a category class names at least one category"
            raise ValueError(msg)

    @property
    def categories(self) -> frozenset[C]:
        """Return the categories naming the class."""
        return self._categories

    @property
    def size_invariant(self) -> bool:
        """Return true: the class is named, not counted."""
        return True

    def admitted(self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the corpus's forms observed under a named category.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the categories are read off.
        """
        return frozenset(
            form
            for form, observed in profile.categories.items()
            if observed & self._categories
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same categories."""
        if type(other) is not CategoryClass:
            return NotImplemented

        return self._categories == other._categories

    @override
    def __hash__(self) -> int:
        """Return a hash over the named categories."""
        return hash((CategoryClass, self._categories))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this class."""
        return f"CategoryClass({set(self._categories)!r})"


class CategoryRole[C: Hashable]:
    """Every form observed under a category at one end of the class sizes.

    The corpus's categories are ordered by how many of its forms bear them,
    smallest first for the closed role and largest first for the open role.
    The class is the declared share of that order, taken to the nearest
    category, and it includes every category tied in size with the last one
    taken, so the selection never depends on how ties are broken. Because the
    class is read off the corpus by size, one declaration addresses the
    analogous categories of grammars whose categories differ.

    Parameters
    ----------
    role : ClassRole
        The end of the size order the class is taken from.
    share : float
        The share of the corpus's categories the class spans.

    Raises
    ------
    TypeError
        If the share is not a number.
    ValueError
        If the share is outside the unit interval.
    """

    __slots__ = ("_role", "_share")

    def __init__(self, role: ClassRole, share: float) -> None:
        self._role = role
        self._share = ensure_fraction("a category role's share", share)

    @property
    def role(self) -> ClassRole:
        """Return the end of the size order the class is taken from."""
        return self._role

    @property
    def share(self) -> float:
        """Return the share of the corpus's categories the class spans."""
        return self._share

    @property
    def size_invariant(self) -> bool:
        """Return true: the class is a share of the category inventory."""
        return True

    def classes(self, profile: CorpusProfile[C], /) -> frozenset[C]:
        """Return the categories of the corpus the role addresses.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the class sizes are read off.
        """
        sizes = profile.class_sizes
        taken = round(self._share * len(sizes))

        if taken == 0:
            return frozenset()

        closed = self._role is ClassRole.CLOSED
        ordered = sorted(sizes.values(), reverse=not closed)
        boundary = ordered[taken - 1]

        return frozenset(
            category
            for category, size in sizes.items()
            if (size <= boundary if closed else size >= boundary)
        )

    def admitted(self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the corpus's forms observed under an addressed category.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the categories are read off.
        """
        addressed = self.classes(profile)

        return frozenset(
            form
            for form, observed in profile.categories.items()
            if observed & addressed
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value addresses the same share of one end."""
        if type(other) is not CategoryRole:
            return NotImplemented

        return self._role is other._role and self._share == other._share

    @override
    def __hash__(self) -> int:
        """Return a hash over the role and the share."""
        return hash((CategoryRole, self._role, self._share))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this class."""
        return f"CategoryRole({self._role!r}, {self._share!r})"


type FormSelector[C: Hashable] = (
    RankBand
    | FrequencyBand
    | FrequencyCount
    | FrequencyQuantile
    | CategoryClass[C]
    | CategoryRole[C]
)
"""A rule choosing the forms of a corpus a policy selects."""

_EVERY_FORM: Final = RankBand(0.0, 1.0)
"""The band admitting every form of a corpus."""


def agreement[C: Hashable](
    first: frozenset[Form], second: frozenset[Form], profile: CorpusProfile[C], /
) -> float | None:
    """Return the chance-corrected agreement of two selections over one corpus.

    Parameters
    ----------
    first : frozenset[Form]
        One selection.
    second : frozenset[Form]
        The other selection.
    profile : CorpusProfile[C]
        The corpus both selections were made on.

    Returns
    -------
    float | None
        One where the two selections coincide, zero where they overlap no more
        than two selections of their sizes drawn at random would, and nothing
        where both selections take every form or both take none, so that
        chance agreement is already certain.

    Raises
    ------
    ValueError
        If a selection names a form the profile does not hold.
    """
    forms = frozenset(profile.counts)
    stray = sorted((first | second) - forms)

    if stray:
        msg = f"a selection names forms the corpus does not hold: {stray}"
        raise ValueError(msg)

    total = len(forms)

    if total == 0:
        return None

    observed = (len(first & second) + len(forms - (first | second))) / total
    expected = (
        len(first) * len(second) + (total - len(first)) * (total - len(second))
    ) / total**2

    return None if expected == 1.0 else (observed - expected) / (1.0 - expected)


class CategoryBudget:
    """What one category contributes to an achieved constraint budget.

    Parameters
    ----------
    types_pinned : int
        How many of the pinned forms bear the category.
    tokens_covered : int
        How many of the covered tokens those forms account for.
    mass_share : float
        Those tokens as a share of the corpus's token mass.

    Raises
    ------
    TypeError
        If a count is not an integer or the share is not a number.
    ValueError
        If a count is negative or the share is outside the unit interval.
    """

    __slots__ = ("_mass_share", "_tokens_covered", "_types_pinned")

    def __init__(
        self, types_pinned: int, tokens_covered: int, mass_share: float
    ) -> None:
        self._types_pinned = ensure_non_negative("a pinned type count", types_pinned)
        self._tokens_covered = ensure_non_negative(
            "a covered token count", tokens_covered
        )
        self._mass_share = ensure_finite("a covered mass share", mass_share)

        if not 0.0 <= self._mass_share <= 1.0:
            msg = f"a covered mass share is a fraction, not {mass_share!r}"
            raise ValueError(msg)

    @property
    def types_pinned(self) -> int:
        """Return how many pinned forms bear the category."""
        return self._types_pinned

    @property
    def tokens_covered(self) -> int:
        """Return how many covered tokens those forms account for."""
        return self._tokens_covered

    @property
    def mass_share(self) -> float:
        """Return those tokens as a share of the corpus's token mass."""
        return self._mass_share

    def _identity(self) -> tuple[int, int, float]:
        """Return the three fields a budget row is equal on."""
        return (self._types_pinned, self._tokens_covered, self._mass_share)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reports the same three quantities."""
        if type(other) is not CategoryBudget:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the two counts and the share."""
        return hash((CategoryBudget, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this row."""
        return (
            f"CategoryBudget({self._types_pinned!r}, {self._tokens_covered!r}, "
            f"{self._mass_share!r})"
        )


class SelectionReport[C: Hashable]:
    """The constraint budget a selection achieved on one corpus.

    A form bearing several categories is counted under each of them, so the
    rows sum to more than the whole where categories overlap.

    Parameters
    ----------
    types_pinned : int
        How many forms the selection pinned.
    tokens_covered : int
        How many tokens those forms cover.
    mass_share : float
        Those tokens as a share of the corpus's token mass.
    by_category : Mapping[C, CategoryBudget]
        The same three quantities for each category the pinned forms bear.
    concordance : float | None
        The rank-class concordance of the corpus the selection was made on,
        or nothing where its counts or class sizes do not vary.
    selector : FormSelector[C]
        The selector that chose the pinned forms, which says whether the
        budget is a share of the corpus or a set of absolute counts.

    Raises
    ------
    TypeError
        If a count is not an integer or a share is not a number.
    ValueError
        If a count is negative, a share is outside the unit interval, or a row
        pins more types or covers more tokens than the selection did.
    """

    __slots__ = (
        "_by_category",
        "_concordance",
        "_mass_share",
        "_selector",
        "_tokens_covered",
        "_types_pinned",
    )

    def __init__(
        self,
        types_pinned: int,
        tokens_covered: int,
        mass_share: float,
        by_category: Mapping[C, CategoryBudget],
        concordance: float | None = None,
        *,
        selector: FormSelector[C],
    ) -> None:
        self._types_pinned = ensure_non_negative("a pinned type count", types_pinned)
        self._tokens_covered = ensure_non_negative(
            "a covered token count", tokens_covered
        )
        self._mass_share = ensure_finite("a covered mass share", mass_share)

        if not 0.0 <= self._mass_share <= 1.0:
            msg = f"a covered mass share is a fraction, not {mass_share!r}"
            raise ValueError(msg)

        for category, row in by_category.items():
            if row.types_pinned > self._types_pinned:
                msg = (
                    f"{category!r} is borne by {row.types_pinned} of the "
                    f"{self._types_pinned} forms the selection pinned"
                )
                raise ValueError(msg)

            if row.tokens_covered > self._tokens_covered:
                msg = (
                    f"{category!r} accounts for {row.tokens_covered} of the "
                    f"{self._tokens_covered} tokens the selection covered"
                )
                raise ValueError(msg)

        self._by_category = MappingProxyType(dict(by_category))
        self._concordance = (
            None if concordance is None else ensure_finite("a concordance", concordance)
        )
        self._selector = selector

    @classmethod
    def of(
        cls,
        profile: CorpusProfile[C],
        covered: Mapping[Form, int],
        selector: FormSelector[C],
        /,
    ) -> SelectionReport[C]:
        """Return the budget a selection achieved against the corpus it read.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the selection was made on.
        covered : Mapping[Form, int]
            How many of each pinned form's tokens the selection covers.
        selector : FormSelector[C]
            The selector that chose the pinned forms.

        Raises
        ------
        ValueError
            If a pinned form is not one the profile holds, or more of its
            tokens are covered than the corpus has.
        """
        stray = sorted(set(covered) - set(profile.counts))

        if stray:
            msg = f"a selection pins forms the corpus does not hold: {stray}"
            raise ValueError(msg)

        for form, tokens in covered.items():
            if tokens > profile.counts[form]:
                msg = (
                    f"{list(form)} covers {tokens} tokens of the "
                    f"{profile.counts[form]} it has"
                )
                raise ValueError(msg)

        types: Counter[C] = Counter()
        tokens_by_category: Counter[C] = Counter()

        for form, tokens in covered.items():
            for category in profile.categories[form]:
                types[category] += 1
                tokens_by_category[category] += tokens

        rows = {
            category: CategoryBudget(
                types[category],
                tokens_by_category[category],
                profile.share(tokens_by_category[category]),
            )
            for category in sorted(
                types, key=lambda named: (-tokens_by_category[named], -types[named])
            )
        }
        total = sum(covered.values())

        return cls(
            len(covered),
            total,
            profile.share(total),
            rows,
            profile.concordance,
            selector=selector,
        )

    @property
    def types_pinned(self) -> int:
        """Return how many forms the selection pinned."""
        return self._types_pinned

    @property
    def tokens_covered(self) -> int:
        """Return how many tokens those forms cover."""
        return self._tokens_covered

    @property
    def mass_share(self) -> float:
        """Return those tokens as a share of the corpus's token mass."""
        return self._mass_share

    @property
    def by_category(self) -> Mapping[C, CategoryBudget]:
        """Return the budget each category the pinned forms bear accounts for."""
        return self._by_category

    @property
    def concordance(self) -> float | None:
        """Return the rank-class concordance of the corpus the selection read."""
        return self._concordance

    @property
    def selector(self) -> FormSelector[C]:
        """Return the selector that chose the pinned forms."""
        return self._selector

    @property
    def size_invariant(self) -> bool:
        """Return whether the selector named a share of the corpus, not counts."""
        return self._selector.size_invariant

    def _identity(
        self,
    ) -> tuple[
        int,
        int,
        float,
        frozenset[tuple[C, CategoryBudget]],
        float | None,
        FormSelector[C],
    ]:
        """Return the fields a report is equal on."""
        return (
            self._types_pinned,
            self._tokens_covered,
            self._mass_share,
            frozenset(self._by_category.items()),
            self._concordance,
            self._selector,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value reports the same achieved budget."""
        if type(other) is not SelectionReport:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts, shares, concordance and selector."""
        return hash((SelectionReport, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return (
            f"SelectionReport({self._types_pinned!r}, {self._tokens_covered!r}, "
            f"{self._mass_share!r}, {dict(self._by_category)!r}, "
            f"{self._concordance!r}, selector={self._selector!r})"
        )


def _observed[C: Hashable](masses: Mapping[C, float]) -> Mapping[C, float]:
    """Return the masses once they are known to name at least one label.

    Parameters
    ----------
    masses : Mapping[C, float]
        The mass observed on each label.

    Raises
    ------
    ValueError
        If no label was observed, or a label bears no positive mass.
    """
    if not masses:
        msg = "a form observed under no type permits nothing"
        raise ValueError(msg)

    for label, mass in masses.items():
        if not isfinite(mass) or mass <= 0.0:
            msg = (
                f"{label!r} was observed with a mass of {mass!r}, which is no "
                "observation"
            )
            raise ValueError(msg)

    return masses


class UnionRule[C: Hashable]:
    """Permits every type a form was observed under, bound at every occurrence."""

    __slots__ = ()

    @property
    def scope(self) -> ScopeKind:
        """Return the type-level scope."""
        return ScopeKind.FORM

    def permitted(self, masses: Mapping[C, float], /) -> frozenset[C]:
        """Return every observed label.

        Parameters
        ----------
        masses : Mapping[C, float]
            The mass observed on each label.

        Raises
        ------
        ValueError
            If no label was observed, or a label bears no positive mass.
        """
        return frozenset(_observed(masses))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is also the union rule."""
        if type(other) is not UnionRule:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the shared hash for union rules."""
        return hash(UnionRule)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this rule."""
        return "UnionRule()"


class MajorityRule[C: Hashable]:
    """Permits the type bearing the most mass, bound at every occurrence.

    A tie goes to the label observed first.
    """

    __slots__ = ()

    @property
    def scope(self) -> ScopeKind:
        """Return the type-level scope."""
        return ScopeKind.FORM

    def permitted(self, masses: Mapping[C, float], /) -> frozenset[C]:
        """Return the label bearing the most mass, alone.

        Parameters
        ----------
        masses : Mapping[C, float]
            The mass observed on each label, in observation order.

        Raises
        ------
        ValueError
            If no label was observed, or a label bears no positive mass.
        """
        observed = _observed(masses)

        return frozenset({max(observed, key=lambda label: observed[label])})

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is also the majority rule."""
        if type(other) is not MajorityRule:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the shared hash for majority rules."""
        return hash(MajorityRule)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this rule."""
        return "MajorityRule()"


class WeightedRule[C: Hashable]:
    """Permits every observed type, each weighted by its share of the mass."""

    __slots__ = ()

    @property
    def scope(self) -> ScopeKind:
        """Return the type-level scope."""
        return ScopeKind.FORM

    def permitted(self, masses: Mapping[C, float], /) -> frozenset[Weighted[C]]:
        """Return every observed label with its share of the total mass.

        Parameters
        ----------
        masses : Mapping[C, float]
            The mass observed on each label.

        Raises
        ------
        ValueError
            If no label was observed, or a label bears no positive mass.
        """
        observed = _observed(masses)
        total = sum(observed.values())

        return frozenset(
            Weighted(label, mass / total) for label, mass in observed.items()
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is also the weighted rule."""
        if type(other) is not WeightedRule:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the shared hash for weighted rules."""
        return hash(WeightedRule)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this rule."""
        return "WeightedRule()"


class PerOccurrenceRule[C: Hashable]:
    """Permits each occurrence's own type, bound at that occurrence alone."""

    __slots__ = ()

    @property
    def scope(self) -> ScopeKind:
        """Return the token-level scope."""
        return ScopeKind.OCCURRENCE

    def permitted(self, masses: Mapping[C, float], /) -> frozenset[C]:
        """Return every label one occurrence was observed under.

        Parameters
        ----------
        masses : Mapping[C, float]
            The mass observed on each label at one occurrence.

        Raises
        ------
        ValueError
            If no label was observed, or a label bears no positive mass.
        """
        return frozenset(_observed(masses))

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is also the per-occurrence rule."""
        if type(other) is not PerOccurrenceRule:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the shared hash for per-occurrence rules."""
        return hash(PerOccurrenceRule)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this rule."""
        return "PerOccurrenceRule()"


class SelectionPolicy[C: Hashable, L: Hashable]:
    """Which constraints a source emits from the evidence it reads.

    Parameters
    ----------
    ambiguity : AmbiguityRule[C, L]
        How a form observed under several types is resolved, and at which scope
        the result binds.
    selector : FormSelector[C]
        Where in the corpus a form must sit to be emitted. Every form by
        default.
    coverage : float
        The fraction of the selected forms, or occurrences, that are emitted.
    seed : int
        The seed the covered fraction is drawn under.

    Raises
    ------
    TypeError
        If the coverage is not a number or the seed is not an integer.
    ValueError
        If the coverage is outside the unit interval or the seed is negative.
    """

    __slots__ = ("_ambiguity", "_coverage", "_seed", "_selector")

    def __init__(
        self,
        ambiguity: AmbiguityRule[C, L],
        *,
        selector: FormSelector[C] = _EVERY_FORM,
        coverage: float = 1.0,
        seed: int = 0,
    ) -> None:
        if isinstance(coverage, bool) or not isinstance(coverage, int | float):
            msg = f"a coverage is not a number: {coverage!r}"
            raise TypeError(msg)

        if not 0.0 <= coverage <= 1.0:
            msg = f"a coverage is a fraction of what is admitted, not {coverage!r}"
            raise ValueError(msg)

        self._ambiguity = ambiguity
        self._selector = selector
        self._coverage = float(coverage)
        self._seed = ensure_non_negative("a selection seed", seed)

    @property
    def ambiguity(self) -> AmbiguityRule[C, L]:
        """Return how a form observed under several types is resolved."""
        return self._ambiguity

    @property
    def scope(self) -> ScopeKind:
        """Return the scope the ambiguity rule binds its result at."""
        return self._ambiguity.scope

    @property
    def selector(self) -> FormSelector[C]:
        """Return where in the corpus an emitted form sits."""
        return self._selector

    @property
    def coverage(self) -> float:
        """Return the fraction of the selected forms or occurrences emitted."""
        return self._coverage

    @property
    def seed(self) -> int:
        """Return the seed the covered fraction is drawn under."""
        return self._seed

    def selected(self, profile: CorpusProfile[C], /) -> frozenset[Form]:
        """Return the corpus's forms the selector admits.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the policy is applied to.
        """
        return self._selector.admitted(profile)

    def report(
        self, profile: CorpusProfile[C], covered: Mapping[Form, int], /
    ) -> SelectionReport[C]:
        """Return the budget a selection under this policy achieved.

        Parameters
        ----------
        profile : CorpusProfile[C]
            The corpus the selection was made on.
        covered : Mapping[Form, int]
            How many of each pinned form's tokens the selection covers.
        """
        return SelectionReport.of(profile, covered, self._selector)

    def with_seed(self, seed: int) -> SelectionPolicy[C, L]:
        """Return the same policy drawing its covered fraction under another seed.

        Parameters
        ----------
        seed : int
            The seed for the new policy.
        """
        return SelectionPolicy(
            self._ambiguity,
            selector=self._selector,
            coverage=self._coverage,
            seed=seed,
        )

    def _identity(
        self,
    ) -> tuple[AmbiguityRule[C, L], FormSelector[C], float, int]:
        """Return the four fields a policy is equal on."""
        return (self._ambiguity, self._selector, self._coverage, self._seed)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value selects the same way under the same seed."""
        if type(other) is not SelectionPolicy:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the rule, the selector, the coverage and the seed."""
        return hash((SelectionPolicy, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        return (
            f"SelectionPolicy({self._ambiguity!r}, selector={self._selector!r}, "
            f"coverage={self._coverage!r}, seed={self._seed!r})"
        )
