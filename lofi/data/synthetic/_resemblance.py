"""The joint a source lexicon was measured to have, and how it is resized.

A source lexicon is read as a joint of category and token mass over its
forms: how many forms bear each category, what fraction of the token mass
each category accounts for, and the exponent of the rank-frequency curve its
forms fall along. Resizing scales every class in proportion to the form
inventory, so a class keeps its share of the forms whether the inventory is
subsampled below the source's or extrapolated above it, and the fitted
exponent is what spreads a class's token share over the ranks it then has.

A source that carries logical forms is read the same way one level down. Its
items are read as a joint of logical-form class and token mass over the
logical constants they are built on, beside the arities its predicates take
and the classes the items of each category bear. Resizing holds a closed
class at the constants the source uses, since a sixth connective cannot be
invented, and gives the predicate and individual classes whatever the
declared inventory leaves them.
"""

from math import exp
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_finite, ensure_name
from lofi.data.synthetic._frequency import log_log_slope
from lofi.data.synthetic._semantics import logical_classes

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi.data.synthetic._ports import SemanticSource, SourceLexicon
    from lofi.data.synthetic._semantics import LogicalClass

__all__ = [
    "LexiconEstimate",
    "SemanticLexiconEstimate",
    "estimate",
    "semantic_estimate",
]

_CLOSED_CLASSES: Final[frozenset[str]] = frozenset(
    {"quantifier", "connective", "identity"}
)
"""The classes whose constants the fragment fixes rather than the inventory size."""


@final
class LexiconEstimate:
    """The joint a source lexicon was measured to have, beside its digest.

    Parameters
    ----------
    digest : str
        The digest of the source the joint was read from, which two
        estimates of the same joint from different sources differ on.
    rows : tuple[tuple[str, int, float], ...]
        The category text, how many of the source's forms bear it, and the
        fraction of the source's token mass it accounts for.
    forms : int
        How many forms the source holds.
    exponent : float
        The exponent fitted to the rank-frequency curve of the source's
        forms, positive where a frequent form is much more frequent than a
        rare one.

    Raises
    ------
    ValueError
        If the digest is empty, no row is given, a category has two rows, a
        class is empty or larger than the source's inventory, a share is not
        positive, the form count is not positive, or the exponent is not
        finite.
    """

    __slots__ = ("_digest", "_exponent", "_forms", "_rows")

    def __init__(
        self,
        digest: str,
        rows: tuple[tuple[str, int, float], ...],
        forms: int,
        exponent: float,
    ) -> None:
        if forms <= 0:
            msg = f"a source lexicon holds no form: {forms!r}"
            raise ValueError(msg)

        if not rows:
            msg = "an estimate of a lexicon needs a category"
            raise ValueError(msg)

        categories = [category for category, _, _ in rows]

        if len(set(categories)) != len(categories):
            msg = f"an estimate names a category twice: {categories!r}"
            raise ValueError(msg)

        for category, size, share in rows:
            ensure_name("an estimated category", category)

            if size <= 0 or size > forms:
                msg = (
                    f"category {category!r} bears {size} of the source's {forms} forms"
                )
                raise ValueError(msg)

            if ensure_finite("an estimated share", share) <= 0.0:
                msg = f"category {category!r} carries no token mass: {share!r}"
                raise ValueError(msg)

        self._digest = ensure_name("a source digest", digest)
        self._rows = rows
        self._forms = forms
        self._exponent = ensure_finite("an estimated exponent", exponent)

    @property
    def digest(self) -> str:
        """Return the digest of the source the joint was read from."""
        return self._digest

    @property
    def rows(self) -> tuple[tuple[str, int, float], ...]:
        """Return each category with its class size and its token share."""
        return self._rows

    @property
    def forms(self) -> int:
        """Return how many forms the source holds."""
        return self._forms

    @property
    def exponent(self) -> float:
        """Return the exponent fitted to the source's rank-frequency curve."""
        return self._exponent

    def at(self, forms: int) -> tuple[tuple[str, int, float], ...]:
        """Return the joint scaled to a form inventory of a given size.

        Each class size is scaled by the ratio of the given inventory to the
        source's and rounded. A class that rounds to nothing is dropped, so
        the categories at a smaller inventory are a subset of the source's,
        and the shares of the categories that remain are relative.

        Parameters
        ----------
        forms : int
            How many forms the scaled inventory holds.

        Returns
        -------
        tuple[tuple[str, int, float], ...]
            The category text, its scaled class size and its token share, in
            the order the estimate holds them.

        Raises
        ------
        ValueError
            If the inventory is not positive, or is too small for any class
            of the source to survive the scaling.
        """
        if forms <= 0:
            msg = f"a scaled inventory holds no form: {forms!r}"
            raise ValueError(msg)

        ratio = forms / self._forms
        kept = tuple(
            (category, round(size * ratio), share)
            for category, size, share in self._rows
            if round(size * ratio) > 0
        )

        if not kept:
            msg = (
                f"an inventory of {forms} forms is too small to resemble a "
                f"source of {self._forms}, whose largest class bears "
                f"{max(size for _, size, _ in self._rows)}"
            )
            raise ValueError(msg)

        return kept

    def _fields(self) -> tuple[str, tuple[tuple[str, int, float], ...], int, float]:
        """Return the fields the estimate is equal on."""
        return (self._digest, self._rows, self._forms, self._exponent)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another estimate read the same joint from the same source.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether the digest, the rows, the form count and the exponent
            agree.
        """
        if type(other) is not LexiconEstimate:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the digest, the rows, the form count and the exponent."""
        return hash((LexiconEstimate, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this estimate."""
        return (
            f"LexiconEstimate({self._digest!r}, {self._rows!r}, "
            f"{self._forms!r}, {self._exponent!r})"
        )


def estimate[T](
    source: SourceLexicon[T],
    render: Callable[[T], str],
    digest: str,
    /,
) -> LexiconEstimate:
    """Return the joint of category and token mass a source lexicon holds.

    A category's class size is how many forms bear it and its token share is
    the mass those forms carry at it, relative to the mass of the whole
    lexicon. The exponent is the negated least-squares slope of log mass on
    log rank over the forms ranked by mass. The masses are read relative to
    the heaviest form, which neither the shares nor the slope depend on.

    Parameters
    ----------
    source : SourceLexicon[T]
        The lexicon to read.
    render : Callable[[T], str]
        The text a category is recorded under, which must parse back to that
        category in the syntax a lexicon drawn from the estimate is read in.
    digest : str
        The digest of the source, recorded so that two sources are told
        apart by the estimate alone.

    Returns
    -------
    LexiconEstimate
        The joint, with its categories ordered by descending share and then
        by text.

    Raises
    ------
    ValueError
        If the source holds no form, holds a form no category bears, or
        holds too few forms for a rank-frequency curve to be fitted.
    """
    forms = source.forms

    if not forms:
        msg = "an estimate of a lexicon needs a form"
        raise ValueError(msg)

    top = max(source.log_mass(form) for form in forms)
    masses = {form: exp(source.log_mass(form) - top) for form in forms}
    bearers: dict[str, set[str]] = {}
    carried: dict[str, float] = {}

    for form in forms:
        held = source.entries(form)

        if not held:
            msg = f"the source holds the form {form!r} under no category"
            raise ValueError(msg)

        for entry in held:
            category = render(entry.category)
            bearers.setdefault(category, set()).add(form)
            carried[category] = carried.get(category, 0.0) + masses[form] * exp(
                entry.weight
            )

    total = sum(carried.values())
    ranked = sorted(masses.values(), reverse=True)
    slope = log_log_slope(
        tuple(float(rank) for rank in range(1, len(ranked) + 1)), ranked
    )

    if slope is None:
        msg = f"a rank-frequency curve needs two forms, not {len(forms)}"
        raise ValueError(msg)

    rows = tuple(
        (category, len(bearers[category]), carried[category] / total)
        for category in sorted(carried, key=lambda name: (-carried[name], name))
    )

    return LexiconEstimate(digest, rows, len(forms), -slope)


def _apportion(total: int, weights: Sequence[int], /) -> tuple[int, ...]:
    """Return a total split over weights by largest remainder.

    Parameters
    ----------
    total : int
        The number to split.
    weights : Sequence[int]
        The positive weight of each part.

    Returns
    -------
    tuple[int, ...]
        One count per weight, summing to the total, an earlier weight
        winning a tie for a spare unit.
    """
    mass = sum(weights)
    quotas = [total * weight / mass for weight in weights]
    held = [int(quota) for quota in quotas]
    order = sorted(range(len(weights)), key=lambda at: (held[at] - quotas[at], at))

    for at in order[: total - sum(held)]:
        held[at] += 1

    return tuple(held)


@final
class SemanticLexiconEstimate:
    """The logical forms a source lexicon was measured to carry, beside its digest.

    Parameters
    ----------
    digest : str
        The digest of the source the joint was read from, which two
        estimates of the same joint from different sources differ on.
    rows : tuple[tuple[LogicalClass, int, float], ...]
        The logical-form class, how many of the source's logical constants
        belong to it, and the fraction of the source's token mass its items
        carry.
    arities : tuple[tuple[int, int], ...]
        Each arity the source's predicates take, with how many predicate
        constants bear it.
    assignment : tuple[tuple[str, LogicalClass, int | None, int], ...]
        The category text, the class its items bear, the arity when that
        class is the predicate class, and how many of the source's items
        bore the two together.
    quantifiers : tuple[str, ...]
        The quantifiers the source's quantifier forms are built on.
    connectives : tuple[str, ...]
        The connectives the source's connective forms are built on.
    exponent : float
        The exponent fitted to the rank-frequency curve of the source's
        logical constants, positive where a frequent constant is much more
        frequent than a rare one.

    Raises
    ------
    ValueError
        If the digest is empty, no row is given, a class is unknown or has
        two rows, a class is empty or carries no token mass, a closed class
        is sized apart from the constants it names, the arity rows do not
        place the predicate constants, no item is assigned, an assignment
        row names a class outside the rows or an arity outside the arity
        rows, a predicate row forces no arity or another row forces one, or
        the exponent is not finite.
    """

    __slots__ = (
        "_arities",
        "_assignment",
        "_connectives",
        "_digest",
        "_exponent",
        "_quantifiers",
        "_rows",
    )

    def __init__(
        self,
        digest: str,
        rows: tuple[tuple[LogicalClass, int, float], ...],
        arities: tuple[tuple[int, int], ...],
        assignment: tuple[tuple[str, LogicalClass, int | None, int], ...],
        *,
        quantifiers: tuple[str, ...] = (),
        connectives: tuple[str, ...] = (),
        exponent: float,
    ) -> None:
        sizes = _class_sizes(rows)
        _check_closed(sizes, quantifiers=quantifiers, connectives=connectives)
        _check_arities(sizes.get("predicate"), arities)
        _check_assignment(assignment, sizes, arities)

        self._digest = ensure_name("a source digest", digest)
        self._rows = rows
        self._arities = arities
        self._assignment = assignment
        self._quantifiers = quantifiers
        self._connectives = connectives
        self._exponent = ensure_finite("an estimated exponent", exponent)

    @property
    def digest(self) -> str:
        """Return the digest of the source the joint was read from."""
        return self._digest

    @property
    def rows(self) -> tuple[tuple[LogicalClass, int, float], ...]:
        """Return each class with its constant count and its token share."""
        return self._rows

    @property
    def arities(self) -> tuple[tuple[int, int], ...]:
        """Return each arity with how many predicate constants bear it."""
        return self._arities

    @property
    def assignment(self) -> tuple[tuple[str, LogicalClass, int | None, int], ...]:
        """Return each category, class and arity with how many items bore them."""
        return self._assignment

    @property
    def quantifiers(self) -> tuple[str, ...]:
        """Return the quantifiers the source's quantifier forms are built on."""
        return self._quantifiers

    @property
    def connectives(self) -> tuple[str, ...]:
        """Return the connectives the source's connective forms are built on."""
        return self._connectives

    @property
    def exponent(self) -> float:
        """Return the exponent fitted to the source's rank-frequency curve."""
        return self._exponent

    @property
    def constants(self) -> int:
        """Return how many logical constants the source's inventory holds."""
        return sum(size for _, size, _ in self._rows)

    def at(self, constants: int) -> tuple[tuple[LogicalClass, int, float], ...]:
        """Return the joint scaled to an inventory of a given number of constants.

        Every class is first apportioned the share of the inventory its size
        in the source earns it. A closed class is then held at the constants
        the source names, since the fragment has no further quantifier,
        connective or identity to give it, and the constants that holding
        frees are apportioned again over the predicate and individual
        classes. A class apportioned nothing is dropped, so the classes at a
        smaller inventory are a subset of the source's.

        Parameters
        ----------
        constants : int
            How many logical constants the scaled inventory holds.

        Returns
        -------
        tuple[tuple[LogicalClass, int, float], ...]
            The class, its scaled constant count and its token share, in the
            order the estimate holds them. The shares are the source's and
            are relative.

        Raises
        ------
        ValueError
            If the inventory is not positive, or is larger than the source's
            closed classes can fill and the source has no open class.
        """
        if constants <= 0:
            msg = f"a scaled inventory holds no constant: {constants!r}"
            raise ValueError(msg)

        sizes = tuple(size for _, size, _ in self._rows)
        held = list(_apportion(constants, sizes))

        for at, (logical_class, size, _) in enumerate(self._rows):
            if logical_class in _CLOSED_CLASSES:
                held[at] = min(held[at], size)

        spare = constants - sum(held)
        opened = tuple(
            at
            for at, (logical_class, _, _) in enumerate(self._rows)
            if logical_class not in _CLOSED_CLASSES
        )

        if spare > 0:
            if not opened:
                msg = (
                    f"an inventory of {constants} constants cannot resemble a "
                    f"source whose {sum(held)} constants are all closed classes"
                )
                raise ValueError(msg)

            for at, extra in zip(
                opened,
                _apportion(spare, tuple(sizes[at] for at in opened)),
                strict=True,
            ):
                held[at] += extra

        return tuple(
            (logical_class, held[at], share)
            for at, (logical_class, _, share) in enumerate(self._rows)
            if held[at] > 0
        )

    def _fields(
        self,
    ) -> tuple[
        str,
        tuple[tuple[LogicalClass, int, float], ...],
        tuple[tuple[int, int], ...],
        tuple[tuple[str, LogicalClass, int | None, int], ...],
        tuple[str, ...],
        tuple[str, ...],
        float,
    ]:
        """Return the fields the estimate is equal on."""
        return (
            self._digest,
            self._rows,
            self._arities,
            self._assignment,
            self._quantifiers,
            self._connectives,
            self._exponent,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another estimate read the same joint from the same source.

        Parameters
        ----------
        other : object
            The value to compare.

        Returns
        -------
        bool
            Whether the digest, the rows, the arities, the assignment, the
            logical constants and the exponent agree.
        """
        if type(other) is not SemanticLexiconEstimate:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields the estimate is equal on."""
        return hash((SemanticLexiconEstimate, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this estimate."""
        return (
            f"SemanticLexiconEstimate({self._digest!r}, {self._rows!r}, "
            f"{self._arities!r}, {self._assignment!r}, "
            f"quantifiers={self._quantifiers!r}, "
            f"connectives={self._connectives!r}, exponent={self._exponent!r})"
        )


def _class_sizes(
    rows: tuple[tuple[LogicalClass, int, float], ...],
) -> dict[LogicalClass, int]:
    """Return the constant count of each class, refusing a joint it cannot size."""
    if not rows:
        msg = "an estimate of a semantic lexicon needs a class"
        raise ValueError(msg)

    sizes: dict[LogicalClass, int] = {}
    classes = logical_classes()

    for logical_class, size, share in rows:
        if logical_class not in classes:
            msg = (
                f"an estimated class is not installed among {classes}: "
                f"{logical_class!r}"
            )
            raise ValueError(msg)

        if logical_class in sizes:
            msg = f"an estimate names the {logical_class} class twice"
            raise ValueError(msg)

        if size <= 0:
            msg = f"the {logical_class} class holds no constant: {size!r}"
            raise ValueError(msg)

        if ensure_finite("an estimated share", share) <= 0.0:
            msg = f"the {logical_class} class carries no token mass: {share!r}"
            raise ValueError(msg)

        sizes[logical_class] = size

    if sizes.get("identity", 1) != 1:
        msg = f"the identity class holds one constant, not {sizes['identity']}"
        raise ValueError(msg)

    return sizes


def _check_closed(
    sizes: dict[LogicalClass, int],
    *,
    quantifiers: tuple[str, ...],
    connectives: tuple[str, ...],
) -> None:
    """Refuse a closed class sized apart from the constants it names."""
    for logical_class, names in (
        ("quantifier", quantifiers),
        ("connective", connectives),
    ):
        held = tuple(
            ensure_name(f"an estimated {logical_class}", name) for name in names
        )

        if len(set(held)) != len(held):
            msg = f"an estimate names a {logical_class} twice: {held!r}"
            raise ValueError(msg)

        if sizes.get(logical_class, 0) != len(held):
            msg = (
                f"the {logical_class} class holds "
                f"{sizes.get(logical_class, 0)} constants but the estimate "
                f"names {len(held)}: {list(held)!r}"
            )
            raise ValueError(msg)


def _check_arities(
    predicates: int | None, arities: tuple[tuple[int, int], ...]
) -> None:
    """Refuse arity rows that do not place the source's predicate constants."""
    seen: set[int] = set()

    for arity, count in arities:
        if arity <= 0:
            msg = f"an estimated arity is not positive: {arity!r}"
            raise ValueError(msg)

        if arity in seen:
            msg = f"an estimate names the arity {arity} twice"
            raise ValueError(msg)

        if count <= 0:
            msg = f"no predicate constant bears the estimated arity {arity}"
            raise ValueError(msg)

        seen.add(arity)

    placed = sum(count for _, count in arities)

    if placed != (predicates or 0):
        msg = (
            f"the arity rows place {placed} predicate constants but the "
            f"predicate class holds {predicates or 0}"
        )
        raise ValueError(msg)


def _check_assignment(
    assignment: tuple[tuple[str, LogicalClass, int | None, int], ...],
    sizes: dict[LogicalClass, int],
    arities: tuple[tuple[int, int], ...],
) -> None:
    """Refuse an assignment the joint cannot bear."""
    if not assignment:
        msg = "an estimate of a semantic lexicon needs an assigned item"
        raise ValueError(msg)

    placed = {arity for arity, _ in arities}
    seen: set[tuple[str, LogicalClass, int | None]] = set()

    for category, logical_class, arity, items in assignment:
        ensure_name("an assigned category", category)

        if (category, logical_class, arity) in seen:
            msg = (
                f"an estimate assigns category {category!r} the "
                f"{logical_class} class at arity {arity!r} twice"
            )
            raise ValueError(msg)

        if logical_class not in sizes:
            msg = (
                f"category {category!r} is assigned the {logical_class} "
                f"class, which the estimate does not size"
            )
            raise ValueError(msg)

        if (arity is None) == (logical_class == "predicate"):
            msg = (
                f"category {category!r} is assigned the {logical_class} "
                f"class at arity {arity!r}, and only a predicate bears one"
            )
            raise ValueError(msg)

        if arity is not None and arity not in placed:
            msg = f"category {category!r} is assigned the unplaced arity {arity}"
            raise ValueError(msg)

        if items <= 0:
            msg = f"no item of category {category!r} bears the {logical_class} class"
            raise ValueError(msg)

        seen.add((category, logical_class, arity))


def semantic_estimate(
    source: SemanticSource, digest: str, /
) -> SemanticLexiconEstimate:
    """Return the logical forms a source lexicon was measured to carry.

    A class's size is how many distinct logical constants its forms are built
    on and its token share is the mass of the items bearing it, relative to
    the mass of every item. The exponent is the negated least-squares slope
    of log mass on log rank over the constants ranked by the mass of the
    items built on them. The masses are read relative to the heaviest item,
    which neither the shares nor the slope depend on.

    Parameters
    ----------
    source : SemanticSource
        The lexicon to read.
    digest : str
        The digest of the source, recorded so that two sources are told
        apart by the estimate alone.

    Returns
    -------
    SemanticLexiconEstimate
        The joint, with the classes in report order, the arities ascending
        and the assignment ordered by category, class and arity.

    Raises
    ------
    ValueError
        If the source holds no item, holds an item bearing no logical form
        or no token mass, or holds too few logical constants for a
        rank-frequency curve to be fitted.
    """
    items = source.items

    if not items:
        msg = "an estimate of a semantic lexicon needs an item"
        raise ValueError(msg)

    top = max(source.log_mass(item) for item in items)
    symbols: dict[LogicalClass, set[str]] = {}
    carried: dict[LogicalClass, float] = {}
    borne: dict[int, set[str]] = {}
    counted: dict[tuple[str, LogicalClass, int | None], int] = {}
    mass: dict[tuple[LogicalClass, str], float] = {}

    for item in items:
        meaning = source.meaning(item)

        if meaning is None:
            msg = f"the source holds the item {item!r} under no logical form"
            raise ValueError(msg)

        held = exp(source.log_mass(item) - top)

        if held <= 0.0:
            msg = f"the source holds the item {item!r} with no token mass"
            raise ValueError(msg)

        logical_class = meaning.logical_class
        category, _ = item
        symbols.setdefault(logical_class, set()).add(meaning.symbol)
        carried[logical_class] = carried.get(logical_class, 0.0) + held
        mass[logical_class, meaning.symbol] = (
            mass.get((logical_class, meaning.symbol), 0.0) + held
        )
        counted[category, logical_class, meaning.arity] = (
            counted.get((category, logical_class, meaning.arity), 0) + 1
        )

        if meaning.arity is not None:
            borne.setdefault(meaning.arity, set()).add(meaning.symbol)

    total = sum(carried.values())
    ranked = sorted(mass.values(), reverse=True)
    slope = log_log_slope(
        tuple(float(rank) for rank in range(1, len(ranked) + 1)), ranked
    )

    if slope is None:
        msg = f"a rank-frequency curve needs two logical constants, not {len(ranked)}"
        raise ValueError(msg)

    classes = logical_classes()

    return SemanticLexiconEstimate(
        digest,
        tuple(
            (logical_class, len(symbols[logical_class]), carried[logical_class] / total)
            for logical_class in classes
            if logical_class in symbols
        ),
        tuple((arity, len(borne[arity])) for arity in sorted(borne)),
        tuple(
            (category, logical_class, arity, counted[category, logical_class, arity])
            for category, logical_class, arity in sorted(
                counted,
                key=lambda key: (key[0], classes.index(key[1]), key[2] or 0),
            )
        ),
        quantifiers=tuple(sorted(symbols.get("quantifier", ()))),
        connectives=tuple(sorted(symbols.get("connective", ()))),
        exponent=-slope,
    )
