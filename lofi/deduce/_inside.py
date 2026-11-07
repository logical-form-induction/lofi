"""Inside weights per exact category, kept without derivation steps."""

from typing import TYPE_CHECKING, Protocol, override

from lofi._native import ChartProgram as _ChartProgram
from lofi._native import InsideTable as _InsideTable

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi.deduce._ports import Semiring
    from lofi.deduce._spans import Cell, SpanIndex

__all__ = ["InsideChart"]

type _NativeItem = tuple[int, int]
type _NativeEdge = tuple[_NativeItem, int, _NativeItem]
type _NativeClosure = tuple[list[_NativeEdge], list[_NativeItem], bool]


class _Identifiers[T](Protocol):
    """Dense identifiers for values Python compares exactly."""

    def identify(self, value: T, /) -> int | None:
        """Return a value's identifier, or refuse it under a declared bound."""

    def lookup(self, value: T, /) -> int | None:
        """Return a value's identifier when it has been registered."""

    def value(self, identifier: int, /) -> T:
        """Return the value at one identifier."""


class _CategoryIdentifiers[T](_Identifiers[T], Protocol):
    """Dense category identifiers that report bound refusals."""

    @property
    def refused_categories(self) -> int:
        """Return how many exact categories a declared bound refused."""


class _ValueTable[T]:
    """Mint dense identifiers while Python decides exact equality."""

    __slots__ = ("_ids", "_values")

    def __init__(self) -> None:
        self._ids: dict[T, int] = {}
        self._values: list[T] = []

    def identify(self, value: T) -> int:
        """Return one value's dense identifier, registering it once."""
        identifier = self.lookup(value)

        if identifier is not None:
            return identifier

        identifier = len(self._values)

        try:
            self._ids[value] = identifier
        except TypeError as error:
            msg = f"an inside-chart value is not hashable: {value!r}"
            raise TypeError(msg) from error

        self._values.append(value)
        return identifier

    @property
    def refused_categories(self) -> int:
        """Return zero because an unconstrained value table refuses nothing."""
        return 0

    def lookup(self, value: T) -> int | None:
        """Return one registered value's identifier."""
        try:
            return self._ids.get(value)
        except TypeError as error:
            msg = f"an inside-chart value is not hashable: {value!r}"
            raise TypeError(msg) from error

    def value(self, identifier: int) -> T:
        """Return the value at one dense identifier."""
        return self._values[identifier]


class InsideChart[T, W]:
    """Stores the inside weight of every exact category over every span.

    A category can be derived under several normal form tags. The chart keeps
    a weight per tag while it is filled, since a normal form reads the tag of
    each premise, and reports a category's weight as the semiring sum over its
    tags. No derivation step is kept, so a weight cannot be unpacked into the
    derivations it sums.

    Parameters
    ----------
    index : SpanIndex
        Span index for the input.
    semiring : Semiring[W]
        Algebra for inside weights.
    normal_form : str
        The name of the normal form whose steps the weights sum over.
    """

    __slots__ = (
        "_categories",
        "_index",
        "_normal_form",
        "_semiring",
        "_table",
        "_tags",
    )

    def __init__(
        self,
        index: SpanIndex,
        semiring: Semiring[W],
        *,
        normal_form: str = "none",
        _identifiers: tuple[_CategoryIdentifiers[T], _Identifiers[str]] | None = None,
    ) -> None:
        self._index = index
        self._semiring: Semiring[W] = semiring
        self._normal_form = normal_form
        if _identifiers is None:
            self._categories = _ValueTable()
            self._tags = _ValueTable()
        else:
            self._categories, self._tags = _identifiers
        self._table = _InsideTable(index.n, semiring, semiring.lowering)

    @property
    def index(self) -> SpanIndex:
        """Return the chart's span index."""
        return self._index

    @property
    def semiring(self) -> Semiring[W]:
        """Return the semiring used for inside weights."""
        return self._semiring

    @property
    def normal_form(self) -> str:
        """Return the name of the normal form whose steps the weights sum over.

        Under any normal form but ``"none"`` a category's weight sums over its
        normal form derivations only, so ``Counting`` counts those.
        """
        return self._normal_form

    @property
    def truncated(self) -> bool:
        """Return whether bounded construction omitted a derivation."""
        return self._table.truncated or self.refused_categories > 0

    @property
    def refused_categories(self) -> int:
        """Return how many exact categories the declared bounds refused."""
        return self._categories.refused_categories

    @property
    def num_categories(self) -> int:
        """Return how many exact categories the chart holds, over every span."""
        return self._table.num_categories

    def mark_truncated(self) -> None:
        """Record that bounded construction omitted a derivation."""
        self._table.mark_truncated()

    def add(self, span: Cell, category: T, tag: str, weight: W, /) -> None:
        """Add one derivation's weight to a category under one tag.

        Parameters
        ----------
        span : Cell
            Half-open token span.
        category : T
            The exact category derived.
        tag : str
            The normal form tag the derivation gives it.
        weight : W
            The derivation's weight.

        Raises
        ------
        ValueError
            If the span falls outside the span index.
        """
        category_id = self._categories.identify(category)

        if category_id is None:
            return

        tag_id = self._tags.identify(tag)

        if tag_id is None:
            msg = f"normal-form tag {tag!r} was refused by its identifier table"
            raise ValueError(msg)

        item = (category_id, tag_id)
        self._table.add(span, item, weight)

    def restrict(self, span: Cell, keep: Iterable[tuple[T, str]], /) -> None:
        """Drop every category and tag at one span but the ones named.

        The chart keeps no steps, so nothing else changes. The drop is not
        recorded as truncation: a caller drops only what no root derivation
        uses.

        Parameters
        ----------
        span : Cell
            Half-open token span.
        keep : Iterable[tuple[T, str]]
            The categories and tags to keep.
        """
        identifiers: list[tuple[int, int]] = []

        for category, tag in keep:
            category_id = self._categories.lookup(category)
            tag_id = self._tags.lookup(tag)

            if category_id is not None and tag_id is not None:
                identifiers.append((category_id, tag_id))

        self._table.restrict(span, identifiers)

    def tagged_at(self, span: Cell, /) -> tuple[tuple[T, str, W], ...]:
        """Return every category, tag and weight over one span, in arrival order.

        Parameters
        ----------
        span : Cell
            Half-open token span.
        """
        categories = self._categories
        tags = self._tags
        return tuple(
            (categories.value(category), tags.value(tag), weight)
            for category, tag, weight in self._table.tagged_at(span)
        )

    def categories_at(self, span: Cell, /) -> tuple[tuple[T, W], ...]:
        """Return the exact categories derived over one span, with their weights.

        Parameters
        ----------
        span : Cell
            Half-open token span.

        Returns
        -------
        tuple[tuple[T, W], ...]
            Each category once, with the semiring sum of its weight over its
            tags, in the order the category arrived.
        """
        categories = self._categories
        return tuple(
            (categories.value(category), weight)
            for category, weight in self._table.categories_at(span)
        )

    def tagged(self, span: Cell, category: T, tag: str, /) -> W:
        """Return one category's weight over one span under one tag.

        Parameters
        ----------
        span : Cell
            Half-open token span.
        category : T
            The exact category.
        tag : str
            The normal form tag.

        Returns
        -------
        W
            The weight, or zero where the category has no derivation there
            under that tag.
        """
        category_id = self._categories.lookup(category)
        tag_id = self._tags.lookup(tag)

        if category_id is None or tag_id is None:
            return self._semiring.zero

        return self._table.tagged(span, (category_id, tag_id))

    def inside(self, span: Cell, category: T, /) -> W:
        """Return one category's weight over one span.

        Parameters
        ----------
        span : Cell
            Half-open token span.
        category : T
            The exact category.

        Returns
        -------
        W
            The semiring sum over its tags, or zero where the category is not
            derived there.
        """
        category_id = self._categories.lookup(category)

        if category_id is None:
            return self._semiring.zero

        return self._table.inside(span, category_id)

    def sweep(
        self,
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
        program: _ChartProgram,
        resolver: object,
        *,
        scoring: object,
    ) -> _NativeClosure:
        """Fill this table through the native drawing and accumulation kernel."""
        return self._table.sweep(
            active_arities,
            unary_depth,
            threads,
            program,
            resolver,
            scoring=scoring,
        )

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the input length and the category count."""
        return f"InsideChart(n={self._index.n}, categories={self.num_categories})"
