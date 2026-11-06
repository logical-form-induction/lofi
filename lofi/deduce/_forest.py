"""Packed derivation forests."""

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, override

from lofi._native import ChartProgram as _ChartProgram
from lofi._native import ForestTable as _ForestTable
from lofi.deduce._inside import _CategoryIdentifiers, _Identifiers, _ValueTable
from lofi.deduce._values import Derivation, Edge, EdgeProduct, ItemKey

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping

    from lofi.deduce._ports import Semiring
    from lofi.deduce._spans import Cell, SpanIndex
    from lofi.deduce._values import PrunedMass

__all__ = ["PackedForest"]

type _NativeItem = tuple[int, int]
type _NativeKey = tuple[tuple[int, int], _NativeItem]
type _NativeEdge[W] = tuple[
    int,
    list[_NativeKey],
    int,
    W,
    int | None,
    int,
]
type _NativeProduct[W] = tuple[
    int,
    list[list[_NativeKey]],
    int,
    W,
    int | None,
    int,
]
type _ForestRows[T, W] = tuple[tuple[ItemKey[T], tuple[EdgeProduct[T, W], ...]], ...]


class _ForestState[T, W](NamedTuple):
    """Fields needed to reconstruct one packed forest."""

    index: SpanIndex
    semiring: Semiring[W]
    normal_form: str
    rows: _ForestRows[T, W]
    truncated: bool
    pruned: tuple[tuple[Cell, W], ...]
    pruned_mass: PrunedMass | None


type _ForestReduction[T, W] = tuple[
    Callable[[_ForestState[T, W]], PackedForest[T, W]],
    tuple[_ForestState[T, W]],
]


def _restore_forest[T, W](state: _ForestState[T, W]) -> PackedForest[T, W]:
    """Reconstruct one packed forest through its native storage boundary."""
    forest = PackedForest(state.index, state.semiring, normal_form=state.normal_form)

    for key, products in state.rows:
        for product in products:
            forest.add_product(key, product)

    if state.truncated:
        forest.mark_truncated()

    for span, weight in state.pruned:
        forest.record_pruned(span, weight)

    if state.pruned_mass is not None:
        forest.record_pruned_mass(state.pruned_mass)

    return forest


class PackedForest[T, W]:
    """Stores items, inside weights, and derivation edges.

    Parameters
    ----------
    index : SpanIndex
        Span index for the input.
    semiring : Semiring[W]
        Algebra for inside weights.
    normal_form : str
        The name of the normal form whose steps the forest holds.
    """

    __slots__ = (
        "_categories",
        "_index",
        "_labels",
        "_normal_form",
        "_pruned",
        "_pruned_mass",
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
        _identifiers: tuple[
            _CategoryIdentifiers[T], _Identifiers[str], _Identifiers[str]
        ]
        | None = None,
    ) -> None:
        self._index = index
        self._normal_form = normal_form
        self._semiring: Semiring[W] = semiring
        if _identifiers is None:
            self._categories = _ValueTable()
            self._tags = _ValueTable()
            self._labels = _ValueTable()
        else:
            self._categories, self._tags, self._labels = _identifiers
        self._table = _ForestTable(index.n, semiring, semiring.lowering)
        self._pruned: dict[Cell, W] = {}
        self._pruned_mass: PrunedMass | None = None

    @property
    def index(self) -> SpanIndex:
        """Return the forest's span index."""
        return self._index

    @property
    def semiring(self) -> Semiring[W]:
        """Return the semiring used for inside weights."""
        return self._semiring

    @property
    def normal_form(self) -> str:
        """Return the name of the normal form whose steps the forest holds.

        Under any normal form but ``"none"`` an item's inside weight sums over
        its normal form derivations only, so ``Counting`` counts those.
        """
        return self._normal_form

    @property
    def num_items(self) -> int:
        """Return how many items the forest holds, over every span."""
        return self._table.num_items

    @property
    def num_edges(self) -> int:
        """Return how many steps the forest holds, over every item."""
        return self._table.num_edges

    @property
    def truncated(self) -> bool:
        """Return whether bounded construction omitted a derivation."""
        return self._table.truncated or self.refused_categories > 0

    @property
    def refused_categories(self) -> int:
        """Return how many exact categories the declared bounds refused."""
        return self._categories.refused_categories

    def mark_truncated(self) -> None:
        """Record that bounded construction omitted a derivation."""
        self._table.mark_truncated()

    @property
    def pruned(self) -> Mapping[Cell, W]:
        """Return, per pruned cell, the summed weight of the items pruned from it.

        Each weight is the semiring sum of the inside weights the dropped items
        held when their cell closed, so it is in the forest's own algebra: a
        log mass under ``log-inside`` and the best dropped log weight under
        ``viterbi``. A cell the pruner left whole has no entry.
        """
        return MappingProxyType(self._pruned)

    @property
    def pruned_mass(self) -> PrunedMass | None:
        """Return the goal mass pruning removed, or None where it was not measured."""
        return self._pruned_mass

    def record_pruned(self, span: Cell, weight: W, /) -> None:
        """Add the weight of items a pruner removed from one cell.

        Raises
        ------
        ValueError
            If the span falls outside the forest span index.
        """
        if not self._valid(span):
            msg = (
                f"pruned span {span!r} is outside the forest span index "
                f"for {self._index.n} tokens"
            )
            raise ValueError(msg)

        held = self._pruned.get(span)
        self._pruned[span] = (
            weight if held is None else self._semiring.plus(held, weight)
        )

    def record_pruned_mass(self, mass: PrunedMass, /) -> None:
        """Record the goal mass an exhaustive reparse found pruning removed."""
        self._pruned_mass = mass

    def add(self, key: ItemKey[T], edge: Edge[T, W], /) -> None:
        """Record one derivation of one item.

        Raises
        ------
        ValueError
            If the concluded item falls outside the span index.
        """
        if not self._valid(key.span):
            msg = (
                f"item span {key.span!r} is outside the forest span index "
                f"for {self._index.n} tokens"
            )
            raise ValueError(msg)

        categories = self._categories
        tags = self._tags
        category = categories.identify(key.reading)
        output = categories.identify(edge.output)
        tag = tags.identify(key.nf_tag)

        if category is None or output is None:
            return

        if tag is None:
            msg = f"normal-form tag {key.nf_tag!r} was refused by its identifier table"
            raise ValueError(msg)

        label = self._labels.identify(edge.rule_name)

        if label is None:
            msg = f"edge label {edge.rule_name!r} was refused by its identifier table"
            raise ValueError(msg)

        native_key = (key.span, (category, tag))
        children: list[_NativeKey] = []

        for child in edge.children:
            child_category = categories.identify(child.reading)
            child_tag = tags.identify(child.nf_tag)

            if child_category is None:
                return

            if child_tag is None:
                msg = (
                    f"normal-form tag {child.nf_tag!r} was refused by its "
                    "identifier table"
                )
                raise ValueError(msg)

            children.append((child.span, (child_category, child_tag)))
        self._table.add(
            native_key,
            (
                label,
                children,
                output,
                edge.weight,
                edge.split,
                edge.order,
            ),
        )

    def add_product(self, key: ItemKey[T], product: EdgeProduct[T, W], /) -> None:
        """Record one exact Cartesian product of derivations."""
        if not self._valid(key.span):
            msg = (
                f"item span {key.span!r} is outside the forest span index "
                f"for {self._index.n} tokens"
            )
            raise ValueError(msg)

        categories = self._categories
        tags = self._tags
        category = categories.identify(key.reading)
        output = categories.identify(product.output)
        tag = tags.identify(key.nf_tag)

        if category is None or output is None:
            return

        if tag is None:
            msg = f"normal-form tag {key.nf_tag!r} was refused by its identifier table"
            raise ValueError(msg)

        label = self._labels.identify(product.rule_name)

        if label is None:
            msg = (
                f"edge label {product.rule_name!r} was refused by its identifier table"
            )
            raise ValueError(msg)

        axes: list[list[_NativeKey]] = []

        for axis in product.axes:
            encoded: list[_NativeKey] = []

            for child in axis:
                child_category = categories.identify(child.reading)
                child_tag = tags.identify(child.nf_tag)

                if child_category is None:
                    return

                if child_tag is None:
                    msg = (
                        f"normal-form tag {child.nf_tag!r} was refused by its "
                        "identifier table"
                    )
                    raise ValueError(msg)

                encoded.append((child.span, (child_category, child_tag)))

            axes.append(encoded)

        self._table.add_product(
            (key.span, (category, tag)),
            (
                label,
                axes,
                output,
                product.weight,
                product.split,
                product.order,
            ),
        )

    def recompute_inside(self, keys: Iterable[ItemKey[T]], /) -> None:
        """Recompute inside weights in the supplied dependency order."""
        native = [encoded for key in keys if (encoded := self._lookup_key(key))]
        self._table.recompute_inside(native)

    def restrict(
        self, span: Cell, keep: Iterable[ItemKey[T]], /, *, truncates: bool = True
    ) -> None:
        """Drop every item at one span but the ones named.

        A step of a kept item that used a dropped one goes with it, and an item
        left with no step at all is dropped in turn. The forest records a
        truncating restriction, such as a beam, but not root restriction.

        Raises
        ------
        ValueError
            If an item to keep is not at the span.
        """
        present = self.items_at(span)

        if not present:
            return

        kept = tuple(keep)
        available = set(present)
        unknown = tuple(key for key in kept if key not in available)

        if unknown:
            names = ", ".join(repr(key) for key in unknown)
            msg = f"span {span!r} holds no item {names}"
            raise ValueError(msg)

        native = [encoded for key in kept if (encoded := self._lookup_key(key))]
        self._table.restrict(span, native, truncates=truncates)

    def items_at(self, span: Cell, /) -> tuple[ItemKey[T], ...]:
        """Return the items covering one span, in arrival order."""
        if not self._valid(span):
            return ()

        return tuple(
            self._decode_key((span, item)) for item in self._table.items_at(span)
        )

    def categories_at(self, span: Cell, /) -> tuple[tuple[T, W], ...]:
        """Return exact categories and their weights summed over tags."""
        if not self._valid(span):
            return ()

        categories = self._categories
        return tuple(
            (categories.value(category), weight)
            for category, weight in self._table.categories_at(span)
        )

    def inside(self, key: ItemKey[T], /) -> W:
        """Return an item's weight, or zero when the item is absent."""
        native = self._lookup_key(key)
        return self._semiring.zero if native is None else self._table.inside(native)

    def edges(self, key: ItemKey[T], /) -> tuple[Edge[T, W], ...]:
        """Return every step that concluded an item, in arrival order."""
        native = self._lookup_key(key)

        if native is None:
            return ()

        return tuple(self._decode_edge(edge) for edge in self._table.edges(native))

    def products(self, key: ItemKey[T], /) -> tuple[EdgeProduct[T, W], ...]:
        """Return exact Cartesian edge products in arrival order."""
        native = self._lookup_key(key)

        if native is None:
            return ()

        return tuple(
            self._decode_product(product) for product in self._table.products(native)
        )

    def term(self, key: ItemKey[T], /) -> T:
        """Return an item's exact conclusion.

        Raises
        ------
        KeyError
            If the item is absent.
        """
        native = self._lookup_key(key)

        if native is None or not self._table.contains(native):
            msg = f"the chart holds no item {key!r}"
            raise KeyError(msg)

        return key.reading

    def derivations(self, key: ItemKey[T], /) -> Iterator[Derivation[T, W]]:
        """Enumerate an item's acyclic derivations, one tree at a time."""
        return self._trees(key, ())

    def sweep(
        self,
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
        program: _ChartProgram,
        resolver: object,
        *,
        scoring: object,
    ) -> None:
        """Fill this forest through the native drawing and storage kernel."""
        self._table.sweep(
            active_arities,
            unary_depth,
            threads,
            program,
            resolver,
            scoring=scoring,
        )

    def _trees(
        self, key: ItemKey[T], path: tuple[ItemKey[T], ...]
    ) -> Iterator[Derivation[T, W]]:
        """Yield derivations reached along one acyclic path."""
        if key in path:
            return

        deeper = (*path, key)

        for edge in self.edges(key):
            for below in self._premises(edge.children, 0, deeper):
                yield Derivation(key, edge, below)

    def _premises(
        self,
        children: tuple[ItemKey[T], ...],
        at: int,
        path: tuple[ItemKey[T], ...],
    ) -> Iterator[tuple[Derivation[T, W], ...]]:
        """Yield one subderivation per remaining premise, in premise order."""
        if at == len(children):
            yield ()
            return

        for tree in self._trees(children[at], path):
            for rest in self._premises(children, at + 1, path):
                yield (tree, *rest)

    def _lookup_key(self, key: ItemKey[T]) -> _NativeKey | None:
        """Return a key's dense representation when its values are registered."""
        category = self._categories.lookup(key.reading)
        tag = self._tags.lookup(key.nf_tag)

        if category is None or tag is None:
            return None

        return (key.span, (category, tag))

    def _decode_key(self, key: _NativeKey) -> ItemKey[T]:
        """Decode one dense forest key."""
        span, (category, tag) = key
        return ItemKey(span, self._categories.value(category), self._tags.value(tag))

    def _decode_edge(self, edge: _NativeEdge[W]) -> Edge[T, W]:
        """Decode one dense forest edge."""
        label, children, output, weight, split, order = edge
        return Edge(
            self._labels.value(label),
            tuple(self._decode_key(child) for child in children),
            self._categories.value(output),
            weight,
            split=split,
            order=order,
        )

    def _decode_product(self, product: _NativeProduct[W]) -> EdgeProduct[T, W]:
        """Decode one dense Cartesian edge product."""
        label, axes, output, weight, split, order = product
        return EdgeProduct(
            self._labels.value(label),
            tuple(tuple(self._decode_key(child) for child in axis) for axis in axes),
            self._categories.value(output),
            weight,
            split=split,
            order=order,
        )

    def _valid(self, span: Cell) -> bool:
        """Return whether a nonempty span belongs to this forest."""
        start, end = span
        return 0 <= start < end <= self._index.n

    @override
    def __reduce__(self) -> _ForestReduction[T, W]:
        """Return the public-data reconstruction call for this forest."""
        rows = tuple(
            (key, self.products(key))
            for span in self._index.spans_by_length()
            for key in self.items_at(span)
        )
        return (
            _restore_forest,
            (
                _ForestState(
                    self._index,
                    self._semiring,
                    self._normal_form,
                    rows,
                    self.truncated,
                    tuple(self._pruned.items()),
                    self._pruned_mass,
                ),
            ),
        )

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the input length and item count."""
        return f"PackedForest(n={self._index.n}, items={self.num_items})"
