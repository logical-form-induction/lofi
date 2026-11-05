"""Goal-relative inside-outside inference over packed edges."""

from math import exp
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Never, final, override

from lofi.deduce._values import Edge, EdgeProduct, ItemKey

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.deduce._ports import Chart, Goal, ProductChart, Semiring
    from lofi.deduce._scorers import SplitLogWeights
    from lofi.deduce._spans import Cell
    from lofi.terms import TargetSet

__all__ = [
    "SHARED_RULE_WEIGHTS",
    "IndependentRuleWeights",
    "Outside",
    "OutsideRuleSource",
    "SharedRuleWeights",
    "goal_partition",
    "outside_from_edges",
    "target_mass",
]

type _OutsideFields[T] = tuple[
    Chart[T, float],
    Semiring[float],
    float,
    tuple[tuple[ItemKey[T], float], ...],
    tuple[tuple[ItemKey[T], float], ...],
    tuple[tuple[ItemKey[T], tuple[EdgeProduct[T, float], ...]], ...],
]
"""Fields needed to reconstruct an outside result."""

type _OutsideReduction[T] = tuple[
    Callable[[_OutsideFields[T]], Outside[T]],
    tuple[_OutsideFields[T]],
]
"""Outside-result pickle reduction."""


@final
class SharedRuleWeights:
    """Outside rule weights read back from the inside pass."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the tag this source is selected by."""
        return "shared-with-inside"

    @property
    def chart_derived(self) -> bool:
        """Return whether the weights come from the chart the sweep reads."""
        return True

    def rule_weight[T](
        self,
        parent: ItemKey[T],
        alternative: Edge[T, float] | EdgeProduct[T, float],
        /,
    ) -> float:
        """Return the weight the inside pass recorded on one edge.

        Parameters
        ----------
        parent : ItemKey[T]
            The item the edge concludes.
        alternative : Edge[T, float] | EdgeProduct[T, float]
            The edge or exact edge product the sweep is descending through.
        """
        del parent

        return alternative.weight

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other value is the same source."""
        if type(other) is not SharedRuleWeights:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the hash the source shares with every other instance."""
        return hash(SharedRuleWeights)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the source."""
        return "SharedRuleWeights()"


@final
class IndependentRuleWeights:
    """Outside rule weights read from a split attention of their own.

    Parameters
    ----------
    attention : SplitLogWeights
        Normalized log weights over the split points of every span the sweep
        descends a binary edge at.
    """

    __slots__ = ("_attention",)

    def __init__(self, attention: SplitLogWeights) -> None:
        object.__setattr__(self, "_attention", attention)

    @property
    def attention(self) -> SplitLogWeights:
        """Return the split weights the sweep reads."""
        return self._attention

    @property
    def name(self) -> str:
        """Return the tag this source is selected by."""
        return "independent"

    @property
    def chart_derived(self) -> bool:
        """Return whether the weights come from the chart the sweep reads."""
        return False

    def rule_weight[T](
        self,
        parent: ItemKey[T],
        alternative: Edge[T, float] | EdgeProduct[T, float],
        /,
    ) -> float:
        """Return the declared weight of one edge.

        An axiom weighs what the chart recorded, there being no rule to weigh.

        Parameters
        ----------
        parent : ItemKey[T]
            The item the edge concludes.
        alternative : Edge[T, float] | EdgeProduct[T, float]
            The edge or exact edge product the sweep is descending through.

        Raises
        ------
        ValueError
            If the edge has premises and no weight is declared for its split.
        """
        if alternative.arity == 0:
            return alternative.weight

        split = alternative.split

        if split is None:
            msg = (
                f"edge {alternative.rule_name!r} at {parent.span!r} has no split "
                "point, "
                f"so no independent outside weight is declared for it"
            )
            raise ValueError(msg)

        try:
            weight = self._attention.weight(parent.span, split)

        except KeyError:
            msg = (
                f"no independent outside weight is declared for split {split} "
                f"of span {parent.span!r}"
            )
            raise ValueError(msg) from None

        return weight

    def __copy__(self) -> IndependentRuleWeights:
        """Reconstruct the source with its shared attention."""
        return IndependentRuleWeights(self._attention)

    def __deepcopy__(
        self, memo: dict[int, IndependentRuleWeights]
    ) -> IndependentRuleWeights:
        """Reconstruct the source with its shared attention."""
        del memo

        return IndependentRuleWeights(self._attention)

    @override
    def __reduce__(
        self,
    ) -> tuple[type[IndependentRuleWeights], tuple[SplitLogWeights]]:
        """Return the source reconstruction call."""
        return (IndependentRuleWeights, (self._attention,))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an independent rule source is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an independent rule source is immutable"

        raise AttributeError(msg)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two sources declare the same weights."""
        if type(other) is not IndependentRuleWeights:
            return NotImplemented

        return self._attention == other._attention

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared weights."""
        return hash((IndependentRuleWeights, self._attention))

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the declared span count."""
        return f"IndependentRuleWeights(spans={len(self._attention.spans)})"


type OutsideRuleSource = SharedRuleWeights | IndependentRuleWeights
"""Where an outside sweep reads the rule weight of an edge from."""

SHARED_RULE_WEIGHTS: Final[SharedRuleWeights] = SharedRuleWeights()
"""The source that reuses what the inside pass recorded."""


class _OutsideData[T](NamedTuple):
    """Live fields for an outside result."""

    chart: Chart[T, float]
    semiring: Semiring[float]
    partition: float
    inside: Mapping[ItemKey[T], float]
    outside: Mapping[ItemKey[T], float]
    admitted: Mapping[ItemKey[T], tuple[EdgeProduct[T, float], ...]]


@final
class Outside[T]:
    """An immutable goal-relative inside-outside result."""

    __slots__ = (
        "_admitted",
        "_chart",
        "_inside",
        "_outside",
        "_partition",
        "_semiring",
    )

    def __init__(self, data: _OutsideData[T], /) -> None:
        object.__setattr__(self, "_chart", data.chart)
        object.__setattr__(self, "_semiring", data.semiring)
        object.__setattr__(self, "_partition", data.partition)
        object.__setattr__(self, "_inside", MappingProxyType(dict(data.inside)))
        object.__setattr__(self, "_outside", MappingProxyType(dict(data.outside)))
        object.__setattr__(self, "_admitted", MappingProxyType(dict(data.admitted)))

    @property
    def partition(self) -> float:
        """Return the goal-relative log partition."""
        return self._partition

    def is_for(self, chart: Chart[T, float], /) -> bool:
        """Return whether this result was computed from `chart`."""
        return chart is self._chart

    def inside(self, key: ItemKey[T], /) -> float:
        """Return an item's goal-relative inside weight."""
        return self._inside.get(key, self._semiring.zero)

    def outside(self, key: ItemKey[T], /) -> float:
        """Return an item's goal-relative outside weight."""
        return self._outside.get(key, self._semiring.zero)

    def constituent_marginal(self, key: ItemKey[T], /) -> float:
        """Return an item's posterior constituent probability."""
        mass = self._semiring.times(self.inside(key), self.outside(key))

        return self._normalize(mass)

    def rule_marginal(self, parent: ItemKey[T], edge: Edge[T, float], /) -> float:
        """Return an edge's posterior rule probability."""
        if not any(
            product.contains(edge) for product in self._admitted.get(parent, ())
        ):
            return 0.0

        mass = self._semiring.times(self.outside(parent), edge.weight)

        for child in edge.children:
            mass = self._semiring.times(mass, self.inside(child))

        return self._normalize(mass)

    def rule_marginals(
        self,
        parent: ItemKey[T],
        /,
    ) -> tuple[tuple[Edge[T, float], float], ...]:
        """Return admitted edges for `parent` and their posterior probabilities."""
        return tuple(
            (edge, self.rule_marginal(parent, edge))
            for product in self._admitted.get(parent, ())
            for edge in product.edges()
        )

    def intersection(self, span: Cell, /) -> tuple[ItemKey[T], ...]:
        """Return inside items with live outside weight in chart order."""
        return tuple(
            key
            for key in self._chart.items_at(span)
            if not self._semiring.is_zero(self.outside(key))
        )

    def _normalize(self, mass: float) -> float:
        """Normalize one log mass by the partition."""
        if self._semiring.is_zero(mass) or self._semiring.is_zero(self._partition):
            return 0.0

        return exp(mass - self._partition)

    def _fields(self) -> _OutsideFields[T]:
        """Return fields in reconstruction order."""
        return (
            self._chart,
            self._semiring,
            self._partition,
            tuple(self._inside.items()),
            tuple(self._outside.items()),
            tuple(self._admitted.items()),
        )

    def __copy__(self) -> Outside[T]:
        """Reconstruct the result with shared dependencies."""
        return _restore_outside(self._fields())

    def __deepcopy__(self, memo: dict[int, Outside[T]]) -> Outside[T]:
        """Reconstruct the result with shared dependencies."""
        del memo

        return _restore_outside(self._fields())

    @override
    def __reduce__(self) -> _OutsideReduction[T]:
        """Return the outside-result reconstruction call."""
        return (_restore_outside, (self._fields(),))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "an outside result is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "an outside result is immutable"

        raise AttributeError(msg)

    @override
    def __repr__(self) -> str:
        """Return a rendering with the partition and item count."""
        return f"Outside(partition={self._partition!r}, items={len(self._inside)})"


def _restore_outside[T](fields: _OutsideFields[T]) -> Outside[T]:
    """Reconstruct an outside result from frozen fields."""
    chart, semiring, partition, inside, outside, admitted = fields

    return Outside(
        _OutsideData(
            chart,
            semiring,
            partition,
            dict(inside),
            dict(outside),
            dict(admitted),
        )
    )


def target_mass[T](
    chart: Chart[T, float],
    outside: Outside[T],
    targets: TargetSet[T],
    span: Cell,
    /,
) -> float:
    """Return the posterior mass of matching items at a span.

    Parameters
    ----------
    chart : Chart[T, float]
        Chart whose items are scored.
    outside : Outside[T]
        Outside result computed from ``chart``.
    targets : TargetSet[T]
        Categories and matching used to select items.
    span : Cell
        Half-open span whose items are scored.

    Raises
    ------
    ValueError
        If ``outside`` was computed from another chart.
    """
    if not outside.is_for(chart):
        msg = "the outside result was computed from another chart"
        raise ValueError(msg)

    return sum(
        (
            outside.constituent_marginal(key)
            for key in chart.items_at(span)
            if any(
                targets.matching.equivalent(key.reading, category)
                for category in targets.categories
            )
        ),
        0.0,
    )


def _items[T](chart: Chart[T, float]) -> tuple[ItemKey[T], ...]:
    """Return every forest item from narrowest span to widest."""
    return tuple(
        key for span in chart.index.spans_by_length() for key in chart.items_at(span)
    )


def _require_log_inside[T](chart: Chart[T, float]) -> Semiring[float]:
    """Return the chart's semiring once it carries log-probability weights."""
    semiring = chart.semiring

    if not semiring.log_probability:
        msg = (
            "outside marginals require log-probability weights, which "
            f"{semiring.name!r} does not carry"
        )
        raise ValueError(msg)

    return semiring


def _inside[T](
    key: ItemKey[T],
    admitted: Mapping[ItemKey[T], tuple[EdgeProduct[T, float], ...]],
    semiring: Semiring[float],
    memo: dict[ItemKey[T], float],
    active: set[ItemKey[T]],
) -> float:
    """Compute one admitted inside weight."""
    known = memo.get(key)

    if known is not None:
        return known

    if key in active:
        msg = f"the admitted forest contains a cycle at {key!r}"
        raise ValueError(msg)

    active.add(key)
    total = semiring.zero

    for product in admitted.get(key, ()):
        contribution = product.weight

        for axis in product.axes:
            axis_total = semiring.zero

            for child in axis:
                axis_total = semiring.plus(
                    axis_total,
                    _inside(child, admitted, semiring, memo, active),
                )

            contribution = semiring.times(contribution, axis_total)

        total = semiring.plus(total, contribution)

    active.remove(key)
    memo[key] = total

    return total


def _parents_first[T](
    roots: tuple[ItemKey[T], ...],
    admitted: Mapping[ItemKey[T], tuple[EdgeProduct[T, float], ...]],
) -> tuple[ItemKey[T], ...]:
    """Order reachable items from roots toward leaves."""
    seen: set[ItemKey[T]] = set()
    active: set[ItemKey[T]] = set()
    finished: list[ItemKey[T]] = []

    def visit(key: ItemKey[T]) -> None:
        """Append one reachable subtree in postorder."""
        if key in seen:
            return

        if key in active:
            msg = f"the admitted forest contains a cycle at {key!r}"
            raise ValueError(msg)

        active.add(key)

        for product in admitted.get(key, ()):
            for axis in product.axes:
                for child in axis:
                    visit(child)

        active.remove(key)
        seen.add(key)
        finished.append(key)

    for root in roots:
        visit(root)

    return tuple(reversed(finished))


def _root_state[T](
    chart: Chart[T, float],
    goal: Goal[T],
    inside: Mapping[ItemKey[T], float],
    semiring: Semiring[float],
) -> tuple[tuple[ItemKey[T], ...], float, dict[ItemKey[T], float]]:
    """Compute the partition and outside seeds for root groups."""
    groups = goal.root_groups(chart)
    roots = tuple(root for group in groups for root in group)
    totals: list[float] = []

    for group in groups:
        total = semiring.zero

        for root in group:
            contribution = semiring.times(
                goal.root_weight(root),
                inside.get(root, semiring.zero),
            )
            total = semiring.plus(total, contribution)

        totals.append(total)

    if not groups:
        return roots, semiring.zero, {}

    weight = goal.combination_weight(len(groups))
    partition = weight

    for total in totals:
        partition = semiring.times(partition, total)

    seeds: dict[ItemKey[T], float] = {}

    for position, group in enumerate(groups):
        seed = weight

        for other_position, total in enumerate(totals):
            if other_position != position:
                seed = semiring.times(seed, total)

        for root in group:
            root_seed = semiring.times(seed, goal.root_weight(root))
            seeds[root] = semiring.plus(
                seeds.get(root, semiring.zero),
                root_seed,
            )

    return roots, partition, seeds


def _goal_relative_inside[T](
    chart: ProductChart[T, float],
    goal: Goal[T],
    semiring: Semiring[float],
) -> tuple[
    tuple[ItemKey[T], ...],
    dict[ItemKey[T], tuple[EdgeProduct[T, float], ...]],
    dict[ItemKey[T], float],
]:
    """Build admitted edge products and their separate inside map."""
    items = _items(chart)
    canonical = {key: key for key in items}
    admitted: dict[ItemKey[T], tuple[EdgeProduct[T, float], ...]] = {}

    for key in items:
        kept: list[EdgeProduct[T, float]] = []

        for held in chart.products(key):
            product = EdgeProduct(
                held.rule_name,
                tuple(
                    tuple(canonical.get(child, child) for child in axis)
                    for axis in held.axes
                ),
                held.output,
                held.weight,
                held.split,
                held.order,
            )
            accepted: list[Edge[T, float]] = []
            complete = True

            for edge in product.edges():
                if goal.admits_edge(chart, key, edge):
                    accepted.append(edge)
                else:
                    complete = False

            if complete:
                kept.append(product)
            else:
                kept.extend(
                    EdgeProduct(
                        edge.rule_name,
                        tuple((child,) for child in edge.children),
                        edge.output,
                        edge.weight,
                        edge.split,
                        edge.order,
                    )
                    for edge in accepted
                )

        admitted[key] = tuple(kept)
    inside: dict[ItemKey[T], float] = {}
    active: set[ItemKey[T]] = set()

    for key in items:
        _inside(key, admitted, semiring, inside, active)

    return items, admitted, inside


def goal_partition[T](chart: ProductChart[T, float], goal: Goal[T], /) -> float:
    """Return a forest's goal-relative log partition.

    This is the partition an outside pass over the same forest and goal
    reports, computed without the outside sweep.

    Parameters
    ----------
    chart : ProductChart[T, float]
        A finished forest under log-probability weights.
    goal : Goal[T]
        The goal the partition is relative to.

    Raises
    ------
    ValueError
        If the forest does not carry log-probability weights, or contains a cycle.
    """
    semiring = _require_log_inside(chart)
    _, _, inside = _goal_relative_inside(chart, goal, semiring)

    return _root_state(chart, goal, inside, semiring)[1]


def outside_from_edges[T](
    chart: ProductChart[T, float],
    goal: Goal[T],
    /,
    *,
    rule_source: OutsideRuleSource = SHARED_RULE_WEIGHTS,
) -> Outside[T]:
    """Compute goal-relative inside weights, outside weights, and marginals.

    The contribution one item makes to a premise of one of its edges is that
    item's outside weight, times the edge's rule weight, times the inside
    weight of every other premise. The premise's own inside weight never
    enters, whichever source the rule weight comes from.

    Parameters
    ----------
    chart : ProductChart[T, float]
        The finished chart to sweep.
    goal : Goal[T]
        The goal the sweep is relative to.
    rule_source : OutsideRuleSource
        Where the sweep reads each edge's rule weight, by default the weight
        the inside pass recorded on the edge.
    """
    semiring = _require_log_inside(chart)
    items, admitted, inside = _goal_relative_inside(chart, goal, semiring)

    roots, partition, seeds = _root_state(chart, goal, inside, semiring)
    outside: dict[ItemKey[T], float] = dict.fromkeys(items, semiring.zero)

    outside.update(seeds)

    for parent in _parents_first(roots, admitted):
        parent_outside = outside.get(parent, semiring.zero)

        for product in admitted.get(parent, ()):
            rule_weight = rule_source.rule_weight(parent, product)
            totals = []

            for axis in product.axes:
                total = semiring.zero

                for child in axis:
                    total = semiring.plus(total, inside[child])

                totals.append(total)

            for position, axis in enumerate(product.axes):
                contribution = semiring.times(parent_outside, rule_weight)

                for sibling_position, total in enumerate(totals):
                    if sibling_position != position:
                        contribution = semiring.times(contribution, total)

                for child in axis:
                    outside[child] = semiring.plus(
                        outside.get(child, semiring.zero),
                        contribution,
                    )

    return Outside(_OutsideData(chart, semiring, partition, inside, outside, admitted))
