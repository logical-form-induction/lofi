"""Constrained high-variance acquisition for judgment collection."""

from __future__ import annotations

from collections import Counter
from functools import partial
from math import ceil, floor, inf
from statistics import fmean
from types import MappingProxyType
from typing import TYPE_CHECKING, Protocol, final, override

from bead.lists.constraints import (
    CategoricalBinning,
    EqualWidthBinning,
    QuantileBinning,
    StdDevBinning,
    ThresholdBinning,
)
from bead.lists.stratification import assign_grid_cells

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.judgment._attributes import AttributeExpression, item_key
from lofi.registry import Registry, RegistryError
from lofi.spec.acquisition import (
    BalanceSettings,
    BatchBalanceSettings,
    BatchCoverageSettings,
    BatchDiversitySettings,
    BatchMinOccurrenceSettings,
    CategoricalBinningSettings,
    ConditionalUniquenessSettings,
    DiversitySettings,
    EqualWidthBinningSettings,
    GridStratificationSettings,
    GroupedQuantileSettings,
    OrderingSettings,
    QuantileBinningSettings,
    QuantileSettings,
    StdDevBinningSettings,
    ThresholdBinningSettings,
    UniquenessSettings,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping, Sequence

    from bead.items.item import Item
    from bead.lists.constraints import BinningSpec

    from lofi.spec.acquisition import (
        BatchConstraintSettings,
        BinningSettings,
        ListConstraintSettings,
        RC21AcquisitionSettings,
    )

__all__ = [
    "RC21_ACQUISITION_PLUGIN",
    "AcquisitionBatch",
    "AcquisitionError",
    "AcquisitionRule",
    "AcquisitionRulePlugin",
    "AcquisitionScore",
    "PoolExhaustedError",
    "RC21Acquisition",
    "acquisition_rule_for",
]


class AcquisitionError(ValueError):
    """A candidate pool cannot satisfy the declared batch constraints."""


class PoolExhaustedError(AcquisitionError):
    """The remaining candidates cannot fill one more batch under the rule."""


class AcquisitionRule(Protocol):
    """Select one batch under a declared acquisition policy."""

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return every attribute expression the rule reads."""
        ...

    def stratum(self, item: Item, /) -> str:
        """Return the stratum an item is allocated under."""
        ...

    def select(
        self,
        candidates: Sequence[Item],
        scores: Iterable[AcquisitionScore],
        performance: Mapping[str, float],
        /,
        *,
        previous: Sequence[AcquisitionBatch],
    ) -> AcquisitionBatch:
        """Return one identity-preserving constrained batch.

        ``previous`` holds the batches the run has already acquired, in order,
        which the declared batch constraints count.
        """
        ...


@final
class AcquisitionScore(FrozenValue):
    """One candidate's expected response."""

    _noun = "acquisition score"
    __slots__ = ("_expected", "_item_id")

    def __init__(self, item_id: str, expected: float) -> None:
        object.__setattr__(
            self, "_item_id", ensure_name("an acquisition item", item_id)
        )
        ensure_finite("an expected response", expected)
        object.__setattr__(self, "_expected", float(expected))

    @property
    def item_id(self) -> str:
        """Return the scored item's identity."""
        return self._item_id

    @property
    def expected(self) -> float:
        """Return its expected response on the declared response scale."""
        return self._expected


@final
class AcquisitionBatch(FrozenValue):
    """One selected batch with its exact bead collection payload.

    Parameters
    ----------
    items : Sequence[Item]
        The selected bead items in batch order.
    shares : Mapping[str, float]
        Each stratum's realized share of the batch.
    variance : float | None
        Variance of the selected candidates' expected responses, or None for a
        batch drawn before any prediction exists.
    performance : Mapping[str, float]
        Each stratum's performance the batch was allocated by, empty for a
        batch drawn before any fit.
    floored : tuple[str, ...]
        The strata whose negative performance was floored at zero when the
        batch was allocated, in sorted order.
    """

    _noun = "acquisition batch"
    __slots__ = ("_floored", "_items", "_performance", "_shares", "_variance")

    def __init__(
        self,
        items: Sequence[Item],
        shares: Mapping[str, float],
        variance: float | None,
        *,
        performance: Mapping[str, float],
        floored: tuple[str, ...],
    ) -> None:
        object.__setattr__(self, "_items", tuple(items))
        object.__setattr__(self, "_shares", MappingProxyType(dict(shares)))
        object.__setattr__(self, "_variance", variance)
        object.__setattr__(self, "_performance", MappingProxyType(dict(performance)))
        object.__setattr__(self, "_floored", floored)

    @property
    def items(self) -> tuple[Item, ...]:
        """Return the selected bead items under their own identities."""
        return self._items

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return selected identities in batch order."""
        return tuple(item_key(item) for item in self._items)

    @property
    def shares(self) -> Mapping[str, float]:
        """Return each stratum's realized share of the batch."""
        return self._shares

    @property
    def variance(self) -> float | None:
        """Return the variance of the selected expected responses, if predicted."""
        return self._variance

    @property
    def performance(self) -> Mapping[str, float]:
        """Return each stratum's performance the batch was allocated by."""
        return self._performance

    @property
    def floored(self) -> tuple[str, ...]:
        """Return the strata whose negative performance was floored at zero."""
        return self._floored


def _quotas(
    performance: Mapping[str, float], batch_size: int, /
) -> tuple[dict[str, int], dict[str, float], tuple[str, ...]]:
    """Return each stratum's batch quota, its target share and the floored strata.

    A stratum whose performance is negative is floored at zero, so it receives
    no proportional share; when no stratum is positive the batch is split
    evenly.
    """
    if not performance:
        msg = "RC2.1 acquisition needs the performance of each stratum"
        raise AcquisitionError(msg)
    checked: dict[str, float] = {}
    floored: list[str] = []
    for stratum, value in performance.items():
        name = ensure_name("an acquisition stratum", stratum)
        ensure_finite(f"stratum {name!r}'s performance", value)
        if value < 0.0:
            floored.append(name)
        checked[name] = max(float(value), 0.0)
    total = sum(checked.values())
    if total > 0.0:
        shares = {stratum: value / total for stratum, value in checked.items()}
    else:
        shares = {stratum: 1.0 / len(checked) for stratum in checked}
    exact = {stratum: share * batch_size for stratum, share in shares.items()}
    quotas = {stratum: int(value) for stratum, value in exact.items()}
    remainder = batch_size - sum(quotas.values())
    order = sorted(
        exact, key=lambda stratum: (-(exact[stratum] - quotas[stratum]), stratum)
    )
    for stratum in order[:remainder]:
        quotas[stratum] += 1
    return quotas, shares, tuple(sorted(floored))


class _Selection(Protocol):
    """One constraint's state while a single batch is selected."""

    def admits(self, item: Item, /) -> bool:
        """Return whether the batch may take the item."""
        ...

    def demands(self, item: Item, /) -> bool:
        """Return whether the item fills a need the batch must meet first."""
        ...

    def take(self, item: Item, /) -> None:
        """Record that the batch took the item."""
        ...

    def violation(self) -> str | None:
        """Return why the completed batch breaks the constraint, if it does."""
        ...


class _Constraint(Protocol):
    """One declared bead constraint over the batches a rule selects."""

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return every attribute expression the constraint reads."""
        ...

    def begin(
        self,
        candidates: Sequence[Item],
        previous: Sequence[AcquisitionBatch],
        batch_size: int,
        /,
    ) -> _Selection:
        """Return the constraint's state for one batch over the candidates."""
        ...


@final
class _UniqueSelection:
    """The values a batch's items already hold under one uniqueness constraint."""

    __slots__ = ("_constraint", "_seen")

    def __init__(self, constraint: _Unique, /) -> None:
        self._constraint = constraint
        self._seen: set[tuple[str, ...]] = set()

    def admits(self, item: Item, /) -> bool:
        """Admit an item whose value no taken item holds, where it applies."""
        return (
            not self._constraint.applies(item)
            or self._constraint.value(item) not in self._seen
        )

    def demands(self, item: Item, /) -> bool:
        """Demand no item."""
        del item
        return False

    def take(self, item: Item, /) -> None:
        """Record the taken item's value, where the constraint applies."""
        if self._constraint.applies(item):
            self._seen.add(self._constraint.value(item))

    def violation(self) -> str | None:
        """Report none, since a repeated value is never taken."""
        return None


@final
class _Unique(FrozenValue):
    """No two selected items share a value, where an optional condition holds."""

    _noun = "uniqueness constraint"
    __slots__ = ("_condition", "_expression")

    def __init__(
        self, expression: AttributeExpression, condition: AttributeExpression | None
    ) -> None:
        object.__setattr__(self, "_expression", expression)
        object.__setattr__(self, "_condition", condition)

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return the value expression, then the condition if one is declared."""
        if self._condition is None:
            return (self._expression,)
        return (self._expression, self._condition)

    def applies(self, item: Item, /) -> bool:
        """Return whether the constraint applies to one item."""
        return self._condition is None or self._condition.holds(item)

    def value(self, item: Item, /) -> tuple[str, ...]:
        """Return the value that is unique across the batch."""
        return self._expression.levels(item)

    def begin(
        self,
        candidates: Sequence[Item],
        previous: Sequence[AcquisitionBatch],
        batch_size: int,
        /,
    ) -> _Selection:
        """Return an empty record of taken values."""
        del candidates, previous, batch_size
        return _UniqueSelection(self)


@final
class _DiversitySelection:
    """The distinct values a batch's items hold under one diversity constraint."""

    __slots__ = ("_constraint", "_seen")

    def __init__(self, constraint: _Diversity, /) -> None:
        self._constraint = constraint
        self._seen: set[tuple[str, ...]] = set()

    def admits(self, item: Item, /) -> bool:
        """Admit every item."""
        del item
        return True

    def demands(self, item: Item, /) -> bool:
        """Demand no item."""
        del item
        return False

    def take(self, item: Item, /) -> None:
        """Record the taken item's value."""
        self._seen.add(self._constraint.expression.levels(item))

    def violation(self) -> str | None:
        """Report a batch holding fewer distinct values than declared."""
        minimum = self._constraint.minimum
        if len(self._seen) >= minimum:
            return None
        return (
            f"the batch has {len(self._seen)} distinct values of "
            f"{self._constraint.expression.text!r}, fewer than the declared {minimum}"
        )


@final
class _Diversity(FrozenValue):
    """A selected batch holds a least number of an expression's values."""

    _noun = "diversity constraint"
    __slots__ = ("_expression", "_minimum")

    def __init__(self, settings: DiversitySettings, /) -> None:
        object.__setattr__(
            self,
            "_expression",
            AttributeExpression(
                settings.property_expression, role="a diversity constraint", many=True
            ),
        )
        object.__setattr__(self, "_minimum", settings.min_unique_values)

    @property
    def expression(self) -> AttributeExpression:
        """Return the expression whose values are counted."""
        return self._expression

    @property
    def minimum(self) -> int:
        """Return the least number of distinct values."""
        return self._minimum

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return the counted expression."""
        return (self._expression,)

    def begin(
        self,
        candidates: Sequence[Item],
        previous: Sequence[AcquisitionBatch],
        batch_size: int,
        /,
    ) -> _Selection:
        """Return an empty record of distinct values."""
        del candidates, previous, batch_size
        return _DiversitySelection(self)


@final
class _CappedSelection:
    """Counts of a batch's items per key, each key held to a cap."""

    __slots__ = ("_cap", "_caps", "_check", "_counts", "_key")

    def __init__(
        self,
        key: Callable[[Item], Hashable],
        caps: Mapping[Hashable, float],
        cap: float,
        check: Callable[[Mapping[Hashable, int]], str | None] | None,
        /,
    ) -> None:
        self._key = key
        self._caps = caps
        self._cap = cap
        self._check = check
        self._counts: dict[Hashable, int] = {}

    def admits(self, item: Item, /) -> bool:
        """Admit an item whose key the batch holds fewer times than its cap."""
        key = self._key(item)
        return self._counts.get(key, 0) < self._caps.get(key, self._cap)

    def demands(self, item: Item, /) -> bool:
        """Demand no item."""
        del item
        return False

    def take(self, item: Item, /) -> None:
        """Count the taken item under its key."""
        key = self._key(item)
        self._counts[key] = self._counts.get(key, 0) + 1

    def violation(self) -> str | None:
        """Report the batch's counts breaking the constraint, if they do."""
        if self._check is None:
            return None
        return self._check(MappingProxyType(self._counts))


@final
class _Balance(FrozenValue):
    """A selected batch holds an expression's levels in balanced counts.

    A level's count deviates from its target by at most the tolerance, as a
    proportion of the target (of one, for a target of zero). Without declared
    targets every level the candidates hold has the batch's mean count as its
    target while the batch is selected, and the realized counts are compared
    with their own mean, as bead compares them.
    """

    _noun = "balance constraint"
    __slots__ = ("_expression", "_targets", "_tolerance")

    def __init__(self, settings: BalanceSettings, /) -> None:
        object.__setattr__(
            self,
            "_expression",
            AttributeExpression(
                settings.property_expression, role="a balance constraint"
            ),
        )
        object.__setattr__(self, "_targets", settings.target_counts)
        object.__setattr__(self, "_tolerance", settings.tolerance)

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return the balanced expression."""
        return (self._expression,)

    def begin(
        self,
        candidates: Sequence[Item],
        previous: Sequence[AcquisitionBatch],
        batch_size: int,
        /,
    ) -> _Selection:
        """Return counts held to each level's target plus its tolerance."""
        del previous
        targets = self._targets
        if targets is None:
            levels = {self._expression.level(item) for item in candidates}
            mean = batch_size / len(levels) if levels else 0.0
            limits: dict[Hashable, float] = {
                level: floor(mean + self._tolerance * max(mean, 1.0))
                for level in levels
            }
        else:
            limits = {
                level: floor(count + self._tolerance * max(count, 1))
                for level, count in targets.items()
            }
        return _CappedSelection(self._expression.level, limits, inf, self._violation)

    def _violation(self, counts: Mapping[Hashable, int], /) -> str | None:
        """Report a count outside its target's tolerance."""
        targets = self._targets
        text = self._expression.text
        if targets is None:
            if not counts:
                return None
            mean = fmean(counts.values())
            deviation = max(
                abs(count - mean) / max(mean, 1.0) for count in counts.values()
            )
            if deviation <= self._tolerance:
                return None
            return (
                f"the batch's counts of {text!r} deviate by {deviation:.6g} from "
                f"their mean, beyond the declared tolerance {self._tolerance:.6g}"
            )
        for level, target in targets.items():
            count = counts.get(level, 0)
            if abs(count - target) / max(target, 1) > self._tolerance:
                return (
                    f"the batch holds {count} items of {text!r} level {level!r}, "
                    f"outside {self._tolerance:.6g} of the declared {target}"
                )
        return None


@final
class _Cells(FrozenValue):
    """A selected batch takes at most a number of items from each grid cell.

    Each candidate is assigned a cell by bead's grid stratification over the
    candidate pool: one bin per dimension, numeric bins fitted within each
    group where a group expression is declared. A quantile constraint is the
    one-dimensional quantile grid and a grouped quantile constraint the same
    grid within groups.
    """

    _noun = "grid stratification constraint"
    __slots__ = ("_binnings", "_categorical", "_dimensions", "_group", "_per_cell")

    def __init__(
        self,
        dimensions: Sequence[tuple[AttributeExpression, BinningSettings]],
        group: AttributeExpression | None,
        per_cell: int,
        /,
    ) -> None:
        object.__setattr__(
            self, "_dimensions", tuple(expression for expression, _ in dimensions)
        )
        object.__setattr__(
            self,
            "_binnings",
            tuple(_bead_binning(binning) for _, binning in dimensions),
        )
        object.__setattr__(
            self,
            "_categorical",
            tuple(binning.binning == "categorical" for _, binning in dimensions),
        )
        object.__setattr__(self, "_group", group)
        object.__setattr__(self, "_per_cell", per_cell)

    @classmethod
    def of(cls, settings: ListConstraintSettings, /) -> _Cells | None:
        """Return a quantile or grid constraint's interior form, or None."""
        match settings:
            case QuantileSettings(
                property_expression=expression,
                n_quantiles=bins,
                items_per_quantile=per_cell,
            ):
                return cls(
                    (
                        (
                            _numeric(expression),
                            QuantileBinningSettings("quantile", bins),
                        ),
                    ),
                    None,
                    per_cell,
                )
            case GroupedQuantileSettings(
                property_expression=expression,
                group_by_expression=group,
                n_quantiles=bins,
                items_per_quantile=per_cell,
            ):
                return cls(
                    (
                        (
                            _numeric(expression),
                            QuantileBinningSettings("quantile", bins),
                        ),
                    ),
                    AttributeExpression(group, role="a quantile group"),
                    per_cell,
                )
            case GridStratificationSettings(
                dimensions=dimensions,
                group_by_expression=group,
                items_per_cell=per_cell,
            ):
                return cls(
                    tuple(
                        (
                            AttributeExpression(
                                dimension.property_expression,
                                role="a grid dimension",
                            )
                            if dimension.binning.binning == "categorical"
                            else _numeric(dimension.property_expression),
                            dimension.binning,
                        )
                        for dimension in dimensions
                    ),
                    None
                    if group is None
                    else AttributeExpression(group, role="a grid group"),
                    per_cell,
                )
        return None

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return every dimension's expression, then the group's if declared."""
        if self._group is None:
            return self._dimensions
        return (*self._dimensions, self._group)

    def begin(
        self,
        candidates: Sequence[Item],
        previous: Sequence[AcquisitionBatch],
        batch_size: int,
        /,
    ) -> _Selection:
        """Return per-cell counts over the candidates' fitted cells."""
        del previous, batch_size
        keys = [item_key(item) for item in candidates]
        by_key = {item_key(item): item for item in candidates}
        group = self._group
        cells = assign_grid_cells(
            keys,
            [
                partial(_level_of if categorical else _number_of, expression, by_key)
                for expression, categorical in zip(
                    self._dimensions, self._categorical, strict=True
                )
            ],
            list(self._binnings),
            None if group is None else partial(_level_of, group, by_key),
        )
        return _CappedSelection(
            lambda item: (
                None if group is None else group.level(item),
                cells[item_key(item)],
            ),
            {},
            self._per_cell,
            None,
        )


def _level_of(
    expression: AttributeExpression, items: Mapping[str, Item], key: str, /
) -> str:
    """Return the level of the item under a key."""
    return expression.level(items[key])


def _number_of(
    expression: AttributeExpression, items: Mapping[str, Item], key: str, /
) -> float:
    """Return the number the item under a key yields."""
    return expression.number(items[key])


def _numeric(expression: str, /) -> AttributeExpression:
    """Return a numeric dimension's expression."""
    return AttributeExpression(expression, role="a numeric dimension", numeric=True)


def _bead_binning(settings: BinningSettings, /) -> BinningSpec:
    """Return bead's binning strategy for resolved settings."""
    match settings:
        case QuantileBinningSettings(n_quantiles=bins):
            return QuantileBinning(binning="quantile", n_quantiles=bins)
        case EqualWidthBinningSettings(n_bins=bins, range_min=low, range_max=high):
            return EqualWidthBinning(
                binning="equal_width", n_bins=bins, range_min=low, range_max=high
            )
        case ThresholdBinningSettings(edges=edges):
            return ThresholdBinning(binning="threshold", edges=edges)
        case StdDevBinningSettings(k_values=offsets):
            return StdDevBinning(binning="stddev", k_values=offsets)
        case CategoricalBinningSettings(categories=levels, include_other=other):
            return CategoricalBinning(
                binning="categorical", categories=levels, include_other=other
            )


@final
class _BatchLevels(FrozenValue):
    """One batch constraint's expression over the levels batches hold."""

    _noun = "batch constraint"
    __slots__ = ("_expression", "_settings")

    def __init__(self, settings: BatchConstraintSettings, /) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(
            self,
            "_expression",
            AttributeExpression(
                settings.property_expression,
                role=f"a batch {settings.constraint_type} constraint",
            ),
        )

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return the expression the constraint counts."""
        return (self._expression,)

    def begin(
        self,
        candidates: Sequence[Item],
        previous: Sequence[AcquisitionBatch],
        batch_size: int,
        /,
    ) -> _Selection:
        """Return the constraint's state given the batches already acquired."""
        level = self._expression.level
        history = Counter(level(item) for batch in previous for item in batch.items)
        lists = Counter(
            value
            for batch in previous
            for value in {level(item) for item in batch.items}
        )
        pooled = {level(item) for item in candidates} | set(history)
        return _BatchSelection(
            self._settings, level, history, lists, pooled, batch_size
        )


@final
class _BatchSelection:
    """A batch constraint's counts over the acquired batches and this one."""

    __slots__ = (
        "_batch_size",
        "_history",
        "_level",
        "_lists",
        "_pooled",
        "_settings",
        "_taken",
    )

    def __init__(
        self,
        settings: BatchConstraintSettings,
        level: Callable[[Item], str],
        history: Counter[str],
        lists: Counter[str],
        pooled: set[str],
        batch_size: int,
        /,
    ) -> None:
        self._settings = settings
        self._level = level
        self._history = history
        self._lists = lists
        self._pooled = pooled
        self._batch_size = batch_size
        self._taken: Counter[str] = Counter()

    def _held(self, value: str, /) -> int:
        """Return how many acquired items hold the level, this batch's included."""
        return self._history[value] + self._taken[value]

    def admits(self, item: Item, /) -> bool:
        """Admit an item the acquired batches leave room for."""
        value = self._level(item)
        match self._settings:
            case BatchDiversitySettings(max_lists_per_value=most):
                return self._taken[value] > 0 or self._lists[value] < most
            case BatchBalanceSettings(target_distribution=shares, tolerance=gap):
                if value not in shares:
                    return True
                total = sum(self._history.values()) + self._batch_size
                return self._held(value) < floor((shares[value] + gap) * total)
        return True

    def demands(self, item: Item, /) -> bool:
        """Demand an item holding a level the acquired batches still need."""
        value = self._level(item)
        match self._settings:
            case BatchCoverageSettings(target_values=targets, min_coverage=share):
                wanted = self._pooled if targets is None else set(targets)
                if not wanted or value not in wanted or self._held(value) > 0:
                    return False
                covered = sum(1 for level in wanted if self._held(level) > 0)
                return covered / len(wanted) < share
            case BatchMinOccurrenceSettings(min_occurrences=least):
                return self._held(value) < least
            case BatchBalanceSettings(target_distribution=shares, tolerance=gap):
                if value not in shares:
                    return False
                total = sum(self._history.values()) + self._batch_size
                return self._held(value) < ceil((shares[value] - gap) * total)
        return False

    def take(self, item: Item, /) -> None:
        """Count the taken item's level."""
        self._taken[self._level(item)] += 1

    def violation(self) -> str | None:
        """Report acquired shares breaking a batch balance constraint."""
        match self._settings:
            case BatchBalanceSettings(
                property_expression=text, target_distribution=shares, tolerance=gap
            ):
                total = sum(self._history.values()) + sum(self._taken.values())
                for value, share in shares.items():
                    realized = self._held(value) / total if total else 0.0
                    if abs(realized - share) > gap:
                        return (
                            f"the acquired batches hold {realized:.6g} of {text!r} "
                            f"level {value!r}, outside {gap:.6g} of {share:.6g}"
                        )
        return None


@final
class _Ordering(FrozenValue):
    """Arrange a selected batch in the order a bead ordering constraint fixes.

    Practice items come first, then the main items, each in blocks of one
    level where blocks are declared, in order of each block's best-ranked
    item. Each position takes the best-ranked remaining item of the current
    block that every precedence pair lets follow the items already placed and
    that lies at least the declared distance after the last item sharing its
    no-adjacent level.
    """

    _noun = "ordering constraint"
    __slots__ = (
        "_adjacent",
        "_block",
        "_gap",
        "_longest",
        "_practice",
        "_precedence",
    )

    def __init__(self, settings: OrderingSettings, /) -> None:
        object.__setattr__(self, "_precedence", settings.precedence_pairs)
        object.__setattr__(
            self,
            "_adjacent",
            None
            if settings.no_adjacent_property is None
            else AttributeExpression(
                settings.no_adjacent_property, role="a no-adjacent ordering"
            ),
        )
        object.__setattr__(
            self,
            "_block",
            None
            if settings.block_by_property is None
            else AttributeExpression(
                settings.block_by_property, role="an ordering block"
            ),
        )
        object.__setattr__(
            self, "_gap", 1 if settings.min_distance is None else settings.min_distance
        )
        object.__setattr__(self, "_longest", settings.max_distance)
        object.__setattr__(
            self,
            "_practice",
            None
            if settings.practice_item_property is None
            else AttributeExpression(
                settings.practice_item_property, role="a practice item", truth=True
            ),
        )

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return every declared ordering expression."""
        return tuple(
            expression
            for expression in (self._adjacent, self._block, self._practice)
            if expression is not None
        )

    def _blocks(self, items: Sequence[Item], /) -> list[list[Item]]:
        """Return the items in blocks: practice items first, then the main ones."""
        practice = self._practice
        segments = (
            [list(items)]
            if practice is None
            else [
                [item for item in items if practice.holds(item)],
                [item for item in items if not practice.holds(item)],
            ]
        )
        block = self._block
        blocks: list[list[Item]] = []
        for segment in segments:
            if block is None:
                blocks.append(segment)
                continue
            grouped: dict[str, list[Item]] = {}
            for item in segment:
                grouped.setdefault(block.level(item), []).append(item)
            blocks.extend(grouped.values())
        return [members for members in blocks if members]

    def arrange(self, items: Sequence[Item], /) -> tuple[Item, ...]:
        """Return the batch in its declared presentation order.

        Raises
        ------
        AcquisitionError
            If a block spans more than the declared distance, or no order
            satisfies the precedence pairs and the no-adjacent distance.
        """
        present = {item_key(item) for item in items}
        before: dict[str, set[str]] = {}
        for earlier, later in self._precedence:
            if earlier in present and later in present:
                before.setdefault(later, set()).add(earlier)
        placed: list[Item] = []
        done: set[str] = set()
        last: dict[str, int] = {}
        adjacent = self._adjacent
        for members in self._blocks(items):
            if self._longest is not None and len(members) - 1 > self._longest:
                msg = (
                    f"the ordering constraint's block of {len(members)} items spans "
                    f"more than the declared {self._longest}"
                )
                raise AcquisitionError(msg)
            pending = list(members)
            while pending:
                position = len(placed)
                chosen = next(
                    (
                        item
                        for item in pending
                        if before.get(item_key(item), set()) <= done
                        and (
                            adjacent is None
                            or (seen := last.get(adjacent.level(item))) is None
                            or position - seen > self._gap
                        )
                    ),
                    None,
                )
                if chosen is None:
                    msg = (
                        "the ordering constraint admits no order of the selected "
                        f"batch at position {position}"
                    )
                    raise AcquisitionError(msg)
                pending.remove(chosen)
                placed.append(chosen)
                done.add(item_key(chosen))
                if adjacent is not None:
                    last[adjacent.level(chosen)] = position
        return tuple(placed)


def _constraint(settings: ListConstraintSettings, /) -> _Constraint | None:
    """Return a list constraint's selection form, or None for size and ordering."""
    match settings:
        case UniquenessSettings(property_expression=expression):
            return _Unique(
                AttributeExpression(
                    expression, role="a uniqueness constraint", many=True
                ),
                None,
            )
        case ConditionalUniquenessSettings(
            property_expression=expression, condition_expression=condition
        ):
            return _Unique(
                AttributeExpression(
                    expression, role="a uniqueness constraint", many=True
                ),
                AttributeExpression(
                    condition, role="a uniqueness condition", truth=True
                ),
            )
        case DiversitySettings():
            return _Diversity(settings)
        case BalanceSettings():
            return _Balance(settings)
    return _Cells.of(settings)


@final
class RC21Acquisition(FrozenValue):
    """Select a high-variance batch under RC2.1 quotas and bead constraints.

    Candidates are allocated to strata by the declared ``allocate_by``
    expression, each stratum's quota proportional to its performance, and are
    ranked by their expected response's squared distance from the pool mean.
    The batch first takes, in rank order, the candidates a declared batch
    coverage, minimum-occurrence or balance constraint still needs, then the
    other candidates in rank order, a candidate any declared constraint does
    not admit being passed over: a uniqueness constraint excludes a repeated
    value, a balance constraint a level at its target, a quantile, grouped
    quantile or grid constraint a full cell, a batch diversity constraint a
    level already in its most batches and a batch balance constraint a level
    above its share. The completed batch is checked against every diversity
    and balance constraint and ordered by a declared ordering constraint. A
    size constraint is checked against the batch size where it is declared.
    Batch constraints count the batches already acquired, each one bead list.
    """

    _noun = "RC2.1 acquisition rule"
    __slots__ = (
        "_allocate_by",
        "_batch_size",
        "_constraints",
        "_ordering",
        "_tolerance",
    )

    def __init__(self, settings: RC21AcquisitionSettings, /) -> None:
        orderings = tuple(
            _Ordering(constraint)
            for constraint in settings.constraints
            if isinstance(constraint, OrderingSettings)
        )
        if len(orderings) > 1:
            msg = "an RC2.1 acquisition declares at most one ordering constraint"
            raise ValueError(msg)
        object.__setattr__(self, "_batch_size", settings.batch_size)
        object.__setattr__(
            self,
            "_allocate_by",
            AttributeExpression(settings.allocate_by, role="an allocation stratum"),
        )
        object.__setattr__(
            self,
            "_constraints",
            (
                *(
                    constraint
                    for declared in settings.constraints
                    if (constraint := _constraint(declared)) is not None
                ),
                *(_BatchLevels(declared) for declared in settings.batch_constraints),
            ),
        )
        object.__setattr__(self, "_ordering", orderings[0] if orderings else None)
        object.__setattr__(self, "_tolerance", settings.share_tolerance)

    @property
    def batch_size(self) -> int:
        """Return the declared batch size."""
        return self._batch_size

    @property
    def expressions(self) -> tuple[AttributeExpression, ...]:
        """Return the allocation expression, then every constraint's expressions."""
        ordering = () if self._ordering is None else self._ordering.expressions
        return (
            self._allocate_by,
            *(
                expression
                for constraint in self._constraints
                for expression in constraint.expressions
            ),
            *ordering,
        )

    def stratum(self, item: Item, /) -> str:
        """Return the stratum an item is allocated under."""
        return self._allocate_by.level(item)

    def select(
        self,
        candidates: Sequence[Item],
        scores: Iterable[AcquisitionScore],
        performance: Mapping[str, float],
        /,
        *,
        previous: Sequence[AcquisitionBatch],
    ) -> AcquisitionBatch:
        """Return a deterministic constrained batch with maximal greedy spread."""
        if not candidates:
            msg = "the acquisition pool is empty"
            raise PoolExhaustedError(msg)
        keys = tuple(item_key(item) for item in candidates)
        if len(set(keys)) != len(keys):
            msg = "the acquisition pool repeats an item identity"
            raise AcquisitionError(msg)
        by_id = {score.item_id: score.expected for score in scores}
        if set(by_id) != set(keys):
            msg = "acquisition scores do not cover the candidate pool"
            raise AcquisitionError(msg)

        quotas, targets, floored = _quotas(performance, self._batch_size)
        strata = {item_key(item): self.stratum(item) for item in candidates}
        unknown = sorted(set(strata.values()) - set(quotas))
        if unknown:
            msg = f"candidate strata have no performance: {unknown!r}"
            raise AcquisitionError(msg)

        center = fmean(by_id.values())
        ranked = sorted(
            candidates,
            key=lambda item: (
                -((by_id[item_key(item)] - center) ** 2),
                item_key(item),
            ),
        )
        states = tuple(
            constraint.begin(candidates, previous, self._batch_size)
            for constraint in self._constraints
        )
        remaining = dict(quotas)
        selected: list[Item] = []
        chosen: set[str] = set()

        def offer(item: Item, /) -> None:
            stratum = strata[item_key(item)]
            if (
                len(selected) == self._batch_size
                or item_key(item) in chosen
                or remaining[stratum] == 0
                or not all(state.admits(item) for state in states)
            ):
                return
            selected.append(item)
            chosen.add(item_key(item))
            for state in states:
                state.take(item)
            remaining[stratum] -= 1

        for item in ranked:
            if any(state.demands(item) for state in states):
                offer(item)
        for item in ranked:
            offer(item)

        if len(selected) != self._batch_size:
            msg = (
                f"the pool satisfies only {len(selected)} of {self._batch_size} "
                "RC2.1 batch positions"
            )
            raise PoolExhaustedError(msg)

        counts = {stratum: quotas[stratum] - remaining[stratum] for stratum in quotas}
        shares = {
            stratum: count / self._batch_size for stratum, count in counts.items()
        }
        for stratum, target in targets.items():
            if abs(shares[stratum] - target) > self._tolerance:
                msg = (
                    f"stratum {stratum!r} has share {shares[stratum]:.6g}, outside "
                    f"{self._tolerance:.6g} of {target:.6g}"
                )
                raise AcquisitionError(msg)

        for state in states:
            violation = state.violation()
            if violation is not None:
                raise AcquisitionError(violation)

        ordered = (
            tuple(selected)
            if self._ordering is None
            else self._ordering.arrange(selected)
        )
        values = tuple(by_id[item_key(item)] for item in ordered)
        mean = fmean(values)
        variance = fmean((value - mean) ** 2 for value in values)
        return AcquisitionBatch(
            ordered, shares, variance, performance=performance, floored=floored
        )

    @override
    def __repr__(self) -> str:
        constrained = tuple(expression.text for expression in self.expressions[1:])
        return (
            f"RC21Acquisition(batch_size={self._batch_size}, "
            f"allocate_by={self._allocate_by.text!r}, constrained={constrained!r}, "
            f"share_tolerance={self._tolerance!r})"
        )


@final
class AcquisitionRulePlugin(FrozenValue):
    """Build the acquisition rule belonging to one registered tag."""

    _noun = "acquisition-rule plugin"
    __slots__ = ("_kind",)

    def __init__(self, kind: str, /) -> None:
        object.__setattr__(self, "_kind", ensure_name("an acquisition kind", kind))

    @property
    def kind(self) -> str:
        """Return the declaration tag this plugin handles."""
        return self._kind

    def construct(self, settings: RC21AcquisitionSettings, /) -> AcquisitionRule:
        """Build the rule after checking the exact resolved tag."""
        if settings.kind != self._kind:
            msg = f"acquisition plugin {self._kind!r} cannot build {settings.kind!r}"
            raise RegistryError(msg, family="acquisition")
        return RC21Acquisition(settings)


RC21_ACQUISITION_PLUGIN = AcquisitionRulePlugin("rc2.1")
"""The registered current RC2.1 acquisition rule."""


def acquisition_rule_for(settings: RC21AcquisitionSettings, /) -> AcquisitionRule:
    """Construct the installed acquisition rule for resolved settings."""
    member = Registry[object]("acquisition").get(settings.kind)
    if not isinstance(member, AcquisitionRulePlugin):
        msg = f"acquisition entry {settings.kind!r} is not an acquisition plugin"
        raise RegistryError(msg, family="acquisition")
    return member.construct(settings)
