"""Declarations of the bead constraints an acquired batch is selected under.

A list constraint holds of each acquired batch; a batch constraint holds of
the sequence of batches a run acquires, each round's batch being one of bead's
lists. Both are declared in bead's serialized shape, discriminated by
``constraint_type``, and name the item attributes they read with a bead DSL
expression over an item's ``item_metadata``, such as ``item['structure']``.
"""

from itertools import pairwise
from math import isclose
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_fraction, ensure_name, ensure_positive

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "BalanceConstraintSpec",
    "BalanceSettings",
    "BatchBalanceConstraintSpec",
    "BatchBalanceSettings",
    "BatchConstraintSettings",
    "BatchConstraintSpec",
    "BatchCoverageConstraintSpec",
    "BatchCoverageSettings",
    "BatchDiversityConstraintSpec",
    "BatchDiversitySettings",
    "BatchMinOccurrenceConstraintSpec",
    "BatchMinOccurrenceSettings",
    "BinningSettings",
    "BinningSpec",
    "CategoricalBinningSettings",
    "CategoricalBinningSpec",
    "ConditionalUniquenessConstraintSpec",
    "ConditionalUniquenessSettings",
    "DiversityConstraintSpec",
    "DiversitySettings",
    "EqualWidthBinningSettings",
    "EqualWidthBinningSpec",
    "GridDimensionSettings",
    "GridDimensionSpec",
    "GridStratificationConstraintSpec",
    "GridStratificationSettings",
    "GroupedQuantileConstraintSpec",
    "GroupedQuantileSettings",
    "ListConstraintSettings",
    "ListConstraintSpec",
    "OrderingConstraintSpec",
    "OrderingPairSpec",
    "OrderingSettings",
    "QuantileBinningSettings",
    "QuantileBinningSpec",
    "QuantileConstraintSpec",
    "QuantileSettings",
    "SizeConstraintSpec",
    "SizeSettings",
    "StdDevBinningSettings",
    "StdDevBinningSpec",
    "ThresholdBinningSettings",
    "ThresholdBinningSpec",
    "UniquenessConstraintSpec",
    "UniquenessSettings",
    "batch_settings",
    "list_settings",
]


def _expression(role: str, value: str, /) -> str:
    """Return a nonempty expression a constraint reads."""
    return ensure_name(role, value)


def _optional_expression(role: str, value: str | None, /) -> str | None:
    """Return an optional nonempty expression a constraint reads."""
    return None if value is None else ensure_name(role, value)


def _quantile_count(value: int, /) -> int:
    """Return a quantile count of at least two."""
    if value < 2:
        msg = f"a quantile count is at least 2: {value!r}"
        raise ValueError(msg)
    return value


class UniquenessSettings(NamedTuple):
    """Resolved bead uniqueness constraint over a selected batch."""

    constraint_type: Literal["uniqueness"]
    property_expression: str


class ConditionalUniquenessSettings(NamedTuple):
    """Resolved bead conditional uniqueness constraint over a selected batch."""

    constraint_type: Literal["conditional_uniqueness"]
    property_expression: str
    condition_expression: str


class DiversitySettings(NamedTuple):
    """Resolved bead diversity constraint over a selected batch."""

    constraint_type: Literal["diversity"]
    property_expression: str
    min_unique_values: int


class SizeSettings(NamedTuple):
    """Resolved bead size constraint over a selected batch."""

    constraint_type: Literal["size"]
    min_size: int | None
    max_size: int | None
    exact_size: int | None


class BalanceSettings(NamedTuple):
    """Resolved bead balance constraint over a selected batch."""

    constraint_type: Literal["balance"]
    property_expression: str
    target_counts: Mapping[str, int] | None
    tolerance: float


class QuantileSettings(NamedTuple):
    """Resolved bead quantile constraint over a selected batch."""

    constraint_type: Literal["quantile"]
    property_expression: str
    n_quantiles: int
    items_per_quantile: int


class GroupedQuantileSettings(NamedTuple):
    """Resolved bead grouped quantile constraint over a selected batch."""

    constraint_type: Literal["grouped_quantile"]
    property_expression: str
    group_by_expression: str
    n_quantiles: int
    items_per_quantile: int


class QuantileBinningSettings(NamedTuple):
    """Resolved equal-frequency binning of one grid dimension."""

    binning: Literal["quantile"]
    n_quantiles: int


class EqualWidthBinningSettings(NamedTuple):
    """Resolved equal-width binning of one grid dimension."""

    binning: Literal["equal_width"]
    n_bins: int
    range_min: float | None
    range_max: float | None


class ThresholdBinningSettings(NamedTuple):
    """Resolved binning of one grid dimension at declared edges."""

    binning: Literal["threshold"]
    edges: tuple[float, ...]


class StdDevBinningSettings(NamedTuple):
    """Resolved binning of one grid dimension at standard-deviation offsets."""

    binning: Literal["stddev"]
    k_values: tuple[float, ...]


class CategoricalBinningSettings(NamedTuple):
    """Resolved binning of one grid dimension by its level names."""

    binning: Literal["categorical"]
    categories: tuple[str, ...] | None
    include_other: bool


type BinningSettings = (
    QuantileBinningSettings
    | EqualWidthBinningSettings
    | ThresholdBinningSettings
    | StdDevBinningSettings
    | CategoricalBinningSettings
)
"""A resolved binning strategy of one grid dimension."""


class GridDimensionSettings(NamedTuple):
    """Resolved axis of a stratification grid."""

    property_expression: str
    binning: BinningSettings


class GridStratificationSettings(NamedTuple):
    """Resolved bead grid stratification constraint over a selected batch."""

    constraint_type: Literal["grid_stratification"]
    dimensions: tuple[GridDimensionSettings, ...]
    group_by_expression: str | None
    items_per_cell: int


class OrderingSettings(NamedTuple):
    """Resolved bead ordering constraint over a selected batch's order."""

    constraint_type: Literal["ordering"]
    precedence_pairs: tuple[tuple[str, str], ...]
    no_adjacent_property: str | None
    block_by_property: str | None
    min_distance: int | None
    max_distance: int | None
    practice_item_property: str | None


type ListConstraintSettings = (
    UniquenessSettings
    | ConditionalUniquenessSettings
    | DiversitySettings
    | SizeSettings
    | BalanceSettings
    | QuantileSettings
    | GroupedQuantileSettings
    | GridStratificationSettings
    | OrderingSettings
)
"""A resolved bead list constraint the RC2.1 selector enforces."""


class BatchCoverageSettings(NamedTuple):
    """Resolved bead coverage constraint over every acquired batch."""

    constraint_type: Literal["coverage"]
    property_expression: str
    target_values: tuple[str, ...] | None
    min_coverage: float


class BatchBalanceSettings(NamedTuple):
    """Resolved bead balance constraint over every acquired batch."""

    constraint_type: Literal["balance"]
    property_expression: str
    target_distribution: Mapping[str, float]
    tolerance: float


class BatchDiversitySettings(NamedTuple):
    """Resolved bead diversity constraint over every acquired batch."""

    constraint_type: Literal["diversity"]
    property_expression: str
    max_lists_per_value: int


class BatchMinOccurrenceSettings(NamedTuple):
    """Resolved bead minimum-occurrence constraint over every acquired batch."""

    constraint_type: Literal["min_occurrence"]
    property_expression: str
    min_occurrences: int


type BatchConstraintSettings = (
    BatchCoverageSettings
    | BatchBalanceSettings
    | BatchDiversitySettings
    | BatchMinOccurrenceSettings
)
"""A resolved bead batch constraint the RC2.1 selector enforces across rounds."""


class _ListDeclaration(Protocol):
    """A validated list-constraint declaration exposing immutable settings."""

    def settings(self) -> ListConstraintSettings:
        """Return the resolved constraint."""
        ...


class ListConstraintSpec(
    dx.TaggedUnion, discriminator="constraint_type", extra="forbid"
):
    """Declare one bead list constraint, in bead's serialized shape."""


class UniquenessConstraintSpec(ListConstraintSpec):
    """No two selected items share the value an expression yields.

    Parameters
    ----------
    property_expression : str
        A bead expression over an item's attributes yielding a level name or a
        list of them.
    """

    constraint_type: Literal["uniqueness"] = "uniqueness"
    property_expression: str

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a uniqueness expression", value)

    def settings(self) -> UniquenessSettings:
        """Discard the boundary declaration into immutable settings."""
        return UniquenessSettings(self.constraint_type, self.property_expression)


class ConditionalUniquenessConstraintSpec(ListConstraintSpec):
    """No two selected items a condition holds of share an expression's value.

    Parameters
    ----------
    property_expression : str
        A bead expression over an item's attributes yielding a level name or a
        list of them.
    condition_expression : str
        A bead expression over an item's attributes yielding a truth value.
    """

    constraint_type: Literal["conditional_uniqueness"] = "conditional_uniqueness"
    property_expression: str
    condition_expression: str

    @dx.validates("property_expression", "condition_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a conditional uniqueness expression", value)

    def settings(self) -> ConditionalUniquenessSettings:
        """Discard the boundary declaration into immutable settings."""
        return ConditionalUniquenessSettings(
            self.constraint_type, self.property_expression, self.condition_expression
        )


class DiversityConstraintSpec(ListConstraintSpec):
    """A selected batch holds a least number of an expression's distinct values.

    Parameters
    ----------
    property_expression : str
        A bead expression over an item's attributes yielding a level name or a
        list of them.
    min_unique_values : int
        The least number of distinct values.
    """

    constraint_type: Literal["diversity"] = "diversity"
    property_expression: str
    min_unique_values: int

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a diversity expression", value)

    @dx.validates("min_unique_values")
    @classmethod
    def _minimum_is_positive(cls, value: int) -> int:
        return ensure_positive("a diversity minimum", value)

    def settings(self) -> DiversitySettings:
        """Discard the boundary declaration into immutable settings."""
        return DiversitySettings(
            self.constraint_type, self.property_expression, self.min_unique_values
        )


class SizeConstraintSpec(ListConstraintSpec):
    """A selected batch's size lies within declared bounds.

    Parameters
    ----------
    min_size, max_size, exact_size : int | None
        The bounds, each None when not declared.
    """

    constraint_type: Literal["size"] = "size"
    min_size: int | None
    max_size: int | None
    exact_size: int | None

    def settings(self) -> SizeSettings:
        """Discard the boundary declaration into immutable settings."""
        return SizeSettings(
            self.constraint_type, self.min_size, self.max_size, self.exact_size
        )


class BalanceConstraintSpec(ListConstraintSpec):
    """A selected batch holds each level of an expression in balanced counts.

    Parameters
    ----------
    property_expression : str
        A bead expression over an item's attributes yielding a level name.
    target_counts : dict[str, int] | None
        The count each named level should reach, or None for equal counts of
        every level the candidate pool holds.
    tolerance : float
        Largest deviation of a level's count from its target, as a proportion
        of the target (of one, for a target of zero).
    """

    constraint_type: Literal["balance"] = "balance"
    property_expression: str
    target_counts: dict[str, int] | None
    tolerance: float

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a balance expression", value)

    @dx.validates("target_counts")
    @classmethod
    def _targets_count_from_zero(
        cls, value: dict[str, int] | None
    ) -> dict[str, int] | None:
        if value is None:
            return value
        for level, count in value.items():
            ensure_name("a balance level", level)
            if count < 0:
                msg = f"a balance target is nonnegative: {level!r} has {count!r}"
                raise ValueError(msg)
        return value

    @dx.validates("tolerance")
    @classmethod
    def _tolerance_is_a_fraction(cls, value: float) -> float:
        return ensure_fraction("a balance tolerance", value)

    def settings(self) -> BalanceSettings:
        """Discard the boundary declaration into immutable settings."""
        return BalanceSettings(
            self.constraint_type,
            self.property_expression,
            None
            if self.target_counts is None
            else MappingProxyType(dict(self.target_counts)),
            self.tolerance,
        )


class QuantileConstraintSpec(ListConstraintSpec):
    """A selected batch spreads over the quantiles of a numeric expression.

    Parameters
    ----------
    property_expression : str
        A bead expression over an item's attributes yielding a number.
    n_quantiles : int
        Equal-frequency bins the candidate pool is divided into, at least two.
    items_per_quantile : int
        Most items of one bin a batch selects.
    """

    constraint_type: Literal["quantile"] = "quantile"
    property_expression: str
    n_quantiles: int
    items_per_quantile: int

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a quantile expression", value)

    @dx.validates("n_quantiles")
    @classmethod
    def _at_least_two(cls, value: int) -> int:
        return _quantile_count(value)

    @dx.validates("items_per_quantile")
    @classmethod
    def _per_bin_is_positive(cls, value: int) -> int:
        return ensure_positive("items per quantile", value)

    def settings(self) -> QuantileSettings:
        """Discard the boundary declaration into immutable settings."""
        return QuantileSettings(
            self.constraint_type,
            self.property_expression,
            self.n_quantiles,
            self.items_per_quantile,
        )


class GroupedQuantileConstraintSpec(ListConstraintSpec):
    """A selected batch spreads over a numeric expression's quantiles per group.

    Parameters
    ----------
    property_expression : str
        A bead expression over an item's attributes yielding a number.
    group_by_expression : str
        A bead expression yielding the level name quantiles are computed within.
    n_quantiles : int
        Equal-frequency bins each group is divided into, at least two.
    items_per_quantile : int
        Most items of one group's bin a batch selects.
    """

    constraint_type: Literal["grouped_quantile"] = "grouped_quantile"
    property_expression: str
    group_by_expression: str
    n_quantiles: int
    items_per_quantile: int

    @dx.validates("property_expression", "group_by_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a grouped quantile expression", value)

    @dx.validates("n_quantiles")
    @classmethod
    def _at_least_two(cls, value: int) -> int:
        return _quantile_count(value)

    @dx.validates("items_per_quantile")
    @classmethod
    def _per_bin_is_positive(cls, value: int) -> int:
        return ensure_positive("items per quantile", value)

    def settings(self) -> GroupedQuantileSettings:
        """Discard the boundary declaration into immutable settings."""
        return GroupedQuantileSettings(
            self.constraint_type,
            self.property_expression,
            self.group_by_expression,
            self.n_quantiles,
            self.items_per_quantile,
        )


class _BinningDeclaration(Protocol):
    """A validated binning declaration exposing immutable settings."""

    def settings(self) -> BinningSettings:
        """Return the resolved binning."""
        ...


class BinningSpec(dx.TaggedUnion, discriminator="binning", extra="forbid"):
    """Declare how one grid dimension's values are binned, in bead's shape."""


class QuantileBinningSpec(BinningSpec):
    """Bin a numeric dimension into equal-frequency bins.

    Parameters
    ----------
    n_quantiles : int
        Bins, at least two.
    """

    binning: Literal["quantile"] = "quantile"
    n_quantiles: int

    @dx.validates("n_quantiles")
    @classmethod
    def _at_least_two(cls, value: int) -> int:
        return _quantile_count(value)

    def settings(self) -> QuantileBinningSettings:
        """Discard the boundary declaration into immutable settings."""
        return QuantileBinningSettings(self.binning, self.n_quantiles)


class EqualWidthBinningSpec(BinningSpec):
    """Bin a numeric dimension into equal-width bins.

    Parameters
    ----------
    n_bins : int
        Bins.
    range_min, range_max : float | None
        The binned range's ends, each None for the candidates' own extreme.
    """

    binning: Literal["equal_width"] = "equal_width"
    n_bins: int
    range_min: float | None
    range_max: float | None

    @dx.validates("n_bins")
    @classmethod
    def _bins_are_positive(cls, value: int) -> int:
        return ensure_positive("an equal-width bin count", value)

    @dx.model_validator()
    def _range_is_ordered(self) -> EqualWidthBinningSpec:
        low, high = self.range_min, self.range_max
        if low is not None and high is not None and not low < high:
            msg = f"an equal-width range ends above its start: {low!r} to {high!r}"
            raise ValueError(msg)
        return self

    def settings(self) -> EqualWidthBinningSettings:
        """Discard the boundary declaration into immutable settings."""
        return EqualWidthBinningSettings(
            self.binning, self.n_bins, self.range_min, self.range_max
        )


class ThresholdBinningSpec(BinningSpec):
    """Bin a numeric dimension at declared edges.

    Parameters
    ----------
    edges : tuple[float, ...]
        Strictly increasing bin edges, at least one.
    """

    binning: Literal["threshold"] = "threshold"
    edges: tuple[float, ...]

    @dx.validates("edges")
    @classmethod
    def _edges_increase(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not low < high for low, high in pairwise(value)):
            msg = f"threshold edges are a nonempty increasing sequence: {value!r}"
            raise ValueError(msg)
        return tuple(ensure_finite("a threshold edge", edge) for edge in value)

    def settings(self) -> ThresholdBinningSettings:
        """Discard the boundary declaration into immutable settings."""
        return ThresholdBinningSettings(self.binning, self.edges)


class StdDevBinningSpec(BinningSpec):
    """Bin a numeric dimension at offsets of standard deviations from its mean.

    Parameters
    ----------
    k_values : tuple[float, ...]
        Strictly increasing offsets, in standard deviations, at least one.
    """

    binning: Literal["stddev"] = "stddev"
    k_values: tuple[float, ...]

    @dx.validates("k_values")
    @classmethod
    def _offsets_increase(cls, value: tuple[float, ...]) -> tuple[float, ...]:
        if not value or any(not low < high for low, high in pairwise(value)):
            msg = f"standard-deviation offsets increase strictly: {value!r}"
            raise ValueError(msg)
        return tuple(ensure_finite("a standard-deviation offset", k) for k in value)

    def settings(self) -> StdDevBinningSettings:
        """Discard the boundary declaration into immutable settings."""
        return StdDevBinningSettings(self.binning, self.k_values)


class CategoricalBinningSpec(BinningSpec):
    """Bin a dimension by its level names.

    Parameters
    ----------
    categories : tuple[str, ...] | None
        The levels, one bin each in this order, or None for every level the
        candidates hold in sorted order.
    include_other : bool
        Whether one further bin holds every level the categories omit; when
        not, such a level is refused.
    """

    binning: Literal["categorical"] = "categorical"
    categories: tuple[str, ...] | None
    include_other: bool

    @dx.validates("categories")
    @classmethod
    def _categories_are_distinct(
        cls, value: tuple[str, ...] | None
    ) -> tuple[str, ...] | None:
        if value is None:
            return value
        if not value or len(set(value)) != len(value):
            msg = f"categorical bins name distinct levels: {value!r}"
            raise ValueError(msg)
        return tuple(ensure_name("a categorical bin", level) for level in value)

    def settings(self) -> CategoricalBinningSettings:
        """Discard the boundary declaration into immutable settings."""
        return CategoricalBinningSettings(
            self.binning, self.categories, self.include_other
        )


class GridDimensionSpec(dx.Model, extra="forbid"):
    """Declare one axis of a stratification grid.

    Parameters
    ----------
    property_expression : str
        A bead expression yielding the dimension's value: a number for a
        numeric binning, a level name for a categorical one.
    binning : BinningSpec
        How the value is binned.
    """

    property_expression: str
    binning: BinningSpec

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a grid dimension expression", value)

    def settings(self) -> GridDimensionSettings:
        """Discard the boundary declaration into immutable settings."""
        return GridDimensionSettings(
            self.property_expression,
            cast("_BinningDeclaration", self.binning).settings(),
        )


class GridStratificationConstraintSpec(ListConstraintSpec):
    """A selected batch spreads over the cells of a binned grid.

    Parameters
    ----------
    dimensions : tuple[GridDimensionSpec, ...]
        The grid's axes, at least one.
    group_by_expression : str | None
        A bead expression yielding the level name numeric bins are computed
        within, or None for bins over the whole pool.
    items_per_cell : int
        Most items of one cell a batch selects.
    """

    constraint_type: Literal["grid_stratification"] = "grid_stratification"
    dimensions: tuple[GridDimensionSpec, ...]
    group_by_expression: str | None
    items_per_cell: int

    @dx.validates("dimensions")
    @classmethod
    def _one_dimension_at_least(
        cls, value: tuple[GridDimensionSpec, ...]
    ) -> tuple[GridDimensionSpec, ...]:
        if not value:
            msg = "a stratification grid has at least one dimension"
            raise ValueError(msg)
        return value

    @dx.validates("group_by_expression")
    @classmethod
    def _group_is_named(cls, value: str | None) -> str | None:
        return _optional_expression("a grid group expression", value)

    @dx.validates("items_per_cell")
    @classmethod
    def _per_cell_is_positive(cls, value: int) -> int:
        return ensure_positive("items per grid cell", value)

    def settings(self) -> GridStratificationSettings:
        """Discard the boundary declaration into immutable settings."""
        return GridStratificationSettings(
            self.constraint_type,
            tuple(dimension.settings() for dimension in self.dimensions),
            self.group_by_expression,
            self.items_per_cell,
        )


class OrderingPairSpec(dx.Model, extra="forbid"):
    """Declare that one selected item is presented before another.

    Parameters
    ----------
    before, after : str
        The identities of the earlier and the later item.
    """

    before: str
    after: str

    @dx.validates("before", "after")
    @classmethod
    def _item_is_named(cls, value: str) -> str:
        return ensure_name("an ordered item", value)


class OrderingConstraintSpec(ListConstraintSpec):
    """Order a selected batch for presentation.

    The batch is written in the order this constraint fixes: practice items
    first, then the main items, each in blocks of one level where blocks are
    declared, each item after every item a precedence pair puts before it, and
    items sharing a level separated where that is declared. Within those rules
    the items keep their selection order, so the order is deterministic.

    Parameters
    ----------
    precedence_pairs : tuple[OrderingPairSpec, ...]
        Pairs of items the earlier of which precedes the later when both are
        selected.
    no_adjacent_property : str | None
        A bead expression yielding the level name no two neighboring items
        may share.
    block_by_property : str | None
        A bead expression yielding the level name items are grouped into
        contiguous blocks by.
    min_distance : int | None
        Least number of items between two that share the no-adjacent level;
        one when not declared. Requires ``no_adjacent_property``.
    max_distance : int | None
        Largest span from the first to the last item of one block. Requires
        ``block_by_property``.
    practice_item_property : str | None
        A bead expression yielding whether an item is a practice item.
    randomize_within_blocks : False
        Refused when true: randomizing presentation within blocks belongs to
        the presentation runtime, and an acquired batch is ordered
        deterministically.
    """

    constraint_type: Literal["ordering"] = "ordering"
    precedence_pairs: tuple[OrderingPairSpec, ...] = ()
    no_adjacent_property: str | None = None
    block_by_property: str | None = None
    min_distance: int | None = None
    max_distance: int | None = None
    practice_item_property: str | None = None
    randomize_within_blocks: Literal[False] = False

    @dx.validates("no_adjacent_property", "block_by_property", "practice_item_property")
    @classmethod
    def _expression_is_named(cls, value: str | None) -> str | None:
        return _optional_expression("an ordering expression", value)

    @dx.validates("min_distance", "max_distance")
    @classmethod
    def _distance_is_positive(cls, value: int | None) -> int | None:
        return None if value is None else ensure_positive("an ordering distance", value)

    @dx.model_validator()
    def _distances_have_their_property(self) -> OrderingConstraintSpec:
        if self.min_distance is not None and self.no_adjacent_property is None:
            msg = "min_distance requires no_adjacent_property"
            raise ValueError(msg)
        if self.max_distance is not None and self.block_by_property is None:
            msg = "max_distance requires block_by_property"
            raise ValueError(msg)
        return self

    def settings(self) -> OrderingSettings:
        """Discard the boundary declaration into immutable settings."""
        return OrderingSettings(
            self.constraint_type,
            tuple((pair.before, pair.after) for pair in self.precedence_pairs),
            self.no_adjacent_property,
            self.block_by_property,
            self.min_distance,
            self.max_distance,
            self.practice_item_property,
        )


class _BatchDeclaration(Protocol):
    """A validated batch-constraint declaration exposing immutable settings."""

    def settings(self) -> BatchConstraintSettings:
        """Return the resolved constraint."""
        ...


class BatchConstraintSpec(
    dx.TaggedUnion, discriminator="constraint_type", extra="forbid"
):
    """Declare one bead batch constraint over every batch a run acquires."""


class BatchCoverageConstraintSpec(BatchConstraintSpec):
    """The acquired batches together cover the levels of an expression.

    Each round's selection first takes, for every level not yet covered, the
    best-ranked admissible candidate holding it, until the covered share
    reaches ``min_coverage``.

    Parameters
    ----------
    property_expression : str
        A bead expression yielding a level name.
    target_values : tuple[str, ...] | None
        The levels to cover, or None for every level the pool holds.
    min_coverage : float
        The share of the target levels to cover.
    """

    constraint_type: Literal["coverage"] = "coverage"
    property_expression: str
    target_values: tuple[str, ...] | None
    min_coverage: float

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a coverage expression", value)

    @dx.validates("min_coverage")
    @classmethod
    def _coverage_is_a_fraction(cls, value: float) -> float:
        return ensure_fraction("a minimum coverage", value)

    def settings(self) -> BatchCoverageSettings:
        """Discard the boundary declaration into immutable settings."""
        return BatchCoverageSettings(
            self.constraint_type,
            self.property_expression,
            self.target_values,
            self.min_coverage,
        )


class BatchBalanceConstraintSpec(BatchConstraintSpec):
    """The acquired batches together hold an expression's levels in proportion.

    Parameters
    ----------
    property_expression : str
        A bead expression yielding a level name.
    target_distribution : dict[str, float]
        Each level's target share, the shares summing to one.
    tolerance : float
        Largest gap between a level's realized share and its target.
    """

    constraint_type: Literal["balance"] = "balance"
    property_expression: str
    target_distribution: dict[str, float]
    tolerance: float

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a batch balance expression", value)

    @dx.validates("target_distribution")
    @classmethod
    def _distribution_sums_to_one(cls, value: dict[str, float]) -> dict[str, float]:
        for level, share in value.items():
            ensure_name("a balanced level", level)
            ensure_fraction(f"level {level!r}'s target share", share)
        total = sum(value.values())
        if not isclose(total, 1.0):
            msg = f"a target distribution sums to {total!r}, not 1"
            raise ValueError(msg)
        return value

    @dx.validates("tolerance")
    @classmethod
    def _tolerance_is_a_fraction(cls, value: float) -> float:
        return ensure_fraction("a batch balance tolerance", value)

    def settings(self) -> BatchBalanceSettings:
        """Discard the boundary declaration into immutable settings."""
        return BatchBalanceSettings(
            self.constraint_type,
            self.property_expression,
            MappingProxyType(dict(self.target_distribution)),
            self.tolerance,
        )


class BatchDiversityConstraintSpec(BatchConstraintSpec):
    """No level of an expression is acquired in more than a number of batches.

    Parameters
    ----------
    property_expression : str
        A bead expression yielding a level name.
    max_lists_per_value : int
        Most batches one level may appear in.
    """

    constraint_type: Literal["diversity"] = "diversity"
    property_expression: str
    max_lists_per_value: int

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a batch diversity expression", value)

    @dx.validates("max_lists_per_value")
    @classmethod
    def _maximum_is_positive(cls, value: int) -> int:
        return ensure_positive("a batch diversity maximum", value)

    def settings(self) -> BatchDiversitySettings:
        """Discard the boundary declaration into immutable settings."""
        return BatchDiversitySettings(
            self.constraint_type, self.property_expression, self.max_lists_per_value
        )


class BatchMinOccurrenceConstraintSpec(BatchConstraintSpec):
    """Every level of an expression is acquired a least number of times.

    Each round's selection first takes, for every level the pool holds that
    the acquired batches hold fewer times than ``min_occurrences``, the
    best-ranked admissible candidates holding it.

    Parameters
    ----------
    property_expression : str
        A bead expression yielding a level name.
    min_occurrences : int
        Least number of acquired items holding each level.
    """

    constraint_type: Literal["min_occurrence"] = "min_occurrence"
    property_expression: str
    min_occurrences: int

    @dx.validates("property_expression")
    @classmethod
    def _expression_is_named(cls, value: str) -> str:
        return _expression("a minimum-occurrence expression", value)

    @dx.validates("min_occurrences")
    @classmethod
    def _minimum_is_positive(cls, value: int) -> int:
        return ensure_positive("a minimum occurrence count", value)

    def settings(self) -> BatchMinOccurrenceSettings:
        """Discard the boundary declaration into immutable settings."""
        return BatchMinOccurrenceSettings(
            self.constraint_type, self.property_expression, self.min_occurrences
        )


def list_settings(constraint: ListConstraintSpec, /) -> ListConstraintSettings:
    """Return one validated list constraint's settings."""
    return cast("_ListDeclaration", constraint).settings()


def batch_settings(constraint: BatchConstraintSpec, /) -> BatchConstraintSettings:
    """Return one validated batch constraint's settings."""
    return cast("_BatchDeclaration", constraint).settings()
