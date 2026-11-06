"""Immutable algorithm policy and swappable symbolic parse capabilities."""

from enum import Enum
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    override,
    runtime_checkable,
)

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.deduce._ports import CellPruner, Semiring
from lofi.deduce._spans import SpanIndex

if TYPE_CHECKING:
    from lofi.deduce._forest import PackedForest
    from lofi.deduce._inside import InsideChart
    from lofi.deduce._ports import NamedRoots
    from lofi.deduce._values import ItemKey
    from lofi.judge import Rule

__all__ = [
    "CKY",
    "CONTIGUOUS_SPANS",
    "NATIVE_CHARTS",
    "ChartKernel",
    "ChartStore",
    "CkyStrategy",
    "ContiguousSpanIndex",
    "InsideSweep",
    "NativeChartStore",
    "ParsePolicy",
    "ParseSettings",
    "ParseStrategy",
    "Reachability",
    "Retain",
    "SpanIndexFactory",
]


class InsideSweep[T](NamedTuple):
    """Unary closure evidence returned by one inside-chart strategy."""

    edges: tuple[tuple[ItemKey[T], Rule[T], ItemKey[T]], ...]
    concluded: tuple[ItemKey[T], ...]
    truncated: bool


@runtime_checkable
class SpanIndexFactory(Protocol):
    """Build the region index used by one symbolic parse."""

    @property
    def name(self) -> str:
        """Return the installed selection name."""

    def build(self, tokens: int, /) -> SpanIndex:
        """Build an index for a token count."""


@runtime_checkable
class ChartKernel[T, W](Protocol):
    """Expose the chart construction and sweep operations a policy composes."""

    def inside_chart(
        self,
        index: SpanIndex,
        semiring: Semiring[W],
        *,
        normal_form: str,
    ) -> InsideChart[T, W]:
        """Construct an inside chart sharing this kernel's identifiers."""

    def forest_chart(
        self,
        index: SpanIndex,
        semiring: Semiring[W],
        pruner: CellPruner[T, W] | None,
        roots: NamedRoots[T] | None,
        *,
        normal_form: str,
        prune_lexical: bool,
    ) -> PackedForest[T, W]:
        """Construct a packed forest sharing this kernel's identifiers."""

    def sweep_inside(
        self,
        chart: InsideChart[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
    ) -> InsideSweep[T]:
        """Fill an inside chart and return its root unary closure."""

    def sweep_forest(
        self,
        chart: PackedForest[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
    ) -> None:
        """Fill a packed forest."""


@runtime_checkable
class ChartStore(Protocol):
    """Construct chart storage without fixing it in the deduction engine."""

    @property
    def name(self) -> str:
        """Return the installed selection name."""

    def inside[T, W](
        self,
        kernel: ChartKernel[T, W],
        index: SpanIndex,
        semiring: Semiring[W],
        *,
        normal_form: str,
    ) -> InsideChart[T, W]:
        """Build inside-only chart storage."""

    def forest[T, W](
        self,
        kernel: ChartKernel[T, W],
        index: SpanIndex,
        semiring: Semiring[W],
        pruner: CellPruner[T, W] | None,
        roots: NamedRoots[T] | None,
        *,
        normal_form: str,
        prune_lexical: bool,
    ) -> PackedForest[T, W]:
        """Build packed forest chart storage."""


@runtime_checkable
class ParseStrategy(Protocol):
    """Choose the order in which a chart kernel admits deduction steps."""

    @property
    def name(self) -> str:
        """Return the installed selection name."""

    def inside[T, W](
        self,
        kernel: ChartKernel[T, W],
        chart: InsideChart[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
        /,
    ) -> InsideSweep[T]:
        """Fill an inside chart under this traversal strategy."""

    def forest[T, W](
        self,
        kernel: ChartKernel[T, W],
        chart: PackedForest[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
        /,
    ) -> None:
        """Fill a packed forest under this traversal strategy."""


class ContiguousSpanIndex(FrozenValue):
    """Build the contiguous half-open span index."""

    __slots__ = ()
    _noun = "span-index factory"

    @property
    def name(self) -> str:
        """Return the installed selection name."""
        return "contiguous"

    def build(self, tokens: int, /) -> SpanIndex:
        """Build the contiguous index for a token count."""
        return SpanIndex(tokens)


class NativeChartStore(FrozenValue):
    """Construct the native inside table or packed forest selected by retention."""

    __slots__ = ()
    _noun = "chart-store factory"

    @property
    def name(self) -> str:
        """Return the installed selection name."""
        return "native"

    def inside[T, W](
        self,
        kernel: ChartKernel[T, W],
        index: SpanIndex,
        semiring: Semiring[W],
        *,
        normal_form: str,
    ) -> InsideChart[T, W]:
        """Construct native inside-only storage."""
        return kernel.inside_chart(index, semiring, normal_form=normal_form)

    def forest[T, W](
        self,
        kernel: ChartKernel[T, W],
        index: SpanIndex,
        semiring: Semiring[W],
        pruner: CellPruner[T, W] | None,
        roots: NamedRoots[T] | None,
        *,
        normal_form: str,
        prune_lexical: bool,
    ) -> PackedForest[T, W]:
        """Construct native packed forest storage."""
        return kernel.forest_chart(
            index,
            semiring,
            pruner,
            roots,
            normal_form=normal_form,
            prune_lexical=prune_lexical,
        )


class CkyStrategy(FrozenValue):
    """Run the kernel's exhaustive bottom-up CKY traversal."""

    __slots__ = ()
    _noun = "parse strategy"

    @property
    def name(self) -> str:
        """Return the installed selection name."""
        return "cky"

    def inside[T, W](
        self,
        kernel: ChartKernel[T, W],
        chart: InsideChart[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
        /,
    ) -> InsideSweep[T]:
        """Fill an inside chart from narrow spans to wide spans."""
        return kernel.sweep_inside(chart, active_arities, unary_depth, threads)

    def forest[T, W](
        self,
        kernel: ChartKernel[T, W],
        chart: PackedForest[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
        /,
    ) -> None:
        """Fill a packed forest from narrow spans to wide spans."""
        kernel.sweep_forest(chart, active_arities, unary_depth, threads)


CKY = CkyStrategy()
"""The exhaustive bottom-up chart strategy."""

CONTIGUOUS_SPANS = ContiguousSpanIndex()
"""The contiguous half-open span-index factory."""

NATIVE_CHARTS = NativeChartStore()
"""The native chart-storage factory."""


class Retain(Enum):
    """What a parse keeps of the derivations it builds."""

    FOREST = "forest"
    """Every item and every step, as a packed forest."""

    INSIDE = "inside"
    """The inside weight of every exact category over every span, and no step."""


class Reachability(Enum):
    """Which chart items one parse constructs."""

    ALL = "all"
    """Every item licensed within the declared structural bounds."""

    ROOTS = "roots"
    """Only items that participate in a derivation of a named root category."""


DEFAULT_REFINEMENT_PAIRS: Final = 1 << 16
"""The default bound on one lexical refinement of a root-reachability plan."""


class ParsePolicy[RetainT: Retain](FrozenValue):
    """The algorithm choices shared by every symbolic parse in one run.

    Parameters
    ----------
    retain : RetainT
        Whether the parse keeps a packed forest or only inside weights.
    reachability : Reachability
        Whether the parse constructs all items or only root-reaching items.
    refinement_pairs : int
        The most binding-and-candidate pairs a root-reachability plan
        enumerates when it narrows one rule row's demands on its lexical
        children. A row past the bound keeps its wider demands, so a lower
        bound plans faster and prunes less without losing a root-reaching
        item. Zero never refines.

    Raises
    ------
    TypeError
        If either choice is not its lowered enum value.
    ValueError
        If the refinement bound is negative.
    """

    __slots__ = ("_reachability", "_refinement_pairs", "_retain")

    _noun = "parse policy"

    def __init__(
        self,
        retain: RetainT,
        reachability: Reachability,
        *,
        refinement_pairs: int = DEFAULT_REFINEMENT_PAIRS,
    ) -> None:
        if not isinstance(retain, Retain):
            msg = f"a parse retention policy is not a Retain value: {retain!r}"
            raise TypeError(msg)
        if not isinstance(reachability, Reachability):
            msg = (
                "a parse reachability policy is not a Reachability value: "
                f"{reachability!r}"
            )
            raise TypeError(msg)

        object.__setattr__(self, "_retain", retain)
        object.__setattr__(self, "_reachability", reachability)
        object.__setattr__(
            self,
            "_refinement_pairs",
            ensure_non_negative("refinement_pairs", refinement_pairs),
        )

    @property
    def retain(self) -> RetainT:
        """Return what a parse keeps."""
        return self._retain

    @property
    def reachability(self) -> Reachability:
        """Return which chart items a parse constructs."""
        return self._reachability

    @property
    def refinement_pairs(self) -> int:
        """Return the bound on one lexical refinement of a reachability plan."""
        return self._refinement_pairs

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two policies make the same choices."""
        if type(other) is not ParsePolicy:
            return NotImplemented

        return (
            self._retain is other._retain
            and self._reachability is other._reachability
            and self._refinement_pairs == other._refinement_pairs
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the algorithm choices."""
        return hash(
            (ParsePolicy, self._retain, self._reachability, self._refinement_pairs)
        )

    @override
    def __repr__(self) -> str:
        """Return an explicit constructor rendering."""
        return (
            f"ParsePolicy(retain={self._retain!r}, "
            f"reachability={self._reachability!r}, "
            f"refinement_pairs={self._refinement_pairs!r})"
        )


class ParseSettings[W](FrozenValue):
    """Installed components of one validated symbolic parse declaration.

    The settings are generic over the weight type of their semiring, so a
    consumer that reads weights of one type, such as the float log
    probabilities A* ranks by, takes settings of that type.
    """

    __slots__ = (
        "_chart_store",
        "_measure_pruned_mass",
        "_policy",
        "_prune_lexical",
        "_pruner",
        "_semiring",
        "_span_index",
        "_strategy",
    )
    _noun = "parse settings"

    def __init__(
        self,
        policy: ParsePolicy[Retain],
        strategy: ParseStrategy,
        span_index: SpanIndexFactory,
        chart_store: ChartStore,
        pruner: CellPruner[object, W] | None,
        semiring: Semiring[W],
        *,
        prune_lexical: bool,
        measure_pruned_mass: bool,
    ) -> None:
        if not isinstance(strategy, ParseStrategy):
            msg = f"a parse strategy does not implement ParseStrategy: {strategy!r}"
            raise TypeError(msg)
        if not isinstance(span_index, SpanIndexFactory):
            msg = (
                "a span-index factory does not implement SpanIndexFactory: "
                f"{span_index!r}"
            )
            raise TypeError(msg)
        if not isinstance(chart_store, ChartStore):
            msg = (
                f"a chart-store factory does not implement ChartStore: {chart_store!r}"
            )
            raise TypeError(msg)
        if pruner is not None and not isinstance(pruner, CellPruner):
            msg = f"a cell pruner does not implement CellPruner: {pruner!r}"
            raise TypeError(msg)
        if not isinstance(semiring, Semiring):
            msg = f"a semiring does not implement Semiring: {semiring!r}"
            raise TypeError(msg)
        if measure_pruned_mass and pruner is None:
            msg = "parse settings measure pruned mass but declare no pruner"
            raise ValueError(msg)

        object.__setattr__(self, "_policy", policy)
        object.__setattr__(self, "_strategy", strategy)
        object.__setattr__(self, "_span_index", span_index)
        object.__setattr__(self, "_chart_store", chart_store)
        object.__setattr__(self, "_pruner", pruner)
        object.__setattr__(self, "_semiring", semiring)
        object.__setattr__(self, "_prune_lexical", prune_lexical)
        object.__setattr__(self, "_measure_pruned_mass", measure_pruned_mass)

    @property
    def policy(self) -> ParsePolicy[Retain]:
        """Return the retention and reachability policy."""
        return self._policy

    @property
    def strategy(self) -> ParseStrategy:
        """Return the chart traversal strategy."""
        return self._strategy

    @property
    def span_index(self) -> SpanIndexFactory:
        """Return the region-index factory."""
        return self._span_index

    @property
    def chart_store(self) -> ChartStore:
        """Return the chart-storage factory."""
        return self._chart_store

    @property
    def pruner(self) -> CellPruner[object, W] | None:
        """Return the cell pruner, or None for exhaustive cells."""
        return self._pruner

    @property
    def semiring(self) -> Semiring[W]:
        """Return the installed symbolic weight algebra."""
        return self._semiring

    @property
    def prune_lexical(self) -> bool:
        """Return whether pruning applies to one-token cells."""
        return self._prune_lexical

    @property
    def measure_pruned_mass(self) -> bool:
        """Return whether a pruned parse is remeasured without its pruner."""
        return self._measure_pruned_mass
