"""Refits that separate a corpus preference from a signature's trainability.

Every signature is fit again at each learning rate and budget the grid names.
Each signature's leave-one-out difference is then read at the learning rate it
scored highest under, so a signature that merely optimizes faster does not
collect the difference a corpus preference would earn. The difference at every
budget forms a curve, and the rank correlation between the orderings at the
largest budgets says whether the ordering has settled.
"""

from itertools import combinations, pairwise
from math import fsum
from typing import TYPE_CHECKING, NamedTuple, Protocol, cast, final, override

from quivers.diagnostics import loo

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi.eval._values import Comparison
from lofi.eval.elpd import ModelComparison, PairwiseDelta, ParetoGate, compare_models
from lofi.stats import paired_cohens_d, paired_permutation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from xarray import DataArray, DataTree

    from lofi._types import JsonValue
    from lofi.eval.elpd import WeightingMethod

__all__ = [
    "BestConfiguration",
    "CellRow",
    "CurvePoint",
    "RankStability",
    "Refit",
    "RefitCell",
    "RefitGrid",
    "RefitStudy",
    "StabilityRule",
    "refit_study",
]


class RefitCell(NamedTuple):
    """One signature at one learning rate and one budget."""

    signature: str
    learning_rate: float
    budget: int


class Refit(Protocol):
    """Fits one signature under one optimization configuration."""

    def __call__(self, cell: RefitCell, /) -> DataTree:
        """Return the fit of one signature at one learning rate and budget.

        Parameters
        ----------
        cell : RefitCell
            The signature to fit and the configuration to fit it under.
        """
        ...


class _PointwiseLoo(Protocol):
    """A pointwise LOO result exposing per-observation densities."""

    @property
    def elpd_i(self) -> DataArray:
        """Return one leave-one-out density per observation."""
        ...


def _increasing_rates(rates: tuple[float, ...]) -> tuple[float, ...]:
    """Return the learning rates, checked to be positive and increasing."""
    checked = tuple(ensure_finite("a grid's learning rate", rate) for rate in rates)

    if not checked:
        msg = "a refit grid names no learning rate"
        raise ValueError(msg)

    for rate in checked:
        if rate <= 0.0:
            msg = f"a grid's learning rate is not positive: {rate!r}"
            raise ValueError(msg)

    if any(later <= earlier for earlier, later in pairwise(checked)):
        msg = f"a grid's learning rates do not increase: {checked!r}"
        raise ValueError(msg)

    return checked


def _increasing_budgets(budgets: tuple[int, ...]) -> tuple[int, ...]:
    """Return the budgets, checked to be positive, increasing and at least two."""
    checked = tuple(
        ensure_non_negative("a grid's budget", budget) for budget in budgets
    )

    if len(checked) < 2:
        msg = f"a curve needs at least two budgets and the grid names {len(checked)}"
        raise ValueError(msg)

    for budget in checked:
        if budget == 0:
            msg = "a grid's budget is zero, which fits nothing"
            raise ValueError(msg)

    if any(later <= earlier for earlier, later in pairwise(checked)):
        msg = f"a grid's budgets do not increase: {checked!r}"
        raise ValueError(msg)

    return checked


def _arms(signatures: tuple[str, ...]) -> tuple[str, ...]:
    """Return the signature names, checked to be at least two and distinct."""
    checked = tuple(
        ensure_name("a grid's signature", signature) for signature in signatures
    )

    if len(checked) < 2:
        msg = f"a difference needs two signatures and the grid names {len(checked)}"
        raise ValueError(msg)

    if len(set(checked)) != len(checked):
        msg = f"a grid names one signature more than once: {checked!r}"
        raise ValueError(msg)

    return checked


@final
class RefitGrid:
    """The signatures, learning rates and budgets a study refits over.

    Parameters
    ----------
    signatures : tuple[str, ...]
        Names of the signatures compared, at least two and all distinct.
    learning_rates : tuple[float, ...]
        Learning rates each signature is fit at, positive and increasing. A grid
        naming one learning rate chooses nothing, so a study run over it reports
        its comparisons as smoke tests.
    budgets : tuple[int, ...]
        Optimization budgets each signature is fit at, positive and increasing,
        at least two so that a curve has somewhere to go.

    Raises
    ------
    TypeError
        If a name is not a string or a number is not numeric.
    ValueError
        If fewer than two signatures or budgets are named, a name is empty or
        repeated, a learning rate or budget is not positive, or either sequence
        does not increase.
    """

    __slots__ = ("_budgets", "_cells", "_learning_rates", "_signatures")

    def __init__(
        self,
        signatures: tuple[str, ...],
        learning_rates: tuple[float, ...],
        budgets: tuple[int, ...],
    ) -> None:
        self._signatures = _arms(signatures)
        self._learning_rates = _increasing_rates(learning_rates)
        self._budgets = _increasing_budgets(budgets)
        self._cells = tuple(
            RefitCell(signature, rate, budget)
            for signature in self._signatures
            for rate in self._learning_rates
            for budget in self._budgets
        )

    @property
    def signatures(self) -> tuple[str, ...]:
        """Return the names of the signatures compared."""
        return self._signatures

    @property
    def learning_rates(self) -> tuple[float, ...]:
        """Return the learning rates each signature is fit at."""
        return self._learning_rates

    @property
    def budgets(self) -> tuple[int, ...]:
        """Return the optimization budgets each signature is fit at."""
        return self._budgets

    @property
    def cells(self) -> tuple[RefitCell, ...]:
        """Return every cell of the grid, signature first and budget last."""
        return self._cells

    @property
    def refits(self) -> bool:
        """Return whether the grid gives a signature more than one configuration."""
        return len(self._learning_rates) > 1

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a grid over the same three sequences."""
        if type(other) is not RefitGrid:
            return NotImplemented

        return (
            self._signatures == other._signatures
            and self._learning_rates == other._learning_rates
            and self._budgets == other._budgets
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the signatures, learning rates and budgets."""
        return hash((RefitGrid, self._signatures, self._learning_rates, self._budgets))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this grid."""
        return (
            f"RefitGrid({self._signatures!r}, {self._learning_rates!r}, "
            f"{self._budgets!r})"
        )


@final
class StabilityRule:
    """A rule for calling an ordering settled over the largest budgets.

    Parameters
    ----------
    tail : int
        How many of the largest budgets the rank correlation is computed over,
        at least two so that there are two orderings to correlate.
    threshold : float
        Rank correlation at or above which the ordering counts as settled.

    Raises
    ------
    TypeError
        If a field is not numeric.
    ValueError
        If fewer than two budget points are asked for, or the threshold lies
        outside the range a rank correlation takes.
    """

    __slots__ = ("_tail", "_threshold")

    def __init__(self, *, tail: int, threshold: float) -> None:
        self._tail = ensure_non_negative("a stability rule's tail", tail)
        self._threshold = ensure_finite("a stability rule's threshold", threshold)

        if self._tail < 2:
            msg = (
                f"a rank correlation needs at least two budget points and the "
                f"rule asks for {tail!r}"
            )
            raise ValueError(msg)

        if not -1.0 <= self._threshold <= 1.0:
            msg = (
                f"a stability rule's threshold is not a rank correlation: {threshold!r}"
            )
            raise ValueError(msg)

    @property
    def tail(self) -> int:
        """Return how many of the largest budgets the statistic is computed over."""
        return self._tail

    @property
    def threshold(self) -> float:
        """Return the correlation at or above which the ordering counts as settled."""
        return self._threshold

    def holds(self, correlation: float) -> bool:
        """Return whether an observed rank correlation settles the ordering.

        Parameters
        ----------
        correlation : float
            The correlation observed over the tail.

        Raises
        ------
        TypeError
            If the correlation is not numeric.
        ValueError
            If the correlation is not finite.
        """
        return ensure_finite("a rank correlation", correlation) >= self._threshold

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a rule with the same two parts."""
        if type(other) is not StabilityRule:
            return NotImplemented

        return self._tail == other._tail and self._threshold == other._threshold

    @override
    def __hash__(self) -> int:
        """Return a hash over the tail and the threshold."""
        return hash((StabilityRule, self._tail, self._threshold))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this rule."""
        return f"StabilityRule(tail={self._tail!r}, threshold={self._threshold!r})"


class CellRow(NamedTuple):
    """One signature's leave-one-out summary at one learning rate and budget."""

    signature: str
    learning_rate: float
    budget: int
    elpd: float
    se: float
    observations: int
    exceeding: int
    reliable: bool

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the row as serializable report columns."""
        return {
            "signature": self.signature,
            "learning_rate": self.learning_rate,
            "budget": self.budget,
            "elpd": self.elpd,
            "se": self.se,
            "observations": self.observations,
            "exceeding": self.exceeding,
            "reliable": self.reliable,
        }


class BestConfiguration(NamedTuple):
    """The learning rate one signature scored highest under at one budget."""

    signature: str
    learning_rate: float
    elpd: float
    reliable: bool

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the choice as serializable report columns."""
        return {
            "signature": self.signature,
            "learning_rate": self.learning_rate,
            "elpd": self.elpd,
            "reliable": self.reliable,
        }


class CurvePoint(NamedTuple):
    """One budget's comparison, each signature at its own best learning rate."""

    budget: int
    best: tuple[BestConfiguration, ...]
    ranking: tuple[str, ...]
    deltas: tuple[PairwiseDelta, ...]


class RankStability(NamedTuple):
    """The agreement between the orderings at the largest budgets."""

    correlation: float
    points: int
    pairs: int
    settled: bool

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the statistic as serializable report columns."""
        return {
            "correlation": self.correlation,
            "points": self.points,
            "pairs": self.pairs,
            "settled": self.settled,
        }


def _rank_correlation(left: Sequence[str], right: Sequence[str]) -> float:
    """Return the rank correlation between two orderings of the same names.

    Both orderings list the same names once each, so no pair ties and the
    correlation is the concordant share of pairs rescaled onto minus one to one.

    Parameters
    ----------
    left : Sequence[str]
        One ordering, best first.
    right : Sequence[str]
        The other ordering of the same names.

    Raises
    ------
    ValueError
        If the two orderings do not rank the same names.
    """
    place = {name: index for index, name in enumerate(right)}

    if set(place) != set(left) or len(place) != len(left):
        msg = f"two orderings rank different names: {left!r} against {right!r}"
        raise ValueError(msg)

    pairs = tuple(combinations(left, 2))
    concordant = sum(1 for one, other in pairs if place[one] < place[other])

    return (2 * concordant - len(pairs)) / len(pairs)


def _stability(curve: tuple[CurvePoint, ...], rule: StabilityRule) -> RankStability:
    """Return the rank correlation over the last few points of a curve.

    Parameters
    ----------
    curve : tuple[CurvePoint, ...]
        The comparison at every budget, in increasing budget order.
    rule : StabilityRule
        How many of the largest budgets to read and what settles the ordering.

    Raises
    ------
    ValueError
        If the curve is shorter than the tail the rule asks for.
    """
    if len(curve) < rule.tail:
        msg = (
            f"a rule reading {rule.tail} budget points was given a curve of "
            f"{len(curve)}"
        )
        raise ValueError(msg)

    tail = tuple(point.ranking for point in curve[-rule.tail :])
    pairs = tuple(combinations(tail, 2))
    correlation = fsum(_rank_correlation(one, other) for one, other in pairs) / len(
        pairs
    )

    return RankStability(
        correlation=correlation,
        points=len(tail),
        pairs=len(pairs),
        settled=rule.holds(correlation),
    )


def _measured(cells: tuple[CellRow, ...], grid: RefitGrid) -> tuple[CellRow, ...]:
    """Return the cell rows, checked to be one per grid cell in grid order."""
    checked = tuple(cells)

    for row in checked:
        if not isinstance(row, CellRow):
            msg = f"a cell row is not one: {row!r}"
            raise TypeError(msg)

    named = tuple(
        RefitCell(row.signature, row.learning_rate, row.budget) for row in checked
    )

    if named != grid.cells:
        msg = (
            f"a study reports {len(named)} cells where its grid names "
            f"{len(grid.cells)}, or reports them out of grid order"
        )
        raise ValueError(msg)

    return checked


def _along(curve: tuple[CurvePoint, ...], grid: RefitGrid) -> tuple[CurvePoint, ...]:
    """Return the curve, checked to hold one point per budget in grid order."""
    checked = tuple(curve)

    for point in checked:
        if not isinstance(point, CurvePoint):
            msg = f"a curve point is not one: {point!r}"
            raise TypeError(msg)

    if tuple(point.budget for point in checked) != grid.budgets:
        msg = (
            f"a curve runs over {[point.budget for point in checked]!r} where "
            f"its grid names {list(grid.budgets)!r}"
        )
        raise ValueError(msg)

    for point in checked:
        chosen = tuple(best.signature for best in point.best)

        if chosen != grid.signatures:
            msg = (
                f"the point at budget {point.budget} chooses for {chosen!r} "
                f"where the grid names {grid.signatures!r}"
            )
            raise ValueError(msg)

        for best in point.best:
            if best.learning_rate not in grid.learning_rates:
                msg = (
                    f"the point at budget {point.budget} chose a learning rate "
                    f"the grid does not name: {best.learning_rate!r}"
                )
                raise ValueError(msg)

        if sorted(point.ranking) != sorted(grid.signatures):
            msg = (
                f"the point at budget {point.budget} ranks {point.ranking!r} "
                f"where the grid names {grid.signatures!r}"
            )
            raise ValueError(msg)

    return checked


def _emitted(
    comparisons: tuple[Comparison, ...], grid: RefitGrid
) -> tuple[Comparison, ...]:
    """Return the comparisons, checked against the grid's pairs and smoke status."""
    checked = tuple(comparisons)

    for comparison in checked:
        if not isinstance(comparison, Comparison):
            msg = f"a study's comparison is not one: {comparison!r}"
            raise TypeError(msg)

    pairs = {frozenset((comparison.left, comparison.right)) for comparison in checked}
    wanted = {frozenset(pair) for pair in combinations(grid.signatures, 2)}

    if pairs != wanted or len(checked) != len(wanted):
        msg = (
            f"a study emits {len(checked)} comparisons where its grid has "
            f"{len(wanted)} signature pairs"
        )
        raise ValueError(msg)

    for comparison in checked:
        if comparison.smoke_test is grid.refits:
            msg = (
                f"the comparison of {comparison.left!r} against "
                f"{comparison.right!r} reports its smoke-test status as "
                f"{comparison.smoke_test!r} under a grid naming "
                f"{len(grid.learning_rates)} learning rates"
            )
            raise ValueError(msg)

    return checked


@final
class RefitStudy:
    """A refit over a grid, its best-configuration curve and its rank stability.

    Parameters
    ----------
    grid : RefitGrid
        The configurations every signature was fit under.
    cells : tuple[CellRow, ...]
        One row per grid cell, in grid order.
    curve : tuple[CurvePoint, ...]
        One comparison per budget, each signature at its own best learning rate,
        in increasing budget order.
    comparisons : tuple[Comparison, ...]
        One comparison per pair of signatures at the largest budget.
    stability : RankStability
        The agreement between the orderings at the largest budgets.

    Raises
    ------
    TypeError
        If a member has an invalid type.
    ValueError
        If the cells are not one per grid cell in grid order, the curve does not
        run over the grid's budgets, a point chooses a learning rate or ranks a
        signature the grid does not name, the comparisons do not cover the pairs
        of signatures once, a comparison's smoke-test flag disagrees with what
        the grid offered to choose from, or the statistic reads more budget
        points than the curve has.
    """

    __slots__ = ("_cells", "_comparisons", "_curve", "_grid", "_stability")

    def __init__(
        self,
        grid: RefitGrid,
        cells: tuple[CellRow, ...],
        curve: tuple[CurvePoint, ...],
        comparisons: tuple[Comparison, ...],
        *,
        stability: RankStability,
    ) -> None:
        if not isinstance(grid, RefitGrid):
            msg = f"a study's grid is not one: {grid!r}"
            raise TypeError(msg)

        if not isinstance(stability, RankStability):
            msg = f"a study's rank stability is not one: {stability!r}"
            raise TypeError(msg)

        self._grid = grid
        self._cells = _measured(cells, grid)
        self._curve = _along(curve, grid)
        self._comparisons = _emitted(comparisons, grid)
        self._stability = stability

        if stability.points > len(self._curve):
            msg = (
                f"a study's stability reads {stability.points} budget points "
                f"from a curve of {len(self._curve)}"
            )
            raise ValueError(msg)

    @property
    def grid(self) -> RefitGrid:
        """Return the configurations every signature was fit under."""
        return self._grid

    @property
    def cells(self) -> tuple[CellRow, ...]:
        """Return one row per grid cell, in grid order."""
        return self._cells

    @property
    def curve(self) -> tuple[CurvePoint, ...]:
        """Return one best-configuration comparison per budget."""
        return self._curve

    @property
    def comparisons(self) -> tuple[Comparison, ...]:
        """Return one comparison per pair of signatures at the largest budget."""
        return self._comparisons

    @property
    def stability(self) -> RankStability:
        """Return the agreement between the orderings at the largest budgets."""
        return self._stability

    @property
    def smoke_test(self) -> bool:
        """Return whether the grid left the best configuration unchosen."""
        return not self._grid.refits

    @property
    def best(self) -> tuple[BestConfiguration, ...]:
        """Return each signature's best learning rate at the largest budget."""
        return self._curve[-1].best

    def rows(self) -> list[dict[str, JsonValue]]:
        """Return the study as long-format report rows.

        Returns
        -------
        list[dict[str, JsonValue]]
            One row per grid cell, then per budget one row for each signature's
            chosen configuration and one for each pairwise difference, then the
            rank-stability row.
        """
        return [
            *({"kind": "cell", **row.to_spec()} for row in self._cells),
            *(
                {"kind": "best", "budget": point.budget, **best.to_spec()}
                for point in self._curve
                for best in point.best
            ),
            *(
                {"kind": "delta", "budget": point.budget, **delta.to_spec()}
                for point in self._curve
                for delta in point.deltas
            ),
            {"kind": "stability", **self._stability.to_spec()},
        ]

    def _identity(
        self,
    ) -> tuple[
        RefitGrid,
        tuple[CellRow, ...],
        tuple[CurvePoint, ...],
        tuple[Comparison, ...],
        RankStability,
    ]:
        """Return the five parts a study is equal on."""
        return (
            self._grid,
            self._cells,
            self._curve,
            self._comparisons,
            self._stability,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a study with the same five parts."""
        if type(other) is not RefitStudy:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five parts the study is equal on."""
        return hash((RefitStudy, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this study."""
        return (
            f"RefitStudy({self._grid!r}, {self._cells!r}, {self._curve!r}, "
            f"{self._comparisons!r}, stability={self._stability!r})"
        )


def _pointwise(fit: DataTree, *, var_name: str | None) -> tuple[float, ...]:
    """Return one leave-one-out log predictive density per observation.

    Parameters
    ----------
    fit : DataTree
        A fitted model with a log likelihood group.
    var_name : str | None
        Observed variable to score on, when the group holds more than one.
    """
    scored = cast("_PointwiseLoo", loo(fit, pointwise=True, var_name=var_name))

    return tuple(float(density) for density in scored.elpd_i.to_numpy().ravel())


def _compared(
    chosen: tuple[RefitCell, ...],
    fits: Mapping[RefitCell, DataTree],
    cache: dict[tuple[RefitCell, ...], ModelComparison],
    *,
    gate: ParetoGate,
    method: WeightingMethod,
    var_name: str | None,
) -> ModelComparison:
    """Return the comparison of one fit per signature, computed once per cell set.

    Parameters
    ----------
    chosen : tuple[RefitCell, ...]
        One cell per signature, in the grid's signature order.
    fits : Mapping[RefitCell, DataTree]
        The fit of every cell of the grid.
    cache : dict[tuple[RefitCell, ...], ModelComparison]
        Comparisons already computed, which this call reads and extends.
    gate : ParetoGate
        Reliability rule the rows are judged under.
    method : WeightingMethod
        Estimator the weights come from.
    var_name : str | None
        Observed variable to compare on.
    """
    held = cache.get(chosen)

    if held is None:
        held = compare_models(
            {cell.signature: fits[cell] for cell in chosen},
            gate=gate,
            method=method,
            var_name=var_name,
        )
        cache[chosen] = held

    return held


def _best(
    signature: str,
    budget: int,
    measured: Mapping[RefitCell, CellRow],
    grid: RefitGrid,
) -> RefitCell:
    """Return the cell one signature scored highest in at one budget.

    Parameters
    ----------
    signature : str
        The signature choosing its configuration.
    budget : int
        The budget it is choosing at.
    measured : Mapping[RefitCell, CellRow]
        The row of every cell of the grid.
    grid : RefitGrid
        The configurations on offer. Where two learning rates score equally the
        smaller one is chosen, so the choice does not depend on grid order.
    """
    candidates = tuple(
        RefitCell(signature, rate, budget) for rate in grid.learning_rates
    )

    return min(candidates, key=lambda cell: (-measured[cell].elpd, cell.learning_rate))


def _point(
    budget: int,
    measured: Mapping[RefitCell, CellRow],
    fits: Mapping[RefitCell, DataTree],
    cache: dict[tuple[RefitCell, ...], ModelComparison],
    *,
    grid: RefitGrid,
    gate: ParetoGate,
    method: WeightingMethod,
    var_name: str | None,
) -> CurvePoint:
    """Return one budget's comparison, each signature at its own best rate.

    Parameters
    ----------
    budget : int
        The budget the point reports.
    measured : Mapping[RefitCell, CellRow]
        The row of every cell of the grid.
    fits : Mapping[RefitCell, DataTree]
        The fit of every cell of the grid.
    cache : dict[tuple[RefitCell, ...], ModelComparison]
        Comparisons already computed, which this call reads and extends.
    grid : RefitGrid
        The configurations every signature was fit under.
    gate : ParetoGate
        Reliability rule the rows are judged under.
    method : WeightingMethod
        Estimator the weights come from.
    var_name : str | None
        Observed variable to compare on.
    """
    chosen = tuple(
        _best(signature, budget, measured, grid) for signature in grid.signatures
    )
    comparison = _compared(
        chosen, fits, cache, gate=gate, method=method, var_name=var_name
    )

    return CurvePoint(
        budget=budget,
        best=tuple(
            BestConfiguration(
                signature=cell.signature,
                learning_rate=cell.learning_rate,
                elpd=measured[cell].elpd,
                reliable=measured[cell].reliable,
            )
            for cell in chosen
        ),
        ranking=comparison.ranking,
        deltas=comparison.deltas,
    )


def refit_study(
    refit: Refit,
    grid: RefitGrid,
    *,
    gate: ParetoGate,
    rule: StabilityRule,
    falsifier: str,
    family: str,
    replicates: int,
    seed: int,
    method: WeightingMethod = "stacking",
    var_name: str | None = None,
) -> RefitStudy:
    """Refit every cell of a grid and compare at each signature's best rate.

    Each cell is fit once. The leave-one-out summary of a cell comes from the
    comparison of the signatures sharing its learning rate and budget, since a
    fit's expected log predictive density does not depend on what it is compared
    against. Each signature then chooses the learning rate it scored highest
    under, separately at every budget, and the difference reported at a budget is
    the difference between those chosen fits.

    Parameters
    ----------
    refit : Refit
        What fits one signature under one configuration.
    grid : RefitGrid
        The configurations every signature is fit under.
    gate : ParetoGate
        Reliability rule over each fit's Pareto shapes.
    rule : StabilityRule
        How many of the largest budgets the rank correlation reads, and what
        settles the ordering.
    falsifier : str
        The outcome that would refute the protocol these comparisons come from.
    family : str
        The family the comparisons are corrected within.
    replicates : int
        How many sign-flip reassignments the paired null draws.
    seed : int
        The seed that null draws under.
    method : WeightingMethod
        Estimator the model weights come from.
    var_name : str | None
        Observed variable to compare on, when a fit's log likelihood group holds
        more than one.

    Returns
    -------
    RefitStudy
        Every cell's row, the best-configuration curve, its rank stability and
        one comparison per pair of signatures at the largest budget.

    Raises
    ------
    TypeError
        If the grid, the gate or the rule is not what it should be.
    ValueError
        If the rule reads more budget points than the grid has.
    """
    if not isinstance(grid, RefitGrid):
        msg = f"a refit study's grid is not one: {grid!r}"
        raise TypeError(msg)

    if not isinstance(rule, StabilityRule):
        msg = f"a refit study's stability rule is not one: {rule!r}"
        raise TypeError(msg)

    if not isinstance(gate, ParetoGate):
        msg = f"a refit study's gate is not one: {gate!r}"
        raise TypeError(msg)

    if rule.tail > len(grid.budgets):
        msg = (
            f"a rule reading {rule.tail} budget points was given a grid of "
            f"{len(grid.budgets)}"
        )
        raise ValueError(msg)

    fits = {cell: refit(cell) for cell in grid.cells}
    cache: dict[tuple[RefitCell, ...], ModelComparison] = {}
    measured: dict[RefitCell, CellRow] = {}

    for budget in grid.budgets:
        for rate in grid.learning_rates:
            shared = tuple(
                RefitCell(signature, rate, budget) for signature in grid.signatures
            )
            rows = {
                row.name: row
                for row in _compared(
                    shared, fits, cache, gate=gate, method=method, var_name=var_name
                ).models
            }

            for cell in shared:
                scored = rows[cell.signature]
                measured[cell] = CellRow(
                    signature=cell.signature,
                    learning_rate=cell.learning_rate,
                    budget=cell.budget,
                    elpd=scored.elpd,
                    se=scored.se,
                    observations=scored.observations,
                    exceeding=scored.exceeding,
                    reliable=scored.reliable,
                )

    curve = tuple(
        _point(
            budget,
            measured,
            fits,
            cache,
            grid=grid,
            gate=gate,
            method=method,
            var_name=var_name,
        )
        for budget in grid.budgets
    )
    final = curve[-1]
    densities = {
        best.signature: _pointwise(
            fits[RefitCell(best.signature, best.learning_rate, final.budget)],
            var_name=var_name,
        )
        for best in final.best
    }

    return RefitStudy(
        grid,
        tuple(measured[cell] for cell in grid.cells),
        curve,
        tuple(
            Comparison(
                delta.left,
                delta.right,
                statistic=delta.delta,
                p_value=paired_permutation(
                    densities[delta.left],
                    densities[delta.right],
                    replicates=replicates,
                    seed=seed,
                    tail="two-sided",
                ),
                effect=paired_cohens_d(densities[delta.left], densities[delta.right]),
                falsifier=falsifier,
                smoke_test=not grid.refits,
                family=family,
            )
            for delta in final.deltas
        ),
        stability=_stability(curve, rule),
    )
