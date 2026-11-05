"""Interchange-intervention accuracy over a preregistered grid of cells.

Each result records the digest of the grid, search, and budget fixed before the
sweep. Cell accuracy is tested against a null fitted to exchanged decisions.
"""

import json
from hashlib import sha256
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, override

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.nn.sites import Site
from lofi.probe._causal import CombinatorAt, NamedVariable, SplitAt, TypeAt
from lofi.probe._das import (
    Assessment,
    BoundlessSearch,
    DistributedAlignment,
    FixedRankSearch,
    SearchStrategy,
)
from lofi.probe._pricing import CostModel, GridSpec

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.nn.ports import Reading
    from lofi.probe._causal import CausalPair, HighLevelVariable, IIAResult
    from lofi.probe._ports import Subject
    from lofi.stats import LabelPermutation, PValue

__all__ = [
    "CellResult",
    "GridCell",
    "Preregistration",
    "Subgrid",
    "SweepReport",
    "UnaffordableError",
    "assess_cell",
    "sweep",
]


def _a_variable(field: str, variable: HighLevelVariable) -> HighLevelVariable:
    """Return a high-level variable, checked to be one.

    Parameters
    ----------
    field : str
        The name to report in a message.
    variable : HighLevelVariable
        The value to check.

    Raises
    ------
    TypeError
        If the value is not a high-level variable.
    """
    if type(variable) not in {TypeAt, SplitAt, CombinatorAt, NamedVariable}:
        msg = f"{field} is a high-level variable, not {variable!r}"
        raise TypeError(msg)

    return variable


def _variable_json(variable: HighLevelVariable) -> dict[str, JsonValue]:
    """Return a high-level variable as a JSON object.

    Parameters
    ----------
    variable : HighLevelVariable
        The variable to render.
    """
    if isinstance(variable, NamedVariable):
        return {
            "kind": variable.kind,
            "namespace": variable.namespace,
            "name": variable.name,
        }

    rendered: dict[str, JsonValue] = {
        "kind": variable.kind,
        "cell": list(variable.cell),
    }

    if isinstance(variable, CombinatorAt):
        rendered["split"] = variable.split

    return rendered


def _site_json(site: Site) -> list[JsonValue]:
    """Return a site as a JSON array of its component, role and key.

    Parameters
    ----------
    site : Site
        The site to render.
    """
    return [site.component, site.role.name, list(site.key)]


class GridCell:
    """One cell of the alignment grid: a variable at one layer and one position.

    Parameters
    ----------
    variable : HighLevelVariable
        The decision of a derivation the cell is about.
    layer : int
        The layer of the model the variable is read at.
    position : int
        The position within that layer the variable is read at.

    Raises
    ------
    TypeError
        If the variable is not a high-level variable, or the layer or the position is
        not an integer.
    ValueError
        If the layer or the position is negative.
    """

    __slots__ = ("_layer", "_position", "_variable")

    def __init__(self, variable: HighLevelVariable, layer: int, position: int) -> None:
        self._variable = _a_variable("a cell's variable", variable)
        self._layer = ensure_non_negative("a cell's layer", layer)
        self._position = ensure_non_negative("a cell's position", position)

    @property
    def variable(self) -> HighLevelVariable:
        """Return the decision the cell is about."""
        return self._variable

    @property
    def layer(self) -> int:
        """Return the layer the variable is read at."""
        return self._layer

    @property
    def position(self) -> int:
        """Return the position within the layer the variable is read at."""
        return self._position

    @property
    def coordinate(self) -> tuple[int, int]:
        """Return the layer and the position together."""
        return (self._layer, self._position)

    def to_json(self) -> dict[str, JsonValue]:
        """Return the cell as a JSON object."""
        return {
            "variable": _variable_json(self._variable),
            "layer": self._layer,
            "position": self._position,
        }

    def _identity(self) -> tuple[HighLevelVariable, int, int]:
        """Return the three fields a cell is equal on."""
        return (self._variable, self._layer, self._position)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value names the same cell."""
        if type(other) is not GridCell:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the variable, the layer and the position."""
        return hash((GridCell, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this cell."""
        return f"GridCell({self._variable!r}, {self._layer!r}, {self._position!r})"


def _unique[E](field: str, members: Iterable[E]) -> tuple[E, ...]:
    """Return the members in the order given, refusing a repeat.

    Parameters
    ----------
    field : str
        The name to report in a message.
    members : Iterable[E]
        The members to check.

    Raises
    ------
    ValueError
        If no member is given, or one is given twice.
    """
    held = tuple(members)

    if not held:
        msg = f"a subgrid names at least one {field}"
        raise ValueError(msg)

    if len(set(held)) != len(held):
        msg = f"a {field} is named twice: {held!r}"
        raise ValueError(msg)

    return held


class Subgrid:
    """The cells a sweep is restricted to, and the site each is read at.

    Parameters
    ----------
    variables : Iterable[HighLevelVariable]
        The decisions swept, in the order their cells are reported.
    layers : Iterable[int]
        The layers swept, in the order their cells are reported.
    positions : Iterable[int]
        The positions swept, in the order their cells are reported.
    sites : Mapping[tuple[int, int], Site]
        The published point each layer and position is read at, keyed by the layer
        and the position, covering the layers and positions swept exactly.

    Raises
    ------
    TypeError
        If a variable is not a high-level variable, a layer or a position is not an
        integer, or a published point is not a site.
    ValueError
        If an axis is empty or names a member twice, if a layer or a position is
        negative, or if a site is missing for a swept layer and position or published
        for one that is not swept.
    """

    __slots__ = ("_cells", "_layers", "_positions", "_sites", "_variables")

    def __init__(
        self,
        variables: Iterable[HighLevelVariable],
        layers: Iterable[int],
        positions: Iterable[int],
        sites: Mapping[tuple[int, int], Site],
    ) -> None:
        self._variables = tuple(
            _a_variable("a subgrid's variable", variable)
            for variable in _unique("variable", variables)
        )
        self._layers = tuple(
            ensure_non_negative("a subgrid's layer", layer)
            for layer in _unique("layer", layers)
        )
        self._positions = tuple(
            ensure_non_negative("a subgrid's position", position)
            for position in _unique("position", positions)
        )
        swept = {
            (layer, position) for layer in self._layers for position in self._positions
        }

        for coordinate, site in sites.items():
            if not isinstance(site, Site):
                msg = f"a layer and a position are read at a Site, not at {site!r}"
                raise TypeError(msg)

            if coordinate not in swept:
                msg = (
                    f"a site is published for layer and position {coordinate!r}, "
                    "which the subgrid does not sweep"
                )
                raise ValueError(msg)

        for coordinate in sorted(swept):
            if coordinate not in sites:
                msg = f"layer and position {coordinate!r} is published at no site"
                raise ValueError(msg)

        self._sites = MappingProxyType(dict(sites))
        self._cells = tuple(
            GridCell(variable, layer, position)
            for variable in self._variables
            for layer in self._layers
            for position in self._positions
        )

    @property
    def variables(self) -> tuple[HighLevelVariable, ...]:
        """Return the decisions swept."""
        return self._variables

    @property
    def layers(self) -> tuple[int, ...]:
        """Return the layers swept."""
        return self._layers

    @property
    def positions(self) -> tuple[int, ...]:
        """Return the positions swept."""
        return self._positions

    @property
    def cells(self) -> tuple[GridCell, ...]:
        """Return every cell, variables outermost and positions innermost."""
        return self._cells

    @property
    def sites(self) -> Mapping[tuple[int, int], Site]:
        """Return the published point of each layer and position."""
        return self._sites

    def __contains__(self, cell: GridCell) -> bool:
        """Return whether a cell is one of the subgrid's."""
        return cell in self._cells

    def __len__(self) -> int:
        """Return how many cells the subgrid holds."""
        return len(self._cells)

    def site_of(self, cell: GridCell) -> Site:
        """Return the published point a cell is read at.

        Parameters
        ----------
        cell : GridCell
            The cell to look up.

        Raises
        ------
        ValueError
            If the cell is not one of the subgrid's.
        """
        if cell not in self._cells:
            msg = f"{cell!r} is not a cell of the preregistered subgrid"
            raise ValueError(msg)

        return self._sites[cell.coordinate]

    def spec(self, search: SearchStrategy) -> GridSpec:
        """Return the shape of the sweep, for pricing.

        Parameters
        ----------
        search : SearchStrategy
            The strategy whose fit and null-refit units the grid pays for.
        """
        return GridSpec(
            (
                ("variable", len(self._variables)),
                ("layer", len(self._layers)),
                ("position", len(self._positions)),
            ),
            fit_unit=search.fit_unit,
            refit_unit=search.refit_unit,
            replicates=_replicates_of(search.null),
        )

    def to_json(self) -> dict[str, JsonValue]:
        """Return the subgrid as a JSON object."""
        return {
            "variables": [_variable_json(variable) for variable in self._variables],
            "layers": list(self._layers),
            "positions": list(self._positions),
            "sites": [
                [layer, position, *_site_json(self._sites[layer, position])]
                for layer in self._layers
                for position in self._positions
            ],
        }

    def _identity(
        self,
    ) -> tuple[
        tuple[HighLevelVariable, ...],
        tuple[int, ...],
        tuple[int, ...],
        frozenset[tuple[tuple[int, int], Site]],
    ]:
        """Return the four fields a subgrid is equal on."""
        return (
            self._variables,
            self._layers,
            self._positions,
            frozenset(self._sites.items()),
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value sweeps the same cells at the same sites."""
        if type(other) is not Subgrid:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the axes and the sites."""
        return hash((Subgrid, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return how many cells the subgrid holds along each axis."""
        return (
            f"Subgrid(variables={len(self._variables)}, layers={len(self._layers)}, "
            f"positions={len(self._positions)})"
        )


class UnaffordableError(Exception):
    """Reports a subgrid whose sweep does not fit the budget.

    Parameters
    ----------
    hours : float
        The accelerator hours the sweep needs.
    budget_hours : float
        The accelerator hours allowed.
    cells : int
        How many cells the subgrid holds.
    """

    def __init__(self, hours: float, budget_hours: float, cells: int) -> None:
        self.hours = hours
        self.budget_hours = budget_hours
        self.cells = cells

        super().__init__(
            f"a sweep over {cells} cells needs {hours:.3g} accelerator hours "
            f"against a budget of {budget_hours:.3g}"
        )


def _replicates_of(null: LabelPermutation[int]) -> int:
    """Return how many refits a null draws.

    A label permutation always draws a positive count of orderings; the count is
    typed as optional only by the null-model protocol, which admits nulls that
    draw none.

    Parameters
    ----------
    null : LabelPermutation[int]
        The permutation each cell is refitted under.
    """
    return cast("int", null.replicates)


class Preregistration:
    """What a sweep committed to before it ran, and the digest of that commitment.

    Parameters
    ----------
    subgrid : Subgrid
        The cells swept and the sites they are read at.
    search : SearchStrategy
        The fixed-rank or boundless strategy every cell is found and tested under.
    costs : CostModel
        The measured prices the sweep is costed against.
    budget_hours : float
        The accelerator hours the sweep may use.

    Raises
    ------
    TypeError
        If the subgrid, the search or the cost model is not one, or the budget is
        not a number.
    ValueError
        If the budget is not finite or is negative.
    UnpricedUnitError
        If the cost model has no price for a unit the sweep needs.
    UnaffordableError
        If the sweep needs more accelerator hours than the budget allows.
    """

    __slots__ = (
        "_budget_hours",
        "_device",
        "_digest",
        "_hours",
        "_search",
        "_subgrid",
    )

    def __init__(
        self,
        subgrid: Subgrid,
        search: SearchStrategy,
        *,
        costs: CostModel,
        budget_hours: float,
    ) -> None:
        if type(subgrid) is not Subgrid:
            msg = f"a sweep is preregistered over a Subgrid, not {subgrid!r}"
            raise TypeError(msg)

        if not isinstance(search, FixedRankSearch | BoundlessSearch):
            msg = f"a sweep is preregistered under a search strategy, not {search!r}"
            raise TypeError(msg)

        if type(costs) is not CostModel:
            msg = f"a sweep is costed by a CostModel, not {costs!r}"
            raise TypeError(msg)

        budget = ensure_finite("a sweep's budget in hours", budget_hours)

        if budget < 0.0:
            msg = f"a sweep's budget in hours is negative: {budget_hours!r}"
            raise ValueError(msg)

        hours = costs.hours(subgrid.spec(search))

        if hours > budget:
            raise UnaffordableError(hours, budget, len(subgrid))

        self._subgrid = subgrid
        self._search = search
        self._budget_hours = budget
        self._hours = hours
        self._device = costs.device
        encoded = json.dumps(
            self.to_json(), ensure_ascii=False, separators=(",", ":"), sort_keys=True
        )
        self._digest = sha256(encoded.encode()).hexdigest()

    @property
    def subgrid(self) -> Subgrid:
        """Return the cells swept and the sites they are read at."""
        return self._subgrid

    @property
    def search(self) -> SearchStrategy:
        """Return the strategy every cell is assessed under."""
        return self._search

    @property
    def budget_hours(self) -> float:
        """Return the accelerator hours the sweep may use."""
        return self._budget_hours

    @property
    def hours(self) -> float:
        """Return the accelerator hours the sweep needs at the measured prices."""
        return self._hours

    @property
    def device(self) -> str | None:
        """Return the device the prices were measured on, or None with no price."""
        return self._device

    @property
    def digest(self) -> str:
        """Return the hexadecimal digest of the commitment."""
        return self._digest

    def to_json(self) -> dict[str, JsonValue]:
        """Return the commitment as a JSON object, which the digest is computed from.

        The measured hours and the device are not part of it, since they describe
        the prices the sweep was costed against rather than the sweep.
        """
        return {
            "subgrid": self._subgrid.to_json(),
            "search": self._search.to_json(),
            "budget_hours": self._budget_hours,
        }

    @override
    def __repr__(self) -> str:
        """Return the subgrid, the hours against the budget, and the digest."""
        return (
            f"Preregistration({self._subgrid!r}, hours={self._hours:.3g}, "
            f"budget_hours={self._budget_hours:.3g}, digest={self._digest[:12]!r})"
        )


class CellResult:
    """One cell's accuracy, its p-value under the cell's null, and the commitment.

    Parameters
    ----------
    cell : GridCell
        The cell assessed.
    site : Site
        The published point the cell was read at.
    assessment : Assessment
        The accuracy and the p-value computed from the cell's own null.
    preregistration : str
        The digest of the commitment the cell was assessed under.

    Raises
    ------
    TypeError
        If the cell, the site or the assessment is not one.
    ValueError
        If the assessment is for a variable other than the cell's, or the digest is
        empty.
    """

    __slots__ = ("_assessment", "_cell", "_preregistration", "_site")

    def __init__(
        self, cell: GridCell, site: Site, assessment: Assessment, preregistration: str
    ) -> None:
        if type(cell) is not GridCell:
            msg = f"a result is for a GridCell, not {cell!r}"
            raise TypeError(msg)

        if not isinstance(site, Site):
            msg = f"a result is read at a Site, not {site!r}"
            raise TypeError(msg)

        if type(assessment) is not Assessment:
            msg = f"a result holds an Assessment, not {assessment!r}"
            raise TypeError(msg)

        if assessment.result.variable != cell.variable:
            msg = (
                f"an assessment for {assessment.result.variable!r} cannot report "
                f"the sweep cell for {cell.variable!r}"
            )
            raise ValueError(msg)

        if not preregistration:
            msg = "a result names the digest of the commitment it was assessed under"
            raise ValueError(msg)

        self._cell = cell
        self._site = site
        self._assessment = assessment
        self._preregistration = preregistration

    @property
    def cell(self) -> GridCell:
        """Return the cell assessed."""
        return self._cell

    @property
    def site(self) -> Site:
        """Return the published point the cell was read at."""
        return self._site

    @property
    def result(self) -> IIAResult:
        """Return the accuracy with its variable, rank and tested null."""
        return self._assessment.result

    @property
    def p_value(self) -> PValue:
        """Return how often a refit on exchanged decisions did as well."""
        return self._assessment.p_value

    @property
    def preregistration(self) -> str:
        """Return the digest of the commitment the cell was assessed under."""
        return self._preregistration

    def _identity(self) -> tuple[GridCell, Site, IIAResult, PValue, str]:
        """Return the five fields a result is equal on."""
        return (
            self._cell,
            self._site,
            self._assessment.result,
            self._assessment.p_value,
            self._preregistration,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same five fields."""
        if type(other) is not CellResult:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the five fields the result is equal on."""
        return hash((CellResult, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the cell, the accuracy and the p-value."""
        return (
            f"CellResult({self._cell!r}, accuracy={self.result.accuracy!r}, "
            f"p_value={self.p_value.value!r})"
        )


class SweepReport:
    """The cell results of one sweep, each checked against its commitment.

    Parameters
    ----------
    preregistration : Preregistration
        The commitment the sweep ran under.
    results : Iterable[CellResult]
        The results, one per assessed cell, in any order.

    Raises
    ------
    TypeError
        If the commitment or a result is not one.
    ValueError
        If a result names another commitment's digest, is for a cell outside the
        subgrid, was read at a site other than the one the subgrid publishes for its
        cell, or is the second result for one cell.
    """

    __slots__ = ("_preregistration", "_results")

    def __init__(
        self, preregistration: Preregistration, results: Iterable[CellResult]
    ) -> None:
        if type(preregistration) is not Preregistration:
            msg = f"a sweep reports under a Preregistration, not {preregistration!r}"
            raise TypeError(msg)

        subgrid = preregistration.subgrid
        held: dict[GridCell, CellResult] = {}

        for result in results:
            if type(result) is not CellResult:
                msg = f"a sweep reports CellResults, not {result!r}"
                raise TypeError(msg)

            if result.preregistration != preregistration.digest:
                msg = (
                    f"a result for {result.cell!r} was assessed under commitment "
                    f"{result.preregistration!r}, not {preregistration.digest!r}"
                )
                raise ValueError(msg)

            site = subgrid.site_of(result.cell)

            if result.site != site:
                msg = (
                    f"a result for {result.cell!r} was read at {result.site!r}, "
                    f"not at the preregistered {site!r}"
                )
                raise ValueError(msg)

            if result.cell in held:
                msg = f"{result.cell!r} is reported twice"
                raise ValueError(msg)

            held[result.cell] = result

        self._preregistration = preregistration
        self._results = tuple(held[cell] for cell in subgrid.cells if cell in held)

    @property
    def preregistration(self) -> Preregistration:
        """Return the commitment the sweep ran under."""
        return self._preregistration

    @property
    def results(self) -> tuple[CellResult, ...]:
        """Return the results, in the subgrid's cell order."""
        return self._results

    @property
    def cells(self) -> tuple[GridCell, ...]:
        """Return the cells assessed, in the subgrid's cell order."""
        return tuple(result.cell for result in self._results)

    @property
    def unassessed(self) -> tuple[GridCell, ...]:
        """Return the subgrid's cells no result was reported for."""
        assessed = frozenset(self.cells)

        return tuple(
            cell for cell in self._preregistration.subgrid.cells if cell not in assessed
        )

    @property
    def p_values(self) -> tuple[PValue, ...]:
        """Return each result's p-value, in the subgrid's cell order."""
        return tuple(result.p_value for result in self._results)

    def result_at(self, cell: GridCell) -> CellResult:
        """Return the result for one cell.

        Parameters
        ----------
        cell : GridCell
            The cell to look up.

        Raises
        ------
        KeyError
            If no result was reported for the cell.
        """
        for result in self._results:
            if result.cell == cell:
                return result

        msg = f"no result was reported for {cell!r}"
        raise KeyError(msg)

    @override
    def __repr__(self) -> str:
        """Return how many of the subgrid's cells the report holds results for."""
        return (
            f"SweepReport(assessed={len(self._results)}, "
            f"cells={len(self._preregistration.subgrid)})"
        )


def assess_cell[ReadingT: Reading, PairT: CausalPair](
    subject: Subject[ReadingT, PairT],
    preregistration: Preregistration,
    cell: GridCell,
    fitting: Sequence[PairT],
    held_out: Sequence[PairT],
) -> CellResult:
    """Return one cell's accuracy and its p-value under the cell's own null.

    Parameters
    ----------
    subject : Subject[ReadingT, PairT]
        The component whose decisions are read off its output.
    preregistration : Preregistration
        The commitment the cell is assessed under.
    cell : GridCell
        The cell to assess, one of the commitment's.
    fitting : Sequence[PairT]
        The pairs the alignment is fitted on, each varying the cell's variable.
    held_out : Sequence[PairT]
        The pairs the accuracy is measured on, held out from the fitting ones.

    Returns
    -------
    CellResult
        The accuracy on the held-out pairs, the p-value of that accuracy against
        refits to the fitting pairs with their decisions exchanged, and the digest
        of the commitment.

    Raises
    ------
    ValueError
        If the cell is not one of the commitment's, or the alignment cannot be
        fitted or assessed on the pairs given.
    """
    site = preregistration.subgrid.site_of(cell)
    alignment = DistributedAlignment(
        subject, site, cell.variable, search=preregistration.search
    )
    assessment = alignment.fit(fitting).assess(held_out)

    return CellResult(cell, site, assessment, preregistration.digest)


def sweep[ReadingT: Reading, PairT: CausalPair](
    subject: Subject[ReadingT, PairT],
    preregistration: Preregistration,
    fitting: Mapping[HighLevelVariable, Sequence[PairT]],
    held_out: Mapping[HighLevelVariable, Sequence[PairT]],
) -> SweepReport:
    """Return the accuracy and p-value of every cell of the preregistered subgrid.

    Parameters
    ----------
    subject : Subject[ReadingT, PairT]
        The component whose decisions are read off its output.
    preregistration : Preregistration
        The commitment the sweep runs under.
    fitting : Mapping[HighLevelVariable, Sequence[PairT]]
        The pairs each variable's alignments are fitted on, keyed by variable.
    held_out : Mapping[HighLevelVariable, Sequence[PairT]]
        The pairs each variable's accuracies are measured on, keyed by variable.

    Raises
    ------
    ValueError
        If a swept variable has no fitting or no held-out pairs, or a cell's
        alignment cannot be fitted or assessed on the pairs given.
    """
    subgrid = preregistration.subgrid

    for variable in subgrid.variables:
        for name, pairs in (("fitting", fitting), ("held-out", held_out)):
            if variable not in pairs:
                msg = f"the sweep has no {name} pairs for {variable!r}"
                raise ValueError(msg)

    return SweepReport(
        preregistration,
        (
            assess_cell(
                subject,
                preregistration,
                cell,
                fitting[cell.variable],
                held_out[cell.variable],
            )
            for cell in subgrid.cells
        ),
    )
