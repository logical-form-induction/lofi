from math import log, sqrt
from statistics import fmean
from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._causal import (
    AlignmentPolicy,
    CombinatorAt,
    CounterfactualPair,
    IIAResult,
    NamedVariable,
    Side,
    SplitAt,
    TypeAt,
)
from lofi.probe._das import (
    AlignmentObjective,
    Annealing,
    Assessment,
    BoundlessSearch,
    FixedRankSearch,
    Schedule,
    SearchStrategy,
)
from lofi.probe._pricing import CostModel, ProbeUnit, UnitPrice, UnpricedUnitError
from lofi.probe._provenance import Provenance
from lofi.probe._sweep import (
    CellResult,
    GridCell,
    Preregistration,
    Subgrid,
    SweepReport,
    UnaffordableError,
    assess_cell,
    sweep,
)
from lofi.stats import (
    Dependence,
    Holm,
    LabelPermutation,
    PValue,
    p_value_against_replicates,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._causal import CellAssignment, HighLevelVariable

    type Pairs = dict[HighLevelVariable, tuple[CounterfactualPair, ...]]

MODEL: Final = ComponentId("toy-transformer")
CATEGORY: Final = TypeAt((0, 2))
SIBLING: Final = TypeAt((2, 4))
SPLIT: Final = SplitAt((0, 4))
COMBINATOR: Final = CombinatorAt((0, 4), 2)
CATEGORIES: Final = ("N", "NP")
DECISIONS: Final[dict[HighLevelVariable, tuple[str, ...]]] = {
    CATEGORY: CATEGORIES,
    SIBLING: ("V", "VP"),
}
LAYERS: Final = (0, 1)
POSITIONS: Final = (1, 3)
WIDTH: Final = 4
GAIN: Final = 6.0
REPLICATES: Final = 19
GENERATED_BY: Final = Provenance("planted", "toy", 0)
POLICY: Final = AlignmentPolicy("first", "exclude", contiguous=True)

# the category is planted in the first coordinate of a hidden basis; the readout
# reads it at full weight and the other three coordinates at half weight
READOUT: Final = torch.tensor([[1.0, 0.5, -0.5, 0.5], [-1.0, -0.5, 0.5, -0.5]])

SITES: Final[dict[tuple[int, int], Site]] = {
    (layer, position): Site(MODEL, Role.RESIDUAL_STREAM, (layer, position))
    for layer in LAYERS
    for position in POSITIONS
}
SIGNAL: Final = (0, 3)
PUBLISHED: Final[SiteMap] = SiteMap.custom(
    MODEL,
    {
        site: SiteSpec(
            ("item", "feature"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        )
        for site in SITES.values()
    },
)


class _Held:
    __slots__ = ("_axes", "_mask", "_site", "_tensor")

    def __init__(self, site: Site, tensor: Tensor, axes: tuple[str, ...]) -> None:
        self._site = site
        self._tensor = tensor
        self._axes = axes
        self._mask = None

    @property
    def site(self) -> Site:
        return self._site

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return self._axes

    @property
    def mask(self) -> Tensor | None:
        return self._mask


class _Scope:
    def __init__(
        self, model: _Toy, sites: frozenset[Site], patches: Mapping[Site, Patch[_Held]]
    ) -> None:
        self._model = model
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Held] = {}

    @property
    def readings(self) -> Mapping[Site, _Held]:
        return self._readings

    def observe(self, site: Site, tensor: Tensor, axes: tuple[str, ...]) -> None:
        if site in self._sites:
            self._readings[site] = _Held(site, tensor, axes)

    def intervene(self, site: Site, tensor: Tensor, axes: tuple[str, ...]) -> Tensor:
        standing = self._patches.get(site)

        if standing is None:
            return tensor

        return standing.stands_in_for(_Held(site, tensor, axes))

    def __enter__(self) -> Self:
        self._model.scopes.append(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        self._model.scopes.remove(self)

        return None


def a_basis(seed: int) -> Tensor:
    orthogonal, _ = torch.linalg.qr(
        torch.randn(WIDTH, WIDTH, generator=torch.Generator().manual_seed(seed))
    )

    return orthogonal


class _Toy:
    """Plants each variable's decision at its own site and noise at the others.

    Each site has its own hidden basis. The fixture records which rows each variable
    uses during fitting and scoring.
    """

    def __init__(
        self, planted: Mapping[HighLevelVariable, tuple[int, int]], seed: int = 0
    ) -> None:
        self._planted = dict(planted)
        self.fitted: dict[HighLevelVariable, set[int]] = {}
        self.scored: dict[HighLevelVariable, set[int]] = {}
        self._bases = {
            coordinate: a_basis(seed * 100 + index)
            for index, coordinate in enumerate(SITES)
        }
        self._noise = torch.randn(
            len(SITES), 2, 512, WIDTH, generator=torch.Generator().manual_seed(seed)
        )
        self.readout = torch.nn.Parameter(READOUT.clone())
        self.scopes: list[_Scope] = []

    @property
    def policy(self) -> AlignmentPolicy:
        return POLICY

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def decisions(self, variable: HighLevelVariable) -> tuple[str | int, ...]:
        return DECISIONS.get(variable, ())

    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        varied = pairs[0].varied
        first = self.decisions(varied)[0]
        decided = [pair.base if side is Side.BASE else pair.variant for pair in pairs]
        signs = torch.tensor([1.0 if at[varied] == first else -1.0 for at in decided])
        rows = torch.tensor([int(at[SPLIT]) for at in decided])
        scores = torch.zeros(len(pairs), len(CATEGORIES))

        if side is Side.BASE:
            recorded = self.fitted if torch.is_grad_enabled() else self.scored
            recorded.setdefault(varied, set()).update(rows.tolist())

        for index, (coordinate, site) in enumerate(SITES.items()):
            planted = self._noise[index, 0 if side is Side.BASE else 1, rows].clone()

            if self._planted.get(varied) == coordinate:
                planted[:, 0] = signs * GAIN

            hidden = planted @ self._bases[coordinate]

            for scope in self.scopes:
                hidden = scope.intervene(site, hidden, ("item", "feature"))
                scope.observe(site, hidden, ("item", "feature"))

            scores = scores + (hidden @ self._bases[coordinate].T) @ self.readout.T

        return scores


def a_pair(
    index: int, base: str, variant: str, variable: HighLevelVariable = CATEGORY
) -> CounterfactualPair:
    before: CellAssignment = {variable: base, SPLIT: index}
    after: CellAssignment = {variable: variant, SPLIT: index}

    return CounterfactualPair(before, after, variable, (0, 4), GENERATED_BY)


def pairs_from(
    count: int, *, seed: int, first: int = 0, variable: HighLevelVariable = CATEGORY
) -> tuple[CounterfactualPair, ...]:
    draws = torch.Generator().manual_seed(seed)
    flips = torch.randint(0, 2, (count,), generator=draws).tolist()
    decisions = DECISIONS[variable]

    return tuple(
        a_pair(first + index, decisions[flip], decisions[1 - flip], variable)
        for index, flip in enumerate(flips)
    )


def rows_of(pairs: Sequence[CounterfactualPair]) -> set[int]:
    return {int(pair.base[SPLIT]) for pair in pairs}


FITS_ON: Final = 32
TESTS_ON: Final = 120
FITTING: Final = pairs_from(FITS_ON, seed=1)
HELD_OUT: Final = pairs_from(TESTS_ON, seed=2, first=FITS_ON)


def a_schedule(seed: int = 0) -> Schedule:
    return Schedule(epochs=1, learning_rate=0.5, batch_size=32, seed=seed)


def a_search(seed: int = 0, replicates: int = REPLICATES) -> FixedRankSearch:
    return FixedRankSearch(
        rank=1,
        schedule=a_schedule(seed),
        null=LabelPermutation(replicates=replicates, seed=seed),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )


def a_price(unit: ProbeUnit, seconds: float, device: str = "cpu") -> UnitPrice:
    return UnitPrice(
        unit,
        seconds_per_run=seconds,
        seconds_fastest=seconds / 2,
        warmup=1,
        repeats=3,
        device=device,
    )


FIT_SECONDS: Final = 36.0
REFIT_SECONDS: Final = 18.0
COSTS: Final = CostModel(
    [
        a_price(ProbeUnit.FIXED_ALIGNMENT_FIT, FIT_SECONDS),
        a_price(ProbeUnit.FIXED_PERMUTATION_REFIT, REFIT_SECONDS),
    ]
)
BOUNDLESS_COSTS: Final = CostModel(
    [
        a_price(ProbeUnit.BOUNDLESS_ALIGNMENT_FIT, FIT_SECONDS),
        a_price(ProbeUnit.BOUNDLESS_PERMUTATION_REFIT, REFIT_SECONDS),
    ]
)


def a_subgrid(
    variables: tuple[HighLevelVariable, ...] = (CATEGORY,),
    layers: tuple[int, ...] = LAYERS,
    positions: tuple[int, ...] = POSITIONS,
) -> Subgrid:
    sites = {
        (layer, position): SITES[layer, position]
        for layer in layers
        for position in positions
    }

    return Subgrid(variables, layers, positions, sites)


def a_preregistration(
    subgrid: Subgrid | None = None,
    search: SearchStrategy | None = None,
    *,
    costs: CostModel = COSTS,
    budget_hours: float = 10.0,
) -> Preregistration:
    return Preregistration(
        a_subgrid() if subgrid is None else subgrid,
        a_search() if search is None else search,
        costs=costs,
        budget_hours=budget_hours,
    )


SIGNAL_CELL: Final = GridCell(CATEGORY, *SIGNAL)
NULL_CELL: Final = GridCell(CATEGORY, 1, 1)


PLANTED_SEARCH: Final = FixedRankSearch(
    rank=1,
    schedule=Schedule(epochs=20, learning_rate=0.2, batch_size=16, seed=0),
    null=LabelPermutation(replicates=REPLICATES, seed=0),
    objective=AlignmentObjective.CROSS_ENTROPY,
)


def a_boundless_search(epochs: int = 3) -> BoundlessSearch:
    return BoundlessSearch(
        boundaries=(0.0, 1.0),
        margin=1e-6,
        annealing=Annealing(initial=1.0, final=0.05),
        schedule=Schedule(epochs=epochs, learning_rate=0.2, batch_size=16, seed=0),
        null=LabelPermutation(replicates=1, seed=0),
        objective=AlignmentObjective.CROSS_ENTROPY,
    )


@pytest.fixture(scope="module")
def planted_report() -> SweepReport:
    return sweep(
        _Toy({CATEGORY: SIGNAL}),
        a_preregistration(search=PLANTED_SEARCH),
        {CATEGORY: FITTING},
        {CATEGORY: HELD_OUT},
    )


def test_a_cell_is_a_variable_at_a_layer_and_a_position() -> None:
    assert SIGNAL_CELL.variable == CATEGORY
    assert SIGNAL_CELL.layer == 0
    assert SIGNAL_CELL.position == 3
    assert SIGNAL_CELL.coordinate == (0, 3)
    assert SIGNAL_CELL.to_json() == {
        "variable": {"kind": "type", "cell": [0, 2]},
        "layer": 0,
        "position": 3,
    }
    assert GridCell(COMBINATOR, 0, 0).to_json()["variable"] == {
        "kind": "combinator",
        "cell": [0, 4],
        "split": 2,
    }
    named = NamedVariable("external", "syntactic-category")
    assert GridCell(named, 0, 0).to_json()["variable"] == {
        "kind": "named",
        "namespace": "external",
        "name": "syntactic-category",
    }


def test_cells_naming_the_same_variable_layer_and_position_are_one_cell() -> None:
    rebuilt = GridCell(TypeAt((0, 2)), 0, 3)

    assert rebuilt == SIGNAL_CELL
    assert hash(rebuilt) == hash(SIGNAL_CELL)
    assert GridCell(SPLIT, 0, 3) != SIGNAL_CELL
    assert GridCell(CATEGORY, 1, 3) != SIGNAL_CELL
    assert GridCell(CATEGORY, 0, 1) != SIGNAL_CELL
    assert SIGNAL_CELL != (CATEGORY, 0, 3)
    assert repr(SIGNAL_CELL) == "GridCell(TypeAt((0, 2)), 0, 3)"


def test_a_cell_is_about_a_high_level_variable_at_a_non_negative_place() -> None:
    with pytest.raises(TypeError, match="high-level variable"):
        GridCell(cast("HighLevelVariable", "category"), 0, 0)

    with pytest.raises(ValueError, match="layer is negative"):
        GridCell(CATEGORY, -1, 0)

    with pytest.raises(ValueError, match="position is negative"):
        GridCell(CATEGORY, 0, -1)


def test_a_subgrid_crosses_its_axes_with_positions_innermost() -> None:
    subgrid = a_subgrid((CATEGORY, SPLIT))

    assert subgrid.variables == (CATEGORY, SPLIT)
    assert subgrid.layers == LAYERS
    assert subgrid.positions == POSITIONS
    assert len(subgrid) == 8
    assert subgrid.cells == tuple(
        GridCell(variable, layer, position)
        for variable in (CATEGORY, SPLIT)
        for layer in LAYERS
        for position in POSITIONS
    )
    assert SIGNAL_CELL in subgrid
    assert GridCell(COMBINATOR, 0, 3) not in subgrid
    assert GridCell(CATEGORY, 2, 3) not in subgrid
    assert repr(subgrid) == "Subgrid(variables=2, layers=2, positions=2)"


def test_a_subgrid_reads_each_cell_at_the_site_published_for_its_place() -> None:
    subgrid = a_subgrid()

    assert subgrid.site_of(SIGNAL_CELL) == SITES[SIGNAL]
    assert subgrid.site_of(NULL_CELL) == SITES[1, 1]
    assert subgrid.sites == SITES

    with pytest.raises(ValueError, match="not a cell of the preregistered subgrid"):
        subgrid.site_of(GridCell(SPLIT, 0, 3))


def test_a_subgrid_is_priced_as_a_grid_of_its_three_axes() -> None:
    spec = a_subgrid((CATEGORY, SPLIT)).spec(a_search())

    assert spec.axes == (("variable", 2), ("layer", 2), ("position", 2))
    assert spec.cells == 8
    assert spec.replicates == REPLICATES
    assert spec.fit_unit is ProbeUnit.FIXED_ALIGNMENT_FIT
    assert spec.refit_unit is ProbeUnit.FIXED_PERMUTATION_REFIT
    assert spec.attribution_runs == 0
    assert spec.pruning_runs == 0


def test_subgrids_over_the_same_cells_and_sites_are_one_subgrid() -> None:
    assert a_subgrid() == a_subgrid()
    assert hash(a_subgrid()) == hash(a_subgrid())
    assert a_subgrid() != a_subgrid((CATEGORY, SPLIT))
    assert a_subgrid() != a_subgrid(layers=(1, 0))
    assert a_subgrid() != a_subgrid(positions=(1,))
    assert a_subgrid() != (CATEGORY,)


def test_a_subgrid_names_every_axis_member_once_and_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least one variable"):
        a_subgrid(())

    with pytest.raises(ValueError, match="at least one layer"):
        a_subgrid(layers=())

    with pytest.raises(ValueError, match="at least one position"):
        a_subgrid(positions=())

    with pytest.raises(ValueError, match="variable is named twice"):
        a_subgrid((CATEGORY, CATEGORY))

    with pytest.raises(ValueError, match="layer is named twice"):
        Subgrid((CATEGORY,), (0, 0), (1,), {(0, 1): SITES[0, 1]})

    with pytest.raises(ValueError, match="position is named twice"):
        Subgrid((CATEGORY,), (0,), (1, 1), {(0, 1): SITES[0, 1]})


def test_a_subgrid_checks_its_variables_layers_and_positions() -> None:
    with pytest.raises(TypeError, match="high-level variable"):
        a_subgrid((cast("HighLevelVariable", "category"),))

    with pytest.raises(ValueError, match="layer is negative"):
        Subgrid((CATEGORY,), (-1,), (1,), {(-1, 1): SITES[0, 1]})

    with pytest.raises(ValueError, match="position is negative"):
        Subgrid((CATEGORY,), (0,), (-1,), {(0, -1): SITES[0, 1]})


def test_a_subgrid_is_published_at_exactly_the_sites_of_its_places() -> None:
    with pytest.raises(ValueError, match=r"\(0, 1\) is published at no site"):
        Subgrid((CATEGORY,), LAYERS, POSITIONS, {SIGNAL: SITES[SIGNAL]})

    with pytest.raises(ValueError, match="which the subgrid does not sweep"):
        Subgrid((CATEGORY,), (0,), (3,), SITES)

    with pytest.raises(TypeError, match="read at a Site"):
        Subgrid(
            (CATEGORY,),
            (0,),
            (3,),
            cast("dict[tuple[int, int], Site]", {SIGNAL: "layers.0.residual"}),
        )


def test_a_preregistration_records_what_the_sweep_committed_to() -> None:
    preregistration = a_preregistration()
    recorded = preregistration.to_json()

    assert preregistration.subgrid == a_subgrid()
    assert preregistration.search is not None
    assert preregistration.budget_hours == 10.0
    assert preregistration.device == "cpu"
    assert recorded["subgrid"] == a_subgrid().to_json()
    assert recorded["search"] == {
        "kind": "fixed-rank",
        "rank": 1,
        "schedule": {
            "epochs": 1,
            "learning_rate": 0.5,
            "batch_size": 32,
            "seed": 0,
        },
        "null": {"name": "label permutation", "replicates": REPLICATES, "seed": 0},
        "objective": "cross-entropy",
    }
    assert recorded["budget_hours"] == 10.0
    assert a_subgrid().to_json()["sites"] == [
        [0, 1, "toy-transformer", "RESIDUAL_STREAM", [0, 1]],
        [0, 3, "toy-transformer", "RESIDUAL_STREAM", [0, 3]],
        [1, 1, "toy-transformer", "RESIDUAL_STREAM", [1, 1]],
        [1, 3, "toy-transformer", "RESIDUAL_STREAM", [1, 3]],
    ]


def test_a_boundless_strategy_is_preregistered_and_priced_under_its_own_units() -> None:
    search = a_boundless_search()
    preregistration = a_preregistration(
        a_subgrid(layers=(0,), positions=(3,)),
        search,
        costs=BOUNDLESS_COSTS,
    )
    recorded = preregistration.to_json()["search"]
    assert isinstance(recorded, dict)

    assert recorded["kind"] == "boundless"
    assert recorded["boundaries"] == [0.0, 1.0]
    assert recorded["annealing"] == {"initial": 1.0, "final": 0.05}
    assert preregistration.subgrid.spec(search).demand() == {
        ProbeUnit.BOUNDLESS_ALIGNMENT_FIT: 1,
        ProbeUnit.BOUNDLESS_PERMUTATION_REFIT: 1,
    }


def test_the_same_sweep_path_accepts_a_boundless_strategy() -> None:
    preregistration = a_preregistration(
        a_subgrid(layers=(0,), positions=(3,)),
        a_boundless_search(epochs=8),
        costs=BOUNDLESS_COSTS,
    )
    report = sweep(
        _Toy({CATEGORY: SIGNAL}),
        preregistration,
        {CATEGORY: FITTING},
        {CATEGORY: HELD_OUT},
    )

    assert len(report.results) == 1
    assert report.results[0].result.rank >= 0


def test_a_preregistration_prices_the_sweep_at_the_measured_prices() -> None:
    preregistration = a_preregistration()
    cells = len(a_subgrid())

    assert preregistration.hours == pytest.approx(
        cells * (FIT_SECONDS + REPLICATES * REFIT_SECONDS) / 3600
    )


def test_the_digest_is_a_function_of_the_commitment_alone() -> None:
    digest = a_preregistration().digest

    assert len(digest) == 64
    assert int(digest, 16) >= 0
    assert a_preregistration().digest == digest
    assert a_preregistration(a_subgrid((CATEGORY, SPLIT))).digest != digest
    assert a_preregistration(a_subgrid(positions=(3, 1))).digest != digest
    assert a_preregistration(search=a_search(seed=1)).digest != digest
    wider = FixedRankSearch(
        rank=2,
        schedule=a_schedule(),
        null=a_search().null,
        objective=AlignmentObjective.CROSS_ENTROPY,
    )
    assert a_preregistration(search=wider).digest != digest
    assert a_preregistration(search=a_search(replicates=9)).digest != digest
    assert a_preregistration(budget_hours=11.0).digest != digest
    assert digest[:12] in repr(a_preregistration())


def test_a_sweep_the_budget_does_not_cover_is_refused_with_the_arithmetic() -> None:
    hours = a_preregistration().hours

    with pytest.raises(
        UnaffordableError, match=r"needs .* against a budget of"
    ) as caught:
        a_preregistration(budget_hours=hours / 2)

    assert caught.value.hours == pytest.approx(hours)
    assert caught.value.budget_hours == pytest.approx(hours / 2)
    assert caught.value.cells == len(a_subgrid())
    assert a_preregistration(budget_hours=hours).hours == pytest.approx(hours)


def test_a_sweep_needing_an_unpriced_unit_is_refused() -> None:
    fits_only = CostModel([a_price(ProbeUnit.FIXED_ALIGNMENT_FIT, FIT_SECONDS)])

    with pytest.raises(UnpricedUnitError, match="fixed_permutation_refit"):
        Preregistration(a_subgrid(), a_search(), costs=fits_only, budget_hours=10.0)


def test_a_preregistration_checks_its_subgrid_search_costs_and_budget() -> None:
    with pytest.raises(TypeError, match="Subgrid"):
        Preregistration(
            cast("Subgrid", a_subgrid().cells),
            a_search(),
            costs=COSTS,
            budget_hours=1.0,
        )

    with pytest.raises(TypeError, match="search strategy"):
        Preregistration(
            a_subgrid(),
            cast("SearchStrategy", a_schedule()),
            costs=COSTS,
            budget_hours=1.0,
        )

    with pytest.raises(TypeError, match="CostModel"):
        Preregistration(
            a_subgrid(), a_search(), costs=cast("CostModel", {}), budget_hours=1.0
        )

    with pytest.raises(ValueError, match="budget in hours is negative"):
        a_preregistration(budget_hours=-1.0)

    with pytest.raises(ValueError, match="not finite"):
        a_preregistration(budget_hours=float("inf"))


def test_a_planted_cell_is_recovered_with_a_small_p_value() -> None:
    preregistration = a_preregistration(search=PLANTED_SEARCH)
    found = assess_cell(
        _Toy({CATEGORY: SIGNAL}), preregistration, SIGNAL_CELL, FITTING, HELD_OUT
    )

    assert found.cell == SIGNAL_CELL
    assert found.site == SITES[SIGNAL]
    assert found.preregistration == preregistration.digest
    assert found.result.accuracy > 0.95
    assert found.result.variable == CATEGORY
    assert found.result.rank == 1
    assert found.p_value.value == pytest.approx(1 / (REPLICATES + 1))
    assert found.p_value.provenance == found.result.null
    assert found.result.null.null == "label permutation"
    assert found.result.null.replicates == REPLICATES


def test_a_cell_outside_the_preregistered_subgrid_is_not_assessed() -> None:
    with pytest.raises(ValueError, match="not a cell of the preregistered subgrid"):
        assess_cell(
            _Toy({CATEGORY: SIGNAL}),
            a_preregistration(),
            GridCell(CATEGORY, 2, 3),
            FITTING,
            HELD_OUT,
        )


def test_the_accuracy_record_holds_no_p_value_and_cannot_enter_a_correction(
    planted_report: SweepReport,
) -> None:
    accuracies = tuple(result.result for result in planted_report.results)

    assert all(isinstance(accuracy, IIAResult) for accuracy in accuracies)

    with pytest.raises(TypeError):
        Holm().apply(cast("Sequence[PValue]", accuracies), 0.05, Dependence.ARBITRARY)

    decisions = Holm().apply(planted_report.p_values, 0.05, Dependence.ARBITRARY)

    assert len(decisions.reject) == len(planted_report.results)


def test_a_sweep_assesses_every_cell_in_the_subgrids_order(
    planted_report: SweepReport,
) -> None:
    preregistration = planted_report.preregistration

    assert planted_report.cells == preregistration.subgrid.cells
    assert planted_report.unassessed == ()
    assert len(planted_report.p_values) == 4
    assert repr(planted_report) == "SweepReport(assessed=4, cells=4)"
    assert isinstance(preregistration.search, FixedRankSearch)

    for result in planted_report.results:
        assert result.preregistration == preregistration.digest
        assert result.site == preregistration.subgrid.site_of(result.cell)
        assert result.result.rank == preregistration.search.rank


def test_the_planted_cell_alone_beats_its_null(planted_report: SweepReport) -> None:
    planted = planted_report.result_at(SIGNAL_CELL)

    assert planted.result.accuracy > 0.95
    assert planted.p_value.value == pytest.approx(1 / (REPLICATES + 1))

    for result in planted_report.results:
        if result.cell != SIGNAL_CELL:
            assert result.result.accuracy < 0.05
            assert result.p_value.value > 1 / (REPLICATES + 1)


def test_a_sweep_assesses_each_cell_on_its_own_variables_held_out_pairs() -> None:
    subject = _Toy({CATEGORY: SIGNAL, SIBLING: (1, 1)})
    both = a_preregistration(a_subgrid((CATEGORY, SIBLING)))
    fitting: Pairs = {
        CATEGORY: FITTING,
        SIBLING: pairs_from(FITS_ON, seed=3, variable=SIBLING),
    }
    held_out: Pairs = {
        CATEGORY: HELD_OUT,
        SIBLING: pairs_from(TESTS_ON, seed=4, first=FITS_ON, variable=SIBLING),
    }
    report = sweep(subject, both, fitting, held_out)

    assert report.cells == both.subgrid.cells
    assert rows_of(FITTING).isdisjoint(rows_of(HELD_OUT))

    for variable in (CATEGORY, SIBLING):
        assert subject.fitted[variable] == rows_of(fitting[variable])
        assert subject.scored[variable] == rows_of(held_out[variable])


def test_a_sweep_over_a_variable_with_no_pairs_is_refused() -> None:
    subject = _Toy({CATEGORY: SIGNAL})
    both = a_preregistration(a_subgrid((CATEGORY, SPLIT)))

    with pytest.raises(ValueError, match="no fitting pairs for SplitAt"):
        sweep(subject, both, {CATEGORY: FITTING}, {CATEGORY: HELD_OUT, SPLIT: HELD_OUT})

    with pytest.raises(ValueError, match="no held-out pairs for SplitAt"):
        sweep(subject, both, {CATEGORY: FITTING, SPLIT: FITTING}, {CATEGORY: HELD_OUT})


def test_a_result_assessed_under_another_commitment_is_refused(
    planted_report: SweepReport,
) -> None:
    other = a_preregistration(search=PLANTED_SEARCH, budget_hours=11.0)

    with pytest.raises(ValueError, match="was assessed under commitment"):
        SweepReport(other, planted_report.results)


def test_a_result_for_a_cell_outside_the_subgrid_is_refused(
    planted_report: SweepReport,
) -> None:
    preregistration = planted_report.preregistration
    found = planted_report.result_at(SIGNAL_CELL)
    elsewhere = CellResult(
        GridCell(SPLIT, *SIGNAL),
        found.site,
        Assessment(
            IIAResult(
                found.result.accuracy,
                SPLIT,
                found.result.rank,
                found.result.null,
            ),
            found.p_value,
        ),
        preregistration.digest,
    )

    with pytest.raises(ValueError, match="not a cell of the preregistered subgrid"):
        SweepReport(preregistration, [elsewhere])


def test_a_result_read_at_another_site_than_the_preregistered_one_is_refused(
    planted_report: SweepReport,
) -> None:
    preregistration = planted_report.preregistration
    found = planted_report.result_at(SIGNAL_CELL)
    moved = CellResult(
        found.cell,
        SITES[1, 1],
        Assessment(found.result, found.p_value),
        preregistration.digest,
    )

    with pytest.raises(ValueError, match="not at the preregistered"):
        SweepReport(preregistration, [moved])


def test_a_cell_reported_twice_is_refused(planted_report: SweepReport) -> None:
    found = planted_report.result_at(SIGNAL_CELL)

    with pytest.raises(ValueError, match="is reported twice"):
        SweepReport(planted_report.preregistration, [found, found])


def test_a_report_orders_results_as_the_subgrid_orders_its_cells(
    planted_report: SweepReport,
) -> None:
    reversed_report = SweepReport(
        planted_report.preregistration, reversed(planted_report.results)
    )

    assert reversed_report.results == planted_report.results
    assert reversed_report.cells == planted_report.preregistration.subgrid.cells


def test_a_named_variable_serializes_and_reports_independently_of_result_order() -> (
    None
):
    variable = NamedVariable("external", "syntactic-category")
    subgrid = a_subgrid((variable,), positions=(1,))
    preregistration = a_preregistration(subgrid)
    tested = p_value_against_replicates(
        0.75,
        (0.25,),
        tail="greater",
        null="label permutation",
        family="permutation",
        exact=False,
        seed=0,
    )
    results = tuple(
        CellResult(
            cell,
            subgrid.site_of(cell),
            Assessment(
                IIAResult(0.75, variable, 1, tested.provenance),
                tested,
            ),
            preregistration.digest,
        )
        for cell in subgrid.cells
    )

    report = SweepReport(preregistration, reversed(results))

    assert report.results == results
    assert all(result.result.variable is variable for result in report.results)
    assert subgrid.to_json()["variables"] == [
        {
            "kind": "named",
            "namespace": "external",
            "name": "syntactic-category",
        }
    ]


def test_a_partial_report_names_the_cells_it_lacks(
    planted_report: SweepReport,
) -> None:
    preregistration = planted_report.preregistration
    found = planted_report.result_at(SIGNAL_CELL)
    partial = SweepReport(preregistration, [found])

    assert partial.results == (found,)
    assert partial.unassessed == tuple(
        cell for cell in preregistration.subgrid.cells if cell != SIGNAL_CELL
    )

    with pytest.raises(KeyError, match="no result was reported"):
        partial.result_at(NULL_CELL)


def test_a_report_checks_its_commitment_and_its_results(
    planted_report: SweepReport,
) -> None:
    with pytest.raises(TypeError, match="Preregistration"):
        SweepReport(cast("Preregistration", a_subgrid()), planted_report.results)

    with pytest.raises(TypeError, match="CellResults"):
        SweepReport(
            planted_report.preregistration,
            cast("list[CellResult]", [planted_report.results[0].result]),
        )


def test_a_result_checks_its_cell_site_assessment_and_digest(
    planted_report: SweepReport,
) -> None:
    found = planted_report.result_at(SIGNAL_CELL)
    assessment = Assessment(found.result, found.p_value)

    with pytest.raises(TypeError, match="GridCell"):
        CellResult(cast("GridCell", SIGNAL), found.site, assessment, "abc")

    with pytest.raises(TypeError, match="Site"):
        CellResult(SIGNAL_CELL, cast("Site", "residual"), assessment, "abc")

    with pytest.raises(TypeError, match="Assessment"):
        CellResult(SIGNAL_CELL, found.site, cast("Assessment", found.result), "abc")

    mismatched = Assessment(
        IIAResult(
            found.result.accuracy,
            SPLIT,
            found.result.rank,
            found.p_value.provenance,
        ),
        found.p_value,
    )
    with pytest.raises(ValueError, match="cannot report the sweep cell"):
        CellResult(SIGNAL_CELL, found.site, mismatched, "abc")

    with pytest.raises(ValueError, match="names the digest"):
        CellResult(SIGNAL_CELL, found.site, assessment, "")


def test_results_recording_the_same_five_fields_are_one_result(
    planted_report: SweepReport,
) -> None:
    found = planted_report.result_at(SIGNAL_CELL)
    rebuilt = CellResult(
        found.cell,
        found.site,
        Assessment(found.result, found.p_value),
        found.preregistration,
    )

    assert rebuilt == found
    assert hash(rebuilt) == hash(found)
    assert found != planted_report.result_at(NULL_CELL)
    assert found != CellResult(
        found.cell, found.site, Assessment(found.result, found.p_value), "other"
    )
    assert found != (found.cell, found.site)
    assert repr(found).startswith(
        "CellResult(GridCell(TypeAt((0, 2)), 0, 3), accuracy="
    )


def rejection_excess(pvalues: Sequence[float], replicates: int) -> float:
    """Return the largest excess rejection rate over its nominal level.

    Permutation p-values lie on ``k / (replicates + 1)``. Tied accuracies
    make them conservative, so calibration bounds excess rejections rather
    than requiring equality with the uniform CDF.

    Parameters
    ----------
    pvalues : Sequence[float]
    replicates : int
    """
    lattice = replicates + 1

    return max(
        sum(1 for p in pvalues if p <= k / lattice) / len(pvalues) - k / lattice
        for k in range(1, lattice + 1)
    )


def calibration_bound(count: int, alpha: float) -> float:
    """Return the one-sided empirical-CDF deviation bound at the given level.

    Parameters
    ----------
    count : int
    alpha : float
    """
    return sqrt(-log(alpha) / (2 * count))


def outcomes_at(accuracy: float) -> list[float]:
    matched = round(accuracy * TESTS_ON)

    return [1.0] * matched + [0.0] * (TESTS_ON - matched)


def test_on_a_null_signal_cell_the_p_values_control_false_positives() -> None:
    refits = 200
    pvalues: list[float] = []
    accuracies: list[float] = []

    for refit in range(refits):
        preregistration = a_preregistration(search=a_search(seed=refit))
        found = assess_cell(
            _Toy({}, seed=refit),
            preregistration,
            NULL_CELL,
            pairs_from(FITS_ON, seed=1000 + refit),
            pairs_from(TESTS_ON, seed=2000 + refit, first=FITS_ON),
        )
        pvalues.append(found.p_value.value)
        accuracies.append(found.result.accuracy)

    assert rejection_excess(pvalues, REPLICATES) < calibration_bound(refits, 0.01)
    assert min(pvalues) == pytest.approx(1 / (REPLICATES + 1))
    assert max(pvalues) == 1.0

    # a null that shuffles the observed outcomes leaves the accuracy where it was,
    # so every p-value is one and fails the nondegeneracy check above
    shuffled = [
        LabelPermutation(replicates=REPLICATES, seed=refit)
        .test(fmean, outcomes_at(accuracy), tail="greater")
        .value
        for refit, accuracy in enumerate(accuracies)
    ]

    assert all(p == 1.0 for p in shuffled)
    assert min(shuffled) > min(pvalues)
