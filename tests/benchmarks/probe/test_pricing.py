"""Wall-clock validation for the probe cost model.

The benchmark prices fixed and boundless fit and refit units on a toy subject,
then requires an unseen sweep estimate to fall within 25 percent of its measured
time.
"""

import itertools
from functools import partial
from statistics import fmean
from typing import TYPE_CHECKING, Final, Self

import pytest
import torch

from lofi import _checks
from lofi.nn.policy import DevicePolicy
from lofi.nn.sites import (
    Access,
    ComponentId,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.probe import (
    AlignmentPolicy,
    Annealing,
    BoundlessSearch,
    CostModel,
    CounterfactualPair,
    DistributedAlignment,
    FixedRankSearch,
    GridSpec,
    ProbeUnit,
    Provenance,
    Schedule,
    Side,
    TypeAt,
    price_unit,
)
from lofi.probe._das import AlignmentObjective
from lofi.stats import LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence
    from types import TracebackType

    from pytest_benchmark.fixture import BenchmarkFixture
    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe import CellAssignment, HighLevelVariable

#: How far the model's seconds may sit from a sweep's wall clock, as a fraction.
TOLERANCE: Final = 0.25

CPU: Final = DevicePolicy(torch.device("cpu"), torch.float32)

PARSER: Final = ComponentId("priced-parser")
HIDDEN: Final = Site(PARSER, Role.CHART_CELL, (0, 2))
CATEGORY: Final = TypeAt((0, 2))
CATEGORIES: Final = ("N", "NP", "S", "PP")
BITS: Final = 2
WIDTH: Final = 6
GENERATED_BY: Final = Provenance("planted", "toy", 0)
POLICY: Final = AlignmentPolicy("first", "exclude", contiguous=True)

CODES: Final = torch.tensor(
    [
        [1.0 if (index >> bit) & 1 else -1.0 for bit in range(BITS)]
        for index in range(len(CATEGORIES))
    ]
)

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    PARSER,
    {
        HIDDEN: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
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


class _Toy:
    """Plants the category at the cell as two signed bits in a rotated basis."""

    def __init__(self, basis: Tensor) -> None:
        self._basis = basis
        self.readout = torch.nn.Parameter(CODES.clone())
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
        return CATEGORIES if variable == CATEGORY else ()

    def run(self, pairs: Sequence[CounterfactualPair], side: Side, /) -> Tensor:
        decided = [pair.base if side is Side.BASE else pair.variant for pair in pairs]
        codes = torch.stack(
            [CODES[CATEGORIES.index(str(at[CATEGORY]))] for at in decided]
        )
        nuisance = torch.arange(WIDTH - BITS, dtype=codes.dtype).expand(len(pairs), -1)
        hidden = torch.cat([codes, nuisance], dim=-1) @ self._basis

        for scope in self.scopes:
            hidden = scope.intervene(HIDDEN, hidden, ("item", "width"))
            scope.observe(HIDDEN, hidden, ("item", "width"))

        return (hidden @ self._basis.T)[:, :BITS] @ self.readout.T


def a_basis() -> Tensor:
    orthogonal, _ = torch.linalg.qr(
        torch.randn(WIDTH, WIDTH, generator=torch.Generator().manual_seed(3))
    )

    return orthogonal


def a_pair(base: str, variant: str) -> CounterfactualPair:
    before: CellAssignment = {CATEGORY: base}
    after: CellAssignment = {CATEGORY: variant}

    return CounterfactualPair(before, after, CATEGORY, (0, 4), GENERATED_BY)


PAIRS: Final = (
    tuple(
        a_pair(base, variant) for base, variant in itertools.permutations(CATEGORIES, 2)
    )
    * 12
)

FITTING: Final = PAIRS[:96]
HELD_OUT: Final = PAIRS[96:144]


def a_fixed_alignment(replicates: int) -> DistributedAlignment[_Held]:
    return DistributedAlignment(
        _Toy(a_basis()),
        HIDDEN,
        CATEGORY,
        search=FixedRankSearch(
            rank=BITS,
            schedule=Schedule(epochs=30, learning_rate=0.05, batch_size=8, seed=0),
            null=LabelPermutation(replicates=replicates, seed=1),
            objective=AlignmentObjective.CROSS_ENTROPY,
        ),
    )


def a_boundless_alignment(replicates: int) -> DistributedAlignment[_Held]:
    return DistributedAlignment(
        _Toy(a_basis()),
        HIDDEN,
        CATEGORY,
        search=BoundlessSearch(
            boundaries=(0.0, float(BITS)),
            margin=1e-6,
            annealing=Annealing(initial=1.0, final=0.05),
            schedule=Schedule(epochs=30, learning_rate=0.05, batch_size=8, seed=0),
            null=LabelPermutation(replicates=replicates, seed=1),
            objective=AlignmentObjective.CROSS_ENTROPY,
        ),
    )


def a_model() -> CostModel:
    """Price the four alignment units on the toy subject."""
    fitting = price_unit(
        ProbeUnit.FIXED_ALIGNMENT_FIT,
        lambda: a_fixed_alignment(1).fit(FITTING),
        policy=CPU,
        warmup=1,
        repeats=3,
    )
    fitted = a_fixed_alignment(1).fit(FITTING)
    refitting = price_unit(
        ProbeUnit.FIXED_PERMUTATION_REFIT,
        lambda: fitted.assess(HELD_OUT),
        policy=CPU,
        warmup=1,
        repeats=3,
    )

    boundless_fitting = price_unit(
        ProbeUnit.BOUNDLESS_ALIGNMENT_FIT,
        lambda: a_boundless_alignment(1).fit(FITTING),
        policy=CPU,
        warmup=1,
        repeats=3,
    )
    boundless = a_boundless_alignment(1).fit(FITTING)
    boundless_refitting = price_unit(
        ProbeUnit.BOUNDLESS_PERMUTATION_REFIT,
        lambda: boundless.assess(HELD_OUT),
        policy=CPU,
        warmup=1,
        repeats=3,
    )

    return CostModel((fitting, refitting, boundless_fitting, boundless_refitting))


def sweep(grid: GridSpec, *, boundless: bool) -> None:
    """Fit and assess one alignment per cell of the grid, serially.

    Parameters
    ----------
    grid : GridSpec
    boundless : bool
        Whether each cell runs the boundless rather than fixed strategy.
    """
    for _ in range(grid.cells):
        alignment = (
            a_boundless_alignment(grid.replicates)
            if boundless
            else a_fixed_alignment(grid.replicates)
        )
        alignment.fit(FITTING).assess(HELD_OUT)


def priced(
    grid: GridSpec, predictions: list[float], *, boundless: bool
) -> tuple[tuple[GridSpec], dict[str, bool]]:
    predictions.append(a_model().seconds(grid))

    return (grid,), {"boundless": boundless}


def a_grid(
    axes: tuple[tuple[str, int], ...], *, replicates: int, boundless: bool
) -> GridSpec:
    fit_unit = (
        ProbeUnit.BOUNDLESS_ALIGNMENT_FIT
        if boundless
        else ProbeUnit.FIXED_ALIGNMENT_FIT
    )
    refit_unit = (
        ProbeUnit.BOUNDLESS_PERMUTATION_REFIT
        if boundless
        else ProbeUnit.FIXED_PERMUTATION_REFIT
    )
    return GridSpec(
        axes, fit_unit=fit_unit, refit_unit=refit_unit, replicates=replicates
    )


def test_pricing_setup_makes_one_prediction_per_round(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)
    grid = a_grid((("variable", 2),), replicates=1, boundless=False)
    predictions: list[float] = []
    setup = partial(priced, grid, predictions, boundless=False)

    for _ in range(3):
        args, kwargs = setup()
        sweep(*args, **kwargs)

    assert len(predictions) == 3


@pytest.mark.parametrize(
    ("grid", "boundless"),
    [
        (a_grid((("variable", 2),), replicates=1, boundless=False), False),
        (
            a_grid(
                (("layer", 1), ("position", 3)),
                replicates=2,
                boundless=False,
            ),
            False,
        ),
        (a_grid((("variable", 1),), replicates=4, boundless=False), False),
        (a_grid((("variable", 2),), replicates=1, boundless=True), True),
        (
            a_grid(
                (("layer", 1), ("position", 3)),
                replicates=2,
                boundless=True,
            ),
            True,
        ),
        (a_grid((("variable", 1),), replicates=4, boundless=True), True),
    ],
    ids=[
        "fixed-two-cells-one-replicate",
        "fixed-three-cells-two-replicates",
        "fixed-one-cell-four",
        "boundless-two-cells-one-replicate",
        "boundless-three-cells-two-replicates",
        "boundless-one-cell-four",
    ],
)
@pytest.mark.usefixtures("single_threaded_cpu")
def test_the_model_reproduces_the_wall_clock_of_a_held_out_sweep(
    benchmark: BenchmarkFixture, grid: GridSpec, *, boundless: bool
) -> None:
    predictions: list[float] = []

    benchmark.pedantic(
        sweep,
        setup=partial(priced, grid, predictions, boundless=boundless),
        rounds=3,
        iterations=1,
        warmup_rounds=0,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the sweep ran but pytest-benchmark timed nothing")

    predicted = fmean(predictions)
    actual = measured.stats.mean
    benchmark.extra_info["cells"] = grid.cells
    benchmark.extra_info["replicates"] = grid.replicates
    benchmark.extra_info["predicted_seconds"] = predicted
    benchmark.extra_info["actual_seconds"] = actual
    benchmark.extra_info["torch_num_threads"] = torch.get_num_threads()

    assert abs(predicted - actual) / actual <= TOLERANCE
