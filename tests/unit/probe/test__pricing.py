import itertools
import pickle
from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import torch

from lofi import _checks
from lofi.nn.policy import DevicePolicy
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe import _pricing
from lofi.probe._causal import AlignmentPolicy, CounterfactualPair, Side, TypeAt
from lofi.probe._das import (
    AlignmentObjective,
    Annealing,
    BoundlessSearch,
    DistributedAlignment,
    FixedRankSearch,
    Schedule,
)
from lofi.probe._pricing import (
    CostModel,
    GridSpec,
    ProbeUnit,
    UnitPrice,
    UnpricedUnitError,
    price_unit,
)
from lofi.probe._provenance import Provenance
from lofi.stats import LabelPermutation

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._causal import CellAssignment, HighLevelVariable

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
    * 4
)

FITTING: Final = PAIRS[:32]
HELD_OUT: Final = PAIRS[32:48]


def an_alignment(replicates: int) -> DistributedAlignment[_Held]:
    return DistributedAlignment(
        _Toy(a_basis()),
        HIDDEN,
        CATEGORY,
        search=FixedRankSearch(
            rank=BITS,
            schedule=Schedule(epochs=6, learning_rate=0.05, batch_size=8, seed=0),
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
            schedule=Schedule(epochs=6, learning_rate=0.05, batch_size=8, seed=0),
            null=LabelPermutation(replicates=replicates, seed=1),
            objective=AlignmentObjective.CROSS_ENTROPY,
        ),
    )


def a_price(
    unit: ProbeUnit = ProbeUnit.FIXED_ALIGNMENT_FIT,
    *,
    seconds: float = 2.0,
    device: str = "cpu",
) -> UnitPrice:
    return UnitPrice(
        unit,
        seconds_per_run=seconds,
        seconds_fastest=seconds / 2,
        warmup=1,
        repeats=3,
        device=device,
    )


def a_grid(
    axes: tuple[tuple[str, int], ...] = (),
    *,
    fit_unit: ProbeUnit = ProbeUnit.FIXED_ALIGNMENT_FIT,
    refit_unit: ProbeUnit = ProbeUnit.FIXED_PERMUTATION_REFIT,
    replicates: int = 0,
    attribution_runs: int = 0,
    pruning_runs: int = 0,
) -> GridSpec:
    return GridSpec(
        axes,
        fit_unit=fit_unit,
        refit_unit=refit_unit,
        replicates=replicates,
        attribution_runs=attribution_runs,
        pruning_runs=pruning_runs,
    )


class Counted:
    def __init__(self) -> None:
        self.runs = 0

    def __call__(self) -> int:
        self.runs += 1

        return self.runs


def an_accelerator() -> DevicePolicy:
    device = torch.accelerator.current_accelerator(check_available=True)

    if device is None:
        pytest.skip("no accelerator to price on")

    return DevicePolicy(device, CPU.dtype)


@pytest.fixture(autouse=True)
def _unchecked(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_checks, "CHECKS", False)


def test_a_unit_is_run_warmup_plus_repeats_times_and_priced_off_the_timed_runs() -> (
    None
):
    body = Counted()

    price = price_unit(ProbeUnit.ATTRIBUTION, body, policy=CPU, warmup=2, repeats=4)

    assert body.runs == 6
    assert price.unit is ProbeUnit.ATTRIBUTION
    assert price.warmup == 2
    assert price.repeats == 4
    assert price.seconds_per_run > 0
    assert 0 < price.seconds_fastest <= price.seconds_per_run
    assert price.device == "cpu"


def test_one_timed_run_is_both_the_mean_and_the_fastest() -> None:
    price = price_unit(ProbeUnit.PRUNING, Counted(), policy=CPU, warmup=0, repeats=1)

    assert price.seconds_fastest == price.seconds_per_run


def test_a_cpu_unit_has_no_allocator_peak_to_read() -> None:
    assert (
        price_unit(ProbeUnit.PRUNING, Counted(), policy=CPU).peak_allocated_bytes
        is None
    )


def test_a_meta_device_is_refused() -> None:
    meta = DevicePolicy(torch.device("meta"), torch.float32)
    body = Counted()

    with pytest.raises(ValueError, match="meta device computes nothing"):
        price_unit(ProbeUnit.FIXED_ALIGNMENT_FIT, body, policy=meta)

    assert body.runs == 0


@pytest.mark.parametrize("repeats", [0, -1])
def test_fewer_than_one_timed_run_is_refused(repeats: int) -> None:
    with pytest.raises(ValueError, match="at least one timed run"):
        price_unit(
            ProbeUnit.FIXED_ALIGNMENT_FIT, Counted(), policy=CPU, repeats=repeats
        )


def test_a_negative_warmup_is_refused() -> None:
    with pytest.raises(ValueError, match="warmup count is not negative"):
        price_unit(ProbeUnit.FIXED_ALIGNMENT_FIT, Counted(), policy=CPU, warmup=-1)


def test_a_checked_build_is_refused_before_anything_runs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_checks, "CHECKS", True)
    body = Counted()

    with pytest.raises(RuntimeError, match="LOFI_CHECK"):
        price_unit(ProbeUnit.FIXED_ALIGNMENT_FIT, body, policy=CPU)

    assert body.runs == 0


def test_the_fastest_run_is_the_minimum_and_the_mean_is_the_mean(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    clock = iter((0.0, 3.0, 10.0, 11.0, 20.0, 25.0))
    monkeypatch.setattr(_pricing, "perf_counter", lambda: next(clock))
    body = Counted()

    price = price_unit(ProbeUnit.PRUNING, body, policy=CPU, warmup=0, repeats=3)

    assert body.runs == 3
    assert price.seconds_fastest == 1.0
    assert price.seconds_per_run == 3.0


def test_a_cpu_unit_never_touches_the_accelerator(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    touched = Counted()
    monkeypatch.setattr(torch.accelerator, "synchronize", lambda _: touched())
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda _: touched()
    )
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda _: touched())

    price_unit(ProbeUnit.PRUNING, Counted(), policy=CPU, warmup=1, repeats=3)

    assert touched.runs == 0


def test_an_accelerator_unit_is_synchronized_before_and_after_every_timed_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    policy = DevicePolicy(torch.device("cuda", 1), CPU.dtype)
    synchronized: list[int] = []
    body = Counted()
    monkeypatch.setattr(
        torch.accelerator, "synchronize", lambda _: synchronized.append(body.runs)
    )
    reset: list[int] = []
    monkeypatch.setattr(
        torch.accelerator, "reset_peak_memory_stats", lambda _: reset.append(body.runs)
    )
    monkeypatch.setattr(torch.accelerator, "max_memory_allocated", lambda _: 4096)

    price = price_unit(ProbeUnit.PRUNING, body, policy=policy, warmup=2, repeats=3)

    assert synchronized == [2, 3, 4, 5]
    assert reset == [2]
    assert price.peak_allocated_bytes == 4096
    assert price.device == str(policy.device)


def test_an_accelerator_peak_is_read_over_the_timed_runs_alone() -> None:
    policy = an_accelerator()
    width = 1024
    earlier = torch.ones(width, width, dtype=policy.dtype, device=policy.device)
    earlier_bytes = earlier.numel() * earlier.element_size()
    later_bytes = width * earlier.element_size()
    del earlier

    price = price_unit(
        ProbeUnit.PRUNING,
        lambda: torch.ones(width, dtype=policy.dtype, device=policy.device),
        policy=policy,
        warmup=1,
        repeats=2,
    )

    assert price.peak_allocated_bytes is not None
    assert later_bytes <= price.peak_allocated_bytes < earlier_bytes


def test_a_price_holds_what_it_was_built_from() -> None:
    price = UnitPrice(
        ProbeUnit.FIXED_PERMUTATION_REFIT,
        seconds_per_run=1.5,
        seconds_fastest=1.0,
        warmup=2,
        repeats=5,
        device="cuda:0",
        peak_allocated_bytes=4096,
    )

    assert price.unit is ProbeUnit.FIXED_PERMUTATION_REFIT
    assert price.seconds_per_run == 1.5
    assert price.seconds_fastest == 1.0
    assert price.warmup == 2
    assert price.repeats == 5
    assert price.device == "cuda:0"
    assert price.peak_allocated_bytes == 4096


def test_a_price_refuses_a_unit_that_is_not_a_probe_unit() -> None:
    with pytest.raises(TypeError, match="ProbeUnit"):
        UnitPrice(
            cast("ProbeUnit", "alignment_fit"),
            seconds_per_run=1.0,
            seconds_fastest=1.0,
            warmup=0,
            repeats=1,
            device="cpu",
        )


@pytest.mark.parametrize("seconds", [0.0, -1.0, float("inf"), float("nan")])
def test_a_price_refuses_a_duration_that_is_not_positive_and_finite(
    seconds: float,
) -> None:
    with pytest.raises(ValueError, match="positive duration"):
        a_price(seconds=seconds)


def test_a_price_refuses_a_duration_that_is_not_a_number() -> None:
    with pytest.raises(TypeError, match="not a number"):
        UnitPrice(
            ProbeUnit.PRUNING,
            seconds_per_run=cast("float", "long"),
            seconds_fastest=1.0,
            warmup=0,
            repeats=1,
            device="cpu",
        )


def test_a_price_refuses_a_fastest_run_slower_than_the_mean() -> None:
    with pytest.raises(ValueError, match="against a mean"):
        UnitPrice(
            ProbeUnit.PRUNING,
            seconds_per_run=1.0,
            seconds_fastest=2.0,
            warmup=0,
            repeats=1,
            device="cpu",
        )


@pytest.mark.parametrize(
    ("field", "value", "match"),
    [
        ("warmup", -1, "negative"),
        ("repeats", 0, "at least one timed run"),
        ("device", "", "empty"),
        ("peak_allocated_bytes", -1, "negative"),
    ],
)
def test_a_price_refuses_a_bad_count_or_an_unnamed_device(
    field: str, value: int | str, match: str
) -> None:
    fields: dict[str, int | str] = {
        "warmup": 1,
        "repeats": 2,
        "device": "cpu",
        "peak_allocated_bytes": 0,
        field: value,
    }

    with pytest.raises(ValueError, match=match):
        UnitPrice(
            ProbeUnit.PRUNING,
            seconds_per_run=1.0,
            seconds_fastest=1.0,
            warmup=cast("int", fields["warmup"]),
            repeats=cast("int", fields["repeats"]),
            device=cast("str", fields["device"]),
            peak_allocated_bytes=cast("int", fields["peak_allocated_bytes"]),
        )


def test_prices_are_equal_on_every_field_and_pickle() -> None:
    price = a_price()

    assert price == a_price()
    assert hash(price) == hash(a_price())
    assert price != a_price(ProbeUnit.PRUNING)
    assert price != a_price(seconds=3.0)
    assert price != a_price(device="cuda:0")
    assert price != "price"
    assert pickle.loads(pickle.dumps(price)) == price
    assert repr(price) == (
        "UnitPrice(<ProbeUnit.FIXED_ALIGNMENT_FIT: 'fixed_alignment_fit'>, "
        "seconds_per_run=2.0, "
        "seconds_fastest=1.0, warmup=1, repeats=3, device='cpu', "
        "peak_allocated_bytes=None)"
    )


def test_a_grid_crosses_its_axes_to_cells() -> None:
    grid = a_grid((("variable", 3), ("layer", 12), ("position", 5)), replicates=2)

    assert grid.axes == (("variable", 3), ("layer", 12), ("position", 5))
    assert grid.cells == 180
    assert grid.replicates == 2
    assert grid.attribution_runs == 0
    assert grid.pruning_runs == 0


def test_a_grid_of_no_axes_is_one_cell() -> None:
    assert a_grid().cells == 1
    assert a_grid().demand() == {ProbeUnit.FIXED_ALIGNMENT_FIT: 1}


def test_a_grid_demands_one_fit_per_cell_and_replicates_refits_per_cell() -> None:
    grid = a_grid(
        (("variable", 2), ("layer", 3)),
        replicates=4,
        attribution_runs=5,
        pruning_runs=1,
    )

    assert dict(grid.demand()) == {
        ProbeUnit.FIXED_ALIGNMENT_FIT: 6,
        ProbeUnit.FIXED_PERMUTATION_REFIT: 24,
        ProbeUnit.ATTRIBUTION: 5,
        ProbeUnit.PRUNING: 1,
    }


def test_a_grid_leaves_a_unit_it_needs_none_of_out_of_its_demand() -> None:
    assert ProbeUnit.FIXED_PERMUTATION_REFIT not in a_grid((("layer", 2),)).demand()
    assert ProbeUnit.ATTRIBUTION not in a_grid((("layer", 2),)).demand()
    assert ProbeUnit.PRUNING not in a_grid((("layer", 2),)).demand()


def test_a_grid_refuses_an_empty_axis_a_repeated_axis_or_an_unnamed_one() -> None:
    with pytest.raises(ValueError, match="no cell"):
        a_grid((("layer", 0),))

    with pytest.raises(ValueError, match="named twice"):
        a_grid((("layer", 2), ("layer", 3)))

    with pytest.raises(ValueError, match="empty"):
        a_grid((("", 2),))

    with pytest.raises(ValueError, match="negative"):
        a_grid((("layer", -2),))


def test_a_grid_requires_one_coherent_alignment_pricing_pair() -> None:
    build = cast("Callable[..., GridSpec]", GridSpec)

    with pytest.raises(TypeError, match="fit_unit"):
        build()

    with pytest.raises(ValueError, match="do not price one alignment strategy"):
        GridSpec(
            (),
            fit_unit=ProbeUnit.FIXED_ALIGNMENT_FIT,
            refit_unit=ProbeUnit.BOUNDLESS_PERMUTATION_REFIT,
        )


def test_a_grid_refuses_a_negative_count() -> None:
    with pytest.raises(ValueError, match="negative"):
        a_grid(replicates=-1)

    with pytest.raises(ValueError, match="negative"):
        a_grid(attribution_runs=-1)

    with pytest.raises(ValueError, match="negative"):
        a_grid(pruning_runs=-1)


def test_grids_are_equal_on_their_axes_and_counts_and_pickle() -> None:
    grid = a_grid((("layer", 2),), replicates=1)

    assert grid == a_grid((("layer", 2),), replicates=1)
    assert hash(grid) == hash(a_grid((("layer", 2),), replicates=1))
    assert grid != a_grid((("layer", 3),), replicates=1)
    assert grid != a_grid((("layer", 2),), replicates=2)
    assert grid != a_grid((("layer", 2),), replicates=1, pruning_runs=1)
    assert grid != "grid"
    assert pickle.loads(pickle.dumps(grid)) == grid
    assert repr(grid) == (
        "GridSpec((('layer', 2),), "
        "fit_unit=<ProbeUnit.FIXED_ALIGNMENT_FIT: 'fixed_alignment_fit'>, "
        "refit_unit=<ProbeUnit.FIXED_PERMUTATION_REFIT: "
        "'fixed_permutation_refit'>, replicates=1, attribution_runs=0, "
        "pruning_runs=0)"
    )


def test_a_model_sums_the_demanded_runs_at_their_measured_prices() -> None:
    model = CostModel(
        (
            a_price(ProbeUnit.FIXED_ALIGNMENT_FIT, seconds=60.0),
            a_price(ProbeUnit.FIXED_PERMUTATION_REFIT, seconds=30.0),
            a_price(ProbeUnit.ATTRIBUTION, seconds=600.0),
        )
    )
    grid = a_grid((("layer", 4), ("position", 3)), replicates=10, attribution_runs=2)

    assert model.seconds(grid) == pytest.approx(12 * 60.0 + 120 * 30.0 + 2 * 600.0)
    assert model.hours(grid) == pytest.approx(model.seconds(grid) / 3600.0)
    assert model.priced == (
        ProbeUnit.FIXED_ALIGNMENT_FIT,
        ProbeUnit.FIXED_PERMUTATION_REFIT,
        ProbeUnit.ATTRIBUTION,
    )
    assert model.device == "cpu"


def test_a_model_lists_its_priced_units_in_declaration_order() -> None:
    model = CostModel(
        (a_price(ProbeUnit.PRUNING), a_price(ProbeUnit.FIXED_ALIGNMENT_FIT))
    )

    assert model.priced == (ProbeUnit.FIXED_ALIGNMENT_FIT, ProbeUnit.PRUNING)


def test_a_model_refuses_a_grid_naming_an_unpriced_unit_by_name() -> None:
    model = CostModel((a_price(ProbeUnit.FIXED_ALIGNMENT_FIT),))
    grid = a_grid((("layer", 2),), replicates=3, pruning_runs=1)

    with pytest.raises(
        UnpricedUnitError, match="fixed_permutation_refit, pruning"
    ) as raised:
        model.hours(grid)

    assert raised.value.unpriced == (
        ProbeUnit.FIXED_PERMUTATION_REFIT,
        ProbeUnit.PRUNING,
    )
    assert raised.value.priced == (ProbeUnit.FIXED_ALIGNMENT_FIT,)
    assert "fixed_alignment_fit" in str(raised.value)


def test_a_model_costs_a_grid_that_needs_only_what_it_prices() -> None:
    model = CostModel((a_price(ProbeUnit.FIXED_ALIGNMENT_FIT, seconds=5.0),))

    assert model.seconds(a_grid((("layer", 3),))) == pytest.approx(15.0)


def test_an_empty_model_prices_nothing_and_refuses_every_grid() -> None:
    model = CostModel(())

    assert model.priced == ()
    assert model.device is None
    assert model.prices == {}

    with pytest.raises(UnpricedUnitError, match="priced: nothing"):
        model.seconds(a_grid())


def test_a_model_refuses_two_prices_for_one_unit() -> None:
    with pytest.raises(ValueError, match="priced twice"):
        CostModel((a_price(), a_price(seconds=3.0)))


def test_a_model_refuses_prices_from_different_devices() -> None:
    with pytest.raises(ValueError, match="one device"):
        CostModel((a_price(), a_price(ProbeUnit.PRUNING, device="cuda:0")))


def test_a_model_reports_itself_as_its_constructor_call() -> None:
    price = a_price()

    assert repr(CostModel((price,))) == f"CostModel([{price!r}])"


def test_the_real_alignment_units_are_priced_and_a_grid_is_costed_from_them() -> None:
    fixed_fit = price_unit(
        ProbeUnit.FIXED_ALIGNMENT_FIT,
        lambda: an_alignment(1).fit(FITTING),
        policy=CPU,
        warmup=1,
        repeats=2,
    )
    fitted = an_alignment(1).fit(FITTING)
    fixed_refit = price_unit(
        ProbeUnit.FIXED_PERMUTATION_REFIT,
        lambda: fitted.assess(HELD_OUT),
        policy=CPU,
        warmup=1,
        repeats=2,
    )
    boundless_fit = price_unit(
        ProbeUnit.BOUNDLESS_ALIGNMENT_FIT,
        lambda: a_boundless_alignment(1).fit(FITTING),
        policy=CPU,
        warmup=1,
        repeats=2,
    )
    boundless = a_boundless_alignment(1).fit(FITTING)
    boundless_refit = price_unit(
        ProbeUnit.BOUNDLESS_PERMUTATION_REFIT,
        lambda: boundless.assess(HELD_OUT),
        policy=CPU,
        warmup=1,
        repeats=2,
    )
    model = CostModel((fixed_fit, fixed_refit, boundless_fit, boundless_refit))
    fixed_grid = a_grid((("variable", 3), ("layer", 2)), replicates=5)
    boundless_grid = a_grid(
        (("variable", 3), ("layer", 2)),
        fit_unit=ProbeUnit.BOUNDLESS_ALIGNMENT_FIT,
        refit_unit=ProbeUnit.BOUNDLESS_PERMUTATION_REFIT,
        replicates=5,
    )

    assert model.priced == tuple(ProbeUnit)[:4]
    assert model.seconds(fixed_grid) == pytest.approx(
        6 * fixed_fit.seconds_per_run + 30 * fixed_refit.seconds_per_run
    )
    assert model.seconds(boundless_grid) == pytest.approx(
        6 * boundless_fit.seconds_per_run + 30 * boundless_refit.seconds_per_run
    )

    with pytest.raises(UnpricedUnitError, match="attribution"):
        model.hours(a_grid(attribution_runs=1))
