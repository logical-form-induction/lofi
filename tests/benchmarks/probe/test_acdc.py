"""Wall-clock cost of receiver-local ACDC against its model forward."""

from functools import partial
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Final, Self

import pytest
import torch

from lofi.deduce import ItemKey
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
    AbsoluteThreshold,
    ACDCResult,
    ComputationGraph,
    GraphEdge,
    GraphNode,
    ProbeUnit,
    Provenance,
    PruningBudget,
    Unit,
    acdc,
    price_unit,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from pytest_benchmark.fixture import BenchmarkFixture
    from torch import Tensor

    from lofi.nn.ports import Patch

#: Number of receiver edges in the benchmark graph.
EDGES: Final = 8

#: Number of activation rows in one forward.
ROWS: Final = 128

#: Width of every activation row.
WIDTH: Final = 1024

#: Absolute wall-clock budget relative to one plain forward.
FORWARD_EQUIVALENTS: Final = 64.0

MODEL: Final = ComponentId("acdc-benchmark")
TARGET: Final = Site(MODEL, Role.RECEIVER_EDGE, ("inputs",))
AXES: Final = ("edge", "width")
CPU: Final = DevicePolicy(torch.device("cpu"), torch.float32)
DATA: Final = Provenance("endpoint-pair", "synthetic-acdc", 1)
POLICY: Final = Provenance("reverse-topological-acdc", "synthetic-acdc", None)


class _Held:
    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    @property
    def site(self) -> Site:
        return TARGET

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return AXES

    @property
    def mask(self) -> Tensor | None:
        return None


class _Scope:
    def __init__(
        self,
        model: _Receiver,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[_Held]],
    ) -> None:
        self._model = model
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Held] = {}

    @property
    def readings(self) -> Mapping[Site, _Held]:
        return self._readings

    def pass_through(self, tensor: Tensor) -> Tensor:
        standing = self._patches.get(TARGET)

        if standing is not None:
            tensor = standing.stands_in_for(_Held(tensor))

        if TARGET in self._sites:
            self._readings[TARGET] = _Held(tensor)

        return tensor

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


class _Receiver:
    """A receiver with a moderately expensive dense readout."""

    def __init__(self, seed: int) -> None:
        draws = torch.Generator().manual_seed(seed)
        self.weight = torch.randn(WIDTH, WIDTH, generator=draws) / WIDTH**0.5
        self.scopes: list[_Scope] = []
        self.calls = 0
        self._published = SiteMap.custom(
            MODEL,
            {
                TARGET: SiteSpec(
                    AXES,
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE | Access.PATCH,
                )
            },
        )

    def site_map(self) -> SiteMap:
        return self._published

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def forward(self, given: Tensor) -> Tensor:
        self.calls += 1
        value = given

        for scope in self.scopes:
            value = scope.pass_through(value)

        return (value @ self.weight).square().sum()


def a_problem() -> tuple[
    _Receiver,
    ComputationGraph[str],
    dict[GraphEdge[str], Unit],
    tuple[GraphEdge[str], ...],
    Tensor,
    Tensor,
]:
    """Return one deterministic receiver-local pruning problem."""
    model = _Receiver(0)
    receiver = GraphNode(MODEL, ItemKey((0, EDGES), "receiver"))
    sources = tuple(
        GraphNode(MODEL, ItemKey((index, index + 1), f"source-{index}"))
        for index in range(EDGES)
    )
    order = tuple(GraphEdge(source, receiver, Role.RECEIVER_EDGE) for source in sources)
    graph = ComputationGraph((*sources, receiver), order)
    targets = {edge: Unit(TARGET, {"edge": index}) for index, edge in enumerate(order)}
    draws = torch.Generator().manual_seed(1)
    clean = torch.randn(ROWS, WIDTH, generator=draws)
    counterfactual = torch.randn(ROWS, WIDTH, generator=draws)

    return model, graph, targets, order, clean, counterfactual


def squared(clean: Tensor, candidate: Tensor) -> Tensor:
    """Return squared output displacement."""
    return (clean - candidate).square().sum()


def prune(
    model: _Receiver,
    graph: ComputationGraph[str],
    targets: Mapping[GraphEdge[str], Unit],
    order: tuple[GraphEdge[str], ...],
    clean: Tensor,
    counterfactual: Tensor,
) -> ACDCResult[str]:
    """Run the actual ACDC kernel over one bound problem."""
    return acdc(
        model,
        graph,
        targets=targets,
        order=order,
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        divergence=squared,
        threshold=AbsoluteThreshold(0.0),
        budget=PruningBudget(EDGES),
        data_provenance=DATA,
        policy_provenance=POLICY,
    )


def seconds_per_forward(model: _Receiver, given: Tensor, *, rounds: int) -> float:
    """Return the median wall seconds of one plain model forward."""
    taken: list[float] = []

    with torch.no_grad():
        model.forward(given)

        for _ in range(rounds):
            start = perf_counter()
            model.forward(given)
            taken.append(perf_counter() - start)

    return median(taken)


def test_acdc_has_a_bounded_forward_equivalent_cost_and_a_real_unit_price(
    benchmark: BenchmarkFixture,
) -> None:
    model, graph, targets, order, clean, counterfactual = a_problem()
    forward_seconds = seconds_per_forward(model, clean, rounds=9)

    benchmark.pedantic(
        prune,
        args=(model, graph, targets, order, clean, counterfactual),
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the ACDC kernel ran but pytest-benchmark timed nothing")

    acdc_seconds = measured.stats.median
    forward_equivalents = acdc_seconds / forward_seconds
    priced = partial(prune, model, graph, targets, order, clean, counterfactual)
    calls_before_price = model.calls
    price = price_unit(
        ProbeUnit.PRUNING,
        priced,
        policy=CPU,
        warmup=1,
        repeats=3,
    )
    calls_per_run = 2 + EDGES
    benchmark.extra_info["edges"] = EDGES
    benchmark.extra_info["forwards"] = calls_per_run
    benchmark.extra_info["forward_seconds"] = forward_seconds
    benchmark.extra_info["acdc_seconds"] = acdc_seconds
    benchmark.extra_info["forward_equivalents"] = forward_equivalents
    benchmark.extra_info["priced_seconds"] = price.seconds_per_run

    assert forward_equivalents <= FORWARD_EQUIVALENTS
    assert price.unit is ProbeUnit.PRUNING
    assert price.seconds_per_run > 0.0
    assert price.device == "cpu"
    assert model.calls - calls_before_price == (price.warmup + price.repeats) * (
        calls_per_run
    )
