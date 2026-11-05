"""Wall-clock cost of EAP-IG against the forward pass it prices.

The benchmark times one plain forward pass and one five-step integrated-gradient
attribution over the same model and inputs.
"""

from itertools import pairwise
from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Final, Self

import pytest
import torch

from lofi.deduce import ItemKey
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
    ComputationGraph,
    EdgeAttribution,
    GraphEdge,
    GraphNode,
    Unit,
    attribute_edges,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from pytest_benchmark.fixture import BenchmarkFixture
    from torch import Tensor

    from lofi.nn.ports import Patch
#: How many integration right endpoints are evaluated.
INTEGRATION_STEPS: Final = 5

#: The absolute forward-equivalent wall-clock budget for five-step EAP-IG.
FORWARD_EQUIVALENTS: Final = 20.0

MODEL: Final = ComponentId("two-path")
WIDTH: Final = 1024
ITEMS: Final = 256
DEPTH: Final = 3
PATHS: Final = ("a", "b")
OUTPUT: Final = "output"
NAMES: Final = (
    *(f"{path}{layer}" for path in PATHS for layer in range(1, DEPTH + 1)),
    OUTPUT,
)
SITES: Final = {name: Site(MODEL, Role.RESIDUAL_STREAM, (name,)) for name in NAMES}
NODES: Final = {name: GraphNode(MODEL, ItemKey((0, 1), name)) for name in NAMES}
AXES: Final = ("item", "width")

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    MODEL,
    {
        site: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        )
        for site in SITES.values()
    },
)


class _Held:
    __slots__ = ("_site", "_tensor")

    def __init__(self, site: Site, tensor: Tensor) -> None:
        self._site = site
        self._tensor = tensor

    @property
    def site(self) -> Site:
        return self._site

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
        model: _TwoPath,
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

    def pass_through(self, site: Site, tensor: Tensor) -> Tensor:
        standing = self._patches.get(site)

        if standing is not None:
            tensor = standing.stands_in_for(_Held(site, tensor))

        if site in self._sites:
            self._readings[site] = _Held(site, tensor)

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


class _TwoPath:
    """Two depth-three paths of square layers, read out at every layer."""

    def __init__(self, seed: int) -> None:
        draws = torch.Generator().manual_seed(seed)
        self.weights = {
            path: [
                torch.randn(WIDTH, WIDTH, generator=draws) / WIDTH**0.5
                for _ in range(DEPTH)
            ]
            for path in PATHS
        }
        self.readouts = {
            path: [torch.randn(WIDTH, generator=draws) for _ in range(DEPTH)]
            for path in PATHS
        }
        self.scopes: list[_Scope] = []

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def at(self, name: str, tensor: Tensor) -> Tensor:
        for scope in self.scopes:
            tensor = scope.pass_through(SITES[name], tensor)

        return tensor

    def forward(self, given: Tensor) -> Tensor:
        streams = dict.fromkeys(PATHS, given)
        readout = torch.zeros(given.shape[0])

        for layer in range(DEPTH):
            for path, stream in streams.items():
                stepped = self.at(
                    f"{path}{layer + 1}", torch.tanh(stream @ self.weights[path][layer])
                )
                streams[path] = stepped
                readout = readout + stepped @ self.readouts[path][layer]

        return self.at(OUTPUT, readout.unsqueeze(-1)).sum()


def a_graph() -> ComputationGraph[str]:
    edges: list[GraphEdge[str]] = []

    for path in PATHS:
        chain = [f"{path}{layer}" for layer in range(1, DEPTH + 1)]
        edges.extend(
            GraphEdge(NODES[source], NODES[target], Role.RESIDUAL_STREAM)
            for source, target in pairwise(chain)
        )
        edges.extend(
            GraphEdge(NODES[source], NODES[OUTPUT], Role.RESIDUAL_STREAM)
            for source in chain
        )

    return ComputationGraph(
        NODES.values(), edges, {NODES[name]: site for name, site in SITES.items()}
    )


def a_pair(seed: int) -> tuple[Tensor, Tensor]:
    draws = torch.Generator().manual_seed(seed)
    clean = torch.randn(ITEMS, WIDTH, generator=draws)
    counterfactual = torch.randn(ITEMS, WIDTH, generator=draws)

    return clean, counterfactual


def seconds_per_forward(model: _TwoPath, given: Tensor, *, rounds: int) -> float:
    """Return the median wall seconds of one forward pass without a gradient."""
    taken: list[float] = []

    with torch.no_grad():
        model.forward(given)

        for _ in range(rounds):
            start = perf_counter()
            model.forward(given)
            taken.append(perf_counter() - start)

    return median(taken)


def _attribute(
    model: _TwoPath,
    graph: ComputationGraph[str],
    clean: Tensor,
    counterfactual: Tensor,
) -> EdgeAttribution[str]:
    return attribute_edges(
        model,
        graph,
        targets={edge: Unit(SITES[edge.source.key.reading]) for edge in graph.edges},
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=INTEGRATION_STEPS,
        dataset="two-path",
    )


def test_attribution_returns_one_score_per_edge() -> None:
    model = _TwoPath(0)
    graph = a_graph()
    clean, counterfactual = a_pair(1)

    attributed = _attribute(model, graph, clean, counterfactual)

    assert len(attributed) == len(graph.edges)


def test_an_attribution_costs_a_bounded_number_of_forward_passes(
    benchmark: BenchmarkFixture,
) -> None:
    model = _TwoPath(0)
    graph = a_graph()
    clean, counterfactual = a_pair(1)
    forward_seconds = seconds_per_forward(model, clean, rounds=9)

    benchmark.pedantic(
        _attribute,
        args=(model, graph, clean, counterfactual),
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the attribution ran but pytest-benchmark timed nothing")

    # use the median to reduce sensitivity to scheduler delays
    attribution_seconds = measured.stats.median
    forward_equivalents = attribution_seconds / forward_seconds
    benchmark.extra_info["edges"] = len(graph.edges)
    benchmark.extra_info["integration_steps"] = INTEGRATION_STEPS
    benchmark.extra_info["forwards"] = INTEGRATION_STEPS + 1
    benchmark.extra_info["backwards"] = INTEGRATION_STEPS
    benchmark.extra_info["forward_seconds"] = forward_seconds
    benchmark.extra_info["attribution_seconds"] = attribution_seconds
    benchmark.extra_info["forward_equivalents"] = forward_equivalents

    assert forward_equivalents <= FORWARD_EQUIVALENTS
