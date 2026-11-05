from itertools import pairwise
from typing import TYPE_CHECKING, Final, Self, cast, override

import pytest
import torch
from tests.conformance._suite import Conformance, ConformanceSuite
from tests.unit.nn.parsers.test__scorer import LEAF_AXES as SCORER_LEAF_AXES
from tests.unit.nn.parsers.test__scorer import build as build_scorer
from tests.unit.nn.parsers.test__scorer import leaves as scorer_leaves

from lofi.deduce import ItemKey, SpanIndex
from lofi.judgment.agreement import rank_correlation
from lofi.nn._values import Encoded
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe import _instrument
from lofi.probe._attribution import (
    EdgeAttribution,
    attribute_edges,
    first_order_attribute_edges,
)
from lofi.probe._circuits import AblationEvidence, AttributionEvidence, Unit
from lofi.probe._graph import ComputationGraph, GraphEdge, GraphNode
from lofi.probe._instrument import instrument
from lofi.probe._interventions import PatchFrom
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.parsers import DeepInsideOutside
    from lofi.nn.ports import Patch, Reading

MODEL: Final = ComponentId("two-path")
WIDTH: Final = 4
ITEMS: Final = 8
DEPTH: Final = 3
PLANTED: Final = "a"
DISTRACTOR: Final = "b"
OUTPUT: Final = "output"
NAMES: Final = (
    *(
        f"{path}{layer}"
        for path in (PLANTED, DISTRACTOR)
        for layer in range(1, DEPTH + 1)
    ),
    OUTPUT,
)
SITES: Final = {name: Site(MODEL, Role.RESIDUAL_STREAM, (name,)) for name in NAMES}
NODES: Final = {name: GraphNode(MODEL, ItemKey((0, 1), name)) for name in NAMES}
AXES: Final = ("item", "width")

# how far the distractor path's copy of the category is scaled down
LEAK: Final = 0.1

# how far the weights sit from zero, which sets how far the paths are from linear
SCALE: Final = 0.5

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
    __slots__ = ("_mask", "_site", "_tensor")

    def __init__(self, site: Site, tensor: Tensor, mask: Tensor | None) -> None:
        self._site = site
        self._tensor = tensor
        self._mask = mask

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
        return self._mask


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

    def pass_through(self, site: Site, tensor: Tensor, mask: Tensor | None) -> Tensor:
        standing = self._patches.get(site)

        if standing is not None:
            tensor = standing.stands_in_for(_Held(site, tensor, mask))

        if site in self._sites:
            self._readings[site] = _Held(site, tensor, mask)

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
    """Two depth-three paths from a category to a readout summed over every layer.

    The planted path reads the category alone. The distractor path reads the
    nuisance coordinates and a scaled-down copy of the category. Every weight is
    positive, so the category's sign propagates coherently along both paths.
    """

    def __init__(
        self,
        seed: int,
        *,
        squash: Callable[[Tensor], Tensor] = torch.tanh,
        published: SiteMap = PUBLISHED,
    ) -> None:
        draws = torch.Generator().manual_seed(seed)

        def a_weight() -> Tensor:
            return torch.randn(WIDTH, WIDTH, generator=draws).abs() * SCALE / WIDTH**0.5

        def a_readout() -> Tensor:
            return torch.randn(WIDTH, generator=draws).abs()

        self.weights = {
            path: [a_weight() for _ in range(DEPTH)] for path in (PLANTED, DISTRACTOR)
        }
        self.readouts = {
            path: [a_readout() for _ in range(DEPTH)] for path in (PLANTED, DISTRACTOR)
        }
        self.squash = squash
        self.published = published
        self.scopes: list[_Scope] = []
        self.forwards = 0

    def site_map(self) -> SiteMap:
        return self.published

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def at(self, name: str, tensor: Tensor, mask: Tensor | None) -> Tensor:
        for scope in self.scopes:
            tensor = scope.pass_through(SITES[name], tensor, mask)

        return tensor

    def forward(self, given: Tensor, mask: Tensor | None = None) -> Tensor:
        self.forwards += 1
        category = given[:, :1]
        nuisance = given[:, 1:]
        streams = {
            PLANTED: category.expand(-1, WIDTH),
            DISTRACTOR: torch.cat([category * LEAK, nuisance], dim=-1),
        }
        readout = torch.zeros(given.shape[0])

        for layer in range(DEPTH):
            for path, stream in streams.items():
                stepped = self.at(
                    f"{path}{layer + 1}",
                    self.squash(stream @ self.weights[path][layer]),
                    mask,
                )
                streams[path] = stepped
                readout = readout + stepped @ self.readouts[path][layer]

        readout = self.at(OUTPUT, readout.unsqueeze(-1), mask)

        return readout.sum()


CURVE_MODEL: Final = ComponentId("scalar-curve")
CURVE_SITE: Final = Site(CURVE_MODEL, Role.RESIDUAL_STREAM, ("source",))
CURVE_SOURCE: Final = GraphNode(CURVE_MODEL, ItemKey((0, 1), "source"))
CURVE_TARGET: Final = GraphNode(CURVE_MODEL, ItemKey((0, 1), "target"))
CURVE_PUBLISHED: Final = SiteMap.custom(
    CURVE_MODEL,
    {
        CURVE_SITE: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        )
    },
)


class _ScalarCurve(_TwoPath):
    """A scalar source followed by a quadratic metric."""

    def __init__(self) -> None:
        self.published = CURVE_PUBLISHED
        self.scopes = []
        self.forwards = 0

    @override
    def forward(self, given: Tensor, mask: Tensor | None = None) -> Tensor:
        self.forwards += 1
        source = given

        for scope in self.scopes:
            source = scope.pass_through(CURVE_SITE, source, mask)

        return source.square().sum()


def a_curve_graph() -> ComputationGraph[str]:
    edge = GraphEdge(CURVE_SOURCE, CURVE_TARGET, Role.RESIDUAL_STREAM)

    return ComputationGraph(
        (CURVE_SOURCE, CURVE_TARGET), (edge,), {CURVE_SOURCE: CURVE_SITE}
    )


def a_graph(published: Mapping[str, Site] | None = None) -> ComputationGraph[str]:
    edges: list[GraphEdge[str]] = []

    for path in (PLANTED, DISTRACTOR):
        chain = [f"{path}{layer}" for layer in range(1, DEPTH + 1)]

        edges.extend(
            GraphEdge(NODES[source], NODES[target], Role.RESIDUAL_STREAM)
            for source, target in pairwise(chain)
        )
        edges.extend(
            GraphEdge(NODES[source], NODES[OUTPUT], Role.RESIDUAL_STREAM)
            for source in chain
        )

    sites = SITES if published is None else published

    return ComputationGraph(
        NODES.values(),
        edges,
        {NODES[name]: site for name, site in sites.items()},
    )


def source_targets(graph: ComputationGraph[str]) -> dict[GraphEdge[str], Unit]:
    """Return each edge's target as the whole point its source is published at."""
    targets: dict[GraphEdge[str], Unit] = {}

    for edge in graph.edges:
        site = graph.site_of(edge.source)

        if site is None:
            pytest.fail(f"{edge!r} leaves an unpublished node")

        targets[edge] = Unit(site)

    return targets


def a_pair(seed: int, items: int = ITEMS) -> tuple[Tensor, Tensor]:
    """Return a batch with the category on and the same batch with it off."""
    draws = torch.Generator().manual_seed(100 + seed)
    clean = torch.randn(items, WIDTH, generator=draws) * 0.5
    clean[:, 0] = 1.0
    counterfactual = clean.clone()
    counterfactual[:, 0] = -1.0

    return clean, counterfactual


def brute_force(
    model: _TwoPath, graph: ComputationGraph[str], clean: Tensor, counterfactual: Tensor
) -> dict[GraphEdge[str], float]:
    """Patch each edge's source with its counterfactual value and measure the metric."""
    sites = frozenset(SITES.values())

    with torch.no_grad(), instrument(model, observe=sites) as corrupted:
        corrupted(lambda: model.forward(counterfactual))

    with torch.no_grad():
        baseline = float(model.forward(clean))

    measured: dict[GraphEdge[str], float] = {}

    for edge in graph.edges:
        site = graph.site_of(edge.source)

        if site is None:
            pytest.fail(f"{edge!r} leaves an unpublished node")

        patch = {site: PatchFrom(corrupted.trace, site)}

        with torch.no_grad(), instrument(model, patch=patch) as patched:
            measured[edge] = float(patched(lambda: model.forward(clean))) - baseline

    return measured


def planted_edges(graph: ComputationGraph[str]) -> frozenset[GraphEdge[str]]:
    return frozenset(
        edge for edge in graph.edges if edge.source.key.reading.startswith(PLANTED)
    )


def an_attribution(
    seed: int, squash: Callable[[Tensor], Tensor] = torch.tanh
) -> tuple[_TwoPath, ComputationGraph[str], Tensor, Tensor, EdgeAttribution[str]]:
    model = _TwoPath(seed, squash=squash)
    graph = a_graph()
    clean, counterfactual = a_pair(seed)
    attributed = first_order_attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        dataset="two-path",
    )

    return model, graph, clean, counterfactual, attributed


GENERATED_BY: Final = Provenance("eap", "two-path", None)


def a_small_graph() -> ComputationGraph[str]:
    return ComputationGraph(
        (NODES["a1"], NODES["a2"], NODES[OUTPUT]),
        (
            GraphEdge(NODES["a1"], NODES["a2"], Role.RESIDUAL_STREAM),
            GraphEdge(NODES["a2"], NODES[OUTPUT], Role.RESIDUAL_STREAM),
        ),
        {name: SITES[name.key.reading] for name in (NODES["a1"], NODES["a2"])},
    )


def a_small_attribution(second: float = -0.5) -> EdgeAttribution[str]:
    graph = a_small_graph()
    first, last = sorted(graph.edges, key=repr)

    return EdgeAttribution(
        graph,
        {
            first: AttributionEvidence(1.5, GENERATED_BY),
            last: AttributionEvidence(second, GENERATED_BY),
        },
    )


class TestAnEdgeAttribution(ConformanceSuite):
    """Tests edge attributions against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[EdgeAttribution[str]]:
        """Return the conformance the laws run an attribution through."""
        return Conformance(
            build=a_small_attribution,
            unequal=a_small_attribution(0.25),
            operations=(
                ("graph", lambda attributed: len(attributed.graph.edges)),
                ("effects", lambda attributed: len(attributed.effects)),
                ("ranked", lambda attributed: len(attributed.ranked())),
                ("len", len),
                ("iter", lambda attributed: len(list(attributed))),
            ),
        )


def test_an_attribution_holds_one_effect_per_edge_of_its_graph() -> None:
    attributed = a_small_attribution()
    first, last = sorted(attributed.graph.edges, key=repr)

    assert len(attributed) == 2
    assert set(attributed) == attributed.graph.edges
    assert attributed[first].effect == 1.5
    assert attributed[last].effect == -0.5
    assert attributed.effects[last] == attributed[last]
    assert repr(attributed) == f"EdgeAttribution({attributed.graph!r}, edges=2)"


def test_an_attribution_ranks_its_edges_by_the_magnitude_of_their_effects() -> None:
    attributed = a_small_attribution(-3.0)
    first, last = sorted(attributed.graph.edges, key=repr)

    assert attributed.ranked() == (last, first)
    assert a_small_attribution(1.5).ranked() == (first, last)


def test_an_effect_for_every_edge_is_required() -> None:
    graph = a_small_graph()
    first, _ = sorted(graph.edges, key=repr)

    with pytest.raises(ValueError, match="has no effect"):
        EdgeAttribution(graph, {first: AttributionEvidence(1.0, GENERATED_BY)})


def test_an_effect_for_an_edge_the_graph_does_not_hold_is_refused() -> None:
    graph = a_small_graph()
    elsewhere = GraphEdge(NODES["b1"], NODES["b2"], Role.RESIDUAL_STREAM)
    effects = dict.fromkeys(
        (*graph.edges, elsewhere), AttributionEvidence(1.0, GENERATED_BY)
    )

    with pytest.raises(ValueError, match="not an edge here"):
        EdgeAttribution(graph, effects)


def test_only_attribution_evidence_stands_as_an_effect() -> None:
    graph = a_small_graph()
    effects = dict.fromkeys(graph.edges, AblationEvidence(1.0, "zero", GENERATED_BY))

    with pytest.raises(TypeError, match="attribution evidence"):
        EdgeAttribution(
            graph, cast("Mapping[GraphEdge[str], AttributionEvidence]", effects)
        )


def test_an_attribution_is_over_a_computation_graph() -> None:
    with pytest.raises(TypeError, match="computation graph"):
        EdgeAttribution(cast("ComputationGraph[str]", NODES["a1"]), {})


@pytest.mark.parametrize("seed", range(5))
def test_the_estimate_ranks_edges_as_brute_force_patching_does(seed: int) -> None:
    model, graph, clean, counterfactual, attributed = an_attribution(seed)
    measured = brute_force(model, graph, clean, counterfactual)
    edges = sorted(graph.edges, key=repr)

    correlation = rank_correlation(
        [abs(attributed[edge].effect) for edge in edges],
        [abs(measured[edge]) for edge in edges],
    )

    assert correlation > 0.9


@pytest.mark.parametrize("seed", range(5))
def test_the_planted_path_outranks_a_random_circuit_of_its_size(seed: int) -> None:
    _, graph, _, _, attributed = an_attribution(seed)
    edges = sorted(graph.edges, key=repr)
    planted = planted_edges(graph)
    draws = torch.Generator().manual_seed(seed)

    def weight(circuit: frozenset[GraphEdge[str]]) -> float:
        return sum(abs(attributed[edge].effect) for edge in circuit)

    compared = 0

    for _ in range(30):
        drawn = torch.randperm(len(edges), generator=draws)[: len(planted)]
        random = frozenset(edges[index] for index in drawn.tolist())

        if random == planted:
            continue

        compared += 1

        assert weight(planted) > weight(random)

    assert compared > 0
    assert attributed.ranked()[: len(planted)] == tuple(
        sorted(planted, key=lambda edge: (-abs(attributed[edge].effect), repr(edge)))
    )


def test_the_estimate_is_exact_where_the_paths_are_linear() -> None:
    model, graph, clean, counterfactual, attributed = an_attribution(
        2, squash=lambda stream: stream
    )
    measured = brute_force(model, graph, clean, counterfactual)

    for edge in graph.edges:
        assert attributed[edge].effect == pytest.approx(measured[edge], rel=1e-4)


def test_integrated_attribution_is_exact_where_the_paths_are_linear() -> None:
    model = _TwoPath(2, squash=lambda stream: stream)
    graph = a_graph()
    clean, counterfactual = a_pair(2)
    attributed = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=7,
        dataset="two-path",
    )
    measured = brute_force(model, graph, clean, counterfactual)

    for edge in graph.edges:
        assert attributed[edge].effect == pytest.approx(measured[edge], rel=1e-4)


def test_integrated_attribution_recovers_the_scalar_right_endpoint_sum() -> None:
    model = _ScalarCurve()
    graph = a_curve_graph()
    (edge,) = graph.edges
    integrated = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=torch.ones(1, 1),
        counterfactual=torch.zeros(1, 1),
        forward=model.forward,
        steps=4,
        dataset="scalar-curve",
    )

    # gradients 2x at x = 1/4, 1/2, 3/4, 1 average to 5/4; the
    # counterfactual-minus-clean source displacement is -1
    assert integrated[edge].effect == -1.25


def test_one_integration_step_is_the_clean_point_first_order_estimate() -> None:
    model = _TwoPath(3)
    graph = a_graph()
    clean, counterfactual = a_pair(3)
    integrated = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=1,
        dataset="two-path",
    )
    first_order = first_order_attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        dataset="two-path",
    )

    for edge in graph.edges:
        assert integrated[edge].effect == first_order[edge].effect

    assert {one.provenance.method for one in integrated.effects.values()} == {"eap-ig"}
    assert {one.provenance.method for one in first_order.effects.values()} == {"eap"}


def test_integrated_attribution_uses_every_right_endpoint_in_caller_order() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    seen: list[Tensor] = []

    def a_forward(given: Tensor) -> Tensor:
        seen.append(given.detach().clone())
        return model.forward(given)

    attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=a_forward,
        steps=4,
        dataset="two-path",
    )

    difference = clean - counterfactual
    expected = [
        counterfactual + difference * fraction for fraction in (0.25, 0.5, 0.75)
    ]
    expected.extend((clean, counterfactual))

    assert len(seen) == len(expected)

    for actual, endpoint in zip(seen, expected, strict=True):
        assert torch.equal(actual, endpoint)


def test_integrated_attribution_costs_steps_plus_one_forwards_and_steps_backwards() -> (
    None
):
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    backwards: list[int] = []

    def a_forward(given: Tensor) -> Tensor:
        metric = model.forward(given)

        if metric.requires_grad:
            metric.register_hook(lambda _gradient: backwards.append(1))

        return metric

    attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=a_forward,
        steps=5,
        dataset="two-path",
    )

    assert model.forwards == 6
    assert backwards == [1] * 5


def test_integrated_and_clean_point_attribution_separate_on_a_saturated_path() -> None:
    model = _TwoPath(1, squash=lambda stream: torch.tanh(4.0 * stream))
    graph = a_graph()
    clean, counterfactual = a_pair(1)
    integrated = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=32,
        dataset="two-path",
    )
    first_order = first_order_attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        dataset="two-path",
    )

    assert any(
        integrated[edge].effect != pytest.approx(first_order[edge].effect, rel=1e-3)
        for edge in graph.edges
    )


def test_integrated_attribution_honors_the_endpoint_masks() -> None:
    model = _TwoPath(4)
    graph = a_graph()
    clean, counterfactual = a_pair(4)
    kept = torch.ones(ITEMS, dtype=torch.bool)
    kept[0] = False
    masked = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=lambda given: model.forward(given, kept),
        steps=5,
        dataset="two-path",
    )
    without = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean[1:],
        counterfactual=counterfactual[1:],
        forward=model.forward,
        steps=5,
        dataset="two-path",
    )

    for edge in graph.edges:
        assert masked[edge].effect == pytest.approx(without[edge].effect, rel=1e-5)


@pytest.mark.parametrize("steps", [0, -1])
def test_an_integration_step_count_must_be_positive(steps: int) -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    with pytest.raises(ValueError, match="integration step count"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            steps=steps,
            dataset="two-path",
        )

    assert model.forwards == 0


@pytest.mark.parametrize("steps", [True, 1.5])
def test_an_integration_step_count_must_be_an_integer(steps: float) -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    with pytest.raises(TypeError, match=r"integration step count.*integer"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            steps=cast("int", steps),
            dataset="two-path",
        )

    assert model.forwards == 0


@pytest.mark.parametrize("endpoint", ["clean", "counterfactual"])
def test_an_attribution_endpoint_must_be_a_tensor(endpoint: str) -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    if endpoint == "clean":
        clean = cast("Tensor", [[1.0]])
    else:
        counterfactual = cast("Tensor", [[-1.0]])

    with pytest.raises(TypeError, match=rf"{endpoint} input is not a tensor"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            steps=2,
            dataset="two-path",
        )

    assert model.forwards == 0


@pytest.mark.parametrize("endpoint", ["clean", "counterfactual"])
def test_an_attribution_endpoint_must_be_floating_point(endpoint: str) -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    if endpoint == "clean":
        clean = clean.to(torch.int64)
    else:
        counterfactual = counterfactual.to(torch.int64)

    with pytest.raises(TypeError, match=rf"{endpoint} input is not floating point"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            steps=2,
            dataset="two-path",
        )

    assert model.forwards == 0


def test_attribution_endpoints_must_have_one_dtype() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    with pytest.raises(ValueError, match="counterfactual input has dtype"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual.to(torch.float64),
            forward=model.forward,
            steps=2,
            dataset="two-path",
        )

    assert model.forwards == 0


def test_integrated_attribution_preserves_inputs_and_repeats_deterministically() -> (
    None
):
    model = _TwoPath(0)
    graph = a_graph()
    clean, counterfactual = a_pair(0)
    original_clean = clean.clone()
    original_counterfactual = counterfactual.clone()
    first = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=4,
        dataset="two-path",
    )
    second = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=4,
        dataset="two-path",
    )

    assert torch.equal(clean, original_clean)
    assert torch.equal(counterfactual, original_counterfactual)
    assert first == second


def test_an_edgeless_graph_requires_no_model_call() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    graph = ComputationGraph((NODES[OUTPUT],), (), {})
    attributed = attribute_edges(
        model,
        graph,
        targets=source_targets(graph),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=4,
        dataset="two-path",
    )

    assert len(attributed) == 0
    assert model.forwards == 0


def test_a_nonfinite_integrated_effect_is_refused_by_site() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    def an_infinite_metric(given: Tensor) -> Tensor:
        return model.forward(given) * torch.tensor(float("inf"))

    with pytest.raises(ValueError, match=r"gradient at .* is not finite"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=an_infinite_metric,
            steps=2,
            dataset="two-path",
        )


def test_a_nonfinite_final_inner_product_is_refused_by_site() -> None:
    model = _ScalarCurve()

    with pytest.raises(ValueError, match=r"attributed effect at .* is not finite"):
        attribute_edges(
            model,
            a_curve_graph(),
            targets=source_targets(a_curve_graph()),
            clean=torch.full((1, 1), 1e154, dtype=torch.float64),
            counterfactual=torch.zeros(1, 1, dtype=torch.float64),
            forward=model.forward,
            steps=1,
            dataset="scalar-curve",
        )


def test_a_path_shape_change_is_refused_at_the_step_where_it_occurs() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    kept = torch.ones(ITEMS, dtype=torch.bool)
    kept[0] = False
    calls = 0

    def a_forward(given: Tensor) -> Tensor:
        nonlocal calls
        calls += 1
        return model.forward(given, kept if calls == 2 else None)

    with pytest.raises(ValueError, match=r"path step 2.*different positions"):
        attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=a_forward,
            steps=3,
            dataset="two-path",
        )

    assert model.forwards == 2


def test_the_estimate_costs_two_forward_passes_and_one_backward_pass() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    backwards: list[int] = []

    def a_clean_pass(given: Tensor) -> Tensor:
        metric = model.forward(given)

        if metric.requires_grad:
            metric.register_hook(lambda _gradient: backwards.append(1))

        return metric

    first_order_attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=a_clean_pass,
        dataset="two-path",
    )

    assert model.forwards == 2
    assert backwards == [1]


def test_every_effect_records_the_estimator_and_the_data_it_ran_over() -> None:
    _, _, _, _, attributed = an_attribution(1)

    assert {evidence.provenance for evidence in attributed.effects.values()} == {
        GENERATED_BY
    }
    assert all(
        evidence.kind == "attribution" for evidence in attributed.effects.values()
    )


def test_edges_leaving_one_node_share_that_nodes_estimate() -> None:
    _, graph, _, _, attributed = an_attribution(1)

    for edge in graph.edges:
        for other in graph.edges:
            if edge.source == other.source:
                assert attributed[edge].effect == attributed[other].effect


def test_the_estimate_reads_the_counterfactual_minus_the_clean_value() -> None:
    model = _TwoPath(3, squash=lambda stream: stream)
    clean, counterfactual = a_pair(3)
    forwards = first_order_attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        dataset="two-path",
    )
    backwards = first_order_attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=counterfactual,
        counterfactual=clean,
        forward=model.forward,
        dataset="two-path",
    )

    for edge in forwards:
        assert forwards[edge].effect == pytest.approx(-backwards[edge].effect)
        assert forwards[edge].effect != 0.0


def test_positions_a_mask_excludes_contribute_nothing_to_an_effect() -> None:
    model = _TwoPath(4)
    clean, counterfactual = a_pair(4)
    kept = torch.ones(ITEMS, dtype=torch.bool)
    kept[0] = False

    masked = first_order_attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=lambda given: model.forward(given, kept),
        dataset="two-path",
    )
    without = first_order_attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean[1:],
        counterfactual=counterfactual[1:],
        forward=model.forward,
        dataset="two-path",
    )
    everything = first_order_attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        dataset="two-path",
    )

    for edge in masked:
        assert masked[edge].effect == pytest.approx(without[edge].effect, rel=1e-5)
        assert masked[edge].effect != pytest.approx(everything[edge].effect)


def _without(
    targets: Mapping[GraphEdge[str], Unit], edge: GraphEdge[str]
) -> dict[GraphEdge[str], Unit]:
    return {key: unit for key, unit in targets.items() if key != edge}


def test_every_edge_needs_exactly_one_target_before_the_model_runs() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    graph = a_graph()
    targets = source_targets(graph)
    first = min(graph.edges, key=repr)
    elsewhere = GraphEdge(NODES[OUTPUT], NODES["a1"], Role.RESIDUAL_STREAM)
    refused: tuple[tuple[object, type[Exception], str], ...] = (
        (_without(targets, first), ValueError, "no target for"),
        ({**targets, elsewhere: Unit(SITES["a1"])}, ValueError, "not an edge here"),
        ({**targets, first: SITES["a1"]}, TypeError, "is not a Unit"),
        ([*targets.items()], TypeError, "not a mapping"),
        (
            {**targets, first: Unit(Site(MODEL, Role.RESIDUAL_STREAM, ("absent",)))},
            ValueError,
            "not published",
        ),
        ({**targets, first: Unit(SITES["a1"], {"span": 0})}, KeyError, "undeclared"),
    )

    for given, kind, message in refused:
        with pytest.raises(kind, match=message):
            first_order_attribute_edges(
                model,
                graph,
                targets=cast("Mapping[GraphEdge[str], Unit]", given),
                clean=clean,
                counterfactual=counterfactual,
                forward=model.forward,
                dataset="two-path",
            )

    assert model.forwards == 0


def test_an_attribution_validates_the_site_map_once_across_its_steps(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    calls: list[SiteMap] = []
    validate = _instrument.ensure_publishes

    def counting(published: SiteMap) -> SiteMap:
        calls.append(published)

        return validate(published)

    monkeypatch.setattr(_instrument, "ensure_publishes", counting)
    attribute_edges(
        model,
        a_graph(),
        targets=source_targets(a_graph()),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        steps=4,
        dataset="two-path",
    )

    assert calls == [PUBLISHED]


def test_a_nondifferentiable_target_is_refused_before_the_model_runs() -> None:
    frozen = dict(PUBLISHED)
    frozen[SITES["b2"]] = SiteSpec(
        AXES,
        DTypeClass.REAL,
        differentiable=False,
        access=Access.OBSERVE | Access.PATCH,
    )
    model = _TwoPath(0, published=SiteMap.custom(MODEL, frozen))
    clean, counterfactual = a_pair(0)

    with pytest.raises(ValueError, match="non-differentiable"):
        first_order_attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            dataset="two-path",
        )

    assert model.forwards == 0


def test_a_target_coordinate_beyond_its_axis_is_refused_by_name() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    graph = a_graph()
    first = min(graph.edges, key=repr)
    targets = {**source_targets(graph), first: Unit(SITES["a1"], {"width": WIDTH})}

    with pytest.raises(ValueError, match=r"indexes 4 on axis 'width'"):
        first_order_attribute_edges(
            model,
            graph,
            targets=targets,
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            dataset="two-path",
        )


def test_coordinate_targets_split_one_point_s_effect_among_its_edges() -> None:
    model = _TwoPath(1)
    clean, counterfactual = a_pair(1)
    graph = a_graph()
    targets = source_targets(graph)
    whole = first_order_attribute_edges(
        model,
        graph,
        targets=targets,
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        dataset="two-path",
    )
    leaving = sorted(
        (edge for edge in graph.edges if edge.source == NODES["a1"]), key=repr
    )
    parts: list[float] = []

    for offset in range(0, WIDTH, len(leaving)):
        coordinates = {
            edge: Unit(SITES["a1"], {"width": offset + position})
            for position, edge in enumerate(leaving)
        }
        split = first_order_attribute_edges(
            model,
            graph,
            targets={**targets, **coordinates},
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            dataset="two-path",
        )
        parts.extend(split[edge].effect for edge in leaving)

    assert len(set(parts)) == WIDTH
    assert sum(parts) == pytest.approx(whole[leaving[0]].effect)


def test_a_point_reached_twice_at_one_position_is_refused() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    graph = a_curve_graph()
    (edge,) = graph.edges

    def twice(given: Tensor) -> Tensor:
        first = model.at("a1", given, None)
        return (first + model.at("a1", given, None)).sum()

    with pytest.raises(RuntimeError, match="twice at one position"):
        first_order_attribute_edges(
            model,
            graph,
            targets={edge: Unit(SITES["a1"])},
            clean=clean,
            counterfactual=counterfactual,
            forward=twice,
            dataset="two-path",
        )


def test_each_gradient_pass_holds_its_readings_to_their_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)
    model = _ScalarCurve()
    model.published = SiteMap.custom(
        CURVE_MODEL,
        {
            CURVE_SITE: SiteSpec(
                AXES,
                DTypeClass.PROBABILITY,
                differentiable=True,
                access=Access.OBSERVE | Access.PATCH,
            )
        },
    )
    graph = a_curve_graph()

    with pytest.raises(ValueError, match=r"the reading at .*source"):
        first_order_attribute_edges(
            model,
            graph,
            targets=source_targets(graph),
            clean=torch.full((1, 1), 2.0),
            counterfactual=torch.zeros(1, 1),
            forward=model.forward,
            dataset="scalar-curve",
        )

    assert model.forwards == 1


def test_a_source_that_permits_no_patching_is_refused_before_the_model_runs() -> None:
    sealed = dict(PUBLISHED)
    sealed[SITES["b2"]] = SiteSpec(
        AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
    )
    model = _TwoPath(0, published=SiteMap.custom(MODEL, sealed))
    clean, counterfactual = a_pair(0)

    with pytest.raises(ValueError, match="cannot be named"):
        first_order_attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            dataset="two-path",
        )

    assert model.forwards == 0


def test_an_attribution_names_the_data_it_ran_over() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    with pytest.raises(ValueError, match="dataset"):
        first_order_attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            dataset="",
        )

    assert model.forwards == 0


def test_a_clean_pass_that_returns_no_scalar_is_refused() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)

    with pytest.raises(ValueError, match="scalar"):
        first_order_attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=lambda given: model.forward(given).expand(2),
            dataset="two-path",
        )


def test_a_counterfactual_of_another_shape_cannot_stand_in() -> None:
    model = _TwoPath(0)
    clean, _ = a_pair(0)
    _, counterfactual = a_pair(0, items=ITEMS + 1)

    with pytest.raises(ValueError, match="cannot stand in"):
        first_order_attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=model.forward,
            dataset="two-path",
        )


def test_a_counterfactual_holding_data_at_other_positions_cannot_stand_in() -> None:
    model = _TwoPath(0)
    clean, counterfactual = a_pair(0)
    kept = torch.ones(ITEMS, dtype=torch.bool)
    kept[0] = False

    with pytest.raises(ValueError, match="different positions"):
        first_order_attribute_edges(
            model,
            a_graph(),
            targets=source_targets(a_graph()),
            clean=clean,
            counterfactual=counterfactual,
            forward=lambda given: model.forward(
                given, kept if torch.equal(given, counterfactual) else None
            ),
            dataset="two-path",
        )


#: How far each receiver edge is nudged along its displacement for a central difference.
NUDGE: Final = 1e-5

#: The words of the receiver chart, enough for edges at three widths.
RECEIVER_WORDS: Final = 4


class _Nudge:
    """Adds a displacement to one receiver edge at the reach that consumes it."""

    def __init__(self, position: int, displacement: Tensor) -> None:
        self._position = position
        self._displacement = displacement

    def stands_in_for(self, reading: Reading, /) -> Tensor:
        mask = reading.mask

        if mask is None or not bool(mask[:, self._position].any()):
            return reading.tensor

        nudged = reading.tensor.clone()
        nudged[:, self._position] += self._displacement

        return nudged


def a_receiver_chart() -> tuple[
    DeepInsideOutside,
    ComputationGraph[str],
    Mapping[GraphEdge[str], Unit],
    Encoded,
    Encoded,
]:
    scorer = build_scorer(dtype=torch.float64)
    index = SpanIndex(RECEIVER_WORDS)
    component = scorer.receiver_site.component
    nodes = {
        span: GraphNode(component, ItemKey(index.span_of(span), "cell"))
        for span in range(index.num_spans)
    }
    targets = {
        GraphEdge(nodes[edge.source], nodes[edge.target], Role.RECEIVER_EDGE): Unit(
            scorer.receiver_site, edge.coordinates
        )
        for edge in scorer.receiver_edges(index)
    }
    graph = ComputationGraph(nodes.values(), targets)
    clean = scorer_leaves(1, RECEIVER_WORDS, seed=5, dtype=torch.float64)
    counterfactual = scorer_leaves(1, RECEIVER_WORDS, seed=17, dtype=torch.float64)

    return scorer, graph, targets, clean, counterfactual


def test_each_receiver_edge_is_attributed_along_its_own_displacement() -> None:
    scorer, graph, targets, clean, counterfactual = a_receiver_chart()
    site = scorer.receiver_site

    def log_partition(given: Tensor) -> Tensor:
        leaves = Encoded(given, SCORER_LEAF_AXES, clean.mask)

        log_z = scorer.readouts(leaves).log_z

        assert log_z is not None

        return log_z.tensor.sum()

    attributed = first_order_attribute_edges(
        scorer,
        graph,
        targets=targets,
        clean=clean.tensor,
        counterfactual=counterfactual.tensor,
        forward=log_partition,
        dataset="receiver-chart",
    )

    with torch.no_grad(), instrument(scorer, observe=frozenset({site})) as observed:
        observed(lambda: log_partition(clean.tensor))

    with torch.no_grad(), instrument(scorer, observe=frozenset({site})) as corrupted:
        corrupted(lambda: log_partition(counterfactual.tensor))

    displacement = corrupted.trace[site].tensor - observed.trace[site].tensor

    for edge, unit in targets.items():
        position = unit.index["edge"]
        nudged: list[float] = []

        for sign in (1.0, -1.0):
            nudge = _Nudge(position, sign * NUDGE * displacement[:, position])

            with (
                torch.no_grad(),
                scorer.instrument(frozenset(), patches={site: nudge}),
            ):
                nudged.append(float(log_partition(clean.tensor)))

        difference = (nudged[0] - nudged[1]) / (2 * NUDGE)

        assert attributed[edge].effect == pytest.approx(difference, rel=1e-4, abs=1e-9)

    assert len({attributed[edge].effect for edge in graph.edges}) == len(graph.edges)
