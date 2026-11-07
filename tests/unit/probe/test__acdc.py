from copy import copy, deepcopy
from math import isfinite
from typing import TYPE_CHECKING, Final, Self, cast

import pytest
import torch
from tests.unit.nn.parsers.test__scorer import LEAF_AXES as SCORER_LEAF_AXES
from tests.unit.nn.parsers.test__scorer import build as build_scorer
from tests.unit.nn.parsers.test__scorer import leaves as scorer_leaves

from lofi.deduce import ItemKey, SpanIndex
from lofi.nn._values import Encoded
from lofi.nn.ports import ReceiverEdge
from lofi.nn.sites import (
    Access,
    ComponentId,
    DTypeClass,
    ReceiverSide,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.probe import (
    AbsoluteThreshold,
    ACDCResult,
    AttributionEvidence,
    ComputationGraph,
    EdgeAttribution,
    GraphEdge,
    GraphNode,
    Provenance,
    PruningBudget,
    ReceiverCircuit,
    RelativeThreshold,
    Unit,
    acdc,
    attribute_edges,
    matched_cardinality_jaccard,
    receiver_circuit,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch, ReceiverLocalChartScorer
    from lofi.probe import PruningThreshold

MODEL: Final = ComponentId("receiver")
OTHER: Final = ComponentId("other")
TARGET: Final = Site(MODEL, Role.RECEIVER_EDGE, ("inputs",))
AXES: Final = ("edge",)


class _Held:
    def __init__(
        self,
        site: Site,
        tensor: Tensor,
        *,
        axes: tuple[str, ...] = AXES,
        mask: Tensor | None = None,
    ) -> None:
        self._site = site
        self._tensor = tensor
        self._axes = axes
        self._mask = mask

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

    def pass_through(self, site: Site, tensor: Tensor) -> Tensor:
        standing = self._patches.get(site)

        if standing is not None:
            tensor = standing.stands_in_for(_Held(site, tensor))

        if site in self._sites:
            mask = self._model.mask_for(tensor)
            self._readings[site] = _Held(
                site,
                tensor,
                axes=self._model.axes_for(tensor),
                mask=mask,
            )

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
    def __init__(
        self,
        *,
        access: Access = Access.OBSERVE | Access.PATCH,
        site: Site = TARGET,
        axes_for: Callable[[Tensor], tuple[str, ...]] | None = None,
        mask_for: Callable[[Tensor], Tensor | None] | None = None,
    ) -> None:
        self._site = site
        self._axes_for = axes_for or (lambda _tensor: AXES)
        self._mask_for = mask_for or (lambda _tensor: None)
        self._published = SiteMap.custom(
            MODEL,
            {
                site: SiteSpec(
                    AXES,
                    DTypeClass.REAL,
                    differentiable=True,
                    access=access,
                )
            },
        )
        self.scopes: list[_Scope] = []
        self.calls = 0
        self.received: list[Tensor] = []
        self.weight = torch.tensor(1.0)

    def axes_for(self, tensor: Tensor) -> tuple[str, ...]:
        return self._axes_for(tensor)

    def mask_for(self, tensor: Tensor) -> Tensor | None:
        return self._mask_for(tensor)

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
            value = scope.pass_through(self._site, value)

        self.received.append(value.detach().clone())

        return value.sum() * self.weight


def a_graph() -> tuple[
    ComputationGraph[str], GraphEdge[str], GraphEdge[str], dict[GraphEdge[str], Unit]
]:
    left = GraphNode(MODEL, ItemKey((0, 1), "left"))
    right = GraphNode(MODEL, ItemKey((1, 2), "right"))
    receiver = GraphNode(MODEL, ItemKey((0, 2), "receiver"))
    left_edge = GraphEdge(left, receiver, Role.RECEIVER_EDGE)
    right_edge = GraphEdge(right, receiver, Role.RECEIVER_EDGE)
    graph = ComputationGraph((left, right, receiver), (left_edge, right_edge))
    targets = {
        left_edge: Unit(TARGET, {"edge": 0}),
        right_edge: Unit(TARGET, {"edge": 1}),
    }

    return graph, left_edge, right_edge, targets


def provenances() -> tuple[Provenance, Provenance]:
    return (
        Provenance("endpoint-pair", "toy", 3),
        Provenance("acdc-policy", "toy", None),
    )


def squared(clean: Tensor, candidate: Tensor) -> Tensor:
    return (clean - candidate).square().sum()


def assign(value: object, name: str, replacement: object) -> None:
    setattr(value, name, replacement)


def run(
    model: _Receiver,
    *,
    budget: PruningBudget | None = None,
    threshold: PruningThreshold | None = None,
    order: tuple[GraphEdge[str], ...] | None = None,
    targets: Mapping[GraphEdge[str], Unit] | None = None,
    clean: Tensor | None = None,
    counterfactual: Tensor | None = None,
    divergence: Callable[[Tensor, Tensor], float | Tensor] = squared,
    forward: Callable[[Tensor], Tensor] | None = None,
) -> ACDCResult[str]:
    graph, left, right, declared = a_graph()
    data, policy = provenances()

    return acdc(
        model,
        graph,
        targets=declared if targets is None else targets,
        order=(right, left) if order is None else order,
        clean=torch.tensor([1.0, 1.0]) if clean is None else clean,
        counterfactual=(
            torch.tensor([0.0, 0.0]) if counterfactual is None else counterfactual
        ),
        forward=model.forward if forward is None else forward,
        divergence=divergence,
        threshold=AbsoluteThreshold(2.0) if threshold is None else threshold,
        budget=PruningBudget(2) if budget is None else budget,
        data_provenance=data,
        policy_provenance=policy,
    )


def empty_result() -> ACDCResult[str]:
    graph: ComputationGraph[str] = ComputationGraph((), ())
    data, policy = provenances()

    return ACDCResult(
        graph,
        tested=(),
        untested=(),
        retained=(),
        pruned=(),
        divergence=0.0,
        threshold=AbsoluteThreshold(0.0),
        cutoff=0.0,
        wall_seconds=0.0,
        budget=PruningBudget(0),
        data_provenance=data,
        policy_provenance=policy,
    )


def test_a_pruning_budget_is_required_validated_and_immutable() -> None:
    budget = PruningBudget(2)

    assert budget.maximum_trials == 2
    assert budget == PruningBudget(2)
    assert hash(budget) == hash(PruningBudget(2))
    assert repr(budget) == "PruningBudget(2)"
    assert copy(budget) is budget
    assert deepcopy(budget) is budget

    with pytest.raises(AttributeError, match="pruning budget"):
        assign(budget, "maximum_trials", 3)

    with pytest.raises(TypeError, match="maximum trials"):
        PruningBudget(maximum_trials=True)

    with pytest.raises(ValueError, match="maximum trials"):
        PruningBudget(-1)


def test_acdc_patches_one_receiver_edge_then_accumulates_accepted_removals() -> None:
    model = _Receiver()
    graph, left, right, _ = a_graph()
    clean = torch.tensor([1.0, 1.0])
    counterfactual = torch.tensor([0.0, 0.0])
    clean_before = clean.clone()
    counterfactual_before = counterfactual.clone()
    weight_before = model.weight.clone()
    data, policy = provenances()

    found = acdc(
        model,
        graph,
        targets=a_graph()[3],
        order=(right, left),
        clean=clean,
        counterfactual=counterfactual,
        forward=model.forward,
        divergence=squared,
        threshold=AbsoluteThreshold(2.0),
        budget=PruningBudget(2),
        data_provenance=data,
        policy_provenance=policy,
    )

    assert found.tested == (right, left)
    assert found.untested == ()
    assert found.pruned == (right,)
    assert found.retained == (left,)
    assert found.divergence == 1.0
    assert found.threshold == AbsoluteThreshold(2.0)
    assert found.cutoff == 2.0
    assert found.scheduling_residual == 0.0
    assert found.budget == PruningBudget(2)
    assert found.data_provenance is data
    assert found.policy_provenance is policy
    assert isfinite(found.wall_seconds)
    assert found.wall_seconds >= 0.0
    assert model.calls == 4
    assert tuple(tuple(value.tolist()) for value in model.received) == (
        (1.0, 1.0),
        (0.0, 0.0),
        (1.0, 0.0),
        (0.0, 0.0),
    )
    assert torch.equal(clean, clean_before)
    assert torch.equal(counterfactual, counterfactual_before)
    assert torch.equal(model.weight, weight_before)
    assert model.scopes == []


def test_a_threshold_tie_is_retained_and_declared_order_breaks_no_ties() -> None:
    model = _Receiver()
    graph, left, right, _ = a_graph()

    found = run(model, threshold=AbsoluteThreshold(1.0))

    assert found.tested == (right, left)
    assert found.pruned == ()
    assert found.retained == (right, left)
    assert found.divergence == 0.0
    assert found.graph == graph


def test_pruning_thresholds_are_validated_immutable_values() -> None:
    absolute = AbsoluteThreshold(0.5)
    relative = RelativeThreshold(0.25)

    assert absolute.bound == 0.5
    assert relative.fraction == 0.25
    assert absolute == AbsoluteThreshold(0.5)
    assert relative == RelativeThreshold(0.25)
    assert hash(relative) == hash(RelativeThreshold(0.25))
    assert absolute != RelativeThreshold(0.5)
    assert repr(absolute) == "AbsoluteThreshold(0.5)"
    assert repr(relative) == "RelativeThreshold(0.25)"
    assert copy(relative) is relative
    assert deepcopy(absolute) is absolute
    assert absolute.cutoff(4.0) == 0.5
    assert relative.cutoff(4.0) == 1.0
    assert absolute.to_spec() == {"kind": "absolute", "bound": 0.5}
    assert relative.to_spec() == {"kind": "relative", "fraction": 0.25}

    with pytest.raises(AttributeError, match="pruning threshold"):
        assign(relative, "fraction", 0.5)

    with pytest.raises(TypeError, match="fraction"):
        RelativeThreshold(fraction=True)

    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ValueError, match="bound"):
            AbsoluteThreshold(invalid)

        with pytest.raises(ValueError, match="fraction"):
            RelativeThreshold(invalid)


def test_a_relative_threshold_scales_with_the_full_corruption_divergence() -> None:
    _, left, right, _ = a_graph()

    found = run(_Receiver(), threshold=RelativeThreshold(0.5))

    assert found.pruned == (right,)
    assert found.retained == (left,)
    assert found.threshold == RelativeThreshold(0.5)
    assert found.cutoff == 2.0


@pytest.mark.parametrize("scale", [1.0, 10.0])
def test_relative_pruning_is_invariant_to_the_divergence_scale(scale: float) -> None:
    _, left, right, _ = a_graph()

    def divergence(clean: Tensor, candidate: Tensor) -> Tensor:
        return scale * squared(clean, candidate)

    found = run(_Receiver(), threshold=RelativeThreshold(0.5), divergence=divergence)

    assert found.pruned == (right,)
    assert found.retained == (left,)
    assert found.cutoff == 2.0 * scale


def test_an_absolute_threshold_ignores_the_full_corruption_divergence() -> None:
    _, left, right, _ = a_graph()

    def divergence(clean: Tensor, candidate: Tensor) -> Tensor:
        return 10.0 * squared(clean, candidate)

    found = run(_Receiver(), threshold=AbsoluteThreshold(2.0), divergence=divergence)

    assert found.pruned == ()
    assert found.retained == (right, left)
    assert found.cutoff == 2.0


def test_zero_full_corruption_retains_every_edge_under_a_relative_threshold() -> None:
    _, left, right, _ = a_graph()

    found = run(
        _Receiver(),
        threshold=RelativeThreshold(0.5),
        counterfactual=torch.ones(2),
    )

    assert found.pruned == ()
    assert found.retained == (right, left)
    assert found.cutoff == 0.0


def test_an_acdc_result_requires_a_declared_threshold_and_a_valid_cutoff() -> None:
    graph: ComputationGraph[str] = ComputationGraph((), ())
    data, policy = provenances()

    def built(threshold: object, cutoff: float) -> ACDCResult[str]:
        return ACDCResult(
            graph,
            tested=(),
            untested=(),
            retained=(),
            pruned=(),
            divergence=0.0,
            threshold=cast("PruningThreshold", threshold),
            cutoff=cutoff,
            wall_seconds=0.0,
            budget=PruningBudget(0),
            data_provenance=data,
            policy_provenance=policy,
        )

    with pytest.raises(TypeError, match="pruning threshold"):
        built(0.5, 0.5)

    with pytest.raises(ValueError, match="cutoff"):
        built(RelativeThreshold(0.5), -1.0)


def test_budget_exhaustion_retains_untested_edges_and_reports_scheduling_residual() -> (
    None
):
    model = _Receiver()
    _, left, right, _ = a_graph()

    found = run(model, budget=PruningBudget(1))

    assert found.tested == (right,)
    assert found.untested == (left,)
    assert found.pruned == (right,)
    assert found.retained == (left,)
    assert found.scheduling_residual == 0.5
    assert model.calls == 3


def test_a_zero_trial_budget_still_fixes_both_nonempty_endpoints() -> None:
    model = _Receiver()
    _, left, right, _ = a_graph()

    found = run(model, budget=PruningBudget(0))

    assert found.tested == ()
    assert found.untested == (right, left)
    assert found.retained == (right, left)
    assert found.pruned == ()
    assert found.scheduling_residual == 1.0
    assert model.calls == 2


def test_an_edgeless_graph_runs_nothing_and_has_zero_residual() -> None:
    class _Untouched:
        def site_map(self) -> SiteMap:
            pytest.fail("an edgeless graph touched the model")

        def instrument(
            self,
            sites: frozenset[Site],
            /,
            *,
            patches: Mapping[Site, Patch[_Held]],
        ) -> _Scope:
            del sites, patches
            pytest.fail("an edgeless graph instrumented the model")

    graph: ComputationGraph[str] = ComputationGraph((), ())
    data, policy = provenances()
    calls = 0

    def forward(given: Tensor) -> Tensor:
        nonlocal calls
        calls += 1

        return given.sum()

    found = acdc(
        _Untouched(),
        graph,
        targets={},
        order=(),
        clean=torch.tensor([1.0]),
        counterfactual=torch.tensor([0.0]),
        forward=forward,
        divergence=squared,
        threshold=AbsoluteThreshold(0.0),
        budget=PruningBudget(0),
        data_provenance=data,
        policy_provenance=policy,
    )

    assert found.tested == found.untested == found.retained == found.pruned == ()
    assert found.divergence == 0.0
    assert found.scheduling_residual == 0.0
    assert found.wall_seconds == 0.0
    assert calls == 0


def test_a_total_reverse_topological_order_is_required() -> None:
    first = GraphNode(MODEL, ItemKey((0, 1), "first"))
    middle = GraphNode(MODEL, ItemKey((0, 2), "middle"))
    last = GraphNode(MODEL, ItemKey((0, 3), "last"))
    upstream = GraphEdge(first, middle, Role.RECEIVER_EDGE)
    downstream = GraphEdge(middle, last, Role.RECEIVER_EDGE)
    graph = ComputationGraph((first, middle, last), (upstream, downstream))
    data, policy = provenances()

    with pytest.raises(ValueError, match="reverse-topological"):
        acdc(
            _Receiver(),
            graph,
            targets={},
            order=(upstream, downstream),
            clean=torch.tensor([1.0]),
            counterfactual=torch.tensor([0.0]),
            forward=lambda value: value,
            divergence=squared,
            threshold=AbsoluteThreshold(0.0),
            budget=PruningBudget(0),
            data_provenance=data,
            policy_provenance=policy,
        )


@pytest.mark.parametrize("kind", ["missing", "repeated", "foreign"])
def test_the_edge_order_must_cover_the_graph_exactly_once(kind: str) -> None:
    model = _Receiver()
    _, _, right, targets = a_graph()
    foreign_node = GraphNode(MODEL, ItemKey((2, 3), "foreign"))
    foreign = GraphEdge(foreign_node, right.target, Role.RECEIVER_EDGE)
    order = {
        "missing": (right,),
        "repeated": (right, right),
        "foreign": (right, foreign),
    }[kind]

    with pytest.raises(ValueError, match="edge order"):
        run(model, order=order, targets=targets)


def test_targets_must_cover_exactly_the_graph_edges() -> None:
    model = _Receiver()
    graph, left, right, targets = a_graph()
    foreign_node = GraphNode(MODEL, ItemKey((2, 3), "foreign"))
    foreign = GraphEdge(foreign_node, right.target, Role.RECEIVER_EDGE)

    with pytest.raises(ValueError, match="no receiver-local target"):
        run(model, targets={right: targets[right]})

    with pytest.raises(ValueError, match="not an edge"):
        run(model, targets={**targets, foreign: targets[left]})

    assert graph.edges == frozenset((left, right))


def test_targets_must_be_receiver_local_units_at_the_receiver_component() -> None:
    model = _Receiver()
    _, left, right, targets = a_graph()
    malformed = cast(
        "Mapping[GraphEdge[str], Unit]",
        {left: object(), right: targets[right]},
    )

    with pytest.raises(TypeError, match="not a Unit"):
        run(model, targets=malformed)

    wrong_role = Unit(Site(MODEL, Role.CHART_CELL), {"edge": 0})

    with pytest.raises(ValueError, match="RECEIVER_EDGE"):
        run(model, targets={left: wrong_role, right: targets[right]})

    wrong_component = Unit(Site(OTHER, Role.RECEIVER_EDGE), {"edge": 0})

    with pytest.raises(ValueError, match="receiver"):
        run(model, targets={left: wrong_component, right: targets[right]})


def test_edge_targets_must_not_overlap() -> None:
    model = _Receiver()
    _, left, right, _ = a_graph()

    with pytest.raises(ValueError, match="overlap"):
        run(
            model,
            targets={
                left: Unit(TARGET, {"edge": 0}),
                right: Unit(TARGET),
            },
        )


@pytest.mark.parametrize("access", [Access.OBSERVE, Access.PATCH])
def test_every_target_must_permit_observation_and_patching(access: Access) -> None:
    model = _Receiver(access=access)

    with pytest.raises(ValueError, match="observation and patching"):
        run(model)

    assert model.calls == 0


def test_target_axes_and_coordinates_are_validated_by_name() -> None:
    model = _Receiver()
    _, left, right, targets = a_graph()

    with pytest.raises(KeyError, match="undeclared axis"):
        run(
            model,
            targets={left: Unit(TARGET, {"row": 0}), right: targets[right]},
        )

    with pytest.raises(ValueError, match="has 2 positions"):
        run(
            model,
            budget=PruningBudget(1),
            targets={left: Unit(TARGET, {"edge": 2}), right: targets[right]},
        )

    assert model.calls == 2


@pytest.mark.parametrize(
    ("clean", "counterfactual", "error", "message"),
    [
        (object(), torch.tensor([0.0, 0.0]), TypeError, "clean input"),
        (
            torch.tensor([1.0, float("nan")]),
            torch.tensor([0.0, 0.0]),
            ValueError,
            "clean input is not finite",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([0.0]),
            ValueError,
            "shape",
        ),
        (
            torch.tensor([1.0, 1.0]),
            torch.tensor([0, 0]),
            ValueError,
            "dtype",
        ),
    ],
)
def test_endpoint_inputs_are_validated_before_execution(
    clean: object,
    counterfactual: Tensor,
    error: type[Exception],
    message: str,
) -> None:
    model = _Receiver()

    with pytest.raises(error, match=message):
        run(model, clean=cast("Tensor", clean), counterfactual=counterfactual)

    assert model.calls == 0


@pytest.mark.parametrize(
    ("divergence", "error", "message"),
    [
        (lambda _clean, _trial: torch.ones(2), ValueError, "not scalar"),
        (
            lambda _clean, _trial: torch.tensor(1),
            TypeError,
            "not floating point",
        ),
        (lambda _clean, _trial: float("nan"), ValueError, "not finite"),
        (lambda _clean, _trial: -1.0, ValueError, "nonnegative"),
        (lambda _clean, _trial: object(), TypeError, "not a scalar number"),
    ],
)
def test_divergence_must_be_a_finite_nonnegative_scalar(
    divergence: Callable[[Tensor, Tensor], object],
    error: type[Exception],
    message: str,
) -> None:
    model = _Receiver()

    with pytest.raises(error, match=message):
        run(
            model,
            divergence=cast("Callable[[Tensor, Tensor], float | Tensor]", divergence),
        )


def test_outputs_must_be_corresponding_finite_floating_tensors() -> None:
    model = _Receiver()
    no_tensor = cast("Callable[[Tensor], Tensor]", lambda _value: 1.0)

    with pytest.raises(TypeError, match="returned no tensor"):
        run(model, forward=no_tensor)

    model = _Receiver()

    with pytest.raises(TypeError, match="not floating point"):
        run(model, forward=lambda _value: torch.tensor(1))

    model = _Receiver()

    with pytest.raises(ValueError, match="output is not finite"):
        run(model, forward=lambda _value: torch.tensor(float("inf")))

    model = _Receiver()

    with pytest.raises(ValueError, match="dtype"):
        run(
            model,
            forward=lambda value: model.forward(value).to(
                torch.float64 if bool(value.sum()) else torch.float32
            ),
        )


def test_endpoint_readings_must_have_corresponding_axes_and_masks() -> None:
    axes_model = _Receiver(
        axes_for=lambda value: ("edge",) if bool(value.sum()) else ("slot",)
    )

    with pytest.raises(ValueError, match="axes"):
        run(axes_model)

    mask_model = _Receiver(
        mask_for=lambda value: torch.tensor([True, bool(value.sum())])
    )

    with pytest.raises(ValueError, match="different positions"):
        run(mask_model)


def test_repeated_calls_make_the_same_decisions() -> None:
    first = run(_Receiver())
    second = run(_Receiver())

    assert first.tested == second.tested
    assert first.untested == second.untested
    assert first.retained == second.retained
    assert first.pruned == second.pruned
    assert first.divergence == second.divergence
    assert first.scheduling_residual == second.scheduling_residual


def test_an_acdc_result_validates_its_immutable_partitions() -> None:
    model = _Receiver()
    found = run(model)

    assert copy(found) is found
    assert deepcopy(found) is found
    assert "tested=2" in repr(found)

    with pytest.raises(AttributeError, match="ACDC result"):
        assign(found, "divergence", 3.0)

    graph, left, right, _ = a_graph()
    data, policy = provenances()

    with pytest.raises(ValueError, match="partition"):
        ACDCResult(
            graph,
            tested=(right,),
            untested=(),
            retained=(left,),
            pruned=(right,),
            divergence=1.0,
            threshold=AbsoluteThreshold(2.0),
            cutoff=2.0,
            wall_seconds=0.0,
            budget=PruningBudget(1),
            data_provenance=data,
            policy_provenance=policy,
        )


def test_matched_cardinality_jaccard_uses_the_eap_ig_ranking() -> None:
    graph, left, right, _ = a_graph()
    data, _ = provenances()
    found = run(_Receiver())
    attribution = EdgeAttribution(
        graph,
        {
            left: AttributionEvidence(0.25, data),
            right: AttributionEvidence(0.75, data),
        },
    )

    assert found.retained == (left,)
    assert matched_cardinality_jaccard(found, attribution) == 0.0


def test_matched_cardinality_jaccard_refuses_a_graph_mismatch() -> None:
    graph, left, right, _ = a_graph()
    data, _ = provenances()
    changed = ComputationGraph(graph.nodes, (left,))
    attribution = EdgeAttribution(
        changed,
        {left: AttributionEvidence(1.0, data)},
    )

    with pytest.raises(ValueError, match="same graph"):
        matched_cardinality_jaccard(run(_Receiver()), attribution)

    assert right in graph.edges


def test_two_empty_circuits_have_matched_cardinality_agreement_one() -> None:
    found = empty_result()
    attribution: EdgeAttribution[str] = EdgeAttribution(found.graph, {})

    assert matched_cardinality_jaccard(found, attribution) == 1.0


@pytest.mark.parametrize("words", [2, 3, 4, 5, 6])
def test_receiver_edges_and_graph_edges_correspond_one_to_one(words: int) -> None:
    scorer = build_scorer()
    index = SpanIndex(words)
    edges = scorer.receiver_edges(index)
    circuit = receiver_circuit(scorer, index, reading="cell")
    units = tuple(circuit.targets.values())

    assert type(circuit) is ReceiverCircuit
    assert len(circuit.graph.edges) == len(edges) == len(circuit.targets)
    assert len(set(units)) == len(units)
    assert {unit.index["edge"] for unit in units} == set(range(len(edges)))
    assert {unit.site for unit in units} == {scorer.receiver_site}
    assert {node.span for node in circuit.graph.nodes} == {
        index.span_of(span) for span in range(index.num_spans)
    }

    for edge, unit in circuit.targets.items():
        receiver = edges[unit.index["edge"]]

        assert edge.source.span == index.span_of(receiver.source)
        assert edge.target.span == index.span_of(receiver.target)
        assert edge.role is Role.RECEIVER_EDGE
        assert edge.source.component == scorer.receiver_site.component


def _width(span: tuple[int, int]) -> int:
    return span[1] - span[0]


@pytest.mark.parametrize("words", [3, 5])
def test_receiver_order_runs_widest_receivers_first_then_by_edge_position(
    words: int,
) -> None:
    scorer = build_scorer()
    circuit = receiver_circuit(scorer, SpanIndex(words), reading="cell")
    keys = tuple(
        (-_width(edge.target.span), circuit.targets[edge].index["edge"])
        for edge in circuit.order
    )

    assert keys == tuple(sorted(keys))
    assert set(circuit.order) == circuit.graph.edges

    if words == 3:
        assert tuple(circuit.targets[edge].index["edge"] for edge in circuit.order) == (
            4,
            5,
            6,
            7,
            0,
            1,
            2,
            3,
        )


class _Declared:
    def __init__(self, edges: tuple[ReceiverEdge, ...]) -> None:
        self._edges = edges

    @property
    def receiver_site(self) -> Site:
        return TARGET

    def receiver_edges(self, index: SpanIndex, /) -> tuple[ReceiverEdge, ...]:
        del index
        return self._edges


def test_a_receiver_mapping_that_is_not_one_to_one_is_refused() -> None:
    index = SpanIndex(2)
    root = index.span_id(0, 2)
    left, right = index.span_id(0, 1), index.span_id(1, 2)
    twice = (
        ReceiverEdge(0, left, root, 0, ReceiverSide.LEFT),
        ReceiverEdge(1, left, root, 0, ReceiverSide.RIGHT),
    )
    shuffled = (
        ReceiverEdge(1, left, root, 0, ReceiverSide.LEFT),
        ReceiverEdge(0, right, root, 0, ReceiverSide.RIGHT),
    )

    for edges, message in ((twice, "one-to-one"), (shuffled, "axis order")):
        declared = cast("ReceiverLocalChartScorer", _Declared(edges))

        with pytest.raises(ValueError, match=message):
            receiver_circuit(declared, index, reading="cell")


def test_both_estimators_run_receiver_locally_on_one_chart_scorer() -> None:
    scorer = build_scorer(dtype=torch.float64)
    words = 4
    circuit = receiver_circuit(scorer, SpanIndex(words), reading="cell")
    clean = scorer_leaves(1, words, seed=5, dtype=torch.float64)
    counterfactual = scorer_leaves(1, words, seed=17, dtype=torch.float64)

    def log_partition(given: Tensor) -> Tensor:
        log_z = scorer.readouts(Encoded(given, SCORER_LEAF_AXES, clean.mask)).log_z

        assert log_z is not None

        return log_z.tensor

    attribution = attribute_edges(
        scorer,
        circuit.graph,
        targets=circuit.targets,
        clean=clean.tensor,
        counterfactual=counterfactual.tensor,
        forward=lambda given: log_partition(given).sum(),
        steps=3,
        dataset="receiver-chart",
    )
    result = acdc(
        scorer,
        circuit.graph,
        targets=circuit.targets,
        order=circuit.order,
        clean=clean.tensor,
        counterfactual=counterfactual.tensor,
        forward=log_partition,
        divergence=lambda reference, trial: (reference - trial).abs().mean(),
        threshold=AbsoluteThreshold(0.0),
        budget=PruningBudget(len(circuit.order)),
        data_provenance=Provenance("receiver chart", "seeds 5 and 17", None),
        policy_provenance=Provenance("budgeted-acdc", "receiver order", None),
    )

    assert result.tested == circuit.order
    assert result.scheduling_residual == 0.0
    assert len({attribution[edge].effect for edge in circuit.graph.edges}) == len(
        circuit.graph.edges
    )
    assert 0.0 <= matched_cardinality_jaccard(result, attribution) <= 1.0
