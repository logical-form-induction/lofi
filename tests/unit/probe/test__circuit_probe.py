from contextlib import nullcontext
from typing import TYPE_CHECKING, Final, Self

import didactic.api as dx
import pytest
import torch

from lofi.deduce import ItemKey
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._acdc import AbsoluteThreshold, RelativeThreshold
from lofi.probe._circuit_probe import PLUGIN, CircuitProbe, CircuitProbeSpec
from lofi.probe._circuits import Unit
from lofi.probe._graph import ComputationGraph, GraphEdge, GraphNode
from lofi.probe._probing import (
    GradientMode,
    PassLayout,
    ProbeBuildContext,
    UnpublishedSiteError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from contextlib import AbstractContextManager
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch

MODEL: Final = ComponentId("circuit-model")
SITE: Final = Site(MODEL, Role.RECEIVER_EDGE, ("input",))
AXES: Final = ("item", "feature")
PUBLISHED: Final[SiteMap] = SiteMap.custom(
    MODEL,
    {
        SITE: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        )
    },
)
CLEAN: Final = torch.tensor(
    [
        [1.0, 0.5],
        [2.0, -0.5],
        [0.5, 3.0],
        [4.0, 1.5],
        [-1.0, 2.0],
        [3.0, -2.0],
    ]
)
COUNTERFACTUAL: Final = torch.tensor(
    [
        [-3.0, 2.0],
        [0.0, -4.0],
        [2.5, -1.0],
        [-2.0, 0.5],
        [4.0, -3.0],
        [-0.5, 1.0],
    ]
)


class _Reading:
    __slots__ = ("_tensor",)

    def __init__(self, tensor: Tensor) -> None:
        self._tensor = tensor

    @property
    def site(self) -> Site:
        return SITE

    @property
    def tensor(self) -> Tensor:
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        return AXES

    @property
    def mask(self) -> None:
        return None


class _Scope:
    __slots__ = ("_model", "_patches", "_readings", "_sites")

    def __init__(
        self,
        model: _CircuitModel,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[_Reading]],
    ) -> None:
        self._model = model
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Reading] = {}

    @property
    def readings(self) -> Mapping[Site, _Reading]:
        return self._readings

    def pass_through(self, tensor: Tensor) -> Tensor:
        standing = self._patches.get(SITE)
        if standing is not None:
            tensor = standing.stands_in_for(_Reading(tensor))
        if SITE in self._sites:
            self._readings[SITE] = _Reading(tensor)
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


class _CircuitModel:
    __slots__ = ("scopes",)

    def __init__(self) -> None:
        self.scopes: list[_Scope] = []

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Reading]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def forward(self, given: Tensor) -> Tensor:
        value = given
        for scope in self.scopes:
            value = scope.pass_through(value)
        return value[:, 0].square() + 0.5 * value[:, 1].square() + value.prod(dim=1)


def _graph() -> tuple[
    ComputationGraph[object],
    tuple[GraphEdge[object], GraphEdge[object]],
    Mapping[GraphEdge[object], Unit],
]:
    left = GraphNode[object](MODEL, ItemKey((0, 1), "left"))
    right = GraphNode[object](MODEL, ItemKey((1, 2), "right"))
    receiver = GraphNode[object](MODEL, ItemKey((0, 2), "receiver"))
    left_edge = GraphEdge(left, receiver, Role.RECEIVER_EDGE)
    right_edge = GraphEdge(right, receiver, Role.RECEIVER_EDGE)
    graph = ComputationGraph(
        (left, right, receiver),
        (left_edge, right_edge),
        sites={left: SITE, right: SITE},
    )
    targets = {
        left_edge: Unit(SITE, {"feature": 0}),
        right_edge: Unit(SITE, {"feature": 1}),
    }
    return graph, (right_edge, left_edge), targets


class _CircuitPass:
    __slots__ = ("_model", "gradients")

    def __init__(self, model: _CircuitModel) -> None:
        self._model = model
        self.gradients: list[GradientMode] = []

    def __call__(self, gradients: GradientMode, /) -> PassLayout:
        self.gradients.append(gradients)
        context: AbstractContextManager[object] = (
            nullcontext() if gradients is GradientMode.ENABLED else torch.no_grad()
        )
        with context:
            self._model.forward(CLEAN)
        words = tuple((f"item-{index}",) for index in range(len(CLEAN)))
        return PassLayout(words, None, tuple(None for _word in words))

    def computation_graph(self) -> ComputationGraph[object]:
        return _graph()[0]

    def circuit_component(self) -> str:
        return str(MODEL)

    def circuit_inputs(self) -> tuple[Tensor, Tensor]:
        return CLEAN, COUNTERFACTUAL

    def circuit_forward(self, endpoint: Tensor, /) -> Tensor:
        return self._model.forward(endpoint)

    def circuit_targets(self) -> Mapping[GraphEdge[object], Unit]:
        return _graph()[2]

    def circuit_order(self) -> tuple[GraphEdge[object], ...]:
        return _graph()[1]


class _Point:
    __slots__ = ("_model", "_pass")

    def __init__(self) -> None:
        self._model = _CircuitModel()
        self._pass = _CircuitPass(self._model)

    @property
    def components(self) -> Mapping[str, _CircuitModel]:
        return {str(MODEL): self._model}

    @property
    def passes(self) -> tuple[_CircuitPass, ...]:
        return (self._pass,)


def _declaration(
    threshold: dict[str, str | float] | None = None,
) -> CircuitProbeSpec:
    return CircuitProbeSpec.model_validate(
        {
            "kind": "circuit",
            "acdc_trials": 2,
            "attribution_steps": 2,
            "acdc_threshold": (
                {"kind": "absolute", "bound": 0.0} if threshold is None else threshold
            ),
            "null_replicates": 3,
            "completeness_draws": 1,
            "bootstrap_replicates": 3,
            "seed": 3,
            "fdr_alpha": 0.1,
            "dependence": "prds",
        }
    )


def test_a_circuit_declaration_requires_an_explicit_acdc_budget() -> None:
    with pytest.raises(dx.ValidationError, match="acdc_trials"):
        CircuitProbeSpec.model_validate({"kind": "circuit"})


@pytest.mark.parametrize(
    ("declared", "threshold"),
    [
        ({"kind": "absolute", "bound": 0.25}, AbsoluteThreshold(0.25)),
        ({"kind": "relative", "fraction": 0.01}, RelativeThreshold(0.01)),
    ],
)
def test_a_circuit_declaration_names_the_kind_of_its_acdc_threshold(
    declared: dict[str, str | float], threshold: AbsoluteThreshold | RelativeThreshold
) -> None:
    assert _declaration(declared).settings().acdc_threshold == threshold


@pytest.mark.parametrize(
    "declared",
    [
        0.01,
        {"kind": "relative", "fraction": -0.5},
        {"kind": "absolute", "bound": float("inf")},
        {"kind": "relative", "bound": 0.01},
        {"kind": "proportional", "fraction": 0.01},
    ],
)
def test_an_undeclared_or_invalid_acdc_threshold_is_refused(
    declared: float | dict[str, str | float],
) -> None:
    with pytest.raises(dx.ValidationError):
        CircuitProbeSpec.model_validate(
            {"kind": "circuit", "acdc_trials": 2, "acdc_threshold": declared}
        )


def test_the_circuit_plugin_validates_its_pass_capability_at_the_build_boundary() -> (
    None
):
    class _GraphOnly:
        def __call__(self, gradients: GradientMode, /) -> PassLayout:
            del gradients
            return PassLayout((), None, ())

        def computation_graph(self) -> ComputationGraph[object]:
            return ComputationGraph((), ())

    class _GraphPoint:
        @property
        def components(self) -> Mapping[str, _CircuitModel]:
            return {}

        @property
        def passes(self) -> tuple[_GraphOnly, ...]:
            return (_GraphOnly(),)

    with pytest.raises(TypeError, match="circuit endpoint"):
        PLUGIN.construct(_declaration(), ProbeBuildContext("circuits", _GraphPoint()))


def test_the_circuit_probe_executes_every_declared_method_and_inferential_field() -> (
    None
):
    point = _Point()
    built = PLUGIN.construct(_declaration(), ProbeBuildContext("circuits", point))
    built.admit(PUBLISHED)

    rows = built.run(point)
    by_measure = {row.measure: row for row in rows}

    assert isinstance(built, CircuitProbe)
    assert point.passes[0].gradients == [GradientMode.ENABLED]
    assert {
        "eap-ig/mean-absolute-effect",
        "acdc/retained-share",
        "acdc/eap-ig-jaccard",
        "faithfulness/sufficiency",
        "faithfulness/incompleteness",
        "faithfulness/minimum-contribution",
        "circuitry/preservation",
        "circuitry/localization",
        "circuitry/minimality",
        "sensitivity/resample/logit-difference",
        "sensitivity/resample/probability",
        "sensitivity/mean/logit-difference",
        "sensitivity/mean/probability",
    } <= by_measure.keys()
    for measure in (
        "circuitry/preservation",
        "circuitry/localization",
        "circuitry/minimality",
    ):
        row = by_measure[measure]
        assert row.null is not None
        assert row.raw_p is not None
        assert row.adjusted_p is not None
        assert row.rejected is not None
        assert row.family == "circuits/matched-circuitry"
        assert row.dependence is not None


def test_a_circuit_probe_admits_only_differentiable_observable_patchable_sites() -> (
    None
):
    settings = _declaration().settings()
    probe = CircuitProbe("circuits", settings)
    inaccessible = SiteMap.custom(
        MODEL,
        {
            SITE: SiteSpec(
                AXES,
                DTypeClass.REAL,
                differentiable=False,
                access=Access.OBSERVE,
            )
        },
    )

    with pytest.raises(UnpublishedSiteError, match="differentiable"):
        probe.admit(inaccessible)
