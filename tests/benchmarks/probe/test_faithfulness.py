"""Wall-clock cost of scoring a circuit alone against the forward pass it prices.

The benchmark times one plain forward pass of a two-path model and one pass
with everything outside a circuit replaced, both on the same inputs, and
requires the ablated pass to cost no more than a fixed number of forward passes,
so that the coordinate masks and the patch machinery stay small beside the
model.
"""

from statistics import median
from time import perf_counter
from typing import TYPE_CHECKING, Final, Self

import pytest
import torch

from lofi.nn.sites import (
    Access,
    ComponentId,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.probe import Ablation, Circuit, Unit, Zero

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from pytest_benchmark.fixture import BenchmarkFixture
    from torch import Tensor

    from lofi.nn.ports import Patch
#: How many forward passes one ablated pass may cost.
FORWARD_EQUIVALENTS: Final = 3.0

MODEL: Final = ComponentId("two-path")
WIDTH: Final = 1024
ITEMS: Final = 256
DEPTH: Final = 3
PATHS: Final = ("a", "b")
NAMES: Final = tuple(
    f"{path}{layer}" for path in PATHS for layer in range(1, DEPTH + 1)
)
SITES: Final = {name: Site(MODEL, Role.RESIDUAL_STREAM, (name,)) for name in NAMES}
AXES: Final = ("item", "width")

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    MODEL,
    {
        site: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=False,
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
    """Two depth-three paths of square layers, read out per item at every layer."""

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

        return readout


def a_circuit() -> Circuit:
    """Return the first path whole and the leading half of the second's last layer."""
    return Circuit(
        (
            *(Unit(SITES[f"a{layer}"]) for layer in range(1, DEPTH + 1)),
            *(
                Unit(SITES[f"b{DEPTH}"], {"width": index})
                for index in range(WIDTH // 2)
            ),
        )
    )


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


def _score(ablation: Ablation[_Held], circuit: Circuit) -> tuple[float, ...]:
    return ablation.only(circuit.units)


def test_scoring_a_circuit_returns_one_score_per_item() -> None:
    model = _TwoPath(0)
    given = torch.randn(ITEMS, WIDTH, generator=torch.Generator().manual_seed(1))
    circuit = a_circuit()
    ablation = Ablation(
        model,
        sites=SITES.values(),
        metric=lambda: model.forward(given),
        intervention=Zero(),
        dataset="two-path",
    )

    scored = _score(ablation, circuit)

    assert len(scored) == ITEMS


def test_scoring_a_circuit_alone_costs_a_bounded_number_of_forward_passes(
    benchmark: BenchmarkFixture,
) -> None:
    model = _TwoPath(0)
    given = torch.randn(ITEMS, WIDTH, generator=torch.Generator().manual_seed(1))
    circuit = a_circuit()
    ablation = Ablation(
        model,
        sites=SITES.values(),
        metric=lambda: model.forward(given),
        intervention=Zero(),
        dataset="two-path",
    )
    forward_seconds = seconds_per_forward(model, given, rounds=9)

    benchmark.pedantic(
        _score,
        args=(ablation, circuit),
        rounds=5,
        iterations=1,
        warmup_rounds=1,
    )

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the ablated pass ran but pytest-benchmark timed nothing")

    # use the median to reduce sensitivity to scheduler delays
    ablated_seconds = measured.stats.median
    forward_equivalents = ablated_seconds / forward_seconds
    benchmark.extra_info["units"] = len(circuit.units)
    benchmark.extra_info["forward_seconds"] = forward_seconds
    benchmark.extra_info["ablated_seconds"] = ablated_seconds
    benchmark.extra_info["forward_equivalents"] = forward_equivalents

    # one forward pass, with room for the masks and the patch machinery
    assert forward_equivalents <= FORWARD_EQUIVALENTS
