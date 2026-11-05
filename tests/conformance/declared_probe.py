"""Reusable laws for probes a run document declares, and a toy trained point.

The toy point holds one chart component publishing a span-vector point that may
be observed and patched, a constituent-marginal point and a log-partition point.
Its held-out passes cover items whose gold constituents are known, and the first
coordinate of every span vector says whether the span is a gold constituent, so
a linear probe can read the constituent label and LEACE can erase it.
"""

import math
from types import MappingProxyType, TracebackType
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, Self, cast, override

import pytest
import torch
from torch import nn

from lofi.deduce import SpanIndex
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe import PassLayout, ProbeValue, UnpublishedSiteError, published_sites
from lofi.probe._probing import GradientMode

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.nn.ports import Patch, Reading
    from lofi.probe import DeclaredProbe, HeldOutPass, Probed, TrainedPoint

__all__ = [
    "CELLS",
    "CHART",
    "LOG_Z",
    "MU",
    "DeclaredProbeSuite",
    "ToyChart",
    "ToyPoint",
    "toy_point",
]

CHART: Final = ComponentId("chart")
CELLS: Final = Site(CHART, Role.CHART_CELL, ("cells",))
MU: Final = Site(CHART, Role.CONSTITUENT_MARGINAL, ("mu",))
LOG_Z: Final = Site(CHART, Role.LOG_PARTITION, ("log_z",))
WIDTH: Final = 3

PUBLISHED: Final = SiteMap.custom(
    CHART,
    {
        CELLS: SiteSpec(
            ("item", "span", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        MU: SiteSpec(
            ("item", "span"),
            DTypeClass.PROBABILITY,
            differentiable=True,
            access=Access.OBSERVE,
        ),
        LOG_Z: SiteSpec(
            ("item",),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
    },
)


class _Reading(NamedTuple):
    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None


class _Scope:
    """One instrumented region over the toy chart."""

    def __init__(
        self,
        owner: ToyChart,
        sites: frozenset[Site],
        patches: Mapping[Site, Patch[_Reading]],
    ) -> None:
        self._owner = owner
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Reading] = {}

    @property
    def readings(self) -> Mapping[Site, _Reading]:
        return MappingProxyType(self._readings)

    def __enter__(self) -> Self:
        self._owner.scopes.append(self)
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self._owner.scopes.remove(self)

    def patched(self, reading: _Reading) -> Tensor:
        patch = self._patches.get(reading.site)
        return reading.tensor if patch is None else patch.stands_in_for(reading)

    def record(self, reading: _Reading) -> None:
        if reading.site in self._sites:
            self._readings[reading.site] = reading


class ToyChart(nn.Module):
    """A chart whose span vectors say whether each span is a gold constituent."""

    def __init__(self) -> None:
        super().__init__()
        self.scale = nn.Parameter(torch.ones(WIDTH))
        self.scopes: list[_Scope] = []

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Reading]]
    ) -> _Scope:
        return _Scope(self, sites, patches)

    def _published(self, reading: _Reading) -> Tensor:
        tensor = reading.tensor

        for scope in self.scopes:
            tensor = scope.patched(reading._replace(tensor=tensor))

        for scope in self.scopes:
            scope.record(reading._replace(tensor=tensor))

        return tensor

    @override
    def forward(self, layout: PassLayout) -> Tensor:
        spans = layout.spans
        assert spans is not None
        items = len(layout.words)
        cells = torch.zeros(items, spans.num_spans, WIDTH)
        live = torch.zeros(items, spans.num_spans, dtype=torch.bool)

        for item, (words, gold) in enumerate(
            zip(layout.words, layout.gold, strict=True)
        ):
            for span in range(spans.num_spans):
                start, end = spans.span_of(span)

                if end <= len(words):
                    live[item, span] = True
                    constituent = gold is not None and (start, end) in gold
                    cells[item, span] = torch.tensor(
                        [float(constituent), float(end - start), float(len(words))]
                    )

        cells = self._published(
            _Reading(CELLS, cells * self.scale, ("item", "span", "width"), live)
        )
        mu = torch.sigmoid(cells[..., 0]) * live
        self._published(_Reading(MU, mu, ("item", "span"), live))
        log_z = (cells.sum(dim=-1) * live).sum(dim=-1)
        self._published(_Reading(LOG_Z, log_z, ("item",), None))

        return log_z


def _layout(sentences: tuple[tuple[str, ...], ...]) -> PassLayout:
    """Return a batch whose gold analyses are right-branching."""
    gold = tuple(
        {(start, len(words)): "X" for start in range(len(words))}
        | {(start, start + 1): "W" for start in range(len(words))}
        for words in sentences
    )

    return PassLayout(sentences, SpanIndex(max(map(len, sentences))), tuple(gold))


class _Pass:
    def __init__(self, chart: ToyChart, layout: PassLayout) -> None:
        self._chart = chart
        self._layout = layout

    def __call__(self, gradients: GradientMode, /) -> PassLayout:
        with torch.set_grad_enabled(gradients is GradientMode.ENABLED):
            self._chart(self._layout)

        return self._layout


class ToyPoint:
    """A trained point holding one toy chart and its held-out passes."""

    def __init__(self, chart: ToyChart | None = None) -> None:
        self.chart = ToyChart() if chart is None else chart
        batches = (
            (("a", "b", "c"), ("d", "e")),
            (("f", "g", "h", "i"), ("j", "k", "l")),
            (("m", "n"), ("o", "p", "q")),
        )
        self._passes = tuple(_Pass(self.chart, _layout(batch)) for batch in batches)

    @property
    def components(self) -> Mapping[str, Probed[Reading]]:
        return MappingProxyType({"chart": self.chart})

    @property
    def passes(self) -> tuple[HeldOutPass, ...]:
        return self._passes


def toy_point() -> ToyPoint:
    """Return a fresh toy trained point."""
    return ToyPoint()


def _modules(point: TrainedPoint) -> dict[str, nn.Module]:
    """Return every component of a point as the module it is."""
    return {
        role: cast("nn.Module", component)
        for role, component in point.components.items()
    }


def _state(point: TrainedPoint) -> dict[str, Tensor]:
    """Return a detached copy of every component's state."""
    return {
        f"{role}.{name}": value.detach().clone()
        for role, module in _modules(point).items()
        for name, value in module.state_dict().items()
    }


def _modes(point: TrainedPoint) -> dict[str, bool]:
    """Return every component's training flag."""
    return {role: module.training for role, module in _modules(point).items()}


class DeclaredProbeSuite:
    """Laws every declared probe obeys against a point it admits.

    A subclass supplies a ``declared_probe`` fixture, a probe the ``trained_point``
    fixture admits, and may override ``trained_point``.
    """

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    unpublished: ClassVar[Mapping[Site, SiteSpec]] = MappingProxyType({})

    @pytest.fixture
    def trained_point(self) -> TrainedPoint:
        return toy_point()

    def test_admits_the_sites_its_point_publishes(
        self, declared_probe: DeclaredProbe, trained_point: TrainedPoint
    ) -> None:
        declared_probe.admit(published_sites(trained_point.components))

    def test_refuses_a_point_that_publishes_nothing(
        self, declared_probe: DeclaredProbe
    ) -> None:
        with pytest.raises(UnpublishedSiteError):
            declared_probe.admit(self.unpublished)

    def test_reports_finite_values_over_published_sites(
        self, declared_probe: DeclaredProbe, trained_point: TrainedPoint
    ) -> None:
        values = declared_probe.run(trained_point)
        published = published_sites(trained_point.components)

        assert values
        assert all(type(value) is ProbeValue for value in values)
        assert all(math.isfinite(value.value) for value in values)
        assert all(value.count > 0 for value in values)
        assert all(value.site is None or value.site in published for value in values)
        assert len({(value.measure, value.site) for value in values}) == len(values)

    def test_is_repeatable(
        self, declared_probe: DeclaredProbe, trained_point: TrainedPoint
    ) -> None:
        assert declared_probe.run(trained_point) == declared_probe.run(trained_point)

    def test_leaves_the_point_as_it_found_it(
        self, declared_probe: DeclaredProbe, trained_point: TrainedPoint
    ) -> None:
        before = _state(trained_point)
        modes = _modes(trained_point)

        declared_probe.run(trained_point)

        after = _state(trained_point)

        assert before.keys() == after.keys()
        assert all(torch.equal(before[name], after[name]) for name in before)
        assert _modes(trained_point) == modes
