from math import inf, log
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self, override

import pytest
import torch
from tests.conformance._sites import SiteObservation
from tests.conformance.chart_scorer import (
    CHART_INSTRUMENTATION_LAWS,
    CHART_SCORER_LAWS,
    ChartInstrumentationConformance,
    ChartInstrumentationSuite,
    ChartScorerConformance,
    ChartScorerSuite,
    published,
)

from lofi._checks import LawViolationError
from lofi.nn._values import ChartReadouts, Encoded, LogDistribution
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch

AXES: Final[Mapping[str, tuple[str, ...]]] = {
    "inside": ("item", "span"),
    "outside": ("item", "span"),
    "alpha": ("item", "span", "split"),
    "mu": ("item", "span"),
    "log_z": ("item",),
    "cells": ("item", "span", "width"),
}

FILLED: Final = frozenset({"inside", "alpha", "mu", "log_z", "cells"})

SCORER: Final = ComponentId("width-chart")

ITEMS: Final = 2

WORDS: Final = 4

NAMEABLE: Final[Mapping[str, tuple[Role, DTypeClass]]] = {
    "outside": (Role.OUTSIDE_SCORE, DTypeClass.REAL),
    "cells": (Role.CHART_CELL, DTypeClass.REAL),
}


class Read(NamedTuple):
    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class Scope:
    def __init__(
        self,
        sites: frozenset[Site],
        enter: Callable[[Scope], None],
        leave: Callable[[Scope], None],
    ) -> None:
        self._sites = sites
        self._enter = enter
        self._leave = leave
        self._readings: dict[Site, Read] = {}
        self._closed = False

    @property
    def readings(self) -> Mapping[Site, Read]:
        return MappingProxyType(self._readings)

    def record(self, reading: Read) -> None:
        if not self._closed and reading.site in self._sites:
            self._readings[reading.site] = reading

    def __enter__(self) -> Self:
        self._enter(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        del kind, error, traceback
        self._closed = True
        self._leave(self)


def spans(length: int) -> tuple[tuple[int, int], ...]:
    """Return every span of at least two words, narrowest first.

    Parameters
    ----------
    length
        The longest item's word count.

    Returns
    -------
    tuple of tuple of int
        The half-open ``[start, end)`` spans, ordered by width and then by start, which
        is the order the span axis runs in.
    """
    return tuple(
        (start, start + width)
        for width in range(2, length + 1)
        for start in range(length - width + 1)
    )


def leaves(items: int, words: int) -> Encoded:
    """Return the leaves of a batch whose items shorten by one word each.

    Parameters
    ----------
    items
        How many items the batch holds.
    words
        How many words its longest item holds.

    Returns
    -------
    Encoded
        Rank 3 over the axes ``("item", "word", "width")`` at double precision, masked
        over the item and word axes so that a shorter item leaves its trailing words as
        padding.
    """
    lengths = [max(2, words - item) for item in range(items)]
    longest = max(lengths)
    values = torch.zeros(items, longest, 1, dtype=torch.float64)
    mask = torch.zeros(items, longest, dtype=torch.bool)

    for item, length in enumerate(lengths):
        values[item, :length, 0] = torch.arange(1, length + 1, dtype=torch.float64)
        mask[item, :length] = True

    return Encoded(values, ("item", "word", "width"), mask)


class Chart:
    def __init__(self) -> None:
        self._scopes: list[Scope] = []

    @property
    def publishes(self) -> frozenset[str]:
        return FILLED

    def site_map(self) -> SiteMap:
        return SiteMap(
            SCORER,
            ComponentKind.PARSER,
            {
                Site(SCORER, NAMEABLE[readout][0], (readout,)): SiteSpec(
                    AXES[readout],
                    NAMEABLE[readout][1],
                    differentiable=True,
                    access=Access.OBSERVE,
                )
                for readout in sorted(self.publishes & frozenset(NAMEABLE))
            },
        )

    def wire(self, readout: str) -> None:
        """Accept a loss over one readout.

        Parameters
        ----------
        readout
            The readout the loss consumes.

        Raises
        ------
        ValueError
            If this scorer does not publish it, since the loss would otherwise read a
            chart cell no pass ever fills.
        """
        if readout not in self.publishes:
            msg = (
                f"no pass fills {readout!r}; this scorer publishes "
                f"{sorted(self.publishes)}"
            )
            raise ValueError(msg)

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Read]]
    ) -> Scope:
        published_sites = self.site_map()

        for site in sites | frozenset(patches):
            if site not in published_sites:
                msg = f"this chart does not publish {site!r}"
                raise ValueError(msg)

        if patches:
            msg = "this chart is observation-only"
            raise ValueError(msg)

        return Scope(sites, self._attach, self._detach)

    def _alpha(
        self, live: torch.Tensor, layout: tuple[tuple[int, int], ...]
    ) -> LogDistribution:
        items = int(live.shape[0])
        widest = max(end - start for start, end in layout)
        weights = torch.full(
            (items, len(layout), widest - 1), -inf, dtype=torch.float64
        )

        for span, (start, end) in enumerate(layout):
            weights[:, span, : end - start - 1] = log(1.0 / (end - start - 1))

        return LogDistribution(
            torch.where(live.unsqueeze(-1), weights, -inf),
            ("item", "span", "split"),
            live,
            normalized_over="split",
        )

    def _inside(
        self,
        values: torch.Tensor,
        live: torch.Tensor,
        layout: tuple[tuple[int, int], ...],
    ) -> LogDistribution:
        scores = torch.stack(
            [
                -values[:, start:end, 0].sum(dim=-1) - float(end - start)
                for start, end in layout
            ],
            dim=-1,
        )

        return LogDistribution(torch.where(live, scores, -inf), ("item", "span"), live)

    def readouts(self, leaf: Encoded, /) -> ChartReadouts:
        mask = leaf.mask
        lengths = (
            torch.full((leaf.size("item"),), leaf.size("word"))
            if mask is None
            else mask.sum(dim=-1)
        )
        layout = spans(leaf.size("word"))
        ends = torch.tensor([end for _, end in layout])
        live = ends.unsqueeze(0) <= lengths.unsqueeze(-1)
        inside = self._inside(leaf.tensor, live, layout)

        readouts = self._record(inside, self._alpha(live, layout), live)

        if self._scopes:
            self._publish(readouts)

        return readouts

    def _reading(self, site: Site, value: LogDistribution | Encoded) -> Read:
        return Read(site, value.tensor, value.axes, value.mask)

    def _publish(self, readouts: ChartReadouts) -> None:
        for name, value in published(readouts).items():
            declared = NAMEABLE.get(name)

            if declared is None:
                continue

            site = Site(SCORER, declared[0], (name,))
            reading = self._reading(site, value)

            for scope in self._scopes:
                scope.record(reading)

    def _attach(self, scope: Scope) -> None:
        self._scopes.append(scope)

    def _detach(self, scope: Scope) -> None:
        if not self._scopes or self._scopes[-1] is not scope:
            msg = "chart scopes close inside out"
            raise RuntimeError(msg)

        self._scopes.pop()

    def _record(
        self, inside: LogDistribution, alpha: LogDistribution, live: torch.Tensor
    ) -> ChartReadouts:
        return ChartReadouts(
            inside=inside,
            alpha=alpha,
            mu=Encoded(inside.tensor.exp(), ("item", "span"), live),
            log_z=LogDistribution(inside.tensor.logsumexp(dim=-1), ("item",)),
            cells=Encoded(inside.tensor.unsqueeze(-1), ("item", "span", "width"), live),
        )


class Loose(Chart):
    @override
    def _alpha(
        self, live: torch.Tensor, layout: tuple[tuple[int, int], ...]
    ) -> LogDistribution:
        items = int(live.shape[0])
        widest = max(end - start for start, end in layout)
        weights = torch.zeros((items, len(layout), widest - 1), dtype=torch.float64)

        return LogDistribution(
            torch.where(live.unsqueeze(-1), weights, -inf),
            ("item", "span", "split"),
            live,
        )


class Zeroed(Chart):
    @override
    def _inside(
        self,
        values: torch.Tensor,
        live: torch.Tensor,
        layout: tuple[tuple[int, int], ...],
    ) -> LogDistribution:
        scored = super()._inside(values, live, layout)

        return LogDistribution(torch.where(live, scored.tensor, 0.0), ("item", "span"))


class Understating(Chart):
    @property
    @override
    def publishes(self) -> frozenset[str]:
        return FILLED - {"log_z"}


class Trusting(Chart):
    @override
    def wire(self, readout: str) -> None: ...


class Copying(Chart):
    @override
    def _reading(self, site: Site, value: LogDistribution | Encoded) -> Read:
        return Read(site, value.tensor.clone(), value.axes, value.mask)


class Drifting(Chart):
    @override
    def _record(
        self, inside: LogDistribution, alpha: LogDistribution, live: torch.Tensor
    ) -> ChartReadouts:
        if self._scopes:
            inside = LogDistribution(inside.tensor + 1.0, inside.axes, inside.mask)

        return super()._record(inside, alpha, live)


class Boasting(Chart):
    @override
    def site_map(self) -> SiteMap:
        return SiteMap(
            SCORER,
            ComponentKind.PARSER,
            {
                **super().site_map(),
                Site(SCORER, Role.OUTSIDE_SCORE, ("outside",)): SiteSpec(
                    AXES["outside"],
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                ),
            },
        )


def _observe(subject: Chart) -> Mapping[Site, LogDistribution | Encoded]:
    """Return what a pass of a scorer exposes, keyed by the site each is under.

    Parameters
    ----------
    subject
        The scorer to run.

    Returns
    -------
    collections.abc.Mapping
        Each filled readout an observer may name, under its site.
    """
    filled = published(subject.readouts(leaves(ITEMS, WORDS)))

    return {
        Site(SCORER, NAMEABLE[name][0], (name,)): value
        for name, value in filled.items()
        if name in NAMEABLE
    }


def _against(build: type[Chart]) -> ChartScorerConformance[Chart]:
    """Return the conformance the laws run one kind of scorer through.

    Parameters
    ----------
    build
        The scorer to build.

    Returns
    -------
    ChartScorerConformance
        The fixture, over the ragged batch the leaves above build.
    """
    return ChartScorerConformance(
        build=build,
        leaves=leaves,
        axes=AXES,
        split_axis="split",
        ragged_axis="span",
        wire=lambda subject, readout: subject.wire(readout),
        sites=SiteObservation(
            build=build,
            observe=_observe,
            read=lambda subject: tuple(
                value.tensor
                for value in published(subject.readouts(leaves(ITEMS, WORDS))).values()
            ),
        ),
    )


def _instrumented(build: type[Chart]) -> ChartInstrumentationConformance[Chart, Read]:
    return ChartInstrumentationConformance(
        build=build,
        leaves=leaves,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        items=ITEMS,
        words=WORDS,
    )


def _red(conformance: ChartScorerConformance[Chart]) -> frozenset[str]:
    """Return the name of every law that fails against a subject.

    Parameters
    ----------
    conformance
        The subject and what the laws need to run against it.

    Returns
    -------
    frozenset of str
        The names of the failing laws.
    """
    suite = ChartScorerSuite()
    red: list[str] = []

    for law in CHART_SCORER_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


def _instrumentation_red(
    conformance: ChartInstrumentationConformance[Chart, Read],
) -> frozenset[str]:
    suite = ChartInstrumentationSuite()
    red: list[str] = []

    for law in CHART_INSTRUMENTATION_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, type[Chart]], ...]] = (
    ("test_split_weights_normalize_per_span", Loose),
    ("test_padding_in_a_log_chart_is_never_an_exact_zero", Zeroed),
    ("test_every_declared_readout_is_present_and_the_rest_absent", Understating),
    ("test_a_readout_no_scorer_publishes_is_refused_at_wiring", Trusting),
    ("test_the_site_map_matches_what_instrumentation_yields", Boasting),
)


class TestAChartScorerThatPublishesWhatItFills(ChartScorerSuite):
    @pytest.fixture
    def conformance(self) -> ChartScorerConformance[Chart]:
        return _against(Chart)


class TestAnInstrumentedChartReturnsWhatItPublishes(ChartInstrumentationSuite):
    @pytest.fixture
    def chart_instrumentation(
        self,
    ) -> ChartInstrumentationConformance[Chart, Read]:
        return _instrumented(Chart)


def test_every_law_holds_of_a_scorer_that_publishes_what_it_fills() -> None:
    assert _red(_against(Chart)) == frozenset()


def test_every_instrumentation_law_holds_of_exact_observation() -> None:
    assert _instrumentation_red(_instrumented(Chart)) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    [
        ("test_a_scope_reads_the_exact_objects_the_chart_pass_returns", Copying),
        ("test_observing_leaves_the_chart_pass_bit_identical", Drifting),
    ],
)
def test_exactly_the_broken_instrumentation_law_goes_red(
    law: str, build: type[Chart]
) -> None:
    assert _instrumentation_red(_instrumented(build)) == frozenset({law})


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[law.removeprefix("test_") for law, _ in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, build: type[Chart]) -> None:
    assert _red(_against(build)) == frozenset({law})


def test_a_shorter_item_leaves_the_spans_past_it_as_padding() -> None:
    readouts = Chart().readouts(leaves(ITEMS, WORDS))
    inside = readouts.inside

    assert inside is not None
    assert inside.mask is not None
    assert inside.axes == ("item", "span")
    assert spans(4)[-1] == (0, 4)
    assert bool(inside.mask[0].all())
    assert not bool(inside.mask[1, -1])
    assert inside.tensor[1, -1] == -inf


def test_the_published_points_describe_the_chart_readouts_the_pass_fills() -> None:
    scorer = Chart()
    points = scorer.site_map()
    filled = published(scorer.readouts(leaves(ITEMS, WORDS)))

    assert {site.key[0] for site in points} == scorer.publishes & frozenset(NAMEABLE)

    for site, spec in points.items():
        assert spec.axes == filled[str(site.key[0])].axes
        assert Access.OBSERVE in spec.access

    assert Site(SCORER, Role.OUTSIDE_SCORE, ("outside",)) not in points
