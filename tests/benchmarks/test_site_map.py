"""Benchmarks site-map validation and the open observation path."""

from statistics import fmean
from time import perf_counter_ns
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self

import pytest
import torch
from tests._controllers import mlp_controller
from tests.conftest import counted_constructions
from torch import nn

from lofi.judge import Action, ActionSet
from lofi.nn.parsers import (
    DeepInsideOutside,
    LogSumExpReduction,
    MarginalSplitWeigher,
    MeanFieldComposition,
    MlpSpanFunction,
)
from lofi.nn.psi import RuleScorer
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
from lofi.probe import (
    ExternalModel,
    ExternalSite,
    ExternalSiteMap,
    SiteMapIso,
    SiteMapSpec,
    _instrument,
    instrument,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType
    from typing import Never

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.nn.ports import Patch

# width used to construct benchmark components
WIDTH: Final = 8

# number of direct forwards averaged beside the benchmarked open-scope forward
OVERHEAD_ROUNDS: Final = 1_000

OBSERVED: Final = ComponentId("observed-parser")
STREAM: Final = Site(OBSERVED, Role.RESIDUAL_STREAM, (0,))
_EMPTY: Final = MappingProxyType({})


class _EmptyScope:
    """An instrument scope for a component whose benchmark observes no point."""

    @property
    def readings(self) -> Mapping[Site, Never]:
        """Return the empty set of readings."""
        return _EMPTY

    def __enter__(self) -> Self:
        """Open the scope."""
        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Close the scope."""


class _FullMap:
    """Publishes the full parser map through the instrumentable port."""

    def __init__(self, published: SiteMap) -> None:
        self._published = published

    def site_map(self) -> SiteMap:
        """Return the parser's complete published map."""
        return self._published

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[Never]],
    ) -> _EmptyScope:
        """Return the empty component scope used after validation."""
        del sites, patches

        return _EmptyScope()


class Published(NamedTuple):
    """Stores declared and resolved forms of a site map.

    Attributes
    ----------
    declared
        The map as text, which is the form it arrives in.
    resolved
        The map a probe reads, which is the form the codec produces.
    subject
        The instrumentable component publishing the resolved map.
    """

    declared: str
    resolved: SiteMap
    subject: _FullMap


class Observed(NamedTuple):
    """Stores the real neural pass and its external observation point."""

    module: nn.Module
    model: ExternalModel
    given: torch.Tensor
    site: Site


class Crossings(NamedTuple):
    """Stores boundary validations and didactic constructions at one instant."""

    validations: int
    models: int


@pytest.fixture(scope="module")
def published() -> Published:
    """Return the declared and resolved benchmark site map.

    Returns
    -------
    Published
        The declaration and the map behind it.
    """
    keep = Action("keep")
    actions = ActionSet((Action("join", "concatenation"), keep), keep)
    component = ComponentId("parser")
    controller = mlp_controller(
        actions,
        width=WIDTH,
        component=ComponentId("controller"),
        hidden=64,
    )
    parser = DeepInsideOutside(
        RuleScorer(controller, width=WIDTH),
        composition=MeanFieldComposition(
            MlpSpanFunction((WIDTH,), WIDTH, layers=1, nonlinearity="tanh"),
            MlpSpanFunction((WIDTH, WIDTH), WIDTH, layers=1, nonlinearity="tanh"),
            width=WIDTH,
        ),
        split_left=MlpSpanFunction(
            (WIDTH, WIDTH), WIDTH, layers=1, nonlinearity="tanh"
        ),
        split_right=MlpSpanFunction(
            (WIDTH, WIDTH), WIDTH, layers=1, nonlinearity="tanh"
        ),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=component,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    )

    resolved = SiteMap(component, ComponentKind.PARSER, parser.site_map())

    return Published(
        SiteMapIso().backward(resolved).model_dump_json(),
        resolved,
        _FullMap(resolved),
    )


@pytest.fixture(scope="module")
def observed() -> Observed:
    """Return one real neural forward with an observe-only hook installed."""
    module = nn.Sequential(nn.Linear(WIDTH, WIDTH), nn.Tanh())
    declared = ExternalSiteMap(
        OBSERVED,
        (
            ExternalSite(
                "0",
                Role.RESIDUAL_STREAM,
                0,
                SiteSpec(
                    ("item", "position", "width"),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                ),
            ),
        ),
    )
    given = torch.linspace(-1.0, 1.0, 6 * WIDTH).reshape(2, 3, WIDTH)

    return Observed(
        module,
        ExternalModel(module, declared),
        given,
        STREAM,
    )


def _open_scope(subject: _FullMap) -> None:
    """Open and close one instrument scope over the subject's full map."""
    with instrument(subject):
        pass


def _forward(observed: Observed) -> None:
    """Run the real neural pass used by the untimed and timed cases."""
    observed.module(observed.given)


def _mean_plain_ns(observed: Observed) -> float:
    """Return the mean wall nanoseconds of the same pass outside a scope."""
    _forward(observed)
    taken: list[int] = []

    for _ in range(OVERHEAD_ROUNDS):
        start = perf_counter_ns()
        _forward(observed)
        taken.append(perf_counter_ns() - start)

    return fmean(taken)


def test_the_published_map_names_exact_roles_keys_and_permissions(
    published: Published,
) -> None:
    permissions = {
        (site.role, site.key): spec.access for site, spec in published.resolved.items()
    }

    assert permissions == {
        (Role.INSIDE_SCORE, ("inside",)): Access.OBSERVE,
        (Role.OUTSIDE_SCORE, ("outside",)): Access.OBSERVE,
        (Role.SPLIT_DISTRIBUTION, ("alpha",)): Access.OBSERVE,
        (Role.CONSTITUENT_MARGINAL, ("mu",)): Access.OBSERVE,
        (Role.LOG_PARTITION, ("log_z",)): Access.OBSERVE,
        (Role.IDENTITY_SCORE, ("identity",)): Access.OBSERVE,
        (Role.CHART_CELL, ("cells",)): Access.OBSERVE,
        (Role.OUTSIDE_CELL, ("outside_cells",)): Access.OBSERVE,
        (Role.SPAN_REPRESENTATION, ("spans",)): Access.OBSERVE,
        (Role.RECEIVER_EDGE, ("receiver",)): Access.OBSERVE | Access.PATCH,
    }


def test_the_declaration_a_scope_reads_resolves_to_what_was_published(
    published: Published,
) -> None:
    resolved = SiteMapIso().forward(SiteMapSpec.model_validate_json(published.declared))

    assert resolved == published.resolved


def test_the_timed_scope_callable_validates_the_full_map_once(
    monkeypatch: pytest.MonkeyPatch,
    published: Published,
) -> None:
    validated: list[SiteMap] = []
    validate = _instrument.ensure_publishes

    def recording(published_map: SiteMap) -> SiteMap:
        validated.append(published_map)

        return validate(published_map)

    monkeypatch.setattr(_instrument, "ensure_publishes", recording)

    _open_scope(published.subject)

    assert validated == [published.resolved]


def test_a_forward_inside_an_open_scope_crosses_no_boundary(
    monkeypatch: pytest.MonkeyPatch,
    observed: Observed,
) -> None:
    validations = 0
    validate = _instrument.ensure_publishes

    def counting(published_map: SiteMap) -> SiteMap:
        nonlocal validations
        validations += 1

        return validate(published_map)

    monkeypatch.setattr(_instrument, "ensure_publishes", counting)

    with (
        counted_constructions() as constructions,
        torch.no_grad(),
        instrument(observed.model, observe=frozenset({observed.site})),
    ):
        opened = Crossings(validations, constructions.count)
        _forward(observed)
        forwarded = Crossings(validations, constructions.count)

    assert opened.validations == 1
    assert forwarded.validations - opened.validations == 0
    assert forwarded.models - opened.models == 0


def test_a_scope_validates_the_map_it_opens_over_within_budget(
    benchmark: BenchmarkFixture,
    published: Published,
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark(_open_scope, published.subject)

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the codec ran but pytest-benchmark timed nothing")

    per_scope_ms = measured.stats.mean * 1e3
    benchmark.extra_info["points"] = len(published.resolved)
    benchmark.extra_info["mean_ms_per_scope"] = per_scope_ms
    benchmark.extra_info["median_ms_per_scope"] = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "P-39"
    benchmark.extra_info["budget_cell"] = "scope-validation"

    # enforce the 10 ms per-scope budget
    if wall_clock_budget.enforced:
        assert per_scope_ms <= 10


def test_an_open_observe_only_scope_reports_its_per_forward_overhead(
    benchmark: BenchmarkFixture,
    observed: Observed,
) -> None:
    with torch.no_grad():
        plain_before_ns = _mean_plain_ns(observed)

        with instrument(observed.model, observe=frozenset({observed.site})):
            benchmark.pedantic(
                _forward,
                args=(observed,),
                iterations=100,
                rounds=100,
                warmup_rounds=5,
            )

        plain_after_ns = _mean_plain_ns(observed)

    measured = benchmark.stats

    if measured is None:
        pytest.fail("the open-scope forward ran but pytest-benchmark timed nothing")

    scoped_ns = measured.stats.mean * 1e9
    plain_ns = fmean((plain_before_ns, plain_after_ns))
    benchmark.extra_info["mean_plain_us_per_forward"] = plain_ns / 1e3
    benchmark.extra_info["mean_open_scope_us_per_forward"] = scoped_ns / 1e3
    benchmark.extra_info["mean_overhead_us_per_forward"] = (scoped_ns - plain_ns) / 1e3
    benchmark.extra_info["observed_points"] = 1
