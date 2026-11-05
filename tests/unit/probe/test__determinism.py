"""Tests for bitwise run-to-run determinism by published site."""

from typing import TYPE_CHECKING, Final, Self

import pytest
import torch

from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe import DeterminismReport, determinism

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch

COMPONENT: Final = ComponentId("determinism-toy")
AXES: Final = ("item",)
FIRST: Final = Site(COMPONENT, Role.RESIDUAL_STREAM, ("first",))
SECOND: Final = Site(COMPONENT, Role.RESIDUAL_STREAM, ("second",))
SITES: Final = frozenset({FIRST, SECOND})
PUBLISHED: Final = SiteMap.custom(
    COMPONENT,
    {
        site: SiteSpec(
            AXES,
            DTypeClass.REAL,
            differentiable=False,
            access=Access.OBSERVE,
        )
        for site in SITES
    },
)


class _Reading:
    """One toy reading passed through an instrument scope."""

    __slots__ = ("_site", "_tensor")

    def __init__(self, site: Site, tensor: Tensor) -> None:
        self._site = site
        self._tensor = tensor

    @property
    def site(self) -> Site:
        """Return the point read."""
        return self._site

    @property
    def tensor(self) -> Tensor:
        """Return the value read."""
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the reading's one named axis."""
        return AXES

    @property
    def mask(self) -> None:
        """Return that every position is live."""
        return None


class _Scope:
    """The open observation region of the toy component."""

    def __init__(self, model: _Toy, sites: frozenset[Site]) -> None:
        self._model = model
        self._sites = sites
        self._readings: dict[Site, _Reading] = {}

    @property
    def readings(self) -> Mapping[Site, _Reading]:
        """Return readings taken during the call."""
        return self._readings

    def observe(self, site: Site, tensor: Tensor) -> Tensor:
        """File a value when its point was requested."""
        if site in self._sites:
            self._readings[site] = _Reading(site, tensor)

        return tensor

    def __enter__(self) -> Self:
        """Attach this region to the component."""
        self._model.scopes.append(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Detach this region from the component."""
        self._model.scopes.remove(self)


class _Toy:
    """A two-site seeded component with one optional stateful divergence."""

    def __init__(
        self,
        *,
        divergent: Site | None = None,
        omit: Site | None = None,
        first_values: tuple[Tensor, ...] | None = None,
    ) -> None:
        self.divergent = divergent
        self.omit = omit
        self.first_values = first_values
        self.calls = 0
        self.scopes: list[_Scope] = []

    def site_map(self) -> SiteMap:
        """Return the two observable points."""
        return PUBLISHED

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[_Reading]],
    ) -> _Scope:
        """Open an observation region; this model permits no patching."""
        if patches:
            msg = "the determinism diagnostic is observe-only"
            raise AssertionError(msg)

        return _Scope(self, sites)

    def _at(self, site: Site, value: Tensor) -> Tensor:
        """Pass one value through every open region."""
        if site == self.omit:
            return value

        for scope in self.scopes:
            value = scope.observe(site, value)

        return value

    def forward(self) -> Tensor:
        """Draw the two seeded readings and optionally perturb one by call count."""
        self.calls += 1
        first = (
            torch.rand(8)
            if self.first_values is None
            else self.first_values[self.calls - 1]
        )
        second = torch.rand(8)

        if self.divergent == FIRST:
            first = first + self.calls

        if self.divergent == SECOND:
            second = second + self.calls

        return self._at(FIRST, first) + self._at(SECOND, second)


def test_five_identically_seeded_calls_agree_at_every_requested_site() -> None:
    model = _Toy()
    report = determinism(model, model.forward, sites=SITES, seed=7)

    assert isinstance(report, DeterminismReport)
    assert model.calls == 5
    assert report.divergent_sites == frozenset()
    assert {row.site: row.divergent_runs for row in report.sites} == {
        FIRST: (),
        SECOND: (),
    }


def test_one_stateful_reading_is_reported_at_exactly_its_site() -> None:
    model = _Toy(divergent=SECOND)
    report = determinism(model, model.forward, sites=SITES, seed=11)

    assert report.divergent_sites == frozenset({SECOND})
    assert report.at(FIRST).deterministic
    assert report.at(SECOND).divergent_runs == (2, 3, 4, 5)


def test_signed_zeroes_are_bitwise_distinct() -> None:
    positive = torch.tensor([0.0], dtype=torch.float32)
    negative = torch.tensor([-0.0], dtype=torch.float32)
    model = _Toy(first_values=(positive, negative, negative, negative, negative))

    report = determinism(model, model.forward, sites=SITES, seed=11)

    assert torch.equal(positive, negative)
    assert report.at(FIRST).divergent_runs == (2, 3, 4, 5)
    assert report.at(SECOND).deterministic


def test_identical_nan_payloads_are_bitwise_equal() -> None:
    payload = torch.tensor([0x7FC01234], dtype=torch.int32).view(torch.float32)
    model = _Toy(first_values=tuple(payload.clone() for _ in range(5)))

    report = determinism(model, model.forward, sites=SITES, seed=11)

    assert torch.isnan(payload).all()
    assert not torch.equal(payload, payload.clone())
    assert report.at(FIRST).deterministic
    assert report.at(SECOND).deterministic


def test_the_diagnostic_restores_the_callers_random_state() -> None:
    torch.manual_seed(13)
    expected = torch.rand(4)
    torch.manual_seed(13)

    model = _Toy()
    determinism(model, model.forward, sites=SITES, seed=19)

    assert torch.equal(torch.rand(4), expected)


def test_a_call_that_never_reaches_a_requested_site_is_refused() -> None:
    model = _Toy(omit=SECOND)

    with pytest.raises(ValueError, match=r"did not reach.*second"):
        determinism(model, model.forward, sites=SITES, seed=0)


def test_a_diagnostic_needs_a_site_and_an_integer_seed() -> None:
    model = _Toy()

    with pytest.raises(ValueError, match="at least one site"):
        determinism(model, model.forward, sites=frozenset(), seed=0)

    with pytest.raises(TypeError, match="not an integer"):
        determinism(model, model.forward, sites=SITES, seed=True)

    with pytest.raises(ValueError, match="negative"):
        determinism(model, model.forward, sites=SITES, seed=-1)


def test_a_report_refuses_a_site_that_was_not_requested() -> None:
    model = _Toy()
    report = determinism(model, model.forward, sites=frozenset({FIRST}), seed=0)

    with pytest.raises(KeyError, match="no result"):
        report.at(SECOND)
