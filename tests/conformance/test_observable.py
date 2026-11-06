from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self, override

import pytest
import torch
from tests.conformance.observable import (
    OBSERVABLE_LAWS,
    ObservableConformance,
    ObservableSuite,
)
from torch import nn

from lofi._checks import LawViolationError
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import (
        Instrumentable,
        InstrumentScope,
        Observable,
        Patch,
        Reading,
    )
PANEL: Final = ComponentId("residual-panel")
LEDGER: Final = ComponentId("cell-ledger")

STREAM: Final = Site(PANEL, Role.RESIDUAL_STREAM)
CELLS: Final = Site(LEDGER, Role.CHART_CELL)
OUTSIDE: Final = Site(LEDGER, Role.OUTSIDE_SCORE)

ROWS: Final = ("row", "width")

STREAM_SPEC: Final = SiteSpec(
    ROWS, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
)
CELL_SPEC: Final = SiteSpec(
    ROWS, DTypeClass.REAL, differentiable=False, access=Access.OBSERVE
)
OUTSIDE_SPEC: Final = SiteSpec(
    ROWS, DTypeClass.LOG_PROBABILITY, differentiable=False, access=Access.OBSERVE
)

WIDTH: Final = 3
HEIGHT: Final = 2


class Read(NamedTuple):
    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class Scope:
    """A region over a set of published points, holding what was read in it.

    Parameters
    ----------
    sites
        The points to keep readings at.
    enter
        Tell the component the region has opened.
    leave
        Tell the component the region has closed.
    """

    def __init__(
        self,
        sites: frozenset[Site],
        enter: Callable[[Scope], None],
        leave: Callable[[], None],
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
        """Keep one reading, where the scope was opened over its point.

        Parameters
        ----------
        reading
            What was read.
        """
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
        self._closed = True
        self._leave()


def _input() -> Tensor:
    return torch.linspace(-1.0, 1.0, HEIGHT * WIDTH).reshape(HEIGHT, WIDTH)


class Panel(nn.Module):
    def __init__(self, scale: float = 1.0) -> None:
        super().__init__()
        torch.manual_seed(3)
        self._layer = nn.Linear(WIDTH, WIDTH)
        self.register_buffer("_rows", _input())
        self.scale = scale
        self._open: Scope | None = None

    def site_map(self) -> SiteMap:
        return SiteMap.custom(PANEL, {STREAM: STREAM_SPEC})

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Read]]
    ) -> Scope:
        if patches:
            msg = "this panel publishes nothing that may be patched"
            raise ValueError(msg)

        return Scope(sites, self._attach, self._detach)

    def compute(self) -> Tensor:
        stream = self._layer(self.get_buffer("_rows")) * self.scale
        self._record(Read(STREAM, stream.detach(), ROWS))

        return stream

    def _record(self, reading: Read) -> None:
        if self._open is not None:
            self._open.record(reading)

    def _attach(self, scope: Scope) -> None:
        self._open = scope

    def _detach(self) -> None:
        self._open = None


class Ledger:
    def __init__(self, floor: float = -8.0) -> None:
        self._table = _input()
        self.floor = floor
        self._open: Scope | None = None

    def site_map(self) -> SiteMap:
        return SiteMap.custom(LEDGER, {CELLS: CELL_SPEC, OUTSIDE: OUTSIDE_SPEC})

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Read]]
    ) -> Scope:
        if patches:
            msg = "this ledger publishes nothing that may be patched"
            raise ValueError(msg)

        return Scope(sites, self._attach, self._detach)

    def compute(self) -> tuple[Tensor, Tensor]:
        inside = self._table * 2.0
        outside = (self._table.abs() + 1e-3).log().clamp(min=self.floor)
        live = torch.tensor([True, False])

        self._record(Read(CELLS, inside, ROWS))
        self._record(Read(OUTSIDE, outside, ROWS, live))

        return (inside, outside)

    def _record(self, reading: Read) -> None:
        if self._open is not None:
            self._open.record(reading)

    def _attach(self, scope: Scope) -> None:
        self._open = scope

    def _detach(self) -> None:
        self._open = None


class Renaming(Panel):
    @override
    def compute(self) -> Tensor:
        stream = self._layer(self.get_buffer("_rows")) * self.scale
        self._record(Read(STREAM, stream.detach(), ("width", "row")))

        return stream


class Flattening(Panel):
    @override
    def compute(self) -> Tensor:
        stream = self._layer(self.get_buffer("_rows")) * self.scale
        self._record(Read(STREAM, stream.detach().reshape(-1), ROWS))

        return stream


class Silent(Panel):
    @override
    def compute(self) -> Tensor:
        return self._layer(self.get_buffer("_rows")) * self.scale


class Reconfiguring(Panel):
    @override
    def compute(self) -> Tensor:
        if self._open is not None:
            self.scale = self.scale * 2.0

        return super().compute()


class Growing(Panel):
    @override
    def compute(self) -> Tensor:
        if self._open is not None and "extra" not in dict(self.named_modules()):
            self.add_module("extra", nn.Linear(WIDTH, WIDTH))

        return super().compute()


class Drifting(Panel):
    @override
    def compute(self) -> Tensor:
        stream = self._layer(self.get_buffer("_rows")) * self.scale

        if self._open is not None:
            stream = stream + 1.0

        self._record(Read(STREAM, stream.detach(), ROWS))

        return stream


class Traveling(Panel):
    @override
    def compute(self) -> Tensor:
        if self._open is not None:
            self.to("meta")

        return super().compute()


def _panel[T: Panel](build: Callable[[], T]) -> ObservableConformance[T, Read]:
    """Return the conformance the laws run one kind of panel through.

    Parameters
    ----------
    build
        The panel to build.

    Returns
    -------
    ObservableConformance
        The fixture, over the rows above.
    """
    return ObservableConformance(
        build=build,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        run=lambda subject: (subject.compute(),),
        configuration=lambda subject: {"scale": subject.scale},
        submodules=lambda subject: frozenset(
            name for name, _ in subject.named_modules()
        ),
        placements=lambda subject: tuple(
            str(tensor.device) for tensor in (*subject.parameters(), *subject.buffers())
        ),
    )


def _ledger() -> ObservableConformance[Ledger, Read]:
    """Return the conformance the laws run the ledger through.

    Returns
    -------
    ObservableConformance
        The fixture, over the table above.
    """
    return ObservableConformance(
        build=Ledger,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        run=lambda subject: subject.compute(),
        configuration=lambda subject: {"floor": subject.floor},
        submodules=lambda _: frozenset(),
        placements=lambda _: ("cpu",),
    )


def _red[SubjectT: Observable, ReadingT: Reading](
    conformance: ObservableConformance[SubjectT, ReadingT],
) -> frozenset[str]:
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
    suite = ObservableSuite()
    red: list[str] = []

    for law in OBSERVABLE_LAWS:
        try:
            law(suite, conformance)
        except LawViolationError:
            red.append(law.__name__)

    return frozenset(red)


_BROKEN: Final[tuple[tuple[str, Callable[[], Panel]], ...]] = (
    ("test_the_site_map_is_what_a_scope_yields", Renaming),
    ("test_the_site_map_is_what_a_scope_yields", Flattening),
    ("test_the_site_map_is_what_a_scope_yields", Silent),
    ("test_observing_leaves_the_observed_alone", Reconfiguring),
    ("test_observing_leaves_the_observed_alone", Growing),
    ("test_observing_leaves_the_observed_alone", Drifting),
    ("test_observing_leaves_the_observed_alone", Traveling),
)


class TestAPanelThatPublishesWhatItReads(ObservableSuite):
    @pytest.fixture
    def conformance(self) -> ObservableConformance[Panel, Read]:
        return _panel(Panel)


class TestALedgerThatPublishesWhatItReads(ObservableSuite):
    @pytest.fixture
    def conformance(self) -> ObservableConformance[Ledger, Read]:
        return _ledger()


def test_both_laws_hold_of_a_panel_that_publishes_what_it_reads() -> None:
    assert _red(_panel(Panel)) == frozenset()


def test_both_laws_hold_of_a_ledger_that_publishes_what_it_reads() -> None:
    assert _red(_ledger()) == frozenset()


@pytest.mark.parametrize(
    ("law", "build"),
    _BROKEN,
    ids=[build.__name__.lower() for _, build in _BROKEN],
)
def test_exactly_the_broken_law_goes_red(law: str, build: Callable[[], Panel]) -> None:
    assert _red(_panel(build)) == frozenset({law})


def test_a_component_publishing_nothing_has_nothing_to_be_read() -> None:
    class Blank(Panel):
        @override
        def site_map(self) -> SiteMap:
            return SiteMap.custom(PANEL, {})

    assert "test_the_site_map_is_what_a_scope_yields" in _red(_panel(Blank))


def test_a_configuration_read_off_the_fixture_is_what_the_purity_law_compares() -> None:
    conformance = _panel(Panel)
    subject = conformance.build()

    assert conformance.configuration(subject) == {"scale": 1.0}
    assert conformance.submodules(subject) == frozenset({"", "_layer"})
    assert conformance.placements(subject) == ("cpu", "cpu", "cpu")


def _published(subject: Observable) -> SiteMap:
    """Return what a component publishes, read through the declared port.

    Parameters
    ----------
    subject
        The component to read.

    Returns
    -------
    SiteMap
        Its published points.
    """
    return subject.site_map()


def _opened(
    subject: Instrumentable[Read], sites: frozenset[Site]
) -> InstrumentScope[Read]:
    """Return a region over those points, opened through the declared port.

    Parameters
    ----------
    subject
        The component to instrument.
    sites
        The points to read.

    Returns
    -------
    InstrumentScope
        The region, not yet entered.
    """
    return subject.instrument(sites, patches={})


def test_the_subjects_are_observable_and_instrumentable_by_the_declared_ports() -> None:
    panel = Panel()
    ledger = Ledger()

    assert frozenset(_published(panel)) == frozenset({STREAM})
    assert frozenset(_published(ledger)) == frozenset({CELLS, OUTSIDE})

    with _opened(panel, frozenset({STREAM})) as scope:
        panel.compute()

    assert frozenset(scope.readings) == frozenset({STREAM})

    with _opened(ledger, frozenset({CELLS})) as other:
        ledger.compute()

    assert frozenset(other.readings) == frozenset({CELLS})


def test_a_scope_keeps_a_reading_only_at_the_points_it_was_opened_over() -> None:
    ledger = Ledger()

    with ledger.instrument(frozenset({OUTSIDE}), patches={}) as scope:
        ledger.compute()

    assert frozenset(scope.readings) == frozenset({OUTSIDE})
    assert scope.readings[OUTSIDE].mask is not None


def test_a_scope_takes_no_reading_once_it_has_closed() -> None:
    ledger = Ledger()
    scope = ledger.instrument(frozenset({CELLS}), patches={})

    with scope:
        ledger.compute()

    scope.record(Read(CELLS, torch.zeros(HEIGHT, WIDTH), ROWS))

    assert torch.equal(scope.readings[CELLS].tensor, _input() * 2.0)
