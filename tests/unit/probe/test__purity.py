from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self, override

import pytest
import torch
from tests.conformance.observable import ObservableConformance, ObservableSuite
from torch import nn

from lofi._checks import LawViolationError
from lofi.nn.policy import DevicePolicy
from lofi.nn.sites import Access, ComponentId, DTypeClass, Role, Site, SiteMap, SiteSpec
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._controller_relations import ControllerRelation
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._instrument import instrument
from lofi.probe._marginals import SpanWeighting
from lofi.probe._purity import Effect, Recording, Touch

if TYPE_CHECKING:
    from collections.abc import Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._diagnose import DiagnosticResult

ENCODER: Final = ComponentId("purity-encoder")
LOGITS: Final = Site(ENCODER, Role.COMBINATOR_LOGITS, (0,))

AXES: Final = ("span", "action")
SPANS: Final = 2
ACTIONS: Final = 3

SPEC: Final = SiteSpec(
    AXES, DTypeClass.LOG_PROBABILITY, differentiable=True, access=Access.OBSERVE
)

POLICY: Final = DevicePolicy(torch.device("cpu"), torch.float32)

BELIEVED: Final = SpanWeighting(torch.tensor([0.75, 0.25]), axes=("span",))


class Read(NamedTuple):
    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class Settings:
    def __init__(self, order: str) -> None:
        self.order = order


class Scope:
    def __init__(self, model: Encoder, sites: frozenset[Site]) -> None:
        self._model = model
        self._sites = sites
        self._readings: dict[Site, Read] = {}

    @property
    def readings(self) -> Mapping[Site, Read]:
        return MappingProxyType(self._readings)

    def record(self, reading: Read) -> None:
        if reading.site in self._sites:
            self._readings[reading.site] = reading

    def __enter__(self) -> Self:
        self._model.open.append(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        self._model.open.remove(self)


class Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        torch.manual_seed(5)
        self.settings = Settings("breadth-first")
        self._layer = nn.Linear(ACTIONS, ACTIONS)
        self.register_buffer(
            "_rows", torch.linspace(-1.0, 1.0, SPANS * ACTIONS).reshape(SPANS, ACTIONS)
        )
        self.open: list[Scope] = []

    def site_map(self) -> SiteMap:
        return SiteMap.custom(ENCODER, {LOGITS: SPEC})

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[Read]]
    ) -> Scope:
        if patches:
            msg = "this encoder publishes nothing that may be patched"
            raise ValueError(msg)

        return Scope(self, sites)

    def compute(self) -> Tensor:
        scored = self._layer(self.get_buffer("_rows")).log_softmax(dim=-1)

        for scope in self.open:
            scope.record(Read(LOGITS, scored.detach(), AXES))

        return scored


class Traveling(Encoder):
    @override
    def compute(self) -> Tensor:
        self.to("meta")

        return super().compute()


def _placements(model: Encoder) -> tuple[str, ...]:
    return tuple(
        str(tensor.device) for tensor in (*model.parameters(), *model.buffers())
    )


def _probe(model: Encoder) -> tuple[Tensor, Mapping[str, DiagnosticResult]]:
    published = model.site_map()

    with instrument(model, observe=frozenset(published)) as region:
        computed = region(model.compute)

    context = DiagnosticContext(
        published,
        relations={
            LOGITS: ControllerRelation(
                SPANS,
                sequences=((0, 1),),
                similar_inputs=((0, 1),),
            )
        },
        weighting=BELIEVED,
    )
    measured = {
        one.tag: one.run(region.trace, context=context)
        for one in installed_diagnostics().matching(context)
    }

    return computed, measured


def test_a_stand_in_reads_through_to_the_component_it_stands_in_for() -> None:
    encoder = Encoder()
    recording = Recording(encoder, POLICY)
    watched = recording.stand_in

    assert watched.settings.order == "breadth-first"
    assert watched.site_map() == encoder.site_map()
    assert torch.equal(watched.compute(), encoder.compute())
    assert recording.touches == ()


def test_a_write_reached_through_an_attribute_is_recorded_under_the_path_it_took() -> (
    None
):
    encoder = Encoder()
    recording = Recording(encoder, POLICY)

    recording.stand_in.settings.order = "depth-first"

    assert recording.touches == (
        Touch(("settings", "order"), Effect.ASSIGNMENT, holder="Settings"),
    )
    assert encoder.settings.order == "depth-first"


def test_a_write_to_the_component_itself_is_recorded_under_its_own_name() -> None:
    encoder = Encoder()
    recording = Recording(encoder, POLICY)

    recording.stand_in.settings = Settings("depth-first")

    assert recording.touches == (
        Touch(("settings",), Effect.ASSIGNMENT, holder="Encoder"),
    )


def test_a_deletion_reached_through_an_attribute_is_recorded_as_a_deletion() -> None:
    encoder = Encoder()
    recording = Recording(encoder, POLICY)

    del recording.stand_in.settings.order

    assert recording.touches == (
        Touch(("settings", "order"), Effect.DELETION, holder="Settings"),
    )
    assert not hasattr(encoder.settings, "order")


def test_a_move_made_through_the_stand_in_is_recorded_as_a_placement() -> None:
    recording = Recording(Encoder(), POLICY)

    recording.stand_in.to("meta")

    assert recording.touches == (Touch(("to",), Effect.PLACEMENT, holder="Encoder"),)


def test_a_submodule_added_through_the_stand_in_is_recorded_as_a_registration() -> None:
    encoder = Encoder()
    recording = Recording(encoder, POLICY)

    recording.stand_in.add_module("extra", nn.Linear(ACTIONS, ACTIONS))

    assert recording.touches == (
        Touch(("add_module",), Effect.REGISTRATION, holder="Encoder"),
    )
    assert "extra" in dict(encoder.named_modules())


def test_reading_a_method_without_calling_it_records_nothing() -> None:
    recording = Recording(Encoder(), POLICY)

    assert recording.stand_in.to is not None
    assert recording.touches == ()


def test_a_component_that_is_not_where_the_run_placed_it_is_refused() -> None:
    elsewhere = DevicePolicy(torch.device("meta"), torch.float32)

    with pytest.raises(ValueError, match="this run is on meta"):
        Recording(Encoder(), elsewhere)


def test_a_block_that_touched_the_component_reports_what_was_touched() -> None:
    recording = Recording(Encoder(), POLICY)

    with (
        pytest.raises(LawViolationError, match=r"assignment at settings\.order"),
        recording as watched,
    ):
        watched.settings.order = "depth-first"

    assert recording.touches == (
        Touch(("settings", "order"), Effect.ASSIGNMENT, holder="Settings"),
    )


def test_a_block_that_left_the_component_elsewhere_is_caught_at_its_close() -> None:
    recording = Recording(Traveling(), POLICY)

    with (
        pytest.raises(ValueError, match="is on meta while this run is on cpu"),
        recording as watched,
    ):
        watched.compute()

    assert recording.touches == ()


def _failing(recording: Recording[Encoder]) -> None:
    with recording as watched:
        watched.settings.order = "depth-first"

        raise ZeroDivisionError


def test_a_failing_block_reports_what_failed_rather_than_what_it_touched() -> None:
    recording = Recording(Encoder(), POLICY)

    with pytest.raises(ZeroDivisionError):
        _failing(recording)

    assert len(recording.touches) == 1


def test_a_diagnostic_run_through_the_stand_in_touches_the_component_nowhere() -> None:
    encoder = Encoder()
    placed = _placements(encoder)
    recording = Recording(encoder, POLICY)

    with recording as watched:
        computed, measured = _probe(watched)

    assert recording.touches == ()
    assert set(measured) == {
        "action-margin",
        "controller-choice",
        "controller-relations",
        "finite-readings",
    }
    assert measured["finite-readings"].value(LOGITS, "nonfinite-share") == 0.0
    assert measured["action-margin"].value(LOGITS, "rows") == float(SPANS)
    assert measured["controller-choice"].value(LOGITS, "rows") == float(SPANS)
    assert measured["controller-relations"].value(LOGITS, "sequences") == 1.0
    assert torch.equal(computed, encoder.compute())
    assert _placements(encoder) == placed
    assert encoder.settings.order == "breadth-first"
    assert frozenset(name for name, _ in encoder.named_modules()) == frozenset(
        {"", "_layer"}
    )


def test_a_touch_is_equal_to_another_taken_at_the_same_path_by_the_same_effect() -> (
    None
):
    one = Touch(("settings", "order"), Effect.ASSIGNMENT, holder="Settings")
    same = Touch(("settings", "order"), Effect.ASSIGNMENT, holder="Settings")
    deeper = Touch(("settings",), Effect.ASSIGNMENT, holder="Settings")

    assert one == same
    assert hash(one) == hash(same)
    assert one != deeper
    assert one != Touch(("settings", "order"), Effect.DELETION, holder="Settings")
    assert one.path == ("settings", "order")
    assert one.effect is Effect.ASSIGNMENT
    assert one.holder == "Settings"
    assert one != ("settings", "order")


def _conformance() -> ObservableConformance[Encoder, Read]:
    return ObservableConformance(
        build=Encoder,
        scope=lambda subject, sites: subject.instrument(sites, patches={}),
        run=lambda subject: (_probe(subject)[0],),
        configuration=lambda subject: {"order": subject.settings.order},
        submodules=lambda subject: frozenset(
            name for name, _ in subject.named_modules()
        ),
        placements=_placements,
    )


class TestAProbeRunLeavesTheComponentAlone(ObservableSuite):
    @pytest.fixture
    def conformance(self) -> ObservableConformance[Encoder, Read]:
        return _conformance()


def test_the_conformance_reads_the_configuration_the_purity_law_compares() -> None:
    conformance = _conformance()
    subject = conformance.build()

    assert conformance.configuration(subject) == {"order": "breadth-first"}
    assert conformance.submodules(subject) == frozenset({"", "_layer"})
    assert conformance.placements(subject) == ("cpu", "cpu", "cpu")


def _meddling(model: Encoder) -> tuple[Tensor, ...]:
    computed, _ = _probe(model)
    model.settings.order = f"{model.settings.order}-swept"

    return (computed,)


def test_the_purity_law_goes_red_where_the_probe_writes_a_configuration_field() -> None:
    suite = ObservableSuite()

    with pytest.raises(LawViolationError, match="the configuration was"):
        suite.test_observing_leaves_the_observed_alone(
            _conformance()._replace(run=_meddling)
        )
