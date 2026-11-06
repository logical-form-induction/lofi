from typing import TYPE_CHECKING, Final, Self, cast, override

import pytest
import torch

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
from lofi.probe import _instrument
from lofi.probe._catalog import installed_diagnostics
from lofi.probe._diagnose import DiagnosticContext
from lofi.probe._instrument import (
    Differentiated,
    GradientSites,
    Instrumented,
    differentiate,
    instrument,
)
from lofi.probe._interventions import (
    MeanFrom,
    Noise,
    PatchFrom,
    Resample,
    Shuffle,
    Sparing,
    Zero,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn.ports import Patch
    from lofi.probe._interventions import Intervention
    from lofi.probe._values import Observation, Trace

ENCODER: Final = ComponentId("toy-encoder")
OTHER: Final = ComponentId("other-encoder")

WEIGHT: Final = Site(ENCODER, Role.PARAMETER, ("scale",))
HIDDEN: Final = Site(ENCODER, Role.RECURRENT_HIDDEN, (0,))
SEALED: Final = Site(ENCODER, Role.GATE, (0,))
ELSEWHERE: Final = Site(OTHER, Role.RECURRENT_HIDDEN, (0,))

# otherwise identical parameter and activation sites
GAIN: Final = Site(ENCODER, Role.PARAMETER, ("gain",))
STREAM: Final = Site(ENCODER, Role.RESIDUAL_STREAM, (0,))
TWINNED: Final = (GAIN, STREAM)

SCORES: Final = Site(ENCODER, Role.SPLIT_DISTRIBUTION, (0,))

# patchable, but declared to pass no gradient
FROZEN: Final = Site(ENCODER, Role.GATE, (1,))

WIDTH: Final = 3
ITEMS: Final = 2

PUBLISHED: Final[SiteMap] = SiteMap.custom(
    ENCODER,
    {
        WEIGHT: SiteSpec(
            ("width",),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        HIDDEN: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        SEALED: SiteSpec(
            ("item", "width"),
            DTypeClass.PROBABILITY,
            differentiable=False,
            access=Access.OBSERVE,
        ),
        GAIN: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        STREAM: SiteSpec(
            ("item", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        SCORES: SiteSpec(
            ("item", "width"),
            DTypeClass.LOG_PROBABILITY,
            differentiable=True,
            access=Access.OBSERVE | Access.PATCH,
        ),
        FROZEN: SiteSpec(
            ("item", "width"),
            DTypeClass.PROBABILITY,
            differentiable=False,
            access=Access.OBSERVE | Access.PATCH,
        ),
    },
)


class _Held:
    __slots__ = ("_axes", "_mask", "_site", "_tensor")

    def __init__(
        self,
        site: Site,
        tensor: Tensor,
        *,
        axes: tuple[str, ...],
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


class _ToyScope:
    def __init__(
        self, model: _Toy, sites: frozenset[Site], patches: Mapping[Site, Patch[_Held]]
    ) -> None:
        self._model = model
        self._sites = sites
        self._patches = patches
        self._readings: dict[Site, _Held] = {}
        self._restore: list[Tensor] = []

    @property
    def readings(self) -> Mapping[Site, _Held]:
        return self._readings

    def observe(
        self,
        site: Site,
        tensor: Tensor,
        axes: tuple[str, ...],
        mask: Tensor | None = None,
    ) -> None:
        if site in self._sites:
            self._readings[site] = _Held(site, tensor, axes=axes, mask=mask)

    def intervene(
        self,
        site: Site,
        tensor: Tensor,
        axes: tuple[str, ...],
        mask: Tensor | None = None,
    ) -> Tensor:
        standing = self._patches.get(site)

        if standing is None:
            return tensor

        return standing.stands_in_for(_Held(site, tensor, axes=axes, mask=mask))

    def __enter__(self) -> Self:
        standing = self._patches.get(WEIGHT)

        if standing is not None:
            self._restore.append(self._model.weight)
            self._model.weight = standing.stands_in_for(
                _Held(WEIGHT, self._model.weight, axes=("width",))
            )

        self._model.scopes.append(self)
        self._model.log.append(f"enter {self._model.name}")

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> bool | None:
        for held in reversed(self._restore):
            self._model.weight = held

        self._restore.clear()
        self._model.scopes.remove(self)
        self._model.log.append(f"exit {self._model.name}")

        return None


class _Toy:
    def __init__(self, name: str, weight: Tensor, log: list[str]) -> None:
        self.name = name
        self.weight = weight
        self.log = log
        self.scopes: list[_ToyScope] = []

    def site_map(self) -> SiteMap:
        return PUBLISHED

    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _ToyScope:
        return _ToyScope(self, sites, patches)

    def forward(self, given: Tensor) -> Tensor:
        hidden = given * self.weight

        for scope in self.scopes:
            scope.observe(WEIGHT, self.weight, ("width",))
            hidden = scope.intervene(HIDDEN, hidden, ("item", "width"))
            scope.observe(HIDDEN, hidden, ("item", "width"))

        return hidden

    def twins(self, given: Tensor) -> dict[Site, Tensor]:
        computed: dict[Site, Tensor] = {}

        for site in TWINNED:
            value = given

            for scope in self.scopes:
                value = scope.intervene(site, value, ("item", "width"))
                scope.observe(site, value, ("item", "width"))

            computed[site] = value

        return computed

    def scored(self, given: Tensor, mask: Tensor) -> Tensor:
        value = given

        for scope in self.scopes:
            value = scope.intervene(SCORES, value, ("item", "width"), mask)
            scope.observe(SCORES, value, ("item", "width"), mask)

        return value

    def gated(self, given: Tensor, mask: Tensor) -> Tensor:
        for scope in self.scopes:
            scope.observe(SEALED, given, ("item", "width"), mask)

        return given


class _Misfiled(_ToyScope):
    @override
    def observe(
        self,
        site: Site,
        tensor: Tensor,
        axes: tuple[str, ...],
        mask: Tensor | None = None,
    ) -> None:
        if site in self._sites:
            self._readings[site] = _Held(ELSEWHERE, tensor, axes=axes, mask=mask)


class _Careless(_Toy):
    @override
    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _ToyScope:
        return _Misfiled(self, sites, patches)


class _Nosy(_ToyScope):
    @override
    def observe(
        self,
        site: Site,
        tensor: Tensor,
        axes: tuple[str, ...],
        mask: Tensor | None = None,
    ) -> None:
        self._readings[site] = _Held(site, tensor, axes=axes, mask=mask)


class _Curious(_Toy):
    @override
    def instrument(
        self, sites: frozenset[Site], /, *, patches: Mapping[Site, Patch[_Held]]
    ) -> _ToyScope:
        return _Nosy(self, sites, patches)


class _Spy(_Toy):
    def __init__(self, name: str, weight: Tensor, log: list[str]) -> None:
        super().__init__(name, weight, log)
        self.calls = 0

    @override
    def forward(self, given: Tensor) -> Tensor:
        self.calls += 1

        return super().forward(given)


class _Oversized:
    tag = "oversized"

    def stands_in_for(self, observation: Observation, /) -> Tensor:
        return torch.zeros(observation.rank + 1)


def a_weight() -> Tensor:
    return torch.tensor([1.0, 2.0, 3.0])


def an_input() -> Tensor:
    return torch.ones(ITEMS, WIDTH)


def a_toy(name: str = "toy", log: list[str] | None = None) -> _Toy:
    return _Toy(name, a_weight(), [] if log is None else log)


def test_a_region_reads_the_points_it_was_opened_over() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({WEIGHT, HIDDEN})) as run:
        computed = run(lambda: model.forward(an_input()))

    assert torch.equal(computed, an_input() * a_weight())
    assert set(run.trace.sites()) == {WEIGHT, HIDDEN}
    assert torch.equal(run.trace[WEIGHT].tensor, a_weight())
    assert run.trace[HIDDEN].axes == ("item", "width")


def test_instrumentation_validates_the_tagged_map_once_at_scope_entry(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = a_toy()
    calls: list[SiteMap] = []
    validate = _instrument.ensure_publishes

    def counting(published: SiteMap) -> SiteMap:
        calls.append(published)

        return validate(published)

    monkeypatch.setattr(_instrument, "ensure_publishes", counting)

    with instrument(model):
        pass

    assert calls == [PUBLISHED]


def test_gradient_sites_validate_the_tagged_map_once_for_every_region(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = a_toy()
    calls: list[SiteMap] = []
    validate = _instrument.ensure_publishes

    def counting(published: SiteMap) -> SiteMap:
        calls.append(published)

        return validate(published)

    monkeypatch.setattr(_instrument, "ensure_publishes", counting)
    sites = GradientSites(model.site_map(), frozenset({HIDDEN}))

    for _ in range(3):
        with differentiate(model, sites=sites) as run:
            run(lambda: model.forward(an_input()))

    assert calls == [PUBLISHED]


def test_an_untagged_mapping_is_refused_before_the_component_scope_opens() -> None:
    class Untagged(_Toy):
        @override
        def site_map(self) -> SiteMap:
            return cast("SiteMap", dict(PUBLISHED))

    model = Untagged("untagged", a_weight(), [])

    with (
        pytest.raises(TypeError, match=r"explicit SiteMap\.custom path"),
        instrument(model),
    ):
        pass

    assert model.log == []


def test_a_real_component_map_is_checked_from_its_own_kind_before_entry() -> None:
    class Mislabeled(_Toy):
        @override
        def site_map(self) -> SiteMap:
            return SiteMap(ENCODER, ComponentKind.PARSER, PUBLISHED)

    model = Mislabeled("mislabeled", a_weight(), [])

    with pytest.raises(ValueError, match="publishes no"), instrument(model):
        pass

    assert model.log == []


def test_a_regions_readings_feed_a_diagnostic_resolved_from_the_same_map() -> None:
    class NonRecurrent(_Toy):
        @override
        def site_map(self) -> SiteMap:
            return SiteMap.custom(ENCODER, {STREAM: PUBLISHED[STREAM]})

    model = NonRecurrent("non-recurrent", a_weight(), [])

    with instrument(model, observe=frozenset({STREAM})) as run:
        run(lambda: model.twins(an_input()))

    context = DiagnosticContext.observing(model)
    resolved = installed_diagnostics().matching(context)
    measured = {one.tag: one.run(run.trace, context=context) for one in resolved}

    assert [one.tag for one in resolved] == ["finite-readings"]
    assert measured["finite-readings"].value(STREAM, "live") == ITEMS * WIDTH
    assert measured["finite-readings"].value(STREAM, "nonfinite-share") == 0.0


def test_the_trace_is_empty_until_a_pass_has_run_and_closed_once_it_has() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({HIDDEN})) as run:
        assert len(run.trace) == 0
        assert not run.ran
        assert not run.trace.closed

        run(lambda: model.forward(an_input()))

        assert len(run.trace) == 1
        assert run.ran

    assert run.trace.closed


def test_a_point_named_for_neither_reading_nor_writing_is_left_alone() -> None:
    model = a_toy()

    with instrument(model) as run:
        computed = run(lambda: model.forward(an_input()))

    assert run.trace.sites() == ()
    assert torch.equal(computed, an_input() * a_weight())


def test_an_intervention_replaces_the_value_for_the_rest_of_the_pass() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({HIDDEN}), patch={HIDDEN: Zero()}) as run:
        computed = run(lambda: model.forward(an_input()))

    assert torch.equal(computed, torch.zeros(ITEMS, WIDTH))
    assert torch.equal(run.trace[HIDDEN].tensor, torch.zeros(ITEMS, WIDTH))


def test_a_patched_weight_is_the_one_it_was_at_entry_once_the_region_closes() -> None:
    model = a_toy()

    with instrument(model, patch={WEIGHT: Zero()}) as run:
        computed = run(lambda: model.forward(an_input()))

        assert torch.equal(model.weight, torch.zeros(WIDTH))

    assert torch.equal(computed, torch.zeros(ITEMS, WIDTH))
    assert torch.equal(model.weight, a_weight())


def a_broken_pass() -> Tensor:
    msg = "the pass did not finish"
    raise RuntimeError(msg)


def patch_and_fail(model: _Toy, opened: list[Instrumented[_Held]]) -> None:
    with instrument(model, patch={WEIGHT: Noise(0.5, 11)}) as run:
        opened.append(run)
        run(a_broken_pass)


def test_a_pass_that_raises_still_leaves_the_component_where_it_was() -> None:
    model = a_toy()
    opened: list[Instrumented[_Held]] = []

    with pytest.raises(RuntimeError, match="did not finish"):
        patch_and_fail(model, opened)

    assert torch.equal(model.weight, a_weight())
    assert model.scopes == []
    assert opened[0].trace.closed


def test_two_regions_nest_and_unwind_innermost_first() -> None:
    log: list[str] = []
    outer = a_toy("outer", log)
    inner = a_toy("inner", log)

    with instrument(outer, patch={WEIGHT: Zero()}):
        assert torch.equal(outer.weight, torch.zeros(WIDTH))

        with instrument(inner, patch={WEIGHT: Zero()}):
            assert torch.equal(inner.weight, torch.zeros(WIDTH))

        assert torch.equal(inner.weight, a_weight())
        assert torch.equal(outer.weight, torch.zeros(WIDTH))

    assert log == ["enter outer", "enter inner", "exit inner", "exit outer"]
    assert torch.equal(outer.weight, a_weight())


def raise_inside_two_regions(outer: _Toy, inner: _Toy) -> None:
    with instrument(outer, patch={WEIGHT: Zero()}):
        assert torch.equal(outer.weight, torch.zeros(WIDTH))

        with instrument(inner, patch={WEIGHT: Zero()}):
            msg = "nothing to see"
            raise ValueError(msg)


def test_a_raise_inside_the_inner_region_restores_both() -> None:
    log: list[str] = []
    outer = a_toy("outer", log)
    inner = a_toy("inner", log)

    with pytest.raises(ValueError, match="nothing to see"):
        raise_inside_two_regions(outer, inner)

    assert log == ["enter outer", "enter inner", "exit inner", "exit outer"]
    assert torch.equal(outer.weight, a_weight())
    assert torch.equal(inner.weight, a_weight())


def test_a_second_pass_in_one_region_is_refused() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({HIDDEN})) as run:
        run(lambda: model.forward(an_input()))

        with pytest.raises(RuntimeError, match="already run"):
            run(lambda: model.forward(an_input()))


def test_a_point_the_component_does_not_publish_is_refused_before_entry() -> None:
    model = a_toy()

    with (
        pytest.raises(ValueError, match="not published"),
        instrument(model, observe=frozenset({ELSEWHERE})),
    ):
        pass

    assert model.log == []


def test_a_point_that_permits_no_patching_is_refused_before_entry() -> None:
    model = a_toy()

    with (
        pytest.raises(ValueError, match="cannot be named"),
        instrument(model, patch={SEALED: Zero()}),
    ):
        pass

    assert model.log == []


def test_something_that_is_not_a_point_is_refused() -> None:
    model = a_toy()
    named = cast("frozenset[Site]", frozenset({"hidden"}))

    with pytest.raises(TypeError, match="not a Site"), instrument(model, observe=named):
        pass


def test_a_reading_filed_under_the_wrong_point_is_refused() -> None:
    model = _Careless("careless", a_weight(), [])

    with (
        pytest.raises(ValueError, match="is not where it came from"),
        instrument(model, observe=frozenset({HIDDEN})) as run,
    ):
        run(lambda: model.forward(an_input()))


def test_the_region_says_whether_it_has_run_and_what_it_holds() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({HIDDEN})) as run:
        assert isinstance(run, Instrumented)
        assert "ran=False" in repr(run)

        run(lambda: model.forward(an_input()))

    assert "ran=True" in repr(run)


def test_an_intervention_is_handed_the_numbers_the_component_published() -> None:
    model = a_toy()

    with instrument(
        model, patch={HIDDEN: Noise(0.0625, 5)}, observe=frozenset()
    ) as run:
        computed = run(lambda: model.forward(an_input()))

    drawn = torch.randn(
        (ITEMS, WIDTH),
        generator=torch.Generator().manual_seed(5),
        dtype=torch.float32,
    )

    assert torch.equal(computed, an_input() * a_weight() + 0.0625 * drawn)
    assert run.trace.sites() == ()


def a_base() -> Tensor:
    return torch.tensor([[1.0, -2.0, 3.0], [0.5, 4.0, -1.5]])


def an_other() -> Tensor:
    return torch.tensor([[7.0, 0.25, -3.0], [2.0, -0.5, 6.0]])


def a_source_trace(model: _Toy) -> Trace:
    with instrument(model, observe=frozenset(TWINNED)) as run:
        run(lambda: model.twins(an_other()))

    return run.trace


def patched_twin(model: _Toy, site: Site, intervention: Intervention) -> Tensor:
    with instrument(
        model, observe=frozenset({site}), patch={site: intervention}
    ) as run:
        computed = run(lambda: model.twins(a_base()))

    assert torch.equal(run.trace[site].tensor, computed[site])

    return computed[site]


# interventions tested at both parameter and activation sites
_VARIANTS: Final[tuple[tuple[str, Callable[[Trace], Intervention]], ...]] = (
    ("zero", lambda _source: Zero()),
    ("mean-from", MeanFrom),
    ("resample", lambda source: Resample(source, 3)),
    ("patch-from", lambda source: PatchFrom(source, GAIN)),
    ("noise", lambda _source: Noise(0.5, 7)),
    ("shuffle", lambda _source: Shuffle(11)),
    ("sparing", lambda _source: Sparing((), Zero())),
)


@pytest.mark.parametrize(
    ("tag", "build"), _VARIANTS, ids=[named for named, _ in _VARIANTS]
)
def test_a_variant_stands_in_the_same_way_at_a_weight_and_at_an_activation(
    tag: str, build: Callable[[Trace], Intervention]
) -> None:
    model = a_toy()
    source = a_source_trace(model)

    assert build(source).tag == tag

    landed = {site: patched_twin(model, site, build(source)) for site in TWINNED}

    assert torch.equal(landed[GAIN], landed[STREAM])
    assert not torch.equal(landed[GAIN], a_base())


def a_scored() -> Tensor:
    return torch.tensor([[-0.5, -1.0, -2.0], [-torch.inf, -torch.inf, -torch.inf]])


def a_live_mask() -> Tensor:
    return torch.tensor([True, False])


def test_a_patch_lands_only_where_the_mask_says_a_position_holds_data() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({SCORES}), patch={SCORES: Zero()}) as run:
        computed = run(lambda: model.scored(a_scored(), a_live_mask()))

    assert torch.equal(computed[0], torch.zeros(WIDTH))
    assert torch.equal(computed[1], torch.full((WIDTH,), -torch.inf))
    assert torch.equal(run.trace[SCORES].tensor, computed)


def test_a_masked_position_keeps_its_log_zero_where_a_mean_is_written_in() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({SCORES})) as run:
        run(lambda: model.scored(a_scored(), a_live_mask()))

    with instrument(
        model, observe=frozenset({SCORES}), patch={SCORES: MeanFrom(run.trace)}
    ) as patched:
        computed = patched(lambda: model.scored(a_scored(), a_live_mask()))

    live = a_scored()[0]
    mean = live.to(torch.float64).mean().item()

    assert torch.equal(computed[0], torch.full((WIDTH,), mean, dtype=live.dtype))
    assert torch.equal(computed[1], torch.full((WIDTH,), -torch.inf))


def test_a_reading_at_a_point_the_region_was_not_opened_over_is_refused() -> None:
    model = _Curious("curious", a_weight(), [])

    with (
        pytest.raises(ValueError, match="not opened over"),
        instrument(model, observe=frozenset({HIDDEN})) as run,
    ):
        run(lambda: model.forward(an_input()))


def patch_a_sealed_point(model: _Spy) -> None:
    with instrument(model, patch={SEALED: Zero()}) as run:
        run(lambda: model.forward(an_input()))


def test_a_patch_at_a_sealed_point_is_refused_before_the_component_runs() -> None:
    model = _Spy("spy", a_weight(), [])

    with pytest.raises(ValueError, match="cannot be named"):
        patch_a_sealed_point(model)

    assert model.calls == 0
    assert model.log == []


def run_with_an_oversized_replacement(model: _Toy) -> None:
    with instrument(model, patch={HIDDEN: cast("Intervention", _Oversized())}) as run:
        run(lambda: model.forward(an_input()))


def test_a_replacement_of_another_points_shape_is_refused() -> None:
    model = a_toy()

    with pytest.raises(ValueError, match="rather than"):
        run_with_an_oversized_replacement(model)

    assert torch.equal(model.weight, a_weight())


def test_a_bound_asked_for_at_the_region_reaches_the_trace_it_hands_back() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({WEIGHT, HIDDEN}), capacity=1) as run:
        run(lambda: model.forward(an_input()))

    assert run.trace.capacity == 1
    assert len(run.trace) == 1
    assert run.trace.evictions == 1
    assert run.trace.nbytes > 0


def test_a_region_asked_to_compress_hands_back_half_precision_readings() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({HIDDEN}), compress=True) as run:
        run(lambda: model.forward(an_input()))

    assert run.trace[HIDDEN].tensor.dtype is torch.float16


def test_a_region_keeps_every_reading_where_no_bound_was_asked_for() -> None:
    model = a_toy()

    with instrument(model, observe=frozenset({WEIGHT, HIDDEN})) as run:
        run(lambda: model.forward(an_input()))

    assert run.trace.capacity is None
    assert run.trace.evictions == 0
    assert len(run.trace) == 2
    assert run.trace[HIDDEN].tensor.dtype is torch.float32


def a_zero_filled_score() -> Tensor:
    return torch.tensor([[-0.5, -1.0, -2.0], [0.0, 0.0, 0.0]])


def a_score_above_zero() -> Tensor:
    return torch.tensor([[0.5, -1.0, -2.0], [-torch.inf, -torch.inf, -torch.inf]])


def read_the_scores(model: _Toy, given: Tensor) -> Trace:
    with instrument(model, observe=frozenset({SCORES})) as run:
        run(lambda: model.scored(given, a_live_mask()))

    return run.trace


def test_a_zero_filled_invalid_position_is_refused_where_a_point_reads_logs(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    with pytest.raises(ValueError, match=r"with 0\.0 rather than -inf"):
        read_the_scores(a_toy(), a_zero_filled_score())


def test_a_reading_above_zero_is_refused_where_a_point_reads_log_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    with pytest.raises(ValueError, match="live positions above zero"):
        read_the_scores(a_toy(), a_score_above_zero())


def test_the_refusal_names_the_point_the_departing_reading_came_from(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    with pytest.raises(ValueError, match="SPLIT_DISTRIBUTION"):
        read_the_scores(a_toy(), a_zero_filled_score())


def test_an_unchecked_region_files_a_reading_that_departs_from_its_declaration(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", False)

    trace = read_the_scores(a_toy(), a_zero_filled_score())

    assert torch.equal(trace[SCORES].tensor, a_zero_filled_score())


def test_a_reading_within_its_declaration_is_filed_the_same_way_either_run(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", False)

    unchecked = read_the_scores(a_toy(), a_scored())

    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    checked = read_the_scores(a_toy(), a_scored())

    assert torch.equal(checked[SCORES].tensor, unchecked[SCORES].tensor)
    assert checked.sites() == unchecked.sites()


def test_the_discipline_a_region_opens_under_is_chosen_as_it_opens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    model = a_toy()

    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    with instrument(model, observe=frozenset({SCORES})) as run:
        monkeypatch.setattr("lofi.probe._instrument.CHECKS", False)

        with pytest.raises(ValueError, match="rather than -inf"):
            run(lambda: model.scored(a_zero_filled_score(), a_live_mask()))


def a_zero_filled_gate() -> Tensor:
    return torch.tensor([[0.5, 0.25, 0.25], [0.0, 0.0, 0.0]])


def a_log_zero_filled_gate() -> Tensor:
    return torch.tensor([[0.5, 0.25, 0.25], [-torch.inf, -torch.inf, -torch.inf]])


def read_the_gate(model: _Toy, given: Tensor) -> Trace:
    with instrument(model, observe=frozenset({SEALED})) as run:
        run(lambda: model.gated(given, a_live_mask()))

    return run.trace


def test_a_zero_filled_invalid_position_is_filed_where_a_point_reads_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    trace = read_the_gate(a_toy(), a_zero_filled_gate())

    assert torch.equal(trace[SEALED].tensor, a_zero_filled_gate())


def test_a_log_zero_fill_is_refused_where_a_point_reads_probabilities(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)

    with pytest.raises(ValueError, match=r"with -inf rather than 0\.0"):
        read_the_gate(a_toy(), a_log_zero_filled_gate())


def test_a_reading_above_zero_is_filed_where_a_point_reads_real_numbers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)
    model = a_toy()

    with instrument(model, observe=frozenset({HIDDEN})) as run:
        run(lambda: model.forward(torch.ones(ITEMS, WIDTH)))

    assert float(run.trace[HIDDEN].tensor.max()) > 0.0


def a_squared_metric(model: _Toy) -> Tensor:
    return (model.forward(an_input()) ** 2).sum()


def test_a_differentiated_region_hands_back_the_gradient_at_each_point() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({WEIGHT, HIDDEN}))
    ) as run:
        metric = run(lambda: a_squared_metric(model))
        gradients = run.gradients(metric)

    assert set(gradients) == {WEIGHT, HIDDEN}
    assert torch.allclose(gradients[HIDDEN], 2.0 * an_input() * a_weight())
    assert torch.allclose(gradients[WEIGHT], 2.0 * ITEMS * a_weight())


def test_a_tap_leaves_the_numbers_the_region_reads_as_they_were() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({WEIGHT, HIDDEN}))
    ) as run:
        computed = run(lambda: model.forward(an_input()))

    assert torch.equal(computed.detach(), an_input() * a_weight())
    assert torch.equal(run.trace[HIDDEN].tensor, an_input() * a_weight())
    assert torch.equal(run.trace[WEIGHT].tensor, a_weight())
    assert not run.trace[HIDDEN].tensor.requires_grad
    assert torch.equal(model.weight, a_weight())


def test_a_metric_computed_from_a_tapped_point_reaches_it_with_a_gradient() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        computed = run(lambda: model.forward(an_input()))

        assert computed.requires_grad


def test_no_gradient_accumulates_on_the_component_a_region_differentiates() -> None:
    model = _Toy("toy", a_weight().requires_grad_(), [])

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({WEIGHT, HIDDEN}))
    ) as run:
        metric = run(lambda: a_squared_metric(model))
        run.gradients(metric)

    assert model.weight.grad is None
    assert torch.equal(model.weight.detach(), a_weight())


def test_a_point_the_metric_does_not_depend_on_has_a_zero_gradient() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset(TWINNED))
    ) as run:
        twins = run(lambda: model.twins(an_input()))
        gradients = run.gradients(twins[GAIN].sum())

    assert torch.equal(gradients[GAIN], torch.ones(ITEMS, WIDTH))
    assert torch.equal(gradients[STREAM], torch.zeros(ITEMS, WIDTH))


def differentiate_a_frozen_point(model: _Spy) -> None:
    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({FROZEN}))
    ) as run:
        run(lambda: model.forward(an_input()))


def test_a_non_differentiable_point_is_refused_before_the_component_runs() -> None:
    model = _Spy("spy", a_weight(), [])

    with pytest.raises(ValueError, match="non-differentiable"):
        differentiate_a_frozen_point(model)

    assert model.calls == 0
    assert model.log == []


def differentiate_a_sealed_point(model: _Spy) -> None:
    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({SEALED}))
    ) as run:
        run(lambda: model.forward(an_input()))


def test_a_point_that_permits_no_patching_cannot_be_differentiated() -> None:
    model = _Spy("spy", a_weight(), [])

    with pytest.raises(ValueError, match="cannot be named"):
        differentiate_a_sealed_point(model)

    assert model.calls == 0


def differentiate_an_unreached_point(model: _Toy) -> None:
    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        run(lambda: model.twins(an_input()))


def test_a_point_the_pass_never_reaches_is_refused_once_the_pass_has_run() -> None:
    with pytest.raises(ValueError, match="never reached"):
        differentiate_an_unreached_point(a_toy())


def test_a_gradient_is_taken_once_and_only_after_a_pass() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        with pytest.raises(RuntimeError, match="no pass has run"):
            run.gradients(torch.zeros(()))

        metric = run(lambda: a_squared_metric(model))
        run.gradients(metric)

        with pytest.raises(RuntimeError, match="already been taken"):
            run.gradients(metric)


def test_a_gradient_is_taken_of_a_differentiable_scalar_only() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        computed = run(lambda: model.forward(an_input()))

        with pytest.raises(ValueError, match="scalar"):
            run.gradients(computed)

        with pytest.raises(ValueError, match="no gradient reaches"):
            run.gradients(computed.detach().sum())


def test_a_differentiated_region_says_whether_it_has_run() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        assert isinstance(run, Differentiated)
        assert not run.ran
        assert repr(run) == "Differentiated(ran=False, taps=1)"

        run(lambda: model.forward(an_input()))

        assert run.ran
        assert len(run.trace) == 1

    assert run.trace.closed


def test_a_bound_and_a_compression_asked_of_a_differentiated_region_reach_it() -> None:
    model = a_toy()

    with differentiate(
        model,
        sites=GradientSites(model.site_map(), frozenset({WEIGHT, HIDDEN})),
        capacity=1,
        compress=True,
    ) as run:
        run(lambda: model.forward(an_input()))

    assert run.trace.capacity == 1
    assert run.trace.evictions == 1
    assert run.trace[HIDDEN].tensor.dtype is torch.float16


def an_item_mask(item: int) -> Tensor:
    return torch.arange(ITEMS) == item


def a_weighed_sum_over_reaches(model: _Toy, masks: tuple[Tensor | None, ...]) -> Tensor:
    metric = torch.zeros(())

    for weight, mask in enumerate(masks, start=1):
        reached = an_input()

        for scope in model.scopes:
            reached = scope.intervene(HIDDEN, reached, ("item", "width"), mask)

        metric = metric + weight * reached.sum()

    return metric


def test_each_position_takes_its_gradient_from_the_one_reach_that_covers_it() -> None:
    model = a_toy()
    masks = tuple(an_item_mask(item) for item in range(ITEMS))

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        metric = run(lambda: a_weighed_sum_over_reaches(model, masks))
        gradients = run.gradients(metric)

    expected = torch.arange(1.0, ITEMS + 1.0).unsqueeze(-1).expand(ITEMS, WIDTH)

    assert torch.equal(gradients[HIDDEN], expected)


def test_a_position_no_reach_covers_has_a_zero_gradient() -> None:
    model = a_toy()

    with differentiate(
        model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
    ) as run:
        metric = run(lambda: a_weighed_sum_over_reaches(model, (an_item_mask(0),)))
        gradients = run.gradients(metric)

    assert torch.equal(gradients[HIDDEN][0], torch.ones(WIDTH))
    assert torch.equal(gradients[HIDDEN][1], torch.zeros(WIDTH))


@pytest.mark.parametrize(
    "masks",
    [(None, None), (an_item_mask(0), None), (an_item_mask(1), an_item_mask(1))],
)
def test_a_point_reached_twice_at_one_position_is_refused(
    masks: tuple[Tensor | None, ...],
) -> None:
    model = a_toy()

    with (
        differentiate(
            model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
        ) as run,
        pytest.raises(RuntimeError, match=r"reached .* twice at one position"),
    ):
        run(lambda: a_weighed_sum_over_reaches(model, masks))


def test_reaches_of_one_point_at_two_shapes_are_refused() -> None:
    model = a_toy()

    def two_shapes() -> Tensor:
        (scope,) = model.scopes
        first = scope.intervene(HIDDEN, an_input(), ("item", "width"), an_item_mask(0))
        second = scope.intervene(
            HIDDEN, torch.ones(ITEMS, WIDTH + 1), ("item", "width"), an_item_mask(1)
        )

        return first.sum() + second.sum()

    with (
        differentiate(
            model, sites=GradientSites(model.site_map(), frozenset({HIDDEN}))
        ) as run,
        pytest.raises(RuntimeError, match="of shape"),
    ):
        run(two_shapes)


def test_a_differentiated_region_holds_its_readings_to_their_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr("lofi.probe._instrument.CHECKS", True)
    model = a_toy()

    with (
        differentiate(
            model, sites=GradientSites(model.site_map(), frozenset({SCORES}))
        ) as run,
        pytest.raises(ValueError, match="rather than -inf"),
    ):
        run(lambda: model.scored(a_zero_filled_score(), a_live_mask()).sum())
