from typing import TYPE_CHECKING, Final, NoReturn, cast, override

import pytest
import torch
from torch import nn

from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteSpec,
)
from lofi.probe._boundary import ExternalModelSpec
from lofi.probe._causal import AlignmentPolicy, NamedVariable, TypeAt
from lofi.probe._external import (
    TORCH_HOOKS,
    Attachment,
    ExternalJointSubject,
    ExternalModel,
    ExternalScope,
    ExternalSite,
    ExternalSiteMap,
    Hooking,
    Output,
    TorchHooks,
)
from lofi.probe._instrument import instrument
from lofi.probe._interventions import Zero
from lofi.probe._roles import ROLE_AXES, ROLE_DTYPE
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from collections.abc import Mapping, MutableMapping, Sequence

    from torch import Tensor

    from lofi.probe._causal import HighLevelVariable
    from lofi.probe._external import Recorder, Release

WIDTH: Final = 4
INNER: Final = 8
BATCH: Final = 2
POSITIONS: Final = 3
LAYERS: Final = 2

GPT: Final = ComponentId("gpt2-toy")
BERT: Final = ComponentId("bert-toy")
LLAMA: Final = ComponentId("llama-toy")

STREAM_AXES: Final = ROLE_AXES[Role.RESIDUAL_STREAM]
STREAM_SPEC: Final = SiteSpec(
    STREAM_AXES,
    ROLE_DTYPE[Role.RESIDUAL_STREAM],
    differentiable=True,
    access=Access.OBSERVE | Access.PATCH,
)


class _Mlp(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.c_fc = nn.Linear(WIDTH, INNER)
        self.c_proj = nn.Linear(INNER, WIDTH)

    @override
    def forward(self, given: Tensor) -> Tensor:
        return self.c_proj(torch.relu(self.c_fc(given)))


class _Block(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.mlp = _Mlp()

    @override
    def forward(self, given: Tensor) -> tuple[Tensor, None]:
        return (given + self.mlp(given), None)


class _Trunk(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.h = nn.ModuleList(_Block() for _ in range(LAYERS))

    def block(self, index: int) -> _Block:
        found = self.h[index]
        assert isinstance(found, _Block)

        return found

    @override
    def forward(self, given: Tensor) -> Tensor:
        for block in self.h:
            given = block(given)[0]

        return given


class _Gpt(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.transformer = _Trunk()

    @override
    def forward(self, given: Tensor) -> Tensor:
        return self.transformer(given)


class _FailingGpt(_Gpt):
    @override
    def forward(self, given: Tensor) -> NoReturn:
        del given
        message = "the forward pass failed"

        raise RuntimeError(message)


class _Intermediate(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(WIDTH, INNER)

    @override
    def forward(self, given: Tensor) -> Tensor:
        return torch.relu(self.dense(given))


class _Output(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.dense = nn.Linear(INNER, WIDTH)

    @override
    def forward(self, given: Tensor, residual: Tensor) -> Tensor:
        return self.dense(given) + residual


class _Layer(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.intermediate = _Intermediate()
        self.output = _Output()

    @override
    def forward(self, given: Tensor) -> Tensor:
        return self.output(self.intermediate(given), given)


class _Encoder(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layer = nn.ModuleList(_Layer() for _ in range(LAYERS))

    def at(self, index: int) -> _Layer:
        found = self.layer[index]
        assert isinstance(found, _Layer)

        return found

    @override
    def forward(self, given: Tensor) -> Tensor:
        for layer in self.layer:
            given = layer(given)

        return given


class _Bert(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.encoder = _Encoder()

    @override
    def forward(self, given: Tensor) -> Tensor:
        return self.encoder(given)


class _LlamaBody(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.layers = nn.ModuleList(_Layer() for _ in range(LAYERS))

    @override
    def forward(self, given: Tensor) -> Tensor:
        for layer in self.layers:
            given = layer(given)

        return given


class _Llama(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.model = _LlamaBody()

    @override
    def forward(self, given: Tensor) -> Tensor:
        return self.model(given)


class _Constant:
    def __init__(self, fill: float) -> None:
        self._fill = fill

    def stands_in_for(self, reading: Observation, /) -> Tensor:
        return torch.full_like(reading.tensor, self._fill)


class _Counting:
    def __init__(self) -> None:
        self.installed: list[tuple[Site, str]] = []
        self.released = 0

    def install(self, attached: Sequence[Attachment], record: Recorder, /) -> Release:
        self.installed.extend((one.site, one.path) for one in attached)
        del record

        def release() -> None:
            self.released += 1

        return release


class _Opening:
    def __init__(self) -> None:
        self.attached: tuple[Attachment, ...] = ()
        self.record: Recorder | None = None

    def install(self, attached: Sequence[Attachment], record: Recorder, /) -> Release:
        self.attached = tuple(attached)
        self.record = record

        return lambda: None


def an_input() -> Tensor:
    return torch.linspace(-1.0, 1.0, BATCH * POSITIONS * WIDTH).reshape(
        BATCH, POSITIONS, WIDTH
    )


def a_gpt_map() -> ExternalSiteMap:
    return ExternalSiteMap(
        GPT,
        tuple(
            site
            for layer in range(LAYERS)
            for site in (
                ExternalSite(
                    f"transformer.h.{layer}",
                    Role.RESIDUAL_STREAM,
                    layer,
                    STREAM_SPEC,
                    select=0,
                ),
                ExternalSite(
                    f"transformer.h.{layer}.mlp.c_fc",
                    Role.FFN_PREACTIVATION,
                    layer,
                    SiteSpec(
                        ROLE_AXES[Role.FFN_PREACTIVATION],
                        ROLE_DTYPE[Role.FFN_PREACTIVATION],
                        differentiable=True,
                        access=Access.OBSERVE | Access.PATCH,
                    ),
                ),
            )
        ),
    )


def a_bert_map() -> ExternalSiteMap:
    return ExternalSiteMap(
        BERT,
        tuple(
            site
            for layer in range(LAYERS)
            for site in (
                ExternalSite(
                    f"encoder.layer.{layer}.output",
                    Role.RESIDUAL_STREAM,
                    layer,
                    STREAM_SPEC,
                ),
                ExternalSite(
                    f"encoder.layer.{layer}.intermediate.dense",
                    Role.FFN_PREACTIVATION,
                    layer,
                    SiteSpec(
                        ROLE_AXES[Role.FFN_PREACTIVATION],
                        ROLE_DTYPE[Role.FFN_PREACTIVATION],
                        differentiable=True,
                        access=Access.OBSERVE | Access.PATCH,
                    ),
                ),
            )
        ),
    )


def stream(model: ComponentId, layer: int) -> Site:
    return Site(model, Role.RESIDUAL_STREAM, (layer,))


def preactivation(model: ComponentId, layer: int) -> Site:
    return Site(model, Role.FFN_PREACTIVATION, (layer,))


def hooked(model: nn.Module) -> int:
    return sum(len(module._forward_hooks) for _, module in model.named_modules())  # noqa: SLF001


def hooks_by_path(model: nn.Module) -> dict[str, int]:
    return {
        name: len(module._forward_hooks)  # noqa: SLF001
        for name, module in model.named_modules()
        if module._forward_hooks  # noqa: SLF001
    }


def test_a_module_path_that_is_empty_is_refused() -> None:
    with pytest.raises(ValueError, match="module path is empty"):
        ExternalSite("", Role.RESIDUAL_STREAM, 0, STREAM_SPEC)


def test_a_module_path_holding_an_empty_segment_is_refused() -> None:
    with pytest.raises(ValueError, match=r"empty segment: 'transformer\.\.h'"):
        ExternalSite("transformer..h", Role.RESIDUAL_STREAM, 0, STREAM_SPEC)


def test_a_declared_point_whose_role_is_not_a_role_is_refused() -> None:
    with pytest.raises(TypeError, match="role is a Role"):
        ExternalSite("transformer.h.0", "RESIDUAL_STREAM", 0, STREAM_SPEC)  # ty: ignore[invalid-argument-type]


def test_a_declared_point_holding_no_specification_is_refused() -> None:
    with pytest.raises(TypeError, match="not a SiteSpec"):
        ExternalSite("transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_AXES)  # ty: ignore[invalid-argument-type]


def test_a_declared_point_in_a_negative_layer_is_refused() -> None:
    with pytest.raises(ValueError, match="layer is negative"):
        ExternalSite("transformer.h.0", Role.RESIDUAL_STREAM, -1, STREAM_SPEC)


def test_a_declared_point_at_a_negative_position_is_refused() -> None:
    with pytest.raises(ValueError, match="position is negative"):
        ExternalSite("transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=-1)


def test_two_declared_points_agreeing_on_every_field_are_equal() -> None:
    one = ExternalSite(
        "transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
    )
    other = ExternalSite(
        "transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
    )

    assert one == other
    assert hash(one) == hash(other)


def test_two_declared_points_differing_in_a_field_are_not_equal() -> None:
    one = ExternalSite(
        "transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
    )

    assert one != ExternalSite(
        "transformer.h.1", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
    )
    assert one != ExternalSite(
        "transformer.h.0", Role.RESIDUAL_STREAM, 1, STREAM_SPEC, select=0
    )
    assert one != ExternalSite("transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC)
    assert one != ExternalSite(
        "transformer.h.0", Role.FFN_PREACTIVATION, 0, STREAM_SPEC, select=0
    )


def test_a_declared_point_repeats_its_constructor_call() -> None:
    declared = ExternalSite(
        "transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
    )

    assert repr(declared).startswith(
        "ExternalSite('transformer.h.0', Role.RESIDUAL_STREAM, 0, "
    )
    assert repr(declared).endswith("select=0)")


def test_an_external_map_without_a_model_name_is_refused() -> None:
    with pytest.raises(ValueError, match="model is empty"):
        ExternalSiteMap(ComponentId(""), ())


def test_an_external_map_holding_something_other_than_a_declaration_is_refused() -> (
    None
):
    with pytest.raises(TypeError, match="not a declaration"):
        ExternalSiteMap(GPT, (STREAM_SPEC,))  # ty: ignore[invalid-argument-type]


def test_an_external_map_declaring_one_point_twice_names_both_paths() -> None:
    first = ExternalSite("transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC)
    again = ExternalSite("transformer.wte", Role.RESIDUAL_STREAM, 0, STREAM_SPEC)

    with pytest.raises(ValueError, match=r"transformer\.h\.0 and at transformer\.wte"):
        ExternalSiteMap(GPT, (first, again))


def test_an_external_map_publishes_one_point_per_declaration_keyed_by_layer() -> None:
    declared = a_gpt_map()

    assert set(declared.published) == {
        stream(GPT, 0),
        stream(GPT, 1),
        preactivation(GPT, 0),
        preactivation(GPT, 1),
    }
    assert declared.published[stream(GPT, 1)].axes == STREAM_AXES


def test_an_external_map_keeps_the_order_its_points_were_declared_in() -> None:
    assert [one.path for one in a_gpt_map().sites] == [
        "transformer.h.0",
        "transformer.h.0.mlp.c_fc",
        "transformer.h.1",
        "transformer.h.1.mlp.c_fc",
    ]


def test_an_external_map_returns_the_declaration_one_point_was_made_under() -> None:
    found = a_gpt_map().declaration(stream(GPT, 1))

    assert found.path == "transformer.h.1"
    assert found.select == 0


def test_an_external_map_refuses_a_point_it_declares_nothing_at() -> None:
    with pytest.raises(ValueError, match="declares no Site"):
        a_gpt_map().declaration(Site(GPT, Role.GATE, (0,)))


def test_two_external_maps_declaring_the_same_points_for_one_model_are_equal() -> None:
    assert a_gpt_map() == a_gpt_map()
    assert hash(a_gpt_map()) == hash(a_gpt_map())
    assert a_gpt_map() != a_bert_map()


def test_an_external_map_reports_its_model_and_how_much_it_declares() -> None:
    assert repr(a_gpt_map()) == "ExternalSiteMap('gpt2-toy', 4 sites)"


def test_a_declared_specification_yields_observations_over_the_declared_axes() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())
    watched = frozenset({stream(GPT, 1), preactivation(GPT, 0)})

    with instrument(external, observe=watched) as region:
        region(lambda: model(an_input()))

    assert set(region.trace) == watched
    assert region.trace[stream(GPT, 1)].axes == STREAM_AXES
    assert tuple(region.trace[stream(GPT, 1)].tensor.shape) == (BATCH, POSITIONS, WIDTH)
    assert region.trace[preactivation(GPT, 0)].axes == ROLE_AXES[Role.FFN_PREACTIVATION]
    assert tuple(region.trace[preactivation(GPT, 0)].tensor.shape) == (
        BATCH,
        POSITIONS,
        INNER,
    )


def test_a_bert_declaration_yields_observations_over_the_declared_axes() -> None:
    model = _Bert()
    external = ExternalModel(model, a_bert_map())
    watched = frozenset({stream(BERT, 0), preactivation(BERT, 1)})

    with instrument(external, observe=watched) as region:
        region(lambda: model(an_input()))

    assert set(region.trace) == watched
    assert region.trace[stream(BERT, 0)].axes == STREAM_AXES
    assert tuple(region.trace[stream(BERT, 0)].tensor.shape) == (
        BATCH,
        POSITIONS,
        WIDTH,
    )


def test_the_reading_at_a_positioned_point_is_what_the_next_layer_was_handed() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())
    given = an_input()

    with instrument(external, observe=frozenset({stream(GPT, 0)})) as region:
        region(lambda: model(given))

    assert torch.equal(
        region.trace[stream(GPT, 0)].tensor, model.transformer.h[0](given)[0]
    )


def test_a_declaration_naming_fewer_axes_than_the_value_has_is_refused_on_reading() -> (
    None
):
    model = _Gpt()
    thin = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.0",
                Role.RESIDUAL_STREAM,
                0,
                SiteSpec(
                    ("batch", "feature"),
                    DTypeClass.REAL,
                    differentiable=True,
                    access=Access.OBSERVE,
                ),
                select=0,
            ),
        ),
    )
    external = ExternalModel(model, thin)

    with (
        pytest.raises(ValueError, match="rank 3 needs 3 axis names"),
        instrument(external, observe=frozenset({stream(GPT, 0)})) as region,
    ):
        region(lambda: model(an_input()))


def test_a_whole_output_point_whose_module_returns_a_tuple_is_refused() -> None:
    model = _Gpt()
    mistaken = ExternalSiteMap(
        GPT, (ExternalSite("transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC),)
    )
    external = ExternalModel(model, mistaken)

    with (
        pytest.raises(TypeError, match="as its whole output"),
        instrument(external, observe=frozenset({stream(GPT, 0)})) as region,
    ):
        region(lambda: model(an_input()))


def test_a_point_declared_at_a_position_whose_module_returns_a_tensor_is_refused() -> (
    None
):
    model = _Gpt()
    mistaken = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.0.mlp.c_fc",
                Role.RESIDUAL_STREAM,
                0,
                STREAM_SPEC,
                select=0,
            ),
        ),
    )
    external = ExternalModel(model, mistaken)

    with (
        pytest.raises(TypeError, match="at position 0 of its output"),
        instrument(external, observe=frozenset({stream(GPT, 0)})) as region,
    ):
        region(lambda: model(an_input()))


def test_a_point_declared_beyond_the_end_of_its_module_output_is_refused() -> None:
    model = _Gpt()
    mistaken = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=2
            ),
        ),
    )
    external = ExternalModel(model, mistaken)

    with (
        pytest.raises(ValueError, match="it returned 2 positions"),
        instrument(external, observe=frozenset({stream(GPT, 0)})) as region,
    ):
        region(lambda: model(an_input()))


def test_a_point_declared_where_its_module_holds_no_tensor_is_refused() -> None:
    model = _Gpt()
    mistaken = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.0", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=1
            ),
        ),
    )
    external = ExternalModel(model, mistaken)

    with (
        pytest.raises(TypeError, match="holds None at position 1"),
        instrument(external, observe=frozenset({stream(GPT, 0)})) as region,
    ):
        region(lambda: model(an_input()))


def test_a_patch_at_a_positioned_point_replaces_what_the_next_layer_reads() -> None:
    model = _Gpt()

    with instrument(
        ExternalModel(model, a_gpt_map()),
        observe=frozenset({preactivation(GPT, 1)}),
        patch={stream(GPT, 0): Zero()},
    ) as region:
        region(lambda: model(an_input()))

    expected = model.transformer.block(1).mlp.c_fc(torch.zeros(BATCH, POSITIONS, WIDTH))

    assert torch.allclose(region.trace[preactivation(GPT, 1)].tensor, expected)


def test_a_patch_at_a_whole_output_point_replaces_what_the_pass_carries_on_with() -> (
    None
):
    model = _Bert()

    with instrument(
        ExternalModel(model, a_bert_map()),
        observe=frozenset({preactivation(BERT, 1)}),
        patch={stream(BERT, 0): Zero()},
    ) as region:
        region(lambda: model(an_input()))

    expected = model.encoder.at(1).intermediate.dense(
        torch.zeros(BATCH, POSITIONS, WIDTH)
    )

    assert torch.allclose(region.trace[preactivation(BERT, 1)].tensor, expected)


def test_a_patch_leaves_every_other_position_of_the_output_as_it_stood() -> None:
    model = _Gpt()
    carried: list[object] = []

    with instrument(
        ExternalModel(model, a_gpt_map()), patch={stream(GPT, 0): Zero()}
    ) as region:
        watcher = model.transformer.block(0).register_forward_hook(
            lambda _m, _a, out: carried.append(out)
        )

        try:
            region(lambda: model(an_input()))
        finally:
            watcher.remove()

    assert len(carried) == 1
    held = carried[0]
    assert isinstance(held, tuple)
    assert len(held) == 2
    assert torch.equal(held[0], torch.zeros(BATCH, POSITIONS, WIDTH))
    assert held[1] is None


def test_the_reading_kept_at_a_patched_point_is_the_value_the_patch_put_there() -> None:
    model = _Bert()

    with instrument(
        ExternalModel(model, a_bert_map()),
        observe=frozenset({stream(BERT, 0)}),
        patch={stream(BERT, 0): Zero()},
    ) as region:
        region(lambda: model(an_input()))

    assert torch.equal(
        region.trace[stream(BERT, 0)].tensor,
        torch.zeros(BATCH, POSITIONS, WIDTH),
    )


def test_a_point_reached_only_to_patch_it_leaves_no_reading() -> None:
    model = _Bert()

    with instrument(
        ExternalModel(model, a_bert_map()), patch={stream(BERT, 0): Zero()}
    ) as region:
        region(lambda: model(an_input()))

    assert len(region.trace) == 0


def test_a_region_leaves_the_model_as_it_stood_on_a_normal_exit() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())

    assert hooked(model) == 0

    with instrument(external, observe=frozenset({stream(GPT, 0)})) as region:
        region(lambda: model(an_input()))
        assert hooked(model) == 1

    assert hooked(model) == 0


def test_a_region_leaves_the_model_as_it_stood_when_the_body_raises() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())
    refusal = RuntimeError("the body raised")

    with (
        pytest.raises(RuntimeError, match="the body raised"),
        instrument(external, observe=frozenset({stream(GPT, 0)})),
    ):
        raise refusal

    assert hooked(model) == 0


def test_an_error_from_the_forward_pass_propagates_out_of_the_region() -> None:
    model = _FailingGpt()
    external = ExternalModel(model, a_gpt_map())

    with (
        instrument(external, observe=frozenset({stream(GPT, 0)})) as region,
        pytest.raises(RuntimeError, match="the forward pass failed"),
    ):
        region(lambda: model(an_input()))

    assert region.trace.sites() == ()
    assert hooked(model) == 0


def test_nested_regions_over_one_model_both_restore_it() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())

    with instrument(external, observe=frozenset({stream(GPT, 0)})) as outer:
        with instrument(external, observe=frozenset({stream(GPT, 1)})) as inner:
            assert hooked(model) == 2
            inner(lambda: model(an_input()))

        assert hooked(model) == 1
        outer(lambda: model(an_input()))

    assert hooked(model) == 0


def test_a_region_that_is_already_open_refuses_to_open_again() -> None:
    scope = ExternalScope((), TORCH_HOOKS)

    with scope, pytest.raises(RuntimeError, match="already open"):
        scope.__enter__()


def test_a_region_that_never_opened_refuses_to_close() -> None:
    scope = ExternalScope((), TORCH_HOOKS)

    with pytest.raises(RuntimeError, match="never opened"):
        scope.__exit__(None, None, None)


def test_a_region_reports_how_many_points_it_holds_and_whether_it_is_open() -> None:
    scope = ExternalScope((), TORCH_HOOKS)

    assert repr(scope) == "ExternalScope(0 sites, open=False)"

    with scope:
        assert repr(scope) == "ExternalScope(0 sites, open=True)"


def test_a_foreign_model_that_is_not_a_module_is_refused() -> None:
    with pytest.raises(TypeError, match="is a torch module"):
        ExternalModel(a_gpt_map(), a_gpt_map())  # ty: ignore[invalid-argument-type]


def test_a_foreign_model_published_through_something_other_than_a_map_is_refused() -> (
    None
):
    with pytest.raises(TypeError, match="not an external map"):
        ExternalModel(_Gpt(), a_gpt_map().published)  # ty: ignore[invalid-argument-type]


def test_a_foreign_model_read_through_something_that_installs_nothing_is_refused() -> (
    None
):
    with pytest.raises(TypeError, match="installs none"):
        ExternalModel(_Gpt(), a_gpt_map(), hooking=object())  # ty: ignore[invalid-argument-type]


def test_a_declaration_naming_a_module_the_model_does_not_hold_is_refused() -> None:
    model = _Gpt()
    missing = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.7", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
            ),
        ),
    )
    external = ExternalModel(model, missing)

    with (
        pytest.raises(ValueError, match=r"'transformer\.h\.7'"),
        instrument(external, observe=frozenset({stream(GPT, 0)})),
    ):
        pass

    assert hooked(model) == 0


def test_a_declaration_naming_part_of_a_path_is_refused_rather_than_matched() -> None:
    model = _Gpt()
    partial = ExternalSiteMap(
        GPT,
        (ExternalSite("transformer.h.0.ml", Role.RESIDUAL_STREAM, 0, STREAM_SPEC),),
    )

    with pytest.raises(ValueError, match=r"'transformer\.h\.0\.ml'"):
        ExternalModel(model, partial).instrument(
            frozenset({stream(GPT, 0)}), patches={}
        )


def test_a_declaration_extending_a_module_path_is_refused_rather_than_matched() -> None:
    model = _Gpt()
    extended = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.0.mlp.c_fc.weight",
                Role.FFN_PREACTIVATION,
                0,
                STREAM_SPEC,
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"c_fc\.weight'"):
        ExternalModel(model, extended).instrument(
            frozenset({preactivation(GPT, 0)}), patches={}
        )


def test_a_region_over_a_point_the_declaration_does_not_name_is_refused() -> None:
    external = ExternalModel(_Gpt(), a_gpt_map())

    with pytest.raises(ValueError, match="declares no Site"):
        external.instrument(frozenset({Site(GPT, Role.GATE, (0,))}), patches={})


def test_a_foreign_model_publishes_what_its_declaration_holds() -> None:
    declared = a_gpt_map()
    published = ExternalModel(_Gpt(), declared).site_map()

    assert published == declared.published
    assert published.component == GPT
    assert published.kind is ComponentKind.CUSTOM


def test_a_foreign_model_reports_the_declaration_it_is_read_through() -> None:
    external = ExternalModel(_Gpt(), a_gpt_map())

    assert repr(external) == "ExternalModel(ExternalSiteMap('gpt2-toy', 4 sites))"
    assert external.declared == a_gpt_map()


def test_a_region_installs_its_reads_through_the_installer_it_was_given() -> None:
    counting = _Counting()
    external = ExternalModel(_Gpt(), a_gpt_map(), hooking=counting)

    with external.instrument(frozenset({stream(GPT, 0)}), patches={}):
        assert counting.installed == [(stream(GPT, 0), "transformer.h.0")]
        assert counting.released == 0

    assert counting.released == 1


def test_an_installer_is_handed_one_attachment_per_point_it_is_to_reach() -> None:
    opening = _Opening()
    external = ExternalModel(_Gpt(), a_gpt_map(), hooking=opening)
    patch = _Constant(1.0)

    with external.instrument(
        frozenset({preactivation(GPT, 0)}), patches={stream(GPT, 1): patch}
    ):
        pass

    by_site = {one.site: one for one in opening.attached}

    assert set(by_site) == {preactivation(GPT, 0), stream(GPT, 1)}
    assert by_site[preactivation(GPT, 0)].observed
    assert by_site[preactivation(GPT, 0)].patch is None
    assert by_site[preactivation(GPT, 0)].select is None
    assert not by_site[stream(GPT, 1)].observed
    assert by_site[stream(GPT, 1)].patch is patch
    assert by_site[stream(GPT, 1)].select == 0
    assert by_site[stream(GPT, 1)].axes == STREAM_AXES


def test_the_installer_files_a_reading_under_the_point_it_was_taken_at() -> None:
    opening = _Opening()
    external = ExternalModel(_Gpt(), a_gpt_map(), hooking=opening)

    with external.instrument(frozenset({stream(GPT, 0)}), patches={}) as scope:
        assert opening.record is not None
        opening.record(
            Observation(stream(GPT, 0), torch.zeros(1, 1, 1), axes=STREAM_AXES)
        )

        assert set(scope.readings) == {stream(GPT, 0)}


def test_the_default_installer_repeats_its_constructor_call() -> None:
    assert repr(TORCH_HOOKS) == "TorchHooks()"
    assert isinstance(TORCH_HOOKS, TorchHooks)
    assert isinstance(TORCH_HOOKS, Hooking)


def test_every_inline_point_is_read_write_and_differentiable() -> None:
    for declared in a_gpt_map().sites:
        assert declared.spec.access == Access.OBSERVE | Access.PATCH
        assert declared.spec.differentiable


def test_a_llama_family_model_declared_inline_is_probed() -> None:
    declared = ExternalModelSpec.model_validate(
        {
            "model": "llama-toy",
            "sites": [
                {
                    "path": "model.layers.1.output",
                    "role": "RESIDUAL_STREAM",
                    "layer": 1,
                    "axes": ["batch", "position", "feature"],
                    "dtype_class": "real",
                    "differentiable": True,
                    "access": ["OBSERVE", "PATCH"],
                }
            ],
        }
    ).site_map()
    model = _Llama()
    external = ExternalModel(model, declared)

    with instrument(external, observe=frozenset({stream(LLAMA, 1)})) as region:
        region(lambda: model(an_input()))

    reading = region.trace[stream(LLAMA, 1)]
    assert reading.axes == STREAM_AXES
    assert tuple(reading.tensor.shape) == (BATCH, POSITIONS, WIDTH)


def test_the_attachment_the_region_installs_names_the_path_it_resolved() -> None:
    opening = _Opening()
    external = ExternalModel(_Bert(), a_bert_map(), hooking=opening)

    with external.instrument(frozenset({preactivation(BERT, 1)}), patches={}):
        pass

    assert opening.attached[0].path == "encoder.layer.1.intermediate.dense"


def test_the_module_an_attachment_holds_is_the_one_the_model_holds_at_the_path() -> (
    None
):
    model = _Gpt()
    opening = _Opening()
    external = ExternalModel(model, a_gpt_map(), hooking=opening)

    with external.instrument(frozenset({preactivation(GPT, 0)}), patches={}):
        pass

    assert opening.attached[0].module is model.transformer.block(0).mlp.c_fc


def test_an_installer_that_is_not_the_default_keeps_the_model_unhooked() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map(), hooking=_Counting())

    with external.instrument(frozenset({stream(GPT, 0)}), patches={}):
        assert hooked(model) == 0


def test_the_published_map_cannot_be_written_to() -> None:
    published = cast("MutableMapping[Site, SiteSpec]", a_gpt_map().published)

    with pytest.raises(TypeError):
        published[stream(GPT, 0)] = STREAM_SPEC


class _SealedGpt(_Gpt):
    @override
    def named_modules(self, *args: object, **named: object) -> NoReturn:
        message = "this model does not enumerate its modules"

        raise AssertionError(message)

    @override
    def named_parameters(self, *args: object, **named: object) -> NoReturn:
        message = "this model does not enumerate its parameters"

        raise AssertionError(message)

    @override
    def state_dict(self, *args: object, **named: object) -> NoReturn:
        message = "this model does not enumerate its state"

        raise AssertionError(message)


class _Refusing(nn.Module):
    @override
    def register_forward_hook(self, hook: object, **named: object) -> NoReturn:
        del hook, named
        message = "this module takes no hook"

        raise RuntimeError(message)


def test_a_region_installs_exactly_one_hook_per_point_and_none_besides() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())
    watched = frozenset({stream(GPT, 0), preactivation(GPT, 1)})

    with instrument(external, observe=watched) as region:
        assert hooks_by_path(model) == {
            "transformer.h.0": 1,
            "transformer.h.1.mlp.c_fc": 1,
        }
        region(lambda: model(an_input()))

    assert hooks_by_path(model) == {}


def test_a_point_named_for_reading_and_for_patching_takes_one_hook_only() -> None:
    model = _Gpt()
    external = ExternalModel(model, a_gpt_map())
    site = stream(GPT, 0)

    with instrument(
        external, observe=frozenset({site}), patch={site: Zero()}
    ) as region:
        assert hooks_by_path(model) == {"transformer.h.0": 1}
        region(lambda: model(an_input()))

    assert torch.equal(region.trace[site].tensor, torch.zeros(BATCH, POSITIONS, WIDTH))


def test_registration_that_fails_partway_takes_back_what_it_installed() -> None:
    model = _Gpt()
    attached = (
        Attachment(
            stream(GPT, 0),
            model.transformer.block(0),
            "transformer.h.0",
            0,
            STREAM_AXES,
            observed=True,
            patch=None,
        ),
        Attachment(
            stream(GPT, 1),
            _Refusing(),
            "transformer.h.1",
            0,
            STREAM_AXES,
            observed=True,
            patch=None,
        ),
    )

    with pytest.raises(RuntimeError, match="takes no hook"):
        TORCH_HOOKS.install(attached, lambda _observation: None)

    assert hooked(model) == 0


def test_a_region_reads_a_model_that_enumerates_nothing_it_holds() -> None:
    model = _SealedGpt()
    external = ExternalModel(model, a_gpt_map())

    with instrument(external, observe=frozenset({stream(GPT, 0)})) as region:
        region(lambda: model(an_input()))

    assert set(region.trace) == {stream(GPT, 0)}
    assert tuple(region.trace[stream(GPT, 0)].tensor.shape) == (
        BATCH,
        POSITIONS,
        WIDTH,
    )


def test_a_region_over_a_model_that_enumerates_nothing_still_refuses_a_bad_path() -> (
    None
):
    missing = ExternalSiteMap(
        GPT,
        (
            ExternalSite(
                "transformer.h.7", Role.RESIDUAL_STREAM, 0, STREAM_SPEC, select=0
            ),
        ),
    )

    with pytest.raises(ValueError, match=r"holds no module there"):
        ExternalModel(_SealedGpt(), missing).instrument(
            frozenset({stream(GPT, 0)}), patches={}
        )


def test_an_external_model_runs_named_inputs_without_mutating_them() -> None:
    external = ExternalModel(_Gpt(), a_gpt_map())
    given = an_input()
    before = given.clone()

    output = external.run({"given": given})

    assert isinstance(output, torch.Tensor)
    assert output.shape == given.shape
    assert torch.equal(given, before)


def test_an_external_model_refuses_an_empty_or_nontensor_input_mapping() -> None:
    external = ExternalModel(_Gpt(), a_gpt_map())

    with pytest.raises(ValueError, match="at least one named input"):
        external.run({})

    with pytest.raises(TypeError, match="is not a tensor"):
        external.run({"given": cast("Tensor", 3)})

    with pytest.raises(TypeError, match="inputs are a mapping"):
        external.run(cast("Mapping[str, Tensor]", 3))


def test_an_external_joint_subject_reaches_only_its_published_site_map() -> None:
    outer = TypeAt((0, 3))
    inner = TypeAt((1, 2))

    def scorer(
        output: Output,
        variable: HighLevelVariable,
        decisions: tuple[str | int, ...],
    ) -> Tensor:
        del variable
        tensor = (
            output if isinstance(output, torch.Tensor) else cast("Tensor", output[0])
        )
        return tensor[:, 0, : len(decisions)]

    declared = a_gpt_map()
    subject = ExternalJointSubject(
        ExternalModel(_Gpt(), declared),
        AlignmentPolicy("all", "exclude", contiguous=True),
        "toy-tokenizer",
        {outer: ("A", "B"), inner: ("A", "B")},
        scorer,
        batch_axis="item",
        row_axis="token",
    )

    assert subject.site_map() == declared.published
    assert subject.decisions(outer) == ("A", "B")

    with pytest.raises(TypeError, match="NamedVariable"):
        ExternalJointSubject(
            ExternalModel(_Gpt(), declared),
            AlignmentPolicy("all", "exclude", contiguous=True),
            "toy-tokenizer",
            {
                outer: ("A", "B"),
                NamedVariable("external", "category"): ("A", "B"),
            },
            scorer,
            batch_axis="item",
            row_axis="token",
        )

    with pytest.raises(ValueError, match="declares no Site"):
        subject.instrument(frozenset({Site(GPT, Role.GATE, (0,))}), patches={})
