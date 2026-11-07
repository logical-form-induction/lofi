from typing import TYPE_CHECKING, Final, Never, cast, override

import didactic.api as dx
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
    SiteMap,
    SiteSpec,
)
from lofi.probe._boundary import (
    ClassifierProbeSettings,
    ClassifierProbeSpec,
    DiagnosticsProbeSettings,
    DiagnosticsProbeSpec,
    ExternalModelSpec,
    ExternalSiteDecl,
    ExternalSiteMapIso,
    InterventionProbeSpec,
    MeanDecl,
    NoiseDecl,
    ProbeSiteDecl,
    ResampleDecl,
    ShuffleDecl,
    SiteDecl,
    SiteInterventionDecl,
    SiteMapIso,
    SiteMapSpec,
    SpanLabels,
    ZeroDecl,
    lower,
)
from lofi.probe._external import ExternalModel
from lofi.probe._instrument import instrument
from lofi.probe._values import Observation, Trace
from lofi.spec import ProbeSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue

PARSER = ComponentId("cky-parser")
CELL = Site(PARSER, Role.CHART_CELL, (0, 4))
SPLITS = Site(PARSER, Role.SPLIT_DISTRIBUTION, ())

PASSES = 100


def a_cell_decl(
    *,
    role: str = "CHART_CELL",
    axes: tuple[str, ...] = ("item", "span", "width"),
    dtype_class: str = "real",
    access: tuple[str, ...] = ("OBSERVE", "PATCH"),
) -> SiteDecl:
    return SiteDecl(
        role=role,
        key=(0, 4),
        axes=axes,
        dtype_class=dtype_class,
        differentiable=True,
        access=access,
    )


def a_spec() -> SiteMapSpec:
    return SiteMapSpec(
        component="cky-parser",
        kind="parser",
        sites=(
            a_cell_decl(),
            SiteDecl(
                role="SPLIT_DISTRIBUTION",
                key=(),
                axes=("item", "span", "split"),
                dtype_class="log-probability",
                differentiable=True,
                access=("OBSERVE",),
            ),
            SiteDecl(
                role="INSIDE_SCORE",
                key=(),
                axes=("item", "span"),
                dtype_class="real",
                differentiable=True,
                access=("OBSERVE",),
            ),
            SiteDecl(
                role="OUTSIDE_SCORE",
                key=(),
                axes=("item", "span"),
                dtype_class="real",
                differentiable=True,
                access=("OBSERVE",),
            ),
        ),
    )


def a_custom_spec(*sites: SiteDecl) -> SiteMapSpec:
    return SiteMapSpec(component="cky-parser", kind="custom", sites=sites)


@pytest.fixture
def crossings(monkeypatch: pytest.MonkeyPatch) -> list[int]:
    counted = [0]
    built = cast("Callable[..., None]", dx.Model.__init__)

    def counting(self: dx.Model, *args: Never, **fields: JsonValue) -> None:
        counted[0] += 1
        built(self, *args, **fields)

    monkeypatch.setattr(dx.Model, "__init__", counting)

    return counted


def test_a_declaration_resolves_to_the_map_a_probe_reads() -> None:
    resolved = SiteMapIso().forward(a_spec())

    assert resolved.component == PARSER
    assert resolved.kind is ComponentKind.PARSER
    assert {CELL, SPLITS} <= set(resolved)
    assert resolved[CELL] == SiteSpec(
        ("item", "span", "width"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE | Access.PATCH,
    )
    assert resolved[SPLITS].dtype_class is DTypeClass.LOG_PROBABILITY
    assert Access.PATCH not in resolved[SPLITS].access


def test_the_codec_round_trips_a_declaration_through_the_interior_form() -> None:
    codec = SiteMapIso()
    declared = a_spec()

    assert codec.backward(codec.forward(declared)) == declared


def test_the_codec_round_trips_an_interior_map_through_the_declared_form() -> None:
    codec = SiteMapIso()
    resolved = codec.forward(a_spec())

    assert codec.forward(codec.backward(resolved)) == resolved


def test_a_permission_set_written_in_another_order_resolves_to_the_same_map() -> None:
    codec = SiteMapIso()
    permuted = a_custom_spec(a_cell_decl(access=("PATCH", "OBSERVE")))

    resolved = codec.forward(permuted)

    assert resolved == codec.forward(a_custom_spec(a_cell_decl()))
    assert codec.backward(resolved).sites[0].access == ("OBSERVE", "PATCH")


def test_a_declaration_forbids_a_key_the_record_does_not_declare() -> None:
    with pytest.raises(dx.ValidationError):
        SiteDecl.model_validate(
            {
                "component": "cky-parser",
                "role": "CHART_CELL",
                "key": (0, 4),
                "axes": ("item", "span", "width"),
                "dtype_class": "real",
                "differentiable": True,
                "access": ("OBSERVE",),
                "bogus": 1,
            }
        )


def test_a_site_map_declaration_forbids_a_key_it_does_not_declare() -> None:
    with pytest.raises(dx.ValidationError):
        SiteMapSpec.model_validate(
            {
                "component": "empty-grounding",
                "kind": "custom",
                "sites": (),
                "extra": 1,
            }
        )


def test_a_site_map_declaration_requires_its_component_and_kind() -> None:
    with pytest.raises(dx.ValidationError):
        SiteMapSpec.model_validate({"sites": ()})


def test_a_custom_declaration_preserves_an_empty_component_view() -> None:
    declared = SiteMapSpec(component="empty-grounding", kind="custom", sites=())
    resolved = SiteMapIso().forward(declared)

    assert resolved.component == ComponentId("empty-grounding")
    assert resolved.kind is ComponentKind.CUSTOM
    assert len(resolved) == 0
    assert SiteMapIso().backward(resolved) == declared


def test_forward_lowering_refuses_a_real_component_role_with_wrong_axes() -> None:
    correct = a_spec()
    declared = SiteMapSpec(
        component="cky-parser",
        kind="parser",
        sites=(a_cell_decl(axes=("item", "width", "span")), *correct.sites[1:]),
    )

    with pytest.raises(ValueError, match="runs over") as refusal:
        SiteMapIso().forward(declared)

    message = str(refusal.value)

    assert repr(CELL) in message
    assert "('item', 'width', 'span')" in message
    assert "('item', 'span', 'width')" in message


def test_forward_lowering_refuses_a_real_component_role_with_wrong_dtype() -> None:
    correct = a_spec()
    declared = SiteMapSpec(
        component="cky-parser",
        kind="parser",
        sites=(
            a_cell_decl(dtype_class="probability"),
            *correct.sites[1:],
        ),
    )

    with pytest.raises(ValueError, match="holds real") as refusal:
        SiteMapIso().forward(declared)

    message = str(refusal.value)

    assert repr(CELL) in message
    assert "as probability" in message


def test_a_site_declaration_cannot_repeat_the_map_s_component() -> None:
    with pytest.raises(dx.ValidationError):
        SiteDecl.model_validate(
            {
                "component": "cky-parser",
                "role": "CHART_CELL",
                "axes": ("item", "span", "width"),
                "dtype_class": "real",
                "differentiable": True,
                "access": ("OBSERVE",),
            }
        )


def test_a_role_that_does_not_exist_is_refused_and_the_declared_ones_named() -> None:
    with pytest.raises(ValueError, match="no role is named 'CHART_SELL'"):
        SiteMapIso().forward(a_custom_spec(a_cell_decl(role="CHART_SELL")))


def test_an_access_member_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="no access is named 'WRITE'"):
        SiteMapIso().forward(a_custom_spec(a_cell_decl(access=("WRITE",))))


def test_a_point_that_permits_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="permits neither observation nor patching"):
        SiteMapIso().forward(a_custom_spec(a_cell_decl(access=())))


def test_a_dtype_class_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="no dtype class is spelled 'float32'"):
        SiteMapIso().forward(a_custom_spec(a_cell_decl(dtype_class="float32")))


def test_a_repeated_axis_name_in_a_declaration_is_refused() -> None:
    with pytest.raises(ValueError, match="declares the axis 'item' twice"):
        SiteMapIso().forward(
            a_custom_spec(a_cell_decl(axes=("item", "item"))),
        )


def test_a_declaration_naming_one_point_twice_is_refused() -> None:
    with pytest.raises(ValueError, match=r"names .* twice"):
        SiteMapIso().forward(a_custom_spec(a_cell_decl(), a_cell_decl()))


def test_lowering_gives_one_reader_per_point_the_scope_asked_for() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []

    readers = lower(resolved, (CELL, SPLITS), collected)

    assert set(readers) == {CELL, SPLITS}


def test_a_reader_files_what_it_was_handed_under_its_own_point() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []
    readers = lower(resolved, (CELL, SPLITS), collected)
    handed = torch.arange(6, dtype=torch.float32).reshape(2, 3)

    readers[SPLITS](handed)

    assert len(collected) == 1
    assert collected[0][0] == SPLITS
    assert torch.equal(collected[0][1], handed)


def test_the_readers_hold_the_tensor_they_were_handed_and_check_nothing() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []
    readers = lower(resolved, (CELL, SPLITS), collected)

    readers[CELL](torch.zeros(1))
    readers[CELL](torch.zeros(9, 9, 9, 9))

    assert [tuple(tensor.shape) for _, tensor in collected] == [(1,), (9, 9, 9, 9)]


def test_validation_happens_once_and_the_open_scope_crosses_no_boundary(
    crossings: list[int],
) -> None:
    declared = a_spec()
    resolved = SiteMapIso().forward(declared)
    collected: list[tuple[Site, torch.Tensor]] = []
    readers = lower(resolved, (CELL, SPLITS), collected)
    handed = torch.zeros(2, 3)

    opened = crossings[0]

    for _ in range(PASSES):
        for read in readers.values():
            read(handed)

    assert crossings[0] - opened == 0
    assert len(collected) == PASSES * len(readers)


def test_a_second_scope_over_the_same_map_reuses_the_validated_form(
    crossings: list[int],
) -> None:
    resolved = SiteMapIso().forward(a_spec())

    reopened = crossings[0]
    first: list[tuple[Site, torch.Tensor]] = []
    second: list[tuple[Site, torch.Tensor]] = []
    opened = lower(resolved, (CELL, SPLITS), first)
    reopened_readers = lower(resolved, (CELL, SPLITS), second)
    opened[CELL](torch.zeros(2, 3))
    reopened_readers[CELL](torch.ones(2, 3))

    assert crossings[0] - reopened == 0
    assert set(opened) == set(reopened_readers) == {CELL, SPLITS}
    assert [site for site, _ in first] == [CELL]
    assert torch.equal(second[0][1], torch.ones(2, 3))


def test_the_counter_moves_when_a_declaration_is_actually_built(
    crossings: list[int],
) -> None:
    before = crossings[0]

    a_cell_decl()

    assert crossings[0] > before


def test_a_point_the_scope_did_not_ask_for_gets_no_reader() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []

    readers = lower(resolved, (SPLITS,), collected)

    assert set(readers) == {SPLITS}
    assert len(readers) == 1


def test_asking_for_no_point_installs_no_reader() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []

    assert lower(resolved, (), collected) == {}


def test_asking_for_one_point_twice_installs_one_reader() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []

    readers = lower(resolved, (CELL, CELL), collected)

    assert len(readers) == 1

    readers[CELL](torch.zeros(2, 3))

    assert len(collected) == 1


def test_asking_for_a_point_the_map_does_not_publish_is_refused_loudly() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []
    elsewhere = Site(ComponentId("other-parser"), Role.CHART_CELL, (0, 4))

    with pytest.raises(ValueError, match="does not publish"):
        lower(resolved, (CELL, elsewhere), collected)

    assert collected == []


def test_the_readers_come_back_in_the_order_the_map_publishes_them() -> None:
    resolved = SiteMapIso().forward(a_spec())
    collected: list[tuple[Site, torch.Tensor]] = []

    assert list(lower(resolved, (SPLITS, CELL), collected)) == [CELL, SPLITS]


MODEL = ComponentId("gpt2-small")
STREAM = Site(MODEL, Role.RESIDUAL_STREAM, (3,))
PREACTIVATION = Site(MODEL, Role.FFN_PREACTIVATION, (3,))


def a_stream_decl(
    *,
    path: str = "transformer.h.3",
    role: str = "RESIDUAL_STREAM",
    layer: int = 3,
    select: int | None = 0,
) -> ExternalSiteDecl:
    return ExternalSiteDecl(
        path=path,
        role=role,
        layer=layer,
        axes=("batch", "position", "feature"),
        dtype_class="real",
        differentiable=True,
        access=("OBSERVE", "PATCH"),
        select=select,
    )


def an_external_spec() -> ExternalModelSpec:
    return ExternalModelSpec(
        model="gpt2-small",
        sites=(
            a_stream_decl(),
            a_stream_decl(
                path="transformer.h.3.mlp.c_fc",
                role="FFN_PREACTIVATION",
                select=None,
            ),
        ),
    )


def test_an_external_declaration_resolves_to_the_map_a_probe_reads() -> None:
    resolved = ExternalSiteMapIso().forward(an_external_spec())

    assert resolved.model == MODEL
    assert set(resolved.published) == {STREAM, PREACTIVATION}
    assert resolved.declaration(STREAM).path == "transformer.h.3"
    assert resolved.declaration(STREAM).select == 0
    assert resolved.declaration(PREACTIVATION).select is None
    assert resolved.published[STREAM] == SiteSpec(
        ("batch", "position", "feature"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE | Access.PATCH,
    )


def test_the_external_codec_round_trips_a_declaration_through_the_resolved_form() -> (
    None
):
    codec = ExternalSiteMapIso()
    declared = an_external_spec()

    assert codec.backward(codec.forward(declared)) == declared


def test_the_external_codec_round_trips_a_resolved_map_through_the_declared_form() -> (
    None
):
    codec = ExternalSiteMapIso()
    resolved = codec.forward(an_external_spec())

    assert codec.forward(codec.backward(resolved)) == resolved


def test_an_external_declaration_forbids_a_key_the_record_does_not_declare() -> None:
    with pytest.raises(dx.ValidationError):
        ExternalSiteDecl.model_validate(
            {
                "path": "transformer.h.3",
                "role": "RESIDUAL_STREAM",
                "layer": 3,
                "axes": ("batch", "position", "feature"),
                "dtype_class": "real",
                "differentiable": True,
                "access": ("OBSERVE",),
                "module": "transformer",
            }
        )


def test_an_external_model_declaration_forbids_a_key_it_does_not_declare() -> None:
    with pytest.raises(dx.ValidationError):
        ExternalModelSpec.model_validate({"model": "gpt2-small", "family": "gpt2"})


def test_an_external_declaration_naming_a_role_that_does_not_exist_is_refused() -> None:
    with pytest.raises(ValueError, match="no role is named 'RESIDUAL'"):
        ExternalSiteMapIso().forward(
            ExternalModelSpec(
                model="gpt2-small", sites=(a_stream_decl(role="RESIDUAL"),)
            )
        )


def test_an_external_declaration_without_a_module_path_is_refused() -> None:
    with pytest.raises(ValueError, match="module path is empty"):
        ExternalSiteMapIso().forward(
            ExternalModelSpec(model="gpt2-small", sites=(a_stream_decl(path=""),))
        )


def test_an_external_declaration_naming_one_point_twice_is_refused() -> None:
    with pytest.raises(ValueError, match="twice"):
        ExternalSiteMapIso().forward(
            ExternalModelSpec(
                model="gpt2-small", sites=(a_stream_decl(), a_stream_decl())
            )
        )


STACKED = ComponentId("stacked")
FIRST = Site(STACKED, Role.RESIDUAL_STREAM, (0,))


def a_stacked_spec() -> ExternalModelSpec:
    return ExternalModelSpec(
        model="stacked",
        sites=(
            ExternalSiteDecl(
                path="0",
                role="RESIDUAL_STREAM",
                layer=0,
                axes=("batch", "position", "feature"),
                dtype_class="real",
                differentiable=True,
                access=("OBSERVE",),
            ),
        ),
    )


class _CustomExternal(ExternalModel):
    @override
    def site_map(self) -> SiteMap:
        return SiteMap.custom(self.declared.model, self.declared.published)


def test_an_external_map_is_validated_once_and_no_pass_crosses_the_boundary(
    crossings: list[int],
) -> None:
    resolved = ExternalSiteMapIso().forward(a_stacked_spec())
    model = nn.Sequential(nn.Linear(4, 4), nn.Linear(4, 4))
    external = _CustomExternal(model, resolved)
    given = torch.zeros(1, 2, 4)

    opened = crossings[0]

    for _ in range(PASSES):
        with instrument(external, observe=frozenset({FIRST})) as region:
            region(lambda: model(given))

            assert region.trace[FIRST].axes == ("batch", "position", "feature")

    assert crossings[0] - opened == 0


def test_a_diagnostics_declaration_lowers_role_names_to_roles() -> None:
    declared = DiagnosticsProbeSpec(roles=("GATE", "CHART_CELL"))

    assert declared.settings() == DiagnosticsProbeSettings(
        roles=frozenset({Role.GATE, Role.CHART_CELL})
    )
    assert DiagnosticsProbeSpec().settings().roles is None


MALFORMED: Final[tuple[dict[str, JsonValue], ...]] = (
    {"kind": "diagnostics", "roles": ["GATES"]},
    {"kind": "classifier", "role": "CELL"},
    {"kind": "classifier", "held_out_fraction": 0.0},
    {"kind": "classifier", "held_out_fraction": 1.0},
    {"kind": "classifier", "labels": "depth"},
    {"kind": "classifier", "block_fractions": [0.5, 0.9]},
    {"kind": "classifier", "block_fractions": [0.6, 0.3, 1.0]},
    {"kind": "classifier", "epochs": 0},
    {"kind": "classifier", "hidden": -1},
    {"kind": "classifier", "learning_rate": 0.0},
    {"kind": "classifier", "control_seed": -1},
    {
        "kind": "intervention",
        "interventions": [
            {
                "site": {"component": "parser", "role": "RECEIVER_EDGE"},
                "intervention": {"kind": "noise", "sigma": 0.0},
            }
        ],
    },
    {"kind": "intervention", "interventions": []},
    {
        "kind": "intervention",
        "outcome": "PARTITION",
        "interventions": [
            {
                "site": {"component": "parser", "role": "RECEIVER_EDGE"},
                "intervention": {"kind": "zero"},
            }
        ],
    },
    {
        "kind": "intervention",
        "interventions": [
            {
                "site": {"component": "parser", "role": "EDGE"},
                "intervention": {"kind": "zero"},
            }
        ],
    },
    {
        "kind": "intervention",
        "interventions": [
            {
                "site": {"component": "parser", "role": "RECEIVER_EDGE"},
                "intervention": {"kind": "zero"},
            },
            {
                "site": {"component": "parser", "role": "RECEIVER_EDGE"},
                "intervention": {"kind": "mean"},
            },
        ],
    },
)


@pytest.mark.parametrize("document", MALFORMED)
def test_a_malformed_probe_declaration_is_refused(
    document: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        ProbeSpec.model_validate(document)


def test_a_classifier_declaration_lowers_every_setting() -> None:
    declared = ClassifierProbeSpec.model_validate(
        {
            "kind": "classifier",
            "role": "NODE_ENCODING",
            "labels": "category",
            "held_out_fraction": 0.25,
            "block_fractions": [0.5, 1.0],
        }
    )

    settings = declared.settings()
    capacity = settings.classifier.capacity

    assert type(settings) is ClassifierProbeSettings
    assert settings[:4] == (Role.NODE_ENCODING, SpanLabels.CATEGORY, 0.25, 0)
    assert (
        capacity.hidden,
        capacity.epochs,
        capacity.learning_rate,
        capacity.seed,
    ) == (
        0,
        100,
        0.1,
        0,
    )
    assert settings.classifier.control.seed == 0
    assert settings.schedule.fractions == (0.5, 1.0)


def test_an_intervention_declaration_names_its_sites_and_draws_its_interventions() -> (
    None
):
    declared = InterventionProbeSpec(
        interventions=tuple(
            SiteInterventionDecl(
                site=ProbeSiteDecl(
                    component="parser", role="RECEIVER_EDGE", key=(key,)
                ),
                intervention=intervention,
            )
            for key, intervention in (
                ("receiver", ZeroDecl()),
                ("other", NoiseDecl(sigma=0.25, seed=3)),
                ("mean", MeanDecl()),
                ("shuffle", ShuffleDecl(seed=1)),
                ("resample", ResampleDecl(seed=2)),
            )
        )
    )
    settings = declared.settings()
    clean = Trace()
    reading = Site(ComponentId("parser"), Role.RECEIVER_EDGE, ("mean",))
    clean.record(Observation(reading, torch.ones(2, 3), axes=("item", "width")))
    clean.close()
    drawn = [source.drawn(clean) for _, source in settings.interventions]

    assert settings.outcome is Role.LOG_PARTITION
    assert [site.key for site, _ in settings.interventions] == [
        ("receiver",),
        ("other",),
        ("mean",),
        ("shuffle",),
        ("resample",),
    ]
    assert [intervention.tag for intervention in drawn] == [
        "zero",
        "noise",
        "mean-from",
        "shuffle",
        "resample",
    ]
    assert repr(drawn[1]) == "Noise(0.25, seed=3)"
    assert torch.equal(
        drawn[2].stands_in_for(
            Observation(reading, torch.zeros(1, 3), axes=("item", "width"))
        ),
        torch.ones(1, 3),
    )
