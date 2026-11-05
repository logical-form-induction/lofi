from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    ReceiverSide,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

ENCODER = ComponentId("term-encoder")

key_parts = st.one_of(
    st.integers(min_value=0, max_value=64),
    st.text(alphabet="abc", min_size=1, max_size=3),
)
keys = st.lists(key_parts, max_size=3).map(tuple)


def test_a_site_keeps_the_three_parts_that_name_the_point() -> None:
    site = Site(ENCODER, Role.RECURRENT_HIDDEN, (3, "left"))

    assert site.component == ENCODER
    assert site.role is Role.RECURRENT_HIDDEN
    assert site.key == (3, "left")


def test_a_role_that_names_one_point_needs_no_key() -> None:
    assert Site(ENCODER, Role.TERM_ENCODING).key == ()


def test_sites_naming_the_same_point_are_one_site() -> None:
    site = Site(ENCODER, Role.ATTENTION_WEIGHTS, (0,))
    same = Site(ENCODER, Role.ATTENTION_WEIGHTS, (0,))

    assert site == same
    assert hash(site) == hash(same)
    assert len({site, same}) == 1
    assert site != Site(ENCODER, Role.ATTENTION_LOGITS, (0,))
    assert site != Site(ENCODER, Role.ATTENTION_WEIGHTS, (1,))
    assert site != Site(ComponentId("decoder"), Role.ATTENTION_WEIGHTS, (0,))
    assert site != (ENCODER, Role.ATTENTION_WEIGHTS, (0,))


def test_a_site_cannot_be_rebound_or_emptied() -> None:
    site = Site(ENCODER, Role.GATE, (2,))

    for field in ("component", "role", "key"):
        with pytest.raises(AttributeError):
            setattr(site, field, None)

        with pytest.raises(AttributeError):
            delattr(site, field)

    assert not hasattr(site, "__dict__")
    assert site.role is Role.GATE


def test_a_site_names_the_component_it_belongs_to() -> None:
    with pytest.raises(ValueError, match="names the component"):
        Site(ComponentId(""), Role.CHART_CELL)


def test_a_role_cannot_arrive_as_a_string() -> None:
    with pytest.raises(TypeError, match="role is a Role"):
        Site(ENCODER, cast("Role", "attention_weights"))


def test_a_role_cannot_be_recovered_from_the_way_a_module_spells_a_name() -> None:
    with pytest.raises(ValueError, match="not a valid Role"):
        Role("gate")

    with pytest.raises(ValueError, match="not a valid Role"):
        Role("attention_weights")


def test_a_key_part_is_an_index_or_a_name() -> None:
    with pytest.raises(TypeError, match="index or a name"):
        Site(ENCODER, Role.CHART_CELL, (True,))

    with pytest.raises(ValueError, match="counts from the start"):
        Site(ENCODER, Role.CHART_CELL, (-1,))

    with pytest.raises(ValueError, match="key part is empty"):
        Site(ENCODER, Role.CHART_CELL, ("",))


def test_access_composes_and_reports_what_it_permits() -> None:
    both = Access.OBSERVE | Access.PATCH

    assert Access.OBSERVE in both
    assert Access.PATCH in both
    assert Access.PATCH not in Access.OBSERVE
    assert both & Access.OBSERVE == Access.OBSERVE
    assert not (Access.OBSERVE & Access.PATCH)


def test_receiver_sides_are_the_two_stable_edge_axis_coordinates() -> None:
    assert tuple(ReceiverSide) == (ReceiverSide.LEFT, ReceiverSide.RIGHT)
    assert ReceiverSide.LEFT.value == 0
    assert ReceiverSide.RIGHT.value == 1
    assert Role.RECEIVER_EDGE is not Role.CHART_CELL


def test_a_site_reads_back_out_of_a_map_keyed_by_it() -> None:
    readings = {
        Site(ENCODER, Role.CHART_CELL, (span,)): float(span) for span in range(4)
    }

    assert readings[Site(ENCODER, Role.CHART_CELL, (2,))] == 2.0
    assert Site(ENCODER, Role.OUTSIDE_SCORE, (2,)) not in readings


@settings(max_examples=50)
@given(
    component=st.text(alphabet="abc", min_size=1, max_size=4),
    role=st.sampled_from(list(Role)),
    key=keys,
)
def test_a_site_is_nothing_but_its_three_parts(
    component: str, role: Role, key: tuple[int | str, ...]
) -> None:
    site = Site(ComponentId(component), role, key)
    rebuilt = Site(ComponentId(component), role, key)
    elsewhere = Site(ComponentId(component + "z"), role, key)

    assert site == rebuilt
    assert hash(site) == hash(rebuilt)
    assert site != elsewhere
    assert site.key == tuple(key)


def a_spec(
    axes: tuple[str, ...] = ("item", "row", "width"),
    dtype_class: DTypeClass = DTypeClass.REAL,
    *,
    differentiable: bool = True,
    access: Access = Access.OBSERVE,
) -> SiteSpec:
    return SiteSpec(axes, dtype_class, differentiable=differentiable, access=access)


def test_a_specification_keeps_the_four_things_it_declares() -> None:
    spec = a_spec(access=Access.OBSERVE | Access.PATCH)

    assert spec.axes == ("item", "row", "width")
    assert spec.rank == 3
    assert spec.dtype_class is DTypeClass.REAL
    assert spec.differentiable is True
    assert Access.PATCH in spec.access


def test_a_point_holding_one_number_declares_no_axis() -> None:
    spec = a_spec(axes=())

    assert spec.axes == ()
    assert spec.rank == 0


def test_the_dtype_classes_are_the_five_the_layer_distinguishes() -> None:
    assert {member.value for member in DTypeClass} == {
        "real",
        "log-probability",
        "probability",
        "integer",
        "mask",
    }


def test_a_log_probability_and_a_probability_are_different_declarations() -> None:
    logs = a_spec(dtype_class=DTypeClass.LOG_PROBABILITY)
    probabilities = a_spec(dtype_class=DTypeClass.PROBABILITY)

    assert logs != probabilities
    assert logs.dtype_class is not probabilities.dtype_class


def test_an_axis_is_addressed_by_name_and_its_position_comes_from_the_spec() -> None:
    spec = a_spec()

    assert spec.position("item") == 0
    assert spec.position("width") == 2


def test_asking_for_an_axis_that_is_not_declared_names_the_ones_that_are() -> None:
    spec = a_spec()

    with pytest.raises(KeyError, match="only"):
        spec.position("span")


def test_a_duplicate_axis_name_is_refused_and_the_message_names_it() -> None:
    with pytest.raises(ValueError, match="'row' twice"):
        a_spec(axes=("row", "width", "row"))


def test_an_empty_axis_name_is_refused_and_the_message_names_its_position() -> None:
    with pytest.raises(ValueError, match="axis 1 of a site specification is empty"):
        a_spec(axes=("item", ""))


def test_an_axis_name_that_is_not_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="not a string"):
        a_spec(axes=cast("tuple[str, ...]", ("item", 3)))


def test_a_specification_permitting_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="permits neither"):
        a_spec(access=Access(0))


def test_a_dtype_class_spelled_as_a_string_is_refused() -> None:
    with pytest.raises(TypeError, match="dtype class is not one"):
        a_spec(dtype_class=cast("DTypeClass", "real"))


def test_differentiability_is_a_boolean_and_not_a_truthy_number() -> None:
    with pytest.raises(TypeError, match="not a boolean"):
        a_spec(differentiable=cast("bool", 1))


def test_access_spelled_as_a_set_of_flags_is_refused() -> None:
    with pytest.raises(TypeError, match="not an Access"):
        a_spec(access=cast("Access", frozenset({Access.OBSERVE})))


def test_specifications_declaring_the_same_four_things_are_one_specification() -> None:
    spec = a_spec()

    assert spec == a_spec()
    assert hash(spec) == hash(a_spec())
    assert len({spec, a_spec()}) == 1
    assert spec != a_spec(axes=("item", "width", "row"))
    assert spec != a_spec(dtype_class=DTypeClass.MASK)
    assert spec != a_spec(differentiable=False)
    assert spec != a_spec(access=Access.PATCH)
    assert spec != ("item", "row", "width")


def test_a_specification_cannot_be_rebound() -> None:
    spec = a_spec()

    for field in ("axes", "dtype_class", "differentiable", "access"):
        with pytest.raises(AttributeError):
            setattr(spec, field, None)

    assert not hasattr(spec, "__dict__")
    assert spec.axes == ("item", "row", "width")


def test_a_site_map_looks_a_specification_up_by_the_point_it_describes() -> None:
    inside = Site(ENCODER, Role.INSIDE_SCORE, (0, 3))
    logits = Site(ENCODER, Role.COMBINATOR_LOGITS)
    published = {
        inside: a_spec(("item", "span"), DTypeClass.LOG_PROBABILITY),
        logits: a_spec(("item", "rule"), DTypeClass.REAL),
    }

    assert published[Site(ENCODER, Role.INSIDE_SCORE, (0, 3))].axes == ("item", "span")
    assert published[logits].dtype_class is DTypeClass.REAL
    assert Site(ENCODER, Role.INSIDE_SCORE, (1, 3)) not in published
    assert Site(ENCODER, Role.CHART_CELL, (0, 3)) not in published
    assert Site(ENCODER, Role.CONSTITUENT_MARGINAL, (0, 3)) not in published


def test_a_site_map_keeps_one_component_one_kind_and_its_points() -> None:
    point = Site(ENCODER, Role.TERM_ENCODING)
    spec = a_spec(("term", "width"))
    published = SiteMap(ENCODER, ComponentKind.TERM_ENCODER, {point: spec})

    assert published.component == ENCODER
    assert published.kind is ComponentKind.TERM_ENCODER
    assert dict(published) == {point: spec}


def test_a_site_map_is_immutable_and_does_not_share_its_input_mapping() -> None:
    point = Site(ENCODER, Role.TERM_ENCODING)
    source = {point: a_spec(("term", "width"))}
    published = SiteMap(ENCODER, ComponentKind.TERM_ENCODER, source)
    source.clear()

    assert tuple(published) == (point,)

    with pytest.raises(TypeError):
        cast("dict[Site, SiteSpec]", published)[point] = a_spec(("other",))

    with pytest.raises(AttributeError):
        published.__setattr__("_kind", ComponentKind.PARSER)

    with pytest.raises(AttributeError):
        published.__setattr__("_component", ComponentId("other"))

    assert published.component == ENCODER
    assert published.kind is ComponentKind.TERM_ENCODER


def test_a_site_map_refuses_a_point_from_another_component() -> None:
    elsewhere = Site(ComponentId("other"), Role.TERM_ENCODING)

    with pytest.raises(ValueError, match="belongs to other"):
        SiteMap(
            ENCODER,
            ComponentKind.TERM_ENCODER,
            {elsewhere: a_spec(("term", "width"))},
        )


def test_a_site_map_refuses_sites_that_are_not_a_mapping_deliberately() -> None:
    with pytest.raises(TypeError, match="sites are a Mapping"):
        SiteMap(
            ENCODER,
            ComponentKind.TERM_ENCODER,
            cast("Mapping[Site, SiteSpec]", ()),
        )


def test_site_map_equality_includes_the_component_and_kind_tags() -> None:
    published = SiteMap.custom(ENCODER, {})
    same = SiteMap.custom(ENCODER, {})
    other_component = SiteMap.custom(ComponentId("other"), {})
    other_kind = SiteMap(ENCODER, ComponentKind.PARSER, {})

    assert published == same
    assert hash(published) == hash(same)
    assert published != other_component
    assert published != other_kind
    assert len({published, same, other_component, other_kind}) == 3
    assert published != {}


def test_a_custom_site_map_takes_the_explicit_custom_path() -> None:
    empty = SiteMap.custom(ENCODER, {})

    assert empty.component == ENCODER
    assert empty.kind is ComponentKind.CUSTOM
    assert len(empty) == 0

    with pytest.raises(ValueError, match=r"SiteMap\.custom"):
        SiteMap(ENCODER, ComponentKind.CUSTOM, {})


def test_the_repr_of_a_specification_shows_its_four_fields() -> None:
    shown = repr(a_spec(("span",), DTypeClass.MASK, differentiable=False))

    assert "('span',)" in shown
    assert "DTypeClass.MASK" in shown
    assert "differentiable=False" in shown
    assert "OBSERVE" in shown
