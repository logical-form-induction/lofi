from pathlib import Path
from typing import Final, cast

import pytest
from tests._controllers import mlp_controller

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
from lofi.probe import _roles
from lofi.probe._roles import (
    INVENTORY,
    ROLE_AXES,
    ROLE_DTYPE,
    RoleInventory,
    SiteMapBuilder,
    ensure_publishes,
)

PARSER: Final = ComponentId("cky-parser")
ENCODER: Final = ComponentId("tree-rnn")
OTHER: Final = ComponentId("hrr")
DECODER: Final = ComponentId("term-decoder")

PROSE: Final = Path(_roles.__file__).with_name("ROLES.md")


def a_parser(*roles: Role) -> SiteMapBuilder:
    builder = SiteMapBuilder.custom(PARSER)

    for role in roles:
        builder.publish(role)

    return builder


def a_whole_parser() -> SiteMapBuilder:
    return a_parser(
        Role.CHART_CELL,
        Role.INSIDE_SCORE,
        Role.OUTSIDE_SCORE,
        Role.SPLIT_DISTRIBUTION,
    )


def a_chart_scorer() -> DeepInsideOutside:
    keep = Action("keep")
    controller = mlp_controller(
        ActionSet((Action("apply", "application"), keep), keep),
        width=4,
        component=ComponentId("controller"),
        hidden=3,
    )

    return DeepInsideOutside(
        RuleScorer(controller, width=4),
        composition=MeanFieldComposition(
            MlpSpanFunction((2,), 4, layers=1, nonlinearity="tanh"),
            MlpSpanFunction((4, 4), 4, layers=1, nonlinearity="tanh"),
            width=4,
        ),
        split_left=MlpSpanFunction((4, 4), 4, layers=1, nonlinearity="tanh"),
        split_right=MlpSpanFunction((4, 4), 4, layers=1, nonlinearity="tanh"),
        outside_weights="contextual",
        root_outside="learned",
        root_init_gain=1.0,
        component=PARSER,
        reduction=LogSumExpReduction(),
        weigher=MarginalSplitWeigher(),
        checkpoint_from_width=None,
    )


def tagged(kind: ComponentKind, published: SiteMap) -> SiteMap:
    return SiteMap(published.component, kind, published)


def scorer_map() -> SiteMap:
    return SiteMap(PARSER, ComponentKind.PARSER, a_chart_scorer().site_map())


def test_the_split_distribution_is_declared_as_the_tree_s_parser_publishes_it() -> None:
    published = scorer_map()
    splits = {
        site: spec
        for site, spec in published.items()
        if site.role is Role.SPLIT_DISTRIBUTION
    }

    assert len(splits) == 1

    for spec in splits.values():
        assert spec.axes == ROLE_AXES[Role.SPLIT_DISTRIBUTION]
        assert spec.dtype_class is ROLE_DTYPE[Role.SPLIT_DISTRIBUTION]


def a_spec(axes: tuple[str, ...], dtype_class: DTypeClass) -> SiteSpec:
    return SiteSpec(axes, dtype_class, differentiable=True, access=Access.OBSERVE)


UNPUBLISHED: Final = (
    Role.RECURRENT_HIDDEN,
    Role.RECURRENT_CELL,
    Role.GATE,
    Role.ATTENTION_LOGITS,
    Role.ATTENTION_WEIGHTS,
)


def test_every_role_with_an_entry_declares_its_axes_and_what_they_mean() -> None:
    declared = set(ROLE_AXES)

    assert declared == set(ROLE_DTYPE)
    assert declared == set(Role) - {Role.PARAMETER, *UNPUBLISHED}

    for role, axes in ROLE_AXES.items():
        assert len(set(axes)) == len(axes)
        assert all(axes)
        assert isinstance(ROLE_DTYPE[role], DTypeClass)


@pytest.mark.parametrize("role", UNPUBLISHED, ids=lambda role: role.name)
def test_a_role_with_no_entry_is_published_with_its_own_specification(
    role: Role,
) -> None:
    builder = SiteMapBuilder.custom(ENCODER)

    with pytest.raises(KeyError, match="publish_at"):
        builder.publish(role)

    spec = a_spec(("cell",), DTypeClass.PROBABILITY)
    built = builder.publish_at(role, spec).build()

    assert built[Site(ENCODER, role)] == spec


def test_the_chart_scorer_s_published_map_passes_as_a_parser() -> None:
    published = scorer_map()

    assert ensure_publishes(published) is published

    for site, spec in published.items():
        assert spec.axes == ROLE_AXES[site.role]
        assert spec.dtype_class is ROLE_DTYPE[site.role]


def test_each_chart_point_is_published_under_a_role_of_its_own() -> None:
    roles = {site.key: site.role for site in scorer_map()}

    assert roles == {
        ("inside",): Role.INSIDE_SCORE,
        ("outside",): Role.OUTSIDE_SCORE,
        ("alpha",): Role.SPLIT_DISTRIBUTION,
        ("mu",): Role.CONSTITUENT_MARGINAL,
        ("log_z",): Role.LOG_PARTITION,
        ("identity",): Role.IDENTITY_SCORE,
        ("cells",): Role.CHART_CELL,
        ("outside_cells",): Role.OUTSIDE_CELL,
        ("spans",): Role.SPAN_REPRESENTATION,
        ("receiver",): Role.RECEIVER_EDGE,
    }


def test_a_parser_publishing_an_incomplete_deep_structure_is_refused() -> None:
    published = SiteMap(
        PARSER,
        ComponentKind.PARSER,
        {
            site: spec
            for site, spec in scorer_map().items()
            if site.role is not Role.INSIDE_SCORE
        },
    )

    with pytest.raises(ValueError, match="INSIDE_SCORE") as refusal:
        ensure_publishes(published)

    assert "deep is half exposed" in str(refusal.value)


def test_a_parser_must_publish_its_cell_vectors() -> None:
    published = SiteMap(
        PARSER,
        ComponentKind.PARSER,
        {
            site: spec
            for site, spec in scorer_map().items()
            if site.role is not Role.CHART_CELL
        },
    )

    with pytest.raises(ValueError, match="publishes no CHART_CELL"):
        ensure_publishes(published)


def a_scorer_map_redeclaring(name: str, spec: SiteSpec) -> SiteMap:
    published = dict(scorer_map())
    (point,) = (site for site in published if site.key == (name,))
    published[point] = spec

    return SiteMap(PARSER, ComponentKind.PARSER, published)


def test_a_scorer_readout_over_other_axes_is_refused_naming_both() -> None:
    published = a_scorer_map_redeclaring(
        "cells", a_spec(("item", "span"), DTypeClass.REAL)
    )

    with pytest.raises(ValueError, match="runs over") as refusal:
        ensure_publishes(published)

    message = str(refusal.value)

    assert repr(Site(PARSER, Role.CHART_CELL, ("cells",))) in message
    assert "over ('item', 'span'), and" in message
    assert "of role CHART_CELL runs over ('item', 'span', 'width')" in message


def test_a_scorer_readout_over_its_role_s_axes_in_another_order_is_refused() -> None:
    published = a_scorer_map_redeclaring(
        "cells", a_spec(("item", "width", "span"), DTypeClass.REAL)
    )

    with pytest.raises(ValueError, match="runs over") as refusal:
        ensure_publishes(published)

    message = str(refusal.value)

    assert repr(Site(PARSER, Role.CHART_CELL, ("cells",))) in message
    assert "over ('item', 'width', 'span'), and" in message
    assert "of role CHART_CELL runs over ('item', 'span', 'width')" in message


def test_a_scorer_readout_of_another_numeric_class_is_refused_naming_both() -> None:
    published = a_scorer_map_redeclaring(
        "mu", a_spec(("item", "span"), DTypeClass.REAL)
    )

    with pytest.raises(ValueError, match="holds probability") as refusal:
        ensure_publishes(published)

    message = str(refusal.value)

    assert repr(Site(PARSER, Role.CONSTITUENT_MARGINAL, ("mu",))) in message
    assert "as real, and a point of role CONSTITUENT_MARGINAL holds probability" in (
        message
    )


READOUTS: Final = ("inside", "outside", "alpha", "mu", "log_z", "identity", "cells")


def the_scorer_s_point(name: str) -> tuple[Site, SiteSpec]:
    (point,) = (
        (site, spec) for site, spec in scorer_map().items() if site.key == (name,)
    )

    return point


@pytest.mark.parametrize("name", READOUTS)
def test_every_scorer_readout_over_other_axes_is_refused_naming_it(name: str) -> None:
    site, spec = the_scorer_s_point(name)
    wrong = ("batch", *spec.axes)
    published = a_scorer_map_redeclaring(name, a_spec(wrong, spec.dtype_class))

    with pytest.raises(ValueError, match="runs over") as refusal:
        ensure_publishes(published)

    message = str(refusal.value)

    assert repr(site) in message
    assert f"over {wrong}, and" in message
    assert f"of role {site.role.name} runs over {ROLE_AXES[site.role]}" in message


@pytest.mark.parametrize("name", READOUTS)
def test_every_scorer_readout_of_another_numeric_class_is_refused_naming_it(
    name: str,
) -> None:
    site, spec = the_scorer_s_point(name)
    wrong = next(dtype for dtype in DTypeClass if dtype is not spec.dtype_class)
    published = a_scorer_map_redeclaring(name, a_spec(spec.axes, wrong))

    with pytest.raises(ValueError, match="holds") as refusal:
        ensure_publishes(published)

    message = str(refusal.value)

    assert repr(site) in message
    assert (
        f"as {wrong.value}, and a point of role {site.role.name} holds "
        f"{ROLE_DTYPE[site.role].value}"
    ) in message


def test_every_kind_of_component_has_an_entry_in_the_inventory() -> None:
    assert set(INVENTORY) == {
        ComponentKind.TOKEN_ENCODER,
        ComponentKind.SUBWORD_POOLER,
        ComponentKind.TERM_ENCODER,
        ComponentKind.TERM_DECODER,
        ComponentKind.CONTROLLER,
        ComponentKind.PARSER,
    }

    for inventory in INVENTORY.values():
        assert inventory.required
        assert inventory.required <= set(ROLE_AXES)


def test_token_rows_and_pooled_words_have_distinct_roles_and_axes() -> None:
    assert INVENTORY[ComponentKind.TOKEN_ENCODER].required == {Role.TOKEN_EMBEDDING}
    assert INVENTORY[ComponentKind.SUBWORD_POOLER].required == {Role.WORD_VECTOR}
    assert ROLE_AXES[Role.TOKEN_EMBEDDING] == ("item", "row", "width")
    assert ROLE_AXES[Role.WORD_VECTOR] == ("item", "word", "width")


def test_term_vectors_and_node_encodings_have_distinct_roles_and_axes() -> None:
    inventory = INVENTORY[ComponentKind.TERM_ENCODER]

    assert inventory.required == {
        Role.TERM_ENCODING,
        Role.NODE_ENCODING,
    }
    assert inventory.capabilities == {
        "recurrence": {
            frozenset({Role.RECURRENT_HIDDEN}),
            frozenset({Role.RECURRENT_HIDDEN, Role.GATE}),
            frozenset({Role.RECURRENT_HIDDEN, Role.RECURRENT_CELL, Role.GATE}),
        }
    }
    assert ROLE_AXES[Role.TERM_ENCODING] == ("term", "width")
    assert ROLE_AXES[Role.NODE_ENCODING] == ("term", "slot", "width")


def test_term_decoders_publish_symbol_distributions_and_one_kind_of_state() -> None:
    inventory = INVENTORY[ComponentKind.TERM_DECODER]

    assert inventory.required == {Role.SYMBOL_DISTRIBUTION}
    assert inventory.alternatives == {
        "state": {
            Role.DECODER_RECURRENT_STATE,
            Role.UNBOUND_VECTOR,
            Role.DECODER_RESIDUAL_STREAM,
        }
    }
    assert ROLE_AXES[Role.SYMBOL_DISTRIBUTION] == ("term", "slot", "symbol")
    assert ROLE_DTYPE[Role.SYMBOL_DISTRIBUTION] is DTypeClass.LOG_PROBABILITY


def test_controllers_publish_log_distributions_over_cell_actions() -> None:
    assert INVENTORY[ComponentKind.CONTROLLER].required == {Role.COMBINATOR_LOGITS}
    assert ROLE_AXES[Role.COMBINATOR_LOGITS] == ("cell", "action")
    assert ROLE_DTYPE[Role.COMBINATOR_LOGITS] is DTypeClass.LOG_PROBABILITY


@pytest.mark.parametrize(
    "state",
    [
        Role.DECODER_RECURRENT_STATE,
        Role.UNBOUND_VECTOR,
        Role.DECODER_RESIDUAL_STREAM,
    ],
)
def test_each_term_decoder_state_alternative_is_allowed(state: Role) -> None:
    published = (
        SiteMapBuilder(DECODER, ComponentKind.TERM_DECODER)
        .publish(Role.SYMBOL_DISTRIBUTION)
        .publish(state)
        .build()
    )

    assert ensure_publishes(published) is published


def test_a_term_decoder_with_no_state_alternative_is_refused() -> None:
    decoder = SiteMapBuilder(DECODER, ComponentKind.TERM_DECODER)
    decoder.publish(Role.SYMBOL_DISTRIBUTION)

    with pytest.raises(ValueError, match="state has no alternative") as refusal:
        decoder.build()

    assert "DECODER_RECURRENT_STATE" in str(refusal.value)
    assert "DECODER_RESIDUAL_STREAM" in str(refusal.value)
    assert "UNBOUND_VECTOR" in str(refusal.value)


def test_a_term_decoder_with_two_state_alternatives_is_refused() -> None:
    decoder = SiteMapBuilder(DECODER, ComponentKind.TERM_DECODER)
    decoder.publish(Role.SYMBOL_DISTRIBUTION)
    decoder.publish(Role.UNBOUND_VECTOR).publish(Role.DECODER_RESIDUAL_STREAM)

    with pytest.raises(ValueError, match="more than one alternative") as refusal:
        decoder.build()

    assert "Exactly one of" in str(refusal.value)


def test_a_map_naming_every_required_role_is_returned_unchanged() -> None:
    built = tagged(ComponentKind.PARSER, a_whole_parser().build())
    checked = ensure_publishes(built)

    assert checked == built
    assert len(built) == 4


def test_a_map_missing_a_required_role_is_refused_naming_what_is_missing() -> None:
    thinner = tagged(
        ComponentKind.PARSER,
        a_parser(Role.CHART_CELL, Role.INSIDE_SCORE, Role.SPLIT_DISTRIBUTION).build(),
    )

    with pytest.raises(ValueError, match="OUTSIDE_SCORE") as refusal:
        ensure_publishes(thinner)

    assert PARSER in str(refusal.value)
    assert "parser" in str(refusal.value)


def an_encoder(*roles: Role) -> SiteMapBuilder:
    builder = (
        SiteMapBuilder.custom(ENCODER)
        .publish(Role.TERM_ENCODING)
        .publish(Role.NODE_ENCODING)
    )

    for role in roles:
        builder.publish(role)

    return builder


def test_half_a_structure_is_refused_naming_the_other_half() -> None:
    half = tagged(
        ComponentKind.TERM_ENCODER,
        an_encoder(Role.RESIDUAL_STREAM).build(),
    )

    with pytest.raises(ValueError, match="FFN_PREACTIVATION") as refusal:
        ensure_publishes(half)

    assert "residual" in str(refusal.value)


def test_a_whole_structure_is_allowed() -> None:
    whole = tagged(
        ComponentKind.TERM_ENCODER,
        an_encoder(Role.RESIDUAL_STREAM, Role.FFN_PREACTIVATION).build(),
    )

    assert ensure_publishes(whole) is whole

    assert len(whole) == 4


def test_half_a_structure_of_roles_with_no_entry_is_refused() -> None:
    half = a_whole_parser().publish_at(
        Role.ATTENTION_WEIGHTS, a_spec(("head",), DTypeClass.PROBABILITY)
    )

    with pytest.raises(ValueError, match="ATTENTION_LOGITS") as refusal:
        ensure_publishes(tagged(ComponentKind.PARSER, half.build()))

    assert "attention" in str(refusal.value)


def test_a_structure_no_part_of_which_is_published_is_owed_by_nobody() -> None:
    encoder = (
        SiteMapBuilder(ENCODER, ComponentKind.TERM_ENCODER)
        .publish(Role.TERM_ENCODING)
        .publish(Role.NODE_ENCODING)
    )

    assert len(encoder.build()) == 2


def a_recurrent_encoder(*roles: Role) -> SiteMapBuilder:
    encoder = SiteMapBuilder(ENCODER, ComponentKind.TERM_ENCODER)
    encoder.publish(Role.TERM_ENCODING).publish(Role.NODE_ENCODING)

    for role in roles:
        meaning = DTypeClass.PROBABILITY if role is Role.GATE else DTypeClass.REAL
        encoder.publish_at(role, a_spec(("step", "width"), meaning))

    return encoder


@pytest.mark.parametrize(
    "roles",
    [
        (Role.RECURRENT_HIDDEN,),
        (Role.RECURRENT_HIDDEN, Role.GATE),
        (Role.RECURRENT_HIDDEN, Role.RECURRENT_CELL, Role.GATE),
    ],
    ids=("vanilla", "gru", "lstm"),
)
def test_each_declared_recurrent_capability_set_is_allowed(
    roles: tuple[Role, ...],
) -> None:
    built = a_recurrent_encoder(*roles).build()

    assert {site.role for site in built} == {
        Role.TERM_ENCODING,
        Role.NODE_ENCODING,
        *roles,
    }


@pytest.mark.parametrize(
    "roles",
    [
        (Role.GATE,),
        (Role.RECURRENT_CELL,),
        (Role.RECURRENT_CELL, Role.GATE),
        (Role.RECURRENT_HIDDEN, Role.RECURRENT_CELL),
    ],
    ids=("gate-without-hidden", "cell-alone", "cell-and-gate", "cell-without-gate"),
)
def test_undeclared_recurrent_role_mixtures_are_refused(
    roles: tuple[Role, ...],
) -> None:
    encoder = a_recurrent_encoder(*roles)

    with pytest.raises(
        ValueError, match="not a declared recurrence capability"
    ) as refusal:
        encoder.build()

    assert all(role.name in str(refusal.value) for role in roles)
    assert "RECURRENT_HIDDEN" in str(refusal.value)
    assert "GATE" in str(refusal.value)


def test_a_point_whose_axes_are_not_its_role_s_is_refused() -> None:
    published = SiteMap(
        PARSER,
        ComponentKind.PARSER,
        {
            Site(PARSER, Role.CHART_CELL): SiteSpec(
                ROLE_AXES[Role.CHART_CELL],
                ROLE_DTYPE[Role.CHART_CELL],
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(PARSER, Role.INSIDE_SCORE): SiteSpec(
                ("cell", "width"),
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(PARSER, Role.OUTSIDE_SCORE): SiteSpec(
                ROLE_AXES[Role.OUTSIDE_SCORE],
                ROLE_DTYPE[Role.OUTSIDE_SCORE],
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(PARSER, Role.SPLIT_DISTRIBUTION): SiteSpec(
                ROLE_AXES[Role.SPLIT_DISTRIBUTION],
                ROLE_DTYPE[Role.SPLIT_DISTRIBUTION],
                differentiable=True,
                access=Access.OBSERVE,
            ),
        },
    )

    with pytest.raises(ValueError, match="runs over") as refusal:
        ensure_publishes(published)

    assert "of role INSIDE_SCORE runs over ('item', 'span')" in str(refusal.value)
    assert "'cell', 'width'" in str(refusal.value)


def test_a_point_whose_numbers_are_not_its_role_s_is_refused() -> None:
    changed = dict(a_whole_parser().build())
    point = Site(PARSER, Role.SPLIT_DISTRIBUTION)
    changed[point] = SiteSpec(
        ROLE_AXES[Role.SPLIT_DISTRIBUTION],
        DTypeClass.PROBABILITY,
        differentiable=True,
        access=Access.OBSERVE,
    )
    published = SiteMap(PARSER, ComponentKind.PARSER, changed)

    with pytest.raises(ValueError, match="holds log-probability") as refusal:
        ensure_publishes(published)

    assert repr(point) in str(refusal.value)


def test_a_map_holding_another_component_s_point_is_refused() -> None:
    mixed = dict(a_whole_parser().build())
    mixed[Site(OTHER, Role.TERM_ENCODING)] = SiteSpec(
        ROLE_AXES[Role.TERM_ENCODING],
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE,
    )

    with pytest.raises(ValueError, match="belongs to") as refusal:
        SiteMap(PARSER, ComponentKind.PARSER, mixed)

    assert OTHER in str(refusal.value)


def test_a_role_declaring_no_axes_is_published_with_its_own_specification() -> None:
    weight = SiteSpec(
        ("row", "column"),
        DTypeClass.REAL,
        differentiable=True,
        access=Access.OBSERVE,
    )
    builder = SiteMapBuilder.custom(ENCODER)

    with pytest.raises(KeyError, match="publish_at"):
        builder.publish(Role.PARAMETER)

    builder.publish_at(Role.PARAMETER, weight, key=("scale",))
    built = builder.build()

    assert built[Site(ENCODER, Role.PARAMETER, ("scale",))] == weight


def test_a_point_published_twice_is_refused() -> None:
    builder = a_parser(Role.CHART_CELL)

    with pytest.raises(ValueError, match="already published"):
        builder.publish(Role.CHART_CELL)


def test_two_points_of_one_role_are_told_apart_by_their_key() -> None:
    builder = SiteMapBuilder.custom(ENCODER)
    builder.publish(Role.TERM_ENCODING, (0,)).publish(Role.TERM_ENCODING, (1,))

    assert len(builder.build()) == 2


def test_a_published_point_carries_the_access_and_meaning_its_role_declares() -> None:
    built = (
        SiteMapBuilder.custom(ENCODER)
        .publish(Role.CONSTITUENT_MARGINAL)
        .publish(Role.TERM_ENCODING, patchable=False, differentiable=False)
        .build()
    )
    marginal = built[Site(ENCODER, Role.CONSTITUENT_MARGINAL)]
    encoding = built[Site(ENCODER, Role.TERM_ENCODING)]

    assert marginal.dtype_class is DTypeClass.PROBABILITY
    assert marginal.access == Access.OBSERVE | Access.PATCH
    assert marginal.axes == ("item", "span")
    assert marginal.differentiable
    assert encoding.access == Access.OBSERVE
    assert not encoding.differentiable


def test_a_builder_hands_out_a_map_it_does_not_go_on_owning() -> None:
    builder = a_whole_parser()
    built = dict(builder.build())
    built.clear()

    assert len(builder.build()) == 4


def test_an_inventory_refuses_what_is_not_a_role_and_an_empty_group() -> None:
    with pytest.raises(TypeError, match="is a Role"):
        RoleInventory(cast("list[Role]", ["TERM_ENCODING"]))

    with pytest.raises(ValueError, match="names no role"):
        RoleInventory((Role.GATE,), {"attention": ()})

    with pytest.raises(ValueError, match="empty"):
        RoleInventory((Role.GATE,), {"": (Role.ATTENTION_LOGITS,)})

    with pytest.raises(ValueError, match="names no role"):
        RoleInventory((Role.GATE,), alternatives={"state": ()})

    with pytest.raises(ValueError, match="empty"):
        RoleInventory((Role.GATE,), alternatives={"": (Role.DECODER_RECURRENT_STATE,)})

    with pytest.raises(ValueError, match="declares no set"):
        RoleInventory((Role.GATE,), capabilities={"recurrence": ()})

    with pytest.raises(ValueError, match="declares an empty set"):
        RoleInventory((Role.GATE,), capabilities={"recurrence": ((),)})

    with pytest.raises(ValueError, match="empty"):
        RoleInventory((Role.GATE,), capabilities={"": ((Role.RECURRENT_HIDDEN,),)})


def table_rows(headed: set[str]) -> dict[str, tuple[str, ...]]:
    rows: dict[str, tuple[str, ...]] = {}

    for line in PROSE.read_text(encoding="utf-8").splitlines():
        stripped = line.strip()

        if not stripped.startswith("|"):
            continue

        cells = tuple(cell.strip() for cell in stripped.strip("|").split("|"))
        heading = cells[0].strip("`")

        if heading in headed:
            rows[heading] = cells

    return rows


def listed_names(cell: str) -> tuple[str, ...]:
    return tuple(part.strip().strip("`") for part in cell.split(",") if part.strip())


def listed_structures(cell: str) -> dict[str, frozenset[str]]:
    if cell == "none":
        return {}

    grouped: dict[str, frozenset[str]] = {}

    for group in cell.split(";"):
        name, _, members = group.partition(":")
        grouped[name.strip().strip("`")] = frozenset(listed_names(members))

    return grouped


def listed_capabilities(cell: str) -> dict[str, frozenset[frozenset[str]]]:
    if cell == "none":
        return {}

    grouped: dict[str, frozenset[frozenset[str]]] = {}

    for group in cell.split(";"):
        name, _, alternatives = group.partition(":")
        sets = (
            frozenset(listed_names(alternative.strip().strip("()")))
            for alternative in alternatives.split("/")
        )
        grouped[name.strip().strip("`")] = frozenset(sets)

    return grouped


def test_the_prose_states_the_axes_and_the_numbers_of_every_role() -> None:
    rows = table_rows(set(Role.__members__))

    assert set(rows) == {role.name for role in ROLE_AXES}
    assert Role.PARAMETER.name not in rows
    assert Role.PARAMETER.name in PROSE.read_text(encoding="utf-8")

    for role, axes in ROLE_AXES.items():
        assert listed_names(rows[role.name][1]) == axes
        assert rows[role.name][2].strip("`") == ROLE_DTYPE[role].value


def test_the_prose_states_what_each_kind_of_component_owes() -> None:
    rows = table_rows({kind.value for kind in INVENTORY})

    assert set(rows) == {kind.value for kind in INVENTORY}

    for kind, inventory in INVENTORY.items():
        required = frozenset(listed_names(rows[kind.value][1]))
        structures = listed_structures(rows[kind.value][2])
        alternatives = listed_structures(rows[kind.value][3])
        capabilities = listed_capabilities(rows[kind.value][4])

        assert required == {role.name for role in inventory.required}
        assert structures == {
            name: frozenset(role.name for role in group)
            for name, group in inventory.structures.items()
        }
        assert alternatives == {
            name: frozenset(role.name for role in group)
            for name, group in inventory.alternatives.items()
        }
        assert capabilities == {
            name: frozenset(
                frozenset(role.name for role in roles) for roles in alternatives
            )
            for name, alternatives in inventory.capabilities.items()
        }


def test_an_inventory_says_what_it_holds() -> None:
    parser = INVENTORY[ComponentKind.PARSER]

    assert "1 required" in repr(parser)
    assert "attention" in repr(parser)
    assert "deep" in repr(parser)
    assert "SiteMapBuilder" in repr(SiteMapBuilder(PARSER, ComponentKind.PARSER))
    assert parser.roles == parser.required | {
        Role.ATTENTION_LOGITS,
        Role.ATTENTION_WEIGHTS,
        Role.INSIDE_SCORE,
        Role.OUTSIDE_SCORE,
        Role.SPLIT_DISTRIBUTION,
        Role.OUTSIDE_CELL,
        Role.SPAN_REPRESENTATION,
    }
