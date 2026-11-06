from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec.spaces import (
    CombinatorFamilySpec,
    Merge,
    SpaceMapForm,
    SpaceRelation,
    SpaceRelationSpec,
    SpaceSpec,
    TypeSpaceMapSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_a_space_binds_one_sort_to_named_components() -> None:
    declared = SpaceSpec.model_validate(
        {
            "sort": "Tm",
            "encoder": "terms",
            "identity": "lf-identity",
            "combinators": {
                "merge": "concatenate",
                "decoders": {
                    "backward-application": "lf-apply",
                    "forward-application": "lf-apply",
                },
            },
        }
    )

    assert declared == SpaceSpec(
        sort="Tm",
        encoder="terms",
        identity="lf-identity",
        combinators=CombinatorFamilySpec(
            merge=Merge.CONCATENATE,
            decoders={
                "backward-application": "lf-apply",
                "forward-application": "lf-apply",
            },
        ),
    )
    assert tuple(cast("CombinatorFamilySpec", declared.combinators).decoders) == (
        "backward-application",
        "forward-application",
    )


def test_a_unary_only_family_declares_no_binary_decoder_or_merge() -> None:
    declared = CombinatorFamilySpec.model_validate({"decoders": {}})

    assert declared.decoders == {}
    assert declared.merge is None
    assert declared.chart_sharing() is None


@pytest.mark.parametrize(
    "payload",
    [
        {"merge": "concatenate", "decoders": {}},
        {"decoders": {"apply": "decoder"}},
    ],
)
def test_binary_decoder_bindings_and_their_merge_are_declared_together(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError, match="exactly"):
        CombinatorFamilySpec.model_validate(payload)


def test_a_category_space_declares_its_relation_to_another_space() -> None:
    declared = SpaceSpec(
        sort="Cat",
        encoder="terms",
        identity="cat-identity",
        relation=SpaceRelationSpec(
            space="types",
            via=SpaceRelation.MAPPED,
            mapping=TypeSpaceMapSpec(form=SpaceMapForm.LINEAR),
        ),
    )

    assert declared.relation == SpaceRelationSpec(
        space="types",
        via=SpaceRelation.MAPPED,
        mapping=TypeSpaceMapSpec(form=SpaceMapForm.LINEAR),
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"space": "types", "via": "mapped"},
        {
            "space": "types",
            "via": "consistency",
            "mapping": {"form": "linear"},
        },
        {
            "space": "types",
            "via": "mapped",
            "mapping": {
                "form": "multilayer",
                "hidden": 4,
                "depth": 2,
            },
        },
    ],
)
def test_a_relation_refuses_an_incomplete_or_unused_map(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        SpaceRelationSpec.model_validate(payload)


def test_a_multilayer_relation_map_keeps_its_declared_architecture() -> None:
    mapping = TypeSpaceMapSpec.model_validate(
        {"form": "multilayer", "hidden": 7, "depth": 3, "dropout": 0.25}
    )

    assert mapping.settings() == (SpaceMapForm.MULTILAYER, 7, 3, 0.25)
    assert mapping.model_dump()["kind"] == "type-space-map"


@pytest.mark.parametrize(
    ("payload", "named"),
    [
        ({"sort": "", "encoder": "terms", "identity": "identity"}, "sort"),
        ({"sort": "Tm", "encoder": "", "identity": "identity"}, "encoder"),
        ({"sort": "Tm", "encoder": "terms", "identity": ""}, "identity"),
        (
            {
                "sort": "Tm",
                "encoder": "terms",
                "identity": "identity",
                "combinators": {
                    "merge": "concatenate",
                    "decoders": {"": "apply"},
                },
            },
            "action",
        ),
        (
            {
                "sort": "Tm",
                "encoder": "terms",
                "identity": "identity",
                "combinators": {
                    "merge": "concatenate",
                    "decoders": {"apply": ""},
                },
            },
            "decoder",
        ),
        (
            {
                "sort": "Cat",
                "encoder": "terms",
                "identity": "identity",
                "relation": {"space": "", "via": "consistency"},
            },
            "space",
        ),
    ],
)
def test_an_unnamed_space_binding_is_refused(
    payload: dict[str, JsonValue], named: str
) -> None:
    with pytest.raises(dx.ValidationError) as failure:
        SpaceSpec.model_validate(payload)

    assert named in str(failure.value)
