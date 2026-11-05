from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec.evaluators.alignment import (
    AlignmentDeclaration,
    AlignmentKind,
    AlignmentSettings,
    AlignmentSpec,
    AlignmentTieBreak,
    IdentityAlignmentSpec,
    ManyToOneAlignmentSpec,
    OneToOneAlignmentSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def _declared(kind: AlignmentKind) -> dict[str, JsonValue]:
    declaration: dict[str, JsonValue] = {
        "kind": kind,
        "fit_on": "dev",
        "score_on": "test",
        "random_floor": {"replicates": 31, "level": 0.9},
        "stability": {"replicates": 41, "level": 0.95},
    }

    if kind != "identity":
        declaration["tie_break"] = "least-gold-label"

    return declaration


@pytest.mark.parametrize(
    ("kind", "variant", "tie_break"),
    [
        ("identity", IdentityAlignmentSpec, None),
        ("many-to-one", ManyToOneAlignmentSpec, "least-gold-label"),
        ("one-to-one", OneToOneAlignmentSpec, "least-gold-label"),
    ],
)
def test_each_alignment_policy_is_declared_with_held_out_fitting(
    kind: AlignmentKind,
    variant: type[AlignmentSpec],
    tie_break: AlignmentTieBreak | None,
) -> None:
    declared = AlignmentSpec.model_validate(_declared(kind))

    assert type(declared) is variant
    assert cast("AlignmentDeclaration", declared).settings() == AlignmentSettings(
        kind=kind,
        tie_break=tie_break,
        fit_on="dev",
        score_on="test",
        floor_replicates=31,
        floor_level=0.9,
        stability_replicates=41,
        stability_level=0.95,
    )
    assert AlignmentSpec.model_validate_json(declared.model_dump_json()) == declared


def test_alignment_partition_names_follow_the_dataset_vocabulary() -> None:
    declared = AlignmentSpec.model_validate(
        _declared("one-to-one") | {"fit_on": "development", "score_on": "test"}
    )

    assert cast("AlignmentDeclaration", declared).settings().fit_on == "development"


@pytest.mark.parametrize(
    "payload",
    [
        _declared("identity") | {"tie_break": "least-gold-label"},
        _declared("many-to-one") | {"tie_break": "first-seen"},
        _declared("one-to-one") | {"fit_on": ""},
        _declared("one-to-one") | {"fit_on": "test"},
        _declared("one-to-one") | {"score_on": "dev"},
        {
            key: value
            for key, value in _declared("many-to-one").items()
            if key != "tie_break"
        },
    ],
)
def test_an_alignment_refuses_an_unused_unknown_or_leaking_choice(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        AlignmentSpec.model_validate(payload)
