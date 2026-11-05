from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec._ports import Dependency
from lofi.spec.components.model import SubwordPoolerSpec
from lofi.spec.layers import FeedForwardLayer, FeedForwardLayerSpec
from lofi.spec.poolers.attention import (
    AttentionSubwordPoolerSettings,
    AttentionSubwordPoolerSpec,
    SubwordContextSettings,
    SubwordContextSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_tag_reaches_the_attention_variant_through_the_family_root() -> None:
    declared = SubwordPoolerSpec.model_validate(
        {
            "kind": "attention",
            "context": {"cell": "lstm", "layers": 2, "bidirectional": False},
            "scorer": [
                {"width": 5, "activation": "relu"},
                {"width": 3, "activation": "tanh", "dropout": 0.1},
            ],
        }
    )

    assert type(declared) is AttentionSubwordPoolerSpec
    assert declared == AttentionSubwordPoolerSpec(
        context=SubwordContextSpec(cell="lstm", layers=2, bidirectional=False),
        scorer=(
            FeedForwardLayerSpec(width=5, activation="relu"),
            FeedForwardLayerSpec(width=3, activation="tanh", dropout=0.1),
        ),
    )
    assert SubwordPoolerSpec.__variants__["attention"] is AttentionSubwordPoolerSpec


def test_the_scorer_reads_rows_as_they_are_unless_a_recurrence_is_declared() -> None:
    declared = AttentionSubwordPoolerSpec()

    assert declared.settings() == AttentionSubwordPoolerSettings(
        context=None, scorer=()
    )
    assert declared.dependencies() == (Dependency("token_encoder", "encoder"),)


def test_an_empty_recurrence_declaration_reads_both_directions_with_one_gru() -> None:
    declared = AttentionSubwordPoolerSpec.model_validate(
        {"kind": "attention", "context": {}}
    )

    assert declared.settings() == AttentionSubwordPoolerSettings(
        context=SubwordContextSettings("gru", layers=1, width=None, bidirectional=True),
        scorer=(),
    )


def test_the_settings_follow_every_declared_field() -> None:
    declared = AttentionSubwordPoolerSpec(
        context=SubwordContextSpec(
            cell="elman-relu", layers=3, width=6, bidirectional=False
        ),
        scorer=(FeedForwardLayerSpec(width=5, activation="gelu", dropout=0.1),),
    )

    assert declared.settings() == AttentionSubwordPoolerSettings(
        context=SubwordContextSettings(
            "elman-relu", layers=3, width=6, bidirectional=False
        ),
        scorer=(FeedForwardLayer(5, "gelu", 0.1),),
    )


@pytest.mark.parametrize(
    ("field", "document"),
    [
        ("heads", {"kind": "attention", "heads": 2}),
        ("temperature", {"kind": "attention", "temperature": 2.0}),
        ("state", {"kind": "attention", "context": {"state": 4}}),
    ],
)
def test_an_unknown_setting_is_refused(
    field: str, document: dict[str, JsonValue]
) -> None:
    with pytest.raises(dx.ValidationError, match=field):
        SubwordPoolerSpec.model_validate(document)


def test_external_layer_names_remain_declared_until_plugin_resolution() -> None:
    declared = SubwordPoolerSpec.model_validate(
        {
            "kind": "attention",
            "context": {"cell": "test-package-cell"},
            "scorer": [{"width": 4, "activation": "test-package-activation"}],
        }
    )

    assert isinstance(declared, AttentionSubwordPoolerSpec)
    assert declared.context is not None
    assert declared.context.cell == "test-package-cell"
    assert declared.scorer == (
        FeedForwardLayerSpec(width=4, activation="test-package-activation"),
    )


@pytest.mark.parametrize("layers", [0, -1])
def test_a_recurrence_of_no_layer_is_refused(layers: int) -> None:
    with pytest.raises(dx.ValidationError, match="a recurrence has a layer"):
        SubwordContextSpec(layers=layers)


@pytest.mark.parametrize("width", [0, -1])
def test_a_state_of_no_coordinate_is_refused(width: int) -> None:
    with pytest.raises(dx.ValidationError, match="width"):
        SubwordContextSpec(width=width)


@pytest.mark.parametrize("width", [0, -1])
def test_a_hidden_scorer_layer_of_no_coordinate_is_refused(width: int) -> None:
    with pytest.raises(dx.ValidationError, match="hidden layer's width"):
        AttentionSubwordPoolerSpec(
            scorer=(FeedForwardLayerSpec(width=width, activation="tanh"),)
        )


def test_the_removed_shared_scorer_width_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="scorer_width"):
        AttentionSubwordPoolerSpec.model_validate(
            {"kind": "attention", "scorer_width": 4}
        )


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "attention", "context": {"cell": "elman-tanh", "layers": 2, '
        '"width": 5, "bidirectional": true}, "scorer": [{"width": 3, '
        '"activation": "silu", "dropout": 0.1}], "declared_output": null}'
    )
    declared = SubwordPoolerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert SubwordPoolerSpec.model_validate_json(declared.model_dump_json()) == declared
