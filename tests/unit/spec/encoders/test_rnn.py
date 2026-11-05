from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.components.model import TermEncoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_the_tag_reaches_the_recurrent_variant_through_the_family_root() -> None:
    declared = TermEncoderSpec.model_validate(
        {
            "kind": "tree-rnn",
            "width": 16,
            "cell": "lstm",
            "layers": 2,
            "norm_cap": 3.0,
            "inventory": "shared-dag",
        }
    )

    assert type(declared) is TreeRnnEncoderSpec
    assert declared == TreeRnnEncoderSpec(
        width=16,
        cell="lstm",
        layers=2,
        norm_cap=3.0,
        inventory="shared-dag",
    )
    assert TermEncoderSpec.__variants__["tree-rnn"] is TreeRnnEncoderSpec


def test_one_layer_runs_and_constructor_vectors_are_read_as_stored_by_default() -> None:
    declared = TreeRnnEncoderSpec(width=16)

    assert declared.cell == "gru"
    assert declared.layers == 1
    assert declared.norm_cap is None
    assert declared.inventory == "uncached"
    assert declared.dependencies() == ()


def test_the_cell_is_any_installed_recurrent_cell_name() -> None:
    declared = TreeRnnEncoderSpec(width=16, cell="elman-relu")

    assert declared.settings().cell == "elman-relu"


def test_the_encoder_lends_no_submodule_to_the_components_it_supplies() -> None:
    assert TreeRnnEncoderSpec(width=16).publications() == frozenset()


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"width": 0}, "a state of no coordinates holds nothing"),
        ({"width": 16, "layers": 0}, "a recurrence has a layer"),
        ({"width": 16, "norm_cap": 0.0}, "a norm cap is above zero"),
        ({"width": 16, "norm_cap": -1.0}, "a norm cap is above zero"),
        ({"width": 16, "cell": ""}, "recurrent cell"),
    ],
)
def test_an_out_of_range_dimension_is_refused_at_the_boundary(
    payload: dict[str, JsonValue], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TermEncoderSpec.model_validate({"kind": "tree-rnn", **payload})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "tree-rnn", "width": 16, "cell": "gru", "layers": 1, '
        '"norm_cap": null, "inventory": "uncached", "declared_output": null}'
    )
    declared = TermEncoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TermEncoderSpec.model_validate_json(declared.model_dump_json()) == declared
