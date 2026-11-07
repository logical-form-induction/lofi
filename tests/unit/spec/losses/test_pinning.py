import didactic.api as dx
import pytest

from lofi.spec import Dependency
from lofi.spec.components.loss import LossSpec
from lofi.spec.losses.pinning import PinningLossSpec


def test_the_pinning_variant_declares_its_term_readout_dependency() -> None:
    declared = LossSpec.model_validate({"kind": "pinning", "readout": "types"})

    assert type(declared) is PinningLossSpec
    assert declared.dependencies() == (Dependency("types", "constraint-readout"),)
    assert declared.constraint_route() == "target"
    assert declared.declared_output is None
    assert declared.required_chart_outputs() == {"alpha", "mu"}


def test_negative_bracket_pinning_has_no_term_readout_dependency() -> None:
    declared = LossSpec.model_validate({"kind": "pinning"})

    assert type(declared) is PinningLossSpec
    assert declared.readout is None
    assert declared.dependencies() == ()
    assert declared.constraint_route() == "negative-bracket"
    assert declared.required_chart_outputs() == {"alpha", "mu"}


@pytest.mark.parametrize(
    "payload",
    [
        {"kind": "pinning", "readout": ""},
        {"kind": "pinning", "readout": True},
        {"kind": "pinning", "readout": "types", "margin": 0.1},
    ],
)
def test_an_empty_wrong_typed_or_extra_field_is_refused(
    payload: dict[str, str | bool | float],
) -> None:
    with pytest.raises(dx.ValidationError):
        LossSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_current_json_shape() -> None:
    document = '{"kind": "pinning", "readout": "types", "declared_output": null}'
    declared = LossSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert LossSpec.model_validate_json(declared.model_dump_json()) == declared
