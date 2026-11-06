import didactic.api as dx
import pytest

from lofi.spec._ports import Dependency
from lofi.spec.components.model import SubwordPoolerSpec
from lofi.spec.poolers.mean import MeanSubwordPoolerSpec


def test_the_tag_reaches_the_mean_variant_through_the_family_root() -> None:
    declared = SubwordPoolerSpec.model_validate({"kind": "mean"})

    assert type(declared) is MeanSubwordPoolerSpec
    assert declared == MeanSubwordPoolerSpec()
    assert SubwordPoolerSpec.__variants__["mean"] is MeanSubwordPoolerSpec


def test_the_pooler_takes_its_rows_and_its_width_from_the_token_encoder() -> None:
    assert MeanSubwordPoolerSpec().dependencies() == (
        Dependency("token_encoder", "encoder"),
    )


def test_a_width_is_not_declared_here_because_the_encoder_fixes_it() -> None:
    with pytest.raises(dx.ValidationError, match="width"):
        SubwordPoolerSpec.model_validate({"kind": "mean", "width": 8})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = '{"kind": "mean", "declared_output": null}'
    declared = SubwordPoolerSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert SubwordPoolerSpec.model_validate_json(declared.model_dump_json()) == declared
