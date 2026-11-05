import didactic.api as dx
import pytest

from lofi.spec.components.model import TokenEncoderSpec
from lofi.spec.tokens.learned import LearnedTokenEncoderSpec


def test_the_tag_reaches_the_learned_variant_through_the_family_root() -> None:
    declared = TokenEncoderSpec.model_validate({"kind": "learned", "width": 64})

    assert type(declared) is LearnedTokenEncoderSpec
    assert declared == LearnedTokenEncoderSpec(width=64)
    assert TokenEncoderSpec.__variants__["learned"] is LearnedTokenEncoderSpec


def test_nothing_is_taken_from_the_graph() -> None:
    declared = LearnedTokenEncoderSpec(width=16)

    assert declared.dependencies() == ()
    assert declared.declared_output is None


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "learned"}, "width"),
        ({"kind": "learned", "width": 0}, "no features"),
        ({"kind": "learned", "width": -1}, "no features"),
        ({"kind": "learned", "width": 8, "vectors": "v.txt"}, "vectors"),
        ({"kind": "learned", "width": 8, "trainable": False}, "trainable"),
    ],
)
def test_a_missing_nonpositive_or_extra_setting_is_refused(
    payload: dict[str, str | int | bool], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TokenEncoderSpec.model_validate(payload)


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "learned", "width": 256, "init_std": 0.125, "declared_output": null}'
    )
    declared = TokenEncoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TokenEncoderSpec.model_validate_json(declared.model_dump_json()) == declared
