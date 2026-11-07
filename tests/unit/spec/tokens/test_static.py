import didactic.api as dx
import pytest

from lofi.spec.components.model import TokenEncoderSpec
from lofi.spec.tokens.static import StaticTokenEncoderSpec


def test_the_tag_reaches_the_static_variant_through_the_family_root() -> None:
    declared = TokenEncoderSpec.model_validate(
        {"kind": "static", "vectors": "vectors.txt", "vectors_format": "word2vec"}
    )

    assert type(declared) is StaticTokenEncoderSpec
    assert declared == StaticTokenEncoderSpec(
        vectors="vectors.txt", vectors_format="word2vec"
    )
    assert TokenEncoderSpec.__variants__["static"] is StaticTokenEncoderSpec


def test_the_layout_defaults_to_glove_and_nothing_is_taken_from_the_graph() -> None:
    declared = StaticTokenEncoderSpec(vectors="vectors.txt")

    assert declared.vectors_format == "glove"
    assert not declared.trainable
    assert declared.dependencies() == ()
    assert declared.declared_output is None


def test_the_embedding_table_can_be_declared_trainable() -> None:
    declared = TokenEncoderSpec.model_validate(
        {"kind": "static", "vectors": "vectors.txt", "trainable": True}
    )

    assert isinstance(declared, StaticTokenEncoderSpec)
    assert declared.trainable


@pytest.mark.parametrize(
    ("payload", "detail"),
    [
        ({"kind": "static"}, "vectors"),
        ({"kind": "static", "vectors": ""}, "a vectors file"),
        ({"kind": "static", "vectors": "v.txt", "trainable": "yes"}, "trainable"),
    ],
)
def test_a_missing_empty_unknown_or_extra_setting_is_refused(
    payload: dict[str, str | bool], detail: str
) -> None:
    with pytest.raises(dx.ValidationError, match=detail):
        TokenEncoderSpec.model_validate(payload)


def test_a_vector_format_name_is_open_to_installed_plugins() -> None:
    declared = StaticTokenEncoderSpec(vectors="v.txt", vectors_format="fasttext")

    assert declared.vectors_format == "fasttext"


def test_an_unknown_tag_is_refused_with_the_tag_named() -> None:
    with pytest.raises(dx.ValidationError, match="'subword'"):
        TokenEncoderSpec.model_validate({"kind": "subword", "vectors": "v.txt"})


def test_the_variant_round_trips_through_its_json_form_byte_for_byte() -> None:
    document = (
        '{"kind": "static", "vectors": "vectors.txt", '
        '"vectors_format": "word2vec", "trainable": false, '
        '"lowercase": true, "oov_init": "normal", '
        '"oov_init_std": 0.1, "seed": 13, '
        '"declared_output": null}'
    )
    declared = TokenEncoderSpec.model_validate_json(document)

    assert declared.model_dump_json() == document
    assert TokenEncoderSpec.model_validate_json(declared.model_dump_json()) == declared
