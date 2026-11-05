"""Tests for cross-lingual transfer identities."""

from typing import TYPE_CHECKING, cast

import pytest

from lofi.eval._transfer import TransferLanguage, UnavailableLanguage

if TYPE_CHECKING:
    from lofi.eval._transfer import SourceClass


def _replace(value: object, name: str, replacement: object) -> None:
    setattr(value, name, replacement)


def test_a_transfer_language_is_a_frozen_serializable_identity() -> None:
    language = TransferLanguage("de", "gold")

    assert language.code == "de"
    assert language.source == "gold"
    assert language.to_spec() == {"language": "de", "source": "gold"}
    assert language == TransferLanguage("de", "gold")
    assert hash(language) == hash(TransferLanguage("de", "gold"))

    with pytest.raises(AttributeError):
        _replace(language, "code", "en")


def test_a_transfer_language_refuses_an_undeclared_annotation_class() -> None:
    with pytest.raises(ValueError, match="annotation class"):
        TransferLanguage("de", cast("SourceClass", "bronze"))


def test_only_a_silver_language_may_be_unavailable() -> None:
    language = TransferLanguage("es", "silver")
    unavailable = UnavailableLanguage(language, "no converted treebank")

    assert unavailable.language is language
    assert unavailable.reason == "no converted treebank"
    assert unavailable == UnavailableLanguage(language, "no converted treebank")
    assert hash(unavailable) == hash(
        UnavailableLanguage(language, "no converted treebank")
    )

    with pytest.raises(ValueError, match="only a silver language"):
        UnavailableLanguage(TransferLanguage("de", "gold"), "missing")


def test_an_unavailable_language_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        UnavailableLanguage(TransferLanguage("es", "silver"), "")
