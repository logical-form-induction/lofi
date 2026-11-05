"""Leaf-lexicon declarations at the language boundary."""

import json
from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.language import (
    InlineLeafLexiconSpec,
    LanguageIso,
    PathLeafLexiconSpec,
    registered_language,
)

if TYPE_CHECKING:
    from pathlib import Path


def _inline() -> InlineLeafLexiconSpec:
    """Return a representative typed-and-inferred leaf source."""
    return InlineLeafLexiconSpec.model_validate(
        {
            "kind": "inline",
            "entries": [
                {"name": "jo", "type": "e", "classes": ["individual"]},
                {"name": "sleep", "type": "infer", "classes": ["predicate"]},
            ],
        }
    )


def test_an_inline_leaf_lexicon_supplies_language_constants() -> None:
    source = registered_language("propositional").with_(leaf_lexicon=_inline())
    language = LanguageIso().infer(source, (("sleep(jo)", "t"),))
    recorded = LanguageIso().backward(language)

    assert language.print_type(language.scheme("jo").body) == "e"
    assert language.print_type(language.scheme("sleep").body) == "e -> t"
    assert recorded.leaf_lexicon is None
    assert tuple(constant.name for constant in recorded.constants[-2:]) == (
        "jo",
        "sleep",
    )


def test_a_path_leaf_lexicon_is_decoded_by_its_didactic_document(
    tmp_path: Path,
) -> None:
    path = tmp_path / "tiny.json"
    path.write_text(
        json.dumps(
            {
                "entries": [
                    {"name": "jo", "type": "e", "classes": ["individual"]},
                    {"name": "sleep", "type": "e -> t", "classes": ["predicate"]},
                ]
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    source = registered_language("propositional").with_(
        leaf_lexicon=PathLeafLexiconSpec(path=str(path))
    )
    language = LanguageIso().forward(source)

    assert language.print_type(language.scheme("sleep").body) == "e -> t"
    assert LanguageIso().backward(language).leaf_lexicon is None


def test_a_path_document_is_validated_before_language_lowering(tmp_path: Path) -> None:
    path = tmp_path / "bad.json"
    path.write_text(
        '{"entries": [{"name": "jo", "typo": "e"}]}', encoding="utf-8", newline="\n"
    )
    source = registered_language("propositional").with_(
        leaf_lexicon=PathLeafLexiconSpec(path=str(path))
    )

    with pytest.raises(dx.ValidationError, match="typo"):
        LanguageIso().forward(source)


def test_a_leaf_source_cannot_redeclare_a_language_constant() -> None:
    source = registered_language("propositional").with_(
        leaf_lexicon=InlineLeafLexiconSpec.model_validate(
            {"kind": "inline", "entries": [{"name": "and", "type": "t"}]}
        )
    )

    with pytest.raises(dx.ValidationError, match="declared twice"):
        LanguageIso().forward(source)


def test_the_three_packaged_leaf_lexicons_are_distinct_signature_sources() -> None:
    base = registered_language("propositional")
    languages = {
        name: LanguageIso().forward(
            base.with_(leaf_lexicon=PathLeafLexiconSpec(path=f"{name}.json"))
        )
        for name in ("extensional", "predicate", "superintensional")
    }

    assert {
        name: language.print_type(language.scheme("think").body)
        for name, language in languages.items()
    } == {
        "extensional": "t -> e -> t",
        "predicate": "(v -> t) -> e -> v -> t",
        "superintensional": "((v -> t) -> t) -> e -> v -> t",
    }
