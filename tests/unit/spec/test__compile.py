import pytest

from lofi.spec._compile import (
    GrammarCompileEnvelope,
    LanguageCompileEnvelope,
    canonical_declaration_bytes,
)


def test_compile_envelopes_are_distinct_frozen_route_values() -> None:
    language = LanguageCompileEnvelope(
        source_version="language-source",
        target_version="language-target",
        canonical_declaration=b"{}",
    )
    grammar = GrammarCompileEnvelope(
        source_version="grammar-source",
        target_version="grammar-target",
        canonical_declaration=b"{}",
    )

    assert type(language) is LanguageCompileEnvelope
    assert type(grammar) is GrammarCompileEnvelope
    assert set(language.model_dump()) == {
        "source_version",
        "target_version",
        "canonical_declaration",
    }
    assert set(grammar.model_dump()) == set(language.model_dump())

    with pytest.raises(AttributeError, match="immutable"):
        language.__setattr__("target_version", "other")

    with pytest.raises(AttributeError, match="immutable"):
        grammar.__setattr__("target_version", "other")


class _Declared:
    """A declaration whose JSON projection names its keys in a given order."""

    def __init__(self, projection: str) -> None:
        self._projection = projection

    def model_dump_json(
        self, *, indent: int | None = None, by_alias: bool = False
    ) -> str:
        del indent, by_alias
        return self._projection


def test_canonical_bytes_do_not_depend_on_key_order() -> None:
    forward = _Declared('{"b": {"y": 1, "x": "é"}, "a": [2, 1]}')
    backward = _Declared('{"a": [2, 1], "b": {"x": "é", "y": 1}}')

    assert canonical_declaration_bytes(forward) == canonical_declaration_bytes(backward)
    assert canonical_declaration_bytes(forward) == (
        b'{"a":[2,1],"b":{"x":"\\u00e9","y":1}}'
    )
