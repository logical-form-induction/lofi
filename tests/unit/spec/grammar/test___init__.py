from typing import TYPE_CHECKING

from lofi.spec import grammar
from lofi.spec.grammar import (
    _archive,
    _erasure,
    _formalism,
    _generation,
    _grammar,
    _identifiers,
    _interpretation,
    _lexicon,
    _lowering,
    _packs,
    _report,
    _signature,
    _values,
)

if TYPE_CHECKING:
    from types import ModuleType


def test_the_package_publishes_what_its_modules_publish() -> None:
    source: dict[str, ModuleType] = {}

    for module in (
        _archive,
        _erasure,
        _formalism,
        _signature,
        _lexicon,
        _grammar,
        _generation,
        _identifiers,
        _interpretation,
        _lowering,
        _packs,
        _report,
        _values,
    ):
        public = tuple(name for name in module.__all__ if not name.endswith("_PLUGIN"))
        source |= dict.fromkeys(public, module)

    assert sorted(grammar.__all__) == sorted(source)

    for name, module in source.items():
        assert vars(grammar)[name] is vars(module)[name]


def test_the_grammar_declaration_and_its_codec_are_public() -> None:
    assert grammar.GrammarDecl is _grammar.GrammarDecl
    assert grammar.GrammarIso is _lowering.GrammarIso
