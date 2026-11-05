"""Packaged declarative grammars for Parallel Meaning Bank CCG data."""

from pathlib import Path
from typing import Final, Literal

from lofi.spec._interpolation import lower_package_resource
from lofi.spec._loader import SpecLoader
from lofi.spec.grammar import GrammarDecl

__all__ = ["PmbGrammarLanguage", "pmb_grammar"]

type PmbGrammarLanguage = Literal["en", "de", "it", "nl"]

_LANGUAGES: Final = frozenset({"en", "de", "it", "nl"})


def pmb_grammar(language: PmbGrammarLanguage, /) -> GrammarDecl:
    """Load the packaged PMB grammar declaration for one corpus language."""
    if language not in _LANGUAGES:
        names = ", ".join(sorted(_LANGUAGES))
        msg = f"unknown PMB grammar language {language!r}; choose one of {names}"
        raise ValueError(msg)
    path = Path(
        lower_package_resource(
            f"lofi.resource:data/datasets/train/dataset/grammar/pmb/{language}.yaml"
        )
    )
    return SpecLoader().compose_traced(path, schema=GrammarDecl).spec
