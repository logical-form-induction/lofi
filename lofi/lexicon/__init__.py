"""Lexical entries and sources."""

from lofi.lexicon._beam import BeamSource, GrowingBeamSource
from lofi.lexicon._clamped import (
    ClampedSource,
    ClampReport,
    DropClampPolicy,
    ErrorPolicy,
    SkipItemPolicy,
)
from lofi.lexicon._errors import (
    ExhaustedBeamError,
    LexiconError,
    UnsatisfiableClampError,
)
from lofi.lexicon._ports import (
    Clamp,
    ClampMode,
    ClampPolicy,
    ClampScope,
    Entry,
    LexicalSource,
    UnsatisfiableAction,
    UnsatisfiablePolicy,
)
from lofi.lexicon._static import StaticLexicon

__all__ = [
    "BeamSource",
    "Clamp",
    "ClampMode",
    "ClampPolicy",
    "ClampReport",
    "ClampScope",
    "ClampedSource",
    "DropClampPolicy",
    "Entry",
    "ErrorPolicy",
    "ExhaustedBeamError",
    "GrowingBeamSource",
    "LexicalSource",
    "LexiconError",
    "SkipItemPolicy",
    "StaticLexicon",
    "UnsatisfiableAction",
    "UnsatisfiableClampError",
    "UnsatisfiablePolicy",
]
