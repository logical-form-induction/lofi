"""Corpus readers and comparison labels."""

from lofi.data.corpora._comparison import GrammarInventory, SignatureLabels
from lofi.data.corpora.conll2008 import (
    AlignedConll2008Sentence,
    CcgbankSentenceIdentity,
    Conll2008AlignmentError,
    Conll2008Sentence,
    MalformedConll2008Error,
    align_conll2008,
    read_conll2008,
)

__all__ = [
    "AlignedConll2008Sentence",
    "CcgbankSentenceIdentity",
    "Conll2008AlignmentError",
    "Conll2008Sentence",
    "GrammarInventory",
    "MalformedConll2008Error",
    "SignatureLabels",
    "align_conll2008",
    "read_conll2008",
]
