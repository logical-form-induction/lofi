"""FraCaS test-suite support."""

from lofi.data.corpora.fracas._codec import (
    FRACAS_NAME,
    FracasAnswer,
    FracasCodec,
    FracasFile,
    FracasIso,
    FracasProblem,
    FracasSection,
    MalformedFracasError,
)

__all__ = [
    "FRACAS_NAME",
    "FracasAnswer",
    "FracasCodec",
    "FracasFile",
    "FracasIso",
    "FracasProblem",
    "FracasSection",
    "MalformedFracasError",
]
