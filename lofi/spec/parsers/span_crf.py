"""Globally normalized span-parser declaration."""

from typing import Final, Literal, NamedTuple, override

import didactic.api as dx

from lofi.spec._ports import Dependency
from lofi.spec.components.model import ParserSpec

__all__ = ["SpanCrfParserSettings", "SpanCrfParserSpec"]

_DEPENDENCIES: Final[tuple[Dependency, ...]] = (Dependency("subword_pooler", "leaves"),)
"""The pooled word vectors from which the scalar potentials are computed."""


class SpanCrfParserSettings(NamedTuple):
    """Resolved settings of a globally normalized span parser."""

    hidden: int


class SpanCrfParserSpec(ParserSpec):
    """Declare a scalar-potential CRF over binary span trees.

    The parser derives its input width from the pooled-word producer. It stores no
    chart vectors and has no controller dependency: its only trainable chart values
    are lexical and binary-split potentials. Log-sum-exp is the parser's defining
    reduction, so it is represented by the parser kind rather than exposed as a
    switch that could silently destroy the posterior semantics.
    """

    kind: Literal["span-crf"] = "span-crf"
    hidden: int

    __axioms__ = (
        dx.axiom("hidden > 0", message="a span-CRF hidden layer has positive width"),
    )

    @override
    def dependencies(self) -> tuple[Dependency, ...]:
        """Return the pooled words from which scalar potentials are scored."""
        return _DEPENDENCIES

    def chart_outputs(self) -> frozenset[str]:
        """Return the complete globally normalized chart analysis."""
        return frozenset(
            {"inside", "outside", "alpha", "mu", "log_z", "entropy", "mbr"}
        )

    def settings(self) -> SpanCrfParserSettings:
        """Return the hidden width used to score binary splits."""
        return SpanCrfParserSettings(self.hidden)
