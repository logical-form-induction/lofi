"""Scoring-convention declaration shared by held-out evaluators."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_non_negative

__all__ = ["ConventionSettings", "ConventionSpec"]


class ConventionSettings(NamedTuple):
    """The eight validated conventions a held-out evaluator is built under.

    Attributes
    ----------
    punctuation : Literal["keep", "strip"]
        Whether punctuation is scored or stripped first.
    unary_chains : Literal["keep", "collapse"]
        Whether a gold unary chain stays a chain or collapses.
    trivial_spans : Literal["keep", "drop"]
        Whether the whole-sentence span and single-word spans are scored.
    minimum_length : int
        Shortest sentence still scored, after punctuation is removed.
    aggregation : Literal["sentence", "corpus"]
        Whether a score averages sentences or pools units.
    coverage : Literal["ignore", "zero"]
        The penalty for an item without an analysis.
    self_arcs : Literal["allow", "forbid"]
        Whether a token may depend on itself.
    gold_derivation : Literal["canonical", "oracle"]
        Which gold derivation a prediction is scored against.
    """

    punctuation: Literal["keep", "strip"]
    unary_chains: Literal["keep", "collapse"]
    trivial_spans: Literal["keep", "drop"]
    minimum_length: int
    aggregation: Literal["sentence", "corpus"]
    coverage: Literal["ignore", "zero"]
    self_arcs: Literal["allow", "forbid"]
    gold_derivation: Literal["canonical", "oracle"]


class ConventionSpec(dx.Model, extra="forbid"):
    """Declares the eight conventions a held-out row is scored and reported under.

    Every convention is required, so no evaluator scores under a convention the
    run document did not state.

    Parameters
    ----------
    punctuation : Literal["keep", "strip"]
        Whether punctuation is scored or stripped first.
    unary_chains : Literal["keep", "collapse"]
        Whether a gold unary chain stays a chain or collapses.
    trivial_spans : Literal["keep", "drop"]
        Whether the whole-sentence span and single-word spans are scored.
    minimum_length : int
        Shortest sentence still scored, after punctuation is removed.
    aggregation : Literal["sentence", "corpus"]
        Whether a score averages sentences or pools units.
    coverage : Literal["ignore", "zero"]
        The penalty for an item without an analysis.
    self_arcs : Literal["allow", "forbid"]
        Whether a token may depend on itself.
    gold_derivation : Literal["canonical", "oracle"]
        Which gold derivation a prediction is scored against.
    """

    punctuation: Literal["keep", "strip"]
    unary_chains: Literal["keep", "collapse"]
    trivial_spans: Literal["keep", "drop"]
    minimum_length: int
    aggregation: Literal["sentence", "corpus"]
    coverage: Literal["ignore", "zero"]
    self_arcs: Literal["allow", "forbid"]
    gold_derivation: Literal["canonical", "oracle"]

    @dx.validates("minimum_length")
    @classmethod
    def _length_is_non_negative(cls, value: int) -> int:
        """Return a non-negative minimum sentence length."""
        return ensure_non_negative("a minimum sentence length", value)

    def settings(self) -> ConventionSettings:
        """Return the declared conventions as the value an evaluator is built under."""
        return ConventionSettings(
            punctuation=self.punctuation,
            unary_chains=self.unary_chains,
            trivial_spans=self.trivial_spans,
            minimum_length=self.minimum_length,
            aggregation=self.aggregation,
            coverage=self.coverage,
            self_arcs=self.self_arcs,
            gold_derivation=self.gold_derivation,
        )
