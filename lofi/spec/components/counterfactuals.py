"""The declared counterfactual population a probe experiment generates.

A run's grammar already fixes the rules, the lexicon and the goal a
counterfactual pair is derived under, so this section declares only what the
substitution adds: the unary closure depth and reading budget of each string,
how a category is written as a decision, the alignment policy a pair's varied
span must satisfy, the seed a string's pairs are ordered under, and the file of
candidate sentences, one per line with its tokens separated by whitespace.
"""

from math import isfinite
from pathlib import Path
from typing import Final, Literal, NamedTuple, Self

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative

__all__ = [
    "COUNTERFACTUAL_RENDERINGS",
    "MINIMUM_SWEEP_THRESHOLDS",
    "CounterfactualFilterSettings",
    "CounterfactualFilterSpec",
    "CounterfactualRendering",
    "CounterfactualSettings",
    "CounterfactualsSpec",
    "ScoreSourceSpec",
]

type CounterfactualRendering = Literal["declared", "s-expression"]
"""How a category is written as the decision at its cell.

``declared`` writes it in the grammar's declared category syntax and
``s-expression`` in the store's canonical s-expression syntax, the two
spellings a signature's ``syntax`` names.
"""

COUNTERFACTUAL_RENDERINGS: Final[tuple[CounterfactualRendering, ...]] = (
    "declared",
    "s-expression",
)
"""The renderings a declaration may name."""


MINIMUM_SWEEP_THRESHOLDS: Final = 5
"""How many distinct thresholds a filter's sweep declares at least."""


class CounterfactualFilterSettings(NamedTuple):
    """The validated settings of one declared score filter."""

    scores: Path
    source: str
    public: bool
    fit_signature: str | None
    primary: float
    sweep: tuple[float, ...]


class ScoreSourceSpec(dx.Model, extra="forbid"):
    """Declare where a filter's sentence scores came from.

    Parameters
    ----------
    name : str
        The score source's stable name.
    public : bool
        Whether the scores come from independent public data.
    fit_signature : str | None
        The grammatical signature the scores were fitted under, or None when
        none was used.
    """

    name: str
    public: bool
    fit_signature: str | None

    @dx.validates("name")
    @classmethod
    def _name_is_named(cls, value: str) -> str:
        """Return a nonempty source name."""
        return ensure_name("a score source's name", value)


class CounterfactualFilterSpec(dx.Model, extra="forbid"):
    """Declare a score filter over the counterfactual population, in three arms.

    Parameters
    ----------
    scores : str
        The JSON Lines file giving each candidate sentence's score, one object
        ``{"tokens": [...], "score": number}`` per line, larger meaning more
        acceptable; it scores exactly the candidate sentences.
    source : ScoreSourceSpec
        Where the scores came from. The primary arm requires public scores
        fitted under no signature.
    primary : float
        The primary arm's inclusive minimum score.
    sweep : tuple[float, ...]
        At least five distinct inclusive minima, one sweep arm each, among
        them the primary one.
    """

    scores: str
    source: ScoreSourceSpec
    primary: float
    sweep: tuple[float, ...]

    @dx.validates("scores")
    @classmethod
    def _scores_are_named(cls, value: str) -> str:
        """Return a nonempty score-file path."""
        return ensure_name("the counterfactual filter scores path", value)

    @dx.model_validator()
    def _supports_its_primary_arm(self) -> Self:
        """Refuse a filter whose sweep or source cannot support the primary arm."""
        if not all(isfinite(value) for value in (self.primary, *self.sweep)):
            msg = "a counterfactual filter's thresholds are finite"
            raise ValueError(msg)

        if len(self.sweep) < MINIMUM_SWEEP_THRESHOLDS:
            msg = (
                "a counterfactual filter's sweep declares at least five "
                f"thresholds, not {len(self.sweep)}"
            )
            raise ValueError(msg)

        if len(set(self.sweep)) != len(self.sweep):
            msg = "a counterfactual filter's sweep declares distinct thresholds"
            raise ValueError(msg)

        if self.primary not in self.sweep:
            msg = "a counterfactual filter's sweep includes the primary threshold"
            raise ValueError(msg)

        if not self.source.public:
            msg = "a counterfactual filter's primary arm requires a public source"
            raise ValueError(msg)

        if self.source.fit_signature is not None:
            msg = (
                "a counterfactual filter's primary arm requires a source with no "
                "fit signature"
            )
            raise ValueError(msg)

        return self

    def settings(self) -> CounterfactualFilterSettings:
        """Return the validated settings."""
        return CounterfactualFilterSettings(
            Path(self.scores),
            self.source.name,
            self.source.public,
            self.source.fit_signature,
            self.primary,
            self.sweep,
        )


class CounterfactualSettings(NamedTuple):
    """The validated settings of one declared counterfactual population."""

    sentences: Path
    unary_depth: int
    readings: int
    render: CounterfactualRendering
    pieces: Literal["first", "last", "all"]
    specials: Literal["exclude", "include"]
    contiguous: bool
    seed: int
    filter: CounterfactualFilterSettings | None


class CounterfactualsSpec(dx.Model, extra="forbid"):
    """Declare the counterfactual population generated under the run's grammar.

    Parameters
    ----------
    sentences : str
        The file of candidate sentences, one per line, tokens separated by
        whitespace.
    unary_depth : int
        How many rounds of unary closure a derivation of one string may take.
    readings : int
        How many derivations of one string are read before reading stops; a pair
        needs two.
    render : Literal["declared", "s-expression"]
        How a category is written as the decision at its cell.
    pieces : Literal["first", "last", "all"]
        Which tokenizer pieces of a word stand for the word.
    specials : Literal["exclude", "include"]
        Whether a varied span keeps the rows no word produced.
    contiguous : bool
        Whether a varied span's rows must be contiguous.
    seed : int
        The seed the order of one string's pairs is drawn under.
    filter : CounterfactualFilterSpec | None
        The score filter reported over the population, or None to report the
        population unfiltered only.
    """

    sentences: str
    unary_depth: int
    readings: int
    render: CounterfactualRendering
    pieces: Literal["first", "last", "all"]
    specials: Literal["exclude", "include"]
    contiguous: bool
    seed: int
    filter: CounterfactualFilterSpec | None = None

    @dx.validates("sentences")
    @classmethod
    def _sentences_are_named(cls, value: str) -> str:
        """Return a nonempty candidate-sentence path."""
        return ensure_name("the counterfactual sentences path", value)

    @dx.validates("unary_depth")
    @classmethod
    def _depth_is_non_negative(cls, value: int) -> int:
        """Return a valid number of unary-closure rounds."""
        return ensure_non_negative("the counterfactual unary_depth", value)

    @dx.validates("readings")
    @classmethod
    def _two_readings_are_read(cls, value: int) -> int:
        """Return a reading budget that admits a pair."""
        if value < 2:
            msg = (
                "the counterfactual readings must be at least two, since a pair "
                f"needs two derivations of one string, not {value!r}"
            )
            raise ValueError(msg)

        return value

    @dx.validates("seed")
    @classmethod
    def _seed_is_non_negative(cls, value: int) -> int:
        """Return a valid generator seed."""
        return ensure_non_negative("the counterfactual seed", value)

    def settings(self) -> CounterfactualSettings:
        """Return the validated settings."""
        return CounterfactualSettings(
            Path(self.sentences),
            self.unary_depth,
            self.readings,
            self.render,
            self.pieces,
            self.specials,
            self.contiguous,
            self.seed,
            None if self.filter is None else self.filter.settings(),
        )
