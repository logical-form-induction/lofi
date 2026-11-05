"""Declarations for local entailment-graph releases."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name, ensure_non_negative
from lofi.spec.lexicon_generators import LexiconGeneratorSpec

__all__ = [
    "EdgeListEntailmentGraphSpec",
    "EntailmentGraphGeneratorSettings",
    "EntailmentGraphGeneratorSpec",
    "EntailmentGraphSpec",
]


class EntailmentGraphSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare a local entailment-graph release."""


class EdgeListEntailmentGraphSpec(EntailmentGraphSpec, extra="forbid"):
    """Declare a directed two-column UTF-8 edge list."""

    kind: Literal["edge-list"] = "edge-list"
    source: str
    language: str
    delimiter: str = "\t"

    @dx.validates("source", "language", "delimiter")
    @classmethod
    def _is_present(cls, value: str) -> str:
        """Return a nonempty source field."""
        return ensure_name("an entailment-graph field", value)


class EntailmentGraphGeneratorSettings(NamedTuple):
    """Resolved proposal of a graph's nodes as one bead lexicon."""

    kind: Literal["entailment-graph"]
    graph: EntailmentGraphSpec
    lexicon: str
    feature: str
    linked: str
    sampled: str
    negative_ratio: int
    seed: int | None


class EntailmentGraphGeneratorSpec(LexiconGeneratorSpec):
    """Propose a local entailment graph's nodes as one bead lexicon.

    Each node is one lexical item in the graph's language, and its features
    record, under ``feature``, every node it is paired with and the pair's
    declared label: ``linked`` for each node a graph link reaches from it, and
    ``sampled`` for each node of a seeded draw of ordered pairs outside the
    graph, ``negative_ratio`` of them per link. An item construction reads the
    pairs and labels through its attribute and admission expressions.

    Parameters
    ----------
    graph : EntailmentGraphSpec
        The local graph release.
    lexicon : str
        Name of the lexicon the nodes form.
    feature : str
        The feature each node records its paired nodes and their labels under.
    linked : str
        The label of a pair a graph link backs.
    sampled : str
        The label of a pair drawn outside the graph.
    negative_ratio : int
        Pairs drawn outside the graph per graph link.
    seed : int | None
        Seed of the outside-pair draw, or None for one derived from the
        construction's namespace.
    """

    kind: Literal["entailment-graph"] = "entailment-graph"
    graph: EntailmentGraphSpec
    lexicon: str
    feature: str
    linked: str
    sampled: str
    negative_ratio: int
    seed: int | None

    @dx.validates("lexicon", "feature", "linked", "sampled")
    @classmethod
    def _is_named(cls, value: str) -> str:
        """Return a nonempty name."""
        return ensure_name("an entailment-graph generator field", value)

    @dx.validates("negative_ratio")
    @classmethod
    def _ratio_is_nonnegative(cls, value: int) -> int:
        """Return a nonnegative negative-per-link ratio."""
        return ensure_non_negative("an entailment-graph negative ratio", value)

    @dx.validates("seed")
    @classmethod
    def _seed_is_nonnegative(cls, value: int | None) -> int | None:
        """Return a nonnegative sampling seed, or None."""
        if value is None:
            return None
        return ensure_non_negative("an entailment-graph seed", value)

    @dx.model_validator()
    def _labels_differ(self) -> EntailmentGraphGeneratorSpec:
        """Validate that a linked pair and a drawn pair are labeled apart."""
        if self.linked == self.sampled:
            msg = f"linked and sampled pairs are both labeled {self.linked!r}"
            raise ValueError(msg)
        return self

    def settings(self) -> EntailmentGraphGeneratorSettings:
        """Discard the boundary declaration into immutable settings."""
        return EntailmentGraphGeneratorSettings(
            self.kind,
            self.graph,
            self.lexicon,
            self.feature,
            self.linked,
            self.sampled,
            self.negative_ratio,
            self.seed,
        )
