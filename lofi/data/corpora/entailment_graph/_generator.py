"""Propose a local entailment graph's nodes as one bead lexicon."""

from random import Random
from typing import TYPE_CHECKING, Final, final
from uuid import uuid5

from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon

from lofi._frozen import FrozenValue
from lofi.data.corpora.entailment_graph._reader import read_entailment_graph
from lofi.data.corpora.entailment_graph.spec import EntailmentGraphGeneratorSpec
from lofi.registry import ComponentPlugin

if TYPE_CHECKING:
    from uuid import UUID

    from lofi.data.corpora.entailment_graph._reader import EntailmentGraph
    from lofi.data.corpora.entailment_graph.spec import (
        EntailmentGraphGeneratorSettings,
    )

__all__ = ["ENTAILMENT_GRAPH_PLUGIN", "EntailmentGraphLexiconGenerator"]

_SEED_BOUND: Final = 2**32
"""One past the largest seed a derived draw seed takes."""


def _outside_pairs(
    graph: EntailmentGraph, count: int, seed: int, /
) -> tuple[tuple[str, str], ...]:
    """Draw distinct ordered node pairs absent from the graph."""
    candidates = [
        (left, right)
        for left in graph.nodes
        for right in graph.nodes
        if left != right and not graph.links(left, right)
    ]
    if count > len(candidates):
        msg = (
            f"the graph has {len(candidates)} outside pairs, fewer than the "
            f"declared negative count {count}"
        )
        raise ValueError(msg)
    Random(seed).shuffle(candidates)
    return tuple(candidates[:count])


@final
class EntailmentGraphLexiconGenerator(FrozenValue):
    """Propose a graph's nodes, each recording its labeled pairs.

    Parameters
    ----------
    settings : EntailmentGraphGeneratorSettings
        The resolved declaration.
    graph : EntailmentGraph
        The declared graph.
    """

    _noun = "entailment-graph lexicon generator"
    __slots__ = ("_graph", "_settings")

    def __init__(
        self, settings: EntailmentGraphGeneratorSettings, graph: EntailmentGraph, /
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_graph", graph)

    def generate(self, namespace: UUID, /) -> Lexicon:
        """Return the graph's nodes in lexical order, identities in ``namespace``."""
        settings = self._settings
        graph = self._graph
        links = tuple(sorted(graph.edges))
        seed = (
            settings.seed
            if settings.seed is not None
            else uuid5(namespace, f"entailment-graph\0{settings.lexicon}").int
            % _SEED_BOUND
        )
        drawn = _outside_pairs(graph, len(links) * settings.negative_ratio, seed)
        paired: dict[str, dict[str, str]] = {node: {} for node in graph.nodes}
        for left, right in links:
            paired[left][right] = settings.linked
        for left, right in drawn:
            paired[left][right] = settings.sampled
        source = f"entailment-graph:{graph.language}"
        return Lexicon(
            name=settings.lexicon,
            language_code=graph.language,
            items=tuple(
                LexicalItem(
                    id=uuid5(namespace, f"lexical-item\0{source}\0{node}"),
                    lemma=node,
                    language_code=graph.language,
                    features={settings.feature: dict(sorted(paired[node].items()))},
                    source=source,
                )
                for node in graph.nodes
            ),
        )


def _build(
    settings: EntailmentGraphGeneratorSettings, context: None, /
) -> EntailmentGraphLexiconGenerator:
    """Read the declared graph and build its lexicon generator."""
    del context
    return EntailmentGraphLexiconGenerator(
        settings, read_entailment_graph(settings.graph)
    )


ENTAILMENT_GRAPH_PLUGIN: Final = ComponentPlugin(
    spec=EntailmentGraphGeneratorSpec, build=_build
)
"""The generator proposing a local entailment graph's nodes as a lexicon."""
