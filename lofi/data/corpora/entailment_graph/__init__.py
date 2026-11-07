"""Local entailment-graph declarations, readers and the lexicon generator."""

from lofi.data.corpora.entailment_graph._generator import (
    ENTAILMENT_GRAPH_PLUGIN,
    EntailmentGraphLexiconGenerator,
)
from lofi.data.corpora.entailment_graph._reader import (
    EDGE_LIST_PLUGIN,
    EntailmentGraph,
    EntailmentGraphReaderPlugin,
    MalformedEntailmentGraphError,
    read_entailment_graph,
)
from lofi.data.corpora.entailment_graph.spec import (
    EdgeListEntailmentGraphSpec,
    EntailmentGraphGeneratorSettings,
    EntailmentGraphGeneratorSpec,
    EntailmentGraphSpec,
)

__all__ = [
    "EDGE_LIST_PLUGIN",
    "ENTAILMENT_GRAPH_PLUGIN",
    "EdgeListEntailmentGraphSpec",
    "EntailmentGraph",
    "EntailmentGraphGeneratorSettings",
    "EntailmentGraphGeneratorSpec",
    "EntailmentGraphLexiconGenerator",
    "EntailmentGraphReaderPlugin",
    "EntailmentGraphSpec",
    "MalformedEntailmentGraphError",
    "read_entailment_graph",
]
