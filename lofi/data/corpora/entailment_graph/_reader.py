"""Registered readers and immutable values for local entailment graphs."""

from pathlib import Path
from typing import TYPE_CHECKING, Protocol, cast, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.registry import Registry, RegistryError

from .spec import EdgeListEntailmentGraphSpec, EntailmentGraphSpec

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "EDGE_LIST_PLUGIN",
    "EntailmentGraph",
    "EntailmentGraphReaderPlugin",
    "MalformedEntailmentGraphError",
    "read_entailment_graph",
]


class MalformedEntailmentGraphError(ValueError):
    """Report a local entailment graph that is not the declared release format."""


@final
class EntailmentGraph(FrozenValue):
    """A finite directed graph over named nodes."""

    _noun = "entailment graph"
    __slots__ = ("_edges", "_language", "_nodes")

    def __init__(self, language: str, edges: Iterable[tuple[str, str]], /) -> None:
        held = frozenset(
            (
                ensure_name("the node a link starts from", antecedent),
                ensure_name("the node a link ends at", consequent),
            )
            for antecedent, consequent in edges
        )
        if not held:
            msg = "an entailment graph has no edges"
            raise MalformedEntailmentGraphError(msg)
        if any(left == right for left, right in held):
            msg = "an entailment graph contains a reflexive edge"
            raise MalformedEntailmentGraphError(msg)
        nodes = tuple(sorted({node for edge in held for node in edge}))
        object.__setattr__(self, "_language", ensure_name("a graph language", language))
        object.__setattr__(self, "_nodes", nodes)
        object.__setattr__(self, "_edges", held)

    @property
    def language(self) -> str:
        """Return the graph's BCP 47 language tag."""
        return self._language

    @property
    def nodes(self) -> tuple[str, ...]:
        """Return nodes in lexical order."""
        return self._nodes

    @property
    def edges(self) -> frozenset[tuple[str, str]]:
        """Return directed node links."""
        return self._edges

    def links(self, antecedent: str, consequent: str, /) -> bool:
        """Return whether the graph contains one directed link."""
        return (antecedent, consequent) in self._edges

    @override
    def __eq__(self, other: object) -> bool:
        return (
            type(other) is EntailmentGraph
            and self._language == other._language
            and self._edges == other._edges
        )

    @override
    def __hash__(self) -> int:
        return hash((EntailmentGraph, self._language, self._edges))


class EntailmentGraphReader[S: EntailmentGraphSpec](Protocol):
    """Read one validated local graph declaration."""

    def __call__(self, declaration: S, /) -> EntailmentGraph:
        """Return the declared graph."""
        ...


@final
class EntailmentGraphReaderPlugin[S: EntailmentGraphSpec](FrozenValue):
    """Pair a graph declaration variant with its one reader."""

    _noun = "entailment-graph reader plugin"
    __slots__ = ("_read", "_spec")

    def __init__(
        self,
        *,
        spec: type[S],
        read: EntailmentGraphReader[S],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_read", read)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin reads."""
        return self._spec

    def read(self, declaration: EntailmentGraphSpec, /) -> EntailmentGraph:
        """Read a declaration of this plugin's exact variant."""
        if type(declaration) is not self._spec:
            msg = (
                f"the entailment-graph reader handles {self._spec.__name__}, not "
                f"{type(declaration).__name__}"
            )
            raise RegistryError(msg, family="entailment_graph_reader")
        return self._read(cast("S", declaration))


def _edge_list(declaration: EdgeListEntailmentGraphSpec, /) -> EntailmentGraph:
    """Read a directed edge from each nonempty line of a local UTF-8 file."""
    source = Path(declaration.source)
    try:
        text = source.read_text(encoding="utf-8")
    except (OSError, UnicodeError) as error:
        msg = f"cannot read entailment graph {source}: {error}"
        raise MalformedEntailmentGraphError(msg) from error

    edges: list[tuple[str, str]] = []
    for number, line in enumerate(text.splitlines(), start=1):
        if not line:
            continue
        fields = line.split(declaration.delimiter)
        if len(fields) != 2 or any(not field for field in fields):
            msg = f"entailment graph line {number} is not one directed edge"
            raise MalformedEntailmentGraphError(msg)
        edges.append((fields[0], fields[1]))
    return EntailmentGraph(declaration.language, edges)


EDGE_LIST_PLUGIN = EntailmentGraphReaderPlugin(
    spec=EdgeListEntailmentGraphSpec,
    read=_edge_list,
)
"""The current reader for local directed edge-list releases."""


def read_entailment_graph(declaration: EntailmentGraphSpec, /) -> EntailmentGraph:
    """Read a graph through the reader installed for its declared kind."""
    tag = cast("str", declaration.kind)
    member = Registry[object]("entailment_graph_reader").get(tag)
    if not isinstance(member, EntailmentGraphReaderPlugin):
        msg = (
            f"the entailment-graph reader entry {declaration.kind!r} "
            "does not name a reader plugin"
        )
        raise RegistryError(msg, family="entailment_graph_reader")
    plugin = cast("EntailmentGraphReaderPlugin[EntailmentGraphSpec]", member)
    return plugin.read(declaration)
