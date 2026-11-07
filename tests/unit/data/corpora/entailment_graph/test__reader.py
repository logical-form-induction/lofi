"""Tests for declared local entailment-graph releases."""

from typing import TYPE_CHECKING

import pytest

from lofi.data.corpora.entailment_graph import (
    EDGE_LIST_PLUGIN,
    EdgeListEntailmentGraphSpec,
    MalformedEntailmentGraphError,
)

if TYPE_CHECKING:
    from pathlib import Path


def test_the_edge_list_reader_preserves_nodes_and_directed_edges(
    tmp_path: Path,
) -> None:
    source = tmp_path / "graph.tsv"
    source.write_text(
        "admire\tlike\ncall\tcontact\nlike\tappreciate\n",
        encoding="utf-8",
        newline="\n",
    )

    graph = EDGE_LIST_PLUGIN.read(
        EdgeListEntailmentGraphSpec(source=str(source), language="en")
    )

    assert graph.language == "en"
    assert graph.nodes == ("admire", "appreciate", "call", "contact", "like")
    assert graph.edges == frozenset(
        {("admire", "like"), ("call", "contact"), ("like", "appreciate")}
    )
    assert graph.links("admire", "like")
    assert not graph.links("like", "admire")


def test_the_reader_refuses_an_empty_or_malformed_local_release(tmp_path: Path) -> None:
    source = tmp_path / "graph.tsv"
    declaration = EdgeListEntailmentGraphSpec(source=str(source), language="en")

    source.write_text("", encoding="utf-8", newline="\n")
    with pytest.raises(MalformedEntailmentGraphError, match="no edges"):
        EDGE_LIST_PLUGIN.read(declaration)

    source.write_text("one\ttwo\tthree\n", encoding="utf-8", newline="\n")
    with pytest.raises(MalformedEntailmentGraphError, match="line 1"):
        EDGE_LIST_PLUGIN.read(declaration)
