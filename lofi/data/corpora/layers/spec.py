"""Layers repository validation."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple

import didactic.api as dx
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import expression as expression_records
from lairs.records import graph as graph_records
from lairs.records import judgment as judgment_records
from lairs.records import segmentation as segmentation_records

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lairs.store.repository import Repository

__all__ = ["RepositoryRecords", "records_at"]


_MODELS: Final[Mapping[str, type[dx.Model]]] = MappingProxyType(
    {
        "pub.layers.corpus.corpus": corpus_records.Corpus,
        "pub.layers.corpus.membership": corpus_records.Membership,
        "pub.layers.expression.expression": expression_records.Expression,
        "pub.layers.segmentation.segmentation": segmentation_records.Segmentation,
        "pub.layers.annotation.annotationLayer": annotation_records.AnnotationLayer,
        "pub.layers.graph.graphNode": graph_records.GraphNode,
        "pub.layers.graph.graphEdge": graph_records.GraphEdge,
        "pub.layers.judgment.experimentDef": judgment_records.ExperimentDef,
        "pub.layers.judgment.judgmentSet": judgment_records.JudgmentSet,
    }
)
"""Known Layers collections and their record models."""


class RepositoryRecords(NamedTuple):
    """Validated records and failures from a Layers repository."""

    records: tuple[tuple[str, dx.Model], ...]
    corpus_uris: tuple[str, ...]
    skips: tuple[str, ...]


def _collection_of(uri: str) -> str:
    """Return the collection named by an AT-URI.

    Parameters
    ----------
    uri : str
        The AT-URI to inspect.

    Returns
    -------
    str
        The collection segment, or an empty string if it is absent.
    """
    segments = uri.removeprefix("at://").split("/")

    return segments[1] if len(segments) > 1 else ""


def records_at(repository: Repository, /) -> RepositoryRecords:
    """Validate known records in a Layers repository."""
    records: list[tuple[str, dx.Model]] = []
    corpus_uris: list[str] = []
    skips: list[str] = []

    for uri in sorted(repository.staged_uris()):
        model = _MODELS.get(_collection_of(uri))

        if model is None:
            continue

        try:
            record = repository.load(uri, model)
        except dx.ValidationError:
            skips.append("a Layers record that fails schema validation")

            continue

        if record is None:
            skips.append("a staged Layers record whose value is absent")

            continue

        records.append((uri, record))

        if isinstance(record, corpus_records.Corpus):
            corpus_uris.append(uri)

    return RepositoryRecords(tuple(records), tuple(corpus_uris), tuple(skips))
