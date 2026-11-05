"""Benchmarks Layers record validation and domain lifting.

Repository-index and path I/O are prepared outside the timed corpus boundary.
"""

import json
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
from lairs import Corpus
from lairs.store.repository import Repository

from lofi.data import (
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    Scalar,
    SpanConstraints,
    Unlabeled,
)
from lofi.data.corpora.layers import LayersCorpusReader
from lofi.data.corpora.layers.spec import records_at

if TYPE_CHECKING:
    from collections.abc import Mapping

    import didactic.api as dx
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi._types import JsonValue
    from lofi.data import Example, Supervision

SAMPLE: Final = Path(__file__).parents[1] / "data" / "layers_sample" / "repository"
RECORDS: Final = 10_000
"""The record population crossed in one benchmark round."""

_CORPUS_COLLECTION: Final = "pub.layers.corpus.corpus"
"""The collection whose selected record owns every scaled membership."""


class MemoryRepository(Repository):
    """An indexed repository of raw record JSON held outside the timer.

    Parameters
    ----------
    records : Mapping[str, str]
        Raw record JSON by AT-URI.
    """

    __slots__ = ("_records", "_uris")

    def __init__(self, records: Mapping[str, str]) -> None:
        self._records = MappingProxyType(dict(records))
        self._uris = tuple(sorted(records))

    @override
    def staged_uris(self) -> list[str]:
        """Return the record identifiers in stable order."""
        return list(self._uris)

    @override
    def load(self, uri: str, model_cls: type[dx.Model]) -> dx.Model | None:
        """Validate one raw record against its generated model.

        Parameters
        ----------
        uri : str
            Record identifier.
        model_cls : type[dx.Model]
            Generated record model.

        Returns
        -------
        dx.Model | None
            Validated record, or None when absent.
        """
        value = self._records.get(uri)
        return None if value is None else model_cls.model_validate_json(value)

    @property
    def raw_values(self) -> tuple[str, ...]:
        """Return the raw record JSON in insertion order."""
        return tuple(self._records.values())


class ScaledRepository(NamedTuple):
    """A reference-consistent record graph and its selected corpus.

    Attributes
    ----------
    repository
        Indexed raw record source.
    corpus_uri
        Corpus record selected by the public reader.
    copies
        Number of complete sample populations in the working tree.
    uris
        Record identifiers written to the repository.
    """

    repository: MemoryRepository
    corpus_uri: str
    copies: int
    uris: tuple[str, ...]


class Crossing(NamedTuple):
    """A reader and the domain examples consumed from it."""

    reader: LayersCorpusReader
    examples_by_split: tuple[tuple[str, tuple[Example[Supervision[str]], ...]], ...]


def _collection(uri: str) -> str:
    """Return the collection segment of an AT-URI.

    Parameters
    ----------
    uri : str
        Record identifier.

    Returns
    -------
    str
        Collection segment.
    """
    return uri.removeprefix("at://").split("/")[1]


def _copy_uri(uri: str, copy: int) -> str:
    """Return a record identifier namespaced to one fixture copy.

    Parameters
    ----------
    uri : str
        Source record identifier.
    copy : int
        Fixture-copy position.

    Returns
    -------
    str
        Unique identifier retaining the source collection.
    """
    prefix, key = uri.rsplit("/", 1)
    return f"{prefix}/{key}-benchmark-{copy:03d}"


def _remap(value: JsonValue, references: Mapping[str, str]) -> JsonValue:
    """Return JSON with every referenced AT-URI replaced.

    Parameters
    ----------
    value : JsonValue
        Source record value.
    references : Mapping[str, str]
        Replacement identifier for each source record.

    Returns
    -------
    JsonValue
        Recursively remapped record value.
    """
    if isinstance(value, str):
        return references.get(value, value)

    if isinstance(value, list):
        return [_remap(entry, references) for entry in value]

    if isinstance(value, dict):
        return {key: _remap(entry, references) for key, entry in value.items()}

    return value


def _references(value: JsonValue) -> frozenset[str]:
    """Return every AT-URI referenced inside JSON.

    Parameters
    ----------
    value : JsonValue
        Parsed record value.

    Returns
    -------
    frozenset[str]
        Referenced record identifiers.
    """
    if isinstance(value, str):
        return frozenset((value,)) if value.startswith("at://") else frozenset()

    if isinstance(value, list):
        return frozenset().union(*(_references(entry) for entry in value))

    if isinstance(value, dict):
        return frozenset().union(*(_references(entry) for entry in value.values()))

    return frozenset()


@pytest.fixture(scope="module")
def scaled_repository() -> ScaledRepository:
    """Build ten thousand valid records from the committed sample.

    Returns
    -------
    ScaledRepository
        The generated repository and selected corpus.
    """
    source = Repository.open(SAMPLE)
    source_uris = tuple(source.staged_uris())
    source_values: dict[str, JsonValue] = {}

    for uri in source_uris:
        value = source.load_raw(uri)

        if value is None:
            pytest.fail(f"the staged sample record {uri} has no value")

        source_values[uri] = value

    if RECORDS % len(source_uris) != 0:
        pytest.fail(
            f"{RECORDS} records cannot hold whole copies of a "
            f"{len(source_uris)}-record sample"
        )

    source_corpora = tuple(
        uri for uri in source_uris if _collection(uri) == _CORPUS_COLLECTION
    )

    if len(source_corpora) != 1:
        pytest.fail(f"the sample has {len(source_corpora)} corpus records")

    copies = RECORDS // len(source_uris)
    corpus_uri = _copy_uri(source_corpora[0], 0)
    records: dict[str, str] = {}
    written: list[str] = []

    for copy in range(copies):
        references = {uri: _copy_uri(uri, copy) for uri in source_uris}
        references[source_corpora[0]] = corpus_uri

        for source_uri in source_uris:
            uri = _copy_uri(source_uri, copy)
            records[uri] = json.dumps(
                _remap(source_values[source_uri], references),
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
            )
            written.append(uri)

    return ScaledRepository(
        MemoryRepository(records),
        corpus_uri,
        copies,
        tuple(written),
    )


def _cross_boundary(scaled: ScaledRepository) -> Crossing:
    """Validate raw records and lift their resolved graph to domain values.

    Parameters
    ----------
    scaled : ScaledRepository
        Reference-consistent raw record graph.

    Returns
    -------
    Crossing
        Reader beside every domain example consumed from it.
    """
    validated = records_at(scaled.repository)
    corpus = Corpus(uri=scaled.corpus_uri)

    for uri, record in validated.records:
        corpus.add_record(uri, record)

    reader = LayersCorpusReader(corpus)
    examples = tuple(
        (split, tuple(reader.examples(split))) for split in reader.splits()
    )
    return Crossing(reader, examples)


@pytest.fixture(scope="module")
def loaded(scaled_repository: ScaledRepository) -> Crossing:
    """Read the scaled repository through the public boundary.

    Parameters
    ----------
    scaled_repository : ScaledRepository
        Reference-consistent repository fixture.

    Returns
    -------
    Crossing
        Validated reader and materialized domain examples.
    """
    return _cross_boundary(scaled_repository)


def test_the_scaled_repository_validates_every_record(
    scaled_repository: ScaledRepository,
) -> None:
    validated = records_at(scaled_repository.repository)

    assert len(scaled_repository.uris) == RECORDS
    assert len(set(scaled_repository.uris)) == RECORDS
    assert tuple(scaled_repository.repository.staged_uris()) == tuple(
        sorted(scaled_repository.uris)
    )
    assert len(validated.records) == RECORDS
    assert validated.skips == ()
    available = frozenset(scaled_repository.uris)
    referenced = frozenset().union(
        *(
            _references(json.loads(value))
            for value in scaled_repository.repository.raw_values
        )
    )
    assert referenced <= available


def test_the_public_reader_resolves_every_scaled_copy(
    scaled_repository: ScaledRepository,
    loaded: Crossing,
) -> None:
    reader = loaded.reader
    by_split = dict(loaded.examples_by_split)

    assert reader.splits() == ("dev", "test", "train")
    assert {split: len(examples) for split, examples in by_split.items()} == {
        "dev": scaled_repository.copies,
        "test": 2 * scaled_repository.copies,
        "train": 5 * scaled_repository.copies,
    }
    assert sum(map(len, by_split.values())) == 8 * scaled_repository.copies
    assert reader.report().emitted == 8 * scaled_repository.copies
    assert reader.report().skips_by_reason == {}
    assert reader.supervision_kinds() == frozenset(
        {
            GoldDependencies,
            GoldDerivation,
            GoldGraph,
            Scalar,
            SpanConstraints,
            Unlabeled,
        }
    )


def test_a_corpus_record_crosses_the_boundary_within_budget(
    benchmark: BenchmarkFixture,
    scaled_repository: ScaledRepository,
    wall_clock_budget: WallClockBudget,
) -> None:
    benchmark.pedantic(
        _cross_boundary,
        args=(scaled_repository,),
        rounds=5,
        warmup_rounds=1,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the reader ran but pytest-benchmark timed nothing")

    per_record_us = measured.stats.median / len(scaled_repository.uris) * 1e6
    benchmark.extra_info["records"] = len(scaled_repository.uris)
    benchmark.extra_info["median_us_per_record"] = per_record_us
    benchmark.extra_info["mean_us_per_record"] = (
        measured.stats.mean / len(scaled_repository.uris) * 1e6
    )
    benchmark.extra_info["source"] = "in-memory raw JSON; path index I/O excluded"
    benchmark.extra_info["budget"] = "corpus-record"

    if wall_clock_budget.enforced:
        assert per_record_us <= 200
