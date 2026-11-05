import json
import tomllib
from importlib.resources import files
from pathlib import Path
from typing import TYPE_CHECKING

import pytest
from lairs.records import annotation, corpus, expression, graph, judgment
from lairs.records import segmentation as segmentation_records
from lairs.store.repository import Repository

if TYPE_CHECKING:
    import didactic.api as dx

SAMPLE = Path(__file__).parent
REPOSITORY = SAMPLE / "repository"
MANIFEST = SAMPLE / "MANIFEST.json"

RECORD_MODELS: dict[str, type[dx.Model]] = {
    "pub.layers.annotation.annotationLayer": annotation.AnnotationLayer,
    "pub.layers.corpus.corpus": corpus.Corpus,
    "pub.layers.corpus.membership": corpus.Membership,
    "pub.layers.expression.expression": expression.Expression,
    "pub.layers.graph.graphEdge": graph.GraphEdge,
    "pub.layers.graph.graphNode": graph.GraphNode,
    "pub.layers.judgment.experimentDef": judgment.ExperimentDef,
    "pub.layers.judgment.judgmentSet": judgment.JudgmentSet,
    "pub.layers.segmentation.segmentation": segmentation_records.Segmentation,
}


def collection_of(uri: str) -> str:
    return uri.removeprefix("at://").split("/")[1]


@pytest.fixture(scope="module")
def repository() -> Repository:
    return Repository.open(REPOSITORY)


@pytest.fixture(scope="module")
def loaded(repository: Repository) -> dict[str, dx.Model]:
    """Return every record of the sample, validated against its Layers model."""
    records: dict[str, dx.Model] = {}

    for uri in repository.staged_uris():
        model_class = RECORD_MODELS[collection_of(uri)]
        record = repository.load(uri, model_class)

        assert record is not None
        records[uri] = record

    return records


def of_type[M: dx.Model](loaded: dict[str, dx.Model], wanted: type[M]) -> list[M]:
    return [record for record in loaded.values() if isinstance(record, wanted)]


def test_every_staged_record_is_a_layers_record_type(repository: Repository) -> None:
    unknown = [
        uri
        for uri in repository.staged_uris()
        if collection_of(uri) not in RECORD_MODELS
    ]

    assert unknown == []


def test_every_record_validates_against_its_layers_model_and_none_is_skipped(
    repository: Repository,
    loaded: dict[str, dx.Model],
) -> None:
    staged = repository.staged_uris()

    assert staged
    assert len(loaded) == len(staged)


def test_the_pinned_lexicon_tree_hash_is_the_one_lairs_vendors() -> None:
    pinned = json.loads(MANIFEST.read_text(encoding="utf-8"))["lexicon_tree_hash"]
    vendored = tomllib.loads(
        files("lairs.lexicons").joinpath("MANIFEST.toml").read_bytes().decode()
    )["provenance"]["lexicon_tree_hash"]

    assert pinned == vendored, (
        "the sample was written against a different Layers lexicon tree; "
        "rebuild it with tests/data/layers_sample/build_sample.py"
    )


def test_the_corpus_spreads_its_expressions_over_three_splits(
    loaded: dict[str, dx.Model],
) -> None:
    memberships = of_type(loaded, corpus.Membership)
    corpus_uri = next(
        uri for uri in loaded if collection_of(uri) == "pub.layers.corpus.corpus"
    )

    assert len(memberships) == 4
    assert {membership.split for membership in memberships} == {"train", "dev", "test"}
    assert {membership.corpusRef for membership in memberships} == {corpus_uri}


def test_every_membership_points_at_an_expression_the_sample_holds(
    loaded: dict[str, dx.Model],
) -> None:
    for membership in of_type(loaded, corpus.Membership):
        assert membership.expressionRef in loaded


def test_the_annotation_layers_cover_a_tree_a_relation_and_a_token_tagging(
    loaded: dict[str, dx.Model],
) -> None:
    layers = of_type(loaded, annotation.AnnotationLayer)
    kinds = {(layer.kind, layer.subkind) for layer in layers}

    assert kinds == {("tree", "ccg"), ("relation", "dependency"), ("token-tag", "ccg")}

    for layer in layers:
        assert layer.expression in loaded
        assert layer.annotations


def test_the_tree_layer_is_a_tree_whose_leaves_cover_every_token(
    loaded: dict[str, dx.Model],
) -> None:
    tree = next(
        layer
        for layer in of_type(loaded, annotation.AnnotationLayer)
        if layer.kind == "tree"
    )
    nodes = {node.uuid.value: node for node in tree.annotations}
    roots = [node for node in tree.annotations if node.parentId is None]
    leaves = [node for node in tree.annotations if not node.childIds]

    assert len(roots) == 1

    for node in tree.annotations:
        for child in node.childIds or ():
            parent = nodes[child.value].parentId

            assert parent is not None
            assert parent.value == node.uuid.value

    positions = sorted(
        leaf.tokenIndex for leaf in leaves if leaf.tokenIndex is not None
    )

    assert positions == [0, 1, 2]
    assert len(positions) == len(leaves)


def test_the_dependency_layer_relates_token_positions(
    loaded: dict[str, dx.Model],
) -> None:
    relations = next(
        layer
        for layer in of_type(loaded, annotation.AnnotationLayer)
        if layer.kind == "relation"
    )

    for arc in relations.annotations:
        assert arc.headIndex is not None
        assert arc.targetIndex is not None
        assert arc.label


def test_every_token_span_recovers_the_word_it_covers(
    loaded: dict[str, dx.Model],
) -> None:
    checked = 0

    for segmentation in of_type(loaded, segmentation_records.Segmentation):
        sentence = loaded[segmentation.expression]

        assert isinstance(sentence, expression.Expression)
        assert sentence.text is not None

        encoded = sentence.text.encode()

        for tokenization in segmentation.tokenizations:
            for token in tokenization.tokens:
                assert token.textSpan is not None
                covered = encoded[token.textSpan.byteStart : token.textSpan.byteEnd]

                assert covered.decode() == token.text
                checked += 1

    assert checked > 0


def test_the_sample_carries_words_whose_characters_are_more_than_one_byte(
    loaded: dict[str, dx.Model],
) -> None:
    wide = [
        token.text
        for segmentation in of_type(loaded, segmentation_records.Segmentation)
        for tokenization in segmentation.tokenizations
        for token in tokenization.tokens
        if token.text is not None and len(token.text.encode()) > len(token.text)
    ]

    assert len(wide) >= 2


def test_one_expression_is_left_with_no_segmentation(
    loaded: dict[str, dx.Model],
) -> None:
    sentences = {
        uri
        for uri in loaded
        if collection_of(uri) == "pub.layers.expression.expression"
    }
    segmented = {
        segmentation.expression
        for segmentation in of_type(loaded, segmentation_records.Segmentation)
    }

    assert len(sentences - segmented) == 1


def test_a_graph_edge_joins_a_sentence_to_a_graph_node(
    loaded: dict[str, dx.Model],
) -> None:
    edges = of_type(loaded, graph.GraphEdge)
    nodes = of_type(loaded, graph.GraphNode)

    assert len(nodes) == 1
    assert len(edges) == 1

    edge = edges[0]

    assert edge.source.recordRef in loaded
    assert edge.target.recordRef in loaded
    assert collection_of(str(edge.target.recordRef)) == "pub.layers.graph.graphNode"


def test_two_annotators_rated_the_same_item(loaded: dict[str, dx.Model]) -> None:
    sets = of_type(loaded, judgment.JudgmentSet)
    experiment = of_type(loaded, judgment.ExperimentDef)[0]
    rated = {
        response.item.recordRef for judged in sets for response in judged.judgments
    }
    annotators = {judged.agent.id for judged in sets if judged.agent is not None}
    ratings = [response.scalarValue for judged in sets for response in judged.judgments]

    assert len(sets) == 2
    assert len(rated) == 1
    assert len(annotators) == 2
    assert len(ratings) == 2
    assert ratings[0] != ratings[1]

    for judged in sets:
        assert judged.experimentRef in loaded

    for rating in ratings:
        assert rating is not None
        assert experiment.scaleMin is not None
        assert experiment.scaleMax is not None
        assert experiment.scaleMin <= rating <= experiment.scaleMax


def test_the_rated_sentence_repeats_one_surface_form(
    loaded: dict[str, dx.Model],
) -> None:
    experiment = of_type(loaded, judgment.ExperimentDef)[0]
    stimuli = experiment.stimulusExpressionRefs

    assert stimuli is not None

    stimulus = loaded[stimuli[0]]

    assert isinstance(stimulus, expression.Expression)
    assert stimulus.text is not None

    words = stimulus.text.split()

    assert len(words) > len(set(words))
