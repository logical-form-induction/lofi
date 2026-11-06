import json
import os
import subprocess
import sys
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, Never, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from lairs import Corpus
from lairs.data import JudgmentStudy
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import defs as defs_records
from lairs.records import expression as expression_records
from lairs.records import graph as graph_records
from lairs.records import judgment as judgment_records
from lairs.records import ontology as ontology_records
from lairs.records import segmentation as segmentation_records
from lairs.store.repository import Repository
from tests.conformance.corpus_reader import CorpusReaderConformance, CorpusReaderSubject

from lofi.data import (
    RESPONSE_MISSING,
    AnalysisCounts,
    Annotation,
    AnnotatorId,
    Categorical,
    CorpusReader,
    EvalDataset,
    Example,
    GoldDependencies,
    GoldDerivation,
    GoldGraph,
    Inventory,
    ItemHasher,
    ItemId,
    LabelBlocker,
    ReachableDerivation,
    Relational,
    RelationalScalar,
    RuleBlocker,
    Scalar,
    SpanConstraints,
    Supervision,
    Unlabeled,
)
from lofi.data.corpora import SignatureLabels
from lofi.data.corpora.layers import (
    READER_NAME,
    ComparisonInventory,
    JudgmentStudyOptions,
    LayersCorpusReader,
    LayersSkip,
    corpus_at,
    judgment_study_at,
    reader_at,
)
from lofi.data.corpora.layers import reader as reader_module
from lofi.data.corpora.layers._translate import lift_tree, lower_tree
from lofi.data.corpora.layers.reader import _models_of
from lofi.data.laws import ITEM_LAWS, ItemFixture, ReaderFixture, check_item_laws
from lofi.judge import NoNormalForm, NormalFormRole, RuleSet
from lofi.terms import Operator, Signature

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence

    import didactic.api as dx
    from lairs.atproto.pds import PdsClient

    from lofi._types import JsonValue
    from lofi.data import (
        Derivation,
        DerivationOutcome,
        LabelMembership,
        ReachabilityReport,
        StepRecovery,
        Token,
    )
    from lofi.judge import Direction, RuleProbe, RuleRestriction, Slot

SAMPLE: Final = (
    Path(__file__).resolve().parents[4] / "data" / "layers_sample" / "repository"
)

AUTHORITY: Final = "did:plc:lofisample"

CREATED: Final = datetime(2026, 1, 1, tzinfo=UTC)

EXPRESSION_NSID: Final = "pub.layers.expression.expression"
MEMBERSHIP_NSID: Final = "pub.layers.corpus.membership"
LAYER_NSID: Final = "pub.layers.annotation.annotationLayer"
EXPERIMENT_NSID: Final = "pub.layers.judgment.experimentDef"
JUDGMENT_SET_NSID: Final = "pub.layers.judgment.judgmentSet"
CORPUS_NSID: Final = "pub.layers.corpus.corpus"

SAMPLE_CORPUS: Final = f"at://{AUTHORITY}/{CORPUS_NSID}/sample"

SLEEPS: Final = f"at://{AUTHORITY}/{EXPRESSION_NSID}/en-sleeps"
REPEATED: Final = f"at://{AUTHORITY}/{EXPRESSION_NSID}/en-repeated"
MANDARIN: Final = f"at://{AUTHORITY}/{EXPRESSION_NSID}/zh-shuijiao"

TREE_LAYER: Final = f"at://{AUTHORITY}/{LAYER_NSID}/en-sleeps-tree"
TAG_LAYER: Final = f"at://{AUTHORITY}/{LAYER_NSID}/en-sleeps-token-tags"
DEPENDENCY_LAYER: Final = f"at://{AUTHORITY}/{LAYER_NSID}/en-sleeps-dependency"

GRAPH_NODE_NSID: Final = "pub.layers.graph.graphNode"
GRAPH_EDGE_NSID: Final = "pub.layers.graph.graphEdge"

GRAPH_NODE: Final = f"at://{AUTHORITY}/{GRAPH_NODE_NSID}/sleep-situation"
GRAPH_EDGE: Final = f"at://{AUTHORITY}/{GRAPH_EDGE_NSID}/en-sleeps-denotes"

ORDINAL_LEVELS: Final = 7

IDENTITY_EXAMPLES: Final = 25

_SAMPLE_IDS = """
import json
import os
import sys
from pathlib import Path

from lofi.data.corpora.layers import reader_at

reader = reader_at(Path(os.environ["LOFI_IDENTITY_SAMPLE"]))
payload = {
    split: [str(item.item_id) for item in reader.items(split)]
    for split in reader.splits()
}
sys.stdout.buffer.write(
    json.dumps(
        payload,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")
)
"""


class LayerProvenance(NamedTuple):
    """Provenance fields for a generated annotation layer."""

    source_method: str | None
    licensing: defs_records.Licensing | None
    eprint_refs: tuple[str, ...] = ()
    reproducibility: defs_records.ReproducibilityInfo | None = None


def uri(collection: str, rkey: str) -> str:
    return f"at://did:plc:test/{collection}/{rkey}"


def load_sample() -> LayersCorpusReader:
    return LayersCorpusReader(corpus_at(SAMPLE))


def sample_ids_under(seed: str) -> bytes:
    environment = {
        **os.environ,
        "LOFI_IDENTITY_SAMPLE": str(SAMPLE),
        "PYTHONHASHSEED": seed,
        "PYTHONIOENCODING": "utf-8",
    }
    finished = subprocess.run(
        [sys.executable, "-"],
        input=_SAMPLE_IDS.encode("utf-8"),
        capture_output=True,
        check=True,
        cwd=Path(__file__).resolve().parents[5],
        env=environment,
        timeout=120.0,
    )

    return finished.stdout


@pytest.fixture(scope="module")
def sample() -> LayersCorpusReader:
    return load_sample()


def built(records: dict[str, dx.Model], *, named: str | None = None) -> Corpus:
    corpus = Corpus(uri=named)

    for at_uri, record in records.items():
        corpus.add_record(at_uri, record)

    return corpus


def built_study(
    records: dict[str, dx.Model], *, named: str | None = None
) -> JudgmentStudy:
    study = JudgmentStudy.new(named)

    for at_uri, record in records.items():
        study.add_record(at_uri, record)

    return study


def expression_of(
    rkey: str,
    text: str | None,
    languages: tuple[str, ...] = ("en",),
) -> expression_records.Expression:
    return expression_records.Expression(
        id=rkey,
        kind="sentence",
        createdAt=CREATED,
        text=text,
        languages=languages,
    )


def membership_of(
    rkey: str,
    split: str | None,
    ordinal: int | None = 0,
) -> corpus_records.Membership:
    return corpus_records.Membership(
        corpusRef=uri(CORPUS_NSID, "one"),
        expressionRef=uri(EXPRESSION_NSID, rkey),
        createdAt=CREATED,
        split=split,
        ordinal=ordinal,
    )


def segmentation_of(
    rkey: str,
    words: tuple[str, ...],
    *,
    spans: bool = False,
    start_at: int = 0,
) -> segmentation_records.Segmentation:
    tokens = tuple(
        segmentation_records.Token(
            tokenIndex=start_at + position,
            text=word,
            textSpan=defs_records.Span(byteStart=0, byteEnd=len(word.encode()))
            if spans
            else None,
        )
        for position, word in enumerate(words)
    )

    return segmentation_records.Segmentation(
        createdAt=CREATED,
        expression=uri(EXPRESSION_NSID, rkey),
        tokenizations=(
            segmentation_records.Tokenization(
                uuid=defs_records.Uuid(value=f"tokenization-{rkey}"),
                kind="whitespace",
                tokens=tokens,
            ),
        ),
    )


def licensing_of(expression: str = "CC-BY-4.0") -> defs_records.Licensing:
    return defs_records.Licensing(
        expression=expression,
        licenses=(defs_records.LicenseRef(spdx=expression, appliesTo="annotations"),),
    )


def layer_of(
    rkey: str,
    kind: str,
    annotations: tuple[annotation_records.Annotation, ...],
    *,
    subkind: str | None = None,
    tokenization: str | None = None,
) -> annotation_records.AnnotationLayer:
    return annotation_records.AnnotationLayer(
        createdAt=CREATED,
        expression=uri(EXPRESSION_NSID, rkey),
        kind=kind,
        subkind=subkind,
        annotations=annotations,
        tokenizationId=None
        if tokenization is None
        else defs_records.Uuid(value=tokenization),
    )


def sourced_layer_of(
    rkey: str,
    kind: str,
    annotations: tuple[annotation_records.Annotation, ...],
    provenance: LayerProvenance,
) -> annotation_records.AnnotationLayer:
    return annotation_records.AnnotationLayer(
        createdAt=CREATED,
        expression=uri(EXPRESSION_NSID, rkey),
        kind=kind,
        annotations=annotations,
        sourceMethod=provenance.source_method,
        licensing=provenance.licensing,
        eprintRefs=provenance.eprint_refs,
        reproducibility=provenance.reproducibility,
    )


def graph_node_of(
    label: str,
    properties: tuple[tuple[str, str], ...] = (),
) -> graph_records.GraphNode:
    return graph_records.GraphNode(
        createdAt=CREATED,
        nodeType="situation",
        label=label,
        properties=defs_records.FeatureMap(
            entries=tuple(
                defs_records.Feature(key=key, value=value) for key, value in properties
            )
        )
        if properties
        else None,
    )


def graph_edge_of(
    source: str,
    target: str,
    edge_type: str = "denotes",
    *,
    metadata: defs_records.AnnotationMetadata | None = None,
) -> graph_records.GraphEdge:
    return graph_records.GraphEdge(
        createdAt=CREATED,
        edgeType=edge_type,
        source=defs_records.ObjectRef(recordRef=source),
        target=defs_records.ObjectRef(recordRef=target),
        metadata=metadata,
    )


def experiment_of(
    *,
    task: str | None,
    minimum: int | None = 1,
    maximum: int | None = 7,
    labels: tuple[str, ...] = (),
    features: tuple[tuple[str, str], ...] = (),
) -> judgment_records.ExperimentDef:
    return judgment_records.ExperimentDef(
        name="a study",
        createdAt=CREATED,
        taskType=task,
        scaleMin=minimum,
        scaleMax=maximum,
        labels=labels,
        features=defs_records.FeatureMap(
            entries=tuple(
                defs_records.Feature(key=key, value=value) for key, value in features
            )
        )
        if features
        else None,
    )


def judgment_set_of(
    rkey: str,
    experiment: str,
    judgments: tuple[judgment_records.Judgment, ...],
) -> judgment_records.JudgmentSet:
    return judgment_records.JudgmentSet(
        experimentRef=experiment,
        createdAt=CREATED,
        agent=defs_records.AgentRef(id=f"annotator-{rkey}"),
        judgments=judgments,
    )


def rating_of(target: str, value: int | None) -> judgment_records.Judgment:
    return judgment_records.Judgment(
        item=defs_records.ObjectRef(recordRef=target),
        scalarValue=value,
    )


def metadata_of(agent: str) -> defs_records.AnnotationMetadata:
    return defs_records.AnnotationMetadata(
        tool="fixture",
        agent=defs_records.AgentRef(id=agent),
    )


def relation_annotation(
    value: str,
    hypothesis: str,
    label: str | None = "entails",
) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=defs_records.Uuid(value=value),
        label=label,
        arguments=(
            annotation_records.ArgumentRef(
                role="hypothesis",
                target=defs_records.ObjectRef(recordRef=hypothesis),
            ),
        ),
    )


def relation_layer_of(
    premise: str,
    annotations: tuple[annotation_records.Annotation, ...],
    agent: str | None = None,
    *,
    provenance: LayerProvenance | None = None,
) -> annotation_records.AnnotationLayer:
    return annotation_records.AnnotationLayer(
        createdAt=CREATED,
        expression=premise,
        kind="relation",
        annotations=annotations,
        metadata=None if agent is None else metadata_of(agent),
        sourceMethod=None if provenance is None else provenance.source_method,
        licensing=None if provenance is None else provenance.licensing,
        eprintRefs=() if provenance is None else provenance.eprint_refs,
        reproducibility=None if provenance is None else provenance.reproducibility,
    )


def inference_experiment_of(
    premise: str,
    hypothesis: str,
) -> judgment_records.ExperimentDef:
    return judgment_records.ExperimentDef(
        name="inference",
        createdAt=CREATED,
        taskType="categorical",
        measureType="inference",
        labels=("entails", "contradicts", "neutral"),
        stimulusExpressionRefs=(premise, hypothesis),
    )


def pair_items(
    premises: tuple[tuple[str, str], ...] = (("premise", "someone slept"),),
    *,
    hypothesis_split: str = "train",
) -> dict[str, dx.Model]:
    records: dict[str, dx.Model] = {}

    for ordinal, (rkey, text) in enumerate(premises):
        records[uri(EXPRESSION_NSID, rkey)] = expression_of(rkey, text)
        records[uri(MEMBERSHIP_NSID, rkey)] = membership_of(
            rkey,
            "train",
            ordinal,
        )
        records[uri("pub.layers.segmentation.segmentation", rkey)] = segmentation_of(
            rkey,
            tuple(text.split()),
        )

    hypothesis = "hypothesis"
    records[uri(EXPRESSION_NSID, hypothesis)] = expression_of(
        hypothesis,
        "someone moved",
    )
    records[uri(MEMBERSHIP_NSID, hypothesis)] = membership_of(
        hypothesis,
        hypothesis_split,
        len(premises),
    )
    records[uri("pub.layers.segmentation.segmentation", hypothesis)] = segmentation_of(
        hypothesis,
        ("someone", "moved"),
    )

    return records


def one_sentence(
    *,
    text: str | None = "a b",
    words: tuple[str, ...] = ("a", "b"),
    split: str | None = "train",
) -> dict[str, dx.Model]:
    return {
        uri(EXPRESSION_NSID, "one"): expression_of("one", text),
        uri(MEMBERSHIP_NSID, "one"): membership_of("one", split),
        uri("pub.layers.segmentation.segmentation", "one"): segmentation_of(
            "one", words
        ),
    }


def duplicate_surface_reader(
    rkeys: tuple[str, str], words: tuple[str, ...]
) -> LayersCorpusReader:
    records: dict[str, dx.Model] = {}

    for ordinal, rkey in enumerate(rkeys):
        records[uri(EXPRESSION_NSID, rkey)] = expression_of(rkey, " ".join(words))
        records[uri(MEMBERSHIP_NSID, rkey)] = membership_of(rkey, "train", ordinal)
        records[uri("pub.layers.segmentation.segmentation", rkey)] = segmentation_of(
            rkey, words
        )

    return LayersCorpusReader(built(records))


@pytest.fixture
def mixed_provenance_reader() -> LayersCorpusReader:
    records: dict[str, dx.Model] = {}

    for ordinal, (rkey, source_method) in enumerate(
        (("manual", "manual-native"), ("automatic", "automatic"))
    ):
        records[uri(EXPRESSION_NSID, rkey)] = expression_of(rkey, rkey)
        records[uri(MEMBERSHIP_NSID, rkey)] = membership_of(rkey, "train", ordinal)
        records[uri("pub.layers.segmentation.segmentation", rkey)] = segmentation_of(
            rkey, (rkey,)
        )
        records[uri(LAYER_NSID, f"{rkey}-tags")] = sourced_layer_of(
            rkey,
            "token-tag",
            (
                annotation_records.Annotation(
                    uuid=defs_records.Uuid(value=f"{rkey}-tag"),
                    label="N",
                    tokenIndex=0,
                ),
            ),
            LayerProvenance(source_method, licensing_of()),
        )

    return LayersCorpusReader(built(records))


class TestLayersReader(CorpusReaderConformance):
    @pytest.fixture
    def subject(self) -> CorpusReaderSubject[str, Never]:
        return CorpusReaderSubject(
            ReaderFixture(load_sample),
            "lofi.data.corpora.layers",
        )


def test_the_committed_sample_ids_are_stable_across_hash_seeds() -> None:
    first = sample_ids_under("1")
    second = sample_ids_under("2")
    decoded = cast("dict[str, list[str]]", json.loads(first))

    assert first
    assert first == second
    assert first == json.dumps(
        decoded,
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    ).encode("utf-8")


@given(
    rkeys=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=2,
        max_size=2,
        unique=True,
    ).map(tuple),
    words=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8).filter(
            lambda value: value != "null"
        ),
        min_size=1,
        max_size=5,
    ).map(tuple),
)
@settings(max_examples=IDENTITY_EXAMPLES)
def test_distinct_source_ids_survive_load_and_hashing(
    rkeys: tuple[str, str], words: tuple[str, ...]
) -> None:
    reader = duplicate_surface_reader(rkeys, words)
    examples = tuple(reader.examples("train"))
    expected = tuple(ItemId(uri(EXPRESSION_NSID, rkey)) for rkey in rkeys)
    surfaces = tuple(
        tuple(token.surface for token in example.item.tokens) for example in examples
    )
    hasher = ItemHasher(
        Inventory("generated surfaces", tuple(dict.fromkeys(words))),
        Inventory("generated categories", ("category",)),
    )
    hashed = tuple(hasher(example) for example in examples)

    assert tuple(example.item.item_id for example in examples) == expected
    assert surfaces == (words, words)
    assert len(set(expected)) == 2
    assert tuple(item.item_id for item in hashed) == expected


def test_only_a_json_null_or_an_absent_text_is_missing() -> None:
    spelled = segmentation_records.Token(tokenIndex=0, text="null")
    absent = segmentation_records.Token(tokenIndex=0)
    present_null = segmentation_records.Token.model_validate_json(
        '{"tokenIndex": 0, "text": null}'
    )

    assert spelled.text == "null"
    assert json.loads(spelled.model_dump_json())["text"] == "null"
    assert absent.text is None
    assert present_null.text is None
    assert (
        segmentation_records.Token.model_validate_json(spelled.model_dump_json())
        == spelled
    )


@given(
    rkeys=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=2,
        max_size=2,
        unique=True,
    ).map(tuple),
    words=st.lists(
        st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=8),
        min_size=1,
        max_size=5,
    ).map(tuple),
)
@settings(max_examples=IDENTITY_EXAMPLES)
def test_predictions_and_gold_join_on_generated_source_ids(
    rkeys: tuple[str, str], words: tuple[str, ...]
) -> None:
    reader = duplicate_surface_reader(rkeys, words)
    items = tuple(reader.items("train"))
    annotations = tuple(
        Annotation(item.item_id, position, f"annotator-{position}")
        for position, item in enumerate(items)
    )
    dataset = EvalDataset(items, annotations)
    predictions = {
        item.item_id: f"prediction-{position}"
        for position, item in enumerate(reversed(items))
    }
    joined = dataset.join(predictions)

    assert tuple(pair.item_id for pair in joined.pairs) == tuple(
        item.item_id for item in items
    )
    assert tuple(pair.gold.value for pair in joined.pairs) == (0, 1)
    assert tuple(pair.prediction for pair in joined.pairs) == (
        "prediction-1",
        "prediction-0",
    )
    assert joined.unpredicted == ()
    assert joined.unmatched == ()


def test_the_reader_is_usable_as_the_port_a_consumer_takes() -> None:
    reader: CorpusReader[str, Never] = load_sample()

    assert reader.name == READER_NAME
    assert reader.grammar is None
    assert reader.lexicon is None
    assert sum(len(reader.items(split)) for split in reader.splits()) == len(
        list(reader.examples(reader.splits()[0]))
    ) + sum(len(reader.items(split)) for split in reader.splits()[1:])


def test_the_splits_are_the_ones_the_memberships_name(
    sample: LayersCorpusReader,
) -> None:
    assert sample.splits() == ("dev", "test", "train")


def test_every_thing_the_corpus_says_about_a_sentence_is_its_own_item(
    sample: LayersCorpusReader,
) -> None:
    ids = [item.item_id for item in sample.items("train")]

    assert TREE_LAYER in ids
    assert TAG_LAYER in ids
    assert DEPENDENCY_LAYER in ids
    assert GRAPH_EDGE in ids

    surfaces = [
        tuple(token.surface for token in item.tokens) for item in sample.items("train")
    ]

    assert surfaces.count(("the", "child", "sleeps")) == 4


def test_two_annotators_rating_one_sentence_give_two_items(
    sample: LayersCorpusReader,
) -> None:
    examples = list(sample.examples("test"))
    annotators = {
        example.supervision.annotator
        for example in examples
        if isinstance(example.supervision, Scalar)
    }

    assert len(examples) == 2
    assert {example.item.item_id for example in examples} == {REPEATED}
    assert annotators == {"sample-annotator-a", "sample-annotator-b"}


def test_a_rating_reads_against_the_scale_its_experiment_states(
    sample: LayersCorpusReader,
) -> None:
    ratings = [
        example.supervision
        for example in sample.examples("test")
        if isinstance(example.supervision, Scalar)
    ]

    assert ratings

    for rating in ratings:
        assert rating.scale.kind == "ordinal"
        assert rating.scale.levels == ORDINAL_LEVELS
        assert rating.scale.minimum == 1.0
        assert rating.scale.maximum == ORDINAL_LEVELS


def test_the_reader_takes_no_scale_of_its_own(sample: LayersCorpusReader) -> None:
    other = LayersCorpusReader(corpus_at(SAMPLE))
    mine = [
        example.supervision.scale
        for example in sample.examples("test")
        if isinstance(example.supervision, Scalar)
    ]
    theirs = [
        example.supervision.scale
        for example in other.examples("test")
        if isinstance(example.supervision, Scalar)
    ]

    assert mine == theirs


def test_a_token_span_recovers_the_word_it_covers(sample: LayersCorpusReader) -> None:
    mandarin = next(item for item in sample.items("dev") if item.item_id == MANDARIN)

    assert mandarin.text is not None
    assert [token.surface for token in mandarin.tokens] == ["孩子", "睡觉"]

    for token in mandarin.tokens:
        assert token.span is not None
        assert token.span.substring(mandarin.text) == token.surface
        assert token.span.length > len(token.surface)


def test_the_analysis_a_tree_layer_records_tiles_the_sentence(
    sample: LayersCorpusReader,
) -> None:
    analyzed = next(
        example
        for example in sample.examples("train")
        if isinstance(example.supervision, GoldDerivation)
    )
    gold = analyzed.supervision

    assert isinstance(gold, GoldDerivation)

    derivation = gold.derivation
    root = derivation.node(derivation.root)

    assert root.label == "S"
    assert root.span.start == 0
    assert root.span.end == len(analyzed.item.tokens)
    assert sorted(
        node.label for node in derivation.nodes.values() if node.is_lexical
    ) == ["N", "NP/N", "S\\NP"]


def test_the_relations_a_dependency_layer_records_arrive_in_order(
    sample: LayersCorpusReader,
) -> None:
    related = next(
        example
        for example in sample.examples("train")
        if isinstance(example.supervision, GoldDependencies)
    )
    gold = related.supervision

    assert isinstance(gold, GoldDependencies)
    assert [(arc.head, arc.dependent, arc.label) for arc in gold.dependencies] == [
        (2, 1, "nsubj"),
        (1, 0, "det"),
    ]


def test_a_token_tag_layer_gives_one_width_one_constraint_per_position(
    sample: LayersCorpusReader,
) -> None:
    tagged = next(
        example
        for example in sample.examples("train")
        if isinstance(example.supervision, SpanConstraints)
    )
    gold = tagged.supervision

    assert isinstance(gold, SpanConstraints)

    constraints = gold.constraints

    assert len(constraints) == len(tagged.item.tokens)
    assert [sorted(constraint.permitted) for constraint in constraints] == [
        ["NP/N"],
        ["N"],
        ["S\\NP"],
    ]


def test_an_unannotated_sentence_arrives_under_the_unlabeled_arm(
    sample: LayersCorpusReader,
) -> None:
    unlabeled = [
        example
        for example in sample.examples("dev")
        if isinstance(example.supervision, Unlabeled)
    ]

    assert len(unlabeled) == 1
    assert unlabeled[0].supervision == Unlabeled()


def test_an_annotation_layer_preserves_its_license_on_the_item(
    sample: LayersCorpusReader,
) -> None:
    analyzed = next(
        example
        for example in sample.examples("train")
        if isinstance(example.supervision, GoldDerivation)
    )
    provenance = analyzed.item.provenance

    assert provenance is not None
    assert provenance.source_method == "manual-native"
    assert provenance.licensing.expression == "CC0-1.0"
    assert [ref.identifier for ref in provenance.licensing.licenses] == ["CC0-1.0"]


def test_mixed_annotation_provenance_survives_selection_and_materialization(
    mixed_provenance_reader: LayersCorpusReader,
) -> None:
    loaded = tuple(mixed_provenance_reader.examples("train"))
    provenances = tuple(example.item.provenance for example in loaded)

    assert all(provenance is not None for provenance in provenances)
    assert [
        (provenance.source_method, provenance.licensing.expression)
        for provenance in provenances
        if provenance is not None
    ] == [
        ("manual-native", "CC-BY-4.0"),
        ("automatic", "CC-BY-4.0"),
    ]

    selected = (
        example
        for example in mixed_provenance_reader.examples("train")
        if (provenance := example.item.provenance) is not None
        and provenance.source_method == "manual-native"
    )
    materialized = tuple(selected)

    assert [example.item.item_id for example in materialized] == [
        ItemId(uri(LAYER_NSID, "manual-tags"))
    ]
    assert materialized[0].item.provenance == provenances[0]


def test_an_annotation_layer_with_incomplete_provenance_remains_unset() -> None:
    records = one_sentence(text="a", words=("a",))
    records[uri(LAYER_NSID, "one-tags")] = sourced_layer_of(
        "one",
        "token-tag",
        (tag("one", "N", 0),),
        LayerProvenance("manual-native", None),
    )
    example = next(iter(LayersCorpusReader(built(records)).examples("train")))

    assert example.item.provenance is None


def test_a_relation_layer_preserves_its_provenance() -> None:
    premise = uri(EXPRESSION_NSID, "premise")
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")
    layer_uri = uri(LAYER_NSID, "pair")
    eprint = "at://did:plc:test/pub.layers.eprint.eprint/relation"
    records = pair_items()
    records[layer_uri] = relation_layer_of(
        premise,
        (relation_annotation("pair", hypothesis),),
        provenance=LayerProvenance("automatic", licensing_of("CC0-1.0"), (eprint,)),
    )
    related = next(
        example
        for example in LayersCorpusReader(built(records)).examples("train")
        if isinstance(example.supervision, Relational)
    )
    provenance = related.item.provenance

    assert provenance is not None
    assert provenance.source_method == "automatic"
    assert provenance.licensing.expression == "CC0-1.0"
    assert provenance.eprint_refs == (eprint,)


def test_the_report_accounts_for_every_candidate(sample: LayersCorpusReader) -> None:
    report = sample.report()
    emitted = sum(len(sample.items(split)) for split in sample.splits())

    assert report.emitted == emitted
    assert report.records_seen == report.emitted + report.skipped
    assert report.skips_by_reason == {}


def test_the_arms_reported_are_the_ones_the_load_emitted(
    sample: LayersCorpusReader,
) -> None:
    emitted = {
        type(example.supervision)
        for split in sample.splits()
        for example in sample.examples(split)
    }

    assert {arm.__name__ for arm in sample.supervision_kinds()} == {
        arm.__name__ for arm in emitted
    }


def test_a_corpus_saying_nothing_reports_the_unlabeled_arm_alone() -> None:
    reader = LayersCorpusReader(built(one_sentence()))

    assert reader.supervision_kinds() == frozenset({Unlabeled})


def test_an_item_with_no_text_still_loads() -> None:
    reader = LayersCorpusReader(built(one_sentence(text=None)))
    item = next(iter(reader.items("train")))

    assert item.text is None
    assert [token.surface for token in item.tokens] == ["a", "b"]


def test_a_membership_naming_no_split_is_a_counted_skip() -> None:
    reader = LayersCorpusReader(built(one_sentence(split=None)))

    assert reader.splits() == ()
    assert reader.report().records_seen == 1
    assert reader.report().skips_by_reason == {LayersSkip.UNSPLIT: 1}


def test_a_membership_naming_an_absent_expression_is_a_counted_skip() -> None:
    records = one_sentence()
    del records[uri(EXPRESSION_NSID, "one")]
    reader = LayersCorpusReader(built(records))
    report = reader.report()

    assert report.emitted == 0
    assert report.records_seen == 1
    assert set(report.skips_by_reason) == {LayersSkip.UNHELD_EXPRESSION}


def test_a_tokenization_with_a_gap_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri("pub.layers.segmentation.segmentation", "one")] = segmentation_of(
        "one", ("a", "b"), start_at=1
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "a tokenization whose positions are not contiguous from zero"
    }


def test_a_tree_layer_with_two_roots_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(LAYER_NSID, "broken")] = layer_of(
        "one",
        "tree",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="a"), label="X", tokenIndex=0
            ),
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="b"), label="Y", tokenIndex=1
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "a tree layer holding other than one root"
    }


def test_a_layer_of_an_unread_kind_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(LAYER_NSID, "other")] = layer_of(
        "one",
        "span",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="a"), label="X", tokenIndex=0
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "an annotation layer of a kind the reader does not read"
    }


def test_one_repository_load_accumulates_each_skip_reason(tmp_path: Path) -> None:
    directory = tmp_path / "reported"
    repository = Repository.init(directory)
    invalid_uri = uri(EXPRESSION_NSID, "invalid")
    repository.save(invalid_uri, expression_of("invalid", "invalid"))
    invalid_path = next((directory / "records").glob("*.json"))
    invalid_path.write_text('{"id": "invalid"}', encoding="utf-8", newline="\n")
    records = one_sentence()
    records[uri(EXPRESSION_NSID, "unread")] = expression_of("unread", "a b")
    records[uri(MEMBERSHIP_NSID, "unread")] = membership_of("unread", "train", 1)
    records[uri("pub.layers.segmentation.segmentation", "unread")] = segmentation_of(
        "unread", ("a", "b")
    )
    records[uri(LAYER_NSID, "unread")] = layer_of(
        "unread",
        "span",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="unread"),
                label="other",
                tokenIndex=0,
            ),
        ),
    )
    records[uri(EXPRESSION_NSID, "gap")] = expression_of("gap", "a b")
    records[uri(MEMBERSHIP_NSID, "gap")] = membership_of("gap", "train", 2)
    records[uri("pub.layers.segmentation.segmentation", "gap")] = segmentation_of(
        "gap", ("a", "b"), start_at=1
    )

    for at_uri, record in records.items():
        repository.save(at_uri, record)

    repository.save(
        uri("pub.layers.ontology.typeDef", "ignored"),
        ontology_records.TypeDef(
            createdAt=CREATED,
            name="ignored",
            typeKind="entity",
            ontologyRef=uri("pub.layers.ontology.ontology", "main"),
        ),
    )

    with pytest.raises(ValueError, match="1 invalid Layers record"):
        corpus_at(directory)

    reader = reader_at(directory)
    report = reader.report()

    assert len(reader.items("train")) == 1
    assert report.records_seen == 4
    assert report.emitted == 1
    assert report.skips_by_reason == {
        "a Layers record that fails schema validation": 1,
        "an annotation layer of a kind the reader does not read": 1,
        "a tokenization whose positions are not contiguous from zero": 1,
    }


def test_the_three_pair_routes_produce_the_same_relational_value() -> None:
    premise = uri(EXPRESSION_NSID, "premise")
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")
    annotated_layer = uri(LAYER_NSID, "annotated-pair")
    elicited_layer = uri(LAYER_NSID, "elicited-pair")
    experiment_uri = uri(EXPERIMENT_NSID, "inference")
    records = pair_items()
    records[annotated_layer] = relation_layer_of(
        premise,
        (relation_annotation("annotated-pair", hypothesis),),
        "annotator-a",
    )
    records[elicited_layer] = relation_layer_of(
        premise,
        (relation_annotation("elicited-pair", hypothesis, None),),
    )
    records[uri(GRAPH_EDGE_NSID, "pair")] = graph_edge_of(
        premise,
        hypothesis,
        "entails",
        metadata=metadata_of("annotator-a"),
    )
    records[experiment_uri] = inference_experiment_of(premise, hypothesis)
    records[uri(JUDGMENT_SET_NSID, "annotator-a")] = judgment_set_of(
        "a",
        experiment_uri,
        (
            judgment_records.Judgment(
                item=defs_records.ObjectRef(
                    recordRef=elicited_layer,
                    objectId=defs_records.Uuid(value="elicited-pair"),
                ),
                categoricalValue="entails",
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))
    examples = [
        example
        for example in reader.examples("train")
        if isinstance(example.supervision, Relational)
    ]

    assert len(examples) == 3
    assert {example.item.item_id for example in examples} == {ItemId(premise)}
    assert {example.supervision for example in examples} == {
        Relational(
            ItemId(hypothesis),
            "entails",
            AnnotatorId("annotator-a"),
        )
    }
    assert reader.supervision_kinds() == frozenset({Relational, Unlabeled})


def test_a_continuous_inference_response_keeps_the_pair_as_one_bounded_item() -> None:
    premise = uri(EXPRESSION_NSID, "premise")
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")
    pair_layer = uri(LAYER_NSID, "continuous-pair")
    experiment_uri = uri(EXPERIMENT_NSID, "continuous-inference")
    records = pair_items()
    records[pair_layer] = relation_layer_of(
        premise,
        (relation_annotation("continuous-pair", hypothesis, None),),
    )
    records[experiment_uri] = judgment_records.ExperimentDef(
        name="continuous inference",
        createdAt=CREATED,
        taskType="continuous",
        measureType="inference",
        scaleMin=0,
        scaleMax=1,
        stimulusExpressionRefs=(premise, hypothesis),
    )
    records[uri(JUDGMENT_SET_NSID, "annotator-a")] = judgment_set_of(
        "a",
        experiment_uri,
        (
            judgment_records.Judgment(
                item=defs_records.ObjectRef(
                    recordRef=pair_layer,
                    objectId=defs_records.Uuid(value="continuous-pair"),
                ),
                continuousValue=judgment_records.DecimalValue(
                    coefficient=73, exponent=-2
                ),
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))
    examples = tuple(
        example
        for example in reader.examples("train")
        if isinstance(example.supervision, RelationalScalar)
    )

    assert len(examples) == 1
    response = examples[0].supervision
    assert examples[0].item.item_id == premise
    assert isinstance(response, RelationalScalar)
    assert response.other == hypothesis
    assert response.relation == "entails"
    assert response.value == 0.73
    assert response.scale.kind == "continuous"
    assert response.annotator == "annotator-a"
    assert reader.supervision_kinds() == frozenset({RelationalScalar, Unlabeled})


def test_identical_premises_paired_to_one_hypothesis_remain_distinct() -> None:
    premises = (("premise-a", "someone slept"), ("premise-b", "someone slept"))
    records = pair_items(premises)
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")

    for rkey, _ in premises:
        records[uri(GRAPH_EDGE_NSID, rkey)] = graph_edge_of(
            uri(EXPRESSION_NSID, rkey),
            hypothesis,
            "neutral",
        )

    reader = LayersCorpusReader(built(records))
    examples = [
        example
        for example in reader.examples("train")
        if isinstance(example.supervision, Relational)
    ]

    pairs: list[tuple[ItemId, ItemId]] = []

    for example in examples:
        supervision = example.supervision

        assert isinstance(supervision, Relational)
        pairs.append((example.item.item_id, supervision.other))

    assert pairs == [
        (ItemId(uri(EXPRESSION_NSID, "premise-a")), ItemId(hypothesis)),
        (ItemId(uri(EXPRESSION_NSID, "premise-b")), ItemId(hypothesis)),
    ]


def test_an_incoming_relation_edge_creates_no_reverse_example() -> None:
    records = pair_items()
    premise = uri(EXPRESSION_NSID, "premise")
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")
    records[uri(GRAPH_EDGE_NSID, "pair")] = graph_edge_of(
        premise,
        hypothesis,
        "contradicts",
    )
    reader = LayersCorpusReader(built(records))
    relations = [
        example
        for example in reader.examples("train")
        if isinstance(example.supervision, Relational)
    ]

    assert len(relations) == 1
    assert relations[0].item.item_id == ItemId(premise)
    supervision = relations[0].supervision

    assert isinstance(supervision, Relational)
    assert supervision.other == ItemId(hypothesis)


@pytest.mark.parametrize("missing_source", [False, True])
def test_an_unresolved_relation_endpoint_is_a_counted_skip(
    *,
    missing_source: bool,
) -> None:
    records = pair_items()
    premise = uri(EXPRESSION_NSID, "premise")
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")
    missing = uri(EXPRESSION_NSID, "missing")
    records[uri(GRAPH_EDGE_NSID, "pair")] = graph_edge_of(
        missing if missing_source else premise,
        hypothesis if missing_source else missing,
        "entails",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {
        "a relation whose endpoint is not an expression record": 1
    }


def test_a_nonexpression_relation_source_is_a_counted_skip() -> None:
    records = pair_items()
    node = uri(GRAPH_NODE_NSID, "source")
    records[node] = graph_node_of("a node")
    records[uri(GRAPH_EDGE_NSID, "pair")] = graph_edge_of(
        node,
        uri(EXPRESSION_NSID, "hypothesis"),
        "entails",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {
        "a relation whose endpoint is not an expression record": 1
    }


def test_relation_endpoints_in_different_splits_are_a_counted_skip() -> None:
    records = pair_items(hypothesis_split="test")
    records[uri(GRAPH_EDGE_NSID, "pair")] = graph_edge_of(
        uri(EXPRESSION_NSID, "premise"),
        uri(EXPRESSION_NSID, "hypothesis"),
        "entails",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {
        "a relation whose endpoints are not members of one split": 1
    }


def test_a_noncanonical_expression_relation_is_a_counted_skip() -> None:
    records = pair_items()
    records[uri(GRAPH_EDGE_NSID, "pair")] = graph_edge_of(
        uri(EXPRESSION_NSID, "premise"),
        uri(EXPRESSION_NSID, "hypothesis"),
        "entailment",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {
        "a relation outside the canonical inference vocabulary": 1
    }


def test_a_relation_annotation_without_a_hypothesis_is_a_counted_skip() -> None:
    records = pair_items()
    records[uri(LAYER_NSID, "pair")] = relation_layer_of(
        uri(EXPRESSION_NSID, "premise"),
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="pair"),
                label="entails",
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {
        "a relation annotation without exactly one hypothesis argument": 1
    }


@pytest.mark.parametrize(
    ("item", "reason"),
    [
        (
            defs_records.ObjectRef(objectId=defs_records.Uuid(value="pair")),
            "an elicited relation naming no pair layer",
        ),
        (
            defs_records.ObjectRef(recordRef=uri(LAYER_NSID, "pair")),
            "an elicited relation naming no pair annotation",
        ),
        (
            defs_records.ObjectRef(
                recordRef=uri(LAYER_NSID, "pair"),
                objectId=defs_records.Uuid(value="missing"),
            ),
            "an elicited relation naming no unique pair annotation",
        ),
        (
            defs_records.ObjectRef(
                recordRef=uri(LAYER_NSID, "pair"),
                objectId=defs_records.Uuid(value="pair"),
                localId=defs_records.Uuid(value="local"),
            ),
            "an elicited relation naming no exact pair reference",
        ),
    ],
)
def test_an_elicited_relation_without_both_pair_references_is_counted(
    item: defs_records.ObjectRef,
    reason: str,
) -> None:
    records = pair_items()
    hypothesis = uri(EXPRESSION_NSID, "hypothesis")
    layer_uri = uri(LAYER_NSID, "pair")
    experiment_uri = uri(EXPERIMENT_NSID, "inference")
    records[layer_uri] = relation_layer_of(
        uri(EXPRESSION_NSID, "premise"),
        (relation_annotation("pair", hypothesis, None),),
    )
    records[experiment_uri] = inference_experiment_of(
        uri(EXPRESSION_NSID, "premise"), hypothesis
    )
    records[uri(JUDGMENT_SET_NSID, "annotator-a")] = judgment_set_of(
        "a",
        experiment_uri,
        (judgment_records.Judgment(item=item, categoricalValue="entails"),),
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {reason: 1}


def test_the_graph_a_corpus_attaches_to_a_sentence_is_its_own_item(
    sample: LayersCorpusReader,
) -> None:
    graphed = next(
        example
        for example in sample.examples("train")
        if isinstance(example.supervision, GoldGraph)
    )
    gold = graphed.supervision

    assert isinstance(gold, GoldGraph)
    assert graphed.item.item_id == GRAPH_EDGE
    assert [node.identifier for node in gold.nodes] == [GRAPH_NODE]
    assert gold.nodes[0].node_type == "situation"
    assert gold.nodes[0].label == "sleep"
    assert dict(gold.nodes[0].properties) == {
        "tense": "present",
        "polarity": "positive",
    }
    assert gold.edges == ()


def test_two_edges_attaching_one_sentence_give_two_items() -> None:
    records = one_sentence()
    records[uri(GRAPH_NODE_NSID, "n")] = graph_node_of("sleep")
    records[uri(GRAPH_EDGE_NSID, "a")] = graph_edge_of(
        uri(EXPRESSION_NSID, "one"), uri(GRAPH_NODE_NSID, "n")
    )
    records[uri(GRAPH_EDGE_NSID, "b")] = graph_edge_of(
        uri(EXPRESSION_NSID, "one"), uri(GRAPH_NODE_NSID, "n"), "describes"
    )
    reader = LayersCorpusReader(built(records))
    ids = [item.item_id for item in reader.items("train")]

    assert ids == [uri(GRAPH_EDGE_NSID, "a"), uri(GRAPH_EDGE_NSID, "b")]
    assert reader.supervision_kinds() == frozenset({GoldGraph})


def test_graph_lookup_visits_only_incident_edges_in_uri_order(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    records = one_sentence()
    records[uri(GRAPH_NODE_NSID, "n")] = graph_node_of("sleep")
    records[uri(GRAPH_NODE_NSID, "m")] = graph_node_of("wake")
    records[uri(GRAPH_NODE_NSID, "o")] = graph_node_of("dream")
    records[uri(GRAPH_NODE_NSID, "x")] = graph_node_of("unrelated-source")
    records[uri(GRAPH_NODE_NSID, "y")] = graph_node_of("unrelated-target")
    records[uri(GRAPH_EDGE_NSID, "a")] = graph_edge_of(
        uri(EXPRESSION_NSID, "one"),
        uri(GRAPH_NODE_NSID, "n"),
    )
    records[uri(GRAPH_EDGE_NSID, "b")] = graph_edge_of(
        uri(EXPRESSION_NSID, "one"),
        uri(GRAPH_NODE_NSID, "n"),
        "describes",
    )
    records[uri(GRAPH_EDGE_NSID, "c")] = graph_edge_of(
        uri(GRAPH_NODE_NSID, "n"),
        uri(GRAPH_NODE_NSID, "m"),
        "follows",
    )
    records[uri(GRAPH_EDGE_NSID, "d")] = graph_edge_of(
        uri(GRAPH_NODE_NSID, "m"),
        uri(GRAPH_NODE_NSID, "o"),
        "causes",
    )
    records[uri(GRAPH_EDGE_NSID, "z-unrelated")] = graph_edge_of(
        uri(GRAPH_NODE_NSID, "x"),
        uri(GRAPH_NODE_NSID, "y"),
        "separate",
    )
    observed: list[str] = []
    original = reader_module.graph_start

    def recording_start(
        edge: graph_records.GraphEdge,
        expression: str,
    ) -> str | None:
        observed.append(edge.edgeType)

        return original(edge, expression)

    monkeypatch.setattr(reader_module, "graph_start", recording_start)
    reader = LayersCorpusReader(built(records))
    examples = tuple(reader.examples("train"))

    assert observed == ["denotes", "describes"]
    assert [example.item.item_id for example in examples] == [
        uri(GRAPH_EDGE_NSID, "a"),
        uri(GRAPH_EDGE_NSID, "b"),
    ]
    assert all(isinstance(example.supervision, GoldGraph) for example in examples)
    first = examples[0].supervision
    assert isinstance(first, GoldGraph)
    assert first == examples[1].supervision
    assert [node.identifier for node in first.nodes] == [
        uri(GRAPH_NODE_NSID, "m"),
        uri(GRAPH_NODE_NSID, "n"),
        uri(GRAPH_NODE_NSID, "o"),
    ]
    assert [(edge.source, edge.target, edge.edge_type) for edge in first.edges] == [
        (
            uri(GRAPH_NODE_NSID, "n"),
            uri(GRAPH_NODE_NSID, "m"),
            "follows",
        ),
        (
            uri(GRAPH_NODE_NSID, "m"),
            uri(GRAPH_NODE_NSID, "o"),
            "causes",
        ),
    ]
    assert observed == ["denotes", "describes"]


def test_examples_reuse_values_decoded_during_load(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0
    original = reader_module.tokens_of

    def counting_tokens(
        text: str | None,
        segmentations: Sequence[segmentation_records.Segmentation],
    ) -> tuple[Token, ...]:
        nonlocal calls
        calls += 1

        return original(text, segmentations)

    monkeypatch.setattr(reader_module, "tokens_of", counting_tokens)
    reader = LayersCorpusReader(built(one_sentence()))
    first = tuple(reader.examples("train"))
    second = tuple(reader.examples("train"))

    assert calls == 1
    assert first == second
    assert first[0] is second[0]


def test_a_graph_edge_naming_no_expression_attaches_no_item() -> None:
    records = one_sentence()
    records[uri(GRAPH_NODE_NSID, "n")] = graph_node_of("sleep")
    records[uri(GRAPH_NODE_NSID, "m")] = graph_node_of("wake")
    records[uri(GRAPH_EDGE_NSID, "a")] = graph_edge_of(
        uri(GRAPH_NODE_NSID, "n"), uri(GRAPH_NODE_NSID, "m"), "follows"
    )
    reader = LayersCorpusReader(built(records))

    assert reader.supervision_kinds() == frozenset({Unlabeled})
    assert reader.report().skips_by_reason == {}


def test_an_edge_attaching_a_sentence_to_no_graph_node_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(GRAPH_EDGE_NSID, "a")] = graph_edge_of(
        uri(EXPRESSION_NSID, "one"), uri(GRAPH_NODE_NSID, "absent")
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "a graph edge attaching an expression to no graph node"
    }


def test_a_graph_node_stating_no_type_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(GRAPH_NODE_NSID, "n")] = graph_records.GraphNode(
        createdAt=CREATED, nodeType="", label="sleep"
    )
    records[uri(GRAPH_EDGE_NSID, "a")] = graph_edge_of(
        uri(EXPRESSION_NSID, "one"), uri(GRAPH_NODE_NSID, "n")
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {LayersSkip.UNREADABLE_GRAPH}


def test_a_relation_layer_of_an_unread_subkind_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(LAYER_NSID, "coref")] = layer_of(
        "one",
        "relation",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="a"),
                label="coref",
                headIndex=1,
                targetIndex=0,
            ),
        ),
        subkind="coreference",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "an annotation layer of a subkind the reader does not read"
    }


def test_a_layer_naming_a_tokenization_the_item_does_not_hold_is_a_counted_skip() -> (
    None
):
    records = one_sentence()
    records[uri(LAYER_NSID, "tagged")] = layer_of(
        "one",
        "token-tag",
        (tag("a", "N", 0), tag("b", "V", 1)),
        tokenization="tokenization-elsewhere",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "an annotation layer naming a tokenization the item does not hold"
    }


def test_a_layer_naming_the_tokenization_the_item_was_read_on_is_emitted() -> None:
    records = one_sentence()
    records[uri(LAYER_NSID, "tagged")] = layer_of(
        "one",
        "token-tag",
        (tag("a", "N", 0), tag("b", "V", 1)),
        tokenization="tokenization-one",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().skips_by_reason == {}
    assert reader.supervision_kinds() == frozenset({SpanConstraints})


def test_a_token_tag_layer_carrying_confidence_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(LAYER_NSID, "tagged")] = layer_of(
        "one",
        "token-tag",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="a"),
                label="N",
                tokenIndex=0,
                confidence=500,
            ),
            tag("b", "V", 1),
        ),
        tokenization="tokenization-one",
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert reader.report().skips_by_reason == {
        "a token-tag layer carrying confidence": 1
    }


def test_a_response_whose_experiment_is_absent_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(JUDGMENT_SET_NSID, "a")] = judgment_set_of(
        "a",
        uri(EXPERIMENT_NSID, "missing"),
        (rating_of(uri(EXPRESSION_NSID, "one"), 3),),
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {LayersSkip.UNHELD_EXPERIMENT}


def test_a_response_holding_no_value_is_emitted_as_missing() -> None:
    records = one_sentence()
    records[uri(EXPERIMENT_NSID, "study")] = experiment_of(task="ordinal-scale")
    records[uri(JUDGMENT_SET_NSID, "a")] = judgment_set_of(
        "a",
        uri(EXPERIMENT_NSID, "study"),
        (rating_of(uri(EXPRESSION_NSID, "one"), None),),
    )
    reader = LayersCorpusReader(built(records))
    response = next(iter(reader.examples("train"))).supervision

    assert isinstance(response, Scalar)
    assert response.value is None
    assert response.features == frozenset({RESPONSE_MISSING})
    assert reader.report().emitted == 1
    assert reader.report().skips_by_reason == {}


def test_a_magnitude_experiment_states_its_modulus_on_the_scale() -> None:
    records = one_sentence()
    records[uri(EXPERIMENT_NSID, "study")] = experiment_of(
        task="magnitude",
        minimum=0,
        maximum=None,
        features=(("modulus", "20"),),
    )
    records[uri(JUDGMENT_SET_NSID, "a")] = judgment_set_of(
        "a",
        uri(EXPERIMENT_NSID, "study"),
        (rating_of(uri(EXPRESSION_NSID, "one"), 40),),
    )
    reader = LayersCorpusReader(built(records))
    response = next(iter(reader.examples("train"))).supervision

    assert isinstance(response, Scalar)
    assert response.scale.kind == "ratio"
    assert response.scale.modulus == 20.0
    assert response.value == 40.0


def test_a_categorical_experiment_reaches_the_label_arm_through_the_reader() -> None:
    records = one_sentence()
    records[uri(EXPERIMENT_NSID, "study")] = experiment_of(
        task="categorical", labels=("yes", "no")
    )
    records[uri(JUDGMENT_SET_NSID, "a")] = judgment_set_of(
        "a",
        uri(EXPERIMENT_NSID, "study"),
        (
            judgment_records.Judgment(
                item=defs_records.ObjectRef(recordRef=uri(EXPRESSION_NSID, "one")),
                categoricalValue="yes",
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))
    chosen = next(iter(reader.examples("train"))).supervision

    assert isinstance(chosen, Categorical)
    assert chosen.label == "yes"
    assert chosen.label_set == frozenset({"yes", "no"})
    assert reader.supervision_kinds() == frozenset({Categorical})


def test_a_categorical_null_is_emitted_as_missing() -> None:
    records = one_sentence()
    records[uri(EXPERIMENT_NSID, "study")] = experiment_of(
        task="categorical", labels=("yes", "no")
    )
    records[uri(JUDGMENT_SET_NSID, "a")] = judgment_set_of(
        "a",
        uri(EXPERIMENT_NSID, "study"),
        (
            judgment_records.Judgment(
                item=defs_records.ObjectRef(recordRef=uri(EXPRESSION_NSID, "one")),
            ),
        ),
    )
    reader = LayersCorpusReader(built(records))
    response = next(iter(reader.examples("train"))).supervision

    assert isinstance(response, Categorical)
    assert response.label is None
    assert response.features == frozenset({RESPONSE_MISSING})
    assert reader.report().emitted == 1


def test_a_loaded_judgment_study_reuses_the_layers_reader() -> None:
    experiment_uri = uri(EXPERIMENT_NSID, "study")
    target = uri(EXPRESSION_NSID, "one")
    records: dict[str, dx.Model] = {
        target: expression_of("one", "a b"),
        experiment_uri: experiment_of(task="ordinal-scale"),
    }
    count = 32
    expected_values: list[float] = []

    for index in range(count):
        rkey = f"annotator-{index:02d}"
        value = index % ORDINAL_LEVELS + 1
        expected_values.append(float(value))
        records[uri(JUDGMENT_SET_NSID, rkey)] = judgment_set_of(
            rkey,
            experiment_uri,
            (rating_of(target, value),),
        )

    reader = LayersCorpusReader(built_study(records, named=experiment_uri))
    examples = list(reader.examples("train"))
    responses = [
        example.supervision
        for example in examples
        if isinstance(example.supervision, Scalar)
    ]
    identities = {
        (example.item.item_id, response.annotator, experiment_uri)
        for example, response in zip(examples, responses, strict=True)
    }

    assert reader.splits() == ("train",)
    assert len(examples) == count
    assert len(identities) == count
    assert {example.item.item_id for example in examples} == {target}
    assert [response.value for response in responses] == expected_values
    assert reader.report().emitted == count
    assert reader.report().skips_by_reason == {}


def test_judgment_study_at_uses_the_lairs_loader(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = JudgmentStudy.new(uri(EXPERIMENT_NSID, "study"))
    calls: list[tuple[str, dict[str, str | bool | PdsClient | None]]] = []

    def load(
        at_uri: str,
        **options: str | bool | PdsClient | None,
    ) -> JudgmentStudy:
        calls.append((at_uri, options))

        return loaded

    monkeypatch.setattr(reader_module, "load_judgment_study", load)

    result = judgment_study_at(
        uri(EXPERIMENT_NSID, "study"),
        options=JudgmentStudyOptions(source="pds", follow_refs=False),
    )

    assert result is loaded
    assert calls == [
        (
            uri(EXPERIMENT_NSID, "study"),
            {
                "source": "pds",
                "cache_dir": None,
                "revision": None,
                "pds_client": None,
                "follow_refs": False,
            },
        )
    ]


def test_a_magnitude_experiment_with_no_modulus_is_a_counted_skip() -> None:
    records = one_sentence()
    records[uri(EXPERIMENT_NSID, "study")] = experiment_of(
        task="magnitude", minimum=0, maximum=None
    )
    records[uri(JUDGMENT_SET_NSID, "a")] = judgment_set_of(
        "a",
        uri(EXPERIMENT_NSID, "study"),
        (rating_of(uri(EXPRESSION_NSID, "one"), 40),),
    )
    reader = LayersCorpusReader(built(records))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "a magnitude experiment stating no modulus"
    }


def test_the_memberships_of_a_split_are_read_in_the_order_the_corpus_states() -> None:
    records: dict[str, dx.Model] = {}

    for rkey, ordinal in (("c", 0), ("a", 1), ("b", 2)):
        records[uri(EXPRESSION_NSID, rkey)] = expression_of(rkey, rkey)
        records[uri(MEMBERSHIP_NSID, rkey)] = membership_of(rkey, "train", ordinal)

    reader = LayersCorpusReader(built(records))

    assert [item.item_id for item in reader.items("train")] == [
        uri(EXPRESSION_NSID, "c"),
        uri(EXPRESSION_NSID, "a"),
        uri(EXPRESSION_NSID, "b"),
    ]


def test_a_split_with_no_ordinals_is_read_in_a_stable_order() -> None:
    records: dict[str, dx.Model] = {}

    for rkey in ("c", "a", "b"):
        records[uri(EXPRESSION_NSID, rkey)] = expression_of(rkey, rkey)
        records[uri(MEMBERSHIP_NSID, rkey)] = membership_of(rkey, "train", None)

    reader = LayersCorpusReader(built(records))

    assert [item.item_id for item in reader.items("train")] == [
        uri(EXPRESSION_NSID, "a"),
        uri(EXPRESSION_NSID, "b"),
        uri(EXPRESSION_NSID, "c"),
    ]


def test_a_split_the_corpus_does_not_have_is_empty() -> None:
    reader = LayersCorpusReader(built(one_sentence()))

    assert len(reader.items("dev")) == 0
    assert list(reader.examples("dev")) == []


def test_reading_a_split_twice_rebuilds_the_same_items() -> None:
    reader = load_sample()
    source = reader.items("train")

    assert tuple(source) == tuple(source)
    assert len(source) == len(tuple(source))


def test_the_repository_is_read_under_the_corpus_record_it_stages() -> None:
    corpus = corpus_at(SAMPLE)

    assert corpus.uri == SAMPLE_CORPUS
    assert set(corpus.expression_uris()) == {
        SLEEPS,
        REPEATED,
        MANDARIN,
        f"at://{AUTHORITY}/{EXPRESSION_NSID}/es-duerme",
    }


def test_a_repository_naming_two_corpora_asks_which_one(tmp_path: Path) -> None:
    repository = Repository.init(tmp_path / "two")

    for rkey in ("one", "other"):
        repository.save(
            uri(CORPUS_NSID, rkey),
            corpus_records.Corpus(name=rkey, createdAt=CREATED),
        )

    repository.commit("two corpora")

    with pytest.raises(ValueError, match="which of"):
        corpus_at(tmp_path / "two")


def test_a_repository_with_no_corpus_record_reads_every_membership(
    tmp_path: Path,
) -> None:
    repository = Repository.init(tmp_path / "loose")

    for at_uri, record in one_sentence().items():
        repository.save(at_uri, record)

    repository.commit("one sentence")
    corpus = corpus_at(tmp_path / "loose")
    reader = LayersCorpusReader(corpus)

    assert corpus.uri is None
    assert len(reader.items("train")) == 1


def test_a_record_of_a_collection_the_reader_does_not_read_is_left_out(
    tmp_path: Path,
) -> None:
    repository = Repository.init(tmp_path / "extra")

    for at_uri, record in one_sentence().items():
        repository.save(at_uri, record)

    repository.save(
        uri("pub.layers.ontology.typeDef", "t"),
        ontology_records.TypeDef(
            createdAt=CREATED,
            name="a type",
            typeKind="entity",
            ontologyRef=uri("pub.layers.ontology.ontology", "o"),
        ),
    )
    repository.commit("one sentence and an ontology type")
    corpus = corpus_at(tmp_path / "extra")

    assert uri("pub.layers.ontology.typeDef", "t") not in corpus.pool.uris()
    assert len(LayersCorpusReader(corpus).items("train")) == 1


SEGMENTATION_NSID: Final = "pub.layers.segmentation.segmentation"

WORDS: Final = ("the", "child", "sleeps")

LEXICON: Final = Inventory("surfaces", WORDS, ("<unknown>",))

CATEGORIES: Final = Inventory("categories", ("NP/N", "N", "S\\NP"))


def tag(value: str, label: str, position: int) -> annotation_records.Annotation:
    return annotation_records.Annotation(
        uuid=defs_records.Uuid(value=value),
        label=label,
        tokenIndex=position,
    )


def item_corpus() -> Corpus:
    records: dict[str, dx.Model] = {
        uri(EXPRESSION_NSID, "tagged"): expression_of("tagged", "the child sleeps"),
        uri(MEMBERSHIP_NSID, "tagged"): membership_of("tagged", "train", 0),
        uri(SEGMENTATION_NSID, "tagged"): segmentation_of("tagged", WORDS),
        uri(LAYER_NSID, "tagged"): layer_of(
            "tagged",
            "token-tag",
            (
                tag("t0", "NP/N", 0),
                tag("t0-other", "N", 0),
                tag("t1", "N", 1),
                tag("t2", "S\\NP", 2),
            ),
        ),
        uri(EXPRESSION_NSID, "plain"): expression_of("plain", "the child sleeps"),
        uri(MEMBERSHIP_NSID, "plain"): membership_of("plain", "train", 1),
        uri(SEGMENTATION_NSID, "plain"): segmentation_of("plain", WORDS),
        uri(EXPRESSION_NSID, "silent"): expression_of("silent", None),
        uri(MEMBERSHIP_NSID, "silent"): membership_of("silent", "train", 2),
        uri(SEGMENTATION_NSID, "silent"): segmentation_of("silent", WORDS),
        uri(EXPRESSION_NSID, "stimulus"): expression_of("stimulus", "the child sleeps"),
        uri(MEMBERSHIP_NSID, "stimulus"): membership_of("stimulus", "test", 0),
        uri(EXPERIMENT_NSID, "study"): experiment_of(task="ordinal-scale"),
        uri(JUDGMENT_SET_NSID, "a"): judgment_set_of(
            "a",
            uri(EXPERIMENT_NSID, "study"),
            (rating_of(uri(EXPRESSION_NSID, "stimulus"), 6),),
        ),
    }

    return built(records)


def hashable(reader: LayersCorpusReader) -> Iterator[Example[Supervision[str]]]:
    for split in reader.splits():
        for example in reader.examples(split):
            if isinstance(example.supervision, Unlabeled | Scalar | SpanConstraints):
                yield example


def only(reader: LayersCorpusReader, rkey: str) -> Example[Supervision[str]]:
    return next(
        example for example in hashable(reader) if example.item.item_id.endswith(rkey)
    )


def item_fixture() -> ItemFixture:
    reader = LayersCorpusReader(item_corpus())

    return ItemFixture(
        ItemHasher(LEXICON, CATEGORIES),
        lambda: hashable(reader),
        only(reader, "tagged"),
        only(reader, "/plain"),
        only(reader, "/silent"),
        ("dog", "runs"),
    )


class TestLayersItems:
    pytestmark = pytest.mark.conformance

    @pytest.mark.parametrize("law", ITEM_LAWS, ids=[law.__name__ for law in ITEM_LAWS])
    def test_the_items_hold_to_each_lesson(
        self,
        law: Callable[[ItemFixture], None],
    ) -> None:
        law(item_fixture())


def test_the_six_lessons_run_in_one_call_over_what_the_reader_read() -> None:
    check_item_laws(item_fixture())


def test_the_corpus_offers_a_position_two_categories_and_the_adapter_keeps_both() -> (
    None
):
    fixture = item_fixture()
    hashed = fixture.hasher(fixture.ambiguous)

    assert hashed.analyses[0] == frozenset(
        {CATEGORIES.index("NP/N"), CATEGORIES.index("N")}
    )
    assert hashed.is_ambiguous


def test_an_item_the_reader_read_with_no_text_hashes_to_its_own_codes() -> None:
    fixture = item_fixture()
    hashed = fixture.hasher(fixture.textless)

    assert fixture.textless.item.text is None
    assert hashed.symbols == tuple(LEXICON.index(word) for word in WORDS)


def test_a_rating_the_reader_read_keeps_the_scale_its_experiment_stated() -> None:
    fixture = item_fixture()
    rated = next(
        example
        for example in fixture.examples()
        if isinstance(example.supervision, Scalar)
    )
    hashed = fixture.hasher(rated)

    assert hashed.response == rated.supervision
    assert hashed.response is not None
    assert hashed.response.scale.levels == ORDINAL_LEVELS


def sample_tree_layer() -> annotation_records.AnnotationLayer:
    for at_uri, record in _models_of(
        corpus_at(SAMPLE), annotation_records.AnnotationLayer
    ):
        if at_uri == TREE_LAYER:
            return record

    raise AssertionError(TREE_LAYER)


def test_the_sample_ccg_tree_layer_survives_a_lift_and_a_lowering() -> None:
    layer = sample_tree_layer()
    lift = lift_tree(layer)
    written = lower_tree(lift)

    assert len(written) == len(layer.annotations)
    assert written == layer.annotations
    assert [source.uuid for source in lift.sources] == [
        annotation.uuid.value for annotation in layer.annotations
    ]


def test_the_sample_ccg_tree_layer_lifts_to_the_analysis_it_records() -> None:
    layer = sample_tree_layer()
    derivation = lift_tree(layer).derivation
    root = derivation.node(derivation.root)

    assert root.span.start == 0
    assert root.span.end == len(derivation.leaves())
    assert [leaf.label for leaf in derivation.leaves()] == [
        annotation.label for annotation in layer.annotations if not annotation.childIds
    ]


SIGNATURE: Final = Signature(
    "cat", ("cat",), (Operator("S", "cat"), Operator("NP", "cat"))
)

ONTOLOGY: Final = "at://did:plc:test/pub.layers.ontology.ontology/ccg"


def typed_tree_records() -> dict[str, dx.Model]:
    records = one_sentence()
    records[uri(LAYER_NSID, "typed")] = layer_of(
        "one",
        "tree",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="r"),
                label="fa",
                ontologyTypeRef=f"{ONTOLOGY}#S",
                childIds=(
                    defs_records.Uuid(value="l"),
                    defs_records.Uuid(value="m"),
                ),
            ),
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="l"),
                label="lex",
                ontologyTypeRef=f"{ONTOLOGY}#NP",
                parentId=defs_records.Uuid(value="r"),
                tokenIndex=0,
            ),
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="m"),
                label="lex",
                ontologyTypeRef=f"{ONTOLOGY}#NP",
                parentId=defs_records.Uuid(value="r"),
                tokenIndex=1,
            ),
        ),
    )

    return records


def test_a_tree_layer_reads_its_categories_against_the_signature_the_run_declares() -> (
    None
):
    reader = LayersCorpusReader(built(typed_tree_records()), SignatureLabels(SIGNATURE))
    analyses = [
        example.supervision
        for example in reader.examples("train")
        if isinstance(example.supervision, GoldDerivation)
    ]

    assert len(analyses) == 1

    derivation = analyses[0].derivation
    root = derivation.node(derivation.root)

    assert root.label == "S"
    assert root.rule == "fa"
    assert [leaf.label for leaf in derivation.leaves()] == ["NP", "NP"]
    assert reader.report().skips_by_reason == {}


def test_a_tree_layer_naming_a_type_the_signature_lacks_is_a_counted_skip() -> None:
    narrower = Signature("cat", ("cat",), (Operator("NP", "cat"),))
    reader = LayersCorpusReader(built(typed_tree_records()), SignatureLabels(narrower))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "an ontology type the run's label inventory does not admit"
    }


def test_a_tree_layer_naming_a_type_under_a_reader_given_no_signature_is_a_skip() -> (
    None
):
    reader = LayersCorpusReader(built(typed_tree_records()))

    assert reader.report().emitted == 0
    assert set(reader.report().skips_by_reason) == {
        "an ontology type reference under a run that declares no label inventory"
    }


class AdmittedSteps:
    __slots__ = ("admitted", "calls")

    def __init__(self, *admitted: tuple[tuple[str, ...], str]) -> None:
        self.admitted = frozenset(admitted)
        self.calls: list[tuple[tuple[str, ...], str]] = []

    def recover(self, inputs: Sequence[str], output: str, /) -> tuple[str, ...]:
        step = tuple(inputs), output
        self.calls.append(step)

        return ("step",) if step in self.admitted else ()


class Exact:
    __slots__ = ()

    @property
    def name(self) -> str:
        return "exact"

    @property
    def has_key(self) -> bool:
        return True

    def key(self, term: str, /) -> str:
        return term

    def equivalent(self, left: str, right: str, /) -> bool:
        return left == right


MATCHING: Final = Exact()


class Relation:
    __slots__ = ("_arity", "_holds", "_name")

    def __init__(
        self,
        name: str,
        arity: int,
        holds: Callable[[tuple[str, ...], str], bool],
    ) -> None:
        self._name = name
        self._arity = arity
        self._holds = holds

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset({"check"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        return False

    @property
    def matching(self) -> Exact:
        return MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    def apply(self, inputs: tuple[str, ...], /) -> Iterator[str]:
        del inputs

        return iter(())

    def admits(self, inputs: tuple[str, ...], output: str, /) -> bool:
        return self._holds(inputs, output)

    def residuate(
        self,
        output: str,
        known: tuple[Slot[str], ...],
        /,
    ) -> Iterator[str]:
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": self._name}


PREDICATION: Final = (("NP", "NP"), "S")


def predication_rules() -> RuleSet[str]:
    return RuleSet(
        "predication",
        (
            Relation(
                "fa",
                2,
                lambda inputs, output: (inputs, output) == PREDICATION,
            ),
        ),
        MATCHING,
        NoNormalForm(),
    )


def typed_tree(
    rkey: str,
    root: str,
    leaves: tuple[str, ...],
    *,
    rule: str = "fa",
) -> annotation_records.AnnotationLayer:
    return layer_of(
        rkey,
        "tree",
        (
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value="r"),
                label=rule,
                ontologyTypeRef=f"{ONTOLOGY}#{root}",
                childIds=tuple(
                    defs_records.Uuid(value=f"l{position}")
                    for position in range(len(leaves))
                ),
            ),
            *(
                annotation_records.Annotation(
                    uuid=defs_records.Uuid(value=f"l{position}"),
                    label="lex",
                    ontologyTypeRef=f"{ONTOLOGY}#{leaf}",
                    parentId=defs_records.Uuid(value="r"),
                    tokenIndex=position,
                )
                for position, leaf in enumerate(leaves)
            ),
        ),
    )


def forest_records() -> dict[str, dx.Model]:
    records: dict[str, dx.Model] = {}

    for ordinal, (rkey, split, root, leaves) in enumerate(
        (
            ("dev-fine", "dev", "S", ("NP", "NP")),
            ("train-fine", "train", "S", ("NP", "NP")),
            ("train-odd", "train", "S", ("N", "NP")),
            ("train-wide", "train", "S", ("NP", "N")),
        )
    ):
        records[uri(EXPRESSION_NSID, rkey)] = expression_of(rkey, "a b")
        records[uri(MEMBERSHIP_NSID, rkey)] = membership_of(rkey, split, ordinal)
        records[uri("pub.layers.segmentation.segmentation", rkey)] = segmentation_of(
            rkey, ("a", "b")
        )
        records[uri(LAYER_NSID, f"{rkey}-tree")] = typed_tree(rkey, root, leaves)

    records[uri(LAYER_NSID, "train-wide-tags")] = layer_of(
        "train-wide",
        "token-tag",
        tuple(
            annotation_records.Annotation(
                uuid=defs_records.Uuid(value=f"tag{position}"),
                label="N",
                tokenIndex=position,
            )
            for position in range(2)
        ),
    )

    return records


WIDER: Final = Signature(
    "cat",
    ("cat",),
    (Operator("S", "cat"), Operator("NP", "cat"), Operator("N", "cat")),
)


def decoded_in_order(
    reader: LayersCorpusReader,
) -> list[tuple[ItemId, Derivation[str]]]:
    return [
        (example.item.item_id, example.supervision.derivation)
        for split in reader.splits()
        for example in reader.examples(split)
        if isinstance(example.supervision, GoldDerivation)
    ]


def outcome_of(reader: LayersCorpusReader, item_id: ItemId) -> DerivationOutcome[str]:
    outcomes = reader.reachability()

    assert outcomes is not None

    return dict(outcomes)[item_id]


def test_a_reader_given_no_comparison_inventory_classifies_nothing() -> None:
    reader = LayersCorpusReader(built(typed_tree_records()), SignatureLabels(SIGNATURE))

    assert reader.reachability() is None
    assert reader.report().analyses is None
    assert reader.report().reachability_report is None


def test_a_decoded_derivation_is_classified_once_beside_its_item() -> None:
    rules = AdmittedSteps(PREDICATION)
    reader = LayersCorpusReader(
        built(typed_tree_records()),
        SignatureLabels(SIGNATURE),
        comparison=ComparisonInventory(SignatureLabels(SIGNATURE), rules),
    )
    (item_id, derivation), *rest = decoded_in_order(reader)

    assert rest == []
    assert reader.reachability() == ((item_id, ReachableDerivation()),)
    assert rules.calls == [PREDICATION]
    assert derivation.node(derivation.root).rule == "fa"
    assert reader.report().analyses == AnalysisCounts(1, 1)

    report = reader.report().reachability_report

    assert report is not None
    assert report.outcomes == (ReachableDerivation(),)
    assert report.checked_steps == 1


def test_the_declared_rule_set_recovers_a_step_through_the_recovery_port() -> None:
    reader = LayersCorpusReader(
        built(forest_records()),
        SignatureLabels(WIDER),
        comparison=ComparisonInventory(SignatureLabels(WIDER), predication_rules()),
    )
    decoded = decoded_in_order(reader)

    assert [outcome.kind for _, outcome in reader.reachability() or ()] == [
        "reachable",
        "reachable",
        "unreachable_by_rule",
        "unreachable_by_rule",
    ]
    assert outcome_of(reader, decoded[2][0]) == RuleBlocker(
        decoded[2][1].root, ("N", "NP"), "S", "fa"
    )


def test_a_narrower_comparison_signature_emits_the_item_and_blocks_it_by_label() -> (
    None
):
    narrower = Signature("cat", ("cat",), (Operator("S", "cat"),))
    reader = LayersCorpusReader(
        built(typed_tree_records()),
        SignatureLabels(SIGNATURE),
        comparison=ComparisonInventory(
            SignatureLabels(narrower), AdmittedSteps(PREDICATION)
        ),
    )
    (item_id, derivation), *_ = decoded_in_order(reader)
    report = reader.report().reachability_report

    assert reader.report().emitted == 1
    assert reader.report().skips_by_reason == {}
    assert outcome_of(reader, item_id) == LabelBlocker(derivation.preorder[1], "NP")
    assert report is not None
    assert report.rule_blocked_derivations == 0
    assert reader.report().analyses == AnalysisCounts(1, 0)


def test_an_unresolvable_step_under_admitted_labels_is_unreachable_by_rule() -> None:
    reader = LayersCorpusReader(
        built(typed_tree_records()),
        SignatureLabels(SIGNATURE),
        comparison=ComparisonInventory(SignatureLabels(SIGNATURE), AdmittedSteps()),
    )
    (item_id, derivation), *_ = decoded_in_order(reader)
    report = reader.report().reachability_report

    assert outcome_of(reader, item_id) == RuleBlocker(
        derivation.root, ("NP", "NP"), "S", "fa"
    )
    assert report is not None
    assert report.label_blocked_derivations == 0


def test_outcomes_follow_stable_item_order_across_splits_and_sum_to_the_decoded() -> (
    None
):
    reader = LayersCorpusReader(
        built(forest_records()),
        SignatureLabels(WIDER),
        comparison=ComparisonInventory(
            SignatureLabels(
                Signature(
                    "cat", ("cat",), (Operator("S", "cat"), Operator("NP", "cat"))
                )
            ),
            AdmittedSteps(PREDICATION),
        ),
    )
    outcomes = reader.reachability()
    decoded = decoded_in_order(reader)
    report = reader.report().reachability_report

    assert outcomes is not None
    assert [item_id for item_id, _ in outcomes] == [item_id for item_id, _ in decoded]
    assert len(decoded) < reader.report().emitted
    assert [outcome.kind for _, outcome in outcomes] == [
        "reachable",
        "reachable",
        "unreachable_by_label",
        "unreachable_by_label",
    ]
    assert report is not None
    assert report.checked_derivations == len(decoded)
    assert (
        report.reachable_derivations
        + report.label_blocked_derivations
        + report.rule_blocked_derivations
    ) == len(decoded)
    assert reader.report().analyses == AnalysisCounts(len(decoded), 2)


def test_the_kernel_runs_once_over_every_decoded_derivation(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[list[Derivation[str]]] = []
    kernel = reader_module.reachability_report

    def counting[Recovered](
        labels: LabelMembership[str],
        rules: StepRecovery[str, Recovered],
        decoded: Iterable[Derivation[str]],
        /,
    ) -> ReachabilityReport[str]:
        held = list(decoded)
        calls.append(held)

        return kernel(labels, rules, held)

    monkeypatch.setattr(reader_module, "reachability_report", counting)
    reader = LayersCorpusReader(
        built(forest_records()),
        SignatureLabels(WIDER),
        comparison=ComparisonInventory(
            SignatureLabels(WIDER), AdmittedSteps(PREDICATION)
        ),
    )

    assert calls == [[derivation for _, derivation in decoded_in_order(reader)]]


def test_the_committed_sample_classifies_its_tree_layers_against_an_inventory() -> None:
    comparison = ComparisonInventory(
        SignatureLabels(SIGNATURE), AdmittedSteps(PREDICATION)
    )
    reader = reader_at(SAMPLE, comparison=comparison)
    outcomes = reader.reachability()
    decoded = decoded_in_order(reader)

    assert outcomes is not None
    assert decoded
    assert [item_id for item_id, _ in outcomes] == [item_id for item_id, _ in decoded]
    assert reader.report().reachability_report is not None
    assert (
        reader.report()
        == LayersCorpusReader(corpus_at(SAMPLE), comparison=comparison).report()
    )
