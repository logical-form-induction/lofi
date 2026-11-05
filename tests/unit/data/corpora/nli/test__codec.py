from collections import Counter
from decimal import Decimal
from pathlib import Path
from typing import Final

import pytest
from hypothesis import strategies as st
from lairs import Corpus
from lairs.data import JudgmentStudy
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import segmentation as segmentation_records
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data import IsoCodec, Relational, RelationalScalar, Unlabeled
from lofi.data.corpora.layers import LayersCorpusReader
from lofi.data.corpora.nli import (
    ChaosNliCodec,
    MalformedNliError,
    MnliCodec,
    NliFile,
    NliFormat,
    SnliCodec,
    UnliCodec,
)
from lofi.data.laws import CodecFixture, IsoFixture
from lofi.spec.datasets import CorpusReleaseSpec

type Subject = CodecSubject[NliFile, CorpusFragment, FragmentRecord]
type NliCodec = IsoCodec[NliFile, CorpusFragment, FragmentRecord]
type LayerModel = (
    annotation_records.AnnotationLayer
    | corpus_records.Corpus
    | corpus_records.Membership
    | expression_records.Expression
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet
    | segmentation_records.Segmentation
)

FIXTURES: Final = Path(__file__).with_name("fixtures")
EPRINT: Final = "at://did:plc:test/pub.layers.eprint.eprint/nli"
INLINE: Final = (
    '{"gold_label":"entailment","pairID":"inline-0",'
    '"sentence1":"Birds fly.","sentence2":"Animals fly."}\n'
)


def specification(release_id: str = "v1") -> CorpusReleaseSpec:
    return CorpusReleaseSpec(
        source_id="synthetic-nli",
        release_id=release_id,
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=(EPRINT,),
    )


def fixture_text(name: str) -> str:
    return (FIXTURES / name).read_text(encoding="utf-8")


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    return fragment.records


def subject_for(codec: NliCodec, sources: tuple[str, ...]) -> Subject:
    strategy = st.sampled_from((*sources, *(source.encode() for source in sources)))
    files = st.sampled_from(
        tuple(codec.iso.backward(codec.decode(src)) for src in sources)
    )
    return CodecSubject(
        IsoFixture(codec.iso, files, files.map(codec.iso.forward), 8),
        CodecFixture(codec, strategy, records_of, 8),
    )


class TestNliCodec(CodecConformance):
    @pytest.fixture(
        params=tuple(NliFormat),
        ids=tuple(NliFormat),
    )
    def subject(self, request: pytest.FixtureRequest) -> Subject:
        name = request.param
        spec = specification()

        if name is NliFormat.SNLI:
            return subject_for(SnliCodec(spec), (fixture_text("snli.jsonl"), INLINE))
        if name is NliFormat.MNLI:
            return subject_for(MnliCodec(spec), (fixture_text("mnli.jsonl"),))
        if name is NliFormat.UNLI:
            return subject_for(UnliCodec(spec), (fixture_text("unli.tsv"),))

        return subject_for(ChaosNliCodec(spec), (fixture_text("chaosnli.jsonl"),))


def model_of(record: FragmentRecord) -> LayerModel:
    models = {
        "pub.layers.annotation.annotationLayer": annotation_records.AnnotationLayer,
        "pub.layers.corpus.corpus": corpus_records.Corpus,
        "pub.layers.corpus.membership": corpus_records.Membership,
        "pub.layers.expression.expression": expression_records.Expression,
        "pub.layers.judgment.experimentDef": judgment_records.ExperimentDef,
        "pub.layers.judgment.judgmentSet": judgment_records.JudgmentSet,
        "pub.layers.segmentation.segmentation": segmentation_records.Segmentation,
    }
    return models[record.nsid].model_validate_json(record.value_json)


def reader_of(fragment: CorpusFragment) -> LayersCorpusReader:
    corpus_record = next(
        record
        for record in fragment.records
        if record.nsid == "pub.layers.corpus.corpus"
    )
    corpus = Corpus(uri=corpus_record.local_id)

    for record in fragment.records:
        corpus.add_record(record.local_id, model_of(record))

    return LayersCorpusReader(corpus)


@pytest.mark.parametrize(
    ("codec", "fixture", "relations"),
    [
        (SnliCodec(specification()), "snli.jsonl", ("entails", "contradicts")),
        (MnliCodec(specification()), "mnli.jsonl", ("neutral", "entails")),
    ],
)
def test_static_labels_round_trip_and_name_resolvable_items(
    codec: NliCodec,
    fixture: str,
    relations: tuple[str, ...],
) -> None:
    source = fixture_text(fixture)
    fragment = codec.decode(source)
    examples = tuple(reader_of(fragment).examples("test"))
    responses = tuple(
        supervision
        for example in examples
        if isinstance((supervision := example.supervision), Relational)
    )
    item_ids = {example.item.item_id for example in examples}

    assert codec.encode(fragment.records) == source
    assert tuple(response.relation for response in responses) == relations
    assert all(response.other in item_ids for response in responses)
    assert sum(isinstance(example.supervision, Unlabeled) for example in examples) == 2


def test_source_ids_are_stable_and_release_scoped() -> None:
    source = fixture_text("snli.jsonl")
    first = SnliCodec(specification()).decode(source)
    second = SnliCodec(specification()).decode(source)
    changed = SnliCodec(specification("v2")).decode(source)

    assert tuple(record.local_id for record in first.records) == tuple(
        record.local_id for record in second.records
    )
    assert {record.local_id for record in first.records}.isdisjoint(
        record.local_id for record in changed.records
    )


def test_chaos_counts_expand_without_aggregation_or_participant_ids() -> None:
    codec = ChaosNliCodec(specification())
    source = fixture_text("chaosnli.jsonl")
    fragment = codec.decode(source)
    responses = tuple(
        example.supervision
        for example in reader_of(fragment).examples("test")
        if isinstance(example.supervision, Relational)
    )

    assert codec.encode(fragment.records) == source
    assert len(responses) == 100
    assert Counter(response.relation for response in responses) == {
        "entails": 34,
        "contradicts": 33,
        "neutral": 33,
    }
    assert all(response.annotator is None for response in responses)

    changed = source.replace('"majority_label":"e"', '"majority_label":"c"')
    changed_responses = tuple(
        example.supervision
        for example in reader_of(codec.decode(changed)).examples("test")
        if isinstance(example.supervision, Relational)
    )
    assert changed_responses == responses


def test_pair_records_retain_source_metadata_and_utf8_spans() -> None:
    fragment = SnliCodec(specification()).decode(fixture_text("snli.jsonl"))
    models = tuple(model_of(record) for record in fragment.records)
    expressions = tuple(
        model for model in models if isinstance(model, expression_records.Expression)
    )
    layers = tuple(
        model
        for model in models
        if isinstance(model, annotation_records.AnnotationLayer)
    )
    segmentations = tuple(
        model
        for model in models
        if isinstance(model, segmentation_records.Segmentation)
    )

    assert len(expressions) == 4
    assert len(layers) == 2
    assert all(layer.sourceMethod == "manual" for layer in layers)
    assert all(layer.eprintRefs == (EPRINT,) for layer in layers)
    assert all(
        layer.licensing is not None and layer.licensing.expression == "CC0-1.0"
        for layer in layers
    )
    cafe = segmentations[0]
    token = cafe.tokenizations[0].tokens[1]
    assert token.textSpan is not None
    assert (token.text, token.textSpan.byteStart, token.textSpan.byteEnd) == (
        "café",
        2,
        7,
    )


def test_unli_round_trips_exact_decimals_through_lairs_and_the_layers_reader() -> None:
    codec = UnliCodec(specification())
    source = fixture_text("unli.tsv")
    fragment = codec.decode(source)
    models = tuple(model_of(record) for record in fragment.records)
    experiment = next(
        model for model in models if isinstance(model, judgment_records.ExperimentDef)
    )
    judgments = tuple(
        judgment
        for model in models
        if isinstance(model, judgment_records.JudgmentSet)
        for judgment in model.judgments
    )
    study = JudgmentStudy.new()

    for record, model in zip(fragment.records, models, strict=True):
        study.add_record(record.local_id, model)

    flattened = tuple(study.judgments())
    continuous = tuple(
        example.supervision
        for example in reader_of(fragment).examples("test")
        if isinstance(example.supervision, RelationalScalar)
    )

    assert codec.encode(fragment.records) == source
    assert (experiment.taskType, experiment.scaleMin, experiment.scaleMax) == (
        "continuous",
        0,
        1,
    )
    assert all(judgment.scalarValue is None for judgment in judgments)
    assert [
        None
        if judgment.continuousValue is None
        else (
            judgment.continuousValue.coefficient,
            judgment.continuousValue.exponent,
        )
        for judgment in judgments
    ] == [(730, -3), (4, -2)]
    assert [row.continuous_value for row in flattened] == [
        Decimal("0.730"),
        Decimal("0.04"),
    ]
    assert [response.value for response in continuous] == [0.73, 0.04]
    assert all(response.scale.kind == "continuous" for response in continuous)
    assert all(response.relation == "entails" for response in continuous)


def test_unli_encoding_reads_the_typed_continuous_judgment_not_layer_metadata() -> None:
    codec = UnliCodec(specification())
    fragment = codec.decode(fixture_text("unli.tsv"))
    changed: list[FragmentRecord] = []
    replaced = False

    for record in fragment.records:
        model = model_of(record)

        if isinstance(model, judgment_records.JudgmentSet) and not replaced:
            held = model.judgments[0]
            exact = held.continuousValue

            if exact is None:
                msg = "the UNLI fixture judgment has no continuous response"
                raise ValueError(msg)

            model = judgment_records.JudgmentSet(
                agent=model.agent,
                createdAt=model.createdAt,
                experimentRef=model.experimentRef,
                features=model.features,
                judgments=(
                    judgment_records.Judgment(
                        item=held.item,
                        continuousValue=judgment_records.DecimalValue(
                            coefficient=731, exponent=exact.exponent
                        ),
                    ),
                ),
                knowledgeRefs=model.knowledgeRefs,
                metadata=model.metadata,
            )
            changed.append(
                FragmentRecord(
                    local_id=record.local_id,
                    nsid=record.nsid,
                    value_json=model.model_dump_json(),
                )
            )
            replaced = True

            continue

        changed.append(record)

    assert replaced
    assert "\t0.731\n" in codec.encode(changed)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "no header"),
        (
            fixture_text("unli.tsv").replace("UNLI label", "probability", 1),
            "exact release header",
        ),
        (
            fixture_text("unli.tsv").replace("0.730", "1.001", 1),
            "invalid UNLI probability",
        ),
    ],
)
def test_malformed_unli_sources_are_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedNliError, match=message):
        UnliCodec(specification()).decode(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "no pairs"),
        (INLINE.replace("entailment", "maybe"), "unknown gold label"),
        (
            fixture_text("chaosnli.jsonl").replace(
                '"label_count":100', '"label_count":99'
            ),
            "disagrees",
        ),
        (
            fixture_text("chaosnli.jsonl").replace('"c":33', '"c":-1'),
            "invalid count",
        ),
    ],
)
def test_malformed_sources_are_refused(source: str, message: str) -> None:
    codec: NliCodec = (
        ChaosNliCodec(specification())
        if "chaos-0" in source
        else SnliCodec(specification())
    )

    with pytest.raises(MalformedNliError, match=message):
        codec.decode(source)
