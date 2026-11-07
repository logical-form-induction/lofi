from pathlib import Path
from typing import Final

import pytest
from hypothesis import strategies as st
from lairs.data import JudgmentStudy
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import resource as resource_records
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data import RESPONSE_MISSING, Scalar
from lofi.data.corpora.layers import LayersCorpusReader
from lofi.data.corpora.megaacceptability import (
    NAME,
    MalformedMegaAcceptabilityError,
    MegaAcceptabilityCodec,
    MegaAcceptabilitySpec,
    MegaAcceptabilityTable,
)
from lofi.data.laws import CodecFixture, IsoFixture

type Subject = CodecSubject[MegaAcceptabilityTable, CorpusFragment, FragmentRecord]

FIXTURES: Final = Path(__file__).with_name("fixtures")
SAMPLE: Final = FIXTURES / "sample.tsv"
V1: Final = (
    "participant\tlist\tpresentationorder\tverb\tframe\tresponse\t"
    "nativeenglish\tsentence\n"
    "p01\t1\t1\tthink\tNP _ that S\t6\tyes\tDana thinks that Lee left.\n"
)
FRAMES: Final = {
    "NP _ that S": "at://did:plc:test/pub.layers.resource.template/that-s",
    "NP _ NP to VP": "at://did:plc:test/pub.layers.resource.template/np-to-vp",
    "NP _ to VP": "at://did:plc:test/pub.layers.resource.template/to-vp",
}
VERBS: Final = {
    verb: f"at://did:plc:test/pub.layers.resource.entry/{verb}"
    for verb in (
        "believe",
        "think",
        "know",
        "say",
        "want",
        "expect",
        "persuade",
        "promise",
        "try",
        "manage",
    )
}
EPRINT: Final = "at://did:plc:test/pub.layers.eprint.eprint/megaacceptability"


def specification() -> MegaAcceptabilitySpec:
    return MegaAcceptabilitySpec(
        source_id="synthetic-megaacceptability",
        release_id="v2",
        spdx="CC0-1.0",
        scale_min=1,
        scale_max=7,
        frame_refs=FRAMES,
        verb_refs=VERBS,
        eprint_refs=(EPRINT,),
    )


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    return fragment.records


def sample_text() -> str:
    return SAMPLE.read_text(encoding="utf-8")


class TestMegaAcceptabilityCodec(CodecConformance):
    @pytest.fixture
    def subject(self) -> Subject:
        codec = MegaAcceptabilityCodec(specification())
        sources = st.sampled_from((sample_text(), V1, sample_text().encode("utf-8")))
        tables = st.sampled_from(
            tuple(
                codec.iso.backward(codec.decode(source))
                for source in (sample_text(), V1)
            )
        )

        return CodecSubject(
            IsoFixture(codec.iso, tables, tables.map(codec.iso.forward), 8),
            CodecFixture(codec, sources, records_of, 8),
        )


def model_of(
    record: FragmentRecord,
) -> (
    expression_records.Expression
    | resource_records.Filling
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet
):
    models = {
        "pub.layers.expression.expression": expression_records.Expression,
        "pub.layers.resource.filling": resource_records.Filling,
        "pub.layers.judgment.experimentDef": judgment_records.ExperimentDef,
        "pub.layers.judgment.judgmentSet": judgment_records.JudgmentSet,
    }
    return models[record.nsid].model_validate_json(record.value_json)


def test_fifty_rows_round_trip_in_exact_source_order() -> None:
    codec = MegaAcceptabilityCodec(specification())
    source = sample_text()
    fragment = codec.decode(source)

    assert len(source.splitlines()) == 51
    assert codec.name == NAME
    assert codec.encode(fragment.records) == source


def test_v1_and_v2_headers_remain_distinct() -> None:
    codec = MegaAcceptabilityCodec(specification())

    assert codec.encode(codec.decode(V1).records) == V1
    assert (
        codec.encode(codec.decode(sample_text()).records)
        .splitlines()[0]
        .endswith("\tversion")
    )


def test_declared_metadata_and_external_resource_refs_survive() -> None:
    fragment = MegaAcceptabilityCodec(specification()).decode(sample_text())
    models = tuple(model_of(record) for record in fragment.records)
    experiment = next(
        model for model in models if isinstance(model, judgment_records.ExperimentDef)
    )
    fillings = tuple(
        model for model in models if isinstance(model, resource_records.Filling)
    )

    assert experiment.scaleMin == 1
    assert experiment.scaleMax == 7
    assert experiment.eprintRefs == (EPRINT,)
    assert experiment.licensing is not None
    assert experiment.licensing.expression == "CC0-1.0"
    assert set(experiment.templateRefs or ()) == set(FRAMES.values())
    assert {filling.templateRef for filling in fillings} == set(FRAMES.values())
    assert {filling.slotFillings[0].entryRef for filling in fillings} == set(
        VERBS.values()
    )


def test_responses_remain_per_participant_without_aggregation() -> None:
    fragment = MegaAcceptabilityCodec(specification()).decode(sample_text())
    response_sets = tuple(
        model
        for model in map(model_of, fragment.records)
        if isinstance(model, judgment_records.JudgmentSet)
    )

    assert len(response_sets) == 5
    assert {
        response_set.agent.id for response_set in response_sets if response_set.agent
    } == {
        "p01",
        "p02",
        "p03",
        "p04",
        "p05",
    }
    assert sum(len(response_set.judgments) for response_set in response_sets) == 50
    missing = tuple(
        judgment
        for response_set in response_sets
        for judgment in response_set.judgments
        if judgment.scalarValue is None
    )
    assert len(missing) == 1
    assert missing[0].features is not None
    assert {(entry.key, entry.value) for entry in missing[0].features.entries} >= {
        ("response_na", "true")
    }


def test_layers_reader_preserves_shared_items_agents_and_missingness() -> None:
    fragment = MegaAcceptabilityCodec(specification()).decode(sample_text())
    experiment_record = next(
        record
        for record in fragment.records
        if record.nsid == "pub.layers.judgment.experimentDef"
    )
    study = JudgmentStudy.new(experiment_record.local_id)

    for record in fragment.records:
        study.add_record(record.local_id, model_of(record))

    examples = tuple(LayersCorpusReader(study).examples("train"))
    responses = tuple(
        example.supervision
        for example in examples
        if isinstance(example.supervision, Scalar)
    )

    assert len(examples) == 50
    assert len(responses) == 50
    assert len({example.item.item_id for example in examples}) == 10
    assert {response.annotator for response in responses} == {
        "p01",
        "p02",
        "p03",
        "p04",
        "p05",
    }
    missing = tuple(response for response in responses if response.value is None)
    assert len(missing) == 1
    assert missing[0].features == frozenset({RESPONSE_MISSING})


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("", "no header"),
        ("participant\tresponse\np01\t7\n", "unsupported header"),
        (V1.replace("\t6\t", "\t8\t"), "outside the declared scale"),
        (V1.replace("\t6\t", "\t\t"), "noninteger response ''"),
        (sample_text().replace("\t7.0\t", "\tNA\t", 1), "v2 response 'NA'"),
        (sample_text().replace("\t7.0\t", "\t7\t", 1), "v2 response '7'"),
    ],
)
def test_malformed_tables_are_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedMegaAcceptabilityError, match=message):
        MegaAcceptabilityCodec(specification()).decode(source)


def test_a_participant_may_rate_one_stimulus_twice() -> None:
    repeated = V1 + V1.splitlines(keepends=True)[1].replace("\t1\t6\t", "\t2\t3\t")
    codec = MegaAcceptabilityCodec(specification())
    fragment = codec.decode(repeated)
    judgments = tuple(
        judgment
        for record in fragment.records
        if record.nsid == "pub.layers.judgment.judgmentSet"
        for judgment in judgment_records.JudgmentSet.model_validate_json(
            record.value_json
        ).judgments
    )

    assert codec.encode(fragment.records) == repeated
    assert len(judgments) == 2
    assert len({judgment.item.recordRef for judgment in judgments}) == 1
