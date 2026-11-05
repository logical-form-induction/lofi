from __future__ import annotations

from typing import TYPE_CHECKING, Final

import pytest
from lairs.data import JudgmentStudy
from lairs.records import annotation as annotation_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records

from lofi.data import Example, Relational, RelationalScalar, Supervision
from lofi.data.corpora.layers import LayersCorpusReader
from lofi.data.corpora.megaattitude import (
    MalformedMegaAttitudeError,
    MegaAttitudeRelationSpec,
    MegaAttitudeSpec,
    MegaIntensionalityCodec,
    MegaNegRaisingCodec,
    MegaVeridicalityCodec,
)

if TYPE_CHECKING:
    from lairs.integrations.codecs import FragmentRecord

VERIDICALITY: Final = (
    "participant\tlist\tpresentationorder\tverb\tframe\tpolarity\tconditional\t"
    "sentence\tveridicality\tacceptability\tnativeenglish\n"
    "487\t0\t1\tsurmise\tNP Ved that S\tpositive\tFalse\t"
    "Someone surmised that a particular thing happened\tmaybe\t6\tTrue\n"
    "585\t32\t1\twant\tNP Ved to VP[+eventive]\tpositive\tFalse\t"
    "A particular person wanted to do a particular thing.\tyes\t7\tTrue\n"
    "585\t32\t2\topt\tNP Ved to VP[-eventive]\tpositive\tFalse\t"
    "A particular person opted to have a particular thing.\tno\t7\tTrue\n"
)
NEG_RAISING: Final = (
    "participant\tlist\tpresentationorder\tverb\tframe\ttense\tsubject\t"
    "sentence1\tsentence2\tnegraising\tacceptability\tnativeenglish\n"
    "229\t81\t8\tabhor\tNP V that S\tpast\tfirst\t"
    "I didn't abhor that a particular thing happened.\t"
    "I abhorred that that thing didn't happen.\t0.30\t0.97\tTrue\n"
)
INTENSIONALITY: Final = (
    "participant\tantecedent_verb\tantecedent_polarity\ttarget\t"
    "antecedent_frame\tconsequent_verb\tconsequent_embedded_tense\t"
    "antecedent_text\tconsequent_text\tresponse\tsanity\n"
    "70\tconsider\tpositive\tsubject\tNP V that S\twant\tpast\t"
    "A considered that C happened\tA wanted C to have happened\t0.780\t4\n"
)


def specification(release: str) -> MegaAttitudeSpec:
    return MegaAttitudeSpec(
        source_id="synthetic-megaattitude",
        release_id=release,
        language="en",
        source_method="elicited",
        spdx="CC-BY-SA-4.0",
    )


def model_of(
    record: FragmentRecord,
) -> (
    annotation_records.AnnotationLayer
    | expression_records.Expression
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet
):
    models = {
        "pub.layers.annotation.annotationLayer": annotation_records.AnnotationLayer,
        "pub.layers.expression.expression": expression_records.Expression,
        "pub.layers.judgment.experimentDef": judgment_records.ExperimentDef,
        "pub.layers.judgment.judgmentSet": judgment_records.JudgmentSet,
    }
    return models[record.nsid].model_validate_json(record.value_json)


def examples_of(
    records: tuple[FragmentRecord, ...],
) -> tuple[Example[Supervision[str]], ...]:
    study = JudgmentStudy.new()

    for record in records:
        study.add_record(record.local_id, model_of(record))

    return tuple(LayersCorpusReader(study).examples("train"))


def test_megaveridicality_generates_consequents_and_keeps_participants() -> None:
    codec = MegaVeridicalityCodec(specification("v2.1"))
    fragment = codec.decode(VERIDICALITY)
    examples = examples_of(fragment.records)
    responses = tuple(
        example.supervision
        for example in examples
        if isinstance(example.supervision, Relational)
    )
    expressions = {
        model.text
        for record in fragment.records
        if isinstance((model := model_of(record)), expression_records.Expression)
    }

    assert codec.encode(fragment.records) == VERIDICALITY
    assert expressions >= {
        "A particular thing happened.",
        "A particular person did a particular thing.",
        "A particular person had a particular thing.",
    }
    assert {(response.annotator, response.relation) for response in responses} == {
        ("487", "neutral"),
        ("585", "entails"),
        ("585", "contradicts"),
    }


def test_megaveridicality_uses_the_declared_relation_table() -> None:
    spec = MegaAttitudeSpec(
        source_id="synthetic-megaattitude",
        release_id="v2.1",
        language="en",
        source_method="elicited",
        spdx="CC-BY-SA-4.0",
        veridicality_relations=(
            MegaAttitudeRelationSpec(response="no", relation="entails"),
            MegaAttitudeRelationSpec(response="maybe", relation="contradicts"),
            MegaAttitudeRelationSpec(response="yes", relation="neutral"),
        ),
    )
    codec = MegaVeridicalityCodec(spec)
    fragment = codec.decode(VERIDICALITY)
    responses = tuple(
        (example.supervision.annotator, example.supervision.relation)
        for example in examples_of(fragment.records)
        if isinstance(example.supervision, Relational)
    )

    assert codec.encode(fragment.records) == VERIDICALITY
    assert ("487", "contradicts") in responses
    assert {relation for _, relation in responses} == {
        "contradicts",
        "neutral",
        "entails",
    }


def test_negraising_likelihood_round_trips_as_an_exact_bounded_pair() -> None:
    codec = MegaNegRaisingCodec(specification("v1.1"))
    fragment = codec.decode(NEG_RAISING)
    response = next(
        example.supervision
        for example in examples_of(fragment.records)
        if isinstance(example.supervision, RelationalScalar)
    )
    judgment = next(
        judgment
        for record in fragment.records
        if isinstance((model := model_of(record)), judgment_records.JudgmentSet)
        for judgment in model.judgments
    )

    assert codec.encode(fragment.records) == NEG_RAISING
    assert response.annotator == "229"
    assert response.value == 0.3
    assert response.relation == "entails"
    assert judgment.continuousValue is not None
    assert (
        judgment.continuousValue.coefficient,
        judgment.continuousValue.exponent,
    ) == (30, -2)


def test_intensionality_likelihood_keeps_the_released_pair_and_participant() -> None:
    codec = MegaIntensionalityCodec(specification("v1"))
    fragment = codec.decode(INTENSIONALITY)
    response = next(
        example.supervision
        for example in examples_of(fragment.records)
        if isinstance(example.supervision, RelationalScalar)
    )
    expressions = {
        model.text
        for record in fragment.records
        if isinstance((model := model_of(record)), expression_records.Expression)
    }

    assert codec.encode(fragment.records) == INTENSIONALITY
    assert expressions >= {
        "A considered that C happened",
        "A wanted C to have happened",
    }
    assert response.annotator == "70"
    assert response.value == 0.78


def test_megaveridicality_generates_the_for_clause_consequent() -> None:
    source = VERIDICALITY.splitlines(keepends=True)[0] + (
        "12\t4\t3\tarrange\tNP Ved for NP to VP\tpositive\tTrue\t"
        "Someone arranged for a particular thing to happen.\tyes\t5\tTrue\n"
    )
    codec = MegaVeridicalityCodec(specification("v2.1"))
    fragment = codec.decode(source)
    expressions = {
        model.text
        for record in fragment.records
        if isinstance((model := model_of(record)), expression_records.Expression)
    }

    assert codec.encode(fragment.records) == source
    assert "A particular thing happened." in expressions


def test_the_consequent_is_the_clause_after_the_matrix_complementizer() -> None:
    source = VERIDICALITY.splitlines(keepends=True)[0] + (
        "12\t4\t3\ttell\tNP was Ved that S\tpositive\tFalse\t"
        "Someone was told that a particular thing that mattered happened.\t"
        "yes\t5\tTrue\n"
    )
    codec = MegaVeridicalityCodec(specification("v2.1"))
    fragment = codec.decode(source)
    expressions = {
        model.text
        for record in fragment.records
        if isinstance((model := model_of(record)), expression_records.Expression)
    }

    assert codec.encode(fragment.records) == source
    assert expressions == {
        "Someone was told that a particular thing that mattered happened.",
        "A particular thing that mattered happened.",
    }


def test_a_blank_veridicality_response_is_kept_without_a_relation() -> None:
    source = VERIDICALITY.replace("\tmaybe\t", "\t\t")
    codec = MegaVeridicalityCodec(specification("v2.1"))
    fragment = codec.decode(source)
    annotators = [
        example.supervision.annotator
        for example in examples_of(fragment.records)
        if isinstance(example.supervision, Relational)
    ]

    assert codec.encode(fragment.records) == source
    assert annotators == ["585", "585"]


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (VERIDICALITY.replace("\tmaybe\t", "\tperhaps\t"), "'perhaps'"),
        (VERIDICALITY.replace("\tsurmise\t", "\t\t"), "line 2"),
        (VERIDICALITY.replace("NP Ved that S", "NP Ved whether S"), "unsupported"),
        (VERIDICALITY.replace(" that a particular", " a particular"), "no ' that '"),
        (VERIDICALITY.replace("\t6\t", "\tsix\t"), "noninteger acceptability"),
        (VERIDICALITY.splitlines(keepends=True)[0], "no response rows"),
        (VERIDICALITY.replace("veridicality", "response", 1), "header"),
    ],
)
def test_a_malformed_veridicality_table_is_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedMegaAttitudeError, match=message):
        MegaVeridicalityCodec(specification("v2.1")).decode(source)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        (NEG_RAISING.replace("\t0.30\t", "\t\t"), "line 2"),
        (NEG_RAISING.replace("\t0.30\t", "\t1.30\t"), "closed unit scale"),
        (NEG_RAISING.replace("\t0.30\t", "\tNaN\t"), "nonfinite"),
        (NEG_RAISING.replace("\t0.30\t", "\thigh\t"), "invalid negraising"),
    ],
)
def test_a_malformed_likelihood_table_is_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedMegaAttitudeError, match=message):
        MegaNegRaisingCodec(specification("v1.1")).decode(source)
