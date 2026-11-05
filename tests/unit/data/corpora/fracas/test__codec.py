import json
from pathlib import Path
from typing import Final

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import segmentation as segmentation_records
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data import IsoCodec
from lofi.data.corpora.fracas import (
    FRACAS_NAME,
    FracasAnswer,
    FracasCodec,
    FracasFile,
    FracasProblem,
    FracasSection,
    MalformedFracasError,
)
from lofi.data.laws import CodecFixture, IsoFixture
from lofi.data.registration import declared_codec_tags
from lofi.spec.datasets import CorpusReleaseSpec

type Subject = CodecSubject[FracasFile, CorpusFragment, FragmentRecord]
type Codec = IsoCodec[FracasFile, CorpusFragment, FragmentRecord]

FIXTURES: Final = Path(__file__).with_name("fixtures")
LAYER_NSID: Final = "pub.layers.annotation.annotationLayer"
HEAD: Final = (
    '<?xml version="1.0" encoding="UTF-8"?>\n<fracas-problems>\n'
    '<comment class="section"> 1 GENERALIZED QUANTIFIERS </comment>\n'
    '<comment class="subsection"> 1.1 Conservativity </comment>\n'
)
TAIL: Final = "</fracas-problems>\n"
ONE: Final = (
    '<problem id="001" fracas_answer="yes">'
    '<p idx="1">An Italian sang.</p><q>Did an Italian sing?</q>'
    "<h>An Italian sang.</h><a>Yes</a></problem>\n"
)


def specification() -> CorpusReleaseSpec:
    return CorpusReleaseSpec(
        source_id="synthetic-fracas",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
        eprint_refs=("at://did:plc:test/pub.layers.eprint.eprint/fracas",),
    )


def document(*bodies: str) -> str:
    return HEAD + "".join(bodies) + TAIL


def fixture_text() -> str:
    return (FIXTURES / "sample.xml").read_text(encoding="utf-8")


def codec() -> FracasCodec:
    return FracasCodec(specification())


def parsed(source: str) -> FracasFile:
    read = codec()
    return read.iso.backward(read.decode(source))


def problem(identifier: str) -> FracasProblem:
    return next(
        held
        for held in parsed(fixture_text()).problems
        if held.problem_id == identifier
    )


def relation_layers(
    fragment: CorpusFragment,
) -> tuple[annotation_records.Annotation, ...]:
    return tuple(
        annotation_records.AnnotationLayer.model_validate_json(
            record.value_json
        ).annotations[0]
        for record in fragment.records
        if record.nsid == LAYER_NSID
    )


def features_of(annotation: annotation_records.Annotation) -> dict[str, str]:
    assert annotation.features is not None
    return {entry.key: entry.value for entry in annotation.features.entries}


def records_of(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    return fragment.records


class TestFracasCodec(CodecConformance):
    @pytest.fixture
    def subject(self) -> Subject:
        read: Codec = codec()
        sources = (fixture_text(), document(ONE))
        strategy = st.sampled_from((*sources, *(text.encode() for text in sources)))
        files = st.sampled_from(
            tuple(read.iso.backward(read.decode(text)) for text in sources)
        )
        return CodecSubject(
            IsoFixture(read.iso, files, files.map(read.iso.forward), 8),
            CodecFixture(read, strategy, records_of, 8),
        )


def test_the_codec_is_declared_under_the_fracas_tag() -> None:
    assert FRACAS_NAME in declared_codec_tags()
    assert codec().name == FRACAS_NAME


def segmentations(
    fragment: CorpusFragment,
) -> tuple[segmentation_records.Segmentation, ...]:
    return tuple(
        segmentation_records.Segmentation.model_validate_json(record.value_json)
        for record in fragment.records
        if record.nsid == "pub.layers.segmentation.segmentation"
    )


def test_a_premise_carries_utf_8_byte_spans_over_its_own_text() -> None:
    premise = "Some delegates at the caf\u00e9 finished the survey."
    tokens = next(
        held.tokenizations[0].tokens
        for held in segmentations(codec().decode(fixture_text()))
        if " ".join(token.text or "" for token in held.tokenizations[0].tokens)
        == premise
    )

    assert tokens[-1].textSpan is not None
    assert tokens[-1].textSpan.byteEnd == len(premise.encode())


def test_every_expression_takes_the_next_membership_ordinal() -> None:
    fragment = codec().decode(fixture_text())
    ordinals = tuple(
        corpus_records.Membership.model_validate_json(record.value_json).ordinal
        for record in fragment.records
        if record.nsid == "pub.layers.corpus.membership"
    )

    assert ordinals == tuple(range(len(ordinals)))


def test_a_multi_premise_problem_keeps_its_premises_as_an_ordered_sequence() -> None:
    held = problem("003")

    assert held.premises == (
        "No delegate finished the survey on time.",
        "Smith is a delegate.",
        "Smith finished the survey.",
    )


def test_a_multi_premise_problem_emits_one_argument_per_premise_in_order() -> None:
    fragment = codec().decode(fixture_text())
    annotation = next(
        held
        for held in relation_layers(fragment)
        if json.loads(features_of(held)["problem_json"])["problem_id"] == "003"
    )
    assert annotation.arguments is not None

    assert tuple(argument.role for argument in annotation.arguments) == (
        "premise-0",
        "premise-1",
        "premise-2",
        "hypothesis",
    )


def test_every_premise_of_a_multi_premise_problem_is_its_own_expression() -> None:
    fragment = codec().decode(document())
    single = codec().decode(fixture_text())
    expressions = tuple(
        record
        for record in single.records
        if record.nsid == "pub.layers.expression.expression"
    )

    assert not fragment.records[1:]
    assert len(expressions) == sum(
        len(held.premises) + (held.hypothesis is not None)
        for held in parsed(fixture_text()).problems
    )


def test_every_problem_carries_a_non_null_section_stratum() -> None:
    fragment = codec().decode(fixture_text())
    sections = tuple(
        features_of(annotation)["section"] for annotation in relation_layers(fragment)
    )

    assert sections
    assert all(FracasSection(section) for section in sections)


def test_the_section_stratum_is_folded_from_the_document_order_of_comments() -> None:
    held = problem("004")

    assert held.section is FracasSection.TEMPORAL
    assert held.subsection == "7.2 Temporal Adverbials"
    assert held.subsubsection == "7.2.1 Indexicals"


def test_a_problem_inherits_the_pattern_gloss_its_run_sits_under() -> None:
    assert problem("001").pattern == "Q As are Bs == Q As are As who are Bs"
    assert problem("002").pattern is None


def test_a_degenerate_problem_reads_with_no_hypothesis() -> None:
    held = problem("005")

    assert held.hypothesis is None
    assert held.question is None
    assert held.source_answer is None
    assert held.answer is FracasAnswer.UNDEF


def test_a_degenerate_problem_emits_premises_and_no_hypothesis_argument() -> None:
    fragment = codec().decode(fixture_text())
    annotation = next(
        held
        for held in relation_layers(fragment)
        if json.loads(features_of(held)["problem_json"])["problem_id"] == "005"
    )
    assert annotation.arguments is not None

    assert tuple(argument.role for argument in annotation.arguments) == ("premise-0",)


def test_a_qualified_answer_keeps_its_source_spelling_and_its_nonstandard_flag() -> (
    None
):
    held = problem("004")

    assert held.answer is FracasAnswer.YES
    assert held.source_answer == "Yes, on one reading"
    assert not held.standard


def test_a_section_numeral_is_read_off_the_released_heading() -> None:
    assert FracasSection.ANAPHORA.number == "3"
    assert FracasSection.TEMPORAL.number == "7"


UNKNOWN_ANSWER: Final = (
    '<problem id="001" fracas_answer="maybe">'
    '<p idx="1">A sang.</p><h>A sang.</h><a>Yes</a></problem>'
)
MISINDEXED: Final = (
    '<problem id="001" fracas_answer="yes">'
    '<p idx="2">A sang.</p><h>A sang.</h><a>Yes</a></problem>'
)
BLANK_PREMISE: Final = (
    '<problem id="001" fracas_answer="yes">'
    '<p idx="1"> </p><h>A sang.</h><a>Yes</a></problem>'
)
NO_PREMISE: Final = (
    '<problem id="001" fracas_answer="yes"><h>A sang.</h><a>Yes</a></problem>'
)
DISAGREEING: Final = (
    '<problem id="001" fracas_answer="yes">'
    '<p idx="1">A sang.</p><h>A sang.</h><a>No</a></problem>'
)
STANDARD_UNDEF: Final = (
    '<problem id="001" fracas_answer="undef"><p idx="1">A sang.</p></problem>'
)
ANSWERED_WITHOUT_HYPOTHESIS: Final = (
    '<problem id="001" fracas_answer="yes"><p idx="1">A sang.</p><a>Yes</a></problem>'
)
UNIDENTIFIED: Final = (
    '<problem fracas_answer="yes">'
    '<p idx="1">A sang.</p><h>A sang.</h><a>Yes</a></problem>'
)


@pytest.mark.parametrize(
    ("body", "message"),
    [
        (UNKNOWN_ANSWER, "declares answer"),
        (MISINDEXED, "is indexed"),
        (BLANK_PREMISE, "empty premise"),
        (NO_PREMISE, "no premise"),
        (DISAGREEING, "source answer reads"),
        (STANDARD_UNDEF, "answered as standard"),
        (ANSWERED_WITHOUT_HYPOTHESIS, "with no hypothesis"),
        (ONE + ONE, "repeats problem id"),
        (UNIDENTIFIED, "no identifier"),
    ],
)
def test_a_malformed_problem_is_refused(body: str, message: str) -> None:
    with pytest.raises(MalformedFracasError, match=message):
        codec().decode(document(body))


def test_a_problem_before_its_section_heading_is_refused() -> None:
    source = '<?xml version="1.0"?>\n<fracas-problems>\n' + ONE + TAIL

    with pytest.raises(MalformedFracasError, match="precedes its section heading"):
        codec().decode(source)


def test_a_new_section_drops_the_subsection_the_last_one_declared() -> None:
    source = (
        HEAD
        + ONE
        + '<comment class="section"> 2 PLURALS </comment>\n'
        + ONE.replace('id="001"', 'id="002"')
        + TAIL
    )

    with pytest.raises(MalformedFracasError, match="precedes its section heading"):
        codec().decode(source)


def test_an_unknown_section_heading_is_refused() -> None:
    source = (
        '<?xml version="1.0"?>\n<fracas-problems>\n'
        '<comment class="section"> 10 PRESUPPOSITION </comment>\n'
        '<comment class="subsection"> 10.1 Triggers </comment>\n' + ONE + TAIL
    )

    with pytest.raises(MalformedFracasError, match="section heading reads"):
        codec().decode(source)


def test_a_subsection_outside_its_section_is_refused() -> None:
    source = (
        '<?xml version="1.0"?>\n<fracas-problems>\n'
        '<comment class="section"> 1 GENERALIZED QUANTIFIERS </comment>\n'
        '<comment class="subsection"> 2.1 Conjoined Noun Phrases </comment>\n'
        + ONE
        + TAIL
    )

    with pytest.raises(MalformedFracasError, match="sits under subsection"):
        codec().decode(source)


def test_an_unknown_comment_class_is_refused() -> None:
    body = '<comment class="appendix"> A </comment>'

    with pytest.raises(MalformedFracasError, match="declares class"):
        codec().decode(document(body))


def test_an_unexpected_element_is_refused() -> None:
    with pytest.raises(MalformedFracasError, match="holds a 'gloss' element"):
        codec().decode(document("<gloss>x</gloss>"))


def test_a_document_with_another_root_is_refused() -> None:
    with pytest.raises(MalformedFracasError, match="rooted at"):
        codec().decode("<problems/>")


def test_text_that_is_not_xml_is_refused() -> None:
    with pytest.raises(MalformedFracasError, match="not well-formed XML"):
        codec().decode("not xml")


def test_bytes_that_are_not_utf_8_are_refused() -> None:
    with pytest.raises(MalformedFracasError, match="not UTF-8"):
        codec().decode(b"\xff\xfe")


def test_a_heading_dropped_in_place_has_no_spelling_and_is_refused() -> None:
    held = problem("004")
    source = FracasFile(
        (held, held._replace(problem_id="006", subsubsection=None)),
    )

    with pytest.raises(MalformedFracasError, match="drops the heading"):
        codec().iso.forward(source)


def test_a_fragment_from_another_codec_is_refused() -> None:
    fragment = CorpusFragment(records=(), source="snli")

    with pytest.raises(MalformedFracasError, match="received 'snli' records"):
        codec().iso.backward(fragment)


def test_a_fragment_holding_a_foreign_record_is_refused() -> None:
    fragment = CorpusFragment(
        records=(
            FragmentRecord(
                local_id="x", nsid="pub.layers.graph.graph", value_json="{}"
            ),
        ),
        source=FRACAS_NAME,
    )

    with pytest.raises(MalformedFracasError, match="fragment contains"):
        codec().iso.backward(fragment)


def rewritten(record: FragmentRecord, value: dict[str, object]) -> FragmentRecord:
    return FragmentRecord(
        local_id=record.local_id,
        nsid=record.nsid,
        value_json=json.dumps(value),
    )


def test_a_relation_layer_missing_codec_metadata_is_refused() -> None:
    fragment = codec().decode(document(ONE))
    others = tuple(record for record in fragment.records if record.nsid != LAYER_NSID)
    layer = next(record for record in fragment.records if record.nsid == LAYER_NSID)
    payload = json.loads(layer.value_json)
    payload["annotations"][0]["features"] = None
    stripped = (*others, rewritten(layer, payload))

    with pytest.raises(MalformedFracasError, match="lacks exact codec metadata"):
        codec().iso.backward(CorpusFragment(records=stripped, source=FRACAS_NAME))


def test_noncontiguous_problem_ordinals_are_refused() -> None:
    fragment = codec().decode(document(ONE))
    shifted = tuple(record for record in fragment.records if record.nsid != LAYER_NSID)
    layer = next(record for record in fragment.records if record.nsid == LAYER_NSID)
    payload = json.loads(layer.value_json)
    entries = payload["annotations"][0]["features"]["entries"]

    for entry in entries:
        if entry["key"] == "problem_ordinal":
            entry["value"] = "3"

    moved = rewritten(layer, payload)

    with pytest.raises(MalformedFracasError, match="not contiguous from zero"):
        codec().iso.backward(
            CorpusFragment(records=(*shifted, moved), source=FRACAS_NAME)
        )


def test_records_outside_the_canonical_form_are_refused() -> None:
    fragment = codec().decode(document(ONE))
    dropped = tuple(
        record
        for record in fragment.records
        if record.nsid != "pub.layers.segmentation.segmentation"
    )

    with pytest.raises(MalformedFracasError, match="canonical form"):
        codec().iso.backward(CorpusFragment(records=dropped, source=FRACAS_NAME))


def test_decoding_into_a_fragment_keeps_the_records_it_was_given_first() -> None:
    read = codec()
    first = read.decode(document(ONE))
    extended = read.decode(fixture_text(), into=first)

    assert extended.records[: len(first.records)] == first.records
