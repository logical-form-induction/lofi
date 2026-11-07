from pathlib import Path
from typing import Final

import pytest
from hypothesis import strategies as st
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation as annotation_records
from lairs.records import expression as expression_records
from lairs.records import segmentation as segmentation_records
from tests.conformance.codec import CodecConformance, CodecSubject

from lofi.data.corpora.pmb import (
    MalformedPmbCcgError,
    PmbCcgCodec,
    PmbFile,
    PmbLeaf,
    PmbTree,
)
from lofi.data.laws import CodecFixture, IsoFixture

type Subject = CodecSubject[PmbFile, CorpusFragment, FragmentRecord]
type LayerRecord = (
    annotation_records.AnnotationLayer
    | expression_records.Expression
    | segmentation_records.Segmentation
)

FIXTURES: Final = Path(__file__).with_name("fixtures")
ALL_RULES: Final = FIXTURES / "all_rules.ccg"
RULES: Final = (
    "fa",
    "ba",
    "rp",
    "lx",
    "bxc",
    "conj",
    "fc",
    "fxc",
    "gbxc",
    "lp",
    "bc",
    "gfc",
    "gbc",
)
SAMPLE: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(1,
 ba(s:dcl,
  lx(np,n,
   t(n,'äußerst',[lemma:'äußerst',from:0,to:7,pos:'RB',sem:'MNR',wordnet:'extremely.r.01',verbnet:['Theme','Agent']])),
  fa(s:dcl\np,
   t((s:dcl\np)/np,'praktiziert',[lemma:practice,from:8,to:19,pos:'VBZ',sem:'ENS',wordnet:'practice.v.03',verbnet:['Agent','Theme']]),
   t(np,'Kunst',[lemma:'kunst',from:20,to:25,pos:'NN',sem:'CON',wordnet:'art.n.01',verbnet:[]])))).
"""
PUNCTUATION: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(punctuation,
 t(np,'[',[lemma:'[',from:0,to:1,pos:'PUNCT',sem:'SYM',wordnet:'left_bracket.n.01',verbnet:[]])).
"""
BACKSLASH: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(backslash,
 t(np,'\\',[lemma:'a\\b',from:0,to:1,pos:'PUNCT',sem:'SYM',wordnet:'backslash.n.01',verbnet:[]])).
"""
JOINED: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(joined,
 rp(np,
  lx(np,n,
   t(n,'Washington~DC',[lemma:'washington~dc',from:20,to:34,pos:'NNP',sem:'GPE',wordnet:'city.n.01'])),
  t(.,'.',[lemma:'.',from:34,to:35,pos:'.',sem:'NIL',wordnet:'O']))).
"""
JOINED_TRIPLE: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(triple,
 t(n,'Washington~D.~C.',[lemma:'washington~d.~c.',from:18,to:35,pos:'NNP',sem:'GPE',wordnet:'city.n.01'])).
"""
BARE_STOPS: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(first,
 t(.,'.',[lemma:'.',from:0,to:1,pos:'.',sem:'NIL',wordnet:'O'])).
ccg(second,
 t(,,',',[lemma:',',from:0,to:1,pos:',',sem:'NIL',wordnet:'O'])).
"""
QUOTED_STOPS: Final = r""":- op(601, xfx, (/)).
:- op(601, xfx, (\)).
ccg(first,
 t('.','.',[lemma:'.',from:0,to:1,pos:'.',sem:'NIL',wordnet:'O'])).
ccg(second,
 t(',',',',[lemma:',',from:0,to:1,pos:',',sem:'NIL',wordnet:'O'])).
"""
# The two declarations every file in PMB 4.0.0 writes under the slash operators.
DATABASE_DIRECTIVES: Final = """:- multifile ccg/2, id/2.
:- discontiguous ccg/2, id/2.
"""
# PMB 4.0.0 English gold, p97/d2625, whose raw text is "He works at a pawn shop."
PAWNSHOP: Final = (FIXTURES / "pawnshop.tags").read_text(encoding="utf-8")


def _fixture_text() -> str:
    """Return the complete synthetic rule fixture."""
    return ALL_RULES.read_text(encoding="utf-8")


def _records(fragment: CorpusFragment) -> tuple[FragmentRecord, ...]:
    """Return the records in a corpus fragment.

    Parameters
    ----------
    fragment
        Corpus fragment under test.
    """
    return fragment.records


def _model(record: FragmentRecord) -> LayerRecord:
    """Return the Layers model for one fragment record.

    Parameters
    ----------
    record
        Fragment record to decode.
    """
    models = {
        "pub.layers.annotation.annotationLayer": annotation_records.AnnotationLayer,
        "pub.layers.expression.expression": expression_records.Expression,
        "pub.layers.segmentation.segmentation": segmentation_records.Segmentation,
    }
    return models[record.nsid].model_validate_json(record.value_json)


def _leaves(node: PmbLeaf | PmbTree) -> tuple[PmbLeaf, ...]:
    """Return the lexical nodes under a derivation node, in surface order.

    Parameters
    ----------
    node
        Derivation node to walk.
    """
    if isinstance(node, PmbLeaf):
        return (node,)

    return tuple(leaf for child in node.children for leaf in _leaves(child))


def _type_references(source: str) -> tuple[str | None, ...]:
    """Return the ontology reference of every annotation decoded from a source.

    Parameters
    ----------
    source
        PMB CCG text to decode.
    """
    return tuple(
        annotation.ontologyTypeRef
        for record in PmbCcgCodec().decode(source).records
        if isinstance((model := _model(record)), annotation_records.AnnotationLayer)
        for annotation in model.annotations
    )


class TestPmbCcgCodec(CodecConformance):
    @pytest.fixture
    def subject(self) -> Subject:
        """Return the codec conformance subject."""
        codec = PmbCcgCodec()
        sources = (SAMPLE, PAWNSHOP, _fixture_text(), SAMPLE.encode("utf-8"))
        files = tuple(codec.iso.backward(codec.decode(source)) for source in sources)
        return CodecSubject(
            IsoFixture(
                codec.iso,
                st.sampled_from(files),
                st.sampled_from(tuple(codec.iso.forward(file) for file in files)),
                8,
            ),
            CodecFixture(codec, st.sampled_from(sources), _records, 8),
        )


def test_slash_operators_use_the_declared_precedence() -> None:
    parsed = PmbCcgCodec().iso.backward(PmbCcgCodec().decode(SAMPLE))
    root = parsed[0].tree

    assert isinstance(root, PmbTree)
    assert root.category == "s:dcl"
    right = root.children[1]
    assert isinstance(right, PmbTree)
    assert right.category == r"s:dcl\np"
    verb = right.children[0]
    assert isinstance(verb, PmbLeaf)
    assert verb.category == r"(s:dcl\np)/np"


@pytest.mark.parametrize("category", ["s/np/np", r"s/np\np"])
def test_unparenthesized_xfx_chains_are_refused(category: str) -> None:
    source = SAMPLE.replace(r"(s:dcl\np)/np", category)

    with pytest.raises(MalformedPmbCcgError, match="xfx operator"):
        PmbCcgCodec().decode(source)


@pytest.mark.parametrize("category", ["(s/np)/np", "s/(np/np)"])
def test_parenthesized_xfx_chains_are_accepted(category: str) -> None:
    source = SAMPLE.replace(r"(s:dcl\np)/np", category)
    parsed = PmbCcgCodec().iso.backward(PmbCcgCodec().decode(source))
    root = parsed[0].tree

    assert isinstance(root, PmbTree)
    right = root.children[1]
    assert isinstance(right, PmbTree)
    verb = right.children[0]
    assert isinstance(verb, PmbLeaf)
    assert verb.category == category


def test_character_offsets_become_utf8_byte_offsets() -> None:
    fragment = PmbCcgCodec().decode(SAMPLE)
    segmentation = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), segmentation_records.Segmentation)
    )
    tokens = segmentation.tokenizations[0].tokens

    assert tuple(token.text for token in tokens) == ("äußerst", "praktiziert", "Kunst")
    assert tuple(
        (token.textSpan.byteStart, token.textSpan.byteEnd)
        for token in tokens
        if token.textSpan is not None
    ) == ((0, 9), (10, 21), (22, 27))


def test_leaf_metadata_survives_the_layers_translation() -> None:
    fragment = PmbCcgCodec().decode(SAMPLE)
    layer = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), annotation_records.AnnotationLayer)
    )
    leaf = next(
        annotation
        for annotation in layer.annotations
        if annotation.anchor is not None
        and annotation.anchor.tokenRef is not None
        and annotation.anchor.tokenRef.tokenIndex == 0
    )

    assert leaf.features is not None
    assert {entry.key: entry.value for entry in leaf.features.entries} == {
        "lemma": "äußerst",
        "pos": "RB",
        "sem": "MNR",
        "wordnet": "extremely.r.01",
        "verbnet": "['Theme','Agent']",
    }


def test_every_source_rule_symbol_remains_a_distinct_label() -> None:
    fragment = PmbCcgCodec().decode(_fixture_text())
    layers = tuple(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), annotation_records.AnnotationLayer)
    )
    labels = tuple(
        annotation.label
        for layer in layers
        for annotation in layer.annotations
        if annotation.label is not None
    )

    assert labels == RULES
    assert len(labels) == len(set(labels))
    assert {"rp", "lp", "lx"} <= set(labels)


def test_decoding_then_encoding_preserves_the_parsed_value() -> None:
    codec = PmbCcgCodec()
    parsed = codec.iso.backward(codec.decode(SAMPLE))
    returned = codec.iso.backward(
        codec.decode(codec.encode(codec.decode(SAMPLE).records))
    )

    assert returned == parsed


def test_a_bare_punctuation_category_reads_as_its_quoted_spelling() -> None:
    codec = PmbCcgCodec()
    bare = codec.iso.backward(codec.decode(BARE_STOPS))
    quoted = codec.iso.backward(codec.decode(QUOTED_STOPS))

    assert tuple(derivation.identifier for derivation in bare) == ("first", "second")
    assert bare == quoted
    assert tuple(
        derivation.tree.category
        for derivation in bare
        if isinstance(derivation.tree, PmbLeaf)
    ) == ("'.'", "','")


def test_a_bare_punctuation_category_interns_as_its_quoted_spelling_does() -> None:
    references = _type_references(BARE_STOPS)

    assert references == _type_references(QUOTED_STOPS)
    assert len(set(references)) == 2


def test_a_joined_token_reads_as_one_token_spanning_the_raw_text() -> None:
    codec = PmbCcgCodec()
    derivation = codec.iso.backward(codec.decode(JOINED))[0]
    tree = derivation.tree

    assert isinstance(tree, PmbTree)
    place = tree.children[0]
    assert isinstance(place, PmbTree)
    leaf = place.children[0]
    assert isinstance(leaf, PmbLeaf)
    assert leaf.text == "Washington~DC"
    assert (leaf.start, leaf.end) == (20, 34)
    assert leaf.surface == "Washington  DC"
    assert leaf.surface != leaf.text


def test_a_joined_token_fills_its_span_in_the_expression() -> None:
    fragment = PmbCcgCodec().decode(JOINED)
    expression = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), expression_records.Expression)
    )
    segmentation = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), segmentation_records.Segmentation)
    )
    tokens = segmentation.tokenizations[0].tokens

    assert expression.text is not None
    assert len(expression.text) == 35
    assert expression.text[20:34] == "Washington  DC"
    assert tuple(token.text for token in tokens) == ("Washington~DC", ".")
    assert tokens[0].textSpan is not None
    assert (tokens[0].textSpan.byteStart, tokens[0].textSpan.byteEnd) == (20, 34)


def test_a_joined_token_spreads_every_separator_across_its_span() -> None:
    codec = PmbCcgCodec()
    leaf = codec.iso.backward(codec.decode(JOINED_TRIPLE))[0].tree

    assert isinstance(leaf, PmbLeaf)
    assert leaf.text == "Washington~D.~C."
    assert leaf.surface == "Washington  D. C."
    assert len(leaf.surface) == leaf.end - leaf.start


@pytest.mark.parametrize("source", [BARE_STOPS, JOINED, JOINED_TRIPLE, PAWNSHOP])
def test_pmb_spellings_survive_a_decode_encode_decode(source: str) -> None:
    codec = PmbCcgCodec()
    parsed = codec.iso.backward(codec.decode(source))
    returned = codec.iso.backward(
        codec.decode(codec.encode(codec.decode(source).records))
    )

    assert returned == parsed


@pytest.mark.parametrize("span", ["to:32", "to:31"])
def test_a_joined_token_with_no_room_for_a_separator_is_refused(span: str) -> None:
    source = JOINED.replace("to:34", span)

    with pytest.raises(MalformedPmbCcgError, match="overruns"):
        PmbCcgCodec().decode(source)


@pytest.mark.parametrize("source", [PUNCTUATION, BACKSLASH])
def test_quoted_punctuation_metadata_round_trips(source: str) -> None:
    codec = PmbCcgCodec()
    parsed = codec.iso.backward(codec.decode(source))
    returned = codec.iso.backward(
        codec.decode(codec.encode(codec.decode(source).records))
    )

    assert returned == parsed


def test_encoding_distinguishes_atom_and_list_metadata() -> None:
    codec = PmbCcgCodec()
    encoded = codec.encode(codec.decode(PUNCTUATION).records)

    assert "lemma:'['" in encoded
    assert "verbnet:[]" in encoded
    assert "lemma:['[']" not in encoded
    assert "verbnet:'[]'" not in encoded


def test_decoding_into_a_fragment_keeps_existing_records_first() -> None:
    codec = PmbCcgCodec()
    first = codec.decode(SAMPLE)
    extended = codec.decode(_fixture_text(), into=first)

    assert extended.records[: len(first.records)] == first.records
    assert len(extended.records) == len(first.records) + len(RULES) * len(first.records)


@pytest.mark.parametrize(
    ("source", "message"),
    [
        ("ccg(1,t(np,dog,[from:0,to:3])).", "does not declare"),
        (
            SAMPLE.replace("op(601, xfx, (/))", "op(600, xfx, (/))"),
            "supported directive",
        ),
        (
            DATABASE_DIRECTIVES.replace("multifile", "dynamic") + SAMPLE,
            "supported directive",
        ),
        (
            DATABASE_DIRECTIVES.replace("ccg/2, id/2", "ccg/2 id/2") + SAMPLE,
            "supported directive",
        ),
        (SAMPLE.replace("to:7", "to:6"), "overruns"),
        (SAMPLE.replace("from:8,to:19", "from:19,to:8"), "ends before it starts"),
        (SAMPLE.replace("ccg(1", "term(1"), "not a ccg term"),
    ],
)
def test_malformed_sources_are_refused(source: str, message: str) -> None:
    with pytest.raises(MalformedPmbCcgError, match=message):
        PmbCcgCodec().decode(source)


def test_the_declarations_every_released_file_carries_are_read() -> None:
    source = SAMPLE.replace("ccg(1,", DATABASE_DIRECTIVES + "ccg(1,")

    assert PmbCcgCodec().decode(source).records == PmbCcgCodec().decode(SAMPLE).records


def test_a_database_declaration_may_precede_the_operator_declarations() -> None:
    source = DATABASE_DIRECTIVES + SAMPLE

    assert PmbCcgCodec().decode(source).records == PmbCcgCodec().decode(SAMPLE).records


def test_a_normalized_token_reads_as_one_token_spanning_the_raw_text() -> None:
    codec = PmbCcgCodec()
    leaves = _leaves(codec.iso.backward(codec.decode(PAWNSHOP))[0].tree)
    shop = leaves[-2]

    assert tuple(leaf.text for leaf in leaves) == (
        "He",
        "works",
        "at",
        "a",
        "pawnshop",
        ".",
    )
    assert (shop.start, shop.end) == (14, 23)
    assert shop.surface == "pawnshop "
    assert shop.surface != shop.text


def test_a_normalized_token_fills_its_span_in_the_expression() -> None:
    fragment = PmbCcgCodec().decode(PAWNSHOP)
    expression = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), expression_records.Expression)
    )
    segmentation = next(
        model
        for record in fragment.records
        if isinstance((model := _model(record)), segmentation_records.Segmentation)
    )
    shop = segmentation.tokenizations[0].tokens[-2]

    assert expression.text == "He works at a pawnshop ."
    assert shop.text == "pawnshop"
    assert shop.textSpan is not None
    assert (shop.textSpan.byteStart, shop.textSpan.byteEnd) == (14, 23)


def test_a_spelling_wider_than_its_span_is_refused() -> None:
    source = PAWNSHOP.replace("from:14, to:23", "from:14, to:21")

    with pytest.raises(MalformedPmbCcgError, match="overruns"):
        PmbCcgCodec().decode(source)
