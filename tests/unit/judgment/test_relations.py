from collections import Counter
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from lairs import Corpus
from lairs.records import annotation as annotation_records
from lairs.records import corpus as corpus_records
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import segmentation as segmentation_records
from tests.conformance.judgment_source import (
    JudgmentSourceSubject,
    JudgmentSourceSuite,
)

import lofi.judgment.relations as relation_module
from lofi.data import Example, Relational
from lofi.data.corpora.layers import LayersCorpusReader
from lofi.data.corpora.nli import ChaosNliCodec
from lofi.judgment import (
    AnonymousParticipantPolicy,
    AttributedItem,
    DesignError,
    ItemFactors,
    JudgmentSkip,
    JudgmentSource,
    RelationSource,
    design_of,
    related_item_key,
)
from lofi.spec.datasets import CorpusReleaseSpec


def test_the_participant_policy_has_one_public_route() -> None:
    assert "AnonymousParticipantPolicy" not in vars(relation_module)


if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from lairs.integrations.codecs import FragmentRecord

    from lofi.data import Supervision

CHAOS: Final = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "corpora"
    / "nli"
    / "fixtures"
    / "chaosnli.jsonl"
)
INVENTORY: Final = ("contradicts", "neutral", "entails")


class Sentence(NamedTuple):
    item_id: str
    languages: tuple[str, ...]


class Entailment(NamedTuple):
    other: str
    relation: str
    annotator: str | None


class Related(NamedTuple):
    item: Sentence
    supervision: Entailment


PREMISE: Final = Sentence("premise-1", ("en",))
OTHER_PREMISE: Final = Sentence("premise-2", ("en",))


def unattributed(
    counts: dict[str, int], premise: Sentence = PREMISE
) -> tuple[Related, ...]:
    return tuple(
        Related(premise, Entailment("hypothesis-1", relation, None))
        for relation, count in counts.items()
        for _ in range(count)
    )


def attributed(
    counts: dict[str, int], premise: Sentence = PREMISE
) -> tuple[Related, ...]:
    return tuple(
        Related(premise, Entailment("hypothesis-1", relation, f"p{position}"))
        for position, relation in enumerate(
            relation for relation, count in counts.items() for _ in range(count)
        )
    )


def test_a_relation_source_answers_the_judgment_source_port() -> None:
    assert isinstance(RelationSource((), inventory=INVENTORY), JudgmentSource)


def test_a_related_item_key_joins_the_related_items_in_order() -> None:
    assert related_item_key(("item-1", "item-2")) == "item-1 -> item-2"
    assert related_item_key(("a", "b", "c")) == "a -> b -> c"
    with pytest.raises(ValueError, match="at least two"):
        related_item_key(("a",))


def test_a_response_is_the_relation_position_in_the_declared_inventory() -> None:
    related = (
        Related(PREMISE, Entailment("hypothesis-1", "entails", "p1")),
        Related(PREMISE, Entailment("hypothesis-1", "contradicts", "p2")),
        Related(PREMISE, Entailment("hypothesis-1", "neutral", "p3")),
    )

    source = RelationSource(related, inventory=INVENTORY)
    design = design_of(source)

    assert source.inventory == INVENTORY
    assert source.response_levels == 3
    assert design.responses.tolist() == [2, 0, 1]
    assert design.item_levels == (related_item_key(("premise-1", "hypothesis-1")),)
    assert design.participant_levels == ("p1", "p2", "p3")


def test_a_hundred_attributed_relations_stay_a_hundred_rows_of_one_pair() -> None:
    source = RelationSource(
        attributed({"entails": 34, "contradicts": 33, "neutral": 33}),
        inventory=INVENTORY,
    )

    design = design_of(source)

    assert design.n_rows == 100
    assert source.skips == {}
    assert design.item_levels == (related_item_key(("premise-1", "hypothesis-1")),)
    assert len(design.participant_levels) == 100
    assert Counter(design.responses.tolist()) == {2: 34, 0: 33, 1: 33}


def test_an_unattributed_relation_is_counted_and_becomes_no_row() -> None:
    related = (
        *attributed({"entails": 2}),
        *unattributed({"neutral": 3}),
    )

    source = RelationSource(related, inventory=INVENTORY)
    design = design_of(source)

    assert design.n_rows == 2
    assert source.skips == {JudgmentSkip.MISSING_PARTICIPANT: 3}
    assert design.participant_levels == ("p0", "p1")
    assert design.responses.tolist() == [2, 2]
    assert source.participant_policy is AnonymousParticipantPolicy.SKIP
    assert source.singleton_participants == 0


def test_each_unattributed_occurrence_can_receive_a_singleton_participant() -> None:
    occurrence = Related(PREMISE, Entailment("hypothesis-1", "entails", None))
    source = RelationSource(
        (occurrence, occurrence),
        inventory=INVENTORY,
        participant_policy=AnonymousParticipantPolicy.SINGLETON,
    )
    design = design_of(source)

    assert source.skips == {}
    assert source.participant_policy is AnonymousParticipantPolicy.SINGLETON
    assert source.singleton_participants == 2
    assert design.n_rows == 2
    assert len(design.participant_levels) == 2
    assert design.participants.tolist() == [0, 1]


def test_singleton_participants_do_not_collide_with_attributed_participants() -> None:
    related = (
        Related(PREMISE, Entailment("hypothesis-1", "entails", "anonymous-0")),
        Related(PREMISE, Entailment("hypothesis-1", "entails", None)),
    )
    source = RelationSource(
        related,
        inventory=INVENTORY,
        participant_policy=AnonymousParticipantPolicy.SINGLETON,
    )
    rows = source.rows()

    assert rows[0].participant == "anonymous-0"
    assert rows[1].participant != rows[0].participant


def test_attributed_relations_have_identical_rows_under_both_policies() -> None:
    related = attributed({"entails": 2, "neutral": 1})
    skipped = RelationSource(
        related,
        inventory=INVENTORY,
        participant_policy=AnonymousParticipantPolicy.SKIP,
    )
    singleton = RelationSource(
        related,
        inventory=INVENTORY,
        participant_policy=AnonymousParticipantPolicy.SINGLETON,
    )

    assert singleton.rows() == skipped.rows()
    assert singleton.skips == skipped.skips == {}
    assert singleton.singleton_participants == skipped.singleton_participants == 0


def test_a_relation_source_records_no_factor_it_was_not_declared() -> None:
    related = (
        *attributed({"entails": 2}),
        *attributed({"neutral": 1}, Sentence("premise-3", ("cmn",))),
    )

    source = RelationSource(related, inventory=INVENTORY)

    assert source.factors == ()
    assert dict(design_of(source).factor_levels) == {}


def test_declared_factors_are_read_from_the_related_item_record() -> None:
    pair = related_item_key(("premise-1", "hypothesis-1"))
    source = RelationSource(
        (*attributed({"entails": 2}), *unattributed({"neutral": 1})),
        inventory=INVENTORY,
        factors=ItemFactors(
            {"genre": "item['genre']"},
            (AttributedItem(pair, {"genre": "fiction"}, {}),),
        ),
    )

    design = design_of(source)

    assert source.factors == ("genre",)
    assert design.factor_levels["genre"] == ("fiction",)
    assert repr(source) == (
        "RelationSource(rows=2, inventory=('contradicts', 'neutral', 'entails'), "
        "factors=('genre',), participant_policy='skip', "
        "singleton_participants=0, skipped=1)"
    )


def test_a_related_key_without_a_record_is_refused_when_factors_are_declared() -> None:
    with pytest.raises(DesignError, match="'premise-1 -> hypothesis-1', so it has"):
        RelationSource(
            attributed({"entails": 1}),
            inventory=INVENTORY,
            factors=ItemFactors({"genre": "item['genre']"}, ()),
        )


def test_a_relation_outside_the_inventory_is_refused() -> None:
    with pytest.raises(DesignError, match="'maybe' is outside the inventory"):
        RelationSource(unattributed({"maybe": 1}), inventory=INVENTORY)


@pytest.mark.parametrize(
    ("inventory", "message"),
    [
        (("entails",), "at least two relations"),
        (("entails", "entails"), "repeats a relation"),
        (("entails", ""), "relation name is empty"),
    ],
)
def test_a_malformed_inventory_is_refused(
    inventory: tuple[str, ...], message: str
) -> None:
    with pytest.raises((DesignError, ValueError), match=message):
        RelationSource((), inventory=inventory)


type LayerModel = (
    annotation_records.AnnotationLayer
    | corpus_records.Corpus
    | corpus_records.Membership
    | expression_records.Expression
    | judgment_records.ExperimentDef
    | judgment_records.JudgmentSet
    | segmentation_records.Segmentation
)


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


class Fixture(NamedTuple):
    relations: tuple[Example[Relational], ...]
    counts: Counter[str]


def relational_examples(
    examples: Iterable[Example[Supervision[str]]],
) -> Iterator[Example[Relational]]:
    for example in examples:
        if isinstance(example.supervision, Relational):
            yield Example(example.item, example.supervision)


def chaos_collection() -> Fixture:
    spec = CorpusReleaseSpec(
        source_id="synthetic-nli",
        release_id="v1",
        split="test",
        language="en",
        source_method="manual",
        spdx="CC0-1.0",
    )
    fragment = ChaosNliCodec(spec).decode(CHAOS.read_text(encoding="utf-8"))
    corpus_record = next(
        record
        for record in fragment.records
        if record.nsid == "pub.layers.corpus.corpus"
    )
    corpus = Corpus(uri=corpus_record.local_id)

    for record in fragment.records:
        corpus.add_record(record.local_id, model_of(record))

    relations = tuple(relational_examples(LayersCorpusReader(corpus).examples("test")))

    return Fixture(
        relations, Counter(related.supervision.relation for related in relations)
    )


def test_the_reader_examples_of_the_chaos_fixture_are_counted_not_attributed() -> None:
    fixture = chaos_collection()
    source = RelationSource(fixture.relations, inventory=INVENTORY)

    design = design_of(source)

    assert fixture.counts == {"entails": 34, "contradicts": 33, "neutral": 33}
    assert all(related.supervision.annotator is None for related in fixture.relations)
    assert source.skips == {
        JudgmentSkip.MISSING_PARTICIPANT: sum(fixture.counts.values())
    }
    assert design.n_rows == 0
    assert design.participant_levels == ()


def test_the_chaos_fixture_becomes_one_singleton_participant_per_label() -> None:
    fixture = chaos_collection()
    source = RelationSource(
        fixture.relations,
        inventory=INVENTORY,
        participant_policy=AnonymousParticipantPolicy.SINGLETON,
    )
    design = design_of(source)

    labels = sum(fixture.counts.values())

    assert source.skips == {}
    assert source.singleton_participants == labels
    assert design.n_rows == labels
    assert len(design.participant_levels) == labels
    assert len(set(design.participant_levels)) == labels
    assert Counter(design.responses.tolist()) == {2: 34, 0: 33, 1: 33}


class TestAttributedRelations(JudgmentSourceSuite):
    @pytest.fixture
    def judgment_source_subject(self) -> JudgmentSourceSubject:
        related = tuple(
            Related(premise, Entailment("hypothesis-1", relation, annotator))
            for premise in (PREMISE, OTHER_PREMISE)
            for annotator, relation in (("p1", "entails"), ("p2", "neutral"))
        )

        return JudgmentSourceSubject(
            RelationSource(related, inventory=INVENTORY),
            len(related),
            related_item_key(("premise-1", "hypothesis-1")),
            2,
            2,
        )


class TestAnonymousRelations(JudgmentSourceSuite):
    @pytest.fixture
    def judgment_source_subject(self) -> JudgmentSourceSubject:
        related = tuple(
            Related(premise, Entailment("hypothesis-1", relation, None))
            for premise in (PREMISE, OTHER_PREMISE)
            for relation in ("entails", "neutral")
        )

        return JudgmentSourceSubject(
            RelationSource(
                related,
                inventory=INVENTORY,
                participant_policy=AnonymousParticipantPolicy.SINGLETON,
            ),
            len(related),
            related_item_key(("premise-1", "hypothesis-1")),
            2,
            2,
        )
