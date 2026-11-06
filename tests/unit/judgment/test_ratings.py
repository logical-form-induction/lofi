from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from lairs.data import JudgmentStudy
from lairs.records import expression as expression_records
from lairs.records import judgment as judgment_records
from lairs.records import resource as resource_records
from tests.conformance.judgment_source import (
    JudgmentSourceSubject,
    JudgmentSourceSuite,
)

from lofi.data import Example, Scalar
from lofi.data.corpora.layers import LayersCorpusReader
from lofi.data.corpora.megaacceptability import (
    MegaAcceptabilityCodec,
    MegaAcceptabilitySpec,
)
from lofi.judgment import (
    AttributedItem,
    DesignError,
    ItemFactors,
    JudgmentSkip,
    JudgmentSource,
    RatingSource,
    design_of,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator

    from lairs.integrations.codecs import FragmentRecord

    from lofi.data import Supervision

SAMPLE: Final = (
    Path(__file__).resolve().parents[1]
    / "data"
    / "corpora"
    / "megaacceptability"
    / "fixtures"
    / "sample.tsv"
)


class Sentence(NamedTuple):
    item_id: str
    languages: tuple[str, ...]


class Levels:
    def __init__(self, levels: int | None, minimum: int = 1) -> None:
        self._levels = levels
        self._minimum = minimum

    @property
    def levels(self) -> int | None:
        return self._levels

    def level_of(self, value: float, /) -> int:
        return int(value) - self._minimum


class Response(NamedTuple):
    value: float | None
    scale: Levels
    annotator: str | None


class Rated(NamedTuple):
    item: Sentence
    supervision: Response


SEVEN: Final = Levels(7)
THREE: Final = Levels(3)
ENGLISH: Final = Sentence("en-1", ("en",))
MANDARIN: Final = Sentence("cmn-1", ("cmn",))


def acceptability(
    items: Iterable[Sentence], annotators: Iterable[str], scale: Levels = SEVEN
) -> tuple[Rated, ...]:
    return tuple(
        Rated(item, Response(float(1 + (position + offset) % 7), scale, annotator))
        for position, item in enumerate(items)
        for offset, annotator in enumerate(annotators)
    )


def test_a_rating_source_answers_the_judgment_source_port() -> None:
    assert isinstance(RatingSource((), levels=7), JudgmentSource)


def test_five_ratings_of_one_sentence_stay_five_rows_of_one_item() -> None:
    source = RatingSource(
        acceptability([ENGLISH], ("p1", "p2", "p3", "p4", "p5")), levels=7
    )

    design = design_of(source)

    assert design.n_rows == 5
    assert design.item_levels == ("en-1",)
    assert design.participant_levels == ("p1", "p2", "p3", "p4", "p5")
    assert design.responses.tolist() == [0, 1, 2, 3, 4]
    assert source.skips == {}


def test_a_source_records_no_factor_it_was_not_declared() -> None:
    source = RatingSource(acceptability([ENGLISH, MANDARIN], ("p1", "p2")), levels=7)

    design = design_of(source)

    assert source.factors == ()
    assert dict(design.factor_levels) == {}


def test_a_language_is_a_factor_only_when_declared() -> None:
    source = RatingSource(
        acceptability([ENGLISH, MANDARIN], ("p1", "p2")),
        levels=7,
        factors=ItemFactors(
            {"language": "item['language']"},
            (
                AttributedItem("en-1", {"language": "en"}, {}),
                AttributedItem("cmn-1", {"language": "cmn"}, {}),
            ),
        ),
    )

    assert design_of(source).factor_levels["language"] == ("cmn", "en")


def test_a_three_level_veridicality_scale_builds_a_three_level_design() -> None:
    rated = tuple(
        Rated(Sentence("v-1", ("en",)), Response(float(level), THREE, f"p{level}"))
        for level in (1, 2, 3)
    )

    design = design_of(RatingSource(rated, levels=3))

    assert design.response_levels == 3
    assert design.responses.tolist() == [0, 1, 2]


def test_declared_factors_are_read_from_the_item_record_in_column_order() -> None:
    record = AttributedItem("en-1", {"verb": "think", "frame": "NP _ that S"}, {})
    source = RatingSource(
        acceptability([ENGLISH], ("p1", "p2")),
        levels=7,
        factors=ItemFactors(
            {"frame": "item['frame']", "verb": "item['verb']"}, (record,)
        ),
    )

    design = design_of(source)

    assert source.factors == ("frame", "verb")
    assert design.factor_levels["verb"] == ("think",)
    assert design.factor_levels["frame"] == ("NP _ that S",)


def test_a_rated_item_without_a_record_is_refused_when_factors_are_declared() -> None:
    record = AttributedItem("other", {"verb": "think"}, {})

    with pytest.raises(DesignError, match="no item has the identity 'en-1'"):
        RatingSource(
            acceptability([ENGLISH], ("p1",)),
            levels=7,
            factors=ItemFactors({"verb": "item['verb']"}, (record,)),
        )


def test_a_missing_response_and_a_missing_participant_are_counted_not_rows() -> None:
    rated = (
        Rated(ENGLISH, Response(None, SEVEN, "p1")),
        Rated(ENGLISH, Response(4.0, SEVEN, None)),
        Rated(ENGLISH, Response(4.0, SEVEN, "p2")),
        Rated(ENGLISH, Response(None, SEVEN, "p3")),
    )

    source = RatingSource(rated, levels=7)

    assert len(source.rows()) == 1
    assert source.skips == {
        JudgmentSkip.MISSING_RESPONSE: 2,
        JudgmentSkip.MISSING_PARTICIPANT: 1,
    }
    assert repr(source) == ("RatingSource(rows=1, levels=7, factors=(), skipped=3)")


def test_a_rating_on_a_scale_of_another_level_count_is_refused() -> None:
    with pytest.raises(
        DesignError, match="scale of 3 levels, and this source is over 7"
    ):
        RatingSource(acceptability([ENGLISH], ("p1",), THREE), levels=7)


def test_a_continuous_probability_is_refused_rather_than_bucketed() -> None:
    graded = (Rated(ENGLISH, Response(0.73, Levels(None), "p1")),)

    with pytest.raises(DesignError, match="without ordered levels"):
        RatingSource(graded, levels=7)


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


class Sample(NamedTuple):
    ratings: tuple[Example[Scalar], ...]
    items: tuple[AttributedItem, ...]
    judgments: int


def scalar_examples(
    examples: Iterable[Example[Supervision[str]]],
) -> Iterator[Example[Scalar]]:
    for example in examples:
        if isinstance(example.supervision, Scalar):
            yield Example(example.item, example.supervision)


def sample_collection() -> Sample:
    spec = MegaAcceptabilitySpec(
        source_id="synthetic-megaacceptability",
        release_id="v2",
        spdx="CC0-1.0",
        scale_min=1,
        scale_max=7,
        frame_refs=FRAMES,
        verb_refs=VERBS,
    )
    fragment = MegaAcceptabilityCodec(spec).decode(SAMPLE.read_text(encoding="utf-8"))
    experiment = next(
        record
        for record in fragment.records
        if record.nsid == "pub.layers.judgment.experimentDef"
    )
    study = JudgmentStudy.new(experiment.local_id)
    items: dict[str, AttributedItem] = {}
    judgments = 0

    for record in fragment.records:
        model = model_of(record)
        study.add_record(record.local_id, model)

        if isinstance(model, judgment_records.JudgmentSet):
            for judgment in model.judgments:
                judgments += 1
                features = {
                    entry.key: entry.value
                    for entry in (
                        judgment.features.entries if judgment.features else ()
                    )
                }
                target = str(judgment.item.recordRef)
                items[target] = AttributedItem(target, features, {})

    ratings = tuple(scalar_examples(LayersCorpusReader(study).examples("train")))

    return Sample(ratings, tuple(items.values()), judgments)


def test_the_reader_examples_of_the_sample_table_build_a_factored_design() -> None:
    sample = sample_collection()
    source = RatingSource(
        sample.ratings,
        levels=7,
        factors=ItemFactors(
            {"verb": "item['verb']", "frame": "item['frame']"}, sample.items
        ),
    )

    design = design_of(source)

    levels = [
        rating.supervision.scale.level_of(rating.supervision.value)
        for rating in sample.ratings
        if rating.supervision.value is not None
    ]

    assert sample.judgments == 50
    assert design.n_rows + sum(source.skips.values()) == sample.judgments
    assert source.skips == {JudgmentSkip.MISSING_RESPONSE: 1}
    assert design.responses.tolist() == levels
    assert len(design.item_levels) == 10
    assert len(design.participant_levels) == 5
    assert set(design.factor_levels["frame"]) <= set(FRAMES)
    assert set(design.factor_levels["verb"]) <= set(VERBS)


class TestSampleTable(JudgmentSourceSuite):
    @pytest.fixture
    def judgment_source_subject(self) -> JudgmentSourceSubject:
        sample = sample_collection()
        source = RatingSource(sample.ratings, levels=7)
        rated = {}

        for rating in sample.ratings:
            if rating.supervision.value is not None:
                rated.setdefault(rating.item.item_id, set()).add(
                    rating.supervision.annotator
                )

        item, participants = max(rated.items(), key=lambda pair: len(pair[1]))

        return JudgmentSourceSubject(
            source, sample.judgments, item, len(participants), len(participants)
        )


class TestSyntheticRatings(JudgmentSourceSuite):
    @pytest.fixture
    def judgment_source_subject(self) -> JudgmentSourceSubject:
        annotators = ("p1", "p2", "p3", "p4", "p5")
        rated = (
            *acceptability([ENGLISH, MANDARIN], annotators),
            Rated(ENGLISH, Response(None, SEVEN, "p6")),
        )

        return JudgmentSourceSubject(
            RatingSource(rated, levels=7), len(rated), "en-1", 5, 5
        )
