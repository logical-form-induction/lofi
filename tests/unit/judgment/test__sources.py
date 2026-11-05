from typing import NamedTuple

import pytest
from tests.conformance.judgment_source import (
    JudgmentSourceSubject,
    JudgmentSourceSuite,
)

from lofi.judgment import (
    AttributedItem,
    BeadJudgmentSource,
    DesignError,
    ItemFactors,
    JudgmentSkip,
    JudgmentSource,
    design_of,
)


class AnnotationRecord(NamedTuple):
    annotator_id: str
    item_id: str
    question_name: str
    response_label: str


class ResponseEncoding(NamedTuple):
    name: str
    n_levels: int
    labels: tuple[str, ...]


def test_bead_records_preserve_item_and_participant_identity_in_the_design() -> None:
    source = BeadJudgmentSource(
        (
            AnnotationRecord("participant-7", "019a-item", "acceptability", "yes"),
            AnnotationRecord("participant-2", "019b-item", "acceptability", "no"),
            AnnotationRecord("participant-7", "019c-item", "inference", "yes"),
        ),
        ResponseEncoding("acceptability", 2, ("no", "yes")),
        factors=ItemFactors(
            {"language": "item['language']"},
            (
                AttributedItem("019a-item", {"language": "en"}, {}),
                AttributedItem("019b-item", {"language": "es"}, {}),
            ),
        ),
    )

    design = design_of(source)

    assert isinstance(source, JudgmentSource)
    assert design.participant_levels == ("participant-2", "participant-7")
    assert design.item_levels == ("019a-item", "019b-item")
    assert design.responses.tolist() == [1, 0]
    assert design.factor_levels["language"] == ("en", "es")
    assert source.skips == {JudgmentSkip.OTHER_QUESTION: 1}


def test_bead_response_labels_follow_the_encoding_order() -> None:
    source = BeadJudgmentSource(
        (
            AnnotationRecord("p", "i0", "rating", "high"),
            AnnotationRecord("p", "i1", "rating", "low"),
            AnnotationRecord("p", "i2", "rating", "middle"),
        ),
        ResponseEncoding("rating", 3, ("low", "middle", "high")),
    )

    assert tuple(row.response for row in source.rows()) == (2, 0, 1)


def test_bead_source_refuses_an_unknown_label_or_missing_factor() -> None:
    encoding = ResponseEncoding("rating", 2, ("no", "yes"))

    with pytest.raises(DesignError, match="outside"):
        BeadJudgmentSource(
            (AnnotationRecord("p", "item", "rating", "maybe"),), encoding
        )

    with pytest.raises(DesignError, match="no item has the identity 'item'"):
        BeadJudgmentSource(
            (AnnotationRecord("p", "item", "rating", "yes"),),
            encoding,
            factors=ItemFactors({"language": "item['language']"}, ()),
        )


@pytest.mark.parametrize(
    "encoding",
    [
        ResponseEncoding("rating", 3, ("no", "yes")),
        ResponseEncoding("rating", 1, ("yes",)),
        ResponseEncoding("rating", 2, ("yes", "yes")),
    ],
)
def test_bead_source_refuses_an_incoherent_encoding(
    encoding: ResponseEncoding,
) -> None:
    with pytest.raises(DesignError):
        BeadJudgmentSource((), encoding)


class TestBeadRecords(JudgmentSourceSuite):
    @pytest.fixture
    def judgment_source_subject(self) -> JudgmentSourceSubject:
        records = (
            AnnotationRecord("p1", "item", "rating", "yes"),
            AnnotationRecord("p2", "item", "rating", "no"),
            AnnotationRecord("p3", "other", "another-question", "yes"),
        )
        return JudgmentSourceSubject(
            BeadJudgmentSource(
                records,
                ResponseEncoding("rating", 2, ("no", "yes")),
            ),
            len(records),
            "item",
            2,
            2,
        )
