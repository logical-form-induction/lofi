from __future__ import annotations

import json
from typing import TYPE_CHECKING

import pytest

from lofi.probe._causal import HighLevelVariable, TypeAt
from lofi.probe._counterfactuals import GeneratedPair
from lofi.probe._filtering import (
    FilterReport,
    filter_counterfactuals,
    read_sentence_scores,
    require_score_coverage,
)
from lofi.probe._provenance import Provenance

if TYPE_CHECKING:
    from pathlib import Path

SCORES = (0.5, 0.9, 0.1, 0.7, 0.3, 0.6)


def a_sentence(index: int) -> tuple[str, ...]:
    return (f"really-{index}", "sleeps")


def a_pair(index: int) -> GeneratedPair:
    varied = TypeAt((1, 2))
    base_assignment: dict[HighLevelVariable, str | int] = {
        TypeAt((0, 1)): "S/S",
        varied: "Sdcl",
        TypeAt((0, 2)): "S",
    }
    variant_assignment: dict[HighLevelVariable, str | int] = dict(base_assignment)
    variant_assignment[varied] = "Sng"

    return GeneratedPair(
        a_sentence(index),
        ("S/S", "Sdcl"),
        ("S/S", "Sng"),
        base_assignment,
        variant_assignment,
        varied,
        "toy:1",
        Provenance("grammar-counterfactual", "unit", index),
    )


PAIRS = tuple(a_pair(index) for index in range(len(SCORES)))
SENTENCE_SCORES = {a_sentence(index): score for index, score in enumerate(SCORES)}


def a_report() -> FilterReport:
    return filter_counterfactuals(
        PAIRS, SENTENCE_SCORES, primary=0.5, sweep=(0.1, 0.3, 0.5, 0.7, 0.9)
    )


def test_filtering_returns_primary_sweep_and_unfiltered_arms_together() -> None:
    report = a_report()

    assert report.primary.kind == "primary"
    assert report.primary.threshold == 0.5
    assert tuple(arm.threshold for arm in report.sweep) == (0.1, 0.3, 0.5, 0.7, 0.9)
    assert all(arm.kind == "sweep" for arm in report.sweep)
    assert report.unfiltered.kind == "unfiltered"
    assert report.unfiltered.threshold is None
    assert report.arms == (report.primary, *report.sweep, report.unfiltered)


def test_each_arm_keeps_the_pairs_at_or_above_its_threshold_in_order() -> None:
    report = a_report()

    assert report.primary.ids == tuple(PAIRS[index].id for index in (0, 1, 3, 5))
    assert report.primary.dropped == 2
    assert tuple(arm.count for arm in report.sweep) == (6, 5, 4, 2, 1)
    assert report.sweep[2].ids == report.primary.ids
    assert report.unfiltered.ids == tuple(pair.id for pair in PAIRS)
    assert report.unfiltered.dropped == 0


def test_a_pair_from_an_unscored_sentence_is_refused() -> None:
    with pytest.raises(ValueError, match="no score"):
        filter_counterfactuals(PAIRS, {a_sentence(0): 1.0}, primary=0.5, sweep=(0.5,))


def test_scores_are_read_from_json_lines_keyed_by_sentence_tokens(
    tmp_path: Path,
) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(
        "".join(
            json.dumps({"tokens": list(sentence), "score": score}) + "\n"
            for sentence, score in SENTENCE_SCORES.items()
        ),
        encoding="utf-8",
    )

    assert dict(read_sentence_scores(path)) == SENTENCE_SCORES


@pytest.mark.parametrize(
    ("line", "match"),
    [
        ('{"tokens": ["a"]}', "exactly"),
        ('{"tokens": [], "score": 1}', "tokens"),
        ('{"tokens": ["a"], "score": true}', "finite"),
        ('{"tokens": ["a"], "score": NaN}', "finite"),
        ('{"tokens": ["a"], "score": 1}\n{"tokens": ["a"], "score": 2}', "twice"),
    ],
)
def test_a_malformed_score_file_is_refused(
    tmp_path: Path, line: str, match: str
) -> None:
    path = tmp_path / "scores.jsonl"
    path.write_text(line + "\n", encoding="utf-8")

    with pytest.raises(ValueError, match=match):
        read_sentence_scores(path)


def test_scores_cover_exactly_the_candidate_sentences() -> None:
    require_score_coverage(SENTENCE_SCORES, list(SENTENCE_SCORES))

    with pytest.raises(ValueError, match="miss 1"):
        require_score_coverage(SENTENCE_SCORES, [*SENTENCE_SCORES, ("other",)])

    with pytest.raises(ValueError, match="score 1"):
        require_score_coverage(SENTENCE_SCORES, list(SENTENCE_SCORES)[1:])
