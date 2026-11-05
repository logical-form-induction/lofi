import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Final

import pytest

from lofi.judgment._design import JudgmentRow, RowColumns
from lofi.judgment.agreement import rank_correlation
from lofi.judgment.ordinal_bounds import (
    OrdinalBoundKind,
    OrdinalBoundPrediction,
    OrdinalBoundResult,
    majority_class_bound,
    modal_response_oracle,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Mapping

LEVELS: Final = 3


class Rows:
    """An ordinal judgment source over rows declared by a test."""

    def __init__(
        self,
        rows: Iterable[JudgmentRow],
        *,
        levels: int = LEVELS,
        skips: Mapping[str, int] = MappingProxyType({}),
    ) -> None:
        self._rows = tuple(rows)
        self._levels = levels
        self._skips = skips

    @property
    def response_levels(self) -> int:
        return self._levels

    @property
    def factors(self) -> tuple[str, ...]:
        return ()

    @property
    def skips(self) -> Mapping[str, int]:
        return self._skips

    def rows(self) -> tuple[JudgmentRow, ...]:
        return self._rows


def source(
    responses: tuple[tuple[str, str, int], ...],
    *,
    levels: int = LEVELS,
    skips: Mapping[str, int] = MappingProxyType({}),
) -> Rows:
    return Rows(
        tuple(
            JudgmentRow(participant, item, response, RowColumns({}, {}))
            for participant, item, response in responses
        ),
        levels=levels,
        skips=skips,
    )


def test_majority_class_probabilities_and_summaries_match_hand_values() -> None:
    fitting = source(
        (
            ("p-0", "fit-0", 0),
            ("p-1", "fit-1", 1),
            ("p-2", "fit-2", 1),
            ("p-3", "fit-3", 2),
        )
    )
    scored = source(
        (
            ("p-a", "item-a", 1),
            ("p-b", "item-b", 0),
            ("p-c", "item-c", 2),
        )
    )

    result = majority_class_bound(fitting, scored)

    assert result.kind is OrdinalBoundKind.MAJORITY_CLASS
    assert result.rows == 3
    assert result.response_levels == LEVELS
    assert (
        tuple(prediction.distribution for prediction in result.predictions)
        == ((0.25, 0.5, 0.25),) * 3
    )
    assert tuple(prediction.predicted_level for prediction in result.predictions) == (
        1,
        1,
        1,
    )
    assert tuple(prediction.expected_level for prediction in result.predictions) == (
        1.0,
        1.0,
        1.0,
    )
    assert result.loss == pytest.approx(
        -(math.log(0.5) + math.log(0.25) + math.log(0.25)) / 3.0
    )
    assert result.accuracy == pytest.approx(1.0 / 3.0)
    assert result.rank_correlation is None


def test_scored_responses_never_enter_the_majority_fit() -> None:
    fitting = source(
        (
            ("p-0", "fit-0", 0),
            ("p-1", "fit-1", 1),
            ("p-2", "fit-2", 1),
        )
    )
    first = majority_class_bound(
        fitting,
        source(
            (
                ("p-a", "item-a", 1),
                ("p-b", "item-b", 1),
                ("p-c", "item-c", 1),
            )
        ),
    )
    second = majority_class_bound(
        fitting,
        source(
            (
                ("p-a", "item-a", 0),
                ("p-b", "item-b", 0),
                ("p-c", "item-c", 2),
            )
        ),
    )

    assert first.predictions == second.predictions
    assert first.accuracy == 1.0
    assert second.accuracy == 0.0
    assert first.loss != second.loss


def test_modal_oracle_probabilities_and_summaries_match_hand_values() -> None:
    table = source(
        (
            ("p-0", "item-a", 0),
            ("p-1", "item-a", 0),
            ("p-2", "item-a", 1),
            ("p-3", "item-a", 1),
            ("p-0", "item-b", 1),
            ("p-1", "item-b", 2),
            ("p-2", "item-b", 2),
        )
    )

    result = modal_response_oracle(table, table)

    expected_levels = (0.5, 0.5, 0.5, 0.5, 5.0 / 3.0, 5.0 / 3.0, 5.0 / 3.0)
    observed = (0, 0, 1, 1, 1, 2, 2)

    assert result.kind is OrdinalBoundKind.MODAL_RESPONSE_ORACLE
    assert tuple(prediction.distribution for prediction in result.predictions) == (
        *((0.5, 0.5, 0.0),) * 4,
        *((0.0, 1.0 / 3.0, 2.0 / 3.0),) * 3,
    )
    assert tuple(prediction.predicted_level for prediction in result.predictions) == (
        0,
        0,
        0,
        0,
        2,
        2,
        2,
    )
    assert tuple(
        prediction.expected_level for prediction in result.predictions
    ) == pytest.approx(expected_levels)
    assert result.loss == pytest.approx(
        -(4.0 * math.log(0.5) + math.log(1.0 / 3.0) + 2.0 * math.log(2.0 / 3.0)) / 7.0
    )
    assert result.accuracy == pytest.approx(4.0 / 7.0)
    assert result.rank_correlation == pytest.approx(
        rank_correlation(expected_levels, observed)
    )


def test_modal_ties_follow_declared_level_order_not_encounter_order() -> None:
    table = source(
        (
            ("p-0", "item", 1),
            ("p-1", "item", 0),
        )
    )

    result = modal_response_oracle(table, table)

    assert result.predictions[0].distribution == (0.5, 0.5, 0.0)
    assert result.predictions[0].predicted_level == 0
    assert result.rank_correlation is None


def test_a_response_absent_from_the_fit_has_infinite_unsmoothed_loss() -> None:
    fitting = source((("p-fit", "fit", 0),))
    scored = source((("p-score", "score", 1),))

    result = majority_class_bound(fitting, scored)

    assert result.predictions[0].distribution == (1.0, 0.0, 0.0)
    assert math.isinf(result.loss)
    assert result.loss > 0.0


@pytest.mark.parametrize(
    ("call", "message"),
    [
        (
            lambda: majority_class_bound(source(()), source((("p", "item", 0),))),
            "fit.*at least one response row",
        ),
        (
            lambda: majority_class_bound(source((("p", "item", 0),)), source(())),
            "score.*at least one response row",
        ),
        (
            lambda: modal_response_oracle(source(()), source((("p", "item", 0),))),
            "table.*at least one response row",
        ),
        (
            lambda: modal_response_oracle(source((("p", "item", 0),)), source(())),
            "score.*at least one response row",
        ),
        (
            lambda: majority_class_bound(
                source((("p", "item", 0),), levels=3),
                source((("p", "item", 0),), levels=4),
            ),
            "3 response levels.*4",
        ),
        (
            lambda: modal_response_oracle(
                source((("p", "item", 0),), levels=3),
                source((("p", "item", 0),), levels=4),
            ),
            "3 response levels.*4",
        ),
        (
            lambda: modal_response_oracle(
                source(
                    (("p", "item", 0),),
                    skips=MappingProxyType({"missing-response": 1}),
                ),
                source((("p", "item", 0),)),
            ),
            "complete scored item groups.*1 skipped",
        ),
    ],
)
def test_invalid_bound_inputs_are_refused(
    call: Callable[[], OrdinalBoundResult], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        call()


def test_the_modal_oracle_refuses_an_unknown_scored_item() -> None:
    table = source((("p", "known", 0),))
    scored = source((("p", "unknown", 0),))

    with pytest.raises(KeyError, match=r"unknown.*absent"):
        modal_response_oracle(table, scored)


@pytest.mark.parametrize(
    "scored_rows",
    [
        (("p-missing", "item", 0),),
        (("p-table", "item", 0), ("p-table", "item", 0)),
        (("p-table", "item", 1),),
    ],
)
def test_the_modal_oracle_requires_every_scored_response_occurrence_in_its_table(
    scored_rows: tuple[tuple[str, str, int], ...],
) -> None:
    table = source((("p-table", "item", 0),))

    with pytest.raises(ValueError, match=r"table is missing 1 occurrence"):
        modal_response_oracle(table, source(scored_rows))


def test_predictions_and_results_are_computed_frozen_values() -> None:
    result = majority_class_bound(
        source((("p-fit", "fit", 0),)),
        source((("p-score", "score", 0),)),
    )
    prediction = result.predictions[0]

    with pytest.raises(TypeError, match="computed from judgment responses"):
        OrdinalBoundPrediction()

    with pytest.raises(TypeError, match="computed from judgment responses"):
        OrdinalBoundResult()

    with pytest.raises(AttributeError, match="immutable"):
        prediction.__setattr__("item", "changed")

    with pytest.raises(AttributeError, match="immutable"):
        result.__setattr__("loss", 0.0)
