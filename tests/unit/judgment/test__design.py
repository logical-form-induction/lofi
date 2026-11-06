from typing import TYPE_CHECKING, cast

import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.judgment import (
    BoundedDesign,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    ClosedScale,
    DesignCodec,
    DesignError,
    JudgmentDesign,
    JudgmentRow,
    RowColumns,
    design_of,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

LEVELS = 5
FACTORS = ("frame", "verb")
READOUTS = ("p_gram", "length")
READOUT_VALUES = {"p_gram": 0.5, "length": 4.0}


def row(
    participant: str,
    item: str,
    response: int,
    *,
    alternatives: tuple[str, ...] | None = None,
) -> JudgmentRow:
    return JudgmentRow(
        participant,
        item,
        response,
        RowColumns(
            factors={"frame": "declarative", "verb": "know"},
            readouts={"p_gram": 0.75, "length": 8.0},
        ),
        alternatives,
    )


def codec() -> DesignCodec:
    return DesignCodec(LEVELS, FACTORS, READOUTS)


def assert_same_design(left: JudgmentDesign, right: JudgmentDesign) -> None:
    assert torch.equal(left.participants, right.participants)
    assert left.participant_levels == right.participant_levels
    assert torch.equal(left.items, right.items)
    assert left.item_levels == right.item_levels
    assert left.factor_levels == right.factor_levels
    assert left.response_levels == right.response_levels
    assert torch.equal(left.responses, right.responses)
    assert left.alternatives == right.alternatives
    assert tuple(left.factors) == tuple(right.factors)
    assert tuple(left.readouts) == tuple(right.readouts)

    for name in left.factors:
        assert torch.equal(left.factors[name], right.factors[name])

    for name in left.readouts:
        assert torch.equal(left.readouts[name], right.readouts[name])


def test_five_participants_on_one_item_remain_five_rows() -> None:
    rows = tuple(row(f"participant-{index}", "item-1", index) for index in range(5))
    design = codec().encode(rows)

    assert design.n_rows == 5
    assert design.participant_levels == tuple(
        f"participant-{index}" for index in range(5)
    )
    assert design.participants.tolist() == [0, 1, 2, 3, 4]
    assert design.item_levels == ("item-1",)
    assert design.items.tolist() == [0, 0, 0, 0, 0]
    assert design.responses.tolist() == [0, 1, 2, 3, 4]


def test_factor_indices_are_dense_and_readouts_are_columns() -> None:
    rows = (
        row("p-1", "i-1", 1),
        JudgmentRow(
            "p-2",
            "i-2",
            2,
            RowColumns(
                {"frame": "interrogative", "verb": "think"},
                {"p_gram": 0.75, "length": 8.0},
            ),
        ),
        JudgmentRow(
            "p-1",
            "i-3",
            3,
            RowColumns(
                {"frame": "declarative", "verb": "think"},
                {"p_gram": 0.75, "length": 8.0},
            ),
        ),
    )
    design = codec().encode(rows)

    assert design.factors["frame"].tolist() == [0, 1, 0]
    assert design.factor_levels["frame"] == ("declarative", "interrogative")
    assert design.factors["verb"].tolist() == [0, 1, 1]
    assert design.factor_levels["verb"] == ("know", "think")
    assert design.readouts["p_gram"].tolist() == [0.75, 0.75, 0.75]
    assert design.readouts["length"].tolist() == [8.0, 8.0, 8.0]
    assert design.participants.dtype is torch.int64
    assert design.responses.dtype is torch.int64
    assert design.readouts["p_gram"].dtype is torch.float64


def test_all_level_codes_are_canonical_under_row_reordering() -> None:
    rows = (
        JudgmentRow(
            "participant-z",
            "item-z",
            1,
            RowColumns(
                {"frame": "question", "verb": "wonder"},
                {"p_gram": 0.25, "length": 4.0},
            ),
        ),
        row("participant-a", "item-a", 2),
    )
    forward = codec().encode(rows)
    reversed_design = codec().encode(reversed(rows))

    assert forward.participant_levels == reversed_design.participant_levels
    assert forward.item_levels == reversed_design.item_levels
    assert forward.factor_levels == reversed_design.factor_levels
    assert forward.participant_levels == ("participant-a", "participant-z")
    assert forward.item_levels == ("item-a", "item-z")
    assert forward.factor_levels["frame"] == ("declarative", "question")
    assert forward.factor_levels["verb"] == ("know", "wonder")


def test_a_row_selection_retains_global_level_tables_and_codes() -> None:
    rows = (
        row("participant-z", "item-z", 1),
        row("participant-a", "item-a", 2),
        row("participant-z", "item-a", 3),
    )
    design = codec().encode(rows)
    selected = design.select_rows((2, 0))

    assert selected.participant_levels == design.participant_levels
    assert selected.item_levels == design.item_levels
    assert selected.factor_levels == design.factor_levels
    assert selected.participants.tolist() == [1, 1]
    assert selected.items.tolist() == [0, 1]
    assert selected.responses.tolist() == [3, 1]


def test_forced_choice_alternatives_keep_presentation_order() -> None:
    rows = (
        row("p-1", "trial-1", 0, alternatives=("left-1", "right-1")),
        row("p-2", "trial-2", 1, alternatives=("right-2", "left-2")),
    )
    design = codec().encode(rows)

    assert design.alternatives == (
        ("left-1", "right-1"),
        ("right-2", "left-2"),
    )
    assert codec().decode(design) == rows


@pytest.mark.parametrize("response", [-1, LEVELS])
def test_a_response_outside_the_declared_range_is_refused(response: int) -> None:
    with pytest.raises(DesignError, match="I-J2"):
        codec().encode((row("p-1", "i-1", response),))


@pytest.mark.parametrize(
    ("factors", "readouts", "message"),
    [
        ({"frame": "x"}, {"p_gram": 0.5, "length": 2.0}, "factors"),
        ({"frame": "x", "verb": "y"}, {"p_gram": 0.5}, "readouts"),
    ],
)
def test_a_row_missing_a_declared_column_is_refused(
    factors: dict[str, str], readouts: dict[str, float], message: str
) -> None:
    incomplete = JudgmentRow("p-1", "i-1", 0, RowColumns(factors, readouts))

    with pytest.raises(DesignError, match=message):
        codec().encode((incomplete,))


def test_a_forced_choice_row_refuses_repeated_or_missing_alternatives() -> None:
    columns = RowColumns({"frame": "x", "verb": "y"}, READOUT_VALUES)

    with pytest.raises(DesignError, match="needs 2"):
        JudgmentRow("p-1", "i-1", 0, columns, ("only",))

    with pytest.raises(DesignError, match="repeats"):
        JudgmentRow("p-1", "i-1", 0, columns, ("same", "same"))


def test_the_design_has_no_public_constructor() -> None:
    with pytest.raises(TypeError, match="constructed by DesignCodec"):
        JudgmentDesign()


def test_the_design_fields_are_frozen() -> None:
    design = codec().encode((row("p-1", "i-1", 0),))

    with pytest.raises(AttributeError, match="immutable"):
        design.__setattr__(
            "_responses",
            torch.tensor([1], dtype=torch.int64),
        )


def test_a_row_repr_is_a_constructor_call() -> None:
    original = row(
        "participant-1",
        "item-1",
        2,
        alternatives=("left", "right"),
    )

    assert repr(original) == (
        "JudgmentRow('participant-1', 'item-1', 2, "
        "RowColumns({'frame': 'declarative', 'verb': 'know'}, "
        "{'p_gram': 0.75, 'length': 8.0}), ('left', 'right'))"
    )


PARTICIPANTS = st.integers(min_value=0, max_value=7).map(
    lambda value: f"participant-{value}"
)
ITEMS = st.integers(min_value=0, max_value=11).map(lambda value: f"item-{value}")
RESPONSES = st.integers(min_value=0, max_value=LEVELS - 1)
FACTOR_LEVELS = st.integers(min_value=0, max_value=4).map(
    lambda value: f"level-{value}"
)
READOUT_VALUE_STRATEGY = st.floats(
    min_value=-20.0,
    max_value=20.0,
    allow_nan=False,
    allow_infinity=False,
)
ALTERNATIVES = st.one_of(
    st.none(),
    st.tuples(ITEMS, ITEMS).filter(lambda values: values[0] != values[1]),
)
ROWS = st.lists(
    st.builds(
        JudgmentRow,
        participant=PARTICIPANTS,
        item=ITEMS,
        response=RESPONSES,
        columns=st.builds(
            RowColumns,
            factors=st.fixed_dictionaries(
                {"frame": FACTOR_LEVELS, "verb": FACTOR_LEVELS}
            ),
            readouts=st.fixed_dictionaries(
                {"p_gram": READOUT_VALUE_STRATEGY, "length": READOUT_VALUE_STRATEGY}
            ),
        ),
        alternatives=ALTERNATIVES,
    ),
    max_size=24,
)


@given(ROWS)
@settings(max_examples=30, deadline=None)
def test_decode_then_encode_is_the_identity_over_generated_designs(
    rows: list[JudgmentRow],
) -> None:
    first = codec().encode(rows)
    decoded = codec().decode(first)
    second = codec().encode(decoded)

    assert decoded == tuple(rows)
    assert_same_design(first, second)


def test_invalid_runtime_row_types_are_refused() -> None:
    with pytest.raises(TypeError, match="not an integer"):
        JudgmentRow(
            "p-1",
            "i-1",
            cast("int", 1.5),
            RowColumns({"frame": "x", "verb": "y"}, READOUT_VALUES),
        )


def bounded_row(response: float, scale: ClosedScale) -> BoundedJudgmentRow:
    return BoundedJudgmentRow(
        "participant-1",
        "item-1",
        response,
        scale,
        RowColumns(
            {"frame": "declarative"},
            {"p_gram": 0.75},
        ),
    )


def test_a_closed_scale_preserves_both_endpoints_exactly() -> None:
    scale = ClosedScale(-10.0, 90.0)

    assert scale.normalize(-10.0) == 0.0
    assert scale.normalize(90.0) == 1.0
    assert repr(scale) == "ClosedScale(-10.0, 90.0)"


@pytest.mark.parametrize(
    ("lower", "upper"),
    [(0.0, 0.0), (1.0, 0.0), (float("nan"), 1.0), (0.0, float("inf"))],
)
def test_a_closed_scale_refuses_malformed_bounds(lower: float, upper: float) -> None:
    with pytest.raises((DesignError, ValueError)):
        ClosedScale(lower, upper)


@pytest.mark.parametrize("response", [-10.1, 90.1])
def test_a_bounded_row_refuses_a_response_outside_its_scale(response: float) -> None:
    with pytest.raises(DesignError, match="outside the closed scale"):
        bounded_row(response, ClosedScale(-10.0, 90.0))


def test_a_bounded_row_requires_a_floating_response() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        BoundedJudgmentRow(
            "participant-1",
            "item-1",
            cast("float", 0),
            ClosedScale(0.0, 1.0),
            RowColumns({"frame": "declarative"}, {"p_gram": 0.75}),
        )


def test_the_bounded_codec_retains_original_values_scales_and_unit_values() -> None:
    rows = (
        bounded_row(0.0, ClosedScale(0.0, 100.0)),
        BoundedJudgmentRow(
            "participant-2",
            "item-2",
            7.0,
            ClosedScale(-3.0, 7.0),
            RowColumns({"frame": "interrogative"}, {"p_gram": 0.25}),
        ),
    )
    codec = BoundedDesignCodec(factors=("frame",), readouts=("p_gram",))
    design = codec.encode(rows)

    assert isinstance(design, BoundedDesign)
    assert design.response_levels is None
    assert design.responses.dtype is torch.float64
    assert design.responses.tolist() == [0.0, 7.0]
    assert design.unit_responses.tolist() == [0.0, 1.0]
    assert design.scales == (ClosedScale(0.0, 100.0), ClosedScale(-3.0, 7.0))
    assert codec.decode(design) == rows


class Rows:
    def __init__(self, rows: tuple[JudgmentRow, ...], *, levels: int) -> None:
        self._rows = rows
        self._levels = levels

    @property
    def response_levels(self) -> int:
        return self._levels

    @property
    def factors(self) -> tuple[str, ...]:
        return ("language",)

    @property
    def skips(self) -> Mapping[str, int]:
        return {}

    def rows(self) -> tuple[JudgmentRow, ...]:
        return self._rows


def test_a_source_is_encoded_at_its_own_levels_and_factors() -> None:
    rows = tuple(
        JudgmentRow(f"p{index}", "item", index, RowColumns({"language": "en"}, {}))
        for index in range(3)
    )

    design = design_of(Rows(rows, levels=3))

    assert design.response_levels == 3
    assert tuple(design.factors) == ("language",)
    assert design.n_rows == 3
    assert design.readouts == {}


def test_a_source_whose_rows_exceed_its_levels_is_refused() -> None:
    rows = (JudgmentRow("p1", "item", 3, RowColumns({"language": "en"}, {})),)

    with pytest.raises(DesignError, match="I-J2"):
        design_of(Rows(rows, levels=3))


ALTERNATIVE_READOUTS = {"p_gram": (0.25, 0.75), "length": (3.0, 8.0)}


def choice_row(
    participant: str,
    item: str,
    response: int,
    *,
    alternatives: tuple[str, ...] = ("left", "right"),
    readouts: Mapping[str, tuple[float, ...]] | None = None,
) -> JudgmentRow:
    return JudgmentRow(
        participant,
        item,
        response,
        RowColumns(
            factors={"frame": "declarative", "verb": "know"},
            readouts={},
            alternative_readouts=dict(
                ALTERNATIVE_READOUTS if readouts is None else readouts
            ),
        ),
        alternatives,
    )


def choice_codec(alternatives: int = 2) -> DesignCodec:
    return DesignCodec(LEVELS, FACTORS, READOUTS, alternatives)


def test_a_forced_choice_design_holds_one_readout_value_per_alternative() -> None:
    rows = (
        choice_row("p-1", "trial-1", 0),
        choice_row(
            "p-2",
            "trial-2",
            1,
            alternatives=("right", "left"),
            readouts={"p_gram": (0.75, 0.25), "length": (8.0, 3.0)},
        ),
    )
    design = choice_codec().encode(rows)

    assert design.alternatives_per_row == 2
    assert design.n_rows == 2
    assert design.n_readout_rows == 4
    assert design.readouts["p_gram"].tolist() == [0.25, 0.75, 0.75, 0.25]
    assert design.readouts["length"].tolist() == [3.0, 8.0, 8.0, 3.0]
    assert design.participants.tolist() == [0, 1]
    assert choice_codec().decode(design) == rows


def test_a_forced_choice_codec_declares_at_least_two_alternatives() -> None:
    with pytest.raises(DesignError, match="at least 2 alternatives: 1"):
        DesignCodec(LEVELS, FACTORS, READOUTS, 1)


def test_a_row_gives_every_alternative_a_value_of_every_readout() -> None:
    with pytest.raises(DesignError, match="'p_gram' has 3 values and the row shows 2"):
        choice_row("p-1", "trial-1", 0, readouts={"p_gram": (0.1, 0.2, 0.3)})


def test_a_readout_per_alternative_needs_alternatives_to_give_it_to() -> None:
    with pytest.raises(DesignError, match="has 2 values and the row shows 0"):
        JudgmentRow(
            "p-1",
            "i-1",
            0,
            RowColumns({}, {}, {"p_gram": (0.25, 0.75)}),
        )


def test_each_codec_reads_the_side_its_rows_declare() -> None:
    with pytest.raises(DesignError, match="per alternative"):
        choice_codec().encode((row("p-1", "i-1", 0, alternatives=("left", "right")),))

    with pytest.raises(DesignError, match="per row"):
        codec().encode((choice_row("p-1", "i-1", 0),))


def test_a_design_shows_the_alternatives_its_codec_declares() -> None:
    with pytest.raises(DesignError, match="row 0 shows 2 alternatives and the design"):
        choice_codec(3).encode((choice_row("p-1", "i-1", 0),))

    with pytest.raises(DesignError, match="shows 2 alternatives per row and its codec"):
        choice_codec(3).decode(choice_codec().encode((choice_row("p-1", "i-1", 0),)))
