from collections import Counter
from typing import Final

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.judgment import (
    ITEM,
    PARTICIPANT,
    RESPONSE,
    ComparisonSpec,
    CoverageError,
    CoverageGate,
    CoverageReport,
    DesignError,
    Fold,
    FoldCoverage,
    JudgmentRow,
    KFold,
    ModelComparisonCriterion,
    ModelComparisonSpec,
    ModelWeighting,
    RowColumns,
    Split,
    splitter_for,
)
from lofi.spec import HeldOutParticipantPolicy

VERB: Final = "verb"
FRAME: Final = "frame"
UNITS: Final = (PARTICIPANT, ITEM, VERB, FRAME)
LEVELS: Final = 7


def row(
    participant: str, item: str, verb: str, frame: str, response: int = 0
) -> JudgmentRow:
    return JudgmentRow(
        participant,
        item,
        response % LEVELS,
        RowColumns(factors={VERB: verb, FRAME: frame}, readouts={}),
    )


def crossed_rows(
    *, participants: int = 6, verbs: int = 4, frames: int = 3, ratings: int = 2
) -> tuple[JudgmentRow, ...]:
    """Return rows in which every participant rates every verb-frame item."""
    return tuple(
        row(
            f"annotator-{participant}",
            f"item-{verb}-{frame}",
            f"verb-{verb}",
            f"frame-{frame}",
            rating + verb + frame,
        )
        for verb in range(verbs)
        for frame in range(frames)
        for participant in range(participants)
        for rating in range(ratings)
    )


def nested_rows() -> tuple[JudgmentRow, ...]:
    """Return rows in which most participants rate a single frame.

    Frame ``frame-0`` is rated by three participants who rate nothing else,
    and by one participant who also rates every other frame. Frames
    ``frame-1`` and ``frame-2`` are rated by two participants each, one of
    whom is that same crossing participant.
    """
    alone = tuple(
        row(f"annotator-{index}", f"item-0-{index}", "verb-0", "frame-0")
        for index in range(3)
    )
    crossing = tuple(
        row("annotator-x", f"item-{frame}-x", f"verb-{frame}", f"frame-{frame}")
        for frame in range(3)
    )

    return (
        *alone,
        *crossing,
        row("annotator-4", "item-1-4", "verb-1", "frame-1"),
        row("annotator-5", "item-2-5", "verb-2", "frame-2"),
    )


def unit_of(record: JudgmentRow, unit: str) -> str:
    if unit == PARTICIPANT:
        return record.participant

    if unit == ITEM:
        return record.item

    return record.factors[unit]


def open_gate() -> CoverageGate:
    return CoverageGate(1.0)


def comparison(
    held_out_unit: str,
    *,
    splitter: str = "kfold",
    folds: int | None = 3,
) -> ComparisonSpec:
    return ComparisonSpec(
        held_out_unit=held_out_unit,
        readout_units={},
        splitter=splitter,
        folds=folds,
        seed=17,
        coverage_threshold=1.0,
        unseen_participant=None,
        models=ModelComparisonSpec(
            candidates=("declared", "twin"),
            candidate="declared",
            criterion=ModelComparisonCriterion.PSIS_LOO,
            weighting=ModelWeighting.STACKING,
            pareto_fraction=0.1,
            pareto_threshold=0.7,
        ),
    )


@pytest.mark.parametrize("held_out_unit", [RESPONSE, ITEM, PARTICIPANT, VERB, FRAME])
def test_a_comparison_lowers_each_declared_grain_to_kfold(
    held_out_unit: str,
) -> None:
    generator = splitter_for(comparison(held_out_unit))
    rows = crossed_rows()
    split = generator.split(rows, open_gate())
    assigned = [fold.units for fold in split.folds]
    expected = (
        {str(index) for index in range(len(rows))}
        if held_out_unit == RESPONSE
        else {unit_of(record, held_out_unit) for record in rows}
    )

    assert isinstance(generator, KFold)
    assert generator.unit == held_out_unit
    assert generator.folds == 3
    assert generator.seed == 17
    assert frozenset().union(*assigned) == expected
    assert sum(len(units) for units in assigned) == len(expected)
    assert Counter(
        record for fold in split.folds for record in fold.held_out
    ) == Counter(rows)


def test_a_by_list_comparison_lowers_to_leave_one_group_out() -> None:
    rows = crossed_rows()
    declared = comparison(
        FRAME,
        splitter="leave-one-group-out",
        folds=None,
    )
    generator = splitter_for(declared)
    split = generator.split(rows, open_gate())

    assert generator.unit == FRAME
    assert len(split.folds) == len({row.factors[FRAME] for row in rows})
    assert all(len(fold.units) == 1 for fold in split.folds)
    assert Counter(
        record for fold in split.folds for record in fold.held_out
    ) == Counter(rows)


def test_response_holdout_assigns_equal_rows_by_occurrence() -> None:
    repeated = row("annotator", "item", "verb", "frame")
    rows = (repeated,) * 7
    split = KFold(RESPONSE, folds=3, seed=5).split(rows, open_gate())
    assigned = [fold.units for fold in split.folds]

    assert frozenset().union(*assigned) == {str(index) for index in range(len(rows))}
    assert sum(len(units) for units in assigned) == len(rows)
    assert sum(len(fold.held_out) for fold in split.folds) == len(rows)

    for fold in split.folds:
        assert len(fold.held_out) == len(fold.units)
        assert len(fold.held_out) + len(fold.retained) == len(rows)


@pytest.mark.parametrize("unit", UNITS)
def test_each_generator_holds_out_every_row_of_a_unit_together(unit: str) -> None:
    rows = crossed_rows()
    split = KFold(unit, folds=3, seed=1).split(rows, open_gate())

    assert len(split.folds) == 3

    for fold in split.folds:
        held_out = {unit_of(record, unit) for record in fold.held_out}
        retained = {unit_of(record, unit) for record in fold.retained}

        assert held_out == fold.units
        assert not held_out & retained
        assert len(fold.held_out) + len(fold.retained) == len(rows)
        assert fold.held_out + fold.retained != ()

    all_units = [fold.units for fold in split.folds]

    assert frozenset().union(*all_units) == {unit_of(record, unit) for record in rows}
    assert sum(len(units) for units in all_units) == len(frozenset().union(*all_units))
    assert Counter(
        record for fold in split.folds for record in fold.held_out
    ) == Counter(rows)


def test_a_row_is_held_out_by_exactly_one_fold_and_retained_by_the_rest() -> None:
    rows = crossed_rows()
    split = KFold(ITEM, folds=4, seed=2).split(rows, open_gate())
    held = Counter(record for fold in split.folds for record in fold.held_out)
    kept = Counter(record for fold in split.folds for record in fold.retained)

    assert held == Counter(rows)
    assert kept == Counter(dict.fromkeys(rows, 3))


def test_the_folds_are_balanced_in_rows_to_within_the_largest_unit() -> None:
    rows = crossed_rows(participants=5, verbs=7, frames=2)
    split = KFold(ITEM, folds=4, seed=3).split(rows, open_gate())
    sizes = [len(fold.held_out) for fold in split.folds]
    largest_item = max(Counter(record.item for record in rows).values())

    assert max(sizes) - min(sizes) <= largest_item


@pytest.mark.parametrize("seed", range(10))
def test_dealing_the_largest_units_first_balances_a_skewed_profile_exactly(
    seed: int,
) -> None:
    profile = (8, 7, 6, 5, 4, 3, 3, 2, 1, 1)
    rows = tuple(
        row(f"annotator-{rating % 3}", f"item-{verb}-{rating}", f"verb-{verb}", "frame")
        for verb, count in enumerate(profile)
        for rating in range(count)
    )

    split = KFold(VERB, folds=2, seed=seed).split(rows, open_gate())

    assert [len(fold.held_out) for fold in split.folds] == [20, 20]


def test_the_same_seed_repeats_the_split_and_another_seed_changes_it() -> None:
    rows = crossed_rows(participants=3, verbs=6, frames=4)
    first = KFold(ITEM, folds=3, seed=7).split(rows, open_gate())
    again = KFold(ITEM, folds=3, seed=7).split(rows, open_gate())
    other = KFold(ITEM, folds=3, seed=8).split(rows, open_gate())

    assert [fold.units for fold in first.folds] == [fold.units for fold in again.folds]
    assert [fold.units for fold in first.folds] != [fold.units for fold in other.folds]


def test_a_participant_split_leaves_every_held_out_participant_unseen() -> None:
    rows = crossed_rows()
    report = KFold(PARTICIPANT, folds=3, seed=1).coverage(rows)

    assert report.unit == PARTICIPANT
    assert report.participants == 6
    assert [coverage.fraction for coverage in report.folds] == [1.0, 1.0, 1.0]
    assert [coverage.participants for coverage in report.folds] == [2, 2, 2]
    assert [coverage.unseen for coverage in report.folds] == [2, 2, 2]
    assert report.largest_fraction == 1.0


@pytest.mark.parametrize("unit", [ITEM, VERB, FRAME])
def test_a_crossed_design_split_by_anything_else_shows_every_participant(
    unit: str,
) -> None:
    report = KFold(unit, folds=3, seed=1).coverage(crossed_rows())

    assert [coverage.fraction for coverage in report.folds] == [0.0, 0.0, 0.0]
    assert [coverage.participants for coverage in report.folds] == [6, 6, 6]
    assert report.largest_fraction == 0.0


def test_a_frame_split_of_a_nested_design_reports_the_unseen_annotators() -> None:
    rows = nested_rows()
    report = KFold(FRAME, folds=3, seed=0).coverage(rows)
    by_frame = {
        next(iter(units)): coverage
        for units, coverage in zip(
            (
                fold.units
                for fold in KFold(FRAME, folds=3, seed=0).split(rows, open_gate()).folds
            ),
            report.folds,
            strict=True,
        )
    }

    assert report.participants == 6
    assert by_frame["frame-0"] == FoldCoverage(
        by_frame["frame-0"].fold, participants=4, unseen=3
    )
    assert by_frame["frame-1"] == FoldCoverage(
        by_frame["frame-1"].fold, participants=2, unseen=1
    )
    assert by_frame["frame-2"] == FoldCoverage(
        by_frame["frame-2"].fold, participants=2, unseen=1
    )
    assert by_frame["frame-0"].fraction == 0.75
    assert report.largest_fraction == 0.75


def test_a_fold_above_the_threshold_is_refused_without_a_policy() -> None:
    rows = nested_rows()

    with pytest.raises(
        CoverageError, match=r"fraction of 0\.750 above the threshold 0\.500"
    ) as refused:
        KFold(FRAME, folds=3, seed=0).split(rows, CoverageGate(0.5))

    assert refused.value.coverage.unseen == 3
    assert refused.value.coverage.participants == 4
    assert refused.value.threshold == 0.5
    assert "no policy" in str(refused.value)


def test_a_fold_above_the_threshold_constructs_under_a_declared_policy() -> None:
    rows = nested_rows()
    gate = CoverageGate(0.5, policy=HeldOutParticipantPolicy.MARGINAL)

    split = KFold(FRAME, folds=3, seed=0).split(rows, gate)
    policies = {fold.coverage.fraction: fold.policy for fold in split.folds}

    assert policies == {
        0.75: HeldOutParticipantPolicy.MARGINAL,
        0.5: HeldOutParticipantPolicy.MARGINAL,
    }


def test_a_declared_policy_applies_to_every_fold_below_the_threshold() -> None:
    rows = nested_rows()
    gate = CoverageGate(1.0, policy=HeldOutParticipantPolicy.PLUG_IN)

    split = KFold(FRAME, folds=3, seed=0).split(rows, gate)

    assert all(fold.coverage.unseen > 0 for fold in split.folds)
    assert all(fold.policy is HeldOutParticipantPolicy.PLUG_IN for fold in split.folds)


def test_a_fold_at_or_below_the_threshold_constructs_without_a_policy() -> None:
    rows = nested_rows()

    split = KFold(FRAME, folds=3, seed=0).split(rows, CoverageGate(0.75))

    assert all(fold.policy is None for fold in split.folds)
    assert split.report.largest_fraction == 0.75


def test_a_participant_split_is_refused_at_any_threshold_below_one() -> None:
    rows = crossed_rows()

    with pytest.raises(
        CoverageError, match="fold 0 holds out 2 participants of whom 2"
    ):
        KFold(PARTICIPANT, folds=3, seed=1).split(rows, CoverageGate(0.99))

    admitted = KFold(PARTICIPANT, folds=3, seed=1).split(
        rows, CoverageGate(0.0, policy=HeldOutParticipantPolicy.PLUG_IN)
    )

    assert all(
        fold.policy is HeldOutParticipantPolicy.PLUG_IN for fold in admitted.folds
    )


def test_the_gate_admits_at_the_threshold_and_refuses_just_above_it() -> None:
    gate = CoverageGate(0.5)
    at = FoldCoverage(0, participants=4, unseen=2)
    above = FoldCoverage(1, participants=4, unseen=3)

    assert gate.admit(at) is None

    with pytest.raises(CoverageError, match="fold 1"):
        gate.admit(above)

    declared = CoverageGate(0.5, policy=HeldOutParticipantPolicy.PLUG_IN)

    assert declared.admit(above) is HeldOutParticipantPolicy.PLUG_IN
    assert declared.admit(at) is HeldOutParticipantPolicy.PLUG_IN
    assert repr(gate) == "CoverageGate(0.5, policy=None)"


def test_the_report_of_a_split_is_the_report_computed_without_a_gate() -> None:
    rows = nested_rows()
    generator = KFold(VERB, folds=3, seed=4)

    split = generator.split(rows, open_gate())
    report = generator.coverage(rows)

    assert split.report.folds == report.folds
    assert split.report.participants == report.participants
    assert split.report.unit == VERB == report.unit
    assert tuple(fold.coverage for fold in split.folds) == report.folds


@pytest.mark.parametrize("threshold", [-0.1, 1.5, float("nan"), float("inf")])
def test_a_threshold_outside_the_unit_interval_is_refused(threshold: float) -> None:
    with pytest.raises(ValueError, match="between zero and one"):
        CoverageGate(threshold)


def test_fewer_levels_than_folds_is_refused() -> None:
    rows = crossed_rows(participants=2, verbs=2, frames=2)

    with pytest.raises(
        DesignError, match="needs at least 3 levels, and the rows have 2"
    ):
        KFold(VERB, folds=3, seed=0).split(rows, open_gate())

    with pytest.raises(DesignError, match="needs at least 3 levels"):
        KFold(PARTICIPANT, folds=3, seed=0).coverage(rows)


def test_a_row_without_a_level_of_the_factor_is_refused() -> None:
    rows = (
        row("annotator-0", "item-0", "verb-0", "frame-0"),
        JudgmentRow("annotator-1", "item-1", 0, RowColumns({VERB: "verb-1"}, {})),
    )

    with pytest.raises(DesignError, match="a split by 'frame' needs a level"):
        KFold(FRAME, folds=2, seed=0).coverage(rows)


@pytest.mark.parametrize("folds", [0, 1])
def test_a_split_has_at_least_two_folds(folds: int) -> None:
    with pytest.raises(ValueError, match="at least 2 folds"):
        KFold(ITEM, folds=folds, seed=0)


def test_a_split_names_its_unit() -> None:
    with pytest.raises(ValueError, match="names the unit"):
        KFold("", folds=2, seed=0)


def test_a_coverage_with_impossible_counts_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot hold out 2 participants of whom 3"):
        FoldCoverage(0, participants=2, unseen=3)

    with pytest.raises(ValueError, match="nonnegative"):
        FoldCoverage(-1, participants=2, unseen=1)

    with pytest.raises(ValueError, match="nonnegative"):
        FoldCoverage(0, participants=2, unseen=-1)


def test_a_coverage_of_no_participants_has_a_fraction_of_zero() -> None:
    assert FoldCoverage(0, participants=0, unseen=0).fraction == 0.0
    assert CoverageReport(ITEM, participants=0, folds=()).largest_fraction == 0.0


def test_a_report_numbers_its_folds_from_zero_in_order() -> None:
    with pytest.raises(ValueError, match="numbered from zero in order"):
        CoverageReport(
            ITEM,
            participants=3,
            folds=(FoldCoverage(1, participants=1, unseen=0),),
        )


def test_coverages_compare_by_fold_and_counts() -> None:
    one = FoldCoverage(0, participants=4, unseen=1)

    assert one == FoldCoverage(0, participants=4, unseen=1)
    assert one != FoldCoverage(1, participants=4, unseen=1)
    assert hash(one) == hash(FoldCoverage(0, participants=4, unseen=1))
    assert repr(one) == "FoldCoverage(0, participants=4, unseen=1)"


def test_the_values_render_their_shape() -> None:
    rows = crossed_rows(participants=2, verbs=2, frames=2)
    generator = KFold(ITEM, folds=2, seed=5)
    split = generator.split(rows, open_gate())
    fold = split.folds[0]

    assert repr(generator) == "KFold('item', folds=2, seed=5)"
    assert repr(fold) == (
        f"Fold(0, units={len(fold.units)}, held_out={len(fold.held_out)}, "
        f"retained={len(fold.retained)})"
    )
    assert repr(split.report) == (
        "CoverageReport(unit='item', participants=2, fractions=[0.000, 0.000])"
    )
    assert repr(split) == f"Split(folds=2, report={split.report!r})"
    assert fold.index == 0
    assert isinstance(fold, Fold)
    assert isinstance(split, Split)
    assert (generator.unit, generator.folds, generator.seed) == (ITEM, 2, 5)


@settings(max_examples=60, deadline=None)
@given(
    rows=st.lists(
        st.tuples(
            st.integers(0, 5), st.integers(0, 7), st.integers(0, 3), st.integers(0, 2)
        ),
        min_size=4,
        max_size=60,
    ),
    unit=st.sampled_from(UNITS),
    folds=st.integers(2, 4),
    seed=st.integers(0, 1000),
)
def test_generated_splits_partition_the_units_and_count_the_coverage(
    rows: list[tuple[int, int, int, int]],
    unit: str,
    folds: int,
    seed: int,
) -> None:
    records = tuple(
        row(f"annotator-{p}", f"item-{i}", f"verb-{v}", f"frame-{f}")
        for p, i, v, f in rows
    )
    levels = {unit_of(record, unit) for record in records}
    generator = KFold(unit, folds=folds, seed=seed)

    if len(levels) < folds:
        with pytest.raises(DesignError, match="needs at least"):
            generator.coverage(records)

        return

    split = generator.split(records, CoverageGate(1.0))
    held_units = [fold.units for fold in split.folds]

    assert all(held_units)
    assert frozenset().union(*held_units) == levels
    assert sum(len(units) for units in held_units) == len(levels)
    assert Counter(
        record for fold in split.folds for record in fold.held_out
    ) == Counter(records)

    for fold in split.folds:
        held = {record.participant for record in fold.held_out}
        seen = {record.participant for record in fold.retained}

        assert fold.coverage.participants == len(held)
        assert fold.coverage.unseen == len(held - seen)
        assert all(unit_of(record, unit) in fold.units for record in fold.held_out)
        assert all(unit_of(record, unit) not in fold.units for record in fold.retained)

    assert split.report.participants == len({record.participant for record in records})
