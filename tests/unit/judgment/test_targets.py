import math
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from tests.conformance.stratified_targets import (
    StratifiedTargetsSubject,
    StratifiedTargetsSuite,
    rows_name_declared_items_and_strata,
)

from lofi._checks import LawViolationError
from lofi.judgment import (
    AgreementBin,
    AgreementStrata,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    ClosedScale,
    DesignCodec,
    ItemTarget,
    JudgmentDesign,
    JudgmentRow,
    JudgmentTargets,
    ProbabilityTargets,
    RowColumns,
    StratifiedMeans,
    StratumMean,
    TargetRow,
    agreement_of,
    stratified_mean,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

INVENTORY: Final = ("contradicts", "neutral", "entails")
LEVELS: Final = len(INVENTORY)
ANNOTATORS: Final = 20

#: Label counts of the contested items of the inference-label fixture.
CONTESTED: Final = ((8, 7, 5), (7, 7, 6), (6, 8, 6), (5, 8, 7), (7, 6, 7))

#: Label counts of the settled items of the inference-label fixture.
SETTLED: Final = ((20, 0, 0), (0, 20, 0), (0, 0, 20), (19, 1, 0), (1, 18, 1))


def item_rows(item: str, counts: Sequence[int]) -> list[JudgmentRow]:
    responses = [level for level, count in enumerate(counts) for _ in range(count)]

    return [
        JudgmentRow(
            f"annotator-{position}", item, level, RowColumns(factors={}, readouts={})
        )
        for position, level in enumerate(responses)
    ]


def design_of_counts(items: Sequence[Sequence[int]]) -> JudgmentDesign:
    rows = [
        row
        for index, counts in enumerate(items)
        for row in item_rows(f"pair-{index}", counts)
    ]

    return DesignCodec(LEVELS).encode(rows)


def inference_labels() -> JudgmentDesign:
    """Return a design shaped like a crowd-labeled inference set.

    Most items are settled and a minority is contested, so an aggregate over
    the rows is dominated by the settled items.
    """
    return design_of_counts([*SETTLED * 9, *CONTESTED * 3])


def uniform_agreement() -> JudgmentDesign:
    """Return a design whose items all agree to the same degree."""
    return design_of_counts([(10, 5, 5), (5, 10, 5), (5, 5, 10)] * 10)


def population_score(targets: JudgmentTargets) -> list[float]:
    """Return the soft-label probability of each row's own response."""
    return [row.distribution[row.response] for row in targets.rows]


def item_score(targets: JudgmentTargets) -> list[float]:
    """Return a score that rises with the item's position and is constant within it."""
    position = {name: index for index, name in enumerate(targets.items)}

    return [float(position[row.item]) for row in targets.rows]


def test_agreement_is_one_when_unanimous_and_zero_when_uniform() -> None:
    assert agreement_of((1.0, 0.0, 0.0)) == 1.0
    assert agreement_of((0.0, 1.0)) == 1.0
    assert agreement_of((1 / 3, 1 / 3, 1 / 3)) == pytest.approx(0.0, abs=1e-12)
    assert agreement_of((0.5, 0.5)) == pytest.approx(0.0, abs=1e-12)


def test_agreement_falls_as_the_distribution_spreads() -> None:
    assert (
        agreement_of((0.9, 0.05, 0.05))
        > agreement_of((0.6, 0.2, 0.2))
        > agreement_of((0.4, 0.3, 0.3))
    )


def test_agreement_is_the_entropy_complement_on_the_uniform_scale() -> None:
    distribution = (0.7, 0.2, 0.1)
    entropy = -sum(p * math.log(p) for p in distribution)

    assert agreement_of(distribution) == pytest.approx(1.0 - entropy / math.log(3))


@pytest.mark.parametrize(
    ("distribution", "detail"),
    [
        ((1.0,), "at least two levels"),
        ((0.5, 0.6), "sum to"),
        ((1.5, -0.5), "negative probability"),
    ],
)
def test_agreement_refuses_what_is_not_a_distribution_over_levels(
    distribution: tuple[float, ...], detail: str
) -> None:
    with pytest.raises(ValueError, match=detail):
        agreement_of(distribution)


def test_the_strata_are_cut_at_the_tertiles_of_the_item_agreements() -> None:
    strata = AgreementStrata([0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0])

    assert strata.lower == pytest.approx(1 / 3)
    assert strata.upper == pytest.approx(2 / 3)
    assert strata.bin_of(0.0) is AgreementBin.LOW
    assert strata.bin_of(0.3) is AgreementBin.LOW
    assert strata.bin_of(0.5) is AgreementBin.MEDIUM
    assert strata.bin_of(2 / 3) is AgreementBin.HIGH
    assert strata.bin_of(1.0) is AgreementBin.HIGH
    assert (
        repr(strata)
        == f"AgreementStrata(lower={strata.lower!r}, upper={strata.upper!r})"
    )


def test_items_agreeing_to_one_degree_all_fall_in_the_high_stratum() -> None:
    strata = AgreementStrata([0.4, 0.4, 0.4])
    single = AgreementStrata([0.4])

    assert strata.lower == strata.upper == 0.4
    assert strata.bin_of(0.4) is AgreementBin.HIGH
    assert single.bin_of(0.4) is AgreementBin.HIGH
    assert single.bin_of(0.3) is AgreementBin.LOW


def test_strata_over_no_item_or_a_nonfinite_agreement_are_refused() -> None:
    with pytest.raises(ValueError, match="at least one item"):
        AgreementStrata([])

    with pytest.raises(ValueError, match="not finite"):
        AgreementStrata([0.5, float("nan")])


def test_an_item_target_holds_the_distribution_of_its_counts() -> None:
    target = ItemTarget("pair-0", (8, 7, 5), 0.5, AgreementBin.LOW)

    assert target.item == "pair-0"
    assert target.counts == (8, 7, 5)
    assert target.responses == 20
    assert target.distribution == (0.4, 0.35, 0.25)
    assert target.agreement == 0.5
    assert target.stratum is AgreementBin.LOW
    assert (
        repr(target)
        == "ItemTarget('pair-0', (8, 7, 5), 0.5, <AgreementBin.LOW: 'low'>)"
    )


def test_an_item_target_over_no_response_is_refused() -> None:
    with pytest.raises(ValueError, match="no response"):
        ItemTarget("pair-0", (0, 0, 0), 1.0, AgreementBin.HIGH)


def test_a_target_row_reads_both_targets_and_the_stratum_off_its_item() -> None:
    target = ItemTarget("pair-0", (8, 7, 5), 0.5, AgreementBin.LOW)
    row = TargetRow("annotator-3", target, 2)

    assert row.participant == "annotator-3"
    assert row.item == "pair-0"
    assert row.response == 2
    assert row.distribution == (0.4, 0.35, 0.25)
    assert row.agreement == 0.5
    assert row.stratum is AgreementBin.LOW
    assert repr(row) == (
        "TargetRow('annotator-3', 'pair-0', 2, <AgreementBin.LOW: 'low'>)"
    )


@pytest.mark.parametrize("response", [-1, 3])
def test_a_target_row_outside_its_item_s_levels_is_refused(response: int) -> None:
    target = ItemTarget("pair-0", (8, 7, 5), 0.5, AgreementBin.LOW)

    with pytest.raises(ValueError, match="lies outside the 3 levels"):
        TargetRow("annotator-3", target, response)


def test_every_row_has_both_targets_and_its_item_s_stratum() -> None:
    source = design_of_counts([(20, 0, 0), (8, 7, 5)])

    targets = JudgmentTargets(source)

    assert len(targets.rows) == source.n_rows == 2 * ANNOTATORS
    assert targets.items["pair-0"].distribution == (1.0, 0.0, 0.0)
    assert targets.items["pair-1"].distribution == (0.4, 0.35, 0.25)
    assert targets.items["pair-0"].stratum is AgreementBin.HIGH
    assert targets.items["pair-1"].stratum is AgreementBin.LOW

    for row in targets.rows:
        assert row.distribution == targets.items[row.item].distribution
        assert row.stratum is targets.items[row.item].stratum

    responses = [row.response for row in targets.rows]

    assert responses == source.responses.tolist()
    assert [row.participant for row in targets.rows[:2]] == [
        "annotator-0",
        "annotator-1",
    ]
    assert repr(targets) == (
        f"JudgmentTargets(rows={2 * ANNOTATORS}, items=2, strata={targets.strata!r})"
    )


def test_repeated_measures_stay_one_row_each() -> None:
    source = DesignCodec(LEVELS).encode(
        [
            JudgmentRow(
                "annotator-0", "pair-0", 0, RowColumns(factors={}, readouts={})
            ),
            JudgmentRow(
                "annotator-0", "pair-0", 2, RowColumns(factors={}, readouts={})
            ),
            JudgmentRow(
                "annotator-1", "pair-0", 2, RowColumns(factors={}, readouts={})
            ),
        ]
    )

    targets = JudgmentTargets(source)

    assert len(targets.rows) == 3
    assert targets.items["pair-0"].counts == (1, 0, 2)


def test_targets_over_an_empty_design_are_refused() -> None:
    with pytest.raises(ValueError, match="the design has none"):
        JudgmentTargets(DesignCodec(LEVELS).encode([]))


def test_a_stratum_mean_reports_no_value_over_no_rows() -> None:
    empty = StratumMean(None, None, None, items=0, rows=0)
    filled = StratumMean(0.5, 0.4, 0.6, items=3, rows=30)

    assert empty.value is None
    assert empty.interval is None
    assert empty.half_width is None
    assert filled.interval == (0.4, 0.6)
    assert filled.half_width == pytest.approx(0.1)
    assert filled.items == 3
    assert filled.rows == 30
    assert repr(filled) == "StratumMean(0.5, 0.4, 0.6, items=3, rows=30)"


def test_a_stratum_mean_is_refused_a_value_over_no_rows_or_rows_without_one() -> None:
    with pytest.raises(ValueError, match="no rows reports a mean"):
        StratumMean(0.5, 0.4, 0.6, items=0, rows=0)

    with pytest.raises(ValueError, match="reports no mean or no interval"):
        StratumMean(0.5, None, 0.6, items=1, rows=5)


def test_the_aggregate_hides_what_the_lowest_stratum_shows_on_inference_labels() -> (
    None
):
    targets = JudgmentTargets(inference_labels())

    means = stratified_mean(targets, population_score(targets), seed=1)
    aggregate = means.aggregate
    lowest = means.lowest

    assert aggregate.value is not None
    assert aggregate.half_width is not None
    assert lowest.value is not None
    assert lowest.half_width is not None
    assert means.strata[AgreementBin.LOW] is lowest
    assert lowest.items == len(CONTESTED) * 3
    assert aggregate.items == len(targets.items)
    assert aggregate.rows == len(targets.rows)
    assert aggregate.value - lowest.value > aggregate.half_width
    assert aggregate.value - lowest.value > lowest.half_width


def test_no_gap_opens_where_every_item_agrees_to_the_same_degree() -> None:
    targets = JudgmentTargets(uniform_agreement())

    means = stratified_mean(targets, item_score(targets), seed=1)
    aggregate = means.aggregate
    lowest = means.lowest

    assert {target.stratum for target in targets.items.values()} == {AgreementBin.HIGH}
    assert lowest is means.strata[AgreementBin.HIGH]
    assert means.strata[AgreementBin.LOW].rows == 0
    assert means.strata[AgreementBin.MEDIUM].rows == 0
    assert aggregate.value is not None
    assert aggregate.half_width is not None
    assert lowest.value is not None
    assert lowest.half_width is not None
    assert aggregate.half_width > 0.0
    assert lowest.value == aggregate.value
    assert abs(aggregate.value - lowest.value) <= aggregate.half_width
    assert abs(aggregate.value - lowest.value) <= lowest.half_width


def test_probability_targets_stratify_relation_values_by_binary_uncertainty() -> None:
    rows = tuple(
        BoundedJudgmentRow(
            f"participant-{position}",
            f"pair-{position // 2}",
            response,
            ClosedScale(0.0, 1.0),
            RowColumns({}, {}),
        )
        for position, response in enumerate((0.25, 0.75, 0.75, 0.75, 1.0, 1.0))
    )
    targets = ProbabilityTargets(BoundedDesignCodec().encode(rows))
    means = stratified_mean(targets, (0.0, 0.0, 1.0, 1.0, 2.0, 2.0), replicates=20)

    assert [row.stratum for row in targets.rows] == [
        AgreementBin.LOW,
        AgreementBin.LOW,
        AgreementBin.MEDIUM,
        AgreementBin.MEDIUM,
        AgreementBin.HIGH,
        AgreementBin.HIGH,
    ]
    assert means.aggregate.rows == 6
    assert all(mean.rows == 2 for mean in means.strata.values())


def test_probability_targets_refuse_a_nonprobability_scale() -> None:
    row = BoundedJudgmentRow(
        "participant-0",
        "item-0",
        4.0,
        ClosedScale(1.0, 7.0),
        RowColumns({}, {}),
    )

    with pytest.raises(ValueError, match=r"response scale.*\[0, 1\]"):
        ProbabilityTargets(BoundedDesignCodec().encode((row,)))


class TestJudgmentTargetsConformance(StratifiedTargetsSuite):
    @pytest.fixture
    def stratified_targets_subject(self) -> StratifiedTargetsSubject:
        targets = JudgmentTargets(design_of_counts(((2, 0, 0), (1, 1, 0), (1, 1, 1))))

        return StratifiedTargetsSubject(
            targets,
            items=3,
            rows=7,
            strata=frozenset(AgreementBin),
        )


class TestProbabilityTargetsConformance(StratifiedTargetsSuite):
    @pytest.fixture
    def stratified_targets_subject(self) -> StratifiedTargetsSubject:
        rows = tuple(
            BoundedJudgmentRow(
                f"participant-{position}",
                f"pair-{position // 2}",
                response,
                ClosedScale(0.0, 1.0),
                RowColumns({}, {}),
            )
            for position, response in enumerate((0.25, 0.75, 0.75, 0.75, 1.0, 1.0))
        )
        targets = ProbabilityTargets(BoundedDesignCodec().encode(rows))

        return StratifiedTargetsSubject(
            targets,
            items=3,
            rows=6,
            strata=frozenset(AgreementBin),
        )


class DefectiveTarget(NamedTuple):
    item: str
    stratum: AgreementBin


class DefectiveTargets(NamedTuple):
    items: dict[str, DefectiveTarget]
    rows: tuple[DefectiveTarget, ...]


def test_the_stratified_target_law_refuses_a_miskeyed_item() -> None:
    targets = DefectiveTargets(
        {"stored": DefectiveTarget("declared", AgreementBin.LOW)},
        (DefectiveTarget("stored", AgreementBin.LOW),),
    )

    with pytest.raises(LawViolationError, match="item keys disagree"):
        rows_name_declared_items_and_strata(
            StratifiedTargetsSubject(
                targets, items=1, rows=1, strata=frozenset({AgreementBin.LOW})
            )
        )


def test_the_stratified_target_law_refuses_a_row_in_another_item_stratum() -> None:
    targets = DefectiveTargets(
        {"item": DefectiveTarget("item", AgreementBin.HIGH)},
        (DefectiveTarget("item", AgreementBin.LOW),),
    )

    with pytest.raises(LawViolationError, match="rows disagree with their item strata"):
        rows_name_declared_items_and_strata(
            StratifiedTargetsSubject(
                targets, items=1, rows=1, strata=frozenset({AgreementBin.LOW})
            )
        )


def test_the_bootstrap_resamples_items_rather_than_rows() -> None:
    targets = JudgmentTargets(design_of_counts([(20, 0, 0), (0, 0, 20)]))
    scores = [1.0] * ANNOTATORS + [0.0] * ANNOTATORS

    means = stratified_mean(targets, scores, replicates=200, seed=2)
    lower, upper = means.aggregate.interval or (None, None)

    assert means.aggregate.value == 0.5
    assert (lower, upper) == (0.0, 1.0)


def test_the_bootstrap_is_reproducible_at_a_seed_and_moves_with_it() -> None:
    targets = JudgmentTargets(inference_labels())
    scores = population_score(targets)

    first = stratified_mean(targets, scores, replicates=100, seed=5)
    again = stratified_mean(targets, scores, replicates=100, seed=5)
    other = stratified_mean(targets, scores, replicates=100, seed=6)

    assert first.aggregate.interval == again.aggregate.interval
    assert first.aggregate.interval != other.aggregate.interval
    assert first.aggregate.value == other.aggregate.value


def test_the_interval_narrows_as_the_level_falls() -> None:
    targets = JudgmentTargets(inference_labels())
    scores = population_score(targets)

    wide = stratified_mean(targets, scores, replicates=300, level=0.99, seed=3)
    narrow = stratified_mean(targets, scores, replicates=300, level=0.5, seed=3)

    assert wide.aggregate.half_width is not None
    assert narrow.aggregate.half_width is not None
    assert wide.aggregate.half_width > narrow.aggregate.half_width


def test_stratified_means_keep_the_strata_in_order() -> None:
    targets = JudgmentTargets(inference_labels())

    means = stratified_mean(targets, population_score(targets), replicates=10)

    assert tuple(means.strata) == tuple(AgreementBin)
    assert isinstance(means, StratifiedMeans)
    assert repr(means).startswith("StratifiedMeans(StratumMean(")


@pytest.mark.parametrize(
    ("scores", "replicates", "level", "detail"),
    [
        ("short", 10, 0.95, "scores were given for"),
        ("given", 0, 0.95, "at least one replicate"),
        ("given", 10, 1.0, "not a proportion"),
        ("nan", 10, 0.95, "not finite"),
    ],
)
def test_a_stratified_mean_refuses_bad_scores_replicates_or_levels(
    scores: str, replicates: int, level: float, detail: str
) -> None:
    targets = JudgmentTargets(design_of_counts([(20, 0, 0), (8, 7, 5)]))
    given = {
        "short": [0.5],
        "given": population_score(targets),
        "nan": [float("nan")] * len(targets.rows),
    }[scores]

    with pytest.raises(ValueError, match=detail):
        stratified_mean(targets, given, replicates=replicates, level=level)


def test_the_item_agreement_is_that_of_its_count_distribution() -> None:
    source = design_of_counts([(8, 7, 5)])

    targets = JudgmentTargets(source)

    assert targets.items["pair-0"].agreement == pytest.approx(
        agreement_of((0.4, 0.35, 0.25))
    )
    assert torch.equal(source.responses, torch.tensor([0] * 8 + [1] * 7 + [2] * 5))
