from math import log, log10, pi
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from quivers.diagnostics import to_datatree_from_svi
from tests.conformance._suite import Conformance, ConformanceSuite

from lofi._checks import LawViolationError
from lofi.eval._values import Comparison
from lofi.eval.elpd import ParetoGate
from lofi.eval.laws import RefitFixture, check_refit_laws
from lofi.eval.refit import (
    BestConfiguration,
    CellRow,
    RankStability,
    RefitCell,
    RefitGrid,
    RefitStudy,
    StabilityRule,
    _pointwise,
    _rank_correlation,
    _stability,
    refit_study,
)

if TYPE_CHECKING:
    from xarray import DataTree

SIGNATURES: Final = ("grammar", "twin")

RATES: Final = (0.001, 0.01, 0.1)

BUDGETS: Final = (4, 8, 64, 256)

MATCHED: Final = ("grammar", "matched")

# how far each signature's fit sits from the sample mean once it has converged
FLOORS: Final = {"grammar": 0.0, "twin": 0.42, "matched": 0.42}

# the learning rate each signature converges fastest under
BEST_RATES: Final = {"grammar": 0.01, "twin": 0.1, "matched": 0.01}

# how much of the remaining distance is still uncovered at one unit of budget
TRANSIENTS: Final = {"grammar": 6.0, "twin": 0.4, "matched": 6.0}

OBSERVATIONS: Final = 40

DRAWS: Final = 200

FALSIFIER: Final = (
    "the twin scores at least as well as the grammar at the largest budget"
)

FAMILY: Final = "optimization-confound"

REPLICATES: Final = 999

SEED: Final = 11

GATE: Final = ParetoGate(fraction=0.1)

RULE: Final = StabilityRule(tail=2, threshold=0.9)


def _sample() -> torch.Tensor:
    """Return one sample from a unit-variance normal with mean two."""
    generator = torch.Generator().manual_seed(0)

    return 2.0 + torch.randn(OBSERVATIONS, generator=generator)


OBSERVED: Final = _sample()


def _offset(cell: RefitCell) -> float:
    """Return how far one cell's fitted mean sits from the sample mean.

    A signature pays a standing penalty for the distance between the learning
    rate it is fit at and the one it converges fastest under, and a further
    penalty that shrinks as its budget grows. The grammar converges to the
    sample mean and the twin does not, while the twin covers its own distance in
    far less budget, so the two orderings disagree until the budget is large.
    """
    missed = abs(log10(cell.learning_rate) - log10(BEST_RATES[cell.signature]))

    return (
        FLOORS[cell.signature]
        + 0.25 * missed
        + TRANSIENTS[cell.signature] / cell.budget
    )


def _fit(observed: torch.Tensor, offset: float) -> DataTree:
    """Return a posterior over a mean that sits one offset from the sample mean."""
    generator = torch.Generator().manual_seed(7)
    means = (
        float(observed.mean())
        + offset
        + torch.randn(DRAWS, generator=generator) / OBSERVATIONS**0.5
    )
    likelihood = -0.5 * (observed[None, :] - means[:, None]) ** 2 - 0.5 * log(2 * pi)

    return to_datatree_from_svi({"mu": means}, log_likelihood={"y": likelihood})


def _refit(cell: RefitCell, /) -> DataTree:
    """Return the fit of one cell, whose distance from the truth the cell decides."""
    return _fit(OBSERVED, _offset(cell))


def _partial(cell: RefitCell, /) -> DataTree:
    """Return a fit scored on one fewer observation for one of the signatures."""
    kept = OBSERVATIONS - int(cell.signature == SIGNATURES[-1])

    return _fit(OBSERVED[:kept], _offset(cell))


class _Drifting:
    """A refit whose fit moves further away on every call."""

    def __init__(self) -> None:
        self._calls = 0

    def __call__(self, cell: RefitCell, /) -> DataTree:
        self._calls += 1

        return _fit(OBSERVED, _offset(cell) + self._calls)


def _densities(fit: DataTree) -> tuple[float, ...]:
    return _pointwise(fit, var_name=None)


def _study(
    rates: tuple[float, ...],
    rule: StabilityRule = RULE,
    signatures: tuple[str, ...] = SIGNATURES,
) -> RefitStudy:
    return refit_study(
        _refit,
        RefitGrid(signatures, rates, BUDGETS),
        gate=GATE,
        rule=rule,
        falsifier=FALSIFIER,
        family=FAMILY,
        replicates=REPLICATES,
        seed=SEED,
    )


def _difference(study: RefitStudy) -> float:
    """Return the grammar's difference over the twin at the largest budget."""
    for comparison in study.comparisons:
        if comparison.left == "grammar" and comparison.right == "twin":
            return comparison.statistic

    message = f"no comparison ran the grammar against the twin: {study!r}"

    raise AssertionError(message)


def _both(grid: RefitGrid) -> tuple[RefitCell, ...]:
    """Return one cell of each signature."""
    return (grid.cells[0], grid.cells[-1])


@pytest.fixture(scope="module")
def grid() -> RefitGrid:
    return RefitGrid(SIGNATURES, RATES, BUDGETS)


@pytest.fixture(scope="module")
def study() -> RefitStudy:
    """Return the study over the whole grid, each signature choosing its own rate."""
    return _study(RATES)


@pytest.fixture(scope="module")
def matched() -> RefitStudy:
    """Return the study of a twin as easy to train as the grammar it copies."""
    return _study(RATES, signatures=MATCHED)


@pytest.fixture(scope="module")
def at_slow_rate() -> RefitStudy:
    """Return the study at the one rate the twin converges slowest under."""
    return _study((RATES[0],))


@pytest.fixture(scope="module")
def at_fast_rate() -> RefitStudy:
    """Return the study at the one rate the twin converges fastest under."""
    return _study((RATES[-1],))


# ---------------------------------------------------------------------------
# the grid


def test_a_grid_needs_two_signatures() -> None:
    with pytest.raises(ValueError, match="two signatures"):
        RefitGrid(("grammar",), RATES, BUDGETS)


def test_a_grid_refuses_a_repeated_signature() -> None:
    with pytest.raises(ValueError, match="more than once"):
        RefitGrid(("grammar", "grammar"), RATES, BUDGETS)


def test_a_grid_refuses_an_unnamed_signature() -> None:
    with pytest.raises(ValueError, match="is empty"):
        RefitGrid(("grammar", ""), RATES, BUDGETS)


def test_a_grid_refuses_a_nonpositive_learning_rate() -> None:
    with pytest.raises(ValueError, match="not positive"):
        RefitGrid(SIGNATURES, (0.0, 0.1), BUDGETS)


def test_a_grid_refuses_learning_rates_out_of_order() -> None:
    with pytest.raises(ValueError, match="do not increase"):
        RefitGrid(SIGNATURES, (0.1, 0.01), BUDGETS)


def test_a_grid_needs_two_budgets() -> None:
    with pytest.raises(ValueError, match="two budgets"):
        RefitGrid(SIGNATURES, RATES, (16,))


def test_a_grid_refuses_a_zero_budget() -> None:
    with pytest.raises(ValueError, match="fits nothing"):
        RefitGrid(SIGNATURES, RATES, (0, 16))


def test_a_grid_refuses_budgets_out_of_order() -> None:
    with pytest.raises(ValueError, match="do not increase"):
        RefitGrid(SIGNATURES, RATES, (64, 16))


def test_a_grid_names_one_cell_per_configuration(grid: RefitGrid) -> None:
    assert len(grid.cells) == len(SIGNATURES) * len(RATES) * len(BUDGETS)
    assert len(set(grid.cells)) == len(grid.cells)


def test_a_grid_of_one_learning_rate_chooses_nothing() -> None:
    assert RefitGrid(SIGNATURES, (0.01,), BUDGETS).refits is False
    assert RefitGrid(SIGNATURES, RATES, BUDGETS).refits is True


# ---------------------------------------------------------------------------
# the stability rule and the rank correlation


def test_a_rule_needs_two_budget_points() -> None:
    with pytest.raises(ValueError, match="at least two budget points"):
        StabilityRule(tail=1, threshold=0.9)


def test_a_rule_refuses_a_threshold_off_the_correlation_scale() -> None:
    with pytest.raises(ValueError, match="not a rank correlation"):
        StabilityRule(tail=2, threshold=1.5)


def test_a_rule_settles_at_its_threshold() -> None:
    rule = StabilityRule(tail=2, threshold=0.9)

    assert rule.holds(0.9)
    assert not rule.holds(0.89)


def test_an_ordering_correlates_perfectly_with_itself() -> None:
    assert _rank_correlation(("a", "b", "c"), ("a", "b", "c")) == 1.0


def test_a_reversed_ordering_correlates_negatively() -> None:
    assert _rank_correlation(("a", "b", "c"), ("c", "b", "a")) == -1.0


def test_one_swap_moves_the_correlation_off_one() -> None:
    assert _rank_correlation(("a", "b", "c"), ("b", "a", "c")) == pytest.approx(1 / 3)


def test_two_orderings_of_different_names_do_not_correlate() -> None:
    with pytest.raises(ValueError, match="different names"):
        _rank_correlation(("a", "b"), ("a", "c"))


# ---------------------------------------------------------------------------
# the driver


def test_the_driver_emits_one_row_per_cell(study: RefitStudy, grid: RefitGrid) -> None:
    emitted = tuple(
        RefitCell(row.signature, row.learning_rate, row.budget) for row in study.cells
    )

    assert emitted == grid.cells


def test_each_signature_refits_at_its_own_best_rate(study: RefitStudy) -> None:
    chosen = {best.signature: best.learning_rate for best in study.best}

    assert chosen == {name: BEST_RATES[name] for name in SIGNATURES}
    assert chosen["grammar"] != chosen["twin"]


def test_the_chosen_rate_is_the_one_that_signature_scored_highest_under(
    study: RefitStudy,
) -> None:
    budget = study.curve[-1].budget

    for best in study.best:
        scored = [
            row.elpd
            for row in study.cells
            if row.signature == best.signature and row.budget == budget
        ]

        assert best.elpd == max(scored)


def test_the_ordering_flips_between_the_smallest_and_the_largest_budget(
    study: RefitStudy,
) -> None:
    assert study.curve[0].ranking[0] == "twin"
    assert study.curve[-1].ranking[0] == "grammar"


def test_the_curve_runs_over_every_budget(study: RefitStudy) -> None:
    assert tuple(point.budget for point in study.curve) == BUDGETS


def test_the_ranking_is_stable_in_the_limit(study: RefitStudy) -> None:
    assert study.stability == RankStability(
        correlation=1.0, points=2, pairs=1, settled=True
    )


def test_a_longer_tail_catches_an_ordering_still_moving(study: RefitStudy) -> None:
    moving = _stability(study.curve, StabilityRule(tail=4, threshold=0.9))

    assert moving.points == len(BUDGETS)
    assert moving.correlation < 1.0
    assert not moving.settled


def test_a_rule_reading_past_the_curve_is_refused(study: RefitStudy) -> None:
    with pytest.raises(ValueError, match="budget points"):
        _stability(study.curve, StabilityRule(tail=len(BUDGETS) + 1, threshold=0.9))


def test_a_rule_reading_past_the_grid_is_refused(grid: RefitGrid) -> None:
    with pytest.raises(ValueError, match="budget points"):
        refit_study(
            _refit,
            grid,
            gate=GATE,
            rule=StabilityRule(tail=len(BUDGETS) + 1, threshold=0.9),
            falsifier=FALSIFIER,
            family=FAMILY,
            replicates=REPLICATES,
            seed=SEED,
        )


def test_the_driver_refuses_something_that_is_not_a_grid() -> None:
    with pytest.raises(TypeError, match="grid is not one"):
        refit_study(
            _refit,
            cast("RefitGrid", SIGNATURES),
            gate=GATE,
            rule=RULE,
            falsifier=FALSIFIER,
            family=FAMILY,
            replicates=REPLICATES,
            seed=SEED,
        )


def test_a_refit_comparison_is_a_finding(study: RefitStudy) -> None:
    assert study.smoke_test is False
    assert len(study.comparisons) == 1

    for comparison in study.comparisons:
        assert comparison.smoke_test is False
        assert comparison.falsifier == FALSIFIER
        assert comparison.family == FAMILY
        assert 0.0 < comparison.p_value.value <= 1.0


def test_a_comparison_without_the_refit_is_a_smoke_test(
    at_slow_rate: RefitStudy,
) -> None:
    assert at_slow_rate.smoke_test is True

    for comparison in at_slow_rate.comparisons:
        assert comparison.smoke_test is True
        assert comparison.falsifier == FALSIFIER


def test_one_shared_learning_rate_misstates_the_difference(
    study: RefitStudy, at_slow_rate: RefitStudy, at_fast_rate: RefitStudy
) -> None:
    assert _difference(at_slow_rate) > _difference(study) > _difference(at_fast_rate)


def test_a_twin_of_equal_trainability_shows_no_confound(
    matched: RefitStudy,
) -> None:
    chosen = {best.signature: best.learning_rate for best in matched.best}
    settled = _stability(matched.curve, StabilityRule(tail=len(BUDGETS), threshold=0.9))

    assert len(set(chosen.values())) == 1
    assert {point.ranking[0] for point in matched.curve} == {"grammar"}
    assert settled.correlation == 1.0
    assert settled.settled


def test_the_rows_cover_the_cells_the_curve_and_the_statistic(
    study: RefitStudy,
) -> None:
    counted: dict[str, int] = {}

    for row in study.rows():
        kind = row["kind"]

        assert isinstance(kind, str)

        counted[kind] = counted.get(kind, 0) + 1

    pairs = len(SIGNATURES) * (len(SIGNATURES) - 1) // 2

    assert counted == {
        "cell": len(study.grid.cells),
        "best": len(BUDGETS) * len(SIGNATURES),
        "delta": len(BUDGETS) * pairs,
        "stability": 1,
    }


def test_every_cell_row_names_its_configuration(study: RefitStudy) -> None:
    for row in study.cells:
        spec = row.to_spec()

        assert spec["signature"] == row.signature
        assert spec["learning_rate"] == row.learning_rate
        assert spec["budget"] == row.budget
        assert spec["observations"] == OBSERVATIONS


# ---------------------------------------------------------------------------
# what a study refuses


def test_a_study_refuses_a_missing_cell(study: RefitStudy) -> None:
    with pytest.raises(ValueError, match="cells"):
        RefitStudy(
            study.grid,
            study.cells[:-1],
            study.curve,
            study.comparisons,
            stability=study.stability,
        )


def test_a_study_refuses_a_curve_off_the_grid(study: RefitStudy) -> None:
    with pytest.raises(ValueError, match="curve runs over"):
        RefitStudy(
            study.grid,
            study.cells,
            study.curve[:-1],
            study.comparisons,
            stability=study.stability,
        )


def test_a_study_refuses_a_smoke_flag_that_contradicts_the_grid(
    study: RefitStudy,
) -> None:
    emitted = study.comparisons[0]
    contradicting = Comparison(
        emitted.left,
        emitted.right,
        statistic=emitted.statistic,
        p_value=emitted.p_value,
        effect=emitted.effect,
        falsifier=emitted.falsifier,
        smoke_test=True,
        family=emitted.family,
    )

    with pytest.raises(ValueError, match="smoke-test status"):
        RefitStudy(
            study.grid,
            study.cells,
            study.curve,
            (contradicting,),
            stability=study.stability,
        )


def test_a_study_refuses_a_statistic_reading_past_its_curve(
    study: RefitStudy,
) -> None:
    with pytest.raises(ValueError, match="budget points"):
        RefitStudy(
            study.grid,
            study.cells,
            study.curve,
            study.comparisons,
            stability=RankStability(
                correlation=1.0, points=len(BUDGETS) + 1, pairs=1, settled=True
            ),
        )


def test_a_study_refuses_a_cell_row_that_is_not_one(study: RefitStudy) -> None:
    with pytest.raises(TypeError, match="cell row is not one"):
        RefitStudy(
            study.grid,
            (*study.cells[:-1], cast("CellRow", study.grid.cells[-1])),
            study.curve,
            study.comparisons,
            stability=study.stability,
        )


# ---------------------------------------------------------------------------
# the protocol laws


def test_a_refit_satisfies_the_protocol_laws(grid: RefitGrid) -> None:
    check_refit_laws(RefitFixture(_refit, _both(grid), _densities))


def test_the_laws_catch_a_refit_that_answers_no_cell() -> None:
    with pytest.raises(LawViolationError, match="no cell"):
        check_refit_laws(RefitFixture(_refit, (), _densities))


def test_the_laws_catch_a_refit_that_does_not_repeat(grid: RefitGrid) -> None:
    drifting = _Drifting()

    with pytest.raises(LawViolationError, match="deterministic"):
        check_refit_laws(RefitFixture(drifting, grid.cells[:1], _densities))


def test_the_laws_catch_fits_scored_on_different_observations(
    grid: RefitGrid,
) -> None:
    with pytest.raises(LawViolationError, match="observation counts"):
        check_refit_laws(RefitFixture(_partial, _both(grid), _densities))


# ---------------------------------------------------------------------------
# value laws


class TestARefitGrid(ConformanceSuite):
    """Tests refit grids against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[RefitGrid]:
        """Return the conformance the laws run a grid through."""
        return Conformance(
            build=lambda: RefitGrid(SIGNATURES, RATES, BUDGETS),
            unequal=RefitGrid(SIGNATURES, RATES, (4, 8, 64, 512)),
            operations=(
                ("signatures", lambda value: value.signatures),
                ("learning_rates", lambda value: value.learning_rates),
                ("budgets", lambda value: value.budgets),
                ("cells", lambda value: value.cells),
                ("refits", lambda value: value.refits),
            ),
        )


class TestAStabilityRule(ConformanceSuite):
    """Tests stability rules against value laws."""

    @pytest.fixture
    def conformance(self) -> Conformance[StabilityRule]:
        """Return the conformance the laws run a rule through."""
        return Conformance(
            build=lambda: StabilityRule(tail=2, threshold=0.9),
            unequal=StabilityRule(tail=3, threshold=0.9),
            operations=(
                ("tail", lambda value: value.tail),
                ("threshold", lambda value: value.threshold),
                ("holds", lambda value: value.holds(1.0)),
            ),
        )


class TestARefitStudy(ConformanceSuite):
    """Tests refit studies against value laws."""

    @pytest.fixture
    def conformance(self, study: RefitStudy) -> Conformance[RefitStudy]:
        """Return the conformance the laws run a study through."""
        return Conformance(
            build=lambda: RefitStudy(
                study.grid,
                study.cells,
                study.curve,
                study.comparisons,
                stability=study.stability,
            ),
            unequal=RefitStudy(
                study.grid,
                study.cells,
                study.curve,
                study.comparisons,
                stability=RankStability(
                    correlation=0.0, points=2, pairs=1, settled=False
                ),
            ),
            operations=(
                ("cells", lambda value: value.cells),
                ("curve", lambda value: value.curve),
                ("best", lambda value: value.best),
                ("stability", lambda value: value.stability),
                ("smoke_test", lambda value: value.smoke_test),
            ),
        )


def test_a_cell_row_and_a_best_configuration_serialize_their_fields() -> None:
    row = CellRow(
        signature="grammar",
        learning_rate=0.01,
        budget=64,
        elpd=-1.0,
        se=0.5,
        observations=OBSERVATIONS,
        exceeding=0,
        reliable=True,
    )
    best = BestConfiguration(
        signature="grammar", learning_rate=0.01, elpd=-1.0, reliable=True
    )

    assert row.to_spec()["elpd"] == row.elpd
    assert best.to_spec()["reliable"] is True
