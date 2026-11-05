from math import log, pi
from typing import TYPE_CHECKING, Final

import pytest
import torch
from quivers.diagnostics import to_datatree_from_svi

from lofi.eval.elpd import ParetoGate
from lofi.eval.laws import RefitFixture, check_refit_laws
from lofi.eval.refit import (
    RefitCell,
    RefitGrid,
    RefitStudy,
    StabilityRule,
    _pointwise,
    refit_study,
)

if TYPE_CHECKING:
    from xarray import DataTree

pytestmark = pytest.mark.integration

SIGNATURES: Final = ("grammar", "twin")

RATES: Final = (0.01, 0.05, 0.25)

BUDGETS: Final = (5, 50, 500, 5000)

OBSERVATIONS: Final = 200

DRAWS: Final = 200

FALSIFIER: Final = (
    "the twin scores at least as well as the grammar at the largest budget"
)

FAMILY: Final = "optimization-confound"

REPLICATES: Final = 999

SEED: Final = 3


def _corpus() -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Return the response and the one predictor each signature reads.

    The response is built from both predictors. The grammar reads the predictor
    that explains more of it and whose spread is small, so a gradient step moves
    its coefficient slowly. The twin reads the predictor that explains less and
    whose spread is wide, so it converges in a few steps and then stops
    improving.
    """
    generator = torch.Generator().manual_seed(0)
    wide = torch.randn(OBSERVATIONS, generator=generator)
    narrow = torch.randn(OBSERVATIONS, generator=generator) * 0.1
    noise = torch.randn(OBSERVATIONS, generator=generator) * 0.5

    return (
        1.0 + 4.0 * narrow + 0.3 * wide + noise,
        {"grammar": narrow, "twin": wide},
    )


RESPONSE, PREDICTORS = _corpus()


def _trained(cell: RefitCell) -> torch.Tensor:
    """Return the coefficients gradient descent reaches under one configuration."""
    design = torch.stack([torch.ones(OBSERVATIONS), PREDICTORS[cell.signature]], dim=1)
    coefficients = torch.zeros(design.shape[1], requires_grad=True)
    optimizer = torch.optim.SGD([coefficients], lr=cell.learning_rate)

    for _ in range(cell.budget):
        optimizer.zero_grad()
        loss = ((design @ coefficients - RESPONSE) ** 2).mean()
        loss.backward()
        optimizer.step()

    return coefficients.detach()


def _refit(cell: RefitCell, /) -> DataTree:
    """Return a Laplace posterior around the point the optimizer reached.

    The residual spread is estimated from the fit itself, and each coefficient
    is drawn around its estimate at the standard error of a least-squares
    coefficient, so a configuration that has not converged is scored as the
    worse fit it is.
    """
    design = torch.stack([torch.ones(OBSERVATIONS), PREDICTORS[cell.signature]], dim=1)
    estimate = _trained(cell)
    spread = float(((design @ estimate - RESPONSE) ** 2).mean()) ** 0.5
    errors = spread / (OBSERVATIONS**0.5 * design.std(dim=0).clamp(min=1.0))
    generator = torch.Generator().manual_seed(SEED)
    drawn = estimate + torch.randn(DRAWS, len(estimate), generator=generator) * errors
    residuals = RESPONSE[None, :] - drawn @ design.T
    likelihood = -0.5 * (residuals / spread) ** 2 - log(spread) - 0.5 * log(2 * pi)

    return to_datatree_from_svi(
        {"coefficients": drawn}, log_likelihood={"y": likelihood}
    )


@pytest.fixture(scope="module")
def grid() -> RefitGrid:
    return RefitGrid(SIGNATURES, RATES, BUDGETS)


@pytest.fixture(scope="module")
def study(grid: RefitGrid) -> RefitStudy:
    """Return the study over real gradient descent at every cell of the grid."""
    return refit_study(
        _refit,
        grid,
        gate=ParetoGate(fraction=0.1),
        rule=StabilityRule(tail=2, threshold=0.9),
        falsifier=FALSIFIER,
        family=FAMILY,
        replicates=REPLICATES,
        seed=SEED,
    )


def test_the_refit_satisfies_the_protocol_laws(grid: RefitGrid) -> None:
    check_refit_laws(
        RefitFixture(
            _refit,
            (grid.cells[0], grid.cells[-1]),
            lambda fit: _pointwise(fit, var_name=None),
        )
    )


def test_the_driver_emits_one_row_per_cell(study: RefitStudy, grid: RefitGrid) -> None:
    emitted = tuple(
        RefitCell(row.signature, row.learning_rate, row.budget) for row in study.cells
    )

    assert emitted == grid.cells


def test_each_signature_is_read_at_its_own_best_configuration(
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


def test_more_budget_does_not_lose_the_grammar_its_fit(study: RefitStudy) -> None:
    reached = [
        next(best.elpd for best in point.best if best.signature == "grammar")
        for point in study.curve
    ]

    assert reached == sorted(reached)


def test_the_ordering_flips_once_the_budget_is_large(study: RefitStudy) -> None:
    assert study.curve[0].ranking[0] == "twin"
    assert study.curve[-1].ranking[0] == "grammar"


def test_the_ranking_is_stable_in_the_limit(study: RefitStudy) -> None:
    assert study.stability.correlation == 1.0
    assert study.stability.settled


def test_the_comparison_is_a_finding_rather_than_a_smoke_test(
    study: RefitStudy,
) -> None:
    assert study.smoke_test is False

    for comparison in study.comparisons:
        assert comparison.smoke_test is False
        assert comparison.falsifier == FALSIFIER
        assert comparison.statistic > 0.0
