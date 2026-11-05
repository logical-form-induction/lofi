import copy
from typing import TYPE_CHECKING, Never

import pytest
import torch
from quivers.inference import MCMC, MCMCResult
from tests.conformance.posterior_fitter import (
    PosteriorFitterConformance,
    PosteriorFitterSuite,
    generated_program,
    small_conformance,
)

from lofi.judgment import (
    NONFINITE_START,
    REFERENCE_NUTS_PLAN,
    InferenceMethod,
    NutsPlan,
    PosteriorFittingError,
    QuiversNutsFitter,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from quivers.continuous import MonadicProgram
    from quivers.inference.guides import Guide


class TestQuiversNutsFitter(PosteriorFitterSuite):
    """Bind the posterior-fitting laws to the Quivers implementation."""

    @pytest.fixture
    def posterior_fitter_conformance(self) -> PosteriorFitterConformance:
        return small_conformance()


def test_the_reference_plan_pins_the_preregistered_protocol() -> None:
    assert REFERENCE_NUTS_PLAN.warmup == 500
    assert REFERENCE_NUTS_PLAN.retained == 1000
    assert REFERENCE_NUTS_PLAN.chains == 4
    assert REFERENCE_NUTS_PLAN.seed == 20261008
    assert QuiversNutsFitter().plan is REFERENCE_NUTS_PLAN
    assert QuiversNutsFitter().method is InferenceMethod.NUTS


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("warmup", 0, "warmup count"),
        ("retained", 3, "at least four"),
        ("chains", 1, "at least two"),
        ("seed", -1, "seed.*negative"),
    ],
)
def test_a_nuts_plan_refuses_counts_that_cannot_supply_diagnostics(
    field: str, value: int, message: str
) -> None:
    arguments = {"warmup": 1, "retained": 4, "chains": 2, "seed": 0}
    arguments[field] = value

    with pytest.raises(ValueError, match=message):
        NutsPlan(**arguments)


@pytest.mark.parametrize("field", ["warmup", "retained", "chains", "seed"])
def test_a_nuts_plan_refuses_boolean_counts(field: str) -> None:
    arguments: dict[str, int | bool] = {
        "warmup": 1,
        "retained": 4,
        "chains": 2,
        "seed": 0,
    }
    arguments[field] = True

    with pytest.raises(TypeError, match="not an integer"):
        NutsPlan(**arguments)


def test_a_nuts_plan_is_immutable() -> None:
    plan = NutsPlan(warmup=1, retained=4, chains=2, seed=0)

    assert copy.copy(plan) is plan
    assert copy.deepcopy(plan) is plan


def test_a_real_nonfinite_initial_position_is_reported_as_a_named_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def nonfinite_initial(
        dimensions: int,
        *,
        generator: torch.Generator | None = None,
    ) -> torch.Tensor:
        del generator
        return torch.full((dimensions,), torch.nan)

    monkeypatch.setattr(torch, "randn", nonfinite_initial)

    with pytest.raises(
        PosteriorFittingError,
        match=rf"chain 0 {NONFINITE_START}",
    ):
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )


def _failing_run(error: BaseException) -> Callable[..., Never]:
    def fail(
        sampler: MCMC,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> Never:
        del sampler, model, x, observations, guide
        raise error

    return fail


@pytest.mark.parametrize(
    "error",
    [RuntimeError("integrator exploded"), TypeError("a site has no shape")],
    ids=["runtime", "type"],
)
def test_a_sampler_defect_propagates_rather_than_as_a_fit_failure(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(MCMC, "run", _failing_run(error))

    with pytest.raises(type(error), match=str(error)):
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )


def test_another_value_error_mentioning_finiteness_propagates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    defect = ValueError("the step size must be finite and positive")
    monkeypatch.setattr(MCMC, "run", _failing_run(defect))

    with pytest.raises(ValueError, match="step size") as raised:
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )

    assert raised.value is defect


def test_an_exhausted_accelerator_is_reported_as_memory_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        MCMC, "run", _failing_run(torch.OutOfMemoryError("CUDA out of memory"))
    )

    with pytest.raises(MemoryError, match="CUDA out of memory") as raised:
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )

    assert not isinstance(raised.value, PosteriorFittingError)


def _sampler_result(*, chains: int, retained: int) -> MCMCResult:
    return MCMCResult(
        samples={"parameter": torch.zeros((chains, retained, 1))},
        log_densities=torch.zeros((chains, retained)),
        acceptance_rates=torch.ones(chains),
        divergence_counts=torch.zeros(chains, dtype=torch.int64),
        r_hat={"parameter": torch.ones(1)},
        ess={"parameter": torch.full((1,), float(chains * retained))},
        num_warmup=1,
        num_samples=retained,
    )


@pytest.mark.parametrize(("chains", "retained"), [(1, 4), (2, 5)])
def test_a_sampler_result_must_match_the_planned_chain_and_draw_axes(
    chains: int,
    retained: int,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def mismatch(
        sampler: MCMC,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> MCMCResult:
        del sampler, model, x, observations, guide
        return _sampler_result(chains=chains, retained=retained)

    monkeypatch.setattr(MCMC, "run", mismatch)

    with pytest.raises(ValueError, match=r"chain/draw axes.*not the plan's \(2, 4\)"):
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )


def _returning(result: MCMCResult) -> Callable[..., MCMCResult]:
    def run(
        sampler: MCMC,
        model: MonadicProgram,
        x: torch.Tensor,
        observations: dict[str, torch.Tensor],
        guide: Guide | None = None,
    ) -> MCMCResult:
        del sampler, model, x, observations, guide
        return result

    return run


def test_a_chain_whose_every_transition_diverged_is_a_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _sampler_result(chains=2, retained=4)
    diverged = MCMCResult(
        samples=result.samples,
        log_densities=result.log_densities,
        acceptance_rates=result.acceptance_rates,
        divergence_counts=torch.tensor([4, 0], dtype=torch.int64),
        r_hat=result.r_hat,
        ess=result.ess,
        num_warmup=1,
        num_samples=4,
    )
    monkeypatch.setattr(MCMC, "run", _returning(diverged))

    with pytest.raises(PosteriorFittingError, match="1 chains diverged"):
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )


def test_a_non_finite_convergence_diagnostic_is_a_fit_failure(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    result = _sampler_result(chains=2, retained=4)
    unconverged = MCMCResult(
        samples=result.samples,
        log_densities=result.log_densities,
        acceptance_rates=result.acceptance_rates,
        divergence_counts=result.divergence_counts,
        r_hat={"parameter": torch.full((1,), torch.nan)},
        ess=result.ess,
        num_warmup=1,
        num_samples=4,
    )
    monkeypatch.setattr(MCMC, "run", _returning(unconverged))

    with pytest.raises(PosteriorFittingError, match="not finite"):
        QuiversNutsFitter(NutsPlan(warmup=1, retained=4, chains=2, seed=0)).fit(
            generated_program()
        )
