"""The pinned Quivers NUTS treatment of compiled response programs."""

from typing import Final, Literal, final

import torch
from quivers.inference import MCMC, MCMCResult, NUTSKernel

from lofi._frozen import FrozenValue
from lofi.judgment._ports import PosteriorFitter, PosteriorFittingError
from lofi.judgment._results import NutsPlan, PosteriorFit
from lofi.judgment._spec import InferenceMethod
from lofi.judgment.compiler import BoundProgram
from lofi.judgment.runtime import QuiversRuntime

__all__ = [
    "NONFINITE_START",
    "REFERENCE_NUTS_PLAN",
    "QuiversNutsFitter",
    "ensure_nuts_fitter",
]


REFERENCE_NUTS_PLAN: Final = NutsPlan(
    warmup=500,
    retained=1000,
    chains=4,
    seed=20261008,
)
"""The preregistered posterior-fit benchmark plan."""

NONFINITE_START: Final = "initial_position must contain only finite values"
"""The words Quivers' ``ValueError`` uses for a chain's non-finite starting point.

Quivers exposes no typed signal for this refusal, so it is recognized by these
words, which both its driver and its kernel use.
"""


def _posterior_fit(result: MCMCResult, plan: NutsPlan) -> PosteriorFit:
    """Translate and validate the public Quivers result against its plan."""
    fitted = PosteriorFit(
        draws=result.samples,
        log_densities=result.log_densities,
        acceptance_rates=result.acceptance_rates,
        divergence_counts=result.divergence_counts,
        r_hat=result.r_hat,
        effective_sample_sizes=result.ess,
    )
    expected = (plan.chains, plan.retained)
    observed = (fitted.chains, fitted.retained)
    if observed != expected:
        msg = f"NUTS returned chain/draw axes {observed!r}, not the plan's {expected!r}"
        raise ValueError(msg)
    return fitted


@final
class QuiversNutsFitter(FrozenValue):
    """Fit a bound response program with Quivers' public NUTS engine."""

    _noun = "Quivers NUTS fitter"

    __slots__ = ("_plan",)

    def __init__(self, plan: NutsPlan = REFERENCE_NUTS_PLAN) -> None:
        if not isinstance(plan, NutsPlan):
            msg = f"a Quivers NUTS fitter's plan is not a NutsPlan: {plan!r}"
            raise TypeError(msg)
        object.__setattr__(self, "_plan", plan)

    @property
    def method(self) -> Literal[InferenceMethod.NUTS]:
        """Return genuine NUTS as the sole inference method."""
        return InferenceMethod.NUTS

    @property
    def plan(self) -> NutsPlan:
        """Return the complete immutable sampling plan."""
        return self._plan

    def fit(self, program: BoundProgram, /) -> PosteriorFit:
        """Draw and diagnose the posterior of a bound response program.

        A fresh kernel and driver are constructed for every invocation. This
        makes repeated fits independent of adaptation state while Quivers
        derives independent chain streams from the plan's root seed.

        Quivers absorbs a non-finite density along a trajectory as a divergent
        transition and refuses a non-finite initial position with a
        ``ValueError`` whose message holds ``NONFINITE_START``. Those
        numerical conditions, a chain whose
        every retained transition diverged, and a non-finite draw, density or
        convergence diagnostic are reported as a ``PosteriorFittingError``.
        An exhausted accelerator is reported as a ``MemoryError``. Any other
        error is a defect and propagates.

        Raises
        ------
        PosteriorFittingError
            If the sampler reports a numerical condition.
        MemoryError
            If the accelerator runs out of memory.
        """
        if not isinstance(program, BoundProgram):
            msg = f"a Quivers NUTS fitter received no bound program: {program!r}"
            raise TypeError(msg)

        try:
            return self._checked(self._sample(program))

        except torch.OutOfMemoryError as error:
            exhausted = str(error)

        except ValueError as error:
            if NONFINITE_START not in str(error):
                raise

            msg = f"NUTS posterior fitting failed: {error}"
            raise PosteriorFittingError(msg) from error

        raise MemoryError(exhausted)

    def _checked(self, result: MCMCResult, /) -> PosteriorFit:
        """Return a sampled result's posterior, refusing a numerical condition."""
        diverged = int((result.divergence_counts >= self._plan.retained).sum())

        if diverged:
            msg = (
                f"NUTS posterior fitting failed: every retained transition of "
                f"{diverged} chains diverged"
            )
            raise PosteriorFittingError(msg)

        tensors = (
            result.log_densities,
            *result.samples.values(),
            *result.r_hat.values(),
            *result.ess.values(),
        )

        if not all(bool(torch.isfinite(tensor).all()) for tensor in tensors):
            msg = (
                "NUTS posterior fitting failed: a draw, density or convergence "
                "diagnostic is not finite"
            )
            raise PosteriorFittingError(msg)

        return _posterior_fit(result, self._plan)

    def _sample(self, program: BoundProgram, /) -> MCMCResult:
        """Run the pinned NUTS driver over a bound program."""
        kernel = NUTSKernel(
            step_size=0.1,
            max_tree_depth=10,
            mass_matrix="diagonal",
            target_accept=0.8,
            divergence_threshold=1000.0,
            adapt_step_size=True,
            adapt_mass_matrix=True,
        )
        sampler = MCMC(
            kernel,
            num_warmup=self._plan.warmup,
            num_samples=self._plan.retained,
            num_chains=self._plan.chains,
            init_strategy="prior",
            seed=self._plan.seed,
        )
        model = QuiversRuntime().program(program)

        return sampler.run(
            model,
            torch.zeros((1, 1), dtype=torch.float64),
            dict(program.host_data),
        )


def ensure_nuts_fitter(value: PosteriorFitter, /) -> PosteriorFitter:
    """Return a posterior fitter only when it declares genuine NUTS.

    The posterior-fitting protocol admits multiple sampling methods. This
    boundary narrows that declared capability to the one NUTS treatment and
    verifies the immutable plan needed to reproduce it.

    Parameters
    ----------
    value : PosteriorFitter
        Declared posterior fitter.

    Returns
    -------
    PosteriorFitter
        Validated genuine-NUTS fitter.

    Raises
    ------
    TypeError
        If the fitter has no immutable plan.
    ValueError
        If the candidate declares a non-NUTS inference method.
    """
    if value.method is not InferenceMethod.NUTS:
        msg = f"a posterior fitter must run genuine NUTS, not {value.method.value!r}"
        raise ValueError(msg)
    if not isinstance(value.plan, NutsPlan):
        msg = f"a posterior fitter has no immutable NutsPlan: {value.plan!r}"
        raise TypeError(msg)
    return value
