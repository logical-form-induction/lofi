"""Conformance checks for genuine NUTS posterior fitters."""

from types import MappingProxyType
from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple

import pytest
import torch

from lofi.judgment import (
    LINEAR,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    FixedDiscrimination,
    InferenceMethod,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    NutsPlan,
    PosteriorFit,
    PosteriorFitter,
    PriorBlock,
    QuiversNutsFitter,
    RandomIntercept,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    linear_steps,
    sample_responses,
    simulated_design,
)
from lofi.judgment.compiler import ITEM, PARTICIPANT, BoundProgram

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "CONFORMANCE_GENERATING_VALUES",
    "CONFORMANCE_RECOVERY_INTERVALS",
    "PosteriorFitterConformance",
    "PosteriorFitterSuite",
    "generated_program",
    "reference_program",
]

SCORE = "score"
LEVELS = 7
CONFORMANCE_GENERATING_VALUES: Final[Mapping[str, tuple[float, ...]]] = (
    MappingProxyType(
        {
            "score_slope": (1.25,),
            "cutpoint_base": (-3.0,),
            "cutpoint_increments": (0.55,) * (LEVELS - 1),
        }
    )
)
"""Known values for every parameter group in the fitted conformance fixture."""

CONFORMANCE_RECOVERY_INTERVALS: Final[Mapping[str, tuple[float, float]]] = (
    MappingProxyType(
        {
            "score_slope": (0.0, 2.5),
            "cutpoint_base": (-6.0, 0.0),
            "cutpoint_increments": (-3.0, 3.0),
        }
    )
)
"""Broad preregistered intervals for posterior means in every fitted group."""


def _spec(*, crossed: bool) -> ResponseSpec:
    predictors = [
        LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),
    ]
    if crossed:
        predictors.extend(
            (
                RandomIntercept(group=PARTICIPANT),
                RandomIntercept(group=ITEM),
            )
        )
    return ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=tuple(predictors),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


def _unobserved_program(
    *, participants: int, items: int, crossed: bool
) -> tuple[BoundProgram, JudgmentDesign]:
    codec = DesignCodec(LEVELS, factors=(), readouts=(SCORE,))
    design = codec.encode(
        JudgmentRow(
            f"participant-{participant}",
            f"item-{item}",
            0,
            RowColumns(
                factors={},
                readouts={SCORE: -2.0 + 4.0 * item / max(items - 1, 1)},
            ),
        )
        for participant in range(participants)
        for item in range(items)
    )
    program = ResponseCompiler({LINEAR: linear_steps}).emit(
        _spec(crossed=crossed), design
    )
    return program, design


def _generating_vector(program: BoundProgram) -> torch.Tensor:
    pieces = []
    for slot in program.layout:
        match slot.site:
            case "score_slope":
                piece = torch.tensor(CONFORMANCE_GENERATING_VALUES[slot.site])
            case "participant_intercept_scale" | "item_intercept_scale":
                piece = torch.tensor([0.2])
            case "participant_intercept_offsets":
                piece = torch.linspace(-0.5, 0.5, slot.size)
            case "item_intercept_offsets":
                piece = torch.linspace(-0.75, 0.75, slot.size)
            case "cutpoint_base":
                piece = torch.tensor(CONFORMANCE_GENERATING_VALUES[slot.site])
            case "cutpoint_increments":
                piece = torch.tensor(CONFORMANCE_GENERATING_VALUES[slot.site])
            case "participant_cutpoint_spread":
                piece = torch.full((slot.size,), 0.1)
            case "participant_cutpoint_offsets":
                piece = torch.zeros(slot.size)
            case _:
                msg = f"the generated fixture has no value for site {slot.site!r}"
                raise ValueError(msg)
        pieces.append(piece)
    return torch.cat(tuple(pieces)).to(torch.float64)


def generated_program() -> BoundProgram:
    """Return the fast deterministic generated fixture for conformance."""
    program, design = _unobserved_program(participants=2, items=4, crossed=False)
    responses = sample_responses(
        program,
        _generating_vector(program),
        generator=torch.Generator().manual_seed(77),
    )
    return ResponseCompiler({LINEAR: linear_steps}).emit(
        _spec(crossed=False), simulated_design(design, responses)
    )


def reference_program() -> BoundProgram:
    """Return the pinned 8-by-16 crossed generated reference fixture."""
    program, design = _unobserved_program(participants=8, items=16, crossed=True)
    responses = sample_responses(
        program,
        _generating_vector(program),
        generator=torch.Generator().manual_seed(20261008),
    )
    return ResponseCompiler({LINEAR: linear_steps}).emit(
        _spec(crossed=True), simulated_design(design, responses)
    )


class PosteriorFitterConformance(NamedTuple):
    """One fitter, a changed-seed fitter, and their generated program."""

    fitter: PosteriorFitter
    changed_seed: PosteriorFitter
    program: BoundProgram

    def run(self) -> tuple[PosteriorFit, PosteriorFit, PosteriorFit]:
        """Run the repeatability, seed-sensitivity, and RNG-isolation laws."""
        state = torch.random.get_rng_state()
        first = self.fitter.fit(self.program)
        again = self.fitter.fit(self.program)
        other = self.changed_seed.fit(self.program)
        current = torch.random.get_rng_state()
        if not torch.equal(state, current):
            msg = "a posterior fitter changed Torch's global random state"
            raise AssertionError(msg)
        return first, again, other


class PosteriorFitterSuite:
    """Reusable laws for posterior-fitting implementations."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_fit_is_repeatable_seed_sensitive_and_rng_isolated(
        self,
        posterior_fitter_conformance: PosteriorFitterConformance,
    ) -> None:
        """Exercise the posterior fit, diagnostic, and recovery laws."""
        conformance = posterior_fitter_conformance
        first, again, other = conformance.run()
        chains = conformance.fitter.plan.chains
        retained = conformance.fitter.plan.retained

        assert isinstance(conformance.fitter, PosteriorFitter)
        assert conformance.fitter.method is InferenceMethod.NUTS
        assert first.chains == chains
        assert first.retained == retained
        assert all(
            draws.shape[:2] == (chains, retained) for draws in first.draws.values()
        )
        assert first.divergence_counts.shape == (chains,)
        assert first.divergence_counts.dtype is torch.int64
        assert first.acceptance_rates.shape == (chains,)
        assert first.diagnostics.method is InferenceMethod.NUTS
        assert first.diagnostics.divergences is not None
        assert first.diagnostics.max_rhat is not None
        assert first.diagnostics.min_ess is not None
        assert all(
            torch.equal(first.draws[name], again.draws[name]) for name in first.draws
        )
        assert any(
            not torch.equal(first.draws[name], other.draws[name])
            for name in first.draws
        )
        assert frozenset(first.draws) == frozenset(CONFORMANCE_GENERATING_VALUES)
        assert frozenset(first.draws) == frozenset(CONFORMANCE_RECOVERY_INTERVALS)
        for name, draws in first.draws.items():
            lower, upper = CONFORMANCE_RECOVERY_INTERVALS[name]
            truth = torch.tensor(CONFORMANCE_GENERATING_VALUES[name])
            posterior_means = draws.mean(dim=(0, 1)).reshape(-1)

            assert bool(((truth >= lower) & (truth <= upper)).all())
            assert bool(((posterior_means >= lower) & (posterior_means <= upper)).all())


def small_conformance() -> PosteriorFitterConformance:
    """Return the Quivers binding at small diagnostic-valid plans."""
    return PosteriorFitterConformance(
        QuiversNutsFitter(NutsPlan(warmup=5, retained=8, chains=2, seed=91)),
        QuiversNutsFitter(NutsPlan(warmup=5, retained=8, chains=2, seed=92)),
        generated_program(),
    )
