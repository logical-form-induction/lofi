import copy
from typing import override

import pytest
import torch
from tests.conformance.posterior_fitter import generated_program

from lofi.judgment import (
    PARTICIPANT,
    BoundProgram,
    ConditionalPointObjective,
    Cumulative,
    DesignCodec,
    FixedDiscrimination,
    InferenceMethod,
    JudgmentRow,
    MapFitter,
    MapInference,
    MarginalPointObjective,
    OffsetTerm,
    PointFit,
    PointFitter,
    PointFittingError,
    PointObjective,
    PriorBlock,
    RandomIntercept,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    Slot,
    ThresholdPriors,
    TruncatedMarginalLikelihood,
)
from lofi.spec.optimizers.adam import AdamOptimizerSpec


def plan(*, steps: int = 3) -> MapInference:
    return MapInference(
        optimizer=AdamOptimizerSpec(learning_rate=0.02),
        steps=steps,
        initial_scale=0.1,
    )


class QuadraticObjective:
    """A one-coordinate support-preserving objective for fitter laws."""

    @property
    def layout(self) -> tuple[Slot, ...]:
        return (Slot("rate", 1),)

    def initial_position(
        self, *, generator: torch.Generator, scale: float
    ) -> torch.Tensor:
        return scale * torch.randn(1, generator=generator, dtype=torch.float64)

    def parameters(self, position: torch.Tensor, /) -> torch.Tensor:
        return position.exp()

    def log_density(self, parameters: torch.Tensor, /) -> torch.Tensor:
        return -((parameters - 1.5) ** 2).sum()


class NonfiniteObjective(QuadraticObjective):
    @override
    def log_density(self, parameters: torch.Tensor, /) -> torch.Tensor:
        return parameters.new_full((), torch.nan)


class NonfiniteInitialObjective(QuadraticObjective):
    @override
    def initial_position(
        self, *, generator: torch.Generator, scale: float
    ) -> torch.Tensor:
        del generator, scale
        return torch.full((1,), torch.nan, dtype=torch.float64)


class NonfiniteGradientObjective(QuadraticObjective):
    @override
    def log_density(self, parameters: torch.Tensor, /) -> torch.Tensor:
        zero = (parameters - parameters).sum()
        return -torch.sqrt(zero)


def marginal_program() -> BoundProgram:
    declaration = ResponseSpec(
        family=Cumulative(),
        levels=3,
        objective=TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=3.0,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8,
            quadrature_points=8,
            quadrature_seed=23,
        ),
        predictor=(
            RandomIntercept(group=PARTICIPANT),
            OffsetTerm(readout="score"),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    rows = (
        JudgmentRow(
            f"p-{index % 2}",
            f"i-{index}",
            index % 3,
            RowColumns({}, {"score": (-0.3, 0.1, 0.5, -0.2)[index]}),
        )
        for index in range(4)
    )
    design = DesignCodec(3, readouts=("score",)).encode(rows)

    return ResponseCompiler({}).emit(declaration, design)


def test_a_point_fit_owns_an_immutable_finite_parameter_snapshot() -> None:
    source = torch.tensor([1.0, 2.0], requires_grad=True)
    fitted = PointFit(parameters=source, log_density=-3.0, steps=4)
    first = fitted.parameters
    first.fill_(99.0)
    source.data.fill_(88.0)

    assert torch.equal(fitted.parameters, torch.tensor([1.0, 2.0]))
    assert fitted.parameters.device.type == "cpu"
    assert fitted.log_density == -3.0
    assert fitted.steps == 4
    assert copy.copy(fitted) is fitted
    assert copy.deepcopy(fitted) is fitted


@pytest.mark.parametrize(
    ("parameters", "log_density", "message"),
    [
        (torch.zeros((1, 1)), 0.0, "not one flat"),
        (torch.tensor([torch.nan]), 0.0, "nonfinite"),
        (torch.zeros(1), torch.inf, "not finite"),
    ],
)
def test_a_point_fit_refuses_an_invalid_result(
    parameters: torch.Tensor, log_density: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        PointFit(parameters=parameters, log_density=log_density, steps=1)


def test_conditional_coordinates_follow_the_full_layout_and_prior_support() -> None:
    program = generated_program()
    objective = ConditionalPointObjective(program)
    position = objective.initial_position(
        generator=torch.Generator().manual_seed(7), scale=0.1
    )
    parameters = objective.parameters(position)
    sites = program.sites(parameters)

    assert isinstance(objective, PointObjective)
    assert objective.layout == program.layout
    assert parameters.shape == (program.size,)
    assert bool(sites["score_slope"] > 0.0)
    assert bool(torch.isfinite(objective.log_density(parameters)))


def test_marginal_coordinates_follow_only_the_population_layout_and_support() -> None:
    program = marginal_program()
    objective = MarginalPointObjective(program, active_groups=(0, 1))
    position = objective.initial_position(
        generator=torch.Generator().manual_seed(11), scale=0.1
    )
    parameters = objective.parameters(position)
    names = tuple(slot.site for slot in objective.layout)
    pieces = torch.split(parameters, [slot.size for slot in objective.layout])
    sites = dict(zip(names, pieces, strict=True))

    assert "participant_intercept_offsets" not in names
    assert "participant_cutpoint_offsets" not in names
    assert "participant_intercept_scale" in names
    assert bool((sites["participant_intercept_scale"] > 0.0).all())
    assert bool(torch.isfinite(objective.log_density(parameters)))


def test_a_map_fit_is_repeatable_seed_sensitive_and_rng_isolated() -> None:
    objective = QuadraticObjective()
    state = torch.random.get_rng_state().clone()
    first = MapFitter(plan(), seed=5).fit(objective)
    again = MapFitter(plan(), seed=5).fit(objective)
    other = MapFitter(plan(), seed=6).fit(objective)

    assert isinstance(MapFitter(plan(), seed=5), PointFitter)
    assert MapFitter(plan(), seed=5).method is InferenceMethod.MAP
    assert torch.equal(first.parameters, again.parameters)
    assert not torch.equal(first.parameters, other.parameters)
    assert first.steps == plan().steps
    assert first.parameters.item() > 0.0
    assert torch.equal(torch.random.get_rng_state(), state)


def test_a_real_conditional_program_is_fitted_in_its_owner_layout() -> None:
    objective = ConditionalPointObjective(generated_program())
    fitted = MapFitter(plan(steps=2), seed=17).fit(objective)

    assert fitted.parameters.numel() == sum(slot.size for slot in objective.layout)
    assert fitted.steps == 2


def test_a_real_marginal_program_is_fitted_in_its_population_layout() -> None:
    objective = MarginalPointObjective(marginal_program(), active_groups=(0, 1))
    fitted = MapFitter(plan(steps=1), seed=19).fit(objective)

    assert fitted.parameters.numel() == sum(slot.size for slot in objective.layout)
    assert fitted.steps == 1


def test_a_nonfinite_density_is_a_named_point_fitting_error() -> None:
    with pytest.raises(PointFittingError, match="nonfinite log density at step 1"):
        MapFitter(plan(), seed=3).fit(NonfiniteObjective())


def test_a_nonfinite_initial_position_is_a_named_point_fitting_error() -> None:
    with pytest.raises(PointFittingError, match="nonfinite initial position"):
        MapFitter(plan(), seed=3).fit(NonfiniteInitialObjective())


def test_a_nonfinite_gradient_is_a_named_point_fitting_error() -> None:
    with pytest.raises(PointFittingError, match="nonfinite gradient at step 1"):
        MapFitter(plan(), seed=3).fit(NonfiniteGradientObjective())


def test_each_objective_refuses_the_other_likelihood_treatment() -> None:
    conditional = generated_program()
    marginal = marginal_program()

    with pytest.raises(ValueError, match="conditional program"):
        ConditionalPointObjective(marginal)

    with pytest.raises(ValueError, match="truncated_marginal"):
        MarginalPointObjective(conditional, active_groups=(0, 1))
