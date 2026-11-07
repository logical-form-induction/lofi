from typing import TYPE_CHECKING

import pytest
import torch

from lofi.judgment import DesignCodec, JudgmentRow, RowColumns
from lofi.judgment.crossed import (
    CredibleIntervals,
    CrossedModel,
    CrossedPosterior,
    CrossedScales,
    PosteriorDraws,
    coverage_of,
    crossed_model_of,
    exact_posterior,
    sample_posterior,
    simulate_crossed,
)

if TYPE_CHECKING:
    from torch import Tensor

LEVELS = 5
READOUTS = ("response", "length")
ROWS = 240
PARTICIPANTS = 12
ITEMS = 9
COEFFICIENTS = 3
SCALES = CrossedScales(0.8, 2.0, (0.7, 0.5))


def design_of(rows: int, participants: int, items: int) -> CrossedModel:
    simulated = simulate_crossed(
        *columns(rows, participants, items),
        SCALES,
        generator=torch.Generator().manual_seed(4),
    )

    return simulated.model


def columns(
    rows: int, participants: int, items: int
) -> tuple[Tensor, tuple[Tensor, Tensor], tuple[int, int]]:
    generator = torch.Generator().manual_seed(1)
    fixed = torch.randn((rows, COEFFICIENTS), generator=generator, dtype=torch.float64)
    fixed[:, 0] = 1.0
    participant = torch.randint(0, participants, (rows,), generator=generator)
    item = torch.randint(0, items, (rows,), generator=generator)

    return fixed, (participant, item), (participants, items)


def log_posterior(model: CrossedModel, parameters: Tensor) -> Tensor:
    fixed = parameters[: model.n_fixed]
    effects = parameters[model.n_fixed :]
    predictor = model.fixed @ fixed + effects[model.stacked_indices].sum(dim=1)
    residual = ((model.responses - predictor) ** 2).sum() / model.scales.residual**2
    penalty = (fixed**2).sum() / model.scales.fixed**2
    spread = (model.effect_prior_precisions * effects**2).sum()

    return -0.5 * (residual + penalty + spread)


def test_a_scale_that_is_not_positive_is_refused() -> None:
    with pytest.raises(ValueError, match="not positive"):
        CrossedScales(0.0, 1.0, (1.0,))

    with pytest.raises(ValueError, match="not positive"):
        CrossedScales(1.0, -1.0, (1.0,))

    with pytest.raises(ValueError, match="not positive"):
        CrossedScales(1.0, 1.0, (1.0, 0.0))


def test_two_scales_written_the_same_way_are_one_key() -> None:
    left = CrossedScales(0.5, 2.0, (1.0, 0.25))
    right = CrossedScales(0.5, 2.0, (1.0, 0.25))
    other = CrossedScales(0.5, 2.0, (1.0, 0.5))

    assert left == right
    assert hash(left) == hash(right)
    assert left != other
    assert len({left, right, other}) == 2


def test_a_model_lays_its_parameters_out_fixed_first_then_factor_by_factor() -> None:
    model = design_of(ROWS, PARTICIPANTS, ITEMS)

    assert model.n_rows == ROWS
    assert model.n_fixed == COEFFICIENTS
    assert model.n_effects == PARTICIPANTS + ITEMS
    assert model.n_parameters == COEFFICIENTS + PARTICIPANTS + ITEMS
    assert model.group_sizes == (PARTICIPANTS, ITEMS)
    assert model.group_offsets == (0, PARTICIPANTS)
    assert int(model.effect_counts.sum()) == ROWS * len(model.group_sizes)
    assert torch.allclose(
        model.effect_prior_precisions[:PARTICIPANTS],
        torch.full((PARTICIPANTS,), 1.0 / SCALES.groups[0] ** 2, dtype=torch.float64),
    )
    assert torch.allclose(
        model.effect_prior_precisions[PARTICIPANTS:],
        torch.full((ITEMS,), 1.0 / SCALES.groups[1] ** 2, dtype=torch.float64),
    )


def test_a_model_refuses_a_level_outside_its_factor() -> None:
    fixed, groups, sizes = columns(ROWS, PARTICIPANTS, ITEMS)
    beyond = groups[0].clone()
    beyond[0] = PARTICIPANTS

    with pytest.raises(ValueError, match="holds levels"):
        CrossedModel(
            fixed,
            (beyond, groups[1]),
            sizes,
            torch.zeros((ROWS,), dtype=torch.float64),
            SCALES,
        )


def test_a_model_refuses_columns_that_do_not_line_up() -> None:
    fixed, groups, sizes = columns(ROWS, PARTICIPANTS, ITEMS)

    with pytest.raises(ValueError, match="one floating response per row"):
        CrossedModel(
            fixed, groups, sizes, torch.zeros((ROWS - 1,), dtype=torch.float64), SCALES
        )

    with pytest.raises(ValueError, match="one integer level per row"):
        CrossedModel(
            fixed,
            (groups[0].to(dtype=torch.float64), groups[1]),
            sizes,
            torch.zeros((ROWS,), dtype=torch.float64),
            SCALES,
        )

    with pytest.raises(ValueError, match="one scale per grouping factor"):
        CrossedModel(
            fixed,
            (groups[0],),
            (PARTICIPANTS,),
            torch.zeros((ROWS,), dtype=torch.float64),
            SCALES,
        )


def test_a_model_refuses_a_response_that_is_not_finite() -> None:
    fixed, groups, sizes = columns(ROWS, PARTICIPANTS, ITEMS)
    responses = torch.zeros((ROWS,), dtype=torch.float64)
    responses[3] = float("nan")

    with pytest.raises(ValueError, match="finite"):
        CrossedModel(fixed, groups, sizes, responses, SCALES)


def test_an_interval_holds_the_values_between_its_bounds() -> None:
    intervals = CredibleIntervals(
        torch.tensor([-1.0, 0.0]), torch.tensor([1.0, 2.0]), 0.9
    )

    assert intervals.level == 0.9
    assert intervals.covers(torch.tensor([0.5, 0.0])).tolist() == [True, True]
    assert intervals.covers(torch.tensor([-2.0, 3.0])).tolist() == [False, False]

    with pytest.raises(ValueError, match="one value each"):
        intervals.covers(torch.tensor([0.0]))


def test_an_interval_whose_bounds_are_the_wrong_way_round_is_refused() -> None:
    with pytest.raises(ValueError, match="above its upper bound"):
        CredibleIntervals(torch.tensor([1.0]), torch.tensor([0.0]), 0.9)

    with pytest.raises(ValueError, match="not a proportion"):
        CredibleIntervals(torch.tensor([0.0]), torch.tensor([1.0]), 1.0)


def test_a_posterior_refuses_a_variance_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="not positive"):
        CrossedPosterior(torch.zeros(2), torch.tensor([1.0, 0.0]))

    with pytest.raises(ValueError, match="one mean and one variance"):
        CrossedPosterior(torch.zeros(2), torch.ones(3))


def test_a_gaussian_interval_reaches_the_standard_normal_quantile() -> None:
    posterior = CrossedPosterior(
        torch.tensor([1.0, -2.0], dtype=torch.float64),
        torch.tensor([4.0, 0.25], dtype=torch.float64),
    )
    intervals = posterior.intervals(0.9)
    quantile = torch.distributions.Normal(0.0, 1.0).icdf(torch.tensor(0.95))
    half = quantile * posterior.variances.sqrt()

    assert torch.allclose(intervals.lower, posterior.means - half)
    assert torch.allclose(intervals.upper, posterior.means + half)


def test_an_empirical_interval_approaches_the_gaussian_one_it_was_drawn_from() -> None:
    generator = torch.Generator().manual_seed(9)
    values = torch.randn((40000, 2), generator=generator, dtype=torch.float64)
    values = values * torch.tensor([2.0, 0.5], dtype=torch.float64)
    empirical = PosteriorDraws(values).intervals(0.9)
    gaussian = CrossedPosterior(
        torch.zeros(2, dtype=torch.float64),
        torch.tensor([4.0, 0.25], dtype=torch.float64),
    ).intervals(0.9)

    assert torch.allclose(empirical.lower, gaussian.lower, atol=0.1)
    assert torch.allclose(empirical.upper, gaussian.upper, atol=0.1)

    with pytest.raises(ValueError, match="at least one row"):
        PosteriorDraws(torch.zeros((0, 2)))


def test_the_exact_posterior_sits_where_the_log_posterior_is_stationary() -> None:
    model = design_of(60, 4, 3)
    posterior = exact_posterior(model)
    parameters = posterior.means.detach().clone().requires_grad_()

    log_posterior(model, parameters).backward()
    gradient = parameters.grad

    assert gradient is not None
    assert float(gradient.abs().max()) < 1e-8


def test_the_exact_variances_are_the_diagonal_of_the_inverse_curvature() -> None:
    model = design_of(60, 4, 3)
    posterior = exact_posterior(model)
    curvature = torch.autograd.functional.hessian(
        lambda parameters: log_posterior(model, parameters),
        posterior.means.detach().clone(),
    )

    assert torch.allclose(
        posterior.variances, torch.diagonal(torch.linalg.inv(-curvature)), atol=1e-10
    )


def test_the_sampler_reproduces_the_posterior_it_draws_from() -> None:
    model = design_of(ROWS, PARTICIPANTS, ITEMS)
    posterior = exact_posterior(model)
    draws = sample_posterior(
        model, draws=20000, generator=torch.Generator().manual_seed(7)
    )

    assert draws.n_draws == 20000
    assert draws.n_parameters == model.n_parameters
    standardized = (
        draws.values.mean(dim=0) - posterior.means
    ) / posterior.variances.sqrt()

    assert float(standardized.abs().max()) < 0.05
    assert torch.allclose(
        draws.values.var(dim=0), posterior.variances, rtol=0.05, atol=0.0
    )


def test_one_seed_fixes_the_sample_and_the_simulation() -> None:
    model = design_of(60, 4, 3)
    first = sample_posterior(
        model, draws=16, generator=torch.Generator().manual_seed(2)
    )
    second = sample_posterior(
        model, draws=16, generator=torch.Generator().manual_seed(2)
    )
    other = sample_posterior(
        model, draws=16, generator=torch.Generator().manual_seed(3)
    )

    assert torch.equal(first.values, second.values)
    assert not torch.equal(first.values, other.values)

    with pytest.raises(ValueError, match="at least one draw"):
        sample_posterior(model, draws=0, generator=torch.Generator().manual_seed(2))


def test_a_simulated_model_carries_the_parameters_its_responses_came_from() -> None:
    simulated = simulate_crossed(
        *columns(ROWS, PARTICIPANTS, ITEMS),
        SCALES,
        generator=torch.Generator().manual_seed(6),
    )

    assert simulated.truth.shape == (simulated.model.n_parameters,)
    assert bool(torch.isfinite(simulated.model.responses).all())

    predictor = simulated.model.fixed @ simulated.truth[
        : simulated.model.n_fixed
    ] + simulated.truth[simulated.model.n_fixed :][simulated.model.stacked_indices].sum(
        dim=1
    )
    residual = simulated.model.responses - predictor

    assert float(residual.std()) == pytest.approx(SCALES.residual, rel=0.2)


def test_coverage_counts_the_parameters_an_interval_holds() -> None:
    intervals = CredibleIntervals(
        torch.zeros(4, dtype=torch.float64), torch.ones(4, dtype=torch.float64), 0.9
    )
    values = torch.tensor([0.5, 0.25, 2.0, -1.0], dtype=torch.float64)

    assert coverage_of(intervals, values) == 0.5


def test_the_exact_posterior_covers_a_drawn_truth_at_about_its_level() -> None:
    generator = torch.Generator().manual_seed(21)
    fixed, groups, sizes = columns(ROWS, PARTICIPANTS, ITEMS)
    covered = [
        coverage_of(
            exact_posterior(simulated.model).intervals(0.9),
            simulated.truth,
        )
        for simulated in (
            simulate_crossed(fixed, groups, sizes, SCALES, generator=generator)
            for _ in range(40)
        )
    ]

    assert sum(covered) / len(covered) == pytest.approx(0.9, abs=0.03)


def test_a_model_read_from_a_design_takes_its_participants_and_items() -> None:
    codec = DesignCodec(LEVELS, (), READOUTS)
    rows = tuple(
        JudgmentRow(
            f"participant-{index % 4}",
            f"item-{index % 3}",
            index % LEVELS,
            RowColumns(
                factors={},
                readouts={"response": float(index % LEVELS), "length": float(index)},
            ),
            None,
        )
        for index in range(24)
    )
    design = codec.encode(rows)
    model = crossed_model_of(
        design, SCALES, response="response", covariates=("length",)
    )

    assert model.n_rows == 24
    assert model.n_fixed == 2
    assert model.group_sizes == (4, 3)
    assert torch.equal(model.fixed[:, 0], torch.ones(24, dtype=torch.float64))
    assert torch.equal(
        model.fixed[:, 1], design.readouts["length"].to(dtype=torch.float64)
    )
    assert torch.equal(model.responses, design.readouts["response"].to(torch.float64))

    with pytest.raises(ValueError, match="were asked for"):
        crossed_model_of(design, SCALES, response="absent")
