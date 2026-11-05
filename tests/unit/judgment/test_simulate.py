import math
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from torch.nn import functional

from lofi.judgment import (
    COMPLEMENT,
    LOGIT,
    AuxiliaryParameter,
    BoundProgram,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    FactorTerm,
    FixedCutpointVariance,
    FixedDiscrimination,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    Name,
    ParameterFormula,
    PriorBlock,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    TriangularTransform,
    backends,
    evaluate,
    log_category_probabilities,
    log_likelihood,
    logit_steps,
    sample_responses,
    simulate,
    simulated_design,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

LOG_MASS: Final = "p_gram"
LOG_COMPLEMENT: Final = "p_gram_complement"
LEVELS: Final = 7
ROWS: Final = 600
PARTICIPANTS: Final = 12
ITEMS: Final = 50
VERBS: Final = 5
REPLICATES: Final = 200
LEVEL: Final = 0.9
BINOMIAL_LEVEL: Final = 0.99
NORMAL_QUANTILE: Final = 1.6448536269514722
SPEC: Final = ResponseSpec(
    family=Cumulative(),
    levels=LEVELS,
    objective=ConditionalLikelihood(),
    predictor=(
        LinkTerm(
            readout=LOG_MASS,
            linking=LinkingSpec(family=LOGIT, columns={COMPLEMENT: LOG_COMPLEMENT}),
        ),
    ),
    identification=FixedDiscrimination(value=1.0),
    priors=PriorBlock(thresholds=ThresholdPriors()),
)
GAPS: Final = torch.tensor([1.0, 1.5, 1.0, 1.0, 1.0, 1.5], dtype=torch.float64)
GENERATING: Final = torch.cat(
    (
        torch.tensor([1.5, -4.0], dtype=torch.float64),
        torch.log(torch.expm1(GAPS)),
    )
)


def design(rows: int = ROWS, *, seed: int = 3) -> JudgmentDesign:
    generator = torch.Generator().manual_seed(seed)
    log_masses = -0.05 - 4.0 * torch.rand(
        rows, generator=generator, dtype=torch.float64
    )
    codec = DesignCodec(LEVELS, factors=("verb",), readouts=(LOG_MASS, LOG_COMPLEMENT))

    return codec.encode(
        JudgmentRow(
            f"annotator-{index % PARTICIPANTS}",
            f"sentence-{index % ITEMS}",
            index % LEVELS,
            RowColumns(
                factors={"verb": f"verb-{index % ITEMS % VERBS}"},
                readouts={
                    LOG_MASS: float(log_masses[index]),
                    LOG_COMPLEMENT: math.log1p(-math.exp(float(log_masses[index]))),
                },
            ),
        )
        for index in range(rows)
    )


def bound(source: JudgmentDesign) -> BoundProgram:
    return ResponseCompiler({LOGIT: logit_steps}).emit(SPEC, source)


def generator(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


def test_the_generating_vector_lays_out_as_the_program_does() -> None:
    program = bound(design())

    assert [slot.site for slot in program.layout] == [
        "p_gram_slope",
        "cutpoint_base",
        "cutpoint_increments",
    ]
    assert program.size == GENERATING.numel()


def test_sampling_is_reproducible_at_a_seed_and_varies_with_it() -> None:
    program = bound(design())

    first = sample_responses(program, GENERATING, generator=generator(7))
    again = sample_responses(program, GENERATING, generator=generator(7))
    other = sample_responses(program, GENERATING, generator=generator(8))

    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert first.dtype is torch.int64
    assert first.shape == (ROWS,)


def test_sampled_levels_follow_the_distribution_the_likelihood_scores() -> None:
    source = design(rows=20_000, seed=5)
    program = bound(source)
    scored = torch.stack(
        [
            log_likelihood(
                bound(simulated_design(source, torch.full((source.n_rows,), level))),
                GENERATING,
            )
            for level in range(LEVELS)
        ],
        dim=1,
    )
    expected = torch.exp(scored)

    sampled = sample_responses(program, GENERATING, generator=generator(1))
    observed = functional.one_hot(sampled, LEVELS).to(torch.float64)

    assert torch.allclose(
        expected.sum(1), torch.ones(source.n_rows, dtype=torch.float64)
    )
    assert int(sampled.min()) >= 0
    assert int(sampled.max()) < LEVELS
    assert len(set(sampled.tolist())) == LEVELS
    assert torch.allclose(observed.mean(0), expected.mean(0), atol=0.01)


def test_the_sampler_and_the_likelihood_read_one_interpreted_program() -> None:
    source = design()
    compiled = ResponseCompiler({LOGIT: logit_steps}).compile(SPEC)
    program = compiled.over(source)
    location, thresholds = backends.latent(program, GENERATING)

    sampled = sample_responses(program, GENERATING, generator=generator(2))
    scored = log_likelihood(program, GENERATING)

    assert simulate.latent is backends.latent
    assert program.program is compiled
    assert hash(program.program) == hash(compiled)
    assert sampled.shape == scored.shape == (source.n_rows,)
    assert torch.equal(
        scored,
        log_category_probabilities(location, thresholds)
        .gather(1, source.responses.unsqueeze(1))
        .squeeze(1),
    )


def test_a_simulated_design_keeps_every_column_but_the_responses() -> None:
    source = design()
    program = bound(source)
    sampled = sample_responses(program, GENERATING, generator=generator(4))

    resampled = simulated_design(source, sampled)

    assert torch.equal(resampled.responses, sampled)
    assert not torch.equal(resampled.responses, source.responses)
    assert torch.equal(resampled.participants, source.participants)
    assert resampled.participant_levels == source.participant_levels
    assert torch.equal(resampled.items, source.items)
    assert resampled.item_levels == source.item_levels
    assert tuple(resampled.factors) == tuple(source.factors) == ("verb",)
    assert all(
        torch.equal(resampled.factors[name], source.factors[name])
        for name in source.factors
    )
    assert dict(resampled.factor_levels) == dict(source.factor_levels)
    assert len(source.factor_levels["verb"]) == VERBS
    assert resampled.response_levels == source.response_levels
    assert tuple(resampled.readouts) == tuple(source.readouts)
    assert all(
        torch.equal(resampled.readouts[name], source.readouts[name])
        for name in source.readouts
    )
    assert resampled.alternatives == source.alternatives


@pytest.mark.parametrize(
    "responses",
    [
        torch.zeros(ROWS, dtype=torch.float64),
        torch.zeros(ROWS + 1, dtype=torch.int64),
        torch.zeros((ROWS, 1), dtype=torch.int64),
    ],
)
def test_a_simulated_design_takes_one_integer_response_per_row(
    responses: Tensor,
) -> None:
    with pytest.raises(ValueError, match="one integer response per row"):
        simulated_design(design(), responses)


def estimands(params: Tensor) -> Tensor:
    """Return what the data identify: the slope and the six cutpoints.

    The base and the first increment are one cutpoint written as two
    numbers, so they are separately pinned by their priors alone and
    their own intervals say nothing about the sampler.
    """
    slope, base, increments = params[:1], params[1], params[2:]

    return torch.cat((slope, base + torch.cumsum(functional.softplus(increments), 0)))


class Laplace(NamedTuple):
    mode: Tensor
    covariance: Tensor

    @property
    def estimates(self) -> Tensor:
        return estimands(self.mode)

    @property
    def standard_error(self) -> Tensor:
        jacobian = torch.autograd.functional.jacobian(estimands, self.mode)

        return torch.sqrt(torch.diagonal(jacobian @ self.covariance @ jacobian.T))

    def covers(self, truth: Tensor) -> Tensor:
        half_width = NORMAL_QUANTILE * self.standard_error

        return (estimands(truth) - self.estimates).abs() <= half_width


def log_prior(program: BoundProgram, params: Tensor) -> Tensor:
    total = params.new_zeros(())

    for draw, values in zip(
        program.program.draws, program.sites(params).values(), strict=True
    ):
        match draw.prior.family, draw.prior.arguments:
            case "Normal", (location, scale):
                total = total - 0.5 * (((values - location) / scale) ** 2).sum()
            case "HalfNormal", (scale,):
                total = total - 0.5 * ((values / scale) ** 2).sum()
            case _:
                msg = f"the fit knows no log density for {draw.prior!r}"

                raise ValueError(msg)

    return total


def fitted(program: BoundProgram) -> Laplace:
    def negative_log_posterior(params: Tensor) -> Tensor:
        return -log_likelihood(program, params).sum() - log_prior(program, params)

    start = torch.cat(
        (
            torch.ones(1, dtype=torch.float64),
            torch.zeros(program.size - 1, dtype=torch.float64),
        )
    ).requires_grad_()
    optimizer = torch.optim.LBFGS(
        [start], lr=1.0, max_iter=200, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = negative_log_posterior(start)
        loss.backward()

        return loss

    for _ in range(3):
        optimizer.step(closure)

    mode = start.detach()
    curvature = torch.autograd.functional.hessian(negative_log_posterior, mode)

    return Laplace(mode, torch.linalg.inv(curvature))


def refit(seed: int, *, generating: Tensor = GENERATING) -> Laplace:
    source = design(seed=seed)
    compiled = ResponseCompiler({LOGIT: logit_steps}).compile(SPEC)
    program = compiled.over(source)
    sampled = sample_responses(program, generating, generator=generator(seed))
    fitting = compiled.over(simulated_design(source, sampled))

    if program.program is not fitting.program or hash(program.program) != hash(
        fitting.program
    ):
        msg = "sampling and fitting must share one compiled response program"

        raise RuntimeError(msg)

    return fitted(fitting)


def test_refitting_a_sample_recovers_the_generating_parameters() -> None:
    fit = refit(1)
    departure = (fit.estimates - estimands(GENERATING)).abs() / fit.standard_error

    assert fit.estimates.shape == (LEVELS,)
    assert bool(torch.isfinite(fit.standard_error).all())
    assert bool((fit.standard_error > 0.0).all())
    assert bool((departure < 3.0).all())
    assert bool(fit.covers(GENERATING).all())


def binomial_interval(
    trials: int, probability: float, *, level: float = BINOMIAL_LEVEL
) -> tuple[int, int]:
    probabilities = [
        math.comb(trials, successes)
        * probability**successes
        * (1.0 - probability) ** (trials - successes)
        for successes in range(trials + 1)
    ]
    tail = (1.0 - level) / 2.0
    cumulative = 0.0
    lower = 0
    upper = trials

    for successes, mass in enumerate(probabilities):
        cumulative += mass

        if cumulative < tail:
            lower = successes + 1

        if cumulative >= 1.0 - tail:
            upper = successes
            break

    return lower, upper


def coverage_over_replicates(generating: Tensor) -> Tensor:
    return torch.stack(
        [
            refit(seed, generating=generating).covers(GENERATING)
            for seed in range(REPLICATES)
        ]
    ).sum(0)


def test_the_ninety_percent_intervals_cover_at_their_nominal_rate() -> None:
    lower, upper = binomial_interval(REPLICATES, LEVEL)

    covered = coverage_over_replicates(GENERATING)

    assert covered.shape == (LEVELS,)
    assert bool(((covered >= lower) & (covered <= upper)).all()), covered.tolist()


def test_a_sampler_drifting_from_the_fitted_model_fails_the_coverage_check() -> None:
    lower, _ = binomial_interval(REPLICATES, LEVEL)
    drifted = GENERATING.clone()
    drifted[0] = 1.5 * GENERATING[0]

    covered = coverage_over_replicates(drifted)

    assert int(covered[0]) < lower, covered.tolist()


SCALE_USE: Final = "scale_use"
SCALE_ROWS: Final = 4000
DISCRIMINATION_COEFFICIENT: Final = 0.7
SCALE_SPEC: Final = ResponseSpec(
    family=Cumulative(),
    levels=LEVELS,
    objective=ConditionalLikelihood(),
    predictor=(
        LinkTerm(
            readout=LOG_MASS,
            linking=LinkingSpec(family=LOGIT, columns={COMPLEMENT: LOG_COMPLEMENT}),
        ),
    ),
    identification=FixedCutpointVariance(value=1.0),
    priors=PriorBlock(thresholds=ThresholdPriors()),
    parameters=(
        ParameterFormula(
            parameter=AuxiliaryParameter.DISCRIMINATION,
            predictor=(FactorTerm(name=SCALE_USE),),
        ),
    ),
)
FLAT_SPEC: Final = ResponseSpec(
    family=Cumulative(),
    levels=LEVELS,
    objective=ConditionalLikelihood(),
    predictor=SCALE_SPEC.predictor,
    identification=SCALE_SPEC.identification,
    priors=SCALE_SPEC.priors,
)
GENERATING_SCALE: Final = torch.cat(
    (
        torch.tensor(
            [
                1.5,
                -DISCRIMINATION_COEFFICIENT / 2.0,
                DISCRIMINATION_COEFFICIENT / 2.0,
                0.0,
            ],
            dtype=torch.float64,
        ),
        torch.log(torch.expm1(GAPS)),
    )
)


def scale_design(rows: int = SCALE_ROWS, *, seed: int = 3) -> JudgmentDesign:
    generator = torch.Generator().manual_seed(seed)
    log_masses = -0.05 - 4.0 * torch.rand(
        rows, generator=generator, dtype=torch.float64
    )
    codec = DesignCodec(
        LEVELS, factors=(SCALE_USE,), readouts=(LOG_MASS, LOG_COMPLEMENT)
    )

    return codec.encode(
        JudgmentRow(
            f"annotator-{index % PARTICIPANTS}",
            f"sentence-{index % ITEMS}",
            index % LEVELS,
            RowColumns(
                factors={SCALE_USE: "narrow" if index % 2 else "wide"},
                readouts={
                    LOG_MASS: float(log_masses[index]),
                    LOG_COMPLEMENT: math.log1p(-math.exp(float(log_masses[index]))),
                },
            ),
        )
        for index in range(rows)
    )


def discrimination_coefficient(params: Tensor) -> Tensor:
    """Return the log discrimination of wide scale use above narrow scale use."""
    return params[2] - params[1]


def gaussian_interval(
    fit: Laplace, readout: Callable[[Tensor], Tensor]
) -> tuple[float, float]:
    jacobian = torch.autograd.functional.jacobian(readout, fit.mode)
    error = torch.sqrt(jacobian @ fit.covariance @ jacobian)
    center = readout(fit.mode)
    half = NORMAL_QUANTILE * error

    return float(center - half), float(center + half)


def scale_programs() -> tuple[BoundProgram, BoundProgram]:
    source = scale_design()
    compiler = ResponseCompiler({LOGIT: logit_steps})

    return compiler.emit(SCALE_SPEC, source), compiler.emit(FLAT_SPEC, source)


def test_the_generating_vector_lays_out_as_the_two_formula_program_does() -> None:
    varying, _ = scale_programs()

    assert [slot.site for slot in varying.layout] == [
        "p_gram_slope",
        "discrimination_scale_use_effects",
        "cutpoint_base",
        "cutpoint_increments",
    ]
    assert varying.size == GENERATING_SCALE.numel()
    assert varying.rows == SCALE_ROWS


def test_a_single_formula_leaves_every_row_at_one_discrimination() -> None:
    varying, constant = scale_programs()
    levels = varying.host_data[SCALE_USE]
    narrow = int(torch.nonzero(levels == 0)[0])
    wide = int(torch.nonzero(levels == 1)[0])

    spread = backends.latent(varying, GENERATING_SCALE).thresholds
    flat = backends.latent(
        constant, torch.zeros(constant.size, dtype=torch.float64)
    ).thresholds

    assert spread.shape == (SCALE_ROWS, LEVELS - 1)
    assert not torch.allclose(spread[narrow], spread[wide])
    assert flat.shape == (LEVELS - 1,)
    assert constant.size == GENERATING_SCALE.numel() - 2


def test_refitting_recovers_the_generating_discrimination_coefficient() -> None:
    source = scale_design()
    compiled = ResponseCompiler({LOGIT: logit_steps}).compile(SCALE_SPEC)
    sampled = sample_responses(
        compiled.over(source), GENERATING_SCALE, generator=generator(11)
    )
    fitting = compiled.over(simulated_design(source, sampled))

    fit = fitted(fitting)
    lower, upper = gaussian_interval(fit, discrimination_coefficient)

    assert fitting.rows == SCALE_ROWS
    assert lower < DISCRIMINATION_COEFFICIENT < upper
    assert lower > 0.0


def test_the_generating_random_effect_correlation_lies_in_its_interval() -> None:
    groups = 600
    generating = 0.65
    offsets = torch.randn(
        (groups, 2),
        generator=torch.Generator().manual_seed(0),
        dtype=torch.float64,
    )
    factor = torch.tensor(
        [
            [1.0, 0.0],
            [generating, math.sqrt(1.0 - generating**2)],
        ],
        dtype=torch.float64,
    )
    effects = evaluate(
        TriangularTransform(Name("factor"), Name("offsets")),
        {"factor": factor, "offsets": offsets},
        like=offsets,
    )
    grid = torch.linspace(-0.95, 0.95, 761, dtype=torch.float64)
    first, second = effects.T
    complement = 1.0 - grid.square()
    log_likelihoods = (
        -0.5 * groups * torch.log(complement)
        - 0.5
        * (
            first.square().sum()
            - 2.0 * grid * (first * second).sum()
            + second.square().sum()
        )
        / complement
    )
    posterior = torch.softmax(log_likelihoods, dim=0)
    cumulative = posterior.cumsum(0)
    lower = grid[torch.searchsorted(cumulative, torch.tensor(0.05))]
    upper = grid[torch.searchsorted(cumulative, torch.tensor(0.95))]

    assert float(lower) < generating < float(upper)
    assert float(lower) > 0.0
