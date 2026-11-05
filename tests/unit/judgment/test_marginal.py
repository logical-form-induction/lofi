import math
from typing import Final

import pytest
import torch

from lofi.judgment import (
    PARTICIPANT,
    PLATE,
    RESPONSE,
    ApproximateMarginalLikelihood,
    Binary,
    Bind,
    BoundProgram,
    ConditionalPlan,
    Cumulative,
    DesignCodec,
    Draw,
    FixedDiscrimination,
    Gather,
    JudgmentRow,
    MarginalLikelihoodError,
    Name,
    ObserveChoice,
    ObserveOrdered,
    OffsetTerm,
    Operator,
    Prior,
    PriorBlock,
    RandomIntercept,
    ResponseCompiler,
    ResponseProgram,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    TruncatedMarginalLikelihood,
    TruncatedMarginalPlan,
    log_likelihood,
)

DTYPE: Final = torch.float64
GROUP_OBJECT: Final = "ByParticipant"
SCORE: Final = "score"
STANDARD_NORMAL: Final = Prior("Normal", 0.0, 1.0)


def plan(*, points: int = 5, seed: int = 17) -> TruncatedMarginalPlan:
    return TruncatedMarginalPlan(
        "truncated_marginal",
        PARTICIPANT,
        3.5,
        1e-8,
        1e-10,
        points,
        seed,
    )


def bound_program(
    rank: int,
    *,
    group_index: torch.Tensor | None = None,
    objective: ConditionalPlan | TruncatedMarginalPlan | None = None,
    groups: tuple[str, ...] = (PARTICIPANT,),
    prior: Prior = STANDARD_NORMAL,
) -> BoundProgram:
    indices = (
        torch.tensor([0, 0, 1, 1], dtype=torch.int64)
        if group_index is None
        else group_index
    )
    responses = torch.tensor([0, 1, 1, 0], dtype=torch.int64)[: indices.numel()]
    scores = torch.tensor([-0.3, 0.2, 0.4, -0.1], dtype=DTYPE)[: indices.numel()]
    local_names = tuple(f"local_{axis}" for axis in range(rank))
    location = Name(SCORE)
    draws = [
        Draw("cutpoint", ("Unit",), Prior("Normal", 0.0, 2.0)),
        Draw("local_scale", ("Unit",), Prior("HalfNormal", 1.0)),
    ]

    for name in local_names:
        draws.append(Draw(name, (GROUP_OBJECT,), prior))
        shifted = Binary(
            Operator.MUL,
            Name("local_scale"),
            Gather(name, PARTICIPANT),
        )
        location = Binary(Operator.ADD, location, shifted)

    program = ResponseProgram(
        levels=2,
        steps=(
            *draws,
            Bind("location", location),
            ObserveOrdered(RESPONSE, PLATE, location="location", thresholds="cutpoint"),
        ),
        objects={GROUP_OBJECT: (PARTICIPANT,)},
        indices={PARTICIPANT: (PARTICIPANT,)},
        readouts=(SCORE,),
        objective=plan(points=5) if objective is None else objective,
        groups=groups,
        cutpoint_object=None,
        threshold_group=None,
        scalar_object="Unit",
    )

    return BoundProgram(
        program,
        sizes={PLATE: indices.numel(), "Unit": 1, GROUP_OBJECT: 2},
        host_data={RESPONSE: responses, PARTICIPANT: indices, SCORE: scores},
    )


def binary_log_likelihood(
    responses: torch.Tensor,
    locations: torch.Tensor,
    cutpoint: torch.Tensor,
) -> torch.Tensor:
    lower = -torch.nn.functional.softplus(locations - cutpoint)
    upper = -torch.nn.functional.softplus(cutpoint - locations)

    return torch.where(responses == 0, lower, upper)


def test_a_scalar_block_follows_the_adaptive_ode_integral() -> None:
    bound = bound_program(1)
    marginal = ApproximateMarginalLikelihood(bound, active_groups=(0, 1))
    population = torch.tensor([0.15, 0.65], dtype=DTYPE)
    grid = torch.linspace(-3.5, 3.5, 100_001, dtype=DTYPE)
    expected = []

    for group in range(2):
        selected = bound.host_data[PARTICIPANT] == group
        locations = bound.host_data[SCORE][selected, None] + 0.65 * grid[None, :]
        responses = bound.host_data[RESPONSE][selected, None]
        row_log = binary_log_likelihood(responses, locations, population[0]).sum(0)
        log_integrand = row_log - 0.5 * (grid.square() + math.log(2.0 * math.pi))
        expected.append(torch.log(torch.trapezoid(log_integrand.exp(), grid)))

    assert marginal.group == PARTICIPANT
    assert marginal.local_width == 1
    assert marginal.population_layout == bound.layout[:2]
    assert torch.allclose(marginal(population), torch.stack(expected).mean(), atol=2e-7)


def test_multivariate_blocks_consume_the_supplied_positive_in_ball_rule() -> None:
    bound = bound_program(2)
    nodes = torch.tensor(
        [[0.0, 0.0], [0.5, -0.25], [-0.75, 0.4], [1.0, 0.5], [-0.2, -1.1]],
        dtype=DTYPE,
    )
    weights = torch.tensor([0.4, 0.6, 0.8, 0.5, 0.7], dtype=DTYPE)
    marginal = ApproximateMarginalLikelihood(
        bound, active_groups=(0, 1), nodes=nodes, weights=weights
    )
    population = torch.tensor([0.15, 0.65], dtype=DTYPE)
    expected = []

    for group in range(2):
        selected = bound.host_data[PARTICIPANT] == group
        locations = (
            bound.host_data[SCORE][selected, None] + 0.65 * nodes.sum(1)[None, :]
        )
        responses = bound.host_data[RESPONSE][selected, None]
        row_log = binary_log_likelihood(responses, locations, population[0]).sum(0)
        local_prior = -0.5 * (nodes.square().sum(1) + 2.0 * math.log(2.0 * math.pi))
        expected.append(torch.logsumexp(weights.log() + row_log + local_prior, dim=0))

    nodes.fill_(99.0)
    weights.fill_(99.0)

    assert torch.allclose(marginal(population), torch.stack(expected).mean())


def test_the_reduction_is_a_uniform_mean_over_whole_groups() -> None:
    index = torch.tensor([0, 1, 1, 1], dtype=torch.int64)
    bound = bound_program(1, group_index=index)
    marginal = ApproximateMarginalLikelihood(bound, active_groups=(0, 1))
    population = torch.tensor([0.0, 0.5], dtype=DTYPE)
    value = marginal(population)
    first = BoundProgram(
        bound.program,
        sizes={PLATE: 1, "Unit": 1, GROUP_OBJECT: 1},
        host_data={
            RESPONSE: bound.host_data[RESPONSE][:1],
            PARTICIPANT: torch.zeros(1, dtype=torch.int64),
            SCORE: bound.host_data[SCORE][:1],
        },
    )
    second = BoundProgram(
        bound.program,
        sizes={PLATE: 3, "Unit": 1, GROUP_OBJECT: 1},
        host_data={
            RESPONSE: bound.host_data[RESPONSE][1:],
            PARTICIPANT: torch.zeros(3, dtype=torch.int64),
            SCORE: bound.host_data[SCORE][1:],
        },
    )
    expected = 0.5 * (
        ApproximateMarginalLikelihood(first, active_groups=(0,))(population)
        + ApproximateMarginalLikelihood(second, active_groups=(0,))(population)
    )

    assert torch.allclose(value, expected)


def test_group_scores_are_exposed_without_reducing_other_groups() -> None:
    marginal = ApproximateMarginalLikelihood(bound_program(1), active_groups=(0, 1))
    population = torch.tensor([0.15, 0.65], dtype=DTYPE)
    first = marginal.group_log_likelihood(population, 0)
    second = marginal.group_log_likelihood(population, 1)

    assert torch.allclose(marginal.log_likelihood(population), (first + second) / 2)

    with pytest.raises(MarginalLikelihoodError, match="not among"):
        marginal.group_log_likelihood(population, 2)


def test_the_population_posterior_counts_each_group_and_the_prior_once() -> None:
    marginal = ApproximateMarginalLikelihood(bound_program(1), active_groups=(0, 1))
    population = torch.tensor([0.15, 0.65], dtype=DTYPE)
    prior = torch.distributions.Normal(0.0, 2.0).log_prob(population[0])
    prior = prior + torch.distributions.HalfNormal(1.0).log_prob(population[1])
    groups = sum(
        (marginal.group_log_likelihood(population, group) for group in (0, 1)),
        start=population.new_zeros(()),
    )

    assert torch.allclose(marginal.log_posterior(population), prior + groups)
    assert not torch.allclose(
        marginal.log_posterior(population),
        prior + marginal.log_likelihood(population),
    )


def test_row_and_active_group_order_do_not_change_the_marginal_objective() -> None:
    bound = bound_program(1)
    order = torch.tensor([3, 1, 0, 2], dtype=torch.int64)
    reordered = BoundProgram(
        bound.program,
        sizes=bound.sizes,
        host_data={name: values[order] for name, values in bound.host_data.items()},
    )
    population = torch.tensor([0.2, 0.7], dtype=DTYPE)
    forward = ApproximateMarginalLikelihood(bound, active_groups=(0, 1))(population)
    backward = ApproximateMarginalLikelihood(reordered, active_groups=(1, 0))

    assert backward.active_groups == (0, 1)
    assert torch.allclose(backward(population), forward)


def test_an_explicit_active_group_reduces_only_its_exact_rows() -> None:
    bound = bound_program(1, group_index=torch.tensor([1, 1], dtype=torch.int64))
    population = torch.tensor([0.0, 0.5], dtype=DTYPE)
    selected = ApproximateMarginalLikelihood(bound, active_groups=(1,))

    assert selected.active_groups == (1,)
    assert bool(torch.isfinite(selected(population)))


@pytest.mark.parametrize("groups", [(), (0, 0), (-1,), (2,)])
def test_invalid_explicit_active_groups_are_refused(
    groups: tuple[int, ...],
) -> None:
    with pytest.raises(MarginalLikelihoodError):
        ApproximateMarginalLikelihood(bound_program(1), active_groups=groups)


def test_the_group_integral_is_not_a_reweighting_of_row_integrals() -> None:
    bound = bound_program(1)
    population = torch.tensor([0.1, 1.2], dtype=DTYPE)
    grouped = ApproximateMarginalLikelihood(bound, active_groups=(0, 1))(population)
    separate = []

    for row in range(bound.rows):
        singleton = BoundProgram(
            bound.program,
            sizes={PLATE: 1, "Unit": 1, GROUP_OBJECT: 1},
            host_data={
                RESPONSE: bound.host_data[RESPONSE][row : row + 1],
                PARTICIPANT: torch.zeros(1, dtype=torch.int64),
                SCORE: bound.host_data[SCORE][row : row + 1],
            },
        )
        separate.append(
            ApproximateMarginalLikelihood(singleton, active_groups=(0,))(population)
        )

    row_reweighting = torch.stack(separate).mean()

    assert abs(float(grouped - row_reweighting)) > 1e-3


def test_zero_local_scale_reduces_to_conditional_score_plus_prior_mass() -> None:
    bound = bound_program(1)
    population = torch.tensor([0.2, 0.0], dtype=DTYPE)
    marginal = ApproximateMarginalLikelihood(bound, active_groups=(0, 1))(population)
    conditional = (
        binary_log_likelihood(
            bound.host_data[RESPONSE], bound.host_data[SCORE], population[0]
        )
        .reshape(2, 2)
        .sum(1)
        .mean()
    )
    prior_mass = math.erf(3.5 / math.sqrt(2.0))

    assert torch.allclose(marginal, conditional + math.log(prior_mass), atol=2e-7)


def test_the_generated_rule_preserves_caller_rng_state() -> None:
    state = torch.random.get_rng_state().clone()

    ApproximateMarginalLikelihood(bound_program(2), active_groups=(0, 1))

    assert torch.equal(torch.random.get_rng_state(), state)


def test_an_all_local_program_uses_the_empty_population_tensor_policy() -> None:
    local_cutpoint = "local_cutpoint"
    local_location = "local_location"
    program = ResponseProgram(
        levels=2,
        steps=(
            Draw(local_cutpoint, (GROUP_OBJECT, "Unit"), STANDARD_NORMAL),
            Draw(local_location, (GROUP_OBJECT,), STANDARD_NORMAL),
            Bind("row_cutpoint", Gather(local_cutpoint, PARTICIPANT)),
            Bind("row_location", Gather(local_location, PARTICIPANT)),
            ObserveOrdered(
                RESPONSE,
                PLATE,
                location="row_location",
                thresholds="row_cutpoint",
            ),
        ),
        objects={GROUP_OBJECT: (PARTICIPANT,)},
        indices={PARTICIPANT: (PARTICIPANT,)},
        readouts=(),
        objective=plan(),
        groups=(PARTICIPANT,),
        cutpoint_object=None,
        threshold_group=None,
        scalar_object="Unit",
    )
    bound = BoundProgram(
        program,
        sizes={PLATE: 2, "Unit": 1, GROUP_OBJECT: 2},
        host_data={
            RESPONSE: torch.tensor([0, 1], dtype=torch.int64),
            PARTICIPANT: torch.arange(2),
        },
    )
    marginal = ApproximateMarginalLikelihood(bound, active_groups=(0, 1))
    population = torch.empty(0, dtype=DTYPE, requires_grad=True)

    value = marginal.log_likelihood(population)

    assert marginal.population_layout == ()
    assert marginal.population_size == 0
    assert marginal.local_width == 2
    assert value.dtype == population.dtype
    assert bool(torch.isfinite(value))


def test_the_public_row_scorer_refuses_a_marginal_program() -> None:
    bound = bound_program(1)

    with pytest.raises(MarginalLikelihoodError, match="population layout"):
        log_likelihood(bound, torch.zeros(bound.size, dtype=DTYPE))


def real_program() -> BoundProgram:
    declaration = ResponseSpec(
        family=Cumulative(),
        levels=3,
        objective=TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=3.0,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8,
            quadrature_points=16,
            quadrature_seed=23,
        ),
        predictor=(RandomIntercept(group=PARTICIPANT), OffsetTerm(readout=SCORE)),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    rows = (
        JudgmentRow(
            f"p-{index % 2}",
            f"i-{index}",
            index % 3,
            RowColumns({}, {SCORE: (-0.3, 0.1, 0.5, -0.2)[index]}),
        )
        for index in range(4)
    )
    design = DesignCodec(3, readouts=(SCORE,)).encode(rows)

    return ResponseCompiler({}).emit(declaration, design)


def test_the_compiler_excludes_the_local_block_and_keeps_gradients() -> None:
    bound = real_program()
    readout = bound.host_data[SCORE].clone().requires_grad_()
    marginal = ApproximateMarginalLikelihood(
        bound.with_readouts({SCORE: readout}), active_groups=(0, 1)
    )
    local_sites = {"participant_intercept_offsets", "participant_cutpoint_offsets"}
    population = torch.full(
        (marginal.population_size,), 0.3, dtype=DTYPE, requires_grad=True
    )

    value = marginal(population)
    value.backward()

    assert local_sites.isdisjoint(slot.site for slot in marginal.population_layout)
    assert local_sites <= {slot.site for slot in bound.layout}
    assert readout.grad is not None
    assert bool(torch.isfinite(readout.grad).all())
    assert bool((readout.grad != 0.0).any())
    assert population.grad is not None
    assert bool(torch.isfinite(population.grad).all())
    position = 0

    for slot in marginal.population_layout:
        if slot.site == "participant_intercept_scale":
            assert bool((population.grad[position : position + slot.size] != 0.0).any())
            break

        position += slot.size
    else:
        pytest.fail("the participant intercept scale is absent from the population")


@pytest.mark.parametrize(
    ("program", "message"),
    [
        (bound_program(1, objective=ConditionalPlan()), "truncated_marginal"),
        (
            bound_program(1, objective=plan()._replace(group="item")),
            "complete 'participant' block",
        ),
        (bound_program(1, groups=(PARTICIPANT, "item")), "crossed groups"),
        (bound_program(0), "no local latent effect"),
        (bound_program(1, prior=Prior("Normal", 1.0, 1.0)), "standard-normal"),
    ],
)
def test_structurally_invalid_programs_are_refused_by_the_named_error(
    program: BoundProgram, message: str
) -> None:
    with pytest.raises(MarginalLikelihoodError, match=message):
        ApproximateMarginalLikelihood(program, active_groups=(0, 1))


def test_an_incomplete_group_is_refused_by_name() -> None:
    program = bound_program(1, group_index=torch.zeros(2, dtype=torch.int64))

    with pytest.raises(MarginalLikelihoodError, match="not exactly"):
        ApproximateMarginalLikelihood(program, active_groups=(0, 1))


@pytest.mark.parametrize(
    ("nodes", "weights", "message"),
    [
        (torch.zeros(4, 2), torch.ones(4), "shape"),
        (torch.zeros(5, 2), torch.tensor([1.0, 1.0, 0.0, 1.0, 1.0]), "positive"),
        (
            torch.tensor([[4.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]]),
            torch.ones(5),
            "inside",
        ),
    ],
)
def test_invalid_multivariate_rules_are_refused(
    nodes: torch.Tensor, weights: torch.Tensor, message: str
) -> None:
    with pytest.raises(MarginalLikelihoodError, match=message):
        ApproximateMarginalLikelihood(
            bound_program(2), active_groups=(0, 1), nodes=nodes, weights=weights
        )


def test_a_population_layout_mismatch_is_refused() -> None:
    marginal = ApproximateMarginalLikelihood(bound_program(1), active_groups=(0, 1))

    with pytest.raises(MarginalLikelihoodError, match="population layout"):
        marginal(torch.zeros(marginal.population_size + 1, dtype=DTYPE))


def test_an_unsupported_response_family_is_refused() -> None:
    program = ResponseProgram(
        levels=2,
        steps=(
            Draw("local", (GROUP_OBJECT,), Prior("Normal", 0.0, 1.0)),
            ObserveChoice(RESPONSE, PLATE, logits=SCORE),
        ),
        objects={GROUP_OBJECT: (PARTICIPANT,)},
        indices={PARTICIPANT: (PARTICIPANT,)},
        readouts=(SCORE,),
        objective=plan(),
        groups=(PARTICIPANT,),
        cutpoint_object=None,
        threshold_group=None,
        scalar_object="Unit",
        alternative_object="Alternative",
        alternatives=2,
    )
    bound = BoundProgram(
        program,
        sizes={PLATE: 2, "Unit": 1, GROUP_OBJECT: 2, "Alternative": 2},
        host_data={
            RESPONSE: torch.zeros(2, dtype=torch.int64),
            PARTICIPANT: torch.arange(2),
            SCORE: torch.zeros(2, 2, dtype=DTYPE),
        },
    )

    with pytest.raises(MarginalLikelihoodError, match="cumulative ordered"):
        ApproximateMarginalLikelihood(bound, active_groups=(0, 1))
