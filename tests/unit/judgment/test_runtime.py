import json
import math
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
import quivers
import quivers.continuous as quivers_continuous
import quivers.effects as quivers_effects
import torch
from quivers import MonadicProgram
from quivers.effects import TraceHandler, clamp, run_program
from torch import nn

from lofi.judgment import (
    RESPONSE,
    Bounded,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    BoundedResponseSpec,
    Choice,
    ClosedScale,
    ConditionalLikelihood,
    CorrelatedRandomSlope,
    Cumulative,
    DesignCodec,
    Draw,
    FactorTerm,
    FixedDiscrimination,
    IdentityLinkOnLogit,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkingSteps,
    LinkTerm,
    Name,
    PriorBlock,
    PriorSpec,
    ProgramRuntime,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ScaleNormalization,
    ThresholdPriors,
    linear_steps,
    log_joint,
    log_likelihood,
    logit_steps,
    morcela_steps,
    probability_steps,
)
from lofi.judgment.runtime import QuiversRuntime, build

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment import BoundProgram

LADDER: Final = Path(__file__).resolve().parents[3] / "tests" / "data" / "judgment"
LEVELS: Final = 7
LM: Final = "lm_log_probability"
UNIGRAM: Final = "unigram_log_probability"
TOKENS: Final = "length"
P_GRAM: Final = "p_gram"
P_GRAM_COMPLEMENT: Final = "p_gram_complement"
POOLED: Final = "encoder_pooled"
ROOT_ENTROPY: Final = "root_entropy"
READOUTS: Final = (
    LM,
    UNIGRAM,
    TOKENS,
    P_GRAM,
    P_GRAM_COMPLEMENT,
    POOLED,
    ROOT_ENTROPY,
)
FACTORS: Final = ("verb", "frame")
SEEDS: Final = 3


class Shape(NamedTuple):
    rows: int
    participants: int
    items: int
    verbs: int
    frames: int


SHAPES: Final = (Shape(24, 4, 6, 3, 2), Shape(35, 5, 7, 4, 3))


def identity_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool = False,
    signed_slope: bool = False,
) -> LinkingSteps:
    del prefix, signed_slope, unit_slope

    return LinkingSteps({}, Name(term.readout))


def compiler() -> ResponseCompiler:
    return ResponseCompiler(
        {
            "morcela": morcela_steps,
            "logit": logit_steps,
            "probability": probability_steps,
            "linear": linear_steps,
            "identity": identity_steps,
        }
    )


def test_every_quivers_builder_dependency_is_publicly_exported() -> None:
    continuous = {
        "ContinuousMorphism",
        "Draw",
        "Euclidean",
        "FixedDistribution",
        "LKJCorrelationFactor",
        "Let",
        "MixedInlineDistribution",
        "MonadicProgram",
        "Observe",
        "OrderedLogistic",
        "PlateDraw",
        "ProductSpace",
        "Step",
        "VectorisedObserve",
        "make_fixed_exponential",
        "make_fixed_halfnormal",
        "make_fixed_lognormal",
        "make_fixed_normal",
        "make_fixed_uniform",
        "reading",
    }
    effects = {"Trace", "TraceHandler", "clamp", "run_program"}

    assert "FinSet" in quivers.__all__
    assert continuous <= set(quivers_continuous.__all__)
    assert effects <= set(quivers_effects.__all__)


def test_quivers_builds_and_scores_a_bounded_continuous_observation() -> None:
    declared = BoundedResponseSpec(
        family=Bounded(scale=0.2),
        predictor=(
            LinkTerm(
                readout=P_GRAM,
                linking=LinkingSpec(
                    family="logit",
                    columns={"complement": P_GRAM_COMPLEMENT},
                ),
            ),
        ),
        identification=IdentityLinkOnLogit(),
    )
    rows = tuple(
        BoundedJudgmentRow(
            "annotator",
            f"item-{position}",
            response,
            ClosedScale(0.0, 10.0),
            RowColumns(
                {},
                {P_GRAM: -0.2 - position, P_GRAM_COMPLEMENT: -1.2},
            ),
        )
        for position, response in enumerate((0.0, 5.0, 10.0))
    )
    design = BoundedDesignCodec(readouts=(P_GRAM, P_GRAM_COMPLEMENT)).encode(rows)
    bound = compiler().compile(declared).over(design)
    runtime = QuiversRuntime()
    params = torch.empty(0, dtype=torch.float64)

    runtime.load(bound.program)
    scored = runtime.observed_log_probabilities(bound, params)

    assert isinstance(build(bound), MonadicProgram)
    assert scored.shape == (3,)
    assert torch.allclose(scored, log_likelihood(bound, params), atol=1e-6, rtol=0.0)


def rows_of(shape: Shape) -> list[JudgmentRow]:
    return [
        JudgmentRow(
            f"annotator-{index % shape.participants}",
            f"sentence-{index % shape.items}",
            (index * 5) % LEVELS,
            RowColumns(
                factors={
                    "verb": f"verb-{index % shape.verbs}",
                    "frame": f"frame-{(index // 2) % shape.frames}",
                },
                readouts={
                    LM: -2.0 - 0.1 * (index % 7),
                    UNIGRAM: -4.0 - 0.2 * (index % 5),
                    TOKENS: 3.0 + index % 5,
                    P_GRAM: -0.25 - 0.5 * (index % 4),
                    P_GRAM_COMPLEMENT: math.log1p(-math.exp(-0.25 - 0.5 * (index % 4))),
                    POOLED: 0.1 * (index % 6),
                    ROOT_ENTROPY: 0.2 + 0.1 * (index % 5),
                },
            ),
        )
        for index in range(shape.rows)
    ]


def design_of(shape: Shape) -> JudgmentDesign:
    return DesignCodec(LEVELS, factors=FACTORS, readouts=READOUTS).encode(
        rows_of(shape)
    )


def choice_program() -> BoundProgram:
    alternatives = 3
    rows = []

    for trial in range(6):
        probabilities = tuple(
            0.1 + 0.1 * ((trial + alternative) % 5)
            for alternative in range(alternatives)
        )
        rows.append(
            JudgmentRow(
                f"annotator-{trial % 2}",
                f"choice-{trial}",
                trial % alternatives,
                RowColumns(
                    {},
                    {},
                    {
                        P_GRAM: tuple(math.log(value) for value in probabilities),
                        P_GRAM_COMPLEMENT: tuple(
                            math.log1p(-value) for value in probabilities
                        ),
                    },
                ),
                tuple(
                    f"choice-{trial}-{alternative}"
                    for alternative in range(alternatives)
                ),
            )
        )

    design = DesignCodec(
        alternatives,
        readouts=(P_GRAM, P_GRAM_COMPLEMENT),
        alternatives=alternatives,
    ).encode(rows)
    declaration = ResponseSpec(
        family=Choice(),
        levels=alternatives,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(
                readout=P_GRAM,
                linking=LinkingSpec(
                    family="logit",
                    columns={"complement": P_GRAM_COMPLEMENT},
                ),
            ),
        ),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock(),
    )

    return compiler().compile(declaration).over(design)


def rung(name: str) -> ResponseSpec:
    return ResponseSpec.model_validate_json(
        (LADDER / "ladder" / f"{name}.json").read_text(encoding="utf-8")
    )


def rung_names() -> tuple[str, ...]:
    index = json.loads((LADDER / "ladder" / "ladder.json").read_text(encoding="utf-8"))

    return (
        *index["rungs"],
        *(name for names in index["ablations"].values() for name in names),
        index["annotator"],
    )


def cumulative_rung_names() -> tuple[str, ...]:
    return tuple(
        name for name in rung_names() if rung(name).family.family == "cumulative"
    )


def witness_draw(draw: Draw, size: int, generator: torch.Generator) -> Tensor:
    standard = torch.randn(size, dtype=torch.float64, generator=generator)
    uniform = torch.rand(size, dtype=torch.float64, generator=generator)

    match draw.prior.family, draw.prior.arguments:
        case "Normal", (location, scale):
            result = location + 0.2 * scale * standard
        case "HalfNormal", (scale,):
            result = 0.1 * scale + 0.2 * scale * standard.abs()
        case "LogNormal", (location, scale):
            result = torch.exp(location + 0.2 * scale * standard)
        case "Uniform", (low, high):
            result = low + (high - low) * uniform
        case "Exponential", (rate,):
            result = (0.1 + 0.2 * standard.abs()) / rate
        case "Cauchy", (location, scale):
            result = location + 0.2 * scale * standard
        case "LKJCholesky", (concentration,):
            del concentration
            correlation = 0.35

            result = torch.tensor(
                [1.0, 0.0, correlation, math.sqrt(1.0 - correlation**2)],
                dtype=torch.float64,
            )
        case _:
            msg = f"no witness draw stands in for the prior {draw.prior!r}"

            raise ValueError(msg)

    return result


def parameters_of(program: BoundProgram, seed: int) -> Tensor:
    generator = torch.Generator().manual_seed(seed)
    draws = {draw.site: draw for draw in program.program.draws}

    return torch.cat(
        [
            witness_draw(draws[slot.site], slot.size, generator)
            for slot in program.layout
        ]
    )


def recorded_score(program: BoundProgram, params: Tensor) -> Tensor:
    handler = TraceHandler()

    with clamp(program.sites(params)), handler:
        run_program(build(program), params.new_zeros((1, 1)), dict(program.host_data))

    return handler.trace.sites[RESPONSE].log_prob


def bound_cases() -> tuple[object, ...]:
    shared = compiler()
    cases = []

    for name in cumulative_rung_names():
        program = shared.compile(rung(name))

        for index, shape in enumerate(SHAPES):
            bound = program.over(design_of(shape))
            cases.extend(
                pytest.param(
                    bound, parameters_of(bound, seed), id=f"{name}-{index}-{seed}"
                )
                for seed in range(SEEDS)
            )

    return tuple(cases)


SCORED_PRIORS: Final = (
    PriorSpec(distribution="Normal", parameters=(0.0, 2.0)),
    PriorSpec(distribution="HalfNormal", parameters=(2.0,)),
    PriorSpec(distribution="LogNormal", parameters=(0.0, 0.5)),
    PriorSpec(distribution="Uniform", parameters=(-1.0, 1.0)),
    PriorSpec(distribution="Exponential", parameters=(2.0,)),
)


def with_prior(prior: PriorSpec) -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(FactorTerm(name="verb"),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors(base=prior)),
        normalization=ScaleNormalization.IDENTITY,
    )


def test_the_runtime_answers_the_port() -> None:
    runtime = QuiversRuntime()

    assert isinstance(runtime, ProgramRuntime)
    assert repr(runtime) == "QuiversRuntime(built=0)"


def test_a_program_is_built_once_for_each_set_of_sizes_and_kept() -> None:
    runtime = QuiversRuntime()
    program = compiler().compile(rung("verb_only"))
    full = program.over(design_of(SHAPES[0]))
    again = program.over(design_of(SHAPES[0]))
    other = program.over(design_of(SHAPES[1]))

    runtime.load(program)

    assert runtime.built == 0
    assert runtime.program(full) is runtime.program(again)
    assert runtime.program(other) is not runtime.program(full)
    assert runtime.built == 2
    assert repr(runtime) == "QuiversRuntime(built=2)"


def test_a_kept_program_scores_the_responses_it_is_run_over() -> None:
    runtime = QuiversRuntime()
    design = design_of(SHAPES[0])
    program = compiler().compile(rung("verb_only"))
    bound = program.over(design)
    shifted = program.over(
        DesignCodec(LEVELS, factors=FACTORS, readouts=READOUTS).encode(
            [
                JudgmentRow(
                    row.participant,
                    row.item,
                    (row.response + 1) % LEVELS,
                    RowColumns(dict(row.factors), dict(row.readouts)),
                )
                for row in rows_of(SHAPES[0])
            ]
        )
    )
    params = parameters_of(bound, 3)

    first = runtime.observed_log_probabilities(bound, params)
    second = runtime.observed_log_probabilities(shifted, params)

    assert runtime.built == 1
    assert torch.allclose(second, log_likelihood(shifted, params), atol=1e-6, rtol=0.0)
    assert not torch.allclose(first, second)
    assert torch.allclose(
        runtime.log_joint(shifted, params),
        log_joint(shifted, params),
        atol=1e-4,
        rtol=1e-5,
    )


@pytest.mark.parametrize("prior", SCORED_PRIORS, ids=lambda prior: prior.distribution)
def test_every_prior_family_the_torch_backend_scores_is_loaded(
    prior: PriorSpec,
) -> None:
    QuiversRuntime().load(compiler().compile(with_prior(prior)))


@pytest.mark.parametrize("prior", SCORED_PRIORS, ids=lambda prior: prior.distribution)
def test_every_prior_family_enters_the_quivers_log_joint_as_the_torch_backend_s(
    prior: PriorSpec,
) -> None:
    bound = compiler().compile(with_prior(prior)).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 3)

    assert torch.allclose(
        QuiversRuntime().log_joint(bound, params),
        log_joint(bound, params),
        atol=1e-4,
        rtol=1e-5,
    )


def test_a_literal_is_written_at_the_precision_of_the_values_it_meets() -> None:
    spec = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(FactorTerm(name="verb"),),
        identification=FixedDiscrimination(value=0.1),
        priors=PriorBlock(thresholds=ThresholdPriors(base=SCORED_PRIORS[0])),
        normalization=ScaleNormalization.IDENTITY,
    )
    bound = compiler().compile(spec).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 4)

    assert torch.allclose(
        QuiversRuntime().observed_log_probabilities(bound, params),
        log_likelihood(bound, params),
        atol=1e-12,
        rtol=0.0,
    )


@pytest.mark.parametrize(
    "prior",
    [
        PriorSpec(distribution="Wibble", parameters=(0.0, 2.0)),
        PriorSpec(distribution="Cauchy", parameters=(0.0, 2.0)),
        PriorSpec(distribution="Normal", parameters=(0.0,)),
        PriorSpec(distribution="HalfNormal", parameters=(1.0, 2.0)),
    ],
    ids=("unknown", "cauchy", "short-normal", "long-half-normal"),
)
def test_a_prior_quivers_builds_no_fixed_distribution_for_is_refused(
    prior: PriorSpec,
) -> None:
    program = compiler().compile(with_prior(prior))

    with pytest.raises(ValueError, match=rf"{prior.distribution}.*no fixed"):
        QuiversRuntime().load(program)

    with pytest.raises(ValueError, match="no fixed"):
        build(program.over(design_of(SHAPES[0])))


def test_a_built_program_draws_every_site_and_returns_the_response() -> None:
    bound = compiler().compile(rung("verb_by_frame")).over(design_of(SHAPES[0]))
    built = build(bound)
    handler = TraceHandler()

    with clamp(bound.sites(parameters_of(bound, 0))), handler:
        run_program(
            built, torch.zeros(1, 1, dtype=torch.float64), dict(bound.host_data)
        )

    traced = handler.trace.sites

    assert isinstance(built, MonadicProgram)
    assert built.domain == built.codomain
    assert all(isinstance(step, quivers_continuous.Step) for step in built.steps)
    assert any(isinstance(step, quivers_continuous.Draw) for step in built.steps)
    assert any(isinstance(step, quivers_continuous.Let) for step in built.steps)
    assert isinstance(built.steps[-1], quivers_continuous.Observe)
    assert {slot.site for slot in bound.layout} <= set(traced)
    assert all(traced[slot.site].value.numel() == slot.size for slot in bound.layout)
    assert {step.site for step in bound.program.body} | {RESPONSE} == set(traced)


def test_an_unclamped_run_draws_every_site_at_the_size_the_layout_gives_it() -> None:
    bound = compiler().compile(rung("intercept_only")).over(design_of(SHAPES[0]))
    handler = TraceHandler()
    given = {
        name: column for name, column in bound.host_data.items() if name != RESPONSE
    }

    with handler:
        run_program(build(bound), torch.zeros(1, 1, dtype=torch.float64), given)

    traced = handler.trace.sites

    assert [traced[slot.site].value.numel() for slot in bound.layout] == [
        slot.size for slot in bound.layout
    ]
    assert traced["cutpoints"].value.shape == (bound.rows, LEVELS - 1)
    assert traced[RESPONSE].value.shape == (bound.rows,)
    assert not traced[RESPONSE].is_observed


def correlated_program(shape: Shape) -> BoundProgram:
    """Return a correlated participant intercept-and-slope program."""
    link = LinkTerm(
        readout=P_GRAM,
        linking=LinkingSpec(family="logit", columns={"complement": P_GRAM_COMPLEMENT}),
    )
    declared = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            link,
            CorrelatedRandomSlope(group="participant", term=link),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )

    return compiler().compile(declared).over(design_of(shape))


def test_quivers_draws_one_lkj_factor_independent_of_group_and_row_count() -> None:
    traced_shapes = []

    for shape in SHAPES:
        bound = correlated_program(shape)
        handler = TraceHandler()
        correlation = next(
            draw.site
            for draw in bound.program.draws
            if draw.prior.family == "LKJCholesky"
        )

        with handler:
            run_program(
                build(bound),
                torch.zeros(1, 1, dtype=torch.float64),
                {
                    name: value
                    for name, value in bound.host_data.items()
                    if name != RESPONSE
                },
            )

        traced_shapes.append(handler.trace.sites[correlation].value.shape)

    assert traced_shapes == [torch.Size((1, 4)), torch.Size((1, 4))]


def test_each_row_is_scored_at_the_traced_location_and_thresholds() -> None:
    runtime = QuiversRuntime()
    bound = (
        compiler().compile(rung("morcela_by_verb_by_frame")).over(design_of(SHAPES[1]))
    )
    params = parameters_of(bound, 7)

    scored = runtime.observed_log_probabilities(bound, params)

    assert scored.shape == (bound.rows,)
    assert scored.dtype is torch.float64
    assert bool((scored < 0.0).all())
    assert torch.allclose(scored, log_likelihood(bound, params), atol=1e-6, rtol=0.0)


def test_public_builder_scores_a_categorical_observation_from_logits() -> None:
    bound = choice_program()
    params = torch.empty(0, dtype=torch.float64)
    runtime = QuiversRuntime()

    per_row = runtime.observed_log_probabilities(bound, params)

    assert per_row.shape == (bound.rows,)
    assert torch.allclose(per_row, log_likelihood(bound, params), atol=1e-12, rtol=0.0)
    assert torch.allclose(
        runtime.log_joint(bound, params),
        log_joint(bound, params),
        atol=1e-12,
        rtol=0.0,
    )


def test_public_builder_gathers_cutpoints_and_differentiates_a_covariate() -> None:
    bound = compiler().compile(rung("p_gram_only")).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 3).requires_grad_()
    covariate = bound.host_data[P_GRAM].clone().requires_grad_()
    scored = bound.with_readouts({P_GRAM: covariate})

    quivers_gradients = torch.autograd.grad(
        QuiversRuntime().log_joint(scored, params),
        (params, covariate),
        retain_graph=True,
    )
    torch_gradients = torch.autograd.grad(
        log_joint(scored, params),
        (params, covariate),
    )
    start = 0
    cutpoint_gradients = []

    for slot in bound.layout:
        end = start + slot.size

        if "cutpoint" in slot.site:
            cutpoint_gradients.append(quivers_gradients[0][start:end])

        start = end

    assert cutpoint_gradients
    assert all(bool((gradient != 0.0).any()) for gradient in cutpoint_gradients)
    assert bool((quivers_gradients[1] != 0.0).all())
    assert all(
        torch.allclose(left, right, atol=1e-4, rtol=1e-4)
        for left, right in zip(quivers_gradients, torch_gradients, strict=True)
    )


def test_the_rows_are_scored_at_the_parameter_vector_s_kind() -> None:
    runtime = QuiversRuntime()
    bound = compiler().compile(rung("verb_only")).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 1)

    double = runtime.observed_log_probabilities(bound, params)
    single = runtime.observed_log_probabilities(bound, params.to(torch.float32))

    assert single.dtype is torch.float32
    assert torch.allclose(double.to(torch.float32), single, atol=1e-4, rtol=0.0)


@pytest.mark.parametrize(("bound", "params"), bound_cases())
def test_the_rows_sum_to_the_score_the_program_itself_recorded(
    bound: BoundProgram, params: Tensor
) -> None:
    per_row = QuiversRuntime().observed_log_probabilities(bound, params)
    scored = recorded_score(bound, params)

    assert scored.shape == ()
    assert scored.dtype is params.dtype
    assert torch.allclose(per_row.sum(), scored, atol=1e-12, rtol=0.0)


@pytest.mark.parametrize(("bound", "params"), bound_cases())
def test_the_quivers_log_joint_is_the_torch_backend_s_log_joint(
    bound: BoundProgram, params: Tensor
) -> None:
    under_quivers = QuiversRuntime().log_joint(bound, params)
    under_torch = log_joint(bound, params)

    assert under_quivers.shape == ()
    assert under_quivers.dtype is torch.float64
    assert torch.allclose(under_quivers, under_torch, atol=1e-4, rtol=1e-5)


def test_public_builder_scores_every_element_of_an_indexed_draw() -> None:
    bound = compiler().compile(rung("verb_by_frame")).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 2)
    runtime = QuiversRuntime()
    sites = bound.sites(params)
    moved = {
        slot.site: sites[slot.site]
        + (0.5 if slot.site == "verb_by_frame_effects" else 0.0)
        for slot in bound.layout
    }
    shifted = torch.cat([moved[slot.site] for slot in bound.layout])
    indexed = next(
        draw for draw in bound.program.draws if draw.site == "verb_by_frame_effects"
    )

    assert indexed.index == ("VerbByFrame",)
    assert bound.sizes[indexed.index[0]] == SHAPES[0].verbs * SHAPES[0].frames
    assert indexed.prior.family == "Normal"
    assert torch.allclose(
        runtime.log_joint(bound, shifted) - runtime.log_joint(bound, params),
        log_joint(bound, shifted) - log_joint(bound, params),
        atol=1e-4,
        rtol=0.0,
    )


class PooledScore(nn.Module):
    """An affine score over one feature, standing in for a neural predictor."""

    def __init__(self) -> None:
        super().__init__()
        self.weight = nn.Parameter(torch.ones(1, dtype=torch.float64))
        self.bias = nn.Parameter(torch.zeros(1, dtype=torch.float64))

    @override
    def forward(self, features: Tensor) -> Tensor:
        """Return one score per row."""
        return self.weight * features + self.bias


def pooled() -> BoundProgram:
    return compiler().compile(rung("pooled_encoder")).over(design_of(SHAPES[0]))


def test_the_quivers_log_joint_reaches_a_module_s_column_through_the_host_data() -> (
    None
):
    bound = pooled()
    params = parameters_of(bound, 4).requires_grad_()
    module = PooledScore()
    scored = bound.with_readouts({POOLED: module(bound.host_data[POOLED])})

    joint = QuiversRuntime().log_joint(scored, params)
    under_quivers = torch.autograd.grad(
        joint, (params, module.weight, module.bias), retain_graph=True
    )
    under_torch = torch.autograd.grad(
        log_joint(scored, params), (params, module.weight, module.bias)
    )

    assert float(under_quivers[1]) != 0.0
    assert float(under_quivers[2]) != 0.0
    assert all(
        torch.allclose(left, right, atol=1e-4, rtol=1e-4)
        for left, right in zip(under_quivers, under_torch, strict=True)
    )


def test_the_quivers_log_joint_reaches_no_parameter_of_a_detached_column() -> None:
    bound = pooled()
    params = parameters_of(bound, 4).requires_grad_()
    module = PooledScore()
    detached = bound.with_readouts({POOLED: module(bound.host_data[POOLED]).detach()})

    QuiversRuntime().log_joint(detached, params).backward()

    assert module.weight.grad is None
    assert module.bias.grad is None
    assert params.grad is not None
    assert bool((params.grad != 0.0).any())


def test_the_quivers_rows_follow_a_module_s_column() -> None:
    bound = pooled()
    params = parameters_of(bound, 4)
    module = PooledScore()
    scored = bound.with_readouts({POOLED: module(bound.host_data[POOLED]) + 1.0})

    per_row = QuiversRuntime().observed_log_probabilities(scored, params)

    assert torch.allclose(per_row, log_likelihood(scored, params), atol=1e-6, rtol=0.0)
    assert not torch.allclose(
        per_row, log_likelihood(bound, params), atol=1e-6, rtol=0.0
    )


def test_a_parameter_vector_of_the_wrong_length_is_refused_before_running() -> None:
    runtime = QuiversRuntime()
    bound = compiler().compile(rung("verb_only")).over(design_of(SHAPES[0]))

    with pytest.raises(ValueError, match=f"over {bound.size} parameters"):
        runtime.observed_log_probabilities(
            bound, torch.zeros(bound.size + 1, dtype=torch.float64)
        )

    with pytest.raises(ValueError, match=f"over {bound.size} parameters"):
        runtime.log_joint(bound, torch.zeros(bound.size + 1, dtype=torch.float64))

    with pytest.raises(TypeError, match="not a floating-point kind"):
        runtime.observed_log_probabilities(
            bound, torch.zeros(bound.size, dtype=torch.int64)
        )
