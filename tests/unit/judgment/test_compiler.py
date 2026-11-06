import json
import math
import os
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
import torch
from quivers.effects import TraceHandler, clamp, run_program

from lofi.judgment import (
    DEFAULT_COEFFICIENT_PRIOR,
    DEFAULT_SCALE_PRIOR,
    ITEM,
    PARTICIPANT,
    PLATE,
    POSITION,
    PREDICTOR,
    RESPONSE,
    UNIT_RESPONSE,
    Apply,
    AuxiliaryParameter,
    Binary,
    Bind,
    Bounded,
    BoundedDesign,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    BoundedResponseSpec,
    BoundProgram,
    Choice,
    ClosedScale,
    ConditionalLikelihood,
    ConditionalPlan,
    Coordinate,
    CorrelatedRandomSlope,
    Cumulative,
    DesignCodec,
    DesignError,
    Draw,
    FactorTerm,
    FixedCutpointVariance,
    FixedDiscrimination,
    Function,
    Gather,
    IdentityLinkOnLogit,
    InteractionTerm,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkingSteps,
    LinkTerm,
    Literal,
    Metric,
    Name,
    NutsPlan,
    ObserveBounded,
    ObserveChoice,
    ObserveOrdered,
    OffsetTerm,
    Operator,
    ParameterFormula,
    Predictor,
    Prior,
    PriorBlock,
    PriorSpec,
    QuiversNutsFitter,
    RandomIntercept,
    RandomSlope,
    ResponseCompiler,
    ResponseProgram,
    ResponseSpec,
    RowColumns,
    ScaleIdentification,
    ScaleNormalization,
    Slot,
    Table,
    ThresholdPriors,
    TriangularTransform,
    TruncatedMarginalLikelihood,
    TruncatedMarginalPlan,
    index_column,
    latent,
    linear_steps,
    log_likelihood,
    logit_steps,
    morcela_steps,
    predictor_site,
    probability_steps,
)
from lofi.judgment.runtime import QuiversRuntime, build

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment import ResponseFamily, Step

TREE: Final = Path(__file__).resolve().parents[3]
LADDER: Final = TREE / "tests" / "data" / "judgment"
DIGEST: Final = """
import sys
from pathlib import Path

from lofi.judgment import ResponseCompiler, ResponseSpec

text = Path(sys.argv[1]).read_text(encoding="utf-8")
spec = ResponseSpec.model_validate_json(text)
sys.stdout.write(ResponseCompiler.installed().compile(spec).digest)
"""
LEVELS: Final = 7
ROWS: Final = 24
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
CUMULATIVE: Final = Cumulative()
PINNED: Final = FixedDiscrimination(value=1.0)
THRESHOLDS: Final = PriorBlock(thresholds=ThresholdPriors())
MORCELA: Final = LinkingSpec(
    family="morcela", columns={"length": TOKENS, "frequency": UNIGRAM}
)
SCALED: Final = LinkingSpec(family="scaled")
FAMILIES: Final = (
    "morcela",
    "logit",
    "probability",
    "linear",
    "identity",
    "scaled",
)


def scaled_steps(
    term: LinkTerm,
    prefix: str,
    /,
    *,
    unit_slope: bool = False,
    signed_slope: bool = False,
) -> LinkingSteps:
    del signed_slope
    if unit_slope:
        return LinkingSteps({}, Name(term.readout))

    return LinkingSteps(
        {f"{prefix}_slope": Prior("HalfNormal", 1.0)},
        Binary(Operator.MUL, Name(f"{prefix}_slope"), Name(term.readout)),
    )


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


def emitter() -> ResponseCompiler:
    return ResponseCompiler(
        {
            "morcela": morcela_steps,
            "logit": logit_steps,
            "probability": probability_steps,
            "linear": linear_steps,
            "identity": identity_steps,
            "scaled": scaled_steps,
        }
    )


def rows(count: int = ROWS) -> list[JudgmentRow]:
    return [
        JudgmentRow(
            f"annotator-{index % 4}",
            f"sentence-{index // 4}",
            index % LEVELS,
            RowColumns(
                factors={"verb": f"verb-{index % 3}", "frame": f"frame-{index % 2}"},
                readouts={
                    LM: -10.0 - index,
                    UNIGRAM: -50.0 - 2.0 * index,
                    TOKENS: 3.0 + index % 5,
                    P_GRAM: -0.25 - 0.5 * (index % 4),
                    P_GRAM_COMPLEMENT: math.log1p(-math.exp(-0.25 - 0.5 * (index % 4))),
                    POOLED: 0.1 * index,
                    ROOT_ENTROPY: 0.2 + 0.1 * (index % 5),
                },
            ),
        )
        for index in range(count)
    ]


def design(count: int = ROWS) -> JudgmentDesign:
    return DesignCodec(LEVELS, factors=FACTORS, readouts=READOUTS).encode(rows(count))


def spec(
    *terms: Predictor,
    family: ResponseFamily = CUMULATIVE,
    identification: ScaleIdentification = PINNED,
    priors: PriorBlock = THRESHOLDS,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
    levels: int = LEVELS,
    parameters: tuple[ParameterFormula, ...] = (),
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=identification,
        priors=priors,
        normalization=normalization,
        parameters=parameters,
    )


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


def run(
    program: BoundProgram,
    *,
    pinned: dict[str, Tensor] | None = None,
) -> tuple[TraceHandler, dict[str, Tensor]]:
    data = dict(program.host_data)
    handler = TraceHandler()

    with clamp(pinned or {}), handler:
        run_program(build(program), torch.zeros(1, 1), data)

    return handler, data


def body(program: BoundProgram) -> tuple[Step, ...]:
    return program.program.body


def assert_quivers_agrees(program: BoundProgram) -> None:
    """Check the built program scores every row as the Torch backend does."""
    params = torch.full((program.size,), 0.1, dtype=torch.float64)
    under_quivers = QuiversRuntime().observed_log_probabilities(program, params)
    under_torch = log_likelihood(program, params)

    assert torch.allclose(under_quivers, under_torch, atol=1e-6, rtol=0.0)


def mentioned(program: BoundProgram) -> frozenset[str]:
    """Return every name the program binds, reads or declares a set under."""
    names = set(program.program.host_columns) | set(program.sizes)

    for step in program.program.body:
        names.add(step.site)

        match step:
            case Draw(index=index):
                names.update(index)
            case Bind(expression=expression):
                names.update(expression.names)

    return frozenset(names)


@pytest.mark.parametrize("name", cumulative_rung_names())
def test_every_ladder_rung_emits_a_program_that_loads_and_runs(name: str) -> None:
    source = design()
    program = emitter().emit(rung(name), source)

    handler, _ = run(program)
    sites = handler.trace.sites
    joint = handler.total_log_joint(program.rows, torch.device("cpu"))

    assert bool(torch.isfinite(joint).all())
    assert sites[RESPONSE].is_observed
    assert torch.equal(sites[RESPONSE].value, source.responses)


def test_the_free_scalar_and_entropy_rungs_draw_signed_slopes() -> None:
    signed = emitter().emit(rung("p_gram_signed"), design())
    entropy = emitter().emit(rung("p_gram_entropy"), design())

    assert Draw("p_gram_slope", ("Unit",), Prior("Normal", 0.0, 1.0)) in body(signed)
    assert Draw("p_gram_slope", ("Unit",), Prior("Normal", 0.0, 1.0)) in body(entropy)
    assert Draw("root_entropy_slope", ("Unit",), Prior("Normal", 0.0, 1.0)) in body(
        entropy
    )
    assert ROOT_ENTROPY not in mentioned(signed)
    assert ROOT_ENTROPY in mentioned(entropy)


def test_the_log_joint_gradient_reaches_a_differentiable_readout_column() -> None:
    program = emitter().emit(rung("morcela_only"), design())
    column = program.host_data[LM].clone().requires_grad_()
    params = torch.full((program.size,), 0.1, dtype=torch.float64)

    QuiversRuntime().log_joint(program.with_readouts({LM: column}), params).backward()

    assert column.grad is not None
    assert column.grad.shape == (ROWS,)
    assert bool((column.grad != 0.0).all())


def test_the_program_sizes_its_sets_from_the_design_s_level_counts() -> None:
    program = emitter().emit(rung("morcela_by_verb_by_frame"), design())

    assert tuple(program.sizes) == (
        PLATE,
        "Cutpoint",
        "Unit",
        "Participant",
        "Frame",
        "Verb",
    )
    assert program.sizes == {
        PLATE: ROWS,
        "Cutpoint": LEVELS - 1,
        "Unit": 1,
        "Participant": 4,
        "Frame": 2,
        "Verb": 3,
    }
    assert program.rows == ROWS
    assert program.levels == LEVELS
    assert program.program.scalar_object == "Unit"


def test_a_participant_intercept_gives_each_row_its_participant_s_cutpoints() -> None:
    source = design()
    program = emitter().emit(rung("intercept_only"), source)

    handler, data = run(program)
    cutpoints = handler.trace.sites["cutpoints"].value
    by_level = handler.trace.sites["participant_cutpoints"].value

    offsets = Draw(
        "participant_cutpoint_offsets",
        ("Participant", "Cutpoint"),
        Prior("Normal", 0.0, 1.0),
    )

    assert offsets in body(program)
    assert Bind("cutpoints", Gather("participant_cutpoints", PARTICIPANT)) in body(
        program
    )
    assert cutpoints.shape == (ROWS, LEVELS - 1)
    assert by_level.shape == (4, LEVELS - 1)
    assert torch.equal(cutpoints, by_level[data[PARTICIPANT]])
    assert program.program.threshold_group == PARTICIPANT


def test_without_a_participant_intercept_the_cutpoints_are_one_row() -> None:
    source = design()
    program = emitter().emit(spec(FactorTerm(name="verb")), source)

    handler, _ = run(program)

    assert "participant_cutpoints" not in mentioned(program)
    assert handler.trace.sites["cutpoints"].value.shape == (LEVELS - 1,)
    assert program.program.threshold_group is None
    assert program.program.host_columns == (RESPONSE, "verb")


def test_an_ordinal_predictor_may_reuse_the_bounded_response_column_name() -> None:
    program = emitter().compile(spec(FactorTerm(name=UNIT_RESPONSE)))

    assert program.host_columns == (RESPONSE, UNIT_RESPONSE)


def test_a_random_intercept_is_non_centered_and_gathered_by_its_group() -> None:
    program = emitter().emit(spec(RandomIntercept(group="frame")), design())

    assert body(program)[:3] == (
        Draw("frame_intercept_scale", ("Unit",), DEFAULT_SCALE_PRIOR),
        Draw("frame_intercept_offsets", ("Frame",), Prior("Normal", 0.0, 1.0)),
        Bind(
            "frame_intercept_term",
            Binary(
                Operator.MUL,
                Name("frame_intercept_scale"),
                Gather("frame_intercept_offsets", "frame"),
            ),
        ),
    )


def test_a_factor_draws_one_coefficient_per_level() -> None:
    program = emitter().emit(spec(FactorTerm(name="verb")), design())

    assert body(program)[:3] == (
        Draw("verb_effects", ("Verb",), DEFAULT_COEFFICIENT_PRIOR),
        Bind("verb_term", Gather("verb_effects", "verb")),
        Bind(PREDICTOR, Name("verb_term")),
    )


def test_an_interaction_of_factors_draws_one_coefficient_per_cell() -> None:
    source = design()
    program = emitter().emit(rung("verb_by_frame"), source)

    _, data = run(program)

    assert program.sizes["VerbByFrame"] == 6
    assert Draw(
        "verb_by_frame_effects", ("VerbByFrame",), DEFAULT_COEFFICIENT_PRIOR
    ) in body(program)
    assert Bind(
        "verb_by_frame_term", Gather("verb_by_frame_effects", "verb_by_frame")
    ) in body(program)
    assert program.program.indices == {
        PARTICIPANT: (PARTICIPANT,),
        "verb_by_frame": ("verb", "frame"),
    }
    assert torch.equal(
        data["verb_by_frame"], source.factors["verb"] * 2 + source.factors["frame"]
    )


def test_an_interaction_of_a_factor_and_a_link_scales_the_linked_value_per_level() -> (
    None
):
    term = InteractionTerm(
        left=FactorTerm(name="verb"),
        right=LinkTerm(readout=P_GRAM, linking=SCALED),
    )
    source = design()
    program = emitter().emit(
        spec(term, identification=FixedCutpointVariance(value=1.0)), source
    )

    handler, data = run(program)

    assert body(program)[:5] == (
        Draw("verb_by_p_gram_effects", ("Verb",), DEFAULT_COEFFICIENT_PRIOR),
        Draw("verb_by_p_gram_p_gram_slope", ("Unit",), Prior("HalfNormal", 1.0)),
        Bind(
            "verb_by_p_gram_p_gram_linked",
            Binary(Operator.MUL, Name("verb_by_p_gram_p_gram_slope"), Name(P_GRAM)),
        ),
        Bind(
            "verb_by_p_gram_term",
            Binary(
                Operator.MUL,
                Gather("verb_by_p_gram_effects", "verb"),
                Name("verb_by_p_gram_p_gram_linked"),
            ),
        ),
        Bind(PREDICTOR, Name("verb_by_p_gram_term")),
    )
    assert handler.trace.sites["verb_by_p_gram_term"].value.shape == (ROWS,)
    assert program.program.readouts == (P_GRAM,)
    assert tuple(data) == (RESPONSE, "verb", P_GRAM)


def test_an_offset_enters_with_a_coefficient_of_one() -> None:
    source = design()
    program = emitter().emit(spec(OffsetTerm(readout=TOKENS)), source)

    handler, _ = run(program)

    assert body(program)[0] == Bind(
        f"{TOKENS}_term", Binary(Operator.MUL, Literal(1.0), Name(TOKENS))
    )
    assert torch.equal(
        handler.trace.sites[f"{TOKENS}_term"].value, source.readouts[TOKENS]
    )


def test_a_link_whose_expression_is_the_column_is_bound_as_one_times_it() -> None:
    source = design()
    program = emitter().emit(rung("pooled_encoder"), source)

    handler, _ = run(program)

    assert Bind(
        f"{POOLED}_term", Binary(Operator.MUL, Literal(1.0), Name(POOLED))
    ) in body(program)
    assert torch.equal(
        handler.trace.sites[f"{POOLED}_term"].value, source.readouts[POOLED]
    )


def test_a_random_slope_over_a_link_multiplies_the_link_s_site_per_group() -> None:
    link = LinkTerm(readout=P_GRAM, linking=SCALED)
    source = design()
    program = emitter().emit(
        spec(link, RandomSlope(group="participant", term=link)), source
    )

    handler, _ = run(program)

    assert body(program)[2:5] == (
        Draw("participant_p_gram_slope_scale", ("Unit",), DEFAULT_SCALE_PRIOR),
        Draw(
            "participant_p_gram_slope_offsets",
            ("Participant",),
            Prior("Normal", 0.0, 1.0),
        ),
        Bind(
            "participant_p_gram_slope_term",
            Binary(
                Operator.MUL,
                Binary(
                    Operator.MUL,
                    Name("participant_p_gram_slope_scale"),
                    Gather("participant_p_gram_slope_offsets", PARTICIPANT),
                ),
                Name("p_gram_term"),
            ),
        ),
    )
    assert (
        body(program).count(Draw("p_gram_slope", ("Unit",), Prior("HalfNormal", 1.0)))
        == 1
    )
    assert handler.trace.sites["participant_p_gram_slope_term"].value.shape == (ROWS,)


def test_a_random_slope_over_a_factor_varies_each_level_within_each_group() -> None:
    source = design()
    program = emitter().emit(
        spec(RandomSlope(group="participant", term=FactorTerm(name="verb"))), source
    )

    _, data = run(program)

    assert program.sizes["ParticipantByVerb"] == 12
    assert Draw(
        "participant_verb_slope_offsets",
        ("ParticipantByVerb",),
        Prior("Normal", 0.0, 1.0),
    ) in body(program)
    assert torch.equal(
        data["participant_by_verb"], source.participants * 3 + source.factors["verb"]
    )


def test_a_random_slope_over_a_link_the_predictor_has_not_declared_is_refused() -> None:
    link = LinkTerm(readout=P_GRAM, linking=SCALED)

    with pytest.raises(ValueError, match="must declare before it"):
        emitter().emit(spec(RandomSlope(group="participant", term=link)), design())


def correlated_program(count: int = ROWS) -> BoundProgram:
    """Return a program with one correlated participant intercept and slope."""
    link = LinkTerm(readout=P_GRAM, linking=SCALED)

    return emitter().emit(
        spec(link, CorrelatedRandomSlope(group=PARTICIPANT, term=link)),
        design(count),
    )


def correlated_parameters(program: BoundProgram, correlation: float) -> Tensor:
    """Return valid parameters with a stated two-coefficient correlation."""
    values = {
        slot.site: torch.zeros(slot.shape or (slot.size,), dtype=torch.float64)
        for slot in program.layout
    }
    values[f"{P_GRAM}_slope"].fill_(1.0)
    values[f"{PARTICIPANT}_{P_GRAM}_correlated_scales"].copy_(
        torch.tensor([1.5, 0.75], dtype=torch.float64)
    )
    values[f"{PARTICIPANT}_{P_GRAM}_correlated_correlation"].copy_(
        torch.tensor(
            [[1.0, 0.0], [correlation, math.sqrt(1.0 - correlation**2)]],
            dtype=torch.float64,
        )
    )
    values[f"{PARTICIPANT}_{P_GRAM}_correlated_offsets"].copy_(
        torch.tensor(
            [[0.2, -0.4], [-0.5, 0.1], [0.7, 0.3], [-0.2, 0.8]],
            dtype=torch.float64,
        )
    )
    values["cutpoint_increments"].fill_(1.0)

    return torch.cat([values[slot.site].reshape(-1) for slot in program.layout])


def test_a_correlated_random_slope_compiles_one_ordered_lkj_block() -> None:
    program = correlated_program()
    name = f"{PARTICIPANT}_{P_GRAM}_correlated"

    assert body(program)[2:8] == (
        Draw(
            f"{name}_scales",
            ("Unit",),
            DEFAULT_SCALE_PRIOR,
            event_shape=(2,),
        ),
        Draw(
            f"{name}_correlation",
            ("Unit",),
            Prior("LKJCholesky", 1.0),
            event_shape=(2, 2),
        ),
        Draw(
            f"{name}_offsets",
            ("Participant",),
            Prior("Normal", 0.0, 1.0),
            event_shape=(2,),
        ),
        Bind(
            f"{name}_coefficients",
            Binary(
                Operator.MUL,
                TriangularTransform(
                    Name(f"{name}_correlation"), Name(f"{name}_offsets")
                ),
                Name(f"{name}_scales"),
            ),
        ),
        Bind(
            f"{name}_term",
            Binary(
                Operator.ADD,
                Coordinate(Gather(f"{name}_coefficients", PARTICIPANT), 0),
                Binary(
                    Operator.MUL,
                    Coordinate(Gather(f"{name}_coefficients", PARTICIPANT), 1),
                    Name(P_GRAM),
                ),
            ),
        ),
        Bind(
            PREDICTOR,
            Binary(Operator.ADD, Name(f"{P_GRAM}_term"), Name(f"{name}_term")),
        ),
    )
    assert program.layout[1:4] == (
        Slot(f"{name}_scales", 2, (2,)),
        Slot(f"{name}_correlation", 4, (2, 2)),
        Slot(f"{name}_offsets", 8, (4, 2)),
    )


def test_a_correlated_random_slope_multiplies_the_covariate_not_its_fixed_term() -> (
    None
):
    linking = LinkingSpec(family="logit", columns={"complement": P_GRAM_COMPLEMENT})
    link = LinkTerm(readout=P_GRAM, linking=linking)
    program = emitter().emit(
        spec(link, CorrelatedRandomSlope(group=PARTICIPANT, term=link)), design()
    )
    first = correlated_parameters(program, 0.35)
    second = first.clone()
    slope = next(slot for slot in program.layout if slot.site == f"{P_GRAM}_slope")
    start = sum(slot.size for slot in program.layout[: program.layout.index(slope)])
    first[start] = 1.25
    second[start] = 3.75

    change = latent(program, second).location - latent(program, first).location
    covariate = program.host_data[P_GRAM] - program.host_data[P_GRAM_COMPLEMENT]
    expected = 2.5 * covariate

    assert torch.allclose(change, expected, atol=1e-12, rtol=0.0)


def test_one_correlation_factor_is_shared_across_group_levels_and_response_rows() -> (
    None
):
    first = correlated_program(ROWS)
    more = correlated_program(2 * ROWS)
    suffix = "correlation"
    first_slot = next(slot for slot in first.layout if slot.site.endswith(suffix))
    more_slot = next(slot for slot in more.layout if slot.site.endswith(suffix))
    draw = next(draw for draw in first.program.draws if draw.site.endswith(suffix))

    assert first_slot == more_slot
    assert (first_slot.size, first_slot.shape) == (4, (2, 2))
    assert draw.index == (first.program.scalar_object,)


def test_the_two_backends_read_the_same_correlated_predictor() -> None:
    program = correlated_program()
    identity = correlated_parameters(program, 0.0)
    correlated = correlated_parameters(program, 0.6)
    torch_rows = log_likelihood(program, correlated)
    quivers_rows = QuiversRuntime().observed_log_probabilities(program, correlated)

    assert not torch.allclose(
        latent(program, identity).location, latent(program, correlated).location
    )
    assert torch.allclose(torch_rows, quivers_rows, atol=1e-6, rtol=0.0)


def test_separate_random_intercept_and_slope_terms_remain_independent() -> None:
    link = LinkTerm(readout=P_GRAM, linking=SCALED)
    program = emitter().emit(
        spec(
            link,
            RandomIntercept(group=PARTICIPANT),
            RandomSlope(group=PARTICIPANT, term=link),
        ),
        design(),
    )

    assert all(draw.prior.family != "LKJCholesky" for draw in program.program.draws)
    assert sum(draw.site.endswith("_scale") for draw in program.program.draws) == 2


def test_a_held_discrimination_other_than_one_scales_the_predictor_and_cutpoints() -> (
    None
):
    source = design()
    program = emitter().emit(
        spec(FactorTerm(name="verb"), identification=FixedDiscrimination(value=2.5)),
        source,
    )

    handler, _ = run(program)

    assert Bind(
        "scaled_predictor", Binary(Operator.MUL, Literal(2.5), Name(PREDICTOR))
    ) in body(program)
    assert Bind(
        "scaled_cutpoints", Binary(Operator.MUL, Literal(2.5), Name("cutpoints"))
    ) in body(program)
    assert "discrimination" not in mentioned(program)
    assert program.observation == ObserveOrdered(
        RESPONSE, PLATE, location="scaled_predictor", thresholds="scaled_cutpoints"
    )
    assert torch.allclose(
        handler.trace.sites["scaled_predictor"].value,
        2.5 * handler.trace.sites[PREDICTOR].value,
    )


def test_a_discrimination_held_at_one_emits_no_scaling() -> None:
    program = emitter().emit(spec(FactorTerm(name="verb")), design())

    assert not any("discrimination" in name for name in mentioned(program))
    assert not any("scaled" in name for name in mentioned(program))


def test_a_drawn_discrimination_is_sampled_from_the_declared_prior() -> None:
    source = design()
    program = emitter().emit(
        spec(
            FactorTerm(name="verb"),
            family=Cumulative(discrimination=True),
            identification=IdentityLinkOnLogit(),
            priors=PriorBlock(
                thresholds=ThresholdPriors(),
                discrimination=PriorSpec(
                    distribution="LogNormal", parameters=(0.0, 0.5)
                ),
            ),
        ),
        source,
    )

    handler, _ = run(program)

    assert Draw("discrimination", ("Unit",), Prior("LogNormal", 0.0, 0.5)) in body(
        program
    )
    assert "discrimination" in handler.trace.latent_sites


def test_a_discrimination_over_a_group_is_drawn_per_level_and_gathered_per_row() -> (
    None
):
    source = design()
    program = emitter().emit(rung("ordinal_2pl"), source)

    handler, data = run(program)
    lines = body(program)

    assert program.sizes["Item"] == 6
    assert Draw("discrimination", ("Item",), Prior("LogNormal", 0.0, 0.5)) in lines
    assert Bind("discrimination_rows", Gather("discrimination", ITEM)) in lines
    assert (
        Bind(
            "discrimination_column",
            Table("discrimination_rows", PLATE, "Unit", width=1),
        )
        in lines
    )
    assert (
        Bind(
            "scaled_predictor",
            Binary(Operator.MUL, Name("discrimination_rows"), Name(PREDICTOR)),
        )
        in lines
    )
    assert (
        Bind(
            "scaled_cutpoints",
            Binary(Operator.MUL, Name("discrimination_column"), Name("cutpoints")),
        )
        in lines
    )
    assert torch.equal(data["item"], source.items)
    assert handler.trace.sites["discrimination"].value.shape == (6,)
    assert handler.trace.sites["discrimination_rows"].value.shape == (ROWS,)
    assert handler.trace.sites["scaled_cutpoints"].value.shape == (ROWS, LEVELS - 1)
    assert torch.allclose(
        handler.trace.sites["cutpoints"].value.var(dim=1, correction=0),
        torch.ones(ROWS),
    )
    assert torch.allclose(
        handler.trace.sites["scaled_cutpoints"].value,
        handler.trace.sites["discrimination"].value[source.items].unsqueeze(1)
        * handler.trace.sites["cutpoints"].value,
    )


def test_a_discrimination_over_a_factor_reads_that_factor_s_index() -> None:
    source = design()
    program = emitter().emit(
        spec(
            RandomIntercept(group="participant"),
            family=Cumulative(discrimination=True, discrimination_group="verb"),
            identification=IdentityLinkOnLogit(),
            priors=PriorBlock(
                thresholds=ThresholdPriors(),
                discrimination=PriorSpec(
                    distribution="LogNormal", parameters=(0.0, 0.5)
                ),
            ),
        ),
        source,
    )

    _, data = run(program)

    assert program.sizes["Verb"] == 3
    assert Draw("discrimination", ("Verb",), Prior("LogNormal", 0.0, 0.5)) in body(
        program
    )
    assert Bind("discrimination_rows", Gather("discrimination", "verb")) in body(
        program
    )
    assert torch.equal(data["verb"], source.factors["verb"])
    assert program.program.host_columns == (RESPONSE, PARTICIPANT, "verb")


def test_the_threshold_priors_are_the_declared_ones() -> None:
    program = emitter().emit(
        spec(
            FactorTerm(name="verb"),
            priors=PriorBlock(
                thresholds=ThresholdPriors(
                    base=PriorSpec(distribution="Cauchy", parameters=(0.0, 2.0)),
                    increment=PriorSpec(distribution="Exponential", parameters=(3.0,)),
                )
            ),
        ),
        design(),
    )

    assert Draw("cutpoint_base", ("Unit",), Prior("Cauchy", 0.0, 2.0)) in body(program)
    assert Draw(
        "cutpoint_increments", ("Cutpoint",), Prior("Exponential", 3.0)
    ) in body(program)


def test_a_held_discrimination_beside_a_family_that_draws_it_is_refused() -> None:
    with pytest.raises(
        dx.ValidationError, match="draws discrimination and the identification"
    ):
        spec(
            FactorTerm(name="verb"),
            family=Cumulative(discrimination=True),
            priors=PriorBlock(
                thresholds=ThresholdPriors(),
                discrimination=PriorSpec(
                    distribution="LogNormal", parameters=(0.0, 0.5)
                ),
            ),
        )


LOGIT: Final = LinkingSpec(family="logit", columns={"complement": P_GRAM_COMPLEMENT})
LINKED: Final = (
    LinkTerm(readout=P_GRAM, linking=LOGIT),
    LinkTerm(readout=LM, linking=MORCELA),
    LinkTerm(readout=POOLED, linking=SCALED),
)


def slope_sites(program: BoundProgram) -> tuple[str, ...]:
    return tuple(
        draw.site for draw in program.program.draws if draw.site.endswith("_slope")
    )


def test_the_identity_link_on_the_logit_scale_holds_every_link_slope_at_one() -> None:
    source = design()
    identified = emitter().emit(
        spec(*LINKED, identification=IdentityLinkOnLogit()), source
    )
    free = emitter().emit(spec(*LINKED), source)

    handler, _ = run(identified)
    lines = body(identified)

    assert slope_sites(identified) == ()
    assert slope_sites(free) == (
        f"{P_GRAM}_slope",
        f"{LM}_slope",
        f"{POOLED}_slope",
    )
    terms = {step.site: step.expression for step in lines if isinstance(step, Bind)}

    assert terms[f"{P_GRAM}_term"] == Binary(
        Operator.MUL,
        Literal(1.0),
        Binary(Operator.SUB, Name(P_GRAM), Name(P_GRAM_COMPLEMENT)),
    )
    assert terms[f"{POOLED}_term"] == Binary(Operator.MUL, Literal(1.0), Name(POOLED))
    morcela = terms[f"{LM}_term"]

    assert isinstance(morcela, Binary)
    assert isinstance(morcela.left, Binary)
    assert (morcela.left.operator, morcela.left.left) == (Operator.MUL, Literal(1.0))
    assert torch.equal(
        handler.trace.sites[f"{P_GRAM}_term"].value,
        source.readouts[P_GRAM] - source.readouts[P_GRAM_COMPLEMENT],
    )
    assert identified.size == free.size - len(slope_sites(free))


def test_a_slope_pinned_away_from_one_under_the_identity_link_is_refused() -> None:
    pinned = LinkingSpec(
        family="logit", columns={"complement": P_GRAM_COMPLEMENT}, fixed={"slope": 2.0}
    )
    with pytest.raises(
        dx.ValidationError, match=r"holds the slope at one, and the term pins"
    ):
        spec(
            LinkTerm(readout=P_GRAM, linking=pinned),
            identification=IdentityLinkOnLogit(),
        )

    assert f"{P_GRAM}_slope" not in mentioned(
        emitter().emit(spec(LinkTerm(readout=P_GRAM, linking=pinned)), design())
    )


def test_a_held_cutpoint_variance_scales_the_emitted_cutpoints() -> None:
    declared = spec(
        FactorTerm(name="verb"), identification=FixedCutpointVariance(value=1.0)
    )
    program = emitter().emit(declared, design())
    params = torch.zeros(program.size, dtype=torch.float64)
    cutpoints = latent(program, params).thresholds

    assert Bind(
        "cutpoint_scales",
        Apply(
            Function.SQRT,
            Binary(Operator.DIV, Literal(1.0), Name("cutpoint_variances")),
        ),
    ) in body(program)
    assert torch.allclose(cutpoints.var(correction=0), cutpoints.new_tensor(1.0))


def test_another_response_family_is_refused() -> None:
    declared = spec(FactorTerm(name="verb"), family=Metric(), priors=PriorBlock())

    with pytest.raises(TypeError, match="'metric' family compiles to no response"):
        emitter().emit(declared, design())


@pytest.mark.parametrize("name", ["p_gram_rank_consistent", "p_gram_corn"])
def test_a_rank_consistent_head_has_no_probabilistic_program(name: str) -> None:
    with pytest.raises(
        TypeError, match="'rank_consistent' family compiles to no response program"
    ):
        emitter().compile(rung(name))


def test_a_normalization_of_the_response_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="declares 'log' normalization"):
        spec(FactorTerm(name="verb"), normalization=ScaleNormalization.LOG)


def test_a_linking_family_the_emitter_does_not_compile_is_refused() -> None:
    declared = spec(LinkTerm(readout=P_GRAM, linking=LinkingSpec(family="probit")))

    with pytest.raises(ValueError, match="declares the linking family 'probit'"):
        ResponseCompiler({"morcela": morcela_steps}).emit(declared, design())


def test_a_design_without_rows_is_refused() -> None:
    with pytest.raises(ValueError, match="the design has none"):
        emitter().emit(rung("intercept_only"), design(0))


def test_a_design_over_other_response_levels_is_refused() -> None:
    with pytest.raises(ValueError, match="over 5 response levels was given"):
        emitter().emit(spec(FactorTerm(name="verb"), levels=5), design())


def test_a_factor_the_design_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match="no grouping factor 'tense'"):
        emitter().emit(spec(FactorTerm(name="tense")), design())


def test_a_readout_the_design_lacks_is_refused() -> None:
    declared = spec(OffsetTerm(readout="absent"))

    with pytest.raises(ValueError, match="reading the readout 'absent'"):
        emitter().emit(declared, design())


@pytest.mark.parametrize("readout", ["cutpoints", RESPONSE, PREDICTOR])
def test_a_readout_named_like_a_site_the_program_binds_is_refused(readout: str) -> None:
    source = DesignCodec(LEVELS, readouts=(readout,)).encode(
        [
            JudgmentRow(
                "annotator",
                f"sentence-{index}",
                index % LEVELS,
                RowColumns(factors={}, readouts={readout: 1.0}),
            )
            for index in range(ROWS)
        ]
    )

    with pytest.raises(ValueError, match="bound twice"):
        emitter().emit(spec(OffsetTerm(readout=readout)), source)


def test_a_factor_whose_object_is_one_the_family_declares_is_refused() -> None:
    source = DesignCodec(LEVELS, factors=("unit",)).encode(
        [
            JudgmentRow(
                "annotator",
                f"sentence-{index}",
                index % LEVELS,
                RowColumns(factors={"unit": f"u{index % 2}"}, readouts={}),
            )
            for index in range(ROWS)
        ]
    )

    with pytest.raises(ValueError, match="'Unit' would index"):
        emitter().emit(spec(FactorTerm(name="unit")), source)


def test_a_factor_named_like_a_product_index_column_is_refused() -> None:
    source = DesignCodec(LEVELS, factors=(*FACTORS, "verb_by_frame")).encode(
        [
            JudgmentRow(
                row.participant,
                row.item,
                row.response,
                RowColumns({**row.factors, "verb_by_frame": "cell"}, {}),
            )
            for row in rows()
        ]
    )
    declared = spec(
        FactorTerm(name="verb_by_frame"),
        InteractionTerm(left=FactorTerm(name="verb"), right=FactorTerm(name="frame")),
    )

    with pytest.raises(ValueError, match="indexes both"):
        emitter().emit(declared, source)


def test_a_factor_that_cannot_name_a_column_is_refused() -> None:
    source = DesignCodec(LEVELS, factors=("verb frame",)).encode(
        [
            JudgmentRow(
                "annotator",
                f"sentence-{index}",
                index % LEVELS,
                RowColumns(factors={"verb frame": "x"}, readouts={}),
            )
            for index in range(ROWS)
        ]
    )

    with pytest.raises(ValueError, match="must be an identifier"):
        emitter().emit(spec(FactorTerm(name="verb frame")), source)


def test_the_host_data_holds_the_response_the_indices_and_the_readouts() -> None:
    source = design()
    program = emitter().emit(rung("morcela_by_verb_by_frame"), source)

    data = program.host_data

    assert tuple(data) == (RESPONSE, PARTICIPANT, "frame", "verb", LM, UNIGRAM, TOKENS)
    assert torch.equal(data[RESPONSE], source.responses)
    assert torch.equal(data[PARTICIPANT], source.participants)
    assert torch.equal(data["verb"], source.factors["verb"])
    assert torch.equal(data[LM], source.readouts[LM])
    assert data[LM].dtype is torch.float64
    assert program.program.host_columns == tuple(data)


def test_a_readout_column_can_be_replaced_by_the_values_another_source_gives() -> None:
    program = emitter().emit(rung("pooled_encoder"), design())
    source = program.host_data[POOLED].clone().requires_grad_()
    column = 2.0 * source

    rebound = program.with_readouts({POOLED: column})

    assert rebound.host_data[POOLED] is column
    assert rebound.host_data[POOLED].grad_fn is not None
    assert not torch.equal(program.host_data[POOLED], column.detach())
    assert torch.equal(rebound.host_data[RESPONSE], program.host_data[RESPONSE])
    assert torch.equal(rebound.host_data[PARTICIPANT], program.host_data[PARTICIPANT])
    assert rebound.sizes == program.sizes
    assert rebound.layout == program.layout
    assert rebound.program == program.program


def test_a_replacement_column_lands_on_the_readout_it_is_named_for() -> None:
    program = emitter().emit(rung("morcela_only"), design())
    readouts = program.program.readouts
    column = torch.full((ROWS,), 7.0, dtype=torch.float64)
    params = torch.full((program.size,), 0.5, dtype=torch.float64)

    assert readouts == (LM, UNIGRAM, TOKENS)

    named = program.with_readouts({TOKENS: column})
    misplaced = program.with_readouts({LM: column})

    assert torch.equal(named.host_data[TOKENS], column)
    assert torch.equal(named.host_data[LM], program.host_data[LM])
    assert torch.equal(named.host_data[UNIGRAM], program.host_data[UNIGRAM])
    assert not torch.equal(
        latent(named, params).location, latent(misplaced, params).location
    )


def test_a_column_for_a_readout_the_program_does_not_read_is_refused() -> None:
    program = emitter().emit(rung("pooled_encoder"), design())

    assert program.program.readouts == (POOLED,)

    with pytest.raises(ValueError, match=f"was given a column for {TOKENS!r}"):
        program.with_readouts({TOKENS: torch.zeros(ROWS, dtype=torch.float64)})


def test_a_replacement_column_of_another_length_or_holding_a_gap_is_refused() -> None:
    program = emitter().emit(rung("pooled_encoder"), design())

    with pytest.raises(DesignError, match=rf"expected \({ROWS},\)"):
        program.with_readouts({POOLED: torch.zeros(ROWS + 1, dtype=torch.float64)})

    with pytest.raises(DesignError, match="contains a nonfinite value"):
        program.with_readouts(
            {POOLED: torch.full((ROWS,), float("nan"), dtype=torch.float64)}
        )


def test_a_compiled_program_binds_to_designs_of_different_dimensions() -> None:
    program = emitter().compile(rung("morcela_by_verb_by_frame"))
    other_verbs = DesignCodec(LEVELS, factors=FACTORS, readouts=READOUTS).encode(
        [
            JudgmentRow(
                row.participant,
                row.item,
                row.response,
                RowColumns(
                    {**row.factors, "verb": f"verb-{index % 4}"}, dict(row.readouts)
                ),
            )
            for index, row in enumerate(rows())
        ]
    )
    full = program.over(design())
    fewer = program.over(design(12))
    wider = program.over(other_verbs)

    assert (full.rows, fewer.rows, wider.rows) == (ROWS, 12, ROWS)
    assert full.sizes["Verb"] == 3
    assert wider.sizes["Verb"] == 4
    assert wider.layout != full.layout
    assert full.size == wider.size - 1
    assert repr(program) == (
        f"ResponseProgram(levels={LEVELS}, host_columns={full.program.host_columns!r})"
    )


def test_a_compiled_program_refuses_a_design_that_cannot_bind_it() -> None:
    program = emitter().compile(rung("morcela_by_verb_by_frame"))
    bare = DesignCodec(LEVELS, factors=FACTORS).encode(
        [
            JudgmentRow(
                row.participant,
                row.item,
                row.response,
                RowColumns(dict(row.factors), {}),
            )
            for row in rows()
        ]
    )
    other_levels = DesignCodec(5, factors=FACTORS, readouts=READOUTS).encode(
        [
            JudgmentRow(
                row.participant,
                row.item,
                row.response % 5,
                RowColumns(dict(row.factors), dict(row.readouts)),
            )
            for row in rows()
        ]
    )

    with pytest.raises(ValueError, match="with the readouts"):
        program.over(bare)

    with pytest.raises(ValueError, match="response levels was given"):
        program.over(other_levels)

    with pytest.raises(ValueError, match="the design has none"):
        program.over(design(0))


def test_the_layout_lays_out_each_drawn_site_flat_in_draw_order() -> None:
    program = emitter().emit(rung("verb_by_frame"), design())

    assert program.layout == (
        Slot("participant_intercept_scale", 1),
        Slot("participant_intercept_offsets", 4),
        Slot("verb_by_frame_effects", 6),
        Slot("cutpoint_base", 1),
        Slot("cutpoint_increments", LEVELS - 1),
        Slot("participant_cutpoint_spread", LEVELS - 1),
        Slot("participant_cutpoint_offsets", 4 * (LEVELS - 1)),
    )
    assert program.size == 1 + 4 + 6 + 1 + 6 + 6 + 24
    assert program.layout is program.layout
    assert tuple(slot.site for slot in program.layout) == tuple(
        draw.site for draw in program.program.draws
    )


def test_the_sites_of_a_parameter_vector_are_its_pieces_in_layout_order() -> None:
    program = emitter().emit(rung("verb_by_frame"), design())
    params = torch.arange(program.size, dtype=torch.float64)

    sites = program.sites(params)

    assert tuple(sites) == tuple(slot.site for slot in program.layout)
    assert torch.equal(sites["participant_intercept_scale"], params[:1])
    assert torch.equal(sites["participant_intercept_offsets"], params[1:5])
    assert torch.equal(sites["participant_cutpoint_offsets"], params[-24:])
    assert all(tuple(sites[slot.site].shape) == (slot.size,) for slot in program.layout)


def test_a_parameter_vector_of_the_wrong_shape_or_kind_is_refused() -> None:
    program = emitter().emit(rung("verb_by_frame"), design())

    with pytest.raises(ValueError, match=f"over {program.size} parameters"):
        program.sites(torch.zeros(program.size + 1, dtype=torch.float64))

    with pytest.raises(ValueError, match=f"over {program.size} parameters"):
        program.sites(torch.zeros((program.size, 1), dtype=torch.float64))

    with pytest.raises(TypeError, match="not a floating-point kind"):
        program.sites(torch.zeros(program.size, dtype=torch.int64))


def test_a_compiled_program_ends_with_its_observation_and_lists_its_draws() -> None:
    program = emitter().compile(rung("intercept_only"))

    assert program.observation == ObserveOrdered(
        RESPONSE, PLATE, location=PREDICTOR, thresholds="cutpoints"
    )
    assert program.steps[-1] == program.observation
    assert program.steps == (*program.body, program.observation)
    assert not any(isinstance(step, ObserveOrdered) for step in program.body)
    assert all(isinstance(draw, Draw) for draw in program.draws)
    assert program.draws[0] == Draw(
        "participant_intercept_scale", ("Unit",), DEFAULT_SCALE_PRIOR
    )
    assert program.objects == {"Participant": (PARTICIPANT,)}
    assert program.indices == {PARTICIPANT: (PARTICIPANT,)}
    assert program.readouts == ()
    assert program.levels == LEVELS


def program_of(*steps: Step) -> ResponseProgram:
    return ResponseProgram(
        levels=LEVELS,
        steps=steps,
        objects={},
        indices={},
        readouts=(P_GRAM,),
        objective=ConditionalPlan(),
        groups=(),
        cutpoint_object="Cutpoint",
        threshold_group=None,
        scalar_object="Unit",
    )


def test_a_program_is_its_draws_and_bindings_then_one_observation() -> None:
    draw = Draw("scale", ("Unit",), Prior("HalfNormal", 1.0))
    bind = Bind(PREDICTOR, Binary(Operator.MUL, Name("scale"), Name(P_GRAM)))
    observe = ObserveOrdered(
        RESPONSE, PLATE, location=PREDICTOR, thresholds="cutpoints"
    )

    program = program_of(draw, bind, observe)

    assert program.body == (draw, bind)
    assert program.observation == observe
    assert program.steps == (draw, bind, observe)
    assert program.draws == (draw,)

    with pytest.raises(TypeError, match=r"ends with its observation.*ends with Bind"):
        program_of(draw, bind)

    with pytest.raises(ValueError, match=r"ends with its observation.*no step"):
        program_of()

    with pytest.raises(TypeError, match=r"observes once, at its end.*at step 1 of 4"):
        program_of(draw, observe, bind, observe)


def test_index_column_reads_the_participant_the_item_and_a_factor() -> None:
    source = design()

    assert index_column(source, PARTICIPANT) == (source.participants, 4)
    assert index_column(source, ITEM) == (source.items, 6)
    assert index_column(source, "frame") == (source.factors["frame"], 2)

    with pytest.raises(ValueError, match="no grouping factor 'tense'"):
        index_column(source, "tense")


def test_the_compiler_names_the_families_it_compiles_and_the_program_its_columns() -> (
    None
):
    program = emitter().emit(rung("intercept_only"), design())

    assert emitter().families == FAMILIES
    assert repr(program) == (
        f"BoundProgram(rows={ROWS}, levels={LEVELS}, "
        f"host_columns=({RESPONSE!r}, {PARTICIPANT!r}))"
    )


CHOICE_LINK: Final = LinkTerm(
    readout=P_GRAM,
    linking=LinkingSpec(family="logit", columns={"complement": P_GRAM_COMPLEMENT}),
)
POSITION_TERM: Final = FactorTerm(name=POSITION)
TRIALS: Final = 12


def choice_design(alternatives: int = 2, levels: int = 2) -> JudgmentDesign:
    trials = [
        JudgmentRow(
            f"annotator-{trial % 4}",
            f"trial-{trial}",
            trial % levels,
            RowColumns(
                factors={"verb": f"verb-{trial % 3}", "frame": f"frame-{trial % 2}"},
                readouts={},
                alternative_readouts={
                    name: tuple(
                        -0.25 - 0.1 * ((trial + at) % 5) for at in range(alternatives)
                    )
                    for name in READOUTS
                },
            ),
            tuple(f"item-{trial}-{at}" for at in range(alternatives)),
        )
        for trial in range(TRIALS)
    ]

    return DesignCodec(
        levels, factors=FACTORS, readouts=READOUTS, alternatives=alternatives
    ).encode(trials)


def choice_spec(
    *terms: Predictor, family: Choice, levels: int = 2, priors: PriorBlock | None = None
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock() if priors is None else priors,
    )


def test_a_rating_design_has_no_presentation_position() -> None:
    with pytest.raises(ValueError, match="no presentation position"):
        index_column(design(), POSITION)


def test_the_presentation_position_runs_over_each_row_s_alternatives() -> None:
    column, levels = index_column(choice_design(3), POSITION)

    assert levels == 3
    assert column.tolist() == [0, 1, 2] * TRIALS


def test_a_choice_program_gathers_every_column_once_per_alternative() -> None:
    program = emitter().compile(
        choice_spec(
            CHOICE_LINK,
            POSITION_TERM,
            RandomSlope(group=PARTICIPANT, term=CHOICE_LINK),
            family=Choice(),
        )
    )
    bound = program.over(choice_design())

    assert program.alternatives == 2
    assert isinstance(program.observation, ObserveChoice)
    assert bound.host_data[PARTICIPANT].tolist() == [
        trial % 4 for trial in range(TRIALS) for _ in range(2)
    ]
    assert bound.host_data[RESPONSE].numel() == TRIALS
    assert bound.host_data[P_GRAM].numel() == 2 * TRIALS


def test_a_tied_choice_program_reads_the_weights_of_the_two_places() -> None:
    program = emitter().compile(
        choice_spec(
            CHOICE_LINK,
            POSITION_TERM,
            family=Choice(ties=True),
            levels=3,
            priors=PriorBlock(thresholds=ThresholdPriors()),
        )
    )
    bound = program.over(choice_design(2, 3))

    assert "alternative_contrast" in program.host_columns
    assert bound.host_data["alternative_contrast"].tolist() == [-1.0, 1.0]
    assert bound.sizes["Cutpoint"] == 2
    assert bound.sizes["Alternative"] == 2


def test_a_program_is_bound_to_a_design_showing_the_alternatives_it_scores() -> None:
    choice = emitter().compile(choice_spec(CHOICE_LINK, family=Choice(), levels=LEVELS))
    rating = emitter().compile(spec(OffsetTerm(readout=P_GRAM)))

    with pytest.raises(ValueError, match="over 7 alternatives per row was given"):
        choice.over(design())

    with pytest.raises(ValueError, match="over None alternatives per row was given"):
        rating.over(choice_design(LEVELS, LEVELS))


def test_a_program_names_both_the_alternative_set_and_the_count() -> None:
    with pytest.raises(ValueError, match="names both the set indexing them"):
        ResponseProgram(
            levels=2,
            steps=(ObserveChoice(RESPONSE, PLATE, logits="choice_logits"),),
            objects={},
            indices={},
            readouts=(P_GRAM,),
            objective=ConditionalPlan(),
            groups=(),
            cutpoint_object=None,
            threshold_group=None,
            scalar_object="Unit",
            alternative_object="Alternative",
        )


def bounded_spec(*, scale: float | None = 0.2) -> BoundedResponseSpec:
    priors = (
        PriorBlock()
        if scale is not None
        else PriorBlock(
            scale=PriorSpec(distribution="LogNormal", parameters=(-1.0, 0.5))
        )
    )

    return BoundedResponseSpec(
        family=Bounded(scale=scale),
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
        priors=priors,
    )


def bounded_design() -> BoundedDesign:
    rows = (
        BoundedJudgmentRow(
            "participant-1",
            "item-low",
            0.0,
            ClosedScale(0.0, 100.0),
            RowColumns({}, {P_GRAM: -2.0, P_GRAM_COMPLEMENT: -0.1}),
        ),
        BoundedJudgmentRow(
            "participant-2",
            "item-high",
            7.0,
            ClosedScale(-3.0, 7.0),
            RowColumns({}, {P_GRAM: -0.1, P_GRAM_COMPLEMENT: -2.0}),
        ),
    )

    return BoundedDesignCodec(readouts=(P_GRAM, P_GRAM_COMPLEMENT)).encode(rows)


def test_a_fixed_bounded_program_has_no_draw_and_reads_normalized_responses() -> None:
    program = emitter().compile(bounded_spec())
    bound = program.over(bounded_design())

    assert program.levels is None
    assert program.host_columns[0] == UNIT_RESPONSE
    assert program.draws == ()
    assert bound.size == 0
    assert bound.host_data[UNIT_RESPONSE].tolist() == [0.0, 1.0]
    assert isinstance(program.observation, ObserveBounded)


def test_a_drawn_bounded_program_lays_out_one_positive_scale_parameter() -> None:
    bound = emitter().compile(bounded_spec(scale=None)).over(bounded_design())

    assert bound.layout == (Slot("bounded_scale", 1),)
    assert bound.size == 1


def test_a_bounded_program_refuses_an_ordinal_design() -> None:
    program = emitter().compile(bounded_spec())

    with pytest.raises(ValueError, match="None response levels"):
        program.over(design())


def test_a_program_weights_every_place_of_a_difference_it_takes() -> None:
    with pytest.raises(ValueError, match="weights each of 2 alternatives"):
        ResponseProgram(
            levels=2,
            steps=(ObserveChoice(RESPONSE, PLATE, logits="choice_logits"),),
            objects={},
            indices={},
            readouts=(P_GRAM,),
            objective=ConditionalPlan(),
            groups=(),
            cutpoint_object=None,
            threshold_group=None,
            scalar_object="Unit",
            alternative_object="Alternative",
            alternatives=2,
            contrast="alternative_contrast",
            contrast_weights=(-1.0, 0.0, 1.0),
        )


def test_a_choice_program_has_no_location_and_thresholds_to_read() -> None:
    bound = (
        emitter()
        .compile(choice_spec(CHOICE_LINK, family=Choice()))
        .over(choice_design())
    )

    with pytest.raises(TypeError, match="ordered observation is scored under"):
        latent(bound, torch.zeros(bound.size, dtype=torch.float64))


VARIANCE: Final = FixedCutpointVariance(value=1.0)
DISCRIMINATION: Final = AuxiliaryParameter.DISCRIMINATION
P_GRAM_TERM: Final = LinkTerm(
    readout=P_GRAM,
    linking=LinkingSpec(family="logit", columns={"complement": P_GRAM_COMPLEMENT}),
)


def predicting_spec(*terms: Predictor) -> ResponseSpec:
    return spec(
        P_GRAM_TERM,
        identification=VARIANCE,
        parameters=(ParameterFormula(parameter=DISCRIMINATION, predictor=terms),),
    )


def test_the_site_of_an_auxiliary_parameter_names_its_logarithm() -> None:
    assert predictor_site(DISCRIMINATION) == "log_discrimination"


def test_a_second_formula_reaches_the_program_beside_the_first() -> None:
    program = emitter().emit(predicting_spec(FactorTerm(name="frame")), design())
    lines = body(program)

    assert Bind(PREDICTOR, Name("p_gram_term")) in lines
    assert Bind("log_discrimination", Name("discrimination_frame_term")) in lines
    assert (
        Bind("discrimination", Apply(Function.EXP, Name("log_discrimination"))) in lines
    )
    assert (
        Bind(
            "scaled_predictor",
            Binary(Operator.MUL, Name("discrimination"), Name(PREDICTOR)),
        )
        in lines
    )
    assert [slot.site for slot in program.layout] == [
        "p_gram_slope",
        "discrimination_frame_effects",
        "cutpoint_base",
        "cutpoint_increments",
    ]
    assert_quivers_agrees(program)


def test_a_single_formula_leaves_the_discrimination_out_of_the_program() -> None:
    both = emitter().emit(predicting_spec(FactorTerm(name="frame")), design())
    alone = emitter().emit(spec(P_GRAM_TERM, identification=VARIANCE), design())

    varying = latent(both, torch.zeros(both.size, dtype=torch.float64))
    constant = latent(alone, torch.zeros(alone.size, dtype=torch.float64))

    assert "discrimination" not in mentioned(alone)
    assert "log_discrimination" not in mentioned(alone)
    assert [slot.site for slot in alone.layout] == [
        "p_gram_slope",
        "cutpoint_base",
        "cutpoint_increments",
    ]
    assert varying.thresholds.shape == (both.rows, LEVELS - 1)
    assert constant.thresholds.shape == (LEVELS - 1,)


def test_two_formulas_over_one_factor_keep_their_coefficients_apart() -> None:
    program = emitter().emit(
        spec(
            P_GRAM_TERM,
            FactorTerm(name="verb"),
            identification=VARIANCE,
            parameters=(
                ParameterFormula(
                    parameter=DISCRIMINATION, predictor=(FactorTerm(name="verb"),)
                ),
            ),
        ),
        design(),
    )
    drawn = [step.site for step in program.program.draws]

    assert "verb_effects" in drawn
    assert "discrimination_verb_effects" in drawn
    assert Bind("verb_term", Gather("verb_effects", "verb")) in body(program)
    assert Bind(
        "discrimination_verb_term", Gather("discrimination_verb_effects", "verb")
    ) in body(program)
    assert sorted(program.program.objects) == ["Verb"]
    assert_quivers_agrees(program)


def test_a_random_slope_varies_the_link_term_of_the_formula_declaring_it() -> None:
    program = emitter().emit(
        predicting_spec(P_GRAM_TERM, RandomSlope(group=PARTICIPANT, term=P_GRAM_TERM)),
        design(),
    )
    lines = body(program)

    assert any(
        isinstance(step, Bind) and step.site == "discrimination_p_gram_term"
        for step in lines
    )
    assert (
        Bind(
            "discrimination_participant_p_gram_slope_term",
            Binary(
                Operator.MUL,
                Binary(
                    Operator.MUL,
                    Name("discrimination_participant_p_gram_slope_scale"),
                    Gather(
                        "discrimination_participant_p_gram_slope_offsets", PARTICIPANT
                    ),
                ),
                Name("discrimination_p_gram_term"),
            ),
        )
        in lines
    )
    assert (
        Bind(
            "log_discrimination",
            Binary(
                Operator.ADD,
                Name("discrimination_p_gram_term"),
                Name("discrimination_participant_p_gram_slope_term"),
            ),
        )
        in lines
    )
    assert_quivers_agrees(program)


def test_a_random_slope_refuses_a_link_term_another_formula_declared() -> None:
    declared = spec(
        P_GRAM_TERM,
        identification=VARIANCE,
        parameters=(
            ParameterFormula(
                parameter=DISCRIMINATION,
                predictor=(RandomSlope(group=PARTICIPANT, term=P_GRAM_TERM),),
            ),
        ),
    )

    with pytest.raises(ValueError, match="predictor must declare before it"):
        emitter().compile(declared)


def test_an_offset_read_by_two_formulas_is_bound_once_for_each() -> None:
    program = emitter().emit(
        spec(
            P_GRAM_TERM,
            OffsetTerm(readout=TOKENS),
            identification=VARIANCE,
            parameters=(
                ParameterFormula(
                    parameter=DISCRIMINATION, predictor=(OffsetTerm(readout=TOKENS),)
                ),
            ),
        ),
        design(),
    )
    lines = body(program)

    once = Binary(Operator.MUL, Literal(1.0), Name(TOKENS))

    assert Bind("length_term", once) in lines
    assert Bind("discrimination_length_term", once) in lines
    assert (
        Bind(
            PREDICTOR,
            Binary(Operator.ADD, Name("p_gram_term"), Name("length_term")),
        )
        in lines
    )
    assert Bind("log_discrimination", Name("discrimination_length_term")) in lines
    assert_quivers_agrees(program)


def test_a_choice_family_refuses_a_formula_over_an_auxiliary_parameter() -> None:
    with pytest.raises(
        dx.ValidationError, match="choice family takes no formula over an auxiliary"
    ):
        spec(
            LinkTerm(readout=POOLED, linking=SCALED),
            family=Choice(),
            identification=VARIANCE,
            priors=PriorBlock(),
            parameters=(
                ParameterFormula(
                    parameter=DISCRIMINATION, predictor=(FactorTerm(name="frame"),)
                ),
            ),
        )


def test_the_installed_compiler_holds_every_listed_linking_family() -> None:
    compiler = ResponseCompiler.installed()
    declared = spec(LinkTerm(readout=P_GRAM, linking=LOGIT))

    assert compiler.families == ("linear", "logit", "morcela", "probability")
    assert body(compiler.emit(declared, design())) == body(
        ResponseCompiler({"logit": logit_steps}).emit(declared, design())
    )

    with pytest.raises(ValueError, match="declares the linking family 'scaled'"):
        compiler.emit(spec(LinkTerm(readout=P_GRAM, linking=SCALED)), design())


def test_a_linking_family_a_distribution_lists_is_installed_without_an_edit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    site = tmp_path / "site"
    site.mkdir()
    module = f"lofi_linking_plugin_{tmp_path.name.replace('-', '_')}"
    (site / f"{module}.py").write_text(
        "from lofi.judgment.linear import linear_steps as steps\n",
        encoding="utf-8",
        newline="\n",
    )
    metadata = site / f"{module}-1.0.dist-info"
    metadata.mkdir()
    (metadata / "METADATA").write_text(
        f"Name: {module.replace('_', '-')}\nVersion: 1.0\n",
        encoding="utf-8",
        newline="\n",
    )
    (metadata / "entry_points.txt").write_text(
        f"[lofi.linking]\nshifted = {module}:steps\n", encoding="utf-8", newline="\n"
    )
    monkeypatch.syspath_prepend(str(site))

    assert ResponseCompiler.installed().families == (
        "linear",
        "logit",
        "morcela",
        "probability",
        "shifted",
    )


def test_the_truncated_marginal_objective_leaves_a_nuts_posterior_unchanged() -> None:
    conditional = spec(
        LinkTerm(readout=P_GRAM, linking=LOGIT), RandomIntercept(group=PARTICIPANT)
    )
    marginal = conditional.with_(
        objective=TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=6.0,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8,
            quadrature_points=1,
            quadrature_seed=0,
        )
    )
    source = design()
    joint, integrated = (
        emitter().emit(each, source) for each in (conditional, marginal)
    )
    plan = NutsPlan(warmup=5, retained=4, chains=2, seed=11)
    drawn, again = (QuiversNutsFitter(plan).fit(each) for each in (joint, integrated))

    assert isinstance(integrated.program.objective, TruncatedMarginalPlan)
    assert isinstance(joint.program.objective, ConditionalPlan)
    assert integrated.program.steps == joint.program.steps
    assert drawn.draws.keys() == again.draws.keys()

    for site, values in drawn.draws.items():
        assert torch.equal(values, again.draws[site])


def digest_in_a_fresh_interpreter(path: Path, seed: str) -> str:
    finished = subprocess.run(
        [sys.executable, "-", str(path)],
        input=DIGEST,
        capture_output=True,
        encoding="utf-8",
        check=True,
        cwd=TREE,
        env={**os.environ, "PYTHONHASHSEED": seed},
    )

    return finished.stdout


def test_a_program_s_digest_identifies_its_content_in_any_interpreter() -> None:
    path = LADDER / "ladder" / "intercept_only.json"
    compiled = ResponseCompiler.installed().compile(rung("intercept_only"))
    verb, again, frame = (
        emitter().compile(spec(FactorTerm(name=name)))
        for name in ("verb", "verb", "frame")
    )

    assert len(compiled.digest) == 64
    assert int(compiled.digest, 16) >= 0
    assert verb.digest == again.digest
    assert verb.digest != frame.digest
    assert {digest_in_a_fresh_interpreter(path, seed) for seed in ("1", "2")} == {
        compiled.digest
    }
