import math
from decimal import Decimal, localcontext
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from quivers.effects import TraceHandler, clamp, run_program
from tests.conformance.linking import LinkingSubject, LinkingSuite

from lofi.judgment import (
    COMPLEMENT,
    LOGIT,
    PROBABILITY,
    SLOPE,
    Apply,
    Binary,
    BoundProgram,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    FixedCutpointVariance,
    FixedDiscrimination,
    Function,
    IdentityLinkOnLogit,
    JudgmentDesign,
    JudgmentRow,
    Linking,
    LinkingSpec,
    LinkTerm,
    Literal,
    LogitLinking,
    Name,
    Operator,
    Prior,
    PriorBlock,
    ProbabilityLinking,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    ScaleIdentification,
    SpecialCase,
    ThresholdPriors,
    latent,
    log_category_probabilities,
    log_likelihood,
    log_odds,
    log_odds_spread,
    logit_steps,
    ordered_cutpoints,
    probability_steps,
    simulated_design,
)
from lofi.judgment.runtime import QuiversRuntime

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

LOG_MASS: Final = "p_gram"
LOG_COMPLEMENT: Final = "p_gram_complement"
LEVELS: Final = 7
ROWS: Final = 40
NEAR_ONE: Final = Decimal("1e-12")


def logit_term(**fixed: float) -> LinkTerm:
    return LinkTerm(
        readout=LOG_MASS,
        linking=LinkingSpec(
            family=LOGIT, columns={COMPLEMENT: LOG_COMPLEMENT}, fixed=fixed
        ),
    )


def probability_term() -> LinkTerm:
    return LinkTerm(readout=LOG_MASS, linking=LinkingSpec(family=PROBABILITY))


def pair_of(log_mass: float) -> dict[str, float]:
    return {LOG_MASS: log_mass, LOG_COMPLEMENT: math.log1p(-math.exp(log_mass))}


def design(rows: int = ROWS, *, seed: int = 3) -> JudgmentDesign:
    generator = torch.Generator().manual_seed(seed)
    log_masses = -0.05 - 4.0 * torch.rand(
        rows, generator=generator, dtype=torch.float64
    )
    codec = DesignCodec(LEVELS, readouts=(LOG_MASS, LOG_COMPLEMENT))

    return codec.encode(
        JudgmentRow(
            f"annotator-{index % 5}",
            f"sentence-{index // 5}",
            index % LEVELS,
            RowColumns(factors={}, readouts=pair_of(float(log_masses[index]))),
        )
        for index in range(rows)
    )


def odds(readouts: Mapping[str, Tensor]) -> Tensor:
    return readouts[LOG_MASS] - readouts[LOG_COMPLEMENT]


def mass(readouts: Mapping[str, Tensor]) -> Tensor:
    return torch.exp(readouts[LOG_MASS])


FREE_SLOPE: Final = torch.tensor([1.5], dtype=torch.float64)
NO_PARAMETERS: Final = torch.empty(0, dtype=torch.float64)


class TestLogitLinking(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            LogitLinking(logit_term()), design(), FREE_SLOPE, {"log_odds": odds}
        )


class TestLogitLinkingWithTheSlopePinnedAtOne(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            LogitLinking(logit_term(slope=1.0)),
            design(),
            NO_PARAMETERS,
            {"log_odds": odds},
        )


class TestLogitLinkingWithTheSlopePinnedElsewhere(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            LogitLinking(logit_term(slope=2.5)), design(), NO_PARAMETERS, {}
        )


class TestProbabilityLinking(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            ProbabilityLinking(probability_term()),
            design(),
            NO_PARAMETERS,
            {"mass": mass},
        )


@pytest.mark.parametrize(
    "linking",
    [LogitLinking(logit_term()), ProbabilityLinking(probability_term())],
)
def test_the_linkings_answer_the_port(linking: Linking) -> None:
    assert isinstance(linking, Linking)


def test_the_logit_domain_is_built_from_the_term_it_was_declared_by() -> None:
    declared = logit_term()
    linking = LogitLinking(declared)

    assert linking.domain.argument == LOG_MASS
    assert linking.domain.readouts == declared.readouts == (LOG_MASS, LOG_COMPLEMENT)
    assert linking.domain.parameter_names == (SLOPE,)
    assert linking.domain.parameters[SLOPE].low == 0.0
    assert linking.domain.columns[LOG_MASS].high == 0.0
    assert linking.domain.columns[LOG_COMPLEMENT].high == 0.0
    assert linking.pinned == {}


def test_the_probability_domain_is_the_log_mass_alone() -> None:
    linking = ProbabilityLinking(probability_term())

    assert linking.domain.argument == LOG_MASS
    assert linking.domain.readouts == (LOG_MASS,)
    assert linking.domain.parameter_names == ()
    assert linking.domain.columns[LOG_MASS].high == 0.0


def test_the_logit_contribution_is_the_slope_times_the_log_odds() -> None:
    source = design()
    linking = LogitLinking(logit_term())

    assert torch.equal(
        linking.contribution(source, FREE_SLOPE),
        1.5 * (source.readouts[LOG_MASS] - source.readouts[LOG_COMPLEMENT]),
    )


def test_the_probability_contribution_is_the_mass() -> None:
    source = design()
    linking = ProbabilityLinking(probability_term())

    assert torch.allclose(
        linking.contribution(source, NO_PARAMETERS),
        torch.exp(source.readouts[LOG_MASS]),
    )


def test_the_log_odds_are_read_from_the_pair_where_a_naive_logit_has_lost_them() -> (
    None
):
    with localcontext() as context:
        context.prec = 50
        mass = 1 - NEAR_ONE
        reference = float(mass.ln() - (1 - mass).ln())

    log_mass = math.log1p(-float(NEAR_ONE))
    log_complement = math.log(float(NEAR_ONE))
    naive = float(torch.logit(torch.tensor(1.0 - float(NEAR_ONE), dtype=torch.float64)))
    columns = DesignCodec(LEVELS, readouts=(LOG_MASS, LOG_COMPLEMENT)).encode(
        [
            JudgmentRow(
                "annotator",
                "sentence",
                0,
                RowColumns(
                    factors={},
                    readouts={LOG_MASS: log_mass, LOG_COMPLEMENT: log_complement},
                ),
            )
        ]
    )
    linked = LogitLinking(logit_term(slope=1.0)).contribution(columns, NO_PARAMETERS)

    assert abs(naive - reference) > 1e-6
    assert abs(float(linked[0]) - reference) < 1e-12
    assert torch.equal(
        linked,
        log_odds(columns.readouts[LOG_MASS], columns.readouts[LOG_COMPLEMENT]),
    )


def test_dynamic_range_is_reported_on_the_log_odds_scale() -> None:
    source = design()
    log_mass = source.readouts[LOG_MASS]
    log_complement = source.readouts[LOG_COMPLEMENT]
    linked = log_odds(log_mass, log_complement)

    assert torch.equal(
        log_odds_spread(log_mass, log_complement), linked.max() - linked.min()
    )
    assert not torch.equal(
        log_odds_spread(log_mass, log_complement),
        torch.exp(log_mass).max() - torch.exp(log_mass).min(),
    )


def test_the_log_odds_are_declared_as_the_case_at_a_unit_slope() -> None:
    assert LogitLinking(logit_term()).special_cases == (
        SpecialCase("log_odds", (1.0,)),
    )
    assert LogitLinking(logit_term(slope=1.0)).special_cases == (
        SpecialCase("log_odds", ()),
    )
    assert LogitLinking(logit_term(slope=2.0)).special_cases == ()
    assert ProbabilityLinking(probability_term()).special_cases == (
        SpecialCase("mass", ()),
    )


def test_a_pinned_slope_leaves_the_domain_and_is_held_in_the_contribution() -> None:
    source = design()
    linking = LogitLinking(logit_term(slope=2.0))

    assert linking.pinned == {SLOPE: 2.0}
    assert linking.domain.parameter_names == ()
    assert torch.equal(
        linking.contribution(source, NO_PARAMETERS),
        LogitLinking(logit_term()).contribution(
            source, torch.tensor([2.0], dtype=torch.float64)
        ),
    )


def test_the_identification_holds_the_slope_at_one_and_the_domain_drops_it() -> None:
    source = design()
    held = LogitLinking(logit_term(), unit_slope=True)
    agreeing = LogitLinking(logit_term(slope=1.0), unit_slope=True)

    assert held.pinned == agreeing.pinned == {SLOPE: 1.0}
    assert held.domain.parameter_names == ()
    assert held.special_cases == (SpecialCase("log_odds", ()),)
    assert torch.equal(
        held.contribution(source, NO_PARAMETERS),
        log_odds(source.readouts[LOG_MASS], source.readouts[LOG_COMPLEMENT]),
    )
    assert repr(held) == repr(agreeing)


def test_a_slope_pinned_away_from_one_is_refused_under_the_identification() -> None:
    with pytest.raises(
        ValueError, match=r"holds the slope at one, and the term pins it at 2\.5"
    ):
        LogitLinking(logit_term(slope=2.5), unit_slope=True)

    with pytest.raises(ValueError, match="holds the slope at one"):
        logit_steps(logit_term(slope=0.5), "p_gram", unit_slope=True)


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (
            LinkTerm(readout=LOG_MASS, linking=LinkingSpec(family="morcela")),
            "declared under the family 'morcela'",
        ),
        (
            LinkTerm(readout=LOG_MASS, linking=LinkingSpec(family=LOGIT)),
            "reads the role 'complement'",
        ),
        (
            LinkTerm(
                readout=LOG_MASS,
                linking=LinkingSpec(
                    family=LOGIT,
                    columns={COMPLEMENT: LOG_COMPLEMENT, "length": "n"},
                ),
            ),
            "reads the role 'complement'",
        ),
        (
            LinkTerm(
                readout=LOG_MASS,
                linking=LinkingSpec(
                    family=LOGIT,
                    columns={COMPLEMENT: LOG_COMPLEMENT},
                    fixed={"beta": 1.0},
                ),
            ),
            r"pinned: \['beta'\]",
        ),
        (
            LinkTerm(
                readout=LOG_MASS,
                linking=LinkingSpec(family=LOGIT, columns={COMPLEMENT: LOG_MASS}),
            ),
            "both the argument and the complement",
        ),
    ],
)
def test_a_logit_term_the_linking_cannot_read_is_refused(
    declared: LinkTerm, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        LogitLinking(declared)


@pytest.mark.parametrize(
    ("declared", "message"),
    [
        (
            LinkTerm(readout=LOG_MASS, linking=LinkingSpec(family=LOGIT)),
            "declared under the family 'logit'",
        ),
        (
            LinkTerm(
                readout=LOG_MASS,
                linking=LinkingSpec(
                    family=PROBABILITY, columns={COMPLEMENT: LOG_COMPLEMENT}
                ),
            ),
            "reads its argument alone",
        ),
        (
            LinkTerm(
                readout=LOG_MASS,
                linking=LinkingSpec(family=PROBABILITY, fixed={SLOPE: 1.0}),
            ),
            "has no parameter",
        ),
    ],
)
def test_a_probability_term_the_linking_cannot_read_is_refused(
    declared: LinkTerm, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        ProbabilityLinking(declared)


def test_the_program_forms_draw_a_free_slope_and_write_a_held_one() -> None:
    free = logit_steps(logit_term(), "p_gram")
    signed = logit_steps(logit_term(), "p_gram", signed_slope=True)
    pinned = logit_steps(logit_term(slope=1.0), "p_gram")
    identified = logit_steps(logit_term(), "p_gram", unit_slope=True)
    probability = probability_steps(probability_term(), "p_gram", unit_slope=True)

    log_odds_of_mass = Binary(Operator.SUB, Name(LOG_MASS), Name(LOG_COMPLEMENT))

    assert tuple(free.draws) == ("p_gram_slope",)
    assert free.draws["p_gram_slope"] == Prior("HalfNormal", 1.0)
    assert signed.draws["p_gram_slope"] == Prior("Normal", 0.0, 1.0)
    assert free.expression == Binary(
        Operator.MUL, Name("p_gram_slope"), log_odds_of_mass
    )
    assert pinned.draws == identified.draws == {}
    assert pinned.expression == Binary(Operator.MUL, Literal(1.0), log_odds_of_mass)
    assert identified.expression == pinned.expression
    assert probability.draws == {}
    assert probability.expression == Apply(Function.EXP, Name(LOG_MASS))


@pytest.mark.parametrize(
    ("term", "linking", "parameters"),
    [
        (logit_term(), LogitLinking(logit_term()), FREE_SLOPE),
        (logit_term(slope=0.5), LogitLinking(logit_term(slope=0.5)), NO_PARAMETERS),
        (probability_term(), ProbabilityLinking(probability_term()), NO_PARAMETERS),
    ],
)
def test_the_program_form_evaluates_to_the_contribution(
    term: LinkTerm, linking: LogitLinking | ProbabilityLinking, parameters: Tensor
) -> None:
    source = design()
    steps = linking.steps(LOG_MASS)
    declared = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(term,),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )
    compiler = ResponseCompiler({LOGIT: logit_steps, PROBABILITY: probability_steps})
    program = compiler.emit(declared, source)
    drawn = dict(zip(steps.draws, parameters, strict=True))
    params = torch.cat(
        [
            drawn[slot.site].reshape(1)
            if slot.site in drawn
            else torch.zeros(slot.size, dtype=torch.float64)
            for slot in program.layout
        ]
    )
    handler = TraceHandler()

    with clamp(program.sites(params)), handler:
        run_program(
            QuiversRuntime().program(program),
            torch.zeros(1, 1, dtype=torch.float64),
            dict(program.host_data),
        )

    assert tuple(steps.draws) == tuple(
        draw.site for draw in program.program.draws if draw.site in drawn
    )
    assert torch.allclose(
        handler.trace.sites[f"{LOG_MASS}_term"].value,
        linking.contribution(source, parameters),
        atol=1e-12,
        rtol=0.0,
    )


def test_the_reprs_name_the_columns_and_the_pins() -> None:
    assert repr(LogitLinking(logit_term(slope=2.0))) == (
        f"LogitLinking(argument={LOG_MASS!r}, complement={LOG_COMPLEMENT!r}, "
        f"pinned={{'slope': 2.0}})"
    )
    assert repr(ProbabilityLinking(probability_term())) == (
        f"ProbabilityLinking(argument={LOG_MASS!r})"
    )


class Fit(NamedTuple):
    slope: float | None
    cutpoint_variance: float


THRESHOLDS: Final = PriorBlock(thresholds=ThresholdPriors())


GENERATING_SLOPE: Final = 1.5
GENERATING_CUTPOINTS: Final = torch.tensor(
    [-3.0, -1.5, -0.5, 0.5, 1.5, 3.0], dtype=torch.float64
)


def simulated(rows: int, *, seed: int) -> tuple[JudgmentDesign, Tensor]:
    source = design(rows, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    latent = LogitLinking(logit_term()).contribution(
        source, torch.tensor([GENERATING_SLOPE], dtype=torch.float64)
    )
    cutpoints = GENERATING_CUTPOINTS + latent.mean()
    probabilities = torch.exp(log_category_probabilities(latent, cutpoints))
    responses = torch.multinomial(probabilities, 1, generator=generator).reshape(-1)

    return source, responses


def pinned_variance(cutpoints: Tensor, variance: float) -> Tensor:
    centered = cutpoints - cutpoints.mean()

    return cutpoints.mean() + centered * math.sqrt(variance) / centered.std(
        correction=0
    )


def minimized(parameters: list[Tensor], objective: Callable[[], Tensor]) -> None:
    optimizer = torch.optim.LBFGS(
        parameters, lr=0.5, max_iter=300, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        loss = objective()
        loss.backward()

        return loss

    for _ in range(4):
        optimizer.step(closure)


def fitted_under_a_pinned_cutpoint_variance(
    source: JudgmentDesign, responses: Tensor, variance: float
) -> Fit:
    linking = LogitLinking(logit_term())
    slope = torch.ones(1, dtype=torch.float64, requires_grad=True)
    base = torch.zeros((), dtype=torch.float64, requires_grad=True)
    spans = torch.zeros(LEVELS - 1, dtype=torch.float64, requires_grad=True)

    def cutpoints() -> Tensor:
        return pinned_variance(ordered_cutpoints(base, spans), variance)

    def objective() -> Tensor:
        log_probabilities = log_category_probabilities(
            linking.contribution(source, slope), cutpoints()
        )

        return -log_probabilities.gather(1, responses.unsqueeze(1)).sum()

    minimized([slope, base, spans], objective)

    return Fit(float(slope.detach()[0]), float(cutpoints().detach().var(correction=0)))


def compiled(
    source: JudgmentDesign, responses: Tensor, identification: ScaleIdentification
) -> BoundProgram:
    declared = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(logit_term(),),
        identification=identification,
        priors=THRESHOLDS,
    )

    return ResponseCompiler({LOGIT: logit_steps}).emit(
        declared, simulated_design(source, responses)
    )


def fitted_program(program: BoundProgram) -> Fit:
    params = torch.zeros(program.size, dtype=torch.float64, requires_grad=True)

    minimized([params], lambda: -log_likelihood(program, params).sum())

    sites = program.sites(params.detach())
    slope = sites.get("p_gram_slope")
    cutpoints = latent(program, params.detach()).thresholds

    return Fit(
        None if slope is None else float(slope[0]),
        float(cutpoints.var(correction=0)),
    )


def test_the_slope_is_fitted_under_a_pinned_cutpoint_variance_and_one_otherwise() -> (
    None
):
    source, responses = simulated(1500, seed=11)
    variance = float(GENERATING_CUTPOINTS.var(correction=0))
    identified = compiled(source, responses, IdentityLinkOnLogit())
    free_signed = compiled(source, responses, FixedCutpointVariance(value=variance))

    constrained = fitted_under_a_pinned_cutpoint_variance(source, responses, variance)
    identity = fitted_program(identified)
    free = fitted_program(free_signed)

    assert [slot.site for slot in identified.layout] == [
        "cutpoint_base",
        "cutpoint_increments",
    ]
    assert [slot.site for slot in free_signed.layout] == [
        "p_gram_slope",
        "cutpoint_base",
        "cutpoint_increments",
    ]
    assert free_signed.program.draws[0].prior == Prior("Normal", 0.0, 1.0)
    assert constrained.slope is not None
    assert abs(constrained.slope - GENERATING_SLOPE) < 0.25
    assert abs(constrained.cutpoint_variance - variance) < 1e-9
    assert identity.slope is None
    assert abs(identity.cutpoint_variance - variance) > 0.5
    assert identity.cutpoint_variance < constrained.cutpoint_variance
    assert free.slope is not None
    assert abs(free.slope - GENERATING_SLOPE) < 0.25
    assert abs(free.cutpoint_variance - variance) < 1e-9
