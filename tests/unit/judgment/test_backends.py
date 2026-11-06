import json
import math
import random
from pathlib import Path
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
import torch
from tests.conformance.response_backend import (
    ResponseBackendCase,
    ResponseBackendConformance,
    ResponseBackendSuite,
    bind_response_backend,
    response_backends_agree,
)
from torch import nn
from torch.distributions import (
    Exponential,
    HalfNormal,
    LKJCholesky,
    LogNormal,
    Normal,
    Uniform,
)

from lofi.judgment import (
    ITEM,
    PARTICIPANT,
    PLATE,
    POSITION,
    RESPONSE,
    Binary,
    Bounded,
    BoundedDesign,
    BoundedDesignCodec,
    BoundedJudgmentRow,
    BoundedResponseSpec,
    BoundProgram,
    Choice,
    ClosedScale,
    ConditionalLikelihood,
    CorrelatedRandomSlope,
    Cumulative,
    CumulativeLinkHead,
    CumulativeParameters,
    DesignCodec,
    Draw,
    FactorTerm,
    FixedCutpointVariance,
    FixedDiscrimination,
    IdentityLinkOnLogit,
    InteractionTerm,
    JudgmentDesign,
    JudgmentRow,
    Latent,
    LinkingSpec,
    LinkingSteps,
    LinkTerm,
    MarginalLikelihoodError,
    Metric,
    Name,
    OffsetTerm,
    Operator,
    Predictor,
    Prior,
    PriorBlock,
    PriorSpec,
    QuiversBackend,
    RandomIntercept,
    RandomSlope,
    ResponseBackend,
    ResponseCompiler,
    ResponseFamily,
    ResponseProgram,
    ResponseSpec,
    RowColumns,
    ScaleIdentification,
    ScaleNormalization,
    ThresholdPriors,
    TorchBackend,
    TruncatedMarginalLikelihood,
    closed_unit_log_prob,
    latent,
    linear_steps,
    log_category_probabilities,
    log_joint,
    log_likelihood,
    log_prior,
    logit_steps,
    morcela_steps,
    probability_steps,
    response_log_probabilities,
    sample_prior,
)
from lofi.judgment.runtime import QuiversRuntime

if TYPE_CHECKING:
    from collections.abc import Iterator

    from torch import Tensor
    from torch.distributions import Distribution

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
GROUPS: Final = (PARTICIPANT, ITEM, *FACTORS)
MORCELA: Final = LinkingSpec(
    family="morcela", columns={"length": TOKENS, "frequency": UNIGRAM}
)
LOGIT: Final = LinkingSpec(family="logit", columns={"complement": P_GRAM_COMPLEMENT})
PROBABILITY: Final = LinkingSpec(family="probability")
IDENTITY: Final = LinkingSpec(family="identity")
SCALED: Final = LinkingSpec(family="scaled")
LINEAR: Final = LinkingSpec(family="linear")
THRESHOLDS: Final = PriorBlock(thresholds=ThresholdPriors())
CUMULATIVE: Final = Cumulative()
PINNED: Final = FixedDiscrimination(value=1.0)
DRAWN_DISCRIMINATION: Final = PriorBlock(
    thresholds=ThresholdPriors(),
    discrimination=PriorSpec(distribution="LogNormal", parameters=(0.0, 0.5)),
)
CUTPOINT_BASE: Final = "cutpoint_base"
DISCRIMINATION: Final = "discrimination"
PARAMETER_VECTORS: Final = 3
FUZZED_SPECIFICATIONS: Final = 500


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


FAMILIES: Final = (
    "morcela",
    "logit",
    "probability",
    "linear",
    "identity",
    "scaled",
)


def compiler() -> ResponseCompiler:
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


class Shape(NamedTuple):
    rows: int
    participants: int
    items: int
    verbs: int
    frames: int


SHAPES: Final = (Shape(24, 4, 6, 3, 2), Shape(35, 5, 7, 4, 3))


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


def spec(
    *terms: Predictor,
    family: ResponseFamily = CUMULATIVE,
    identification: ScaleIdentification = PINNED,
    priors: PriorBlock = THRESHOLDS,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=identification,
        priors=priors,
        normalization=normalization,
    )


def marginal_spec() -> ResponseSpec:
    """Return the current participant-local marginal declaration."""
    return ResponseSpec(
        family=CUMULATIVE,
        levels=LEVELS,
        objective=TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=3.0,
            relative_tolerance=1e-6,
            absolute_tolerance=1e-8,
            quadrature_points=16,
            quadrature_seed=19,
        ),
        predictor=(RandomIntercept(group=PARTICIPANT),),
        identification=PINNED,
        priors=THRESHOLDS,
    )


@pytest.mark.parametrize("kind", ["torch", "quivers"])
def test_a_row_backend_refuses_to_silently_score_the_marginal_treatment(
    kind: str,
) -> None:
    shared = compiler()
    backend = (
        TorchBackend(shared)
        if kind == "torch"
        else QuiversBackend(shared, QuiversRuntime())
    )
    source = design_of(Shape(rows=8, participants=2, items=8, verbs=2, frames=2))
    program = backend.compile(marginal_spec())
    params = torch.zeros(program.over(source).size, dtype=torch.float64)

    with pytest.raises(MarginalLikelihoodError, match="population layout"):
        backend.log_likelihood(program, params, source)


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
    """Return one value per element of a site, at a fifth of its prior's scale."""
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


def backends() -> tuple[QuiversBackend, ResponseCompiler]:
    shared = compiler()

    return QuiversBackend(shared, QuiversRuntime()), shared


def ladder_cases() -> tuple[ResponseBackendCase, ...]:
    shared = compiler()
    cases = []

    for name in cumulative_rung_names():
        declared = rung(name)
        program = shared.compile(declared)

        for shape in SHAPES:
            design = design_of(shape)
            bound = program.over(design)

            cases.extend(
                ResponseBackendCase(declared, parameters_of(bound, seed), design)
                for seed in range(PARAMETER_VECTORS)
            )

    return tuple(cases)


def conformance() -> ResponseBackendConformance:
    quivers, shared = backends()

    return ResponseBackendConformance(
        (
            bind_response_backend("torch", TorchBackend(shared)),
            bind_response_backend("quivers", quivers),
        ),
        ladder_cases(),
    )


class TestTorchAndQuiversAgree(ResponseBackendSuite):
    @pytest.fixture
    def conformance(self) -> ResponseBackendConformance:
        return conformance()


def test_the_law_binds_every_cumulative_fixture_at_three_vectors_and_two_designs() -> (
    None
):
    cases = ladder_cases()
    names = cumulative_rung_names()
    index = json.loads((LADDER / "ladder" / "ladder.json").read_text(encoding="utf-8"))
    heads = {name for names in index["ablations"].values() for name in names}

    assert set(rung_names()) - set(names) < heads
    assert {*index["rungs"], index["annotator"]} <= set(names)
    assert "p_gram_probability" in names
    assert "ordinal_2pl" in names
    assert len(cases) == len(names) * PARAMETER_VECTORS * len(SHAPES)
    assert all(case.design.n_rows > 0 for case in cases)
    assert {case.design.n_rows for case in cases} == {shape.rows for shape in SHAPES}


def test_the_two_backends_answer_the_port_and_share_one_compiler() -> None:
    shared = compiler()
    runtime = QuiversRuntime()
    quivers = QuiversBackend(shared, runtime)
    torch_backend = TorchBackend(shared)

    assert isinstance(torch_backend, ResponseBackend)
    assert isinstance(quivers, ResponseBackend)
    assert torch_backend.compiler is quivers.compiler
    assert (
        torch_backend.compile(rung("verb_only")).steps
        == quivers.compile(rung("verb_only")).steps
    )
    assert repr(torch_backend) == f"TorchBackend(families={FAMILIES!r})"
    assert repr(quivers) == (
        f"QuiversBackend(families={FAMILIES!r}, runtime=QuiversRuntime(built=0))"
    )
    assert quivers.runtime is runtime


@pytest.mark.parametrize("name", ["p_gram_rank_consistent", "p_gram_corn"])
def test_rank_consistent_heads_explicitly_refuse_both_probabilistic_backends(
    name: str,
) -> None:
    shared = compiler()
    probabilistic = (TorchBackend(shared), QuiversBackend(shared, QuiversRuntime()))

    for backend in probabilistic:
        with pytest.raises(
            TypeError, match="'rank_consistent' family compiles to no response program"
        ):
            backend.compile(rung(name))


def test_the_quivers_backend_asks_its_runtime_to_load_each_compiled_program() -> None:
    class Recording:
        def __init__(self) -> None:
            self.loaded: list[ResponseProgram] = []

        def load(self, program: ResponseProgram, /) -> None:
            self.loaded.append(program)

        def observed_log_probabilities(
            self, program: BoundProgram, params: Tensor, /
        ) -> Tensor:
            return QuiversRuntime().observed_log_probabilities(program, params)

    shared = compiler()
    runtime = Recording()
    quivers = QuiversBackend(shared, runtime)

    program = quivers.compile(rung("verb_only"))
    other = quivers.compile(rung("frame_only"))

    assert runtime.loaded == [program, other]
    assert other.steps != program.steps


def test_a_prior_quivers_lacks_is_refused_when_the_quivers_backend_compiles() -> None:
    quivers, shared = backends()
    declared = spec(
        LinkTerm(readout=P_GRAM, linking=LOGIT),
        family=Cumulative(discrimination=True),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock(
            thresholds=ThresholdPriors(),
            discrimination=PriorSpec(distribution="Wibble", parameters=(0.0, 0.5)),
        ),
    )
    program = TorchBackend(shared).compile(declared)

    assert Prior("Wibble", 0.0, 0.5) in {draw.prior for draw in program.draws}

    with pytest.raises(ValueError, match=r"Prior\('Wibble', 0.0, 0.5\).*no fixed"):
        quivers.compile(declared)


def test_the_law_names_a_backend_that_compiles_what_the_other_refuses() -> None:
    quivers, shared = backends()
    design = design_of(SHAPES[0])
    declared = spec(
        FactorTerm(name="verb"),
        family=Cumulative(discrimination=True),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock(
            thresholds=ThresholdPriors(),
            discrimination=PriorSpec(distribution="Wibble", parameters=(0.0, 0.5)),
        ),
    )
    bound = shared.compile(declared).over(design)
    cases = tuple(
        ResponseBackendCase(
            declared, torch.full((bound.size,), 0.1 * seed, dtype=torch.float64), design
        )
        for seed in range(PARAMETER_VECTORS)
    )

    with pytest.raises(Exception, match=r"asymmetric compilation.*failed=quivers"):
        response_backends_agree(
            ResponseBackendConformance(
                (
                    bind_response_backend("torch", TorchBackend(shared)),
                    bind_response_backend("quivers", quivers),
                ),
                cases,
            )
        )


def test_per_response_log_likelihoods_agree_on_a_design_with_responses() -> None:
    quivers, shared = backends()
    design = design_of(SHAPES[1])
    program = shared.compile(rung("morcela_by_verb_by_frame"))
    params = parameters_of(program.over(design), 11)

    under_torch = TorchBackend(shared).log_likelihood(program, params, design)
    under_quivers = quivers.log_likelihood(program, params, design)

    assert design.n_rows == SHAPES[1].rows
    assert under_torch.shape == (design.n_rows,)
    assert torch.allclose(under_torch, under_quivers, atol=1e-6, rtol=0.0)
    assert bool((under_torch < 0.0).all())
    assert not torch.allclose(under_torch, under_torch.roll(1))


class Algebra:
    """A seeded source of specifications drawn from the term algebra."""

    def __init__(self, seed: int) -> None:
        self._random = random.Random(seed)

    def factor(self) -> FactorTerm:
        return FactorTerm(name=self._random.choice(FACTORS))

    def link(self) -> LinkTerm:
        return self._random.choice(
            (
                LinkTerm(readout=LM, linking=MORCELA),
                LinkTerm(readout=P_GRAM, linking=LOGIT),
                LinkTerm(readout=P_GRAM, linking=PROBABILITY),
                LinkTerm(readout=ROOT_ENTROPY, linking=LINEAR),
                LinkTerm(readout=POOLED, linking=IDENTITY),
                LinkTerm(readout=POOLED, linking=SCALED),
            )
        )

    def fixed(self) -> Predictor:
        choice = self._random.random()

        if choice < 0.35:
            return self.factor()

        if choice < 0.7:
            return self.link()

        left, right = self.factor(), self.factor()

        if left == right:
            right = self.link()

        return InteractionTerm(left=left, right=right)

    def term(self, declared: tuple[Predictor, ...]) -> Predictor:
        choice = self._random.random()

        if choice < 0.4:
            return self.fixed()

        if choice < 0.7:
            return RandomIntercept(group=self._random.choice(GROUPS))

        if choice < 0.85:
            links = [term for term in declared if isinstance(term, LinkTerm)]
            varying = (
                self._random.choice(links)
                if links and self._random.random() < 0.7
                else self.fixed()
            )

            return RandomSlope(group=self._random.choice(GROUPS), term=varying)

        return OffsetTerm(readout=self._random.choice(READOUTS))

    def specification(self) -> ResponseSpec:
        terms: list[Predictor] = []

        for _ in range(self._random.randint(1, 4)):
            term = self.term(tuple(terms))

            if term not in terms:
                terms.append(term)

        choice = self._random.random()
        family: ResponseFamily = CUMULATIVE
        priors = THRESHOLDS
        identification: ScaleIdentification = PINNED

        if choice < 0.05:
            family, priors = Metric(), PriorBlock()
        elif choice < 0.2:
            family, priors = Cumulative(discrimination=True), DRAWN_DISCRIMINATION
            identification = IdentityLinkOnLogit()
        elif choice < 0.3:
            identification = FixedDiscrimination(value=1.25)
        elif choice < 0.35:
            identification = FixedCutpointVariance(value=1.0)

        if isinstance(identification, FixedDiscrimination) and any(
            isinstance(term, InteractionTerm) and term.readouts for term in terms
        ):
            identification = IdentityLinkOnLogit()

        return spec(
            *terms,
            family=family,
            identification=identification,
            priors=priors,
        )

    def specifications(self, count: int) -> Iterator[ResponseSpec]:
        for _ in range(count):
            yield self.specification()


class Outcome(NamedTuple):
    compiled: int
    refused: int
    asymmetric: int
    disagreeing: int
    lowest: float


def fuzz(count: int, seed: int) -> Outcome:
    quivers, shared = backends()
    torch_backend = TorchBackend(shared)
    design = design_of(SHAPES[0])
    compiled = refused = asymmetric = disagreeing = 0
    lowest = 0.0

    for index, declared in enumerate(Algebra(seed).specifications(count)):
        outcomes = []

        for backend in (torch_backend, quivers):
            try:
                compiled_program = backend.compile(declared)
            except (TypeError, ValueError) as error:
                outcomes.append((type(error), str(error)))
            else:
                outcomes.append(compiled_program)

        first, second = outcomes

        if isinstance(first, tuple) or isinstance(second, tuple):
            refused += 1

            if first != second:
                asymmetric += 1

            continue

        compiled += 1
        params = parameters_of(first.over(design), index)
        under_torch = torch_backend.log_likelihood(first, params, design)
        under_quivers = quivers.log_likelihood(second, params, design)
        lowest = min(lowest, float(under_torch.min()))

        if under_torch.shape != under_quivers.shape or not torch.allclose(
            under_torch, under_quivers, atol=1e-6, rtol=0.0
        ):
            disagreeing += 1

    return Outcome(compiled, refused, asymmetric, disagreeing, lowest)


def test_five_hundred_specifications_from_the_term_algebra_never_disagree() -> None:
    outcome = fuzz(FUZZED_SPECIFICATIONS, seed=2024)

    assert outcome.compiled + outcome.refused == FUZZED_SPECIFICATIONS
    assert outcome.compiled > FUZZED_SPECIFICATIONS // 2
    assert outcome.refused > 0
    assert outcome.lowest < -5.0
    assert outcome.asymmetric == 0
    assert outcome.disagreeing == 0


def test_a_refusal_is_the_same_refusal_under_both_backends() -> None:
    quivers, shared = backends()
    refused = (
        spec(FactorTerm(name="verb"), family=Metric(), priors=PriorBlock()),
        spec(
            RandomSlope(group=PARTICIPANT, term=LinkTerm(readout=P_GRAM, linking=LOGIT))
        ),
        spec(LinkTerm(readout=P_GRAM, linking=LinkingSpec(family="probit"))),
    )

    for declared in refused:
        with pytest.raises((TypeError, ValueError)) as under_torch:
            TorchBackend(shared).compile(declared)

        with pytest.raises((TypeError, ValueError)) as under_quivers:
            quivers.compile(declared)

        assert type(under_torch.value) is type(under_quivers.value)
        assert str(under_torch.value) == str(under_quivers.value)


def test_a_shifted_runtime_is_caught_by_the_law() -> None:
    class ShiftedRuntime:
        def __init__(self) -> None:
            self._runtime = QuiversRuntime()

        def load(self, program: ResponseProgram, /) -> None:
            self._runtime.load(program)

        def observed_log_probabilities(
            self, program: BoundProgram, params: Tensor, /
        ) -> Tensor:
            return self._runtime.observed_log_probabilities(program, params) + 1e-5

    shared = compiler()
    cases = ladder_cases()[:3]

    with pytest.raises(Exception, match="numerical disagreement"):
        response_backends_agree(
            ResponseBackendConformance(
                (
                    bind_response_backend("torch", TorchBackend(shared)),
                    bind_response_backend(
                        "quivers", QuiversBackend(shared, ShiftedRuntime())
                    ),
                ),
                cases,
            )
        )


def test_the_torch_backend_reproduces_the_cumulative_link_head() -> None:
    shared = compiler()
    design = design_of(SHAPES[0])
    declared = spec(FactorTerm(name="verb"))
    program = shared.compile(declared)
    bound = program.over(design)
    params = parameters_of(bound, 3)
    sites = bound.sites(params)

    class VerbEffects:
        @property
        def terms(self) -> tuple[Predictor, ...]:
            return declared.predictor

        def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
            return sites["verb_effects"][design.factors["verb"]]

    head = CumulativeLinkHead(declared, VerbEffects())
    under_head = head.log_likelihood(
        design,
        CumulativeParameters(sites["cutpoint_base"][0], sites["cutpoint_increments"]),
    )

    assert torch.allclose(
        log_likelihood(bound, params), under_head, atol=1e-12, rtol=0.0
    )


def item_response_reference(
    bound: BoundProgram, params: Tensor, design: JudgmentDesign
) -> Tensor:
    """Score each response under a graded response model written out directly.

    A participant's ability and an item's difficulty add on the latent
    scale, the item's discrimination multiplies the distance of every
    participant threshold from that sum, and a level's probability is the
    difference of the two adjacent cumulative logistic probabilities.
    """
    sites = bound.sites(params)
    thresholds = design.response_levels - 1
    participants = design.participants
    items = design.items
    ability = (
        sites["participant_intercept_scale"]
        * sites["participant_intercept_offsets"][participants]
    )
    difficulty = sites["item_intercept_scale"] * sites["item_intercept_offsets"][items]
    discrimination = sites["discrimination"][items]
    shifts = sites["participant_cutpoint_offsets"].reshape(-1, thresholds)
    increments = (
        sites["cutpoint_increments"] + sites["participant_cutpoint_spread"] * shifts
    )
    cutpoints = sites["cutpoint_base"] + torch.cumsum(
        torch.nn.functional.softplus(increments), dim=1
    )
    centered = cutpoints - cutpoints.mean(dim=1, keepdim=True)
    cutpoints = sites["cutpoint_base"] + centered / torch.sqrt(
        torch.mean(centered.square(), dim=1, keepdim=True)
    )
    distance = cutpoints[participants] - (ability + difficulty).unsqueeze(1)
    cumulative = torch.sigmoid(discrimination.unsqueeze(1) * distance)
    edges = torch.cat(
        (
            torch.zeros((design.n_rows, 1), dtype=params.dtype),
            cumulative,
            torch.ones((design.n_rows, 1), dtype=params.dtype),
        ),
        dim=1,
    )
    probabilities = edges[:, 1:] - edges[:, :-1]

    return torch.log(probabilities.gather(1, design.responses.unsqueeze(1))).squeeze(1)


def test_the_ordinal_two_parameter_fixture_scores_as_a_graded_response_model() -> None:
    shared = compiler()
    design = design_of(SHAPES[1])
    bound = shared.compile(rung("ordinal_2pl")).over(design)
    params = parameters_of(bound, 7)
    perturbed = params + 0.1

    assert torch.allclose(
        log_likelihood(bound, params),
        item_response_reference(bound, params, design),
        atol=1e-6,
        rtol=0.0,
    )
    assert not torch.allclose(
        log_likelihood(bound, perturbed),
        item_response_reference(bound, params, design),
        atol=1e-6,
        rtol=0.0,
    )


def test_the_ordinal_two_parameter_fixture_draws_its_four_parameter_families() -> None:
    design = design_of(SHAPES[1])
    bound = compiler().compile(rung("ordinal_2pl")).over(design)
    sizes = {slot.site: slot.size for slot in bound.layout}

    assert sizes["discrimination"] == len(design.item_levels)
    assert sizes["item_intercept_offsets"] == len(design.item_levels)
    assert sizes["participant_intercept_offsets"] == len(design.participant_levels)
    assert sizes["participant_cutpoint_offsets"] == len(design.participant_levels) * (
        LEVELS - 1
    )


def test_the_latent_is_what_the_likelihood_scores_each_row_under() -> None:
    shared = compiler()
    design = design_of(SHAPES[0])
    bound = shared.compile(rung("ordinal_2pl")).over(design)
    params = parameters_of(bound, 2)

    location, thresholds = latent(bound, params)

    assert isinstance(latent(bound, params), Latent)
    assert location.shape == (design.n_rows,)
    assert thresholds.shape == (design.n_rows, LEVELS - 1)
    assert torch.equal(
        log_likelihood(bound, params),
        log_category_probabilities(location, thresholds)
        .gather(1, design.responses.unsqueeze(1))
        .squeeze(1),
    )


def test_the_torch_log_likelihood_is_differentiable_in_every_site() -> None:
    shared = compiler()
    design = design_of(SHAPES[0])
    bound = shared.compile(rung("morcela_by_verb_by_frame")).over(design)
    params = parameters_of(bound, 5).requires_grad_()

    log_likelihood(bound, params).sum().backward()

    assert params.grad is not None
    assert params.grad.shape == params.shape
    assert bool((params.grad != 0.0).all())


def declared_density(draw: Draw, values: Tensor) -> float:
    """Return the log density of a site's values under the prior it declares."""
    half = 0.5 * math.log(2.0 * math.pi)

    match draw.prior.family, draw.prior.arguments:
        case "Normal", (location, scale):
            standardized = (values - location) / scale

            return float((-0.5 * standardized**2 - math.log(scale) - half).sum())
        case "HalfNormal", (scale,):
            standardized = values / scale

            return float(
                (-0.5 * standardized**2 - math.log(scale) - half + math.log(2.0)).sum()
            )
        case "LogNormal", (location, scale):
            standardized = (torch.log(values) - location) / scale

            return float(
                (
                    -0.5 * standardized**2 - math.log(scale) - half - torch.log(values)
                ).sum()
            )
        case "Uniform", (low, high):
            inside = (values >= low) & (values < high)
            held = values.new_full((), -math.log(high - low))

            return float(torch.where(inside, held, -math.inf).sum())
        case "Exponential", (rate,):
            held = math.log(rate) - rate * values

            return float(torch.where(values >= 0.0, held, -math.inf).sum())

    msg = f"no closed form stands in for the prior {draw.prior!r}"

    raise ValueError(msg)


@pytest.mark.parametrize("name", cumulative_rung_names())
def test_the_log_prior_is_every_drawn_site_s_density_under_its_own_prior(
    name: str,
) -> None:
    bound = compiler().compile(rung(name)).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 6)
    sites = bound.sites(params)
    expected = sum(
        declared_density(draw, sites[draw.site]) for draw in bound.program.draws
    )

    assert len(bound.program.draws) > 1
    assert log_prior(bound, params).shape == ()
    assert math.isclose(float(log_prior(bound, params)), expected, rel_tol=1e-9)


def test_the_log_joint_is_the_declared_prior_and_every_row_s_likelihood() -> None:
    program = compiler().compile(rung("morcela_by_verb_by_frame"))
    bound = program.over(design_of(SHAPES[0]))
    params = parameters_of(bound, 6)
    sites = bound.sites(params)
    prior = sum(
        declared_density(draw, sites[draw.site]) for draw in bound.program.draws
    )
    likelihood = float(log_likelihood(bound, params).sum())
    joint = float(log_joint(bound, params))

    assert log_joint(bound, params).shape == ()
    assert math.isclose(joint, prior + likelihood, rel_tol=1e-9)
    assert not math.isclose(joint, likelihood, rel_tol=1e-6)


def held_at(program: BoundProgram, site: str, value: float, seed: int) -> Tensor:
    """Return a drawn parameter vector with one site's elements held at a value."""
    params = parameters_of(program, seed)
    start = 0

    for slot in program.layout:
        if slot.site == site:
            params[start : start + slot.size] = value

        start += slot.size

    return params


def based_on(distribution: str, parameters: tuple[float, ...]) -> BoundProgram:
    """Return a bound program whose lowest threshold is drawn from a prior."""
    declared = spec(
        RandomIntercept(group=PARTICIPANT),
        priors=PriorBlock(
            thresholds=ThresholdPriors(
                base=PriorSpec(distribution=distribution, parameters=parameters)
            )
        ),
    )

    return compiler().compile(declared).over(design_of(SHAPES[0]))


@pytest.mark.parametrize(
    ("distribution", "parameters", "legal", "outside"),
    [
        ("HalfNormal", (1.0,), (0.0, 0.5, 4.0), -0.5),
        ("LogNormal", (0.0, 0.5), (0.05, 1.5, 9.0), -0.5),
        ("LogNormal", (0.0, 0.5), (0.05, 1.5, 9.0), 0.0),
        ("Uniform", (0.0, 1.0), (0.1, 0.5, 0.9), 2.0),
        ("Uniform", (0.0, 1.0), (0.1, 0.5, 0.9), -1.0),
        ("Exponential", (1.0,), (0.0, 0.5, 6.0), -3.0),
        ("Exponential", (1.0,), (0.0, 0.5, 6.0), -40.0),
    ],
)
def test_a_site_outside_its_prior_s_support_has_no_density_at_all(
    distribution: str,
    parameters: tuple[float, ...],
    legal: tuple[float, ...],
    outside: float,
) -> None:
    bound = based_on(distribution, parameters)
    params = held_at(bound, CUTPOINT_BASE, outside, 7)
    tracked = params.clone().requires_grad_()
    densities = [
        float(log_prior(bound, held_at(bound, CUTPOINT_BASE, value, 7)))
        for value in legal
    ]
    illegal = float(log_prior(bound, params))
    joint = float(log_joint(bound, params))
    gradient = torch.autograd.grad(log_prior(bound, tracked), tracked)[0]

    assert all(math.isfinite(density) for density in densities)
    assert not math.isnan(illegal)
    assert illegal == -math.inf
    assert illegal < min(densities)
    assert joint == -math.inf
    assert bool(torch.isfinite(gradient).all())
    assert not bool(bound.sites(gradient)[CUTPOINT_BASE].any())


PROBED: Final = (-40.0, -3.0, -0.5, 0.0, 0.25, 0.5, 2.0, 40.0)


def doubled(value: float) -> Tensor:
    """Return a double-precision scalar holding a value."""
    return torch.tensor(value, dtype=torch.float64)


@pytest.mark.parametrize(
    ("distribution", "parameters", "family"),
    [
        ("Normal", (0.5, 2.0), Normal(doubled(0.5), doubled(2.0))),
        ("HalfNormal", (2.0,), HalfNormal(doubled(2.0))),
        ("LogNormal", (0.0, 0.5), LogNormal(doubled(0.0), doubled(0.5))),
        ("Uniform", (-1.0, 3.0), Uniform(doubled(-1.0), doubled(3.0))),
        ("Exponential", (1.5,), Exponential(doubled(1.5))),
    ],
)
def test_a_site_has_a_density_exactly_where_its_own_family_has_one(
    distribution: str, parameters: tuple[float, ...], family: Distribution
) -> None:
    bound = based_on(distribution, parameters)
    support = family.support
    densities = {
        value: float(log_prior(bound, held_at(bound, CUTPOINT_BASE, value, 7)))
        for value in PROBED
    }

    assert support is not None

    held = {value: bool(support.check(doubled(value))) for value in PROBED}

    assert len(set(held.values())) == (1 if distribution == "Normal" else 2)
    assert all(math.isfinite(densities[value]) == held[value] for value in PROBED), (
        densities
    )
    assert all(densities[value] == -math.inf for value in PROBED if not held[value]), (
        densities
    )


@pytest.mark.parametrize(
    ("distribution", "parameters", "value"),
    [
        ("Normal", (0.5, 2.0), -1.25),
        ("HalfNormal", (2.0,), 1.25),
        ("LogNormal", (0.0, 0.5), 1.25),
        ("Uniform", (-1.0, 3.0), 1.25),
        ("Exponential", (1.5,), 1.25),
    ],
)
def test_every_family_with_a_declared_density_scores_a_legal_value_in_closed_form(
    distribution: str, parameters: tuple[float, ...], value: float
) -> None:
    bound = based_on(distribution, parameters)
    params = held_at(bound, CUTPOINT_BASE, value, 7)
    sites = bound.sites(params)
    base = next(draw for draw in bound.program.draws if draw.site == CUTPOINT_BASE)
    rest = sum(
        declared_density(draw, sites[draw.site])
        for draw in bound.program.draws
        if draw.site != CUTPOINT_BASE
    )

    assert base.prior.family == distribution
    assert base.prior.arguments == parameters
    assert math.isclose(
        float(log_prior(bound, params)),
        declared_density(base, sites[CUTPOINT_BASE]) + rest,
        rel_tol=1e-9,
    )


def test_a_discrimination_below_its_support_leaves_the_joint_no_density() -> None:
    declared = spec(
        RandomIntercept(group=PARTICIPANT),
        family=Cumulative(discrimination=True),
        identification=IdentityLinkOnLogit(),
        priors=DRAWN_DISCRIMINATION,
    )
    bound = compiler().compile(declared).over(design_of(SHAPES[0]))
    illegal = held_at(bound, DISCRIMINATION, -1.0, 7)
    legal = held_at(bound, DISCRIMINATION, 1.0, 7)

    assert math.isnan(float(log_likelihood(bound, illegal).sum()))
    assert float(log_prior(bound, illegal)) == -math.inf
    assert float(log_joint(bound, illegal)) == -math.inf
    assert math.isfinite(float(log_joint(bound, legal)))


def test_a_site_drawn_over_the_whole_line_has_a_density_everywhere_on_it() -> None:
    bound = based_on("Normal", (0.0, 1.0))
    densities = [
        float(log_prior(bound, held_at(bound, CUTPOINT_BASE, value, 7)))
        for value in (-40.0, 0.0, 40.0)
    ]

    assert all(math.isfinite(density) for density in densities)
    assert densities[1] > densities[0]
    assert densities[1] > densities[2]


def test_the_prior_term_of_the_log_joint_carries_a_gradient_of_its_own() -> None:
    bound = (
        compiler().compile(rung("morcela_by_verb_by_frame")).over(design_of(SHAPES[0]))
    )
    params = parameters_of(bound, 6).requires_grad_()
    joint = torch.autograd.grad(log_joint(bound, params), params)[0]
    prior = torch.autograd.grad(log_prior(bound, params), params)[0]
    likelihood = torch.autograd.grad(log_likelihood(bound, params).sum(), params)[0]
    constant = {
        draw.site for draw in bound.program.draws if draw.prior.family == "Uniform"
    }
    curved = {
        site: values
        for site, values in bound.sites(prior).items()
        if site not in constant
    }

    assert len(curved) > 1
    assert all(bool((values != 0.0).any()) for values in curved.values())
    assert torch.allclose(joint, prior + likelihood, atol=1e-12, rtol=0.0)
    assert not torch.allclose(joint, likelihood, atol=1e-6, rtol=0.0)


def correlated() -> BoundProgram:
    """Return a compiled correlated participant intercept-and-slope block."""
    link = LinkTerm(readout=P_GRAM, linking=LOGIT)
    declared = spec(
        link,
        CorrelatedRandomSlope(group=PARTICIPANT, term=link, concentration=2.0),
    )

    return compiler().compile(declared).over(design_of(SHAPES[0]))


def test_the_torch_prior_scores_the_correlation_as_one_lkj_event() -> None:
    bound = correlated()
    params = parameters_of(bound, 5).requires_grad_()
    correlation_site = next(
        draw.site for draw in bound.program.draws if draw.prior.family == "LKJCholesky"
    )
    correlation = bound.sites(params)[correlation_site]
    scored = log_prior(bound, params)
    gradient = bound.sites(torch.autograd.grad(scored, params)[0])[correlation_site]
    expected = LKJCholesky(2, 2.0).log_prob(correlation)
    expected_gradient = torch.autograd.grad(expected, correlation, retain_graph=True)[0]

    assert correlation.shape == (2, 2)
    assert torch.allclose(gradient, expected_gradient, atol=1e-12, rtol=0.0)
    assert bool((gradient != 0.0).any())


def test_a_prior_draw_is_laid_out_and_lies_where_every_prior_has_density() -> None:
    bound = correlated()
    drawn = sample_prior(bound, generator=torch.Generator().manual_seed(3))
    correlation_site = next(
        draw.site for draw in bound.program.draws if draw.prior.family == "LKJCholesky"
    )
    factor = bound.sites(drawn)[correlation_site]

    assert drawn.shape == (bound.size,)
    assert drawn.dtype == torch.float64
    assert bool(torch.isfinite(log_prior(bound, drawn)))
    assert torch.allclose(
        (factor @ factor.T).diagonal(), torch.ones(2, dtype=torch.float64)
    )


def test_prior_draws_follow_a_seed_and_advance_their_stream_alone() -> None:
    bound = correlated()
    generator = torch.Generator().manual_seed(11)
    torch.manual_seed(0)
    before = torch.get_rng_state()

    first = sample_prior(bound, generator=generator)
    second = sample_prior(bound, generator=generator)
    again = sample_prior(bound, generator=torch.Generator().manual_seed(11))

    assert torch.equal(first, again)
    assert not torch.equal(first, second)
    assert torch.equal(torch.get_rng_state(), before)


def test_prior_draws_of_a_site_follow_the_prior_it_declares() -> None:
    bound = (
        compiler()
        .compile(spec(LinkTerm(readout=P_GRAM, linking=LOGIT)))
        .over(design_of(SHAPES[0]))
    )
    generator = torch.Generator().manual_seed(5)
    base = next(draw for draw in bound.program.draws if draw.site == "cutpoint_base")
    location, scale = base.prior.arguments
    values = torch.stack(
        [
            bound.sites(sample_prior(bound, generator=generator))[base.site]
            for _ in range(2000)
        ]
    )

    assert abs(float(values.mean()) - location) < 4 * scale / math.sqrt(2000)
    assert abs(float(values.std()) / scale - 1.0) < 0.1


def test_quivers_and_torch_agree_on_lkj_density_differences_and_gradients() -> None:
    bound = correlated()
    first = parameters_of(bound, 7)
    second = first.clone()
    correlation_slot = next(
        slot for slot in bound.layout if slot.site.endswith("_correlation")
    )
    position = bound.layout.index(correlation_slot)
    start = sum(slot.size for slot in bound.layout[:position])
    other = 0.65
    second[start : start + correlation_slot.size] = torch.tensor(
        [1.0, 0.0, other, math.sqrt(1.0 - other**2)], dtype=torch.float64
    )
    runtime = QuiversRuntime()
    torch_difference = log_joint(bound, second) - log_joint(bound, first)
    quivers_difference = runtime.log_joint(bound, second) - runtime.log_joint(
        bound, first
    )
    torch_params = second.detach().requires_grad_()
    quivers_params = second.detach().requires_grad_()
    torch_gradient = torch.autograd.grad(log_joint(bound, torch_params), torch_params)[
        0
    ]
    quivers_gradient = torch.autograd.grad(
        runtime.log_joint(bound, quivers_params), quivers_params
    )[0]
    free_coordinates = slice(start + 2, start + correlation_slot.size)

    assert torch.allclose(quivers_difference, torch_difference, atol=1e-4, rtol=1e-5)
    assert torch.allclose(
        quivers_gradient[free_coordinates],
        torch_gradient[free_coordinates],
        atol=1e-4,
        rtol=1e-5,
    )


def test_a_site_drawn_from_a_family_without_a_declared_density_is_refused() -> None:
    declared = spec(
        RandomIntercept(group=PARTICIPANT),
        priors=PriorBlock(
            thresholds=ThresholdPriors(
                base=PriorSpec(distribution="Cauchy", parameters=(0.0, 1.0))
            )
        ),
    )
    bound = compiler().compile(declared).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 1)

    with pytest.raises(ValueError, match=r"Cauchy.*which has no density here"):
        log_prior(bound, params)

    with pytest.raises(ValueError, match=r"Cauchy.*which has no density here"):
        log_joint(bound, params)


def test_the_prior_and_the_joint_refuse_a_vector_the_layout_does_not_fit() -> None:
    bound = (
        compiler().compile(rung("morcela_by_verb_by_frame")).over(design_of(SHAPES[0]))
    )

    for score in (log_prior, log_joint):
        with pytest.raises(TypeError, match="not a floating-point kind"):
            score(bound, torch.zeros(bound.size, dtype=torch.int64))

        with pytest.raises(ValueError, match=f"over {bound.size} parameters"):
            score(bound, torch.zeros(bound.size + 1, dtype=torch.float64))


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


def predicting(*terms: Predictor) -> BoundProgram:
    """Return a bound program whose rows read a pooled column and a probability."""
    declared = spec(
        LinkTerm(readout=P_GRAM, linking=LOGIT),
        LinkTerm(readout=POOLED, linking=SCALED),
        *terms,
    )

    return compiler().compile(declared).over(design_of(SHAPES[0]))


def test_a_module_s_column_reaches_its_parameters_through_the_log_joint() -> None:
    bound = compiler().compile(rung("pooled_encoder")).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 4).requires_grad_()
    module = PooledScore()
    scored = bound.with_readouts({POOLED: module(bound.host_data[POOLED])})

    joint, weight, bias = torch.autograd.grad(
        log_joint(scored, params),
        (params, module.weight, module.bias),
        retain_graph=True,
    )
    prior = torch.autograd.grad(log_prior(scored, params), params)[0]
    likelihood = torch.autograd.grad(log_likelihood(scored, params).sum(), params)[0]

    assert float(weight) != 0.0
    assert float(bias) != 0.0
    assert bool((joint != 0.0).any())
    assert torch.allclose(joint, prior + likelihood, atol=1e-12, rtol=0.0)
    assert not torch.allclose(joint, likelihood, atol=1e-6, rtol=0.0)
    assert not torch.allclose(joint, prior, atol=1e-6, rtol=0.0)


def test_a_module_s_column_enters_the_readout_it_is_named_for_and_no_other() -> None:
    bound = predicting(RandomIntercept(group=PARTICIPANT))
    params = parameters_of(bound, 4)
    column = torch.linspace(0.5, 1.5, bound.rows, dtype=torch.float64)
    named = bound.with_readouts({POOLED: column})
    misplaced = bound.with_readouts({P_GRAM: column})

    assert bound.program.readouts == (P_GRAM, P_GRAM_COMPLEMENT, POOLED)
    assert torch.equal(named.host_data[POOLED], column)
    assert torch.equal(named.host_data[P_GRAM], bound.host_data[P_GRAM])
    assert torch.equal(
        named.host_data[P_GRAM_COMPLEMENT], bound.host_data[P_GRAM_COMPLEMENT]
    )
    assert not torch.allclose(
        latent(named, params).location, latent(bound, params).location
    )
    assert not torch.allclose(
        latent(named, params).location, latent(misplaced, params).location
    )
    assert float(log_joint(named, params)) != float(log_joint(misplaced, params))


def test_a_module_feeding_one_of_several_readouts_is_differentiated_through() -> None:
    bound = predicting(RandomIntercept(group=PARTICIPANT))
    params = parameters_of(bound, 4).requires_grad_()
    module = PooledScore()
    scored = bound.with_readouts({POOLED: module(bound.host_data[POOLED])})
    detached = bound.with_readouts({POOLED: module(bound.host_data[POOLED]).detach()})

    joint, weight, bias = torch.autograd.grad(
        log_joint(scored, params),
        (params, module.weight, module.bias),
        retain_graph=True,
    )
    prior = torch.autograd.grad(log_prior(scored, params), params)[0]
    likelihood = torch.autograd.grad(log_likelihood(scored, params).sum(), params)[0]
    control = torch.autograd.grad(
        log_joint(detached, params),
        (params, module.weight, module.bias),
        allow_unused=True,
    )

    assert float(weight) != 0.0
    assert float(bias) != 0.0
    assert torch.allclose(joint, prior + likelihood, atol=1e-12, rtol=0.0)
    assert not torch.allclose(joint, likelihood, atol=1e-6, rtol=0.0)
    assert control[0] is not None
    assert bool((control[0] != 0.0).any())
    assert control[1] is None
    assert control[2] is None


def test_a_detached_module_column_reaches_no_parameter_of_its_module() -> None:
    bound = compiler().compile(rung("pooled_encoder")).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 4).requires_grad_()
    module = PooledScore()
    detached = module(bound.host_data[POOLED]).detach()
    scored = bound.with_readouts({POOLED: detached})

    log_joint(scored, params).backward()

    assert module.weight.grad is None
    assert module.bias.grad is None
    assert params.grad is not None
    assert bool((params.grad != 0.0).any())


def test_the_quivers_runtime_scores_a_module_s_column_as_the_torch_backend_does() -> (
    None
):
    bound = compiler().compile(rung("pooled_encoder")).over(design_of(SHAPES[0]))
    params = parameters_of(bound, 4)
    module = PooledScore()
    scored = bound.with_readouts({POOLED: module(bound.host_data[POOLED])})

    per_row = QuiversRuntime().observed_log_probabilities(scored, params)

    assert per_row.shape == (bound.rows,)
    assert torch.allclose(per_row, log_likelihood(scored, params), atol=1e-6, rtol=0.0)


def test_a_parameter_vector_of_the_wrong_length_is_refused_before_scoring() -> None:
    shared = compiler()
    design = design_of(SHAPES[0])
    program = shared.compile(rung("verb_only"))
    bound = program.over(design)

    with pytest.raises(ValueError, match=f"over {bound.size} parameters"):
        TorchBackend(shared).log_likelihood(
            program, torch.zeros(bound.size + 1, dtype=torch.float64), design
        )


def test_the_observation_scores_each_row_in_log_space() -> None:
    shared = compiler()
    design = design_of(SHAPES[0])
    bound = shared.compile(rung("intercept_only")).over(design)
    params = parameters_of(bound, 9)
    sites = bound.sites(params)
    shifts = sites["participant_cutpoint_offsets"].reshape(-1, LEVELS - 1)
    spans = torch.cumsum(
        torch.nn.functional.softplus(
            sites["cutpoint_increments"] + sites["participant_cutpoint_spread"] * shifts
        ),
        dim=-1,
    )
    cutpoints = (sites["cutpoint_base"] + spans)[design.participants]
    latent = (
        sites["participant_intercept_scale"]
        * (sites["participant_intercept_offsets"][design.participants])
    )
    expected = log_category_probabilities(latent, cutpoints).gather(
        1, design.responses.unsqueeze(1)
    )

    assert torch.allclose(
        response_log_probabilities(bound, params),
        log_category_probabilities(latent, cutpoints),
    )
    assert torch.allclose(log_likelihood(bound, params), expected.squeeze(1))
    assert bound.host_data[RESPONSE] is design.responses
    assert bound.sizes[PLATE] == design.n_rows


CHOICE_TRIALS: Final = 128
CHOICE_LINK: Final = LinkTerm(readout=P_GRAM, linking=LOGIT)
POSITION_TERM: Final = FactorTerm(name=POSITION)
CHOICE_DRAWN: Final = PriorBlock(
    discrimination=PriorSpec(distribution="LogNormal", parameters=(0.0, 0.5))
)
NO_PRIORS: Final = PriorBlock()


def choice_mass(trial: int, at: int) -> float:
    return 0.05 + 0.9 * (((trial * 7 + at * 5) % 13) / 13.0)


def choice_design(alternatives: int, levels: int) -> JudgmentDesign:
    trials = tuple(
        JudgmentRow(
            f"annotator-{trial % 6}",
            f"trial-{trial}",
            (trial * 5) % levels,
            RowColumns(
                factors={},
                readouts={},
                alternative_readouts={
                    P_GRAM: tuple(
                        math.log(choice_mass(trial, at)) for at in range(alternatives)
                    ),
                    P_GRAM_COMPLEMENT: tuple(
                        math.log1p(-choice_mass(trial, at))
                        for at in range(alternatives)
                    ),
                },
            ),
            tuple(f"item-{trial}-{at}" for at in range(alternatives)),
        )
        for trial in range(CHOICE_TRIALS)
    )

    return DesignCodec(
        levels,
        readouts=(P_GRAM, P_GRAM_COMPLEMENT),
        alternatives=alternatives,
    ).encode(trials)


def choice_spec(
    *terms: Predictor,
    family: Choice,
    levels: int,
    priors: PriorBlock = NO_PRIORS,
) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=terms,
        identification=IdentityLinkOnLogit(),
        priors=priors,
    )


def choice_declarations() -> tuple[tuple[ResponseSpec, JudgmentDesign], ...]:
    return (
        (
            choice_spec(CHOICE_LINK, POSITION_TERM, family=Choice(), levels=2),
            choice_design(2, 2),
        ),
        (
            choice_spec(
                CHOICE_LINK,
                POSITION_TERM,
                family=Choice(ties=True),
                levels=3,
                priors=PriorBlock(thresholds=ThresholdPriors()),
            ),
            choice_design(2, 3),
        ),
        (
            choice_spec(
                CHOICE_LINK,
                POSITION_TERM,
                RandomSlope(group=PARTICIPANT, term=CHOICE_LINK),
                family=Choice(discrimination=True),
                levels=3,
                priors=CHOICE_DRAWN,
            ),
            choice_design(3, 3),
        ),
    )


@pytest.mark.parametrize("index", [0, 1, 2], ids=("two", "ties", "three"))
def test_the_quivers_joint_of_a_choice_is_the_torch_joint(index: int) -> None:
    declared, design = choice_declarations()[index]
    bound = compiler().compile(declared).over(design)
    params = parameters_of(bound, 5)

    assert torch.allclose(
        QuiversRuntime().log_joint(bound, params),
        log_joint(bound, params),
        atol=1e-4,
        rtol=1e-5,
    )


def choice_cases() -> tuple[ResponseBackendCase, ...]:
    shared = compiler()
    cases = []

    for declared, design in choice_declarations():
        bound = shared.compile(declared).over(design)
        cases.extend(
            ResponseBackendCase(declared, parameters_of(bound, seed), design)
            for seed in range(PARAMETER_VECTORS)
        )

    return tuple(cases)


def choice_conformance() -> ResponseBackendConformance:
    quivers, shared = backends()

    return ResponseBackendConformance(
        (
            bind_response_backend("torch", TorchBackend(shared)),
            bind_response_backend("quivers", quivers),
        ),
        choice_cases(),
    )


class TestTorchAndQuiversAgreeOnChoices(ResponseBackendSuite):
    @pytest.fixture
    def conformance(self) -> ResponseBackendConformance:
        return choice_conformance()


def test_the_law_binds_every_choice_declaration_at_three_vectors() -> None:
    cases = choice_cases()

    assert len(cases) == len(choice_declarations()) * PARAMETER_VECTORS
    assert {case.design.n_rows for case in cases} == {CHOICE_TRIALS}
    assert {case.design.alternatives_per_row for case in cases} == {2, 3}
    assert {case.spec.family.ties for case in cases} == {False, True}


def test_per_response_choice_log_likelihoods_agree_on_a_hundred_trials() -> None:
    quivers, shared = backends()
    declared, design = choice_declarations()[0]
    program = shared.compile(declared)
    params = parameters_of(program.over(design), 17)

    under_torch = TorchBackend(shared).log_likelihood(program, params, design)
    under_quivers = quivers.log_likelihood(program, params, design)
    distributions = response_log_probabilities(program.over(design), params)

    assert design.n_rows >= 100
    assert under_torch.shape == (design.n_rows,)
    assert distributions.shape == (design.n_rows, design.response_levels)
    assert torch.allclose(
        distributions.exp().sum(dim=-1),
        torch.ones(design.n_rows, dtype=torch.float64),
    )
    assert torch.allclose(
        distributions.gather(1, design.responses.unsqueeze(1)).squeeze(1),
        under_torch,
    )
    assert torch.allclose(under_torch, under_quivers, atol=1e-6, rtol=0.0)
    assert bool((under_torch < 0.0).all())
    assert not torch.allclose(under_torch, under_torch.roll(1))


def test_the_two_backends_compile_one_choice_declaration() -> None:
    quivers, shared = backends()
    declared = choice_declarations()[2][0]

    assert (
        TorchBackend(shared).compile(declared).steps == quivers.compile(declared).steps
    )


BOUNDED_TRIALS: Final = 128


def bounded_declaration(
    *terms: Predictor, scale: float | None = 0.2
) -> BoundedResponseSpec:
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
            *terms,
        ),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock()
        if scale is not None
        else PriorBlock(
            scale=PriorSpec(distribution="LogNormal", parameters=(-1.0, 0.5))
        ),
    )


def bounded_trials() -> BoundedDesign:
    rows = tuple(
        BoundedJudgmentRow(
            f"annotator-{trial % 8}",
            f"item-{trial}",
            0.0
            if trial == 0
            else 100.0
            if trial == BOUNDED_TRIALS - 1
            else float((trial * 37) % 101),
            ClosedScale(0.0, 100.0),
            RowColumns(
                {},
                {
                    P_GRAM: math.log(0.05 + 0.9 * (trial / BOUNDED_TRIALS)),
                    P_GRAM_COMPLEMENT: math.log(0.95 - 0.9 * (trial / BOUNDED_TRIALS)),
                },
            ),
        )
        for trial in range(BOUNDED_TRIALS)
    )

    return BoundedDesignCodec(readouts=(P_GRAM, P_GRAM_COMPLEMENT)).encode(rows)


@pytest.mark.parametrize("scale", [0.2, 1e16])
def test_per_response_bounded_densities_agree_at_a_hundred_trials_and_endpoints(
    scale: float,
) -> None:
    quivers, shared = backends()
    declared = bounded_declaration(scale=scale)
    design = bounded_trials()
    program = shared.compile(declared)
    params = torch.empty(0, dtype=torch.float64)
    under_torch = TorchBackend(shared).log_likelihood(program, params, design)
    under_quivers = quivers.log_likelihood(program, params, design)
    eta = design.readouts[P_GRAM] - design.readouts[P_GRAM_COMPLEMENT]
    expected = closed_unit_log_prob(
        design.unit_responses,
        torch.sigmoid(eta),
        torch.tensor(scale, dtype=torch.float64),
    )

    assert design.n_rows >= 100
    assert design.unit_responses[[0, -1]].tolist() == [0.0, 1.0]
    assert bool(torch.isfinite(under_torch).all())
    assert torch.allclose(under_torch, under_quivers, atol=1e-6, rtol=0.0)
    assert torch.allclose(under_torch, expected, atol=1e-12, rtol=1e-12)

    with pytest.raises(TypeError, match="finite response distribution"):
        response_log_probabilities(program.over(design), params)


@pytest.mark.parametrize("backend_name", ["torch", "quivers"])
def test_bounded_scale_and_predictor_parameters_receive_gradients(
    backend_name: str,
) -> None:
    quivers, shared = backends()
    declared = bounded_declaration(
        RandomIntercept(group=PARTICIPANT),
        scale=None,
    )
    design = bounded_trials()
    program = shared.compile(declared)
    bound = program.over(design)
    params = torch.full(
        (bound.size,),
        0.2,
        dtype=torch.float64,
        requires_grad=True,
    )
    backend = TorchBackend(shared) if backend_name == "torch" else quivers

    backend.log_likelihood(program, params, design).sum().backward()

    for name in program.indices:
        assert bound.host_data[name].dtype is torch.int64

    assert params.grad is not None
    assert params.grad.shape == params.shape
    assert bool(torch.isfinite(params.grad).all())
    assert bool((params.grad != 0.0).all())
