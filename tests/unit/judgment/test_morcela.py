from itertools import combinations
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
import torch
from quivers.effects import TraceHandler, clamp, run_program
from tests.conformance.linking import LinkingSubject, LinkingSuite

from lofi.judgment import (
    FAMILY,
    FREQUENCY,
    LENGTH,
    PARAMETERS,
    Apply,
    Binary,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    FixedDiscrimination,
    Function,
    JudgmentDesign,
    JudgmentRow,
    Linking,
    LinkingSpec,
    LinkTerm,
    Literal,
    MorcelaLinking,
    Name,
    Operator,
    Prior,
    PriorBlock,
    ResponseCompiler,
    ResponseSpec,
    RowColumns,
    SpecialCase,
    ThresholdPriors,
    log_category_probabilities,
    morcela_steps,
    ordered_cutpoints,
)
from lofi.judgment.morcela import morcela_rungs
from lofi.judgment.runtime import QuiversRuntime

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment import Expression

LM: Final = "lm_log_probability"
UNIGRAM: Final = "unigram_log_probability"
TOKENS: Final = "length"
LEVELS: Final = 7
ROWS: Final = 40


def term(**fixed: float) -> LinkTerm:
    return LinkTerm(
        readout=LM,
        linking=LinkingSpec(
            family=FAMILY, columns={LENGTH: TOKENS, FREQUENCY: UNIGRAM}, fixed=fixed
        ),
    )


def design(rows: int = ROWS, *, seed: int = 3) -> JudgmentDesign:
    generator = torch.Generator().manual_seed(seed)
    lengths = torch.randint(3, 25, (rows,), generator=generator).to(torch.float64)
    rates = 3.0 + torch.rand(rows, generator=generator, dtype=torch.float64)
    unigram_rates = 4.0 + torch.rand(rows, generator=generator, dtype=torch.float64)
    codec = DesignCodec(LEVELS, readouts=(LM, UNIGRAM, TOKENS))

    return codec.encode(
        JudgmentRow(
            f"annotator-{index % 5}",
            f"sentence-{index // 5}",
            index % LEVELS,
            RowColumns(
                factors={},
                readouts={
                    LM: float(-rates[index] * lengths[index]),
                    UNIGRAM: float(-unigram_rates[index] * lengths[index]),
                    TOKENS: float(lengths[index]),
                },
            ),
        )
        for index in range(rows)
    )


def slor(readouts: Mapping[str, Tensor]) -> Tensor:
    return (readouts[LM] - readouts[UNIGRAM]) / readouts[TOKENS]


def log_probability(readouts: Mapping[str, Tensor]) -> Tensor:
    return readouts[LM]


REFERENCES: Final = {"slor": slor, "log_probability": log_probability}
INSIDE: Final = torch.tensor([1.5, 0.4, 8.0, 0.75], dtype=torch.float64)


def length_only(readouts: Mapping[str, Tensor]) -> Tensor:
    return readouts[LM] / readouts[TOKENS]


def unigram_frequency_only(readouts: Mapping[str, Tensor]) -> Tensor:
    return readouts[LM] - readouts[UNIGRAM]


SURFACE_REFERENCES: Final = {
    "slor": slor,
    "log_probability": log_probability,
    "length_only": length_only,
    "unigram_frequency_only": unigram_frequency_only,
}
SURFACE_PINS: Final = {
    "slor": {
        "slope": 1.0,
        "frequency": 1.0,
        "length": 0.0,
        "normalization": 1.0,
    },
    "log_probability": {
        "slope": 1.0,
        "frequency": 0.0,
        "length": 0.0,
        "normalization": 0.0,
    },
    "length_only": {
        "slope": 1.0,
        "frequency": 0.0,
        "length": 0.0,
        "normalization": 1.0,
    },
    "unigram_frequency_only": {
        "slope": 1.0,
        "frequency": 1.0,
        "length": 0.0,
        "normalization": 0.0,
    },
}


def separating_design() -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(LM, UNIGRAM, TOKENS))
    values = ((-12.0, -5.0, 3.0), (-7.0, -11.0, 2.0), (-40.0, -55.0, 10.0))

    return codec.encode(
        JudgmentRow(
            "annotator",
            f"sentence-{index}",
            index,
            RowColumns(
                factors={},
                readouts={LM: lm, UNIGRAM: unigram, TOKENS: length},
            ),
        )
        for index, (lm, unigram, length) in enumerate(values)
    )


def surface_linking(name: str) -> MorcelaLinking:
    spec = morcela_rungs(length=TOKENS, frequency=UNIGRAM)[name]

    return MorcelaLinking(LinkTerm(readout=LM, linking=spec))


def test_the_four_surface_statistic_rungs_are_exact_morcela_parameterizations() -> None:
    rungs = morcela_rungs(length=TOKENS, frequency=UNIGRAM)

    assert tuple(rungs) == tuple(SURFACE_REFERENCES)
    assert all(isinstance(spec, LinkingSpec) for spec in rungs.values())
    assert all(spec.family == FAMILY for spec in rungs.values())
    assert all(
        spec.columns == {LENGTH: TOKENS, FREQUENCY: UNIGRAM} for spec in rungs.values()
    )
    assert {name: dict(spec.fixed) for name, spec in rungs.items()} == SURFACE_PINS


@pytest.mark.parametrize("name", tuple(SURFACE_REFERENCES))
def test_each_surface_statistic_rung_agrees_with_its_closed_form(name: str) -> None:
    source = separating_design()
    linking = surface_linking(name)
    actual = linking.contribution(source, torch.empty(0, dtype=torch.float64))
    expected = SURFACE_REFERENCES[name](source.readouts)

    assert linking.domain.parameter_names == ()
    torch.testing.assert_close(actual, expected, atol=1e-9, rtol=0.0)


def test_the_surface_statistic_fixture_separates_every_pair_of_rungs() -> None:
    source = separating_design()
    outputs = {
        name: surface_linking(name).contribution(
            source, torch.empty(0, dtype=torch.float64)
        )
        for name in SURFACE_REFERENCES
    }

    for (left_name, left), (right_name, right) in combinations(outputs.items(), 2):
        assert not torch.allclose(left, right, atol=1e-9, rtol=0.0), (
            left_name,
            right_name,
        )


@pytest.mark.parametrize(
    ("left", "right", "different"),
    [
        ("log_probability", "length_only", {"normalization"}),
        ("log_probability", "unigram_frequency_only", {"frequency"}),
        ("length_only", "slor", {"frequency"}),
        ("unigram_frequency_only", "slor", {"normalization"}),
        ("log_probability", "slor", {"frequency", "normalization"}),
        (
            "length_only",
            "unigram_frequency_only",
            {"frequency", "normalization"},
        ),
    ],
)
def test_surface_statistic_rungs_differ_in_exactly_the_declared_pins(
    left: str, right: str, different: set[str]
) -> None:
    rungs = morcela_rungs(length=TOKENS, frequency=UNIGRAM)

    assert set(rungs[left].fixed) == set(rungs[right].fixed) == set(PARAMETERS)
    assert {
        parameter
        for parameter in PARAMETERS
        if rungs[left].fixed[parameter] != rungs[right].fixed[parameter]
    } == different


class TestMorcelaLinking(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(MorcelaLinking(term()), design(), INSIDE, REFERENCES)


class TestMorcelaLinkingWithTheNormalizationPinned(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        return LinkingSubject(
            MorcelaLinking(term(normalization=1.0)),
            design(),
            INSIDE[:3],
            {"slor": slor},
        )


class TestMorcelaLinkingAsSlor(LinkingSuite):
    @pytest.fixture
    def linking_subject(self) -> LinkingSubject:
        pinned = dict(zip(PARAMETERS, (1.0, 1.0, 0.0, 1.0), strict=True))

        return LinkingSubject(
            MorcelaLinking(term(**pinned)),
            design(),
            torch.empty(0, dtype=torch.float64),
            {"slor": slor},
        )


def test_the_linking_answers_the_port() -> None:
    assert isinstance(MorcelaLinking(term()), Linking)


def test_the_domain_is_built_from_the_term_it_was_declared_by() -> None:
    declared = term()
    linking = MorcelaLinking(declared)

    assert linking.domain.argument == declared.readout
    assert linking.domain.readouts == declared.readouts
    assert linking.domain.parameter_names == PARAMETERS
    assert linking.pinned == {}


def test_the_declared_intervals_cover_a_sentence_and_slor_s_values() -> None:
    domain = MorcelaLinking(term()).domain

    assert domain.columns[TOKENS].covers(1.0)
    assert domain.columns[LM].high == 0.0
    assert domain.columns[UNIGRAM].high == 0.0
    assert domain.parameters["slope"].low == 0.0
    assert domain.parameters["frequency"].covers(1.0)
    assert domain.parameters["length"].covers(0.0)
    assert domain.parameters["normalization"].covers(0.0)
    assert domain.parameters["normalization"].covers(1.0)


def test_the_contribution_is_equation_one_scaled_and_normalized() -> None:
    source = design()
    linking = MorcelaLinking(term())
    slope, frequency, length, normalization = INSIDE.tolist()
    expected = (
        slope
        * (source.readouts[LM] - frequency * source.readouts[UNIGRAM] + length)
        / source.readouts[TOKENS] ** normalization
    )

    assert torch.allclose(linking.contribution(source, INSIDE), expected)


def test_slor_and_the_raw_log_probability_are_declared_at_their_vectors() -> None:
    cases = MorcelaLinking(term()).special_cases

    assert cases == (
        SpecialCase("slor", (1.0, 1.0, 0.0, 1.0)),
        SpecialCase("log_probability", (1.0, 0.0, 0.0, 0.0)),
    )


def test_a_pinned_parameter_leaves_the_domain_and_projects_the_cases() -> None:
    linking = MorcelaLinking(term(normalization=1.0, slope=1.0))

    assert linking.pinned == {"normalization": 1.0, "slope": 1.0}
    assert linking.domain.parameter_names == ("frequency", "length")
    assert linking.special_cases == (SpecialCase("slor", (1.0, 0.0)),)


def test_a_pin_disagreeing_with_every_case_declares_none() -> None:
    linking = MorcelaLinking(term(frequency=0.5))

    assert linking.domain.parameter_names == ("slope", "length", "normalization")
    assert linking.special_cases == ()


def test_a_pinned_parameter_is_held_at_its_value_in_the_contribution() -> None:
    source = design()
    free = MorcelaLinking(term())
    pinned = MorcelaLinking(term(frequency=0.4, normalization=0.75))

    assert torch.equal(
        pinned.contribution(source, torch.tensor([1.5, 8.0], dtype=torch.float64)),
        free.contribution(source, INSIDE),
    )


def test_pinning_every_parameter_fits_nothing() -> None:
    source = design()
    linking = MorcelaLinking(
        term(slope=1.0, frequency=0.0, length=0.0, normalization=0.0)
    )

    assert linking.domain.parameter_names == ()
    assert linking.special_cases == (SpecialCase("log_probability", ()),)
    assert torch.equal(
        linking.contribution(source, torch.empty(0, dtype=torch.float64)),
        source.readouts[LM],
    )


def test_the_frequency_correction_credits_a_rarer_sentence_of_equal_probability() -> (
    None
):
    rare = torch.tensor([-60.0], dtype=torch.float64)
    common = torch.tensor([-30.0], dtype=torch.float64)
    same = {
        LM: torch.tensor([-40.0], dtype=torch.float64),
        TOKENS: torch.tensor([10.0]),
    }
    linking = MorcelaLinking(term())
    parameters = torch.tensor([1.0, 0.5, 0.0, 1.0], dtype=torch.float64)

    class Columns:
        def __init__(self, unigram: Tensor) -> None:
            self.readouts = {**same, UNIGRAM: unigram}
            self.n_rows = 1

    assert float(linking.contribution(Columns(rare), parameters)) > float(
        linking.contribution(Columns(common), parameters)
    )


def test_another_family_is_refused() -> None:
    declared = LinkTerm(readout=LM, linking=LinkingSpec(family="logit"))

    with pytest.raises(ValueError, match="declared under the family 'logit'"):
        MorcelaLinking(declared)


@pytest.mark.parametrize(
    "columns",
    [{}, {LENGTH: TOKENS}, {LENGTH: TOKENS, FREQUENCY: UNIGRAM, "extra": "x"}],
)
def test_a_term_without_exactly_the_two_roles_is_refused(
    columns: dict[str, str],
) -> None:
    declared = LinkTerm(readout=LM, linking=LinkingSpec(family=FAMILY, columns=columns))

    with pytest.raises(ValueError, match="reads the roles"):
        MorcelaLinking(declared)


def test_a_pin_on_a_parameter_the_family_lacks_is_refused() -> None:
    with pytest.raises(ValueError, match=r"pinned: \['beta'\]"):
        MorcelaLinking(term(beta=1.0))


@pytest.mark.parametrize(
    "columns", [{LENGTH: LM, FREQUENCY: UNIGRAM}, {LENGTH: TOKENS, FREQUENCY: TOKENS}]
)
def test_one_column_under_two_roles_is_refused(columns: dict[str, str]) -> None:
    declared = LinkTerm(readout=LM, linking=LinkingSpec(family=FAMILY, columns=columns))

    with pytest.raises(ValueError, match="reads three columns"):
        MorcelaLinking(declared)


def test_the_identification_holds_the_slope_at_one_beside_the_term_s_pins() -> None:
    source = design()
    held = MorcelaLinking(term(length=-2.5), unit_slope=True)
    pinned = MorcelaLinking(term(slope=1.0, length=-2.5))
    parameters = torch.tensor([0.4, 0.75], dtype=torch.float64)

    assert held.pinned == pinned.pinned == {"slope": 1.0, "length": -2.5}
    assert held.domain.parameter_names == ("frequency", "normalization")
    assert torch.equal(
        held.contribution(source, parameters),
        pinned.contribution(source, parameters),
    )
    assert tuple(morcela_steps(term(), "lm", unit_slope=True).draws) == (
        "lm_frequency",
        "lm_length",
        "lm_normalization",
    )


def test_a_slope_pinned_away_from_one_is_refused_under_the_identification() -> None:
    with pytest.raises(
        ValueError, match=r"holds the slope at one, and the term pins it at 2\.0"
    ):
        MorcelaLinking(term(slope=2.0), unit_slope=True)

    with pytest.raises(ValueError, match="holds the slope at one"):
        morcela_steps(term(slope=0.5), "lm", unit_slope=True)


def equation_one(frequency: Expression, length: Expression) -> Expression:
    corrected = Binary(
        Operator.ADD,
        Binary(Operator.SUB, Name(LM), Binary(Operator.MUL, frequency, Name(UNIGRAM))),
        length,
    )
    normalizer = Apply(
        Function.EXP,
        Binary(
            Operator.MUL, Name("lm_normalization"), Apply(Function.LOG, Name(TOKENS))
        ),
    )

    return Binary(
        Operator.DIV, Binary(Operator.MUL, Name("lm_slope"), corrected), normalizer
    )


def test_the_program_form_draws_each_fitted_parameter_and_writes_the_pins() -> None:
    free = morcela_steps(term(), "lm")
    pinned = morcela_steps(term(frequency=1.0, length=-2.5), "lm")

    assert tuple(free.draws) == tuple(f"lm_{name}" for name in PARAMETERS)
    assert free.draws["lm_slope"].family == "HalfNormal"
    assert free.expression == equation_one(Name("lm_frequency"), Name("lm_length"))
    assert tuple(pinned.draws) == ("lm_slope", "lm_normalization")
    assert pinned.expression == equation_one(Literal(1.0), Literal(-2.5))


def test_the_program_form_evaluates_to_the_contribution() -> None:
    source = design()
    declared = term(length=-2.5)
    linking = MorcelaLinking(declared)
    steps = linking.steps(LM)
    program = (
        ResponseCompiler({FAMILY: morcela_steps})
        .compile(
            ResponseSpec(
                family=Cumulative(),
                levels=LEVELS,
                objective=ConditionalLikelihood(),
                predictor=(declared,),
                identification=FixedDiscrimination(value=1.0),
                priors=PriorBlock(thresholds=ThresholdPriors()),
            )
        )
        .over(source)
    )
    parameters = torch.tensor([1.5, 0.4, 0.75], dtype=torch.float64)
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

    assert set(steps.draws) <= {draw.site for draw in program.program.draws}
    assert torch.allclose(
        handler.trace.sites[f"{LM}_term"].value,
        linking.contribution(source, parameters),
        atol=1e-12,
        rtol=0.0,
    )


def test_the_program_form_names_the_prior_of_each_parameter() -> None:
    steps = morcela_steps(term(), "lm")

    assert steps.draws["lm_slope"] == Prior("HalfNormal", 1.0)
    assert steps.draws["lm_frequency"] == Prior("Normal", 1.0, 1.0)
    assert steps.draws["lm_length"] == Prior("Normal", 0.0, 10.0)
    assert steps.draws["lm_normalization"] == Prior("Uniform", 0.0, 1.0)


def test_the_repr_names_the_columns_and_the_pins() -> None:
    assert repr(MorcelaLinking(term(slope=2.0))) == (
        f"MorcelaLinking(argument={LM!r}, length={TOKENS!r}, "
        f"frequency={UNIGRAM!r}, pinned={{'slope': 2.0}})"
    )


class Fit(NamedTuple):
    frequency: float
    length: float


GENERATING: Final = Fit(0.4, 8.0)


def simulated(rows: int, *, seed: int) -> tuple[JudgmentDesign, Tensor]:
    source = design(rows, seed=seed)
    generator = torch.Generator().manual_seed(seed + 1)
    latent = MorcelaLinking(term(normalization=1.0)).contribution(
        source,
        torch.tensor(
            [1.5, GENERATING.frequency, GENERATING.length], dtype=torch.float64
        ),
    )
    cutpoints = torch.tensor([-3.0, -1.5, -0.5, 0.5, 1.5, 3.0], dtype=torch.float64)
    cutpoints = cutpoints + latent.mean()
    probabilities = torch.exp(log_category_probabilities(latent, cutpoints))
    responses = torch.multinomial(probabilities, 1, generator=generator).reshape(-1)

    return source, responses


def fitted(source: JudgmentDesign, responses: Tensor) -> Fit:
    linking = MorcelaLinking(term(normalization=1.0))
    parameters = torch.tensor([1.0, 1.0, 0.0], dtype=torch.float64, requires_grad=True)
    base = torch.zeros((), dtype=torch.float64, requires_grad=True)
    spans = torch.zeros(LEVELS - 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.LBFGS(
        [parameters, base, spans], lr=0.5, max_iter=300, line_search_fn="strong_wolfe"
    )

    def closure() -> Tensor:
        optimizer.zero_grad()
        latent = linking.contribution(source, parameters)
        log_probabilities = log_category_probabilities(
            latent, ordered_cutpoints(base, spans)
        )
        loss = -log_probabilities.gather(1, responses.unsqueeze(1)).sum()
        loss.backward()

        return loss

    for _ in range(4):
        optimizer.step(closure)

    _, frequency, length = parameters.detach().tolist()

    return Fit(frequency, length)


def test_the_fitted_corrections_differ_from_slor_s_fixed_values() -> None:
    source, responses = simulated(1500, seed=11)

    fit = fitted(source, responses)

    assert abs(fit.frequency - GENERATING.frequency) < abs(fit.frequency - 1.0)
    assert abs(fit.length - GENERATING.length) < abs(fit.length - 0.0)
    assert abs(fit.frequency - GENERATING.frequency) < 0.2
    assert abs(fit.length - GENERATING.length) < 3.0
