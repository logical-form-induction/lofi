import math
from itertools import pairwise
from typing import TYPE_CHECKING, Literal, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.latent_predictor import (
    LatentPredictorSubject,
    LatentPredictorSuite,
)
from tests.conformance.response_model import (
    ResponseModelObservation,
    ResponseModelSubject,
    ResponseModelSuite,
)

from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    CumulativeLinkHead,
    CumulativeParameters,
    DesignCodec,
    FixedCutpointVariance,
    FixedDiscrimination,
    JudgmentRow,
    Metric,
    OrdinalHead,
    Predictor,
    PriorBlock,
    PriorSpec,
    ResponseSpec,
    RowColumns,
    RowEffects,
    ScaleNormalization,
    ThresholdPriors,
    log_category_probabilities,
    ordered_cutpoints,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.judgment import JudgmentDesign

READOUT = "p_gram"
LEVELS = 7
BASE = -3.0
SPANS = (0.5, 0.2, -0.3, 1.0, 0.7, 0.1)
RAW_DISCRIMINATION = 0.4
STABILITY_GAP = 30.0
TAIL_GAP = 40.0
SPAN_OFFSETS = (-8.0, -8.0, 0.0, 0.0, 8.0, 0.0)


class Readout(Predictor, extra="forbid"):
    kind: Literal["ordinal_head_readout"] = "ordinal_head_readout"
    name: str


class ReadoutPredictor:
    def __init__(self, name: str) -> None:
        self._terms = (Readout(name=name),)
        self._name = name

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        return design.readouts[self._name]


class ShortPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        return super().latent_scores(design)[:-1]


class SquarePredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        scores = super().latent_scores(design)

        return torch.stack((scores, scores), dim=1)


class CountingPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> Tensor:
        return super().latent_scores(design).to(torch.int64)


def spec(
    *,
    levels: int = LEVELS,
    discrimination: bool = False,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
) -> ResponseSpec:
    priors = PriorBlock(
        thresholds=ThresholdPriors(),
        discrimination=(
            PriorSpec(distribution="LogNormal", parameters=(0.0, 1.0))
            if discrimination
            else None
        ),
    )

    return ResponseSpec(
        family=Cumulative(discrimination=discrimination),
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=(Readout(name=READOUT),),
        identification=(
            FixedCutpointVariance(value=1.0)
            if discrimination
            else FixedDiscrimination(value=1.0)
        ),
        priors=priors,
        normalization=normalization,
    )


def head(declared: ResponseSpec) -> CumulativeLinkHead:
    return CumulativeLinkHead(declared, ReadoutPredictor(READOUT))


def parameters(
    declared: ResponseSpec,
    *,
    dtype: torch.dtype = torch.float64,
) -> CumulativeParameters:
    cutpoints = declared.levels - 1
    spans = tuple(SPANS[position % len(SPANS)] for position in range(cutpoints))

    return CumulativeParameters(
        torch.tensor(BASE, dtype=dtype),
        torch.tensor(spans, dtype=dtype),
        torch.tensor(RAW_DISCRIMINATION, dtype=dtype)
        if declared.family.draws_discrimination
        else None,
    )


def observation(
    declared: ResponseSpec,
    latent: Tensor,
) -> ResponseModelObservation:
    subject, given = head(declared), parameters(declared)

    return ResponseModelObservation(
        subject.cutpoints(given),
        subject.cumulative_probabilities(latent, given),
        subject.probabilities(latent, given),
    )


def observation_under_span_offsets(
    declared: ResponseSpec,
    latent: Tensor,
) -> ResponseModelObservation:
    subject, given = head(declared), parameters(declared)
    offsets = torch.tensor(
        tuple(
            SPAN_OFFSETS[position % len(SPAN_OFFSETS)]
            for position in range(declared.levels - 1)
        ),
        dtype=latent.dtype,
    )
    effects = RowEffects(torch.zeros_like(latent), offsets.expand(latent.numel(), -1))

    return ResponseModelObservation(
        subject.cutpoints(given, effects=effects)[0],
        subject.cumulative_probabilities(latent, given, effects=effects),
        subject.probabilities(latent, given, effects=effects),
    )


def upper_tail_log_probabilities(latent: float, cutpoints: Tensor) -> Tensor:
    offsets = [float(cutpoint) - latent for cutpoint in cutpoints]

    return torch.tensor(
        (
            -math.log1p(math.exp(-offsets[0])),
            *(
                math.log(math.exp(-lower) - math.exp(-upper))
                - math.log1p(math.exp(-lower))
                - math.log1p(math.exp(-upper))
                for lower, upper in pairwise(offsets)
            ),
            -math.log1p(math.exp(offsets[-1])),
        ),
        dtype=torch.float64,
    )


def naive_probabilities(latent: Tensor, cutpoints: Tensor) -> Tensor:
    below = torch.sigmoid(cutpoints.unsqueeze(0) - latent.unsqueeze(1))

    return torch.cat(
        (below[:, :1], below[:, 1:] - below[:, :-1], 1.0 - below[:, -1:]),
        dim=1,
    )


def design_of(responses: tuple[int, ...], scores: tuple[float, ...]) -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(READOUT,))
    rows = tuple(
        JudgmentRow(
            f"annotator-{position % 3}",
            f"item-{position % 2}",
            response,
            RowColumns(factors={}, readouts={READOUT: score}),
        )
        for position, (response, score) in enumerate(
            zip(responses, scores, strict=True)
        )
    )

    return codec.encode(rows)


def adversarial_parameters() -> tuple[CumulativeParameters, ...]:
    generator = torch.Generator().manual_seed(4703)
    drawn = []

    for index in range(128):
        scale = 10.0 ** (index % 7 - 3)
        drawn.append(
            CumulativeParameters(
                torch.randn(1, generator=generator, dtype=torch.float64)[0] * scale,
                torch.randn(
                    LEVELS - 1,
                    generator=generator,
                    dtype=torch.float64,
                )
                * scale,
            )
        )

    drawn.append(
        CumulativeParameters(
            torch.tensor(0.0, dtype=torch.float64),
            torch.tensor((-1e4, -8e2, -1e2, 0.0, 5e1, 1e4), dtype=torch.float64),
        )
    )

    return tuple(drawn)


class TestTheCumulativeLinkHeadObeysTheResponseModelLaws(ResponseModelSuite):
    @pytest.fixture(
        params=[
            pytest.param(LEVELS, id="seven-levels"),
            pytest.param(2, id="two-levels"),
            pytest.param(3, id="three-levels"),
        ]
    )
    def response_model_subject(
        self,
        request: pytest.FixtureRequest,
    ) -> ResponseModelSubject:
        return ResponseModelSubject(spec(levels=request.param), observation)


class TestTheCumulativeLinkHeadObeysTheResponseModelLawsUnderSpanOffsets(
    ResponseModelSuite
):
    @pytest.fixture(
        params=[
            pytest.param(LEVELS, id="seven-levels"),
            pytest.param(2, id="two-levels"),
            pytest.param(3, id="three-levels"),
        ]
    )
    def response_model_subject(
        self,
        request: pytest.FixtureRequest,
    ) -> ResponseModelSubject:
        return ResponseModelSubject(
            spec(levels=request.param), observation_under_span_offsets
        )


class TestTheReadoutEvaluatorObeysTheLatentPredictorLaws(LatentPredictorSuite):
    @pytest.fixture
    def latent_predictor_subject(self) -> LatentPredictorSubject:
        return LatentPredictorSubject(
            ReadoutPredictor(READOUT),
            design_of((1, 5, 0, 6), (0.25, -1.5, 3.0, 0.0)),
        )


def test_the_head_reports_one_cutpoint_between_adjacent_levels() -> None:
    subject = head(spec())

    assert subject.levels == LEVELS
    assert subject.cutpoint_count == LEVELS - 1
    assert subject.cutpoints(parameters(spec())).numel() == LEVELS - 1


def test_the_head_exposes_the_evaluator_of_the_declared_terms() -> None:
    declared = spec()
    subject = head(declared)

    assert subject.predictor.terms == declared.predictor


def test_the_cutpoints_never_fall_out_of_order_under_adversarial_parameters() -> None:
    drawn = adversarial_parameters()
    subject = head(spec())

    for given in drawn:
        gaps = torch.diff(subject.cutpoints(given))

        assert bool(torch.all(torch.isfinite(gaps)))
        assert float(torch.min(gaps)) >= 0.0


def test_the_cutpoints_rise_strictly_when_every_span_is_negative() -> None:
    given = CumulativeParameters(
        torch.tensor(5.0, dtype=torch.float64),
        torch.full((LEVELS - 1,), -20.0, dtype=torch.float64),
    )
    cutpoints = head(spec()).cutpoints(given)

    assert float(torch.min(torch.diff(cutpoints))) > 0.0
    assert float(cutpoints[0]) > 5.0


def test_the_cutpoints_rise_from_the_base_by_softplus_transformed_spans() -> None:
    base = torch.tensor(BASE, dtype=torch.float64)
    spans = torch.tensor(SPANS, dtype=torch.float64)
    expected = base + torch.cumsum(torch.nn.functional.softplus(spans), dim=-1)

    assert torch.equal(ordered_cutpoints(base, spans), expected)


def test_a_larger_latent_score_lowers_every_cumulative_probability() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-1.0, 2.0], dtype=torch.float64)
    cumulative = subject.cumulative_probabilities(latent, given)

    assert bool(torch.all(cumulative[1] < cumulative[0]))


def test_the_cumulative_function_is_the_sigmoid_of_the_offset() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-1.0, 0.25, 4.0], dtype=torch.float64)
    offsets = subject.cutpoints(given).unsqueeze(0) - latent.unsqueeze(1)

    assert torch.equal(
        subject.cumulative_probabilities(latent, given),
        torch.sigmoid(offsets),
    )


def test_the_probabilities_match_direct_differencing_at_moderate_offsets() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.linspace(-4.0, 4.0, 33, dtype=torch.float64)
    direct = naive_probabilities(latent, subject.cutpoints(given))

    assert torch.allclose(
        subject.probabilities(latent, given),
        direct,
        atol=1e-12,
        rtol=0.0,
    )


def test_the_binary_head_splits_the_logistic_at_its_one_cutpoint() -> None:
    declared = spec(levels=2)
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-2.0, 0.0, 3.0], dtype=torch.float64)
    offsets = subject.cutpoints(given) - latent
    expected = torch.stack((torch.sigmoid(offsets), torch.sigmoid(-offsets)), dim=1)

    assert torch.allclose(
        subject.probabilities(latent, given),
        expected,
        atol=1e-15,
        rtol=0.0,
    )


def test_the_distribution_stays_proper_at_a_latent_score_of_thirty() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-STABILITY_GAP, STABILITY_GAP], dtype=torch.float64)
    probabilities = subject.probabilities(latent, given)

    assert bool(torch.all(torch.isfinite(probabilities)))
    assert float(torch.min(probabilities)) > 0.0
    assert torch.allclose(
        torch.sum(probabilities, dim=1),
        torch.ones(2, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_direct_differencing_loses_the_tail_the_head_keeps() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-STABILITY_GAP], dtype=torch.float64)
    cutpoints = subject.cutpoints(given)
    direct = naive_probabilities(latent, cutpoints)
    departure = torch.abs(
        torch.log(direct) - log_category_probabilities(latent, cutpoints)
    )

    assert float(torch.max(departure)) > 1e-6


def test_direct_differencing_zeroes_the_tail_the_head_keeps_in_single_precision() -> (
    None
):
    declared = spec()
    subject = head(declared)
    given = parameters(declared, dtype=torch.float32)
    latent = torch.tensor([-STABILITY_GAP], dtype=torch.float32)
    cutpoints = subject.cutpoints(given)

    assert float(torch.min(naive_probabilities(latent, cutpoints))) == 0.0
    assert float(torch.min(subject.probabilities(latent, given))) > 0.0
    assert float(torch.sum(subject.probabilities(latent, given))) == pytest.approx(1.0)


def test_the_head_passes_a_gradient_to_the_cutpoints_at_the_stability_gap() -> None:
    base = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
    spans = torch.tensor(SPANS, dtype=torch.float64, requires_grad=True)
    latent = torch.tensor([-STABILITY_GAP, 0.0, STABILITY_GAP], dtype=torch.float64)
    given = CumulativeParameters(base, spans)
    head(spec()).log_probabilities(latent, given).sum().backward()

    assert base.grad is not None
    assert spans.grad is not None
    assert bool(torch.all(torch.isfinite(spans.grad)))
    assert bool(torch.all(spans.grad != 0.0))
    assert bool(torch.isfinite(base.grad))


def test_discrimination_scales_both_the_cutpoints_and_the_latent_score() -> None:
    declared = spec(discrimination=True)
    subject, given = head(declared), parameters(declared)
    scale = torch.nn.functional.softplus(
        torch.tensor(RAW_DISCRIMINATION, dtype=torch.float64)
    )
    latent = torch.tensor([-1.5, 0.5], dtype=torch.float64)
    unscaled = ordered_cutpoints(given.base, given.spans)

    assert torch.equal(subject.discrimination(given), scale)
    assert torch.allclose(subject.cutpoints(given), scale * unscaled, rtol=0.0)
    assert torch.allclose(
        subject.probabilities(latent, given),
        naive_probabilities(scale * latent, scale * unscaled),
        atol=1e-12,
        rtol=0.0,
    )
    assert torch.allclose(
        subject.cumulative_probabilities(latent, given),
        torch.sigmoid((scale * unscaled).unsqueeze(0) - (scale * latent).unsqueeze(1)),
        atol=1e-12,
        rtol=0.0,
    )


def test_a_larger_discrimination_moves_mass_onto_the_level_the_score_falls_in() -> None:
    subject = head(spec(discrimination=True))
    latent = torch.tensor([-5.0], dtype=torch.float64)
    lowest = [
        float(
            subject.probabilities(
                latent,
                CumulativeParameters(
                    torch.tensor(BASE, dtype=torch.float64),
                    torch.tensor(SPANS, dtype=torch.float64),
                    torch.tensor(raw, dtype=torch.float64),
                ),
            )[0, 0]
        )
        for raw in (-2.0, 0.0, 2.0, 4.0)
    ]

    assert lowest == sorted(lowest)
    assert lowest[0] < lowest[-1]


def test_a_family_that_fixes_discrimination_applies_one() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)

    assert float(subject.discrimination(given)) == 1.0
    assert torch.equal(
        subject.cutpoints(given),
        ordered_cutpoints(given.base, given.spans),
    )


def test_discrimination_stays_positive_for_any_unconstrained_value() -> None:
    declared = spec(discrimination=True)
    subject = head(declared)

    for raw in (-1e3, -20.0, 0.0, 20.0):
        given = CumulativeParameters(
            torch.tensor(BASE, dtype=torch.float64),
            torch.tensor(SPANS, dtype=torch.float64),
            torch.tensor(raw, dtype=torch.float64),
        )

        assert float(subject.discrimination(given)) >= 0.0
        assert float(torch.min(torch.diff(subject.cutpoints(given)))) >= 0.0


def test_the_log_likelihood_reads_the_response_recorded_in_each_row() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    responses = (0, 6, 3, 1)
    scores = (0.5, -2.0, 1.25, STABILITY_GAP)
    design = design_of(responses, scores)
    every = subject.log_probabilities(subject.latent_scores(design), given)
    expected = torch.stack(
        [every[position, response] for position, response in enumerate(responses)]
    )

    assert torch.equal(subject.log_likelihood(design, given), expected)


def test_the_log_likelihood_keeps_one_row_per_repeated_measure() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    design = design_of((2, 4, 2), (1.0, 1.0, 1.0))

    assert subject.log_likelihood(design, given).shape == (3,)


def test_the_head_scores_a_design_through_the_declared_readout() -> None:
    declared = spec()
    subject = head(declared)
    design = design_of((1, 5), (0.25, -1.5))

    assert torch.equal(subject.latent_scores(design), design.readouts[READOUT])


def test_a_head_refuses_a_specification_of_another_family() -> None:
    declared = ResponseSpec(
        family=Metric(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(Readout(name=READOUT),),
        identification=FixedDiscrimination(value=1.0),
    )

    with pytest.raises(TypeError, match="defined for the cumulative family"):
        CumulativeLinkHead(declared, ReadoutPredictor(READOUT))


def test_a_head_refuses_an_evaluator_of_undeclared_terms() -> None:
    with pytest.raises(ValueError, match="against the declared terms"):
        CumulativeLinkHead(spec(), ReadoutPredictor("length"))


@pytest.mark.parametrize("levels", [LEVELS - 1, LEVELS + 1])
def test_a_head_refuses_parameters_giving_another_number_of_cutpoints(
    levels: int,
) -> None:
    given = parameters(spec(levels=levels))

    with pytest.raises(ValueError, match="needs 6 cutpoints"):
        head(spec()).cutpoints(given)


def test_a_head_refuses_discrimination_its_family_does_not_draw() -> None:
    given = parameters(spec(discrimination=True))

    with pytest.raises(ValueError, match="does not draw discrimination"):
        head(spec()).cutpoints(given)


def test_a_discriminating_head_refuses_parameters_without_discrimination() -> None:
    given = parameters(spec())

    with pytest.raises(ValueError, match="draws discrimination"):
        head(spec(discrimination=True)).cutpoints(given)


def test_a_head_refuses_a_design_over_another_number_of_levels() -> None:
    declared = spec(levels=LEVELS - 1)
    subject = CumulativeLinkHead(declared, ReadoutPredictor(READOUT))

    with pytest.raises(ValueError, match="was given a design over 7"):
        subject.log_likelihood(design_of((1,), (0.5,)), parameters(declared))


def test_a_head_refuses_an_evaluator_that_scores_another_design() -> None:
    subject = CumulativeLinkHead(spec(), ShortPredictor(READOUT))

    with pytest.raises(ValueError, match="2 rows was scored with 1"):
        subject.latent_scores(design_of((1, 3), (0.5, 0.75)))


def test_a_head_refuses_a_latent_score_that_is_not_one_per_row() -> None:
    declared = spec()

    with pytest.raises(ValueError, match="latent score has rank 2"):
        head(declared).probabilities(
            torch.zeros((2, 2), dtype=torch.float64),
            parameters(declared),
        )


def test_parameters_refuse_a_base_that_is_not_a_scalar() -> None:
    with pytest.raises(ValueError, match="base has rank 1"):
        CumulativeParameters(
            torch.zeros(1, dtype=torch.float64),
            torch.tensor(SPANS, dtype=torch.float64),
        )


def test_parameters_refuse_an_empty_span_vector() -> None:
    with pytest.raises(ValueError, match="one span per cutpoint"):
        CumulativeParameters(
            torch.tensor(BASE, dtype=torch.float64),
            torch.zeros(0, dtype=torch.float64),
        )


def test_parameters_refuse_a_span_vector_of_another_rank() -> None:
    with pytest.raises(ValueError, match="spans has rank 2"):
        CumulativeParameters(
            torch.tensor(BASE, dtype=torch.float64),
            torch.zeros((2, 2), dtype=torch.float64),
        )


def test_parameters_refuse_a_dtype_disagreement() -> None:
    with pytest.raises(ValueError, match="spans hold"):
        CumulativeParameters(
            torch.tensor(BASE, dtype=torch.float64),
            torch.tensor(SPANS, dtype=torch.float32),
        )


def test_parameters_refuse_a_discrimination_dtype_disagreement() -> None:
    with pytest.raises(ValueError, match="discrimination holds"):
        CumulativeParameters(
            torch.tensor(BASE, dtype=torch.float64),
            torch.tensor(SPANS, dtype=torch.float64),
            torch.tensor(RAW_DISCRIMINATION, dtype=torch.float32),
        )


def test_parameters_refuse_values_that_are_not_floating_point() -> None:
    with pytest.raises(TypeError, match="floating-point"):
        CumulativeParameters(
            torch.tensor(0, dtype=torch.int64),
            torch.zeros(LEVELS - 1, dtype=torch.int64),
        )


def test_parameters_refuse_a_discrimination_that_is_not_a_scalar() -> None:
    with pytest.raises(ValueError, match="discrimination has rank 1"):
        CumulativeParameters(
            torch.tensor(BASE, dtype=torch.float64),
            torch.tensor(SPANS, dtype=torch.float64),
            torch.zeros(1, dtype=torch.float64),
        )


def test_the_upper_tail_keeps_every_level_positive_and_accurate() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-TAIL_GAP], dtype=torch.float64)
    log_probabilities = subject.log_probabilities(latent, given)
    expected = upper_tail_log_probabilities(-TAIL_GAP, subject.cutpoints(given))

    assert bool(torch.all(torch.isfinite(log_probabilities)))
    assert bool(torch.all(subject.probabilities(latent, given) > 0.0))
    assert torch.allclose(log_probabilities[0], expected, atol=0.0, rtol=1e-12)


def test_the_upper_tail_passes_a_finite_gradient_to_every_parameter() -> None:
    base = torch.tensor(BASE, dtype=torch.float64, requires_grad=True)
    spans = torch.tensor(SPANS, dtype=torch.float64, requires_grad=True)
    latent = torch.tensor([-TAIL_GAP], dtype=torch.float64)
    head(spec()).log_probabilities(
        latent, CumulativeParameters(base, spans)
    ).sum().backward()

    assert base.grad is not None
    assert spans.grad is not None
    assert bool(torch.isfinite(base.grad))
    assert bool(torch.all(torch.isfinite(spans.grad)))
    assert bool(torch.all(spans.grad != 0.0))


def test_the_dominant_level_keeps_its_log_probability_at_widely_split_cutpoints() -> (
    None
):
    cutpoints = torch.tensor((-TAIL_GAP, TAIL_GAP), dtype=torch.float64)
    latent = torch.zeros(1, dtype=torch.float64)
    dominant = float(log_category_probabilities(latent, cutpoints)[0, 1])
    expected = math.log1p(-2.0 / (1.0 + math.exp(TAIL_GAP)))

    assert dominant < 0.0
    assert dominant == pytest.approx(expected, rel=1e-9, abs=0.0)


def test_the_probabilities_stay_normalized_when_two_cutpoints_nearly_coincide() -> None:
    declared = spec(levels=3)
    subject = head(declared)
    given = CumulativeParameters(
        torch.tensor(0.0, dtype=torch.float64),
        torch.tensor((-38.0, -38.0), dtype=torch.float64),
    )
    latent = torch.tensor([0.0], dtype=torch.float64)
    probabilities = subject.probabilities(latent, given)

    assert bool(torch.all(torch.isfinite(probabilities)))
    assert float(torch.sum(probabilities)) == pytest.approx(1.0, abs=1e-12)


@pytest.mark.parametrize(
    "normalization",
    [ScaleNormalization.LOG, ScaleNormalization.UNIT_INTERVAL],
)
def test_a_head_s_specification_refuses_a_normalization_of_the_level(
    normalization: ScaleNormalization,
) -> None:
    with pytest.raises(dx.ValidationError, match="normalization of the response"):
        spec(normalization=normalization)


def test_a_head_accepts_the_identity_normalization() -> None:
    declared = spec(normalization=ScaleNormalization.IDENTITY)
    subject = head(declared)

    assert subject.levels == LEVELS


def test_the_discrimination_refuses_parameters_of_another_shape() -> None:
    given = parameters(spec(levels=LEVELS + 1))

    with pytest.raises(ValueError, match="needs 6 cutpoints"):
        head(spec()).discrimination(given)


def test_the_discrimination_refuses_parameters_its_family_does_not_draw() -> None:
    given = parameters(spec(discrimination=True))

    with pytest.raises(ValueError, match="does not draw discrimination"):
        head(spec()).discrimination(given)


def test_a_head_refuses_an_evaluator_whose_scores_are_not_one_per_row() -> None:
    subject = CumulativeLinkHead(spec(), SquarePredictor(READOUT))

    with pytest.raises(ValueError, match="latent score has rank 2"):
        subject.latent_scores(design_of((1, 3), (0.5, 0.75)))


def test_a_head_refuses_an_evaluator_whose_scores_are_not_floating_point() -> None:
    subject = CumulativeLinkHead(spec(), CountingPredictor(READOUT))

    with pytest.raises(TypeError, match="not a floating-point kind"):
        subject.latent_scores(design_of((1, 3), (0.5, 0.75)))


def test_parameters_refuse_field_reassignment_and_deletion() -> None:
    given = parameters(spec())

    with pytest.raises(AttributeError, match="parameters are immutable"):
        given.__setattr__("_base", torch.tensor(99.0, dtype=torch.float64))

    with pytest.raises(AttributeError, match="parameters are immutable"):
        given.__delattr__("_spans")


def effects_of(
    latent: tuple[float, ...],
    spans: tuple[tuple[float, ...], ...] | None,
    *,
    dtype: torch.dtype = torch.float64,
) -> RowEffects:
    return RowEffects(
        torch.tensor(latent, dtype=dtype),
        None if spans is None else torch.tensor(spans, dtype=dtype),
    )


def test_row_effects_report_their_offsets_and_row_count() -> None:
    effects = effects_of((0.5, -0.5), ((0.1,) * 6, (0.2,) * 6))

    assert effects.rows == 2
    assert torch.equal(effects.latent, torch.tensor([0.5, -0.5], dtype=torch.float64))
    assert effects.spans is not None
    assert effects.spans.shape == (2, 6)
    assert effects_of((1.0,), None).spans is None


def test_row_effects_add_offset_by_offset_and_keep_whichever_spans_exist() -> None:
    latent_only = effects_of((1.0, 2.0), None)
    with_spans = effects_of((0.5, 0.5), ((1.0,) * 6, (2.0,) * 6))
    doubled = with_spans + with_spans
    mixed = latent_only + with_spans

    assert torch.equal(doubled.latent, torch.tensor([1.0, 1.0], dtype=torch.float64))
    assert doubled.spans is not None
    assert with_spans.spans is not None
    assert torch.equal(doubled.spans, 2.0 * with_spans.spans)
    assert torch.equal(mixed.latent, torch.tensor([1.5, 2.5], dtype=torch.float64))
    assert mixed.spans is not None
    assert torch.equal(mixed.spans, with_spans.spans)
    assert (latent_only + latent_only).spans is None
    reversed_spans = (with_spans + latent_only).spans
    assert reversed_spans is not None
    assert torch.equal(reversed_spans, with_spans.spans)


@pytest.mark.parametrize(
    ("other", "message"),
    [
        (effects_of((0.0,), None), "over 2 rows were added to 1"),
        (effects_of((0.0, 0.0, 0.0), ((0.0,) * 6,) * 3), "over 2 rows were added to 3"),
        (effects_of((0.0, 0.0), None, dtype=torch.float32), "float64 were added to"),
        (effects_of((0.0, 0.0), ((0.0,) * 5,) * 2), "over 6 spans were added to 5"),
    ],
)
def test_row_effects_refuse_to_add_offsets_over_other_rows_dtypes_or_spans(
    other: RowEffects, message: str
) -> None:
    effects = effects_of((0.0, 0.0), ((0.0,) * 6,) * 2)

    with pytest.raises(ValueError, match=message):
        _ = effects + other

    with pytest.raises(ValueError, match="were added to"):
        _ = other + effects


def test_row_effects_refuse_offsets_that_do_not_line_up() -> None:
    with pytest.raises(ValueError, match="latent offset has rank 2"):
        RowEffects(torch.zeros(2, 1, dtype=torch.float64))

    with pytest.raises(ValueError, match="span offsets has rank 1"):
        RowEffects(
            torch.zeros(2, dtype=torch.float64), torch.zeros(6, dtype=torch.float64)
        )

    with pytest.raises(ValueError, match="2 latent scores and 3 rows of spans"):
        effects_of((0.0, 0.0), ((0.0,) * 6,) * 3)

    with pytest.raises(ValueError, match=r"holds torch\.float64 and its span offsets"):
        RowEffects(torch.zeros(2, dtype=torch.float64), torch.zeros(2, 6))

    with pytest.raises(TypeError, match="not a floating-point kind"):
        RowEffects(torch.zeros(2, dtype=torch.int64))


def test_row_effects_refuse_field_reassignment_and_deletion() -> None:
    effects = effects_of((0.0,), None)

    with pytest.raises(AttributeError, match="row effects are immutable"):
        effects.__setattr__("_latent", torch.zeros(1, dtype=torch.float64))

    with pytest.raises(AttributeError, match="row effects are immutable"):
        effects.__delattr__("_latent")


def test_a_latent_offset_shifts_the_score_before_the_cutpoints_are_read() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([-1.0, 0.5, 2.0], dtype=torch.float64)
    offsets = torch.tensor([0.75, -2.0, 0.0], dtype=torch.float64)
    effects = RowEffects(offsets)

    assert torch.equal(
        subject.probabilities(latent, given, effects=effects),
        subject.probabilities(latent + offsets, given),
    )
    assert torch.equal(
        subject.cumulative_probabilities(latent, given, effects=effects),
        subject.cumulative_probabilities(latent + offsets, given),
    )
    assert torch.equal(
        subject.cutpoints(given, effects=effects), subject.cutpoints(given)
    )


def test_a_span_offset_gives_each_row_its_own_ordered_cutpoints() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    span_offsets = torch.tensor(
        ((0.5, -3.0, 1.0, 0.0, -0.25, 2.0), (-40.0, -40.0, 0.0, 0.0, 40.0, 0.0)),
        dtype=torch.float64,
    )
    effects = RowEffects(torch.zeros(2, dtype=torch.float64), span_offsets)
    cutpoints = subject.cutpoints(given, effects=effects)
    latent = torch.tensor([0.0, 1.0], dtype=torch.float64)

    assert cutpoints.shape == (2, LEVELS - 1)
    assert float(torch.min(torch.diff(cutpoints, dim=1))) >= 0.0

    for row in range(2):
        assert torch.equal(
            cutpoints[row],
            ordered_cutpoints(given.base, given.spans + span_offsets[row]),
        )
        assert torch.allclose(
            subject.probabilities(latent, given, effects=effects)[row],
            subject.probabilities(
                latent[row : row + 1],
                CumulativeParameters(given.base, given.spans + span_offsets[row]),
            )[0],
            rtol=0.0,
            atol=1e-15,
        )


def test_the_distribution_stays_proper_under_extreme_offsets_that_vary_by_row() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    generator = torch.Generator().manual_seed(613)
    rows = 256
    latent = torch.rand(rows, generator=generator, dtype=torch.float64) * 24.0 - 12.0
    effects = RowEffects(
        torch.randn(rows, generator=generator, dtype=torch.float64) * TAIL_GAP,
        torch.randn(rows, LEVELS - 1, generator=generator, dtype=torch.float64)
        * TAIL_GAP,
    )
    cutpoints = subject.cutpoints(given, effects=effects)
    cumulative = subject.cumulative_probabilities(latent, given, effects=effects)
    probabilities = subject.probabilities(latent, given, effects=effects)

    assert cutpoints.shape == (rows, LEVELS - 1)
    assert bool(torch.all(torch.diff(cutpoints, dim=1) >= 0.0))
    assert bool(torch.all(torch.isfinite(cumulative)))
    assert bool(torch.all(torch.diff(cumulative, dim=1) >= 0.0))
    assert bool(torch.all(torch.isfinite(probabilities)))
    assert bool(torch.all(probabilities >= 0.0))
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(rows, dtype=torch.float64),
        rtol=0.0,
        atol=1e-6,
    )


def test_discrimination_scales_the_offsets_with_the_rest_of_the_model() -> None:
    declared = spec(discrimination=True)
    subject, given = head(declared), parameters(declared)
    scale = subject.discrimination(given)
    latent = torch.tensor([0.25, -1.0], dtype=torch.float64)
    effects = effects_of((1.0, -0.5), ((0.3,) * 6, (-0.3,) * 6))

    assert effects.spans is not None
    assert torch.allclose(
        subject.cutpoints(given, effects=effects),
        scale * ordered_cutpoints(given.base, given.spans + effects.spans),
        rtol=0.0,
    )
    assert torch.allclose(
        subject.cumulative_probabilities(latent, given, effects=effects),
        torch.sigmoid(
            subject.cutpoints(given, effects=effects)
            - (scale * (latent + effects.latent)).unsqueeze(1)
        ),
        rtol=0.0,
        atol=1e-12,
    )


def test_the_log_likelihood_applies_the_effects_of_each_row() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    responses = (0, 6, 3)
    design = design_of(responses, (0.5, -2.0, 1.25))
    effects = effects_of((0.4, -0.4, 0.0), ((0.1,) * 6, (-0.1,) * 6, (0.0,) * 6))
    every = subject.log_probabilities(
        subject.latent_scores(design), given, effects=effects
    )
    expected = torch.stack(
        [every[position, response] for position, response in enumerate(responses)]
    )
    scored = subject.log_likelihood(design, given, effects=effects)

    assert torch.equal(scored, expected)
    assert not torch.equal(scored, subject.log_likelihood(design, given))


def test_effects_pass_a_gradient_to_their_offsets() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent_offsets = torch.zeros(2, dtype=torch.float64, requires_grad=True)
    span_offsets = torch.zeros(2, LEVELS - 1, dtype=torch.float64, requires_grad=True)
    latent = torch.tensor([-STABILITY_GAP, 0.5], dtype=torch.float64)
    subject.log_probabilities(
        latent, given, effects=RowEffects(latent_offsets, span_offsets)
    ).sum().backward()

    assert latent_offsets.grad is not None
    assert span_offsets.grad is not None
    assert bool(torch.all(torch.isfinite(span_offsets.grad)))
    assert bool(torch.all(span_offsets.grad != 0.0))


def test_a_head_refuses_effects_over_another_number_of_rows() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)
    latent = torch.tensor([0.0, 1.0], dtype=torch.float64)

    with pytest.raises(ValueError, match="2 latent scores were given effects over 3"):
        subject.probabilities(latent, given, effects=effects_of((0.0, 0.0, 0.0), None))


def test_a_head_refuses_effects_over_another_number_of_cutpoints() -> None:
    declared = spec()
    subject, given = head(declared), parameters(declared)

    with pytest.raises(ValueError, match="6 cutpoints was given effects over 5"):
        subject.cutpoints(given, effects=effects_of((0.0,), ((0.0,) * 5,)))


TRAINED_CUTPOINTS = (-2.0, -0.5, 1.0, 2.5)


def trained() -> OrdinalHead:
    return OrdinalHead(5, TRAINED_CUTPOINTS)


def test_the_default_cutpoints_are_evenly_spaced_about_zero() -> None:
    held = OrdinalHead(5).cutpoints().tolist()

    assert len(held) == 4
    assert held == pytest.approx([-1.5, -0.5, 0.5, 1.5], rel=1e-9)
    assert sum(held) == pytest.approx(0.0, abs=1e-9)


def test_the_declared_cutpoints_are_recovered_from_the_parameters() -> None:
    subject = trained()

    assert subject.levels == 5
    assert subject.cutpoint_count == 4
    assert subject.cutpoints().tolist() == pytest.approx(
        list(TRAINED_CUTPOINTS), rel=1e-9
    )
    assert repr(subject) == "OrdinalHead(levels=5)"


def test_the_cutpoints_stay_ordered_under_any_parameter_values() -> None:
    subject = trained()
    generator = torch.Generator().manual_seed(11)

    for _ in range(50):
        with torch.no_grad():
            subject.base.copy_(torch.randn((), generator=generator) * 10.0)
            subject.spans.copy_(torch.randn(4, generator=generator) * 10.0)

        held = subject.cutpoints()

        assert bool((held[1:] > held[:-1]).all())


def test_the_trainable_head_scores_a_latent_score_as_the_functional_head_does() -> None:
    subject = trained()
    latent = torch.tensor([-3.0, 0.2, 4.0], dtype=torch.float64)

    scored = subject.log_probabilities(latent)

    assert torch.allclose(
        scored, log_category_probabilities(latent, subject.cutpoints())
    )
    assert torch.allclose(scored.exp().sum(dim=1), torch.ones(3, dtype=torch.float64))


def test_the_trainable_head_parameters_are_its_base_and_spans() -> None:
    named = dict(trained().named_parameters())

    assert set(named) == {"base", "spans"}
    assert named["base"].shape == ()
    assert named["spans"].shape == (4,)


def test_the_gradient_of_the_observed_level_reaches_both_parameters() -> None:
    subject = trained()
    latent = torch.tensor([0.3, -1.0], dtype=torch.float64)
    responses = torch.tensor([2, 0])

    subject.log_probabilities(latent).gather(1, responses.unsqueeze(1)).sum().backward()

    assert subject.base.grad is not None
    assert subject.spans.grad is not None
    assert float(subject.base.grad.abs()) > 0.0
    assert float(subject.spans.grad.abs().sum()) > 0.0


@pytest.mark.parametrize("levels", [0, 1])
def test_fewer_than_two_levels_is_refused(levels: int) -> None:
    with pytest.raises(ValueError, match="levels"):
        OrdinalHead(levels)


def test_cutpoints_of_another_count_than_the_levels_need_are_refused() -> None:
    with pytest.raises(ValueError, match="cutpoints"):
        OrdinalHead(5, (-1.0, 1.0))


def test_cutpoints_out_of_order_are_refused() -> None:
    with pytest.raises(ValueError, match="increasing"):
        OrdinalHead(5, (-2.0, 1.0, -0.5, 2.5))


def test_a_cutpoint_that_is_not_finite_is_refused() -> None:
    with pytest.raises(ValueError, match="cutpoint 1"):
        OrdinalHead(5, (-2.0, math.inf, 1.0, 2.5))
