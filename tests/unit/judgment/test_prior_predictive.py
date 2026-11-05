import math
from typing import TYPE_CHECKING, Final, Never

import pytest
import torch
from quivers.inference import SVI

from lofi.judgment import (
    LINEAR,
    CalibrationCampaign,
    CalibrationSeeds,
    ConditionalLikelihood,
    Cumulative,
    DesignCodec,
    FixedDiscrimination,
    InferenceMethod,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    NutsPlan,
    PosteriorFittingError,
    Predictor,
    PriorBlock,
    QuiversGuideFitter,
    QuiversNutsFitter,
    RandomIntercept,
    ResponseCompiler,
    ResponseProgram,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    ensure_participant_thresholds,
    latent,
    linear_steps,
    participant_cutpoints,
    prior_predictive,
    prior_predictive_calibration,
    sample_prior,
)
from lofi.judgment.compiler import ITEM, PARTICIPANT

if TYPE_CHECKING:
    from lofi.judgment import BoundProgram, PosteriorFit

LEVELS: Final = 5
SCORE: Final = "score"
METHODS: Final = (
    InferenceMethod.NUTS,
    InferenceMethod.STRUCTURED,
    InferenceMethod.MEAN_FIELD,
)
GUIDES: Final = METHODS[1:]
RATE: Final = 0.05
SEEDS: Final = CalibrationSeeds(simulation=1, inference=2, rank=3)


def design() -> JudgmentDesign:
    codec = DesignCodec(LEVELS, factors=(), readouts=(SCORE,))

    return codec.encode(
        JudgmentRow(
            f"participant-{participant}",
            f"item-{item}",
            (participant + item) % LEVELS,
            RowColumns(factors={}, readouts={SCORE: -1.5 + item}),
        )
        for participant in range(2)
        for item in range(4)
    )


def compiled(*grouped: Predictor) -> ResponseProgram:
    response = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            LinkTerm(readout=SCORE, linking=LinkingSpec(family=LINEAR)),
            *grouped,
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )

    return ResponseCompiler({LINEAR: linear_steps}).compile(response)


def crossed() -> ResponseProgram:
    return compiled(RandomIntercept(group=PARTICIPANT), RandomIntercept(group=ITEM))


def test_a_guide_fit_draws_one_reproducible_chain_over_the_program_layout() -> None:
    program = crossed().over(design())
    torch.manual_seed(0)
    before = torch.get_rng_state()

    first, again, other = (
        QuiversGuideFitter(
            InferenceMethod.MEAN_FIELD,
            iterations=10,
            draws=6,
            learning_rate=RATE,
            seed=seed,
        )
        .fit(program)
        .draws
        for seed in (4, 4, 5)
    )

    assert first.shape == (1, 6, program.size)
    assert first.dtype == torch.float64
    assert torch.equal(first, again)
    assert not torch.equal(first, other)
    assert torch.equal(torch.get_rng_state(), before)


def test_the_structured_and_mean_field_guides_fit_different_families() -> None:
    program = crossed().over(design())

    structured, mean_field = (
        QuiversGuideFitter(method, iterations=10, draws=6, learning_rate=RATE, seed=4)
        .fit(program)
        .draws
        for method in GUIDES
    )

    assert structured.shape == mean_field.shape
    assert not torch.allclose(structured, mean_field)


def test_a_guide_is_fitted_at_its_declared_learning_rate() -> None:
    program = crossed().over(design())
    slow, fast = (
        QuiversGuideFitter(
            InferenceMethod.MEAN_FIELD,
            iterations=10,
            draws=6,
            learning_rate=rate,
            seed=4,
        )
        for rate in (0.01, 0.1)
    )

    assert (slow.learning_rate, fast.learning_rate) == (0.01, 0.1)
    assert not torch.equal(slow.fit(program).draws, fast.fit(program).draws)


@pytest.mark.parametrize("rate", [0.0, -0.1, math.nan, math.inf])
def test_a_guide_s_learning_rate_is_positive_and_finite(rate: float) -> None:
    with pytest.raises(ValueError, match="learning rate"):
        QuiversGuideFitter(
            InferenceMethod.MEAN_FIELD,
            iterations=1,
            draws=1,
            learning_rate=rate,
            seed=0,
        )


@pytest.mark.parametrize(
    ("method", "iterations", "draws", "match"),
    [
        (InferenceMethod.NUTS, 1, 1, "structured or the mean-field"),
        (InferenceMethod.MAP, 1, 1, "structured or the mean-field"),
        (InferenceMethod.STRUCTURED, 0, 1, "guide iterations"),
        (InferenceMethod.MEAN_FIELD, 1, 0, "guide draws"),
    ],
)
def test_a_guide_fitter_builds_only_a_guide_fitted_and_drawn_at_least_once(
    method: InferenceMethod, iterations: int, draws: int, match: str
) -> None:
    with pytest.raises(ValueError, match=match):
        QuiversGuideFitter(
            method, iterations=iterations, draws=draws, learning_rate=RATE, seed=0
        )


def test_each_participant_s_cutpoints_are_those_their_rows_are_observed_under() -> None:
    source = design()
    program = crossed().over(source)
    params = sample_prior(program, generator=torch.Generator().manual_seed(2))
    thresholds = latent(program, params).thresholds

    table = participant_cutpoints(program, source, params)

    assert table.shape == (2, LEVELS - 1)
    assert torch.equal(table[source.participants], thresholds)
    assert not torch.equal(table[0], table[1])


def test_cutpoints_shared_by_every_participant_are_refused() -> None:
    ensure_participant_thresholds(crossed())

    with pytest.raises(ValueError, match="do not vary over participants"):
        ensure_participant_thresholds(compiled(RandomIntercept(group=ITEM)))


def _failing_step(error: BaseException) -> object:
    def fail(inference: SVI, *arguments: object) -> Never:
        del inference, arguments
        raise error

    return fail


def a_guide_fit() -> torch.Tensor:
    return (
        QuiversGuideFitter(
            InferenceMethod.MEAN_FIELD,
            iterations=2,
            draws=2,
            learning_rate=RATE,
            seed=0,
        )
        .fit(crossed().over(design()))
        .draws
    )


def test_an_exhausted_accelerator_in_a_guide_fit_is_memory_exhaustion(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    exhausted = torch.OutOfMemoryError("CUDA out of memory")
    monkeypatch.setattr(SVI, "step", _failing_step(exhausted))

    with pytest.raises(MemoryError, match="CUDA out of memory") as raised:
        a_guide_fit()

    assert not isinstance(raised.value, PosteriorFittingError)
    assert raised.value.__cause__ is exhausted


@pytest.mark.parametrize(
    "error",
    [
        ValueError("Expected parameter loc to satisfy Real(), found invalid values"),
        torch.linalg.LinAlgError("the covariance is not positive-definite"),
    ],
    ids=["invalid-parameter", "not-positive-definite"],
)
def test_a_numerical_condition_in_a_guide_fit_is_a_fitting_failure(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SVI, "step", _failing_step(error))

    with pytest.raises(PosteriorFittingError, match="guide fitting failed") as raised:
        a_guide_fit()

    assert raised.value.__cause__ is error


@pytest.mark.parametrize(
    "error",
    [RuntimeError("an estimator was misconfigured"), TypeError("a site has no shape")],
    ids=["runtime", "type"],
)
def test_a_defect_in_a_guide_fit_propagates_as_itself(
    error: Exception, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(SVI, "step", _failing_step(error))

    with pytest.raises(type(error)) as raised:
        a_guide_fit()

    assert raised.value is error


def test_every_replicate_and_method_takes_its_own_seed_from_its_stream() -> None:
    seeds = CalibrationSeeds(simulation=1, inference=2, rank=3)
    again = CalibrationSeeds(simulation=1, inference=2, rank=3)
    other = CalibrationSeeds(simulation=9, inference=8, rank=7)
    inference = {
        seeds.inference(method, replicate) for method in METHODS for replicate in (0, 1)
    }

    assert seeds.rank == 3
    assert seeds.simulation(0) != seeds.simulation(1)
    assert len(inference) == len(METHODS) * 2
    assert [seeds.simulation(0), seeds.inference(METHODS[0], 0)] == [
        again.simulation(0),
        again.inference(METHODS[0], 0),
    ]
    assert seeds.simulation(0) != other.simulation(0)
    assert seeds.inference(METHODS[0], 0) != other.inference(METHODS[0], 0)


def calibrate(replicates: int) -> CalibrationCampaign:
    return prior_predictive_calibration(
        crossed(),
        design(),
        replicates=replicates,
        guide_iterations=20,
        guide_learning_rate=RATE,
        plan=NutsPlan(warmup=5, retained=8, chains=2, seed=0),
        seeds=SEEDS,
    )


@pytest.fixture(scope="module")
def campaign() -> tuple[CalibrationCampaign, list[ResponseProgram]]:
    checked: list[ResponseProgram] = []

    def counted(program: ResponseProgram, /) -> None:
        checked.append(program)

    with pytest.MonkeyPatch.context() as patched:
        patched.setattr(prior_predictive, "ensure_participant_thresholds", counted)
        calibrated = calibrate(2)

    return calibrated, checked


def test_calibration_leaves_the_threshold_check_to_its_caller(
    campaign: tuple[CalibrationCampaign, list[ResponseProgram]],
) -> None:
    assert campaign[1] == []


def test_a_calibration_reduces_every_replicate_fitted_by_every_method(
    campaign: tuple[CalibrationCampaign, list[ResponseProgram]],
) -> None:
    program = crossed()
    report = campaign[0].report

    assert report is not None
    assert report.replicates == 2
    assert report.rank_seed == SEEDS.rank
    assert [summary.block for summary in report.sbc] == sorted(
        draw.site for draw in program.draws
    )
    assert {summary.posterior_samples for summary in report.sbc} == {16}
    assert {summary.method for summary in report.coverage} == set(METHODS)
    assert {summary.participants for summary in report.cutpoints} == {4}
    assert {summary.cutpoint for summary in report.cutpoints} == set(range(LEVELS - 1))


def test_every_replicate_records_its_seed_and_each_method_s_diagnostics(
    campaign: tuple[CalibrationCampaign, list[ResponseProgram]],
) -> None:
    records = campaign[0].records
    sites = {draw.site for draw in crossed().draws}

    assert [record.replicate for record in records] == [0, 1]
    assert [record.seed for record in records] == [
        SEEDS.simulation(0),
        SEEDS.simulation(1),
    ]
    assert campaign[0].failed == ()

    for record in records:
        nuts = record.nuts

        assert record.retained
        assert record.failures == {}
        assert nuts is not None
        assert len(nuts.divergences) == 2
        assert all(count >= 0 for count in nuts.divergences)
        assert len(nuts.acceptance_rates) == 2
        assert set(nuts.r_hat) == sites
        assert set(nuts.effective_sample_sizes) == sites
        assert nuts.max_rhat == max(nuts.r_hat.values())
        assert nuts.min_ess == min(nuts.effective_sample_sizes.values())
        assert set(record.evidence_bounds) == set(GUIDES)
        assert all(math.isfinite(bound) for bound in record.evidence_bounds.values())


def _refusing_nuts(
    monkeypatch: pytest.MonkeyPatch, *, replicates: tuple[int, ...]
) -> str:
    fit = QuiversNutsFitter.fit
    refused = {SEEDS.inference(InferenceMethod.NUTS, each) for each in replicates}
    message = "NUTS posterior fitting failed: every retained transition diverged"

    def refusing(fitter: QuiversNutsFitter, program: BoundProgram, /) -> PosteriorFit:
        if fitter.plan.seed in refused:
            raise PosteriorFittingError(message)

        return fit(fitter, program)

    monkeypatch.setattr(QuiversNutsFitter, "fit", refusing)

    return message


def test_a_failed_fit_is_retained_and_the_other_replicates_are_reduced(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    message = _refusing_nuts(monkeypatch, replicates=(0,))

    calibrated = calibrate(2)
    failed, fitted = calibrated.records

    assert calibrated.failed == (failed,)
    assert not failed.retained
    assert failed.failures == {InferenceMethod.NUTS: message}
    assert failed.nuts is None
    assert set(failed.evidence_bounds) == set(GUIDES)
    assert fitted.retained
    assert calibrated.report is not None
    assert calibrated.report.replicates == 1
    assert {summary.participants for summary in calibrated.report.cutpoints} == {2}


def test_a_campaign_retaining_no_replicate_has_no_report(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _refusing_nuts(monkeypatch, replicates=(0,))

    calibrated = calibrate(1)

    assert calibrated.report is None
    assert calibrated.failed == calibrated.records


def test_a_calibration_runs_at_least_one_replicate() -> None:
    with pytest.raises(ValueError, match="calibration replicates"):
        prior_predictive_calibration(
            crossed(),
            design(),
            replicates=0,
            guide_iterations=1,
            guide_learning_rate=RATE,
            plan=NutsPlan(warmup=5, retained=8, chains=2, seed=0),
            seeds=CalibrationSeeds(simulation=1, inference=2, rank=3),
        )
