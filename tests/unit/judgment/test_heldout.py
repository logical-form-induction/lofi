import ast
from pathlib import Path
from types import MappingProxyType
from typing import Final

import didactic.api as dx
import pytest
import torch

import lofi.judgment
from lofi.judgment import (
    PARTICIPANT,
    BoundProgram,
    ConditionalLikelihood,
    CoverageGate,
    Cumulative,
    DesignCodec,
    FactorTerm,
    FixedCutpointVariance,
    FixedDiscrimination,
    FoldPredictions,
    HeldOutPredictor,
    JudgmentDesign,
    JudgmentRow,
    KFold,
    Measurement,
    OffsetTerm,
    PosteriorVarianceRow,
    PredictiveDensityRow,
    PriorBlock,
    PriorSpec,
    PseudoRSquaredRow,
    RandomIntercept,
    RandomSlope,
    ResponseCompiler,
    ResponseProgram,
    ResponseSpec,
    RowColumns,
    ThresholdPriors,
    TruncatedMarginalLikelihood,
    fold_predictions,
    posterior_variance_row,
    predictive_density_row,
    predictive_log_density,
    pseudo_r_squared,
)
from lofi.judgment.backends import latent
from lofi.judgment.heldout import MarginalPointPredictor
from lofi.judgment.marginal import ApproximateMarginalLikelihood
from lofi.judgment.ordinal import log_category_probabilities
from lofi.spec import HeldOutParticipantPolicy

LEVELS: Final = 5
SCORE: Final = "score"
VERB: Final = "verb"
ITEMS: Final = 6
DRAWS: Final = 40
POLICIES: Final = (HeldOutParticipantPolicy.MARGINAL, HeldOutParticipantPolicy.PLUG_IN)
MARGINAL_LEVELS: Final = 2


def intercept_spec() -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(RandomIntercept(group=PARTICIPANT), OffsetTerm(readout=SCORE)),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


def slope_spec() -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(
            RandomIntercept(group=PARTICIPANT),
            FactorTerm(name=VERB),
            RandomSlope(group=PARTICIPANT, term=FactorTerm(name=VERB)),
        ),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


def item_spec() -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(RandomIntercept(group="item"), OffsetTerm(readout=SCORE)),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


def discrimination_spec(distribution: str, *parameters: float) -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(discrimination=True, discrimination_group=PARTICIPANT),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(OffsetTerm(readout=SCORE),),
        identification=FixedCutpointVariance(value=1.0),
        priors=PriorBlock(
            thresholds=ThresholdPriors(),
            discrimination=PriorSpec(distribution=distribution, parameters=parameters),
        ),
    )


def marginal_spec() -> ResponseSpec:
    return ResponseSpec(
        family=Cumulative(),
        levels=MARGINAL_LEVELS,
        objective=TruncatedMarginalLikelihood(
            group=PARTICIPANT,
            radius=4.0,
            relative_tolerance=1e-7,
            absolute_tolerance=1e-9,
            quadrature_points=512,
            quadrature_seed=23,
        ),
        predictor=(RandomIntercept(group=PARTICIPANT), OffsetTerm(readout=SCORE)),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )


def compiled(spec: ResponseSpec) -> ResponseProgram:
    return ResponseCompiler({}).compile(spec)


def codec() -> DesignCodec:
    return DesignCodec(LEVELS, factors=(VERB,), readouts=(SCORE,))


def rows(participants: tuple[str, ...], *, offset: int = 0) -> tuple[JudgmentRow, ...]:
    return tuple(
        JudgmentRow(
            participant,
            f"item-{item + offset}",
            (item + position) % LEVELS,
            RowColumns({VERB: f"verb-{item % 2}"}, {SCORE: 0.4 * item - 1.0}),
        )
        for position, participant in enumerate(participants)
        for item in range(ITEMS)
    )


def design(participants: tuple[str, ...], *, offset: int = 0) -> JudgmentDesign:
    return codec().encode(rows(participants, offset=offset))


FITTED: Final = ("a", "b", "c", "d")
UNSEEN: Final = ("e", "f")


def draws_of(
    fit: BoundProgram, *, spread: float, seed: int = 0, count: int = DRAWS
) -> torch.Tensor:
    """Return synthetic draws with every participant scale held at ``spread``."""
    generator = torch.Generator().manual_seed(seed)
    pieces = []

    for slot in fit.layout:
        shape = (count, slot.size)
        noise = torch.randn(shape, dtype=torch.float64, generator=generator)

        if slot.site.endswith(("_scale", "_spread")):
            pieces.append(torch.full(shape, spread, dtype=torch.float64))
        elif slot.site == "cutpoint_base":
            pieces.append(-1.5 + 0.1 * noise)
        elif slot.site == "discrimination":
            pieces.append(1.0 + 0.1 * noise.abs())
        elif slot.site.endswith("_offsets"):
            pieces.append(noise)
        else:
            pieces.append(0.2 * noise)

    return torch.cat(pieces, dim=1)


def predictor(
    spec: ResponseSpec, *, spread: float, samples: int = 8, seed: int = 0
) -> HeldOutPredictor:
    program = compiled(spec)
    fitted = design(FITTED)

    return HeldOutPredictor(
        program,
        fitted,
        draws_of(program.over(fitted), spread=spread, seed=seed),
        samples=samples,
    )


def seeded(seed: int) -> torch.Generator:
    return torch.Generator().manual_seed(seed)


class _Point:
    """A test point implementing the predictor's narrow fit capability."""

    __slots__ = ("_parameters",)

    def __init__(self, parameters: torch.Tensor) -> None:
        self._parameters = parameters

    @property
    def parameters(self) -> torch.Tensor:
        return self._parameters


def marginal_codec() -> DesignCodec:
    return DesignCodec(MARGINAL_LEVELS, readouts=(SCORE,))


def marginal_rows(
    responses: tuple[tuple[str, int], ...],
    *,
    reverse: bool = False,
) -> tuple[JudgmentRow, ...]:
    records = tuple(
        JudgmentRow(
            participant,
            f"retained-{position}",
            response,
            RowColumns({}, {SCORE: 0.0}),
        )
        for position, (participant, response) in enumerate(responses)
    )

    return tuple(reversed(records)) if reverse else records


def marginal_point(
    program: ResponseProgram,
    retained: JudgmentDesign,
    *,
    local_scale: float,
) -> _Point:
    kernel = ApproximateMarginalLikelihood(
        program.over(retained),
        active_groups=tuple(range(len(retained.participant_levels))),
    )
    pieces = []

    for slot in kernel.population_layout:
        if slot.site == "participant_intercept_scale":
            pieces.append(torch.full((slot.size,), local_scale, dtype=torch.float64))
        elif slot.site == "participant_cutpoint_spread":
            pieces.append(torch.zeros(slot.size, dtype=torch.float64))
        else:
            pieces.append(torch.zeros(slot.size, dtype=torch.float64))

    return _Point(torch.cat(pieces))


def marginal_predictor(
    retained_rows: tuple[JudgmentRow, ...], *, local_scale: float
) -> MarginalPointPredictor:
    program = compiled(marginal_spec())
    codec_ = marginal_codec()
    retained = codec_.encode(retained_rows)

    return MarginalPointPredictor(
        program,
        codec_,
        retained,
        marginal_point(program, retained, local_scale=local_scale),
    )


def candidate(
    *participants: str, score: float = 0.0, reverse: bool = False
) -> JudgmentDesign:
    rows_ = tuple(
        JudgmentRow(
            participant,
            f"candidate-{position}",
            0,
            RowColumns({}, {SCORE: score}),
        )
        for position, participant in enumerate(participants)
    )

    return marginal_codec().encode(tuple(reversed(rows_)) if reverse else rows_)


def plug_in_probabilities(
    predictor_: MarginalPointPredictor, design_: JudgmentDesign
) -> torch.Tensor:
    program = compiled(marginal_spec())
    bound = program.over(design_)
    retained = marginal_codec().encode(marginal_rows((("a", 0), ("a", 1))))
    population_layout = ApproximateMarginalLikelihood(
        program.over(retained), active_groups=(0,)
    ).population_layout
    values = predictor_.parameters
    split = torch.split(values, [slot.size for slot in population_layout])
    population = {
        slot.site: value for slot, value in zip(population_layout, split, strict=True)
    }
    parameters = torch.cat(
        tuple(
            population.get(slot.site, values.new_zeros(slot.size))
            for slot in bound.layout
        )
    )
    location, thresholds = latent(bound, parameters)

    return torch.exp(log_category_probabilities(location, thresholds))


def assert_proper(
    probabilities: torch.Tensor, rows_expected: int, levels: int = LEVELS
) -> None:
    assert probabilities.shape == (rows_expected, levels)
    assert bool((probabilities > 0.0).all())
    assert torch.allclose(
        probabilities.sum(dim=1), torch.ones(rows_expected, dtype=torch.float64)
    )


def test_a_seen_point_prediction_conditions_on_its_retained_responses() -> None:
    positive = marginal_predictor(
        marginal_rows((("a", 1), ("a", 1), ("a", 1), ("b", 0))),
        local_scale=3.0,
    )
    negative = marginal_predictor(
        marginal_rows((("a", 0), ("a", 0), ("a", 0), ("b", 0))),
        local_scale=3.0,
    )
    targets = candidate("a", "new")
    after_positive = positive.probabilities(targets)
    after_negative = negative.probabilities(targets)

    assert_proper(after_positive, 2, MARGINAL_LEVELS)
    assert after_positive[0, 1] > after_negative[0, 1]
    assert torch.allclose(after_positive[1], after_negative[1], atol=1e-12, rtol=0.0)


def test_a_new_participant_uses_the_local_prior_instead_of_a_plug_in() -> None:
    fitted = marginal_predictor(
        marginal_rows((("a", 0), ("a", 1), ("b", 0), ("b", 1))),
        local_scale=4.0,
    )
    target = candidate("new", score=1.5)
    integrated = fitted.probabilities(target)
    plugged_in = plug_in_probabilities(fitted, target)

    assert_proper(integrated, 1, MARGINAL_LEVELS)
    assert_proper(plugged_in, 1, MARGINAL_LEVELS)
    assert float((integrated - plugged_in).abs().max()) > 0.05


def test_zero_local_variance_agrees_with_the_population_mean_plug_in() -> None:
    fitted = marginal_predictor(
        marginal_rows((("a", 1), ("a", 1), ("b", 0), ("b", 0))),
        local_scale=0.0,
    )
    targets = candidate("a", "new", score=1.0)

    assert torch.allclose(
        fitted.probabilities(targets),
        plug_in_probabilities(fitted, targets),
        atol=1e-12,
        rtol=0.0,
    )


def test_marginal_point_predictions_ignore_row_group_and_target_order() -> None:
    responses = (("b", 0), ("a", 1), ("b", 1), ("a", 1))
    forward = marginal_predictor(marginal_rows(responses), local_scale=2.0)
    backward = marginal_predictor(
        marginal_rows(responses, reverse=True), local_scale=2.0
    )
    expected = forward.probabilities(candidate("a", "new", "b"))
    reordered = backward.probabilities(candidate("a", "new", "b", reverse=True))

    assert torch.allclose(reordered.flip(0), expected, atol=1e-12, rtol=0.0)


def test_a_marginal_point_predictor_preserves_an_empty_target() -> None:
    fitted = marginal_predictor(marginal_rows((("a", 0), ("a", 1))), local_scale=1.0)

    assert fitted.probabilities(marginal_codec().encode(())).shape == (
        0,
        MARGINAL_LEVELS,
    )


@pytest.mark.parametrize("spec", [intercept_spec(), slope_spec()])
def test_the_conventions_agree_when_the_participant_variance_is_zero(
    spec: ResponseSpec,
) -> None:
    both = predictor(spec, spread=0.0).both(design(UNSEEN), generator=seeded(1))

    assert_proper(both.plug_in, both.rows)
    assert torch.allclose(both.marginal, both.plug_in, rtol=0.0, atol=1e-12)


@pytest.mark.parametrize("spec", [intercept_spec(), slope_spec()])
def test_the_conventions_differ_when_the_participant_variance_is_high(
    spec: ResponseSpec,
) -> None:
    both = predictor(spec, spread=3.0).both(design(UNSEEN), generator=seeded(1))

    assert_proper(both.marginal, both.rows)
    assert_proper(both.plug_in, both.rows)
    assert float((both.marginal - both.plug_in).abs().max()) > 0.05


def test_the_plug_in_prediction_does_not_depend_on_the_generator() -> None:
    fitted = predictor(intercept_spec(), spread=3.0)
    held_out = design(UNSEEN)
    first, second = (
        fitted.probabilities(
            held_out, policy=HeldOutParticipantPolicy.PLUG_IN, generator=seeded(seed)
        )
        for seed in (1, 2)
    )

    assert torch.equal(first, second)


def test_the_marginal_prediction_draws_the_unseen_effects_from_the_generator() -> None:
    fitted = predictor(intercept_spec(), spread=3.0, samples=1)
    held_out = design(UNSEEN)
    first, second = (
        fitted.probabilities(
            held_out, policy=HeldOutParticipantPolicy.MARGINAL, generator=seeded(seed)
        )
        for seed in (1, 2)
    )

    assert not torch.allclose(first, second)


def test_more_samples_per_draw_bring_two_marginal_predictions_closer() -> None:
    held_out = design(UNSEEN)

    def spread_of(samples: int) -> float:
        fitted = predictor(intercept_spec(), spread=3.0, samples=samples)
        first, second = (
            fitted.probabilities(
                held_out,
                policy=HeldOutParticipantPolicy.MARGINAL,
                generator=seeded(seed),
            )
            for seed in (1, 2)
        )

        return float((first - second).abs().max())

    assert spread_of(64) < spread_of(1)


@pytest.mark.parametrize("policy", POLICIES)
def test_parameters_over_the_fitted_design_are_the_draws(
    policy: HeldOutParticipantPolicy,
) -> None:
    fitted = predictor(slope_spec(), spread=3.0, samples=3)
    vectors = fitted.parameters(design(FITTED), policy=policy, generator=seeded(1))
    repeats = fitted.samples if policy is HeldOutParticipantPolicy.MARGINAL else 1

    assert torch.equal(vectors, fitted.draws.repeat_interleave(repeats, dim=0))


@pytest.mark.parametrize("spec", [intercept_spec(), slope_spec()])
@pytest.mark.parametrize("policy", POLICIES)
def test_predictions_follow_the_rows_when_the_row_order_changes(
    spec: ResponseSpec, policy: HeldOutParticipantPolicy
) -> None:
    fitted = predictor(spec, spread=3.0)
    forward = codec().encode(rows(FITTED))
    backward = codec().encode(tuple(reversed(rows(FITTED))))

    assert backward.participant_levels == forward.participant_levels

    expected = fitted.probabilities(forward, policy=policy, generator=seeded(1))
    reordered = fitted.probabilities(backward, policy=policy, generator=seeded(1))

    assert torch.allclose(reordered.flip(0), expected)


def test_a_seen_participant_keeps_its_fitted_effects_beside_an_unseen_one() -> None:
    fitted = predictor(intercept_spec(), spread=3.0)
    mixed = design(("a", "e"))
    alone = design(("a",))
    both = fitted.both(mixed, generator=seeded(1))
    seen = mixed.participants == 0

    assert int(seen.sum()) == ITEMS
    assert torch.allclose(both.marginal[seen], both.plug_in[seen])
    assert torch.allclose(
        both.plug_in[seen],
        fitted.probabilities(
            alone, policy=HeldOutParticipantPolicy.PLUG_IN, generator=seeded(1)
        ),
    )
    assert not torch.allclose(both.marginal[~seen], both.plug_in[~seen])


@pytest.mark.parametrize("policy", POLICIES)
def test_an_unseen_level_of_another_factor_is_refused(
    policy: HeldOutParticipantPolicy,
) -> None:
    fitted = predictor(item_spec(), spread=3.0)

    with pytest.raises(ValueError, match="'item'"):
        fitted.probabilities(
            design(("a",), offset=ITEMS), policy=policy, generator=seeded(1)
        )


def test_a_participant_site_with_a_lognormal_prior_plugs_in_its_mean() -> None:
    program = compiled(discrimination_spec("LogNormal", 0.0, 0.5))
    fitted = design(FITTED)
    held_out = design(("a", "e"))
    draws = draws_of(program.over(fitted), spread=1.0)
    vectors = HeldOutPredictor(program, fitted, draws, samples=1).parameters(
        held_out, policy=HeldOutParticipantPolicy.PLUG_IN, generator=seeded(1)
    )
    discrimination = torch.stack(
        [program.over(held_out).sites(vector)["discrimination"] for vector in vectors]
    )

    fitted_discrimination = torch.stack(
        [program.over(fitted).sites(draw)["discrimination"] for draw in draws]
    )

    assert torch.equal(discrimination[:, 0], fitted_discrimination[:, 0])
    assert torch.allclose(
        discrimination[:, 1],
        torch.full((DRAWS,), 0.125, dtype=torch.float64).exp(),
    )


def test_a_participant_site_with_a_lognormal_prior_is_sampled_positive() -> None:
    program = compiled(discrimination_spec("LogNormal", 0.0, 0.5))
    fitted = design(FITTED)
    held_out = design(UNSEEN)
    draws = draws_of(program.over(fitted), spread=1.0)
    vectors = HeldOutPredictor(program, fitted, draws, samples=4).parameters(
        held_out, policy=HeldOutParticipantPolicy.MARGINAL, generator=seeded(1)
    )
    discrimination = torch.stack(
        [program.over(held_out).sites(vector)["discrimination"] for vector in vectors]
    )

    assert discrimination.shape == (4 * DRAWS, len(UNSEEN))
    assert bool((discrimination > 0.0).all())
    assert float(discrimination.std()) > 0.0


@pytest.mark.parametrize("policy", POLICIES)
def test_a_participant_site_with_a_prior_of_no_mean_is_refused(
    policy: HeldOutParticipantPolicy,
) -> None:
    program = compiled(discrimination_spec("Cauchy", 0.0, 1.0))
    fitted = design(FITTED)
    draws = draws_of(program.over(fitted), spread=1.0)
    predictor_ = HeldOutPredictor(program, fitted, draws, samples=1)

    with pytest.raises(ValueError, match="'discrimination'"):
        predictor_.parameters(design(UNSEEN), policy=policy, generator=seeded(1))


def test_the_draws_and_the_sample_count_are_validated() -> None:
    program = compiled(intercept_spec())
    fitted = design(FITTED)
    draws = draws_of(program.over(fitted), spread=1.0)

    with pytest.raises(ValueError, match="parameters"):
        HeldOutPredictor(program, fitted, draws[0], samples=1)

    with pytest.raises(ValueError, match="parameters"):
        HeldOutPredictor(program, fitted, draws[:, :-1], samples=1)

    with pytest.raises(ValueError, match="parameters"):
        HeldOutPredictor(program, fitted, draws[:0], samples=1)

    with pytest.raises(TypeError, match="floating"):
        HeldOutPredictor(program, fitted, draws.to(torch.int64), samples=1)

    with pytest.raises(ValueError, match="at least one sample"):
        HeldOutPredictor(program, fitted, draws, samples=0)

    with pytest.raises(TypeError, match="not an integer"):
        HeldOutPredictor(program, fitted, draws, samples=True)


def test_the_predictor_reports_its_draws_and_samples() -> None:
    fitted = predictor(intercept_spec(), spread=1.0, samples=3)

    assert fitted.samples == 3
    assert fitted.draws.shape == (DRAWS, fitted.fit.size)
    assert fitted.program == compiled(intercept_spec())
    assert repr(fitted) == (
        f"HeldOutPredictor(draws={DRAWS}, parameters={fitted.fit.size}, samples=3)"
    )


def test_fold_predictions_cover_both_conventions_on_a_participant_held_out_fold() -> (
    None
):
    program = compiled(intercept_spec())
    split = KFold(PARTICIPANT, folds=2, seed=3).split(rows(FITTED), CoverageGate(1.0))

    for fold in split.folds:
        assert fold.coverage.fraction == 1.0

        retained = codec().encode(fold.retained)
        both = fold_predictions(
            program,
            codec(),
            fold,
            draws_of(program.over(retained), spread=3.0),
            samples=4,
            generator=seeded(fold.index),
        )

        assert both.rows == len(fold.held_out)
        assert_proper(both.marginal, both.rows)
        assert_proper(both.plug_in, both.rows)
        assert not torch.allclose(both.marginal, both.plug_in)


def test_fold_predictions_are_read_by_policy_and_immutable() -> None:
    marginal = torch.full((3, LEVELS), 1.0 / LEVELS, dtype=torch.float64)
    plug_in = torch.eye(LEVELS, dtype=torch.float64)[:3]
    both = FoldPredictions(marginal, plug_in)

    assert both.under(HeldOutParticipantPolicy.MARGINAL) is marginal
    assert both.under(HeldOutParticipantPolicy.PLUG_IN) is plug_in
    assert (both.rows, both.levels) == (3, LEVELS)
    assert repr(both) == f"FoldPredictions(rows=3, levels={LEVELS})"

    with pytest.raises(AttributeError, match="immutable"):
        both.__setattr__("_marginal", plug_in)

    with pytest.raises(AttributeError, match="immutable"):
        both.__delattr__("_plug_in")

    with pytest.raises(ValueError, match="shapes"):
        FoldPredictions(marginal, plug_in[:2])

    with pytest.raises(ValueError, match="shapes"):
        FoldPredictions(marginal[0], plug_in[0])


def test_predictive_log_density_reads_the_recorded_level() -> None:
    probabilities = torch.tensor(
        [[0.5, 0.25, 0.25], [0.1, 0.2, 0.7]], dtype=torch.float64
    )
    responses = torch.tensor([1, 2])

    assert torch.allclose(
        predictive_log_density(probabilities, responses),
        torch.tensor([0.25, 0.7], dtype=torch.float64).log(),
    )


def test_predictive_density_row_reduces_and_names_held_out_rows() -> None:
    by_fit = {
        "p_gram_only": torch.tensor([-2.0, -3.0, -7.0], dtype=torch.float64),
        "morcela_by_frame": torch.tensor([-1.0, -2.0, -6.0], dtype=torch.float64),
        "morcela_by_verb_by_frame": torch.tensor(
            [-0.5, -1.5, -4.0], dtype=torch.float64
        ),
    }
    rows = tuple(
        predictive_density_row(name, densities) for name, densities in by_fit.items()
    )

    assert all(isinstance(row, PredictiveDensityRow) for row in rows)
    assert tuple(row.fit for row in rows) == tuple(by_fit)
    assert tuple(row.predictive_density.value for row in rows) == pytest.approx(
        tuple(float(densities.mean()) for densities in by_fit.values())
    )
    assert all(row.predictive_density.n == 3 for row in rows)
    assert all(row.predictive_density.unit == "response" for row in rows)


@pytest.mark.parametrize(
    "densities",
    [
        torch.empty(0, dtype=torch.float64),
        torch.ones((2, 2), dtype=torch.float64),
        torch.tensor([float("nan")], dtype=torch.float64),
    ],
)
def test_predictive_density_row_refuses_invalid_inputs(densities: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="predictive densities"):
        predictive_density_row("morcela_by_frame", densities)


def test_predictive_density_row_refuses_nonfloating_values() -> None:
    with pytest.raises(TypeError, match="must be floating point"):
        predictive_density_row("morcela_by_frame", torch.tensor([-2, -1]))


def test_posterior_variance_row_reduces_draws_with_an_equal_tail_interval() -> None:
    draws = torch.tensor([0.0, 0.5, 1.0, 2.0, 4.0], dtype=torch.float64)
    level = 0.8
    tail = (1.0 - level) / 2.0
    expected = torch.quantile(
        draws, torch.tensor([tail, 1.0 - tail], dtype=draws.dtype)
    )
    row = posterior_variance_row("morcela_by_frame", "frame", draws, level=level)

    assert isinstance(row, PosteriorVarianceRow)
    assert (row.fit, row.group) == ("morcela_by_frame", "frame")
    assert row.posterior_variance.value == pytest.approx(float(draws.mean()))
    assert row.posterior_variance.lower == pytest.approx(float(expected[0]))
    assert row.posterior_variance.upper == pytest.approx(float(expected[1]))
    assert row.posterior_variance.level == level
    assert row.posterior_variance.n == draws.numel()
    assert row.posterior_variance.unit == "posterior_draw"


@pytest.mark.parametrize(
    ("draws", "level", "message"),
    [
        (torch.empty(0, dtype=torch.float64), 0.95, "nonempty vector"),
        (torch.ones((2, 2), dtype=torch.float64), 0.95, "nonempty vector"),
        (torch.tensor([float("inf")]), 0.95, "finite"),
        (torch.tensor([-0.1, 0.2]), 0.95, "nonnegative"),
        (torch.tensor([0.1, 0.2]), 1.0, "proportion"),
    ],
)
def test_posterior_variance_row_refuses_invalid_inputs(
    draws: torch.Tensor, level: float, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        posterior_variance_row("morcela_by_frame", "frame", draws, level=level)


def test_posterior_variance_row_refuses_nonfloating_values() -> None:
    with pytest.raises(TypeError, match="must be floating point"):
        posterior_variance_row("morcela_by_frame", "frame", torch.tensor([0, 1]))


def test_report_rows_refuse_unnamed_or_undefined_results() -> None:
    density = Measurement(value=-1.0, n=2, unit="response")
    interval = Measurement(
        value=0.2,
        n=10,
        unit="posterior_draw",
        lower=0.1,
        upper=0.3,
        level=0.8,
    )

    with pytest.raises(dx.ValidationError, match="fit"):
        PredictiveDensityRow(fit="", predictive_density=density)

    with pytest.raises(dx.ValidationError, match="no predictive density"):
        PredictiveDensityRow(
            fit="morcela_by_frame",
            predictive_density=Measurement(value=None, n=2, unit="response"),
        )

    with pytest.raises(dx.ValidationError, match="group"):
        PosteriorVarianceRow(
            fit="morcela_by_frame", group="", posterior_variance=interval
        )

    with pytest.raises(dx.ValidationError, match="posterior interval"):
        PosteriorVarianceRow(
            fit="morcela_by_frame",
            group="frame",
            posterior_variance=Measurement(value=0.2, n=10, unit="posterior_draw"),
        )


def likelihood(
    value: float | None, *, n: int = 200, unit: str = "response"
) -> Measurement:
    return Measurement(value=value, n=n, unit=unit)


def test_every_pseudo_r_squared_row_names_its_reference() -> None:
    fitted = likelihood(-0.6)
    references = MappingProxyType(
        {
            "intercept_only": likelihood(-1.0),
            "frame_only": likelihood(-0.8),
        }
    )
    rows = tuple(
        pseudo_r_squared("morcela_only", fitted, reference=name, references=references)
        for name in references
    )

    assert all(isinstance(row, PseudoRSquaredRow) for row in rows)
    assert tuple(row.fit for row in rows) == ("morcela_only", "morcela_only")
    assert tuple(row.reference for row in rows) == ("intercept_only", "frame_only")
    assert rows[0].pseudo_r_squared.value == pytest.approx(0.4, abs=1e-12)
    assert rows[1].pseudo_r_squared.value == pytest.approx(0.25, abs=1e-12)
    assert rows[0].pseudo_r_squared.value != rows[1].pseudo_r_squared.value
    assert all(row.pseudo_r_squared.n == 200 for row in rows)
    assert all(row.pseudo_r_squared.unit == "response" for row in rows)


def test_a_pseudo_r_squared_row_requires_a_named_fit_and_reference() -> None:
    measured = likelihood(0.4)

    with pytest.raises(dx.ValidationError, match="fit"):
        PseudoRSquaredRow(fit="", reference="intercept_only", pseudo_r_squared=measured)

    with pytest.raises(dx.ValidationError, match="reference"):
        PseudoRSquaredRow(fit="morcela_only", reference="", pseudo_r_squared=measured)


@pytest.mark.parametrize(
    ("reference", "message"),
    [
        (Measurement(value=-1.0, n=199, unit="response"), "same units"),
        (Measurement(value=-1.0, n=200, unit="item"), "same units"),
        (Measurement(value=0.0, n=200, unit="response"), "zero"),
        (Measurement(value=None, n=200, unit="response"), "undefined"),
    ],
)
def test_pseudo_r_squared_refuses_an_incomparable_reference(
    reference: Measurement, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        pseudo_r_squared(
            "morcela_only",
            likelihood(-0.6),
            reference="intercept_only",
            references=MappingProxyType({"intercept_only": reference}),
        )


def test_pseudo_r_squared_refuses_a_reference_the_mapping_does_not_name() -> None:
    with pytest.raises(ValueError, match="missing"):
        pseudo_r_squared(
            "morcela_only",
            likelihood(-0.6),
            reference="missing",
            references=MappingProxyType({"intercept_only": likelihood(-1.0)}),
        )


def test_pseudo_r_squared_refuses_an_undefined_fitted_likelihood() -> None:
    with pytest.raises(ValueError, match="undefined"):
        pseudo_r_squared(
            "morcela_only",
            likelihood(None),
            reference="intercept_only",
            references=MappingProxyType({"intercept_only": likelihood(-1.0)}),
        )


def test_a_pseudo_r_squared_row_round_trips_through_its_wire_form() -> None:
    row = pseudo_r_squared(
        "morcela_only",
        likelihood(-0.6),
        reference="frame_only",
        references=MappingProxyType({"frame_only": likelihood(-0.8)}),
    )

    assert PseudoRSquaredRow.model_validate_json(row.model_dump_json()) == row


def test_no_signature_in_the_package_defaults_the_policy() -> None:
    package = Path(lofi.judgment.__file__).parent
    defaulted = []

    for source in sorted(package.glob("*.py")):
        text = source.read_text(encoding="utf-8")

        for node in ast.walk(ast.parse(text)):
            if not isinstance(node, ast.FunctionDef):
                continue

            arguments = node.args
            positional = arguments.posonlyargs + arguments.args
            defaults = [None] * (len(positional) - len(arguments.defaults))
            defaults += arguments.defaults
            pairs = [
                *zip(positional, defaults, strict=True),
                *zip(arguments.kwonlyargs, arguments.kw_defaults, strict=True),
            ]

            for argument, default in pairs:
                annotation = (
                    ""
                    if argument.annotation is None
                    else ast.get_source_segment(text, argument.annotation) or ""
                )
                selected = default is not None and not (
                    isinstance(default, ast.Constant) and default.value is None
                )

                if "HeldOutParticipantPolicy" in annotation and selected:
                    defaulted.append(
                        f"{source.name}:{node.name}({argument.arg}="
                        f"{ast.get_source_segment(text, default)})"
                    )

    assert defaulted == []
