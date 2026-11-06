"""Tests of the metric head, including the sign-inversion case.

The constructed data set follows the demonstration in Liddell and Kruschke
(2018), "Analyzing ordinal data with metric models: What could possibly go
wrong?", Journal of Experimental Social Psychology 79, 328-348.
"""

import json
import math
import random
from pathlib import Path
from typing import Final, Literal, override

import pytest
import torch

from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    CumulativeLinkHead,
    CumulativeParameters,
    DesignCodec,
    FixedDiscrimination,
    JudgmentDesign,
    JudgmentRow,
    Metric,
    MetricHead,
    MetricParameters,
    Predictor,
    PriorBlock,
    ResponseSpec,
    RowColumns,
    RowEffects,
    ScaleNormalization,
    ThresholdPriors,
    item_response_weights,
    normalized_responses,
    ridit_scores,
)

LADDER: Final = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "judgment" / "ladder"
)
GOLDEN: Final = (
    Path(__file__).resolve().parents[3]
    / "tests"
    / "data"
    / "judgment"
    / "ridit_golden.json"
)

READOUT = "p_gram"
LEVELS = 7
SPREAD = 0.4

FACTOR = "condition"
THRESHOLDS = (-2.0, -0.5, 1.5, 3.5, 5.0, 6.5)
CONDITIONS = (("consistent", 0.0, 1.5), ("variable", -2.0, 6.0))
ITEMS = 8
PARTICIPANTS = 20
STEPS = 600
RATE = 0.15


class Readout(Predictor, extra="forbid"):
    kind: Literal["metric_head_readout"] = "metric_head_readout"
    name: str


class Condition(Predictor, extra="forbid"):
    kind: Literal["metric_head_condition"] = "metric_head_condition"
    factor: str


class ReadoutPredictor:
    def __init__(self, name: str) -> None:
        self._terms = (Readout(name=name),)
        self._name = name

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return design.readouts[self._name]


class ShortPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return super().latent_scores(design)[:-1]


class SquarePredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        scores = super().latent_scores(design)

        return torch.stack((scores, scores), dim=1)


class CountingPredictor(ReadoutPredictor):
    @override
    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return super().latent_scores(design).to(torch.int64)


class DeclaredPredictor:
    """An evaluator implementing whatever terms a specification declares."""

    def __init__(self, terms: tuple[Predictor, ...]) -> None:
        self._terms = terms

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        return torch.zeros(design.n_rows, dtype=torch.float64)


class ConditionPredictor:
    """One fitted coefficient per condition, in the order the case names them."""

    def __init__(self, terms: tuple[Predictor, ...], names: tuple[str, ...]) -> None:
        self._terms = terms
        self._names = names
        self.coefficients = torch.zeros(
            len(names), dtype=torch.float64, requires_grad=True
        )

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    def coefficient(self, name: str) -> float:
        return float(self.coefficients.detach()[self._names.index(name)])

    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        order = torch.tensor(
            [self._names.index(level) for level in design.factor_levels[FACTOR]],
            dtype=torch.int64,
        )

        return self.coefficients[order[design.factors[FACTOR]]]


def spec(
    *,
    levels: int = LEVELS,
    normalization: ScaleNormalization = ScaleNormalization.IDENTITY,
) -> ResponseSpec:
    return ResponseSpec(
        family=Metric(),
        levels=levels,
        objective=ConditionalLikelihood(),
        predictor=(Readout(name=READOUT),),
        identification=FixedDiscrimination(value=1.0),
        normalization=normalization,
    )


def head(declared: ResponseSpec) -> MetricHead:
    return MetricHead(declared, ReadoutPredictor(READOUT))


def parameters(spread: float = SPREAD) -> MetricParameters:
    return MetricParameters(torch.tensor(spread, dtype=torch.float64))


def readout_design(
    judgments: tuple[tuple[str, int, float], ...],
    *,
    levels: int = LEVELS,
) -> JudgmentDesign:
    codec = DesignCodec(levels, readouts=(READOUT,))

    return codec.encode(
        JudgmentRow(
            participant,
            f"item{position}",
            level,
            RowColumns({}, {READOUT: score}),
        )
        for position, (participant, level, score) in enumerate(judgments)
    )


def condition_spec(family: Cumulative | Metric, priors: PriorBlock) -> ResponseSpec:
    return ResponseSpec(
        family=family,
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(Condition(factor=FACTOR),),
        identification=FixedDiscrimination(value=1.0),
        priors=priors,
    )


def level_counts(location: float, spread: float, total: int) -> tuple[int, ...]:
    """Return how many of a condition's responses fall at each level.

    The generating model is a cumulative logit at the case's thresholds, and
    the counts are its probabilities allocated by largest remainder, so the
    data set is fixed by the location and the spread alone.
    """
    cumulative = [
        1.0 / (1.0 + math.exp(-(threshold - location) / spread))
        for threshold in THRESHOLDS
    ]
    edges = (0.0, *cumulative, 1.0)
    exact = [total * (edges[level + 1] - edges[level]) for level in range(LEVELS)]
    counts = [int(value) for value in exact]
    remainders = sorted(
        range(LEVELS), key=lambda level: exact[level] - counts[level], reverse=True
    )

    for level in remainders[: total - sum(counts)]:
        counts[level] += 1

    return tuple(counts)


def constructed_design() -> JudgmentDesign:
    """Return the case: two conditions differing in location and in spread.

    Every participant judges every item, so each condition's responses are
    dealt one to each participant-by-item cell, and the items of a condition
    receive nearly the same distribution as one another.
    """
    rows: list[JudgmentRow] = []

    for name, location, spread in CONDITIONS:
        counts = level_counts(location, spread, ITEMS * PARTICIPANTS)
        levels = [level for level, count in enumerate(counts) for _ in range(count)]

        rows.extend(
            JudgmentRow(
                f"participant{(position // ITEMS) % PARTICIPANTS}",
                f"{name}-item{position % ITEMS}",
                level,
                RowColumns({FACTOR: name}, {}),
            )
            for position, level in enumerate(levels)
        )

    return DesignCodec(LEVELS, factors=(FACTOR,)).encode(rows)


def names_of() -> tuple[str, ...]:
    return tuple(name for name, _, _ in CONDITIONS)


def fit_cumulative(design: JudgmentDesign) -> ConditionPredictor:
    declared = condition_spec(Cumulative(), PriorBlock(thresholds=ThresholdPriors()))
    predictor = ConditionPredictor(declared.predictor, names_of())
    subject = CumulativeLinkHead(declared, predictor)
    base = torch.tensor(-3.0, dtype=torch.float64, requires_grad=True)
    spans = torch.zeros(LEVELS - 1, dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([predictor.coefficients, base, spans], lr=RATE)

    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = -subject.log_likelihood(design, CumulativeParameters(base, spans)).mean()
        loss.backward()
        optimizer.step()

    return predictor


def fit_metric(design: JudgmentDesign) -> ConditionPredictor:
    declared = condition_spec(Metric(), PriorBlock())
    predictor = ConditionPredictor(declared.predictor, names_of())
    subject = MetricHead(declared, predictor)
    spread = torch.zeros((), dtype=torch.float64, requires_grad=True)
    optimizer = torch.optim.Adam([predictor.coefficients, spread], lr=RATE)

    for _ in range(STEPS):
        optimizer.zero_grad()
        loss = -subject.log_likelihood(design, MetricParameters(spread)).mean()
        loss.backward()
        optimizer.step()

    return predictor


def condition_rows(design: JudgmentDesign, name: str) -> torch.Tensor:
    return design.factors[FACTOR] == design.factor_levels[FACTOR].index(name)


def item_means(design: JudgmentDesign, name: str) -> torch.Tensor:
    mask = condition_rows(design, name)
    items = design.items[mask]
    responses = design.responses[mask].to(dtype=torch.float64)

    return torch.stack(
        [responses[items == item].mean() for item in items.unique()],
    )


def test_the_head_reports_the_declared_levels_and_normalization() -> None:
    subject = head(spec(normalization=ScaleNormalization.RIDIT))

    assert subject.levels == LEVELS
    assert subject.normalization is ScaleNormalization.RIDIT
    assert repr(subject) == "MetricHead(levels=7, normalization='ridit')"


def test_the_head_exposes_the_evaluator_of_the_declared_terms() -> None:
    predictor = ReadoutPredictor(READOUT)

    assert MetricHead(spec(), predictor).predictor is predictor


def test_a_head_refuses_a_specification_of_another_family() -> None:
    declared = ResponseSpec(
        family=Cumulative(),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(Readout(name=READOUT),),
        identification=FixedDiscrimination(value=1.0),
        priors=PriorBlock(thresholds=ThresholdPriors()),
    )

    with pytest.raises(TypeError, match="metric family alone"):
        MetricHead(declared, ReadoutPredictor(READOUT))


def test_a_head_refuses_an_evaluator_of_undeclared_terms() -> None:
    with pytest.raises(ValueError, match="against the declared terms"):
        MetricHead(spec(), ReadoutPredictor("length"))


def test_a_head_refuses_the_normalization_it_has_no_ratio_scale_for() -> None:
    declared = spec(normalization=ScaleNormalization.LOG)

    with pytest.raises(ValueError, match="logarithm"):
        MetricHead(declared, ReadoutPredictor(READOUT))


@pytest.mark.parametrize(
    "normalization",
    [
        ScaleNormalization.IDENTITY,
        ScaleNormalization.RIDIT,
        ScaleNormalization.UNIT_INTERVAL,
    ],
)
def test_a_head_accepts_every_normalization_it_applies(
    normalization: ScaleNormalization,
) -> None:
    assert head(spec(normalization=normalization)).normalization is normalization


def test_the_identity_observes_the_recorded_level_itself() -> None:
    design = readout_design((("a", 0, 0.5), ("a", 3, 0.5), ("a", 6, 0.5)))

    assert torch.equal(
        head(spec()).observations(design),
        torch.tensor([0.0, 3.0, 6.0], dtype=torch.float64),
    )


def test_unit_interval_scaling_divides_the_level_by_the_highest() -> None:
    design = readout_design((("a", 0, 0.5), ("a", 3, 0.5), ("a", 6, 0.5)))
    subject = head(spec(normalization=ScaleNormalization.UNIT_INTERVAL))

    assert torch.allclose(
        subject.observations(design),
        torch.tensor([0.0, 0.5, 1.0], dtype=torch.float64),
    )


def test_ridit_scaling_gives_a_level_the_midpoint_of_its_own_interval() -> None:
    design = readout_design(
        (("a", 0, 0.5), ("a", 0, 0.5), ("a", 1, 0.5), ("a", 3, 0.5))
    )

    assert torch.allclose(
        ridit_scores(design),
        torch.tensor([0.25, 0.25, 0.625, 0.875], dtype=torch.float64),
    )


def test_ridit_scaling_reads_each_participant_against_their_own_responses() -> None:
    design = readout_design(
        (
            ("low", 0, 0.5),
            ("low", 1, 0.5),
            ("low", 2, 0.5),
            ("high", 2, 0.5),
            ("high", 5, 0.5),
            ("high", 6, 0.5),
        )
    )
    scores = ridit_scores(design)

    assert scores[2] == pytest.approx(5.0 / 6.0)
    assert scores[3] == pytest.approx(1.0 / 6.0)


def test_ridit_scaling_puts_an_unvarying_participant_at_the_midpoint() -> None:
    design = readout_design((("a", 4, 0.5), ("a", 4, 0.5), ("a", 4, 0.5)))

    assert torch.allclose(
        ridit_scores(design), torch.full((3,), 0.5, dtype=torch.float64)
    )


def test_the_logarithm_is_refused_as_a_normalization_of_a_level() -> None:
    design = readout_design((("a", 0, 0.5), ("a", 3, 0.5)))

    with pytest.raises(ValueError, match="ratio-scale"):
        normalized_responses(design, ScaleNormalization.LOG)


def item_weight_design() -> JudgmentDesign:
    codec = DesignCodec(LEVELS, readouts=(READOUT,))

    return codec.encode(
        (
            JudgmentRow("a", "first", 0, RowColumns({}, {READOUT: 0.0})),
            JudgmentRow("a", "second", 3, RowColumns({}, {READOUT: 0.0})),
            JudgmentRow("b", "first", 0, RowColumns({}, {READOUT: 0.0})),
            JudgmentRow("b", "second", 6, RowColumns({}, {READOUT: 0.0})),
        )
    )


def test_the_identity_acceptability_channel_is_one_per_item() -> None:
    design = item_weight_design()

    assert design.item_levels == ("first", "second")
    assert torch.equal(
        item_response_weights(design),
        torch.ones(2, dtype=torch.float64),
    )


@pytest.mark.parametrize(
    ("normalization", "expected"),
    [
        (ScaleNormalization.RIDIT, (0.25, 0.75)),
        (ScaleNormalization.UNIT_INTERVAL, (0.0, 0.75)),
    ],
)
def test_item_response_weights_average_scaled_rows_by_item(
    normalization: ScaleNormalization,
    expected: tuple[float, float],
) -> None:
    design = item_weight_design()

    assert torch.allclose(
        item_response_weights(design, normalization),
        torch.tensor(expected, dtype=torch.float64),
    )


def test_item_response_weights_refuse_logarithmic_scaling() -> None:
    with pytest.raises(ValueError, match="ratio-scale"):
        item_response_weights(item_weight_design(), ScaleNormalization.LOG)


def test_the_head_refuses_a_design_over_another_number_of_levels() -> None:
    design = readout_design((("a", 0, 0.5), ("a", 3, 0.5)), levels=LEVELS + 1)

    with pytest.raises(ValueError, match="levels was given a design over"):
        head(spec()).observations(design)


def test_the_likelihood_is_the_gaussian_density_of_the_normalized_response() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25), ("a", 6, 2.0)))
    subject = head(spec())
    given = parameters()
    expected = torch.distributions.Normal(
        design.readouts[READOUT], subject.spread(given)
    ).log_prob(subject.observations(design))

    assert torch.allclose(subject.log_likelihood(design, given), expected)


def test_the_likelihood_reads_the_response_through_the_declared_normalization() -> None:
    design = readout_design((("a", 0, 0.5), ("a", 1, 0.5), ("a", 6, 0.5)))
    given = parameters()
    identity = head(spec()).log_likelihood(design, given)
    ridit = head(spec(normalization=ScaleNormalization.RIDIT)).log_likelihood(
        design, given
    )

    assert not torch.allclose(identity, ridit)


def test_the_spread_stays_positive_for_any_unconstrained_value() -> None:
    subject = head(spec())
    values = torch.tensor([-40.0, -1.0, 0.0, 30.0], dtype=torch.float64)
    spreads = torch.stack([subject.spread(MetricParameters(v)) for v in values])

    assert bool((spreads > 0.0).all())
    assert torch.allclose(spreads, torch.nn.functional.softplus(values))


def test_a_wider_spread_lowers_the_density_of_a_well_predicted_response() -> None:
    design = readout_design((("a", 3, 3.0),))
    subject = head(spec())
    narrow = subject.log_likelihood(design, parameters(0.1))
    wide = subject.log_likelihood(design, parameters(4.0))

    assert float(narrow) > float(wide)


def test_the_prediction_is_the_latent_score_itself() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25)))

    assert torch.equal(head(spec()).predictions(design), design.readouts[READOUT])


def test_a_latent_offset_shifts_every_prediction_it_is_given() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25)))
    offsets = torch.tensor([0.5, -1.5], dtype=torch.float64)
    shifted = head(spec()).predictions(design, effects=RowEffects(offsets))

    assert torch.allclose(shifted, design.readouts[READOUT] + offsets)


def test_the_head_refuses_effects_that_offset_cutpoint_spans() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25)))
    effects = RowEffects(
        torch.zeros(2, dtype=torch.float64),
        torch.zeros(2, LEVELS - 1, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="no cutpoints"):
        head(spec()).predictions(design, effects=effects)


def test_the_head_refuses_effects_over_another_number_of_rows() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25)))
    effects = RowEffects(torch.zeros(3, dtype=torch.float64))

    with pytest.raises(ValueError, match="effects over 3 rows"):
        head(spec()).predictions(design, effects=effects)


def test_the_head_refuses_an_evaluator_that_scores_another_design() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25)))
    subject = MetricHead(spec(), ShortPredictor(READOUT))

    with pytest.raises(ValueError, match="was scored with"):
        subject.latent_scores(design)


def test_the_head_refuses_a_latent_score_that_is_not_one_per_row() -> None:
    design = readout_design((("a", 1, -0.5), ("a", 4, 0.25)))
    subject = MetricHead(spec(), SquarePredictor(READOUT))

    with pytest.raises(ValueError, match="rank 2, expected 1"):
        subject.latent_scores(design)


def test_the_head_refuses_a_latent_score_of_a_nonfloating_kind() -> None:
    design = readout_design((("a", 1, -1.0), ("a", 4, 2.0)))
    subject = MetricHead(spec(), CountingPredictor(READOUT))

    with pytest.raises(TypeError, match="not a floating-point kind"):
        subject.latent_scores(design)


def test_parameters_refuse_a_spread_that_is_not_a_scalar() -> None:
    with pytest.raises(ValueError, match="rank 1, expected 0"):
        MetricParameters(torch.zeros(1, dtype=torch.float64))


def test_parameters_refuse_a_spread_that_is_not_floating_point() -> None:
    with pytest.raises(TypeError, match="not a floating-point kind"):
        MetricParameters(torch.zeros((), dtype=torch.int64))


def test_parameters_refuse_reassignment_and_deletion() -> None:
    given = parameters()

    with pytest.raises(AttributeError, match="parameters are immutable"):
        given.__setattr__("_spread", torch.zeros((), dtype=torch.float64))

    with pytest.raises(AttributeError, match="parameters are immutable"):
        given.__delattr__("_spread")


def test_the_declared_metric_ablation_builds_this_head() -> None:
    declared = ResponseSpec.model_validate_json(
        (LADDER / "p_gram_metric.json").read_text(encoding="utf-8")
    )
    subject = MetricHead(declared, DeclaredPredictor(declared.predictor))

    assert isinstance(declared.family, Metric)
    assert subject.levels == declared.levels
    assert subject.normalization is declared.normalization


def test_the_case_generates_the_two_conditions_it_declares() -> None:
    design = constructed_design()

    assert design.n_rows == 2 * ITEMS * PARTICIPANTS
    assert len(design.item_levels) == 2 * ITEMS
    assert len(design.participant_levels) == PARTICIPANTS
    assert sorted(design.factor_levels[FACTOR]) == sorted(names_of())


def test_the_metric_head_reports_each_condition_s_mean_response() -> None:
    design = constructed_design()
    fitted = fit_metric(design)
    responses = design.responses.to(dtype=torch.float64)

    for name in names_of():
        mean = float(responses[condition_rows(design, name)].mean())

        assert fitted.coefficient(name) == pytest.approx(mean, abs=1e-6)


def test_the_metric_head_inverts_the_ordering_the_cumulative_head_recovers() -> None:
    design = constructed_design()
    metric = fit_metric(design)
    cumulative = fit_cumulative(design)
    higher, lower = names_of()

    assert CONDITIONS[0][1] > CONDITIONS[1][1]
    assert cumulative.coefficient(higher) > cumulative.coefficient(lower)
    assert metric.coefficient(higher) < metric.coefficient(lower)


def test_averaging_across_participants_per_item_keeps_the_inversion() -> None:
    design = constructed_design()
    higher, lower = names_of()

    assert float(item_means(design, higher).mean()) < float(
        item_means(design, lower).mean()
    )


def test_every_item_of_the_case_carries_every_participant_once() -> None:
    design = constructed_design()
    pairs = {
        (int(item), int(participant))
        for item, participant in zip(design.items, design.participants, strict=True)
    }

    assert len(pairs) == design.n_rows


type GoldenRow = tuple[str, int, float]


def golden_document() -> tuple[int, tuple[tuple[str, tuple[GoldenRow, ...]], ...]]:
    """Return the committed level count and each named case's rows."""
    document = json.loads(GOLDEN.read_text(encoding="utf-8"))
    cases = tuple(
        (
            str(case["name"]),
            tuple(
                (str(row["participant"]), int(row["response"]), float(row["ridit"]))
                for row in case["rows"]
            ),
        )
        for case in document["cases"]
    )

    return int(document["levels"]), cases


GOLDEN_LEVELS, GOLDEN_CASES = golden_document()


def empirical_midpoint(responses: tuple[int, ...], level: int) -> float:
    """Return the midpoint of the interval a level occupies in a distribution."""
    below = sum(1 for other in responses if other < level)
    at = sum(1 for other in responses if other == level)

    return (below + at / 2) / len(responses)


def scattered_judgments(seed: int) -> tuple[GoldenRow, ...]:
    """Return judgments whose participants and levels are interleaved at random."""
    source = random.Random(seed)
    participants = [f"participant{index}" for index in range(9)]

    return tuple(
        (source.choice(participants), source.randrange(LEVELS), 0.5) for _ in range(240)
    )


@pytest.mark.parametrize(
    "rows", [pytest.param(rows, id=name) for name, rows in GOLDEN_CASES]
)
def test_ridit_scaling_reproduces_the_golden_vector(
    rows: tuple[GoldenRow, ...],
) -> None:
    design = readout_design(
        tuple((participant, level, 0.5) for participant, level, _ in rows),
        levels=GOLDEN_LEVELS,
    )
    expected = torch.tensor([score for _, _, score in rows], dtype=torch.float64)

    assert torch.allclose(ridit_scores(design), expected, rtol=0.0, atol=1e-9)


def test_ridit_scaling_matches_the_definition_read_one_participant_at_a_time() -> None:
    judgments = scattered_judgments(7)
    own: dict[str, tuple[int, ...]] = {}

    for participant, level, _ in judgments:
        own[participant] = (*own.get(participant, ()), level)

    expected = torch.tensor(
        [
            empirical_midpoint(own[participant], level)
            for participant, level, _ in judgments
        ],
        dtype=torch.float64,
    )

    assert torch.allclose(
        ridit_scores(readout_design(judgments)), expected, rtol=0.0, atol=1e-9
    )
