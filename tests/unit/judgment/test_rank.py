import math
from pathlib import Path
from typing import Final, cast

import pytest
import torch

from lofi.judgment import (
    ConditionalLikelihood,
    DesignCodec,
    IdentityLinkOnLogit,
    JudgmentDesign,
    JudgmentRow,
    LinkingSpec,
    LinkTerm,
    Predictor,
    PriorBlock,
    ResponseSpec,
    RowColumns,
    latent_contribution,
)
from lofi.judgment._spec import RankConsistent, RankConsistentScheme
from lofi.judgment.mass import LogitLinking
from lofi.judgment.rank import RankConsistentHead, RankConsistentParameters

READOUT: Final = "p_gram"
COMPLEMENT: Final = "p_gram_complement"
LEVELS: Final = 7
GENERATED_ROWS: Final = 100_000
LADDER: Final = (
    Path(__file__).resolve().parents[3] / "tests" / "data" / "judgment" / "ladder"
)
TERM: Final = LinkTerm(
    readout=READOUT,
    linking=LinkingSpec(family="logit", columns={"complement": COMPLEMENT}),
)


class ReadoutPredictor:
    """An evaluator applying one declared unit-slope logit link."""

    def __init__(self, term: LinkTerm = TERM) -> None:
        self._terms = (term,)
        self._linking = LogitLinking(term, unit_slope=True)

    @property
    def terms(self) -> tuple[Predictor, ...]:
        return self._terms

    @property
    def linking(self) -> LogitLinking:
        """Return the unit-slope linking that evaluates the term."""
        return self._linking

    def latent_scores(self, design: JudgmentDesign, /) -> torch.Tensor:
        parameters = design.readouts[READOUT].new_empty(0)

        return latent_contribution(self._linking, design, parameters)


def spec(scheme: RankConsistentScheme) -> ResponseSpec:
    return ResponseSpec(
        family=RankConsistent(scheme=scheme),
        levels=LEVELS,
        objective=ConditionalLikelihood(),
        predictor=(TERM,),
        identification=IdentityLinkOnLogit(),
        priors=PriorBlock(),
    )


def head(scheme: RankConsistentScheme) -> RankConsistentHead:
    return RankConsistentHead(spec(scheme), ReadoutPredictor())


def _log_sigmoid(value: float) -> float:
    """Return a scalar log sigmoid without overflowing.

    Parameters
    ----------
    value : float
        Log odds to transform.
    """
    return -max(-value, 0.0) - math.log1p(math.exp(-abs(value)))


def design(scores: tuple[float, ...], responses: tuple[int, ...]) -> JudgmentDesign:
    return DesignCodec(LEVELS, readouts=(READOUT, COMPLEMENT)).encode(
        JudgmentRow(
            "annotator",
            f"item-{position}",
            response,
            RowColumns(
                {},
                {
                    READOUT: _log_sigmoid(score),
                    COMPLEMENT: _log_sigmoid(-score),
                },
            ),
        )
        for position, (score, response) in enumerate(
            zip(scores, responses, strict=True)
        )
    )


@pytest.fixture(scope="module")
def generated_design() -> JudgmentDesign:
    return DesignCodec(LEVELS, readouts=(READOUT, COMPLEMENT)).encode(
        JudgmentRow(
            "annotator",
            f"item-{position % 11}",
            position % LEVELS,
            RowColumns(
                {},
                {
                    READOUT: _log_sigmoid(((position % 2001) - 1000) / 100.0),
                    COMPLEMENT: _log_sigmoid(-((position % 2001) - 1000) / 100.0),
                },
            ),
        )
        for position in range(GENERATED_ROWS)
    )


def parameters(
    scheme: RankConsistentScheme,
    *,
    requires_grad: bool = False,
) -> RankConsistentParameters:
    weights = (
        (0.75,)
        if scheme is RankConsistentScheme.CORAL
        else (0.3, -0.5, 0.7, 0.2, -0.4, 0.6)
    )

    return RankConsistentParameters(
        torch.tensor(weights, dtype=torch.float64, requires_grad=requires_grad),
        torch.tensor(
            (3.0, 2.0, 1.0, 0.0, -1.0, -2.0),
            dtype=torch.float64,
            requires_grad=requires_grad,
        ),
    )


def test_the_head_reports_its_levels_scheme_and_predictor() -> None:
    subject = head(RankConsistentScheme.CORAL)

    assert subject.levels == LEVELS
    assert subject.task_count == LEVELS - 1
    assert subject.scheme is RankConsistentScheme.CORAL
    assert subject.predictor.terms == (TERM,)
    assert repr(subject) == "RankConsistentHead(levels=7, scheme='coral')"


def test_the_declared_logit_is_evaluated_at_unit_slope_without_parameters() -> None:
    source = design((-2.0, 0.0, 3.5), (0, 3, 6))
    predictor = ReadoutPredictor()

    assert predictor.linking.domain.parameter_names == ()
    assert dict(predictor.linking.pinned) == {"slope": 1.0}
    assert torch.allclose(
        predictor.latent_scores(source),
        torch.tensor((-2.0, 0.0, 3.5), dtype=torch.float64),
        atol=1e-15,
        rtol=0.0,
    )


@pytest.mark.parametrize("scheme", tuple(RankConsistentScheme))
def test_probabilities_are_proper_and_exceedance_is_nonincreasing_over_generated_rows(
    scheme: RankConsistentScheme,
    generated_design: JudgmentDesign,
) -> None:
    subject = head(scheme)
    given = parameters(scheme)
    exceedance = subject.exceedance_probabilities(generated_design, given)
    probabilities = subject.probabilities(generated_design, given)

    assert exceedance.shape == (GENERATED_ROWS, LEVELS - 1)
    assert probabilities.shape == (GENERATED_ROWS, LEVELS)
    assert bool(torch.isfinite(exceedance).all())
    assert bool(torch.isfinite(probabilities).all())
    assert bool((exceedance[:, :-1] >= exceedance[:, 1:]).all())
    assert bool((probabilities >= 0.0).all())
    assert torch.allclose(
        probabilities.sum(dim=1),
        torch.ones(GENERATED_ROWS, dtype=torch.float64),
        atol=1e-12,
        rtol=0.0,
    )


def test_corn_exceedance_is_the_exact_conditional_probability_chain() -> None:
    source = design((-2.0, -0.25, 1.5), (0, 3, 6))
    given = parameters(RankConsistentScheme.CORN)
    latent = ReadoutPredictor().latent_scores(source)
    conditionals = torch.sigmoid(latent.unsqueeze(1) * given.weights + given.biases)
    expected = torch.cumprod(conditionals, dim=1)

    assert torch.allclose(
        head(RankConsistentScheme.CORN).exceedance_probabilities(source, given),
        expected,
        atol=0.0,
        rtol=1e-15,
    )


def test_category_probabilities_are_the_ordered_exceedance_differences() -> None:
    source = design((-1.0, 0.5, 2.0), (1, 3, 5))

    for scheme in RankConsistentScheme:
        subject = head(scheme)
        given = parameters(scheme)
        exceedance = subject.exceedance_probabilities(source, given)
        expected = torch.cat(
            (
                1.0 - exceedance[:, :1],
                exceedance[:, :-1] - exceedance[:, 1:],
                exceedance[:, -1:],
            ),
            dim=1,
        )

        assert torch.allclose(
            subject.probabilities(source, given), expected, atol=1e-15, rtol=1e-15
        )


def test_coral_uses_one_shared_weight_and_corn_uses_one_per_task() -> None:
    biases = torch.tensor((3.0, 2.0, 1.0, 0.0, -1.0, -2.0), dtype=torch.float64)
    shared = RankConsistentParameters(torch.tensor((0.5,), dtype=torch.float64), biases)
    independent = RankConsistentParameters(
        torch.ones(LEVELS - 1, dtype=torch.float64), biases
    )
    source = design((0.0,), (0,))

    head(RankConsistentScheme.CORAL).probabilities(source, shared)
    head(RankConsistentScheme.CORN).probabilities(source, independent)

    with pytest.raises(ValueError, match=r"coral.*needs 1 weights"):
        head(RankConsistentScheme.CORAL).probabilities(source, independent)

    with pytest.raises(ValueError, match=r"corn.*needs 6 weights"):
        head(RankConsistentScheme.CORN).probabilities(source, shared)


def test_coral_refuses_biases_that_do_not_order_the_direct_tasks() -> None:
    source = design((0.0,), (0,))
    reversed_biases = RankConsistentParameters(
        torch.tensor((1.0,), dtype=torch.float64),
        torch.arange(LEVELS - 1, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match="strictly decreasing"):
        head(RankConsistentScheme.CORAL).probabilities(source, reversed_biases)


@pytest.mark.parametrize("scheme", tuple(RankConsistentScheme))
def test_extreme_log_probabilities_are_finite(
    scheme: RankConsistentScheme,
) -> None:
    source = design((-10_000.0, 10_000.0), (0, LEVELS - 1))
    weights = (
        torch.tensor((100.0,), dtype=torch.float64)
        if scheme is RankConsistentScheme.CORAL
        else torch.tensor((100.0, -90.0, 80.0, -70.0, 60.0, -50.0), dtype=torch.float64)
    )
    given = RankConsistentParameters(
        weights,
        torch.tensor((300.0, 200.0, 100.0, 0.0, -100.0, -200.0), dtype=torch.float64),
    )
    log_probabilities = head(scheme).log_probabilities(source, given)

    assert bool(torch.isfinite(log_probabilities).all())
    assert log_probabilities.shape == (2, LEVELS)


@pytest.mark.parametrize("scheme", tuple(RankConsistentScheme))
def test_likelihood_gradients_reach_every_declared_parameter(
    scheme: RankConsistentScheme,
) -> None:
    source = design(
        (-2.5, -1.25, -0.5, 0.25, 1.0, 1.75, 2.5),
        (0, 1, 2, 3, 4, 5, 6),
    )
    given = parameters(scheme, requires_grad=True)
    loss = -head(scheme).log_likelihood(source, given).sum()

    loss.backward()

    assert given.weights.grad is not None
    assert given.biases.grad is not None
    assert bool(torch.isfinite(given.weights.grad).all())
    assert bool(torch.isfinite(given.biases.grad).all())
    assert bool((given.weights.grad != 0.0).all())
    assert bool((given.biases.grad != 0.0).all())


@pytest.mark.parametrize(
    ("fixture", "scheme"),
    [
        ("p_gram_rank_consistent", RankConsistentScheme.CORAL),
        ("p_gram_corn", RankConsistentScheme.CORN),
    ],
)
def test_both_rank_consistent_fixtures_construct_the_ordinary_head(
    fixture: str,
    scheme: RankConsistentScheme,
) -> None:
    declared = ResponseSpec.model_validate_json(
        (LADDER / f"{fixture}.json").read_text(encoding="utf-8")
    )
    term = cast("LinkTerm", declared.predictor[0])
    subject = RankConsistentHead(declared, ReadoutPredictor(term))

    assert subject.scheme is scheme
    assert subject.predictor.terms == declared.predictor
