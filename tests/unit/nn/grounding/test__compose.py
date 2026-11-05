from itertools import product
from math import inf, isfinite, log, nan
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.language import language

from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import _compose
from lofi.nn.grounding._compose import (
    Composed,
    PairTable,
    compose,
    pair_log_weights,
)
from lofi.nn.grounding._values import (
    Best,
    GradientPath,
    Marginal,
    Tempered,
)

if TYPE_CHECKING:
    from lofi.terms import Term

LANGUAGE: Final = language(("cold", "dry", "rain", "wet"))
STORE: Final = LANGUAGE.store
RAIN: Final = LANGUAGE.constant("rain")
COLD: Final = LANGUAGE.constant("cold")
WET: Final = LANGUAGE.constant("wet")
DRY: Final = LANGUAGE.constant("dry")


def _weights(*values: float) -> torch.Tensor:
    return torch.tensor(values, dtype=torch.float64).log().requires_grad_()


def _weighted(terms: tuple[Term, ...], weights: torch.Tensor) -> WeightedTerms:
    return WeightedTerms(terms, weights, torch.tensor(-torch.inf), Estimate.EXACT)


def _joint(*values: float) -> torch.Tensor:
    return _weights(*values)


def _table(joint: torch.Tensor, *values: float | None) -> PairTable:
    return PairTable.from_values(joint, values)


def _probability(composed: Composed) -> float:
    assert composed.log_probability is not None

    return float(composed.log_probability.detach().exp())


def test_pair_log_weights_run_over_the_premise_and_then_the_hypothesis() -> None:
    premise_weights = _weights(0.75, 0.25)
    hypothesis_weights = _weights(0.5, 0.5)
    premise = _weighted((RAIN, COLD), premise_weights)
    hypothesis = _weighted((WET, DRY), hypothesis_weights)
    joint = pair_log_weights(premise, hypothesis)
    pairs = list(product(premise.terms, hypothesis.terms))

    assert pairs == [(RAIN, WET), (RAIN, DRY), (COLD, WET), (COLD, DRY)]
    assert joint.detach().exp().tolist() == pytest.approx([0.375, 0.375, 0.125, 0.125])

    joint.sum().backward()

    assert premise_weights.grad is not None
    assert premise_weights.grad.tolist() == [2.0, 2.0]
    assert hypothesis_weights.grad is not None
    assert hypothesis_weights.grad.tolist() == [2.0, 2.0]


def test_a_table_holds_one_truth_value_and_one_decision_per_pair() -> None:
    joint = _joint(0.5, 0.3, 0.2)
    table = _table(joint, 1.0, None, 0.0)

    assert table.log_joint is joint
    assert table.truth.tolist() == [1.0, 0.0, 0.0]
    assert table.decided.tolist() == [True, False, True]
    assert len(table) == 3
    assert repr(table) == "PairTable(pairs=3, decided=2)"


def test_a_table_is_immutable() -> None:
    table = _table(_joint(1.0), 1.0)

    with pytest.raises(AttributeError, match="immutable"):
        table.__setattr__("_truth", table.truth)

    with pytest.raises(AttributeError, match="immutable"):
        table.__delattr__("_truth")


def test_a_table_refuses_mismatched_shapes_and_precisions() -> None:
    joint = _joint(0.5, 0.5)

    with pytest.raises(ValueError, match="one value per pair"):
        PairTable.from_values(joint, [1.0])

    with pytest.raises(ValueError, match="truth values"):
        PairTable(
            joint, torch.zeros(3, dtype=torch.float64), torch.ones(2, dtype=torch.bool)
        )

    with pytest.raises(ValueError, match="decided mask"):
        PairTable(
            joint, torch.zeros(2, dtype=torch.float64), torch.ones(3, dtype=torch.bool)
        )

    with pytest.raises(TypeError, match="floating point"):
        PairTable(
            joint, torch.zeros(2, dtype=torch.int64), torch.ones(2, dtype=torch.bool)
        )

    with pytest.raises(TypeError, match="boolean"):
        PairTable(
            joint, torch.zeros(2, dtype=torch.float64), torch.ones(2, dtype=torch.int64)
        )


def test_a_decided_truth_value_outside_the_unit_interval_is_refused_under_checks(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(_compose, "CHECKS", True)
    joint = _joint(0.5, 0.5)
    decided = torch.tensor([True, False])

    with pytest.raises(ValueError, match="unit interval"):
        PairTable(joint, torch.tensor([1.5, 0.0], dtype=torch.float64), decided)

    PairTable(joint, torch.tensor([1.0, 7.0], dtype=torch.float64), decided)


@pytest.mark.parametrize("path", list(GradientPath), ids=lambda path: path.name.lower())
def test_best_takes_the_heaviest_verdict_alone_where_marginal_weighs_it(
    path: GradientPath,
) -> None:
    joint = _joint(0.51, 0.49)

    best = compose([_table(joint, 1.0, 0.0)], composition=Best(), gradient_path=path)
    marginal = compose(
        [_table(joint, 1.0, 0.0)], composition=Marginal(), gradient_path=path
    )
    tempered = compose(
        [_table(joint, 1.0, 0.0)], composition=Tempered(1.0), gradient_path=path
    )
    reversed_best = compose(
        [_table(joint, 0.0, 1.0)], composition=Best(), gradient_path=path
    )

    assert _probability(best) == 1.0
    assert _probability(marginal) == pytest.approx(0.51)
    assert _probability(tempered) == pytest.approx(0.51)
    assert reversed_best.log_probability is not None
    assert float(reversed_best.log_probability) == -inf
    assert not reversed_best.log_probability.requires_grad
    assert float(best.residual) == pytest.approx(0.0)


def test_an_abstained_pair_moves_its_mass_to_the_residual() -> None:
    joint = _joint(0.5, 0.3, 0.2)
    composed = compose(
        [_table(joint, 1.0, None, 0.0)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert _probability(composed) == pytest.approx(0.5 / 0.7)
    assert float(composed.residual) == pytest.approx(0.3)


def test_a_table_deciding_nothing_composes_to_no_estimate_and_a_full_residual() -> None:
    composed = compose(
        [_table(_joint(0.5, 0.5), None, None)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert composed.log_probability is None
    assert float(composed.residual) == 1.0


def test_truncated_beams_leave_their_omitted_mass_in_the_residual() -> None:
    composed = compose(
        [_table(_joint(0.4, 0.2), 1.0, 1.0)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert _probability(composed) == pytest.approx(1.0)
    assert float(composed.residual) == pytest.approx(0.4)


def test_every_pair_entailing_is_a_certainty_that_still_sits_in_the_graph() -> None:
    joint = _joint(0.5, 0.5)
    composed = compose(
        [_table(joint, 1.0, 1.0)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert composed.log_probability is not None
    assert float(composed.log_probability.detach()) == 0.0
    assert composed.log_probability.requires_grad

    composed.log_probability.backward()

    assert joint.grad is not None
    assert joint.grad.tolist() == pytest.approx([0.0, 0.0], abs=1e-12)


def test_every_pair_refuted_is_an_impossibility_attached_to_nothing() -> None:
    for path in GradientPath:
        composed = compose(
            [_table(_joint(0.5, 0.5), 0.0, 0.0)],
            composition=Marginal(),
            gradient_path=path,
        )

        assert composed.log_probability is not None
        assert float(composed.log_probability) == -inf
        assert not composed.log_probability.requires_grad


def test_the_pathwise_gradient_is_that_of_the_renormalized_marginal() -> None:
    joint = _joint(0.51, 0.49)
    composed = compose(
        [_table(joint, 1.0, 0.0)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert composed.log_probability is not None

    composed.log_probability.backward()

    assert joint.grad is not None
    assert joint.grad.tolist() == pytest.approx([0.49, -0.49])


def test_the_score_function_gradient_is_the_surrogates() -> None:
    joint = _joint(0.51, 0.49)
    composed = compose(
        [_table(joint, 1.0, 0.0)],
        composition=Marginal(),
        gradient_path=GradientPath.SCORE_FUNCTION,
    )

    assert composed.log_probability is not None
    assert _probability(composed) == pytest.approx(0.51)

    composed.log_probability.backward()

    assert joint.grad is not None
    assert joint.grad.tolist() == pytest.approx([1.0, 0.0])


def test_a_baseline_at_the_estimate_makes_the_two_paths_agree() -> None:
    gradients: list[list[float]] = []

    for path in GradientPath:
        joint = _joint(0.4, 0.35, 0.25)
        composed = compose(
            [_table(joint, 1.0, 0.0, 1.0)],
            composition=Marginal(),
            gradient_path=path,
            baseline=0.65,
        )

        assert composed.log_probability is not None
        assert _probability(composed) == pytest.approx(0.65)

        composed.log_probability.backward()

        assert joint.grad is not None

        gradients.append(joint.grad.tolist())

    assert gradients[0] == pytest.approx(gradients[1])


def test_a_baseline_moves_the_score_function_gradient_and_not_the_estimate() -> None:
    estimates: list[float] = []
    gradients: list[list[float]] = []

    for baseline in (0.0, 0.5):
        joint = _joint(0.51, 0.49)
        composed = compose(
            [_table(joint, 1.0, 0.0)],
            composition=Marginal(),
            gradient_path=GradientPath.SCORE_FUNCTION,
            baseline=baseline,
        )

        assert composed.log_probability is not None

        composed.log_probability.backward()

        assert joint.grad is not None

        estimates.append(_probability(composed))
        gradients.append(joint.grad.tolist())

    assert estimates[0] == estimates[1]
    assert gradients[0] != pytest.approx(gradients[1])


def test_the_best_rule_reaches_one_pair_and_the_marginal_reaches_every_pair() -> None:
    reached: list[list[bool]] = []

    for composition in (Best(), Marginal()):
        joint = _joint(0.4, 0.35, 0.25)
        composed = compose(
            [_table(joint, 1.0, 0.0, 1.0)],
            composition=composition,
            gradient_path=GradientPath.PATHWISE,
        )

        assert composed.log_probability is not None

        composed.log_probability.backward()

        assert joint.grad is not None

        reached.append([entry != 0.0 for entry in joint.grad.tolist()])

    assert reached == [[True, False, False], [True, True, True]]


def test_a_relaxed_truth_value_takes_the_gradient_on_the_pathwise_path_alone() -> None:
    for path, expected in (
        (GradientPath.PATHWISE, True),
        (GradientPath.SCORE_FUNCTION, False),
    ):
        joint = _joint(0.5, 0.5)
        truth = torch.tensor([0.8, 0.2], dtype=torch.float64, requires_grad=True)
        composed = compose(
            [PairTable(joint, truth, torch.tensor([True, True]))],
            composition=Marginal(),
            gradient_path=path,
        )

        assert composed.log_probability is not None
        assert _probability(composed) == pytest.approx(0.5)

        composed.log_probability.backward()

        assert (truth.grad is not None and bool(truth.grad.abs().sum() > 0)) is expected


def test_alignments_compose_by_the_mean_the_max_and_the_power_mean() -> None:
    joint = _joint(0.5, 0.5)
    entailing = _table(joint, 1.0, 1.0)
    refuting = _table(joint, 0.0, 0.0)
    mixed = _table(joint, 1.0, 0.0)

    for composition, expected in (
        (Marginal(), 0.5),
        (Best(), 1.0),
        (Tempered(0.5), (0.5 * (1.0 + 0.0)) ** 0.5),
    ):
        composed = compose(
            [entailing, refuting],
            composition=composition,
            gradient_path=GradientPath.PATHWISE,
        )

        assert _probability(composed) == pytest.approx(expected)

    assert _probability(
        compose(
            [mixed, mixed], composition=Marginal(), gradient_path=GradientPath.PATHWISE
        )
    ) == pytest.approx(0.5)


def test_an_undecided_alignment_drops_out_of_the_estimate_and_into_the_residual() -> (
    None
):
    joint = _joint(0.5, 0.5)
    composed = compose(
        [_table(joint, 1.0, 0.0), _table(joint, None, None)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )
    best = compose(
        [_table(joint, 1.0, 0.0), _table(joint, None, None)],
        composition=Best(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert _probability(composed) == pytest.approx(0.5)
    assert float(composed.residual) == pytest.approx(0.5)
    assert _probability(best) == pytest.approx(1.0)
    assert float(best.residual) == pytest.approx(1.0)


def test_the_gradient_over_alignments_reaches_each_alignments_weights() -> None:
    premise_weights = _weights(0.5, 0.5)
    hypothesis_weights = _weights(1.0)
    joint = pair_log_weights(
        _weighted((RAIN, COLD), premise_weights),
        _weighted((WET,), hypothesis_weights),
    )
    composed = compose(
        [_table(joint, 1.0, 0.0), _table(joint, 1.0, 1.0)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert _probability(composed) == pytest.approx(0.75)
    assert composed.log_probability is not None

    composed.log_probability.backward()

    assert premise_weights.grad is not None
    assert premise_weights.grad.tolist() == pytest.approx([1.0 / 6.0, -1.0 / 6.0])


def test_composition_needs_a_table_and_a_finite_baseline() -> None:
    with pytest.raises(ValueError, match="at least one alignment"):
        compose([], composition=Marginal(), gradient_path=GradientPath.PATHWISE)

    with pytest.raises(ValueError, match="baseline"):
        compose(
            [_table(_joint(1.0), 1.0)],
            composition=Marginal(),
            gradient_path=GradientPath.SCORE_FUNCTION,
            baseline=nan,
        )

    with pytest.raises(TypeError, match="baseline"):
        compose(
            [_table(_joint(1.0), 1.0)],
            composition=Marginal(),
            gradient_path=GradientPath.SCORE_FUNCTION,
            baseline=cast("float", "mean"),
        )


def test_a_tiny_probability_is_composed_in_log_space() -> None:
    joint = torch.tensor([log(1e-300), log(1.0 - 1e-300)], dtype=torch.float64)
    composed = compose(
        [_table(joint, 1.0, 0.0)],
        composition=Marginal(),
        gradient_path=GradientPath.PATHWISE,
    )

    assert composed.log_probability is not None
    assert float(composed.log_probability) == pytest.approx(log(1e-300))


def test_a_score_function_estimate_below_the_floor_of_probability_space_is_finite() -> (
    None
):
    gradients: list[list[float]] = []

    for path in GradientPath:
        joint = torch.tensor([-800.0, 0.0], dtype=torch.float64, requires_grad=True)
        composed = compose(
            [_table(joint, 1.0, 0.0)], composition=Marginal(), gradient_path=path
        )

        assert composed.log_probability is not None
        assert float(composed.log_probability.detach()) == pytest.approx(-800.0)

        composed.log_probability.backward()

        assert joint.grad is not None
        assert all(isfinite(entry) for entry in joint.grad.tolist())

        gradients.append(joint.grad.tolist())

    assert gradients == [
        pytest.approx([1.0, -1.0]),
        pytest.approx([1.0, 0.0]),
        pytest.approx([1.0, 0.0]),
    ]


def _gap(composition: Marginal | Best | Tempered) -> tuple[float, float]:
    """Return the residual and the gap it is measured against under one rule.

    The heavy pair is refuted and the light pair undecided in the truncated
    table, and proved in the one it is compared with.
    """
    truncated = compose(
        [_table(_joint(0.99, 0.01), 0.0, None)],
        composition=composition,
        gradient_path=GradientPath.PATHWISE,
    )
    exhaustive = compose(
        [_table(_joint(0.99, 0.01), 0.0, 1.0)],
        composition=composition,
        gradient_path=GradientPath.PATHWISE,
    )

    assert truncated.log_probability is not None
    assert float(truncated.log_probability) == -inf

    return float(truncated.residual), _probability(exhaustive)


def test_the_residual_bounds_the_gap_under_the_marginal_rule_alone() -> None:
    residual, gap = _gap(Marginal())

    assert residual == pytest.approx(0.01)
    assert gap <= residual + 1e-12

    residual, gap = _gap(Tempered(1.0))

    assert gap <= residual + 1e-12

    for temperature in (2.0, 4.0):
        residual, gap = _gap(Tempered(temperature))

        assert residual == pytest.approx(0.01)
        assert gap > residual


@pytest.mark.parametrize("path", list(GradientPath), ids=lambda path: path.name.lower())
def test_a_certainty_over_beams_summing_to_one_is_a_prediction_at_every_split(
    path: GradientPath,
) -> None:
    for numerator in range(1, 97):
        share = numerator / 97
        joint = pair_log_weights(
            _weighted((RAIN, COLD), _weights(share, 1.0 - share)),
            _weighted((WET, DRY), _weights(1.0 - share, share)),
        )
        composed = compose(
            [_table(joint, 1.0, 1.0, 1.0, 1.0)],
            composition=Marginal(),
            gradient_path=path,
        )
        assert composed.log_probability is not None
        assert float(composed.log_probability.detach()) == pytest.approx(0.0, abs=1e-12)
        assert float(composed.residual) == pytest.approx(0.0, abs=1e-12)


def test_heaviest_keeps_the_top_entries_up_to_the_width() -> None:
    weights = _joint(0.1, 0.5, 0.2, 0.2)

    assert _compose.heaviest(weights, 1) == frozenset({1})
    assert _compose.heaviest(weights, 3) == frozenset({1, 2, 3})
    assert _compose.heaviest(weights, 10) == frozenset({0, 1, 2, 3})
    assert _compose.heaviest(weights, 0) == frozenset()
