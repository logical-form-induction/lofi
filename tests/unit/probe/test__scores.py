from collections.abc import Callable
from math import exp, log
from typing import Final, cast

import pytest
import torch
from torch import Tensor

from lofi.nn.sites import ComponentId, DTypeClass, Role, Site
from lofi.probe._scores import (
    best_two,
    certainty,
    consistency,
    cosine_stability,
    energy_concentration,
    js_divergence,
    js_divergence_score,
    kl_divergence_score,
    maximum_probability,
    norm_preservation,
    normalized_entropy,
    normalized_js_divergence,
    smoothness,
    sparsity,
    spectral_entropy,
    top_decile_power,
    top_two_margin,
)
from lofi.probe._values import Observation

CONTROLLER: Final = ComponentId("chart-controller")
WEIGHING: Final = Site(CONTROLLER, Role.COMBINATOR_LOGITS)
STATE: Final = Site(CONTROLLER, Role.RECURRENT_HIDDEN)

REAL: Final = DTypeClass.REAL
LOGS: Final = DTypeClass.LOG_PROBABILITY
PROBABILITIES: Final = DTypeClass.PROBABILITY

type Scaled = Callable[[float], Observation]
type Masked = Callable[[Observation], Observation]


def rows(numbers: list[list[float]], mask: Tensor | None = None) -> Observation:
    return Observation(
        WEIGHING, torch.tensor(numbers), axes=("row", "action"), mask=mask
    )


def logged(numbers: list[list[float]], mask: Tensor | None = None) -> Observation:
    return Observation(
        WEIGHING, torch.tensor(numbers).log(), axes=("row", "action"), mask=mask
    )


def steps(numbers: list[list[list[float]]], mask: Tensor | None = None) -> Observation:
    return Observation(
        STATE, torch.tensor(numbers), axes=("row", "step", "width"), mask=mask
    )


def sigmoid(number: float) -> float:
    return 1.0 / (1.0 + exp(-number))


# a distribution on three outcomes, its entropy, and the log of its width
SKEWED: Final = [[0.5, 0.25, 0.25]]
SKEWED_ENTROPY: Final = 0.5 * log(2) + 0.5 * log(4)
SKEWED_NORMALIZED: Final = SKEWED_ENTROPY / log(3)


def test_the_maximum_probability_is_the_largest_outcome_probability() -> None:
    assert maximum_probability(
        rows(SKEWED), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([0.5])


def test_a_maximum_over_log_probabilities_is_raised_exactly_once() -> None:
    reading = rows([[-1.0, -2.0, -3.0]])

    assert maximum_probability(
        reading, LOGS, over="action"
    ).tensor.tolist() == pytest.approx([exp(-1.0)])


def test_a_maximum_over_probabilities_is_not_renormalized() -> None:
    reading = rows([[0.5, 0.25, 0.125]])

    assert maximum_probability(
        reading, PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([0.5])


def test_a_probability_within_boundary_slack_is_clamped_to_one() -> None:
    reading = rows([[1.0 + 1e-7, 0.0]])

    assert maximum_probability(
        reading, PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([1.0])


def test_a_maximum_over_real_weights_normalizes_them_first() -> None:
    shifted = Observation(
        WEIGHING, torch.tensor(SKEWED).log() + 7.0, axes=("row", "action")
    )

    assert maximum_probability(shifted, REAL, over="action").tensor.tolist() == (
        pytest.approx([0.5])
    )


def test_the_maximum_uses_live_actions_and_preserves_the_other_axes() -> None:
    reading = Observation(
        WEIGHING,
        torch.tensor([[[0.25, 0.75, 0.0], [0.0, 0.0, 0.0]]]),
        axes=("row", "step", "action"),
        mask=torch.tensor([[[True, True, False], [False, False, False]]]),
    )
    scored = maximum_probability(reading, PROBABILITIES, over="action")

    assert scored.axes == ("row", "step")
    assert scored.tensor.flatten().tolist() == pytest.approx([0.75, 0.0])
    assert scored.mask is not None
    assert scored.mask.tolist() == [[True, False]]


def test_the_normalized_entropy_matches_the_hand_computed_value() -> None:
    assert normalized_entropy(
        rows(SKEWED), PROBABILITIES, over="action"
    ).tensor.tolist() == (pytest.approx([SKEWED_NORMALIZED]))


def test_the_normalized_entropy_of_a_uniform_distribution_is_one() -> None:
    assert normalized_entropy(
        rows([[0.25, 0.25, 0.25, 0.25]]), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([1.0])


def test_the_normalized_entropy_of_a_distribution_on_one_outcome_is_zero() -> None:
    assert normalized_entropy(
        rows([[0.0, 1.0, 0.0]]), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([0.0])


def test_an_entropy_over_log_probabilities_is_raised_exactly_once() -> None:
    assert normalized_entropy(logged(SKEWED), LOGS, over="action").tensor.tolist() == (
        pytest.approx([SKEWED_NORMALIZED])
    )


def test_an_entropy_over_real_weights_normalizes_them_first() -> None:
    # the logs of the skewed distribution, shifted by a constant, softmax back to it
    shifted = Observation(
        WEIGHING, torch.tensor(SKEWED).log() + 7.0, axes=("row", "action")
    )

    assert normalized_entropy(shifted, REAL, over="action").tensor.tolist() == (
        pytest.approx([SKEWED_NORMALIZED])
    )


def test_the_entropy_counts_only_the_live_outcomes_of_a_row() -> None:
    reading = Observation(
        WEIGHING,
        torch.tensor([[[0.5, 0.5, 0.0]]]),
        axes=("row", "step", "action"),
        mask=torch.tensor([[[True, True, False]]]),
    )

    assert normalized_entropy(
        reading, PROBABILITIES, over="action"
    ).tensor.flatten().tolist() == (pytest.approx([1.0]))


def test_a_row_with_one_live_outcome_has_no_entropy() -> None:
    reading = Observation(
        WEIGHING,
        torch.tensor([[[1.0, 0.0, 0.0]]]),
        axes=("row", "step", "action"),
        mask=torch.tensor([[[True, False, False]]]),
    )

    assert normalized_entropy(
        reading, PROBABILITIES, over="action"
    ).tensor.tolist() == [[0.0]]


def test_a_row_that_holds_no_data_is_masked_off_under_real_weights() -> None:
    reading = rows([[1.0, 1.0], [3.0, 4.0]], mask=torch.tensor([True, False]))
    result = normalized_entropy(reading, REAL, over="action")

    assert result.mask is not None
    assert result.mask.tolist() == [True, False]
    assert result.tensor[0].item() == pytest.approx(1.0)


def test_real_weights_are_normalized_over_the_live_outcomes_only() -> None:
    # the masked-off weight would take all the mass if it were normalized with the rest
    reading = Observation(
        WEIGHING,
        torch.tensor([[[0.0, 0.0, 100.0]]]),
        axes=("row", "step", "action"),
        mask=torch.tensor([[[True, True, False]]]),
    )

    assert normalized_entropy(
        reading, REAL, over="action"
    ).tensor.flatten().tolist() == (pytest.approx([1.0]))


def test_a_count_reading_weighs_no_outcomes() -> None:
    with pytest.raises(ValueError, match="weigh no outcomes"):
        normalized_entropy(rows([[1.0, 2.0]]), DTypeClass.INTEGER, over="action")


def test_a_declaration_that_is_not_a_dtype_class_is_refused() -> None:
    with pytest.raises(TypeError, match="DTypeClass"):
        normalized_entropy(
            rows([[1.0, 2.0]]), cast("DTypeClass", "real"), over="action"
        )


def test_a_distribution_over_one_outcome_has_no_normalized_entropy() -> None:
    with pytest.raises(ValueError, match="no gap"):
        normalized_entropy(rows([[1.0]]), PROBABILITIES, over="action")


def test_certainty_is_the_complement_of_the_normalized_entropy() -> None:
    assert certainty(rows(SKEWED), PROBABILITIES, over="action").tensor.tolist() == (
        pytest.approx([1.0 - SKEWED_NORMALIZED])
    )


def test_the_top_two_margin_is_the_gap_between_the_two_best_probabilities() -> None:
    assert top_two_margin(
        rows(SKEWED), PROBABILITIES, over="action"
    ).tensor.tolist() == (pytest.approx([0.25]))


def test_the_top_two_margin_over_real_weights_is_taken_between_probabilities() -> None:
    shifted = Observation(
        WEIGHING, torch.tensor(SKEWED).log() + 7.0, axes=("row", "action")
    )

    assert top_two_margin(shifted, REAL, over="action").tensor.tolist() == (
        pytest.approx([0.25])
    )


def test_the_best_two_are_the_two_largest_numbers_largest_first() -> None:
    assert best_two(
        rows([[3.0, 1.0, 2.0], [-1.0, -3.0, -2.0]]), over="action"
    ).tolist() == [
        [3.0, 2.0],
        [-1.0, -2.0],
    ]


def test_the_best_two_of_a_reading_with_one_position_do_not_exist() -> None:
    with pytest.raises(ValueError, match="no gap"):
        best_two(rows([[3.0]]), over="action")


def test_the_best_two_are_taken_along_the_named_axis_wherever_it_lies() -> None:
    reading = Observation(
        WEIGHING, torch.tensor([[3.0, 1.0], [2.0, 4.0]]), axes=("action", "row")
    )

    assert best_two(reading, over="action").tolist() == [[3.0, 2.0], [4.0, 1.0]]


def test_sparsity_is_the_fraction_of_entries_negligible_beside_the_largest() -> None:
    reading = rows([[0.0, 0.001, 1.0, 0.5]])

    assert sparsity(reading, REAL, over="action").tensor.tolist() == pytest.approx(
        [0.5]
    )


def test_sparsity_takes_its_tolerance_as_a_fraction_of_the_largest_entry() -> None:
    reading = rows([[0.0, 0.001, 1.0, 0.5]])

    assert sparsity(reading, REAL, over="action", tolerance=0.6).tensor.tolist() == (
        pytest.approx([0.75])
    )


def test_a_vector_of_zeros_is_entirely_sparse() -> None:
    assert sparsity(rows([[0.0, 0.0]]), REAL, over="action").tensor.tolist() == [1.0]


def test_sparsity_counts_only_live_entries_and_measures_against_them() -> None:
    # the masked-off entry is the largest, and it neither counts nor sets the scale
    reading = Observation(
        WEIGHING,
        torch.tensor([[[0.0, 1.0, 100.0]]]),
        axes=("row", "step", "action"),
        mask=torch.tensor([[[True, True, False]]]),
    )

    assert sparsity(reading, REAL, over="action").tensor.tolist() == [[0.5]]


def test_sparsity_over_log_probabilities_is_taken_over_the_probabilities() -> None:
    reading = logged([[0.0, 0.001, 0.5, 0.499]])

    assert sparsity(reading, LOGS, over="action").tensor.tolist() == pytest.approx(
        [0.5]
    )


def test_a_mask_reading_has_no_sparsity() -> None:
    reading = Observation(
        WEIGHING, torch.tensor([[True, False]]), axes=("row", "action")
    )

    with pytest.raises(ValueError, match="no magnitude"):
        sparsity(reading, DTypeClass.MASK, over="action")


def test_norm_preservation_matches_the_hand_computed_value() -> None:
    before = rows([[3.0, 0.0]])
    after = rows([[6.0, 0.0]])

    assert norm_preservation(before, after, REAL, over="action").tensor.tolist() == (
        pytest.approx([sigmoid(1.0 - log(2))])
    )


def test_a_transformation_that_keeps_the_norm_scores_the_sigmoid_of_one() -> None:
    before = rows([[3.0, 4.0]])
    after = rows([[0.0, 5.0]])

    assert norm_preservation(before, after, REAL, over="action").tensor.tolist() == (
        pytest.approx([sigmoid(1.0)])
    )


def test_a_transformation_that_zeroes_a_vector_preserves_nothing() -> None:
    before = rows([[3.0, 4.0], [0.0, 0.0]])
    after = rows([[0.0, 0.0], [3.0, 4.0]])

    assert norm_preservation(before, after, REAL, over="action").tensor.tolist() == [
        0.0,
        0.0,
    ]


def test_two_zero_vectors_preserve_each_other() -> None:
    assert norm_preservation(
        rows([[0.0, 0.0]]), rows([[0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([sigmoid(1.0)])


def test_readings_of_two_shapes_cannot_be_paired() -> None:
    with pytest.raises(ValueError, match="one shape"):
        norm_preservation(
            rows([[1.0, 2.0]]), rows([[1.0, 2.0, 3.0]]), REAL, over="action"
        )


def test_a_pairwise_score_is_read_at_the_first_readings_point() -> None:
    before = rows([[3.0, 4.0]])
    after = Observation(STATE, torch.tensor([[0.0, 5.0]]), axes=("row", "action"))

    assert norm_preservation(before, after, REAL, over="action").site is WEIGHING
    assert kl_divergence_score(after, before, REAL, over="action").site is STATE
    assert js_divergence_score(after, before, REAL, over="action").site is STATE


def test_readings_over_different_axes_cannot_be_paired() -> None:
    left = rows(HALVES)
    right = Observation(WEIGHING, torch.tensor(QUARTER), axes=("column", "action"))

    with pytest.raises(ValueError, match="one set of axes"):
        kl_divergence_score(left, right, PROBABILITIES, over="action")


def test_readings_masked_differently_cannot_be_paired() -> None:
    before = rows([[1.0, 2.0], [3.0, 4.0]], mask=torch.tensor([True, False]))
    after = rows([[1.0, 2.0], [3.0, 4.0]], mask=torch.tensor([True, True]))

    with pytest.raises(ValueError, match="masked differently"):
        norm_preservation(before, after, REAL, over="action")


ALTERNATING: Final = [[[1.0, 0.0], [0.0, 1.0]]]


def test_consistency_matches_the_hand_computed_value() -> None:
    # each outcome takes the values one and zero, whose population variance is 0.25
    assert consistency(
        steps(ALTERNATING), PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == pytest.approx([exp(-0.25)])


def test_a_distribution_that_never_changes_is_perfectly_consistent() -> None:
    assert consistency(
        steps([[[0.3, 0.7], [0.3, 0.7], [0.3, 0.7]]]),
        PROBABILITIES,
        across="step",
        over="width",
    ).tensor.tolist() == pytest.approx([1.0])


def test_consistency_ignores_a_step_that_holds_no_data() -> None:
    reading = steps(
        [[[0.3, 0.7], [0.3, 0.7], [0.0, 0.0]]],
        mask=torch.tensor([[True, True, False]]),
    )

    assert consistency(
        reading, PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([1.0]))


def test_consistency_over_real_weights_normalizes_each_step() -> None:
    # every step is the same distribution up to a constant shift
    reading = steps([[[0.0, 1.0], [5.0, 6.0]]])

    assert consistency(reading, REAL, across="step", over="width").tensor.tolist() == (
        pytest.approx([1.0])
    )


def test_the_step_axis_and_the_outcome_axis_must_differ() -> None:
    with pytest.raises(ValueError, match="one axis"):
        consistency(steps(ALTERNATING), PROBABILITIES, across="width", over="width")


def test_smoothness_matches_the_hand_computed_value() -> None:
    assert smoothness(
        steps(ALTERNATING), PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == pytest.approx([exp(-1.0)])


def test_smoothness_averages_over_every_transition() -> None:
    # the transitions move by 0.2 per outcome and by nothing, for a mean of 0.1
    reading = steps([[[0.5, 0.5], [0.3, 0.7], [0.3, 0.7]]])

    assert smoothness(
        reading, PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([exp(-0.1)]))


def test_a_single_step_is_perfectly_smooth() -> None:
    assert smoothness(
        steps([[[0.5, 0.5]]]), PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == [1.0]


def test_smoothness_skips_a_transition_into_a_step_that_holds_no_data() -> None:
    reading = steps(
        [[[0.5, 0.5], [0.3, 0.7], [0.0, 0.0]]],
        mask=torch.tensor([[True, True, False]]),
    )

    assert smoothness(
        reading, PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([exp(-0.2)]))


def test_smoothness_skips_a_transition_out_of_a_step_that_holds_no_data() -> None:
    reading = steps(
        [[[0.0, 0.0], [0.5, 0.5], [0.3, 0.7]]],
        mask=torch.tensor([[False, True, True]]),
    )

    assert smoothness(
        reading, PROBABILITIES, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([exp(-0.2)]))


def test_cosine_stability_of_orthogonal_consecutive_states_is_zero() -> None:
    reading = steps([[[1.0, 0.0], [0.0, 1.0], [1.0, 0.0]]])

    assert cosine_stability(
        reading, REAL, across="step", over="width"
    ).tensor.tolist() == [0.0]


def test_cosine_stability_averages_the_cosines_of_consecutive_pairs() -> None:
    # the pairs have cosines one and minus one
    reading = steps([[[1.0, 0.0], [2.0, 0.0], [-3.0, 0.0]]])

    assert cosine_stability(
        reading, REAL, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([0.0]))


def test_cosine_stability_of_states_that_only_grow_is_one() -> None:
    reading = steps([[[1.0, 2.0], [2.0, 4.0]]])

    assert cosine_stability(
        reading, REAL, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([1.0]))


def test_a_zero_state_has_cosine_zero_with_its_neighbor() -> None:
    reading = steps([[[1.0, 2.0], [0.0, 0.0]]])

    assert cosine_stability(
        reading, REAL, across="step", over="width"
    ).tensor.tolist() == [0.0]


def test_cosine_stability_skips_a_pair_with_a_step_that_holds_no_data() -> None:
    reading = steps(
        [[[1.0, 0.0], [2.0, 0.0], [0.0, 1.0]]],
        mask=torch.tensor([[True, True, False]]),
    )

    assert cosine_stability(
        reading, REAL, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([1.0]))


def test_cosine_stability_skips_a_pair_whose_first_step_holds_no_data() -> None:
    reading = steps(
        [[[0.0, 1.0], [1.0, 0.0], [2.0, 0.0]]],
        mask=torch.tensor([[False, True, True]]),
    )

    assert cosine_stability(
        reading, REAL, across="step", over="width"
    ).tensor.tolist() == (pytest.approx([1.0]))


def test_a_single_state_is_perfectly_stable() -> None:
    assert cosine_stability(
        steps([[[1.0, 2.0]]]), REAL, across="step", over="width"
    ).tensor.tolist() == [1.0]


HALVES: Final = [[0.5, 0.5]]
QUARTER: Final = [[0.25, 0.75]]
HALVES_FROM_QUARTER: Final = 0.5 * log(2) + 0.5 * log(2 / 3)


def test_the_kl_score_matches_the_hand_computed_value() -> None:
    assert kl_divergence_score(
        rows(HALVES), rows(QUARTER), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([sigmoid(HALVES_FROM_QUARTER)])


def test_the_kl_score_of_a_distribution_against_itself_is_one_half() -> None:
    assert kl_divergence_score(
        rows(QUARTER), rows(QUARTER), PROBABILITIES, over="action"
    ).tensor.tolist() == [0.5]


def test_the_kl_score_is_taken_from_the_left_reading_against_the_right() -> None:
    forward = kl_divergence_score(
        rows(HALVES), rows(QUARTER), PROBABILITIES, over="action"
    )
    backward = kl_divergence_score(
        rows(QUARTER), rows(HALVES), PROBABILITIES, over="action"
    )

    assert forward.tensor.item() != pytest.approx(backward.tensor.item())


def test_the_kl_score_over_log_probabilities_matches_the_probability_domain() -> None:
    assert kl_divergence_score(
        logged(HALVES), logged(QUARTER), LOGS, over="action"
    ).tensor.tolist() == pytest.approx([sigmoid(HALVES_FROM_QUARTER)])


def test_the_kl_score_is_one_where_the_right_rules_out_what_the_left_weighs() -> None:
    assert kl_divergence_score(
        rows(HALVES), rows([[1.0, 0.0]]), PROBABILITIES, over="action"
    ).tensor.tolist() == [1.0]


def test_the_js_score_of_disjoint_distributions_is_two_thirds() -> None:
    assert js_divergence_score(
        rows([[1.0, 0.0]]), rows([[0.0, 1.0]]), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([2.0 / 3.0])


def test_the_js_score_of_a_distribution_against_itself_is_one_half() -> None:
    assert js_divergence_score(
        rows(QUARTER), rows(QUARTER), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([0.5])


def test_the_js_score_is_symmetric() -> None:
    forward = js_divergence_score(
        rows(HALVES), rows(QUARTER), PROBABILITIES, over="action"
    )
    backward = js_divergence_score(
        rows(QUARTER), rows(HALVES), PROBABILITIES, over="action"
    )

    assert forward.tensor.item() == pytest.approx(backward.tensor.item())


def test_the_js_score_matches_the_hand_computed_value() -> None:
    # the mixture of the halves and the quarter is (0.375, 0.625)
    mixture = [0.375, 0.625]
    from_halves = sum(p * log(p / m) for p, m in zip(HALVES[0], mixture, strict=True))
    from_quarter = sum(q * log(q / m) for q, m in zip(QUARTER[0], mixture, strict=True))

    assert js_divergence_score(
        rows(HALVES), rows(QUARTER), PROBABILITIES, over="action"
    ).tensor.tolist() == pytest.approx([sigmoid(0.5 * (from_halves + from_quarter))])


def test_raw_js_divergence_has_zero_and_log_two_at_its_extremes() -> None:
    identical = js_divergence(
        rows(QUARTER), rows(QUARTER), PROBABILITIES, over="action"
    )
    disjoint = js_divergence(
        rows([[1.0, 0.0]]), rows([[0.0, 1.0]]), PROBABILITIES, over="action"
    )

    assert identical.tensor.tolist() == pytest.approx([0.0])
    assert disjoint.tensor.tolist() == pytest.approx([log(2)])


def test_raw_js_divergence_matches_live_outcomes_under_a_mask() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])
    left = rows([[0.5, 0.5, 0.0], [1.0, 0.0, 0.0]], mask=mask)
    right = rows([[0.25, 0.75, 0.0], [1.0, 0.0, 0.0]], mask=mask)
    mixture = [0.375, 0.625]
    from_left = sum(p * log(p / m) for p, m in zip(HALVES[0], mixture, strict=True))
    from_right = sum(q * log(q / m) for q, m in zip(QUARTER[0], mixture, strict=True))
    result = js_divergence(left, right, PROBABILITIES, over="action")

    assert result.tensor.tolist() == pytest.approx(
        [0.5 * (from_left + from_right), 0.0]
    )
    assert result.mask is not None
    assert result.mask.tolist() == [True, True]


def test_normalized_js_divergence_has_zero_and_one_at_its_extremes() -> None:
    identical = normalized_js_divergence(
        rows(QUARTER), rows(QUARTER), PROBABILITIES, over="action"
    )
    disjoint = normalized_js_divergence(
        rows([[1.0, 0.0]]), rows([[0.0, 1.0]]), PROBABILITIES, over="action"
    )

    assert identical.tensor.tolist() == pytest.approx([0.0])
    assert disjoint.tensor.tolist() == pytest.approx([1.0])


def test_normalized_js_divergence_is_symmetric_and_preserves_outer_axes() -> None:
    left = rows(HALVES)
    right = rows(QUARTER)
    forward = normalized_js_divergence(left, right, PROBABILITIES, over="action")
    backward = normalized_js_divergence(right, left, PROBABILITIES, over="action")

    assert forward.site is WEIGHING
    assert forward.axes == ("row",)
    assert forward.tensor.item() == pytest.approx(backward.tensor.item())


def test_energy_concentration_matches_the_hand_computed_values() -> None:
    # one entry holds everything; four equal entries need three to reach nine tenths;
    # energies nine and sixteen need only the sixteen
    reading = rows([[1.0, 0.0, 0.0, 0.0], [1.0, 1.0, 1.0, 1.0], [3.0, 4.0, 0.0, 0.0]])

    assert energy_concentration(reading, REAL, over="action").tensor.tolist() == (
        pytest.approx([1.0, 0.25, 0.75])
    )


def test_an_entry_that_lands_exactly_on_nine_tenths_is_within_it() -> None:
    # energies nine and one: the nine alone is exactly nine tenths of the total
    assert energy_concentration(
        rows([[3.0, 1.0, 0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([0.75])


def test_a_vector_of_zeros_concentrates_no_energy() -> None:
    assert energy_concentration(
        rows([[0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == [0.0]


def test_energy_concentration_counts_only_live_entries() -> None:
    reading = Observation(
        WEIGHING,
        torch.tensor([[[1.0, 1.0, 5.0, 0.0]]]),
        axes=("row", "step", "action"),
        mask=torch.tensor([[[True, True, False, False]]]),
    )

    assert energy_concentration(
        reading, REAL, over="action"
    ).tensor.flatten().tolist() == (pytest.approx([0.5]))


def test_the_spectral_entropy_of_a_constant_vector_is_zero() -> None:
    assert spectral_entropy(
        rows([[1.0, 1.0, 1.0, 1.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([0.0])


def test_the_spectral_entropy_of_an_impulse_is_one() -> None:
    assert spectral_entropy(
        rows([[1.0, 0.0, 0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([1.0])


def test_the_spectral_entropy_matches_the_hand_computed_value() -> None:
    # the transform of (1, 0, 1, 0) is (2, 0, 2, 0): two equal bins of four
    assert spectral_entropy(
        rows([[1.0, 0.0, 1.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([log(2) / log(4)])


def test_a_vector_of_zeros_has_no_spectral_entropy() -> None:
    assert spectral_entropy(
        rows([[0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == [0.0]


def test_a_spectrum_over_one_bin_has_no_shape() -> None:
    with pytest.raises(ValueError, match="no gap"):
        spectral_entropy(rows([[1.0]]), REAL, over="action")


def test_the_top_decile_power_of_a_constant_vector_is_one() -> None:
    assert top_decile_power(
        rows([[1.0, 1.0, 1.0, 1.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([1.0])


def test_the_top_decile_power_of_an_impulse_is_its_share_of_bins() -> None:
    # four bins keep one, which holds a quarter of a flat spectrum
    assert top_decile_power(
        rows([[1.0, 0.0, 0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == pytest.approx([0.25])


def test_the_top_decile_keeps_a_tenth_of_the_bins_once_there_are_enough() -> None:
    # twenty bins keep two, which hold a tenth of a flat spectrum
    impulse = [[1.0] + [0.0] * 19]

    assert top_decile_power(rows(impulse), REAL, over="action").tensor.tolist() == (
        pytest.approx([0.1])
    )


def test_a_vector_of_zeros_has_no_top_decile_power() -> None:
    assert top_decile_power(
        rows([[0.0, 0.0]]), REAL, over="action"
    ).tensor.tolist() == [0.0]


def test_an_axis_the_reading_does_not_declare_is_refused() -> None:
    with pytest.raises(KeyError, match="width"):
        normalized_entropy(rows(SKEWED), PROBABILITIES, over="width")


def test_a_reading_that_contradicts_its_declared_class_is_refused() -> None:
    with pytest.raises(ValueError, match="above zero"):
        normalized_entropy(rows([[0.5, 0.5]]), LOGS, over="action")


def a_real_reading(factor: float, seed: int = 7) -> Observation:
    generator = torch.Generator().manual_seed(seed)
    numbers = torch.randn((3, 4, 12), generator=generator) * factor

    return Observation(STATE, numbers, axes=("row", "step", "width"))


def another_real_reading(factor: float) -> Observation:
    return a_real_reading(factor, seed=11)


# every score, over real readings scaled by one factor, with the closed range its
# docstring declares
RANGED: Final = (
    pytest.param(
        lambda f: normalized_entropy(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="normalized_entropy",
    ),
    pytest.param(
        lambda f: certainty(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="certainty",
    ),
    pytest.param(
        lambda f: top_two_margin(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="top_two_margin",
    ),
    pytest.param(
        lambda f: sparsity(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="sparsity",
    ),
    pytest.param(
        lambda f: norm_preservation(
            a_real_reading(f), another_real_reading(f), REAL, over="width"
        ),
        0.0,
        sigmoid(1.0),
        id="norm_preservation",
    ),
    pytest.param(
        lambda f: consistency(a_real_reading(f), REAL, across="step", over="width"),
        exp(-0.25),
        1.0,
        id="consistency",
    ),
    pytest.param(
        lambda f: smoothness(a_real_reading(f), REAL, across="step", over="width"),
        exp(-1.0),
        1.0,
        id="smoothness",
    ),
    pytest.param(
        lambda f: cosine_stability(
            a_real_reading(f), REAL, across="step", over="width"
        ),
        -1.0,
        1.0,
        id="cosine_stability",
    ),
    pytest.param(
        lambda f: kl_divergence_score(
            a_real_reading(f), another_real_reading(f), REAL, over="width"
        ),
        0.5,
        1.0,
        id="kl_divergence_score",
    ),
    pytest.param(
        lambda f: js_divergence_score(
            a_real_reading(f), another_real_reading(f), REAL, over="width"
        ),
        0.5,
        sigmoid(log(2)),
        id="js_divergence_score",
    ),
    pytest.param(
        lambda f: energy_concentration(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="energy_concentration",
    ),
    pytest.param(
        lambda f: spectral_entropy(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="spectral_entropy",
    ),
    pytest.param(
        lambda f: top_decile_power(a_real_reading(f), REAL, over="width"),
        0.0,
        1.0,
        id="top_decile_power",
    ),
)

# the scores that read magnitudes rather than a softmax, which scaling cannot move
SCALE_FREE: Final = (
    pytest.param(
        lambda f: sparsity(a_real_reading(f), REAL, over="width"), id="sparsity"
    ),
    pytest.param(
        lambda f: norm_preservation(
            a_real_reading(f), another_real_reading(f), REAL, over="width"
        ),
        id="norm_preservation",
    ),
    pytest.param(
        lambda f: cosine_stability(
            a_real_reading(f), REAL, across="step", over="width"
        ),
        id="cosine_stability",
    ),
    pytest.param(
        lambda f: energy_concentration(a_real_reading(f), REAL, over="width"),
        id="energy_concentration",
    ),
    pytest.param(
        lambda f: spectral_entropy(a_real_reading(f), REAL, over="width"),
        id="spectral_entropy",
    ),
    pytest.param(
        lambda f: top_decile_power(a_real_reading(f), REAL, over="width"),
        id="top_decile_power",
    ),
)


@pytest.mark.parametrize("factor", [1e-6, 1.0, 1e6])
@pytest.mark.parametrize(("score", "low", "high"), RANGED)
def test_every_score_stays_finite_and_in_its_range_under_extreme_scaling(
    score: Scaled, low: float, high: float, factor: float
) -> None:
    result = score(factor)

    assert result.site is STATE
    assert result.axes in {("row",), ("row", "step")}
    assert result.mask is None
    assert torch.isfinite(result.tensor).all()
    assert (result.tensor >= low - 1e-6).all()
    assert (result.tensor <= high + 1e-6).all()


@pytest.mark.parametrize(("score", "low", "high"), RANGED)
def test_a_score_over_two_distinct_readings_is_not_constant(
    score: Scaled, low: float, high: float
) -> None:
    # a pair of readings scores somewhere strictly inside its range, so the table is
    # not passing on the degenerate value a reading takes against itself
    result = score(1.0).tensor

    assert (result > low + 1e-3).any()
    assert (result < high - 1e-3).any()


@pytest.mark.parametrize("score", SCALE_FREE)
def test_a_score_over_magnitudes_is_unchanged_by_scaling(score: Scaled) -> None:
    unit = score(1.0).tensor.flatten().tolist()

    assert score(1e-6).tensor.flatten().tolist() == pytest.approx(unit, abs=1e-4)
    assert score(1e6).tensor.flatten().tolist() == pytest.approx(unit, abs=1e-4)


# every score over a masked reading whose last row holds no data
DEAD_LAST_ROW: Final = (
    pytest.param(
        lambda o: normalized_entropy(o, REAL, over="width"), id="normalized_entropy"
    ),
    pytest.param(lambda o: certainty(o, REAL, over="width"), id="certainty"),
    pytest.param(lambda o: top_two_margin(o, REAL, over="width"), id="top_two_margin"),
    pytest.param(lambda o: sparsity(o, REAL, over="width"), id="sparsity"),
    pytest.param(
        lambda o: norm_preservation(o, o, REAL, over="width"), id="norm_preservation"
    ),
    pytest.param(
        lambda o: consistency(o, REAL, across="step", over="width"), id="consistency"
    ),
    pytest.param(
        lambda o: smoothness(o, REAL, across="step", over="width"), id="smoothness"
    ),
    pytest.param(
        lambda o: cosine_stability(o, REAL, across="step", over="width"),
        id="cosine_stability",
    ),
    pytest.param(
        lambda o: kl_divergence_score(o, o, REAL, over="width"),
        id="kl_divergence_score",
    ),
    pytest.param(
        lambda o: js_divergence_score(o, o, REAL, over="width"),
        id="js_divergence_score",
    ),
    pytest.param(
        lambda o: energy_concentration(o, REAL, over="width"),
        id="energy_concentration",
    ),
    pytest.param(
        lambda o: spectral_entropy(o, REAL, over="width"), id="spectral_entropy"
    ),
    pytest.param(
        lambda o: top_decile_power(o, REAL, over="width"), id="top_decile_power"
    ),
)


@pytest.mark.parametrize("score", DEAD_LAST_ROW)
def test_every_score_masks_off_a_row_that_holds_no_data(score: Masked) -> None:
    reading = Observation(
        STATE,
        torch.randn((3, 4, 12), generator=torch.Generator().manual_seed(7)),
        axes=("row", "step", "width"),
        mask=torch.tensor([True, True, False]),
    )
    result = score(reading)

    assert result.mask is not None
    assert result.mask.tolist() == [True, True, False]


def test_a_score_names_the_axes_it_did_not_reduce_in_their_own_order() -> None:
    reading = Observation(
        WEIGHING, torch.rand((2, 3, 4)), axes=("row", "action", "step")
    )
    result = certainty(reading, REAL, over="action")

    assert result.site is WEIGHING
    assert result.axes == ("row", "step")
    assert tuple(result.tensor.shape) == (2, 4)


def test_a_score_that_reduces_a_masked_axis_keeps_a_row_with_any_live_step() -> None:
    reading = steps(
        [[[0.3, 0.7], [0.3, 0.7]], [[0.0, 0.0], [0.3, 0.7]], [[0.0, 0.0], [0.0, 0.0]]],
        mask=torch.tensor([[True, True], [False, True], [False, False]]),
    )
    result = consistency(reading, PROBABILITIES, across="step", over="width")

    assert result.mask is not None
    assert result.mask.tolist() == [True, True, False]


def test_a_score_keeps_a_mask_that_stops_short_of_the_reduced_axis() -> None:
    reading = steps(
        [[[0.3, 0.7], [0.0, 0.0]], [[0.3, 0.7], [0.3, 0.7]]],
        mask=torch.tensor([[True, False], [True, True]]),
    )
    result = normalized_entropy(reading, PROBABILITIES, over="width")

    assert result.mask is not None
    assert result.mask.tolist() == [[True, False], [True, True]]
