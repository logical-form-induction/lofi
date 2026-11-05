import copy
import pickle
from math import inf, nan, sqrt
from typing import TYPE_CHECKING, Protocol, cast

import pytest

from lofi.eval.grammaticality import (
    grammaticality_report,
    noise_adjusted_auc,
)

if TYPE_CHECKING:
    from collections.abc import Sequence


class _ReferenceMetrics(Protocol):
    """The reference functions used in the numerical comparison."""

    def roc_auc_score(
        self, labels: Sequence[int], scores: Sequence[float]
    ) -> float: ...

    def average_precision_score(
        self, labels: Sequence[int], scores: Sequence[float]
    ) -> float: ...

    def roc_curve(
        self,
        labels: Sequence[int],
        scores: Sequence[float],
        *,
        drop_intermediate: bool,
    ) -> tuple[Sequence[float], Sequence[float], Sequence[float]]: ...

    def accuracy_score(
        self, labels: Sequence[int], predictions: Sequence[int]
    ) -> float: ...

    def matthews_corrcoef(
        self, labels: Sequence[int], predictions: Sequence[int]
    ) -> float: ...


LABELS = (0, 0, 1, 1)
SCORES = (0.1, 0.4, 0.35, 0.8)
FLIP_RATE = 0.1


def test_a_worked_battery_reports_every_metric_and_its_noise_ceiling() -> None:
    report = grammaticality_report(LABELS, SCORES, flip_rate=FLIP_RATE)

    assert report.n == 4
    assert report.positives == 2
    assert report.roc_auc == 0.75
    assert report.pr_auc == pytest.approx(5 / 6)
    assert report.youden_threshold == 0.8
    assert report.accuracy_at_youden == 0.75
    assert report.accuracy_at_half == 0.75
    assert report.mcc == pytest.approx(2 / sqrt(12))
    assert report.noise_adjusted_auc == 0.8125
    assert report.auc_ceiling == 0.9
    assert report.flip_rate == FLIP_RATE
    assert report.primary == report.roc_auc


def test_the_battery_matches_the_installed_reference_implementation() -> None:
    reference = cast(
        "_ReferenceMetrics",
        pytest.importorskip("sklearn.metrics", exc_type=ImportError),
    )
    labels = (0, 0, 0, 1, 0, 0, 1, 0, 1, 1)
    scores = (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95)
    report = grammaticality_report(labels, scores, flip_rate=0.0)
    false_positive, true_positive, thresholds = reference.roc_curve(
        labels, scores, drop_intermediate=False
    )
    best = max(
        range(len(thresholds)),
        key=lambda index: (
            true_positive[index] - false_positive[index],
            thresholds[index],
        ),
    )
    youden_predictions = tuple(int(score >= thresholds[best]) for score in scores)
    half_predictions = tuple(int(score >= 0.5) for score in scores)

    assert report.roc_auc == pytest.approx(reference.roc_auc_score(labels, scores))
    assert report.pr_auc == pytest.approx(
        reference.average_precision_score(labels, scores)
    )
    assert report.youden_threshold == pytest.approx(thresholds[best])
    assert report.accuracy_at_youden == pytest.approx(
        reference.accuracy_score(labels, youden_predictions)
    )
    assert report.accuracy_at_half == pytest.approx(
        reference.accuracy_score(labels, half_predictions)
    )
    assert report.mcc == pytest.approx(
        reference.matthews_corrcoef(labels, half_predictions)
    )


def test_tied_scores_enter_together_and_input_order_cannot_break_the_tie() -> None:
    labels = (0, 1, 0, 1)
    scores = (0.5, 0.5, 0.1, 0.9)
    report = grammaticality_report(labels, scores, flip_rate=0.0)
    reversed_report = grammaticality_report(
        tuple(reversed(labels)), tuple(reversed(scores)), flip_rate=0.0
    )

    assert report.roc_auc == pytest.approx(7 / 8)
    assert report.pr_auc == pytest.approx(5 / 6)
    assert report.youden_threshold == 0.9
    assert reversed_report == report


def test_the_youden_tie_break_selects_the_greatest_observed_threshold() -> None:
    report = grammaticality_report(LABELS, SCORES, flip_rate=0.0)

    assert report.youden_threshold == 0.8
    assert report.accuracy_at_youden == 0.75


def test_the_youden_threshold_includes_the_all_negative_roc_endpoint() -> None:
    reference = cast(
        "_ReferenceMetrics",
        pytest.importorskip("sklearn.metrics", exc_type=ImportError),
    )
    labels = (1, 1, 0, 0)
    scores = (0.1, 0.2, 0.8, 0.9)
    report = grammaticality_report(labels, scores, flip_rate=0.0)
    false_positive, true_positive, thresholds = reference.roc_curve(
        labels, scores, drop_intermediate=False
    )
    best = max(
        range(len(thresholds)),
        key=lambda index: (
            true_positive[index] - false_positive[index],
            thresholds[index],
        ),
    )

    assert report.youden_threshold == thresholds[best] == inf
    assert report.accuracy_at_youden == 0.5


def test_matthews_correlation_is_zero_when_prediction_has_one_class() -> None:
    report = grammaticality_report((0, 1), (0.6, 0.6), flip_rate=0.0)

    assert report.mcc == 0.0
    assert report.accuracy_at_half == 0.5
    assert report.youden_threshold == inf


def test_a_score_of_exactly_one_half_is_classified_as_grammatical() -> None:
    report = grammaticality_report((0, 1), (0.4, 0.5), flip_rate=0.0)

    assert report.accuracy_at_half == 1.0
    assert report.mcc == 1.0


def test_reports_are_immutable_copyable_hashable_and_pickleable() -> None:
    report = grammaticality_report(LABELS, SCORES, flip_rate=FLIP_RATE)

    assert copy.copy(report) == report
    assert copy.deepcopy(report) == report
    assert pickle.loads(pickle.dumps(report)) == report
    assert hash(report) == hash(pickle.loads(pickle.dumps(report)))

    with pytest.raises(AttributeError):
        report.__setattr__("roc_auc", 0.0)


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ((), (), "one label of each class"),
        ((0, 0), (0.1, 0.2), "one label of each class"),
        ((1, 1), (0.8, 0.9), "one label of each class"),
        ((0, 1), (0.1,), "do not align"),
        ((0, 2), (0.1, 0.9), "not binary"),
        ((0, 1), (-0.1, 0.9), "unit interval"),
        ((0, 1), (0.1, 1.1), "unit interval"),
        ((0, 1), (nan, 0.9), "not finite"),
    ],
)
def test_the_battery_refuses_undefined_or_malformed_inputs(
    labels: tuple[int, ...], scores: tuple[float, ...], message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        grammaticality_report(labels, scores, flip_rate=0.0)


@pytest.mark.parametrize("flip_rate", [-0.1, 0.5, 0.6, inf, nan])
def test_the_adjustment_refuses_a_rate_for_which_it_is_undefined(
    flip_rate: float,
) -> None:
    with pytest.raises(ValueError, match=r"flip rate|flip-rate"):
        noise_adjusted_auc(0.75, flip_rate=flip_rate)


def test_the_adjustment_requires_the_experimental_flip_rate() -> None:
    no_arguments: dict[str, float] = {}

    with pytest.raises(TypeError, match="flip_rate"):
        noise_adjusted_auc(0.75, **no_arguments)


def test_the_adjustment_refuses_an_invalid_observed_area() -> None:
    with pytest.raises(ValueError, match="unit interval"):
        noise_adjusted_auc(1.1, flip_rate=0.1)

    with pytest.raises(ValueError, match="not finite"):
        noise_adjusted_auc(nan, flip_rate=0.1)


def test_known_symmetric_noise_recovers_the_clean_auc_at_twenty_thousand_items() -> (
    None
):
    per_class = 10_000
    flip_every = 10
    flip_rate = 1.0 / flip_every
    labels = tuple(label for label in (0, 1) for _ in range(per_class))
    scores = tuple(
        (label + (position + 0.5) / per_class) / 2.0
        for label in (0, 1)
        for position in range(per_class)
    )
    noisy = tuple(
        1 - label if position % flip_every == 0 else label
        for label in (0, 1)
        for position in range(per_class)
    )
    clean = grammaticality_report(labels, scores, flip_rate=0.0)
    observed = grammaticality_report(noisy, scores, flip_rate=flip_rate)

    assert len(labels) == 20_000
    assert sum(left != right for left, right in zip(labels, noisy, strict=True)) == (
        len(labels) * flip_rate
    )
    assert observed.roc_auc == pytest.approx(0.9, abs=1e-4)
    assert observed.auc_ceiling == 0.9
    assert abs(observed.noise_adjusted_auc - clean.roc_auc) < 0.02
