"""Binary grammaticality-prediction metrics."""

from itertools import groupby
from math import fsum, inf, sqrt
from typing import TYPE_CHECKING, NamedTuple

from lofi._checks import ensure_finite

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

__all__ = [
    "GrammaticalityReport",
    "grammaticality_report",
    "noise_adjusted_auc",
]


type _Example = tuple[int, float]
"""A binary grammaticality label and its predicted probability."""

type _ScoreGroup = tuple[float, int, int]
"""A score followed by positive and negative counts tied at that score."""

type _Scored = tuple[float, bool]
"""A grammaticality score and whether its item is positive."""


def _ranks(scores: Sequence[float]) -> list[float]:
    """Return one-based ranks, averaging the positions occupied by ties."""
    ordered = sorted(range(len(scores)), key=lambda at: scores[at])
    ranks = [0.0] * len(scores)
    placed = 0

    for _, run in groupby(ordered, key=lambda at: scores[at]):
        tied = list(run)
        shared = placed + (len(tied) + 1) / 2.0

        for at in tied:
            ranks[at] = shared

        placed += len(tied)

    return ranks


def _area_under_curve(scored: Sequence[_Scored], /) -> float | None:
    """Return the probability that a positive item outscores a negative one."""
    positives = sum(1 for _, positive in scored if positive)
    negatives = len(scored) - positives

    if positives == 0 or negatives == 0:
        return None

    ranks = _ranks([score for score, _ in scored])
    summed = fsum(
        rank for rank, (_, positive) in zip(ranks, scored, strict=True) if positive
    )

    return (summed - positives * (positives + 1) / 2.0) / (positives * negatives)


class GrammaticalityReport(NamedTuple):
    """The grammaticality-prediction metric battery.

    ROC-AUC is the primary metric. PR-AUC is average precision, so a tied score
    group enters the precision-recall curve in one step. The Youden threshold is
    the greatest ROC threshold maximizing sensitivity minus
    false-positive rate, including positive infinity for the all-negative
    prediction; predictions at either reported threshold are positive when their
    score is at least the threshold. Matthews correlation is computed at one half.

    Attributes
    ----------
    n : int
        Number of scored items.
    positives : int
        Number of items labeled grammatical.
    roc_auc : float
        Pairwise area under the receiver-operating-characteristic curve.
    pr_auc : float
        Stepwise area under the precision-recall curve, or average precision.
    youden_threshold : float
        Greatest threshold on the ROC curve attaining the maximum Youden index.
    accuracy_at_youden : float
        Accuracy at ``youden_threshold``.
    accuracy_at_half : float
        Accuracy at probability one half.
    mcc : float
        Matthews correlation at probability one half.
    noise_adjusted_auc : float
        ROC-AUC adjusted for the declared symmetric label-flip rate under balanced
        latent class prevalence.
    auc_ceiling : float
        Maximum observed ROC-AUC under the declared flip rate.
    flip_rate : float
        Symmetric label-flip rate used for adjustment.
    """

    n: int
    positives: int
    roc_auc: float
    pr_auc: float
    youden_threshold: float
    accuracy_at_youden: float
    accuracy_at_half: float
    mcc: float
    noise_adjusted_auc: float
    auc_ceiling: float
    flip_rate: float

    @property
    def primary(self) -> float:
        """Return ROC-AUC, the battery's primary metric."""
        return self.roc_auc


def _flip_rate(value: float) -> float:
    """Return a finite symmetric flip rate for which adjustment is defined."""
    rate = ensure_finite("a grammaticality label-flip rate", value)

    if not 0.0 <= rate < 0.5:
        msg = f"a grammaticality label-flip rate must lie in [0, 0.5), not {value!r}"
        raise ValueError(msg)

    return rate


def noise_adjusted_auc(roc_auc: float, *, flip_rate: float) -> float:
    r"""Adjust ROC-AUC for a declared symmetric label-flip rate.

    The specified attenuation assumes balanced latent class prevalence.

    Parameters
    ----------
    roc_auc : float
        Observed ROC-AUC in the unit interval.
    flip_rate : float
        Finite symmetric label-flip rate in $[0,\tfrac12)$. This argument is
        required because the adjustment is undefined at one half and cannot infer
        an experimental noise rate from predictions.

    Returns
    -------
    float
        $a_{\mathrm{adjusted}}=\tfrac12+(a-\tfrac12)/(1-2\eta)$,
        where $a$ is `roc_auc` and $\eta$ is `flip_rate`.

    Raises
    ------
    TypeError
        If an argument is not numeric or is a boolean.
    ValueError
        If an argument is not finite, ROC-AUC is outside the unit interval, or the
        flip rate is outside $[0,\tfrac12)$.
    """
    area = ensure_finite("a grammaticality ROC-AUC", roc_auc)

    if not 0.0 <= area <= 1.0:
        msg = f"a grammaticality ROC-AUC is outside the unit interval: {roc_auc!r}"
        raise ValueError(msg)

    rate = _flip_rate(flip_rate)

    return 0.5 + (area - 0.5) / (1.0 - 2.0 * rate)


def _label(value: int, position: int) -> int:
    """Return a binary label as zero or one."""
    if isinstance(value, bool):
        return int(value)

    if isinstance(value, int) and value in (0, 1):
        return value

    msg = f"grammaticality label {position} is not binary: {value!r}"
    raise ValueError(msg)


def _examples(
    labels: Iterable[int], p_gram: Iterable[float]
) -> tuple[tuple[_Example, ...], int]:
    """Collect and validate aligned binary labels and probabilities."""
    held_labels = tuple(
        _label(label, position) for position, label in enumerate(labels)
    )
    held_scores = tuple(
        ensure_finite(f"grammaticality probability {position}", score)
        for position, score in enumerate(p_gram)
    )

    if len(held_labels) != len(held_scores):
        msg = (
            f"{len(held_labels)} grammaticality labels do not align with "
            f"{len(held_scores)} probabilities"
        )
        raise ValueError(msg)

    for position, score in enumerate(held_scores):
        if not 0.0 <= score <= 1.0:
            msg = (
                f"grammaticality probability {position} is outside the unit "
                f"interval: {score!r}"
            )
            raise ValueError(msg)

    positives = sum(held_labels)

    if positives == 0 or positives == len(held_labels):
        msg = "grammaticality ROC-AUC requires at least one label of each class"
        raise ValueError(msg)

    return tuple(zip(held_labels, held_scores, strict=True)), positives


def _groups(examples: Sequence[_Example]) -> tuple[_ScoreGroup, ...]:
    """Return tied-score groups in descending score order."""
    ordered = sorted(examples, key=lambda example: example[1], reverse=True)
    groups: list[_ScoreGroup] = []
    start = 0

    while start < len(ordered):
        score = ordered[start][1]
        stop = start + 1

        while stop < len(ordered) and ordered[stop][1] == score:
            stop += 1

        positives = sum(label for label, _ in ordered[start:stop])
        groups.append((score, positives, stop - start - positives))
        start = stop

    return tuple(groups)


def _threshold_metrics(
    groups: Sequence[_ScoreGroup], positives: int, negatives: int
) -> tuple[float, float, float]:
    """Return average precision and the selected Youden threshold and accuracy."""
    true_positives = 0
    false_positives = 0
    previous_positives = 0
    average_precision = 0.0
    total = positives + negatives
    best_youden_numerator = 0
    best_threshold = inf
    best_accuracy = negatives / total

    for score, positive_count, negative_count in groups:
        true_positives += positive_count
        false_positives += negative_count
        average_precision += (
            (true_positives - previous_positives)
            * true_positives
            / (positives * (true_positives + false_positives))
        )
        previous_positives = true_positives
        youden_numerator = true_positives * negatives - false_positives * positives

        if youden_numerator <= best_youden_numerator:
            continue

        best_youden_numerator = youden_numerator
        best_threshold = score
        best_accuracy = (true_positives + negatives - false_positives) / total

    return average_precision, best_threshold, best_accuracy


def _at_half(examples: Sequence[_Example]) -> tuple[float, float]:
    """Return accuracy and Matthews correlation at probability one half."""
    true_positives = sum(label == 1 and score >= 0.5 for label, score in examples)
    false_positives = sum(label == 0 and score >= 0.5 for label, score in examples)
    false_negatives = sum(label == 1 and score < 0.5 for label, score in examples)
    true_negatives = len(examples) - true_positives - false_positives - false_negatives
    accuracy = (true_positives + true_negatives) / len(examples)
    radicand = (
        (true_positives + false_positives)
        * (true_positives + false_negatives)
        * (true_negatives + false_positives)
        * (true_negatives + false_negatives)
    )
    correlation = (
        0.0
        if radicand == 0
        else (true_positives * true_negatives - false_positives * false_negatives)
        / sqrt(radicand)
    )

    return accuracy, correlation


def grammaticality_report(
    labels: Iterable[int],
    p_gram: Iterable[float],
    *,
    flip_rate: float,
) -> GrammaticalityReport:
    r"""Compute binary grammaticality-prediction metrics.

    Parameters
    ----------
    labels : Iterable[int]
        Binary labels, with one denoting grammaticality.
    p_gram : Iterable[float]
        Predicted grammaticality probabilities in the unit interval.
    flip_rate : float
        Finite symmetric label-flip rate in $[0,\tfrac12)$.

    Returns
    -------
    GrammaticalityReport
        ROC-AUC, average precision, thresholded accuracy, Matthews correlation,
        and the declared-noise adjustment.

    Raises
    ------
    TypeError
        If a probability or flip rate is not numeric or is a boolean.
    ValueError
        If the inputs differ in length, a label is not binary, a probability is
        outside the unit interval, either class is absent, or the flip rate is not
        finite and below one half.
    """
    examples, positives = _examples(labels, p_gram)
    negatives = len(examples) - positives
    groups = _groups(examples)
    roc_auc = _area_under_curve(
        tuple((score, bool(label)) for label, score in examples)
    )

    if roc_auc is None:
        msg = "grammaticality ROC-AUC requires at least one label of each class"
        raise ValueError(msg)

    pr_auc, threshold, youden_accuracy = _threshold_metrics(
        groups, positives, negatives
    )
    half_accuracy, mcc = _at_half(examples)
    rate = _flip_rate(flip_rate)

    return GrammaticalityReport(
        n=len(examples),
        positives=positives,
        roc_auc=roc_auc,
        pr_auc=pr_auc,
        youden_threshold=threshold,
        accuracy_at_youden=youden_accuracy,
        accuracy_at_half=half_accuracy,
        mcc=mcc,
        noise_adjusted_auc=noise_adjusted_auc(roc_auc, flip_rate=rate),
        auc_ceiling=1.0 - rate,
        flip_rate=rate,
    )
