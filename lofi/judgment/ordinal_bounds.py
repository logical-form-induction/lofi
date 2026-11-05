"""Deterministic comparison bounds over ordinal judgment responses."""

import math
from collections import Counter
from enum import StrEnum
from typing import TYPE_CHECKING, final, override

from lofi._frozen import FrozenValue
from lofi.judgment._design import design_of
from lofi.judgment.agreement import rank_correlation

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judgment._design import JudgmentDesign
    from lofi.judgment._ports import JudgmentSource

__all__ = [
    "OrdinalBoundKind",
    "OrdinalBoundPrediction",
    "OrdinalBoundResult",
    "majority_class_bound",
    "modal_response_oracle",
]

_MINIMUM_CORRELATION_ROWS = 3
"""The minimum response count for a rank correlation."""


class OrdinalBoundKind(StrEnum):
    """The fixed interpretation of an ordinal comparison bound."""

    #: One distribution estimated from fitting responses alone.
    MAJORITY_CLASS = "majority-class"

    #: Per-item distributions estimated from the complete scored table.
    MODAL_RESPONSE_ORACLE = "modal-response-oracle"


@final
class OrdinalBoundPrediction(FrozenValue):
    """One scored row's deterministic ordinal-bound prediction.

    Instances are produced by the bound functions so the predicted level and
    expected level cannot disagree with the probability vector.
    """

    __slots__ = (
        "_distribution",
        "_expected_level",
        "_item",
        "_participant",
        "_predicted_level",
    )

    _noun = "ordinal-bound prediction"

    def __init__(self) -> None:
        """Prevent construction apart from a computed bound."""
        msg = "an ordinal-bound prediction is computed from judgment responses"

        raise TypeError(msg)

    @property
    def participant(self) -> str:
        """Return the participant whose response is predicted."""
        return self._participant

    @property
    def item(self) -> str:
        """Return the item whose response is predicted."""
        return self._item

    @property
    def distribution(self) -> tuple[float, ...]:
        """Return response probabilities in declared ordinal level order."""
        return self._distribution

    @property
    def predicted_level(self) -> int:
        """Return the first highest-probability level in declared order."""
        return self._predicted_level

    @property
    def expected_level(self) -> float:
        """Return the probability-weighted zero-based response level."""
        return self._expected_level

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another prediction has the same fields."""
        if type(other) is not OrdinalBoundPrediction:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the prediction fields."""
        return hash((OrdinalBoundPrediction, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the row identity, predicted level and probability vector."""
        return (
            f"OrdinalBoundPrediction(participant={self._participant!r}, "
            f"item={self._item!r}, predicted_level={self._predicted_level!r}, "
            f"distribution={self._distribution!r})"
        )

    def _identity(self) -> tuple[str, str, tuple[float, ...], int, float]:
        """Return the fields in comparison order."""
        return (
            self._participant,
            self._item,
            self._distribution,
            self._predicted_level,
            self._expected_level,
        )


@final
class OrdinalBoundResult(FrozenValue):
    """Computed predictions and summaries for one ordinal comparison bound.

    Instances are produced by the bound functions so the summary values cannot
    be supplied independently of the predictions and recorded responses.
    """

    __slots__ = (
        "_accuracy",
        "_kind",
        "_loss",
        "_predictions",
        "_rank_correlation",
    )

    _noun = "ordinal-bound result"

    def __init__(self) -> None:
        """Prevent construction apart from a computed bound."""
        msg = "an ordinal-bound result is computed from judgment responses"

        raise TypeError(msg)

    @property
    def kind(self) -> OrdinalBoundKind:
        """Return the fixed interpretation of this bound."""
        return self._kind

    @property
    def predictions(self) -> tuple[OrdinalBoundPrediction, ...]:
        """Return predictions in scored-row order."""
        return self._predictions

    @property
    def loss(self) -> float:
        """Return mean categorical negative log probability."""
        return self._loss

    @property
    def accuracy(self) -> float:
        """Return the fraction whose predicted level equals the response."""
        return self._accuracy

    @property
    def rank_correlation(self) -> float | None:
        """Return expected-level Spearman correlation, when defined."""
        return self._rank_correlation

    @property
    def rows(self) -> int:
        """Return the number of scored response rows."""
        return len(self._predictions)

    @property
    def response_levels(self) -> int:
        """Return the number of ordered response levels."""
        return len(self._predictions[0].distribution)

    @override
    def __repr__(self) -> str:
        """Return the bound kind, row count and summaries."""
        return (
            f"OrdinalBoundResult(kind={self._kind!r}, rows={self.rows}, "
            f"loss={self._loss!r}, accuracy={self._accuracy!r}, "
            f"rank_correlation={self._rank_correlation!r})"
        )


def _nonempty(source: JudgmentSource, role: str) -> JudgmentDesign:
    """Return a source's validated nonempty design.

    Parameters
    ----------
    source : JudgmentSource
        Source whose rows are encoded and validated.
    role : str
        Source role used in an empty-input error.

    Raises
    ------
    ValueError
        If the source has no response row.
    """
    design = design_of(source)

    if design.n_rows == 0:
        msg = f"{role} needs at least one response row"

        raise ValueError(msg)

    return design


def _same_scale(
    reference: JudgmentDesign,
    scored: JudgmentDesign,
    *,
    role: str,
) -> None:
    """Refuse a scored design over another ordinal response scale.

    Parameters
    ----------
    reference : JudgmentDesign
        Design whose response scale determines the predictions.
    scored : JudgmentDesign
        Design whose rows are scored.
    role : str
        Reference role used in the error message.

    Raises
    ------
    ValueError
        If the two level counts differ.
    """
    if reference.response_levels != scored.response_levels:
        msg = (
            f"{role} has {reference.response_levels} response levels and the "
            f"scored rows have {scored.response_levels}"
        )

        raise ValueError(msg)


def _distribution(responses: Sequence[int], levels: int) -> tuple[float, ...]:
    """Return empirical response probabilities in declared level order.

    Parameters
    ----------
    responses : Sequence[int]
        Validated zero-based responses.
    levels : int
        Number of ordered response levels.
    """
    counts = [0] * levels

    for response in responses:
        counts[response] += 1

    total = len(responses)

    return tuple(count / total for count in counts)


def _prediction(
    participant: str,
    item: str,
    distribution: tuple[float, ...],
) -> OrdinalBoundPrediction:
    """Return a prediction derived from an empirical distribution.

    Parameters
    ----------
    participant : str
        Participant whose response is predicted.
    item : str
        Item whose response is predicted.
    distribution : tuple[float, ...]
        Response probabilities in declared level order.
    """
    highest = max(distribution)
    predicted = distribution.index(highest)
    expected = math.fsum(
        level * probability for level, probability in enumerate(distribution)
    )
    built = object.__new__(OrdinalBoundPrediction)
    object.__setattr__(built, "_participant", participant)
    object.__setattr__(built, "_item", item)
    object.__setattr__(built, "_distribution", distribution)
    object.__setattr__(built, "_predicted_level", predicted)
    object.__setattr__(built, "_expected_level", expected)

    return built


def _result(
    kind: OrdinalBoundKind,
    predictions: tuple[OrdinalBoundPrediction, ...],
    responses: Sequence[int],
) -> OrdinalBoundResult:
    """Return summaries computed from predictions and recorded responses.

    Parameters
    ----------
    kind : OrdinalBoundKind
        Fixed interpretation of the computed bound.
    predictions : tuple[OrdinalBoundPrediction, ...]
        Predictions in scored-row order.
    responses : Sequence[int]
        Recorded responses in the same order.
    """
    losses = tuple(
        (
            math.inf
            if prediction.distribution[response] == 0.0
            else -math.log(prediction.distribution[response])
        )
        for prediction, response in zip(predictions, responses, strict=True)
    )
    accuracy = sum(
        prediction.predicted_level == response
        for prediction, response in zip(predictions, responses, strict=True)
    ) / len(responses)
    expected = tuple(prediction.expected_level for prediction in predictions)
    held_responses = tuple(responses)
    correlation = None

    if (
        len(held_responses) >= _MINIMUM_CORRELATION_ROWS
        and len(frozenset(expected)) > 1
        and len(frozenset(held_responses)) > 1
    ):
        correlation = rank_correlation(expected, held_responses)

    built = object.__new__(OrdinalBoundResult)
    object.__setattr__(built, "_kind", kind)
    object.__setattr__(built, "_predictions", predictions)
    object.__setattr__(built, "_loss", math.fsum(losses) / len(losses))
    object.__setattr__(built, "_accuracy", accuracy)
    object.__setattr__(built, "_rank_correlation", correlation)

    return built


def _predictions(
    design: JudgmentDesign,
    distributions: dict[str, tuple[float, ...]],
) -> tuple[OrdinalBoundPrediction, ...]:
    """Return predictions by looking up each scored row's item.

    Parameters
    ----------
    design : JudgmentDesign
        Validated rows to predict.
    distributions : dict[str, tuple[float, ...]]
        Empirical response probabilities by item.

    Raises
    ------
    KeyError
        If a scored item is absent from the distributions.
    """
    participants = design.participants.tolist()
    items = design.items.tolist()
    built: list[OrdinalBoundPrediction] = []

    for participant, item in zip(participants, items, strict=True):
        item_id = design.item_levels[item]

        if item_id not in distributions:
            msg = f"the scored item {item_id!r} is absent from the oracle table"

            raise KeyError(msg)

        built.append(
            _prediction(
                design.participant_levels[participant],
                item_id,
                distributions[item_id],
            )
        )

    return tuple(built)


def _response_rows(
    design: JudgmentDesign,
) -> Counter[tuple[str, str, int]]:
    """Return participant, item and response rows with their multiplicities.

    Parameters
    ----------
    design : JudgmentDesign
        Validated response rows to count.
    """
    return Counter(
        (
            design.participant_levels[participant],
            design.item_levels[item],
            response,
        )
        for participant, item, response in zip(
            design.participants.tolist(),
            design.items.tolist(),
            design.responses.tolist(),
            strict=True,
        )
    )


def _require_scored_rows_in_table(
    table: JudgmentDesign,
    scored: JudgmentDesign,
) -> None:
    """Require every scored response occurrence to occur in the oracle table.

    Parameters
    ----------
    table : JudgmentDesign
        Complete scored response table.
    scored : JudgmentDesign
        Rows whose responses the oracle summarizes.

    Raises
    ------
    ValueError
        If a scored response occurrence is absent from the complete table.
    """
    missing = _response_rows(scored) - _response_rows(table)

    if missing:
        row, count = next(iter(missing.items()))
        participant, item, response = row
        occurrence = "occurrence" if count == 1 else "occurrences"
        msg = (
            f"the complete scored table is missing {count} {occurrence} of "
            f"participant {participant!r}, item {item!r}, response {response}"
        )

        raise ValueError(msg)


def majority_class_bound(
    fitting: JudgmentSource,
    scored: JudgmentSource,
    /,
) -> OrdinalBoundResult:
    """Return the fit-only global majority-class comparison bound.

    The empirical response distribution is estimated from fitting rows alone
    and applied unchanged to every scored row. Its first modal level is the
    class prediction.

    Parameters
    ----------
    fitting : JudgmentSource
        Responses used to estimate the global response distribution.
    scored : JudgmentSource
        Responses predicted and summarized without entering the estimate.

    Raises
    ------
    ValueError
        If either source is empty or the response scales differ.
    """
    fitted = _nonempty(fitting, "a majority-class fit")
    evaluated = _nonempty(scored, "a majority-class score")
    _same_scale(fitted, evaluated, role="the majority-class fit")
    distribution = _distribution(fitted.responses.tolist(), fitted.response_levels)
    predictions = tuple(
        _prediction(
            evaluated.participant_levels[participant],
            evaluated.item_levels[item],
            distribution,
        )
        for participant, item in zip(
            evaluated.participants.tolist(),
            evaluated.items.tolist(),
            strict=True,
        )
    )

    return _result(
        OrdinalBoundKind.MAJORITY_CLASS,
        predictions,
        evaluated.responses.tolist(),
    )


def modal_response_oracle(
    complete_scored_table: JudgmentSource,
    scored: JudgmentSource,
    /,
) -> OrdinalBoundResult:
    """Return the per-item modal-response oracle over a complete scored table.

    The oracle estimates each item's empirical response distribution from the
    complete table that includes the responses being scored. It is therefore
    labeled as an oracle and never as a fitted model.

    Parameters
    ----------
    complete_scored_table : JudgmentSource
        Complete responses used to estimate each item's distribution.
    scored : JudgmentSource
        Responses predicted and summarized by those item distributions.

    Raises
    ------
    KeyError
        If a scored item is absent from the complete table.
    ValueError
        If either source is empty, the response scales differ, or the oracle
        table reports a skipped response or omits a scored response occurrence.
    """
    skipped = sum(complete_scored_table.skips.values())

    if skipped != 0:
        msg = (
            "a modal-response oracle needs complete scored item groups and the "
            f"table reports {skipped} skipped responses"
        )

        raise ValueError(msg)

    table = _nonempty(complete_scored_table, "a modal-response oracle table")
    evaluated = _nonempty(scored, "a modal-response oracle score")
    _same_scale(table, evaluated, role="the modal-response oracle table")
    grouped: dict[str, list[int]] = {}

    for item, response in zip(
        table.items.tolist(), table.responses.tolist(), strict=True
    ):
        grouped.setdefault(table.item_levels[item], []).append(response)

    distributions = {
        item: _distribution(responses, table.response_levels)
        for item, responses in grouped.items()
    }
    predictions = _predictions(evaluated, distributions)
    _require_scored_rows_in_table(table, evaluated)

    return _result(
        OrdinalBoundKind.MODAL_RESPONSE_ORACLE,
        predictions,
        evaluated.responses.tolist(),
    )
