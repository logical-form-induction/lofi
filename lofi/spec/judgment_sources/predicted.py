"""Declaration of judgments supplied by a trained LoFI point."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_name

if TYPE_CHECKING:
    from collections.abc import Mapping

__all__ = [
    "PredictedJudgmentSourceSettings",
    "PredictedJudgmentSourceSpec",
    "decode_predicted_judgment_source",
]


class PredictedJudgmentSourceSettings(NamedTuple):
    """Resolved producer, items, inputs, factors and response space."""

    producer_run: str
    producer_point: str
    item_set: str
    inputs: Mapping[str, str]
    factors: Mapping[str, str]
    question: str
    labels: tuple[str, ...]
    participant: str


class PredictedJudgmentSourceSpec(dx.Model, extra="forbid"):
    """Name a trained point and the items whose predictions are judgments.

    Parameters
    ----------
    producer_run : str
        The run holding the producer point.
    producer_point : str
        The producer point.
    item_set : str
        The judgment item construction document the items are built from,
        relative to the directory holding this document.
    inputs : dict[str, str]
        The rendered element of every item each input role the point publishes
        reads, keyed by role.
    factors : dict[str, str]
        The bead expression each factor every row records is read by over the
        item's attributes, keyed by the factor's name.
    question : str
        The response space the predictions answer.
    labels : tuple[str, ...]
        The response labels in model order.
    participant : str
        The participant identity every predicted row is recorded under.
    """

    kind: Literal["predicted"] = "predicted"
    producer_run: str
    producer_point: str
    item_set: str
    inputs: dict[str, str]
    factors: dict[str, str]
    question: str
    labels: tuple[str, ...]
    participant: str = "lofi-prediction"

    @dx.validates(
        "producer_run", "producer_point", "item_set", "question", "participant"
    )
    @classmethod
    def _identities_are_named(cls, value: str) -> str:
        return ensure_name("a predicted-judgment identity", value)

    @dx.validates("inputs", "factors")
    @classmethod
    def _mappings_are_named(cls, value: dict[str, str]) -> dict[str, str]:
        return {
            ensure_name("a predicted-source name", name): ensure_name(
                f"{name!r}'s declared value", declared
            )
            for name, declared in value.items()
        }

    @dx.validates("labels")
    @classmethod
    def _labels_are_a_response_space(cls, labels: tuple[str, ...]) -> tuple[str, ...]:
        held = tuple(
            ensure_name("a predicted response label", label) for label in labels
        )
        if len(held) < 2 or len(set(held)) != len(held):
            msg = f"predicted judgments need at least two distinct labels: {held!r}"
            raise ValueError(msg)
        return held

    def settings(self) -> PredictedJudgmentSourceSettings:
        """Discard the boundary declaration into immutable settings."""
        return PredictedJudgmentSourceSettings(
            self.producer_run,
            self.producer_point,
            self.item_set,
            MappingProxyType(dict(self.inputs)),
            MappingProxyType(dict(self.factors)),
            self.question,
            self.labels,
            self.participant,
        )


def decode_predicted_judgment_source(data: bytes, /) -> PredictedJudgmentSourceSpec:
    """Decode and validate one predicted-judgment source document."""
    return PredictedJudgmentSourceSpec.model_validate_json(data)
