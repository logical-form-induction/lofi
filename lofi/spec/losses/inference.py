"""Inference-loss declaration."""

from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast, override

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.spec.components.loss import LossSpec

if TYPE_CHECKING:
    from lofi.spec.components.data import TrainingBatchKind

__all__ = [
    "InferenceLossSettings",
    "InferenceLossSpec",
    "InferenceResponseDeclaration",
    "InferenceResponseSpec",
    "LabelResponseSettings",
    "LabelResponseSpec",
    "ResponseSettingsDeclaration",
]


class InferenceResponseDeclaration(NamedTuple):
    """Resolved response tag and its immutable settings."""

    kind: str
    settings: object


class ResponseSettingsDeclaration(Protocol):
    """A response declaration variant that resolves interior settings."""

    def settings(self) -> object:
        """Return immutable response settings."""
        ...

    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the data capabilities this response consumes."""
        ...

    def grounding_query(self) -> Literal["observed-label", "full"]:
        """Return which proof questions the response can score."""
        ...


class InferenceResponseSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how observed responses supervise grounded label predictions."""


class LabelResponseSettings(NamedTuple):
    """Resolved categorical-label response settings."""


class LabelResponseSpec(InferenceResponseSpec):
    """Declare categorical inference label supervision."""

    kind: Literal["label"] = "label"

    def settings(self) -> LabelResponseSettings:
        """Return the empty categorical-label settings."""
        return LabelResponseSettings()

    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return that categorical labels need no capability beyond pairs."""
        return frozenset()

    def grounding_query(self) -> Literal["observed-label"]:
        """Return that label training may omit the label-opposed question."""
        return "observed-label"


class InferenceLossSettings(NamedTuple):
    """Settings for one response-supervised inference objective."""

    signal: Literal["likelihood", "reward"]
    response: InferenceResponseDeclaration
    auxiliary_weight: float


class InferenceLossSpec(LossSpec):
    """Declare a loss over the model's grounded inference distribution."""

    kind: Literal["inference"] = "inference"
    signal: Literal["likelihood", "reward"]
    response: InferenceResponseSpec
    auxiliary_weight: float = 0.0

    @dx.validates("auxiliary_weight")
    @classmethod
    def _auxiliary_weight_is_nonnegative(cls, value: float) -> float:
        held = ensure_finite("an inference auxiliary weight", value)

        if held < 0.0:
            msg = f"an inference auxiliary weight is non-negative, not {value!r}"
            raise ValueError(msg)

        return held

    def settings(self) -> InferenceLossSettings:
        """Return the settings the inference objective is built from."""
        return InferenceLossSettings(
            signal=self.signal,
            response=InferenceResponseDeclaration(
                cast("str", self.response.kind),
                cast("ResponseSettingsDeclaration", self.response).settings(),
            ),
            auxiliary_weight=self.auxiliary_weight,
        )

    @override
    def grounding_signal(self) -> str:
        """Return the training signal this inference objective consumes."""
        return self.signal

    @override
    def grounding_query(self) -> Literal["observed-label", "full"]:
        """Return the question policy owned by the declared response."""
        return cast("ResponseSettingsDeclaration", self.response).grounding_query()

    @override
    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the data capability required by the declared response."""
        return cast(
            "ResponseSettingsDeclaration", self.response
        ).required_training_batches()
