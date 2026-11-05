"""Ordinal-judgment response declarations for inference training."""

from math import fsum
from typing import Literal, NamedTuple, Self

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.lf import InferenceLabel
from lofi.spec.components.data import TrainingBatchKind
from lofi.spec.losses.inference import InferenceResponseSpec

__all__ = ["JudgmentResponseSettings", "JudgmentResponseSpec"]


class JudgmentResponseSettings(NamedTuple):
    """A response probability row for each inference label."""

    probabilities: tuple[tuple[InferenceLabel, tuple[float, ...]], ...]


class JudgmentResponseSpec(InferenceResponseSpec):
    """Declare an ordinal response instrument conditional on inference labels."""

    kind: Literal["judgment"] = "judgment"
    probabilities: dict[str, tuple[float, ...]]

    @dx.model_validator()
    def _rows_are_probability_distributions(self) -> Self:
        """Validate equally sized nonempty probability rows."""
        try:
            tuple(InferenceLabel(label) for label in self.probabilities)
        except ValueError as error:
            msg = f"a judgment instrument names an unknown inference label: {error}"
            raise ValueError(msg) from error

        rows = tuple(self.probabilities.values())
        widths = {len(row) for row in rows}

        if len(widths) != 1 or not widths or next(iter(widths)) < 2:
            msg = (
                "a judgment instrument has equally sized rows with at least two levels"
            )
            raise ValueError(msg)

        for row in rows:
            checked = tuple(ensure_finite("a judgment probability", one) for one in row)

            if (
                any(one < 0.0 for one in checked)
                or not abs(fsum(checked) - 1.0) <= 1e-6
            ):
                msg = (
                    f"a judgment instrument row is a probability distribution: {row!r}"
                )
                raise ValueError(msg)

        return self

    def settings(self) -> JudgmentResponseSettings:
        """Return response rows keyed by their declared inference labels."""
        return JudgmentResponseSettings(
            tuple(
                (InferenceLabel(label), probabilities)
                for label, probabilities in self.probabilities.items()
            )
        )

    def required_training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return that the response consumes aligned ordinal judgments."""
        return frozenset({TrainingBatchKind.JUDGMENTS})

    def grounding_query(self) -> Literal["full"]:
        """Return that judgments require both entailment questions."""
        return "full"
