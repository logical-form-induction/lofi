"""Held-out inference label evaluator declaration."""

from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Self,
    override,
)

import didactic.api as dx

from lofi._checks import ensure_name, ensure_positive
from lofi.lf import RELATION_LABELS
from lofi.spec.components.run import EvaluationData, EvaluatorSpec, IntervalSettings
from lofi.spec.evaluators.conventions import ConventionSettings, ConventionSpec
from lofi.spec.provers import ProofBudgetSpec

if TYPE_CHECKING:
    from lofi.lf.provers import ProofBudget

__all__ = [
    "InferenceEvaluatorSettings",
    "InferenceEvaluatorSpec",
    "InferencePartitions",
    "InferencePartitionsSpec",
    "InferenceProofSettings",
    "InferenceProofSpec",
    "LabelReductionSpec",
    "inference_evaluation_metrics",
]


class InferencePartitions(NamedTuple):
    """Exact development and test partition names."""

    fit_on: str
    score_on: str


class InferencePartitionsSpec(dx.Model, extra="forbid"):
    """Declare disjoint control fitting and scoring partitions."""

    fit_on: str
    score_on: str

    @dx.model_validator()
    def _partitions_are_disjoint(self) -> Self:
        ensure_name("an inference fitting partition", self.fit_on)
        ensure_name("an inference scoring partition", self.score_on)
        if self.fit_on == self.score_on:
            msg = "inference fitting and scoring partitions are disjoint"
            raise ValueError(msg)
        return self

    def settings(self) -> InferencePartitions:
        """Return the immutable partition selection."""
        return InferencePartitions(self.fit_on, self.score_on)


class InferenceProofSettings(NamedTuple):
    """Budget and joint beam depth of proof evaluation."""

    budget: ProofBudget
    depth: int


class InferenceProofSpec(dx.Model, extra="forbid"):
    """Evaluate published forms with the interpretation's shared proof service."""

    budget: ProofBudgetSpec
    depth: int

    @dx.validates("depth")
    @classmethod
    def _depth_is_positive(cls, value: int) -> int:
        return ensure_positive("an inference proof beam depth", value)

    def settings(self) -> InferenceProofSettings:
        """Return the bounded proof treatment."""
        return InferenceProofSettings(self.budget.build(), self.depth)


def inference_evaluation_metrics(classes: tuple[str, ...], /) -> tuple[str, ...]:
    """Return the inference metrics reported for a declared class inventory."""
    count = len(classes)

    if count < 2:
        msg = f"an inference evaluation needs at least two classes, not {count}"
        raise ValueError(msg)

    prefix = f"inference-label-{count}-way"

    return (
        prefix,
        f"{prefix}-macro-f1",
        *(f"{prefix}-{label}-precision" for label in classes),
        *(f"{prefix}-{label}-recall" for label in classes),
        *(f"{prefix}-{label}-f1" for label in classes),
        f"{prefix}-abstention",
    )


class LabelReductionSpec(dx.Model, extra="forbid"):
    """Declare a total two-way coarsening of the inference label scale."""

    mapping: dict[str, str]

    @dx.validates("mapping")
    @classmethod
    def _mapping_is_total_and_two_way(cls, mapping: dict[str, str]) -> dict[str, str]:
        """Return a named coarsening whose image contains exactly two classes."""
        held = {
            ensure_name("an inference reduction source", source): ensure_name(
                f"the class {source!r} reduces to", target
            )
            for source, target in mapping.items()
        }
        images = frozenset(held.values())

        if len(images) != 2:
            msg = f"an inference reduction leaves {len(images)} classes, not two"
            raise ValueError(msg)

        if len(held) <= len(images):
            msg = "an inference reduction must coarsen its declared source labels"
            raise ValueError(msg)

        return dict(sorted(held.items()))

    def settings(self) -> tuple[tuple[str, str], ...]:
        """Return the reduction as immutable source-target pairs."""
        return tuple(self.mapping.items())


class InferenceEvaluatorSettings(NamedTuple):
    """Validated settings used to build an inference label evaluator."""

    conventions: ConventionSettings
    reduction: tuple[tuple[str, str], ...] | None
    partitions: InferencePartitions
    bins: int
    proof: InferenceProofSettings | None
    interval: IntervalSettings


class InferenceEvaluatorSpec(EvaluatorSpec):
    """Declare inference label evaluation over held-out sentence pairs.

    ``reduction`` is required. A null value reports the data's native scale; a
    mapping explicitly declares a two-way coarsening of that scale.
    """

    kind: Literal["inference"] = "inference"
    conventions: ConventionSpec
    reduction: LabelReductionSpec | None
    partitions: InferencePartitionsSpec
    bins: int
    proof: InferenceProofSpec | None

    @dx.validates("bins")
    @classmethod
    def _bins_are_positive(cls, value: int) -> int:
        return ensure_positive("an inference calibration bin count", value)

    def classes(self, labels: tuple[str, ...] | None = None, /) -> tuple[str, ...]:
        """Return scored classes for a data-owned native label inventory."""
        native = (
            tuple(sorted(label.value for label in RELATION_LABELS))
            if labels is None
            else labels
        )
        reduction = self.reduction

        if reduction is None:
            return native

        declared = frozenset(native)
        reduced = frozenset(reduction.mapping)

        if reduced != declared:
            msg = (
                f"an inference reduction maps {sorted(reduced)}, not the data's "
                f"label inventory {sorted(declared)}"
            )
            raise ValueError(msg)

        return tuple(sorted(set(reduction.mapping.values())))

    @override
    def fit_split(self) -> str:
        """Return the exact retained partition used to fit controls."""
        return self.partitions.fit_on

    @override
    def score_split(self) -> str:
        """Return the exact retained partition used for scoring."""
        return self.partitions.score_on

    @override
    def evaluation_data(self) -> EvaluationData:
        """Return that the evaluator scores labeled inference pairs."""
        return EvaluationData.INFERENCE_PAIRS

    @override
    def metrics(self, labels: tuple[str, ...] = (), /) -> tuple[str, ...]:
        """Return rows for the data's label inventory.

        Parameters
        ----------
        labels : tuple[str, ...]
            The inference label inventory the run's data owns; empty reads the
            canonical logical-relation response scale.
        """
        return self._metrics(self.classes(labels or None))

    def _metrics(self, classes: tuple[str, ...], /) -> tuple[str, ...]:
        """Return accuracy, macro-F1, and every per-class P/R/F1 row."""
        proof = self.proof
        return (
            *inference_evaluation_metrics(classes),
            "entailment-coverage",
            "entailment-calibration-ece",
            "entailment-calibration-brier",
            *(
                ()
                if proof is None
                else (
                    "entailment-preserving-rate",
                    f"entailment-preserving-rate-at-{proof.depth}",
                    "entailment-preserving-coverage",
                    "entailment-preserving-abstention",
                    "entailment-preserving-both",
                    "relaxation-gap",
                    "proof-calls",
                    "proof-memo-hits",
                    "proof-cpu-seconds",
                    "proof-questions",
                    "proof-wall-seconds",
                    "proof-abstentions",
                )
            ),
            "majority-class",
            "hypothesis-only",
        )

    def settings(self) -> InferenceEvaluatorSettings:
        """Return the settings the evaluator plugin consumes."""
        return InferenceEvaluatorSettings(
            conventions=self.conventions.settings(),
            reduction=None if self.reduction is None else self.reduction.settings(),
            partitions=self.partitions.settings(),
            bins=self.bins,
            proof=None if self.proof is None else self.proof.settings(),
            interval=self.interval.settings(),
        )
