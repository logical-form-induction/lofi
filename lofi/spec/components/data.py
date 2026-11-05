"""The data section of a run and the component families it declares."""

from enum import StrEnum
from typing import TYPE_CHECKING, Final, Protocol, Self, runtime_checkable

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec.components.base import ComponentSpec
from lofi.spec.constraint_policy import ConstraintPolicySection, PolicyDomain

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.spec.language import LanguageDecl

TRAINING_SPLIT: Final = "train"
"""The split a dataset assigns every occurrence it does not retain."""

_DEFAULT_FEATURIZER_PADDING: Final = "<pad>"
_DEFAULT_UNKNOWN_FORM: Final = "<unk>"
_DEFAULT_RESERVED_FORMS: Final = (_DEFAULT_FEATURIZER_PADDING, _DEFAULT_UNKNOWN_FORM)
_DEFAULT_VALIDATION_SPLIT: Final = "validation"

__all__ = [
    "TRAINING_SPLIT",
    "ConstraintSourceSpec",
    "DataSection",
    "DatasetSection",
    "DatasetSpec",
    "FeaturizerSpec",
    "GoldAggregationSpec",
    "HeldOutDatasetSpec",
    "InferenceLabeledDatasetSpec",
    "LearnerLanguageDatasetSpec",
    "RetainedDatasetSpec",
    "RetainedPairSideDerivationsSpec",
    "TokenizerSpec",
    "TrainingAllocationSpec",
    "TrainingBatchKind",
]


class TrainingBatchKind(StrEnum):
    """A kind of training batch a dataset's lowering yields to a trainer."""

    #: Sentences drawn from a compiled grammar, lowered with that grammar's
    #: lexicon, codec and type store.
    SENTENCES = "sentences"
    #: Premise and hypothesis pairs, each labeled with an inference judgment.
    INFERENCE_PAIRS = "inference-pairs"
    #: Certified logical forms aligned to the surface items they interpret.
    GOLD_FORMS = "gold-forms"
    #: Ordinal responses aligned to premise-hypothesis pairs.
    JUDGMENTS = "judgments"


@runtime_checkable
class HeldOutDatasetSpec(Protocol):
    """A dataset declaration that names the split withheld during fitting."""

    def held_out_split(self) -> str:
        """Return the validated held-out split name."""
        ...


@runtime_checkable
class RetainedDatasetSpec(Protocol):
    """A dataset declaration assigning occurrences to named retained splits."""

    def retained_splits(self) -> frozenset[str]:
        """Return the exact names of the retained splits."""
        ...


@runtime_checkable
class TrainingAllocationSpec(Protocol):
    """A dataset declaration deriving its training count from the total count."""

    def training_occurrences(self, total: int, /) -> int:
        """Return occurrences left for fitting after exact split allocation."""
        ...


@runtime_checkable
class RetainedPairSideDerivationsSpec(RetainedDatasetSpec, Protocol):
    """A pair dataset retaining gold derivations for both sides of each pair."""

    def retained_pair_side_derivation_splits(self) -> frozenset[str]:
        """Return exact splits whose pair sides retain gold derivations."""
        ...


@runtime_checkable
class InferenceLabeledDatasetSpec(Protocol):
    """A pair dataset that owns an ordered inference-label vocabulary."""

    def inference_labels(self) -> tuple[str, ...]:
        """Return output labels in the model-column order declared by the data."""
        ...


@runtime_checkable
class LearnerLanguageDatasetSpec(Protocol):
    """A dataset declaration that supplies the model's learner language."""

    def learner_language(self) -> LanguageDecl:
        """Return the language shared by the dataset's source grammar."""
        ...


class DatasetSpec(ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declares a dataset."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists dataset plugins."""
        return "dataset"

    def observed_occurrences(self) -> int | None:
        """Return how many item occurrences the declaration fixes the dataset to hold.

        Returns
        -------
        int | None
            The count, or None where it is known only once the dataset is read.
        """
        return None

    def with_derived_seeds(self, derive: Callable[[str], int], path: str, /) -> Self:
        """Return the dataset with each derived data seed replaced by the one derived.

        A dataset declaring no derived seed is returned as it is.

        Parameters
        ----------
        derive : Callable[[str], int]
            The seed a point derives for a dotted path.
        path : str
            The dotted path of the dataset in the run document.
        """
        del derive, path

        return self

    def training_batches(self) -> frozenset[TrainingBatchKind]:
        """Return the kinds of training batch the dataset's lowering yields.

        A variant that declares none yields no batch a trainer can consume.
        """
        return frozenset()


class FeaturizerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a featurizer."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "featurizer"


class TokenizerSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares how a collected corpus is segmented into tokens."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists tokenizers."""
        return "tokenizer"


class GoldAggregationSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares how repeated corpus responses determine one gold response."""

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists gold aggregators."""
        return "gold_aggregation"


class ConstraintSourceSpec(
    ComponentSpec, dx.TaggedUnion, discriminator="kind", extra="forbid"
):
    """Declares a constraint source and the label domain it serves.

    The run selects the source's constraints under the policy its data section
    declares for that domain.
    """

    policy: PolicyDomain

    @classmethod
    def family(cls) -> str:
        """Return the family whose entry-point group lists the plugins."""
        return "constraint_source"


class DatasetSection(dx.Model, extra="forbid"):
    """Declare one named training dataset and its input transformation.

    Parameters
    ----------
    dataset : DatasetSpec
        Dataset component owned by this role.
    featurizer : FeaturizerSpec | None
        Optional featurizer component.
    constraint_source : ConstraintSourceSpec | None
        Optional source of selected parser span constraints.
    constraint_policy : ConstraintPolicySection
        Independent syntactic and semantic constraint-selection policies.
    reserved_forms : tuple[str, ...]
        The token forms reserved ahead of the dataset's own forms, in index
        order. A featurizer pads with one of them.
    unknown_form : str
        The reserved form a token outside the dataset's vocabulary is read as.
    """

    dataset: DatasetSpec
    featurizer: FeaturizerSpec | None = None
    constraint_source: ConstraintSourceSpec | None = None
    constraint_policy: ConstraintPolicySection = dx.field(
        default_factory=ConstraintPolicySection
    )
    reserved_forms: tuple[str, ...] = _DEFAULT_RESERVED_FORMS
    unknown_form: str = _DEFAULT_UNKNOWN_FORM

    @dx.validates("reserved_forms")
    @classmethod
    def _reserved_forms_are_distinct_names(
        cls, values: tuple[str, ...]
    ) -> tuple[str, ...]:
        """Return named reserved forms, each declared once."""
        held = tuple(ensure_name("a reserved form", value) for value in values)

        if len(set(held)) != len(held):
            msg = f"reserved_forms names one reserved form twice: {held!r}"
            raise ValueError(msg)

        return held

    @dx.model_validator()
    def _unknown_form_is_reserved(self) -> Self:
        """Require the unknown-token form to be one of the reserved forms."""
        if self.unknown_form not in self.reserved_forms:
            msg = (
                f"unknown_form {self.unknown_form!r} is not one of the reserved "
                f"forms {self.reserved_forms!r}"
            )
            raise ValueError(msg)

        return self


class DataSection(dx.Model, extra="forbid"):
    """Declare the named datasets one run may train across."""

    datasets: dict[str, DatasetSection]

    @dx.validates("datasets")
    @classmethod
    def _datasets_are_named(
        cls, values: dict[str, DatasetSection]
    ) -> dict[str, DatasetSection]:
        """Validate one nonempty, name-ordered dataset mapping."""
        if not values:
            msg = "data.datasets declares at least one training dataset"
            raise ValueError(msg)

        return {
            ensure_name("a training-dataset role", name): values[name]
            for name in sorted(values)
        }

    def inference_labels(self) -> tuple[str, ...]:
        """Return the one inference-label inventory the run's datasets own.

        Returns
        -------
        tuple[str, ...]
            The labels in the model-column order the data declares.

        Raises
        ------
        ValueError
            If no dataset, or datasets declaring different inventories, own one.
        """
        inventories = {
            dataset.inference_labels()
            for section in self.datasets.values()
            if isinstance((dataset := section.dataset), InferenceLabeledDatasetSpec)
        }

        if len(inventories) != 1:
            msg = (
                "an inference evaluator needs one data-owned label inventory, not "
                f"{tuple(sorted(inventories))!r}"
            )
            raise ValueError(msg)

        return next(iter(inventories))
