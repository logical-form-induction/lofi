import didactic.api as dx
import pytest
from tests.unit.spec.components.conftest import SampleDataset, WindowFeaturizer, minimal

from lofi.spec.components.data import (
    DataSection,
    DatasetSection,
    HeldOutDatasetSpec,
    InferenceLabeledDatasetSpec,
    RetainedPairSideDerivationsSpec,
    TrainingAllocationSpec,
)
from lofi.spec.constraint_policy import (
    AmbiguityMode,
    ConstraintPolicySection,
    ConstraintPolicySpec,
    QuantileSelectorSpec,
)


def test_an_existing_data_document_gets_two_neutral_constraint_policies() -> None:
    data = minimal().data

    assert data.datasets["train"].constraint_policy == ConstraintPolicySection()
    assert data.datasets["train"].constraint_policy.syntactic.coverage == 0.0
    assert data.datasets["train"].constraint_policy.semantic.coverage == 0.0
    assert (
        data.datasets["train"].constraint_policy.syntactic.ambiguity
        is AmbiguityMode.UNION
    )
    assert (
        data.datasets["train"].constraint_policy.semantic.ambiguity
        is AmbiguityMode.UNION
    )


def test_syntactic_and_semantic_constraint_policies_are_independent_slots() -> None:
    syntactic = ConstraintPolicySpec(coverage=1.0)
    semantic = ConstraintPolicySpec(
        ambiguity=AmbiguityMode.MAJORITY,
        selector=QuantileSelectorSpec(share=0.25),
        coverage=0.5,
        seed=9,
    )
    data = DataSection(
        datasets={
            "train": DatasetSection(
                dataset=SampleDataset(),
                featurizer=WindowFeaturizer(),
                constraint_policy=ConstraintPolicySection(
                    syntactic=syntactic, semantic=semantic
                ),
            )
        }
    )

    assert data.datasets["train"].constraint_policy.syntactic == syntactic
    assert data.datasets["train"].constraint_policy.semantic == semantic
    assert (
        data.datasets["train"].constraint_policy.syntactic
        != data.datasets["train"].constraint_policy.semantic
    )


def test_a_dataset_variant_that_declares_no_batches_yields_none() -> None:
    assert SampleDataset().training_batches() == frozenset()


class _HeldOutDataset:
    def held_out_split(self) -> str:
        return "validation"


def test_a_held_out_split_is_an_explicit_dataset_capability() -> None:
    assert isinstance(_HeldOutDataset(), HeldOutDatasetSpec)
    assert not isinstance(SampleDataset(), HeldOutDatasetSpec)


class _InferenceLabeledDataset:
    def inference_labels(self) -> tuple[str, ...]:
        return ("entails", "non-entails")


def test_an_inference_label_inventory_is_an_explicit_dataset_capability() -> None:
    assert isinstance(_InferenceLabeledDataset(), InferenceLabeledDatasetSpec)
    assert not isinstance(SampleDataset(), InferenceLabeledDatasetSpec)


class _PairSideDerivations:
    def retained_splits(self) -> frozenset[str]:
        return frozenset(("dev", "test"))

    def training_occurrences(self, total: int, /) -> int:
        return total - 2

    def retained_pair_side_derivation_splits(self) -> frozenset[str]:
        return self.retained_splits()


def test_retained_pair_side_derivations_are_an_explicit_dataset_capability() -> None:
    assert isinstance(_PairSideDerivations(), RetainedPairSideDerivationsSpec)
    assert isinstance(_PairSideDerivations(), TrainingAllocationSpec)
    assert not isinstance(SampleDataset(), RetainedPairSideDerivationsSpec)


def test_a_dataset_section_declares_its_reserved_forms_with_defaults() -> None:
    section = DatasetSection(dataset=SampleDataset())

    assert section.reserved_forms == ("<pad>", "<unk>")
    assert section.unknown_form == "<unk>"

    declared = DatasetSection(
        dataset=SampleDataset(),
        reserved_forms=("<p>", "<bos>", "<u>"),
        unknown_form="<u>",
    )

    assert declared.reserved_forms == ("<p>", "<bos>", "<u>")
    assert declared.unknown_form == "<u>"


def test_a_dataset_section_refuses_an_unreserved_unknown_form() -> None:
    with pytest.raises(dx.ValidationError, match="not one of the reserved forms"):
        DatasetSection(dataset=SampleDataset(), unknown_form="<missing>")


def test_a_dataset_section_refuses_a_repeated_reserved_form() -> None:
    with pytest.raises(dx.ValidationError, match="names one reserved form twice"):
        DatasetSection(dataset=SampleDataset(), reserved_forms=("<unk>", "<unk>"))


def test_a_data_section_without_a_label_inventory_refuses_to_name_one() -> None:
    data = DataSection(datasets={"train": DatasetSection(dataset=SampleDataset())})

    with pytest.raises(ValueError, match="one data-owned label inventory"):
        data.inference_labels()
