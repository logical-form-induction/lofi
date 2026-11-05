import pytest

from lofi.data.synthetic.spec._splits import (
    split_assignments,
    training_occurrences,
    validate_retained,
    validate_training_split,
)


def test_named_retained_allocations_partition_occurrences_deterministically() -> None:
    assignments = split_assignments(
        20, {"test": 0.2, "dev": 0.2}, 7, training_split="train"
    )

    assert assignments.count("train") == 12
    assert assignments.count("dev") == 4
    assert assignments.count("test") == 4
    assert assignments == split_assignments(
        20, {"dev": 0.2, "test": 0.2}, 7, training_split="train"
    )
    assert assignments != split_assignments(
        20, {"dev": 0.2, "test": 0.2}, 8, training_split="train"
    )
    assert training_occurrences(20, {"dev": 0.2, "test": 0.2}) == 12


def test_a_fraction_too_small_for_one_item_leaves_that_item_for_training() -> None:
    assert split_assignments(1, {"validation": 0.25}, 0, training_split="train") == (
        "train",
    )
    assert training_occurrences(1, {"validation": 0.25}) == 1


@pytest.mark.parametrize(
    "retained",
    [{}, {"": 0.2}, {"dev": 0.0}, {"dev": 1.0}, {"dev": 0.5, "test": 0.5}],
)
def test_invalid_retained_allocations_are_refused(retained: dict[str, float]) -> None:
    with pytest.raises(ValueError, match="retained"):
        validate_retained(retained)


def test_occurrences_not_retained_take_the_declared_training_split() -> None:
    assignments = split_assignments(10, {"dev": 0.3}, 4, training_split="fit")

    assert assignments.count("fit") == 7
    assert assignments.count("dev") == 3
    assert "train" not in assignments


def test_a_retained_split_may_not_share_the_training_split_name() -> None:
    with pytest.raises(ValueError, match="is the training split"):
        validate_training_split("fit", {"fit": 0.2})

    assert validate_training_split("fit", {"dev": 0.2}) == "fit"
