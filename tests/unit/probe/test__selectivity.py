from typing import TYPE_CHECKING, Final

import pytest
import torch

from lofi.probe._selectivity import (
    ClassifierProbe,
    ControlTask,
    ProbeCapacity,
    ProbeResult,
    SpanInventory,
)

if TYPE_CHECKING:
    from torch import Tensor

CLASSES: Final = 4
FEATURES: Final = 8
VOCABULARY: Final = 6

CENTERS: Final = torch.randn(
    CLASSES, FEATURES, generator=torch.Generator().manual_seed(5), dtype=torch.float64
)

WORDS: Final = torch.randn(
    VOCABULARY,
    FEATURES,
    generator=torch.Generator().manual_seed(9),
    dtype=torch.float64,
)

# a probe wide enough and fitted long enough to memorize the labels it is given
HIGH: Final = ProbeCapacity(hidden=64, epochs=300, learning_rate=0.05, seed=3)

# a probe small enough that a law check running it six times stays quick
SMALL: Final = ProbeCapacity(hidden=8, epochs=40, learning_rate=0.1, seed=3)


def labels_of(spans: int, seed: int) -> list[int]:
    draws = torch.Generator().manual_seed(seed)

    return torch.randint(CLASSES, (spans,), generator=draws).tolist()


def noise(spans: int, seed: int) -> Tensor:
    draws = torch.Generator().manual_seed(seed + 1000)

    return torch.randn(spans, FEATURES, generator=draws, dtype=torch.float64)


def separable(spans: int, seed: int) -> SpanInventory:
    """Return spans whose vectors cluster by their label, each key its own."""
    labels = labels_of(spans, seed)
    vectors = CENTERS[labels] + 0.5 * noise(spans, seed)
    keys = [f"span-{seed}-{at}" for at in range(spans)]

    return SpanInventory(vectors, keys, labels, classes=CLASSES)


def scattered(spans: int, seed: int) -> SpanInventory:
    """Return spans whose vectors say nothing about their label, each key its own."""
    labels = labels_of(spans, seed)
    keys = [f"span-{seed}-{at}" for at in range(spans)]

    return SpanInventory(noise(spans, seed), keys, labels, classes=CLASSES)


def recurring(spans: int, seed: int) -> SpanInventory:
    """Return spans drawn from a small vocabulary, labeled by the word alone."""
    draws = torch.Generator().manual_seed(seed)
    which = torch.randint(VOCABULARY, (spans,), generator=draws).tolist()
    vectors = WORDS[which] + 0.1 * noise(spans, seed)
    keys = [f"word-{at}" for at in which]
    labels = [at % CLASSES for at in which]

    return SpanInventory(vectors, keys, labels, classes=CLASSES)


def small(spans: int, seed: int) -> SpanInventory:
    """Return a two-class inventory of three coordinates, for the cheap checks."""
    draws = torch.Generator().manual_seed(seed)
    labels = torch.randint(2, (spans,), generator=draws).tolist()
    vectors = torch.randn(spans, 3, generator=draws, dtype=torch.float64)
    keys = [f"span-{seed}-{at}" for at in range(spans)]

    return SpanInventory(vectors, keys, labels, classes=2)


def test_an_inventory_holds_its_vectors_keys_labels_and_classes() -> None:
    vectors = torch.zeros(3, 5, dtype=torch.float64)

    inventory = SpanInventory(vectors, ["a", "b", "a"], [0, 1, 2], classes=3)

    assert torch.equal(inventory.vectors, vectors)
    assert inventory.keys == ("a", "b", "a")
    assert inventory.labels == (0, 1, 2)
    assert inventory.classes == 3
    assert inventory.features == 5
    assert len(inventory) == 3


def test_an_inventory_refuses_vectors_that_are_not_a_floating_point_tensor() -> None:
    with pytest.raises(TypeError, match="floating point"):
        SpanInventory(
            torch.zeros(2, 3, dtype=torch.int64), ["a", "b"], [0, 1], classes=2
        )


def test_an_inventory_refuses_something_that_is_not_a_tensor() -> None:
    with pytest.raises(TypeError, match="floating point"):
        SpanInventory([[1.0, 2.0]], ["a"], [0], classes=2)  # ty: ignore[invalid-argument-type]


def test_an_inventory_refuses_vectors_that_are_not_rank_two() -> None:
    with pytest.raises(ValueError, match="one vector per span"):
        SpanInventory(
            torch.zeros(2, 3, 4, dtype=torch.float64), ["a", "b"], [0, 1], classes=2
        )


def test_an_inventory_refuses_keys_that_are_not_one_per_span() -> None:
    with pytest.raises(ValueError, match="2 keys"):
        SpanInventory(
            torch.zeros(3, 2, dtype=torch.float64), ["a", "b"], [0, 1, 0], classes=2
        )


def test_an_inventory_refuses_labels_that_are_not_one_per_span() -> None:
    with pytest.raises(ValueError, match="1 labels"):
        SpanInventory(
            torch.zeros(3, 2, dtype=torch.float64), ["a", "b", "c"], [0], classes=2
        )


def test_an_inventory_refuses_holding_no_span() -> None:
    with pytest.raises(ValueError, match="no span"):
        SpanInventory(torch.zeros(0, 2, dtype=torch.float64), [], [], classes=2)


def test_an_inventory_refuses_a_vector_of_no_coordinates() -> None:
    with pytest.raises(ValueError, match="no coordinates"):
        SpanInventory(
            torch.zeros(2, 0, dtype=torch.float64), ["a", "b"], [0, 1], classes=2
        )


def test_an_inventory_refuses_an_empty_key() -> None:
    with pytest.raises(ValueError, match="span key"):
        SpanInventory(
            torch.zeros(2, 2, dtype=torch.float64), ["a", ""], [0, 1], classes=2
        )


def test_an_inventory_refuses_a_label_outside_the_class_count() -> None:
    with pytest.raises(ValueError, match="labeled 2 among 2"):
        SpanInventory(
            torch.zeros(2, 2, dtype=torch.float64), ["a", "b"], [0, 2], classes=2
        )


def test_an_inventory_refuses_a_negative_label() -> None:
    with pytest.raises(ValueError, match="span label"):
        SpanInventory(
            torch.zeros(2, 2, dtype=torch.float64), ["a", "b"], [0, -1], classes=2
        )


def test_an_inventory_refuses_fewer_than_two_classes() -> None:
    with pytest.raises(ValueError, match="one class"):
        SpanInventory(
            torch.zeros(2, 2, dtype=torch.float64), ["a", "b"], [0, 0], classes=1
        )


def test_relabeling_keeps_the_vectors_the_keys_and_the_classes() -> None:
    inventory = small(4, 1)

    relabeled = inventory.relabeled([1, 0, 1, 0])

    assert torch.equal(relabeled.vectors, inventory.vectors)
    assert relabeled.keys == inventory.keys
    assert relabeled.classes == inventory.classes
    assert relabeled.labels == (1, 0, 1, 0)


def test_relabeling_refuses_the_wrong_number_of_labels() -> None:
    with pytest.raises(ValueError, match="labels"):
        small(4, 1).relabeled([1, 0])


def test_an_inventory_reports_its_shape() -> None:
    assert repr(small(4, 1)) == "SpanInventory(4 spans, features=3, classes=2)"


def test_a_control_task_labels_one_key_alike_wherever_it_appears() -> None:
    task = ControlTask(seed=7)
    inventory = SpanInventory(
        torch.zeros(4, 2, dtype=torch.float64),
        ["a", "b", "a", "b"],
        [0, 0, 0, 0],
        classes=3,
    )

    drawn = task.over(inventory).labels

    assert drawn[0] == drawn[2]
    assert drawn[1] == drawn[3]
    assert drawn[0] == task.label("a", 3)


def test_a_control_task_labels_a_key_alike_in_another_inventory() -> None:
    task = ControlTask(seed=7)
    here = SpanInventory(torch.zeros(1, 2, dtype=torch.float64), ["a"], [0], classes=3)
    there = SpanInventory(
        torch.ones(2, 2, dtype=torch.float64), ["z", "a"], [1, 1], classes=3
    )

    assert task.over(here).labels[0] == task.over(there).labels[1]


def test_a_control_task_draws_below_the_class_count() -> None:
    task = ControlTask(seed=7)

    assert all(task.label(f"key-{at}", 3) in {0, 1, 2} for at in range(50))


def test_another_seed_draws_another_assignment() -> None:
    inventory = separable(40, 1)

    first = ControlTask(seed=7).assignment(inventory)
    second = ControlTask(seed=8).assignment(inventory)

    assert first.keys() == second.keys()
    assert first != second


def test_the_same_seed_draws_the_same_assignment() -> None:
    inventory = separable(40, 1)

    assert ControlTask(seed=7).assignment(inventory) == ControlTask(seed=7).assignment(
        inventory
    )


def test_an_assignment_holds_one_label_per_distinct_key() -> None:
    inventory = SpanInventory(
        torch.zeros(3, 2, dtype=torch.float64), ["b", "a", "b"], [0, 0, 0], classes=3
    )

    assignment = ControlTask(seed=7).assignment(inventory)

    assert list(assignment) == ["a", "b"]


def test_a_control_task_keeps_the_span_inventory_it_relabels() -> None:
    inventory = separable(12, 1)

    controlled = ControlTask(seed=7).over(inventory)

    assert torch.equal(controlled.vectors, inventory.vectors)
    assert controlled.keys == inventory.keys
    assert controlled.classes == inventory.classes
    assert controlled.labels != inventory.labels


def test_a_control_task_refuses_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="control-task seed"):
        ControlTask(seed=-1)


def test_a_control_task_refuses_an_empty_key() -> None:
    with pytest.raises(ValueError, match="span key"):
        ControlTask(seed=7).label("", 3)


def test_a_control_task_refuses_a_class_count_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="class count"):
        ControlTask(seed=7).label("a", 0)


def test_control_tasks_are_equal_on_their_seed() -> None:
    assert ControlTask(seed=7) == ControlTask(seed=7)
    assert ControlTask(seed=7) != ControlTask(seed=8)
    assert hash(ControlTask(seed=7)) == hash(ControlTask(seed=7))
    assert ControlTask(seed=7) != 7


def test_a_control_task_reports_its_seed() -> None:
    assert repr(ControlTask(seed=7)) == "ControlTask(seed=7)"


def test_a_linear_capacity_counts_one_weight_per_coordinate_and_class() -> None:
    capacity = ProbeCapacity(hidden=0, epochs=1, learning_rate=0.1, seed=0)

    assert capacity.parameters(5, 3) == 5 * 3 + 3


def test_a_hidden_capacity_counts_both_layers() -> None:
    capacity = ProbeCapacity(hidden=4, epochs=1, learning_rate=0.1, seed=0)

    assert capacity.parameters(5, 3) == 5 * 4 + 4 + 4 * 3 + 3


def test_a_capacity_refuses_a_class_count_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="class count"):
        ProbeCapacity(hidden=0, epochs=1, learning_rate=0.1, seed=0).parameters(5, 0)


def test_a_capacity_refuses_a_coordinate_count_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="coordinate count"):
        ProbeCapacity(hidden=0, epochs=1, learning_rate=0.1, seed=0).parameters(0, 3)


def test_a_capacity_refuses_zero_epochs() -> None:
    with pytest.raises(ValueError, match="epoch count"):
        ProbeCapacity(hidden=4, epochs=0, learning_rate=0.1, seed=0)


def test_a_capacity_refuses_a_negative_hidden_width() -> None:
    with pytest.raises(ValueError, match="hidden width"):
        ProbeCapacity(hidden=-1, epochs=1, learning_rate=0.1, seed=0)


def test_a_capacity_refuses_a_rate_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="learning rate"):
        ProbeCapacity(hidden=4, epochs=1, learning_rate=0.0, seed=0)


def test_a_capacity_refuses_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="probe seed"):
        ProbeCapacity(hidden=4, epochs=1, learning_rate=0.1, seed=-1)


def test_capacities_are_equal_on_their_four_fields() -> None:
    assert ProbeCapacity(hidden=8, epochs=40, learning_rate=0.1, seed=3) == SMALL
    assert SMALL != HIGH
    assert hash(SMALL) == hash(
        ProbeCapacity(hidden=8, epochs=40, learning_rate=0.1, seed=3)
    )
    assert SMALL != 8


def test_a_capacity_reports_the_call_that_rebuilds_it() -> None:
    assert repr(SMALL) == (
        "ProbeCapacity(hidden=8, epochs=40, learning_rate=0.1, seed=3)"
    )


def test_a_result_reports_selectivity_as_the_gap_between_its_accuracies() -> None:
    result = ProbeResult(0.8, 0.3, control_seed=4, capacity=SMALL, parameters=12)

    assert result.accuracy == 0.8
    assert result.control_accuracy == 0.3
    assert result.selectivity == pytest.approx(0.5)
    assert result.control_seed == 4
    assert result.capacity == SMALL
    assert result.parameters == 12


def test_a_result_refuses_an_accuracy_outside_zero_to_one() -> None:
    with pytest.raises(ValueError, match="probe accuracy is not a proportion"):
        ProbeResult(1.5, 0.3, control_seed=4, capacity=SMALL, parameters=12)


def test_a_result_refuses_a_control_accuracy_outside_zero_to_one() -> None:
    with pytest.raises(ValueError, match="control-task accuracy is not a proportion"):
        ProbeResult(0.5, -0.1, control_seed=4, capacity=SMALL, parameters=12)


def test_a_result_refuses_a_capacity_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="ProbeCapacity"):
        ProbeResult(0.5, 0.3, control_seed=4, capacity=8, parameters=12)  # ty: ignore[invalid-argument-type]


def test_a_result_refuses_a_negative_seed() -> None:
    with pytest.raises(ValueError, match="control-task seed"):
        ProbeResult(0.5, 0.3, control_seed=-1, capacity=SMALL, parameters=12)


def test_a_result_refuses_a_weight_count_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="weight count"):
        ProbeResult(0.5, 0.3, control_seed=4, capacity=SMALL, parameters=0)


def test_a_result_reports_both_accuracies_the_gap_and_the_seed() -> None:
    result = ProbeResult(0.8, 0.3, control_seed=4, capacity=SMALL, parameters=12)

    assert repr(result) == (
        "ProbeResult(accuracy=0.8, control_accuracy=0.3, "
        "selectivity=0.5, control_seed=4)"
    )


def test_selectivity_is_positive_where_the_vectors_predict_the_labels() -> None:
    probe = ClassifierProbe(HIGH, ControlTask(seed=11))

    result = probe.run(separable(200, 1), separable(200, 2))

    assert result.accuracy > 0.9
    assert result.control_accuracy < 0.4
    assert result.selectivity > 0.4


def test_selectivity_is_near_zero_where_the_labels_are_random() -> None:
    probe = ClassifierProbe(HIGH, ControlTask(seed=11))

    result = probe.run(scattered(200, 1), scattered(200, 2))

    assert abs(result.selectivity) < 0.1


def test_a_recurring_span_key_leaves_a_high_accuracy_unselective() -> None:
    probe = ClassifierProbe(HIGH, ControlTask(seed=11))

    result = probe.run(recurring(150, 1), recurring(150, 2))

    assert result.accuracy > 0.9
    assert result.control_accuracy > 0.9
    assert abs(result.selectivity) < 0.1


def test_a_probe_reports_the_seed_the_capacity_and_the_weight_count() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))
    fitting = small(40, 1)

    result = probe.run(fitting, small(40, 2))

    assert result.control_seed == 11
    assert result.capacity == SMALL
    assert result.parameters == SMALL.parameters(fitting.features, fitting.classes)


def test_a_probe_runs_the_same_way_twice() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))
    fitting, held_out = small(40, 1), small(40, 2)

    first = probe.run(fitting, held_out)
    second = probe.run(fitting, held_out)

    assert first.accuracy == second.accuracy
    assert first.control_accuracy == second.control_accuracy


def test_the_real_task_does_not_move_with_the_control_seed() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))
    fitting, held_out = separable(60, 1), separable(60, 2)

    first = probe.run(fitting, held_out)
    second = probe.under(ControlTask(seed=12)).run(fitting, held_out)

    assert first.accuracy == second.accuracy
    assert second.control_seed == 12


def test_another_control_task_keeps_the_capacity() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))

    other = probe.under(ControlTask(seed=12))

    assert other.capacity == SMALL
    assert other.control == ControlTask(seed=12)


def test_a_probe_refuses_inventories_of_different_widths() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))

    with pytest.raises(ValueError, match="3 coordinates"):
        probe.run(small(10, 1), separable(10, 2))


def test_a_probe_refuses_inventories_of_different_class_counts() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))
    wide = SpanInventory(
        torch.zeros(4, 3, dtype=torch.float64),
        ["a", "b", "c", "d"],
        [0, 1, 2, 0],
        classes=3,
    )

    with pytest.raises(ValueError, match="2 classes cannot be scored over 3"):
        probe.run(small(4, 1), wide)


def test_a_probe_refuses_something_that_is_not_an_inventory() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))

    with pytest.raises(TypeError, match="SpanInventory"):
        probe.run(small(4, 1), "spans")  # ty: ignore[invalid-argument-type]


def test_a_probe_refuses_a_capacity_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="ProbeCapacity"):
        ClassifierProbe(8, ControlTask(seed=11))  # ty: ignore[invalid-argument-type]


def test_a_probe_refuses_a_control_task_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="ControlTask"):
        ClassifierProbe(SMALL, 11)  # ty: ignore[invalid-argument-type]


def test_a_probe_reports_its_capacity_and_its_control_task() -> None:
    probe = ClassifierProbe(SMALL, ControlTask(seed=11))

    assert repr(probe) == f"ClassifierProbe({SMALL!r}, ControlTask(seed=11))"
