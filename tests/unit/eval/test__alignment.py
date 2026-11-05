from typing import TYPE_CHECKING, cast

import pytest

from lofi.eval._alignment import (
    Alignment,
    Contingency,
    apply_alignment,
)

if TYPE_CHECKING:
    from types import ModuleType

_COUNTS = {
    ("u0", "NP"): 40,
    ("u0", "VP"): 3,
    ("u1", "VP"): 25,
    ("u2", "NP"): 7,
}


@pytest.fixture
def torch_module() -> ModuleType:
    return pytest.importorskip("torch")


def _dev() -> Contingency:
    return Contingency(_COUNTS, split="dev")


def test_a_contingency_reports_its_inventories_its_total_and_a_missing_pair() -> None:
    table = _dev()

    assert table.induced_labels == ("u0", "u1", "u2")
    assert table.gold_labels == ("NP", "VP")
    assert table.total == 75
    assert table.count("u0", "NP") == 40
    assert table.count("u1", "NP") == 0
    assert table.count("u9", "NP") == 0
    assert table.split == "dev"


def test_a_contingency_copies_the_counts_it_was_given() -> None:
    counts = dict(_COUNTS)
    table = Contingency(counts, split="dev")
    counts[("u3", "NP")] = 100

    assert table.induced_labels == ("u0", "u1", "u2")
    assert table.total == 75


def test_no_field_of_a_contingency_can_be_rebound() -> None:
    table = _dev()

    for field in (
        "counts",
        "split",
        "induced_labels",
        "gold_labels",
        "total",
        "_counts",
        "_total",
    ):
        with pytest.raises(AttributeError):
            setattr(table, field, None)

    with pytest.raises(AttributeError):
        delattr(table, "_counts")

    assert not hasattr(table, "__dict__")
    assert table.total == 75


def test_an_empty_label_is_not_a_label() -> None:
    with pytest.raises(ValueError, match="an induced label is empty"):
        Contingency({("", "NP"): 1}, split="dev")

    with pytest.raises(ValueError, match="a gold label is empty"):
        Contingency({("u0", ""): 1}, split="dev")


def test_a_label_that_is_not_a_string_is_not_a_label() -> None:
    counts = cast("dict[tuple[str, str], int]", {(7, "NP"): 1})

    with pytest.raises(TypeError, match="an induced label is not a string"):
        Contingency(counts, split="dev")

    with pytest.raises(TypeError, match="a gold label is not a string"):
        Contingency(cast("dict[tuple[str, str], int]", {("u0", 7): 1}), split="dev")


def test_a_count_is_a_non_negative_integer() -> None:
    with pytest.raises(ValueError, match="negative"):
        Contingency({("u0", "NP"): -1}, split="dev")

    with pytest.raises(TypeError, match="not an integer"):
        Contingency({("u0", "NP"): True}, split="dev")


def test_a_contingency_serializes_its_counts_in_one_order() -> None:
    forward = Contingency(_COUNTS, split="dev")
    backward = Contingency(dict(reversed(list(_COUNTS.items()))), split="dev")

    assert forward.to_spec() == backward.to_spec()
    assert forward.to_spec()["counts"] == [
        ["u0", "NP", 40],
        ["u0", "VP", 3],
        ["u1", "VP", 25],
        ["u2", "NP", 7],
    ]


def test_contingencies_are_equal_on_their_counts_and_their_split() -> None:
    assert _dev() == Contingency(dict(reversed(list(_COUNTS.items()))), split="dev")
    assert hash(_dev()) == hash(Contingency(_COUNTS, split="dev"))
    assert _dev() != Contingency(_COUNTS, split="train")
    assert _dev() != "dev"


def test_an_alignment_maps_induced_labels_and_reports_what_it_left_out() -> None:
    alignment = Alignment({"u0": "NP", "u1": "VP"}, _dev())

    assert alignment.apply("u0") == "NP"
    assert alignment.apply("u1") == "VP"
    assert alignment.apply("u2") is None
    assert alignment.split == "dev"
    assert alignment.contingency == _dev()


def test_injectivity_is_read_off_the_map_rather_than_claimed() -> None:
    assert Alignment({"u0": "NP", "u1": "VP"}, _dev()).is_injective
    assert not Alignment({"u0": "NP", "u2": "NP"}, _dev()).is_injective
    assert Alignment({}, _dev()).is_injective


def test_an_alignment_over_a_label_nobody_counted_is_refused() -> None:
    with pytest.raises(ValueError, match="an induced label does not occur"):
        Alignment({"u9": "NP"}, _dev())

    with pytest.raises(ValueError, match="a gold label does not occur"):
        Alignment({"u0": "SBAR"}, _dev())


def test_an_alignment_accepts_any_named_fitting_split() -> None:
    for split in ("development", "validation-fold-7", "test"):
        table = Contingency(_COUNTS, split=split)

        assert Alignment({"u0": "NP"}, table).split == split


def test_an_alignment_serializes_to_one_text_whatever_order_it_was_built_in() -> None:
    first = Alignment({"u0": "NP", "u1": "VP"}, _dev())
    second = Alignment({"u1": "VP", "u0": "NP"}, _dev())

    assert first.serialize() == second.serialize()
    assert first == second
    assert hash(first) == hash(second)
    assert '"pairs": [["u0", "NP"], ["u1", "VP"]]' in first.serialize()
    assert '"split": "dev"' in first.serialize()


def test_alignments_differ_when_the_counts_behind_them_differ() -> None:
    fewer = Contingency({("u0", "NP"): 1}, split="dev")

    assert Alignment({"u0": "NP"}, _dev()) != Alignment({"u0": "NP"}, fewer)
    assert Alignment({"u0": "NP"}, _dev()) != "u0"


def test_no_field_of_an_alignment_can_be_rebound() -> None:
    alignment = Alignment({"u0": "NP"}, _dev())

    for field in (
        "mapping",
        "contingency",
        "split",
        "is_injective",
        "_mapping",
        "_contingency",
    ):
        with pytest.raises(AttributeError):
            setattr(alignment, field, None)

    with pytest.raises(AttributeError):
        delattr(alignment, "_mapping")

    assert not hasattr(alignment, "__dict__")
    assert alignment.apply("u0") == "NP"


def test_neither_value_takes_a_tensor(torch_module: ModuleType) -> None:
    tensor = torch_module.tensor(1.0)

    with pytest.raises(TypeError):
        Contingency({("u0", "NP"): tensor}, split="dev")

    with pytest.raises(TypeError):
        Contingency({(tensor, "NP"): 1}, split="dev")

    with pytest.raises(TypeError):
        Alignment({"u0": "NP"}, tensor)


_TEST_COUNTS = {
    ("u0", "NP"): 30,
    ("u0", "VP"): 5,
    ("u1", "VP"): 20,
    ("u2", "NP"): 5,
    ("u3", "NP"): 10,
}


def _fitted() -> Alignment:
    return Alignment({"u0": "NP", "u1": "VP", "u2": "VP"}, _dev())


def test_a_frozen_alignment_scores_a_split_it_was_never_fitted_on() -> None:
    scored = apply_alignment(
        _fitted(), Contingency(_TEST_COUNTS, split="test"), unit="token"
    )

    assert scored.value == pytest.approx(50 / 70)
    assert scored.n == 70
    assert scored.resampling.unit == "token"
    assert scored.components["matched"] == 50.0


def test_an_applied_score_carries_the_induced_inventory_it_was_read_over() -> None:
    scored = apply_alignment(
        _fitted(), Contingency(_TEST_COUNTS, split="test"), unit="token"
    )

    assert scored.components["induced_labels"] == 4.0
    assert scored.components["gold_labels"] == 2.0
    assert scored.components["fitted_induced_labels"] == 3.0


def test_a_label_the_correspondence_never_saw_costs_the_score() -> None:
    scored = apply_alignment(
        _fitted(), Contingency(_TEST_COUNTS, split="test"), unit="token"
    )

    assert scored.components["unaligned"] == 10.0
    assert scored.value is not None
    assert scored.value < 1.0


def test_applying_an_alignment_is_a_pure_function_of_the_frozen_map() -> None:
    table = Contingency(_TEST_COUNTS, split="test")
    reversed_counts = Contingency(
        dict(reversed(list(_TEST_COUNTS.items()))), split="test"
    )

    assert apply_alignment(_fitted(), table, unit="token") == apply_alignment(
        _fitted(), reversed_counts, unit="token"
    )
    assert apply_alignment(_fitted(), table, unit="token") == apply_alignment(
        Alignment({"u0": "NP", "u1": "VP", "u2": "VP"}, _dev()), table, unit="token"
    )


def test_applying_an_alignment_to_an_empty_table_reports_no_value() -> None:
    scored = apply_alignment(_fitted(), Contingency({}, split="test"), unit="token")

    assert scored.value is None
    assert scored.n == 0


def test_applying_reads_the_scored_counts_and_not_the_fitted_ones() -> None:
    on_dev = apply_alignment(_fitted(), _dev(), unit="token")
    on_test = apply_alignment(
        _fitted(), Contingency(_TEST_COUNTS, split="test"), unit="token"
    )

    assert on_dev.value == pytest.approx(65 / 75)
    assert on_dev.value != on_test.value


def test_applying_refuses_an_alignment_a_contingency_or_a_unit_that_is_not_one() -> (
    None
):
    table = Contingency(_TEST_COUNTS, split="test")

    with pytest.raises(TypeError, match="an applied alignment is not one"):
        apply_alignment(cast("Alignment", table), table, unit="token")

    with pytest.raises(TypeError, match="a scored contingency is not one"):
        apply_alignment(_fitted(), cast("Contingency", _fitted()), unit="token")

    with pytest.raises(ValueError, match="a scored unit is empty"):
        apply_alignment(_fitted(), table, unit="")
