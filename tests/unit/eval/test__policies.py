from itertools import permutations
from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.eval._alignment import Contingency, apply_alignment
from lofi.eval._policies import (
    ManyToOne,
    OneToOne,
    _Assignment,
    _assignment_values,
    _costs,
)
from lofi.eval._ports import TIE_BREAKS, TieBreak

if TYPE_CHECKING:
    from collections.abc import Mapping

# hand-computed four-by-four assignment
_SQUARE: Final[Mapping[tuple[str, str], int]] = {
    ("u0", "g0"): 9,
    ("u0", "g1"): 1,
    ("u0", "g2"): 0,
    ("u0", "g3"): 0,
    ("u1", "g0"): 8,
    ("u1", "g1"): 7,
    ("u1", "g2"): 1,
    ("u1", "g3"): 0,
    ("u2", "g0"): 0,
    ("u2", "g1"): 2,
    ("u2", "g2"): 6,
    ("u2", "g3"): 1,
    ("u3", "g0"): 0,
    ("u3", "g1"): 0,
    ("u3", "g2"): 3,
    ("u3", "g3"): 5,
}


def _dev(counts: Mapping[tuple[str, str], int]) -> Contingency:
    return Contingency(counts, split="dev")


@st.composite
def _tables(draw: st.DrawFn) -> Contingency:
    rows = draw(st.integers(min_value=1, max_value=4))
    columns = draw(st.integers(min_value=1, max_value=4))

    return Contingency(
        {
            (f"u{row}", f"g{column}"): draw(st.integers(min_value=0, max_value=4))
            for row in range(rows)
            for column in range(columns)
        },
        split="dev",
    )


@st.composite
def _matrices(draw: st.DrawFn) -> list[list[int]]:
    size = draw(st.integers(min_value=1, max_value=5))

    return [
        [draw(st.integers(min_value=0, max_value=12)) for _ in range(size)]
        for _ in range(size)
    ]


def _matched(mapping: Mapping[str, str], table: Contingency) -> int:
    return sum(table.count(induced, gold) for induced, gold in mapping.items())


def _best_assignment(
    table: Contingency, tie_break: TieBreak
) -> tuple[dict[str, str], int]:
    induced = table.induced_labels
    gold = table.gold_labels
    size = max(len(induced), len(gold))
    columns: list[str | None] = [*gold, *(None,) * (size - len(gold))]
    marginals = {
        label: sum(table.count(row, label) for row in induced) for label in gold
    }
    best: tuple[int, int, tuple[int, ...]] = (-1, -1, ())
    winner: dict[str, str] = {}

    for chosen in permutations(columns, len(induced)):
        pairs = {
            row: column
            for row, column in zip(induced, chosen, strict=True)
            if column is not None
        }
        key = (
            sum(table.count(row, column) for row, column in pairs.items()),
            sum(marginals[column] for column in pairs.values())
            if tie_break == "largest-gold-marginal"
            else 0,
            tuple(
                -(gold.index(column) if column is not None else len(gold))
                for column in chosen
            ),
        )

        if key > best:
            best, winner = key, pairs

    return (
        {row: column for row, column in winner.items() if table.count(row, column) > 0},
        max(best[0], 0),
    )


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_many_to_one_takes_each_labels_most_frequent_gold_label(
    tie_break: TieBreak,
) -> None:
    fitted = ManyToOne(tie_break=tie_break).fit(_dev(_SQUARE))

    assert dict(fitted.mapping) == {"u0": "g0", "u1": "g0", "u2": "g2", "u3": "g3"}
    assert not fitted.is_injective


def test_many_to_one_accuracy_is_the_summed_row_maxima_over_the_total() -> None:
    table = _dev(_SQUARE)
    fitted = ManyToOne(tie_break="least-gold-label").fit(table)
    scored = apply_alignment(fitted, table, unit="token")
    maxima = sum(
        max(table.count(induced, gold) for gold in table.gold_labels)
        for induced in table.induced_labels
    )

    assert maxima == 9 + 8 + 6 + 5
    assert table.total == 43
    assert scored.value == maxima / table.total
    assert scored.components["induced_labels"] == 4.0


def test_a_hand_computed_four_by_four_case_reaches_the_one_to_one_optimum() -> None:
    table = _dev(_SQUARE)
    fitted = OneToOne(tie_break="least-gold-label").fit(table)

    # optimal diagonal total: 9 + 7 + 6 + 5
    assert dict(fitted.mapping) == {"u0": "g0", "u1": "g1", "u2": "g2", "u3": "g3"}
    assert _matched(fitted.mapping, table) == 27
    assert fitted.is_injective


def test_the_one_to_one_optimum_beats_the_greedy_choice() -> None:
    # greedy total: 8 + 1
    table = _dev({("u0", "g0"): 7, ("u0", "g1"): 1, ("u1", "g0"): 8, ("u1", "g1"): 0})
    fitted = OneToOne(tie_break="least-gold-label").fit(table)

    assert dict(fitted.mapping) == {"u0": "g1", "u1": "g0"}
    assert _matched(fitted.mapping, table) == 9


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_a_rectangular_table_is_handled_by_padding_to_square(
    tie_break: TieBreak,
) -> None:
    wide = _dev(
        {
            ("u0", "g0"): 4,
            ("u0", "g1"): 1,
            ("u0", "g2"): 0,
            ("u1", "g0"): 1,
            ("u1", "g1"): 6,
            ("u1", "g2"): 2,
        }
    )
    tall = _dev(
        {
            ("u0", "g0"): 4,
            ("u0", "g1"): 1,
            ("u1", "g0"): 1,
            ("u1", "g1"): 6,
            ("u2", "g0"): 3,
            ("u2", "g1"): 2,
        }
    )
    policy = OneToOne(tie_break=tie_break)

    assert dict(policy.fit(wide).mapping) == {"u0": "g0", "u1": "g1"}
    assert dict(policy.fit(tall).mapping) == {"u0": "g0", "u1": "g1"}


def test_an_induced_label_the_table_never_counted_is_left_unassigned() -> None:
    table = _dev({("u0", "g0"): 5, ("u1", "g0"): 0, ("u1", "g1"): 0})

    assert dict(ManyToOne(tie_break="least-gold-label").fit(table).mapping) == {
        "u0": "g0"
    }
    assert dict(OneToOne(tie_break="least-gold-label").fit(table).mapping) == {
        "u0": "g0"
    }


def test_splitting_an_induced_label_weakly_raises_the_many_to_one_score() -> None:
    whole = _dev({("u0", "g0"): 6, ("u0", "g1"): 4, ("u1", "g1"): 9})
    split = _dev(
        {("u0a", "g0"): 6, ("u0a", "g1"): 1, ("u0b", "g1"): 3, ("u1", "g1"): 9}
    )
    policy = ManyToOne(tie_break="least-gold-label")
    before = apply_alignment(policy.fit(whole), whole, unit="token")
    after = apply_alignment(policy.fit(split), split, unit="token")

    assert before.value == 15 / 19
    assert after.value == 18 / 19
    assert after.components["induced_labels"] > before.components["induced_labels"]


def test_splitting_an_induced_label_can_lower_the_one_to_one_score() -> None:
    whole = _dev({("u0", "g0"): 10, ("u1", "g1"): 10})
    split = _dev({("u0a", "g0"): 5, ("u0b", "g0"): 5, ("u1", "g1"): 10})
    policy = OneToOne(tie_break="least-gold-label")
    before = apply_alignment(policy.fit(whole), whole, unit="token")
    after = apply_alignment(policy.fit(split), split, unit="token")

    assert before.value == 1.0
    assert after.value == 0.75


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_a_fit_is_the_same_map_however_the_counts_were_accumulated(
    tie_break: TieBreak,
) -> None:
    forward = _dev(_SQUARE)
    backward = Contingency(dict(reversed(list(_SQUARE.items()))), split="dev")

    for policy in (ManyToOne(tie_break=tie_break), OneToOne(tie_break=tie_break)):
        assert policy.fit(forward).serialize() == policy.fit(backward).serialize()


def test_the_two_tie_breaks_settle_a_many_to_one_tie_differently() -> None:
    tied = _dev({("u0", "g0"): 2, ("u0", "g1"): 2, ("u1", "g0"): 0, ("u1", "g1"): 9})

    assert ManyToOne(tie_break="least-gold-label").fit(tied).apply("u0") == "g0"
    assert ManyToOne(tie_break="largest-gold-marginal").fit(tied).apply("u0") == "g1"


def test_the_two_tie_breaks_settle_a_one_to_one_tie_differently() -> None:
    # both assignments total eight. g1 has the larger marginal
    tied = _dev({("u0", "g0"): 5, ("u0", "g1"): 5, ("u1", "g1"): 1, ("u1", "g2"): 3})
    least = OneToOne(tie_break="least-gold-label").fit(tied)
    marginal = OneToOne(tie_break="largest-gold-marginal").fit(tied)

    assert dict(least.mapping) == {"u0": "g0", "u1": "g2"}
    assert dict(marginal.mapping) == {"u0": "g1", "u1": "g2"}
    assert _matched(least.mapping, tied) == _matched(marginal.mapping, tied) == 8


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
@given(table=_tables())
@settings(max_examples=80)
def test_the_solver_reproduces_a_brute_force_optimum(
    tie_break: TieBreak, table: Contingency
) -> None:
    expected, total = _best_assignment(table, tie_break)
    fitted = OneToOne(tie_break=tie_break).fit(table)

    assert dict(fitted.mapping) == expected
    assert _matched(fitted.mapping, table) == total


@given(cost=_matrices())
@settings(max_examples=80)
def test_the_solver_finds_the_cheapest_permutation(cost: list[list[int]]) -> None:
    taken = _Assignment(cost).solve()
    cheapest = min(
        sum(cost[row][column] for row, column in enumerate(order))
        for order in permutations(range(len(cost)))
    )

    assert sorted(taken) == list(range(len(cost)))
    assert sum(cost[row][column] for row, column in enumerate(taken)) == cheapest


def test_the_value_matrix_is_padded_square_and_its_costs_start_at_zero() -> None:
    table = _dev({("u0", "g0"): 1, ("u0", "g1"): 0, ("u1", "g1"): 1})
    wide = _dev({("u0", "g0"): 1, ("u0", "g1"): 0, ("u0", "g2"): 2})
    values = _assignment_values(table, tie_break="largest-gold-marginal")
    padded = _assignment_values(wide, tie_break="least-gold-label")
    costs = _costs(values)

    assert [len(row) for row in values] == [2, 2]
    assert [len(row) for row in padded] == [3, 3, 3]
    assert min(min(row) for row in costs) == 0
    assert all(cost >= 0 for row in costs for cost in row)


@pytest.mark.parametrize("policy_type", [ManyToOne, OneToOne])
def test_a_policy_refuses_an_undeclared_tie_breaking_order(
    policy_type: type[ManyToOne | OneToOne],
) -> None:
    with pytest.raises(ValueError, match="not a declared one"):
        policy_type(tie_break=cast("TieBreak", "whatever-comes-first"))


def test_a_policy_publishes_its_name_its_order_and_its_injectivity() -> None:
    many = ManyToOne(tie_break="least-gold-label")
    one = OneToOne(tie_break="largest-gold-marginal")

    assert (many.name, many.tie_break, many.injective) == (
        "many-to-one",
        "least-gold-label",
        False,
    )
    assert (one.name, one.tie_break, one.injective) == (
        "one-to-one",
        "largest-gold-marginal",
        True,
    )


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_neither_policy_fits_on_the_test_split(tie_break: TieBreak) -> None:
    on_test = Contingency(_SQUARE, split="test")

    for policy in (ManyToOne(tie_break=tie_break), OneToOne(tie_break=tie_break)):
        with pytest.raises(ValueError, match="test"):
            policy.fit(on_test)


@pytest.mark.parametrize("tie_break", TIE_BREAKS)
def test_an_empty_table_fits_an_empty_correspondence(tie_break: TieBreak) -> None:
    empty = Contingency({}, split="dev")

    for policy in (ManyToOne(tie_break=tie_break), OneToOne(tie_break=tie_break)):
        assert dict(policy.fit(empty).mapping) == {}
