from itertools import pairwise

import pytest

from lofi.deduce._spans import Cell, SpanIndex

# every span of a three-token input, in ascending span-id order.
SPANS_AT_THREE = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]


def test_counts_every_span_of_a_short_input() -> None:
    index = SpanIndex(3)

    assert index.n == 3
    assert index.num_spans == 6
    assert index.max_splits == 2
    assert list(index.spans_by_length()) == SPANS_AT_THREE


def test_an_empty_input_has_no_spans() -> None:
    index = SpanIndex(0)

    assert index.num_spans == 0
    assert index.max_splits == 0
    assert list(index.spans_by_length()) == []
    assert index.left_child_ids == ()
    assert index.split_mask == ()


def test_a_single_token_has_one_span_and_no_splits() -> None:
    index = SpanIndex(1)

    assert index.num_spans == 1
    assert index.max_splits == 0
    assert list(index.splits(0, 1)) == []
    assert index.split_mask == ((),)


def test_a_negative_length_is_rejected() -> None:
    with pytest.raises(ValueError, match="non-negative"):
        SpanIndex(-1)


def test_span_ids_are_ascending_and_contiguous_by_width() -> None:
    index = SpanIndex(3)

    assert [index.span_id(i, j) for i, j in SPANS_AT_THREE] == [0, 1, 2, 3, 4, 5]


def test_span_of_inverts_span_id() -> None:
    index = SpanIndex(5)

    for i, j in index.spans_by_length():
        assert index.span_of(index.span_id(i, j)) == (i, j)


def test_span_of_rejects_an_id_outside_the_range() -> None:
    index = SpanIndex(3)

    with pytest.raises(IndexError, match="range\\(6\\)"):
        index.span_of(6)

    with pytest.raises(IndexError, match="outside"):
        index.span_of(-1)


def test_splits_are_the_interior_cut_points() -> None:
    index = SpanIndex(4)

    assert list(index.splits(0, 4)) == [1, 2, 3]
    assert list(index.splits(1, 3)) == [2]
    assert list(index.splits(2, 3)) == []


def test_split_id_addresses_a_row_and_a_column() -> None:
    index = SpanIndex(4)

    assert index.split_id(0, 4, 1) == (index.span_id(0, 4), 0)
    assert index.split_id(0, 4, 3) == (index.span_id(0, 4), 2)
    assert index.split_id(1, 3, 2) == (index.span_id(1, 3), 0)


def test_spans_by_length_honors_a_minimum_width() -> None:
    index = SpanIndex(3)

    assert list(index.spans_by_length(min_length=2)) == [(0, 2), (1, 3), (0, 3)]
    assert list(index.spans_by_length(min_length=4)) == []


def test_spans_by_length_desc_reverses_the_widths_only() -> None:
    index = SpanIndex(3)

    assert list(index.spans_by_length_desc()) == [
        (0, 3),
        (0, 2),
        (1, 3),
        (0, 1),
        (1, 2),
        (2, 3),
    ]
    assert list(index.spans_by_length_desc(min_length=3)) == [(0, 3)]


def test_a_span_covers_one_contiguous_region() -> None:
    assert SpanIndex(3).fanout == 1
    assert SpanIndex(0).fanout == 1


def test_the_leaves_are_the_width_one_spans() -> None:
    assert list(SpanIndex(3).leaves()) == [(0, 1), (1, 2), (2, 3)]
    assert list(SpanIndex(1).leaves()) == [(0, 1)]
    assert list(SpanIndex(0).leaves()) == []


def test_regions_by_size_is_the_span_enumeration_under_another_name() -> None:
    index = SpanIndex(3)

    assert list(index.regions_by_size()) == SPANS_AT_THREE
    assert list(index.regions_by_size()) == list(index.spans_by_length())


def test_binary_decompositions_are_the_split_pairs() -> None:
    index = SpanIndex(3)

    assert list(index.decompositions((0, 3), 2)) == [
        ((0, 1), (1, 3)),
        ((0, 2), (2, 3)),
    ]
    assert list(index.decompositions((2, 3), 2)) == []


def test_a_unary_decomposition_is_the_span_itself() -> None:
    index = SpanIndex(3)

    assert list(index.decompositions((0, 3), 1)) == [((0, 3),)]
    assert list(index.decompositions((2, 3), 1)) == [((2, 3),)]


def test_a_ternary_decomposition_cuts_at_two_points() -> None:
    index = SpanIndex(4)

    assert list(index.decompositions((0, 4), 3)) == [
        ((0, 1), (1, 2), (2, 4)),
        ((0, 1), (1, 3), (3, 4)),
        ((0, 2), (2, 3), (3, 4)),
    ]
    assert list(index.decompositions((0, 2), 3)) == []


def test_a_non_empty_span_has_no_nullary_decomposition() -> None:
    index = SpanIndex(2)

    assert list(index.decompositions((0, 2), 0)) == []
    assert list(index.decompositions((0, 1), 0)) == []


def test_a_negative_arity_is_rejected() -> None:
    index = SpanIndex(2)

    with pytest.raises(ValueError, match="non-negative"):
        list(index.decompositions((0, 2), -1))


def test_every_decomposition_covers_the_region_contiguously() -> None:
    index = SpanIndex(6)

    for arity in (1, 2, 3):
        for parts in index.decompositions((0, 6), arity):
            assert len(parts) == arity
            assert parts[0][0] == 0
            assert parts[-1][1] == 6
            assert all(left[1] == right[0] for left, right in pairwise(parts))
            assert all(start < end for start, end in parts)


def test_span_ids_are_a_bijection_onto_the_span_range() -> None:
    index = SpanIndex(6)
    spans = list(index.spans_by_length())
    ids = [index.span_id(i, j) for i, j in spans]

    assert index.num_spans == 21
    assert sorted(ids) == list(range(index.num_spans))
    assert [index.span_of(sid) for sid in ids] == spans


def test_the_binary_sweep_pairs_each_span_with_its_splits() -> None:
    index = SpanIndex(3)

    assert list(index.binary_sweep()) == [
        ((0, 2), (0, 1), (1, 2)),
        ((1, 3), (1, 2), (2, 3)),
        ((0, 3), (0, 1), (1, 3)),
        ((0, 3), (0, 2), (2, 3)),
    ]


def test_the_binary_sweep_skips_a_span_that_cannot_be_cut() -> None:
    assert list(SpanIndex(1).binary_sweep()) == []
    assert list(SpanIndex(0).binary_sweep()) == []


def test_the_binary_sweep_reaches_a_span_only_after_its_parts() -> None:
    index = SpanIndex(5)
    completed: set[Cell] = set()
    previous: Cell | None = None

    for parent, left, right in index.binary_sweep():
        if previous is not None and parent != previous:
            completed.add(previous)

        for child in (left, right):
            assert child[1] - child[0] == 1 or child in completed

        previous = parent


def test_the_binary_sweep_and_the_split_mask_count_the_same_splits() -> None:
    index = SpanIndex(5)

    assert len(list(index.binary_sweep())) == 20
    assert sum(sum(row) for row in index.split_mask) == 20
    assert all(len(row) == index.max_splits for row in index.split_mask)


def test_gather_rows_are_rectangular_and_masked() -> None:
    index = SpanIndex(3)
    wide = index.span_id(0, 3)
    narrow = index.span_id(0, 1)

    assert index.left_child_ids[wide] == (index.span_id(0, 1), index.span_id(0, 2))
    assert index.right_child_ids[wide] == (index.span_id(1, 3), index.span_id(2, 3))
    assert index.split_mask[wide] == (True, True)

    assert index.split_mask[narrow] == (False, False)
    assert len(index.left_child_ids[narrow]) == index.max_splits


def test_gather_rows_agree_with_split_enumeration() -> None:
    index = SpanIndex(5)

    for i, j in index.spans_by_length():
        sid = index.span_id(i, j)
        mask = index.split_mask[sid]
        left = index.left_child_ids[sid]
        right = index.right_child_ids[sid]
        splits = list(index.splits(i, j))

        assert len(mask) == index.max_splits
        assert [c for c, keep in zip(left, mask, strict=True) if keep] == [
            index.span_id(i, k) for k in splits
        ]
        assert [c for c, keep in zip(right, mask, strict=True) if keep] == [
            index.span_id(k, j) for k in splits
        ]


def test_gather_rows_are_built_once() -> None:
    index = SpanIndex(4)

    assert index.left_child_ids is index.left_child_ids
    assert index.right_child_ids is index.right_child_ids
    assert index.split_mask is index.split_mask


def test_indices_over_the_same_length_are_one_value() -> None:
    assert SpanIndex(4) == SpanIndex(4)
    assert SpanIndex(4) != SpanIndex(5)
    assert SpanIndex(4) != 4
    assert len({SpanIndex(4), SpanIndex(4), SpanIndex(5)}) == 2
    assert repr(SpanIndex(4)) == "SpanIndex(4)"


def test_the_input_length_cannot_be_reassigned() -> None:
    index = SpanIndex(4)

    for field, value in (("n", 5), ("num_spans", 0), ("max_splits", 99)):
        with pytest.raises(AttributeError):
            setattr(index, field, value)

    assert not hasattr(index, "__dict__")
    assert index.n == 4
