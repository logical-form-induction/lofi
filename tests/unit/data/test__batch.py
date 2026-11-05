from typing import TYPE_CHECKING, cast, override

import pytest
import torch

from lofi._types import ItemId
from lofi.data import _batch
from lofi.data._batch import Alignment, Batch, Mask

if TYPE_CHECKING:
    from tests.conftest import ChecksReload

IDS = (ItemId("s/1"), ItemId("s/2"))


def alignment() -> Alignment:
    return Alignment(tuple((index, index + 1) for index in range(3)))


def mask() -> Mask:
    return Mask(torch.tensor([[True, True, True], [True, True, False]]))


def vectors() -> torch.Tensor:
    return torch.arange(12, dtype=torch.float32).reshape(2, 3, 2)


def build() -> Batch:
    return _batch.Batch(
        vectors(),
        (3, 2),
        mask=mask(),
        alignments=(alignment(), alignment()),
        item_ids=IDS,
    )


def reading() -> tuple[
    tuple[float, ...],
    tuple[int, ...],
    tuple[bool, ...],
    tuple[tuple[int, tuple[tuple[int, int] | None, ...]], ...],
    tuple[ItemId, ...],
    int,
]:
    built = build()
    return (
        tuple(built.vectors.flatten().tolist()),
        built.lengths,
        tuple(built.mask.values.flatten().tolist()),
        tuple((one.token_count, one.rows) for one in built.alignments),
        built.item_ids,
        len(built),
    )


def test_an_alignment_says_which_rows_align_to_each_token() -> None:
    subject = Alignment(((0, 2), (2, 3), (3, 6)))

    assert subject.token_count == 3
    assert subject.rows_for(1) == (2, 3)
    assert subject.is_total
    assert subject.uncovered == ()


def test_an_alignment_names_unaligned_tokens() -> None:
    subject = Alignment(((0, 1), None, None, (1, 2)))

    assert not subject.is_total
    assert subject.uncovered == (1, 2)
    assert subject.rows_for(1) is None


def test_an_alignment_refuses_an_empty_interval() -> None:
    with pytest.raises(ValueError, match="aligned to no row"):
        Alignment(((1, 1), None))


def test_an_alignment_refuses_intervals_that_run_out_of_token_order() -> None:
    with pytest.raises(ValueError, match="before the"):
        Alignment(((2, 4), (1, 3)))


def test_an_alignment_refuses_a_position_the_item_has_no_token_at() -> None:
    with pytest.raises(IndexError, match="has none at position"):
        Alignment(((0, 1), None)).rows_for(5)


def test_two_alignments_over_one_correspondence_are_one_value() -> None:
    assert alignment() == alignment()
    assert hash(alignment()) == hash(alignment())
    assert alignment() != Alignment((*alignment().rows, None))
    assert len({alignment(), alignment()}) == 1


def test_an_alignment_reads_in_token_order_whatever_order_it_was_handed() -> None:
    subject = Alignment(((0, 1), (1, 2), (2, 3)))

    assert subject.rows == ((0, 1), (1, 2), (2, 3))


def test_an_alignment_keeps_no_handle_on_the_sequence_it_was_given() -> None:
    rows = [(0, 1)]
    subject = Alignment(tuple(rows))
    rows[0] = (9, 10)

    assert subject.rows_for(0) == (0, 1)


def test_a_mask_reads_the_shape_it_covers_off_the_tensor() -> None:
    subject = mask()

    assert subject.items == 2
    assert subject.width == 3
    assert bool(subject.values[1, 2]) is False


def test_a_mask_over_the_wrong_rank_is_refused() -> None:
    with pytest.raises(ValueError, match="rank 2"):
        Mask(torch.tensor([True, False]))

    with pytest.raises(ValueError, match="rank 2"):
        Mask(torch.ones(2, 3, 4, dtype=torch.bool))


def test_a_mask_holds_booleans() -> None:
    with pytest.raises(TypeError, match="holds booleans"):
        Mask(torch.ones(2, 3))


def test_a_batch_contains_its_lengths_mask_and_alignments() -> None:
    built = build()

    assert len(built) == 2
    assert built.lengths == (3, 2)
    assert built.item_ids == IDS
    assert built.mask.width == 3
    assert built.alignments[0].is_total
    assert repr(built) == "Batch(items=2, width=3)"


def test_a_batch_refuses_a_negative_length_whatever_the_flag_says() -> None:
    with pytest.raises(ValueError, match="negative"):
        Batch(
            vectors(),
            (3, -1),
            mask=mask(),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_a_batch_refuses_lengths_ids_and_alignments_that_disagree() -> None:
    with pytest.raises(ValueError, match="came with 1 alignments"):
        Batch(
            vectors(),
            (3, 2),
            mask=mask(),
            alignments=(alignment(),),
            item_ids=IDS,
        )

    with pytest.raises(ValueError, match="came with 1 ids"):
        Batch(
            vectors(),
            (3, 2),
            mask=mask(),
            alignments=(alignment(), alignment()),
            item_ids=IDS[:1],
        )


def test_the_guard_rejects_a_batch_whose_lengths_miss_the_leading_axis(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match="leading axis holds 2 items"):
        _batch.Batch(
            vectors(),
            (3, 2, 1),
            mask=mask(),
            alignments=(alignment(), alignment(), alignment()),
            item_ids=(*IDS, ItemId("s/3")),
        )


def test_a_batch_always_requires_a_padding_mask() -> None:
    with pytest.raises(TypeError, match="requires a padding mask"):
        Batch(
            vectors(),
            (3, 2),
            mask=cast("Mask", None),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_the_guard_rejects_a_mask_covering_a_different_number_of_items(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match=r"shape \(3, 3\).+shape \(2, 3\)"):
        _batch.Batch(
            vectors(),
            (3, 2),
            mask=Mask(torch.ones(3, 3, dtype=torch.bool)),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_the_guard_rejects_vectors_without_item_and_position_axes(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match="item axis and a position axis"):
        _batch.Batch(
            torch.ones(2),
            (1, 1),
            mask=Mask(torch.ones(2, 1, dtype=torch.bool)),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_the_guard_rejects_a_mask_with_a_different_position_axis(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match=r"shape \(2, 2\).+shape \(2, 3\)"):
        _batch.Batch(
            vectors(),
            (2, 2),
            mask=Mask(torch.ones(2, 2, dtype=torch.bool)),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_the_guard_rejects_a_mask_wider_than_the_longest_item(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match=r"width 4.+longest item has length 3"):
        _batch.Batch(
            torch.ones(2, 4, 2),
            (3, 2),
            mask=Mask(
                torch.tensor([[True, True, True, False], [True, True, False, False]])
            ),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_the_guard_rejects_mask_counts_that_disagree_with_lengths(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match=r"row counts \(2, 3\).+lengths are \(3, 2\)"):
        _batch.Batch(
            vectors(),
            (3, 2),
            mask=Mask(torch.tensor([[True, True, False], [True, True, True]])),
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_zero_length_rows_are_legal(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    built = _batch.Batch(
        torch.empty(2, 0, 4),
        (0, 0),
        mask=Mask(torch.empty(2, 0, dtype=torch.bool)),
        alignments=(Alignment(()), Alignment(())),
        item_ids=IDS,
    )

    assert built.mask.width == 0
    assert built.lengths == (0, 0)


def test_the_guard_rejects_an_alignment_leaving_a_token_on_no_row(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match=r"leaves tokens \(1,\) on no row"):
        _batch.Batch(
            vectors(),
            (3, 2),
            mask=mask(),
            alignments=(alignment(), Alignment(((0, 1), None, (1, 2)))),
            item_ids=IDS,
        )


def test_the_guard_is_real_in_both_directions(
    reload_with_checks: ChecksReload,
) -> None:
    wrong_counts = Mask(torch.tensor([[True, True, False], [True, True, True]]))

    reload_with_checks(_batch, on=False)
    admitted = _batch.Batch(
        vectors(),
        (3, 2),
        mask=wrong_counts,
        alignments=(alignment(), alignment()),
        item_ids=IDS,
    )

    assert admitted.mask is wrong_counts

    reload_with_checks(_batch, on=True)

    with pytest.raises(ValueError, match="row counts"):
        _batch.Batch(
            vectors(),
            (3, 2),
            mask=wrong_counts,
            alignments=(alignment(), alignment()),
            item_ids=IDS,
        )


def test_the_golden_batch_reads_the_same_under_both_settings(
    reload_with_checks: ChecksReload,
) -> None:
    reload_with_checks(_batch, on=True)
    checked = reading()

    reload_with_checks(_batch, on=False)
    unchecked = reading()

    assert checked == unchecked
    assert checked[1] == (3, 2)
    assert checked[5] == 2


def test_a_batch_takes_on_nothing_after_it_is_built() -> None:
    built = build()

    for field in ("lengths", "mask", "attached"):
        with pytest.raises(AttributeError):
            setattr(built, field, None)


class _MaskedStub(torch.nn.Module):
    @override
    def forward(self, batch: Batch) -> torch.Tensor:
        return batch.vectors * batch.mask.values.unsqueeze(-1)


def test_a_real_module_zeros_padded_contributions_and_gradients() -> None:
    held = vectors().detach().requires_grad_()
    batch = Batch(
        held,
        (3, 2),
        mask=mask(),
        alignments=(alignment(), alignment()),
        item_ids=IDS,
    )

    contributions = _MaskedStub()(batch)
    contributions.sum().backward()
    gradient = held.grad

    assert torch.equal(contributions[1, 2], torch.zeros(2))
    assert gradient is not None
    assert torch.equal(gradient[1, 2], torch.zeros(2))
    assert torch.equal(gradient[batch.mask.values], torch.ones(5, 2))
