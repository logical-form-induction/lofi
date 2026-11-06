from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.nn.sites import ComponentId, Role, Site
from lofi.probe._values import Observation, ReadingCache, Trace

if TYPE_CHECKING:
    from collections.abc import Callable

PARSER = ComponentId("cky-parser")
CELL = Site(PARSER, Role.CHART_CELL, (0, 4))
SPLITS = Site(PARSER, Role.SPLIT_DISTRIBUTION, (0, 4))

AXES = ("item", "span", "width")


def a_reading() -> torch.Tensor:
    return torch.arange(24, dtype=torch.float32).reshape(2, 3, 4)


def an_observation(
    site: Site = CELL,
    tensor: torch.Tensor | None = None,
    *,
    axes: tuple[str, ...] = AXES,
    mask: torch.Tensor | None = None,
) -> Observation:
    return Observation(
        site,
        a_reading() if tensor is None else tensor,
        axes=axes,
        mask=mask,
    )


def test_an_observation_keeps_the_four_things_it_records() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])
    observed = an_observation(mask=mask)

    assert observed.site is CELL
    assert observed.axes == AXES
    assert observed.rank == 3
    assert torch.equal(observed.tensor, a_reading())
    assert observed.mask is not None
    assert torch.equal(observed.mask, mask)


def test_an_observation_holds_no_autograd_node_even_when_read_from_a_graph() -> None:
    parameter = torch.zeros(2, 3, 4, requires_grad=True)
    live = parameter * 2.0

    observed = an_observation(tensor=live)

    assert live.grad_fn is not None
    assert observed.tensor.grad_fn is None
    assert observed.tensor.requires_grad is False


def test_selecting_on_one_axis_matches_the_slice_taken_by_hand() -> None:
    reading = a_reading()

    selected = an_observation(tensor=reading).select(span=1)

    assert torch.equal(selected.tensor, reading[:, 1])
    assert selected.axes == ("item", "width")


def test_selecting_on_several_axes_matches_the_slice_taken_by_hand() -> None:
    reading = a_reading()

    selected = an_observation(tensor=reading).select(item=1, width=2)

    assert torch.equal(selected.tensor, reading[1, :, 2])
    assert selected.axes == ("span",)


def test_selecting_on_every_axis_leaves_the_one_number_at_that_coordinate() -> None:
    reading = a_reading()

    selected = an_observation(tensor=reading).select(item=1, span=2, width=3)

    assert selected.rank == 0
    assert selected.tensor.item() == reading[1, 2, 3].item()


def test_selecting_no_axis_keeps_the_whole_reading() -> None:
    observed = an_observation()

    assert torch.equal(observed.select().tensor, observed.tensor)
    assert observed.select().axes == AXES


def test_a_selection_stays_at_the_site_it_was_read_from_and_chains() -> None:
    reading = a_reading()

    selected = an_observation(tensor=reading).select(item=0).select(width=3)

    assert selected.site is CELL
    assert selected.axes == ("span",)
    assert torch.equal(selected.tensor, reading[0, :, 3])


def test_a_selection_carries_the_mask_of_the_axes_it_kept() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])

    selected = an_observation(mask=mask).select(span=2)

    assert selected.mask is not None
    assert torch.equal(selected.mask, mask[:, 2])


def test_a_selection_under_the_masks_reach_leaves_the_mask_alone() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])

    selected = an_observation(mask=mask).select(width=1)

    assert selected.mask is not None
    assert torch.equal(selected.mask, mask)


def test_a_reading_with_no_mask_is_live_at_every_position() -> None:
    covering = an_observation().covering()

    assert covering.dtype is torch.bool
    assert tuple(covering.shape) == (2, 3, 4)
    assert bool(covering.all())


def test_a_mask_over_the_outer_axes_covers_the_axes_beneath_it() -> None:
    mask = torch.tensor([[True, True, False], [True, False, False]])

    covering = an_observation(mask=mask).covering()

    assert tuple(covering.shape) == (2, 3, 4)
    assert int(covering.sum()) == int(mask.sum()) * 4
    assert bool(covering[0, 0].all())
    assert not bool(covering[1, 2].any())


def test_a_mask_of_the_readings_own_rank_covers_each_number_on_its_own() -> None:
    mask = torch.arange(24).reshape(2, 3, 4) % 2 == 0

    covering = an_observation(mask=mask).covering()

    assert torch.equal(covering, mask)


def test_naming_an_axis_the_observation_does_not_declare_is_an_error() -> None:
    observed = an_observation()

    with pytest.raises(KeyError, match="no axis 'split' was observed here"):
        observed.select(split=0)


def test_a_coordinate_beyond_its_axis_is_refused_and_the_axis_is_named() -> None:
    with pytest.raises(ValueError, match="axis 'span' has 3 positions"):
        an_observation().select(span=3)


def test_a_coordinate_counting_from_the_end_is_refused() -> None:
    with pytest.raises(ValueError, match="counts from the start"):
        an_observation().select(item=-1)


def test_a_coordinate_that_is_not_an_integer_is_refused() -> None:
    with pytest.raises(TypeError, match="is not an integer"):
        an_observation().select(item=cast("int", "0"))


def test_the_extent_of_an_axis_is_read_by_its_name() -> None:
    observed = an_observation()

    assert observed.extent("item") == 2
    assert observed.extent("width") == 4

    with pytest.raises(KeyError, match="only"):
        observed.extent("split")


def test_the_position_of_an_axis_is_read_by_its_name() -> None:
    observed = an_observation()

    assert observed.position("item") == 0
    assert observed.position("span") == 1
    assert observed.position("width") == 2

    with pytest.raises(KeyError, match="only"):
        observed.position("split")


def test_axis_names_that_do_not_cover_the_reading_are_refused() -> None:
    with pytest.raises(ValueError, match="needs 3 axis names"):
        an_observation(axes=("item", "span"))


def test_a_repeated_axis_name_is_refused_and_the_message_names_it() -> None:
    with pytest.raises(ValueError, match="'span' twice"):
        an_observation(axes=("span", "span", "width"))


def test_a_mask_that_does_not_cover_the_outermost_axes_is_refused() -> None:
    with pytest.raises(ValueError, match="does not cover the outermost axes"):
        an_observation(mask=torch.ones(2, 5, dtype=torch.bool))


def test_a_mask_spanning_more_axes_than_the_reading_has_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot cover a reading of rank"):
        an_observation(mask=torch.ones(2, 3, 4, 1, dtype=torch.bool))


def test_a_mask_that_does_not_hold_truth_values_is_refused() -> None:
    with pytest.raises(TypeError, match="holds truth values"):
        an_observation(mask=torch.ones(2, 3))


def test_a_reading_that_is_not_a_tensor_is_refused() -> None:
    with pytest.raises(TypeError, match="is not a tensor"):
        Observation(CELL, cast("torch.Tensor", [1.0, 2.0]), axes=("width",))


def test_a_site_that_is_not_a_site_is_refused() -> None:
    with pytest.raises(TypeError, match="is not a Site"):
        Observation(cast("Site", "cky-parser"), a_reading(), axes=AXES)


def test_a_trace_reads_back_what_was_recorded_at_each_point() -> None:
    trace = Trace()
    cell = an_observation()
    splits = an_observation(SPLITS, torch.zeros(2, 3), axes=("item", "split"))

    trace.record(cell)
    trace.record(splits)

    assert len(trace) == 2
    assert trace[CELL] is cell
    assert trace[SPLITS] is splits
    assert CELL in trace
    assert trace.sites() == (CELL, SPLITS)
    assert dict(trace.items()) == {CELL: cell, SPLITS: splits}
    assert list(trace) == [CELL, SPLITS]


def test_a_trace_files_a_reading_under_the_point_it_came_from() -> None:
    trace = Trace()

    trace.record(an_observation(SPLITS, torch.zeros(2, 3), axes=("item", "split")))

    assert trace.sites() == (SPLITS,)


def test_a_trace_refuses_a_reading_once_its_scope_has_closed() -> None:
    trace = Trace()
    trace.close()

    assert trace.closed is True

    with pytest.raises(RuntimeError, match="closed before a reading"):
        trace.record(an_observation())

    assert len(trace) == 0


def test_closing_a_trace_twice_leaves_it_in_one_state() -> None:
    trace = Trace()
    trace.record(an_observation())
    trace.close()
    trace.close()

    assert trace.closed is True
    assert len(trace) == 1


def test_a_trace_refuses_a_second_reading_at_a_point_it_already_holds() -> None:
    trace = Trace()
    first = an_observation()
    trace.record(first)

    with pytest.raises(RuntimeError, match="already holds a reading"):
        trace.record(an_observation(tensor=torch.zeros(2, 3, 4)))

    assert trace[CELL] is first


def test_a_trace_records_observations_and_nothing_else() -> None:
    trace = Trace()

    with pytest.raises(TypeError, match="records observations"):
        trace.record(cast("Observation", a_reading()))


def test_asking_a_trace_for_a_point_it_never_read_names_the_point() -> None:
    trace = Trace()

    with pytest.raises(KeyError, match="nothing was read at"):
        _ = trace[CELL]


def test_a_trace_can_be_read_while_its_scope_is_still_open() -> None:
    trace = Trace()
    trace.record(an_observation())

    assert trace.closed is False
    assert trace[CELL].rank == 3


def test_a_reading_offers_no_way_in_by_position() -> None:
    observed = an_observation()
    select = cast("Callable[..., Observation]", observed.select)

    assert not hasattr(observed, "__getitem__")

    with pytest.raises(TypeError):
        select(0)


def a_store(*, capacity: int | None = None, compress: bool = False) -> ReadingCache:
    return ReadingCache(capacity=capacity, compress=compress)


def a_point(index: int) -> Site:
    return Site(PARSER, Role.CHART_CELL, (0, index))


def test_a_store_hands_back_the_reading_it_was_given_at_each_point() -> None:
    store = a_store()
    cell = an_observation()
    splits = an_observation(SPLITS, torch.zeros(2, 3), axes=("item", "split"))

    store.put(cell)
    store.put(splits)

    assert store.get(CELL) is cell
    assert store.get(SPLITS) is splits
    assert len(store) == 2
    assert list(store) == [CELL, SPLITS]
    assert store.items() == ((CELL, cell), (SPLITS, splits))


def test_a_store_counts_the_lookups_that_found_a_reading_and_those_that_did_not() -> (
    None
):
    store = a_store()
    store.put(an_observation())

    assert store.get(CELL) is not None

    with pytest.raises(KeyError, match="no reading is held at"):
        store.get(SPLITS)

    with pytest.raises(KeyError, match="no reading is held at"):
        store.get(SPLITS)

    assert store.hits == 1
    assert store.misses == 2
    assert store.evictions == 0


def test_asking_whether_a_store_holds_a_point_counts_neither_way() -> None:
    store = a_store()
    store.put(an_observation())

    assert CELL in store
    assert SPLITS not in store
    assert store.hits == 0
    assert store.misses == 0


def test_a_store_past_its_bound_drops_the_reading_that_has_gone_longest_unread() -> (
    None
):
    store = a_store(capacity=2)

    for index in range(2):
        store.put(an_observation(a_point(index), torch.zeros(2), axes=("width",)))

    store.put(an_observation(a_point(2), torch.zeros(2), axes=("width",)))

    assert len(store) == 2
    assert store.evictions == 1
    assert a_point(0) not in store
    assert list(store) == [a_point(1), a_point(2)]


def test_reading_one_back_moves_it_out_of_the_way_of_the_next_eviction() -> None:
    store = a_store(capacity=2)

    for index in range(2):
        store.put(an_observation(a_point(index), torch.zeros(2), axes=("width",)))

    assert store.recency() == (a_point(0), a_point(1))

    store.get(a_point(0))

    assert store.recency() == (a_point(1), a_point(0))

    store.put(an_observation(a_point(2), torch.zeros(2), axes=("width",)))

    assert a_point(0) in store
    assert a_point(1) not in store


def test_a_store_evicts_no_further_than_its_bound_however_many_arrive() -> None:
    store = a_store(capacity=3)
    arriving = 10

    for index in range(arriving):
        store.put(an_observation(a_point(index), torch.zeros(2), axes=("width",)))

    assert len(store) == 3
    assert store.evictions == arriving - 3
    assert list(store) == [a_point(7), a_point(8), a_point(9)]


def test_a_store_keeps_what_arrived_in_the_order_it_arrived_however_it_was_read() -> (
    None
):
    store = a_store()

    for index in range(3):
        store.put(an_observation(a_point(index), torch.zeros(2), axes=("width",)))

    store.get(a_point(0))
    store.get(a_point(1))

    assert list(store) == [a_point(0), a_point(1), a_point(2)]
    assert store.recency() == (a_point(2), a_point(0), a_point(1))


def test_a_reading_arriving_again_at_one_point_replaces_the_one_held_there() -> None:
    store = a_store()
    later = an_observation(tensor=torch.ones(2, 3, 4))

    store.put(an_observation())
    store.put(later)

    assert len(store) == 1
    assert store.get(CELL) is later
    assert store.evictions == 0


def test_a_store_accounts_for_the_bytes_its_readings_occupy() -> None:
    store = a_store()
    mask = torch.tensor([True, False, True, False])
    numbers = torch.zeros(4, dtype=torch.float32)

    store.put(an_observation(CELL, numbers, axes=("width",), mask=mask))

    assert store.nbytes == numbers.nelement() * 4 + mask.nelement()


def test_the_accounting_falls_back_as_a_reading_is_dropped() -> None:
    store = a_store(capacity=1)

    store.put(an_observation(a_point(0), torch.zeros(8), axes=("width",)))

    held = store.nbytes

    store.put(an_observation(a_point(1), torch.zeros(8), axes=("width",)))

    assert store.nbytes == held
    assert store.evictions == 1


def test_a_compressing_store_keeps_its_readings_at_half_precision() -> None:
    store = a_store(compress=True)
    numbers = torch.tensor([1.5, 2.5], dtype=torch.float32)

    store.put(an_observation(CELL, numbers, axes=("width",)))

    kept = store.get(CELL)

    assert kept.tensor.dtype is torch.float16
    assert torch.equal(kept.tensor.to(torch.float32), numbers)
    assert store.nbytes == numbers.nelement() * 2
    assert store.compresses is True


def test_a_store_that_does_not_compress_leaves_the_reading_as_it_came() -> None:
    store = a_store()
    numbers = torch.tensor([1.5, 2.5], dtype=torch.float32)

    store.put(an_observation(CELL, numbers, axes=("width",)))

    assert store.get(CELL).tensor.dtype is torch.float32
    assert store.nbytes == numbers.nelement() * 4


def test_compression_leaves_a_reading_that_has_no_half_precision_form_alone() -> None:
    store = a_store(compress=True)
    counts = torch.tensor([1, 2, 3], dtype=torch.int64)

    store.put(an_observation(CELL, counts, axes=("width",)))

    assert store.get(CELL).tensor.dtype is torch.int64


def test_compression_keeps_the_mask_over_a_reading_as_truth_values() -> None:
    store = a_store(compress=True)
    mask = torch.tensor([True, False])

    store.put(
        an_observation(CELL, torch.zeros(2, 3), axes=("item", "width"), mask=mask)
    )

    kept = store.get(CELL)

    assert kept.mask is not None
    assert kept.mask.dtype is torch.bool
    assert torch.equal(kept.mask, mask)


def test_compression_is_what_sends_a_number_beyond_half_precision_to_infinity() -> None:
    beyond = torch.tensor([65504.0 * 2], dtype=torch.float32)

    plain = a_store()
    plain.put(an_observation(CELL, beyond, axes=("width",)))

    compressing = a_store(compress=True)
    compressing.put(an_observation(CELL, beyond, axes=("width",)))

    assert bool(torch.isfinite(plain.get(CELL).tensor).all())
    assert not bool(torch.isfinite(compressing.get(CELL).tensor).all())


def test_a_store_that_could_hold_nothing_is_refused() -> None:
    with pytest.raises(ValueError, match="would drop every reading"):
        ReadingCache(capacity=0)


def test_a_bound_that_is_not_a_whole_count_is_refused() -> None:
    with pytest.raises(ValueError, match="capacity is negative"):
        ReadingCache(capacity=-1)

    with pytest.raises(TypeError, match="capacity is not an integer"):
        ReadingCache(capacity=cast("int", 2.0))


def test_a_compression_flag_that_is_not_a_boolean_is_refused() -> None:
    with pytest.raises(TypeError, match="compression flag is not a boolean"):
        ReadingCache(compress=cast("bool", "half"))


def test_a_store_holds_observations_and_nothing_else() -> None:
    store = a_store()

    with pytest.raises(TypeError, match="holds observations"):
        store.put(cast("Observation", a_reading()))


def test_a_store_says_what_it_holds_what_it_cost_and_what_it_counted() -> None:
    store = a_store(capacity=4)
    store.put(an_observation())

    assert repr(store) == (
        f"ReadingCache(1 of 4, bytes={store.nbytes!r}, hits=0, misses=0, "
        f"evictions=0, compresses=False)"
    )


def test_a_trace_keeps_its_readings_in_a_bounded_store() -> None:
    trace = Trace(capacity=2)

    for index in range(3):
        trace.record(an_observation(a_point(index), torch.zeros(2), axes=("width",)))

    assert trace.capacity == 2
    assert trace.evictions == 1
    assert len(trace) == 2
    assert trace.nbytes > 0

    with pytest.raises(KeyError, match="nothing was read at"):
        _ = trace[a_point(0)]

    assert trace.misses == 1

    assert trace[a_point(2)].rank == 1
    assert trace.hits == 1


def test_a_trace_keeps_every_reading_where_no_bound_was_asked_for() -> None:
    trace = Trace()

    for index in range(64):
        trace.record(an_observation(a_point(index), torch.zeros(2), axes=("width",)))

    assert trace.capacity is None
    assert trace.evictions == 0
    assert len(trace) == 64


def test_a_compressing_trace_records_at_half_precision() -> None:
    trace = Trace(compress=True)

    trace.record(an_observation())

    assert trace[CELL].tensor.dtype is torch.float16


def test_a_recurrent_pair_is_two_readings_at_two_points_rather_than_a_tuple() -> None:
    trace = Trace()
    hidden_at = Site(PARSER, Role.RECURRENT_HIDDEN, (0,))
    cell_at = Site(PARSER, Role.RECURRENT_CELL, (0,))
    hidden = torch.zeros(2, 3)
    cell = torch.ones(2, 3)

    trace.record(Observation(hidden_at, hidden, axes=("item", "width")))
    trace.record(Observation(cell_at, cell, axes=("item", "width")))

    assert trace.sites() == (hidden_at, cell_at)
    assert torch.equal(trace[hidden_at].tensor, hidden)
    assert torch.equal(trace[cell_at].tensor, cell)

    with pytest.raises(TypeError, match="reading is not a tensor"):
        Observation(
            hidden_at,
            cast("torch.Tensor", (hidden, cell)),
            axes=("item", "width"),
        )
