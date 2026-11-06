from typing import TYPE_CHECKING

from lofi.deduce import Cell, SpanIndex

if TYPE_CHECKING:
    from lofi.formalism._ports import Region, RegionIndex

# half-open spans of a three-token input, shortest first.
REGIONS_AT_THREE = [(0, 1), (1, 2), (2, 3), (0, 2), (1, 3), (0, 3)]


def _fanout(index: RegionIndex[Cell]) -> int:
    return index.fanout


def _leaves(index: RegionIndex[Cell]) -> list[Cell]:
    return list(index.leaves())


def _regions(index: RegionIndex[Cell]) -> list[Cell]:
    return list(index.regions_by_size())


def _decompositions(
    index: RegionIndex[Cell], region: Cell, arity: int
) -> list[tuple[Cell, ...]]:
    return list(index.decompositions(region, arity))


def _sweep(index: RegionIndex[Cell]) -> list[tuple[Cell, Cell, Cell]]:
    return [
        (region, parts[0], parts[1])
        for region in index.regions_by_size()
        for parts in index.decompositions(region, 2)
    ]


def _reached_after_its_parts(index: RegionIndex[Cell], arity: int) -> bool:
    reached: set[Cell] = set()

    for region in index.regions_by_size():
        for parts in index.decompositions(region, arity):
            if any(part not in reached for part in parts):
                return False

        reached.add(region)

    return True


def _chart(index: RegionIndex[Cell]) -> dict[Region, int]:
    chart: dict[Region, int] = {
        region: order for order, region in enumerate(index.regions_by_size())
    }

    return chart


def test_the_span_index_is_the_region_index_at_fan_out_one() -> None:
    assert _fanout(SpanIndex(4)) == 1
    assert _fanout(SpanIndex(0)) == 1


def test_the_leaves_are_the_one_token_regions() -> None:
    assert _leaves(SpanIndex(4)) == [(0, 1), (1, 2), (2, 3), (3, 4)]
    assert _leaves(SpanIndex(1)) == [(0, 1)]
    assert _leaves(SpanIndex(0)) == []


def test_regions_by_size_is_the_span_enumeration() -> None:
    assert _regions(SpanIndex(3)) == REGIONS_AT_THREE
    assert _regions(SpanIndex(5)) == list(SpanIndex(5).spans_by_length())
    assert _regions(SpanIndex(0)) == []


def test_a_region_is_reached_only_after_the_regions_it_divides_into() -> None:
    index = SpanIndex(5)

    for arity in (2, 3):
        assert _reached_after_its_parts(index, arity)


def test_decompositions_through_the_protocol_are_the_split_pairs() -> None:
    index = SpanIndex(3)

    assert _decompositions(index, (0, 3), 2) == [((0, 1), (1, 3)), ((0, 2), (2, 3))]
    assert _decompositions(index, (0, 3), 1) == [((0, 3),)]
    assert _decompositions(index, (0, 3), 0) == []
    assert _decompositions(index, (2, 3), 2) == []


def test_a_sweep_against_the_protocol_answers_as_the_span_sweep_does() -> None:
    assert len(_sweep(SpanIndex(5))) == 20

    for n in range(6):
        index = SpanIndex(n)

        assert _sweep(index) == list(index.binary_sweep())


def test_a_chart_addresses_one_cell_per_region() -> None:
    index = SpanIndex(4)
    chart = _chart(index)

    assert len(chart) == index.num_spans
    assert chart[(0, 1)] == 0
    assert chart[(0, 4)] == index.num_spans - 1
    assert chart[index.span_of(index.span_id(1, 3))] == chart[(1, 3)]
