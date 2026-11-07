import itertools

import didactic.api as dx
import pytest
from hypothesis import given, reject, settings
from hypothesis import strategies as st

from lofi.run import ContractionError, GridDesignSpec
from lofi.run._boundary.experiment import GeneratorSpec, ResolutionIVRung
from lofi.run._contraction import (
    DropInOrder,
    ExpandedAxis,
    ExpandedSweep,
    Fallback,
    PreregisteredCells,
    ResolutionIVFraction,
    SeedFloor,
    point_key,
)


def axis(key: str, *values: str | int) -> ExpandedAxis:
    return ExpandedAxis(key, values, selects_group=False)


def sweep(
    *grid: ExpandedAxis,
    yoked: tuple[tuple[ExpandedAxis, ...], ...] = (),
    seeds: tuple[int, ...] = (0, 1, 2),
) -> ExpandedSweep:
    return ExpandedSweep(
        design=GridDesignSpec(),
        grid=grid,
        yoked=yoked,
        clauses=(),
        seeds=seeds,
        expected_points=99,
    )


def runs(contracted: ExpandedSweep) -> list[dict[str, int]]:
    """Return the sign of every axis at every position of the last yoked group."""
    group = contracted.yoked[-1]

    return [
        {
            column.key: -1 if value == 0 else 1
            for column, value in zip(group, position, strict=True)
        }
        for position in zip(*(column.points for column in group), strict=True)
    ]


def test_a_value_list_is_frozen_and_read_back_fresh() -> None:
    held = ExpandedAxis("loss", ([1, {"b": 2}],), selects_group=False)

    first = held.points[0]
    assert isinstance(first, list)
    first.append(3)

    assert held.points == ([1, {"b": 2}],)
    assert held == ExpandedAxis("loss", ([1, {"b": 2}],), selects_group=False)
    assert point_key({"b": 1, "a": 2}) == point_key({"a": 2, "b": 1})


def test_preregistered_cells_drop_unrealized_values_of_grid_and_yoked_axes() -> None:
    declared = sweep(
        axis("a", 1, 2, 3),
        yoked=((axis("b", "x", "y", "z"), axis("c", 7, 8, 9)),),
    )
    rung = PreregisteredCells(
        {"a": frozenset({point_key(1), point_key(2)}), "b": frozenset({point_key("y")})}
    )

    contracted = rung.contract(declared)

    assert rung.step == "unpreregistered-cells"
    assert contracted.sweep.grid == (axis("a", 1, 2),)
    assert contracted.sweep.yoked == ((axis("b", "y"), axis("c", 8)),)
    assert contracted.dropped == (3, "x", "z")
    assert contracted.sweep.expected_points is None


def test_a_rung_refuses_to_empty_an_axis_or_name_an_absent_one() -> None:
    declared = sweep(axis("a", 1, 2))

    with pytest.raises(ContractionError, match="every value"):
        PreregisteredCells({"a": frozenset()}).contract(declared)

    with pytest.raises(ContractionError, match="no axis of the sweep assigns 'z'"):
        PreregisteredCells({"z": frozenset()}).contract(declared)


def test_drop_in_order_removes_the_next_listed_value_still_assigned() -> None:
    rung = DropInOrder("languages", "lang", ("nld", "eng"))
    declared = sweep(axis("lang", "deu", "eng", "nld"))

    once = rung.contract(declared)
    twice = rung.contract(once.sweep)

    assert once.dropped == ("nld",)
    assert twice.dropped == ("eng",)
    assert twice.sweep.grid == (axis("lang", "deu"),)

    with pytest.raises(ContractionError, match="no value left"):
        rung.contract(twice.sweep)


def test_the_seed_floor_keeps_the_first_seeds() -> None:
    contracted = SeedFloor(3).contract(sweep(axis("a", 1), seeds=(5, 6, 7, 8, 9)))

    assert contracted.sweep.seeds == (5, 6, 7)
    assert contracted.dropped == (8, 9)

    with pytest.raises(ContractionError, match="already at the floor"):
        SeedFloor(3).contract(contracted.sweep)


def test_the_fallback_keeps_the_sweep() -> None:
    declared = sweep(axis("a", 1, 2))

    assert Fallback().contract(declared).sweep == declared
    assert Fallback().step == "fallback"


def test_a_quarter_fraction_of_six_factors_has_sixteen_runs_clear_of_interactions() -> (
    None
):
    names = "abcdef"
    declared = sweep(*(axis(name, 0, 1) for name in names), axis("other", "x", "y"))
    fraction = ResolutionIVFraction((("e", ("a", "b", "c")), ("f", ("b", "c", "d"))))

    contracted = fraction.contract(declared).sweep
    signs = runs(contracted)

    assert contracted.grid == (axis("other", "x", "y"),)
    assert len(signs) == 16
    assert len({tuple(run.values()) for run in signs}) == 16

    for factor in names:
        for left, right in itertools.combinations(names, 2):
            assert sum(run[factor] * run[left] * run[right] for run in signs) == 0


def test_a_fraction_reads_each_axis_first_value_as_its_negative_sign() -> None:
    names = "abcde"
    declared = sweep(*(axis(name, 0, 1) for name in names))
    fraction = ResolutionIVFraction((("e", ("a", "b", "c", "d")),))

    signs = runs(fraction.contract(declared).sweep)

    assert len(signs) == 16
    assert all(run["e"] == run["a"] * run["b"] * run["c"] * run["d"] for run in signs)


def test_the_fraction_refuses_an_axis_it_cannot_halve() -> None:
    fraction = ResolutionIVFraction((("d", ("a", "b", "c")),))

    with pytest.raises(ContractionError, match="exactly two values"):
        fraction.contract(sweep(axis("a", 0, 1, 2), *(axis(n, 0, 1) for n in "bcd")))

    with pytest.raises(ContractionError, match="not grid axes"):
        fraction.contract(sweep(*(axis(n, 0, 1) for n in "abc")))


@st.composite
def resolution_iv_designs(
    draw: st.DrawFn,
) -> tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]]:
    basics = tuple(f"x{index}" for index in range(draw(st.integers(3, 6))))
    count = draw(st.integers(1, 3))
    generators = tuple(
        (
            f"g{index}",
            tuple(
                draw(
                    st.lists(
                        st.sampled_from(basics),
                        min_size=3,
                        max_size=len(basics),
                        unique=True,
                    )
                )
            ),
        )
        for index in range(count)
    )

    try:
        ResolutionIVRung(
            generators=tuple(
                GeneratorSpec(factor=factor, word=word) for factor, word in generators
            )
        )

    except dx.ValidationError:
        reject()

    return basics, generators


@settings(max_examples=40, deadline=None)
@given(resolution_iv_designs())
def test_every_declared_resolution_iv_fraction_clears_main_effects_of_interactions(
    design: tuple[tuple[str, ...], tuple[tuple[str, tuple[str, ...]], ...]],
) -> None:
    basics, generators = design
    names = (*basics, *(factor for factor, _ in generators))
    used = {key for _, word in generators for key in word}
    contracted = ResolutionIVFraction(generators).contract(
        sweep(*(axis(name, 0, 1) for name in names))
    )
    signs = runs(contracted.sweep)
    design_names = tuple(signs[0])

    assert tuple(column.key for column in contracted.sweep.grid) == tuple(
        name for name in basics if name not in used
    )
    assert len(signs) == 2 ** len(used)

    for factor in design_names:
        assert sum(run[factor] for run in signs) == 0

        for left, right in itertools.combinations(design_names, 2):
            assert sum(run[factor] * run[left] * run[right] for run in signs) == 0
