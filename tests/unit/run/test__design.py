"""Tests for installed sweep-design plugins."""

from __future__ import annotations

from importlib.metadata import EntryPoint, EntryPoints
from typing import TYPE_CHECKING, cast

import pytest
from tests.support.installed import install

from lofi.run import (
    Axis,
    FractionalFactorialDesignSpec,
    GeneratorSpec,
    GridDesignSpec,
    LatinHypercubeDesignSpec,
    RandomDesignSpec,
    Sweep,
    design_assignments,
)

if TYPE_CHECKING:
    import lofi.run._design as design_module

DESIGN_POINTS = EntryPoints(
    tuple(
        EntryPoint(name=name, value=value, group="lofi.sweep_design")
        for name, value in (
            ("grid", "lofi.run._design:GRID_PLUGIN"),
            ("random", "lofi.run._design:RANDOM_PLUGIN"),
            ("latin-hypercube", "lofi.run._design:LATIN_HYPERCUBE_PLUGIN"),
            (
                "fractional-factorial",
                "lofi.run._design:FRACTIONAL_FACTORIAL_PLUGIN",
            ),
        )
    )
)


@pytest.fixture(autouse=True)
def _design_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    install(monkeypatch, "sweep_design", DESIGN_POINTS)


def _record(
    assignment: design_module.SweepAssignment,
) -> tuple[tuple[str, object], ...]:
    return tuple((axis.key, value) for axis, value in assignment)


def test_grid_design_crosses_grid_axes_and_advances_yoked_axes_together() -> None:
    sweep = Sweep(
        grid=(Axis(path="width", values=(1, 2)),),
        yoked=(
            (
                Axis(path="depth", values=(3, 4)),
                Axis(path="active", values=(True, False)),
            ),
        ),
    )

    assert tuple(map(_record, design_assignments(GridDesignSpec(), sweep))) == (
        (("width", 1), ("depth", 3), ("active", True)),
        (("width", 1), ("depth", 4), ("active", False)),
        (("width", 2), ("depth", 3), ("active", True)),
        (("width", 2), ("depth", 4), ("active", False)),
    )


def test_random_design_is_seeded_and_samples_without_replacement() -> None:
    sweep = Sweep(
        grid=(
            Axis(path="width", values=(1, 2)),
            Axis(path="depth", values=(3, 4)),
        )
    )
    declaration = RandomDesignSpec(points=3, seed=8)
    first = tuple(map(_record, design_assignments(declaration, sweep)))

    assert first == tuple(map(_record, design_assignments(declaration, sweep)))
    assert len(first) == len(set(first)) == 3


def test_latin_hypercube_design_stratifies_each_dimension() -> None:
    sweep = Sweep(
        grid=(
            Axis(path="width", values=(0, 1, 2, 3)),
            Axis(path="depth", values=(4, 5, 6, 7)),
        )
    )
    assignments = design_assignments(LatinHypercubeDesignSpec(points=4, seed=11), sweep)
    records = tuple(dict(_record(assignment)) for assignment in assignments)

    assert sorted(cast("int", record["width"]) for record in records) == [0, 1, 2, 3]
    assert sorted(cast("int", record["depth"]) for record in records) == [4, 5, 6, 7]


def test_fractional_factorial_design_obeys_the_declared_generator() -> None:
    sweep = Sweep(
        grid=tuple(Axis(path=key, values=(0, 1)) for key in ("a", "b", "c", "d"))
    )
    declaration = FractionalFactorialDesignSpec(
        generators=(GeneratorSpec(factor="d", word=("a", "b", "c")),)
    )
    records = tuple(
        dict(_record(assignment))
        for assignment in design_assignments(declaration, sweep)
    )

    assert len(records) == 8
    assert all(
        cast("int", record["d"])
        == sum(cast("int", record[key]) for key in ("a", "b", "c")) % 2
        for record in records
    )
