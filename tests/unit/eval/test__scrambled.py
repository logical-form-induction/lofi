import random
import subprocess
import sys
from pathlib import Path
from typing import cast

import pytest

from lofi.eval._scrambled import (
    EmpiricalTwinNull,
    SignatureAssignments,
    TwinFit,
    empirical_twin_null,
    sample_exchangeable_signature_twins,
    sample_signature_twins,
)


def _assignments(
    primitives: tuple[str, ...] = ("S", "NP", "N"),
    constructors: tuple[str, ...] = ("forward", "backward"),
    combinators: tuple[str, ...] = ("application", "composition"),
) -> SignatureAssignments:
    return SignatureAssignments(
        primitives,
        constructors,
        combinators,
    )


def test_signature_assignments_are_frozen_hashable_values() -> None:
    assignments = _assignments()

    assert assignments.primitives == ("S", "NP", "N")
    assert assignments.constructors == ("forward", "backward")
    assert assignments.combinators == ("application", "composition")
    assert assignments.scramble_space == 23
    assert assignments == _assignments()
    assert hash(assignments) == hash(_assignments())

    with pytest.raises(AttributeError):
        assignments.__setattr__("primitives", ("N",))


@pytest.mark.parametrize(
    ("field", "value", "error"),
    [
        ("primitives", ("S", "S"), ValueError),
        ("constructors", ("forward", ""), ValueError),
        ("combinators", ("application", 4), TypeError),
        ("primitives", "S", TypeError),
    ],
)
def test_signature_assignments_refuse_invalid_inventories(
    field: str,
    value: tuple[str, ...] | tuple[str, int] | str | int,
    error: type[Exception],
) -> None:
    supplied: dict[str, tuple[str, ...] | tuple[str, int] | str | int | bool] = {
        "primitives": ("S", "NP"),
        "constructors": ("forward", "backward"),
        "combinators": ("application", "composition"),
    }
    supplied[field] = value

    with pytest.raises(error):
        SignatureAssignments(
            supplied["primitives"],
            supplied["constructors"],
            supplied["combinators"],
        )


def test_signature_twin_sampling_is_distinct_inventory_matched_and_local() -> None:
    observed = _assignments()
    global_state = random.getstate()
    twins = sample_signature_twins(observed, count=12, seed=29)

    assert random.getstate() == global_state
    assert twins == sample_signature_twins(observed, count=12, seed=29)
    assert len(twins) == 12
    assert len(set(twins)) == 12
    assert observed not in twins

    for twin in twins:
        assert set(twin.primitives) == set(observed.primitives)
        assert set(twin.constructors) == set(observed.constructors)
        assert set(twin.combinators) == set(observed.combinators)
        assert len(twin.primitives) == len(observed.primitives)
        assert len(twin.constructors) == len(observed.constructors)
        assert len(twin.combinators) == len(observed.combinators)


def test_signature_twin_sampling_can_exhaust_the_exact_finite_space() -> None:
    observed = _assignments(
        primitives=("S", "NP"),
        constructors=("forward", "backward"),
        combinators=("application",),
    )
    twins = sample_signature_twins(observed, count=3, seed=7)

    assert observed.scramble_space == 3
    assert len(set(twins)) == observed.scramble_space
    assert observed not in twins


def test_signature_twins_only_exchange_constructors_with_the_same_shape() -> None:
    observed = _assignments(
        primitives=("S", "NP", "N"),
        constructors=("forward", "backward", "feature"),
        combinators=("application", "composition"),
    )
    twins = sample_exchangeable_signature_twins(
        observed,
        primitive_shapes=("atomic", "atomic", "atomic"),
        constructor_shapes=("binary", "binary", "unary"),
        count=3,
        seed=19,
    )

    assert len(set(twins)) == 3
    assert all(twin.constructors[2] == "feature" for twin in twins)
    assert all(set(twin.constructors[:2]) == {"forward", "backward"} for twin in twins)


def test_signature_twin_sampling_handles_a_space_larger_than_platform_range() -> None:
    observed = _assignments(
        primitives=tuple(f"P{index}" for index in range(21)),
        constructors=(),
        combinators=(),
    )

    twins = sample_signature_twins(observed, count=3, seed=31)

    assert observed.scramble_space > sys.maxsize
    assert len(set(twins)) == 3
    assert observed not in twins


def test_signature_twin_sampling_refuses_an_invalid_or_undersized_request() -> None:
    observed = _assignments(
        primitives=("S", "NP"),
        constructors=("forward", "backward"),
        combinators=("application",),
    )

    with pytest.raises(ValueError, match="fewer than three"):
        sample_signature_twins(observed, count=2, seed=0)

    with pytest.raises(ValueError, match="fewer than the 4 requested"):
        sample_signature_twins(observed, count=4, seed=0)

    with pytest.raises(ValueError, match="negative"):
        sample_signature_twins(observed, count=3, seed=-1)

    with pytest.raises(TypeError):
        sample_signature_twins(observed, count=True, seed=0)

    with pytest.raises(TypeError):
        sample_signature_twins(
            cast("SignatureAssignments", ("not", "assignments")),
            count=3,
            seed=0,
        )


def test_twin_fit_requires_finite_scalar_elpd() -> None:
    assignments = _assignments()
    fit = TwinFit(assignments, -14)

    assert fit.assignments is assignments
    assert fit.elpd == -14.0
    assert fit == TwinFit(assignments, -14.0)
    assert hash(fit) == hash(TwinFit(assignments, -14.0))

    with pytest.raises(ValueError, match="not finite"):
        TwinFit(assignments, float("nan"))

    with pytest.raises(ValueError, match="not finite"):
        TwinFit(assignments, float("inf"))

    with pytest.raises(TypeError):
        TwinFit(assignments, elpd=True)

    with pytest.raises(TypeError):
        TwinFit(cast("SignatureAssignments", ("not", "assignments")), -14.0)


def test_empirical_twin_null_retains_ordered_fits_deltas_and_midrank() -> None:
    observed_assignments = _assignments(
        primitives=("S", "NP"),
        constructors=("forward", "backward"),
    )
    twin_assignments = sample_signature_twins(
        observed_assignments,
        count=3,
        seed=3,
    )
    observed = TwinFit(observed_assignments, 5.0)
    twins = tuple(
        TwinFit(assignments, elpd)
        for assignments, elpd in zip(
            twin_assignments,
            (3.0, 5.0, 7.0),
            strict=True,
        )
    )
    result = empirical_twin_null(observed, twins, declared_twins=3)

    assert result.observed is observed
    assert result.twins == twins
    assert result.declared_twins == 3
    assert result.deltas == (2.0, 0.0, -2.0)
    assert result.quantile == 0.5
    assert result == EmpiricalTwinNull(observed, twins, declared_twins=3)
    assert hash(result) == hash(EmpiricalTwinNull(observed, twins, declared_twins=3))

    with pytest.raises(AttributeError):
        result.__setattr__("quantile", 0.0)


def test_empirical_twin_null_uses_the_full_tied_midrank_distribution() -> None:
    observed_assignments = _assignments()
    twin_assignments = sample_signature_twins(observed_assignments, count=4, seed=11)
    observed = TwinFit(observed_assignments, 1.0)
    twins = tuple(
        TwinFit(assignments, elpd)
        for assignments, elpd in zip(
            twin_assignments,
            (0.0, 0.5, 1.0, 2.0),
            strict=True,
        )
    )

    result = empirical_twin_null(observed, twins, declared_twins=3)

    assert result.deltas == (1.0, 0.5, 0.0, -1.0)
    assert result.quantile == 0.625
    assert len(result.twins) == 4


def test_empirical_twin_null_refuses_incomplete_or_invalid_fits() -> None:
    observed_assignments = _assignments()
    twin_assignments = sample_signature_twins(observed_assignments, count=4, seed=17)
    observed = TwinFit(observed_assignments, 2.0)
    fits = tuple(TwinFit(assignments, 1.0) for assignments in twin_assignments)

    with pytest.raises(ValueError, match="fewer than three"):
        empirical_twin_null(observed, fits, declared_twins=2)

    with pytest.raises(ValueError, match="2 fits for 3 declared"):
        empirical_twin_null(observed, fits[:2], declared_twins=3)

    with pytest.raises(ValueError, match="identity assignment"):
        empirical_twin_null(
            observed,
            (TwinFit(observed_assignments, 1.0), *fits[:2]),
            declared_twins=3,
        )

    with pytest.raises(ValueError, match="repeats an earlier assignment"):
        empirical_twin_null(
            observed,
            (fits[0], fits[0], fits[1]),
            declared_twins=3,
        )

    changed_inventory = _assignments(primitives=("S", "VP", "N"))

    with pytest.raises(ValueError, match="changes a signature-family inventory"):
        empirical_twin_null(
            observed,
            (TwinFit(changed_inventory, 1.0), *fits[:2]),
            declared_twins=3,
        )

    with pytest.raises(TypeError):
        empirical_twin_null(
            observed,
            (fits[0], cast("TwinFit", "not a fit"), fits[1]),
            declared_twins=3,
        )

    with pytest.raises(TypeError):
        empirical_twin_null(cast("TwinFit", "not a fit"), fits, declared_twins=3)


def test_scrambled_kernel_imports_no_fitting_or_optional_elpd_module() -> None:
    script = """
import sys

import lofi.eval._scrambled

assert "lofi.eval.elpd" not in sys.modules
assert "lofi.train" not in sys.modules
"""
    finished = subprocess.run(
        [sys.executable, "-"],
        input=script,
        capture_output=True,
        encoding="utf-8",
        check=False,
        cwd=Path(__file__).parents[3],
    )

    assert finished.returncode == 0, finished.stderr
