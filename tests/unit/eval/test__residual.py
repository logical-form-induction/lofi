from typing import TYPE_CHECKING, cast

import pytest

from lofi.eval._rescoring import DeclaredBudget
from lofi.eval._residual import (
    NO_ENUMERATION_BUDGET,
    RESIDUAL_SOURCES,
    Approximation,
    Budget,
    NoEnumerationBudget,
    Residual,
    ResidualSummary,
    residual_summary,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.eval._residual import ResidualSource


class _Budget:
    __slots__ = ("_beam", "_depth", "_mass")

    def __init__(self, depth: int, beam: int, mass: float | None = None) -> None:
        self._depth = depth
        self._beam = beam
        self._mass = mass

    @property
    def depth(self) -> int:
        """Return how many levels an enumeration was allowed to produce."""
        return self._depth

    @property
    def beam(self) -> int:
        """Return how many candidates a search kept."""
        return self._beam

    @property
    def mass(self) -> float | None:
        """Return the largest omitted mass, if a bound was set."""
        return self._mass


def _measured(costs: Mapping[str, float]) -> Residual:
    return Residual(cast("Mapping[ResidualSource, float]", costs))


def _full() -> Residual:
    return Residual(dict.fromkeys(RESIDUAL_SOURCES, 0.25))


def test_a_row_carries_its_budget_and_the_residual_measured_under_it() -> None:
    approximation = Approximation(_Budget(3, 8, 0.05), residual=_full())
    spec = approximation.to_spec()

    assert approximation.is_measured
    assert spec["budget_depth"] == 3
    assert spec["budget_beam"] == 8
    assert spec["budget_mass"] == 0.05
    assert spec["residual_measured"] is True
    assert all(spec[f"residual_{source}"] == 0.25 for source in RESIDUAL_SOURCES)


def test_an_unmeasured_residual_is_absent_rather_than_zero() -> None:
    approximation = Approximation(_Budget(3, 8), residual=None)
    spec = approximation.to_spec()

    assert approximation.residual is None
    assert not approximation.is_measured
    assert spec["residual_measured"] is False
    assert spec["budget_mass"] is None
    for source in RESIDUAL_SOURCES:
        assert spec[f"residual_{source}"] is None


def test_a_measured_zero_and_an_unmeasured_source_are_different_rows() -> None:
    zeroed = Approximation(
        _Budget(3, 8), residual=Residual(dict.fromkeys(RESIDUAL_SOURCES, 0.0))
    )
    unmeasured = Approximation(_Budget(3, 8), residual=None)
    partial = Approximation(_Budget(3, 8), residual=_measured({"chart-cap": 0.0}))

    assert zeroed.is_measured
    assert not unmeasured.is_measured
    assert not partial.is_measured
    assert zeroed != unmeasured
    assert zeroed.to_spec()["residual_chart-cap"] == 0.0
    assert unmeasured.to_spec()["residual_chart-cap"] is None
    assert partial.to_spec()["residual_chart-cap"] == 0.0
    assert partial.to_spec()["residual_beam-truncation"] is None


def test_a_residual_names_the_sources_nobody_measured() -> None:
    partial = _measured({"beam-truncation": 0.3, "prover-timeout": 2.0})

    assert partial.unmeasured == ("chart-cap", "enumeration-budget")
    assert not partial.is_measured
    assert _full().unmeasured == ()
    assert _full().is_measured
    assert Residual({}).unmeasured == RESIDUAL_SOURCES


def test_a_residual_serializes_every_source_as_a_column() -> None:
    spec = _measured({"chart-cap": 7.0}).to_spec()

    assert list(spec) == list(RESIDUAL_SOURCES)
    assert spec["chart-cap"] == 7.0
    assert spec["beam-truncation"] is None


def test_a_summary_counts_the_rows_that_measured_nothing() -> None:
    rows = [
        Approximation(_Budget(3, 8), residual=_full()),
        Approximation(_Budget(3, 8), residual=None),
        Approximation(_Budget(4, 8), residual=_measured({"chart-cap": 1.0})),
    ]
    summary = residual_summary(rows)

    assert summary.rows == 3
    assert summary.measured == 1
    assert summary.unmeasured == 2
    assert summary.by_source["chart-cap"] == 1
    assert summary.by_source["beam-truncation"] == 2
    assert summary.by_source["prover-timeout"] == 2


def test_a_summary_over_no_rows_counts_nothing() -> None:
    summary = residual_summary([])

    assert summary.rows == 0
    assert summary.measured == 0
    assert summary.unmeasured == 0
    assert dict(summary.by_source) == {}
    assert all(summary.by_source.get(source, 0) == 0 for source in RESIDUAL_SOURCES)


def test_a_summary_refuses_a_row_that_is_not_an_approximation() -> None:
    with pytest.raises(TypeError, match="not one"):
        residual_summary([cast("Approximation", _Budget(3, 8))])


def test_a_residual_refuses_an_undeclared_source_or_a_negative_cost() -> None:
    with pytest.raises(TypeError, match="not a declared one"):
        _measured({"gpu-memory": 1.0})

    with pytest.raises(ValueError, match="is negative"):
        _measured({"chart-cap": -1.0})

    with pytest.raises(ValueError, match="not finite"):
        _measured({"chart-cap": float("nan")})

    with pytest.raises(TypeError, match="not a number"):
        _measured({"chart-cap": cast("float", "a lot")})


def test_an_approximation_refuses_a_budget_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="publishes no depth"):
        Approximation(cast("Budget", 3), residual=None)

    with pytest.raises(TypeError, match="residual is not one"):
        Approximation(_Budget(3, 8), residual=cast("Residual", 0.5))

    with pytest.raises(ValueError, match="depth is negative"):
        Approximation(_Budget(-1, 8), residual=None)

    with pytest.raises(TypeError, match="mass bound is not a number"):
        Approximation(_Budget(3, 8, cast("float", "half")), residual=None)


def test_an_approximation_is_equal_on_the_accuracy_it_bought() -> None:
    one = Approximation(_Budget(3, 8, 0.1), residual=_full())
    other = Approximation(_Budget(3, 8, 0.1), residual=_full())

    assert one == other
    assert hash(one) == hash(other)
    assert one != Approximation(_Budget(4, 8, 0.1), residual=_full())
    assert one != Approximation(_Budget(3, 8, 0.1), residual=None)
    assert one != "a budget"
    assert repr(one).startswith("Approximation(")

    for field in ("budget", "residual", "is_measured"):
        with pytest.raises(AttributeError):
            setattr(one, field, None)


def test_a_residual_and_a_summary_are_equal_hashable_and_frozen() -> None:
    one = _full()
    summary = residual_summary([Approximation(_Budget(3, 8), residual=one)])

    assert one == _full()
    assert hash(one) == hash(_full())
    assert one != Residual({})
    assert one != "a residual"
    assert repr(one).startswith("Residual(")
    assert summary == residual_summary([Approximation(_Budget(9, 9), residual=_full())])
    assert hash(summary) == hash(
        residual_summary([Approximation(_Budget(9, 9), residual=_full())])
    )
    assert summary != residual_summary([])
    assert summary != "a summary"
    assert repr(summary).startswith("ResidualSummary(1, 1, ")
    assert summary != residual_summary(
        [Approximation(NO_ENUMERATION_BUDGET, residual=None)]
    )

    for field in ("measurements", "unmeasured", "is_measured"):
        with pytest.raises(AttributeError):
            setattr(one, field, None)


def test_a_summary_refuses_more_measured_rows_than_rows() -> None:
    with pytest.raises(ValueError, match="measured rows and 0 rows"):
        ResidualSummary(1, 2, {}, not_applicable=0)

    with pytest.raises(ValueError, match="row count is negative"):
        ResidualSummary(-1, 0, {}, not_applicable=0)


def test_a_row_under_no_enumeration_budget_declares_so_in_its_columns() -> None:
    approximation = Approximation(NO_ENUMERATION_BUDGET, residual=None)
    enumerated = Approximation(_Budget(3, 8, 0.05), residual=_full())

    assert approximation.to_spec() == {
        "budget_kind": "none",
        "budget_depth": None,
        "budget_beam": None,
        "budget_mass": None,
        "residual_measured": False,
        **{f"residual_{source}": None for source in RESIDUAL_SOURCES},
    }
    assert enumerated.to_spec()["budget_kind"] == "enumeration"
    assert not approximation.applies
    assert enumerated.applies
    assert not approximation.is_measured


def test_no_enumeration_budget_is_a_declared_value_not_a_zero_budget() -> None:
    declared = Approximation(NoEnumerationBudget(), residual=None)

    assert declared == Approximation(NO_ENUMERATION_BUDGET, residual=None)
    assert hash(declared) == hash(Approximation(NO_ENUMERATION_BUDGET, residual=None))
    assert declared != Approximation(DeclaredBudget(0, 0, mass=None), residual=None)
    assert NoEnumerationBudget() == NO_ENUMERATION_BUDGET
    assert DeclaredBudget(0, 0, mass=None) != NO_ENUMERATION_BUDGET
    assert repr(NO_ENUMERATION_BUDGET) == "NoEnumerationBudget()"
    assert not isinstance(NO_ENUMERATION_BUDGET, Budget)


def test_a_residual_is_refused_where_no_enumeration_budget_applies() -> None:
    with pytest.raises(ValueError, match="no enumeration budget"):
        Approximation(NO_ENUMERATION_BUDGET, residual=_full())


def test_a_summary_counts_rows_under_no_enumeration_budget_apart() -> None:
    rows = [
        Approximation(_Budget(3, 8), residual=_full()),
        Approximation(_Budget(3, 8), residual=None),
        Approximation(NO_ENUMERATION_BUDGET, residual=None),
        Approximation(NO_ENUMERATION_BUDGET, residual=None),
    ]
    summary = residual_summary(rows)

    assert summary.rows == 4
    assert summary.measured == 1
    assert summary.unmeasured == 1
    assert summary.not_applicable == 2
    assert dict(summary.by_source) == dict.fromkeys(RESIDUAL_SOURCES, 1)


def test_a_summary_refuses_more_classified_rows_than_rows() -> None:
    with pytest.raises(ValueError, match="among 2 rows"):
        ResidualSummary(2, 1, {}, not_applicable=2)
