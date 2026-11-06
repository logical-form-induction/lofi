import json
from typing import cast

import pytest

from lofi.eval._prereg import (
    DeclaredHypothesis,
    Preregistration,
    PreregistrationAudit,
    PreregistrationError,
    UndeclaredFamilyError,
    check_preregistration,
    read_preregistration,
)
from lofi.eval._values import Comparison
from lofi.stats import Interval, p_value_against_replicates

_GRID: dict[str, list[str | int]] = {
    "grammar": ["ab", "ccg", "tlg"],
    "inventory": [4, 8],
}


def _hypotheses() -> list[DeclaredHypothesis]:
    return [
        DeclaredHypothesis(
            "ccg-over-ab", "ccg", "ab", family="grammar-comparison", margin=0.01
        ),
        DeclaredHypothesis("tlg-over-ab", "tlg", "ab", family="grammar-comparison"),
        DeclaredHypothesis(
            "wide-over-narrow", "wide", "narrow", family="inventory-comparison"
        ),
    ]


def _plan() -> Preregistration:
    return Preregistration(_GRID, _hypotheses())


def _comparison(left: str, right: str, *, family: str) -> Comparison:
    return Comparison(
        left,
        right,
        statistic=1.4,
        p_value=p_value_against_replicates(
            1.4,
            [0.2, 0.1, 0.5],
            tail="greater",
            null="permuted-arms",
            family="permutation",
            exact=False,
            seed=0,
        ),
        effect=0.2,
        falsifier=f"{left} scores no better than {right}",
        smoke_test=False,
        family=family,
        ci=Interval(0.1, 0.3, 0.95),
    )


def test_the_plan_carries_its_grid_and_the_size_that_grid_crosses_to() -> None:
    plan = _plan()

    assert dict(plan.grid) == {"grammar": ("ab", "ccg", "tlg"), "inventory": (4, 8)}
    assert plan.grid_size == 6
    assert plan.declared == 3
    assert plan.families == frozenset({"grammar-comparison", "inventory-comparison"})
    assert plan == _plan()
    assert hash(plan) == hash(_plan())
    assert plan != "a plan"
    assert repr(plan).startswith("Preregistration({'grammar'")


def test_a_hypothesis_is_found_under_either_ordering_of_its_arms() -> None:
    plan = _plan()

    assert plan.hypothesis("ccg", "ab").name == "ccg-over-ab"
    assert plan.hypothesis("ab", "ccg").name == "ccg-over-ab"
    assert plan.hypothesis("ab", "ccg").pair == ("ab", "ccg")
    assert plan.margin("ab", "ccg") == 0.01

    with pytest.raises(PreregistrationError, match="declares no hypothesis"):
        plan.hypothesis("ccg", "tlg")


def test_a_margin_for_an_undeclared_pair_is_an_error_and_never_a_default() -> None:
    plan = _plan()

    with pytest.raises(PreregistrationError, match="declares no hypothesis"):
        plan.margin("ccg", "tlg")

    with pytest.raises(PreregistrationError, match="declares no equivalence margin"):
        plan.margin("tlg", "ab")


def test_a_hypothesis_reads_out_with_its_family_and_its_margin() -> None:
    declared = _hypotheses()[0]

    assert declared.to_spec() == {
        "name": "ccg-over-ab",
        "left": "ccg",
        "right": "ab",
        "family": "grammar-comparison",
        "margin": 0.01,
    }
    assert _hypotheses()[1].to_spec()["margin"] is None
    assert declared == _hypotheses()[0]
    assert hash(declared) == hash(_hypotheses()[0])
    assert declared != _hypotheses()[1]
    assert declared != "ccg-over-ab"
    assert repr(declared).startswith("DeclaredHypothesis('ccg-over-ab'")


def test_a_hypothesis_refuses_one_arm_twice_or_a_margin_that_is_not_positive() -> None:
    with pytest.raises(ValueError, match="one arm on both sides"):
        DeclaredHypothesis("h", "ccg", "ccg", family="f")

    with pytest.raises(ValueError, match="margin that is not positive"):
        DeclaredHypothesis("h", "ccg", "ab", family="f", margin=0.0)

    with pytest.raises(ValueError, match="family is empty"):
        DeclaredHypothesis("h", "ccg", "ab", family="")


def test_a_plan_refuses_an_empty_factor_a_repeated_level_or_no_hypotheses() -> None:
    with pytest.raises(ValueError, match="declares no levels"):
        Preregistration({"grammar": []}, _hypotheses())

    with pytest.raises(ValueError, match="the level 'ab' twice"):
        Preregistration({"grammar": ["ab", "ab"]}, _hypotheses())

    with pytest.raises(ValueError, match="declares no hypotheses at all"):
        Preregistration(_GRID, [])

    with pytest.raises(TypeError, match="declared hypothesis is not one"):
        Preregistration(_GRID, [cast("DeclaredHypothesis", "ccg-over-ab")])


def test_a_plan_refuses_one_pair_or_one_name_declared_twice() -> None:
    with pytest.raises(ValueError, match="both declare the comparison"):
        Preregistration(
            _GRID,
            [
                DeclaredHypothesis("one", "ccg", "ab", family="f"),
                DeclaredHypothesis("another", "ab", "ccg", family="f"),
            ],
        )

    with pytest.raises(ValueError, match="is declared twice"):
        Preregistration(
            _GRID,
            [
                DeclaredHypothesis("one", "ccg", "ab", family="f"),
                DeclaredHypothesis("one", "tlg", "ab", family="f"),
            ],
        )


def test_a_report_whose_families_are_all_declared_is_accepted() -> None:
    audit = check_preregistration(
        _plan(),
        [
            _comparison("ccg", "ab", family="grammar-comparison"),
            _comparison("ab", "tlg", family="grammar-comparison"),
        ],
    )

    assert audit.tested == ("ccg-over-ab", "tlg-over-ab")
    assert audit.untested == ("wide-over-narrow",)
    assert audit.undeclared == ()
    assert audit.ratio == pytest.approx(2 / 3)
    assert audit.preregistration == _plan()


def test_a_report_naming_an_undeclared_family_is_rejected() -> None:
    with pytest.raises(UndeclaredFamilyError, match="does not declare"):
        check_preregistration(
            _plan(), [_comparison("ccg", "ab", family="a-family-invented-later")]
        )


def test_a_declared_pair_moved_into_another_family_is_rejected() -> None:
    with pytest.raises(PreregistrationError, match="where the preregistration"):
        check_preregistration(
            _plan(), [_comparison("ccg", "ab", family="inventory-comparison")]
        )


def test_a_pair_the_plan_never_declared_is_recorded_rather_than_counted() -> None:
    audit = check_preregistration(
        _plan(),
        [
            _comparison("ccg", "tlg", family="grammar-comparison"),
            _comparison("tlg", "ccg", family="grammar-comparison"),
        ],
    )

    assert audit.tested == ()
    assert audit.undeclared == (("ccg", "tlg"),)
    assert audit.ratio == 0.0


def test_one_hypothesis_tested_twice_is_counted_once() -> None:
    audit = check_preregistration(
        _plan(),
        [
            _comparison("ccg", "ab", family="grammar-comparison"),
            _comparison("ab", "ccg", family="grammar-comparison"),
        ],
    )

    assert audit.tested == ("ccg-over-ab",)
    assert audit.ratio == pytest.approx(1 / 3)


def test_the_gate_refuses_a_plan_or_a_comparison_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="preregistration is not one"):
        check_preregistration(cast("Preregistration", _GRID), [])

    with pytest.raises(TypeError, match="checked comparison is not one"):
        check_preregistration(_plan(), [cast("Comparison", "ccg beats ab")])


def test_the_audit_states_the_grid_beside_what_was_tested_of_it() -> None:
    audit = check_preregistration(
        _plan(), [_comparison("ccg", "ab", family="grammar-comparison")]
    )
    spec = audit.to_spec()

    assert spec == {
        "grid_size": 6,
        "declared": 3,
        "tested": 1,
        "ratio": pytest.approx(1 / 3),
        "untested": ["tlg-over-ab", "wide-over-narrow"],
        "undeclared": [],
    }
    assert json.loads(json.dumps(audit.to_spec())) == audit.to_spec()
    assert audit == check_preregistration(
        _plan(), [_comparison("ccg", "ab", family="grammar-comparison")]
    )
    assert hash(audit) == hash(PreregistrationAudit(_plan(), ["ccg-over-ab"], []))
    assert audit != "an audit"
    assert repr(audit).startswith("PreregistrationAudit(Preregistration(")


def test_an_audit_refuses_a_hypothesis_the_plan_does_not_declare() -> None:
    with pytest.raises(ValueError, match="which the plan does not declare"):
        PreregistrationAudit(_plan(), ["not-declared"], [])

    with pytest.raises(ValueError, match="counts the hypothesis"):
        PreregistrationAudit(_plan(), ["ccg-over-ab", "ccg-over-ab"], [])

    with pytest.raises(TypeError, match="audit's preregistration is not one"):
        PreregistrationAudit(cast("Preregistration", _GRID), [], [])


def test_a_plan_is_read_from_the_file_it_was_registered_in() -> None:
    text = json.dumps(
        {
            "grid": _GRID,
            "hypotheses": [entry.to_spec() for entry in _hypotheses()],
        }
    )
    plan = read_preregistration(text)

    assert plan == _plan()
    assert plan.to_spec() == {
        "grid": {"grammar": ["ab", "ccg", "tlg"], "inventory": [4, 8]},
        "grid_size": 6,
        "hypotheses": [entry.to_spec() for entry in _hypotheses()],
    }
    assert json.loads(json.dumps(plan.to_spec())) == plan.to_spec()


def test_a_plan_refuses_a_file_that_is_not_the_shape_a_plan_is_written_in() -> None:
    with pytest.raises(PreregistrationError, match="not JSON"):
        read_preregistration("{")

    with pytest.raises(PreregistrationError, match="does not validate"):
        read_preregistration(cast("str", 3))

    with pytest.raises(PreregistrationError, match="does not validate"):
        read_preregistration("[]")

    with pytest.raises(PreregistrationError, match="does not validate"):
        read_preregistration(json.dumps({"grid": {}}))

    with pytest.raises(PreregistrationError, match="does not validate"):
        read_preregistration(json.dumps({"grid": {"grammar": "ab"}, "hypotheses": []}))

    with pytest.raises(PreregistrationError, match="does not validate"):
        read_preregistration(
            json.dumps({"grid": _GRID, "hypotheses": [{"name": "one"}]})
        )

    with pytest.raises(PreregistrationError, match="was refused"):
        read_preregistration(json.dumps({"grid": _GRID, "hypotheses": []}))
