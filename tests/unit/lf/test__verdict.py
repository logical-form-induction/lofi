from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest

from lofi.lf._verdict import (
    ABSTENTION_CAUSES,
    Abstain,
    Countermodel,
    CounterSatisfiable,
    Graded,
    Proof,
    Proved,
    Refuted,
    SearchBound,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.lf._verdict import AbstentionCause
    from lofi.terms import Term


class _Structure:
    """A crisp finite structure for checking countermodel evidence."""

    def __init__(self, values: Mapping[str, bool]) -> None:
        self._values = MappingProxyType(dict(values))

    @property
    def identity(self) -> str:
        """Return this fixture structure's stable identity."""
        return "fixture-structure"

    @property
    def rendered(self) -> str:
        """Return this fixture structure's inspectable representation."""
        return ", ".join(
            f"{name}={str(value).lower()}" for name, value in self._values.items()
        )

    def holds(self, formula: Term, /) -> bool:
        """Return the declared truth value of a fixture formula."""
        return self._values[cast("str", formula)]


def _term(text: str) -> Term:
    """Treat fixture names as opaque terms for the structure protocol."""
    return cast("Term", text)


def _proof(text: str) -> Proof:
    return Proof("fixture", text)


def _counter(text: str) -> CounterSatisfiable:
    return CounterSatisfiable("fixture", "CounterSatisfiable", text)


def test_a_proof_holds_the_certificate_it_was_given() -> None:
    proof = _proof("resolution refutation")

    assert Proved(proof).proof is proof


def test_a_refutation_holds_the_countermodel_it_was_given() -> None:
    evidence = _counter("p=true, q=false")

    assert Refuted(evidence).evidence is evidence


def test_an_abstention_holds_the_reason_it_was_given() -> None:
    assert Abstain("the wall clock ran out").reason == "the wall clock ran out"


def test_a_proved_verdict_refuses_untyped_evidence() -> None:
    with pytest.raises(TypeError, match="requires a Proof"):
        Proved(cast("Proof", "resolution refutation"))


def test_a_refuted_verdict_refuses_untyped_evidence() -> None:
    with pytest.raises(TypeError, match="requires CounterSatisfiable or Countermodel"):
        Refuted(cast("CounterSatisfiable", "p=true, q=false"))


@pytest.mark.parametrize(
    "verdict",
    [Proved(_proof("something")), Refuted(_counter("something")), Abstain("something")],
)
def test_a_verdict_refuses_reassignment_and_deletion(
    verdict: Proved | Refuted | Abstain,
) -> None:
    with pytest.raises(AttributeError, match="immutable"):
        verdict.__setattr__("_reason", "other")

    with pytest.raises(AttributeError, match="immutable"):
        verdict.__delattr__("_reason")


def test_the_four_kinds_never_compare_equal_on_the_same_text() -> None:
    assert Proved(_proof("x")) == Proved(_proof("x"))
    assert Proved(_proof("x")) != Proved(_proof("y"))
    assert Proved(_proof("x")) != Refuted(_counter("x"))
    assert Refuted(_counter("x")) != Abstain("x")
    assert Abstain("x") != Proved(_proof("x"))
    assert Graded("fixture", 0.5, "x") != Proved(_proof("x"))
    assert Proved(_proof("x")) != "x"


def test_the_four_kinds_hash_apart_on_the_same_text() -> None:
    proved = Proved(_proof("x"))
    refuted = Refuted(_counter("x"))
    graded = Graded("fixture", 0.5, "x")
    held = {proved, refuted, graded, Abstain("x"), Proved(_proof("x"))}

    assert len(held) == 4
    assert hash(proved) == hash(Proved(_proof("x")))
    assert hash(proved) != hash(refuted)
    assert hash(refuted) != hash(Abstain("x"))
    assert hash(Abstain("x")) != hash(proved)
    assert hash(proved) != hash("x")
    assert hash(refuted) != hash("x")
    assert hash(Abstain("x")) != hash("x")


@pytest.mark.parametrize(
    ("verdict", "rendered"),
    [
        (Proved(_proof("c")), "Proved(Proof('fixture', 'c', ()))"),
        (
            Refuted(_counter("m")),
            "Refuted(CounterSatisfiable('fixture', 'CounterSatisfiable', 'm'))",
        ),
        (Graded("fixture", 0.5, "g"), "Graded('fixture', 0.5, 'g')"),
        (Abstain("r"), "Abstain('r', cause='gave-up')"),
    ],
)
def test_a_verdict_renders_as_the_call_that_rebuilds_it(
    verdict: Proved | Refuted | Graded | Abstain, rendered: str
) -> None:
    assert repr(verdict) == rendered


def test_an_unclassified_abstention_names_the_prover_giving_up() -> None:
    assert Abstain("the wall clock ran out").cause == "gave-up"


def test_abstentions_on_one_reason_are_told_apart_by_their_cause() -> None:
    held = {
        Abstain(
            "stopped",
            cause=cause,
            bound=SearchBound("atoms", 8) if cause == "search-bound" else None,
        )
        for cause in ABSTENTION_CAUSES
    }

    assert len(held) == len(ABSTENTION_CAUSES)
    assert {verdict.cause for verdict in held} == set(ABSTENTION_CAUSES)
    assert Abstain("stopped", cause="timeout") == Abstain("stopped", cause="timeout")
    assert Abstain("stopped", cause="timeout") != Abstain("stopped", cause="gave-up")
    assert hash(Abstain("stopped", cause="timeout")) != hash(
        Abstain("stopped", cause="gave-up")
    )


def test_an_abstention_accepts_an_open_named_cause() -> None:
    verdict = Abstain("stopped", cause=cast("AbstentionCause", "license-expired"))

    assert verdict.cause == "license-expired"


@pytest.mark.parametrize("degree", [0.0, 0.25, 1.0])
def test_a_graded_verdict_holds_unit_interval_evidence(degree: float) -> None:
    verdict = Graded("fuzzy-prover", degree, "reported degree")

    assert verdict.degree == degree
    assert verdict.system == "fuzzy-prover"


@pytest.mark.parametrize("degree", [-0.01, 1.01, float("inf")])
def test_a_graded_verdict_refuses_a_degree_outside_the_unit_interval(
    degree: float,
) -> None:
    with pytest.raises(ValueError, match="graded entailment degree"):
        Graded("fuzzy-prover", degree, "reported degree")


def test_a_proof_has_typed_system_derivation_and_unsat_core() -> None:
    proof = Proof("vampire", "resolution refutation", ("axiom-2", "premise"))

    assert proof.system == "vampire"
    assert proof.derivation == "resolution refutation"
    assert proof.unsat_core == ("axiom-2", "premise")


def test_a_counter_satisfiable_status_is_typed_refutation_evidence() -> None:
    evidence = CounterSatisfiable("vampire", "CounterSatisfiable", "model found")

    assert evidence.system == "vampire"
    assert evidence.status == "CounterSatisfiable"
    assert evidence.detail == "model found"


def test_a_countermodel_checks_every_assumption_and_the_negated_goal() -> None:
    model = Countermodel(
        _Structure(
            {
                "premise": True,
                "theory": True,
                "negated-goal": True,
            }
        ),
        (_term("premise"),),
        _term("negated-goal"),
        theory=(_term("theory"),),
    )

    assert model.structure.identity == "fixture-structure"
    assert model.premises == (_term("premise"),)
    assert model.negated_goal == _term("negated-goal")
    assert model.theory == (_term("theory"),)


@pytest.mark.parametrize("false_formula", ["premise", "theory", "negated-goal"])
def test_a_countermodel_refuses_any_required_formula_it_falsifies(
    false_formula: str,
) -> None:
    values = {
        "premise": True,
        "theory": True,
        "negated-goal": True,
    }
    values[false_formula] = False

    with pytest.raises(ValueError, match=false_formula):
        Countermodel(
            _Structure(values),
            (_term("premise"),),
            _term("negated-goal"),
            theory=(_term("theory"),),
        )


def test_a_search_bound_names_the_exhausted_resource_and_limit() -> None:
    bound = SearchBound("atoms", 8)

    assert bound.kind == "atoms"
    assert bound.limit == 8


def test_a_search_bound_abstention_carries_its_exhausted_bound() -> None:
    bound = SearchBound("atoms", 8)
    verdict = Abstain("atom cap reached", cause="search-bound", bound=bound)

    assert verdict.bound is bound
    assert repr(verdict) == (
        "Abstain('atom cap reached', cause='search-bound', "
        "bound=SearchBound('atoms', 8))"
    )


def test_a_search_bound_and_its_cause_are_required_together() -> None:
    with pytest.raises(ValueError, match="requires its exhausted SearchBound"):
        Abstain("atom cap reached", cause="search-bound")

    with pytest.raises(ValueError, match="only a search-bound"):
        Abstain("timed out", cause="timeout", bound=SearchBound("atoms", 8))
