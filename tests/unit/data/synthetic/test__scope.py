"""Tests for the reading enumeration and the equivalence quotient."""

from typing import TYPE_CHECKING, Final

import pytest
from tests.unit.data.synthetic.conftest import (
    APPLICATION,
    FLAT_ENTRIES,
    JOINED_ENTRIES,
    JOINED_INVENTORY,
    SCOPING,
    flat_generator,
    grammar,
    interpretation,
    joined_generator,
    scoped_generator,
    spurious_generator,
)

from lofi.data.synthetic import SyntheticGenerator, readings
from lofi.data.synthetic.spec.interpretation import lower_interpretation
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec.grammar import GrammarIso
from lofi.spec.language import LanguageIso, registered_language
from lofi.spec.provers import LabelingProverSpec, ProofBudgetSpec, TruthTableProverSpec

if TYPE_CHECKING:
    from tests.unit.data.synthetic.conftest import Generator

    from lofi.data.synthetic import Readings


SCOPED: Final = ("every-student", "reads", "a-book")
JOINED: Final = ("john-too", "reads", "mary-too")
CONJOINED: Final = ("john", "sleeps", "and", "mary", "sleeps")
IMPATIENT: Final = 1e-9


def _readings(generator: Generator, forms: tuple[str, ...]) -> Readings:
    """Read a string's forms through the module function rather than the method."""
    chart = generator.parse(forms)
    interpretation = generator.interpretation

    assert interpretation is not None

    return readings(
        interpretation,
        chart,
        generator.grammar.goal.roots(chart),
    )


def test_two_quantifiers_scoped_two_ways_are_two_readings() -> None:
    generator = scoped_generator()
    held = generator.readings(SCOPED)
    interpretation = generator.interpretation

    assert interpretation is not None
    assert held.derivations == generator.derivation_count(SCOPED)
    assert held.derivations > held.count
    assert held.count == 2
    assert {interpretation.render(form) for form in held.forms} == {
        "forall x0. student(x0) => exists x1. book(x1) & read(x0)(x1)",
        "exists x0. book(x0) & forall x1. student(x1) => read(x1)(x0)",
    }


def test_the_module_function_and_the_method_read_the_same_forms() -> None:
    generator = scoped_generator()
    direct = _readings(generator, SCOPED)
    held = generator.readings(SCOPED)

    assert direct.forms == held.forms
    assert direct.derivations == held.derivations


def test_reading_equivalence_reuses_the_shared_proof_memo() -> None:
    """Repeated reading grouping reaches the same service memo."""
    generator = joined_generator()
    interpretation = generator.interpretation
    assert interpretation is not None

    first = generator.readings(JOINED)
    after_first = interpretation.service.report
    second = generator.readings(JOINED)
    after_second = interpretation.service.report

    assert second.forms == first.forms
    assert after_first.calls > 0
    assert after_second.calls == after_first.calls
    assert after_second.memo_hits > after_first.memo_hits


def test_a_spent_proof_call_budget_leaves_uncached_readings_apart() -> None:
    """A zero-call budget turns both equivalence questions into abstentions."""
    languages = LanguageIso()
    assembly, interpreted = lower_interpretation(
        grammar("joined", JOINED_ENTRIES, SCOPING),
        registered_language("propositional").with_(
            logic=registered_language("propositional").logic.with_(
                prover=TruthTableProverSpec(timeout=5.0),
                labeling=LabelingProverSpec(budget=ProofBudgetSpec(calls_per_step=0)),
            )
        ),
        interpretation(JOINED_INVENTORY),
        GrammarIso(languages),
        languages,
    )
    generator = SyntheticGenerator(
        assembly,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        interpretation=interpreted,
    )

    held = generator.readings(JOINED)
    report = interpreted.service.report

    assert held.count == 2
    assert held.undecided == 1
    assert report.calls == 0
    assert report.abstentions["budget-spent"] == 2


def test_two_scopings_that_entail_each_other_are_one_reading() -> None:
    languages = LanguageIso()
    assembly, interpreted = lower_interpretation(
        grammar("joined", JOINED_ENTRIES, SCOPING),
        registered_language("propositional").with_(
            logic=registered_language("propositional").logic.with_(
                prover=TruthTableProverSpec(timeout=IMPATIENT)
            )
        ),
        interpretation(JOINED_INVENTORY),
        GrammarIso(languages),
        languages,
    )
    impatient = SyntheticGenerator(
        assembly,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        interpretation=interpreted,
    )
    apart = impatient.readings(JOINED)
    held = joined_generator().readings(JOINED)
    function = impatient.interpretation

    assert function is not None
    assert apart.derivations == held.derivations == 3
    assert apart.count == 2
    assert apart.undecided == 1
    assert held.count == 1
    assert held.undecided == 0

    left, right = apart.forms

    assert function.render(left) != function.render(right)
    assert function.decide(left, right).label == "undecided"
    assert function.decide(right, left).label == "undecided"


def test_a_string_with_many_derivations_can_have_one_reading() -> None:
    generator = spurious_generator()
    held = generator.readings(CONJOINED)
    interpretation = generator.interpretation

    assert interpretation is not None
    assert held.derivations > 20
    assert held.count == 1
    assert interpretation.render(held.forms[0]) == "sleep(john) & sleep(mary)"


def test_a_string_with_one_derivation_has_one_reading() -> None:
    generator = flat_generator()

    for forms in (("john", "sleeps"), ("john", "loves", "mary")):
        held = generator.readings(forms)

        assert held.derivations == 1
        assert held.count == 1
        assert held.undecided == 0


def test_an_undecided_equivalence_leaves_two_forms_apart() -> None:
    generator = scoped_generator()
    held = generator.readings(("a-student", "reads", "a-book"))

    assert held.undecided == 1
    assert held.count == 2


def test_an_undecided_equivalence_can_be_counted_as_one_reading() -> None:
    """The declared equivalent policy joins a pair the oracle cannot decide."""
    generator = scoped_generator()
    forms = ("a-student", "reads", "a-book")
    chart = generator.parse(forms)
    interpreted = generator.interpretation
    assert interpreted is not None

    held = readings(
        interpreted,
        chart,
        generator.grammar.goal.roots(chart),
        undecided="equivalent",
    )

    assert held.undecided == 1
    assert held.count == 1


def test_readings_need_an_interpretation() -> None:
    bare = SyntheticGenerator(
        GrammarIso().forward(
            grammar("flat", FLAT_ENTRIES, APPLICATION),
            registered_language("propositional"),
        ),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )

    with pytest.raises(ValueError, match="none is declared"):
        bare.readings(("john", "sleeps"))


def test_a_readings_record_names_its_counts() -> None:
    held = scoped_generator().readings(SCOPED)

    assert repr(held) == (
        f"Readings(count=2, derivations={held.derivations}, undecided=1)"
    )
