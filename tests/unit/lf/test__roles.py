from lofi.lf._roles import (
    EQUALITY,
    EXCLUSIVE_OR,
    EXISTENTIAL,
    GENERALIZED_UNIVERSAL,
    UNIVERSAL,
)


def test_an_open_truth_function_supplies_all_of_its_semantics() -> None:
    assert EXCLUSIVE_OR.name == "exclusive-or"
    assert EXCLUSIVE_OR.arity == 2
    assert EXCLUSIVE_OR.truth((False, True))
    assert not EXCLUSIVE_OR.truth((True, True))
    assert EXCLUSIVE_OR.truth_table == (False, True, True, False)
    assert EXCLUSIVE_OR.tptp(("p", "q")) == "(p <~> q)"
    assert EXCLUSIVE_OR.thf(("p", "q")) == "(p <~> q)"
    assert EXCLUSIVE_OR.smt(("p", "q")) == "(xor p q)"
    assert EXCLUSIVE_OR.model_operation is None


def test_model_roles_publish_semantic_operations_independent_of_their_names() -> None:
    assert EQUALITY.model_operation == "equality"
    assert UNIVERSAL.model_operation == "universal"
    assert EXISTENTIAL.model_operation == "existential"
    assert GENERALIZED_UNIVERSAL.model_operation == "restricted-universal"
