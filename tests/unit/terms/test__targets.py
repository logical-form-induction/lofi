import pickle
from copy import copy, deepcopy

import pytest

from lofi.terms._matching import Identity
from lofi.terms._targets import TargetSet


def test_a_target_set_freezes_categories_and_retains_its_matching() -> None:
    matching = Identity[str]()
    declared = ["S", "NP"]
    targets = TargetSet(declared, matching)
    declared.append("N")

    assert targets.categories == ("S", "NP")
    assert targets.matching is matching
    assert targets.matching_name == "identity"
    assert repr(targets) == "TargetSet(categories=('S', 'NP'), matching='identity')"


def test_matching_identity_participates_in_target_set_equality() -> None:
    matching = Identity[str]()
    targets = TargetSet(("S", "NP"), matching)
    equal = TargetSet(("S", "NP"), matching)
    other_matching = TargetSet(("S", "NP"), Identity[str]())

    assert targets == equal
    assert hash(targets) == hash(equal)
    assert targets != other_matching
    assert targets != TargetSet(("S",), matching)


@pytest.mark.parametrize("field", ["_categories", "_matching"])
def test_a_target_set_refuses_mutation_and_deletion(field: str) -> None:
    targets = TargetSet(("S",), Identity[str]())

    with pytest.raises(AttributeError, match="immutable"):
        targets.__setattr__(field, ())

    with pytest.raises(AttributeError, match="immutable"):
        targets.__delattr__(field)


def test_a_target_set_has_explicit_copy_and_pickle_behavior() -> None:
    matching = Identity[str]()
    targets = TargetSet(("S", "NP"), matching)
    restored = pickle.loads(pickle.dumps(targets))

    assert copy(targets) is targets
    assert deepcopy(targets) is targets
    assert restored.categories == targets.categories
    assert restored.matching_name == targets.matching_name
    assert restored.matching is not matching
