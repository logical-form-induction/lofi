from copy import copy, deepcopy
from typing import Final

import pytest
from tests.conformance.evaluator import packaged_held_out_pass

from lofi.data import LabelMembership
from lofi.data.corpora import GrammarInventory, SignatureLabels
from lofi.terms import Operator, Signature

WIDE: Final = Signature(
    "cat",
    ("cat",),
    (Operator("S", "cat"), Operator("NP", "cat"), Operator("N", "cat")),
)

NARROW: Final = Signature("cat", ("cat",), (Operator("S", "cat"),))

FEATURED: Final = Signature(
    "cat",
    ("cat",),
    (Operator("S", "cat"),),
    features={"S": ("dcl", "q")},
)


def test_the_labels_admit_exactly_the_operators_the_signature_declares() -> None:
    labels = SignatureLabels(WIDE)

    assert isinstance(labels, LabelMembership)
    assert labels.signature is WIDE
    assert all(labels.contains(name) for name in ("S", "NP", "N"))
    assert not labels.contains("PP")
    assert not labels.contains("")


def test_the_labels_admit_the_featured_operators_the_signature_derives() -> None:
    labels = SignatureLabels(FEATURED)

    assert {operator.name for operator in FEATURED.operators} > {"S"}
    assert all(labels.contains(operator.name) for operator in FEATURED.operators)


def test_a_narrower_signature_admits_fewer_labels() -> None:
    wide = SignatureLabels(WIDE)
    narrow = SignatureLabels(NARROW)
    names = {operator.name for operator in WIDE.operators}

    assert {name for name in names if narrow.contains(name)} < {
        name for name in names if wide.contains(name)
    }


def test_the_labels_are_equal_on_the_signature_they_share_by_reference() -> None:
    redeclared = Signature(
        "cat",
        ("cat",),
        (Operator("S", "cat"), Operator("NP", "cat"), Operator("N", "cat")),
    )

    assert SignatureLabels(WIDE) == SignatureLabels(WIDE)
    assert hash(SignatureLabels(WIDE)) == hash(SignatureLabels(WIDE))
    assert SignatureLabels(WIDE) != SignatureLabels(redeclared)
    assert SignatureLabels(WIDE) != SignatureLabels(NARROW)
    assert SignatureLabels(WIDE) != WIDE
    assert repr(SignatureLabels(NARROW)) == f"SignatureLabels({NARROW!r})"


@pytest.mark.parametrize("field", ["_signature", "_names"])
def test_the_labels_refuse_reassignment_and_deletion(field: str) -> None:
    labels = SignatureLabels(WIDE)

    with pytest.raises(AttributeError, match="immutable"):
        setattr(labels, field, ())

    with pytest.raises(AttributeError, match="immutable"):
        delattr(labels, field)

    assert copy(labels) is labels
    assert deepcopy(labels) is labels
    assert labels.__reduce__() == (SignatureLabels, (WIDE,))


def test_a_grammar_inventory_admits_parsed_labels_and_refuses_unparsable_ones() -> None:
    inventory = GrammarInventory(packaged_held_out_pass().grammar)

    assert inventory.contains("NP")
    assert not inventory.contains("((")
    assert not inventory.contains("undeclared")
    assert inventory.recover(("((",), "NP") == ()
