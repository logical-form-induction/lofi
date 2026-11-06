from math import inf, log
from typing import TYPE_CHECKING

import pytest
from tests.unit.data.synthetic.conftest import semantic_sample_rows, semantic_source

from lofi.lexicon import Entry, StaticLexicon

if TYPE_CHECKING:
    from lofi.data.synthetic._ports import SemanticSource, SourceLexicon


def _lexicon() -> StaticLexicon[str]:
    return StaticLexicon(
        (
            Entry("dog", "N", None, log(3.0)),
            Entry("dog", "N/N", None, log(1.0)),
            Entry("runs", "S\\NP", None, log(2.0)),
        )
    )


def test_a_static_lexicon_enumerates_forms_entries_and_mass_through_the_protocol() -> (
    None
):
    source: SourceLexicon[str] = _lexicon()

    assert source.forms == ("dog", "runs")
    assert source.log_mass("dog") == pytest.approx(log(4.0))
    assert {entry.category: entry.weight for entry in source.entries("dog")} == (
        pytest.approx({"N": log(0.75), "N/N": log(0.25)})
    )
    assert [entry.weight for entry in source.entries("runs")] == [0.0]


def test_the_protocol_reports_a_form_the_lexicon_does_not_hold_as_absent() -> None:
    source: SourceLexicon[str] = _lexicon()

    assert source.entries("cat") == ()
    assert source.log_mass("cat") == -inf


def test_a_semantic_sample_enumerates_items_and_meanings_through_the_protocol() -> None:
    source: SemanticSource = semantic_source()
    john = source.meaning(("NP", "john"))
    dogs = source.meaning(("NP", "dogs"))

    assert len(source.items) == len(semantic_sample_rows())
    assert source.items[0] == ("NP/N", "the")
    assert john is not None
    assert (john.logical_class, john.symbol, john.arity) == ("constant", "john", None)
    assert dogs is not None
    assert (dogs.logical_class, dogs.symbol, dogs.arity) == ("predicate", "dog", 1)
    assert source.log_mass(("NP", "john")) == pytest.approx(log(210.0))


def test_the_semantic_protocol_reports_an_item_the_source_lacks_as_absent() -> None:
    source: SemanticSource = semantic_source()

    assert source.meaning(("NP", "cat")) is None
    assert source.log_mass(("NP", "cat")) == -inf
