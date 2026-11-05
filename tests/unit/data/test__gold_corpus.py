"""Tests for gold sentence corpora read from treebanks."""

from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest
from tests.unit.data.test_ccgbank import _AUTO, _dataset

from lofi._types import HeadSelection
from lofi.data import LoadReport
from lofi.data._gold_corpus import gold_corpus, lexical_counts

if TYPE_CHECKING:
    from pathlib import Path


def test_a_corpus_of_canonical_steps_rebuilds_to_the_same_corpus(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.auto"
    source.write_text(_AUTO, encoding="utf-8", newline="\n")
    dataset = _dataset(source)

    rebuilt = gold_corpus(
        dataset.corpus.examples,
        dataset.source_grammar,
        LoadReport(2, 2),
        seed=dataset.corpus.seed,
        selections=MappingProxyType({}),
    )

    assert rebuilt.corpus.examples == dataset.corpus.examples
    assert rebuilt.digest == dataset.source_digest
    assert rebuilt.report.analyses is not None
    assert rebuilt.report.analyses.checked == 2
    assert lexical_counts(rebuilt.corpus.examples)["John", "NP"] == 2


def test_a_treebank_without_examples_is_refused(tmp_path: Path) -> None:
    source = tmp_path / "sample.auto"
    source.write_text(_AUTO, encoding="utf-8", newline="\n")
    grammar = _dataset(source).source_grammar

    with pytest.raises(ValueError, match="emitted no examples"):
        gold_corpus(
            (),
            grammar,
            LoadReport(0, 0),
            seed=0,
            selections=MappingProxyType({}),
        )


def test_a_step_whose_head_selection_names_no_recovered_rule_is_refused(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.auto"
    source.write_text(_AUTO, encoding="utf-8", newline="\n")
    dataset = _dataset(source)
    grammar = dataset.source_grammar
    absent_slot = 2

    with pytest.raises(ValueError, match="identifies 0 compiled rules"):
        gold_corpus(
            dataset.corpus.examples,
            grammar,
            LoadReport(2, 2),
            seed=0,
            selections=MappingProxyType(
                {
                    rule.name: HeadSelection("functor", absent_slot)
                    for rule in grammar.grammar.rules
                }
            ),
        )
