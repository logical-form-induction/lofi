"""Installed CCGbank dataset lowering and estimation."""

from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.data.ccgbank import PLUGIN, CcgbankDataset, lexicon_estimate
from lofi.data.datasets import ReservedForms
from lofi.data.synthetic.spec.corpus import ResemblingLexiconSpec
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.nn._tensorize import TorchCodeTensorizer
from lofi.spec.components.data import RetainedDatasetSpec, TrainingAllocationSpec
from lofi.spec.datasets import CcgbankDatasetSpec, ccgbank_grammar
from lofi.spec.grammar import GrammarIso, RootGoalSpec
from lofi.spec.language import LanguageDecl

if TYPE_CHECKING:
    from pathlib import Path

_AUTO = """ID=wsj_0001.1 PARSER=GOLD NUMPARSE=1
(<T S[dcl] 0 2> (<L NP NNP NNP John NP>) (<L S[dcl]\\NP VBZ VBZ sleeps S[dcl]\\NP>) )
ID=wsj_0001.2 PARSER=GOLD NUMPARSE=1
(<T S[dcl] 0 2> (<L NP NNP NNP Mary NP>) \
(<T S[dcl]\\NP 0 2> (<L (S[dcl]\\NP)/NP VBZ VBZ sees \
(S[dcl]\\NP)/NP>) (<L NP NNP NNP John NP>) ) )
"""


def _language() -> LanguageDecl:
    return LanguageDecl.model_validate(
        {
            "name": "ccgbank-test",
            "atomic_types": [{"name": "e"}, {"name": "t"}],
            "constructors": [
                {
                    "name": "->",
                    "kind": ["Ty", "Ty", "Ty"],
                    "notation": {
                        "style": "infix",
                        "spelling": "->",
                        "precedence": 1,
                        "associativity": "right",
                    },
                }
            ],
            "constants": [],
            "binders": {"style": "curry"},
            "logic": {"equality": False},
        }
    )


def _declaration(path: Path) -> CcgbankDatasetSpec:
    return CcgbankDatasetSpec(
        source=str(path),
        splits=("train",),
        grammar=ccgbank_grammar(),
        language=_language(),
    )


def _dataset(path: Path) -> CcgbankDataset:
    checked = PLUGIN.check(_declaration(path))
    return cast(
        "CcgbankDataset",
        PLUGIN.lower(checked, ParsePolicy(Retain.FOREST, Reachability.ALL)),
    )


def test_the_declaration_names_held_out_rows(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path / "sample.auto")

    assert declaration.held_out_split() == "validation"
    assert declaration.with_(validation_split="development").held_out_split() == (
        "development"
    )

    with pytest.raises(dx.ValidationError, match="validation split is empty"):
        declaration.with_(validation_split="")


def test_the_declaration_exposes_selected_retained_partitions(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path / "sample.auto").with_(
        splits=("train", "validation", "development")
    )

    assert isinstance(declaration, RetainedDatasetSpec)
    assert not isinstance(declaration, TrainingAllocationSpec)
    assert declaration.retained_splits() == frozenset(("validation", "development"))


def test_the_plugin_lowers_a_corpus_with_empirical_reports(tmp_path: Path) -> None:
    source = tmp_path / "sample.auto"
    source.write_text(_AUTO, encoding="utf-8", newline="\n")

    dataset = _dataset(source)
    PLUGIN.validate(dataset)

    assert dataset.report.records_seen == 2
    assert dataset.report.emitted == 2
    assert dataset.report.reachability == 1.0
    assert dataset.report.normal_form_conformance == 1.0
    assert tuple(example.item.item_id for example in dataset.corpus.examples) == (
        "wsj_0001.1",
        "wsj_0001.2",
    )
    vocabulary = PLUGIN.vocabulary(
        dataset, ReservedForms(("<pad>", "<unk>"), unknown="<unk>")
    )
    assert vocabulary.forms == ("John", "sleeps", "Mary", "sees")
    assert dict(vocabulary.reserved) == {"<pad>": 0, "<unk>": 1}
    assert vocabulary.unknown == "<unk>"
    rules = dataset.source_grammar.grammar.rules
    recorded = tuple(
        node.rule
        for example in dataset.corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if not node.is_lexical
    )
    assert recorded == (
        "application:\\",
        "application:/",
        "application:\\",
    )
    assert set(recorded) <= {rule.name for rule in rules}
    assert all(
        node.rule is None
        for example in dataset.corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if node.is_lexical
    )


def test_the_corpus_estimate_draws_through_a_document_declaration(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.auto"
    source.write_text(_AUTO, encoding="utf-8", newline="\n")
    dataset = _dataset(source)

    measured = lexicon_estimate(dataset)
    declaration = ccgbank_grammar().with_(
        goal=RootGoalSpec(categories=("NP",)),
        lexicon=ResemblingLexiconSpec(forms=measured.forms, estimate=measured, seed=5),
    )
    assembly = GrammarIso().forward(declaration, _language())

    assert measured.forms == 4
    assert len(measured.digest) == 64
    assert assembly.lexicon.forms == ("w0", "w1", "w2", "w3")


def test_the_estimate_digest_changes_with_the_source(tmp_path: Path) -> None:
    first = tmp_path / "first.auto"
    second = tmp_path / "second.auto"
    first.write_text(_AUTO, encoding="utf-8", newline="\n")
    second.write_text(_AUTO.replace("Mary", "Alice"), encoding="utf-8", newline="\n")

    first_estimate = lexicon_estimate(_dataset(first))
    second_estimate = lexicon_estimate(_dataset(second))

    assert first_estimate.with_(digest=second_estimate.digest) != first_estimate
    assert first_estimate.digest != second_estimate.digest


def test_the_corpus_trains_a_parser_on_its_declared_training_split(
    tmp_path: Path,
) -> None:
    source = tmp_path / "sample.auto"
    source.write_text(_AUTO, encoding="utf-8", newline="\n")
    dataset = _dataset(source)
    sentences = PLUGIN.sentences(dataset)
    resources = PLUGIN.resources(
        dataset,
        ReservedForms(("<pad>", "<unk>"), unknown="<unk>"),
        TorchCodeTensorizer(),
    )

    assert sentences.corpus is dataset.corpus
    assert sentences.source_grammar is dataset.source_grammar
    assert sentences.source_interpretation is None
    assert {
        token.surface
        for example in dataset.corpus.examples
        for token in example.item.tokens
    } <= set(resources.hasher.lexicon.ordinary)

    with pytest.raises(ValueError, match="not inference pairs"):
        PLUGIN.pairs(dataset)


def test_the_training_split_is_one_the_declaration_reads(tmp_path: Path) -> None:
    declaration = _declaration(tmp_path / "sample.auto").with_(
        splits=("train", "validation")
    )

    assert declaration.with_(training_split="validation").retained_splits() == (
        frozenset(("train",))
    )

    with pytest.raises(dx.ValidationError, match="does not read from"):
        declaration.with_(training_split="test")
