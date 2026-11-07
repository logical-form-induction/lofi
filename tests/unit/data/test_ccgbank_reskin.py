from collections import Counter
from pathlib import Path
from typing import Final, cast

from lofi.data import Derivation, DerivationNode
from lofi.data.ccgbank_reskin import PLUGIN
from lofi.data.corpora.ccgbank.reader import CcgbankCorpusReader
from lofi.data.synthetic import SyntheticDataset, SyntheticGenerator
from lofi.data.synthetic.spec.ccgbank import (
    CcgbankReskinDatasetSpec,
    FrequencyReskinSpec,
)
from lofi.data.synthetic.spec.corpus import (
    CorruptionOperatorsSpec,
    CorruptionProportionSpec,
    CorruptionSpec,
)
from lofi.data.synthetic.spec.operators import SwapSpec, WeightedCorruptionOperatorSpec
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.language import registered_language

FIXTURE: Final = Path(__file__).with_name("fixtures") / "ccgbank_reskin.auto"
POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _declaration(
    corruption: CorruptionSpec | None = None,
) -> CcgbankReskinDatasetSpec:
    return CcgbankReskinDatasetSpec(
        source=str(FIXTURE),
        splits=("train",),
        grammar=ccgbank_grammar(),
        language=registered_language("propositional"),
        mapping=FrequencyReskinSpec(prefix="type", minimum_frequency=2),
        seed=11,
        corruption=corruption,
    )


def _lower(
    corruption: CorruptionSpec | None = None,
) -> SyntheticDataset:
    checked = PLUGIN.check(_declaration(corruption))
    dataset = cast("SyntheticDataset", PLUGIN.lower(checked, POLICY))
    PLUGIN.validate(dataset)
    return dataset


def _core(dataset: SyntheticDataset, label: str) -> str:
    grammar = dataset.source_grammar
    term = grammar.syntax.parse(grammar.store, label)
    return grammar.syntax.render(grammar.store, term.core)


def _stripped(
    dataset: SyntheticDataset, derivation: Derivation[str]
) -> Derivation[str]:
    return Derivation(
        derivation.root,
        {
            node_id: DerivationNode(
                node.span,
                _core(dataset, node.label) if node.is_lexical else node.label,
                rule=node.rule,
                children=node.children,
                features=node.features,
            )
            for node_id, node in derivation.nodes.items()
        },
    )


def _leaf_counts(derivations: tuple[Derivation[str], ...]) -> Counter[str]:
    return Counter(
        leaf.label for derivation in derivations for leaf in derivation.leaves()
    )


def test_the_reskin_keeps_source_derivations_and_empirical_leaf_frequencies() -> None:
    source = tuple(CcgbankCorpusReader(FIXTURE, splits=("train",)).examples("train"))
    dataset = _lower()
    examples = dataset.corpus.examples

    assert tuple(example.item.item_id for example in examples) == tuple(
        example.item.item_id for example in source[:4]
    )
    assert tuple(example.supervision.derivation for example in examples) == tuple(
        _stripped(dataset, example.supervision.derivation) for example in source[:4]
    )
    assert _leaf_counts(
        tuple(example.supervision.derivation for example in examples)
    ) == Counter({"NP": 4, "S\\NP": 4})
    assert all(
        "[" not in leaf.label
        for example in examples
        for leaf in example.supervision.derivation.leaves()
    )

    category_by_form = {
        form: dataset.source_grammar.syntax.render(
            dataset.source_grammar.store,
            dataset.source_grammar.lexicon.entries(form)[0].category,
        )
        for form in dataset.source_grammar.lexicon.forms
    }
    assert category_by_form == {"type0": "NP", "type1": "S\\NP"}
    assert (
        tuple(
            tuple(token.surface for token in example.item.tokens)
            for example in examples
        )
        == (("type0", "type1"),) * 4
    )


def test_equal_reskin_declarations_produce_the_same_corpus() -> None:
    assert _lower() == _lower()


def test_declared_exact_corruption_replaces_the_requested_share_with_negatives() -> (
    None
):
    corruption = CorruptionSpec(
        proportion=CorruptionProportionSpec(value=0.5, draw="exact"),
        operators=CorruptionOperatorsSpec(
            entries=(WeightedCorruptionOperatorSpec(operator=SwapSpec()),)
        ),
        attempts=8,
    )
    dataset = _lower(corruption)
    report = dataset.corpus.corruption

    assert report is not None
    assert len(dataset.corpus.examples) == 2
    assert len(dataset.corpus.corrupted) == 2
    assert report.corrupted == 2
    assert report.achieved_proportion == 0.5
    assert Counter(label.true_label for label in dataset.corpus.labels) == Counter(
        {"grammatical": 2, "ungrammatical": 2}
    )

    certifier = SyntheticGenerator(dataset.source_grammar, policy=POLICY)
    for record in dataset.corpus.corrupted:
        forms = tuple(token.surface for token in record.item.tokens)
        assert certifier.certify(forms) == record.certificate
