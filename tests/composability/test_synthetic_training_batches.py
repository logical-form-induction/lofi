from typing import Final

from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data.datasets import ReservedForms
from lofi.data.featurizers import unlabeled
from lofi.data.sampler import SequentialBatchSampler
from lofi.data.synthetic.spec.corpus import (
    SyntheticCorpusDecl,
    SyntheticCorpusIso,
    UniformLengthSpec,
)
from lofi.data.synthetic.training import (
    SyntheticTrainingBatch,
    corpus_featurizer_resources,
    synthetic_training_batches,
)
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.nn._tensorize import TorchCodeTensorizer
from lofi.spec import ApplicationSpec, RuleSetSpec
from lofi.spec.featurizers.unlabeled import UnlabeledFeaturizerSpec
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

CATEGORIES: Final = ("NP", "S\\NP", "(S\\NP)/NP", "NP/NP")
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _declaration() -> SyntheticCorpusDecl:
    grammar = GrammarDecl(
        name="synthetic-training",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(
                    form=f"word-{index}",
                    category=CATEGORIES[index % len(CATEGORIES)],
                )
                for index in range(8)
            )
        ),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )
    return SyntheticCorpusDecl(
        retained={"validation": 0.25},
        grammar=grammar,
        language=registered_language("propositional"),
        size=7,
        seed=5,
        length=UniformLengthSpec(min_length=2, max_length=3),
    )


def test_a_lowered_declaration_reaches_real_featurizer_batches() -> None:
    dataset = SyntheticCorpusIso(policy=FOREST_POLICY).forward(_declaration())
    resources = corpus_featurizer_resources(
        dataset,
        ReservedForms(("<pad>", "<unk>"), unknown="<unk>"),
        TorchCodeTensorizer(),
    )
    featurizer = unlabeled.PLUGIN.construct(
        UnlabeledFeaturizerSpec(padding="<pad>"), resources
    )

    batches = synthetic_training_batches(
        dataset.corpus,
        featurizer,
        dataset.source_interpretation,
        occurrence_positions=tuple(range(len(dataset.corpus.observed()))),
        constraint_batcher=None,
        batch_sampler=SequentialBatchSampler(),
        batch_size=3,
        sampler_seed=17,
        step=9,
    )
    lowered = tuple(
        (item_id, words, label)
        for batch in batches
        for item_id, words, label in zip(
            batch.item_ids,
            batch.words,
            batch.labels,
            strict=True,
        )
    )
    observed_labels = {
        label.item_id: label.observed_label == "grammatical"
        for label in dataset.corpus.labels
    }

    assert all(isinstance(batch, SyntheticTrainingBatch) for batch in batches)
    assert sorted(map(len, batches)) == [1, 3, 3]
    assert all(batch.step == 9 for batch in batches)
    assert {item_id for item_id, _, _ in lowered} == set(observed_labels)

    for item_id, words, label in lowered:
        assert words == tuple(
            token.surface for token in resources.lookup.items[item_id].tokens
        )
        assert label is observed_labels[item_id]

    for batch in batches:
        assert batch.featurized.batch.item_ids == batch.item_ids
        assert batch.featurized.batch.lengths == batch.lengths
        assert batch.featurized.batch.vectors.shape[:2] == (
            len(batch),
            max(batch.lengths),
        )
