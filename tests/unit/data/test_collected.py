"""Tests for collected-corpus dataset plugins."""

import json
import subprocess
import sys
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.unit.data.test__collected_reading import (
    FIXTURES,
    POLICY,
    collected_language,
    nli_release,
)

from lofi.data import Example, Relational
from lofi.data._epochs import EpochBatching
from lofi.data.collected import (
    PMB_CCG_PLUGIN,
    SNLI_PLUGIN,
    CollectedCorpus,
    CorpusDataset,
    PairCorpusDataset,
    collected_featurizer_resources,
)
from lofi.data.datasets import ReservedForms
from lofi.data.featurizers.pair import PairFeaturizer
from lofi.data.sampler import SequentialBatchSampler
from lofi.lf import THREE_WAY, InferenceLabel
from lofi.nn._tensorize import TorchCodeTensorizer
from lofi.spec.datasets import (
    CorpusSourceSpec,
    PmbCcgDatasetSpec,
    SnliDatasetSpec,
    ccgbank_grammar,
    pmb_grammar,
)
from lofi.spec.gold import MajorityGoldSpec
from lofi.spec.language import registered_language
from lofi.spec.tokenizers import RegexTokenizerSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lofi.data._ports import Featurizer
    from lofi.data.collected import PmbCcgDataset
    from lofi.data.featurizers._core import FeaturizedBatch

PROBE: Final = """\
import json
import sys

import lofi.data.collected

print(json.dumps("lairs" in sys.modules))
"""
"""Report whether importing the collected plugins loads the corpus codecs."""


def test_importing_the_plugins_loads_no_release_reader() -> None:
    completed = subprocess.run(
        [sys.executable, "-c", PROBE],
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    assert json.loads(completed.stdout) is False


def test_snli_builds_with_non_whitespace_tokens_and_feeds_pair_featurization() -> None:
    """Feed collected relational examples through the published pair resources."""
    declaration = SnliDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "nli" / "fixtures" / "snli.jsonl"),
                release=nli_release(),
            ),
        ),
        grammar=ccgbank_grammar(),
        language=collected_language(),
        tokenizer=RegexTokenizerSpec(pattern=r"\w+|[^\w\s]"),
        aggregation=MajorityGoldSpec(),
    )
    dataset = cast(
        "CorpusDataset", SNLI_PLUGIN.lower(SNLI_PLUGIN.check(declaration), POLICY)
    )
    examples = dataset.corpus.observed("test")

    with pytest.raises(ValueError, match="declared training split"):
        SNLI_PLUGIN.pairs(dataset)

    resources = collected_featurizer_resources(
        dataset,
        ReservedForms(("<pad>", "<unk>"), unknown="<unk>"),
        TorchCodeTensorizer(),
    )
    batch = PairFeaturizer(
        resources.hasher,
        "<pad>",
        resources.tensorizer,
        resources.lookup,
        Relational,
    )(examples)

    assert tuple(token.surface for token in examples[0].item.tokens)[-1] == "."
    assert batch.member_rows == ((0, 1), (2, 3))
    responses = tuple(
        response for response in batch.supervision if isinstance(response, Relational)
    )
    assert len(responses) == len(batch.supervision)
    assert tuple(response.relation for response in responses) == (
        "entails",
        "contradicts",
    )


def _snli_line(pair: str, relation: str, premise: str, hypothesis: str) -> str:
    return json.dumps(
        {
            "annotator_labels": [relation],
            "gold_label": relation,
            "pairID": pair,
            "sentence1": premise,
            "sentence2": hypothesis,
        }
    )


def _snli_sources(tmp_path: Path, /) -> tuple[CorpusSourceSpec, ...]:
    train = tmp_path / "train.jsonl"
    validation = tmp_path / "validation.jsonl"
    train.write_text(
        "\n".join(
            (
                _snli_line("p-0", "entailment", "A dog runs.", "A dog moves."),
                _snli_line("p-1", "contradiction", "A dog runs.", "No dog runs."),
            )
        )
        + "\n",
        encoding="utf-8",
        newline="\n",
    )
    validation.write_text(
        _snli_line("p-2", "neutral", "A cat sleeps.", "A cat dreams.") + "\n",
        encoding="utf-8",
        newline="\n",
    )
    return tuple(
        CorpusSourceSpec(source=str(path), release=nli_release().with_(split=split))
        for path, split in ((train, "train"), (validation, "validation"))
    )


def test_an_snli_corpus_trains_its_declared_split_through_the_labeling(
    tmp_path: Path,
) -> None:
    declaration = SnliDatasetSpec(
        sources=_snli_sources(tmp_path),
        training_split="train",
        labeling="rte",
        grammar=ccgbank_grammar(),
        language=collected_language(),
    )
    dataset = cast(
        "PairCorpusDataset", SNLI_PLUGIN.lower(SNLI_PLUGIN.check(declaration), POLICY)
    )
    training = SNLI_PLUGIN.pairs(dataset)
    resources = collected_featurizer_resources(
        dataset,
        ReservedForms(("<pad>", "<unk>"), unknown="<unk>"),
        TorchCodeTensorizer(),
    )
    featurizer = cast(
        "Featurizer[Relational, FeaturizedBatch[Relational]]",
        PairFeaturizer(
            resources.hasher,
            "<pad>",
            resources.tensorizer,
            resources.lookup,
            Relational,
        ),
    )
    epoch = training.epochs(
        featurizer,
        EpochBatching(
            SequentialBatchSampler(), batch_size=4, training_seed=0, validation_seed=1
        ),
    ).at(0)

    assert training.corpus is dataset
    assert declaration.retained_splits() == frozenset({"validation"})
    assert tuple(pair.relation for pair in dataset.corpus_pairs) == (
        InferenceLabel.ENTAILS,
        InferenceLabel.CONTRADICTS,
        InferenceLabel.NEUTRAL,
    )
    assert tuple(label for batch in epoch.training for label in batch.labels) == (
        InferenceLabel.ENTAILS,
        InferenceLabel.NON_ENTAILS,
    )
    assert tuple(
        label for batch in epoch.retained["validation"] for label in batch.labels
    ) == (InferenceLabel.NON_ENTAILS,)


def test_a_collected_pair_corpus_refuses_a_relation_outside_the_logical_relations(
    tmp_path: Path,
) -> None:
    source = tmp_path / "odd.jsonl"
    source.write_text(
        _snli_line("p-0", "entailment", "A dog runs.", "A dog moves.") + "\n",
        encoding="utf-8",
        newline="\n",
    )
    declaration = SnliDatasetSpec(
        sources=(CorpusSourceSpec(source=str(source), release=nli_release()),),
        grammar=ccgbank_grammar(),
        language=collected_language(),
    )
    dataset = cast(
        "PairCorpusDataset", SNLI_PLUGIN.lower(SNLI_PLUGIN.check(declaration), POLICY)
    )
    corpus = dataset.corpus

    with pytest.raises(ValueError, match="not one of the logical relations"):
        PairCorpusDataset(
            dataset.declaration,
            dataset.source_grammar,
            CollectedCorpus(
                {
                    "test": tuple(
                        Example(
                            example.item,
                            Relational(example.supervision.other, "paraphrase"),
                        )
                        if isinstance(example.supervision, Relational)
                        else example
                        for example in corpus.examples("test")
                    )
                },
                corpus.report,
            ),
            THREE_WAY,
            training_split=None,
        )


def test_a_pmb_ccg_corpus_trains_a_parser_on_its_declared_split() -> None:
    entry = CorpusSourceSpec(
        source=str(FIXTURES / "pmb" / "fixtures" / "pawnshop.tags"),
        release=nli_release().with_(source_id="pmb", split="train"),
    )
    declaration = PmbCcgDatasetSpec(
        sources=(entry,),
        training_split="train",
        grammar=pmb_grammar("en"),
        language=registered_language("propositional"),
    )
    dataset = cast(
        "PmbCcgDataset",
        PMB_CCG_PLUGIN.lower(PMB_CCG_PLUGIN.check(declaration), POLICY),
    )
    sentences = PMB_CCG_PLUGIN.sentences(dataset)

    assert sentences.corpus is dataset.corpus
    assert sentences.source_grammar is dataset.source_grammar
    assert "training_split='train'" in repr(sentences)


def test_a_collected_corpus_without_a_training_split_refuses_to_train() -> None:
    declaration = PmbCcgDatasetSpec(
        sources=(
            CorpusSourceSpec(
                source=str(FIXTURES / "pmb" / "fixtures" / "pawnshop.tags"),
                release=nli_release(),
            ),
        ),
        grammar=pmb_grammar("en"),
        language=registered_language("propositional"),
    )
    dataset = PMB_CCG_PLUGIN.lower(PMB_CCG_PLUGIN.check(declaration), POLICY)

    with pytest.raises(ValueError, match="declared training split"):
        PMB_CCG_PLUGIN.sentences(dataset)
