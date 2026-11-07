"""Tests for the synthetic training-batch package surface."""

import lofi.data.synthetic.training._pairs as pairs
import lofi.data.synthetic.training._sentences as sentences
from lofi.data.synthetic.training import (
    CorpusSentences,
    FormTypingTarget,
    InferencePairs,
    InferenceTrainingBatch,
    SyntheticTrainingBatch,
    corpus_featurizer_resources,
    corpus_sentences,
    inference_training_batches,
    synthetic_inference_featurizer_resources,
    synthetic_training_batches,
)


def test_sentence_training_names_are_published() -> None:
    assert CorpusSentences is sentences.CorpusSentences
    assert FormTypingTarget is sentences.FormTypingTarget
    assert SyntheticTrainingBatch is sentences.SyntheticTrainingBatch
    assert corpus_featurizer_resources is sentences.corpus_featurizer_resources
    assert (
        synthetic_inference_featurizer_resources
        is sentences.synthetic_inference_featurizer_resources
    )
    assert corpus_sentences is sentences.corpus_sentences
    assert synthetic_training_batches is sentences.synthetic_training_batches


def test_pair_training_names_are_published() -> None:
    assert InferencePairs is pairs.InferencePairs
    assert InferenceTrainingBatch is pairs.InferenceTrainingBatch
    assert inference_training_batches is pairs.inference_training_batches
