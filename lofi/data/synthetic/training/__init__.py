"""Featurized training batches drawn from synthetic corpora."""

from lofi.data.synthetic.training._pairs import (
    CorpusPair,
    InferencePairs,
    InferenceTrainingBatch,
    PairCorpus,
    inference_training_batches,
)
from lofi.data.synthetic.training._sentences import (
    CorpusSentences,
    FormTypingTarget,
    SentenceCorpus,
    SyntheticTrainingBatch,
    corpus_featurizer_resources,
    corpus_sentences,
    synthetic_inference_featurizer_resources,
    synthetic_training_batches,
)

__all__ = [
    "CorpusPair",
    "CorpusSentences",
    "FormTypingTarget",
    "InferencePairs",
    "InferenceTrainingBatch",
    "PairCorpus",
    "SentenceCorpus",
    "SyntheticTrainingBatch",
    "corpus_featurizer_resources",
    "corpus_sentences",
    "inference_training_batches",
    "synthetic_inference_featurizer_resources",
    "synthetic_training_batches",
]
