"""Featurizers for training supervision."""

from lofi.data.featurizers._core import (
    DeferredSupervisionError,
    FeaturizedBatch,
    FeaturizerPlugin,
    MixedSupervisionError,
    SupervisionDisposition,
    disposition_for,
    featurizer_plugin,
    require_featurizable,
)

__all__ = [
    "DeferredSupervisionError",
    "FeaturizedBatch",
    "FeaturizerPlugin",
    "MixedSupervisionError",
    "SupervisionDisposition",
    "disposition_for",
    "featurizer_plugin",
    "require_featurizable",
]
