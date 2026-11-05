from lofi.data import featurizers
from lofi.data.featurizers._core import (
    DeferredSupervisionError,
    FeaturizedBatch,
    MixedSupervisionError,
    SupervisionDisposition,
)


def test_the_package_reexports_shared_featurizer_values() -> None:
    assert featurizers.DeferredSupervisionError is DeferredSupervisionError
    assert featurizers.FeaturizedBatch is FeaturizedBatch
    assert featurizers.MixedSupervisionError is MixedSupervisionError
    assert featurizers.SupervisionDisposition is SupervisionDisposition
