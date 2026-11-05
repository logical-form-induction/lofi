from lofi.data.sampler import (
    LengthBucketBatchSampler,
    RandomBatchSampler,
    SequentialBatchSampler,
)
from lofi.spec.batch_samplers import (
    LengthBucketBatchSamplerSpec,
    RandomBatchSamplerSpec,
    SequentialBatchSamplerSpec,
)
from lofi.training_plugins.batch_sampler import (
    LENGTH_BUCKET_PLUGIN,
    RANDOM_PLUGIN,
    SEQUENTIAL_PLUGIN,
)


def test_batch_sampler_plugins_build_their_declared_runtime_strategies() -> None:
    length_bucket = LENGTH_BUCKET_PLUGIN.construct(
        LengthBucketBatchSamplerSpec(bucket_factor=4, reshuffle_each_epoch=True),
        None,
    )
    random = RANDOM_PLUGIN.construct(
        RandomBatchSamplerSpec(reshuffle_each_epoch=False), None
    )
    sequential = SEQUENTIAL_PLUGIN.construct(SequentialBatchSamplerSpec(), None)

    assert isinstance(length_bucket, LengthBucketBatchSampler)
    assert length_bucket.bucket_factor == 4
    assert length_bucket.reshuffle_each_epoch is True
    assert isinstance(random, RandomBatchSampler)
    assert random.reshuffle_each_epoch is False
    assert isinstance(sequential, SequentialBatchSampler)
    assert sequential.reshuffle_each_epoch is False
