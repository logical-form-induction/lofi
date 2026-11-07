"""Batch-sampler construction plugins."""

from typing import Final

from lofi.data.sampler import (
    LengthBucketBatchSampler,
    RandomBatchSampler,
    SequentialBatchSampler,
)
from lofi.registry import ComponentPlugin
from lofi.spec.batch_samplers import (
    LengthBucketBatchSamplerSpec,
    LengthBucketSettings,
    RandomBatchSamplerSpec,
    RandomSettings,
    SequentialBatchSamplerSpec,
)

__all__ = ["LENGTH_BUCKET_PLUGIN", "RANDOM_PLUGIN", "SEQUENTIAL_PLUGIN"]


def _length_bucket(
    settings: LengthBucketSettings, context: None, /
) -> LengthBucketBatchSampler:
    del context
    return LengthBucketBatchSampler(
        settings.bucket_factor,
        reshuffle_each_epoch=settings.reshuffle_each_epoch,
    )


def _random(settings: RandomSettings, context: None, /) -> RandomBatchSampler:
    del context
    return RandomBatchSampler(reshuffle_each_epoch=settings.reshuffle_each_epoch)


def _sequential(settings: None, context: None, /) -> SequentialBatchSampler:
    del settings, context
    return SequentialBatchSampler()


LENGTH_BUCKET_PLUGIN: Final = ComponentPlugin(
    spec=LengthBucketBatchSamplerSpec,
    build=_length_bucket,
)
RANDOM_PLUGIN: Final = ComponentPlugin(spec=RandomBatchSamplerSpec, build=_random)
SEQUENTIAL_PLUGIN: Final = ComponentPlugin(
    spec=SequentialBatchSamplerSpec,
    build=_sequential,
)
