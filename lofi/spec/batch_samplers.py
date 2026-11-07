"""Batch-sampler declarations."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi._checks import ensure_positive
from lofi.spec.components.trainer import BatchSamplerSpec

__all__ = [
    "LengthBucketBatchSamplerSpec",
    "LengthBucketSettings",
    "RandomBatchSamplerSpec",
    "RandomSettings",
    "SequentialBatchSamplerSpec",
]


class LengthBucketSettings(NamedTuple):
    """Resolved length-bucket sampler settings."""

    bucket_factor: int
    reshuffle_each_epoch: bool


class RandomSettings(NamedTuple):
    """Resolved random sampler settings."""

    reshuffle_each_epoch: bool


class LengthBucketBatchSamplerSpec(BatchSamplerSpec):
    """Declare shuffled length buckets and their padding-locality factor.

    Parameters
    ----------
    bucket_factor : int
        Positive number of batches sorted together by item length.
    reshuffle_each_epoch : bool
        Whether each epoch derives a distinct deterministic partition seed.
    """

    kind: Literal["length-bucket"] = "length-bucket"
    bucket_factor: int
    reshuffle_each_epoch: bool

    @dx.validates("bucket_factor")
    @classmethod
    def _factor_is_positive(cls, value: int) -> int:
        """Validate the number of batches in one length bucket."""
        return ensure_positive("bucket_factor", value)

    def reshuffles(self) -> bool:
        """Return whether each epoch reshuffles the length buckets."""
        return self.reshuffle_each_epoch

    def settings(self) -> LengthBucketSettings:
        """Return the resolved length-bucket settings."""
        return LengthBucketSettings(self.bucket_factor, self.reshuffle_each_epoch)


class RandomBatchSamplerSpec(BatchSamplerSpec):
    """Declare a random partition of the epoch's item indices."""

    kind: Literal["random"] = "random"
    reshuffle_each_epoch: bool

    def reshuffles(self) -> bool:
        """Return whether each epoch draws a new random partition."""
        return self.reshuffle_each_epoch

    def settings(self) -> RandomSettings:
        """Return the resolved random-partition settings."""
        return RandomSettings(self.reshuffle_each_epoch)


class SequentialBatchSamplerSpec(BatchSamplerSpec):
    """Declare batches in stable item-index order."""

    kind: Literal["sequential"] = "sequential"

    def reshuffles(self) -> bool:
        """Return false because sequential order is fixed."""
        return False

    def settings(self) -> None:
        """Return no settings for the fixed sequential order."""
