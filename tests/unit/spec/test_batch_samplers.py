"""Tests for batch-sampler declarations."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

import didactic.api as dx
import pytest

from lofi.spec import BatchSamplerSpec, ConfiguredSpec, EpochPartitionSpec
from lofi.spec.batch_samplers import (
    LengthBucketBatchSamplerSpec,
    LengthBucketSettings,
    RandomBatchSamplerSpec,
    RandomSettings,
    SequentialBatchSamplerSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


@pytest.mark.parametrize(
    ("document", "expected", "settings"),
    [
        (
            {
                "kind": "length-bucket",
                "bucket_factor": 8,
                "reshuffle_each_epoch": True,
            },
            LengthBucketBatchSamplerSpec(bucket_factor=8, reshuffle_each_epoch=True),
            LengthBucketSettings(bucket_factor=8, reshuffle_each_epoch=True),
        ),
        (
            {"kind": "random", "reshuffle_each_epoch": False},
            RandomBatchSamplerSpec(reshuffle_each_epoch=False),
            RandomSettings(reshuffle_each_epoch=False),
        ),
        (
            {"kind": "sequential"},
            SequentialBatchSamplerSpec(),
            None,
        ),
    ],
)
def test_every_sampler_round_trips_through_the_family(
    document: dict[str, JsonValue],
    expected: BatchSamplerSpec,
    settings: object,
) -> None:
    declared = BatchSamplerSpec.model_validate(document)

    assert declared == expected
    assert isinstance(declared, EpochPartitionSpec)
    assert declared.reshuffles() is bool(document.get("reshuffle_each_epoch", False))
    assert cast("ConfiguredSpec[object]", declared).settings() == settings
    assert BatchSamplerSpec.model_validate_json(declared.model_dump_json()) == declared


@pytest.mark.parametrize(
    "document",
    [
        {
            "kind": "length-bucket",
            "bucket_factor": 0,
            "reshuffle_each_epoch": False,
        },
        {"kind": "sequential", "reshuffle_each_epoch": True},
        {"kind": "sequential", "reshuffle_each_epoch": False},
        {"kind": "random"},
        {"kind": "random", "reshuffle_each_epoch": False, "shuffle": True},
    ],
)
def test_invalid_or_superseded_sampler_documents_are_refused(
    document: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        BatchSamplerSpec.model_validate(document)


def test_sequential_order_needs_no_shuffle_setting() -> None:
    declared = BatchSamplerSpec.model_validate({"kind": "sequential"})

    assert declared.model_dump() == {"kind": "sequential", "declared_output": None}
    assert cast("ConfiguredSpec[object]", declared).settings() is None
