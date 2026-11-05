"""Composition tests for installed batch-sampler choices."""

import json
from typing import TYPE_CHECKING, cast

from didactic.settings import load_document

from lofi.registry import Registry, plugin_for
from lofi.run import packaged_options, section_root
from lofi.spec import BatchSamplerSpec, ConfiguredSpec
from lofi.spec._plugins import load_plugins

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.data import BatchSampler


def _built(document: dict[str, JsonValue]) -> BatchSampler:
    """Validate and construct one declared batch sampler."""
    spec = BatchSamplerSpec.model_validate_json(json.dumps(document))
    declaration = cast("ConfiguredSpec[object]", spec)

    return cast("BatchSampler", plugin_for(declaration).construct(declaration, None))


def test_packaged_batch_samplers_validate_discover_construct_and_partition() -> None:
    loaded = load_plugins()
    options = packaged_options("trainer.batch_sampler")
    lengths = (1, 9, 2, 8, 3, 7, 4, 6, 5)

    assert options == ("length-bucket", "random", "sequential")

    for name in options:
        document = cast(
            "dict[str, JsonValue]",
            load_document(section_root("trainer.batch_sampler") / f"{name}.yaml"),
        )
        partition = _built(document).partition(lengths, 3, seed=73)

        assert name in loaded
        assert sorted(index for batch in partition for index in batch) == list(
            range(len(lengths))
        )

    registry = Registry[object]("batch_sampler")

    assert registry.available() == options


def test_declared_bucket_factor_changes_padding_occupancy() -> None:
    document = cast(
        "dict[str, JsonValue]",
        load_document(section_root("trainer.batch_sampler") / "length-bucket.yaml"),
    )
    local = _built({**document, "bucket_factor": 1})
    global_ = _built({**document, "bucket_factor": 16})
    lengths = (1, 16, 2, 15, 3, 14, 4, 13, 5, 12, 6, 11, 7, 10, 8, 9)

    local_partition = local.partition(lengths, 4, seed=31)
    global_partition = global_.partition(lengths, 4, seed=31)

    assert global_partition.occupancy.rate > local_partition.occupancy.rate
