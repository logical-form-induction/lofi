"""Benchmark one self-contained local bundle restoration."""

from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest

from lofi.data import LicenseRef, Licensing, Provenance
from lofi.run import ExperimentSeed, RandomStreams, plan_training, resolve_device_policy
from lofi.run.hub import ModelRelease, export_hub_model, load_hub_bundle
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.train import prepare_training

if TYPE_CHECKING:
    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

CONFIG = Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
CODEC: Final = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))
LOAD_DIAGNOSTIC_MS: Final = 6_000.0
DERIVATION = ExperimentSeed(41).point("hub-benchmark", 0)


def _median_ms(benchmark: BenchmarkFixture) -> float:
    """Return the measured median beside the local-load diagnostic ceiling."""
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the Hub loader ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["diagnostic_ceiling_ms"] = LOAD_DIAGNOSTIC_MS
    benchmark.extra_info["median_ms"] = median_ms

    return median_ms


def test_loading_a_local_model_bundle_meets_its_diagnostic_ceiling(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    wall_clock_budget: WallClockBudget,
) -> None:
    """Time one config decode, plan, validation, construction, and state restore.

    The plan compiles the grammar and lowers its data. The public loader
    performs each stage exactly once.
    """
    run_spec = CODEC(CONFIG)
    blueprint = plan_training(run_spec)
    streams = RandomStreams(
        DERIVATION.policy(accelerator=None, samplers=blueprint.sampler_streams)
    )
    streams.apply()
    prepared = prepare_training(
        blueprint, streams, resolve_device_policy(blueprint.run_spec)
    )
    release = ModelRelease(
        "benchmark-parser",
        ("lofi/synthetic-parser",),
        ("en",),
        Provenance(
            "generated",
            Licensing(
                "CC-BY-4.0",
                (LicenseRef("CC-BY-4.0", "synthetic-corpus"),),
            ),
            (),
            None,
        ),
        grammar_identity=blueprint.grammar_fingerprint,
        codec_identity=blueprint.codec_fingerprint,
        action_set_identity=blueprint.action_set_fingerprint,
        corpus_revision=blueprint.corpus_revision,
        pretrained_resources={},
    )
    bundle = tmp_path / "bundle"
    export_hub_model(bundle, release, blueprint, prepared, DERIVATION)

    def load() -> str:
        return load_hub_bundle(bundle).release.model_name

    loaded_name = benchmark.pedantic(load, rounds=5, warmup_rounds=1, iterations=1)

    assert loaded_name == "benchmark-parser"
    if wall_clock_budget.enforced:
        assert _median_ms(benchmark) <= LOAD_DIAGNOSTIC_MS
