from typing import TYPE_CHECKING, Final

import pytest
import torch

from lofi.train import (
    CheckpointManifest,
    ComponentRecord,
    DirectoryStore,
    PrecisionRecord,
    RngRecord,
    StateReference,
)

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from pytest_benchmark.fixture import BenchmarkFixture
    from tests.benchmarks.conftest import WallClockBudget

    from lofi.train import StateTree

# One save and one load with minimal representative state blocks.
BUDGET_MS: Final = 20.0

ROLES: Final = (
    "terms",
    "forward_application",
    "backward_application",
    "forward_composition",
    "backward_composition",
    "controller",
)


def _record(role: str) -> ComponentRecord:
    return ComponentRecord(
        tag="hrr" if "application" in role or "composition" in role else role,
        spec_fingerprint=f"spec-{role}",
        signature_fingerprint="sig-b2",
        codec_identity="codec-c3",
        state=StateReference(key=role, digest="d4"),
        labels={name: column for column, name in enumerate(ROLES[1:5])}
        if role == "controller"
        else None,
    )


def _manifest() -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run-benchmark",
        step=1000,
        data_position=0,
        code_version="0.1.0",
        action_set="actions-a1",
        components={role: _record(role) for role in ROLES},
        rng=RngRecord(streams={"python": "9f3a", "numpy": "77e1", "torch": "1c20"}),
        precision=PrecisionRecord(
            compute_dtype="bfloat16",
            autocast_device_type="cpu",
            gradient_scaling=False,
        ),
        optimizer=StateReference(key="optimizer", digest="a7"),
        scaler=StateReference(key="scaler", digest="b8"),
        scheduler=StateReference(key="scheduler", digest="c9"),
    )


def _blocks() -> Mapping[str, StateTree]:
    # one scalar per block, so the timing is the machinery rather than the weights
    held: dict[str, StateTree] = {role: {"weight": torch.tensor(0.5)} for role in ROLES}
    held["optimizer"] = {
        "state": {0: {"step": torch.tensor(1000.0)}},
        "param_groups": [],
    }
    held["scaler"] = {}
    held["scheduler"] = {"last_epoch": 1000, "base_lrs": [0.001]}

    return held


def _median_ms(benchmark: BenchmarkFixture, cell: str) -> float:
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the checkpoint operation ran but pytest-benchmark timed nothing")

    median_ms = measured.stats.median * 1e3
    benchmark.extra_info["budget"] = "checkpoint-roundtrip"
    benchmark.extra_info["budget_cell"] = cell
    benchmark.extra_info["budget_ms"] = BUDGET_MS
    benchmark.extra_info["median_ms"] = median_ms

    return median_ms


def test_saving_a_checkpoint_lists_its_handle(tmp_path: Path) -> None:
    store = DirectoryStore(tmp_path / "checkpoints")

    handle = store.save(_manifest(), _blocks())

    assert handle in store.list()


def test_saving_a_checkpoint_with_minimal_state_meets_its_budget(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    wall_clock_budget: WallClockBudget,
) -> None:
    store = DirectoryStore(tmp_path / "checkpoints")
    manifest = _manifest()
    blocks = _blocks()
    benchmark.pedantic(
        store.save, args=(manifest, blocks), rounds=50, warmup_rounds=2, iterations=1
    )

    if wall_clock_budget.enforced:
        assert _median_ms(benchmark, "checkpoint-save") <= BUDGET_MS


def test_loading_a_checkpoint_returns_its_manifest_and_roles(tmp_path: Path) -> None:
    store = DirectoryStore(tmp_path / "checkpoints")
    manifest = _manifest()
    handle = store.save(manifest, _blocks())

    loaded = store.load(handle, manifest)

    assert loaded.manifest == manifest
    assert frozenset(loaded.states) == manifest.roles


def test_loading_a_checkpoint_with_minimal_state_meets_its_budget(
    benchmark: BenchmarkFixture,
    tmp_path: Path,
    wall_clock_budget: WallClockBudget,
) -> None:
    store = DirectoryStore(tmp_path / "checkpoints")
    manifest = _manifest()
    handle = store.save(manifest, _blocks())
    benchmark.pedantic(
        store.load, args=(handle, manifest), rounds=50, warmup_rounds=2, iterations=1
    )

    if wall_clock_budget.enforced:
        assert _median_ms(benchmark, "checkpoint-load") <= BUDGET_MS
