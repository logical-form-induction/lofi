import json
import subprocess
import sys
from pathlib import Path
from typing import TYPE_CHECKING, Never

import pytest

import lofi.train as train_module
from lofi.run import CHECKPOINTS_NAME, MANIFEST_NAME, ManifestStatus
from lofi.run._entrypoint import main
from lofi.train import DirectoryStore, read_manifest

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.train import (
        Checkpoint,
        CheckpointId,
        CheckpointManifest,
        CheckpointStore,
        FittingReport,
        MetricLog,
        PreparedTraining,
        RestorableRandomStreams,
        ResumePosition,
        StateTree,
    )

CONFIG = Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-parser.yaml"
DISTANCE_CONFIG = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-distance.yaml"
)


@pytest.mark.integration
def test_the_module_entrypoint_composes_production_training(tmp_path: Path) -> None:
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "lofi.run",
            "run",
            "--config",
            str(CONFIG),
            "--root",
            str(tmp_path),
            "--seed",
            "17",
            "--experiment-seed",
            "5",
        ],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
    (directory,) = (tmp_path / "runs").glob("[0-9a-f]*")
    assert json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))[
        "status"
    ] == (ManifestStatus.COMPLETED.value)
    assert DirectoryStore(directory / CHECKPOINTS_NAME).list()


@pytest.mark.integration
def test_the_default_cli_executor_trains_and_then_serves_the_completed_run(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    argv = [
        "run",
        "--config",
        str(CONFIG),
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path),
    ]

    assert main(argv) == 0
    (directory,) = (tmp_path / "runs").glob("[0-9a-f]*")
    manifest = json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))
    checkpoints = DirectoryStore(directory / CHECKPOINTS_NAME)
    handles = checkpoints.list()
    fittings = [
        read_manifest(
            (directory / CHECKPOINTS_NAME / handle / "manifest.json").read_text(
                encoding="utf-8"
            )
        ).fitting
        for handle in handles
    ]
    assert all(fitting is not None for fitting in fittings)
    kinds = [fitting.kind for fitting in fittings if fitting is not None]
    before = tuple(handles)

    assert manifest["status"] == ManifestStatus.COMPLETED.value
    assert kinds.count("validation-best") == 2
    assert kinds.count("resume-latest") == 2
    assert (directory / "config.sources.json").is_file()
    assert manifest["accelerator"]
    assert not (directory / "env.json").exists()

    def forbidden(*args: object, **kwargs: object) -> Never:
        del args, kwargs
        msg = "a cache hit prepared live training"
        raise AssertionError(msg)

    monkeypatch.setattr(train_module, "prepare_training", forbidden)

    assert main(argv) == 0
    assert tuple(checkpoints.list()) == before


@pytest.mark.integration
def test_a_killed_run_resumes_only_from_the_latest_eligible_checkpoint(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    original_store = DirectoryStore

    class InterruptingStore:
        def __init__(self, root: Path) -> None:
            self._inner = original_store(root)
            self._saved = 0

        def list(self) -> tuple[CheckpointId, ...]:
            return self._inner.list()

        def manifest(self, checkpoint: CheckpointId) -> CheckpointManifest:
            return self._inner.manifest(checkpoint)

        def remove(self, checkpoint: CheckpointId) -> None:
            self._inner.remove(checkpoint)

        def load(
            self, checkpoint: CheckpointId, expected: CheckpointManifest
        ) -> Checkpoint[StateTree]:
            return self._inner.load(checkpoint, expected)

        def save(
            self,
            manifest: CheckpointManifest,
            states: Mapping[str, StateTree],
        ) -> CheckpointId:
            handle = self._inner.save(manifest, states)
            self._saved += 1

            if self._saved == 2:
                raise KeyboardInterrupt

            return handle

    argv = [
        "run",
        "--config",
        str(CONFIG),
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path),
    ]
    monkeypatch.setattr(train_module, "DirectoryStore", InterruptingStore)

    with pytest.raises(KeyboardInterrupt):
        main(argv)

    monkeypatch.setattr(train_module, "DirectoryStore", original_store)
    assert main(argv) == 0
    (directory,) = (tmp_path / "runs").glob("[0-9a-f]*")
    checkpoints = original_store(directory / CHECKPOINTS_NAME)
    manifests = [
        read_manifest(
            (directory / CHECKPOINTS_NAME / handle / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for handle in checkpoints.list()
    ]

    assert all(manifest.fitting is not None for manifest in manifests)
    assert [
        (manifest.step, manifest.fitting.kind)
        for manifest in manifests
        if manifest.fitting is not None
    ] == [
        (1, "validation-best"),
        (1, "resume-latest"),
        (2, "validation-best"),
        (2, "resume-latest"),
    ]


@pytest.mark.integration
def test_the_production_cli_runs_term_encoder_only_distance_pretraining(
    tmp_path: Path,
) -> None:
    argv = [
        "run",
        "--config",
        str(DISTANCE_CONFIG),
        "--root",
        str(tmp_path),
        "--seed",
        "29",
        "--experiment-seed",
        "5",
    ]

    assert main(argv) == 0
    (directory,) = (tmp_path / "runs").glob("[0-9a-f]*")
    store = DirectoryStore(directory / CHECKPOINTS_NAME)
    manifests = tuple(
        read_manifest(
            (directory / CHECKPOINTS_NAME / handle / "manifest.json").read_text(
                encoding="utf-8"
            )
        )
        for handle in store.list()
    )

    assert tuple(manifest.step for manifest in manifests) == (2, 4)
    assert tuple(manifest.data_position for manifest in manifests) == (4, 8)
    assert all(set(manifest.components) == {"terms"} for manifest in manifests)
    assert all(
        manifest.fitting is not None
        and manifest.fitting.unit == "steps"
        and manifest.fitting.kind == "resume-latest"
        for manifest in manifests
    )


def _checkpoints(root: Path) -> tuple[Path, tuple[CheckpointId, ...]]:
    """Return a store root's one run directory and its checkpoint handles."""
    (directory,) = (root / "runs").glob("[0-9a-f]*")

    return directory, DirectoryStore(directory / CHECKPOINTS_NAME).list()


@pytest.mark.integration
def test_a_run_killed_after_its_final_checkpoint_completes_on_resume_untrained(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = train_module.fit_training

    def killed_after_fitting(
        prepared: PreparedTraining,
        store: CheckpointStore[StateTree],
        manifest: CheckpointManifest,
        streams: RestorableRandomStreams,
        position: ResumePosition,
        /,
        *,
        sinks: Sequence[MetricLog],
    ) -> Never:
        fit(prepared, store, manifest, streams, position, sinks=sinks)

        raise KeyboardInterrupt

    def untrained(*arguments: object) -> Never:
        del arguments
        msg = "a run whose final checkpoint was saved was trained again"
        raise AssertionError(msg)

    argv = [
        "run",
        "--config",
        str(CONFIG),
        "--experiment-seed",
        "5",
        "--root",
        str(tmp_path),
    ]
    monkeypatch.setattr(train_module, "fit_training", killed_after_fitting)

    with pytest.raises(KeyboardInterrupt):
        main(argv)

    directory, saved = _checkpoints(tmp_path)

    assert json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))[
        "status"
    ] == (ManifestStatus.KILLED.value)

    monkeypatch.setattr(train_module, "fit_training", untrained)

    assert main(argv) == 0
    assert _checkpoints(tmp_path)[1] == saved
    assert json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))[
        "status"
    ] == (ManifestStatus.COMPLETED.value)


@pytest.mark.integration
def test_an_interrupted_attempt_is_resumed_from_its_checkpoint_in_process(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    fit = train_module.fit_training
    calls: list[int] = []

    def interrupted_once(
        prepared: PreparedTraining,
        store: CheckpointStore[StateTree],
        manifest: CheckpointManifest,
        streams: RestorableRandomStreams,
        position: ResumePosition,
        /,
        *,
        sinks: Sequence[MetricLog],
    ) -> FittingReport:
        calls.append(1)
        report = fit(prepared, store, manifest, streams, position, sinks=sinks)

        if len(calls) == 1:
            raise InterruptedError

        return report

    monkeypatch.setattr(train_module, "fit_training", interrupted_once)

    assert (
        main(
            [
                "run",
                "--config",
                str(CONFIG),
                "--experiment-seed",
                "5",
                "--infrastructure-attempts",
                "2",
                "--root",
                str(tmp_path),
            ]
        )
        == 0
    )

    directory, saved = _checkpoints(tmp_path)

    assert calls == [1]
    assert len(saved) == 4
    assert json.loads((directory / MANIFEST_NAME).read_text(encoding="utf-8"))[
        "status"
    ] == (ManifestStatus.COMPLETED.value)
