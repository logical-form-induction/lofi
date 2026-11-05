"""Production composition and recovery for configured distance pretraining."""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING

import pytest
import torch

import lofi.train as train_module
from lofi.run import CHECKPOINTS_NAME
from lofi.run._entrypoint import main
from lofi.train import DirectoryStore

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.train import Checkpoint, CheckpointId, CheckpointManifest, StateTree

CONFIG = (
    Path(__file__).resolve().parents[2] / "lofi/configs/run/synthetic-distance.yaml"
)


def _final_term_encoder(root: Path, /) -> StateTree:
    """Return term-encoder state from the final production checkpoint."""
    (directory,) = (root / "runs").glob("[0-9a-f]*")
    store = DirectoryStore(directory / CHECKPOINTS_NAME)
    handle = store.list()[-1]

    return torch.load(
        directory / CHECKPOINTS_NAME / handle / "roles/terms.pt",
        map_location="cpu",
        weights_only=True,
    )


@pytest.mark.integration
def test_interrupted_production_distance_fitting_resumes_bitwise(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    control = tmp_path / "control"
    interrupted = tmp_path / "interrupted"
    argv = ["run", "--config", str(CONFIG), "--seed", "31", "--experiment-seed", "5"]

    assert main([*argv, "--root", str(control)]) == 0
    original_store = DirectoryStore

    class InterruptingStore:
        def __init__(self, root: Path) -> None:
            self._inner = original_store(root)
            self._saved = False

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

            if not self._saved:
                self._saved = True
                raise KeyboardInterrupt

            return handle

    monkeypatch.setattr(train_module, "DirectoryStore", InterruptingStore)

    with pytest.raises(KeyboardInterrupt):
        main([*argv, "--root", str(interrupted)])

    monkeypatch.setattr(train_module, "DirectoryStore", original_store)
    assert main([*argv, "--root", str(interrupted)]) == 0
    expected = _final_term_encoder(control)
    found = _final_term_encoder(interrupted)

    torch.testing.assert_close(found, expected, rtol=0.0, atol=0.0)
