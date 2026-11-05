from pathlib import Path
from typing import Final

import pytest

from lofi.run import _entrypoint

CONFIG: Final = (
    Path(__file__).resolve().parents[3] / "lofi/configs/run/synthetic-parser.yaml"
)


@pytest.fixture(scope="session")
def trained_run(tmp_path_factory: pytest.TempPathFactory) -> Path:
    """Return the directory of one completed packaged synthetic-parser run."""
    root = tmp_path_factory.mktemp("trained")
    status = _entrypoint.main(
        ["run", "--config", str(CONFIG), "--experiment-seed", "5", "--root", str(root)]
    )

    assert status == 0

    (directory,) = (
        path for path in (root / "runs").iterdir() if not path.name.startswith(".")
    )

    return directory
