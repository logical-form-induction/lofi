import subprocess
import sys
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from pathlib import Path


def run_module(*arguments: str, cwd: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-m", "lofi.run", *arguments],
        check=False,
        capture_output=True,
        encoding="utf-8",
        cwd=cwd,
    )


def test_the_module_entry_runs_a_command_and_exits_with_its_status(
    tmp_path: Path,
) -> None:
    completed = run_module("configs", cwd=tmp_path)

    assert completed.returncode == 0
    assert "model.parser" in completed.stdout


def test_the_module_entry_exits_with_a_usage_status_without_a_command(
    tmp_path: Path,
) -> None:
    completed = run_module(cwd=tmp_path)

    assert completed.returncode == 2
    assert "usage: lofi" in completed.stderr
