import os
from typing import TYPE_CHECKING

from lofi.lf.provers._sandbox import sandbox_environment

if TYPE_CHECKING:
    import pytest


def test_a_sandboxed_prover_keeps_its_home_and_temporary_files_in_the_sandbox(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.delenv("SYSTEMROOT", raising=False)

    assert sandbox_environment("sandbox") == {
        "HOME": "sandbox",
        "LANG": "C",
        "LC_ALL": "C",
        "PATH": os.defpath,
        "TEMP": "sandbox",
        "TMP": "sandbox",
        "TMPDIR": "sandbox",
    }


def test_a_sandboxed_prover_inherits_the_windows_system_root(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("SYSTEMROOT", "C:\\Windows")

    assert sandbox_environment("sandbox")["SYSTEMROOT"] == "C:\\Windows"


def test_a_sandboxed_prover_inherits_nothing_else_from_its_parent(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setenv("LOFI_SANDBOX_LEAK", "leaked")

    assert "LOFI_SANDBOX_LEAK" not in sandbox_environment("sandbox")
