import shutil
import subprocess
from typing import TYPE_CHECKING, cast

import pytest

from lofi.run import (
    ExecutionEnvironment,
    SourceRevisionError,
    UnfrozenRevisionError,
    package_checkout,
    require_frozen_revision,
    source_revision,
)
from lofi.run import _environment as environment_module

if TYPE_CHECKING:
    from collections.abc import MutableMapping
    from pathlib import Path


GIT = shutil.which("git") or "git"


def test_an_execution_environment_copies_and_freezes_its_facts() -> None:
    packages = {"torch": "2.14.0"}
    environment = ExecutionEnvironment(packages, "abc123", "cpu")
    packages["torch"] = "changed"

    assert dict(environment.packages) == {"torch": "2.14.0"}
    assert environment.git_revision == "abc123"
    assert environment.accelerator == "cpu"

    with pytest.raises(TypeError):
        cast("MutableMapping[str, str]", environment.packages)["torch"] = "changed"

    with pytest.raises(AttributeError):
        environment.__setattr__("git_revision", "changed")


def test_the_source_revision_is_the_repository_head() -> None:
    expected = subprocess.run(
        (GIT, "rev-parse", "HEAD"),
        check=True,
        capture_output=True,
        encoding="utf-8",
    ).stdout.strip()

    assert source_revision() == expected


def test_a_failed_git_inspection_is_a_named_boundary_error(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(environment_module, "package_checkout", lambda: tmp_path)
    monkeypatch.setattr(
        environment_module.subprocess,
        "run",
        lambda argv, **_: subprocess.CompletedProcess(argv, 1, "", "bad git"),
    )

    with pytest.raises(SourceRevisionError, match="bad git"):
        source_revision()


def test_execution_environment_reads_each_runtime_fact_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(
        environment_module, "installed_packages", lambda: {"lofi": "0.1.0"}
    )
    monkeypatch.setattr(environment_module, "source_revision", lambda: "abc123")
    monkeypatch.setattr(environment_module, "accelerator_runtime", lambda: "cpu")

    environment = environment_module.execution_environment()

    assert dict(environment.packages) == {"lofi": "0.1.0"}
    assert environment.git_revision == "abc123"
    assert environment.accelerator == "cpu"


def git(root: Path, *arguments: str) -> str:
    completed = subprocess.run(
        (GIT, "-C", str(root), *arguments),
        check=True,
        capture_output=True,
        encoding="utf-8",
    )

    return completed.stdout.strip()


@pytest.fixture
def frozen(tmp_path: Path) -> Path:
    git(tmp_path, "init", "--quiet", "--initial-branch", "main")
    git(tmp_path, "config", "user.name", "Freeze Test")
    git(tmp_path, "config", "user.email", "freeze@example.org")
    git(tmp_path, "config", "commit.gpgSign", "false")
    git(tmp_path, "config", "tag.gpgSign", "false")
    (tmp_path / "experiment.yaml").write_text(
        "role: headline\n", encoding="utf-8", newline="\n"
    )
    git(tmp_path, "add", "--all")
    git(tmp_path, "commit", "--quiet", "--message", "Gate commit")
    git(tmp_path, "tag", "--annotate", "--message", "freeze/1", "freeze/1")

    return tmp_path


def test_a_checkout_of_the_tagged_tree_returns_the_tagged_commit(
    frozen: Path,
) -> None:
    (frozen / "runs").mkdir()
    (frozen / "runs" / "manifest.json").write_text("{}", encoding="utf-8", newline="\n")

    assert require_frozen_revision(frozen, "freeze/1") == git(
        frozen, "rev-parse", "HEAD"
    )


def test_the_source_revision_of_a_named_checkout_is_its_head(frozen: Path) -> None:
    (frozen / "experiment.yaml").write_text(
        "kind: analysis\n", encoding="utf-8", newline="\n"
    )

    assert source_revision(frozen) == git(frozen, "rev-parse", "HEAD")


def test_a_checkout_without_the_tag_is_refused(frozen: Path) -> None:
    with pytest.raises(UnfrozenRevisionError, match="has no tag freeze/2"):
        require_frozen_revision(frozen, "freeze/2")


def test_a_checkout_past_the_tag_is_refused(frozen: Path) -> None:
    (frozen / "later.txt").write_text("later\n", encoding="utf-8", newline="\n")
    git(frozen, "add", "--all")
    git(frozen, "commit", "--quiet", "--message", "Later")

    with pytest.raises(UnfrozenRevisionError, match="that freeze/1 tags"):
        require_frozen_revision(frozen, "freeze/1")


def test_a_checkout_modifying_a_tracked_file_is_refused(frozen: Path) -> None:
    (frozen / "experiment.yaml").write_text(
        "kind: analysis\n", encoding="utf-8", newline="\n"
    )

    with pytest.raises(UnfrozenRevisionError, match=r"experiment\.yaml"):
        require_frozen_revision(frozen, "freeze/1")


def test_a_directory_outside_any_repository_is_refused(tmp_path: Path) -> None:
    with pytest.raises(UnfrozenRevisionError, match="has no tag"):
        require_frozen_revision(tmp_path, "freeze/1")


def test_the_package_checkout_is_the_checkout_holding_the_run_package() -> None:
    assert (package_checkout() / "lofi" / "run" / "_environment.py").is_file()
