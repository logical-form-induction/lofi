from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from scripts.preflight.public_release import CommandResult

from scripts.preflight import public_release

_REMOTE_URL = "https://github.com/example/lofi.git"
_REPOSITORY = "example/lofi"


class ReleaseRunner:
    def __init__(
        self,
        remote: Path,
        *,
        expected_document: tuple[str, str] | None = None,
    ) -> None:
        self.remote = remote
        self.expected_document = expected_document
        self.calls: list[tuple[tuple[str, ...], Path]] = []
        self.documentation_heads: list[str] = []
        self.documentation_contents: list[str] = []
        self.gitleaks_roots: list[Path] = []
        self.gitleaks_saw_remote_only = False
        self.gitleaks_saw_remote_tag = False
        self.gitleaks_configs: list[str] = []

    def _documentation_result(self, cwd: Path, /) -> CommandResult:
        head = public_release.run_command(("git", "rev-parse", "HEAD"), cwd=cwd)
        assert head.returncode == 0
        self.documentation_heads.append(head.stdout.strip())

        if self.expected_document is not None:
            relative, expected = self.expected_document
            content = (cwd / relative).read_text(encoding="utf-8")
            self.documentation_contents.append(content)

            if content != expected:
                return CommandResult(1, "", "unexpected documentation source")

        return CommandResult(0, "", "")

    def _gitleaks_result(
        self,
        command: tuple[str, ...],
        cwd: Path,
        /,
    ) -> CommandResult:
        if command == ("gitleaks", "version"):
            return CommandResult(0, "8.30.1\n", "")

        self.gitleaks_roots.append(cwd)
        config = Path(command[command.index("--config") + 1])
        ignore = Path(command[command.index("--gitleaks-ignore-path") + 1])
        self.gitleaks_configs.append(config.read_text(encoding="utf-8"))
        assert ignore.read_text(encoding="utf-8") == ""
        assert "--ignore-gitleaks-allow" in command
        remote_only = public_release.run_command(
            (
                "git",
                "show-ref",
                "--verify",
                "--quiet",
                "refs/heads/private-material",
            ),
            cwd=cwd,
        )
        self.gitleaks_saw_remote_only = remote_only.returncode == 0
        remote_tag = public_release.run_command(
            (
                "git",
                "show-ref",
                "--verify",
                "--quiet",
                "refs/tags/private-tag",
            ),
            cwd=cwd,
        )
        self.gitleaks_saw_remote_tag = remote_tag.returncode == 0
        assert command[2] == str(cwd)
        return CommandResult(0, "", "")

    def __call__(
        self,
        command: tuple[str, ...],
        /,
        *,
        cwd: Path,
    ) -> CommandResult:
        self.calls.append((command, cwd))

        if command[:4] == ("git", "clone", "--mirror", "--no-local"):
            mapped = (*command[:4], str(self.remote), command[5])
            return public_release.run_command(mapped, cwd=cwd)

        if command[0] == "git":
            return public_release.run_command(command, cwd=cwd)

        if command[0] == "gitleaks":
            return self._gitleaks_result(command, cwd)

        if command[0] == "uv":
            return self._documentation_result(cwd)

        sha = _main_sha(self.remote)

        if command[:2] == ("gh", "api"):
            assert command[2] == f"repos/{_REPOSITORY}/commits/main"
            return CommandResult(0, f"{sha}\n", "")

        if command[:3] == ("gh", "run", "list"):
            assert command[command.index("--repo") + 1] == _REPOSITORY
            return CommandResult(
                0,
                json.dumps(
                    [
                        {
                            "headSha": sha,
                            "status": "completed",
                            "conclusion": "success",
                            "url": "https://example.test/run/1",
                        }
                    ]
                ),
                "",
            )

        raise AssertionError(command)


def _run(root: Path, *command: str) -> None:
    result = public_release.run_command(tuple(command), cwd=root)
    assert result.returncode == 0, result.stderr


def _commit(root: Path, *paths: str) -> None:
    _run(root, "git", "add", "--", *paths)
    _run(
        root,
        "git",
        "-c",
        "user.name=LoFI Test",
        "-c",
        "user.email=lofi@example.test",
        "commit",
        "-m",
        "fixture",
    )


def _push(root: Path, remote: Path, refspec: str) -> None:
    _run(root, "git", "push", str(remote), refspec)


def _main_sha(remote: Path) -> str:
    result = public_release.run_command(
        ("git", "rev-parse", "refs/heads/main"),
        cwd=remote,
    )
    assert result.returncode == 0, result.stderr
    return result.stdout.strip()


def _repository(base: Path) -> tuple[Path, Path]:
    root = base / "checkout"
    remote = base / "remote.git"
    root.mkdir()
    remote.mkdir()
    _run(remote, "git", "init", "--bare")
    _run(root, "git", "init", "--initial-branch=main")
    (root / "LICENSE").write_text(
        (Path(public_release.__file__).parents[2] / "LICENSE").read_text(
            encoding="utf-8"
        ),
        encoding="utf-8",
        newline="\n",
    )
    _commit(root, "LICENSE")
    _run(root, "git", "remote", "add", "origin", _REMOTE_URL)
    _push(root, remote, "HEAD:refs/heads/main")
    return root, remote


def test_a_clean_repository_passes_all_six_preconditions(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, remote = _repository(tmp_path)
    runner = ReleaseRunner(remote)

    assert public_release.main(["--root", str(root)], runner=runner) == 0

    output = capsys.readouterr()
    assert output.err == ""
    assert output.out.splitlines() == [
        "PASS: license present",
        "PASS: whole-history secret scan clean",
        "PASS: funded-proposal text absent",
        "PASS: planning content absent",
        "PASS: strict documentation build passes",
        "PASS: CI green on main",
    ]
    commands = [command for command, _cwd in runner.calls]
    assert any(command[0] == "gitleaks" for command in commands)
    assert any(command[0] == "uv" for command in commands)
    assert any(command[:3] == ("gh", "run", "list") for command in commands)
    gh_commands = {command[:2] for command in commands if command[0] == "gh"}
    assert gh_commands == {("gh", "api"), ("gh", "run")}
    assert len(runner.gitleaks_configs) == 1
    assert "[extend]\nuseDefault = true\n" in runner.gitleaks_configs[0]
    assert "allowlist" not in runner.gitleaks_configs[0].casefold()


def test_deleted_proposal_and_planning_paths_are_both_named(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    root, remote = _repository(tmp_path)
    proposal = root / "release-notes.txt"
    proposal.write_text(
        "Project Summary\nPrivate funded work.\n", encoding="utf-8", newline="\n"
    )
    planning = root / "planning" / "private.md"
    planning.parent.mkdir()
    planning.write_text("private plan\n", encoding="utf-8", newline="\n")
    _commit(root, "release-notes.txt", "planning/private.md")
    proposal.unlink()
    planning.unlink()
    _commit(root, "release-notes.txt", "planning/private.md")
    _push(root, remote, "HEAD:refs/heads/main")

    assert (
        public_release.main(
            ["--root", str(root)],
            runner=ReleaseRunner(remote),
        )
        == 1
    )

    output = capsys.readouterr()
    assert "FAIL: funded-proposal text absent\n  release-notes.txt\n" in output.err
    assert "FAIL: planning content absent\n  planning/private.md\n" in output.err


def test_proposal_scan_handles_markup_case_and_binary_path_variants(
    tmp_path: Path,
) -> None:
    root, remote = _repository(tmp_path)
    markdown = root / "release-notes.md"
    markdown.write_text(
        "# pRoJeCt **SuMmArY**\nprivate narrative\n",
        encoding="utf-8",
        newline="\n",
    )
    latex = root / "narrative.tex"
    latex.write_text(
        "\\section{INTELLECTUAL MERIT}\nprivate narrative\n",
        encoding="utf-8",
        newline="\n",
    )
    binary = root / "archive" / "results.PDF"
    binary.parent.mkdir()
    binary.write_bytes(b"%PDF-1.7\x00private")
    lfs = root / "archive" / "appendix.pdf"
    lfs.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'0' * 64}\n"
        "size 100000\n",
        encoding="utf-8",
        newline="\n",
    )
    concatenated = root / "archive" / "grantapplication.txt"
    concatenated.write_text("private narrative\n", encoding="utf-8", newline="\n")
    office = root / "archive" / "results.docx"
    office.write_bytes(b"PK\x03\x04private")
    _commit(
        root,
        "release-notes.md",
        "narrative.tex",
        "archive/results.PDF",
        "archive/appendix.pdf",
        "archive/grantapplication.txt",
        "archive/results.docx",
    )
    _push(root, remote, "HEAD:refs/heads/main")

    results = public_release.preconditions(root, runner=ReleaseRunner(remote))
    proposal = next(
        result for result in results if result.name == "funded-proposal text absent"
    )

    assert not proposal.passed
    assert proposal.details == (
        "archive/appendix.pdf",
        "archive/grantapplication.txt",
        "archive/results.PDF",
        "archive/results.docx",
        "narrative.tex",
        "release-notes.md",
    )


def test_proposal_scan_ignores_embedded_literals_and_nsf_substrings(
    tmp_path: Path,
) -> None:
    root, remote = _repository(tmp_path)
    fixture = root / "scanner_fixture.py"
    fixture.write_text(
        'document.write_text("Project Summary\\nsynthetic fixture\\n")\n',
        encoding="utf-8",
        newline="\n",
    )
    transfer = root / "transfer.py"
    transfer.write_text("value = 'ordinary transfer'\n", encoding="utf-8", newline="\n")
    transformer = root / "transformer.py"
    transformer.write_text(
        "value = 'ordinary transformer'\n", encoding="utf-8", newline="\n"
    )
    _commit(root, "scanner_fixture.py", "transfer.py", "transformer.py")
    _push(root, remote, "HEAD:refs/heads/main")

    results = public_release.preconditions(root, runner=ReleaseRunner(remote))
    proposal = next(
        result for result in results if result.name == "funded-proposal text absent"
    )

    assert proposal.passed
    assert proposal.details == ()


def test_proposal_scan_reports_each_path_that_held_renamed_text(
    tmp_path: Path,
) -> None:
    root, remote = _repository(tmp_path)
    original = root / "release-notes.txt"
    renamed = root / "neutral.txt"
    original.write_text(
        "Project Summary\nPrivate funded work.\n", encoding="utf-8", newline="\n"
    )
    _commit(root, "release-notes.txt")
    original.rename(renamed)
    _run(root, "git", "add", "-A")
    _run(
        root,
        "git",
        "-c",
        "user.name=LoFI Test",
        "-c",
        "user.email=lofi@example.test",
        "commit",
        "-m",
        "rename fixture",
    )
    _push(root, remote, "HEAD:refs/heads/main")

    results = public_release.preconditions(root, runner=ReleaseRunner(remote))
    proposal = next(
        result for result in results if result.name == "funded-proposal text absent"
    )

    assert not proposal.passed
    assert proposal.details == ("neutral.txt", "release-notes.txt")


def test_remote_only_refs_are_mirrored_for_every_history_check(
    tmp_path: Path,
) -> None:
    root, remote = _repository(tmp_path)
    _run(root, "git", "switch", "-c", "private-material")
    planning = root / "planning" / "private.md"
    planning.parent.mkdir()
    planning.write_text("private plan\n", encoding="utf-8", newline="\n")
    _commit(root, "planning/private.md")
    _push(root, remote, "HEAD:refs/heads/private-material")
    proposal = root / "remote-report.pdf"
    proposal.write_text(
        "version https://git-lfs.github.com/spec/v1\n"
        f"oid sha256:{'1' * 64}\n"
        "size 100000\n",
        encoding="utf-8",
        newline="\n",
    )
    _commit(root, "remote-report.pdf")
    _run(root, "git", "tag", "private-tag")
    _push(root, remote, "refs/tags/private-tag:refs/tags/private-tag")
    _run(root, "git", "switch", "main")
    _run(root, "git", "branch", "-D", "private-material")
    _run(root, "git", "tag", "-d", "private-tag")
    runner = ReleaseRunner(remote)

    results = public_release.preconditions(root, runner=runner)
    proposal_result = next(
        result for result in results if result.name == "funded-proposal text absent"
    )
    planning_result = next(
        result for result in results if result.name == "planning content absent"
    )

    assert not proposal_result.passed
    assert proposal_result.details == ("remote-report.pdf",)
    assert not planning_result.passed
    assert planning_result.details == ("planning/private.md",)
    assert runner.gitleaks_saw_remote_only
    assert runner.gitleaks_saw_remote_tag
    assert runner.gitleaks_roots
    assert all(mirror != root for mirror in runner.gitleaks_roots)


def test_gitleaks_ignores_hostile_environment_and_inline_allowlist(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    if shutil.which("gitleaks") is None:
        pytest.skip("gitleaks is not installed")

    root, _remote = _repository(tmp_path)
    secret = root / "secret.env"
    value = "".join(  # noqa: FLY002 - keep the detector fixture out of history.
        ("q7W4e9R2t5Y8u1I6", "o3P0aS9dF4gH7jK2")
    )
    secret.write_text(
        f'API_KEY="{value}" # gitleaks:allow\n',
        encoding="utf-8",
        newline="\n",
    )
    _commit(root, "secret.env")
    hostile = tmp_path / "hostile.toml"
    hostile.write_text(
        "[extend]\nuseDefault = true\n"
        '[[allowlists]]\ndescription = "ignore all"\npaths = [".*"]\n',
        encoding="utf-8",
        newline="\n",
    )
    monkeypatch.setenv("GITLEAKS_CONFIG", str(hostile))
    monkeypatch.setenv("GITLEAKS_CONFIG_TOML", hostile.read_text(encoding="utf-8"))

    result = public_release.secret_scan_precondition(root)

    assert not result.passed
    assert result.name == "whole-history secret scan clean"


def test_license_rejects_empty_or_non_mit_text(
    tmp_path: Path,
) -> None:
    root, remote = _repository(tmp_path)
    (root / "LICENSE").write_text("", encoding="utf-8", newline="\n")
    _commit(root, "LICENSE")
    _push(root, remote, "HEAD:refs/heads/main")

    result = public_release.license_precondition(remote, _main_sha(remote))

    assert not result.passed
    assert result.details == ("target LICENSE is not the canonical LoFI MIT license",)


def test_license_rejects_a_symlink_blob(tmp_path: Path) -> None:
    root, remote = _repository(tmp_path)
    (root / "link-target").write_bytes(b"README.md")
    blob = public_release.run_command(
        ("git", "hash-object", "-w", "link-target"), cwd=root
    )
    assert blob.returncode == 0, blob.stderr
    _run(
        root,
        "git",
        "update-index",
        "--cacheinfo",
        f"120000,{blob.stdout.strip()},LICENSE",
    )
    _run(
        root,
        "git",
        "-c",
        "user.name=LoFI Test",
        "-c",
        "user.email=lofi@example.test",
        "commit",
        "-m",
        "fixture",
    )
    _push(root, remote, "HEAD:refs/heads/main")

    result = public_release.license_precondition(remote, _main_sha(remote))

    assert not result.passed
    assert result.details == (
        "target main does not contain a regular root-level LICENSE file",
    )


def test_documentation_uses_locked_detached_target_commit(
    tmp_path: Path,
) -> None:
    root, remote = _repository(tmp_path)
    document = root / "docs" / "sentinel.txt"
    document.parent.mkdir()
    document.write_text("committed\n", encoding="utf-8", newline="\n")
    (root / "uv.lock").write_text("committed lock\n", encoding="utf-8", newline="\n")
    _commit(root, "docs/sentinel.txt", "uv.lock")
    _push(root, remote, "HEAD:refs/heads/main")
    document.write_text("dirty fix\n", encoding="utf-8", newline="\n")
    (root / "uv.lock").write_text("dirty lock\n", encoding="utf-8", newline="\n")
    runner = ReleaseRunner(
        remote,
        expected_document=("docs/sentinel.txt", "committed\n"),
    )

    results = public_release.preconditions(root, runner=runner)
    docs = next(
        result
        for result in results
        if result.name == "strict documentation build passes"
    )

    assert docs.passed
    assert runner.documentation_contents == ["committed\n"]
    assert runner.documentation_heads == [_main_sha(remote)]
    assert document.read_text(encoding="utf-8") == "dirty fix\n"
    assert (root / "uv.lock").read_text(encoding="utf-8") == "dirty lock\n"
    uv_command = next(command for command, _cwd in runner.calls if command[0] == "uv")
    assert "--locked" in uv_command
    assert "--strict" in uv_command


def test_missing_root_fails_all_six_without_a_traceback(
    tmp_path: Path,
    capsys: pytest.CaptureFixture[str],
) -> None:
    missing = tmp_path / "missing"

    assert public_release.main(["--root", str(missing)]) == 1

    output = capsys.readouterr()
    assert output.out == ""
    assert output.err.count("FAIL: ") == 6
    assert "target origin lookup exited with status 126" in output.err
    assert "Traceback" not in output.err


def test_ci_must_belong_to_the_exact_mirrored_main_commit(tmp_path: Path) -> None:
    local_sha = "a" * 40
    target_sha = "b" * 40

    def mismatched(
        command: tuple[str, ...],
        /,
        *,
        cwd: Path,
    ) -> CommandResult:
        assert cwd == tmp_path

        if command[:3] == ("git", "rev-parse", "HEAD"):
            return CommandResult(0, f"{local_sha}\n", "")

        if command[:2] == ("gh", "api"):
            return CommandResult(0, f"{target_sha}\n", "")

        raise AssertionError(command)

    result = public_release.ci_precondition(
        tmp_path,
        target_sha,
        _REPOSITORY,
        runner=mismatched,
    )

    assert not result.passed
    assert result.name == "CI green on main"
    assert result.details == (
        (
            f"local HEAD {local_sha}, mirrored main {target_sha}, and GitHub main "
            f"{target_sha} do not all match"
        ),
    )


def test_a_non_green_latest_run_names_its_state(tmp_path: Path) -> None:
    sha = "c" * 40

    def failed(
        command: tuple[str, ...],
        /,
        *,
        cwd: Path,
    ) -> CommandResult:
        assert cwd == tmp_path

        if command[:3] == ("git", "rev-parse", "HEAD"):
            return CommandResult(0, f"{sha}\n", "")

        if command[:2] == ("gh", "api"):
            return CommandResult(0, f"{sha}\n", "")

        if command[:3] == ("gh", "run", "list"):
            assert command[command.index("--repo") + 1] == _REPOSITORY
            return CommandResult(
                0,
                json.dumps(
                    [
                        {
                            "headSha": sha,
                            "status": "completed",
                            "conclusion": "failure",
                            "url": "https://example.test/run/2",
                        }
                    ]
                ),
                "",
            )

        raise AssertionError(command)

    result = public_release.ci_precondition(
        tmp_path,
        sha,
        _REPOSITORY,
        runner=failed,
    )

    assert not result.passed
    assert result.details == (
        "status=completed; conclusion=failure; url=https://example.test/run/2",
    )
