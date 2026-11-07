"""Check every condition that must hold before a repository becomes public.

The command is deliberately read-only with respect to the source repository.
It mirrors every ref advertised by ``origin``, checks the mirrored ``main``
commit in a disposable checkout, and compares that commit with the GitHub CI
result for the same repository. It never changes the repository's visibility.

Run it from the repository that is to become public::

    uv run python scripts/preflight/public_release.py
"""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
from argparse import ArgumentParser
from pathlib import Path, PurePosixPath
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, cast
from urllib.parse import urlsplit

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue

_HEADING_SEPARATOR: Final = "[[:space:][:punct:]]+"
_PROPOSAL_HEADINGS: Final = (
    f"(Project{_HEADING_SEPARATOR}Summary|"
    f"Project{_HEADING_SEPARATOR}Description|"
    f"Intellectual{_HEADING_SEPARATOR}Merit|"
    f"Broader{_HEADING_SEPARATOR}Impacts|"
    f"Results{_HEADING_SEPARATOR}from{_HEADING_SEPARATOR}Prior"
    f"{_HEADING_SEPARATOR}NSF{_HEADING_SEPARATOR}Support)"
)
_PROPOSAL_CONTENT_PATTERN: Final = (
    f"(^[[:space:]#>*_`~-]*{_PROPOSAL_HEADINGS})|"
    f"(^[[:space:]]*\\\\(sub)*section[[:space:]]*\\{{"
    f"[[:space:]#>*_`~-]*{_PROPOSAL_HEADINGS})"
)
_PROPOSAL_PATH_TOKENS: Final = frozenset(
    {"proposal", "proposals", "grant", "grants", "nsf"}
)
_PROPOSAL_PATH_PHRASES: Final = (
    "proposal",
    "grant",
    "projectdescription",
    "projectsummary",
    "intellectualmerit",
    "broaderimpacts",
)
_OPAQUE_DOCUMENT_SUFFIXES: Final = frozenset(
    {
        ".doc",
        ".docx",
        ".key",
        ".odt",
        ".pages",
        ".pdf",
        ".ppt",
        ".pptx",
        ".rtf",
        ".xls",
        ".xlsx",
    }
)
_PATH_TOKEN: Final = re.compile("[^a-z]+")
_GITHUB_SCP_REMOTE: Final = re.compile(
    r"^(?:[^@/]+@)?github\.com:(?P<path>[^?#]+)$",
    re.IGNORECASE,
)
_GITHUB_COMPONENT: Final = re.compile(r"[A-Za-z0-9_.-]+")
_OBJECT_ID: Final = re.compile(r"[0-9a-f]{40}(?:[0-9a-f]{24})?")
_PRECONDITION_NAMES: Final = (
    "license present",
    "whole-history secret scan clean",
    "funded-proposal text absent",
    "planning content absent",
    "strict documentation build passes",
    "CI green on main",
)
_GITLEAKS_VERSION: Final = "8.30.1"
_GITLEAKS_CONFIG: Final = """\
title = "LoFI public-release secret scan"

[extend]
useDefault = true
"""
_EXPECTED_MIT_LICENSE: Final = """\
MIT License

Copyright (c) 2026 Aaron Steven White

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
"""


class CommandResult(NamedTuple):
    """The captured result of one read-only command."""

    returncode: int
    stdout: str
    stderr: str


class CommandRunner(Protocol):
    """Run one argument vector from a repository root."""

    def __call__(
        self,
        command: tuple[str, ...],
        /,
        *,
        cwd: Path,
    ) -> CommandResult:
        """Return the command's status and captured streams."""
        ...


class Precondition(NamedTuple):
    """The result of one named public-release precondition."""

    name: str
    passed: bool
    details: tuple[str, ...] = ()


class TargetRepository(NamedTuple):
    """One immutable snapshot of the repository that would become public."""

    mirror: Path
    checkout: Path
    github_repository: str
    main_sha: str


def run_command(command: tuple[str, ...], /, *, cwd: Path) -> CommandResult:
    """Run one fixed argument vector and capture its result.

    Parameters
    ----------
    command
        Executable name followed by its arguments.
    cwd
        Repository root from which to run it.

    Returns
    -------
    CommandResult
        Exit status and decoded standard streams. A missing executable is a
        normal nonzero result so that the owning precondition can name it.
    """
    executable = shutil.which(command[0])

    if executable is None:
        return CommandResult(127, "", f"executable not found: {command[0]}")

    environment = None

    if command[0] == "gitleaks":
        environment = {
            key: value
            for key, value in os.environ.items()
            if not key.startswith("GITLEAKS_")
        }

    try:
        completed = subprocess.run(
            (executable, *command[1:]),
            cwd=cwd,
            check=False,
            capture_output=True,
            env=environment,
            text=True,
        )
    except (OSError, UnicodeError) as error:
        return CommandResult(126, "", f"{type(error).__name__}: {error}")

    return CommandResult(completed.returncode, completed.stdout, completed.stderr)


def _command_failure(command: str, result: CommandResult) -> tuple[str, ...]:
    """Return a compact failure that does not echo command output."""
    return (f"{command} exited with status {result.returncode}",)


def _github_repository(remote_url: str, /) -> str | None:
    """Return ``owner/repository`` for a GitHub remote URL."""
    match = _GITHUB_SCP_REMOTE.fullmatch(remote_url)

    if match is not None:
        remote_path = match.group("path")
    else:
        parsed = urlsplit(remote_url)

        if parsed.hostname is None or parsed.hostname.casefold() != "github.com":
            return None

        remote_path = parsed.path

    normalized = remote_path.strip("/").removesuffix(".git")

    parts = normalized.split("/")

    if len(parts) != 2 or not all(
        part and _GITHUB_COMPONENT.fullmatch(part) for part in parts
    ):
        return None

    return "/".join(parts)


def _failed_preconditions(details: tuple[str, ...], /) -> tuple[Precondition, ...]:
    """Return all six named preconditions as failures."""
    return tuple(
        Precondition(name=name, passed=False, details=details)
        for name in _PRECONDITION_NAMES
    )


def _target_origin(
    root: Path,
    /,
    *,
    runner: CommandRunner,
) -> tuple[tuple[str, str] | None, tuple[str, ...]]:
    """Return the origin URL and its GitHub repository identity."""
    remote = runner(("git", "remote", "get-url", "origin"), cwd=root)

    if remote.returncode != 0:
        return None, _command_failure("target origin lookup", remote)

    remote_lines = remote.stdout.splitlines()

    if len(remote_lines) != 1 or not remote_lines[0]:
        return None, ("target origin lookup returned no single URL",)

    remote_url = remote_lines[0]
    github_repository = _github_repository(remote_url)

    if github_repository is None:
        return None, ("target origin is not an unambiguous GitHub repository",)

    return (remote_url, github_repository), ()


def _prepare_target(
    root: Path,
    directory: Path,
    /,
    *,
    runner: CommandRunner,
) -> tuple[TargetRepository | None, tuple[str, ...]]:
    """Mirror the target origin and check out its exact ``main`` commit."""
    origin, origin_errors = _target_origin(root, runner=runner)

    if origin is None:
        return None, origin_errors

    remote_url, github_repository = origin

    mirror = directory / "repository.git"
    cloned = runner(
        (
            "git",
            "clone",
            "--mirror",
            "--no-local",
            remote_url,
            str(mirror),
        ),
        cwd=root,
    )

    if cloned.returncode != 0:
        return None, _command_failure("target repository mirror", cloned)

    resolved = runner(
        ("git", "rev-parse", "--verify", "refs/heads/main^{commit}"),
        cwd=mirror,
    )

    if resolved.returncode != 0:
        return None, _command_failure("target main lookup", resolved)

    main_sha = resolved.stdout.strip()

    if _OBJECT_ID.fullmatch(main_sha) is None:
        return None, ("target main lookup returned a malformed object id",)

    checkout = directory / "checkout"
    checked_out = runner(
        (
            "git",
            "--git-dir",
            str(mirror),
            "worktree",
            "add",
            "--detach",
            str(checkout),
            main_sha,
        ),
        cwd=root,
    )

    if checked_out.returncode != 0:
        return None, _command_failure("target main checkout", checked_out)

    return (
        TargetRepository(
            mirror=mirror,
            checkout=checkout,
            github_repository=github_repository,
            main_sha=main_sha,
        ),
        (),
    )


def _nul_paths(output: str) -> frozenset[str]:
    """Return the nonempty paths from NUL-delimited Git output."""
    return frozenset(path for path in output.split("\0") if path)


def historical_paths(
    root: Path,
    /,
    *,
    runner: CommandRunner = run_command,
) -> tuple[frozenset[str] | None, tuple[str, ...]]:
    """Return paths committed anywhere in the mirrored advertised refs.

    Parameters
    ----------
    root
        Repository root to inspect.
    runner
        Read-only command capability.

    Returns
    -------
    tuple[frozenset[str] | None, tuple[str, ...]]
        Historical paths and no error, or no paths and a named error.
    """
    result = runner(
        ("git", "log", "--all", "--format=", "--name-only", "-z"),
        cwd=root,
    )

    if result.returncode != 0:
        return None, _command_failure("git history path inventory", result)

    return _nul_paths(result.stdout), ()


def _named_proposal(path: str) -> bool:
    """Return whether a historical path names likely proposal material."""
    if PurePosixPath(path).suffix.casefold() in _OPAQUE_DOCUMENT_SUFFIXES:
        return True

    for part in PurePosixPath(path).parts:
        tokens = tuple(token for token in _PATH_TOKEN.split(part.casefold()) if token)

        if _PROPOSAL_PATH_TOKENS.intersection(tokens):
            return True

        collapsed = "".join(tokens)

        if any(phrase in collapsed for phrase in _PROPOSAL_PATH_PHRASES):
            return True

    return False


def _proposal_marker_paths(
    root: Path,
    /,
    *,
    runner: CommandRunner,
) -> tuple[frozenset[str] | None, tuple[str, ...]]:
    """Return paths containing any distinctive funded-proposal heading."""
    revisions = runner(("git", "rev-list", "--all"), cwd=root)

    if revisions.returncode != 0:
        return None, _command_failure("git proposal-content scan", revisions)

    commits = tuple(dict.fromkeys(revisions.stdout.splitlines()))

    if any(_OBJECT_ID.fullmatch(commit) is None for commit in commits):
        return None, ("git proposal-content scan returned a malformed object id",)

    paths: set[str] = set()

    for commit in commits:
        result = runner(
            (
                "git",
                "grep",
                "-I",
                "-i",
                "-l",
                "-z",
                "-E",
                _PROPOSAL_CONTENT_PATTERN,
                commit,
                "--",
            ),
            cwd=root,
        )

        if result.returncode == 1 and not result.stdout:
            continue

        if result.returncode != 0:
            return None, _command_failure("git proposal-content scan", result)

        prefix = f"{commit}:"

        for record in result.stdout.split("\0"):
            if not record:
                continue

            if not record.startswith(prefix) or len(record) == len(prefix):
                return None, ("git proposal-content scan returned a malformed path",)

            paths.add(record.removeprefix(prefix))

    return frozenset(paths), ()


def license_precondition(
    root: Path,
    main_sha: str,
    /,
    *,
    runner: CommandRunner = run_command,
) -> Precondition:
    """Check that target ``main`` contains the regular canonical MIT license."""
    entry = runner(("git", "ls-tree", "-z", main_sha, "--", "LICENSE"), cwd=root)

    if entry.returncode != 0:
        details = _command_failure("target LICENSE tree lookup", entry)
        return Precondition(name="license present", passed=False, details=details)

    fields = entry.stdout.removesuffix("\0").split(maxsplit=2)

    if (
        len(fields) != 3
        or fields[0] not in {"100644", "100755"}
        or fields[1] != "blob"
        or not fields[2].endswith("\tLICENSE")
    ):
        return Precondition(
            name="license present",
            passed=False,
            details=("target main does not contain a regular root-level LICENSE file",),
        )

    content = runner(("git", "show", f"{main_sha}:LICENSE"), cwd=root)

    if content.returncode != 0:
        details = _command_failure("target LICENSE content lookup", content)
        return Precondition(name="license present", passed=False, details=details)

    passed = content.stdout == _EXPECTED_MIT_LICENSE
    details = (
        () if passed else ("target LICENSE is not the canonical LoFI MIT license",)
    )

    return Precondition("license present", passed, details)


def secret_scan_precondition(
    root: Path,
    /,
    *,
    runner: CommandRunner = run_command,
) -> Precondition:
    """Run pinned gitleaks rules over every ref without repository allowlists."""
    version = runner(("gitleaks", "version"), cwd=root)

    if version.returncode != 0:
        details = _command_failure("gitleaks version check", version)
        return Precondition(
            name="whole-history secret scan clean",
            passed=False,
            details=details,
        )

    if version.stdout.strip() != _GITLEAKS_VERSION:
        found_version = version.stdout.strip() or "<missing>"
        return Precondition(
            name="whole-history secret scan clean",
            passed=False,
            details=(
                f"gitleaks {_GITLEAKS_VERSION} is required; found {found_version}",
            ),
        )

    try:
        with TemporaryDirectory(prefix="lofi-gitleaks-") as directory:
            config = Path(directory) / "gitleaks.toml"
            ignore = Path(directory) / "gitleaksignore"
            config.write_text(_GITLEAKS_CONFIG, encoding="utf-8", newline="\n")
            ignore.write_text("", encoding="utf-8", newline="\n")
            result = runner(
                (
                    "gitleaks",
                    "git",
                    str(root),
                    "--config",
                    str(config),
                    "--gitleaks-ignore-path",
                    str(ignore),
                    "--ignore-gitleaks-allow",
                    "--log-opts=--all",
                    "--redact",
                    "--no-banner",
                ),
                cwd=root,
            )
    except OSError:
        return Precondition(
            name="whole-history secret scan clean",
            passed=False,
            details=("gitleaks scan could not create its audited configuration",),
        )

    passed = result.returncode == 0
    details = () if passed else _command_failure("gitleaks whole-history scan", result)

    return Precondition("whole-history secret scan clean", passed, details)


def proposal_precondition(
    root: Path,
    paths: frozenset[str] | None,
    path_errors: tuple[str, ...],
    /,
    *,
    runner: CommandRunner = run_command,
) -> Precondition:
    """Check that no reachable commit contains funded-proposal material."""
    if paths is None:
        return Precondition(
            name="funded-proposal text absent",
            passed=False,
            details=path_errors,
        )

    marker_paths, marker_errors = _proposal_marker_paths(root, runner=runner)

    if marker_paths is None:
        return Precondition(
            name="funded-proposal text absent",
            passed=False,
            details=marker_errors,
        )

    proposal_paths = sorted(
        marker_paths | frozenset(path for path in paths if _named_proposal(path))
    )

    return Precondition(
        name="funded-proposal text absent",
        passed=not proposal_paths,
        details=tuple(proposal_paths),
    )


def planning_precondition(
    paths: frozenset[str] | None,
    path_errors: tuple[str, ...],
    /,
) -> Precondition:
    """Check that no reachable commit contains the root planning tree."""
    if paths is None:
        return Precondition(
            name="planning content absent",
            passed=False,
            details=path_errors,
        )

    planning_paths = tuple(
        sorted(
            path
            for path in paths
            if PurePosixPath(path).parts
            and PurePosixPath(path).parts[0].casefold() == "planning"
        )
    )

    return Precondition(
        name="planning content absent",
        passed=not planning_paths,
        details=planning_paths,
    )


def documentation_precondition(
    root: Path,
    /,
    *,
    runner: CommandRunner = run_command,
) -> Precondition:
    """Build documentation strictly into a disposable directory."""
    try:
        with TemporaryDirectory(prefix="lofi-public-docs-") as directory:
            result = runner(
                (
                    "uv",
                    "run",
                    "--locked",
                    "--with-requirements",
                    "docs/requirements.txt",
                    "mkdocs",
                    "build",
                    "--strict",
                    "--site-dir",
                    directory,
                ),
                cwd=root,
            )
    except OSError:
        return Precondition(
            name="strict documentation build passes",
            passed=False,
            details=("strict documentation build could not create its site directory",),
        )

    passed = result.returncode == 0
    details = () if passed else _command_failure("strict documentation build", result)

    return Precondition("strict documentation build passes", passed, details)


def _decoded_runs(output: str) -> tuple[dict[str, JsonValue], ...] | None:
    """Decode the structured output of ``gh run list``."""
    try:
        decoded = cast("JsonValue", json.loads(output))
    except json.JSONDecodeError:
        return None

    if not isinstance(decoded, list):
        return None

    runs: list[dict[str, JsonValue]] = []

    for value in decoded:
        if not isinstance(value, dict) or not all(
            isinstance(key, str) for key in value
        ):
            return None

        runs.append(value)

    return tuple(runs)


def ci_precondition(
    root: Path,
    target_sha: str,
    repository: str,
    /,
    *,
    runner: CommandRunner = run_command,
) -> Precondition:
    """Check that local and target ``main`` are the same green commit."""
    local = runner(("git", "rev-parse", "HEAD"), cwd=root)

    if local.returncode != 0:
        return Precondition(
            name="CI green on main",
            passed=False,
            details=_command_failure("local HEAD lookup", local),
        )

    remote = runner(
        (
            "gh",
            "api",
            f"repos/{repository}/commits/main",
            "--jq",
            ".sha",
        ),
        cwd=root,
    )

    if remote.returncode != 0:
        return Precondition(
            name="CI green on main",
            passed=False,
            details=_command_failure("remote main lookup", remote),
        )

    local_sha = local.stdout.strip()
    remote_sha = remote.stdout.strip()

    if not local_sha or local_sha != target_sha or remote_sha != target_sha:
        details = (
            (
                f"local HEAD {local_sha or '<missing>'}, mirrored main {target_sha}, "
                f"and GitHub main {remote_sha or '<missing>'} do not all match"
            ),
        )

        return Precondition(name="CI green on main", passed=False, details=details)

    listed = runner(
        (
            "gh",
            "run",
            "list",
            "--repo",
            repository,
            "--workflow",
            "ci.yml",
            "--branch",
            "main",
            "--limit",
            "20",
            "--json",
            "headSha,status,conclusion,url",
        ),
        cwd=root,
    )

    if listed.returncode != 0:
        return Precondition(
            name="CI green on main",
            passed=False,
            details=_command_failure("CI run lookup", listed),
        )

    runs = _decoded_runs(listed.stdout)

    if runs is None:
        details = ("gh returned malformed CI run data",)
        passed = False
    else:
        matching = next(
            (run for run in runs if run.get("headSha") == target_sha),
            None,
        )

        if matching is None:
            details = (f"no CI run found for main commit {target_sha}",)
            passed = False
        else:
            status = matching.get("status")
            conclusion = matching.get("conclusion")
            passed = status == "completed" and conclusion == "success"
            details = (
                ()
                if passed
                else (
                    (
                        f"status={status}; conclusion={conclusion}; "
                        f"url={matching.get('url')}"
                    ),
                )
            )

    return Precondition(name="CI green on main", passed=passed, details=details)


def preconditions(
    root: Path,
    /,
    *,
    runner: CommandRunner = run_command,
) -> tuple[Precondition, ...]:
    """Run and return all six public-release preconditions."""
    try:
        with TemporaryDirectory(prefix="lofi-public-target-") as directory:
            target, target_errors = _prepare_target(
                root,
                Path(directory),
                runner=runner,
            )

            if target is None:
                return _failed_preconditions(target_errors)

            paths, path_errors = historical_paths(target.mirror, runner=runner)

            return (
                license_precondition(
                    target.mirror,
                    target.main_sha,
                    runner=runner,
                ),
                secret_scan_precondition(target.mirror, runner=runner),
                proposal_precondition(
                    target.mirror,
                    paths,
                    path_errors,
                    runner=runner,
                ),
                planning_precondition(paths, path_errors),
                documentation_precondition(target.checkout, runner=runner),
                ci_precondition(
                    root,
                    target.main_sha,
                    target.github_repository,
                    runner=runner,
                ),
            )
    except OSError:
        return _failed_preconditions(
            ("target repository mirror could not create temporary storage",)
        )


def main(
    argv: Sequence[str] | None = None,
    /,
    *,
    runner: CommandRunner = run_command,
) -> int:
    """Run every precondition, print each result, and return a process status."""
    parser = ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--root",
        type=Path,
        default=Path.cwd(),
        help="repository to inspect (default: current directory)",
    )
    arguments = parser.parse_args(argv)

    try:
        root = arguments.root.resolve()
    except OSError:
        results = _failed_preconditions(("repository root could not be resolved",))
    else:
        results = preconditions(root, runner=runner)

    for result in results:
        stream = sys.stdout if result.passed else sys.stderr
        status = "PASS" if result.passed else "FAIL"
        print(f"{status}: {result.name}", file=stream)

        for detail in result.details:
            print(f"  {detail}", file=stream)

    return 0 if all(result.passed for result in results) else 1


if __name__ == "__main__":
    raise SystemExit(main())
