"""Repository prose ships the public package and its contribution contract."""

import re
import shutil
import subprocess
from pathlib import Path
from typing import Final

TREE: Final = Path(__file__).resolve().parents[2]


def _read(name: str) -> str:
    return (TREE / name).read_text(encoding="utf-8")


def _level_two_section(document: str, title: str) -> str:
    marker = f"## {title}\n"
    start = document.index(marker) + len(marker)
    remainder = document[start:]
    stop = remainder.find("\n## ")

    if stop == -1:
        return remainder

    return remainder[:stop]


def _git_ignores(candidate: str) -> bool:
    git = shutil.which("git")

    assert git is not None

    completed = subprocess.run(
        [git, "-C", str(TREE), "check-ignore", "--no-index", "--quiet", candidate],
        check=False,
    )

    return completed.returncode == 0


def test_the_license_grants_and_conditions_the_mit_rights() -> None:
    license_text = _read("LICENSE")

    assert license_text.startswith(
        "MIT License\n\nCopyright (c) 2026 Aaron Steven White\n"
    )
    assert "Permission is hereby granted, free of charge" in license_text
    assert (
        "to use, copy, modify, merge, publish, distribute, sublicense" in license_text
    )
    assert (
        "copyright notice and this permission notice shall be included" in license_text
    )
    assert 'THE SOFTWARE IS PROVIDED "AS IS"' in license_text
    assert "AUTHORS OR COPYRIGHT HOLDERS BE LIABLE" in license_text
    assert license_text.endswith("OTHER DEALINGS IN THE\nSOFTWARE.\n")


def test_the_readme_names_lofi_before_describing_it() -> None:
    first_line = _read("README.md").splitlines()[0]

    assert first_line == "# LoFI"


def test_the_readme_install_block_names_the_distribution_once() -> None:
    installation = _level_two_section(_read("README.md"), "Installation")
    blocks = re.findall(r"```[^\n]*\n(.*?)\n```", installation, flags=re.DOTALL)

    assert len(blocks) == 1
    assert len(re.findall(r"\blofi\b", blocks[0], flags=re.IGNORECASE)) == 1
    assert "@ file://" in blocks[0]
    assert "pypi" not in installation.casefold()
    assert "package index" not in installation.casefold()


def test_the_contribution_conventions_cover_the_operational_boundaries() -> None:
    conventions = _level_two_section(_read("CONTRIBUTING.md"), "Conventions").casefold()
    required_terms = (
        "protocols",
        "didactic",
        "typing.any",
        "frozen",
        "named boundary",
        "import graph",
        "side effects",
        "published sites",
        "numpy-style docstrings",
        "suppression comments",
        "empirical result",
    )

    assert all(term in conventions for term in required_terms)


def test_the_unreleased_package_rule_removes_every_superseded_route() -> None:
    section = _level_two_section(_read("CONTRIBUTING.md"), "One current interface")
    current = " ".join(section.split())
    prohibited_routes = (
        "aliases",
        "adapters",
        "deprecated spellings",
        "migration paths",
        "fallback parsers",
        "duplicate registry tags",
        "version branches",
        "compatibility shims",
        "old configuration readers",
    )

    assert "has not been released" in current
    assert "remove the superseded route" in current
    assert all(route in current for route in prohibited_routes)


def test_the_code_of_conduct_has_standards_scope_and_a_reporting_route() -> None:
    conduct = _read("CODE_OF_CONDUCT.md")
    headings = set(re.findall(r"^## (.+)$", conduct, flags=re.MULTILINE))

    assert {
        "Our pledge",
        "Our standards",
        "Enforcement responsibilities",
        "Scope",
        "Reporting and enforcement",
        "Attribution",
    } <= headings
    assert "aaronstevenwhite@gmail.com" in conduct
    assert "Contributor Covenant" in conduct
    assert "harassment-free" in conduct
    assert "[INSERT" not in conduct


def test_the_changelog_opens_its_release_history_with_unreleased() -> None:
    changelog = _read("CHANGELOG.md")
    releases = re.findall(r"^## \[[^]]+\](?: - .+)?$", changelog, flags=re.MULTILINE)
    unreleased = _level_two_section(changelog, "[Unreleased]")

    assert releases[0] == "## [Unreleased]"
    assert "Keep a Changelog" in changelog
    assert "Semantic Versioning" in changelog
    assert "### Added" in unreleased
    assert re.search(r"^- \S", unreleased, flags=re.MULTILINE) is not None
    assert "[Unreleased]: https://" in changelog


def test_data_and_local_files_are_ignored_only_at_the_repository_root() -> None:
    assert _git_ignores("data/example.txt")
    assert _git_ignores(".local/example.txt")
    assert not _git_ignores("nested/data/example.txt")
    assert not _git_ignores("nested/.local/example.txt")
