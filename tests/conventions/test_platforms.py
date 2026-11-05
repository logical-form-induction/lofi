"""CI tests, builds and installs on exactly the platforms the package declares."""

import tomllib
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml

if TYPE_CHECKING:
    from lofi._types import JsonValue

TREE: Final = Path(__file__).resolve().parents[2]
WORKFLOWS: Final = TREE / ".github" / "workflows"
HOSTED: Final = ("ubuntu-latest", "macos-latest", "windows-latest")
WHEELS: Final = {
    "manylinux_2_28-x86_64": "x86_64-unknown-linux-gnu",
    "manylinux_2_28-aarch64": "aarch64-unknown-linux-gnu",
    "macos-arm64": "aarch64-apple-darwin",
    "windows-x86_64": "x86_64-pc-windows-msvc",
}
CLASSIFIED: Final = (
    "Operating System :: MacOS",
    "Operating System :: Microsoft :: Windows",
    "Operating System :: POSIX :: Linux",
)


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)

    return value


def _items(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)

    return value


def _text(value: JsonValue) -> str:
    assert isinstance(value, str)

    return value


def _job(workflow: str, job: str) -> dict[str, JsonValue]:
    loaded = yaml.safe_load((WORKFLOWS / workflow).read_text(encoding="utf-8"))

    return _mapping(_mapping(_mapping(loaded)["jobs"])[job])


def _matrix(job: dict[str, JsonValue]) -> dict[str, JsonValue]:
    return _mapping(_mapping(job["strategy"])["matrix"])


def _included(job: dict[str, JsonValue]) -> list[dict[str, JsonValue]]:
    return [_mapping(row) for row in _items(_matrix(job)["include"])]


def test_the_full_suite_runs_on_every_hosted_operating_system() -> None:
    job = _job("ci.yml", "test")

    assert _text(job["runs-on"]) == "${{ matrix.os }}"
    assert tuple(_text(name) for name in _items(_matrix(job)["os"])) == HOSTED
    assert _mapping(job["strategy"])["fail-fast"] is False


def test_each_hosted_suite_run_turns_absolute_wall_clock_budgets_off() -> None:
    for name in ("test", "test-optimized", "test-unchecked"):
        steps = [_mapping(step) for step in _items(_job("ci.yml", name)["steps"])]
        (tests,) = [step for step in steps if step.get("name") == "Tests"]

        assert "-o wall_clock_budgets=false" in _text(tests["run"])


def test_the_static_gates_run_once() -> None:
    assert _text(_job("ci.yml", "static")["runs-on"]) == "ubuntu-latest"


def test_a_wheel_is_built_for_each_supported_target() -> None:
    rows = _included(_job("rust.yml", "wheels"))

    assert {_text(row["platform"]): _text(row["target"]) for row in rows} == WHEELS


def test_each_built_wheel_is_installed_on_a_runner_of_its_own_platform() -> None:
    built = {
        _text(row["platform"]): _text(row["runner"])
        for row in _included(_job("rust.yml", "wheels"))
    }
    installed = {
        _text(row["platform"]): _text(row["runner"])
        for row in _included(_job("rust.yml", "wheel-install"))
    }

    assert installed == built


def test_the_classifiers_name_exactly_the_wheel_operating_systems() -> None:
    project = tomllib.loads((TREE / "pyproject.toml").read_text(encoding="utf-8"))
    classifiers = _items(_mapping(project["project"])["classifiers"])
    systems = tuple(
        sorted(
            _text(name)
            for name in classifiers
            if _text(name).startswith("Operating System ::")
        )
    )

    assert systems == CLASSIFIED
