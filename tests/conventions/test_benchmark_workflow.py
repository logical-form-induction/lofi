"""Hosted runners record timings and compare commits on the same machine."""

import os
import shutil
import subprocess
from pathlib import Path
from typing import TYPE_CHECKING, Final

import yaml
from pytest_benchmark.utils import parse_compare_fail

if TYPE_CHECKING:
    from lofi._types import JsonValue

TREE: Final = Path(__file__).resolve().parents[2]
WORKFLOWS: Final = TREE / ".github" / "workflows"
BENCHMARK_WORKFLOW: Final = WORKFLOWS / "bench.yml"

CHECKOUT: Final = "actions/checkout@3d3c42e5aac5ba805825da76410c181273ba90b1"
SETUP_UV: Final = "astral-sh/setup-uv@20cfd1bf945f4377ade1205e4dbc17946fc9a30d"
RUST_CACHE: Final = "Swatinem/rust-cache@6323deb102c322ba6fcbdcafc7e3dddab59af2b6"
UPLOAD: Final = "actions/upload-artifact@cf430e030ddbb5b0abf93d22962f4752f3646cd9"
HOSTED: Final = ("ubuntu-latest", "macos-latest", "windows-latest")
BASH: Final = shutil.which("bash") or "bash"
GIT: Final = shutil.which("git") or "git"


def _mapping(value: JsonValue) -> dict[str, JsonValue]:
    assert isinstance(value, dict)

    return value


def _items(value: JsonValue) -> list[JsonValue]:
    assert isinstance(value, list)

    return value


def _text(value: JsonValue) -> str:
    assert isinstance(value, str)

    return value


def _workflow() -> dict[str, JsonValue]:
    loaded = yaml.safe_load(BENCHMARK_WORKFLOW.read_text(encoding="utf-8"))

    return _mapping(loaded)


def _job(name: str) -> dict[str, JsonValue]:
    return _mapping(_mapping(_workflow()["jobs"])[name])


def _steps(job: str = "benchmarks") -> list[dict[str, JsonValue]]:
    return [_mapping(step) for step in _items(_job(job)["steps"])]


def _named_step(name: str, job: str = "benchmarks") -> dict[str, JsonValue]:
    matching = [step for step in _steps(job) if step.get("name") == name]

    assert len(matching) == 1

    return matching[0]


def test_the_workflow_directory_contains_only_the_four_declared_routes() -> None:
    files = {path.name for path in WORKFLOWS.iterdir() if path.is_file()}

    assert files == {"bench.yml", "ci.yml", "docs.yml", "rust.yml"}


def test_the_benchmark_workflow_has_each_required_trigger() -> None:
    source = BENCHMARK_WORKFLOW.read_text(encoding="utf-8")
    trigger_block = source.split("\npermissions:", maxsplit=1)[0]

    assert "\non:\n" in trigger_block
    assert "  push:\n    branches: [main]\n" in trigger_block
    assert "  pull_request:\n" in trigger_block
    assert "  workflow_dispatch:\n" in trigger_block
    assert '  schedule:\n    - cron: "0 7 * * 1"\n' in trigger_block


def test_scheduled_and_manual_runs_enable_the_full_benchmark_grid() -> None:
    assert "if" not in _job("instructions")
    assert _text(_job("relative")["if"]) == "github.event_name != 'schedule'"
    assert "if" not in _job("benchmarks")
    scope = _text(_mapping(_job("benchmarks")["env"])["BENCHMARK_SCOPE"])
    assert "github.event_name == 'schedule'" in scope
    assert "github.event_name == 'workflow_dispatch'" in scope
    assert "'full'" in scope
    assert "'bounded'" in scope
    script = _text(_named_step("Record timings")["run"])
    assert '"$BENCHMARK_SCOPE" = full' in script
    assert "--run-slow" in script
    assert '"${scope[@]}"' in script


def test_the_workflow_has_one_job_per_measure() -> None:
    jobs = _mapping(_workflow()["jobs"])

    assert tuple(jobs) == ("benchmarks", "relative", "instructions")


def test_all_timing_jobs_use_hosted_runners() -> None:
    for name in ("benchmarks", "relative"):
        job = _job(name)
        assert job["runs-on"] == "${{ matrix.os }}"
        assert _mapping(_mapping(job["strategy"])["matrix"])["os"] == list(HOSTED)
    assert _job("instructions")["runs-on"] == "ubuntu-latest"


def test_the_workflow_installs_the_locked_tree_with_pinned_actions() -> None:
    workflow = _workflow()

    assert _mapping(workflow["permissions"]) == {"contents": "read"}

    action_steps = [step for step in _steps() if "uses" in step]

    assert [_text(step["uses"]) for step in action_steps] == [
        CHECKOUT,
        SETUP_UV,
        RUST_CACHE,
        UPLOAD,
    ]
    assert _mapping(action_steps[0]["with"]) == {"persist-credentials": False}
    assert _mapping(action_steps[1]["with"]) == {
        "python-version": "3.14.5",
        "enable-cache": True,
    }
    assert _text(_named_step("Install")["run"]) == "uv sync --locked"


def test_every_benchmark_is_recorded_without_cross_machine_thresholds() -> None:
    script = _text(_named_step("Record timings")["run"])
    assert "uv run pytest tests/benchmarks --benchmark-only" in script
    assert "-o wall_clock_budgets=false" in script
    assert '--benchmark-json="$RUNNER_TEMP/benchmarks.json"' in script
    assert "--benchmark-compare" not in script
    assert "--benchmark-save" not in script


def test_timing_results_are_retained_even_when_a_benchmark_fails() -> None:
    step = _named_step("Upload timings")
    assert step["if"] == "${{ !cancelled() }}"
    assert step["uses"] == UPLOAD
    assert _mapping(step["with"]) == {
        "name": "benchmarks-${{ matrix.os }}",
        "path": "${{ runner.temp }}/benchmarks.json",
        "if-no-files-found": "error",
    }


def test_the_relative_comparison_runs_on_every_hosted_operating_system() -> None:
    job = _job("relative")
    strategy = _mapping(job["strategy"])

    assert strategy["fail-fast"] is False
    assert tuple(
        _text(name) for name in _items(_mapping(strategy["matrix"])["os"])
    ) == (HOSTED)


def test_the_relative_comparison_builds_with_pinned_actions_and_toolchain() -> None:
    action_steps = [step for step in _steps("relative") if "uses" in step]

    assert [_text(step["uses"]) for step in action_steps] == [
        CHECKOUT,
        SETUP_UV,
        RUST_CACHE,
    ]
    assert _mapping(action_steps[0]["with"]) == {
        "persist-credentials": False,
        "fetch-depth": 2,
    }
    assert _mapping(action_steps[1]["with"]) == {
        "python-version": "3.14.5",
        "enable-cache": True,
    }
    assert _text(_named_step("Rust toolchain", "relative")["run"]) == (
        "rustup toolchain install"
    )


def test_the_declared_relative_threshold_is_a_regression_check() -> None:
    declared = _text(_mapping(_job("relative")["env"])["RELATIVE_REGRESSION"])
    check = parse_compare_fail(declared)

    assert check.field == "median"
    assert check.threshold > 0


def _find_parent(tmp_path: Path, commits: int) -> tuple[str, str, str]:
    """Run the parent lookup in a fresh repository of ``commits`` commits.

    Returns the step's log, its outputs and its job summary.
    """
    script = _text(_named_step("Find the parent", "relative")["run"])
    repository = tmp_path / "repository"
    repository.mkdir()
    outputs = tmp_path / "outputs"
    summary = tmp_path / "summary"
    outputs.touch()
    summary.touch()
    git = (GIT, "-c", "user.name=CI", "-c", "user.email=ci@example.invalid")
    subprocess.run((*git, "init", "--quiet"), cwd=repository, check=True)

    for index in range(commits):
        subprocess.run(
            (*git, "commit", "--quiet", "--allow-empty", "-m", f"commit {index}"),
            cwd=repository,
            check=True,
        )

    result = subprocess.run(
        (BASH, "-c", script),
        cwd=repository,
        env={
            "PATH": os.environ["PATH"],
            "GITHUB_OUTPUT": str(outputs),
            "GITHUB_STEP_SUMMARY": str(summary),
        },
        check=True,
        capture_output=True,
        text=True,
    )

    return (
        result.stdout,
        outputs.read_text(encoding="utf-8"),
        summary.read_text(encoding="utf-8"),
    )


def test_a_commit_without_a_parent_skips_the_comparison_with_a_notice(
    tmp_path: Path,
) -> None:
    log, outputs, summary = _find_parent(tmp_path, 1)

    assert outputs.splitlines() == ["found=false"]
    assert "::notice title=No relative timing comparison::" in log
    assert "has no parent commit" in log
    assert "has no parent commit" in summary


def test_a_commit_with_a_parent_is_compared_against_it(tmp_path: Path) -> None:
    log, outputs, summary = _find_parent(tmp_path, 2)

    assert outputs.splitlines() == ["found=true"]
    assert "::notice" not in log
    assert summary == ""


def test_each_timing_step_waits_for_a_parent() -> None:
    lookup = _named_step("Find the parent", "relative")

    assert lookup["id"] == "parent"
    assert lookup["shell"] == "bash"

    for name in (
        "Check out the parent",
        "Time the parent",
        "Time this commit against its parent",
    ):
        step = _named_step(name, "relative")

        assert step["if"] == "steps.parent.outputs.found == 'true'"


def test_the_parent_is_timed_on_the_same_machine_without_absolute_budgets() -> None:
    checkout = _named_step("Check out the parent", "relative")
    parent = _named_step("Time the parent", "relative")
    script = _text(parent["run"])

    assert 'git worktree add --detach "$RUNNER_TEMP/parent" HEAD^1' in _text(
        checkout["run"]
    )
    assert parent["working-directory"] == "${{ runner.temp }}/parent"
    assert parent["shell"] == "bash"
    assert "uv sync --locked" in script
    assert "uv run pytest tests/benchmarks --benchmark-only" in script
    assert "-o wall_clock_budgets=false" in script
    assert '--benchmark-storage="$RUNNER_TEMP/relative"' in script
    assert "--benchmark-save=parent" in script
    assert "--benchmark-compare" not in script


def test_the_commit_fails_on_the_declared_regression_against_its_parent() -> None:
    step = _named_step("Time this commit against its parent", "relative")
    script = _text(step["run"])

    assert step["shell"] == "bash"
    assert "working-directory" not in step
    assert "uv sync --locked" in script
    assert "uv run pytest tests/benchmarks --benchmark-only" in script
    assert "-o wall_clock_budgets=false" in script
    assert '--benchmark-storage="$RUNNER_TEMP/relative"' in script
    assert "--benchmark-compare=0001" in script
    assert '--benchmark-compare-fail="$RELATIVE_REGRESSION"' in script
    assert "--benchmark-save" not in script


def test_instruction_counts_compare_against_the_base_on_linux() -> None:
    declared = _text(_mapping(_job("instructions")["env"])["INSTRUCTION_REGRESSION"])
    compare = _text(
        _named_step("Counts at this commit against the base", "instructions")["run"]
    )

    assert declared.startswith("ir=")
    assert declared.endswith("%")
    assert "--save-baseline=base" in _text(
        _named_step("Counts at the base commit", "instructions")["run"]
    )
    assert '--callgrind-limits="$INSTRUCTION_REGRESSION"' in compare
    assert "--baseline=base" in compare


def test_relative_tool_setup_waits_for_an_available_parent() -> None:
    steps = _steps("relative")
    lookup = next(i for i, step in enumerate(steps) if step.get("id") == "parent")

    for index, step in enumerate(steps):
        if (
            step.get("uses") in (SETUP_UV, RUST_CACHE)
            or step.get("name") == "Rust toolchain"
        ):
            assert index > lookup
            assert step["if"] == "steps.parent.outputs.found == 'true'"


def test_instruction_counts_select_only_the_custom_benchmark_harnesses() -> None:
    for name in ("Counts at the base commit", "Counts at this commit against the base"):
        script = _text(_named_step(name, "instructions")["run"])

        assert "--bench '*'" in script
