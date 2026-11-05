"""The benchmark workflow splits its measures between pinned and hosted runners.

The pinned host holds the absolute wall-clock budgets and the committed
baselines. Each hosted operating system compares a commit's timings with its
parent's on one machine, and Linux compares instruction counts.
"""

import json
import os
import shutil
import subprocess
import sys
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
RUNNER: Final = ("self-hosted", "macOS", "ARM64", "lofi-benchmark")
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


def _fullname_check(script: str) -> str:
    marker = 'if ! uv run python - "$baseline" "$current_result" <<\'PY\'\n'

    return script.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _selection_check(script: str) -> str:
    marker = "\"$saved_fullnames\" <<'PY'\n"

    return script.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


def _coverage_check(script: str) -> str:
    marker = "\"$unowned_selector\" <<'PY'\n"

    return script.split(marker, maxsplit=1)[1].split("\nPY\n", maxsplit=1)[0]


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


def test_a_scheduled_run_counts_instructions_and_times_nothing() -> None:
    assert "if" not in _job("instructions")
    assert _text(_job("relative")["if"]) == "github.event_name != 'schedule'"
    assert _text(_job("benchmarks")["if"]).startswith(
        "github.event_name != 'schedule' && ("
    )


def test_the_workflow_has_one_job_per_measure() -> None:
    jobs = _mapping(_workflow()["jobs"])

    assert tuple(jobs) == ("benchmarks", "relative", "instructions")


def test_only_the_absolute_budgets_select_the_pinned_benchmark_host() -> None:
    job = _job("benchmarks")

    assert tuple(_text(label) for label in _items(job["runs-on"])) == RUNNER
    assert _text(_job("relative")["runs-on"]) == "${{ matrix.os }}"
    assert _text(_job("instructions")["runs-on"]) == "ubuntu-latest"


def test_only_same_repository_pull_requests_reach_the_self_hosted_runner() -> None:
    jobs = _mapping(_workflow()["jobs"])
    job = _mapping(jobs["benchmarks"])

    assert _text(job["if"]) == (
        "github.event_name != 'schedule' && ("
        "github.event_name != 'pull_request' || "
        "github.event.pull_request.head.repo.full_name == github.repository)"
    )


def test_the_workflow_installs_the_locked_tree_with_pinned_actions() -> None:
    workflow = _workflow()

    assert _mapping(workflow["permissions"]) == {"contents": "read"}

    action_steps = [step for step in _steps() if "uses" in step]

    assert [_text(step["uses"]) for step in action_steps] == [CHECKOUT, SETUP_UV]
    assert _mapping(action_steps[0]["with"]) == {"persist-credentials": False}
    assert _mapping(action_steps[1]["with"]) == {
        "python-version": "3.14.5",
        "enable-cache": True,
    }
    assert _text(_named_step("Install")["run"]) == "uv sync --locked"


def test_each_tracked_baseline_owns_one_blocking_comparison() -> None:
    step = _named_step("Time budgets and compare committed baselines")
    script = _text(step["run"])
    loop = script.split(
        "while IFS=$'\\t' read -r baseline selector_file; do", maxsplit=1
    )[1].split('done < "$plan"', maxsplit=1)[0]
    committed = tuple((TREE / "tests" / "benchmarks" / "baselines").glob("*/*.json"))

    assert step["shell"] == "bash"
    assert committed
    assert "git ls-files -- 'tests/benchmarks/baselines/*/*.json'" in script
    assert 'if [[ ! -s "$baseline_list" ]]' in script
    assert script.count("--collect-only") == 1
    assert 'if hasattr(item, "fixturenames")' in script
    assert 'and "benchmark" in item.fixturenames' in script
    assert script.count("uv run pytest") == 2
    unowned = script.split("if ((${#unowned_selectors[@]})); then", maxsplit=1)[
        1
    ].split("while IFS=$'\\t' read -r baseline selector_file; do", maxsplit=1)[0]
    assert 'uv run pytest "${unowned_selectors[@]}" --benchmark-only' in unowned
    assert "--benchmark-compare" not in unowned
    assert "while IFS=$'\\t' read -r baseline selector_file; do" in script
    assert loop.count("uv run pytest") == 1
    assert "uv run pytest tests/benchmarks" not in script
    assert 'if ! uv run pytest "${selectors[@]}"' in loop
    assert "--benchmark-only" in loop
    assert "--benchmark-storage=tests/benchmarks/baselines" in loop
    assert '--benchmark-compare="$baseline"' in loop
    assert "--benchmark-compare-fail=mean:10%" in loop
    assert '--benchmark-json="$current_result"' in loop
    assert 'mktemp "${RUNNER_TEMP}/lofi-benchmark-current.XXXXXX"' in loop
    assert 'uv run python - "$baseline" "$current_result"' in loop
    assert "status=1" in loop
    assert 'done < "$plan"' in script
    assert 'exit "$status"' in script
    assert all(path.name not in script for path in committed)


def test_the_selection_plan_refuses_duplicate_saved_fullnames(tmp_path: Path) -> None:
    script = _text(_named_step("Time budgets and compare committed baselines")["run"])
    baseline_list = tmp_path / "baselines.txt"
    selector_directory = tmp_path / "selectors"
    selector_directory.mkdir()
    plan = tmp_path / "plan.tsv"
    saved_fullnames = tmp_path / "saved.txt"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    fullname = "tests/benchmarks/test_live.py::test_live"
    first.write_text(
        json.dumps({"benchmarks": [{"fullname": fullname}]}),
        encoding="utf-8",
        newline="\n",
    )
    second.write_text(
        json.dumps({"benchmarks": [{"fullname": fullname}]}),
        encoding="utf-8",
        newline="\n",
    )
    baseline_list.write_text(f"{first}\n{second}\n", encoding="utf-8", newline="\n")

    result = subprocess.run(
        (
            sys.executable,
            "-c",
            _selection_check(script),
            str(baseline_list),
            str(selector_directory),
            str(plan),
            str(saved_fullnames),
        ),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    assert fullname in result.stdout
    assert str(first) in result.stdout
    assert str(second) in result.stdout


def test_the_selection_plan_assigns_each_saved_fullname_once(tmp_path: Path) -> None:
    script = _text(_named_step("Time budgets and compare committed baselines")["run"])
    baseline_list = tmp_path / "baselines.txt"
    selector_directory = tmp_path / "selectors"
    selector_directory.mkdir()
    plan = tmp_path / "plan.tsv"
    saved_fullnames = tmp_path / "saved.txt"
    first = tmp_path / "first.json"
    second = tmp_path / "second.json"
    first_fullname = "tests/benchmarks/test_first.py::test_first"
    second_fullname = "tests/benchmarks/test_second.py::test_second[param]"
    first.write_text(
        json.dumps({"benchmarks": [{"fullname": first_fullname}]}),
        encoding="utf-8",
        newline="\n",
    )
    second.write_text(
        json.dumps({"benchmarks": [{"fullname": second_fullname}]}),
        encoding="utf-8",
        newline="\n",
    )
    baseline_list.write_text(f"{first}\n{second}\n", encoding="utf-8", newline="\n")

    result = subprocess.run(
        (
            sys.executable,
            "-c",
            _selection_check(script),
            str(baseline_list),
            str(selector_directory),
            str(plan),
            str(saved_fullnames),
        ),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode == 0
    assignments = [
        line.split("\t") for line in plan.read_text(encoding="utf-8").splitlines()
    ]
    assert [assignment[0] for assignment in assignments] == [str(first), str(second)]
    selected = [
        Path(assignment[1]).read_text(encoding="utf-8").splitlines()
        for assignment in assignments
    ]
    assert selected == [[first_fullname], [second_fullname]]
    assert saved_fullnames.read_text(encoding="utf-8").splitlines() == [
        first_fullname,
        second_fullname,
    ]


def test_the_collected_benchmark_partition_refuses_stale_saved_rows(
    tmp_path: Path,
) -> None:
    script = _text(_named_step("Time budgets and compare committed baselines")["run"])
    saved = tmp_path / "saved.txt"
    current = tmp_path / "current.txt"
    unowned_selector = tmp_path / "unowned.txt"
    shared = "tests/benchmarks/test_live.py::test_live"
    stale = "tests/benchmarks/test_old.py::test_old"
    unowned = "tests/benchmarks/test_new.py::test_new"
    saved.write_text(f"{shared}\n{stale}\n", encoding="utf-8", newline="\n")
    current.write_text(f"{shared}\n{unowned}\n", encoding="utf-8", newline="\n")

    result = subprocess.run(
        (
            sys.executable,
            "-c",
            _coverage_check(script),
            str(saved),
            str(current),
            str(unowned_selector),
        ),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    assert f"stale saved fullname: {stale}" in result.stdout
    assert f"stale saved fullname: {shared}" not in result.stdout


def test_the_collected_benchmark_partition_writes_each_unowned_node_once(
    tmp_path: Path,
) -> None:
    script = _text(_named_step("Time budgets and compare committed baselines")["run"])
    saved = tmp_path / "saved.txt"
    current = tmp_path / "current.txt"
    unowned_selector = tmp_path / "unowned.txt"
    shared = "tests/benchmarks/test_live.py::test_live"
    first_unowned = "tests/benchmarks/test_absolute.py::test_first"
    second_unowned = "tests/benchmarks/test_absolute.py::test_second"
    saved.write_text(f"{shared}\n", encoding="utf-8", newline="\n")
    current.write_text(
        f"{second_unowned}\n{shared}\n{first_unowned}\n", encoding="utf-8", newline="\n"
    )

    result = subprocess.run(
        (
            sys.executable,
            "-c",
            _coverage_check(script),
            str(saved),
            str(current),
            str(unowned_selector),
        ),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode == 0
    assert result.stdout == ""
    assert unowned_selector.read_text(encoding="utf-8").splitlines() == [
        first_unowned,
        second_unowned,
    ]


def test_the_fullname_check_separates_an_orphaned_saved_row(tmp_path: Path) -> None:
    script = _text(_named_step("Time budgets and compare committed baselines")["run"])
    baseline = tmp_path / "baseline.json"
    current = tmp_path / "current.json"
    baseline.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {"fullname": "benchmarks/test_live.py::test_live"},
                    {"fullname": "benchmarks/test_old.py::test_orphan"},
                ]
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    current.write_text(
        json.dumps(
            {
                "benchmarks": [
                    {"fullname": "benchmarks/test_live.py::test_live"},
                ]
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    result = subprocess.run(
        (sys.executable, "-c", _fullname_check(script), str(baseline), str(current)),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert result.returncode == 1
    assert "benchmarks/test_old.py::test_orphan" in result.stdout
    assert "missing: benchmarks/test_old.py::test_orphan" in result.stdout
    assert "missing: benchmarks/test_live.py::test_live" not in result.stdout

    current.write_text(
        baseline.read_text(encoding="utf-8"), encoding="utf-8", newline="\n"
    )
    accepted = subprocess.run(
        (sys.executable, "-c", _fullname_check(script), str(baseline), str(current)),
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert accepted.returncode == 0
    assert accepted.stdout == ""


def test_the_workflow_neither_writes_baselines_nor_switches_debug_checks() -> None:
    source = BENCHMARK_WORKFLOW.read_text(encoding="utf-8")
    absolute = _text(_named_step("Time budgets and compare committed baselines")["run"])

    assert "--benchmark-save" not in absolute
    assert "env" not in _job("benchmarks")
    assert "upload-artifact" not in source
    assert "LOFI_CHECK" not in source


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
