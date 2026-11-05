from __future__ import annotations

import os
import random
from hashlib import sha256
from pathlib import Path
from typing import TYPE_CHECKING, cast

import pytest
import torch

from lofi.run import RandomStreams, SeedPolicy
from scripts.ci import determinism

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from lofi._types import JsonValue


class FakeProcess:
    def __init__(
        self,
        directory: str,
        payload: bytes,
        status: int,
    ) -> None:
        self._directory = Path(directory)
        self._payload = payload
        self._status = status
        self.exitcode: int | None = None
        self.hash_seed: str | None = None
        self.joined = False

    def start(self) -> None:
        self.hash_seed = os.environ.get("PYTHONHASHSEED")

        if self._status == 0:
            self._directory.mkdir(parents=True)
            (self._directory / "values.json").write_bytes(self._payload)

    def join(self) -> None:
        self.joined = True
        self.exitcode = self._status


class FakeContext:
    def __init__(self, payloads: Sequence[bytes], statuses: Sequence[int]) -> None:
        self._payloads = tuple(payloads)
        self._statuses = tuple(statuses)
        self.processes: list[FakeProcess] = []
        self.targets: list[Callable[[str, int], None]] = []
        self.seeds: list[int] = []
        self.names: list[str] = []
        self.methods: list[str] = []
        self.Process = self.process

    def process(
        self,
        *,
        target: Callable[[str, int], None],
        args: tuple[str, int],
        name: str,
    ) -> FakeProcess:
        index = len(self.processes)
        process = FakeProcess(args[0], self._payloads[index], self._statuses[index])
        self.processes.append(process)
        self.targets.append(target)
        self.seeds.append(args[1])
        self.names.append(name)

        return process


def _context(
    monkeypatch: pytest.MonkeyPatch,
    payloads: Sequence[bytes],
    statuses: Sequence[int] = (0, 0),
) -> FakeContext:
    context = FakeContext(payloads, statuses)

    def get_context(method: str) -> FakeContext:
        context.methods.append(method)

        return context

    monkeypatch.setattr(determinism.multiprocessing, "get_context", get_context)

    return context


def test_the_fixed_seed_seeds_every_stream_as_the_run_streams_do() -> None:
    previous = torch.are_deterministic_algorithms_enabled()

    try:
        determinism.configure_determinism(17)
        drawn = (random.random(), torch.rand(3).tolist())
        RandomStreams(SeedPolicy(17)).apply()
        expected = (random.random(), torch.rand(3).tolist())

        assert drawn == expected
        assert torch.are_deterministic_algorithms_enabled()

        determinism.configure_determinism(18)

        assert (random.random(), torch.rand(3).tolist()) != expected

    finally:
        torch.use_deterministic_algorithms(mode=previous)


def test_the_existing_recorder_writes_canonical_values_in_isolation(
    tmp_path: Path,
) -> None:
    source = Path("tests/golden/test_agreement.py")
    golden = Path("tests/golden/values.json")
    source_before = source.read_bytes()
    golden_before = golden.read_bytes()
    directory = tmp_path / "run"

    recorded = determinism.write_recording(directory)

    assert recorded == directory / "values.json"
    assert recorded.read_bytes() == golden_before
    assert source.read_bytes() == source_before
    assert golden.read_bytes() == golden_before
    assert not (directory / source.name).exists()


def test_the_copied_recorder_is_removed_when_recording_fails(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    directory = tmp_path / "run"

    def fail(path: str, *, run_name: str) -> dict[str, JsonValue]:
        assert Path(path) == directory / "test_agreement.py"
        assert run_name == "__main__"
        msg = "recording failed"

        raise RuntimeError(msg)

    monkeypatch.setattr(determinism.runpy, "run_path", fail)

    with pytest.raises(RuntimeError, match="recording failed"):
        determinism.write_recording(directory)

    assert not (directory / "test_agreement.py").exists()


def test_the_digest_covers_the_exact_recording_bytes(tmp_path: Path) -> None:
    recording = tmp_path / "values.json"
    payload = b'{"value":1}\n'
    recording.write_bytes(payload)

    assert determinism.recording_digest(recording) == sha256(payload).hexdigest()

    recording.write_bytes(b'{"value":1}')

    assert determinism.recording_digest(recording) != sha256(payload).hexdigest()


def test_differences_name_changed_missing_and_indexed_values() -> None:
    first = cast(
        "JsonValue",
        {"metric": {"score": 1.0}, "rows": ["kept", 2], "only_first": True},
    )
    second = cast(
        "JsonValue",
        {"metric": {"score": -1.0}, "rows": ["kept", 3, 4], "only_second": 5},
    )

    assert determinism.differing_values(first, second) == (
        "metric.score: run-1=1.0; run-2=-1.0",
        "only_first: run-1=true; run-2=<missing>",
        "only_second: run-1=<missing>; run-2=5",
        "rows[1]: run-1=2; run-2=3",
        "rows[2]: run-1=<missing>; run-2=4",
    )
    assert determinism.differing_values(0.0, -0.0) == ("<root>: run-1=0.0; run-2=-0.0",)
    assert determinism.differing_values(first, first) == ()


def test_two_spawned_runs_share_the_model_seed_and_randomize_the_hash_seed(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
    tmp_path: Path,
) -> None:
    payload = b'{"metric":{"score":1.0}}\n'
    context = _context(monkeypatch, (payload, payload))
    summary = tmp_path / "summary.md"
    summary.write_text("existing\n", encoding="utf-8", newline="\n")
    monkeypatch.setenv("GITHUB_STEP_SUMMARY", str(summary))
    monkeypatch.setenv("PYTHONHASHSEED", "preserved")

    assert determinism.main(["--model-seed", "31"]) == 0

    output = capsys.readouterr()
    digest = sha256(payload).hexdigest()
    assert context.methods == ["spawn"]
    assert context.targets == [determinism.run_once, determinism.run_once]
    assert context.seeds == [31, 31]
    assert context.names == ["lofi-determinism-run-1", "lofi-determinism-run-2"]
    assert all(process.hash_seed == "random" for process in context.processes)
    assert all(process.joined for process in context.processes)
    assert os.environ["PYTHONHASHSEED"] == "preserved"
    assert output.out == (
        f"run-1 SHA-256: {digest}\n"
        f"run-2 SHA-256: {digest}\n"
        "recordings are byte-identical\n"
    )
    assert output.err == ""
    assert summary.read_text(encoding="utf-8") == (
        "existing\n"
        "### Determinism\n\n"
        f"- run-1 SHA-256: {digest}\n"
        f"- run-2 SHA-256: {digest}\n"
        "- recordings are byte-identical\n"
    )


def test_a_mismatch_reports_both_digests_and_the_differing_value(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = b'{"metric":{"score":1.0}}\n'
    second = b'{"metric":{"score":-1.0}}\n'
    _context(monkeypatch, (first, second))
    monkeypatch.delenv("PYTHONHASHSEED", raising=False)

    assert determinism.main([]) == 1

    output = capsys.readouterr()
    assert output.out == (
        f"run-1 SHA-256: {sha256(first).hexdigest()}\n"
        f"run-2 SHA-256: {sha256(second).hexdigest()}\n"
    )
    assert output.err == (
        "recordings differ\ndiffering value metric.score: run-1=1.0; run-2=-1.0\n"
    )
    assert "PYTHONHASHSEED" not in os.environ


def test_equal_json_with_a_newline_difference_reports_the_first_unequal_byte(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    first = b'{"metric":{"score":1.0}}'
    second = first + b"\n"
    _context(monkeypatch, (first, second))

    assert determinism.main([]) == 1

    output = capsys.readouterr()
    assert output.out == (
        f"run-1 SHA-256: {sha256(first).hexdigest()}\n"
        f"run-2 SHA-256: {sha256(second).hexdigest()}\n"
    )
    assert output.err == (
        "recordings differ\n"
        f"first differing byte at offset {len(first)}: "
        "run-1=<EOF>; run-2=0x0a\n"
    )


def test_a_failed_suite_stops_before_comparison(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    context = _context(monkeypatch, (b"", b""), statuses=(5, 0))

    assert determinism.main([]) == 5

    output = capsys.readouterr()
    assert len(context.processes) == 1
    assert output.out == ""
    assert output.err == "run-1 failed with exit code 5\n"


def test_the_worker_seeds_tests_and_the_recording_independently(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    events: list[tuple[str, int | tuple[str, ...] | Path]] = []

    def configure(seed: int) -> None:
        events.append(("seed", seed))

    def run_tests(arguments: list[str]) -> pytest.ExitCode:
        events.append(("pytest", tuple(arguments)))

        return pytest.ExitCode.OK

    def record(directory: Path) -> Path:
        events.append(("record", directory))

        return directory / "values.json"

    monkeypatch.setattr(determinism, "_ROOT", tmp_path)
    monkeypatch.setattr(determinism, "configure_determinism", configure)
    monkeypatch.setattr(determinism.pytest, "main", run_tests)
    monkeypatch.setattr(determinism, "write_recording", record)

    determinism.run_once(str(tmp_path / "recording"), 37)

    assert events == [
        ("seed", 37),
        ("pytest", ("tests/unit", "tests/golden")),
        ("seed", 37),
        ("record", tmp_path / "recording"),
    ]


def test_the_worker_does_not_record_after_a_test_failure(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    seeds: list[int] = []
    recordings: list[Path] = []

    def fail_tests(arguments: list[str]) -> pytest.ExitCode:
        assert arguments == ["tests/unit", "tests/golden"]

        return pytest.ExitCode.TESTS_FAILED

    monkeypatch.setattr(determinism, "_ROOT", tmp_path)
    monkeypatch.setattr(determinism, "configure_determinism", seeds.append)
    monkeypatch.setattr(determinism.pytest, "main", fail_tests)
    monkeypatch.setattr(determinism, "write_recording", recordings.append)

    with pytest.raises(SystemExit) as raised:
        determinism.run_once(str(tmp_path / "recording"), 41)

    assert raised.value.code == int(pytest.ExitCode.TESTS_FAILED)
    assert seeds == [41]
    assert recordings == []
