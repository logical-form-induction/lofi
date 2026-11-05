"""Run the unit and golden suites twice and compare their recordings."""

from __future__ import annotations

import json
import multiprocessing
import os
import runpy
import shutil
import sys
from argparse import ArgumentParser
from hashlib import sha256
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch

from lofi.run import RandomStreams, SeedPolicy

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue

_ROOT: Final = Path(__file__).resolve().parents[2]
_RECORDER: Final = _ROOT / "tests" / "golden" / "test_agreement.py"
_RECORDING: Final = "values.json"
_RUNS: Final = ("run-1", "run-2")


def configure_determinism(seed: int) -> None:
    """Seed every random stream from one seed policy and require deterministic torch.

    The streams are seeded through the same random streams a run's points are
    seeded through, so a recording is drawn under the derivation training uses.

    Parameters
    ----------
    seed : int
        Fixed model seed for one run, the root of the seed policy.
    """
    RandomStreams(SeedPolicy(seed)).apply()
    torch.use_deterministic_algorithms(mode=True)


def write_recording(directory: Path) -> Path:
    """Run the golden recorder from an isolated directory.

    Parameters
    ----------
    directory : Path
        New directory that receives the recording.

    Returns
    -------
    Path
        Path to the generated recording.
    """
    directory.mkdir(parents=True)
    copied = directory / _RECORDER.name
    shutil.copyfile(_RECORDER, copied)

    try:
        runpy.run_path(str(copied), run_name="__main__")

    finally:
        copied.unlink()

    return directory / _RECORDING


def recording_digest(path: Path) -> str:
    """Return the SHA-256 digest of the exact recording bytes.

    Parameters
    ----------
    path : Path
        Recording to digest.

    Returns
    -------
    str
        Full hexadecimal digest.
    """
    return sha256(path.read_bytes()).hexdigest()


def _encoded(value: JsonValue) -> str:
    """Return one JSON value in a compact canonical spelling."""
    return json.dumps(value, allow_nan=False, separators=(",", ":"), sort_keys=True)


def _path(parent: str, child: str) -> str:
    """Return a dotted recording path."""
    return child if not parent else f"{parent}.{child}"


def differing_values(
    first: JsonValue,
    second: JsonValue,
    path: str = "",
) -> tuple[str, ...]:
    """Return named leaf differences between two decoded recordings.

    Parameters
    ----------
    first : JsonValue
        Value recorded by the first run.
    second : JsonValue
        Value recorded by the second run.
    path : str
        Path already traversed by a recursive call.

    Returns
    -------
    tuple[str, ...]
        Differences in canonical path order.
    """
    differences: list[str] = []

    if isinstance(first, dict) and isinstance(second, dict):
        for name in sorted(first.keys() | second.keys()):
            nested = _path(path, name)

            if name not in first:
                differences.append(
                    f"{nested}: run-1=<missing>; run-2={_encoded(second[name])}"
                )

            elif name not in second:
                differences.append(
                    f"{nested}: run-1={_encoded(first[name])}; run-2=<missing>"
                )

            else:
                differences.extend(differing_values(first[name], second[name], nested))

        return tuple(differences)

    if isinstance(first, list) and isinstance(second, list):
        shared = min(len(first), len(second))

        for index in range(shared):
            nested = f"{path}[{index}]" if path else f"[{index}]"
            differences.extend(differing_values(first[index], second[index], nested))

        for index in range(shared, len(first)):
            nested = f"{path}[{index}]" if path else f"[{index}]"
            differences.append(
                f"{nested}: run-1={_encoded(first[index])}; run-2=<missing>"
            )

        for index in range(shared, len(second)):
            nested = f"{path}[{index}]" if path else f"[{index}]"
            differences.append(
                f"{nested}: run-1=<missing>; run-2={_encoded(second[index])}"
            )

        return tuple(differences)

    if _encoded(first) != _encoded(second):
        differences.append(
            f"{path or '<root>'}: run-1={_encoded(first)}; run-2={_encoded(second)}"
        )

    return tuple(differences)


def _first_differing_byte(first: bytes, second: bytes) -> str:
    """Return the first byte-level difference between two unequal recordings.

    Parameters
    ----------
    first : bytes
        Exact recording bytes from the first run.
    second : bytes
        Exact recording bytes from the second run.

    Returns
    -------
    str
        The zero-based offset and both byte values, using ``<EOF>`` when one
        recording ends at the offset.

    Raises
    ------
    ValueError
        If the recordings are byte-identical.
    """
    shared = min(len(first), len(second))

    for offset in range(shared):
        if first[offset] != second[offset]:
            return (
                f"first differing byte at offset {offset}: "
                f"run-1=0x{first[offset]:02x}; run-2=0x{second[offset]:02x}"
            )

    if len(first) != len(second):
        first_value = "<EOF>" if len(first) == shared else f"0x{first[shared]:02x}"
        second_value = "<EOF>" if len(second) == shared else f"0x{second[shared]:02x}"

        return (
            f"first differing byte at offset {shared}: "
            f"run-1={first_value}; run-2={second_value}"
        )

    msg = "byte-identical recordings have no differing byte"
    raise ValueError(msg)


def _decoded(path: Path) -> JsonValue:
    """Return one decoded recording."""
    return cast("JsonValue", json.loads(path.read_text(encoding="utf-8")))


def _write_summary(lines: Sequence[str]) -> None:
    """Append human-readable results to the configured CI summary.

    Parameters
    ----------
    lines : Sequence[str]
        Result lines to append.
    """
    summary = os.environ.get("GITHUB_STEP_SUMMARY")

    if summary is None:
        return

    with Path(summary).open("a", encoding="utf-8", newline="\n") as handle:
        handle.write("### Determinism\n\n")

        handle.writelines(f"- {line}\n" for line in lines)


def run_once(directory: str, model_seed: int) -> None:
    """Run one seeded suite and write its golden recording.

    Parameters
    ----------
    directory : str
        Isolated directory for the recording.
    model_seed : int
        Fixed model seed for the run.
    """
    previous = Path.cwd()
    os.chdir(_ROOT)

    try:
        configure_determinism(model_seed)
        status = pytest.main(["tests/unit", "tests/golden"])

        if status != pytest.ExitCode.OK:
            raise SystemExit(int(status))

        configure_determinism(model_seed)
        write_recording(Path(directory))

    finally:
        os.chdir(previous)


def _parser() -> ArgumentParser:
    """Return the command-line parser."""
    parser = ArgumentParser(description=__doc__)
    parser.add_argument(
        "--model-seed",
        type=int,
        default=0,
        help="fixed seed shared by both test runs",
    )

    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """Run both suites and compare their exact golden recordings.

    Parameters
    ----------
    argv : Sequence[str] | None
        Arguments without the program name, or ``None`` to read the process arguments.

    Returns
    -------
    int
        Zero when both suites pass and their recordings are byte-identical.
    """
    arguments = _parser().parse_args(argv)
    context = multiprocessing.get_context("spawn")
    results: list[str] = []

    with TemporaryDirectory(prefix="lofi-determinism-") as temporary:
        root = Path(temporary)
        recordings: list[Path] = []

        for label in _RUNS:
            directory = root / label
            process = context.Process(
                target=run_once,
                args=(str(directory), arguments.model_seed),
                name=f"lofi-determinism-{label}",
            )
            previous_hash_seed = os.environ.get("PYTHONHASHSEED")
            os.environ["PYTHONHASHSEED"] = "random"

            try:
                process.start()

            finally:
                if previous_hash_seed is None:
                    del os.environ["PYTHONHASHSEED"]

                else:
                    os.environ["PYTHONHASHSEED"] = previous_hash_seed

            process.join()
            status = process.exitcode

            if status != 0:
                code = 1 if status is None or status < 0 else status
                line = f"{label} failed with exit code {status!r}"
                sys.stderr.write(f"{line}\n")
                _write_summary((*results, line))

                return code

            recording = directory / _RECORDING
            digest = recording_digest(recording)
            line = f"{label} SHA-256: {digest}"
            sys.stdout.write(f"{line}\n")
            results.append(line)
            recordings.append(recording)

        first_bytes = recordings[0].read_bytes()
        second_bytes = recordings[1].read_bytes()

        if first_bytes == second_bytes:
            line = "recordings are byte-identical"
            sys.stdout.write(f"{line}\n")
            results.append(line)
            _write_summary(results)

            return 0

        differences = differing_values(_decoded(recordings[0]), _decoded(recordings[1]))
        line = "recordings differ"
        sys.stderr.write(f"{line}\n")
        results.append(line)

        for difference in differences:
            named = f"differing value {difference}"
            sys.stderr.write(f"{named}\n")
            results.append(named)

        if not differences:
            byte_difference = _first_differing_byte(first_bytes, second_bytes)
            sys.stderr.write(f"{byte_difference}\n")
            results.append(byte_difference)

        _write_summary(results)

        return 1


if __name__ == "__main__":
    raise SystemExit(main())
