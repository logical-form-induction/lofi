import threading
from pathlib import Path
from typing import TYPE_CHECKING

import pytest

from lofi._files import REPLACE_ATTEMPTS, replace_path

if TYPE_CHECKING:
    from collections.abc import Callable


def _staged(tmp_path: Path) -> tuple[Path, Path]:
    target = tmp_path / "record.json"
    staged = tmp_path / ".record.json.staged"
    target.write_bytes(b"old")
    staged.write_bytes(b"new")

    return staged, target


def test_a_staged_file_replaces_its_target(tmp_path: Path) -> None:
    staged, target = _staged(tmp_path)

    replace_path(staged, target)

    assert target.read_bytes() == b"new"
    assert not staged.exists()


def test_a_replacement_waits_for_a_reader_holding_the_target_open(
    tmp_path: Path,
) -> None:
    staged, target = _staged(tmp_path)
    reader = target.open("rb")
    release = threading.Timer(0.2, reader.close)
    release.start()

    try:
        replace_path(staged, target)

    finally:
        release.join()
        reader.close()

    assert target.read_bytes() == b"new"


def _refusing(times: int) -> tuple[list[int], Callable[[Path, Path], Path]]:
    calls: list[int] = []
    rename = Path.replace

    def refuse(self: Path, target: Path) -> Path:
        calls.append(len(calls))

        if len(calls) <= times:
            msg = "the target is open in another process"
            raise PermissionError(msg)

        return rename(self, target)

    return calls, refuse


def test_a_refused_replacement_is_tried_again(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged, target = _staged(tmp_path)
    calls, refuse = _refusing(2)
    monkeypatch.setattr(Path, "replace", refuse)

    replace_path(staged, target, interval=0.0)

    assert len(calls) == 3
    assert target.read_bytes() == b"new"


def test_a_replacement_refused_every_attempt_raises(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    staged, target = _staged(tmp_path)
    calls, refuse = _refusing(REPLACE_ATTEMPTS)
    monkeypatch.setattr(Path, "replace", refuse)

    with pytest.raises(PermissionError, match="another process"):
        replace_path(staged, target, interval=0.0)

    assert len(calls) == REPLACE_ATTEMPTS
    assert target.read_bytes() == b"old"
