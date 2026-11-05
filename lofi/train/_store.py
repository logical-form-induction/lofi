"""A directory of checkpoints, one subdirectory per save, with blocks keyed by role."""

import json
import os
import shutil
import sys
from hashlib import file_digest, sha256
from io import BufferedWriter, FileIO
from pathlib import Path
from pickle import UnpicklingError
from tempfile import mkdtemp
from typing import TYPE_CHECKING, Final, cast, override

import torch
from torch import Tensor

from lofi._checks import ensure_name
from lofi.train._boundary import disagreements, read_manifest, write_manifest
from lofi.train._checkpoint import (
    Checkpoint,
    CheckpointId,
    CorruptCheckpointError,
    FingerprintMismatchError,
    UnknownCheckpointError,
)

if TYPE_CHECKING:
    from collections.abc import Buffer, Iterable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.train._boundary import CheckpointManifest

__all__ = ["DirectoryStore", "StateTree"]

type StateTree = (
    Tensor
    | bool
    | int
    | float
    | str
    | list[StateTree]
    | tuple[StateTree, ...]
    | dict[str, StateTree]
    | dict[int, StateTree]
    | None
)
"""What one saved block holds: tensors and plain values nested in containers."""

_MANIFEST: Final = "manifest.json"
_BLOCKS: Final = "blocks.json"
_ROLES: Final = "roles"
_RUN_STATE: Final = "run_state"
_SUFFIX: Final = ".pt"
_ORDINAL_WIDTH: Final = 6
_SEPARATORS: Final = frozenset({"/", "\\", "\0"})


class _HashingWriter(BufferedWriter):
    """A buffered file writer that digests every byte it is handed.

    Parameters
    ----------
    raw : FileIO
        The open file the bytes are written through to.
    """

    def __init__(self, raw: FileIO) -> None:
        super().__init__(raw)

        self._digest = sha256()

    @override
    def write(self, buffer: Buffer, /) -> int:
        """Digest the bytes and write them through.

        Parameters
        ----------
        buffer : Buffer
            The bytes to write.
        """
        self._digest.update(buffer)

        return super().write(buffer)

    def hexdigest(self) -> str:
        """Return the digest of everything written so far."""
        return self._digest.hexdigest()


def _file_name(field: str, value: str) -> str:
    """Return a name usable as one path component.

    Parameters
    ----------
    field : str
        What the name names, for the message.
    value : str
        The name.

    Raises
    ------
    TypeError
        If the name is not a string.
    ValueError
        If the name is empty, is a dot name, starts with a dot, or holds a path
        separator.
    """
    name = ensure_name(field, value)

    if name.startswith(".") or any(separator in name for separator in _SEPARATORS):
        msg = f"{field} is not a file name: {name!r}"

        raise ValueError(msg)

    return name


def _distinct_on_disk(field: str, names: Iterable[str]) -> None:
    """Refuse names a case-insensitive file system would store as one.

    Parameters
    ----------
    field : str
        What the names name, for the message.
    names : Iterable[str]
        The names, each already a file name.

    Raises
    ------
    ValueError
        If two names differ only in case.
    """
    by_fold: dict[str, str] = {}

    for name in sorted(names):
        folded = name.casefold()

        if folded in by_fold:
            msg = (
                f"{field} {by_fold[folded]!r} and {name!r} differ only in case, "
                f"which a case-insensitive file system stores as one file"
            )

            raise ValueError(msg)

        by_fold[folded] = name


def _sync(path: Path) -> None:
    """Flush a directory entry to disk.

    Windows cannot open a directory to flush it, so there an entry is as
    durable as the file system records it.

    Parameters
    ----------
    path : Path
        The directory whose entries were just written or renamed.
    """
    if sys.platform == "win32":
        return

    descriptor = os.open(path, os.O_RDONLY)

    try:
        os.fsync(descriptor)

    finally:
        os.close(descriptor)


def _write_block(path: Path, state: StateTree) -> str:
    """Write one block and return the digest of its bytes.

    Parameters
    ----------
    path : Path
        Where to write, under a directory that exists.
    state : StateTree
        What to write.
    """
    with _HashingWriter(FileIO(path, "w")) as handle:
        torch.save(state, handle)
        handle.flush()
        os.fsync(handle.fileno())

        return handle.hexdigest()


def _write_text(path: Path, text: str) -> None:
    """Write one text file and flush it to disk.

    Parameters
    ----------
    path : Path
        Where to write, under a directory that exists.
    text : str
        What to write.
    """
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        handle.write(text)
        handle.flush()
        os.fsync(handle.fileno())


def _read_block(
    checkpoint: CheckpointId,
    path: Path,
    digest: str,
    name: str,
) -> StateTree:
    """Read one block back, refusing bytes other than the ones written.

    Parameters
    ----------
    checkpoint : CheckpointId
        The checkpoint the block belongs to, for the message.
    path : Path
        Where the block was written.
    digest : str
        The digest the block was written with.
    name : str
        The role or reference key the block holds state for, for the message.

    Raises
    ------
    CorruptCheckpointError
        If the block is missing, its bytes no longer digest to what was recorded,
        or it holds anything other than tensors and plain values.
    """
    try:
        with path.open("rb") as handle:
            found = file_digest(handle, "sha256").hexdigest()

    except OSError as error:
        detail = f"the block for {name!r} cannot be read: {error}"

        raise CorruptCheckpointError(checkpoint, detail) from error

    if found != digest:
        detail = (
            f"the block for {name!r} was written with digest {digest} and now "
            f"digests to {found}"
        )

        raise CorruptCheckpointError(checkpoint, detail)

    try:
        loaded = torch.load(path, map_location="cpu", weights_only=True)

    except (UnpicklingError, RuntimeError, OSError) as error:
        detail = f"the block for {name!r} holds more than weights: {error}"

        raise CorruptCheckpointError(checkpoint, detail) from error

    return cast("StateTree", loaded)


def _digests(
    checkpoint: CheckpointId,
    record: JsonValue,
    section: str,
) -> dict[str, str]:
    """Return one section of the block index as a mapping of names to digests.

    Parameters
    ----------
    checkpoint : CheckpointId
        The checkpoint the index belongs to, for the message.
    record : JsonValue
        The decoded JSON of the whole index.
    section : str
        Which section to read.

    Raises
    ------
    CorruptCheckpointError
        If the index or the section is not an object of strings.
    """
    digests = record.get(section) if isinstance(record, dict) else None

    if not isinstance(digests, dict):
        detail = f"the block index holds no {section!r} object"

        raise CorruptCheckpointError(checkpoint, detail)

    read: dict[str, str] = {}

    for name, digest in digests.items():
        if not isinstance(digest, str):
            detail = f"the block index entry {name!r} in {section!r} is not a digest"

            raise CorruptCheckpointError(checkpoint, detail)

        read[name] = digest

    return read


def _ordinal(name: str) -> int | None:
    """Return the save ordinal a directory name starts with, or None for none.

    Parameters
    ----------
    name : str
        A directory name under the store's root.
    """
    head, separator, _ = name.partition("-")

    if separator and len(head) == _ORDINAL_WIDTH and head.isdigit():
        return int(head)

    return None


class DirectoryStore:
    """Checkpoints under one root directory, one subdirectory per save.

    A checkpoint's directory holds ``manifest.json``, a block index ``blocks.json``
    recording the digest of every block, one block per component under
    ``roles/<role>.pt`` and one per run-state reference under
    ``run_state/<key>.pt``. A save is written to a staging directory and renamed
    into place once every file is flushed, so a checkpoint is either listed in
    full or not at all. Handles order saves oldest first.

    Parameters
    ----------
    root : Path
        The directory the checkpoints are written under, created on first save.
    """

    __slots__ = ("_root",)

    def __init__(self, root: Path) -> None:
        self._root = root

    @property
    def root(self) -> Path:
        """Return the directory the checkpoints are written under."""
        return self._root

    def _entries(self) -> Iterator[tuple[int, str]]:
        """Yield the ordinal and name of every checkpoint directory."""
        if not self._root.is_dir():
            return

        for entry in self._root.iterdir():
            ordinal = _ordinal(entry.name)

            if ordinal is not None and entry.is_dir():
                yield ordinal, entry.name

    def _directory(self, checkpoint: CheckpointId) -> Path:
        """Return the directory a handle names.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle.

        Raises
        ------
        UnknownCheckpointError
            If the handle is not the name of a checkpoint directory under the root.
        """
        if (
            not checkpoint
            or checkpoint.startswith(".")
            or any(separator in checkpoint for separator in _SEPARATORS)
            or _ordinal(checkpoint) is None
            or not (self._root / checkpoint).is_dir()
        ):
            raise UnknownCheckpointError(checkpoint)

        return self._root / checkpoint

    def save(
        self,
        manifest: CheckpointManifest,
        states: Mapping[str, StateTree],
        /,
    ) -> CheckpointId:
        """Write a checkpoint and return the handle it was written under.

        Parameters
        ----------
        manifest : CheckpointManifest
            What the checkpoint records about the run it is written from.
        states : Mapping[str, StateTree]
            One block per role the manifest names, keyed by that role, and one per
            run-state reference the manifest holds, keyed by the reference's key.

        Returns
        -------
        CheckpointId
            The handle the checkpoint may be read back under.

        Raises
        ------
        ValueError
            If the blocks are not keyed by exactly the roles and reference keys the
            manifest names, or if a role, a key or the run id cannot name a file.
        """
        roles = manifest.roles
        keys = frozenset(manifest.references)
        given = frozenset(states)
        missing = sorted((roles | keys) - given)
        unnamed = sorted(given - (roles | keys))

        if missing or unnamed:
            msg = (
                f"the manifest names the roles {sorted(roles)} and the run-state "
                f"keys {sorted(keys)}; the blocks lack {missing} and add {unnamed}"
            )

            raise ValueError(msg)

        for role in roles:
            _file_name("a role", role)

        for key in keys:
            _file_name("a run-state key", key)

        _distinct_on_disk("the roles", roles)
        _distinct_on_disk("the run-state keys", keys)

        run_id = _file_name("the run id", manifest.run_id)

        self._root.mkdir(parents=True, exist_ok=True)

        ordinal = max((held for held, _ in self._entries()), default=-1) + 1
        handle = CheckpointId(
            f"{ordinal:0{_ORDINAL_WIDTH}d}-{run_id}-step{manifest.step}"
        )
        staging = Path(mkdtemp(prefix=f".{handle}.", dir=self._root))

        try:
            (staging / _ROLES).mkdir()
            (staging / _RUN_STATE).mkdir()

            index = {
                _ROLES: {
                    role: _write_block(
                        staging / _ROLES / f"{role}{_SUFFIX}", states[role]
                    )
                    for role in sorted(roles)
                },
                _RUN_STATE: {
                    key: _write_block(
                        staging / _RUN_STATE / f"{key}{_SUFFIX}", states[key]
                    )
                    for key in sorted(keys)
                },
            }

            _write_text(staging / _MANIFEST, write_manifest(manifest))
            _write_text(staging / _BLOCKS, json.dumps(index, indent=2, sort_keys=True))
            _sync(staging / _ROLES)
            _sync(staging / _RUN_STATE)
            _sync(staging)
            staging.rename(self._root / handle)

        except BaseException:
            shutil.rmtree(staging, ignore_errors=True)

            raise

        _sync(self._root)

        return handle

    def load(
        self,
        checkpoint: CheckpointId,
        expected: CheckpointManifest,
        /,
    ) -> Checkpoint[StateTree]:
        """Load a checkpoint.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle the checkpoint was written under.
        expected : CheckpointManifest
            The manifest the reading run would write for itself.

        Returns
        -------
        Checkpoint[StateTree]
            The saved manifest, the block of each role and the block of each
            run-state reference.

        Raises
        ------
        UnknownCheckpointError
            If the store holds no checkpoint under that handle.
        FingerprintMismatchError
            If the saved manifest disagrees with the expected one, listing every
            disagreement rather than the first.
        CorruptCheckpointError
            If the manifest or the block index cannot be read as written, the index
            does not cover the manifest's roles and references, or a block's bytes
            are not the ones written.
        """
        directory = self._directory(checkpoint)
        saved = self.manifest(checkpoint)
        reasons = disagreements(expected, saved)

        if reasons:
            raise FingerprintMismatchError(checkpoint, reasons)

        try:
            record = cast(
                "JsonValue",
                json.loads((directory / _BLOCKS).read_text(encoding="utf-8")),
            )

        except (OSError, ValueError) as error:
            detail = f"the block index cannot be read: {error}"

            raise CorruptCheckpointError(checkpoint, detail) from error

        role_digests = _digests(checkpoint, record, _ROLES)
        run_state_digests = _digests(checkpoint, record, _RUN_STATE)

        if frozenset(role_digests) != saved.roles:
            detail = (
                f"the manifest names the roles {sorted(saved.roles)} and the block "
                f"index holds {sorted(role_digests)}"
            )

            raise CorruptCheckpointError(checkpoint, detail)

        if frozenset(run_state_digests) != frozenset(saved.references):
            detail = (
                f"the manifest references run state under "
                f"{sorted(saved.references)} and the block index holds "
                f"{sorted(run_state_digests)}"
            )

            raise CorruptCheckpointError(checkpoint, detail)

        states = {
            role: _read_block(
                checkpoint, directory / _ROLES / f"{role}{_SUFFIX}", digest, role
            )
            for role, digest in role_digests.items()
        }
        run_state = {
            key: _read_block(
                checkpoint, directory / _RUN_STATE / f"{key}{_SUFFIX}", digest, key
            )
            for key, digest in run_state_digests.items()
        }

        return Checkpoint(checkpoint, saved, states, run_state=run_state)

    def manifest(self, checkpoint: CheckpointId, /) -> CheckpointManifest:
        """Return the manifest a checkpoint was written with, without its blocks.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle the checkpoint was written under.

        Raises
        ------
        UnknownCheckpointError
            If the store holds no checkpoint under that handle.
        CorruptCheckpointError
            If the manifest cannot be read as written.
        """
        directory = self._directory(checkpoint)

        try:
            return read_manifest((directory / _MANIFEST).read_text(encoding="utf-8"))

        except (OSError, ValueError) as error:
            raise CorruptCheckpointError(checkpoint, str(error)) from error

    def remove(self, checkpoint: CheckpointId, /) -> None:
        """Remove a checkpoint's directory.

        The directory is first renamed out of the listing, so an interrupted
        removal leaves a staging directory the listing ignores rather than a
        partial checkpoint.

        Parameters
        ----------
        checkpoint : CheckpointId
            The handle the checkpoint was written under.

        Raises
        ------
        UnknownCheckpointError
            If the store holds no checkpoint under that handle.
        """
        directory = self._directory(checkpoint)
        discarded = Path(mkdtemp(prefix=f".{checkpoint}.removed.", dir=self._root))
        target = discarded / checkpoint
        directory.rename(target)
        _sync(self._root)
        shutil.rmtree(discarded)

    def list(self) -> tuple[CheckpointId, ...]:
        """Return checkpoint identifiers in storage order.

        Returns
        -------
        tuple[CheckpointId, ...]
            The handles, oldest first.
        """
        return tuple(CheckpointId(name) for _, name in sorted(self._entries()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this store."""
        return f"DirectoryStore({self._root!r})"
