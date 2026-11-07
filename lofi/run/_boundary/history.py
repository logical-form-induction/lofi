"""Committed run records, their codec, and the run history that stores them."""

import errno
import json
import os
import sys
from contextlib import contextmanager
from importlib import import_module
from pathlib import Path
from tempfile import TemporaryDirectory
from typing import TYPE_CHECKING, Final, Protocol, cast, override

import didactic.api as dx

if sys.platform == "win32":
    import msvcrt
else:
    import fcntl

from lofi.run._provenance import CommittedRun
from lofi.spec import (
    ArchivedLanguage,
    IndexedLanguage,
    RunSpec,
    fingerprint_run,
    require_unchanged,
)

if TYPE_CHECKING:
    from collections.abc import Iterator

    from lofi._types import JsonValue

__all__ = [
    "CommittedLanguageRecord",
    "CommittedRunIso",
    "CommittedRunRecord",
    "RunHistory",
    "UncommittedRunError",
]


class _Panproto(Protocol):
    """The Panproto exception used at repository operations."""

    VcsError: type[Exception]


def _panproto() -> _Panproto:
    """Load the optional repository substrate at its first operation."""
    return cast("_Panproto", import_module("panproto"))


class CommittedLanguageRecord(dx.Model, extra="forbid"):
    """Records one indexed language of a committed run, each byte string in hex.

    Parameters
    ----------
    declaration : str
        The declaration's canonical bytes.
    version : str
        The theory format version.
    specification : str
        The canonical theory specification.
    digest : str
        The specification's ``sha256`` in lowercase hex.
    source_version : str
        The version the declaration was lowered from.
    target_version : str
        The version it was lowered to.
    """

    declaration: str
    version: str = dx.field(nominal=True)
    specification: str
    digest: str = dx.field(nominal=True)
    source_version: str = dx.field(nominal=True)
    target_version: str = dx.field(nominal=True)


class CommittedRunRecord(dx.Model, extra="forbid"):
    """Records one committed run, each byte string in hex beside its identifier.

    Parameters
    ----------
    point_id : str
        The point the run directory is named by.
    configuration : str
        The resolved configuration file's bytes.
    configuration_identifier : str
        The ``sha256`` of those bytes in lowercase hex.
    fingerprint : str
        The boundary fingerprint file's bytes.
    fingerprint_identifier : str
        The ``sha256`` of those bytes in lowercase hex.
    languages : dict[str, CommittedLanguageRecord]
        Every indexed language keyed by its declared name.
    """

    point_id: str = dx.field(nominal=True)
    configuration: str
    configuration_identifier: str = dx.field(nominal=True)
    fingerprint: str
    fingerprint_identifier: str = dx.field(nominal=True)
    languages: dict[str, CommittedLanguageRecord]


class CommittedRunIso(dx.Iso[CommittedRun, CommittedRunRecord]):
    """Converts a committed run to and from the record a history stores.

    Decoding checks each recorded identifier against the bytes the record
    holds, and never against a re-encoding of what those bytes decode to.
    """

    @override
    def forward(self, a: CommittedRun, /) -> CommittedRunRecord:
        """Encode a committed run."""
        return CommittedRunRecord(
            point_id=a.point_id,
            configuration=a.configuration.hex(),
            configuration_identifier=a.configuration_identifier,
            fingerprint=a.fingerprint.hex(),
            fingerprint_identifier=a.fingerprint_identifier,
            languages={
                name: CommittedLanguageRecord(
                    declaration=language.declaration.hex(),
                    version=language.theory.version,
                    specification=language.theory.specification.hex(),
                    digest=language.theory.digest,
                    source_version=language.theory.route.source_version,
                    target_version=language.theory.route.target_version,
                )
                for name, language in a.languages.items()
            },
        )

    @override
    def backward(self, b: CommittedRunRecord, /) -> CommittedRun:
        """Decode a committed run.

        Raises
        ------
        TamperedRecordError
            If a recorded identifier is not the identifier of its bytes.
        ValueError
            If a byte string is not lowercase hex, a theory digest is not the
            digest of its specification, or the run fails the committed run's
            own checks.
        """
        run = CommittedRun(
            b.point_id,
            configuration=_hex_bytes("configuration", b.configuration),
            fingerprint=_hex_bytes("fingerprint", b.fingerprint),
            languages={
                name: ArchivedLanguage(
                    _hex_bytes("declaration", language.declaration),
                    IndexedLanguage(
                        language.version,
                        _hex_bytes("specification", language.specification),
                        language.digest,
                        dx.LoweringRoute(
                            language.source_version, language.target_version
                        ),
                    ),
                )
                for name, language in b.languages.items()
            },
        )
        run.require_identifiers(
            configuration=b.configuration_identifier,
            fingerprint=b.fingerprint_identifier,
        )

        return run


def _hex_bytes(subject: str, text: str) -> bytes:
    """Return the bytes a lowercase hex string spells.

    Raises
    ------
    ValueError
        If the text is not lowercase hex with an even number of digits.
    """
    try:
        stored = bytes.fromhex(text)

    except ValueError as refused:
        msg = f"a committed {subject} is not hex: {text!r}"
        raise ValueError(msg) from refused

    if stored.hex() != text:
        msg = f"a committed {subject} is not lowercase hex: {text!r}"
        raise ValueError(msg)

    return stored


class UncommittedRunError(LookupError):
    """A point that a run history holds no commit for.

    Parameters
    ----------
    point_id : str
        The point looked up.
    """

    __slots__ = ("point_id",)

    def __init__(self, point_id: str) -> None:
        self.point_id = point_id
        super().__init__(f"no run is committed for point {point_id}")


class RunHistory:
    """The version history of the runs committed under one directory.

    The history is a repository in the directory, opened when one is there and
    initialized otherwise. Its first commit persists the exact current
    ``RunSpec`` schema, stages the committed-run record as the stable schema at
    HEAD, and binds one dataset to each. Every later commit reuses the persisted
    ``RunSpec`` schema for the resolved configuration and the HEAD schema for the
    committed-run record. A tag named by the point identifier points at the
    commit holding both datasets.

    Replay follows the tag to the commit it persists and reads the dataset
    stored there, so a run is found by the identifiers the repository holds on
    disk rather than by the rows of its log.

    Opening and committing hold an exclusive lock on ``.history.lock`` in the
    directory, so processes sharing one runs directory, as the tasks of an
    array job do, stage and commit one run at a time.

    Parameters
    ----------
    root : Path
        The directory the repository is kept in, created if absent.
    """

    __slots__ = ("_repository", "_root")

    def __init__(self, root: Path) -> None:
        panproto = _panproto()
        with _exclusive(root):
            try:
                repository = dx.Repository.open(root)

            except panproto.VcsError:
                repository = dx.Repository.init(root)

        self._root = root
        self._repository = repository

    @property
    def root(self) -> Path:
        """Return the directory the repository is kept in."""
        return self._root

    def committed(self) -> dict[str, str]:
        """Return the commit each committed point's tag points at, by point."""
        return dict(sorted(self._repository.list_tags()))

    def commit(self, run: CommittedRun, /) -> str:
        """Commit a run and tag its commit with the point identifier.

        A run whose point is already committed with the same record is not
        committed again.

        Parameters
        ----------
        run : CommittedRun
            The run to commit.

        Returns
        -------
        str
            The commit the point's tag points at.

        Raises
        ------
        ValueError
            If the point is already committed with a different record.
        """
        _require_current_boundary(run)
        _current_configuration(run)

        with _exclusive(self._root):
            held = self.committed().get(run.point_id)

            if held is not None:
                if self.replay(run.point_id) != run:
                    msg = (
                        f"point {run.point_id} is committed at {held} with a "
                        "different record"
                    )
                    raise ValueError(msg)

                return held

            record = CommittedRunIso().forward(run).model_dump_json()
            head = self._repository.head()

            with TemporaryDirectory() as staging:
                configuration = Path(staging) / "configuration.json"
                configuration.write_bytes(run.configuration)
                metadata = Path(staging) / "record.json"
                metadata.write_text(record, encoding="utf-8", newline="\n")

                if head is None:
                    self._repository.add(RunSpec)
                    self._repository.add_data(
                        configuration,
                        key=_configuration_key(run.point_id),
                    )
                    self._repository.add(CommittedRunRecord)
                else:
                    self._repository.add_data(
                        configuration,
                        key=_configuration_key(run.point_id),
                        schema_id=self._configuration_schema_id(head),
                    )

                self._repository.add_data(metadata, key=run.point_id)

            commit = self._repository.commit(run.point_id, author="lofi")
            self._repository.create_tag(run.point_id, commit)

            return commit

    def replay(self, point_id: str, /) -> CommittedRun:
        """Return the run committed for a point, read from its tagged commit.

        Parameters
        ----------
        point_id : str
            The point whose tag is followed.

        Raises
        ------
        UncommittedRunError
            If no tag names the point, or its commit holds no single record
            for it under its key.
        TamperedRecordError
            If the record's identifiers are not those of its bytes.
        panproto.VcsError
            If a stored object no longer hashes to its identifier.
        """
        panproto = _panproto()
        try:
            commit = self._repository.resolve_ref(point_id)

        except panproto.VcsError as refused:
            raise UncommittedRunError(point_id) from refused

        datasets = self._repository.decoded_data_at(commit)
        expected_keys = {_configuration_key(point_id), point_id}

        if (
            {dataset.key for dataset in datasets} != expected_keys
            or len(datasets) != len(expected_keys)
            or any(dataset.record_count != 1 for dataset in datasets)
        ):
            raise UncommittedRunError(point_id)

        records = _records(datasets, point_id)
        configurations = _records(datasets, _configuration_key(point_id))

        record = records[0] if len(records) == 1 else None
        configuration = configurations[0] if len(configurations) == 1 else None

        if not isinstance(record, dict) or not isinstance(configuration, dict):
            raise UncommittedRunError(point_id)

        run = CommittedRunIso().backward(CommittedRunRecord.model_validate(record))

        if run.point_id != point_id:
            raise UncommittedRunError(point_id)

        _require_current_boundary(run)
        persisted = _current_configuration(run)

        if configuration != persisted:
            msg = (
                f"point {point_id} does not commit the exact current resolved "
                "configuration"
            )
            raise ValueError(msg)

        return run

    def _configuration_schema_id(self, head: str, /) -> str:
        """Return the persisted configuration schema used by the latest run."""
        datasets = [
            dataset
            for dataset in self._repository.data_at(head)
            if dataset.key is not None and dataset.key.endswith(_CONFIGURATION_SUFFIX)
        ]

        if len(datasets) != 1 or datasets[0].record_count != 1:
            msg = "the run history HEAD has no unique configuration dataset"
            raise ValueError(msg)

        return datasets[0].schema_id

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"RunHistory({self._root!r})"


if sys.platform == "win32":

    def _hold(descriptor: int, /) -> None:
        """Block until a descriptor holds the first byte of its file.

        A Windows lock request retries for about ten seconds and then reports
        a deadlock, so the request is repeated until it succeeds.
        """
        while True:
            try:
                msvcrt.locking(descriptor, msvcrt.LK_LOCK, 1)

            except OSError as busy:
                if busy.errno != errno.EDEADLK:
                    raise

            else:
                return

    def _release(descriptor: int, /) -> None:
        """Release the first byte of the file a descriptor holds."""
        msvcrt.locking(descriptor, msvcrt.LK_UNLCK, 1)

else:

    def _hold(descriptor: int, /) -> None:
        """Block until a descriptor holds its file's exclusive lock."""
        fcntl.flock(descriptor, fcntl.LOCK_EX)

    def _release(descriptor: int, /) -> None:
        """Release the exclusive lock a descriptor holds."""
        fcntl.flock(descriptor, fcntl.LOCK_UN)


@contextmanager
def _exclusive(root: Path) -> Iterator[None]:
    """Hold the exclusive lock of a history directory, creating the directory.

    The lock is the operating system's lock on ``.history.lock``, so the
    operating system releases it when its holder exits, however it exits.
    """
    root.mkdir(parents=True, exist_ok=True)
    descriptor = os.open(root / ".history.lock", os.O_RDWR | os.O_CREAT)

    try:
        _hold(descriptor)

        try:
            yield

        finally:
            _release(descriptor)

    finally:
        os.close(descriptor)


_CONFIGURATION_SUFFIX: Final = ":configuration"


def _configuration_key(point_id: str, /) -> str:
    """Return the repository key of one point's resolved configuration."""
    return f"{point_id}{_CONFIGURATION_SUFFIX}"


def _records(
    datasets: list[dx.DecodedCommittedDataset], key: str, /
) -> list[JsonValue]:
    """Return the decoded records held under one committed dataset key."""
    return [
        record
        for dataset in datasets
        if dataset.key == key
        for record in dataset.records
    ]


def _require_current_boundary(run: CommittedRun, /) -> None:
    """Refuse a committed run resolved against another current boundary."""
    current = fingerprint_run(
        {name: language.theory for name, language in run.languages.items()}
    )
    require_unchanged(run.boundary, current)


def _current_configuration(run: CommittedRun, /) -> dict[str, JsonValue]:
    """Return a complete configuration validated against the current schema."""
    persisted = cast("JsonValue", json.loads(run.configuration))

    if not isinstance(persisted, dict):
        msg = f"point {run.point_id} does not hold a configuration object"
        raise TypeError(msg)

    resolved = RunSpec.model_validate(persisted)

    if persisted != json.loads(resolved.model_dump_json()):
        msg = (
            f"point {run.point_id} does not hold a complete current resolved "
            "configuration"
        )
        raise ValueError(msg)

    return persisted
