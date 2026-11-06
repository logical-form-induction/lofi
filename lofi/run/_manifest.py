"""The files a run directory holds and the records they are read back as.

A run directory holds four files. ``manifest.json`` is the experiment
manifest, written as running when the run opens and rewritten with a
terminal status when it closes. The other three are written once, before
training starts: ``config.resolved.json`` is the validated run
specification, ``config.sources.json`` is the source of each of its leaves,
and ``schema.fingerprint`` is the fingerprint record of the configuration
boundary the specification was validated at. The environment the run executed
under, its package versions, source revision and accelerator runtime, and the
seed policy are the manifest's, and no other file repeats them.

A point an experiment submits is bound to that experiment. ``point.json`` records
the freeze the point was submitted under and the seed derivation of its run
seed; it is written when the point is submitted and again when it is opened
for execution, and it is the submission ledger the experiment's status is read
from. ``rows.json`` holds the report rows the point emitted, with the freeze
and seed derivation they were produced under, and is written as the point
completes; a point is completed only when its manifest is closed as completed
and its rows are recorded. ``evaluation-artifacts/<role>/`` holds deterministic
files published by the point's held-out evaluators. ``failure.json`` records the
attempts a point has spent with its seed derivation: an attempt in progress or
interrupted, or a
numerical or infrastructure failure with its diagnostic. The point of a
judgment fit experiment records its judgment fit manifest as ``fit.json``.

A directory an experiment's sweep was priced in holds ``price.json``, the
experiment price with every raw timing it was read off, which is read back as
the price it was written from.
"""

import platform
from importlib import import_module
from importlib.metadata import version
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, override

from lofi._files import replace_path
from lofi.run._boundary.manifest import (
    BindingIso,
    FailureIso,
    ManifestIso,
    PointBinding,
    PointFailure,
    PriceIso,
    RecordedRows,
    RowsIso,
    decode_manifest,
    decode_provenance,
    decode_run_spec,
    indented_json,
    read_file,
)
from lofi.spec import (
    ArchivedLanguage,
    FingerprintIso,
    Provenance,
    RunFingerprint,
    RunSpec,
    decode_fingerprint,
)
from lofi.spec.grammar import archive_bytes

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi.run._boundary.manifest import ExperimentManifest
    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._training import TrainingBlueprint
    from lofi.run._values import ExperimentPrice

__all__ = [
    "FAILURE_NAME",
    "FIT_MANIFEST_NAME",
    "INTERPRETER",
    "MANIFEST_NAME",
    "POINT_NAME",
    "PRICE_NAME",
    "RECORDED_DISTRIBUTIONS",
    "REPORTED_NAME",
    "RESOLVED_NAME",
    "SCHEMA_FINGERPRINT_NAME",
    "SOURCES_NAME",
    "RunRecord",
    "StoredRecord",
    "accelerator_runtime",
    "installed_packages",
    "read_binding",
    "read_failure",
    "read_manifest",
    "read_price",
    "read_record",
    "read_rows",
    "remove_failure",
    "replace_file",
    "training_record",
    "write_binding",
    "write_failure",
    "write_manifest",
    "write_price",
    "write_record",
    "write_rows",
]

MANIFEST_NAME: Final = "manifest.json"
"""The file inside a run directory that holds the experiment manifest."""

RESOLVED_NAME: Final = "config.resolved.json"
"""The file inside a run directory that holds the validated run specification."""

SOURCES_NAME: Final = "config.sources.json"
"""The file inside a run directory that holds the source of each leaf."""

SCHEMA_FINGERPRINT_NAME: Final = "schema.fingerprint"
"""The file inside a run directory that holds the configuration fingerprint."""


PRICE_NAME: Final = "price.json"
"""The file inside a priced directory that holds the pricing report."""

POINT_NAME: Final = "point.json"
"""The file inside a submitted point's directory that binds it to its experiment."""

REPORTED_NAME: Final = "rows.json"
"""The file inside a completed point's directory that holds its report rows."""

FAILURE_NAME: Final = "failure.json"
"""The file inside a failed point's directory that records its failure."""

FIT_MANIFEST_NAME: Final = "fit.json"
"""The file inside a judgment fit point's directory that holds its manifest."""

RECORDED_DISTRIBUTIONS: Final = (
    "didactic",
    "lairs",
    "numpy",
    "panproto",
    "quivers",
    "torch",
    "transformers",
)
"""The distributions every manifest names a version for."""

FRAMEWORK: Final = "torch"
"""The recorded distribution whose build the accelerator runtime is read from."""

INTERPRETER: Final = "python"
"""The name the interpreter's version is recorded under beside the distributions."""


class RunRecord:
    """What a run directory holds beside its manifest, with the declarations it commits.

    Parameters
    ----------
    provenance : Provenance
        The source of each leaf of the resolved configuration.
    fingerprint : RunFingerprint
        The fingerprint of the configuration boundary the run specification
        was validated at.
    declarations : Mapping[str, bytes]
        The canonical bytes of each indexed language's declaration, keyed by
        the declared name. The run's history commits each beside the theory
        the fingerprint names for it.

    Raises
    ------
    ValueError
        If the declarations do not name exactly the languages the
        fingerprint indexes.
    TypeError
        If a declaration is not bytes.
    """

    __slots__ = ("_fingerprint", "_languages", "_provenance")

    def __init__(
        self,
        *,
        provenance: Provenance,
        fingerprint: RunFingerprint,
        declarations: Mapping[str, bytes],
    ) -> None:
        if sorted(declarations) != sorted(fingerprint.languages):
            msg = (
                f"a run record declares languages {sorted(declarations)} under a "
                f"fingerprint indexing {sorted(fingerprint.languages)}"
            )
            raise ValueError(msg)

        self._provenance = provenance
        self._fingerprint = fingerprint
        self._languages = MappingProxyType(
            {
                name: ArchivedLanguage(declarations[name], theory)
                for name, theory in sorted(fingerprint.languages.items())
            }
        )

    @property
    def provenance(self) -> Provenance:
        """Return the source of each configuration leaf."""
        return self._provenance

    @property
    def fingerprint(self) -> RunFingerprint:
        """Return the configuration-boundary fingerprint."""
        return self._fingerprint

    @property
    def languages(self) -> Mapping[str, ArchivedLanguage]:
        """Return each indexed declaration with the theory the fingerprint names."""
        return self._languages

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if type(other) is not RunRecord:
            return NotImplemented

        return (
            self._provenance,
            self._fingerprint,
            dict(self._languages),
        ) == (
            other._provenance,
            other._fingerprint,
            dict(other._languages),
        )

    @override
    def __hash__(self) -> int:
        """Hash every field."""
        return hash(
            (
                RunRecord,
                self._provenance,
                self._fingerprint,
                tuple(self._languages.items()),
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        declarations = {
            name: language.declaration for name, language in self._languages.items()
        }

        return (
            f"RunRecord(provenance={self._provenance!r}, "
            f"fingerprint={self._fingerprint!r}, "
            f"declarations={declarations!r})"
        )


def training_record(
    point: SweepPoint,
    blueprint: TrainingBlueprint,
    /,
) -> RunRecord:
    """Return persisted provenance and indexed declarations for one run."""
    provenance = point.provenance

    if provenance is None:
        msg = f"the executable point {point.point_id!r} carries no provenance"
        raise ValueError(msg)

    return RunRecord(
        provenance=provenance,
        fingerprint=blueprint.run_fingerprint,
        declarations={"grammar": archive_bytes(blueprint.learner.grammar)},
    )


class StoredRecord(NamedTuple):
    """The files of a run directory other than its manifest, read back."""

    resolved: RunSpec
    provenance: Provenance
    fingerprint: RunFingerprint


def installed_packages() -> dict[str, str]:
    """Return the interpreter's version and every recorded distribution's by name.

    A distribution is recorded under the version its metadata names, except
    the framework, which is recorded under the version it reports at runtime.
    The framework's metadata can omit the local build label, such as
    ``+cu130`` or ``+cpu``, that names the accelerator build installed, and two
    builds of one release need not compute the same numbers.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If a recorded distribution is not installed.
    ModuleNotFoundError
        If the framework is not installed.
    """
    packages = {
        INTERPRETER: platform.python_version(),
        **{name: version(name) for name in RECORDED_DISTRIBUTIONS},
    }
    packages[FRAMEWORK] = str(import_module(FRAMEWORK).__version__)

    return packages


def accelerator_runtime() -> str:
    """Return the accelerator runtime the installed framework was built against.

    The result is ``cuda`` followed by the CUDA version, ``mps`` followed by
    the operating system version that provides Metal, or ``cpu`` when the
    framework was built against neither.

    Raises
    ------
    ModuleNotFoundError
        If the framework is not installed.
    """
    torch = import_module(FRAMEWORK)
    cuda = torch.version.cuda

    if cuda is not None:
        return f"cuda {cuda}"

    if torch.backends.mps.is_built():
        return f"mps {platform.mac_ver()[0]}"

    return "cpu"


def write_manifest(directory: Path, manifest: ExperimentManifest, /) -> None:
    """Write a manifest into a run directory, replacing any it holds.

    Parameters
    ----------
    directory : Path
        The run directory, created if absent.
    manifest : ExperimentManifest
        The manifest to write.
    """
    directory.mkdir(parents=True, exist_ok=True)
    replace_file(directory / MANIFEST_NAME, ManifestIso().forward(manifest))


def read_manifest(directory: Path, /) -> ExperimentManifest | None:
    """Return the manifest a run directory holds, or None when it holds none.

    Parameters
    ----------
    directory : Path
        The run directory.

    Raises
    ------
    ValueError
        If the file does not validate as a manifest.
    """
    path = directory / MANIFEST_NAME

    if not path.is_file():
        return None

    return decode_manifest(path.read_bytes())


def write_price(directory: Path, price: ExperimentPrice, /) -> None:
    """Write an experiment price into a directory, replacing any it holds.

    Parameters
    ----------
    directory : Path
        The directory, created if absent.
    price : ExperimentPrice
        The price to write.
    """
    directory.mkdir(parents=True, exist_ok=True)
    replace_file(directory / PRICE_NAME, PriceIso().forward(price))


def read_price(directory: Path, /) -> ExperimentPrice | None:
    """Return the experiment price a directory holds, or None when it holds none.

    Parameters
    ----------
    directory : Path
        The directory read.

    Raises
    ------
    ValueError
        If the file does not validate as an experiment price.
    """
    return read_file(directory / PRICE_NAME, PriceIso())


def write_record(
    directory: Path, manifest: ExperimentManifest, record: RunRecord, /
) -> None:
    """Write the three files a run directory holds beside its manifest.

    Parameters
    ----------
    directory : Path
        The run directory, created if absent.
    manifest : ExperimentManifest
        The manifest whose run specification is the resolved configuration
        and whose package versions name the interpreter and every recorded
        distribution.
    record : RunRecord
        The sources and fingerprint to write.

    Raises
    ------
    ValueError
        If the manifest names no version for the interpreter or a recorded
        distribution.
    """
    missing = sorted(
        name
        for name in (*RECORDED_DISTRIBUTIONS, INTERPRETER)
        if name not in manifest.package_versions
    )

    if missing:
        msg = f"the manifest names no version for {', '.join(missing)}"
        raise ValueError(msg)

    directory.mkdir(parents=True, exist_ok=True)
    replace_file(directory / RESOLVED_NAME, indented_json(manifest.run_spec))
    replace_file(directory / SOURCES_NAME, indented_json(record.provenance))
    replace_file(
        directory / SCHEMA_FINGERPRINT_NAME,
        FingerprintIso().forward(record.fingerprint),
    )


def read_record(directory: Path, /) -> StoredRecord:
    """Read back the files a run directory holds beside its manifest.

    Parameters
    ----------
    directory : Path
        The run directory.

    Raises
    ------
    FileNotFoundError
        If one of the three files is absent.
    ValueError
        If a file does not validate as the record it holds.
    """
    return StoredRecord(
        decode_run_spec((directory / RESOLVED_NAME).read_bytes()),
        decode_provenance((directory / SOURCES_NAME).read_bytes()),
        decode_fingerprint((directory / SCHEMA_FINGERPRINT_NAME).read_bytes()),
    )


def write_binding(directory: Path, binding: PointBinding, /) -> None:
    """Bind a point's directory to the experiment that submitted it.

    Parameters
    ----------
    directory : Path
        The point's directory, created if absent.
    binding : PointBinding
        The freeze and seed derivation the point is submitted under.
    """
    directory.mkdir(parents=True, exist_ok=True)
    replace_file(directory / POINT_NAME, BindingIso().forward(binding))


def read_binding(directory: Path, /) -> PointBinding | None:
    """Return the binding a point's directory records, or None when it records none.

    Parameters
    ----------
    directory : Path
        The point's directory.

    Raises
    ------
    ValueError
        If the file does not record a freeze and a seed derivation.
    """
    return read_file(directory / POINT_NAME, BindingIso())


def write_rows(directory: Path, recorded: RecordedRows, /) -> None:
    """Write the report rows a completed point emitted, replacing any it holds.

    Parameters
    ----------
    directory : Path
        The point's directory, created if absent.
    recorded : RecordedRows
        The rows, in emission order, with the freeze and seed derivation they
        were produced under.

    Raises
    ------
    ValueError
        If a row holds a value JSON cannot hold, such as NaN.
    """
    directory.mkdir(parents=True, exist_ok=True)
    replace_file(directory / REPORTED_NAME, RowsIso().forward(recorded))


def read_rows(directory: Path, /) -> RecordedRows | None:
    """Return the rows a point recorded, or None when it recorded none.

    Parameters
    ----------
    directory : Path
        The point's directory.

    Raises
    ------
    ValueError
        If the file does not record a freeze, a seed derivation and a list of
        row objects.
    """
    return read_file(directory / REPORTED_NAME, RowsIso())


def write_failure(directory: Path, failure: PointFailure, /) -> None:
    """Record a point's failure in its directory.

    Parameters
    ----------
    directory : Path
        The point's directory, created if absent.
    failure : PointFailure
        The failure to record.
    """
    directory.mkdir(parents=True, exist_ok=True)
    replace_file(directory / FAILURE_NAME, FailureIso().forward(failure))


def remove_failure(directory: Path, /) -> None:
    """Remove the failure a point's directory records, if it records one.

    Parameters
    ----------
    directory : Path
        The point's directory.
    """
    (directory / FAILURE_NAME).unlink(missing_ok=True)


def read_failure(directory: Path, /) -> PointFailure | None:
    """Return the failure a point's directory records, or None when it records none.

    Parameters
    ----------
    directory : Path
        The point's directory.

    Raises
    ------
    ValueError
        If the file does not record a failure kind, a diagnostic, a positive
        attempt count and a seed derivation.
    """
    return read_file(directory / FAILURE_NAME, FailureIso())


def replace_file(path: Path, content: bytes) -> None:
    """Write a file's whole content so that a reader sees either version.

    Parameters
    ----------
    path : Path
        The file to write, under a directory that exists.
    content : bytes
        The new content.
    """
    staged = path.with_name(f".{path.name}.staged")
    staged.write_bytes(content)
    replace_path(staged, path)
