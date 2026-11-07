"""Configuration-boundary fingerprints and the substrate versions they record."""

from hashlib import sha256
from importlib.metadata import version
from json import dumps, loads
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, get_args, override

import didactic.api as dx

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.spec._plugins import load_plugins
from lofi.spec.components.run import RunSpec

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.spec._coerce import Annotation

__all__ = [
    "SUBSTRATE_DISTRIBUTIONS",
    "BoundaryMismatchError",
    "FingerprintIso",
    "IndexedLanguage",
    "IndexedLanguageRecord",
    "LoweringRouteRecord",
    "RunFingerprint",
    "RunFingerprintRecord",
    "decode_fingerprint",
    "fingerprint_run",
    "installed_versions",
    "require_unchanged",
    "schema_uris",
]

SUBSTRATE_DISTRIBUTIONS: Final = ("didactic", "didactic-settings", "lairs", "panproto")
"""The distributions whose installed versions every fingerprint records."""


@final
class IndexedLanguage(FrozenValue):
    """A checked theory's version, specification and digest with its exact route.

    Parameters
    ----------
    version : str
        The theory format version.
    specification : bytes
        The canonical theory specification.
    digest : str
        The specification's ``sha256`` in lowercase hex.
    route : dx.LoweringRoute
        The source and target versions the theory was lowered between.

    Raises
    ------
    TypeError
        If the version or digest is not a string, the specification is not
        bytes, or the route is not a lowering route.
    ValueError
        If the version, specification or digest is empty, or the digest is not
        the digest of the specification.
    """

    _noun = "fingerprint value"

    __slots__ = ("_digest", "_route", "_specification", "_version")

    def __init__(
        self,
        version: str,
        specification: bytes,
        digest: str,
        route: dx.LoweringRoute,
    ) -> None:
        held_version = ensure_name("a theory version", version)

        if not isinstance(specification, bytes):
            msg = f"a theory specification is not bytes: {specification!r}"
            raise TypeError(msg)

        if not specification:
            msg = "a theory specification is empty"
            raise ValueError(msg)

        held_digest = ensure_name("a theory digest", digest)
        computed = sha256(specification).hexdigest()

        if held_digest != computed:
            msg = (
                f"theory digest {held_digest!r} does not match canonical "
                f"specification digest {computed!r}"
            )
            raise ValueError(msg)

        if type(route) is not dx.LoweringRoute:
            msg = f"an indexed language route is not a lowering route: {route!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_version", held_version)
        object.__setattr__(self, "_specification", specification)
        object.__setattr__(self, "_digest", held_digest)
        object.__setattr__(self, "_route", route)

    @classmethod
    def from_versions(
        cls,
        version: str,
        specification: bytes,
        digest: str,
        *,
        source_version: str,
        target_version: str,
    ) -> IndexedLanguage:
        """Build an indexed language from explicit lowering versions."""
        return cls(
            version,
            specification,
            digest,
            dx.LoweringRoute(source_version, target_version),
        )

    @property
    def version(self) -> str:
        """Return the theory format version."""
        return self._version

    @property
    def specification(self) -> bytes:
        """Return the canonical theory specification."""
        return self._specification

    @property
    def digest(self) -> str:
        """Return the canonical specification digest."""
        return self._digest

    @property
    def route(self) -> dx.LoweringRoute:
        """Return the exact source-to-target version pair."""
        return self._route

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every field and the route."""
        if type(other) is not IndexedLanguage:
            return NotImplemented

        return (
            self._version == other._version
            and self._specification == other._specification
            and self._digest == other._digest
            and self._route == other._route
        )

    @override
    def __hash__(self) -> int:
        """Hash every field and the route."""
        return hash(
            (
                IndexedLanguage,
                self._version,
                self._specification,
                self._digest,
                self._route,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"IndexedLanguage({self._version!r}, {self._specification!r}, "
            f"{self._digest!r}, {self._route!r})"
        )


@final
class RunFingerprint(FrozenValue):
    """The content a configuration boundary is identified by, with its digest.

    The document is canonical JSON: keys sorted, no whitespace, every non-ASCII
    character escaped, encoded as UTF-8. It holds the schema URI of every model
    the run specification reaches by qualified name, every indexed language by
    name with its artifact version, digest, hex specification and route
    endpoints, and every recorded package by name with its version. The digest
    is the ``sha256`` of the document in lowercase hex.

    Parameters
    ----------
    schemas : Mapping[str, str]
        The canonical schema URI of every model the run specification reaches,
        itself included, keyed by the model's qualified name.
    languages : Mapping[str, IndexedLanguage]
        Every indexed language keyed by its declared name.
    packages : Mapping[str, str]
        Every recorded distribution keyed by its name.

    Raises
    ------
    TypeError
        If a name, version or schema URI is not a string, or a language is not
        an ``IndexedLanguage``.
    ValueError
        If no schema is named, or a name, version or schema URI is empty.
    """

    _noun = "fingerprint value"

    __slots__ = ("_digest", "_document", "_languages", "_packages", "_schemas")

    def __init__(
        self,
        schemas: Mapping[str, str],
        languages: Mapping[str, IndexedLanguage],
        packages: Mapping[str, str],
    ) -> None:
        if not schemas:
            msg = "a fingerprint names the schema of at least one model"
            raise ValueError(msg)

        held_schemas = {
            ensure_name("a model name", name): ensure_name(
                "a schema URI", schemas[name]
            )
            for name in sorted(schemas)
        }
        held_languages = {
            ensure_name("an indexed language name", name): _language(languages[name])
            for name in sorted(languages)
        }
        held_packages = {
            ensure_name("a package name", name): ensure_name(
                "a package version", packages[name]
            )
            for name in sorted(packages)
        }
        document = dumps(
            {
                "languages": {
                    name: _language_fields(language)
                    for name, language in held_languages.items()
                },
                "packages": held_packages,
                "schemas": held_schemas,
            },
            sort_keys=True,
            separators=(",", ":"),
            ensure_ascii=True,
        ).encode("utf-8")

        object.__setattr__(self, "_schemas", held_schemas)
        object.__setattr__(self, "_languages", held_languages)
        object.__setattr__(self, "_packages", held_packages)
        object.__setattr__(self, "_document", document)
        object.__setattr__(self, "_digest", sha256(document).hexdigest())

    @property
    def schemas(self) -> Mapping[str, str]:
        """Return the schema URI of every reached model by name, in name order."""
        return MappingProxyType(self._schemas)

    @property
    def languages(self) -> Mapping[str, IndexedLanguage]:
        """Return the indexed languages by name, in name order."""
        return MappingProxyType(self._languages)

    @property
    def packages(self) -> Mapping[str, str]:
        """Return the recorded package versions by name, in name order."""
        return MappingProxyType(self._packages)

    @property
    def document(self) -> bytes:
        """Return the canonical JSON the digest is computed over."""
        return self._document

    @property
    def digest(self) -> str:
        """Return the document's ``sha256`` in lowercase hex."""
        return self._digest

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the canonical documents."""
        if type(other) is not RunFingerprint:
            return NotImplemented

        return self._document == other._document

    @override
    def __hash__(self) -> int:
        """Hash the digest."""
        return hash((RunFingerprint, self._digest))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"RunFingerprint({self._schemas!r}, {self._languages!r}, "
            f"{self._packages!r})"
        )


class LoweringRouteRecord(dx.Model, extra="forbid"):
    """Records the versions an indexed language was lowered between.

    Attributes
    ----------
    source : str
        The version the declaration was lowered from.
    target : str
        The version it was lowered to.
    """

    source: str
    target: str

    @dx.validates("source", "target")
    @classmethod
    def _version_is_named(cls, value: str) -> str:
        """Return a nonempty version."""
        return ensure_name("a route version", value)


class IndexedLanguageRecord(dx.Model, extra="forbid"):
    """Records one indexed language, its specification in lowercase hex.

    Attributes
    ----------
    digest : str
        The specification's ``sha256`` in lowercase hex.
    route : LoweringRouteRecord
        The versions the theory was lowered between.
    specification : str
        The canonical theory specification in lowercase hex.
    version : str
        The theory format version.
    """

    digest: str
    route: LoweringRouteRecord
    specification: str
    version: str

    @dx.validates("specification")
    @classmethod
    def _specification_is_hex(cls, value: str) -> str:
        """Return a lowercase hex specification."""
        try:
            decoded = bytes.fromhex(value)

        except ValueError as failure:
            msg = f"a theory specification is not lowercase hex: {value!r}"
            raise ValueError(msg) from failure

        if decoded.hex() != value:
            msg = f"a theory specification is not lowercase hex: {value!r}"
            raise ValueError(msg)

        return value

    def language(self) -> IndexedLanguage:
        """Return the indexed language the record holds.

        Raises
        ------
        ValueError
            If a field is empty or the digest is not the specification's.
        """
        return IndexedLanguage(
            self.version,
            bytes.fromhex(self.specification),
            self.digest,
            dx.LoweringRoute(self.route.source, self.route.target),
        )


class RunFingerprintRecord(dx.Model, extra="forbid"):
    """Records a configuration-boundary fingerprint beside its digest.

    Attributes
    ----------
    digest : str
        The ``sha256`` of the canonical document in lowercase hex.
    languages : dict[str, IndexedLanguageRecord]
        Every indexed language keyed by its declared name.
    packages : dict[str, str]
        Every recorded distribution's version keyed by its name.
    schemas : dict[str, str]
        The schema URI of every reached model keyed by its qualified name.
    """

    digest: str
    languages: dict[str, IndexedLanguageRecord]
    packages: dict[str, str]
    schemas: dict[str, str]

    def fingerprint(self) -> RunFingerprint:
        """Return the fingerprint the record holds, its digest recomputed.

        Raises
        ------
        ValueError
            If a field fails the fingerprint's checks, or the recorded digest
            is not the digest of the rebuilt document.
        """
        rebuilt = RunFingerprint(
            self.schemas,
            {name: record.language() for name, record in self.languages.items()},
            self.packages,
        )

        if self.digest != rebuilt.digest:
            msg = (
                f"fingerprint digest {self.digest!r} does not match the rebuilt "
                f"document's digest {rebuilt.digest!r}"
            )
            raise ValueError(msg)

        return rebuilt


class FingerprintIso(dx.Iso[RunFingerprint, bytes]):
    """Converts a fingerprint to and from the JSON bytes a run directory holds.

    The bytes are the canonical document's fields with the digest beside them,
    indented for reading.
    """

    @override
    def forward(self, a: RunFingerprint, /) -> bytes:
        """Encode a fingerprint."""
        fields: dict[str, JsonValue] = loads(a.document)
        fields["digest"] = a.digest

        return (dumps(fields, indent=2) + "\n").encode()

    @override
    def backward(self, b: bytes, /) -> RunFingerprint:
        """Decode a fingerprint, recomputing every digest."""
        return RunFingerprintRecord.model_validate_json(b).fingerprint()


def decode_fingerprint(data: bytes, /) -> RunFingerprint:
    """Decode a fingerprint's JSON bytes, refusing any that do not validate.

    Parameters
    ----------
    data : bytes
        The bytes a fingerprint was written as.

    Returns
    -------
    RunFingerprint
        The fingerprint, its every digest recomputed.

    Raises
    ------
    ValueError
        If the bytes are not JSON, do not validate as a fingerprint record, or
        record a digest their content does not give.
    """
    try:
        return FingerprintIso().backward(data)

    except dx.ValidationError as refused:
        msg = f"the bytes do not validate as a run fingerprint: {refused}"
        raise ValueError(msg) from refused


class BoundaryMismatchError(ValueError):
    """A current fingerprint that differs from the committed one.

    Parameters
    ----------
    committed : RunFingerprint
        The fingerprint the record was committed with.
    fresh : RunFingerprint
        The fingerprint computed from the current boundary.
    members : tuple[str, ...]
        The dotted paths of the document members that differ.
    """

    __slots__ = ("committed", "fresh", "members")

    def __init__(
        self,
        *,
        committed: RunFingerprint,
        fresh: RunFingerprint,
        members: tuple[str, ...],
    ) -> None:
        self.committed = committed
        self.fresh = fresh
        self.members: tuple[str, ...] = members

        listed = ", ".join(members)
        msg = (
            f"committed configuration boundary {committed.digest} does not match "
            f"current boundary {fresh.digest} at {listed}"
        )

        super().__init__(msg)


def installed_versions() -> dict[str, str]:
    """Return the installed version of every substrate distribution by name.

    Raises
    ------
    importlib.metadata.PackageNotFoundError
        If a substrate distribution is not installed.
    """
    return {name: version(name) for name in SUBSTRATE_DISTRIBUTIONS}


def _models_in(annotation: Annotation, /) -> Iterator[type[dx.Model]]:
    """Yield every model a field annotation names, at any depth of nesting."""
    if isinstance(annotation, type) and issubclass(annotation, dx.Model):
        yield annotation

        return

    for argument in get_args(annotation):
        yield from _models_in(argument)


def schema_uris(root: type[dx.Model], /) -> dict[str, str]:
    """Return the schema URI of every model a root model reaches, by qualified name.

    A model's own schema URI names each nested model only by its name, so a
    field added to a nested model leaves it unchanged. The models reached are
    the root, every model a field annotation names at any depth (inside a
    tuple, a mapping or an optional), and every registered variant of a tagged
    union reached, followed until no new model appears.

    Parameters
    ----------
    root : type[dx.Model]
        The model the walk starts from.

    Returns
    -------
    dict[str, str]
        By ``<module>.<qualified name>``, the model's schema URI.
    """
    reached: dict[type[dx.Model], None] = {}
    pending: list[type[dx.Model]] = [root]

    while pending:
        model = pending.pop()

        if model in reached:
            continue

        reached[model] = None

        if issubclass(model, dx.TaggedUnion):
            pending.extend(model.__variants__.values())

        for spec in model.__field_specs__.values():
            pending.extend(_models_in(spec.annotation))

    return {
        f"{model.__module__}.{model.__qualname__}": dx.schema_uri(model)
        for model in reached
    }


def fingerprint_run(
    languages: Mapping[str, IndexedLanguage],
    *,
    packages: Mapping[str, str] | None = None,
) -> RunFingerprint:
    """Fingerprint the run specification schema with the given languages.

    Declaration plugins are loaded first, so the schemas cover every installed
    plugin arm rather than whichever happen to be imported, and the schema of
    every model the run specification reaches enters the fingerprint.

    Parameters
    ----------
    languages : Mapping[str, IndexedLanguage]
        Every indexed language keyed by its declared name.
    packages : Mapping[str, str] | None
        The package versions to record, or the installed substrate versions.
    """
    load_plugins()

    return RunFingerprint(
        schema_uris(RunSpec),
        languages,
        installed_versions() if packages is None else packages,
    )


def require_unchanged(committed: RunFingerprint, fresh: RunFingerprint) -> None:
    """Refuse a fresh fingerprint that differs from the committed one.

    Parameters
    ----------
    committed : RunFingerprint
        The fingerprint the record was committed with.
    fresh : RunFingerprint
        The fingerprint computed from the current boundary.

    Raises
    ------
    BoundaryMismatchError
        If any member of the two documents differs.
    """
    if committed == fresh:
        return

    members = _differing("", loads(committed.document), loads(fresh.document))

    raise BoundaryMismatchError(committed=committed, fresh=fresh, members=members)


def _language_fields(language: IndexedLanguage, /) -> dict[str, JsonValue]:
    """Return a language's fields as its canonical document holds them."""
    return {
        "digest": language.digest,
        "route": {
            "source": language.route.source_version,
            "target": language.route.target_version,
        },
        "specification": language.specification.hex(),
        "version": language.version,
    }


def _language(value: IndexedLanguage) -> IndexedLanguage:
    """Return an indexed language."""
    if type(value) is not IndexedLanguage:
        msg = f"an indexed language is not an IndexedLanguage: {value!r}"
        raise TypeError(msg)

    return value


def _differing(prefix: str, left: JsonValue, right: JsonValue) -> tuple[str, ...]:
    """Return the dotted paths at which two JSON values differ."""
    if isinstance(left, dict) and isinstance(right, dict):
        return tuple(
            path
            for name in sorted(left.keys() | right.keys())
            for path in _differing(
                f"{prefix}.{name}" if prefix else name,
                left.get(name),
                right.get(name),
            )
        )

    return () if left == right else (prefix,)
