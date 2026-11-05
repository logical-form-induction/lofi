"""The exact bytes a run committed and the identifiers they are checked against.

A committed run holds two byte strings exactly as its run directory persisted
them: the resolved configuration and the boundary fingerprint record. It also
holds every indexed language the run was resolved against, as the declaration's
canonical bytes with the theory specification it checked to. The identifier of
a byte string is its ``sha256`` in lowercase hex, computed over the stored bytes
themselves. Nothing is identified by re-encoding what the bytes decode to.
"""

from hashlib import sha256
from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.spec import (
    ArchivedLanguage,
    RunFingerprint,
    decode_fingerprint,
    require_unchanged,
)
from lofi.spec.grammar import rebuild_grammar

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.spec.grammar import GrammarAssembly

__all__ = ["CommittedRun", "TamperedRecordError", "identify"]


class TamperedRecordError(ValueError):
    """Stored bytes that do not hash to the identifier recorded for them.

    Parameters
    ----------
    point_id : str
        The point whose record was checked.
    subject : str
        Which byte string failed, ``configuration`` or ``fingerprint``.
    recorded : str
        The identifier the record holds.
    computed : str
        The identifier of the bytes as they are now.
    """

    __slots__ = ("computed", "point_id", "recorded", "subject")

    def __init__(
        self, *, point_id: str, subject: str, recorded: str, computed: str
    ) -> None:
        self.point_id = point_id
        self.subject = subject
        self.recorded = recorded
        self.computed = computed
        super().__init__(
            f"the {subject} of point {point_id} hashes to {computed}, "
            f"not to the committed identifier {recorded}"
        )


def identify(stored: bytes, /) -> str:
    """Return the identifier of a stored byte string.

    Parameters
    ----------
    stored : bytes
        The bytes exactly as they were persisted.

    Returns
    -------
    str
        Their ``sha256`` in lowercase hex.
    """
    return sha256(stored).hexdigest()


@final
class CommittedRun(FrozenValue):
    """One run's persisted bytes with the indexed languages it was resolved against.

    Parameters
    ----------
    point_id : str
        The point the run directory is named by.
    configuration : bytes
        The resolved configuration file's bytes.
    fingerprint : bytes
        The boundary fingerprint file's bytes.
    languages : Mapping[str, ArchivedLanguage]
        Every indexed language keyed by its declared name.

    Raises
    ------
    TypeError
        If the configuration or the fingerprint is not bytes, or a language is
        not an ``ArchivedLanguage``.
    ValueError
        If the point identifier, the configuration or the fingerprint is empty,
        the fingerprint does not read as a fingerprint record, or the languages
        are not exactly the ones the fingerprint names with the theories it
        names.
    """

    _noun = "committed run"

    __slots__ = ("_boundary", "_configuration", "_fingerprint", "_languages", "_point")

    def __init__(
        self,
        point_id: str,
        *,
        configuration: bytes,
        fingerprint: bytes,
        languages: Mapping[str, ArchivedLanguage],
    ) -> None:
        point = ensure_name("a point identifier", point_id)
        held_configuration = _stored("configuration", configuration)
        held_fingerprint = _stored("fingerprint", fingerprint)
        boundary = decode_fingerprint(held_fingerprint)
        held_languages = {
            name: _archived(languages[name]) for name in sorted(languages)
        }

        if tuple(held_languages) != tuple(boundary.languages):
            msg = (
                f"point {point} commits languages {list(held_languages)} under a "
                f"fingerprint naming {list(boundary.languages)}"
            )
            raise ValueError(msg)

        for name, language in held_languages.items():
            if language.theory != boundary.languages[name]:
                msg = (
                    f"the archived theory of language {name!r} is not the theory "
                    f"the fingerprint of point {point} names"
                )
                raise ValueError(msg)

        object.__setattr__(self, "_point", point)
        object.__setattr__(self, "_configuration", held_configuration)
        object.__setattr__(self, "_fingerprint", held_fingerprint)
        object.__setattr__(self, "_boundary", boundary)
        object.__setattr__(self, "_languages", held_languages)

    @property
    def point_id(self) -> str:
        """Return the point identifier."""
        return self._point

    @property
    def configuration(self) -> bytes:
        """Return the resolved configuration's persisted bytes."""
        return self._configuration

    @property
    def fingerprint(self) -> bytes:
        """Return the fingerprint record's persisted bytes."""
        return self._fingerprint

    @property
    def boundary(self) -> RunFingerprint:
        """Return the fingerprint the record's bytes read as."""
        return self._boundary

    @property
    def languages(self) -> Mapping[str, ArchivedLanguage]:
        """Return the indexed languages by name, in name order."""
        return dict(self._languages)

    @property
    def configuration_identifier(self) -> str:
        """Return the identifier of the configuration bytes."""
        return identify(self._configuration)

    @property
    def fingerprint_identifier(self) -> str:
        """Return the identifier of the fingerprint bytes."""
        return identify(self._fingerprint)

    def require_identifiers(self, *, configuration: str, fingerprint: str) -> None:
        """Check recorded identifiers against this run's bytes.

        Parameters
        ----------
        configuration : str
            The identifier recorded for the configuration.
        fingerprint : str
            The identifier recorded for the fingerprint.

        Raises
        ------
        TamperedRecordError
            If either identifier is not the identifier of the bytes held.
        """
        for subject, recorded, computed in (
            ("configuration", configuration, self.configuration_identifier),
            ("fingerprint", fingerprint, self.fingerprint_identifier),
        ):
            if recorded != computed:
                raise TamperedRecordError(
                    point_id=self._point,
                    subject=subject,
                    recorded=recorded,
                    computed=computed,
                )

    def require_persisted(self, *, configuration: bytes, fingerprint: bytes) -> None:
        """Check that a run directory still holds the bytes this run committed.

        Parameters
        ----------
        configuration : bytes
            The configuration file's bytes as the directory holds them now.
        fingerprint : bytes
            The fingerprint file's bytes as the directory holds them now.

        Raises
        ------
        TamperedRecordError
            If either file's identifier differs from the committed one.
        """
        for subject, recorded, persisted in (
            ("configuration", self.configuration_identifier, configuration),
            ("fingerprint", self.fingerprint_identifier, fingerprint),
        ):
            computed = identify(persisted)

            if computed != recorded:
                raise TamperedRecordError(
                    point_id=self._point,
                    subject=subject,
                    recorded=recorded,
                    computed=computed,
                )

    def grammar(
        self,
        name: str,
        current: RunFingerprint,
        /,
    ) -> GrammarAssembly:
        """Return the native grammar an indexed declaration builds at this boundary.

        Parameters
        ----------
        name : str
            The declared name of the indexed language.
        current : RunFingerprint
            The current configuration boundary.

        Raises
        ------
        lofi.spec.BoundaryMismatchError
            If the committed and current boundaries differ.
        KeyError
            If the run indexes no language under the name.
        lofi.spec.grammar.ArchiveMismatchError
            If the declaration does not rebuild the indexed theory.
        """
        require_unchanged(self._boundary, current)

        return rebuild_grammar(self._languages[name])

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the point, both byte strings and the languages."""
        if type(other) is not CommittedRun:
            return NotImplemented

        return (
            self._point == other._point
            and self._configuration == other._configuration
            and self._fingerprint == other._fingerprint
            and self._languages == other._languages
        )

    @override
    def __hash__(self) -> int:
        """Hash the point and both byte strings."""
        return hash((CommittedRun, self._point, self._configuration, self._fingerprint))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            f"CommittedRun({self._point!r}, configuration={self._configuration!r}, "
            f"fingerprint={self._fingerprint!r}, languages={self._languages!r})"
        )


def _stored(subject: str, value: bytes) -> bytes:
    """Return a nonempty byte string."""
    if not isinstance(value, bytes):
        msg = f"a committed {subject} is not bytes: {value!r}"
        raise TypeError(msg)

    if not value:
        msg = f"a committed {subject} is empty"
        raise ValueError(msg)

    return value


def _archived(value: ArchivedLanguage) -> ArchivedLanguage:
    """Return an archived language."""
    if type(value) is not ArchivedLanguage:
        msg = f"a committed language is not an ArchivedLanguage: {value!r}"
        raise TypeError(msg)

    return value
