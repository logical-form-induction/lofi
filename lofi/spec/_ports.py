"""Specification-layer protocols."""

import enum
from typing import Protocol, override, runtime_checkable

import didactic.api as dx

from lofi._checks import ensure_name
from lofi.spec._errors import UnknownSettingError

__all__ = [
    "ChartOutputConsumer",
    "ChartOutputProvider",
    "ConfigSource",
    "Dependency",
    "LeafOrigin",
    "Provenance",
]


@runtime_checkable
class ChartOutputProvider(Protocol):
    """A parser declaration naming the chart values its runtime publishes."""

    def chart_outputs(self) -> frozenset[str]:
        """Return the runtime chart readouts available to consumers."""
        ...


@runtime_checkable
class ChartOutputConsumer(Protocol):
    """A declaration naming the chart values it requires."""

    def required_chart_outputs(self) -> frozenset[str]:
        """Return the runtime chart readouts required by this declaration."""
        ...


class ConfigSource(enum.StrEnum):
    """Identifies a configuration source."""

    #: The schema itself: the default it declares for the field, or the
    #: value it computes from other settings.
    DEFAULT = "default"

    #: A group fragment the document pulled in from the search path.
    GROUP = "group"

    #: The document the loader was pointed at.
    FILE = "file"

    #: The named profile selected on top of that document.
    PROFILE = "profile"

    #: Settings handed to the loader in process, as one point of a sweep is.
    EXTRA = "extra"

    #: An override typed on the command line.
    OVERRIDE = "override"

    @property
    def precedence(self) -> int:
        """Return source precedence from zero."""
        return list(ConfigSource).index(self)

    def outranks(self, other: ConfigSource) -> bool:
        """Return whether this source outranks another.

        Parameters
        ----------
        other : ConfigSource
            Source to compare.

        Returns
        -------
        bool
            Whether this source has higher precedence.
        """
        return self.precedence > other.precedence


class LeafOrigin(dx.Model, extra="forbid"):
    """Where one configuration leaf's value came from.

    Parameters
    ----------
    source : ConfigSource
        The kind of layer that set the leaf.
    layer : str
        Which member of that source: the profile name, the fragment or
        document file, or the override expression. Empty when the source
        has one anonymous member, as the schema default and the in-process
        settings do.
    expression : str | None
        The ``${...}`` text the value was resolved from, when the layer wrote
        an expression rather than a value.
    """

    source: ConfigSource
    layer: str = ""
    expression: str | None = None


class Provenance(dx.Model, extra="forbid"):
    """Records configuration provenance.

    Parameters
    ----------
    origins : dict[str, LeafOrigin]
        Origin keyed by dotted configuration path.
    """

    origins: dict[str, LeafOrigin]

    @dx.validates("origins")
    @classmethod
    def _paths_are_dotted(cls, origins: dict[str, LeafOrigin]) -> dict[str, LeafOrigin]:
        """Validate and order origin paths.

        Parameters
        ----------
        origins : dict[str, LeafOrigin]
            Origin keyed by dotted path.

        Returns
        -------
        dict[str, LeafOrigin]
            Origins ordered by path.

        Raises
        ------
        ValueError
            If a path is empty or contains an empty segment.
        """
        for path in origins:
            if not path or any(not segment for segment in path.split(".")):
                msg = f"{path!r} is not a dotted path"

                raise ValueError(msg)

        return dict(sorted(origins.items()))

    @property
    def paths(self) -> tuple[str, ...]:
        """Return dotted paths in order."""
        return tuple(self.origins)

    def origin_of(self, path: str, /) -> LeafOrigin:
        """Return the origin of one configuration leaf.

        Parameters
        ----------
        path : str
            Dotted configuration path.

        Returns
        -------
        LeafOrigin
            The source that set the leaf, which of its members, and the
            expression the value was resolved from, if any.

        Raises
        ------
        UnknownSettingError
            If no source set ``path``.
        """
        found = self.origins.get(path)

        if found is None:
            raise UnknownSettingError(path=path, declared=self.paths)

        return found

    def source_of(self, path: str, /) -> ConfigSource:
        """Return the source of one configuration leaf.

        Parameters
        ----------
        path : str
            Dotted configuration path.

        Returns
        -------
        ConfigSource
            Source that set the leaf.

        Raises
        ------
        UnknownSettingError
            If no source set ``path``.
        """
        return self.origin_of(path).source


class Dependency:
    """A component dependency.

    Parameters
    ----------
    role : str
        Producer role.
    port : str
        Consumer input port.

    Raises
    ------
    TypeError
        If either value is not a string.
    ValueError
        If either value is empty.
    """

    __slots__ = ("_port", "_role")

    def __init__(self, role: str, port: str) -> None:
        self._role = ensure_name("role", role)
        self._port = ensure_name("port", port)

    @property
    def role(self) -> str:
        """Return the producer role."""
        return self._role

    @property
    def port(self) -> str:
        """Return the consumer port."""
        return self._port

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two dependencies name the same role and port."""
        if not isinstance(other, Dependency):
            return NotImplemented

        return self._role == other._role and self._port == other._port

    @override
    def __hash__(self) -> int:
        """Return a hash over the role and the port."""
        return hash((self._role, self._port))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this dependency."""
        return f"Dependency({self._role!r}, {self._port!r})"
