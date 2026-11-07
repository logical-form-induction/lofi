"""The installed diagnostics and their resolution by the roles a model publishes.

A diagnostic is installed by listing its class in the ``lofi.diagnostic``
entry point group under the tag the class declares. Reading the group builds a
catalog, which checks every listed declaration once and then resolves the
diagnostics a set of published roles supports.
"""

from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_name
from lofi.nn.sites import Role
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._ports import Diagnostic

__all__ = ["DiagnosticCatalog", "installed_diagnostics"]


def _checked(listed: str, kind: type[Diagnostic], /) -> type[Diagnostic]:
    """Return a diagnostic class once its declarations are shown well formed.

    Raises
    ------
    RegistryError
        If the class declares a tag other than the one it is listed under.
    TypeError
        If its role or score declarations are not frozen sets of their kind.
    ValueError
        If its tag or a score name is empty.
    """
    tag = ensure_name(f"the tag of {kind.__name__}", kind.tag)

    if tag != listed:
        msg = f"{kind.__name__} declares the tag {tag!r} and is listed as {listed!r}"
        raise RegistryError(msg, family="diagnostic")

    if not isinstance(kind.requires, frozenset) or not all(
        isinstance(role, Role) for role in kind.requires
    ):
        msg = (
            f"{kind.__name__} requires {kind.requires!r}, which is not a "
            "frozenset of roles"
        )
        raise TypeError(msg)

    if not isinstance(kind.scores, frozenset):
        msg = (
            f"{kind.__name__} scores {kind.scores!r}, which is not a frozenset of names"
        )
        raise TypeError(msg)

    for name in kind.scores:
        ensure_name(f"a score name of {kind.__name__}", name)

    return kind


class DiagnosticCatalog:
    """Diagnostic classes indexed by the tags they declare.

    Parameters
    ----------
    kinds : Mapping[str, type[Diagnostic]]
        Each diagnostic class, under the tag it is listed as.

    Raises
    ------
    RegistryError
        If a class declares a tag other than the one it is listed under.
    TypeError
        If a class's role or score declarations are not frozen sets of their kind.
    ValueError
        If a tag or a score name is empty.
    """

    __slots__ = ("_kinds",)

    def __init__(self, kinds: Mapping[str, type[Diagnostic]]) -> None:
        self._kinds: Mapping[str, type[Diagnostic]] = MappingProxyType(
            {tag: _checked(tag, kinds[tag]) for tag in sorted(kinds)}
        )

    @property
    def tags(self) -> tuple[str, ...]:
        """Return the cataloged tags, in ascending order."""
        return tuple(self._kinds)

    def kind(self, tag: str) -> type[Diagnostic]:
        """Return the diagnostic cataloged under one tag.

        Parameters
        ----------
        tag : str
            The tag to resolve.

        Returns
        -------
        type[Diagnostic]
            The cataloged diagnostic class.

        Raises
        ------
        KeyError
            If no diagnostic is cataloged under the tag; the message lists the
            tags that are.
        """
        try:
            return self._kinds[tag]

        except KeyError as error:
            msg = f"no diagnostic is tagged {tag!r}; {self.tags} are installed"
            raise KeyError(msg) from error

    def matching(self, context: DiagnosticContext) -> tuple[Diagnostic, ...]:
        """Return one of every diagnostic the published roles support.

        Parameters
        ----------
        context : DiagnosticContext
            The roles published by the components under study.

        Returns
        -------
        tuple[Diagnostic, ...]
            New applicable instances in ascending tag order.
        """
        return tuple(
            kind() for kind in self._kinds.values() if kind.requires <= context.roles
        )

    def __contains__[TagT](self, tag: TagT) -> bool:
        """Return whether a tag is cataloged here.

        Parameters
        ----------
        tag : TagT
            The tag to test.
        """
        return tag in self._kinds

    def __len__(self) -> int:
        """Return how many diagnostics are cataloged here."""
        return len(self._kinds)

    @override
    def __repr__(self) -> str:
        """Return the cataloged tags."""
        return f"DiagnosticCatalog({list(self.tags)!r})"


def installed_diagnostics() -> DiagnosticCatalog:
    """Return the catalog of every diagnostic the ``lofi.diagnostic`` group lists.

    Returns
    -------
    DiagnosticCatalog
        Each listed diagnostic class, under the tag it is listed as.

    Raises
    ------
    RegistryError
        If a listed class declares a tag other than the one it is listed under.
    TypeError
        If a listed class's role or score declarations are malformed.
    ValueError
        If a listed tag or score name is empty.
    """
    listed: Registry[type[Diagnostic]] = Registry("diagnostic")

    return DiagnosticCatalog(dict(listed.members()))
