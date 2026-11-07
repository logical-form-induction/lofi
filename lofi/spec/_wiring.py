"""Component-graph dependency resolution."""

from types import MappingProxyType
from typing import TYPE_CHECKING, NamedTuple, Protocol

from lofi._checks import ensure_name
from lofi.spec._errors import (
    DependencyCycleError,
    LinearizationDisagreementError,
    UnresolvedDependencyError,
    WiringError,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi._types import SubmoduleDeclaration
    from lofi.spec._ports import Dependency

__all__ = [
    "DependencyPlan",
    "DependencySource",
    "SharingSource",
    "check_codec_identity",
    "check_realized_sharing",
    "resolve_dependencies",
    "resolve_sharing",
]


class DependencySource(Protocol):
    """A component declaration that names its producer dependencies."""

    def dependencies(self) -> tuple[Dependency, ...]:
        """Return dependencies in declaration order."""
        ...


class SharingSource(Protocol):
    """A component declaration that names the submodules it lends and holds."""

    def sharing(self) -> Mapping[str, SubmoduleDeclaration]:
        """Return the producer submodules this component holds, by input port."""
        ...

    def publications(self) -> frozenset[str]:
        """Return the names of the submodules this component lends."""
        ...


class DependencyPlan(NamedTuple):
    """A canonical dependency-first order and its producer bindings.

    Attributes
    ----------
    order : tuple[str, ...]
        Every role, each after the roles it depends on.
    producers : Mapping[tuple[str, str], str]
        The producer role supplying each consumer role and input port.
    declarations : Mapping[str, tuple[Dependency, ...]]
        Each role's dependencies, read once, in declaration order.
    """

    order: tuple[str, ...]
    producers: Mapping[tuple[str, str], str]
    declarations: Mapping[str, tuple[Dependency, ...]]


def resolve_dependencies(
    components: Mapping[str, DependencySource],
) -> DependencyPlan:
    """Resolve one component mapping before any component is constructed.

    Parameters
    ----------
    components : Mapping[str, DependencySource]
        Component declarations keyed by role.

    Returns
    -------
    DependencyPlan
        The construction order and the producer of every consumer port. Roles
        are placed in repeated lexical passes, each pass placing every role whose
        producers are already placed.

    Raises
    ------
    UnresolvedDependencyError
        If a dependency names a role no component fills.
    WiringError
        If two dependencies arrive on one consumer port.
    DependencyCycleError
        If the dependencies form a cycle.
    """
    copied = dict(components)

    for role in copied:
        ensure_name("component role", role)

    ordered = {role: copied[role] for role in sorted(copied)}
    declarations = MappingProxyType(
        {role: tuple(component.dependencies()) for role, component in ordered.items()}
    )
    producers: dict[tuple[str, str], str] = {}

    for consumer, dependencies in declarations.items():
        for dependency in dependencies:
            if dependency.role not in ordered:
                raise UnresolvedDependencyError(
                    consumer=consumer,
                    producer=dependency.role,
                    port=dependency.port,
                    reason="no component fills that role",
                )

            arrival = (consumer, dependency.port)
            first = producers.get(arrival)

            if first is not None:
                msg = (
                    f"{consumer}'s {dependency.port} names both {first} "
                    f"and {dependency.role}"
                )
                raise WiringError(msg, left=consumer, right=dependency.role)

            producers[arrival] = dependency.role

    return DependencyPlan(
        _dependency_order(declarations),
        MappingProxyType(producers),
        declarations,
    )


def resolve_sharing(
    components: Mapping[str, SharingSource],
    producers: Mapping[tuple[str, str], str],
) -> Mapping[tuple[str, str], SubmoduleDeclaration]:
    """Resolve every declared submodule sharing before any component is constructed.

    Parameters
    ----------
    components : Mapping[str, SharingSource]
        Component declarations keyed by role.
    producers : Mapping[tuple[str, str], str]
        The producer role supplying each consumer role and input port.

    Returns
    -------
    Mapping[tuple[str, str], SubmoduleDeclaration]
        Each declaration keyed by the consumer role and the port it arrives on.

    Raises
    ------
    WiringError
        If a declaration names a port no dependency supplies, or a submodule its
        producer does not publish.
    """
    sharing: dict[tuple[str, str], SubmoduleDeclaration] = {}

    for consumer in sorted(components):
        for port, declaration in components[consumer].sharing().items():
            producer = producers.get((consumer, port))

            if producer is None:
                msg = (
                    f"no dependency supplies {consumer}'s {port} for submodule sharing"
                )
                raise WiringError(msg, left=consumer, right=consumer)

            if declaration.name not in components[producer].publications():
                msg = (
                    f"{producer} publishes no {declaration.name} submodule for "
                    f"{consumer}'s {port} to hold"
                )
                raise WiringError(msg, left=consumer, right=producer)

            sharing[consumer, port] = declaration

    return MappingProxyType(sharing)


def check_codec_identity[V](
    owner: str,
    codec: V,
    readers: Mapping[str, V],
    /,
) -> None:
    """Refuse any built component that reads terms through another codec.

    Parameters
    ----------
    owner : str
        The name of whatever supplied ``codec`` to the graph.
    codec : V
        The one codec every reader must hold.
    readers : Mapping[str, V]
        The codec each reading component holds, keyed by role.

    Raises
    ------
    LinearizationDisagreementError
        If a reader holds an object other than ``codec``, even one equal to it.
    """
    for role in sorted(readers):
        held = readers[role]

        if held is not codec:
            raise LinearizationDisagreementError(
                left=owner,
                left_codec=f"{type(codec).__name__} at {id(codec):#x}",
                right=role,
                right_codec=f"{type(held).__name__} at {id(held):#x}",
            )


def check_realized_sharing[C](
    components: Mapping[str, C],
    producers: Mapping[tuple[str, str], str],
    sharing: Mapping[tuple[str, str], SubmoduleDeclaration],
    matches: Callable[[C, C, SubmoduleDeclaration | None], bool],
) -> None:
    """Refuse any built edge whose realized sharing differs from its declaration.

    Parameters
    ----------
    components : Mapping[str, C]
        Built components keyed by role.
    producers : Mapping[tuple[str, str], str]
        The producer role supplying each consumer role and input port.
    sharing : Mapping[tuple[str, str], SubmoduleDeclaration]
        The declared sharing keyed by consumer role and port; an edge absent from
        it must share nothing.
    matches : Callable[[C, C, SubmoduleDeclaration | None], bool]
        Whether a consumer and its producer realize a declaration.

    Raises
    ------
    WiringError
        If an edge's consumer and producer do not realize its declaration.
    """
    for (consumer, port), producer in producers.items():
        if not matches(
            components[consumer], components[producer], sharing.get((consumer, port))
        ):
            msg = (
                f"realized submodule sharing from {producer} to {consumer}'s "
                f"{port} differs from its declaration"
            )
            raise WiringError(msg, left=consumer, right=producer)


def _dependency_order(
    declarations: Mapping[str, tuple[Dependency, ...]],
) -> tuple[str, ...]:
    """Return the lexical-pass topological order of component roles."""
    placed: list[str] = []
    left = dict.fromkeys(declarations)

    while left:
        placed_before = len(placed)

        for role in tuple(left):
            if all(dependency.role in placed for dependency in declarations[role]):
                placed.append(role)
                del left[role]

        if len(placed) == placed_before:
            raise DependencyCycleError(
                cycle=_dependency_cycle(declarations, tuple(left))
            )

    return tuple(placed)


def _dependency_cycle(
    declarations: Mapping[str, tuple[Dependency, ...]],
    left: tuple[str, ...],
) -> tuple[str, ...]:
    """Return one deterministic cycle among the roles left unordered."""
    walk = [left[0]]

    while True:
        following = next(
            dependency.role
            for dependency in declarations[walk[-1]]
            if dependency.role in left
        )

        if following in walk:
            return tuple(walk[walk.index(following) :])

        walk.append(following)
