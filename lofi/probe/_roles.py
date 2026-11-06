"""Required model-site roles."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Self, override

from lofi._checks import ensure_name
from lofi.nn.sites import (
    Access,
    ComponentId,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

__all__ = [
    "INVENTORY",
    "ROLE_AXES",
    "ROLE_DTYPE",
    "ComponentKind",
    "RoleInventory",
    "SiteMapBuilder",
    "ensure_publishes",
]


ROLE_AXES: Final[Mapping[Role, tuple[str, ...]]] = MappingProxyType(
    {
        Role.TOKEN_EMBEDDING: ("item", "row", "width"),
        Role.WORD_VECTOR: ("item", "word", "width"),
        Role.TERM_ENCODING: ("term", "width"),
        Role.NODE_ENCODING: ("term", "slot", "width"),
        Role.SYMBOL_DISTRIBUTION: ("term", "slot", "symbol"),
        Role.DECODER_RECURRENT_STATE: ("term", "slot", "width"),
        Role.UNBOUND_VECTOR: ("term", "slot", "width"),
        Role.DECODER_RESIDUAL_STREAM: ("term", "slot", "width"),
        Role.RESIDUAL_STREAM: ("batch", "position", "feature"),
        Role.FFN_PREACTIVATION: ("batch", "position", "feature"),
        Role.CHART_CELL: ("item", "span", "width"),
        Role.OUTSIDE_CELL: ("item", "span", "width"),
        Role.SPAN_REPRESENTATION: ("item", "span", "span_width"),
        Role.RECEIVER_EDGE: ("item", "edge", "width"),
        Role.INSIDE_SCORE: ("item", "span"),
        Role.OUTSIDE_SCORE: ("item", "span"),
        Role.SPLIT_DISTRIBUTION: ("item", "span", "split"),
        Role.CONSTITUENT_MARGINAL: ("item", "span"),
        Role.LOG_PARTITION: ("item",),
        Role.IDENTITY_SCORE: ("item", "span", "split"),
        Role.COMBINATOR_LOGITS: ("cell", "action"),
    }
)
"""Role axes in outermost-first order."""

ROLE_DTYPE: Final[Mapping[Role, DTypeClass]] = MappingProxyType(
    {
        Role.TOKEN_EMBEDDING: DTypeClass.REAL,
        Role.WORD_VECTOR: DTypeClass.REAL,
        Role.TERM_ENCODING: DTypeClass.REAL,
        Role.NODE_ENCODING: DTypeClass.REAL,
        Role.SYMBOL_DISTRIBUTION: DTypeClass.LOG_PROBABILITY,
        Role.DECODER_RECURRENT_STATE: DTypeClass.REAL,
        Role.UNBOUND_VECTOR: DTypeClass.REAL,
        Role.DECODER_RESIDUAL_STREAM: DTypeClass.REAL,
        Role.RESIDUAL_STREAM: DTypeClass.REAL,
        Role.FFN_PREACTIVATION: DTypeClass.REAL,
        Role.CHART_CELL: DTypeClass.REAL,
        Role.OUTSIDE_CELL: DTypeClass.REAL,
        Role.SPAN_REPRESENTATION: DTypeClass.REAL,
        Role.RECEIVER_EDGE: DTypeClass.REAL,
        Role.INSIDE_SCORE: DTypeClass.REAL,
        Role.OUTSIDE_SCORE: DTypeClass.REAL,
        Role.SPLIT_DISTRIBUTION: DTypeClass.LOG_PROBABILITY,
        Role.CONSTITUENT_MARGINAL: DTypeClass.PROBABILITY,
        Role.LOG_PARTITION: DTypeClass.REAL,
        Role.IDENTITY_SCORE: DTypeClass.REAL,
        Role.COMBINATOR_LOGITS: DTypeClass.LOG_PROBABILITY,
    }
)
"""The numeric class required for each role."""


class RoleInventory:
    """Stores required, grouped, alternative, and capability roles for a kind.

    Parameters
    ----------
    required : Iterable[Role]
        The roles every component of the kind publishes.
    structures : Mapping[str, Iterable[Role]]
        The groups of roles that come together, keyed by the name of the structure.
    alternatives : Mapping[str, Iterable[Role]]
        The named groups from which a component publishes exactly one role.
    capabilities : Mapping[str, Iterable[Iterable[Role]]]
        The named families whose published roles must equal one declared set or be
        absent.

    Raises
    ------
    TypeError
        If a role is not a role.
    ValueError
        If a group name is empty, a group is empty, or a capability has no nonempty
        declared set.
    """

    __slots__ = ("_alternatives", "_capabilities", "_required", "_structures")

    def __init__(
        self,
        required: Iterable[Role],
        structures: Mapping[str, Iterable[Role]] = MappingProxyType({}),
        alternatives: Mapping[str, Iterable[Role]] = MappingProxyType({}),
        capabilities: Mapping[str, Iterable[Iterable[Role]]] = MappingProxyType({}),
    ) -> None:
        self._required = _roles("a required role", required)
        self._structures = _groups("structure", structures)
        self._alternatives = _groups("alternative", alternatives)
        self._capabilities = _capability_sets(capabilities)

    @property
    def required(self) -> frozenset[Role]:
        """Return the roles every component of the kind publishes."""
        return self._required

    @property
    def structures(self) -> Mapping[str, frozenset[Role]]:
        """Return the groups of roles that come together, keyed by structure."""
        return self._structures

    @property
    def alternatives(self) -> Mapping[str, frozenset[Role]]:
        """Return the groups from which exactly one role is published."""
        return self._alternatives

    @property
    def capabilities(self) -> Mapping[str, frozenset[frozenset[Role]]]:
        """Return the named families of allowed capability sets."""
        return self._capabilities

    @property
    def roles(self) -> frozenset[Role]:
        """Return every role the inventory names, required and conditional alike."""
        return self._required.union(
            *self._structures.values(),
            *self._alternatives.values(),
            *(roles for sets in self._capabilities.values() for roles in sets),
        )

    @override
    def __repr__(self) -> str:
        """Return how many roles are required and which groups are named."""
        return (
            f"RoleInventory({len(self._required)} required, "
            f"structures={sorted(self._structures)}, "
            f"alternatives={sorted(self._alternatives)}, "
            f"capabilities={sorted(self._capabilities)})"
        )


def _groups(
    noun: str, given: Mapping[str, Iterable[Role]]
) -> Mapping[str, frozenset[Role]]:
    """Return validated named groups of roles.

    Parameters
    ----------
    noun : str
        The singular name for one kind of group.
    given : Mapping[str, Iterable[Role]]
        The group members, keyed by group name.

    Returns
    -------
    Mapping[str, frozenset[Role]]
        An immutable copy of the named groups.

    Raises
    ------
    TypeError
        If a member is not a role.
    ValueError
        If a group name is empty or a group names no role.
    """
    grouped: dict[str, frozenset[Role]] = {}

    for group, members in given.items():
        named = ensure_name(f"a {noun}'s name", group)
        roles = _roles(f"a role of the {named} {noun}", members)

        if not roles:
            msg = f"the {named} {noun} names no role"
            raise ValueError(msg)

        grouped[named] = roles

    return MappingProxyType(grouped)


def _capability_sets(
    given: Mapping[str, Iterable[Iterable[Role]]],
) -> Mapping[str, frozenset[frozenset[Role]]]:
    """Return validated named families of capability sets.

    Parameters
    ----------
    given : Mapping[str, Iterable[Iterable[Role]]]
        The allowed role sets, keyed by capability-family name.

    Returns
    -------
    Mapping[str, frozenset[frozenset[Role]]]
        An immutable copy of the named families.

    Raises
    ------
    TypeError
        If a member is not a role.
    ValueError
        If a family name is empty, a family declares no set, or a set is empty.
    """
    grouped: dict[str, frozenset[frozenset[Role]]] = {}

    for group, alternatives in given.items():
        named = ensure_name("a capability family's name", group)
        sets = frozenset(
            _roles(f"a role of the {named} capability", roles) for roles in alternatives
        )

        if not sets:
            msg = f"the {named} capability family declares no set"
            raise ValueError(msg)

        if any(not roles for roles in sets):
            msg = f"the {named} capability family declares an empty set"
            raise ValueError(msg)

        grouped[named] = sets

    return MappingProxyType(grouped)


def _roles(what: str, given: Iterable[Role]) -> frozenset[Role]:
    """Return a set of roles.

    Parameters
    ----------
    what : str
        What the roles are for, for the message.
    given : Iterable[Role]
        The roles to check.

    Returns
    -------
    frozenset[Role]
        Them, as a set.

    Raises
    ------
    TypeError
        If a value is not a role.
    """
    collected = frozenset(given)

    for role in collected:
        if not isinstance(role, Role):
            msg = f"{what} is a Role, not {role!r}"
            raise TypeError(msg)

    return collected


_ATTENTION: Final[str] = "attention"
"""The published name of an attention structure."""

_ATTENDING: Final = (Role.ATTENTION_LOGITS, Role.ATTENTION_WEIGHTS)
"""The pre- and post-attention score roles."""

_DECODER_STATES: Final = (
    Role.DECODER_RECURRENT_STATE,
    Role.UNBOUND_VECTOR,
    Role.DECODER_RESIDUAL_STREAM,
)
"""The mutually exclusive state representations a term decoder may expose."""

INVENTORY: Final[Mapping[ComponentKind, RoleInventory]] = MappingProxyType(
    {
        ComponentKind.TOKEN_ENCODER: RoleInventory((Role.TOKEN_EMBEDDING,)),
        ComponentKind.SUBWORD_POOLER: RoleInventory((Role.WORD_VECTOR,)),
        ComponentKind.TERM_ENCODER: RoleInventory(
            (Role.TERM_ENCODING, Role.NODE_ENCODING),
            {
                _ATTENTION: _ATTENDING,
                "residual": (Role.RESIDUAL_STREAM, Role.FFN_PREACTIVATION),
            },
            capabilities={
                "recurrence": (
                    (Role.RECURRENT_HIDDEN,),
                    (Role.RECURRENT_HIDDEN, Role.GATE),
                    (Role.RECURRENT_HIDDEN, Role.RECURRENT_CELL, Role.GATE),
                )
            },
        ),
        ComponentKind.TERM_DECODER: RoleInventory(
            (Role.SYMBOL_DISTRIBUTION,),
            {_ATTENTION: _ATTENDING},
            {"state": _DECODER_STATES},
        ),
        ComponentKind.CONTROLLER: RoleInventory((Role.COMBINATOR_LOGITS,)),
        ComponentKind.PARSER: RoleInventory(
            (Role.CHART_CELL,),
            {
                _ATTENTION: _ATTENDING,
                "deep": (
                    Role.INSIDE_SCORE,
                    Role.OUTSIDE_SCORE,
                    Role.SPLIT_DISTRIBUTION,
                ),
            },
            capabilities={
                "span-context": ((Role.OUTSIDE_CELL, Role.SPAN_REPRESENTATION),)
            },
        ),
    }
)
"""Published sites required from each component kind."""


def _named(roles: Iterable[Role]) -> str:
    """Return role names in one message, in ascending order.

    Parameters
    ----------
    roles : Iterable[Role]
        The roles to name.

    Returns
    -------
    str
        Their names, comma separated.
    """
    return ", ".join(sorted(role.name for role in roles))


def ensure_publishes(published: SiteMap) -> SiteMap:
    """Return a component's map.

    Parameters
    ----------
    published : SiteMap
        The map, including the component and kind that govern it.

    Returns
    -------
    SiteMap
        The validated map.

    Raises
    ------
    ValueError
        If the kind has no inventory or if required roles, structural completeness,
        role axes, or the numeric class of a role are invalid.
    """
    if published.kind is ComponentKind.CUSTOM:
        return published

    kind = published.kind
    component = published.component
    inventory = INVENTORY.get(kind)

    if inventory is None:
        msg = f"the {kind.value} component kind has no declared role inventory"
        raise ValueError(msg)

    roles = frozenset(site.role for site in published)

    missing = inventory.required - roles

    if missing:
        msg = (
            f"{component} is a {kind.value} and publishes no {_named(missing)}, "
            "so what can be probed of it is thinner than the kind allows"
        )
        raise ValueError(msg)

    for structure, group in inventory.structures.items():
        held = group & roles

        if held and held != group:
            msg = (
                f"{component} publishes {_named(held)} and not "
                f"{_named(group - held)}, so its {structure} is half exposed"
            )
            raise ValueError(msg)

    for alternative, choices in inventory.alternatives.items():
        held = choices & roles

        if len(held) != 1:
            if not held:
                msg = (
                    f"{component} publishes none of {_named(choices)}, so its "
                    f"{alternative} has no alternative"
                )
            else:
                msg = (
                    f"{component} publishes {_named(held)}, so its {alternative} "
                    f"has more than one alternative. Exactly one of "
                    f"{_named(choices)} is allowed"
                )

            raise ValueError(msg)

    for capability, choices in inventory.capabilities.items():
        named = frozenset().union(*choices)
        held = named & roles

        if held and held not in choices:
            allowed = "; ".join(_named(choice) for choice in sorted(choices, key=len))
            msg = (
                f"{component} publishes {_named(held)}, which is not a declared "
                f"{capability} capability. Allowed sets are {allowed}"
            )
            raise ValueError(msg)

    for site, spec in published.items():
        declared = ROLE_AXES.get(site.role)
        meaning = ROLE_DTYPE.get(site.role)

        if meaning is not None and spec.dtype_class is not meaning:
            msg = (
                f"{component} publishes {site!r} as {spec.dtype_class.value}, and a "
                f"point of role {site.role.name} holds {meaning.value}"
            )
            raise ValueError(msg)

        if declared is not None and spec.axes != declared:
            msg = (
                f"{component} publishes {site!r} over {spec.axes}, and a point "
                f"of role {site.role.name} runs over {declared}"
            )
            raise ValueError(msg)

    return published


class SiteMapBuilder:
    """Declares one component's published points by role rather than by axis.

    Parameters
    ----------
    component : ComponentId
        The component every point belongs to.
    kind : ComponentKind
        The shared role inventory the component claims.

    Raises
    ------
    ValueError
        If the component is empty.
    """

    __slots__ = ("_component", "_kind", "_published")

    def __init__(self, component: ComponentId, kind: ComponentKind) -> None:
        if kind is ComponentKind.CUSTOM:
            msg = "a custom builder is constructed through SiteMapBuilder.custom"
            raise ValueError(msg)

        if not isinstance(kind, ComponentKind):
            msg = f"a builder's kind is not a ComponentKind: {kind!r}"
            raise TypeError(msg)

        self._component = ComponentId(ensure_name("a builder's component", component))
        self._kind = kind
        self._published: dict[Site, SiteSpec] = {}

    @classmethod
    def custom(cls, component: ComponentId) -> Self:
        """Return a builder for a component-specific map.

        Parameters
        ----------
        component : ComponentId
            The component every point belongs to.

        Returns
        -------
        Self
            The empty builder.
        """
        built = object.__new__(cls)
        object.__setattr__(
            built,
            "_component",
            ComponentId(ensure_name("a builder's component", component)),
        )
        object.__setattr__(built, "_kind", ComponentKind.CUSTOM)
        object.__setattr__(built, "_published", {})

        return built

    @property
    def component(self) -> ComponentId:
        """Return the component every point belongs to."""
        return self._component

    @property
    def kind(self) -> ComponentKind:
        """Return the component kind whose role inventory applies."""
        return self._kind

    def publish(
        self,
        role: Role,
        key: tuple[int | str, ...] = (),
        *,
        patchable: bool = True,
        differentiable: bool = True,
    ) -> Self:
        """Record one point of a role, over the axes that role declares.

        Parameters
        ----------
        role : Role
            What the value at the point plays in the computation.
        key : tuple[int | str, ...]
            Which point among those published for the role.
        patchable : bool
            Whether the point may be written to as well as read.
        differentiable : bool
            Whether a gradient flows through the value there.

        Returns
        -------
        Self
            This builder.

        Raises
        ------
        KeyError
            If the role has no declared axes, so its points are published with
            ``publish_at``.
        ValueError
            If the point is already published.
        """
        if role not in ROLE_AXES:
            msg = (
                f"{role.name} declares no axes, so a point of it states its own "
                "specification with publish_at"
            )
            raise KeyError(msg)

        spec = SiteSpec(
            ROLE_AXES[role],
            ROLE_DTYPE[role],
            differentiable=differentiable,
            access=Access.OBSERVE | Access.PATCH if patchable else Access.OBSERVE,
        )

        return self.publish_at(role, spec, key=key)

    def publish_at(
        self, role: Role, spec: SiteSpec, *, key: tuple[int | str, ...] = ()
    ) -> Self:
        """Record one point whose specification the component states itself.

        Parameters
        ----------
        role : Role
            What the value at the point plays in the computation.
        spec : SiteSpec
            What is found there, which the component states because its role declares no
            axes.
        key : tuple[int | str, ...]
            Which point among those published for the role.

        Returns
        -------
        Self
            This builder.

        Raises
        ------
        TypeError
            If the specification is not one.
        ValueError
            If the point is already published.
        """
        if not isinstance(spec, SiteSpec):
            msg = f"what is published at a point is not a SiteSpec: {spec!r}"
            raise TypeError(msg)

        site = Site(self._component, role, key)

        if site in self._published:
            msg = f"{site!r} is already published by this builder"
            raise ValueError(msg)

        self._published[site] = spec

        return self

    def build(self) -> SiteMap:
        """Return the map built so far, held to its kind's inventory.

        Returns
        -------
        SiteMap
            The points published, in the order they were published.

        Raises
        ------
        ValueError
            If the map is thinner than its inventory.
        """
        if self._kind is ComponentKind.CUSTOM:
            return SiteMap.custom(self._component, self._published)

        return ensure_publishes(SiteMap(self._component, self._kind, self._published))

    @override
    def __repr__(self) -> str:
        """Return the component and how many points it has published so far."""
        return (
            f"SiteMapBuilder({self._component!r}, {self._kind!r}, "
            f"{len(self._published)} sites)"
        )
