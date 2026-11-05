"""Model instrumentation sites and metadata."""

from collections.abc import Iterator, Mapping
from enum import Enum, Flag, auto
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, override

from lofi._checks import ensure_name, refuse_reassignment
from lofi._frozen import FrozenValue
from lofi._types import ComponentId

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = [
    "Access",
    "ComponentId",
    "ComponentKind",
    "DTypeClass",
    "ReceiverSide",
    "Role",
    "Site",
    "SiteMap",
    "SiteSpec",
]


class _CustomPath:
    """Permission used only by the explicit custom site-map constructor."""

    __slots__ = ()


_CUSTOM_PATH = _CustomPath()


class ComponentKind(Enum):
    """Enumerates neural component kinds whose sites share one role inventory."""

    @staticmethod
    @override
    def _generate_next_value_(
        name: str, start: int, count: int, last_values: list[str]
    ) -> str:
        """Return the wire spelling of a component-kind member."""
        del start, count, last_values

        return name.lower().replace("_", "-")

    #: A component-specific view with no shared role inventory.
    CUSTOM = auto()

    #: Encodes input tokens as contextual rows.
    TOKEN_ENCODER = auto()

    #: Pools contextual rows into word vectors.
    SUBWORD_POOLER = auto()

    #: Encodes a term as a vector.
    TERM_ENCODER = auto()

    #: Scores or generates term structures.
    TERM_DECODER = auto()

    #: Selects a combinator for chart cells.
    CONTROLLER = auto()

    #: Computes chart values.
    PARSER = auto()


class Role(Enum):
    """Identifies a value's role in a computation."""

    #: The rows a token encoder produces for the input.
    TOKEN_EMBEDDING = auto()

    #: The vectors a subword pooler produces for input words.
    WORD_VECTOR = auto()

    #: The vector a term encoder produces for a term.
    TERM_ENCODING = auto()

    #: The vector a term encoder produces for one term node or slot.
    NODE_ENCODING = auto()

    #: The distribution a term decoder produces over symbols at one slot.
    SYMBOL_DISTRIBUTION = auto()

    #: The state a recurrent term decoder carries between slots.
    DECODER_RECURRENT_STATE = auto()

    #: The vector a term decoder has not yet bound to a symbol.
    UNBOUND_VECTOR = auto()

    #: The stream a residual term decoder reads from and writes back to.
    DECODER_RESIDUAL_STREAM = auto()

    #: The hidden state a recurrent cell passes between steps.
    RECURRENT_HIDDEN = auto()

    #: The memory a recurrent cell passes between steps, where it has one.
    RECURRENT_CELL = auto()

    #: A recurrent gate or bounded candidate activation.
    GATE = auto()

    #: Attention scores before normalization.
    ATTENTION_LOGITS = auto()

    #: Attention scores after normalization.
    ATTENTION_WEIGHTS = auto()

    #: The stream a residual block reads from and writes back to.
    RESIDUAL_STREAM = auto()

    #: A feed-forward block's activations before its non-linearity.
    FFN_PREACTIVATION = auto()

    #: The vector a chart holds for one span.
    CHART_CELL = auto()

    #: The vector a chart computes from one span's surrounding contexts.
    OUTSIDE_CELL = auto()

    #: The concatenated inside and outside representation of one span.
    SPAN_REPRESENTATION = auto()

    #: A chart-cell activation where one receiver consumes it.
    RECEIVER_EDGE = auto()

    #: The log inside weight a chart holds for one span.
    INSIDE_SCORE = auto()

    #: The distribution over the ways one span splits.
    SPLIT_DISTRIBUTION = auto()

    #: The log outside weight a chart holds for one span.
    #: It is one number per span, the outside counterpart of ``INSIDE_SCORE``, and
    #: not a vector like ``CHART_CELL``.
    OUTSIDE_SCORE = auto()

    #: The probability that one span is a constituent of the item.
    CONSTITUENT_MARGINAL = auto()

    #: The log partition over every derivation of one item.
    LOG_PARTITION = auto()

    #: The log weight of the identity action at one split of one span.
    IDENTITY_SCORE = auto()

    #: Scores over the combinators a controller chooses between.
    COMBINATOR_LOGITS = auto()

    #: A weight the component learned, as opposed to an activation.
    PARAMETER = auto()


class Access(Flag):
    """Specifies permitted access to a site."""

    #: The value at the point may be read.
    OBSERVE = auto()

    #: The value at the point may be replaced for the rest of the pass.
    PATCH = auto()


class ReceiverSide(Enum):
    """Identifies which input of a binary chart receiver an edge enters."""

    #: The receiver's left input.
    LEFT = 0

    #: The receiver's right input.
    RIGHT = 1


def _checked_part(part: int | str) -> int | str:
    """Return one key part.

    Parameters
    ----------
    part : int | str
        The part to check.

    Returns
    -------
    int | str
        The part, unchanged.

    Raises
    ------
    TypeError
        If the part is neither an integer nor a string, or is a bool, which reaches a
        key only by mistake.
    ValueError
        If the part is a negative index.
    """
    if isinstance(part, bool) or not isinstance(part, int | str):
        msg = f"a key part is an index or a name, not {part!r}"
        raise TypeError(msg)

    if isinstance(part, int):
        if part < 0:
            msg = f"a key part counts from the start, so it is not {part}"
            raise ValueError(msg)

        return part

    if not part:
        msg = "a key part is empty"
        raise ValueError(msg)

    return part


class Site:
    """Identifies a published component value by role and key.

    Parameters
    ----------
    component : ComponentId
        The component the point belongs to.
    role : Role
        What the value at the point plays in the computation.
    key : tuple[int | str, ...]
        Which point among those the component publishes for that role: a layer index, a
        span, an action's name.

    Raises
    ------
    TypeError
        If the role is not a role, or a key part is neither an integer nor a string.
    ValueError
        If the component is empty, or a key part is negative or an empty string.
    """

    __slots__ = ("_component", "_key", "_role")

    def __init__(
        self,
        component: ComponentId,
        role: Role,
        key: tuple[int | str, ...] = (),
    ) -> None:
        if not component:
            msg = "a site names the component it belongs to"
            raise ValueError(msg)

        if not isinstance(role, Role):
            msg = f"a site's role is a Role, not {role!r}"
            raise TypeError(msg)

        self._component = component
        self._role = role
        self._key = tuple(_checked_part(part) for part in key)

    @property
    def component(self) -> ComponentId:
        """Return the component the point belongs to."""
        return self._component

    @property
    def role(self) -> Role:
        """Return the computational role of the site value."""
        return self._role

    @property
    def key(self) -> tuple[int | str, ...]:
        """Return the key distinguishing sites with the same role."""
        return self._key

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a site with the same three parts.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not Site:
            return NotImplemented

        return (
            self._component == other._component
            and self._role is other._role
            and self._key == other._key
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the component, the role and the key."""
        return hash((Site, self._component, self._role, self._key))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this site."""
        return f"Site({self._component!r}, {self._role}, {self._key!r})"


class DTypeClass(Enum):
    """Classifies numeric values independently of storage dtype."""

    #: A number with no declared range: a logit, a weight, a hidden state.
    REAL = "real"

    #: The logarithm of a probability, at or below zero.
    LOG_PROBABILITY = "log-probability"

    #: A probability, from zero to one.
    PROBABILITY = "probability"

    #: A whole number: a count, an index, a label.
    INTEGER = "integer"

    #: A truth value distinguishing data from padding.
    MASK = "mask"


class SiteSpec:
    """Describes a site's axes, numeric class, gradient behavior, and access.

    Parameters
    ----------
    axes : Iterable[str]
        The name of each axis, outermost first.
    dtype_class : DTypeClass
        What the numbers there mean.
    differentiable : bool
        Whether a gradient flows through the value at the point.
    access : Access
        What the point permits, as one or more members of ``Access``.

    Raises
    ------
    TypeError
        If the dtype class, the differentiability flag or the access set is not of its
        declared type, or if an axis name is not a string.
    ValueError
        If an axis name is empty, if two axes share a name, or if the access set permits
        nothing.
    """

    __slots__ = ("_access", "_axes", "_differentiable", "_dtype_class")

    def __init__(
        self,
        axes: Iterable[str],
        dtype_class: DTypeClass,
        *,
        differentiable: bool,
        access: Access,
    ) -> None:
        if not isinstance(dtype_class, DTypeClass):
            msg = f"a site specification's dtype class is not one: {dtype_class!r}"
            raise TypeError(msg)

        if not isinstance(differentiable, bool):
            msg = (
                "a site specification's differentiability is not a boolean: "
                f"{differentiable!r}"
            )
            raise TypeError(msg)

        if not isinstance(access, Access):
            msg = f"a site specification's access is not an Access: {access!r}"
            raise TypeError(msg)

        named = tuple(
            ensure_name(f"axis {position} of a site specification", name)
            for position, name in enumerate(axes)
        )

        seen: set[str] = set()

        for name in named:
            if name in seen:
                msg = f"a site specification declares the axis {name!r} twice"
                raise ValueError(msg)

            seen.add(name)

        if not access:
            msg = "a site specification permits neither observation nor patching"
            raise ValueError(msg)

        self._axes = named
        self._dtype_class = dtype_class
        self._differentiable = differentiable
        self._access = access

    @property
    def axes(self) -> tuple[str, ...]:
        """Return the name of each axis, outermost first."""
        return self._axes

    @property
    def rank(self) -> int:
        """Return the number of tensor axes."""
        return len(self._axes)

    @property
    def dtype_class(self) -> DTypeClass:
        """Return the numeric class of the site value."""
        return self._dtype_class

    @property
    def differentiable(self) -> bool:
        """Return whether a gradient flows through the value at the point."""
        return self._differentiable

    @property
    def access(self) -> Access:
        """Return the permitted site access."""
        return self._access

    def position(self, axis: str) -> int:
        """Return the position of a named axis.

        Parameters
        ----------
        axis : str
            The axis name to look up.

        Returns
        -------
        int
            Its position, outermost first.

        Raises
        ------
        KeyError
            If the specification declares no axis of that name.
        """
        for position, name in enumerate(self._axes):
            if name == axis:
                return position

        msg = f"no axis {axis!r} is declared here, only {self._axes}"
        raise KeyError(msg)

    def _identity(self) -> tuple[tuple[str, ...], DTypeClass, bool, Access]:
        """Return the four fields a specification is equal on."""
        return (self._axes, self._dtype_class, self._differentiable, self._access)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value declares the same four fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not SiteSpec:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields the specification is equal on."""
        return hash((SiteSpec, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this specification."""
        return (
            f"SiteSpec({self._axes!r}, {self._dtype_class!r}, "
            f"differentiable={self._differentiable!r}, access={self._access!r})"
        )


class SiteMap(FrozenValue, Mapping[Site, SiteSpec]):
    """An immutable map of the sites published by one component.

    Parameters
    ----------
    component : ComponentId
        The component every point belongs to.
    kind : ComponentKind
        The shared role inventory the component claims.
    sites : Mapping[Site, SiteSpec]
        The points the component publishes.

    Raises
    ------
    TypeError
        If the kind, a point, or its specification has the wrong value type.
    ValueError
        If the component is empty, the custom kind bypasses the explicit custom
        constructor, or a point belongs to another component.
    """

    __slots__ = ("_component", "_kind", "_sites")

    _noun = "site map"

    def __init__(
        self,
        component: ComponentId,
        kind: ComponentKind,
        sites: Mapping[Site, SiteSpec],
        *,
        _custom: _CustomPath | None = None,
    ) -> None:
        if kind is ComponentKind.CUSTOM and _custom is not _CUSTOM_PATH:
            msg = "a custom site map is constructed through SiteMap.custom"
            raise ValueError(msg)

        if _custom is not None and kind is not ComponentKind.CUSTOM:
            msg = "only a custom site map takes the custom construction path"
            raise ValueError(msg)

        self._initialize(component, kind, sites)

    def _initialize(
        self,
        component: ComponentId,
        kind: ComponentKind,
        sites: Mapping[Site, SiteSpec],
    ) -> None:
        try:
            object.__getattribute__(self, "_sites")

        except AttributeError:
            pass

        else:
            refuse_reassignment(self._noun)

        named = ComponentId(ensure_name("a site map's component", component))

        if not isinstance(kind, ComponentKind):
            msg = f"a site map's kind is not a ComponentKind: {kind!r}"
            raise TypeError(msg)

        if not isinstance(sites, Mapping):
            msg = f"a site map's sites are a Mapping, not {sites!r}"
            raise TypeError(msg)

        copied: dict[Site, SiteSpec] = {}

        for site, spec in sites.items():
            if not isinstance(site, Site):
                msg = f"a site map key is not a Site: {site!r}"
                raise TypeError(msg)

            if not isinstance(spec, SiteSpec):
                msg = f"what is published at {site!r} is not a SiteSpec: {spec!r}"
                raise TypeError(msg)

            if site.component != named:
                msg = (
                    f"{named} cannot publish {site!r}, which belongs to "
                    f"{site.component}"
                )
                raise ValueError(msg)

            copied[site] = spec

        object.__setattr__(self, "_component", named)
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_sites", MappingProxyType(copied))

    @classmethod
    def custom(cls, component: ComponentId, sites: Mapping[Site, SiteSpec]) -> Self:
        """Return an explicitly component-specific site map.

        Parameters
        ----------
        component : ComponentId
            The component every point belongs to.
        sites : Mapping[Site, SiteSpec]
            The points the component publishes without a shared role inventory.

        Returns
        -------
        Self
            The immutable custom map.
        """
        return cls(component, ComponentKind.CUSTOM, sites, _custom=_CUSTOM_PATH)

    @property
    def component(self) -> ComponentId:
        """Return the component every point belongs to."""
        return self._component

    @property
    def kind(self) -> ComponentKind:
        """Return the component kind whose role inventory applies."""
        return self._kind

    @override
    def __getitem__(self, site: Site) -> SiteSpec:
        """Return what the component publishes at one point."""
        return self._sites[site]

    @override
    def __iter__(self) -> Iterator[Site]:
        """Return an iterator over published points in declaration order."""
        return iter(self._sites)

    @override
    def __len__(self) -> int:
        """Return the number of published points."""
        return len(self._sites)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another site map has the same tag and points."""
        if type(other) is not SiteMap:
            return NotImplemented

        return (
            self._component == other._component
            and self._kind is other._kind
            and self._sites == other._sites
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the component, kind, and published points."""
        return hash(
            (SiteMap, self._component, self._kind, frozenset(self._sites.items()))
        )

    @override
    def __repr__(self) -> str:
        """Return the component, kind, and number of published points."""
        return f"SiteMap({self._component!r}, {self._kind!r}, {len(self._sites)} sites)"
