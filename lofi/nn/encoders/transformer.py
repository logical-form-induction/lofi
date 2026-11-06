"""Transformer encoding of linearized terms under a structural declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self, assert_never, final, override

import torch
from torch import Tensor, nn

from lofi.nn._layers import activation
from lofi.nn._values import Encoded
from lofi.nn.policy import DevicePolicy, place
from lofi.nn.ports import (
    Consumption,
    RelationKind,
    StructuralRelation,
)
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.nn.symbols import SymbolTable
from lofi.nn.transformer import (
    BLOCK_INPUT,
    BLOCK_OUTPUT,
    BLOCK_POST_ATTENTION,
    AttentionBlock,
    transformer_structure,
)
from lofi.registry import ComponentPlugin
from lofi.spec.encoders.transformer import (
    TransformerTermEncoderSettings,
    TransformerTermEncoderSpec,
)
from lofi.terms import BatchEncoding, Encoding

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from lofi._types import SharedSubmodule
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import DeclaredStructure, NeuralCodecView, PlacedEncoding
    from lofi.nn.ports import Patch, StructuralDeclaration
    from lofi.nn.sites import ComponentId

__all__ = [
    "TransformerEncoderSettings",
    "TransformerTermEncoder",
]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of a term encoding."""

_NODE_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of a node encoding."""

_ATTENTION_AXES: Final[tuple[str, ...]] = (
    "batch",
    "position",
    "attended-position",
)
"""The axes of one head's attention scores."""

_ACTIVATION_AXES: Final[tuple[str, ...]] = ("batch", "position", "feature")
"""The axes of an internal stream or feedforward activation."""


class TransformerEncoderSettings(NamedTuple):
    """Transformer term-encoder dimensions.

    Attributes
    ----------
    width : int
        The model and output width.
    heads : int
        The number of attention heads.
    layers : int
        The number of Transformer layers.
    feedforward_width : int
        The hidden width of each feedforward block.
    norm_cap : float | None
        The largest norm a symbol vector is read at, or None to read symbol vectors
        as stored.
    activation : str
        The installed ``lofi.activation`` entry each feedforward block applies.
    """

    width: int
    heads: int
    layers: int
    feedforward_width: int
    norm_cap: float | None
    activation: str


def _encoder_sites(component: ComponentId, *, layers: int, heads: int) -> SiteMap:
    """Return all sites published by an encoder.

    Parameters
    ----------
    component : ComponentId
        The component identifier assigned by the run.
    layers : int
        The number of Transformer layers.
    heads : int
        The number of attention heads in each layer.
    """
    published = {
        Site(component, Role.TERM_ENCODING): SiteSpec(
            _TERM_AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
        Site(component, Role.NODE_ENCODING): SiteSpec(
            _NODE_AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
    }

    for layer in range(layers):
        published[Site(component, Role.RESIDUAL_STREAM, (layer, BLOCK_INPUT))] = (
            SiteSpec(
                _ACTIVATION_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
        )

        for head in range(heads):
            published[Site(component, Role.ATTENTION_LOGITS, (layer, head))] = SiteSpec(
                _ATTENTION_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
            published[Site(component, Role.ATTENTION_WEIGHTS, (layer, head))] = (
                SiteSpec(
                    _ATTENTION_AXES,
                    DTypeClass.PROBABILITY,
                    differentiable=True,
                    access=Access.OBSERVE,
                )
            )

        published[
            Site(component, Role.RESIDUAL_STREAM, (layer, BLOCK_POST_ATTENTION))
        ] = SiteSpec(
            _ACTIVATION_AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        )
        published[Site(component, Role.FFN_PREACTIVATION, (layer,))] = SiteSpec(
            _ACTIVATION_AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        )
        published[Site(component, Role.RESIDUAL_STREAM, (layer, BLOCK_OUTPUT))] = (
            SiteSpec(
                _ACTIVATION_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
        )

    return SiteMap(
        component,
        ComponentKind.TERM_ENCODER,
        published,
    )


class _Reading(NamedTuple):
    """One tensor observed at a Transformer site."""

    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class _InstrumentScope:
    """One nestable observation region over selected Transformer sites."""

    def __init__(
        self,
        sites: frozenset[Site],
        enter: Callable[[_InstrumentScope], None],
        leave: Callable[[_InstrumentScope], None],
    ) -> None:
        self._sites = sites
        self._enter = enter
        self._leave = leave
        self._readings: dict[Site, _Reading] = {}
        self._entered = False
        self._closed = False

    @property
    def readings(self) -> Mapping[Site, _Reading]:
        """Return the latest reading taken at every selected site."""
        return MappingProxyType(self._readings)

    def record(self, reading: _Reading) -> None:
        """Keep a reading when this open scope selected its site."""
        if not self._closed and reading.site in self._sites:
            self._readings[reading.site] = reading

    def __enter__(self) -> Self:
        """Open this observation region."""
        if self._entered:
            msg = "a Transformer instrumentation scope cannot be entered twice"
            raise RuntimeError(msg)

        self._entered = True
        self._enter(self)

        return self

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Close this observation region."""
        del kind, error, traceback
        self._closed = True
        self._leave(self)


def _read(
    served: DeclaredStructure, relation: StructuralRelation, *, policy: DevicePolicy
) -> Tensor:
    """Return one relation's tensor for the layout.

    Parameters
    ----------
    served : DeclaredStructure
        The relations the encoder declared, read from its codec.
    relation : StructuralRelation
        The relation to read.
    policy : DevicePolicy
        Where this run's tensors live.

    Returns
    -------
    Tensor
        One value per slot for a unary relation, and one value per ordered pair of
        slots for a pairwise relation.
    """
    accessors = {
        StructuralRelation.POSITION: served.position,
        StructuralRelation.DEPTH: served.depth,
        StructuralRelation.ARITY: served.arity,
        StructuralRelation.PARENT: served.parent,
        StructuralRelation.ANCESTOR: served.ancestor,
        StructuralRelation.SIBLING: served.sibling,
        StructuralRelation.TREE_DISTANCE: served.tree_distance,
        StructuralRelation.SAME_CONSTITUENT: served.same_constituent,
        StructuralRelation.SPAN_CONTAINS: served.span_contains,
    }

    return accessors[relation](policy=policy)


class _TreeShape(NamedTuple):
    """The tree the ancestor relation of a layout describes, one value per slot.

    Attributes
    ----------
    parents : Tensor
        The slot each slot descends from, with the root sentinel at a root.
    sides : Tensor
        The rank of each slot, in slot order, among the slots sharing its parent.
    depths : Tensor
        The number of edges between each slot and its root.
    """

    parents: Tensor
    sides: Tensor
    depths: Tensor


def _tree_shape(ancestor: Tensor) -> _TreeShape:
    """Return the tree an ancestor relation describes.

    Parameters
    ----------
    ancestor : Tensor
        True where the read slot is the reading slot or one it descends from, of
        rank two over the reading slot and the read slot, in boolean dtype.

    Returns
    -------
    _TreeShape
        The parent, the rank among siblings and the depth of every slot, each of
        rank one in ``int64``.
    """
    slots = ancestor.shape[0]
    eye = torch.eye(slots, dtype=torch.bool, device=ancestor.device)
    above = ancestor & ~eye
    depths = above.sum(dim=1)
    nearest = torch.where(above, depths.unsqueeze(0), -1)
    parents = torch.where(
        depths.gt(0), nearest.argmax(dim=1), torch.full_like(depths, Encoding.NO_PARENT)
    )
    same_parent = parents.unsqueeze(1).eq(parents.unsqueeze(0)) & depths.gt(0)
    earlier = (
        torch.arange(slots, device=ancestor.device)
        .unsqueeze(0)
        .lt(torch.arange(slots, device=ancestor.device).unsqueeze(1))
    )
    sides = (same_parent & earlier).sum(dim=1)

    return _TreeShape(parents, sides, depths)


class _Structure(NamedTuple):
    """What a declaration builds a Transformer encoder from.

    Attributes
    ----------
    embedded : tuple[Tensor, ...]
        The unary relations added to a slot's input, each as its value at each slot.
    paths : _TreeShape | None
        The tree whose paths from the root are embedded, or None where the ancestor
        relation is not read as an embedding.
    biased : tuple[tuple[Tensor, int], ...]
        The pairwise relations biasing attention, each as the table row to read at
        each pair of slots and the number of rows its table holds.
    admitted : Tensor
        True where the reading slot attends to the read slot, true throughout when
        no mask is declared.
    """

    embedded: tuple[Tensor, ...]
    paths: _TreeShape | None
    biased: tuple[tuple[Tensor, int], ...]
    admitted: Tensor


def _between(value: Tensor) -> Tensor:
    """Return where a pairwise relation holds between two slots in either direction.

    Parameters
    ----------
    value : Tensor
        The relation's tensor, of rank two over the reading slot and the read slot,
        in boolean dtype.

    Returns
    -------
    Tensor
        True where the relation holds from either slot to the other, of the
        tensor's own shape and dtype.
    """
    return value | value.T


def _masked(relation: StructuralRelation, value: Tensor) -> Tensor:
    """Return the pairs a mask admits attention between, from a pairwise relation.

    Parameters
    ----------
    relation : StructuralRelation
        The relation read.
    value : Tensor
        Its tensor, of rank two over the reading slot and the read slot.

    Returns
    -------
    Tensor
        True where the relation holds between the two slots in either direction,
        of the tensor's own shape, in boolean dtype.

    Raises
    ------
    ValueError
        If the relation is one value per slot, or counts edges between two slots
        rather than holding between them or not.
    """
    if relation.kind is RelationKind.UNARY:
        msg = (
            f"a mask confines attention to the pairs a relation holds between, "
            f"and {relation.value} is one value per slot"
        )
        raise ValueError(msg)

    if value.dtype is not torch.bool:
        msg = (
            f"{relation.value} counts edges between two slots, and a mask needs a "
            f"relation that holds between them or not"
        )
        raise ValueError(msg)

    return _between(value)


def _biased(
    relation: StructuralRelation, value: Tensor, *, capacity: int
) -> tuple[Tensor, int]:
    """Return the bias table row each pair of slots reads, and the table's size.

    Parameters
    ----------
    relation : StructuralRelation
        The relation read.
    value : Tensor
        Its tensor: one value per slot for a unary relation, one per ordered pair
        of slots for a pairwise one.
    capacity : int
        How many slots the layout holds.

    Returns
    -------
    tuple[Tensor, int]
        The row to read at each pair, of rank two over the reading slot and the
        read slot in ``int64``, and how many rows the table holds. A unary relation
        is read by the difference of the two slots' values, a relation that holds
        or not by whether it holds between them in either direction, and a count
        by its value.
    """
    if relation.kind is RelationKind.UNARY:
        difference = value.unsqueeze(1) - value.unsqueeze(0)

        return difference + capacity - 1, 2 * capacity - 1

    if value.dtype is torch.bool:
        return _between(value).to(torch.int64), 2

    return value, capacity


def _structure(
    served: DeclaredStructure, *, capacity: int, policy: DevicePolicy
) -> _Structure:
    """Return what a declaration builds an encoder from.

    A mask admits attention between two slots in either direction of the relation,
    since a term is read at its root and its root descends from nothing, and a
    bias over a relation that holds or not reads it the same way, so a mask is the
    bias that is minus infinity off the relation. The ancestor relation read as an
    embedding embeds each slot's path from the root.

    Parameters
    ----------
    served : DeclaredStructure
        The relations the encoder declared, read from its codec.
    capacity : int
        How many slots the layout holds.
    policy : DevicePolicy
        Where this run's tensors live.

    Raises
    ------
    ValueError
        If a relation is declared as a recurrence, if a unary relation is declared
        as a mask, if a count is declared as a mask, or if a pairwise relation other
        than the ancestor is declared as an embedding.
    """
    embedded: list[Tensor] = []
    biased: list[tuple[Tensor, int]] = []
    paths: _TreeShape | None = None
    eye = torch.eye(capacity, dtype=torch.bool, device=policy.device)
    admitted = eye.clone()
    restricted = False

    for relation, consumption in served.declaration.items():
        value = _read(served, relation, policy=policy)

        match consumption:
            case Consumption.EMBEDDING if relation.kind is RelationKind.UNARY:
                embedded.append(value)
            case Consumption.EMBEDDING if relation is StructuralRelation.ANCESTOR:
                paths = _tree_shape(value)
            case Consumption.EMBEDDING:
                msg = (
                    f"an embedding is one vector per slot, and {relation.value} "
                    f"is one value per pair of slots"
                )
                raise ValueError(msg)
            case Consumption.MASK:
                restricted = True
                admitted |= _masked(relation, value)
            case Consumption.BIAS:
                biased.append(_biased(relation, value, capacity=capacity))
            case Consumption.RECURRENCE:
                msg = (
                    f"this encoder attends over every slot at once and runs no "
                    f"recurrence, so it cannot read {relation.value} as one"
                )
                raise ValueError(msg)
            case unreachable:
                assert_never(unreachable)

    return _Structure(
        tuple(embedded),
        paths,
        tuple(biased),
        admitted if restricted else torch.ones_like(eye),
    )


@final
class _TreePositionEncoding(nn.Module):
    """One vector per slot, built along its path from the root.

    A root reads a learned vector. Every other slot reads a bounded transform of
    its parent's vector plus a learned vector for its rank among its siblings.

    Parameters
    ----------
    width : int
        The position-vector width.
    shape : _TreeShape
        The parent, the rank among siblings and the depth of every slot.
    """

    _parents: Tensor
    _sides: Tensor
    _depths: Tensor

    def __init__(self, *, width: int, shape: _TreeShape) -> None:
        super().__init__()
        capacity = int(shape.parents.shape[0])
        self._root = nn.Parameter(torch.zeros(width))
        self._child = nn.Embedding(capacity, width)
        self._inherit = nn.Linear(width, width, bias=False)
        self.register_buffer("_parents", shape.parents.clamp_min(0))
        self.register_buffer("_sides", shape.sides)
        self.register_buffer("_depths", shape.depths)
        self._levels = int(shape.depths.max())

    @override
    def forward(self) -> Tensor:
        """Return the position of every slot, over the slot and the width."""
        capacity = int(self._parents.shape[0])
        positions = self._root.unsqueeze(0).expand(capacity, -1)
        offsets = self._child(self._sides)

        for depth in range(1, self._levels + 1):
            inherited = positions[self._parents]
            proposed = torch.tanh(self._inherit(inherited) + offsets)
            positions = torch.where(
                self._depths.eq(depth).unsqueeze(-1), proposed, positions
            )

        return positions


@final
class TransformerTermEncoder(nn.Module):
    """A Transformer over a term codec's slots, reading declared structure.

    Declared unary relations contribute embeddings, while pairwise relations provide
    attention masks or per-head biases. Every occupied slot attends to itself, and no
    slot attends to an unoccupied position.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    component : ComponentId
        The component identifier assigned by the run.
    settings : TransformerEncoderSettings
        The model dimensions and the symbol norm cap.
    structure : StructuralDeclaration
        The relations this encoder reads from its codec, each with how it uses it.

    Raises
    ------
    TypeError
        If the norm cap is not a number, or if a key of the structure is not a
        relation or a value not a consumption.
    ValueError
        If the norm cap is not finite or not above zero, the structure names a
        relation the codec does not serve, or it names a consumption this encoder
        cannot build from the relation.
    """

    _features: Tensor
    _bias_rows: Tensor
    _admitted: Tensor

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        component: ComponentId,
        settings: TransformerEncoderSettings,
        structure: StructuralDeclaration,
    ) -> None:
        super().__init__()
        served = codec.serve(structure)
        model_width = settings.width
        head_count = settings.heads
        layer_count = settings.layers
        inner_width = settings.feedforward_width
        self._symbols = SymbolTable(
            symbols=codec.n_symbols, width=model_width, cap=settings.norm_cap
        )
        self._blocks = nn.ModuleList(
            AttentionBlock(
                model_width, head_count, inner_width, activation(settings.activation)
            )
            for _ in range(layer_count)
        )
        self._norm = nn.LayerNorm(model_width)
        policy = DevicePolicy(self._symbols.weight.device, self._symbols.weight.dtype)
        built = _structure(served, capacity=codec.capacity, policy=policy)
        self._embeddings = nn.ModuleList(
            nn.Embedding(codec.capacity, model_width) for _ in built.embedded
        )
        self._paths = (
            None
            if built.paths is None
            else _TreePositionEncoding(width=model_width, shape=built.paths)
        )
        self._bias_tables = nn.ParameterList(
            nn.Parameter(torch.randn(head_count, rows, dtype=policy.dtype))
            for _, rows in built.biased
        )
        self.register_buffer(
            "_features",
            torch.stack(built.embedded)
            if built.embedded
            else torch.zeros(
                (0, codec.capacity), dtype=torch.int64, device=policy.device
            ),
        )
        self.register_buffer(
            "_bias_rows",
            torch.stack([rows for rows, _ in built.biased])
            if built.biased
            else torch.zeros(
                (0, codec.capacity, codec.capacity),
                dtype=torch.int64,
                device=policy.device,
            ),
        )
        self.register_buffer("_admitted", built.admitted)
        self._codec = codec
        self._structure = served.declaration
        self._sites = _encoder_sites(component, layers=layer_count, heads=head_count)
        self._component = component
        self._scopes: list[_InstrumentScope] = []
        self._capacity = codec.capacity
        self._width = model_width

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis."""
        return self._width

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view supplied at construction."""
        return self._codec

    @property
    def structure(self) -> StructuralDeclaration:
        """Return the relations this encoder reads from its codec, and how."""
        return self._structure

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the submodules another component may hold, which are none."""
        return MappingProxyType({})

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the symbol table."""
        return DevicePolicy(self._symbols.weight.device, self._symbols.weight.dtype)

    def _positions(self) -> Tensor:
        """Return the structural input added at every slot.

        Returns
        -------
        Tensor
            Rank 2 over the slot and the width, zero where the declaration embeds
            nothing.
        """
        positions = self._symbols.weight.new_zeros((self._capacity, self._width))

        for table, values in zip(self._embeddings, self._features, strict=True):
            positions = positions + table(values)

        if self._paths is not None:
            positions = positions + self._paths()

        return positions

    def _attention(self, occupied: Tensor) -> Tensor:
        """Return the term added to the attention weights of every layer.

        Parameters
        ----------
        occupied : Tensor
            True where a slot holds a node, of rank two over the term and the slot.

        Returns
        -------
        Tensor
            Rank 4 over the term, the head, the reading slot and the read slot, of
            size one over the head where no bias is declared, at minus infinity
            where the reading slot does not attend to the read slot or the read
            slot holds no node and is not the reading slot, and at the declared
            biases elsewhere.
        """
        terms, slots = occupied.shape
        eye = torch.eye(slots, dtype=torch.bool, device=occupied.device)
        readable = self._admitted.unsqueeze(0) & (occupied.unsqueeze(1) | eye)
        weight = self._symbols.weight
        zero = weight.new_zeros(())
        refused = weight.new_full((), -torch.inf)
        biased = torch.where(readable, zero, refused).reshape(terms, 1, slots, slots)

        for table, rows in zip(self._bias_tables, self._bias_rows, strict=True):
            biased = biased + table[:, rows].unsqueeze(0)

        return biased

    def _transform(self, symbols: Tensor, occupied: Tensor, /) -> Tensor:
        """Return contextual node vectors with unoccupied slots zeroed.

        Parameters
        ----------
        symbols : Tensor
            One symbol vector per term slot, over the term, slot and width axes.
        occupied : Tensor
            True where a slot holds a node, of rank two over the term and the slot.
        """
        stream = torch.where(
            occupied.unsqueeze(-1),
            symbols + self._positions().unsqueeze(0),
            torch.zeros_like(symbols),
        )

        if occupied.shape[0] == 0:
            return stream

        attention = self._attention(occupied)
        observe = self._observe if self._scopes else None

        for layer, block in enumerate(self._blocks):
            stream = block(
                stream,
                attention,
                occupied=occupied,
                layer=layer,
                observe=observe,
            )

        contextual = self._norm(stream)

        return torch.where(
            occupied.unsqueeze(-1), contextual, torch.zeros_like(contextual)
        )

    def _hard_nodes(self, terms: BatchEncoding) -> tuple[Tensor, PlacedEncoding]:
        """Return contextual nodes and their placed channels.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.
        """
        placed = self._codec.place_batch(terms, policy=self._policy())
        symbols = self._codec.symbol_rows(placed, self._symbols.rows)

        return self._transform(symbols.tensor, placed.mask), placed

    @staticmethod
    def _terms(
        nodes: Tensor, placed: PlacedEncoding, structure: BatchEncoding, /
    ) -> Encoded:
        """Return the contextual vector at each root.

        A term's root is the occupied slot its own parent channel marks with the
        root sentinel, wherever the layout put it.

        Parameters
        ----------
        nodes : Tensor
            Contextual vectors for every slot.
        placed : PlacedEncoding
            The aligned parent and occupancy channels.
        structure : BatchEncoding
            The source rows, read to decide whether a term mask is needed.
        """
        roots = placed.mask & placed.parents.eq(Encoding.NO_PARENT)
        selected = torch.where(roots.unsqueeze(-1), nodes, torch.zeros_like(nodes)).sum(
            dim=1
        )
        mask = (
            None
            if all(row.length > 0 for row in structure.rows)
            else placed.mask.any(dim=1)
        )

        return Encoded(selected, _TERM_AXES, mask)

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        """Return one contextual vector per term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            Contextual vectors over the term and width axes, masked only for rows
            that hold no term.
        """
        nodes, placed = self._hard_nodes(terms)

        if self._scopes:
            self._observe(Role.NODE_ENCODING, (), nodes, placed.mask)

        encoded = self._terms(nodes, placed, terms)

        if self._scopes:
            self._observe(Role.TERM_ENCODING, (), encoded.tensor, encoded.mask)

        return encoded

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        """Return one contextual vector per occupied term slot.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            Contextual vectors over the term, slot, and width axes, masked by
            occupancy and zero at unoccupied slots.
        """
        nodes, placed = self._hard_nodes(terms)

        if self._scopes:
            self._observe(Role.NODE_ENCODING, (), nodes, placed.mask)

        return Encoded(nodes, _NODE_AXES, placed.mask)

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[_Reading]],
    ) -> _InstrumentScope:
        """Return a nestable observation scope over published sites.

        Parameters
        ----------
        sites : frozenset[Site]
            The published points to observe.
        patches : Mapping[Site, Patch[_Reading]]
            Requested replacements, all of which this observation-only component
            refuses.

        Raises
        ------
        ValueError
            If a requested site is not published, or if any patch is requested.
        """
        for site in sites:
            if site not in self._sites:
                msg = f"the Transformer does not publish the observation site {site!r}"
                raise ValueError(msg)

        for site in patches:
            if site not in self._sites:
                msg = f"the Transformer does not publish the patch site {site!r}"
                raise ValueError(msg)

            msg = f"the Transformer site {site!r} is observation-only"
            raise ValueError(msg)

        return _InstrumentScope(sites, self._attach, self._detach)

    def _observe(
        self,
        role: Role,
        key: tuple[int | str, ...],
        tensor: Tensor,
        mask: Tensor | None,
        /,
    ) -> None:
        """Record a real forward value in every open scope selecting its site."""
        if not self._scopes:
            return

        site = Site(self._component, role, key)
        reading = _Reading(site, tensor.detach(), self._sites[site].axes, mask)

        for scope in self._scopes:
            scope.record(reading)

    def _attach(self, scope: _InstrumentScope) -> None:
        """Add one scope to the active nesting stack."""
        self._scopes.append(scope)

    def _detach(self, scope: _InstrumentScope) -> None:
        """Remove the innermost scope from the active nesting stack."""
        if not self._scopes or self._scopes[-1] is not scope:
            msg = "Transformer instrumentation scopes must close inside out"
            raise RuntimeError(msg)

        self._scopes.pop()

    def site_map(self) -> SiteMap:
        """Return every published output and internal observation site."""
        return self._sites


def _build(
    settings: TransformerTermEncoderSettings, context: BuildContext, /
) -> TransformerTermEncoder:
    """Build and place a Transformer term encoder over the run's codec.

    Parameters
    ----------
    settings : TransformerTermEncoderSettings
        The model dimensions, the declared structure and the symbol norm cap.
    context : BuildContext
        The codec, the component handle and the device policy.
    """
    return place(
        TransformerTermEncoder(
            codec=context.codec,
            component=context.component,
            settings=TransformerEncoderSettings(
                width=settings.width,
                heads=settings.heads,
                layers=settings.layers,
                feedforward_width=settings.feedforward_width,
                norm_cap=settings.norm_cap,
                activation=settings.activation,
            ),
            structure=transformer_structure(settings.structure),
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=TransformerTermEncoderSpec, build=_build)
"""The Transformer term encoder's plugin."""
