"""Recurrent encoding of linearized terms, bottom up over their operator trees."""

from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    Literal,
    NamedTuple,
    Self,
    final,
)

import torch
from torch import nn
from torch.nn import functional

from lofi.nn._layers import recurrent_step
from lofi.nn._values import Encoded
from lofi.nn.arity import ArityBound, ArityHeadScheme, ArityUnsupportedError
from lofi.nn.codec import NeuralCodecView, reflexive_closure
from lofi.nn.embeddings import ChoiceEmbedding
from lofi.nn.inventory import (
    EntangledInventory,
    InventoryTopology,
    LevelBatch,
)
from lofi.nn.policy import DevicePolicy, place
from lofi.nn.ports import ArityScheme, Consumption, StructuralRelation
from lofi.nn.sites import (
    Access,
    ComponentKind,
    DTypeClass,
    Role,
    Site,
    SiteMap,
    SiteSpec,
)
from lofi.registry import ComponentPlugin
from lofi.spec.encoders.rnn import TreeRnnEncoderSettings, TreeRnnEncoderSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi._types import SharedSubmodule
    from lofi.nn._layers import CellStep, Gate, RecurrentStep
    from lofi.nn._values import LogDistribution
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import PlacedEncoding
    from lofi.nn.ports import Patch, StructuralDeclaration
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import RecurrentCell
    from lofi.terms import BatchEncoding, ChoiceIndex

__all__ = ["TreeRnnEncoder"]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of a term encoding."""

_NODE_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of a node encoding."""

_RECURRENT_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of one recurrent activation."""

_STRUCTURE: Final[StructuralDeclaration] = MappingProxyType(
    {StructuralRelation.PARENT: Consumption.RECURRENCE}
)
"""The one relation the recurrence reads from its codec: the parent, as what each
slot's state is computed from."""


type InventoryPass = Literal["uncached", "shared-dag"]


def _encoder_sites(
    component: ComponentId,
    *,
    layers: int,
    gates: tuple[Gate, ...],
) -> SiteMap:
    """Return all sites published by an encoder.

    Parameters
    ----------
    component : ComponentId
        The component identifier assigned by the run.
    layers : int
        The number of recurrent layers.
    gates : tuple[Gate, ...]
        The activations each recurrent layer publishes.
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
        Site(component, Role.PARAMETER, ("choice-embedding",)): SiteSpec(
            ("choice", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
    }

    for layer in range(layers):
        published[Site(component, Role.RECURRENT_HIDDEN, (layer,))] = SiteSpec(
            _RECURRENT_AXES,
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        )
        for gate in gates:
            published[Site(component, Role.GATE, (layer, gate.name))] = SiteSpec(
                _RECURRENT_AXES,
                gate.dtype,
                differentiable=True,
                access=Access.OBSERVE,
            )

    return SiteMap(component, ComponentKind.TERM_ENCODER, published)


class _Reading(NamedTuple):
    """One tensor observed at a recurrent encoder site."""

    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class _InstrumentScope:
    """One nestable observation region over selected recurrent encoder sites."""

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
            msg = "a recurrent instrumentation scope cannot be entered twice"
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


class _Layout(NamedTuple):
    """The tree a layout's parent relation describes, one value per slot.

    Attributes
    ----------
    children : Tensor
        The child slot at each argument position of every slot, in slot order, over
        the slot and position axes, at zero beyond a slot's own children.
    has_child : Tensor
        True where a slot has a child at an argument position, of the same shape.
    depths : Tensor
        The number of edges between each slot and the root, over the slot axis.
    roots : Tensor
        True at a slot no slot is the parent of, over the slot axis.
    """

    children: Tensor
    has_child: Tensor
    depths: Tensor
    roots: Tensor


def _layout(parent: Tensor) -> _Layout:
    """Return the tree a parent relation describes.

    Parameters
    ----------
    parent : Tensor
        True where the read slot is the reading slot's parent, of rank two over the
        reading slot and the read slot, in boolean dtype.
    """
    slots = parent.shape[0]
    address = torch.arange(slots, device=parent.device)
    is_child = parent.T
    counts = is_child.sum(dim=1)
    branching = int(counts.max())
    position = is_child.cumsum(dim=1).sub(1)
    target = torch.where(is_child, position, torch.full_like(position, branching))
    table = torch.zeros((slots, branching + 1), dtype=torch.int64, device=parent.device)
    table.scatter_(1, target, address.unsqueeze(0).expand(slots, slots))
    has_child = (
        torch.arange(branching, device=parent.device)
        .unsqueeze(0)
        .lt(counts.unsqueeze(1))
    )
    depths = reflexive_closure(parent).sum(dim=1).sub(1)

    return _Layout(table[:, :branching], has_child, depths, ~parent.any(dim=1))


def _gathered(states: Tensor, slots: Tensor) -> Encoded:
    """Return the state held at one named slot of every term.

    Parameters
    ----------
    states : Tensor
        The state at every slot, over the term, slot, layer and width axes.
    slots : Tensor
        Which slot to read for each term and slot, over the term and slot axes.

    Returns
    -------
    Encoded
        The states read, over the term, slot, layer and width axes.
    """
    index = slots.reshape(*slots.shape, 1, 1).expand_as(states)

    return Encoded(torch.gather(states, 1, index), ("term", "slot", "layer", "width"))


@final
class TreeRnnEncoder(nn.Module):
    """A recurrent encoder running bottom up over a term codec's slots.

    Every leaf starts from one learned position-independent state; internal states
    combine their children's states by arity before reading the constructor code.
    Term and node outputs expose the top recurrent layer. A one-layer encoder whose
    cell carries no memory apart from its hidden state can continue a synthetic
    operator tree from a live term vector exactly; other recurrent configurations do
    not expose a complete continuation state.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    choices : ChoiceIndex
        The language's payload-aware alphabet.
    component : ComponentId
        The component identifier assigned by the run.
    scheme : ArityScheme
        The rank-indexed family merging a slot's children, whose declared width is the
        width this encoder runs and returns at.
    cell : RecurrentCell
        The installed recurrent cell each layer steps with.
    layers : int
        How many recurrent layers each slot holds.
    norm_cap : float | None
        The largest norm a constructor vector is read at, or None to read constructor
        vectors as stored.
    """

    _children: Tensor
    _choice_arities: Tensor
    _has_child: Tensor
    _depths: Tensor
    _roots: Tensor

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        scheme: ArityScheme,
        cell: RecurrentCell,
        layers: int,
        norm_cap: float | None,
        inventory: InventoryPass,
    ) -> None:
        super().__init__()
        served = codec.serve(_STRUCTURE)
        width = scheme.width
        depth = layers

        self._embedding = ChoiceEmbedding(
            index=choices, width=width, component=component
        )
        self._start = nn.Parameter(torch.empty(depth, width))
        nn.init.normal_(self._start)
        steps = tuple(
            recurrent_step(cell, inputs=width, width=width) for _ in range(depth)
        )
        self._cells = nn.ModuleList(steps)
        self._steps: tuple[RecurrentStep, ...] = steps
        self._gates = steps[0].gates
        self._gate_names = tuple(gate.name for gate in self._gates)
        self._carries_memory = steps[0].carries_memory
        policy = DevicePolicy(
            self._embedding.weight.device, self._embedding.weight.dtype
        )
        layout = _layout(served.parent(policy=policy))
        self.register_buffer("_children", layout.children)
        self.register_buffer(
            "_choice_arities",
            torch.tensor(
                (-1, *(choices.arity(choice) for choice in range(1, choices.width))),
                dtype=torch.int64,
                device=layout.children.device,
            ),
        )
        self.register_buffer("_has_child", layout.has_child)
        self.register_buffer("_depths", layout.depths)
        self.register_buffer("_roots", layout.roots)
        self._scheme = scheme
        self._codec = codec
        self._choices = choices
        self._norm_cap = norm_cap
        self._inventory = inventory
        self._structure = served.declaration
        self._sites = _encoder_sites(component, layers=depth, gates=self._gates)
        self._cell = cell
        self._component = component
        self._scopes: list[_InstrumentScope] = []
        self._layers = depth
        self._width = width

    @property
    def output_width(self) -> int:
        """Return the size of the feature axis, which the scheme declares."""
        return self._width

    @property
    def layers(self) -> int:
        """Return how many recurrent layers each slot holds."""
        return self._layers

    @property
    def cell(self) -> RecurrentCell:
        """Return the recurrent transition applied at every layer."""
        return self._cell

    @property
    def inventory(self) -> InventoryPass:
        """Return how repeated subterms are evaluated within one hard batch."""
        return self._inventory

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view supplied at construction."""
        return self._codec

    @property
    def choices(self) -> ChoiceIndex:
        """Return the language choice index read by the encoder."""
        return self._choices

    @property
    def scheme(self) -> ArityScheme:
        """Return the combination family supplied at construction."""
        return self._scheme

    @property
    def structure(self) -> StructuralDeclaration:
        """Return the relations this encoder reads from its codec, and how."""
        return self._structure

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return the submodules another component may hold, which are none."""
        return MappingProxyType({})

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the choice embedding."""
        return DevicePolicy(self._embedding.weight.device, self._embedding.weight.dtype)

    def _choice_rows(self) -> Tensor:
        """Return choice rows under the declared norm cap."""
        weight = self._embedding.weight

        if self._norm_cap is None:
            return weight

        norms = weight.norm(dim=1, keepdim=True)
        return weight * (self._norm_cap / norms.clamp_min(self._norm_cap))

    def _leaf_states(self, slots: int, terms: int) -> Tensor:
        """Return the shared state every slot with no child starts from.

        Parameters
        ----------
        slots : int
            How many slots the linearization holds.
        terms : int
            How many terms the batch holds.

        Returns
        -------
        Tensor
            The starting state, over the term, slot, layer and width axes.
        """
        return self._start.reshape(1, 1, self._layers, self._width).expand(
            terms, slots, self._layers, self._width
        )

    def _advance(
        self,
        previous: Tensor,
        memory: Tensor,
        inputs: Tensor,
    ) -> tuple[Tensor, Tensor, tuple[CellStep, ...]]:
        """Return the state and gates the cells reach from a start and an input.

        Parameters
        ----------
        previous : Tensor
            The hidden state each layer starts from, over the term, slot, layer and
            width axes.
        memory : Tensor
            The cell memory each layer starts from, over the same axes.
        inputs : Tensor
            The constructor vector each slot reads, over the term, slot and width axes.

        Returns
        -------
        tuple[Tensor, Tensor, tuple[CellStep, ...]]
            The hidden state and memory each layer reaches, over the term, slot,
            layer and width axes, and each layer's flattened state and activations.
            The lowest layer reads the constructor vector and every layer above it
            reads the layer below.
        """
        terms, slots = inputs.shape[0], inputs.shape[1]
        started = previous.reshape(terms * slots, self._layers, self._width)
        remembered = memory.reshape(terms * slots, self._layers, self._width)
        below = inputs.reshape(terms * slots, self._width)
        reached: list[CellStep] = []

        for layer, cell in enumerate(self._steps):
            step = cell(below, started[:, layer, :], remembered[:, layer, :])
            below = step.hidden
            reached.append(step)

        states = torch.stack([step.hidden for step in reached], dim=1).reshape(
            terms, slots, self._layers, self._width
        )
        memories = torch.stack([step.memory for step in reached], dim=1).reshape(
            terms, slots, self._layers, self._width
        )

        return states, memories, tuple(reached)

    def _child_slots(self, occupied: Tensor) -> tuple[Tensor, Tensor]:
        """Return the occupied children of every slot, and how many each slot has.

        A slot's children are the slots the layout hangs off it that a node occupies,
        and their argument order is the order the layout put them in.

        Parameters
        ----------
        occupied : Tensor
            True where a slot holds a node, of rank two over the term and the slot.

        Returns
        -------
        tuple[Tensor, Tensor]
            The child slot at each argument position, over the term, slot and position
            axes, and the child count of each slot, over the term and slot axes. A
            position beyond a slot's own child count reads zero.
        """
        terms, slots = occupied.shape
        branching = int(self._children.shape[1])
        children = self._children.unsqueeze(0).expand(terms, slots, branching)
        live = occupied[:, self._children] & self._has_child.unsqueeze(0)
        counts = live.sum(dim=2)
        position = live.cumsum(dim=2).sub(1)
        target = torch.where(live, position, torch.full_like(position, branching))
        table = torch.zeros(
            (terms, slots, branching + 1), dtype=torch.int64, device=occupied.device
        )
        table.scatter_(2, target, children)

        return table[:, :, :branching], counts

    def _states(
        self,
        symbols: Tensor,
        occupied: Tensor,
        *,
        injected: Tensor | None = None,
        injected_slots: tuple[int, ...] | None = None,
    ) -> Tensor:
        """Return the recurrent state at every slot of a batch.

        Parameters
        ----------
        symbols : Tensor
            The constructor vector at each slot, over the term, slot and width axes.
        occupied : Tensor
            True where a slot holds a node, of rank two over the term and the slot.

        Returns
        -------
        Tensor
            The state at every slot, over the term, slot, layer and width axes, at zero
            wherever no node occupies the slot.

        Raises
        ------
        ArityUnsupportedError
            If a slot has a child count the combination family serves no module at.
        """
        terms, slots = occupied.shape
        states = symbols.new_zeros((terms, slots, self._layers, self._width))
        memories = symbols.new_zeros((terms, slots, self._layers, self._width))
        fixed = torch.zeros_like(occupied)

        if injected is not None:
            if injected_slots is None or len(injected_slots) != terms:
                msg = "one injected synthetic-tree slot is required per term"
                raise ValueError(msg)

            expected = (terms, self._layers, self._width)

            if tuple(injected.shape) != expected:
                msg = (
                    "injected synthetic-tree states have shape "
                    f"{tuple(injected.shape)}, "
                    f"not {expected}"
                )
                raise ValueError(msg)

            rows = torch.arange(terms, device=occupied.device)
            held_slots = torch.tensor(
                injected_slots, dtype=torch.long, device=occupied.device
            )

            if bool(held_slots.lt(0).any()) or bool(held_slots.ge(slots).any()):
                msg = f"a synthetic-tree leaf slot lies outside [0, {slots})"
                raise ValueError(msg)

            if not bool(occupied[rows, held_slots].all()):
                msg = "a synthetic-tree leaf slot does not hold an operator"
                raise ValueError(msg)

            states[rows, held_slots] = injected
            memories[rows, held_slots] = injected
            fixed[rows, held_slots] = True
        gates = (
            {
                name: [
                    symbols.new_zeros((terms, slots, self._width))
                    for _ in range(self._layers)
                ]
                for name in self._gate_names
            }
            if self._scopes
            else None
        )

        if not bool(occupied.any()):
            self._publish_recurrence(states, gates, occupied)

            return states

        child_slots, counts = self._child_slots(occupied)
        leaves = self._leaf_states(slots, terms)
        depths = self._depths.unsqueeze(0)

        for depth in range(int(self._depths.max()), -1, -1):
            at_depth = occupied & depths.eq(depth)

            if not bool(at_depth.any()):
                continue

            for rank in sorted({int(each) for each in counts[at_depth].unique()}):
                previous = (
                    leaves
                    if rank == 0
                    else self._scheme.combine(
                        tuple(
                            _gathered(states, child_slots[:, :, position])
                            for position in range(rank)
                        )
                    ).tensor
                )
                memory = (
                    torch.zeros_like(leaves)
                    if rank == 0
                    else torch.stack(
                        tuple(
                            _gathered(memories, child_slots[:, :, position]).tensor
                            for position in range(rank)
                        )
                    ).sum(dim=0)
                )
                selected = at_depth & counts.eq(rank) & ~fixed
                advanced, remembered, steps = self._advance(previous, memory, symbols)
                states = torch.where(
                    selected.reshape(terms, slots, 1, 1),
                    advanced,
                    states,
                )
                memories = torch.where(
                    selected.reshape(terms, slots, 1, 1),
                    remembered,
                    memories,
                )

                if gates is not None:
                    selected_rows = selected.reshape(terms, slots, 1)

                    for layer, step in enumerate(steps):
                        for name, values in zip(
                            self._gate_names, step.gates, strict=True
                        ):
                            gates[name][layer] = torch.where(
                                selected_rows,
                                values.reshape(terms, slots, self._width),
                                gates[name][layer],
                            )

        self._publish_recurrence(states, gates, occupied)

        return states

    def _soft_states(self, probabilities: Tensor) -> Tensor:
        """Return mean-field states from payload-aware choice marginals.

        Each slot conditions its expected choice embedding on operator arity, then
        mixes the corresponding recurrent branches by their marginal mass. Padding
        contributes the zero state. Thus a one-hot distribution follows exactly the
        same branch as hard encoding, while a non-degenerate distribution never
        enumerates or selects a term.
        """
        terms, slots = probabilities.shape[:2]
        embeddings = self._choice_rows()
        states = probabilities.new_zeros((terms, slots, self._layers, self._width))
        memories = probabilities.new_zeros((terms, slots, self._layers, self._width))
        leaves = self._leaf_states(slots, terms)
        depths = self._depths.unsqueeze(0)
        arities = self._choice_arities
        served = (
            0,
            *(rank for rank in self._scheme.ranks if rank <= self._children.shape[1]),
        )
        unsupported = arities.ge(0) & ~torch.isin(arities, arities.new_tensor(served))

        unsupported_mass = probabilities[..., unsupported].gt(0).any(dim=(0, 1))

        if bool(unsupported_mass.any()):
            arity = int(arities[unsupported][unsupported_mass][0])
            msg = (
                f"soft encoding assigns positive mass to operator arity {arity}, "
                f"and the encoder serves {served!r}"
            )
            raise ArityUnsupportedError(msg)

        gates = (
            {
                name: [
                    probabilities.new_zeros((terms, slots, self._width))
                    for _ in range(self._layers)
                ]
                for name in self._gate_names
            }
            if self._scopes
            else None
        )

        for depth in range(int(self._depths.max()), -1, -1):
            at_depth = depths.eq(depth)
            mixed_states = torch.zeros_like(states)
            mixed_memories = torch.zeros_like(memories)

            for rank in served:
                at_rank = arities.eq(rank)
                weighted = probabilities[..., at_rank]
                mass = weighted.sum(dim=-1)
                symbol_total = weighted @ embeddings[at_rank]
                symbols = torch.where(
                    mass.unsqueeze(-1).gt(0),
                    symbol_total
                    / mass.unsqueeze(-1).clamp_min(torch.finfo(mass.dtype).tiny),
                    torch.zeros_like(symbol_total),
                )
                previous = (
                    leaves
                    if rank == 0
                    else self._scheme.combine(
                        tuple(
                            _gathered(
                                states,
                                self._children[:, position]
                                .unsqueeze(0)
                                .expand(terms, slots),
                            )
                            for position in range(rank)
                        )
                    ).tensor
                )
                memory = (
                    torch.zeros_like(leaves)
                    if rank == 0
                    else torch.stack(
                        tuple(
                            _gathered(
                                memories,
                                self._children[:, position]
                                .unsqueeze(0)
                                .expand(terms, slots),
                            ).tensor
                            for position in range(rank)
                        )
                    ).sum(dim=0)
                )
                advanced, remembered, steps = self._advance(previous, memory, symbols)
                weight = mass.reshape(terms, slots, 1, 1)
                mixed_states = mixed_states + weight * advanced
                mixed_memories = mixed_memories + weight * remembered

                if gates is not None:
                    gate_weight = mass.reshape(terms, slots, 1)

                    for layer, step in enumerate(steps):
                        for name, values in zip(
                            self._gate_names, step.gates, strict=True
                        ):
                            gates[name][layer] = gates[name][
                                layer
                            ] + gate_weight * values.reshape(terms, slots, self._width)

            selected = at_depth.reshape(1, slots, 1, 1)
            states = torch.where(selected, mixed_states, states)
            memories = torch.where(selected, mixed_memories, memories)

        live = probabilities[..., arities.ge(0)].sum(dim=-1).gt(0)
        self._publish_recurrence(states, gates, live)

        return states

    def _publish_recurrence(
        self,
        states: Tensor,
        gates: dict[str, list[Tensor]] | None,
        occupied: Tensor,
    ) -> None:
        """Publish exact recurrent activations when an observation scope is open.

        Parameters
        ----------
        states : Tensor
            The final hidden states over term, slot, layer and width.
        gates : dict[str, list[Tensor]] | None
            Each named activation at every layer, or none when no scope was open
            during its computation.
        occupied : Tensor
            The occupancy mask over term and slot.
        """
        if gates is None:
            return

        for layer in range(self._layers):
            self._observe(
                Role.RECURRENT_HIDDEN,
                (layer,),
                states[:, :, layer, :],
                occupied,
            )

            for name in self._gate_names:
                self._observe(Role.GATE, (layer, name), gates[name][layer], occupied)

    def _hard_states(self, terms: BatchEncoding) -> tuple[Tensor, PlacedEncoding]:
        """Return the states of a batch of terms and their placed channels.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.
        """
        placed = self._codec.place_batch(terms, policy=self._policy())
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=self._policy().device,
        ).reshape(len(terms.rows), self._codec.capacity)
        rows = functional.embedding(codes, self._choice_rows())

        if self._scopes:
            self._observe(
                Role.PARAMETER,
                ("choice-embedding",),
                self._choice_rows(),
                None,
            )

        if self._inventory == "uncached" or self._scopes:
            return self._states(rows, placed.mask), placed

        return self._shared_states(terms), placed

    def _inventory_level(self, batch: LevelBatch, /) -> Tensor:
        """Return packed hidden and memory states for one shared-DAG level."""
        nodes = int(batch.choices.numel())
        state_width = self._layers * self._width
        inputs = functional.embedding(batch.choices, self._choice_rows()).reshape(
            nodes, 1, self._width
        )
        packed = inputs.new_zeros((nodes, 2 * state_width))

        for rank in sorted({int(value) for value in batch.arities.unique()}):
            selected = batch.arities.eq(rank)
            count = int(selected.sum())

            if rank == 0:
                previous = self._start.reshape(1, 1, self._layers, self._width).expand(
                    count, 1, self._layers, self._width
                )
                memory = torch.zeros_like(previous)
            else:
                children = batch.children

                if children is None:
                    msg = "a nonleaf inventory level has no child embeddings"
                    raise RuntimeError(msg)

                held = children[selected, :rank]
                hidden = held[..., :state_width].reshape(
                    count, rank, self._layers, self._width
                )
                remembered = held[..., state_width:].reshape(
                    count, rank, self._layers, self._width
                )
                previous = self._scheme.combine(
                    tuple(
                        Encoded(
                            hidden[:, position],
                            ("term", "layer", "width"),
                        )
                        for position in range(rank)
                    )
                ).tensor.unsqueeze(1)
                memory = remembered.sum(dim=1).unsqueeze(1)

            reached, memories, _ = self._advance(
                previous,
                memory,
                inputs[selected],
            )
            packed[selected] = torch.cat(
                (
                    reached.reshape(count, state_width),
                    memories.reshape(count, state_width),
                ),
                dim=1,
            )

        return packed

    def _shared_states(self, terms: BatchEncoding, /) -> Tensor:
        """Encode hard terms by evaluating each shared addressed subterm once."""
        if not terms.rows:
            return self._choice_rows().new_zeros(
                (0, self._codec.capacity, self._layers, self._width)
            )

        topology = InventoryTopology(terms, self._choices)

        if topology.node_count == 0:
            return self._choice_rows().new_zeros(
                (
                    len(terms.rows),
                    self._codec.capacity,
                    self._layers,
                    self._width,
                )
            )

        inventory = EntangledInventory(topology).to(device=self._policy().device)
        packed = inventory(self._inventory_level).placed()
        state_width = self._layers * self._width

        return packed[..., :state_width].reshape(
            len(terms.rows),
            self._codec.capacity,
            self._layers,
            self._width,
        )

    def _terms(
        self, states: Tensor, placed: PlacedEncoding, structure: BatchEncoding
    ) -> Encoded:
        """Return the topmost state at each term's root.

        Parameters
        ----------
        states : Tensor
            The state at every slot, over the term, slot, layer and width axes.
        placed : PlacedEncoding
            The occupancy channel the states were computed from.
        structure : BatchEncoding
            The source rows, read to decide whether a term mask is needed.

        Returns
        -------
        Encoded
            One vector per term, over the term and width axes, masked only where a row
            holds no term.
        """
        top = states[:, :, -1, :]
        roots = placed.mask & self._roots.unsqueeze(0)
        selected = torch.where(roots.unsqueeze(-1), top, torch.zeros_like(top)).sum(
            dim=1
        )
        mask = (
            None
            if all(row.length > 0 for row in structure.rows)
            else placed.mask.any(dim=1)
        )

        return Encoded(selected, _TERM_AXES, mask)

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            The topmost state at each root, over the term and width axes, masked only
            where a row holds no term.
        """
        states, placed = self._hard_states(terms)
        nodes = states[:, :, -1, :]

        if self._scopes:
            self._observe(Role.NODE_ENCODING, (), nodes, placed.mask)

        encoded = self._terms(states, placed, terms)

        if self._scopes:
            self._observe(Role.TERM_ENCODING, (), encoded.tensor, encoded.mask)

        return encoded

    def continue_synthetic(
        self,
        structure: BatchEncoding,
        leaves: Encoded,
        /,
        *,
        leaf_slots: tuple[int, ...],
    ) -> Encoded:
        """Continue a synthetic tree from one supplied live leaf per row.

        This structural continuation is exact for a single layer whose cell carries
        no separate memory: its complete child state is the published hidden vector.
        A cell with memory, such as an LSTM, retains a distinct memory vector, and a
        multilayer recurrence retains hidden vectors below the published top layer,
        so neither exposes enough state to continue.

        Raises
        ------
        ValueError
            If this encoder does not publish the complete recurrent state required
            for exact continuation, or if the supplied leaf batch is misaligned.
        """
        self.validate_synthetic_continuation()

        expected_axes = ("term", "width")

        if leaves.axes != expected_axes:
            msg = f"synthetic-tree leaves run over {expected_axes}, not {leaves.axes}"
            raise ValueError(msg)

        if leaves.size("width") != self._width:
            msg = (
                f"synthetic-tree leaves have width {leaves.size('width')}, "
                f"not encoder width {self._width}"
            )
            raise ValueError(msg)

        if leaves.tensor.shape[0] != len(structure.rows):
            msg = (
                f"synthetic-tree leaves have {leaves.tensor.shape[0]} rows beside "
                f"{len(structure.rows)} structures"
            )
            raise ValueError(msg)

        placed = self._codec.place_batch(structure, policy=self._policy())
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in structure.rows),
            dtype=torch.int64,
            device=self._policy().device,
        ).reshape(len(structure.rows), self._codec.capacity)
        symbols = functional.embedding(codes, self._choice_rows())
        injected = leaves.tensor.reshape(len(structure.rows), 1, self._width)
        states = self._states(
            symbols,
            placed.mask,
            injected=injected,
            injected_slots=leaf_slots,
        )

        return self._terms(states, placed, structure)

    def validate_synthetic_continuation(self) -> None:
        """Refuse unless published vectors are complete recurrent states."""
        if self._layers != 1 or self._carries_memory:
            msg = (
                "synthetic-tree continuation requires a one-layer TreeRNN whose cell "
                "carries no separate memory, not "
                f"{self._layers} layer(s) with cell {self._cell!r}"
            )
            raise ValueError(msg)

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per occupied slot of each term.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to encode.

        Returns
        -------
        Encoded
            The topmost state at each slot, over the term, slot and width axes, masked
            by occupancy and at zero where no node occupies the slot.
        """
        states, placed = self._hard_states(terms)
        nodes = states[:, :, -1, :]

        if self._scopes:
            self._observe(Role.NODE_ENCODING, (), nodes, placed.mask)

        return Encoded(nodes, _NODE_AXES, placed.mask)

    def encode_soft(
        self,
        choices: LogDistribution,
        /,
        *,
        structure: BatchEncoding,
    ) -> Encoded:
        """Return one vector per term for a distribution over its symbols.

        Parameters
        ----------
        choices : LogDistribution
            The log probability of each choice at each slot, over the term, slot and
            choice axes and normalized over the choice axis.
        structure : BatchEncoding
            The occupancy channel aligned with the term and slot axes. Its symbol
            channel is ignored.

        Returns
        -------
        Encoded
            The topmost state at each root, over the term and width axes, masked only
            where a row holds no term.

        Raises
        ------
        ValueError
            If the distribution's axes, term and slot shape, or occupancy mask disagree
            with the structure.
        """
        expected_axes = ("term", "slot", "choice")

        if choices.axes != expected_axes:
            msg = f"a choice distribution runs over {expected_axes}, not {choices.axes}"
            raise ValueError(msg)

        if choices.normalized_over != "choice":
            msg = (
                "a choice distribution is normalized over 'choice', not "
                f"{choices.normalized_over!r}"
            )
            raise ValueError(msg)

        if choices.size("choice") != self._choices.width:
            msg = (
                f"the choice distribution has width {choices.size('choice')}, "
                f"not {self._choices.width}"
            )
            raise ValueError(msg)

        policy = self._policy()
        placed = self._codec.place_batch(structure, policy=policy)
        probabilities = choices.tensor.to(
            device=policy.device, dtype=policy.dtype
        ).exp()

        if tuple(probabilities.shape[:2]) != tuple(placed.mask.shape):
            msg = (
                f"the choice distribution has term and slot shape "
                f"{tuple(probabilities.shape[:2])}, not {tuple(placed.mask.shape)}"
            )
            raise ValueError(msg)

        if choices.mask is not None and not torch.equal(
            choices.mask.to(device=policy.device), placed.mask
        ):
            msg = "the choice-distribution mask differs from the structure's occupancy"
            raise ValueError(msg)

        if self._scopes:
            self._observe(
                Role.PARAMETER,
                ("choice-embedding",),
                self._choice_rows(),
                None,
            )

        states = self._soft_states(probabilities)
        nodes = states[:, :, -1, :]

        if self._scopes:
            self._observe(Role.NODE_ENCODING, (), nodes, placed.mask)

        encoded = self._terms(states, placed, structure)

        if self._scopes:
            self._observe(Role.TERM_ENCODING, (), encoded.tensor, encoded.mask)

        return encoded

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
                msg = f"the recurrent encoder does not publish {site!r}"
                raise ValueError(msg)

        for site in patches:
            if site not in self._sites:
                msg = f"the recurrent encoder does not publish {site!r}"
                raise ValueError(msg)

            msg = f"the recurrent encoder site {site!r} is observation-only"
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
            msg = "recurrent instrumentation scopes must close inside out"
            raise RuntimeError(msg)

        self._scopes.pop()

    def site_map(self) -> SiteMap:
        """Return every published output and recurrent observation site."""
        return self._sites


def _build(
    settings: TreeRnnEncoderSettings, context: BuildContext, /
) -> TreeRnnEncoder:
    """Build and place a recurrent term encoder over the run's codec.

    The heads that merge a slot's children are built for the arities the codec's
    own signature reaches, so a grammar that never builds a constructor of some
    arity builds no head for it.

    Parameters
    ----------
    settings : TreeRnnEncoderSettings
        The width, the layer count and the symbol norm cap.
    context : BuildContext
        The codec, the component handle and the device policy.
    """
    codec = context.codec
    operators = codec.codec.index.signature.operators
    ranks = tuple(sorted({operator.arity for operator in operators if operator.arity}))

    return place(
        TreeRnnEncoder(
            codec=codec,
            choices=context.choices,
            component=context.component,
            scheme=ArityHeadScheme(
                width=settings.width,
                reached=operators,
                bound=ArityBound(ranks) if ranks else None,
            ),
            cell=settings.cell,
            layers=settings.layers,
            norm_cap=settings.norm_cap,
            inventory=settings.inventory,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=TreeRnnEncoderSpec, build=_build)
"""The recurrent term encoder's plugin."""
