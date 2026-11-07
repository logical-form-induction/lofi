"""Recurrent encoding over the declared linearization order."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Self, assert_never, final

import torch
from torch import nn

from lofi._types import ModuleSharing, SharedSubmodule
from lofi.nn._layers import recurrent_step
from lofi.nn._values import Encoded
from lofi.nn.embeddings import ChoiceEmbedding
from lofi.nn.policy import DevicePolicy, place
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
from lofi.spec.encoders.linear import LinearRnnEncoderSettings, LinearRnnEncoderSpec
from lofi.terms import ChoiceIndex, Encoding

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi.nn._layers import RecurrentStep
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.ports import Patch
    from lofi.nn.sites import ComponentId
    from lofi.spec.encoders.linear import DirectionMerge
    from lofi.spec.layers import RecurrentCell
    from lofi.terms import BatchEncoding

__all__ = ["LinearRnnEncoder"]

_TERM_AXES: Final = ("term", "width")
_NODE_AXES: Final = ("term", "slot", "width")
_STATE_AXES: Final = ("term", "slot", "width")

type _Direction = Literal["forward", "bidirectional"]
type _Variables = Literal["index", "level"]


class _Reading(NamedTuple):
    """One observed encoder tensor."""

    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class _InstrumentScope:
    """A nestable observation region for one linear encoder."""

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
        """Return the latest reading at every selected site."""
        return MappingProxyType(self._readings)

    def record(self, reading: _Reading) -> None:
        """Keep a reading selected by this open region."""
        if not self._closed and reading.site in self._sites:
            self._readings[reading.site] = reading

    def __enter__(self) -> Self:
        """Open the observation region."""
        if self._entered:
            msg = "a linear encoder instrumentation scope cannot be entered twice"
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
        """Close the observation region."""
        del kind, error, traceback
        self._closed = True
        self._leave(self)


def _merged(merge: DirectionMerge | None, forward: Tensor, backward: Tensor) -> Tensor:
    """Return the forward and backward states combined as the declaration says."""
    match merge:
        case "mean":
            return 0.5 * (forward + backward)
        case "sum":
            return forward + backward
        case "concatenation":
            return torch.cat((forward, backward), dim=-1)
        case None:
            msg = "a bidirectional recurrence resolved without a merge"
            raise ValueError(msg)
        case unreachable:
            assert_never(unreachable)


def _sites(component: ComponentId, layers: int, direction: _Direction) -> SiteMap:
    """Return the encoder's declared output and recurrent state sites."""
    published = {
        Site(component, Role.TERM_ENCODING): SiteSpec(
            _TERM_AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
        ),
        Site(component, Role.NODE_ENCODING): SiteSpec(
            _NODE_AXES, DTypeClass.REAL, differentiable=True, access=Access.OBSERVE
        ),
        Site(component, Role.PARAMETER, ("choice-embedding",)): SiteSpec(
            ("choice", "width"),
            DTypeClass.REAL,
            differentiable=True,
            access=Access.OBSERVE,
        ),
    }
    names = ("forward",) if direction == "forward" else ("forward", "backward")
    for name in names:
        for layer in range(layers):
            published[Site(component, Role.RECURRENT_HIDDEN, (name, layer))] = SiteSpec(
                _STATE_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            )
    return SiteMap(component, ComponentKind.TERM_ENCODER, published)


def _steps(cell: RecurrentCell, width: int, layers: int) -> tuple[RecurrentStep, ...]:
    """Return one single-step layer of the named recurrent cell per layer."""
    return tuple(recurrent_step(cell, inputs=width, width=width) for _ in range(layers))


def _child_sides(row: Encoding) -> tuple[int, ...]:
    """Return each slot's argument position under its encoded parent."""
    seen: dict[int, int] = {}
    sides: list[int] = []
    for slot, parent in enumerate(row.parents):
        if parent == Encoding.NO_PARENT:
            sides.append(0)
            continue
        side = seen.get(parent, 0)
        sides.append(side)
        if row.occupied[slot]:
            seen[parent] = side + 1
    return tuple(sides)


def _level_codes(index: ChoiceIndex, row: Encoding, /) -> tuple[int, ...]:
    """Return choice codes whose bound-variable payload is its absolute level."""
    codes = list(index.codes(row))
    sides = _child_sides(row)
    for slot, code in enumerate(tuple(codes)):
        if code == ChoiceIndex.PAD:
            continue
        choice = index.choice(code)
        if not choice.operator.is_bound_variable:
            continue
        payload = choice.payload
        if payload is None:
            msg = f"bound-variable choice {code} has no index"
            raise ValueError(msg)
        scopes = 0
        child = slot
        parent = row.parents[child]
        while parent != Encoding.NO_PARENT:
            parent_op = index.signature.operator(row.symbols[parent])
            scopes += sum(
                site.sort == choice.operator.sort
                for site in parent_op.scope(sides[child])
            )
            child = parent
            parent = row.parents[child]
        level = scopes - payload - 1
        if level < 0:
            msg = (
                f"bound-variable index {payload} at slot {slot} exceeds its "
                f"{scopes} in-scope sites"
            )
            raise ValueError(msg)
        codes[slot] = index.index(choice.operator, level)
    return tuple(codes)


@final
class LinearRnnEncoder(nn.Module):
    """Encode payload-aware choices in their codec linearization order."""

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        settings: LinearRnnEncoderSettings,
    ) -> None:
        super().__init__()
        self._embedding = ChoiceEmbedding(
            index=choices, width=settings.width, component=component
        )
        forward = _steps(settings.cell, settings.width, settings.layers)
        backward = (
            _steps(settings.cell, settings.width, settings.layers)
            if settings.direction == "bidirectional"
            else None
        )
        self._forward = nn.ModuleList(forward)
        self._backward = None if backward is None else nn.ModuleList(backward)
        self._forward_steps = forward
        self._backward_steps = backward
        self._codec = codec
        self._choices = choices
        self._component = component
        self._cell = settings.cell
        self._layers = settings.layers
        self._direction = settings.direction
        self._merge = settings.merge
        self._variables = settings.variables
        self._width = settings.width
        self._sites = _sites(component, settings.layers, settings.direction)
        self._scopes: list[_InstrumentScope] = []

    @property
    def output_width(self) -> int:
        """Return the encoding width, doubled when two directions are concatenated."""
        return 2 * self._width if self._merge == "concatenation" else self._width

    @property
    def codec(self) -> NeuralCodecView:
        """Return the exact codec view read by this encoder."""
        return self._codec

    @property
    def cell(self) -> RecurrentCell:
        """Return the declared recurrent cell."""
        return self._cell

    @property
    def layers(self) -> int:
        """Return the number of recurrent layers."""
        return self._layers

    @property
    def direction(self) -> _Direction:
        """Return the declared recurrence direction."""
        return self._direction

    @property
    def variables(self) -> _Variables:
        """Return how bound-variable identities are embedded."""
        return self._variables

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Publish the one choice embedding available for declared tying."""
        return MappingProxyType(
            {
                "choice-embedding": SharedSubmodule(
                    self._embedding,
                    ModuleSharing(in_parameters=True, in_state_dict=True),
                )
            }
        )

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the choice table."""
        return DevicePolicy(self._embedding.weight.device, self._embedding.weight.dtype)

    def _codes(self, terms: BatchEncoding) -> Tensor:
        """Place payload-aware choice codes under the declared variable identity."""
        rows = tuple(
            self._choices.codes(row)
            if self._variables == "index"
            else _level_codes(self._choices, row)
            for row in terms.rows
        )
        return torch.tensor(
            rows, dtype=torch.int64, device=self._policy().device
        ).reshape(len(rows), self._codec.capacity)

    @staticmethod
    def _advance(
        cells: tuple[RecurrentStep, ...],
        inputs: Tensor,
        hidden: Tensor,
        memory: Tensor,
    ) -> tuple[Tensor, Tensor]:
        """Advance every layer by one choice row."""
        following_hidden: list[Tensor] = []
        following_memory: list[Tensor] = []
        below = inputs
        for layer, cell in enumerate(cells):
            step = cell(below, hidden[layer], memory[layer])
            following_hidden.append(step.hidden)
            following_memory.append(step.memory)
            below = step.hidden
        return torch.stack(following_hidden), torch.stack(following_memory)

    def _run_direction(
        self,
        inputs: Tensor,
        occupied: Tensor,
        cells: tuple[RecurrentStep, ...],
        slots: range,
        name: str,
    ) -> tuple[Tensor, Tensor]:
        """Return per-slot and final states from one traversal direction."""
        terms, capacity, width = inputs.shape
        hidden = inputs.new_zeros((self._layers, terms, width))
        memory = torch.zeros_like(hidden)
        states = inputs.new_zeros((terms, capacity, self._layers, width))
        for slot in slots:
            reached, remembered = self._advance(cells, inputs[:, slot], hidden, memory)
            live = occupied[:, slot].reshape(1, terms, 1)
            hidden = torch.where(live, reached, hidden)
            memory = torch.where(live, remembered, memory)
            states[:, slot] = hidden.permute(1, 0, 2) * live.permute(1, 0, 2)
        for layer in range(self._layers):
            self._observe(
                Role.RECURRENT_HIDDEN,
                (name, layer),
                states[:, :, layer],
                occupied,
            )
        return states[:, :, -1], hidden[-1]

    def _encode(self, terms: BatchEncoding) -> tuple[Encoded, Encoded]:
        """Return term and node encodings from one recurrent pass."""
        placed = self._codec.place_batch(terms, policy=self._policy())
        inputs = self._embedding(self._codes(terms))
        forward_nodes, forward_term = self._run_direction(
            inputs,
            placed.mask,
            self._forward_steps,
            range(self._codec.capacity),
            "forward",
        )
        if self._backward_steps is None:
            nodes = forward_nodes
            term = forward_term
        else:
            backward_nodes, backward_term = self._run_direction(
                inputs,
                placed.mask,
                self._backward_steps,
                range(self._codec.capacity - 1, -1, -1),
                "backward",
            )
            nodes = _merged(self._merge, forward_nodes, backward_nodes)
            term = _merged(self._merge, forward_term, backward_term)
        nodes = nodes * placed.mask.unsqueeze(-1)
        term_mask = placed.mask.any(dim=1)
        mask = None if bool(term_mask.all()) else term_mask
        encoded_nodes = Encoded(nodes, _NODE_AXES, placed.mask)
        encoded_term = Encoded(term, _TERM_AXES, mask)
        self._observe(
            Role.PARAMETER,
            ("choice-embedding",),
            self._embedding.weight,
            None,
        )
        self._observe(Role.NODE_ENCODING, (), nodes, placed.mask)
        self._observe(Role.TERM_ENCODING, (), term, mask)
        return encoded_term, encoded_nodes

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per term."""
        return self._encode(terms)[0]

    def encode_nodes(self, terms: BatchEncoding, /) -> Encoded:
        """Return one vector per occupied linearization slot."""
        return self._encode(terms)[1]

    def instrument(
        self,
        sites: frozenset[Site],
        /,
        *,
        patches: Mapping[Site, Patch[_Reading]],
    ) -> _InstrumentScope:
        """Return an observation-only scope over published sites."""
        for site in (*sites, *patches):
            if site not in self._sites:
                msg = f"the linear encoder does not publish {site!r}"
                raise ValueError(msg)
        if patches:
            site = next(iter(patches))
            msg = f"the linear encoder site {site!r} is observation-only"
            raise ValueError(msg)
        return _InstrumentScope(sites, self._scopes.append, self._detach)

    def _observe(
        self,
        role: Role,
        key: tuple[int | str, ...],
        tensor: Tensor,
        mask: Tensor | None,
    ) -> None:
        """Record one forward value in every selecting scope."""
        if not self._scopes:
            return
        site = Site(self._component, role, key)
        reading = _Reading(site, tensor.detach(), self._sites[site].axes, mask)
        for scope in self._scopes:
            scope.record(reading)

    def _detach(self, scope: _InstrumentScope) -> None:
        """Remove the innermost observation scope."""
        if not self._scopes or self._scopes[-1] is not scope:
            msg = "linear encoder scopes must close inside out"
            raise RuntimeError(msg)
        self._scopes.pop()

    def site_map(self) -> SiteMap:
        """Return every published encoder site."""
        return self._sites


def _build(
    settings: LinearRnnEncoderSettings, context: BuildContext, /
) -> LinearRnnEncoder:
    """Build and place one linear-recurrent encoder."""
    return place(
        LinearRnnEncoder(
            codec=context.codec,
            choices=context.choices,
            component=context.component,
            settings=settings,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=LinearRnnEncoderSpec, build=_build)
"""The linear-recurrent term encoder plugin."""
