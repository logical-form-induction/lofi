"""Autoregressive recurrence over the declared term linearization."""

from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    Self,
    cast,
    final,
)

import torch
from torch import nn

from lofi.nn._contexts import teacher_contexts
from lofi.nn._guards import checked_state
from lofi.nn._layers import recurrent_step
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.embeddings import ChoiceEmbedding
from lofi.nn.heads import TermHeads
from lofi.nn.policy import DevicePolicy, place
from lofi.nn.ports import (
    Consumption,
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
from lofi.registry import ComponentPlugin
from lofi.spec.decoders.linear import LinearRnnDecoderSettings, LinearRnnDecoderSpec
from lofi.terms import ChoiceIndex, HoleContext, Term

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch import Tensor

    from lofi._types import SharedSubmodule
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.ports import Patch, StructuralDeclaration
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import RecurrentCell
    from lofi.terms import BatchEncoding

__all__ = ["LinearRnnDecoder", "LinearRnnState"]

_TERM_AXES: Final = ("term", "width")
_ROW_AXES: Final = ("term", "slot", "choice")
_STRUCTURE: Final = MappingProxyType(
    {StructuralRelation.POSITION: Consumption.RECURRENCE}
)


class _EncoderProducer(Protocol):
    """The represented-space producer a linear decoder reads."""

    @property
    def output_width(self) -> int:
        """Return the width of one encoded term."""
        ...

    @property
    def shared(self) -> Mapping[str, SharedSubmodule[nn.Module]]:
        """Return explicitly published submodules by name."""
        ...


class LinearRnnState(NamedTuple):
    """The persistent recurrent state of one term frontier."""

    hidden: Tensor
    memory: Tensor
    current: Tensor
    steps: int


class _Reading(NamedTuple):
    """One observed decoder tensor."""

    site: Site
    tensor: Tensor
    axes: tuple[str, ...]
    mask: Tensor | None = None


class _InstrumentScope:
    """A nestable observation region for one linear decoder."""

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
            msg = "a linear decoder instrumentation scope cannot be entered twice"
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


def _sites(component: ComponentId) -> SiteMap:
    """Return the decoder's published output and state sites."""
    return SiteMap(
        component,
        ComponentKind.TERM_DECODER,
        {
            Site(component, Role.SYMBOL_DISTRIBUTION): SiteSpec(
                ("term", "slot", "symbol"),
                DTypeClass.LOG_PROBABILITY,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(component, Role.DECODER_RECURRENT_STATE): SiteSpec(
                ("term", "slot", "width"),
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
        },
    )


@final
class LinearRnnDecoder(nn.Module):
    """Decode terms one payload-aware choice at a time in linearization order."""

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        input_width: int,
        settings: LinearRnnDecoderSettings,
        embedding: ChoiceEmbedding | None,
    ) -> None:
        super().__init__()
        width = settings.width
        if embedding is not None and embedding.index is not choices:
            msg = "a tied choice embedding and decoder must share one choice index"
            raise ValueError(msg)
        if embedding is not None and embedding.output_width != width:
            msg = (
                f"a tied choice embedding has width {embedding.output_width}, "
                f"not decoder width {width}"
            )
            raise ValueError(msg)
        self._embedding = (
            ChoiceEmbedding(index=choices, width=width, component=component)
            if embedding is None
            else embedding
        )
        steps = tuple(
            recurrent_step(settings.cell, inputs=width, width=width)
            for _ in range(settings.layers)
        )
        self._cells = nn.ModuleList(steps)
        self._steps = steps
        memory = steps[0].carries_memory
        self._heads = TermHeads(
            index=choices, input_width=width, kind=settings.choice_heads
        )
        self._input_projection = nn.Linear(input_width, settings.layers * width)
        self._memory_projection = (
            nn.Linear(input_width, settings.layers * width) if memory else None
        )
        self._root_hidden = nn.Parameter(torch.empty(settings.layers, width))
        self._root_memory = (
            nn.Parameter(torch.empty(settings.layers, width)) if memory else None
        )
        self._start = nn.Parameter(torch.empty(width))
        nn.init.normal_(self._root_hidden, std=width**-0.5)
        if self._root_memory is not None:
            nn.init.normal_(self._root_memory, std=width**-0.5)
        nn.init.normal_(self._start, std=width**-0.5)
        self._codec = codec
        self._choices = choices
        self._component = component
        self._cell = settings.cell
        self._layers = settings.layers
        self._input_width = input_width
        self._width = width
        self._sites = _sites(component)
        self._scopes: list[_InstrumentScope] = []
        operator_index = codec.codec.index
        self._symbol_choices = tuple(
            tuple(
                code
                for code in range(choices.width)
                if (
                    operator_index.PAD
                    if code == ChoiceIndex.PAD
                    else operator_index.index(choices.operator(code))
                )
                == symbol
            )
            for symbol in range(operator_index.width)
        )

    @property
    def codec(self) -> NeuralCodecView:
        """Return the exact codec view this decoder fills."""
        return self._codec

    @property
    def input_width(self) -> int:
        """Return the width of an encoded input term."""
        return self._input_width

    @property
    def width(self) -> int:
        """Return the recurrent-state width."""
        return self._width

    @property
    def cell(self) -> RecurrentCell:
        """Return the declared recurrent transition."""
        return self._cell

    @property
    def layers(self) -> int:
        """Return the number of recurrent layers."""
        return self._layers

    @property
    def structure(self) -> StructuralDeclaration:
        """Return the linear position recurrence consumed by the decoder."""
        return _STRUCTURE

    def admits(self, term: Term, /) -> bool:
        """Return whether a completed frontier belongs to the decoder language."""
        return term.op.sort in self._choices.signature.sorts

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the learned root state."""
        return DevicePolicy(self._root_hidden.device, self._root_hidden.dtype)

    def _placed(self, encoded: Encoded, /) -> tuple[Tensor, Tensor | None]:
        """Validate and place one batch of encoded inputs."""
        held = checked_state(encoded, self._input_width, "linear decoder")
        if held.axes != _TERM_AXES:
            msg = f"linear decoder inputs run over {_TERM_AXES}, not {held.axes}"
            raise ValueError(msg)
        if held.mask is not None and held.mask.ndim != 1:
            msg = "linear decoder inputs are masked over the term axis"
            raise ValueError(msg)
        policy = self._policy()
        return held.tensor.to(device=policy.device, dtype=policy.dtype), held.mask

    def _initial(self, vectors: Tensor) -> tuple[Tensor, Tensor]:
        """Return learned per-layer root states conditioned on encoded inputs."""
        terms = vectors.shape[0]
        projected = self._input_projection(vectors).reshape(
            terms, self._layers, self._width
        )
        hidden = projected + self._root_hidden.unsqueeze(0)
        if self._memory_projection is None:
            memory = torch.zeros_like(hidden)
        else:
            root_memory = self._root_memory
            if root_memory is None:
                msg = "a decoder whose cell carries memory requires learned root memory"
                raise RuntimeError(msg)
            memory = self._memory_projection(vectors).reshape_as(hidden)
            memory = memory + root_memory.unsqueeze(0)
        return hidden.permute(1, 0, 2), memory.permute(1, 0, 2)

    def _advance(
        self, inputs: Tensor, hidden: Tensor, memory: Tensor
    ) -> tuple[Tensor, Tensor, Tensor]:
        """Advance every recurrent layer once."""
        following_hidden: list[Tensor] = []
        following_memory: list[Tensor] = []
        below = inputs
        for layer, cell in enumerate(self._steps):
            step = cell(below, hidden[layer], memory[layer])
            following_hidden.append(step.hidden)
            following_memory.append(step.memory)
            below = step.hidden
        reached_hidden = torch.stack(following_hidden)
        reached_memory = torch.stack(following_memory)
        return reached_hidden, reached_memory, reached_hidden[-1]

    def start(self, inputs: Encoded, /) -> LinearRnnState:
        """Return one frontier initialized by learned per-layer root states."""
        vectors, mask = self._placed(inputs)
        if vectors.shape[0] != 1:
            msg = f"one linear frontier starts from one input, not {vectors.shape[0]}"
            raise ValueError(msg)
        if mask is not None and not bool(mask[0]):
            msg = "one linear frontier starts from an unmasked input"
            raise ValueError(msg)
        hidden, memory = self._initial(vectors)
        hidden, memory, current = self._advance(
            self._start.unsqueeze(0), hidden, memory
        )
        return LinearRnnState(hidden, memory, current[0], 0)

    def weigh(
        self, state: LinearRnnState, /, *, context: HoleContext
    ) -> LogDistribution:
        """Return the legal choice distribution at the next frontier hole."""
        row = self._heads(state.current.unsqueeze(0), (context,))
        return LogDistribution(
            row.tensor[0],
            ("choice",),
            None if row.mask is None else row.mask[0],
            normalized_over="choice",
        )

    def fill(self, state: LinearRnnState, /, *, choices: Tensor) -> LinearRnnState:
        """Advance one frontier through one payload-aware choice."""
        if choices.ndim != 0:
            msg = f"one frontier choice is scalar, not shape {tuple(choices.shape)}"
            raise ValueError(msg)
        choice = int(choices.item())
        if not ChoiceIndex.PAD < choice < self._choices.width:
            msg = f"choice {choice} lies outside [1, {self._choices.width})"
            raise ValueError(msg)
        hidden, memory, current = self._advance(
            self._embedding(choices.reshape(1)), state.hidden, state.memory
        )
        return LinearRnnState(hidden, memory, current[0], state.steps + 1)

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        """Return one teacher-forced choice row per linearized slot."""
        vectors, input_mask = self._placed(inputs)
        placed = self._codec.place_batch(terms, policy=self._policy())
        if placed.term_count != vectors.shape[0]:
            msg = (
                f"{placed.term_count} teacher terms arrived beside "
                f"{vectors.shape[0]} encoded inputs"
            )
            raise ValueError(msg)
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=self._policy().device,
        ).reshape(placed.term_count, self._codec.capacity)
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        hidden, memory = self._initial(vectors)
        hidden, memory, current = self._advance(
            self._start.unsqueeze(0).expand(placed.term_count, -1), hidden, memory
        )
        rows: list[Tensor] = []
        states: list[Tensor] = []
        for slot in range(self._codec.capacity):
            states.append(current)
            row = self._heads(current, tuple(per_term[slot] for per_term in contexts))
            live = placed.mask[:, slot]
            values = row.tensor.masked_fill(~live.unsqueeze(1), -torch.inf)
            rows.append(values)
            reached_hidden, reached_memory, reached = self._advance(
                self._embedding(codes[:, slot]), hidden, memory
            )
            selected = live.reshape(1, placed.term_count, 1)
            hidden = torch.where(selected, reached_hidden, hidden)
            memory = torch.where(selected, reached_memory, memory)
            current = torch.where(live.unsqueeze(1), reached, current)
        values = torch.stack(rows, dim=1)
        mask = placed.mask
        if input_mask is not None:
            mask = mask & input_mask.unsqueeze(1)
            values = values.masked_fill(~mask.unsqueeze(2), -torch.inf)
        state_values = torch.stack(states, dim=1)
        state_values = state_values.masked_fill(~mask.unsqueeze(2), 0.0)
        symbols = self._symbol_rows(values)
        self._observe(Role.SYMBOL_DISTRIBUTION, symbols, mask)
        self._observe(Role.DECODER_RECURRENT_STATE, state_values, mask)
        return LogDistribution(values, _ROW_AXES, mask, normalized_over="choice")

    def _symbol_rows(self, choices: Tensor, /) -> Tensor:
        """Collapse payload-aware choice mass onto operator symbols."""
        rows = []
        for codes in self._symbol_choices:
            if not codes:
                rows.append(choices.new_full(choices.shape[:-1], -torch.inf))
                continue
            selected = choices[..., torch.tensor(codes, device=choices.device)]
            rows.append(torch.logsumexp(selected, dim=-1))
        return torch.stack(rows, dim=-1)

    def site_map(self) -> SiteMap:
        """Return the decoder's declared choice-row and recurrent-state sites."""
        return self._sites

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
                msg = f"the linear decoder does not publish {site!r}"
                raise ValueError(msg)
        if patches:
            site = next(iter(patches))
            msg = f"the linear decoder site {site!r} is observation-only"
            raise ValueError(msg)
        return _InstrumentScope(sites, self._scopes.append, self._detach)

    def _observe(self, role: Role, tensor: Tensor, mask: Tensor | None) -> None:
        """Record one teacher-forced value in every selecting scope."""
        if not self._scopes:
            return
        site = Site(self._component, role)
        reading = _Reading(site, tensor.detach(), self._sites[site].axes, mask)
        for scope in self._scopes:
            scope.record(reading)

    def _detach(self, scope: _InstrumentScope) -> None:
        """Remove the innermost observation scope."""
        if not self._scopes or self._scopes[-1] is not scope:
            msg = "linear decoder scopes must close inside out"
            raise RuntimeError(msg)
        self._scopes.pop()


def _tied_embedding(
    producer: _EncoderProducer,
    choices: ChoiceIndex,
    width: int,
    /,
) -> ChoiceEmbedding:
    """Return the producer's explicitly published choice embedding."""
    published = producer.shared.get("choice-embedding")
    if published is None:
        msg = "a tied linear decoder's encoder publishes no choice-embedding"
        raise ValueError(msg)
    module = published.module
    module = cast("ChoiceEmbedding", module)
    if module.index is not choices or module.output_width != width:
        msg = "a tied choice embedding must share the decoder index and width"
        raise ValueError(msg)
    return module


def _build(
    settings: LinearRnnDecoderSettings, context: BuildContext, /
) -> LinearRnnDecoder:
    """Build and place one linear-recurrent decoder."""
    producer = cast("_EncoderProducer", context.producer("encoder"))
    tied = (
        None
        if settings.tie is None
        else _tied_embedding(producer, context.choices, settings.width)
    )
    return place(
        LinearRnnDecoder(
            codec=context.codec,
            choices=context.choices,
            component=context.component,
            input_width=producer.output_width,
            settings=settings,
            embedding=tied,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=LinearRnnDecoderSpec, build=_build)
"""The linear-recurrent term decoder plugin."""
