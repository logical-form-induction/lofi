"""Masked diffusion over the slots of a term codec, denoised in one call."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Self, cast, final, override

import torch
from torch import Tensor, nn

from lofi._checks import ensure_finite, ensure_non_negative
from lofi.nn._contexts import slot_paths, teacher_contexts
from lofi.nn._guards import checked_state, positive
from lofi.nn._layers import activation
from lofi.nn._values import Encoded, LogDistribution
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
from lofi.spec.decoders.diffusion import (
    DiffusionTermDecoderSettings,
    DiffusionTermDecoderSpec,
)
from lofi.terms import ChoiceIndex, HoleContext, Term

if TYPE_CHECKING:
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import NeuralCodecView, PlacedEncoding
    from lofi.nn.ports import StructuralDeclaration, TermEncoder
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import HeadKind
    from lofi.terms import BatchEncoding

__all__ = [
    "DiffusionDecoderSettings",
    "DiffusionTermDecoder",
    "NoiseSchedule",
]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of an encoded batch of terms."""

_STREAM_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of one stream position per slot of each term."""

_CHOICE_AXES: Final[tuple[str, ...]] = ("term", "slot", "choice")
"""The axes of one payload-aware choice row per slot."""

_STRUCTURE: Final[StructuralDeclaration] = MappingProxyType(
    {StructuralRelation.POSITION: Consumption.EMBEDDING}
)
"""The one relation read from the codec: the position of a slot, as the vector
added to the slot's input."""


def _checked_beta(step: int, value: float) -> float:
    """Return the fraction of the remaining choices one step masks.

    Parameters
    ----------
    step : int
        Which step the fraction belongs to, for the message.
    value : float
        The fraction to check.

    Raises
    ------
    TypeError
        If it is not a number.
    ValueError
        If it is not above zero and at most one.
    """
    beta = ensure_finite(f"the masking fraction of step {step}", value)

    if not 0.0 < beta <= 1.0:
        msg = f"a masking fraction lies in (0, 1], and step {step} masks {beta}"
        raise ValueError(msg)

    return beta


@final
class NoiseSchedule:
    """The fraction of a term's choices masked at each step of a diffusion.

    At step ``t``, ``beta_t`` masks each visible choice and ``alpha_bar_t`` records
    its probability of surviving through that step. The final step leaves every
    choice masked for the start of the reverse process.

    Parameters
    ----------
    betas : tuple[float, ...]
        The masking fraction of each step, in step order, each above zero and at
        most one, the last exactly one.

    Raises
    ------
    TypeError
        If a fraction is not a number.
    ValueError
        If the schedule has no step, if a fraction is not above zero and at most one,
        or if the last step leaves a choice visible.
    """

    __slots__ = ("_alpha_bars", "_betas")

    def __init__(self, betas: tuple[float, ...]) -> None:
        if not betas:
            msg = "a noise schedule has at least one step"
            raise ValueError(msg)

        checked = tuple(_checked_beta(step, beta) for step, beta in enumerate(betas))

        if checked[-1] != 1.0:
            msg = (
                f"the last step of a noise schedule masks every choice still "
                f"visible, and step {len(checked) - 1} masks {checked[-1]}"
            )
            raise ValueError(msg)

        alpha_bars: list[float] = []
        visible = 1.0

        for beta in checked:
            visible *= 1.0 - beta
            alpha_bars.append(visible)

        self._betas = checked
        self._alpha_bars = tuple(alpha_bars)

    @classmethod
    def linear(cls, steps: int, *, first: float) -> Self:
        """Return a schedule whose fractions rise evenly from a first value to one.

        Parameters
        ----------
        steps : int
            How many steps the schedule holds.
        first : float
            The masking fraction of the first step, which is also the last step of a
            schedule of one step and so must then be one.

        Raises
        ------
        TypeError
            If the step count is not an integer, or the first fraction is not a
            number.
        ValueError
            If the step count is not positive, if the first fraction is not above
            zero and at most one, or if a schedule of one step is given a first
            fraction below one.
        """
        count = positive("the step count of a noise schedule", steps)

        if count == 1:
            return cls((first,))

        rising = (
            first + (1.0 - first) * step / (count - 1) for step in range(count - 1)
        )

        return cls((*rising, 1.0))

    @property
    def steps(self) -> int:
        """Return how many steps the schedule holds."""
        return len(self._betas)

    @property
    def betas(self) -> tuple[float, ...]:
        """Return the masking fraction of each step, in step order."""
        return self._betas

    @property
    def alpha_bars(self) -> tuple[float, ...]:
        """Return the probability a choice is still visible after each step.

        Returns
        -------
        tuple[float, ...]
            One probability per step, in step order, falling to zero at the last.
        """
        return self._alpha_bars

    def _checked_step(self, step: int) -> int:
        """Return a step of this schedule.

        Parameters
        ----------
        step : int
            The step to check.

        Raises
        ------
        TypeError
            If it is not an integer.
        ValueError
            If it is negative.
        IndexError
            If the schedule holds no such step.
        """
        at = ensure_non_negative("a diffusion step", step)

        if at >= len(self._betas):
            msg = (
                f"this schedule holds {len(self._betas)} steps, so step {at} is not one"
            )
            raise IndexError(msg)

        return at

    def previous_alpha_bar(self, step: int, /) -> float:
        """Return the probability a choice is visible on entering a step.

        Parameters
        ----------
        step : int
            The step being entered.

        Returns
        -------
        float
            The cumulative coefficient of the step before, or one on entering the
            first step, before which nothing is masked.

        Raises
        ------
        TypeError
            If the step is not an integer.
        ValueError
            If it is negative.
        IndexError
            If the schedule holds no such step.
        """
        at = self._checked_step(step)

        if at == 0:
            return 1.0

        return self._alpha_bars[at - 1]

    def reveal_probability(self, step: int, /) -> float:
        """Return the probability the reverse process reveals a masked choice at a step.

        Parameters
        ----------
        step : int
            The step the reverse process is undoing.

        Returns
        -------
        float
            The share of the choices masked after the step that were still visible on
            entering it. At the first step it is one, so that nothing stays masked
            once the reverse process ends, and at the last step it is the coefficient
            on entering that step, since nothing is visible after it.

        Raises
        ------
        TypeError
            If the step is not an integer.
        ValueError
            If it is negative.
        IndexError
            If the schedule holds no such step.
        """
        at = self._checked_step(step)
        after = self._alpha_bars[at]

        return (self.previous_alpha_bar(at) - after) / (1.0 - after)

    def draw(
        self,
        count: int,
        *,
        device: torch.device,
        generator: torch.Generator | None = None,
    ) -> Tensor:
        """Return steps drawn uniformly from the schedule.

        Parameters
        ----------
        count : int
            How many steps to draw.
        device : torch.device
            Where to put the result.
        generator : torch.Generator | None
            The source the draw reads, or None to read the ambient one.

        Returns
        -------
        Tensor
            Rank 1 over the draw, in ``int64``, each entry a step of the schedule.

        Raises
        ------
        TypeError
            If the count is not an integer.
        ValueError
            If it is negative.
        """
        return torch.randint(
            len(self._betas),
            (ensure_non_negative("a draw count", count),),
            generator=generator,
            device=device,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another schedule masks the same fraction at every step."""
        if type(other) is not NoiseSchedule:
            return NotImplemented

        return self._betas == other._betas

    @override
    def __hash__(self) -> int:
        """Return a hash over the masking fractions."""
        return hash((NoiseSchedule, self._betas))

    @override
    def __repr__(self) -> str:
        """Return an evaluable representation."""
        return f"NoiseSchedule({self._betas!r})"


class DiffusionDecoderSettings(NamedTuple):
    """Diffusion decoder dimensions.

    Attributes
    ----------
    input_width : int
        The width of the vector each term is encoded to.
    width : int
        The stream width.
    layers : int
        The number of denoising layers.
    feedforward_width : int
        The hidden width of each feedforward block.
    activation : str
        The installed ``lofi.activation`` entry each feedforward block applies.
    """

    input_width: int
    width: int
    layers: int
    feedforward_width: int
    activation: str


def _decoder_sites(component: ComponentId) -> SiteMap:
    """Return the stream site published by a diffusion decoder.

    Parameters
    ----------
    component : ComponentId
        The component identifier assigned by the run.

    Returns
    -------
    SiteMap
        The choice distribution and the residual stream it is read from.
    """
    return SiteMap(
        component,
        ComponentKind.TERM_DECODER,
        {
            Site(component, Role.SYMBOL_DISTRIBUTION): SiteSpec(
                _CHOICE_AXES,
                DTypeClass.LOG_PROBABILITY,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(component, Role.DECODER_RESIDUAL_STREAM): SiteSpec(
                _STREAM_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
        },
    )


class _Block(nn.Module):
    """One pre-norm denoising layer mixing across the slots and then across the width.

    Parameters
    ----------
    width : int
        The stream width.
    capacity : int
        How many slots the stream runs over.
    inner_width : int
        The hidden width of the feedforward block.
    activation : nn.Module
        The elementwise map the feedforward block applies.
    """

    def __init__(
        self, width: int, capacity: int, inner_width: int, activation: nn.Module
    ) -> None:
        super().__init__()
        self._activation = activation
        self._mixing_norm = nn.LayerNorm(width)
        self._mix = nn.Linear(capacity, capacity)
        self._feedforward_norm = nn.LayerNorm(width)
        self._expand = nn.Linear(width, inner_width)
        self._contract = nn.Linear(inner_width, width)

    @override
    def forward(self, stream: Tensor, /) -> Tensor:
        """Return the stream after one layer.

        Parameters
        ----------
        stream : Tensor
            Rank three over the term, the slot and the width.

        Returns
        -------
        Tensor
            The stream, of the same shape, each slot having read every other.
        """
        mixed = self._mix(self._mixing_norm(stream).transpose(1, 2)).transpose(1, 2)
        stream = stream + mixed

        return stream + self._contract(
            self._activation(self._expand(self._feedforward_norm(stream)))
        )


class _DiffusionFrontier(NamedTuple):
    """One encoded input and the choices already revealed to its denoiser."""

    vectors: Tensor
    weights: Tensor
    revealed: Tensor
    pending: tuple[tuple[int, ...], ...]
    filled: int


@final
class DiffusionTermDecoder(nn.Module):
    """A masked diffusion over the fixed slots of a term codec.

    The forward process masks choices under the schedule; the denoiser reads the
    choices already filled, mask, positions, step, and encoded term. Frontier and
    teacher-forced decoding use the same canonical-order reveals.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    choices : ChoiceIndex
        The language-owned payload-aware choice alphabet.
    component : ComponentId
        The component identifier assigned by the run.
    settings : DiffusionDecoderSettings
        The model dimensions.
    schedule : NoiseSchedule
        The fraction of the visible choices each step masks.
    """

    _keep: Tensor
    _positions: Tensor
    _reveal: Tensor

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        settings: DiffusionDecoderSettings,
        schedule: NoiseSchedule,
        choice_heads: HeadKind,
    ) -> None:
        super().__init__()

        served = codec.serve(_STRUCTURE)
        skeleton = codec.codec.skeleton
        if skeleton is None:
            msg = "a diffusion decoder requires a fixed path-addressed layout"
            raise ValueError(msg)
        if choices.signature is not codec.codec.index.signature:
            msg = "a diffusion decoder's choices and codec share one signature"
            raise ValueError(msg)
        input_width = settings.input_width
        width = settings.width
        layer_count = settings.layers
        inner_width = settings.feedforward_width

        self._choices_table = nn.Embedding(
            choices.width, width, padding_idx=ChoiceIndex.PAD
        )
        self._masked = nn.Parameter(torch.zeros(width))
        self._places = nn.Embedding(codec.capacity, width)
        self._times = nn.Embedding(schedule.steps, width)
        self._project = nn.Linear(input_width, width)
        self._layers = nn.ModuleList(
            _Block(width, codec.capacity, inner_width, activation(settings.activation))
            for _ in range(layer_count)
        )
        self._norm = nn.LayerNorm(width)
        self._choice_heads = TermHeads(
            index=choices,
            input_width=width,
            kind=choice_heads,
        )
        policy = DevicePolicy(self._project.weight.device, self._project.weight.dtype)
        self.register_buffer("_positions", served.position(policy=policy))
        self.register_buffer(
            "_keep",
            torch.tensor(schedule.alpha_bars, dtype=policy.dtype, device=policy.device),
        )
        self.register_buffer(
            "_reveal",
            torch.tensor(
                tuple(schedule.reveal_probability(t) for t in range(schedule.steps)),
                dtype=policy.dtype,
                device=policy.device,
            ),
        )
        self._codec = codec
        self._choices = choices
        self._schedule = schedule
        self._sites = _decoder_sites(component)
        self._input_width = input_width
        self._width = width
        self._paths = slot_paths(skeleton)
        self._path_slots = MappingProxyType(
            {path: slot for slot, path in enumerate(self._paths)}
        )
        self._slot_legal = tuple(
            tuple(
                code != ChoiceIndex.PAD
                and codec.codec.legal_symbols[slot][
                    codec.codec.index.index(choices.operator(code))
                ]
                for code in range(choices.width)
            )
            for slot in range(codec.capacity)
        )

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view supplied at construction."""
        return self._codec

    @property
    def schedule(self) -> NoiseSchedule:
        """Return the noise schedule supplied at construction."""
        return self._schedule

    @property
    def structure(self) -> StructuralDeclaration:
        """Return the relations this decoder reads from its codec, and how."""
        return _STRUCTURE

    @property
    def input_width(self) -> int:
        """Return the width of the vector this decoder reads a term from."""
        return self._input_width

    @property
    def width(self) -> int:
        """Return the stream width."""
        return self._width

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the readout weight.

        Returns
        -------
        DevicePolicy
            Where this decoder's own parameters live, and at what precision.
        """
        return DevicePolicy(self._project.weight.device, self._project.weight.dtype)

    def _placed(self, encoded: Encoded) -> tuple[Tensor, Tensor | None]:
        """Return the encoded terms beside this decoder's parameters, and their mask.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            The vectors, of rank two over the term and the width, at the parameters'
            device and precision, and the mask over the term, or None where every row
            holds a term.

        Raises
        ------
        ValueError
            If the value does not run over the term and the width, runs over a width
            other than the one this decoder reads, or is masked over an axis other
            than the term.
        """
        held = checked_state(encoded, self._input_width, "decoder")

        if held.axes != _TERM_AXES:
            msg = f"an encoded batch of terms runs over {_TERM_AXES}, not {held.axes}"
            raise ValueError(msg)

        mask = held.mask

        if mask is not None and mask.ndim != 1:
            msg = f"an encoded batch of terms is masked over the term, not {held.axes}"
            raise ValueError(msg)

        policy = self._policy()

        return (held.tensor.to(device=policy.device, dtype=policy.dtype), mask)

    def _beside(self, vectors: Tensor, terms: BatchEncoding) -> PlacedEncoding:
        """Return the terms laid beside encoded vectors as placed channels.

        Parameters
        ----------
        vectors : Tensor
            One vector per term, of rank two over the term and the width.
        terms : BatchEncoding
            The linearized terms, in the order of the term axis.

        Returns
        -------
        PlacedEncoding
            The channels, of rank two over the term and the slot.

        Raises
        ------
        ValueError
            If there is not one linearization per encoded term.
        """
        placed = self._codec.place_batch(terms, policy=self._policy())

        if placed.term_count != vectors.shape[0]:
            msg = (
                f"{placed.term_count} linearized terms arrived beside "
                f"{vectors.shape[0]} encoded ones"
            )
            raise ValueError(msg)

        return placed

    def _one_hot(self, choices: Tensor) -> Tensor:
        """Return the weight each choice has when one choice per slot is given.

        Parameters
        ----------
        choices : Tensor
            The choice at each slot, over its own leading axes.

        Returns
        -------
        Tensor
            One at each given choice and zero elsewhere, with the choice axis appended.
        """
        policy = self._policy()
        held = torch.zeros(
            (*choices.shape, self._choices.width),
            dtype=policy.dtype,
            device=policy.device,
        )

        return held.scatter(-1, choices.unsqueeze(-1), 1.0)

    def _choice_rows(
        self,
        stream: Tensor,
        contexts: tuple[tuple[HoleContext, ...], ...],
    ) -> LogDistribution:
        """Return structurally legal choice rows from a complete denoiser stream."""
        rows: list[Tensor] = []
        masks: list[Tensor] = []

        for slot in range(self._codec.capacity):
            row = self._choice_row(
                stream[:, slot],
                tuple(per_term[slot] for per_term in contexts),
                (slot,) * stream.shape[0],
            )
            rows.append(row.tensor)
            if row.mask is None:
                msg = "a diffusion choice row must carry its legality mask"
                raise RuntimeError(msg)
            masks.append(row.mask)

        return LogDistribution(
            torch.stack(rows, dim=1),
            _CHOICE_AXES,
            torch.stack(masks, dim=1),
            "choice",
        )

    def _choice_row(
        self,
        stream: Tensor,
        contexts: tuple[HoleContext, ...],
        slots: tuple[int, ...],
    ) -> LogDistribution:
        """Return legal choice rows for aligned term states and layout slots."""
        row = self._choice_heads(stream, contexts)
        structural = torch.stack(
            tuple(
                torch.tensor(
                    self._slot_legal[slot], dtype=torch.bool, device=stream.device
                )
                for slot in slots
            )
        )
        legal = structural if row.mask is None else structural & row.mask
        if bool((~legal.any(dim=1)).any()):
            msg = "a diffusion decoder reached a slot with no admissible choice"
            raise ValueError(msg)
        values = torch.log_softmax(row.tensor.masked_fill(~legal, -torch.inf), dim=1)
        return LogDistribution(values, ("term", "choice"), legal, "choice")

    def _step(self, filled: int, count: int, *, device: torch.device) -> Tensor:
        """Return the declared diffusion time for a canonical frontier depth."""
        at = max(
            0,
            self._schedule.steps
            - 1
            - filled * self._schedule.steps // self._codec.capacity,
        )
        return torch.full((count,), at, dtype=torch.int64, device=device)

    def _denoise(
        self, vectors: Tensor, weights: Tensor, revealed: Tensor, steps: Tensor
    ) -> Tensor:
        """Return the stream one denoising call reports.

        Parameters
        ----------
        vectors : Tensor
            One vector per term, of rank two over the term and the width.
        weights : Tensor
            The weight each choice has at each slot as what the slot shows the
            denoiser, of rank three over the term, the slot and the choice.
        revealed : Tensor
            True where a slot shows its choice rather than the mask, of rank two over
            the term and the slot.
        steps : Tensor
            The step each term is denoised at, of rank one over the term.

        Returns
        -------
        Tensor
            The stream, of rank three over the term, the slot and the width.
        """
        shown = weights @ self._choices_table.weight
        source = torch.where(
            revealed.unsqueeze(-1), shown, self._masked.reshape(1, 1, -1)
        )
        stream = (
            source
            + self._places(self._positions).unsqueeze(0)
            + self._times(steps).unsqueeze(1)
            + self._project(vectors).unsqueeze(1)
        )

        for layer in self._layers:
            stream = layer(stream)

        return self._norm(stream)

    def _all_masked(self, vectors: Tensor) -> Tensor:
        """Return the stream of the last step with every slot masked.

        Parameters
        ----------
        vectors : Tensor
            One vector per term, of rank two over the term and the width.

        Returns
        -------
        Tensor
            The stream, of rank three over the term, the slot and the width.
        """
        count = vectors.shape[0]
        capacity = self._codec.capacity
        weights = torch.zeros(
            (count, capacity, self._choices.width),
            dtype=vectors.dtype,
            device=vectors.device,
        )
        revealed = torch.zeros(
            (count, capacity), dtype=torch.bool, device=vectors.device
        )
        steps = torch.full(
            (count,), self._schedule.steps - 1, dtype=torch.int64, device=vectors.device
        )

        return self._denoise(vectors, weights, revealed, steps)

    def states(self, encoded: Encoded, /) -> Encoded:
        """Return the stream an observing pass reads its rows from.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("term", "slot", "width")``, real valued, masked as
            the encoded terms are, from the last step with every slot masked.

        Raises
        ------
        ValueError
            If the encoded terms are shaped wrong.
        """
        vectors, mask = self._placed(encoded)
        stream = self._all_masked(vectors)

        return Encoded(stream, _STREAM_AXES, mask)

    def denoised(
        self,
        encoded: Encoded,
        /,
        *,
        terms: BatchEncoding,
        steps: Tensor,
        revealed: Tensor,
    ) -> LogDistribution:
        """Return the rows the denoiser reports when shown part of each term.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.
        terms : BatchEncoding
            The linearized terms whose choices the revealed slots show, in the order
            of the term axis.
        steps : Tensor
            The step each term is denoised at, over the axis ``("term",)``, in
            ``int64``.
        revealed : Tensor
            True where a slot shows the term's choice rather than the mask, over the
            axes ``("term", "slot")``, in boolean dtype.

        Returns
        -------
        LogDistribution
            Rank 3 over the axes ``("term", "slot", "choice")``, normalized over the
            choice axis, masked as the encoded terms are, with negative infinity at
            every slot of a masked term.

        Raises
        ------
        ValueError
            If the encoded terms are shaped wrong, if there is not one linearization
            per encoded term, if there is not one step per term or a step is outside
            the schedule, or if the revealed slots are not one flag per slot per term.
        """
        vectors, mask = self._placed(encoded)
        placed = self._beside(vectors, terms)
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=vectors.device,
        )
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        count = vectors.shape[0]

        if steps.shape != (count,) or steps.dtype is not torch.int64:
            msg = (
                f"a denoising step is one int64 entry per term, so steps run over "
                f"({count},), not {tuple(steps.shape)} in {steps.dtype}"
            )
            raise ValueError(msg)

        if bool(steps.lt(0).any()) or bool(steps.ge(self._schedule.steps).any()):
            msg = (
                f"this schedule holds {self._schedule.steps} steps, and a step "
                f"named here lies outside them: {steps.tolist()}"
            )
            raise ValueError(msg)

        if revealed.shape != codes.shape or revealed.dtype is not torch.bool:
            msg = (
                f"the revealed slots are one boolean per slot per term, over "
                f"{tuple(codes.shape)}, not {tuple(revealed.shape)} in "
                f"{revealed.dtype}"
            )
            raise ValueError(msg)

        stream = self._denoise(
            vectors,
            self._one_hot(codes),
            revealed.to(vectors.device),
            steps.to(vectors.device),
        )
        rows = self._choice_rows(stream, contexts)
        values = rows.tensor.masked_fill(~placed.mask.unsqueeze(2), -torch.inf)
        live = placed.mask
        if mask is not None:
            live = live & mask.unsqueeze(1)
            values = values.masked_fill(~mask.reshape(-1, 1, 1), -torch.inf)
        return LogDistribution(values, _CHOICE_AXES, live, normalized_over="choice")

    def bound(
        self,
        encoded: Encoded,
        /,
        *,
        terms: BatchEncoding,
        generator: torch.Generator | None = None,
    ) -> LogDistribution:
        """Return a one-draw estimate of the evidence lower bound of each term.

        The estimator samples one schedule step per term and reweights masked-slot
        log probabilities by the reveal probability and schedule length. It includes
        padding slots and therefore bounds the observing score.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.
        terms : BatchEncoding
            The linearized terms to bound, in the order of the term axis.
        generator : torch.Generator | None
            The source the draws read, or None to read the ambient one.

        Returns
        -------
        LogDistribution
            Rank 1 over the axis ``("term",)``, unnormalized and at most zero, masked
            as the encoded terms are, with negative infinity at a masked term.

        Raises
        ------
        ValueError
            If the encoded terms are shaped wrong, or if there is not one
            linearization per encoded term.
        """
        vectors, mask = self._placed(encoded)
        placed = self._beside(vectors, terms)
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=vectors.device,
        )
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        count = vectors.shape[0]
        steps = self._schedule.draw(count, device=vectors.device, generator=generator)
        thrown = torch.rand(
            codes.shape,
            generator=generator,
            dtype=vectors.dtype,
            device=vectors.device,
        )
        revealed = thrown.lt(self._keep[steps].unsqueeze(1))
        stream = self._denoise(vectors, self._one_hot(codes), revealed, steps)
        rows = self._choice_rows(stream, contexts).tensor
        held = rows.gather(2, codes.unsqueeze(-1)).squeeze(-1)
        target = placed.mask & ~revealed
        masked = torch.where(target, held, torch.zeros_like(held)).sum(dim=1)
        estimate = self._schedule.steps * self._reveal[steps] * masked

        if mask is not None:
            estimate = estimate.masked_fill(~mask, -torch.inf)

        return LogDistribution(estimate, ("term",), mask)

    def admits(self, term: Term, /) -> bool:
        """Return whether a completed term belongs to this decoder's language."""
        return term.op.sort in self._choices.signature.sorts

    def start(self, inputs: Encoded, /) -> _DiffusionFrontier:
        """Return one diffusion frontier with every slot masked."""
        vectors, mask = self._placed(inputs)
        if vectors.shape[0] != 1:
            msg = (
                f"one diffusion frontier starts from one input, not {vectors.shape[0]}"
            )
            raise ValueError(msg)
        if mask is not None and not bool(mask[0]):
            msg = "one diffusion frontier starts from an unmasked input"
            raise ValueError(msg)
        weights = vectors.new_zeros((1, self._codec.capacity, self._choices.width))
        revealed = torch.zeros(
            (1, self._codec.capacity), dtype=torch.bool, device=vectors.device
        )
        return _DiffusionFrontier(vectors, weights, revealed, ((),), 0)

    def weigh(
        self, state: _DiffusionFrontier, /, *, context: HoleContext
    ) -> LogDistribution:
        """Return the denoiser row at the frontier's next masked slot."""
        if not state.pending:
            msg = "a complete diffusion frontier has no next choice"
            raise ValueError(msg)
        expected = state.pending[0]
        if context.path != expected:
            msg = f"the next diffusion path is {expected!r}, not {context.path!r}"
            raise ValueError(msg)
        slot = self._path_slots[expected]
        stream = self._denoise(
            state.vectors,
            state.weights,
            state.revealed,
            self._step(state.filled, 1, device=state.vectors.device),
        )
        rows = self._choice_row(stream[:, slot], (context,), (slot,))
        mask = None if rows.mask is None else rows.mask[0]
        return LogDistribution(rows.tensor[0], ("choice",), mask, "choice")

    def fill(
        self, state: _DiffusionFrontier, /, *, choices: Tensor
    ) -> _DiffusionFrontier:
        """Return the frontier after revealing one payload-aware choice."""
        if choices.ndim != 0:
            msg = f"one frontier choice is scalar, not shape {tuple(choices.shape)}"
            raise ValueError(msg)
        choice = int(choices.item())
        if not ChoiceIndex.PAD < choice < self._choices.width:
            msg = f"choice {choice} lies outside [1, {self._choices.width})"
            raise ValueError(msg)
        if not state.pending:
            msg = "a complete diffusion frontier cannot be filled"
            raise ValueError(msg)
        path = state.pending[0]
        slot = self._path_slots[path]
        if not self._slot_legal[slot][choice]:
            msg = (
                f"choice {choice} at path {path!r} exceeds the "
                "decoder's fixed term layout"
            )
            raise ValueError(msg)
        opened = tuple(
            (*path, side) for side in range(self._choices.operator(choice).arity)
        )
        if any(child not in self._path_slots for child in opened):
            msg = (
                f"choice {choice} at path {path!r} exceeds the "
                "decoder's fixed term layout"
            )
            raise ValueError(msg)
        pending = tuple(
            sorted((*state.pending[1:], *opened), key=lambda held: (len(held), held))
        )
        weights = state.weights.clone()
        weights[:, slot] = self._one_hot(choices.reshape(1))
        revealed = state.revealed.clone()
        revealed[:, slot] = True
        return _DiffusionFrontier(
            state.vectors, weights, revealed, pending, state.filled + 1
        )

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        """Return canonical-order teacher rows without exposing the next choice."""
        vectors, input_mask = self._placed(inputs)
        placed = self._beside(vectors, terms)
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=vectors.device,
        )
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        per_term: list[Tensor] = []

        for term in range(vectors.shape[0]):
            weights = vectors.new_zeros((1, self._codec.capacity, self._choices.width))
            revealed = torch.zeros(
                (1, self._codec.capacity), dtype=torch.bool, device=vectors.device
            )
            reported: dict[int, Tensor] = {}
            occupied = tuple(
                sorted(
                    (slot for slot, live in enumerate(placed.mask[term]) if bool(live)),
                    key=lambda slot: (len(self._paths[slot]), self._paths[slot]),
                )
            )

            for filled, slot in enumerate(occupied):
                stream = self._denoise(
                    vectors[term : term + 1],
                    weights,
                    revealed,
                    self._step(filled, 1, device=vectors.device),
                )
                row = self._choice_row(
                    stream[:, slot], (contexts[term][slot],), (slot,)
                )
                reported[slot] = row.tensor[0]
                weights = weights.clone()
                weights[:, slot] = self._one_hot(codes[term, slot].reshape(1))
                revealed = revealed.clone()
                revealed[:, slot] = True

            absent = vectors.new_full((self._choices.width,), -torch.inf)
            per_term.append(
                torch.stack(
                    tuple(
                        reported.get(slot, absent)
                        for slot in range(self._codec.capacity)
                    )
                )
            )

        tensor = torch.stack(per_term)
        live = placed.mask
        if input_mask is not None:
            live = live & input_mask.unsqueeze(1)
        tensor = tensor.masked_fill(~live.unsqueeze(2), -torch.inf)
        return LogDistribution(tensor, _CHOICE_AXES, live, normalized_over="choice")

    def site_map(self) -> SiteMap:
        """Return the published choice distribution and residual-stream sites."""
        return self._sites


def _build(
    settings: DiffusionTermDecoderSettings, context: BuildContext, /
) -> DiffusionTermDecoder:
    """Build and place a diffusion decoder denoising every slot of a term at once.

    Parameters
    ----------
    settings : DiffusionTermDecoderSettings
        The model dimensions and the masking schedule.
    context : BuildContext
        The codec, the wired encoder whose output width conditions the denoising,
        the component handle and the device policy.
    """
    encoder = cast("TermEncoder", context.producer("encoder"))

    return place(
        DiffusionTermDecoder(
            codec=context.codec,
            choices=context.choices,
            component=context.component,
            settings=DiffusionDecoderSettings(
                encoder.output_width,
                settings.width,
                settings.layers,
                settings.feedforward_width,
                settings.activation,
            ),
            schedule=NoiseSchedule.linear(
                settings.noise_steps, first=settings.first_masking
            ),
            choice_heads=settings.choice_heads,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=DiffusionTermDecoderSpec, build=_build)
"""The diffusion term decoder's plugin."""
