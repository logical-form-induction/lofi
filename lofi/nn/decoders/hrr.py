"""Holographic decoding of linearized terms by unbinding addresses from a codebook."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import torch
from torch import Tensor, nn

from lofi.nn._contexts import slot_paths, teacher_contexts
from lofi.nn._guards import checked_state
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
from lofi.nn.vsa import unbind
from lofi.registry import ComponentPlugin
from lofi.spec.decoders.hrr import (
    HrrTermDecoderSettings,
    HrrTermDecoderSpec,
)
from lofi.terms import ChoiceIndex, HoleContext, Term

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.build import BuildContext
    from lofi.nn.codec import NeuralCodecView, PlacedEncoding
    from lofi.nn.encoders import HolographicCodebook, HrrTermEncoder
    from lofi.nn.ports import StructuralDeclaration
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import HeadKind
    from lofi.terms import BatchEncoding, Sort

__all__ = ["HolographicTermDecoder"]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of an encoded batch of terms."""

_UNBOUND_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of the vector recovered at each slot of each term."""

_CHOICE_AXES: Final[tuple[str, ...]] = ("term", "slot", "choice")
"""The axes of one choice row per fixed-layout slot."""

_STRUCTURE: Final[StructuralDeclaration] = MappingProxyType(
    {StructuralRelation.POSITION: Consumption.EMBEDDING}
)
"""The one relation read from the codec: the position of a slot, as the address
whose role the slot is unbound with."""


def _decoder_sites(component: ComponentId) -> SiteMap:
    """Return the unbound-vector site published by a holographic decoder.

    Parameters
    ----------
    component : ComponentId
        The component identifier assigned by the run.

    Returns
    -------
    SiteMap
        The choice distribution and the unbound vectors it is read from.
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
            Site(component, Role.UNBOUND_VECTOR): SiteSpec(
                _UNBOUND_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
        },
    )


class _HolographicFrontier(NamedTuple):
    """Unbound slot states and the fixed-layout paths still open."""

    states: Tensor
    pending: tuple[tuple[int, ...], ...]


class HolographicTermDecoder(nn.Module):
    """A teacher-forced decoder that unbinds each linearized term slot.

    Each slot is unbound with its position role and read through the language's shared
    payload-aware choice heads. This common implementation works with both fixed and
    term-dependent layouts. A fixed path-addressed layout additionally selects the
    private frontier-and-marginal specialization through the component plugin.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    component : ComponentId
        The component identifier assigned by the run.
    choices : ChoiceIndex
        The language-owned payload-aware choice alphabet.
    codebook : HolographicCodebook
        The codes and roles the matching encoder binds with, held as a submodule.

    Raises
    ------
    ValueError
        If the codebook holds a number of codes other than the codec's symbol count
        or a number of roles other than the codec's capacity.
    """

    _addresses: Tensor

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        codebook: HolographicCodebook,
        choice_heads: HeadKind,
    ) -> None:
        super().__init__()

        if codebook.codes.count != codec.n_symbols:
            msg = (
                f"the codebook holds {codebook.codes.count} codes beside a codec "
                f"of {codec.n_symbols} symbols"
            )
            raise ValueError(msg)

        if codebook.roles.count != codec.capacity:
            msg = (
                f"the codebook holds {codebook.roles.count} roles beside a codec "
                f"of {codec.capacity} slots"
            )
            raise ValueError(msg)

        if choices.signature is not codec.codec.index.signature:
            msg = "a holographic decoder's choices and codec share one signature"
            raise ValueError(msg)

        self._codebook = codebook
        policy = self._policy()
        self.register_buffer(
            "_addresses",
            codec.serve(_STRUCTURE).position(policy=policy),
            persistent=False,
        )
        self._heads = TermHeads(
            index=choices, input_width=codebook.width, kind=choice_heads
        )
        self._codec = codec
        self._choices = choices
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
        self._sites = _decoder_sites(component)

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view supplied at construction."""
        return self._codec

    @property
    def codebook(self) -> HolographicCodebook:
        """Return the codes and roles this decoder unbinds against."""
        return self._codebook

    @property
    def input_width(self) -> int:
        """Return the width of the vector this decoder reads a term from."""
        return self._codebook.width

    @property
    def structure(self) -> StructuralDeclaration:
        """Return the relations this decoder reads from its codec, and how."""
        return _STRUCTURE

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the constructor codes.

        Returns
        -------
        DevicePolicy
            Where the codebook's atoms live, and at what precision.
        """
        atoms = self._codebook.roles.atoms

        return DevicePolicy(atoms.device, atoms.dtype)

    def _placed(self, encoded: Encoded) -> tuple[Tensor, Tensor | None]:
        """Return the encoded terms beside the codebook, and their mask.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.

        Returns
        -------
        tuple[Tensor, Tensor | None]
            The vectors, of rank two over the term and the width, at the codebook's
            device and precision, and the mask over the term, or None where every
            row holds a term.

        Raises
        ------
        ValueError
            If the value does not run over the term and the width, runs over a width
            other than the atoms' dimension, or is masked over an axis other than
            the term.
        """
        held = checked_state(encoded, self.input_width, "decoder")

        if held.axes != _TERM_AXES:
            msg = f"an encoded batch of terms runs over {_TERM_AXES}, not {held.axes}"
            raise ValueError(msg)

        mask = held.mask

        if mask is not None and mask.ndim != 1:
            msg = f"an encoded batch of terms is masked over the term, not {held.axes}"
            raise ValueError(msg)

        policy = self._policy()

        return (held.tensor.to(device=policy.device, dtype=policy.dtype), mask)

    def _placed_terms(self, terms: BatchEncoding, count: int) -> PlacedEncoding:
        """Return a batch of terms as placed channels.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms, in the order of the term axis.
        count : int
            How many encoded terms the batch lies beside.

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

        if placed.term_count != count:
            msg = (
                f"{placed.term_count} linearized terms arrived beside "
                f"{count} encoded ones"
            )
            raise ValueError(msg)

        return placed

    def _unbound(self, vectors: Tensor) -> Tensor:
        """Return the vector recovered at each slot of each term.

        Parameters
        ----------
        vectors : Tensor
            One vector per term, of rank two over the term and the width.

        Returns
        -------
        Tensor
            Each term unbound with the role at each slot's address, of rank three
            over the term, the slot and the width.
        """
        roles = self._codebook.roles(self._addresses)

        return unbind(vectors.unsqueeze(1), roles.unsqueeze(0))

    def _choice_row(
        self,
        states: Tensor,
        contexts: tuple[HoleContext, ...],
        slots: tuple[int, ...],
        /,
    ) -> LogDistribution:
        """Return structurally legal normalized choice rows for slot states."""
        row = self._heads(states, contexts)
        structural = torch.tensor(
            tuple(self._slot_legal[slot] for slot in slots),
            dtype=torch.bool,
            device=states.device,
        )
        legal = structural if row.mask is None else structural & row.mask

        if bool((~legal.any(dim=1)).any()):
            msg = "a holographic decoder reached a slot with no admissible choice"
            raise ValueError(msg)

        values = torch.log_softmax(row.tensor.masked_fill(~legal, -torch.inf), dim=1)
        return LogDistribution(values, ("term", "choice"), legal, "choice")

    def unbound(self, encoded: Encoded, /) -> Encoded:
        """Return the vector recovered at each slot of each term.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("term", "slot", "width")``, real valued, masked
            as the encoded terms are.

        Raises
        ------
        ValueError
            If the encoded terms are shaped wrong.
        """
        vectors, mask = self._placed(encoded)

        return Encoded(self._unbound(vectors), _UNBOUND_AXES, mask)

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        """Return one payload-aware teacher row per linearized slot."""
        vectors, input_mask = self._placed(inputs)
        placed = self._placed_terms(terms, vectors.shape[0])
        states = self._unbound(vectors)
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        values: list[Tensor] = []

        for slot in range(self._codec.capacity):
            row = self._choice_row(
                states[:, slot],
                tuple(per_term[slot] for per_term in contexts),
                (slot,) * vectors.shape[0],
            )
            values.append(row.tensor)

        tensor = torch.stack(values, dim=1)
        live = placed.mask
        if input_mask is not None:
            live = live & input_mask.unsqueeze(1)
        tensor = tensor.masked_fill(~live.unsqueeze(2), -torch.inf)
        return LogDistribution(tensor, _CHOICE_AXES, live, normalized_over="choice")

    def site_map(self) -> SiteMap:
        """Return the published choice distribution and unbound-vector sites."""
        return self._sites


@final
class _PathAddressedHolographicTermDecoder(HolographicTermDecoder):
    """Add frontier and marginal decoding for a fixed path-addressed layout."""

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        codebook: HolographicCodebook,
        choice_heads: HeadKind,
    ) -> None:
        super().__init__(
            codec=codec,
            choices=choices,
            component=component,
            codebook=codebook,
            choice_heads=choice_heads,
        )
        skeleton = codec.codec.skeleton
        if skeleton is None:
            msg = "a path-addressed holographic decoder needs a fixed skeleton"
            raise ValueError(msg)
        self._paths = slot_paths(skeleton)
        self._path_slots = MappingProxyType(
            {path: slot for slot, path in enumerate(self._paths)}
        )

    def _fixed_contexts(
        self, root_sort: Sort, /
    ) -> tuple[tuple[HoleContext, ...], tuple[bool, ...]]:
        """Return fixed contexts and reachability under one requested root sort."""
        if root_sort not in self._choices.signature.sorts:
            msg = f"a marginal root names undeclared sort {root_sort!r}"
            raise ValueError(msg)
        empty: Mapping[str, int] = MappingProxyType({})
        contexts: list[HoleContext] = []
        reachable: list[bool] = []

        for path in self._paths:
            sort = root_sort
            live = True
            if path:
                parent_slot = self._path_slots[path[:-1]]
                live = reachable[parent_slot]
                if live:
                    parent_sort = contexts[parent_slot].sort
                    side = path[-1]
                    operators = tuple(
                        self._choices.operator(code)
                        for code, (structural, typed) in enumerate(
                            zip(
                                self._slot_legal[parent_slot],
                                self._choices.legal_choices[parent_sort],
                                strict=True,
                            )
                        )
                        if structural
                        and typed
                        and self._choices.operator(code).arity > side
                    )
                    if any(operator.scope(side) for operator in operators):
                        msg = (
                            "marginal holographic rows are undefined below a "
                            "binding site"
                        )
                        raise ValueError(msg)
                    sorts = frozenset(
                        operator.arg_sorts[side] for operator in operators
                    )
                    live = bool(sorts)
                    if len(sorts) > 1:
                        msg = (
                            "a marginal holographic decoder needs one fixed sort at "
                            f"path {path!r}, not {tuple(sorted(sorts))!r}"
                        )
                        raise ValueError(msg)
                    if live:
                        sort = next(iter(sorts))

            contexts.append(
                HoleContext(
                    path,
                    sort,
                    empty,
                    empty,
                    self._codec.capacity - 1,
                )
            )
            reachable.append(live)

        return tuple(contexts), tuple(reachable)

    def admits(self, term: Term, /) -> bool:
        """Return whether a completed term belongs to this decoder's language."""
        return term.op.sort in self._choices.signature.sorts

    def start(self, inputs: Encoded, /) -> _HolographicFrontier:
        """Return one frontier holding every unbound fixed-slot state."""
        vectors, mask = self._placed(inputs)
        if vectors.shape[0] != 1:
            msg = (
                "one holographic frontier starts from one input, not "
                f"{vectors.shape[0]}"
            )
            raise ValueError(msg)
        if mask is not None and not bool(mask[0]):
            msg = "one holographic frontier starts from an unmasked input"
            raise ValueError(msg)
        return _HolographicFrontier(self._unbound(vectors)[0], ((),))

    def weigh(
        self, state: _HolographicFrontier, /, *, context: HoleContext
    ) -> LogDistribution:
        """Return the choice distribution at the frontier's next path."""
        if not state.pending:
            msg = "a complete holographic frontier has no next choice"
            raise ValueError(msg)
        expected = state.pending[0]
        if context.path != expected:
            msg = f"the next holographic path is {expected!r}, not {context.path!r}"
            raise ValueError(msg)
        slot = self._path_slots[expected]
        row = self._choice_row(state.states[slot].unsqueeze(0), (context,), (slot,))
        return LogDistribution(
            row.tensor[0],
            ("choice",),
            None if row.mask is None else row.mask[0],
            "choice",
        )

    def fill(
        self, state: _HolographicFrontier, /, *, choices: Tensor
    ) -> _HolographicFrontier:
        """Advance one choice without changing independent unbound slot states."""
        if choices.ndim != 0:
            msg = f"one frontier choice is scalar, not shape {tuple(choices.shape)}"
            raise ValueError(msg)
        choice = int(choices.item())
        if not ChoiceIndex.PAD < choice < self._choices.width:
            msg = f"choice {choice} lies outside [1, {self._choices.width})"
            raise ValueError(msg)
        if not state.pending:
            msg = "a complete holographic frontier cannot be filled"
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
        return _HolographicFrontier(state.states, pending)

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: Sort) -> LogDistribution:
        """Return independent payload-aware rows at every fixed-layout slot."""
        vectors, input_mask = self._placed(inputs)
        states = self._unbound(vectors)
        contexts, reachable = self._fixed_contexts(root_sort)
        values: list[Tensor] = []
        masks: list[Tensor] = []

        for slot in range(self._codec.capacity):
            if not reachable[slot]:
                values.append(
                    states.new_full((vectors.shape[0], self._choices.width), -torch.inf)
                )
                masks.append(
                    torch.zeros(
                        (vectors.shape[0], self._choices.width),
                        dtype=torch.bool,
                        device=states.device,
                    )
                )
                continue
            row = self._choice_row(
                states[:, slot],
                (contexts[slot],) * vectors.shape[0],
                (slot,) * vectors.shape[0],
            )
            values.append(row.tensor)
            masks.append(
                row.mask
                if row.mask is not None
                else torch.ones_like(row.tensor, dtype=torch.bool)
            )

        tensor = torch.stack(values, dim=1)
        mask = torch.stack(masks, dim=1)
        if input_mask is not None:
            mask &= input_mask.reshape(-1, 1, 1)
            tensor = tensor.masked_fill(~input_mask.reshape(-1, 1, 1), -torch.inf)
        return LogDistribution(tensor, _CHOICE_AXES, mask, normalized_over="choice")


def _build(
    settings: HrrTermDecoderSettings, context: BuildContext, /
) -> HolographicTermDecoder:
    """Build and place a holographic decoder over its encoder's codebook.

    Parameters
    ----------
    settings : HrrTermDecoderSettings
        The decoder's settings, which hold nothing.
    context : BuildContext
        The codec, the wired encoder whose codebook the decoder unbinds with, the
        component handle and the device policy.
    """
    encoder = cast("HrrTermEncoder", context.producer("encoder"))

    implementation = (
        _PathAddressedHolographicTermDecoder
        if context.codec.path_addressed
        else HolographicTermDecoder
    )
    return place(
        implementation(
            codec=context.codec,
            choices=context.choices,
            component=context.component,
            codebook=encoder.codebook,
            choice_heads=settings.choice_heads,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=HrrTermDecoderSpec, build=_build)
"""The holographic term decoder's plugin."""
