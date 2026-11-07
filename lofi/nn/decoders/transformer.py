"""Transformer decoding of linearized terms under a structural declaration."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, Protocol, assert_never, cast, final

import torch
from torch import Tensor, nn

from lofi.nn._contexts import slot_paths, teacher_contexts
from lofi.nn._guards import checked_state
from lofi.nn._layers import activation
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import NeuralCodecView, PlacedEncoding, reflexive_closure
from lofi.nn.heads import TermHeads
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
from lofi.nn.transformer import AttentionBlock, transformer_structure
from lofi.registry import ComponentPlugin
from lofi.spec.decoders.transformer import (
    TransformerTermDecoderSettings,
    TransformerTermDecoderSpec,
)
from lofi.terms import ChoiceIndex, HoleContext, Term

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.build import BuildContext
    from lofi.nn.codec import DeclaredStructure
    from lofi.nn.ports import StructuralDeclaration, TermEncoder
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import HeadKind
    from lofi.terms import BatchEncoding, Sort

__all__ = ["TransformerDecoderSettings", "TransformerTermDecoder"]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of an encoded batch of terms."""

_STREAM_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of one stream position per slot of each term."""

_CHOICE_AXES: Final[tuple[str, ...]] = ("term", "slot", "choice")
"""The axes of one payload-aware choice row per slot."""


class TransformerDecoderSettings(NamedTuple):
    """Transformer decoder dimensions.

    Attributes
    ----------
    input_width : int
        The width of the vector each term is encoded to.
    width : int
        The model and stream width.
    heads : int
        The number of attention heads.
    layers : int
        The number of Transformer layers.
    feedforward_width : int
        The hidden width of each feedforward block.
    activation : str
        The installed ``lofi.activation`` entry each feedforward block applies.
    """

    input_width: int
    width: int
    heads: int
    layers: int
    feedforward_width: int
    activation: str


def _decoder_sites(component: ComponentId) -> SiteMap:
    """Return the stream site published by a Transformer decoder.

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


class _Accessor(Protocol):
    """Reads one relation's tensor for a layout."""

    def __call__(self, *, policy: DevicePolicy) -> Tensor:
        """Return the relation's tensor, placed under a policy.

        Parameters
        ----------
        policy : DevicePolicy
            Where this run's tensors live.
        """


def _read(
    served: DeclaredStructure, relation: StructuralRelation, *, policy: DevicePolicy
) -> Tensor:
    """Return one relation's tensor for the layout.

    Parameters
    ----------
    served : DeclaredStructure
        The relations the decoder declared, read from its codec.
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
    accessors: Mapping[StructuralRelation, _Accessor] = {
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


def _held(relation: StructuralRelation, value: Tensor, what: str) -> Tensor:
    """Return a pairwise relation as the pairs it holds between.

    Parameters
    ----------
    relation : StructuralRelation
        The relation read.
    value : Tensor
        Its tensor, of rank two over the reading slot and the read slot.
    what : str
        The consumption being built, for the message.

    Returns
    -------
    Tensor
        The tensor, in boolean dtype.

    Raises
    ------
    ValueError
        If the relation is a count rather than one that holds or not.
    """
    if value.dtype is not torch.bool:
        msg = (
            f"{relation.value} counts edges between two slots, and {what} needs a "
            f"relation that holds between them or not"
        )
        raise ValueError(msg)

    return value


def _unary_mask(value: Tensor) -> Tensor:
    """Return the attention a unary relation admits.

    Parameters
    ----------
    value : Tensor
        The relation's value at each slot, of rank one.

    Returns
    -------
    Tensor
        True where the read slot's value is at most the reading slot's, of rank two
        over the reading slot and the read slot.
    """
    return value.unsqueeze(0).le(value.unsqueeze(1))


def _unary_feeds(value: Tensor) -> Tensor:
    """Return the slots a unary recurrence feeds each slot from.

    Parameters
    ----------
    value : Tensor
        The relation's value at each slot, of rank one.

    Returns
    -------
    Tensor
        True where the read slot's value is the greatest below the reading slot's,
        of rank two over the reading slot and the read slot.
    """
    below = value.unsqueeze(0).lt(value.unsqueeze(1))
    nearest = torch.where(below, value.unsqueeze(0), value.min() - 1).amax(dim=1)

    return below & value.unsqueeze(0).eq(nearest.unsqueeze(1))


def _levels(feeds: Tensor) -> Tensor:
    """Return the pass at which each slot is settled under a recurrence.

    Parameters
    ----------
    feeds : Tensor
        True where the reading slot is fed from the read slot, of rank two, with
        no chain of feeds returning to its start.

    Returns
    -------
    Tensor
        Zero at a slot fed from nowhere, and one more than the greatest level among
        the slots each other slot is fed from, of rank one in ``int64``.
    """
    fed = feeds.any(dim=1)
    levels = torch.zeros(feeds.shape[0], dtype=torch.int64, device=feeds.device)

    for _ in range(feeds.shape[0]):
        above = torch.where(feeds, levels.unsqueeze(0), -1).amax(dim=1) + 1
        levels = torch.where(fed, above, levels)

    return levels


def _admitted(attend: Tensor, feeds: Tensor, *, restricted: bool) -> Tensor:
    """Return where the reading slot attends to the read slot.

    Parameters
    ----------
    attend : Tensor
        True where a mask admits the reading slot to attend to the read slot, of
        rank two, true on the diagonal.
    feeds : Tensor
        True where the reading slot's input takes the choice made at the read
        slot, of rank two.
    restricted : bool
        Whether a mask or a recurrence was declared, so that attention is confined
        to what they admit rather than open to every slot.

    Returns
    -------
    Tensor
        True where the mask admits attention or where a chain of feeds reaches, of
        the mask's own shape and dtype, or true throughout when nothing restricts
        attention.

    Raises
    ------
    ValueError
        If a chain of feeds returns to its start, or if a slot is admitted to
        attend, through any chain of admitted attention, a slot fed its own choice.
    """
    eye = torch.eye(attend.shape[0], dtype=torch.bool, device=attend.device)
    reached = reflexive_closure(feeds)

    if bool((reached & reached.T & ~eye).any()):
        msg = (
            "a recurrence settles a slot after the slots it is fed from, and this "
            "declaration feeds two slots from each other"
        )
        raise ValueError(msg)

    admitted = (attend | reached) if restricted else torch.ones_like(eye)
    seen = reflexive_closure(admitted).to(torch.int64) @ feeds.to(torch.int64)
    exposed = seen.diagonal().nonzero()

    if exposed.numel():
        slot = int(exposed[0])
        msg = (
            f"a slot fed a choice made at another slot must not be read by that "
            f"slot, and this declaration admits slot {slot} to attend, through the "
            f"slots it is admitted to, a slot that is fed its own choice"
        )
        raise ValueError(msg)

    return admitted


class _Structure(NamedTuple):
    """What a declaration builds a Transformer decoder from.

    Attributes
    ----------
    embedded : tuple[tuple[StructuralRelation, Tensor], ...]
        The relations added to a slot's input, each as its value at each slot.
    biased : tuple[tuple[StructuralRelation, Tensor, int], ...]
        The relations biasing attention, each as the table row to read at each pair
        of slots and the number of rows its table holds.
    attend : Tensor
        True where the reading slot attends to the read slot.
    feeds : Tensor
        True where the reading slot's input takes the choice made at the read slot.
    """

    embedded: tuple[tuple[StructuralRelation, Tensor], ...]
    biased: tuple[tuple[StructuralRelation, Tensor, int], ...]
    attend: Tensor
    feeds: Tensor


class _TransformerFrontier(NamedTuple):
    """One encoded input, its filled choice weights and its open paths."""

    vectors: Tensor
    weights: Tensor
    pending: tuple[tuple[int, ...], ...]


def _structure(
    served: DeclaredStructure, *, capacity: int, policy: DevicePolicy
) -> _Structure:
    """Return what a declaration builds a decoder from.

    Parameters
    ----------
    served : DeclaredStructure
        The relations the decoder declared, read from its codec.
    capacity : int
        How many slots the layout holds.
    policy : DevicePolicy
        Where this run's tensors live.

    Raises
    ------
    ValueError
        If a pairwise relation is declared as an embedding, if a count is declared
        as a mask or a recurrence, if a recurrence makes two slots depend on each
        other, or if a mask admits a slot to attend, through any chain of admitted
        attention, a slot that is fed its own choice.
    """
    embedded: list[tuple[StructuralRelation, Tensor]] = []
    biased: list[tuple[StructuralRelation, Tensor, int]] = []
    eye = torch.eye(capacity, dtype=torch.bool, device=policy.device)
    attend = eye.clone()
    feeds = torch.zeros_like(eye)
    restricted = False

    for relation, consumption in served.declaration.items():
        value = _read(served, relation, policy=policy)
        unary = relation.kind is RelationKind.UNARY

        match consumption:
            case Consumption.EMBEDDING:
                if not unary:
                    msg = (
                        f"an embedding is one vector per slot, and {relation.value} "
                        f"is one value per pair of slots"
                    )
                    raise ValueError(msg)

                embedded.append((relation, value))
            case Consumption.MASK:
                restricted = True
                attend |= (
                    _unary_mask(value) if unary else _held(relation, value, "a mask")
                )
            case Consumption.BIAS:
                if unary:
                    difference = value.unsqueeze(1) - value.unsqueeze(0)
                    biased.append(
                        (relation, difference + capacity - 1, 2 * capacity - 1)
                    )
                elif value.dtype is torch.bool:
                    biased.append((relation, value.to(torch.int64), 2))
                else:
                    biased.append((relation, value, capacity))
            case Consumption.RECURRENCE:
                restricted = True
                feeds |= (
                    _unary_feeds(value)
                    if unary
                    else _held(relation, value, "a recurrence") & ~eye
                )
            case unreachable:
                assert_never(unreachable)

    admitted = _admitted(attend, feeds, restricted=restricted)

    return _Structure(tuple(embedded), tuple(biased), admitted, feeds)


@final
class TransformerTermDecoder(nn.Module):
    """A Transformer over the fixed slots of a term codec, shaped by a declaration.

    The structural declaration controls embeddings, attention masks and biases, and
    choice recurrences. Frontier and teacher-forced passes reveal only earlier
    choices in the fixed layout order; marginal rows feed each earlier row's mass
    through the same backbone.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    choices : ChoiceIndex
        The language-owned payload-aware choice alphabet.
    component : ComponentId
        The component identifier assigned by the run.
    settings : TransformerDecoderSettings
        The model dimensions.
    structure : StructuralDeclaration
        The relations this decoder reads from the codec, each with how it uses it.

    Raises
    ------
    TypeError
        If the declaration is not keyed by relations with consumptions as values.
    ValueError
        If the declaration names a relation the codec does not serve, if it names a
        consumption this decoder cannot build from the relation, or if it lets a slot
        read its own choice.
    """

    _mask: Tensor
    _bias_rows: Tensor
    _features: Tensor
    _feeds: Tensor
    _fed: Tensor
    _levels: Tensor

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        component: ComponentId,
        settings: TransformerDecoderSettings,
        structure: StructuralDeclaration,
        choice_heads: HeadKind,
    ) -> None:
        super().__init__()

        served = codec.serve(structure)
        skeleton = codec.codec.skeleton
        if skeleton is None:
            msg = "a Transformer decoder requires a fixed path-addressed layout"
            raise ValueError(msg)
        if choices.signature is not codec.codec.index.signature:
            msg = "a Transformer decoder's choices and codec share one signature"
            raise ValueError(msg)
        input_width = settings.input_width
        model_width = settings.width
        head_count = settings.heads
        layer_count = settings.layers
        inner_width = settings.feedforward_width
        self._choices_table: nn.Embedding | None = nn.Embedding(
            choices.width, model_width, padding_idx=ChoiceIndex.PAD
        )
        self._start = nn.Parameter(torch.zeros(model_width))
        self._project = nn.Linear(input_width, model_width)
        self._layers = nn.ModuleList(
            AttentionBlock(
                model_width, head_count, inner_width, activation(settings.activation)
            )
            for _ in range(layer_count)
        )
        self._norm = nn.LayerNorm(model_width)
        self._choice_heads = TermHeads(
            index=choices,
            input_width=model_width,
            kind=choice_heads,
        )
        policy = DevicePolicy(self._project.weight.device, self._project.weight.dtype)
        built = _structure(served, capacity=codec.capacity, policy=policy)

        if not bool(built.feeds.any()):
            self._choices_table = None

        self._embeddings = nn.ModuleList(
            nn.Embedding(codec.capacity, model_width) for _ in built.embedded
        )
        self._bias_tables = nn.ParameterList(
            nn.Parameter(torch.zeros(head_count, rows, dtype=policy.dtype))
            for _, _, rows in built.biased
        )
        zero = torch.zeros((), dtype=policy.dtype, device=policy.device)
        refused = torch.full((), -torch.inf, dtype=policy.dtype, device=policy.device)
        self.register_buffer("_mask", torch.where(built.attend, zero, refused))
        self.register_buffer(
            "_bias_rows",
            torch.stack([rows for _, rows, _ in built.biased])
            if built.biased
            else torch.zeros(
                (0, codec.capacity, codec.capacity),
                dtype=torch.int64,
                device=policy.device,
            ),
        )
        self.register_buffer(
            "_features",
            torch.stack([value for _, value in built.embedded])
            if built.embedded
            else torch.zeros(
                (0, codec.capacity), dtype=torch.int64, device=policy.device
            ),
        )
        self.register_buffer("_feeds", built.feeds.to(policy.dtype))
        self.register_buffer("_fed", built.feeds.any(dim=1))
        self.register_buffer("_levels", _levels(built.feeds))
        self._codec = codec
        self._choices = choices
        self._structure = served.declaration
        self._recurrent = bool(built.feeds.any())
        self._sites = _decoder_sites(component)
        self._input_width = input_width
        self._width = model_width
        self._heads = head_count
        self._paths = slot_paths(skeleton)
        self._path_slots = MappingProxyType(
            {path: slot for slot, path in enumerate(self._paths)}
        )
        self._order = tuple(
            sorted(
                range(codec.capacity),
                key=lambda slot: (len(self._paths[slot]), self._paths[slot]),
            )
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
    def structure(self) -> StructuralDeclaration:
        """Return the relations this decoder reads from its codec, and how."""
        return self._structure

    @property
    def attention_mask(self) -> Tensor:
        """Return the additive mask each slot attends under, before any bias."""
        return self._mask

    @property
    def recurrent(self) -> bool:
        """Return whether some slot is fed the choice made at another."""
        return self._recurrent

    @property
    def input_width(self) -> int:
        """Return the width of the vector this decoder reads a term from."""
        return self._input_width

    @property
    def width(self) -> int:
        """Return the model and stream width."""
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

    def _attention(self) -> Tensor:
        """Return the term added to the attention weights of every layer.

        Returns
        -------
        Tensor
            The mask plus every declared bias, of rank four over the term, the head,
            the reading slot and the read slot, of size one over the term, and over
            the head as well when no bias is declared.
        """
        biased = self._mask.reshape(1, 1, *self._mask.shape)

        for table, rows in zip(self._bias_tables, self._bias_rows, strict=True):
            biased = biased + table[:, rows].unsqueeze(0)

        return biased

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
                            "marginal Transformer rows are undefined below a "
                            "binding site"
                        )
                        raise ValueError(msg)
                    sorts = frozenset(
                        operator.arg_sorts[side] for operator in operators
                    )
                    live = bool(sorts)
                    if len(sorts) > 1:
                        msg = (
                            "a marginal Transformer decoder needs one fixed sort at "
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

    def _choice_row(
        self,
        states: Tensor,
        contexts: tuple[HoleContext, ...],
        slots: tuple[int, ...],
        /,
    ) -> LogDistribution:
        """Return structurally legal normalized choice rows for slot states."""
        row = self._choice_heads(states, contexts)
        structural = torch.tensor(
            tuple(self._slot_legal[slot] for slot in slots),
            dtype=torch.bool,
            device=states.device,
        )
        legal = structural if row.mask is None else structural & row.mask
        if bool((~legal.any(dim=1)).any()):
            msg = "a Transformer decoder reached a slot with no admissible choice"
            raise ValueError(msg)
        values = torch.log_softmax(row.tensor.masked_fill(~legal, -torch.inf), dim=1)
        return LogDistribution(values, ("term", "choice"), legal, "choice")

    def _pass(self, vectors: Tensor, weights: Tensor) -> Tensor:
        """Return the stream one assignment of choice feedback gives.

        Parameters
        ----------
        vectors : Tensor
            One vector per term, of rank two over the term and the width.
        weights : Tensor
            The weight each choice has at each slot as what that slot feeds the
            slots it is a recurrence dependency of, of rank three over the term,
            the slot and the choice.

        Returns
        -------
        Tensor
            The stream, of rank three over the term, the slot and the width.
        """
        if self._choices_table is None:
            source = self._start.expand(
                vectors.shape[0], self._codec.capacity, self._width
            )
        else:
            fed = self._feeds.unsqueeze(0) @ (weights @ self._choices_table.weight)
            source = torch.where(
                self._fed.reshape(1, -1, 1), fed, self._start.reshape(1, 1, -1)
            )

        for table, values in zip(self._embeddings, self._features, strict=True):
            source = source + table(values).unsqueeze(0)

        stream = source + self._project(vectors).unsqueeze(1)
        attention = self._attention()

        for layer in self._layers:
            stream = layer(stream, attention)

        return self._norm(stream)

    def _one_hot(self, choices: Tensor) -> Tensor:
        """Return the weight each choice has when one choice per slot is given.

        Parameters
        ----------
        choices : Tensor
            The choice at each slot, of rank two over the term and the slot.

        Returns
        -------
        Tensor
            One at the given choice of each slot and zero elsewhere, of rank three over
            the term, the slot and the choice, at this decoder's own precision.
        """
        policy = self._policy()
        held = torch.zeros(
            (*choices.shape, self._choices.width),
            dtype=policy.dtype,
            device=policy.device,
        )

        return held.scatter(-1, choices.unsqueeze(-1), 1.0)

    def states(self, encoded: Encoded, /, *, terms: BatchEncoding) -> Encoded:
        """Return the stream at each slot under the choices a term supplies.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.
        terms : BatchEncoding
            The linearized terms whose choices the transformer feeds, in the order of
            the term axis.

        Returns
        -------
        Encoded
            Rank 3 over the axes ``("term", "slot", "width")``, real valued, masked as
            the encoded terms are.

        Raises
        ------
        ValueError
            If the encoded terms are shaped wrong, or if there is not one
            linearization per encoded term.
        """
        vectors, mask = self._placed(encoded)
        self._beside(vectors, terms)
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=vectors.device,
        )
        stream = self._pass(vectors, self._one_hot(codes))

        return Encoded(stream, _STREAM_AXES, mask)

    def admits(self, term: Term, /) -> bool:
        """Return whether a completed term belongs to this decoder's language."""
        return term.op.sort in self._choices.signature.sorts

    def start(self, inputs: Encoded, /) -> _TransformerFrontier:
        """Return one frontier with no choice revealed."""
        vectors, mask = self._placed(inputs)
        if vectors.shape[0] != 1:
            msg = (
                "one Transformer frontier starts from one input, not "
                f"{vectors.shape[0]}"
            )
            raise ValueError(msg)
        if mask is not None and not bool(mask[0]):
            msg = "one Transformer frontier starts from an unmasked input"
            raise ValueError(msg)
        weights = vectors.new_zeros((1, self._codec.capacity, self._choices.width))
        return _TransformerFrontier(vectors, weights, ((),))

    def weigh(
        self, state: _TransformerFrontier, /, *, context: HoleContext
    ) -> LogDistribution:
        """Return the row at the next slot under choices already filled."""
        if not state.pending:
            msg = "a complete Transformer frontier has no next choice"
            raise ValueError(msg)
        expected = state.pending[0]
        if context.path != expected:
            msg = f"the next Transformer path is {expected!r}, not {context.path!r}"
            raise ValueError(msg)
        slot = self._path_slots[expected]
        stream = self._pass(state.vectors, state.weights)
        row = self._choice_row(stream[:, slot], (context,), (slot,))
        return LogDistribution(
            row.tensor[0],
            ("choice",),
            None if row.mask is None else row.mask[0],
            "choice",
        )

    def fill(
        self, state: _TransformerFrontier, /, *, choices: Tensor
    ) -> _TransformerFrontier:
        """Return the frontier after revealing one payload-aware choice."""
        if choices.ndim != 0:
            msg = f"one frontier choice is scalar, not shape {tuple(choices.shape)}"
            raise ValueError(msg)
        choice = int(choices.item())
        if not ChoiceIndex.PAD < choice < self._choices.width:
            msg = f"choice {choice} lies outside [1, {self._choices.width})"
            raise ValueError(msg)
        if not state.pending:
            msg = "a complete Transformer frontier cannot be filled"
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
        return _TransformerFrontier(state.vectors, weights, pending)

    def rows(self, inputs: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        """Return teacher-forced choice rows without exposing later choices."""
        vectors, input_mask = self._placed(inputs)
        placed = self._beside(vectors, terms)
        codes = torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=vectors.device,
        )
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        weights = vectors.new_zeros(
            (vectors.shape[0], self._codec.capacity, self._choices.width)
        )
        reported: list[tuple[int, Tensor]] = []

        for slot in self._order:
            stream = self._pass(vectors, weights)
            row = self._choice_row(
                stream[:, slot],
                tuple(per_term[slot] for per_term in contexts),
                (slot,) * vectors.shape[0],
            )
            reported.append((slot, row.tensor))
            live = placed.mask[:, slot]
            shown = self._one_hot(codes[:, slot])
            weights = weights.clone()
            weights[:, slot] = torch.where(
                live.unsqueeze(1), shown, torch.zeros_like(shown)
            )

        tensor = torch.stack(
            tuple(row for _slot, row in sorted(reported, key=lambda item: item[0])),
            dim=1,
        )
        live = placed.mask
        if input_mask is not None:
            live = live & input_mask.unsqueeze(1)
        tensor = tensor.masked_fill(~live.unsqueeze(2), -torch.inf)
        return LogDistribution(tensor, _CHOICE_AXES, live, normalized_over="choice")

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: Sort) -> LogDistribution:
        """Return choice rows after feeding earlier rows' expected embeddings."""
        vectors, input_mask = self._placed(inputs)
        contexts, reachable = self._fixed_contexts(root_sort)
        weights = vectors.new_zeros(
            (vectors.shape[0], self._codec.capacity, self._choices.width)
        )
        reported: list[tuple[int, LogDistribution]] = []

        for slot in self._order:
            if not reachable[slot]:
                values = vectors.new_full(
                    (vectors.shape[0], self._choices.width), -torch.inf
                )
                mask = torch.zeros_like(values, dtype=torch.bool)
                reported.append(
                    (slot, LogDistribution(values, ("term", "choice"), mask, "choice"))
                )
                continue
            stream = self._pass(vectors, weights)
            row = self._choice_row(
                stream[:, slot],
                (contexts[slot],) * vectors.shape[0],
                (slot,) * vectors.shape[0],
            )
            reported.append((slot, row))
            weights = weights.clone()
            weights[:, slot] = row.tensor.exp()

        ordered = tuple(
            row for _slot, row in sorted(reported, key=lambda item: item[0])
        )
        tensor = torch.stack(tuple(row.tensor for row in ordered), dim=1)
        mask = torch.stack(
            tuple(
                row.mask
                if row.mask is not None
                else torch.ones_like(row.tensor, dtype=torch.bool)
                for row in ordered
            ),
            dim=1,
        )
        if input_mask is not None:
            mask &= input_mask.reshape(-1, 1, 1)
            tensor = tensor.masked_fill(~input_mask.reshape(-1, 1, 1), -torch.inf)
        return LogDistribution(tensor, _CHOICE_AXES, mask, normalized_over="choice")

    def site_map(self) -> SiteMap:
        """Return the published choice distribution and residual-stream sites."""
        return self._sites


def _build(
    settings: TransformerTermDecoderSettings, context: BuildContext, /
) -> TransformerTermDecoder:
    """Build and place a Transformer decoder attending over the slots it fills.

    Parameters
    ----------
    settings : TransformerTermDecoderSettings
        The model dimensions and declared attention structure.
    context : BuildContext
        The codec, the wired encoder whose output width the stream is projected
        from, the component handle and the device policy.
    """
    encoder = cast("TermEncoder", context.producer("encoder"))

    return place(
        TransformerTermDecoder(
            codec=context.codec,
            choices=context.choices,
            component=context.component,
            settings=TransformerDecoderSettings(
                encoder.output_width,
                settings.width,
                settings.heads,
                settings.layers,
                settings.feedforward_width,
                settings.activation,
            ),
            structure=transformer_structure(settings.structure),
            choice_heads=settings.choice_heads,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=TransformerTermDecoderSpec, build=_build)
"""The Transformer term decoder's plugin."""
