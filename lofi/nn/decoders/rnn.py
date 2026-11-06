"""Recursive tree decoding of linearized terms."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final

import torch
from torch import Tensor, nn

from lofi.nn._contexts import teacher_contexts
from lofi.nn._guards import checked_state
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import DISTRIBUTION_AXES, NeuralCodecView, PlacedEncoding
from lofi.nn.heads import TermHeads
from lofi.nn.policy import DevicePolicy, place
from lofi.nn.ports import (
    Consumption,
    NoAdmissibleChoiceError,
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
from lofi.nn.type_directed import TermAdmission, term_admission
from lofi.registry import ComponentPlugin
from lofi.spec.decoders.rnn import (
    RnnTermDecoderSettings,
    RnnTermDecoderSpec,
)
from lofi.terms import BatchEncoding, ChoiceIndex, Encoding, HoleContext, Term

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.nn.build import BuildContext
    from lofi.nn.ports import StructuralDeclaration, TermEncoder
    from lofi.nn.sites import ComponentId
    from lofi.spec.layers import HeadKind
    from lofi.terms import Operator, Path, Sort

__all__ = [
    "RecursiveTermDecoder",
    "TreeDecoderSettings",
]

_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")
"""The axes of an encoded batch of terms."""

_STATE_AXES: Final[tuple[str, ...]] = ("term", "slot", "width")
"""The axes of one state per slot of each term."""

_CHOICE_DISTRIBUTION_AXES: Final = ("term", "slot", "choice")
"""The axis a slot's mass sums over."""

_STRUCTURE: Final[StructuralDeclaration] = MappingProxyType(
    {StructuralRelation.PARENT: Consumption.RECURRENCE}
)
"""The one relation the recurrence reads from its codec: the parent, as what each
slot's state is computed from."""


class _SlotStructure(NamedTuple):
    """Which slot each slot descends from, and at which argument position.

    Attributes
    ----------
    parents : tuple[int, ...]
        The slot each slot descends from, with the root sentinel at the root.
    sides : tuple[int, ...]
        The argument position each slot occupies under the slot it descends from.
    branching : int
        The most argument positions any slot has under it.
    """

    parents: tuple[int, ...]
    sides: tuple[int, ...]
    branching: int


def _slot_structure(codec: NeuralCodecView, *, policy: DevicePolicy) -> _SlotStructure:
    """Return the parent and argument position of every slot of a codec's layout.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    policy : DevicePolicy
        Where this run's tensors live.

    Returns
    -------
    _SlotStructure
        The parent of every slot, read from the parent relation the codec serves,
        and the argument position of every slot, which is its rank in slot order
        among the slots sharing its parent.

    Raises
    ------
    ValueError
        If the codec does not serve the parent relation, because which slot holds a
        child depends on the term rather than on the layout.
    """
    parent = codec.serve(_STRUCTURE).parent(policy=policy)
    parents = [Encoding.NO_PARENT] * codec.capacity
    sides = [0] * codec.capacity
    seen = [0] * codec.capacity

    for slot, row in enumerate(parent.tolist()):
        for above, held in enumerate(row):
            if held:
                parents[slot] = above
                sides[slot] = seen[above]
                seen[above] += 1

    return _SlotStructure(tuple(parents), tuple(sides), max(seen))


def _marginal_contexts(
    codec: NeuralCodecView,
    choices: ChoiceIndex,
    structure: _SlotStructure,
    /,
    *,
    root_sort: Sort,
) -> tuple[tuple[HoleContext, ...], tuple[bool, ...]]:
    """Return the choice-independent contexts under one requested root sort.

    Marginal rows require one sort at each slot before any parent choice is known.
    Binding is excluded because the scope at a descendant likewise depends on the
    parent choices being marginalized.
    """
    index = codec.codec.index

    if choices.signature is not index.signature:
        msg = "a marginal decoder's choices and codec must share one signature"
        raise ValueError(msg)

    if root_sort not in choices.signature.sorts:
        msg = f"the signature declares no root sort {root_sort!r}"
        raise ValueError(msg)

    legal = codec.codec.legal_symbols
    offered = tuple(
        tuple(
            index.operator(symbol)
            for symbol, admitted in enumerate(per_slot)
            if admitted and symbol != index.PAD
        )
        for per_slot in legal
    )
    paths: list[tuple[int, ...]] = [()] * codec.capacity
    contexts: list[HoleContext] = []
    reachable_operators: list[tuple[Operator, ...]] = []
    reachable_slots: list[bool] = []
    empty = MappingProxyType({})

    for slot in range(codec.capacity):
        parent = structure.parents[slot]
        expected = root_sort

        if parent == Encoding.NO_PARENT:
            sorts = frozenset({root_sort})
        elif not reachable_slots[parent]:
            sorts = frozenset()
        else:
            side = structure.sides[slot]
            paths[slot] = (*paths[parent], side)
            sorts = frozenset(
                operator.arg_sorts[side]
                for operator in reachable_operators[parent]
                if side < operator.arity
            )

        if len(sorts) > 1:
            msg = (
                f"a marginal tree decoder needs one fixed sort at slot {slot}, "
                f"not {tuple(sorted(sorts))!r}"
            )
            raise ValueError(msg)

        live = bool(sorts)
        if live:
            expected = next(iter(sorts))
            operators = tuple(
                operator for operator in offered[slot] if operator.sort == expected
            )
        else:
            operators = ()

        if any(operator.is_binder for operator in operators):
            msg = f"marginal rows over {root_sort!r} reach a binding operator"
            raise ValueError(msg)

        reachable_operators.append(operators)
        reachable_slots.append(live)

        contexts.append(
            HoleContext(
                paths[slot],
                expected,
                empty,
                empty,
                codec.capacity - 1,
            )
        )

    return tuple(contexts), tuple(reachable_slots)


class TreeDecoderSettings(NamedTuple):
    """Tree decoder dimensions.

    Attributes
    ----------
    input_width : int
        The width of the vector each term is encoded to.
    width : int
        The width of the state the recursion passes between slots.
    """

    input_width: int
    width: int


class _TreePosition(NamedTuple):
    """One position of a frontier decode.

    Attributes
    ----------
    vector : Tensor
        The state at the slot from which the slots below it are read, of rank one
        over the width.
    path : tuple[int, ...]
        The argument path from the root to this frontier position.
    """

    vector: Tensor
    path: tuple[int, ...]


class _TreeFrontier(NamedTuple):
    """The next tree position and the positions that follow it."""

    current: _TreePosition | None
    pending: tuple[_TreePosition, ...]
    decisions: tuple[tuple[Path, int], ...]


def _decoder_sites(component: ComponentId) -> SiteMap:
    """Return the state site published by a tree decoder.

    Parameters
    ----------
    component : ComponentId
        The component identifier assigned by the run.

    Returns
    -------
    SiteMap
        The symbol distribution and the recurrent state it is read from.
    """
    return SiteMap(
        component,
        ComponentKind.TERM_DECODER,
        {
            Site(component, Role.SYMBOL_DISTRIBUTION): SiteSpec(
                DISTRIBUTION_AXES,
                DTypeClass.LOG_PROBABILITY,
                differentiable=True,
                access=Access.OBSERVE,
            ),
            Site(component, Role.DECODER_RECURRENT_STATE): SiteSpec(
                _STATE_AXES,
                DTypeClass.REAL,
                differentiable=True,
                access=Access.OBSERVE,
            ),
        },
    )


@final
class RecursiveTermDecoder(nn.Module):
    """A tree recurrence over the fixed slots of a term codec.

    Frontier and teacher-forced passes condition each child state on its parent's
    payload-aware choice and argument position. Marginal rows mix those child states
    under the complete parent-choice distribution.

    Parameters
    ----------
    codec : NeuralCodecView
        The grammar's tensor view of its term codec.
    choices : ChoiceIndex
        The language choices aligned with the grammar's operators.
    admission : TermAdmission
        The declared policy for completed decoder terms.
    component : ComponentId
        The component identifier assigned by the run.
    settings : TreeDecoderSettings
        The model dimensions.

    Raises
    ------
    ValueError
        If the codec does not serve the parent relation.
    """

    def __init__(
        self,
        *,
        codec: NeuralCodecView,
        choices: ChoiceIndex,
        admission: TermAdmission,
        component: ComponentId,
        settings: TreeDecoderSettings,
        choice_heads: HeadKind,
    ) -> None:
        super().__init__()

        input_width = settings.input_width
        width = settings.width
        self._choice_symbols = nn.Embedding(
            choices.width, width, padding_idx=ChoiceIndex.PAD
        )
        self._root = nn.Linear(input_width, width)
        self._choice_heads = TermHeads(
            index=choices,
            input_width=width,
            kind=choice_heads,
        )
        policy = DevicePolicy(self._root.weight.device, self._root.weight.dtype)
        structure = _slot_structure(codec, policy=policy)
        self._down = nn.ModuleList(
            nn.Linear(width, width) for _ in range(structure.branching)
        )
        self._codec = codec
        self._choices = choices
        self._admission = admission
        self._structure = structure
        marginal_slots: dict[str, tuple[tuple[HoleContext, ...], tuple[bool, ...]]] = {}
        marginal_refusals: dict[str, str] = {}

        for root_sort in choices.signature.sorts:
            try:
                marginal_slots[root_sort] = _marginal_contexts(
                    codec,
                    choices,
                    structure,
                    root_sort=root_sort,
                )
            except ValueError as error:
                marginal_refusals[root_sort] = str(error)

        self._marginal_slots: Mapping[
            str, tuple[tuple[HoleContext, ...], tuple[bool, ...]]
        ] = MappingProxyType(marginal_slots)
        self._marginal_refusals: Mapping[str, str] = MappingProxyType(marginal_refusals)
        self._sites = _decoder_sites(component)
        self._input_width = input_width
        self._width = width

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view supplied at construction."""
        return self._codec

    def admits(self, term: Term, /) -> bool:
        """Return whether one completed term satisfies the declared type policy."""
        return self._admission.admits(term)

    @property
    def input_width(self) -> int:
        """Return the width of the vector this decoder reads a term from."""
        return self._input_width

    @property
    def width(self) -> int:
        """Return the width of the state the recursion passes between slots."""
        return self._width

    @property
    def structure(self) -> StructuralDeclaration:
        """Return the relations this decoder reads from its codec, and how."""
        return _STRUCTURE

    def _policy(self) -> DevicePolicy:
        """Return placement derived from the root projection.

        Returns
        -------
        DevicePolicy
            Where this decoder's own parameters live, and at what precision.
        """
        return DevicePolicy(self._root.weight.device, self._root.weight.dtype)

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

    def _placed_conditioning_terms(
        self, terms: BatchEncoding, count: int
    ) -> PlacedEncoding:
        """Return validated teacher terms as placed channels.

        Parameters
        ----------
        terms : BatchEncoding
            The teacher encodings under this decoder's exact codec and layout.
        count : int
            How many encoded terms the teachers lie beside.

        Returns
        -------
        PlacedEncoding
            The validated channels, placed beside the decoder parameters.
        """
        self._validate_terms(terms)

        return self._placed_terms(terms, count)

    def _validate_terms(self, terms: BatchEncoding) -> None:
        """Raise unless terms belong to this decoder's exact codec layout.

        Parameters
        ----------
        terms : BatchEncoding
            The linearized terms to validate before recurrence.

        Raises
        ------
        TypeError
            If the batch, a row, a channel, or a channel value has the wrong type.
        ValueError
            If a channel has the wrong width, a row differs from the codec's fixed
            layout, or its symbols and occupancy do not form a term under this codec.
        """
        if not isinstance(terms, BatchEncoding):
            msg = f"teacher terms are a BatchEncoding, not {terms!r}"
            raise TypeError(msg)

        if not isinstance(terms.rows, tuple):
            msg = "teacher term rows are a tuple"
            raise TypeError(msg)

        skeleton = self._codec.codec.skeleton

        if skeleton is None:
            msg = "a recurrent decoder requires a fixed term layout"
            raise ValueError(msg)

        capacity = self._codec.capacity
        pad = self._codec.codec.index.PAD

        for term, row in enumerate(terms.rows):
            if not isinstance(row, Encoding):
                msg = f"teacher term {term} is an Encoding, not {row!r}"
                raise TypeError(msg)

            channels = (
                ("symbols", row.symbols, int),
                ("parents", row.parents, int),
                ("depths", row.depths, int),
                ("occupancy", row.occupied, bool),
            )

            for name, channel, expected_type in channels:
                if not isinstance(channel, tuple):
                    msg = f"teacher term {term} {name} are a tuple"
                    raise TypeError(msg)

                if len(channel) != capacity:
                    msg = (
                        f"teacher term {term} {name} cover {len(channel)} slots "
                        f"beside this codec's capacity of {capacity}"
                    )
                    raise ValueError(msg)

                for slot, value in enumerate(channel):
                    valid = isinstance(value, expected_type)

                    if expected_type is int:
                        valid = valid and not isinstance(value, bool)

                    if not valid:
                        msg = (
                            f"teacher term {term} {name} at slot {slot} has "
                            f"the wrong value {value!r}"
                        )
                        raise TypeError(msg)

            if row.payloads is not None:
                if not isinstance(row.payloads, tuple):
                    msg = f"teacher term {term} payloads are a tuple"
                    raise TypeError(msg)

                if len(row.payloads) != capacity:
                    msg = (
                        f"teacher term {term} payloads cover {len(row.payloads)} "
                        f"slots beside this codec's capacity of {capacity}"
                    )
                    raise ValueError(msg)

                for slot, payload in enumerate(row.payloads):
                    if not isinstance(payload, int) or isinstance(payload, bool):
                        msg = (
                            f"teacher term {term} payloads at slot {slot} have "
                            f"the wrong value {payload!r}"
                        )
                        raise TypeError(msg)

            if row.parents != skeleton.parents:
                msg = f"teacher term {term} parents do not match this codec's layout"
                raise ValueError(msg)

            if row.depths != skeleton.depths:
                msg = f"teacher term {term} depths do not match this codec's layout"
                raise ValueError(msg)

            for slot, symbol in enumerate(row.symbols):
                if not 0 <= symbol < self._codec.n_symbols:
                    msg = (
                        f"teacher term {term} slot {slot} holds symbol {symbol} "
                        f"beside this codec's alphabet of {self._codec.n_symbols}"
                    )
                    raise ValueError(msg)

                if not self._codec.codec.legal_symbols[slot][symbol]:
                    msg = (
                        f"teacher term {term} slot {slot} holds symbol {symbol}, "
                        "which this codec does not admit there"
                    )
                    raise ValueError(msg)

            expected = [False] * capacity
            expected[0] = True

            for slot in range(1, capacity):
                parent = self._structure.parents[slot]
                parent_symbol = row.symbols[parent]
                expected[slot] = (
                    expected[parent]
                    and self._structure.sides[slot] < self._codec.arities[parent_symbol]
                )

            for slot, (occupied, required, symbol) in enumerate(
                zip(row.occupied, expected, row.symbols, strict=True)
            ):
                if occupied != required:
                    msg = (
                        f"teacher term {term} occupancy at slot {slot} is "
                        f"{occupied}, but this codec requires {required}"
                    )
                    raise ValueError(msg)

                if occupied == (symbol == pad):
                    state = "occupied" if occupied else "empty"
                    msg = (
                        f"teacher term {term} slot {slot} is {state}, but holds "
                        f"symbol {symbol} under this codec"
                    )
                    raise ValueError(msg)

    def _slot_paths(self) -> tuple[tuple[int, ...], ...]:
        """Return each fixed slot's argument path."""
        paths: list[tuple[int, ...]] = [()] * self._codec.capacity

        for slot in range(1, self._codec.capacity):
            parent = self._structure.parents[slot]
            paths[slot] = (*paths[parent], self._structure.sides[slot])

        return tuple(paths)

    def _teacher_decisions(
        self, encoding: Encoding
    ) -> tuple[tuple[tuple[Path, int], ...], ...]:
        """Return the preceding frontier choices at every teacher slot."""
        codes = self._choices.codes(encoding)
        paths = self._slot_paths()
        occupied = tuple(
            sorted(
                (slot for slot, held in enumerate(encoding.occupied) if held),
                key=lambda slot: (len(paths[slot]), paths[slot]),
            )
        )
        decisions: tuple[tuple[Path, int], ...] = ()
        preceding = cast(
            "list[tuple[tuple[Path, int], ...]]",
            [decisions] * self._codec.capacity,
        )

        for slot in occupied:
            preceding[slot] = decisions
            decisions = (*decisions, (paths[slot], codes[slot]))

        return tuple(preceding)

    @staticmethod
    def _conditioned_choices(
        row: LogDistribution,
        admitted: Tensor,
        occupied: Tensor,
        /,
    ) -> Tensor:
        """Condition choice rows on their declared type-directed masks."""
        mask = admitted if row.mask is None else row.mask & admitted
        refused = occupied & ~mask.any(dim=1)

        if bool(refused.any()):
            positions = tuple(int(at) for at in refused.nonzero().reshape(-1).tolist())
            msg = f"type-directed teacher rows admit no choice for terms {positions!r}"
            raise ValueError(msg)

        values = row.tensor.masked_fill(~mask, -torch.inf)
        return torch.log_softmax(values, dim=1)

    def _teacher_states(self, vectors: Tensor, choices: Tensor) -> Tensor:
        """Return slot states conditioned on payload-aware teacher choices."""
        states = [torch.tanh(self._root(vectors))]

        for slot in range(1, self._codec.capacity):
            parent = self._structure.parents[slot]
            step = self._down[self._structure.sides[slot]]
            states.append(
                torch.tanh(
                    step(states[parent]) + self._choice_symbols(choices[:, parent])
                )
            )

        return torch.stack(states, dim=1)

    def _choice_codes(self, terms: BatchEncoding, /) -> Tensor:
        """Place each teacher's payload-aware choice codes."""
        return torch.tensor(
            tuple(self._choices.codes(row) for row in terms.rows),
            dtype=torch.int64,
            device=self._policy().device,
        )

    def rows(
        self,
        encoded: Encoded,
        /,
        *,
        terms: BatchEncoding,
    ) -> LogDistribution:
        """Return choice rows along the exact supplied teacher path.

        Parameters
        ----------
        encoded : Encoded
            One vector per term, over the axes ``("term", "width")``.
        terms : BatchEncoding
            One linearized teacher term per encoded term, under this decoder's codec
            and fixed layout.

        Returns
        -------
        LogDistribution
            Rank 3 over the axes ``("term", "slot", "choice")``, normalized over the
            choice axis, masked as the encoded terms are, with negative infinity at
            every slot of a masked term.

        Raises
        ------
        TypeError
            If the terms are malformed.
        ValueError
            If the encoded terms are shaped wrong, there is not one teacher per
            encoded term, or the teachers do not belong to the decoder's codec and
            layout.
        """
        vectors, mask = self._placed(encoded)
        placed = self._placed_conditioning_terms(terms, vectors.shape[0])
        choices = self._choice_codes(terms)
        states = self._teacher_states(vectors, choices)
        contexts = tuple(teacher_contexts(self._choices, row) for row in terms.rows)
        decisions = tuple(self._teacher_decisions(row) for row in terms.rows)
        held = []

        for slot in range(self._codec.capacity):
            row = self._choice_heads(
                states[:, slot],
                tuple(per_term[slot] for per_term in contexts),
            )
            admitted = torch.tensor(
                tuple(
                    self._admission.admitted_choices(
                        per_term[slot],
                        context[slot],
                        self._choices,
                    )
                    if term.occupied[slot]
                    else (True,) * self._choices.width
                    for per_term, context, term in zip(
                        decisions,
                        contexts,
                        terms.rows,
                        strict=True,
                    )
                ),
                dtype=torch.bool,
                device=row.tensor.device,
            )
            fits = torch.tensor(
                tuple(
                    tuple(
                        True
                        if not term.occupied[slot]
                        else choice != ChoiceIndex.PAD
                        and self._fits_at(context[slot].path, choice)
                        for choice in range(self._choices.width)
                    )
                    for context, term in zip(contexts, terms.rows, strict=True)
                ),
                dtype=torch.bool,
                device=row.tensor.device,
            )
            held.append(
                self._conditioned_choices(
                    row,
                    admitted & fits,
                    placed.mask[:, slot],
                )
            )

        rows = torch.stack(tuple(held), dim=1)

        if mask is not None:
            rows = rows.masked_fill(~mask.reshape(-1, 1, 1), -torch.inf)

        return LogDistribution(
            rows, _CHOICE_DISTRIBUTION_AXES, mask, normalized_over="choice"
        )

    def _marginal_row(
        self,
        states: Tensor,
        /,
        *,
        context: HoleContext,
    ) -> LogDistribution:
        """Return the payload-aware row at one choice-independent slot."""
        row = self._choice_heads(states, (context,) * states.shape[0])
        admitted = torch.tensor(
            self._admission.admitted_choices((), context, self._choices),
            dtype=torch.bool,
            device=row.tensor.device,
        ).expand(states.shape[0], -1)
        fits = torch.tensor(
            tuple(
                choice != ChoiceIndex.PAD and self._fits_at(context.path, choice)
                for choice in range(self._choices.width)
            ),
            dtype=torch.bool,
            device=row.tensor.device,
        ).expand(states.shape[0], -1)
        mask = admitted & fits

        if row.mask is not None:
            mask &= row.mask

        refused = ~mask.any(dim=1)

        if bool(refused.any()):
            positions = tuple(int(at) for at in refused.nonzero().reshape(-1))
            msg = f"marginal rows admit no choice for terms {positions!r}"
            raise NoAdmissibleChoiceError(msg)

        values = torch.log_softmax(row.tensor.masked_fill(~mask, -torch.inf), dim=1)
        return LogDistribution(values, ("term", "choice"), mask, "choice")

    def marginal_rows(self, inputs: Encoded, /, *, root_sort: Sort) -> LogDistribution:
        """Return payload-aware rows after mixing over every parent choice.

        The requested root sort fixes one context per layout slot. At a child, the
        recurrent state is the probability-weighted mixture of the states induced by
        every payload-aware parent choice, so constants sharing an operator symbol
        remain distinct throughout the recurrence.

        Raises
        ------
        ValueError
            If the signature declares no such root sort, or if marginalization from
            that sort reaches a binder or a choice-dependent child sort.
        """
        refusal = self._marginal_refusals.get(root_sort)

        if refusal is not None:
            raise ValueError(refusal)

        held = self._marginal_slots.get(root_sort)

        if held is None:
            msg = f"the signature declares no root sort {root_sort!r}"
            raise ValueError(msg)
        contexts, reachable = held

        vectors, input_mask = self._placed(inputs)
        root = torch.tanh(self._root(vectors))
        states = [root]
        rows = [self._marginal_row(root, context=contexts[0])]

        for slot in range(1, self._codec.capacity):
            parent = self._structure.parents[slot]
            if not reachable[slot]:
                states.append(states[parent])
                values = vectors.new_full(
                    (vectors.shape[0], self._choices.width), -torch.inf
                )
                mask = torch.zeros_like(values, dtype=torch.bool)
                rows.append(LogDistribution(values, ("term", "choice"), mask, "choice"))
                continue
            step = self._down[self._structure.sides[slot]]
            candidates = torch.tanh(
                step(states[parent]).unsqueeze(1)
                + self._choice_symbols.weight.unsqueeze(0)
            )
            state = torch.bmm(rows[parent].tensor.exp().unsqueeze(1), candidates)
            state = state.squeeze(1)
            states.append(state)
            rows.append(self._marginal_row(state, context=contexts[slot]))

        values = torch.stack(tuple(row.tensor for row in rows), dim=1)
        mask = torch.stack(
            tuple(
                row.mask
                if row.mask is not None
                else torch.ones_like(row.tensor, dtype=torch.bool)
                for row in rows
            ),
            dim=1,
        )

        if input_mask is not None:
            mask &= input_mask.reshape(-1, 1, 1)
            values = values.masked_fill(~input_mask.reshape(-1, 1, 1), -torch.inf)

        return LogDistribution(
            values,
            _CHOICE_DISTRIBUTION_AXES,
            mask,
            normalized_over="choice",
        )

    def start(self, encoded: Encoded, /) -> _TreeFrontier:
        """Return the root state of one term frontier.

        Parameters
        ----------
        encoded : Encoded
            One vector for the one term, over the axes ``("term", "width")``.

        Raises
        ------
        ValueError
            If the encoded terms are shaped wrong, if there is not exactly one of
            them, or if the mask holds no term.
        """
        vectors, mask = self._placed(encoded)

        if vectors.shape[0] != 1:
            msg = (
                f"a stepwise decode starts from one term, and {vectors.shape[0]} "
                f"were encoded"
            )
            raise ValueError(msg)

        if mask is not None and not bool(mask[0]):
            msg = "a stepwise decode starts from a term, and the mask holds none"
            raise ValueError(msg)

        root = _TreePosition(torch.tanh(self._root(vectors[0])), ())
        return _TreeFrontier(root, (), ())

    def weigh(
        self, state: _TreeFrontier, /, *, context: HoleContext
    ) -> LogDistribution:
        """Return the log probability of each choice at the next hole.

        Parameters
        ----------
        state : _TreeFrontier
            The frontier state reached so far.

        Returns
        -------
        LogDistribution
            Rank 1 over the axis ``("choice",)``, normalized over it, masked where the
            codec admits a symbol at the state's slot beneath the symbol taken above
            it, with negative infinity at every symbol it refuses there.
        """
        current = state.current

        if current is None:
            msg = "a complete decoder frontier has no next hole to weigh"
            raise ValueError(msg)

        row = self._choice_heads(current.vector.unsqueeze(0), (context,))
        admitted = torch.tensor(
            self._admission.admitted_choices(
                state.decisions,
                context,
                self._choices,
            ),
            dtype=torch.bool,
            device=row.tensor.device,
        )
        fits = torch.tensor(
            tuple(
                False
                if choice == ChoiceIndex.PAD
                else self._fits_at(current.path, choice)
                for choice in range(self._choices.width)
            ),
            dtype=torch.bool,
            device=row.tensor.device,
        )
        mask = admitted & fits

        if row.mask is not None:
            mask &= row.mask[0]

        if not bool(mask.any()):
            msg = f"type-directed decoding admits no choice at path {context.path!r}"
            raise NoAdmissibleChoiceError(msg)

        conditioned = row.tensor[0].masked_fill(~mask, -torch.inf)
        conditioned = torch.log_softmax(conditioned, dim=0)

        return LogDistribution(
            conditioned,
            ("choice",),
            mask,
            normalized_over="choice",
        )

    def _fits_at(self, path: Path, choice: int, /) -> bool:
        """Return whether a choice and all its children fit the fixed layout."""
        slot = 0

        try:
            for side in path:
                slot = self._codec.child_index(slot, side)

            for side in range(self._choices.arity(choice)):
                self._codec.child_index(slot, side)

        except IndexError:
            return False

        return True

    def _frontier_child(
        self, state: _TreePosition, choice: int, child: int
    ) -> _TreePosition:
        """Return one child state conditioned on a payload-aware choice."""
        if not 0 < choice < self._choices.width:
            msg = (
                f"the learner language holds {self._choices.width} choices, "
                f"not {choice}"
            )
            raise ValueError(msg)

        if not 0 <= child < len(self._down):
            msg = f"the decoder reaches {len(self._down)} argument sides, not {child}"
            raise ValueError(msg)

        step = self._down[child]

        return _TreePosition(
            torch.tanh(step(state.vector) + self._choice_symbols.weight[choice]),
            (*state.path, child),
        )

    def fill(self, state: _TreeFrontier, /, *, choices: Tensor) -> _TreeFrontier:
        """Return the frontier state after filling its next hole.

        Parameters
        ----------
        state : _TreeFrontier
            The frontier whose next position is filled.
        choices : Tensor
            One scalar choice id.

        Raises
        ------
        ValueError
            If the frontier is complete, the choice is not scalar, or the choice is
            outside this decoder's alphabet.
        """
        current = state.current

        if current is None:
            msg = "a complete decoder frontier has no hole to fill"
            raise ValueError(msg)

        if choices.ndim != 0:
            msg = f"one frontier choice is scalar, not of shape {tuple(choices.shape)}"
            raise ValueError(msg)

        choice = int(choices.item())

        if not 0 < choice < self._choices.width:
            msg = (
                f"the learner language holds {self._choices.width} choices, "
                f"not {choice}"
            )
            raise ValueError(msg)

        if not self._fits_at(current.path, choice):
            msg = (
                f"choice {choice} at path {current.path!r} exceeds the decoder's "
                "fixed term layout"
            )
            raise ValueError(msg)

        selected = self._choices.choice(choice)
        decisions = (*state.decisions, (current.path, choice))
        opened = tuple(
            self._frontier_child(current, choice, child)
            for child in range(selected.operator.arity)
        )
        pending = tuple(
            sorted(
                (*state.pending, *opened), key=lambda held: (len(held.path), held.path)
            )
        )

        if not pending:
            return _TreeFrontier(None, (), decisions)

        return _TreeFrontier(pending[0], pending[1:], decisions)

    def site_map(self) -> SiteMap:
        """Return the published symbol distribution and recurrent state sites."""
        return self._sites


def _build(
    settings: RnnTermDecoderSettings, context: BuildContext, /
) -> RecursiveTermDecoder:
    """Build and place a recursive decoder reading its encoder's terms down to slots.

    Parameters
    ----------
    settings : RnnTermDecoderSettings
        The width of the state the recursion passes between slots.
    context : BuildContext
        The codec, the wired encoder whose output width the recursion starts from,
        the component handle and the device policy.
    """
    encoder = cast("TermEncoder", context.producer("encoder"))

    return place(
        RecursiveTermDecoder(
            codec=context.codec,
            choices=context.choices,
            admission=term_admission(context.language, settings.type_directed),
            component=context.component,
            settings=TreeDecoderSettings(encoder.output_width, settings.width),
            choice_heads=settings.choice_heads,
        ),
        context.policy,
    )


PLUGIN: Final = ComponentPlugin(spec=RnnTermDecoderSpec, build=_build)
"""The recursive term decoder's plugin."""
