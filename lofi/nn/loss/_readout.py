"""Term distributions, complete terms and target masses read from chart cells."""

from enum import IntEnum
from types import MappingProxyType
from typing import (
    TYPE_CHECKING,
    Final,
    NamedTuple,
    Protocol,
    cast,
    override,
    runtime_checkable,
)

import torch

from lofi._checks import ensure_name
from lofi._types import ReadoutId
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.codec import DISTRIBUTION_AXES
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._decoding import decoding_strategy
from lofi.nn.loss._linearize import LinearizedTargets, linearize_targets
from lofi.nn.loss._mapping import SOFTMAX, Entmax
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._values import SpanReadout, SpanTerms
from lofi.nn.ports import (
    FrontierDecoder,
    MarginalDecoder,
    TeacherForcedDecoder,
)
from lofi.nn.readouts import ProofTermPushforward
from lofi.nn.strategies import (
    paired_exhaustive,
    score,
)
from lofi.registry import ComponentPlugin
from lofi.spec.readouts.terms import (
    DecodedMeaningSettings,
    MeaningInductionSettings,
    ProofTermMeaningSettings,
    ReadoutMass,
    ReadoutSource,
    TermsReadoutSettings,
    TermsReadoutSpec,
)
from lofi.terms import BatchEncoding
from lofi.typecheck import meaning_weighting

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.judge import Action
    from lofi.nn.build import BuildContext
    from lofi.nn.codec import NeuralCodecView
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.loss._parser import SpanMarginals, SplitPosterior
    from lofi.nn.spaces import Space
    from lofi.spec.decoding import DecodingSettings
    from lofi.terms import ChoiceIndex, Encoding, TargetSet, Term, TermCodec, TermStore


@runtime_checkable
class _RepresentedSpace(Protocol):
    """The represented-space identity a readout consumes at construction."""

    @property
    def sort(self) -> str:
        """Return the root sort represented by the space."""

    @property
    def codec(self) -> NeuralCodecView:
        """Return the exact codec view shared by the space's components."""


__all__ = [
    "CELL_AXES",
    "MASS_AXES",
    "TERMS_READOUT_AXES",
    "ReadoutMass",
    "TermsPublication",
    "TermsReadout",
    "log_complement",
    "log_mass",
]


class TermsPublication(NamedTuple):
    """One marginal readout's published values and payload-aware choice rows."""

    readouts: Mapping[ReadoutId, SpanReadout]
    choices: SpanReadout


CELL_AXES: Final[tuple[str, ...]] = (*SPAN_AXES, "width")
"""Per-cell vector axes."""

TERMS_READOUT_AXES: Final[tuple[str, ...]] = (*SPAN_AXES, *DISTRIBUTION_AXES[1:])
"""Per-cell type distribution axes."""

MASS_AXES: Final[tuple[str, ...]] = (*SPAN_AXES, "outcome")
"""Per-cell target-mass axes, the outcome axis holding one log probability pair."""

#: The axes a decoder reads one vector per term over.
_TERM_AXES: Final[tuple[str, ...]] = ("term", "width")


class _Outcome(IntEnum):
    """Positions along the outcome axis of a published mass pair."""

    #: The span reads as a member of the target set.
    TARGET = 0

    #: The span reads as anything else.
    ELSEWHERE = 1


class _Declared(NamedTuple):
    """One target set as the readout scores it.

    Attributes
    ----------
    key : ReadoutId
        The identifier the set's mass pair is published under.
    rows : tuple[Encoding, ...]
        The distinct linearized members, in declaration order.
    targets : TargetSet[Term]
        The symbolic members and matching that own this published mass.
    """

    key: ReadoutId
    rows: tuple[Encoding, ...]
    targets: TargetSet[Term]


def log_mass(pair: LogDistribution, /) -> Tensor:
    """Return the log mass of the target set from a published mass pair.

    Parameters
    ----------
    pair : LogDistribution
        A published mass readout's whole distribution, or one cell of it, over the
        outcome axis innermost.

    Returns
    -------
    Tensor
        The log mass, with the outcome axis removed.

    Raises
    ------
    ValueError
        If the value has no outcome axis.
    """
    return pair.tensor.select(pair.axis(MASS_AXES[-1]), _Outcome.TARGET)


def log_complement(pair: LogDistribution, /) -> Tensor:
    """Return the log mass everywhere but the target set from a published mass pair.

    Parameters
    ----------
    pair : LogDistribution
        A published mass readout's whole distribution, or one cell of it, over the
        outcome axis innermost.

    Returns
    -------
    Tensor
        The log of one minus the mass, with the outcome axis removed.

    Raises
    ------
    ValueError
        If the value has no outcome axis.
    """
    return pair.tensor.select(pair.axis(MASS_AXES[-1]), _Outcome.ELSEWHERE)


def _live_cells(mask: Tensor | None, cells: tuple[int, int]) -> Tensor | None:
    """Return which cells hold a vector, or None where every cell does.

    Parameters
    ----------
    mask : Tensor | None
        The mask the cells arrived with, over the item or over the item and the span,
        or None where every cell holds a vector.
    cells : tuple[int, int]
        How many items and how many spans there are.

    Returns
    -------
    Tensor | None
        A boolean tensor of rank 2 over the item and the span, or None.

    Raises
    ------
    ValueError
        If the mask reaches into a cell's features.
    """
    if mask is None:
        return None

    if mask.ndim > len(cells):
        msg = (
            f"a cell is decoded whole, and a mask of rank {mask.ndim} marks part of one"
        )
        raise ValueError(msg)

    widened = mask.reshape(tuple(mask.shape) + (1,) * (len(cells) - mask.ndim))

    return widened.expand(cells)


def _computed_rows(rows: LogDistribution) -> Tensor | None:
    """Return which decoded terms the decoder computed, or None where it computed all.

    Parameters
    ----------
    rows : LogDistribution
        What the decoder returned, one row of slots per term.

    Returns
    -------
    Tensor | None
        A boolean tensor of rank 1 over the terms, or None.

    Raises
    ------
    ValueError
        If the decoder masks whole slots rather than whole terms or individual
        choices.
    """
    if rows.mask is None:
        return None

    if rows.mask.ndim == 1:
        return rows.mask

    if rows.mask.ndim == 3:
        computed = rows.mask.any(dim=(1, 2))

        return None if bool(computed.all()) else computed

    msg = (
        "a cell is decoded whole, and the decoder masked part of one over "
        f"{rows.mask_axes}"
    )
    raise ValueError(msg)


def _canonical_choice_rows(rows: LogDistribution, pad: int, /) -> Tensor:
    """Fill every unreachable slot of a computed row with one-hot padding.

    Marginal decoders mask individual choices and may mask every choice at a fixed
    layout slot that the represented root sort cannot reach. A published span value
    is masked only cell by cell, so those inner slots need a normalized value of
    their own. Whole uncomputed terms remain negative infinity for the outer cell
    mask to exclude.
    """
    mask = rows.mask

    if mask is None or mask.ndim != 3:
        return rows.tensor

    computed = mask.any(dim=(1, 2))
    unreachable = computed.unsqueeze(1) & ~mask.any(dim=2)
    padding = rows.tensor.new_full(rows.tensor.shape, -torch.inf)
    padding[..., pad] = 0.0

    return torch.where(unreachable.unsqueeze(2), padding, rows.tensor)


def _declared(targets: Mapping[ReadoutId, LinearizedTargets]) -> tuple[_Declared, ...]:
    """Return the declared target sets with their members made distinct.

    Parameters
    ----------
    targets : Mapping[ReadoutId, LinearizedTargets]
        The linearized target sets, keyed by the id each is published under.

    Raises
    ------
    TypeError
        If an id is not a string.
    ValueError
        If an id is empty, or a set has no member.
    """
    declared: list[_Declared] = []

    for key, linearized in targets.items():
        rows = tuple(dict.fromkeys(linearized.encoded.rows))

        if not rows:
            msg = f"the target set {key!r} has no member to put mass on"
            raise ValueError(msg)

        declared.append(
            _Declared(
                ReadoutId(ensure_name("a readout id", key)),
                rows,
                linearized.targets,
            )
        )

    return tuple(declared)


def _log_of(probabilities: Tensor, held: Tensor) -> Tensor:
    """Return the log of the held probabilities, and negative infinity elsewhere.

    Parameters
    ----------
    probabilities : Tensor
        Values in the unit interval.
    held : Tensor
        True where a value is read, and false where the result is negative infinity
        regardless of the value.

    Returns
    -------
    Tensor
        The log where held and the value is positive, and negative infinity elsewhere,
        with a finite gradient everywhere.
    """
    positive = held & (probabilities > 0.0)
    logged = torch.log(torch.where(positive, probabilities, probabilities.new_ones(())))

    return torch.where(positive, logged, probabilities.new_full((), -torch.inf))


def _log_sum(weights: Tensor) -> Tensor:
    """Return the log of the summed exponentials along the innermost axis.

    Parameters
    ----------
    weights : Tensor
        Log weights, negative infinity where a weight is zero.

    Returns
    -------
    Tensor
        The log sum along the innermost axis, negative infinity where every weight
        is, with a finite gradient everywhere.
    """
    some = weights.gt(-torch.inf).any(dim=-1)
    safe = torch.where(some.unsqueeze(-1), weights, weights.new_zeros(()))

    return torch.where(some, safe.logsumexp(dim=-1), weights.new_full((), -torch.inf))


def _log_one_minus(log_mass: Tensor) -> Tensor:
    """Return the log of one minus the exponential of a log probability.

    Parameters
    ----------
    log_mass : Tensor
        Log probabilities, at most zero.

    Returns
    -------
    Tensor
        The log of the remaining mass, negative infinity where the log probability
        is zero, with a finite gradient everywhere.
    """
    below = log_mass < 0.0
    safe = torch.where(below, log_mass, log_mass.new_full((), -1.0))
    remaining = torch.log(-torch.expm1(safe))

    return torch.where(below, remaining, log_mass.new_full((), -torch.inf))


def _member_log_probabilities(rows: Tensor, target: _Declared) -> Tensor:
    """Return the log probability the published rows give each member at every cell.

    A member's probability is the product, over the slots its nodes fill, of the
    symbol probabilities the rows hold at those slots. A row that gives one of those
    symbols exactly zero mass therefore puts the member at negative infinity.

    Parameters
    ----------
    rows : Tensor
        The published log probabilities over the item, the span, the slot and the
        symbol.
    target : _Declared
        The set whose members are read off them.

    Returns
    -------
    Tensor
        The log probability over the item, the span and the member.

    Raises
    ------
    ValueError
        If a member has other slots than the rows, or a symbol the rows do not hold.
    """
    _, _, slots, symbols = rows.shape
    codes = torch.tensor(tuple(row.symbols for row in target.rows), device=rows.device)
    filled = torch.tensor(
        tuple(row.occupied for row in target.rows), device=rows.device
    )

    if codes.shape[1] != slots:
        msg = (
            f"a member of the target set {target.key!r} fills {codes.shape[1]} "
            f"slots, and the decoder reads {slots}"
        )
        raise ValueError(msg)

    if int(codes.max()) >= symbols:
        msg = (
            f"a member of the target set {target.key!r} names symbol "
            f"{int(codes.max())}, and the decoder reads {symbols}"
        )
        raise ValueError(msg)

    picked = rows[:, :, torch.arange(slots, device=rows.device), codes]

    return torch.where(filled, picked, rows.new_zeros(())).sum(dim=-1)


def _log_constituent_factor(
    marginals: SpanMarginals, spans: SpanIndex, items: int
) -> tuple[Tensor, Tensor]:
    """Return the log constituent factor of every cell, and where it is defined.

    The factor is one at every span of width one within an item's length and at
    the item's root span, the constituent marginal at every other span a derivation
    may build, and undefined elsewhere, a padded position beyond the item's length
    included.

    Parameters
    ----------
    marginals : SpanMarginals
        The chart's constituent marginals.
    spans : SpanIndex
        The addressing the readout is over.
    items : int
        How many items the readout covers.

    Returns
    -------
    tuple[Tensor, Tensor]
        The log factor over the item and the span, negative infinity where undefined,
        and a boolean tensor of the same shape that is true where it is defined.

    Raises
    ------
    ValueError
        If the marginals are over another span addressing or another number of items
        than the readout.
    """
    if marginals.spans != spans:
        msg = (
            f"the marginals are over {marginals.spans!r}, and the readout over "
            f"{spans!r}"
        )
        raise ValueError(msg)

    if marginals.items != items:
        msg = f"the marginals cover {marginals.items} items, and the readout {items}"
        raise ValueError(msg)

    probabilities = marginals.constituent_factor()
    defined = probabilities.mask

    if defined is None:
        msg = "a complete constituent factor names its live cells"
        raise ValueError(msg)

    return _log_of(probabilities.tensor, defined), defined


class TermsReadout:
    """Publishes what a decoder reads at every cell of a chart.

    The decoder publishes a type distribution for every computed cell. Each declared
    target set also receives its log mass and complement, scaled by the cell's
    constituent probability.

    Parameters
    ----------
    decoder : object
        The decoder providing marginal rows or a complete-term frontier.
    codec : NeuralCodecView
        The fixed term layout whose symbols the readout publishes.
    index : ChoiceIndex
        The payload-aware choices returned by the decoder's rows.
    reads : ReadoutSource
        Whether the decoder consumes each chart cell's inside vector or its
        concatenated inside-and-outside span representation.
    decoding : DecodingSettings | None
        The complete-term strategy, or None to publish marginal choice rows.
    key : ReadoutId
        The identifier the type distribution is published under.
    targets : Mapping[ReadoutId, LinearizedTargets]
        The target sets whose mass is published, each linearized through the codec
        the decoder reads, keyed by the id the wiring assigns it.
    strategy : ReadoutMass
        How a member's probability at a cell is computed.
    mapping : Entmax
        The mapping the decoder's scores go through before they are published. The
        mapping is invariant to a shift shared by a row, so a decoder that already
        normalizes its scores publishes the same distribution as the scores behind
        them. A sparse mapping leaves most symbols at exactly zero mass, and a target
        member needing one of them takes no mass under either strategy.
    meanings : MeaningInductionSettings
        Whether every span is decoded directly or only lexical spans are decoded and
        internal spans are constructed through semantic proof actions.
    proof_terms : ProofTermPushforward | None
        The semantic pushforward for proof-term induction, or None for direct
        decoding.
    approximation : ApproximationBound | None
        Effective residual budget for this readout's exhaustive enumeration. On the
        proof-term route it bounds word decoding, and the pushforward checks derived
        phrase cells against the route's own bound.
    run_approximation : ApproximationBound | None
        Run-level residual budget before this readout's explicit override.

    Raises
    ------
    TypeError
        If an id is not a string.
    ValueError
        If the decoder does not implement the capability its route needs, if an id is
        empty or published twice, if a target set has no member, or if the coupled or
        sampling strategy is declared over a decoder without teacher-forced rows.
    """

    __slots__ = (
        "_approximation",
        "_clamp",
        "_codec",
        "_decoder",
        "_decoding",
        "_decoding_strategy",
        "_index",
        "_key",
        "_mapping",
        "_max_nodes",
        "_meanings",
        "_proof_terms",
        "_reads",
        "_root_sort",
        "_run_approximation",
        "_store",
        "_strategy",
        "_symbol_choices",
        "_targets",
    )

    def __init__(
        self,
        decoder: object,
        /,
        *,
        codec: NeuralCodecView,
        index: ChoiceIndex,
        reads: ReadoutSource,
        decoding: DecodingSettings | None,
        key: ReadoutId,
        targets: Mapping[ReadoutId, LinearizedTargets],
        strategy: ReadoutMass,
        mapping: Entmax,
        store: TermStore | None,
        root_sort: str,
        max_nodes: int | None,
        clamp: Term | None,
        meanings: MeaningInductionSettings,
        proof_terms: ProofTermPushforward | None,
        approximation: ApproximationBound | None,
        run_approximation: ApproximationBound | None,
    ) -> None:
        if not isinstance(reads, ReadoutSource):
            msg = f"a term readout source must be a ReadoutSource, not {reads!r}"
            raise TypeError(msg)

        if not isinstance(meanings, (DecodedMeaningSettings, ProofTermMeaningSettings)):
            msg = f"a meaning induction route is declared settings, not {meanings!r}"
            raise TypeError(msg)

        if (meanings.pushforward() is not None) != (proof_terms is not None):
            msg = "only proof-term meaning induction carries a proof-term pushforward"
            raise ValueError(msg)

        if decoding is None and not isinstance(decoder, MarginalDecoder):
            msg = (
                f"{type(decoder).__name__} does not provide marginal choice rows, "
                "so it cannot publish one distribution per cell"
            )
            raise TypeError(msg)

        if index.signature is not codec.codec.index.signature:
            msg = "a term readout's choices and codec must share one signature"
            raise ValueError(msg)

        declared = _declared(targets)
        keys = [key, *(target.key for target in declared)]

        if len(set(keys)) != len(keys):
            msg = f"a readout publishes each id once, and these repeat: {keys}"
            raise ValueError(msg)

        if (
            decoding is None
            and strategy is ReadoutMass.COUPLED
            and declared
            and not isinstance(decoder, TeacherForcedDecoder)
        ):
            msg = (
                f"{type(decoder).__name__} does not provide teacher-forced rows, "
                "so it cannot couple a target's choices"
            )
            raise TypeError(msg)

        decoding_kind = (
            None if decoding is None else decoding_strategy(decoding.strategy)
        )

        if decoding is not None and decoding_kind is not None:
            if not isinstance(decoder, FrontierDecoder):
                msg = f"{type(decoder).__name__} does not implement frontier decoding"
                raise TypeError(msg)

            if decoding_kind.rescores and not isinstance(decoder, TeacherForcedDecoder):
                msg = (
                    f"{type(decoder).__name__} samples terms but provides no "
                    "teacher-forced rows for rescoring"
                )
                raise TypeError(msg)

            if store is None or max_nodes is None:
                msg = (
                    "a decoding readout needs its language index, store, sort and bound"
                )
                raise ValueError(msg)

            if decoding_kind.enumerates and approximation is None:
                override = decoding.residual_mass

                if override is None:
                    msg = "an exhaustive term readout requires a residual-mass bound"
                    raise ValueError(msg)

                approximation = ApproximationBound(override)

        self._decoder = decoder
        self._codec = codec
        self._reads = reads
        self._decoding = decoding
        self._decoding_strategy = decoding_kind
        self._index = index
        self._store = store
        self._root_sort = root_sort
        self._max_nodes = max_nodes
        self._clamp = clamp
        self._meanings = meanings
        self._proof_terms = proof_terms
        self._approximation = approximation
        self._run_approximation = run_approximation
        choice_symbols = tuple(
            codec.codec.index.PAD
            if choice == index.PAD
            else codec.codec.index.index(index.operator(choice))
            for choice in range(index.width)
        )
        self._symbol_choices = tuple(
            tuple(
                choice for choice, held in enumerate(choice_symbols) if held == symbol
            )
            for symbol in range(codec.n_symbols)
        )
        self._key = ReadoutId(ensure_name("a readout id", key))
        self._targets = declared
        self._strategy = strategy
        self._mapping = mapping

    @property
    def key(self) -> ReadoutId:
        """Return the identifier the type distribution is published under."""
        return self._key

    @property
    def reads(self) -> ReadoutSource:
        """Return which chart representation the decoder consumes."""
        return self._reads

    @property
    def decoding(self) -> DecodingSettings | None:
        """Return the complete-term strategy, or None for a mode readout."""
        return self._decoding

    @property
    def meanings(self) -> MeaningInductionSettings:
        """Return how complete terms are induced at internal chart spans."""
        return self._meanings

    @property
    def approximation(self) -> ApproximationBound | None:
        """Return the residual budget governing this readout's enumeration."""
        return self._approximation

    @property
    def run_approximation(self) -> ApproximationBound | None:
        """Return the run-level residual budget before component overrides."""
        return self._run_approximation

    @property
    def targets(self) -> tuple[ReadoutId, ...]:
        """Return the ids the declared target sets' masses are published under."""
        return tuple(target.key for target in self._targets)

    @property
    def target_encodings(self) -> Mapping[ReadoutId, BatchEncoding]:
        """Return each declared target set's distinct members, keyed by its id."""
        return MappingProxyType(
            {target.key: BatchEncoding(target.rows) for target in self._targets}
        )

    @property
    def target_sets(self) -> Mapping[ReadoutId, TargetSet[Term]]:
        """Return symbolic target sets from the boundary that declared their mass."""
        return MappingProxyType(
            {target.key: target.targets for target in self._targets}
        )

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec the decoder's slots and symbols are laid out by."""
        return self._codec

    @property
    def strategy(self) -> ReadoutMass:
        """Return how a member's probability at a cell is computed."""
        return self._strategy

    @property
    def mapping(self) -> Entmax:
        """Return the mapping the decoder's scores go through before publication."""
        return self._mapping

    def publish(
        self,
        cells: Encoded,
        /,
        *,
        spans: SpanIndex,
        marginals: SpanMarginals | None = None,
    ) -> Mapping[ReadoutId, SpanReadout]:
        """Return what this readout publishes at every cell a step filled.

        The decoder runs once over cells kept by the mask; excluded cells remain
        uncomputed. Target mass is published only where the constituent factor and all
        required member weights are available.

        Parameters
        ----------
        cells : Encoded
            One vector per cell, over the axes ``CELL_AXES``, masked over the item or
            over the item and the span to the cells the chart filled.
        spans : SpanIndex
            The addressing the span axis is over.
        marginals : SpanMarginals | None
            The chart's constituent marginals, which every declared target set's mass
            is weighted by, or None where no target set is declared.

        Returns
        -------
        Mapping[ReadoutId, SpanReadout]
            Under the readout's own id, the type distribution over the axes
            ``TERMS_READOUT_AXES``, normalized as the decoder normalizes and computed
            exactly at the cells the mask kept. Under each target set's id, its mass
            pair over the axes ``MASS_AXES``, normalized over the outcome axis.

        Raises
        ------
        ValueError
            If the cells are not over ``CELL_AXES``, if there is not one per span the
            index addresses, if the mask reaches into a cell's features, if the decoder
            returns its distribution over other axes than it declares, if the decoder
            masks part of a cell, if a target set is declared and no marginals are
            given or the marginals are over other items or spans, or if a target's
            linearization does not fit the decoder's slots.
        """
        return self.publish_with_choices(
            cells, spans=spans, marginals=marginals
        ).readouts

    def publish_with_choices(
        self,
        cells: Encoded,
        /,
        *,
        spans: SpanIndex,
        marginals: SpanMarginals | None = None,
    ) -> TermsPublication:
        """Return published values together with the decoder's choice marginals."""
        if self._decoding is not None:
            msg = "a strategy readout publishes complete terms through decode"
            raise ValueError(msg)

        expected_axes = (
            (*SPAN_AXES, "width")
            if self._reads is ReadoutSource.INSIDE
            else (*SPAN_AXES, "span_width")
        )

        if cells.axes != expected_axes:
            msg = f"cells run over {expected_axes}, not {cells.axes}"
            raise ValueError(msg)

        items, count, width = cells.tensor.shape

        if count != spans.num_spans:
            msg = (
                f"there are {count} cells per item, and the index addresses "
                f"{spans.num_spans}"
            )
            raise ValueError(msg)

        live = _live_cells(cells.mask, (items, count))
        vectors = cells.tensor.reshape(items * count, width)
        kept = None if live is None else live.reshape(-1)
        selected = vectors if kept is None else vectors[kept]
        types, choices = self._types(selected, kept, spans, (items, count))
        published = {self._key: types}

        if self._targets:
            if marginals is None:
                msg = (
                    "a target set's mass is weighted by the constituent marginal, "
                    "and none was given"
                )
                raise ValueError(msg)

            factor, defined = _log_constituent_factor(marginals, spans, items)

            for target in self._targets:
                published[target.key] = self._mass(
                    target, types, selected, kept, factor, defined
                )

        return TermsPublication(MappingProxyType(published), choices)

    def decode(
        self,
        cells: Encoded,
        /,
        *,
        spans: SpanIndex,
        splits: SplitPosterior | None = None,
        actions: Mapping[Action, LogDistribution] | None = None,
    ) -> SpanTerms:
        """Decode weighted complete terms at every computed chart cell."""
        settings = self._decoding
        strategy = self._decoding_strategy
        index = self._index
        store = self._store
        root_sort = self._root_sort
        max_nodes = self._max_nodes

        if (
            settings is None
            or strategy is None
            or store is None
            or root_sort is None
            or max_nodes is None
        ):
            msg = "a mode readout publishes distributions through publish"
            raise ValueError(msg)

        expected_axes = (
            (*SPAN_AXES, "width")
            if self._reads is ReadoutSource.INSIDE
            else (*SPAN_AXES, "span_width")
        )

        if cells.axes != expected_axes:
            msg = f"cells run over {expected_axes}, not {cells.axes}"
            raise ValueError(msg)

        items, count, width = cells.tensor.shape

        if count != spans.num_spans:
            msg = (
                f"there are {count} cells per item, and the index addresses "
                f"{spans.num_spans}"
            )
            raise ValueError(msg)

        live = _live_cells(cells.mask, (items, count))
        decoded: dict[tuple[int, int], WeightedTerms] = {}
        proof_terms = self._proof_terms

        if proof_terms is not None and splits is None:
            msg = "proof-term meaning induction requires a split posterior"
            raise ValueError(msg)

        if proof_terms is not None and actions is None:
            msg = "proof-term meaning induction requires chart action weights"
            raise ValueError(msg)

        span_ids = (
            range(count)
            if proof_terms is None
            else tuple(spans.span_id(i, j) for i, j in spans.leaves())
        )

        for item in range(items):
            for span in span_ids:
                if live is not None and not bool(live[item, span]):
                    continue

                inputs = Encoded(
                    cells.tensor[item, span].reshape(1, width),
                    _TERM_AXES,
                )
                decoded[item, span] = strategy.decode(
                    cast("FrontierDecoder[object]", self._decoder),
                    inputs,
                    settings,
                    codec=self._codec,
                    index=index,
                    store=store,
                    root_sort=root_sort,
                    max_nodes=max_nodes,
                    clamp=self._clamp,
                    approximation=self._approximation,
                )

        if proof_terms is None:
            return SpanTerms(spans, decoded)

        if splits is None or actions is None:
            msg = "proof-term meaning induction requires chart decisions"
            raise RuntimeError(msg)

        return SpanTerms(
            spans,
            proof_terms.derive(
                decoded,
                spans=spans,
                splits=splits,
                actions=actions,
            ),
        )

    def score_terms(self, inputs: Encoded, terms: Sequence[Term], /) -> LogDistribution:
        """Return each supplied complete term's log probability from its input.

        The term readout remains the owner of the decoder and its symbolic codec;
        objectives supply only aligned vectors and symbolic terms.
        """
        held = tuple(terms)

        if inputs.axes != _TERM_AXES:
            msg = f"complete-term inputs run over {_TERM_AXES}, not {inputs.axes}"
            raise ValueError(msg)

        if inputs.tensor.shape[0] != len(held):
            msg = (
                f"complete-term scoring received {inputs.tensor.shape[0]} inputs "
                f"for {len(held)} terms"
            )
            raise ValueError(msg)

        decoder = self._decoder
        if not isinstance(decoder, TeacherForcedDecoder):
            msg = f"{type(decoder).__name__} does not provide teacher-forced rows"
            raise TypeError(msg)

        encoded = self._codec.codec.encode_batch(held)
        scored = score(decoder, inputs, encoded, index=self._index)

        if scored.axes != (_TERM_AXES[0],) or scored.tensor.shape != (len(held),):
            msg = (
                "complete-term scoring returns one weight over the term axis, not "
                f"axes {scored.axes} and shape {tuple(scored.tensor.shape)}"
            )
            raise ValueError(msg)

        return scored

    def decode_input[StateT](
        self,
        decoder: FrontierDecoder[StateT],
        inputs: Encoded,
        /,
        *,
        space: Space,
        approximation: ApproximationBound | None = None,
    ) -> WeightedTerms:
        """Decode one vector through this readout's declared strategy.

        The supplied decoder may be the identity decoder or a combinator decoder of
        the same represented space. The readout remains the single owner of the
        symbolic alphabet, root sort, term bound, clamp and strategy settings.
        """
        settings = self._decoding
        strategy = self._decoding_strategy

        if (
            settings is None
            or strategy is None
            or self._store is None
            or self._max_nodes is None
        ):
            msg = "a mode readout has no complete-term strategy"
            raise ValueError(msg)

        return strategy.decode(
            decoder,
            inputs,
            settings,
            codec=space.codec,
            index=space.choices,
            store=space.store,
            root_sort=space.sort,
            max_nodes=space.max_nodes,
            clamp=self._clamp,
            approximation=approximation or self._approximation,
        )

    def paired_inputs[LeftStateT, RightStateT](
        self,
        reference: FrontierDecoder[LeftStateT],
        candidate: FrontierDecoder[RightStateT],
        reference_inputs: Encoded,
        candidate_inputs: Encoded,
        /,
        *,
        space: Space,
        approximation: ApproximationBound | None = None,
    ) -> tuple[WeightedTerms, WeightedTerms]:
        """Decode aligned reference and candidate states on one term frontier."""
        settings = self._decoding
        strategy = self._decoding_strategy

        if settings is None or strategy is None or not strategy.enumerates:
            msg = "paired coherence requires an enumerating term readout"
            raise ValueError(msg)

        if (
            self._store is None
            or self._max_nodes is None
            or settings.cap is None
            or settings.exact_residual is None
        ):
            msg = "an exhaustive term readout has incomplete symbolic settings"
            raise ValueError(msg)

        bound = approximation or self._approximation

        if bound is None:
            msg = "paired exhaustive decoding requires a residual-mass bound"
            raise ValueError(msg)

        reference_terms, candidate_terms = paired_exhaustive(
            reference,
            reference_inputs,
            candidate,
            candidate_inputs,
            index=space.choices,
            store=space.store,
            root_sort=space.sort,
            max_nodes=space.max_nodes,
            residual_mass=bound.residual_mass,
            exact_residual=settings.exact_residual,
            cap=settings.cap,
            clamp=self._clamp,
        )
        component = f"term readout {self._key!r}"
        return (
            bound.check_terms(reference_terms, component=component),
            bound.check_terms(candidate_terms, component=component),
        )

    def _types(
        self,
        selected: Tensor,
        kept: Tensor | None,
        spans: SpanIndex,
        cells: tuple[int, int],
    ) -> tuple[SpanReadout, SpanReadout]:
        """Return the type distribution the decoder reads at the selected cells.

        The decoder's rows go through the configured mapping before they are
        published, so a sparse mapping leaves most symbols at negative infinity.

        Parameters
        ----------
        selected : Tensor
            One vector per cell the mask kept, of rank two over the cell and the width.
        kept : Tensor | None
            Which of every item's cells the mask kept, of rank one, or None where it
            kept them all.
        spans : SpanIndex
            The addressing the span axis is over.
        cells : tuple[int, int]
            How many items and how many spans there are.

        Raises
        ------
        ValueError
            If the decoder returns its distribution over other axes than it declares,
            or masks part of a cell.
        """
        items, count = cells
        decoder = self._decoder
        if not isinstance(decoder, MarginalDecoder):
            msg = "a strategy readout has no distribution mode"
            raise TypeError(msg)

        rows = decoder.marginal_rows(
            Encoded(selected, _TERM_AXES), root_sort=self._root_sort
        )

        choice_axes = ("term", "slot", "choice")
        if rows.axes != choice_axes:
            msg = f"the decoder returned {rows.axes}, and declares {choice_axes}"
            raise ValueError(msg)

        if rows.normalized_over != "choice":
            msg = "marginal decoder rows normalize over the choice axis"
            raise ValueError(msg)

        if rows.tensor.shape[-1] != self._index.width:
            msg = (
                f"the decoder returned {rows.tensor.shape[-1]} choices, and the "
                f"language declares {self._index.width}"
            )
            raise ValueError(msg)

        choice_values = _canonical_choice_rows(rows, self._index.PAD)
        symbol_rows = []

        for choices in self._symbol_choices:
            if not choices:
                symbol_rows.append(
                    choice_values.new_full(choice_values.shape[:-1], -torch.inf)
                )
                continue

            held = choice_values[
                ..., torch.tensor(choices, device=choice_values.device)
            ]
            finite = held.isfinite().any(dim=-1)
            safe = torch.where(finite.unsqueeze(-1), held, held.new_zeros(()))
            collapsed = torch.logsumexp(safe, dim=-1)
            symbol_rows.append(torch.where(finite, collapsed, -torch.inf))

        collapsed = torch.stack(tuple(symbol_rows), dim=-1)
        scored = self._mapping.project(collapsed)
        decoded = _computed_rows(rows)
        _, slots, symbols = scored.shape
        choice_width = choice_values.shape[-1]

        if kept is None and decoded is None:
            return (
                SpanReadout(
                    spans,
                    LogDistribution(
                        scored.reshape(items, count, slots, symbols),
                        TERMS_READOUT_AXES,
                        normalized_over="symbol",
                    ),
                ),
                SpanReadout(
                    spans,
                    LogDistribution(
                        choice_values.reshape(items, count, slots, choice_width),
                        (*SPAN_AXES, "slot", "choice"),
                        normalized_over="choice",
                    ),
                ),
            )

        full = scored.new_full((items * count, slots, symbols), -torch.inf)
        full_choices = choice_values.new_full(
            (items * count, slots, choice_width), -torch.inf
        )
        computed = torch.zeros(items * count, dtype=torch.bool, device=scored.device)
        target = slice(None) if kept is None else kept
        full[target] = scored
        full_choices[target] = choice_values
        computed[target] = True if decoded is None else decoded

        return (
            SpanReadout(
                spans,
                LogDistribution(
                    full.reshape(items, count, slots, symbols),
                    TERMS_READOUT_AXES,
                    computed.reshape(items, count),
                    normalized_over="symbol",
                ),
            ),
            SpanReadout(
                spans,
                LogDistribution(
                    full_choices.reshape(items, count, slots, choice_width),
                    (*SPAN_AXES, "slot", "choice"),
                    computed.reshape(items, count),
                    normalized_over="choice",
                ),
            ),
        )

    def _mass(
        self,
        target: _Declared,
        types: SpanReadout,
        selected: Tensor,
        kept: Tensor | None,
        factor: Tensor,
        defined: Tensor,
    ) -> SpanReadout:
        """Return one target set's mass pair at every cell.

        Parameters
        ----------
        target : _Declared
            The set whose mass is published.
        types : SpanReadout
            The type distribution this step published.
        selected : Tensor
            One vector per cell the mask kept, of rank two over the cell and the width.
        kept : Tensor | None
            Which of every item's cells the mask kept, of rank one, or None where it
            kept them all.
        factor : Tensor
            The log constituent factor at every cell.
        defined : Tensor
            True at every cell the factor is defined at.

        Raises
        ------
        ValueError
            If the target's linearization does not fit the decoder's slots.
        """
        if self._strategy is ReadoutMass.FACTORIZED:
            members, scored = self._factorized(target, types)

        else:
            members, scored = self._coupled(target, types, selected, kept)

        computed = types.computed & defined & scored
        absent = members.new_full((), -torch.inf)
        mass = torch.where(computed, factor + members, absent).clamp(max=0.0)
        rest = torch.where(computed, _log_one_minus(mass), absent)
        pair = torch.stack((mass, rest), dim=-1)

        return SpanReadout(
            types.spans,
            LogDistribution(pair, MASS_AXES, computed, normalized_over=MASS_AXES[-1]),
        )

    def _factorized(
        self, target: _Declared, types: SpanReadout
    ) -> tuple[Tensor, Tensor]:
        """Return the log summed member probability at every cell, read off the rows.

        Parameters
        ----------
        target : _Declared
            The set whose members are summed.
        types : SpanReadout
            The type distribution this step published.

        Returns
        -------
        tuple[Tensor, Tensor]
            The log sum over the item and the span, and a boolean tensor of the same
            shape that is true everywhere, since the rows hold every cell.

        Raises
        ------
        ValueError
            If a member has other slots than the rows, or a symbol the rows do not
            hold.
        """
        totals = _member_log_probabilities(types.distribution.tensor, target)

        return _log_sum(totals), types.computed.new_ones(types.computed.shape)

    def _coupled(
        self,
        target: _Declared,
        types: SpanReadout,
        selected: Tensor,
        kept: Tensor | None,
    ) -> tuple[Tensor, Tensor]:
        """Return the log summed member weight at every cell, scored by the decoder.

        A member the published rows leave at exactly zero mass is off the type
        distribution this readout publishes, so it takes no weight here either. Under
        the dense mapping the rows keep every symbol the decoder admits and no member
        is dropped.

        Parameters
        ----------
        target : _Declared
            The set whose members are scored.
        types : SpanReadout
            The type distribution this step published.
        selected : Tensor
            One vector per cell the mask kept, of rank two over the cell and the width.
        kept : Tensor | None
            Which of every item's cells the mask kept, of rank one, or None where it
            kept them all.

        Returns
        -------
        tuple[Tensor, Tensor]
            The log sum over the item and the span, and a boolean tensor of the same
            shape that is true where the decoder scored every member.

        Raises
        ------
        ValueError
            If the decoder returns its weights over other axes than it declares, if a
            member has other slots than the published rows, or if it names a symbol
            they do not hold.
        """
        items, count = types.computed.shape
        cells, _ = selected.shape
        members = len(target.rows)
        published = _member_log_probabilities(types.distribution.tensor, target)
        carried = published.reshape(items * count, members)
        supported = (carried if kept is None else carried[kept]) > -torch.inf
        decoder = self._decoder
        if not isinstance(decoder, TeacherForcedDecoder):
            msg = f"{type(decoder).__name__} does not provide teacher-forced rows"
            raise TypeError(msg)

        weights = score(
            decoder,
            Encoded(selected.repeat_interleave(members, dim=0), _TERM_AXES),
            BatchEncoding(target.rows * cells),
            index=self._index,
        )

        if weights.axes != _TERM_AXES[:1]:
            msg = f"the decoder returned {weights.axes}, and declares {_TERM_AXES[:1]}"
            raise ValueError(msg)

        given = weights.tensor.reshape(cells, members)
        absent = given.new_full((), -torch.inf)
        summed = _log_sum(torch.where(supported, given, absent))
        decided = (
            summed.new_ones(cells, dtype=torch.bool)
            if weights.mask is None
            else weights.mask.reshape(cells, members).all(dim=-1)
        )

        if kept is None:
            return summed.reshape(items, count), decided.reshape(items, count)

        full = summed.new_full((items * count,), -torch.inf)
        scored = torch.zeros(items * count, dtype=torch.bool, device=summed.device)
        full[kept] = summed
        scored[kept] = decided

        return full.reshape(items, count), scored.reshape(items, count)

    @override
    def __repr__(self) -> str:
        """Return the ids, decoder type, route and mass strategy."""
        return (
            f"TermsReadout({type(self._decoder).__name__}, "
            f"reads={self._reads!r}, decoding={self._decoding!r}, "
            f"meanings={self._meanings.kind!r}, "
            f"key={self._key!r}, "
            f"targets={self.targets!r}, strategy={self._strategy.name}, "
            f"mapping={self._mapping!r})"
        )


def _build(settings: TermsReadoutSettings, context: BuildContext, /) -> TermsReadout:
    """Bind a term readout to its decoder and to the run's target sets.

    The component handle names the distribution the readout publishes. The
    declared output key names the target-mass pair of the declared target source,
    and every other target set of the context is published under its own name.

    Parameters
    ----------
    settings : TermsReadoutSettings
        The represented space, output key, target source, optional complete-term
        strategy and mass strategy.
    context : BuildContext
        The decoders by role, the shared codec and the target sets by source.

    Raises
    ------
    KeyError
        If the decoder role names no decoder the run built.
    TypeError
        If the space identity lacks the capability required by the readout route.
    ValueError
        If the target source names no target set of the run, if the decoder does
        not share the run's codec view, or if a target id is published twice.
    """
    role = str(context.component)
    decoder = context.producer("identity")
    space = context.producer("space")

    if not isinstance(space, _RepresentedSpace):
        msg = f"term readout {role!r} is not wired to a represented space"
        raise TypeError(msg)

    if space.codec is not context.codec:
        msg = (
            f"the space identity and term readout {role!r} do not "
            "share the exact codec view"
        )
        raise ValueError(msg)

    source = None if settings.targets is None else str(settings.targets)

    if source is not None and source not in context.targets:
        known = ", ".join(sorted(context.targets)) or "no target source"
        msg = f"term readout {role!r} names target source {source!r}; known: {known}"
        raise ValueError(msg)

    codec = cast("TermCodec", context.codec.codec)
    linearized: dict[ReadoutId, LinearizedTargets] = {}

    if source is not None:
        output = ReadoutId(settings.output or source)
        linearized[output] = linearize_targets(context.targets[source], codec)

    for name, additional in context.targets.items():
        if name == source:
            continue

        key = ReadoutId(name)

        if key in linearized:
            msg = f"term readout {role!r} publishes target id {key!r} twice"
            raise ValueError(msg)

        linearized[key] = linearize_targets(additional, codec)

    decoding = settings.decoding
    run_approximation = context.approximation
    approximation = run_approximation
    clamp = None
    proof_terms = None

    if decoding is not None and decoding.clamp is not None:
        surface = context.language.syntax.parse(decoding.clamp)
        clamp = context.language.syntax.desugar(surface).term

    if decoding is not None and decoding_strategy(decoding.strategy).enumerates:
        override = decoding.residual_mass

        if override is not None:
            approximation = ApproximationBound(override)

        if approximation is None:
            msg = f"term readout {role!r} requires a residual-mass bound"
            raise ValueError(msg)

    pushed = settings.meanings.pushforward()

    if pushed is not None:
        meaning = context.meaning

        if decoding is None:
            msg = "proof-term meaning induction requires a lexical decoder"
            raise ValueError(msg)

        if meaning is None:
            msg = f"proof-term readout {role!r} requires a dataset interpretation"
            raise ValueError(msg)

        if meaning.actions.calculus.store is not context.language.store:
            msg = (
                f"proof-term readout {role!r} and its interpretation do not share "
                "one term store"
            )
            raise ValueError(msg)

        proof_terms = ProofTermPushforward(
            context.actions,
            meaning,
            cap=decoding_strategy(decoding.strategy).frontier(decoding),
            rule_arities={rule.name: rule.arity for rule in context.rules.rules},
            weighting=meaning_weighting(pushed.weighting),
            bound=ApproximationBound(pushed.residual_mass),
        )

    return TermsReadout(
        decoder,
        codec=context.codec,
        index=context.choices,
        reads=settings.reads,
        key=ReadoutId(ensure_name("a readout role", role)),
        targets=linearized,
        decoding=decoding,
        strategy=settings.mass,
        mapping=SOFTMAX,
        store=context.language.store if decoding is not None else None,
        root_sort=space.sort,
        max_nodes=context.codec.capacity if decoding is not None else None,
        clamp=clamp,
        meanings=settings.meanings,
        proof_terms=proof_terms,
        approximation=approximation,
        run_approximation=run_approximation,
    )


PLUGIN: Final = ComponentPlugin(spec=TermsReadoutSpec, build=_build)
"""The term readout's plugin."""
