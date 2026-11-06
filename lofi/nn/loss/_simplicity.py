"""The simplicity prior: the expected description length of a readout's terms."""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_name
from lofi._types import ReadoutId
from lofi.nn.loss._description_cost import DescriptionCost, description_cost
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import TERMS_READOUT_AXES
from lofi.registry import ComponentPlugin
from lofi.spec.losses.simplicity import (
    SimplicityLossSettings,
    SimplicityLossSpec,
    SimplicityScope,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn.codec import TermCoding
    from lofi.nn.loss._ports import LossComponentBindings, PublishedOutputs
    from lofi.nn.loss._values import SpanReadout

__all__ = [
    "SimplicityPriorObjective",
    "SimplicityScope",
    "expected_description_lengths",
]


def _paddable(codec: TermCoding) -> tuple[int, ...]:
    """Return the slots a term may leave empty, in slot order.

    Parameters
    ----------
    codec : TermCoding
        The codec whose layout says which symbols each slot may hold.
    """
    return tuple(
        slot
        for slot, allowed in enumerate(codec.legal_symbols)
        if allowed[codec.index.PAD]
    )


def _lexical(spans: SpanIndex, like: Tensor) -> Tensor:
    """Return which spans the lexicon assigns a term to.

    Parameters
    ----------
    spans : SpanIndex
        The addressing the span axis is over.
    like : Tensor
        A tensor on the device the mask is built for.

    Returns
    -------
    Tensor
        A boolean tensor of rank 1 over the spans, true at those of width one.
    """
    mask = like.new_zeros((spans.num_spans,), dtype=torch.bool)

    for i, j in spans.leaves():
        mask[spans.span_id(i, j)] = True

    return mask


def _expected_description_lengths(
    scores: Tensor,
    computed: Tensor,
    spans: SpanIndex,
    costs: tuple[float, ...],
    scope: SimplicityScope,
    /,
) -> tuple[Tensor, Tensor]:
    """Return expected constructor cost per cell and the cells in scope."""
    priced = computed

    if scope is SimplicityScope.LEXICON:
        priced = priced & _lexical(spans, scores).unsqueeze(0)

    symbol_costs = scores.new_tensor(costs)
    lengths = (scores.exp() * symbol_costs).sum(dim=(-1, -2))
    return lengths, priced


def expected_description_lengths(
    published: SpanReadout,
    spans: SpanIndex,
    codec: TermCoding,
    scope: SimplicityScope,
    symbol_cost: DescriptionCost,
    /,
) -> tuple[Tensor, Tensor]:
    """Return each chart cell's expected description length and pricing mask.

    This is the shared lexical-pricing kernel used by the simplicity objective and
    held-out reporting. It validates the term distribution against the exact codec
    that owns its slots and symbols, then returns the unaggregated expectation so a
    caller can take either a per-cell mean or a per-item total.

    Parameters
    ----------
    published : SpanReadout
        Per-cell marginal distribution over term symbols.
    spans : SpanIndex
        Addressing of the readout's span axis.
    codec : TermCoding
        Codec whose slots and symbol index the distribution represents.
    scope : SimplicityScope
        Cells whose expected cost is included.
    symbol_cost : DescriptionCost
        Declared constructor cost policy.

    Returns
    -------
    tuple[Tensor, Tensor]
        Expected description length at every cell and the boolean cells in scope.

    Raises
    ------
    ValueError
        If the publication axes or shape disagree with the codec, or if the cost
        policy does not assign zero to padding and one cost per symbol.
    """
    if published.spans.n != spans.n:
        msg = (
            f"the term readout spans inputs of length {published.spans.n}, and the "
            f"description-length index spans length {spans.n}"
        )
        raise ValueError(msg)

    if published.axes != TERMS_READOUT_AXES:
        msg = (
            f"description length prices a readout over {TERMS_READOUT_AXES}, not "
            f"{published.axes}"
        )
        raise ValueError(msg)

    distribution = published.distribution
    shape = (distribution.size("slot"), distribution.size("symbol"))
    expected = (codec.capacity, codec.n_symbols)

    if shape != expected:
        msg = (
            f"the term readout holds {shape[0]} slots of {shape[1]} symbols, and "
            f"the codec lays out {expected[0]} slots of {expected[1]}"
        )
        raise ValueError(msg)

    costs = symbol_cost.costs(codec.index)

    if len(costs) != codec.n_symbols:
        msg = (
            f"the description-cost policy returned {len(costs)} costs for "
            f"{codec.n_symbols} symbols"
        )
        raise ValueError(msg)

    pad = codec.index.PAD

    if costs[pad] != 0.0:
        msg = f"the padding symbol has description cost {costs[pad]!r}, not 0"
        raise ValueError(msg)

    return _expected_description_lengths(
        distribution.tensor,
        published.computed,
        spans,
        costs,
        scope,
    )


class SimplicityPriorObjective[BatchT]:
    """Prices a readout's terms by how many nodes they are expected to occupy.

    The prior sums expected declared constructor cost over every slot and averages it
    across cells in scope. Padding has cost zero; the default scope prices width-one
    cells.

    Parameters
    ----------
    readout : ReadoutId
        The id the readout is published under.
    codec : TermCoding
        The codec whose operator index assigns the padding code and whose layout says
        which slots a term may leave empty.
    scope : SimplicityScope
        Which cells are priced.
    symbol_cost : DescriptionCost
        Policy assigning a cost to every constructor in the codec's alphabet.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the readout id or the name is not a string.
    ValueError
        If the readout id or the name is empty.
    """

    __slots__ = (
        "_codec",
        "_name",
        "_pad",
        "_paddable",
        "_readout",
        "_scope",
        "_symbol_cost",
    )

    def __init__(
        self,
        *,
        readout: ReadoutId,
        codec: TermCoding,
        scope: SimplicityScope,
        symbol_cost: DescriptionCost,
        name: str,
    ) -> None:
        self._readout = ReadoutId(ensure_name("the readout id", readout))
        self._pad = codec.index.PAD
        self._paddable = _paddable(codec)
        costs = symbol_cost.costs(codec.index)

        if len(costs) != codec.n_symbols:
            msg = (
                f"the description-cost policy returned {len(costs)} costs for "
                f"{codec.n_symbols} symbols"
            )
            raise ValueError(msg)

        if costs[self._pad] != 0.0:
            msg = f"the padding symbol has description cost {costs[self._pad]!r}, not 0"
            raise ValueError(msg)

        self._codec = codec
        self._symbol_cost = symbol_cost
        self._scope = scope
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"spans", "readouts"})

    @property
    def readout(self) -> ReadoutId:
        """Return the id the readout is published under."""
        return self._readout

    @property
    def scope(self) -> SimplicityScope:
        """Return which cells are priced."""
        return self._scope

    def _published(self, readouts: Mapping[ReadoutId, SpanReadout]) -> SpanReadout:
        """Return what the configured readout published.

        Parameters
        ----------
        readouts : Mapping[ReadoutId, SpanReadout]
            What every readout published, keyed by id.

        Raises
        ------
        ValueError
            If nothing was published under the configured id.
        """
        published = readouts.get(self._readout)

        if published is None:
            offered = ", ".join(sorted(readouts)) or "nothing"
            msg = (
                f"{self._name} reads the readout {self._readout!r}, and the step "
                f"published {offered}"
            )
            raise ValueError(msg)

        return published

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return the expected constructor cost per priced cell.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the priced cells the readout computed, counted in those cells,
            reporting the mean padding probability over their paddable slots as
            ``padding``.

        Raises
        ------
        ValueError
            If the step published nothing under the configured readout id, or if what
            it published is not one distribution over the codec's symbols at each of
            its slots.
        """
        published = self._published(outputs.readouts)
        scores = published.distribution.tensor
        lengths, priced = expected_description_lengths(
            published,
            outputs.spans,
            self._codec,
            self._scope,
            self._symbol_cost,
        )

        count = int(priced.sum())

        if count == 0:
            return LossTerm.inapplicable(scores)

        selected = scores[priced]
        padding = selected[:, list(self._paddable), self._pad].exp()
        description_length = lengths[priced]

        return LossTerm(
            description_length.sum() / count,
            count,
            diagnostics={"padding": padding.detach().mean()},
        )

    @override
    def __repr__(self) -> str:
        """Return the name, the readout id and the scope."""
        return (
            f"SimplicityPriorObjective(name={self._name!r}, "
            f"readout={self._readout!r}, scope={self._scope!r})"
        )


def _build[BatchT](
    settings: SimplicityLossSettings, context: LossComponentBindings, /
) -> SimplicityPriorObjective[BatchT]:
    """Build the prior over the declared readout's distribution and codec.

    Parameters
    ----------
    settings : SimplicityLossSettings
        The term-readout role and the cells priced.
    context : LossComponentBindings
        The declared name and the built components the role names.

    Raises
    ------
    TypeError
        If the role holds a component of another family.
    ValueError
        If the role is not built.
    """
    readout = context.terms_readout(settings.readout)

    return SimplicityPriorObjective(
        readout=readout.key,
        codec=readout.codec.codec,
        scope=settings.scope,
        symbol_cost=description_cost(settings.symbol_cost),
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=SimplicityLossSpec, build=_build)
"""The simplicity prior's plugin."""
