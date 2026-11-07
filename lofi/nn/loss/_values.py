"""Restricted storage for parser outputs."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import CHECKS, ensure_name
from lofi.nn._values import DerivationScoreChart, LogDistribution
from lofi.nn.loss._errors import UncomputedCellError, UndeclaredOutputError
from lofi.nn.loss._parser import SPAN_AXES

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from torch import Tensor

    from lofi._types import ReadoutId
    from lofi.deduce import SpanIndex
    from lofi.judge import Action
    from lofi.nn._values import Encoded
    from lofi.nn.forms import WeightedTerms
    from lofi.nn.grounding import GroundedBatch
    from lofi.nn.loss._parser import SpanMarginals, SplitPosterior


__all__ = [
    "OUTPUT_FIELDS",
    "OUTPUT_NAMES",
    "OUTSIDE_CHART",
    "CellChart",
    "ObjectiveOutputs",
    "RecordingOutputs",
    "SpanReadout",
    "SpanTerms",
]

#: The leading axis of a gather over chart cells, one row per cell read.
_CELL_AXIS: Final = "cell"

OUTPUT_FIELDS: Final[tuple[str, ...]] = (
    "spans",
    "chart",
    "splits",
    "scores",
    "marginals",
    "actions",
    "readouts",
    "choice_rows",
    "terms",
    "forms",
    "entailment",
    "cells",
    "outside_cells",
    "entropy",
    "mbr",
    "derivations",
    "batch",
)
"""Output fields in constructor order."""

OUTSIDE_CHART: Final[str] = "chart.outside"
"""The outside-chart output name."""

OUTPUT_NAMES: Final[frozenset[str]] = frozenset(OUTPUT_FIELDS) | {OUTSIDE_CHART}
"""Output names available to objectives."""


def _field_of(name: str) -> str:
    """Return the field a declared name reads.

    Parameters
    ----------
    name : str
        A name from the published vocabulary.
    """
    field, _, _ = name.partition(".")

    return field


class CellChart:
    """Stores inside and optional outside scores by span.

    Parameters
    ----------
    inside : LogDistribution
        Log inside scores per span.
    outside : LogDistribution | None
        Log outside scores per span, or None where the parser ran no outside pass.

    Raises
    ------
    ValueError
        When the debug checks are on and the two passes are not over the same axes.
    """

    __slots__ = ("_inside", "_outside")

    def __init__(
        self,
        inside: LogDistribution,
        outside: LogDistribution | None = None,
    ) -> None:
        if CHECKS and outside is not None and outside.axes != inside.axes:
            msg = (
                f"the outside pass is over the axes {outside.axes} and the "
                f"inside pass over {inside.axes}"
            )
            raise ValueError(msg)

        self._inside = inside
        self._outside = outside

    @property
    def inside(self) -> LogDistribution:
        """Return log inside scores per span."""
        return self._inside

    @property
    def outside(self) -> LogDistribution | None:
        """Return log outside scores per span, or None where no outside pass ran."""
        return self._outside

    @override
    def __repr__(self) -> str:
        """Return the chart axes and outside-score availability."""
        return (
            f"CellChart(axes={self._inside.axes!r}, "
            f"outside={self._outside is not None})"
        )


def _checked_readout_axes(axes: tuple[str, ...]) -> tuple[str, ...]:
    """Return the axes of a per-span readout.

    Parameters
    ----------
    axes : tuple[str, ...]
        The axes the published value declares.

    Returns
    -------
    tuple[str, ...]
        The axes, unchanged.

    Raises
    ------
    ValueError
        If the value does not run over the item and then the span outermost.
    """
    if axes[: len(SPAN_AXES)] != SPAN_AXES:
        msg = f"a readout runs over {SPAN_AXES} outermost, and this one over {axes}"
        raise ValueError(msg)

    return axes


def _computed_cells(distribution: LogDistribution) -> Tensor:
    """Return which cells a published value holds a computed entry for.

    Parameters
    ----------
    distribution : LogDistribution
        The published value, over the item and then the span outermost.

    Returns
    -------
    Tensor
        A boolean tensor of rank 2 over the item and the span.

    Raises
    ------
    ValueError
        If the value is masked below the span axis, which would mark a cell as computed
        in part.
    """
    tensor = distribution.tensor
    mask = distribution.mask
    cells = tuple(tensor.shape[: len(SPAN_AXES)])

    if mask is None:
        return tensor.new_ones(cells, dtype=torch.bool)

    if mask.ndim > len(SPAN_AXES):
        msg = (
            "a readout is computed cell by cell, and a mask over "
            f"{distribution.mask_axes} marks part of a cell"
        )
        raise ValueError(msg)

    widened = mask.reshape(tuple(mask.shape) + (1,) * (len(cells) - mask.ndim))

    return widened.expand(cells)


class SpanReadout:
    """Stores what one readout published at every cell of a chart for one step.

    Parameters
    ----------
    spans : SpanIndex
        The addressing the span axis is over.
    distribution : LogDistribution
        The published values, over the item and then the span outermost and the
        readout's own axes within, masked over the item or over the item and the span
        to the cells the readout computed.

    Raises
    ------
    ValueError
        If the value does not run over the item and then the span outermost, if its
        span axis is not as long as the index has spans, or if it is masked below the
        span axis.
    """

    __slots__ = ("_computed", "_distribution", "_spans")

    def __init__(self, spans: SpanIndex, distribution: LogDistribution) -> None:
        _checked_readout_axes(distribution.axes)

        if distribution.size(SPAN_AXES[1]) != spans.num_spans:
            msg = (
                f"a readout holds {distribution.size(SPAN_AXES[1])} spans, and the "
                f"index addresses {spans.num_spans}"
            )
            raise ValueError(msg)

        self._spans = spans
        self._distribution = distribution
        self._computed = _computed_cells(distribution)

    @property
    def spans(self) -> SpanIndex:
        """Return the addressing the span axis is over."""
        return self._spans

    @property
    def distribution(self) -> LogDistribution:
        """Return the published values over every cell, computed or not."""
        return self._distribution

    @property
    def axes(self) -> tuple[str, ...]:
        """Return what each axis of the published values means, outermost first."""
        return self._distribution.axes

    @property
    def computed(self) -> Tensor:
        """Return a rank-2 boolean tensor, true at a cell the readout computed."""
        return self._computed

    @property
    def items(self) -> int:
        """Return the batch size."""
        return int(self._distribution.tensor.shape[0])

    def _address(self, item: int, i: int, j: int) -> int:
        """Return the span id of one cell, refusing a cell the chart does not have.

        Parameters
        ----------
        item : int
            Which item of the batch to read.
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Raises
        ------
        ValueError
            If the span is empty or reaches outside the input.
        IndexError
            If there is no such item.
        """
        if not 0 <= i < j <= self._spans.n:
            msg = (
                f"the span [{i}, {j}) is not a span of an input of length "
                f"{self._spans.n}"
            )
            raise ValueError(msg)

        if not 0 <= item < self.items:
            msg = f"there is no item {item} in a readout over {self.items} items"
            raise IndexError(msg)

        return self._spans.span_id(i, j)

    def at(self, item: int, i: int, j: int) -> LogDistribution:
        """Return what the readout published at one cell.

        Parameters
        ----------
        item : int
            Which item of the batch to read.
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.

        Returns
        -------
        LogDistribution
            The cell's values over the readout's own axes, attached to the published
            tensor's autograd graph and normalized over the axis the published value
            is.

        Raises
        ------
        ValueError
            If the span is empty or reaches outside the input.
        IndexError
            If there is no such item.
        UncomputedCellError
            If the readout never computed that cell.
        """
        span = self._address(item, i, j)

        if not bool(self._computed[item, span]):
            raise UncomputedCellError(item=item, span=(i, j))

        return LogDistribution(
            self._distribution.tensor[item, span],
            self._distribution.axes[len(SPAN_AXES) :],
            normalized_over=self._distribution.normalized_over,
        )

    def gather(self, cells: Sequence[tuple[int, int, int]]) -> LogDistribution:
        """Return what the readout published at several cells, stacked in order.

        Every address is checked on the host, and the computed mask is consulted once
        for all of them, so a read of many cells costs one gather however many there
        are.

        Parameters
        ----------
        cells : Sequence[tuple[int, int, int]]
            The cells to read, each as an item, a span start and an exclusive span
            end, in the order the rows are returned in.

        Returns
        -------
        LogDistribution
            One row per cell over a leading cell axis and then the readout's own
            axes, attached to the published tensor's autograd graph and normalized
            over the axis the published value is.

        Raises
        ------
        ValueError
            If a span is empty or reaches outside the input.
        IndexError
            If a cell names an item the readout does not hold.
        UncomputedCellError
            If the readout never computed one of the cells, naming the first such
            cell in the given order.
        """
        held = self._distribution.tensor
        items = torch.tensor(
            [item for item, _, _ in cells], dtype=torch.long, device=held.device
        )
        spans = torch.tensor(
            [self._address(item, i, j) for item, i, j in cells],
            dtype=torch.long,
            device=held.device,
        )
        computed = self._computed[items, spans]

        if not bool(computed.all()):
            first = int(computed.logical_not().nonzero()[0])
            item, i, j = cells[first]
            raise UncomputedCellError(item=item, span=(i, j))

        return LogDistribution(
            held[items, spans],
            (_CELL_AXIS, *self._distribution.axes[len(SPAN_AXES) :]),
            normalized_over=self._distribution.normalized_over,
        )

    @override
    def __repr__(self) -> str:
        """Return the axes and how many cells were computed."""
        return (
            f"SpanReadout(axes={self._distribution.axes!r}, "
            f"computed={int(self._computed.sum())} of {self._computed.numel()})"
        )


class SpanTerms:
    """Weighted complete terms decoded at selected cells of one chart."""

    __slots__ = ("_cells", "_spans")

    def __init__(
        self,
        spans: SpanIndex,
        cells: Mapping[tuple[int, int], WeightedTerms],
        /,
    ) -> None:
        held: dict[tuple[int, int], WeightedTerms] = {}

        for address, terms in cells.items():
            item, span = address

            if item < 0:
                msg = f"a decoded cell has negative item index {item}"
                raise ValueError(msg)

            if not 0 <= span < spans.num_spans:
                msg = (
                    f"a decoded cell has span index {span}, outside the "
                    f"{spans.num_spans} addressed spans"
                )
                raise ValueError(msg)

            held[address] = terms

        self._spans = spans
        self._cells: Mapping[tuple[int, int], WeightedTerms] = MappingProxyType(
            dict(sorted(held.items()))
        )

    @property
    def spans(self) -> SpanIndex:
        """Return the span addressing the decoded cells use."""
        return self._spans

    @property
    def cells(self) -> Mapping[tuple[int, int], WeightedTerms]:
        """Return weighted terms keyed by item position and span id."""
        return self._cells

    def at(self, item: int, span: int, /) -> WeightedTerms:
        """Return the weighted terms decoded at one chart cell."""
        return self._cells[item, span]

    def __len__(self) -> int:
        """Return the number of decoded cells."""
        return len(self._cells)


class ObjectiveOutputs[BatchT]:
    """Stores parser outputs for one training step.

    Parameters
    ----------
    spans : SpanIndex
        The span addressing every chart index in the bundle is over.
    chart : CellChart
        The inside scores per span, and the outside scores where they were computed.
    batch : BatchT
        What the featurizer handed this step, which the neural layer passes through
        without naming.
    splits : SplitPosterior | None
        The posterior over the ways each span divides, or None where no parser published
        one.
    scores : LogDistribution | None
        The scalar log inside score of each item, which is the partition a normalizing
        objective divides by, or None where none was published.
    marginals : SpanMarginals | None
        The constituent and rule marginals of the chart, or None where none was
        published.
    actions : Mapping[Action, LogDistribution] | None
        The log weight of each action per span, keyed by the action itself.
    readouts : Mapping[ReadoutId, SpanReadout] | None
        What each readout published, keyed by the readout's id.
    choice_rows : Mapping[ReadoutId, SpanReadout] | None
        Payload-aware decoder marginals keyed by the readout whose decoder produced
        them.
    terms : Mapping[ReadoutId, SpanTerms] | None
        Weighted complete terms decoded by each strategy readout, keyed by the
        readout's id and then by chart cell.
    forms : Mapping[ReadoutId, WeightedTerms] | None
        The weighted logical forms each side published, keyed by its side id.
    entailment : GroundedBatch | None
        The inference grounding's batched label distribution, or None where no
        grounding was run.
    cells : Encoded | None
        The representation the chart holds for each span, or None where none was
        published.
    outside_cells : Encoded | None
        The contextual outside representation of each span, or None where none was
        published.
    entropy : Encoded | None
        Derivation entropy per item, or None where none was published.
    mbr : Encoded | None
        The MBR tree as a constituent indicator per span, or None where unavailable.
    derivations : DerivationScoreChart[Hashable] | None
        Gold derivation scoring against the parser's normal-form-refined chart.
    truncated : tuple[tuple[int, ...], ...]
        The positions of the words the token encoder omitted from each item, one
        entry per item, or empty where no token encoding was published.

    Raises
    ------
    TypeError
        When the debug checks are on and a readout id is not a string.
    ValueError
        When the debug checks are on and a readout id is empty, or a readout is over
        another span addressing than the bundle.
    """

    __slots__ = (
        "_actions",
        "_batch",
        "_cells",
        "_chart",
        "_choice_rows",
        "_derivations",
        "_entailment",
        "_entropy",
        "_forms",
        "_marginals",
        "_mbr",
        "_outside_cells",
        "_readouts",
        "_scores",
        "_spans",
        "_splits",
        "_terms",
        "_truncated",
    )

    def __init__(
        self,
        *,
        spans: SpanIndex,
        chart: CellChart,
        batch: BatchT,
        splits: SplitPosterior | None = None,
        scores: LogDistribution | None = None,
        marginals: SpanMarginals | None = None,
        actions: Mapping[Action, LogDistribution] | None = None,
        readouts: Mapping[ReadoutId, SpanReadout] | None = None,
        choice_rows: Mapping[ReadoutId, SpanReadout] | None = None,
        terms: Mapping[ReadoutId, SpanTerms] | None = None,
        forms: Mapping[ReadoutId, WeightedTerms] | None = None,
        entailment: GroundedBatch | None = None,
        cells: Encoded | None = None,
        outside_cells: Encoded | None = None,
        entropy: Encoded | None = None,
        mbr: Encoded | None = None,
        derivations: DerivationScoreChart[Hashable] | None = None,
        truncated: tuple[tuple[int, ...], ...] = (),
    ) -> None:
        if CHECKS:
            for key, readout in (readouts or {}).items():
                ensure_name("a readout id", key)

                if readout.spans != spans:
                    msg = (
                        f"the readout {key!r} is over {readout.spans!r}, and the "
                        f"bundle over {spans!r}"
                    )
                    raise ValueError(msg)

            for key, rows in (choice_rows or {}).items():
                ensure_name("a readout id", key)

                if rows.spans != spans:
                    msg = (
                        f"the choice rows {key!r} are over {rows.spans!r}, and the "
                        f"bundle over {spans!r}"
                    )
                    raise ValueError(msg)

            for key in forms or {}:
                ensure_name("a readout id", key)

            for key, decoded in (terms or {}).items():
                ensure_name("a readout id", key)

                if decoded.spans != spans:
                    msg = (
                        f"the decoded terms {key!r} are over {decoded.spans!r}, "
                        f"and the bundle over {spans!r}"
                    )
                    raise ValueError(msg)

        self._spans = spans
        self._chart = chart
        self._batch = batch
        self._splits = splits
        self._scores = scores
        self._marginals = marginals
        self._actions: Mapping[Action, LogDistribution] = MappingProxyType(
            dict(actions or {})
        )
        self._readouts: Mapping[ReadoutId, SpanReadout] = MappingProxyType(
            dict(readouts or {})
        )
        self._choice_rows: Mapping[ReadoutId, SpanReadout] = MappingProxyType(
            dict(choice_rows or {})
        )
        self._terms: Mapping[ReadoutId, SpanTerms] = MappingProxyType(dict(terms or {}))
        self._forms: Mapping[ReadoutId, WeightedTerms] = MappingProxyType(
            dict(forms or {})
        )
        self._entailment = entailment
        self._cells = cells
        self._outside_cells = outside_cells
        self._entropy = entropy
        self._mbr = mbr
        self._derivations = derivations
        self._truncated = truncated

    @property
    def spans(self) -> SpanIndex:
        """Return the span addressing every chart index in the bundle is over."""
        return self._spans

    @property
    def chart(self) -> CellChart:
        """Return per-span inside and outside scores."""
        return self._chart

    @property
    def splits(self) -> SplitPosterior | None:
        """Return per-span split posteriors, if available."""
        return self._splits

    @property
    def scores(self) -> LogDistribution | None:
        """Return per-item log inside scores, if available."""
        return self._scores

    @property
    def marginals(self) -> SpanMarginals | None:
        """Return the chart's two marginals if published."""
        return self._marginals

    @property
    def actions(self) -> Mapping[Action, LogDistribution]:
        """Return the log weight of each action per span, keyed by the action itself."""
        return self._actions

    @property
    def readouts(self) -> Mapping[ReadoutId, SpanReadout]:
        """Return what each readout published, keyed by the readout's id."""
        return self._readouts

    @property
    def choice_rows(self) -> Mapping[ReadoutId, SpanReadout]:
        """Return payload-aware decoder marginals keyed by readout id."""
        return self._choice_rows

    @property
    def terms(self) -> Mapping[ReadoutId, SpanTerms]:
        """Return weighted complete terms keyed by their readout id."""
        return self._terms

    @property
    def forms(self) -> Mapping[ReadoutId, WeightedTerms]:
        """Return weighted logical forms keyed by their side id."""
        return self._forms

    @property
    def entailment(self) -> GroundedBatch | None:
        """Return the grounded inference distribution, if one was published."""
        return self._entailment

    @property
    def cells(self) -> Encoded | None:
        """Return the representation of each span if published."""
        return self._cells

    @property
    def outside_cells(self) -> Encoded | None:
        """Return each span's contextual outside representation if published."""
        return self._outside_cells

    @property
    def entropy(self) -> Encoded | None:
        """Return derivation entropy per item if published."""
        return self._entropy

    @property
    def mbr(self) -> Encoded | None:
        """Return MBR-tree constituent indicators if published."""
        return self._mbr

    @property
    def derivations(self) -> DerivationScoreChart[Hashable] | None:
        """Return gold derivation scoring if published."""
        return self._derivations

    @property
    def batch(self) -> BatchT:
        """Return the input batch passed through the neural layer."""
        return self._batch

    @property
    def truncated(self) -> tuple[tuple[int, ...], ...]:
        """Return the positions of the words omitted from each item, if published."""
        return self._truncated

    @property
    def published(self) -> frozenset[str]:
        """Return the names this bundle has a value for."""
        present = {
            "spans": True,
            "chart": True,
            "batch": True,
            "splits": self._splits is not None,
            "scores": self._scores is not None,
            "marginals": self._marginals is not None,
            "actions": bool(self._actions),
            "readouts": bool(self._readouts),
            "choice_rows": bool(self._choice_rows),
            "terms": bool(self._terms),
            "forms": bool(self._forms),
            "entailment": self._entailment is not None,
            "cells": self._cells is not None,
            "outside_cells": self._outside_cells is not None,
            "entropy": self._entropy is not None,
            "mbr": self._mbr is not None,
            "derivations": self._derivations is not None,
            OUTSIDE_CHART: self._chart.outside is not None,
        }

        return frozenset(name for name, available in present.items() if available)

    @override
    def __repr__(self) -> str:
        """Return the names the bundle has a value for."""
        names = ", ".join(sorted(self.published))

        return f"ObjectiveOutputs({names})"


class RecordingOutputs[BatchT]:
    """Restricted objective access to published outputs.

    Parameters
    ----------
    outputs : ObjectiveOutputs[BatchT]
        The bundle to read through.
    objective : str
        The name of the objective reading it, for the message.
    requires : frozenset[str]
        The names that objective declared.
    """

    __slots__ = ("_admitted", "_objective", "_outputs", "_touched")

    def __init__(
        self,
        outputs: ObjectiveOutputs[BatchT],
        *,
        objective: str,
        requires: frozenset[str],
    ) -> None:
        self._outputs = outputs
        self._objective = objective
        self._admitted = frozenset(_field_of(name) for name in requires)
        self._touched: set[str] = set()

    @property
    def touched(self) -> frozenset[str]:
        """Return the fields read through this proxy so far."""
        return frozenset(self._touched)

    def _admit(self, field: str) -> None:
        """Record a read of one field.

        Parameters
        ----------
        field : str
            The field being read.

        Raises
        ------
        UndeclaredOutputError
            If the objective did not declare the field.
        """
        if field not in self._admitted:
            raise UndeclaredOutputError(
                objective=self._objective,
                field=field,
                declared=self._admitted,
            )

        self._touched.add(field)

    @property
    def spans(self) -> SpanIndex:
        """Return the span addressing every chart index in the bundle is over."""
        self._admit("spans")

        return self._outputs.spans

    @property
    def chart(self) -> CellChart:
        """Return per-span inside and outside scores."""
        self._admit("chart")

        return self._outputs.chart

    @property
    def splits(self) -> SplitPosterior | None:
        """Return per-span split posteriors, if available."""
        self._admit("splits")

        return self._outputs.splits

    @property
    def scores(self) -> LogDistribution | None:
        """Return per-item log inside scores, if available."""
        self._admit("scores")

        return self._outputs.scores

    @property
    def marginals(self) -> SpanMarginals | None:
        """Return the chart's two marginals if published."""
        self._admit("marginals")

        return self._outputs.marginals

    @property
    def actions(self) -> Mapping[Action, LogDistribution]:
        """Return the log weight of each action per span, keyed by the action itself."""
        self._admit("actions")

        return self._outputs.actions

    @property
    def readouts(self) -> Mapping[ReadoutId, SpanReadout]:
        """Return what each readout published, keyed by the readout's id."""
        self._admit("readouts")

        return self._outputs.readouts

    @property
    def choice_rows(self) -> Mapping[ReadoutId, SpanReadout]:
        """Return payload-aware decoder marginals keyed by readout id."""
        self._admit("choice_rows")

        return self._outputs.choice_rows

    @property
    def terms(self) -> Mapping[ReadoutId, SpanTerms]:
        """Return weighted complete terms keyed by their readout id."""
        self._admit("terms")

        return self._outputs.terms

    @property
    def forms(self) -> Mapping[ReadoutId, WeightedTerms]:
        """Return weighted logical forms keyed by their side id."""
        self._admit("forms")

        return self._outputs.forms

    @property
    def entailment(self) -> GroundedBatch | None:
        """Return the grounded inference distribution, if one was published."""
        self._admit("entailment")

        return self._outputs.entailment

    @property
    def cells(self) -> Encoded | None:
        """Return the representation of each span if published."""
        self._admit("cells")

        return self._outputs.cells

    @property
    def outside_cells(self) -> Encoded | None:
        """Return each span's contextual outside representation if published."""
        self._admit("outside_cells")

        return self._outputs.outside_cells

    @property
    def entropy(self) -> Encoded | None:
        """Return derivation entropy per item if published."""
        self._admit("entropy")

        return self._outputs.entropy

    @property
    def mbr(self) -> Encoded | None:
        """Return MBR-tree constituent indicators if published."""
        self._admit("mbr")

        return self._outputs.mbr

    @property
    def derivations(self) -> DerivationScoreChart[Hashable] | None:
        """Return gold derivation scoring if published."""
        self._admit("derivations")

        return self._outputs.derivations

    @property
    def batch(self) -> BatchT:
        """Return the input batch passed through the neural layer."""
        self._admit("batch")

        return self._outputs.batch

    @override
    def __repr__(self) -> str:
        """Return the objective name and accessed outputs."""
        read = ", ".join(sorted(self._touched)) or "nothing"

        return f"RecordingOutputs({self._objective}, read={read})"
