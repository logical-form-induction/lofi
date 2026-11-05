"""Vector-space anchors for constrained chart cells."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

import torch
from torch import nn
from torch.nn import functional

from lofi._checks import ensure_finite, ensure_name
from lofi._types import ReadoutId
from lofi.nn.loss._constraint import BatchLocalConstraint
from lofi.nn.loss._errors import MissingOutputError, UncomputedCellError
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import CELL_AXES
from lofi.registry import ComponentPlugin
from lofi.spec.losses.vector_anchor import (
    AnchorAggregateName,
    AnchorDistanceName,
    VectorAnchorLossSettings,
    VectorAnchorLossSpec,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn._values import Encoded
    from lofi.nn.loss._ports import (
        ConstrainedItems,
        LossComponentBindings,
        PublishedOutputs,
    )
    from lofi.nn.ports import TermEncoder
    from lofi.terms import BatchEncoding, Term

__all__ = ["AnchorAggregateName", "AnchorDistanceName", "VectorAnchorObjective"]

_CELL_RANK = len(CELL_AXES)
"""The rank of a chart-cell representation."""

_CELL_MASK_RANK = _CELL_RANK - 1
"""The widest mask that selects whole chart cells."""

_TARGET_AXES = ("term", "width")
"""The axes of encoded target vectors."""


def _pairwise_distance(
    selected: Tensor, target: Tensor, distance: AnchorDistanceName, /
) -> Tensor:
    """Return every selected-cell distance to every target anchor."""
    if distance is AnchorDistanceName.COSINE:
        return 1.0 - functional.cosine_similarity(
            selected.unsqueeze(1), target.unsqueeze(0), dim=-1
        )

    return (selected.unsqueeze(1) - target.unsqueeze(0)).pow(2).sum(dim=-1)


def _aggregate_distance(distances: Tensor, aggregate: AnchorAggregateName, /) -> Tensor:
    """Return one distance for each selected cell."""
    if aggregate is AnchorAggregateName.NEAREST:
        return distances.min(dim=-1).values

    return distances.mean(dim=-1)


def _targets_by_readout(
    targets: Mapping[ReadoutId, BatchEncoding],
) -> Mapping[ReadoutId, BatchEncoding]:
    """Return validated target encodings keyed by readout id.

    Parameters
    ----------
    targets : Mapping[ReadoutId, BatchEncoding]
        The target terms for each localized constraint identity.

    Raises
    ------
    TypeError
        If a readout id is not a string.
    ValueError
        If no target set is given, a readout id is empty, or a target set is empty.
    """
    checked: dict[ReadoutId, BatchEncoding] = {}

    for readout, terms in targets.items():
        key = ReadoutId(ensure_name("a vector-anchor readout id", readout))

        if not terms.rows:
            msg = f"the vector-anchor target {key!r} has no term"
            raise ValueError(msg)

        checked[key] = terms

    if not checked:
        msg = "a vector-anchor objective needs at least one target set"
        raise ValueError(msg)

    return MappingProxyType(checked)


def _checked_target(readout: ReadoutId, target: Encoded, *, count: int) -> Tensor:
    """Return one detached target-vector matrix.

    Parameters
    ----------
    readout : ReadoutId
        The identity the target vectors are stored under.
    target : Encoded
        The vectors returned by the term encoder.
    count : int
        The number of target terms supplied to the encoder.

    Raises
    ------
    TypeError
        If the encoder returns non-floating values.
    ValueError
        If the vectors have the wrong axes or row count, have no feature, or mask a
        target term.
    """
    if target.axes != _TARGET_AXES:
        msg = (
            f"the target encoder returned {target.axes} for {readout!r}, and vector "
            f"anchors read {_TARGET_AXES}"
        )
        raise ValueError(msg)

    if target.tensor.shape[0] != count:
        msg = (
            f"the target encoder returned {target.tensor.shape[0]} vectors for "
            f"{count} terms under {readout!r}"
        )
        raise ValueError(msg)

    if target.tensor.shape[1] == 0:
        msg = f"the target encoder returned empty vectors under {readout!r}"
        raise ValueError(msg)

    if not target.tensor.is_floating_point():
        msg = f"the target vectors under {readout!r} are not floating point"
        raise TypeError(msg)

    if target.mask is not None and not bool(target.mask.all()):
        msg = f"the target encoder masked a term under {readout!r}"
        raise ValueError(msg)

    return target.tensor.detach().clone()


def _buffer_name(readout: ReadoutId) -> str:
    """Return a stable buffer name for a readout id.

    Parameters
    ----------
    readout : ReadoutId
        The target identity to encode.
    """
    return f"target_{readout.encode('utf-8').hex()}"


def _live_cells(cells: Encoded, spans: SpanIndex) -> Tensor:
    """Return which chart cells hold vectors.

    Parameters
    ----------
    cells : Encoded
        The published chart-cell representations.
    spans : SpanIndex
        The addressing of the span axis.

    Raises
    ------
    ValueError
        If the vectors do not run over the declared cell axes and span index, or if
        their mask divides a vector's features.
    """
    if cells.axes != CELL_AXES:
        msg = f"vector anchors read cells over {CELL_AXES}, not {cells.axes}"
        raise ValueError(msg)

    if cells.tensor.ndim != _CELL_RANK:
        msg = f"a chart cell is one vector, not a tensor of rank {cells.tensor.ndim}"
        raise ValueError(msg)

    if cells.tensor.shape[1] != spans.num_spans:
        msg = (
            f"the chart holds {cells.tensor.shape[1]} spans, and its index addresses "
            f"{spans.num_spans}"
        )
        raise ValueError(msg)

    mask = cells.mask
    shape = tuple(cells.tensor.shape[:_CELL_MASK_RANK])

    if mask is None:
        return cells.tensor.new_ones(shape, dtype=torch.bool)

    if mask.ndim > _CELL_MASK_RANK:
        msg = (
            f"the chart cells are masked over {cells.mask_axes}, so the mask divides "
            "a cell's features"
        )
        raise ValueError(msg)

    if mask.ndim == 1:
        return mask.unsqueeze(1).expand(shape)

    return mask


class VectorAnchorObjective[T, ScopeT: Hashable]:
    """Binds target terms for a vector-space chart objective.

    Parameters
    ----------
    targets : Mapping[ReadoutId, BatchEncoding]
        The target terms for each localized constraint identity.
    name : str
        The name the term is logged and namespaced under.
    distance : AnchorDistanceName
        Geometry used between cells and anchors.
    aggregate : AnchorAggregateName
        How distances to several permitted anchors are combined.
    margin : float
        Minimum distance required by a negated constraint.

    Raises
    ------
    TypeError
        If a readout id or the name is not a string.
    ValueError
        If no target set is given, a target set is empty, or an id or name is empty.
    """

    __slots__ = ("_aggregate", "_distance", "_margin", "_name", "_targets")

    def __init__(
        self,
        *,
        targets: Mapping[ReadoutId, BatchEncoding],
        name: str,
        distance: AnchorDistanceName,
        aggregate: AnchorAggregateName,
        margin: float,
    ) -> None:
        self._targets = _targets_by_readout(targets)
        self._name = ensure_name("the objective's name", name)
        self._distance = distance
        self._aggregate = aggregate
        checked_margin = ensure_finite("the vector-anchor margin", margin)

        if checked_margin < 0.0:
            msg = f"the vector-anchor margin is nonnegative, not {margin!r}"
            raise ValueError(msg)

        self._margin = checked_margin

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs the bound objective reads."""
        return frozenset({"batch", "cells", "spans"})

    @property
    def targets(self) -> Mapping[ReadoutId, BatchEncoding]:
        """Return the target terms keyed by localized constraint identity."""
        return self._targets

    @property
    def distance(self) -> AnchorDistanceName:
        """Return the geometry used between cells and anchors."""
        return self._distance

    @property
    def aggregate(self) -> AnchorAggregateName:
        """Return how distances to several anchors are combined."""
        return self._aggregate

    @property
    def margin(self) -> float:
        """Return the minimum distance required by a negated constraint."""
        return self._margin

    def bind(self, context: TermEncoder, /) -> _BoundVectorAnchorObjective[T, ScopeT]:
        """Return an objective holding the encoder's detached target vectors.

        Parameters
        ----------
        context : TermEncoder
            The trained term encoder that maps target terms into chart-vector space.

        Raises
        ------
        TypeError
            If an encoded target is not floating point.
        ValueError
            If an encoded target has incompatible axes, shape, width, or mask.
        """
        vectors: dict[ReadoutId, Tensor] = {}

        with torch.no_grad():
            for readout in sorted(self._targets):
                terms = self._targets[readout]
                vectors[readout] = _checked_target(
                    readout,
                    context.encode(terms),
                    count=len(terms.rows),
                )

        return _BoundVectorAnchorObjective(
            name=self._name,
            targets=vectors,
            distance=self._distance,
            aggregate=self._aggregate,
            margin=self._margin,
        )

    @override
    def __repr__(self) -> str:
        """Return the name and target readout ids."""
        return (
            f"VectorAnchorObjective(name={self._name!r}, "
            f"targets={tuple(sorted(self._targets))!r}, "
            f"distance={self._distance!r}, aggregate={self._aggregate!r}, "
            f"margin={self._margin!r})"
        )


class _BoundVectorAnchorObjective[T, ScopeT: Hashable](nn.Module):
    """Scores constrained chart cells against fixed target vectors."""

    def __init__(
        self,
        *,
        name: str,
        targets: Mapping[ReadoutId, Tensor],
        distance: AnchorDistanceName,
        aggregate: AnchorAggregateName,
        margin: float,
    ) -> None:
        super().__init__()
        self._name = name
        self._distance = distance
        self._aggregate = aggregate
        self._margin = margin
        self._buffers_by_readout = MappingProxyType(
            {readout: _buffer_name(readout) for readout in sorted(targets)}
        )

        for readout, buffer in self._buffers_by_readout.items():
            self.register_buffer(buffer, targets[readout])

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"batch", "cells", "spans"})

    @property
    def readouts(self) -> tuple[ReadoutId, ...]:
        """Return the target identities in stable order."""
        return tuple(self._buffers_by_readout)

    @property
    def distance(self) -> AnchorDistanceName:
        """Return the geometry used between cells and anchors."""
        return self._distance

    @property
    def aggregate(self) -> AnchorAggregateName:
        """Return how distances to several anchors are combined."""
        return self._aggregate

    @property
    def margin(self) -> float:
        """Return the minimum distance required by a negated constraint."""
        return self._margin

    def _target(self, readout: ReadoutId) -> Tensor:
        """Return the target vectors stored under a readout id.

        Parameters
        ----------
        readout : ReadoutId
            The localized constraint identity to read.

        Raises
        ------
        ValueError
            If no target was bound under the id.
        """
        buffer = self._buffers_by_readout.get(readout)

        if buffer is None:
            declared = ", ".join(self._buffers_by_readout)
            msg = (
                f"{self._name} is handed a constraint reading {readout!r}, which has "
                f"no target; declared: {declared}"
            )
            raise ValueError(msg)

        return self.get_buffer(buffer)

    def evaluate(
        self, outputs: PublishedOutputs[ConstrainedItems[T, ScopeT]], /
    ) -> LossTerm:
        """Return the mean declared-distance penalty over constrained cells.

        Positive constraints minimize the declared aggregate distance. Negated
        constraints use a hinge that is zero once that distance reaches the declared
        margin.

        Parameters
        ----------
        outputs : PublishedOutputs[ConstrainedItems[T, ScopeT]]
            What the graph published for the step.

        Raises
        ------
        MissingOutputError
            If the step published no chart-cell vectors.
        UncomputedCellError
            If a constraint addresses a masked chart cell.
        IndexError
            If a constraint names an item outside the batch.
        TypeError
            If the batch hands over a negative bracket, which names no target.
        ValueError
            If the cells have incompatible axes, span addressing, width, or mask, a
            constraint span falls outside the index, or no target was bound under its
            readout id.
        """
        cells = outputs.cells

        if cells is None:
            raise MissingOutputError(
                objective=self._name,
                missing=frozenset({"cells"}),
                published=frozenset(),
            )

        live = _live_cells(cells, outputs.spans)
        items = int(cells.tensor.shape[0])
        grouped: dict[ReadoutId, list[tuple[int, int, bool, float]]] = {
            readout: [] for readout in self._buffers_by_readout
        }

        for constraint in outputs.batch.constraints:
            if not isinstance(constraint, BatchLocalConstraint):
                msg = (
                    f"{self._name} anchors target-bearing constraints and cannot "
                    f"score a negative bracket: {constraint!r}"
                )
                raise TypeError(msg)

            target = self._target(constraint.readout)

            if target.shape[1] != cells.tensor.shape[2]:
                msg = (
                    f"the target vectors under {constraint.readout!r} have width "
                    f"{target.shape[1]}, and the chart cells have width "
                    f"{cells.tensor.shape[2]}"
                )
                raise ValueError(msg)

            held = grouped[constraint.readout]

            for item in constraint.item_positions:
                if item >= items:
                    msg = f"there is no item {item} in a batch of {items} items"
                    raise IndexError(msg)

                for start, end in constraint.spans:
                    if end > outputs.spans.n:
                        msg = (
                            f"the span [{start}, {end}) reaches past an input of "
                            f"length {outputs.spans.n}"
                        )
                        raise ValueError(msg)

                    span = outputs.spans.span_id(start, end)

                    if not bool(live[item, span]):
                        raise UncomputedCellError(item=item, span=(start, end))

                    held.append(
                        (
                            item,
                            span,
                            constraint.negate,
                            1.0 if constraint.penalty is None else constraint.penalty,
                        )
                    )

        losses: list[Tensor] = []
        gradient_scales: list[Tensor] = []
        count = 0

        for readout, instances in grouped.items():
            if not instances:
                continue

            target = self._target(readout)
            item_ids = torch.tensor(
                [item for item, _, _, _ in instances], device=cells.tensor.device
            )
            span_ids = torch.tensor(
                [span for _, span, _, _ in instances], device=cells.tensor.device
            )
            selected = cells.tensor[item_ids, span_ids]
            distance = _aggregate_distance(
                _pairwise_distance(selected, target, self._distance), self._aggregate
            )
            negated = torch.tensor(
                [held for _, _, held, _ in instances], device=cells.tensor.device
            )
            weights = cells.tensor.new_tensor([weight for _, _, _, weight in instances])
            positive = distance
            negative = functional.relu(self._margin - distance)
            losses.append(torch.where(negated, negative, positive) * weights)
            gradient_scales.append(weights)
            count += len(instances)

        if not losses:
            return LossTerm.inapplicable(cells.tensor)

        return LossTerm(
            torch.cat(losses).mean(),
            count,
            diagnostics={"gradient_scale": torch.cat(gradient_scales).detach().mean()},
        )

    @override
    def __repr__(self) -> str:
        """Return the name and target readout ids."""
        return (
            f"_BoundVectorAnchorObjective(name={self._name!r}, "
            f"targets={tuple(self._buffers_by_readout)!r}, "
            f"distance={self._distance!r}, aggregate={self._aggregate!r}, "
            f"margin={self._margin!r})"
        )


def _build(
    settings: VectorAnchorLossSettings, context: LossComponentBindings, /
) -> _BoundVectorAnchorObjective[Term, Hashable]:
    """Build anchors at the term encoder's vectors for one readout's target sets.

    The vectors are those of the term encoder as it stands when the objective is
    built, and they are held fixed from then on.

    Parameters
    ----------
    settings : VectorAnchorLossSettings
        The term-readout role whose target sets are anchored.
    context : LossComponentBindings
        The declared name, the built components the role names, and the type
        encoder.

    Raises
    ------
    ValueError
        If the role is not built, the graph built no term encoder, the readout
        holds no target set, or an encoded target has incompatible axes, shape,
        width, or mask.
    """
    readout = context.terms_readout(settings.readout)
    anchors: VectorAnchorObjective[Term, Hashable] = VectorAnchorObjective(
        targets=readout.target_encodings,
        name=context.name,
        distance=settings.distance,
        aggregate=settings.aggregate,
        margin=settings.margin,
    )

    return anchors.bind(context.readout_space(settings.readout).encoder)


PLUGIN: Final = ComponentPlugin(spec=VectorAnchorLossSpec, build=_build)
"""The vector-anchor objective's plugin."""
