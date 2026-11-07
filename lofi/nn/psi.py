"""Rule and split scores for chart parsing."""

from math import inf
from typing import TYPE_CHECKING, Final, override

import torch
from torch import nn

from lofi.nn._values import Encoded, LogDistribution

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.judge import ActionSet
    from lofi.nn.ports import Controller

__all__ = ["RuleScorer", "RuleScores"]

_ITEM_AXIS: Final[str] = "item"
"""The batch axis."""

_SPAN_AXIS: Final[str] = "span"
"""The span axis."""

_SPLIT_AXIS: Final[str] = "split"
"""The split axis."""

_ACTION_AXIS: Final[str] = "action"
"""The action axis."""

_WIDTH_AXIS: Final[str] = "width"
"""The feature axis."""

_CELL_AXES: Final[tuple[str, ...]] = (_SPAN_AXIS, _ITEM_AXIS, _WIDTH_AXIS)
"""Packed chart cell axes."""

_PAIR_AXES: Final[tuple[str, ...]] = ("cell", _WIDTH_AXIS)
"""Controller input axes."""

_WEIGHT_AXES: Final[tuple[str, ...]] = ("cell", _ACTION_AXIS)
"""Controller output axes."""

_SCORE_AXES: Final[tuple[str, ...]] = (
    _ITEM_AXIS,
    _SPAN_AXIS,
    _SPLIT_AXIS,
    _ACTION_AXIS,
)
"""Rule-score axis order."""

_SPLIT_AXES: Final[tuple[str, ...]] = (_ITEM_AXIS, _SPAN_AXIS, _SPLIT_AXIS)
"""Split-score axes after action marginalization."""

_ACTION_DIM: Final[int] = _SCORE_AXES.index(_ACTION_AXIS)
"""The action-axis position."""


def _gathered(
    cells: Tensor,
    children: tuple[tuple[int, ...], ...],
    /,
    *,
    items: int,
    splits: int,
) -> Tensor:
    """Return one child cell per item, span and split, laid out as a run of pairs.

    Parameters
    ----------
    cells : Tensor
        The chart's cells, rank 3 of shape (span, item, width).
    children : tuple[tuple[int, ...], ...]
        Per span id, the span ids of that side's children over all splits.
    items : int
        How many items the batch holds.
    splits : int
        How many split positions a span's row holds.

    Returns
    -------
    Tensor
        Rank 2 of shape (item * span * split, width) in the cells' floating point dtype,
        running over items outermost and splits innermost.
    """
    spans = int(cells.shape[0])
    width = int(cells.shape[2])
    ids = cells.new_tensor(children, dtype=torch.long).reshape(-1)
    found = cells.index_select(0, ids).reshape(spans, splits, items, width)

    return found.permute(2, 0, 1, 3).reshape(items * spans * splits, width)


class RuleScores:
    """Stores action and split scores for a parser pass.

    Parameters
    ----------
    scores : Tensor
        The scores before the split mask reaches them, rank 4 of shape (item, span,
        split, action) at floating precision.
    mask : Tensor
        True where a span has a real split there, rank 3 of shape (item, span, split).
    index : SpanIndex
        The span addressing the scores are laid out under.
    null : int | None
        The action column holding the weight that no combinator applies, which a
        split's weight leaves out, or None where the action set declares no null
        action.

    Raises
    ------
    TypeError
        If the scores are not of a floating point dtype, or the mask is not boolean.
    ValueError
        If the scores are not rank 4, if the mask does not cover their three outermost
        axes, if the span and split axes are not the ones the index addresses, or if
        the null column is not one of the action columns.
    """

    __slots__ = ("_index", "_mask", "_null", "_raw", "_scores", "_splits")

    def __init__(
        self, scores: Tensor, mask: Tensor, index: SpanIndex, *, null: int | None
    ) -> None:
        if not scores.is_floating_point():
            msg = f"a rule score is a log weight, not {scores.dtype}"
            raise TypeError(msg)

        if scores.ndim != len(_SCORE_AXES):
            msg = (
                f"a rule score runs over {_SCORE_AXES}, so it has rank "
                f"{len(_SCORE_AXES)}, not {scores.ndim}"
            )
            raise ValueError(msg)

        if mask.dtype is not torch.bool:
            msg = f"a split mask holds booleans, not {mask.dtype}"
            raise TypeError(msg)

        if tuple(mask.shape) != tuple(scores.shape[: len(_SPLIT_AXES)]):
            msg = (
                f"a mask of shape {tuple(mask.shape)} does not cover "
                f"{_SPLIT_AXES} of a score of shape {tuple(scores.shape)}"
            )
            raise ValueError(msg)

        wanted = (index.num_spans, index.max_splits)
        held = (int(scores.shape[1]), int(scores.shape[2]))

        if held != wanted:
            msg = (
                f"an index over {wanted[0]} spans of at most {wanted[1]} splits "
                f"was given scores over {held[0]} and {held[1]}"
            )
            raise ValueError(msg)

        columns = int(scores.shape[_ACTION_DIM])

        if null is not None and not 0 <= null < columns:
            msg = f"the null column {null} is not one of the {columns} action columns"
            raise ValueError(msg)

        self._raw = scores
        self._mask = mask
        self._index = index
        self._null = null
        self._scores: LogDistribution | None = None
        self._splits: LogDistribution | None = None

    @property
    def index(self) -> SpanIndex:
        """Return the span index used by these scores."""
        return self._index

    @property
    def mask(self) -> Tensor:
        """Return the rank-3 valid-split mask over items, spans, and splits."""
        return self._mask

    @property
    def log_scores(self) -> LogDistribution:
        """Return the log weight of every action at every split of every span.

        Returns
        -------
        LogDistribution
            Rank 4 over the axes ``("item", "span", "split", "action")``, at the
            scorer's own floating precision and unnormalized, masked over the item, span
            and split axes.
        """
        held = self._scores

        if held is None:
            filled = self._raw.masked_fill(~self._mask.unsqueeze(_ACTION_DIM), -inf)
            held = LogDistribution(filled, _SCORE_AXES, self._mask)
            self._scores = held

        return held

    @property
    def split_scores(self) -> LogDistribution:
        """Return split scores marginalized over every action but the null action.

        Returns
        -------
        LogDistribution
            Rank 3 over the axes ``("item", "span", "split")``, at the scorer's own
            floating precision and unnormalized, masked over all three.
        """
        held = self._splits

        if held is None:
            raw = self._raw
            null = self._null

            if null is not None:
                raw = torch.cat(
                    (raw[..., :null], raw[..., null + 1 :]), dim=_ACTION_DIM
                )

            total = raw.logsumexp(dim=_ACTION_DIM).masked_fill(~self._mask, -inf)
            held = LogDistribution(total, _SPLIT_AXES, self._mask)
            self._splits = held

        return held

    @override
    def __repr__(self) -> str:
        """Return the shape the scores were computed over."""
        return f"RuleScores(shape={tuple(self._raw.shape)}, n={self._index.n})"


class RuleScorer(nn.Module):
    """Scores actions for every valid span split.

    Parameters
    ----------
    controller : Controller
        What weighs an action against a pair of cells.
    width : int
        How wide a chart cell's vector is.
    register_controller : bool
        Whether this scorer owns the controller as a submodule. A recurrent chart
        registers its controller once on the stack and gives each layer a borrowing
        scorer.
    """

    def __init__(
        self,
        controller: Controller,
        /,
        *,
        width: int,
        register_controller: bool = True,
    ) -> None:
        super().__init__()

        if register_controller:
            self._controller = controller
        else:
            object.__setattr__(self, "_controller", controller)

        self._compatibility = nn.Linear(width, width, bias=False)
        self._compatibility_bias = nn.Parameter(torch.zeros(1))
        self._width = width

    @property
    def action_set(self) -> ActionSet:
        """Return the actions weighed at a split, in the order the columns take."""
        return self._controller.action_set

    @property
    def controller(self) -> Controller:
        """Return the controller whose action distribution this scorer adjusts."""
        return self._controller

    @property
    def input_width(self) -> int:
        """Return the size of the feature axis of the cells this scorer reads."""
        return self._width

    def score(self, cells: Encoded, /, *, index: SpanIndex) -> RuleScores:
        """Return the log weight of every action at every split of every span.

        Parameters
        ----------
        cells : Encoded
            The chart's cells, over the axes ``("span", "item", "width")``, which is the
            packed store's own layout.
        index : SpanIndex
            The span addressing the cells are laid out under.

        Returns
        -------
        RuleScores
            The scores over the axes ``("item", "span", "split", "action")`` and the
            split mask they are read under.

        Raises
        ------
        ValueError
            If cell axes or extents disagree with the scorer or index, or controller
            weights use different axes.
        """
        if cells.axes != _CELL_AXES:
            msg = f"a chart's cells run over {_CELL_AXES}, not over {cells.axes}"
            raise ValueError(msg)

        if cells.size(_SPAN_AXIS) != index.num_spans:
            msg = (
                f"an index over {index.num_spans} spans was handed cells for "
                f"{cells.size(_SPAN_AXIS)}"
            )
            raise ValueError(msg)

        if cells.size(_WIDTH_AXIS) != self._width:
            msg = (
                f"this scorer reads cells {self._width} wide, not "
                f"{cells.size(_WIDTH_AXIS)}"
            )
            raise ValueError(msg)

        data = cells.tensor
        items = cells.size(_ITEM_AXIS)
        splits = index.max_splits
        left = _gathered(data, index.left_child_ids, items=items, splits=splits)
        right = _gathered(data, index.right_child_ids, items=items, splits=splits)
        weights = self._controller.weigh(
            Encoded(left, _PAIR_AXES), right=Encoded(right, _PAIR_AXES)
        )

        if weights.axes != _WEIGHT_AXES:
            msg = f"a controller weighs over {_WEIGHT_AXES}, not over {weights.axes}"
            raise ValueError(msg)

        compatibility = (self._compatibility(left) * right).sum(-1, keepdim=True)
        scored = weights.tensor + compatibility + self._compatibility_bias
        real = data.new_tensor(index.split_mask, dtype=torch.bool)
        shape = (items, index.num_spans, splits)
        actions = self._controller.action_set
        null = actions.null

        return RuleScores(
            scored.reshape(*shape, weights.size(_ACTION_AXIS)),
            real.unsqueeze(0).expand(shape),
            index,
            null=None if null is None else actions.index(null.name),
        )
