"""Best-derivation and minimum-Bayes-risk decoding over the tag-refined chart."""

from math import inf
from typing import TYPE_CHECKING, Final

import torch

from lofi.nn.parsers._inside import _root_ids
from lofi.nn.parsers._tagged import ScoreChannels, span_marginals
from lofi.nn.parsers._values import TaggedTree

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn.parsers._tagged import NormalFormTransitions, TaggedInsideKernel

__all__ = ["TaggedDecoder"]

_LEXICAL: Final = 0
"""Backpointer kind for a leaf read off the lexical channel."""

_UNARY: Final = 1
"""Backpointer kind for a step read off the unary channel."""

_BINARY: Final = 2
"""Backpointer kind for a step read off the split channel."""

_RECORD_WIDTH: Final = 5
"""Backpointer fields: the kind, two step indices, and two premise ranks."""

type _Weights = list[list[list[float]]]
"""One item's log weights, read by span, then state, then rank."""

type _Records = list[list[list[list[int]]]]
"""One item's backpointers, read by span, then state, then rank, then field."""


class _Ranking[T]:
    """Ranks the derivations of every state of every span by summed log weight.

    Parameters
    ----------
    transitions : NormalFormTransitions[T]
        The compiled normal form whose steps the ranking may take.
    channels : ScoreChannels
        The lexical, combining-rule, and unary log weights.
    index : SpanIndex
        The span addressing every channel is laid out under.
    width : int
        How many derivations a state keeps.
    """

    __slots__ = ("_back", "_channels", "_index", "_transitions", "_weights", "_width")

    def __init__(
        self,
        transitions: NormalFormTransitions[T],
        channels: ScoreChannels,
        index: SpanIndex,
        width: int,
        /,
    ) -> None:
        self._transitions = transitions
        self._channels = channels
        self._index = index
        self._width = width
        self._weights = channels.leaf.new_full(
            (
                int(channels.leaf.shape[0]),
                index.num_spans,
                transitions.chart_states,
                width,
            ),
            -inf,
        )
        self._back = self._records(tuple(self._weights.shape))
        self._sweep()

    @property
    def weights(self) -> Tensor:
        """Return descending log weights over item, span, state and rank."""
        return self._weights

    @property
    def back(self) -> Tensor:
        """Return the choice behind every ranked derivation."""
        return self._back

    def _records(self, shape: tuple[int, ...], /) -> Tensor:
        """Return zeroed backpointers of one slab's shape.

        Parameters
        ----------
        shape : tuple[int, ...]
            The chart shape.
        """
        return torch.zeros(
            (*shape, _RECORD_WIDTH),
            dtype=torch.long,
            device=self._weights.device,
        )

    def _leaves(self) -> Tensor:
        """Return one-token spans holding each category score at the lexical tag."""
        leaf = self._channels.leaf
        transitions = self._transitions
        tags = len(transitions.tags)
        categories = len(transitions.categories)
        shape = (int(leaf.shape[0]), int(leaf.shape[1]))
        ranked = leaf.new_full((*shape, categories, self._width), -inf)
        ranked[..., 0] = leaf
        columns = leaf.new_tensor(
            [
                category * tags + transitions.lexical_tag
                for category in range(categories)
            ],
            dtype=torch.long,
        )

        return leaf.new_full(
            (*shape, transitions.chart_states, self._width), -inf
        ).index_copy(2, columns, ranked)

    def _close(self, weights: Tensor, back: Tensor, unary: Tensor, /) -> None:
        """Fold one slab's retained unary derivations into its ranking, in place.

        Parameters
        ----------
        weights : Tensor
            The accumulated weights.
        back : Tensor
            The binary backpointer chart.
        unary : Tensor
            The unary backpointer chart.
        """
        closure = self._transitions.closure
        width = self._width
        ranks = torch.arange(width, device=weights.device)
        premises = closure.reachable(weights[..., 0].isfinite())
        offset = 0

        for state, edges in closure.schedule:
            pooled = [weights[:, :, state, :]]
            records = [back[:, :, state, :, :]]

            for position, edge in enumerate(edges):
                source = closure.state(edge.source_category, edge.source_tag)
                lifted = weights[:, :, source, :] + unary[..., edge.rule].unsqueeze(-1)
                pooled.append(
                    lifted.masked_fill(~premises[..., source].unsqueeze(-1), -inf)
                )
                record = torch.zeros_like(records[0])
                record[..., 0] = _UNARY
                record[..., 1] = offset + position
                record[..., 2] = ranks
                records.append(record)

            offset += len(edges)
            best = torch.cat(pooled, dim=-1).topk(width, dim=-1)
            weights[:, :, state, :] = best.values
            back[:, :, state, :, :] = torch.cat(records, dim=-2).gather(
                -2, best.indices.unsqueeze(-1).expand(-1, -1, -1, _RECORD_WIDTH)
            )

    def _premises(
        self,
        ids: tuple[tuple[int, ...], ...],
        columns: list[int],
        slab: tuple[int, int],
    ) -> Tensor:
        """Return one premise slot's ranked weights, by span, split and transition.

        Parameters
        ----------
        ids : tuple[tuple[int, ...], ...]
            The state identifiers.
        columns : list[int]
            The participating output columns.
        slab : tuple[int, int]
            The span bounds for this chart slab.
        """
        start, stop = slab
        rows = self._weights.new_tensor(ids, dtype=torch.long)[start:stop].reshape(-1)
        shape = (
            int(self._weights.shape[0]),
            stop - start,
            self._index.max_splits,
            self._transitions.chart_states,
            self._width,
        )

        return (
            self._weights.index_select(1, rows)
            .reshape(shape)
            .index_select(3, self._weights.new_tensor(columns, dtype=torch.long))
        )

    def _combine(self, scores: Tensor, slab: tuple[int, int]) -> tuple[Tensor, Tensor]:
        """Return one slab's ranking and backpointers, built from its children.

        Parameters
        ----------
        scores : Tensor
            The scores to inspect.
        slab : tuple[int, int]
            The span bounds for this chart slab.
        """
        start, stop = slab
        batch = int(self._weights.shape[0])
        spans = stop - start
        width = self._width
        candidates = self._transitions.edges
        index = self._index
        joined = (
            self._premises(
                index.left_child_ids,
                [step.left for step in candidates],
                slab,
            ).unsqueeze(-1)
            + self._premises(
                index.right_child_ids,
                [step.right for step in candidates],
                slab,
            ).unsqueeze(-2)
            + scores[:, start:stop].unsqueeze(-1).unsqueeze(-1)
        )
        grouped: list[list[int]] = [[] for _ in range(self._transitions.chart_states)]

        for position, step in enumerate(candidates):
            grouped[step.output].append(position)

        ranked: list[Tensor] = []
        records: list[Tensor] = []
        per_transition = width * width

        for columns in grouped:
            if not columns:
                ranked.append(self._weights.new_full((batch, spans, width), -inf))
                records.append(self._records((batch, spans, width)))
                continue

            held = self._weights.new_tensor(columns, dtype=torch.long)
            best = (
                joined.index_select(3, held)
                .reshape(batch, spans, -1)
                .topk(width, dim=-1)
            )
            per_split = len(columns) * per_transition
            column = best.indices // per_split
            local = (best.indices - column * per_split) // per_transition
            pair = best.indices - column * per_split - local * per_transition
            left = pair // width
            ranked.append(best.values)
            records.append(
                torch.stack(
                    (
                        torch.full_like(best.indices, _BINARY),
                        column,
                        held[local],
                        left,
                        pair - left * width,
                    ),
                    dim=-1,
                )
            )

        return (torch.stack(ranked, dim=2), torch.stack(records, dim=2))

    def _sweep(self) -> None:
        """Fill the ranking span by span, narrowest first."""
        index = self._index
        transitions = self._transitions
        unary = self._channels.unary
        scores = transitions.mask_scores(self._channels.rule, index).log_scores
        weights = self._leaves()
        records = self._records(tuple(weights.shape))

        if unary is not None:
            self._close(weights, records, unary[:, : index.n])

        self._weights[:, : index.n] = weights
        self._back[:, : index.n] = records

        for span_width in range(2, index.n + 1):
            start = index.span_id(0, span_width)
            stop = start + index.n - span_width + 1

            if transitions.edges:
                weights, records = self._combine(scores, (start, stop))
            else:
                shape = (
                    int(self._weights.shape[0]),
                    stop - start,
                    transitions.chart_states,
                )
                weights = self._weights.new_full((*shape, self._width), -inf)
                records = self._records((*shape, self._width))

            if unary is not None:
                self._close(weights, records, unary[:, start:stop])

            self._weights[:, start:stop] = weights
            self._back[:, start:stop] = records


class _Reader[T]:
    """Reads one item's ranked derivations off its weights and backpointers.

    Parameters
    ----------
    transitions : NormalFormTransitions[T]
        The compiled normal form the backpointers index.
    index : SpanIndex
        The span addressing the chart is laid out under.
    weights : _Weights
        One item's log weights, by span, state and rank.
    back : _Records
        One item's backpointers, by span, state, rank and field.
    """

    __slots__ = ("_back", "_index", "_transitions", "_weights")

    def __init__(
        self,
        transitions: NormalFormTransitions[T],
        index: SpanIndex,
        weights: _Weights,
        back: _Records,
        /,
    ) -> None:
        self._transitions = transitions
        self._index = index
        self._weights = weights
        self._back = back

    def _children(self, span: int, state: int, rank: int, /) -> tuple[TaggedTree, ...]:
        """Return visible children, flattening one hidden prefix state."""
        if state < self._transitions.states:
            return (self.tree(span, state, rank),)

        kind, split, edge_index, left_rank, right_rank = self._back[span][state][rank]

        if kind != _BINARY:
            msg = f"hidden chart state {state} has no combining backpointer"
            raise RuntimeError(msg)

        edge = self._transitions.edges[edge_index]

        if edge.scored:
            msg = f"hidden chart state {state} was concluded by a scored rule edge"
            raise RuntimeError(msg)

        start, end = self._index.span_of(span)
        cut = start + split + 1

        return (
            *self._children(self._index.span_id(start, cut), edge.left, left_rank),
            *self._children(self._index.span_id(cut, end), edge.right, right_rank),
        )

    def tree(self, span: int, state: int, rank: int, /) -> TaggedTree:
        """Return one ranked derivation of one state of one span.

        Parameters
        ----------
        span : int
            The span id the derivation covers.
        state : int
            The flat category and tag column it concludes at.
        rank : int
            Which of that state's derivations to read.

        Returns
        -------
        TaggedTree
            The derivation, with its premises resolved.
        """
        tags = len(self._transitions.tags)
        cell = self._index.span_of(span)
        category, tag = divmod(state, tags)
        total = self._weights[span][state][rank]
        kind, first, second, left_rank, right_rank = self._back[span][state][rank]

        if kind == _LEXICAL:
            return TaggedTree(cell, (category, tag), None, (), total)

        if kind == _UNARY:
            edge = self._transitions.closure.retained[first]
            source = edge.source_category * tags + edge.source_tag

            return TaggedTree(
                cell,
                (category, tag),
                edge.rule,
                (self.tree(span, source, second),),
                total,
            )

        start, end = cell
        cut = start + first + 1
        edge = self._transitions.edges[second]

        if not edge.scored or edge.transition is None:
            msg = f"visible chart state {state} was concluded by a hidden prefix edge"
            raise RuntimeError(msg)

        transition = self._transitions.transitions[edge.transition]
        premises = (
            *self._children(self._index.span_id(start, cut), edge.left, left_rank),
            *self._children(self._index.span_id(cut, end), edge.right, right_rank),
        )

        return TaggedTree(cell, (category, tag), transition.rule, premises, total)


def _supported(scores: Tensor, counts: Tensor, gain: Tensor, /) -> Tensor:
    """Return each step's gain where the chart gives it mass, and -inf elsewhere.

    Parameters
    ----------
    scores : Tensor
        The scores to inspect.
    counts : Tensor
        The expected counts.
    gain : Tensor
        The bracket gain.
    """
    return torch.where(
        (counts > 0.0) & scores.isfinite(), gain, gain.new_full((), -inf)
    )


class TaggedDecoder[T]:
    """Reads derivations off a normal-form-refined chart.

    Parameters
    ----------
    kernel : TaggedInsideKernel[T]
        The kernel whose compiled normal form every decoded derivation obeys. Minimum
        Bayes risk also reads its posterior, so it needs log-sum-exp reduction, but
        k-best ranks by summed log weight under either kernel reduction.
    k : int
        How many derivations each state keeps and each item returns.

    Raises
    ------
    ValueError
        If k is nonpositive.
    """

    __slots__ = ("_k", "_kernel")

    def __init__(self, kernel: TaggedInsideKernel[T], /, *, k: int = 1) -> None:
        if k < 1:
            msg = f"a k-best decode keeps at least one derivation, not {k}"
            raise ValueError(msg)

        self._kernel = kernel
        self._k = k

    @property
    def kernel(self) -> TaggedInsideKernel[T]:
        """Return the kernel whose normal form the derivations obey."""
        return self._kernel

    @property
    def k(self) -> int:
        """Return how many derivations each state keeps."""
        return self._k

    def k_best(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> tuple[tuple[TaggedTree, ...], ...]:
        """Return each item's best few normal-form derivations, heaviest first.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights, the last present
            exactly when the rule set declares a unary rule.
        index : SpanIndex
            The span addressing every channel is laid out under.
        goals : Sequence[T] | None
            The root categories a derivation may conclude at, or None for all of them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        tuple[tuple[TaggedTree, ...], ...]
            One tuple per item, holding that item's derivations in descending order of
            summed log weight, and fewer than k of them where the chart holds fewer. A
            derivation's weight is the sum of the scores its steps read.

        Raises
        ------
        TypeError
            If a channel is not of a floating point dtype, or if the lengths are not of
            an integer dtype.
        ValueError
            If the index is empty, a channel has a shape the index and the transitions
            do not call for, a goal is outside the inventory, or a checked length is
            outside the index.
        """
        transitions = self._kernel.transitions
        width = self._k
        batch = transitions.checked_channels(channels, index)
        columns = transitions.goal_categories(goals)
        tags = len(transitions.tags)
        ranking = _Ranking(transitions, channels, index, width)
        weights = ranking.weights
        admitted = weights.new_tensor(
            [state // tags in columns for state in range(transitions.states)],
            dtype=torch.bool,
        )
        roots = _root_ids(weights[..., 0, 0], index, lengths)
        at_root = weights.gather(
            1, roots.reshape(batch, 1, 1, 1).expand(-1, 1, transitions.states, width)
        ).squeeze(1)
        best = (
            at_root.masked_fill(~admitted.unsqueeze(-1), -inf)
            .reshape(batch, -1)
            .topk(width, dim=-1)
        )
        held: list[_Weights] = weights.tolist()
        table: list[_Records] = ranking.back.tolist()
        chosen = best.indices.tolist()
        totals = best.values.tolist()

        return tuple(
            tuple(
                _Reader(transitions, index, held[item], table[item]).tree(
                    root, flat // width, flat % width
                )
                for flat, total in zip(chosen[item], totals[item], strict=True)
                if total != -inf
            )
            for item, root in enumerate(roots.tolist())
        )

    def mbr_trees(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> tuple[TaggedTree | None, ...]:
        """Return each item's derivation of greatest expected bracket gain.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        goals : Sequence[T] | None
            The root categories a derivation may conclude at, or None for all of them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        tuple[TaggedTree | None, ...]
            One entry per item, holding the derivation and the expected number of
            constituents it gets right, or None where the item has no derivation. Each
            combining step is worth the constituent marginal of the span it builds,
            and a unary step is worth nothing.

        Raises
        ------
        TypeError
            If a channel is not of a floating point dtype, or if the lengths are not of
            an integer dtype.
        ValueError
            If the kernel does not use log-sum-exp reduction, the index is empty, a
            channel has a shape the index and the transitions do not call for,
            a goal is outside the inventory, or a checked length is outside the index.
        """
        kernel = self._kernel
        counts = kernel.expected_counts(channels, index, goals=goals, lengths=lengths)
        marginals = span_marginals(counts, index)
        unary = channels.unary
        gains = ScoreChannels(
            _supported(channels.leaf, counts.leaf, torch.zeros_like(channels.leaf)),
            _supported(
                channels.rule,
                counts.rule,
                marginals.unsqueeze(-1).unsqueeze(-1).expand_as(channels.rule),
            ),
            None
            if unary is None or counts.unary is None
            else _supported(unary, counts.unary, torch.zeros_like(unary)),
        )
        found = TaggedDecoder(kernel, k=1).k_best(
            gains, index, goals=goals, lengths=lengths
        )

        return tuple(trees[0] if trees else None for trees in found)
