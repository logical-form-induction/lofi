"""Shared neural tensor containers and choices."""

from __future__ import annotations

from collections.abc import Hashable
from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Protocol, override

import torch

from lofi._checks import CHECKS, NumericalFailureError, ensure_finite, ensure_name
from lofi._frozen import FrozenValue
from lofi.judge import ActionSet

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi._alignment import Alignment
    from lofi.judge import Action
    from lofi.nn._derivation import DerivationTree

__all__ = [
    "ChartActionScores",
    "ChartReadouts",
    "DerivationScoreChart",
    "DerivationSetScore",
    "Encoded",
    "InferenceSignal",
    "LogDistribution",
    "NormalFormDerivationScores",
    "TokenEncoding",
    "TruncationReport",
]


class DerivationSetScore(FrozenValue):
    """Store one derivation score and the partition it is normalized against."""

    __slots__ = ("_log_partition", "_normal_form", "_score")

    _noun = "derivation-set score"

    def __init__(
        self, normal_form: str, score: float | None, log_partition: float, /
    ) -> None:
        object.__setattr__(
            self, "_normal_form", ensure_name("a normal form", normal_form)
        )
        object.__setattr__(
            self,
            "_score",
            None if score is None else ensure_finite("a derivation score", score),
        )
        object.__setattr__(
            self,
            "_log_partition",
            ensure_finite("a derivation-set log partition", log_partition),
        )

    @property
    def normal_form(self) -> str:
        """Return the filter defining the derivation set."""
        return self._normal_form

    @property
    def score(self) -> float | None:
        """Return the gold derivation score, or none when the set excludes it."""
        return self._score

    @property
    def log_partition(self) -> float:
        """Return the log of the derivation set's total weight."""
        return self._log_partition

    def _fields(self) -> tuple[str, float | None, float]:
        """Return fields in constructor order."""
        return (self._normal_form, self._score, self._log_partition)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another score has the same fields."""
        if type(other) is not DerivationSetScore:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the score fields."""
        return hash((DerivationSetScore, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this score."""
        return (
            f"DerivationSetScore({self._normal_form!r}, {self._score!r}, "
            f"{self._log_partition!r})"
        )


class NormalFormDerivationScores(FrozenValue):
    """Store one derivation scored under a declared and unrestricted filter."""

    __slots__ = ("_constrained", "_unconstrained")

    _noun = "normal-form derivation scores"

    def __init__(
        self,
        constrained: DerivationSetScore,
        unconstrained: DerivationSetScore,
        /,
    ) -> None:
        object.__setattr__(self, "_constrained", constrained)
        object.__setattr__(self, "_unconstrained", unconstrained)

    @property
    def constrained(self) -> DerivationSetScore:
        """Return the score under the parser's declared normal form."""
        return self._constrained

    @property
    def unconstrained(self) -> DerivationSetScore:
        """Return the score under the unrestricted normal form."""
        return self._unconstrained

    def _fields(self) -> tuple[DerivationSetScore, DerivationSetScore]:
        """Return fields in constructor order."""
        return (self._constrained, self._unconstrained)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another pair holds the same two scores."""
        if type(other) is not NormalFormDerivationScores:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over both derivation-set scores."""
        return hash((NormalFormDerivationScores, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this pair."""
        return (
            f"NormalFormDerivationScores({self._constrained!r}, "
            f"{self._unconstrained!r})"
        )


class DerivationScoreChart[T: Hashable](Protocol):
    """Score gold derivations against two sets published by one chart."""

    @property
    def normal_form(self) -> str:
        """Return the parser's declared normal form."""

    def score(
        self, item: int, derivation: DerivationTree[T], /
    ) -> NormalFormDerivationScores:
        """Score one batch item's gold derivation under both filters."""


class InferenceSignal(StrEnum):
    """Name the scalar signal derived from a grounded label probability."""

    LIKELIHOOD = "likelihood"
    REWARD = "reward"


#: The name of the axis a batch's items run along.
_ITEM_AXIS: Final[str] = "item"

#: The name of the axis a token encoder's rows run along.
_ROW_AXIS: Final[str] = "row"

#: The axes of a token encoder's vectors, outermost first.
_TOKEN_AXES: Final[tuple[str, ...]] = (_ITEM_AXIS, _ROW_AXIS, "width")

#: How far a normalized axis may sit from summing to one, in log space.
_NORMALIZATION_ATOL: Final[float] = 1e-4

#: The one layout in which chart action scores cross a component boundary.
_CHART_ACTION_AXES: Final[tuple[str, ...]] = (
    "item",
    "span",
    "split",
    "action",
)


def _checked_axes(axes: tuple[str, ...], tensor: Tensor) -> tuple[str, ...]:
    """Return the axis names.

    Parameters
    ----------
    axes : tuple[str, ...]
        One name per axis, outermost first.
    tensor : Tensor
        The tensor the names describe.

    Returns
    -------
    tuple[str, ...]
        The names, unchanged.

    Raises
    ------
    ValueError
        If the names do not cover the tensor's axes exactly, if a name is empty, or if a
        name is used twice.
    """
    if len(axes) != tensor.ndim:
        msg = (
            f"a tensor of rank {tensor.ndim} needs {tensor.ndim} axis names, "
            f"but {len(axes)} were given: {axes}"
        )
        raise ValueError(msg)

    if not all(axes):
        msg = f"an axis name is empty: {axes}"
        raise ValueError(msg)

    if len(set(axes)) != len(axes):
        msg = f"an axis name is used twice, so it names neither axis: {axes}"
        raise ValueError(msg)

    return axes


def _checked_mask(mask: Tensor | None, tensor: Tensor) -> Tensor | None:
    """Return the validity mask.

    Parameters
    ----------
    mask : Tensor | None
        True where a position holds data, over the outermost axes of the tensor, or None
        where every position does.
    tensor : Tensor
        The tensor the mask covers.

    Returns
    -------
    Tensor | None
        The mask, unchanged.

    Raises
    ------
    TypeError
        If the mask is not of boolean dtype.
    ValueError
        If the mask covers no axis, covers more axes than the tensor has, or disagrees
        with the tensor about the extent of an axis it covers.
    """
    if mask is None:
        return None

    if mask.dtype is not torch.bool:
        msg = f"a mask holds booleans, not {mask.dtype}"
        raise TypeError(msg)

    if not 1 <= mask.ndim <= tensor.ndim:
        msg = (
            f"a mask covers the outermost axes of a tensor of rank "
            f"{tensor.ndim}, so its rank is between 1 and {tensor.ndim}, "
            f"not {mask.ndim}"
        )
        raise ValueError(msg)

    covered = tuple(tensor.shape[: mask.ndim])

    if tuple(mask.shape) != covered:
        msg = (
            f"a mask of shape {tuple(mask.shape)} does not cover the outermost "
            f"axes of a tensor of shape {tuple(tensor.shape)}"
        )
        raise ValueError(msg)

    return mask


def _aligned(mask: Tensor, ndim: int) -> Tensor:
    """Return a mask reshaped to broadcast against a tensor of a given rank.

    Parameters
    ----------
    mask : Tensor
        A mask over the outermost axes.
    ndim : int
        The rank to broadcast against.

    Returns
    -------
    Tensor
        A boolean view of the mask with a trailing axis of extent one for every axis the
        mask does not cover.
    """
    return mask.reshape(tuple(mask.shape) + (1,) * (ndim - mask.ndim))


class _AxisTensor:
    """Stores a tensor with named axes and an optional validity mask.

    Parameters
    ----------
    tensor : Tensor
        The values.
    axes : tuple[str, ...]
        One name per axis, outermost first.
    mask : Tensor | None
        True where a position holds data, over the outermost axes, or None where every
        position does.

    Raises
    ------
    TypeError
        If the mask is not of boolean dtype.
    ValueError
        If the axis names do not cover the tensor's axes exactly, if a name is empty or
        repeated, or if the mask does not cover the outermost axes.
    """

    __slots__ = ("_axes", "_mask", "_tensor")

    def __init__(
        self,
        tensor: Tensor,
        axes: tuple[str, ...],
        mask: Tensor | None = None,
    ) -> None:
        self._tensor = tensor
        self._axes = _checked_axes(axes, tensor)
        self._mask = _checked_mask(mask, tensor)

    @property
    def tensor(self) -> Tensor:
        """Return the values."""
        return self._tensor

    @property
    def axes(self) -> tuple[str, ...]:
        """Return what each axis means, outermost first."""
        return self._axes

    @property
    def mask(self) -> Tensor | None:
        """Return the validity mask, or None if every position is valid."""
        return self._mask

    @property
    def mask_axes(self) -> tuple[str, ...]:
        """Return the axes the mask covers, which are the outermost ones."""
        if self._mask is None:
            return ()

        return self._axes[: self._mask.ndim]

    def axis(self, name: str) -> int:
        """Return the position of a named axis.

        Parameters
        ----------
        name : str
            The axis to locate.

        Returns
        -------
        int
            Its position, counting from the outermost.

        Raises
        ------
        ValueError
            If the value declares no axis under that name.
        """
        try:
            return self._axes.index(name)

        except ValueError as error:
            msg = f"there is no axis named {name!r}; the axes are {self._axes}"
            raise ValueError(msg) from error

    def size(self, name: str) -> int:
        """Return the extent of a named axis.

        Parameters
        ----------
        name : str
            The axis to measure.

        Returns
        -------
        int
            How many positions it holds.

        Raises
        ------
        ValueError
            If the value declares no axis under that name.
        """
        return int(self._tensor.shape[self.axis(name)])

    @override
    def __repr__(self) -> str:
        """Return a concise representation."""
        return (
            f"{type(self).__name__}(shape={tuple(self._tensor.shape)}, "
            f"axes={self._axes!r}, masked={self._mask is not None})"
        )


class Encoded(_AxisTensor):
    """Stores a real-valued tensor with named axes and an optional mask.

    Parameters
    ----------
    tensor : Tensor
        The values.
    axes : tuple[str, ...]
        One name per axis, outermost first.
    mask : Tensor | None
        True where a position holds data, over the outermost axes, or None where every
        position does.

    Raises
    ------
    TypeError
        If the mask is not of boolean dtype.
    ValueError
        If the axis names do not cover the tensor's axes exactly, if a name is empty or
        repeated, or if the mask does not cover the outermost axes.
    """

    __slots__ = ()


class LogDistribution(_AxisTensor):
    """Stores log-space values with an optional normalized axis.

    Parameters
    ----------
    tensor : Tensor
        The log-space values, at floating precision.
    axes : tuple[str, ...]
        One name per axis, outermost first.
    mask : Tensor | None
        True where a position holds data, over the outermost axes, or None where every
        position does.
    normalized_over : str | None
        The axis whose mass sums to one, or None for an unnormalized score.

    Raises
    ------
    TypeError
        If the tensor is not of floating dtype, or the mask is not boolean.
    ValueError
        If axis names, mask coverage, or the normalized axis are invalid.
    """

    __slots__ = ("_normalized_over",)

    def __init__(
        self,
        tensor: Tensor,
        axes: tuple[str, ...],
        mask: Tensor | None = None,
        normalized_over: str | None = None,
    ) -> None:
        if not tensor.is_floating_point():
            msg = f"a log-space value is floating point, not {tensor.dtype}"
            raise TypeError(msg)

        super().__init__(tensor, axes, mask)

        # validate the normalized axis before storing it
        self._normalized_over = (
            None if normalized_over is None else self._axes[self.axis(normalized_over)]
        )

        if CHECKS:
            self._reject_padded_zeros()

            if self._normalized_over is not None:
                self._reject_unnormalized(self._normalized_over)

    @property
    def normalized_over(self) -> str | None:
        """Return the axis whose mass sums to one, or None for an unnormalized score."""
        return self._normalized_over

    def _reject_padded_zeros(self) -> None:
        """Raise if a masked position holds an exact zero.

        Raises
        ------
        ValueError
            If any position the mask excludes holds ``0.0``, which reads as probability
            one.
        """
        if self._mask is None:
            return

        padded = ~_aligned(self._mask, self._tensor.ndim)

        if bool((self._tensor.eq(0.0) & padded).any()):
            msg = (
                "a masked position holds an exact zero, which reads as "
                "probability one. an absent log weight is negative infinity"
            )
            raise ValueError(msg)

    def _reject_unnormalized(self, name: str) -> None:
        """Raise if the mass along an axis does not sum to one.

        Parameters
        ----------
        name : str
            The normalized axis.

        Raises
        ------
        NumericalFailureError
            If a live position's log mass along the axis is not a number or is
            positive infinity.
        ValueError
            If a live position's mass along the axis is otherwise not one.
        """
        dim = self.axis(name)
        values = self._tensor

        if self._mask is not None:
            values = values.masked_fill(~_aligned(self._mask, values.ndim), -torch.inf)

        total = values.logsumexp(dim=dim)
        off = ~(total.abs() <= _NORMALIZATION_ATOL)
        nonfinite = total.isnan() | total.isposinf()

        if self._mask is not None:
            live = _aligned(
                self._mask.any(dim=dim) if self._mask.ndim > dim else self._mask,
                off.ndim,
            )
            off = off & live
            nonfinite = nonfinite & live

        if bool(nonfinite.any()):
            quantity = f"the log mass over the axis named {name!r}"
            raise NumericalFailureError(quantity, float(total[nonfinite][0].detach()))

        if bool(off.any()):
            msg = (
                f"the mass over the axis named {name!r} does not sum to one at "
                f"{int(off.sum())} of {off.numel()} positions"
            )
            raise ValueError(msg)


class TruncationReport:
    """Names the words an encoder dropped from one item.

    Parameters
    ----------
    positions : tuple[int, ...]
        Where each dropped word stood in the item, in increasing order.
    words : tuple[str, ...]
        The dropped words, in the same order.

    Raises
    ------
    ValueError
        If there is not one word per position, if a position is negative, or if the
        positions do not form one contiguous range.
    """

    __slots__ = ("_positions", "_words")

    def __init__(self, positions: tuple[int, ...], words: tuple[str, ...]) -> None:
        if len(positions) != len(words):
            msg = (
                f"a truncation report names one word per position, but "
                f"{len(words)} words were given for {len(positions)} positions"
            )
            raise ValueError(msg)

        previous: int | None = None

        for position in positions:
            if position < 0 or (previous is not None and position != previous + 1):
                msg = (
                    "the dropped positions increase by one through one contiguous "
                    f"range, so they cannot be {positions}"
                )
                raise ValueError(msg)
            previous = position

        self._positions = tuple(positions)
        self._words = tuple(words)

    @property
    def positions(self) -> tuple[int, ...]:
        """Return where each dropped word stood in the item."""
        return self._positions

    @property
    def words(self) -> tuple[str, ...]:
        """Return the dropped words, in item order."""
        return self._words

    def __len__(self) -> int:
        """Return how many words were dropped."""
        return len(self._words)

    def __bool__(self) -> bool:
        """Return whether any word was dropped."""
        return bool(self._words)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report names the same words at the same positions."""
        if type(other) is not TruncationReport:
            return NotImplemented

        return self._positions == other._positions and self._words == other._words

    @override
    def __hash__(self) -> int:
        """Return a hash over the positions and the words."""
        return hash((TruncationReport, self._positions, self._words))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this report."""
        return f"TruncationReport({self._positions!r}, {self._words!r})"


class ChartActionScores(FrozenValue):
    """Pair a chart scorer's ordered actions with its per-split log scores.

    Parameters
    ----------
    action_set : ActionSet
        The exact ordered action inventory that assigns meaning to the final axis.
    scores : LogDistribution
        Unnormalized log scores over ``("item", "span", "split", "action")``,
        masked over exactly the first three axes.

    Raises
    ------
    TypeError
        If the inventory is not an action set.
    ValueError
        If the score axes or mask are not exact, or if the action extent differs
        from the inventory size.
    """

    __slots__ = ("_action_set", "_by_action", "_scores")

    _noun = "chart action scores"

    def __init__(self, action_set: ActionSet, scores: LogDistribution) -> None:
        if not isinstance(action_set, ActionSet):
            msg = f"a chart action inventory is an ActionSet, not {action_set!r}"
            raise TypeError(msg)

        if scores.axes != _CHART_ACTION_AXES:
            msg = (
                f"chart action scores run over {_CHART_ACTION_AXES}, not over "
                f"{scores.axes}"
            )
            raise ValueError(msg)

        if scores.normalized_over is not None:
            msg = (
                "chart action scores are unnormalized parser scores, not a "
                f"distribution normalized over {scores.normalized_over!r}"
            )
            raise ValueError(msg)

        actions = action_set.actions
        extent = scores.size("action")

        if extent != len(actions):
            msg = (
                f"the action axis has extent {extent}, but the action inventory "
                f"holds {len(actions)} actions"
            )
            raise ValueError(msg)

        if scores.mask is None or scores.mask_axes != _CHART_ACTION_AXES[:-1]:
            msg = (
                "chart action scores require a valid-split mask over exactly "
                f"{_CHART_ACTION_AXES[:-1]}, not {scores.mask_axes}"
            )
            raise ValueError(msg)

        projected = {
            action: LogDistribution(
                scores.tensor[..., column],
                _CHART_ACTION_AXES[:-1],
                scores.mask,
            )
            for column, action in enumerate(actions)
        }
        object.__setattr__(self, "_action_set", action_set)
        object.__setattr__(self, "_scores", scores)
        object.__setattr__(self, "_by_action", MappingProxyType(projected))

    @property
    def action_set(self) -> ActionSet:
        """Return the exact inventory assigning actions to score columns."""
        return self._action_set

    @property
    def scores(self) -> LogDistribution:
        """Return the rank-four chart action scores."""
        return self._scores

    @property
    def by_action(self) -> Mapping[Action, LogDistribution]:
        """Return one gradient-preserving projection under each inventory action."""
        return self._by_action

    @property
    def identity(self) -> LogDistribution:
        """Return the named instrumentation projection for the identity action."""
        return self._by_action[self._action_set.identity]

    @override
    def __repr__(self) -> str:
        """Return the inventory size and score shape."""
        return (
            f"ChartActionScores(actions={len(self._action_set.actions)}, "
            f"shape={tuple(self._scores.tensor.shape)})"
        )


def _checked_words(
    words: Sequence[Sequence[str]] | None,
    alignments: tuple[Alignment, ...],
) -> tuple[tuple[str, ...], ...] | None:
    """Return the input words of each item.

    Parameters
    ----------
    words : Sequence[Sequence[str]] | None
        The words of each item, in item order and then in word order, or None where
        the encoding does not record them.
    alignments : tuple[Alignment, ...]
        One alignment per item, whose entries the words are checked against.

    Returns
    -------
    tuple[tuple[str, ...], ...] | None
        The words as tuples, or None where none were given.

    Raises
    ------
    ValueError
        If there is not one word sequence per alignment, or if an item's words and
        its alignment disagree about how many words the item holds.
    """
    if words is None:
        return None

    held = tuple(tuple(item) for item in words)

    if len(held) != len(alignments):
        msg = (
            f"a token encoding records the words of each of its {len(alignments)} "
            f"items, but received the words of {len(held)}"
        )
        raise ValueError(msg)

    for item, (item_words, alignment) in enumerate(zip(held, alignments, strict=True)):
        if len(item_words) != alignment.word_count:
            msg = (
                f"item {item} holds {len(item_words)} words and its alignment "
                f"holds {alignment.word_count} entries"
            )
            raise ValueError(msg)

    return held


def _check_alignments(
    alignments: tuple[Alignment, ...],
    mask: Tensor,
    row_count: int,
    *,
    words_recorded: bool,
) -> None:
    """Check that every alignment names live rows and omits only a suffix.

    Parameters
    ----------
    alignments : tuple[Alignment, ...]
        One alignment per item.
    mask : Tensor
        The live rows of each item.
    row_count : int
        The extent of the row axis.
    words_recorded : bool
        Whether the encoding records the original words.

    Raises
    ------
    ValueError
        If a retained span exceeds the row axis or includes a masked row, if an
        alignment has an interior missing entry, or if an incomplete alignment has no
        recorded words.
    """
    for item, alignment in enumerate(alignments):
        for word, span in enumerate(alignment.rows):
            if span is None:
                continue

            start, end = span

            if end > row_count:
                msg = (
                    f"word {word} of item {item} is aligned to rows [{start}, {end}), "
                    f"beyond the row extent {row_count}"
                )
                raise ValueError(msg)

            if not bool(mask[item, start:end].all()):
                msg = (
                    f"word {word} of item {item} is aligned to rows [{start}, {end}), "
                    "which includes a masked row"
                )
                raise ValueError(msg)

        dropped = alignment.dropped

        if not dropped:
            continue

        suffix = tuple(range(dropped[0], alignment.word_count))

        if dropped != suffix:
            msg = (
                f"item {item} omits the word positions {dropped}, but an incomplete "
                f"alignment may omit only one contiguous suffix such as {suffix}"
            )
            raise ValueError(msg)

        if not words_recorded:
            msg = (
                f"item {item} has an incomplete alignment at {dropped}, so the token "
                "encoding must record its original words"
            )
            raise ValueError(msg)


class TokenEncoding:
    """Stores encoder rows, their mask, and word alignment.

    Parameters
    ----------
    vectors : Encoded
        The encoder's rows, over the axes ``("item", "row", "width")``, with a mask over
        the item and row axes.
    alignments : tuple[Alignment, ...]
        One alignment per item, in the order of the item axis.
    words : Sequence[Sequence[str]] | None
        The words of each item, in item order and then in word order, or None for a
        complete alignment that does not retain them.

    Raises
    ------
    ValueError
        If the vectors are not over those three axes, if they have no mask or one that
        leaves the row axis uncovered, if there is not exactly one alignment per item,
        if an alignment names an absent row, if an incomplete alignment does not omit
        one suffix or has no recorded words, or if the words of an item do not number
        its alignment's entries.
    """

    __slots__ = ("_alignments", "_lengths", "_vectors", "_words")

    def __init__(
        self,
        vectors: Encoded,
        alignments: tuple[Alignment, ...],
        *,
        words: Sequence[Sequence[str]] | None = None,
    ) -> None:
        if vectors.axes != _TOKEN_AXES:
            msg = f"a token encoding is over the axes {_TOKEN_AXES}, not {vectors.axes}"
            raise ValueError(msg)

        row_axis = vectors.axis(_ROW_AXIS)
        mask = vectors.mask

        if mask is None or mask.ndim != row_axis + 1:
            msg = (
                "a token encoding requires a mask over its item and row axes, "
                "since the rows of a batch of sentences are ragged"
            )
            raise ValueError(msg)

        if len(alignments) != vectors.size(_ITEM_AXIS):
            msg = (
                f"a token encoding requires one alignment per item, but received "
                f"{len(alignments)} for {vectors.size(_ITEM_AXIS)} items"
            )
            raise ValueError(msg)

        self._vectors = vectors
        self._alignments = tuple(alignments)
        self._lengths = mask.sum(dim=row_axis)
        _check_alignments(
            self._alignments,
            mask,
            vectors.size(_ROW_AXIS),
            words_recorded=words is not None,
        )
        self._words = _checked_words(words, self._alignments)

    @property
    def vectors(self) -> Encoded:
        """Return the encoder's rows, with their axes and their padding mask."""
        return self._vectors

    @property
    def lengths(self) -> Tensor:
        """Return the valid row count for each item."""
        return self._lengths

    @property
    def alignments(self) -> tuple[Alignment, ...]:
        """Return the word-to-row correspondence for each item, in item order."""
        return self._alignments

    @property
    def words(self) -> tuple[tuple[str, ...], ...] | None:
        """Return the input words of each item, or None if they were not recorded."""
        return self._words

    @property
    def truncated(self) -> tuple[tuple[int, ...], ...]:
        """Return the positions of the words omitted from each item."""
        return tuple(alignment.dropped for alignment in self._alignments)

    @property
    def truncation(self) -> tuple[TruncationReport, ...] | None:
        """Return the words omitted from each item, or None if no words were recorded.

        Returns
        -------
        tuple[TruncationReport, ...] | None
            One report per item, empty for an item that lost no word, naming each
            dropped word beside the position it stood at.
        """
        if self._words is None:
            return None

        return tuple(
            TruncationReport(
                alignment.dropped,
                tuple(item_words[position] for position in alignment.dropped),
            )
            for alignment, item_words in zip(self._alignments, self._words, strict=True)
        )

    @override
    def __repr__(self) -> str:
        """Return the batch shape and truncation count."""
        lost = sum(1 for dropped in self.truncated if dropped)

        return (
            f"TokenEncoding(shape={tuple(self._vectors.tensor.shape)}, "
            f"truncated={lost})"
        )


class ChartReadouts:
    """Stores optional outputs published by a chart scorer.

    Parameters
    ----------
    inside : LogDistribution | None
        Log inside scores per span.
    outside : LogDistribution | None
        Log outside scores per span.
    alpha : LogDistribution | None
        The log posterior over the splits of a span.
    mu : Encoded | None
        The probability that a span is a constituent.
    log_z : LogDistribution | None
        The log partition, one per item.
    actions : ChartActionScores | None
        The complete ordered chart-action score inventory, where a formalism has one.
    cells : Encoded | None
        The inside-vector representation the chart holds for each span.
    outside_cells : Encoded | None
        The vector representation of each span's surrounding context.
    spans : Encoded | None
        The span representation formed by concatenating its inside and outside
        vectors.
    entropy : Encoded | None
        Derivation entropy per item, in nats.
    mbr : Encoded | None
        The minimum-Bayes-risk tree as a binary constituent indicator per span.
    derivations : DerivationScoreChart[Hashable] | None
        Gold-derivation scoring against the tagged chart's declared and
        unrestricted derivation sets.

    Raises
    ------
    ValueError
        If two readouts name the same axis and disagree about its extent.
    """

    __slots__ = (
        "_actions",
        "_alpha",
        "_cells",
        "_derivations",
        "_entropy",
        "_inside",
        "_log_z",
        "_mbr",
        "_mu",
        "_outside",
        "_outside_cells",
        "_spans",
    )

    def __init__(
        self,
        *,
        inside: LogDistribution | None = None,
        outside: LogDistribution | None = None,
        alpha: LogDistribution | None = None,
        mu: Encoded | None = None,
        log_z: LogDistribution | None = None,
        actions: ChartActionScores | None = None,
        cells: Encoded | None = None,
        outside_cells: Encoded | None = None,
        spans: Encoded | None = None,
        entropy: Encoded | None = None,
        mbr: Encoded | None = None,
        derivations: DerivationScoreChart[Hashable] | None = None,
    ) -> None:
        self._inside = inside
        self._outside = outside
        self._alpha = alpha
        self._mu = mu
        self._log_z = log_z
        self._actions = actions
        self._cells = cells
        self._outside_cells = outside_cells
        self._spans = spans
        self._entropy = entropy
        self._mbr = mbr
        self._derivations = derivations

        self._reject_disagreeing_axes()

    @property
    def inside(self) -> LogDistribution | None:
        """Return log inside scores per span if published."""
        return self._inside

    @property
    def outside(self) -> LogDistribution | None:
        """Return log outside scores per span if published."""
        return self._outside

    @property
    def alpha(self) -> LogDistribution | None:
        """Return per-span log split posteriors, if available."""
        return self._alpha

    @property
    def mu(self) -> Encoded | None:
        """Return per-span constituent marginals, if available."""
        return self._mu

    @property
    def log_z(self) -> LogDistribution | None:
        """Return the log partition per item if published."""
        return self._log_z

    @property
    def actions(self) -> ChartActionScores | None:
        """Return the complete chart-action score inventory if published."""
        return self._actions

    @property
    def identity(self) -> LogDistribution | None:
        """Return the identity action's instrumentation projection if published."""
        return None if self._actions is None else self._actions.identity

    @property
    def cells(self) -> Encoded | None:
        """Return the inside-vector representation of each span if published."""
        return self._cells

    @property
    def outside_cells(self) -> Encoded | None:
        """Return the surrounding-context vector of each span if published."""
        return self._outside_cells

    @property
    def spans(self) -> Encoded | None:
        """Return the concatenated inside and outside span representation."""
        return self._spans

    @property
    def entropy(self) -> Encoded | None:
        """Return derivation entropy per item, in nats."""
        return self._entropy

    @property
    def mbr(self) -> Encoded | None:
        """Return the MBR tree as a binary constituent indicator per span."""
        return self._mbr

    @property
    def derivations(self) -> DerivationScoreChart[Hashable] | None:
        """Return published gold-derivation scoring, if available."""
        return self._derivations

    def _published(self) -> tuple[tuple[str, _AxisTensor], ...]:
        """Return the readouts that are present, paired with their names.

        Returns
        -------
        tuple[tuple[str, _AxisTensor], ...]
            One pair per published readout, in declaration order.
        """
        published = (
            ("inside", self._inside),
            ("outside", self._outside),
            ("alpha", self._alpha),
            ("mu", self._mu),
            ("log_z", self._log_z),
            (
                "actions",
                None if self._actions is None else self._actions.scores,
            ),
            ("identity", self.identity),
            ("cells", self._cells),
            ("outside_cells", self._outside_cells),
            ("spans", self._spans),
            ("entropy", self._entropy),
            ("mbr", self._mbr),
        )

        return tuple(
            (name, readout) for name, readout in published if readout is not None
        )

    def _reject_disagreeing_axes(self) -> None:
        """Raise where two published readouts size a shared axis differently.

        Raises
        ------
        ValueError
            If two readouts name the same axis with different extents.
        """
        seen: dict[str, tuple[str, int]] = {}

        for name, readout in self._published():
            for axis in readout.axes:
                extent = readout.size(axis)
                first = seen.setdefault(axis, (name, extent))

                if first[1] != extent:
                    msg = (
                        f"{first[0]} and {name} disagree about the axis "
                        f"named {axis!r}: {first[1]} against {extent}"
                    )
                    raise ValueError(msg)

    @override
    def __repr__(self) -> str:
        """Return which readouts are present."""
        names = [name for name, _ in self._published()]

        if self._derivations is not None:
            names.append("derivations")

        return f"ChartReadouts({', '.join(names)})"
