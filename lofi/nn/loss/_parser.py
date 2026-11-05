"""Parser outputs consumed by objectives."""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import CHECKS
from lofi.nn._values import Encoded
from lofi.nn.loss._errors import InvalidCutpointError, UncomputedCellError

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.nn._values import LogDistribution

__all__ = [
    "SPAN_AXES",
    "SPLIT_AXES",
    "SpanMarginals",
    "SplitPosterior",
]

SPAN_AXES: Final[tuple[str, ...]] = ("item", "span")
"""Per-span tensor axes."""

SPLIT_AXES: Final[tuple[str, ...]] = ("item", "span", "split")
"""Per-cutpoint tensor axes."""


def _checked_axes(
    axes: tuple[str, ...],
    expected: tuple[str, ...],
    *,
    field: str,
) -> None:
    """Raise unless a value has the required axes.

    Parameters
    ----------
    axes : tuple[str, ...]
        The axes the value declares.
    expected : tuple[str, ...]
        The axes the contract names.
    field : str
        The field the value arrived as, for the message.

    Raises
    ------
    ValueError
        If the two differ.
    """
    if axes != expected:
        msg = f"{field} is over the axes {axes}, and this contract is over {expected}"
        raise ValueError(msg)


def _checked_extent(
    tensor: Tensor,
    spans: SpanIndex,
    *,
    field: str,
    splits: bool,
) -> None:
    """Raise unless a tensor's span and cutpoint axes match the span index.

    Parameters
    ----------
    tensor : Tensor
        The values.
    spans : SpanIndex
        The addressing they are over.
    field : str
        The field the value arrived as, for the message.
    splits : bool
        Whether the tensor has a cutpoint axis to check as well.

    Raises
    ------
    ValueError
        If the span axis is not as long as the index has spans, or the cutpoint axis is
        not as wide as its widest span's split count.
    """
    if tensor.shape[1] != spans.num_spans:
        msg = (
            f"{field} holds {tensor.shape[1]} spans, and the index addresses "
            f"{spans.num_spans}"
        )
        raise ValueError(msg)

    if splits and tensor.shape[2] != spans.max_splits:
        msg = (
            f"{field} holds {tensor.shape[2]} cutpoints per span, and the "
            f"index's widest span has {spans.max_splits}"
        )
        raise ValueError(msg)


def _checked_span(spans: SpanIndex, i: int, j: int) -> None:
    """Raise unless a span lies inside the index.

    Parameters
    ----------
    spans : SpanIndex
        The addressing being read.
    i : int
        The start of the span, inclusive.
    j : int
        The end of the span, exclusive.

    Raises
    ------
    ValueError
        If the span is empty or reaches outside the input.
    """
    if not 0 <= i < j <= spans.n:
        msg = f"the span [{i}, {j}) is not a span of an input of length {spans.n}"
        raise ValueError(msg)


class SplitPosterior:
    """Stores per-span split probabilities and valid cutpoints.

    Parameters
    ----------
    spans : SpanIndex
        The addressing the span and cutpoint axes are over.
    log_probabilities : LogDistribution
        The log probability of each cutpoint of each span, over the axes ``SPLIT_AXES``,
        normalized over the cutpoint axis, and masked to the cutpoints each span
        actually has.

    Attributes
    ----------
    spans : SpanIndex
        The addressing the span and cutpoint axes are over.
    log_probabilities : LogDistribution
        The log probability of each cutpoint of each span.
    valid : Tensor
        True at a cutpoint a span has, of rank 3 and boolean dtype.
    items : int
        How many items the posterior covers.

    Raises
    ------
    ValueError
        If axes, normalization, mask coverage, or span and cutpoint extents are invalid.
    """

    __slots__ = ("_log_probabilities", "_spans", "_valid")

    def __init__(
        self,
        spans: SpanIndex,
        log_probabilities: LogDistribution,
    ) -> None:
        _checked_axes(log_probabilities.axes, SPLIT_AXES, field="a split posterior")

        if log_probabilities.normalized_over != SPLIT_AXES[-1]:
            msg = (
                f"a split posterior's mass sums over {SPLIT_AXES[-1]!r}, and "
                f"this one names {log_probabilities.normalized_over!r}"
            )
            raise ValueError(msg)

        valid = log_probabilities.mask

        if valid is None:
            msg = (
                "a split posterior requires a validity mask over its cutpoints, "
                "since a padded cutpoint reads as probability one in log space"
            )
            raise ValueError(msg)

        if valid.ndim != len(SPLIT_AXES):
            msg = (
                f"a split posterior's mask covers its cutpoint axis, so its "
                f"rank is {len(SPLIT_AXES)}, not {valid.ndim}"
            )
            raise ValueError(msg)

        _checked_extent(
            log_probabilities.tensor,
            spans,
            field="a split posterior",
            splits=True,
        )

        self._spans = spans
        self._log_probabilities = log_probabilities
        self._valid = valid

    @property
    def spans(self) -> SpanIndex:
        """Return the addressing the span and cutpoint axes are over."""
        return self._spans

    @property
    def log_probabilities(self) -> LogDistribution:
        """Return the log probability of each cutpoint of each span."""
        return self._log_probabilities

    @property
    def valid(self) -> Tensor:
        """Return a rank-3 boolean mask of valid cutpoints."""
        return self._valid

    @property
    def items(self) -> int:
        """Return the batch size."""
        return int(self._log_probabilities.tensor.shape[0])

    def at(self, item: int, i: int, j: int, k: int) -> Tensor:
        """Return the log probability that one span divides at one cutpoint.

        Parameters
        ----------
        item : int
            Which item of the batch to read.
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.
        k : int
            The cutpoint, which divides the span into ``[i, k)`` and ``[k, j)``.

        Returns
        -------
        Tensor
            A scalar in the posterior's floating dtype, attached to its autograd graph.

        Raises
        ------
        ValueError
            If the span is empty or reaches outside the input.
        IndexError
            If there is no such item.
        InvalidCutpointError
            If the cutpoint does not divide the span, or the mask marks it as one the
            span does not have.
        """
        _checked_span(self._spans, i, j)

        if k not in self._spans.splits(i, j):
            raise InvalidCutpointError(item=item, span=(i, j), cut=k)

        span, column = self._spans.split_id(i, j, k)

        if not bool(self.valid[item, span, column]):
            raise InvalidCutpointError(item=item, span=(i, j), cut=k)

        return self._log_probabilities.tensor[item, span, column]

    def mass(self) -> Tensor:
        """Return how much probability each span's valid cutpoints hold.

        Returns
        -------
        Tensor
            A tensor of rank 2 over item and span, of the posterior's floating dtype.
        """
        scores = self._log_probabilities.tensor
        live = torch.where(self.valid, scores, scores.new_full((), -torch.inf))

        return live.logsumexp(dim=len(SPLIT_AXES) - 1).exp()

    @override
    def __repr__(self) -> str:
        """Return the axes and valid-position count."""
        return (
            f"SplitPosterior(items={self.items}, spans={self._spans.n}, "
            f"valid={int(self.valid.sum())})"
        )


class SpanMarginals:
    """Stores constituent and rule marginals.

    Parameters
    ----------
    spans : SpanIndex
        The addressing the span and cutpoint axes are over.
    constituent : Encoded
        The probability that each span is a constituent, over ``SPAN_AXES``, masked to
        the spans a binary derivation may build for each item.
    rule : Encoded
        The probability that each span is built at each cutpoint, over ``SPLIT_AXES``,
        masked to the cutpoints each span has.
    lengths : Tensor
        How many tokens each item holds, of rank 1 and integer dtype.

    Attributes
    ----------
    spans : SpanIndex
        The addressing the span and cutpoint axes are over.
    constituent : Encoded
        The probability that each span is a constituent.
    rule : Encoded
        The probability that each span is built at each cutpoint.
    built : Tensor
        Which spans a derivation may build, which is the constituent marginal's mask.
    lengths : Tensor
        How many tokens each item holds.
    items : int
        How many items the marginals cover.

    Raises
    ------
    TypeError
        If the lengths are not of integer dtype.
    ValueError
        If axes, masks, span or cutpoint extents, or item lengths disagree with the
        index.
    """

    __slots__ = ("_built", "_constituent", "_cutpoints", "_lengths", "_rule", "_spans")

    def __init__(
        self,
        spans: SpanIndex,
        constituent: Encoded,
        rule: Encoded,
        lengths: Tensor,
    ) -> None:
        _checked_axes(constituent.axes, SPAN_AXES, field="a constituent marginal")
        _checked_axes(rule.axes, SPLIT_AXES, field="a rule marginal")
        _checked_extent(
            constituent.tensor,
            spans,
            field="a constituent marginal",
            splits=False,
        )
        _checked_extent(rule.tensor, spans, field="a rule marginal", splits=True)

        built = constituent.mask
        cutpoints = rule.mask

        if built is None or cutpoints is None:
            msg = "a span marginal names which spans and cutpoints are real"
            raise ValueError(msg)

        self._spans = spans
        self._constituent = constituent
        self._rule = rule
        self._cutpoints = cutpoints
        self._built = built
        self._lengths = _checked_lengths(lengths, spans, constituent.tensor.shape[0])

        if CHECKS:
            _reject_leaf_constituents(spans, built)

    @property
    def spans(self) -> SpanIndex:
        """Return the addressing the span and cutpoint axes are over."""
        return self._spans

    @property
    def constituent(self) -> Encoded:
        """Return the probability that each span is a constituent."""
        return self._constituent

    @property
    def rule(self) -> Encoded:
        """Return the probability that each span is built at each cutpoint."""
        return self._rule

    @property
    def built(self) -> Tensor:
        """Return which spans a derivation may build, over the item and span axes."""
        return self._built

    @property
    def lengths(self) -> Tensor:
        """Return the token count for each item."""
        return self._lengths

    @property
    def items(self) -> int:
        """Return the batch size."""
        return int(self._constituent.tensor.shape[0])

    def constituent_factor(self) -> Encoded:
        """Return the complete constituent probability at every live span.

        A binary derivation contains every lexical span and its root with probability
        one. Proper internal spans take their parser marginal. Padded spans remain
        outside the returned mask. This is the one definition used by both target
        readouts and negative-bracket supervision.
        """
        probabilities = self._constituent.tensor
        device = probabilities.device
        lengths = self._lengths.to(device)
        fixed = torch.zeros(
            self.items,
            self._spans.num_spans,
            dtype=torch.bool,
            device=device,
        )
        leaves = tuple(self._spans.leaves())

        if leaves:
            leaf_ids = [self._spans.span_id(i, j) for i, j in leaves]
            ends = torch.tensor([j for _, j in leaves], device=device)
            fixed[:, leaf_ids] = ends <= lengths.unsqueeze(1)

        roots = torch.tensor(
            [self._spans.span_id(0, width) for width in range(1, self._spans.n + 1)],
            device=device,
        )
        fixed[torch.arange(self.items, device=device), roots[lengths - 1]] = True
        defined = fixed | self._built
        values = torch.where(
            fixed,
            probabilities.new_ones(()),
            torch.where(self._built, probabilities, probabilities.new_zeros(())),
        )

        return Encoded(values, SPAN_AXES, defined)

    def gather_constituent(self, cells: Sequence[tuple[int, int, int]], /) -> Tensor:
        """Return complete constituent probabilities for addressed chart cells.

        Raises
        ------
        ValueError
            If a span is empty or reaches outside the input, or a defined probability
            lies outside the unit interval.
        IndexError
            If a cell names an item these marginals do not hold.
        UncomputedCellError
            If a cell is padding rather than a span of that item.
        """
        factor = self.constituent_factor()
        mask = factor.mask

        if mask is None:
            msg = "a complete constituent factor names its live cells"
            raise ValueError(msg)

        addressed: list[int] = []
        items: list[int] = []

        for item, i, j in cells:
            _checked_span(self._spans, i, j)

            if not 0 <= item < self.items:
                msg = f"there is no item {item} in marginals over {self.items} items"
                raise IndexError(msg)

            items.append(item)
            addressed.append(self._spans.span_id(i, j))

        item_index = torch.tensor(items, dtype=torch.long, device=factor.tensor.device)
        span_index = torch.tensor(
            addressed, dtype=torch.long, device=factor.tensor.device
        )
        computed = mask[item_index, span_index]

        if not bool(computed.all()):
            first = int(computed.logical_not().nonzero()[0])
            item, i, j = cells[first]
            raise UncomputedCellError(item=item, span=(i, j))

        gathered = factor.tensor[item_index, span_index]
        proper = gathered.isfinite() & (gathered >= 0.0) & (gathered <= 1.0)

        if not bool(proper.all()):
            first = int(proper.logical_not().nonzero()[0])
            item, i, j = cells[first]
            msg = (
                f"the constituent probability at span [{i}, {j}) of item {item} "
                f"is outside the unit interval: {float(gathered[first])!r}"
            )
            raise ValueError(msg)

        return gathered

    def log_nonconstituent(self, cells: Sequence[tuple[int, int, int]], /) -> Tensor:
        """Return exact log nonconstituent mass with finite endpoint gradients.

        A constituent probability of one has zero complementary mass, whose log is
        negative infinity. The safe branch preserves that exact value without
        evaluating ``log1p`` at its singular endpoint, so its gradient is finite.

        Raises
        ------
        ValueError
            If a span is empty or reaches outside the input, or a defined probability
            lies outside the unit interval.
        IndexError
            If a cell names an item these marginals do not hold.
        UncomputedCellError
            If a cell is padding rather than a span of that item.
        """
        probability = self.gather_constituent(cells)
        below_one = probability < 1.0
        safe = torch.where(below_one, probability, probability.new_zeros(()))
        complement = torch.log1p(-safe)

        return torch.where(
            below_one,
            complement,
            probability.new_full((), -torch.inf),
        )

    def at(self, item: int, i: int, j: int) -> Tensor:
        """Return the probability that one span is a constituent.

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
        Tensor
            A scalar in the marginal's floating dtype, attached to its autograd graph.

        Raises
        ------
        ValueError
            If the span is empty or reaches outside the input.
        IndexError
            If there is no such item.
        """
        return self.gather_constituent(((item, i, j),))[0]

    def at_cutpoint(self, item: int, i: int, j: int, k: int) -> Tensor:
        """Return the probability that one span is built at one cutpoint.

        Parameters
        ----------
        item : int
            Which item of the batch to read.
        i : int
            The start of the span, inclusive.
        j : int
            The end of the span, exclusive.
        k : int
            The cutpoint, which divides the span into ``[i, k)`` and ``[k, j)``.

        Returns
        -------
        Tensor
            A scalar in the marginal's floating dtype, attached to its autograd graph.

        Raises
        ------
        ValueError
            If the span is empty or reaches outside the input.
        IndexError
            If there is no such item.
        InvalidCutpointError
            If the cutpoint does not divide the span, or the mask marks it as one the
            span does not have.
        """
        _checked_span(self._spans, i, j)

        if k not in self._spans.splits(i, j):
            raise InvalidCutpointError(item=item, span=(i, j), cut=k)

        span, column = self._spans.split_id(i, j, k)

        if not bool(self._cutpoints[item, span, column]):
            raise InvalidCutpointError(item=item, span=(i, j), cut=k)

        return self._rule.tensor[item, span, column]

    def constituent_mass(self) -> Tensor:
        """Return how much constituent mass each item's spans hold together.

        Returns
        -------
        Tensor
            A tensor of rank 1 over the items, of the marginal's floating dtype.
        """
        values = self._constituent.tensor
        kept = torch.where(self._built, values, values.new_zeros(()))

        return kept.sum(dim=len(SPAN_AXES) - 1)

    @override
    def __repr__(self) -> str:
        """Return the axes and item lengths."""
        return (
            f"SpanMarginals(items={self.items}, spans={self._spans.n}, "
            f"lengths={self._lengths.tolist()})"
        )


def _checked_lengths(lengths: Tensor, spans: SpanIndex, items: int) -> Tensor:
    """Return the item lengths.

    Parameters
    ----------
    lengths : Tensor
        How many tokens each item holds.
    spans : SpanIndex
        The addressing the marginals are over.
    items : int
        How many items the marginals cover.

    Returns
    -------
    Tensor
        The lengths, unchanged.

    Raises
    ------
    TypeError
        If the lengths are not of integer dtype.
    ValueError
        If they are not of rank one, do not cover the items, or name a length outside
        the range the index addresses.
    """
    if lengths.is_floating_point() or lengths.dtype is torch.bool:
        msg = f"a token count is an integer, not {lengths.dtype}"
        raise TypeError(msg)

    if lengths.ndim != 1 or int(lengths.shape[0]) != items:
        msg = (
            f"the lengths have shape {tuple(lengths.shape)}, and there are "
            f"{items} items to give a length to"
        )
        raise ValueError(msg)

    if items and not bool(((lengths >= 1) & (lengths <= spans.n)).all()):
        msg = (
            f"an item's length is outside the range 1 to {spans.n} the index "
            f"addresses: {lengths.tolist()}"
        )
        raise ValueError(msg)

    return lengths


def _reject_leaf_constituents(spans: SpanIndex, mask: Tensor) -> None:
    """Raise if the constituent mask keeps a span no derivation builds.

    Parameters
    ----------
    spans : SpanIndex
        The addressing the marginal is over.
    mask : Tensor
        True where a span's marginal is real.

    Raises
    ------
    ValueError
        If a span of width one is kept.
    """
    leaves = [spans.span_id(i, j) for i, j in spans.leaves()]

    if leaves and bool(mask[:, leaves].any()):
        msg = (
            "the constituent mask keeps a span of width one, which no binary "
            "derivation builds"
        )
        raise ValueError(msg)
