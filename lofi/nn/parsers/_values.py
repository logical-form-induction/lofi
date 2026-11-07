"""Parser result containers."""

from typing import TYPE_CHECKING, NamedTuple, Never, override

from lofi.nn.parsers._states import BINARY

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.deduce import Cell, SpanIndex

__all__ = ["BracketedTree", "ChanneledChart", "ScoredChart", "TaggedTree"]


def _checked(name: str, tensor: Tensor, rank: int) -> Tensor:
    """Return one of a chart's tensors.

    Parameters
    ----------
    name : str
        What to call the tensor in a message.
    tensor : Tensor
        The tensor to check.
    rank : int
        How many axes it runs over.

    Returns
    -------
    Tensor
        The tensor, unchanged.

    Raises
    ------
    TypeError
        If it is not of a floating point dtype.
    ValueError
        If it does not have the rank its axes call for.
    """
    if not tensor.is_floating_point():
        msg = f"{name} accumulates in a floating point dtype, not {tensor.dtype}"
        raise TypeError(msg)

    if tensor.ndim != rank:
        msg = f"{name} runs over {rank} axes, so it has rank {rank}, not {tensor.ndim}"
        raise ValueError(msg)

    return tensor


class ChanneledChart:
    """Stores scalar inside values, split posteriors, and mixed vectors.

    Parameters
    ----------
    inside : Tensor
        Rank-2 floating-point scores shaped ``(item, span)``.
    alpha : Tensor
        Rank-3 floating-point split posteriors shaped ``(item, span, split)``.
    cells : Tensor
        Rank-3 floating-point vectors shaped ``(span, item, width)``.
    index : SpanIndex
        The span index shared by all three tensors.

    Raises
    ------
    TypeError
        If any tensor is not floating point.
    ValueError
        If a tensor has the wrong rank or an incompatible axis extent.
    """

    __slots__ = ("_alpha", "_cells", "_index", "_inside")

    def __init__(
        self, inside: Tensor, alpha: Tensor, cells: Tensor, index: SpanIndex, /
    ) -> None:
        # validate ranks before checking axis extents
        _checked("an inside chart", inside, 2)
        _checked("a split posterior", alpha, 3)
        _checked("a vector chart", cells, 3)

        batch = int(inside.shape[0])
        scores = (batch, index.num_spans)
        splits = (batch, index.num_spans, index.max_splits)
        leading = (index.num_spans, batch)

        if tuple(inside.shape) != scores:
            msg = f"an inside chart has shape {scores}, not {tuple(inside.shape)}"
            raise ValueError(msg)

        if tuple(alpha.shape) != splits:
            msg = f"a split posterior has shape {splits}, not {tuple(alpha.shape)}"
            raise ValueError(msg)

        if tuple(cells.shape[:2]) != leading:
            msg = (
                f"a vector chart runs over {leading} before its width axis, "
                f"not over {tuple(cells.shape[:2])}"
            )
            raise ValueError(msg)

        self._inside = inside
        self._alpha = alpha
        self._cells = cells
        self._index = index

    @property
    def index(self) -> SpanIndex:
        """Return the span index shared by the charts."""
        return self._index

    @property
    def inside(self) -> Tensor:
        """Return rank-2 log inside scores over items and spans."""
        return self._inside

    @property
    def alpha(self) -> Tensor:
        """Return rank-3 log split posteriors over items, spans, and splits."""
        return self._alpha

    @property
    def cells(self) -> Tensor:
        """Return rank-3 cell vectors over spans, items, and features."""
        return self._cells

    @property
    def batch(self) -> int:
        """Return the batch size."""
        return int(self._inside.shape[0])

    @property
    def width(self) -> int:
        """Return the cell-vector width."""
        return int(self._cells.shape[-1])

    @override
    def __repr__(self) -> str:
        """Return the input length, the batch and the width of one cell."""
        return (
            f"ChanneledChart(n={self._index.n}, batch={self.batch}, width={self.width})"
        )


class ScoredChart(NamedTuple):
    """Stores a channeled chart beside the split scores its sweep read.

    Parameters
    ----------
    chart : ChanneledChart
        The scalar inside values, the split posteriors and the mixed vectors.
    split_scores : Tensor
        Rank-3 floating-point log weights shaped ``(item, span, split)``, with
        negative infinity at every column a span does not have.
    """

    chart: ChanneledChart
    split_scores: Tensor


class BracketedTree:
    """Stores one binary bracketing and its decoding score.

    Parameters
    ----------
    constituents : tuple[Cell, ...]
        The spans of two tokens or more the tree holds, in ascending span id order.
    total : float
        What the decode that produced the tree maximized.
    """

    __slots__ = ("_constituents", "_total")

    def __init__(self, constituents: tuple[Cell, ...], total: float, /) -> None:
        self._constituents = constituents
        self._total = total

    @property
    def constituents(self) -> tuple[Cell, ...]:
        """Return the spans of two tokens or more the tree holds, in span id order."""
        return self._constituents

    @property
    def total(self) -> float:
        """Return the decoding score."""
        return self._total

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two trees hold the same constituents."""
        if type(other) is not BracketedTree:
            return NotImplemented

        return self._constituents == other._constituents

    @override
    def __hash__(self) -> int:
        """Return a hash consistent with equality."""
        return hash((BracketedTree, self._constituents))

    @override
    def __repr__(self) -> str:
        """Return the constituents and the total."""
        return f"BracketedTree({self._constituents!r}, {self._total!r})"


type _Identity = tuple["Cell", tuple[int, int], int | None, tuple["_Identity", ...]]
"""A derivation's spans, states and rules, nested as its premises are."""


def _identity(tree: TaggedTree, /) -> _Identity:
    """Return the fields equality compares, premises resolved first.

    Parameters
    ----------
    tree : TaggedTree
        The tagged tree.
    """
    return (
        tree.span,
        (tree.category, tree.tag),
        tree.rule,
        tuple(_identity(child) for child in tree.children),
    )


class TaggedTree:
    """Stores one derivation over categories and normal form tags.

    Parameters
    ----------
    span : Cell
        The tokens the derivation covers.
    state : tuple[int, int]
        The category and tag indices its conclusion occupies.
    rule : int | None
        The index of the rule that concluded it in the combining or unary channel, or
        None for a lexical leaf.
    children : tuple[TaggedTree, ...]
        The premise derivations in rule order.
    total : float
        The summed weight the decode ranked the subtree by.
    """

    __slots__ = ("_children", "_rule", "_span", "_state", "_total")

    def __init__(
        self,
        span: Cell,
        state: tuple[int, int],
        rule: int | None,
        children: tuple[TaggedTree, ...],
        total: float,
        /,
    ) -> None:
        object.__setattr__(self, "_span", span)
        object.__setattr__(self, "_state", state)
        object.__setattr__(self, "_rule", rule)
        object.__setattr__(self, "_children", children)
        object.__setattr__(self, "_total", total)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a tagged derivation is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a tagged derivation is immutable"
        raise AttributeError(msg)

    @property
    def span(self) -> Cell:
        """Return the tokens the derivation covers."""
        return self._span

    @property
    def category(self) -> int:
        """Return the category index of its conclusion."""
        return self._state[0]

    @property
    def tag(self) -> int:
        """Return the normal form tag index of its conclusion."""
        return self._state[1]

    @property
    def rule(self) -> int | None:
        """Return the concluding rule's channel index, or None for a lexical leaf."""
        return self._rule

    @property
    def children(self) -> tuple[TaggedTree, ...]:
        """Return the premise derivations."""
        return self._children

    @property
    def total(self) -> float:
        """Return the decoding score of the subtree."""
        return self._total

    def constituents(self) -> tuple[Cell, ...]:
        """Return the spans of two tokens or more the derivation brackets.

        Returns
        -------
        tuple[Cell, ...]
            Each combining step's span, outermost first.
        """
        found: list[Cell] = []

        if len(self._children) >= BINARY:
            found.append(self._span)

        for child in self._children:
            found.extend(child.constituents())

        return tuple(found)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether two derivations have the same spans, states and rules."""
        if type(other) is not TaggedTree:
            return NotImplemented

        return _identity(self) == _identity(other)

    @override
    def __hash__(self) -> int:
        """Return a hash consistent with equality."""
        return hash((TaggedTree, _identity(self)))

    @override
    def __repr__(self) -> str:
        """Return the span, the state, the rule and the premise count."""
        return (
            f"TaggedTree({self._span!r}, {self._state!r}, {self._rule!r}, "
            f"{len(self._children)} premises, {self._total!r})"
        )
