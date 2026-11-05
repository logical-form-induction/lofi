"""Proof-figure and bracketed-string renderings of derivations."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.deduce import Derivation

__all__ = ["BraceString", "ProofFigure"]

_NO_SYMBOLS: Final[Mapping[str, str]] = MappingProxyType({})
"""Rule-name fallback for missing display symbols."""

_GAP: Final = 1
"""The minimum gap between proof-figure columns."""


def _postorder[T, W](derivation: Derivation[T, W]) -> list[Derivation[T, W]]:
    """Return derivation nodes in postorder.

    Parameters
    ----------
    derivation : Derivation[T, W]
        Tree to traverse.

    Returns
    -------
    list[Derivation[T, W]]
        Nodes ordered with children before parents.
    """
    order: list[Derivation[T, W]] = []
    stack: list[Derivation[T, W]] = [derivation]

    while stack:
        node = stack.pop()
        order.append(node)
        stack.extend(node.children)

    order.reverse()

    return order


def _starts(widths: list[int], gaps: list[int]) -> list[int]:
    """Return the starting offset of each figure column.

    Parameters
    ----------
    widths : list[int]
        Column widths.
    gaps : list[int]
        Spaces following each column except the last.

    Returns
    -------
    list[int]
        Zero-based column offsets.
    """
    offsets = [0] * len(widths)
    running = 0

    for at in range(1, len(widths)):
        running += widths[at - 1] + gaps[at - 1]
        offsets[at] = running

    return offsets


def _extent(widths: list[int], gaps: list[int], first: int, last: int) -> int:
    """Return the width of a column range, including gaps.

    Parameters
    ----------
    widths : list[int]
        Column widths.
    gaps : list[int]
        Spaces following each column except the last.
    first : int
        First column in the range.
    last : int
        Exclusive end of the range.

    Returns
    -------
    int
        Width in characters.
    """
    return sum(widths[first:last]) + sum(gaps[first : last - 1])


def _place(line: list[str], at: int, text: str) -> None:
    """Write text into a figure line, extending it as needed.

    Parameters
    ----------
    line : list[str]
        Mutable character buffer.
    at : int
        Starting offset.
    text : str
        Text to write.
    """
    if at + len(text) > len(line):
        line.extend(" " * (at + len(text) - len(line)))

    line[at : at + len(text)] = list(text)


class ProofFigure[T]:
    """Renders derivations as proof figures.

    Parameters
    ----------
    tokens : Sequence[str]
        Input tokens in position order.
    spell : Callable[[T], str]
        Function that renders an output term.
    symbols : Mapping[str, str]
        Display symbols keyed by rule name.
    """

    __slots__ = ("_spell", "_symbols", "_tokens")

    def __init__(
        self,
        tokens: Sequence[str],
        spell: Callable[[T], str],
        symbols: Mapping[str, str] = _NO_SYMBOLS,
    ) -> None:
        self._tokens: tuple[str, ...] = tuple(tokens)
        self._spell: Callable[[T], str] = spell
        self._symbols: dict[str, str] = dict(symbols)

    @property
    def name(self) -> str:
        """Return the style name."""
        return "proof-figure"

    def render[W](self, derivation: Derivation[T, W], /) -> str:
        """Return a derivation drawn as a proof figure.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Tree to render.

        Returns
        -------
        str
            Figure without trailing spaces or a trailing newline.

        Raises
        ------
        ValueError
            If the derivation extends beyond the configured tokens.
        """
        first, last = derivation.key.span

        if last > len(self._tokens):
            msg = (
                f"the derivation covers tokens up to {last} and the style holds "
                f"{len(self._tokens)}"
            )
            raise ValueError(msg)

        nodes = _postorder(derivation)
        heights = _heights(nodes)
        widths = [len(token) for token in self._tokens[first:last]]
        gaps = [_GAP] * max(len(widths) - 1, 0)

        self._reserve(nodes, first, last, widths, gaps)

        return self._draw(nodes, heights, first, widths, gaps)

    def _reserve[W](
        self,
        nodes: list[Derivation[T, W]],
        first: int,
        last: int,
        widths: list[int],
        gaps: list[int],
    ) -> None:
        """Widen columns and gaps until every node fits.

        Parameters
        ----------
        nodes : list[Derivation[T, W]]
            Nodes in postorder.
        first : int
            First token covered by the derivation.
        last : int
            Exclusive end of the derivation span.
        widths : list[int]
            Mutable column widths.
        gaps : list[int]
            Mutable gaps between columns.
        """
        for node in nodes:
            if not node.children:
                continue

            end = node.key.span[1]

            if end < last:
                wanted = len(self._symbol(node)) + _GAP
                gaps[end - 1 - first] = max(gaps[end - 1 - first], wanted)

        for node in nodes:
            begin, end = node.key.span
            wanted = len(self._spell(node.edge.output))
            have = _extent(widths, gaps, begin - first, end - first)

            if have < wanted:
                widths[end - 1 - first] += wanted - have

    def _draw[W](
        self,
        nodes: list[Derivation[T, W]],
        heights: dict[Derivation[T, W], int],
        first: int,
        widths: list[int],
        gaps: list[int],
    ) -> str:
        """Return the proof figure for a reserved layout.

        Parameters
        ----------
        nodes : list[Derivation[T, W]]
            Nodes in postorder.
        heights : dict[Derivation[T, W], int]
            Height of each node above its deepest leaf.
        first : int
            First token covered by the derivation.
        widths : list[int]
            Reserved column widths.
        gaps : list[int]
            Reserved gaps between columns.

        Returns
        -------
        str
            Rendered proof figure.
        """
        offsets = _starts(widths, gaps)
        rows = 2 + 2 * max(heights.values())
        lines: list[list[str]] = [[] for _ in range(rows)]

        for node in nodes:
            begin, end = node.key.span
            left = offsets[begin - first]
            across = _extent(widths, gaps, begin - first, end - first)
            row = 2 * heights[node]
            drawn = self._spell(node.edge.output)

            if node.children:
                _place(lines[row], left, "-" * across + self._symbol(node))
            else:
                covered = " ".join(self._tokens[begin:end])
                _place(lines[row], left + (across - len(covered)) // 2, covered)

            _place(lines[row + 1], left + (across - len(drawn)) // 2, drawn)

        return "\n".join("".join(line).rstrip() for line in lines)

    def _symbol[W](self, node: Derivation[T, W]) -> str:
        """Return a step's inference-line symbol.

        Parameters
        ----------
        node : Derivation[T, W]
            Derivation step.

        Returns
        -------
        str
            Configured symbol or the rule name when no symbol is configured.
        """
        rule = node.edge.rule_name

        return self._symbols.get(rule, rule)

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the input length and the symbol count."""
        return f"ProofFigure(tokens={len(self._tokens)}, symbols={len(self._symbols)})"


class BraceString[T]:
    """Renders derivations in CCGbank brace notation.

    Parameters
    ----------
    tokens : Sequence[str]
        Input tokens in position order.
    tags : Sequence[str]
        Part-of-speech tag for each token.
    spell : Callable[[T], str]
        Function that renders an output term.
    heads : Mapping[str, int]
        Zero-based head-premise index keyed by rule name.

    Raises
    ------
    ValueError
        If token and tag counts differ.
    """

    __slots__ = ("_heads", "_spell", "_tags", "_tokens")

    def __init__(
        self,
        tokens: Sequence[str],
        tags: Sequence[str],
        spell: Callable[[T], str],
        heads: Mapping[str, int],
    ) -> None:
        if len(tokens) != len(tags):
            msg = f"{len(tokens)} tokens cannot align with {len(tags)} tags"
            raise ValueError(msg)

        self._tokens: tuple[str, ...] = tuple(tokens)
        self._tags: tuple[str, ...] = tuple(tags)
        self._spell: Callable[[T], str] = spell
        self._heads: dict[str, int] = dict(heads)

    @property
    def name(self) -> str:
        """Return the style name."""
        return "brace-string"

    def render[W](self, derivation: Derivation[T, W], /) -> str:
        """Return a derivation as a bracketed string.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Tree to render.

        Returns
        -------
        str
            Single-line bracketed derivation.

        Raises
        ------
        ValueError
            If the derivation exceeds the configured tokens or a rule lacks a
            valid head-premise index.
        """
        last = derivation.key.span[1]

        if last > len(self._tokens):
            msg = (
                f"the derivation covers tokens up to {last} and the style holds "
                f"{len(self._tokens)}"
            )
            raise ValueError(msg)

        return self._node(derivation)

    def _node[W](self, derivation: Derivation[T, W]) -> str:
        """Return a bracketed rendering of a derivation subtree.

        Parameters
        ----------
        derivation : Derivation[T, W]
            Subtree to render.

        Returns
        -------
        str
            Bracketed subtree.

        Raises
        ------
        ValueError
            If a nonunary rule lacks a valid head-premise index.
        """
        category = self._spell(derivation.edge.output)
        children = derivation.children
        first, last = derivation.key.span

        if not children:
            word = "_".join(self._tokens[first:last])
            tag = self._tags[first]

            return f"(<L {category} {tag} {tag} {word} {category}>)"

        rule = derivation.edge.rule_name
        at = 0 if len(children) == 1 else self._heads.get(rule, -1)

        if at < 0:
            msg = f"the head table names no premise of {rule!r}"
            raise ValueError(msg)

        if at >= len(children):
            msg = (
                f"the head table names premise {at} of {rule!r}, which has "
                f"{len(children)} premises here"
            )
            raise ValueError(msg)

        inside = " ".join(self._node(child) for child in children)

        return f"(<T {category} {at} {len(children)}> {inside} )"

    @override
    def __repr__(self) -> str:
        """Return a rendering naming the input length and the rules covered."""
        return f"BraceString(tokens={len(self._tokens)}, heads={len(self._heads)})"


def _heights[T, W](
    nodes: list[Derivation[T, W]],
) -> dict[Derivation[T, W], int]:
    """Return each node's height above its deepest leaf.

    Parameters
    ----------
    nodes : list[Derivation[T, W]]
        Nodes in postorder.

    Returns
    -------
    dict[Derivation[T, W], int]
        Node heights, with leaves at zero.
    """
    heights: dict[Derivation[T, W], int] = {}

    for node in nodes:
        heights[node] = (
            0
            if not node.children
            else 1 + max(heights[child] for child in node.children)
        )

    return heights
