"""Ordered tree edit distance for embedding regression."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._native import Term
    from lofi.terms._values import Operator

__all__ = [
    "NodeLabel",
    "RelabelCost",
    "TreeEditDistance",
    "class_sensitive_relabel",
]


type NodeLabel = tuple[Operator, int | None]
"""One term node's operator and payload."""

type RelabelCost = Callable[[NodeLabel, NodeLabel], int]
"""A cost for replacing one node label with another."""


def class_sensitive_relabel(left: NodeLabel, right: NodeLabel, /) -> int:
    """Return the class-sensitive cost of replacing one node label."""
    if left == right:
        return 0

    left_operator, _ = left
    right_operator, _ = right

    return 1 if left_operator.family == right_operator.family else 2


def _postorder(
    term: Term,
) -> tuple[tuple[Term, ...], tuple[int, ...], tuple[int, ...]]:
    """Return postorder nodes, leftmost descendants, and keyroots."""
    nodes: list[Term] = []
    leftmost = [0]

    def visit(current: Term) -> None:
        first_leaf: int | None = None

        for position, child in enumerate(current.args):
            visit(child)

            if position == 0:
                first_leaf = leftmost[len(nodes)]

        nodes.append(current)
        leftmost.append(len(nodes) if first_leaf is None else first_leaf)

    visit(term)
    last_for_leaf: dict[int, int] = {}

    for index in range(1, len(nodes) + 1):
        last_for_leaf[leftmost[index]] = index

    return tuple(nodes), tuple(leftmost), tuple(sorted(last_for_leaf.values()))


def _distance(
    left: Term,
    right: Term,
    relabel: RelabelCost,
) -> int:
    """Return Zhang-Shasha distance between two nonempty trees."""
    left_nodes, leftmost_left, left_keyroots = _postorder(left)
    right_nodes, leftmost_right, right_keyroots = _postorder(right)
    tree_distances = [[0] * (len(right_nodes) + 1) for _ in range(len(left_nodes) + 1)]

    for left_root in left_keyroots:
        left_start = leftmost_left[left_root]

        for right_root in right_keyroots:
            right_start = leftmost_right[right_root]
            forest_distances = [
                [0] * (len(right_nodes) + 1) for _ in range(len(left_nodes) + 1)
            ]

            for left_index in range(left_start, left_root + 1):
                forest_distances[left_index][right_start - 1] = (
                    forest_distances[left_index - 1][right_start - 1] + 1
                )

            for right_index in range(right_start, right_root + 1):
                forest_distances[left_start - 1][right_index] = (
                    forest_distances[left_start - 1][right_index - 1] + 1
                )

            for left_index in range(left_start, left_root + 1):
                for right_index in range(right_start, right_root + 1):
                    delete = forest_distances[left_index - 1][right_index] + 1
                    insert = forest_distances[left_index][right_index - 1] + 1

                    if (
                        leftmost_left[left_index] == left_start
                        and leftmost_right[right_index] == right_start
                    ):
                        replace = forest_distances[left_index - 1][
                            right_index - 1
                        ] + relabel(
                            (
                                left_nodes[left_index - 1].op,
                                left_nodes[left_index - 1].payload,
                            ),
                            (
                                right_nodes[right_index - 1].op,
                                right_nodes[right_index - 1].payload,
                            ),
                        )
                        distance = min(delete, insert, replace)
                        forest_distances[left_index][right_index] = distance
                        tree_distances[left_index][right_index] = distance

                    else:
                        align = (
                            forest_distances[leftmost_left[left_index] - 1][
                                leftmost_right[right_index] - 1
                            ]
                            + tree_distances[left_index][right_index]
                        )
                        forest_distances[left_index][right_index] = min(
                            delete, insert, align
                        )

    return tree_distances[-1][-1]


class TreeEditDistance:
    """Zhang-Shasha distance with a caller-owned cache."""

    __slots__ = ("_cache", "_relabel")

    def __init__(self, relabel: RelabelCost = class_sensitive_relabel) -> None:
        self._relabel = relabel
        self._cache: dict[
            tuple[Term | None, Term | None],
            int,
        ] = {}

    def __call__(self, left: Term | None, right: Term | None, /) -> int:
        """Return the ordered tree edit distance between two terms."""
        pair = (left, right)
        cached = self._cache.get(pair)

        if cached is not None:
            return cached

        if left is None:
            distance = 0 if right is None else right.size

        elif right is None:
            distance = left.size

        else:
            distance = _distance(left, right, self._relabel)

        self._cache[pair] = distance

        return distance

    def clear(self) -> None:
        """Release all cached term pairs."""
        self._cache.clear()
