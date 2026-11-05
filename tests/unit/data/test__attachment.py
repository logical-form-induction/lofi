"""Tests for the corpus-generic attachment-tree value."""

from copy import copy, deepcopy
from pickle import dumps, loads
from typing import cast

import pytest

from lofi.data import AttachmentTree, Dependency, Token, attachment_tree
from lofi.data.deps import HeadProjection


def test_a_tree_validates_heads_root_and_reachability() -> None:
    with pytest.raises(ValueError, match="are given"):
        AttachmentTree(("a", "b"), (0,))
    with pytest.raises(ValueError, match="heads itself"):
        AttachmentTree(("a", "b"), (0, 2))
    with pytest.raises(ValueError, match="outside 2 tokens"):
        AttachmentTree(("a", "b"), (0, 3))
    with pytest.raises(ValueError, match="0 tokens attach to the root"):
        AttachmentTree(("a", "b"), (2, 1))
    with pytest.raises(ValueError, match="2 tokens attach to the root"):
        AttachmentTree(("a", "b"), (0, 0))
    with pytest.raises(ValueError, match="does not reach"):
        AttachmentTree(("a", "b", "c"), (0, 3, 2))
    with pytest.raises(ValueError, match="punctuation position 0"):
        AttachmentTree(("a",), (0,), punctuation=(0,))
    with pytest.raises(TypeError, match="not an integer"):
        AttachmentTree(("a",), cast("tuple[int, ...]", (0.0,)))


def test_a_tree_is_a_frozen_reconstructible_value() -> None:
    tree = AttachmentTree(("a", "b"), (0, 1), punctuation=(2,))
    same = AttachmentTree(("a", "b"), (0, 1), punctuation=[2])

    assert tree == same
    assert hash(tree) == hash(same)
    assert loads(dumps(tree)) == tree
    assert copy(tree) is tree
    assert deepcopy(tree) is tree
    assert tree.arcs() == {(0, 1), (1, 2)}
    assert tree.head_of(2) == 1
    assert repr(tree) == "AttachmentTree(('a', 'b'), (0, 1), [2])"

    field = "_heads"
    with pytest.raises(AttributeError, match="immutable"):
        setattr(tree, field, (0, 0))


def test_a_projection_converts_once_through_the_published_data_value() -> None:
    projection = HeadProjection(
        (1,),
        (
            Dependency(2, 1, "dependent"),
            Dependency(0, 2, "root"),
        ),
    )
    tokens = (Token(0, "a"), Token(1, "b"))

    tree = attachment_tree(projection, tokens, punctuation=(2,))

    assert tree == AttachmentTree(("a", "b"), (2, 0), punctuation=(2,))

    with pytest.raises(ValueError, match="has index"):
        attachment_tree(projection, tuple(reversed(tokens)))
    with pytest.raises(ValueError, match="more than one head"):
        attachment_tree(
            HeadProjection(
                projection.heads,
                (*projection.dependencies, Dependency(0, 1, "other")),
            ),
            tokens,
        )
