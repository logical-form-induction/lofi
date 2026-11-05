from typing import Final

import pytest

from lofi.terms import Operator, Signature, TermStore, at_path, positions, replace_at

SIGNATURE: Final = Signature(
    "path-categories",
    ("Cat",),
    (
        Operator("n", "Cat"),
        Operator("np", "Cat"),
        Operator("s", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("ternary", "Cat", arg_sorts=("Cat", "Cat", "Cat")),
    ),
)
STORE: Final = TermStore(SIGNATURE)
N: Final = STORE.op(SIGNATURE.by_name("n"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
S: Final = STORE.op(SIGNATURE.by_name("s"))
FWD: Final = SIGNATURE.by_name("fwd")
BWD: Final = SIGNATURE.by_name("bwd")
TERNARY: Final = SIGNATURE.by_name("ternary")
INTRANSITIVE: Final = STORE.op(BWD, S, NP)
TRANSITIVE: Final = STORE.op(FWD, INTRANSITIVE, NP)
DITRANSITIVE: Final = STORE.op(FWD, TRANSITIVE, NP)


def test_the_empty_and_nested_paths_read_the_tree_shape() -> None:
    assert at_path(DITRANSITIVE, ()) is DITRANSITIVE
    assert at_path(DITRANSITIVE, (0,)) is TRANSITIVE
    assert at_path(DITRANSITIVE, (1,)) is NP
    assert at_path(DITRANSITIVE, (0, 0)) is INTRANSITIVE
    assert at_path(DITRANSITIVE, (0, 0, 0)) is S
    assert at_path(DITRANSITIVE, (0, 0, 1)) is NP


def test_positions_are_root_first_and_left_to_right_in_preorder() -> None:
    term = STORE.op(TERNARY, STORE.op(FWD, S, NP), N, STORE.op(BWD, NP, S))

    assert positions(term, lambda _term: True) == (
        (),
        (0,),
        (0, 0),
        (0, 1),
        (1,),
        (2,),
        (2, 0),
        (2, 1),
    )
    assert positions(term, lambda candidate: candidate.op.arity == 0) == (
        (0, 0),
        (0, 1),
        (1,),
        (2, 0),
        (2, 1),
    )


def test_replacement_preserves_the_functional_result() -> None:
    result = replace_at(STORE, DITRANSITIVE, (0,), INTRANSITIVE)

    assert result is TRANSITIVE
    assert DITRANSITIVE.args == (TRANSITIVE, NP)


def test_nested_replacement_rebuilds_only_the_ancestor_spine() -> None:
    result = replace_at(STORE, DITRANSITIVE, (0, 0, 0), N)

    assert at_path(result, (0, 0, 0)) is N
    assert result is not DITRANSITIVE
    assert result.args[1] is NP
    assert result.args[0].args[1] is NP
    assert result.args[0].args[0].args[1] is NP
    assert DITRANSITIVE.args == (TRANSITIVE, NP)
    assert at_path(DITRANSITIVE, (0, 0, 0)) is S


def test_replacing_a_path_with_its_value_returns_the_original_identity() -> None:
    assert replace_at(STORE, DITRANSITIVE, (0, 0), INTRANSITIVE) is DITRANSITIVE


def test_root_replacement_returns_the_supplied_value() -> None:
    assert replace_at(STORE, DITRANSITIVE, (), N) is N


@pytest.mark.parametrize("path", [(0,), (-1,), (2,), (0, 1, 0)])
def test_an_invalid_child_names_the_complete_path_and_failing_node(
    path: tuple[int, ...],
) -> None:
    with pytest.raises(IndexError) as caught:
        at_path(N if len(path) == 1 else DITRANSITIVE, path)

    assert repr(path) in str(caught.value)
    assert repr(N if len(path) == 1 else NP) in str(caught.value)


def test_replacement_reports_an_invalid_path_before_rebuilding() -> None:
    before = STORE.misses

    with pytest.raises(IndexError, match=r"path \(0, 2\).+no child at index 2"):
        replace_at(STORE, DITRANSITIVE, (0, 2), N)

    assert STORE.misses == before
    assert DITRANSITIVE.args == (TRANSITIVE, NP)
