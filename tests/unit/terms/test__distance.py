from typing import Final

import pytest
from hypothesis import given, settings
from tests.support.term_reference import categories
from tests.support.term_strategies import terms

from lofi.terms import (
    NodeLabel,
    Operator,
    Signature,
    Term,
    TermStore,
    TreeEditDistance,
    class_sensitive_relabel,
)

CATEGORY_SIGNATURE: Final = categories()
CATEGORY_STORE: Final = TermStore(CATEGORY_SIGNATURE)
NP: Final = CATEGORY_STORE.op(CATEGORY_SIGNATURE.by_name("np"))
S: Final = CATEGORY_STORE.op(CATEGORY_SIGNATURE.by_name("s"))
FWD: Final = CATEGORY_SIGNATURE.by_name("fwd")
BWD: Final = CATEGORY_SIGNATURE.by_name("bwd")
INTRANSITIVE: Final = CATEGORY_STORE.op(BWD, S, NP)
TRANSITIVE: Final = CATEGORY_STORE.op(FWD, INTRANSITIVE, NP)

ORDERED_TREE_CASES: Final = (
    (S, NP, 1),
    (S, INTRANSITIVE, 2),
    (S, TRANSITIVE, 4),
    (INTRANSITIVE, TRANSITIVE, 2),
    (TRANSITIVE, INTRANSITIVE, 2),
    (TRANSITIVE, None, 5),
    (None, TRANSITIVE, 5),
)


PUBLISHED_SIGNATURE: Final = Signature(
    "zhang-shasha-figure-4",
    ("Tree",),
    (
        Operator("a", "Tree", family="letter"),
        Operator("b", "Tree", family="letter"),
        Operator("e", "Tree", family="letter"),
        Operator("c", "Tree", arg_sorts=("Tree",), family="letter"),
        Operator(
            "d",
            "Tree",
            arg_sorts=("Tree", "Tree"),
            family="letter",
        ),
        Operator(
            "f",
            "Tree",
            arg_sorts=("Tree", "Tree"),
            family="letter",
        ),
    ),
)
PUBLISHED_STORE: Final = TermStore(PUBLISHED_SIGNATURE)
A: Final = PUBLISHED_STORE.op(PUBLISHED_SIGNATURE.by_name("a"))
B: Final = PUBLISHED_STORE.op(PUBLISHED_SIGNATURE.by_name("b"))
E: Final = PUBLISHED_STORE.op(PUBLISHED_SIGNATURE.by_name("e"))
C: Final = PUBLISHED_SIGNATURE.by_name("c")
D: Final = PUBLISHED_SIGNATURE.by_name("d")
F: Final = PUBLISHED_SIGNATURE.by_name("f")
C_B: Final = PUBLISHED_STORE.op(C, B)
D_A_C: Final = PUBLISHED_STORE.op(D, A, C_B)
FIRST: Final = PUBLISHED_STORE.op(F, D_A_C, E)
D_A_B: Final = PUBLISHED_STORE.op(D, A, B)
C_D: Final = PUBLISHED_STORE.op(C, D_A_B)
SECOND: Final = PUBLISHED_STORE.op(F, C_D, E)
FIRST_POSTORDER: Final = (A, B, C_B, D_A_C, E, FIRST)
SECOND_POSTORDER: Final = (A, B, D_A_B, C_D, E, SECOND)
PUBLISHED_DISTANCES: Final = (
    (0, 1, 2, 3, 1, 5),
    (1, 0, 2, 3, 1, 5),
    (2, 1, 2, 2, 2, 4),
    (3, 3, 1, 2, 4, 4),
    (1, 1, 3, 4, 0, 5),
    (5, 5, 3, 3, 5, 2),
)


@pytest.mark.parametrize(("left", "right", "expected"), ORDERED_TREE_CASES)
def test_the_seven_reference_inputs_have_ordered_tree_distances(
    left: Term | None,
    right: Term | None,
    expected: int,
) -> None:
    assert TreeEditDistance()(left, right) == expected


def test_zhang_and_shasha_figure_eight_subtree_distances_are_reproduced() -> None:
    distance = TreeEditDistance()
    observed = tuple(
        tuple(distance(left, right) for right in SECOND_POSTORDER)
        for left in FIRST_POSTORDER
    )

    assert observed == PUBLISHED_DISTANCES


def test_ordered_siblings_cannot_be_swapped_without_edits() -> None:
    left = CATEGORY_STORE.op(FWD, S, NP)
    right = CATEGORY_STORE.op(FWD, NP, S)

    assert TreeEditDistance()(left, right) == 2


def test_node_labels_include_operator_family_and_payload() -> None:
    payload_operator = Operator(
        "symbol",
        "Tree",
        payload_sort="constant",
        family="lexical",
    )

    assert class_sensitive_relabel((S.op, S.payload), (S.op, S.payload)) == 0
    assert class_sensitive_relabel((S.op, S.payload), (NP.op, NP.payload)) == 1
    assert class_sensitive_relabel((S.op, S.payload), (FWD, None)) == 2
    assert class_sensitive_relabel((payload_operator, 1), (payload_operator, 2)) == 1


def test_a_custom_relabel_cost_controls_node_replacement() -> None:
    def free_relabel(_left: NodeLabel, _right: NodeLabel) -> int:
        return 0

    distance = TreeEditDistance(relabel=free_relabel)

    assert distance(S, NP) == 0


def test_cache_pairs_are_ordered_owned_by_one_instance_and_released() -> None:
    calls: list[tuple[NodeLabel, NodeLabel]] = []

    def counted(left: NodeLabel, right: NodeLabel) -> int:
        calls.append((left, right))

        return class_sensitive_relabel(left, right)

    first = TreeEditDistance(relabel=counted)
    second = TreeEditDistance(relabel=counted)

    assert first(S, NP) == 1
    assert first(S, NP) == 1
    assert len(calls) == 1
    assert first(NP, S) == 1
    assert second(S, NP) == 1
    assert len(calls) == 3

    first.clear()

    assert first(S, NP) == 1
    assert len(calls) == 4


GENERATED_TERMS: Final = terms(
    CATEGORY_STORE,
    sort="Cat",
    max_size=7,
    closed=True,
)


@settings(max_examples=2_000, deadline=None)
@given(left=GENERATED_TERMS, middle=GENERATED_TERMS, right=GENERATED_TERMS)
def test_default_distance_obeys_metric_laws_over_thousands_of_generated_terms(
    left: Term,
    middle: Term,
    right: Term,
) -> None:
    distance = TreeEditDistance()
    left_right = distance(left, right)
    left_middle = distance(left, middle)
    middle_right = distance(middle, right)

    assert left_right >= 0
    assert distance(left, left) == 0
    assert left_right == distance(right, left)
    assert left_right <= left_middle + middle_right
