from typing import Final

from lofi.nn._contexts import child_sides, slot_paths, teacher_contexts
from lofi.terms import (
    ChoiceIndex,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    TermCodec,
    TermStore,
)

SIGNATURE: Final = Signature(
    "decoder-context",
    ("term",),
    (
        Operator("a", "term", family="primitive"),
        Operator("b", "term", family="primitive"),
        Operator("pair", "term", arg_sorts=("term", "term"), family="constructor"),
    ),
)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"term": 3})
STORE: Final = TermStore(SIGNATURE)
A: Final = STORE.op(SIGNATURE.operator(1))
B: Final = STORE.op(SIGNATURE.operator(2))
TERM: Final = STORE.op(SIGNATURE.operator(3), STORE.op(SIGNATURE.operator(3), A, B), A)
ROW: Final = TermCodec(OperatorIndex(SIGNATURE), PreorderArityLayout(7)).encode(TERM)


def test_a_slot_reads_its_argument_path_from_its_parents() -> None:
    assert child_sides(ROW)[:5] == (0, 0, 0, 1, 1)
    assert slot_paths(ROW)[:5] == ((), (0,), (0, 0), (0, 1), (1,))


def test_teacher_contexts_are_the_contexts_frontier_decoding_reaches() -> None:
    contexts = teacher_contexts(CHOICES, ROW)

    assert [context.path for context in contexts[:5]] == [
        (),
        (0,),
        (0, 0),
        (0, 1),
        (1,),
    ]
    assert [context.remaining for context in contexts[:5]] == [6, 4, 2, 2, 2]
    assert all(context.sort == "term" for context in contexts)
