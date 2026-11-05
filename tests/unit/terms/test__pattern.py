from typing import Final, cast

import pytest
from hypothesis import given, settings
from tests.support.term_strategies import terms

from lofi.terms import (
    Bindings,
    MVar,
    Node,
    Operator,
    Pattern,
    PVar,
    Signature,
    Spine,
    Term,
    TermStore,
    Vocabulary,
    build,
    match,
)

TY: Final = "Ty"
SIGNATURE: Final = Signature(
    "pattern-categories",
    (TY,),
    (
        Operator("n", TY),
        Operator("np", TY),
        Operator("s", TY),
        Operator("comma", TY),
        Operator("conj", TY),
        Operator("fwd", TY, arg_sorts=(TY, TY)),
        Operator("bwd", TY, arg_sorts=(TY, TY)),
    ),
    features={"s": ("dcl", "q")},
    vocabularies=(Vocabulary(TY, tuple(f"label{at}" for at in range(8))),),
)
STORE: Final = TermStore(SIGNATURE)
N: Final = STORE.op(SIGNATURE.by_name("n"))
NP: Final = STORE.op(SIGNATURE.by_name("np"))
S: Final = STORE.op(SIGNATURE.by_name("s"))
COMMA: Final = STORE.op(SIGNATURE.by_name("comma"))
CONJ: Final = STORE.op(SIGNATURE.by_name("conj"))
FWD: Final = SIGNATURE.by_name("fwd")
BWD: Final = SIGNATURE.by_name("bwd")


def _schema(term: Term, path: tuple[int, ...] = ()) -> Pattern:
    """Return a pattern that binds each leaf and payload separately."""
    suffix = "root" if not path else "_".join(map(str, path))

    if not term.args:
        return MVar(f"leaf_{suffix}", term.op.sort)

    payload = None if term.payload is None else PVar(f"payload_{suffix}")

    return Node(
        PVar(f"operator_{suffix}"),
        tuple(_schema(child, (*path, at)) for at, child in enumerate(term.args)),
        payload,
    )


@settings(max_examples=2000, deadline=None)
@given(terms(STORE, sort=TY, max_size=8, closed=True))
def test_matching_then_building_returns_the_same_interned_term(
    term: Term,
) -> None:
    pattern = _schema(term)
    bindings = match(pattern, term)

    assert bindings is not None
    assert build(pattern, STORE, bindings) is term


def test_repeated_term_variables_use_pointer_identity() -> None:
    same = STORE.op(FWD, N, N)
    different = STORE.op(FWD, N, NP)
    pattern = Node("fwd", (MVar("X"), MVar("X")))

    assert match(pattern, same) is not None
    assert match(pattern, different) is None


def test_payload_variables_match_and_rebuild_integer_payloads() -> None:
    labeled = STORE.const(TY, "label7")
    pattern = Node("#const[Ty]", payload=PVar("symbol"))
    bindings = match(pattern, labeled)

    assert bindings is not None
    assert bindings.payloads == {"symbol": 7}
    assert build(pattern, STORE, bindings) is labeled


def test_payload_variables_cover_slash_directions_and_featured_operators() -> None:
    featured = STORE.op(SIGNATURE.featured("s", "dcl"))
    functor = STORE.op(FWD, featured, NP)
    pattern = Node(
        PVar("slash"),
        (Node(PVar("feature")), MVar("argument")),
    )
    bindings = match(pattern, functor)

    assert bindings is not None
    assert bindings.payloads == {"slash": "fwd", "feature": "s[dcl]"}
    assert build(pattern, STORE, bindings) is functor


def _spine(order: int, core: Term = N) -> Term:
    """Build a left-result spine of an exact order."""
    built = core

    for at in range(order):
        op = FWD if at % 2 == 0 else BWD
        built = STORE.op(op, built, NP)

    return built


@pytest.mark.parametrize("order", range(4))
def test_spine_peeling_and_building_respect_each_declared_order(order: int) -> None:
    term = _spine(order)
    pattern = Spine(
        "context",
        MVar("core"),
        ops=("fwd", "bwd"),
        slot=0,
        min_order=order,
        max_order=order,
    )
    bindings = match(pattern, term)

    assert bindings is not None
    assert len(bindings.spines["context"]) == order
    assert bindings.terms["core"] is N
    assert build(pattern, STORE, bindings) is term


def test_spine_matching_extends_bindings_across_composition_premises() -> None:
    first_pattern = Node("fwd", (MVar("X"), MVar("Y")))
    second_pattern = Spine(
        "context",
        MVar("Y"),
        ops=("fwd", "bwd"),
        min_order=1,
        max_order=3,
    )
    conclusion = Spine(
        "context",
        MVar("X"),
        ops=("fwd", "bwd"),
        min_order=1,
        max_order=3,
    )
    first = STORE.op(FWD, S, N)
    second = _spine(3)

    bindings = match(first_pattern, first)

    assert bindings is not None

    bindings = match(second_pattern, second, bindings)

    assert bindings is not None
    assert build(conclusion, STORE, bindings) is _spine(3, S)


def test_a_spine_refuses_orders_outside_its_bounds() -> None:
    pattern = Spine(
        "context",
        Node("n"),
        ops=("fwd", "bwd"),
        min_order=2,
        max_order=3,
    )

    assert match(pattern, _spine(1)) is None

    with pytest.raises(ValueError, match="min_order 2 exceeds max_order 1"):
        Spine("context", Node("n"), ops=("fwd",), min_order=2, max_order=1)


def test_bindings_are_read_only() -> None:
    bindings = match(MVar("X"), N)

    assert bindings is not None

    with pytest.raises(TypeError):
        cast("dict[str, Term]", bindings.terms)["Y"] = NP


type PatternValue = MVar | PVar | Node | Spine | Bindings


@pytest.mark.parametrize(
    "value",
    [
        MVar("X"),
        PVar("slash"),
        Node("n"),
        Spine("context", MVar("X"), ("fwd",), max_order=3),
        Bindings(),
    ],
)
def test_pattern_values_are_slotted_and_frozen(value: PatternValue) -> None:
    with pytest.raises(TypeError):
        vars(value)

    with pytest.raises(AttributeError, match="immutable"):
        value.__setattr__("_probe", 1)


def test_pattern_values_use_field_equality_hashing_and_repr() -> None:
    first_bindings = match(
        Spine("context", MVar("X"), ("fwd",), min_order=1, max_order=3),
        _spine(1),
    )
    second_bindings = match(
        Spine("context", MVar("X"), ("fwd",), min_order=1, max_order=3),
        _spine(1),
    )

    assert first_bindings is not None
    assert second_bindings is not None

    pairs: tuple[tuple[PatternValue, PatternValue], ...] = (
        (MVar("X"), MVar("X")),
        (PVar("slash"), PVar("slash")),
        (Node("fwd", (MVar("X"), MVar("Y"))), Node("fwd", (MVar("X"), MVar("Y")))),
        (
            Spine("context", MVar("X"), ("fwd",), min_order=1, max_order=3),
            Spine("context", MVar("X"), ("fwd",), min_order=1, max_order=3),
        ),
        (first_bindings, second_bindings),
    )

    for left, right in pairs:
        assert left == right
        assert hash(left) == hash(right)
        assert repr(left) == repr(right)


type CoercionPattern = tuple[tuple[Pattern, ...], Pattern]

CCGBANK_COERCIONS: Final[dict[str, CoercionPattern]] = {
    "n_to_np": ((Node("n"),), Node("np")),
    "clausal_adjunct": (
        (
            Spine(
                "clause",
                Node(PVar("sentence")),
                ops=("fwd", "bwd"),
                min_order=1,
                max_order=1,
            ),
        ),
        Node(PVar("slash"), (MVar("X"), MVar("X"))),
    ),
    "unlike_coordination": (
        (Node("conj"), MVar("X")),
        Node(PVar("featured_result")),
    ),
    "appositive_extraposition": (
        (Node("comma"), Node("np")),
        Node(PVar("slash"), (MVar("X"), MVar("X"))),
    ),
}
"""Four CCGbank coercions as pattern pairs."""


type PatternPart = Pattern | PVar | str | int | None


def _parts(pattern: Pattern) -> tuple[PatternPart, ...]:
    """Return the values nested in a pattern."""
    if isinstance(pattern, MVar):
        return (pattern, pattern.name, pattern.sort)

    if isinstance(pattern, Node):
        return (
            pattern,
            pattern.op,
            pattern.payload,
            *(part for child in pattern.args for part in _parts(child)),
        )

    return (
        pattern,
        pattern.name,
        *pattern.ops,
        pattern.slot,
        pattern.min_order,
        pattern.max_order,
        *_parts(pattern.core),
    )


def test_four_ccgbank_coercion_pairs_are_data_without_callables() -> None:
    assert tuple(CCGBANK_COERCIONS) == (
        "n_to_np",
        "clausal_adjunct",
        "unlike_coordination",
        "appositive_extraposition",
    )
    patterns = tuple(
        pattern
        for premises, conclusion in CCGBANK_COERCIONS.values()
        for pattern in (*premises, conclusion)
    )
    assert not any(callable(part) for pattern in patterns for part in _parts(pattern))


def test_build_reports_an_unbound_variable() -> None:
    with pytest.raises(KeyError, match="unbound term variable 'X'"):
        build(MVar("X"), STORE, Bindings())
