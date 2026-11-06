import ast
from itertools import product
from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, overload

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.term_reference import categories, formulas

from lofi._checks import LawViolationError
from lofi.terms import (
    BindingSite,
    Choice,
    ChoiceIndex,
    CompleteTree,
    Encoding,
    EncodingOverflowError,
    EncodingShapeError,
    HoleContext,
    Interner,
    Operator,
    OperatorIndex,
    PreorderArityLayout,
    Signature,
    Term,
    TermCodec,
    TermStore,
    UnknownOperatorError,
    UnreachedArityDiagnostic,
    Vocabulary,
)
from lofi.terms._codec import heap_order
from lofi.terms.laws import (
    LAYOUT_LAWS,
    SIGNATURE_INVARIANTS,
    AlphabetFixture,
    LayoutFixture,
    check_layout_laws,
    check_signature_invariants,
    round_trip_is_identity,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from hypothesis.strategies import SearchStrategy

    from lofi.nn.codec._ports import TermCoding

BUILDERS: Final = (categories, formulas)

NAMES: Final = tuple(build().name for build in BUILDERS)


def choice_signature() -> Signature:
    """Return a signature exercising every choice payload."""
    return Signature(
        "choices",
        ("Ty", "Tm"),
        (
            Operator("atom", "Ty", family="constructor"),
            Operator(
                "lam",
                "Tm",
                arg_sorts=("Tm",),
                binding=(BindingSite((0,), "Tm"),),
                family="binder",
            ),
            Operator("eval", "Tm", arg_sorts=("Tm", "Tm"), family="application"),
        ),
        vocabularies=(Vocabulary("Ty", ("e", "t")), Vocabulary("Tm", ("dog",))),
        open_sorts=("Ty", "Tm"),
        holes=True,
    )


def test_choice_index_is_the_single_payload_aware_alphabet() -> None:
    signature = choice_signature()
    index = ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )

    expected = (
        Choice(signature.by_name("#const[Ty]"), 0),
        Choice(signature.by_name("#const[Ty]"), 1),
        Choice(signature.by_name("#fvar[Ty]"), 0),
        Choice(signature.by_name("atom"), None),
        Choice(signature.by_name("#const[Tm]"), 0),
        Choice(signature.by_name("#bvar[Tm]"), 0),
        Choice(signature.by_name("#bvar[Tm]"), 1),
        Choice(signature.by_name("#bvar[Tm]"), 2),
        Choice(signature.by_name("#fvar[Tm]"), 0),
        Choice(signature.by_name("#fvar[Tm]"), 1),
        Choice(signature.by_name("lam"), None),
        Choice(signature.by_name("eval"), None),
    )

    assert tuple(index.choice(code) for code in range(1, index.width)) == expected
    assert all(
        index.index(choice.operator, choice.payload) == code
        for code, choice in enumerate(expected, 1)
    )


def test_choice_order_is_sort_major_and_family_contiguous() -> None:
    signature = choice_signature()
    index = ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )

    keys = tuple(
        (index.operator(code).sort, index.operator(code).family)
        for code in range(1, index.width)
    )

    family_order = tuple(
        dict.fromkeys(
            (operator.sort, operator.family) for operator in signature.operators
        )
    )

    assert keys == tuple(sorted(keys, key=family_order.index))


def test_choice_codes_combine_operator_and_payload_channels() -> None:
    signature = choice_signature()
    store = TermStore(signature)
    index = ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )
    codec = TermCodec(OperatorIndex(signature), PreorderArityLayout(3))
    constant = store.op(signature.constant("Tm"), payload=0)
    term = store.op(signature.by_name("lam"), constant)
    encoded = codec.encode(term)

    assert index.codes(encoded) == (
        index.index(signature.by_name("lam")),
        index.index(signature.constant("Tm"), 0),
        index.PAD,
    )


def test_ranked_variable_choices_instantiate_valid_store_terms() -> None:
    signature = choice_signature()
    store = TermStore(signature)
    index = ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )
    free = index.instantiate(store, index.index(signature.free("Tm"), 1))
    bound = index.instantiate(store, index.index(signature.bound("Tm"), 2))

    assert store.atom_name(free.payload) == "$decoded1"
    assert bound is store.bvar("Tm", 2)


def test_choice_legality_is_sort_specific_and_never_admits_padding_or_holes() -> None:
    signature = choice_signature()
    index = ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )

    for sort, row in index.legal_choices.items():
        assert len(row) == index.width
        assert not row[index.PAD]
        assert not any(
            row[choice] and index.operator(choice).family == "hole"
            for choice in range(1, index.width)
        )
        assert {
            index.operator(choice).sort for choice, allowed in enumerate(row) if allowed
        } == {sort}


def test_choice_legality_respects_frontier_size_and_scope() -> None:
    signature = choice_signature()
    index = ChoiceIndex(
        signature,
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )
    empty = MappingProxyType({})
    root = index.legal(HoleContext((), "Tm", empty, empty, 0))
    scoped = index.legal(HoleContext((0,), "Tm", MappingProxyType({"Tm": 1}), empty, 1))
    bound = index.index(signature.bound("Tm"), 0)
    binder = index.index(signature.by_name("lam"))

    assert not root[bound]
    assert not root[binder]
    assert scoped[bound]
    assert scoped[binder]


def test_choice_families_are_contiguous_ranges_over_choices() -> None:
    index = ChoiceIndex(
        choice_signature(),
        depth_bounds={"Ty": 2, "Tm": 3},
        free_ranks={"Ty": 1, "Tm": 2},
    )

    for (sort, family), held in index.family_slices.items():
        assert held
        assert {
            (index.operator(choice).sort, index.operator(choice).family)
            for choice in held
        } == {(sort, family)}


class CodecStore(Interner, Protocol):
    """A term interner whose signature supplies the codec alphabet."""

    @property
    def signature(self) -> Signature:
        """Return the signature whose operators the store accepts."""
        ...

    def fvar(self, sort: str, name: str, /) -> Term:
        """Return the interned free variable with this sort and name."""
        ...


def alphabet(
    build: Callable[[], Signature],
) -> AlphabetFixture[OperatorIndex, Operator]:
    return AlphabetFixture(
        table=OperatorIndex(build()),
        rebuild=lambda: OperatorIndex(build()),
        operators=lambda index: index.signature.operators,
        code=lambda index, op: index.index(op),
        operator=lambda index, code: index.operator(code),
        arity=lambda index, op: index.arity(index.index(op)),
        sort=lambda _index, op: op.sort,
        family=lambda _index, op: op.family,
        digest=lambda index: index.digest,
        pad=OperatorIndex.PAD,
    )


def test_an_operator_and_its_code_are_two_views_of_one_row() -> None:
    index = OperatorIndex(categories())

    for op in index.signature.operators:
        assert index.operator(index.index(op)) is op
        assert index.index(op.name) == index.index(op)


def test_the_alphabet_is_as_wide_as_the_operators_and_the_padding_code() -> None:
    index = OperatorIndex(categories())

    assert index.width == len(index.signature.operators) + 1
    assert sorted(index.index(op) for op in index.signature.operators) == list(
        range(1, index.width)
    )


def test_the_padding_code_is_padding_and_is_neither_nullary_nor_branching() -> None:
    index = OperatorIndex(formulas())

    assert OperatorIndex.PAD == Signature.PAD
    assert index.is_pad(OperatorIndex.PAD)
    assert not index.is_nullary(OperatorIndex.PAD)

    with pytest.raises(UnknownOperatorError) as caught:
        index.arity(OperatorIndex.PAD)

    assert caught.value.operator == OperatorIndex.PAD


def test_no_code_an_operator_holds_reads_as_padding() -> None:
    index = OperatorIndex(formulas())

    assert not any(index.is_pad(index.index(op)) for op in index.signature.operators)


def test_the_nullary_codes_are_exactly_the_codes_of_the_nullary_operators() -> None:
    index = OperatorIndex(formulas())
    nullary = {index.index(op) for op in index.signature.by_arity(0)}

    assert nullary == {code for code in range(1, index.width) if index.is_nullary(code)}
    assert not index.is_nullary(index.width)
    assert not index.is_nullary(-1)


def test_an_atom_and_a_constructor_can_never_land_on_one_code() -> None:
    index = OperatorIndex(formulas())
    branching = [index.index(op) for op in index.signature.operators if op.arity > 0]

    assert branching
    assert not any(index.is_nullary(code) for code in branching)


def test_the_arity_of_a_code_is_the_arity_of_the_operator_it_stands_for() -> None:
    index = OperatorIndex(formulas())
    arities = [index.arity(code) for code in range(1, index.width)]

    assert arities == [op.arity for op in index.signature.operators]


def test_a_code_outside_the_alphabet_has_no_operator_and_no_arity() -> None:
    index = OperatorIndex(categories())

    for code in (-1, index.width):
        with pytest.raises(UnknownOperatorError):
            index.operator(code)

        with pytest.raises(UnknownOperatorError):
            index.arity(code)


def test_an_operator_of_another_declaration_cannot_borrow_a_code() -> None:
    index = OperatorIndex(categories())
    impostor = Operator("fwd", "Cat", arg_sorts=("Cat",), family="slash")

    with pytest.raises(UnknownOperatorError):
        index.index(impostor)


def test_every_family_is_a_block_of_codes_holding_that_family_alone() -> None:
    index = OperatorIndex(categories())
    slices = index.family_slices

    assert set(slices) == {(op.sort, op.family) for op in index.signature.operators}

    for (sort, family), block in slices.items():
        assert {index.operator(code).family for code in block} == {family}
        assert {index.operator(code).sort for code in block} == {sort}


def test_the_family_blocks_cover_the_alphabet_once_and_leave_padding_out() -> None:
    index = OperatorIndex(formulas())
    covered = [code for block in index.family_slices.values() for code in block]

    assert sorted(covered) == list(range(1, index.width))
    assert OperatorIndex.PAD not in covered


def test_the_family_view_cannot_be_written_through() -> None:
    index = OperatorIndex(categories())

    with pytest.raises(TypeError):
        cast("dict[tuple[str, str], range]", index.family_slices)[("Cat", "x")] = range(
            1
        )


def test_a_family_declared_around_another_is_still_one_block() -> None:
    operators = (
        Operator("np", "Cat", family="primitive"),
        Operator("gap", "Cat", family="gap"),
        Operator("n", "Cat", family="primitive"),
        Operator("app", "Cat", arg_sorts=("Cat", "Cat"), family="application"),
        Operator("not", "Cat", arg_sorts=("Cat",), family="primitive"),
    )
    index = OperatorIndex(Signature("mixed", ("Cat",), operators))

    assert index.family_slices == {
        ("Cat", "primitive"): range(1, 4),
        ("Cat", "gap"): range(4, 5),
        ("Cat", "application"): range(5, 6),
    }


def test_one_family_name_in_two_sorts_is_two_blocks() -> None:
    operators = (
        Operator("e", "Ty", family="primitive"),
        Operator("top", "Lf", family="primitive"),
        Operator("t", "Ty", family="primitive"),
    )
    index = OperatorIndex(Signature("mixed", ("Ty", "Lf"), operators))

    assert index.family_slices == {
        ("Ty", "primitive"): range(1, 3),
        ("Lf", "primitive"): range(3, 4),
    }


def test_the_digest_is_the_signature_s_and_moves_with_the_declaration() -> None:
    sorts = ("Cat",)
    here = (Operator("np", "Cat"), Operator("n", "Cat"))
    index = OperatorIndex(Signature("mixed", sorts, here))
    reordered = OperatorIndex(Signature("mixed", sorts, tuple(reversed(here))))

    assert index.digest == index.signature.digest
    assert index.digest == OperatorIndex(Signature("mixed", sorts, here)).digest
    assert index.digest != reordered.digest


def test_an_index_renders_as_the_signature_it_codes_and_its_width() -> None:
    index = OperatorIndex(categories())

    assert repr(index) == f"OperatorIndex('categories', {index.width} codes)"


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
@pytest.mark.parametrize(
    "law",
    SIGNATURE_INVARIANTS,
    ids=[law.__name__ for law in SIGNATURE_INVARIANTS],
)
def test_every_signature_invariant_holds_of_the_index_over_a_reference(
    law: Callable[[AlphabetFixture[OperatorIndex, Operator]], None],
    build: Callable[[], Signature],
) -> None:
    law(alphabet(build))


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
def test_running_the_family_in_one_call_runs_it_against_an_index(
    build: Callable[[], Signature],
) -> None:
    check_signature_invariants(alphabet(build))


if TYPE_CHECKING:

    def check_neural_codec_boundary(codec: TermCodec) -> None:
        neural_codec: TermCoding = codec
        del neural_codec


# Hypothesis inspects this alias while building recursive strategies.
type Strategy = SearchStrategy[Term]

CAPACITY: Final = 7

THIRTEEN: Final = 13

CODEC_STORE: Final = TermStore(formulas())

OPEN_CCGBANK_PRIMITIVES: Final = (
    "s",
    "np",
    "n",
    "pp",
    "conj",
    ".",
    ",",
    ":",
    ";",
    "lrb",
    "rrb",
)
OPEN_CCGBANK_FEATURES: Final = (
    (
        "s",
        (
            "dcl",
            "wq",
            "q",
            "qem",
            "em",
            "bem",
            "b",
            "frg",
            "for",
            "intj",
            "inv",
            "to",
            "pss",
            "pt",
            "ng",
            "adj",
            "as",
            "poss",
            "asup",
        ),
    ),
    ("n", ("num",)),
    ("np", ("expl", "thr", "nb", "conj")),
)


def open_ccgbank_signature() -> Signature:
    """Return the open CCGBank primitive and feature declaration."""
    primitives = tuple(Operator(name, "Cat") for name in OPEN_CCGBANK_PRIMITIVES)

    return Signature(
        "open-ccgbank",
        ("Cat",),
        (
            *primitives,
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
        features=dict(OPEN_CCGBANK_FEATURES),
    )


def open_ccgbank_categories(store: TermStore) -> tuple[Term, ...]:
    """Return every declared atom and depth-two functor over the open inventory.

    Parameters
    ----------
    store
        The store holding the open CCGBank signature.
    """
    atoms = tuple(store.op(op) for op in store.signature.by_arity(0))

    return (
        *atoms,
        *(
            store.op(op, left, right)
            for op in store.signature.by_arity(2)
            for left, right in product(atoms, repeat=2)
        ),
    )


def leaf(store: CodecStore, op: Operator) -> Strategy:
    if op.payload_sort == "none":
        return st.just(store.op(op))

    if op.payload_sort == "free":
        return st.integers(0, 2).map(lambda at: store.fvar(op.sort, f"x{at}"))

    return st.integers(0, 2).map(lambda code: store.op(op, payload=code))


def terms(store: CodecStore, *, max_leaves: int) -> Strategy:
    inventory = store.signature.operators
    base = st.one_of([leaf(store, op) for op in inventory if op.arity == 0])

    def grow(children: Strategy) -> Strategy:
        return st.one_of(
            [
                st.tuples(*[children] * op.arity).map(
                    lambda args, op=op: store.op(op, *args)
                )
                for op in inventory
                if op.arity
            ]
        )

    return st.recursive(base, grow, max_leaves=max_leaves)


def terms_to_depth(store: CodecStore, depth: int) -> Strategy:
    """Return terms whose structural depth does not exceed a bound.

    Parameters
    ----------
    store
        The store whose signature supplies operators.
    depth
        The maximum structural depth.
    """
    inventory = store.signature.operators
    base = st.one_of([leaf(store, op) for op in inventory if op.arity == 0])

    if depth == 1:
        return base

    children = terms_to_depth(store, depth - 1)
    branches = [
        st.tuples(*[children] * op.arity).map(lambda args, op=op: store.op(op, *args))
        for op in inventory
        if op.arity
    ]

    return st.one_of(base, *branches)


def term_depth(term: Term) -> int:
    """Return the number of levels in a term.

    Parameters
    ----------
    term
        The term to measure.
    """
    return 1 + max((term_depth(arg) for arg in term.args), default=0)


class Unshared:
    """A store that allocates a new node for every call."""

    def __init__(self, signature: Signature) -> None:
        self.signature = signature

    def op(
        self,
        op: Operator,
        /,
        *args: Term,
        payload: int | None = None,
    ) -> Term:
        """Return a node no earlier call can have returned.

        Parameters
        ----------
        op
            The root operator.
        args
            The node arguments.
        payload
            The optional operator payload.
        """
        fresh = TermStore(self.signature)

        def copy(term: Term) -> Term:
            """Rebuild one argument in the fresh store.

            Parameters
            ----------
            term
                The argument to rebuild.
            """
            return fresh.op(
                term.op,
                *(copy(arg) for arg in term.args),
                payload=term.payload,
            )

        return fresh.op(op, *(copy(arg) for arg in args), payload=payload)

    def fvar(self, sort: str, name: str, /) -> Term:
        """Return a free variable no earlier call can have returned."""
        return TermStore(self.signature).fvar(sort, name)


@overload
def layout_fixture(
    build: Callable[[], Signature],
) -> LayoutFixture[Term, TermStore, OperatorIndex]: ...


@overload
def layout_fixture(
    build: Callable[[], Signature],
    store: CodecStore,
) -> LayoutFixture[Term, CodecStore, OperatorIndex]: ...


def layout_fixture(
    build: Callable[[], Signature],
    store: CodecStore | None = None,
) -> (
    LayoutFixture[Term, TermStore, OperatorIndex]
    | LayoutFixture[Term, CodecStore, OperatorIndex]
):
    if store is None:
        interning = TermStore(build())

        return LayoutFixture(
            layout=PreorderArityLayout(CAPACITY),
            terms=terms(interning, max_leaves=8),
            store=interning,
            index=OperatorIndex(interning.signature),
            size=lambda t: t.size,
            fits=lambda t: t.size <= CAPACITY,
        )

    return LayoutFixture(
        layout=PreorderArityLayout(CAPACITY),
        terms=terms(store, max_leaves=8),
        store=store,
        index=OperatorIndex(store.signature),
        size=lambda t: t.size,
        fits=lambda t: t.size <= CAPACITY,
    )


def cat(store: TermStore, name: str, *args: Term) -> Term:
    return store.op(store.signature.by_name(name), *args)


def sentence(store: TermStore) -> Term:
    return cat(store, "s")


def forward(store: TermStore) -> Term:
    return cat(store, "fwd", cat(store, "s"), cat(store, "np"))


def transitive(store: TermStore) -> Term:
    return cat(
        store,
        "fwd",
        cat(store, "bwd", cat(store, "s"), cat(store, "np")),
        cat(store, "np"),
    )


def sequence(names: tuple[str, ...], index: OperatorIndex) -> Encoding:
    codes = tuple(index.index(name) for name in names)
    padding = (OperatorIndex.PAD,) * (CAPACITY - len(codes))

    return Encoding(
        symbols=codes + padding,
        parents=(Encoding.NO_PARENT,) * CAPACITY,
        depths=(0,) * CAPACITY,
        occupied=(True,) * len(codes) + (False,) * len(padding),
    )


def spine(store: TermStore, deep: int) -> Term:
    fwd = store.signature.by_name("fwd")
    right = store.op(store.signature.by_name("np"))
    built = store.op(store.signature.by_name("s"))

    for _ in range(deep):
        built = store.op(fwd, built, right)

    return built


def deep_term(store: TermStore, depth: int) -> Term:
    """Return a term whose first argument reaches a requested depth.

    Parameters
    ----------
    store
        The store used to build the term.
    depth
        The requested first-argument depth.
    """
    op = next(
        candidate
        for candidate in store.signature.operators
        if candidate.arity and candidate.arg_sorts[0] == candidate.sort
    )
    leaves = {
        sort: next(
            store.op(candidate)
            if candidate.payload_sort == "none"
            else store.op(candidate, payload=0)
            for candidate in store.signature.by_arity(0)
            if candidate.sort == sort
        )
        for sort in op.arg_sorts
    }
    built = leaves[op.sort]

    for _ in range(depth):
        args = (built, *(leaves[sort] for sort in op.arg_sorts[1:]))
        built = store.op(op, *args)

    return built


def exact_completion_signature() -> Signature:
    """Return the small signature used to check exact completion."""
    return Signature(
        "exact-completion",
        ("Term",),
        (
            Operator("leaf", "Term"),
            Operator("unary", "Term", arg_sorts=("Term",)),
            Operator("binary", "Term", arg_sorts=("Term", "Term")),
        ),
    )


def terms_with_maximum_depth(store: TermStore, depth: int) -> tuple[Term, ...]:
    """Return every declared term whose depth does not exceed the limit.

    Parameters
    ----------
    store
        The store whose signature supplies the operators.
    depth
        The inclusive depth limit.
    """
    leaves = tuple(store.op(op) for op in store.signature.by_arity(0))

    if depth == 1:
        return leaves

    children = terms_with_maximum_depth(store, depth - 1)
    unary = tuple(
        store.op(op, child) for op in store.signature.by_arity(1) for child in children
    )
    binary = tuple(
        store.op(op, left, right)
        for op in store.signature.by_arity(2)
        for left, right in product(children, repeat=2)
    )

    return (*leaves, *unary, *binary)


def terms_with_maximum_size(store: TermStore, max_nodes: int) -> tuple[Term, ...]:
    """Return every declared term whose size does not exceed the limit.

    Parameters
    ----------
    store
        The store whose signature supplies the operators.
    max_nodes
        The inclusive node limit.
    """
    by_size: list[tuple[Term, ...]] = [()]

    for size in range(1, max_nodes + 1):
        held: list[Term] = []

        if size == 1:
            held.extend(store.op(op) for op in store.signature.by_arity(0))

        held.extend(
            store.op(op, child)
            for op in store.signature.by_arity(1)
            for child in by_size[size - 1]
        )
        held.extend(
            store.op(op, left, right)
            for op in store.signature.by_arity(2)
            for left_size in range(1, size - 1)
            for left in by_size[left_size]
            for right in by_size[size - left_size - 1]
        )
        by_size.append(tuple(held))

    return tuple(term for terms_at_size in by_size for term in terms_at_size)


def completion_symbols_by_prefix(
    encodings: set[tuple[int, ...]],
) -> dict[tuple[int, ...], frozenset[int]]:
    """Return the next symbols observed after every reachable prefix.

    Parameters
    ----------
    encodings
        The full-width encodings to group.
    """
    grouped: dict[tuple[int, ...], set[int]] = {}

    for symbols in encodings:
        for slot, symbol in enumerate(symbols):
            grouped.setdefault(symbols[:slot], set()).add(symbol)

    return {prefix: frozenset(symbols) for prefix, symbols in grouped.items()}


def complete_layout_fixture(
    build: Callable[[], Signature], order: Literal["df", "bf"]
) -> LayoutFixture[Term, TermStore, OperatorIndex]:
    """Return a fixture for one complete-tree order.

    Parameters
    ----------
    build
        The signature factory.
    order
        The depth-first or breadth-first slot order.
    """
    store = TermStore(build())
    depth = 3
    generated = terms(store, max_leaves=8)
    corpus = st.one_of(generated, st.just(deep_term(store, depth)))

    return LayoutFixture(
        layout=CompleteTree(
            store.signature, store.signature.operators, order=order, depth=depth
        ),
        terms=corpus,
        store=store,
        index=OperatorIndex(store.signature),
        size=lambda term: term.size,
        fits=lambda term: term_depth(term) <= depth,
        max_examples=100,
    )


def test_a_codec_retains_its_index_and_layout_and_caches_their_metadata() -> None:
    signature = categories()
    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, depth=3)
    codec = TermCodec(index, layout)

    assert codec.index is index
    assert codec.layout is layout
    assert codec.n_symbols == index.width
    assert codec.capacity == layout.capacity
    assert codec.skeleton is layout.skeleton
    assert codec.skeleton is codec.skeleton
    assert codec.legal_symbols is codec.legal_symbols


@pytest.mark.parametrize("order", ["df", "bf"])
def test_a_complete_tree_codec_exposes_fixed_structure_and_static_masks(
    order: Literal["df", "bf"],
) -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = CompleteTree(
        store.signature, store.signature.operators, order=order, depth=3
    )
    codec = TermCodec(index, layout)
    skeleton = codec.skeleton

    assert skeleton is not None
    assert skeleton.symbols == (OperatorIndex.PAD,) * codec.capacity
    assert skeleton.occupied == (False,) * codec.capacity
    assert skeleton.payloads is None
    assert skeleton.parents == layout.encode(transitive(store), index).parents
    assert skeleton.depths == layout.encode(transitive(store), index).depths
    assert len(codec.legal_symbols) == codec.capacity
    assert all(len(row) == codec.n_symbols for row in codec.legal_symbols)
    assert not codec.legal_symbols[0][OperatorIndex.PAD]
    assert all(row[OperatorIndex.PAD] for row in codec.legal_symbols[1:])
    assert not any(
        codec.legal_symbols[slot][index.index("fwd")]
        for slot, depth in enumerate(skeleton.depths)
        if depth == 2
    )


def test_a_preorder_codec_has_term_specific_structure_and_static_masks() -> None:
    index = OperatorIndex(categories())
    codec = TermCodec(index, PreorderArityLayout(CAPACITY))

    assert codec.skeleton is None
    assert len(codec.legal_symbols) == CAPACITY
    assert not codec.legal_symbols[0][OperatorIndex.PAD]
    assert all(row[OperatorIndex.PAD] for row in codec.legal_symbols[1:])
    assert not codec.legal_symbols[-1][index.index("fwd")]
    assert codec.legal_symbols[-1][index.index("s")]


@pytest.mark.parametrize("kind", ["complete", "preorder"])
def test_static_masks_follow_the_owner_s_code_order(kind: str) -> None:
    signature = Signature(
        "mixed-arities",
        ("Cat",),
        (
            Operator("binary", "Cat", arg_sorts=("Cat", "Cat"), family="shared"),
            Operator("atom", "Cat", family="shared"),
            Operator("unary", "Cat", arg_sorts=("Cat",), family="shared"),
        ),
    )
    index = OperatorIndex(signature)
    layout: CompleteTree | PreorderArityLayout = (
        CompleteTree(signature, signature.operators, depth=2)
        if kind == "complete"
        else PreorderArityLayout(4)
    )
    rows = layout.static_symbols(index)

    assert tuple(op.name for op in signature.operators) != (
        "binary",
        "atom",
        "unary",
    )

    for slot, row in enumerate(rows):
        assert len(row) == index.width
        assert row[index.PAD] == (slot != 0)

        if isinstance(layout, CompleteTree):
            heap_slot = heap_order(layout.branching, layout.depth, layout.order)[slot]
            expected = tuple(
                op.arity == 0
                or (
                    op.arity <= layout.branching
                    and layout.branching * heap_slot + op.arity < layout.capacity
                )
                for op in signature.operators
            )
        else:
            room = layout.capacity - slot - 1
            expected = tuple(op.arity <= room for op in signature.operators)

        assert tuple(row[index.index(op)] for op in signature.operators) == expected

    foreign = Operator("atom", "Cat", family="foreign")
    with pytest.raises(UnknownOperatorError):
        index.index(foreign)


@pytest.mark.parametrize(
    "kind",
    ["complete-df", "complete-bf", "preorder"],
)
def test_path_and_slot_maps_are_inverse_for_every_current_layout(
    kind: str,
) -> None:
    signature = categories()
    store = TermStore(signature)
    layout: CompleteTree | PreorderArityLayout = (
        PreorderArityLayout(CAPACITY)
        if kind == "preorder"
        else CompleteTree(
            signature,
            signature.operators,
            order="df" if kind == "complete-df" else "bf",
            depth=3,
        )
    )
    codec = TermCodec(OperatorIndex(store.signature), layout)
    term = transitive(store)
    path_slots = codec.path_to_slot(term)
    slot_paths = codec.slot_to_path(term)

    assert tuple(sorted(path_slots)) == ((), (0,), (0, 0), (0, 1), (1,))
    assert len(path_slots) == len(slot_paths) == term.size
    assert all(slot_paths[slot] == path for path, slot in path_slots.items())

    with pytest.raises(TypeError):
        cast("dict[tuple[int, ...], int]", path_slots)[()] = 1

    with pytest.raises(TypeError):
        cast("dict[int, tuple[int, ...]]", slot_paths)[0] = ()


def test_the_tree_mask_matches_the_transitive_ancestor_rows() -> None:
    blocked = float("-inf")
    parents = (Encoding.NO_PARENT, 0, 1, 1, 0, 4, 4)
    expected = (
        (0.0, blocked, blocked, blocked, blocked, blocked, blocked),
        (0.0, 0.0, blocked, blocked, blocked, blocked, blocked),
        (0.0, 0.0, 0.0, blocked, blocked, blocked, blocked),
        (0.0, 0.0, blocked, 0.0, blocked, blocked, blocked),
        (0.0, blocked, blocked, blocked, 0.0, blocked, blocked),
        (0.0, blocked, blocked, blocked, 0.0, 0.0, blocked),
        (0.0, blocked, blocked, blocked, 0.0, blocked, 0.0),
    )
    codec = TermCodec(OperatorIndex(categories()), PreorderArityLayout(7))
    mask = codec.tree_mask(parents)

    assert mask == expected
    assert type(mask) is tuple
    assert all(type(row) is tuple for row in mask)


@settings(max_examples=1000, deadline=None)
@given(terms_to_depth(CODEC_STORE, 3))
def test_complete_codec_static_masks_admit_every_encoded_symbol(
    term: Term,
) -> None:
    index = OperatorIndex(CODEC_STORE.signature)
    codec = TermCodec(
        index,
        CompleteTree(
            CODEC_STORE.signature,
            CODEC_STORE.signature.operators,
            depth=3,
        ),
    )
    encoding = codec.encode(term)

    assert all(
        codec.legal_symbols[slot][code] for slot, code in enumerate(encoding.symbols)
    )


@settings(max_examples=1000, deadline=None)
@given(terms(CODEC_STORE, max_leaves=8))
def test_preorder_codec_static_masks_admit_every_encoded_symbol(
    term: Term,
) -> None:
    codec = TermCodec(
        OperatorIndex(CODEC_STORE.signature), PreorderArityLayout(CAPACITY)
    )

    if term.size > codec.capacity:
        return

    encoding = codec.encode(term)

    assert all(
        codec.legal_symbols[slot][code] for slot, code in enumerate(encoding.symbols)
    )


def test_a_codec_batch_preserves_every_single_encoding_channel() -> None:
    store = TermStore(formulas())
    index = OperatorIndex(store.signature)
    codec = TermCodec(index, PreorderArityLayout(CAPACITY))
    signature = store.signature
    atom = store.fvar("Lf", "x")
    pred = store.op(signature.constant("Lf"), payload=3)
    applied = store.op(signature.by_name("app"), pred, atom)
    expected = (codec.encode(atom), codec.encode(applied))
    batch = codec.encode_batch((atom, applied))

    assert batch.rows == expected
    assert batch.symbols == tuple(row.symbols for row in expected)
    assert batch.parents == tuple(row.parents for row in expected)
    assert batch.depths == tuple(row.depths for row in expected)
    assert batch.mask == tuple(row.occupied for row in expected)
    assert batch.payloads == tuple(row.payloads for row in expected)
    assert batch.lengths == (atom.size, applied.size)
    assert tuple(codec.decode(row, store) for row in batch.rows) == (atom, applied)


def test_features_and_payloads_use_independent_codec_channels() -> None:
    category_store = TermStore(categories())
    category_index = OperatorIndex(category_store.signature)
    category_codec = TermCodec(category_index, PreorderArityLayout(CAPACITY))
    featured = category_store.op(category_store.signature.featured("s", "dcl"))
    category_encoding = category_codec.encode(featured)
    formula_store = TermStore(formulas())
    formula_index = OperatorIndex(formula_store.signature)
    formula_codec = TermCodec(formula_index, PreorderArityLayout(CAPACITY))
    payload = formula_store.op(formula_store.signature.bound("Lf"), payload=5)
    formula_encoding = formula_codec.encode(payload)

    assert category_store.signature.features
    assert not category_store.signature.has_payloads
    assert category_encoding.symbols[0] == category_index.index(featured.op)
    assert category_encoding.payloads is None
    assert category_codec.decode(category_encoding, category_store) is featured
    assert not formula_store.signature.features
    assert formula_store.signature.has_payloads
    assert formula_encoding.payloads is not None
    assert formula_encoding.payloads[0] == 5
    assert formula_codec.decode(formula_encoding, formula_store) is payload


def test_the_open_ccgbank_declaration_round_trips_several_hundred_categories() -> None:
    store = TermStore(open_ccgbank_signature())
    inventory = open_ccgbank_categories(store)
    index = OperatorIndex(store.signature)
    codecs = (
        TermCodec(
            index,
            CompleteTree(store.signature, store.signature.operators, depth=2),
        ),
        TermCodec(index, PreorderArityLayout(3)),
    )

    assert len(inventory) == 2485
    assert store.signature.features == dict(OPEN_CCGBANK_FEATURES)
    assert any(term.op.erases_to is not None for term in inventory)

    for codec in codecs:
        encodings = codec.encode_batch(inventory)

        assert encodings.lengths.count(1) == 35
        assert encodings.lengths.count(3) == 2450
        assert encodings.payloads is None

        for term, encoding in zip(inventory, encodings.rows, strict=True):
            assert encoding.width == codec.capacity
            assert encoding.length == term.size
            assert len(encoding.parents) == codec.capacity
            assert len(encoding.depths) == codec.capacity
            assert codec.decode(encoding, store) is term


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
@pytest.mark.parametrize("order", ["df", "bf"])
@pytest.mark.parametrize("law", LAYOUT_LAWS, ids=[law.__name__ for law in LAYOUT_LAWS])
def test_every_layout_law_holds_of_a_complete_tree(
    law: Callable[[LayoutFixture[Term, TermStore, OperatorIndex]], None],
    build: Callable[[], Signature],
    order: Literal["df", "bf"],
) -> None:
    law(complete_layout_fixture(build, order))


def test_complete_tree_capacity_is_the_k_ary_geometric_sum() -> None:
    binary = categories()
    ternary = formulas()
    quaternary = Signature(
        "quaternary",
        ("Lf",),
        (
            Operator("atom", "Lf"),
            Operator("four", "Lf", arg_sorts=("Lf",) * 4),
        ),
    )

    assert CompleteTree(binary, binary.operators, depth=1).capacity == 1
    assert CompleteTree(binary, binary.operators, depth=3).capacity == 7
    assert CompleteTree(ternary, ternary.operators, depth=3).capacity == THIRTEEN
    assert CompleteTree(quaternary, quaternary.operators, depth=2).capacity == 5


def test_complete_tree_computes_branching_from_reached_operator_arities() -> None:
    signature = formulas()
    reached = tuple(op for op in signature.operators if op.arity < 3)
    layout = CompleteTree(signature, reached, depth=5)

    assert layout.branching == 2
    assert layout.capacity == 31
    assert layout.diagnostics == (UnreachedArityDiagnostic("ite", 3),)
    assert layout.diagnostics[0].message == "operator 'ite' declares unreached arity 3"

    repeated = CompleteTree(signature, (*reached, *reached), depth=5)

    assert repeated.branching == layout.branching
    assert repeated.diagnostics == layout.diagnostics


@pytest.mark.parametrize(
    "foreign",
    [
        Operator("foreign", "Lf", arg_sorts=("Lf",) * 4),
        Operator(
            "ite",
            "Lf",
            arg_sorts=("Lf", "Lf", "Lf"),
            family="connective",
        ),
    ],
    ids=["undeclared name", "structural twin"],
)
def test_complete_tree_refuses_a_reached_operator_not_owned_by_the_signature(
    foreign: Operator,
) -> None:
    signature = formulas()

    with pytest.raises(UnknownOperatorError) as caught:
        CompleteTree(signature, (*signature.operators, foreign), depth=2)

    assert caught.value.operator == foreign.name


def test_unreached_arity_diagnostics_name_every_affected_operator() -> None:
    signature = Signature(
        "two-ternaries",
        ("Lf",),
        (
            Operator("atom", "Lf"),
            Operator("pair", "Lf", arg_sorts=("Lf", "Lf")),
            Operator("choose", "Lf", arg_sorts=("Lf",) * 3),
            Operator("conjoin", "Lf", arg_sorts=("Lf",) * 3),
        ),
    )
    reached = tuple(op for op in signature.operators if op.arity < 3)
    layout = CompleteTree(signature, reached, depth=2)

    assert layout.diagnostics == (
        UnreachedArityDiagnostic("choose", 3),
        UnreachedArityDiagnostic("conjoin", 3),
    )

    one_ternary = (*reached, signature.by_name("choose"))
    generalized = CompleteTree(signature, one_ternary, depth=2)

    assert generalized.branching == 3
    assert generalized.diagnostics == ()


@pytest.mark.parametrize(
    ("depth", "capacity", "symbols"),
    [
        (2, 3, "040102"),
        (3, 7, "04010000020000"),
        (4, 15, "040100000000000002000000000000"),
        (5, 31, "04010000000000000000000000000000020000000000000000000000000000"),
        (
            6,
            63,
            (
                "0401000000000000000000000000000000000000000000000000000000000000"
                "02000000000000000000000000000000000000000000000000000000000000"
            ),
        ),
    ],
)
def test_computed_binary_layout_has_exact_encoded_bytes_at_configured_depths(
    depth: int, capacity: int, symbols: str
) -> None:
    signature = Signature(
        "binary-row",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("np", "Cat"),
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    index = OperatorIndex(signature)
    term = store.op(
        signature.by_name("bwd"),
        store.op(signature.by_name("s")),
        store.op(signature.by_name("np")),
    )
    written = CompleteTree(
        signature, signature.operators, order="df", depth=depth
    ).encode(term, index)

    assert written.width == capacity
    assert bytes(written.symbols).hex() == symbols


def test_complete_tree_exposes_its_fixed_layout_parameters() -> None:
    signature = formulas()
    layout = CompleteTree(signature, signature.operators, order="bf", depth=4)

    assert layout.name == "complete-tree"
    assert layout.order == "bf"
    assert layout.branching == 3
    assert layout.depth == 4
    assert layout.path_addressed
    assert repr(layout) == "CompleteTree(order='bf', branching=3, depth=4)"


def test_complete_tree_parameters_have_frozen_value_semantics() -> None:
    signature = formulas()
    layout = CompleteTree(signature, signature.operators, order="bf", depth=4)
    same = CompleteTree(signature, signature.operators, order="bf", depth=4)

    assert layout == same
    assert hash(layout) == hash(same)
    assert layout != CompleteTree(signature, signature.operators, order="df", depth=4)
    assert layout != ("bf", 3, 4)

    with pytest.raises(AttributeError, match="immutable"):
        layout.__setattr__("_order", "df")


@pytest.mark.parametrize(
    "build",
    [
        lambda: CompleteTree(categories(), categories().operators, depth=0),
        lambda: CompleteTree(
            categories(),
            categories().operators,
            order=cast("Literal['df', 'bf']", "sideways"),
            depth=2,
        ),
    ],
)
def test_complete_tree_refuses_invalid_shape_parameters(
    build: Callable[[], CompleteTree],
) -> None:
    with pytest.raises(ValueError, match=r"depth|order"):
        build()


@pytest.mark.parametrize(
    "build",
    [
        lambda: CompleteTree(
            categories(), categories().operators, depth=cast("int", 2.0)
        ),
    ],
)
def test_complete_tree_refuses_noninteger_shape_parameters(
    build: Callable[[], CompleteTree],
) -> None:
    with pytest.raises(TypeError):
        build()


def test_the_single_heap_order_has_depth_first_addresses() -> None:
    assert heap_order(2, 3, "df") == (0, 1, 3, 4, 2, 5, 6)
    assert heap_order(2, 3, "bf") == tuple(range(7))


def test_only_the_codec_module_defines_heap_order() -> None:
    package = Path(__file__).resolve().parents[3] / "lofi" / "terms"
    owners = []

    for module in package.glob("*.py"):
        tree = ast.parse(module.read_text(encoding="utf-8"))

        if any(
            isinstance(node, ast.FunctionDef) and node.name == "heap_order"
            for node in ast.walk(tree)
        ):
            owners.append(module.name)

    assert owners == ["_codec.py"]


def test_the_depth_first_row_has_exact_symbols_and_parents() -> None:
    signature = Signature(
        "depth-first-row",
        ("Cat",),
        (
            Operator("s", "Cat"),
            Operator("np", "Cat"),
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        ),
    )
    store = TermStore(signature)
    index = OperatorIndex(signature)
    term = store.op(
        signature.by_name("bwd"),
        store.op(signature.by_name("s")),
        store.op(signature.by_name("np")),
    )
    written = CompleteTree(signature, signature.operators, order="df", depth=3).encode(
        term, index
    )

    assert written.symbols == (4, 1, 0, 0, 2, 0, 0)
    assert written.parents == (-1, 0, 1, 1, 0, 4, 4)
    assert written.depths == (0, 1, 2, 2, 1, 2, 2)
    assert written.occupied == (True, True, False, False, True, False, False)


def test_a_ternary_operator_uses_each_declared_child_position() -> None:
    store = TermStore(formulas())
    signature = store.signature

    for name in ("x0", "x1", "x2"):
        store.atom("Lf", name)

    index = OperatorIndex(signature)
    layout = CompleteTree(signature, signature.operators, depth=2)
    children = (
        store.op(signature.bound("Lf"), payload=1),
        store.fvar("Lf", "x2"),
        store.op(signature.constant("Lf"), payload=3),
    )
    term = store.op(signature.by_name("ite"), *children)
    written = layout.encode(term, index)

    assert written.symbols == (
        index.index("ite"),
        *(index.index(t.op) for t in children),
    )
    assert written.parents == (-1, 0, 0, 0)
    assert written.occupied == (True, True, True, True)
    assert written.payloads == (0, 1, 2, 3)
    assert layout.decode(written, store, index) is term


def test_a_constructor_wider_than_the_tree_is_refused_whole() -> None:
    store = TermStore(formulas())
    signature = store.signature
    index = OperatorIndex(signature)
    leaf = store.op(signature.bound("Lf"), payload=0)
    term = store.op(signature.by_name("ite"), leaf, leaf, leaf)
    reachable = tuple(op for op in signature.operators if op.arity < 3)
    layout = CompleteTree(signature, reachable, depth=3)

    with pytest.raises(EncodingOverflowError) as caught:
        layout.encode(term, index)

    assert (caught.value.capacity, caught.value.size) == (7, term.size)


def test_complete_tree_decode_refuses_a_missing_child_and_a_trailing_node() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = CompleteTree(store.signature, store.signature.operators, depth=3)
    branch = layout.encode(forward(store), index)
    symbols = list(branch.symbols)
    symbols[4] = OperatorIndex.PAD
    missing = branch._replace(symbols=tuple(symbols))

    with pytest.raises(EncodingShapeError) as exhausted:
        layout.decode(missing, store, index)

    alone = layout.encode(sentence(store), index)
    symbols = list(alone.symbols)
    symbols[1] = index.index("np")
    trailing = alone._replace(symbols=tuple(symbols))

    with pytest.raises(EncodingShapeError) as extra:
        layout.decode(trailing, store, index)

    assert (exhausted.value.reason, exhausted.value.slot) == ("exhausted", 4)
    assert (extra.value.reason, extra.value.slot) == ("trailing", 1)


AGREEMENT_DEPTH: Final = 4
AGREEMENT_BRANCHING: Final = 3
AGREEMENT_STORE: Final = TermStore(formulas())
AGREEMENT_TERMS: Final = terms_to_depth(AGREEMENT_STORE, AGREEMENT_DEPTH)


@settings(max_examples=3000, deadline=None)
@given(AGREEMENT_TERMS)
def test_breadth_first_is_one_permutation_of_depth_first_over_thousands(
    term: Term,
) -> None:
    index = OperatorIndex(AGREEMENT_STORE.signature)
    depth_first = CompleteTree(
        AGREEMENT_STORE.signature,
        AGREEMENT_STORE.signature.operators,
        order="df",
        depth=AGREEMENT_DEPTH,
    )
    breadth_first = CompleteTree(
        AGREEMENT_STORE.signature,
        AGREEMENT_STORE.signature.operators,
        order="bf",
        depth=AGREEMENT_DEPTH,
    )
    df = depth_first.encode(term, index)
    bf = breadth_first.encode(term, index)
    df_heap = heap_order(AGREEMENT_BRANCHING, AGREEMENT_DEPTH, "df")
    bf_heap = heap_order(AGREEMENT_BRANCHING, AGREEMENT_DEPTH, "bf")
    df_positions = {heap: slot for slot, heap in enumerate(df_heap)}
    permutation = tuple(df_positions[heap] for heap in bf_heap)
    inverse = [0] * len(permutation)

    for target, source in enumerate(permutation):
        inverse[source] = target

    assert bf.symbols == tuple(df.symbols[slot] for slot in permutation)
    assert bf.depths == tuple(df.depths[slot] for slot in permutation)
    assert bf.occupied == tuple(df.occupied[slot] for slot in permutation)
    assert bf.parents == tuple(
        Encoding.NO_PARENT
        if df.parents[slot] == Encoding.NO_PARENT
        else inverse[df.parents[slot]]
        for slot in permutation
    )
    assert bf.payloads == (
        None
        if df.payloads is None
        else tuple(df.payloads[slot] for slot in permutation)
    )
    assert depth_first.decode(df, AGREEMENT_STORE, index) is term
    assert breadth_first.decode(bf, AGREEMENT_STORE, index) is term


@settings(max_examples=250, deadline=None)
@given(AGREEMENT_TERMS)
def test_tree_masks_agree_across_layouts_under_the_slot_correspondence(
    term: Term,
) -> None:
    index = OperatorIndex(AGREEMENT_STORE.signature)
    complete_codec = TermCodec(
        index,
        CompleteTree(
            AGREEMENT_STORE.signature,
            AGREEMENT_STORE.signature.operators,
            order="df",
            depth=AGREEMENT_DEPTH,
        ),
    )
    preorder_codec = TermCodec(index, PreorderArityLayout(complete_codec.capacity))
    complete = complete_codec.encode(term)
    preorder = preorder_codec.encode(term)
    complete_slots = tuple(
        slot for slot, occupied in enumerate(complete.occupied) if occupied
    )
    preorder_slots = tuple(
        slot for slot, occupied in enumerate(preorder.occupied) if occupied
    )
    correspondence = tuple(zip(complete_slots, preorder_slots, strict=True))
    complete_mask = complete_codec.tree_mask(complete.parents)
    preorder_mask = preorder_codec.tree_mask(preorder.parents)
    aligned_complete = tuple(
        tuple(complete_mask[row][column] for column, _ in correspondence)
        for row, _ in correspondence
    )
    aligned_preorder = tuple(
        tuple(preorder_mask[row][column] for _, column in correspondence)
        for _, row in correspondence
    )

    assert len(correspondence) == term.size
    assert tuple(complete.symbols[slot] for slot in complete_slots) == tuple(
        preorder.symbols[slot] for slot in preorder_slots
    )
    assert tuple(complete.depths[slot] for slot in complete_slots) == tuple(
        preorder.depths[slot] for slot in preorder_slots
    )
    assert aligned_complete == aligned_preorder


@pytest.mark.parametrize("order", ["df", "bf"])
def test_a_nullary_symbol_is_legal_in_the_deepest_slot(
    order: Literal["df", "bf"],
) -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = CompleteTree(
        store.signature, store.signature.operators, order=order, depth=3
    )
    full = layout.encode(cat(store, "fwd", forward(store), forward(store)), index)
    prefix = full.symbols[:-1]
    allowed = layout.legal_next(prefix, index)

    assert full.occupied[-1]
    assert index.is_nullary(full.symbols[-1])
    assert allowed[full.symbols[-1]]
    assert not allowed[index.index("fwd")]
    assert layout.decode(full, store, index) is cat(
        store, "fwd", forward(store), forward(store)
    )


@pytest.mark.parametrize(
    ("kind", "expected_terms"),
    [("df", 13), ("bf", 13), ("preorder", 89)],
    ids=("complete depth first", "complete breadth first", "preorder"),
)
def test_legal_next_is_exactly_the_set_of_completable_symbols(
    kind: Literal["df", "bf", "preorder"], expected_terms: int
) -> None:
    signature = exact_completion_signature()
    store = TermStore(signature)
    index = OperatorIndex(signature)

    if kind == "preorder":
        layout: CompleteTree | PreorderArityLayout = PreorderArityLayout(CAPACITY)
        fitting = terms_with_maximum_size(store, layout.capacity)
    else:
        layout = CompleteTree(
            signature,
            signature.operators,
            order=kind,
            depth=3,
        )
        fitting = terms_with_maximum_depth(store, layout.depth)

    full_width = {layout.encode(term, index).symbols for term in fitting}
    expected_by_prefix = completion_symbols_by_prefix(full_width)

    assert len(fitting) == expected_terms
    assert len(full_width) == expected_terms
    assert {len(prefix) for prefix in expected_by_prefix} == set(range(layout.capacity))

    for prefix, expected in expected_by_prefix.items():
        mask = layout.legal_next(prefix, index)
        actual = frozenset(code for code, legal in enumerate(mask) if legal)

        assert actual, prefix
        assert actual == expected, prefix


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
@pytest.mark.parametrize("law", LAYOUT_LAWS, ids=[law.__name__ for law in LAYOUT_LAWS])
def test_every_layout_law_holds_of_the_preorder_layout(
    law: Callable[[LayoutFixture[Term, TermStore, OperatorIndex]], None],
    build: Callable[[], Signature],
) -> None:
    law(layout_fixture(build))


@pytest.mark.conformance
@pytest.mark.parametrize("build", BUILDERS, ids=NAMES)
def test_running_the_layout_family_in_one_call_runs_it_against_the_layout(
    build: Callable[[], Signature],
) -> None:
    check_layout_laws(layout_fixture(build))


def test_the_round_trip_catches_a_store_that_decodes_into_a_second_node() -> None:
    store = Unshared(categories())

    with pytest.raises(LawViolationError, match="round trip"):
        round_trip_is_identity(layout_fixture(categories, store))


@pytest.mark.parametrize(
    ("written", "spells"),
    [
        (("s",), sentence),
        (("fwd", "s", "np"), forward),
        (("fwd", "bwd", "s", "np", "np"), transitive),
    ],
    ids=["a nullary", "one branching", "two branching"],
)
def test_the_symbols_a_term_spells_decode_to_that_term(
    written: tuple[str, ...],
    spells: Callable[[TermStore], Term],
) -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    expected = spells(store)

    assert layout.decode(sequence(written, index), store, index) is expected
    assert layout.encode(expected, index).symbols == sequence(written, index).symbols


def test_a_featured_term_is_written_out_rather_than_its_erased_core() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    dcl = store.signature.featured("s", "dcl")
    over = store.op(store.signature.by_name("fwd"), store.op(dcl), cat(store, "np"))
    written = layout.encode(over, index)

    assert over.core is not over
    assert written.symbols[1] == index.index(dcl)
    assert written.symbols != layout.encode(over.core, index).symbols
    assert layout.decode(written, store, index) is over


@pytest.mark.parametrize(
    ("written", "reason", "slot"),
    [
        ((), "exhausted", 0),
        (("fwd", "s"), "exhausted", 2),
        (("s", "np"), "trailing", 1),
    ],
    ids=["the empty sequence", "an argument still owed", "a trailing token"],
)
def test_a_sequence_that_is_not_one_term_is_refused(
    written: tuple[str, ...], reason: str, slot: int
) -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)

    with pytest.raises(EncodingShapeError) as caught:
        layout.decode(sequence(written, index), store, index)

    assert caught.value.reason == reason
    assert caught.value.slot == slot
    assert caught.value.layout == layout.name


def test_a_trailing_token_is_refused_rather_than_read_as_a_prefix() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    alone = sequence(("s",), index)

    assert layout.decode(alone, store, index) is sentence(store)

    with pytest.raises(EncodingShapeError):
        layout.decode(sequence(("s", "np"), index), store, index)


def test_a_full_sequence_still_owing_a_subterm_is_refused() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(2)
    codes = (index.index("fwd"), index.index("fwd"))
    full = Encoding(codes, (Encoding.NO_PARENT, 0), (0, 1), (True, True))

    with pytest.raises(EncodingShapeError) as caught:
        layout.decode(full, store, index)

    assert (caught.value.reason, caught.value.slot) == ("exhausted", full.width)


def test_the_capacity_is_the_node_count_it_was_declared_with() -> None:
    assert PreorderArityLayout(THIRTEEN).capacity == THIRTEEN
    assert PreorderArityLayout(0).capacity == 0
    assert repr(PreorderArityLayout(THIRTEEN)) == f"PreorderArityLayout({THIRTEEN})"
    assert PreorderArityLayout(THIRTEEN).name == "preorder-arity"


def test_a_node_count_that_is_not_a_count_is_refused() -> None:
    with pytest.raises(ValueError, match="max_nodes"):
        PreorderArityLayout(-1)

    with pytest.raises(TypeError, match="max_nodes"):
        PreorderArityLayout(cast("int", 2.0))


def test_a_thirteen_node_term_takes_thirteen_slots_however_deep_it_is() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(THIRTEEN)
    deep = spine(store, 6)
    fwd = store.signature.by_name("fwd")
    inner = forward(store)
    bushy = store.op(
        fwd,
        store.op(fwd, inner, inner),
        store.op(fwd, inner, store.op(store.signature.by_name("s"))),
    )
    written = [layout.encode(each, index) for each in (deep, bushy)]

    assert deep.size == THIRTEEN
    assert bushy.size == THIRTEEN
    assert [enc.width for enc in written] == [THIRTEEN, THIRTEEN]
    assert [enc.length for enc in written] == [THIRTEEN, THIRTEEN]
    assert max(written[0].depths) != max(written[1].depths)


def test_a_term_of_more_nodes_than_slots_is_refused_whole() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    over = spine(store, 4)

    with pytest.raises(EncodingOverflowError) as caught:
        layout.encode(over, index)

    assert (caught.value.capacity, caught.value.size) == (CAPACITY, over.size)


def test_no_slot_of_the_layout_denotes_a_path() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    shallow = layout.encode(forward(store), index)
    deeper = layout.encode(transitive(store), index)

    assert not layout.path_addressed
    assert shallow.depths[2] != deeper.depths[2]


def test_the_payload_channel_is_there_for_a_signature_that_declares_one() -> None:
    store = TermStore(formulas())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    signature = store.signature
    body = store.op(
        signature.by_name("app"),
        store.op(signature.bound("Lf"), payload=1),
        store.op(signature.constant("Lf"), payload=2),
    )
    written = layout.encode(store.op(signature.by_name("lam"), body), index)

    assert written.payloads is not None
    assert written.payloads[2:4] == (1, 2)
    assert layout.decode(written, store, index) is store.op(
        signature.by_name("lam"), body
    )


def test_the_payload_channel_is_absent_where_no_operator_takes_a_payload() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    written = PreorderArityLayout(CAPACITY).encode(spine(store, 1), index)

    assert not store.signature.has_payloads
    assert written.payloads is None


def test_a_partial_term_with_a_hole_round_trips_through_the_codec() -> None:
    signature = categories().with_holes()
    store = TermStore(signature)
    index = OperatorIndex(signature)
    codec = TermCodec(index, PreorderArityLayout(3))
    skeleton = store.op(
        signature.by_name("fwd"),
        store.op(signature.by_name("s")),
        store.hole("Cat"),
    )
    encoded = codec.encode(skeleton)

    assert index.is_nullary(index.index("#hole[Cat]"))
    assert codec.decode(encoded, store) is skeleton
    assert encoded.symbols == (
        index.index("fwd"),
        index.index("s"),
        index.index("#hole[Cat]"),
    )


def test_two_terms_differing_only_in_a_payload_do_not_decode_to_one_term() -> None:
    store = TermStore(formulas())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    var = store.signature.bound("Lf")
    here = store.op(var, payload=0)
    there = store.op(var, payload=1)

    assert layout.encode(here, index).symbols == layout.encode(there, index).symbols
    assert layout.decode(layout.encode(here, index), store, index) is here
    assert layout.decode(layout.encode(there, index), store, index) is there


def test_only_padding_may_follow_a_complete_term() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(CAPACITY)
    allowed = layout.legal_next((index.index("s"),), index)

    assert allowed[OperatorIndex.PAD]
    assert sum(allowed) == 1


def test_a_symbol_owing_more_subterms_than_the_slots_left_is_refused() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(2)
    allowed = layout.legal_next((), index)

    assert not any(allowed[index.index(op)] for op in index.signature.by_arity(2))
    assert all(allowed[index.index(op)] for op in index.signature.by_arity(0))


def test_a_prefix_that_padded_before_its_term_was_whole_admits_only_padding() -> None:
    index = OperatorIndex(categories())
    layout = PreorderArityLayout(CAPACITY)
    allowed = layout.legal_next((index.index("fwd"), OperatorIndex.PAD), index)

    assert allowed[OperatorIndex.PAD]
    assert sum(allowed) == 1


def test_nothing_follows_a_linearization_that_fills_every_slot() -> None:
    store = TermStore(categories())
    index = OperatorIndex(store.signature)
    layout = PreorderArityLayout(1)
    allowed = layout.legal_next((index.index("s"),), index)

    assert not any(allowed)
    assert len(allowed) == index.width
