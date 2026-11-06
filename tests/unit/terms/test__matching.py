from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.term_reference import categories, formulas
from tests.support.term_strategies import terms

from lofi.terms import (
    Alpha,
    ExactMatching,
    FeatureErasure,
    Identity,
    Matching,
    OperatorIndex,
    PreorderArityLayout,
    Product,
    RootMatching,
    Signature,
    SignatureStableMatching,
    Subsumption,
    Term,
    TermStore,
    close_binder,
    open_binder,
    substitute,
)
from lofi.terms.laws import MATCHING_LAWS, MatchingFixture, check_matching_laws

if TYPE_CHECKING:
    from collections.abc import Callable

EXAMPLES: Final = 25


class Node:
    """A sample operator with its arguments."""

    __slots__ = ("args", "op")

    def __init__(self, op: str, args: tuple[Node, ...]) -> None:
        """Hold an operator and the terms it was applied to."""
        self.op = op
        self.args = args

    @override
    def __repr__(self) -> str:
        """Return the operator, with its arguments where it has any."""
        return f"{self.op}{list(self.args)}" if self.args else self.op


TABLE: Final[dict[tuple[str, tuple[Node, ...]], Node]] = {}
"""Intern table keyed by operator and identity-compared arguments."""


def node(op: str, *args: Node) -> Node:
    slot = (op, args)
    held = TABLE.get(slot)

    if held is None:
        held = TABLE[slot] = Node(op, args)

    return held


NP: Final = node("NP")
S: Final = node("S")
POOL: Final = (NP, S, node("/", S, NP), node("/", NP, NP), node("\\", S, NP))


def under_slash(t: Node) -> Node:
    return node("/", t, NP)


def under_backslash(t: Node) -> Node:
    return node("\\", S, t)


def under_type_change(t: Node) -> Node:
    return node("tc", t)


def fixture(relation: Matching[Node] | None = None) -> MatchingFixture[Node]:
    return MatchingFixture(
        Identity[Node]() if relation is None else relation,
        st.sampled_from(POOL),
        st.sampled_from((under_slash, under_backslash, under_type_change)),
        EXAMPLES,
    )


def test_the_sample_algebra_interns_so_one_node_is_one_object() -> None:
    assert node("/", S, NP) is node("/", S, NP)
    assert node("/", S, NP) is not node("/", NP, S)


@pytest.mark.parametrize(
    "law",
    MATCHING_LAWS,
    ids=[law.__name__ for law in MATCHING_LAWS],
)
def test_identity_holds_to_each_matching_law(
    law: Callable[[MatchingFixture[Node]], None],
) -> None:
    law(fixture())


def test_running_the_family_in_one_call_runs_it_against_identity() -> None:
    check_matching_laws(fixture())


@pytest.mark.parametrize(
    "law",
    MATCHING_LAWS,
    ids=[law.__name__ for law in MATCHING_LAWS],
)
def test_alpha_holds_to_each_matching_law(
    law: Callable[[MatchingFixture[Node]], None],
) -> None:
    law(fixture(Alpha[Node]()))


def test_running_the_family_in_one_call_runs_it_against_alpha() -> None:
    check_matching_laws(fixture(Alpha[Node]()))


def test_identity_is_the_matching_port_at_run_time() -> None:
    assert isinstance(Identity[Node](), Matching)


def test_identity_names_itself_and_offers_representatives() -> None:
    relation = Identity[Node]()

    assert relation.name == "identity"
    assert relation.has_key
    assert isinstance(relation, SignatureStableMatching)
    assert relation.signature_extension_stable
    assert repr(relation) == "Identity()"


def test_each_matching_declares_whether_it_is_identity() -> None:
    assert isinstance(Identity(), ExactMatching)
    assert Identity().exact
    assert Alpha().exact
    assert not FeatureErasure().exact
    assert not Subsumption().exact
    assert Product(FeatureErasure(), Identity()).exact
    assert not Product(FeatureErasure(), Subsumption()).exact


def test_a_term_represents_its_own_class() -> None:
    relation = Identity[Node]()
    term = node("/", S, NP)

    assert relation.key(term) is term


def test_two_equal_values_that_are_two_objects_are_not_related() -> None:
    relation = Identity[tuple[int, ...]]()
    one = tuple(range(3))
    other = tuple(range(3))

    assert one == other
    assert one is not other
    assert not relation.equivalent(one, other)
    assert relation.equivalent(one, one)


class Nominal(NamedTuple):
    """A term with explicit binder names."""

    head: str
    payload: int | None
    arguments: tuple[Nominal, ...]
    binders: tuple[tuple[int, ...], ...]


FORMULA_STORE: Final = TermStore(formulas())
FORMULA_SIGNATURE: Final = FORMULA_STORE.signature
BOUND: Final = FORMULA_SIGNATURE.bound("Lf")
FREE: Final = FORMULA_SIGNATURE.free("Lf")
APPLICATION: Final = FORMULA_SIGNATURE.by_name("app")
BINDER: Final = FORMULA_SIGNATURE.by_name("lam")
PREDICATE: Final = FORMULA_STORE.op(FORMULA_SIGNATURE.constant("Lf"), payload=0)

CATEGORY_STORE: Final = TermStore(categories())
CATEGORY_SIGNATURE: Final = CATEGORY_STORE.signature
CATEGORY_NP: Final = CATEGORY_STORE.op(CATEGORY_SIGNATURE.by_name("np"))
CATEGORY_S: Final = CATEGORY_STORE.op(CATEGORY_SIGNATURE.by_name("s"))
CATEGORY_DCL: Final = CATEGORY_STORE.op(CATEGORY_SIGNATURE.featured("s", "dcl"))
CATEGORY_B: Final = CATEGORY_STORE.op(CATEGORY_SIGNATURE.featured("s", "b"))
CATEGORY_FWD: Final = CATEGORY_SIGNATURE.by_name("fwd")

CATEGORY_TERMS = terms(
    CATEGORY_STORE,
    sort="Cat",
    max_size=8,
    closed=True,
)
FORMULA_TERMS = terms(
    FORMULA_STORE,
    sort="Lf",
    max_size=8,
    closed=True,
)


def category_as_result(term: Term) -> Term:
    """Build a forward functor with the term as its result."""
    return CATEGORY_STORE.op(CATEGORY_FWD, term, CATEGORY_NP)


def category_as_argument(term: Term) -> Term:
    """Build a forward functor with the term as its argument."""
    return CATEGORY_STORE.op(CATEGORY_FWD, CATEGORY_S, term)


def formula_as_function(term: Term) -> Term:
    """Apply the term to a fixed predicate."""
    return FORMULA_STORE.op(APPLICATION, term, PREDICATE)


def formula_as_argument(term: Term) -> Term:
    """Apply a fixed predicate to the term."""
    return FORMULA_STORE.op(APPLICATION, PREDICATE, term)


CATEGORY_CONTEXTS = st.sampled_from((category_as_result, category_as_argument))
FORMULA_CONTEXTS = st.sampled_from((formula_as_function, formula_as_argument))

CATEGORY_MATCHINGS: Final[tuple[Matching[Term], ...]] = (
    Identity(),
    FeatureErasure(),
    Alpha(),
)
FORMULA_MATCHINGS: Final[tuple[Matching[Term], ...]] = (
    Identity(),
    FeatureErasure(),
    Alpha(),
)


@pytest.mark.parametrize(
    "relation",
    CATEGORY_MATCHINGS,
    ids=lambda relation: relation.name,
)
def test_each_matching_holds_over_generated_featureful_terms(
    relation: Matching[Term],
) -> None:
    check_matching_laws(
        MatchingFixture(relation, CATEGORY_TERMS, CATEGORY_CONTEXTS, EXAMPLES * 8)
    )


@pytest.mark.parametrize(
    "relation",
    FORMULA_MATCHINGS,
    ids=lambda relation: relation.name,
)
def test_each_matching_holds_over_generated_feature_free_terms(
    relation: Matching[Term],
) -> None:
    check_matching_laws(
        MatchingFixture(relation, FORMULA_TERMS, FORMULA_CONTEXTS, EXAMPLES * 8)
    )


def test_feature_erasure_uses_the_precomputed_core() -> None:
    relation = FeatureErasure()

    assert isinstance(relation, Matching)
    assert relation.name == "feature-erasure"
    assert relation.has_key
    assert isinstance(relation, SignatureStableMatching)
    assert relation.signature_extension_stable
    assert relation.key(CATEGORY_DCL) is CATEGORY_S
    assert relation.key(CATEGORY_S) is CATEGORY_S
    assert relation.equivalent(CATEGORY_DCL, CATEGORY_B)
    assert not relation.equivalent(CATEGORY_DCL, CATEGORY_NP)
    assert repr(relation) == "FeatureErasure()"


def test_product_holds_over_generated_featureful_terms() -> None:
    relation = Product(
        FeatureErasure(),
        Alpha[Term](),
    )

    check_matching_laws(
        MatchingFixture(
            relation,
            CATEGORY_TERMS,
            st.just(category_as_result),
            EXAMPLES * 8,
        )
    )


def test_product_holds_over_generated_feature_free_terms() -> None:
    relation = Product(
        FeatureErasure(),
        Alpha[Term](),
    )

    check_matching_laws(
        MatchingFixture(
            relation,
            FORMULA_TERMS,
            st.just(formula_as_function),
            EXAMPLES * 8,
        )
    )


def test_product_requires_both_relations_and_compiles_their_intersection() -> None:
    relation = Product(
        FeatureErasure(),
        Alpha[Term](),
    )

    assert isinstance(relation, Matching)
    assert not isinstance(relation, SignatureStableMatching)
    assert relation.name == "product(feature-erasure,alpha)"
    assert relation.has_key
    assert relation.equivalent(CATEGORY_DCL, CATEGORY_DCL)
    assert not relation.equivalent(CATEGORY_DCL, CATEGORY_B)
    assert relation.key(CATEGORY_DCL) is CATEGORY_DCL
    assert relation.key(CATEGORY_B) is CATEGORY_B
    assert repr(relation) == "Product(FeatureErasure(), Alpha())"


def test_subsumption_does_not_claim_signature_stable_representatives() -> None:
    assert not isinstance(Subsumption(), SignatureStableMatching)


def test_subsumption_keys_preserve_repeated_free_variable_identity() -> None:
    relation = Subsumption()
    x = FORMULA_STORE.fvar("Lf", "x")
    y = FORMULA_STORE.fvar("Lf", "y")
    z = FORMULA_STORE.fvar("Lf", "z")
    xx = FORMULA_STORE.op(APPLICATION, x, x)
    yy = FORMULA_STORE.op(APPLICATION, y, y)
    yz = FORMULA_STORE.op(APPLICATION, y, z)

    assert relation.has_key
    assert relation.equivalent(xx, yy)
    assert not relation.equivalent(xx, yz)
    assert relation.key(xx) is relation.key(yy)
    assert relation.key(xx) is not relation.key(yz)


def binder(body: Term) -> Term:
    """Bind a used variable over a generated body."""
    variable = FORMULA_STORE.op(BOUND, payload=0)

    return FORMULA_STORE.op(BINDER, FORMULA_STORE.op(APPLICATION, variable, body))


BINDERS = terms(
    FORMULA_STORE,
    sort="Lf",
    max_size=10,
    closed=True,
).map(binder)


def rename(
    term: Term,
    old: tuple[int, ...],
    new: tuple[int, ...],
) -> Term:
    """Rename free atoms through repeated free substitution."""
    renamed = term

    for source, target in zip(old, new, strict=True):
        replacement = FORMULA_STORE.op(FREE, payload=target)
        renamed = substitute(FORMULA_STORE, renamed, source, replacement)

    return renamed


def nominalize(
    term: Term,
    *,
    first_name: int,
) -> Nominal:
    """Convert a closed term to a separate nominal representation."""
    next_name = first_name

    def convert(node: Term, scope: tuple[int, ...]) -> Nominal:
        nonlocal next_name

        if node.op.payload_sort == "index" and node.payload is not None:
            return Nominal("variable", scope[node.payload], (), ())

        if node.op.payload_sort == "free":
            return Nominal("variable", node.payload, (), ())

        arguments: list[Nominal] = []
        binders: list[tuple[int, ...]] = []

        for position, argument in enumerate(node.args):
            count = len(node.op.scope(position))
            names = tuple(range(next_name, next_name - count, -1))
            next_name -= count
            arguments.append(convert(argument, (*names, *scope)))
            binders.append(names)

        return Nominal(node.op.name, node.payload, tuple(arguments), tuple(binders))

    return convert(term, ())


def nominally_alpha_equivalent(left: Nominal, right: Nominal) -> bool:
    """Compare nominal terms modulo consistent bound-name changes."""

    def equivalent(
        here: Nominal,
        there: Nominal,
        here_scope: tuple[int, ...],
        there_scope: tuple[int, ...],
    ) -> bool:
        if here.head == "variable" or there.head == "variable":
            if here.head != there.head:
                return False

            here_bound = (
                here_scope.index(here.payload) if here.payload in here_scope else None
            )
            there_bound = (
                there_scope.index(there.payload)
                if there.payload in there_scope
                else None
            )

            if here_bound is not None or there_bound is not None:
                return here_bound == there_bound

            return here.payload == there.payload

        if (
            here.head != there.head
            or here.payload != there.payload
            or len(here.arguments) != len(there.arguments)
            or tuple(map(len, here.binders)) != tuple(map(len, there.binders))
        ):
            return False

        return all(
            equivalent(
                here_argument,
                there_argument,
                (*here_names, *here_scope),
                (*there_names, *there_scope),
            )
            for here_argument, there_argument, here_names, there_names in zip(
                here.arguments,
                there.arguments,
                here.binders,
                there.binders,
                strict=True,
            )
        )

    return equivalent(left, right, (), ())


def test_alpha_names_identity_matching() -> None:
    relation = Alpha[Node]()

    assert isinstance(relation, Identity)
    assert isinstance(relation, Matching)
    assert relation.name == "alpha"
    assert relation.has_key
    assert repr(relation) == "Alpha()"
    assert Alpha.key is Identity.key
    assert Alpha.equivalent is Identity.equivalent
    assert relation.key(NP) is NP
    assert relation.equivalent(NP, NP)
    assert not relation.equivalent(NP, S)


def test_the_nominal_oracle_distinguishes_bound_and_free_renaming() -> None:
    bound_here = Nominal(
        "lam",
        None,
        (Nominal("variable", -1, (), ()),),
        ((-1,),),
    )
    bound_there = Nominal(
        "lam",
        None,
        (Nominal("variable", -7, (), ()),),
        ((-7,),),
    )
    free_there = Nominal(
        "lam",
        None,
        (Nominal("variable", 7, (), ()),),
        ((-7,),),
    )

    assert nominally_alpha_equivalent(bound_here, bound_there)
    assert not nominally_alpha_equivalent(bound_here, free_there)


@settings(max_examples=3_000, deadline=None)
@given(BINDERS)
def test_nominal_alpha_variants_are_one_node_and_one_encoding(
    term: Term,
) -> None:
    opened, atoms = open_binder(FORMULA_STORE, term)
    fresh = FORMULA_STORE.fresh("Lf", opened.free, len(atoms))
    variant = rename(opened, atoms, fresh)
    restored = close_binder(FORMULA_STORE, variant, fresh)
    relation = Alpha[Term]()
    layout = PreorderArityLayout(term.size)
    index = OperatorIndex(FORMULA_SIGNATURE)

    assert atoms != fresh
    assert layout.encode(opened, index) != layout.encode(variant, index)
    assert nominally_alpha_equivalent(
        nominalize(term, first_name=-1),
        nominalize(restored, first_name=-10_000),
    )
    assert restored is term
    assert relation.equivalent(restored, term)
    assert relation.key(restored) is relation.key(term)
    assert layout.encode(relation.key(restored), index) == layout.encode(
        relation.key(term), index
    )


def _signature(*operators: tuple[str, str | None]) -> Signature:
    return cast(
        "Signature",
        SimpleNamespace(
            operators=tuple(
                SimpleNamespace(name=name, erases_to=erases_to)
                for name, erases_to in operators
            )
        ),
    )


FEATURED: Final = (("f", None), ("f[x]", "f"), ("f[y]", "f"), ("g", None))
"""A signature with one operator core declared under two features."""


@pytest.mark.parametrize(
    "matching", [Identity[Term](), Alpha[Term](), Subsumption()], ids=repr
)
def test_a_root_preserving_congruence_identifies_a_root_only_with_itself(
    matching: RootMatching,
) -> None:
    assert matching.identified_roots("f[x]", _signature(*FEATURED)) == {"f[x]"}


def test_feature_erasure_identifies_every_operator_with_the_same_core() -> None:
    signature = _signature(*FEATURED)

    assert FeatureErasure().identified_roots("f[x]", signature) == {
        "f",
        "f[x]",
        "f[y]",
    }
    assert FeatureErasure().identified_roots("g", signature) == {"g"}


def test_a_product_identifies_the_roots_both_components_identify() -> None:
    signature = _signature(*FEATURED)
    product = Product[Term](FeatureErasure(), Identity[Term]())

    assert product.identified_roots("f[x]", signature) == {"f[x]"}
    assert isinstance(product, RootMatching)
