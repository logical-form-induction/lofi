from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.typing_system import TypingSystemConformance, TypingSystemSuite
from tests.language import (
    apply as language_apply,
)
from tests.language import (
    binary as language_binary,
)
from tests.language import (
    language,
    lexicon,
)
from tests.language import (
    unary as language_unary,
)

from lofi.terms import (
    MVar,
    Node,
    NotLocallyClosedError,
    RegionStateError,
    StoreFullError,
    Term,
    TermStore,
)
from lofi.typecheck import (
    Constraint,
    LexicalSignature,
    Scheme,
    TypeChecker,
    TypingResult,
    TypingRule,
    TypingSystem,
    UnknownSymbolError,
)

if TYPE_CHECKING:
    from collections.abc import Mapping


CONSTANTS: Final = (
    "ambient-symbol",
    "arbitrary-entity",
    "arbitrary-event",
    "arbitrary-predicate",
    "arbitrary-truth",
    "checker-annotated-proposition",
    "checker-foreign-scheme",
    "checker-undeclared",
    "conformance-truth",
    "declared-identity",
    "descendant-entity",
    "descendant-truth",
    "directed",
    "equality-entity",
    "equality-entity-again",
    "equality-event",
    "fixed-rule-entity",
    "if-left",
    "if-right",
    "iff-left",
    "iff-right",
    "ill-typed-entity",
    "ill-typed-event",
    "normalize-entity",
    "normalize-event",
    "ordered-entity",
    "ordered-event",
    "ordered-kind-entity",
    "transaction-compound",
    "transaction-event",
    "two-system-proposition",
)
LANGUAGE: Final = language(CONSTANTS)
LF_STORE: Final = LANGUAGE.store
TYPE_STORE: Final = LANGUAGE.store
ARROW: Final = LANGUAGE.store.signature.by_name("->")
ENTITY: Final = LANGUAGE.read_type("e")
EVENT: Final = LANGUAGE.read_type("v")
TRUTH: Final = LANGUAGE.read_type("t")

A = MVar("A", LANGUAGE.type_sort)
B = MVar("B", LANGUAGE.type_sort)
T = Node("#const[Ty]", payload=TRUTH.payload)
FUNCTION = Node("->", (A, B))
RULES: Final[Mapping[str, TypingRule]] = LANGUAGE.rules
EMPTY_SIGNATURE: Final = LexicalSignature({})


def arrow(domain: Term, codomain: Term) -> Term:
    """Build a semantic function type."""
    return TYPE_STORE.op(ARROW, domain, codomain)


def apply(
    function: Term,
    argument: Term,
) -> Term:
    """Build a logical-form application."""
    return language_apply(LANGUAGE, function, argument)


def unary(name: str, body: Term) -> Term:
    """Build a unary logical-form node."""
    return language_unary(LANGUAGE, name, body)


def binary(
    name: str,
    left: Term,
    right: Term,
) -> Term:
    """Build a binary logical-form node."""
    return language_binary(LANGUAGE, name, left, right)


def declare(
    name: str,
    scheme: Scheme,
    signature: dict[int, Scheme],
) -> Term:
    """Declare and build a logical-form symbol."""
    term = LANGUAGE.constant(name)
    payload = term.payload

    assert payload is not None
    signature[payload] = scheme

    return term


def test_checker_normalizes_every_node_and_infers_the_identity_scheme() -> None:
    term = unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 0))
    checker = TypeChecker(LF_STORE)
    result = checker.check(term, EMPTY_SIGNATURE, RULES)
    scheme = result.scheme

    assert result.is_well_typed
    assert set(result.assignment) == {(), (0,)}
    assert scheme is not None
    assert len(scheme.quantified) == 1
    variable = scheme.body.args[0]
    assert scheme.body is arrow(variable, variable)
    assert scheme.quantified == (variable.payload,)
    assert result.assignment[()] is scheme.body
    assert result.assignment[(0,)] is variable


def test_root_failures_are_ordered_and_withhold_the_scheme() -> None:
    signature: dict[int, Scheme] = {}
    entity = declare("ordered-entity", Scheme(ENTITY), signature)
    event = declare("ordered-event", Scheme(EVENT), signature)
    result = TypeChecker(LF_STORE).check(
        binary("and", entity, event),
        lexicon(LANGUAGE, signature),
        RULES,
    )

    assert result.residual == (
        Constraint("unification", ENTITY, TRUTH, (0,)),
        Constraint("unification", EVENT, TRUTH, ()),
    )
    assert result.scheme is None
    assert not result.is_well_typed


def test_a_descendant_failure_retains_the_available_root_scheme() -> None:
    signature: dict[int, Scheme] = {}
    entity = declare("descendant-entity", Scheme(ENTITY), signature)
    proposition = declare("descendant-truth", Scheme(TRUTH), signature)
    term = unary("lam", binary("and", entity, proposition))
    result = TypeChecker(LF_STORE).check(
        term,
        lexicon(LANGUAGE, signature),
        RULES,
    )
    scheme = result.scheme

    assert tuple(constraint.path for constraint in result.residual) == ((0, 0),)
    assert scheme is not None
    assert len(scheme.quantified) == 1
    assert scheme.body.args[1] is TRUTH


def test_self_application_becomes_a_located_occurs_residual() -> None:
    variable = LF_STORE.bvar(LANGUAGE.term_sort, 0)
    term = unary("lam", apply(variable, variable))
    result = TypeChecker(LF_STORE).check(
        term,
        EMPTY_SIGNATURE,
        RULES,
    )

    assert len(result.residual) == 1
    residual = result.residual[0]
    assert residual.kind == "infinite_type"
    assert residual.path == (0,)
    assert residual.left.free & residual.right.free
    assert result.scheme is not None


def test_failed_compound_unification_leaks_no_binding() -> None:
    signature: dict[int, Scheme] = {}
    compound = declare(
        "transaction-compound",
        Scheme(arrow(ENTITY, ENTITY)),
        signature,
    )
    event = declare("transaction-event", Scheme(EVENT), signature)
    result = TypeChecker(LF_STORE).check(
        apply(compound, event),
        lexicon(LANGUAGE, signature),
        RULES,
    )

    assert result.residual == (
        Constraint(
            "unification",
            EVENT,
            ENTITY,
            (),
        ),
    )
    assert result.assignment[(1,)] is EVENT


def test_later_successes_normalize_earlier_residuals() -> None:
    signature: dict[int, Scheme] = {}
    entity = declare("normalize-entity", Scheme(ENTITY), signature)
    event = declare("normalize-event", Scheme(EVENT), signature)
    rules = dict(RULES)
    rules[LANGUAGE.calculus.app] = TypingRule(
        LANGUAGE.calculus.app,
        (None, None),
        (A, A),
        T,
    )
    result = TypeChecker(LF_STORE).check(
        apply(entity, event),
        lexicon(LANGUAGE, signature),
        rules,
    )

    assert result.residual == (Constraint("unification", EVENT, ENTITY, ()),)


def test_ambient_unquantified_scheme_atoms_remain_rigid() -> None:
    ambient = TYPE_STORE.fvar("Ty", "ambient-rigid")
    signature: dict[int, Scheme] = {}
    term = declare("ambient-symbol", Scheme(ambient), signature)
    result = TypeChecker(LF_STORE).check(
        term,
        lexicon(LANGUAGE, signature),
        RULES,
    )

    assert result.scheme == Scheme(ambient)
    assert result.scheme is not None
    assert result.scheme.quantified == ()


def test_fixed_rule_atoms_remain_rigid_and_unquantified() -> None:
    rigid = TYPE_STORE.fvar("Ty", "fixed-rule-rigid")
    atom = rigid.payload

    assert atom is not None
    signature: dict[int, Scheme] = {}
    entity = declare("fixed-rule-entity", Scheme(ENTITY), signature)
    rules = dict(RULES)
    rules[LANGUAGE.calculus.app] = TypingRule(
        LANGUAGE.calculus.app,
        (None, None),
        (A, A),
        Node("#fvar[Ty]", payload=atom),
    )
    result = TypeChecker(LF_STORE).check(
        apply(entity, entity),
        lexicon(LANGUAGE, signature),
        rules,
    )

    assert result.assignment[()] is rigid
    assert result.scheme == Scheme(rigid)


def test_polymorphic_symbol_instances_are_regeneralized() -> None:
    variable = TYPE_STORE.fvar("Ty", "declared-polymorphic")
    atom = variable.payload

    assert atom is not None
    signature: dict[int, Scheme] = {}
    term = declare(
        "declared-identity",
        Scheme(arrow(variable, variable), (atom,)),
        signature,
    )
    result = TypeChecker(LF_STORE).check(
        term,
        lexicon(LANGUAGE, signature),
        RULES,
    )
    scheme = result.scheme

    assert scheme is not None
    assert len(scheme.quantified) == 1
    assert scheme.body.args[0] is scheme.body.args[1]
    assert scheme.quantified == (scheme.body.args[0].payload,)
    assert scheme.body is not arrow(variable, variable)


def test_compatible_and_conflicting_leaf_annotations_preserve_inference() -> None:
    variable = TYPE_STORE.fvar("Ty", "checker-annotated-polymorphic")
    atom = variable.payload

    assert atom is not None
    signature: dict[int, Scheme] = {}
    proposition = declare(
        "checker-annotated-proposition",
        Scheme(variable, (atom,)),
        signature,
    )
    term = unary("not", proposition)
    checker = TypeChecker(LF_STORE)
    held_signature = lexicon(LANGUAGE, signature)
    unannotated = checker.check(term, held_signature, RULES)
    compatible = checker.check(
        term,
        held_signature,
        RULES,
        annotations={(1,): TRUTH},
    )
    conflicting = checker.check(
        term,
        held_signature,
        RULES,
        annotations={(1,): ENTITY},
    )

    assert compatible == unannotated
    assert conflicting.assignment == unannotated.assignment
    assert conflicting.scheme == unannotated.scheme
    assert conflicting.residual == (Constraint("annotation", TRUTH, ENTITY, (1,)),)


def test_injected_constraints_refine_the_returned_path_assignment() -> None:
    variable = TYPE_STORE.fvar("Ty", "checker-injected-polymorphic")
    atom = variable.payload

    assert atom is not None
    signature: dict[int, Scheme] = {}
    proposition = declare(
        "checker-annotated-proposition",
        Scheme(variable, (atom,)),
        signature,
    )
    checker = TypeChecker(LF_STORE)
    constrained = checker.check(
        proposition,
        lexicon(LANGUAGE, signature),
        RULES,
        constraints={(): ENTITY},
    )

    assert constrained.assignment == {(): ENTITY}
    assert constrained.scheme == Scheme(ENTITY)
    assert constrained.is_well_typed


def test_a_conflicting_injected_constraint_is_located() -> None:
    signature: dict[int, Scheme] = {}
    proposition = declare("conformance-truth", Scheme(TRUTH), signature)
    result = TypeChecker(LF_STORE).check(
        proposition,
        lexicon(LANGUAGE, signature),
        RULES,
        constraints={(): ENTITY},
    )

    assert result.assignment == {(): TRUTH}
    assert result.residual == (Constraint("unification", TRUTH, ENTITY, ()),)
    assert result.scheme is None


def test_a_consumer_reads_injected_node_types_at_codec_slots() -> None:
    term = unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 0))
    result = TypeChecker(LF_STORE).check(
        term,
        EMPTY_SIGNATURE,
        RULES,
        constraints={(0,): EVENT},
    )
    path_slots = LANGUAGE.codec.at(LANGUAGE.term_sort).path_to_slot(term)
    assignment_by_slot = {
        slot: result.assignment[path] for path, slot in path_slots.items()
    }

    assert assignment_by_slot[path_slots[()]] is arrow(EVENT, EVENT)
    assert assignment_by_slot[path_slots[(0,)]] is EVENT


def test_ordinary_and_annotation_failures_retain_their_order_and_kinds() -> None:
    signature: dict[int, Scheme] = {}
    entity = declare("ordered-kind-entity", Scheme(ENTITY), signature)
    result = TypeChecker(LF_STORE).check(
        unary("not", entity),
        lexicon(LANGUAGE, signature),
        RULES,
        annotations={(1,): EVENT},
    )

    assert result.residual == (
        Constraint("unification", ENTITY, TRUTH, ()),
        Constraint("annotation", ENTITY, EVENT, (1,)),
    )


def test_a_binder_annotation_constrains_the_introduced_binding() -> None:
    term = unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 0))
    result = TypeChecker(LF_STORE).check(
        term,
        EMPTY_SIGNATURE,
        RULES,
        annotations={(): EVENT},
    )

    assert result.is_well_typed
    assert result.assignment[()] is arrow(EVENT, EVENT)
    assert result.assignment[(0,)] is EVENT
    assert result.scheme == Scheme(arrow(EVENT, EVENT))


def test_nested_binder_annotations_follow_nearest_binder_indices() -> None:
    term = unary("lam", unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 1)))
    result = TypeChecker(LF_STORE).check(
        term,
        EMPTY_SIGNATURE,
        RULES,
        annotations={(): ENTITY, (0,): EVENT, (0, 0): ENTITY},
    )

    assert result.is_well_typed
    assert result.assignment[()] is arrow(ENTITY, arrow(EVENT, ENTITY))
    assert result.assignment[(0, 0)] is ENTITY


@pytest.mark.parametrize("name", ["if", "iff"])
def test_all_binary_truth_rules_execute(name: str) -> None:
    signature: dict[int, Scheme] = {}
    left = declare(f"{name}-left", Scheme(TRUTH), signature)
    right = declare(f"{name}-right", Scheme(TRUTH), signature)
    result = TypeChecker(LF_STORE).check(
        binary(name, left, right),
        lexicon(LANGUAGE, signature),
        RULES,
    )

    assert result.is_well_typed
    assert result.assignment[()] is TRUTH


def test_the_equality_rule_shares_one_operand_type() -> None:
    signature: dict[int, Scheme] = {}
    left = declare("equality-entity", Scheme(ENTITY), signature)
    same = declare("equality-entity-again", Scheme(ENTITY), signature)
    different = declare("equality-event", Scheme(EVENT), signature)
    checker = TypeChecker(LF_STORE)
    held_signature = lexicon(LANGUAGE, signature)
    accepted = checker.check(binary("eq", left, same), held_signature, RULES)
    rejected = checker.check(binary("eq", left, different), held_signature, RULES)

    assert accepted.is_well_typed
    assert accepted.assignment[()] is TRUTH
    assert rejected.residual == (Constraint("unification", EVENT, ENTITY, ()),)


def test_checker_satisfies_the_typing_protocol() -> None:
    checker = TypeChecker(LF_STORE)
    variable = TYPE_STORE.fvar("Ty", "protocol-general")

    assert isinstance(checker, TypingSystem)
    assert checker.store is LANGUAGE.store
    assert checker.subsumes(variable, ENTITY) == {variable.payload: ENTITY}

    for slot in checker.__slots__:
        with pytest.raises(AttributeError, match="immutable"):
            checker.__setattr__(slot, TYPE_STORE)

        with pytest.raises(AttributeError, match="immutable"):
            checker.__delattr__(slot)


@pytest.fixture
def conformance() -> TypingSystemConformance[LexicalSignature]:
    """Build a conformance fixture for the real checker."""
    signature: dict[int, Scheme] = {}
    logical_form = declare("conformance-truth", Scheme(TRUTH), signature)
    general_atom = TYPE_STORE.fvar("Ty", "conformance-general")
    general = arrow(general_atom, general_atom)
    specific = arrow(ENTITY, ENTITY)
    unrelated = arrow(ENTITY, TRUTH)

    return TypingSystemConformance(
        TypeChecker(LF_STORE),
        logical_form,
        lexicon(LANGUAGE, signature),
        RULES,
        TypingResult({(): TRUTH}, (), Scheme(TRUTH)),
        general,
        specific,
        unrelated,
    )


class TestTypeChecker(TypingSystemSuite):
    """Real type-checker conformance cases."""


def test_checker_preserves_generation_boundary_errors() -> None:
    checker = TypeChecker(LF_STORE)

    with pytest.raises(NotLocallyClosedError):
        checker.check(LF_STORE.bvar(LANGUAGE.term_sort, 0), EMPTY_SIGNATURE, RULES)

    undeclared = LANGUAGE.constant("checker-undeclared")
    symbol = cast("int", undeclared.payload)
    term = unary("not", undeclared)
    declared = dict(LANGUAGE.lexicon.schemes)
    del declared[symbol]

    with pytest.raises(UnknownSymbolError) as caught:
        checker.check(term, LexicalSignature(declared), RULES)

    assert caught.value.symbol == symbol
    assert caught.value.term is undeclared
    assert caught.value.path == (1,)

    malformed = dict(RULES)
    malformed[LANGUAGE.calculus.app] = TypingRule(
        LANGUAGE.calculus.lam,
        (None, None),
        (T, T),
        T,
    )

    with pytest.raises(ValueError, match="maps 'eval' to rule 'lam'"):
        checker.check(
            unary("not", LF_STORE.fvar(LANGUAGE.term_sort, "checker-free")),
            LANGUAGE.lexicon,
            malformed,
        )


def test_checker_rejects_a_sibling_logical_form_store() -> None:
    sibling = TermStore(LANGUAGE.store.signature)
    foreign = sibling.fvar(LANGUAGE.term_sort, "foreign-source")
    checker = TypeChecker(LF_STORE)

    with pytest.raises(ValueError, match=r"outside.*language store lineage"):
        checker.check(foreign, EMPTY_SIGNATURE, RULES)


def test_checker_accepts_an_ancestor_in_its_active_source_lineage() -> None:
    root = TermStore(LANGUAGE.store.signature)
    term = root.fvar(LANGUAGE.term_sort, "ancestor-source")

    with root.region("source-child", max_nodes=8) as child:
        result = TypeChecker(child).check(
            term,
            EMPTY_SIGNATURE,
            RULES,
        )

        assert result.is_well_typed
        assert result.scheme is not None


def test_checker_refuses_a_released_or_exhausted_language_store() -> None:
    root = TermStore(LANGUAGE.store.signature)
    term: Term | None = None
    checker: TypeChecker | None = None

    with root.region("released-checker", max_nodes=8) as child:
        term = child.fvar(LANGUAGE.term_sort, "released-source")
        checker = TypeChecker(child)

    assert term is not None
    assert checker is not None

    with pytest.raises(RegionStateError, match="released"):
        checker.check(term, EMPTY_SIGNATURE, RULES)

    exhausted = TermStore(LANGUAGE.store.signature, max_nodes=1)
    variable = exhausted.fvar(LANGUAGE.term_sort, "exhausted")

    with pytest.raises(StoreFullError):
        TypeChecker(exhausted).check(
            variable,
            EMPTY_SIGNATURE,
            RULES,
        )


def test_checker_refuses_a_released_logical_form_store() -> None:
    root = TermStore(LANGUAGE.store.signature)
    term: Term | None = None
    checker: TypeChecker | None = None

    with root.region("released-source", max_nodes=8) as child:
        term = child.fvar(LANGUAGE.term_sort, "released-source-term")
        checker = TypeChecker(child)

    assert term is not None
    assert checker is not None

    with pytest.raises(RegionStateError, match="released"):
        checker.check(term, EMPTY_SIGNATURE, RULES)


ARBITRARY_SIGNATURE: dict[int, Scheme] = {}
ARBITRARY_ENTITY = declare("arbitrary-entity", Scheme(ENTITY), ARBITRARY_SIGNATURE)
ARBITRARY_EVENT = declare("arbitrary-event", Scheme(EVENT), ARBITRARY_SIGNATURE)
ARBITRARY_TRUTH = declare("arbitrary-truth", Scheme(TRUTH), ARBITRARY_SIGNATURE)
ARBITRARY_PREDICATE = declare(
    "arbitrary-predicate",
    Scheme(arrow(ENTITY, TRUTH)),
    ARBITRARY_SIGNATURE,
)
ARBITRARY_BAD = apply(ARBITRARY_ENTITY, ARBITRARY_EVENT)


def arbitrary_term(code: int) -> Term:
    """Build a closed logical form from one generated byte."""
    leaves = (
        ARBITRARY_ENTITY,
        ARBITRARY_EVENT,
        ARBITRARY_TRUTH,
        ARBITRARY_PREDICATE,
        LF_STORE.fvar(LANGUAGE.term_sort, "arbitrary-free"),
    )
    left = leaves[code % len(leaves)]
    right = leaves[(code // len(leaves)) % len(leaves)]
    constructors = (
        left,
        unary("not", left),
        unary("exists", left),
        unary("forall", left),
        unary("lam", left),
        binary("and", left, right),
        binary("or", left, right),
        apply(left, right),
    )

    return constructors[code % len(constructors)]


@given(codes=st.binary(min_size=999, max_size=999))
@settings(max_examples=1, deadline=None)
def test_check_is_total_for_a_thousand_arbitrary_closed_terms(codes: bytes) -> None:
    checker = TypeChecker(LF_STORE)
    terms = (ARBITRARY_BAD, *(arbitrary_term(code) for code in codes))
    signature = lexicon(LANGUAGE, ARBITRARY_SIGNATURE)
    results = tuple(checker.check(term, signature, RULES) for term in terms)

    assert len(results) == 1_000
    assert all(isinstance(result, TypingResult) for result in results)
    assert any(result.residual for result in results)


@given(
    case=st.sampled_from(("apply", "and", "eq", "not")),
    wrappers=st.integers(min_value=0, max_value=3),
)
@settings(max_examples=120, deadline=None)
def test_generated_ill_typed_forms_return_typed_residuals(
    case: str,
    wrappers: int,
) -> None:
    signature: dict[int, Scheme] = {}
    entity = declare("ill-typed-entity", Scheme(ENTITY), signature)
    event = declare("ill-typed-event", Scheme(EVENT), signature)

    if case == "apply":
        term = apply(entity, event)
    elif case == "and":
        term = binary("and", entity, event)
    elif case == "eq":
        term = binary("eq", entity, event)
    else:
        term = unary("not", entity)

    for _ in range(wrappers):
        term = unary("lam", term)

    result = TypeChecker(LF_STORE).check(
        term,
        lexicon(LANGUAGE, signature),
        RULES,
    )

    assert isinstance(result, TypingResult)
    assert result.residual

    for residual in result.residual:
        assert residual.kind in ("unification", "annotation", "infinite_type")
        assert isinstance(residual.path, tuple)
        assert TYPE_STORE.contains(residual.left)
        assert TYPE_STORE.contains(residual.right)


@given(
    target=st.sampled_from((ENTITY, EVENT, TRUTH, arrow(ENTITY, TRUTH))),
    wrappers=st.integers(min_value=0, max_value=6),
)
@settings(max_examples=200, deadline=None)
def test_type_directed_identity_applications_retain_the_target(
    target: Term,
    wrappers: int,
) -> None:
    signature: dict[int, Scheme] = {}
    term = declare("directed", Scheme(target), signature)

    for _ in range(wrappers):
        term = apply(unary("lam", LF_STORE.bvar(LANGUAGE.term_sort, 0)), term)

    result = TypeChecker(LF_STORE).check(
        term,
        lexicon(LANGUAGE, signature),
        RULES,
    )

    assert result.is_well_typed
    assert result.assignment[()] is target
    assert result.scheme == Scheme(target)


def test_a_foreign_scheme_remains_a_boundary_error() -> None:
    foreign = TermStore(LANGUAGE.store.signature)
    term = LANGUAGE.constant("checker-foreign-scheme")
    payload = cast("int", term.payload)
    entity = foreign.const(LANGUAGE.type_sort, "e")

    with pytest.raises(ValueError, match="outside the unifier's store lineage"):
        TypeChecker(LF_STORE).check(
            term,
            LexicalSignature({payload: Scheme(entity)}),
            RULES,
        )
