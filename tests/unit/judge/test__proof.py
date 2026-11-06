import json
from copy import copy, deepcopy
from pickle import dumps
from typing import TYPE_CHECKING, ClassVar, Final, Literal, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.rules import declared_rules

from lofi.judge import (
    PROOF_LOGICS,
    CompositionWitness,
    DerivedAction,
    ElaboratedSchema,
    Exhaustion,
    GeneralizedComposition,
    Inhabitants,
    ProofCalculus,
    ProofLogic,
    ProofSchema,
    ProofSearch,
    ProofWitness,
    StructuralRules,
    derive_action,
    elaborate_step,
    inhabitants,
)
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PatternSpec,
    RuleSetSpec,
    RuleSpec,
    SpinePatternSpec,
    SubstitutionSpec,
    TermPatternSpec,
    TypeRaisingSpec,
    VariableOperatorSpec,
    VariablePayloadSpec,
)
from lofi.spec.grammar import FunctorSpec, SignatureSpec
from lofi.terms import (
    BindingSite,
    Calculus,
    Normalization,
    Operator,
    OpName,
    Signature,
    Term,
    TermStore,
    UnknownOperatorError,
    Vocabulary,
)
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from lofi._types import JsonValue
    from lofi.judge import Rule
    from lofi.spec.grammar import AdoptedRules


LANGUAGE_SIGNATURE: Final = Signature(
    "proof-language",
    ("Cat", "Type"),
    (
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("arrow", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("type-x", "Type"),
        Operator("type-y", "Type"),
        Operator("type-z", "Type"),
        Operator("type-conj", "Type"),
        Operator("->", "Type", arg_sorts=("Type", "Type")),
    ),
    vocabularies=(Vocabulary("Cat", ("x", "x-conj", "y", "z", "conj")),),
    constant_cores={"Cat": {"x-conj": "x"}},
    open_sorts=("Type",),
)
PROOF_SIGNATURE: Final = Signature(
    "proof-terms",
    ("Proof",),
    (
        Operator("unit", "Proof"),
        Operator("app", "Proof", arg_sorts=("Proof", "Proof")),
        Operator(
            "lam", "Proof", arg_sorts=("Proof",), binding=(BindingSite((0,), "Proof"),)
        ),
    ),
)
STORE: Final = TermStore(LANGUAGE_SIGNATURE)
PROOF_STORE: Final = TermStore(PROOF_SIGNATURE)
ARROW: Final = LANGUAGE_SIGNATURE.by_name("->")
TYPE_ATOMS: Final = {
    name: STORE.op(LANGUAGE_SIGNATURE.by_name(f"type-{name}"))
    for name in ("x", "y", "z", "conj")
}
ARROW_IMAGE: Final = STORE.op(
    ARROW,
    STORE.fvar("Type", "a1"),
    STORE.fvar("Type", "a0"),
)
ERASURE: Final = Erasure(
    STORE,
    TYPE_ATOMS,
    {"fwd": ARROW_IMAGE, "bwd": ARROW_IMAGE, "arrow": ARROW_IMAGE},
    category_sort="Cat",
    function_type="->",
)
FOREIGN_STORE: Final = TermStore(LANGUAGE_SIGNATURE)
MALFORMED_TYPE_SIGNATURE: Final = Signature(
    "malformed-proof-language",
    ("Cat", "Type"),
    (
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("type-x", "Type"),
        Operator("type-y", "Type"),
        Operator("type-z", "Type"),
        Operator("type-conj", "Type"),
        Operator(
            "->",
            "Type",
            arg_sorts=("Type", "Type"),
            binding=(BindingSite((0,), "Type"),),
        ),
    ),
    vocabularies=(Vocabulary("Cat", ("x", "x-conj", "y", "z", "conj")),),
    constant_cores={"Cat": {"x-conj": "x"}},
)
MALFORMED_TYPE_STORE: Final = TermStore(MALFORMED_TYPE_SIGNATURE)


class ForeignAtomErasure:
    """Structural erasure with one foreign atom image."""

    @property
    def store(self) -> TermStore:
        return STORE

    @property
    def category_sort(self) -> str:
        return "Cat"

    @property
    def atoms(self) -> dict[OpName, Term]:
        return {
            **TYPE_ATOMS,
            "x": FOREIGN_STORE.op(LANGUAGE_SIGNATURE.by_name("type-x")),
        }

    @property
    def constructors(self) -> dict[OpName, Term]:
        return dict(ERASURE.constructors)

    @property
    def function_type(self) -> OpName:
        return "->"

    @property
    def target_sort(self) -> str:
        return "Type"

    def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
        return ERASURE.image(constructor, arguments)

    def __call__(self, category: Term, /) -> Term:
        return ERASURE(category)


class BindingArrowErasure:
    """Structural erasure with an invalid binding arrow."""

    @property
    def store(self) -> TermStore:
        return MALFORMED_TYPE_STORE

    @property
    def category_sort(self) -> str:
        return "Cat"

    @property
    def atoms(self) -> dict[OpName, Term]:
        return {
            name: MALFORMED_TYPE_STORE.op(
                MALFORMED_TYPE_SIGNATURE.by_name(f"type-{name}")
            )
            for name in ("x", "y", "z", "conj")
        }

    @property
    def constructors(self) -> dict[OpName, Term]:
        return {}

    @property
    def function_type(self) -> OpName:
        return "->"

    @property
    def target_sort(self) -> str:
        return "Type"

    def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
        del constructor, arguments

        return self.atoms["x"]

    def __call__(self, category: Term, /) -> Term:
        del category

        return self.atoms["x"]


class LambdaCalculus:
    """Proof calculus backed by the caller's term store."""

    @property
    def store(self) -> TermStore:
        return PROOF_STORE

    @property
    def operators(self) -> Calculus:
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        return self.free_variable(f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        return PROOF_STORE.fvar("Proof", name)

    def bound_variable(self, index: int, /) -> Term:
        return PROOF_STORE.bvar("Proof", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        return PROOF_STORE.op(
            PROOF_SIGNATURE.by_name("app"),
            function,
            argument,
        )

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return PROOF_STORE.bind("lam", variable, body)


CALCULUS: Final = LambdaCalculus()


def _normalization(fuel: int) -> Normalization:
    """Return the declared proof-term normalization at one search bound."""
    return Normalization(
        fuel=fuel,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )


def _constructor(name: str, spelling: str) -> ConstructorSpec:
    return ConstructorSpec(
        name=name,
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling=spelling, precedence=10, associativity="left"
        ),
    )


DIRECTED: Final = SignatureSpec(
    name="proof-categories",
    primitives=("x", "y", "z"),
    constructors=(_constructor("fwd", "/"), _constructor("bwd", "\\")),
    functors=(
        FunctorSpec(
            name="fwd", constructor="fwd", argument_side="right", opposite="bwd"
        ),
        FunctorSpec(
            name="bwd", constructor="bwd", argument_side="left", opposite="fwd"
        ),
    ),
)
"""The directed category signature the rules are compiled over.

Its constructors and category sort are the ones :data:`ERASURE` reads, so the
schemata of a rule compiled over it erase against the proof language."""

UNDIRECTED: Final = SignatureSpec(
    name="proof-undirected-categories",
    primitives=("x", "y", "z"),
    constructors=(_constructor("arrow", "->"),),
    functors=(
        FunctorSpec(
            name="arrow", constructor="arrow", argument_side="either", opposite="arrow"
        ),
    ),
)
"""The undirected category signature with one self-opposite constructor."""


def _adopted(declaration: RuleSpec, signature: SignatureSpec) -> AdoptedRules:
    return declared_rules(signature, RuleSetSpec(name="proof", rules=(declaration,)))


def _rule(declaration: RuleSpec, signature: SignatureSpec = DIRECTED) -> Rule[Term]:
    """Compile one rule through the native grammar compiler."""
    (rule,) = _adopted(declaration, signature).rule_set

    return rule


def _schema(declaration: RuleSpec, signature: SignatureSpec = DIRECTED) -> ProofSchema:
    return cast("ProofSchema", _rule(declaration, signature))


def _composition_rule(
    declaration: RuleSpec, signature: SignatureSpec = DIRECTED
) -> GeneralizedComposition:
    return cast("GeneralizedComposition", _rule(declaration, signature))


X_PATTERN: Final = TermPatternSpec(name="X", sort="Cat")
Y_PATTERN: Final = TermPatternSpec(name="Y", sort="Cat")
Z_PATTERN: Final = TermPatternSpec(name="Z", sort="Cat")


def _node(operator: str, *arguments: PatternSpec) -> NodePatternSpec:
    return NodePatternSpec(
        operator=FixedOperatorSpec(name=operator), arguments=arguments
    )


def _arrow(argument: Term, result: Term) -> Term:
    return STORE.op(ARROW, argument, result)


def _application_rule() -> ProofSchema:
    return _schema(
        PatternRuleSpec(
            name="application",
            premises=(_node("fwd", X_PATTERN, Y_PATTERN), Y_PATTERN),
            conclusion=X_PATTERN,
        )
    )


def _backward_application_rule() -> ProofSchema:
    return _schema(
        PatternRuleSpec(
            name="backward-application",
            premises=(Y_PATTERN, _node("bwd", X_PATTERN, Y_PATTERN)),
            conclusion=X_PATTERN,
        )
    )


def _kuhlmann(view: str, order: int) -> ProofSchema:
    return _schema(
        ApplicationSpec(view=view)
        if order == 0
        else CompositionSpec(view=view, order=order, direction="kuhlmann")
    )


def test_structural_ports_accept_judge_native_implementations() -> None:
    assert isinstance(CALCULUS, ProofCalculus)
    assert isinstance(_application_rule(), ProofSchema)


def test_application_derives_the_expected_interned_proof() -> None:
    (action,) = derive_action(
        _application_rule(),
        ERASURE,
        calculus=CALCULUS,
        normalization=_normalization(20),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )
    expected = CALCULUS.application(CALCULUS.hypothesis(0), CALCULUS.hypothesis(1))

    assert action.inhabitants.terms == (expected,)
    assert action.inhabitants.exhausted == frozenset()
    assert all(PROOF_STORE.contains(term) for term in action.inhabitants.terms)
    assert action.inhabitants.terms[0].free != 0


def test_golden_actions_have_exact_normal_inhabitants() -> None:
    p0 = CALCULUS.hypothesis(0)
    p1 = CALCULUS.hypothesis(1)
    argument0 = CALCULUS.free_variable("golden-argument-0")
    argument1 = CALCULUS.free_variable("golden-argument-1")
    application = CALCULUS.application
    abstraction = CALCULUS.abstraction
    cases = (
        (
            _backward_application_rule(),
            application(p1, p0),
        ),
        (
            _kuhlmann("fwd", 1),
            abstraction(
                "golden-argument-0",
                application(p0, application(p1, argument0)),
            ),
        ),
        (
            _kuhlmann("bwd", 1),
            abstraction(
                "golden-argument-0",
                application(p1, application(p0, argument0)),
            ),
        ),
        (
            _kuhlmann("fwd", 2),
            abstraction(
                "golden-argument-0",
                abstraction(
                    "golden-argument-1",
                    application(
                        p0,
                        application(application(p1, argument0), argument1),
                    ),
                ),
            ),
        ),
        (
            _kuhlmann("bwd", 2),
            abstraction(
                "golden-argument-0",
                abstraction(
                    "golden-argument-1",
                    application(
                        p1,
                        application(application(p0, argument0), argument1),
                    ),
                ),
            ),
        ),
        (
            _schema(TypeRaisingSpec(view="fwd", arguments=None)),
            abstraction(
                "golden-argument-0",
                application(argument0, p0),
            ),
        ),
        (
            _schema(TypeRaisingSpec(view="bwd", arguments=None)),
            abstraction(
                "golden-argument-0",
                application(argument0, p0),
            ),
        ),
        (
            _schema(SubstitutionSpec(view="fwd", order=1)),
            abstraction(
                "golden-argument-0",
                application(
                    application(p0, argument0),
                    application(p1, argument0),
                ),
            ),
        ),
        (
            _schema(SubstitutionSpec(view="bwd", order=1)),
            abstraction(
                "golden-argument-0",
                application(
                    application(p1, argument0),
                    application(p0, argument0),
                ),
            ),
        ),
    )

    for rule, expected in cases:
        actions = derive_action(
            rule,
            ERASURE,
            calculus=CALCULUS,
            normalization=_normalization(40),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=2, max_inhabitants=64
            ),
        )

        assert actions
        assert all(action.inhabitants.terms == (expected,) for action in actions)
        assert all(not action.inhabitants.exhausted for action in actions)


@pytest.mark.parametrize(
    "premises",
    [
        (X_PATTERN,),
        (_node("fwd", X_PATTERN, Y_PATTERN), Y_PATTERN),
        (
            _node("fwd", _node("fwd", X_PATTERN, Z_PATTERN), Y_PATTERN),
            Y_PATTERN,
            Z_PATTERN,
        ),
    ],
)
def test_derive_action_supports_unary_binary_and_ternary_schemata(
    premises: tuple[PatternSpec, ...],
) -> None:
    rule = _schema(
        PatternRuleSpec(name="ranked", premises=premises, conclusion=X_PATTERN)
    )
    (action,) = derive_action(
        rule,
        ERASURE,
        calculus=CALCULUS,
        normalization=_normalization(30),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )

    assert action.inhabitants.terms


def test_substitution_requires_contraction() -> None:
    x, y, z = (TYPE_ATOMS[name] for name in ("x", "y", "z"))
    context = (_arrow(z, _arrow(y, x)), _arrow(z, y))
    goal = _arrow(z, x)
    linear = inhabitants(
        context,
        goal,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(30),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )
    contraction = inhabitants(
        context,
        goal,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(30),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=2, max_inhabitants=64
        ),
    )

    assert linear.terms == ()
    assert linear.exhausted == frozenset({"uses"})
    assert contraction.terms
    assert set(linear.terms) <= set(contraction.terms)


@pytest.mark.parametrize(
    ("logic", "expected_terms"),
    [
        ("relevant", 0),
        ("linear", 0),
        ("affine", 1),
        ("intuitionistic", 1),
    ],
)
def test_declared_weakening_controls_whether_an_assumption_may_go_unused(
    logic: ProofLogic,
    expected_terms: int,
) -> None:
    x, y = TYPE_ATOMS["x"], TYPE_ATOMS["y"]
    rules = PROOF_LOGICS[logic]
    found = inhabitants(
        (x, y),
        y,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(10),
        search=ProofSearch(
            contraction=rules.contraction,
            weakening=rules.weakening,
            max_uses=1,
            max_inhabitants=64,
        ),
    )

    assert len(found.terms) == expected_terms


def test_each_named_logic_declares_its_structural_rules() -> None:
    assert {
        "relevant": StructuralRules(contraction=True, weakening=False),
        "linear": StructuralRules(contraction=False, weakening=False),
        "affine": StructuralRules(contraction=False, weakening=True),
        "intuitionistic": StructuralRules(contraction=True, weakening=True),
    } == PROOF_LOGICS


def test_a_search_without_contraction_refuses_more_than_one_use() -> None:
    with pytest.raises(ValueError, match="without contraction"):
        ProofSearch(contraction=False, weakening=True, max_uses=2, max_inhabitants=1)

    search = ProofSearch(
        contraction=True, weakening=False, max_uses=2, max_inhabitants=1
    )

    assert search.contraction
    assert not search.weakening
    assert search.requires_use


def test_linear_use_bound_is_a_rule_not_search_exhaustion() -> None:
    x = TYPE_ATOMS["x"]
    found = inhabitants(
        (_arrow(x, x), x),
        x,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(10),
        search=ProofSearch(
            contraction=False, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )
    expected = CALCULUS.application(
        CALCULUS.hypothesis(0),
        CALCULUS.hypothesis(1),
    )

    assert found.terms == (expected,)
    assert found.exhausted == frozenset()


def test_inhabitant_budget_reports_a_truncated_enumeration() -> None:
    x = TYPE_ATOMS["x"]
    found = inhabitants(
        (_arrow(x, x), x),
        x,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(20),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=2, max_inhabitants=1
        ),
    )

    assert len(found.terms) == 1
    assert "inhabitants" in found.exhausted


def test_native_composition_substitution_and_raising_publish_actions() -> None:
    rules = (
        _kuhlmann("fwd", 0),
        _kuhlmann("fwd", 2),
        _schema(SubstitutionSpec(view="fwd", order=1)),
        _schema(TypeRaisingSpec(view="fwd", arguments=None)),
    )
    derived = tuple(
        derive_action(
            rule,
            ERASURE,
            calculus=CALCULUS,
            normalization=_normalization(40),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=2, max_inhabitants=64
            ),
        )
        for rule in rules
    )

    assert all(actions for actions in derived)
    assert all(action.inhabitants.terms for actions in derived for action in actions)


def test_self_opposite_rules_retain_both_physical_orientations() -> None:
    rule = _schema(ApplicationSpec(view="arrow"), UNDIRECTED)
    actions = derive_action(
        rule,
        ERASURE,
        calculus=CALCULUS,
        normalization=_normalization(20),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )

    assert len(actions) == 2
    assert actions[0].sequent[0] == tuple(reversed(actions[1].sequent[0]))
    assert tuple(action.inhabitants.terms for action in actions) == (
        (CALCULUS.application(CALCULUS.hypothesis(0), CALCULUS.hypothesis(1)),),
        (CALCULUS.application(CALCULUS.hypothesis(1), CALCULUS.hypothesis(0)),),
    )


def test_fuel_exhaustion_is_reported_without_mislabeling_uses() -> None:
    x, y = TYPE_ATOMS["x"], TYPE_ATOMS["y"]
    found = inhabitants(
        (_arrow(y, x), y),
        x,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(1),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=3, max_inhabitants=64
        ),
    )

    assert found.terms == ()
    assert found.exhausted == frozenset({"fuel"})


def test_partial_findings_retain_their_exhaustion_tags() -> None:
    x = TYPE_ATOMS["x"]
    context = (_arrow(x, x), x)
    expected = CALCULUS.application(
        CALCULUS.hypothesis(0),
        CALCULUS.hypothesis(1),
    )
    fuel_limited = inhabitants(
        context,
        x,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(2),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=2, max_inhabitants=64
        ),
    )
    use_limited = inhabitants(
        context,
        x,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(10),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )

    assert fuel_limited.terms == (expected,)
    assert fuel_limited.exhausted == frozenset({"fuel"})
    assert expected in use_limited.terms
    assert use_limited.exhausted == frozenset({"uses"})


def test_nondeterministic_patterns_publish_no_schema() -> None:
    rules = tuple(
        _schema(declaration)
        for declaration in (
            PatternRuleSpec(
                name="operator-variable",
                premises=(
                    NodePatternSpec(
                        operator=VariableOperatorSpec(name="F"),
                        arguments=(X_PATTERN, Y_PATTERN),
                    ),
                ),
                conclusion=X_PATTERN,
            ),
            PatternRuleSpec(
                name="payload-variable",
                premises=(
                    NodePatternSpec(
                        operator=FixedOperatorSpec(name="#const[Cat]"),
                        payload=VariablePayloadSpec(name="P"),
                    ),
                ),
                conclusion=X_PATTERN,
            ),
            PatternRuleSpec(
                name="spine-variable",
                premises=(
                    SpinePatternSpec(
                        name="K", core=X_PATTERN, operators=("fwd",), max_order=None
                    ),
                ),
                conclusion=X_PATTERN,
            ),
        )
    )

    for rule in rules:
        assert rule.proof_schemata == ()
        assert (
            derive_action(
                rule,
                ERASURE,
                calculus=CALCULUS,
                normalization=_normalization(20),
                search=ProofSearch(
                    contraction=True, weakening=False, max_uses=1, max_inhabitants=64
                ),
            )
            == ()
        )


def test_derive_action_rejects_a_target_without_variable_channels() -> None:
    signature = Signature(
        "closed-proof-language",
        ("Cat", "Type"),
        (
            Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("type-x", "Type"),
            Operator("type-y", "Type"),
            Operator("type-z", "Type"),
            Operator("type-conj", "Type"),
            Operator("->", "Type", arg_sorts=("Type", "Type")),
        ),
        vocabularies=(Vocabulary("Cat", ("x", "x-conj", "y", "z", "conj")),),
        constant_cores={"Cat": {"x-conj": "x"}},
    )
    closed_store = TermStore(signature)

    class ClosedTargetErasure:
        """Expose a closed type signature through the structural protocol."""

        store = closed_store
        category_sort = "Cat"
        atoms: ClassVar[dict[OpName, Term]] = {
            name: closed_store.op(signature.by_name(f"type-{name}"))
            for name in ("x", "y", "z", "conj")
        }
        constructors: ClassVar[dict[OpName, Term]] = {}
        function_type = "->"
        target_sort = "Type"

        def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
            del constructor, arguments

            return self.atoms["x"]

        def __call__(self, category: Term, /) -> Term:
            return self.atoms[category.core.op.name]

    erasure = ClosedTargetErasure()

    with pytest.raises(UnknownOperatorError):
        derive_action(
            _application_rule(),
            erasure,
            calculus=CALCULUS,
            normalization=_normalization(20),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


def test_derive_action_rejects_a_foreign_erasure_image() -> None:
    rule = _schema(
        PatternRuleSpec(
            name="fixed-atom",
            premises=(ConstantPatternSpec(name="x"),),
            conclusion=ConstantPatternSpec(name="x"),
        )
    )

    with pytest.raises(ValueError, match="outside its store lineage"):
        derive_action(
            rule,
            ForeignAtomErasure(),
            calculus=CALCULUS,
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


def test_derive_action_rejects_a_binding_target_arrow() -> None:
    with pytest.raises(ValueError, match="lacks a valid variable or arrow channel"):
        derive_action(
            _application_rule(),
            BindingArrowErasure(),
            calculus=CALCULUS,
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


class MissingBoundCalculus:
    """Proof calculus without the required bound-variable channel."""

    @property
    def store(self) -> TermStore:
        return PROOF_STORE

    @property
    def operators(self) -> Calculus:
        return CALCULUS.operators

    def hypothesis(self, index: int, /) -> Term:
        return CALCULUS.hypothesis(index)

    def free_variable(self, name: str, /) -> Term:
        return CALCULUS.free_variable(name)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        return CALCULUS.application(function, argument)

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return CALCULUS.abstraction(variable, body)


class MissingFreeCalculus:
    """Proof calculus without the required free-variable channel."""

    @property
    def store(self) -> TermStore:
        return PROOF_STORE

    @property
    def operators(self) -> Calculus:
        return CALCULUS.operators

    def hypothesis(self, index: int, /) -> Term:
        return PROOF_STORE.fvar("Proof", f"p{index}")

    def bound_variable(self, index: int, /) -> Term:
        return PROOF_STORE.bvar("Proof", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        return PROOF_STORE.op(PROOF_SIGNATURE.by_name("app"), function, argument)

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return PROOF_STORE.bind("lam", variable, body)


@pytest.mark.parametrize(
    "malformed",
    [
        cast("ProofCalculus", MissingBoundCalculus()),
        cast("ProofCalculus", MissingFreeCalculus()),
    ],
)
def test_inhabitants_rejects_a_missing_variable_channel(
    malformed: ProofCalculus,
) -> None:

    with pytest.raises(TypeError, match="lacks a required construction channel"):
        inhabitants(
            (TYPE_ATOMS["x"],),
            TYPE_ATOMS["x"],
            calculus=malformed,
            arrow="->",
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


class NonvariableHypothesisCalculus(LambdaCalculus):
    """Proof calculus with a malformed hypothesis channel."""

    @override
    def hypothesis(self, index: int, /) -> Term:
        del index

        return PROOF_STORE.op(PROOF_SIGNATURE.by_name("unit"))


def test_inhabitants_rejects_a_store_owned_nonvariable_hypothesis() -> None:
    with pytest.raises(ValueError, match="did not return a free variable"):
        inhabitants(
            (TYPE_ATOMS["x"],),
            TYPE_ATOMS["x"],
            calculus=NonvariableHypothesisCalculus(),
            arrow="->",
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


class UnstableHypothesisCalculus(LambdaCalculus):
    """Proof calculus with an unstable hypothesis channel."""

    def __init__(self) -> None:
        self._calls = 0

    @override
    def hypothesis(self, index: int, /) -> Term:
        self._calls += 1

        return self.free_variable(f"unstable-{index}-{self._calls}")


class NoninjectiveHypothesisCalculus(LambdaCalculus):
    """Proof calculus with a noninjective hypothesis channel."""

    @override
    def hypothesis(self, index: int, /) -> Term:
        del index

        return self.free_variable("shared-hypothesis")


class UnstableThirdHypothesisCalculus(LambdaCalculus):
    """Proof calculus whose third hypothesis changes between calls."""

    def __init__(self) -> None:
        self._third_calls = 0

    @override
    def hypothesis(self, index: int, /) -> Term:
        if index != 2:
            return CALCULUS.hypothesis(index)

        self._third_calls += 1

        return self.free_variable(f"unstable-third-{self._third_calls}")


@pytest.mark.parametrize(
    ("calculus", "message"),
    [
        (UnstableHypothesisCalculus(), "not stable"),
        (NoninjectiveHypothesisCalculus(), "not injective"),
    ],
)
def test_inhabitants_rejects_invalid_hypothesis_identity(
    calculus: ProofCalculus,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        inhabitants(
            (),
            _arrow(TYPE_ATOMS["x"], TYPE_ATOMS["x"]),
            calculus=calculus,
            arrow="->",
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


def test_inhabitants_rechecks_every_materialized_hypothesis() -> None:
    with pytest.raises(ValueError, match="not stable"):
        inhabitants(
            (TYPE_ATOMS["x"], TYPE_ATOMS["y"], TYPE_ATOMS["z"]),
            TYPE_ATOMS["x"],
            calculus=UnstableThirdHypothesisCalculus(),
            arrow="->",
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


type MalformedChannel = Literal[
    "free-variable",
    "bound-variable",
    "application",
    "abstraction",
]


class MalformedCalculus(LambdaCalculus):
    """Proof calculus with one malformed construction channel."""

    def __init__(self, channel: MalformedChannel) -> None:
        self._channel = channel

    @override
    def hypothesis(self, index: int, /) -> Term:
        return CALCULUS.hypothesis(index)

    @override
    def free_variable(self, name: str, /) -> Term:
        if self._channel == "free-variable":
            return PROOF_STORE.op(PROOF_SIGNATURE.by_name("unit"))

        return CALCULUS.free_variable(name)

    @override
    def bound_variable(self, index: int, /) -> Term:
        if self._channel == "bound-variable":
            return PROOF_STORE.op(PROOF_SIGNATURE.by_name("unit"))

        return CALCULUS.bound_variable(index)

    @override
    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        if self._channel == "application":
            return PROOF_STORE.op(PROOF_SIGNATURE.by_name("unit"))

        return CALCULUS.application(function, argument)

    @override
    def abstraction(self, variable: str, body: Term, /) -> Term:
        if self._channel == "abstraction":
            return PROOF_STORE.op(PROOF_SIGNATURE.by_name("unit"))

        return CALCULUS.abstraction(variable, body)


@pytest.mark.parametrize(
    "channel",
    ["free-variable", "bound-variable", "application", "abstraction"],
)
def test_inhabitants_rejects_malformed_construction_channels(
    channel: MalformedChannel,
) -> None:
    with pytest.raises(ValueError, match=rf"{channel} channel returned a malformed"):
        inhabitants(
            (TYPE_ATOMS["x"],),
            TYPE_ATOMS["x"],
            calculus=MalformedCalculus(channel),
            arrow="->",
            normalization=_normalization(10),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
        )


class CollidingHypothesisCalculus(LambdaCalculus):
    """Proof calculus whose hypothesis occupies the first binder candidate."""

    @override
    def hypothesis(self, index: int, /) -> Term:
        return self.free_variable(f"__proof_bound_0_{index}")


def test_binder_variables_do_not_capture_caller_hypotheses() -> None:
    calculus = CollidingHypothesisCalculus()
    x = TYPE_ATOMS["x"]
    argument = calculus.free_variable("__proof_bound_0_1")
    expected = calculus.abstraction(
        "__proof_bound_0_1",
        calculus.application(calculus.hypothesis(0), argument),
    )
    found = inhabitants(
        (_arrow(x, x),),
        _arrow(x, x),
        calculus=calculus,
        arrow="->",
        normalization=_normalization(10),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )

    assert found.terms == (expected,)


def test_results_are_frozen_identity_copies_and_refuse_pickle() -> None:
    first = CALCULUS.abstraction("alpha-left", CALCULUS.free_variable("alpha-left"))
    second = CALCULUS.abstraction(
        "alpha-right",
        CALCULUS.free_variable("alpha-right"),
    )
    found = Inhabitants((first, second, first))
    action = DerivedAction(((TYPE_ATOMS["x"],), TYPE_ATOMS["x"]), found)

    assert first is second
    assert found.terms == (first,)

    for value in (found, action):
        assert copy(value) is value
        assert deepcopy(value) is value

        with pytest.raises(TypeError, match="live interned terms"):
            dumps(value)

        with pytest.raises(AttributeError, match="immutable"):
            value.__setattr__("_terms", ())


def test_results_reject_unknown_exhaustion_tags() -> None:
    unknown = cast("tuple[Exhaustion, ...]", ("order",))

    with pytest.raises(ValueError, match="unknown proof-search exhaustion"):
        Inhabitants((), unknown)


@pytest.mark.parametrize(
    ("fuel", "max_uses", "message"),
    [
        (-1, 1, "normalization fuel is negative"),
        (1, 0, "max_uses must be positive"),
    ],
)
def test_inhabitants_rejects_invalid_bounds(
    fuel: int,
    max_uses: int,
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        inhabitants(
            (TYPE_ATOMS["x"],),
            TYPE_ATOMS["x"],
            calculus=CALCULUS,
            arrow="->",
            normalization=_normalization(fuel),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=max_uses, max_inhabitants=64
            ),
        )


def _check(
    proof: Term,
    expected: Term,
    hypotheses: tuple[Term, ...],
    bound: tuple[Term, ...] = (),
) -> bool:
    """Check a proof against a type without using the search implementation."""
    if expected.op.name == "->":
        if proof.op.name != "lam":
            return False

        domain, codomain = expected.args

        return _check(proof.args[0], codomain, hypotheses, (domain, *bound))

    if proof.op.name == "lam":
        return False

    inferred = _infer(proof, hypotheses, bound)

    return inferred is expected


def _infer(
    proof: Term,
    hypotheses: tuple[Term, ...],
    bound: tuple[Term, ...],
) -> Term | None:
    """Infer a non-abstraction proof type independently."""
    if proof.op.payload_sort == "free" and proof.payload is not None:
        name = PROOF_STORE.atom_name(proof.payload)

        if name.startswith("p") and name[1:].isdigit():
            index = int(name[1:])

            return hypotheses[index] if index < len(hypotheses) else None

        return None

    if proof.op.payload_sort == "index" and proof.payload is not None:
        return bound[proof.payload] if proof.payload < len(bound) else None

    if proof.op.name != "app":
        return None

    function = _infer(proof.args[0], hypotheses, bound)

    if function is None or function.op.name != "->":
        return None

    argument, result = function.args

    return result if _check(proof.args[1], argument, hypotheses, bound) else None


def _is_beta_normal(proof: Term) -> bool:
    """Check that a proof has no beta redex."""
    if proof.op.name == "app" and proof.args[0].op.name == "lam":
        return False

    return all(_is_beta_normal(argument) for argument in proof.args)


def _contains(proof: Term, target: Term) -> bool:
    """Check whether a proof contains one interned term."""
    return proof is target or any(
        _contains(argument, target) for argument in proof.args
    )


def _uses_bound(proof: Term, index: int) -> bool:
    """Check whether a proof uses one surrounding bound variable."""
    if proof.op.payload_sort == "index":
        return proof.payload == index

    return any(
        _uses_bound(argument, index + len(proof.op.scope(position)))
        for position, argument in enumerate(proof.args)
    )


def _binders_are_relevant(proof: Term) -> bool:
    """Check that every proof binder owns a use in its body."""
    if proof.op.name == "lam" and not _uses_bound(proof.args[0], 0):
        return False

    return all(_binders_are_relevant(argument) for argument in proof.args)


@settings(max_examples=120, deadline=None)
@given(
    st.lists(
        st.sampled_from(tuple(TYPE_ATOMS[name] for name in ("x", "y", "z"))),
        min_size=1,
        max_size=2,
    ),
    st.lists(
        st.sampled_from(tuple(TYPE_ATOMS[name] for name in ("x", "y", "z"))),
        min_size=0,
        max_size=2,
    ),
    st.sampled_from(tuple(TYPE_ATOMS[name] for name in ("x", "y", "z"))),
)
def test_generated_inhabitants_satisfy_normality_and_relevance(
    arguments: list[Term],
    goal_arguments: list[Term],
    result: Term,
) -> None:
    function = result

    for argument in reversed((*arguments, *goal_arguments)):
        function = _arrow(argument, function)

    goal = result

    for argument in reversed(goal_arguments):
        goal = _arrow(argument, goal)

    context = (function, *arguments)
    found = inhabitants(
        context,
        goal,
        calculus=CALCULUS,
        arrow="->",
        normalization=_normalization(30),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )
    hypotheses = tuple(CALCULUS.hypothesis(at) for at in range(len(context)))

    assert found.terms
    assert all(_check(proof, goal, context) for proof in found.terms)
    assert all(_is_beta_normal(proof) for proof in found.terms)
    assert all(_binders_are_relevant(proof) for proof in found.terms)
    assert all(
        all(_contains(proof, hypothesis) for hypothesis in hypotheses)
        for proof in found.terms
    )


def test_higher_use_bounds_are_monotone() -> None:
    atoms = tuple(TYPE_ATOMS[name] for name in ("x", "y", "z"))

    for argument in atoms:
        for result in atoms:
            context = (_arrow(argument, result), argument)
            lower = inhabitants(
                context,
                result,
                calculus=CALCULUS,
                arrow="->",
                normalization=_normalization(10),
                search=ProofSearch(
                    contraction=True, weakening=False, max_uses=1, max_inhabitants=64
                ),
            )
            higher = inhabitants(
                context,
                result,
                calculus=CALCULUS,
                arrow="->",
                normalization=_normalization(10),
                search=ProofSearch(
                    contraction=True, weakening=False, max_uses=2, max_inhabitants=64
                ),
            )

            assert set(lower.terms) <= set(higher.terms)


def _step_erasure(adopted: AdoptedRules, constructors: tuple[str, ...]) -> Erasure:
    """Erase the compiled categories into the types of their learner language."""
    store = adopted.context.store
    image = store.op(
        store.signature.by_name("->"), store.fvar("Ty", "a1"), store.fvar("Ty", "a0")
    )

    return Erasure(
        store,
        {
            "x": store.const("Ty", "e"),
            "y": store.const("Ty", "t"),
            "z": store.const("Ty", "v"),
        },
        dict.fromkeys(constructors, image),
        category_sort="Cat",
        function_type="->",
    )


_STEP_RULES: Final = _adopted(ApplicationSpec(view="fwd"), DIRECTED)
STEP_ERASURE: Final = _step_erasure(_STEP_RULES, ("fwd", "bwd"))
_STEP_CONTEXT: Final = _STEP_RULES.context
FUNCTORS, _ = _STEP_CONTEXT.functors["fwd"]
CATEGORY_ATOMS: Final = {
    name: _STEP_CONTEXT.syntax.parse(_STEP_CONTEXT.store, name)
    for name in ("x", "y", "z")
}

type Role = Literal["forward", "backward"]

SLASHES: Final = ("fwd", "bwd")

MIXED_FRAMES: Final = (
    (),
    ("fwd",),
    ("bwd",),
    ("fwd", "fwd"),
    ("bwd", "fwd"),
    ("fwd", "bwd"),
    ("bwd", "bwd"),
    ("fwd", "fwd", "fwd"),
    ("bwd", "fwd", "bwd"),
    ("fwd", "bwd", "fwd"),
)

P0: Final = CALCULUS.hypothesis(0)
P1: Final = CALCULUS.hypothesis(1)
HAND_0: Final = CALCULUS.free_variable("hand-written-0")
HAND_1: Final = CALCULUS.free_variable("hand-written-1")
HAND_2: Final = CALCULUS.free_variable("hand-written-2")


def _by_hand(function: Term, argument: Term) -> Term:
    return CALCULUS.application(function, argument)


HAND_WRITTEN: Final = {
    ("forward", 0): _by_hand(P0, P1),
    ("forward", 1): CALCULUS.abstraction(
        "hand-written-0", _by_hand(P0, _by_hand(P1, HAND_0))
    ),
    ("forward", 2): CALCULUS.abstraction(
        "hand-written-0",
        CALCULUS.abstraction(
            "hand-written-1",
            _by_hand(P0, _by_hand(_by_hand(P1, HAND_0), HAND_1)),
        ),
    ),
    ("forward", 3): CALCULUS.abstraction(
        "hand-written-0",
        CALCULUS.abstraction(
            "hand-written-1",
            CALCULUS.abstraction(
                "hand-written-2",
                _by_hand(P0, _by_hand(_by_hand(_by_hand(P1, HAND_0), HAND_1), HAND_2)),
            ),
        ),
    ),
    ("backward", 0): _by_hand(P1, P0),
    ("backward", 1): CALCULUS.abstraction(
        "hand-written-0", _by_hand(P1, _by_hand(P0, HAND_0))
    ),
    ("backward", 2): CALCULUS.abstraction(
        "hand-written-0",
        CALCULUS.abstraction(
            "hand-written-1",
            _by_hand(P1, _by_hand(_by_hand(P0, HAND_0), HAND_1)),
        ),
    ),
    ("backward", 3): CALCULUS.abstraction(
        "hand-written-0",
        CALCULUS.abstraction(
            "hand-written-1",
            CALCULUS.abstraction(
                "hand-written-2",
                _by_hand(P1, _by_hand(_by_hand(_by_hand(P0, HAND_0), HAND_1), HAND_2)),
            ),
        ),
    ),
}


def _folded(role: Role, order: int) -> Term:
    """Write the composed term at one order by folding over its arguments."""
    principal, container = (P0, P1) if role == "forward" else (P1, P0)
    names = tuple(f"folded-{at}" for at in range(order))
    applied = container

    for name in names:
        applied = CALCULUS.application(applied, CALCULUS.free_variable(name))

    term = CALCULUS.application(principal, applied)

    for name in reversed(names):
        term = CALCULUS.abstraction(name, term)

    return term


def _slash_of(role: Role) -> OpName:
    return "fwd" if role == "forward" else "bwd"


def _spine(base: Term, frames: tuple[OpName, ...]) -> Term:
    """Wrap a category in a result spine whose frames run shallowest first."""
    built = base

    for slash in reversed(frames):
        built = FUNCTORS.make(slash, built, CATEGORY_ATOMS["z"])

    return built


def _step(role: Role, frames: tuple[OpName, ...]) -> tuple[tuple[Term, ...], Term]:
    """Build the premises and conclusion of one composition step."""
    slash = _slash_of(role)
    principal = FUNCTORS.make(slash, CATEGORY_ATOMS["x"], CATEGORY_ATOMS["y"])
    container = _spine(CATEGORY_ATOMS["y"], frames)
    inputs = (principal, container) if role == "forward" else (container, principal)

    return inputs, _spine(CATEGORY_ATOMS["x"], frames)


def _composition(role: Role, frames: tuple[OpName, ...]) -> GeneralizedComposition:
    """Build the rule that licenses a step of one order."""
    view = _slash_of(role)

    return _composition_rule(
        ApplicationSpec(view=view)
        if not frames
        else CompositionSpec(view=view, order=None, direction="unrestricted")
    )


def test_the_folded_term_agrees_with_every_hand_written_one() -> None:
    assert all(
        _folded(role, order) is HAND_WRITTEN[(role, order)]
        for role in ("forward", "backward")
        for order in range(4)
    )


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("frames", MIXED_FRAMES, ids="-".join)
def test_a_composition_step_elaborates_the_hand_written_term(
    role: Role, frames: tuple[OpName, ...]
) -> None:
    rule = _composition(role, frames)
    inputs, conclusion = _step(role, frames)
    action = elaborate_step(rule, inputs, conclusion, STEP_ERASURE, calculus=CALCULUS)

    assert action is not None
    assert action.sequent == (
        tuple(STEP_ERASURE(premise) for premise in inputs),
        STEP_ERASURE(conclusion),
    )
    assert action.inhabitants.terms == (HAND_WRITTEN[(role, len(frames))],)
    assert action.inhabitants.exhausted == frozenset()

    (proof,) = action.inhabitants.terms
    premises, goal = action.sequent

    assert _check(proof, goal, premises)
    assert _is_beta_normal(proof)
    assert _binders_are_relevant(proof)


@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("frames", MIXED_FRAMES, ids="-".join)
def test_the_witness_reports_the_spine_the_step_composed_over(
    role: Role, frames: tuple[OpName, ...]
) -> None:
    rule = _composition(role, frames)
    inputs, conclusion = _step(role, frames)
    witnesses = rule.witnesses(inputs, conclusion)
    slash = _slash_of(role)

    assert rule.witness_is_unique is True
    assert len(witnesses) == 1
    (witness,) = witnesses
    assert isinstance(witness, ProofWitness)
    assert witness.order == len(frames)
    assert witness.frames == frames
    assert witness.slash == slash
    assert witness.principal == (0 if role == "forward" else 1)
    assert witness.container == (1 if role == "forward" else 0)
    assert witness.crossed == (bool(frames) and frames[-1] != slash)
    assert rule.applied_order(inputs, conclusion) == len(frames)


def test_coincident_self_opposite_inputs_preserve_both_physical_orientations() -> None:
    adopted = _adopted(
        CompositionSpec(view="arrow", order=None, direction="unrestricted"), UNDIRECTED
    )
    (rule,) = cast("tuple[GeneralizedComposition]", tuple(adopted.rule_set))
    functors, _ = adopted.context.functors["arrow"]
    x = adopted.context.syntax.parse(adopted.context.store, "x")
    coincident = functors.make("arrow", x, x)
    inputs = (coincident, coincident)
    witnesses = rule.witnesses(inputs, coincident)
    action = elaborate_step(
        rule, inputs, coincident, _step_erasure(adopted, ("arrow",)), calculus=CALCULUS
    )
    expected = (
        CompositionWitness(0, 1, "arrow", ("arrow",)),
        CompositionWitness(1, 0, "arrow", ("arrow",)),
    )

    assert rule.witness_is_unique is False
    assert witnesses == expected
    assert rule.applied_order(inputs, coincident) == 1
    assert action is not None
    assert action.inhabitants.terms == tuple(
        witness.elaborate(CALCULUS) for witness in expected
    )


@settings(max_examples=40, deadline=None)
@given(
    st.lists(st.sampled_from(SLASHES), min_size=4, max_size=11),
    st.sampled_from(("forward", "backward")),
)
def test_a_randomized_larger_order_elaborates_the_folded_term(
    frames: list[OpName], role: Role
) -> None:
    held = tuple(frames)
    rule = _composition(role, held)
    inputs, conclusion = _step(role, held)
    action = elaborate_step(rule, inputs, conclusion, STEP_ERASURE, calculus=CALCULUS)
    witnesses = rule.witnesses(inputs, conclusion)

    assert len(witnesses) == 1
    (witness,) = witnesses
    assert witness.frames == held
    assert action is not None
    assert action.inhabitants.terms == (_folded(role, len(held)),)

    (proof,) = action.inhabitants.terms
    premises, goal = action.sequent

    assert _check(proof, goal, premises)
    assert _binders_are_relevant(proof)


def test_the_schema_publishes_no_sequent_for_search_and_elaborates_instead() -> None:
    rule = _composition("forward", ("fwd",))
    inputs, conclusion = _step("forward", ("fwd", "bwd"))

    assert rule.order is None
    assert rule.proof_schemata == ()
    assert isinstance(rule, ElaboratedSchema)
    assert (
        derive_action(
            rule,
            ERASURE,
            calculus=CALCULUS,
            normalization=_normalization(40),
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=2, max_inhabitants=64
            ),
        )
        == ()
    )

    action = elaborate_step(rule, inputs, conclusion, STEP_ERASURE, calculus=CALCULUS)

    assert action is not None
    assert action.inhabitants.terms == (HAND_WRITTEN[("forward", 2)],)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_the_elaborated_term_is_the_one_search_finds_at_a_declared_order(
    order: int,
) -> None:
    frames = ("fwd",) * order
    declared = _kuhlmann("fwd", order)
    schema = _composition("forward", frames)
    inputs, conclusion = _step("forward", frames)
    actions = derive_action(
        declared,
        ERASURE,
        calculus=CALCULUS,
        normalization=_normalization(40),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=2, max_inhabitants=64
        ),
    )
    elaborated = elaborate_step(
        schema, inputs, conclusion, STEP_ERASURE, calculus=CALCULUS
    )

    assert actions
    assert elaborated is not None
    assert {action.inhabitants.terms for action in actions} == {
        elaborated.inhabitants.terms
    }


def test_no_step_is_elaborated_where_the_rule_licenses_none() -> None:
    rule = _composition("forward", ("fwd",))
    inputs, conclusion = _step("forward", ("fwd",))

    assert (
        elaborate_step(
            rule, inputs, CATEGORY_ATOMS["z"], STEP_ERASURE, calculus=CALCULUS
        )
        is None
    )
    assert elaborate_step(rule, (), conclusion, ERASURE, calculus=CALCULUS) is None
    assert (
        elaborate_step(
            _composition("backward", ("fwd",)),
            inputs,
            conclusion,
            STEP_ERASURE,
            calculus=CALCULUS,
        )
        is None
    )


class FixedProofWitness:
    """A third-party witness returning one fixed term."""

    def __init__(self, term: Term) -> None:
        self._term = term

    def elaborate(self, calculus: ProofCalculus, /) -> Term:
        del calculus

        return self._term


class FixedElaboratedSchema:
    """A third-party schema exposing one fixed proof witness."""

    def __init__(self, term: Term) -> None:
        self._witness = FixedProofWitness(term)

    @property
    def witness_is_unique(self) -> bool:
        return True

    def witnesses(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> tuple[ProofWitness, ...]:
        del inputs, output

        return (self._witness,)


def test_elaboration_rejects_a_foreign_witness_term() -> None:
    foreign_store = TermStore(PROOF_SIGNATURE)
    inputs, conclusion = _step("forward", ("fwd",))
    schema = FixedElaboratedSchema(foreign_store.fvar("Proof", "p0"))

    with pytest.raises(ValueError, match="witness channel returned a foreign term"):
        elaborate_step(schema, inputs, conclusion, STEP_ERASURE, calculus=CALCULUS)


def test_elaboration_rejects_a_witness_with_the_wrong_hypothesis_mask() -> None:
    inputs, conclusion = _step("forward", ("fwd",))
    schema = FixedElaboratedSchema(CALCULUS.hypothesis(0))

    with pytest.raises(ValueError, match=r"hypothesis mask .* not .* for arity 2"):
        elaborate_step(schema, inputs, conclusion, STEP_ERASURE, calculus=CALCULUS)


def test_a_witness_round_trips_through_serialization() -> None:
    witness = CompositionWitness(1, 0, "bwd", ("fwd", "bwd", "fwd"))
    encoded = json.dumps(dict(witness.to_spec()), sort_keys=True)
    restored = CompositionWitness.from_spec(json.loads(encoded))

    assert restored == witness
    assert hash(restored) == hash(witness)
    assert restored.order == witness.order
    assert restored.crossed == witness.crossed
    assert json.dumps(dict(restored.to_spec()), sort_keys=True) == encoded
    assert repr(witness) == "CompositionWitness(1, 0, 'bwd', ('fwd', 'bwd', 'fwd'))"


def test_a_witness_compares_on_its_fields_alone() -> None:
    witness = CompositionWitness(0, 1, "fwd", ("fwd",))

    assert witness == CompositionWitness(0, 1, "fwd", ("fwd",))
    assert witness != CompositionWitness(1, 0, "fwd", ("fwd",))
    assert witness != CompositionWitness(0, 1, "bwd", ("fwd",))
    assert witness != CompositionWitness(0, 1, "fwd", ("bwd",))
    assert witness.__eq__(witness.to_spec()) is NotImplemented
    assert copy(witness) is witness
    assert deepcopy(witness) is witness


@pytest.mark.parametrize(
    ("principal", "container", "slash", "frames", "error", "message"),
    [
        (0, 0, "fwd", (), ValueError, "both principal and container"),
        (-1, 1, "fwd", (), ValueError, "principal is negative"),
        (0, -1, "fwd", (), ValueError, "container is negative"),
        (0, 1, "", (), ValueError, "slash is empty"),
        (0, 1, "fwd", ("",), ValueError, r"frames\[0\] is empty"),
        (True, 1, "fwd", (), TypeError, "principal is not an integer"),
    ],
)
def test_a_malformed_witness_is_refused(
    principal: int,
    container: int,
    slash: OpName,
    frames: tuple[OpName, ...],
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        CompositionWitness(principal, container, slash, frames)


@pytest.mark.parametrize(
    "spec",
    [
        {"container": 1, "slash": "fwd", "frames": []},
        {"principal": 0, "slash": "fwd", "frames": []},
        {"principal": 0, "container": 1, "frames": []},
        {"principal": 0, "container": 1, "slash": "fwd"},
        {"principal": True, "container": 1, "slash": "fwd", "frames": []},
        {"principal": 0, "container": False, "slash": "fwd", "frames": []},
        {"principal": 0, "container": 1, "slash": 7, "frames": []},
        {"principal": 0, "container": 1, "slash": "fwd", "frames": "fwd"},
        {"principal": 0, "container": 1, "slash": "fwd", "frames": [1]},
    ],
)
def test_a_malformed_witness_declaration_is_refused(
    spec: dict[str, JsonValue],
) -> None:
    with pytest.raises(ValueError, match="malformed composition witness"):
        CompositionWitness.from_spec(spec)


def test_a_spine_binder_does_not_capture_a_colliding_hypothesis() -> None:
    calculus = CollidingHypothesisCalculus()
    rule = _composition("forward", ("fwd",))
    inputs, conclusion = _step("forward", ("fwd",))
    principal = calculus.hypothesis(0)
    container = calculus.hypothesis(1)
    argument = calculus.free_variable("__proof_bound_0_2")
    expected = calculus.abstraction(
        "__proof_bound_0_2",
        calculus.application(principal, calculus.application(container, argument)),
    )
    action = elaborate_step(rule, inputs, conclusion, STEP_ERASURE, calculus=calculus)

    assert action is not None
    assert action.inhabitants.terms == (expected,)

    (proof,) = action.inhabitants.terms

    assert _contains(proof, principal)
    assert _contains(proof, container)
