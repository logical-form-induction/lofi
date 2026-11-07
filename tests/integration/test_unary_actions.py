from typing import Final

from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.derivations import FormLexicon, axiom, step
from tests.support.rules import ccg_signature, declared_rules

from lofi.interpret import (
    MeaningReading,
    MeaningSelection,
    StringReading,
    compile_meaning_actions,
    read,
)
from lofi.judge import (
    DerivedAction,
    ProofSearch,
    Rule,
    derive_action,
)
from lofi.lexicon import Entry
from lofi.spec import (
    ApplicationSpec,
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    RuleSpec,
    TypeRaisingSpec,
)
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.grammar import (
    AdoptedRules,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    OperatorSpec,
)
from lofi.spec.language import AtomicTypeSpec, registered_language
from lofi.terms import (
    BindingSite,
    Calculus,
    MVar,
    Node,
    Normalization,
    Operator,
    Signature,
    Term,
    TermStore,
)
from lofi.typecheck import Erasure

BASE_LANGUAGE: Final = registered_language("propositional")
LANGUAGE: Final = BASE_LANGUAGE.with_(
    atomic_types=(
        *BASE_LANGUAGE.atomic_types,
        *(
            AtomicTypeSpec(name=name, kind="individual")
            for name in ("n", "np", "s", "marker")
        ),
    )
)
CATEGORIES: Final = ccg_signature(
    "N",
    "NP",
    "S",
    "conj",
    operators=(
        OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
        OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
    ),
    conjunctions=ConjunctionsSpec(
        markers=("conj",),
        pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
    ),
    name="unary-integration-language",
)


def _compile(*specs: RuleSpec) -> AdoptedRules:
    """Compile a rule inventory over the shared category signature."""
    return declared_rules(
        CATEGORIES,
        RuleSetSpec(name="unary-integration", rules=specs),
        matching=FeatureErasureMatchingSpec(),
        language=LANGUAGE,
    )


ASSEMBLY: Final = _compile(ApplicationSpec(view="/")).context
CATEGORY_STORE: Final = ASSEMBLY.store
CATEGORY_SIGNATURE: Final = CATEGORY_STORE.signature
SYNTAX: Final = ASSEMBLY.syntax
N: Final = SYNTAX.parse(CATEGORY_STORE, "N")
NP: Final = SYNTAX.parse(CATEGORY_STORE, "NP")
S: Final = SYNTAX.parse(CATEGORY_STORE, "S")
S_DCL_BACK_NP: Final = SYNTAX.parse(CATEGORY_STORE, "S[dcl]\\NP")

TYPE_N: Final = CATEGORY_STORE.const("Ty", "n")
TYPE_NP: Final = CATEGORY_STORE.const("Ty", "np")
TYPE_S: Final = CATEGORY_STORE.const("Ty", "s")
TYPE_MARKER: Final = CATEGORY_STORE.const("Ty", "marker")
TYPE_ARROW: Final = CATEGORY_SIGNATURE.by_name("->")
TYPE_ARROW_IMAGE: Final = CATEGORY_STORE.op(
    TYPE_ARROW,
    CATEGORY_STORE.fvar("Ty", "a1"),
    CATEGORY_STORE.fvar("Ty", "a0"),
)
ERASURE: Final = Erasure(
    CATEGORY_STORE,
    {"N": TYPE_N, "NP": TYPE_NP, "S": TYPE_S, "conj": TYPE_MARKER},
    {"/": TYPE_ARROW_IMAGE, "\\": TYPE_ARROW_IMAGE},
    category_sort="Cat",
    function_type="->",
)

MEANING_SIGNATURE: Final = Signature(
    "unary-integration-meanings",
    ("Meaning",),
    (
        Operator("kim", "Meaning"),
        Operator("app", "Meaning", arg_sorts=("Meaning", "Meaning")),
        Operator(
            "lam",
            "Meaning",
            arg_sorts=("Meaning",),
            binding=(BindingSite((0,), "Meaning"),),
        ),
    ),
)
MEANING_STORE: Final = TermStore(MEANING_SIGNATURE)
KIM: Final = MEANING_STORE.op(MEANING_SIGNATURE.by_name("kim"))


class LambdaCalculus:
    """A lambda-calculus adapter for the fixture term store."""

    __slots__ = ()

    @property
    def store(self) -> TermStore:
        """Return the meaning store."""
        return MEANING_STORE

    @property
    def operators(self) -> Calculus:
        """Return the lambda-calculus operators."""
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        """Return an indexed hypothesis."""
        return self.free_variable(f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        """Return a free variable."""
        return self.store.fvar("Meaning", name)

    def bound_variable(self, index: int, /) -> Term:
        """Return a bound variable."""
        return self.store.bvar("Meaning", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        """Return an application."""
        return self.store.op(MEANING_SIGNATURE.by_name("app"), function, argument)

    def abstraction(
        self,
        variable: str,
        body: Term,
        /,
    ) -> Term:
        """Return an abstraction."""
        return self.store.bind("lam", variable, body)


CALCULUS: Final = LambdaCalculus()
NORMALIZATION: Final = Normalization(
    fuel=40,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)
N_PATTERN: Final = ConstantPatternSpec(name="N")
NP_PATTERN: Final = ConstantPatternSpec(name="NP")
N_TO_NP: Final = PatternRuleSpec(
    name="n-to-np-integration",
    premises=(N_PATTERN,),
    conclusion=NP_PATTERN,
    head_policy="pass-through",
)
TYPE_RAISING: Final = TypeRaisingSpec(view="/", targets=("S",), arguments=None)
CCGBANK_ADJUNCTS: Final = ccgbank_grammar().rules.rules[3:7]


def _build(spec: RuleSpec) -> Rule[Term]:
    """Build one low-level runtime rule fixture."""
    (rule,) = _compile(spec).rule_set
    return rule


def _derive(rule: Rule[Term]) -> tuple[DerivedAction, ...]:
    """Derive every action for one assembled rule."""
    schema = ASSEMBLY.proof_schema(rule)

    assert schema is not None

    return derive_action(
        schema,
        ERASURE,
        calculus=CALCULUS,
        normalization=Normalization(
            fuel=12,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
    )


def _category_node(name: str) -> Node:
    """Return a constant-pattern node from the declared category vocabulary."""
    identifier = CATEGORY_SIGNATURE.vocabulary("Cat").id(name)

    return Node("#const[Cat]", payload=identifier)


def test_public_unary_declarations_reach_semantic_reading() -> None:
    raising = _build(TYPE_RAISING)
    coercion = _build(N_TO_NP)
    adjuncts = tuple(_build(spec) for spec in CCGBANK_ADJUNCTS)
    rules = (raising, coercion, *adjuncts)
    derived = {rule.name: _derive(rule) for rule in rules}
    type_variable = CATEGORY_STORE.fvar("Ty", "X")
    raised_type = CATEGORY_STORE.op(
        TYPE_ARROW,
        CATEGORY_STORE.op(TYPE_ARROW, type_variable, TYPE_S),
        TYPE_S,
    )

    assert ASSEMBLY.proof_schema(raising) is raising
    assert ASSEMBLY.proof_schema(coercion) is coercion
    assert derived[raising.name][0].sequent == ((type_variable,), raised_type)
    assert derived[raising.name][0].inhabitants.exhausted == frozenset()
    assert len(derived[raising.name][0].inhabitants.terms) == 1
    assert derived[coercion.name][0].sequent == ((TYPE_N,), TYPE_NP)
    assert derived[coercion.name][0].inhabitants.terms == ()
    assert derived[coercion.name][0].inhabitants.exhausted == frozenset()

    expected_adjunct = (
        (CATEGORY_STORE.op(TYPE_ARROW, TYPE_NP, TYPE_S),),
        CATEGORY_STORE.op(TYPE_ARROW, type_variable, type_variable),
    )
    adjunct_names = frozenset(rule.name for rule in adjuncts)
    expected_adjunct_names = (
        "clausal-adjunct-backward-backward",
        "clausal-adjunct-backward-forward",
        "clausal-adjunct-forward-backward",
        "clausal-adjunct-forward-forward",
    )
    expected_slashes = (("\\", "\\"), ("\\", "/"), ("/", "\\"), ("/", "/"))

    assert tuple(rule.name for rule in adjuncts) == expected_adjunct_names

    for rule, (source_slash, target_slash) in zip(
        adjuncts,
        expected_slashes,
        strict=True,
    ):
        schema = ASSEMBLY.proof_schema(rule)

        assert schema is rule
        assert schema.proof_schemata == (
            (
                (Node(source_slash, (_category_node("S"), _category_node("NP"))),),
                Node(
                    target_slash,
                    (MVar("X", "Cat"), MVar("X", "Cat")),
                ),
            ),
        )
        (action,) = derived[rule.name]

        assert action.sequent == expected_adjunct
        assert action.inhabitants.terms == ()
        assert action.inhabitants.exhausted == frozenset()

    compiled = compile_meaning_actions(
        rules,
        {coercion.name: CALCULUS.hypothesis(0)},
        erasure=ERASURE,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
        selection=MeaningSelection("unique"),
    )
    reading = MeaningReading[Term](
        compiled,
        normalization=NORMALIZATION,
        policy="eager",
    )
    binder = "__proof_bound_0_0"
    expected_raised = CALCULUS.abstraction(
        binder,
        CALCULUS.application(CALCULUS.free_variable(binder), KIM),
    )

    assert compiled.uncovered == adjunct_names
    assert compiled.ambiguous == frozenset()
    assert set(compiled.rules) == {raising.name, coercion.name}
    kim_np = Entry("Kim", NP, KIM)
    kim_n = Entry("Kim", N, KIM)
    clause = Entry("clause", S_DCL_BACK_NP, KIM)
    lexicon = FormLexicon((raising, coercion, *adjuncts), (kim_np, kim_n, clause))

    assert read(
        step(raising, (axiom(kim_np),), next(raising.apply((NP,))), 0.0),
        reading,
        lexicon,
    ) == (expected_raised,)
    assert read(step(coercion, (axiom(kim_n),), NP, 0.0), reading, lexicon) == (KIM,)

    for rule, (_, target_slash) in zip(adjuncts, expected_slashes, strict=True):
        assert read(
            step(
                rule,
                (axiom(clause),),
                SYNTAX.parse(CATEGORY_STORE, f"S{target_slash}S"),
                0.0,
            ),
            reading,
            lexicon,
        ) == (None,)


@settings(max_examples=100, deadline=None)
@given(st.lists(st.integers(min_value=1, max_value=3), max_size=12))
def test_generic_fold_handles_unary_binary_and_ternary_steps(
    additional_arities: list[int],
) -> None:
    unary = _build(N_TO_NP)
    binary = _build(ApplicationSpec(view="/"))
    ternary = _build(
        PatternRuleSpec(
            name="n-conj-n-integration",
            premises=(N_PATTERN, N_PATTERN, N_PATTERN),
            conclusion=N_PATTERN,
            head_policy="left-conjunct",
        )
    )
    rules: dict[int, Rule[Term]] = {
        1: unary,
        2: binary,
        3: ternary,
    }
    entries = [Entry("x0", N, "x0")]
    tree = axiom(entries[0])
    serial = 1

    for arity in (1, 2, 3, *additional_arities):
        extra = tuple(
            Entry(f"x{at}", N, f"x{at}") for at in range(serial, serial + arity - 1)
        )
        entries.extend(extra)
        tree = step(
            rules[arity],
            (tree, *(axiom(entry, at) for at, entry in enumerate(extra, serial))),
            N,
            0.0,
        )
        serial += arity - 1

    lexicon = FormLexicon(rules.values(), entries)
    expected = tuple(entry.form for entry in entries)

    assert read(tree, StringReading[Term, str](), lexicon) == (expected,)
