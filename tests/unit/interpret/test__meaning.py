from copy import copy, deepcopy
from gc import get_referents
from typing import TYPE_CHECKING, Final, Literal, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance._suite import Conformance, ConformanceSuite
from tests.conformance.interpretation import InterpretationSuite
from tests.support.derivations import FormLexicon, axiom, step
from tests.support.rules import declared_rules

from lofi.interpret import (
    NORMALIZATION_POLICIES,
    NORMALIZES_STEPS,
    AmbiguousMeaningError,
    CategoryWitnessRequiredError,
    InhabitantSelection,
    Interpretation,
    MeaningActions,
    MeaningReading,
    MeaningSelection,
    NormalizationPolicy,
    StepContext,
    compile_meaning_actions,
    read,
    run_action,
)
from lofi.interpret.laws import InterpretationFixture
from lofi.judge import (
    PROOF_LOGICS,
    PatternRule,
    ProofCalculus,
    ProofSearch,
    Rule,
)
from lofi.lexicon import Entry
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import FunctorSpec, SignatureSpec
from lofi.terms import (
    BindingSite,
    Calculus,
    MVar,
    NonTerminatingError,
    Normalization,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
    normalize,
)
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from lofi.grammars.ccg import CCGAssembly
    from lofi.judge import PatternRule
    from lofi.spec import PatternSpec, RuleSpec


def _constructor(name: str, spelling: str) -> ConstructorSpec:
    """Declare one binary category constructor."""
    return ConstructorSpec(
        name=name,
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling=spelling, precedence=10, associativity="left"
        ),
    )


DIRECTED: Final = SignatureSpec(
    name="meaning-language",
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
UNDIRECTED: Final = SignatureSpec(
    name="meaning-undirected",
    primitives=("x", "y", "z"),
    constructors=(_constructor("fwd", "/"),),
    functors=(
        FunctorSpec(
            name="fwd", constructor="fwd", argument_side="either", opposite="fwd"
        ),
    ),
)


def _rule(spec: RuleSpec, /, *, signature: SignatureSpec = DIRECTED) -> Rule[Term]:
    """Compile one rule over a category signature."""
    (rule,) = declared_rules(
        signature, RuleSetSpec(name="meaning", rules=(spec,))
    ).rule_set

    return rule


def _category(name: str, /) -> TermPatternSpec:
    """Return a category variable pattern."""
    return TermPatternSpec(name=name, sort="Cat")


def _node(slash: str, /, *arguments: PatternSpec) -> NodePatternSpec:
    """Return a functor pattern over one constructor."""
    return NodePatternSpec(operator=FixedOperatorSpec(name=slash), arguments=arguments)


def pattern_rule(
    name: str, premises: tuple[PatternSpec, ...], conclusion: PatternSpec, /
) -> PatternRule:
    """Compile one pattern rule over the directed categories."""
    return cast(
        "PatternRule",
        _rule(PatternRuleSpec(name=name, premises=premises, conclusion=conclusion)),
    )


def _composition(*, signature: SignatureSpec = DIRECTED) -> Rule[Term]:
    """Compile forward composition at every finite order."""
    return _rule(
        CompositionSpec(view="fwd", order=None, direction="unrestricted"),
        signature=signature,
    )


def _context(signature: SignatureSpec, /) -> CCGAssembly:
    """Return the compiled rule context of a category signature."""
    return declared_rules(
        signature, RuleSetSpec(name="meaning", rules=(ApplicationSpec(view="fwd"),))
    ).context


def _erasure(context: CCGAssembly, /) -> Erasure:
    """Erase each category of a compiled context to a type of its store."""
    store = context.store
    arrow = store.op(
        store.signature.by_name("->"),
        store.fvar("Ty", "a1"),
        store.fvar("Ty", "a0"),
    )

    return Erasure(
        store,
        {
            "x": store.const("Ty", "e"),
            "y": store.const("Ty", "v"),
            "z": store.const("Ty", "t"),
        },
        dict.fromkeys(context.functors, arrow),
        category_sort="Cat",
        function_type="->",
    )


DIRECTED_CONTEXT: Final = _context(DIRECTED)
UNDIRECTED_CONTEXT: Final = _context(UNDIRECTED)
LANGUAGE_STORE: Final = DIRECTED_CONTEXT.store

MEANING_SIGNATURE: Final = Signature(
    "meaning-terms",
    ("Meaning",),
    (
        Operator("app", "Meaning", arg_sorts=("Meaning", "Meaning")),
        Operator(
            "lam",
            "Meaning",
            arg_sorts=("Meaning",),
            binding=(BindingSite((0,), "Meaning"),),
        ),
    ),
    vocabularies=(Vocabulary("Meaning", ("a", "foreign", "k")),),
)
MEANING_STORE: Final = TermStore(MEANING_SIGNATURE)
FOREIGN_MEANING_STORE: Final = TermStore(MEANING_SIGNATURE)
ERASURE: Final = _erasure(DIRECTED_CONTEXT)
UNDIRECTED_ERASURE: Final = _erasure(UNDIRECTED_CONTEXT)


class LambdaCalculus:
    """A lambda-calculus adapter for the fixture term store."""

    @property
    def store(self) -> TermStore:
        return MEANING_STORE

    @property
    def operators(self) -> Calculus:
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        return MEANING_STORE.fvar("Meaning", f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        return MEANING_STORE.fvar("Meaning", name)

    def bound_variable(self, index: int, /) -> Term:
        return MEANING_STORE.bvar("Meaning", index)

    def application(
        self,
        function: Term,
        argument: Term,
        /,
    ) -> Term:
        return MEANING_STORE.op(
            MEANING_SIGNATURE.by_name("app"),
            function,
            argument,
        )

    def abstraction(
        self,
        variable: str,
        body: Term,
        /,
    ) -> Term:
        return MEANING_STORE.bind("lam", variable, body)


CALCULUS: Final[ProofCalculus] = LambdaCalculus()
NORMALIZATION: Final = Normalization(
    fuel=40,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)


FUNCTORS: Final = DIRECTED_CONTEXT.functors["fwd"][0]
UNDIRECTED_FUNCTORS: Final = UNDIRECTED_CONTEXT.functors["fwd"][0]
IDENTITY_RULE: Final = pattern_rule("identity", (_category("X"),), _category("X"))
FORWARD_APPLICATION: Final = pattern_rule(
    "forward-application",
    (_node("fwd", _category("X"), _category("Y")), _category("Y")),
    _category("X"),
)
BACKWARD_APPLICATION: Final = pattern_rule(
    "backward-application",
    (_category("Y"), _node("bwd", _category("X"), _category("Y"))),
    _category("X"),
)
X: Final = LANGUAGE_STORE.const("Cat", "x")
A: Final = MEANING_STORE.const("Meaning", "a")
K: Final = MEANING_STORE.const("Meaning", "k")
SEARCH: Final = ProofSearch(
    contraction=True, weakening=False, max_uses=1, max_inhabitants=64
)
UNIQUE_SELECTION: Final = MeaningSelection("unique")


def _identity_rule(name: str) -> PatternRule:
    variable = _category("X")
    return pattern_rule(name, (variable,), variable)


def _compile(
    rules: tuple[Rule[Term], ...],
    supplied: dict[str, Term] | None = None,
    *,
    selection: MeaningSelection = UNIQUE_SELECTION,
    search: ProofSearch = SEARCH,
    erasure: Erasure = ERASURE,
) -> MeaningActions:
    return compile_meaning_actions(
        rules,
        supplied,
        erasure=erasure,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        search=search,
        selection=selection,
    )


def _meaning_actions(actions: dict[str, Term]) -> MeaningActions:
    known: dict[str, Rule[Term]] = {
        IDENTITY_RULE.name: IDENTITY_RULE,
        FORWARD_APPLICATION.name: FORWARD_APPLICATION,
        BACKWARD_APPLICATION.name: BACKWARD_APPLICATION,
    }
    return _compile(
        tuple(known.get(name, _identity_rule(name)) for name in actions),
        actions,
    )


def test_entry_and_derivation_keep_category_and_meaning_carriers_distinct() -> None:
    entry: Entry[Term, Term] = Entry("a", X, A)
    actions = _compile((IDENTITY_RULE,))
    reading = MeaningReading[Term](
        actions,
        normalization=NORMALIZATION,
        policy="eager",
    )
    interpretation: Interpretation[
        Term,
        Term,
        Term,
    ] = reading
    derivation = step(IDENTITY_RULE, (axiom(entry),), X, 0.0)
    (meaning,) = read(
        derivation, interpretation, FormLexicon((IDENTITY_RULE,), (entry,))
    )

    assert meaning is A


def test_one_carrier_entry_annotation_uses_the_default() -> None:
    entry: Entry[str] = Entry("a", "NP", "a")

    assert entry.meaning == "a"


def test_compile_records_gaps_and_ambiguity_before_applying_overrides() -> None:
    p0 = CALCULUS.hypothesis(0)
    p1 = CALCULUS.hypothesis(1)
    binary = CALCULUS.application(p0, p1)
    x = _category("X")
    y = _category("Y")
    rules = (
        _identity_rule("unique"),
        pattern_rule("uncovered", (x,), y),
        pattern_rule("ambiguous", (x, x), x),
        pattern_rule("limited", (x, x), x),
        pattern_rule("supplied-only", (x,), y),
    )
    actions = _compile(
        rules,
        {"ambiguous": binary, "supplied-only": p0},
        search=ProofSearch(
            contraction=False, weakening=True, max_uses=1, max_inhabitants=1
        ),
    )

    assert actions.rules == ("unique", "ambiguous", "supplied-only")
    assert actions.resolve("unique", None) == (p0,)
    assert actions.resolve("ambiguous", None) == (binary,)
    assert actions.resolve("supplied-only", None) == (p0,)
    assert actions.uncovered == frozenset(("uncovered", "limited"))
    assert actions.ambiguous == frozenset()
    assert actions.exhausted == {"limited": frozenset(("inhabitants",))}


def test_compile_refuses_an_override_for_an_absent_rule() -> None:
    with pytest.raises(ValueError, match=r"absent.*ghost"):
        _compile((IDENTITY_RULE,), {"ghost": CALCULUS.hypothesis(0)})


def test_compile_refuses_a_foreign_supplied_action() -> None:
    foreign = FOREIGN_MEANING_STORE.fvar("Meaning", "p0")

    with pytest.raises(ValueError, match="foreign to the calculus"):
        _compile((IDENTITY_RULE,), {IDENTITY_RULE.name: foreign})


def test_compile_refuses_a_supplied_action_with_the_wrong_hypotheses() -> None:
    with pytest.raises(ValueError, match=r"hypothesis mask.*arity 1"):
        _compile((IDENTITY_RULE,), {IDENTITY_RULE.name: A})


def test_compile_refuses_duplicate_rule_names() -> None:
    duplicate = _identity_rule(IDENTITY_RULE.name)

    with pytest.raises(ValueError, match=r"received rule.*twice"):
        _compile((IDENTITY_RULE, duplicate))


@pytest.mark.parametrize(
    ("kind", "selected"),
    [
        ("unique", ()),
        ("first", (K,)),
        ("smallest", (A,)),
        ("all", (K, A)),
    ],
)
def test_inhabitant_selection_is_declared(
    kind: Literal["unique", "first", "smallest", "all"],
    selected: tuple[Term, ...],
) -> None:
    larger = CALCULUS.application(K, A)
    candidates = (larger, A) if kind == "smallest" else (K, A)
    assert MeaningSelection(kind).select(candidates) == selected


def test_compile_refuses_overlapping_schematic_sequents() -> None:
    left = MVar("Left", "Cat")
    right = MVar("Right", "Cat")

    class OverlappingRule:
        name = "overlap"
        arity = 1
        proof_schemata = ((((left,), left)), (((right,), right)))

    actions = compile_meaning_actions(
        (cast("Rule[Term]", OverlappingRule()),),
        erasure=ERASURE,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        search=SEARCH,
        selection=MeaningSelection("all"),
    )

    assert actions.rules == ()
    assert actions.ambiguous == frozenset(("overlap",))
    assert actions.uncovered == frozenset(("overlap",))


def test_sequent_resolution_preserves_repeated_variable_constraints() -> None:
    variable = _category("Same")
    rule = pattern_rule(
        "repeat",
        (variable, variable),
        variable,
    )
    actions = _compile(
        (rule,),
        selection=MeaningSelection("first"),
        search=ProofSearch(
            contraction=False, weakening=True, max_uses=1, max_inhabitants=64
        ),
    )
    matching = StepContext(rule, X, 0.0, 0, (X, X))
    mismatching = StepContext(
        rule,
        X,
        0.0,
        0,
        (X, LANGUAGE_STORE.const("Cat", "y")),
    )

    assert actions.resolve("repeat", matching) == (CALCULUS.hypothesis(0),)
    assert actions.resolve("repeat", mismatching) == ()


@pytest.mark.parametrize(
    ("logic", "selection", "covered", "uncovered", "ambiguous"),
    [
        ("relevant", "unique", 0, 2, 0),
        ("linear", "all", 0, 2, 0),
        ("affine", "unique", 1, 1, 1),
        ("intuitionistic", "first", 2, 0, 1),
        ("intuitionistic", "smallest", 2, 0, 1),
        ("intuitionistic", "all", 2, 0, 1),
    ],
)
def test_logic_and_selection_sweep_changes_action_coverage(
    logic: Literal["relevant", "linear", "affine", "intuitionistic"],
    selection: Literal["unique", "first", "smallest", "all"],
    covered: int,
    uncovered: int,
    ambiguous: int,
) -> None:
    x = _category("X")
    y = _category("Y")
    search = ProofSearch(
        contraction=PROOF_LOGICS[logic].contraction,
        weakening=PROOF_LOGICS[logic].weakening,
        max_uses=1,
        max_inhabitants=64,
    )
    actions = _compile(
        (
            pattern_rule("weakening", (x, y), y),
            pattern_rule("choice", (x, x), x),
        ),
        selection=MeaningSelection(selection),
        search=search,
    )

    assert len(actions) == covered
    assert len(actions.uncovered) == uncovered
    assert len(actions.ambiguous) == ambiguous


def test_meaning_actions_are_frozen_snapshots() -> None:
    source = {"identity": CALCULUS.hypothesis(0)}
    actions = _compile((_identity_rule("identity"),), source)
    source.clear()

    assert actions.rules == ("identity",)
    assert copy(actions) is actions
    assert deepcopy(actions) is actions
    assert actions == _meaning_actions({"identity": CALCULUS.hypothesis(0)})
    assert hash(actions) == hash(_meaning_actions({"identity": CALCULUS.hypothesis(0)}))

    field = "_programs"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(actions, field, {})


def test_run_action_substitutes_all_children_simultaneously() -> None:
    p0 = CALCULUS.hypothesis(0)
    p1 = CALCULUS.hypothesis(1)
    action = CALCULUS.application(p0, p1)

    result = run_action(
        action,
        (p1, A),
        calculus=CALCULUS,
        normalization=NORMALIZATION,
    )

    assert result is CALCULUS.application(p1, A)


@pytest.mark.parametrize(
    ("action", "children"),
    [
        (CALCULUS.hypothesis(0), ()),
        (
            CALCULUS.application(CALCULUS.hypothesis(0), CALCULUS.hypothesis(1)),
            (A,),
        ),
    ],
)
def test_run_action_rejects_a_child_arity_mismatch(
    action: Term,
    children: tuple[Term, ...],
) -> None:
    with pytest.raises(ValueError, match="child arity"):
        run_action(
            action,
            children,
            calculus=CALCULUS,
            normalization=NORMALIZATION,
        )


def test_run_action_rejects_foreign_actions_and_children() -> None:
    foreign = FOREIGN_MEANING_STORE.const("Meaning", "foreign")

    with pytest.raises(ValueError, match="store lineage"):
        run_action(foreign, (), calculus=CALCULUS, normalization=NORMALIZATION)

    with pytest.raises(ValueError, match="store lineage"):
        run_action(
            CALCULUS.hypothesis(0),
            (foreign,),
            calculus=CALCULUS,
            normalization=NORMALIZATION,
        )


class RepeatedHypotheses(LambdaCalculus):
    """A calculus with one hypothesis shared across positions."""

    @override
    def hypothesis(self, index: int, /) -> Term:
        del index

        return MEANING_STORE.fvar("Meaning", "repeated")


def test_run_action_rejects_a_noninjective_hypothesis_channel() -> None:
    repeated = RepeatedHypotheses()
    action = repeated.hypothesis(0)

    with pytest.raises(ValueError, match="not injective"):
        run_action(action, (A,), calculus=repeated, normalization=NORMALIZATION)


class UnstableHypotheses(LambdaCalculus):
    """A calculus whose hypothesis changes on each request."""

    def __init__(self) -> None:
        self._next = 0

    @override
    def hypothesis(self, index: int, /) -> Term:
        term = MEANING_STORE.fvar("Meaning", f"unstable-{index}-{self._next}")
        self._next += 1

        return term


def test_run_action_rejects_an_unstable_hypothesis_channel() -> None:
    unstable = UnstableHypotheses()
    action = unstable.hypothesis(0)

    with pytest.raises(ValueError, match="not stable"):
        run_action(action, (A,), calculus=unstable, normalization=NORMALIZATION)


def test_the_reading_returns_none_for_a_missing_meaning_or_action() -> None:
    reading = MeaningReading[Term](
        _compile(()),
        normalization=NORMALIZATION,
        policy="eager",
    )

    empty = Entry[Term, Term]("empty", X)
    context = StepContext(IDENTITY_RULE, X, 0.0, 0, (X,))

    assert reading.axiom(empty) is None
    assert reading.step(context, (A,)) is None
    assert not reading.supports(IDENTITY_RULE)


def test_the_reading_rejects_a_foreign_lexical_meaning() -> None:
    reading = MeaningReading[Term](
        _compile(()),
        normalization=NORMALIZATION,
        policy="eager",
    )
    foreign = FOREIGN_MEANING_STORE.const("Meaning", "foreign")

    with pytest.raises(ValueError, match="store lineage"):
        reading.axiom(Entry("foreign", X, foreign))


def test_the_reading_reports_its_identity_and_coverage() -> None:
    reading = MeaningReading[Term](
        _meaning_actions({IDENTITY_RULE.name: CALCULUS.hypothesis(0)}),
        normalization=NORMALIZATION,
        policy="eager",
    )

    assert reading.name == "meaning"
    assert reading.refines_items
    assert reading.supports(IDENTITY_RULE)
    assert reading.normalization is NORMALIZATION
    assert repr(reading) == (
        f"MeaningReading(actions=1, policy='eager', normalization={NORMALIZATION!r})"
    )


def test_a_scalar_reading_names_plural_selected_actions_as_ambiguous() -> None:
    variable = _category("X")
    rule = pattern_rule(
        "plural",
        (variable, variable),
        variable,
    )
    actions = _compile(
        (rule,),
        selection=MeaningSelection("all"),
        search=ProofSearch(
            contraction=False, weakening=True, max_uses=1, max_inhabitants=64
        ),
    )
    reading = MeaningReading[Term](
        actions,
        normalization=NORMALIZATION,
        policy="eager",
    )
    context = StepContext(rule, X, 0.0, 0, (X, X))

    assert not reading.supports(rule)

    with pytest.raises(AmbiguousMeaningError, match="requires exactly one"):
        reading.step(context, (A, K))


def _redex_reading(
    policy: NormalizationPolicy,
) -> MeaningReading[Term]:
    """Build a reading whose one action applies its first child to its second."""
    action = CALCULUS.application(CALCULUS.hypothesis(0), CALCULUS.hypothesis(1))

    return MeaningReading[Term](
        _meaning_actions({FORWARD_APPLICATION.name: action}),
        normalization=NORMALIZATION,
        policy=policy,
    )


def _applied(
    reading: MeaningReading[Term],
) -> Term | None:
    """Read the identity function applied to a constant under a reading."""
    identity = CALCULUS.abstraction("x", CALCULUS.free_variable("x"))
    functor = FUNCTORS.make("fwd", X, X)
    entries = (Entry("id", functor, identity), Entry("a", X, A))
    derivation = step(
        FORWARD_APPLICATION,
        (axiom(entries[0]), axiom(entries[1], 1)),
        X,
        0.0,
    )
    (meaning,) = read(derivation, reading, FormLexicon((FORWARD_APPLICATION,), entries))

    return meaning


def test_the_lazy_policy_keeps_the_redex_the_eager_policy_contracts() -> None:
    identity = CALCULUS.abstraction("x", CALCULUS.free_variable("x"))
    eager = _applied(_redex_reading("eager"))
    lazy = _applied(_redex_reading("lazy"))

    assert eager is A
    assert lazy is CALCULUS.application(identity, A)
    assert lazy is not eager


def test_the_declared_fuel_and_policy_control_normalization_refusal() -> None:
    variable = CALCULUS.bound_variable(0)
    self_application = CALCULUS.application(variable, variable)
    abstraction = CALCULUS.abstraction("x", self_application)
    action = CALCULUS.application(CALCULUS.hypothesis(0), CALCULUS.hypothesis(0))
    settings = Normalization(
        fuel=1,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )
    context = StepContext(IDENTITY_RULE, X, 0.0, 0, (X,))
    eager = MeaningReading[Term](
        _meaning_actions({IDENTITY_RULE.name: action}),
        normalization=settings,
        policy="eager",
    )
    lazy = MeaningReading[Term](
        _meaning_actions({IDENTITY_RULE.name: action}),
        normalization=settings,
        policy="lazy",
    )

    with pytest.raises(NonTerminatingError):
        eager.step(context, (abstraction,))

    assert lazy.step(context, (abstraction,)) is CALCULUS.application(
        abstraction, abstraction
    )


def test_the_two_policies_agree_up_to_normalization_by_pointer_equality() -> None:
    eager = _applied(_redex_reading("eager"))
    lazy = _applied(_redex_reading("lazy"))

    assert lazy is not None
    assert (
        normalize(
            MEANING_STORE,
            lazy,
            CALCULUS.operators,
            normalization=NORMALIZATION,
        )
        is eager
    )


def test_the_lazy_policy_reports_itself() -> None:
    reading = _redex_reading("lazy")

    assert reading.policy == "lazy"
    assert repr(reading) == (
        f"MeaningReading(actions=1, policy='lazy', normalization={NORMALIZATION!r})"
    )


def test_an_undeclared_policy_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="normalization policy"):
        MeaningReading[Term](
            _compile(()),
            normalization=NORMALIZATION,
            policy=cast("NormalizationPolicy", "sometimes"),
        )


def _spine(
    base: Term, order: int, /, *, context: CCGAssembly = DIRECTED_CONTEXT
) -> Term:
    """Return a forward result spine of one composition order."""
    functors, slash = context.functors["fwd"]
    built = base

    for _ in range(order):
        built = functors.make(slash, built, context.store.const("Cat", "z"))

    return built


def _composition_step(
    order: int, /, *, context: CCGAssembly = DIRECTED_CONTEXT
) -> tuple[tuple[Term, Term], Term]:
    """Return one forward-composition step of the requested order."""
    functors, slash = context.functors["fwd"]
    x = context.store.const("Cat", "x")
    y = context.store.const("Cat", "y")
    principal = functors.make(slash, x, y)
    container = _spine(y, order, context=context)

    return (principal, container), _spine(x, order, context=context)


@pytest.mark.parametrize("order", [1, 2, 3])
def test_unbounded_composition_resolves_each_witnessed_order(order: int) -> None:
    rule = _composition()
    actions = _compile((rule,))
    premises, conclusion = _composition_step(order)
    context = StepContext(rule, conclusion, 0.0, 0, premises)
    reading = MeaningReading[Term](
        actions,
        normalization=NORMALIZATION,
        policy="lazy",
    )

    assert actions.rules == (rule.name,)
    assert len(actions) == 1
    assert actions.uncovered == frozenset()
    assert actions.ambiguous == frozenset()
    assert actions.exhausted == {}
    assert len(actions.resolve(rule.name, context)) == 1
    assert reading.supports(rule)
    assert reading.step(context, (A, K)) is not None

    with pytest.raises(CategoryWitnessRequiredError, match="concrete derivation-step"):
        actions.resolve(rule.name, None)


def test_unbounded_composition_returns_no_action_for_an_invalid_step() -> None:
    rule = _composition()
    actions = _compile((rule,))
    premises, _ = _composition_step(2)
    context = StepContext(
        rule,
        LANGUAGE_STORE.const("Cat", "z"),
        0.0,
        0,
        premises,
    )
    reading = MeaningReading[Term](
        actions,
        normalization=NORMALIZATION,
        policy="eager",
    )

    assert actions.resolve(rule.name, context) == ()
    assert reading.step(context, (A, K)) is None


def test_a_supplied_action_replaces_witness_elaboration() -> None:
    rule = _composition()
    override = CALCULUS.application(CALCULUS.hypothesis(0), CALCULUS.hypothesis(1))
    actions = _compile((rule,), {rule.name: override})
    premises, conclusion = _composition_step(3)
    context = StepContext(rule, conclusion, 0.0, 0, premises)

    assert actions.resolve(rule.name, None) == (override,)
    assert actions.resolve(rule.name, context) == (override,)


def test_an_empty_schema_without_a_step_witness_remains_uncovered() -> None:
    class EmptySchema:
        name = "empty-schema"
        arity = 1
        proof_schemata = ()

    actions = compile_meaning_actions(
        (cast("Rule[Term]", EmptySchema()),),
        erasure=ERASURE,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        search=SEARCH,
        selection=UNIQUE_SELECTION,
    )

    assert actions.rules == ()
    assert actions.uncovered == frozenset(("empty-schema",))


def test_a_self_opposite_witness_resolves_both_physical_orientations() -> None:
    rule = _composition(signature=UNDIRECTED)
    actions = _compile((rule,), erasure=UNDIRECTED_ERASURE)
    premises, conclusion = _composition_step(2, context=UNDIRECTED_CONTEXT)
    forward = StepContext(rule, conclusion, 0.0, 0, premises)
    backward = StepContext(rule, conclusion, 0.0, 0, tuple(reversed(premises)))
    forward_terms = actions.resolve(rule.name, forward)
    backward_terms = actions.resolve(rule.name, backward)

    assert len(forward_terms) == 1
    assert len(backward_terms) == 1
    assert forward_terms != backward_terms


@pytest.mark.parametrize(
    ("selection", "count"),
    [
        ("unique", 0),
        ("first", 1),
        ("smallest", 1),
        ("all", 2),
    ],
)
def test_witness_inhabitant_selection_controls_coincident_self_opposites(
    selection: InhabitantSelection,
    count: int,
) -> None:
    rule = _composition(signature=UNDIRECTED)
    x = UNDIRECTED_CONTEXT.store.const("Cat", "x")
    coincident = UNDIRECTED_FUNCTORS.make("fwd", x, x)
    premises = (coincident, coincident)
    context = StepContext(rule, coincident, 0.0, 0, premises)
    actions = _compile(
        (rule,), selection=MeaningSelection(selection), erasure=UNDIRECTED_ERASURE
    )
    selected = actions.resolve(rule.name, context)
    reading = MeaningReading[Term](
        actions,
        normalization=NORMALIZATION,
        policy="eager",
    )

    assert len(selected) == count
    scalar = selection != "all"
    assert actions.is_scalar(rule.name) is scalar
    assert actions.ambiguous == frozenset((rule.name,))
    assert reading.supports(rule) is scalar

    if selection == "all":
        assert selected[0] is not selected[1]

        with pytest.raises(AmbiguousMeaningError, match="resolves to 2"):
            reading.step(context, (A, K))


type _DrawnRule = PatternRule
"""A pattern rule a Hypothesis draw compiles."""


@st.composite
def _rule_shapes(draw: st.DrawFn) -> _DrawnRule:
    """Return a varied real pattern rule."""
    arity = draw(st.integers(min_value=1, max_value=3))
    variables = tuple(_category(f"X{at}") for at in range(arity))
    operator = draw(st.sampled_from(("fwd", "bwd")))
    left = draw(st.integers(min_value=0, max_value=arity - 1))
    right = draw(st.integers(min_value=0, max_value=arity - 1))
    premises: list[PatternSpec] = list(variables)

    if draw(st.booleans()):
        wrapped = draw(st.integers(min_value=0, max_value=arity - 1))
        premises[wrapped] = _node(operator, variables[wrapped], variables[right])

    conclusion: PatternSpec

    if draw(st.booleans()):
        conclusion = variables[draw(st.integers(min_value=0, max_value=arity - 1))]
    else:
        conclusion = _node(operator, variables[left], variables[right])

    name = draw(st.text(alphabet="abcdefghijklmnopqrstuvwxyz", min_size=1, max_size=12))

    return pattern_rule(name, tuple(premises), conclusion)


@settings(max_examples=100, deadline=None)
@given(_rule_shapes())
def test_rule_object_graphs_keep_interpretations_outside_rules(
    rule: _DrawnRule,
) -> None:
    stored = get_referents(rule)

    for value in tuple(stored):
        if isinstance(value, tuple | frozenset | list | dict):
            stored.extend(get_referents(value))

    assert not isinstance(rule, Interpretation)
    assert not any(isinstance(value, Interpretation) for value in stored)


def test_golden_actions_beta_reduce_for_application_composition_and_raising() -> None:
    identity = CALCULUS.abstraction("x", CALCULUS.free_variable("x"))
    constant = CALCULUS.abstraction("x", K)
    composition = _rule(CompositionSpec(view="fwd", order=1, direction="unrestricted"))
    raising = _rule(TypeRaisingSpec(view="fwd", targets=("x",), arguments=None))
    functor = FUNCTORS.make("fwd", X, X)
    backward = FUNCTORS.make("bwd", X, X)
    raised = next(iter(raising.apply((X,))))
    rules_and_children = (
        (FORWARD_APPLICATION, (functor, X), X, (identity, A), A),
        (BACKWARD_APPLICATION, (X, backward), X, (A, identity), A),
        (composition, (functor, functor), functor, (constant, identity), constant),
        (
            raising,
            (X,),
            raised,
            (A,),
            CALCULUS.abstraction(
                "f",
                CALCULUS.application(CALCULUS.free_variable("f"), A),
            ),
        ),
    )

    for rule, premises, conclusion, children, expected in rules_and_children:
        compiled = _compile((rule,))
        reading = MeaningReading[Term](
            compiled,
            normalization=NORMALIZATION,
            policy="eager",
        )
        context = StepContext(rule, conclusion, 0.0, 0, tuple(premises))

        assert reading.step(context, children) is expected


@pytest.fixture(params=NORMALIZATION_POLICIES)
def interpretation_conformance(
    request: pytest.FixtureRequest,
) -> InterpretationFixture[
    Term,
    Term,
    Term,
]:
    entry = Entry("a", X, A)
    other = Entry("k", X, K)
    derivation = step(IDENTITY_RULE, (axiom(entry),), X, 0.0)
    reading = MeaningReading[Term](
        _meaning_actions({IDENTITY_RULE.name: CALCULUS.hypothesis(0)}),
        normalization=NORMALIZATION,
        policy=request.param,
    )

    return InterpretationFixture(
        reading,
        FormLexicon((IDENTITY_RULE,), (entry, other)),
        st.just(derivation),
        st.just((axiom(entry), axiom(other))),
    )


class TestMeaningReading(InterpretationSuite):
    """Interpretation-law tests for MeaningReading."""


class TestMeaningActions(ConformanceSuite):
    """Value-law tests for MeaningActions."""

    @pytest.fixture
    def conformance(self) -> Conformance[MeaningActions]:
        p0 = CALCULUS.hypothesis(0)

        return Conformance(
            build=lambda: _meaning_actions({"identity": p0}),
            unequal=_meaning_actions({"identity": CALCULUS.application(p0, p0)}),
            operations=(
                ("rules", lambda actions: actions.rules),
                ("scalar", lambda actions: actions.is_scalar("identity")),
                ("uncovered", lambda actions: actions.uncovered),
                ("ambiguous", lambda actions: actions.ambiguous),
                (
                    "exhausted",
                    lambda actions: tuple(
                        sorted(
                            (name, tuple(sorted(bounds)))
                            for name, bounds in actions.exhausted.items()
                        )
                    ),
                ),
                (
                    "resolve",
                    lambda actions: tuple(
                        repr(term) for term in actions.resolve("identity", None)
                    ),
                ),
            ),
        )


def test_each_declared_policy_says_whether_rule_applications_are_normalized() -> None:
    assert NORMALIZES_STEPS == {"eager": True, "lazy": False}
    assert set(NORMALIZES_STEPS) == set(NORMALIZATION_POLICIES)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [("unique", ()), ("first", (A,)), ("smallest", (A,)), ("all", (A, K))],
)
def test_each_declared_selection_keeps_its_inhabitants(
    kind: InhabitantSelection, expected: tuple[Term, ...]
) -> None:
    assert MeaningSelection(kind).select((A, K)) == expected
    assert MeaningSelection(kind).select(()) == ()
