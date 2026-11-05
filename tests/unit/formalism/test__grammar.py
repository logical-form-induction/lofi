from copy import copy, deepcopy
from hashlib import sha256
from typing import Final, cast

import pytest
from tests.support.rules import declared_rules

from lofi.deduce import CoverGoal, RootGoal
from lofi.formalism._grammar import Formalism, Grammar
from lofi.formalism._profile import (
    ProfileBound,
    ProfileUnsupportedError,
)
from lofi.formalism._values import Budget, FormalismDeclaration, TheoryArtifact
from lofi.judge import (
    ClosureReport,
    NoNormalForm,
    Rule,
    RuleSet,
    capacity_report,
)
from lofi.lexicon import Clamp
from lofi.spec import (
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar import SignatureSpec
from lofi.terms import (
    Identity,
    TargetSet,
    Term,
    TermStore,
)


def _node(name: str, *arguments: str) -> NodePatternSpec:
    return NodePatternSpec(
        operator=FixedOperatorSpec(name=name),
        arguments=tuple(
            TermPatternSpec(name=argument, sort="Cat") for argument in arguments
        ),
    )


ADOPTED: Final = declared_rules(
    SignatureSpec(
        name="formalism-test",
        primitives=("A", "B"),
        constructors=(
            ConstructorSpec(
                name="pair",
                kind=("Cat", "Cat", "Cat"),
                notation=InfixConstructorNotationSpec(
                    spelling="*", precedence=10, associativity="left"
                ),
            ),
        ),
    ),
    RuleSetSpec(
        name="formalism-test",
        rules=(
            PatternRuleSpec(
                name="unary",
                premises=(ConstantPatternSpec(name="A"),),
                conclusion=ConstantPatternSpec(name="B"),
            ),
            PatternRuleSpec(
                name="binary",
                premises=(
                    TermPatternSpec(name="X", sort="Cat"),
                    TermPatternSpec(name="Y", sort="Cat"),
                ),
                conclusion=_node("pair", "X", "Y"),
            ),
            PatternRuleSpec(
                name="axiom",
                premises=(),
                conclusion=ConstantPatternSpec(name="A"),
            ),
        ),
    ),
)
STORE: Final = ADOPTED.context.store
SIGNATURE: Final = STORE.signature
SYNTAX: Final = ADOPTED.context.syntax
A: Final = SYNTAX.parse(STORE, "A")
B: Final = SYNTAX.parse(STORE, "B")
UNARY, BINARY, AXIOM = ADOPTED.rule_set.rules
MATCHING: Final = ADOPTED.rule_set.matching
NORMAL_FORM: Final = NoNormalForm[Term]()
TARGETS: Final = TargetSet((B,), MATCHING)
CLOSURE: Final = ClosureReport(
    (A, B),
    saturated=True,
    steps=1,
    budget=8,
)
THEORY: Final = TheoryArtifact.from_specification(
    "panproto-0.74",
    b'{"families":["Category"],"order":2}',
)


def _budget(
    *,
    depth: int = 5,
    mass: float = 1.0,
    beam: int = 8,
    unary_rounds: int = 1,
    threads: int = 1,
    beta_fuel: int = 1_000,
    max_category_arity: int | None = None,
    max_category_size: int | None = None,
) -> Budget:
    return Budget(
        depth=depth,
        mass=mass,
        beam=beam,
        unary_rounds=unary_rounds,
        threads=threads,
        beta_fuel=beta_fuel,
        max_category_arity=max_category_arity,
        max_category_size=max_category_size,
    )


BUDGET: Final = _budget()
PIN: Final = Clamp((0, 1), category=(A,))


def _rules(
    inventory: tuple[Rule[Term], ...] = (UNARY, BINARY),
) -> RuleSet[Term]:
    return RuleSet("test-rules", inventory, MATCHING, NORMAL_FORM)


def _declaration(
    *,
    theory: TheoryArtifact = THEORY,
    fanout: int = 1,
    lexicalized: bool = True,
    budget: Budget = BUDGET,
    clamps: tuple[Clamp[Term], ...] = (PIN,),
) -> FormalismDeclaration:
    return FormalismDeclaration(
        CLOSURE,
        theory,
        fanout,
        budget,
        capacity_report(_rules()),
        lexicalized=lexicalized,
        clamps=clamps,
    )


def _formalism(
    *,
    rules: RuleSet[Term] | None = None,
    targets: TargetSet[Term] = TARGETS,
    declaration: FormalismDeclaration | None = None,
) -> Formalism:
    return Formalism(
        "test",
        STORE,
        _rules() if rules is None else rules,
        targets,
        RootGoal(targets),
        _declaration() if declaration is None else declaration,
    )


def test_theory_artifact_checks_and_exposes_canonical_bytes() -> None:
    specification = b'{"index":"Cat","order":2}'
    artifact = TheoryArtifact.from_specification("panproto-0.74", specification)

    assert artifact.version == "panproto-0.74"
    assert artifact.specification is specification
    assert artifact.digest == sha256(specification).hexdigest()
    assert artifact == TheoryArtifact(
        "panproto-0.74",
        specification,
        sha256(specification).hexdigest(),
    )
    assert hash(artifact) == hash(
        TheoryArtifact(
            "panproto-0.74",
            specification,
            sha256(specification).hexdigest(),
        )
    )
    assert repr(artifact).startswith("TheoryArtifact('panproto-0.74'")


def test_theory_artifact_refuses_a_wrong_digest() -> None:
    with pytest.raises(ValueError, match="does not match"):
        TheoryArtifact("panproto-0.74", b"{}", "0" * 64)


@pytest.mark.parametrize(
    ("version", "specification", "error"),
    [
        ("", b"{}", ValueError),
        ("panproto-0.74", b"", ValueError),
        ("panproto-0.74", cast("bytes", "{}"), TypeError),
    ],
)
def test_theory_artifact_refuses_invalid_fields(
    version: str,
    specification: bytes,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        TheoryArtifact.from_specification(version, specification)


def test_budget_exposes_one_approximation_policy() -> None:
    budget = Budget(
        depth=7,
        mass=0.95,
        beam=12,
        unary_rounds=3,
        threads=8,
        beta_fuel=400,
        max_category_arity=3,
        max_category_size=11,
    )

    assert budget.depth == 7
    assert budget.mass == 0.95
    assert budget.beam == 12
    assert budget.unary_rounds == 3
    assert budget.threads == 8
    assert budget.beta_fuel == 400
    assert budget.max_category_arity == 3
    assert budget.max_category_size == 11
    rebuilt = Budget(
        depth=7,
        mass=0.95,
        beam=12,
        unary_rounds=3,
        threads=8,
        beta_fuel=400,
        max_category_arity=3,
        max_category_size=11,
    )
    assert budget == rebuilt
    assert hash(budget) == hash(rebuilt)
    assert repr(budget).startswith("Budget(depth=7")


def test_compile_derives_the_profile_from_rules_and_closure() -> None:
    formalism = _formalism(declaration=_declaration(fanout=2, lexicalized=False))

    assert formalism.requirements.ranks == frozenset({1, 2})
    assert formalism.requirements.unary_closure
    assert formalism.requirements.category_set == "finite"
    assert formalism.requirements.restriction_mode == "unrestricted"
    assert formalism.profile.fanout == 2
    assert formalism.profile.ranks == formalism.requirements.ranks
    assert formalism.profile.unary_closure == formalism.requirements.unary_closure
    assert formalism.profile.category_set == formalism.requirements.category_set
    assert not formalism.profile.lexicalized


def test_compiled_grammar_retains_every_native_runtime_value() -> None:
    rules = _rules()
    formalism = _formalism(rules=rules)
    grammar = formalism.compile()

    assert grammar.name == "test"
    assert grammar.signature is SIGNATURE
    assert grammar.store is STORE
    assert grammar.rules is rules
    assert grammar.matching is MATCHING
    assert grammar.normal_form is NORMAL_FORM
    assert grammar.closure is formalism.closure
    assert formalism.targets is TARGETS
    assert grammar.targets is TARGETS
    formalism_goal = cast("RootGoal[Term]", formalism.goal)
    grammar_goal = cast("RootGoal[Term]", grammar.goal)

    assert formalism_goal.targets is TARGETS
    assert grammar.goal is formalism.goal
    assert grammar_goal.targets is TARGETS
    assert grammar.profile is formalism.profile
    assert grammar.budget is BUDGET
    assert grammar.capacity is formalism.capacity
    assert grammar.capacity.result == "not_characterized"
    assert grammar.clamps == (PIN,)
    assert grammar.clamps[0] is PIN
    assert grammar.theory_version == THEORY.version
    assert grammar.theory_specification is THEORY.specification
    assert grammar.theory_digest == THEORY.digest
    assert grammar.fingerprint == formalism.fingerprint
    assert len(grammar.fingerprint) == 64


def test_repeated_compile_returns_one_shared_grammar() -> None:
    formalism = _formalism()
    grammar = formalism.compile()

    assert formalism.compile() is grammar
    assert copy(grammar) is grammar
    assert deepcopy(grammar) is grammar


def test_direct_grammar_construction_is_refused() -> None:
    formalism = _formalism()

    with pytest.raises(TypeError, match=r"Formalism\.compile"):
        Grammar(formalism)

    grammar = formalism.compile()

    assert formalism.compile() is grammar


def test_compile_derives_rank_zero_and_a_bound_refuses_it_structurally() -> None:
    formalism = _formalism(rules=_rules((AXIOM,)))

    assert formalism.requirements.ranks == frozenset({0})
    assert formalism.profile.ranks == frozenset({0})
    grammar = formalism.compile((ProfileBound("axiom chart", 1, frozenset({0})),))

    with pytest.raises(ProfileUnsupportedError) as refusal:
        formalism.compile((ProfileBound("lofi.deduce.parse", 1, frozenset({1, 2, 3})),))

    assert "lofi.deduce.parse" in str(refusal.value)
    assert "[0]" in str(refusal.value)
    assert formalism.compile() is grammar


def test_every_bound_is_checked_even_after_compilation_is_cached() -> None:
    formalism = _formalism()
    accepted = ProfileBound("general", 1, frozenset({1, 2}))
    rank_bound = ProfileBound("binary-only", 1, frozenset({2}))
    grammar = formalism.compile((accepted,))

    with pytest.raises(ProfileUnsupportedError, match="binary-only"):
        formalism.compile((accepted, rank_bound))

    assert formalism.compile((accepted,)) is grammar


def test_fanout_refusal_names_the_consumer_and_extension_point() -> None:
    formalism = _formalism(declaration=_declaration(fanout=2))
    bound = ProfileBound("span chart", 1, frozenset({1, 2}), "a region index")

    with pytest.raises(ProfileUnsupportedError) as refusal:
        formalism.compile((bound,))

    assert "span chart" in str(refusal.value)
    assert "fan-out 2" in str(refusal.value)
    assert "a region index" in str(refusal.value)


def test_rule_inventory_changes_the_profile_and_fingerprint() -> None:
    unary = _formalism(rules=_rules((UNARY,)))
    mixed = _formalism()

    assert unary.profile.ranks == frozenset({1})
    assert mixed.profile.ranks == frozenset({1, 2})
    assert unary.fingerprint != mixed.fingerprint


def test_rule_declaration_order_changes_the_fingerprint() -> None:
    forward = _formalism(rules=_rules((UNARY, BINARY)))
    backward = _formalism(rules=_rules((BINARY, UNARY)))

    assert forward.profile == backward.profile
    assert forward.fingerprint != backward.fingerprint


def test_target_categories_change_the_goal_and_fingerprint() -> None:
    a_targets = TargetSet((A,), MATCHING)
    b_targets = TargetSet((B,), MATCHING)
    a_formalism = _formalism(targets=a_targets)
    b_formalism = _formalism(targets=b_targets)

    a_goal = cast("RootGoal[Term]", a_formalism.goal)
    b_goal = cast("RootGoal[Term]", b_formalism.goal)

    assert a_goal.targets is a_targets
    assert b_goal.targets is b_targets
    assert a_goal.categories == (A,)
    assert b_goal.categories == (B,)
    assert a_formalism.fingerprint != b_formalism.fingerprint


def test_the_declared_goal_kind_and_settings_enter_the_compiled_grammar() -> None:
    cover = CoverGoal(3, penalty=0.5)
    formalism = Formalism(
        "cover",
        STORE,
        _rules(),
        TARGETS,
        cover,
        _declaration(),
    )
    grammar = formalism.compile()

    assert grammar.goal is cover
    assert formalism.fingerprint != _formalism().fingerprint


def test_root_category_weights_change_the_grammar_fingerprint() -> None:
    neutral = _formalism()
    weighted = Formalism(
        "test",
        STORE,
        _rules(),
        TARGETS,
        RootGoal(TARGETS, (-0.5,)),
        _declaration(),
    )

    assert neutral.fingerprint != weighted.fingerprint


def test_indexed_declaration_and_order_bound_change_theory_and_grammar_digests() -> (
    None
):
    order_two = TheoryArtifact.from_specification(
        "panproto-0.74",
        b'{"families":["Category"],"order":2}',
    )
    order_three = TheoryArtifact.from_specification(
        "panproto-0.74",
        b'{"families":["Category","Mode"],"order":3}',
    )
    first = _formalism(declaration=_declaration(theory=order_two))
    second = _formalism(declaration=_declaration(theory=order_three))

    assert order_two.digest != order_three.digest
    assert first.fingerprint != second.fingerprint


def test_identical_native_declarations_and_theory_bytes_fingerprint_equally() -> None:
    first_declaration = _declaration(
        theory=TheoryArtifact.from_specification("v1", b'{"x":1}')
    )
    second_declaration = _declaration(
        theory=TheoryArtifact.from_specification("v1", b'{"x":1}')
    )
    first = _formalism(declaration=first_declaration)
    second = _formalism(declaration=second_declaration)

    assert first_declaration is not second_declaration
    assert first_declaration != second_declaration
    assert first is not second
    assert first != second
    assert first.fingerprint == second.fingerprint
    assert first.compile() is not second.compile()
    assert first.compile() != second.compile()
    assert first.compile().fingerprint == second.compile().fingerprint


def test_clamp_declaration_changes_the_fingerprint() -> None:
    unclamped = _formalism(declaration=_declaration(clamps=()))
    clamped = _formalism(declaration=_declaration(clamps=(PIN,)))

    assert unclamped.fingerprint != clamped.fingerprint


def test_a_unary_inventory_requires_unary_closure_fuel() -> None:
    with pytest.raises(ValueError, match="unary-closure"):
        _formalism(declaration=_declaration(budget=_budget(unary_rounds=0)))


def test_an_empty_target_set_is_refused() -> None:
    with pytest.raises(ValueError, match=r"formalism 'test'.*empty target set"):
        _formalism(targets=TargetSet((), MATCHING))


def test_target_matching_must_be_the_rule_matching_object() -> None:
    other_matching = Identity[Term]()

    with pytest.raises(
        ValueError,
        match=(
            r"formalism 'test' target matching 'identity'.*"
            r"rule matching 'identity' by identity"
        ),
    ):
        _formalism(targets=TargetSet((B,), other_matching))


def test_a_foreign_target_is_refused_by_name() -> None:
    foreign_store = TermStore(SIGNATURE)
    foreign = SYNTAX.parse(foreign_store, "B")

    with pytest.raises(
        ValueError,
        match=r"outside formalism 'test'.*store lineage",
    ) as refusal:
        _formalism(targets=TargetSet((foreign,), MATCHING))

    assert repr(foreign) in str(refusal.value)
    assert "formalism 'test'" in str(refusal.value)
    assert "store lineage" in str(refusal.value)


def test_an_unreachable_target_is_refused_with_the_closure() -> None:
    pair = STORE.op(SIGNATURE.by_name("pair"), A, A)

    with pytest.raises(
        ValueError,
        match=r"unreachable in closure ClosureReport",
    ) as refusal:
        _formalism(targets=TargetSet((pair,), MATCHING))

    assert repr(pair) in str(refusal.value)
    assert repr(CLOSURE) in str(refusal.value)
    assert "unreachable" in str(refusal.value)


class _SortMatching:
    """Match fixture terms by result sort."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the fixture matching name."""
        return "sort"

    @property
    def has_key(self) -> bool:
        """Return whether the matching supports representative keys."""
        return False

    def key(self, term: Term, /) -> Term:
        """Return the input term because keys are unavailable.

        Parameters
        ----------
        term
            Fixture term to return unchanged.
        """
        return term

    def equivalent(
        self,
        left: Term,
        right: Term,
        /,
    ) -> bool:
        """Return whether the terms have the same result sort.

        Parameters
        ----------
        left
            First fixture term.
        right
            Second fixture term.
        """
        return left.op.sort == right.op.sort


def test_a_matching_equivalent_target_is_reachable() -> None:
    matching = _SortMatching()
    rules = RuleSet("sort-matching", (), matching, NORMAL_FORM)
    closure = ClosureReport((A,), saturated=True, steps=0, budget=1)
    targets = TargetSet((B,), matching)
    declaration = FormalismDeclaration(
        closure,
        THEORY,
        1,
        BUDGET,
        capacity_report(rules),
        lexicalized=True,
        clamps=(PIN,),
    )
    formalism = Formalism(
        "matching-equivalent",
        STORE,
        rules,
        targets,
        RootGoal(targets),
        declaration,
    )

    assert formalism.targets is targets
    assert cast("RootGoal[Term]", formalism.goal).categories == (B,)


def test_foreign_closure_and_clamp_terms_are_refused() -> None:
    foreign_store = TermStore(SIGNATURE)
    foreign = SYNTAX.parse(foreign_store, "A")
    foreign_closure = ClosureReport(
        (foreign,),
        saturated=True,
        steps=0,
        budget=1,
    )

    with pytest.raises(ValueError, match="closure term"):
        Formalism(
            "foreign-closure",
            STORE,
            _rules(),
            TARGETS,
            RootGoal(TARGETS),
            FormalismDeclaration(
                foreign_closure,
                THEORY,
                1,
                BUDGET,
                capacity_report(_rules()),
                lexicalized=True,
            ),
        )

    with pytest.raises(ValueError, match="clamp category"):
        _formalism(
            declaration=_declaration(clamps=(Clamp((0, 1), category=(foreign,)),))
        )


@pytest.mark.parametrize(
    "value",
    [BUDGET, THEORY, _declaration(), _formalism(), _formalism().compile()],
)
def test_formalism_values_are_shared_and_frozen(
    value: (Budget | TheoryArtifact | FormalismDeclaration | Formalism | Grammar),
) -> None:
    assert copy(value) is value
    assert deepcopy(value) is value

    with pytest.raises(AttributeError, match="immutable"):
        value.__setattr__("_name", "changed")
