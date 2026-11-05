from itertools import product
from typing import TYPE_CHECKING, Final, Literal

import pytest
from hypothesis import given, seed, settings
from hypothesis import strategies as st
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import (
    FORWARD_ENUMERATION,
    HOLE,
    NormalForm,
    Rule,
    RuleSet,
    TypeRaising,
)
from lofi.spec import RuleSetSpec, TypeRaisingSpec
from lofi.spec._rule_laws import LawGrammar, LawView
from lofi.spec.grammar import OperatorSpec
from lofi.terms import (
    Identity,
    MVar,
    Node,
    Spine,
    Term,
    at_path,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.judge import NormalFormDecisionTable
    from lofi.terms import Pattern

type SlashRole = Literal["forward", "backward"]
"""Distinguished slash role."""

GRAMMARS: Final = {
    view: LawGrammar(view) for view in ("direct", "swapped", "undirected")
}
VIEWS: Final[tuple[LawView, ...]] = ("direct", "swapped", "undirected")
DIRECT_GRAMMAR: Final = GRAMMARS["direct"]
DIRECT: Final = DIRECT_GRAMMAR.functor("forward")[0]
X: Final = DIRECT_GRAMMAR.atom("x")
T: Final = DIRECT_GRAMMAR.atom("t")
U: Final = DIRECT_GRAMMAR.atom("u")
OTHER: Final = DIRECT_GRAMMAR.atom("other")


def _raising(
    grammar: LawGrammar,
    role: SlashRole,
    targets: tuple[str, ...] | None,
    arguments: tuple[str, ...] | None = None,
) -> TypeRaising:
    """Compile one type-raising rule over a law grammar."""
    rule = grammar.rule(
        TypeRaisingSpec(
            view=grammar.view_name(role), targets=targets, arguments=arguments
        )
    )

    assert type(rule) is TypeRaising

    return rule


def _raised(
    grammar: LawGrammar, role: SlashRole, input_term: Term, target: Term
) -> Term:
    """Construct one type-raised term."""
    functors, slash = grammar.functor(role)
    inner = functors.make(functors.opposite(slash), target, input_term)

    return functors.make(slash, target, inner)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("targeted", [True, False], ids=("targeted", "untargeted"))
def test_the_equation_covers_all_twelve_declared_variants(
    view: LawView,
    role: SlashRole,
    *,
    targeted: bool,
) -> None:
    grammar = GRAMMARS[view]
    functors, _ = grammar.functor(role)
    x, t, u = (grammar.atom(name) for name in ("x", "t", "u"))
    rule = _raising(grammar, role, ("t", "u") if targeted else None)
    expected = tuple(_raised(grammar, role, x, target) for target in (t, u))
    outputs = tuple(rule.apply((x,)))

    assert isinstance(rule, Rule)
    assert rule.probe is None
    assert rule.arity == 1
    assert rule.residual_slots == frozenset({0})
    assert rule.directions == (
        frozenset({"forward", "check", "residual"})
        if targeted
        else frozenset({"check", "residual"})
    )
    assert rule.forward_is_complete is targeted
    assert outputs == (expected if targeted else ())

    for target, output in zip((t, u), expected, strict=True):
        inner = at_path(output, functors.argument_path)

        assert at_path(output, functors.result_path) == target
        assert at_path(inner, functors.result_path) == target
        assert at_path(inner, functors.argument_path) == x
        assert rule.admits((x,), output)
        assert tuple(rule.residuate(output, (HOLE,))) == (x,)


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("role", ["forward", "backward"])
def test_targeted_admission_filters_outer_targets(
    view: LawView,
    role: SlashRole,
) -> None:
    grammar = GRAMMARS[view]
    x = grammar.atom("x")
    rule = _raising(grammar, role, ("t",))
    declared = _raised(grammar, role, x, grammar.atom("t"))
    undeclared = _raised(grammar, role, x, grammar.atom("u"))

    assert rule.admits((x,), declared)
    assert not rule.admits((x,), undeclared)
    assert tuple(rule.residuate(declared, (HOLE,))) == (x,)
    assert tuple(rule.residuate(undeclared, (HOLE,))) == ()


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("role", ["forward", "backward"])
def test_untargeted_admission_accepts_every_well_shaped_target(
    view: LawView,
    role: SlashRole,
) -> None:
    grammar = GRAMMARS[view]
    x = grammar.atom("x")
    rule = _raising(grammar, role, None)

    for target in ("t", "u", "other"):
        output = _raised(grammar, role, x, grammar.atom(target))

        assert rule.admits((x,), output)
        assert tuple(rule.residuate(output, (HOLE,))) == (x,)

    assert tuple(rule.apply((x,))) == ()


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("role", ["forward", "backward"])
def test_structural_mismatches_are_rejected(
    view: LawView,
    role: SlashRole,
) -> None:
    grammar = GRAMMARS[view]
    functors, slash = grammar.functor(role)
    opposite = functors.opposite(slash)
    x, t, u, other = (grammar.atom(name) for name in ("x", "t", "u", "other"))
    rule = _raising(grammar, role, None)
    wrong_input = _raised(grammar, role, other, t)
    mismatches = [functors.make(slash, t, functors.make(opposite, u, x)), wrong_input]

    if view != "undirected":
        mismatches.extend(
            (
                functors.make(opposite, t, functors.make(opposite, t, x)),
                functors.make(slash, t, functors.make(slash, t, x)),
            )
        )

    for output in mismatches:
        assert not rule.admits((x,), output)
        assert tuple(rule.residuate(output, (HOLE,))) == (
            (other,) if output == wrong_input else ()
        )


@pytest.mark.parametrize("view", VIEWS)
@pytest.mark.parametrize("role", ["forward", "backward"])
@pytest.mark.parametrize("targeted", [True, False], ids=("targeted", "untargeted"))
def test_only_the_declared_arguments_raise(
    view: LawView,
    role: SlashRole,
    *,
    targeted: bool,
) -> None:
    grammar = GRAMMARS[view]
    functors, slash = grammar.functor(role)
    x, t, other = (grammar.atom(name) for name in ("x", "t", "other"))
    composite = functors.make(slash, x, other)
    rule = _raising(grammar, role, ("t",) if targeted else None, ("x",))

    for undeclared in (other, composite):
        output = _raised(grammar, role, undeclared, t)

        assert tuple(rule.apply((undeclared,))) == ()
        assert not rule.admits((undeclared,), output)
        assert tuple(rule.residuate(output, (HOLE,))) == ()

    declared = _raised(grammar, role, x, t)

    assert tuple(rule.apply((x,))) == ((declared,) if targeted else ())
    assert rule.admits((x,), declared)
    assert tuple(rule.residuate(declared, (HOLE,))) == (x,)
    assert rule.to_spec()["arguments"] == [_term_array(x)]
    assert len(rule.proof_schemata) == 1
    assert not any(isinstance(premise, MVar) for (premise,), _ in rule.proof_schemata)


def test_raising_arguments_default_to_the_argument_atoms() -> None:
    assert TypeRaisingSpec(view="/").arguments == ("N", "S")


def test_malformed_queries_return_no_values() -> None:
    output = _raised(DIRECT_GRAMMAR, "forward", X, T)
    rule = _raising(DIRECT_GRAMMAR, "forward", ("t",))

    assert tuple(rule.apply(())) == ()
    assert tuple(rule.apply((X, T))) == ()
    assert not rule.admits((), output)
    assert not rule.admits((X, T), output)
    assert not rule.admits((X,), OTHER)
    assert tuple(rule.residuate(output, ())) == ()
    assert tuple(rule.residuate(output, (X,))) == ()
    assert tuple(rule.residuate(output, (HOLE, HOLE))) == ()


def _term_array(term: Term) -> JsonValue:
    """Encode a term as its JSON array declaration."""
    return [
        term.op.name,
        term.payload,
        [_term_array(argument) for argument in term.args],
    ]


LEAVES: Final = (X, T, U, OTHER)
TERM_STRATEGY: Final = st.recursive(
    st.sampled_from(LEAVES),
    lambda children: st.tuples(
        st.sampled_from(sorted(DIRECT.slashes)), children, children
    ).map(lambda parts: DIRECT.make(*parts)),
    max_leaves=20,
)


@pytest.mark.parametrize("targeted", [True, False], ids=("targeted", "untargeted"))
@seed(20260912)
@settings(max_examples=1_000, deadline=None, database=None)
@given(input_term=TERM_STRATEGY, output=TERM_STRATEGY)
def test_admission_is_total_on_a_thousand_generated_well_sorted_calls(
    input_term: Term,
    output: Term,
    *,
    targeted: bool,
) -> None:
    rule = _raising(DIRECT_GRAMMAR, "forward", ("t", "u") if targeted else None)

    assert isinstance(rule.admits((input_term,), output), bool)
    assert not rule.admits((), output)
    assert not rule.admits((input_term, input_term), output)


class AllowAll:
    """Permissive derivation-step filter."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the filter name."""
        return "allow-all"

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the single tag."""
        return frozenset({"item"})

    def lexical_tag(self) -> str:
        """Return the lexical tag."""
        return "item"

    def decisions(self, rules: Sequence[Rule[Term]], /) -> NormalFormDecisionTable:
        """Defer every finite tag state to exact step admission."""
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "premises")
            for rule in rules
        )

    def tag(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        /,
    ) -> str:
        """Return the derived tag."""
        del rule, children

        return "item"

    def admits(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Accept a derivation step."""
        del rule, children, inputs, output

        return True


def test_rule_set_supporting_reads_the_declared_directions() -> None:
    targeted = _raising(DIRECT_GRAMMAR, "forward", ("t",))
    untargeted = _raising(DIRECT_GRAMMAR, "forward", None)
    normal_form: NormalForm[Term] = AllowAll()
    rules = RuleSet("raising", (targeted, untargeted), Identity(), normal_form)

    assert rules.supporting("forward") == (targeted,)
    assert rules.supporting("check") == (targeted, untargeted)
    assert rules.supporting("residual") == (targeted, untargeted)

    findings = rules.validate(FORWARD_ENUMERATION, strict=False)

    assert len(findings) == 1
    assert findings[0].code == "forward_enumeration_unavailable"
    assert findings[0].rules == (untargeted.name,)


def test_the_json_declaration_preserves_recursive_targets_and_none() -> None:
    compound = DIRECT.make(DIRECT.forward_slash, T, U)
    targeted = _raising(DIRECT_GRAMMAR, "forward", ("t/u",))
    untargeted = _raising(DIRECT_GRAMMAR, "forward", None)

    assert targeted.targets == (compound,)
    assert targeted.to_spec() == {
        "rule": "type-raising",
        "matching": "identity",
        "slash": "fwd",
        "targets": [_term_array(compound)],
        "arguments": None,
        "restriction": None,
    }
    assert untargeted.to_spec()["targets"] is None


def test_the_rule_is_frozen_slotted_and_declarative() -> None:
    rule = _raising(DIRECT_GRAMMAR, "forward", ("t",))

    assert rule.functors.slashes == DIRECT.slashes
    assert rule.matching.name == "identity"
    assert rule.slash == DIRECT.forward_slash
    assert rule.targets == (T,)
    assert rule.name == "type-raising:fwd:targets=t"

    with pytest.raises(TypeError):
        vars(rule)

    with pytest.raises(AttributeError, match="immutable"):
        rule.__setattr__("_slash", DIRECT.backward_slash)


def test_the_rule_name_uses_declared_vocabulary_constant_names() -> None:
    adopted = declared_rules(
        ccg_signature(
            "S", operators=(OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),)
        ),
        RuleSetSpec(
            name="raising",
            rules=(
                TypeRaisingSpec(view="/", targets=("S[dcl]",), arguments=("S[dcl]",)),
            ),
        ),
    )
    (rule,) = adopted.rule_set
    target = adopted.context.syntax.parse(adopted.context.store, "S[dcl]")

    assert str(target) == "S[dcl]"
    assert repr(target).startswith("#const[Cat](")
    assert rule.name == "type-raising:/:targets=S[dcl]:arguments=S[dcl]"


def _metavariable_sorts(pattern: Pattern) -> frozenset[str]:
    """Return the sorts of every metavariable in one pattern."""
    match pattern:
        case MVar():
            return frozenset({pattern.sort})
        case Node():
            return frozenset().union(*map(_metavariable_sorts, pattern.args))
        case Spine():
            return _metavariable_sorts(pattern.core)


def test_schema_metavariables_carry_the_functor_category_sort() -> None:
    rule = _raising(DIRECT_GRAMMAR, "forward", None)
    sorts = {
        sort
        for premises, conclusion in rule.reachability_schemata
        for pattern in (*premises, conclusion)
        for sort in _metavariable_sorts(pattern)
    }

    assert sorts == {DIRECT.category_sort}
