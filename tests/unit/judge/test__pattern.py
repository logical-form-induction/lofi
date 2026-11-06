import json
from itertools import product
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.support.rules import declared_rules

from lofi.judge import HOLE, PatternRule, Rule, RuleSet, Slot
from lofi.spec import (
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PatternSpec,
    PrefixConstructorNotationSpec,
    RuleSetSpec,
    SpinePatternSpec,
    TermPatternSpec,
    VariableOperatorSpec,
)
from lofi.spec.grammar import (
    FeatureErasureMatchingSpec,
    IdentityMatchingSpec,
    OperatorSpec,
    ProductMatchingSpec,
    SignatureSpec,
)
from lofi.terms import MVar, Node, Spine, Term, build, match

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from lofi.judge import NormalFormDecisionTable
    from lofi.spec.grammar import MatchingSpec


def _constructor(name: str) -> ConstructorSpec:
    return ConstructorSpec(
        name=name,
        kind=("Cat", "Cat", "Cat"),
        notation=PrefixConstructorNotationSpec(spelling=name),
    )


SIGNATURE: Final = SignatureSpec(
    name="pattern-rules",
    primitives=("left", "right", "s"),
    constructors=tuple(_constructor(name) for name in ("pair", "fwd", "bwd")),
)
FEATURED: Final = SignatureSpec(
    name="featured",
    primitives=("s", "np"),
    operators=(
        OperatorSpec(name="s[dcl]", sort="Cat", erases_to="s"),
        OperatorSpec(name="s[ng]", sort="Cat", erases_to="s"),
    ),
    constructors=(_constructor("bwd"),),
)
X: Final = TermPatternSpec(name="X", sort="Cat")
Y: Final = TermPatternSpec(name="Y", sort="Cat")


def _atom(name: str) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


def _node(operator: str, *arguments: PatternSpec) -> NodePatternSpec:
    return NodePatternSpec(
        operator=FixedOperatorSpec(name=operator), arguments=arguments
    )


def _rule(
    declaration: PatternRuleSpec,
    *,
    signature: SignatureSpec = SIGNATURE,
    matching: MatchingSpec | None = None,
) -> PatternRule:
    """Compile one pattern rule through the native grammar compiler."""
    (rule,) = declared_rules(
        signature,
        RuleSetSpec(name=declaration.name, rules=(declaration,)),
        matching=matching,
    ).rule_set

    return cast("PatternRule", rule)


PAIRING_DECLARATION: Final = PatternRuleSpec(
    name="pair-right",
    premises=(X, _atom("right")),
    conclusion=_node("pair", X, _atom("right")),
)
PAIRING: Final = _rule(PAIRING_DECLARATION)
UNARY_DECLARATION: Final = PatternRuleSpec(
    name="atom-to-phrase", premises=(_atom("left"),), conclusion=_atom("right")
)
UNARY: Final = _rule(UNARY_DECLARATION)
_ANCHOR: Final = declared_rules(
    SIGNATURE, RuleSetSpec(name="anchor", rules=(UNARY_DECLARATION,))
).context
STORE: Final = _ANCHOR.store
SYNTAX: Final = _ANCHOR.syntax


def _term(text: str) -> Term:
    return SYNTAX.parse(STORE, text)


def _pattern(term: Term) -> Node:
    """Return the ground pattern of one term."""
    return Node(
        term.op.name, tuple(_pattern(child) for child in term.args), term.payload
    )


LEFT: Final = _term("left")
RIGHT: Final = _term("right")
S: Final = _term("s")
PAIR_LEFT: Final = _term("pair(left, right)")
PAIR_RIGHT: Final = _term("pair(right, right)")
VARIABLE: Final = MVar("X", "Cat")


class _KeepAll:
    """Permissive derivation-step filter."""

    @property
    def name(self) -> str:
        """Return the normal-form name."""
        return "keep-all"

    @property
    def alphabet(self) -> frozenset[str]:
        """Return the single item tag."""
        return frozenset({"item"})

    def lexical_tag(self) -> str:
        """Select the lexical item tag."""
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
        """Select the derived item tag."""
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
        """Accept every derivation step."""
        del rule, children, inputs, output

        return True


def test_a_unary_constant_rule_supports_every_exact_direction() -> None:
    assert UNARY.arity == 1
    assert UNARY.directions == frozenset({"forward", "check", "residual"})
    assert UNARY.residual_slots == frozenset({0})
    assert UNARY.forward_is_complete
    assert UNARY.probe is not None
    assert UNARY.probe.alternatives == ((frozenset({LEFT.op.name}),),)
    assert list(UNARY.apply((LEFT,))) == [RIGHT]
    assert UNARY.admits((LEFT,), RIGHT)
    assert not UNARY.admits((RIGHT,), RIGHT)
    assert list(UNARY.residuate(RIGHT, (HOLE,))) == [LEFT]


def test_a_rule_set_selects_the_unary_pattern_rule_by_arity() -> None:
    rules = RuleSet(
        "patterns",
        (PAIRING, UNARY),
        PAIRING.matching,
        _KeepAll(),
    )

    assert rules.of_arity(1) == (UNARY,)
    assert rules.of_arity(2) == (PAIRING,)


def test_unary_queries_are_total_on_malformed_shapes() -> None:
    assert list(UNARY.apply(())) == []
    assert list(UNARY.apply((LEFT, RIGHT))) == []
    assert not UNARY.admits((), RIGHT)
    assert not UNARY.admits((LEFT, RIGHT), RIGHT)
    assert list(UNARY.residuate(RIGHT, ())) == []
    assert list(UNARY.residuate(RIGHT, (LEFT,))) == []
    assert list(UNARY.residuate(RIGHT, (HOLE, HOLE))) == []


def test_the_unary_declaration_rebuilds_the_same_relation() -> None:
    spec = UNARY.to_spec()
    rebuilt = _rule(UNARY_DECLARATION)

    assert json.loads(json.dumps(spec)) == spec
    assert rebuilt.to_spec() == spec
    assert rebuilt.arity == UNARY.arity
    assert list(rebuilt.apply((LEFT,))) == [RIGHT]
    assert rebuilt.admits((LEFT,), RIGHT)
    assert list(rebuilt.residuate(RIGHT, (HOLE,))) == [LEFT]


def test_a_fully_bound_conclusion_supports_every_direction() -> None:
    assert isinstance(PAIRING, Rule)
    assert PAIRING.directions == frozenset({"forward", "check", "residual"})
    assert PAIRING.residual_slots == frozenset({0, 1})
    assert PAIRING.forward_is_complete
    assert PAIRING.probe is not None
    assert PAIRING.probe.alternatives == ((None, frozenset({RIGHT.op.name})),)
    assert list(PAIRING.apply((LEFT, RIGHT))) == [PAIR_LEFT]
    assert PAIRING.admits((LEFT, RIGHT), PAIR_LEFT)


SENTENCE_COERCION: Final = PatternRuleSpec(
    name="sentence-coercion",
    premises=(
        SpinePatternSpec(
            name="context", core=_atom("s"), operators=("fwd", "bwd"), max_order=3
        ),
    ),
    conclusion=NodePatternSpec(
        operator=VariableOperatorSpec(name="slash"), arguments=(X, X)
    ),
)


def test_unbound_conclusion_variables_make_a_rule_relational() -> None:
    rule = _rule(SENTENCE_COERCION)

    assert rule.directions == frozenset({"check"})
    assert rule.residual_slots == frozenset()
    assert not rule.forward_is_complete
    assert rule.probe is not None
    assert rule.probe.alternatives == ((frozenset({S.op.name, "fwd", "bwd"}),),)
    assert list(rule.apply((S,))) == []
    assert rule.admits((S,), _term("fwd(left, left)"))
    assert rule.admits((_term("bwd(s, right)"),), _term("bwd(right, right)"))
    assert not rule.admits((S,), _term("fwd(left, right)"))


def test_finite_targets_enumerate_an_unbound_conclusion_variable() -> None:
    rule = _rule(
        PatternRuleSpec(
            name="inject-left",
            premises=(TermPatternSpec(name="A", sort="Cat"),),
            conclusion=_node(
                "pair",
                TermPatternSpec(name="A", sort="Cat"),
                TermPatternSpec(name="B", sort="Cat"),
            ),
            targets=("left", "right"),
        )
    )
    left = _term("pair(s, left)")
    right = _term("pair(s, right)")

    assert rule.directions == frozenset({"forward", "check", "residual"})
    assert rule.forward_is_complete
    assert rule.targets == (LEFT, RIGHT)
    assert list(rule.apply((S,))) == [left, right]
    assert rule.admits((S,), left)
    assert rule.admits((S,), right)
    assert not rule.admits((S,), _term("pair(s, s)"))


def test_finite_targets_require_an_unbound_term_variable() -> None:
    with pytest.raises(ValueError, match="unbound conclusion term variable"):
        _rule(
            PatternRuleSpec(
                name="identity",
                premises=(TermPatternSpec(name="A", sort="Cat"),),
                conclusion=TermPatternSpec(name="A", sort="Cat"),
                targets=("left",),
            )
        )


def test_a_nonempty_spine_probe_uses_only_frame_heads() -> None:
    rule = _rule(
        PatternRuleSpec(
            name="nonempty-spine",
            premises=(
                SpinePatternSpec(
                    name="context",
                    core=_atom("s"),
                    operators=("fwd", "bwd"),
                    min_order=1,
                    max_order=3,
                ),
            ),
            conclusion=X,
        )
    )

    assert rule.probe is not None
    assert rule.probe.alternatives == ((frozenset({"fwd", "bwd"}),),)


@pytest.mark.parametrize(
    ("matching", "indexed"),
    [
        (
            ProductMatchingSpec(
                left=FeatureErasureMatchingSpec(), right=IdentityMatchingSpec()
            ),
            True,
        ),
        (FeatureErasureMatchingSpec(), False),
    ],
    ids=("declared-exact", "coarser"),
)
def test_a_head_probe_is_inferred_only_under_a_declared_exact_matching(
    matching: MatchingSpec, *, indexed: bool
) -> None:
    rule = _rule(
        PatternRuleSpec(
            name="headed",
            premises=(
                SpinePatternSpec(
                    name="context",
                    core=_atom("s"),
                    operators=("fwd", "bwd"),
                    min_order=1,
                    max_order=3,
                ),
            ),
            conclusion=X,
        ),
        matching=matching,
    )

    assert (rule.probe is not None) is indexed


def test_fully_unconstrained_premise_heads_decline_indexing() -> None:
    rule = _rule(
        PatternRuleSpec(
            name="wildcard",
            premises=(
                X,
                NodePatternSpec(operator=VariableOperatorSpec(name="op"), arguments=()),
            ),
            conclusion=X,
        )
    )

    assert rule.probe is None


def test_admits_returns_false_for_malformed_pairs() -> None:
    assert not PAIRING.admits((), PAIR_LEFT)
    assert not PAIRING.admits((LEFT,), PAIR_LEFT)
    assert not PAIRING.admits((LEFT, RIGHT, RIGHT), PAIR_LEFT)
    assert not PAIRING.admits((LEFT, LEFT), PAIR_LEFT)


def test_residuation_constructs_each_determined_premise() -> None:
    assert list(PAIRING.residuate(PAIR_LEFT, (HOLE, RIGHT))) == [LEFT]
    assert list(PAIRING.residuate(PAIR_LEFT, (LEFT, HOLE))) == [RIGHT]
    assert list(PAIRING.residuate(PAIR_RIGHT, (HOLE, RIGHT))) == [RIGHT]


def test_residuation_requires_exactly_one_hole() -> None:
    assert list(PAIRING.residuate(PAIR_LEFT, (LEFT, RIGHT))) == []
    assert list(PAIRING.residuate(PAIR_LEFT, (HOLE, HOLE))) == []
    assert list(PAIRING.residuate(PAIR_LEFT, (HOLE,))) == []


def test_underdetermined_inverse_construction_returns_empty() -> None:
    rule = _rule(SENTENCE_COERCION)

    assert list(rule.residuate(_term("fwd(left, left)"), (HOLE,))) == []


def test_residual_capability_is_derived_for_each_premise() -> None:
    rule = _rule(PatternRuleSpec(name="left-projection", premises=(X, Y), conclusion=X))

    assert rule.residual_slots == frozenset({0})
    assert rule.directions == frozenset({"forward", "check", "residual"})
    assert list(rule.residuate(LEFT, (HOLE, RIGHT))) == [LEFT]
    assert list(rule.residuate(LEFT, (LEFT, HOLE))) == []


def test_root_spines_enumerate_bindings_before_later_premises_match() -> None:
    twice = _term("fwd(fwd(left, right), right)")
    rule = _rule(
        PatternRuleSpec(
            name="shared-spine-core",
            premises=(
                SpinePatternSpec(
                    name="context", core=X, operators=("fwd",), max_order=2
                ),
                X,
            ),
            conclusion=X,
        )
    )

    assert list(rule.apply((twice, twice))) == [twice]


def test_the_declaration_contains_only_json_data() -> None:
    spec = PAIRING.to_spec()

    assert json.loads(json.dumps(spec)) == spec
    assert spec["rule"] == "pattern"
    assert spec["matching"] == "identity"


def test_pattern_rule_is_frozen_and_slotted() -> None:
    assert PAIRING.name == "pair-right"
    assert PAIRING.arity == 2
    assert PAIRING.premises == (VARIABLE, _pattern(RIGHT))
    assert PAIRING.conclusion == Node("pair", (VARIABLE, _pattern(RIGHT)))

    with pytest.raises(TypeError):
        vars(PAIRING)

    with pytest.raises(AttributeError, match="immutable"):
        PAIRING.__setattr__("_name", "changed")


class _MalformedRule:
    """Supplies residuation probes with the public slot type."""

    @staticmethod
    def run(
        rule: PatternRule,
        output: Term,
        known: tuple[Slot[Term], ...],
    ) -> Iterator[Term]:
        """Return the rule's residual iterator."""
        return rule.residuate(output, known)


def test_slot_annotations_accept_the_shared_hole_value() -> None:
    known = cast("tuple[Slot[Term], ...]", (HOLE, RIGHT))

    assert list(_MalformedRule.run(PAIRING, PAIR_LEFT, known)) == [LEFT]


LIFT_PAIR: Final = PatternRuleSpec(
    name="lift-pair",
    premises=(
        SpinePatternSpec(
            name="C", core=X, operators=("fwd", "bwd"), slot=0, max_order=2
        ),
        _atom("right"),
    ),
    conclusion=SpinePatternSpec(
        name="C",
        core=_node("pair", X, _atom("right")),
        operators=("fwd", "bwd"),
        max_order=2,
    ),
)


def test_a_bounded_spine_presents_one_deterministic_sequent_per_context() -> None:
    rule = _rule(LIFT_PAIR)
    schemata = rule.proof_schemata
    sibling = MVar("C#0.1", "Cat")
    paired = Node("pair", (VARIABLE, _pattern(RIGHT)))

    assert len(schemata) == 1 + 2 + 4
    assert schemata[0] == ((VARIABLE, _pattern(RIGHT)), paired)
    assert schemata[1] == (
        (Node("fwd", (VARIABLE, sibling)), _pattern(RIGHT)),
        Node("fwd", (paired, sibling)),
    )
    assert all(
        not isinstance(pattern, Spine)
        for premises, conclusion in schemata
        for pattern in (*premises, conclusion)
    )


def test_every_bounded_spine_sequent_agrees_with_forward_application() -> None:
    rule = _rule(LIFT_PAIR)
    deep = _term("bwd(fwd(left, s), right)")
    outputs = set(rule.apply((deep, RIGHT)))
    instances = {
        build(conclusion, STORE, bindings)
        for premises, conclusion in rule.proof_schemata
        if (bindings := match(premises[0], deep)) is not None
    }

    assert instances == outputs
    assert _term("bwd(fwd(pair(left, right), s), right)") in outputs


def test_an_unbounded_spine_presents_no_sequent() -> None:
    unbounded = _rule(
        PatternRuleSpec(
            name="unbounded",
            premises=(
                SpinePatternSpec(name="C", core=X, operators=("fwd",), max_order=None),
            ),
            conclusion=X,
        )
    )

    assert unbounded.proof_schemata == ()


def test_frames_are_drawn_only_from_declared_operators_with_the_slot() -> None:
    rule = _rule(
        PatternRuleSpec(
            name="frames",
            premises=(
                SpinePatternSpec(
                    name="C", core=X, operators=("fwd",), slot=1, max_order=1
                ),
            ),
            conclusion=X,
        )
    )

    assert rule.proof_schemata == (
        ((VARIABLE,), VARIABLE),
        ((Node("fwd", (MVar("C#0.0", "Cat"), VARIABLE)),), VARIABLE),
    )


def test_spine_occurrences_with_different_slots_share_only_the_empty_context() -> None:
    rule = _rule(
        PatternRuleSpec(
            name="slots",
            premises=(
                SpinePatternSpec(
                    name="C", core=X, operators=("fwd",), slot=0, max_order=2
                ),
            ),
            conclusion=SpinePatternSpec(
                name="C", core=X, operators=("fwd",), slot=1, max_order=2
            ),
        )
    )

    assert rule.proof_schemata == (((VARIABLE,), VARIABLE),)


def test_a_variable_free_premise_names_the_one_term_it_matches() -> None:
    assert UNARY.ground_premises == (LEFT,)
    assert PAIRING.ground_premises == (None, RIGHT)


_FEATURED_CONTEXT: Final = declared_rules(
    FEATURED,
    RuleSetSpec(
        name="featured-anchor",
        rules=(PatternRuleSpec(name="anchor", premises=(_atom("np"),), conclusion=X),),
    ),
).context


def _featured(text: str) -> Term:
    return _FEATURED_CONTEXT.syntax.parse(_FEATURED_CONTEXT.store, text)


def _verb(head: str) -> Term:
    return _featured(f"bwd({head}, np)")


def _featured_rule(declaration: PatternRuleSpec, *, erased: bool = True) -> PatternRule:
    return _rule(
        declaration,
        signature=FEATURED,
        matching=FeatureErasureMatchingSpec() if erased else None,
    )


def test_a_ground_premise_matches_its_whole_class_under_a_keyed_congruence() -> None:
    premise = _node("bwd", _atom("s[dcl]"), _atom("np"))
    erased = _featured_rule(
        PatternRuleSpec(name="erased", premises=(premise,), conclusion=_atom("np"))
    )
    exact = _featured_rule(
        PatternRuleSpec(name="exact", premises=(premise,), conclusion=_atom("np")),
        erased=False,
    )
    np = _featured("np")

    for head in ("s", "s[dcl]", "s[ng]"):
        assert tuple(erased.apply((_verb(head),))) == (np,)
        assert erased.admits((_verb(head),), np)

    assert tuple(exact.apply((_verb("s[dcl]"),))) == (np,)
    assert tuple(exact.apply((_verb("s[ng]"),))) == ()
    assert tuple(exact.apply((_verb("s"),))) == ()
    assert erased.premises == (_pattern(_verb("s[dcl]")),)
    assert erased.ground_premises == (_verb("s"),)
    assert exact.ground_premises == (_verb("s[dcl]"),)


def test_reachability_schemata_name_feature_erasure_classes() -> None:
    rule = _featured_rule(
        PatternRuleSpec(
            name="erased",
            premises=(_node("bwd", _atom("s[dcl]"), _atom("np")),),
            conclusion=_atom("s[ng]"),
        )
    )

    assert rule.proof_schemata == (
        ((_pattern(_verb("s[dcl]")),), _pattern(_featured("s[ng]"))),
    )
    assert rule.reachability_schemata == (
        ((_pattern(_verb("s")),), _pattern(_featured("s"))),
    )


def test_ground_subpatterns_beside_variables_match_their_class() -> None:
    rule = _featured_rule(
        PatternRuleSpec(
            name="lift",
            premises=(_node("bwd", _atom("s[dcl]"), X),),
            conclusion=_node("bwd", _atom("s[dcl]"), X),
        )
    )

    assert tuple(rule.apply((_verb("s[ng]"),))) == (_verb("s[dcl]"),)
    assert rule.admits((_verb("s"),), _verb("s[ng]"))
    assert tuple(rule.residuate(_verb("s[ng]"), (HOLE,))) == (_verb("s[dcl]"),)
