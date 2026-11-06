"""Rule declaration and open-plugin boundary tests."""

import tomllib
from collections import abc
from copy import copy, deepcopy
from functools import partial
from importlib.metadata import EntryPoint, EntryPoints
from pathlib import Path
from pickle import dumps, loads
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, get_args

import didactic.api as dx
import pytest
from tests.support.installed import install
from tests.support.term_reference import infix_syntax
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.grammars.ccg import CCGAssembly, FunctorView
from lofi.judge import (
    GeneralizedComposition,
    GeneralizedSubstitution,
    NoNormalForm,
    PatternRule,
    Rule,
    RuleRestriction,
    RuleSet,
    TypeRaising,
)
from lofi.registry import RegistryError, TagCollisionError, UnknownTagError
from lofi.registry import _registry as registry_module
from lofi.spec import (
    AbsorptionSpec,
    ApplicationSpec,
    BuiltRuleSet,
    ClausalAdjunctSpec,
    CoercionSetSpec,
    CoercionSpec,
    CombinatorSetSpec,
    CompositionSpec,
    ConstantPatternSpec,
    CoordinationSpec,
    FixedOperatorSpec,
    FixedPayloadSpec,
    HeadPolicy,
    HeadSelection,
    NodePatternSpec,
    OperatorPatternSpec,
    PatternRuleSpec,
    PatternSpec,
    PayloadPatternSpec,
    RulePlugin,
    RuleRestrictionSpec,
    RuleSetSpec,
    RuleSpec,
    SpinePatternSpec,
    SubstitutionSpec,
    TermPatternSpec,
    TypeRaisingSpec,
    VariableOperatorSpec,
    VariablePayloadSpec,
)
from lofi.spec import _rule as rule_module
from lofi.spec._matching import IdentityMatchingSpec
from lofi.spec.grammar import (
    GrammarDecl,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language
from lofi.terms import (
    FeatureErasure,
    MVar,
    Node,
    Operator,
    PVar,
    Signature,
    Spine,
    TermStore,
    UnknownConstantError,
    Vocabulary,
    _compile_grammar,
)

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import Protocol

    from lofi._types import JsonValue
    from lofi.terms import Term, _CompiledRule

    class _PatternDeclaration(Protocol):
        """A declaration that builds a term pattern."""

        def build(self, context: CCGAssembly, /) -> object:
            """Build the term pattern."""
            ...


SIGNATURE: Final = Signature(
    "rule-spec",
    ("Cat", "Meaning"),
    (
        Operator("meaning", "Meaning"),
        Operator("/", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("\\", "Cat", arg_sorts=("Cat", "Cat")),
    ),
    vocabularies=(
        Vocabulary("Cat", ("N", "NP", "S", "S[pss]", "S[adj]", "S[adj][conj]")),
    ),
    constant_cores={
        "Cat": {
            "S[pss]": "S",
            "S[adj]": "S",
            "S[adj][conj]": "S",
        }
    },
)
STORE: Final = TermStore(SIGNATURE)
SYNTAX: Final = infix_syntax(SIGNATURE, ("/", "/"), ("\\", "\\"))
FUNCTORS: Final = FunctorView(STORE, "/", "\\", result=0, argument=1)
ASSEMBLY: Final = CCGAssembly(
    STORE,
    SYNTAX,
    partial(SYNTAX.parse, STORE),
    {"/": (FUNCTORS, "/"), "\\": (FUNCTORS, "\\")},
    FeatureErasure(),
    None,
    NoNormalForm(),
    category_sort="Cat",
    categories=("S", "S\\NP", "NP", "S/N", "S/NP"),
)
X: Final = TermPatternSpec(name="X")
Y: Final = TermPatternSpec(name="Y", sort="Cat")
RULE_POINTS: Final = EntryPoints(
    EntryPoint(name=name, value=value, group="lofi.rule")
    for name, value in (
        ("absorption", "lofi.spec._ccg_rules:ABSORPTION_PLUGIN"),
        ("application", "lofi.spec._ccg_rules:APPLICATION_PLUGIN"),
        ("clausal-adjunct", "lofi.spec._ccg_rules:CLAUSAL_ADJUNCT_PLUGIN"),
        ("coercion", "lofi.spec._ccg_rules:COERCION_PLUGIN"),
        ("composition", "lofi.spec._ccg_rules:COMPOSITION_PLUGIN"),
        ("coordination", "lofi.spec._ccg_rules:COORDINATION_PLUGIN"),
        ("pattern", "lofi.spec._ccg_rules:PATTERN_PLUGIN"),
        ("substitution", "lofi.spec._ccg_rules:SUBSTITUTION_PLUGIN"),
        ("type-raising", "lofi.spec._ccg_rules:TYPE_RAISING_PLUGIN"),
    )
)


@pytest.fixture(autouse=True)
def _installed_rule_plugins(monkeypatch: pytest.MonkeyPatch) -> None:
    """Expose this checkout's declared rule plugins to the shared test runtime."""
    install(monkeypatch, "rule", RULE_POINTS)


def _build(declaration: RuleSetSpec, /) -> BuiltRuleSet[RuleSet[Term]]:
    """Build one rule set from the native plan used in production."""
    language = registered_language("propositional")
    grammar = GrammarDecl(
        name="rule-plan-fixture",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("N", "NP", "S"),
            operators=tuple(
                OperatorSpec(name=name, sort="Cat", erases_to="S")
                for name in ("S[pss]", "S[adj]", "S[adj][conj]")
            ),
        ),
        rules=declaration,
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="fixture", category="N"),)
        ),
    )
    plan = _compile_grammar(
        language.compile_envelope.canonical_declaration,
        grammar.compile_envelope.canonical_declaration,
    )
    context = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        {"/": (FUNCTORS, "/"), "\\": (FUNCTORS, "\\")},
        FeatureErasure(),
        None,
        NoNormalForm(),
        category_sort="Cat",
        categories=plan.runtime_categories,
        rule_matchings=grammar.prepared_plugins.matching_rows,
    )

    return grammar.prepared_plugins.rules.adopt(tuple(plan.rules), context)


def _build_one(declaration: RuleSpec, /) -> Rule[Term]:
    """Adopt one declaration from the authoritative native rule plan."""
    (rule,) = _build(RuleSetSpec(name="single-rule", rules=(declaration,))).rule_set
    return rule


def test_native_plan_materialization_scans_rule_plugins_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls = 0

    def counted() -> EntryPoints:
        nonlocal calls
        calls += 1

        return RULE_POINTS

    install(monkeypatch, "rule", ())
    current = registry_module.entry_points

    def listed() -> tuple[EntryPoint, ...]:
        return (*current(), *counted())

    monkeypatch.setattr(registry_module, "entry_points", listed)
    _build(
        RuleSetSpec(
            name="one-scan",
            rules=(
                PatternRuleSpec(name="identity", premises=(X,), conclusion=X),
                CombinatorSetSpec(application=True),
                CoercionSetSpec(label="lexical", entries=("N => NP",)),
            ),
        )
    )

    assert calls == 1


def test_pattern_values_build_against_the_published_signature_context() -> None:
    node = NodePatternSpec(
        operator=VariableOperatorSpec(name="F"),
        arguments=(X,),
        payload=VariablePayloadSpec(name="P"),
    )
    spine = SpinePatternSpec(
        name="C",
        core=ConstantPatternSpec(name="N"),
        operators=("/", "\\"),
        slot=1,
        min_order=1,
        max_order=None,
    )

    assert node.build(ASSEMBLY) == Node(
        PVar("F"), (MVar("X", "Cat"),), payload=PVar("P")
    )
    assert spine.build(ASSEMBLY) == Spine(
        "C",
        Node("#const[Cat]", payload=SIGNATURE.vocabulary("Cat").id("N")),
        ("/", "\\"),
        slot=1,
        min_order=1,
        max_order=None,
    )

    for declaration in (node, spine):
        restored = PatternSpec.model_validate_json(declaration.model_dump_json())
        assert restored == declaration
        assert cast("_PatternDeclaration", restored).build(
            ASSEMBLY
        ) == declaration.build(ASSEMBLY)


def test_a_constant_pattern_resolves_a_declared_category_atom() -> None:
    pattern = ConstantPatternSpec(name="NP")

    assert pattern.build(ASSEMBLY) == Node(
        "#const[Cat]", payload=SIGNATURE.vocabulary("Cat").id("NP")
    )
    assert PatternSpec.model_validate_json(pattern.model_dump_json()) == pattern

    with pytest.raises(UnknownConstantError, match="Undeclared"):
        ConstantPatternSpec(name="Undeclared").build(ASSEMBLY)


def test_a_term_variable_defaults_to_the_category_sort_and_checks_named_sorts() -> None:
    assert X.build(ASSEMBLY) == MVar("X", "Cat")
    assert TermPatternSpec(name="M", sort="Meaning").build(ASSEMBLY) == MVar(
        "M", "Meaning"
    )

    with pytest.raises(ValueError, match="Nonexistent"):
        TermPatternSpec(name="bad", sort="Nonexistent").build(ASSEMBLY)


def test_application_is_a_distinct_declaration_and_runtime_identity() -> None:
    forward = ApplicationSpec(view="/")
    backward = ApplicationSpec(view="\\")
    restored = RuleSpec.model_validate_json(forward.model_dump_json())
    built = _build(RuleSetSpec(name="application", rules=(forward, backward)))

    assert type(restored) is ApplicationSpec
    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "application:/",
        "application:\\",
    )
    assert dict(built.head_selections) == {
        "application:/": HeadSelection("functor", 0),
        "application:\\": HeadSelection("functor", 1),
    }


def test_rule_categories_lower_once_in_runtime_rule_order() -> None:
    """Resolve finite conclusion and consumption declarations as categories."""
    forward = ApplicationSpec(
        view="/",
        introduces=("S", "S\\NP"),
        consumes=("NP",),
    )
    backward = ApplicationSpec(view="\\")

    built = _build(RuleSetSpec(name="category-effects", rules=(forward, backward)))

    assert built.rule_set.introduces == (
        (ASSEMBLY.category("S"), ASSEMBLY.category("S\\NP")),
        None,
    )
    assert built.rule_set.consumes == ((ASSEMBLY.category("NP"),), None)


@pytest.mark.parametrize(
    ("field", "description"),
    [("introduces", "introduced"), ("consumes", "consumed")],
)
def test_a_finite_rule_category_declaration_is_nonempty(
    field: str, description: str
) -> None:
    """Distinguish a finite declaration from the unrestricted case."""
    with pytest.raises(dx.ValidationError, match=f"{description} categories"):
        ApplicationSpec.model_validate({"view": "/", field: []})


def test_a_pattern_rule_lowers_finite_unbound_conclusion_targets() -> None:
    declaration = PatternRuleSpec(
        name="inject-left",
        premises=(X,),
        conclusion=NodePatternSpec(
            operator=FixedOperatorSpec(name="/"),
            arguments=(X, Y),
        ),
        targets=("N", "NP"),
    )
    restored = RuleSpec.model_validate_json(declaration.model_dump_json())
    rule = _build_one(declaration)
    source = ASSEMBLY.category("S")

    assert restored == declaration
    assert isinstance(rule, PatternRule)
    assert list(rule.apply((source,))) == [
        ASSEMBLY.category("S/N"),
        ASSEMBLY.category("S/NP"),
    ]


def test_a_pattern_rule_refuses_an_empty_finite_target_inventory() -> None:
    with pytest.raises(dx.ValidationError, match="finite pattern-rule targets"):
        PatternRuleSpec(
            name="inject-left",
            premises=(X,),
            conclusion=NodePatternSpec(
                operator=FixedOperatorSpec(name="/"),
                arguments=(X, Y),
            ),
            targets=(),
        )


def test_composition_is_positive_order_and_names_its_direction_restriction() -> None:
    steedman = CompositionSpec(direction="steedman", order=2, view="/")
    kuhlmann = CompositionSpec(direction="kuhlmann", order=2, view="/")
    left = _build_one(steedman)
    right = _build_one(kuhlmann)

    assert isinstance(left, GeneralizedComposition)
    assert left.direction == "steedman"
    assert left.name == "composition:/:2:harmonic:steedman"
    assert right.name == "composition:/:2:harmonic:kuhlmann"
    assert left.name != right.name

    with pytest.raises(dx.ValidationError, match="positive"):
        CompositionSpec(direction="unrestricted", order=0, view="/")


def test_substitution_may_range_over_every_positive_finite_order() -> None:
    declaration = SubstitutionSpec(order=None, view="/")
    restored = RuleSpec.model_validate_json(declaration.model_dump_json())
    built = _build_one(restored)

    assert isinstance(built, GeneralizedSubstitution)
    assert built.order is None
    assert built.name == "substitution:/:any:harmonic"
    assert built.proof_schemata == ()

    with pytest.raises(dx.ValidationError, match="positive"):
        SubstitutionSpec(order=0, view="/")


def test_type_raising_names_its_finite_target_inventory() -> None:
    s = _build_one(TypeRaisingSpec(view="/", targets=("S",), arguments=None))
    np = _build_one(TypeRaisingSpec(view="/", targets=("NP",), arguments=None))

    assert s.name == "type-raising:/:targets=S"
    assert np.name == "type-raising:/:targets=NP"
    assert s.name != np.name


def test_every_shipped_declaration_builds_through_its_installed_plugin() -> None:
    declarations = (
        TypeRaisingSpec(view="/", targets=("S",), arguments=None),
        PatternRuleSpec(name="identity", premises=(X,), conclusion=X),
        AbsorptionSpec(
            name="punctuation",
            premises=(X, Y),
            conclusion=X,
            absorbing=("N",),
            side="right",
        ),
        ClausalAdjunctSpec(name="adjunct", source_view="\\", target_view="\\"),
    )
    rules = tuple(_build_one(declaration) for declaration in declarations)

    assert isinstance(rules[0], TypeRaising)
    assert isinstance(rules[1], PatternRule)
    assert tuple(rule.name for rule in rules) == (
        "type-raising:/:targets=S",
        "identity",
        "punctuation",
        "adjunct",
    )
    assert all(rule.matching.name == ASSEMBLY.matching.name for rule in rules)


def test_rule_set_derives_restriction_mode_and_every_head_selection() -> None:
    declared = RuleSetSpec(
        name="fixture",
        rules=(
            ApplicationSpec(
                view="/",
                restriction=RuleRestrictionSpec(primary_targets=("S",)),
            ),
            PatternRuleSpec(name="identity", premises=(X,), conclusion=X),
            AbsorptionSpec(
                name="absorb-left",
                premises=(X, Y),
                conclusion=Y,
                absorbing=("N",),
                side="left",
            ),
        ),
    )
    built = _build(declared)

    assert declared.restriction_mode == "grammar_specific"
    assert isinstance(built, BuiltRuleSet)
    assert built.rule_set.restriction_mode == "grammar_specific"
    assert dict(built.head_selections) == {
        "application:/:only(targets=S)": HeadSelection("functor", 0),
        "identity": HeadSelection("pass-through", 0),
        "absorb-left": HeadSelection("non-absorbing", 1),
    }
    assert type(built.head_policies) is MappingProxyType
    assert type(built.head_selections) is MappingProxyType


def test_one_rule_list_expands_set_entries_in_their_declared_positions() -> None:
    declared = RuleSetSpec(
        name="interleaved",
        rules=(
            CoercionSetSpec(label="lexical", entries=("N => NP",)),
            PatternRuleSpec(name="identity", premises=(X,), conclusion=X),
            CombinatorSetSpec(application=True),
        ),
    )
    built = _build(declared)

    assert declared.plugin_tags == ("coercion", "pattern", "application")
    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "lexical:0:N=>NP",
        "identity",
        "application:/",
        "application:\\",
    )


def test_a_declaration_publishes_its_member_tags_and_whether_it_only_applies() -> None:
    application = CombinatorSetSpec(application=True)
    raising = CombinatorSetSpec(
        application=True, type_raising=True, raising_arguments=None
    )
    coercions = CoercionSetSpec(label="lexical", entries=("N => NP",))

    assert application.member_tags() == ("application",)
    assert raising.member_tags() == ("application", "type-raising")
    assert coercions.member_tags() == ("coercion",)
    assert ApplicationSpec(view="/").member_tags() == ("application",)
    assert application.yields_only_application()
    assert ApplicationSpec(view="/").yields_only_application()
    assert not raising.yields_only_application()
    assert not coercions.yields_only_application()
    assert not CompositionSpec(
        direction="unrestricted", order=1, view="/"
    ).yields_only_application()


def test_a_combinator_set_hands_its_restriction_to_every_member() -> None:
    declared = RuleSetSpec(
        name="restricted",
        rules=(
            CombinatorSetSpec(
                application=True,
                restriction=RuleRestrictionSpec(primary_targets=("S",)),
            ),
        ),
    )
    built = _build(declared)

    assert declared.restriction_mode == "grammar_specific"
    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "application:/:only(targets=S)",
        "application:\\:only(targets=S)",
    )


@pytest.mark.parametrize("tag", ["application", "composition", "coercions"])
def test_a_rule_whose_head_is_fixed_by_its_kind_declares_no_head_policy(
    tag: str,
) -> None:
    fields: dict[str, dict[str, JsonValue]] = {
        "application": {"view": "/"},
        "composition": {"direction": "unrestricted", "order": 1, "view": "/"},
        "coercions": {"label": "lexical", "entries": ["N => NP"]},
    }

    with pytest.raises(dx.ValidationError):
        RuleSpec.model_validate({"rule": tag, "head_policy": "functor", **fields[tag]})


def test_a_combinator_set_expands_orders_over_the_published_functor_view() -> None:
    declared = RuleSetSpec(
        name="schema",
        rules=(
            CombinatorSetSpec(
                composition=(1, None),
                crossed_substitution=(2,),
                type_raising=True,
                raising_targets=("S",),
                raising_arguments=None,
            ),
        ),
    )
    built = _build(declared)

    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "application:/",
        "application:\\",
        "composition:/:1:harmonic:unrestricted",
        "composition:/:any:harmonic:unrestricted",
        "composition:\\:1:harmonic:unrestricted",
        "composition:\\:any:harmonic:unrestricted",
        "substitution:/:2:crossed",
        "substitution:\\:2:crossed",
        "type-raising:/:targets=S",
        "type-raising:\\:targets=S",
    )


def test_a_coercion_set_parses_ground_surface_categories_and_derives_names() -> None:
    declared = RuleSetSpec(
        name="coercions",
        rules=(
            CoercionSetSpec(
                label="lexical",
                entries=(
                    "N => NP",
                    "S[pss]/NP => S[adj][conj]\\NP",
                    "N => S",
                ),
            ),
        ),
    )
    built = _build(declared)

    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "lexical:0:N=>NP",
        "lexical:1:S[pss]/NP=>S[adj][conj]\\NP",
        "lexical:2:N=>S",
    )
    n = SYNTAX.parse(STORE, "N")
    np = SYNTAX.parse(STORE, "NP")
    assert tuple(built.rule_set.rules[0].apply((n,))) == (np,)
    assert tuple(built.rule_set.rules[2].apply((n,))) == (SYNTAX.parse(STORE, "S"),)


def test_a_malformed_surface_coercion_names_its_set_and_member() -> None:
    declared = CoercionSetSpec(label="lexical", entries=("N -> NP",))

    with pytest.raises(ValueError, match=r"coercion set\[0\] member\[0\]"):
        _build(RuleSetSpec(name="malformed", rules=(declared,)))


def test_a_large_coercion_set_keeps_repeated_sources_and_deep_categories() -> None:
    targets = ["S"]

    while len(targets) < 64:
        targets.append(f"({targets[-1]})/NP")

    entries = tuple(
        f"{source} => {target}" for target in targets for source in ("N", "NP", "S")
    )[:190]
    built = _build(
        RuleSetSpec(
            name="large coercions",
            rules=(CoercionSetSpec(label="lexical", entries=entries),),
        )
    )

    assert len(built.rule_set.rules) == 190
    assert len(built.head_selections) == 190
    assert built.rule_set.rules[3].name == "lexical:3:N=>(S)/NP"

    with pytest.raises(ValueError, match="'lexical:0:N=>S' and 'lexical:190:N=>S'"):
        _build(
            RuleSetSpec(
                name="repeated coercions",
                rules=(
                    CoercionSetSpec(label="lexical", entries=(*entries, entries[0])),
                ),
            )
        )


def test_restricted_and_unrestricted_rules_on_one_slash_keep_distinct_names() -> None:
    built = _build(
        RuleSetSpec(
            name="restricted twins",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(
                    view="/",
                    restriction=RuleRestrictionSpec(primary_targets=("S", "NP")),
                ),
                CompositionSpec(direction="unrestricted", order=1, view="/"),
                CompositionSpec(
                    direction="unrestricted",
                    order=1,
                    view="/",
                    restriction=RuleRestrictionSpec(secondary_patterns=("S/NP",)),
                ),
                SubstitutionSpec(order=1, view="/"),
                SubstitutionSpec(
                    order=1,
                    view="/",
                    restriction=RuleRestrictionSpec(secondary_patterns=("S/NP",)),
                ),
                TypeRaisingSpec(view="/", targets=("S",), arguments=None),
                TypeRaisingSpec(
                    view="/",
                    targets=("S",),
                    restriction=RuleRestrictionSpec(primary_targets=("NP",)),
                    arguments=None,
                ),
            ),
        )
    )

    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "application:/",
        "application:/:only(targets=NP,S)",
        "composition:/:1:harmonic:unrestricted",
        "composition:/:1:harmonic:unrestricted:only(patterns=S/NP)",
        "substitution:/:1:harmonic",
        "substitution:/:1:harmonic:only(patterns=S/NP)",
        "type-raising:/:targets=S",
        "type-raising:/:targets=S:only(targets=NP)",
    )


def test_duplicate_derived_rule_names_are_refused() -> None:
    declared = RuleSetSpec(
        name="duplicates",
        rules=(ApplicationSpec(view="\\"), ApplicationSpec(view="\\")),
    )

    with pytest.raises(ValueError, match="declares the same relation twice"):
        _build(declared)


def test_runtime_boundary_values_are_frozen_and_have_explicit_pickle_behavior() -> None:
    selection = HeadSelection("non-absorbing", 1)
    built = _build(RuleSetSpec(name="fixture", rules=(ApplicationSpec(view="/"),)))

    assert selection == HeadSelection("non-absorbing", 1)
    assert repr(selection) == "HeadSelection('non-absorbing', 1)"
    assert copy(selection) is selection
    assert deepcopy(selection) is selection
    assert loads(dumps(selection)) == selection
    assert copy(built) is built
    assert deepcopy(built) is built

    with pytest.raises(AttributeError, match="immutable"):
        selection.__setattr__("_slot", 0)

    with pytest.raises(TypeError, match="runtime rule identities"):
        dumps(built)


def test_pattern_values_cover_every_operator_payload_and_term_shape() -> None:
    fixed_operator = FixedOperatorSpec(name="meaning")
    variable_operator = VariableOperatorSpec(name="F")
    fixed_payload = FixedPayloadSpec(value=7)
    variable_payload = VariablePayloadSpec(name="P")
    nested = NodePatternSpec(
        operator=fixed_operator,
        arguments=(
            NodePatternSpec(
                operator=variable_operator,
                arguments=(X,),
                payload=variable_payload,
            ),
        ),
        payload=fixed_payload,
    )

    assert isinstance(fixed_operator, OperatorPatternSpec)
    assert isinstance(variable_operator, OperatorPatternSpec)
    assert isinstance(fixed_payload, PayloadPatternSpec)
    assert isinstance(variable_payload, PayloadPatternSpec)
    assert nested.build(ASSEMBLY) == Node(
        "meaning",
        (Node(PVar("F"), (MVar("X", "Cat"),), payload=PVar("P")),),
        payload=7,
    )

    restored = PatternSpec.model_validate_json(nested.model_dump_json())
    assert restored == nested
    assert cast("_PatternDeclaration", restored).build(ASSEMBLY) == nested.build(
        ASSEMBLY
    )


@pytest.mark.parametrize(
    "declaration",
    [
        ApplicationSpec(view="/"),
        CompositionSpec(direction="unrestricted", order=None, view="/"),
        SubstitutionSpec(order=None, view="\\"),
        TypeRaisingSpec(view="/", targets=("S",), arguments=None),
        ClausalAdjunctSpec(name="adjunct", source_view="\\", target_view="\\"),
        PatternRuleSpec(name="identity", premises=(X,), conclusion=X),
        AbsorptionSpec(
            name="absorb",
            premises=(X, Y),
            conclusion=X,
            absorbing=("N",),
            side="right",
        ),
        CoordinationSpec(
            name="coordinate",
            premises=(X, Y),
            conclusion=X,
            phase="mark",
            core_relation="same",
            head_policy="non-absorbing",
        ),
        CoercionSpec(label="lexical", member=0, source="N", target="NP"),
    ],
)
def test_every_shipped_declaration_round_trips_through_the_open_root(
    declaration: RuleSpec,
) -> None:
    restored = RuleSpec.model_validate_json(declaration.model_dump_json())

    assert type(restored) is type(declaration)
    assert restored == declaration


def test_rule_plugin_is_a_frozen_structural_construction_boundary() -> None:
    def adopt(context: int, _compiled: _CompiledRule) -> tuple[str, HeadSelection]:
        return "adopted" * context, HeadSelection("pass-through", context)

    plugin = RulePlugin(
        spec=str,
        adopt=adopt,
    )

    assert plugin.spec is str
    assert repr(plugin) == "RulePlugin(spec=str)"
    assert copy(plugin) is plugin
    assert deepcopy(plugin) is plugin

    with pytest.raises(AttributeError, match="immutable"):
        plugin.__setattr__("_spec", int)


class _RulePoint:
    """A controlled installed rule entry point."""

    __slots__ = ("_member", "dist", "group", "name", "value")

    def __init__(self, name: str, value: str, member: object) -> None:
        self.name = name
        self.value = value
        self.group = "lofi.rule"
        self.dist = None
        self._member = member

    def load(self) -> object:
        """Return the controlled entry-point member."""
        return self._member


def _application_plugin() -> RulePlugin[ApplicationSpec, object, object]:
    return RulePlugin(
        spec=ApplicationSpec,
        adopt=lambda context, _compiled: (
            context,
            HeadSelection("functor", None),
        ),
    )


def test_rule_plugin_resolution_selects_one_exact_tag_and_spec(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    plugin = _application_plugin()

    def selected() -> tuple[_RulePoint, ...]:
        return (_RulePoint("application", "fixture:APPLICATION", plugin),)

    install(monkeypatch, "rule", cast("tuple[EntryPoint, ...]", selected()))

    assert rule_module.rule_plugin(ApplicationSpec(view="/")) is plugin


@pytest.mark.parametrize(
    ("points", "error", "message"),
    [
        ((), UnknownTagError, "no rule is registered as 'application'"),
        (
            (
                _RulePoint("application", "a:PLUGIN", _application_plugin()),
                _RulePoint("application", "b:PLUGIN", _application_plugin()),
            ),
            TagCollisionError,
            "each declare 'application'",
        ),
        (
            (_RulePoint("application", "fixture:NOT_PLUGIN", object()),),
            RegistryError,
            "not a RulePlugin",
        ),
        (
            (
                _RulePoint(
                    "application",
                    "fixture:WRONG_PLUGIN",
                    RulePlugin(
                        spec=CompositionSpec,
                        adopt=lambda context, _compiled: (
                            context,
                            HeadSelection("functor", None),
                        ),
                    ),
                ),
            ),
            RegistryError,
            "builds CompositionSpec, not ApplicationSpec",
        ),
    ],
)
def test_rule_plugin_resolution_rejects_ambiguous_or_invalid_entries(
    monkeypatch: pytest.MonkeyPatch,
    points: tuple[_RulePoint, ...],
    error: type[Exception],
    message: str,
) -> None:
    def selected() -> tuple[_RulePoint, ...]:
        return points

    install(monkeypatch, "rule", cast("tuple[EntryPoint, ...]", selected()))

    with pytest.raises(error, match=message):
        rule_module.rule_plugin(ApplicationSpec(view="/"))


def test_project_metadata_names_every_shipped_rule_plugin_once() -> None:
    project = tomllib.loads(
        (Path(__file__).parents[3] / "pyproject.toml").read_text(encoding="utf-8")
    )
    declared = project["project"]["entry-points"]["lofi.rule"]

    assert set(declared.items()) == {
        ("absorption", "lofi.spec._ccg_rules:ABSORPTION_PLUGIN"),
        ("application", "lofi.spec._ccg_rules:APPLICATION_PLUGIN"),
        ("clausal-adjunct", "lofi.spec._ccg_rules:CLAUSAL_ADJUNCT_PLUGIN"),
        ("coercion", "lofi.spec._ccg_rules:COERCION_PLUGIN"),
        ("composition", "lofi.spec._ccg_rules:COMPOSITION_PLUGIN"),
        ("coordination", "lofi.spec._ccg_rules:COORDINATION_PLUGIN"),
        ("pattern", "lofi.spec._ccg_rules:PATTERN_PLUGIN"),
        ("substitution", "lofi.spec._ccg_rules:SUBSTITUTION_PLUGIN"),
        ("type-raising", "lofi.spec._ccg_rules:TYPE_RAISING_PLUGIN"),
    }
    assert len(declared) == 9


def test_native_plan_expands_every_combinator_family_in_declared_order() -> None:
    declaration = CombinatorSetSpec(
        application=False,
        composition=(1, None),
        crossed_composition=(2,),
        substitution=(3,),
        crossed_substitution=(4,),
        direction="kuhlmann",
        type_raising=True,
        raising_targets=("S", "NP"),
        raising_arguments=None,
    )

    built = _build(RuleSetSpec(name="families", rules=(declaration,)))

    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "composition:/:1:harmonic:kuhlmann",
        "composition:/:any:harmonic:kuhlmann",
        "composition:\\:1:harmonic:kuhlmann",
        "composition:\\:any:harmonic:kuhlmann",
        "composition:/:2:crossed:kuhlmann",
        "composition:\\:2:crossed:kuhlmann",
        "substitution:/:3:harmonic",
        "substitution:\\:3:harmonic",
        "substitution:/:4:crossed",
        "substitution:\\:4:crossed",
        "type-raising:/:targets=S,NP",
        "type-raising:\\:targets=S,NP",
    )


def test_native_plan_keeps_explicit_combinator_and_coercion_groups_ordered() -> None:
    explicit = PatternRuleSpec(name="identity", premises=(X,), conclusion=X)
    declaration = RuleSetSpec(
        name="ordered",
        rules=(
            explicit,
            CombinatorSetSpec(application=True),
            CoercionSetSpec(label="lexical", entries=("N => NP",)),
        ),
    )

    built = _build(declaration)

    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "identity",
        "application:/",
        "application:\\",
        "lexical:0:N=>NP",
    )


def test_native_plan_trims_coercions_and_preserves_member_positions() -> None:
    declaration = CoercionSetSpec(
        label="lexical",
        entries=(" N => NP ", "NP=>S"),
    )

    built = _build(RuleSetSpec(name="trimmed", rules=(declaration,)))

    assert tuple(rule.name for rule in built.rule_set.rules) == (
        "lexical:0:N=>NP",
        "lexical:1:NP=>S",
    )


def test_a_rule_declares_no_matching_unless_it_names_one() -> None:
    exact = IdentityMatchingSpec()

    assert ApplicationSpec(view="/").matching is None
    assert RuleSpec.model_validate_json(
        ApplicationSpec(view="/", matching=exact).model_dump_json()
    ) == ApplicationSpec(view="/", matching=exact)


def test_coercion_and_combinator_sets_hand_their_matching_to_every_member() -> None:
    exact = IdentityMatchingSpec()
    coercions = CoercionSetSpec(
        label="lexical", entries=("N => NP", "NP => S"), matching=exact
    )
    combinators = CombinatorSetSpec(
        composition=(1,),
        substitution=(1,),
        type_raising=True,
        matching=exact,
        raising_arguments=None,
    )

    built = _build(
        RuleSetSpec(
            name="matching",
            rules=(combinators, coercions),
        )
    )
    assert len(built.rule_set.rules) == 10
    assert {rule.matching.name for rule in built.rule_set.rules} == {"identity"}


@pytest.mark.parametrize("entry", ["N", "N =>", "=> NP", "N => NP => S"])
def test_coercion_sets_reject_every_malformed_surface_entry(entry: str) -> None:
    with pytest.raises(ValueError, match=r"coercion set\[0\] member\[0\]"):
        _build(
            RuleSetSpec(
                name="malformed",
                rules=(CoercionSetSpec(label="lexical", entries=(entry,)),),
            )
        )


def _empty_restriction() -> RuleRestrictionSpec:
    return RuleRestrictionSpec()


@pytest.mark.parametrize(
    ("constructor", "message"),
    [
        (lambda: ApplicationSpec(view=""), "needs a view"),
        (
            lambda: CompositionSpec(direction="unrestricted", order=0, view="/"),
            "positive",
        ),
        (
            lambda: CompositionSpec(direction="unrestricted", order=-1, view="/"),
            "positive",
        ),
        (lambda: SubstitutionSpec(order=0, view="/"), "positive"),
        (lambda: SubstitutionSpec(order=1, view=""), "needs a view"),
        (
            lambda: TypeRaisingSpec(view="", targets=None, arguments=None),
            "needs a view",
        ),
        (lambda: TypeRaisingSpec(view="/", targets=(), arguments=None), "nonempty"),
        (
            lambda: TypeRaisingSpec(view="/", targets=("",), arguments=None),
            "needs a name",
        ),
        (
            lambda: ClausalAdjunctSpec(name="", source_view="\\", target_view="/"),
            "needs a name",
        ),
        (
            lambda: ClausalAdjunctSpec(name="adjunct", source_view="", target_view="/"),
            "needs a source view",
        ),
        (
            lambda: ClausalAdjunctSpec(
                name="adjunct", source_view="\\", target_view=""
            ),
            "needs a target view",
        ),
        (
            lambda: ClausalAdjunctSpec(
                name="adjunct",
                source_view="\\",
                target_view="/",
                source_core="",
            ),
            "needs a source core",
        ),
        (
            lambda: ClausalAdjunctSpec(
                name="adjunct",
                source_view="\\",
                target_view="/",
                argument="",
            ),
            "needs an argument",
        ),
        (
            lambda: PatternRuleSpec(name="", premises=(X,), conclusion=X),
            "needs a name",
        ),
        (
            lambda: AbsorptionSpec(
                name="absorb",
                premises=(X, Y),
                conclusion=X,
                absorbing=(),
                side="right",
            ),
            "needs an absorbing category",
        ),
        (
            lambda: AbsorptionSpec(
                name="absorb",
                premises=(X, Y),
                conclusion=X,
                absorbing=("",),
                side="right",
            ),
            "category needs a name",
        ),
        (
            lambda: AbsorptionSpec(
                name="absorb",
                premises=(X, Y),
                conclusion=X,
                absorbing=("N",),
                side="right",
                other_must_be_nonabsorbing=False,
            ),
            "reject two absorbing premises",
        ),
        (
            lambda: CoordinationSpec(
                name="coordinate",
                premises=(X, Y),
                conclusion=X,
                phase="mark",
                core_relation="same",
                head_policy="left-conjunct",
            ),
            "exact head policy",
        ),
        (
            lambda: CoordinationSpec(
                name="coordinate",
                premises=(X, Y),
                conclusion=X,
                phase="merge",
                core_relation="different",
                head_policy="left-conjunct",
            ),
            "only a marking relation",
        ),
        (
            lambda: CoordinationSpec(
                name="coordinate",
                premises=(X, Y),
                conclusion=X,
                phase="merge",
                core_relation="same",
                head_policy="non-absorbing",
            ),
            "exact head policy",
        ),
        (
            lambda: CoordinationSpec(
                name="coordinate",
                premises=(X, Y),
                conclusion=X,
                phase="mark",
                core_relation="same",
                head_policy="non-absorbing",
                conjunction_feature="",
            ),
            "needs a conjunction feature",
        ),
        (
            lambda: CoordinationSpec(
                name="coordinate",
                premises=(X, Y),
                conclusion=X,
                phase="mark",
                core_relation="same",
                head_policy="non-absorbing",
                targets=(),
            ),
            "targets must be nonempty",
        ),
        (
            lambda: CoordinationSpec(
                name="coordinate",
                premises=(X, Y),
                conclusion=X,
                phase="mark",
                core_relation="same",
                head_policy="non-absorbing",
                targets=("",),
            ),
            "target needs a name",
        ),
        (
            lambda: CoercionSpec(label="", member=0, source="N", target="NP"),
            "needs a label",
        ),
        (
            lambda: CoercionSpec(label="lexical", member=-1, source="N", target="NP"),
            "nonnegative",
        ),
        (
            lambda: CoercionSpec(label="lexical", member=0, source="", target="NP"),
            "needs a source category",
        ),
        (
            lambda: CoercionSetSpec(label="lexical", entries=()),
            "needs an entry",
        ),
        (
            lambda: CombinatorSetSpec(composition=(0,)),
            "positive or unbounded",
        ),
        (
            lambda: CombinatorSetSpec(type_raising=False, raising_targets=("S",)),
            "require type raising",
        ),
        (_empty_restriction, "target or pattern"),
        (
            lambda: RuleRestrictionSpec(primary_targets=("",)),
            "entry needs a name",
        ),
        (
            lambda: SpinePatternSpec(
                name="C",
                core=X,
                operators=("/",),
                min_order=2,
                max_order=1,
            ),
            "too small",
        ),
        (
            lambda: SpinePatternSpec(name="C", core=X, operators=("",), max_order=None),
            "operator needs a name",
        ),
        (lambda: RuleSetSpec(name=""), "needs a name"),
    ],
)
def test_declaration_axioms_reject_every_current_invalid_constraint(
    constructor: Callable[[], dx.Model],
    message: str,
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        constructor()


@pytest.mark.parametrize(
    ("policy", "slot", "error", "message"),
    [
        ("functor", True, TypeError, "integer or None"),
        ("functor", -1, ValueError, "nonnegative"),
        ("unknown", 0, ValueError, "unknown head policy"),
    ],
)
def test_head_selection_rejects_invalid_runtime_values(
    policy: str,
    slot: int | None,
    error: type[Exception],
    message: str,
) -> None:
    with pytest.raises(error, match=message):
        HeadSelection(cast("HeadPolicy", policy), slot)


def test_built_rule_set_rejects_incomplete_or_invalid_selection_snapshots() -> None:
    runtime = object()

    with pytest.raises(TypeError, match="mapping"):
        BuiltRuleSet(runtime, (), cast("abc.Mapping[str, HeadSelection]", ()))

    with pytest.raises(ValueError, match="every runtime rule"):
        BuiltRuleSet(runtime, ("one", "two"), {"one": HeadSelection("functor", 0)})

    with pytest.raises(TypeError, match="only HeadSelection"):
        BuiltRuleSet(
            runtime,
            ("one",),
            cast(
                "abc.Mapping[str, HeadSelection]",
                {"one": "functor"},
            ),
        )


def test_rule_restrictions_lower_once_for_grammar_specific_sets() -> None:
    restriction = RuleRestrictionSpec(primary_targets=("S",))
    built = _build(
        RuleSetSpec(
            name="restricted",
            rules=(ApplicationSpec(view="/", restriction=restriction),),
        )
    )

    assert built.rule_set.restriction_mode == "grammar_specific"
    assert built.rule_set.rules[0].restriction == RuleRestriction(("S",))


def _reaches_boundary_model(value: object, seen: set[int]) -> bool:
    """Return whether a runtime object graph retains a declaration model."""
    if isinstance(value, dx.Model):
        return True

    identity = id(value)
    if identity in seen or isinstance(
        value, (str, bytes, int, float, bool, type(None))
    ):
        return False

    seen.add(identity)

    if isinstance(value, abc.Mapping):
        return any(
            _reaches_boundary_model(key, seen) or _reaches_boundary_model(member, seen)
            for key, member in value.items()
        )

    if isinstance(value, (tuple, list, set, frozenset)):
        return any(_reaches_boundary_model(member, seen) for member in value)

    slots = (
        slot
        for base in type(value).__mro__
        for slot in getattr(base, "__slots__", ())
        if isinstance(slot, str) and slot != "__weakref__"
    )
    return any(
        _reaches_boundary_model(getattr(value, slot), seen)
        for slot in slots
        if hasattr(value, slot)
    )


def test_runtime_rules_do_not_retain_or_reconstruct_boundary_models(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    built = _build(
        RuleSetSpec(
            name="runtime",
            rules=(PatternRuleSpec(name="identity", premises=(X,), conclusion=X),),
        )
    )
    rule = built.rule_set.rules[0]
    crossings = [0]

    assert not _reaches_boundary_model(built, set())

    def counted(self: dx.Model, **values: JsonValue) -> None:
        del self, values
        crossings[0] += 1

    monkeypatch.setattr(dx.Model, "__init__", counted)

    category = SYNTAX.parse(STORE, "N")
    for _ in range(100):
        assert tuple(rule.apply((category,))) == (category,)

    assert crossings == [0]


def test_root_declarations_refuse_unknown_tags_and_extra_fields() -> None:
    with pytest.raises(dx.ValidationError):
        RuleSpec.model_validate({"rule": "unknown", "head_policy": "functor"})

    with pytest.raises(dx.ValidationError):
        PatternRuleSpec.model_validate(
            {
                "name": "identity",
                "premises": [{"pattern": "term", "name": "X"}],
                "conclusion": {"pattern": "term", "name": "X"},
                "surplus": True,
            }
        )


def test_public_rule_declarations_and_head_policies_are_complete() -> None:
    assert tuple(RuleSpec.__variants__) == (
        "application",
        "composition",
        "substitution",
        "type-raising",
        "clausal-adjunct",
        "pattern",
        "absorption",
        "coordination",
        "coercion",
        "coercions",
        "combinators",
    )
    assert get_args(HeadPolicy.__value__) == (
        "functor",
        "pass-through",
        "non-absorbing",
        "left-conjunct",
    )


def test_a_rule_declaration_names_the_functor_views_it_uses() -> None:
    assert ApplicationSpec(view="/").functor_views() == ("/",)
    assert TypeRaisingSpec(
        view="\\", targets=("S",), arguments=None
    ).functor_views() == ("\\",)
    assert ClausalAdjunctSpec(
        name="adjunct", source_view="/", target_view="\\"
    ).functor_views() == ("/", "\\")
    assert (
        PatternRuleSpec(name="identity", premises=(X,), conclusion=X).functor_views()
        == ()
    )
