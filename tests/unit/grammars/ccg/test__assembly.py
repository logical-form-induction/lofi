import ast
import json
import pickle
from copy import copy, deepcopy
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import TYPE_CHECKING, Final, cast

import didactic.api as dx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.support.rules import ccg_signature, declared_rules
from tests.support.term_reference import infix_syntax

from lofi._native import SpanFree
from lofi.deduce import (
    CoverGoal,
    LogInside,
    NativeTermRule,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    parse,
)
from lofi.grammars.ccg import CCGAssembly, FunctorView
from lofi.grammars.ccg._assembly import _pattern_probe
from lofi.judge import (
    HOLE,
    Conjunctions,
    GeneralizedComposition,
    NoNormalForm,
    PatternRule,
    ProofSchema,
    RuleSet,
)
from lofi.judge.laws import RuleFixture, check_rule_laws
from lofi.lexicon import Entry, StaticLexicon
from lofi.spec import (
    AbsorptionSpec,
    ApplicationSpec,
    ClausalAdjunctSpec,
    CoercionSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    CoordinationSpec,
    DeclarationError,
    InfixConstructorNotationSpec,
    PatternRuleSpec,
    RuleContext,
    RuleRestrictionSpec,
    RuleSetSpec,
    RuleSpec,
    SubstitutionSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec._compile import canonical_declaration_bytes
from lofi.spec._matching import (
    AlphaMatchingSpec,
    FeatureErasureMatchingSpec,
    IdentityMatchingSpec,
)
from lofi.spec.baselines.bisk_hockenmaier import bisk_hockenmaier_grammar
from lofi.spec.grammar import (
    AdoptedRules,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FunctorSpec,
    GrammarDecl,
    GrammarIso,
    MatchingSpec,
    OperatorSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language
from lofi.terms import (
    Alpha,
    Bindings,
    FeatureErasure,
    Identity,
    MVar,
    Node,
    Operator,
    RepresentativeKernel,
    SExpSyntax,
    Signature,
    TargetSet,
    Term,
    TermMatching,
    TermStore,
    Vocabulary,
    build,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue
    from lofi.deduce import Cell, PackedForest
    from lofi.judge import CompositionRestriction, Rule
    from lofi.spec import BuiltRuleSet
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Matching


class _UnitScorer:
    """Assign zero log weight to every lexical and rule edge."""

    __slots__ = ()
    lowering = SpanFree()

    def axiom_weight(
        self,
        position: int,
        entry: Entry[Term],
        /,
    ) -> float:
        """Return the multiplicative unit in log space."""
        del position, entry

        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        """Return the multiplicative unit in log space."""
        del rule, inputs, output, span, split

        return 0.0


SIGNATURE_SPEC: Final = ccg_signature(
    "N",
    "NP",
    "S",
    "conj",
    ".",
    ",",
    ":",
    ";",
    "LRB",
    "RRB",
    operators=(
        OperatorSpec(name="N[conj]", sort="Cat", erases_to="N"),
        OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
        OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
        OperatorSpec(name="S[ng]", sort="Cat", erases_to="S"),
        OperatorSpec(name="S[conj]", sort="Cat", erases_to="S"),
        OperatorSpec(name="\\[conj]", sort="Cat", erases_to="\\"),
    ),
    conjunctions=ConjunctionsSpec(
        name="uppercase-ccgbank",
        markers=("conj", ","),
        pairs=(
            ConjunctionPairSpec(unmarked="N", marked="N[conj]"),
            ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),
            ConjunctionPairSpec(unmarked="S", marked="S[conj]"),
            ConjunctionPairSpec(unmarked="S\\NP", marked="(S\\NP)[conj]"),
        ),
    ),
    name="uppercase-ccgbank",
)
"""The declared uppercase CCGbank category signature every fixture compiles over."""


def _adopted(
    declaration: RuleSetSpec, signature: SignatureSpec = SIGNATURE_SPEC, /
) -> AdoptedRules:
    """Compile one rule inventory over a declared signature under feature erasure."""
    return declared_rules(signature, declaration, matching=FeatureErasureMatchingSpec())


def _built(
    declaration: RuleSetSpec, signature: SignatureSpec = SIGNATURE_SPEC, /
) -> BuiltRuleSet[RuleSet[Term]]:
    """Compile one rule inventory and return its rules with their head selections."""
    return _adopted(declaration, signature).built


COMPILED: Final = _adopted(
    RuleSetSpec(name="application", rules=(ApplicationSpec(view="/"),))
).context
"""The rule context the declared signature compiles to."""
STORE: Final = COMPILED.store
SIGNATURE: Final = STORE.signature
SYNTAX: Final = COMPILED.syntax
N: Final = SYNTAX.parse(STORE, "N")
N_CONJ: Final = SYNTAX.parse(STORE, "N[conj]")
NP: Final = SYNTAX.parse(STORE, "NP")
NP_CONJ: Final = SYNTAX.parse(STORE, "NP[conj]")
S_DCL: Final = SYNTAX.parse(STORE, "S[dcl]")
S_NG: Final = SYNTAX.parse(STORE, "S[ng]")
S_CONJ: Final = SYNTAX.parse(STORE, "S[conj]")
S_BACK_NP: Final = SYNTAX.parse(STORE, "S\\NP")
S_DCL_BACK_NP: Final = SYNTAX.parse(STORE, "S[dcl]\\NP")
S_NG_BACK_NP: Final = SYNTAX.parse(STORE, "S[ng]\\NP")
S_DCL_OVER_NP: Final = SYNTAX.parse(STORE, "S[dcl]/NP")
N_CONJ_BACK_NP: Final = SYNTAX.parse(STORE, "N[conj]\\NP")
S_DCL_BACK_N: Final = SYNTAX.parse(STORE, "S[dcl]\\N")
S_BACK_NP_CONJ: Final = SYNTAX.parse(STORE, "(S\\NP)[conj]")
NP_BACK_NP: Final = SYNTAX.parse(STORE, "NP\\NP")
NP_BACK_N: Final = SYNTAX.parse(STORE, "NP\\N")
NP_OVER_NP: Final = SYNTAX.parse(STORE, "NP/NP")
S_DCL_BACK_S: Final = SYNTAX.parse(STORE, "S[dcl]\\S")
CONJ: Final = SYNTAX.parse(STORE, "conj")
PERIOD: Final = SYNTAX.parse(STORE, ".")
COMMA: Final = SYNTAX.parse(STORE, ",")
COLON: Final = SYNTAX.parse(STORE, ":")
SEMICOLON: Final = SYNTAX.parse(STORE, ";")
LRB: Final = SYNTAX.parse(STORE, "LRB")
RRB: Final = SYNTAX.parse(STORE, "RRB")
CATEGORY_SOURCES: Final = (
    "N",
    "N[conj]",
    "NP",
    "NP[conj]",
    "S",
    "S[dcl]",
    "S[ng]",
    "S[conj]",
    "conj",
    ".",
    ",",
    ":",
    ";",
    "LRB",
    "RRB",
    "S\\NP",
    "S[dcl]\\NP",
    "S[ng]\\NP",
    "S[dcl]/NP",
    "N[conj]\\NP",
    "S[dcl]\\N",
    "(S\\NP)[conj]",
    "NP\\NP",
    "NP\\N",
    "NP/NP",
    "S[dcl]\\S",
    "S\\S",
)
CONJUNCTIONS: Final = cast("Conjunctions[Term]", COMPILED.conjunctions)
FUNCTORS: Final = FunctorView(STORE, "/", "\\", result=0, argument=1)


def _views(functors: FunctorView, /) -> dict[str, tuple[FunctorView, str]]:
    """Name both orientations of one runtime functor view."""
    return {
        functors.forward_slash: (functors, functors.forward_slash),
        functors.backward_slash: (functors, functors.backward_slash),
    }


ASSEMBLY: Final = CCGAssembly(
    STORE,
    SYNTAX,
    partial(SYNTAX.parse, STORE),
    _views(FUNCTORS),
    FeatureErasure(),
    CONJUNCTIONS,
    NoNormalForm[Term](),
    category_sort="Cat",
    categories=CATEGORY_SOURCES,
    rule_matchings=(
        (
            canonical_declaration_bytes(IdentityMatchingSpec()).decode(),
            Identity(),
        ),
        (canonical_declaration_bytes(AlphaMatchingSpec()).decode(), Alpha()),
    ),
)
X: Final = TermPatternSpec(name="X", sort="Cat")
Y: Final = TermPatternSpec(name="Y", sort="Cat")


def _one(declaration: RuleSpec, /) -> Rule[Term]:
    """Compile one rule over the declared signature."""
    (rule,) = _built(RuleSetSpec(name="single-rule", rules=(declaration,))).rule_set
    return rule


N_PATTERN: Final = ConstantPatternSpec(name="N")
NP_PATTERN: Final = ConstantPatternSpec(name="NP")
CONJ_PATTERN: Final = ConstantPatternSpec(name="conj")
PUNCTUATION: Final = (".", ",", ":", ";", "LRB", "RRB")
LAW_TERMS: Final = (
    N,
    N_CONJ,
    NP,
    NP_CONJ,
    S_DCL,
    S_NG,
    S_CONJ,
    S_BACK_NP,
    S_DCL_BACK_NP,
    S_NG_BACK_NP,
    S_DCL_OVER_NP,
    N_CONJ_BACK_NP,
    S_DCL_BACK_N,
    S_BACK_NP_CONJ,
    NP_BACK_NP,
    NP_BACK_N,
    NP_OVER_NP,
    S_DCL_BACK_S,
    CONJ,
    PERIOD,
    COMMA,
    COLON,
    SEMICOLON,
    LRB,
    RRB,
)


def _declaration(*, unlike_targets: tuple[str, ...] | None = None) -> RuleSetSpec:
    """Build the shared cross-layer declaration."""
    return RuleSetSpec(
        name="ccgbank-exceptions",
        rules=(
            ApplicationSpec(view="/"),
            TypeRaisingSpec(view="\\", targets=("S[dcl]",), arguments=None),
            PatternRuleSpec(
                name="n-to-np",
                premises=(N_PATTERN,),
                conclusion=NP_PATTERN,
                head_policy="pass-through",
            ),
            AbsorptionSpec(
                name="punctuation-left",
                premises=(X, Y),
                conclusion=Y,
                absorbing=PUNCTUATION,
                side="left",
            ),
            AbsorptionSpec(
                name="punctuation-right",
                premises=(X, Y),
                conclusion=X,
                absorbing=PUNCTUATION,
                side="right",
            ),
            CoordinationSpec(
                name="coordination-mark",
                premises=(CONJ_PATTERN, X),
                conclusion=X,
                phase="mark",
                core_relation="same",
                head_policy="non-absorbing",
            ),
            CoordinationSpec(
                name="coordination-merge",
                premises=(X, X),
                conclusion=X,
                phase="merge",
                core_relation="same",
                head_policy="left-conjunct",
            ),
            CoordinationSpec(
                name="coordination-unlike",
                premises=(CONJ_PATTERN, Y),
                conclusion=X,
                phase="mark",
                core_relation="different",
                head_policy="non-absorbing",
                targets=unlike_targets,
            ),
        ),
    )


def _typed_context(
    assembly: CCGAssembly,
) -> RuleContext[
    Rule[Term],
    RuleSet[Term],
    Term,
]:
    """Return the structural rule construction context."""
    return assembly


def _clausal_adjunct() -> ClausalAdjunctSpec:
    """Build the shared clausal-adjunct declaration."""
    return ClausalAdjunctSpec(
        name="backward-clausal-adjunct",
        source_view="\\",
        target_view="\\",
    )


def _assembly_law_terms(rule_name: str) -> tuple[Term, ...]:
    """Return a compact term domain that exercises one assembly rule."""
    if rule_name.startswith("punctuation-"):
        return (COMMA, NP)

    if rule_name == "coordination-mark":
        return (CONJ, NP, N)

    if rule_name == "coordination-merge":
        return (NP, NP_CONJ)

    return (CONJ, N, NP_CONJ)


def test_the_assembly_is_a_frozen_structural_build_context() -> None:
    context = _typed_context(ASSEMBLY)

    assert context is ASSEMBLY
    assert ASSEMBLY.store is STORE
    assert ASSEMBLY.syntax is SYNTAX
    assert ASSEMBLY.conjunctions is CONJUNCTIONS
    assert ASSEMBLY.functors == _views(FUNCTORS)
    assert copy(ASSEMBLY) is ASSEMBLY
    assert deepcopy(ASSEMBLY) is ASSEMBLY

    with pytest.raises(AttributeError, match="immutable"):
        ASSEMBLY.__setattr__("_syntax", SYNTAX)


def test_an_assembly_without_conjunctions_runs_under_the_declared_matching() -> None:
    matching = FeatureErasure()
    assembly = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        _views(FUNCTORS),
        matching,
        None,
        NoNormalForm[Term](),
        category_sort="Cat",
        categories=CATEGORY_SOURCES,
    )

    assert assembly.conjunctions is None
    assert assembly.matching is matching
    assert assembly.matching.equivalent(NP_CONJ, NP)
    assert ASSEMBLY.matching.equivalent(NP_CONJ, NP) is False
    assert ASSEMBLY.matching.name.startswith("ccg-feature-erasure:uppercase-ccgbank:")

    plain = _adopted(
        RuleSetSpec(name="application", rules=(ApplicationSpec(view="/"),)),
        SIGNATURE_SPEC.with_(conjunctions=None),
    )
    (application,) = plain.rule_set.rules
    store = plain.context.store
    syntax = plain.context.syntax

    assert plain.context.conjunctions is None
    assert plain.context.matching.equivalent(
        syntax.parse(store, "NP[conj]"), syntax.parse(store, "NP")
    )
    assert tuple(
        application.apply((syntax.parse(store, "S[dcl]/NP"), syntax.parse(store, "NP")))
    ) == (syntax.parse(store, "S[dcl]"),)


def _rule(matching: MatchingSpec, slash: str) -> Rule[Term]:
    """Compile the declared application rule over one slash under one matching."""
    (application,) = declared_rules(
        SIGNATURE_SPEC,
        RuleSetSpec(name="application", rules=(ApplicationSpec(view=slash),)),
        matching=matching,
    ).rule_set.rules

    return application


def _marked(matching: str) -> GrammarAssembly:
    """Build a declared grammar with a conjunction inventory under one matching."""
    coordination = (
        CoordinationSpec(
            name="coordination-mark",
            premises=(CONJ_PATTERN, X),
            conclusion=X,
            phase="mark",
            core_relation="same",
            head_policy="non-absorbing",
        ),
        CoordinationSpec(
            name="coordination-merge",
            premises=(X, X),
            conclusion=X,
            phase="merge",
            core_relation="same",
            head_policy="left-conjunct",
        ),
    )
    declaration = GrammarDecl.model_validate(
        {
            "name": "marked",
            "signature": {
                "primitives": ["S", "NP", "conj"],
                "operators": [
                    {"name": "NP[conj]", "sort": "Cat", "erases_to": "NP"},
                    {"name": "S[dcl]", "sort": "Cat", "erases_to": "S"},
                ],
                "constructors": [
                    {
                        "name": "/",
                        "kind": ["Cat", "Cat", "Cat"],
                        "notation": {
                            "style": "infix",
                            "spelling": "/",
                            "precedence": 50,
                            "associativity": "left",
                        },
                    },
                    {
                        "name": "\\",
                        "kind": ["Cat", "Cat", "Cat"],
                        "notation": {
                            "style": "infix",
                            "spelling": "\\",
                            "precedence": 50,
                            "associativity": "left",
                        },
                    },
                ],
                "functors": [
                    {
                        "name": "/",
                        "constructor": "/",
                        "argument_side": "right",
                        "opposite": "\\",
                    },
                    {
                        "name": "\\",
                        "constructor": "\\",
                        "argument_side": "left",
                        "opposite": "/",
                    },
                ],
                "conjunctions": {
                    "markers": ["conj"],
                    "pairs": [{"unmarked": "NP", "marked": "NP[conj]"}],
                },
            },
            "rules": {
                "name": "application",
                "rules": [
                    {"rule": "application", "view": "\\"},
                    *(json.loads(rule.model_dump_json()) for rule in coordination),
                ],
            },
            "lexicon": {
                "kind": "inline",
                "entries": [
                    {"form": "a", "category": "NP"},
                    {"form": "and", "category": "conj"},
                    {"form": "b", "category": "S[dcl]\\NP"},
                ],
            },
            "goal": {"kind": "root", "categories": ["S[dcl]"]},
            "matching": {"kind": matching},
        }
    )

    return GrammarIso().forward(declaration, registered_language("propositional"))


def _rules_by_name(assembly: GrammarAssembly) -> dict[str, Rule[Term]]:
    """Return a built grammar's rules by name."""
    return {rule.name: rule for rule in assembly.grammar.rules.rules}


def test_a_declared_feature_erasure_grammar_keeps_its_conjunction_marks() -> None:
    assembly = _marked("feature-erasure")
    rules = _rules_by_name(assembly)

    def category(text: str) -> Term:
        return assembly.syntax.parse(assembly.store, text)

    matching = assembly.grammar.rules.matching
    application = rules["application:\\"]

    assert matching.equivalent(category("S[dcl]"), category("S"))
    assert matching.equivalent(category("S[dcl]\\NP"), category("S\\NP"))
    assert matching.equivalent(category("NP[conj]"), category("NP[conj]"))
    assert not matching.equivalent(category("NP[conj]"), category("NP"))
    assert tuple(application.apply((category("NP"), category("S\\NP")))) == (
        category("S"),
    )
    assert (
        tuple(application.apply((category("NP[conj]"), category("S[dcl]\\NP")))) == ()
    )


@pytest.mark.parametrize("matching", ["identity", "alpha"])
def test_a_declared_grammar_marks_conjunctions_under_its_own_matching(
    matching: str,
) -> None:
    assembly = _marked(matching)
    rules = _rules_by_name(assembly)

    def category(text: str) -> Term:
        return assembly.syntax.parse(assembly.store, text)

    built = assembly.grammar.rules.matching
    application = rules["application:\\"]

    assert built.name == matching
    assert not built.equivalent(category("S[dcl]"), category("S"))
    assert not built.equivalent(category("NP[conj]"), category("NP"))
    assert built.equivalent(category("NP[conj]"), category("NP[conj]"))
    assert tuple(application.apply((category("NP"), category("S[dcl]\\NP")))) == (
        category("S[dcl]"),
    )
    assert tuple(application.apply((category("NP"), category("S\\S")))) == ()
    assert (
        tuple(application.apply((category("NP[conj]"), category("S[dcl]\\NP")))) == ()
    )
    assert tuple(
        rules["coordination-mark"].apply((category("conj"), category("NP")))
    ) == (category("NP[conj]"),)
    assert tuple(
        rules["coordination-merge"].apply((category("NP"), category("NP[conj]")))
    ) == (category("NP"),)


@pytest.mark.parametrize("matching", ["identity", "feature-erasure"])
def test_a_root_directed_parse_keeps_a_coordination_under_its_matching(
    matching: str,
) -> None:
    assembly = _marked(matching)
    grammar = assembly.grammar

    def rooted(reachability: Reachability) -> tuple[str, ...]:
        chart = parse(
            ("a", "and", "a", "b"),
            rules=grammar.rules,
            source=assembly.lexicon,
            semiring=LogInside(),
            scorer=_UnitScorer(),
            goal=grammar.goal,
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, reachability),
        )
        return tuple(
            sorted(
                assembly.syntax.render(assembly.store, key.reading)
                for key in chart.items_at((1, 3))
            )
        )

    assert rooted(Reachability.ALL) == ("NP[conj]",)
    assert rooted(Reachability.ROOTS) == ("NP[conj]",)


def test_conjunction_marks_compose_with_the_declared_matching() -> None:
    feature_erasure = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        _views(FUNCTORS),
        FeatureErasure(),
        CONJUNCTIONS,
        NoNormalForm[Term](),
        category_sort="Cat",
        categories=CATEGORY_SOURCES,
    )
    identity = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        _views(FUNCTORS),
        Identity[Term](),
        CONJUNCTIONS,
        NoNormalForm[Term](),
        category_sort="Cat",
        categories=CATEGORY_SOURCES,
    )

    assert identity.matching.name == "identity"
    assert feature_erasure.matching.name.startswith(
        "ccg-feature-erasure:uppercase-ccgbank:"
    )
    assert identity.matching.name != feature_erasure.matching.name
    assert feature_erasure.matching.equivalent(S_DCL_BACK_NP, S_BACK_NP)
    assert not identity.matching.equivalent(S_DCL_BACK_NP, S_BACK_NP)
    assert not identity.matching.equivalent(NP_CONJ, NP)
    assert identity.matching.equivalent(NP_CONJ, NP_CONJ)
    erased = FeatureErasureMatchingSpec()
    exact = IdentityMatchingSpec()

    assert tuple(_rule(erased, "\\").apply((S_DCL, S_DCL_BACK_S))) == (S_DCL,)
    assert tuple(_rule(exact, "\\").apply((S_DCL, S_DCL_BACK_S))) == ()
    assert tuple(_rule(exact, "\\").apply((S_NG, S_DCL_BACK_S))) == ()


class _Keyless:
    """Feature erasure that offers no representative keys."""

    __slots__ = ()

    @property
    def name(self) -> str:
        """Return the relation name."""
        return "keyless"

    @property
    def has_key(self) -> bool:
        """Report that no representative keys are available."""
        return False

    @property
    def representatives(self) -> RepresentativeKernel:
        """Compile feature-erasure representatives for native rule execution."""
        return RepresentativeKernel.core()

    def key(self, term: Term, /) -> Term:
        """Return the feature-erased term."""
        return term.core

    def equivalent(self, left: Term, right: Term, /) -> bool:
        """Return whether two terms erase to the same term."""
        return left.core is right.core


def test_conjunction_marks_offer_no_keys_over_a_keyless_matching() -> None:
    matching = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        _views(FUNCTORS),
        _Keyless(),
        CONJUNCTIONS,
        NoNormalForm[Term](),
        category_sort="Cat",
        categories=CATEGORY_SOURCES,
    ).matching

    assert not matching.has_key
    assert matching.name.startswith("ccg-keyless:uppercase-ccgbank:")
    assert matching.equivalent(S_DCL_BACK_NP, S_BACK_NP)
    assert not matching.equivalent(NP_CONJ, NP)


MARKED: Final = frozenset(marked for _, marked in CONJUNCTIONS.pairs)


@given(
    st.sampled_from((Identity[Term](), FeatureErasure())),
    st.sampled_from(LAW_TERMS),
    st.sampled_from(LAW_TERMS),
)
def test_a_marked_matching_refines_the_declared_matching_only_at_marks(
    base: TermMatching, left: Term, right: Term
) -> None:
    matching = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        _views(FUNCTORS),
        base,
        CONJUNCTIONS,
        NoNormalForm[Term](),
        category_sort="Cat",
        categories=CATEGORY_SOURCES,
    ).matching
    equivalent = matching.equivalent(left, right)

    assert matching.has_key is base.has_key
    assert equivalent is (matching.key(left) is matching.key(right))
    assert equivalent is matching.equivalent(right, left)

    if left not in MARKED and right not in MARKED:
        assert equivalent is base.equivalent(left, right)
    elif (left in MARKED) is not (right in MARKED):
        assert not equivalent
    else:
        assert equivalent is any(
            first is left and second is right and base.equivalent(one, other)
            for one, first in CONJUNCTIONS.pairs
            for other, second in CONJUNCTIONS.pairs
        )


def test_assembly_values_copy_by_identity_and_refuse_pickle() -> None:
    rules = _built(_declaration()).rule_set.rules
    values = (ASSEMBLY, FUNCTORS, ASSEMBLY.matching, rules[3], rules[5])

    for value in values:
        assert copy(value) is value
        assert deepcopy(value) is value

        with pytest.raises(TypeError, match="cannot be pickled"):
            pickle.dumps(value)


def test_category_resolution_preserves_corpus_case_and_slashes() -> None:
    assert ASSEMBLY.category("N") is N
    assert ASSEMBLY.category("S[dcl]\\NP") is SYNTAX.parse(STORE, "S[dcl]\\NP")

    with pytest.raises(ValueError, match="was not declared by the CCG assembly"):
        ASSEMBLY.category("n")


def test_category_resolution_rejects_terms_of_another_sort() -> None:
    signature = Signature(
        "two-sort-ccg",
        ("Cat", "Meaning"),
        (
            Operator("meaning", "Meaning"),
            Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
            Operator("\\", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        ),
        vocabularies=(Vocabulary("Cat", ("N", "N[conj]", "conj")),),
        constant_cores={"Cat": {"N[conj]": "N"}},
    )
    store = TermStore(signature)
    syntax = infix_syntax(signature, ("/", "/"), ("\\", "\\"))
    assembly_syntax = SExpSyntax(signature)
    n = syntax.parse(store, "N")
    with pytest.raises(ValueError, match="expected category sort"):
        CCGAssembly(
            store,
            assembly_syntax,
            partial(assembly_syntax.parse, store),
            _views(FunctorView(store, "/", "\\", result=0, argument=1)),
            FeatureErasure(),
            Conjunctions(
                "two-sort-ccg",
                (syntax.parse(store, "conj"),),
                ((n, syntax.parse(store, "N[conj]")),),
            ),
            NoNormalForm[Term](),
            category_sort="Cat",
            categories=('("meaning")',),
        )


def test_generic_factories_retain_the_assembly_matching() -> None:
    declarations = (
        ApplicationSpec(view="/"),
        SubstitutionSpec(order=1, view="/"),
        TypeRaisingSpec(view="\\", targets=("S[dcl]",), arguments=None),
    )
    adopted = _adopted(RuleSetSpec(name="generic", rules=declarations))

    assert all(rule.matching is adopted.context.matching for rule in adopted.rule_set)


def test_every_rule_kind_matches_under_the_congruence_it_declares() -> None:
    exact = IdentityMatchingSpec()
    declarations = (
        *_declaration().rules,
        CompositionSpec(direction="unrestricted", order=1, view="/"),
        SubstitutionSpec(order=1, view="/"),
        CoercionSpec(label="lexical", member=0, source="N", target="NP"),
        _clausal_adjunct(),
    )

    for declaration in declarations:
        adopted = _adopted(RuleSetSpec(name="single-rule", rules=(declaration,)))
        (rule,) = adopted.rule_set

        assert rule.matching is adopted.context.matching
        assert type(_one(declaration.with_(matching=exact)).matching) is Identity


def test_a_coercion_declared_exact_fires_only_on_the_featured_premise_it_names() -> (
    None
):
    declared = CoercionSpec(
        label="lexical", member=0, source="S[dcl]\\NP", target="NP\\NP"
    )
    erased = _one(declared)
    exact = _one(declared.with_(matching=IdentityMatchingSpec()))

    assert tuple(exact.apply((S_DCL_BACK_NP,))) == (NP_BACK_NP,)
    assert tuple(exact.apply((S_NG_BACK_NP,))) == ()
    assert tuple(exact.apply((S_BACK_NP,))) == ()
    assert tuple(erased.apply((S_DCL_BACK_NP,))) == (NP_BACK_NP,)
    assert tuple(erased.apply((S_NG_BACK_NP,))) == (NP_BACK_NP,)
    assert tuple(erased.apply((S_BACK_NP,))) == (NP_BACK_NP,)
    assert exact.admits((S_DCL_BACK_NP,), NP_BACK_NP)
    assert not exact.admits((S_NG_BACK_NP,), NP_BACK_NP)
    assert erased.admits((S_NG_BACK_NP,), NP_BACK_NP)


def test_assembly_returns_native_and_private_wrapper_proof_schemas() -> None:
    rules = _built(_declaration()).rule_set.rules
    expected = (
        (((X.build(ASSEMBLY), Y.build(ASSEMBLY)), Y.build(ASSEMBLY)),),
        (((X.build(ASSEMBLY), Y.build(ASSEMBLY)), X.build(ASSEMBLY)),),
        (
            (
                (CONJ_PATTERN.build(ASSEMBLY), X.build(ASSEMBLY)),
                X.build(ASSEMBLY),
            ),
        ),
        (((X.build(ASSEMBLY), X.build(ASSEMBLY)), X.build(ASSEMBLY)),),
        (
            (
                (CONJ_PATTERN.build(ASSEMBLY), Y.build(ASSEMBLY)),
                X.build(ASSEMBLY),
            ),
        ),
    )

    for rule in rules:
        schema = ASSEMBLY.proof_schema(rule)

        assert schema is rule
        assert isinstance(schema, ProofSchema)

    for rule, schemata in zip(rules[3:], expected, strict=True):
        schema = ASSEMBLY.proof_schema(rule)

        assert schema is not None
        assert schema.proof_schemata == schemata


def test_clausal_adjunct_schema_keeps_structure_but_omits_the_feature() -> None:
    rule = _one(_clausal_adjunct())
    schema = ASSEMBLY.proof_schema(rule)
    variable = MVar("X", "Cat")

    assert schema is rule
    assert schema.proof_schemata == (
        (
            (
                Node(
                    "\\",
                    (
                        Node(
                            "#const[Cat]",
                            payload=SIGNATURE.vocabulary("Cat").id("S"),
                        ),
                        Node(
                            "#const[Cat]",
                            payload=SIGNATURE.vocabulary("Cat").id("NP"),
                        ),
                    ),
                ),
            ),
            Node("\\", (variable, variable)),
        ),
    )


def test_clausal_adjunct_checks_featured_sources_and_repeated_targets() -> None:
    built = _built(
        RuleSetSpec(
            name="clausal-adjunct",
            rules=(_clausal_adjunct(),),
        )
    )
    rule = built.rule_set.rules[0]

    assert rule.name == "backward-clausal-adjunct"
    assert rule.arity == 1
    assert rule.directions == frozenset({"check"})
    assert rule.residual_slots == frozenset()
    assert not rule.forward_is_complete
    assert rule.matching is built.rule_set.matching
    assert rule.restriction is None
    assert rule.probe is None
    assert dict(built.head_policies) == {"backward-clausal-adjunct": "pass-through"}
    assert rule.admits((S_DCL_BACK_NP,), NP_BACK_NP)
    assert rule.admits((S_NG_BACK_NP,), NP_BACK_NP)
    assert rule.admits((S_DCL_BACK_NP,), S_DCL_BACK_S)
    assert tuple(rule.apply((S_DCL_BACK_NP,))) == ()
    assert isinstance(rule, NativeTermRule)
    assert tuple(rule.lowering.apply((S_DCL_BACK_NP,))) == ()
    assert tuple(rule.residuate(NP_BACK_NP, (HOLE,))) == ()


@pytest.mark.parametrize(
    ("source_slash", "target_slash", "source", "output"),
    [
        ("\\", "\\", S_DCL_BACK_NP, NP_BACK_NP),
        ("\\", "/", S_DCL_BACK_NP, NP_OVER_NP),
        ("/", "\\", S_DCL_OVER_NP, NP_BACK_NP),
        ("/", "/", S_DCL_OVER_NP, NP_OVER_NP),
    ],
)
def test_clausal_adjunct_accepts_every_slash_pair(
    source_slash: str,
    target_slash: str,
    source: Term,
    output: Term,
) -> None:
    rule = _one(
        ClausalAdjunctSpec(
            name="clausal-adjunct",
            source_view=source_slash,
            target_view=target_slash,
        )
    )

    assert rule.admits((source,), output)


def test_clausal_adjunct_rejects_every_out_of_domain_shape() -> None:
    rule = _one(_clausal_adjunct())

    assert not rule.admits((S_BACK_NP,), NP_BACK_NP)
    assert not rule.admits((S_DCL_OVER_NP,), NP_BACK_NP)
    assert not rule.admits((N_CONJ_BACK_NP,), NP_BACK_NP)
    assert not rule.admits((S_DCL_BACK_N,), NP_BACK_NP)
    assert not rule.admits((S_DCL_BACK_NP,), NP_OVER_NP)
    assert not rule.admits((S_DCL_BACK_NP,), NP_BACK_N)
    assert not rule.admits((), NP_BACK_NP)
    assert not rule.admits((S_DCL_BACK_NP, S_DCL_BACK_NP), NP_BACK_NP)

    foreign_store = TermStore(SIGNATURE)
    foreign_source = SYNTAX.parse(foreign_store, "S[dcl]\\NP")
    foreign_output = SYNTAX.parse(foreign_store, "NP\\NP")

    assert not rule.admits((foreign_source,), NP_BACK_NP)
    assert not rule.admits((S_DCL_BACK_NP,), foreign_output)


@pytest.mark.parametrize(
    ("source_slash", "target_slash"),
    [("|", "\\"), ("\\", "|")],
)
def test_clausal_adjunct_rejects_unknown_views(
    source_slash: str,
    target_slash: str,
) -> None:
    declaration = ClausalAdjunctSpec(
        name="bad-clausal-adjunct",
        source_view=source_slash,
        target_view=target_slash,
    )

    with pytest.raises(dx.ValidationError, match="names '\\|', which is no functor"):
        _one(declaration)


def test_clausal_adjunct_rejects_a_nonatomic_argument() -> None:
    declaration = ClausalAdjunctSpec(
        name="bad-clausal-adjunct",
        source_view="\\",
        target_view="\\",
        argument="S\\NP",
    )

    with pytest.raises(ValueError, match="argument must be an atom"):
        _one(declaration)


@pytest.mark.parametrize("source_core", ["S\\NP", "S[dcl]"])
def test_clausal_adjunct_rejects_a_nonatomic_or_featured_source_core(
    source_core: str,
) -> None:
    declaration = ClausalAdjunctSpec(
        name="bad-clausal-adjunct",
        source_view="\\",
        target_view="\\",
        source_core=source_core,
    )

    with pytest.raises(ValueError, match="source core must be an unfeatured atom"):
        _one(declaration)


def test_clausal_adjunct_is_frozen_and_declares_its_runtime_shape() -> None:
    rule = _one(_clausal_adjunct())

    assert copy(rule) is rule
    assert deepcopy(rule) is rule
    assert rule.to_spec() == {
        "rule": "ccg-clausal-adjunct",
        "name": "backward-clausal-adjunct",
        "source_view": "\\",
        "target_view": "\\",
        "source_core": "S",
        "argument": "NP",
        "head_policy": "pass-through",
        "matching": ASSEMBLY.matching.name,
        "restriction": None,
    }

    with pytest.raises(AttributeError, match="immutable"):
        rule.__setattr__("_name", "changed")

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(rule)


def test_a_restricted_clausal_adjunct_is_refused() -> None:
    declaration = ClausalAdjunctSpec(
        name="restricted-clausal-adjunct",
        source_view="\\",
        target_view="\\",
        restriction=RuleRestrictionSpec(primary_targets=("S",)),
    )

    with pytest.raises(DeclarationError, match="no primary premise"):
        _one(declaration)


def test_clausal_adjunct_satisfies_every_check_only_rule_law() -> None:
    declaration = _clausal_adjunct()
    rule = _one(declaration)

    def rebuild(spec: Mapping[str, JsonValue]) -> Rule[Term]:
        """Rebuild the declared clausal-adjunct rule."""
        rebuilt = _one(declaration)

        if rebuilt.to_spec() != spec:
            msg = "the rebuilt clausal-adjunct rule changed its declaration"
            raise ValueError(msg)

        return rebuilt

    check_rule_laws(
        RuleFixture(
            rule,
            st.sampled_from(LAW_TERMS),
            rebuild,
            max_examples=300,
        )
    )


def test_the_cross_layer_declaration_builds_under_one_matching_identity() -> None:
    built = _built(_declaration())
    rule_set = built.rule_set
    rules = rule_set.rules

    assert tuple(rule.name for rule in rule_set) == (
        "application:/",
        "type-raising:\\:targets=S[dcl]",
        "n-to-np",
        "punctuation-left",
        "punctuation-right",
        "coordination-mark",
        "coordination-merge",
        "coordination-unlike",
    )
    assert all(rule.matching is rule_set.matching for rule in rules)
    assert rule_set.matching.name == ASSEMBLY.matching.name
    assert type(rule_set.normal_form) is NoNormalForm
    assert rules[2].probe is not None
    assert rules[2].probe.alternatives == ((frozenset({"#const[Cat]"}),),)
    assert rule_set.index is not None
    assert rule_set.index.unary_candidates(N) == (rules[1], rules[2])
    assert rule_set.index.unary_candidates(NP) == (rules[1],)
    assert dict(built.head_policies) == {
        "application:/": "functor",
        "type-raising:\\:targets=S[dcl]": "pass-through",
        "n-to-np": "pass-through",
        "punctuation-left": "non-absorbing",
        "punctuation-right": "non-absorbing",
        "coordination-mark": "non-absorbing",
        "coordination-merge": "left-conjunct",
        "coordination-unlike": "non-absorbing",
    }
    assert tuple(rules[2].apply((N,))) == (NP,)


def _forest_snapshot(
    forest: PackedForest[Term, float],
) -> tuple[object, ...]:
    """Return every chart item, edge, and inside weight in arrival order."""
    return tuple(
        (
            span,
            tuple(
                (
                    key,
                    forest.term(key),
                    forest.inside(key),
                    forest.edges(key),
                )
                for key in forest.items_at(span)
            ),
        )
        for span in forest.index.regions_by_size()
    )


@pytest.mark.parametrize(
    ("sentence", "entries"),
    [
        (
            ("functor", "argument"),
            (
                Entry("functor", S_DCL_OVER_NP),
                Entry("argument", NP),
            ),
        ),
        (
            ("punctuation", "noun-phrase"),
            (
                Entry("punctuation", COMMA),
                Entry("noun-phrase", NP),
            ),
        ),
        (
            ("left", "and", "right"),
            (
                Entry("left", NP),
                Entry("and", CONJ),
                Entry("right", NP),
            ),
        ),
        (
            ("and", "noun"),
            (
                Entry("and", CONJ),
                Entry("noun", N),
            ),
        ),
    ],
    ids=["application", "punctuation", "coordination", "unlike-coordination"],
)
def test_indexed_and_unindexed_ccg_charts_are_exactly_equal(
    sentence: tuple[str, ...],
    entries: tuple[Entry[Term], ...],
) -> None:
    """Require indexing to preserve exhaustive chart contents and counts."""
    indexed = _built(_declaration(unlike_targets=("NP",))).rule_set
    plain = RuleSet(
        indexed.name,
        indexed.rules,
        indexed.matching,
        indexed.normal_form,
        indexed.restriction_mode,
    )
    source = StaticLexicon(entries)
    scorer = _UnitScorer()
    goal = CoverGoal(len(sentence))
    indexed_forest = parse(
        sentence,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=indexed,
        source=source,
        semiring=LogInside(),
        scorer=scorer,
        goal=goal,
        unary_depth=2,
    )
    plain_forest = parse(
        sentence,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=plain,
        source=source,
        semiring=LogInside(),
        scorer=scorer,
        goal=goal,
        unary_depth=2,
    )

    assert indexed.index is not None
    assert plain.index is None
    assert indexed_forest.num_items == plain_forest.num_items
    assert indexed_forest.num_edges == plain_forest.num_edges
    assert _forest_snapshot(indexed_forest) == _forest_snapshot(plain_forest)


def test_ccg_index_excludes_exceptional_rules_with_impossible_heads() -> None:
    """Require exceptional-rule probes to reject irrelevant premise pairs."""
    rules = _built(_declaration(unlike_targets=("NP",))).rule_set

    if rules.index is None:
        pytest.fail("the CCG assembly did not attach its binary-rule index")

    application = {rule.name for rule in rules.index.candidates(S_DCL_OVER_NP, NP)}
    marking = {rule.name for rule in rules.index.candidates(CONJ, NP)}
    merging = {rule.name for rule in rules.index.candidates(NP, NP_CONJ)}

    assert "punctuation-left" not in application
    assert "punctuation-right" in application
    assert "coordination-mark" not in application
    assert "coordination-merge" not in application
    assert "coordination-unlike" not in application
    assert "coordination-mark" in marking
    assert "coordination-unlike" in marking
    assert "coordination-merge" in merging


def test_absorption_requires_exactly_one_declared_punctuation_category() -> None:
    rules = _built(_declaration()).rule_set.rules
    left = rules[3]
    right = rules[4]

    assert isinstance(left, NativeTermRule)
    assert isinstance(right, NativeTermRule)
    assert tuple(left.apply((COMMA, NP))) == (NP,)
    assert tuple(right.apply((NP, PERIOD))) == (NP,)
    assert tuple(left.lowering.apply((COMMA, NP))) == (NP,)
    assert tuple(right.lowering.apply((NP, PERIOD))) == (NP,)
    assert tuple(left.apply((COMMA, PERIOD))) == ()
    assert tuple(right.apply((COMMA, PERIOD))) == ()
    assert not left.admits((COMMA, PERIOD), PERIOD)
    assert tuple(left.residuate(NP, (HOLE, NP))) == (
        PERIOD,
        COMMA,
        COLON,
        SEMICOLON,
        LRB,
        RRB,
    )
    assert tuple(right.residuate(NP, (NP, HOLE))) == (
        PERIOD,
        COMMA,
        COLON,
        SEMICOLON,
        LRB,
        RRB,
    )
    left_rows = left.reachability_schemata
    right_rows = right.reachability_schemata

    assert len(left_rows) == len(right_rows) == len(PUNCTUATION)
    assert all(premises[1] == conclusion for premises, conclusion in left_rows)
    assert all(premises[0] == conclusion for premises, conclusion in right_rows)
    assert all(not isinstance(premises[0], MVar) for premises, _ in left_rows)
    assert all(not isinstance(premises[1], MVar) for premises, _ in right_rows)


def test_coordination_phases_mark_and_merge_atomic_and_complex_categories() -> None:
    rules = _built(_declaration()).rule_set.rules
    mark = rules[5]
    merge = rules[6]

    assert isinstance(mark, NativeTermRule)
    assert isinstance(merge, NativeTermRule)
    assert tuple(mark.apply((CONJ, NP))) == (NP_CONJ,)
    assert tuple(mark.apply((CONJ, S_BACK_NP))) == (S_BACK_NP_CONJ,)
    assert tuple(merge.apply((NP, NP_CONJ))) == (NP,)
    assert tuple(merge.apply((S_BACK_NP, S_BACK_NP_CONJ))) == (S_BACK_NP,)
    assert tuple(mark.lowering.apply((CONJ, NP))) == (NP_CONJ,)
    assert tuple(merge.lowering.apply((NP, NP_CONJ))) == (NP,)
    assert tuple(merge.apply((N, NP_CONJ))) == ()
    assert tuple(mark.residuate(NP_CONJ, (HOLE, NP))) == (CONJ,)
    assert tuple(merge.residuate(NP, (NP, HOLE))) == (NP_CONJ,)
    assert tuple(
        (
            tuple(build(premise, STORE, Bindings()) for premise in premises),
            build(conclusion, STORE, Bindings()),
        )
        for premises, conclusion in merge.reachability_schemata
    ) == tuple(((plain, marked), plain) for plain, marked in CONJUNCTIONS.pairs)


def test_coordination_merge_survives_root_reachability() -> None:
    built = _built(_declaration()).rule_set
    merge = built.rules[6]
    rules = RuleSet(
        "coordination-merge",
        (merge,),
        built.matching,
        built.normal_form,
    )
    source = StaticLexicon((Entry("left", NP), Entry("right", NP_CONJ)))
    scorer = _UnitScorer()

    def parsed(reachability: Reachability) -> PackedForest[Term, float]:
        return parse(
            ("left", "right"),
            rules=rules,
            source=source,
            semiring=LogInside(),
            scorer=scorer,
            goal=RootGoal(TargetSet((NP,), built.matching)),
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, reachability),
        )

    unrestricted = parsed(Reachability.ALL)
    rooted = parsed(Reachability.ROOTS)

    assert rooted.categories_at((0, 2)) == unrestricted.categories_at((0, 2))
    assert rooted.categories_at((0, 2)) == ((NP, 0.0),)


def test_unlike_coordination_has_no_forward_enumeration_without_targets() -> None:
    unlike = _built(_declaration()).rule_set.rules[7]

    assert unlike.directions == frozenset({"check", "residual"})
    assert not unlike.forward_is_complete
    assert tuple(unlike.apply((CONJ, N))) == ()
    assert unlike.admits((CONJ, N), NP_CONJ)
    assert not unlike.admits((CONJ, NP), NP_CONJ)
    assert tuple(unlike.residuate(NP_CONJ, (HOLE, N))) == (CONJ,)


def test_finite_unlike_targets_enable_complete_forward_enumeration() -> None:
    unlike = _built(_declaration(unlike_targets=("NP",))).rule_set.rules[7]

    assert unlike.directions == frozenset({"forward", "check", "residual"})
    assert unlike.forward_is_complete
    assert isinstance(unlike, ProofSchema)
    assert unlike.proof_schemata == (
        (
            (CONJ_PATTERN.build(ASSEMBLY), Y.build(ASSEMBLY)),
            Node(NP_CONJ.op.name, payload=NP_CONJ.payload),
        ),
    )
    assert tuple(unlike.apply((CONJ, N))) == (NP_CONJ,)
    assert tuple(unlike.apply((CONJ, NP))) == ()


@pytest.mark.parametrize(
    ("rule_name", "unlike_targets"),
    [
        ("punctuation-left", None),
        ("punctuation-right", None),
        ("coordination-mark", None),
        ("coordination-merge", None),
        ("coordination-unlike", None),
        ("coordination-unlike", ("NP",)),
    ],
)
def test_assembly_specific_rules_satisfy_every_rule_law(
    rule_name: str,
    unlike_targets: tuple[str, ...] | None,
) -> None:
    rule_set = _built(_declaration(unlike_targets=unlike_targets)).rule_set
    rule = next(candidate for candidate in rule_set if candidate.name == rule_name)

    def rebuild(spec: Mapping[str, JsonValue]) -> Rule[Term]:
        """Rebuild the selected declared rule."""
        rebuilt = _built(_declaration(unlike_targets=unlike_targets)).rule_set
        selected = next(
            candidate for candidate in rebuilt if candidate.name == rule_name
        )

        if selected.to_spec() != spec:
            msg = f"rebuilt rule {rule_name!r} changed its declaration"
            raise ValueError(msg)

        return selected

    check_rule_laws(
        RuleFixture(
            rule,
            st.sampled_from(_assembly_law_terms(rule_name)),
            rebuild,
            max_examples=300,
        )
    )


def test_assembly_rejects_a_syntax_for_another_signature() -> None:
    other = Signature(
        "other",
        ("Cat",),
        (
            Operator("/", "Cat", arg_sorts=("Cat", "Cat")),
            Operator("\\", "Cat", arg_sorts=("Cat", "Cat")),
        ),
        vocabularies=(Vocabulary("Cat", ()),),
    )

    with pytest.raises(ValueError, match="syntax must use the store signature"):
        CCGAssembly(
            STORE,
            infix_syntax(other, ("/", "/"), ("\\", "\\")),
            partial(SYNTAX.parse, STORE),
            _views(FUNCTORS),
            FeatureErasure(),
            CONJUNCTIONS,
            NoNormalForm[Term](),
            category_sort="Cat",
        )


def test_pattern_factory_retains_no_boundary_declaration() -> None:
    declaration = PatternRuleSpec(
        name="n-to-np",
        premises=(N_PATTERN,),
        conclusion=NP_PATTERN,
    )
    rule = _one(declaration)

    assert isinstance(rule, PatternRule)
    assert rule.premises == (N_PATTERN.build(ASSEMBLY),)
    assert rule.conclusion == NP_PATTERN.build(ASSEMBLY)


def test_production_assembly_imports_no_specification_package() -> None:
    source = Path("lofi/grammars/ccg/_assembly.py").read_text(encoding="utf-8")
    imported = {
        alias.name
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or ""
        for node in ast.walk(ast.parse(source))
        if isinstance(node, ast.ImportFrom)
    }

    assert all(not name.startswith("lofi.spec") for name in imported)


def test_a_composition_declaration_without_an_order_builds_the_schema() -> None:
    rule = _one(
        CompositionSpec(
            direction="kuhlmann",
            order=None,
            view="/",
        )
    )

    assert isinstance(rule, GeneralizedComposition)
    assert rule.direction == "kuhlmann"
    assert rule.order is None
    assert rule.name == "composition:/:any:harmonic:kuhlmann"
    assert rule.to_spec()["order"] is None


ARROW_SIGNATURE: Final = Signature(
    "arrow",
    ("Cat", "Meaning"),
    (
        Operator("m", "Meaning"),
        Operator("->", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("pair", "Meaning", arg_sorts=("Meaning", "Meaning")),
        Operator("un", "Cat", arg_sorts=("Cat",)),
        Operator("tag", "Cat", arg_sorts=("Cat", "Cat"), family="tag"),
    ),
    vocabularies=(Vocabulary("Cat", ("S", "NP")),),
)
ARROW_STORE: Final = TermStore(ARROW_SIGNATURE)


def test_an_assembly_refuses_functors_of_another_sort_and_invalid_selections() -> None:
    syntax = SExpSyntax(ARROW_SIGNATURE)
    matching = FeatureErasure()
    normal_form = NoNormalForm[Term]()
    arrow = FunctorView(ARROW_STORE, "->", "->", result=0, argument=1)
    meaning = Signature(
        "meaning-first",
        ("Meaning", "Cat"),
        (Operator("->", "Meaning", arg_sorts=("Meaning", "Meaning")),),
        vocabularies=(Vocabulary("Cat", ()),),
    )
    meaning_store = TermStore(meaning)

    with pytest.raises(ValueError, match="not the category sort"):
        CCGAssembly(
            meaning_store,
            SExpSyntax(meaning),
            partial(SExpSyntax(meaning).parse, meaning_store),
            _views(FunctorView(meaning_store, "->", "->", result=0, argument=1)),
            matching,
            None,
            normal_form,
            category_sort="Cat",
        )

    with pytest.raises(ValueError, match="which it does not contain"):
        CCGAssembly(
            ARROW_STORE,
            syntax,
            partial(syntax.parse, ARROW_STORE),
            {"=>": (arrow, "pair")},
            matching,
            None,
            normal_form,
            category_sort="Cat",
        )

    spelled = CCGAssembly(
        ARROW_STORE,
        syntax,
        partial(syntax.parse, ARROW_STORE),
        {"=>": (arrow, "->")},
        matching,
        None,
        normal_form,
        category_sort="Cat",
    )
    assert spelled.functors["=>"] == (arrow, "->")

    built = _built(
        RuleSetSpec(name="undirected", rules=(ApplicationSpec(view="=>"),)),
        SignatureSpec(
            name="arrow",
            primitives=("S", "NP"),
            constructors=(
                ConstructorSpec(
                    name="arrow",
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling="->", precedence=10, associativity="left"
                    ),
                ),
            ),
            functors=(
                FunctorSpec(
                    name="=>",
                    constructor="arrow",
                    argument_side="either",
                    opposite="=>",
                ),
            ),
        ),
    )
    (rule,) = built.rule_set

    assert rule.name == "application:arrow"
    assert built.head_selections["application:arrow"].slot is None


def _universe() -> tuple[Term, ...]:
    """Return categories whose result spines reach depth two over both slashes."""
    atoms = (NP, S_DCL)
    slashes = ("/", "\\")
    simple = tuple(
        FUNCTORS.make(slash, result, argument)
        for slash in slashes
        for result in atoms
        for argument in atoms
    )
    deeper = tuple(
        FUNCTORS.make(slash, inner, argument)
        for slash in slashes
        for inner in simple
        for argument in atoms
    )
    wrapped = tuple(
        FUNCTORS.make(slash, result, inner)
        for slash in slashes
        for result in atoms
        for inner in simple
    )

    return (*atoms, *simple, *deeper, *wrapped)


UNIVERSE: Final = _universe()
POOL: Final[tuple[RuleSpec, ...]] = (
    ApplicationSpec(view="/"),
    ApplicationSpec(view="\\"),
    *(
        CompositionSpec(direction=direction, order=order, view=slash, crossed=crossed)
        for direction in ("unrestricted", "steedman", "kuhlmann")
        for order in (1, 2, None)
        for slash in ("/", "\\")
        for crossed in (False, True)
    ),
    *(
        SubstitutionSpec(order=1, view=slash, crossed=crossed)
        for slash in ("/", "\\")
        for crossed in (False, True)
    ),
)


def _conclusions(declaration: RuleSpec) -> tuple[frozenset[Term], ...]:
    """Return a rule's conclusions on every premise pair of the universe."""
    rule = _one(declaration)

    return tuple(
        frozenset(rule.apply((left, right))) for left in UNIVERSE for right in UNIVERSE
    )


CONCLUSIONS: Final = tuple(_conclusions(declaration) for declaration in POOL)


def test_harmonic_and_crossed_composition_coincide_only_without_a_restriction() -> None:
    def declared(direction: CompositionRestriction) -> RuleSetSpec:
        return RuleSetSpec(
            name="composition",
            rules=(
                CompositionSpec(direction=direction, order=2, view="/"),
                CompositionSpec(direction=direction, order=2, view="/", crossed=True),
            ),
        )

    with pytest.raises(ValueError, match="same relation twice") as refused:
        _built(declared("unrestricted"))

    assert "composition:/:2:harmonic:unrestricted" in str(refused.value)
    assert "composition:/:2:crossed:unrestricted" in str(refused.value)
    assert len(_built(declared("steedman")).rule_set) == 2
    assert len(_built(declared("kuhlmann")).rule_set) == 2


def test_first_order_steedman_and_kuhlmann_composition_coincide() -> None:
    declaration = RuleSetSpec(
        name="composition",
        rules=(
            CompositionSpec(direction="steedman", order=1, view="\\"),
            CompositionSpec(direction="kuhlmann", order=1, view="\\"),
        ),
    )

    with pytest.raises(ValueError, match="differ only in 'direction'"):
        _built(declaration)


ABSORPTION: Final = AbsorptionSpec(
    name="absorb", premises=(X, Y), conclusion=Y, absorbing=PUNCTUATION, side="left"
)
MARKING: Final = CoordinationSpec(
    name="mark",
    premises=(CONJ_PATTERN, X),
    conclusion=X,
    phase="mark",
    core_relation="same",
    head_policy="non-absorbing",
)


@pytest.mark.parametrize(
    ("first", "second"),
    [
        (ABSORPTION, ABSORPTION.with_(name="other", side="right")),
        (ABSORPTION, ABSORPTION.with_(name="other", absorbing=PUNCTUATION[:1])),
        (
            MARKING,
            MARKING.with_(name="other", phase="merge", head_policy="left-conjunct"),
        ),
        (_clausal_adjunct(), _clausal_adjunct().with_(name="other", target_view="/")),
    ],
)
def test_rules_differing_in_one_declared_parameter_do_not_coincide(
    first: RuleSpec, second: RuleSpec
) -> None:
    def declared(*rules: RuleSpec) -> RuleSetSpec:
        return RuleSetSpec(name="pair", rules=rules)

    with pytest.raises(ValueError, match="and 'other' differ only in 'name'"):
        _built(declared(first, first.with_(name="other")))

    assert len(_built(declared(first, second)).rule_set) == 2


def test_pattern_rules_differing_only_in_name_coincide() -> None:
    declaration = RuleSetSpec(
        name="patterns",
        rules=(
            PatternRuleSpec(name="one", premises=(N_PATTERN,), conclusion=NP_PATTERN),
            PatternRuleSpec(name="two", premises=(N_PATTERN,), conclusion=NP_PATTERN),
        ),
    )

    with pytest.raises(ValueError, match="'one' and 'two' differ only in 'name'"):
        _built(declaration)

    exact = declaration.rules[1].with_(matching=IdentityMatchingSpec())

    assert (
        len(_built(declaration.with_(rules=(declaration.rules[0], exact))).rule_set)
        == 2
    )


@settings(max_examples=200, deadline=None, database=None)
@given(st.lists(st.integers(0, len(POOL) - 1), min_size=2, max_size=8, unique=True))
def test_no_accepted_rule_set_holds_two_rules_agreeing_on_every_premise_pair(
    chosen: list[int],
) -> None:
    declaration = RuleSetSpec(name="generated", rules=tuple(POOL[at] for at in chosen))
    agreeing = any(
        CONCLUSIONS[left] == CONCLUSIONS[right]
        for position, left in enumerate(chosen)
        for right in chosen[position + 1 :]
    )

    if agreeing:
        with pytest.raises(ValueError, match="declares the same relation twice"):
            _built(declaration)
    else:
        assert len(_built(declaration).rule_set) == len(chosen)


def test_a_pair_is_refused_exactly_when_it_agrees_on_every_premise_pair() -> None:
    refused = 0

    for position, left in enumerate(POOL):
        for offset, right in enumerate(POOL[position + 1 :], start=position + 1):
            declaration = RuleSetSpec(name="pair", rules=(left, right))
            agreeing = CONCLUSIONS[position] == CONCLUSIONS[offset]

            if agreeing:
                refused += 1

                with pytest.raises(ValueError, match="same relation twice"):
                    _built(declaration)
            else:
                assert len(_built(declaration).rule_set) == 2

    assert refused > 0
    assert refused == len(POOL) - len(set(CONCLUSIONS))


def test_a_rule_declaring_the_grammars_matching_shares_the_grammars_congruence() -> (
    None
):
    declared = CoercionSpec(
        label="lexical", member=0, source="S[dcl]\\NP", target="NP\\NP"
    ).with_(matching=FeatureErasureMatchingSpec())
    rule_set = _built(
        RuleSetSpec(name="declared", rules=(ApplicationSpec(view="/"), declared))
    ).rule_set
    application, coercion = rule_set

    assert coercion.matching is application.matching
    assert rule_set.packing is rule_set.matching
    assert not _one(declared).admits((S_BACK_NP_CONJ,), NP_BACK_NP)


@pytest.mark.parametrize(
    ("declared", "relation"),
    [(IdentityMatchingSpec(), Identity), (AlphaMatchingSpec(), Identity)],
)
def test_a_congruence_that_keeps_marks_apart_is_not_refined(
    declared: IdentityMatchingSpec | AlphaMatchingSpec, relation: type
) -> None:
    built = declared.build()

    if not isinstance(built, TermMatching):
        msg = f"test grammar matching {built.name!r} has no native representatives"
        raise TypeError(msg)

    assembly = CCGAssembly(
        STORE,
        SYNTAX,
        partial(SYNTAX.parse, STORE),
        _views(FUNCTORS),
        built,
        CONJUNCTIONS,
        NoNormalForm[Term](),
        category_sort="Cat",
    )
    rule = _one(ApplicationSpec(view="/", matching=declared))

    assert assembly.matching is built
    assert isinstance(rule.matching, relation)
    assert rule.matching.name == built.name


UNARY_UNIVERSE: Final = (
    SYNTAX.parse(STORE, "S"),
    S_DCL,
    S_NG,
    S_CONJ,
    S_BACK_NP,
    S_DCL_BACK_NP,
    S_NG_BACK_NP,
    S_BACK_NP_CONJ,
    NP,
    NP_CONJ,
    N,
)
UNARY_POOL: Final[tuple[RuleSpec, ...]] = tuple(
    CoercionSpec(label="pool", member=0, source=source, target=target).with_(
        matching=matching
    )
    for source in ("S[dcl]\\NP", "S[ng]\\NP", "S\\NP", "S[dcl]", "S[ng]", "S")
    for target in ("NP\\NP", "NP")
    for matching in (None, IdentityMatchingSpec(), FeatureErasureMatchingSpec())
)


def _unary_conclusions(declaration: RuleSpec) -> tuple[frozenset[Term], ...]:
    """Return a unary rule's conclusions on every premise of the universe."""
    rule = _one(declaration)

    return tuple(frozenset(rule.apply((premise,))) for premise in UNARY_UNIVERSE)


UNARY_CONCLUSIONS: Final = tuple(
    _unary_conclusions(declaration) for declaration in UNARY_POOL
)


def test_a_unary_pair_is_refused_exactly_when_it_agrees_on_every_premise() -> None:
    refused = 0

    for position, left in enumerate(UNARY_POOL):
        for offset, right in enumerate(UNARY_POOL[position + 1 :], start=position + 1):
            declaration = RuleSetSpec(
                name="pair",
                rules=(left, right.with_(member=1)),
            )

            if UNARY_CONCLUSIONS[position] == UNARY_CONCLUSIONS[offset]:
                refused += 1

                with pytest.raises(ValueError, match="same relation twice"):
                    _built(declaration)
            else:
                assert len(_built(declaration).rule_set) == 2

    assert refused > 0


def test_coercions_naming_members_of_one_class_are_refused_only_when_erased() -> None:
    declaration = RuleSetSpec(
        name="coercions",
        rules=(
            CoercionSpec(label="c", member=0, source="S[dcl]\\NP", target="NP\\NP"),
            CoercionSpec(label="c", member=1, source="S[ng]\\NP", target="NP\\NP"),
        ),
    )

    with pytest.raises(ValueError, match="same relation twice"):
        _built(declaration)

    exact = declaration.with_(
        rules=tuple(
            rule.with_(matching=IdentityMatchingSpec()) for rule in declaration.rules
        )
    )

    assert len(_built(exact).rule_set) == 2


def test_a_fixed_premise_probe_admits_every_root_its_matching_identifies() -> None:
    signature = cast(
        "Signature",
        SimpleNamespace(
            operators=tuple(
                SimpleNamespace(name=name, erases_to=erases_to)
                for name, erases_to in (("f", None), ("f[x]", "f"), ("g", None))
            )
        ),
    )
    premises = (Node("f[x]", ()), MVar("x"))

    erased = _pattern_probe(premises, FeatureErasure(), signature)
    exact = _pattern_probe(premises, Identity[Term](), signature)

    assert erased is not None
    assert erased.alternatives == ((frozenset({"f", "f[x]"}), None),)
    assert exact is not None
    assert exact.alternatives == ((frozenset({"f[x]"}), None),)
    assert _pattern_probe(premises, cast("Matching[Term]", object()), signature) is None


def test_a_conjunction_marking_constructor_maps_one_category_to_another() -> None:
    grammar = bisk_hockenmaier_grammar()
    signature = grammar.signature
    conjunctions = signature.conjunctions

    if conjunctions is None:
        pytest.fail("the packaged baseline grammar declares no conjunctions")

    binary = grammar.with_(
        signature=signature.with_(conjunctions=conjunctions.with_(marking="/"))
    )

    with pytest.raises(ValueError, match="does not map one 'Cat' category"):
        GrammarIso().forward(binary, registered_language("propositional"))
