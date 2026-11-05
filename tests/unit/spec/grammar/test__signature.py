import json
from functools import partial
from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.declarations import SMALL, Spine, atom_text, category_text, signatures
from tests.support.rules import declared_rules

from lofi.grammars.ccg import CCGAssembly, FunctorView
from lofi.judge import NoNormalForm, PatternRule, RuleSet
from lofi.spec import (
    ApplicationSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PrefixConstructorNotationSpec,
    RuleRestrictionSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec.grammar import (
    BindingSiteSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    FunctorSpec,
    IdentityMatchingSpec,
    MatchingSpec,
    NoNormalFormSpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import LanguageIso, registered_language
from lofi.terms import (
    BindingSite,
    ConstructorSyntax,
    FeatureErasure,
    Identity,
    Operator,
    SExpSyntax,
    Signature,
    Term,
    TermMatching,
    TermStore,
    TextParseError,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import JsonValue
    from lofi.judge import Rule
    from lofi.spec.grammar import AdoptedRules
    from lofi.terms import TargetSet, TermSyntax

X: Final = TermPatternSpec(name="X", sort="Cat")
Y: Final = TermPatternSpec(name="Y", sort="Cat")
ARROW: Final = (
    ConstructorSpec(
        name="arrow",
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling="->", precedence=10, associativity="left"
        ),
    ),
)
ARROW_VIEWS: Final = (
    FunctorSpec(
        name="arrow",
        constructor="arrow",
        argument_side="either",
        opposite="arrow",
    ),
)


def slashes() -> tuple[ConstructorSpec, ...]:
    """Return the directed slash constructor pair."""
    return (_constructor("/"), _constructor("\\"))


def slash_views() -> tuple[FunctorSpec, ...]:
    """Return the directed views over the packaged slash pair."""
    return (
        FunctorSpec(name="/", constructor="/", argument_side="right", opposite="\\"),
        FunctorSpec(name="\\", constructor="\\", argument_side="left", opposite="/"),
    )


def _constructor(
    name: str, *, arity: int = 2, spelling: str | None = None
) -> ConstructorSpec:
    return ConstructorSpec(
        name=name,
        kind=("Cat",) * (arity + 1),
        notation=InfixConstructorNotationSpec(
            spelling=name if spelling is None else spelling,
            precedence=10,
            associativity="left",
        ),
    )


def _node(
    name: str, *arguments: TermPatternSpec | NodePatternSpec | ConstantPatternSpec
) -> NodePatternSpec | ConstantPatternSpec:
    if not arguments:
        return ConstantPatternSpec(name=name)

    return NodePatternSpec(operator=FixedOperatorSpec(name=name), arguments=arguments)


def _explicit() -> SignatureSpec:
    return SignatureSpec(
        name="cfg",
        primitives=("S", "NP", "VP"),
        constructors=(_constructor("pair"),),
        syntax="s-expression",
    )


def _ccg() -> SignatureSpec:
    return SignatureSpec(
        name="ccg",
        primitives=("S", "NP", "conj"),
        operators=(
            OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
            OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
        ),
        constructors=slashes(),
        functors=slash_views(),
        conjunctions=ConjunctionsSpec(
            markers=("conj",),
            pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
        ),
    )


def _store(declared: SignatureSpec, /) -> TermStore:
    language = registered_language("propositional")
    bounds = dict(language.bounds)
    bounds[declared.category_sort] = bounds.pop(language.category_sort)
    language = language.with_(category_sort=declared.category_sort, bounds=bounds)

    return LanguageIso().extend(language, declared.extend).store


def _signature(declared: SignatureSpec, /) -> Signature:
    return _store(declared).signature


def _term_matching(declared: MatchingSpec, /) -> TermMatching:
    """Return the native representative relation a grammar runtime requires."""
    built = declared.build()

    if not isinstance(built, TermMatching):
        msg = f"test grammar matching {built.name!r} has no native representatives"
        raise TypeError(msg)

    return built


def _rules() -> RuleSetSpec:
    return RuleSetSpec(
        name="cfg-rules",
        rules=(
            PatternRuleSpec(
                name="s-from-np-vp",
                premises=(_node("NP"), _node("VP")),
                conclusion=_node("S"),
                head_policy="non-absorbing",
            ),
        ),
    )


def _runtime(
    declared: SignatureSpec,
    *,
    categories: Sequence[str] = (),
) -> tuple[TermStore, TermSyntax, CCGAssembly]:
    """Return a declaration's store, syntax and rule constructor."""
    store = _store(declared)
    syntax, context = _build_runtime(
        declared,
        store,
        _term_matching(FeatureErasureMatchingSpec()),
        categories=categories,
    )

    assert type(context) is CCGAssembly

    return store, syntax, context


def _build_runtime(
    declared: SignatureSpec,
    store: TermStore,
    matching: TermMatching,
    *,
    categories: Sequence[str] = (),
) -> tuple[TermSyntax, CCGAssembly]:
    """Build a signature runtime with one required category resolver."""
    prepared = declared.prepare_runtime()
    syntax = prepared.syntax(store)
    context = prepared.build(
        store,
        syntax,
        partial(syntax.parse, store),
        matching,
        NoNormalFormSpec().build(),
        categories=categories,
    )
    return syntax, context


def _compiled(
    declared: SignatureSpec,
    rules: RuleSetSpec,
    /,
    *,
    matching: MatchingSpec | None = None,
) -> AdoptedRules:
    """Compile a rule inventory over a signature through the native compiler."""
    return declared_rules(
        declared,
        rules,
        matching=FeatureErasureMatchingSpec() if matching is None else matching,
    )


def _application(*slashes: str) -> RuleSetSpec:
    return RuleSetSpec(
        name="application",
        rules=tuple(ApplicationSpec(view=view) for view in slashes or ("/", "\\")),
    )


def _explicit_runtime() -> tuple[
    TermStore,
    TermSyntax,
    RuleSet[Term],
    TargetSet[Term],
]:
    """Return the store, syntax, rules and goal of the s-expression toy grammar."""
    adopted = _compiled(_explicit(), _rules(), matching=IdentityMatchingSpec())
    store = adopted.context.store
    syntax = adopted.context.syntax
    rules = adopted.rule_set
    goal = RootGoalSpec(categories=('("#const[Cat]" "S")',)).build(
        partial(syntax.parse, store), rules.matching
    )

    return store, syntax, rules, goal


def _primitives(count: int) -> tuple[str, ...]:
    return ("S", *(f"P{index}" for index in range(1, count)))


def _ccg_runtime(
    primitives: int,
) -> tuple[
    TermStore,
    TermSyntax,
    RuleSet[Term],
    TargetSet[Term],
]:
    """Return the store, syntax, rules and goal of a K-primitive CCG."""
    declared = SignatureSpec(
        primitives=_primitives(primitives),
        constructors=slashes(),
        functors=slash_views(),
    )
    adopted = _compiled(declared, _application())
    store = adopted.context.store
    syntax = adopted.context.syntax
    rules = adopted.rule_set
    goal = RootGoalSpec().build(partial(syntax.parse, store), rules.matching)

    return store, syntax, rules, goal


def _derives(
    rule: Rule[Term],
    store: TermStore,
    syntax: TermSyntax,
    *categories: str,
) -> bool:
    """Return whether a binary rule derives ``S`` from two categories."""
    left, right = (syntax.parse(store, category) for category in categories)

    return rule.admits((left, right), syntax.parse(store, "S"))


def test_an_operator_declaration_builds_the_operator_it_names() -> None:
    declared = OperatorSpec(
        name="lam",
        sort="LF",
        arg_sorts=("LF",),
        binding=(BindingSiteSpec(arguments=(0,), sort="LF"),),
        family="binder",
        erases_to=None,
    )

    assert declared.build() == Operator(
        "lam",
        "LF",
        arg_sorts=("LF",),
        binding=(BindingSite((0,), "LF"),),
        family="binder",
    )
    assert (
        OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S").build().erases_to == "S"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {"name": "", "sort": "Cat"},
        {"name": "f", "sort": ""},
        {"name": "f", "sort": "Cat", "arg_sorts": ["Cat", ""]},
        {"name": "f", "sort": "Cat", "family": ""},
        {"name": "f", "sort": "Cat", "arg_sorts": ["Cat"], "binds": [1, 0]},
        {"name": "f", "sort": "Cat", "arg_sorts": ["Cat"], "binds": [-1]},
        {"name": "f", "sort": "Cat", "erases_to": ""},
        {"name": "f", "sort": "Cat", "payload_sort": "tensor"},
        {"name": "f", "sort": "Cat", "arity": 2},
    ],
)
def test_an_operator_declaration_refuses_a_malformed_field(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        OperatorSpec.model_validate(payload)


def test_a_constructor_builds_an_operator_of_its_arity_over_the_category_sort() -> None:
    built = ConstructorSpec(
        name="triple",
        kind=("Cat", "Cat", "Cat", "Cat"),
        notation=PrefixConstructorNotationSpec(spelling="triple"),
    ).build()

    assert built == Operator(
        "triple", "Cat", arg_sorts=("Cat", "Cat", "Cat"), family="constructor"
    )


@pytest.mark.parametrize(
    "payload",
    [
        {},
        {
            "name": "",
            "kind": ["Cat", "Cat"],
            "notation": {"style": "prefix", "spelling": "f"},
        },
        {
            "name": "f",
            "kind": ["Cat"],
            "notation": {"style": "prefix", "spelling": "f"},
        },
        {
            "name": "f",
            "kind": ["Cat", ""],
            "notation": {"style": "prefix", "spelling": "f"},
        },
        {
            "name": "f",
            "kind": ["Cat", "Cat"],
            "notation": {"style": "prefix", "spelling": ""},
        },
        {
            "name": "f",
            "kind": ["Cat", "Cat"],
            "notation": {"style": "sideways", "spelling": "f"},
        },
    ],
)
def test_a_constructor_declaration_refuses_a_malformed_field(
    payload: dict[str, JsonValue],
) -> None:
    with pytest.raises(dx.ValidationError):
        ConstructorSpec.model_validate(payload)


def test_a_signature_builds_primitives_operators_and_constructors_in_code_order() -> (
    None
):
    signature = _signature(_ccg())

    assert signature.name == "propositional:ccg"
    assert signature.vocabulary("Cat").names == (
        "S",
        "NP",
        "conj",
        "S[dcl]",
        "NP[conj]",
    )
    assert signature.by_name("/").family == "constructor"
    assert signature.by_name("\\").arg_sorts == ("Cat", "Cat")
    assert signature.constant_core("Cat", "NP[conj]") == "NP"


def test_a_signature_reads_its_features_off_the_featured_operators() -> None:
    declared = SignatureSpec(
        name="featured",
        primitives=("S",),
        operators=(
            OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
            OperatorSpec(name="S[q]", sort="Cat", erases_to="S"),
            OperatorSpec(name="\\[conj]", sort="Cat", erases_to="\\"),
        ),
        constructors=slashes(),
        allows_vacuous_binding=False,
    )
    signature = _signature(declared)

    assert dict(signature.features) == {"\\": ("conj",)}
    assert signature.vocabulary("Cat").names == (
        "S",
        "S[dcl]",
        "S[q]",
    )
    assert signature.constant_core("Cat", "S[q]") == "S"
    assert signature.by_name("\\[conj]").arg_sorts == ("Cat", "Cat")
    assert signature.allows_vacuous_binding is False
    assert declared.category_sort == "Cat"


def test_a_signature_selects_colon_delimited_category_features() -> None:
    declared = SignatureSpec(
        name="colon-features",
        primitives=("s", "np"),
        operators=(OperatorSpec(name="s:dcl", sort="Cat", erases_to="s"),),
        constructors=slashes(),
        functors=slash_views(),
        feature_delimiter="colon",
    )
    store, syntax, context = _runtime(declared)

    category = syntax.parse(store, "s:dcl\\np")

    assert syntax.render(store, category) == "s:dcl\\np"
    assert context.matching.equivalent(category.args[0], syntax.parse(store, "s"))
    with pytest.raises(TextParseError):
        syntax.parse(store, "s[dcl]\\np")


def test_a_signature_extension_preserves_the_language_kernel_families() -> None:
    language = registered_language("propositional")
    declared = _ccg()
    lowerer = LanguageIso()
    plain = lowerer.extend(language, declared.extend).store.signature
    constant = lowerer.extend(
        language.with_(constant_family="lexical-constant"), declared.extend
    ).store.signature
    variable = lowerer.extend(
        language.with_(variable_family="logical-variable"), declared.extend
    ).store.signature
    custom = lowerer.extend(
        language.with_(
            constant_family="lexical-constant",
            variable_family="logical-variable",
        ),
        declared.extend,
    ).store.signature

    assert custom.constant_family == "lexical-constant"
    assert custom.variable_family == "logical-variable"
    assert custom.constant(language.type_sort).family == "lexical-constant"
    assert custom.constant(declared.category_sort).family == "lexical-constant"
    assert custom.free(language.type_sort).family == "logical-variable"
    assert custom.bound(language.term_sort).family == "logical-variable"
    assert len({plain.digest, constant.digest, variable.digest, custom.digest}) == 4


def test_the_category_sort_is_declared_rather_than_positional() -> None:
    base = registered_language("propositional")
    bounds = dict(base.bounds)
    bounds["Feature"] = bounds.pop(base.type_sort)
    language = base.with_(
        type_sort="Feature",
        constructors=tuple(
            constructor.with_(kind=("Feature",) * len(constructor.kind))
            for constructor in base.constructors
        ),
        bounds=bounds,
    )
    declared = SignatureSpec(
        name="featured-categories",
        sorts=("Feature", "Cat"),
        category_sort="Cat",
        primitives=("S", "NP"),
        constructors=slashes(),
        functors=slash_views(),
        syntax="declared",
    )

    lowered = LanguageIso().extend(language, declared.extend)
    signature = lowered.store.signature

    assert declared.category_sort == "Cat"
    assert signature.sorts == ("Tm", "Feature", "Cat")
    assert signature.by_name("/").sort == "Cat"
    assert signature.vocabulary("Cat").names == ("S", "NP")


def test_a_signature_round_trips_and_keeps_its_constructor_order() -> None:
    declared = _ccg()
    again = SignatureSpec.model_validate_json(declared.model_dump_json())

    assert again == declared
    assert [constructor.name for constructor in again.constructors] == ["/", "\\"]


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sorts", ["Cat", "Cat"], "sort 'Cat' is declared twice"),
        (
            "primitives",
            ["S", "NP", "S"],
            "category primitive 'S' is declared twice",
        ),
        (
            "operators",
            [
                {"name": "S[dcl]", "sort": "Cat", "erases_to": "S"},
                {"name": "S[dcl]", "sort": "Cat", "erases_to": "S"},
            ],
            "operator 'S[dcl]' is declared twice",
        ),
        (
            "vocabularies",
            {"Cat": ["N", "N"]},
            "vocabulary 'Cat' declares 'N' twice",
        ),
        (
            "vocabularies",
            {"Other": ["x", "x"]},
            "vocabulary 'Other' declares 'x' twice",
        ),
        (
            "vocabularies",
            {"Cat": ["S", ""]},
            "vocabulary 'Cat' declares a constant with an empty name",
        ),
    ],
)
def test_a_signature_locates_a_repeated_declaration(
    field: str, value: JsonValue, message: str
) -> None:
    payload = json.loads(_ccg().model_dump_json()) | {field: value}

    with pytest.raises(dx.ValidationError) as refused:
        SignatureSpec.model_validate(payload)

    assert [(entry.loc, entry.msg) for entry in refused.value.entries] == [
        ((field,), message)
    ]


@pytest.mark.parametrize(
    ("changes", "message"),
    [
        (
            {"vocabularies": {"Cat": ["S"]}},
            (
                "category name 'S' is declared by both primitives[0] "
                "and vocabularies['Cat'][0]"
            ),
        ),
        (
            {"primitives": ["S", "NP", "conj", "S[dcl]"]},
            (
                "category name 'S[dcl]' is declared by both primitives[3] "
                "and operators[0]"
            ),
        ),
        (
            {"vocabularies": {"Cat": ["S[dcl]"]}},
            (
                "category name 'S[dcl]' is declared by both operators[0] "
                "and vocabularies['Cat'][0]"
            ),
        ),
    ],
)
def test_a_signature_locates_a_category_name_with_two_sources(
    changes: dict[str, JsonValue], message: str
) -> None:
    payload = json.loads(_ccg().model_dump_json()) | changes

    with pytest.raises(dx.ValidationError) as refused:
        SignatureSpec.model_validate(payload)

    assert [(entry.loc, entry.msg) for entry in refused.value.entries] == [
        ((), message)
    ]


def test_the_packaged_slash_pair_has_declared_infix_notation() -> None:
    assert [
        (constructor.name, constructor.notation.spelling) for constructor in slashes()
    ] == [("/", "/"), ("\\", "\\")]


@pytest.mark.parametrize(
    ("operators", "message"),
    [
        ([{"name": "N", "sort": "Cat"}], "belongs among the primitives"),
        (
            [{"name": "pair", "sort": "Cat", "arg_sorts": ["Cat", "Cat"]}],
            "belongs among the constructors",
        ),
        ([{"name": "NPconj", "sort": "Cat", "erases_to": "NP"}], "bracketed feature"),
        ([{"name": "NP[]", "sort": "Cat", "erases_to": "NP"}], "bracketed feature"),
        ([{"name": "NP[x]", "sort": "Other", "erases_to": "NP"}], "another sort"),
        ([{"name": "VP[x]", "sort": "Cat", "erases_to": "VP"}], "does not declare"),
        (
            [
                {"name": "NP[x]", "sort": "Cat", "erases_to": "NP"},
                {"name": "NP[x][y]", "sort": "Cat", "erases_to": "NP[x]"},
            ],
            "does not declare",
        ),
        (
            [{"name": "NP[x]", "sort": "Cat", "erases_to": "NP", "family": "f"}],
            "takes its shape",
        ),
        (
            [
                {
                    "name": "/[x]",
                    "sort": "Cat",
                    "arg_sorts": ["Cat", "Cat"],
                    "erases_to": "/",
                }
            ],
            "takes its shape",
        ),
    ],
)
def test_a_signature_refuses_an_operator_with_another_declaration(
    operators: list[JsonValue], message: str
) -> None:
    payload = json.loads(_ccg().model_dump_json()) | {
        "sorts": ["Cat", "Other"],
        "operators": operators,
    }

    with pytest.raises(dx.ValidationError, match=message):
        SignatureSpec.model_validate(payload)


def test_a_grammar_signature_refuses_a_noncategory_sort() -> None:
    declared = SignatureSpec(
        sorts=("Cat", "LF"),
        primitives=("S",),
        operators=(
            OperatorSpec(
                name="lam",
                sort="LF",
                arg_sorts=("LF",),
                binding=(BindingSiteSpec(arguments=(0,), sort="LF"),),
            ),
            OperatorSpec(
                name="bind",
                sort="Cat",
                arg_sorts=("Cat",),
                binding=(BindingSiteSpec(arguments=(0,), sort="Cat"),),
            ),
        ),
        syntax="s-expression",
    )

    with pytest.raises(ValueError, match=r"sorts outside the language.*LF"):
        _signature(declared)


@pytest.mark.parametrize(
    "views",
    [
        (FunctorSpec(name="f", constructor="/", argument_side="right", opposite="b"),),
        (
            FunctorSpec(name="f", constructor="/", argument_side="right", opposite="b"),
            FunctorSpec(
                name="b", constructor="\\", argument_side="right", opposite="f"
            ),
        ),
    ],
)
def test_a_signature_refuses_functor_roles_that_describe_no_view(
    views: tuple[FunctorSpec, ...],
) -> None:
    with pytest.raises(dx.ValidationError):
        SignatureSpec(primitives=("S",), constructors=slashes(), functors=views)


def test_a_signature_refuses_functors_that_disagree_on_their_slots() -> None:
    views = (
        FunctorSpec(name="/", constructor="/", argument_side="right", opposite="\\"),
        FunctorSpec(
            name="\\",
            constructor="\\",
            argument_side="left",
            opposite="/",
            result=1,
            argument=0,
        ),
    )

    with pytest.raises(dx.ValidationError, match="disagree on slots"):
        SignatureSpec(primitives=("S",), constructors=slashes(), functors=views)


@pytest.mark.parametrize(
    "payload",
    [
        {"name": ""},
        {"sorts": []},
        {"sorts": ["Cat", ""]},
        {"primitives": ["S", ""]},
        {"constructors": []},
        {"syntax": "prefix"},
        {"kind": "ccg"},
        {"features": {"S": ["dcl"]}},
        {"conjunctions": {"markers": [], "pairs": [{"unmarked": "NP", "marked": "N"}]}},
        {"conjunctions": {"markers": ["conj"], "pairs": []}},
        {
            "conjunctions": {
                "markers": ["conj"],
                "pairs": [{"unmarked": "", "marked": "N"}],
            }
        },
    ],
)
def test_a_signature_refuses_a_malformed_field(
    payload: dict[str, JsonValue],
) -> None:
    complete = json.loads(_ccg().model_dump_json()) | payload

    with pytest.raises(dx.ValidationError):
        SignatureSpec.model_validate(complete)


def test_an_infix_signature_shares_one_syntax_between_the_assembly_and_the_caller() -> (
    None
):
    store, syntax, context = _runtime(_ccg(), categories=("(S[dcl]\\NP)/NP",))

    assert type(syntax) is ConstructorSyntax
    assert context.syntax is syntax
    assert context.store is store
    assert context.category("(S[dcl]\\NP)/NP") is syntax.parse(store, "(S[dcl]\\NP)/NP")

    functors, slash = context.functors["/"]

    assert type(functors) is FunctorView
    assert slash == "/"
    assert (functors.forward_slash, functors.backward_slash) == ("/", "\\")

    conjunctions = context.conjunctions

    assert conjunctions is not None
    assert conjunctions.markers == (syntax.parse(store, "conj"),)
    assert conjunctions.pairs == (
        (syntax.parse(store, "NP"), syntax.parse(store, "NP[conj]")),
    )
    assert (
        context.matching.equivalent(
            syntax.parse(store, "NP[conj]"), syntax.parse(store, "NP")
        )
        is False
    )

    adopted = _compiled(_ccg(), _application("/"))
    (rule,) = adopted.rule_set

    assert rule.matching is adopted.context.matching
    assert rule.name == "application:/"


def test_a_signature_without_conjunctions_runs_under_its_declared_matching() -> None:
    declared = SignatureSpec(
        primitives=("S", "NP"), constructors=slashes(), functors=slash_views()
    )
    store, syntax, context = _runtime(declared)

    assert context.conjunctions is None
    assert type(context.matching) is FeatureErasure
    assert len(_compiled(declared, _application("/", "\\")).rule_set.rules) == 2
    assert syntax.render(store, syntax.parse(store, "S\\NP")) == "S\\NP"


def test_the_conjunction_markers_are_the_reserved_categories() -> None:
    store, syntax, context = _runtime(_ccg())
    conjunctions = context.conjunctions

    assert conjunctions is not None
    assert frozenset(conjunctions.markers) == frozenset({syntax.parse(store, "conj")})


def test_an_undirected_arrow_composes_with_its_argument_on_either_side() -> None:
    declared = SignatureSpec(
        primitives=("S", "NP"), constructors=ARROW, functors=ARROW_VIEWS
    )
    adopted = _compiled(declared, _application("arrow"))
    store = adopted.context.store
    syntax = adopted.context.syntax
    functors, slash = adopted.context.functors["arrow"]

    assert slash == "arrow"
    assert functors.forward_slash == functors.backward_slash == "arrow"

    (rule,) = adopted.rule_set

    assert rule.name == "application:arrow"
    assert _derives(rule, store, syntax, "S->NP", "NP")
    assert _derives(rule, store, syntax, "NP", "S->NP")


def test_switching_the_constructors_alone_flips_directed_to_undirected() -> None:
    directed = SignatureSpec(
        primitives=("S", "NP"), constructors=slashes(), functors=slash_views()
    )
    undirected = directed.with_(
        constructors=(_constructor("/"),),
        functors=(
            FunctorSpec(
                name="/", constructor="/", argument_side="either", opposite="/"
            ),
        ),
    )
    rules = _application("/")
    derived: dict[str, tuple[bool, bool]] = {}

    for label, declared in (("directed", directed), ("undirected", undirected)):
        adopted = _compiled(declared, rules)
        store = adopted.context.store
        syntax = adopted.context.syntax
        (rule,) = adopted.rule_set
        derived[label] = (
            _derives(rule, store, syntax, "S/NP", "NP"),
            _derives(rule, store, syntax, "NP", "S/NP"),
        )

    assert derived == {"directed": (True, False), "undirected": (True, True)}


def test_a_functor_at_other_slots_puts_its_result_second() -> None:
    declared = SignatureSpec(
        primitives=("S", "NP"),
        constructors=(_constructor("|"),),
        functors=(
            FunctorSpec(
                name="|",
                constructor="|",
                argument_side="either",
                opposite="|",
                result=1,
                argument=0,
            ),
        ),
    )
    adopted = _compiled(declared, _application("|"))
    store = adopted.context.store
    syntax = adopted.context.syntax
    (rule,) = adopted.rule_set

    assert _derives(rule, store, syntax, "NP|S", "NP")
    assert not _derives(rule, store, syntax, "S|NP", "NP")


def test_a_rule_names_a_functor_view_independently_of_surface_spelling() -> None:
    declared = SignatureSpec(
        primitives=("S", "NP"),
        constructors=(
            _constructor("fwd", spelling="/"),
            _constructor("bwd", spelling="\\"),
        ),
        functors=(
            FunctorSpec(
                name="right", constructor="fwd", argument_side="right", opposite="left"
            ),
            FunctorSpec(
                name="left", constructor="bwd", argument_side="left", opposite="right"
            ),
        ),
    )
    adopted = _compiled(declared, _application("right", "left"))
    store = adopted.context.store
    syntax = adopted.context.syntax
    right, left = adopted.rule_set

    assert declared.functor_names == frozenset({"right", "left"})
    assert syntax.parse(store, "S/NP").op.name == "fwd"
    assert _derives(right, store, syntax, "S/NP", "NP")
    assert _derives(left, store, syntax, "NP", "S\\NP")


def test_the_functor_slots_are_read_from_the_declared_functor_roles() -> None:
    directed = SignatureSpec(
        primitives=("S", "NP"),
        constructors=(
            _constructor("over"),
            _constructor("under"),
            _constructor("pair"),
        ),
        functors=(
            FunctorSpec(
                name="over",
                constructor="over",
                argument_side="right",
                opposite="under",
                result=1,
                argument=0,
            ),
            FunctorSpec(
                name="under",
                constructor="under",
                argument_side="left",
                opposite="over",
                result=1,
                argument=0,
            ),
        ),
        syntax="s-expression",
    )
    undirected = SignatureSpec(
        primitives=("S", "NP"),
        constructors=ARROW,
        functors=ARROW_VIEWS,
    )

    assert directed.functor_slots == {"over": (1, 0), "under": (1, 0)}
    assert undirected.functor_slots == {"arrow": (0, 1)}
    assert _explicit().functor_slots == {}


def test_a_spelled_constructor_without_functor_roles_is_read_but_not_a_functor() -> (
    None
):
    declared = SignatureSpec(
        primitives=("S", "NP"),
        constructors=(*slashes(), _constructor("pair", spelling="*")),
        functors=slash_views(),
    )
    store, syntax, _ = _runtime(declared)

    assert syntax.parse(store, "S*NP").op.name == "pair"
    assert declared.functor_names == frozenset({"/", "\\"})

    with pytest.raises(dx.ValidationError, match="names '\\*', which is no functor"):
        _compiled(declared, _application("*"))


def test_a_functor_family_rule_is_refused_without_a_functor_view() -> None:
    _, _, context = _runtime(_explicit())

    assert context.functors == {}

    with pytest.raises(dx.ValidationError, match="names 'pair', which is no functor"):
        _compiled(_explicit(), _application("pair"))


def test_an_s_expression_signature_reads_s_expressions_and_builds_pattern_rules() -> (
    None
):
    declared = _explicit()
    store = _store(declared)
    signature = store.signature
    syntax, context = _build_runtime(
        declared,
        store,
        _term_matching(IdentityMatchingSpec()),
        categories=('("pair" ("#const[Cat]" "NP") ("#const[Cat]" "VP"))',),
    )

    assert type(syntax) is SExpSyntax
    assert syntax.signature is signature
    assert type(context) is CCGAssembly
    np = syntax.parse(store, '("#const[Cat]" "NP")')
    vp = syntax.parse(store, '("#const[Cat]" "VP")')

    assert context.category(
        '("pair" ("#const[Cat]" "NP") ("#const[Cat]" "VP"))'
    ) is store.op(
        signature.by_name("pair"),
        np,
        vp,
    )

    built = _compiled(declared, _rules(), matching=IdentityMatchingSpec()).built
    rule_set = built.rule_set

    assert type(rule_set) is RuleSet
    assert rule_set.name == "cfg-rules"
    assert [rule.name for rule in rule_set] == ["s-from-np-vp"]
    assert built.head_selections["s-from-np-vp"].slot == 1

    (rule,) = rule_set

    assert type(rule) is PatternRule
    assert rule.matching is rule_set.matching
    assert type(rule_set.matching) is Identity
    assert type(rule_set.normal_form) is NoNormalForm
    assert rule.restriction is None


def test_an_s_expression_signature_keeps_a_declared_restriction() -> None:
    declared = _explicit()
    rules = RuleSetSpec(
        name="restricted",
        rules=(
            PatternRuleSpec(
                name="s-from-vp",
                premises=(_node("VP"),),
                conclusion=_node("S"),
                head_policy="pass-through",
                restriction=RuleRestrictionSpec(
                    primary_targets=('("#const[Cat]" "VP")',)
                ),
            ),
        ),
    )
    rules = _compiled(declared, rules, matching=IdentityMatchingSpec()).built
    (rule,) = rules.rule_set

    assert rule.restriction is not None
    assert rule.restriction.primary_targets == frozenset({'("#const[Cat]" "VP")'})
    assert rules.rule_set.restriction_mode == "grammar_specific"


@settings(max_examples=8 * SMALL.examples, deadline=None)
@given(signatures(SMALL), st.data())
def test_a_generated_signature_builds_and_reads_each_declared_constructor(
    declared: SignatureSpec, data: st.DataObject
) -> None:
    store = _store(declared)
    signature = store.signature
    syntax, _ = _build_runtime(
        declared,
        store,
        _term_matching(FeatureErasureMatchingSpec()),
    )
    sort = declared.category_sort

    assert set(declared.primitives) <= set(signature.vocabulary(sort).names)
    assert all(
        signature.by_name(constructor.name).arity == constructor.arity
        for constructor in declared.constructors
    )
    assert all(
        signature.constant_core(sort, operator.name) == operator.erases_to
        for operator in declared.operators
    )

    result, argument = data.draw(
        st.lists(st.sampled_from(declared.primitives), min_size=2, max_size=2)
    )
    direction = data.draw(st.sampled_from(("forward", "backward")))
    spine = Spine(
        atom_text(declared, result), ((atom_text(declared, argument), direction),)
    )
    term = syntax.parse(store, category_text(declared, spine))
    role = next(
        view
        for view in declared.functors
        if view.argument_side
        in ({"right", "either"} if direction == "forward" else {"left", "either"})
    )

    assert store.payload_name(term.args[role.result]) == result
    assert store.payload_name(term.args[role.argument]) == argument
    assert syntax.parse(store, syntax.render(store, term)) is term
