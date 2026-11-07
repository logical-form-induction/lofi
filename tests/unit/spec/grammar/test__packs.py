import didactic.api as dx
import pytest

from lofi.spec import (
    ApplicationSpec,
    CoercionSetSpec,
    CombinatorSetSpec,
    CompositionSpec,
    ConstructorSpec,
    InfixConstructorNotationSpec,
    PatternRuleSpec,
    RuleSetSpec,
    RuleSpec,
)
from lofi.spec.grammar import (
    CapacityEvidenceSpec,
    FunctorSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
    declare_grammar,
)
from lofi.spec.grammar._packs import (
    ABFormalismSpec,
    CFGFormalismSpec,
    CFGProductionSpec,
    ModeOrderSpec,
    ModeRelationSpec,
    TLGFormalismSpec,
    UndirectedCCGFormalismSpec,
)


def _lexicon(*entries: tuple[str, str]) -> InlineLexiconSpec:
    return InlineLexiconSpec(
        entries=tuple(
            LexicalEntrySpec(form=form, category=category) for form, category in entries
        )
    )


def _slashes() -> SignatureSpec:
    constructors = tuple(
        ConstructorSpec(
            name=name,
            kind=("Cat", "Cat", "Cat"),
            notation=InfixConstructorNotationSpec(
                spelling=name, precedence=10, associativity="left"
            ),
        )
        for name in ("/", "\\")
    )
    return SignatureSpec(
        name="slashes",
        primitives=("S", "NP"),
        constructors=constructors,
        functors=(
            FunctorSpec(
                name="/", constructor="/", argument_side="right", opposite="\\"
            ),
            FunctorSpec(
                name="\\", constructor="\\", argument_side="left", opposite="/"
            ),
        ),
    )


def test_cfg_expands_arbitrary_arity_productions_to_the_generic_boundary() -> None:
    source = CFGFormalismSpec(
        nonterminals=("S", "NP", "VP", "Adv"),
        productions=(
            CFGProductionSpec(
                name="ternary",
                parent="S",
                children=("NP", "VP", "Adv"),
            ),
        ),
        lexicon=_lexicon(("kim", "NP"), ("runs", "VP"), ("today", "Adv")),
        goal="S",
    )

    declared = declare_grammar(source)
    (rule,) = declared.rules.rules

    assert declared.formalism == "declared"
    assert isinstance(rule, PatternRuleSpec)
    assert len(rule.premises) == 3
    assert declared.budget == source.budget

    with pytest.raises(dx.ValidationError):
        CFGFormalismSpec.model_validate(
            source.model_dump() | {"layout": {"kind": "preorder-arity", "max_nodes": 8}}
        )


def test_ab_expands_only_application_rules() -> None:
    evidence = CapacityEvidenceSpec(
        rule_family="pure_ccg",
        bounded_degree=True,
        prefix_closed=True,
        application_target_restrictions=False,
    )
    source = ABFormalismSpec(
        name="ab",
        signature=_slashes(),
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
            ),
        ),
        lexicon=_lexicon(("kim", "NP"), ("runs", "S\\NP")),
        goal=RootGoalSpec(categories=("S",)),
        capacity_evidence=evidence,
    )

    declared = declare_grammar(source)

    assert declared.formalism == "declared"
    assert tuple(rule.rule for rule in declared.rules.rules) == (
        "application",
        "application",
    )
    assert declared.capacity_evidence == evidence

    with pytest.raises(dx.ValidationError):
        ABFormalismSpec.model_validate(
            source.model_dump() | {"layout": {"kind": "complete-tree", "depth": 5}}
        )


def _ab(*rules: RuleSpec) -> ABFormalismSpec:
    return ABFormalismSpec(
        name="ab",
        signature=_slashes(),
        rules=RuleSetSpec(name="ab", rules=rules),
        lexicon=_lexicon(("kim", "NP"), ("runs", "S\\NP")),
        goal=RootGoalSpec(categories=("S",)),
    )


def test_ab_admits_a_combinator_set_that_expands_only_into_application() -> None:
    source = _ab(CombinatorSetSpec(application=True))

    assert declare_grammar(source).rules.plugin_tags == ("application",)


@pytest.mark.parametrize(
    "rule",
    [
        CombinatorSetSpec(application=True, composition=(1,)),
        CompositionSpec(direction="unrestricted", order=1, view="/"),
        CoercionSetSpec(label="lexical", entries=("NP => S",)),
    ],
    ids=("combinators", "composition", "coercions"),
)
def test_ab_refuses_a_rule_that_expands_beyond_application(rule: RuleSpec) -> None:
    with pytest.raises(dx.ValidationError, match="non-application rules"):
        _ab(ApplicationSpec(view="/"), rule)


def test_undirected_ccg_requires_one_self_opposite_functor() -> None:
    source = UndirectedCCGFormalismSpec(
        name="undirected",
        signature=SignatureSpec(
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
                    name="arrow",
                    constructor="arrow",
                    argument_side="either",
                    opposite="arrow",
                ),
            ),
            syntax="s-expression",
        ),
        rules=RuleSetSpec(name="none", rules=()),
        lexicon=_lexicon(("kim", '("#const[Cat]" "NP")')),
    )

    assert declare_grammar(source).signature == source.signature

    with pytest.raises(dx.ValidationError):
        UndirectedCCGFormalismSpec.model_validate(
            source.model_dump() | {"layout": {"kind": "preorder-arity", "max_nodes": 8}}
        )

    with pytest.raises(dx.ValidationError, match="self-opposite"):
        UndirectedCCGFormalismSpec(
            name="directed",
            signature=_slashes(),
            rules=source.rules,
            lexicon=source.lexicon,
        )


def test_tlg_expands_each_mode_and_preserves_the_hypothesis_bound() -> None:
    source = TLGFormalismSpec(
        primitives=("S", "NP"),
        modes=ModeOrderSpec(
            modes=("structural", "linear"),
            order=(ModeRelationSpec(lower="linear", upper="structural"),),
        ),
        lexicon=_lexicon(("kim", '("#const[Cat]" "NP")')),
        goal="S",
        max_hypotheses=2,
    )

    declared = declare_grammar(source)

    assert declared.max_hypotheses == 2
    assert len(declared.signature.constructors) == 10
    assert declared.matching.kind == "modal"

    with pytest.raises(dx.ValidationError):
        TLGFormalismSpec.model_validate(
            source.model_dump() | {"layout": {"kind": "preorder-arity", "max_nodes": 8}}
        )
