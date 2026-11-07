import sys
from collections import Counter
from copy import copy, deepcopy
from hashlib import sha256
from json import dumps, loads
from os import sep
from pathlib import Path
from random import Random
from subprocess import run
from textwrap import dedent
from types import BuiltinFunctionType, FrameType
from typing import TYPE_CHECKING, Final, Literal, cast, override

import didactic.api as dx
import panproto
import pytest
from hypothesis import given, settings
from tests.declarations import SMALL, category_language, grammars
from tests.language import language
from tests.support import grammar_theory_oracle
from tests.support.grammar_theory_oracle import (
    DERIVATION_FAMILY,
    LEXICAL_OPERATION,
    CheckedTheory,
    DerivationConstructor,
    mirror,
    theory_bytes,
)
from tests.support.installed import entries, install
from tests.support.rules import declared_rules
from tests.unit.spec.grammar.test__signature import slash_views, slashes
from tests.unit.spec.test__matching import _isolated_variants

from lofi._native import SpanFree
from lofi.deduce import ParsePolicy, Reachability, Retain, Viterbi, parse
from lofi.formalism import (
    Budget,
    Grammar,
    ProfileBound,
    ProfileUnsupportedError,
)
from lofi.judge import (
    NoNormalForm,
    NormalFormRole,
    PatternRule,
    RuleSet,
)
from lofi.lexicon import StaticLexicon
from lofi.registry import Registry
from lofi.spec import (
    ApplicationSpec,
    CombinatorSetSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    DeclarationError,
    DerivationIndexError,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PrefixConstructorNotationSpec,
    RuleRestrictionSpec,
    RuleSetSpec,
    RuleSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec._rule import (
    FixedPayloadSpec,
    PatternSpec,
    SpinePatternSpec,
    VariableOperatorSpec,
    VariablePayloadSpec,
)
from lofi.spec._rule_laws import LawGrammar
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.grammar import (
    GRAMMAR_SOURCE_VERSION,
    GRAMMAR_TARGET_VERSION,
    AdoptedRules,
    AlphaMatchingSpec,
    BudgetSpec,
    CapacityEvidenceSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GeneratedLexiconSpec,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    HockenmaierBiskNormalFormSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    NoNormalFormSpec,
    NormalFormPlugin,
    NormalFormSpec,
    OperatorSpec,
    ParametricAmbiguityLawSpec,
    PowerLawRankLawSpec,
    ProductMatchingSpec,
    RootGoalSpec,
    SignatureSpec,
    SubsumptionMatchingSpec,
    UniformRankLawSpec,
    canonical_bytes,
    encode_identifier,
)
from lofi.spec.language import LanguageIso, SortBoundSpec, registered_language
from lofi.terms import (
    SUBSTRATE_PACKAGES,
    BindingSite,
    CompleteTree,
    ConstructorSyntax,
    Identity,
    MVar,
    Node,
    Operator,
    PVar,
    SExpSyntax,
    Signature,
    Spine,
    Term,
    TermCodec,
    TermStore,
    Vocabulary,
    _compile_grammar,
    _CompiledGrammar,
    beta_normal,
    substrate_reaches,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Mapping

    from lofi._types import JsonValue
    from lofi.deduce import Cell, PackedForest
    from lofi.judge import (
        Direction,
        GeneralizedComposition,
        PatternRule,
        ProofWitness,
        Rule,
        RuleProbe,
        RuleRestriction,
        SchematicSequent,
        Slot,
    )
    from lofi.lexicon import Entry
    from lofi.terms import Matching, Pattern, TermRuleKernel
    from lofi.typecheck import TypingResult


X: Final = TermPatternSpec(name="X", sort="Cat")
Y: Final = TermPatternSpec(name="Y", sort="Cat")


def _node(
    name: str, *arguments: TermPatternSpec | NodePatternSpec | ConstantPatternSpec
) -> NodePatternSpec | ConstantPatternSpec:
    if not arguments:
        return ConstantPatternSpec(name=name)

    return NodePatternSpec(operator=FixedOperatorSpec(name=name), arguments=arguments)


def _ccg() -> GrammarDecl:
    return GrammarDecl(
        name="toy-ccg",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "N", "conj"),
            operators=(
                OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
                OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
            ),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
                CompositionSpec(direction="unrestricted", order=1, view="/"),
                TypeRaisingSpec(view="/", targets=("S[dcl]",), arguments=None),
                PatternRuleSpec(
                    name="n-to-np",
                    premises=(_node("N"),),
                    conclusion=_node("NP"),
                    head_policy="pass-through",
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="john", category="NP"),
                LexicalEntrySpec(form="sleeps", category="S[dcl]\\NP"),
                LexicalEntrySpec(form="sees", category="(S[dcl]\\NP)/NP", weight=-1.0),
                LexicalEntrySpec(form="dogs", category="N"),
            )
        ),
        goal=RootGoalSpec(categories=("S[dcl]",)),
        matching=FeatureErasureMatchingSpec(),
        normal_form=HockenmaierBiskNormalFormSpec(),
    )


def _ccg_with_capacity_evidence() -> GrammarDecl:
    """Return the CCG declaration with explicit theorem hypotheses."""
    source = _ccg()

    return GrammarDecl(
        name=source.name,
        signature=source.signature,
        rules=source.rules,
        lexicon=source.lexicon,
        goal=source.goal,
        matching=source.matching,
        normal_form=source.normal_form,
        budget=source.budget,
        max_hypotheses=source.max_hypotheses,
        capacity_evidence=CapacityEvidenceSpec(
            rule_family="pure_ccg",
            bounded_degree=True,
            prefix_closed=True,
            application_target_restrictions=False,
        ),
    )


def test_one_crossing_parses_each_shared_category_source_once(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    source = _ccg()
    signature = source.signature.with_(
        conjunctions=ConjunctionsSpec(
            markers=("S[dcl]",),
            pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
        )
    )
    lexicon = cast("InlineLexiconSpec", source.lexicon)
    declaration = source.with_(
        signature=signature,
        lexicon=lexicon.with_(
            entries=(
                *lexicon.entries,
                LexicalEntrySpec(form="root", category="S[dcl]"),
            )
        ),
    )
    calls: Counter[str] = Counter()
    parse = ConstructorSyntax.parse

    def counted_parse(
        syntax: ConstructorSyntax,
        store: TermStore,
        category: str,
        /,
    ) -> Term:
        calls[category] += 1
        return parse(syntax, store, category)

    monkeypatch.setattr(ConstructorSyntax, "parse", counted_parse)

    assembly = GrammarIso().forward(
        declaration,
        registered_language("propositional"),
    )

    assert calls["S[dcl]"] == 1
    assert (
        assembly.syntax.render(
            assembly.language.store,
            assembly.grammar.targets.categories[0],
        )
        == "S[dcl]"
    )


def _cfg() -> GrammarDecl:
    return GrammarDecl(
        name="toy-cfg",
        signature=SignatureSpec(
            name="cfg", primitives=("S", "NP", "VP"), syntax="s-expression"
        ),
        rules=RuleSetSpec(
            name="productions",
            rules=(
                PatternRuleSpec(
                    name="s",
                    premises=(_node("NP"), _node("VP")),
                    conclusion=_node("S"),
                    head_policy="pass-through",
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="john", category='("#const[Cat]" "NP")'),
                LexicalEntrySpec(form="sleeps", category='("#const[Cat]" "VP")'),
            )
        ),
        goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
        budget=BudgetSpec(unary_rounds=0, closure_rounds=1, closure_nodes=8),
    )


TRANSIENTS: Final = (dx.LoweringRoute,)
"""Boundary route values that must not survive into a built graph."""


def _substrate_code_entered[R](run: Callable[[], R]) -> tuple[str, ...]:
    """Return the substrate functions a workload calls, in call order.

    A Python frame counts when its code lives under the didactic or panproto
    package, and a builtin call counts when the function or its receiver is
    defined there.
    """
    roots = tuple(
        str(Path(module).resolve().parent) + sep
        for module in (dx.__file__, panproto.__file__)
    )
    entered: list[str] = []

    def record[A](frame: FrameType, event: str, called: A) -> None:
        if event == "call":
            code = frame.f_code

            if code.co_filename.startswith(roots):
                entered.append(code.co_qualname)
        elif event == "c_call" and isinstance(called, BuiltinFunctionType):
            owner = type(called.__self__).__module__
            module = called.__module__ or ""

            if module.partition(".")[0] in SUBSTRATE_PACKAGES or (
                owner.partition(".")[0] in SUBSTRATE_PACKAGES
            ):
                entered.append(called.__qualname__)

    sys.setprofile(record)

    try:
        run()
    finally:
        sys.setprofile(None)

    return tuple(entered)


class _Unweighted:
    lowering = SpanFree()

    """A scorer giving every axiom and edge zero log weight."""

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
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
        del rule, inputs, output, span, split

        return 0.0


def _sentences(count: int) -> tuple[tuple[str, ...], ...]:
    """Return sentences over the toy lexicon, each up to six tokens long."""
    forms = ("john", "sees", "dogs", "sleeps")
    drawn = Random(count)

    return tuple(
        tuple(drawn.choice(forms) for _ in range(drawn.randint(1, 6)))
        for _ in range(count)
    )


@pytest.mark.parametrize("declaration", [_ccg(), _cfg()], ids=lambda d: d.name)
def test_lowering_then_inverting_returns_the_declaration(
    declaration: GrammarDecl,
) -> None:
    iso = GrammarIso()
    language = registered_language("propositional")
    assembly = iso.forward(declaration, language)

    assert iso.backward(assembly) == declaration
    assert iso.forward(iso.backward(assembly), language) is assembly


@pytest.mark.parametrize("declaration", [_ccg(), _cfg()], ids=lambda d: d.name)
def test_a_declaration_compiles_once_and_rebuilds_identically(
    declaration: GrammarDecl,
) -> None:
    iso = GrammarIso()
    language = registered_language("propositional")
    first = iso.forward(declaration, language)
    again = GrammarDecl.model_validate_json(canonical_bytes(declaration))

    assert iso.forward(again, language) is first
    assert iso.forward(declaration.with_(name=declaration.name), language) is first

    other = GrammarIso().forward(again, language)

    assert other is not first
    assert other.grammar.signature is not first.grammar.signature
    assert other == first
    assert hash(other) == hash(first)
    assert other.grammar.fingerprint == first.grammar.fingerprint
    assert other.grammar.theory_digest == first.grammar.theory_digest
    assert other.declaration == first.declaration == canonical_bytes(declaration)
    assert other.declaration_digest == sha256(first.declaration).hexdigest()


def test_equal_term_signatures_do_not_merge_distinct_logic_profiles() -> None:
    iso = GrammarIso()
    declaration = _ccg()
    first_order = registered_language("propositional")
    higher_order = first_order.with_(logic=first_order.logic.with_(order="higher"))

    first = iso.forward(declaration, first_order)
    higher = iso.forward(declaration, higher_order)

    assert higher is not first
    assert first.language.profile.fragment.order == "first"
    assert higher.language.profile.fragment.order == "higher"
    assert higher.grammar.theory_digest == first.grammar.theory_digest


@pytest.mark.parametrize(
    ("declaration", "entry"),
    [
        (_ccg(), LexicalEntrySpec(form="john", category="NP")),
        (_cfg(), LexicalEntrySpec(form="john", category='("#const[Cat]" "NP")')),
    ],
    ids=("toy-ccg", "toy-cfg"),
)
def test_grammars_declared_over_one_language_share_its_store_and_signature(
    declaration: GrammarDecl, entry: LexicalEntrySpec
) -> None:
    iso = GrammarIso()
    language = registered_language("propositional")
    first = iso.forward(declaration, language)
    target = declaration.goal.categories[0]
    relexicalized = declaration.with_(
        name="relexicalized",
        lexicon=InlineLexiconSpec(
            entries=(entry, LexicalEntrySpec(form="sentence", category=target))
        ),
    )
    second = iso.forward(relexicalized, language)
    signature = first.grammar.signature

    assert second is not first
    assert second.language is first.language
    assert second.grammar.store is first.grammar.store
    assert second.grammar.signature is signature
    assert second.grammar.theory_digest == first.grammar.theory_digest
    assert second.language.codec.index.signature is signature
    assert second.syntax.signature is signature

    (held,) = first.lexicon.entries(entry.form)
    (shared,) = second.lexicon.entries(entry.form)

    assert shared.category is held.category


def test_native_theory_identity_ignores_presentation_but_tracks_relations() -> None:
    language = registered_language("propositional")
    source = _cfg()

    def plan(declaration: GrammarDecl) -> _CompiledGrammar:
        return _compile_grammar(
            language.compile_envelope.canonical_declaration,
            declaration.compile_envelope.canonical_declaration,
        )

    baseline = plan(source)
    renamed = plan(source.with_(rules=source.rules.with_(name="renamed-rules")))
    budgeted = plan(
        source.with_(budget=source.budget.with_(beam=source.budget.beam + 1))
    )
    bounded = plan(source.with_(max_hypotheses=source.max_hypotheses + 1))
    retargeted = plan(
        source.with_(goal=RootGoalSpec(categories=('("#const[Cat]" "NP")',)))
    )
    uninterpreted = plan(source.with_(interpretations=()))
    evidenced = plan(
        source.with_(
            capacity_evidence=CapacityEvidenceSpec(
                rule_family="pure_ccg",
                bounded_degree=True,
                prefix_closed=True,
                application_target_restrictions=False,
            )
        )
    )
    np_identity = source.with_(
        rules=RuleSetSpec(
            name="unary",
            rules=(
                PatternRuleSpec(
                    name="identity",
                    premises=(_node("NP"),),
                    conclusion=_node("NP"),
                    head_policy="pass-through",
                ),
            ),
        )
    )
    vp_identity = np_identity.with_(
        rules=RuleSetSpec(
            name="unary",
            rules=(
                PatternRuleSpec(
                    name="identity",
                    premises=(_node("VP"),),
                    conclusion=_node("VP"),
                    head_policy="pass-through",
                ),
            ),
        )
    )
    np_plan = plan(np_identity)
    vp_plan = plan(vp_identity)

    for presentation_only in (
        renamed,
        budgeted,
        bounded,
        retargeted,
        uninterpreted,
        evidenced,
    ):
        assert presentation_only.rule_fingerprint == baseline.rule_fingerprint
        assert presentation_only.theory_constructors == baseline.theory_constructors
    assert np_plan.rule_fingerprint != vp_plan.rule_fingerprint
    assert np_plan.theory_constructors != vp_plan.theory_constructors


@pytest.mark.parametrize("declaration", [_ccg(), _cfg()], ids=lambda d: d.name)
def test_converters_given_one_language_converter_share_the_extended_language(
    declaration: GrammarDecl,
) -> None:
    languages = LanguageIso()
    grammar = GrammarIso(languages)
    language = registered_language("propositional")
    first = grammar.forward(declaration, language)
    second = grammar.forward(
        GrammarDecl.model_validate_json(canonical_bytes(declaration)), language
    )

    assert second is first
    assert grammar.languages is languages
    assert second.language is first.language


def test_equal_explicit_language_declarations_share_one_native_assembly() -> None:
    declaration = _ccg()
    source = registered_language("propositional")
    first_language = LanguageIso().extend(source, declaration.signature.extend)
    second_language = LanguageIso().extend(source, declaration.signature.extend)
    iso = GrammarIso()

    first = iso.forward_language(declaration, first_language)
    second = iso.forward_language(declaration, second_language)

    assert first.language is not first_language
    assert first.store is first.language.store
    assert first.grammar.signature is first.language.store.signature
    assert second is first
    assert iso.forward_language(declaration, first_language) is first


def test_production_lowering_cannot_reach_superseded_python_compilers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    declaration = _ccg()
    language = registered_language("propositional")

    def unreachable(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "a superseded Python compiler was reached"
        raise AssertionError(msg)

    monkeypatch.setattr(SignatureSpec, "extend", unreachable)

    assert not hasattr(RuleSetSpec, "expanded")
    assert not hasattr(RuleSetSpec, "build")
    assert not hasattr(RuleSpec, "build")
    assert not hasattr(CombinatorSetSpec, "expand")
    grammar = GrammarIso().forward(declaration, language)

    assert grammar.store is grammar.language.store
    assert grammar.grammar.signature is grammar.language.store.signature


def test_canonical_bytes_are_ascii_compact_with_sorted_keys() -> None:
    declaration = _ccg()
    encoded = canonical_bytes(declaration)
    sorted_keys = dumps(
        loads(declaration.model_dump_json()),
        separators=(",", ":"),
        ensure_ascii=True,
        sort_keys=True,
    ).encode()

    assert encoded == declaration.compile_envelope.canonical_declaration
    assert encoded == sorted_keys
    assert encoded.isascii()
    assert b" " not in encoded.split(b'"sleeps"')[0]

    accented = declaration.with_(
        lexicon=InlineLexiconSpec(
            entries=(LexicalEntrySpec(form="ni\u00f1o", category="NP"),)
        )
    )
    escaped = canonical_bytes(accented)

    assert escaped.isascii()
    assert b'"ni\\u00f1o"' in escaped
    assert GrammarDecl.model_validate_json(escaped) == accented
    assert [
        operator["name"] for operator in loads(encoded)["signature"]["operators"]
    ] == ["S[dcl]", "NP[conj]"]
    assert [
        constructor["name"]
        for constructor in loads(encoded)["signature"]["constructors"]
    ] == ["/", "\\"]
    assert GrammarDecl.model_validate_json(encoded) == declaration

    swapped = declaration.with_(
        signature=declaration.signature.with_(
            operators=(
                OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
                OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
            )
        )
    )

    assert canonical_bytes(swapped) != encoded
    assert GrammarIso().forward(
        swapped, registered_language("propositional")
    ).grammar.signature.vocabulary("Cat").id("S[dcl]") != (
        GrammarIso()
        .forward(declaration, registered_language("propositional"))
        .grammar.signature.vocabulary("Cat")
        .id("S[dcl]")
    )


def test_adopting_a_grammars_rules_builds_neither_its_lexicon_nor_its_goal() -> None:
    declaration = _ccg().with_(goal=RootGoalSpec(categories=("conj",)))
    iso = GrammarIso()

    with pytest.raises(DeclarationError) as refused:
        iso.forward(declaration, registered_language("propositional"))

    assert refused.value.path == "grammar.goal.categories"

    adopted = iso.rules(declaration, registered_language("propositional"))

    assert type(adopted) is AdoptedRules
    assert adopted.rule_set is adopted.built.rule_set
    assert [rule.name for rule in adopted.rule_set] == [
        "application:/",
        "application:\\",
        "composition:/:1:harmonic:unrestricted",
        "type-raising:/:targets=S[dcl]",
        "n-to-np",
    ]
    assert adopted.built.head_selections["n-to-np"].policy == "pass-through"
    assert adopted.context.store.contains(adopted.resolve_category("S[dcl]"))


def test_adopted_rules_over_one_signature_share_its_linked_store() -> None:
    iso = GrammarIso()
    language = registered_language("propositional")
    first = iso.rules(_ccg(), language)
    second = iso.rules(
        _ccg().with_(
            rules=RuleSetSpec(name="forward", rules=(ApplicationSpec(view="/"),))
        ),
        language,
    )
    syntax = first.context.syntax
    store = first.context.store
    (application,) = second.rule_set

    assert second.context.store is store
    assert tuple(
        application.apply((syntax.parse(store, "S[dcl]/NP"), syntax.parse(store, "NP")))
    ) == (syntax.parse(store, "S[dcl]"),)


def test_adopting_rules_refuses_an_inventory_the_native_compiler_refuses() -> None:
    declaration = _ccg().with_(
        rules=RuleSetSpec(
            name="twice",
            rules=(
                PatternRuleSpec(
                    name="n-to-np",
                    premises=(_node("N"),),
                    conclusion=_node("NP"),
                    head_policy="pass-through",
                ),
                PatternRuleSpec(
                    name="n-to-np-again",
                    premises=(_node("N"),),
                    conclusion=_node("NP"),
                    head_policy="pass-through",
                ),
            ),
        )
    )

    with pytest.raises(DeclarationError, match="same relation twice"):
        GrammarIso().rules(declaration, registered_language("propositional"))


def test_a_ccg_assembly_holds_the_compiled_grammar_lexicon_syntax_and_language() -> (
    None
):
    assembly = GrammarIso().forward(_ccg(), registered_language("propositional"))
    grammar = assembly.grammar
    codec = assembly.language.codec.at(assembly.language.category_sort)

    assert type(grammar) is Grammar
    assert grammar.name == "toy-ccg"
    assert type(grammar.rules) is RuleSet
    assert [rule.name for rule in grammar.rules] == [
        "application:/",
        "application:\\",
        "composition:/:1:harmonic:unrestricted",
        "type-raising:/:targets=S[dcl]",
        "n-to-np",
    ]
    assert grammar.normal_form.name == "hockenmaier-bisk"
    assert grammar.profile.ranks == frozenset({1, 2})
    assert grammar.profile.unary_closure is True
    assert grammar.budget.unary_rounds == 1
    assert grammar.goal.accepts is not None
    assert (
        grammar.theory_version == f"{GRAMMAR_SOURCE_VERSION}->{GRAMMAR_TARGET_VERSION}"
    )

    assert type(assembly.lexicon) is StaticLexicon
    assert assembly.lexicon.forms == ("dogs", "john", "sees", "sleeps")
    assert assembly.store is grammar.store
    assert type(assembly.syntax) is ConstructorSyntax
    assert (
        assembly.syntax.render(
            assembly.store, assembly.lexicon.entries("sees")[0].category
        )
        == "S[dcl]\\NP/NP"
    )
    assert grammar.targets.categories == (
        assembly.syntax.parse(assembly.store, "S[dcl]"),
    )
    assert grammar.targets.matching is grammar.matching

    assert type(codec) is TermCodec
    assert codec is assembly.language.codec.at(assembly.language.category_sort)
    assert codec.index.signature is grammar.signature
    assert type(codec.layout) is CompleteTree
    assert codec.encode(grammar.targets.categories[0]).length == 1
    assert assembly.source_version == GRAMMAR_SOURCE_VERSION
    assert assembly.target_version == GRAMMAR_TARGET_VERSION
    assert assembly.category_measure.constructors == (("/", 0), ("\\", 0))

    assert grammar.signature.by_name("/").family == "constructor"
    assert grammar.signature.constant_core("Cat", "S[dcl]") == "S"
    assert grammar.signature.constant_core("Cat", "NP[conj]") == "NP"


def test_a_generated_lexicon_lowers_against_the_compiled_rules_and_goal() -> None:
    declaration = _ccg().with_(
        lexicon=GeneratedLexiconSpec(
            forms=30,
            categories=8,
            category_shares=PowerLawRankLawSpec(exponent=1.0),
            coupling=0.0,
            ambiguity=ParametricAmbiguityLawSpec(
                mean=2.0, dispersion=0.5, maximum=4, placement=0.5
            ),
            rank_law=UniformRankLawSpec(),
            seed=4,
        )
    )
    assembly = GrammarIso().forward(declaration, registered_language("propositional"))
    lexicon = assembly.lexicon
    categories = {
        entry.category for form in lexicon.forms for entry in lexicon.entries(form)
    }

    atoms: set[str] = set()
    pending = list(categories)

    while pending:
        term = pending.pop()
        atoms.update(() if term.args else (term.op.name,))
        pending.extend(term.args)

    assert len(categories) == 8
    assert all(assembly.store.contains(category) for category in categories)
    assert "conj" not in atoms
    assert GrammarIso().backward(assembly) == declaration
    assert (
        GrammarIso()
        .forward(declaration, registered_language("propositional"))
        .lexicon.forms
        == lexicon.forms
    )


def test_a_cfg_assembly_reads_s_expressions_through_the_category_codec() -> None:
    assembly = GrammarIso().forward(_cfg(), registered_language("propositional"))
    grammar = assembly.grammar
    codec = assembly.language.codec.at(assembly.language.category_sort)

    assert type(assembly.syntax) is SExpSyntax
    assert type(codec.layout) is CompleteTree
    assert assembly.category_measure.constructors == ()
    assert grammar.profile.ranks == frozenset({2})
    assert grammar.profile.unary_closure is False
    assert grammar.matching.name == "identity"
    assert grammar.normal_form.name == "none"
    assert [rule.name for rule in grammar.rules] == ["s"]
    assert "VP" in grammar.signature.vocabulary("Cat")
    assert assembly.lexicon.entries("john")[0].category is assembly.syntax.parse(
        assembly.store, '("#const[Cat]" "NP")'
    )
    assert (
        codec.decode(codec.encode(grammar.targets.categories[0]), assembly.store)
        is grammar.targets.categories[0]
    )


def test_a_malformed_declared_result_index_fails_the_check_stage() -> None:
    malformed = _ccg().with_(
        rules=RuleSetSpec(
            name="malformed",
            rules=(
                PatternRuleSpec(
                    name="malformed",
                    premises=(_node("NP"),),
                    conclusion=_node("/", X),
                    head_policy="pass-through",
                ),
            ),
        )
    )
    fault = "arity 1"

    with pytest.raises(DeclarationError, match=fault):
        GrammarIso().forward(malformed, registered_language("propositional"))


@pytest.mark.parametrize("declaration", [_ccg(), _cfg()], ids=lambda d: d.name)
def test_the_assembly_reaches_no_substrate_object_or_boundary_transient(
    declaration: GrammarDecl,
) -> None:
    assembly = GrammarIso().forward(declaration, registered_language("propositional"))

    assert substrate_reaches(assembly, refused=TRANSIENTS) == ()
    assert type(assembly.declaration) is bytes
    assert type(assembly.grammar.theory_digest) is str


def test_the_crossing_does_not_reenter_the_diagnostic_theory_compiler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    uncrossed = _cfg().with_(name="substrate-probe")

    def refused(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "production reentered the differential theory oracle"
        raise AssertionError(msg)

    monkeypatch.setattr(grammar_theory_oracle, "mirror", refused)

    assembly = GrammarIso().forward(uncrossed, registered_language("propositional"))

    assert assembly.grammar.name == "substrate-probe"


def test_the_crossing_uses_only_plugins_prepared_with_the_declarations(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    grammar = _ccg().with_(name="prepared-plugin-crossing")
    language = registered_language("propositional")

    def refused(registry: Registry[object], /, *_: object) -> None:
        msg = f"crossing rediscovered {registry.group!r} entry points"
        raise AssertionError(msg)

    monkeypatch.setattr(Registry, "entries", refused)

    assembly = GrammarIso().forward(grammar, language)

    assert assembly.grammar.name == grammar.name


def test_parsing_and_typing_after_the_crossing_enter_no_substrate_code() -> None:
    assembly = GrammarIso().forward(_ccg(), registered_language("propositional"))
    grammar = assembly.grammar
    sentences = _sentences(200)
    semantic = language(("john", "sleep"), types={"john": "e", "sleep": "e -> t"})
    lf_store = semantic.store
    checker = semantic.checker
    rules = semantic.rules
    calculus = semantic.calculus
    sleeps = semantic.constant("sleep")
    john = semantic.constant("john")
    lexicon = semantic.lexicon
    body = semantic.apply(sleeps, john)
    parsed: list[int] = []
    forests: list[PackedForest[Term, float]] = []
    results: list[TypingResult] = []
    typed: list[bool] = []
    normal: list[Term] = []

    def workload() -> None:
        for sentence in sentences:
            forest = parse(
                sentence,
                policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
                rules=grammar.rules,
                source=assembly.lexicon,
                semiring=Viterbi(),
                scorer=_Unweighted(),
                goal=grammar.goal,
                unary_depth=grammar.budget.unary_rounds,
            )
            forests.append(forest)
            parsed.append(len(tuple(grammar.goal.roots(forest))))

        for serial in range(100):
            variable = f"x{serial}"
            argument = lf_store.fvar(semantic.term_sort, variable)
            redex = semantic.apply(
                lf_store.bind(calculus.lam, variable, semantic.apply(sleeps, argument)),
                john,
            )
            normal.append(beta_normal(lf_store, redex, calculus, fuel=10))
            result = checker.check(body, lexicon, rules)
            results.append(result)
            typed.append(result.is_well_typed)

    entered = _substrate_code_entered(workload)

    assert entered == ()
    assert len(parsed) == 200
    assert sum(parsed) > 0
    assert typed == [True] * 100
    assert normal == [body] * 100

    built = {
        "forests": forests,
        "results": results,
        "normal": normal,
        "checker": checker,
        "rules": rules,
        "lexicon": lexicon,
        "store": lf_store,
    }

    assert substrate_reaches(built, refused=TRANSIENTS) == ()


def test_the_declared_congruence_and_budget_are_lowered() -> None:
    budget = BudgetSpec(
        depth=3,
        mass=0.5,
        beam=2,
        unary_rounds=0,
        beta_fuel=10,
        closure_rounds=1,
        closure_nodes=8,
    )
    declaration = _cfg().with_(budget=budget, matching=AlphaMatchingSpec())
    grammar = (
        GrammarIso().forward(declaration, registered_language("propositional")).grammar
    )
    default = GrammarIso().forward(_cfg(), registered_language("propositional")).grammar

    assert grammar.matching.name == "alpha"
    assert all(rule.matching is grammar.matching for rule in grammar.rules)
    assert grammar.profile.fanout == 1
    assert grammar.profile.lexicalized is True
    assert grammar.budget == Budget(
        depth=3,
        mass=0.5,
        beam=2,
        unary_rounds=0,
        threads=1,
        beta_fuel=10,
        max_category_arity=None,
        max_category_size=None,
    )
    assert default.profile.fanout == 1
    assert default.profile.lexicalized is True
    assert default.budget == Budget(
        depth=5,
        mass=1.0,
        beam=8,
        unary_rounds=0,
        threads=1,
        beta_fuel=1_000,
        max_category_arity=None,
        max_category_size=None,
    )
    assert grammar.fingerprint != default.fingerprint


def test_the_compile_result_carries_declared_capacity_hypotheses() -> None:
    language = registered_language("propositional")
    declared = GrammarIso().forward(_ccg_with_capacity_evidence(), language).grammar
    undeclared = GrammarIso().forward(_ccg(), language).grammar

    assert declared.capacity.rule_family == "pure_ccg"
    assert not declared.capacity.theorem_applicable
    assert declared.capacity.reason == (
        "the inventory is not exclusively generalized CCG composition"
    )
    assert undeclared.capacity.rule_family is None
    assert declared.fingerprint != undeclared.fingerprint


@pytest.mark.parametrize(
    "matching",
    [
        ProductMatchingSpec(
            left=AlphaMatchingSpec(),
            right=FeatureErasureMatchingSpec(),
        ),
        SubsumptionMatchingSpec(),
    ],
    ids=("product", "subsumption"),
)
def test_product_and_subsumption_grammar_matchings_compile_native_representatives(
    matching: ProductMatchingSpec | SubsumptionMatchingSpec,
) -> None:
    declaration = _cfg().with_(matching=matching)
    assembly = GrammarIso().forward(declaration, registered_language("propositional"))

    assert assembly.grammar.matching.has_key
    assert all(
        rule.matching is assembly.grammar.matching for rule in assembly.grammar.rules
    )


def _native_theory_rows(
    plan: _CompiledGrammar, /
) -> tuple[
    tuple[str, str, tuple[str, ...], tuple[str, ...], int, tuple[str, ...], str],
    ...,
]:
    """Return the native derivation records as immutable rows."""
    return tuple(
        (
            rule,
            constructor,
            tuple(inferred),
            tuple(declared),
            premises,
            tuple(premise_indices),
            conclusion_index,
        )
        for (
            rule,
            constructor,
            inferred,
            declared,
            premises,
            premise_indices,
            conclusion_index,
        ) in plan.theory_constructors
    )


def _oracle_theory_rows(
    assembly: GrammarAssembly, /
) -> tuple[
    tuple[str, str, tuple[str, ...], tuple[str, ...], int, tuple[str, ...], str],
    ...,
]:
    """Return the differential oracle's derivation records."""
    grammar = assembly.grammar
    checked = mirror(
        grammar.name,
        grammar.signature,
        grammar.rules,
        assembly.language.category_sort,
    )

    return tuple(
        (
            constructor.rule,
            constructor.constructor,
            constructor.inferred,
            constructor.declared,
            constructor.premises,
            constructor.premise_indices,
            constructor.conclusion_index,
        )
        for constructor in checked.constructors
    )


@pytest.mark.parametrize(
    "declaration",
    [
        ccgbank_grammar(),
        _ccg(),
        _ccg().with_(
            matching=ProductMatchingSpec(
                left=AlphaMatchingSpec(),
                right=FeatureErasureMatchingSpec(),
            )
        ),
    ],
    ids=("packaged-ccgbank", "feature-erasure", "product-recomputation"),
)
def test_native_theory_is_differentially_equal_to_the_python_oracle(
    declaration: GrammarDecl,
) -> None:
    """Keep the native theory equal to the independent Python oracle."""
    language = registered_language("propositional")
    native = _compile_grammar(
        language.compile_envelope.canonical_declaration,
        declaration.compile_envelope.canonical_declaration,
    )
    assembly = GrammarIso().forward(declaration, language)

    assert len(assembly.grammar.rules) == len(native.rules)
    assert _native_theory_rows(native) == _oracle_theory_rows(assembly)
    checked = mirror(
        assembly.grammar.name,
        assembly.grammar.signature,
        assembly.grammar.rules,
        assembly.language.category_sort,
    )
    assert tuple(native.elaborated_rules) == checked.elaborated


def test_native_pattern_plan_exposes_only_typed_runtime_records() -> None:
    language = registered_language("propositional")
    declaration = ccgbank_grammar()
    native = _compile_grammar(
        language.compile_envelope.canonical_declaration,
        declaration.compile_envelope.canonical_declaration,
    )
    compiled = next(rule for rule in native.rules if rule.kind == "pattern")

    assert not hasattr(compiled, "recipe")
    assert compiled.name is not None
    assert compiled.head_policy is not None
    assert compiled.normal_form_role is not None
    assert compiled.conclusion is not None
    assert compiled.conclusion.kind in {"constant", "node", "spine", "term"}
    assert all(
        premise.kind in {"constant", "node", "spine", "term"}
        for premise in compiled.premises
    )


def test_a_late_plugin_declares_the_same_native_and_runtime_theory() -> None:
    probe = dedent(
        """
        from typing import Literal, NamedTuple

        import didactic.api as dx
        from tests.unit.spec.grammar.test__lowering import (
            _cfg,
            _native_theory_rows,
            _oracle_theory_rows,
        )

        from lofi._types import JsonValue
        from lofi.spec import (
            ConstantPatternSpec,
            NormalFormRoleSpec,
            PatternSpec,
            RulePlugin,
            RuleSetSpec,
            RuleSpec,
        )
        from lofi.spec._ccg_rules import _adopt_pattern
        from lofi.registry import _registry as registry_module
        from lofi.spec.grammar import GrammarIso
        from lofi.spec.language import registered_language
        from lofi.terms import _compile_grammar


        class LatePatternSpec(RuleSpec, extra="forbid"):
            rule: Literal["late-theory-probe"] = "late-theory-probe"
            head_policy: Literal["pass-through"] = "pass-through"
            name: str
            premises: tuple[PatternSpec, ...]
            conclusion: PatternSpec
            targets: tuple[str, ...] | None = None
            normal_form_role: NormalFormRoleSpec = dx.field(
                default_factory=NormalFormRoleSpec
            )
            runtime: dict[str, JsonValue]
            theory: dict[str, JsonValue]


        plugin = RulePlugin(
            spec=LatePatternSpec,
            adopt=_adopt_pattern,
        )


        class RulePoint(NamedTuple):
            name: str
            value: str
            group: str
            member: object
            dist: None = None

            def load(self) -> object:
                return self.member


        installed = registry_module.entry_points


        def with_late_plugin() -> tuple[object, ...]:
            points = tuple(installed())
            return (
                *points,
                RulePoint(
                    "late-theory-probe",
                    "tests:late-pattern-plugin",
                    "lofi.rule",
                    plugin,
                ),
            )


        registry_module.entry_points = with_late_plugin
        premises = (
            ConstantPatternSpec(name="NP"),
            ConstantPatternSpec(name="VP"),
        )
        source = _cfg().with_(
            rules=RuleSetSpec(
                name="late",
                rules=(
                    LatePatternSpec(
                        name="late-s",
                        premises=premises,
                        conclusion=ConstantPatternSpec(name="S"),
                        runtime={"kind": "pattern", "head_slot": 0},
                        theory={
                            "forward_complete": True,
                            "kind": "schemata",
                            "name": "late-s",
                            "schemata": [
                                {
                                    "conclusion": {
                                        "name": "S",
                                        "pattern": "constant",
                                        "sort": None,
                                    },
                                    "premises": [
                                        {
                                            "name": "NP",
                                            "pattern": "constant",
                                            "sort": None,
                                        },
                                        {
                                            "name": "VP",
                                            "pattern": "constant",
                                            "sort": None,
                                        },
                                    ],
                                }
                            ],
                        },
                    ),
                ),
            )
        )
        language = registered_language("propositional")
        native = _compile_grammar(
            language.compile_envelope.canonical_declaration,
            source.compile_envelope.canonical_declaration,
        )
        assembly = GrammarIso().forward(source, language)

        assert _native_theory_rows(native) == _oracle_theory_rows(assembly)
        assert tuple(native.elaborated_rules) == ()
        """
    )
    completed = run(
        [sys.executable, "-c", probe],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr


def test_a_rule_subsumption_matching_compiles_and_applies_natively() -> None:
    declaration = _cfg().with_(
        rules=RuleSetSpec(
            name="subsuming-production",
            rules=(
                PatternRuleSpec(
                    name="s",
                    premises=(_node("NP"), _node("VP")),
                    conclusion=_node("S"),
                    head_policy="pass-through",
                    matching=SubsumptionMatchingSpec(),
                ),
            ),
        )
    )

    assembly = GrammarIso().forward(declaration, registered_language("propositional"))
    (rule,) = tuple(assembly.grammar.rules)
    inputs = (
        assembly.lexicon.entries("john")[0].category,
        assembly.lexicon.entries("sleeps")[0].category,
    )

    assert rule.matching.name == "subsumption"
    assert tuple(rule.apply(inputs)) == assembly.grammar.targets.categories


def _flat_rule(*premises: str) -> GrammarDecl:
    return _cfg().with_(
        rules=RuleSetSpec(
            name="flat",
            rules=(
                PatternRuleSpec(
                    name="flat",
                    premises=tuple(_node(name) for name in premises),
                    conclusion=_node("S"),
                    head_policy="pass-through",
                ),
            ),
        )
    )


def test_lowering_checks_the_symbolic_engine_bound_by_default() -> None:
    ternary = (
        GrammarIso()
        .forward(_flat_rule("NP", "VP", "VP"), registered_language("propositional"))
        .grammar
    )

    assert ternary.profile.ranks == frozenset({3})

    with pytest.raises(ProfileUnsupportedError) as refusal:
        GrammarIso().forward(
            _flat_rule("NP", "VP", "VP", "VP"), registered_language("propositional")
        )

    assert "lofi.deduce.parse" in str(refusal.value)
    assert "[4]" in str(refusal.value)


def test_lowering_refuses_an_arity_a_supplied_consumer_cannot_serve() -> None:
    with pytest.raises(ProfileUnsupportedError) as refusal:
        GrammarIso(
            bounds=(ProfileBound("DeepInsideOutside", 1, frozenset({1, 2})),)
        ).forward(_flat_rule("NP", "VP", "VP"), registered_language("propositional"))

    message = str(refusal.value)

    assert "DeepInsideOutside" in message
    assert "[3]" in message


def test_a_compilation_under_one_bound_cannot_satisfy_another_bound() -> None:
    declaration = _flat_rule("NP", "VP", "VP")
    language = registered_language("propositional")
    accepting = ProfileBound("ternary", 1, frozenset({3}))
    refusing = ProfileBound("binary", 1, frozenset({1, 2}))

    accepted = GrammarIso(bounds=(accepting,)).forward(declaration, language)

    assert accepted.grammar.profile.ranks == frozenset({3})

    with pytest.raises(ProfileUnsupportedError, match="binary"):
        GrammarIso(bounds=(refusing,)).forward(declaration, language)


def test_lowering_checks_every_consumer_bound() -> None:
    binary = ProfileBound("binary", 1, frozenset({2}))
    unary = ProfileBound("unary", 1, frozenset({1}))

    with pytest.raises(ProfileUnsupportedError, match="unary"):
        GrammarIso(bounds=(binary, unary)).forward(
            _cfg(), registered_language("propositional")
        )


def _axiom(restriction: RuleRestrictionSpec | None) -> PatternRuleSpec:
    return PatternRuleSpec(
        name="axiom",
        premises=(),
        conclusion=_node("S"),
        head_policy="functor",
        restriction=restriction,
    )


_PRODUCTION = PatternRuleSpec(
    name="s",
    premises=(_node("NP"), _node("VP")),
    conclusion=_node("S"),
    head_policy="pass-through",
)


@pytest.mark.parametrize(
    ("rules", "expected"),
    [
        ((_axiom(None),), "unlexicalized"),
        ((_PRODUCTION,), "lexicalized"),
        ((_PRODUCTION, _axiom(None)), "unlexicalized"),
    ],
    ids=["axiom", "production", "mixed"],
)
def test_lowering_derives_lexicalization_from_nullary_rules(
    rules: tuple[PatternRuleSpec, ...],
    expected: str,
) -> None:
    declaration = _cfg().with_(rules=RuleSetSpec(name="axiomatic", rules=rules))
    accepting = ProfileBound("axiom engine", 1, frozenset({0, 2}))
    grammar = (
        GrammarIso(bounds=(accepting,))
        .forward(declaration, registered_language("propositional"))
        .grammar
    )

    assert grammar.profile.lexicalized is (expected == "lexicalized")


def test_a_restricted_axiom_is_refused_since_it_has_no_premise() -> None:
    restricted = _axiom(RuleRestrictionSpec(primary_targets=("S",)))
    declaration = _cfg().with_(rules=RuleSetSpec(name="axiomatic", rules=(restricted,)))
    accepting = ProfileBound("axiom engine", 1, frozenset({0, 2}))

    with pytest.raises(DeclarationError, match="no primary premise"):
        GrammarIso(bounds=(accepting,)).forward(
            declaration, registered_language("propositional")
        )


def test_the_closure_limits_come_from_the_declared_budget() -> None:
    ccg = _ccg()
    fingerprints = {
        GrammarIso()
        .forward(
            ccg.with_(budget=ccg.budget.with_(closure_rounds=rounds)),
            registered_language("propositional"),
        )
        .grammar.fingerprint
        for rounds in (1, 2, 3)
    }

    assert len(fingerprints) == 3

    cfg = _cfg()
    starved = cfg.with_(budget=cfg.budget.with_(closure_nodes=2))

    with pytest.raises(ValueError, match="unreachable"):
        GrammarIso().forward(starved, registered_language("propositional"))


def test_an_undeclared_pattern_sort_is_located_at_the_rule_declaration() -> None:
    source = _ccg()
    rules = RuleSetSpec(
        name="malformed",
        rules=(
            PatternRuleSpec(
                name="malformed",
                premises=(TermPatternSpec(name="X", sort="Nonexistent"),),
                conclusion=TermPatternSpec(name="X", sort="Nonexistent"),
                head_policy="pass-through",
            ),
        ),
    )

    malformed = source.with_(rules=rules)
    with pytest.raises(DeclarationError) as caught:
        GrammarIso().forward(malformed, registered_language("propositional"))

    assert 'unknown sort "Nonexistent"' in str(caught.value)


def test_an_unreadable_goal_is_located_at_its_category_declaration() -> None:
    source = _ccg().with_(goal=RootGoalSpec(categories=("Q",)))

    with pytest.raises(DeclarationError) as caught:
        GrammarIso().forward(source, registered_language("propositional"))

    assert caught.value.path == "grammar.goal.categories"
    assert "operator is not declared" in str(caught.value)


def test_an_unreachable_goal_names_its_closure_and_configuration_path() -> None:
    source = _ccg().with_(goal=RootGoalSpec(categories=("conj",)))

    with pytest.raises(DeclarationError) as caught:
        GrammarIso().forward(source, registered_language("propositional"))

    assert caught.value.path == "grammar.goal.categories"
    assert "targets ('conj',) are unreachable in ClosureReport(" in str(caught.value)


def test_a_lexical_category_deeper_than_the_language_codec_is_located() -> None:
    source = _ccg()
    language = registered_language("propositional")
    bounds = dict(language.bounds)
    bounds[language.category_sort] = SortBoundSpec(
        layout="complete-tree",
        depth=2,
    )

    with pytest.raises(DeclarationError) as caught:
        GrammarIso().forward(source, language.with_(bounds=bounds))

    assert caught.value.path == "language.bounds.Cat.depth"
    assert "occupies 3 levels" in str(caught.value)


@settings(max_examples=4 * SMALL.examples, deadline=None)
@given(grammars(SMALL))
def test_a_generated_grammar_lowers_and_inverts_to_its_declaration(
    declaration: GrammarDecl,
) -> None:
    iso = GrammarIso()
    declared_language = category_language(
        declaration, registered_language("propositional")
    )
    assembly = iso.forward(declaration, declared_language)
    syntax = assembly.syntax
    store = assembly.grammar.store

    assert iso.backward(assembly) == declaration
    assert GrammarIso().forward(declaration, declared_language) == assembly
    assert len(assembly.grammar.rules) >= sum(
        rule.member_tags() == (rule.rule,) for rule in declaration.rules.rules
    )
    lexicon = cast("InlineLexiconSpec", declaration.lexicon)

    assert {
        (entry.form, syntax.parse(store, entry.category)) for entry in lexicon.entries
    } == {
        (form, entry.category)
        for form in assembly.lexicon.forms
        for entry in assembly.lexicon.entries(form)
    }


COERCIONS: Final = r"""
{
  "name": "coercions",
  "rules": [
    {"rule": "application", "view": "/"},
    {"rule": "application", "view": "\\"},
    {
      "rule": "coercions",
      "label": "exact",
      "entries": ["S[to]\\NP => NP\\NP"],
      "matching": {"kind": "identity"}
    },
    {"rule": "coercions", "label": "erased", "entries": ["S[to]\\NP => S/S"]}
  ]
}
"""


def _coerced(declaration: GrammarDecl, form: str) -> dict[str, str]:
    """Return each rule concluding at a one-token parse, with its conclusion."""
    assembly = GrammarIso().forward(declaration, registered_language("propositional"))
    grammar = assembly.grammar
    forest = parse(
        (form,),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=grammar.rules,
        source=assembly.lexicon,
        semiring=Viterbi(),
        scorer=_Unweighted(),
        goal=grammar.goal,
        unary_depth=1,
    )

    return {
        edge.rule_name: assembly.syntax.render(grammar.store, edge.output)
        for key in forest.items_at((0, 1))
        for edge in forest.edges(key)
        if edge.children
    }


def test_a_rule_declared_exact_matches_only_the_featured_premise_it_names() -> None:
    base = _ccg()
    signature = base.signature.with_(
        operators=(
            *base.signature.operators,
            OperatorSpec(name="S[to]", sort="Cat", erases_to="S"),
        )
    )
    declaration = GrammarDecl.model_validate_json(
        base.with_(
            signature=signature,
            rules=RuleSetSpec.model_validate_json(COERCIONS),
            lexicon=InlineLexiconSpec(
                entries=(
                    LexicalEntrySpec(form="to-leave", category="S[to]\\NP"),
                    LexicalEntrySpec(form="left", category="S[dcl]\\NP"),
                    LexicalEntrySpec(form="leave", category="S\\NP"),
                )
            ),
            goal=RootGoalSpec(categories=("S\\NP",)),
            normal_form=NoNormalFormSpec(),
        ).model_dump_json()
    )
    exact = "exact:0:S[to]\\NP=>NP\\NP"
    erased = "erased:0:S[to]\\NP=>S/S"
    rules = (
        GrammarIso()
        .forward(declaration, registered_language("propositional"))
        .grammar.rules
    )

    assert declaration.matching == FeatureErasureMatchingSpec()
    assert rules.matching.name.startswith("ccg-feature-erasure")
    assert rules.packing.name == "identity"
    assert _coerced(declaration, "to-leave") == {exact: "NP\\NP", erased: "S/S"}
    assert _coerced(declaration, "left") == {erased: "S/S"}
    assert _coerced(declaration, "leave") == {erased: "S/S"}


def _composition_document(direction: str) -> GrammarDecl:
    rules = f"""
    {{
      "name": "composition",
      "rules": [
        {{
          "rule": "combinators",
          "composition": [1, 2],
          "crossed_composition": [1, 2],
          "direction": "{direction}"
        }}
      ]
    }}
    """

    return GrammarDecl.model_validate_json(
        _ccg()
        .with_(
            rules=RuleSetSpec.model_validate_json(rules),
            normal_form=NoNormalFormSpec(),
        )
        .model_dump_json()
    )


def test_a_document_declaring_crossing_that_fixes_no_frame_is_refused() -> None:
    with pytest.raises(ValueError, match="same relation twice") as refused:
        GrammarIso().forward(
            _composition_document("unrestricted"), registered_language("propositional")
        )

    for order in (1, 2):
        for slash in ("/", "\\"):
            harmonic = f"composition:{slash}:{order}:harmonic:unrestricted"
            crossed = f"composition:{slash}:{order}:crossed:unrestricted"

            assert (f"{harmonic!r} and {crossed!r} differ only in 'crossed'") in str(
                refused.value
            )


@pytest.mark.parametrize("direction", ["steedman", "kuhlmann"])
def test_a_document_declaring_crossing_that_fixes_a_frame_is_accepted(
    direction: str,
) -> None:
    rules = (
        GrammarIso()
        .forward(_composition_document(direction), registered_language("propositional"))
        .grammar.rules
    )

    assert len(rules) == 10


SIGNATURE: Final = Signature(
    "mixed",
    ("Cat", "Other", "LF"),
    (
        Operator("A", "Cat"),
        Operator("B", "Cat"),
        Operator("thing", "Other"),
        Operator("pair", "Cat", arg_sorts=("Cat", "Cat"), family="pair"),
        Operator("wrap", "Cat", arg_sorts=("Cat",), family="wrap"),
        Operator("box", "Cat", arg_sorts=("Cat",), family="wrap"),
        Operator(
            "lam",
            "LF",
            arg_sorts=("LF",),
            binding=(BindingSite((0,), "LF"),),
            family="binder",
        ),
        Operator("c", "LF"),
        Operator("quote", "Cat", arg_sorts=("LF",), family="quote"),
    ),
    vocabularies=(Vocabulary("Cat", ("k",)),),
    features={"A": ("f",)},
)
STORE: Final = TermStore(SIGNATURE)
IDX: Final = "#const[Cat]"
OP_IDX: Final = f"op_{encode_identifier(IDX)}"
MATCHING: Final = Identity[Term]()
THEORY_X: Final = MVar("X", "Cat")
THEORY_Y: Final = MVar("Y", "Cat")


DECLARED: Final = SignatureSpec(
    name="mixed",
    primitives=("A",),
    constructors=tuple(
        ConstructorSpec(
            name=name,
            kind=kind,
            notation=PrefixConstructorNotationSpec(spelling=name),
        )
        for name, kind in (
            ("pair", ("Cat", "Cat", "Cat")),
            ("wrap", ("Cat", "Cat")),
            ("box", ("Cat", "Cat")),
        )
    ),
    syntax="s-expression",
)
"""A declared signature carrying the mixed signature's category constructors."""
DECLARED_X: Final = TermPatternSpec(name="X", sort="Cat")
DECLARED_Y: Final = TermPatternSpec(name="Y", sort="Cat")


def _declared(*rules: PatternRuleSpec) -> tuple[Rule[Term], ...]:
    """Compile pattern rules over the declared signature."""
    return declared_rules(
        DECLARED, RuleSetSpec(name="mixed-rules", rules=rules)
    ).rule_set.rules


def _pairing() -> Rule[Term]:
    """Compile the rule pairing its two premises."""
    (rule,) = _declared(
        PatternRuleSpec(
            name="pairing",
            premises=(DECLARED_X, DECLARED_Y),
            conclusion=NodePatternSpec(
                operator=FixedOperatorSpec(name="pair"),
                arguments=(DECLARED_X, DECLARED_Y),
            ),
        )
    )

    return rule


def _index(value: int) -> NodePatternSpec:
    """Declare the category constant at one fixed payload."""
    return NodePatternSpec(
        operator=FixedOperatorSpec(name=IDX), payload=FixedPayloadSpec(value=value)
    )


def _rules(*rules: Rule[Term]) -> RuleSet[Term]:
    return RuleSet("mixed-rules", rules, MATCHING, NoNormalForm())


def _checked(*rules: Rule[Term]) -> CheckedTheory:
    return mirror("mixed", SIGNATURE, _rules(*rules), "Cat")


def _constructors(checked: CheckedTheory) -> dict[str, dict[str, JsonValue]]:
    ops: JsonValue = checked.theory.to_dict()["ops"]

    assert isinstance(ops, list)

    declared: dict[str, dict[str, JsonValue]] = {}

    for op in ops:
        assert isinstance(op, dict)

        name = op["name"]

        assert isinstance(name, str)

        declared[name] = op

    return declared


def _app(operation: str, *arguments: dx.Term) -> dx.App:
    return dx.App(operation, arguments)


def _lexical(category: dx.Term) -> dx.Term:
    return _app(LEXICAL_OPERATION, category)


class _Schemaless:
    """A rule offering no proof schemata."""

    @property
    def name(self) -> str:
        return "schemaless"

    @property
    def arity(self) -> int:
        return 1

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset({"forward", "check"})

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        return True

    @property
    def matching(self) -> Matching[Term]:
        return MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        raise NotImplementedError

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return ()

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        del inputs

        return iter(())

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        del inputs, output

        return False

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        del output, known

        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "dishonest"}


class _Dishonest(_Schemaless):
    """A rule claiming complete forward enumeration with a free conclusion index."""

    @property
    @override
    def name(self) -> str:
        return "dishonest"

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        return (((THEORY_X,), THEORY_Y),)


class _Schema(_Schemaless):
    """A rule presenting one proof schema exactly as given."""

    def __init__(
        self,
        name: str,
        premises: tuple[Pattern, ...],
        conclusion: Pattern,
        *,
        complete: bool = True,
    ) -> None:
        self._name = name
        self._schema = (premises, conclusion)
        self._complete = complete

    @property
    @override
    def name(self) -> str:
        return self._name

    @property
    @override
    def arity(self) -> int:
        return len(self._schema[0])

    @property
    @override
    def forward_is_complete(self) -> bool:
        return self._complete

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        return (self._schema,)


class _Nondeterministic(_Schemaless):
    """A rule whose schema quantifies over operators, payloads or contexts."""

    def __init__(self, schema: SchematicSequent) -> None:
        self._schema = schema

    @property
    @override
    def name(self) -> str:
        return "nondeterministic"

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        return (self._schema,)


def test_the_theory_mirrors_each_operator_with_its_arity_sorts_and_binders() -> None:
    declared = _constructors(_checked())
    one_binder = {
        "App": {"op": "successor", "args": [{"App": {"op": "zero", "args": []}}]}
    }

    assert [name for name in declared if name.startswith("op_")] == [
        f"op_{encode_identifier(op.name)}" for op in SIGNATURE.operators
    ]
    assert declared["op_A"] == {"name": "op_A", "inputs": [], "output": "sort_Cat"}
    assert declared["op_thing"]["output"] == "sort_Other"
    assert declared["op_pair"]["inputs"] == [
        ("argument_0", "sort_Cat", "No"),
        ("argument_1", "sort_Cat", "No"),
    ]
    assert declared["op_lam"] == {
        "name": "op_lam",
        "inputs": [
            ("argument_0", {"name": "bound_LF", "args": [one_binder]}, "No"),
        ],
        "output": "sort_LF",
    }
    assert declared[OP_IDX]["inputs"] == [("payload", "payload_constant", "No")]
    assert declared["op_quote"]["inputs"] == [("argument_0", "sort_LF", "No")]
    assert declared["op_A_5b_f_5d_"]["output"] == "sort_Cat"


def test_the_theory_sorts_operators_by_their_declared_positions() -> None:
    builder = _checked().builder
    a = _app("op_A")

    assert builder.infer_sort(_app("op_pair", a, _app("op_B"))) == dx.SortExpr(
        "sort_Cat", ()
    )
    assert builder.infer_sort(
        _app(
            "op_lam",
            _app("bind_LF", _app("successor", _app("zero")), _app("op_c")),
        )
    ) == dx.SortExpr("sort_LF", ())

    with pytest.raises(dx.GADTDeclarationError):
        builder.infer_sort(_app("op_pair", a, _app("op_thing")))

    with pytest.raises(dx.GADTDeclarationError):
        builder.infer_sort(_app("op_lam", _app("op_c")))

    with pytest.raises(dx.GADTDeclarationError):
        builder.infer_sort(_app(OP_IDX))


def test_a_rule_schema_becomes_a_constructor_whose_premises_fix_its_index() -> None:
    checked = _checked(_pairing())
    (record,) = checked.constructors

    assert record == DerivationConstructor(
        "pairing",
        "rule_0_pairing",
        ("index_X", "index_Y"),
        (),
        2,
        (
            '{"binders":0,"kind":"variable","name":"index_X","sort":"Cat"}',
            '{"binders":0,"kind":"variable","name":"index_Y","sort":"Cat"}',
        ),
        (
            '{"arguments":['
            '{"binders":0,"kind":"variable","name":"index_X","sort":"Cat"},'
            '{"binders":0,"kind":"variable","name":"index_Y","sort":"Cat"}'
            '],"binders":0,"kind":"node","operator":"pair","payload":null,'
            '"sort":"Cat"}'
        ),
    )

    derivation = _app(
        record.constructor, _lexical(_app("op_A")), _lexical(_app("op_B"))
    )

    assert checked.builder.infer_sort(derivation) == dx.SortExpr(
        DERIVATION_FAMILY, (_app("op_pair", _app("op_A"), _app("op_B")),)
    )


def test_a_shared_premise_variable_must_agree_across_premises() -> None:
    checked = _checked(
        *_declared(
            PatternRuleSpec(
                name="same", premises=(DECLARED_X, DECLARED_X), conclusion=DECLARED_X
            )
        )
    )
    (record,) = checked.constructors

    assert record.inferred == ("index_X",)

    a = _lexical(_app("op_A"))

    assert checked.builder.infer_sort(_app(record.constructor, a, a)) == (
        dx.SortExpr(DERIVATION_FAMILY, (_app("op_A"),))
    )

    with pytest.raises(dx.GADTDeclarationError, match="index mismatch"):
        checked.builder.infer_sort(_app(record.constructor, a, _lexical(_app("op_B"))))


def test_a_check_only_rule_takes_its_free_result_index_explicitly() -> None:
    checked = _checked(
        _Schema("unlike", (Node("A"), THEORY_Y), THEORY_X, complete=False)
    )
    (record,) = checked.constructors

    assert record.inferred == ("index_Y",)
    assert record.declared == ("index_X",)
    assert checked.builder.infer_sort(
        _app(
            record.constructor,
            _app("op_B"),
            _lexical(_app("op_A")),
            _lexical(_app("op_A")),
        )
    ) == dx.SortExpr(DERIVATION_FAMILY, (_app("op_B"),))


def test_a_rule_claiming_completeness_with_a_free_index_is_malformed() -> None:
    with pytest.raises(DerivationIndexError, match="index_Y") as caught:
        _checked(_Dishonest())

    assert caught.value.rule == "dishonest"


def test_a_rule_without_schemata_cannot_be_mirrored() -> None:
    with pytest.raises(DerivationIndexError, match="no deterministic schema") as caught:
        _checked(_Schemaless())

    assert caught.value.rule == "schemaless"


@pytest.mark.parametrize(
    "premise",
    [
        NodePatternSpec(
            operator=VariableOperatorSpec(name="f"), arguments=(DECLARED_X,)
        ),
        NodePatternSpec(
            operator=FixedOperatorSpec(name=IDX),
            payload=VariablePayloadSpec(name="n"),
        ),
        SpinePatternSpec(
            name="C",
            core=ConstantPatternSpec(name="A"),
            operators=("wrap",),
            max_order=None,
        ),
    ],
    ids=["operator variable", "payload variable", "unbounded spine context"],
)
def test_a_pattern_rule_over_a_nondeterministic_pattern_offers_no_schema(
    premise: PatternSpec,
) -> None:
    (rule,) = _declared(
        PatternRuleSpec(name="open", premises=(premise,), conclusion=DECLARED_X)
    )

    assert cast("PatternRule", rule).proof_schemata == ()

    with pytest.raises(DerivationIndexError, match="no deterministic schema"):
        _checked(rule)


def test_a_bounded_spine_rule_compiles_one_constructor_per_context() -> None:
    (lift,) = _declared(
        PatternRuleSpec(
            name="lift",
            premises=(
                SpinePatternSpec(
                    name="C",
                    core=DECLARED_X,
                    operators=("wrap", "box"),
                    min_order=1,
                    max_order=3,
                ),
            ),
            conclusion=SpinePatternSpec(
                name="C",
                core=NodePatternSpec(
                    operator=FixedOperatorSpec(name="pair"),
                    arguments=(DECLARED_X, DECLARED_X),
                ),
                operators=("wrap", "box"),
                min_order=1,
                max_order=3,
            ),
        )
    )
    checked = _checked(lift)
    records = checked.constructors

    assert len(records) == 2 + 4 + 8
    assert {record.rule for record in records} == {"lift"}
    assert all(record.declared == () for record in records)
    assert len(set(_constructors(checked)) & {r.constructor for r in records}) == 14
    assert checked.elaborated == ()


@pytest.mark.parametrize(
    ("premise", "fault"),
    [
        (Node(PVar("f"), (THEORY_X,)), "operator variable 'f'"),
        (
            Spine("C", Node("A"), ("wrap",), slot=0, min_order=0, max_order=2),
            "spine context 'C'",
        ),
    ],
    ids=["operator variable", "spine context"],
)
def test_a_plugin_schema_quantifying_over_operators_is_malformed(
    premise: Pattern, fault: str
) -> None:
    with pytest.raises(DerivationIndexError, match=fault):
        _checked(_Nondeterministic(((premise,), THEORY_X)))


@pytest.mark.parametrize(
    ("conclusion", "fault"),
    [
        (Node("thing"), "sort 'Other'"),
        (Node("pair", (THEORY_X,)), "arity 1"),
        (Node("A", payload=3), "takes no payload"),
        (Node(IDX), "requires a payload"),
    ],
)
def test_a_malformed_declared_result_index_fails(conclusion: Node, fault: str) -> None:
    with pytest.raises(DerivationIndexError, match=fault) as caught:
        _checked(_Schema("bad", (THEORY_X,), conclusion))

    assert caught.value.rule == "bad"


def test_a_variable_at_two_sorts_is_malformed() -> None:
    shared = MVar("Z", "Cat")

    with pytest.raises(DerivationIndexError, match="stands at sorts"):
        _checked(
            _Schema("crossed", (Node("wrap", (shared,)),), Node("quote", (shared,)))
        )


def test_a_fixed_payload_becomes_one_literal_per_value() -> None:
    checked = _checked(
        *_declared(
            PatternRuleSpec(name="three", premises=(_index(3),), conclusion=_index(3)),
            PatternRuleSpec(name="minus", premises=(_index(-1),), conclusion=_index(3)),
        )
    )
    three, minus = checked.constructors
    literal = _app(OP_IDX, _app("payload_constant__33_"))

    assert three.inferred == ()
    assert minus.inferred == ()
    assert checked.builder.infer_sort(
        _app(three.constructor, _lexical(literal))
    ) == dx.SortExpr(DERIVATION_FAMILY, (literal,))
    assert checked.builder.infer_sort(
        _app(
            minus.constructor,
            _lexical(_app(OP_IDX, _app("payload_constant__2d_1"))),
        )
    ) == dx.SortExpr(DERIVATION_FAMILY, (literal,))


def test_a_plugin_schema_may_carry_a_payload_variable_as_an_index() -> None:
    schema = ((Node(IDX, payload=PVar("n")),), Node(IDX, payload=PVar("n")))
    checked = _checked(
        _Nondeterministic(schema),
        *_declared(
            PatternRuleSpec(name="three", premises=(_index(3),), conclusion=_index(3))
        ),
    )
    record, _ = checked.constructors
    literal = _app(OP_IDX, _app("payload_constant__33_"))

    assert record.inferred == ("payload_n",)
    assert checked.builder.infer_sort(
        _app(record.constructor, _lexical(literal))
    ) == dx.SortExpr(DERIVATION_FAMILY, (literal,))


def test_theory_bytes_are_canonical_and_name_the_route() -> None:
    checked = _checked(_pairing())
    encoded = theory_bytes(
        checked.theory, source_version="decl-1", target_version="native-1"
    )
    again = _checked(_pairing())

    assert encoded == theory_bytes(
        again.theory, source_version="decl-1", target_version="native-1"
    )
    assert encoded != theory_bytes(
        checked.theory, source_version="decl-2", target_version="native-1"
    )
    assert encoded != theory_bytes(
        checked.theory, source_version="decl-1", target_version="native-2"
    )
    assert b'"route":{"source":"decl-1","target":"native-1"}' in encoded
    assert encoded.isascii()

    renamed = mirror(
        "other",
        SIGNATURE,
        _rules(_pairing()),
        "Cat",
    )

    assert (
        theory_bytes(renamed.theory, source_version="decl-1", target_version="native-1")
        != encoded
    )


def test_a_checked_theory_is_immutable_and_copies_to_itself() -> None:
    checked = _checked()

    assert type(checked.theory) is panproto.Theory
    assert copy(checked) is checked
    assert deepcopy(checked) is checked

    with pytest.raises(AttributeError, match="immutable"):
        checked.__setattr__("_operators", ())

    with pytest.raises(AttributeError, match="immutable"):
        checked.__delattr__("_operators")


class _Elaborated(_Schemaless):
    """A rule that elaborates the proof term of each step it licenses."""

    @property
    @override
    def name(self) -> str:
        return "elaborated"

    @property
    def witness_is_unique(self) -> bool:
        """Report that a licensed step would have one witness."""
        return True

    def witnesses(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> tuple[ProofWitness, ...]:
        del inputs, output

        return ()


SLASH_SIGNATURE: Final = Signature(
    "slashes",
    ("Cat",),
    (
        Operator("x", "Cat"),
        Operator("y", "Cat"),
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
    ),
)
SLASH_GRAMMAR: Final = LawGrammar("direct")
"""The compiled slash pair whose rules the slash signature's theory mirrors."""
FUNCTORS, _ = SLASH_GRAMMAR.functor("forward")
SLASH_ATOMS: Final = {name: SLASH_GRAMMAR.atom(name) for name in ("x", "y")}


def _composition_rules() -> tuple[Rule[Term], ...]:
    """Compile application and composition at every finite order over one slash."""
    return SLASH_GRAMMAR.rules(
        (
            ApplicationSpec(view="fwd"),
            CompositionSpec(view="fwd", order=None, direction="unrestricted"),
        )
    )


def _slash_spine(base: Term, frames: tuple[str, ...]) -> Term:
    """Wrap a category in a result spine whose frames run shallowest first."""
    built = base

    for slash in reversed(frames):
        built = FUNCTORS.make(slash, built, SLASH_ATOMS["y"])

    return built


def test_a_rule_elaborating_its_steps_takes_no_constructor() -> None:
    checked = _checked(
        _pairing(),
        _Elaborated(),
    )
    (record,) = checked.constructors

    assert record.rule == "pairing"
    assert checked.elaborated == ("elaborated",)


def test_a_grammar_declaring_every_finite_order_compiles_a_derivation_theory() -> None:
    application, schema = _composition_rules()
    rules = RuleSet(
        "composition-rules", (application, schema), application.matching, NoNormalForm()
    )
    checked = mirror("composing", SLASH_SIGNATURE, rules, "Cat")
    (record,) = checked.constructors

    assert record.rule == application.name == "application:fwd"
    assert record.inferred == ("index_X", "index_Y")
    assert (
        checked.elaborated
        == (schema.name,)
        == ("composition:fwd:any:harmonic:unrestricted",)
    )

    derivation = _app(
        record.constructor,
        _lexical(_app("op_fwd", _app("op_x"), _app("op_y"))),
        _lexical(_app("op_y")),
    )

    assert checked.builder.infer_sort(derivation) == dx.SortExpr(
        DERIVATION_FAMILY, (_app("op_x"),)
    )


def test_the_rule_the_theory_leaves_unpresented_still_witnesses_a_step() -> None:
    schema = cast("GeneralizedComposition", _composition_rules()[1])
    frames = ("bwd", "fwd", "bwd")
    principal = FUNCTORS.make("fwd", SLASH_ATOMS["x"], SLASH_ATOMS["y"])
    premises = (principal, _slash_spine(SLASH_ATOMS["y"], frames))
    conclusion = _slash_spine(SLASH_ATOMS["x"], frames)
    witnesses = schema.witnesses(premises, conclusion)

    assert schema.proof_schemata == ()
    assert schema.witness_is_unique is True
    assert len(witnesses) == 1
    (witness,) = witnesses
    assert witness.order == len(frames)
    assert witness.frames == frames
    assert witness.principal == 0
    assert witness.crossed is True


def test_an_installed_normal_form_crosses_the_native_boundary(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    with _isolated_variants(NormalFormSpec):

        class BoundedNormalFormSpec(NormalFormSpec, extra="forbid"):
            kind: Literal["bounded-fixture"] = "bounded-fixture"
            bound: int = 2

        plugin = NormalFormPlugin(
            spec=BoundedNormalFormSpec, build=lambda _declaration: NoNormalForm()
        )
        install(
            monkeypatch,
            "normal_form",
            (
                *Registry[object]("normal_form").entries(),
                *entries("normal_form", {"bounded-fixture": plugin}),
            ),
        )
        grammar = _ccg().with_(normal_form=BoundedNormalFormSpec(bound=3))
        assembly = GrammarIso().forward(grammar, registered_language("propositional"))

    assert assembly.grammar.name == grammar.name
