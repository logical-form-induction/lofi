import ast
from functools import partial
from pathlib import Path
from typing import TYPE_CHECKING, Final

import pytest
from tests.support.term_reference import infix_syntax

from lofi.grammars.ccg import CCGAssembly, FunctorView
from lofi.judge import HOLE, Conjunctions, NoNormalForm
from lofi.spec import (
    AbsorptionSpec,
    ClausalAdjunctSpec,
    CombinatorSetSpec,
    CoordinationSpec,
    HeadSelection,
    PatternRuleSpec,
    RuleSetSpec,
    RuleSpec,
)
from lofi.spec.datasets import ccgbank as ccgbank_spec
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.language import registered_language
from lofi.terms import (
    FeatureErasure,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
    _compile_grammar,
)

if TYPE_CHECKING:
    from lofi.judge import Rule


SIGNATURE: Final = Signature(
    "synthetic-ccgbank-exceptions",
    ("Cat",),
    (
        Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
        Operator("\\", "Cat", arg_sorts=("Cat", "Cat"), family="slash"),
    ),
    vocabularies=(
        Vocabulary(
            "Cat",
            (
                "N",
                "N[conj]",
                "NP",
                "NP[conj]",
                "S",
                "S[dcl]",
                "S[ng]",
                "conj",
                ".",
                ",",
                ":",
                ";",
                "LRB",
                "RRB",
            ),
        ),
    ),
    constant_cores={
        "Cat": {
            "N[conj]": "N",
            "NP[conj]": "NP",
            "S[dcl]": "S",
            "S[ng]": "S",
        }
    },
)
STORE: Final = TermStore(SIGNATURE)
SYNTAX: Final = infix_syntax(SIGNATURE, ("/", "/"), ("\\", "\\"))


def _category(source: str) -> Term:
    """Parse one synthetic category."""
    return SYNTAX.parse(STORE, source)


N: Final = _category("N")
N_CONJ: Final = _category("N[conj]")
NP: Final = _category("NP")
NP_CONJ: Final = _category("NP[conj]")
CONJ: Final = _category("conj")
PERIOD: Final = _category(".")
COMMA: Final = _category(",")
COLON: Final = _category(":")
SEMICOLON: Final = _category(";")
LRB: Final = _category("LRB")
RRB: Final = _category("RRB")
PUNCTUATION: Final = (PERIOD, COMMA, COLON, SEMICOLON, LRB, RRB)
MARKERS: Final = (CONJ, *PUNCTUATION)
CONJUNCTIONS: Final = Conjunctions(
    "synthetic-ccgbank-exceptions",
    MARKERS,
    ((N, N_CONJ), (NP, NP_CONJ)),
)
FUNCTORS: Final = FunctorView(STORE, "/", "\\", result=0, argument=1)
GRAMMAR: Final = ccgbank_grammar()
RULE_SET: Final = GRAMMAR.rules
RULES: Final = tuple(
    rule for rule in RULE_SET.rules if not isinstance(rule, CombinatorSetSpec)
)
DECLARED_RULES: Final = RuleSetSpec(name="ccgbank-exceptional", rules=RULES)
PLANNED_GRAMMAR: Final = GRAMMAR.with_(rules=DECLARED_RULES)
LANGUAGE: Final = registered_language("propositional")
PLAN: Final = _compile_grammar(
    LANGUAGE.compile_envelope.canonical_declaration,
    PLANNED_GRAMMAR.compile_envelope.canonical_declaration,
)
ASSEMBLY: Final = CCGAssembly(
    STORE,
    SYNTAX,
    partial(SYNTAX.parse, STORE),
    {"/": (FUNCTORS, "/"), "\\": (FUNCTORS, "\\")},
    FeatureErasure(),
    CONJUNCTIONS,
    NoNormalForm[Term](),
    category_sort="Cat",
    categories=PLAN.runtime_categories,
)
BUILT: Final = PLANNED_GRAMMAR.prepared_plugins.rules.adopt(
    tuple(PLAN.rules),
    ASSEMBLY,
)

EXPECTED_NAMES: Final = (
    "n-to-np",
    "punctuation-left",
    "punctuation-right",
    "clausal-adjunct-backward-backward",
    "clausal-adjunct-backward-forward",
    "clausal-adjunct-forward-backward",
    "clausal-adjunct-forward-forward",
    "appositive-np-comma-forward",
    "appositive-comma-np-backward",
    "appositive-comma-np-verbal",
    "coordination-mark-conj",
    "coordination-mark-period",
    "coordination-mark-comma",
    "coordination-mark-colon",
    "coordination-mark-semicolon",
    "coordination-mark-lrb",
    "coordination-mark-rrb",
    "coordination-merge",
    "coordination-unlike",
    "compound-conj-n",
)


def _rule(name: str) -> Rule[Term]:
    """Return one built exceptional rule by name."""
    return next(rule for rule in BUILT.rule_set if rule.name == name)


def test_the_ordered_inventory_has_twenty_unique_named_declarations() -> None:
    assert RULE_SET.name == "ccgbank"
    assert tuple(spec.name for spec in RULES) == EXPECTED_NAMES
    assert len(set(EXPECTED_NAMES)) == len(EXPECTED_NAMES) == 20
    assert sum(isinstance(spec, PatternRuleSpec) for spec in RULES) == 5
    assert sum(isinstance(spec, AbsorptionSpec) for spec in RULES) == 2
    assert sum(isinstance(spec, ClausalAdjunctSpec) for spec in RULES) == 4
    assert sum(isinstance(spec, CoordinationSpec) for spec in RULES) == 9


def test_every_declaration_and_the_inventory_round_trip_through_json() -> None:
    restored_rules = tuple(
        RuleSpec.model_validate_json(spec.model_dump_json()) for spec in RULE_SET.rules
    )
    restored_set = RuleSetSpec.model_validate_json(RULE_SET.model_dump_json())

    assert restored_rules == RULE_SET.rules
    assert restored_set == RULE_SET
    assert restored_set.rules == restored_rules


def test_every_runtime_rule_uses_the_assembly_matching_identity() -> None:
    rules = BUILT.rule_set

    assert tuple(rule.name for rule in rules) == EXPECTED_NAMES
    assert rules.matching is ASSEMBLY.matching
    assert all(rule.matching is ASSEMBLY.matching for rule in rules)
    assert tuple(BUILT.head_policies.values()) == (
        "pass-through",
        "non-absorbing",
        "non-absorbing",
        "pass-through",
        "pass-through",
        "pass-through",
        "pass-through",
        "left-conjunct",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "non-absorbing",
        "left-conjunct",
        "non-absorbing",
        "non-absorbing",
    )


def test_every_declaration_derives_its_physical_head_selection() -> None:
    assert dict(BUILT.head_selections) == dict(
        zip(
            EXPECTED_NAMES,
            (
                HeadSelection("pass-through", 0),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 0),
                HeadSelection("pass-through", 0),
                HeadSelection("pass-through", 0),
                HeadSelection("pass-through", 0),
                HeadSelection("pass-through", 0),
                HeadSelection("left-conjunct", 0),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
                HeadSelection("left-conjunct", 0),
                HeadSelection("non-absorbing", 1),
                HeadSelection("non-absorbing", 1),
            ),
            strict=True,
        )
    )


def test_unary_and_appositive_patterns_preserve_exact_category_equations() -> None:
    assert tuple(_rule("n-to-np").apply((N,))) == (NP,)
    assert tuple(_rule("compound-conj-n").apply((CONJ, N))) == (N,)
    assert tuple(_rule("compound-conj-n").apply((COMMA, N))) == ()
    assert tuple(_rule("appositive-np-comma-forward").apply((NP, COMMA))) == (
        _category("S/S"),
    )
    assert tuple(_rule("appositive-comma-np-backward").apply((COMMA, NP))) == (
        _category("S\\S"),
    )
    assert tuple(_rule("appositive-comma-np-verbal").apply((COMMA, NP))) == (
        _category("(S\\NP)\\(S\\NP)"),
    )


@pytest.mark.parametrize("punctuation", PUNCTUATION)
def test_punctuation_absorption_keeps_the_nonpunctuation_category(
    punctuation: Term,
) -> None:
    left = _rule("punctuation-left")
    right = _rule("punctuation-right")

    assert tuple(left.apply((punctuation, NP))) == (NP,)
    assert tuple(right.apply((NP, punctuation))) == (NP,)
    assert tuple(left.apply((punctuation, COMMA))) == ()
    assert tuple(right.apply((COMMA, punctuation))) == ()


@pytest.mark.parametrize(
    ("name", "source_slash", "target_slash"),
    [
        ("clausal-adjunct-backward-backward", "\\", "\\"),
        ("clausal-adjunct-backward-forward", "\\", "/"),
        ("clausal-adjunct-forward-backward", "/", "\\"),
        ("clausal-adjunct-forward-forward", "/", "/"),
    ],
)
def test_clausal_adjuncts_cover_every_source_and_target_slash_pair(
    name: str,
    source_slash: str,
    target_slash: str,
) -> None:
    rule = _rule(name)
    dcl_source = _category(f"S[dcl]{source_slash}NP")
    ng_source = _category(f"S[ng]{source_slash}NP")
    unfeatured_source = _category(f"S{source_slash}NP")
    unrelated_source = _category(f"N[conj]{source_slash}NP")
    output = _category(f"NP{target_slash}NP")

    assert rule.admits((dcl_source,), output)
    assert rule.admits((ng_source,), output)
    assert not rule.admits((unfeatured_source,), output)
    assert not rule.admits((unrelated_source,), output)
    assert tuple(rule.apply((dcl_source,))) == ()


@pytest.mark.parametrize(
    ("name", "marker"),
    [
        ("coordination-mark-conj", CONJ),
        ("coordination-mark-period", PERIOD),
        ("coordination-mark-comma", COMMA),
        ("coordination-mark-colon", COLON),
        ("coordination-mark-semicolon", SEMICOLON),
        ("coordination-mark-lrb", LRB),
        ("coordination-mark-rrb", RRB),
    ],
)
def test_each_coordination_marker_is_fixed_in_its_own_declaration(
    name: str,
    marker: Term,
) -> None:
    rule = _rule(name)
    other = COMMA if marker is CONJ else CONJ

    assert tuple(rule.apply((marker, NP))) == (NP_CONJ,)
    assert tuple(rule.apply((other, NP))) == ()


def test_coordination_merge_and_unlike_relations_are_distinct() -> None:
    merge = _rule("coordination-merge")
    unlike = _rule("coordination-unlike")

    assert tuple(merge.apply((NP, NP_CONJ))) == (NP,)
    assert unlike.directions == frozenset({"check", "residual"})
    assert not unlike.forward_is_complete
    assert unlike.admits((CONJ, N), NP_CONJ)
    assert not unlike.admits((COMMA, N), NP_CONJ)
    assert tuple(unlike.apply((CONJ, N))) == ()
    assert tuple(unlike.residuate(NP_CONJ, (HOLE, N))) == (CONJ,)


def test_the_boundary_module_defines_no_rule_inventory_or_engine_import() -> None:
    path = Path(ccgbank_spec.__file__)
    tree = ast.parse(path.read_text(encoding="utf-8"))
    imports = {
        alias.name
        for node in ast.walk(tree)
        if isinstance(node, ast.Import)
        for alias in node.names
    } | {
        node.module or "" for node in ast.walk(tree) if isinstance(node, ast.ImportFrom)
    }

    assigned = {
        node.id
        for node in ast.walk(tree)
        if isinstance(node, ast.Name) and isinstance(node.ctx, ast.Store)
    }

    assert not any("EXCEPTIONAL" in name for name in assigned)
    assert all(not name.startswith(("lofi.judge", "lofi.deduce")) for name in imports)
