from typing import Final

import didactic.api as dx
import pytest

from lofi.formalism import TheoryArtifact
from lofi.spec import (
    ArchivedLanguage,
    ConstantPatternSpec,
    IndexedLanguage,
    PatternRuleSpec,
)
from lofi.spec._rule import RuleSetSpec
from lofi.spec.grammar import (
    GRAMMAR_SOURCE_VERSION,
    GRAMMAR_TARGET_VERSION,
    ArchiveMismatchError,
    BudgetSpec,
    CapacityEvidenceSpec,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    RootGoalSpec,
    SignatureSpec,
    archive_bytes,
    canonical_bytes,
    rebuild_grammar,
)
from lofi.spec.grammar._lowering import _native_plan, _native_theory_bytes
from lofi.spec.language import LanguageDecl, registered_language

ROUTE: Final = dx.LoweringRoute(GRAMMAR_SOURCE_VERSION, GRAMMAR_TARGET_VERSION)


LANGUAGE: Final = registered_language("propositional")


def constant(name: str) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


def declaration(
    *,
    operators: tuple[str, ...] = ("S", "NP", "VP"),
    rules: tuple[PatternRuleSpec, ...] = (),
    entries: tuple[LexicalEntrySpec, ...] = (),
) -> GrammarDecl:
    sentence = PatternRuleSpec(
        name="s",
        premises=(constant("NP"), constant("VP")),
        conclusion=constant("S"),
        head_policy="pass-through",
    )

    return GrammarDecl(
        name="toy-cfg",
        signature=SignatureSpec(
            name="cfg", primitives=tuple(operators), syntax="s-expression"
        ),
        rules=RuleSetSpec(name="productions", rules=(sentence, *rules)),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="john", category='("#const[Cat]" "NP")'),
                LexicalEntrySpec(form="sleeps", category='("#const[Cat]" "VP")'),
                *entries,
            )
        ),
        goal=RootGoalSpec(categories=('("#const[Cat]" "S")',)),
        budget=BudgetSpec(unary_rounds=0, closure_rounds=1, closure_nodes=16),
    )


def assemble(source: GrammarDecl, language: LanguageDecl = LANGUAGE) -> GrammarAssembly:
    return GrammarIso().forward(source, language)


def archived(source: GrammarDecl, route: dx.LoweringRoute = ROUTE) -> ArchivedLanguage:
    assembly = assemble(source)
    artifact = assembly.grammar

    return ArchivedLanguage(
        archive_bytes(assembly),
        IndexedLanguage(
            artifact.theory_version,
            artifact.theory_specification,
            artifact.theory_digest,
            route,
        ),
    )


def native_theory(
    source: GrammarDecl, language: LanguageDecl = LANGUAGE, /
) -> TheoryArtifact:
    """Return the authoritative theory record without assembling consumers."""
    plan = _native_plan(
        canonical_bytes(source),
        language.compile_envelope.canonical_declaration,
        name=source.name,
    )

    return TheoryArtifact.from_specification(
        f"{GRAMMAR_SOURCE_VERSION}->{GRAMMAR_TARGET_VERSION}",
        _native_theory_bytes(
            plan,
            source_version=GRAMMAR_SOURCE_VERSION,
            target_version=GRAMMAR_TARGET_VERSION,
        ),
    )


DETERMINED: Final = declaration(
    operators=("S", "NP", "VP", "D"),
    rules=(
        PatternRuleSpec(
            name="np",
            premises=(constant("D"), constant("NP")),
            conclusion=constant("NP"),
            head_policy="pass-through",
        ),
    ),
    entries=(LexicalEntrySpec(form="the", category='("#const[Cat]" "D")'),),
)


def test_an_archived_declaration_rebuilds_its_native_grammar() -> None:
    assembly = rebuild_grammar(archived(DETERMINED))

    assert assembly.grammar.name == "toy-cfg"
    assert {rule.name for rule in assembly.grammar.rules} == {"s", "np"}
    assert set(assembly.lexicon.forms) == {"john", "sleeps", "the"}
    assert assembly.source_version == GRAMMAR_SOURCE_VERSION
    assert assembly.target_version == GRAMMAR_TARGET_VERSION


def test_the_rebuilt_grammar_carries_the_archived_theory() -> None:
    language = archived(DETERMINED)
    grammar = rebuild_grammar(language).grammar

    assert grammar.theory_digest == language.theory.digest
    assert grammar.theory_specification == language.theory.specification
    assert grammar.theory_version == language.theory.version


def test_a_declaration_that_lowers_to_another_theory_is_refused() -> None:
    honest = archived(DETERMINED)
    mismatched = ArchivedLanguage(archive_bytes(assemble(declaration())), honest.theory)

    with pytest.raises(ArchiveMismatchError, match="lowers to theory") as refused:
        rebuild_grammar(mismatched)

    assert refused.value.archived == honest.theory.digest
    assert refused.value.rebuilt != honest.theory.digest


def test_an_archived_route_the_lowerer_does_not_implement_is_refused() -> None:
    language = archived(
        DETERMINED, dx.LoweringRoute(GRAMMAR_SOURCE_VERSION, "native-9")
    )

    with pytest.raises(dx.UnsupportedLoweringRouteError):
        rebuild_grammar(language)


def test_an_unsupported_route_is_refused_before_malformed_declaration_bytes() -> None:
    unsupported = archived(
        DETERMINED, dx.LoweringRoute(GRAMMAR_SOURCE_VERSION, "native-9")
    )
    malformed = ArchivedLanguage(b"not-json", unsupported.theory)

    with pytest.raises(dx.UnsupportedLoweringRouteError):
        rebuild_grammar(malformed)


def test_two_rebuilds_create_independent_unified_language_stores() -> None:
    first = rebuild_grammar(archived(DETERMINED))
    second = rebuild_grammar(archived(DETERMINED))

    assert first.store is first.language.store
    assert second.store is second.language.store
    assert first.grammar.signature is not second.grammar.signature


def test_adding_an_operator_changes_the_archived_theory() -> None:
    plain = archived(declaration())
    determined = archived(DETERMINED)

    assert plain.theory.digest != determined.theory.digest
    assert plain.theory.version == determined.theory.version


def test_nonsemantic_grammar_fields_preserve_the_native_theory_identity() -> None:
    source = declaration()
    baseline = native_theory(source)
    evidence = CapacityEvidenceSpec(
        rule_family="pure_ccg",
        bounded_degree=True,
        prefix_closed=True,
        application_target_restrictions=False,
    )
    variants = (
        source.with_(
            lexicon=InlineLexiconSpec(
                entries=(
                    LexicalEntrySpec(form="john", category='("#const[Cat]" "NP")'),
                    LexicalEntrySpec(form="sleeps", category='("#const[Cat]" "VP")'),
                    LexicalEntrySpec(form="again", category='("#const[Cat]" "VP")'),
                )
            )
        ),
        source.with_(budget=source.budget.with_(beam=4)),
        source.with_(goal=RootGoalSpec(categories=('("#const[Cat]" "NP")',))),
        source.with_(interpretations=()),
        source.with_(max_hypotheses=1),
        source.with_(capacity_evidence=evidence),
    )

    for variant in variants:
        theory = native_theory(variant)
        assert theory.specification == baseline.specification
        assert theory.digest == baseline.digest


def test_a_rule_change_changes_the_native_theory_identity() -> None:
    source = declaration()
    unary = PatternRuleSpec(
        name="np-copy",
        premises=(constant("NP"),),
        conclusion=constant("NP"),
        head_policy="pass-through",
    )

    assert (
        native_theory(source).digest
        != native_theory(declaration(rules=(unary,))).digest
    )


def test_nonsemantic_language_capacity_preserves_the_native_theory_identity() -> None:
    source = declaration()
    baseline = native_theory(source)
    expanded = LANGUAGE.with_(store_nodes=LANGUAGE.store_nodes + 1)

    changed = native_theory(source, expanded)

    assert changed.specification == baseline.specification
    assert changed.digest == baseline.digest
