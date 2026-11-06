"""Tests for the declared interpretation function."""

from typing import TYPE_CHECKING, Final

import pytest
from tests.support.derivations import axiom, step
from tests.unit.spec.grammar.test__signature import slash_views, slashes
from tests.verdicts import proved

from lofi.data.synthetic import (
    EntailmentCertificate,
    IllTypedFormError,
    InterpretationError,
    InterpretationFunction,
    LambdaCalculus,
    SymbolInventory,
    UninterpretableDerivationError,
    interpretation_function,
)
from lofi.interpret import MeaningSelection, StringReading, read
from lofi.judge import ProofSearch
from lofi.lexicon import Entry
from lofi.lf import (
    LogicFragment,
    Proved,
    ProverBound,
    Refuted,
    TruthTableOracle,
)
from lofi.lf.provers import ProofBudget, ProofService
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    MeaningInterpretationSpec,
    NamedInterpretationSpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
    StringInterpretationSpec,
    default_interpretations,
)
from lofi.spec.language import (
    ConstantSpec,
    LanguageDecl,
    LanguageIso,
    registered_language,
)
from lofi.terms import Normalization
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from lofi.deduce import Derivation
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term

type Interpretation = InterpretationFunction

LEXICON: Final[tuple[tuple[str, str, str], ...]] = (
    ("john", "NP", "john"),
    ("mary", "NP", "mary"),
    ("sleeps", "S\\NP", "\\x. sleep(x)"),
    ("loves", "(S\\NP)/NP", "\\y. \\x. love(x)(y)"),
    ("and", "(S\\S)/S", "\\q. \\p. p & q"),
    ("not", "S/S", "\\p. ~p"),
)
TYPES: Final[Mapping[str, str]] = {"S": "t", "NP": "e", "conj": "e"}
INVENTORY: Final = SymbolInventory(
    {
        "individual": {"john": 0, "mary": 0},
        "predicate": {"sleep": 1, "love": 2},
    }
)
BOUND: Final = ProverBound(
    "truth-table",
    LogicFragment(
        order="first",
        connectives=("and", "or", "if", "iff", "not"),
    ),
)
APPLICATION: Final = "application:/"
BACKWARD: Final = "application:\\"
COMPOSITION: Final = "composition:/:1:harmonic:unrestricted"
UNBOUNDED_COMPOSITION: Final = "composition:/:any:harmonic:unrestricted"
RAISING: Final = "type-raising:/:targets=S"
JUXTAPOSITION: Final = "juxtaposition"


def _grammar_declaration(
    entries: Sequence[tuple[str, str]],
    *,
    juxtaposition: bool = False,
    composition_order: int | None = 1,
    interpretations: tuple[NamedInterpretationSpec, ...] | None = None,
) -> GrammarDecl:
    rules = [
        ApplicationSpec(view="/"),
        ApplicationSpec(view="\\"),
        CompositionSpec(direction="unrestricted", order=composition_order, view="/"),
        TypeRaisingSpec(view="/", targets=("S",), arguments=None),
    ]

    if juxtaposition:
        rules.append(
            PatternRuleSpec(
                name=JUXTAPOSITION,
                premises=(
                    ConstantPatternSpec(name="NP"),
                    ConstantPatternSpec(name="NP"),
                ),
                conclusion=ConstantPatternSpec(name="S"),
            )
        )

    return GrammarDecl(
        name="interpreted",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "conj"),
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(name="interpreted", rules=tuple(rules)),
        interpretations=(
            default_interpretations() if interpretations is None else interpretations
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, category in entries
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )


def _language_declaration(
    inventory: SymbolInventory = INVENTORY,
) -> LanguageDecl:
    declared = registered_language("propositional")
    constants = (
        *(
            ConstantSpec(name=name, type="e", classes=("individual",))
            for name in inventory.in_role("individual")
        ),
        *(
            ConstantSpec(
                name=name,
                type=" -> ".join((*("e" for _ in range(arity)), "t")),
                classes=("predicate",),
            )
            for name, arity in inventory.in_role("predicate").items()
        ),
        *(
            ConstantSpec(
                name=name,
                type=" -> ".join((*("e" for _ in range(arity)), "e")),
                classes=("function",),
            )
            for name, arity in inventory.in_role("function").items()
        ),
    )

    return declared.with_(constants=(*declared.constants, *constants))


def _language(inventory: SymbolInventory = INVENTORY) -> LanguageAssembly:
    return LanguageIso().forward(_language_declaration(inventory))


def _grammar(
    entries: Sequence[tuple[str, str]],
    *,
    juxtaposition: bool = False,
    composition_order: int | None = 1,
    inventory: SymbolInventory = INVENTORY,
) -> GrammarAssembly:
    declaration = _grammar_declaration(
        entries,
        juxtaposition=juxtaposition,
        composition_order=composition_order,
    )
    language = LanguageIso().extend(
        _language_declaration(inventory), declaration.signature.extend
    )

    return GrammarIso().forward_language(declaration, language)


def test_a_second_named_interpretation_is_selected_from_a_document() -> None:
    declared = _grammar_declaration(
        [(form, category) for form, category, _ in LEXICON],
        interpretations=(
            NamedInterpretationSpec(
                name="logical-form", reading=MeaningInterpretationSpec()
            ),
            NamedInterpretationSpec(name="surface", reading=StringInterpretationSpec()),
        ),
    )
    document = GrammarDecl.model_validate_json(declared.model_dump_json())
    language = LanguageIso().extend(_language_declaration(), document.signature.extend)
    grammar = GrammarIso().forward_language(document, language)
    interpretation = _interpretation(grammar=grammar)
    surface = interpretation.interpretation("surface")
    category = grammar.syntax.parse(grammar.store, "NP")

    assert tuple(grammar.grammar.interpretations) == ("logical-form", "surface")
    assert type(surface) is StringReading
    assert read(
        axiom(interpretation.primary("john", category)),
        surface,
        interpretation.primaries,
    ) == (("john",),)

    with pytest.raises(InterpretationError, match="declares no interpretation"):
        interpretation.interpretation("missing")


def _oracle(language: LanguageAssembly, atoms: int = 64) -> TruthTableOracle:
    return TruthTableOracle(language.store, language.profile, bound=BOUND, atoms=atoms)


def _interpretation(
    *,
    entries: Sequence[tuple[str, str, str]] = LEXICON,
    grammar: GrammarAssembly | None = None,
    category_atoms: Mapping[str, str] = TYPES,
    constructors: Mapping[str, str] | None = None,
    inventory: SymbolInventory = INVENTORY,
    meanings: Mapping[tuple[str, str], str] | None = None,
    supplied: Mapping[str, str] | None = None,
    atoms: int = 64,
    normalization: Normalization | None = None,
) -> Interpretation:
    held = (
        _grammar(
            [(form, category) for form, category, _ in entries], inventory=inventory
        )
        if grammar is None
        else grammar
    )
    language = held.language

    store = language.store
    constructor_images = {
        name: language.syntax.read_type(text, quantified=("a0", "a1"))
        for name, text in (
            {"/": "a1 -> a0", "\\": "a1 -> a0"}
            if constructors is None
            else constructors
        ).items()
    }
    erasure = Erasure(
        store,
        {name: language.read_type(text) for name, text in category_atoms.items()},
        constructor_images,
        category_sort=held.language.category_sort,
        function_type=language.function_type,
    )
    declared_meanings = (
        {(form, category): meaning for form, category, meaning in entries}
        if meanings is None
        else meanings
    )
    rendered_meanings = {
        (
            form,
            held.syntax.render(
                held.store,
                held.syntax.parse(held.store, category),
            ),
        ): (meaning,)
        for (form, category), meaning in declared_meanings.items()
    }

    return interpretation_function(
        held,
        language,
        erasure=erasure,
        inventory=inventory,
        meanings=rendered_meanings,
        supplied={} if supplied is None else supplied,
        service=ProofService(
            lambda: _oracle(language, atoms),
            profile=language.profile,
            workers=1,
            batch_size=16,
            timeout=5.0,
        ),
        budget=ProofBudget(),
        truth="t",
        normalization=(
            Normalization(
                fuel=40,
                order="leftmost-outermost",
                eta="off",
                associative_commutative=(),
            )
            if normalization is None
            else normalization
        ),
        normalization_policy="eager",
        proof_search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
        meaning_selection=MeaningSelection("unique"),
    )


@pytest.fixture(scope="module")
def interpretation() -> Interpretation:
    return _interpretation()


def test_the_calculus_builds_stable_hypotheses_and_terms() -> None:
    language = _language()
    calculus = LambdaCalculus(language.store, language.calculus)
    first = calculus.hypothesis(0)

    assert calculus.store is language.store
    assert calculus.operators == language.calculus
    assert calculus.hypothesis(0) is first
    assert calculus.hypothesis(1) is not first
    assert calculus.free_variable("p0") is first
    assert calculus.bound_variable(0).op.name == "#bvar[Tm]"
    applied = calculus.application(first, calculus.hypothesis(1))

    assert applied.op.name == language.calculus.app
    assert applied.args == (first, calculus.hypothesis(1))
    abstracted = calculus.abstraction("p0", applied)

    assert abstracted.op.name == language.calculus.lam
    assert not abstracted.free & first.free

    with pytest.raises(ValueError, match="premise index"):
        calculus.hypothesis(-1)


def test_the_actions_of_application_composition_and_raising_are_derived(
    interpretation: Interpretation,
) -> None:
    actions = interpretation.actions
    render = interpretation.render

    assert set(actions.rules) == {APPLICATION, BACKWARD, COMPOSITION, RAISING}
    expected = {
        APPLICATION: "p0(p1)",
        BACKWARD: "p1(p0)",
        COMPOSITION: "\\x0. p0(p1(x0))",
        RAISING: "\\x0. x0(p0)",
    }

    for rule, rendered in expected.items():
        terms = actions.resolve(rule, None)

        assert len(terms) == 1
        assert render(terms[0]) == rendered

    assert actions.uncovered == frozenset()
    assert actions.ambiguous == frozenset()


def test_every_lexical_entry_bears_its_declared_meaning(
    interpretation: Interpretation,
) -> None:
    entries = interpretation.entries

    assert len(entries) == len(LEXICON)
    assert {form for form, _ in entries} == {form for form, _, _ in LEXICON}

    for (form, category), meanings in entries.items():
        entry = interpretation.primary(form, category)

        assert entry == Entry(form, category, meanings[0])
        assert interpretation.alternatives(form, category) == tuple(
            Entry(form, category, meaning) for meaning in meanings
        )
        assert entry.meaning is meanings[0]
        assert all(interpretation.store.contains(meaning) for meaning in meanings)

    assert interpretation.inventory is INVENTORY
    assert interpretation.language.name == "propositional"
    assert interpretation.service.prover == BOUND.prover
    assert interpretation.calculus.store is interpretation.store


def test_an_entry_or_a_rule_the_interpretation_lacks_is_refused(
    interpretation: Interpretation,
) -> None:
    (form, category), _ = next(iter(interpretation.entries.items()))

    with pytest.raises(UninterpretableDerivationError, match="assigns 'bob' no"):
        interpretation.primary("bob", category)

    with pytest.raises(UninterpretableDerivationError, match="has no rule 'swap'"):
        interpretation.rule("swap")

    assert interpretation.rule(APPLICATION).name == APPLICATION
    assert interpretation.primary(form, category).form == form


def _leaf(interpretation: Interpretation, form: str) -> Derivation[Term, float]:
    (held_form, category), _ = next(
        (key, meaning)
        for key, meaning in interpretation.entries.items()
        if key[0] == form
    )

    return axiom(interpretation.primary(held_form, category))


def _conclusion(interpretation: Interpretation, form: str) -> Term:
    return _leaf(interpretation, form).edge.output


def test_reading_a_derivation_composes_the_meanings_through_the_actions(
    interpretation: Interpretation,
) -> None:
    john = _leaf(interpretation, "john")
    mary = _leaf(interpretation, "mary")
    loves = _leaf(interpretation, "loves")
    sentence = _conclusion(interpretation, "sleeps").args[0]
    verb_phrase = _conclusion(interpretation, "sleeps")
    predicate = step(interpretation.rule(APPLICATION), (loves, mary), verb_phrase, 0.0)
    clause = step(interpretation.rule(BACKWARD), (john, predicate), sentence, 0.0)
    (meaning,) = interpretation.read(clause, interpretation.primaries)

    assert interpretation.render(meaning) == "love(john)(mary)"
    assert interpretation.formula(meaning) is meaning
    assert interpretation.read_text("love(john)(mary)") is meaning
    assert interpretation.read(john, interpretation.primaries) == (
        interpretation.primary("john", john.edge.output).meaning,
    )


def test_an_unbounded_composition_action_reads_a_third_order_derivation() -> None:
    entries = (
        ("john", "NP", "john"),
        ("predicate", "S/NP", "\\x. sleep(x)"),
        ("container", "((NP/NP)/NP)/NP", "\\x. \\y. \\z. x"),
    )
    grammar = _grammar(
        tuple((form, category) for form, category, _ in entries),
        composition_order=None,
    )
    interpretation = _interpretation(entries=entries, grammar=grammar)
    predicate = _leaf(interpretation, "predicate")
    container = _leaf(interpretation, "container")
    conclusion = grammar.syntax.parse(grammar.store, "((S/NP)/NP)/NP")
    rule = interpretation.rule(UNBOUNDED_COMPOSITION)
    (meaning,) = interpretation.read(
        step(rule, (predicate, container), conclusion, 0.0), interpretation.primaries
    )

    assert meaning is not None
    assert interpretation.store.contains(meaning)
    assert UNBOUNDED_COMPOSITION in interpretation.actions.rules
    assert len(interpretation.actions) == 4
    assert interpretation.actions.uncovered == frozenset()
    assert interpretation.actions.ambiguous == frozenset()
    assert interpretation.actions.exhausted == {}


def test_a_form_that_is_no_formula_is_declined_by_the_type_check(
    interpretation: Interpretation,
) -> None:
    john = interpretation.read_text("john")

    assert interpretation.check(john).is_well_typed

    with pytest.raises(IllTypedFormError, match="does not type as a formula"):
        interpretation.formula(john)

    with pytest.raises(IllTypedFormError, match="outside the inventory"):
        interpretation.formula(interpretation.read_text("sleep(bob)"))

    with pytest.raises(IllTypedFormError, match="leaves 1 constraints"):
        interpretation.formula(interpretation.read_text("sleep(sleep)"))


def test_a_decision_is_certified_and_only_its_own_certificate_is_verified(
    interpretation: Interpretation,
) -> None:
    both = interpretation.read_text("sleep(john) & sleep(mary)")
    one = interpretation.read_text("sleep(john)")
    proof = interpretation.decide(both, one)
    refutation = interpretation.decide(one, both)

    assert isinstance(proof.verdict, Proved)
    assert isinstance(refutation.verdict, Refuted)
    assert proof.prover == "truth-table"
    assert proof.profile_digest == interpretation.language.profile.digest
    assert proof.timeout == interpretation.service.timeout
    assert interpretation.verify(both, one, proof)
    assert interpretation.verify(one, both, refutation)
    assert not interpretation.verify(both, one, refutation)
    assert not interpretation.verify(one, both, proof)

    for tampered in (
        EntailmentCertificate(
            proved("trust me"),
            prover=proof.prover,
            profile_digest=proof.profile_digest,
            timeout=proof.timeout,
        ),
        EntailmentCertificate(
            proof.verdict,
            prover="z3",
            profile_digest=proof.profile_digest,
            timeout=proof.timeout,
        ),
        EntailmentCertificate(
            proof.verdict,
            prover=proof.prover,
            profile_digest="0" * 64,
            timeout=proof.timeout,
        ),
    ):
        assert not interpretation.verify(both, one, tampered)

    under_another_clock = EntailmentCertificate(
        proof.verdict,
        prover=proof.prover,
        profile_digest=proof.profile_digest,
        timeout=2.0,
    )

    assert not interpretation.verify(both, one, under_another_clock)


def test_the_interpretation_names_its_language_entries_and_actions(
    interpretation: Interpretation,
) -> None:
    assert repr(interpretation) == (
        f"InterpretationFunction('propositional', entries={len(LEXICON)}, "
        f"actions=4, inventory={INVENTORY!r})"
    )


def test_a_supplied_action_overrides_the_derived_one() -> None:
    interpretation = _interpretation(supplied={APPLICATION: "\\f. \\a. f(a) & f(a)"})
    terms = interpretation.actions.resolve(APPLICATION, None)

    assert len(terms) == 1
    assert interpretation.render(terms[0]) == "p0(p1) & p0(p1)"


def test_a_rule_with_no_derived_action_is_covered_by_a_supplied_one() -> None:
    grammar = _grammar(
        [(form, category) for form, category, _ in LEXICON], juxtaposition=True
    )

    with pytest.raises(
        InterpretationError, match=f"uncovered: \\['{JUXTAPOSITION}'\\]"
    ):
        _interpretation(grammar=grammar)

    interpretation = _interpretation(
        grammar=grammar, supplied={JUXTAPOSITION: "\\a. \\b. love(a)(b)"}
    )
    john = _leaf(interpretation, "john")
    mary = _leaf(interpretation, "mary")
    sentence = _conclusion(interpretation, "sleeps").args[0]
    (meaning,) = interpretation.read(
        step(interpretation.rule(JUXTAPOSITION), (john, mary), sentence, 0.0),
        interpretation.primaries,
    )

    assert interpretation.render(meaning) == "love(john)(mary)"


def test_an_exhausted_action_search_names_the_bound_in_its_refusal() -> None:
    normalization = Normalization(
        fuel=1,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )

    with pytest.raises(InterpretationError, match=r"bound-exhausted: .*'fuel'"):
        _interpretation(normalization=normalization)


def test_an_interpretation_report_distinguishes_action_failure_kinds(
    interpretation: Interpretation,
) -> None:
    assert interpretation.report() == {
        "actions": 4,
        "uncovered": "[]",
        "ambiguous": "[]",
        "bound-exhausted": "{}",
    }


@pytest.mark.parametrize(
    ("build", "fault"),
    [
        (
            lambda: _interpretation(meanings={(f, c): m for f, c, m in LEXICON[1:]}),
            "assigns 'john' no meaning at 'NP'",
        ),
        (
            lambda: _interpretation(
                meanings={**{(f, c): m for f, c, m in LEXICON}, ("bob", "NP"): "john"}
            ),
            "names entries the lexicon lacks: \\[\\('bob', 'NP'\\)\\]",
        ),
        (
            lambda: _interpretation(
                meanings={
                    **{(f, c): m for f, c, m in LEXICON},
                    ("john", "NP"): "john(",
                }
            ),
            "the meaning of 'john' at 'NP' 'john\\(' cannot be read",
        ),
        (
            lambda: _interpretation(
                meanings={**{(f, c): m for f, c, m in LEXICON}, ("john", "NP"): "bob"}
            ),
            "the meaning of 'john' at 'NP' names a symbol outside the inventory",
        ),
        (
            lambda: _interpretation(
                meanings={
                    **{(f, c): m for f, c, m in LEXICON},
                    ("john", "NP"): "sleep(john)",
                }
            ),
            "the meaning of 'john' at 'NP' does not type as the erasure",
        ),
        (
            lambda: _interpretation(
                meanings={
                    **{(f, c): m for f, c, m in LEXICON},
                    ("sleeps", "S\\NP"): "\\x. sleep(sleep)",
                }
            ),
            "the meaning of 'sleeps' at 'S\\\\\\\\NP'.*cannot be read",
        ),
        (
            lambda: _interpretation(supplied={"swap": "\\a. \\b. a"}),
            "actions for rules the grammar lacks: \\['swap'\\]",
        ),
        (
            lambda: _interpretation(supplied={APPLICATION: "\\f. \\a. sleep(john)"}),
            f"the action of rule '{APPLICATION}' does not use each of its 2",
        ),
        (
            lambda: _interpretation(supplied={APPLICATION: "\\f. \\a. f(a"}),
            f"the action of rule '{APPLICATION}' .* cannot be read",
        ),
        (
            lambda: _interpretation(
                inventory=SymbolInventory(
                    {
                        "individual": {"john": 0},
                        "predicate": {"sleep": 1, "love": 2},
                    }
                )
            ),
            "the meaning of 'mary' at 'NP' names a symbol outside the inventory",
        ),
    ],
)
def test_an_interpretation_that_does_not_cover_its_grammar_is_refused(
    build: Callable[[], Interpretation], fault: str
) -> None:
    with pytest.raises(InterpretationError, match=fault):
        build()


def test_the_inventory_types_every_symbol_by_its_kind_and_arity() -> None:
    interpretation = _interpretation(
        entries=(
            ("john", "NP", "john"),
            ("sleeps", "S\\NP", "\\x. sleep(father(x))"),
        ),
        inventory=SymbolInventory(
            {
                "individual": {"john": 0},
                "predicate": {"sleep": 1},
                "function": {"father": 1},
            }
        ),
    )
    store = interpretation.store
    signature = interpretation.signature
    types = interpretation.language.store
    entity = interpretation.language.read_type("e")
    arrow = types.signature.by_name("->")

    assert len(signature) == len(interpretation.language.constant_classes)
    assert (
        signature[
            _payload(store.const(interpretation.language.term_sort, "john").payload)
        ].body
        is entity
    )
    assert signature[
        _payload(store.const(interpretation.language.term_sort, "father").payload)
    ].body is types.op(arrow, entity, entity)
    assert signature[
        _payload(store.const(interpretation.language.term_sort, "sleep").payload)
    ].body is types.op(arrow, entity, interpretation.truth)
    assert interpretation.formula(interpretation.read_text("sleep(father(john))"))


def _payload(payload: int | None) -> int:
    if payload is None:
        msg = "a constant has a payload"
        raise ValueError(msg)

    return payload


SCOPING_ENTRY: Final = (
    "reads",
    "(S\\NP)/(S\\(S/NP))",
    "\\Q. \\x. Q(\\y. love(x)(y))",
)


def test_a_meaning_more_general_than_its_erasure_is_read_at_the_erasure() -> None:
    interpretation = _interpretation(entries=(*LEXICON, SCOPING_ENTRY))
    rendered = sorted(
        interpretation.render(meaning)
        for (form, _), meanings in interpretation.entries.items()
        if form == "reads"
        for meaning in meanings
    )

    assert rendered == ["\\x0. \\x1. x0(\\x2. love(x1)(x2))"]
