import json
from hashlib import sha256
from typing import TYPE_CHECKING, Final, cast

import didactic.api as dx
import pytest
from hypothesis import strategies as st
from tests.support.derivations import axiom, step
from tests.unit.data.synthetic.spec.conftest import (
    INTERPRETED_CONSTANTS,
    INTERPRETED_ENTRIES,
    INTERPRETED_TYPES,
    ISO_LAW_NAMES,
    Dataset,
    IsoLaw,
    _interpretation,
    _interpreted,
    _interpreted_grammar,
    _projection,
)

from lofi.data.laws import ISO_LAWS, IsoFixture
from lofi.data.synthetic import (
    InterpretationFunction,
    SyntheticDataset,
    SyntheticGenerator,
    UninterpretableDerivationError,
)
from lofi.data.synthetic.spec.corpus import (
    SOURCE_VERSION,
    TARGET_VERSION,
    LabelNoiseRateSpec,
    LabelNoiseSpec,
    SyntheticCorpusDecl,
    SyntheticCorpusIso,
    SyntheticCorpusLowerer,
    UniformLengthSpec,
    canonical_bytes,
    fingerprint,
)
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    LexicalMeaningSourceSpec,
    MeaningSpec,
    ProofSearchSpec,
    StructuredMeaningSourceSpec,
    lower_interpretation,
)
from lofi.data.synthetic.spec.semantic_lexicon import preset
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    DeclarationError,
    InfixConstructorNotationSpec,
    PatternRuleSpec,
    PrefixConstructorNotationSpec,
    RuleSetSpec,
    fixed_seed,
)
from lofi.spec.grammar import (
    BudgetSpec,
    FunctorSpec,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    LexiconStructure,
    ParametricAmbiguityLawSpec,
    ParametricJointSpec,
    PowerLawRankLawSpec,
    RootGoalSpec,
    SignatureMorphismErasureSpec,
    SignatureSpec,
    UniformRankLawSpec,
)
from lofi.spec.language import (
    ConstantSpec,
    InfixNotationSpec,
    LanguageDecl,
    LanguageIso,
    PrefixNotationSpec,
    registered_language,
)
from lofi.spec.normalization import NormalizationSpec
from lofi.spec.provers import SmtProverSpec, TruthTableProverSpec

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi._types import JsonValue


FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)


def _interpreted_projection(dataset: Dataset) -> str:
    corpus = dataset.corpus
    document: JsonValue = {
        "base": _projection(dataset),
        "forms": [[form.item_id, form.text] for form in corpus.forms.values()],
        "ill_typed": [
            [record.item_id, record.kind, record.text, repr(record.certificate)]
            for record in corpus.ill_typed
        ],
        "ill_typing": repr(corpus.ill_typing),
    }

    return sha256(json.dumps(document, sort_keys=True).encode()).hexdigest()


def _lower(
    declaration: InterpretationSpec,
    *,
    grammar: GrammarDecl | None = None,
    language: LanguageDecl | None = None,
) -> tuple[GrammarAssembly, InterpretationFunction]:
    languages = LanguageIso()

    return lower_interpretation(
        _interpreted_grammar() if grammar is None else grammar,
        registered_language("propositional") if language is None else language,
        declaration,
        GrammarIso(languages),
        languages,
    )


def _grammar_with_meanings(
    entries: tuple[tuple[str, str, str], ...],
) -> GrammarDecl:
    """Return the interpreted grammar with the supplied entry meanings."""
    grammar = _interpreted_grammar()

    return grammar.with_(
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category, meaning=meaning)
                for form, category, meaning in entries
            )
        )
    )


def test_typed_constant_classes_become_inventory_roles() -> None:
    declaration = _interpretation()
    _, interpretation = _lower(declaration)

    assert interpretation.inventory.roles == {
        "individual": {"john": 0, "mary": 0},
        "predicate": {"sleep": 1, "love": 2},
    }
    assert interpretation.language.checker.store is interpretation.store


def test_the_language_logic_is_the_only_labeling_prover_authority() -> None:
    fields = InterpretationSpec.__field_specs__

    assert "oracle" not in fields
    assert "budget" not in fields


def _arrow_grammar() -> GrammarDecl:
    """Return a grammar over one self-opposite arrow and a non-functor pair."""
    return GrammarDecl(
        name="arrow",
        signature=SignatureSpec(
            primitives=("S", "NP"),
            constructors=(
                ConstructorSpec(
                    name="arrow",
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling="|", precedence=50, associativity="left"
                    ),
                ),
                ConstructorSpec(
                    name="pair",
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling="*", precedence=60, associativity="left"
                    ),
                ),
            ),
            functors=(
                FunctorSpec(
                    name="|",
                    constructor="arrow",
                    argument_side="either",
                    opposite="|",
                ),
            ),
        ),
        rules=RuleSetSpec(name="application", rules=(ApplicationSpec(view="|"),)),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="john", category="NP", meaning="john"),
                LexicalEntrySpec(
                    form="sleeps", category="S|NP", meaning="\\x. sleep(x)"
                ),
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
    )


def _arrow_interpretation(
    constructors: dict[str, str],
) -> InterpretationSpec:
    return InterpretationSpec(
        erasure=SignatureMorphismErasureSpec(
            atoms={"S": "t", "NP": "e"}, constructors=constructors
        ),
        constants=INTERPRETED_CONSTANTS[:3],
        meanings=MeaningSpec(source=LexicalMeaningSourceSpec()),
        actions={"application:arrow": "\\a. \\f. f(a)"},
    )


def test_an_interpretation_declares_no_nonfunctor_images_by_default() -> None:
    assert _interpretation().erasure.constructors == {}


def test_meaning_and_proof_search_declarations_build_interior_policies() -> None:
    meanings = MeaningSpec(
        source=LexicalMeaningSourceSpec(),
        inhabitant_selection="all",
        search_budget=7,
    )
    search = ProofSearchSpec(logic="affine").build(
        max_inhabitants=meanings.search_budget
    )

    assert meanings.selection().kind == "all"
    assert (search.contraction, search.weakening) == (False, True)
    assert search.max_uses == 1
    assert search.max_inhabitants == 7


def test_proof_search_declaration_refuses_contraction_in_substructural_logics() -> None:
    with pytest.raises(dx.ValidationError, match="at most once"):
        ProofSearchSpec(logic="linear", max_uses=2)


def test_functor_images_come_from_the_grammar_and_others_from_the_declaration() -> None:
    declaration = _arrow_interpretation({"pair": "a1 -> a0"})
    _, interpretation = _lower(declaration, grammar=_arrow_grammar())

    assert interpretation.inventory.in_role("predicate") == {"sleep": 1}


def test_self_opposite_dynamic_witness_ambiguity_reaches_the_report() -> None:
    grammar = _arrow_grammar().with_(
        rules=RuleSetSpec(
            name="application-and-composition",
            rules=(
                ApplicationSpec(view="|"),
                CompositionSpec(direction="unrestricted", order=None, view="|"),
            ),
        )
    )
    _, interpretation = _lower(
        _arrow_interpretation({"pair": "a1 -> a0"}),
        grammar=grammar,
    )
    (composition,) = tuple(
        name for name in interpretation.actions.rules if name.startswith("composition:")
    )

    assert interpretation.actions.ambiguous == frozenset((composition,))
    assert interpretation.report()["ambiguous"] == f'["{composition}"]'


def test_renamed_slashes_take_their_images_from_their_roles() -> None:
    renamed = _interpreted_grammar().signature.with_(
        constructors=(
            ConstructorSpec(
                name="over",
                kind=("Cat", "Cat", "Cat"),
                notation=InfixConstructorNotationSpec(
                    spelling="/", precedence=50, associativity="left"
                ),
            ),
            ConstructorSpec(
                name="under",
                kind=("Cat", "Cat", "Cat"),
                notation=InfixConstructorNotationSpec(
                    spelling="\\", precedence=50, associativity="left"
                ),
            ),
        ),
        functors=(
            FunctorSpec(
                name="/", constructor="over", argument_side="right", opposite="\\"
            ),
            FunctorSpec(
                name="\\", constructor="under", argument_side="left", opposite="/"
            ),
        ),
    )
    declaration = _interpretation()
    _, interpretation = _lower(
        declaration, grammar=_interpreted_grammar().with_(signature=renamed)
    )

    assert interpretation.inventory.in_role("predicate") == {"sleep": 1, "love": 2}


def test_a_constructor_without_a_declared_image_is_refused() -> None:
    with pytest.raises(DeclarationError, match=r"coverage has missing \('pair',\)"):
        _lower(_arrow_interpretation({}), grammar=_arrow_grammar())


def test_an_image_restated_for_a_functor_constructor_is_refused() -> None:
    declaration = _arrow_interpretation(
        {
            "arrow": "a1 -> a0",
            "pair": "a1 -> a0",
        }
    )

    with pytest.raises(DeclarationError, match=r"functor constructors \('arrow',\)"):
        _lower(declaration, grammar=_arrow_grammar())


def test_an_image_restated_for_a_slash_is_refused() -> None:
    declaration = _interpretation(constructors={"/": "a1 -> a0"})

    with pytest.raises(DeclarationError, match=r"functor constructors \('/',\)"):
        _lower(declaration)


def _constant(name: str) -> ConstantPatternSpec:
    return ConstantPatternSpec(name=name)


def _proof_sweep_grammar() -> GrammarDecl:
    """Return the finite grammar used to sweep proof declarations."""
    return GrammarDecl(
        name="proof-sweep",
        signature=SignatureSpec(
            constructors=(
                ConstructorSpec(
                    name="arrow",
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling="|", precedence=50, associativity="left"
                    ),
                ),
            ),
            functors=(
                FunctorSpec(
                    name="|",
                    constructor="arrow",
                    argument_side="either",
                    opposite="|",
                ),
            ),
            primitives=("X", "Y"),
        ),
        rules=RuleSetSpec(
            name="proof-sweep",
            rules=(
                PatternRuleSpec(
                    name="weakening",
                    premises=(_constant("X"), _constant("Y")),
                    conclusion=_constant("Y"),
                ),
                PatternRuleSpec(
                    name="choice",
                    premises=(_constant("Y"), _constant("Y")),
                    conclusion=_constant("Y"),
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="f", category="X", meaning="fx"),
                LexicalEntrySpec(form="p", category="Y", meaning="p"),
            )
        ),
        goal=RootGoalSpec(categories=("Y",)),
    )


@pytest.mark.parametrize(
    ("logic", "selection", "covered", "uncovered", "ambiguous"),
    [
        ("relevant", "unique", None, ("choice", "weakening"), ()),
        ("linear", "all", None, ("choice", "weakening"), ()),
        ("affine", "unique", None, ("choice",), ("choice",)),
        ("intuitionistic", "first", 2, (), ("choice",)),
        ("intuitionistic", "smallest", 2, (), ("choice",)),
        ("intuitionistic", "all", 2, (), ("choice",)),
    ],
)
def test_document_sweep_changes_proof_action_coverage(
    logic: str,
    selection: str,
    covered: int | None,
    uncovered: tuple[str, ...],
    ambiguous: tuple[str, ...],
) -> None:
    declaration = InterpretationSpec.model_validate(
        {
            "erasure": {
                "kind": "signature-morphism",
                "atoms": {"X": "e", "Y": "t"},
            },
            "constants": (
                {"name": "fx", "type": "e", "classes": ("individual",)},
                {"name": "p", "type": "t", "classes": ("proposition",)},
            ),
            "meanings": {
                "source": {"kind": "lexical"},
                "inhabitant_selection": selection,
                "search_budget": 8,
            },
            "proof_search": {"logic": logic, "max_uses": 1},
        }
    )

    if covered is None:
        with pytest.raises(DeclarationError) as caught:
            _lower(declaration, grammar=_proof_sweep_grammar())

        assert f"ambiguous: {list(ambiguous)!r}" in str(caught.value)
        assert f"uncovered: {list(uncovered)!r}" in str(caught.value)
        return

    _, interpretation = _lower(declaration, grammar=_proof_sweep_grammar())
    assert len(interpretation.actions) == covered
    assert interpretation.actions.uncovered == frozenset(uncovered)
    assert interpretation.actions.ambiguous == frozenset(ambiguous)


def test_plural_selected_actions_refuse_a_scalar_gold_reading() -> None:
    declaration = InterpretationSpec.model_validate(
        {
            "erasure": {
                "kind": "signature-morphism",
                "atoms": {"X": "e", "Y": "t"},
            },
            "constants": (
                {"name": "fx", "type": "e", "classes": ("individual",)},
                {"name": "p", "type": "t", "classes": ("proposition",)},
            ),
            "meanings": {
                "source": {"kind": "lexical"},
                "inhabitant_selection": "all",
                "search_budget": 8,
            },
            "proof_search": {"logic": "intuitionistic", "max_uses": 1},
        }
    )
    assembly, interpretation = _lower(
        declaration,
        grammar=_proof_sweep_grammar(),
    )
    category = assembly.syntax.parse(assembly.store, "Y")
    premise = axiom(interpretation.primary("p", category))
    derivation = step(
        interpretation.rule("choice"),
        (premise, premise),
        category,
        0.0,
    )

    with pytest.raises(
        UninterpretableDerivationError,
        match="assigns the derivation several meanings",
    ):
        interpretation.read(derivation, interpretation.primaries)


def test_constructors_named_as_slashes_without_roles_take_declared_images() -> None:
    grammar = GrammarDecl(
        name="slashless",
        signature=SignatureSpec(
            primitives=("S", "NP", "VP"),
            constructors=(
                ConstructorSpec(
                    name="/",
                    kind=("Cat", "Cat", "Cat"),
                    notation=PrefixConstructorNotationSpec(spelling="/"),
                ),
                ConstructorSpec(
                    name="\\",
                    kind=("Cat", "Cat", "Cat"),
                    notation=PrefixConstructorNotationSpec(spelling="\\"),
                ),
            ),
            syntax="s-expression",
        ),
        rules=RuleSetSpec(
            name="productions",
            rules=(
                PatternRuleSpec(
                    name="s",
                    premises=(_constant("NP"), _constant("VP")),
                    conclusion=_constant("S"),
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
    )
    assembly = GrammarIso().forward(grammar, registered_language("propositional"))
    declaration = _arrow_interpretation({"/": "a0 -> a1", "\\": "a1 -> a0"})

    assert assembly.category_measure.constructors == ()
    assert declaration.erasure.constructors == {
        "/": "a0 -> a1",
        "\\": "a1 -> a0",
    }


def test_interpretation_round_trips_with_typed_constants() -> None:
    declaration = _interpretation()

    assert (
        InterpretationSpec.model_validate_json(declaration.model_dump_json())
        == declaration
    )
    assert declaration.constants == INTERPRETED_CONSTANTS


def test_normalization_lowers_with_the_grammar_owned_fuel() -> None:
    declaration = _interpretation().with_(
        normalization=NormalizationSpec(
            order="weak-head",
            policy="lazy",
            eta="contract",
            associative_commutative=("and",),
        )
    )
    grammar = _interpreted_grammar().with_(budget=BudgetSpec(beta_fuel=17))
    _, interpretation = _lower(declaration, grammar=grammar)

    assert interpretation.normalization.fuel == 17
    assert interpretation.normalization.order == "weak-head"
    assert interpretation.normalization.eta == "contract"
    assert interpretation.normalization.associative_commutative == ("and",)
    assert interpretation.normalization_policy == "lazy"


def test_normalization_refuses_an_unknown_associative_commutative_constant() -> None:
    declaration = _interpretation().with_(
        normalization=NormalizationSpec(associative_commutative=("missing",))
    )

    with pytest.raises(DeclarationError, match="missing"):
        _lower(declaration)


def test_normalization_refuses_a_nonbinary_associative_commutative_constant() -> None:
    declaration = _interpretation().with_(
        normalization=NormalizationSpec(associative_commutative=("not",))
    )

    with pytest.raises(DeclarationError, match="not binary"):
        _lower(declaration)


def test_meaning_sources_are_one_tagged_document_surface() -> None:
    lexical = LexicalMeaningSourceSpec()
    structured = StructuredMeaningSourceSpec(preset="natural")

    assert lexical.model_dump()["kind"] == "lexical"
    assert structured.model_dump()["kind"] == "structured"
    assert structured.semantic_lexicon() == preset("natural")


def test_a_lexical_meaning_source_requires_an_entry_meaning() -> None:
    grammar = _interpreted_grammar()
    lexicon = cast("InlineLexiconSpec", grammar.lexicon)
    grammar = grammar.with_(
        lexicon=InlineLexiconSpec(
            entries=tuple(entry.with_(meaning=None) for entry in lexicon.entries)
        )
    )

    with pytest.raises(DeclarationError, match="needs a meaning on a lexical entry"):
        _lower(_interpretation(), grammar=grammar)


def test_a_lexical_entry_cannot_list_the_same_meaning_twice() -> None:
    grammar = _interpreted_grammar()
    lexicon = cast("InlineLexiconSpec", grammar.lexicon)
    grammar = grammar.with_(
        lexicon=lexicon.with_(entries=(*lexicon.entries, lexicon.entries[0]))
    )

    with pytest.raises(DeclarationError, match="lists one meaning twice"):
        _lower(_interpretation(), grammar=grammar)


def _generated_interpreted(
    constants: int | None = None, *, meanings_per_item: float = 1.0
) -> SyntheticCorpusDecl:
    """Return a generated surface lexicon interpreted by generated meanings."""
    base = _interpreted_grammar("generated-meanings")
    grammar = base.with_(
        signature=base.signature.with_(primitives=("S", "NP", "N", "conj")),
        lexicon=LexiconStructure(
            forms=24,
            joint=ParametricJointSpec(
                categories=("NP", "S\\NP"),
                category_shares=PowerLawRankLawSpec(exponent=0.0),
            ),
            ambiguity=ParametricAmbiguityLawSpec(
                mean=1.0, dispersion=0.0, maximum=1, placement=0.0
            ),
            rank_law=UniformRankLawSpec(),
            seed=7,
        ),
    )
    source = (
        StructuredMeaningSourceSpec(preset="natural")
        if constants is None
        else StructuredMeaningSourceSpec(
            structure=preset("natural").with_(constants=constants)
        )
    )
    structure = source.semantic_lexicon()
    interpretation = InterpretationSpec(
        erasure=SignatureMorphismErasureSpec(
            atoms={"S": "t", "NP": "e", "N": "⟨e,t⟩", "conj": "e"}
        ),
        constants=(),
        meanings=MeaningSpec(source=source),
    )
    interpretation = interpretation.with_(
        meanings=MeaningSpec(
            source=StructuredMeaningSourceSpec(
                structure=structure.with_(
                    polysemy=ParametricAmbiguityLawSpec(
                        mean=meanings_per_item,
                        dispersion=0.0,
                        maximum=max(1, int(meanings_per_item + 0.5)),
                        placement=0.0,
                    )
                )
            )
        ),
    )

    return SyntheticCorpusDecl(
        retained={"validation": 0.25},
        grammar=grammar,
        language=registered_language("neo-davidsonian").with_(
            logic=registered_language("neo-davidsonian").logic.with_(
                prover=SmtProverSpec()
            )
        ),
        size=12,
        seed=3,
        length=UniformLengthSpec(min_length=2, max_length=2),
        interpretation=interpretation,
    )


def test_a_generated_surface_lexicon_lowers_under_the_natural_semantic_preset() -> None:
    dataset = SyntheticCorpusIso(policy=FOREST_POLICY).forward(_generated_interpreted())
    interpretation = dataset.source_interpretation

    assert interpretation is not None
    assert interpretation.semantic_lexicon is not None
    assert dataset.semantic is not None
    assert dataset.semantic.concordance == pytest.approx(1.0)
    assert dataset.corpus.forms
    assert dataset.corpus.labels


def test_sweeping_the_semantic_inventory_changes_the_symbols_used_report() -> None:
    smaller = (
        SyntheticCorpusIso(policy=FOREST_POLICY)
        .forward(_generated_interpreted(20))
        .semantic
    )
    larger = (
        SyntheticCorpusIso(policy=FOREST_POLICY)
        .forward(_generated_interpreted(170))
        .semantic
    )

    assert smaller is not None
    assert larger is not None
    assert smaller.joint.realized_size == 20
    assert larger.joint.realized_size == 170
    assert smaller.symbols_used != larger.symbols_used


def test_declared_polysemy_is_reported_as_meanings_per_surface_form() -> None:
    report = (
        SyntheticCorpusIso(policy=FOREST_POLICY)
        .forward(_generated_interpreted(meanings_per_item=2.0))
        .semantic
    )

    assert report is not None
    assert report.per_form_meanings.counts == {2: 24}


def test_inventory_roles_do_not_assume_a_result_type() -> None:
    declaration = _interpretation(
        constants=(
            *INTERPRETED_CONSTANTS,
            ConstantSpec(
                name="chooser",
                type="e -> t",
                classes=("choice", "higher-order"),
            ),
        ),
    )
    _, interpretation = _lower(declaration)

    assert interpretation.inventory.in_role("choice") == {"chooser": 1}
    assert interpretation.inventory.in_role("higher-order") == {"chooser": 1}


def test_an_inventory_constant_needs_a_declared_role() -> None:
    declaration = _interpretation(
        constants=(
            *INTERPRETED_CONSTANTS,
            ConstantSpec(name="orphan", type="e"),
        ),
    )

    with pytest.raises(DeclarationError, match="needs an inventory role"):
        _lower(declaration)


def test_the_oracle_serves_the_connectives_a_language_declares_by_role() -> None:
    renamed = registered_language("propositional").with_(
        name="renamed",
        logic=registered_language("propositional").logic.with_(negation="nicht"),
        constants=(
            ConstantSpec(
                name="nicht",
                type="t -> t",
                classes=("sentential", "negation"),
                notation=PrefixNotationSpec(ascii="~", unicode="¬", precedence=6),
            ),
            ConstantSpec(
                name="und",
                type="t -> t -> t",
                classes=("sentential", "conjunction"),
                notation=InfixNotationSpec(
                    ascii="&", unicode="∧", precedence=4, associativity="left"
                ),
            ),
        ),
    )
    meanings = (
        *INTERPRETED_ENTRIES[:2],
        ("sleeps", "S\\NP", "\\x. ~sleep(x) & sleep(x)"),
        *INTERPRETED_ENTRIES[3:],
    )
    _, interpretation = _lower(
        _interpretation(), grammar=_grammar_with_meanings(meanings), language=renamed
    )

    assert interpretation.service.prover == "truth-table"


def test_an_inferred_constant_takes_the_type_its_categories_demand() -> None:
    declaration = _interpretation(
        constants=(
            *INTERPRETED_CONSTANTS[:2],
            ConstantSpec(name="sleep", type="infer", classes=("predicate",)),
            INTERPRETED_CONSTANTS[3],
        )
    )
    _, interpretation = _lower(declaration)
    language = interpretation.language

    assert language.print_type(language.scheme("sleep").body) == "e -> t"
    assert interpretation.inventory.in_role("predicate") == {"sleep": 1, "love": 2}


def test_an_interpreted_declaration_lowers_to_forms() -> None:
    declaration = _interpreted()
    iso = SyntheticCorpusIso(policy=FOREST_POLICY)
    dataset = iso.forward(declaration)
    corpus = dataset.corpus
    interpretation = dataset.source_interpretation

    assert interpretation is not None
    assert len(corpus.forms) == len(corpus.examples) == 10
    generator = SyntheticGenerator(
        dataset.source_grammar,
        policy=FOREST_POLICY,
        interpretation=interpretation,
    )

    assert generator.knobs == corpus.knobs

    for example in corpus.examples:
        term = interpretation.read_text(corpus.forms[example.item.item_id].text)

        assert interpretation.formula(term) is term

    assert iso.backward(dataset) == declaration
    assert SyntheticCorpusDecl.model_validate_json(canonical_bytes(declaration)) == (
        declaration
    )


def test_two_lowerings_of_an_interpreted_declaration_agree_on_forms() -> None:
    declaration = _interpreted(
        noise=LabelNoiseSpec(
            grammatical=LabelNoiseRateSpec(rate=0.3),
            ungrammatical=LabelNoiseRateSpec(rate=0.3),
        )
    )
    first = SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)
    second = SyntheticCorpusIso(
        GrammarIso(), LanguageIso(), policy=FOREST_POLICY
    ).forward(declaration)

    assert first == second
    assert _interpreted_projection(first) == _interpreted_projection(second)
    assert first.corpus.forms == second.corpus.forms


@pytest.mark.parametrize(
    "changed",
    [
        _interpreted(
            language=registered_language("propositional").with_(
                logic=registered_language("propositional").logic.with_(
                    prover=TruthTableProverSpec(timeout=4.0)
                )
            )
        ),
        _interpreted(
            interpretation=_interpretation(
                constants=(
                    *INTERPRETED_CONSTANTS,
                    ConstantSpec(name="p", type="e -> t", classes=("predicate",)),
                )
            )
        ),
        _interpreted(
            grammar=_grammar_with_meanings(
                (
                    *INTERPRETED_ENTRIES[:2],
                    ("sleeps", "S\\NP", "\\x. love(x)(x)"),
                    *INTERPRETED_ENTRIES[3:],
                )
            )
        ),
        _interpreted(
            interpretation=_interpretation(actions={"application:/": "\\f. \\a. f(a)"})
        ),
    ],
    ids=["timeout", "constants", "meaning", "action"],
)
def test_changing_the_interpretation_changes_the_fingerprint(
    changed: SyntheticCorpusDecl,
) -> None:
    declaration = _interpreted()
    iso = SyntheticCorpusIso(policy=FOREST_POLICY)

    assert changed != declaration
    assert canonical_bytes(changed) != canonical_bytes(declaration)
    assert fingerprint(changed) != fingerprint(declaration)
    assert iso.forward(changed).fingerprint != iso.forward(declaration).fingerprint


def test_changing_a_meaning_changes_the_forms_drawn() -> None:
    declaration = _interpreted()
    relabeled = _interpreted(
        grammar=_grammar_with_meanings(
            (
                *INTERPRETED_ENTRIES[:2],
                ("sleeps", "S\\NP", "\\x. love(x)(x)"),
                *INTERPRETED_ENTRIES[3:],
            )
        )
    )
    iso = SyntheticCorpusIso(policy=FOREST_POLICY)
    before = iso.forward(declaration).corpus
    after = iso.forward(relabeled).corpus

    assert tuple(example.item.text for example in before.examples) == tuple(
        example.item.text for example in after.examples
    )
    assert before.forms != after.forms


@pytest.mark.conformance
@pytest.mark.parametrize("law", ISO_LAWS, ids=ISO_LAW_NAMES)
def test_the_lowering_holds_to_each_iso_law_under_an_interpretation(
    law: IsoLaw,
) -> None:
    declarations = st.builds(
        _interpreted,
        size=st.integers(min_value=1, max_value=5),
        seed=st.integers(min_value=0, max_value=1_000),
    )
    law(
        IsoFixture(
            SyntheticCorpusIso(policy=FOREST_POLICY),
            sources=declarations,
            targets=declarations.map(SyntheticCorpusIso(policy=FOREST_POLICY).forward),
            max_examples=4,
        )
    )


@pytest.mark.parametrize(
    ("build", "message"),
    [
        (
            lambda: SignatureMorphismErasureSpec(atoms={"S": ""}),
            "needs a type term",
        ),
        (
            lambda: LexicalEntrySpec(form="a", category="NP", meaning=""),
            "logical form",
        ),
        (lambda: TruthTableProverSpec(timeout=0.0), "a prover timeout is positive"),
        (lambda: TruthTableProverSpec(timeout=float("inf")), "not finite"),
        (
            lambda: NormalizationSpec(associative_commutative=("and", "and")),
            "distinct",
        ),
        (
            lambda: InterpretationSpec(
                entity="t",
                erasure=SignatureMorphismErasureSpec(atoms=INTERPRETED_TYPES),
                constants=INTERPRETED_CONSTANTS,
                meanings=MeaningSpec(source=LexicalMeaningSourceSpec()),
            ),
            "two types",
        ),
    ],
)
def test_the_interpretation_declaration_refuses_a_setting_outside_its_range(
    build: Callable[[], dx.Model], message: str
) -> None:
    with pytest.raises(dx.ValidationError, match=message):
        build()


def test_the_dataset_join_refuses_a_constant_declared_twice() -> None:
    declaration = _interpretation(
        constants=(
            *INTERPRETED_CONSTANTS,
            ConstantSpec(name="and", type="t -> t -> t"),
        )
    )

    with pytest.raises(DeclarationError, match=r"already declares.*\['and'\]"):
        _lower(declaration)


def _corpus_without_johns_meaning() -> SyntheticCorpusDecl:
    """Return the interpreted fixture with one lexical meaning absent."""
    corpus = _interpreted()
    grammar = _interpreted_grammar()
    lexicon = cast("InlineLexiconSpec", grammar.lexicon)
    entries = tuple(
        entry.with_(meaning=None) if entry.form == "john" else entry
        for entry in lexicon.entries
    )

    return corpus.with_(grammar=grammar.with_(lexicon=lexicon.with_(entries=entries)))


@pytest.mark.parametrize(
    ("declaration", "message"),
    [
        (
            _corpus_without_johns_meaning(),
            "assigns 'john' no meaning at 'NP'",
        ),
        (
            _interpreted(
                interpretation=_interpretation(constants=INTERPRETED_CONSTANTS[:2])
            ),
            "outside the inventory",
        ),
        (
            _interpreted(interpretation=_interpretation(atoms={"S": "t"})),
            "atom map coverage",
        ),
        (
            _interpreted(
                interpretation=_interpretation(
                    atoms={**INTERPRETED_TYPES, "NP": "⟨e,w⟩"}
                )
            ),
            "'w' at character offset 3",
        ),
        (
            _interpreted(
                interpretation=_interpretation(
                    atoms={**INTERPRETED_TYPES, "NP": "e"}, type_syntax="s-expression"
                )
            ),
            "expected '\\('",
        ),
    ],
    ids=["meaning", "symbol", "types", "angle", "s-expression"],
)
def test_an_interpretation_that_does_not_cover_its_grammar_is_refused_at_check(
    declaration: SyntheticCorpusDecl, message: str
) -> None:
    with pytest.raises(DeclarationError, match=message):
        SyntheticCorpusIso(policy=FOREST_POLICY).forward(declaration)


def test_the_lowerer_refuses_a_dataset_whose_forms_disagree() -> None:
    lowerer = SyntheticCorpusLowerer()
    declaration = _interpreted()
    checked = lowerer.check(declaration)
    plain = SyntheticGenerator(checked.source_grammar, policy=FOREST_POLICY).sample(
        declaration.size,
        seed=fixed_seed("seed", declaration.seed),
        length=checked.length,
    )

    assert checked.source_interpretation is not None

    with pytest.raises(DeclarationError, match="forms against the declared"):
        lowerer.validate(
            SyntheticDataset(
                canonical_bytes(declaration),
                checked.source_grammar,
                plain,
                source_version=SOURCE_VERSION,
                target_version=TARGET_VERSION,
            )
        )

    lowerer.validate(lowerer.lower(checked, FOREST_POLICY))


def test_types_lower_alike_in_angle_brackets_and_s_expressions() -> None:
    sexp = {
        "S": '("#const[Ty]" "t")',
        "NP": '("#const[Ty]" "e")',
        "conj": '("#const[Ty]" "e")',
    }
    angle = SyntheticCorpusIso(policy=FOREST_POLICY).forward(_interpreted())
    spelled = SyntheticCorpusIso(policy=FOREST_POLICY).forward(
        _interpreted(
            interpretation=_interpretation(atoms=sexp, type_syntax="s-expression")
        )
    )

    def drawn(dataset: Dataset) -> list[tuple[str, ...]]:
        corpus = dataset.corpus

        return [
            (
                " ".join(token.surface for token in example.item.tokens),
                repr(example.supervision),
                corpus.forms[example.item.item_id].text,
            )
            for example in corpus.examples
        ]

    assert angle.fingerprint != spelled.fingerprint
    assert drawn(angle)
    assert drawn(angle) == drawn(spelled)
