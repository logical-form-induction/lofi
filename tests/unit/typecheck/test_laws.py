from functools import cache
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance.coherence_square import (
    CoherenceSquareConformance,
    CoherenceSquareSuite,
    SentenceBatch,
)
from tests.language import grammar_language
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi._checks import LawViolationError
from lofi.lexicon import Entry
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstructorSpec,
    PrefixConstructorNotationSpec,
    RuleSetSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import (
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    NoNormalFormSpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import (
    ConstantSpec,
    LanguageIso,
    registered_language,
)
from lofi.terms import Term, UnknownConstantError
from lofi.typecheck import (
    CLAMP_PROJECTION_LAWS,
    CategoryProjection,
    ClampProjectionFixture,
    CoherenceFixture,
    CoherenceItem,
    CoherenceReport,
    Erasure,
    LexicalSignature,
    PairProjection,
    Scheme,
    TypingLawFixture,
    check_clamp_projection_laws,
    check_typing_laws,
    coherence_square,
)
from lofi.typecheck.laws import (
    a_meaning_is_read_exactly_when_declared,
    a_projection_reads_the_same_reading_the_same_way,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.judge import ProofSchema
    from lofi.spec.language import LanguageDecl


class _MeaningLess:
    """A projection declaring a meaning it never reads."""

    @property
    def has_meaning(self) -> bool:
        return True

    def category(self, reading: str, /) -> str:
        return reading

    def meaning(self, reading: str, /) -> str | None:
        del reading

        return None


class _Fresh:
    """A projection building a new category on every read."""

    @property
    def has_meaning(self) -> bool:
        return False

    def category(self, reading: str, /) -> list[str]:
        return [reading]

    def meaning(self, reading: str, /) -> list[str] | None:
        del reading

        return None


def test_the_typing_square_accepts_a_subsuming_principal_type() -> None:
    fixture = TypingLawFixture(
        logical_form="lambda f",
        category="S/(S\\NP)",
        typeof=lambda _logical_form: "(e -> A) -> A",
        erase=lambda _category: "(e -> t) -> t",
        subsumes=lambda general, specific: (
            MappingProxyType({0: "t"})
            if (general, specific) == ("(e -> A) -> A", "(e -> t) -> t")
            else None
        ),
    )

    check_typing_laws(fixture)


def test_an_unmet_typing_square_is_an_explicit_law_target() -> None:
    fixture = TypingLawFixture(
        logical_form="bo",
        category="NP",
        typeof=lambda _logical_form: "t",
        erase=lambda _category: "e",
        subsumes=lambda _general, _specific: None,
    )

    with pytest.raises(LawViolationError, match="does not subsume"):
        check_typing_laws(fixture)


# ---------------------------------------------------------------------------
# the square over hand-built items, against the native checker
# ---------------------------------------------------------------------------


LANGUAGE: Final = grammar_language(
    SignatureSpec(
        name="square-categories",
        primitives=("NP", "S"),
        constructors=tuple(
            ConstructorSpec(
                name=name,
                kind=("Cat", "Cat", "Cat"),
                notation=PrefixConstructorNotationSpec(spelling=name),
            )
            for name in ("fwd", "bwd")
        ),
    ),
    ("bo", "mary", "sleep"),
    types={"bo": "e", "mary": "e", "sleep": "e -> t"},
)
LF_STORE: Final = LANGUAGE.store
CHECKER: Final = LANGUAGE.checker
RULES: Final = LANGUAGE.rules
ENTITY: Final = LANGUAGE.read_type("e")
TRUTH: Final = LANGUAGE.read_type("t")
ARROW: Final = LF_STORE.signature.by_name(LANGUAGE.function_type)
ARROW_IMAGE: Final = LF_STORE.op(
    ARROW,
    LF_STORE.fvar("Ty", "a1"),
    LF_STORE.fvar("Ty", "a0"),
)
ERASURE: Final = Erasure(
    LF_STORE,
    {"NP": ENTITY, "S": TRUTH},
    {"fwd": ARROW_IMAGE, "bwd": ARROW_IMAGE},
    category_sort=LANGUAGE.category_sort,
    function_type=LANGUAGE.function_type,
)
NP: Final = LF_STORE.const(LANGUAGE.category_sort, "NP")
S: Final = LF_STORE.const(LANGUAGE.category_sort, "S")
VERB_PHRASE: Final = LF_STORE.op(LF_STORE.signature.by_name("bwd"), S, NP)
RAISED: Final = LF_STORE.op(LF_STORE.signature.by_name("fwd"), S, VERB_PHRASE)


def _arrow(domain: Term, codomain: Term) -> Term:
    return LF_STORE.op(ARROW, domain, codomain)


def _atom(name: str) -> int:
    atom = LANGUAGE.constant(name).payload

    assert atom is not None

    return atom


SIGNATURE: Final = LexicalSignature(
    {
        _atom("bo"): Scheme(ENTITY),
        _atom("mary"): Scheme(ENTITY),
        _atom("sleep"): Scheme(_arrow(ENTITY, TRUTH)),
    }
)


def _form(text: str) -> Term:
    return LANGUAGE.syntax.desugar(LANGUAGE.syntax.parse(text)).term


def _square(
    *items: CoherenceItem[Term],
    erase: Mapping[Term, Term] | None = None,
) -> CoherenceReport[Term]:
    return coherence_square(
        CoherenceFixture(
            items,
            CHECKER,
            SIGNATURE,
            RULES,
            ERASURE if erase is None else erase.__getitem__,
        )
    )


def test_the_square_commutes_on_an_equality_and_on_a_strict_instance() -> None:
    report = _square(
        CoherenceItem((0, 1), NP, (_form("bo"), _form("mary"))),
        CoherenceItem((0, 1), RAISED, (_form("\\f. f(bo)"),)),
        CoherenceItem((1, 2), VERB_PHRASE, (_form("sleep"),)),
    )
    plain, other, raised, verb = report.instances

    assert report.item_count == 3
    assert len(report.instances) == 4
    assert other.logical_form is _form("mary")
    assert not other.strict
    assert plain.inferred is ENTITY
    assert plain.erased is ENTITY
    assert not plain.strict
    assert not plain.substitution
    assert raised.erased is _arrow(_arrow(ENTITY, TRUTH), TRUTH)
    assert raised.inferred is not raised.erased
    assert raised.inferred.free
    assert raised.strict
    assert set(raised.substitution.values()) == {TRUTH}
    assert not verb.strict
    assert report.strict == (raised,)
    assert report.strict_items == {((0, 1), RAISED, "")}


def test_items_packed_on_one_span_and_category_are_told_apart_by_tag() -> None:
    report = _square(
        CoherenceItem((0, 1), RAISED, (_form("\\f. f(bo)"),), "tr"),
        CoherenceItem((0, 1), RAISED, (_form("\\f. f(mary)"),), "fc"),
        CoherenceItem((0, 1), RAISED, (_form("\\f. f(bo)"),)),
    )

    assert [instance.tag for instance in report.instances] == ["tr", "fc", ""]
    assert report.strict_items == {
        ((0, 1), RAISED, "tr"),
        ((0, 1), RAISED, "fc"),
        ((0, 1), RAISED, ""),
    }


def test_a_tagged_item_that_fails_the_square_is_named_with_its_tag() -> None:
    with pytest.raises(LawViolationError, match=r"\(0, 1\) .* 'tr' reads no logical"):
        _square(CoherenceItem((0, 1), NP, (), "tr"))


def test_a_strict_instance_is_one_the_erased_type_does_not_subsume_back() -> None:
    (raised,) = _square(CoherenceItem((0, 1), RAISED, (_form("\\f. f(bo)"),))).instances

    assert CHECKER.subsumes(raised.inferred, raised.erased) is not None
    assert CHECKER.subsumes(raised.erased, raised.inferred) is None


def test_alpha_renamed_principal_types_are_not_strict_instances() -> None:
    renamed = LF_STORE.fvar(LANGUAGE.type_sort, "renamed")
    erased = _arrow(renamed, renamed)
    (instance,) = _square(
        CoherenceItem((0, 1), NP, (_form("\\x. x"),)),
        erase={NP: erased},
    ).instances

    assert instance.inferred is not instance.erased
    assert CHECKER.subsumes(instance.inferred, instance.erased) is not None
    assert CHECKER.subsumes(instance.erased, instance.inferred) is not None
    assert not instance.strict


def test_the_square_matches_one_way_so_a_more_general_erasure_fails() -> None:
    general = LF_STORE.fvar(LANGUAGE.type_sort, "alpha")

    with pytest.raises(LawViolationError, match="does not instantiate"):
        _square(
            CoherenceItem((0, 1), NP, (_form("bo"),)),
            erase={NP: general},
        )

    assert CHECKER.subsumes(general, ENTITY) is not None


def test_a_logical_form_the_erased_category_does_not_instantiate_fails() -> None:
    with pytest.raises(LawViolationError, match=r"\(0, 1\).*does not instantiate"):
        _square(CoherenceItem((0, 1), NP, (_form("sleep"),)))


def test_an_ill_typed_logical_form_fails_the_square() -> None:
    with pytest.raises(LawViolationError, match="not well typed"):
        _square(CoherenceItem((0, 1), S, (_form("bo(bo)"),)))

    rooted = CHECKER.check(_form("sleep(bo(bo))"), SIGNATURE, RULES)

    assert rooted.scheme is not None
    assert not rooted.is_well_typed

    with pytest.raises(LawViolationError, match="not well typed"):
        _square(CoherenceItem((0, 1), S, (_form("sleep(bo(bo))"),)))


def test_an_item_reading_no_logical_form_fails_the_square() -> None:
    with pytest.raises(LawViolationError, match="reads no logical form"):
        _square(CoherenceItem((0, 1), NP, ()))


def test_a_batch_with_no_item_checks_nothing_and_says_so() -> None:
    with pytest.raises(LawViolationError, match="no chart item was offered"):
        _square()


def test_every_failing_derivation_in_a_batch_is_named() -> None:
    with pytest.raises(LawViolationError) as caught:
        _square(
            CoherenceItem((0, 1), NP, (_form("bo"), _form("sleep"))),
            CoherenceItem((1, 2), S, (_form("bo(bo)"),)),
            CoherenceItem((2, 3), NP, ()),
        )

    lines = str(caught.value).splitlines()

    assert len(lines) == 3
    assert "does not instantiate" in lines[0]
    assert "not well typed" in lines[1]
    assert "reads no logical form" in lines[2]


# ---------------------------------------------------------------------------
# the square over the shipped grammar and the shipped language
# ---------------------------------------------------------------------------

# form, category and meaning, the meaning in the shipped surface syntax with
# de Bruijn indices. The verbs conclude the declarative feature, which the
# rules match under erasure and the semantic erasure forgets; the
# object-position quantifiers take the noun, the verb and the subject in that
# order; one adverb binds a variable its body never uses and the other repeats
# its proposition.
LEXICON: Final = (
    ("Bo", "NP", "bo"),
    ("Mary", "NP", "mary"),
    ("Kim", "NP", "kim"),
    ("sleeps", "S[dcl]\\NP", "sleep"),
    ("runs", "S[dcl]\\NP", "run"),
    ("likes", "(S[dcl]\\NP)/NP", "like"),
    (
        "sees",
        "(S[dcl]\\NP)/NP",
        (
            "\\object. \\subject. exists event. "
            "(see(event) & (agent(subject)(event) & theme(object)(event)))"
        ),
    ),
    ("surely", "S/S", "\\proposition. exists entity. proposition"),
    ("maybe", "S/S", "\\proposition. proposition & proposition"),
    (
        "every",
        "(S/(S\\NP))/N",
        "\\restrictor. \\scope. forall entity. (~restrictor(entity) | scope(entity))",
    ),
    (
        "some",
        "(S/(S\\NP))/N",
        "\\restrictor. \\scope. exists entity. (restrictor(entity) & scope(entity))",
    ),
    (
        "every",
        "((S\\NP)\\((S\\NP)/NP))/N",
        (
            "\\restrictor. \\relation. \\subject. forall object. "
            "(~restrictor(object) | relation(object)(subject))"
        ),
    ),
    (
        "some",
        "((S\\NP)\\((S\\NP)/NP))/N",
        (
            "\\restrictor. \\relation. \\subject. exists object. "
            "(restrictor(object) & relation(object)(subject))"
        ),
    ),
    ("dog", "N", "dog"),
    ("cat", "N", "cat"),
    ("big", "N/N", "\\predicate. \\entity. (big(entity) & predicate(entity))"),
)

SENTENCES: Final = (
    "Bo sleeps",
    "Mary runs",
    "surely Kim runs",
    "Bo likes Mary",
    "Mary sees Bo",
    "maybe Kim likes Kim",
    "every dog sleeps",
    "some cat runs",
    "every dog likes Mary",
    "some cat sees Bo",
    "Bo likes every cat",
    "Mary sees some dog",
    "every dog likes some cat",
    "some big dog sleeps",
    "Bo sees every big cat",
    "every big dog likes Kim",
    "some cat likes every big dog",
    "every big big dog sees Mary",
)


def _grammar() -> GrammarDecl:
    return GrammarDecl(
        name="square-ccg",
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
            name="combinatory",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
                CompositionSpec(direction="unrestricted", order=1, view="/"),
                CompositionSpec(direction="unrestricted", order=1, view="\\"),
                TypeRaisingSpec(view="/", targets=("S",), arguments=None),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, category, _ in LEXICON
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        normal_form=NoNormalFormSpec(),
    )


def _malformed() -> tuple[tuple[LanguageDecl, type[Exception]], ...]:
    standard = registered_language("neo-davidsonian")
    unknown_type = standard.with_(
        constants=(
            *standard.constants,
            ConstantSpec(name="broken", type="truth"),
        )
    )

    return ((unknown_type, UnknownConstantError),)


@cache
def _conformance() -> CoherenceSquareConformance:
    declaration = _grammar()
    types = {
        "agent": "e -> e -> t",
        "big": "e -> t",
        "bo": "e",
        "cat": "e -> t",
        "dog": "e -> t",
        "kim": "e",
        "like": "e -> e -> t",
        "mary": "e",
        "run": "e -> t",
        "see": "e -> t",
        "sleep": "e -> t",
        "theme": "e -> e -> t",
    }
    base = registered_language("neo-davidsonian")
    standing = {constant.name for constant in base.constants}
    declared_language = base.with_(
        bounds={
            **base.bounds,
            base.category_sort: base.bounds[base.category_sort].with_(depth=5),
        },
        constants=(
            *base.constants,
            *(
                ConstantSpec(name=name, type=type_)
                for name, type_ in types.items()
                if name not in standing
            ),
        ),
    )
    semantic = LanguageIso().extend(declared_language, declaration.signature.extend)
    grammar = GrammarIso().forward_language(declaration, semantic)
    entity = semantic.read_type("e")
    truth = semantic.read_type("t")
    predicate = semantic.read_type("e -> t")
    relation = semantic.read_type("e -> e -> t")
    schemes: dict[str, Term] = {
        "bo": entity,
        "mary": entity,
        "kim": entity,
        "sleep": predicate,
        "run": predicate,
        "like": relation,
        "see": predicate,
        "agent": relation,
        "theme": relation,
        "dog": predicate,
        "cat": predicate,
        "big": predicate,
    }
    entries = tuple(
        Entry(
            form,
            grammar.syntax.parse(grammar.store, category),
            semantic.syntax.desugar(semantic.syntax.parse(meaning)).term,
        )
        for form, category, meaning in LEXICON
    )

    return CoherenceSquareConformance(
        grammar,
        semantic,
        entries,
        MappingProxyType(
            {rule.name: cast("ProofSchema", rule) for rule in grammar.grammar.rules}
        ),
        schemes,
        # the conjunction marker appears in no entry, so its image is never read.
        {"S": truth, "NP": entity, "N": predicate, "conj": truth},
        {
            name: semantic.store.op(
                semantic.store.signature.by_name(semantic.function_type),
                semantic.store.fvar(semantic.type_sort, "a1"),
                semantic.store.fvar(semantic.type_sort, "a0"),
            )
            for name in ("/", "\\")
        },
        SentenceBatch(
            tuple(tuple(sentence.split()) for sentence in SENTENCES),
            items=375,
            instances=507,
            strict_items=237,
        ),
        SentenceBatch(
            (("Bo", "likes", "Mary"),), items=14, instances=16, strict_items=7
        ),
        _malformed(),
    )


class TestCoherenceSquare(CoherenceSquareSuite):
    """The square over the shipped CCG pack and the neo-Davidsonian language."""

    @pytest.fixture
    def coherence_conformance(self) -> CoherenceSquareConformance:
        return _conformance()


# ---------------------------------------------------------------------------
# the clamp projections
# ---------------------------------------------------------------------------


def test_the_shipped_projections_satisfy_every_projection_law() -> None:
    check_clamp_projection_laws(
        ClampProjectionFixture(CategoryProjection[str](), ("S", "NP"))
    )
    check_clamp_projection_laws(
        ClampProjectionFixture(PairProjection[str](), (("S", "m"), ("NP", "k")))
    )
    assert len(CLAMP_PROJECTION_LAWS) == 2


def test_a_projection_declaring_an_unread_meaning_breaks_the_law() -> None:
    fixture = ClampProjectionFixture(_MeaningLess(), ("S",))

    with pytest.raises(LawViolationError, match="declares has_meaning=True"):
        a_meaning_is_read_exactly_when_declared(fixture)


def test_a_projection_offered_no_reading_checks_nothing_and_says_so() -> None:
    with pytest.raises(LawViolationError, match="no reading was offered"):
        a_meaning_is_read_exactly_when_declared(
            ClampProjectionFixture(CategoryProjection[str](), ())
        )


def test_a_projection_reading_fresh_categories_breaks_the_law() -> None:
    fixture = ClampProjectionFixture(_Fresh(), ("S",))

    with pytest.raises(LawViolationError, match="reads two categories"):
        a_projection_reads_the_same_reading_the_same_way(fixture)
