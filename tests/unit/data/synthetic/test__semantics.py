"""Tests for the logical forms assigned to lexical items."""

from collections import Counter
from math import log
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.support.installed import entries, install
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data.synthetic import (
    Histogram,
    LexiconReport,
    LogicalClassContext,
    LogicalClassPlugin,
    Meaning,
    SemanticLexicon,
    SemanticReport,
    SyntheticGenerator,
    UniformLength,
    logical_classes,
)
from lofi.data.synthetic._semantics import assign, inventory
from lofi.deduce import ParsePolicy, Reachability, Retain
from lofi.lf import Application, Atom, Lambda
from lofi.spec import ApplicationSpec, RuleSetSpec
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    SignatureSpec,
)
from lofi.spec.language import LanguageIso, registered_language

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.data.synthetic import SyntheticCorpus
    from lofi.data.synthetic._semantics import (
        AssignmentRow,
        Inventory,
        LogicalClass,
    )
    from lofi.lf import Expression
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

QUANTIFIERS: Final = (("forall", None), ("exists", None))
CONNECTIVES: Final = ("and", "if", "iff", "not", "or")
STANDARD_LANGUAGE: Final = LanguageIso().forward(registered_language("neo-davidsonian"))
ENTRIES: Final = (
    ("w0", "NP"),
    ("w1", "NP"),
    ("w2", "S\\NP"),
    ("w3", "S\\NP"),
    ("w4", "(S\\NP)/NP"),
    ("w5", "NP/NP"),
    ("w5", "S\\NP"),
)


def _uniform(count: int) -> tuple[float, ...]:
    return (1.0,) * count


def _inventory(**sizes: int) -> Inventory:
    return inventory(
        tuple(
            (logical_class, size, _uniform(size))
            for logical_class, size in sizes.items()
        ),
        language=STANDARD_LANGUAGE,
        quantifiers=QUANTIFIERS,
        connectives=CONNECTIVES,
        arities=((1, 2), (2, 1)) if sizes.get("predicate") == 3 else (),
    )


def _grammar() -> GrammarDecl:
    return GrammarDecl(
        name="toy",
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
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, category in ENTRIES
            )
        ),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )


@pytest.fixture(scope="module")
def assembly() -> GrammarAssembly:
    return GrammarIso().forward(_grammar(), registered_language("propositional"))


@pytest.fixture(scope="module")
def corpus(assembly: GrammarAssembly) -> SyntheticCorpus:
    return SyntheticGenerator(
        assembly, policy=ParsePolicy(Retain.FOREST, Reachability.ALL)
    ).sample(60, seed=3, length=UniformLength(2, 4))


def _render(
    assembly: GrammarAssembly,
) -> dict[Term, str]:
    lexicon = assembly.lexicon

    return {
        entry.category: assembly.syntax.render(assembly.store, entry.category)
        for form in lexicon.forms
        for entry in lexicon.entries(form)
    }


def _category(assembly: GrammarAssembly, text: str, /) -> str:
    """Return category text rendered by the fixture grammar's syntax."""
    category = assembly.syntax.parse(assembly.store, text)

    return assembly.syntax.render(assembly.store, category)


def _rows(
    categories: Mapping[str, tuple[AssignmentRow, ...]] | None = None,
    *,
    default: tuple[AssignmentRow, ...] = (("predicate", None, 1.0),),
) -> dict[str, tuple[AssignmentRow, ...]]:
    held = dict.fromkeys((category for _, category in ENTRIES), default)
    held.update(categories or {})

    return held


def _assigned(
    assembly: GrammarAssembly,
    rows: Mapping[str, tuple[AssignmentRow, ...]],
    forms: Inventory | None = None,
    shares: Mapping[LogicalClass, float] | None = None,
    seed: int = 1,
) -> SemanticLexicon:
    held = _inventory(predicate=3, constant=2, quantifier=2) if forms is None else forms

    return assign(
        assembly.lexicon,
        _render(assembly),
        held,
        {logical_class: 1.0 / len(held) for logical_class in held}
        if shares is None
        else shares,
        {_category(assembly, category): row for category, row in rows.items()},
        seed=seed,
    )


def _leaf_counts(corpus: SyntheticCorpus) -> Counter[tuple[str, str]]:
    counts: Counter[tuple[str, str]] = Counter()

    for example in corpus.examples:
        surfaces = [token.surface for token in example.item.tokens]

        for leaf in example.supervision.derivation.leaves():
            counts[leaf.label, surfaces[leaf.span.start]] += 1

    return counts


def _lambda_depth(expression: Expression) -> int:
    depth = 0

    while isinstance(expression, Lambda):
        depth += 1
        expression = expression.body

    return depth


def _atoms(expression: Expression) -> set[str]:
    if isinstance(expression, Lambda):
        return _atoms(expression.body)

    if isinstance(expression, Application):
        return _atoms(expression.function) | _atoms(expression.argument)

    return {expression.name}


SYNTAX = STANDARD_LANGUAGE.syntax


def test_a_meaning_holds_its_class_symbol_arity_and_text() -> None:
    held = Meaning("predicate", "p1", arity=2, text="λx2.λx1.p1(x1)(x2)")

    assert (held.logical_class, held.symbol, held.arity) == ("predicate", "p1", 2)
    assert held == Meaning("predicate", "p1", arity=2, text="λx2.λx1.p1(x1)(x2)")
    assert hash(held) == hash(Meaning("predicate", "p1", arity=2, text=held.text))
    assert held != Meaning("predicate", "p1", arity=1, text="λx1.p1(x1)")
    assert repr(held) == (
        "Meaning('predicate', 'p1', arity=2, text='λx2.λx1.p1(x1)(x2)')"
    )
    assert Meaning("adverb", "quickly", text="quickly").logical_class == "adverb"


def test_a_registered_logical_class_builds_its_own_inventory(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    def build(size: int, context: LogicalClassContext, /) -> tuple[Meaning, ...]:
        assert context.quantifiers == QUANTIFIERS
        return tuple(
            Meaning("modal", f"m{index}", text=f"m{index}")
            for index in range(1, size + 1)
        )

    plugin = LogicalClassPlugin("modal", 5, build)

    install(monkeypatch, "lf_class", entries("lf_class", {"modal": plugin}))

    held = inventory(
        (("modal", 2, _uniform(2)),),
        language=STANDARD_LANGUAGE,
        quantifiers=QUANTIFIERS,
        connectives=CONNECTIVES,
    )

    assert logical_classes() == ("modal",)
    assert tuple(form.symbol for form, _ in held["modal"]) == ("m1", "m2")
    assert Meaning("constant", "c1", text="c1").arity is None


@pytest.mark.parametrize(
    ("logical_class", "symbol", "arity", "text", "error"),
    [
        ("", "a", None, "a", ValueError),
        ("predicate", "p1", 0, "p1", ValueError),
        ("predicate", "p1", 1.5, "p1", TypeError),
        ("constant", "", None, "c1", ValueError),
        ("identity", "λx.x", None, "", ValueError),
    ],
)
def test_a_meaning_refuses_a_malformed_class_arity_or_name(
    logical_class: str,
    symbol: str,
    arity: float | None,
    text: str,
    error: type[Exception],
) -> None:
    with pytest.raises(error):
        Meaning(
            logical_class,
            symbol,
            arity=cast("int | None", arity),
            text=text,
        )


def test_the_inventory_builds_one_parseable_form_per_class_and_rank() -> None:
    held = _inventory(quantifier=2, connective=5, predicate=3, identity=1, constant=2)

    assert [form.symbol for form, _ in held["quantifier"]] == ["forall", "exists"]
    assert [form.symbol for form, _ in held["connective"]] == [
        "and",
        "if",
        "iff",
        "not",
        "or",
    ]
    assert [form.arity for form, _ in held["connective"]] == [2, 2, 2, 1, 2]
    assert [(form.symbol, form.arity) for form, _ in held["predicate"]] == [
        ("p1", 1),
        ("p2", 1),
        ("p3", 2),
    ]
    assert [form.text for form, _ in held["identity"]] == ["λx.x"]
    assert [form.text for form, _ in held["constant"]] == ["c1", "c2"]

    for logical_class, forms in held.items():
        for form, weight in forms:
            parsed = SYNTAX.parse(form.text)

            assert weight == 1.0
            assert form.logical_class == logical_class
            assert (
                _lambda_depth(parsed)
                == {
                    "quantifier": 2,
                    "connective": 1 if form.symbol == "not" else 2,
                    "predicate": form.arity,
                    "identity": 1,
                    "constant": 0,
                }[logical_class]
            )

    binders = {
        form.symbol: _atoms(SYNTAX.parse(form.text)) for form, _ in held["quantifier"]
    }

    assert binders == {
        "exists": {"P", "Q", "and", "exists", "x"},
        "forall": {"P", "Q", "if", "forall", "x"},
    }
    assert SYNTAX.parse(held["connective"][3][0].text) == Lambda(
        "p", None, Application(Atom("not"), Atom("p"))
    )
    assert SYNTAX.parse(held["predicate"][2][0].text) == Lambda(
        "x2",
        None,
        Lambda(
            "x1",
            None,
            Application(
                Application(Atom("p3"), Atom("x1")),
                Atom("x2"),
            ),
        ),
    )


def test_a_generalized_quantifier_applies_as_a_constant() -> None:
    held = inventory(
        (("quantifier", 3, _uniform(3)),),
        language=STANDARD_LANGUAGE,
        quantifiers=(*QUANTIFIERS, ("most", None)),
        connectives=CONNECTIVES,
    )
    most = next(form for form, _ in held["quantifier"] if form.symbol == "most")

    assert SYNTAX.parse(most.text) == Lambda(
        "P",
        None,
        Lambda(
            "Q",
            None,
            Application(
                Application(Atom("most"), Atom("P")),
                Atom("Q"),
            ),
        ),
    )


def test_quantifier_size_one_selects_the_first_declared_template() -> None:
    held = inventory(
        (("quantifier", 1, (1.0,)),),
        language=STANDARD_LANGUAGE,
        quantifiers=(
            ("exists", "λP.λQ.exists x.and(Q(x))(P(x))"),
            ("forall", "λP.λQ.forall x.if(P(x))(Q(x))"),
        ),
        connectives=CONNECTIVES,
    )

    selected = held["quantifier"][0][0]
    assert selected.symbol == "exists"
    assert selected.text == "λP.λQ.exists x.and(Q(x))(P(x))"


def test_the_inventory_keeps_the_rank_weights_it_is_given() -> None:
    held = inventory(
        (("constant", 3, (3.0, 1.5, 0.5)),),
        language=STANDARD_LANGUAGE,
        quantifiers=(),
        connectives=(),
    )

    assert [weight for _, weight in held["constant"]] == [3.0, 1.5, 0.5]


def test_a_connective_form_reads_its_arity_from_the_run_language() -> None:
    source = registered_language("propositional")
    language = LanguageIso().forward(
        source.with_(
            constants=tuple(
                constant.with_(type="t -> t -> t -> t")
                if constant.name == "and"
                else constant
                for constant in source.constants
            )
        )
    )
    held = inventory(
        (("connective", 1, (1.0,)),),
        language=language,
        quantifiers=(),
        connectives=("and",),
    )

    connective = held["connective"][0][0]

    assert connective.arity == 3
    assert connective.text == "λp.λq.λr.and(p)(q)(r)"


@pytest.mark.parametrize(
    ("classes", "quantifiers", "connectives", "arities", "fragment"),
    [
        ((("quantifier", 3, _uniform(3)),), QUANTIFIERS, CONNECTIVES, (), "3 but"),
        ((("quantifier", 1, _uniform(1)),), (), CONNECTIVES, (), "0 quant"),
        ((("connective", 6, _uniform(6)),), QUANTIFIERS, CONNECTIVES, (), "6 but"),
        (
            (("quantifier", 2, _uniform(2)),),
            QUANTIFIERS,
            frozenset(("and",)),
            (),
            "restricts by the connective 'if'",
        ),
        ((("identity", 2, _uniform(2)),), QUANTIFIERS, CONNECTIVES, (), "one form"),
        (
            (("predicate", 3, _uniform(3)),),
            QUANTIFIERS,
            CONNECTIVES,
            ((1, 1), (2, 1)),
            "places 2 forms",
        ),
        (
            (("predicate", 2, _uniform(2)),),
            QUANTIFIERS,
            CONNECTIVES,
            ((0, 2),),
            "not positive",
        ),
        ((("constant", 2, _uniform(3)),), QUANTIFIERS, CONNECTIVES, (), "3 rank"),
        (
            (("constant", 1, _uniform(1)), ("constant", 1, _uniform(1))),
            QUANTIFIERS,
            CONNECTIVES,
            (),
            "twice",
        ),
    ],
)
def test_the_inventory_refuses_what_the_fragment_or_the_rows_cannot_bear(
    classes: tuple[tuple[LogicalClass, int, tuple[float, ...]], ...],
    quantifiers: tuple[tuple[str, str | None], ...],
    connectives: tuple[str, ...],
    arities: tuple[tuple[int, int], ...],
    fragment: str,
) -> None:
    with pytest.raises(ValueError, match=fragment):
        inventory(
            classes,
            language=STANDARD_LANGUAGE,
            quantifiers=quantifiers,
            connectives=connectives,
            arities=arities,
        )


def test_every_item_draws_a_form_of_a_class_its_category_is_assigned(
    assembly: GrammarAssembly,
) -> None:
    rows = _rows(
        {
            "NP": (("constant", None, 1.0),),
            "S\\NP": (("predicate", 1, 1.0),),
            "(S\\NP)/NP": (("predicate", 2, 1.0),),
            "NP/NP": (("quantifier", None, 1.0),),
        }
    )
    held = _assigned(assembly, rows)

    assert set(held.meanings) == {
        (_category(assembly, category), form) for form, category in ENTRIES
    }
    assert len(held) == len(ENTRIES)
    assert {
        item: meanings[0].logical_class for item, meanings in held.meanings.items()
    } == {
        ("NP", "w0"): "constant",
        ("NP", "w1"): "constant",
        ("S\\NP", "w2"): "predicate",
        ("S\\NP", "w3"): "predicate",
        (_category(assembly, "(S\\NP)/NP"), "w4"): "predicate",
        ("NP/NP", "w5"): "quantifier",
        ("S\\NP", "w5"): "predicate",
    }
    assert held[_category(assembly, "(S\\NP)/NP"), "w4"][0].arity == 2
    assert {held["S\\NP", form][0].arity for form in ("w2", "w3", "w5")} == {1}
    assert held.sizes == {"predicate": 3, "constant": 2, "quantifier": 2}


def test_the_same_seed_draws_the_same_assignment_and_another_varies_it(
    assembly: GrammarAssembly,
) -> None:
    rows = _rows(default=tuple((held, None, 1.0) for held in logical_classes()))
    forms = _inventory(quantifier=2, connective=5, predicate=3, identity=1, constant=2)
    first = _assigned(assembly, rows, forms, seed=7)

    assert first == _assigned(assembly, rows, forms, seed=7)
    assert hash(first) == hash(_assigned(assembly, rows, forms, seed=7))
    assert any(
        _assigned(assembly, rows, forms, seed=seed) != first for seed in range(8, 14)
    )


def test_a_class_share_weighs_the_rows_a_category_may_draw(
    assembly: GrammarAssembly,
) -> None:
    rows = _rows(default=(("constant", None, 1.0), ("predicate", None, 1.0)))
    forms = _inventory(predicate=3, constant=2)
    drawn = Counter(
        meaning.logical_class
        for seed in range(30)
        for meanings in _assigned(
            assembly, rows, forms, {"predicate": 1e-9, "constant": 1.0}, seed=seed
        ).meanings.values()
        for meaning in meanings
    )

    assert drawn == {"constant": 30 * len(ENTRIES)}

    balanced = Counter(
        meaning.logical_class
        for seed in range(30)
        for meanings in _assigned(
            assembly, rows, forms, {"predicate": 0.5, "constant": 0.5}, seed=seed
        ).meanings.values()
        for meaning in meanings
    )

    assert balanced["constant"] > 0
    assert balanced["predicate"] > 0


def test_a_row_weight_tilts_the_draw_within_a_category(
    assembly: GrammarAssembly,
) -> None:
    rows = _rows(default=(("constant", None, 1e-9), ("predicate", None, 1.0)))
    forms = _inventory(predicate=3, constant=2)
    drawn = Counter(
        meaning.logical_class
        for seed in range(20)
        for meanings in _assigned(
            assembly, rows, forms, {"predicate": 0.5, "constant": 0.5}, seed=seed
        ).meanings.values()
        for meaning in meanings
    )

    assert drawn == {"predicate": 20 * len(ENTRIES)}


def test_an_assignment_row_can_select_connectives_by_arity(
    assembly: GrammarAssembly,
) -> None:
    rows = _rows(default=(("connective", 1, 1.0),))
    forms = _inventory(connective=5)
    held = _assigned(
        assembly,
        rows,
        forms,
        {"connective": 1.0},
        seed=0,
    )

    assert {
        (meaning.symbol, meaning.arity)
        for meanings in held.meanings.values()
        for meaning in meanings
    } == {("not", 1)}


@pytest.mark.parametrize(
    ("rows", "match"),
    [
        (
            {category: (("predicate", None, 1.0),) for _, category in ENTRIES[:2]},
            "no row for category",
        ),
        (_rows(default=(("identity", None, 1.0),)), "does not declare"),
        (_rows(default=(("predicate", 3, 1.0),)), "forces arity 3"),
    ],
)
def test_the_draw_refuses_a_category_without_a_usable_row(
    assembly: GrammarAssembly,
    rows: Mapping[str, tuple[AssignmentRow, ...]],
    match: str,
) -> None:
    with pytest.raises(ValueError, match=match):
        _assigned(assembly, rows)


def test_the_draw_refuses_a_category_without_a_rendering(
    assembly: GrammarAssembly,
) -> None:
    with pytest.raises(ValueError, match="no rendering"):
        assign(
            assembly.lexicon,
            {},
            _inventory(predicate=3),
            {"predicate": 1.0},
            _rows(),
            seed=1,
        )


def test_the_lexicon_refuses_a_form_outside_its_inventory() -> None:
    forms = {
        logical_class: tuple(form for form, _ in held)
        for logical_class, held in _inventory(predicate=3, constant=2).items()
    }
    stray = Meaning("constant", "c9", text="c9")

    with pytest.raises(ValueError, match="outside the constant inventory"):
        SemanticLexicon(
            {("NP", "w0"): (stray,)},
            forms,
            {"predicate": 0.5, "constant": 0.5},
        )

    with pytest.raises(ValueError, match="classes with shares"):
        SemanticLexicon({}, forms, {"predicate": 1.0})

    held = SemanticLexicon({}, forms, {"predicate": 0.5, "constant": 0.5})

    assert len(held) == 0
    assert held.sizes == {"predicate": 3, "constant": 2}
    assert (
        repr(held)
        == "SemanticLexicon(items=0, inventory={'predicate': 3, 'constant': 2})"
    )


def test_the_report_counts_every_leaf_of_every_gold_derivation(
    assembly: GrammarAssembly, corpus: SyntheticCorpus
) -> None:
    rows = _rows(
        {
            "NP": (("constant", None, 1.0),),
            "NP/NP": (("quantifier", None, 1.0),),
            "S\\NP": (("predicate", 1, 1.0),),
            "(S\\NP)/NP": (("predicate", 2, 1.0),),
        }
    )
    lexicon = _assigned(assembly, rows)
    report = SemanticReport.of(lexicon, (corpus,))
    leaves = _leaf_counts(corpus)
    by_form: Counter[Meaning] = Counter()

    for item, count in leaves.items():
        for meaning in lexicon[item]:
            by_form[meaning] += count

    assert report.lexicon is lexicon
    assert sum(report.joint.token_counts.values()) == sum(leaves.values()) > 0
    assert report.joint.class_sizes == {"predicate": 3, "constant": 2, "quantifier": 2}
    assert report.joint.forms == Histogram.of(
        by_form[form] for forms in lexicon.inventory.values() for form in forms
    )
    assert report.joint.classes["constant"] == Histogram.of(
        by_form[form] for form in lexicon.inventory["constant"]
    )
    assert report.arities == Histogram({1: 3, 2: 1})
    assert report.cells == {
        _category(assembly, "NP"): {"constant": 2},
        _category(assembly, "NP/NP"): {"quantifier": 1},
        _category(assembly, "S\\NP"): {"predicate": 3},
        _category(assembly, "(S\\NP)/NP"): {"predicate": 1},
    }
    assert report.items == {"predicate": 4, "constant": 2, "quantifier": 1}
    assert report.item_shares == {
        "predicate": 4 / 7,
        "constant": 2 / 7,
        "quantifier": 1 / 7,
    }
    assert report == SemanticReport.of(lexicon, (corpus,))
    assert hash(report) == hash(SemanticReport.of(lexicon, (corpus,)))
    assert repr(report).startswith("SemanticReport(SemanticLexicon(items=7")

    joint = SemanticReport.of(lexicon, (corpus, corpus))
    assert joint.joint.token_counts == {
        logical_class: count * 2
        for logical_class, count in report.joint.token_counts.items()
    }


def test_the_report_requires_at_least_one_corpus(
    assembly: GrammarAssembly,
) -> None:
    with pytest.raises(ValueError, match="at least one corpus"):
        SemanticReport.of(_assigned(assembly, _rows()), ())


def test_the_concordance_is_one_when_the_category_fixes_the_class(
    assembly: GrammarAssembly, corpus: SyntheticCorpus
) -> None:
    rows = _rows(
        {
            "NP": (("constant", None, 1.0),),
            "NP/NP": (("constant", None, 1.0),),
            "S\\NP": (("predicate", None, 1.0),),
            "(S\\NP)/NP": (("quantifier", None, 1.0),),
        }
    )
    report = SemanticReport.of(_assigned(assembly, rows), (corpus,))

    assert report.concordance == pytest.approx(1.0)


def test_the_concordance_is_zero_when_the_class_ignores_the_category(
    assembly: GrammarAssembly, corpus: SyntheticCorpus
) -> None:
    forms = _inventory(predicate=3, constant=2)
    both = (("predicate", None, 1.0), ("constant", None, 1.0))
    lexicon = SemanticLexicon(
        {
            ("NP", "w0"): (forms["predicate"][0][0],),
            ("NP", "w1"): (forms["constant"][0][0],),
            ("S\\NP", "w2"): (forms["predicate"][1][0],),
            ("S\\NP", "w3"): (forms["constant"][1][0],),
        },
        {held: tuple(form for form, _ in forms[held]) for held in forms},
        {"predicate": 0.5, "constant": 0.5},
    )
    report = SemanticReport(
        lexicon,
        LexiconReport(
            {"predicate": Histogram({0: 3}), "constant": Histogram({0: 2})},
            Histogram({0: 5}),
            Histogram({1: 5}),
            None,
        ),
        Histogram({1: 2}),
        {
            "NP": {"predicate": 1, "constant": 1},
            "S\\NP": {"predicate": 1, "constant": 1},
        },
    )

    assert report.concordance == pytest.approx(0.0)
    assert (
        SemanticReport.of(
            _assigned(assembly, _rows(default=both)), (corpus,)
        ).concordance
        is not None
    )


def test_the_concordance_is_the_information_over_the_smaller_entropy() -> None:
    forms = _inventory(predicate=3, constant=2)
    lexicon = SemanticLexicon(
        {
            ("NP", "w0"): (forms["predicate"][0][0],),
            ("NP", "w1"): (forms["predicate"][1][0],),
            ("S\\NP", "w2"): (forms["constant"][0][0],),
            ("(S\\NP)/NP", "w4"): (forms["constant"][1][0],),
        },
        {held: tuple(form for form, _ in forms[held]) for held in forms},
        {"predicate": 0.5, "constant": 0.5},
    )
    report = SemanticReport(
        lexicon,
        LexiconReport(
            {"predicate": Histogram({0: 3}), "constant": Histogram({0: 2})},
            Histogram({0: 5}),
            Histogram({1: 5}),
            None,
        ),
        Histogram({1: 2}),
        {
            "NP": {"predicate": 2},
            "S\\NP": {"constant": 1},
            "(S\\NP)/NP": {"constant": 1},
        },
    )
    class_entropy = -(0.5 * log(0.5) + 0.5 * log(0.5))
    category_entropy = -(0.5 * log(0.5) + 2 * 0.25 * log(0.25))

    assert class_entropy < category_entropy
    assert report.concordance == pytest.approx(1.0)


def test_the_concordance_is_absent_without_variation() -> None:
    forms = _inventory(predicate=3)
    inventory_forms: dict[LogicalClass, tuple[Meaning, ...]] = {
        "predicate": tuple(form for form, _ in forms["predicate"])
    }
    lexicon = SemanticLexicon(
        {("NP", "w0"): (forms["predicate"][0][0],)},
        inventory_forms,
        {"predicate": 1.0},
    )
    joint = LexiconReport(
        {"predicate": Histogram({0: 3})},
        Histogram({0: 3}),
        Histogram({1: 3}),
        None,
    )

    assert (
        SemanticReport(
            lexicon, joint, Histogram({}), {"NP": {"predicate": 1}}
        ).concordance
        is None
    )

    empty = SemanticLexicon({}, inventory_forms, {"predicate": 1.0})

    assert SemanticReport(empty, joint, Histogram({}), {}).concordance is None


def test_the_report_refuses_a_leaf_the_assignment_does_not_cover(
    assembly: GrammarAssembly, corpus: SyntheticCorpus
) -> None:
    full = _assigned(assembly, _rows())
    partial = SemanticLexicon(
        {item: meaning for item, meaning in full.meanings.items() if item[1] != "w0"},
        full.inventory,
        full.shares,
    )

    with pytest.raises(ValueError, match="'w0' as 'NP' no form"):
        SemanticReport.of(partial, (corpus,))


def test_the_report_refuses_cells_that_disagree_with_the_lexicon(
    assembly: GrammarAssembly, corpus: SyntheticCorpus
) -> None:
    lexicon = _assigned(assembly, _rows())
    measured = SemanticReport.of(lexicon, (corpus,))

    with pytest.raises(ValueError, match="count 1 meanings but the lexicon holds 7"):
        SemanticReport(
            lexicon, measured.joint, measured.arities, {"NP": {"predicate": 1}}
        )

    with pytest.raises(ValueError, match="outside the inventory"):
        SemanticReport(
            lexicon,
            measured.joint,
            measured.arities,
            {"NP": {"identity": len(lexicon)}},
        )

    with pytest.raises(ValueError, match="joint reports classes outside"):
        SemanticReport(
            lexicon,
            LexiconReport(
                {"identity": Histogram({0: 1})},
                Histogram({0: 1}),
                Histogram({1: 1}),
                None,
            ),
            measured.arities,
            measured.cells,
        )
