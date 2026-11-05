from copy import copy, deepcopy
from typing import Final

import pytest
from tests.language import language, lexicon

from lofi._types import ItemId
from lofi.terms import (
    NonTerminatingError,
    Normalization,
    Term,
    TreeEditDistance,
)
from lofi.typecheck import (
    EvaluationItem,
    EvaluationReport,
    Scheme,
    TypeAgreement,
    evaluation_report,
)

VOCABULARY: Final = (
    "bo",
    "father",
    "jo",
    "run",
    "see",
    "sleep",
    "tall",
)


LANGUAGE: Final = language(VOCABULARY)
STORE: Final = LANGUAGE.store
CHECKER: Final = LANGUAGE.checker
RULES: Final = LANGUAGE.rules
CALCULUS: Final = LANGUAGE.calculus
ARROW: Final = STORE.signature.by_name("->")
ENTITY: Final = LANGUAGE.read_type("e")
TRUTH: Final = LANGUAGE.read_type("t")
PREDICATE: Final = LANGUAGE.read_type("e -> t")
RELATION: Final = LANGUAGE.read_type("e -> e -> t")
FUNCTION: Final = LANGUAGE.read_type("e -> e")


def form(source: str) -> Term:
    """Parse and desugar one logical form into the shared store.

    Parameters
    ----------
    source : str
        Logical-form text.
    """
    return LANGUAGE.syntax.desugar(LANGUAGE.syntax.parse(source)).term


def atom(name: str) -> int:
    """Return the constant atom of one symbol.

    Parameters
    ----------
    name : str
        Symbol name.
    """
    payload = form(name).payload

    if payload is None:
        msg = f"{name!r} is not a constant"
        raise ValueError(msg)

    return payload


SIGNATURE: Final = lexicon(
    LANGUAGE,
    {
        atom("bo"): Scheme(ENTITY),
        atom("jo"): Scheme(ENTITY),
        atom("sleep"): Scheme(PREDICATE),
        atom("run"): Scheme(PREDICATE),
        atom("tall"): Scheme(PREDICATE),
        atom("see"): Scheme(RELATION),
        atom("father"): Scheme(FUNCTION),
    },
)

GOLDEN_PAIRS: Final[tuple[tuple[str, str], ...]] = (
    ("λx.sleep(x)", "λy.sleep(y)"),
    ("sleep(bo)", "sleep(bo)"),
    ("sleep(bo)", "(λx.sleep(x))(bo)"),
    ("λx.λy.see(x)(y)", "λa.λb.see(a)(b)"),
    ("λx.λy.see(x)(y)", "λx.λy.see(y)(x)"),
    ("λx.x", "λy.y"),
    ("λx.father(x)", "λx.x"),
    ("λx.x", "λx.father(x)"),
    ("sleep(bo)", "run(bo)"),
    ("sleep(bo)", "sleep(jo)"),
    ("sleep(bo) ∧ run(jo)", "run(jo) ∧ sleep(bo)"),
    ("¬sleep(bo)", "sleep(bo)"),
    ("∃x.sleep(x)", "∃y.sleep(y)"),
    ("∃x.sleep(x)", "∀x.sleep(x)"),
    ("sleep(bo)", "sleep(sleep)"),
    ("sleep(sleep)", "sleep(bo)"),
    ("bo", "jo"),
    ("bo", "sleep(bo)"),
    ("λx.sleep(x)", "sleep"),
    ("λx.see(x)(bo)", "λy.see(y)(bo)"),
    ("λx.see(bo)(x)", "λx.see(x)(bo)"),
    ("father(father(bo))", "father(bo)"),
    ("(λx.λy.see(x)(y))(bo)(jo)", "see(bo)(jo)"),
    ("(λf.f(bo))(sleep)", "sleep(bo)"),
    ("λf.f(bo)", "λg.g(bo)"),
    ("λf.f(bo)", "λf.f(jo)"),
    ("λf.f(bo)", "λf.sleep(bo)"),
    ("λf.sleep(bo)", "λf.f(bo)"),
    ("∃x.see(x)(bo)", "∃x.see(bo)(x)"),
    ("sleep(bo) ⇒ run(bo)", "run(bo) ⇒ sleep(bo)"),
    ("sleep(bo) ⇔ run(bo)", "sleep(bo) ∧ run(bo)"),
    ("bo = jo", "jo = bo"),
    ("bo = jo", "bo = bo"),
    ("λx.λy.x", "λa.λb.a"),
    ("λx.λy.x", "λx.λy.y"),
    ("λx.λy.x", "λx.λy.bo"),
    ("λx.λy.bo", "λx.λy.x"),
    ("λx.λy.bo", "λx.λy.jo"),
    ("sleep(bo) ∧ run(jo)", "sleep(bo)"),
    ("λx.sleep(x) ∧ run(x)", "λx.run(x) ∧ sleep(x)"),
    ("λx.sleep(x) ∧ run(x)", "λx.sleep(x)"),
    ("λx.(λy.sleep(y))(x)", "λz.sleep(z)"),
    ("λx.see(x)", "λx.λy.see(x)(y)"),
    ("tall(bo)", "sleep(bo)"),
    ("tall(father(bo))", "tall(bo)"),
    ("∃x.sleep(x) ∧ tall(x)", "∃x.tall(x) ∧ sleep(x)"),
    ("∃x.sleep(x) ∧ tall(x)", "∃x.sleep(x)"),
    ("λx.x", "λx.λy.x"),
    ("see(bo)(jo)", "see(jo)(bo)"),
    ("father(bo)", "bo"),
)


def report(
    pairs: dict[ItemId, tuple[Term, Term]],
    *,
    agreement: TypeAgreement = TypeAgreement.SUBSUMES,
    fuel: int = 10_000,
) -> EvaluationReport:
    """Evaluate gold and predicted forms keyed by item id.

    Parameters
    ----------
    pairs : dict[ItemId, tuple[Term, Term]]
        Gold and predicted form of each item.
    fuel : int
        Reduction steps allowed per normalization.
    """
    gold = {item_id: pair[0] for item_id, pair in pairs.items()}
    predicted = {item_id: pair[1] for item_id, pair in pairs.items()}

    return evaluation_report(
        gold,
        predicted,
        CHECKER,
        SIGNATURE,
        RULES,
        STORE,
        CALCULUS,
        agreement=agreement,
        normalization=Normalization(
            fuel=fuel,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
    )


def single(gold: str, predicted: str) -> EvaluationItem:
    """Evaluate one gold and predicted pair.

    Parameters
    ----------
    gold : str
        Gold logical-form text.
    predicted : str
        Predicted logical-form text.
    """
    found = report({ItemId("only"): (form(gold), form(predicted))})

    return found.items[0]


def golden_pairs() -> dict[ItemId, tuple[Term, Term]]:
    """Build the fifty-pair golden table keyed by position."""
    return {
        ItemId(f"item-{position:02d}"): (form(gold), form(predicted))
        for position, (gold, predicted) in enumerate(GOLDEN_PAIRS)
    }


def test_alpha_variant_forms_are_an_exact_match() -> None:
    item = single("λx.λy.see(x)(y)", "λa.λb.see(a)(b)")

    assert item.exact
    assert item.gold is item.predicted
    assert item.distance == 0
    assert item.agrees


def test_beta_reducible_forms_normalize_before_comparison() -> None:
    item = single("see(bo)(jo)", "(λx.λy.see(x)(y))(bo)(jo)")

    assert item.exact
    assert item.predicted is form("see(bo)(jo)")
    assert item.distance == 0


def test_non_variant_forms_are_not_an_exact_match() -> None:
    item = single("λx.λy.see(x)(y)", "λx.λy.see(y)(x)")

    assert not item.exact
    assert item.distance > 0
    assert item.agrees


def test_a_strictly_more_general_prediction_agrees_in_type() -> None:
    item = single("λx.father(x)", "λx.x")

    assert item.agrees
    assert not item.exact
    assert item.gold_type is FUNCTION
    assert item.predicted_type is not None
    assert CHECKER.subsumes(item.predicted_type, FUNCTION) is not None


def test_a_strictly_more_specific_prediction_does_not_agree_in_type() -> None:
    item = single("λx.x", "λx.father(x)")

    assert not item.agrees
    assert item.predicted_type is FUNCTION
    assert item.gold_type is not None
    assert CHECKER.subsumes(item.gold_type, FUNCTION) is not None


def test_unification_agrees_without_requiring_the_prediction_to_be_more_general() -> (
    None
):
    found = report(
        {ItemId("only"): (form("λx.x"), form("λx.father(x)"))},
        agreement=TypeAgreement.UNIFIES,
    )

    assert found.agreement is TypeAgreement.UNIFIES
    assert found.items[0].agrees


def test_an_ill_typed_prediction_does_not_agree_and_keeps_no_type() -> None:
    item = single("sleep(bo)", "sleep(sleep)")

    assert not item.agrees
    assert item.gold_type is TRUTH
    assert item.predicted_type is None
    assert not item.exact


def test_an_ill_typed_gold_form_agrees_with_nothing() -> None:
    item = single("sleep(sleep)", "sleep(bo)")

    assert not item.agrees
    assert item.gold_type is None
    assert item.predicted_type is TRUTH


def test_edit_distance_is_the_term_kernel_distance_over_normal_forms() -> None:
    gold = form("tall(father(bo))")
    predicted = form("(λx.tall(x))(bo)")
    kernel: TreeEditDistance = TreeEditDistance()

    found = report({ItemId("only"): (gold, predicted)})

    assert found.items[0].distance == kernel(form("tall(bo)"), gold)
    assert found.items[0].distance != kernel(predicted, gold)
    assert found.mean_edit_distance == found.items[0].distance


def test_items_with_identical_surfaces_stay_distinct_under_their_ids() -> None:
    gold = form("sleep(bo)")
    pairs = {
        ItemId("first"): (gold, form("sleep(bo)")),
        ItemId("second"): (gold, form("run(bo)")),
    }

    found = report(pairs)

    assert [item.item_id for item in found.items] == ["first", "second"]
    assert found.items[0].gold is found.items[1].gold
    assert found.items[0].exact
    assert not found.items[1].exact
    assert found.exact_match == 0.5
    assert found.type_agreement == 1.0
    assert found.mean_edit_distance == 0.5


def test_items_are_ordered_by_item_id_not_by_insertion() -> None:
    pairs = {
        ItemId("b"): (form("bo"), form("bo")),
        ItemId("a"): (form("jo"), form("jo")),
        ItemId("c"): (form("bo"), form("jo")),
    }

    found = report(pairs)

    assert [item.item_id for item in found.items] == ["a", "b", "c"]


def test_mismatched_item_ids_are_named() -> None:
    gold = {ItemId("kept"): form("bo"), ItemId("gold-only"): form("jo")}
    predicted = {ItemId("kept"): form("bo"), ItemId("predicted-only"): form("jo")}

    with pytest.raises(ValueError, match=r"missing=\('gold-only',\)") as caught:
        evaluation_report(
            gold,
            predicted,
            CHECKER,
            SIGNATURE,
            RULES,
            STORE,
            CALCULUS,
            agreement=TypeAgreement.SUBSUMES,
            normalization=Normalization(
                fuel=40,
                order="leftmost-outermost",
                eta="off",
                associative_commutative=(),
            ),
        )

    assert "extra=('predicted-only',)" in str(caught.value)


def test_an_empty_evaluation_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one gold item"):
        evaluation_report(
            {},
            {},
            CHECKER,
            SIGNATURE,
            RULES,
            STORE,
            CALCULUS,
            agreement=TypeAgreement.SUBSUMES,
            normalization=Normalization(
                fuel=40,
                order="leftmost-outermost",
                eta="off",
                associative_commutative=(),
            ),
        )

    with pytest.raises(ValueError, match="at least one item"):
        EvaluationReport(TypeAgreement.SUBSUMES, ())


def test_an_empty_item_id_is_refused() -> None:
    with pytest.raises(ValueError, match="evaluation item id"):
        report({ItemId(""): (form("bo"), form("bo"))})


def test_the_report_is_frozen_and_identity_preserving() -> None:
    found = report({ItemId("only"): (form("bo"), form("bo"))})

    with pytest.raises(AttributeError, match="immutable"):
        found.__setattr__("_items", ())

    with pytest.raises(AttributeError, match="immutable"):
        found.__delattr__("_items")

    assert copy(found) is found
    assert deepcopy(found) is found
    assert repr(found) == (
        "EvaluationReport(agreement='subsumes', items=1, exact_match=1.0, "
        "type_agreement=1.0, "
        "mean_edit_distance=0.0)"
    )


def test_the_fifty_pair_golden_table_scores() -> None:
    pairs = golden_pairs()

    found = report(pairs)

    assert len(found.items) == 50
    assert found.exact_match == pytest.approx(0.24)
    assert found.type_agreement == pytest.approx(0.8)
    assert found.mean_edit_distance == pytest.approx(1.68)


def test_every_golden_exact_match_has_zero_distance_and_agrees() -> None:
    found = report(golden_pairs())

    for item in found.items:
        if item.exact:
            assert item.distance == 0
            assert item.agrees

        if item.gold_type is None or item.predicted_type is None:
            assert not item.agrees


def test_a_form_without_a_normal_form_within_the_fuel_is_an_error() -> None:
    omega = form("(λx.x(x))(λx.x(x))")

    with pytest.raises(NonTerminatingError):
        report({ItemId("only"): (form("bo"), omega)}, fuel=8)
