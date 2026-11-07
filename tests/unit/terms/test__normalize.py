from typing import TYPE_CHECKING, Final, cast

import pytest
from hypothesis import given, settings
from tests.support.term_reference import CALCULUS, formulas
from tests.support.term_strategies import terms

from lofi.terms import (
    ETA_CONTRACTION,
    NORMALIZATION_ORDERS,
    BindingSite,
    Calculus,
    NonTerminatingError,
    Normalization,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
    beta_normal,
    beta_reducts,
    eta_contract,
    normalize,
    weak_head,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.terms import EtaPolicy, NormalizationOrder

STORE: Final = TermStore(
    formulas().with_vocabulary(
        Vocabulary("Lf", ("constant", "argument", "function", "result"))
    )
)
SIGNATURE: Final = STORE.signature
BOUND: Final = SIGNATURE.bound("Lf")
CONSTANT: Final = SIGNATURE.constant("Lf")
APPLICATION: Final = SIGNATURE.by_name(CALCULUS.app)
BINDER: Final = SIGNATURE.by_name(CALCULUS.lam)
NORMALIZATION_FUEL: Final = 100

type Expression = (
    str | int | tuple[str, Expression] | tuple[str, Expression, Expression]
)

GOLDEN_REDUCTIONS: Final[tuple[tuple[Expression, Expression], ...]] = (
    ("constant", "constant"),
    (("app", ("lam", 0), "argument"), "argument"),
    (("app", ("lam", 1), "argument"), 1),
    (
        ("app", ("lam", ("app", "function", 0)), "argument"),
        ("app", "function", "argument"),
    ),
    (
        ("app", ("lam", ("lam", ("app", 1, 0))), "function"),
        ("lam", ("app", "function", 0)),
    ),
)
"""Five reference reductions."""


def build(expression: Expression) -> Term:
    """Build one compact golden expression."""
    if isinstance(expression, int):
        return STORE.op(BOUND, payload=expression)

    if isinstance(expression, str):
        return STORE.const("Lf", expression)

    if len(expression) == 2:
        _, body = expression

        return STORE.op(BINDER, build(body))

    _, function, value = expression

    return STORE.op(APPLICATION, build(function), build(value))


@pytest.mark.parametrize(("source", "expected"), GOLDEN_REDUCTIONS)
def test_the_reference_golden_reductions_hold(
    source: Expression,
    expected: Expression,
) -> None:
    assert beta_normal(
        STORE, build(source), CALCULUS, fuel=NORMALIZATION_FUEL
    ) is build(expected)


def test_leftmost_outermost_reduction_does_not_evaluate_a_discarded_argument() -> None:
    variable = STORE.op(BOUND, payload=0)
    self_apply = STORE.op(APPLICATION, variable, variable)
    looping_abstraction = STORE.op(BINDER, self_apply)
    omega = STORE.op(APPLICATION, looping_abstraction, looping_abstraction)
    constant = build("result")
    discard = STORE.op(BINDER, constant)
    term = STORE.op(APPLICATION, discard, omega)

    assert beta_normal(STORE, term, CALCULUS, fuel=1) is constant


def test_one_step_reducts_are_root_first_and_left_to_right() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    left = STORE.op(APPLICATION, identity, build("argument"))
    right = STORE.op(APPLICATION, identity, build("result"))
    term = STORE.op(APPLICATION, left, right)

    assert beta_reducts(STORE, term, CALCULUS) == (
        STORE.op(APPLICATION, build("argument"), right),
        STORE.op(APPLICATION, left, build("result")),
    )


def test_one_step_reducts_open_and_reclose_enclosing_binders() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    term = STORE.op(BINDER, STORE.op(APPLICATION, identity, variable))
    expected = STORE.op(BINDER, variable)

    assert beta_reducts(STORE, term, CALCULUS) == (expected,)
    assert expected.is_closed


def test_nontermination_names_the_operation_and_fuel_budget() -> None:
    variable = STORE.op(BOUND, payload=0)
    body = STORE.op(APPLICATION, variable, variable)
    abstraction = STORE.op(BINDER, body)
    omega = STORE.op(APPLICATION, abstraction, abstraction)

    with pytest.raises(NonTerminatingError) as caught:
        beta_normal(STORE, omega, CALCULUS, fuel=7)

    assert (caught.value.operation, caught.value.budget) == (
        "beta normalization",
        7,
    )


def test_weak_head_nontermination_names_its_operation() -> None:
    variable = STORE.op(BOUND, payload=0)
    body = STORE.op(APPLICATION, variable, variable)
    abstraction = STORE.op(BINDER, body)
    omega = STORE.op(APPLICATION, abstraction, abstraction)

    with pytest.raises(NonTerminatingError) as caught:
        weak_head(STORE, omega, CALCULUS, fuel=5)

    assert (caught.value.operation, caught.value.budget) == (
        "weak-head beta normalization",
        5,
    )


def test_normalization_opens_and_recloses_enclosing_binders() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    term = STORE.op(BINDER, STORE.op(APPLICATION, identity, variable))
    expected = STORE.op(BINDER, variable)

    assert term.is_closed
    assert beta_normal(STORE, term, CALCULUS, fuel=NORMALIZATION_FUEL) is expected
    assert expected.is_closed


def test_weak_head_stops_at_a_lambda_and_full_normalization_continues() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    term = STORE.op(BINDER, STORE.op(APPLICATION, identity, variable))

    assert weak_head(STORE, term, CALCULUS, fuel=NORMALIZATION_FUEL) is term
    assert beta_normal(STORE, term, CALCULUS, fuel=NORMALIZATION_FUEL) is not term


def test_weak_head_does_not_reduce_an_argument_below_a_neutral_head() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    argument = STORE.op(APPLICATION, identity, build("argument"))
    term = STORE.op(APPLICATION, build("function"), argument)

    assert weak_head(STORE, term, CALCULUS, fuel=NORMALIZATION_FUEL) is term
    assert beta_normal(STORE, term, CALCULUS, fuel=NORMALIZATION_FUEL) is STORE.op(
        APPLICATION,
        build("function"),
        build("argument"),
    )


def test_the_declared_order_selects_full_or_weak_head_reduction() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    term = STORE.op(BINDER, STORE.op(APPLICATION, identity, variable))
    full = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )
    weak = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="weak-head",
        eta="off",
        associative_commutative=(),
    )

    assert normalize(STORE, term, CALCULUS, normalization=weak) is term
    assert normalize(STORE, term, CALCULUS, normalization=full) is not term


def test_the_declared_eta_policy_changes_the_normal_form() -> None:
    function = build("function")
    variable = STORE.op(BOUND, payload=0)
    expanded = STORE.op(BINDER, STORE.op(APPLICATION, function, variable))
    off = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )
    contract = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="leftmost-outermost",
        eta="contract",
        associative_commutative=(),
    )

    assert normalize(STORE, expanded, CALCULUS, normalization=off) is expanded
    assert normalize(STORE, expanded, CALCULUS, normalization=contract) is function
    assert eta_contract(STORE, expanded, CALCULUS) is function


def test_eta_contraction_preserves_an_outer_binding_without_capture() -> None:
    outer = STORE.op(BOUND, payload=1)
    inner = STORE.op(BOUND, payload=0)
    expanded = STORE.op(
        BINDER,
        STORE.op(BINDER, STORE.op(APPLICATION, outer, inner)),
    )
    expected = STORE.op(BINDER, STORE.op(BOUND, payload=0))

    assert eta_contract(STORE, expanded, CALCULUS) is expected
    assert expected.is_closed


def test_eta_keeps_an_abstraction_when_the_variable_occurs_in_the_function() -> None:
    variable = STORE.op(BOUND, payload=0)
    expanded = STORE.op(
        BINDER,
        STORE.op(APPLICATION, variable, variable),
    )

    assert eta_contract(STORE, expanded, CALCULUS) is expanded


def test_declared_associative_commutative_operators_have_one_normal_form() -> None:
    left = worked(
        "and", constant("bo"), worked("and", constant("mary"), constant("LIKE"))
    )
    right = worked(
        "and", worked("and", constant("LIKE"), constant("bo")), constant("mary")
    )
    exact = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=(),
    )
    quotient = Normalization(
        fuel=NORMALIZATION_FUEL,
        order="leftmost-outermost",
        eta="off",
        associative_commutative=("and",),
    )

    assert normalize(
        WORKED_STORE, left, WORKED_CALCULUS, normalization=exact
    ) is not normalize(WORKED_STORE, right, WORKED_CALCULUS, normalization=exact)
    assert normalize(
        WORKED_STORE, left, WORKED_CALCULUS, normalization=quotient
    ) is normalize(WORKED_STORE, right, WORKED_CALCULUS, normalization=quotient)


@pytest.mark.parametrize(
    "normalization",
    [
        lambda: Normalization(
            fuel=0,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
        lambda: Normalization(
            fuel=1,
            order=cast("NormalizationOrder", "outside-in"),
            eta="off",
            associative_commutative=(),
        ),
        lambda: Normalization(
            fuel=1,
            order="weak-head",
            eta=cast("EtaPolicy", "expand"),
            associative_commutative=(),
        ),
        lambda: Normalization(
            fuel=1,
            order="weak-head",
            eta="off",
            associative_commutative=("and", "and"),
        ),
    ],
)
def test_a_normalization_declaration_refuses_an_invalid_setting(
    normalization: Callable[[], Normalization],
) -> None:
    with pytest.raises((TypeError, ValueError)):
        normalization()


@settings(max_examples=300, deadline=None)
@given(terms(STORE, sort="Lf", max_size=7, closed=True))
def test_full_normalization_of_a_weak_head_result_agrees_on_closed_terms(
    term: Term,
) -> None:
    direct = beta_normal(STORE, term, CALCULUS, fuel=100)
    head = weak_head(STORE, term, CALCULUS, fuel=100)
    staged = beta_normal(
        STORE,
        head,
        CALCULUS,
        fuel=100,
    )

    assert staged is direct
    assert direct.is_closed
    assert head.is_closed
    assert beta_normal(STORE, direct, CALCULUS, fuel=0) is direct


def test_reduction_does_not_modify_its_input() -> None:
    variable = STORE.op(BOUND, payload=0)
    identity = STORE.op(BINDER, variable)
    argument = build("argument")
    term = STORE.op(APPLICATION, identity, argument)
    before = term.key()
    children = term.args

    assert beta_normal(STORE, term, CALCULUS, fuel=NORMALIZATION_FUEL) is argument
    assert term.key() == before
    assert term.args is children
    assert term.args[0] is identity


def test_a_calculus_with_different_operator_names_uses_the_same_normalizer() -> None:
    expression = "Expression"
    apply = Operator("apply", expression, arg_sorts=(expression, expression))
    lambda_ = Operator(
        "lambda",
        expression,
        arg_sorts=(expression,),
        binding=(BindingSite((0,), expression),),
    )
    signature = Signature(
        "alternate",
        (expression,),
        (lambda_, apply),
        vocabularies=(Vocabulary(expression, ("value",)),),
    )
    store = TermStore(signature)
    value = store.const(expression, "value")
    identity = store.op(lambda_, store.bvar(expression, 0))
    term = store.op(apply, identity, value)

    assert (
        beta_normal(
            store,
            term,
            Calculus(lam="lambda", app="apply"),
            fuel=NORMALIZATION_FUEL,
        )
        is value
    )


MEANING: Final = "Meaning"
WORKED_SIGNATURE: Final = Signature(
    "worked-example",
    (MEANING,),
    (
        Operator(
            "lambda",
            MEANING,
            arg_sorts=(MEANING,),
            binding=(BindingSite((0,), MEANING),),
        ),
        Operator(
            "exists",
            MEANING,
            arg_sorts=(MEANING,),
            binding=(BindingSite((0,), MEANING),),
        ),
        Operator("and", MEANING, arg_sorts=(MEANING, MEANING)),
        Operator("apply", MEANING, arg_sorts=(MEANING, MEANING)),
    ),
    vocabularies=(Vocabulary(MEANING, ("bo", "mary", "LIKE", "AGENT", "THEME")),),
)
WORKED_STORE: Final = TermStore(WORKED_SIGNATURE)
WORKED_CALCULUS: Final = Calculus(lam="lambda", app="apply")


def worked(name: str, *arguments: Term) -> Term:
    """Build one worked-example term."""
    return WORKED_STORE.op(WORKED_SIGNATURE.by_name(name), *arguments)


def index(value: int) -> Term:
    """Build one worked-example bound variable."""
    return WORKED_STORE.bvar(MEANING, value)


def constant(name: str) -> Term:
    """Build one worked-example constant."""
    return WORKED_STORE.const(MEANING, name)


def apply(function: Term, value: Term) -> Term:
    """Apply one worked-example function."""
    return worked("apply", function, value)


def test_three_derivations_yield_one_normalized_meaning() -> None:
    bo = constant("bo")
    mary = constant("mary")
    like = constant("LIKE")
    agent = constant("AGENT")
    theme = constant("THEME")
    event = index(0)
    subject = index(1)
    object_ = index(2)
    event_body = apply(like, event)
    agent_body = apply(apply(agent, subject), event)
    theme_body = apply(apply(theme, object_), event)
    likes = worked(
        "lambda",
        worked(
            "lambda",
            worked(
                "exists",
                worked("and", event_body, worked("and", agent_body, theme_body)),
            ),
        ),
    )
    raised_bo = worked("lambda", apply(index(0), bo))
    direct = apply(apply(likes, mary), bo)
    raised = apply(raised_bo, apply(likes, mary))
    composed = apply(
        worked("lambda", apply(raised_bo, apply(likes, index(0)))),
        mary,
    )
    expected = worked(
        "exists",
        worked(
            "and",
            apply(like, index(0)),
            worked(
                "and",
                apply(apply(agent, bo), index(0)),
                apply(apply(theme, mary), index(0)),
            ),
        ),
    )
    normalized = {
        beta_normal(
            WORKED_STORE,
            derivation,
            WORKED_CALCULUS,
            fuel=NORMALIZATION_FUEL,
        )
        for derivation in (direct, raised, composed)
    }

    assert normalized == {expected}
    assert next(iter(normalized)) is expected


def test_a_church_abstraction_reduces_its_body_and_drops_its_annotation() -> None:
    types, forms = "Ty", "Tm"
    signature = Signature(
        "church",
        (types, forms),
        (
            Operator(
                "lam",
                forms,
                arg_sorts=(types, forms),
                binding=(BindingSite((1,), forms, 0),),
            ),
            Operator("app", forms, arg_sorts=(forms, forms)),
        ),
        vocabularies=(Vocabulary(types, ("e",)), Vocabulary(forms, ("jo", "f"))),
        classifiers={forms: types},
    )
    store = TermStore(signature)
    app = signature.by_name("app")
    entity = store.const(types, "e")
    jo, f = store.const(forms, "jo"), store.const(forms, "f")
    identity = store.op(signature.by_name("lam"), entity, store.bvar(forms, 0))
    nested = store.op(
        signature.by_name("lam"),
        entity,
        store.op(app, store.op(app, identity, store.bvar(forms, 0)), f),
    )

    assert (
        beta_normal(
            store,
            store.op(app, identity, jo),
            CALCULUS,
            fuel=NORMALIZATION_FUEL,
        )
        is jo
    )
    assert beta_normal(store, nested, CALCULUS, fuel=NORMALIZATION_FUEL) is store.op(
        signature.by_name("lam"), entity, store.op(app, store.bvar(forms, 0), f)
    )


def test_the_declared_names_select_the_traversal_and_the_eta_flag() -> None:
    lazy = Normalization(
        fuel=4, order="weak-head", eta="contract", associative_commutative=()
    )
    eager = Normalization(
        fuel=4, order="leftmost-outermost", eta="off", associative_commutative=()
    )

    assert lazy.contracts_eta
    assert not eager.contracts_eta
    assert set(NORMALIZATION_ORDERS) == {"leftmost-outermost", "weak-head"}
    assert ETA_CONTRACTION == {"off": False, "contract": True}
