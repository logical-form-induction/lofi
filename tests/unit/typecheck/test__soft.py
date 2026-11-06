from math import log
from types import MappingProxyType
from typing import Final, Literal

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lofi.terms import (
    MVar,
    Node,
    Operator,
    PVar,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)
from lofi.typecheck import (
    OPERATOR_FIELD,
    PAYLOAD_FIELD,
    WHOLE_TERM,
    SoftTypingAddress,
    SoftTypingDistributions,
    SoftTypingInstance,
    TypingRule,
    compile_soft_typing,
    soft_typing_reference,
)

SIGNATURE: Final = Signature(
    "soft-typing",
    ("Ty",),
    (
        Operator("e", "Ty"),
        Operator("t", "Ty"),
        Operator("u", "Ty"),
        Operator("v", "Ty"),
        Operator("->", "Ty", arg_sorts=("Ty", "Ty")),
    ),
    vocabularies=(Vocabulary("Ty", ("zero", "one")),),
)
STORE: Final = TermStore(SIGNATURE)
E: Final = STORE.op(SIGNATURE.by_name("e"))
T: Final = STORE.op(SIGNATURE.by_name("t"))
U: Final = STORE.op(SIGNATURE.by_name("u"))
V: Final = STORE.op(SIGNATURE.by_name("v"))
ARROW: Final = SIGNATURE.by_name("->")
LABEL: Final = SIGNATURE.constant("Ty")
A: Final = MVar("A")
B: Final = MVar("B")


def arrow(left: Term, right: Term) -> Term:
    """Build a function type.

    Parameters
    ----------
    left : Term
        Domain type.
    right : Term
        Codomain type.
    """
    return STORE.op(ARROW, left, right)


def point(term: Term) -> MappingProxyType[Term, float]:
    """Build a point-mass distribution.

    Parameters
    ----------
    term : Term
        Type receiving all mass.
    """
    return MappingProxyType({term: 1.0})


def binary(
    first: Term,
    second: Term,
    probability: float,
) -> MappingProxyType[Term, float]:
    """Build a binary distribution.

    Parameters
    ----------
    first : Term
        First support value.
    second : Term
        Second support value.
    probability : float
        Mass assigned to the first value.
    """
    return MappingProxyType({first: probability, second: 1.0 - probability})


def lam_distributions(
    domain: MappingProxyType[Term, float],
    body: MappingProxyType[Term, float],
) -> SoftTypingDistributions:
    """Build marginals for a lambda typing-rule application.

    Parameters
    ----------
    domain : MappingProxyType[Term, float]
        Domain marginal at both occurrences of ``A``.
    body : MappingProxyType[Term, float]
        Codomain marginal at both occurrences of ``B``.
    """
    return SoftTypingDistributions(
        binds=({(): domain},),
        premises=({(): body},),
        result={
            (): point(arrow(E, T)),
            (0,): domain,
            (1,): body,
        },
    )


def instance(
    op: str,
    distributions: SoftTypingDistributions,
    *,
    constituent_marginal: float = 1.0,
    split_probability: float = 1.0,
) -> SoftTypingInstance:
    """Build one reference instance.

    Parameters
    ----------
    op : str
        Typing-rule operator.
    distributions : SoftTypingDistributions
        Node marginals to score.
    constituent_marginal : float
        Posterior mass on the constituent.
    split_probability : float
        Conditional mass on the split.
    """
    return SoftTypingInstance(
        (0, 2),
        1,
        op,
        distributions,
        constituent_marginal,
        split_probability,
    )


def test_the_compiled_plan_resolves_enforcement_and_direction() -> None:
    rule = TypingRule(
        "selected",
        (A, None),
        (A, A),
        A,
        enforced=(1,),
        direction="both",
    )

    plan = compile_soft_typing(rule)

    assert plan.binding_count == 2
    assert plan.premise_count == 2
    assert plan.required_bindings == ()
    assert plan.targets == ()
    assert len(plan.comparisons) == 2
    assert plan.comparisons[0].source.address == SoftTypingAddress("premise", 1, ())
    assert plan.comparisons[0].target.address == SoftTypingAddress("result", 0, ())
    assert plan.comparisons[1].source == plan.comparisons[0].target
    assert plan.comparisons[1].target == plan.comparisons[0].source


def test_the_compiled_node_test_distinguishes_literal_and_variable_fields() -> None:
    rule = TypingRule(
        "payload",
        (),
        (),
        Node("#const[Ty]", payload=PVar("index")),
    )

    plan = compile_soft_typing(rule)

    assert len(plan.targets) == 1
    assert plan.targets[0].node.operator == "#const[Ty]"
    assert plan.targets[0].node.operator_variable is None
    assert plan.targets[0].node.payload is None
    assert plan.targets[0].node.payload_variable == "index"


def test_the_compiled_plan_requires_active_binding_marginals() -> None:
    rule = TypingRule("binder", (A,), (A,), A)
    plan = compile_soft_typing(rule)

    assert plan.required_bindings == (0,)

    with pytest.raises(ValueError, match="no binding distributions"):
        soft_typing_reference(
            {rule.op: rule},
            (
                instance(
                    rule.op,
                    SoftTypingDistributions(
                        (None,),
                        ({(): point(E)},),
                        {(): point(E)},
                    ),
                ),
            ),
        )


@pytest.mark.parametrize(
    ("distributions", "message"),
    [
        (SoftTypingDistributions((), ({(): point(E)},), {(): point(E)}), "bindings"),
        (SoftTypingDistributions((None,), (), {(): point(E)}), "premises"),
    ],
)
def test_distribution_tuple_arity_matches_the_compiled_plan(
    distributions: SoftTypingDistributions,
    message: str,
) -> None:
    rule = TypingRule("identity", (None,), (A,), A)

    with pytest.raises(ValueError, match=message):
        soft_typing_reference(
            {rule.op: rule},
            (instance(rule.op, distributions),),
        )


def test_lambda_masks_its_metavariable_children() -> None:
    rule = TypingRule("lam", (A,), (B,), Node("->", (A, B)))
    distributions = lam_distributions(point(E), point(T))
    distributions = distributions._replace(
        result={
            (): MappingProxyType({arrow(E, T): 0.75, E: 0.25}),
            (0,): point(E),
            (1,): point(T),
        }
    )

    report = soft_typing_reference({"lam": rule}, (instance("lam", distributions),))

    assert report.total == pytest.approx(-log(0.75))


def test_each_declared_rule_gets_one_term() -> None:
    lam = TypingRule("lam", (A,), (B,), Node("->", (A, B)))
    constant = TypingRule("constant", (), (), Node("t"))

    report = soft_typing_reference(
        {"lam": lam, "constant": constant},
        (instance("lam", lam_distributions(point(E), point(T))),),
    )

    assert [term.op for term in report.terms] == ["lam", "constant"]
    assert [term.instance_count for term in report.terms] == [1, 0]


def test_the_sentence_loss_uses_rule_marginals_not_a_flat_sum() -> None:
    rule = TypingRule("constant", (), (), Node("t"))
    half = MappingProxyType({T: 0.5, E: 0.5})
    distributions = SoftTypingDistributions((), (), {(): half})
    instances = (
        instance(
            "constant",
            distributions,
            constituent_marginal=0.8,
            split_probability=0.25,
        ),
        instance(
            "constant",
            distributions,
            constituent_marginal=0.2,
            split_probability=0.5,
        ),
    )

    report = soft_typing_reference({"constant": rule}, instances)

    assert report.terms[0].unweighted == pytest.approx(2.0 * -log(0.5))
    assert report.total == pytest.approx((0.8 * 0.25 + 0.2 * 0.5) * -log(0.5))


def test_zero_rule_marginal_masks_an_impossible_local_target() -> None:
    rule = TypingRule("constant", (), (), Node("t"))
    distributions = SoftTypingDistributions((), (), {(): point(E)})

    report = soft_typing_reference(
        {"constant": rule},
        (
            instance(
                "constant",
                distributions,
                constituent_marginal=0.0,
            ),
        ),
    )

    assert report.terms[0].unweighted == float("inf")
    assert report.total == 0.0


@settings(max_examples=120)
@given(
    first=st.floats(min_value=0.01, max_value=0.99),
    second=st.floats(min_value=0.01, max_value=0.99),
)
def test_repeated_operator_variables_preserve_equality(
    first: float,
    second: float,
) -> None:
    rule = TypingRule(
        "same-head",
        (),
        (),
        Node("->", (Node(PVar("head")), Node(PVar("head")))),
        direction="up",
    )
    root = arrow(E, E)
    distributions = SoftTypingDistributions(
        (),
        (),
        {
            (): point(root),
            (0,): binary(E, T, first),
            (1,): binary(E, T, second),
        },
    )
    expected = -(first * log(second) + (1.0 - first) * log(1.0 - second))

    report = soft_typing_reference(
        {"same-head": rule},
        (instance("same-head", distributions),),
    )

    assert report.total == pytest.approx(expected)


def test_repeated_payload_variables_preserve_equality() -> None:
    zero = STORE.op(LABEL, payload=0)
    one = STORE.op(LABEL, payload=1)
    rule = TypingRule(
        "same-payload",
        (),
        (),
        Node(
            "->",
            (
                Node("#const[Ty]", payload=PVar("index")),
                Node("#const[Ty]", payload=PVar("index")),
            ),
        ),
    )
    distributions = SoftTypingDistributions(
        (),
        (),
        {
            (): point(arrow(zero, one)),
            (0,): point(zero),
            (1,): point(one),
        },
    )

    report = soft_typing_reference(
        {"same-payload": rule},
        (instance("same-payload", distributions),),
    )

    assert report.total == float("inf")


def test_a_payload_variable_rejects_payloadless_mass() -> None:
    rule = TypingRule("payload", (), (), Node(PVar("op"), payload=PVar("value")))
    distributions = SoftTypingDistributions((), (), {(): point(E)})

    report = soft_typing_reference(
        {"payload": rule},
        (instance("payload", distributions),),
    )

    assert report.total == float("inf")


@settings(max_examples=120)
@given(
    domain_probability=st.floats(min_value=0.01, max_value=0.99),
    body_probability=st.floats(min_value=0.01, max_value=0.99),
    variable=st.sampled_from(("A", "B")),
)
def test_substitution_at_metavariable_positions_preserves_the_loss(
    domain_probability: float,
    body_probability: float,
    variable: Literal["A", "B"],
) -> None:
    rule = TypingRule("lam", (A,), (B,), Node("->", (A, B)))
    domain = binary(E, T, domain_probability)
    body = binary(E, T, body_probability)
    replacement_domain = binary(U, V, domain_probability) if variable == "A" else domain
    replacement_body = binary(U, V, body_probability) if variable == "B" else body
    original = lam_distributions(domain, body)
    substituted = lam_distributions(replacement_domain, replacement_body)

    before = soft_typing_reference({"lam": rule}, (instance("lam", original),))
    after = soft_typing_reference({"lam": rule}, (instance("lam", substituted),))

    assert after.total == pytest.approx(before.total)


@settings(max_examples=120)
@given(
    first=st.floats(min_value=0.01, max_value=0.99),
    second=st.floats(min_value=0.01, max_value=0.99),
    direction=st.sampled_from(("up", "down", "both")),
)
def test_direction_selects_the_source_and_receiver(
    first: float,
    second: float,
    direction: Literal["up", "down", "both"],
) -> None:
    rule = TypingRule("identity", (None,), (A,), A, direction=direction)
    distributions = SoftTypingDistributions(
        (None,),
        ({(): binary(E, T, first)},),
        {(): binary(E, T, second)},
    )
    forward = -(first * log(second) + (1.0 - first) * log(1.0 - second))
    backward = -(second * log(first) + (1.0 - second) * log(1.0 - first))
    expected = {
        "up": forward,
        "down": backward,
        "both": forward + backward,
    }[direction]

    report = soft_typing_reference(
        {"identity": rule},
        (instance("identity", distributions),),
    )

    assert report.total == pytest.approx(expected)


@settings(max_examples=120)
@given(
    first=st.floats(min_value=0.01, max_value=0.99),
    second=st.floats(min_value=0.01, max_value=0.99),
    enforced=st.integers(min_value=0, max_value=1),
)
def test_only_enforced_premises_contribute(
    first: float,
    second: float,
    enforced: int,
) -> None:
    rule = TypingRule(
        "choose",
        (None, None),
        (Node("e"), Node("e")),
        Node("t"),
        enforced=(enforced,),
    )
    distributions = SoftTypingDistributions(
        (None, None),
        (
            {(): binary(E, T, first)},
            {(): binary(E, T, second)},
        ),
        {(): point(T)},
    )
    selected = first if enforced == 0 else second

    report = soft_typing_reference(
        {"choose": rule},
        (instance("choose", distributions),),
    )

    assert report.total == pytest.approx(-log(selected))


def test_a_required_node_marginal_must_be_present() -> None:
    rule = TypingRule("constant", (), (), Node("t"))
    distributions = SoftTypingDistributions((), (), {})

    with pytest.raises(ValueError, match="supplies no marginal"):
        soft_typing_reference(
            {"constant": rule},
            (instance("constant", distributions),),
        )


def test_each_projected_field_reads_its_part_of_a_matched_node() -> None:
    labeled = STORE.op(LABEL, payload=1)

    assert WHOLE_TERM.value(E) is E
    assert OPERATOR_FIELD.value(arrow(E, T)) == "->"
    assert PAYLOAD_FIELD.value(labeled) == 1
    assert (WHOLE_TERM.name, OPERATOR_FIELD.name, PAYLOAD_FIELD.name) == (
        "term",
        "operator",
        "payload",
    )

    with pytest.raises(ValueError, match="has no payload"):
        PAYLOAD_FIELD.value(E)
