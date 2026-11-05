import math
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, override

import didactic.api as dx
import pytest
import torch
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.objective import ObjectiveLawSuite

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._typing import (
    SoftTypingObjective,
    TensorSoftTypingDistributions,
    TensorSoftTypingInstance,
)
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
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
    SoftTypingDistributions,
    SoftTypingInstance,
    TypingRule,
    Unifier,
    soft_typing_reference,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from lofi.nn.loss._ports import LossTerm
    from lofi.terms import OpName


SIGNATURE: Final = Signature(
    "tensor-soft-typing",
    ("Ty",),
    (
        Operator("e", "Ty"),
        Operator("t", "Ty"),
        Operator("u", "Ty"),
        Operator("->", "Ty", arg_sorts=("Ty", "Ty")),
    ),
    vocabularies=(Vocabulary("Ty", ("zero", "one")),),
    open_sorts=("Ty",),
)
STORE: Final = TermStore(SIGNATURE)
E: Final = STORE.op(SIGNATURE.by_name("e"))
T: Final = STORE.op(SIGNATURE.by_name("t"))
U: Final = STORE.op(SIGNATURE.by_name("u"))
ARROW: Final = SIGNATURE.by_name("->")
LABEL: Final = SIGNATURE.constant("Ty")
A: Final = MVar("A")
B: Final = MVar("B")
SPANS: Final = SpanIndex(2)
CONSTANT: Final = TypingRule("constant", (), (), Node("t"))
LAMBDA: Final = TypingRule("lambda", (A,), (B,), Node("->", (A, B)))
UNIFIER: Final = Unifier(STORE)


class Batch(NamedTuple):
    typing_instances: tuple[TensorSoftTypingInstance, ...]


def scalar(value: float) -> Tensor:
    """Build a differentiable float64 scalar probability.

    Parameters
    ----------
    value : float
        Initial probability.
    """
    return torch.tensor(value, dtype=torch.float64, requires_grad=True)


def gradient(value: Tensor) -> Tensor:
    """Read a gradient required by a test.

    Parameters
    ----------
    value : Tensor
        Differentiable scalar whose gradient was computed.

    Returns
    -------
    Tensor
        Computed gradient.

    Raises
    ------
    AssertionError
        If no gradient was computed.
    """
    found = value.grad

    if found is None:
        msg = "the tensor has no gradient"
        raise AssertionError(msg)

    return found


def arrow(left: Term, right: Term) -> Term:
    """Build a function type.

    Parameters
    ----------
    left : Term
        Domain.
    right : Term
        Codomain.
    """
    return STORE.op(ARROW, left, right)


def probabilities(
    entries: Sequence[tuple[Term, float]],
) -> MappingProxyType[Term, Tensor]:
    """Build a tensor-valued categorical marginal.

    Parameters
    ----------
    entries : Sequence[tuple[Term, float]]
        Terms and their probabilities.
    """
    return MappingProxyType({term: scalar(value) for term, value in entries})


def constant_instance(
    probability: float = 0.75,
    *,
    op: OpName = "constant",
    constituent: float = 0.8,
    split: float = 0.5,
) -> TensorSoftTypingInstance:
    """Build one nullary typing-rule application.

    Parameters
    ----------
    probability : float
        Marginal probability of the declared result.
    op : OpName
        Rule operator.
    constituent : float
        Constituent marginal.
    split : float
        Conditional split probability.
    """
    return TensorSoftTypingInstance(
        (0, 2),
        1,
        op,
        TensorSoftTypingDistributions(
            (),
            (),
            {(): probabilities(((T, probability), (E, 1.0 - probability)))},
        ),
        scalar(constituent),
        scalar(split),
    )


def lambda_instance(
    principal: Term,
    erased: Term,
    *,
    root_probability: float = 1.0,
    binding_probability: float = 1.0,
    result_probability: float = 1.0,
) -> TensorSoftTypingInstance:
    """Build one lambda-rule application with coherence types.

    Parameters
    ----------
    principal : Term
        Principal type whose metavariables may bind.
    erased : Term
        Fixed category erasure.
    root_probability : float
        Probability assigned to the erased root.
    binding_probability : float
        Probability assigned to the ground binding type.
    result_probability : float
        Probability assigned to that type at the matching result node.
    """
    right = erased.args[1]
    root = probabilities(((erased, root_probability), (E, 1.0 - root_probability)))
    binding = probabilities(((E, binding_probability), (T, 1.0 - binding_probability)))
    result_left = probabilities(
        ((E, result_probability), (T, 1.0 - result_probability))
    )
    result = {
        (): root,
        (0,): result_left,
        (1,): probabilities(((right, 1.0),)),
    }

    for position, child in enumerate(right.args):
        result[(1, position)] = probabilities(((child, 1.0),))

    return TensorSoftTypingInstance(
        (0, 2),
        1,
        LAMBDA.op,
        TensorSoftTypingDistributions(
            ({(): binding},),
            ({(): probabilities(((right, 1.0),))},),
            result,
        ),
        scalar(1.0),
        scalar(1.0),
        principal,
        erased,
    )


def cells(items: int = 1) -> Encoded:
    """Build differentiable float64 cell encodings with item, span, and width axes.

    Parameters
    ----------
    items : int
        Number of batch items.
    """
    return Encoded(
        torch.ones(
            items,
            SPANS.num_spans,
            1,
            dtype=torch.float64,
            requires_grad=True,
        ),
        ("item", "span", "width"),
    )


def bundle(
    probability: float = 0.75,
    *,
    instances: tuple[TensorSoftTypingInstance, ...] | None = None,
) -> ObjectiveOutputs[Batch]:
    """Build an objective bundle over tensor soft-typing instances.

    Parameters
    ----------
    probability : float
        Probability assigned to the constant rule's result.
    instances : tuple[TensorSoftTypingInstance, ...] | None
        Explicit instances, or ``None`` to build one constant instance.
    """
    held = (constant_instance(probability),) if instances is None else instances
    anchor = cells(max(1, len(held)))
    scores = LogDistribution(
        anchor.tensor.new_zeros((anchor.tensor.shape[0], SPANS.num_spans)),
        SPAN_AXES,
    )

    return ObjectiveOutputs(
        spans=SPANS,
        chart=CellChart(scores),
        batch=Batch(held),
        cells=anchor,
    )


def objective() -> SoftTypingObjective:
    """Build the constant-rule objective."""
    return SoftTypingObjective(CONSTANT)


def instance_tensors(
    outputs: ObjectiveOutputs[Batch],
) -> tuple[Tensor, ...]:
    """Return tensors the constant-rule loss depends on.

    Parameters
    ----------
    outputs : ObjectiveOutputs[Batch]
        Bundle to inspect.
    """
    matched = tuple(
        instance
        for instance in outputs.batch.typing_instances
        if instance.op == CONSTANT.op
    )

    if not matched:
        held = outputs.cells

        assert held is not None

        return (held.tensor,)

    tensors = []

    for instance in matched:
        tensors.extend(
            (
                instance.distributions.result[()][T],
                instance.constituent_marginal,
                instance.split_probability,
            )
        )

    return tuple(tensors)


def by_hand(outputs: ObjectiveOutputs[Batch]) -> float:
    """Return the constant-rule sum computed directly.

    Parameters
    ----------
    outputs : ObjectiveOutputs[Batch]
        Bundle to score.
    """
    losses = []

    for instance in outputs.batch.typing_instances:
        if instance.op != CONSTANT.op:
            continue

        probability = float(instance.distributions.result[()][T].detach())
        weight = float(
            (instance.constituent_marginal * instance.split_probability).detach()
        )
        losses.append(weight * -math.log(probability))

    return sum(losses)


def permuted(outputs: ObjectiveOutputs[Batch]) -> ObjectiveOutputs[Batch]:
    """Return the bundle with its instances reversed.

    Parameters
    ----------
    outputs : ObjectiveOutputs[Batch]
        Bundle to reorder.
    """
    return bundle(instances=tuple(reversed(outputs.batch.typing_instances)))


def refilled(outputs: ObjectiveOutputs[Batch]) -> ObjectiveOutputs[Batch]:
    """Return the unchanged bundle, which has no masked probability positions.

    Parameters
    ----------
    outputs : ObjectiveOutputs[Batch]
        Bundle to retain.
    """
    return outputs


def inapplicable() -> ObjectiveOutputs[Batch]:
    """Build a bundle containing no constant-rule application."""
    return bundle(instances=(constant_instance(op="modal"),))


def models_built(call: Callable[[], LossTerm]) -> int:
    """Return how many didactic models one evaluation constructs.

    Parameters
    ----------
    call : Callable[[], LossTerm]
        Evaluation to count around.
    """
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


def fixture() -> ObjectiveFixture[Batch]:
    """Build the constant-rule conformance fixture."""
    return ObjectiveFixture(
        objective=objective(),
        outputs=lambda: bundle(
            instances=(constant_instance(0.75), constant_instance(0.6))
        ),
        inputs=instance_tensors,
        unweighted=by_hand,
        monotone=lambda: (bundle(0.25), bundle(0.9)),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=inapplicable,
        tolerance=1e-9,
    )


class TestSoftTypingObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Batch]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that reducing accepted result mass raises the loss."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def as_reference(
    instance: TensorSoftTypingInstance,
) -> SoftTypingInstance:
    """Detach a tensor instance for the symbolic reference.

    Parameters
    ----------
    instance : TensorSoftTypingInstance
        Tensor-valued rule application.
    """

    def marginal(
        nodes: Mapping[tuple[int, ...], Mapping[Term, Tensor]],
    ) -> dict[tuple[int, ...], dict[Term, float]]:
        return {
            path: {term: float(value.detach()) for term, value in values.items()}
            for path, values in nodes.items()
        }

    distributions = instance.distributions

    return SoftTypingInstance(
        instance.span,
        instance.split,
        instance.op,
        SoftTypingDistributions(
            tuple(
                None if held is None else marginal(held) for held in distributions.binds
            ),
            tuple(marginal(held) for held in distributions.premises),
            marginal(distributions.result),
        ),
        float(instance.constituent_marginal.detach()),
        float(instance.split_probability.detach()),
    )


def evaluated(
    rule: TypingRule,
    instance: TensorSoftTypingInstance,
) -> float:
    """Return the tensor objective's value for one rule application.

    Parameters
    ----------
    rule : TypingRule
        Rule to score.
    instance : TensorSoftTypingInstance
        Application to score.
    """
    return float(
        SoftTypingObjective(rule).evaluate(bundle(instances=(instance,))).value.detach()
    )


def test_the_tensor_executor_agrees_with_the_symbolic_reference() -> None:
    instance = constant_instance()

    actual = evaluated(CONSTANT, instance)
    expected = soft_typing_reference(
        {CONSTANT.op: CONSTANT},
        (as_reference(instance),),
    )

    assert actual == pytest.approx(expected.total)


@pytest.mark.parametrize("direction", ["up", "down", "both"])
def test_the_compiled_direction_selects_source_and_target(
    direction: Literal["up", "down", "both"],
) -> None:
    rule = TypingRule("identity", (None,), (A,), A, direction=direction)
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (None,),
            ({(): probabilities(((E, 0.8), (T, 0.2)))},),
            {(): probabilities(((E, 0.3), (T, 0.7)))},
        ),
        scalar(1.0),
        scalar(1.0),
    )

    actual = evaluated(rule, instance)
    expected = soft_typing_reference(
        {rule.op: rule},
        (as_reference(instance),),
    )

    assert actual == pytest.approx(expected.total)


def test_the_compiled_enforcement_omits_an_unenforced_premise() -> None:
    rule = TypingRule(
        "quantifier",
        (None, None),
        (Node("e"), Node("e")),
        Node("t"),
        enforced=(1,),
    )
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (None, None),
            (
                {(): probabilities(((E, 0.1), (T, 0.9)))},
                {(): probabilities(((E, 0.7), (T, 0.3)))},
            ),
            {(): probabilities(((T, 1.0),))},
        ),
        scalar(1.0),
        scalar(1.0),
    )

    actual = evaluated(rule, instance)
    expected = soft_typing_reference(
        {rule.op: rule},
        (as_reference(instance),),
    )

    assert actual == pytest.approx(expected.total)
    assert actual == pytest.approx(-math.log(0.7))


@settings(max_examples=120)
@given(
    first=st.floats(min_value=0.01, max_value=0.99),
    second=st.floats(min_value=0.01, max_value=0.99),
    result=st.floats(min_value=0.01, max_value=0.99),
    enforced=st.integers(min_value=0, max_value=1),
    direction=st.sampled_from(("up", "down", "both")),
)
def test_tensor_and_symbolic_backends_agree_across_compiled_plans(
    first: float,
    second: float,
    result: float,
    enforced: int,
    direction: Literal["up", "down", "both"],
) -> None:
    rule = TypingRule(
        "choose",
        (None, None),
        (A, A),
        A,
        enforced=(enforced,),
        direction=direction,
    )
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (None, None),
            (
                {(): probabilities(((E, first), (T, 1.0 - first)))},
                {(): probabilities(((E, second), (T, 1.0 - second)))},
            ),
            {(): probabilities(((E, result), (T, 1.0 - result)))},
        ),
        scalar(0.8),
        scalar(0.5),
    )

    actual = evaluated(rule, instance)
    expected = soft_typing_reference(
        {rule.op: rule},
        (as_reference(instance),),
    )

    assert actual == pytest.approx(expected.total, rel=1e-10, abs=1e-10)


def test_operator_projections_agree_with_the_reference_and_take_gradients() -> None:
    rule = TypingRule(
        "same-operator",
        (),
        (),
        Node("->", (Node(PVar("head")), Node(PVar("head")))),
    )
    first = probabilities(((E, 0.8), (T, 0.2)))
    second = probabilities(((E, 0.3), (T, 0.7)))
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (),
            (),
            {
                (): probabilities(((arrow(E, T), 1.0),)),
                (0,): first,
                (1,): second,
            },
        ),
        scalar(1.0),
        scalar(1.0),
    )
    term = SoftTypingObjective(rule).evaluate(bundle(instances=(instance,)))
    expected = soft_typing_reference(
        {rule.op: rule},
        (as_reference(instance),),
    )

    term.value.backward()
    first_gradient = first[E].grad
    second_gradient = second[E].grad

    assert term.value.item() == pytest.approx(expected.total)
    assert first_gradient is not None
    assert second_gradient is not None
    assert float(first_gradient.abs()) > 0.0
    assert float(second_gradient.abs()) > 0.0


def test_payload_projections_agree_with_the_reference_and_take_gradients() -> None:
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
    first = probabilities(((zero, 0.6), (one, 0.4)))
    second = probabilities(((zero, 0.2), (one, 0.8)))
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (),
            (),
            {
                (): probabilities(((arrow(zero, one), 1.0),)),
                (0,): first,
                (1,): second,
            },
        ),
        scalar(1.0),
        scalar(1.0),
    )
    term = SoftTypingObjective(rule).evaluate(bundle(instances=(instance,)))
    expected = soft_typing_reference(
        {rule.op: rule},
        (as_reference(instance),),
    )

    term.value.backward()
    first_gradient = first[zero].grad
    second_gradient = second[zero].grad

    assert term.value.item() == pytest.approx(expected.total)
    assert first_gradient is not None
    assert second_gradient is not None
    assert float(first_gradient.abs()) > 0.0
    assert float(second_gradient.abs()) > 0.0


def test_a_partial_subsumption_mask_scores_only_ground_result_nodes() -> None:
    variable = STORE.fvar("Ty", "answer")
    erased = arrow(E, arrow(E, T))
    instance = lambda_instance(
        arrow(E, variable),
        erased,
        root_probability=0.8,
    )
    objective = SoftTypingObjective(LAMBDA, subsumes=UNIFIER.subsumes)
    term = objective.evaluate(bundle(instances=(instance,)))
    root = instance.distributions.result[()][erased]
    masked = tuple(
        probability
        for path, marginal in instance.distributions.result.items()
        if path[:1] == (1,)
        for probability in marginal.values()
    )
    premise = tuple(instance.distributions.premises[0][()].values())

    term.value.backward()

    assert term.value.item() == pytest.approx(-math.log(0.8))
    assert term.diagnostics["mask_density"].item() == pytest.approx(2 / 5)
    assert 0.0 < term.diagnostics["mask_density"].item() < 1.0
    assert term.diagnostics["mask_density"].grad_fn is None
    assert root.grad is not None
    assert float(root.grad.abs()) > 0.0
    assert all(
        float(gradient(probability)) == 0.0 for probability in (*masked, *premise)
    )


def test_a_fully_bound_principal_variable_masks_value_and_gradient() -> None:
    variable = STORE.fvar("Ty", "answer")
    rule = TypingRule("identity", (None,), (A,), A)
    premise = probabilities(((E, 0.8), (T, 0.2)))
    result = probabilities(((E, 0.2), (T, 0.8)))
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions((None,), ({(): premise},), {(): result}),
        scalar(1.0),
        scalar(1.0),
        variable,
        E,
    )
    term = SoftTypingObjective(rule, subsumes=UNIFIER.subsumes).evaluate(
        bundle(instances=(instance,))
    )

    term.value.backward()

    assert term.value.item() == 0.0
    assert term.diagnostics["mask_density"].item() == 0.0
    assert all(float(gradient(probability)) == 0.0 for probability in premise.values())
    assert all(float(gradient(probability)) == 0.0 for probability in result.values())


def test_twelve_coherent_items_score_exactly_zero_under_subsumption() -> None:
    variable = STORE.fvar("Ty", "answer")
    erased = arrow(E, T)
    principals = (arrow(E, variable),) * 6 + (erased,) * 6
    instances = tuple(lambda_instance(principal, erased) for principal in principals)
    term = SoftTypingObjective(LAMBDA, subsumes=UNIFIER.subsumes).evaluate(
        bundle(instances=instances)
    )

    assert term.count == 12
    assert term.value.item() == 0.0
    assert sum(principal is not erased for principal in principals) == 6


def test_a_mistyped_ground_node_has_positive_loss() -> None:
    erased = arrow(E, T)
    instance = lambda_instance(
        erased,
        erased,
        binding_probability=0.8,
        result_probability=0.2,
    )
    term = SoftTypingObjective(LAMBDA, subsumes=UNIFIER.subsumes).evaluate(
        bundle(instances=(instance,))
    )

    assert term.value.item() > 0.0
    assert term.diagnostics["mask_density"].item() == 1.0


def test_288_acceptance_items_match_once_with_strict_instances() -> None:
    variable = STORE.fvar("Ty", "answer")
    erased = arrow(E, T)
    principals = tuple(
        arrow(E, variable) if position % 3 == 0 else erased for position in range(288)
    )
    instances = tuple(lambda_instance(principal, erased) for principal in principals)
    calls = 0
    strict = 0

    def counted(principal: Term, specific: Term) -> Mapping[int, Term] | None:
        """Count one-way matches and strict substitutions."""
        nonlocal calls, strict
        calls += 1
        substitution = UNIFIER.subsumes(principal, specific)

        if substitution:
            strict += 1

        return substitution

    term = SoftTypingObjective(LAMBDA, subsumes=counted).evaluate(
        bundle(instances=instances)
    )

    assert term.count == 288
    assert term.value.item() == 0.0
    assert calls == 288
    assert 0 < strict < 288
    assert term.diagnostics["mask_density"].item() == pytest.approx(8 / 9)


def test_variables_in_the_erased_type_cannot_bind() -> None:
    erased_variable = STORE.fvar("Ty", "erased")
    instance = constant_instance()._replace(principal=E, erased=erased_variable)
    objective = SoftTypingObjective(CONSTANT, subsumes=UNIFIER.subsumes)

    assert UNIFIER.subsumes(E, erased_variable) is None
    assert UNIFIER.subsumes(erased_variable, E) is not None

    with pytest.raises(ValueError, match="does not subsume"):
        objective.evaluate(bundle(instances=(instance,)))


def test_coherence_types_require_a_configured_matcher() -> None:
    instance = constant_instance()._replace(principal=T, erased=T)

    with pytest.raises(ValueError, match="no matcher"):
        objective().evaluate(bundle(instances=(instance,)))


def test_coherence_types_must_be_supplied_as_a_pair() -> None:
    instance = constant_instance()._replace(principal=T)

    with pytest.raises(ValueError, match="both principal and erased"):
        SoftTypingObjective(CONSTANT, subsumes=UNIFIER.subsumes).evaluate(
            bundle(instances=(instance,))
        )


def test_a_configured_matcher_requires_types_on_every_instance() -> None:
    calls = 0

    def counted(principal: Term, specific: Term) -> Mapping[int, Term] | None:
        nonlocal calls
        calls += 1

        return UNIFIER.subsumes(principal, specific)

    with pytest.raises(ValueError, match="requires coherence types"):
        SoftTypingObjective(CONSTANT, subsumes=counted).evaluate(
            bundle(instances=(constant_instance(),))
        )

    assert calls == 0


def test_adding_a_modal_rule_adds_exactly_one_named_term() -> None:
    base = {
        "constant": CONSTANT,
        "identity": TypingRule("identity", (None,), (A,), A),
        "lambda": TypingRule("lambda", (A,), (B,), Node("->", (A, B))),
        "quantifier": TypingRule(
            "quantifier",
            (A,),
            (Node("t"),),
            Node("t"),
            enforced=(0,),
        ),
    }
    modal = TypingRule("necessarily", (None,), (Node("t"),), Node("t"))
    extended = {**base, modal.op: modal}
    original = SoftTypingObjective.from_rules(base)
    added = SoftTypingObjective.from_rules(extended)

    assert tuple(term.name for term in original) == tuple(base)
    assert tuple(term.name for term in added) == tuple(extended)
    assert len(added) == len(original) + 1
    assert added[-1].rule is modal


def test_the_suite_reports_one_term_per_rule_declaration() -> None:
    rules = {
        "constant": CONSTANT,
        "modal": TypingRule("modal", (None,), (Node("t"),), Node("t")),
    }
    suite = ObjectiveSuite(
        [(1.0, objective) for objective in SoftTypingObjective.from_rules(rules)]
    )
    result = suite.evaluate(bundle())

    assert tuple(result.terms) == tuple(rules)
    assert result.terms["constant"].count == 1
    assert result.terms["modal"].count == 0
    assert result.total.item() == pytest.approx(by_hand(bundle()))


def test_distribution_arity_must_match_the_compiled_plan() -> None:
    rule = TypingRule("identity", (None,), (A,), A)
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (),
            ({(): probabilities(((E, 1.0),))},),
            {(): probabilities(((E, 1.0),))},
        ),
        scalar(1.0),
        scalar(1.0),
    )

    with pytest.raises(ValueError, match="has 1 bindings"):
        SoftTypingObjective(rule).evaluate(bundle(instances=(instance,)))


def test_an_active_binding_must_supply_marginals() -> None:
    rule = TypingRule("binder", (A,), (Node("t"),), Node("t"))
    instance = TensorSoftTypingInstance(
        (0, 2),
        1,
        rule.op,
        TensorSoftTypingDistributions(
            (None,),
            ({(): probabilities(((T, 1.0),))},),
            {(): probabilities(((T, 1.0),))},
        ),
        scalar(1.0),
        scalar(1.0),
    )

    with pytest.raises(ValueError, match="no binding distributions"):
        SoftTypingObjective(rule).evaluate(bundle(instances=(instance,)))


def test_a_rule_key_must_match_the_declared_operator() -> None:
    with pytest.raises(ValueError, match="disagrees"):
        SoftTypingObjective.from_rules({"other": CONSTANT})


def test_the_objective_requires_a_cell_tensor_for_a_connected_zero() -> None:
    outputs = bundle()
    without_cells = ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=outputs.batch,
    )

    with pytest.raises(MissingOutputError, match="cells"):
        objective().evaluate(without_cells)


def test_repr_names_the_declared_operator() -> None:
    assert repr(objective()) == "SoftTypingObjective(op='constant')"
