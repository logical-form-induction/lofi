import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests._controllers import mlp_controller
from tests.conformance.objective import ObjectiveLawSuite

from lofi.data import SampledPair
from lofi.deduce import SpanIndex
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.loss._controller import ControllerSupervisionObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.sites import ComponentId
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from torch import Tensor

    from lofi.nn.controllers import MlpController
    from lofi.nn.loss._ports import LossTerm
    from lofi.nn.ports import Controller, TermEncoder
    from lofi.terms import Term


SIGNATURE: Final = Signature(
    "controlled",
    ("T",),
    (
        Operator("A", "T", family="atom"),
        Operator("B", "T", family="atom"),
        Operator("f", "T", arg_sorts=("T", "T"), family="functor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CODEC: Final = TermCodec(
    INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=3)
)
STORE: Final = TermStore(SIGNATURE)
CAPACITY: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
WIDTH: Final = CAPACITY * SYMBOLS
A: Final = STORE.op(SIGNATURE.by_name("A"))
B: Final = STORE.op(SIGNATURE.by_name("B"))
IDENTITY: Final = Action("identity")
APPLY: Final = Action("apply", "application")
COMPOSE: Final = Action("compose", "composition")
RAISE: Final = Action("raise", "type-raising")
ACTIONS: Final = ActionSet((IDENTITY, APPLY, COMPOSE), IDENTITY)
REVERSED: Final = ActionSet((COMPOSE, APPLY, IDENTITY), IDENTITY)


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


class Instances(NamedTuple):
    instances: tuple[SampledPair[Term, Action], ...]


class IndicatorEncoder:
    """Encodes a type as its occupied slot-by-symbol indicators, each scaled."""

    def __init__(self) -> None:
        self.weight = torch.full(
            (CAPACITY, SYMBOLS), 1.5, dtype=torch.float64
        ).requires_grad_()

    @property
    def output_width(self) -> int:
        return WIDTH

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        occupied = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        indicators = torch.nn.functional.one_hot(symbols, SYMBOLS).to(torch.float64)
        kept = indicators * occupied.unsqueeze(-1).to(torch.float64)
        vectors = (kept * self.weight).reshape(symbols.shape[0], -1)
        held = occupied.any(dim=1)

        return Encoded(vectors, ("term", "width"), None if bool(held.all()) else held)


def probe(symbol: str) -> Tensor:
    """Return a vector reading how much of one symbol a pair of encodings holds.

    Parameters
    ----------
    symbol : str
    """
    table = torch.zeros(CAPACITY, SYMBOLS, dtype=torch.float64)
    table[:, INDEX.index(SIGNATURE.by_name(symbol))] = 1.0

    return table.reshape(WIDTH).requires_grad_()


class TableController:
    """Weighs each action by a vector held under the action's identity."""

    def __init__(self, action_set: ActionSet, table: Mapping[Action, Tensor]) -> None:
        self._action_set = action_set
        self.table = dict(table)

    @property
    def action_set(self) -> ActionSet:
        return self._action_set

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        summed = left.tensor + right.tensor
        columns = torch.stack(
            [summed @ self.table[action] for action in self._action_set.actions],
            dim=1,
        )
        masks = [mask for mask in (left.mask, right.mask) if mask is not None]

        if not masks:
            return LogDistribution(columns, ("cell", "action"))

        live = masks[0] if len(masks) == 1 else masks[0] & masks[1]

        return LogDistribution(
            columns.masked_fill(~live.unsqueeze(1), -torch.inf),
            ("cell", "action"),
            live,
        )


class LogMaskingController(TableController):
    """Masks a dead pair by adding the log of its mask, so its weights are all -inf."""

    @override
    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        weighed = super().weigh(left, right=right)
        live = weighed.mask

        if live is None:
            return weighed

        summed = left.tensor + right.tensor
        columns = torch.stack(
            [summed @ self.table[action] for action in self.action_set.actions],
            dim=1,
        )
        masked = columns + torch.log(live.to(columns.dtype)).unsqueeze(1)

        return LogDistribution(masked, ("cell", "action"), live)


class UnmaskedController(TableController):
    """Weighs every pair regardless of the daughters' masks."""

    @override
    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        weighed = super().weigh(
            Encoded(left.tensor, left.axes), right=Encoded(right.tensor, right.axes)
        )

        return LogDistribution(weighed.tensor.log_softmax(dim=-1), weighed.axes)


class RefusingController(TableController):
    """Weighs no action at the last pair regardless of its daughters."""

    @override
    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        weighed = super().weigh(left, right=right)
        live = torch.ones(weighed.tensor.shape[0], dtype=torch.bool)
        live[-1] = False

        return LogDistribution(
            weighed.tensor.masked_fill(~live.unsqueeze(1), -torch.inf),
            weighed.axes,
            live,
        )


class WideController(TableController):
    @override
    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        weighed = super().weigh(left, right=right)

        return LogDistribution(weighed.tensor[:, :2], ("cell", "action"))


ENCODER: Final = IndicatorEncoder()
TABLE: Final[Mapping[Action, Tensor]] = {
    IDENTITY: torch.zeros(WIDTH, dtype=torch.float64).requires_grad_(),
    APPLY: probe("A"),
    COMPOSE: probe("B"),
}
CONTROLLER: Final = TableController(ACTIONS, TABLE)


def objective(
    controller: TableController = CONTROLLER,
    *,
    encoder: IndicatorEncoder = ENCODER,
) -> ControllerSupervisionObjective[Instances]:
    return ControllerSupervisionObjective(
        cast("TermEncoder", encoder), CODEC, cast("Controller", controller)
    )


def bundle_of(batch: Instances) -> ObjectiveOutputs[Instances]:
    return ObjectiveOutputs(
        spans=SpanIndex(1),
        chart=CellChart(
            LogDistribution(
                torch.zeros(len(batch.instances), 1, dtype=torch.float64), SPAN_AXES
            )
        ),
        batch=batch,
    )


def batch_over(*instances: SampledPair[Term, Action]) -> Instances:
    return Instances(instances)


def bundle_over(
    *instances: SampledPair[Term, Action],
) -> ObjectiveOutputs[Instances]:
    return bundle_of(batch_over(*instances))


INSTANCES: Final = (
    SampledPair((A, f(A, B)), f(A, f(A, B)), (APPLY,)),
    SampledPair((B, B), f(B, B), (COMPOSE,)),
    SampledPair((f(A, B), A), f(f(A, B), A), (APPLY, COMPOSE)),
    SampledPair((A, B), B, ()),
)


def bundle() -> ObjectiveOutputs[Instances]:
    return bundle_over(*INSTANCES)


def permuted(outputs: ObjectiveOutputs[Instances]) -> ObjectiveOutputs[Instances]:
    order = (3, 0, 2, 1)
    batch = outputs.batch

    return bundle_of(Instances(tuple(batch.instances[position] for position in order)))


def refilled(outputs: ObjectiveOutputs[Instances]) -> ObjectiveOutputs[Instances]:
    batch = outputs.batch

    return bundle_of(
        Instances(
            (*batch.instances, SampledPair((A,), f(A, B), (RAISE,))),
        )
    )


def refilled_on_the_left(
    outputs: ObjectiveOutputs[Instances],
) -> ObjectiveOutputs[Instances]:
    return refilled(outputs)


def empty_bundle() -> ObjectiveOutputs[Instances]:
    return bundle_of(Instances((SampledPair((A,), A, (RAISE,)),)))


def inputs_of(outputs: ObjectiveOutputs[Instances]) -> tuple[Tensor, ...]:
    del outputs

    return (ENCODER.weight, TABLE[APPLY], TABLE[COMPOSE])


def held(term: Term, symbol: str) -> float:
    """Return how much of one symbol a type's encoding holds, by hand.

    Parameters
    ----------
    term : Term
    symbol : str
    """
    encoding = CODEC.encode(term)
    code = INDEX.index(SIGNATURE.by_name(symbol))

    return sum(
        ENCODER.weight[slot, code].item()
        for slot, read in enumerate(encoding.symbols)
        if encoding.occupied[slot] and read == code
    )


def logits_of(instance: SampledPair[Term, Action]) -> dict[Action, float]:
    return {
        IDENTITY: 0.0,
        APPLY: held(instance.inputs[0], "A") + held(instance.inputs[1], "A"),
        COMPOSE: held(instance.inputs[0], "B") + held(instance.inputs[1], "B"),
    }


def instance_nll(instance: SampledPair[Term, Action]) -> float:
    logits = logits_of(instance)
    partition = math.log(sum(math.exp(logit) for logit in logits.values()))
    label = instance.actions or (IDENTITY,)

    return partition - math.log(sum(math.exp(logits[action]) for action in label))


def by_hand(outputs: ObjectiveOutputs[Instances]) -> float:
    batch = outputs.batch
    scored = [
        instance_nll(instance)
        for instance in batch.instances
        if len(instance.inputs) == 2
    ]

    return sum(scored) / len(scored) if scored else 0.0


def models_built(call: Callable[[], LossTerm]) -> int:
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


def fixture() -> ObjectiveFixture[Instances]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (
            bundle_over(SampledPair((A, A), f(A, A), (COMPOSE,))),
            bundle_over(SampledPair((A, A), f(A, A), (APPLY,))),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
        tolerance=1e-9,
    )


class TestControllerSupervisionObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Instances]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a label the controller weighs lightly scores above a heavy one.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_value_is_the_mean_negative_log_mass_of_the_labels() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.count == len(INSTANCES)
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)


def test_an_instance_two_actions_admit_is_scored_on_their_total_mass() -> None:
    shared = SampledPair((f(A, B), A), f(f(A, B), A), (APPLY, COMPOSE))
    logits = logits_of(shared)
    partition = math.log(sum(math.exp(logit) for logit in logits.values()))

    term = objective().evaluate(bundle_over(shared))

    assert term.value.item() == pytest.approx(
        partition - math.log(math.exp(logits[APPLY]) + math.exp(logits[COMPOSE])),
        rel=1e-12,
    )
    assert (
        term.value.item()
        < objective()
        .evaluate(bundle_over(SampledPair((f(A, B), A), f(f(A, B), A), (APPLY,))))
        .value.item()
    )


def test_an_instance_no_action_admits_is_scored_on_the_identity() -> None:
    refused = SampledPair((A, B), B, ())
    logits = logits_of(refused)
    partition = math.log(sum(math.exp(logit) for logit in logits.values()))

    unlabeled = objective().evaluate(bundle_over(refused))
    identity = objective().evaluate(bundle_over(SampledPair((A, B), B, (IDENTITY,))))

    assert unlabeled.value.item() == pytest.approx(partition, rel=1e-12)
    assert identity.value.item() == pytest.approx(unlabeled.value.item(), rel=1e-12)


NONE_APPLIES: Final = Action("none-applies")
NULL_CONTROLLER: Final = TableController(
    ActionSet((IDENTITY, APPLY, COMPOSE, NONE_APPLIES), IDENTITY, null=NONE_APPLIES),
    {**TABLE, NONE_APPLIES: (2.0 * probe("A")).detach()},
)


def test_an_instance_no_action_admits_is_scored_on_a_declared_null_action() -> None:
    supervised = objective(NULL_CONTROLLER)

    unlabeled = supervised.evaluate(bundle_over(SampledPair((A, B), B, ())))
    null = supervised.evaluate(bundle_over(SampledPair((A, B), B, (NONE_APPLIES,))))
    identity = supervised.evaluate(bundle_over(SampledPair((A, B), B, (IDENTITY,))))

    assert unlabeled.value.item() == pytest.approx(null.value.item(), rel=1e-12)
    assert unlabeled.value.item() != pytest.approx(identity.value.item(), rel=1e-6)


def test_the_label_is_resolved_by_action_identity_not_by_column() -> None:
    outputs = bundle()
    forward = objective(TableController(ACTIONS, TABLE)).evaluate(outputs)
    backward = objective(TableController(REVERSED, TABLE)).evaluate(outputs)

    assert backward.value.item() == pytest.approx(forward.value.item(), rel=1e-12)
    assert backward.diagnostics["accuracy"].item() == pytest.approx(
        forward.diagnostics["accuracy"].item()
    )
    assert objective(TableController(REVERSED, TABLE)).action_set == REVERSED


def test_the_accuracy_is_the_share_of_instances_whose_heaviest_action_is_labeled() -> (
    None
):
    term = objective().evaluate(bundle())
    hits = [
        max(logits_of(instance), key=lambda action: logits_of(instance)[action])
        in (instance.actions or (IDENTITY,))
        for instance in INSTANCES
    ]

    assert 0 < sum(hits) < len(hits)
    assert term.diagnostics["accuracy"].item() == pytest.approx(
        sum(hits) / len(hits), rel=1e-12
    )
    assert term.diagnostics["accuracy"].grad_fn is None


def test_a_label_naming_an_action_the_controller_does_not_weigh_is_refused() -> None:
    with pytest.raises(ValueError, match="raise"):
        objective().evaluate(bundle_over(SampledPair((A, B), f(A, B), (RAISE,))))


def test_a_label_naming_an_action_of_the_same_name_under_another_rule_is_refused() -> (
    None
):
    renamed = Action("apply", "backward-application")

    with pytest.raises(ValueError, match="backward-application"):
        objective().evaluate(bundle_over(SampledPair((A, B), f(A, B), (renamed,))))


def test_a_controller_favoring_the_label_scores_lower_after_a_step() -> None:
    table = {
        IDENTITY: torch.zeros(WIDTH, dtype=torch.float64).requires_grad_(),
        APPLY: probe("A"),
        COMPOSE: probe("B"),
    }
    built = objective(TableController(ACTIONS, table))
    outputs = bundle_over(SampledPair((B, B), f(B, B), (APPLY,)))

    before = built.evaluate(outputs)
    before.value.backward()
    gradient = table[APPLY].grad

    assert gradient is not None

    with torch.no_grad():
        table[APPLY] -= 0.1 * gradient

    assert built.evaluate(outputs).value.item() < before.value.item()


def test_an_instance_missing_a_daughter_is_neither_scored_nor_counted() -> None:
    encoder = IndicatorEncoder()
    controller = LogMaskingController(ACTIONS, TABLE)
    complete = objective(controller, encoder=encoder).evaluate(bundle())
    padded = objective(controller, encoder=encoder).evaluate(refilled(bundle()))

    padded.value.backward()

    assert padded.count == len(INSTANCES)
    assert padded.value.item() == pytest.approx(complete.value.item(), rel=1e-12)
    assert padded.diagnostics["accuracy"].item() == pytest.approx(
        complete.diagnostics["accuracy"].item(), rel=1e-12
    )
    assert encoder.weight.grad is not None
    assert bool(torch.isfinite(encoder.weight.grad).all())


@pytest.mark.parametrize("side", ["left", "right"])
def test_a_dead_daughter_is_dropped_whichever_side_it_is_on_and_whoever_masks(
    side: str,
) -> None:
    encoder = IndicatorEncoder()
    controller = UnmaskedController(ACTIONS, TABLE)
    padding = refilled_on_the_left if side == "left" else refilled
    complete = objective(controller, encoder=encoder).evaluate(bundle())
    padded = objective(controller, encoder=encoder).evaluate(padding(bundle()))

    padded.value.backward()

    assert padded.count == len(INSTANCES)
    assert padded.value.item() == pytest.approx(complete.value.item(), rel=1e-12)
    assert padded.diagnostics["accuracy"].item() == pytest.approx(
        complete.diagnostics["accuracy"].item(), rel=1e-12
    )
    assert encoder.weight.grad is not None
    assert bool(torch.isfinite(encoder.weight.grad).all())


def test_a_pair_the_controller_itself_masks_is_neither_scored_nor_counted() -> None:
    controller = RefusingController(ACTIONS, TABLE)
    outputs = bundle()

    complete = objective().evaluate(bundle_over(*INSTANCES[:-1]))
    refused = objective(controller).evaluate(outputs)

    assert refused.count == len(INSTANCES) - 1
    assert refused.value.item() == pytest.approx(complete.value.item(), rel=1e-12)
    assert refused.diagnostics["accuracy"].item() == pytest.approx(
        complete.diagnostics["accuracy"].item(), rel=1e-12
    )


def shipped() -> tuple[TreeRnnEncoder, MlpController]:
    """Return the tree recurrence encoder and the controller over the fixture codec."""
    torch.manual_seed(5)
    encoder = TreeRnnEncoder(
        codec=NeuralCodecView(CODEC),
        choices=ChoiceIndex(
            SIGNATURE,
            depth_bounds=dict.fromkeys(SIGNATURE.sorts, CODEC.capacity),
        ),
        component=ComponentId("tree-encoder"),
        scheme=ArityHeadScheme(width=8, reached=SIGNATURE.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    ).to(torch.float64)
    controller = mlp_controller(
        ACTIONS,
        width=encoder.output_width,
        component=ComponentId("controller"),
        hidden=4,
    ).to(torch.float64)

    return encoder, controller


@pytest.mark.parametrize("side", ["left", "right"])
def test_the_shipped_controller_scores_no_pair_missing_a_daughter(side: str) -> None:
    encoder, controller = shipped()
    built = ControllerSupervisionObjective(encoder, CODEC, controller)
    padding = refilled_on_the_left if side == "left" else refilled
    outputs = bundle()
    batch = outputs.batch

    complete = built.evaluate(outputs)
    padded = built.evaluate(padding(outputs))
    left = encoder.encode(
        CODEC.encode_batch(tuple(instance.inputs[0] for instance in batch.instances))
    )
    right = encoder.encode(
        CODEC.encode_batch(tuple(instance.inputs[1] for instance in batch.instances))
    )
    weighed = controller.weigh(
        Encoded(left.tensor, ("cell", "width")),
        right=Encoded(right.tensor, ("cell", "width")),
    )
    admitted = torch.tensor(
        [
            [action in (label or (IDENTITY,)) for action in ACTIONS.actions]
            for instance in batch.instances
            for label in (instance.actions,)
        ]
    )
    by_hand = -torch.where(admitted, weighed.tensor, -torch.inf).logsumexp(dim=-1)

    assert weighed.mask is None
    assert complete.count == len(INSTANCES)
    assert complete.value.item() == pytest.approx(by_hand.mean().item(), rel=1e-12)
    assert padded.count == complete.count
    assert padded.value.item() == pytest.approx(complete.value.item(), rel=1e-12)


def test_a_batch_holding_no_pair_is_inapplicable() -> None:
    term = objective().evaluate(empty_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


def test_a_controller_that_does_not_weigh_every_action_is_refused() -> None:
    with pytest.raises(ValueError, match="3 actions"):
        objective(WideController(ACTIONS, TABLE)).evaluate(bundle())


def test_nonbinary_instances_are_left_to_their_rule_family() -> None:
    term = objective().evaluate(
        bundle_over(
            SampledPair((A,), A, (RAISE,)),
            SampledPair((A, B), f(A, B), (APPLY,)),
        )
    )

    assert term.count == 1


def test_the_objective_declares_what_it_reads() -> None:
    built = objective()

    assert built.requires == frozenset({"batch"})
    assert built.name == "controller-supervision"
    assert built.action_set == ACTIONS
    assert "TableController" in repr(built)


def test_a_blank_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        ControllerSupervisionObjective(
            cast("TermEncoder", ENCODER),
            CODEC,
            cast("Controller", CONTROLLER),
            name="",
        )


def test_the_term_reaches_the_suite_under_its_name() -> None:
    outputs = bundle()
    built = objective()

    result = ObjectiveSuite([(1.5, built)]).evaluate(outputs)

    assert result.total.item() == pytest.approx(
        1.5 * built.evaluate(outputs).value.item(), rel=1e-12
    )
    assert set(result.diagnostics) == {
        "controller-supervision/loss",
        "controller-supervision/accuracy",
    }
