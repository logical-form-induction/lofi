import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite

from lofi.data import SampledPair
from lofi.deduce import SpanIndex
from lofi.judge import Action
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._combinator import CombinatorSupervisionObjective
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._schedule import TeacherForcingSchedule
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.ports import ForwardMode, ForwardOptions, ensure_mode
from lofi.terms import (
    BatchEncoding,
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

    from lofi.nn.loss._ports import (
        CombinatorScorer,
        LossTerm,
    )
    from lofi.terms import Term


SIGNATURE: Final = Signature(
    "composed",
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
FUNCTOR: Final = INDEX.index(SIGNATURE.by_name("f"))
PREFIX: Final = (0, 1, 4)
FAVORED: Final = 3.0
A: Final = STORE.op(SIGNATURE.by_name("A"))
B: Final = STORE.op(SIGNATURE.by_name("B"))
APPLY: Final = Action("apply", "application")
COMPOSE: Final = Action("compose", "composition")
RAISE: Final = Action("raise", "type-raising")
HORIZON: Final = 100


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


class Instances(NamedTuple):
    instances: tuple[SampledPair[Term, Action], ...]
    step: int
    horizon: int


class IndicatorEncoder:
    """Encodes a type as its slot-by-symbol indicators, each scaled by a weight."""

    def __init__(self) -> None:
        weight = torch.ones(CAPACITY, SYMBOLS, dtype=torch.float64)
        weight[:, INDEX.index(SIGNATURE.by_name("A"))] = FAVORED
        self.weight = weight.requires_grad_()

    @property
    def output_width(self) -> int:
        return CAPACITY * SYMBOLS

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        indicators = torch.nn.functional.one_hot(symbols, SYMBOLS).to(torch.float64)
        vectors = (indicators * self.weight).reshape(symbols.shape[0], -1)
        live = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        held = live.any(dim=1)

        return Encoded(vectors, ("term", "width"), None if bool(held.all()) else held)


class ApplicationScorer:
    """Reads the functor at the root and each daughter's prefix below it, at a gain."""

    def __init__(
        self, gain: float = 1.0, encoder: IndicatorEncoder | None = None
    ) -> None:
        self.gain = torch.tensor(gain, dtype=torch.float64).requires_grad_()
        self.passes: list[ForwardOptions] = []
        self.encoder = ENCODER if encoder is None else encoder

    @property
    def arity(self) -> int:
        return 2

    @property
    def slots(self) -> int:
        return CAPACITY

    @property
    def modes(self) -> frozenset[ForwardMode]:
        return frozenset({ForwardMode.OBSERVE, ForwardMode.SAMPLE})

    def logits(self, left: Tensor, right: Tensor) -> Tensor:
        rows = left.shape[0]
        first = left.reshape(rows, CAPACITY, SYMBOLS)
        second = right.reshape(rows, CAPACITY, SYMBOLS)
        root = torch.zeros(rows, SYMBOLS, dtype=torch.float64)
        root[:, FUNCTOR] = FAVORED
        slots = [root, *(first[:, slot] for slot in PREFIX)]
        slots.extend(second[:, slot] for slot in PREFIX)

        return torch.stack(slots, dim=1) * self.gain

    def score(
        self,
        premises: tuple[tuple[Term, ...], ...],
        /,
        *,
        conclusions: tuple[Term, ...],
        options: ForwardOptions,
    ) -> LogDistribution:
        ensure_mode(options, self.modes)
        self.passes.append(options)
        left = self.encoder.encode(
            CODEC.encode_batch(tuple(inputs[0] for inputs in premises))
        )
        right = self.encoder.encode(
            CODEC.encode_batch(tuple(inputs[1] for inputs in premises))
        )
        logits = self.logits(left.tensor, right.tensor).log_softmax(dim=-1)
        terms = CODEC.encode_batch(conclusions)
        symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        summed = logits.gather(2, symbols.unsqueeze(-1)).squeeze(-1).sum(dim=1)
        masks = [mask for mask in (left.mask, right.mask) if mask is not None]

        if not masks:
            return LogDistribution(summed, ("term",))

        live = masks[0] if len(masks) == 1 else masks[0] & masks[1]

        return LogDistribution(summed.masked_fill(~live, -torch.inf), ("term",), live)

    def exact_matches(
        self,
        premises: tuple[tuple[Term, ...], ...],
        /,
        *,
        conclusions: tuple[Term, ...],
        options: ForwardOptions,
    ) -> Tensor:
        ensure_mode(options, self.modes)
        left = self.encoder.encode(
            CODEC.encode_batch(tuple(inputs[0] for inputs in premises))
        )
        right = self.encoder.encode(
            CODEC.encode_batch(tuple(inputs[1] for inputs in premises))
        )
        predicted = self.logits(left.tensor, right.tensor).argmax(dim=-1)
        targets = CODEC.encode_batch(conclusions)
        symbols = torch.tensor(targets.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        occupied = torch.tensor(targets.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        return ((predicted == symbols) | ~occupied).all(dim=-1)


class ObservingScorer(ApplicationScorer):
    @property
    @override
    def modes(self) -> frozenset[ForwardMode]:
        return frozenset({ForwardMode.OBSERVE})


class WideScorer(ApplicationScorer):
    @override
    def score(
        self,
        premises: tuple[tuple[Term, ...], ...],
        /,
        *,
        conclusions: tuple[Term, ...],
        options: ForwardOptions,
    ) -> LogDistribution:
        scored = super().score(premises, conclusions=conclusions, options=options)

        return LogDistribution(scored.tensor.unsqueeze(-1), ("term", "slot"))


ENCODER: Final = IndicatorEncoder()
DECODERS: Final[Mapping[Action, ApplicationScorer]] = {
    APPLY: ApplicationScorer(1.0),
    COMPOSE: ApplicationScorer(0.5),
}


def objective(
    decoders: Mapping[Action, ApplicationScorer] = DECODERS,
    *,
    schedule: TeacherForcingSchedule | None = None,
) -> CombinatorSupervisionObjective[Instances]:
    return CombinatorSupervisionObjective(
        cast("Mapping[Action, CombinatorScorer]", decoders),
        schedule=schedule,
    )


def bundle_of(
    batch: Instances, *, items: int | None = None
) -> ObjectiveOutputs[Instances]:
    rows = len(batch.instances) if items is None else items

    return ObjectiveOutputs(
        spans=SpanIndex(1),
        chart=CellChart(
            LogDistribution(torch.zeros(rows, 1, dtype=torch.float64), SPAN_AXES)
        ),
        batch=batch,
    )


def batch_over(
    *instances: SampledPair[Term, Action], step: int = 0, horizon: int = HORIZON
) -> Instances:
    return Instances(instances, step, horizon)


def bundle_over(
    *instances: SampledPair[Term, Action], step: int = 0, horizon: int = HORIZON
) -> ObjectiveOutputs[Instances]:
    return bundle_of(batch_over(*instances, step=step, horizon=horizon))


INSTANCES: Final = (
    SampledPair((A, B), f(A, B), (APPLY,)),
    SampledPair((B, A), f(B, A), (COMPOSE,)),
    SampledPair((f(A, B), A), f(f(A, B), A), (APPLY, COMPOSE)),
    SampledPair((A, A), f(A, A), ()),
)


def bundle() -> ObjectiveOutputs[Instances]:
    return bundle_over(*INSTANCES)


def permuted(outputs: ObjectiveOutputs[Instances]) -> ObjectiveOutputs[Instances]:
    order = (2, 0, 3, 1)
    batch = outputs.batch

    return bundle_of(
        Instances(
            tuple(batch.instances[position] for position in order),
            batch.step,
            batch.horizon,
        )
    )


def refilled(outputs: ObjectiveOutputs[Instances]) -> ObjectiveOutputs[Instances]:
    batch = outputs.batch

    return bundle_of(
        Instances(
            (*batch.instances, SampledPair((A, B), f(A, B), ())),
            batch.step,
            batch.horizon,
        )
    )


def unlabeled_bundle() -> ObjectiveOutputs[Instances]:
    return bundle_over(
        SampledPair((A, B), f(A, B), ()),
        SampledPair((B, B), f(B, B), ()),
    )


def inputs_of(outputs: ObjectiveOutputs[Instances]) -> tuple[Tensor, ...]:
    del outputs

    return (ENCODER.weight, DECODERS[APPLY].gain, DECODERS[COMPOSE].gain)


def indicators(term: Term) -> list[list[float]]:
    encoding = CODEC.encode(term)
    rows = [[0.0] * SYMBOLS for _ in range(CAPACITY)]

    for slot, symbol in enumerate(encoding.symbols):
        rows[slot][symbol] = ENCODER.weight[slot, symbol].item()

    return rows


def instance_nll(instance: SampledPair[Term, Action], gain: float) -> float:
    left = indicators(instance.inputs[0])
    right = indicators(instance.inputs[1])
    root = [FAVORED if symbol == FUNCTOR else 0.0 for symbol in range(SYMBOLS)]
    slots = [root, *(left[slot] for slot in PREFIX), *(right[slot] for slot in PREFIX)]
    total = 0.0

    for logits, symbol in zip(
        slots, CODEC.encode(instance.output).symbols, strict=True
    ):
        scaled = [gain * logit for logit in logits]
        total += math.log(sum(math.exp(logit) for logit in scaled)) - scaled[symbol]

    return total


def by_hand(outputs: ObjectiveOutputs[Instances]) -> float:
    batch = outputs.batch
    scored = []

    for instance in batch.instances:
        label = instance.actions
        scored.extend(
            instance_nll(instance, DECODERS[action].gain.item()) for action in label
        )

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
            bundle_over(SampledPair((A, B), f(B, A), (APPLY,))),
            bundle_over(SampledPair((A, B), f(A, B), (APPLY,))),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=unlabeled_bundle,
        tolerance=1e-9,
    )


class TestCombinatorSupervisionObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Instances]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a conclusion the decoder cannot read scores above one it can."""
        the_value_is_monotone_where_the_fixture_says(conformance)


def test_the_value_is_the_mean_over_labels_of_the_slot_summed_nll() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.count == 4
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.diagnostics["per_slot"].item() == pytest.approx(
        by_hand(outputs) / CAPACITY, rel=1e-12
    )
    assert term.diagnostics["drawn"].item() == 0.0
    assert term.diagnostics["exact_match"].item() == 1.0


def test_an_instance_two_actions_admit_trains_both_decoders() -> None:
    shared = SampledPair((f(A, B), A), f(f(A, B), A), (APPLY, COMPOSE))
    both = objective().evaluate(bundle_over(shared))

    assert both.count == 2
    assert both.value.item() == pytest.approx(
        (instance_nll(shared, 1.0) + instance_nll(shared, 0.5)) / 2, rel=1e-12
    )


def test_an_instance_no_action_admits_is_neither_scored_nor_counted() -> None:
    labeled = objective().evaluate(bundle_over(*INSTANCES[:3]))
    padded = objective().evaluate(bundle())

    assert padded.count == labeled.count
    assert padded.value.item() == pytest.approx(labeled.value.item(), rel=1e-12)


def test_a_batch_no_action_admits_is_inapplicable() -> None:
    term = objective().evaluate(unlabeled_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


def test_the_schedule_names_the_pass_each_decoder_runs_at_the_batch_s_step() -> None:
    decoders = {APPLY: ApplicationScorer(), COMPOSE: ApplicationScorer()}
    built = objective(decoders)

    first = built.evaluate(bundle_over(*INSTANCES, step=0))
    quarter = built.evaluate(bundle_over(*INSTANCES, step=HORIZON // 4))
    late = built.evaluate(bundle_over(*INSTANCES, step=HORIZON - 1))

    assert [options.mode for options in decoders[APPLY].passes] == [
        ForwardMode.OBSERVE,
        ForwardMode.SAMPLE,
        ForwardMode.SAMPLE,
    ]
    assert decoders[COMPOSE].passes[1].sampling_rate == pytest.approx(0.5)
    assert first.diagnostics["drawn"].item() == 0.0
    assert quarter.diagnostics["drawn"].item() == pytest.approx(0.5)
    assert late.diagnostics["drawn"].item() == 1.0
    assert built.schedule == TeacherForcingSchedule()


def test_a_schedule_that_never_draws_admits_a_decoder_that_only_observes() -> None:
    observing = {APPLY: ObservingScorer(), COMPOSE: ObservingScorer()}
    built = objective(observing, schedule=TeacherForcingSchedule(None))

    term = built.evaluate(bundle_over(*INSTANCES, step=HORIZON - 1))

    assert term.diagnostics["drawn"].item() == 0.0
    assert all(
        options.mode is ForwardMode.OBSERVE for options in observing[APPLY].passes
    )


def test_a_decoder_that_does_not_run_a_scheduled_pass_is_refused() -> None:
    with pytest.raises(ValueError, match="SAMPLE"):
        objective({APPLY: ApplicationScorer(), COMPOSE: ObservingScorer()})


def test_a_family_with_no_decoder_is_refused() -> None:
    with pytest.raises(ValueError, match="at least one decoder"):
        objective({})


def test_the_decoders_are_keyed_by_action_identity_not_by_position() -> None:
    outputs = bundle_over(INSTANCES[0], INSTANCES[2])
    forward = objective({APPLY: DECODERS[APPLY], COMPOSE: DECODERS[COMPOSE]}).evaluate(
        outputs
    )
    backward = objective({COMPOSE: DECODERS[COMPOSE], APPLY: DECODERS[APPLY]}).evaluate(
        outputs
    )
    swapped = objective({APPLY: DECODERS[COMPOSE], COMPOSE: DECODERS[APPLY]}).evaluate(
        outputs
    )

    assert backward.value.item() == forward.value.item()
    assert swapped.value.item() != pytest.approx(forward.value.item())


def test_a_label_naming_an_action_without_a_decoder_is_refused() -> None:
    with pytest.raises(ValueError, match="raise"):
        objective().evaluate(bundle_over(SampledPair((A, B), f(A, B), (RAISE,))))


def test_a_decoder_more_confident_in_the_conclusion_scores_lower() -> None:
    outputs = bundle_over(SampledPair((A, B), f(A, B), (APPLY,)))

    unsure = objective({APPLY: ApplicationScorer(0.5)}).evaluate(outputs)
    sure = objective({APPLY: ApplicationScorer(2.0)}).evaluate(outputs)

    assert sure.value.item() < unsure.value.item()


def test_the_value_reaches_the_encoder_and_every_labeled_decoder() -> None:
    encoder = IndicatorEncoder()
    decoders = {
        APPLY: ApplicationScorer(encoder=encoder),
        COMPOSE: ApplicationScorer(encoder=encoder),
    }

    objective(decoders).evaluate(bundle()).value.backward()

    assert encoder.weight.grad is not None
    assert bool(encoder.weight.grad.abs().sum() > 0.0)
    assert all(
        decoder.gain.grad is not None and decoder.gain.grad.item() != 0.0
        for decoder in decoders.values()
    )


def test_a_decoder_that_does_not_weigh_one_conclusion_per_instance_is_refused() -> None:
    with pytest.raises(ValueError, match="terms"):
        objective({APPLY: WideScorer(), COMPOSE: WideScorer()}).evaluate(bundle())


def test_an_instance_at_the_wrong_arity_is_refused() -> None:
    wrong = SampledPair((A,), f(A, B), (APPLY,))

    with pytest.raises(ValueError, match="arity 1"):
        objective().evaluate(bundle_over(wrong))


def test_the_objective_declares_what_it_reads() -> None:
    built = objective()

    assert built.requires == frozenset({"batch"})
    assert built.name == "combinator-supervision"
    assert dict(built.decoders) == dict(DECODERS)
    assert "apply: ApplicationScorer" in repr(built)


def test_a_blank_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        CombinatorSupervisionObjective(
            cast("Mapping[Action, CombinatorScorer]", DECODERS),
            name="",
        )


def test_the_term_reaches_the_suite_under_its_name() -> None:
    outputs = bundle()
    built = objective()

    result = ObjectiveSuite([(0.25, built)]).evaluate(outputs)

    assert result.total.item() == pytest.approx(
        0.25 * built.evaluate(outputs).value.item(), rel=1e-12
    )
    assert set(result.diagnostics) == {
        "combinator-supervision/loss",
        "combinator-supervision/exact_match",
        "combinator-supervision/per_slot",
        "combinator-supervision/drawn",
    }
