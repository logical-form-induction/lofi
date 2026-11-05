from itertools import pairwise
from typing import TYPE_CHECKING, Final, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.train_step import TrainStepConformance

from lofi.data import (
    ArityBranching,
    PairDistribution,
    TypeDistribution,
    UniformChildDepth,
    UniformExhaustiveWeighting,
    term_depth,
)
from lofi.data.synthetic import (
    ExhaustiveTypeSampler,
    RecursiveTypeSampler,
    RulePairSampler,
)
from lofi.judge import Action, ActionSet
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.loss._autoencoder import TypeAutoencoderObjective
from lofi.nn.loss._combinator import CombinatorSupervisionObjective
from lofi.nn.loss._controller import ControllerSupervisionObjective
from lofi.nn.loss._ports import (
    LossTerm,
)
from lofi.nn.loss._schedule import TeacherForcingSchedule
from lofi.nn.ports import ForwardMode, ForwardOptions, ensure_mode
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
    TreeEditDistance,
)
from lofi.train import (
    CheckpointManifest,
    ComponentRecord,
    LossAssembly,
    OptimizationResult,
    PrecisionRecord,
    RngRecord,
    StateReference,
    StepGraphError,
    WeightedObjective,
    disagreements,
    edges,
)
from lofi.train._pretraining import (
    ControllerStep,
    DistanceBatcher,
    Draw,
    LinearDepthCurriculum,
    PairDraws,
    RuleBatch,
    TermsBatch,
    TermsStep,
    Trained,
    TrainedController,
    TypeBatch,
    distance_batch_seed,
    scheduled_depth,
    terms_batch_seed,
)
from lofi.train.laws import StepFixture

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from torch import Tensor

    from lofi.data import PairSampler
    from lofi.nn.loss._ports import (
        CombinatorScorer,
        PublishedOutputs,
    )
    from lofi.nn.ports import Controller, TermEncoder
    from lofi.terms import BatchEncoding, Term
    from lofi.train import StepDeclaration, TrainResult


SIGNATURE: Final = Signature(
    "pretrained",
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
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"T": CODEC.capacity})
CAPACITY: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
WIDTH: Final = CAPACITY * SYMBOLS

IDENTITY: Final = Action("identity")
APPLY: Final = Action("apply", "application")
SWAP: Final = Action("swap", "swapping")
ACTIONS: Final = ActionSet((IDENTITY, APPLY, SWAP), IDENTITY)
REVERSED: Final = ActionSet((SWAP, APPLY, IDENTITY), IDENTITY)

TYPES: Final = TypeDistribution(2, {}, max_depth=3)
DAUGHTERS: Final = TypeDistribution(2, {}, max_depth=3)
PAIRS: Final = PairDistribution(max_depth=3)
AVAILABLE: Final = frozenset({"type_distribution", "pair_distribution", "draw"})
HORIZON: Final = 10


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


class Application:
    @property
    def name(self) -> str:
        return "application"

    @property
    def arity(self) -> int:
        return 2

    @property
    def directions(self) -> frozenset[str]:
        return frozenset({"forward"})

    @property
    def forward_is_complete(self) -> bool:
        return True

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        yield f(inputs[0], inputs[1])

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return output == f(inputs[0], inputs[1])


class Swapping(Application):
    @property
    @override
    def name(self) -> str:
        return "swapping"

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        yield f(inputs[1], inputs[0])

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return output == f(inputs[1], inputs[0])


class Rules:
    def of_arity(self, k: int) -> tuple[Application, ...]:
        return (Application(), Swapping()) if k == 2 else ()

    def __iter__(self) -> Iterator[Application]:
        return iter((Application(), Swapping()))


TYPE_SAMPLER: Final = RecursiveTypeSampler(STORE, max_size=7, sort="T")
DISTANCE_SAMPLER: Final = ExhaustiveTypeSampler(
    STORE, 7, sort="T", weighting=UniformExhaustiveWeighting()
)
DISTANCE_CODEC: Final = TermCodec(
    INDEX, CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=4)
)
PAIR_SAMPLER: Final = cast(
    "PairSampler[Term, Action]",
    RulePairSampler(TYPE_SAMPLER, Rules(), ACTIONS),
)


def _rows(terms: BatchEncoding) -> tuple[Tensor, Tensor]:
    symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
    occupied = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)

    return symbols, occupied


def _summed(logits: Tensor, terms: BatchEncoding) -> Tensor:
    symbols, occupied = _rows(terms)
    picked = logits.log_softmax(dim=-1).gather(2, symbols.unsqueeze(-1)).squeeze(-1)

    return (picked * occupied.to(picked.dtype)).sum(dim=1)


def _live(left: Encoded, right: Encoded) -> Tensor | None:
    masks = [mask for mask in (left.mask, right.mask) if mask is not None]

    if not masks:
        return None

    return masks[0] if len(masks) == 1 else masks[0] & masks[1]


class IndicatorEncoder:
    """Encodes a type as its slot-by-symbol indicators, each scaled by a weight."""

    def __init__(self, scale: float = 1.0) -> None:
        self.weight = torch.full(
            (CAPACITY, SYMBOLS), scale, dtype=torch.float64
        ).requires_grad_()

    @property
    def output_width(self) -> int:
        return WIDTH

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        symbols, occupied = _rows(terms)
        indicators = torch.nn.functional.one_hot(symbols, SYMBOLS).to(torch.float64)
        kept = indicators * occupied.unsqueeze(-1).to(torch.float64)
        vectors = (kept * self.weight).reshape(symbols.shape[0], WIDTH)
        held = occupied.any(dim=1)

        return Encoded(vectors, ("term", "width"), None if bool(held.all()) else held)


class ReadingDecoder:
    """Weighs a type by reading its own indicators back out of the encoding."""

    def __init__(self) -> None:
        self.gain = torch.tensor(1.0, dtype=torch.float64).requires_grad_()

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        logits = encoded.tensor.reshape(-1, CAPACITY, SYMBOLS) * self.gain
        normalized = logits.log_softmax(dim=-1)

        return LogDistribution(
            normalized,
            ("term", "slot", "choice"),
            encoded.mask,
            normalized_over="choice",
        )


class PairScorer:
    """Weighs a conclusion by reading the sum of its daughters' indicators."""

    def __init__(self, encoder: IndicatorEncoder) -> None:
        self.gain = torch.tensor(1.0, dtype=torch.float64).requires_grad_()
        self.passes: list[ForwardOptions] = []
        self.encoder = encoder

    @property
    def arity(self) -> int:
        return 2

    @property
    def slots(self) -> int:
        return CAPACITY

    @property
    def modes(self) -> frozenset[ForwardMode]:
        return frozenset({ForwardMode.OBSERVE, ForwardMode.SAMPLE})

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
        logits = (left.tensor + right.tensor).reshape(-1, CAPACITY, SYMBOLS)
        summed = _summed(logits * self.gain, CODEC.encode_batch(conclusions))
        live = _live(left, right)

        if live is None:
            return LogDistribution(summed, ("term",))

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
        logits = (left.tensor + right.tensor).reshape(-1, CAPACITY, SYMBOLS)
        targets = CODEC.encode_batch(conclusions)
        symbols = torch.tensor(targets.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        occupied = torch.tensor(targets.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        return (((logits * self.gain).argmax(dim=-1) == symbols) | ~occupied).all(
            dim=-1
        )


class TableController:
    """Weighs each action by a learned vector over the summed daughters."""

    def __init__(self, action_set: ActionSet) -> None:
        self._action_set = action_set
        self.table = torch.zeros(
            len(action_set.actions), WIDTH, dtype=torch.float64
        ).requires_grad_()

    @property
    def action_set(self) -> ActionSet:
        return self._action_set

    def weigh(self, left: Encoded, /, *, right: Encoded) -> LogDistribution:
        columns = (left.tensor + right.tensor) @ self.table.T
        live = _live(left, right)

        if live is None:
            return LogDistribution(columns, ("cell", "action"))

        return LogDistribution(
            columns.masked_fill(~live.unsqueeze(1), -torch.inf),
            ("cell", "action"),
            live,
        )


class Descent:
    """Plain gradient descent, reporting whether it stepped."""

    def __init__(self, parameters: Sequence[Tensor], rate: float = 0.1) -> None:
        self.parameters = tuple(parameters)
        self.rate = rate
        self.calls = 0

    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        self.calls += 1

        for parameter in self.parameters:
            parameter.grad = None

        loss = forward()

        if not bool(torch.isfinite(loss)):
            return OptimizationResult(loss_finite=False, stepped=False, scale=None)

        loss.backward()

        with torch.no_grad():
            for parameter in self.parameters:
                if parameter.grad is not None:
                    parameter.sub_(self.rate * parameter.grad)

        return OptimizationResult(loss_finite=True, stepped=True, scale=None)


class Accumulating(Descent):
    """Runs the forward and backward passes and never steps the parameters."""

    @override
    def step(self, forward: Callable[[], Tensor], /) -> OptimizationResult:
        self.calls += 1
        loss = forward()
        loss.backward()

        return OptimizationResult(loss_finite=True, stepped=False, scale=None)


class Witness[BatchT]:
    """Records every batch it is handed and contributes a zero term over it."""

    def __init__(self, name: str = "witness") -> None:
        self._name = name
        self.batches: list[BatchT] = []

    @property
    def name(self) -> str:
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"batch"})

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        batch = outputs.batch
        self.batches.append(batch)

        return LossTerm(
            torch.zeros((), dtype=torch.float64, requires_grad=True),
            len(cast("RuleBatch | TypeBatch", batch)),
        )


class Diverging[BatchT](Witness[BatchT]):
    """Contributes an infinite term over every batch."""

    @override
    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        counted = super().evaluate(outputs)

        return LossTerm(
            torch.full((), torch.inf, dtype=torch.float64, requires_grad=True),
            counted.count,
        )


class Pretraining:
    """The three steps over one encoder, with the parameters the policy moves."""

    def __init__(
        self,
        *,
        rate: float = 0.1,
        distance_weight: float = 0.5,
        schedule: TeacherForcingSchedule | None = None,
        encoder: IndicatorEncoder | None = None,
    ) -> None:
        self.encoder = IndicatorEncoder() if encoder is None else encoder
        self.decoder = ReadingDecoder()
        self.scorers = {
            APPLY: PairScorer(self.encoder),
            SWAP: PairScorer(self.encoder),
        }
        self.controller = TableController(ACTIONS)
        self.rate = rate
        self.distance_weight = distance_weight
        self.schedule = schedule
        self.policies: list[Descent] = []

    def parameters(self) -> tuple[Tensor, ...]:
        return (
            self.encoder.weight,
            self.decoder.gain,
            *(scorer.gain for scorer in self.scorers.values()),
            self.controller.table,
        )

    def policy(self) -> Descent:
        built = Descent(self.parameters(), self.rate)
        self.policies.append(built)

        return built

    def terms_step(
        self, *, with_pairs: bool = True, name: str = "categories"
    ) -> TermsStep:
        """Return one joint generated-term step over the represented fixture."""
        autoencoder = TypeAutoencoderObjective[TermsBatch](
            cast("TermEncoder", self.encoder), self.decoder, CHOICES
        )
        objectives = [WeightedObjective(autoencoder.name, 1.0, autoencoder)]
        roles = ["terms", "identity"]

        if with_pairs:
            supervision = CombinatorSupervisionObjective[TermsBatch](
                cast("dict[Action, CombinatorScorer]", self.scorers),
                schedule=self.schedule,
            )
            objectives.append(WeightedObjective(supervision.name, 1.0, supervision))
            roles.extend(("application", "swapping"))

        assembly = LossAssembly[TermsBatch](tuple(objectives))

        return TermsStep(
            TYPE_SAMPLER,
            TYPES,
            PAIR_SAMPLER if with_pairs else None,
            PAIRS if with_pairs else None,
            CODEC,
            assembly,
            self.policy(),
            roles=roles,
            name=name,
        )

    def controller_step(
        self, *, role: str = "controller", name: str | None = None
    ) -> ControllerStep:
        supervision = ControllerSupervisionObjective[RuleBatch](
            cast("TermEncoder", self.encoder),
            CODEC,
            cast("Controller", self.controller),
        )
        assembly = LossAssembly[RuleBatch](
            (WeightedObjective(supervision.name, 1.0, supervision),)
        )

        return ControllerStep(
            PAIR_SAMPLER,
            self.controller.action_set,
            assembly,
            self.policy(),
            role=role,
            name=name,
        )


def test_distance_batch_seeds_are_pure_functions_of_stream_and_position() -> None:
    assert distance_batch_seed(17, 4) == distance_batch_seed(17, 4)
    assert distance_batch_seed(17, 4) != distance_batch_seed(17, 5)
    assert distance_batch_seed(17, 4) != distance_batch_seed(18, 4)


def test_terms_batch_seeds_are_pure_functions_of_stream_and_position() -> None:
    assert terms_batch_seed(17, 4) == terms_batch_seed(17, 4)
    assert terms_batch_seed(17, 4) != terms_batch_seed(17, 5)
    assert terms_batch_seed(17, 4) != terms_batch_seed(18, 4)


def test_the_depth_curriculum_grows_monotonically_through_available_depths() -> None:
    depths = tuple(range(1, 6))
    curriculum = LinearDepthCurriculum(2, 9)
    scheduled = tuple(scheduled_depth(depths, step, curriculum) for step in range(12))

    assert scheduled[0] == 2
    assert scheduled[-1] == 5
    assert all(left <= right for left, right in pairwise(scheduled))
    assert scheduled_depth(depths, 0, None) == 5


def test_ball_batches_respect_the_radius_and_zero_pairs_a_type_with_itself() -> None:
    radius_one = DistanceBatcher(
        DISTANCE_SAMPLER,
        radius=1,
        curriculum=None,
        branching=ArityBranching({}),
        child_depth=UniformChildDepth(),
        constructor_weights={},
    )
    radius_zero = DistanceBatcher(
        DISTANCE_SAMPLER,
        radius=0,
        curriculum=None,
        branching=ArityBranching({}),
        child_depth=UniformChildDepth(),
        constructor_weights={},
    )
    requested = Draw(31, 128, step=0, horizon=1)
    distance = TreeEditDistance()
    nearby = radius_one.batch(DISTANCE_CODEC, requested)
    identical = radius_zero.batch(DISTANCE_CODEC, requested)

    assert all(distance(left, right) <= 1 for left, right in nearby.pairs)
    assert all(left is right for left, right in identical.pairs)


def test_disabling_the_ball_pairs_consecutive_installed_sampler_draws() -> None:
    batcher = DistanceBatcher(
        DISTANCE_SAMPLER,
        radius=None,
        curriculum=None,
        branching=ArityBranching({}),
        child_depth=UniformChildDepth(),
        constructor_weights={},
    )
    requested = Draw(43, 7, step=0, horizon=1)
    depth = max(map(term_depth, DISTANCE_SAMPLER.inventory))
    distribution = TypeDistribution(depth, {}, max_depth=depth)
    drawn, _ = DISTANCE_SAMPLER.sample(14, distribution, seed=43)
    batch = batcher.batch(DISTANCE_CODEC, requested)

    assert batch.pairs == tuple(zip(drawn[0::2], drawn[1::2], strict=True))


class SeenPairDraws(PairDraws):
    __slots__ = ("_seen",)

    def __init__(self, pair_distribution: PairDistribution, draw: Draw) -> None:
        super().__init__(pair_distribution, draw)
        object.__setattr__(self, "_seen", set())

    @property
    @override
    def pair_distribution(self) -> PairDistribution:
        self._seen.add("pair_distribution")

        return super().pair_distribution

    @property
    @override
    def draw(self) -> Draw:
        self._seen.add("draw")

        return super().draw

    @property
    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)


def draw(count: int = 6, *, seed: int = 7, step: int = 0) -> Draw:
    return Draw(seed, count, step=step, horizon=HORIZON)


def pair_draws(count: int = 6, *, seed: int = 7, step: int = 0) -> SeenPairDraws:
    return SeenPairDraws(PAIRS, draw(count, seed=seed, step=step))


def graph() -> tuple[StepDeclaration, ...]:
    built = Pretraining()

    return (built.terms_step(), built.controller_step())


def witnessed_terms_step(
    witness: Witness[TermsBatch],
    policy: Descent | None = None,
    *,
    with_pairs: bool = True,
) -> TermsStep:
    assembly = LossAssembly[TermsBatch](
        (WeightedObjective(witness.name, 1.0, witness),)
    )

    return TermsStep(
        TYPE_SAMPLER,
        TYPES,
        PAIR_SAMPLER if with_pairs else None,
        PAIRS if with_pairs else None,
        CODEC,
        assembly,
        Descent(()) if policy is None else policy,
        roles=("terms", "identity"),
        name="witnessed",
    )


class TestTheTermsStepConforms(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[Draw, tuple[Trained, ...]]:
        return StepFixture(
            step=Pretraining().terms_step(),
            inputs=draw,
            reads=lambda _: frozenset({"draw"}),
            idle=lambda: draw(0),
            variants=(
                Pretraining(schedule=TeacherForcingSchedule(None)).terms_step(),
                Pretraining(
                    rate=0.5, schedule=TeacherForcingSchedule(0.2)
                ).terms_step(),
            ),
            graph=graph(),
            available=AVAILABLE,
        )


class TestTheControllerStepConforms(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[SeenPairDraws, TrainedController]:
        return StepFixture(
            step=Pretraining().controller_step(),
            inputs=pair_draws,
            reads=lambda inputs: inputs.seen,
            idle=lambda: pair_draws(0),
            variants=(
                Pretraining(rate=0.01).controller_step(),
                Pretraining(rate=1.0).controller_step(),
            ),
            graph=graph(),
            available=AVAILABLE,
        )


def test_a_draw_is_validated_and_frozen() -> None:
    request = Draw(3, 4, step=1, horizon=HORIZON)

    assert (request.seed, request.count, request.step, request.horizon) == (
        3,
        4,
        1,
        HORIZON,
    )
    assert request == Draw(3, 4, step=1, horizon=HORIZON)
    assert request != Draw(3, 4, step=2, horizon=HORIZON)
    assert hash(request) == hash(Draw(3, 4, step=1, horizon=HORIZON))
    assert repr(request) == "Draw(3, 4, step=1, horizon=10)"

    with pytest.raises(AttributeError, match="immutable"):
        request.__setattr__("count", 5)

    with pytest.raises(ValueError, match="count"):
        Draw(3, -1, step=0, horizon=HORIZON)

    with pytest.raises(ValueError, match="no steps"):
        Draw(3, 4, step=0, horizon=0)

    flag = True

    with pytest.raises(TypeError, match="seed"):
        Draw(flag, 4, step=0, horizon=HORIZON)


def test_the_draw_values_compare_by_distribution_and_draw() -> None:
    pairs = PairDraws(PAIRS, draw())

    assert pairs == PairDraws(PAIRS, draw())
    assert pairs != PairDraws(PairDistribution(max_depth=2), draw())
    assert hash(pairs) == hash(PairDraws(PAIRS, draw()))
    assert "Draw(7, 6" in repr(pairs)

    with pytest.raises(AttributeError, match="immutable"):
        pairs.__setattr__("draw", draw(1))


def test_the_outputs_name_the_role_and_compare_by_it() -> None:
    assert Trained("encoder") == Trained("encoder")
    assert Trained("encoder") != Trained("decoder")
    assert hash(Trained("encoder")) == hash(Trained("encoder"))
    assert repr(Trained("encoder")) == "Trained('encoder')"

    labeled = TrainedController("controller", {"apply": 1, "identity": 0})

    assert labeled == TrainedController("controller", {"identity": 0, "apply": 1})
    assert labeled != TrainedController("controller", {"identity": 1, "apply": 0})
    assert hash(labeled) == hash(TrainedController("controller", labeled.labels))
    assert "controller" in repr(labeled)

    with pytest.raises(AttributeError, match="immutable"):
        labeled.__setattr__("role", "other")

    with pytest.raises(ValueError, match="role"):
        Trained("")


def test_the_batch_holds_the_drawn_terms_and_the_labeled_rule_instances() -> None:
    witness = Witness[TermsBatch]()
    drawn, _ = TYPE_SAMPLER.sample(5, TYPES, seed=11)
    sampled, _ = PAIR_SAMPLER.sample(5, PAIRS, seed=11)

    witnessed_terms_step(witness).run(draw(5, seed=11, step=4))
    (batch,) = witness.batches

    assert len(batch) == 5
    assert batch.terms == CODEC.encode_batch(drawn)
    assert batch.instances == sampled
    assert (batch.step, batch.horizon) == (4, HORIZON)
    assert {action for pair in batch.instances for action in pair.actions} <= {
        APPLY,
        SWAP,
    }


def test_a_draw_is_a_function_of_its_seed() -> None:
    first = Witness[TermsBatch]()
    second = Witness[TermsBatch]()

    witnessed_terms_step(first).run(draw(4, seed=5))
    witnessed_terms_step(second).run(draw(4, seed=5))
    witnessed_terms_step(second).run(draw(4, seed=6))

    assert first.batches[0].instances == second.batches[0].instances
    assert first.batches[0].instances != second.batches[1].instances


def test_the_terms_step_lowers_the_reconstruction_loss_it_trains() -> None:
    step = Pretraining(rate=0.5).terms_step(with_pairs=False)
    losses = [
        step.run(draw(8, seed=1, step=at)).result.metrics["loss/type-autoencoder"]
        for at in range(6)
    ]

    assert losses[-1] < losses[0]


def test_a_request_for_nothing_takes_no_step_and_reports_no_item() -> None:
    built = Pretraining()
    results: list[TrainResult] = [
        built.terms_step().run(draw(0, step=4)).result,
        built.controller_step().run(pair_draws(0, step=4)).result,
    ]

    for reported, policy in zip(results, built.policies, strict=True):
        assert policy.calls == 0
        assert reported.items == 0
        assert reported.step == 4
        assert "optimizer/stepped" not in reported.metrics
        assert not any(key.startswith("loss/") for key in reported.metrics)

    assert results[0].metrics["terms/sample/draws"] == 0.0
    assert results[1].metrics["sample/emitted"] == 0.0


def test_a_policy_that_does_not_step_leaves_the_step_count_where_it_was() -> None:
    witness = Witness[TermsBatch]()
    step = witnessed_terms_step(witness, Accumulating(()))

    reported = step.run(draw(3, step=5)).result

    assert reported.step == 5
    assert reported.items == 3
    assert reported.metrics["optimizer/stepped"] == 0.0
    assert reported.metrics["optimizer/loss_finite"] == 1.0


def test_a_nonfinite_loss_is_reported_rather_than_raised_or_stepped() -> None:
    step = witnessed_terms_step(Diverging[TermsBatch]("diverging"))

    reported = step.run(draw(3, step=5)).result

    assert reported.step == 5
    assert reported.items == 3
    assert reported.metrics["optimizer/stepped"] == 0.0
    assert reported.metrics["optimizer/loss_finite"] == 0.0
    assert "loss/diverging" not in reported.metrics


def test_the_terms_step_runs_the_pass_the_schedule_names_at_the_step() -> None:
    built = Pretraining(schedule=TeacherForcingSchedule(0.5))
    step = built.terms_step()

    early = step.run(draw(4, step=0)).result
    late = step.run(draw(4, step=5)).result
    passes = [
        options.mode for scorer in built.scorers.values() for options in scorer.passes
    ]

    assert early.metrics["loss/combinator-supervision/drawn"] == 0.0
    assert late.metrics["loss/combinator-supervision/drawn"] == 1.0
    assert ForwardMode.OBSERVE in passes
    assert ForwardMode.SAMPLE in passes
    assert early.items == 4
    assert late.step == 6


def test_the_terms_step_evaluates_exact_match_without_fitting() -> None:
    built = Pretraining()
    evaluated = built.terms_step().evaluate(draw(5))

    assert evaluated.items == 5
    assert 0.0 <= evaluated.metrics["loss/type-autoencoder/exact_match"] <= 1.0
    assert 0.0 <= evaluated.metrics["loss/combinator-supervision/exact_match"] <= 1.0
    assert built.policies[-1].calls == 0


def test_the_terms_step_jointly_trains_the_space_encoder_and_decoders() -> None:
    outcome = Pretraining().terms_step().run(draw(5, seed=19, step=2))
    reported = outcome.result

    assert outcome.outputs == (
        Trained("terms"),
        Trained("identity"),
        Trained("application"),
        Trained("swapping"),
    )
    assert reported.items == 5
    assert reported.step == 3
    assert reported.metrics["terms/sample/draws"] == 5.0
    assert reported.metrics["pairs/sample/emitted"] == 5.0
    assert reported.metrics["loss/type-autoencoder"] > 0.0
    assert reported.metrics["loss/combinator-supervision"] > 0.0
    assert reported.metrics["optimizer/stepped"] == 1.0


def test_a_space_without_combinators_trains_its_encoder_and_identity_decoder() -> None:
    outcome = Pretraining().terms_step(with_pairs=False).run(draw(3, seed=23))

    assert outcome.outputs == (Trained("terms"), Trained("identity"))
    assert outcome.result.items == 3
    assert "loss/type-autoencoder" in outcome.result.metrics
    assert "loss/combinator-supervision" not in outcome.result.metrics
    assert not any(key.startswith("pairs/") for key in outcome.result.metrics)


def test_the_controller_step_carries_the_declared_label_map() -> None:
    step = Pretraining().controller_step()

    outcome = step.run(pair_draws(5))

    assert step.labels == {"identity": 0, "apply": 1, "swap": 2}
    assert outcome.outputs == TrainedController(
        "controller", {"identity": 0, "apply": 1, "swap": 2}
    )
    assert outcome.result.items == 5
    assert outcome.result.metrics["loss/controller-supervision"] > 0.0
    assert "loss/controller-supervision/accuracy" in outcome.result.metrics
    assert "labels={'identity': 0" in repr(step)


def test_the_controller_step_lowers_the_supervision_loss_it_trains() -> None:
    step = Pretraining(rate=0.5).controller_step()
    losses = [
        step.run(pair_draws(8, seed=2, step=at)).result.metrics[
            "loss/controller-supervision"
        ]
        for at in range(6)
    ]

    assert losses[-1] < losses[0]


def test_the_controller_step_evaluates_accuracy_without_fitting() -> None:
    built = Pretraining()
    evaluated = built.controller_step().evaluate(pair_draws(5))

    assert evaluated.items == 5
    assert 0.0 <= evaluated.metrics["loss/controller-supervision/accuracy"] <= 1.0
    assert built.policies[-1].calls == 0


def _manifest(labels: dict[str, int]) -> CheckpointManifest:
    return CheckpointManifest(
        run_id="run",
        step=1,
        data_position=0,
        code_version="0",
        action_set="actions-a1",
        components={
            "controller": ComponentRecord(
                tag="table",
                spec_fingerprint="spec",
                signature_fingerprint="sig",
                codec_identity="codec",
                state=StateReference(key="controller", digest="d"),
                labels=labels,
            )
        },
        rng=RngRecord(streams={}),
        precision=PrecisionRecord(
            compute_dtype="float64", autocast_device_type="cpu", gradient_scaling=False
        ),
    )


def test_a_reordered_action_set_is_reported_at_load_rather_than_loaded() -> None:
    built = Pretraining()
    saved = _manifest(dict(built.controller_step().run(pair_draws(2)).outputs.labels))
    same = ControllerStep(
        PAIR_SAMPLER,
        ACTIONS,
        built.controller_step().assembly,
        Descent(()),
        role="controller",
    )
    reordered = ControllerStep(
        PAIR_SAMPLER,
        REVERSED,
        built.controller_step().assembly,
        Descent(()),
        role="controller",
    )

    assert disagreements(_manifest(dict(same.labels)), saved) == ()

    (reason,) = disagreements(_manifest(dict(reordered.labels)), saved)

    assert "'controller'" in reason
    assert "built with labels {'swap': 0, 'apply': 1, 'identity': 2}" in reason
    assert "saved with {'identity': 0, 'apply': 1, 'swap': 2}" in reason


def test_the_controller_step_depends_on_no_decoder_training() -> None:
    steps = graph()
    controller = steps[-1]

    assert controller.requires == frozenset({"pair_distribution", "draw"})
    assert controller.produces == frozenset({"controller"})
    assert [
        edge for edge in edges(steps, AVAILABLE) if edge.consumer == controller.name
    ] == []
    assert edges(steps, AVAILABLE) == ()

    with pytest.raises(StepGraphError, match="nothing publishes"):
        edges(steps)


def test_two_steps_training_one_role_cannot_share_a_graph() -> None:
    built = Pretraining()
    steps = (built.terms_step(), built.terms_step())

    with pytest.raises(StepGraphError, match="named"):
        edges(steps, AVAILABLE)

    renamed = (
        built.terms_step(name="first"),
        built.terms_step(name="second"),
    )

    with pytest.raises(StepGraphError, match="both publish"):
        edges(renamed, AVAILABLE)


def test_a_step_is_named_after_its_role_unless_told_otherwise() -> None:
    built = Pretraining()

    assert built.controller_step().name == "pretrain:controller"
    assert built.controller_step(name="ctrl").name == "ctrl"
    assert built.controller_step(role="ctl").role == "ctl"
    assert "role='controller'" in repr(built.controller_step())
    assert "name='categories'" in repr(built.terms_step())

    with pytest.raises(ValueError, match="role"):
        built.controller_step(role="")

    with pytest.raises(ValueError, match="name"):
        built.terms_step(name="")

    with pytest.raises(ValueError, match="name"):
        built.controller_step(name="")


def models_built(call: Callable[[], None]) -> int:
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


def test_a_hundred_steps_of_each_kind_construct_no_boundary_model() -> None:
    built = Pretraining()
    steps = 100

    def run() -> None:
        terms = built.terms_step()
        controller = built.controller_step()

        for at in range(steps):
            terms.run(draw(4, seed=at, step=at))
            controller.run(pair_draws(4, seed=at, step=at))

    assert models_built(run) == 0
