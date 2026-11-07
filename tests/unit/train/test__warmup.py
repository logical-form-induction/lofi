from typing import TYPE_CHECKING, Final, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.train_step import TrainStepConformance

from lofi.lexicon import Entry, StaticLexicon
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.encoders import (
    HolographicEncoderSettings,
    HrrTermEncoder,
    TransformerEncoderSettings,
    TransformerTermEncoder,
    TreeRnnEncoder,
)
from lofi.nn.loss._ports import (
    AmbiguousTokens,
    LossTerm,
)
from lofi.nn.loss._warmup import TokenWarmupObjective
from lofi.nn.ports import StandardDeclaration
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.embedder import (
    ReservedTokens,
    StaticTokenEmbedder,
    TokenVocabulary,
)
from lofi.nn.tokens.pool import FirstSubwordPooler
from lofi.nn.tokens.transfer import AmbiguityPolicy
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    Term,
    TermCodec,
    TermStore,
)
from lofi.train import LossAssembly, OptimizationResult, WeightedObjective, edges
from lofi.train._pretraining import Trained
from lofi.train._warmup import TokenWarmupStep, Utterances, WarmupBatch, transfer_rows
from lofi.train.laws import StepFixture

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

    from torch import Tensor

    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.ports import (
        SubwordPooler,
        TeacherForcedDecoder,
        TermEncoder,
        TokenEncoder,
    )
    from lofi.terms import BatchEncoding


SIGNATURE: Final = Signature(
    "warmed",
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
VIEW: Final = NeuralCodecView(CODEC)
STORE: Final = TermStore(SIGNATURE)
CAPACITY: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
CHOICES: Final = ChoiceIndex(
    SIGNATURE, depth_bounds=dict.fromkeys(SIGNATURE.sorts, CODEC.capacity)
)
WIDTH: Final = CAPACITY * SYMBOLS
A: Final = STORE.op(SIGNATURE.by_name("A"))
B: Final = STORE.op(SIGNATURE.by_name("B"))
RESERVED: Final = ReservedTokens({"<pad>": 0, "<unk>": 1})
WORDS: Final = ("amb", "one", "plain", "bare")
VOCABULARY: Final = TokenVocabulary(WORDS, reserved=RESERVED, unknown="<unk>")
AVAILABLE: Final = frozenset({"candidates", "items", "step"})


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


AB: Final = f(A, B)
BA: Final = f(B, A)
LEXICON: Final = StaticLexicon[Term](
    (
        Entry("amb", AB),
        Entry("amb", BA),
        Entry("amb", BA, weight=-1.0),
        Entry("one", A),
        Entry("plain", B),
    )
)


def _rows(terms: BatchEncoding) -> tuple[Tensor, Tensor]:
    symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
    occupied = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)

    return symbols, occupied


class IndicatorEncoder:
    """Encodes a type as its slot-by-symbol indicators, each scaled by a weight."""

    def __init__(self) -> None:
        self.weight = torch.ones(
            CAPACITY, SYMBOLS, dtype=torch.float64
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
    """Weighs a type by reading each slot's logits off a vector."""

    def __init__(self) -> None:
        self.gain = torch.tensor(1.0, dtype=torch.float64).requires_grad_()

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        del terms
        logits = encoded.tensor.reshape(-1, CAPACITY, SYMBOLS) * self.gain

        if encoded.mask is None:
            return LogDistribution(
                logits.log_softmax(dim=-1),
                ("term", "slot", "choice"),
                normalized_over="choice",
            )

        return LogDistribution(
            logits.log_softmax(dim=-1).masked_fill(
                ~encoded.mask.reshape(-1, 1, 1), -torch.inf
            ),
            ("term", "slot", "choice"),
            encoded.mask,
            normalized_over="choice",
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


class Witness:
    """Records every batch it is handed and contributes a zero term over it."""

    def __init__(self) -> None:
        self.batches: list[WarmupBatch] = []

    @property
    def name(self) -> str:
        return "witness"

    @property
    def requires(self) -> frozenset[str]:
        return frozenset({"batch"})

    def evaluate(self, outputs: PublishedOutputs[WarmupBatch], /) -> LossTerm:
        batch = outputs.batch
        self.batches.append(batch)

        return LossTerm(
            torch.zeros((), dtype=torch.float64, requires_grad=True), len(batch)
        )


def indicators(term: Term) -> Tensor:
    return IndicatorEncoder().encode(CODEC.encode_batch((term,))).tensor[0].detach()


def table() -> Tensor:
    rows = torch.zeros(VOCABULARY.size, WIDTH, dtype=torch.float64)
    rows[VOCABULARY.index_of("amb")] = 0.1 * indicators(AB) + 0.3 * indicators(BA)
    rows[VOCABULARY.index_of("one")] = torch.linspace(-1.0, 1.0, WIDTH).to(rows.dtype)
    rows[VOCABULARY.index_of("plain")] = -indicators(A)
    rows[VOCABULARY.index_of("bare")] = 0.5 * indicators(B)

    return rows


def embedder(rows: Tensor | None = None) -> StaticTokenEmbedder:
    return StaticTokenEmbedder(
        table() if rows is None else rows,
        vocabulary=VOCABULARY,
        component=ComponentId("embedder"),
        trainable=True,
        window=None,
        load_report=None,
    )


class Warmup:
    """One warm-up step over the indicator encoder, with the parameters it moves."""

    def __init__(self, *, rate: float = 0.1, rows: Tensor | None = None) -> None:
        self.tokens = embedder(rows)
        self.encoder = IndicatorEncoder()
        self.decoder = ReadingDecoder()
        self.rate = rate
        self.policies: list[Descent] = []

    def parameters(self) -> tuple[Tensor, ...]:
        return (*self.tokens.parameters(), self.encoder.weight, self.decoder.gain)

    def policy(self) -> Descent:
        built = Descent(self.parameters(), self.rate)
        self.policies.append(built)

        return built

    def objective(self) -> TokenWarmupObjective[WarmupBatch]:
        return TokenWarmupObjective(
            cast("TokenEncoder", self.tokens),
            cast(
                "SubwordPooler", FirstSubwordPooler(WIDTH, component=ComponentId("p"))
            ),
            cast("TermEncoder", self.encoder),
            cast("TeacherForcedDecoder", self.decoder),
            CHOICES,
        )

    def step(
        self,
        *,
        role: str = "token_encoder",
        name: str | None = None,
    ) -> TokenWarmupStep:
        objective = self.objective()
        assembly = LossAssembly[WarmupBatch](
            (WeightedObjective(objective.name, 1.0, objective),)
        )

        return TokenWarmupStep(CODEC, assembly, self.policy(), role=role, name=name)


class SeenUtterances(Utterances):
    __slots__ = ("_seen",)

    def __init__(
        self,
        items: Sequence[Sequence[str]],
        candidates: Sequence[Sequence[Sequence[Term]]],
        step: int,
    ) -> None:
        super().__init__(items, candidates, step)
        object.__setattr__(self, "_seen", set())

    @property
    @override
    def items(self) -> tuple[tuple[str, ...], ...]:
        self._seen.add("items")

        return super().items

    @property
    @override
    def step(self) -> int:
        self._seen.add("step")

        return super().step

    @property
    @override
    def candidates(self) -> tuple[tuple[tuple[Term, ...], ...], ...]:
        self._seen.add("candidates")

        return super().candidates

    @property
    def seen(self) -> frozenset[str]:
        return frozenset(self._seen)


ITEMS: Final = (("amb", "one"), ("plain",), ("one", "amb", "plain"))


def admitted(
    items: Sequence[Sequence[str]],
) -> tuple[tuple[tuple[Term, ...], ...], ...]:
    return tuple(
        tuple(
            tuple(
                dict.fromkeys(
                    entry.category
                    for offered, entry, _ in LEXICON.axioms(words)
                    if offered == position
                )
            )
            for position in range(len(words))
        )
        for words in items
    )


def utterances(
    items: Sequence[Sequence[str]] = ITEMS, *, step: int = 0
) -> SeenUtterances:
    return SeenUtterances(items, admitted(items), step)


def witnessed(witness: Witness) -> TokenWarmupStep:
    assembly = LossAssembly[WarmupBatch]((WeightedObjective("witness", 1.0, witness),))

    return TokenWarmupStep(CODEC, assembly, Descent(()), role="token_encoder")


class TestTheTokenWarmupStepConforms(TrainStepConformance):
    @pytest.fixture
    def conformance(self) -> StepFixture[SeenUtterances, Trained]:
        return StepFixture(
            step=Warmup().step(),
            inputs=utterances,
            reads=lambda inputs: inputs.seen,
            idle=lambda: utterances(()),
            variants=(Warmup(rate=0.01).step(), Warmup(rate=0.5).step()),
            graph=(Warmup().step(),),
            available=AVAILABLE,
        )


def test_a_request_is_frozen_and_compares_by_items_and_position() -> None:
    candidates = (((A,), (B,)), ((AB,),))
    request = Utterances([["a", "b"], ["c"]], candidates, 2)

    assert request.items == (("a", "b"), ("c",))
    assert request.candidates == candidates
    assert request.step == 2
    assert request == Utterances((("a", "b"), ("c",)), candidates, 2)
    assert request != Utterances((("a", "b"), ("c",)), candidates, 3)
    assert hash(request) == hash(Utterances((("a", "b"), ("c",)), candidates, 2))
    assert repr(request) == (
        f"Utterances((('a', 'b'), ('c',)), candidates={candidates!r}, step=2)"
    )

    with pytest.raises(AttributeError, match="immutable"):
        request.__setattr__("step", 0)

    with pytest.raises(ValueError, match="step"):
        Utterances((), (), -1)

    with pytest.raises(ValueError, match="align"):
        Utterances((("a",),), (), 0)


def test_the_batch_holds_every_admitted_type_once_at_its_position() -> None:
    witness = Witness()

    witnessed(witness).run(utterances())
    (batch,) = witness.batches

    assert isinstance(batch, AmbiguousTokens)
    assert batch.words == ITEMS
    assert len(batch) == 3
    assert batch.positions == (
        (0, 0),
        (0, 0),
        (0, 1),
        (1, 0),
        (2, 0),
        (2, 1),
        (2, 1),
        (2, 2),
    )
    assert batch.types == CODEC.encode_batch((AB, BA, A, B, A, AB, BA, B))
    assert repr(batch) == "WarmupBatch(items=3, candidates=8)"

    with pytest.raises(AttributeError, match="immutable"):
        batch.__setattr__("words", ())


def test_the_target_of_an_ambiguous_token_is_its_nearest_admitted_type() -> None:
    built = Warmup(rate=0.5)
    step = built.step()
    row = built.tokens.encode_words((("amb",),)).vectors.tensor[0, 0].detach().clone()
    candidates = {AB: indicators(AB), BA: indicators(BA)}
    distances = {
        term: (row - vector).pow(2).mean().item() for term, vector in candidates.items()
    }
    nearest = min(distances, key=distances.__getitem__)
    other = next(term for term in candidates if term is not nearest)

    reported = step.run(utterances((("amb",),))).result
    moved = built.tokens.encode_words((("amb",),)).vectors.tensor[0, 0].detach()

    assert nearest is BA
    assert reported.metrics["loss/token-warmup/distance"] == pytest.approx(
        distances[nearest]
    )
    assert (moved - candidates[nearest]).pow(2).mean().item() < distances[nearest]
    assert (moved - candidates[other]).pow(2).mean().item() > distances[nearest]


def test_the_mean_distance_is_the_mean_of_each_token_least_distance() -> None:
    built = Warmup()
    step = built.step()
    rows = built.tokens.encode_words(ITEMS).vectors.tensor.detach()
    admitted = {
        "amb": (indicators(AB), indicators(BA)),
        "one": (indicators(A),),
        "plain": (indicators(B),),
    }
    least = [
        min(
            (rows[item, word] - vector).pow(2).mean().item()
            for vector in admitted[token]
        )
        for item, words in enumerate(ITEMS)
        for word, token in enumerate(words)
    ]

    reported = step.run(utterances()).result

    assert reported.items == 3
    assert reported.metrics["warmup/candidates"] == 8.0
    assert reported.metrics["loss/token-warmup/distance"] == pytest.approx(
        sum(least) / len(least)
    )
    assert (
        reported.metrics["loss/token-warmup"]
        > reported.metrics["loss/token-warmup/distance"]
    )


def test_the_warm_up_lowers_the_loss_it_trains() -> None:
    step = Warmup(rate=0.5).step()
    losses = [
        step.run(utterances(step=at)).result.metrics["loss/token-warmup"]
        for at in range(6)
    ]

    assert losses[-1] < losses[0]


def test_an_item_the_source_admits_nothing_for_is_dropped_not_raised() -> None:
    built = Warmup()
    step = built.step()

    reported = step.run(
        utterances((("amb",), ("bare", "unseen"), ("one",)), step=3)
    ).result

    assert reported.items == 2
    assert reported.dropped == 1
    assert reported.step == 4
    assert reported.metrics["warmup/candidates"] == 3.0
    assert reported.metrics["optimizer/stepped"] == 1.0

    silent = step.run(utterances((("bare",), ("unseen",)), step=4)).result

    assert silent.items == 0
    assert silent.dropped == 2
    assert silent.step == 4
    assert silent.metrics == {"warmup/candidates": 0.0}
    assert built.policies[0].calls == 1


def test_a_request_for_nothing_takes_no_step_and_reports_no_item() -> None:
    built = Warmup()
    outcome = built.step().run(utterances((), step=5))

    assert outcome.outputs == Trained("token_encoder")
    assert outcome.result.items == 0
    assert outcome.result.dropped == 0
    assert outcome.result.step == 5
    assert outcome.result.metrics == {"warmup/candidates": 0.0}
    assert built.policies[0].calls == 0


def test_two_steps_read_the_same_immutable_occurrence_candidates() -> None:
    before = LEXICON.axioms(("amb", "one", "plain"))
    built = Warmup()
    first = built.step()
    second = built.step()

    first.run(utterances())
    second.run(utterances(step=1))

    assert LEXICON.axioms(("amb", "one", "plain")) == before
    assert len(LEXICON) == 4
    assert "source=" not in repr(first)
    assert "source=" not in repr(second)
    assert first.produces == second.produces == frozenset({"token_encoder"})
    assert first.requires == frozenset({"candidates", "items", "step"})


def test_the_edges_of_a_warm_up_step_are_its_items_and_position() -> None:
    step = Warmup().step()

    assert edges((step,), AVAILABLE) == ()
    assert step.name == "warmup:token_encoder"
    assert Warmup().step(name="seed").name == "seed"
    assert Warmup().step(role="tok").role == "tok"

    with pytest.raises(ValueError, match="role"):
        Warmup().step(role="")

    with pytest.raises(ValueError, match="name"):
        Warmup().step(name="")


def test_transfer_rows_seeds_each_token_row_under_the_policy() -> None:
    encoder = IndicatorEncoder()
    rows = table()
    kept = rows.clone()
    amb = VOCABULARY.index_of("amb")
    bare = VOCABULARY.index_of("bare")
    reports = {}
    written = {}

    for policy in AmbiguityPolicy:
        written[policy], reports[policy] = transfer_rows(
            rows,
            LEXICON,
            encoder=cast("TermEncoder", encoder),
            codec=CODEC,
            vocabulary=VOCABULARY,
            policy=policy,
        )

    assert torch.equal(rows, kept)
    assert torch.equal(
        written[AmbiguityPolicy.MEAN][amb], 0.5 * (indicators(AB) + indicators(BA))
    )
    assert torch.equal(written[AmbiguityPolicy.FIRST][amb], indicators(AB))
    assert torch.equal(written[AmbiguityPolicy.SKIP][amb], rows[amb])

    for policy in AmbiguityPolicy:
        assert torch.equal(written[policy][VOCABULARY.index_of("one")], indicators(A))
        assert torch.equal(written[policy][VOCABULARY.index_of("plain")], indicators(B))
        assert torch.equal(written[policy][bare], rows[bare])
        assert reports[policy].rows_considered == 3
        assert reports[policy].rows_unmatched == 0
        assert not written[policy].requires_grad

    assert reports[AmbiguityPolicy.MEAN].rows_initialized == 3
    assert reports[AmbiguityPolicy.SKIP].rows_initialized == 2
    assert reports[AmbiguityPolicy.SKIP].rows_skipped == 1


def test_transfer_over_a_source_admitting_nothing_writes_nothing() -> None:
    rows = table()

    written, report = transfer_rows(
        rows,
        StaticLexicon[Term](()),
        encoder=cast("TermEncoder", IndicatorEncoder()),
        codec=CODEC,
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.MEAN,
    )

    assert torch.equal(written, rows)
    assert written is not rows
    assert report.rows_considered == 0
    assert report.rows_initialized == 0


class WithholdingEncoder:
    """Encodes as the indicator encoder does, holding no vector for some types."""

    def __init__(self, withheld: Sequence[Term]) -> None:
        self.inner = IndicatorEncoder()
        self.withheld = frozenset(CODEC.encode_batch(withheld).rows)

    @property
    def output_width(self) -> int:
        return self.inner.output_width

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        encoded = self.inner.encode(terms)
        held = torch.tensor([row not in self.withheld for row in terms.rows])

        return Encoded(encoded.tensor, encoded.axes, held)


def test_transfer_offers_no_row_the_encoder_holds_no_vector_for() -> None:
    rows = table()
    amb = VOCABULARY.index_of("amb")
    plain = VOCABULARY.index_of("plain")
    written = {}
    reports = {}

    for policy in (AmbiguityPolicy.MEAN, AmbiguityPolicy.SKIP):
        written[policy], reports[policy] = transfer_rows(
            rows,
            LEXICON,
            encoder=cast("TermEncoder", WithholdingEncoder((BA, B))),
            codec=CODEC,
            vocabulary=VOCABULARY,
            policy=policy,
        )

    for policy, seeded in written.items():
        assert torch.equal(seeded[amb], indicators(AB))
        assert torch.equal(seeded[plain], rows[plain])
        assert torch.equal(seeded[VOCABULARY.index_of("one")], indicators(A))
        assert reports[policy].rows_considered == 2
        assert reports[policy].rows_initialized == 2
        assert reports[policy].rows_skipped == 0


def rnn() -> TreeRnnEncoder:
    return TreeRnnEncoder(
        codec=VIEW,
        choices=ChoiceIndex(
            SIGNATURE,
            depth_bounds=dict.fromkeys(SIGNATURE.sorts, CODEC.capacity),
        ),
        component=ComponentId("rnn"),
        scheme=ArityHeadScheme(width=8, reached=SIGNATURE.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    )


def hrr() -> HrrTermEncoder:
    return HrrTermEncoder(
        codec=VIEW,
        component=ComponentId("hrr"),
        settings=HolographicEncoderSettings(width=12, magnitude=1.0, floor=1e-3),
    )


def transformer() -> TransformerTermEncoder:
    return TransformerTermEncoder(
        codec=VIEW,
        component=ComponentId("transformer"),
        settings=TransformerEncoderSettings(
            6, 2, 1, 16, norm_cap=None, activation="gelu"
        ),
        structure=StandardDeclaration.POSITIONS.declaration,
    )


SHIPPED: Final = (rnn, hrr, transformer)


@pytest.mark.parametrize("build", SHIPPED, ids=[build.__name__ for build in SHIPPED])
def test_the_warm_up_seeds_and_trains_through_every_shipped_term_encoder(
    build: Callable[[], TermEncoder],
) -> None:
    torch.manual_seed(5)
    encoder = cast("torch.nn.Module", build()).to(torch.float64)
    width = cast("TermEncoder", encoder).output_width
    decoder = RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("readout"),
        settings=TreeDecoderSettings(width, 6),
        choice_heads="flat",
    ).to(torch.float64)
    rows, report = transfer_rows(
        torch.zeros(VOCABULARY.size, width, dtype=torch.float64),
        LEXICON,
        encoder=cast("TermEncoder", encoder),
        codec=CODEC,
        vocabulary=VOCABULARY,
        policy=AmbiguityPolicy.FIRST,
    )
    encoded = cast("TermEncoder", encoder).encode(CODEC.encode_batch((AB, A, B)))

    assert report.rows_initialized == 3
    assert rows.shape == (VOCABULARY.size, width)
    assert torch.allclose(rows[VOCABULARY.index_of("amb")], encoded.tensor[0])
    assert torch.allclose(rows[VOCABULARY.index_of("one")], encoded.tensor[1])
    assert torch.allclose(rows[VOCABULARY.index_of("plain")], encoded.tensor[2])

    tokens = StaticTokenEmbedder(
        rows.clone(),
        vocabulary=VOCABULARY,
        component=ComponentId("embedder"),
        trainable=True,
        window=None,
        load_report=None,
    )
    objective = TokenWarmupObjective[WarmupBatch](
        tokens,
        FirstSubwordPooler(width, component=ComponentId("pooler")),
        cast("TermEncoder", encoder),
        decoder,
        CHOICES,
    )
    assembly = LossAssembly[WarmupBatch](
        (WeightedObjective(objective.name, 1.0, objective),)
    )
    parameters = (
        *tokens.parameters(),
        *encoder.parameters(),
        *decoder.parameters(),
    )
    step = TokenWarmupStep(
        CODEC, assembly, Descent(parameters, 0.05), role="token_encoder"
    )

    reported = step.run(utterances()).result

    assert reported.items == 3
    assert reported.metrics["optimizer/stepped"] == 1.0
    assert reported.metrics["optimizer/loss_finite"] == 1.0
    assert reported.metrics["loss/token-warmup/distance"] == pytest.approx(0.0)
    assert reported.metrics["loss/token-warmup/likelihood"] > 0.0


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


def test_a_hundred_warm_up_steps_construct_no_boundary_model() -> None:
    step = Warmup().step()

    def run() -> None:
        for at in range(100):
            step.run(utterances(step=at))

    assert models_built(run) == 0
