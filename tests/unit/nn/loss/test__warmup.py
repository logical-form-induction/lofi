import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite

from lofi.deduce import SpanIndex
from lofi.nn._values import Encoded, LogDistribution
from lofi.nn.arity import ArityHeadScheme
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder, TreeDecoderSettings
from lofi.nn.encoders import TreeRnnEncoder
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
)
from lofi.nn.loss._warmup import (
    TokenWarmupObjective,
    nearest_candidates,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.embedder import (
    ReservedTokens,
    StaticTokenEmbedder,
    TokenVocabulary,
)
from lofi.nn.tokens.pool import FirstSubwordPooler
from lofi.nn.tokens.window import REPORT, ContextWindow
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.terms import (
    BatchEncoding,
    ChoiceIndex,
    CompleteTree,
    Encoding,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
    TermStore,
)

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.nn.loss._ports import LossTerm
    from lofi.nn.ports import (
        SubwordPooler,
        TeacherForcedDecoder,
        TermEncoder,
        TokenEncoder,
    )
    from lofi.terms import Term


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
STORE: Final = TermStore(SIGNATURE)
CAPACITY: Final = CODEC.capacity
SYMBOLS: Final = CODEC.n_symbols
CHOICES: Final = ChoiceIndex(
    SIGNATURE, depth_bounds=dict.fromkeys(SIGNATURE.sorts, CODEC.capacity)
)
WIDTH: Final = CAPACITY * SYMBOLS
A: Final = STORE.op(SIGNATURE.by_name("A"))
B: Final = STORE.op(SIGNATURE.by_name("B"))
FAVORED: Final = 3.0
RESERVED: Final = ReservedTokens({"<pad>": 0, "<unk>": 1})
WORDS: Final = ("amb", "one", "near", "far", "twin")
VOCABULARY: Final = TokenVocabulary(WORDS, reserved=RESERVED, unknown="<unk>")


def f(left: Term, right: Term) -> Term:
    return STORE.op(SIGNATURE.by_name("f"), left, right)


AB: Final = f(A, B)
BA: Final = f(B, A)


class Tokens(NamedTuple):
    words: tuple[tuple[str, ...], ...]
    types: BatchEncoding
    positions: tuple[tuple[int, int], ...]


class IndicatorEncoder:
    """Encodes a type as its slot-by-symbol indicators, each scaled by a weight."""

    def __init__(self) -> None:
        weight = torch.ones(CAPACITY, SYMBOLS, dtype=torch.float64)
        weight[:, INDEX.index(SIGNATURE.by_name("A"))] = FAVORED
        self.weight = weight.requires_grad_()

    @property
    def output_width(self) -> int:
        return WIDTH

    def encode(self, terms: BatchEncoding, /) -> Encoded:
        symbols = torch.tensor(terms.symbols, dtype=torch.long).reshape(-1, CAPACITY)
        indicators = torch.nn.functional.one_hot(symbols, SYMBOLS).to(torch.float64)
        vectors = (indicators * self.weight).reshape(symbols.shape[0], -1)
        live = torch.tensor(terms.mask, dtype=torch.bool).reshape(-1, CAPACITY)
        held = live.any(dim=1)

        return Encoded(vectors, ("term", "width"), None if bool(held.all()) else held)


class ReadingDecoder:
    """Weighs a type by reading each slot's logits off a vector, at a gain."""

    def __init__(self, gain: float = 1.0) -> None:
        self.gain = torch.tensor(gain, dtype=torch.float64).requires_grad_()

    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        rows = encoded.tensor.shape[0]
        logits = encoded.tensor.reshape(rows, CAPACITY, SYMBOLS) * self.gain
        del terms
        mask = encoded.mask

        if mask is not None:
            logits = logits.masked_fill(~mask.reshape(-1, 1, 1), -torch.inf)

        return LogDistribution(
            logits.log_softmax(dim=-1),
            ("term", "slot", "choice"),
            mask,
            normalized_over="choice",
        )


class ForgetfulDecoder(ReadingDecoder):
    """Reads like its parent and then drops the last type it weighed."""

    @override
    def rows(self, encoded: Encoded, /, *, terms: BatchEncoding) -> LogDistribution:
        scored = super().rows(encoded, terms=terms)
        kept = torch.ones(scored.tensor.shape[0], dtype=torch.bool)
        kept[-1] = False

        return LogDistribution(
            scored.tensor.masked_fill(~kept.reshape(-1, 1, 1), -torch.inf),
            scored.axes,
            kept,
            normalized_over="choice",
        )


def indicators(term: Term) -> Tensor:
    """Return the type's encoding under a fresh indicator encoder, detached.

    Parameters
    ----------
    term : Term
    """
    return IndicatorEncoder().encode(CODEC.encode_batch((term,))).tensor[0].detach()


def twin_row() -> Tensor:
    """Return a vector as far from AB as from BA that the readout weighs apart.

    Start at the encodings' midpoint, then change one differing slot so Euclidean
    distance remains tied while the readout favors one type.
    """
    row = 0.5 * (indicators(AB) + indicators(BA))
    slot = next(
        at
        for at, (left, right) in enumerate(
            zip(CODEC.encode(AB).symbols, CODEC.encode(BA).symbols, strict=True)
        )
        if left != right
    )
    row[slot * SYMBOLS + INDEX.index(SIGNATURE.by_name("A"))] += 1.0
    row[slot * SYMBOLS + INDEX.index(SIGNATURE.by_name("B"))] += FAVORED

    return row


def table(amb: Tensor | None = None) -> Tensor:
    """Return the embedding table, with the ambiguous token's row as given.

    Parameters
    ----------
    amb : Tensor | None
    """
    rows = torch.zeros(VOCABULARY.size, WIDTH, dtype=torch.float64)
    rows[VOCABULARY.index_of("amb")] = (
        0.1 * indicators(AB) + 0.3 * indicators(BA) if amb is None else amb
    )
    rows[VOCABULARY.index_of("one")] = torch.linspace(-1.0, 1.0, WIDTH).to(rows.dtype)
    rows[VOCABULARY.index_of("near")] = indicators(A)
    rows[VOCABULARY.index_of("far")] = -2.0 * indicators(A)
    rows[VOCABULARY.index_of("twin")] = twin_row()

    return rows


def embedder(
    amb: Tensor | None = None, *, window: ContextWindow | None = None
) -> StaticTokenEmbedder:
    return StaticTokenEmbedder(
        table(amb),
        vocabulary=VOCABULARY,
        component=ComponentId("embedder"),
        trainable=True,
        window=window,
        load_report=None,
    )


def pooler() -> FirstSubwordPooler:
    return FirstSubwordPooler(WIDTH, component=ComponentId("pooler"))


EMBEDDER: Final = embedder()
POOLER: Final = pooler()
ENCODER: Final = IndicatorEncoder()
DECODER: Final = ReadingDecoder()


def objective(
    tokens: StaticTokenEmbedder = EMBEDDER,
    encoder: IndicatorEncoder = ENCODER,
    decoder: ReadingDecoder = DECODER,
) -> TokenWarmupObjective[Tokens]:
    return TokenWarmupObjective(
        cast("TokenEncoder", tokens),
        cast("SubwordPooler", POOLER),
        cast("TermEncoder", encoder),
        cast("TeacherForcedDecoder", decoder),
        CHOICES,
    )


def dead_row() -> Encoding:
    """Return a row holding no type, whose slots read as some symbol anyway."""
    filled = CODEC.encode(f(f(A, B), f(B, A)))

    return Encoding(
        tuple(2 for _ in filled.symbols),
        filled.parents,
        filled.depths,
        tuple(False for _ in filled.occupied),
    )


def bundle_of(
    words: tuple[tuple[str, ...], ...],
    rows: tuple[Encoding, ...],
    positions: tuple[tuple[int, int], ...],
) -> ObjectiveOutputs[Tokens]:
    return ObjectiveOutputs(
        spans=SpanIndex(1),
        chart=CellChart(
            LogDistribution(torch.zeros(len(words), 1, dtype=torch.float64), SPAN_AXES)
        ),
        batch=Tokens(words, BatchEncoding(rows), positions),
    )


def bundle_over(
    words: tuple[tuple[str, ...], ...],
    candidates: tuple[tuple[tuple[int, int], Term], ...],
) -> ObjectiveOutputs[Tokens]:
    terms = tuple(term for _, term in candidates)
    positions = tuple(position for position, _ in candidates)

    return bundle_of(words, CODEC.encode_batch(terms).rows, positions)


def bundle() -> ObjectiveOutputs[Tokens]:
    """Return two items: an ambiguous token beside a plain one, and a lone one."""
    return bundle_over(
        (("amb", "one"), ("near",)),
        (((0, 0), AB), ((0, 1), A), ((0, 0), BA), ((1, 0), A)),
    )


def permuted(outputs: ObjectiveOutputs[Tokens]) -> ObjectiveOutputs[Tokens]:
    batch = outputs.batch
    rows = batch.types.rows
    order = (3, 0, 2, 1)
    swapped = {0: 1, 1: 0}

    return bundle_of(
        (batch.words[1], batch.words[0]),
        tuple(rows[index] for index in order),
        tuple(
            (swapped[item], word) for item, word in (batch.positions[i] for i in order)
        ),
    )


def refilled(outputs: ObjectiveOutputs[Tokens]) -> ObjectiveOutputs[Tokens]:
    """Return the bundle with a dead candidate added to the ambiguous token.

    Parameters
    ----------
    outputs : ObjectiveOutputs[Tokens]
    """
    batch = outputs.batch

    return bundle_of(
        batch.words, (*batch.types.rows, dead_row()), (*batch.positions, (0, 0))
    )


def empty_bundle() -> ObjectiveOutputs[Tokens]:
    return bundle_of((("amb",),), (dead_row(), dead_row()), ((0, 0), (0, 0)))


def inputs_of(outputs: ObjectiveOutputs[Tokens]) -> tuple[Tensor, ...]:
    del outputs
    (weight,) = tuple(EMBEDDER.parameters())

    return (weight, ENCODER.weight, DECODER.gain)


def slot_nll(vector: Tensor, slot: int, symbol: int, *, gain: float = 1.0) -> float:
    """Return the negative log probability of one symbol at one slot, by hand.

    Parameters
    ----------
    vector : Tensor
    slot : int
    symbol : int
    gain : float
    """
    logits = [gain * vector[slot * SYMBOLS + other].item() for other in range(SYMBOLS)]

    return math.log(sum(math.exp(logit) for logit in logits)) - logits[symbol]


def by_hand(outputs: ObjectiveOutputs[Tokens], rows: Tensor | None = None) -> float:
    """Return the value by looping over the tokens and their candidates.

    Parameters
    ----------
    outputs : ObjectiveOutputs[Tokens]
    rows : Tensor | None
    """
    batch = outputs.batch
    vectors = table() if rows is None else rows
    encoded = ENCODER.encode(batch.types).tensor.detach()
    nearest: dict[tuple[int, int], tuple[float, int]] = {}

    for row, (position, encoding) in enumerate(
        zip(batch.positions, batch.types.rows, strict=True)
    ):
        if not any(encoding.occupied):
            continue

        item, word = position
        vector = vectors[VOCABULARY.index_of(batch.words[item][word])]
        distance = ((vector - encoded[row]) ** 2).mean().item()

        if position not in nearest or distance < nearest[position][0]:
            nearest[position] = (distance, row)

    if not nearest:
        return 0.0

    distances = 0.0
    likelihoods = 0.0

    for (item, word), (distance, row) in nearest.items():
        vector = vectors[VOCABULARY.index_of(batch.words[item][word])]
        distances += distance
        likelihoods += sum(
            slot_nll(vector, slot, symbol)
            for slot, (symbol, occupied) in enumerate(
                zip(
                    batch.types.rows[row].symbols,
                    batch.types.rows[row].occupied,
                    strict=True,
                )
            )
            if occupied
        )

    return (distances + likelihoods) / len(nearest)


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


def fixture() -> ObjectiveFixture[Tokens]:
    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (
            bundle_over((("far",),), (((0, 0), A),)),
            bundle_over((("near",),), (((0, 0), A),)),
        ),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        inapplicable=empty_bundle,
        tolerance=1e-9,
    )


class TestTokenWarmupObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Tokens]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that a token far from its type scores above one sitting on it.

        Parameters
        ----------
        conformance : ObjectiveFixture[BatchT]
        """
        the_value_is_monotone_where_the_fixture_says(conformance)


@pytest.mark.parametrize("seed", range(12))
def test_the_nearest_candidate_of_each_token_is_what_a_loop_finds(seed: int) -> None:
    generator = torch.Generator().manual_seed(seed)
    tokens = 5
    owners = torch.randint(0, tokens, (17,), generator=generator)
    distances = torch.rand(17, dtype=torch.float64, generator=generator).mul(4).floor()
    distances[torch.rand(17, generator=generator) < 0.25] = torch.inf
    expected: dict[int, tuple[float, int]] = {}

    for row, (owner, distance) in enumerate(
        zip(owners.tolist(), distances.tolist(), strict=True)
    ):
        if math.isinf(distance):
            continue

        if owner not in expected or distance < expected[owner][0]:
            expected[owner] = (distance, row)

    nearest = nearest_candidates(distances, owners=owners, tokens=tokens)

    assert nearest.tokens.tolist() == sorted(expected)
    assert nearest.rows.tolist() == [expected[token][1] for token in sorted(expected)]


def test_the_value_is_the_mean_least_distance_plus_the_mean_nll() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.count == 3
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.value.item() == pytest.approx(
        term.diagnostics["distance"].item() + term.diagnostics["likelihood"].item(),
        rel=1e-12,
    )
    assert term.diagnostics["distance"].grad_fn is None
    assert term.diagnostics["likelihood"].grad_fn is None


def test_the_selected_candidate_is_the_one_a_loop_over_the_candidates_finds() -> None:
    outputs = bundle()
    rows = table()
    amb = rows[VOCABULARY.index_of("amb")]
    one = rows[VOCABULARY.index_of("one")]
    near = rows[VOCABULARY.index_of("near")]
    least = {
        "amb": min(((amb - indicators(t)) ** 2).mean().item() for t in (AB, BA)),
        "one": ((one - indicators(A)) ** 2).mean().item(),
        "near": ((near - indicators(A)) ** 2).mean().item(),
    }

    term = objective().evaluate(outputs)

    assert ((amb - indicators(BA)) ** 2).mean().item() == least["amb"]
    assert term.diagnostics["distance"].item() == pytest.approx(
        sum(least.values()) / len(least), rel=1e-12
    )


@pytest.mark.parametrize("gold", [AB, BA])
@pytest.mark.parametrize("fraction", [0.5, 1.0])
def test_moving_the_embedding_toward_any_gold_type_lowers_the_value(
    gold: Term, fraction: float
) -> None:
    outputs = bundle_over((("amb",),), (((0, 0), AB), ((0, 0), BA)))
    start = table()[VOCABULARY.index_of("amb")]
    moved = start + fraction * (indicators(gold) - start)

    before = objective(embedder()).evaluate(outputs).value.item()
    after = objective(embedder(moved)).evaluate(outputs).value.item()

    assert after < before
    assert after == pytest.approx(by_hand(outputs, table(moved)), rel=1e-12)


def test_the_selection_follows_the_embedding_rather_than_a_fixed_target() -> None:
    outputs = bundle_over((("amb",),), (((0, 0), AB), ((0, 0), BA)))

    at_ab = objective(embedder(indicators(AB))).evaluate(outputs)
    at_ba = objective(embedder(indicators(BA))).evaluate(outputs)

    assert at_ab.diagnostics["distance"].item() == 0.0
    assert at_ba.diagnostics["distance"].item() == 0.0


def nll(vector: Tensor, term: Term) -> float:
    """Return the negative log probability of a type against a vector, by hand.

    Parameters
    ----------
    vector : Tensor
    term : Term
    """
    encoded = CODEC.encode(term)
    return sum(
        slot_nll(vector, slot, symbol)
        for slot, (symbol, occupied) in enumerate(
            zip(encoded.symbols, encoded.occupied, strict=True)
        )
        if occupied
    )


def test_among_candidates_at_one_distance_the_earliest_row_is_selected() -> None:
    first = bundle_over((("twin",),), (((0, 0), AB), ((0, 0), BA)))
    second = bundle_over((("twin",),), (((0, 0), BA), ((0, 0), AB)))
    twin = table()[VOCABULARY.index_of("twin")]
    to_ab = ((twin - indicators(AB)) ** 2).mean().item()
    to_ba = ((twin - indicators(BA)) ** 2).mean().item()

    from_first = objective().evaluate(first)
    from_second = objective().evaluate(second)

    assert to_ab == to_ba
    assert abs(nll(twin, AB) - nll(twin, BA)) > 0.5
    assert from_first.diagnostics["distance"].item() == to_ab
    assert from_second.diagnostics["distance"].item() == to_ab
    assert from_first.diagnostics["likelihood"].item() == pytest.approx(
        nll(twin, AB), rel=1e-12
    )
    assert from_second.diagnostics["likelihood"].item() == pytest.approx(
        nll(twin, BA), rel=1e-12
    )
    assert from_first.value.item() == pytest.approx(by_hand(first), rel=1e-12)
    assert from_second.value.item() == pytest.approx(by_hand(second), rel=1e-12)


def test_the_value_reaches_every_component() -> None:
    tokens = embedder()
    encoder = IndicatorEncoder()
    decoder = ReadingDecoder()
    (weight,) = tuple(tokens.parameters())

    objective(tokens, encoder, decoder).evaluate(bundle()).value.backward()

    assert weight.grad is not None
    assert encoder.weight.grad is not None
    assert decoder.gain.grad is not None
    assert bool(weight.grad[VOCABULARY.index_of("amb")].abs().sum() > 0.0)
    assert bool(weight.grad[VOCABULARY.index_of("far")].abs().sum() == 0.0)
    assert bool(encoder.weight.grad.abs().sum() > 0.0)
    assert decoder.gain.grad.item() != 0.0


def test_a_candidate_holding_no_type_is_never_selected() -> None:
    outputs = bundle()
    with_dead = objective().evaluate(refilled(outputs))

    assert with_dead.count == 3
    assert with_dead.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)


def test_a_word_no_row_encodes_is_never_selected() -> None:
    outputs = bundle_over((("near", "one"),), (((0, 0), A), ((0, 1), A), ((0, 1), AB)))
    whole = objective(embedder()).evaluate(outputs)
    cut = objective(embedder(window=ContextWindow(1, overflow=REPORT))).evaluate(
        outputs
    )
    alone = objective(embedder()).evaluate(bundle_over((("near",),), (((0, 0), A),)))

    assert whole.count == 2
    assert cut.count == 1
    assert cut.value.item() == pytest.approx(alone.value.item(), rel=1e-12)


def test_a_token_with_no_live_candidate_is_neither_scored_nor_counted() -> None:
    scored = bundle_over((("near", "one"),), (((0, 0), A),))
    padded = bundle_of(
        (("near", "one"),), (*scored.batch.types.rows, dead_row()), ((0, 0), (0, 1))
    )

    alone = objective().evaluate(scored)
    beside = objective().evaluate(padded)

    assert alone.count == 1
    assert beside.count == 1
    assert beside.value.item() == pytest.approx(alone.value.item(), rel=1e-12)


def test_a_batch_holding_no_live_candidate_is_inapplicable() -> None:
    term = objective().evaluate(empty_bundle())

    assert term.count == 0
    assert term.value.item() == 0.0
    assert term.value.grad_fn is not None
    assert term.diagnostics == {}


def test_a_candidate_without_a_position_is_refused() -> None:
    outputs = bundle_of((("amb",),), CODEC.encode_batch((AB, BA)).rows, ((0, 0),))

    with pytest.raises(ValueError, match="2 candidate types at 1 positions"):
        objective().evaluate(outputs)


@pytest.mark.parametrize("position", [(1, 0), (0, 2)])
def test_a_position_beyond_the_batch_is_refused(position: tuple[int, int]) -> None:
    outputs = bundle_over((("amb", "one"),), ((position, AB),))

    with pytest.raises(ValueError, match="beyond the 1 items and 2 words"):
        objective().evaluate(outputs)


def test_a_readout_that_drops_a_selected_type_is_refused() -> None:
    with pytest.raises(ValueError, match="weighed 2 of the 3"):
        objective(decoder=ForgetfulDecoder()).evaluate(bundle())


def test_the_objective_declares_what_it_reads() -> None:
    built = objective()

    assert built.requires == frozenset({"batch"})
    assert built.name == "token-warmup"
    assert "StaticTokenEmbedder" in repr(built)
    assert "FirstSubwordPooler" in repr(built)
    assert "IndicatorEncoder" in repr(built)
    assert "ReadingDecoder" in repr(built)


def test_a_blank_name_is_refused() -> None:
    with pytest.raises(ValueError, match="name"):
        TokenWarmupObjective(
            cast("TokenEncoder", EMBEDDER),
            cast("SubwordPooler", POOLER),
            cast("TermEncoder", ENCODER),
            cast("TeacherForcedDecoder", DECODER),
            CHOICES,
            name="",
        )


def test_the_term_reaches_the_suite_under_its_name() -> None:
    outputs = bundle()
    built = objective()

    result = ObjectiveSuite([(2.0, built)]).evaluate(outputs)

    assert result.total.item() == pytest.approx(
        2.0 * built.evaluate(outputs).value.item(), rel=1e-12
    )
    assert set(result.diagnostics) == {
        "token-warmup/loss",
        "token-warmup/distance",
        "token-warmup/likelihood",
    }


def shipped() -> tuple[StaticTokenEmbedder, TreeRnnEncoder, RecursiveTermDecoder]:
    """Return a table, the tree recurrence encoder and decoder over the codec."""
    view = NeuralCodecView(CODEC)
    torch.manual_seed(3)
    encoder = TreeRnnEncoder(
        codec=view,
        choices=CHOICES,
        component=ComponentId("tree-encoder"),
        scheme=ArityHeadScheme(width=8, reached=SIGNATURE.operators),
        cell="gru",
        layers=1,
        norm_cap=None,
        inventory="uncached",
    ).to(torch.float64)
    decoder = RecursiveTermDecoder(
        codec=view,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("tree-decoder"),
        settings=TreeDecoderSettings(encoder.output_width, 6),
        choice_heads="flat",
    ).to(torch.float64)
    rows = torch.randn(VOCABULARY.size, encoder.output_width, dtype=torch.float64)
    tokens = StaticTokenEmbedder(
        rows,
        vocabulary=VOCABULARY,
        component=ComponentId("embedder"),
        trainable=True,
        window=None,
        load_report=None,
    )

    return tokens, encoder, decoder


def test_the_value_over_the_shipped_components_is_finite_and_reaches_them() -> None:
    tokens, encoder, decoder = shipped()
    built = TokenWarmupObjective(
        tokens,
        FirstSubwordPooler(encoder.output_width, component=ComponentId("pooler")),
        encoder,
        decoder,
        CHOICES,
    )

    term = built.evaluate(bundle())
    term.value.backward()

    assert term.count == 3
    assert math.isfinite(term.value.item())
    assert term.diagnostics["distance"].item() > 0.0
    assert term.diagnostics["likelihood"].item() > 0.0

    for module in (tokens, encoder, decoder):
        gradients = [
            parameter.grad
            for parameter in module.parameters()
            if parameter.grad is not None
        ]

        assert gradients
        assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
        assert any(bool((gradient != 0.0).any()) for gradient in gradients)
