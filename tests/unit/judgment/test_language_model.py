import math
from collections import Counter
from typing import TYPE_CHECKING, Final, NamedTuple, override

import pytest
import torch
from tests.conformance.lm_score import LmScoreSubject, LmScoreSuite
from torch import nn

from lofi._artifacts import PinnedCheckpoint
from lofi.judgment import DesignError, JudgmentRow, LmScore, RowColumns, SentenceScore
from lofi.judgment.language_model import (
    CausalLmScore,
    LmReadoutColumns,
    UnattestedTokenError,
    UnigramTable,
    lm_readout_design,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from torch import Tensor

BOS: Final = "<s>"
VOCABULARY: Final = (BOS, "the", "dog", "cat", "barked", "slept", "on", "a", "mat")
SENTENCES: Final = ("the dog barked", "a cat slept on the mat", "the cat barked")
COUNTS: Final = {
    "the": 7,
    "dog": 2,
    "cat": 3,
    "barked": 1,
    "slept": 2,
    "on": 4,
    "a": 6,
    "mat": 1,
}
CHECKPOINT: Final = PinnedCheckpoint(
    "an-org/a-model", "0123456789abcdef" * 2 + "01234567"
)
SEED: Final = 5


class Output(NamedTuple):
    logits: Tensor


class PrefixModel(nn.Module):
    """Predicts each next token from the running mean of the embedded prefix."""

    def __init__(self) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(SEED)
        self.embedding = nn.Embedding(len(VOCABULARY), 4)
        self.output = nn.Linear(4, len(VOCABULARY))

        with torch.no_grad():
            for parameter in self.parameters():
                parameter.copy_(torch.randn(parameter.shape, generator=generator))

    @override
    def forward(self, *, input_ids: Tensor) -> Output:
        embedded = self.embedding(input_ids)
        positions = torch.arange(1, input_ids.shape[1] + 1).reshape(1, -1, 1)

        return Output(self.output(embedded.cumsum(dim=1) / positions))


class WhitespaceTokenizer:
    """Splits on whitespace over a closed vocabulary."""

    def __init__(self, bos_token_id: int | None = 0) -> None:
        self.bos_token_id = bos_token_id

    def encode(self, text: str, *, add_special_tokens: bool) -> list[int]:
        if add_special_tokens:
            pytest.fail("the scorer asked the tokenizer to add special tokens")

        return [VOCABULARY.index(word) for word in text.split()]

    def convert_ids_to_tokens(self, ids: list[int]) -> list[str]:
        return [VOCABULARY[index] for index in ids]


class Loader:
    """Hands out one model and tokenizer and records what it was asked for."""

    def __init__(
        self, model: PrefixModel, tokenizer: WhitespaceTokenizer | None = None
    ) -> None:
        self.model = model
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.requests: list[PinnedCheckpoint] = []

    @property
    def dtype(self) -> torch.dtype:
        return torch.float32

    def load_causal_model(self, checkpoint: PinnedCheckpoint) -> PrefixModel:
        self.requests.append(checkpoint)

        return self.model

    def load_sentence_tokenizer(
        self, checkpoint: PinnedCheckpoint
    ) -> WhitespaceTokenizer:
        self.requests.append(checkpoint)

        return self.tokenizer


def scorer(model: PrefixModel | None = None) -> CausalLmScore:
    return CausalLmScore(
        Loader(model or PrefixModel()), UnigramTable(COUNTS), checkpoint=CHECKPOINT
    )


def recomputed(model: PrefixModel, sentence: str) -> float:
    ids = [VOCABULARY.index(word) for word in (BOS, *sentence.split())]
    context = torch.tensor([ids])

    with torch.no_grad():
        logits = model(input_ids=context).logits[0].double()

    return -float(
        nn.functional.cross_entropy(logits[:-1], context[0, 1:], reduction="sum")
    )


class TestCausalLmScore(LmScoreSuite):
    @pytest.fixture
    def lm_score_subject(self) -> LmScoreSubject:
        return LmScoreSubject(scorer(), SENTENCES, SEED)


def test_a_causal_scorer_implements_the_score_protocol() -> None:
    assert isinstance(scorer(), LmScore)


def test_each_log_probability_is_recomputed_from_the_model_logits() -> None:
    model = PrefixModel()
    scores = scorer(model).score(SENTENCES)

    for sentence, score in zip(SENTENCES, scores, strict=True):
        assert float(score.log_probability) == pytest.approx(
            recomputed(model, sentence), abs=1e-9
        )


def test_the_first_token_is_scored_after_the_beginning_of_sequence_token() -> None:
    model = PrefixModel()
    (score,) = scorer(model).score(["dog"])

    assert float(score.log_probability) == pytest.approx(
        recomputed(model, "dog"), abs=1e-9
    )


def test_each_unigram_log_probability_is_a_log_relative_frequency() -> None:
    total = sum(COUNTS.values())
    (score,) = scorer().score(["a cat slept on the mat"])
    expected = [math.log(COUNTS[token] / total) for token in score.tokens]

    assert score.unigram_log_probabilities.tolist() == pytest.approx(expected)
    assert float(score.unigram_log_probability) == pytest.approx(sum(expected))


def test_a_score_carries_the_sentence_tokens_and_their_count() -> None:
    (score,) = scorer().score(["a cat slept on the mat"])

    assert score.tokens == ("a", "cat", "slept", "on", "the", "mat")
    assert score.length == 6


def test_the_scorer_loads_and_records_its_checkpoint() -> None:
    loader = Loader(PrefixModel())
    scoring = CausalLmScore(loader, UnigramTable(COUNTS), checkpoint=CHECKPOINT)

    assert scoring.checkpoint == CHECKPOINT
    assert loader.requests == [CHECKPOINT, CHECKPOINT]


def test_the_scorer_puts_its_model_in_evaluation_mode() -> None:
    model = PrefixModel().train()
    scorer(model)

    assert not model.training


def test_a_token_absent_from_the_unigram_table_is_refused_by_name() -> None:
    table = UnigramTable(
        {piece: count for piece, count in COUNTS.items() if piece != "mat"}
    )
    scoring = CausalLmScore(Loader(PrefixModel()), table, checkpoint=CHECKPOINT)

    with pytest.raises(UnattestedTokenError, match="'mat'") as raised:
        scoring.score(["the cat slept on the mat"])

    assert raised.value.unattested == "mat"


def test_a_tokenizer_without_a_beginning_of_sequence_token_is_refused() -> None:
    with pytest.raises(ValueError, match="beginning-of-sequence"):
        CausalLmScore(
            Loader(PrefixModel(), WhitespaceTokenizer(bos_token_id=None)),
            UnigramTable(COUNTS),
            checkpoint=CHECKPOINT,
        )


def test_a_sentence_without_tokens_is_refused() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        scorer().score([" "])


def test_a_unigram_table_holds_log_relative_frequencies() -> None:
    table = UnigramTable({"a": 1, "b": 3})

    assert table.log_frequencies(("b", "a", "b")).tolist() == pytest.approx(
        [math.log(0.75), math.log(0.25), math.log(0.75)]
    )
    assert table.log_frequencies(("a",)).dtype == torch.float64
    assert len(table) == 2


@pytest.mark.parametrize("count", [0, -2])
def test_a_unigram_table_refuses_a_count_below_one(count: int) -> None:
    with pytest.raises(ValueError, match="count"):
        UnigramTable({"a": 1, "b": count})


def test_an_empty_unigram_table_is_refused() -> None:
    with pytest.raises(ValueError, match="no tokens"):
        UnigramTable({})


class CountingScorer:
    """Scores a sentence by its word count and records every request."""

    def __init__(self) -> None:
        self.requests: list[str] = []

    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        self.requests.extend(sentences)

        return tuple(
            SentenceScore(
                tuple(sentence.split()),
                torch.full((len(sentence.split()),), -2.0),
                torch.full((len(sentence.split()),), -5.0),
            )
            for sentence in sentences
        )


class Source:
    """Three judgments of two items, one with a declared factor."""

    def __init__(self, rows: Iterable[JudgmentRow]) -> None:
        self._rows = tuple(rows)

    @property
    def response_levels(self) -> int:
        return 7

    @property
    def factors(self) -> tuple[str, ...]:
        return ("frame",)

    @property
    def skips(self) -> Mapping[str, int]:
        return {}

    def rows(self) -> Iterable[JudgmentRow]:
        return self._rows


def source() -> Source:
    return Source(
        JudgmentRow(participant, item, response, RowColumns({"frame": frame}, {}))
        for participant, item, response, frame in (
            ("p1", "i1", 3, "NP V"),
            ("p2", "i1", 5, "NP V"),
            ("p1", "i2", 1, "NP V NP"),
        )
    )


READOUTS: Final = LmReadoutColumns("log_p", "unigram_log_p", "tokens")
LM_LOG_PROBABILITY, UNIGRAM_LOG_PROBABILITY, SENTENCE_LENGTH = READOUTS.columns
SENTENCE_OF: Final = {"i1": "the dog barked", "i2": "a cat slept on the mat"}


def test_the_readout_design_joins_three_declared_readouts_to_every_row() -> None:
    design = lm_readout_design(source(), SENTENCE_OF, CountingScorer(), READOUTS)
    items = [design.item_levels[int(index)] for index in design.items]

    assert tuple(design.readouts) == (
        LM_LOG_PROBABILITY,
        UNIGRAM_LOG_PROBABILITY,
        SENTENCE_LENGTH,
    )
    assert items == ["i1", "i1", "i2"]
    assert design.readouts[LM_LOG_PROBABILITY].tolist() == [-6.0, -6.0, -12.0]
    assert design.readouts[UNIGRAM_LOG_PROBABILITY].tolist() == [-15.0, -15.0, -30.0]
    assert design.readouts[SENTENCE_LENGTH].tolist() == [3.0, 3.0, 6.0]
    assert sorted(design.responses.tolist()) == [1, 3, 5]
    assert tuple(design.factors) == ("frame",)


def test_the_readout_design_scores_each_item_once() -> None:
    counting = CountingScorer()
    lm_readout_design(source(), SENTENCE_OF, counting, READOUTS)

    assert Counter(counting.requests) == Counter(SENTENCE_OF.values())


def test_an_item_without_a_sentence_is_refused_by_name() -> None:
    with pytest.raises(DesignError, match="'i2'"):
        lm_readout_design(
            source(), {"i1": "the dog barked"}, CountingScorer(), READOUTS
        )


def test_the_readout_columns_are_named_once_each() -> None:
    with pytest.raises(ValueError, match="repeat a column"):
        LmReadoutColumns("score", "score", "tokens")
