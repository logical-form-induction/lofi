import math
import secrets
from typing import TYPE_CHECKING, Final, override

import pytest
import torch
from tests.conformance.lm_score import (
    LmScoreSubject,
    LmScoreSuite,
    scores_are_deterministic_at_a_fixed_seed,
    scores_are_invariant_to_batch_composition_and_padding,
)

from lofi._checks import LawViolationError
from lofi.judgment import SentenceScore

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

SENTENCES: Final = (
    "the dog barked",
    "a very long sentence about the dog that barked all night",
    "birds sing",
    "the cat slept on the warm mat",
)
SEED: Final = 11
PAD: Final = ""


def unigram_log_probability(token: str) -> float:
    return -math.log1p(len(token))


def bigram_log_probability(previous: str, token: str) -> float:
    return -0.5 - 0.1 * len(token) - (0.3 if previous[:1] == token[:1] else 0.0)


class BigramScorer:
    """Scores each token from its predecessor, padding a batch to one length."""

    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        tokenized = [tuple(sentence.split()) for sentence in sentences]
        width = max(len(tokens) for tokens in tokenized)
        padded = [tokens + (PAD,) * (width - len(tokens)) for tokens in tokenized]
        table = self._table(padded)

        return tuple(
            SentenceScore(
                tokens,
                table[row, : len(tokens)],
                torch.tensor([unigram_log_probability(token) for token in tokens]),
            )
            for row, tokens in enumerate(tokenized)
        )

    def _table(self, padded: list[tuple[str, ...]]) -> Tensor:
        return torch.tensor(
            [
                [
                    bigram_log_probability(
                        tokens[position - 1] if position else "", token
                    )
                    if token != PAD
                    else 0.0
                    for position, token in enumerate(tokens)
                ]
                for tokens in padded
            ]
        )


class PaddedReader(BigramScorer):
    """Normalizes each row over the padded width, pads included."""

    @override
    def _table(self, padded: list[tuple[str, ...]]) -> Tensor:
        table = super()._table(padded)

        return table - torch.logsumexp(table, dim=1, keepdim=True)


class UnigramPaddedReader(BigramScorer):
    """Divides each unigram log-probability by the padded width."""

    @override
    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        width = max(len(sentence.split()) for sentence in sentences)

        return tuple(
            SentenceScore(
                score.tokens,
                score.log_probabilities,
                score.unigram_log_probabilities / width,
            )
            for score in super().score(sentences)
        )


class PositionReader(BigramScorer):
    """Shifts each row by its position in the batch."""

    @override
    def _table(self, padded: list[tuple[str, ...]]) -> Tensor:
        table = super()._table(padded)

        return table - torch.arange(len(padded)).unsqueeze(1) * 0.01


class SeededNoise(BigramScorer):
    """Adds noise drawn from the torch generator."""

    @override
    def _table(self, padded: list[tuple[str, ...]]) -> Tensor:
        table = super()._table(padded)

        return table - 0.01 * torch.rand_like(table)


class UnseededNoise(BigramScorer):
    """Adds noise the torch seed does not fix."""

    @override
    def _table(self, padded: list[tuple[str, ...]]) -> Tensor:
        table = super()._table(padded)

        return table - secrets.randbelow(1000) / 100_000.0


class Retokenizer(BigramScorer):
    """Splits differently when a batch holds exactly two sentences."""

    @override
    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        if len(sentences) != 2:
            return super().score(sentences)

        return tuple(
            SentenceScore(
                (sentence.replace(" ", "_"),),
                torch.tensor([-1.0]),
                torch.tensor([-1.0]),
            )
            for sentence in sentences
        )


class Dropper(BigramScorer):
    """Returns one score fewer than the batch has sentences."""

    @override
    def score(self, sentences: Sequence[str], /) -> tuple[SentenceScore, ...]:
        return super().score(sentences)[:-1]


def subject_of(
    scorer: BigramScorer, sentences: tuple[str, ...] = SENTENCES
) -> LmScoreSubject:
    return LmScoreSubject(scorer, sentences, SEED)


class TestBigramScorer(LmScoreSuite):
    @pytest.fixture
    def lm_score_subject(self) -> LmScoreSubject:
        return subject_of(BigramScorer())


def test_a_scorer_reading_the_padded_width_fails_the_batch_law() -> None:
    subject = subject_of(PaddedReader())

    scores_are_deterministic_at_a_fixed_seed(subject)

    with pytest.raises(LawViolationError, match="scoring beside"):
        scores_are_invariant_to_batch_composition_and_padding(subject)


def test_a_scorer_padding_the_unigram_table_fails_the_batch_law() -> None:
    with pytest.raises(LawViolationError, match="moved the unigram log-probabilities"):
        scores_are_invariant_to_batch_composition_and_padding(
            subject_of(UnigramPaddedReader())
        )


def test_a_scorer_reading_the_batch_position_fails_the_batch_law() -> None:
    with pytest.raises(LawViolationError, match="reversed batch"):
        scores_are_invariant_to_batch_composition_and_padding(
            subject_of(PositionReader())
        )


def test_seeded_noise_is_deterministic_and_yet_depends_on_the_batch() -> None:
    subject = subject_of(SeededNoise())

    scores_are_deterministic_at_a_fixed_seed(subject)

    with pytest.raises(LawViolationError, match="moved the log-probabilities"):
        scores_are_invariant_to_batch_composition_and_padding(subject)


def test_unseeded_noise_fails_the_determinism_law() -> None:
    with pytest.raises(LawViolationError, match="second scoring at the same seed"):
        scores_are_deterministic_at_a_fixed_seed(subject_of(UnseededNoise()))


def test_a_scorer_whose_tokens_depend_on_the_batch_fails_the_batch_law() -> None:
    with pytest.raises(LawViolationError, match="tokenized"):
        scores_are_invariant_to_batch_composition_and_padding(subject_of(Retokenizer()))


def test_a_scorer_returning_fewer_scores_than_sentences_is_refused() -> None:
    with pytest.raises(LawViolationError, match="scored as 3 scores"):
        scores_are_deterministic_at_a_fixed_seed(subject_of(Dropper()))


def test_a_subject_with_too_few_sentences_is_refused() -> None:
    with pytest.raises(LawViolationError, match="supplies 2 sentences"):
        scores_are_deterministic_at_a_fixed_seed(
            subject_of(BigramScorer(), SENTENCES[:2])
        )


def test_a_subject_repeating_a_sentence_is_refused() -> None:
    with pytest.raises(LawViolationError, match="repeats a sentence"):
        scores_are_deterministic_at_a_fixed_seed(
            subject_of(BigramScorer(), (*SENTENCES[:2], SENTENCES[0]))
        )


def test_a_subject_whose_sentences_never_pad_is_refused() -> None:
    same_length = ("the dog barked", "a cat slept", "birds sing loudly")

    with pytest.raises(LawViolationError, match="so no batch pads"):
        scores_are_invariant_to_batch_composition_and_padding(
            subject_of(BigramScorer(), same_length)
        )
