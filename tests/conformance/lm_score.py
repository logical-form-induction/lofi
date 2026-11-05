"""Reusable laws for language-model scorers of sentences."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import pytest
import torch

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from lofi.judgment import LmScore, SentenceScore

__all__ = [
    "LmScoreSubject",
    "LmScoreSuite",
    "scores_are_deterministic_at_a_fixed_seed",
    "scores_are_invariant_to_batch_composition_and_padding",
]

#: Largest departure between two scorings of one sentence the laws accept.
BATCH_ATOL: Final = 1e-6

#: Fewest sentences a subject supplies.
MINIMUM_SENTENCES: Final = 3


class LmScoreSubject(NamedTuple):
    """Inputs for the language-model scorer conformance laws.

    Attributes
    ----------
    scorer : LmScore
        The scorer under test.
    sentences : tuple[str, ...]
        At least three sentences of at least two distinct token counts, so
        that scoring one beside a longer one pads it.
    seed : int
        Seed the determinism law fixes before each scoring.
    """

    scorer: LmScore
    sentences: tuple[str, ...]
    seed: int


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a scoring law violation.

    Parameters
    ----------
    detail : str
        Description of the failed condition.
    law : str
        Name of the failed law.

    Raises
    ------
    LawViolationError
        Always raised with the law name and detail.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _scored(
    subject: LmScoreSubject, batch: tuple[str, ...], *, law: str
) -> tuple[SentenceScore, ...]:
    """Return the scores of a batch, checked to be one per sentence.

    Parameters
    ----------
    subject : LmScoreSubject
        Scorer under test.
    batch : tuple[str, ...]
        Sentences to score together.
    law : str
        Name of the law reporting a defect.

    Raises
    ------
    LawViolationError
        If the scorer returns another number of scores than sentences.
    """
    torch.manual_seed(subject.seed)
    scores = tuple(subject.scorer.score(batch))

    if len(scores) != len(batch):
        _fail(
            f"{len(batch)} sentences were scored as {len(scores)} scores",
            law=law,
        )

    return scores


def _check_subject(subject: LmScoreSubject, *, law: str) -> None:
    """Check that a subject supplies sentences the laws can compare.

    Parameters
    ----------
    subject : LmScoreSubject
        Subject to check.
    law : str
        Name of the law reporting a defect.

    Raises
    ------
    LawViolationError
        If the subject supplies too few sentences, repeats one, or supplies
        sentences of one token count only.
    """
    sentences = subject.sentences

    if len(sentences) < MINIMUM_SENTENCES:
        _fail(
            f"the subject supplies {len(sentences)} sentences and the laws need "
            f"{MINIMUM_SENTENCES}",
            law=law,
        )

    if len(frozenset(sentences)) != len(sentences):
        _fail(f"the subject repeats a sentence: {sentences!r}", law=law)

    lengths = {score.length for score in _scored(subject, sentences, law=law)}

    if len(lengths) < 2:
        _fail(
            f"every subject sentence has {lengths.pop()} tokens, so no batch pads "
            f"one beside a longer one",
            law=law,
        )


def _compare(
    reference: SentenceScore,
    other: SentenceScore,
    *,
    context: str,
    law: str,
) -> None:
    """Check that two scores of one sentence agree.

    Parameters
    ----------
    reference : SentenceScore
        Score to compare against.
    other : SentenceScore
        Score that must agree with it.
    context : str
        How the other score was obtained, for the message.
    law : str
        Name of the law reporting a defect.

    Raises
    ------
    LawViolationError
        If the tokens differ, or a vector departs by more than the tolerance.
    """
    if other.tokens != reference.tokens:
        _fail(
            f"{context} tokenized {reference.tokens!r} as {other.tokens!r}",
            law=law,
        )

    vectors = (
        ("log-probabilities", reference.log_probabilities, other.log_probabilities),
        (
            "unigram log-probabilities",
            reference.unigram_log_probabilities,
            other.unigram_log_probabilities,
        ),
    )

    for name, expected, actual in vectors:
        if not torch.allclose(actual, expected, atol=BATCH_ATOL, rtol=0.0):
            departure = float(torch.max(torch.abs(actual - expected)))
            _fail(
                f"{context} moved the {name} of {reference.tokens!r} by "
                f"{departure} (tolerance {BATCH_ATOL})",
                law=law,
            )


def scores_are_deterministic_at_a_fixed_seed(subject: LmScoreSubject, /) -> None:
    """Check that scoring the sentences twice at one seed repeats the scores.

    Parameters
    ----------
    subject : LmScoreSubject
        Scorer and the sentences it is checked over.

    Raises
    ------
    LawViolationError
        If a second scoring at the same seed differs from the first.
    """
    law = "scores_are_deterministic_at_a_fixed_seed"
    _check_subject(subject, law=law)
    first = _scored(subject, subject.sentences, law=law)
    second = _scored(subject, subject.sentences, law=law)

    for before, after in zip(first, second, strict=True):
        _compare(before, after, context="a second scoring at the same seed", law=law)


def scores_are_invariant_to_batch_composition_and_padding(
    subject: LmScoreSubject, /
) -> None:
    """Check that a sentence scores the same in three batch shapes.

    Each sentence is scored on its own, beside the longest sentence, and in
    the full batch in reverse order, so that the padded length and the
    position within the batch both vary.

    Parameters
    ----------
    subject : LmScoreSubject
        Scorer and the sentences it is checked over.

    Raises
    ------
    LawViolationError
        If a sentence's score depends on the batch it is scored in.
    """
    law = "scores_are_invariant_to_batch_composition_and_padding"
    _check_subject(subject, law=law)
    sentences = subject.sentences
    alone = {
        sentence: _scored(subject, (sentence,), law=law)[0] for sentence in sentences
    }
    longest = max(sentences, key=lambda sentence: alone[sentence].length)
    reversed_batch = tuple(reversed(sentences))
    in_reverse = dict(
        zip(reversed_batch, _scored(subject, reversed_batch, law=law), strict=True)
    )

    for sentence in sentences:
        paired = (sentence, longest) if sentence != longest else (longest, sentences[0])
        beside = _scored(subject, paired, law=law)[0]
        _compare(
            alone[sentence],
            beside,
            context=f"scoring beside {paired[1]!r}",
            law=law,
        )
        _compare(
            alone[sentence],
            in_reverse[sentence],
            context="scoring in the reversed batch",
            law=law,
        )


class LmScoreSuite:
    """Reusable language-model scorer conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_scores_are_deterministic_at_a_fixed_seed(
        self, lm_score_subject: LmScoreSubject
    ) -> None:
        """Run the determinism law.

        Parameters
        ----------
        lm_score_subject : LmScoreSubject
            Scorer and the sentences it is checked over.
        """
        scores_are_deterministic_at_a_fixed_seed(lm_score_subject)

    def test_scores_are_invariant_to_batch_composition_and_padding(
        self, lm_score_subject: LmScoreSubject
    ) -> None:
        """Run the batch-invariance law.

        Parameters
        ----------
        lm_score_subject : LmScoreSubject
            Scorer and the sentences it is checked over.
        """
        scores_are_invariant_to_batch_composition_and_padding(lm_score_subject)
