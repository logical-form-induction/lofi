"""Reusable laws for latent-predictor evaluators."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import pytest
import torch

from lofi._checks import LawViolationError
from lofi.judgment import Predictor

if TYPE_CHECKING:
    from lofi.judgment import JudgmentDesign, LatentPredictor

__all__ = [
    "LATENT_PREDICTOR_LAWS",
    "LatentPredictorSubject",
    "LatentPredictorSuite",
    "declared_terms_are_stable",
    "latent_scores_are_repeatable",
    "latent_scores_cover_every_row",
]


class LatentPredictorSubject(NamedTuple):
    """An evaluator and one design for it to score."""

    predictor: LatentPredictor
    design: JudgmentDesign


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a latent-predictor law violation.

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


def declared_terms_are_stable(subject: LatentPredictorSubject, /) -> None:
    """Check that the evaluator names a fixed, nonempty tuple of declared terms.

    Parameters
    ----------
    subject : LatentPredictorSubject
        Evaluator and one design for it to score.

    Raises
    ------
    LawViolationError
        If the terms are not a nonempty tuple of declared predictor terms, or two
        reads of them disagree.
    """
    law = "declared_terms_are_stable"
    terms = subject.predictor.terms

    if not isinstance(terms, tuple):
        _fail(f"terms are a {type(terms).__name__}, expected a tuple", law=law)

    if not terms:
        _fail("terms are empty, and a linear predictor has at least one", law=law)

    for position, term in enumerate(terms):
        if not isinstance(term, Predictor):
            _fail(
                f"term {position} is a {type(term).__name__}, which is not a "
                f"declared predictor term",
                law=law,
            )

    if subject.predictor.terms != terms:
        _fail("two reads of the terms disagree", law=law)


def latent_scores_cover_every_row(subject: LatentPredictorSubject, /) -> None:
    """Check the shape, kind and finiteness of the scores over one design.

    Parameters
    ----------
    subject : LatentPredictorSubject
        Evaluator and one design for it to score.

    Raises
    ------
    LawViolationError
        If the scores are not one finite floating-point value per design row.
    """
    law = "latent_scores_cover_every_row"
    scores = subject.predictor.latent_scores(subject.design)
    expected_shape = (subject.design.n_rows,)

    if scores.shape != expected_shape:
        _fail(
            f"scores have shape {tuple(scores.shape)}, expected {expected_shape}",
            law=law,
        )

    if not scores.is_floating_point():
        _fail(
            f"scores hold {scores.dtype}, which is not a floating-point kind", law=law
        )

    if not bool(torch.all(torch.isfinite(scores))):
        _fail("scores contain a nonfinite value", law=law)


def latent_scores_are_repeatable(subject: LatentPredictorSubject, /) -> None:
    """Check that scoring one design twice gives the same scores.

    Parameters
    ----------
    subject : LatentPredictorSubject
        Evaluator and one design for it to score.

    Raises
    ------
    LawViolationError
        If two scorings of one design differ, or the second changes the responses
        recorded in the design.
    """
    law = "latent_scores_are_repeatable"
    responses = subject.design.responses.clone()
    first = subject.predictor.latent_scores(subject.design).clone()
    second = subject.predictor.latent_scores(subject.design)

    if not bool(torch.equal(first, second)):
        departure = float(torch.max(torch.abs(first - second)))
        _fail(f"two scorings differ by up to {departure}", law=law)

    if not bool(torch.equal(responses, subject.design.responses)):
        _fail("scoring changed the responses recorded in the design", law=law)


class LatentPredictorSuite:
    """Reusable latent-predictor conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_declared_terms_are_stable(
        self,
        latent_predictor_subject: LatentPredictorSubject,
    ) -> None:
        """Run the declared-terms law.

        Parameters
        ----------
        latent_predictor_subject : LatentPredictorSubject
            Evaluator and one design for it to score.
        """
        declared_terms_are_stable(latent_predictor_subject)

    def test_latent_scores_cover_every_row(
        self,
        latent_predictor_subject: LatentPredictorSubject,
    ) -> None:
        """Run the score-coverage law.

        Parameters
        ----------
        latent_predictor_subject : LatentPredictorSubject
            Evaluator and one design for it to score.
        """
        latent_scores_cover_every_row(latent_predictor_subject)

    def test_latent_scores_are_repeatable(
        self,
        latent_predictor_subject: LatentPredictorSubject,
    ) -> None:
        """Run the repeatability law.

        Parameters
        ----------
        latent_predictor_subject : LatentPredictorSubject
            Evaluator and one design for it to score.
        """
        latent_scores_are_repeatable(latent_predictor_subject)


LATENT_PREDICTOR_LAWS: Final = (
    LatentPredictorSuite.test_declared_terms_are_stable,
    LatentPredictorSuite.test_latent_scores_cover_every_row,
    LatentPredictorSuite.test_latent_scores_are_repeatable,
)
