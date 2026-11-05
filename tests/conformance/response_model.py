"""Reusable laws for response-model likelihoods."""

from typing import TYPE_CHECKING, ClassVar, Final, NamedTuple, NoReturn

import pytest
import torch

from lofi._checks import LawViolationError

if TYPE_CHECKING:
    from collections.abc import Callable

    from torch import Tensor

    from lofi.judgment import ResponseSpec

__all__ = [
    "RESPONSE_MODEL_LAWS",
    "ResponseModelObservation",
    "ResponseModelSubject",
    "ResponseModelSuite",
    "category_probabilities_are_valid",
    "cumulative_probabilities_are_nondecreasing",
    "cutpoint_count_matches_levels",
    "extreme_probabilities_are_stable",
]

_GENERATED_ROWS: Final = 256
_LATENT_LIMIT: Final = 12.0
_PROBABILITY_ATOL: Final = 1e-6
_RANDOM_SEED: Final = 1907
_STABILITY_GAP: Final = 30.0


class ResponseModelObservation(NamedTuple):
    """Observable likelihood values for one latent-score batch."""

    cutpoints: Tensor
    cumulative_probabilities: Tensor
    probabilities: Tensor


class ResponseModelSubject(NamedTuple):
    """A response specification and its observable evaluator."""

    spec: ResponseSpec
    observe: Callable[[ResponseSpec, Tensor], ResponseModelObservation]


def _fail(detail: str, *, law: str) -> NoReturn:
    """Raise a response-model law violation.

    Parameters
    ----------
    detail
        Description of the failed condition.
    law
        Name of the failed law.

    Raises
    ------
    LawViolationError
        Always raised with the law name and detail.
    """
    message = f"{law}: {detail}"

    raise LawViolationError(message)


def _generated_latent_scores() -> Tensor:
    """Generate the latent-score rows used by distribution laws."""
    generator = torch.Generator().manual_seed(_RANDOM_SEED)

    return (
        torch.rand(_GENERATED_ROWS, generator=generator, dtype=torch.float64)
        * (2.0 * _LATENT_LIMIT)
        - _LATENT_LIMIT
    )


def _stable_log_probabilities(eta: Tensor, cutpoints: Tensor) -> Tensor:
    """Compute logistic category probabilities in log space.

    Parameters
    ----------
    eta
        Latent score for each response row.
    cutpoints
        Ordered cutpoints separating adjacent response levels.
    """
    offsets = cutpoints.unsqueeze(0) - eta.unsqueeze(1)
    first = -torch.nn.functional.softplus(-offsets[:, :1])
    lower = offsets[:, :-1]
    upper = offsets[:, 1:]
    middle = (
        lower
        + torch.log(torch.expm1(upper - lower))
        - torch.nn.functional.softplus(lower)
        - torch.nn.functional.softplus(upper)
    )
    last = -torch.nn.functional.softplus(offsets[:, -1:])

    return torch.cat((first, middle, last), dim=1)


def category_probabilities_are_valid(subject: ResponseModelSubject, /) -> None:
    """Check category probability bounds and normalization.

    Parameters
    ----------
    subject
        Response specification and observable evaluator.

    Raises
    ------
    LawViolationError
        If the evaluator returns an invalid shape, a nonfinite or negative value, or
        a row whose probabilities do not sum to one within tolerance.
    """
    law = "category_probabilities_are_valid"
    eta = _generated_latent_scores()
    observation = subject.observe(subject.spec, eta)
    probabilities = observation.probabilities
    expected_shape = (len(eta), subject.spec.levels)

    if probabilities.shape != expected_shape:
        _fail(
            f"probabilities have shape {tuple(probabilities.shape)}, expected "
            f"{expected_shape}",
            law=law,
        )

    if not bool(torch.all(torch.isfinite(probabilities))):
        _fail("probabilities contain a nonfinite value", law=law)

    if bool(torch.any(probabilities < 0.0)):
        minimum = float(torch.min(probabilities))
        _fail(f"minimum probability is {minimum}", law=law)

    totals = torch.sum(probabilities, dim=1)
    deviations = torch.abs(totals - 1.0)
    maximum_deviation = float(torch.max(deviations))

    if maximum_deviation > _PROBABILITY_ATOL:
        _fail(
            f"maximum row-sum deviation is {maximum_deviation} "
            f"(tolerance {_PROBABILITY_ATOL})",
            law=law,
        )


def cumulative_probabilities_are_nondecreasing(
    subject: ResponseModelSubject,
    /,
) -> None:
    """Check cumulative probability order across response levels.

    Parameters
    ----------
    subject
        Response specification and observable evaluator.

    Raises
    ------
    LawViolationError
        If the cumulative probabilities have the wrong shape, contain a nonfinite
        value, or decrease between adjacent levels.
    """
    law = "cumulative_probabilities_are_nondecreasing"
    eta = _generated_latent_scores()
    cumulative = subject.observe(subject.spec, eta).cumulative_probabilities
    expected_shape = (len(eta), subject.spec.levels - 1)

    if cumulative.shape != expected_shape:
        _fail(
            f"cumulative probabilities have shape {tuple(cumulative.shape)}, "
            f"expected {expected_shape}",
            law=law,
        )

    if not bool(torch.all(torch.isfinite(cumulative))):
        _fail("cumulative probabilities contain a nonfinite value", law=law)

    differences = torch.diff(cumulative, dim=1)

    if bool(torch.any(differences < 0.0)):
        minimum = float(torch.min(differences))
        _fail(f"minimum adjacent difference is {minimum}", law=law)


def cutpoint_count_matches_levels(subject: ResponseModelSubject, /) -> None:
    """Check that the model exposes one cutpoint between adjacent levels.

    Parameters
    ----------
    subject
        Response specification and observable evaluator.

    Raises
    ------
    LawViolationError
        If the evaluator does not expose exactly one fewer cutpoint than levels.
    """
    law = "cutpoint_count_matches_levels"
    eta = torch.zeros(1, dtype=torch.float64)
    cutpoints = subject.observe(subject.spec, eta).cutpoints
    expected_shape = (subject.spec.levels - 1,)

    if cutpoints.shape != expected_shape:
        _fail(
            f"cutpoints have shape {tuple(cutpoints.shape)}, expected {expected_shape}",
            law=law,
        )


def extreme_probabilities_are_stable(subject: ResponseModelSubject, /) -> None:
    """Check stable log-CDF differencing at a gap of thirty.

    Parameters
    ----------
    subject
        Response specification and observable evaluator.

    Raises
    ------
    LawViolationError
        If a returned shape is invalid, an extreme probability is nonfinite or
        nonpositive, or log probabilities differ from the reference beyond tolerance.
    """
    law = "extreme_probabilities_are_stable"
    initial = subject.observe(subject.spec, torch.zeros(1, dtype=torch.float64))
    cutpoints = initial.cutpoints
    expected_cutpoints = subject.spec.levels - 1

    if cutpoints.shape != (expected_cutpoints,):
        _fail(
            f"cutpoints have shape {tuple(cutpoints.shape)}, expected "
            f"{(expected_cutpoints,)}",
            law=law,
        )

    eta = torch.cat((cutpoints - _STABILITY_GAP, cutpoints + _STABILITY_GAP))
    observed = subject.observe(subject.spec, eta)
    expected_shape = (len(eta), subject.spec.levels)

    if observed.probabilities.shape != expected_shape:
        _fail(
            f"probabilities have shape {tuple(observed.probabilities.shape)}, "
            f"expected {expected_shape}",
            law=law,
        )

    reference = _stable_log_probabilities(eta, cutpoints)

    if not bool(torch.all(torch.isfinite(observed.probabilities))):
        _fail("probabilities contain a nonfinite value at gap thirty", law=law)

    if bool(torch.any(observed.probabilities <= 0.0)):
        _fail("probabilities contain a nonpositive value at gap thirty", law=law)

    differences = torch.abs(torch.log(observed.probabilities) - reference)
    maximum_difference = float(torch.max(differences))

    if maximum_difference > _PROBABILITY_ATOL:
        _fail(
            f"maximum log-probability difference is {maximum_difference} "
            f"at gap thirty (tolerance {_PROBABILITY_ATOL})",
            law=law,
        )


class ResponseModelSuite:
    """Reusable response-model conformance laws."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_category_probabilities_are_valid(
        self,
        response_model_subject: ResponseModelSubject,
    ) -> None:
        """Run the category-probability law.

        Parameters
        ----------
        response_model_subject
            Response specification and observable evaluator.
        """
        category_probabilities_are_valid(response_model_subject)

    def test_cumulative_probabilities_are_nondecreasing(
        self,
        response_model_subject: ResponseModelSubject,
    ) -> None:
        """Run the cumulative-probability law.

        Parameters
        ----------
        response_model_subject
            Response specification and observable evaluator.
        """
        cumulative_probabilities_are_nondecreasing(response_model_subject)

    def test_cutpoint_count_matches_levels(
        self,
        response_model_subject: ResponseModelSubject,
    ) -> None:
        """Run the cutpoint-count law.

        Parameters
        ----------
        response_model_subject
            Response specification and observable evaluator.
        """
        cutpoint_count_matches_levels(response_model_subject)

    def test_extreme_probabilities_are_stable(
        self,
        response_model_subject: ResponseModelSubject,
    ) -> None:
        """Run the extreme-probability law.

        Parameters
        ----------
        response_model_subject
            Response specification and observable evaluator.
        """
        extreme_probabilities_are_stable(response_model_subject)


RESPONSE_MODEL_LAWS: Final = (
    ResponseModelSuite.test_category_probabilities_are_valid,
    ResponseModelSuite.test_cumulative_probabilities_are_nondecreasing,
    ResponseModelSuite.test_cutpoint_count_matches_levels,
    ResponseModelSuite.test_extreme_probabilities_are_stable,
)
