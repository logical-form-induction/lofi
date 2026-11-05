"""Local discriminating witnesses for response-model laws."""

from typing import TYPE_CHECKING, Final, Literal

import pytest
import torch
from tests.conformance.response_model import (
    ResponseModelObservation,
    ResponseModelSubject,
    ResponseModelSuite,
    category_probabilities_are_valid,
    cumulative_probabilities_are_nondecreasing,
    cutpoint_count_matches_levels,
    extreme_probabilities_are_stable,
)

from lofi._checks import LawViolationError
from lofi.judgment import (
    ConditionalLikelihood,
    Cumulative,
    FixedDiscrimination,
    Predictor,
    PriorBlock,
    ResponseSpec,
    ScaleNormalization,
    ThresholdPriors,
)

if TYPE_CHECKING:
    from collections.abc import Callable

_NORMALIZATION_DEFECT: Final = 2e-6
_NEGATIVE_PROBABILITY: Final = -0.125
_ORDER_DEFECT: Final = 0.01
_POSITIVE_GAP_SCALE: Final = 1.1


class FixtureReadout(Predictor, extra="forbid"):
    """A local latent-score predictor declaration."""

    kind: Literal["response_model_fixture_readout"] = "response_model_fixture_readout"
    name: str


SPEC: Final = ResponseSpec(
    family=Cumulative(),
    levels=5,
    objective=ConditionalLikelihood(),
    predictor=(FixtureReadout(name="p_gram"),),
    identification=FixedDiscrimination(value=1.0),
    priors=PriorBlock(thresholds=ThresholdPriors()),
    normalization=ScaleNormalization.IDENTITY,
)

ALTERNATE_SPEC: Final = ResponseSpec(
    family=Cumulative(),
    levels=3,
    objective=ConditionalLikelihood(),
    predictor=(FixtureReadout(name="alternate_readout"),),
    identification=FixedDiscrimination(value=2.0),
    priors=PriorBlock(thresholds=ThresholdPriors()),
    normalization=ScaleNormalization.IDENTITY,
)


def _cutpoints(spec: ResponseSpec) -> torch.Tensor:
    """Build ordered cutpoints from the declared level count.

    Parameters
    ----------
    spec
        Response specification defining the level count.
    """
    return torch.linspace(-3.0, 3.0, spec.levels - 1, dtype=torch.float64)


def _stable_log_probabilities(
    eta: torch.Tensor,
    cutpoints: torch.Tensor,
) -> torch.Tensor:
    """Compute the local witness probabilities in log space.

    Parameters
    ----------
    eta
        Latent score for each response row.
    cutpoints
        Ordered response cutpoints.
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


def _stable_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Evaluate the conforming local witness.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    cutpoints = _cutpoints(spec)
    offsets = cutpoints.unsqueeze(0) - eta.unsqueeze(1)
    log_probabilities = _stable_log_probabilities(eta, cutpoints)

    return ResponseModelObservation(
        cutpoints=cutpoints,
        cumulative_probabilities=torch.sigmoid(offsets),
        probabilities=torch.exp(log_probabilities),
    )


def _negative_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Return normalized rows containing a negative category probability.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    observation = _stable_observation(spec, eta)
    probabilities = observation.probabilities.clone()
    transferred = probabilities[:, 0] - _NEGATIVE_PROBABILITY
    probabilities[:, 0] = _NEGATIVE_PROBABILITY
    probabilities[:, 1] += transferred

    return ResponseModelObservation(
        observation.cutpoints,
        observation.cumulative_probabilities,
        probabilities,
    )


def _unnormalized_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Return nonnegative rows whose category probabilities do not sum to one.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    observation = _stable_observation(spec, eta)
    probabilities = observation.probabilities.clone()
    probabilities[:, 0] += _NORMALIZATION_DEFECT

    return ResponseModelObservation(
        observation.cutpoints,
        observation.cumulative_probabilities,
        probabilities,
    )


def _decreasing_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Return cumulative probabilities that decrease between two levels.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    observation = _stable_observation(spec, eta)
    cumulative = observation.cumulative_probabilities.clone()
    cumulative[:, 1] = cumulative[:, 0] - _ORDER_DEFECT

    return ResponseModelObservation(
        observation.cutpoints,
        cumulative,
        observation.probabilities,
    )


def _wrong_cutpoint_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Return one fewer cutpoint than the declaration requires.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    observation = _stable_observation(spec, eta)

    return ResponseModelObservation(
        observation.cutpoints[:-1],
        observation.cumulative_probabilities,
        observation.probabilities,
    )


def _negative_gap_naive_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Subtract logistic CDF values only below every cutpoint.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    stable = _stable_observation(spec, eta)
    cutpoints = _cutpoints(spec)
    cumulative = torch.sigmoid(cutpoints.unsqueeze(0) - eta.unsqueeze(1))
    naive_probabilities = torch.cat(
        (
            cumulative[:, :1],
            torch.diff(cumulative, dim=1),
            1.0 - cumulative[:, -1:],
        ),
        dim=1,
    )
    probabilities = stable.probabilities.clone()
    negative_gap = eta < torch.min(cutpoints)
    probabilities[negative_gap] = naive_probabilities[negative_gap]

    return ResponseModelObservation(
        cutpoints,
        stable.cumulative_probabilities,
        probabilities,
    )


def _positive_gap_defect_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Perturb only probabilities above every declared cutpoint.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.
    """
    observation = _stable_observation(spec, eta)
    probabilities = observation.probabilities.clone()
    positive_gap = eta > torch.max(observation.cutpoints)
    added = probabilities[positive_gap, 0] * (_POSITIVE_GAP_SCALE - 1.0)
    probabilities[positive_gap, 0] += added
    probabilities[positive_gap, -1] -= added

    return ResponseModelObservation(
        observation.cutpoints,
        observation.cumulative_probabilities,
        probabilities,
    )


def _alternate_observation(
    spec: ResponseSpec,
    eta: torch.Tensor,
) -> ResponseModelObservation:
    """Evaluate a non-default response specification.

    Parameters
    ----------
    spec
        Response specification defining all model options.
    eta
        Latent score for each response row.

    Raises
    ------
    ValueError
        If the evaluator does not receive the three-level specification.
    """
    if spec.levels != ALTERNATE_SPEC.levels:
        message = "the alternate witness requires the three-level specification"

        raise ValueError(message)

    return _stable_observation(spec, eta)


def _subject(
    observe: Callable[[ResponseSpec, torch.Tensor], ResponseModelObservation],
) -> ResponseModelSubject:
    """Bind one local evaluator to the shared response specification.

    Parameters
    ----------
    observe
        Observable evaluator to bind.
    """
    return ResponseModelSubject(SPEC, observe)


class TestStableResponseModel(ResponseModelSuite):
    """Conformance binding for the stable local witness."""

    @pytest.fixture
    def response_model_subject(self) -> ResponseModelSubject:
        """Return the conforming local subject."""
        return _subject(_stable_observation)


class TestAlternateResponseModel(ResponseModelSuite):
    """Conformance binding for a three-level non-default specification."""

    @pytest.fixture
    def response_model_subject(self) -> ResponseModelSubject:
        """Return the alternate conforming subject."""
        return ResponseModelSubject(ALTERNATE_SPEC, _alternate_observation)


def test_probability_law_rejects_negative_values() -> None:
    """Verify that the probability law rejects a negative category."""
    with pytest.raises(
        LawViolationError,
        match=r"category_probabilities_are_valid.*minimum probability",
    ):
        category_probabilities_are_valid(_subject(_negative_observation))


def test_probability_law_rejects_unnormalized_rows() -> None:
    """Verify that the probability law rejects excess row mass."""
    with pytest.raises(
        LawViolationError,
        match=r"category_probabilities_are_valid.*row-sum deviation",
    ):
        category_probabilities_are_valid(_subject(_unnormalized_observation))


def test_cumulative_law_rejects_a_decreasing_step() -> None:
    """Verify that the cumulative law rejects a decrease by level."""
    with pytest.raises(
        LawViolationError,
        match=r"cumulative_probabilities_are_nondecreasing.*minimum adjacent",
    ):
        cumulative_probabilities_are_nondecreasing(_subject(_decreasing_observation))


def test_cutpoint_law_rejects_the_wrong_count() -> None:
    """Verify that the cutpoint law derives its count from the specification."""
    with pytest.raises(
        LawViolationError,
        match=r"cutpoint_count_matches_levels.*expected \(4,\)",
    ):
        cutpoint_count_matches_levels(_subject(_wrong_cutpoint_observation))


def test_stability_law_requires_the_negative_gap() -> None:
    """Verify that the stability law rejects naive negative-gap differencing."""
    with pytest.raises(
        LawViolationError,
        match=r"extreme_probabilities_are_stable.*gap thirty",
    ):
        extreme_probabilities_are_stable(_subject(_negative_gap_naive_observation))


def test_stability_law_requires_the_positive_gap() -> None:
    """Verify that the stability law evaluates the positive gap."""
    with pytest.raises(
        LawViolationError,
        match=r"extreme_probabilities_are_stable.*gap thirty",
    ):
        extreme_probabilities_are_stable(_subject(_positive_gap_defect_observation))
