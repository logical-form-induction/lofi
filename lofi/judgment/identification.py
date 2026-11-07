"""Reductions for diagnosing and reporting latent-scale identification."""

from typing import TYPE_CHECKING

from lofi.judgment._spec import ConstraintCost, Measurement, ScaleProfile

if TYPE_CHECKING:
    from collections.abc import Iterable

__all__ = ["constraint_cost", "scale_profile"]


def scale_profile(
    scales: Iterable[float],
    log_posterior: Iterable[float],
    /,
    *,
    minimum_tail_drop: float,
) -> ScaleProfile:
    """Reduce an observed latent-scale scan to its ridge diagnostic.

    Parameters
    ----------
    scales : Iterable[float]
        Ordered positive scale factors bracketing one.
    log_posterior : Iterable[float]
        Log-posterior value observed at each scale factor.
    minimum_tail_drop : float
        Smallest drop from the peak required at both endpoints.

    Returns
    -------
    ScaleProfile
        The validated profile with its computed peak, endpoint drops and
        ridge verdict.
    """
    return ScaleProfile(
        scales=tuple(scales),
        log_posterior=tuple(log_posterior),
        minimum_tail_drop=minimum_tail_drop,
    )


def constraint_cost(
    constrained: Measurement,
    unconstrained: Measurement,
    /,
) -> ConstraintCost:
    """Return unconstrained minus constrained predictive density.

    Parameters
    ----------
    constrained : Measurement
        Predictive density of the identified fit.
    unconstrained : Measurement
        Predictive density of the otherwise matched free-scale fit.

    Returns
    -------
    ConstraintCost
        The validated pair and its computed signed difference.
    """
    return ConstraintCost(
        constrained=constrained,
        unconstrained=unconstrained,
    )
