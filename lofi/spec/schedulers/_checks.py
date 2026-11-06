"""Validation shared by the scheduler declarations."""

from lofi._checks import ensure_finite

__all__ = ["warmup_start_factor"]


def warmup_start_factor(value: float, /) -> float:
    """Return a warmup's starting fraction of the learning rate, in $(0, 1]$.

    Parameters
    ----------
    value : float
        The fraction of the declared learning rate the warmup starts from.

    Raises
    ------
    ValueError
        If the fraction is not finite or lies outside $(0, 1]$.
    """
    checked = ensure_finite("warmup_start_factor", value)

    if not 0.0 < checked <= 1.0:
        msg = f"warmup_start_factor lies outside (0, 1]: {value!r}"

        raise ValueError(msg)

    return checked
