"""Validation shared by the optimizer declarations."""

from lofi._checks import ensure_finite

__all__ = ["moment_decays", "nonnegative", "positive"]


def positive(field: str, value: float, /) -> float:
    """Return a positive finite hyperparameter.

    Parameters
    ----------
    field : str
        The field name the message names.
    value : float
        The value to validate.

    Raises
    ------
    ValueError
        If the value is not finite or not positive.
    """
    checked = ensure_finite(field, value)

    if checked <= 0.0:
        msg = f"{field} is not positive: {value!r}"

        raise ValueError(msg)

    return checked


def nonnegative(field: str, value: float, /) -> float:
    """Return a nonnegative finite hyperparameter.

    Parameters
    ----------
    field : str
        The field name the message names.
    value : float
        The value to validate.

    Raises
    ------
    ValueError
        If the value is not finite or is negative.
    """
    checked = ensure_finite(field, value)

    if checked < 0.0:
        msg = f"{field} is negative: {value!r}"

        raise ValueError(msg)

    return checked


def moment_decays(values: tuple[float, ...], /) -> tuple[float, ...]:
    """Return two finite moment-decay rates in ``[0, 1)``.

    Parameters
    ----------
    values : tuple[float, ...]
        The first- and second-moment decay rates.

    Raises
    ------
    ValueError
        If there are not exactly two rates, or a rate is not finite or lies
        outside ``[0, 1)``.
    """
    if len(values) != 2:
        msg = f"betas has length {len(values)}, not 2: {values!r}"

        raise ValueError(msg)

    checked = tuple(
        ensure_finite(f"betas[{position}]", value)
        for position, value in enumerate(values)
    )

    for position, value in enumerate(checked):
        if not 0.0 <= value < 1.0:
            msg = f"betas[{position}] lies outside [0, 1): {value!r}"

            raise ValueError(msg)

    return checked
