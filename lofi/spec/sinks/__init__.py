"""Metric sink declarations."""

from typing import TYPE_CHECKING, cast

import didactic.api as dx

if TYPE_CHECKING:
    from collections.abc import Callable


def _validates[**P, R](
    *field_names: str,
) -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a field-validator decorator."""
    return cast(
        "Callable[[Callable[P, R]], Callable[P, R]]",
        dx.validates(*field_names),
    )


def _model_validator[**P, R]() -> Callable[[Callable[P, R]], Callable[P, R]]:
    """Return a model-validator decorator."""
    return cast(
        "Callable[[Callable[P, R]], Callable[P, R]]",
        dx.model_validator(),
    )
