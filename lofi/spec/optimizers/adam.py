"""Adam optimizer declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi.spec.components.trainer import (
    _DEFAULT_ADAM_BETAS,
    _DEFAULT_ADAM_EPSILON,
    _DEFAULT_ADAM_LEARNING_RATE,
    _DEFAULT_UNREGULARIZED_WEIGHT_DECAY,
    OptimizerSpec,
)
from lofi.spec.optimizers._checks import moment_decays, nonnegative, positive

__all__ = ["AdamOptimizerSettings", "AdamOptimizerSpec"]


class AdamOptimizerSettings(NamedTuple):
    """The settings an Adam optimizer is built from.

    Attributes
    ----------
    learning_rate : float
        The step size.
    betas : tuple[float, float]
        The exponential-decay rates of the first and second moments.
    epsilon : float
        The numerical-stability term.
    weight_decay : float
        The L2 penalty added to each gradient.
    """

    learning_rate: float
    betas: tuple[float, float]
    epsilon: float
    weight_decay: float


class AdamOptimizerSpec(OptimizerSpec):
    """Declares Adam optimization, whose weight decay is an L2 penalty on the gradient.

    Attributes
    ----------
    kind : Literal["adam"]
        Registry tag.
    learning_rate : float
        Positive finite step size.
    betas : tuple[float, ...]
        Two finite exponential-decay rates in ``[0, 1)``.
    epsilon : float
        Positive finite numerical-stability term.
    weight_decay : float
        Finite nonnegative L2 penalty.
    """

    kind: Literal["adam"] = "adam"
    learning_rate: float = _DEFAULT_ADAM_LEARNING_RATE
    betas: tuple[float, ...] = _DEFAULT_ADAM_BETAS
    epsilon: float = _DEFAULT_ADAM_EPSILON
    weight_decay: float = _DEFAULT_UNREGULARIZED_WEIGHT_DECAY

    @dx.validates("learning_rate")
    @classmethod
    def _learning_rate_is_positive(cls, value: float) -> float:
        """Validate the optimizer step size."""
        return positive("learning_rate", value)

    @dx.validates("betas")
    @classmethod
    def _betas_are_rates(cls, values: tuple[float, ...]) -> tuple[float, ...]:
        """Validate the two moment-decay rates."""
        return moment_decays(values)

    @dx.validates("epsilon")
    @classmethod
    def _epsilon_is_positive(cls, value: float) -> float:
        """Validate the numerical-stability term."""
        return positive("epsilon", value)

    @dx.validates("weight_decay")
    @classmethod
    def _weight_decay_is_nonnegative(cls, value: float) -> float:
        """Validate the L2 penalty."""
        return nonnegative("weight_decay", value)

    def settings(self) -> AdamOptimizerSettings:
        """Return the settings the plugin builds this optimizer from."""
        return AdamOptimizerSettings(
            learning_rate=self.learning_rate,
            betas=(self.betas[0], self.betas[1]),
            epsilon=self.epsilon,
            weight_decay=self.weight_decay,
        )
