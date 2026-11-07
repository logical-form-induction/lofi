"""Stochastic gradient descent declaration."""

from typing import Literal, NamedTuple

import didactic.api as dx

from lofi.spec.components.trainer import (
    _DEFAULT_UNREGULARIZED_WEIGHT_DECAY,
    OptimizerSpec,
)
from lofi.spec.optimizers._checks import nonnegative, positive

__all__ = ["SGDOptimizerSettings", "SGDOptimizerSpec"]


class SGDOptimizerSettings(NamedTuple):
    """The settings a stochastic gradient descent optimizer is built from.

    Attributes
    ----------
    learning_rate : float
        The step size.
    momentum : float
        The momentum factor, zero for plain descent.
    dampening : float
        The dampening applied to the momentum.
    weight_decay : float
        The L2 penalty added to each gradient.
    nesterov : bool
        Whether the momentum is Nesterov momentum.
    """

    learning_rate: float
    momentum: float
    dampening: float
    weight_decay: float
    nesterov: bool


class SGDOptimizerSpec(OptimizerSpec):
    """Declares stochastic gradient descent, with optional momentum.

    Attributes
    ----------
    kind : Literal["sgd"]
        Registry tag.
    learning_rate : float
        Positive finite step size.
    momentum : float
        Finite nonnegative momentum factor.
    dampening : float
        Finite nonnegative dampening of the momentum.
    weight_decay : float
        Finite nonnegative L2 penalty.
    nesterov : bool
        Whether the momentum is Nesterov momentum, which needs a positive
        momentum and no dampening.
    """

    kind: Literal["sgd"] = "sgd"
    learning_rate: float = 1e-2
    momentum: float = 0.0
    dampening: float = 0.0
    weight_decay: float = _DEFAULT_UNREGULARIZED_WEIGHT_DECAY
    nesterov: bool = False

    @dx.validates("learning_rate")
    @classmethod
    def _learning_rate_is_positive(cls, value: float) -> float:
        """Validate the optimizer step size."""
        return positive("learning_rate", value)

    @dx.validates("momentum", "dampening", "weight_decay")
    @classmethod
    def _factor_is_nonnegative(cls, value: float) -> float:
        """Validate a momentum, dampening or penalty factor."""
        return nonnegative("an SGD factor", value)

    @dx.model_validator()
    def _nesterov_has_undamped_momentum(self) -> SGDOptimizerSpec:
        """Validate that Nesterov momentum has momentum and no dampening.

        Returns
        -------
        SGDOptimizerSpec
            This declaration after validation.

        Raises
        ------
        ValueError
            If Nesterov momentum is declared without positive momentum or with
            dampening.
        """
        if self.nesterov and (self.momentum <= 0.0 or self.dampening != 0.0):
            msg = (
                f"nesterov momentum needs a positive momentum and no dampening, "
                f"not momentum {self.momentum!r} and dampening {self.dampening!r}"
            )

            raise ValueError(msg)

        return self

    def settings(self) -> SGDOptimizerSettings:
        """Return the settings the plugin builds this optimizer from."""
        return SGDOptimizerSettings(
            learning_rate=self.learning_rate,
            momentum=self.momentum,
            dampening=self.dampening,
            weight_decay=self.weight_decay,
            nesterov=self.nesterov,
        )
