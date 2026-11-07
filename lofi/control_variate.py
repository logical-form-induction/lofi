"""Declared control variates for sampled policy gradients."""

from __future__ import annotations

from math import fsum
from typing import (
    TYPE_CHECKING,
    Literal,
    NamedTuple,
    Protocol,
    final,
    override,
    runtime_checkable,
)

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

__all__ = [
    "ControlVariate",
    "ControlVariateDeclaration",
    "ControlVariatePlugin",
    "ControlVariateSettingsDeclaration",
    "ControlVariateSpec",
    "MovingControlVariate",
    "MovingControlVariateSettings",
    "MovingControlVariateSpec",
]


@runtime_checkable
class ControlVariate(Protocol):
    """The control variate subtracted from sampled-form payments."""

    @property
    def value(self) -> float:
        """Return what the next step subtracts from each payment."""
        ...

    def observe(self, rewards: Sequence[float], /) -> None:
        """Update the control variate from one step's decided payments."""
        ...


class ControlVariateDeclaration(NamedTuple):
    """Resolved tag and settings for one registered control variate."""

    kind: str
    settings: object


class MovingControlVariateSettings(NamedTuple):
    """Resolved settings for an exponential moving control variate."""

    decay: float
    initial: float


class ControlVariateSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare a policy gradient control variate."""


class ControlVariateSettingsDeclaration(Protocol):
    """A control-variate declaration variant that resolves its interior settings."""

    def settings(self) -> object:
        """Return the control variate's immutable interior settings."""
        ...


class MovingControlVariateSpec(ControlVariateSpec):
    """Declare an exponential moving mean of decided rewards."""

    kind: Literal["moving"] = "moving"
    decay: float
    initial: float = 0.0

    @dx.validates("decay")
    @classmethod
    def _decay_is_a_probability_below_one(cls, decay: float) -> float:
        """Return a finite decay in the half-open unit interval."""
        held = ensure_finite("a control-variate decay", decay)

        if not 0.0 <= held < 1.0:
            msg = f"a control-variate decay lies in [0, 1), not {decay!r}"
            raise ValueError(msg)

        return held

    @dx.validates("initial")
    @classmethod
    def _initial_is_finite(cls, initial: float) -> float:
        """Return the finite value used before the first observation."""
        return ensure_finite("an initial control-variate value", initial)

    def settings(self) -> MovingControlVariateSettings:
        """Return the control variate's immutable interior settings."""
        return MovingControlVariateSettings(self.decay, self.initial)


@final
class MovingControlVariate:
    """Track an exponential moving mean over decided step rewards."""

    __slots__ = ("_decay", "_value")

    def __init__(self, *, decay: float, initial: float) -> None:
        self._decay = decay
        self._value = initial

    @property
    def value(self) -> float:
        """Return the value the next step subtracts."""
        return self._value

    def observe(self, rewards: Sequence[float], /) -> None:
        """Move toward the mean of one step's decided rewards."""
        if not rewards:
            return

        mean = fsum(
            ensure_finite("a control-variate reward", one) for one in rewards
        ) / len(rewards)
        self._value = self._decay * self._value + (1.0 - self._decay) * mean

    @override
    def __repr__(self) -> str:
        """Return the current value and declared decay."""
        return f"MovingControlVariate(decay={self._decay!r}, value={self._value!r})"


class ControlVariatePlugin[S](FrozenValue):
    """Build one runtime control variate from its resolved settings."""

    _noun = "control-variate plugin"
    __slots__ = ("_build", "_settings")

    def __init__(
        self,
        *,
        settings: type[S],
        build: Callable[[S], ControlVariate],
    ) -> None:
        object.__setattr__(self, "_settings", settings)
        object.__setattr__(self, "_build", build)

    def construct(self, settings: object, /) -> ControlVariate:
        """Build a control variate after checking the registered settings type."""
        if not isinstance(settings, self._settings):
            msg = (
                f"a control-variate plugin reads {self._settings.__name__}, "
                f"not {type(settings).__name__}"
            )
            raise TypeError(msg)

        return self._build(settings)


def _moving(settings: MovingControlVariateSettings, /) -> MovingControlVariate:
    """Build the declared exponential moving control variate."""
    return MovingControlVariate(decay=settings.decay, initial=settings.initial)


MOVING_PLUGIN = ControlVariatePlugin(
    settings=MovingControlVariateSettings, build=_moving
)
