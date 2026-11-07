"""Boundary declaration of a registered fuzzy truth algebra."""

from typing import NamedTuple

import didactic.api as dx

from lofi._checks import ensure_finite, ensure_name
from lofi.registry import Registry

__all__ = ["FuzzyLogicSettings", "FuzzyLogicSpec"]


def _installed(family: str, noun: str, value: str) -> str:
    """Return a tag that an installed entry of one fuzzy family registers."""
    name = ensure_name(f"a {noun}", value)
    installed = Registry[object](family).available()

    if name not in installed:
        available = ", ".join(installed) or "(none)"
        msg = f"unknown {noun} {name!r}; installed {noun}s are {available}"
        raise ValueError(msg)

    return name


class FuzzyLogicSettings(NamedTuple):
    """Resolved interior settings for one registered fuzzy truth algebra."""

    tnorm: str
    implication: str
    quantifier: str
    equality: str
    margin: float


class FuzzyLogicSpec(dx.Model, extra="forbid"):
    """Declare every model-theoretic choice used by fuzzy evaluation."""

    tnorm: str
    implication: str
    quantifier: str
    equality: str
    margin: float

    @dx.validates("tnorm")
    @classmethod
    def _tnorm_is_named(cls, value: str) -> str:
        """Return a t-norm tag that an installed fuzzy-logic plugin registers."""
        return _installed("fuzzy_logic", "t-norm", value)

    @dx.validates("implication")
    @classmethod
    def _implication_is_named(cls, value: str) -> str:
        """Return an implication tag an installed plugin registers."""
        return _installed("fuzzy_implication", "implication", value)

    @dx.validates("quantifier")
    @classmethod
    def _quantifier_is_named(cls, value: str) -> str:
        """Return a quantifier tag an installed plugin registers."""
        return _installed("fuzzy_quantifier", "quantifier", value)

    @dx.validates("equality")
    @classmethod
    def _equality_is_named(cls, value: str) -> str:
        """Return an equality tag an installed plugin registers."""
        return _installed("fuzzy_equality", "equality", value)

    @dx.validates("margin")
    @classmethod
    def _margin_is_nonnegative(cls, value: float) -> float:
        """Return a finite, nonnegative equality margin."""
        held = ensure_finite("an equality margin", value)

        if held < 0.0:
            msg = f"an equality margin is nonnegative, not {value!r}"
            raise ValueError(msg)

        return held

    def settings(self) -> FuzzyLogicSettings:
        """Discard the boundary declaration into immutable interior settings."""
        return FuzzyLogicSettings(
            self.tnorm,
            self.implication,
            self.quantifier,
            self.equality,
            self.margin,
        )
