"""Shared declarations for action-controller variants."""

from typing import Final, Literal, NamedTuple, Protocol, cast

import didactic.api as dx

from lofi._checks import ensure_finite
from lofi.registry import Registry

__all__ = [
    "ControllerFeature",
    "ControllerInput",
    "ControllerNormalizerSettings",
    "ControllerNormalizerSpec",
    "ControllerRaising",
    "EntmaxControllerNormalizerSpec",
    "SoftmaxControllerNormalizerSpec",
]

type ControllerInput = Literal["inside", "span"]
"""The chart representation handed to a controller."""

type ControllerFeature = str
"""The installed ``lofi.pair_feature`` name of one child pair feature."""

type ControllerRaising = Literal["none", "soft"]
"""Whether the controller differentiably chooses type raising targets."""


class ControllerNormalizerSettings(NamedTuple):
    """Resolved action-normalization settings.

    Attributes
    ----------
    alpha : float
        Which entmax the scores are mapped through; one is the softmax.
    bisections : int
        How many halvings the entmax threshold search takes, zero for the softmax.
    """

    alpha: float
    bisections: int


class ControllerNormalizerSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """Declare how action scores become a distribution."""


class _DeclaredNormalizer(Protocol):
    """Expose the lowering every concrete normalizer declaration implements."""

    def settings(self) -> ControllerNormalizerSettings:
        """Return checked interior settings."""
        ...


class SoftmaxControllerNormalizerSpec(ControllerNormalizerSpec):
    """Normalize action scores with softmax."""

    kind: Literal["softmax"] = "softmax"

    def settings(self) -> ControllerNormalizerSettings:
        """Return softmax settings."""
        return ControllerNormalizerSettings(1.0, 0)


class EntmaxControllerNormalizerSpec(ControllerNormalizerSpec):
    """Normalize action scores with alpha-entmax."""

    kind: Literal["entmax"] = "entmax"
    alpha: float = 1.5
    bisections: int = 50

    __axioms__ = (
        dx.axiom(
            "bisections > 0",
            message="an entmax threshold search takes at least one bisection",
        ),
    )

    @dx.validates("alpha")
    @classmethod
    def _alpha_is_in_the_entmax_family(cls, value: float) -> float:
        """Return a finite alpha at least one."""
        alpha = ensure_finite("a controller entmax alpha", value)

        if alpha < 1.0:
            msg = f"a controller entmax alpha is at least one, not {value}"
            raise ValueError(msg)

        return alpha

    def settings(self) -> ControllerNormalizerSettings:
        """Return entmax settings."""
        return ControllerNormalizerSettings(self.alpha, self.bisections)


def _softmax() -> SoftmaxControllerNormalizerSpec:
    """Return the default action normalizer."""
    return SoftmaxControllerNormalizerSpec()


_DEFAULT_INPUT: Final[ControllerInput] = "inside"
"""The shared default chart representation of controller declarations."""

_DEFAULT_FEATURES: Final[tuple[ControllerFeature, ...]] = ("left", "right")
"""The shared default child pair features of controller declarations."""

_DEFAULT_NORMALIZER: Final = dx.field(default_factory=_softmax)
"""The shared default action normalizer of controller declarations."""


class _CommonDeclaration(Protocol):
    """Expose the settings shared by concrete controller declarations."""

    @property
    def input(self) -> ControllerInput:
        """Return which chart representation is read."""
        ...

    @property
    def features(self) -> tuple[ControllerFeature, ...]:
        """Return the ordered pair features supplied to the scoring head."""
        ...

    @property
    def normalizer(self) -> ControllerNormalizerSpec:
        """Return the declared map from scores to action weights."""
        ...

    def raising(self) -> ControllerRaising:
        """Return the declared type raising treatment."""
        ...


def checked_features(
    features: tuple[ControllerFeature, ...], /
) -> tuple[ControllerFeature, ...]:
    """Return a nonempty feature sequence without repeated contributions."""
    if not features:
        msg = "a controller needs at least one child-pair feature"
        raise ValueError(msg)

    if len(set(features)) != len(features):
        msg = f"a controller repeats child-pair features: {features!r}"
        raise ValueError(msg)

    installed = Registry[object]("pair_feature").available()
    unknown = tuple(feature for feature in features if feature not in installed)

    if unknown:
        available = ", ".join(installed) or "(none)"
        msg = (
            f"unknown child-pair features {unknown!r}; installed features are "
            f"{available}"
        )
        raise ValueError(msg)

    return features


def common_settings(
    declaration: _CommonDeclaration, /
) -> tuple[
    ControllerInput,
    tuple[ControllerFeature, ...],
    ControllerNormalizerSettings,
    ControllerRaising,
]:
    """Return settings shared by every controller runtime."""
    normalizer = cast("_DeclaredNormalizer", declaration.normalizer)
    return (
        declaration.input,
        declaration.features,
        normalizer.settings(),
        declaration.raising(),
    )
