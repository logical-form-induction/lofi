"""Open declarations for premise--hypothesis pairing strategies."""

from typing import Literal, Protocol, cast

import didactic.api as dx

from lofi._frozen import FrozenValue
from lofi.data.synthetic._pairing import (
    IndependentPairing,
    OverlapMatchedPairing,
    PairingStrategy,
)
from lofi.registry import resolve_plugin

__all__ = [
    "IndependentPairingSpec",
    "OverlapMatchedPairingSpec",
    "PairingPlugin",
    "PairingSpec",
    "pairing_plugin",
]


class PairingSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """An installed strategy for drawing premise--hypothesis candidates."""

    def build(self) -> PairingStrategy:
        """Build the selected installed pairing strategy."""
        return pairing_plugin(self).construct(self)


class IndependentPairingSpec(PairingSpec, extra="forbid"):
    """Draw the two sides independently with replacement."""

    kind: Literal["independent"] = "independent"


class OverlapMatchedPairingSpec(PairingSpec, extra="forbid"):
    """Draw among candidate pairs with maximum predicate overlap."""

    kind: Literal["overlap-matched"] = "overlap-matched"


class PairingBuilder[Declaration: PairingSpec](Protocol):
    """Build a pairing strategy from its validated declaration."""

    def __call__(self, declaration: Declaration, /) -> PairingStrategy:
        """Build the strategy."""
        ...


class PairingPlugin[Declaration: PairingSpec](FrozenValue):
    """Pair one open pairing declaration with its runtime builder."""

    _noun = "pairing-strategy plugin"
    __slots__ = ("_build", "_spec")

    def __init__(
        self, *, spec: type[Declaration], build: PairingBuilder[Declaration]
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(self, declaration: Declaration, /) -> PairingStrategy:
        """Build one strategy from a validated declaration."""
        return self._build(declaration)


def pairing_plugin(declaration: PairingSpec, /) -> PairingPlugin[PairingSpec]:
    """Return the installed pairing plugin selected by a declaration."""
    return cast(
        "PairingPlugin[PairingSpec]",
        resolve_plugin(
            "pair_strategy",
            cast("str", declaration.kind),
            PairingPlugin,
            type(declaration),
        ),
    )


def _independent(_: IndependentPairingSpec, /) -> IndependentPairing:
    """Build independent pairing."""
    return IndependentPairing()


def _overlap_matched(_: OverlapMatchedPairingSpec, /) -> OverlapMatchedPairing:
    """Build overlap-matched pairing."""
    return OverlapMatchedPairing()


INDEPENDENT_PLUGIN = PairingPlugin(
    spec=IndependentPairingSpec,
    build=_independent,
)
OVERLAP_MATCHED_PLUGIN = PairingPlugin(
    spec=OverlapMatchedPairingSpec,
    build=_overlap_matched,
)
