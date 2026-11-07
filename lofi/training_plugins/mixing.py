"""Installed builders for declared training-task mixing policies."""

from typing import TYPE_CHECKING, Final, cast

from lofi._frozen import FrozenValue
from lofi.mixing import (
    Interleaving,
    MixingSpec,
    ProportionalMixingSpec,
    RatioMixingSpec,
    RoundRobinMixingSpec,
    TemperatureMixingSpec,
)
from lofi.registry import resolve_plugin

if TYPE_CHECKING:
    from collections.abc import Callable

__all__ = ["MixingPlugin", "mixing"]


class MixingPlugin[S: MixingSpec](FrozenValue):
    """Pair one mixing declaration with its runtime interleaving builder."""

    _noun = "mixing plugin"
    __slots__ = ("_build", "_spec")

    def __init__(self, *, spec: type[S], build: Callable[[S], Interleaving]) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[S]:
        """Return the exact declaration variant this plugin builds."""
        return self._spec

    def construct(self, declaration: S, /) -> Interleaving:
        """Build one immutable runtime interleaving."""
        return self._build(declaration)


def mixing(declaration: MixingSpec, /) -> Interleaving:
    """Build a declared mixing policy through its unique installed plugin."""
    plugin = cast(
        "MixingPlugin[MixingSpec]",
        resolve_plugin(
            "mixing", cast("str", declaration.kind), MixingPlugin, type(declaration)
        ),
    )

    return plugin.construct(declaration)


def _round_robin(declaration: RoundRobinMixingSpec, /) -> Interleaving:
    del declaration
    return Interleaving.round_robin()


def _proportional(declaration: ProportionalMixingSpec, /) -> Interleaving:
    del declaration
    return Interleaving.proportional()


def _ratio(declaration: RatioMixingSpec, /) -> Interleaving:
    return Interleaving.ratio(declaration.weights)


def _temperature(declaration: TemperatureMixingSpec, /) -> Interleaving:
    return Interleaving.temperature(declaration.temperature)


ROUND_ROBIN_PLUGIN: Final = MixingPlugin(
    spec=RoundRobinMixingSpec,
    build=_round_robin,
)
PROPORTIONAL_PLUGIN: Final = MixingPlugin(
    spec=ProportionalMixingSpec,
    build=_proportional,
)
RATIO_PLUGIN: Final = MixingPlugin(
    spec=RatioMixingSpec,
    build=_ratio,
)
TEMPERATURE_PLUGIN: Final = MixingPlugin(
    spec=TemperatureMixingSpec,
    build=_temperature,
)
