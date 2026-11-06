"""Declared pricing cost models resolved through installed plugins.

An experiment's ``pricing`` block is a declaration variant of the
``PricingSpec`` union and an installed plugin of the ``lofi.pricing``
entry point group. The plugin builds the cost model a price is read under:
how many bootstrap replicates its quantiles are read off and which reference
timings, if any, the stacks' own timings are interpolated over.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.run._seed import PointSeed
    from lofi.run._values import GridTiming
    from lofi.spec import RunSpec

__all__ = [
    "CostModel",
    "GridTimer",
    "PricingPlugin",
    "cost_model",
]


class GridTimer(Protocol):
    """Times a reference stack's step over a length-by-batch grid."""

    def time_grid(
        self,
        stacks: Mapping[tuple[int, int], RunSpec],
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> GridTiming:
        """Return the timed grid."""
        ...


class CostModel(Protocol):
    """The cost model an experiment's training steps are priced under."""

    @property
    def resamples(self) -> int:
        """Return the bootstrap replicates the price's quantiles are read off."""
        ...

    def reference(
        self, timer: GridTimer, /, *, device: str, warmup: int, repeats: int
    ) -> GridTiming | None:
        """Return the reference timings stacks are priced over, or None.

        Parameters
        ----------
        timer : GridTimer
            What times the reference stack.
        device : str
            The device the price is timed on.
        warmup : int
            The untimed steps run before the timed ones.
        repeats : int
            The timed steps.
        """
        ...


class _PricingDeclaration(Protocol):
    """A validated declaration selecting an installed cost model."""

    @property
    def kind(self) -> object:
        """Return the installed cost-model tag."""
        ...


class PricingPlugin[S: _PricingDeclaration](FrozenValue):
    """Build one cost model from its exact declaration variant."""

    _noun = "pricing plugin"
    __slots__ = ("_build", "_spec")

    def __init__(self, *, spec: type[S], build: Callable[[S], CostModel]) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin consumes."""
        return self._spec

    def construct(self, declaration: S, /) -> CostModel:
        """Build the declared cost model."""
        return self._build(declaration)


def cost_model(declaration: _PricingDeclaration, /) -> CostModel:
    """Build one declared cost model through its unique installed plugin.

    Raises
    ------
    RegistryError
        If no plugin or several are installed under the declaration's tag, or
        the entry names no pricing plugin or one for another declaration.
    """
    plugin = cast(
        "PricingPlugin[_PricingDeclaration]",
        resolve_plugin(
            "pricing", cast("str", declaration.kind), PricingPlugin, type(declaration)
        ),
    )

    return plugin.construct(declaration)
