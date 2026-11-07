"""Built-in pricing cost-model plugins."""

from __future__ import annotations

from pathlib import Path
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final

from lofi._frozen import FrozenValue
from lofi.run._boundary.experiment import LengthBatchPricing, StackBootstrapPricing
from lofi.run._cost_model import PricingPlugin
from lofi.run._seed import ExperimentSeed
from lofi.spec import PACKAGED_CONFIG_ROOT, RunSpecCodec
from lofi.spec._interpolation import lower_package_resource

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.run._cost_model import CostModel, GridTimer
    from lofi.run._seed import PointSeed
    from lofi.run._values import GridTiming
    from lofi.spec import RunSpec

__all__ = [
    "LENGTH_BATCH_PLUGIN",
    "STACK_BOOTSTRAP_PLUGIN",
    "LengthBatchCostModel",
    "StackBootstrapCostModel",
    "reference_stacks",
]


@final
class StackBootstrapCostModel(FrozenValue):
    """Prices every stack off a random sample of its own timed steps.

    Parameters
    ----------
    resamples : int
        The bootstrap replicates the price's quantiles are read off.
    """

    _noun = "stack-bootstrap cost model"
    __slots__ = ("_resamples",)

    def __init__(self, resamples: int) -> None:
        object.__setattr__(self, "_resamples", resamples)

    @property
    def resamples(self) -> int:
        """Return the bootstrap replicates the price's quantiles are read off."""
        return self._resamples

    def reference(
        self, timer: GridTimer, /, *, device: str, warmup: int, repeats: int
    ) -> GridTiming | None:
        """Return no reference timings, since each stack is priced off its own."""
        del timer, device, warmup, repeats
        return None


@final
class LengthBatchCostModel(FrozenValue):
    """Prices a step by interpolation over sentence length and batch size.

    Parameters
    ----------
    resamples : int
        The bootstrap replicates the price's quantiles are read off.
    stacks : Mapping[tuple[int, int], RunSpec]
        The reference run document composed at each sentence length and batch
        size of the grid, lengths outermost.
    point : PointSeed
        The point every reference stack is planned for.
    """

    _noun = "length-batch cost model"
    __slots__ = ("_point", "_resamples", "_stacks")

    def __init__(
        self,
        resamples: int,
        stacks: Mapping[tuple[int, int], RunSpec],
        point: PointSeed,
    ) -> None:
        object.__setattr__(self, "_resamples", resamples)
        object.__setattr__(self, "_stacks", MappingProxyType(dict(stacks)))
        object.__setattr__(self, "_point", point)

    @property
    def resamples(self) -> int:
        """Return the bootstrap replicates the price's quantiles are read off."""
        return self._resamples

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the sentence lengths the grid spans."""
        return tuple(dict.fromkeys(length for length, _ in self._stacks))

    @property
    def batches(self) -> tuple[int, ...]:
        """Return the batch sizes the grid spans."""
        return tuple(dict.fromkeys(batch for _, batch in self._stacks))

    @property
    def stacks(self) -> Mapping[tuple[int, int], RunSpec]:
        """Return the reference document composed at each cell of the grid."""
        return self._stacks

    def reference(
        self, timer: GridTimer, /, *, device: str, warmup: int, repeats: int
    ) -> GridTiming | None:
        """Return the reference stack's step timed at every cell of the grid."""
        return timer.time_grid(
            self._stacks,
            self._point,
            device=device,
            warmup=warmup,
            repeats=repeats,
        )


def reference_stacks(
    declaration: LengthBatchPricing, /
) -> Mapping[tuple[int, int], RunSpec]:
    """Return the declared reference document composed at every cell of the grid.

    Parameters
    ----------
    declaration : LengthBatchPricing
        The validated pricing declaration.

    Raises
    ------
    ConfigError
        If the reference document, under some cell's overrides, does not
        validate as a run specification.
    """
    path = Path(lower_package_resource(declaration.reference))
    codec = RunSpecCodec(search_path=(PACKAGED_CONFIG_ROOT,))

    return MappingProxyType(
        {
            cell: codec(path, overrides=declaration.overrides(*cell))
            for cell in declaration.cells()
        }
    )


def _stack_bootstrap(declaration: StackBootstrapPricing, /) -> CostModel:
    """Build the stack-bootstrap cost model."""
    return StackBootstrapCostModel(declaration.resamples)


def _length_batch(declaration: LengthBatchPricing, /) -> CostModel:
    """Build the length-by-batch interpolation cost model."""
    reference = Path(lower_package_resource(declaration.reference))

    return LengthBatchCostModel(
        declaration.resamples,
        reference_stacks(declaration),
        ExperimentSeed(declaration.seed).point(reference.stem, 0),
    )


STACK_BOOTSTRAP_PLUGIN: Final = PricingPlugin(
    spec=StackBootstrapPricing, build=_stack_bootstrap
)
LENGTH_BATCH_PLUGIN: Final = PricingPlugin(spec=LengthBatchPricing, build=_length_batch)
