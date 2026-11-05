"""Run-level report values: evaluation rows, and what an experiment costs to train."""

from enum import StrEnum
from importlib import import_module
from typing import TYPE_CHECKING, Final, NamedTuple, cast, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    import numpy as np
    from numpy.typing import NDArray

    from lofi.data import BatchShape
    from lofi.training_plugins import TrainingLength


__all__ = [
    "COST_MODEL",
    "GRID_COST_MODEL",
    "HOUR_DECIMALS",
    "SECONDS_PER_HOUR",
    "ExperimentPrice",
    "GridCell",
    "GridTiming",
    "HardwareDescriptor",
    "MemoryBudget",
    "RunPhase",
    "StackPrice",
    "StackTiming",
]

SECONDS_PER_HOUR: Final = 3600
"""How many seconds an hour holds."""

HOUR_DECIMALS: Final = 6
"""The decimal places an experiment's accelerator-hours are reported and declared to."""

COST_MODEL: Final = "experiment-total-mean-bootstrap"
"""The cost model that reads each stack off its own timed steps.

Each distinct stack's training step is timed at randomly sampled steps, a point
costs its declared steps at the mean timed step, and the experiment's P50 and P90
are quantiles of the summed hours over bootstrap resamples of every stack's
timings, the stacks summed inside each replicate.
"""

GRID_COST_MODEL: Final = "length-batch-interpolation"
"""The cost model that interpolates a step over length and batch.

A reference run document's training step is timed at every cell of a grid of
sentence lengths and batch sizes, a point costs its declared steps at the step
seconds interpolated bilinearly at its stack's mean sentence length and batch
size, and the experiment's P50 and P90 are quantiles of the summed hours over
bootstrap resamples of every cell's timings.
"""

type _NumpyArray = NDArray[np.float64]
"""A one-dimensional array of bootstrapped seconds, one per replicate."""


class MemoryBudget:
    """A ceiling on the bytes one training step holds at its peak.

    Parameters
    ----------
    max_bytes : int
        The most bytes a step may hold at its peak, a positive count the
        device policy declaring it has validated.
    """

    __slots__ = ("_max_bytes",)

    def __init__(self, max_bytes: int) -> None:
        self._max_bytes = max_bytes

    @property
    def max_bytes(self) -> int:
        """Return the ceiling in bytes."""
        return self._max_bytes

    def exceeded_by(self, peak: int | None, /) -> bool:
        """Return whether a measured peak lies above the ceiling.

        Parameters
        ----------
        peak : int | None
            The measured peak, or None where no allocator was read.
        """
        return peak is not None and peak > self._max_bytes

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the ceilings."""
        if type(other) is not MemoryBudget:
            return NotImplemented

        return self._max_bytes == other._max_bytes

    @override
    def __hash__(self) -> int:
        """Hash the ceiling."""
        return hash((MemoryBudget, self._max_bytes))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"MemoryBudget({self._max_bytes!r})"


def _numpy() -> ModuleType:
    """Return NumPy at the call site that bootstraps an experiment's hours.

    Raises
    ------
    RuntimeError
        If NumPy is not installed.
    """
    try:
        return import_module("numpy")

    except ModuleNotFoundError as error:
        msg = "bootstrapping an experiment price requires the lofi[nn] capability"
        raise RuntimeError(msg) from error


class RunPhase(StrEnum):
    """A phase of executing one training point, which its wall time is kept by.

    Preparation draws the corpus and builds the model, recovery restores the
    latest resumable checkpoint, fitting trains what remains, and scoring runs
    the held-out evaluators the run declares.
    """

    PREPARATION = "preparation"
    RECOVERY = "recovery"
    FITTING = "fitting"
    SCORING = "scoring"


@final
class StackTiming(FrozenValue):
    """What repeated training steps of one built stack took.

    Parameters
    ----------
    steps_per_unit : int
        The training steps one unit of the declared training length holds:
        the batches of one epoch, or one for a length declared in steps.
    positions : Sequence[int]
        The position of each timed step in the stack's training, drawn at
        random: the batch of the epoch it read, or the step of the horizon.
    seconds : Sequence[float]
        The wall seconds of each timed step, in the order they ran.
    peak_allocated_bytes : int | None
        The most bytes the accelerator allocator held across the timed steps,
        or None on a device with no allocator to read.
    shape : BatchShape | None
        The mean sentence length and batch size of the stack's training
        batches, or None for a stack whose steps read no sentences.

    Raises
    ------
    ValueError
        If a unit holds no step, nothing was timed, a timed step took no time
        or a time that is not finite, a position is negative or the positions
        are not one per timed step, the peak is negative, or a shape is not
        positive and finite.
    """

    _noun = "stack timing"

    __slots__ = (
        "_peak_allocated_bytes",
        "_positions",
        "_seconds",
        "_shape",
        "_steps_per_unit",
    )

    def __init__(
        self,
        *,
        steps_per_unit: int,
        positions: Sequence[int],
        seconds: Sequence[float],
        peak_allocated_bytes: int | None,
        shape: BatchShape | None,
    ) -> None:
        per_unit = ensure_non_negative("a step count per unit", steps_per_unit)
        timed = tuple(ensure_finite("a timed step", second) for second in seconds)
        sampled = tuple(
            ensure_non_negative("a sampled position", position)
            for position in positions
        )

        if per_unit == 0:
            msg = "a unit of training length holds at least one step"

            raise ValueError(msg)

        if not timed:
            msg = "a timing holds at least one timed step"

            raise ValueError(msg)

        if min(timed) <= 0:
            msg = f"a timed step takes time: {timed!r}"

            raise ValueError(msg)

        if len(sampled) != len(timed):
            msg = (
                f"a timing holds one sampled position per timed step, not "
                f"{len(sampled)} for {len(timed)}"
            )

            raise ValueError(msg)

        if (
            shape is not None
            and min(
                ensure_finite("a mean sentence length", shape.sentence_length),
                ensure_finite("a mean batch size", shape.batch_size),
            )
            <= 0
        ):
            msg = f"a batch shape is positive: {shape!r}"

            raise ValueError(msg)

        object.__setattr__(self, "_shape", shape)
        object.__setattr__(self, "_steps_per_unit", per_unit)
        object.__setattr__(self, "_positions", sampled)
        object.__setattr__(self, "_seconds", timed)
        object.__setattr__(
            self,
            "_peak_allocated_bytes",
            None
            if peak_allocated_bytes is None
            else ensure_non_negative("a peak allocation", peak_allocated_bytes),
        )

    @property
    def steps_per_unit(self) -> int:
        """Return the training steps one unit of training length holds."""
        return self._steps_per_unit

    @property
    def positions(self) -> tuple[int, ...]:
        """Return the sampled position of each timed step, in the order they ran."""
        return self._positions

    @property
    def seconds(self) -> tuple[float, ...]:
        """Return the wall seconds of each timed step, in the order they ran."""
        return self._seconds

    @property
    def peak_allocated_bytes(self) -> int | None:
        """Return the peak the allocator held, or None where none was read."""
        return self._peak_allocated_bytes

    @property
    def shape(self) -> BatchShape | None:
        """Return the mean sentence length and batch size of the training batches."""
        return self._shape

    def _fields(
        self,
    ) -> tuple[int, tuple[int, ...], tuple[float, ...], int | None, BatchShape | None]:
        """Return every field, in declaration order."""
        return (
            self._steps_per_unit,
            self._positions,
            self._seconds,
            self._peak_allocated_bytes,
            self._shape,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if type(other) is not StackTiming:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash every field."""
        return hash((StackTiming, *self._fields()))

    @override
    def __repr__(self) -> str:
        """Return every field of the timing."""
        return (
            f"StackTiming(steps_per_unit={self._steps_per_unit!r}, "
            f"positions={self._positions!r}, "
            f"seconds={self._seconds!r}, "
            f"peak_allocated_bytes={self._peak_allocated_bytes!r}, "
            f"shape={self._shape!r})"
        )


@final
class StackPrice(FrozenValue):
    """What the points sharing one built stack cost to train.

    Every point of a stack composes to the same run document and differs only
    in its seed, so each trains for the same number of steps at the same rate.

    Parameters
    ----------
    stack : str
        The identifier of the stack's resolved run document.
    points : Sequence[str]
        The identifiers of the points that train the stack, in sweep order.
    length : TrainingLength
        The training length the stack's run document declares.
    timing : StackTiming
        What repeated training steps of the stack took.
    budget : MemoryBudget
        The activation-memory ceiling the stack's device policy declares, which
        its peak is read against.

    Raises
    ------
    ValueError
        If the stack is unnamed, no point or a repeated point trains it, or
        the declared length is not a positive count.
    """

    _noun = "stack price"

    __slots__ = ("_budget", "_length", "_points", "_stack", "_timing")

    def __init__(
        self,
        *,
        stack: str,
        points: Sequence[str],
        length: TrainingLength,
        timing: StackTiming,
        budget: MemoryBudget,
    ) -> None:
        held = tuple(ensure_name("a priced point", point) for point in points)

        if not held:
            msg = "a priced stack is trained by at least one point"

            raise ValueError(msg)

        if len(set(held)) != len(held):
            msg = f"a priced stack lists each point once: {held!r}"

            raise ValueError(msg)

        if ensure_non_negative("a training length", length.count) == 0:
            msg = "a declared training length is at least one unit"

            raise ValueError(msg)

        object.__setattr__(self, "_stack", ensure_name("a priced stack", stack))
        object.__setattr__(self, "_points", held)
        object.__setattr__(self, "_length", length)
        object.__setattr__(self, "_timing", timing)
        object.__setattr__(self, "_budget", budget)

    @property
    def stack(self) -> str:
        """Return the identifier of the stack's resolved run document."""
        return self._stack

    @property
    def points(self) -> tuple[str, ...]:
        """Return the points that train the stack, in sweep order."""
        return self._points

    @property
    def length(self) -> TrainingLength:
        """Return the training length the stack's run document declares."""
        return self._length

    @property
    def timing(self) -> StackTiming:
        """Return what repeated training steps of the stack took."""
        return self._timing

    @property
    def budget(self) -> MemoryBudget:
        """Return the activation-memory ceiling the stack's peak is read against."""
        return self._budget

    @property
    def over_budget(self) -> bool:
        """Return whether the stack's measured peak lies above its ceiling."""
        return self._budget.exceeded_by(self._timing.peak_allocated_bytes)

    @property
    def seconds(self) -> tuple[float, ...]:
        """Return the wall seconds of each timed step."""
        return self._timing.seconds

    @property
    def peak_allocated_bytes(self) -> int | None:
        """Return the peak the allocator held, or None where none was read."""
        return self._timing.peak_allocated_bytes

    @property
    def memory_bytes(self) -> int:
        """Return the bytes one step of the stack is known to need at most.

        A measured peak is the figure. A stack timed where no allocator was
        read is bounded by the activation budget its device policy declares.
        """
        peak = self._timing.peak_allocated_bytes

        return self._budget.max_bytes if peak is None else peak

    @property
    def steps(self) -> int:
        """Return the training steps one point takes over its declared length."""
        return self._length.count * self._timing.steps_per_unit

    @property
    def mean_seconds(self) -> float:
        """Return the mean seconds of one timed step."""
        return sum(self._timing.seconds) / len(self._timing.seconds)

    @property
    def mean_hours(self) -> float:
        """Return the accelerator-hours every point costs at the mean timed step."""
        return len(self._points) * self.steps * self.mean_seconds / SECONDS_PER_HOUR

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if type(other) is not StackPrice:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash every field."""
        return hash((StackPrice, self._fields()))

    def _fields(
        self,
    ) -> tuple[str, tuple[str, ...], TrainingLength, StackTiming, MemoryBudget]:
        """Return every field, in declaration order."""
        return (self._stack, self._points, self._length, self._timing, self._budget)

    @override
    def __repr__(self) -> str:
        """Return the stack, its points and its steps."""
        return (
            f"StackPrice({self._stack!r}, points={len(self._points)!r}, "
            f"steps={self.steps!r})"
        )


@final
class HardwareDescriptor(FrozenValue):
    """The machine an experiment's training steps were timed on.

    Parameters
    ----------
    device_name : str
        The name of the device the steps ran on: the accelerator's product
        name, or the device kind where the framework reports no name.
    driver : str | None
        The accelerator driver version, or None where none was read.
    cuda : str | None
        The CUDA runtime the framework was built against, or None for a build
        without one.
    torch : str
        The framework version.
    host_cpu : str
        The host processor, as the platform names it.

    Raises
    ------
    ValueError
        If a name or version is empty.
    """

    _noun = "hardware descriptor"

    __slots__ = ("_cuda", "_device_name", "_driver", "_host_cpu", "_torch")

    def __init__(
        self,
        *,
        device_name: str,
        driver: str | None,
        cuda: str | None,
        torch: str,
        host_cpu: str,
    ) -> None:
        object.__setattr__(
            self, "_device_name", ensure_name("a device name", device_name)
        )
        object.__setattr__(
            self,
            "_driver",
            None if driver is None else ensure_name("a driver version", driver),
        )
        object.__setattr__(
            self, "_cuda", None if cuda is None else ensure_name("a CUDA version", cuda)
        )
        object.__setattr__(self, "_torch", ensure_name("a torch version", torch))
        object.__setattr__(self, "_host_cpu", ensure_name("a host processor", host_cpu))

    @property
    def device_name(self) -> str:
        """Return the name of the device the steps ran on."""
        return self._device_name

    @property
    def driver(self) -> str | None:
        """Return the accelerator driver version, or None where none was read."""
        return self._driver

    @property
    def cuda(self) -> str | None:
        """Return the CUDA runtime version, or None for a build without one."""
        return self._cuda

    @property
    def torch(self) -> str:
        """Return the framework version."""
        return self._torch

    @property
    def host_cpu(self) -> str:
        """Return the host processor."""
        return self._host_cpu

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if type(other) is not HardwareDescriptor:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash every field."""
        return hash((HardwareDescriptor, self._fields()))

    def _fields(self) -> tuple[str, str | None, str | None, str, str]:
        """Return every field, in declaration order."""
        return (
            self._device_name,
            self._driver,
            self._cuda,
            self._torch,
            self._host_cpu,
        )

    @override
    def __repr__(self) -> str:
        """Return the device and framework the descriptor names."""
        return (
            f"HardwareDescriptor(device_name={self._device_name!r}, "
            f"torch={self._torch!r})"
        )


class GridCell(NamedTuple):
    """The timed steps of the reference stack at one sentence length and batch size."""

    sentence_length: int
    batch_size: int
    seconds: tuple[float, ...]


def _segment(axis: tuple[int, ...], value: float, /) -> tuple[int, float]:
    """Return the grid segment a value interpolates on, and its fraction along it.

    A value outside the axis extrapolates along the nearest end segment, so its
    fraction is negative below the first line and above one past the last.
    """
    upper = next(
        (index for index, line in enumerate(axis) if line >= value), len(axis) - 1
    )
    lower = min(max(upper - 1, 0), len(axis) - 2)

    return lower, (value - axis[lower]) / (axis[lower + 1] - axis[lower])


@final
class GridTiming(FrozenValue):
    """Timed steps of the reference stack over a grid of lengths and batch sizes.

    Parameters
    ----------
    cells : Sequence[GridCell]
        One cell per sentence length and batch size, lengths outermost, each
        axis ascending.

    Raises
    ------
    ValueError
        If either axis holds fewer than two distinct ascending positive
        values, the cells do not cover the grid in order, a cell holds a timed
        step count other than the first cell's, or a timed step takes no time
        or a time that is not finite.
    """

    _noun = "grid timing"

    __slots__ = ("_batches", "_cells", "_lengths")

    def __init__(self, cells: Sequence[GridCell]) -> None:
        held = tuple(
            GridCell(
                cell.sentence_length,
                cell.batch_size,
                tuple(ensure_finite("a timed step", second) for second in cell.seconds),
            )
            for cell in cells
        )
        lengths = tuple(dict.fromkeys(cell.sentence_length for cell in held))
        batches = tuple(dict.fromkeys(cell.batch_size for cell in held))

        for name, axis in (("sentence lengths", lengths), ("batch sizes", batches)):
            if len(axis) < 2 or axis != tuple(sorted(axis)) or axis[0] <= 0:
                msg = f"a grid spans two or more ascending positive {name}: {axis!r}"

                raise ValueError(msg)

        expected = [(length, batch) for length in lengths for batch in batches]
        found = [(cell.sentence_length, cell.batch_size) for cell in held]

        if found != expected:
            msg = f"the grid cells are not every length by every batch: {found!r}"

            raise ValueError(msg)

        counts = {len(cell.seconds) for cell in held}

        if len(counts) != 1 or 0 in counts:
            msg = f"every grid cell holds one equal, positive count of steps: {counts}"

            raise ValueError(msg)

        if min(second for cell in held for second in cell.seconds) <= 0:
            msg = "a timed grid step takes time"

            raise ValueError(msg)

        object.__setattr__(self, "_cells", held)
        object.__setattr__(self, "_lengths", lengths)
        object.__setattr__(self, "_batches", batches)

    @property
    def cells(self) -> tuple[GridCell, ...]:
        """Return the cells, lengths outermost."""
        return self._cells

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the sentence lengths, ascending."""
        return self._lengths

    @property
    def batches(self) -> tuple[int, ...]:
        """Return the batch sizes, ascending."""
        return self._batches

    @property
    def repeats(self) -> int:
        """Return the timed steps each cell holds."""
        return len(self._cells[0].seconds)

    def weights(self, shape: BatchShape, /) -> tuple[float, ...]:
        """Return each cell's coefficient in the step seconds at a batch shape.

        The step seconds at a shape are the cells' step seconds weighted by
        these coefficients: bilinear interpolation inside the grid, and linear
        extrapolation along the nearest end segment outside it.

        Parameters
        ----------
        shape : BatchShape
            The mean sentence length and batch size interpolated at.

        Returns
        -------
        tuple[float, ...]
            One coefficient per cell, in cell order, summing to one.
        """
        row, across = _segment(self._lengths, shape.sentence_length)
        column, down = _segment(self._batches, shape.batch_size)
        corners = {
            (row, column): (1 - across) * (1 - down),
            (row, column + 1): (1 - across) * down,
            (row + 1, column): across * (1 - down),
            (row + 1, column + 1): across * down,
        }
        width = len(self._batches)

        return tuple(
            corners.get(divmod(index, width), 0.0) for index in range(len(self._cells))
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the cells."""
        if type(other) is not GridTiming:
            return NotImplemented

        return self._cells == other._cells

    @override
    def __hash__(self) -> int:
        """Hash the cells."""
        return hash((GridTiming, self._cells))

    @override
    def __repr__(self) -> str:
        """Return the grid's axes."""
        return f"GridTiming(lengths={self._lengths!r}, batches={self._batches!r})"


@final
class ExperimentPrice(FrozenValue):
    """What an experiment costs to train, read off timed training steps.

    Each point of a stack takes its declared steps, so the experiment's total is
    the sum over stacks of points times steps times the seconds of one step. The
    P50 and P90 are quantiles of that total under a seeded bootstrap, drawn from
    one NumPy generator seeded by the recorded seed, and the quantiles
    interpolate linearly between order statistics, so the figures are recomputed
    from the record alone.

    Without a grid, a stack's step seconds are read off its own timed steps,
    which are a random sample of the steps it trains: each replicate draws every
    stack in the recorded order as an array of resamples by repeats indices into
    its timed seconds and takes each resample's mean. With a grid, a stack's
    step seconds are the grid's step seconds interpolated at the stack's batch
    shape: each replicate draws every cell in order the same way, and each
    stack's seconds are the cell means weighted by the stack's coefficients.

    Parameters
    ----------
    experiment : str
        The name of the priced experiment.
    freeze_id : str
        The freeze the experiment runs against.
    sweep_id : str
        The identifier of the experiment's expanded points, in sweep order.
    device : str
        The device every stack was timed on, index included.
    hardware : HardwareDescriptor
        The machine the device belongs to.
    warmup : int
        The untimed steps each stack ran before its timed ones.
    repeats : int
        The timed steps each stack and each grid cell ran.
    resamples : int
        The bootstrap replicates the quantiles are read off.
    bootstrap_seed : int
        The seed of the generator the replicates are drawn from.
    stacks : Sequence[StackPrice]
        One price per distinct stack, in the order the sweep first reaches it.
    grid : GridTiming | None
        The reference stack's timed steps over lengths and batch sizes, or None to
        read every stack off its own timed steps.

    Raises
    ------
    ValueError
        If a name is empty, the warmup or the bootstrap seed is negative, no
        resample is drawn, no stack is priced, a stack or grid cell holds a
        timing count other than the repeats, two stacks share an identifier,
        a point trains two stacks, or a grid prices a stack with no batch shape.
    """

    _noun = "experiment price"

    __slots__ = (
        "_bootstrap_seed",
        "_device",
        "_experiment",
        "_freeze_id",
        "_grid",
        "_hardware",
        "_repeats",
        "_resamples",
        "_stacks",
        "_sweep_id",
        "_warmup",
    )

    def __init__(
        self,
        *,
        experiment: str,
        freeze_id: str,
        sweep_id: str,
        device: str,
        hardware: HardwareDescriptor,
        warmup: int,
        repeats: int,
        resamples: int,
        bootstrap_seed: int,
        stacks: Sequence[StackPrice],
        grid: GridTiming | None,
    ) -> None:
        held = tuple(stacks)

        if not held:
            msg = "an experiment price holds at least one stack"

            raise ValueError(msg)

        if ensure_non_negative("a resample count", resamples) == 0:
            msg = "a bootstrap draws at least one resample"

            raise ValueError(msg)

        for stack in held:
            if len(stack.seconds) != repeats:
                msg = (
                    f"the stack {stack.stack!r} holds {len(stack.seconds)} timed "
                    f"steps, not the {repeats} the price declares"
                )

                raise ValueError(msg)

        if grid is not None:
            if grid.repeats != repeats:
                msg = (
                    f"the grid holds {grid.repeats} timed steps per cell, not the "
                    f"{repeats} the price declares"
                )

                raise ValueError(msg)

            shapeless = [stack.stack for stack in held if stack.timing.shape is None]

            if shapeless:
                msg = (
                    f"the stacks {shapeless!r} read no sentences, so a grid over "
                    f"sentence length cannot price them"
                )

                raise ValueError(msg)

        identifiers = [stack.stack for stack in held]

        if len(set(identifiers)) != len(identifiers):
            msg = f"an experiment price lists each stack once: {identifiers!r}"

            raise ValueError(msg)

        points = [point for stack in held for point in stack.points]

        if len(set(points)) != len(points):
            msg = "a point of an experiment trains exactly one stack"

            raise ValueError(msg)

        object.__setattr__(
            self, "_experiment", ensure_name("an experiment", experiment)
        )
        object.__setattr__(self, "_freeze_id", ensure_name("a freeze", freeze_id))
        object.__setattr__(self, "_sweep_id", ensure_name("a sweep", sweep_id))
        object.__setattr__(self, "_device", ensure_name("a priced device", device))
        object.__setattr__(self, "_hardware", hardware)
        object.__setattr__(
            self, "_warmup", ensure_non_negative("a warmup count", warmup)
        )
        object.__setattr__(self, "_repeats", repeats)
        object.__setattr__(self, "_resamples", resamples)
        object.__setattr__(
            self,
            "_bootstrap_seed",
            ensure_non_negative("a bootstrap seed", bootstrap_seed),
        )
        object.__setattr__(self, "_stacks", held)
        object.__setattr__(self, "_grid", grid)

    @property
    def experiment(self) -> str:
        """Return the name of the priced experiment."""
        return self._experiment

    @property
    def freeze_id(self) -> str:
        """Return the freeze the experiment runs against."""
        return self._freeze_id

    @property
    def sweep_id(self) -> str:
        """Return the identifier of the experiment's expanded points."""
        return self._sweep_id

    @property
    def cost_model(self) -> str:
        """Return the cost model that produced the figures."""
        return COST_MODEL if self._grid is None else GRID_COST_MODEL

    @property
    def device(self) -> str:
        """Return the device every stack was timed on."""
        return self._device

    @property
    def hardware(self) -> HardwareDescriptor:
        """Return the machine the device belongs to."""
        return self._hardware

    @property
    def warmup(self) -> int:
        """Return the untimed steps each stack ran before its timed ones."""
        return self._warmup

    @property
    def repeats(self) -> int:
        """Return the timed steps each stack ran."""
        return self._repeats

    @property
    def resamples(self) -> int:
        """Return the bootstrap replicates the quantiles are read off."""
        return self._resamples

    @property
    def bootstrap_seed(self) -> int:
        """Return the seed of the generator the replicates are drawn from."""
        return self._bootstrap_seed

    @property
    def stacks(self) -> tuple[StackPrice, ...]:
        """Return one price per distinct stack, in sweep order."""
        return self._stacks

    @property
    def grid(self) -> GridTiming | None:
        """Return the grid the stacks are interpolated over, or None."""
        return self._grid

    @property
    def points(self) -> int:
        """Return how many points the experiment trains."""
        return sum(len(stack.points) for stack in self._stacks)

    def _step_seconds(self) -> tuple[_NumpyArray, ...]:
        """Return each stack's bootstrapped seconds per step, one per replicate."""
        np = _numpy()
        generator = np.random.default_rng(self._bootstrap_seed)

        def means(seconds: tuple[float, ...]) -> _NumpyArray:
            timed = np.asarray(seconds)
            draws = generator.integers(
                0, timed.size, size=(self._resamples, timed.size)
            )

            return timed[draws].mean(axis=1)

        if self._grid is None:
            return tuple(means(stack.seconds) for stack in self._stacks)

        grid = self._grid
        cells = np.stack([means(cell.seconds) for cell in grid.cells], axis=1)

        return tuple(
            cells @ np.asarray(grid.weights(cast("BatchShape", stack.timing.shape)))
            for stack in self._stacks
        )

    def _quantile_hours(self, quantile: float, /) -> float:
        """Return one quantile of the experiment's bootstrapped total hours.

        Parameters
        ----------
        quantile : float
            The quantile, between zero and one.
        """
        np = _numpy()
        totals = np.zeros(self._resamples)

        for stack, seconds in zip(self._stacks, self._step_seconds(), strict=True):
            totals += (len(stack.points) * stack.steps * seconds) / SECONDS_PER_HOUR

        return float(np.quantile(totals, quantile))

    @property
    def p50_hours(self) -> float:
        """Return the median of the experiment's bootstrapped total hours."""
        return self._quantile_hours(0.5)

    @property
    def p90_hours(self) -> float:
        """Return the ninetieth percentile of the experiment's bootstrapped hours."""
        return self._quantile_hours(0.9)

    def point_seconds(self, quantile: float, /) -> dict[str, float]:
        """Return one quantile of each point's bootstrapped training seconds.

        Parameters
        ----------
        quantile : float
            The quantile, between zero and one.

        Returns
        -------
        dict[str, float]
            By point identifier, the quantile of its declared steps times its
            stack's bootstrapped seconds per step.
        """
        np = _numpy()

        return {
            point: float(np.quantile(stack.steps * seconds, quantile))
            for stack, seconds in zip(self._stacks, self._step_seconds(), strict=True)
            for point in stack.points
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every field."""
        if type(other) is not ExperimentPrice:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Hash every field."""
        return hash((ExperimentPrice, self._fields()))

    def _fields(
        self,
    ) -> tuple[
        str,
        str,
        str,
        str,
        HardwareDescriptor,
        int,
        int,
        int,
        int,
        tuple[StackPrice, ...],
        GridTiming | None,
    ]:
        """Return every field, in declaration order."""
        return (
            self._experiment,
            self._freeze_id,
            self._sweep_id,
            self._device,
            self._hardware,
            self._warmup,
            self._repeats,
            self._resamples,
            self._bootstrap_seed,
            self._stacks,
            self._grid,
        )

    @override
    def __repr__(self) -> str:
        """Return the experiment and its two figures."""
        return (
            f"ExperimentPrice({self._experiment!r}, p50_hours={self.p50_hours!r}, "
            f"p90_hours={self.p90_hours!r})"
        )
