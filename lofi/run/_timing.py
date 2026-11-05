"""Times the real training step of the stack one run document declares.

The stack is planned, seeded and prepared exactly as a training run prepares
it, then placed on the requested device under the precision the run document
declares. Its fitting plan supplies the step and the batches the step reads.
The steps run are a random sample, with replacement and seeded by the root
seed, of the steps the stack trains: a parser run reads batches drawn from
across one epoch, and a distance run reads the draws of steps drawn from
across its horizon. The timing records the position each timed step read,
so the sample is part of the record. Each timed step closes on a device
synchronization, so a clock read covers the work the step queued.

A grid is timed on a reference stack instead, declared once per cell by its
own run document and prepared on the timed device: one untimed step at every
cell, then the timed steps taken one cell at a time round the whole grid, so a
busy stretch of the machine lands on every cell rather than on one. Each cell
steps over the first batch of its epoch, which holds exactly the cell's
sentence length and batch size.
"""

import platform
from pathlib import Path
from random import Random
from time import perf_counter
from typing import TYPE_CHECKING, final, override

import torch

from lofi.data import BatchShape
from lofi.nn.policy import DeviceKind, DevicePolicy
from lofi.run._device import device_scope, resolve_device_policy
from lofi.run._seed import RandomStreams, SeedPolicy
from lofi.run._training import plan_training
from lofi.run._values import (
    GridCell,
    GridTiming,
    HardwareDescriptor,
    StackTiming,
)
from lofi.train import prepare_training

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.run._seed import PointSeed
    from lofi.spec import RunSpec
    from lofi.train import TimedStack

__all__ = [
    "ReferenceShapeError",
    "ReferenceStep",
    "TrainingStepTimer",
    "prepare_reference",
]


class ReferenceShapeError(ValueError):
    """A reference stack whose first batch is not the grid cell it is timed at."""


def _device(device: str, /) -> torch.device:
    """Return the device a name spells.

    Raises
    ------
    ValueError
        If the name spells no device, or a meta device, on which nothing runs.
    """
    try:
        requested = torch.device(device)

    except RuntimeError as unknown:
        msg = f"{device!r} names no device"

        raise ValueError(msg) from unknown

    if requested.type == "meta":
        msg = "a meta device computes nothing, so a step on it has no price"

        raise ValueError(msg)

    return requested


@final
class TrainingStepTimer:
    """Builds the stack a run document declares and times its training step."""

    __slots__ = ()

    def time(
        self,
        spec: RunSpec,
        point: PointSeed,
        /,
        *,
        device: str,
        warmup: int,
        repeats: int,
    ) -> StackTiming:
        """Build a run's stack, run its step untimed, then time sampled steps.

        The stack is planned for the point, so a data seed the run derives is
        drawn as the point draws it. The warmup and timed steps run at
        positions drawn uniformly, with replacement, from the stack's training,
        by a generator seeded with the point's run seed.

        Parameters
        ----------
        spec : RunSpec
            The validated run document whose stack is built.
        point : PointSeed
            The point the stack is planned for, whose run seed it is built
            under.
        device : str
            The device the stack is built and timed on, such as ``cpu``,
            ``mps`` or ``cuda:1``, of the kind the run's device policy
            declares.
        warmup : int
            How many steps run untimed before the first timed one.
        repeats : int
            How many steps are timed.

        Returns
        -------
        StackTiming
            The steps one unit of training length holds, the position and the
            seconds of each timed step and the allocator's peak, or no peak on
            the processor.

        Raises
        ------
        ValueError
            If the device names no device, is a meta device, on which nothing
            runs, or is of a different kind from the device the run's device
            policy declares, whose activation budget the peak is read against.
        """
        requested = _device(device)
        family = spec.device_policy.device.partition(":")[0]

        if requested.type != family:
            msg = (
                f"the run declares a {family} device, so its step is not timed on "
                f"{device!r}"
            )

            raise ValueError(msg)

        seed = point.root

        with device_scope(spec) as declared:
            policy = DevicePolicy(requested, declared.dtype, declared.autocast_dtype)
            accelerated = policy.kind is not DeviceKind.CPU
            workload = _prepared(spec, point, policy)

            sampler = Random(seed)
            drawn = [
                sampler.randrange(workload.positions) for _ in range(warmup + repeats)
            ]
            positions = drawn[warmup:]

            for position in drawn[:warmup]:
                workload.run(position)

            if accelerated:
                torch.accelerator.synchronize(requested)
                torch.accelerator.reset_peak_memory_stats(requested)

            seconds: list[float] = []

            for position in positions:
                start = perf_counter()
                workload.run(position)

                if accelerated:
                    torch.accelerator.synchronize(requested)

                seconds.append(perf_counter() - start)

            peak = (
                int(torch.accelerator.max_memory_allocated(requested))
                if accelerated
                else None
            )

        return StackTiming(
            steps_per_unit=workload.steps_per_unit,
            positions=positions,
            seconds=seconds,
            peak_allocated_bytes=peak,
            shape=workload.shape,
        )

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
        """Time a reference stack's step at every sentence length and batch size.

        Every stack is placed on the timed device and prepared under the
        process settings the first one declares.

        Parameters
        ----------
        stacks : Mapping[tuple[int, int], RunSpec]
            The reference stack declared at each sentence length and batch
            size of the grid, lengths outermost.
        point : PointSeed
            The point every reference stack is planned for.
        device : str
            The device the stacks are built and timed on.
        warmup : int
            How many untimed steps run at every cell before any is timed.
        repeats : int
            How many steps are timed at every cell.

        Returns
        -------
        GridTiming
            The timed seconds of every cell, lengths outermost.

        Raises
        ------
        ValueError
            If the device names no device or is a meta device, or the stacks
            declare different device policies.
        ReferenceShapeError
            If a cell's first batch is not of the cell's length and size.
        """
        _device(device)
        placed = {
            cell: spec.with_(device_policy=spec.device_policy.with_(device=device))
            for cell, spec in stacks.items()
        }
        policies = {spec.device_policy for spec in placed.values()}

        if len(policies) != 1:
            msg = "every reference stack of a pricing grid declares one device policy"

            raise ValueError(msg)

        with device_scope(next(iter(placed.values()))):
            steps = {
                cell: prepare_reference(spec, point, device=device)
                for cell, spec in placed.items()
            }

            for (length, batch), step in steps.items():
                if step.shape != BatchShape(float(length), float(batch)):
                    msg = (
                        f"the reference stack of cell ({length}, {batch}) steps "
                        f"over a first batch of shape {tuple(step.shape)}"
                    )

                    raise ReferenceShapeError(msg)

            def run(cell: tuple[int, int]) -> float:
                start = perf_counter()
                steps[cell].run()

                return perf_counter() - start

            for _ in range(warmup):
                for cell in steps:
                    run(cell)

            timed: dict[tuple[int, int], list[float]] = {cell: [] for cell in steps}

            for _ in range(repeats):
                for cell in steps:
                    timed[cell].append(run(cell))

        return GridTiming(
            [
                GridCell(length, batch, tuple(timed[length, batch]))
                for length, batch in steps
            ]
        )

    def hardware(self, device: str, /) -> HardwareDescriptor:
        """Return the machine a device belongs to, as the framework reports it.

        Parameters
        ----------
        device : str
            The device steps are timed on.

        Raises
        ------
        ValueError
            If the device names no device.
        """
        requested = _device(device)
        cuda = requested.type == "cuda"
        driver = Path("/proc/driver/nvidia/version")

        return HardwareDescriptor(
            device_name=(
                torch.cuda.get_device_name(requested) if cuda else requested.type
            ),
            driver=(
                driver.read_text(encoding="utf-8").splitlines()[0].strip()
                if cuda and driver.is_file()
                else None
            ),
            cuda=torch.version.cuda,
            torch=torch.__version__,
            host_cpu=platform.processor() or platform.machine(),
        )

    @override
    def __repr__(self) -> str:
        """Return the timer's name."""
        return "TrainingStepTimer()"


@final
class ReferenceStep:
    """One reference stack prepared on a device, stepping over its first batch.

    Parameters
    ----------
    workload : TimedStack
        The prepared stack's steps.
    device : torch.device
        Where the stack executes.
    """

    __slots__ = ("_accelerated", "_device", "_workload")

    def __init__(self, workload: TimedStack, device: torch.device, /) -> None:
        self._workload = workload
        self._device = device
        self._accelerated = (
            DevicePolicy(device, torch.float32).kind is not DeviceKind.CPU
        )

    @property
    def device(self) -> torch.device:
        """Return where the stack executes."""
        return self._device

    @property
    def shape(self) -> BatchShape:
        """Return the sentence length and size of the batch every step reads."""
        shape = self._workload.shape_at(0)

        if shape is None:
            msg = "a reference stack's step reads sentences"

            raise ReferenceShapeError(msg)

        return shape

    def run(self) -> None:
        """Run one training step over the first batch and wait for the device."""
        self._workload.run(0)

        if self._accelerated:
            torch.accelerator.synchronize(self._device)


def _prepared(spec: RunSpec, point: PointSeed, policy: DevicePolicy, /) -> TimedStack:
    """Return a run's planned, seeded and prepared stack on a device policy."""
    blueprint = plan_training(spec, point=point)
    accelerated = policy.kind is not DeviceKind.CPU
    streams = RandomStreams(
        SeedPolicy(
            point.root,
            accelerator=str(policy.device) if accelerated else None,
            samplers=blueprint.sampler_streams,
        )
    )
    streams.apply()

    return prepare_training(blueprint, streams, policy).plan.timed_stack()


def prepare_reference(
    spec: RunSpec, point: PointSeed, /, *, device: str
) -> ReferenceStep:
    """Prepare a reference stack whose first batch is one grid cell's.

    Parameters
    ----------
    spec : RunSpec
        The reference run document composed at one cell.
    point : PointSeed
        The point the stack is planned for.
    device : str
        The device the stack is prepared on, under the precision its device
        policy declares.

    Raises
    ------
    ValueError
        If the device names no device or is a meta device.
    """
    requested = _device(device)
    declared = resolve_device_policy(
        spec.with_(device_policy=spec.device_policy.with_(device=device))
    )
    policy = DevicePolicy(requested, declared.dtype, declared.autocast_dtype)
    return ReferenceStep(_prepared(spec, point, policy), requested)
