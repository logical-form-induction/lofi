"""Prices an experiment from timed training steps of the stacks its points declare.

An experiment's points compose to run documents. Points whose documents agree
differ only in their seed and share one stack, so each distinct stack is built
once and its training step is timed over a seeded random sample of its steps on
one device. The training steps a point takes follow from the training length
its trainer's plugin states and from the steps one unit of that length holds in
the built stack. The experiment's accelerator-hours at the median and at the
ninetieth percentile are quantiles of its total hours under a seeded bootstrap
over those timings, and the price keeps every raw timing, the resample count,
the seed and the machine the device belongs to, so the figures can be
recomputed from the record and read against the hardware they were measured on.

An experiment whose document declares a length-by-batch grid is priced off the
step of its declared reference run document timed at every cell of the grid
instead, each stack's step interpolated at its own mean sentence length and
batch size.

Only an experiment whose configuration kind trains a timed step is priced:
a kind whose points train no step this cost model times, such as a judgment
or baseline fit, is refused rather than given a figure. Each stack is built
under the run seed its first point derives from the experiment's root seed, as
training builds it. Nothing is priced whose training length is not declared.
A trainer that stops early at the latest after its declared epochs is priced
at those epochs; a
trainer whose plugin states no training length is refused before any stack is
built. Each stack's peak allocation is read against the activation-memory
budget its own run document's device policy declares.
"""

import json
from hashlib import sha256
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from lofi import _checks
from lofi.run._boundary.sweep import resolve_point
from lofi.run._experiment import expand_experiment
from lofi.run._launch import sweep_id
from lofi.run._values import ExperimentPrice, MemoryBudget, StackPrice
from lofi.spec import RunSpecCodec
from lofi.training_plugins import trainer_plugin

if TYPE_CHECKING:
    from collections.abc import Mapping
    from pathlib import Path

    from lofi.run._boundary.sweep import SweepPoint
    from lofi.run._experiment import Experiment
    from lofi.run._seed import PointSeed
    from lofi.run._values import GridTiming, HardwareDescriptor, StackTiming
    from lofi.spec import RunSpec
    from lofi.training_plugins import TrainingLength

__all__ = [
    "StepTimer",
    "UndeclaredTrainingLengthError",
    "UnpriceableExperimentError",
    "price_experiment",
    "training_length",
]


class UndeclaredTrainingLengthError(ValueError):
    """A run document that declares no training length to price.

    Attributes
    ----------
    trainer : str
        The registry tag of the trainer the run document declares.
    """

    def __init__(self, trainer: str) -> None:
        self.trainer = trainer
        super().__init__(
            f"the trainer {trainer!r} declares neither epochs nor steps, so its "
            f"training length is undeclared and it has no price"
        )


class UnpriceableExperimentError(ValueError):
    """An experiment over a configuration whose points train no timed step.

    Attributes
    ----------
    experiment : str
        The experiment refused.
    kind : str
        The kind of configuration document the experiment sweeps.
    """

    def __init__(self, experiment: str, kind: str) -> None:
        self.experiment = experiment
        self.kind = kind
        super().__init__(
            f"the experiment {experiment!r} sweeps a {kind} document, whose points "
            f"train no step this cost model times, so it has no price"
        )


@runtime_checkable
class StepTimer(Protocol):
    """Builds the stack a run document declares and times its training step."""

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
        """Build a run's stack, run its step untimed, then time repeated steps.

        Parameters
        ----------
        spec : RunSpec
            The validated run document whose stack is built.
        point : PointSeed
            The point the stack is planned for, whose run seed it is built
            under and which derives any data seed the run declares as derived.
        device : str
            The device the stack is built and timed on.
        warmup : int
            How many steps run untimed before the first timed one.
        repeats : int
            How many steps are timed.

        Raises
        ------
        ValueError
            If the device is of a different kind from the device the run's
            device policy declares, whose activation budget the peak is read
            against.
        """
        ...

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
        """Time a reference stack's step at every length and batch size.

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
        """
        ...

    def hardware(self, device: str, /) -> HardwareDescriptor:
        """Return the machine a device belongs to.

        Parameters
        ----------
        device : str
            The device steps are timed on.
        """
        ...


def training_length(spec: RunSpec, /) -> TrainingLength:
    """Return the training length one run document declares.

    The installed plugin of the run's trainer states the length, in the unit
    its training counts: a parser trainer's epochs, which cap any early stop,
    or a distance trainer's steps.

    Parameters
    ----------
    spec : RunSpec
        The validated run document.

    Raises
    ------
    UndeclaredTrainingLengthError
        If the trainer declares no training length.
    """
    trainer = spec.trainer
    length = trainer_plugin(trainer).price(trainer)

    if length is None:
        raise UndeclaredTrainingLengthError(str(trainer.kind))

    return length


def _stack(point: SweepPoint, /) -> str:
    """Return the identifier of the run document a point composes to."""
    canonical = json.dumps(dict(point.config), separators=(",", ":"), sort_keys=True)

    return sha256(canonical.encode()).hexdigest()[:16]


def price_experiment(
    experiment: Experiment,
    root: Path,
    timer: StepTimer,
    /,
    *,
    device: str,
    warmup: int,
    repeats: int,
) -> ExperimentPrice:
    """Time the training step of every distinct stack an experiment declares.

    Parameters
    ----------
    experiment : Experiment
        The experiment whose sweep is expanded and priced.
    root : Path
        The repository root whose configuration groups the points compose with.
    timer : StepTimer
        Builds each stack and times its training step.
    device : str
        The device every stack is timed on, of the kind every point's device
        policy declares.
    warmup : int
        The untimed steps each stack runs before its timed ones.
    repeats : int
        The timed steps each stack runs, at least two.

    Returns
    -------
    ExperimentPrice
        Every raw timing, the grid the experiment declares timed when it
        declares one, and the machine timed on, with the bootstrap that reads
        the P50 and P90 accelerator-hours off them seeded by the experiment's
        root.

    Raises
    ------
    ValueError
        If fewer than two steps would be timed, the warmup is negative, or the
        timer refuses the device as not of the kind a point declares.
    RuntimeError
        If the debug checks are on, since a checked build times the checks.
    UnpriceableExperimentError
        If the experiment's configuration kind trains no timed step; nothing is
        timed.
    UndeclaredTrainingLengthError
        If any point's run document declares no training length; nothing is
        timed.
    ValueError
        If the experiment declares a grid and a stack reads no sentences.
    """
    if not experiment.config.timed:
        raise UnpriceableExperimentError(experiment.name, experiment.config.kind)

    if repeats < 2:
        msg = f"a bootstrap is read off at least two timed runs, not {repeats}"

        raise ValueError(msg)

    if warmup < 0:
        msg = f"a warmup count is not negative: {warmup}"

        raise ValueError(msg)

    if _checks.CHECKS:
        msg = (
            "LOFI_CHECK is on, so a step would be timed with the debug checks in "
            "it; price with LOFI_CHECK=0"
        )

        raise RuntimeError(msg)

    points = expand_experiment(experiment, root)
    codec = RunSpecCodec()
    stacks: dict[str, tuple[RunSpec, TrainingLength, list[SweepPoint]]] = {}

    for point in points:
        key = _stack(point)

        if key in stacks:
            stacks[key][2].append(point)
            continue

        spec = resolve_point(point, codec)
        stacks[key] = (spec, training_length(spec), [point])

    priced = [
        StackPrice(
            stack=key,
            points=[point.point_id for point in members],
            length=length,
            timing=timer.time(
                spec,
                experiment.seed_policy.point(members[0].point_id, members[0].seed),
                device=device,
                warmup=warmup,
                repeats=repeats,
            ),
            budget=MemoryBudget(spec.device_policy.activation_memory_bytes),
        )
        for key, (spec, length, members) in stacks.items()
    ]

    pricing = experiment.pricing

    return ExperimentPrice(
        experiment=experiment.name,
        freeze_id=experiment.freeze_id,
        sweep_id=sweep_id(points),
        device=device,
        hardware=timer.hardware(device),
        warmup=warmup,
        repeats=repeats,
        resamples=pricing.resamples,
        bootstrap_seed=experiment.seed_policy.root,
        stacks=priced,
        grid=pricing.reference(timer, device=device, warmup=warmup, repeats=repeats),
    )
