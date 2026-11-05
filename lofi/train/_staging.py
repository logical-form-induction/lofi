"""The stages of a training protocol and the run that fits them in order."""

from typing import TYPE_CHECKING, Never, final, override

from lofi._checks import (
    ensure_name,
    ensure_non_negative,
    ensure_positive,
    refuse_reassignment,
)
from lofi.train._boundary import staged_manifest
from lofi.train._checkpoint import CheckpointId
from lofi.train._grouping import resolve_groups

if TYPE_CHECKING:
    from collections.abc import Iterator, Sequence

    from torch.nn import Module

    from lofi.train._groups import ActiveGroups, ParameterGroups
    from lofi.train._ports import CheckpointStore, StageFitting
    from lofi.train._steps import TrainResult

__all__ = ["Stage", "StageCompletion", "StagedProtocol"]


@final
class Stage:
    """One stage of a training protocol: what it trains, and for how long.

    A stage names the declared parameter groups it trains, and every other
    declared group is frozen for its duration. What makes a protocol staged is
    thus the sequence of these values a run is configured with, rather than a
    step written for any one of them.

    Parameters
    ----------
    name : str
        What the stage is called, unique within a protocol.
    active : ActiveGroups
        The declared groups the stage trains.
    epochs : int
        How many epochs the stage fits for.

    Raises
    ------
    TypeError
        If the name is not a string or the epoch count is not an integer.
    ValueError
        If the name is empty or the stage fits for no epoch.
    """

    __slots__ = ("_active", "_epochs", "_name")

    def __init__(self, name: str, active: ActiveGroups, epochs: int = 1) -> None:
        object.__setattr__(self, "_name", ensure_name("a stage name", name))
        object.__setattr__(self, "_active", active)
        object.__setattr__(
            self, "_epochs", ensure_positive("a stage's epoch count", epochs)
        )

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("stage")

    @property
    def name(self) -> str:
        """Return what the stage is called."""
        return self._name

    @property
    def active(self) -> ActiveGroups:
        """Return the declared groups the stage trains."""
        return self._active

    @property
    def epochs(self) -> int:
        """Return how many epochs the stage fits for."""
        return self._epochs

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another stage trains the same groups for as long."""
        if type(other) is not Stage:
            return NotImplemented

        return (
            self._name == other._name
            and self._active == other._active
            and self._epochs == other._epochs
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the name, the active groups and the epoch count."""
        return hash((Stage, self._name, self._active, self._epochs))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this stage."""
        return f"Stage({self._name!r}, {self._active!r}, {self._epochs!r})"


@final
class StageCompletion:
    """Records what one stage of a protocol trained and where it was written.

    Parameters
    ----------
    stage : str
        The stage's name.
    checkpoint : CheckpointId
        The handle the stage's checkpoint was written under.
    trained : frozenset[str]
        The registered names of the parameters the stage left active.
    frozen : frozenset[str]
        The registered names of the parameters the stage held fixed.
    results : Sequence[TrainResult]
        What each of the stage's epochs reported, in order.

    Raises
    ------
    TypeError
        If the stage name or the handle is not a string.
    ValueError
        If the stage name or the handle is empty, if no epoch is reported, or if
        a parameter is both trained and frozen.
    """

    __slots__ = ("_checkpoint", "_frozen", "_results", "_stage", "_trained")

    def __init__(
        self,
        stage: str,
        checkpoint: CheckpointId,
        trained: frozenset[str],
        frozen: frozenset[str],
        results: Sequence[TrainResult],
    ) -> None:
        reported = tuple(results)

        if not reported:
            msg = f"the stage {stage!r} reports no epoch"

            raise ValueError(msg)

        both = sorted(trained & frozen)

        if both:
            msg = f"the stage {stage!r} both trained and froze {both}"

            raise ValueError(msg)

        object.__setattr__(self, "_stage", ensure_name("a stage name", stage))
        object.__setattr__(
            self, "_checkpoint", CheckpointId(ensure_name("checkpoint", checkpoint))
        )
        object.__setattr__(self, "_trained", trained)
        object.__setattr__(self, "_frozen", frozen)
        object.__setattr__(self, "_results", reported)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("stage completion")

    @property
    def stage(self) -> str:
        """Return the stage's name."""
        return self._stage

    @property
    def checkpoint(self) -> CheckpointId:
        """Return the handle the stage's checkpoint was written under."""
        return self._checkpoint

    @property
    def trained(self) -> frozenset[str]:
        """Return the registered names of the parameters the stage left active."""
        return self._trained

    @property
    def frozen(self) -> frozenset[str]:
        """Return the registered names of the parameters the stage held fixed."""
        return self._frozen

    @property
    def results(self) -> tuple[TrainResult, ...]:
        """Return what each of the stage's epochs reported, in order."""
        return self._results

    @property
    def step(self) -> int:
        """Return the optimizer-step count the stage ended at."""
        return self._results[-1].step

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another completion records the same stage the same way."""
        if type(other) is not StageCompletion:
            return NotImplemented

        return (
            self._stage == other._stage
            and self._checkpoint == other._checkpoint
            and self._trained == other._trained
            and self._frozen == other._frozen
            and self._results == other._results
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the stage, the handle, the parameters and the results."""
        return hash(
            (
                StageCompletion,
                self._stage,
                self._checkpoint,
                self._trained,
                self._frozen,
                self._results,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the stage, the handle and how much it trained and froze."""
        return (
            f"StageCompletion(stage={self._stage!r}, "
            f"checkpoint={self._checkpoint!r}, trained={len(self._trained)}, "
            f"frozen={len(self._frozen)}, epochs={len(self._results)})"
        )


@final
class StagedProtocol[StateT]:
    """Fits declared stages in order over one model's declared parameter groups.

    Before each stage the declaration and the stage's active set are resolved
    against the model, which is what freezes every parameter the stage does not
    train, and after it the state is checkpointed under a manifest naming the
    stage. Nothing here knows which stage is which: a protocol that fits a chart
    parser on judgments, then only the map that reads types off its cells, then
    both together, is three of these values in a sequence.

    Parameters
    ----------
    stages : Sequence[Stage]
        The stages, in the order they are fitted.
    groups : ParameterGroups
        The parameter groups the model's components declare.
    model : Module
        The wired model each stage resolves the declaration against.
    fitting : StageFitting[StateT]
        What fits the parameters a stage leaves active, and what a stage's
        checkpoint is written from.
    store : CheckpointStore[StateT]
        Where each stage's checkpoint is written.

    Raises
    ------
    ValueError
        If no stage is given or two stages share a name.
    """

    __slots__ = ("_fitting", "_groups", "_model", "_stages", "_store")

    def __init__(
        self,
        stages: Sequence[Stage],
        groups: ParameterGroups,
        model: Module,
        fitting: StageFitting[StateT],
        store: CheckpointStore[StateT],
    ) -> None:
        held = tuple(stages)

        if not held:
            msg = "a staged protocol fits at least one stage"

            raise ValueError(msg)

        names = [stage.name for stage in held]
        repeated = sorted({name for name in names if names.count(name) > 1})

        if repeated:
            msg = f"two stages are called {repeated}"

            raise ValueError(msg)

        self._stages = held
        self._groups = groups
        self._model = model
        self._fitting = fitting
        self._store = store

    @property
    def stages(self) -> tuple[Stage, ...]:
        """Return the stages, in the order they are fitted."""
        return self._stages

    def __iter__(self) -> Iterator[Stage]:
        """Yield the stages in the order they are fitted."""
        return iter(self._stages)

    def __len__(self) -> int:
        """Return how many stages the protocol fits."""
        return len(self._stages)

    def run(self, step: int = 0) -> tuple[StageCompletion, ...]:
        """Fit every stage in order and return what each of them trained.

        Parameters
        ----------
        step : int
            The count of optimizer steps the run has taken before the first
            stage.

        Returns
        -------
        tuple[StageCompletion, ...]
            One completion per stage, in the order they were fitted.

        Raises
        ------
        TypeError
            If the count is not an integer.
        ValueError
            If the count is negative.
        GroupError
            If a stage names a group the declaration does not hold, or if the
            declaration does not fit the model.
        """
        taken = ensure_non_negative("step", step)
        completions: list[StageCompletion] = []

        for position, stage in enumerate(self._stages):
            resolution = resolve_groups(self._groups, stage.active, self._model)
            results: list[TrainResult] = []

            for _ in range(stage.epochs):
                reported = self._fitting.fit(resolution, taken)
                taken = reported.step
                results.append(reported)

            manifest, states = self._fitting.checkpoint(taken)
            written = staged_manifest(
                manifest,
                name=stage.name,
                active=stage.active.names,
                position=position,
            )
            completions.append(
                StageCompletion(
                    stage.name,
                    self._store.save(written, states),
                    resolution.trainable,
                    resolution.frozen,
                    results,
                )
            )

        return tuple(completions)

    @override
    def __repr__(self) -> str:
        """Return the stages in the order they are fitted."""
        return f"StagedProtocol({list(self._stages)!r})"
