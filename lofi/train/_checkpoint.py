"""Stored checkpoints and read errors."""

from types import MappingProxyType
from typing import TYPE_CHECKING, NewType, final, override

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.train._boundary import CheckpointManifest
    from lofi.train._ports import CheckpointStore

__all__ = [
    "Checkpoint",
    "CheckpointError",
    "CheckpointId",
    "CheckpointPolicy",
    "CorruptCheckpointError",
    "FingerprintMismatchError",
    "UnknownCheckpointError",
]

CheckpointId = NewType("CheckpointId", str)
"""A store-assigned checkpoint handle."""


class CheckpointError(Exception):
    """Reports a checkpoint read failure."""


class FingerprintMismatchError(CheckpointError):
    """Reports an incompatible checkpoint.

    Parameters
    ----------
    checkpoint : CheckpointId
        The checkpoint that was asked for.
    reasons : Iterable[str]
        One sentence per disagreement, as the manifest comparison reported them.
    """

    __slots__ = ("_reasons",)

    def __init__(self, checkpoint: CheckpointId, reasons: Iterable[str]) -> None:
        stated = tuple(reasons)
        listed = "".join(f"\n  {reason}" for reason in stated)

        super().__init__(f"{checkpoint} cannot be read into this run:{listed}")

        self._reasons = stated

    @property
    def reasons(self) -> tuple[str, ...]:
        """Return incompatibility reasons in report order."""
        return self._reasons


class UnknownCheckpointError(CheckpointError):
    """Reports an unknown checkpoint.

    Parameters
    ----------
    checkpoint : CheckpointId
        The checkpoint that was asked for.
    """

    __slots__ = ()

    def __init__(self, checkpoint: CheckpointId) -> None:
        super().__init__(f"no checkpoint is stored under {checkpoint}")


class CorruptCheckpointError(CheckpointError):
    """Reports a stored checkpoint that cannot be read as written.

    Parameters
    ----------
    checkpoint : CheckpointId
        The checkpoint that was asked for.
    detail : str
        What was found in place of what the store wrote.
    """

    __slots__ = ()

    def __init__(self, checkpoint: CheckpointId, detail: str) -> None:
        super().__init__(
            f"{checkpoint} is not the checkpoint that was written: {detail}"
        )


class Checkpoint[StateT]:
    """A loaded checkpoint.

    Parameters
    ----------
    checkpoint : CheckpointId
        The handle the state was read under.
    manifest : CheckpointManifest
        What the checkpoint records about the run it was written from.
    states : Mapping[str, StateT]
        The saved state of each component, keyed by the role it fills.
    run_state : Mapping[str, StateT] | None
        The saved optimizer, scaler and scheduler state, keyed by the name the
        manifest's reference to each was written under, or None for a checkpoint
        read without run state.

    Raises
    ------
    TypeError
        If the handle is not a string.
    ValueError
        If the handle is empty, if the roles of the state mapping are not the roles
        the manifest names, or if the run state is keyed by a name the manifest
        holds no reference under.
    """

    __slots__ = ("_checkpoint", "_manifest", "_run_state", "_states")

    def __init__(
        self,
        checkpoint: CheckpointId,
        manifest: CheckpointManifest,
        states: Mapping[str, StateT],
        *,
        run_state: Mapping[str, StateT] | None = None,
    ) -> None:
        held = frozenset(states)

        if held != manifest.roles:
            msg = (
                f"the manifest names the roles {sorted(manifest.roles)} and the "
                f"state is keyed by {sorted(held)}"
            )

            raise ValueError(msg)

        blocks: dict[str, StateT] = {} if run_state is None else dict(run_state)
        unreferenced = sorted(frozenset(blocks) - frozenset(manifest.references))

        if unreferenced:
            msg = (
                f"the manifest references run state under "
                f"{sorted(manifest.references)} and the run state is keyed by "
                f"{unreferenced} as well"
            )

            raise ValueError(msg)

        self._checkpoint = CheckpointId(ensure_name("checkpoint", checkpoint))
        self._manifest = manifest
        self._states = MappingProxyType(dict(states))
        self._run_state = MappingProxyType(blocks)

    @property
    def checkpoint(self) -> CheckpointId:
        """Return the checkpoint identifier."""
        return self._checkpoint

    @property
    def manifest(self) -> CheckpointManifest:
        """Return the checkpoint manifest."""
        return self._manifest

    @property
    def states(self) -> Mapping[str, StateT]:
        """Return component state by role."""
        return self._states

    @property
    def run_state(self) -> Mapping[str, StateT]:
        """Return optimizer, scaler and scheduler state by reference key."""
        return self._run_state

    @override
    def __repr__(self) -> str:
        """Return the handle and the roles the checkpoint holds state for."""
        roles = ", ".join(sorted(self._states))

        return f"Checkpoint({self._checkpoint!r}, roles=[{roles}])"


@final
class CheckpointPolicy(FrozenValue):
    """When fitting writes resumable checkpoints and how many of each kind it keeps.

    Parameters
    ----------
    interval : int
        Positive cadence in the fitting's unit of progress, epochs or steps.
    keep : int
        Positive number of the newest resumable checkpoints kept.
    keep_best : int
        Number of the newest validation-best checkpoints kept, zero for a
        fitting that selects none.

    Raises
    ------
    ValueError
        If the cadence or the resumable retention is not positive, or the
        validation-best retention is negative.
    """

    _noun = "checkpoint policy"

    __slots__ = ("_interval", "_keep", "_keep_best")

    def __init__(self, interval: int, keep: int, keep_best: int) -> None:
        object.__setattr__(
            self, "_interval", ensure_positive("a checkpoint interval", interval)
        )
        object.__setattr__(self, "_keep", ensure_positive("a checkpoint keep", keep))
        object.__setattr__(
            self, "_keep_best", ensure_non_negative("a best-checkpoint keep", keep_best)
        )

    @property
    def interval(self) -> int:
        """Return the cadence in the fitting's unit of progress."""
        return self._interval

    @property
    def keep(self) -> int:
        """Return how many of the newest resumable checkpoints are kept."""
        return self._keep

    @property
    def keep_best(self) -> int:
        """Return how many of the newest validation-best checkpoints are kept."""
        return self._keep_best

    def due(self, completed: int, horizon: int, /) -> bool:
        """Return whether a resumable checkpoint is written at a position.

        Parameters
        ----------
        completed : int
            The epochs or steps completed so far.
        horizon : int
            The epochs or steps the fitting runs to.

        Returns
        -------
        bool
            Whether the position falls on the cadence or is the horizon.
        """
        return completed % self._interval == 0 or completed == horizon

    def removable(
        self, kinds: Iterable[tuple[CheckpointId, str | None]], /
    ) -> tuple[CheckpointId, ...]:
        """Return the checkpoints the retention no longer keeps.

        Parameters
        ----------
        kinds : Iterable[tuple[CheckpointId, str | None]]
            Each stored checkpoint, oldest first, with the fitting kind its
            manifest records, or None for one written outside fitting, which
            is never removed.

        Returns
        -------
        tuple[CheckpointId, ...]
            The resumable checkpoints older than the newest ``keep`` and the
            validation-best ones older than the newest ``keep_best``, oldest
            first.
        """
        listed = tuple(kinds)
        latest = [handle for handle, kind in listed if kind == "resume-latest"]
        best = [handle for handle, kind in listed if kind == "validation-best"]
        dropped = frozenset(latest[: max(len(latest) - self._keep, 0)]) | frozenset(
            best[: max(len(best) - self._keep_best, 0)]
        )

        return tuple(handle for handle, _ in listed if handle in dropped)

    def prune[StateT](
        self, store: CheckpointStore[StateT], /
    ) -> tuple[CheckpointId, ...]:
        """Remove every stored checkpoint the retention no longer keeps.

        Parameters
        ----------
        store : CheckpointStore[StateT]
            The run's checkpoint store.

        Returns
        -------
        tuple[CheckpointId, ...]
            The checkpoints removed, oldest first.
        """
        kinds: list[tuple[CheckpointId, str | None]] = []

        for handle in store.list():
            fitting = store.manifest(handle).fitting
            kinds.append((handle, None if fitting is None else str(fitting.kind)))

        removed = self.removable(kinds)

        for handle in removed:
            store.remove(handle)

        return removed

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another policy has the same cadence and retention."""
        if type(other) is not CheckpointPolicy:
            return NotImplemented

        return (self._interval, self._keep, self._keep_best) == (
            other._interval,
            other._keep,
            other._keep_best,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the cadence and retention."""
        return hash((CheckpointPolicy, self._interval, self._keep, self._keep_best))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this policy."""
        return (
            f"CheckpointPolicy({self._interval!r}, {self._keep!r}, {self._keep_best!r})"
        )
