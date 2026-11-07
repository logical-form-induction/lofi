"""Completed-step checkpoint restoration."""

from typing import TYPE_CHECKING, cast, final, override

from torch import Tensor

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.train._store import StateTree

__all__ = ["ResumePosition"]


def _validate_state_tree(name: str, state: StateTree, /) -> None:
    """Refuse a value outside the checkpoint state-tree wire type.

    Parameters
    ----------
    name : str
        The path naming the candidate value.
    state : StateTree
        The candidate value.

    Raises
    ------
    TypeError
        If the value is not a recursively well-formed state tree.
    """
    if isinstance(state, Tensor) or state is None:
        return

    if type(state) in {bool, int, float, str}:
        return

    if isinstance(state, (list, tuple)):
        for index, member in enumerate(state):
            _validate_state_tree(f"{name}[{index}]", member)

        return

    if isinstance(state, dict):
        keys = tuple(state)
        string_keys = all(type(key) is str for key in keys)
        integer_keys = all(type(key) is int for key in keys)

        if not string_keys and not integer_keys:
            msg = f"state tree {name} mixes or uses unsupported mapping keys"
            raise TypeError(msg)

        for key, member in state.items():
            _validate_state_tree(f"{name}[{key!r}]", member)

        return

    msg = f"state tree {name} contains unsupported value {type(state).__name__!r}"
    raise TypeError(msg)


def _copy_state_tree(state: StateTree, /) -> StateTree:
    """Return recursively detached tensor and container state."""
    if isinstance(state, Tensor):
        return state.detach().clone()

    if isinstance(state, list):
        return [_copy_state_tree(member) for member in state]

    if isinstance(state, tuple):
        return tuple(_copy_state_tree(member) for member in state)

    if isinstance(state, dict):
        if all(type(key) is str for key in state):
            string_state = cast("dict[str, StateTree]", state)

            return {
                key: _copy_state_tree(member) for key, member in string_state.items()
            }

        integer_state = cast("dict[int, StateTree]", state)

        return {key: _copy_state_tree(member) for key, member in integer_state.items()}

    return state


def _validate_compatible_state(
    name: str,
    saved: StateTree,
    live: StateTree,
    /,
) -> None:
    """Refuse saved state that cannot replace one live state-tree value.

    Parameters
    ----------
    name : str
        The path naming the two values.
    saved : StateTree
        The value recovered from the checkpoint.
    live : StateTree
        The corresponding value expected by the live consumer.

    Raises
    ------
    TypeError
        If a value or mapping key has the wrong type.
    ValueError
        If container structure or tensor metadata disagree.
    """
    _validate_state_tree(name, saved)

    if isinstance(live, Tensor):
        if not isinstance(saved, Tensor):
            msg = f"state {name} is not a tensor"
            raise TypeError(msg)

        if saved.shape != live.shape:
            msg = (
                f"state {name} has shape {tuple(saved.shape)!r}; "
                f"the live value has shape {tuple(live.shape)!r}"
            )
            raise ValueError(msg)

        if saved.dtype != live.dtype:
            msg = (
                f"state {name} has dtype {saved.dtype}; "
                f"the live value has dtype {live.dtype}"
            )
            raise ValueError(msg)

        if saved.layout != live.layout:
            msg = (
                f"state {name} has layout {saved.layout}; "
                f"the live value has layout {live.layout}"
            )
            raise ValueError(msg)

        return

    if type(saved) is not type(live):
        msg = (
            f"state {name} has type {type(saved).__name__!r}; "
            f"the live value has type {type(live).__name__!r}"
        )
        raise TypeError(msg)

    if isinstance(live, (list, tuple)) and isinstance(saved, (list, tuple)):
        if len(saved) != len(live):
            msg = (
                f"state {name} has length {len(saved)}; "
                f"the live value has length {len(live)}"
            )
            raise ValueError(msg)

        for index, (saved_member, live_member) in enumerate(
            zip(saved, live, strict=True)
        ):
            _validate_compatible_state(f"{name}[{index}]", saved_member, live_member)

        return

    if isinstance(live, dict) and isinstance(saved, dict):
        saved_mapping = cast("dict[str | int, StateTree]", saved)
        live_mapping = cast("dict[str | int, StateTree]", live)
        missing = sorted(set(live_mapping) - set(saved_mapping), key=repr)
        extra = sorted(set(saved_mapping) - set(live_mapping), key=repr)

        if missing or extra:
            msg = f"state {name} has missing keys {missing} and extra keys {extra}"
            raise ValueError(msg)

        for key in sorted(live_mapping, key=repr):
            _validate_compatible_state(
                f"{name}[{key!r}]", saved_mapping[key], live_mapping[key]
            )


@final
class ResumePosition(FrozenValue):
    """A recovered completed step and next data position.

    Parameters
    ----------
    step : int
        The number of optimizer steps already completed.
    data_position : int
        The next item position the resumed run reads.
    """

    _noun = "resume position"

    __slots__ = ("_data_position", "_step")

    def __init__(self, step: int, data_position: int) -> None:
        object.__setattr__(self, "_step", ensure_non_negative("a resume step", step))
        object.__setattr__(
            self,
            "_data_position",
            ensure_non_negative("a resume data position", data_position),
        )

    @property
    def step(self) -> int:
        """Return the number of completed optimizer steps."""
        return self._step

    @property
    def data_position(self) -> int:
        """Return the next data position."""
        return self._data_position

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another position names the same two counters."""
        if type(other) is not ResumePosition:
            return NotImplemented

        return (self._step, self._data_position) == (
            other._step,
            other._data_position,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the two counters."""
        return hash((ResumePosition, self._step, self._data_position))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this position."""
        return f"ResumePosition({self._step!r}, {self._data_position!r})"


def _restore_checkpoint(
    restore_components: Callable[[], None],
    restore_optimization: Callable[[], None],
    restore_random: Callable[[], None],
    position: ResumePosition,
    /,
) -> ResumePosition:
    """Restore preflighted component, optimization and random state in order.

    Parameters
    ----------
    restore_components : Callable[[], None]
        The prepared component-state restore operation.
    restore_optimization : Callable[[], None]
        The prepared optimization restore operation.
    restore_random : Callable[[], None]
        The prepared random-stream restore operation.
    position : ResumePosition
        The counters recovered from the checkpoint.

    Returns
    -------
    ResumePosition
        The supplied recovered position after all state is restored.
    """
    restore_components()
    restore_optimization()
    restore_random()

    return position
