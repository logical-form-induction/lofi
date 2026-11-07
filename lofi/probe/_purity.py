"""Stand-ins that record what a probe does to the component it observes."""

from enum import Enum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast, override

import torch
from torch import Tensor

from lofi._checks import LawViolationError
from lofi.nn.policy import ensure_placed

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping
    from types import TracebackType

    from torch.nn import Module

    from lofi.nn.policy import DevicePolicy

__all__ = ["Effect", "Recording", "Touch"]


class Effect(Enum):
    """Names what a stand-in saw done to the component behind it."""

    ASSIGNMENT = "assignment"
    DELETION = "deletion"
    PLACEMENT = "placement"
    REGISTRATION = "registration"


_EFFECTS: Final[Mapping[str, Effect]] = MappingProxyType(
    {
        "add_module": Effect.REGISTRATION,
        "cpu": Effect.PLACEMENT,
        "cuda": Effect.PLACEMENT,
        "mps": Effect.PLACEMENT,
        "register_buffer": Effect.REGISTRATION,
        "register_module": Effect.REGISTRATION,
        "register_parameter": Effect.REGISTRATION,
        "to": Effect.PLACEMENT,
        "to_empty": Effect.PLACEMENT,
        "xpu": Effect.PLACEMENT,
    }
)
"""The methods a stand-in records a call to, under what calling one does."""

_TRANSPARENT: Final = (
    type(None),
    bool,
    int,
    float,
    complex,
    str,
    bytes,
    bytearray,
    tuple,
    list,
    set,
    frozenset,
    dict,
    MappingProxyType,
    range,
    slice,
    Enum,
    Tensor,
    torch.device,
    torch.dtype,
    torch.Size,
)
"""The values a stand-in hands back as themselves rather than behind another."""


class Touch:
    """Records one effect a stand-in saw, and where on the component it landed.

    Parameters
    ----------
    path : tuple[str, ...]
        The attribute names followed from the component to reach what was touched,
        the name touched last.
    effect : Effect
        What was done there.
    holder : str
        The name of the type the effect landed on.
    """

    __slots__ = ("_effect", "_holder", "_path")

    def __init__(self, path: tuple[str, ...], effect: Effect, *, holder: str) -> None:
        self._path = path
        self._effect = effect
        self._holder = holder

    @property
    def path(self) -> tuple[str, ...]:
        """Return the attribute names followed to reach what was touched."""
        return self._path

    @property
    def effect(self) -> Effect:
        """Return what was done there."""
        return self._effect

    @property
    def holder(self) -> str:
        """Return the name of the type the effect landed on."""
        return self._holder

    def _identity(self) -> tuple[tuple[str, ...], Effect, str]:
        """Return the three fields a touch is equal on."""
        return (self._path, self._effect, self._holder)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value records the same three fields."""
        if type(other) is not Touch:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the three fields the touch is equal on."""
        return hash((Touch, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this touch."""
        return f"Touch({self._path!r}, {self._effect!r}, holder={self._holder!r})"


class _Log:
    """Collects what the stand-ins over one component saw."""

    __slots__ = ("_touches",)

    def __init__(self) -> None:
        self._touches: list[Touch] = []

    def record(self, touch: Touch) -> None:
        """Keep one effect.

        Parameters
        ----------
        touch : Touch
            What was seen.
        """
        self._touches.append(touch)

    @property
    def touches(self) -> tuple[Touch, ...]:
        """Return what was seen, in the order it happened."""
        return tuple(self._touches)


class _Watched[**P, R]:
    """Stands in for a method, recording a call to it before making it.

    Parameters
    ----------
    call : Callable[P, R]
        The method stood in for.
    touch : Touch | None
        What to record when it is called, or None where calling it records nothing.
    log : _Log
        Where what is recorded goes.
    """

    __slots__ = ("_call", "_log", "_touch")

    def __init__(self, call: Callable[P, R], touch: Touch | None, log: _Log) -> None:
        self._call = call
        self._touch = touch
        self._log = log

    def __call__(self, *args: P.args, **kwargs: P.kwargs) -> R:
        """Record the call, where it is one that is recorded, and then make it.

        Returns
        -------
        R
            What the method returned.
        """
        if self._touch is not None:
            self._log.record(self._touch)

        return self._call(*args, **kwargs)

    @override
    def __repr__(self) -> str:
        """Return the method stood in for and what calling it records."""
        return f"_Watched({self._call!r}, {self._touch!r})"


class _StandIn[TargetT]:
    """Stands in for one object, recording what is done to it through the stand-in.

    Reading an attribute yields another stand-in, so a write several attributes deep
    is recorded under the path it took. Calling a method yields what the method
    returned, without a stand-in over it.

    Parameters
    ----------
    target : TargetT
        The object stood in for.
    path : tuple[str, ...]
        The attribute names followed from the component to reach it.
    log : _Log
        Where what is done to it goes.
    """

    __slots__ = ("_standin_log", "_standin_path", "_standin_target")

    def __init__(self, target: TargetT, path: tuple[str, ...], log: _Log) -> None:
        object.__setattr__(self, "_standin_target", target)
        object.__setattr__(self, "_standin_path", path)
        object.__setattr__(self, "_standin_log", log)

    def __getattr__[ValueT](self, name: str) -> ValueT:
        """Return one attribute of the object stood in for.

        Parameters
        ----------
        name : str
            The attribute to read.

        Returns
        -------
        ValueT
            The value itself where it holds no state a write could reach, and a
            stand-in over it otherwise.
        """
        target = self._standin_target
        path = (*self._standin_path, name)
        value = getattr(target, name)

        if callable(value):
            effect = _EFFECTS.get(name)
            touched = (
                None
                if effect is None
                else Touch(path, effect, holder=type(target).__name__)
            )

            return cast("ValueT", _Watched(value, touched, self._standin_log))

        if isinstance(value, _TRANSPARENT):
            return cast("ValueT", value)

        return cast("ValueT", _StandIn(value, path, self._standin_log))

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> None:
        """Record an assignment and then make it.

        Parameters
        ----------
        name : str
            The attribute assigned to.
        value : ValueT
            What it is assigned.
        """
        target = self._standin_target
        self._standin_log.record(
            Touch(
                (*self._standin_path, name),
                Effect.ASSIGNMENT,
                holder=type(target).__name__,
            )
        )
        setattr(target, name, value)

    @override
    def __delattr__(self, name: str) -> None:
        """Record a deletion and then make it.

        Parameters
        ----------
        name : str
            The attribute deleted.
        """
        target = self._standin_target
        self._standin_log.record(
            Touch(
                (*self._standin_path, name),
                Effect.DELETION,
                holder=type(target).__name__,
            )
        )
        delattr(target, name)

    @override
    def __repr__(self) -> str:
        """Return the path this stand-in was reached by and what it stands in for."""
        return f"_StandIn({self._standin_target!r}, {self._standin_path!r})"


class Recording[SubjectT: Module]:
    """Holds a component's stand-in and what was done to it through the stand-in.

    A probe is handed the stand-in in place of the component. Assignments and
    deletions made through it are recorded at the path they took, as are calls to
    the methods that move a module or register a submodule. What the component does
    to itself through its own reference is not seen here, and neither is a write to
    a tensor the stand-in handed back.

    Parameters
    ----------
    subject : SubjectT
        The component to be observed.
    policy : DevicePolicy
        Where this run's tensors are placed and what precision they are stored in.

    Raises
    ------
    ValueError
        If a parameter or buffer of the component is off the run's device, or is a
        floating one held in another precision.
    """

    __slots__ = ("_log", "_policy", "_stand_in", "_subject")

    def __init__(self, subject: SubjectT, policy: DevicePolicy) -> None:
        self._subject = ensure_placed(subject, policy)
        self._policy = policy
        self._log = _Log()
        self._stand_in: _StandIn[SubjectT] = _StandIn(subject, (), self._log)

    @property
    def stand_in(self) -> SubjectT:
        """Return what to hand a probe in the component's place."""
        return cast("SubjectT", self._stand_in)

    @property
    def touches(self) -> tuple[Touch, ...]:
        """Return what was done to the component, in the order it happened."""
        return self._log.touches

    def __enter__(self) -> SubjectT:
        """Return what to hand a probe in the component's place."""
        return self.stand_in

    def __exit__(
        self,
        kind: type[BaseException] | None,
        error: BaseException | None,
        traceback: TracebackType | None,
        /,
    ) -> None:
        """Hold the component to having been left where and as it was.

        Raises
        ------
        LawViolationError
            If anything was done to the component through the stand-in.
        ValueError
            If a parameter or buffer has left the run's device or precision.
        """
        if error is not None:
            return

        ensure_placed(self._subject, self._policy)
        touches = self._log.touches

        if touches:
            done = ", ".join(
                f"{touch.effect.value} at {'.'.join(touch.path)}" for touch in touches
            )
            msg = (
                f"observing {type(self._subject).__name__} touched it: {done}. "
                "a probe reads what a component publishes and writes nothing"
            )

            raise LawViolationError(msg)

    @override
    def __repr__(self) -> str:
        """Return what is observed and how much has been recorded."""
        return (
            f"Recording({type(self._subject).__name__}, "
            f"touches={len(self._log.touches)})"
        )
