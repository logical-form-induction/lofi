"""Detection of substrate objects retained in built value graphs.

The walk follows containers, attributes, slots, bound methods, function
defaults and closures, and remaining garbage-collector referents. It reports
didactic, panproto, and caller-refused classes with their paths, while treating
program objects and primitive values as leaves.
"""

from collections.abc import Mapping
from contextlib import suppress
from gc import get_referents
from types import (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    CodeType,
    FrameType,
    FunctionType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodDescriptorType,
    MethodType,
    MethodWrapperType,
    ModuleType,
    TracebackType,
    WrapperDescriptorType,
)
from typing import Final, NamedTuple, Protocol

__all__ = ["SUBSTRATE_PACKAGES", "Reach", "substrate_reaches"]

SUBSTRATE_PACKAGES: Final = frozenset({"didactic", "panproto"})
"""The top-level packages whose objects never survive a boundary."""

_PROGRAM: Final = (
    BuiltinFunctionType,
    ClassMethodDescriptorType,
    CodeType,
    FrameType,
    GetSetDescriptorType,
    MemberDescriptorType,
    MethodDescriptorType,
    MethodWrapperType,
    ModuleType,
    TracebackType,
    WrapperDescriptorType,
)
"""Object kinds that lead out of a value graph and into the program."""


class _Held(Protocol):
    """Any object a built graph may hold, judged by its class alone."""


class Reach(NamedTuple):
    """One foreign object found in a built graph.

    Attributes
    ----------
    path : str
        The attributes, indices and keys leading from the root to the object.
        An attribute reads ``.name``, a sequence member ``[3]``, a mapping
        value ``['key']``, a mapping key ``<key 3>``, a set member ``{3}``,
        a bound method's receiver ``.__self__``, a function's default
        arguments ``.__defaults__[0]`` and ``.__kwdefaults__['name']``, a
        closure cell ``<closure name>`` and a referent the walk found only
        through the garbage collector ``<referent 3>``. The root has the
        empty path.
    held : type
        The class judged foreign: the object's class, or the object itself
        when the object reached is a class.
    instance : bool
        Whether the object reached is an instance of the held class rather
        than the class itself.
    """

    path: str
    held: type
    instance: bool


def _defined_by_the_substrate(cls: type) -> bool:
    """Return whether a class or one of its bases is a substrate class."""
    return any(
        base.__module__.partition(".")[0] in SUBSTRATE_PACKAGES for base in cls.__mro__
    )


def _foreign(cls: type, refused: tuple[type, ...]) -> bool:
    """Return whether a class is a substrate class or a refused one.

    Parameters
    ----------
    refused : tuple[type, ...]
        The additional classes to treat as foreign.
    """
    return _defined_by_the_substrate(cls) or issubclass(cls, refused)


def _slot_members(value: _Held) -> list[tuple[str, _Held]]:
    """Return the set slot values of an object under their slot names.

    Parameters
    ----------
    value : _Held
        The object to inspect.
    """
    members: list[tuple[str, _Held]] = []

    for base in type(value).__mro__:
        for name, member in vars(base).items():
            if not isinstance(member, MemberDescriptorType):
                continue

            with suppress(AttributeError):
                members.append((f".{name}", member.__get__(value, type(value))))

    return members


def _function_members(function: FunctionType) -> list[tuple[str, _Held]]:
    """Return the default arguments and closure cell contents of a function.

    Parameters
    ----------
    function : FunctionType
        The function to inspect.
    """
    members: list[tuple[str, _Held]] = [
        (f".__defaults__[{position}]", member)
        for position, member in enumerate(function.__defaults__ or ())
    ]
    members.extend(
        (f".__kwdefaults__[{name!r}]", member)
        for name, member in (function.__kwdefaults__ or {}).items()
    )

    for name, cell in zip(
        function.__code__.co_freevars, function.__closure__ or (), strict=True
    ):
        with suppress(ValueError):
            members.append((f"<closure {name}>", cell.cell_contents))

    return members


def _children(value: _Held) -> list[tuple[str, _Held]]:
    """Return the values an object holds, each under one path segment.

    Parameters
    ----------
    value : _Held
        The object to inspect.
    """
    named: list[tuple[str, _Held]] = []

    if isinstance(value, FunctionType):
        return _function_members(value)

    if isinstance(value, Mapping):
        for position, (key, member) in enumerate(value.items()):
            named.append((f"<key {position}>", key))
            named.append((f"[{key!r}]", member))
    elif isinstance(value, tuple | list):
        named.extend((f"[{position}]", member) for position, member in enumerate(value))
    elif isinstance(value, set | frozenset):
        named.extend(
            (f"{{{position}}}", member) for position, member in enumerate(value)
        )
    elif isinstance(value, MethodType):
        named.append((".__self__", value.__self__))
        named.append((".__func__", value.__func__))

        return named

    attributes = vars(value) if type(value).__dictoffset__ else {}
    named.extend((f".{name}", member) for name, member in attributes.items())
    named.extend(_slot_members(value))
    covered = {id(member) for _, member in named}
    covered.add(id(attributes))
    covered.add(id(type(value)))
    named.extend(
        (f"<referent {position}>", referent)
        for position, referent in enumerate(get_referents(value))
        if id(referent) not in covered
    )

    return named


def substrate_reaches[R](
    root: R, /, *, refused: tuple[type, ...] = ()
) -> tuple[Reach, ...]:
    """Return every foreign object a built graph reaches, with its path.

    Parameters
    ----------
    root : R
        The built runtime value to walk.
    refused : tuple[type, ...]
        Classes to report beside the substrate's own: a lowerer, a checked
        intermediate, or any other boundary transient that must not survive
        into the built graph. Subclasses are reported too.

    Returns
    -------
    tuple[Reach, ...]
        One record per foreign object, in the order the walk found them, or
        the empty tuple when the graph holds native values only. A foreign
        object is reported once and not walked into. Strings, bytes, numbers
        and ``None`` are leaves, and a class the graph reaches is judged
        without being walked into.
    """
    found: list[Reach] = []
    seen: set[int] = set()
    kept: list[_Held] = []
    pending: list[tuple[str, _Held]] = [("", root)]

    while pending:
        path, value = pending.pop()

        if value is None or isinstance(value, str | bytes | int | float | complex):
            continue

        if id(value) in seen:
            continue

        seen.add(id(value))
        kept.append(value)

        if isinstance(value, type):
            if _foreign(value, refused):
                found.append(Reach(path, value, instance=False))

            continue

        if _foreign(type(value), refused):
            found.append(Reach(path, type(value), instance=True))

            continue

        if isinstance(value, _PROGRAM):
            continue

        pending.extend(
            (path + segment, child) for segment, child in reversed(_children(value))
        )

    return tuple(found)
