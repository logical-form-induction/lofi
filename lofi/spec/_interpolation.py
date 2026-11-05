"""LoFI's interpolation resolvers and the run context they read.

Three resolvers sit above the configuration engine's built-ins. ``${lofi.path:rel}``
joins a relative path against the run's data root, ``${lofi.pkg:group/option}`` names
a configuration fragment packaged with LoFI, and ``${lofi.seed:name}`` derives a
sub-seed for a named stream.

A derived sub-seed is a function of the run's seed and the name alone rather than a
draw from a stream, so a name gets the same sub-seed in every process that composes
the same run, whichever names are asked for and in whatever order. That is what lets
two runs of one configuration shuffle the same data and initialize the same weights
while a third run asks for one stream more.

The data root and the seed are not leaves of the specification, so the resolvers read
them from a run context installed around the composition that needs them.
"""

from contextlib import contextmanager
from contextvars import ContextVar
from hashlib import blake2b
from pathlib import Path, PurePosixPath, PureWindowsPath
from typing import TYPE_CHECKING, Final, NamedTuple, cast

from didactic.settings import InterpolationError, ResolverFn, register_resolver

from lofi._checks import ensure_name, ensure_non_negative
from lofi.registry import Registry, entry_point_group
from lofi.spec._groups import PACKAGED_CONFIG_ROOT

if TYPE_CHECKING:
    from collections.abc import Iterator

__all__ = [
    "FRAGMENT_SUFFIXES",
    "RESOLVER_GROUP",
    "RunContext",
    "active_run_context",
    "derive_seed",
    "register_resolvers",
    "run_context",
]

FRAGMENT_SUFFIXES: Final = (".yaml", ".yml", ".toml", ".json")
"""The suffixes a packaged fragment is looked for under, in the order tried."""

_RESOURCE_PREFIX: Final = "lofi.resource:"

RESOLVER_GROUP: Final = entry_point_group("resolver")
"""Entry-point group containing interpolation resolver callables."""


class RunContext(NamedTuple):
    """What the resolvers read while a run's configuration is interpolated.

    Attributes
    ----------
    data_root : Path
        The absolute directory ``${lofi.path:rel}`` joins its argument against.
    seed : int
        The run's seed, which every ``${lofi.seed:name}`` derives from.
    """

    data_root: Path
    seed: int


_ACTIVE: ContextVar[RunContext | None] = ContextVar("lofi_run_context", default=None)


def active_run_context() -> RunContext | None:
    """Return the run context the resolvers read, or ``None`` outside a run.

    Returns
    -------
    RunContext | None
        The innermost context installed on this thread or task.
    """
    return _ACTIVE.get()


@contextmanager
def run_context(context: RunContext, /) -> Iterator[RunContext]:
    """Install a run context for the resolvers to read inside the block.

    Parameters
    ----------
    context : RunContext
        The data root and seed the resolvers read.

    Yields
    ------
    RunContext
        The context that was installed.
    """
    token = _ACTIVE.set(context)

    try:
        yield context

    finally:
        _ACTIVE.reset(token)


def derive_seed(seed: int, name: str, /) -> int:
    """Return the sub-seed a named stream takes under a run's seed.

    Parameters
    ----------
    seed : int
        The run's seed.
    name : str
        The stream the sub-seed is for.

    Returns
    -------
    int
        A seed below ``2 ** 32``, determined by the run's seed and the name.

    Raises
    ------
    TypeError
        If the seed is not an integer or the name is not a string.
    ValueError
        If the seed is negative or the name is empty.
    """
    ensure_non_negative("run seed", seed)
    ensure_name("stream name", name)
    digest = blake2b(
        f"{seed}:{name}".encode(), digest_size=4, person=b"lofi.seed"
    ).digest()

    return int.from_bytes(digest, "big")


def register_resolvers(*, replace: bool = False) -> tuple[str, ...]:
    """Register LoFI's resolvers with the configuration engine.

    Parameters
    ----------
    replace : bool
        Whether a name another resolver already holds may be taken over. Off by
        default, so shadowing a resolver a plugin registered is loud.

    Returns
    -------
    tuple[str, ...]
        The names registered, in the order they are called under.

    Raises
    ------
    ValueError
        If a name is already registered and ``replace`` is false.
    """
    resolvers = _installed_resolvers()

    for name, resolver in resolvers:
        register_resolver(name, resolver, replace=replace)

    return tuple(name for name, _ in resolvers)


def _installed_resolvers() -> tuple[tuple[str, ResolverFn], ...]:
    """Load every installed resolver after checking the discovery boundary.

    Returns
    -------
    tuple[tuple[str, ResolverFn], ...]
        Resolver names and callables in stable name and value order.

    Raises
    ------
    TypeError
        If an entry point names a value that is not callable.
    ValueError
        If more than one entry point claims one resolver name.
    """
    installed: list[tuple[str, ResolverFn]] = []

    for name, member in Registry[object]("resolver").members():
        if not callable(member):
            msg = f"resolver {name!r} names {member!r}, which is not callable"
            raise TypeError(msg)

        installed.append((name, cast("ResolverFn", member)))

    return tuple(installed)


def _argument(resolver: str, args: tuple[str, ...]) -> str:
    """Return the single argument a resolver was called with.

    Parameters
    ----------
    resolver : str
        The name the resolver is called under, for the refusal.
    args : tuple[str, ...]
        The arguments the expression supplied.

    Returns
    -------
    str
        The one argument, which the engine has already stripped of surrounding
        whitespace.

    Raises
    ------
    InterpolationError
        If the expression supplied no argument, more than one, or an empty one.
    """
    if len(args) != 1 or not args[0]:
        supplied = ", ".join(repr(arg) for arg in args) or "nothing"
        msg = f"${{{resolver}:...}} takes one argument; got {supplied}"
        raise InterpolationError(msg)

    return args[0]


def _context(resolver: str) -> RunContext:
    """Return the installed run context, refusing the expression when there is none.

    Parameters
    ----------
    resolver : str
        The name the resolver is called under, for the refusal.

    Returns
    -------
    RunContext
        The innermost installed context.

    Raises
    ------
    InterpolationError
        If no run context is installed.
    """
    context = _ACTIVE.get()

    if context is None:
        msg = (
            f"${{{resolver}:...}} needs a run context; no data root or seed is in scope"
        )
        raise InterpolationError(msg)

    return context


def _relative_parts(resolver: str, declared: str) -> tuple[str, ...]:
    """Return the components of a path declared relative to a root.

    Parameters
    ----------
    resolver : str
        The name the resolver is called under, for the refusal.
    declared : str
        The path text the expression supplied, in POSIX form.

    Returns
    -------
    tuple[str, ...]
        The ordinary name components, with any ``.`` component and any repeated or
        trailing separator dropped.

    Raises
    ------
    InterpolationError
        If the text is absolute, drive-qualified, home-relative, spelled with a
        backslash or a ``..`` component, or names nothing at all. Resolver arguments
        reach this function losslessly, so every backslash spelling is refused here.
    """
    if "\\" in declared:
        msg = f"${{{resolver}:{declared}}} is spelled with a backslash"
        raise InterpolationError(msg)

    if declared.startswith("~"):
        msg = f"${{{resolver}:{declared}}} is home-relative"
        raise InterpolationError(msg)

    relative = PurePosixPath(declared)

    if relative.is_absolute():
        msg = f"${{{resolver}:{declared}}} is absolute, so it would discard the root"
        raise InterpolationError(msg)

    if PureWindowsPath(declared).drive:
        msg = (
            f"${{{resolver}:{declared}}} is drive-qualified, "
            f"so it names no path below the root"
        )
        raise InterpolationError(msg)

    parts = tuple(part for part in relative.parts if part != ".")

    if ".." in parts:
        msg = f"${{{resolver}:{declared}}} is spelled with a '..' component"
        raise InterpolationError(msg)

    if not parts:
        msg = f"${{{resolver}:{declared}}} names no path below the root"
        raise InterpolationError(msg)

    return parts


def _lofi_path(*args: str) -> str:
    """Return ``${lofi.path:rel}``: ``rel`` joined against the run's data root.

    Parameters
    ----------
    *args : str
        The path, relative to the data root and in POSIX form.

    Returns
    -------
    str
        The absolute path, as text.

    Raises
    ------
    InterpolationError
        If no run context is installed, or the path is not one relative to a root.
    """
    declared = _argument("lofi.path", args)
    root = _context("lofi.path").data_root

    return str(root.joinpath(*_relative_parts("lofi.path", declared)))


def _lofi_pkg(*args: str) -> str:
    """Return ``${lofi.pkg:group/option}``: the path of a packaged fragment.

    Parameters
    ----------
    *args : str
        The group and the option in it, separated by ``/``.

    Returns
    -------
    str
        The absolute path of the fragment, as text.

    Raises
    ------
    InterpolationError
        If the argument names no group, or the packaged library holds no such
        option under any of the suffixes tried.
    """
    declared = _argument("lofi.pkg", args)
    parts = _relative_parts("lofi.pkg", declared)

    if len(parts) < 2:
        msg = f"${{lofi.pkg:{declared}}} names no group; write 'group/option'"
        raise InterpolationError(msg)

    stem = PACKAGED_CONFIG_ROOT.joinpath(*parts)
    tried = [stem.with_name(stem.name + suffix) for suffix in FRAGMENT_SUFFIXES]
    found = [candidate for candidate in tried if candidate.is_file()]

    if not found:
        paths = ", ".join(str(candidate) for candidate in tried)
        msg = f"${{lofi.pkg:{declared}}} matches no packaged fragment; tried {paths}"
        raise InterpolationError(msg)

    if len(found) > 1:
        names = ", ".join(candidate.name for candidate in found)
        msg = f"${{lofi.pkg:{declared}}} is ambiguous: {names}"
        raise InterpolationError(msg)

    return str(found[0])


def lower_package_resource(value: str, /) -> str:
    """Resolve one package-resource reference to an absolute path.

    A string beginning with ``lofi.resource:`` names an exact file below the
    packaged configuration root. Any other string is returned unchanged.

    Parameters
    ----------
    value : str
        A validated setting that may name a packaged file.

    Returns
    -------
    str
        The absolute path the reference names, or the value as given.

    Raises
    ------
    ValueError
        If a reference is malformed or names no packaged file.
    """
    if not value.startswith(_RESOURCE_PREFIX):
        return value

    declared = value.removeprefix(_RESOURCE_PREFIX)

    try:
        parts = _relative_parts("lofi.resource", declared)

    except InterpolationError as error:
        raise ValueError(str(error)) from error

    resource = PACKAGED_CONFIG_ROOT.joinpath(*parts)

    if not resource.is_file():
        msg = f"package resource {value!r} names no file: {resource}"
        raise ValueError(msg)

    return str(resource)


def _lofi_seed(*args: str) -> int:
    """Return ``${lofi.seed:name}``: the sub-seed a named stream takes.

    Parameters
    ----------
    *args : str
        The name of the stream.

    Returns
    -------
    int
        The sub-seed the name takes under the run's seed.

    Raises
    ------
    InterpolationError
        If no run context is installed, or the expression names no stream.
    """
    name = _argument("lofi.seed", args)

    return derive_seed(_context("lofi.seed").seed, name)
