"""Declared sweep launchers resolved through installed plugins.

A launcher plugin builds its backend from one launch context, which offers
every runtime resource a launcher may need: the experiment's store, its stop
rules, the in-process executor its configuration kind builds, the resolved
site profile, the task program, the price record and the command runner. A
launcher reads its own settings off its declaration, so a new launcher is a
new declaration variant and plugin, with no edit to the context or to the
command that launches it.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin
from lofi.run._array import ArrayJob, ArrayLauncher
from lofi.run._boundary.experiment import (
    ArrayLaunchSpec,
    LaunchSpec,
    RemoteLaunchSpec,
    SerialLaunchSpec,
)
from lofi.run._remote import RemoteLauncher
from lofi.run._serial import SerialLauncher

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.run._boundary.site import Launcher, SiteProfile
    from lofi.run._launch import CommandRunner
    from lofi.run._serial import ExecutePoint
    from lofi.run._store import ExperimentStore
    from lofi.run._values import ExperimentPrice

__all__ = [
    "ARRAY_PLUGIN",
    "REMOTE_PLUGIN",
    "SERIAL_PLUGIN",
    "LaunchContext",
    "LauncherPlugin",
    "launcher_plugin",
]


class LaunchContext(Protocol):
    """The runtime resources an installed launcher plugin builds a backend from.

    Each resource is built when a launcher first asks for it.
    """

    @property
    def store(self) -> ExperimentStore:
        """Return the store the experiment's points execute under."""
        ...

    @property
    def stop(self) -> Callable[[], bool]:
        """Return the experiment's stop rules, read after each point run here."""
        ...

    @property
    def run(self) -> CommandRunner:
        """Return the runner of a site's commands."""
        ...

    def site(self) -> SiteProfile:
        """Return the resolved site profile."""
        ...

    def program(self) -> tuple[str, ...]:
        """Return the argument vector, after the interpreter, one task runs."""
        ...

    def execute(self) -> ExecutePoint:
        """Return the executor that claims and executes each point in this process."""
        ...

    def price(self) -> ExperimentPrice:
        """Return the price record of the experiment's sweep."""
        ...


class LauncherPlugin[S: LaunchSpec](FrozenValue):
    """Build one launcher from its exact declaration variant.

    A launcher whose submitted work outlives the launching process declares
    how that work is canceled once a stop rule fires; a launcher that runs
    its points before returning declares none, since it leaves nothing to
    cancel.
    """

    _noun = "launcher plugin"
    __slots__ = ("_build", "_cancel", "_requires_site", "_spec")

    def __init__(
        self,
        *,
        spec: type[S],
        build: Callable[[S, LaunchContext], Launcher],
        requires_site: bool,
        cancel: Callable[[S, LaunchContext], None] | None = None,
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)
        object.__setattr__(self, "_requires_site", requires_site)
        object.__setattr__(self, "_cancel", cancel)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin consumes."""
        return self._spec

    @property
    def requires_site(self) -> bool:
        """Return whether the launcher submits its points to a site."""
        return self._requires_site

    def construct(self, declaration: S, context: LaunchContext, /) -> Launcher:
        """Build the declared launcher from the launch context."""
        return self._build(declaration, context)

    def cancel(self, declaration: S, context: LaunchContext, /) -> bool:
        """Cancel the declared launcher's outstanding work, if it leaves any.

        Returns
        -------
        bool
            Whether the launcher declares a cancellation, which was then run.
        """
        if self._cancel is None:
            return False

        self._cancel(declaration, context)

        return True


def launcher_plugin(declaration: LaunchSpec, /) -> LauncherPlugin[LaunchSpec]:
    """Return the unique installed plugin selected by a launch declaration."""
    return cast(
        "LauncherPlugin[LaunchSpec]",
        resolve_plugin(
            "launcher", cast("str", declaration.kind), LauncherPlugin, type(declaration)
        ),
    )


def _serial(declaration: SerialLaunchSpec, context: LaunchContext, /) -> Launcher:
    """Build the backend that executes each point in this process, in order."""
    del declaration
    return SerialLauncher(context.store, execute=context.execute(), stop=context.stop)


def _array(declaration: ArrayLaunchSpec, context: LaunchContext, /) -> Launcher:
    """Build the backend that submits the pending points as one array job."""
    return ArrayLauncher(
        context.site(),
        store=context.store,
        program=context.program(),
        chunk_size=declaration.chunk_size,
        price=context.price(),
        throttle=declaration.throttle,
        run=context.run,
    )


def _cancel_array(declaration: ArrayLaunchSpec, context: LaunchContext, /) -> None:
    """Cancel every task the experiment's array submissions left with the site."""
    del declaration
    ArrayJob(context.site(), context.store.identity, run=context.run).cancel()


def _remote(declaration: RemoteLaunchSpec, context: LaunchContext, /) -> Launcher:
    """Build the backend that opens one remote session per pending point."""
    del declaration
    return RemoteLauncher(
        context.site(), store=context.store, program=context.program(), run=context.run
    )


SERIAL_PLUGIN = LauncherPlugin(
    spec=SerialLaunchSpec,
    build=_serial,
    requires_site=False,
)
ARRAY_PLUGIN = LauncherPlugin(
    spec=ArrayLaunchSpec,
    build=_array,
    requires_site=True,
    cancel=_cancel_array,
)
REMOTE_PLUGIN = LauncherPlugin(
    spec=RemoteLaunchSpec,
    build=_remote,
    requires_site=True,
)
