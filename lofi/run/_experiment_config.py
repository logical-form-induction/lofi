"""Declared experiment-configuration kinds resolved through installed plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin
from lofi.run._boundary.experiment import ExperimentConfigSpec

if TYPE_CHECKING:
    from collections.abc import Callable
    from pathlib import Path

__all__ = [
    "ExperimentConfigPlugin",
    "experiment_config",
]


class ExperimentConfigPlugin[S: ExperimentConfigSpec, R](FrozenValue):
    """Build one runtime experiment configuration from its declaration."""

    _noun = "experiment-configuration plugin"
    __slots__ = ("_build", "_spec")

    def __init__(self, *, spec: type[S], build: Callable[[S, Path], R]) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin consumes."""
        return self._spec

    def construct(self, declaration: S, path: Path, /) -> R:
        """Build the runtime configuration for the resolved document path."""
        return self._build(declaration, path)


def experiment_config[R](declaration: ExperimentConfigSpec, path: Path, /) -> R:
    """Build one experiment configuration through its installed plugin."""
    plugin = cast(
        "ExperimentConfigPlugin[ExperimentConfigSpec, R]",
        resolve_plugin(
            "experiment_config",
            cast("str", declaration.kind),
            ExperimentConfigPlugin,
            type(declaration),
        ),
    )

    return plugin.construct(declaration, path)
