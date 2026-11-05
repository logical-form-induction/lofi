"""Declared multiple-comparison corrections resolved through installed plugins."""

from __future__ import annotations

from typing import TYPE_CHECKING, Protocol, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.stats import Correction

__all__ = [
    "CorrectionPlugin",
    "correction",
]


class _CorrectionDeclaration(Protocol):
    """A validated declaration selecting an installed correction."""

    @property
    def kind(self) -> object:
        """Return the installed correction tag."""
        ...


class CorrectionPlugin[S: _CorrectionDeclaration](FrozenValue):
    """Build one statistical correction from its exact declaration variant."""

    _noun = "correction plugin"
    __slots__ = ("_build", "_spec")

    def __init__(self, *, spec: type[S], build: Callable[[S], Correction]) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin consumes."""
        return self._spec

    def construct(self, declaration: S, /) -> Correction:
        """Build the declared correction."""
        return self._build(declaration)


def correction(declaration: _CorrectionDeclaration, /) -> Correction:
    """Resolve one declared correction through its unique installed plugin."""
    tag = declaration.kind

    if not isinstance(tag, str):
        msg = f"a correction declaration has no string tag: {tag!r}"
        raise TypeError(msg)

    plugin = cast(
        "CorrectionPlugin[_CorrectionDeclaration]",
        resolve_plugin("correction", tag, CorrectionPlugin, type(declaration)),
    )

    return plugin.construct(declaration)
