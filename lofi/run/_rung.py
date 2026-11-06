"""Declared contraction rungs resolved through installed plugins.

A rung of an experiment's contraction ladder is a declaration variant of the
``RungSpec`` union and an installed plugin of the ``lofi.contraction_rung``
entry point group, which lowers the declaration to the sweep transformation it
states. The plugin reads what it needs of the experiment from one rung
context, and refuses a declaration the experiment cannot contract by, so a
new rung is a new declaration variant and plugin with no edit to the loader.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, NamedTuple, Protocol, cast

from lofi._frozen import FrozenValue
from lofi.registry import resolve_plugin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.run._contraction import Rung

__all__ = [
    "RungContext",
    "RungPlugin",
    "contraction_rung",
]


class RungContext(NamedTuple):
    """What a contraction rung reads of the experiment it contracts.

    Attributes
    ----------
    spanned : Mapping[str, tuple[str, frozenset[str]]]
        For each preregistered factor, the axis that spans it and the axis
        values that realize its levels.
    seeds : int
        The seeds the sweep declares.
    minimum_seeds : int
        The fewest seeds a reported cell rests on.
    """

    spanned: Mapping[str, tuple[str, frozenset[str]]]
    seeds: int
    minimum_seeds: int


class _RungDeclaration(Protocol):
    """A validated declaration selecting an installed contraction rung."""

    @property
    def kind(self) -> object:
        """Return the installed rung tag."""
        ...


class RungPlugin[S: _RungDeclaration](FrozenValue):
    """Lower one contraction rung from its exact declaration variant."""

    _noun = "contraction-rung plugin"
    __slots__ = ("_build", "_spec")

    def __init__(
        self, *, spec: type[S], build: Callable[[S, RungContext], Rung]
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[S]:
        """Return the declaration variant this plugin consumes."""
        return self._spec

    def construct(self, declaration: S, context: RungContext, /) -> Rung:
        """Lower the declared rung against the experiment it contracts.

        Raises
        ------
        ValueError
            If the experiment cannot contract by the declared rung.
        """
        return self._build(declaration, context)


def contraction_rung(declaration: _RungDeclaration, context: RungContext, /) -> Rung:
    """Lower one declared rung through its unique installed plugin.

    Raises
    ------
    RegistryError
        If no plugin or several are installed under the rung's tag, or the
        entry names no rung plugin or one for another declaration.
    ValueError
        If the experiment cannot contract by the rung.
    """
    plugin = cast(
        "RungPlugin[_RungDeclaration]",
        resolve_plugin(
            "contraction_rung",
            cast("str", declaration.kind),
            RungPlugin,
            type(declaration),
        ),
    )

    return plugin.construct(declaration, context)
