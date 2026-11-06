"""Declarations for the named interpretations of a grammar."""

from typing import TYPE_CHECKING, Literal, cast

import didactic.api as dx

from lofi._frozen import FrozenValue
from lofi.interpret import (
    CategoryInterpretation,
    DependencyInterpretation,
    InterpretationDeclaration,
    MeaningInterpretation,
    StringInterpretation,
)
from lofi.registry import resolve_plugin

if TYPE_CHECKING:
    from collections import abc
    from collections.abc import Mapping

    from lofi._types import HeadSelection

__all__ = [
    "CategoryInterpretationSpec",
    "DependencyInterpretationSpec",
    "InterpretationPlugin",
    "InterpretationSpec",
    "MeaningInterpretationSpec",
    "NamedInterpretationSpec",
    "PreparedInterpretation",
    "StringInterpretationSpec",
    "default_interpretations",
    "interpretation_plugin",
]


class InterpretationSpec(dx.TaggedUnion, discriminator="kind", extra="forbid"):
    """One selectable derivation interpretation implementation."""


class CategoryInterpretationSpec(InterpretationSpec, extra="forbid"):
    """Select the exact category at each derivation node."""

    kind: Literal["category"] = "category"


class StringInterpretationSpec(InterpretationSpec, extra="forbid"):
    """Select the surface string yielded by a derivation."""

    kind: Literal["string"] = "string"


class DependencyInterpretationSpec(InterpretationSpec, extra="forbid"):
    """Select dependencies projected by the grammar's rule heads."""

    kind: Literal["dependency"] = "dependency"


class MeaningInterpretationSpec(InterpretationSpec, extra="forbid"):
    """Select the logical form yielded by semantic actions."""

    kind: Literal["meaning"] = "meaning"


class InterpretationPlugin[Declaration](FrozenValue):
    """Pair one interpretation declaration with its runtime builder."""

    _noun = "interpretation plugin"
    __slots__ = ("_build", "_prepare", "_spec")

    def __init__(
        self,
        *,
        spec: type[Declaration],
        prepare: abc.Callable[[Declaration], object],
        build: abc.Callable[
            [str, object, Mapping[str, HeadSelection]],
            InterpretationDeclaration,
        ],
    ) -> None:
        object.__setattr__(self, "_spec", spec)
        object.__setattr__(self, "_prepare", prepare)
        object.__setattr__(self, "_build", build)

    @property
    def spec(self) -> type[Declaration]:
        """Return the declaration class this plugin builds."""
        return self._spec

    def construct(
        self,
        name: str,
        declaration: Declaration,
        head_selections: Mapping[str, HeadSelection],
        /,
    ) -> InterpretationDeclaration:
        """Build one named runtime interpretation declaration."""
        return self._build(name, self._prepare(declaration), head_selections)

    def prepare(self, declaration: Declaration, /) -> object:
        """Prepare one interpretation at declaration validation."""
        return self._prepare(declaration)

    def construct_prepared(
        self,
        name: str,
        prepared: object,
        head_selections: Mapping[str, HeadSelection],
        /,
    ) -> InterpretationDeclaration:
        """Build one runtime interpretation from prepared metadata."""
        return self._build(name, prepared, head_selections)


class PreparedInterpretation(FrozenValue):
    """One selected interpretation and its validation-time plan."""

    _noun = "prepared interpretation"
    __slots__ = ("_name", "_plan", "_plugin")

    def __init__(
        self,
        name: str,
        plugin: InterpretationPlugin[InterpretationSpec],
        plan: object,
        /,
    ) -> None:
        object.__setattr__(self, "_name", name)
        object.__setattr__(self, "_plugin", plugin)
        object.__setattr__(self, "_plan", plan)

    def construct(
        self, head_selections: Mapping[str, HeadSelection], /
    ) -> InterpretationDeclaration:
        """Build the prepared runtime interpretation."""
        return self._plugin.construct_prepared(
            self._name,
            self._plan,
            head_selections,
        )


def interpretation_plugin(
    declaration: InterpretationSpec, /
) -> InterpretationPlugin[InterpretationSpec]:
    """Return the installed plugin selected by an interpretation declaration."""
    plugin = resolve_plugin(
        "interpretation",
        cast("str", declaration.kind),
        InterpretationPlugin,
        type(declaration),
    )

    return cast("InterpretationPlugin[InterpretationSpec]", plugin)


class NamedInterpretationSpec(dx.Model, extra="forbid"):
    """One document-level name bound to a derivation interpretation."""

    name: str
    reading: InterpretationSpec

    __axioms__ = (dx.axiom("len(name) > 0", message="an interpretation needs a name"),)

    def build(
        self,
        head_selections: Mapping[str, HeadSelection],
        /,
    ) -> InterpretationDeclaration:
        """Build the named runtime declaration against the grammar's rule heads."""
        return interpretation_plugin(self.reading).construct(
            self.name,
            self.reading,
            head_selections,
        )


def default_interpretations() -> tuple[NamedInterpretationSpec, ...]:
    """Return the four canonical grammar interpretation declarations."""
    return (
        NamedInterpretationSpec(name="category", reading=CategoryInterpretationSpec()),
        NamedInterpretationSpec(name="string", reading=StringInterpretationSpec()),
        NamedInterpretationSpec(
            name="dependency", reading=DependencyInterpretationSpec()
        ),
        NamedInterpretationSpec(name="meaning", reading=MeaningInterpretationSpec()),
    )


def _build_category(
    name: str,
    prepared: object,
    head_selections: Mapping[str, HeadSelection],
    /,
) -> InterpretationDeclaration:
    """Build a named exact-category interpretation."""
    del prepared, head_selections
    return CategoryInterpretation(name)


def _build_string(
    name: str,
    prepared: object,
    head_selections: Mapping[str, HeadSelection],
    /,
) -> InterpretationDeclaration:
    """Build a named surface-string interpretation."""
    del prepared, head_selections
    return StringInterpretation(name)


def _build_dependency(
    name: str,
    prepared: object,
    head_selections: Mapping[str, HeadSelection],
    /,
) -> InterpretationDeclaration:
    """Build a named dependency interpretation from declared rule heads."""
    del prepared
    return DependencyInterpretation(
        name,
        {
            rule_name: selection.slot
            for rule_name, selection in head_selections.items()
            if selection.slot is not None
        },
    )


def _build_meaning(
    name: str,
    prepared: object,
    head_selections: Mapping[str, HeadSelection],
    /,
) -> InterpretationDeclaration:
    """Build a named logical form interpretation declaration."""
    del prepared, head_selections
    return MeaningInterpretation(name)


def _prepare_builtin(declaration: InterpretationSpec, /) -> None:
    """Return the empty plan shared by fieldless built-in declarations."""
    del declaration


CATEGORY_INTERPRETATION_PLUGIN = InterpretationPlugin(
    spec=CategoryInterpretationSpec,
    prepare=_prepare_builtin,
    build=_build_category,
)
STRING_INTERPRETATION_PLUGIN = InterpretationPlugin(
    spec=StringInterpretationSpec,
    prepare=_prepare_builtin,
    build=_build_string,
)
DEPENDENCY_INTERPRETATION_PLUGIN = InterpretationPlugin(
    spec=DependencyInterpretationSpec,
    prepare=_prepare_builtin,
    build=_build_dependency,
)
MEANING_INTERPRETATION_PLUGIN = InterpretationPlugin(
    spec=MeaningInterpretationSpec,
    prepare=_prepare_builtin,
    build=_build_meaning,
)
