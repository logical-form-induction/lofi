"""Reusable laws for a component built from its declaration through its plugin."""

from __future__ import annotations

from typing import TYPE_CHECKING, ClassVar, NamedTuple, get_origin

import pytest

from lofi.registry import plugin_for

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from lofi.spec import ConfiguredSpec

__all__ = ["ConfiguredComponentConformance", "ConfiguredComponentFixture"]


class ConfiguredComponentFixture(NamedTuple):
    """One declaration, the context it is built in, and what the build shows.

    Attributes
    ----------
    declaration : ConfiguredSpec[object]
        A validated declaration of the variant under test.
    context : object
        What the family's builders take beside the declaration's settings.
    expected_type : type
        The exact type of the component the declaration selects.
    observe : Callable[[object], Mapping[str, bool]]
        Named evidence that the component holds what its settings and its
        context describe.
    unchanged : Callable[[], bool]
        Whether the context is as it was before the build.
    """

    declaration: ConfiguredSpec[object]
    context: object
    expected_type: type
    observe: Callable[[object], Mapping[str, bool]]
    unchanged: Callable[[], bool]


def _defaults(function: object, /) -> tuple[object, ...]:
    """Return the positional and keyword defaults a function declares."""
    positional = getattr(function, "__defaults__", None) or ()
    keyword = getattr(function, "__kwdefaults__", None) or {}

    return (*positional, *keyword.values())


class ConfiguredComponentConformance:
    """Laws every component built through its family's plugin obeys."""

    pytestmark: ClassVar[list[pytest.MarkDecorator]] = [pytest.mark.conformance]

    def test_the_installed_plugin_pairs_the_declarations_class(
        self, configured_component_fixture: ConfiguredComponentFixture
    ) -> None:
        """Resolve the declaration's family and tag to a plugin for its class."""
        declaration = configured_component_fixture.declaration

        assert plugin_for(declaration).spec is type(declaration)

    def test_the_component_holds_what_its_declaration_and_context_describe(
        self, configured_component_fixture: ConfiguredComponentFixture
    ) -> None:
        """Build the component and observe its type and what it was built from."""
        fixture = configured_component_fixture
        built = plugin_for(fixture.declaration).construct(
            fixture.declaration, fixture.context
        )
        observations = fixture.observe(built)

        assert type(built) is fixture.expected_type
        assert observations
        assert all(observations.values()), observations
        assert fixture.unchanged()

    def test_each_build_is_a_separate_component(
        self, configured_component_fixture: ConfiguredComponentFixture
    ) -> None:
        """Build twice from one declaration and context."""
        fixture = configured_component_fixture
        plugin = plugin_for(fixture.declaration)
        first = plugin.construct(fixture.declaration, fixture.context)
        second = plugin.construct(fixture.declaration, fixture.context)

        assert first is not second

    def test_neither_the_builder_nor_the_constructor_declares_a_default(
        self, configured_component_fixture: ConfiguredComponentFixture
    ) -> None:
        """Keep every default on the declaration's fields."""
        fixture = configured_component_fixture
        plugin = plugin_for(fixture.declaration)
        built = get_origin(fixture.expected_type) or fixture.expected_type

        assert _defaults(plugin.build) == ()

        if built.__module__.startswith("lofi."):
            assert _defaults(built.__init__) == ()
