import copy
from typing import NamedTuple

import pytest

from lofi.nn.controllers import mlp as controller_module
from lofi.registry import ComponentPlugin, Registry, RegistryError, plugin_for
from lofi.spec.controllers import MlpControllerSettings, MlpControllerSpec
from lofi.spec.layers import FeedForwardLayerSpec
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSettings, RnnSpanDecoderSpec


class Built(NamedTuple):
    """What the test builder returns."""

    hidden: int
    context: str


def build(settings: MlpControllerSettings, context: str, /) -> Built:
    """Return the settings' hidden width beside the context."""
    return Built(settings.hidden[0].width, context)


def build_span(settings: RnnSpanDecoderSettings, context: str, /) -> str:
    """Return the context a span decoder would be built in."""
    del settings

    return context


PLUGIN = ComponentPlugin(spec=MlpControllerSpec, build=build)


def test_a_plugin_builds_from_its_declarations_settings_and_a_context() -> None:
    assert PLUGIN.spec is MlpControllerSpec
    assert PLUGIN.build is build
    declaration = MlpControllerSpec(
        hidden=(FeedForwardLayerSpec(width=5, activation="tanh"),)
    )
    assert PLUGIN.construct(declaration, "run") == Built(5, "run")


def test_a_plugin_is_frozen_and_names_its_declaration() -> None:
    assert copy.copy(PLUGIN) is PLUGIN
    assert repr(PLUGIN) == "ComponentPlugin(spec=MlpControllerSpec)"

    with pytest.raises(AttributeError):
        PLUGIN.__setattr__("_spec", RnnSpanDecoderSpec)


def test_a_declaration_resolves_to_the_plugin_its_family_lists() -> None:
    assert plugin_for(MlpControllerSpec()) is controller_module.PLUGIN


def test_an_entry_naming_something_other_than_a_plugin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: build)

    with pytest.raises(RegistryError, match="not a ComponentPlugin"):
        plugin_for(MlpControllerSpec())


def test_an_entry_naming_another_variants_plugin_is_refused(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    other = ComponentPlugin(spec=RnnSpanDecoderSpec, build=build_span)
    monkeypatch.setattr(Registry, "get", lambda _registry, _tag: other)

    with pytest.raises(RegistryError, match="builds RnnSpanDecoderSpec"):
        plugin_for(MlpControllerSpec())
