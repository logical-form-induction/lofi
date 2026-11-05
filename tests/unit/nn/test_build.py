import copy
import dataclasses
import pickle
from typing import TYPE_CHECKING, Final, Literal, cast

import pytest
from tests.conformance.component_plugin import plugin_context

from lofi._artifacts import ArtifactFetcher, HttpArchiveTransport
from lofi.nn._snapshots import HubSnapshotTransport
from lofi.nn.build import BuildContext, build_component
from lofi.nn.controllers import MlpController
from lofi.nn.tokens.pretrained import HuggingFacePretrainedLoader
from lofi.registry import UnknownTagError
from lofi.spec import ControllerSpec
from lofi.spec.controllers import MlpControllerSpec
from lofi.spec.encoders.hrr import HrrTermEncoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec
from lofi.spec.layers import FeedForwardLayerSpec
from lofi.spec.poolers.attention import AttentionSubwordPoolerSpec

if TYPE_CHECKING:
    from pathlib import Path

    import didactic.api as dx

    from lofi.nn.encoders import HrrTermEncoder, TreeRnnEncoder
    from lofi.nn.tokens.pool import AttentionSubwordPooler

WIDTH: Final = 4


class Cells:
    """A producer of vectors at the test width."""

    output_width = WIDTH


def with_default(
    monkeypatch: pytest.MonkeyPatch, spec: type[dx.Model], field: str, *, value: object
) -> None:
    """Change the default one declaration field takes, for the test's duration."""
    held = spec.__field_specs__[field]
    monkeypatch.setitem(
        spec.__field_specs__, field, dataclasses.replace(held, default=value)
    )


def test_a_context_holds_the_run_resources_and_the_wired_producers() -> None:
    cells = Cells()
    context = plugin_context(component="controller", producers={"encoder": cells})

    assert context.component == "controller"
    assert context.producer("encoder") is cells
    assert dict(context.producers) == {"encoder": cells}
    assert dict(context.targets) == {}
    assert context.meaning is None
    assert context.categories
    assert context.rules.name == "plugin-context"


def test_a_port_no_producer_is_wired_to_is_refused_with_the_wired_ports() -> None:
    context = plugin_context(component="pooler", producers={"leaves": Cells()})

    with pytest.raises(KeyError, match=r"'pooler' reads 'encoder'.*leaves"):
        context.producer("encoder")


def test_a_context_for_another_component_shares_every_run_resource(
    tmp_path: Path,
) -> None:
    context = plugin_context(component="model")
    loader = HuggingFacePretrainedLoader(
        ArtifactFetcher(
            tmp_path,
            archives=HttpArchiveTransport(),
            snapshots=HubSnapshotTransport(),
            offline=True,
        )
    )
    other = context.for_component(context.component, {"encoder": Cells()})
    loaded = other.with_loader(loader)
    targeted = other.with_targets({})

    assert other.language is context.language
    assert other.choices is context.language.codec.choices
    assert other.codec is context.codec
    assert other.vocabulary is context.vocabulary
    assert other.actions is context.actions
    assert other.rules is context.rules
    assert other.categories is context.categories
    assert other.policy is context.policy
    assert other.meaning is context.meaning
    assert loaded.loader is loader
    assert loaded.producers == other.producers
    assert targeted.producers == other.producers


def test_a_context_without_a_declared_cache_refuses_to_name_a_loader() -> None:
    with pytest.raises(LookupError, match="declares no artifacts cache"):
        _ = plugin_context().loader


def test_a_context_is_frozen_and_refuses_serialization() -> None:
    context = plugin_context()

    assert copy.deepcopy(context) is context

    with pytest.raises(TypeError, match="cannot be pickled"):
        pickle.dumps(context)


def test_a_declaration_is_built_through_the_plugin_its_family_lists() -> None:
    policy = plugin_context().policy
    built = build_component(
        MlpControllerSpec(hidden=(FeedForwardLayerSpec(width=3, activation="tanh"),)),
        plugin_context(component="controller", producers={"cells": Cells()}),
    )

    assert type(built) is MlpController
    assert built.input_width == WIDTH
    assert all(parameter.dtype is policy.dtype for parameter in built.parameters())


class UnlistedControllerSpec(ControllerSpec):
    """A controller variant no entry point lists."""

    kind: Literal["unlisted-controller"] = "unlisted-controller"

    def settings(self) -> tuple[()]:
        """Return no settings."""
        return ()


def test_a_declaration_of_a_tag_no_plugin_lists_is_refused() -> None:
    with pytest.raises(UnknownTagError):
        build_component(UnlistedControllerSpec(), plugin_context())


def test_changing_a_declaration_default_changes_the_built_controller(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = plugin_context(component="controller", producers={"cells": Cells()})
    before = cast("MlpController", build_component(MlpControllerSpec(), context))
    with_default(
        monkeypatch,
        MlpControllerSpec,
        "hidden",
        value=(FeedForwardLayerSpec(width=7, activation="tanh"),),
    )
    after = cast("MlpController", build_component(MlpControllerSpec(), context))

    assert before.state_dict()["_head._network.0.weight"].shape[0] == 64
    assert after.state_dict()["_head._network.0.weight"].shape[0] == 7


def test_changing_a_declaration_default_changes_the_built_encoder_and_pooler(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    context = plugin_context(component="built", producers={"encoder": Cells()})
    with_default(monkeypatch, TreeRnnEncoderSpec, "layers", value=3)
    with_default(
        monkeypatch,
        AttentionSubwordPoolerSpec,
        "scorer",
        value=(FeedForwardLayerSpec(width=WIDTH, activation="relu"),),
    )
    with_default(monkeypatch, HrrTermEncoderSpec, "magnitude", value=0.5)
    encoder = cast(
        "TreeRnnEncoder", build_component(TreeRnnEncoderSpec(width=WIDTH), context)
    )
    pooler = cast(
        "AttentionSubwordPooler",
        build_component(AttentionSubwordPoolerSpec(), context),
    )
    holographic = cast(
        "HrrTermEncoder", build_component(HrrTermEncoderSpec(width=WIDTH), context)
    )

    assert encoder.layers == 3
    assert sum(parameter.numel() for parameter in pooler.parameters()) == (
        WIDTH * WIDTH + WIDTH + WIDTH
    )
    assert holographic.codebook.codes.magnitude == 0.5


def test_the_context_type_is_the_one_the_builders_take() -> None:
    assert isinstance(plugin_context(), BuildContext)
