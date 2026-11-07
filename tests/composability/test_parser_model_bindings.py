import subprocess
import sys
from typing import TYPE_CHECKING, Final, cast

import torch
from didactic.settings import compose
from tests.conformance.component_plugin import plugin_context
from tests.support.rules import ccg_signature, declared_rules

from lofi.judge import Action, ActionSet
from lofi.nn.build import build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import RecursiveTermDecoder
from lofi.nn.loss._readout import TermsReadout
from lofi.nn.parsers import TensorChart
from lofi.nn.parsers import _channel as channel_module
from lofi.nn.parsers import _outside as outside_module
from lofi.nn.policy import DevicePolicy
from lofi.nn.sites import ComponentId
from lofi.nn.tokens.embedder import (
    ReservedTokens,
    StaticTokenEmbedder,
    TokenVocabulary,
)
from lofi.nn.tokens.pool import FirstSubwordPooler
from lofi.run import section_root
from lofi.spec import (
    DataModelLanguageSpec,
    ModelSection,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.spec._plugins import load_plugins
from lofi.spec.controllers import MlpControllerSpec
from lofi.spec.decoders.rnn import RnnTermDecoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec
from lofi.spec.layers import FeedForwardLayerSpec
from lofi.spec.parsers.chart import InsideOutsideParserSpec
from lofi.spec.readouts import ReadoutSource, TargetSource, TermsReadoutSpec
from lofi.spec.spaces import CombinatorFamilySpec, Merge, SpaceSpec
from lofi.terms import (
    CompleteTree,
    Identity,
    Operator,
    OperatorIndex,
    Signature,
    TargetSet,
    Term,
    TermCodec,
    TermStore,
)
from lofi.train import ChartForward
from lofi.train._spaces import wire_spaces

if TYPE_CHECKING:
    from collections.abc import Mapping

    import pytest

    from lofi.deduce import SpanIndex
    from lofi.nn.build import BuildContext
    from lofi.nn.parsers import DeepInsideOutside
    from lofi.nn.ports import Controller, SubwordPooler, TokenEncoder
    from lofi.nn.spaces import CombinatorFamily


WIDTH: Final = 4
KEEP: Final = Action("keep")
APPLY: Final = Action("apply", "application")
COMPOSE: Final = Action("compose", "composition")
ACTIONS: Final = ActionSet((KEEP, APPLY, COMPOSE), KEEP)
SIGNATURE: Final = Signature(
    "configured-parser-model",
    ("Cat",),
    (
        Operator("S", "Cat", family="primitive"),
        Operator("NP", "Cat", family="primitive"),
    ),
)
CODEC: Final = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=1),
)
VIEW: Final = NeuralCodecView(CODEC)
STORE: Final = TermStore(SIGNATURE)
MATCHING: Final = Identity[Term]()
TARGETS: Final = TargetSet((STORE.op(SIGNATURE.by_name("S")),), MATCHING)
POLICY: Final = DevicePolicy(torch.device("cpu"), torch.float64)


def _binary_rule(name: str, kept: str) -> PatternRuleSpec:
    """Declare one binary fixture rule keeping one named premise."""
    return PatternRuleSpec(
        name=name,
        premises=(
            TermPatternSpec(name="left", sort="Cat"),
            TermPatternSpec(name="right", sort="Cat"),
        ),
        conclusion=TermPatternSpec(name=kept, sort="Cat"),
    )


RULES: Final = declared_rules(
    ccg_signature("S", "NP"),
    RuleSetSpec(
        name="configured-parser-model",
        rules=(
            _binary_rule("application", "left"),
            _binary_rule("composition", "right"),
        ),
    ),
).rule_set


class Batch:
    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        return (("dog", "runs"),)


class Cells:
    """A producer of chart cells at the test width."""

    output_width = WIDTH


def context(role: str, producers: Mapping[str, object]) -> BuildContext:
    """Return the test run's context for one component."""
    return plugin_context(
        codec=VIEW,
        component=role,
        producers=producers,
        policy=POLICY,
        actions=ACTIONS,
        rules=RULES,
        targets={"grammar-goal": TARGETS},
    )


def model_section() -> ModelSection:
    """Return one represented category space with two composing decoders."""
    return ModelSection(
        language=DataModelLanguageSpec(),
        parser=InsideOutsideParserSpec(width=WIDTH),
        encoders={"terms": TreeRnnEncoderSpec(width=WIDTH)},
        decoders={
            "identity": RnnTermDecoderSpec(width=WIDTH),
            "application": RnnTermDecoderSpec(width=WIDTH),
            "composition": RnnTermDecoderSpec(width=WIDTH),
        },
        spaces={
            "categories": SpaceSpec(
                sort="Cat",
                encoder="terms",
                identity="identity",
                combinators=CombinatorFamilySpec(
                    merge=Merge.CHART_COMBINE,
                    decoders={"apply": "application", "compose": "composition"},
                ),
            )
        },
        readouts={
            "types": TermsReadoutSpec(
                space="categories",
                reads=ReadoutSource.INSIDE,
                targets=TargetSource.GRAMMAR_GOAL,
                output="root-mass",
            )
        },
    )


def test_installed_bindings_preserve_one_inventory_by_name_through_the_forward(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    stores = 0

    def packed_store(
        index: SpanIndex,
        like: torch.Tensor,
        /,
        *,
        batch: int,
        width: int,
    ) -> TensorChart:
        nonlocal stores
        stores += 1
        return TensorChart(index, like, batch=batch, width=width)

    monkeypatch.setattr(channel_module, "TensorChart", packed_store, raising=False)
    monkeypatch.setattr(outside_module, "TensorChart", packed_store, raising=False)
    controller = cast(
        "Controller",
        build_component(
            MlpControllerSpec(
                hidden=(FeedForwardLayerSpec(width=6, activation="tanh"),)
            ),
            context("controller", {"cells": Cells()}),
        ),
    )
    vocabulary = TokenVocabulary(
        ("dog", "runs"), reserved=ReservedTokens({"<unk>": 0}), unknown="<unk>"
    )
    tokens = StaticTokenEmbedder(
        torch.randn(vocabulary.size, WIDTH, dtype=torch.float64),
        vocabulary=vocabulary,
        component=ComponentId("tokens"),
        trainable=True,
        window=None,
        load_report=None,
    )
    pooler = FirstSubwordPooler(WIDTH, component=ComponentId("pooler"))
    chart = cast(
        "DeepInsideOutside",
        build_component(
            InsideOutsideParserSpec(width=WIDTH),
            context("parser", {"leaves": pooler, "rules": controller}),
        ),
    )
    graph = wire_spaces(
        model_section(),
        context("spaces", {}),
        parser=chart,
        pretrained_resources={},
        role_targets={"types": {"grammar-goal": TARGETS}},
        erasure=None,
    )
    readout = graph.readouts["types"]
    family = cast("CombinatorFamily", graph.spaces["categories"].combinators)
    forward = ChartForward(
        cast("TokenEncoder", tokens),
        cast("SubwordPooler", pooler),
        chart,
        readouts=(readout,),
        actions=ACTIONS,
    )
    outputs = forward.publish(Batch())

    assert stores == 3
    assert controller.action_set is ACTIONS
    assert family.actions is ACTIONS
    assert chart.rules.action_set is ACTIONS
    assert tuple(outputs.actions) == ACTIONS.actions
    assert all(
        published is declared
        for published, declared in zip(outputs.actions, ACTIONS.actions, strict=True)
    )
    assert tuple(family.decoders) == (APPLY, COMPOSE)
    assert all(
        isinstance(bound, RecursiveTermDecoder) for bound in family.decoders.values()
    )
    assert isinstance(readout, TermsReadout)
    assert readout.key == "types"
    assert readout.targets == ("root-mass",)
    assert outputs.marginals is not None
    assert tuple(outputs.readouts) == ("types", "root-mass")
    assert bool(outputs.readouts[readout.targets[0]].computed.all())


def test_model_declaration_and_action_bindings_ignore_mapping_order() -> None:
    forward = model_section().with_(
        readouts={
            "zeta": TermsReadoutSpec(space="categories", reads=ReadoutSource.INSIDE),
            "alpha": TermsReadoutSpec(space="categories", reads=ReadoutSource.INSIDE),
        },
        spaces={
            "categories": SpaceSpec(
                sort="Cat",
                encoder="terms",
                identity="identity",
                combinators=CombinatorFamilySpec(
                    merge=Merge.CHART_COMBINE,
                    decoders={"compose": "composition", "apply": "application"},
                ),
            )
        },
    )
    reversed_declaration = model_section().with_(
        readouts=dict(reversed(forward.readouts.items())),
        spaces={
            "categories": SpaceSpec(
                sort="Cat",
                encoder="terms",
                identity="identity",
                combinators=CombinatorFamilySpec(
                    merge=Merge.CHART_COMBINE,
                    decoders={"apply": "application", "compose": "composition"},
                ),
            )
        },
    )

    assert forward == reversed_declaration
    assert tuple(forward.readouts) == ("alpha", "zeta")
    family = cast("CombinatorFamilySpec", forward.spaces["categories"].combinators)
    assert tuple(family.decoders) == ("apply", "compose")


def test_packaged_controller_readout_and_family_fragments_compose() -> None:
    load_plugins()
    model = compose(
        None,
        schema=ModelSection,
        base={
            "language": {"from": "data"},
            "spaces": {
                "categories": {
                    "sort": "Cat",
                    "encoder": "terms",
                    "identity": "identity",
                    "combinators": {
                        "merge": "concatenate",
                        "decoders": {
                            "apply": "application",
                            "compose": "application",
                        },
                    },
                }
            },
        },
        groups={
            "parser": "inside-outside",
            "controller": "mlp",
            "encoders.terms": "hrr",
            "decoders.identity": "hrr",
            "decoders.application": "hrr",
            "readouts": "types",
        },
        search_path=(section_root("model"),),
    )

    assert model.controller is not None
    assert model.controller.model_dump()["kind"] == "mlp"
    readout = cast("TermsReadoutSpec", model.readouts["types"])
    assert readout.space == "categories"
    assert readout.output == "sentences"
    assert readout.targets is TargetSource.GRAMMAR_GOAL
    family = cast("CombinatorFamilySpec", model.spaces["categories"].combinators)
    assert family.decoders == {"apply": "application", "compose": "application"}


def test_every_binding_family_is_discovered_in_a_fresh_process() -> None:
    script = """
from lofi.registry import Registry
from lofi.spec import ControllerSpec, ReadoutSpec
from lofi.spec._plugins import load_plugins

loaded = load_plugins()
assert 'mlp' in loaded
assert 'terms' in loaded
assert ControllerSpec.model_validate({'kind': 'mlp'}).kind == 'mlp'
assert ReadoutSpec.model_validate({
    'kind': 'terms', 'space': 'categories', 'reads': 'inside',
}).kind == 'terms'
assert Registry('controller').get('mlp')
assert Registry('readout').get('terms')
assert Registry('term_decoder').get('tree-rnn')
assert Registry('chart').get('inside-outside')
"""
    completed = subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        encoding="utf-8",
    )

    assert completed.returncode == 0, completed.stderr
