from importlib.metadata import entry_points
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.component_plugin import (
    ComponentPluginConformance,
    ComponentPluginFixture,
    plugin_context,
)
from tests.pretrained_stand_ins import StandInLoader
from tests.support.rules import ccg_signature, declared_rules
from torch import nn

from lofi.data.datasets import ArtifactDatasetPlugin, DatasetPlugin
from lofi.judge import Action, ActionSet
from lofi.nn.build import BuildContext, build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.controllers import (
    BilinearController,
    MlpController,
    TransformerController,
)
from lofi.nn.decoders import (
    DiffusionTermDecoder,
    HolographicTermDecoder,
    LinearRnnDecoder,
    RecursiveTermDecoder,
    TransformerTermDecoder,
)
from lofi.nn.encoders import (
    HrrTermEncoder,
    LinearRnnEncoder,
    TransformerTermEncoder,
    TreeRnnEncoder,
)
from lofi.nn.loss._readout import TermsReadout
from lofi.nn.parsers import (
    DeepInsideOutside,
    FlatSpanEncoder,
    SpanCrfParser,
    TaggedChartScorer,
)
from lofi.nn.policy import DevicePolicy
from lofi.nn.ports import Observable, StandardDeclaration
from lofi.nn.spaces import Space
from lofi.nn.tokens.decoder import RecurrentSpanDecoder
from lofi.nn.tokens.embedder import (
    ReservedTokens,
    StaticTokenEmbedder,
    TokenVocabulary,
)
from lofi.nn.tokens.pool import (
    AttentionSubwordPooler,
    FirstSubwordPooler,
    MeanSubwordPooler,
)
from lofi.nn.tokens.pretrained import PretrainedTokenEncoder
from lofi.registry import ComponentPlugin, entry_point_group
from lofi.spec import (
    COMPONENT_FAMILIES,
    ControllerSpec,
    ParserSpec,
    PatternRuleSpec,
    ReadoutSpec,
    RulePlugin,
    RuleSetSpec,
    SpanDecoderSpec,
    SubwordPoolerSpec,
    TermDecoderSpec,
    TermEncoderSpec,
    TermPatternSpec,
    TokenEncoderSpec,
    component_groups,
)
from lofi.spec.artifacts import HubSourceSpec
from lofi.spec.controllers import (
    BilinearControllerSpec,
    MlpControllerSpec,
    TransformerControllerSpec,
)
from lofi.spec.decoders.diffusion import DiffusionTermDecoderSpec
from lofi.spec.decoders.hrr import HrrTermDecoderSpec
from lofi.spec.decoders.linear import LinearRnnDecoderSpec
from lofi.spec.decoders.rnn import RnnTermDecoderSpec
from lofi.spec.decoders.transformer import TransformerTermDecoderSpec
from lofi.spec.encoders.hrr import HrrTermEncoderSpec
from lofi.spec.encoders.linear import LinearRnnEncoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec
from lofi.spec.encoders.transformer import TransformerTermEncoderSpec
from lofi.spec.layers import FeedForwardLayerSpec
from lofi.spec.parsers.chart import InsideOutsideParserSpec
from lofi.spec.parsers.flat import FlatParserSpec
from lofi.spec.parsers.span_crf import SpanCrfParserSpec
from lofi.spec.parsers.tagged import TaggedParserSpec
from lofi.spec.poolers.attention import AttentionSubwordPoolerSpec, SubwordContextSpec
from lofi.spec.poolers.first import FirstSubwordPoolerSpec
from lofi.spec.poolers.mean import MeanSubwordPoolerSpec
from lofi.spec.readouts import ReadoutSource, TargetSource, TermsReadoutSpec
from lofi.spec.span_decoders.rnn import RnnSpanDecoderSpec
from lofi.spec.tokens.learned import LearnedTokenEncoderSpec
from lofi.spec.tokens.llm2vec import LLM2VecTokenEncoderSpec
from lofi.spec.tokens.pretrained import ContextWindowSpec, PretrainedTokenEncoderSpec
from lofi.spec.tokens.static import StaticTokenEncoderSpec
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
from lofi.training_plugins import TrainerPlugin

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence
    from pathlib import Path

    from torch import Tensor

    from lofi._artifacts import PinnedCheckpoint
    from lofi.nn.ports import Controller
    from lofi.nn.tokens.pretrained import (
        DecoderModel,
        EncoderArchitecturePlugin,
    )
    from lofi.spec import ConfiguredSpec


WIDTH: Final = 8
HEADS: Final = 2
LAYERS: Final = 1
INNER: Final = 16

ATOM: Final = Operator("atom", "T", family="primitive")
PAIR: Final = Operator("pair", "T", arg_sorts=("T", "T"), family="constructor")
SIGNATURE: Final = Signature("installed-component-plugins", ("T",), (ATOM, PAIR))
CODEC: Final = TermCodec(
    OperatorIndex(SIGNATURE),
    CompleteTree(SIGNATURE, SIGNATURE.operators, order="bf", depth=3),
)
VIEW: Final = NeuralCodecView(CODEC)
VOCABULARY: Final = TokenVocabulary(
    ("dog", "cat"), reserved=ReservedTokens({"<unk>": 0}), unknown="<unk>"
)
STORE: Final = TermStore(SIGNATURE)
MATCHING: Final = Identity[Term]()
POLICY: Final = DevicePolicy(torch.device("cpu"), torch.float64)
IDENTITY: Final = Action("identity")
COMPOSE: Final = Action("compose", "compose")
ACTIONS: Final = ActionSet((IDENTITY, COMPOSE), IDENTITY)
_ADOPTED: Final = declared_rules(
    ccg_signature("atom"),
    RuleSetSpec(
        name="installed-component-plugins",
        rules=(
            PatternRuleSpec(
                name="compose",
                premises=(
                    TermPatternSpec(name="left", sort="Cat"),
                    TermPatternSpec(name="right", sort="Cat"),
                ),
                conclusion=TermPatternSpec(name="left", sort="Cat"),
            ),
        ),
    ),
)
RULES: Final = _ADOPTED.rule_set
CATEGORY: Final = _ADOPTED.context.syntax.parse(_ADOPTED.context.store, "atom")
GOAL: Final = TargetSet((CATEGORY,), RULES.matching)
TARGETS: Final = TargetSet((STORE.op(ATOM),), MATCHING)


class _Configuration:
    def __init__(self, hidden_size: int) -> None:
        self.hidden_size = hidden_size
        self.num_hidden_layers = 1


class _Stack(nn.Module):
    def __init__(self, hidden_size: int) -> None:
        super().__init__()
        self.config = _Configuration(hidden_size)
        self.projection = nn.Linear(hidden_size, hidden_size)


class _Batch:
    def __init__(
        self, input_ids: Tensor, attention_mask: Tensor, offset_mapping: Tensor
    ) -> None:
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.offset_mapping = offset_mapping


class _Tokenizer:
    pad_token_id = 0
    model_max_length = 512

    def __call__(
        self,
        texts: Sequence[str],
        *,
        padding: bool,
        return_tensors: str,
        truncation: bool,
        return_offsets_mapping: bool,
    ) -> _Batch:
        del padding, return_tensors, truncation, return_offsets_mapping
        items = [text.split(" ") if text else [] for text in texts]
        longest = max((len(words) for words in items), default=0)
        rows = [
            [index + 1 for index in range(len(words))] + [0] * (longest - len(words))
            for words in items
        ]
        offsets = torch.zeros((len(items), longest, 2), dtype=torch.long)
        tensor = torch.tensor(rows, dtype=torch.long)

        return _Batch(tensor, tensor.ne(self.pad_token_id).to(torch.long), offsets)


class _Loader:
    def __init__(self, hidden_size: int) -> None:
        self._stack = _Stack(hidden_size)

    def load_encoder(
        self,
        checkpoint: PinnedCheckpoint,
        *,
        architecture: EncoderArchitecturePlugin,
        dtype: torch.dtype,
    ) -> _Stack:
        del checkpoint, architecture, dtype

        return self._stack

    def load_decoder(
        self, checkpoint: PinnedCheckpoint, *, dtype: torch.dtype
    ) -> DecoderModel:
        words = tuple(f"word-{index}" for index in range(16))

        return StandInLoader(words, width=WIDTH, seed=0).load_decoder(
            checkpoint, dtype=dtype
        )

    def load_tokenizer(self, checkpoint: PinnedCheckpoint) -> _Tokenizer:
        del checkpoint

        return _Tokenizer()


def context(
    role: str,
    producers: Mapping[str, object],
    targets: Mapping[str, TargetSet[Term]],
) -> BuildContext:
    """Return the fixture run's context for one component."""
    return plugin_context(
        component=role,
        policy=POLICY,
        codec=VIEW,
        vocabulary=VOCABULARY,
        actions=ACTIONS,
        rules=RULES,
        categories=(CATEGORY,),
        producers=producers,
        targets=targets,
        loader=_Loader(WIDTH),
    )


def built(declaration: ConfiguredSpec[object], role: str) -> nn.Module:
    """Return a producer built through its plugin in the fixture run."""
    return cast("nn.Module", build_component(declaration, context(role, {}, {})))


type _TermDecoder = (
    HolographicTermDecoder
    | LinearRnnDecoder
    | RecursiveTermDecoder
    | TransformerTermDecoder
    | DiffusionTermDecoder
)
type _TermEncoder = (
    HrrTermEncoder | LinearRnnEncoder | TreeRnnEncoder | TransformerTermEncoder
)
type _Case = tuple[
    ConfiguredSpec[object],
    type,
    Mapping[str, object],
    Callable[[object], Mapping[str, bool]],
]


def _component(role: str) -> Callable[[object], Mapping[str, bool]]:
    """Return an observation that the built module names its assigned handle."""

    def observe(module: object) -> Mapping[str, bool]:
        return {"component": cast("Observable", module).site_map().component == role}

    return observe


def _cases(vectors: Path) -> Mapping[str, _Case]:
    """Return one declaration per installed variant with its producers."""
    tokens = built(
        StaticTokenEncoderSpec(vectors=str(vectors), trainable=True), "tokens"
    )
    pooler = cast(
        "FirstSubwordPooler",
        build_component(
            FirstSubwordPoolerSpec(), context("pooler", {"encoder": tokens}, {})
        ),
    )
    hrr = built(HrrTermEncoderSpec(width=WIDTH), "hrr")
    linear = built(LinearRnnEncoderSpec(width=WIDTH), "linear")
    rnn = built(TreeRnnEncoderSpec(width=WIDTH), "rnn")
    controller = cast(
        "MlpController",
        build_component(
            MlpControllerSpec(
                hidden=(FeedForwardLayerSpec(width=4, activation="tanh"),)
            ),
            context("controller", {"cells": rnn}, {}),
        ),
    )
    decoder = cast(
        "RecursiveTermDecoder",
        build_component(
            RnnTermDecoderSpec(width=WIDTH),
            context("decoder", {"encoder": rnn}, {}),
        ),
    )

    def static(module: object) -> Mapping[str, bool]:
        encoder = cast("StaticTokenEmbedder", module)

        return {
            "vocabulary": encoder.vocabulary is VOCABULARY,
            "file width": encoder.output_width == 2,
            "trainable": all(p.requires_grad for p in encoder.parameters()),
        }

    def learned(module: object) -> Mapping[str, bool]:
        encoder = cast("StaticTokenEmbedder", module)

        return {"width": encoder.output_width == WIDTH}

    def pretrained(module: object) -> Mapping[str, bool]:
        encoder = cast("PretrainedTokenEncoder", module)

        return {
            "checkpoint": encoder.checkpoint.identity == f"fixture@{'e' * 40}",
            "window": encoder.window is not None and encoder.window.rows == 6,
            "trainable": encoder.trainable,
        }

    def pooled(module: object) -> Mapping[str, bool]:
        return {
            "encoder width": cast("FirstSubwordPooler", module).output_width
            == cast("StaticTokenEmbedder", tokens).output_width
        }

    def attention(module: object) -> Mapping[str, bool]:
        state = cast("AttentionSubwordPooler", module).state_dict()

        return {
            "context": "_context.weight_ih_l0_reverse" in state,
            "scorer": "_query.3.weight" in state,
        }

    def encoder(module: object) -> Mapping[str, bool]:
        held = cast("_TermEncoder", module)

        return {"codec": held.codec is VIEW, "width": held.output_width == WIDTH}

    def term_decoder(producer: nn.Module) -> Callable[[object], Mapping[str, bool]]:
        def observe(module: object) -> Mapping[str, bool]:
            held = cast("_TermDecoder", module)

            return {
                "codec": held.codec is VIEW,
                "input width": held.input_width
                == cast("TreeRnnEncoder", producer).output_width,
            }

        return observe

    def transformer_decoder(module: object) -> Mapping[str, bool]:
        evidence = dict(term_decoder(rnn)(module))
        held = cast("TransformerTermDecoder", module)
        evidence["structure"] = (
            held.structure == StandardDeclaration.ANCESTOR_AND_SIBLING_BIAS.declaration
        )

        return evidence

    def chart(module: object) -> Mapping[str, bool]:
        held = cast("DeepInsideOutside", module)

        return {"rules": held.rules.action_set is controller.action_set}

    def tagged(module: object) -> Mapping[str, bool]:
        held = cast("TaggedChartScorer", module)

        return {
            "rules": held.rules.action_set is controller.action_set,
            "unary depth": held.kernel.transitions.closure.depth == 2,
            "k best": held.decoder.k == 3,
        }

    def flat(module: object) -> Mapping[str, bool]:
        span = cast("FlatSpanEncoder", module)

        return {
            "width": span.width == 3,
            "features": span.features == ("bilinear",),
            "initialization": span.projection_initialization == "identity",
        }

    def span_crf(module: object) -> Mapping[str, bool]:
        parser = cast("SpanCrfParser", module)

        return {
            "input width": parser.input_width == pooler.output_width,
            "hidden width": parser.hidden == 5,
            "global outputs": parser.publishes
            == {"inside", "outside", "alpha", "mu", "log_z", "entropy", "mbr"},
        }

    def acting(module: object) -> Mapping[str, bool]:
        held = cast("Controller", module)

        return {
            "actions": held.action_set is ACTIONS,
            "width": held.input_width == WIDTH,
        }

    def span(module: object) -> Mapping[str, bool]:
        return {"width": cast("RecurrentSpanDecoder", module).input_width == WIDTH}

    space = Space(
        "T",
        codec=VIEW,
        choices=context("space", {}, {}).choices,
        store=STORE,
        max_nodes=VIEW.capacity,
        encoder=cast("TreeRnnEncoder", rnn),
        identity=decoder,
        combinators=None,
    )

    def readout(module: object) -> Mapping[str, bool]:
        held = cast("TermsReadout", module)

        return {"key": held.key == "types", "codec": held.codec is VIEW}

    return MappingProxyType(
        {
            "token_encoder:static": (
                StaticTokenEncoderSpec(vectors=str(vectors), trainable=True),
                StaticTokenEmbedder,
                {},
                static,
            ),
            "token_encoder:learned": (
                LearnedTokenEncoderSpec(width=WIDTH),
                StaticTokenEmbedder,
                {},
                learned,
            ),
            "token_encoder:pretrained": (
                PretrainedTokenEncoderSpec(
                    checkpoint=HubSourceSpec(repository="fixture", revision="e" * 40),
                    trainable=True,
                    window=ContextWindowSpec(rows=6, overflow="refuse"),
                ),
                PretrainedTokenEncoder,
                {},
                pretrained,
            ),
            "token_encoder:llm2vec": (
                LLM2VecTokenEncoderSpec(
                    checkpoint=HubSourceSpec(repository="fixture", revision="e" * 40),
                    trainable=True,
                    window=ContextWindowSpec(rows=6, overflow="refuse"),
                ),
                PretrainedTokenEncoder,
                {},
                pretrained,
            ),
            "subword_pooler:first": (
                FirstSubwordPoolerSpec(),
                FirstSubwordPooler,
                {"encoder": tokens},
                pooled,
            ),
            "subword_pooler:mean": (
                MeanSubwordPoolerSpec(),
                MeanSubwordPooler,
                {"encoder": tokens},
                pooled,
            ),
            "subword_pooler:attention": (
                AttentionSubwordPoolerSpec(
                    context=SubwordContextSpec(),
                    scorer=(FeedForwardLayerSpec(width=4, activation="tanh"),),
                ),
                AttentionSubwordPooler,
                {"encoder": tokens},
                attention,
            ),
            "term_encoder:hrr": (
                HrrTermEncoderSpec(width=WIDTH),
                HrrTermEncoder,
                {},
                encoder,
            ),
            "term_encoder:linear-rnn": (
                LinearRnnEncoderSpec(width=WIDTH),
                LinearRnnEncoder,
                {},
                encoder,
            ),
            "term_encoder:tree-rnn": (
                TreeRnnEncoderSpec(width=WIDTH),
                TreeRnnEncoder,
                {},
                encoder,
            ),
            "term_encoder:transformer": (
                TransformerTermEncoderSpec(
                    width=WIDTH, heads=HEADS, layers=LAYERS, feedforward_width=INNER
                ),
                TransformerTermEncoder,
                {},
                encoder,
            ),
            "term_decoder:hrr": (
                HrrTermDecoderSpec(),
                HolographicTermDecoder,
                {"encoder": hrr},
                term_decoder(hrr),
            ),
            "term_decoder:linear-rnn": (
                LinearRnnDecoderSpec(width=WIDTH),
                LinearRnnDecoder,
                {"encoder": linear},
                term_decoder(linear),
            ),
            "term_decoder:tree-rnn": (
                RnnTermDecoderSpec(width=WIDTH),
                RecursiveTermDecoder,
                {"encoder": rnn},
                term_decoder(rnn),
            ),
            "term_decoder:transformer": (
                TransformerTermDecoderSpec(
                    width=WIDTH,
                    heads=HEADS,
                    layers=LAYERS,
                    feedforward_width=INNER,
                    structure="ancestor-and-sibling-bias",
                ),
                TransformerTermDecoder,
                {"encoder": rnn},
                transformer_decoder,
            ),
            "term_decoder:diffusion": (
                DiffusionTermDecoderSpec(
                    width=WIDTH,
                    layers=LAYERS,
                    feedforward_width=INNER,
                    noise_steps=4,
                    first_masking=0.2,
                ),
                DiffusionTermDecoder,
                {"encoder": rnn},
                term_decoder(rnn),
            ),
            "controller:mlp": (
                MlpControllerSpec(
                    hidden=(FeedForwardLayerSpec(width=4, activation="tanh"),)
                ),
                MlpController,
                {"cells": rnn},
                acting,
            ),
            "controller:bilinear": (
                BilinearControllerSpec(features=("left", "product")),
                BilinearController,
                {"cells": rnn},
                acting,
            ),
            "controller:transformer": (
                TransformerControllerSpec(
                    heads=2,
                    feedforward_width=7,
                    activation="relu",
                ),
                TransformerController,
                {"cells": rnn},
                acting,
            ),
            "chart:inside-outside": (
                InsideOutsideParserSpec(width=WIDTH),
                DeepInsideOutside,
                {"leaves": pooler, "rules": controller},
                chart,
            ),
            "chart:tagged": (
                TaggedParserSpec(width=WIDTH, unary_depth=2, k_best=3),
                TaggedChartScorer,
                {"leaves": pooler, "rules": controller},
                tagged,
            ),
            "chart:flat": (
                FlatParserSpec(
                    width=3,
                    features=("bilinear",),
                    projection_initialization="identity",
                ),
                FlatSpanEncoder,
                {"leaves": pooler},
                flat,
            ),
            "chart:span-crf": (
                SpanCrfParserSpec(hidden=5),
                SpanCrfParser,
                {"leaves": pooler},
                span_crf,
            ),
            "span_decoder:rnn-span-decoder": (
                RnnSpanDecoderSpec(),
                RecurrentSpanDecoder,
                {"cells": rnn},
                span,
            ),
            "readout:terms": (
                TermsReadoutSpec(
                    space="categories",
                    reads=ReadoutSource.INSIDE,
                    targets=TargetSource.GRAMMAR_GOAL,
                    output="sentences",
                ),
                TermsReadout,
                {"identity": decoder, "space": space},
                readout,
            ),
        }
    )


MODEL_GROUPS: Final = tuple(
    f"lofi.{root.family()}"
    for root in (
        TokenEncoderSpec,
        SubwordPoolerSpec,
        TermEncoderSpec,
        TermDecoderSpec,
        ControllerSpec,
        ParserSpec,
        ReadoutSpec,
        SpanDecoderSpec,
    )
)


def _installed() -> tuple[str, ...]:
    """Return every installed entry of the model groups as family:tag."""
    return tuple(
        f"{group.removeprefix('lofi.')}:{point.name}"
        for group in MODEL_GROUPS
        for point in sorted(entry_points(group=group), key=lambda point: point.name)
    )


@pytest.fixture(params=_installed())
def component_plugin_fixture(
    request: pytest.FixtureRequest, tmp_path: Path
) -> ComponentPluginFixture:
    """Return one installed variant's declaration in the fixture run."""
    case = cast("str", request.param)
    vectors = tmp_path / "vectors.txt"
    vectors.write_text("dog 1.0 2.0\ncat 3.0 4.0\n", encoding="utf-8", newline="\n")
    declaration, expected, producers, observe = _cases(vectors)[case]
    role = "types" if case == "readout:terms" else case.split(":")[0]
    targets = {
        "chart:tagged": {"grammar-goal": GOAL},
        "readout:terms": {"grammar-goal": TARGETS},
    }.get(case, {})

    def observed(module: object) -> Mapping[str, bool]:
        evidence = dict(observe(module))

        if isinstance(module, Observable):
            evidence.update(_component(role)(module))

        return evidence

    return ComponentPluginFixture(
        declaration, context(role, producers, targets), expected, observed
    )


class TestInstalledComponentPlugins(ComponentPluginConformance):
    """Conformance tests for every installed component plugin."""


def test_every_component_family_has_its_own_group() -> None:
    assert component_groups() == tuple(f"lofi.{name}" for name in COMPONENT_FAMILIES)


def test_every_installed_entry_names_a_plugin_for_the_variant_its_tag_selects() -> None:
    for group in (entry_point_group("rule"), *component_groups()):
        for point in entry_points(group=group):
            plugin = point.load()

            assert isinstance(
                plugin,
                ComponentPlugin
                | ArtifactDatasetPlugin
                | DatasetPlugin
                | RulePlugin
                | TrainerPlugin,
            ), point

            if isinstance(plugin, RulePlugin):
                assert group == "lofi.rule"
                assert plugin.spec.__field_specs__["rule"].default == point.name
            else:
                assert plugin.spec.family() == group.removeprefix("lofi.")
                assert plugin.spec.__field_specs__["kind"].default == point.name
