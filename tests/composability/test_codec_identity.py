"""The one-codec law over every graph the encoder and decoder matrix builds."""

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, cast

import pytest
import torch
from tests.conformance.component_plugin import plugin_context
from tests.unit.spec.grammar.test__signature import slash_views, slashes
from torch import nn

from lofi.nn.build import build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.sharing import module_sharing_matches
from lofi.spec import ApplicationSpec, RuleSetSpec, WiringError
from lofi.spec._wiring import (
    check_realized_sharing,
    resolve_sharing,
)
from lofi.spec.decoders.diffusion import DiffusionTermDecoderSpec
from lofi.spec.decoders.hrr import HrrTermDecoderSpec
from lofi.spec.decoders.rnn import RnnTermDecoderSpec
from lofi.spec.decoders.transformer import TransformerTermDecoderSpec
from lofi.spec.encoders.hrr import HrrTermEncoderSpec
from lofi.spec.encoders.rnn import TreeRnnEncoderSpec
from lofi.spec.encoders.transformer import TransformerTermEncoderSpec
from lofi.spec.grammar import (
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    HockenmaierBiskNormalFormSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import SortBoundSpec, registered_language

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import SubmoduleDeclaration
    from lofi.nn.sharing import (
        ModuleSharingProducer,
        ModuleSharingSubject,
        StateTensor,
    )
    from lofi.spec import ComponentSpec, ConfiguredSpec
    from lofi.spec.grammar import GrammarAssembly

type Producer = ModuleSharingProducer[nn.Parameter, StateTensor]
type Subject = ModuleSharingSubject[nn.Parameter, StateTensor]
type Grammar = GrammarAssembly
type TagPair = tuple[str, str]

ENCODER_ROLE: Final = "terms"
DECODER_ROLE: Final = "decoder"
ENCODER_PORT: Final = "encoder"
CODEBOOK: Final = "codebook"

WIDTH: Final = 8
HEADS: Final = 2
LAYERS: Final = 1
FEEDFORWARD_WIDTH: Final = 16
NOISE_STEPS: Final = 4
FIRST_MASKING: Final = 0.2
SEED: Final = 11

DEPTH: Final = 3
BREADTH_FIRST: Final = "bf"
DEPTH_FIRST: Final = "df"
ORDERS: Final = (BREADTH_FIRST, DEPTH_FIRST)

NO_CODEBOOK: Final = f"publishes no {CODEBOOK} submodule"


class CodecHolder(Protocol):
    """A component that reads and writes terms through one codec view."""

    @property
    def codec(self) -> NeuralCodecView:
        """Return the codec view the component was built with."""


def _declaration() -> GrammarDecl:
    """Return the toy grammar compiled against either language slot order."""
    return GrammarDecl(
        name="codec-identity",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "N", "conj"),
            operators=(
                OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
                OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
            ),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(
            name="application",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="john", category="NP"),
                LexicalEntrySpec(form="sleeps", category="S[dcl]\\NP"),
            )
        ),
        goal=RootGoalSpec(categories=("S[dcl]",)),
        matching=FeatureErasureMatchingSpec(),
        normal_form=HockenmaierBiskNormalFormSpec(),
    )


def _compile(order: Literal["df", "bf"], /) -> Grammar:
    """Compile the toy grammar and return the assembly that owns its codec."""
    declaration = _declaration()
    language_declaration = registered_language("propositional")
    bounds = dict(language_declaration.bounds)
    bounds[language_declaration.category_sort] = SortBoundSpec(
        layout="complete-tree", depth=DEPTH, order=order
    )
    language = language_declaration.with_(bounds=bounds)

    return GrammarIso().forward(declaration, language)


GRAMMAR: Final = _compile(BREADTH_FIRST)
REBUILT: Final = _compile(BREADTH_FIRST)
REORDERED: Final = _compile(DEPTH_FIRST)

CODEC: Final = GRAMMAR.language.codec.at(GRAMMAR.language.category_sort)
REBUILT_CODEC: Final = REBUILT.language.codec.at(REBUILT.language.category_sort)
REORDERED_CODEC: Final = REORDERED.language.codec.at(REORDERED.language.category_sort)

VIEW: Final = NeuralCodecView(CODEC)
REBUILT_VIEW: Final = NeuralCodecView(REBUILT_CODEC)
REORDERED_VIEW: Final = NeuralCodecView(REORDERED_CODEC)
SECOND_VIEW: Final = NeuralCodecView(CODEC)


def divergent_roles(graph: Graph, /) -> tuple[str, ...]:
    """Return the roles not holding, by identity, the codec the first role holds."""
    held = tuple(
        (role, cast("CodecHolder", graph.component(role)).codec) for role in graph.order
    )
    _, first = held[0]

    return tuple(role for role, codec in held[1:] if codec is not first)


def order_fields(module: nn.Module, /) -> tuple[str, ...]:
    """Return the names under which a module tree keeps a slot order of its own."""
    return tuple(
        name
        for held in module.modules()
        for name, value in vars(held).items()
        if value in ORDERS
    )


class Graph:
    """The two built components of one pair, by role, in construction order."""

    __slots__ = ("_components",)

    def __init__(self, components: Mapping[str, nn.Module]) -> None:
        self._components = dict(components)

    @property
    def order(self) -> tuple[str, ...]:
        """Return the roles in construction order."""
        return tuple(self._components)

    def component(self, role: str, /) -> nn.Module:
        """Return the component filling a role."""
        return self._components[role]


class Registration(NamedTuple):
    """A registered component's declaration."""

    spec: ComponentSpec

    @property
    def tag(self) -> str:
        return cast("str", self.spec.model_dump()["kind"])


HRR_ENCODER: Final = Registration(HrrTermEncoderSpec(width=WIDTH))
RNN_ENCODER: Final = Registration(TreeRnnEncoderSpec(width=WIDTH, layers=LAYERS))
TRANSFORMER_ENCODER: Final = Registration(
    TransformerTermEncoderSpec(
        width=WIDTH, heads=HEADS, layers=LAYERS, feedforward_width=FEEDFORWARD_WIDTH
    )
)

HRR_DECODER: Final = Registration(HrrTermDecoderSpec())
RNN_DECODER: Final = Registration(RnnTermDecoderSpec(width=WIDTH))
TRANSFORMER_DECODER: Final = Registration(
    TransformerTermDecoderSpec(
        width=WIDTH,
        heads=HEADS,
        layers=LAYERS,
        feedforward_width=FEEDFORWARD_WIDTH,
        structure="parent-recurrence",
    )
)
DIFFUSION_DECODER: Final = Registration(
    DiffusionTermDecoderSpec(
        width=WIDTH,
        layers=LAYERS,
        feedforward_width=FEEDFORWARD_WIDTH,
        noise_steps=NOISE_STEPS,
        first_masking=FIRST_MASKING,
    )
)

ENCODERS: Final = (HRR_ENCODER, RNN_ENCODER, TRANSFORMER_ENCODER)
DECODERS: Final = (HRR_DECODER, RNN_DECODER, TRANSFORMER_DECODER, DIFFUSION_DECODER)
ENCODER_TAGS: Final[Mapping[str, Registration]] = {
    registration.tag: registration for registration in ENCODERS
}
DECODER_TAGS: Final[Mapping[str, Registration]] = {
    registration.tag: registration for registration in DECODERS
}

REFUSED: Final[frozenset[TagPair]] = frozenset(
    {
        (RNN_ENCODER.tag, HRR_DECODER.tag),
        (TRANSFORMER_ENCODER.tag, HRR_DECODER.tag),
    }
)
PAIRS: Final = tuple(
    (encoder.tag, decoder.tag) for encoder in ENCODERS for decoder in DECODERS
)
BUILT: Final = tuple(pair for pair in PAIRS if pair not in REFUSED)


def _sharing_matches(
    consumer: nn.Module, producer: nn.Module, declaration: SubmoduleDeclaration | None
) -> bool:
    """Compare the realized sharing of one dependency edge with its declaration."""
    return module_sharing_matches(
        cast("Subject", consumer), cast("Producer", producer), declaration
    )


def _wire(
    pair: TagPair, /, *, encoding: NeuralCodecView, decoding: NeuralCodecView
) -> Graph:
    """Wire one pair, giving the encoder and the decoder the views named for them."""
    encoder_tag, decoder_tag = pair
    encoder = ENCODER_TAGS[encoder_tag]
    decoder = DECODER_TAGS[decoder_tag]
    torch.manual_seed(SEED)
    specifications = {ENCODER_ROLE: encoder.spec, DECODER_ROLE: decoder.spec}
    producers = MappingProxyType({(DECODER_ROLE, ENCODER_PORT): ENCODER_ROLE})
    sharing = resolve_sharing(specifications, producers)
    built_encoder = cast(
        "nn.Module",
        build_component(
            cast("ConfiguredSpec[object]", encoder.spec),
            plugin_context(codec=encoding, component=ENCODER_ROLE),
        ),
    )
    built_decoder = cast(
        "nn.Module",
        build_component(
            cast("ConfiguredSpec[object]", decoder.spec),
            plugin_context(
                codec=decoding,
                component=DECODER_ROLE,
                producers={ENCODER_PORT: built_encoder},
            ),
        ),
    )
    components = {ENCODER_ROLE: built_encoder, DECODER_ROLE: built_decoder}
    check_realized_sharing(components, producers, sharing, _sharing_matches)

    return Graph(components)


def _graph(pair: TagPair, /) -> Graph:
    """Wire one pair over the single codec its grammar owns."""
    return _wire(pair, encoding=VIEW, decoding=VIEW)


class _Linearized(nn.Module):
    """A module that keeps a slot order in its own state."""

    def __init__(self, order: str) -> None:
        super().__init__()
        self._order = order


BUILT_ROWS: Final = [pytest.param(pair, id=":".join(pair)) for pair in BUILT]
REGISTERED_ROWS: Final = [
    *(pytest.param(encoder, id=f"encoder-{encoder.tag}") for encoder in ENCODERS),
    *(pytest.param(decoder, id=f"decoder-{decoder.tag}") for decoder in DECODERS),
]
REFUSED_ROWS: Final = [
    pytest.param(pair, id=":".join(pair)) for pair in sorted(REFUSED)
]


@pytest.mark.parametrize("pair", BUILT_ROWS)
def test_one_codec_per_grammar(pair: TagPair) -> None:
    graph = _graph(pair)
    held = [cast("CodecHolder", graph.component(role)).codec for role in graph.order]

    assert set(graph.order) == {ENCODER_ROLE, DECODER_ROLE}
    assert divergent_roles(graph) == ()
    assert len({id(view) for view in held}) == 1
    assert all(view.codec is CODEC for view in held)


def test_the_matrix_builds_every_pair_it_does_not_refuse() -> None:
    assert len(PAIRS) == len(ENCODERS) * len(DECODERS)
    assert set(BUILT) | REFUSED == set(PAIRS)
    assert set(BUILT) & REFUSED == set()
    assert len(ENCODER_TAGS) == len(ENCODERS)
    assert len(DECODER_TAGS) == len(DECODERS)


@pytest.mark.parametrize("pair", REFUSED_ROWS)
def test_a_refused_pair_builds_no_graph_to_hold_a_codec(pair: TagPair) -> None:
    with pytest.raises(WiringError, match=NO_CODEBOOK):
        _graph(pair)


@pytest.mark.parametrize("pair", BUILT_ROWS)
def test_a_second_codec_of_one_declaration_is_a_counterexample(pair: TagPair) -> None:
    graph = _wire(pair, encoding=VIEW, decoding=REBUILT_VIEW)

    assert GRAMMAR == REBUILT
    assert CODEC is not REBUILT_CODEC
    assert CODEC.skeleton == REBUILT_CODEC.skeleton
    assert CODEC.index.digest == REBUILT_CODEC.index.digest
    assert divergent_roles(graph) == (DECODER_ROLE,)


@pytest.mark.parametrize("pair", BUILT_ROWS)
def test_a_second_view_of_one_codec_is_a_counterexample(pair: TagPair) -> None:
    graph = _wire(pair, encoding=VIEW, decoding=SECOND_VIEW)

    assert SECOND_VIEW is not VIEW
    assert SECOND_VIEW.codec is VIEW.codec
    assert divergent_roles(graph) == (DECODER_ROLE,)


@pytest.mark.parametrize("pair", BUILT_ROWS)
def test_a_breadth_first_parent_over_depth_first_children_is_a_counterexample(
    pair: TagPair,
) -> None:
    graph = _wire(pair, encoding=VIEW, decoding=REORDERED_VIEW)

    assert CODEC.skeleton != REORDERED_CODEC.skeleton
    assert CODEC.capacity == REORDERED_CODEC.capacity
    assert divergent_roles(graph) == (DECODER_ROLE,)


@pytest.mark.parametrize("pair", BUILT_ROWS)
def test_no_module_of_a_built_graph_keeps_a_slot_order(pair: TagPair) -> None:
    graph = _graph(pair)

    for role in graph.order:
        assert order_fields(graph.component(role)) == ()


@pytest.mark.parametrize("order", ORDERS)
def test_a_module_keeping_a_slot_order_is_found_by_the_same_walk(order: str) -> None:
    assert order_fields(_Linearized(order)) == ("_order",)
    assert order_fields(nn.Sequential(nn.Linear(WIDTH, WIDTH), _Linearized(order))) == (
        "_order",
    )


@pytest.mark.parametrize("registration", REGISTERED_ROWS)
def test_only_the_language_sort_bound_names_a_slot_order(
    registration: Registration,
) -> None:
    component = registration.spec.model_dump()
    bound = SortBoundSpec(
        layout="complete-tree", order=BREADTH_FIRST, depth=DEPTH
    ).model_dump()

    assert [value for value in component.values() if value in ORDERS] == []
    assert [value for value in bound.values() if value in ORDERS] == [BREADTH_FIRST]
