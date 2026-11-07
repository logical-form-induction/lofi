"""Composability of n-ary terms, layouts, and neural decoders."""

from typing import TYPE_CHECKING, Final, cast

import pytest
import torch
from tests.conformance.component_plugin import plugin_context

from lofi.nn.arity import ArityBound, ArityUnsupportedError
from lofi.nn.build import build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import HolographicTermDecoder, RecursiveTermDecoder
from lofi.nn.encoders import HrrTermEncoder
from lofi.nn.ports import TeacherForcedDecoder
from lofi.nn.sites import ComponentId
from lofi.nn.strategies import score
from lofi.spec import ConstructorSpec, PrefixConstructorNotationSpec
from lofi.spec.decoders.hrr import HrrTermDecoderSpec
from lofi.spec.decoders.rnn import RnnTermDecoderSpec
from lofi.spec.encoders.hrr import HrrTermEncoderSpec
from lofi.spec.grammar import SignatureSpec
from lofi.spec.language import (
    LanguageAssembly,
    LanguageIso,
    SortBoundSpec,
    registered_language,
)
from lofi.terms import (
    CompleteTree,
    PreorderArityLayout,
)

if TYPE_CHECKING:
    from collections.abc import Callable

type DecoderFactory = Callable[
    [NeuralCodecView, HrrTermEncoder, LanguageAssembly], TeacherForcedDecoder
]

WIDTH: Final = 8
ENCODER: Final = ComponentId("encoder")
DECODER: Final = ComponentId("decoder")

SIGNATURE_DECLARATION: Final = SignatureSpec(
    name="ternary-capabilities",
    sorts=("T",),
    category_sort="T",
    primitives=("a", "b", "c"),
    constructors=(
        ConstructorSpec(
            name="triple",
            kind=("T", "T", "T", "T"),
            notation=PrefixConstructorNotationSpec(spelling="triple"),
        ),
    ),
    syntax="s-expression",
)


def _language(category_bound: SortBoundSpec, /) -> LanguageAssembly:
    language = registered_language("propositional")
    renamed = language.with_(
        category_sort="T",
        bounds={
            language.type_sort: language.bounds[language.type_sort],
            language.term_sort: language.bounds[language.term_sort],
            "T": category_bound,
        },
    )
    return LanguageIso().extend(renamed, SIGNATURE_DECLARATION.extend)


def _holographic_decoder(
    codec: NeuralCodecView,
    encoder: HrrTermEncoder,
    language: LanguageAssembly,
) -> TeacherForcedDecoder:
    """Build the holographic decoder sharing its encoder's codebook.

    Parameters
    ----------
    codec : NeuralCodecView
        The tensor view of the preorder codec.
    encoder : HrrTermEncoder
        The matching encoder that owns the shared codebook.
    """
    return cast(
        "TeacherForcedDecoder",
        build_component(
            HrrTermDecoderSpec(),
            plugin_context(
                codec=codec,
                language=language,
                component=DECODER,
                producers={"encoder": encoder},
            ),
        ),
    )


def _recursive_decoder(
    codec: NeuralCodecView,
    encoder: HrrTermEncoder,
    language: LanguageAssembly,
) -> TeacherForcedDecoder:
    """Build the recursive decoder over a complete-tree codec.

    Parameters
    ----------
    codec : NeuralCodecView
        The tensor view of the complete-tree codec.
    encoder : HrrTermEncoder
        The encoder supplying the decoder's input width.
    """
    return cast(
        "TeacherForcedDecoder",
        build_component(
            RnnTermDecoderSpec(width=WIDTH),
            plugin_context(
                codec=codec,
                language=language,
                component=DECODER,
                producers={"encoder": encoder},
            ),
        ),
    )


@pytest.mark.parametrize(
    ("category_bound", "layout_type", "decoder_factory", "decoder_type"),
    [
        pytest.param(
            SortBoundSpec(layout="preorder-arity", depth=2, nodes=4),
            PreorderArityLayout,
            _holographic_decoder,
            HolographicTermDecoder,
            id="preorder-arity+holographic",
        ),
        pytest.param(
            SortBoundSpec(layout="complete-tree", order="bf", depth=2),
            CompleteTree,
            _recursive_decoder,
            RecursiveTermDecoder,
            id="complete-tree+recursive",
        ),
    ],
)
def test_ternary_terms_run_through_supported_layout_and_decoder_paths(
    category_bound: SortBoundSpec,
    layout_type: type[CompleteTree | PreorderArityLayout],
    decoder_factory: DecoderFactory,
    decoder_type: type[object],
) -> None:
    language = _language(category_bound)
    signature = language.store.signature
    operators = {operator.name: operator for operator in signature.operators}
    store = language.store
    vocabulary = signature.vocabulary("T")
    constant = signature.constant("T")
    term = store.op(
        operators["triple"],
        store.op(constant, payload=vocabulary.id("a")),
        store.op(constant, payload=vocabulary.id("b")),
        store.op(constant, payload=vocabulary.id("c")),
    )
    codec = language.codec.at("T")
    layout = codec.layout
    terms = codec.encode_batch((term,))
    view = NeuralCodecView(codec)
    encoder = cast(
        "HrrTermEncoder",
        build_component(
            HrrTermEncoderSpec(width=WIDTH),
            plugin_context(codec=view, language=language, component=ENCODER),
        ),
    )
    decoder = decoder_factory(view, encoder, language)
    scored = score(decoder, encoder.encode(terms), terms, index=language.codec.choices)

    assert codec.decode(terms.rows[0], store) is term
    assert type(layout) is layout_type
    assert isinstance(decoder, decoder_type)
    assert scored.axes == ("term",)
    assert tuple(scored.tensor.shape) == (1,)
    assert bool(torch.isfinite(scored.tensor).all())


def test_a_rank_indexed_family_refuses_only_an_omitted_ternary_rank() -> None:
    signature = _language(
        SortBoundSpec(layout="complete-tree", depth=2)
    ).store.signature
    bound = ArityBound((1, 2))

    with pytest.raises(ArityUnsupportedError) as refusal:
        bound.served(signature.operators)

    message = str(refusal.value)

    assert bound.reason in message
    assert "arity 3" in message
    assert "'triple'" in message
