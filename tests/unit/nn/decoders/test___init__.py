"""Public decoder inventory and neural-source exception discipline."""

import ast
import inspect
from collections.abc import Callable
from pathlib import Path
from typing import Final, NamedTuple

import pytest

import lofi.nn as neural
import lofi.nn.decoders as decoder_api
from lofi.nn.codec import NeuralCodecView
from lofi.nn.decoders import (
    DiffusionDecoderSettings,
    DiffusionTermDecoder,
    HolographicTermDecoder,
    LinearRnnDecoder,
    NoiseSchedule,
    RecursiveTermDecoder,
    TransformerDecoderSettings,
    TransformerTermDecoder,
    TreeDecoderSettings,
)
from lofi.nn.encoders import HolographicCodebook
from lofi.nn.ports import StandardDeclaration
from lofi.nn.sites import ComponentId
from lofi.nn.type_directed import UNRESTRICTED_ADMISSION
from lofi.spec.decoders.linear import LinearRnnDecoderSettings
from lofi.terms import (
    ChoiceIndex,
    CompleteTree,
    Operator,
    OperatorIndex,
    Signature,
    TermCodec,
)

type ShippedDecoder = (
    DiffusionTermDecoder
    | HolographicTermDecoder
    | LinearRnnDecoder
    | RecursiveTermDecoder
    | TransformerTermDecoder
)
type DecoderClass = type[ShippedDecoder]
type Builder = Callable[[], ShippedDecoder]


class Architecture(NamedTuple):
    """A shipped decoder class beside a construction using the shared codec."""

    decoder: DecoderClass
    build: Builder


SIGNATURE: Final = Signature(
    "decoder-inventory",
    ("Cat",),
    (
        Operator("N", "Cat", family="primitive"),
        Operator("S", "Cat", family="primitive"),
        Operator("/", "Cat", arg_sorts=("Cat", "Cat"), family="constructor"),
    ),
)
INDEX: Final = OperatorIndex(SIGNATURE)
CHOICES: Final = ChoiceIndex(SIGNATURE, depth_bounds={"Cat": 2})
LAYOUT: Final = CompleteTree(SIGNATURE, SIGNATURE.operators, order="df", depth=2)
VIEW: Final = NeuralCodecView(TermCodec(INDEX, LAYOUT))
WIDTH: Final = 4
NN_ROOT: Final = Path(inspect.getfile(neural)).parent
BLANKET: Final = frozenset({"<bare>", "BaseException", "Exception"})


def diffusion() -> DiffusionTermDecoder:
    """Return the shipped diffusion decoder over the shared codec."""
    return DiffusionTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=ComponentId("diffusion"),
        settings=DiffusionDecoderSettings(WIDTH, WIDTH, 1, WIDTH, "gelu"),
        schedule=NoiseSchedule((1.0,)),
        choice_heads="flat",
    )


def holographic() -> HolographicTermDecoder:
    """Return the shipped holographic decoder over the shared codec."""
    codebook = HolographicCodebook(
        symbols=VIEW.n_symbols,
        capacity=VIEW.capacity,
        width=WIDTH,
        magnitude=1.0,
        floor=1e-3,
    )
    return HolographicTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=ComponentId("holographic"),
        codebook=codebook,
        choice_heads="flat",
    )


def linear() -> LinearRnnDecoder:
    """Return the shipped linear-recurrent decoder over the shared codec."""
    return LinearRnnDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=ComponentId("linear"),
        input_width=WIDTH,
        settings=LinearRnnDecoderSettings(WIDTH, "gru", 1, "flat", None),
        embedding=None,
    )


def recursive() -> RecursiveTermDecoder:
    """Return the shipped recursive decoder over the shared codec."""
    return RecursiveTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        admission=UNRESTRICTED_ADMISSION,
        component=ComponentId("recursive"),
        settings=TreeDecoderSettings(WIDTH, WIDTH),
        choice_heads="flat",
    )


def transformer() -> TransformerTermDecoder:
    """Return the shipped Transformer decoder over the shared codec."""
    return TransformerTermDecoder(
        codec=VIEW,
        choices=CHOICES,
        component=ComponentId("transformer"),
        settings=TransformerDecoderSettings(WIDTH, WIDTH, 1, 1, WIDTH, "gelu"),
        structure=StandardDeclaration.PARENT_RECURRENCE.declaration,
        choice_heads="flat",
    )


ARCHITECTURES: Final = (
    Architecture(DiffusionTermDecoder, diffusion),
    Architecture(HolographicTermDecoder, holographic),
    Architecture(LinearRnnDecoder, linear),
    Architecture(RecursiveTermDecoder, recursive),
    Architecture(TransformerTermDecoder, transformer),
)


def _caught_names(caught: ast.expr | None) -> frozenset[str]:
    """Return the terminal names of exceptions caught by one handler."""
    match caught:
        case None:
            return frozenset({"<bare>"})
        case ast.Name(id=name) | ast.Attribute(attr=name):
            return frozenset({name})
        case ast.Tuple(elts=members):
            return frozenset[str]().union(*(_caught_names(each) for each in members))
        case _:
            return frozenset()


def _unsafe_handlers(source: str) -> tuple[frozenset[str], ...]:
    """Return each bare or blanket exception set in a Python source string."""
    unsafe = []

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.ExceptHandler):
            caught = _caught_names(node.type) & BLANKET

            if caught:
                unsafe.append(caught)

    return tuple(unsafe)


def test_the_package_publishes_the_tree_recurrence() -> None:
    assert decoder_api.RecursiveTermDecoder is RecursiveTermDecoder
    assert decoder_api.TreeDecoderSettings is TreeDecoderSettings


def test_the_package_publishes_the_declared_transformer() -> None:
    assert decoder_api.TransformerTermDecoder is TransformerTermDecoder
    assert decoder_api.TransformerDecoderSettings is TransformerDecoderSettings


def test_the_public_decoder_api_exports_exactly_the_five_shipped_architectures() -> (
    None
):
    exported = frozenset(
        name
        for name in decoder_api.__all__
        if name.endswith("TermDecoder") or name == "LinearRnnDecoder"
    )
    enumerated = frozenset(
        architecture.decoder.__name__ for architecture in ARCHITECTURES
    )
    expected = frozenset(
        {
            "DiffusionTermDecoder",
            "HolographicTermDecoder",
            "LinearRnnDecoder",
            "RecursiveTermDecoder",
            "TransformerTermDecoder",
        }
    )

    assert exported == expected
    assert enumerated == expected
    assert len(ARCHITECTURES) == len(expected)


@pytest.mark.parametrize(
    "architecture",
    ARCHITECTURES,
    ids=("diffusion", "holographic", "linear", "recursive", "transformer"),
)
def test_every_shipped_decoder_reaches_a_nonempty_primitive_inventory(
    architecture: Architecture,
) -> None:
    decoder = architecture.build()
    index = decoder.codec.codec.index
    codes = next(
        held
        for (_, family), held in index.family_slices.items()
        if family == "primitive"
    )
    primitives = tuple(index.operator(code) for code in codes)

    assert type(decoder) is architecture.decoder
    assert primitives
    assert all(operator.family == "primitive" for operator in primitives)


def test_the_exception_detector_finds_bare_and_blanket_handlers_only() -> None:
    planted = """
try:
    run()
except:
    recover()
try:
    run()
except Exception:
    recover()
try:
    run()
except (ValueError, builtins.BaseException):
    recover()
try:
    run()
except ValueError:
    recover()
"""

    assert _unsafe_handlers(planted) == (
        frozenset({"<bare>"}),
        frozenset({"Exception"}),
        frozenset({"BaseException"}),
    )


def test_no_neural_module_has_a_bare_or_blanket_exception_handler() -> None:
    found = {
        path.relative_to(NN_ROOT): _unsafe_handlers(path.read_text(encoding="utf-8"))
        for path in NN_ROOT.rglob("*.py")
        if _unsafe_handlers(path.read_text(encoding="utf-8"))
    }

    assert found == {}
