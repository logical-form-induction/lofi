"""Neural term-codec interfaces and tensor views."""

from lofi.nn.codec._ports import CodecMetadata, TermCoding
from lofi.nn.codec._relations import DeclaredStructure, reflexive_closure
from lofi.nn.codec._structure import DISTRIBUTION_AXES, soft_symbol_rows
from lofi.nn.codec._values import PlacedEncoding
from lofi.nn.codec._view import NeuralCodecView

__all__ = [
    "DISTRIBUTION_AXES",
    "CodecMetadata",
    "DeclaredStructure",
    "NeuralCodecView",
    "PlacedEncoding",
    "TermCoding",
    "reflexive_closure",
    "soft_symbol_rows",
]
