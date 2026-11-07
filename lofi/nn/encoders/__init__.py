"""Shared machinery and implementations for term and span encoders."""

from lofi._types import SharedSubmodule
from lofi.nn.encoders.hrr import (
    HolographicCodebook,
    HolographicEncoderSettings,
    HrrTermEncoder,
)
from lofi.nn.encoders.linear import LinearRnnEncoder
from lofi.nn.encoders.rnn import TreeRnnEncoder
from lofi.nn.encoders.transformer import (
    TransformerEncoderSettings,
    TransformerTermEncoder,
)

__all__ = [
    "HolographicCodebook",
    "HolographicEncoderSettings",
    "HrrTermEncoder",
    "LinearRnnEncoder",
    "SharedSubmodule",
    "TransformerEncoderSettings",
    "TransformerTermEncoder",
    "TreeRnnEncoder",
]
