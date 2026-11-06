"""Normalized English MegaAcceptability table encoding."""

from lofi.data.corpora.megaacceptability._codec import (
    NAME,
    MalformedMegaAcceptabilityError,
    MegaAcceptabilityCodec,
    MegaAcceptabilityIso,
    MegaAcceptabilityRow,
    MegaAcceptabilityTable,
    TableVersion,
    parse_table,
)
from lofi.data.corpora.megaacceptability.spec import MegaAcceptabilitySpec

__all__ = [
    "NAME",
    "MalformedMegaAcceptabilityError",
    "MegaAcceptabilityCodec",
    "MegaAcceptabilityIso",
    "MegaAcceptabilityRow",
    "MegaAcceptabilitySpec",
    "MegaAcceptabilityTable",
    "TableVersion",
    "parse_table",
]
