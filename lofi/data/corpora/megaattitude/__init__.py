"""Participant-level MegaAttitude release support."""

from lofi.data.corpora.megaattitude._codec import (
    MEGAINTENSIONALITY_NAME,
    MEGANEGRAISING_NAME,
    MEGAVERIDICALITY_NAME,
    MalformedMegaAttitudeError,
    MegaAttitudeFile,
    MegaAttitudeFormat,
    MegaAttitudeIso,
    MegaIntensionalityCodec,
    MegaIntensionalityRow,
    MegaNegRaisingCodec,
    MegaNegRaisingRow,
    MegaVeridicalityCodec,
    MegaVeridicalityRow,
)
from lofi.data.corpora.megaattitude.spec import (
    MegaAttitudeRelationSpec,
    MegaAttitudeSpec,
)

__all__ = [
    "MEGAINTENSIONALITY_NAME",
    "MEGANEGRAISING_NAME",
    "MEGAVERIDICALITY_NAME",
    "MalformedMegaAttitudeError",
    "MegaAttitudeFile",
    "MegaAttitudeFormat",
    "MegaAttitudeIso",
    "MegaAttitudeRelationSpec",
    "MegaAttitudeSpec",
    "MegaIntensionalityCodec",
    "MegaIntensionalityRow",
    "MegaNegRaisingCodec",
    "MegaNegRaisingRow",
    "MegaVeridicalityCodec",
    "MegaVeridicalityRow",
]
