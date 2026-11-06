import didactic.api as dx
import pytest

from lofi.data.corpora.megaacceptability import MegaAcceptabilitySpec


def declared(scale_min: int = 1, scale_max: int = 7) -> MegaAcceptabilitySpec:
    return MegaAcceptabilitySpec(
        source_id="synthetic",
        release_id="v2",
        spdx="CC0-1.0",
        frame_refs={
            "NP _ that S": "at://did:plc:test/pub.layers.resource.template/frame"
        },
        verb_refs={"think": "at://did:plc:test/pub.layers.resource.entry/think"},
        scale_min=scale_min,
        scale_max=scale_max,
    )


def test_scale_bounds_must_be_ordered() -> None:
    with pytest.raises(dx.ValidationError, match="minimum exceeds"):
        declared(scale_min=8, scale_max=7)


def test_resource_reference_maps_must_be_present() -> None:
    with pytest.raises(dx.ValidationError, match="references cannot be empty"):
        MegaAcceptabilitySpec(
            source_id="synthetic",
            release_id="v2",
            spdx="CC0-1.0",
            scale_min=1,
            scale_max=7,
            frame_refs={},
            verb_refs={"think": "at://did:plc:test/pub.layers.resource.entry/think"},
        )
