"""Tests for baseline-fit boundary models and codecs."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Literal

import didactic.api as dx
import pytest

from lofi.run._boundary.baselines import (
    BaselineFitCodec,
    BaselineFitManifest,
    BaselineFitSpec,
    decode_baseline_manifest,
    encode_baseline_manifest,
)
from lofi.run._boundary.manifest import encode_seed_derivation
from lofi.run._seed import ExperimentSeed
from lofi.spec.baselines import BiskHockenmaierBaselineSpec
from lofi.spec.components.baseline import BaselineSystemSpec

if TYPE_CHECKING:
    from pathlib import Path

    from lofi._types import JsonValue


class EmptyBaseline(BaselineSystemSpec):
    """A test system that promises no report row."""

    kind: Literal["empty-boundary-test"] = "empty-boundary-test"


def document() -> dict[str, JsonValue]:
    """Return one complete baseline-fit document."""
    return {
        "system": {
            "kind": "bisk-hockenmaier",
            "ccgbank_source": "licensed-ccgbank-auto",
            "dependency_gold_source": "licensed-johansson-nugues-conll",
            "reference_source": "lofi.resource:configs/eval/bisk_hockenmaier_2012.json",
            "reference_tolerance": 0.02,
            "nominal_category": "N",
            "sentence_category": "S",
            "coordination_category": "conj",
            "gold_sentence_atom": "S",
            "gold_coordination_atom": "conj",
            "nominal_tags": ["NN"],
            "verbal_tags": ["VB", "VBG"],
            "coordination_tags": ["CC"],
            "other_tags": ["JJ", "."],
            "punctuation_tags": ["."],
            "headed_or_modifier_arguments": ["N", "S"],
            "sentence_arguments": ["N"],
            "em_max_iterations": 50,
            "convergence_tolerance": 1e-6,
        }
    }


def test_the_codec_composes_and_traces_a_baseline_fit(tmp_path: Path) -> None:
    path = tmp_path / "fit.json"
    path.write_text(json.dumps(document()), encoding="utf-8", newline="\n")
    codec = BaselineFitCodec()

    assert BiskHockenmaierBaselineSpec.family() == "baseline_system"

    composition = codec.compose_traced(path, overrides=("system.em_max_iterations=4",))

    assert composition.spec.system.kind == "bisk-hockenmaier"
    assert composition.spec.system.metrics() == (
        "supertagging-length-10",
        "dependencies-length-10",
        "supertagging-length-20",
        "dependencies-length-20",
        "supertagging-all-lengths",
        "dependencies-all-lengths",
    )
    assert (
        json.loads(composition.spec.model_dump_json())["system"]["em_max_iterations"]
        == 4
    )
    assert (
        composition.provenance.origin_of("system.em_max_iterations").source.value
        == "override"
    )


def test_a_system_that_promises_no_metric_is_refused() -> None:
    with pytest.raises(dx.ValidationError, match="at least one"):
        BaselineFitSpec(system=EmptyBaseline())


def test_the_manifest_round_trip_is_canonical() -> None:
    fit = BaselineFitSpec.model_validate(document())
    derived = ExperimentSeed(7).point("point", 2)
    manifest = BaselineFitManifest(
        fit=fit,
        source_digests={"ccgbank-auto": f"sha256:{'0' * 64}"},
        seed_derivation=encode_seed_derivation(derived),
        package_versions={"lofi": "0.1"},
        git_revision="0123abcd",
    )

    encoded = encode_baseline_manifest(manifest)

    assert decode_baseline_manifest(encoded) == manifest
    assert encoded == encode_baseline_manifest(decode_baseline_manifest(encoded))
