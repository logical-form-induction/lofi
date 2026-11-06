"""Tests for the baseline declaration package surface."""

from lofi.spec import baselines
from lofi.spec.baselines.bisk_hockenmaier import (
    BiskHockenmaierBaselineSettings,
    BiskHockenmaierBaselineSpec,
)
from lofi.spec.baselines.lm_prover import (
    LmProverBaselineSettings,
    LmProverBaselineSpec,
)


def test_the_package_exports_its_declaration_and_resolved_settings() -> None:
    assert baselines.BiskHockenmaierBaselineSpec is BiskHockenmaierBaselineSpec
    assert baselines.BiskHockenmaierBaselineSettings is BiskHockenmaierBaselineSettings
    assert baselines.LmProverBaselineSpec is LmProverBaselineSpec
    assert baselines.LmProverBaselineSettings is LmProverBaselineSettings
    assert baselines.__all__ == [
        "BiskHockenmaierBaselineSettings",
        "BiskHockenmaierBaselineSpec",
        "FormalizationPipeline",
        "LincPipeline",
        "LincPipelineSpec",
        "LmProverBaselineSettings",
        "LmProverBaselineSpec",
        "LmProverPipelineSpec",
        "LogicLmPipeline",
        "LogicLmPipelineSpec",
    ]
