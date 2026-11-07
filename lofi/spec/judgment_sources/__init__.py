"""Declared sources of judgment datasets."""

from lofi.spec.judgment_sources.predicted import (
    PredictedJudgmentSourceSettings,
    PredictedJudgmentSourceSpec,
    decode_predicted_judgment_source,
)

__all__ = [
    "PredictedJudgmentSourceSettings",
    "PredictedJudgmentSourceSpec",
    "decode_predicted_judgment_source",
]
