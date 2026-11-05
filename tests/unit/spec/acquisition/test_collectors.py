"""Tests for the declarations of active-learning collectors."""

from typing import TYPE_CHECKING, Final

import didactic.api as dx
import pytest

from lofi.spec.acquisition import (
    ActiveCollectorSpec,
    BeadCollectorSettings,
    BeadCollectorSpec,
    DrawnEffectsCollectorSettings,
    DrawnEffectsCollectorSpec,
    LatentRangeSettings,
    PredictorTruthCollectorSettings,
    PredictorTruthCollectorSpec,
    RegisteredDesignCollectorSettings,
    RegisteredDesignCollectorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue

RANGE: Final[dict[str, JsonValue]] = {"lower": -3.0, "upper": 3.0}


def test_a_bead_collector_resolves_to_its_settings_in_the_collector_family() -> None:
    declared = ActiveCollectorSpec.model_validate({"kind": "bead"})

    assert isinstance(declared, BeadCollectorSpec)
    assert declared.settings() == BeadCollectorSettings("bead")
    assert BeadCollectorSpec.family() == "active_collector"


def test_a_bead_collector_declares_no_other_field() -> None:
    with pytest.raises(dx.ValidationError, match="directory"):
        ActiveCollectorSpec.model_validate({"kind": "bead", "directory": "rounds"})


def test_a_bead_collector_does_not_declare_its_own_encoding() -> None:
    with pytest.raises(dx.ValidationError, match="encoding"):
        ActiveCollectorSpec.model_validate({"kind": "bead", "encoding": "e.json"})


def test_a_predictor_truth_collector_resolves_its_column_participants_and_scale() -> (
    None
):
    declared = ActiveCollectorSpec.model_validate(
        {
            "kind": "predictor-truth",
            "column": "p_gram",
            "participants": 4,
            "scale": "standardized",
            "latent_range": {"lower": -3.0, "upper": 3.0},
        }
    )

    assert isinstance(declared, PredictorTruthCollectorSpec)
    assert declared.settings() == PredictorTruthCollectorSettings(
        "predictor-truth",
        "p_gram",
        4,
        "standardized",
        LatentRangeSettings(-3.0, 3.0),
    )


def test_a_registered_design_collector_resolves_its_design_tag() -> None:
    declared = ActiveCollectorSpec.model_validate(
        {
            "kind": "registered-design",
            "design": "known-random-effects",
            "join": "item['design_item']",
        }
    )

    assert isinstance(declared, RegisteredDesignCollectorSpec)
    assert declared.settings() == RegisteredDesignCollectorSettings(
        "registered-design", "known-random-effects", "item['design_item']"
    )


def test_a_drawn_effects_collector_resolves_its_participants_and_range() -> None:
    declared = ActiveCollectorSpec.model_validate(
        {
            "kind": "drawn-effects",
            "participants": 3,
            "latent_range": {"lower": -2.0, "upper": 4.0},
        }
    )

    assert isinstance(declared, DrawnEffectsCollectorSpec)
    assert declared.settings() == DrawnEffectsCollectorSettings(
        "drawn-effects", 3, LatentRangeSettings(-2.0, 4.0)
    )


@pytest.mark.parametrize(
    ("declared", "reason"),
    [
        (
            {
                "kind": "predictor-truth",
                "column": "p_gram",
                "participants": 0,
                "scale": "raw",
                "latent_range": RANGE,
            },
            "participants",
        ),
        (
            {
                "kind": "predictor-truth",
                "column": "",
                "participants": 2,
                "scale": "raw",
                "latent_range": RANGE,
            },
            "predictor column",
        ),
        (
            {
                "kind": "predictor-truth",
                "column": "p_gram",
                "participants": 2,
                "scale": "ranked",
                "latent_range": RANGE,
            },
            "scale",
        ),
        (
            {
                "kind": "predictor-truth",
                "column": "p_gram",
                "participants": 2,
                "latent_range": RANGE,
            },
            "scale",
        ),
        (
            {
                "kind": "predictor-truth",
                "column": "p_gram",
                "participants": 2,
                "scale": "raw",
            },
            "latent_range",
        ),
        ({"kind": "drawn-effects", "participants": 2}, "latent_range"),
        (
            {
                "kind": "drawn-effects",
                "participants": 2,
                "latent_range": {"lower": 1.0, "upper": 1.0},
            },
            "latent range",
        ),
        (
            {
                "kind": "drawn-effects",
                "participants": 2,
                "latent_range": {"lower": float("-inf"), "upper": 1.0},
            },
            "latent range",
        ),
        (
            {"kind": "registered-design", "design": "", "join": None},
            "registered design",
        ),
        (
            {"kind": "drawn-effects", "participants": -1, "latent_range": RANGE},
            "participants",
        ),
        (
            {
                "kind": "drawn-effects",
                "participants": 2,
                "latent_range": RANGE,
                "seed": 4,
            },
            "seed",
        ),
    ],
)
def test_a_malformed_simulated_collector_is_refused(
    declared: dict[str, JsonValue], reason: str
) -> None:
    with pytest.raises(dx.ValidationError, match=reason):
        ActiveCollectorSpec.model_validate(declared)
