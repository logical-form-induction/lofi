"""Derivation-entropy declaration tests."""

from lofi.spec import LossSpec
from lofi.spec.losses.derivation_entropy import DerivationEntropyLossSpec


def test_the_entropy_objective_is_a_closed_loss_variant() -> None:
    declared = LossSpec.model_validate(
        {"kind": "derivation-entropy", "direction": "maximize"}
    )

    assert type(declared) is DerivationEntropyLossSpec
    assert declared == DerivationEntropyLossSpec(direction="maximize")
    assert declared.required_chart_outputs() == {"entropy"}


def test_the_entropy_direction_round_trips_through_settings() -> None:
    declared = DerivationEntropyLossSpec(direction="minimize")

    assert declared.settings().direction == "minimize"
