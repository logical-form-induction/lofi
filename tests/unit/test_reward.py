from typing import cast

import didactic.api as dx
import pytest

from lofi.reward import ContradictionDefinition, InferenceReward, RewardSpec


def test_a_declared_reward_maps_label_probability_to_expected_payoff() -> None:
    reward = RewardSpec(
        proved=2.0,
        refuted=-0.5,
        contradiction="negated-hypothesis",
        abstention="exclude",
    ).build()

    assert reward.pay(0.0) == pytest.approx(-0.5)
    assert reward.pay(0.4) == pytest.approx(0.5)
    assert reward.pay(1.0) == pytest.approx(2.0)
    assert reward.probability_baseline(0.5) == pytest.approx(0.4)
    assert reward.contradiction == "negated-hypothesis"


def test_a_reward_refuses_unordered_or_nonfinite_payoffs() -> None:
    with pytest.raises(dx.ValidationError, match="proved payoff"):
        RewardSpec(
            proved=0.0,
            refuted=0.0,
            contradiction="negated-hypothesis",
            abstention="exclude",
        )

    with pytest.raises(ValueError, match="probability"):
        InferenceReward(
            proved=1.0,
            refuted=0.0,
            contradiction="negated-hypothesis",
        ).pay(1.1)

    with pytest.raises(ValueError, match="unknown contradiction definition"):
        InferenceReward(
            proved=1.0,
            refuted=0.0,
            contradiction=cast("ContradictionDefinition", "other"),
        )
