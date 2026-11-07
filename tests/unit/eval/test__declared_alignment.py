import pytest

from lofi.eval._alignment import Contingency
from lofi.eval._declared_alignment import (
    fit_declared_alignment,
    predict_relabeling_base,
    score_declared_alignment,
    stable_declared_alignments,
)
from lofi.spec.evaluators.alignment import AlignmentKind, AlignmentSettings


def _settings(kind: AlignmentKind) -> AlignmentSettings:
    return AlignmentSettings(
        kind=kind,
        tie_break=None if kind == "identity" else "least-gold-label",
        fit_on="dev",
        score_on="test",
        floor_replicates=31,
        floor_level=0.9,
        stability_replicates=31,
        stability_level=0.9,
    )


DEV = Contingency(
    {
        ("u0", "g2"): 20,
        ("u1", "g0"): 18,
        ("u2", "g1"): 16,
        ("u0", "g0"): 1,
        ("u1", "g1"): 1,
        ("u2", "g2"): 1,
    },
    split="dev",
)
TEST = Contingency(
    {
        ("u0", "g2"): 9,
        ("u1", "g0"): 8,
        ("u2", "g1"): 7,
        ("u0", "g0"): 1,
    },
    split="test",
)


@pytest.mark.parametrize("kind", ["many-to-one", "one-to-one"])
def test_a_declared_policy_recovers_a_planted_permutation_on_dev_then_freezes_it(
    kind: AlignmentKind,
) -> None:
    fitted = fit_declared_alignment(DEV, _settings(kind))
    scored = score_declared_alignment(fitted, TEST, seed=7)

    assert dict(fitted.alignment.mapping) == {"u0": "g2", "u1": "g0", "u2": "g1"}
    assert fitted.alignment.split == "dev"
    assert scored.observed.value == pytest.approx(24 / 25)
    assert scored.floor.result.n == TEST.total
    assert scored.observed.value is not None
    assert scored.observed.value > scored.floor.value
    assert tuple(scored.rows) == (f"{kind}-accuracy", "random-relabeling-floor")


def test_identity_is_a_declared_current_policy_rather_than_an_implicit_default() -> (
    None
):
    fitted = fit_declared_alignment(DEV, _settings("identity"))
    scored = score_declared_alignment(fitted, TEST, seed=3)

    assert dict(fitted.alignment.mapping) == {}
    assert scored.observed.value == 0.0


def test_scoring_refuses_the_development_split_and_fitting_refuses_test() -> None:
    fitted = fit_declared_alignment(DEV, _settings("many-to-one"))

    with pytest.raises(ValueError, match="score_on"):
        score_declared_alignment(fitted, DEV, seed=0)

    with pytest.raises(ValueError, match="fit_on"):
        fit_declared_alignment(TEST, _settings("many-to-one"))


def test_stability_uses_only_the_frozen_development_alignments() -> None:
    fitted = tuple(
        fit_declared_alignment(DEV, _settings("one-to-one")) for _ in range(4)
    )
    stability = stable_declared_alignments(fitted, seed=11)

    assert stability.seeds == 4
    assert stability.agreement.value == 1.0


def test_the_relabeling_base_prediction_is_derived_from_p_c_and_ambiguity() -> None:
    prediction = predict_relabeling_base(
        primitive_count=7,
        constructor_count=3,
        ambiguity_budget=2,
    )

    assert prediction.group_size == 30_240
    assert prediction.base_size == 3
    assert prediction.to_spec() == {
        "primitive_count": 7,
        "constructor_count": 3,
        "ambiguity_budget": 2,
        "relabeling_group_size": 30_240,
        "predicted_base_size": 3,
    }


def test_a_held_out_split_of_one_token_reports_a_floor_without_an_interval() -> None:
    fitted = fit_declared_alignment(DEV, _settings("many-to-one"))
    single = Contingency({("u0", "g2"): 1}, split="test")
    scored = score_declared_alignment(fitted, single, seed=7)
    floor = scored.rows["random-relabeling-floor"]

    assert scored.observed.value == pytest.approx(1.0)
    assert floor.value == pytest.approx(1.0)
    assert floor.n == 1
    assert floor.ci is None
