from hashlib import sha256
from math import log
from typing import TYPE_CHECKING, Final, cast

import pytest

from lofi.eval._alignment import Contingency
from lofi.eval._clustering import (
    ClusteringScore,
    adjusted_mutual_information,
    adjusted_rand,
    clustering_scores,
    completeness,
    homogeneity,
    v_measure,
)
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping

# hand-computed two-by-two clustering
_SMALL: Final[Mapping[tuple[str, str], int]] = {
    ("u0", "g0"): 2,
    ("u0", "g1"): 1,
    ("u1", "g1"): 3,
}

_CORRECTED: Final = frozenset({"adjusted-mutual-information", "adjusted-rand"})


def _dev(counts: Mapping[tuple[str, str], int]) -> Contingency:
    return Contingency(counts, split="dev")


def _drawn(token: int, labels: int, salt: bytes) -> int:
    digest = sha256(salt + token.to_bytes(8, "big")).digest()

    return int.from_bytes(digest[:8], "big") % labels


def _plogp(share: float) -> float:
    return 0.0 if share == 0.0 else -share * log(share)


def _value(score: ClusteringScore) -> float:
    number = score.result.value

    if number is None:
        pytest.fail(f"{score.name} reported no value on a table that has one")

    return number


def test_the_five_scores_ship_and_only_the_adjusted_two_are_chance_corrected() -> None:
    scores = clustering_scores(_dev(_SMALL), unit="token")

    assert set(scores) == {
        "homogeneity",
        "completeness",
        "v-measure",
        "adjusted-mutual-information",
        "adjusted-rand",
    }
    assert {name for name, score in scores.items() if score.chance_corrected} == (
        _CORRECTED
    )


def test_homogeneity_and_completeness_match_the_entropies_written_out() -> None:
    table = _dev(_SMALL)
    induced_entropy = _plogp(3 / 6) + _plogp(3 / 6)
    gold_entropy = _plogp(2 / 6) + _plogp(4 / 6)
    joint_entropy = _plogp(2 / 6) + _plogp(1 / 6) + _plogp(3 / 6)

    assert _value(homogeneity(table, unit="token")) == pytest.approx(
        1.0 - (joint_entropy - induced_entropy) / gold_entropy
    )
    assert _value(homogeneity(table, unit="token")) == pytest.approx(0.5)
    assert _value(completeness(table, unit="token")) == pytest.approx(
        1.0 - (joint_entropy - gold_entropy) / induced_entropy
    )


def test_v_measure_at_beta_one_is_the_harmonic_mean_of_the_two() -> None:
    table = _dev(_SMALL)
    first = _value(homogeneity(table, unit="token"))
    second = _value(completeness(table, unit="token"))

    assert _value(v_measure(table, unit="token")) == pytest.approx(
        2.0 * first * second / (first + second)
    )


def test_v_measure_leans_on_homogeneity_below_beta_one_and_away_above_it() -> None:
    table = _dev(_SMALL)
    first = _value(homogeneity(table, unit="token"))
    second = _value(completeness(table, unit="token"))
    scores = [
        _value(v_measure(table, unit="token", beta=beta)) for beta in (0.0, 0.5, 2.0)
    ]

    assert first > second
    assert scores[0] == pytest.approx(first)
    assert scores[0] > scores[1] > scores[2]
    assert v_measure(table, unit="token", beta=0.5).result.components["beta"] == 0.5


def test_adjusted_rand_matches_the_pair_counts_written_out() -> None:
    # cell pairs: 1 + 0 + 3. row pairs: 3 + 3. column pairs: 1 + 6
    joined = 1 + 0 + 3
    expected = (3 + 3) * (1 + 6) / 15
    room = (3 + 3 + 1 + 6) / 2 - expected

    assert _value(adjusted_rand(_dev(_SMALL), unit="token")) == pytest.approx(
        (joined - expected) / room
    )


def test_homogeneity_is_one_for_the_maximally_refined_labeling() -> None:
    refined = _dev({(f"u{index}", f"g{index % 2}"): 1 for index in range(8)})
    scores = clustering_scores(refined, unit="token")

    assert _value(scores["homogeneity"]) == 1.0
    assert _value(scores["completeness"]) < 1.0


def test_a_random_labeling_of_five_thousand_tokens_is_near_zero_when_adjusted() -> None:
    counts: dict[tuple[str, str], int] = {}

    for token in range(5000):
        pair = (
            f"u{_drawn(token, 6, b'induced')}",
            f"g{_drawn(token, 4, b'gold')}",
        )
        counts[pair] = counts.get(pair, 0) + 1

    table = _dev(counts)
    scores = clustering_scores(table, unit="token")

    assert table.total == 5000
    assert len(table.induced_labels) == 6
    assert len(table.gold_labels) == 4
    assert abs(_value(scores["adjusted-mutual-information"])) < 0.02
    assert abs(_value(scores["adjusted-rand"])) < 0.02
    assert _value(scores["homogeneity"]) > 0.0
    assert _value(scores["v-measure"]) > 0.0


def test_the_uncorrected_scores_rise_as_the_induced_inventory_is_cut_finer() -> None:
    coarse = _dev({("u0", "g0"): 8, ("u0", "g1"): 8})
    split = _dev(
        {
            ("u0", "g0"): 6,
            ("u0", "g1"): 2,
            ("u1", "g0"): 2,
            ("u1", "g1"): 6,
        }
    )
    finest = _dev({(f"u{index}", f"g{index % 2}"): 1 for index in range(16)})
    rising = [
        _value(homogeneity(table, unit="token")) for table in (coarse, split, finest)
    ]

    assert rising[0] == 0.0
    assert rising[0] < rising[1] < rising[2] == 1.0
    assert _value(adjusted_mutual_information(finest, unit="token")) < 1.0


def test_every_score_reports_the_inventory_sizes_and_the_token_count() -> None:
    table = _dev(_SMALL)

    for score in clustering_scores(table, unit="span").values():
        assert score.result.components["induced_labels"] == 2.0
        assert score.result.components["gold_labels"] == 2.0
        assert score.result.n == 6
        assert score.result.resampling.unit == "span"


def test_an_empty_table_scores_nothing_rather_than_zero() -> None:
    for score in clustering_scores(Contingency({}, split="dev"), unit="token").values():
        assert score.result.value is None
        assert score.result.n == 0


def test_a_single_label_on_each_side_leaves_the_adjusted_scores_undefined() -> None:
    trivial = _dev({("u0", "g0"): 12})
    scores = clustering_scores(trivial, unit="token")

    assert _value(scores["homogeneity"]) == 1.0
    assert _value(scores["completeness"]) == 1.0
    assert scores["adjusted-mutual-information"].result.value is None
    assert scores["adjusted-rand"].result.value is None


def test_a_score_that_is_not_chance_corrected_states_its_inventory_size() -> None:
    bare = MetricResult(0.5, 10, Resampling("token"))

    with pytest.raises(ValueError, match="induced_labels"):
        ClusteringScore("homogeneity", bare, chance_corrected=False)

    assert ClusteringScore("adjusted-rand", bare, chance_corrected=True).result is bare


def test_a_score_refuses_a_result_that_is_not_one_or_a_flag_that_is_not_a_bool() -> (
    None
):
    result = MetricResult(
        0.5, 10, Resampling("token"), components={"induced_labels": 3}
    )

    with pytest.raises(TypeError, match="result is not one"):
        ClusteringScore(
            "homogeneity", cast("MetricResult", 0.5), chance_corrected=False
        )

    with pytest.raises(TypeError, match="flag is not a bool"):
        ClusteringScore("homogeneity", result, chance_corrected=cast("bool", 1))

    with pytest.raises(ValueError, match="name is empty"):
        ClusteringScore("", result, chance_corrected=True)


def test_a_score_is_equal_hashable_and_rebuilt_by_its_repr() -> None:
    table = _dev(_SMALL)
    one = homogeneity(table, unit="token")
    other = homogeneity(table, unit="token")

    assert one == other
    assert hash(one) == hash(other)
    assert one != adjusted_rand(table, unit="token")
    assert one != "homogeneity"
    assert repr(one).startswith("ClusteringScore('homogeneity'")

    for field in ("name", "result", "chance_corrected"):
        with pytest.raises(AttributeError):
            setattr(one, field, None)


def test_a_negative_or_infinite_beta_is_refused() -> None:
    table = _dev(_SMALL)

    with pytest.raises(ValueError, match="beta is negative"):
        v_measure(table, unit="token", beta=-1.0)

    with pytest.raises(ValueError, match="not finite"):
        v_measure(table, unit="token", beta=float("inf"))


def test_an_empty_unit_is_refused_by_every_score() -> None:
    table = _dev(_SMALL)

    with pytest.raises(ValueError, match="a scored unit is empty"):
        homogeneity(table, unit="")

    with pytest.raises(ValueError, match="a scored unit is empty"):
        clustering_scores(table, unit="")
