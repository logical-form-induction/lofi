from math import log
from typing import Final, cast

import pytest

from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._stability import AlignmentStability, alignment_stability
from lofi.eval._values import MINIMUM_SEEDS, MetricResult, Resampling
from lofi.stats import Bootstrap, Interval

_COUNTS: Final = {
    ("i0", "NP"): 5,
    ("i0", "VP"): 1,
    ("i1", "VP"): 4,
    ("i1", "NP"): 1,
    ("i2", "PP"): 3,
    ("i2", "NP"): 1,
}

_TABLE: Final = Contingency(_COUNTS, split="dev")


def _bootstrap(seed: int = 7) -> Bootstrap:
    return Bootstrap(unit="seed", replicates=199, seed=seed, level=0.95)


def _alignment(mapping: dict[str, str]) -> Alignment:
    return Alignment(mapping, _TABLE)


_SETTLED: Final = tuple(
    _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}) for _ in range(MINIMUM_SEEDS)
)

_WANDERING: Final = (
    _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}),
    _alignment({"i0": "VP", "i1": "NP", "i2": "PP"}),
    _alignment({"i0": "PP", "i1": "NP", "i2": "VP"}),
    _alignment({"i0": "VP", "i1": "PP", "i2": "NP"}),
)


def test_seeds_that_recover_one_map_agree_everywhere() -> None:
    stability = alignment_stability(_SETTLED, bootstrap=_bootstrap())

    assert stability.agreement.value == 1.0
    assert stability.seeds == MINIMUM_SEEDS
    assert stability.labels == ("i0", "i1", "i2")
    assert stability.dropped == ()
    assert all(value == 0.0 for value in stability.entropy.values())


def test_the_agreement_matches_the_count_of_agreeing_pairs_by_hand() -> None:
    # three agreements across all label pairs
    stability = alignment_stability(_WANDERING, bootstrap=_bootstrap())
    pairs = MINIMUM_SEEDS * (MINIMUM_SEEDS - 1) // 2

    assert stability.agreement.value == pytest.approx(3.0 / (3.0 * pairs))
    assert stability.agreement.components["pairs"] == float(pairs)
    assert stability.agreement.components["labels"] == 3.0


def test_each_entropy_matches_the_distribution_of_that_label_by_hand() -> None:
    stability = alignment_stability(_WANDERING, bootstrap=_bootstrap())
    # each label distribution has counts 2, 1, 1
    twice_and_two_singletons = -(0.5 * log(0.5) + 2.0 * 0.25 * log(0.25))

    for label in ("i0", "i1", "i2"):
        assert stability.entropy[label] == pytest.approx(twice_and_two_singletons)


def test_a_label_that_lands_somewhere_new_every_seed_reaches_the_ceiling() -> None:
    table = Contingency(
        {("i0", gold): 1 for gold in ("NP", "VP", "PP", "AP")}, split="dev"
    )
    scattered = tuple(
        Alignment({"i0": gold}, table) for gold in ("NP", "VP", "PP", "AP")
    )
    stability = alignment_stability(scattered, bootstrap=_bootstrap())

    assert stability.entropy["i0"] == pytest.approx(log(MINIMUM_SEEDS))
    assert stability.agreement.value == 0.0


def test_the_entropies_separate_a_settled_label_from_a_wandering_one() -> None:
    mixed = (
        _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}),
        _alignment({"i0": "NP", "i1": "NP", "i2": "PP"}),
        _alignment({"i0": "NP", "i1": "PP", "i2": "PP"}),
        _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}),
    )
    stability = alignment_stability(mixed, bootstrap=_bootstrap())

    assert stability.entropy["i0"] == 0.0
    assert stability.entropy["i2"] == 0.0
    assert stability.entropy["i1"] > 0.0


def test_the_interval_is_drawn_over_the_seeds_and_covers_the_estimate() -> None:
    stability = alignment_stability(_WANDERING, bootstrap=_bootstrap())
    interval = stability.agreement.ci

    assert interval is not None
    assert interval.level == 0.95
    assert interval.covers(stability.agreement.value or 0.0)
    assert stability.agreement.resampling.unit == "seed"
    assert stability.agreement.resampling.seed == 7


def test_two_runs_of_one_seed_draw_the_same_interval() -> None:
    first = alignment_stability(_WANDERING, bootstrap=_bootstrap())
    second = alignment_stability(_WANDERING, bootstrap=_bootstrap())

    assert first == second
    assert hash(first) == hash(second)
    assert first != alignment_stability(_SETTLED, bootstrap=_bootstrap())
    assert first != "a stability"


def test_a_label_some_seed_left_unassigned_is_dropped_rather_than_scored() -> None:
    partial = (
        _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}),
        _alignment({"i0": "NP", "i1": "VP"}),
        _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}),
        _alignment({"i0": "NP", "i1": "VP", "i2": "PP"}),
    )
    stability = alignment_stability(partial, bootstrap=_bootstrap())

    assert stability.labels == ("i0", "i1")
    assert stability.dropped == ("i2",)
    assert stability.agreement.value == 1.0
    assert set(stability.entropy) == {"i0", "i1"}
    assert stability.agreement.components["dropped"] == 1.0


def test_seeds_sharing_no_assigned_label_are_refused() -> None:
    disjoint = (
        _alignment({"i0": "NP"}),
        _alignment({"i1": "VP"}),
        _alignment({"i2": "PP"}),
        _alignment({"i0": "NP"}),
    )

    with pytest.raises(ValueError, match="nothing to compare"):
        alignment_stability(disjoint, bootstrap=_bootstrap())


def test_too_few_seeds_are_refused() -> None:
    with pytest.raises(ValueError, match="at least 4"):
        alignment_stability(_SETTLED[:3], bootstrap=_bootstrap())


def test_an_alignment_or_a_bootstrap_that_is_not_one_is_refused() -> None:
    with pytest.raises(TypeError, match="recovered alignment is not one"):
        alignment_stability(
            [*_SETTLED[:3], cast("Alignment", _TABLE)], bootstrap=_bootstrap()
        )

    with pytest.raises(TypeError, match="bootstrap is not one"):
        alignment_stability(_SETTLED, bootstrap=cast("Bootstrap", 99))


def test_the_stability_writes_out_every_column_a_report_prints() -> None:
    spec = alignment_stability(_WANDERING, bootstrap=_bootstrap()).to_spec()

    assert spec["seeds"] == MINIMUM_SEEDS
    assert spec["agreement_ci_level"] == 0.95
    assert spec["labels"] == ["i0", "i1", "i2"]
    assert spec["dropped"] == []
    assert isinstance(spec["entropy"], dict)
    assert set(spec["entropy"]) == {"i0", "i1", "i2"}


def test_a_stability_refuses_an_agreement_that_is_not_a_result() -> None:
    with pytest.raises(TypeError, match="agreement is not a result"):
        AlignmentStability(
            cast("MetricResult", 0.5), {}, labels=(), dropped=(), seeds=4
        )


def test_a_stability_refuses_entropies_that_are_not_over_the_shared_labels() -> None:
    result = MetricResult(0.5, 4, Resampling("seed", 1), ci=Interval(0.2, 0.8, 0.95))

    with pytest.raises(ValueError, match="reports entropies for"):
        AlignmentStability(result, {"i9": 0.0}, labels=("i0",), dropped=(), seeds=4)


def test_a_stability_refuses_a_label_that_is_both_shared_and_dropped() -> None:
    result = MetricResult(0.5, 4, Resampling("seed", 1), ci=Interval(0.2, 0.8, 0.95))

    with pytest.raises(ValueError, match="both shared and dropped"):
        AlignmentStability(
            result, {"i0": 0.0}, labels=("i0",), dropped=("i0",), seeds=4
        )


def test_a_stability_repr_names_its_five_fields() -> None:
    text = repr(alignment_stability(_SETTLED, bootstrap=_bootstrap()))

    assert text.startswith("AlignmentStability(MetricResult(")
    assert "labels=['i0', 'i1', 'i2']" in text
    assert "seeds=4" in text
