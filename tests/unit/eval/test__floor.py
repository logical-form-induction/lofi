from collections import Counter
from contextlib import suppress
from copy import copy, deepcopy
from itertools import product
from pickle import dumps, loads
from typing import Final, cast

import pytest

from lofi.eval._alignment import Contingency
from lofi.eval._floor import (
    Floor,
    FlooredScore,
    _expected_accuracy,
    random_relabeling_floor,
)
from lofi.eval._values import MetricResult, Resampling
from lofi.stats import Bootstrap, Interval

GOLD: Final = ("NP", "VP", "PP", "AP")
MARGINALS: Final = (40, 30, 20, 10)


def _table(marginals: tuple[int, ...]) -> Contingency:
    return Contingency(
        {
            (f"u{index}", label): count
            for index, (label, count) in enumerate(
                zip(GOLD[: len(marginals)], marginals, strict=True)
            )
        },
        split="test",
    )


def _enumerated(marginals: tuple[int, ...], inventory: int) -> float:
    gold = [
        label
        for label, count in zip(GOLD[: len(marginals)], marginals, strict=True)
        for _ in range(count)
    ]
    total_score = 0.0
    labelings = 0

    for labeling in product(range(inventory), repeat=len(gold)):
        clusters = [Counter[str]() for _ in range(inventory)]

        for label, induced in zip(gold, labeling, strict=True):
            clusters[induced][label] += 1

        correct = sum(max(cluster.values(), default=0) for cluster in clusters)
        total_score += correct / len(gold)
        labelings += 1

    return total_score / labelings


@pytest.mark.parametrize(
    ("marginals", "inventory"),
    [
        ((2, 2), 2),
        ((3, 1), 2),
        ((2, 1, 1), 3),
        ((2, 2), 3),
        ((1, 1, 1, 1), 2),
        ((4, 2), 1),
        ((4,), 3),
    ],
)
def test_the_closed_form_is_the_mean_over_every_labeling_scored_for_real(
    marginals: tuple[int, ...], inventory: int
) -> None:
    expected = _enumerated(marginals, inventory)
    floor = random_relabeling_floor(
        _table(marginals),
        inventory=inventory,
        bootstrap=Bootstrap(unit="token", replicates=19, seed=7, level=0.95),
    )

    assert _expected_accuracy(marginals, inventory) == pytest.approx(expected)
    assert floor.value == pytest.approx(expected)


def test_the_floor_rises_with_the_inventory_and_climbs_toward_one() -> None:
    sizes = (1, 2, 4, 8, 16, 32, 64, 128, 1000, 20_000, 10**12, 10**15)
    floors = [_expected_accuracy(MARGINALS, size) for size in sizes]
    tokens = _table(MARGINALS).total
    cross_label_pairs = (
        tokens * tokens - sum(count * count for count in MARGINALS)
    ) / 2

    assert floors == sorted(floors)
    assert len(set(floors)) == len(floors)
    assert floors[0] == pytest.approx(max(MARGINALS) / tokens)
    assert floors[-1] > 0.99
    assert floors[-1] < 1.0
    assert floors[-2] == pytest.approx(
        1.0 - cross_label_pairs / (tokens * 10**12), abs=1e-14
    )
    assert 0.8 < _expected_accuracy((50, 50), 100) < 0.9


def test_the_floor_remains_stable_for_large_samples_and_inventories() -> None:
    assert 0.5 < _expected_accuracy((2500, 2500), 2) < 0.52

    sizes = (10**6, 10**9, 10**12, 10**15, 10**17)
    floors = [_expected_accuracy((1, 1), size) for size in sizes]

    assert floors == sorted(floors)

    for size, value in zip(sizes, floors, strict=True):
        assert value == pytest.approx(1.0 - 1.0 / (2 * size), abs=1e-15)


def test_an_empty_table_has_no_floor_to_climb() -> None:
    assert _expected_accuracy((), 3) == 0.0
    assert _expected_accuracy((0, 0), 3) == 0.0


def test_a_floor_is_computed_over_the_counts_with_a_resampled_interval() -> None:
    table = _table(MARGINALS)
    drawn = random_relabeling_floor(
        table,
        inventory=4,
        bootstrap=Bootstrap(unit="token", replicates=99, seed=3, level=0.95),
    )

    assert drawn.metric == "many-to-one"
    assert drawn.inventory == 4
    assert drawn.value == pytest.approx(_expected_accuracy(MARGINALS, 4))
    assert drawn.ci is not None
    assert drawn.ci.covers(drawn.value)
    assert drawn.ci.level == pytest.approx(0.95)
    assert drawn.result.n == sum(MARGINALS)
    assert drawn.result.resampling.unit == "token"
    assert drawn.result.components["inventory"] == 4.0


def test_a_floor_is_the_same_number_every_time_it_is_drawn() -> None:
    table = _table(MARGINALS)

    def _drawn() -> Floor:
        return random_relabeling_floor(
            table,
            inventory=7,
            bootstrap=Bootstrap(unit="token", replicates=99, seed=11, level=0.95),
        )

    assert _drawn() == _drawn()
    assert _drawn().to_spec() == _drawn().to_spec()


def test_a_floor_reads_only_the_gold_totals_of_the_table() -> None:
    # preserve gold marginals under a different induced partition
    spread = Contingency(
        {
            ("u0", "VP"): 30,
            ("u1", "NP"): 20,
            ("u2", "NP"): 20,
            ("u2", "PP"): 20,
            ("u3", "AP"): 10,
        },
        split="test",
    )
    bootstrap = Bootstrap(unit="token", replicates=99, seed=3, level=0.95)

    assert random_relabeling_floor(
        spread, inventory=4, bootstrap=bootstrap
    ) == random_relabeling_floor(_table(MARGINALS), inventory=4, bootstrap=bootstrap)


def test_a_floor_refuses_invalid_boundaries_directly() -> None:
    table = _table(MARGINALS)
    bootstrap = Bootstrap(unit="token", replicates=19, seed=3, level=0.95)

    with pytest.raises(ValueError, match="below one"):
        random_relabeling_floor(table, inventory=0, bootstrap=bootstrap)

    with pytest.raises(ValueError, match="at least one token, not 0"):
        random_relabeling_floor(
            Contingency({}, split="test"), inventory=4, bootstrap=bootstrap
        )

    with pytest.raises(ValueError, match="must resample 'token'"):
        random_relabeling_floor(
            table,
            inventory=4,
            bootstrap=Bootstrap(unit="sentence", replicates=19, seed=3, level=0.95),
        )

    with pytest.raises(TypeError, match="contingency is not one"):
        random_relabeling_floor(
            cast("Contingency", MARGINALS), inventory=4, bootstrap=bootstrap
        )

    with pytest.raises(TypeError, match="bootstrap is not one"):
        random_relabeling_floor(table, inventory=4, bootstrap=cast("Bootstrap", 99))


def test_a_floor_over_one_token_reports_its_value_without_an_interval() -> None:
    drawn = random_relabeling_floor(
        Contingency({("u0", "NP"): 1}, split="test"),
        inventory=4,
        bootstrap=Bootstrap(unit="token", replicates=19, seed=3, level=0.95),
    )

    assert drawn.value == pytest.approx(_enumerated((1,), 4))
    assert drawn.ci is None
    assert drawn.result.n == 1
    assert drawn.result.ci is None
    assert drawn.to_spec()["floor"] == pytest.approx(1.0)
    assert drawn.to_spec()["floor_ci_lo"] is None
    assert drawn.to_spec()["floor_ci_hi"] is None
    assert drawn.to_spec()["floor_ci_level"] is None
    assert loads(dumps(drawn)) == drawn


def test_zero_cells_and_one_gold_label_have_their_exact_floors() -> None:
    bootstrap = Bootstrap(unit="token", replicates=39, seed=7, level=0.95)
    zero_cell = _table((2, 0, 1))
    single_gold = _table((4,))

    zero_floor = random_relabeling_floor(zero_cell, inventory=3, bootstrap=bootstrap)
    single_floor = random_relabeling_floor(
        single_gold, inventory=3, bootstrap=bootstrap
    )

    assert zero_floor.value == pytest.approx(_enumerated((2, 0, 1), 3))
    assert single_floor.value == pytest.approx(1.0)
    assert _expected_accuracy((1000,), 1_000_000_000) == 1.0


def test_floor_computation_does_not_mutate_its_inputs() -> None:
    counts = {
        ("u0", "NP"): 3,
        ("u1", "VP"): 2,
        ("u1", "NP"): 1,
    }
    original = dict(counts)
    table = Contingency(counts, split="test")
    bootstrap = Bootstrap(unit="token", replicates=39, seed=7, level=0.95)
    bootstrap_state = (
        bootstrap.unit,
        bootstrap.replicates,
        bootstrap.seed,
        bootstrap.level,
        bootstrap.groups,
    )

    random_relabeling_floor(table, inventory=3, bootstrap=bootstrap)

    assert counts == original
    assert dict(table.counts) == original
    assert (
        bootstrap.unit,
        bootstrap.replicates,
        bootstrap.seed,
        bootstrap.level,
        bootstrap.groups,
    ) == bootstrap_state


def test_a_floor_refuses_grouped_token_bootstrapping() -> None:
    table = _table(MARGINALS)

    with pytest.raises(ValueError, match="individual tokens without groups"):
        random_relabeling_floor(
            table,
            inventory=3,
            bootstrap=Bootstrap(
                unit="token",
                replicates=39,
                seed=7,
                groups=("sentence",) * table.total,
                level=0.95,
            ),
        )


def test_a_floor_without_a_value_or_an_interval_is_not_constructible() -> None:
    scheme = Resampling("token", 3)

    with pytest.raises(ValueError, match="reports no value"):
        Floor("many-to-one", 4, MetricResult(None, 0, scheme))

    with pytest.raises(ValueError, match="reports no interval over 100 tokens"):
        Floor("many-to-one", 4, MetricResult(0.4, 100, scheme))

    with pytest.raises(ValueError, match="interval over 1 token"):
        Floor(
            "many-to-one",
            4,
            MetricResult(0.4, 1, scheme, ci=Interval(0.3, 0.5, 0.95)),
        )

    with pytest.raises(TypeError, match="result is not one"):
        Floor("many-to-one", 4, cast("MetricResult", 0.4))

    with pytest.raises(ValueError, match="empty"):
        Floor("", 4, MetricResult(0.4, 100, scheme, ci=Interval(0.3, 0.5, 0.95)))

    with pytest.raises(ValueError, match="below one"):
        Floor(
            "many-to-one",
            0,
            MetricResult(0.4, 100, scheme, ci=Interval(0.3, 0.5, 0.95)),
        )

    with pytest.raises(ValueError, match="must resample 'token'"):
        Floor(
            "many-to-one",
            4,
            MetricResult(
                0.4,
                100,
                Resampling("sentence", 3),
                ci=Interval(0.3, 0.5, 0.95),
            ),
        )


def _floor(value: float = 0.42) -> Floor:
    return Floor(
        "many-to-one",
        4,
        MetricResult(
            value,
            100,
            Resampling("token", 3),
            components={"induced_labels": 4.0},
            ci=Interval(value - 0.02, value + 0.02, 0.95),
        ),
    )


def _observed(
    value: float | None = 0.71,
    *,
    count: int = 100,
    unit: str = "token",
    inventory: float | None = 4.0,
) -> MetricResult:
    components = {} if inventory is None else {"induced_labels": inventory}

    return MetricResult(
        value,
        count,
        Resampling(unit, 3),
        components=components,
    )


def test_an_assignment_row_holds_its_observation_and_its_floor_together() -> None:
    observed = _observed()
    row = FlooredScore("many-to-one", observed, _floor())

    assert row.observed == observed
    assert row.observed is not observed
    assert row.floor.value == pytest.approx(0.42)
    assert row.excess == pytest.approx(0.29)
    assert row.beats_floor
    assert row.to_spec()["observed"] == pytest.approx(0.71)
    assert row.to_spec()["floor"] == pytest.approx(0.42)
    assert row.to_spec()["inventory"] == 4


def test_an_observation_inside_the_floors_interval_does_not_beat_it() -> None:
    inside = FlooredScore("many-to-one", _observed(0.43), _floor())
    absent = FlooredScore("many-to-one", _observed(None), _floor())

    assert not inside.beats_floor
    assert absent.excess is None
    assert absent.beats_floor is None
    assert absent.to_spec()["observed"] is None


def test_a_floor_without_an_interval_is_beaten_by_an_observation_above_it() -> None:
    floor = Floor(
        "many-to-one",
        4,
        MetricResult(0.25, 1, Resampling("token", 3)),
    )
    above = FlooredScore("many-to-one", _observed(1.0, count=1), floor)
    level = FlooredScore("many-to-one", _observed(0.25, count=1), floor)

    assert floor.ci is None
    assert above.beats_floor is True
    assert above.excess == pytest.approx(0.75)
    assert level.beats_floor is False
    assert above.to_spec()["floor_ci_hi"] is None


def test_an_assignment_row_refuses_a_part_that_is_not_what_it_should_be() -> None:
    observed = _observed()

    with pytest.raises(TypeError, match="not a result"):
        FlooredScore("many-to-one", cast("MetricResult", 0.71), _floor())

    with pytest.raises(TypeError, match="floor is not one"):
        FlooredScore("many-to-one", observed, cast("Floor", observed))

    with pytest.raises(ValueError, match="metric 'one-to-one'"):
        FlooredScore("one-to-one", observed, _floor())

    with pytest.raises(ValueError, match="counts 99 tokens"):
        FlooredScore("many-to-one", _observed(count=99), _floor())

    with pytest.raises(ValueError, match="counts 'sentence'"):
        FlooredScore("many-to-one", _observed(unit="sentence"), _floor())

    with pytest.raises(ValueError, match="does not report"):
        FlooredScore("many-to-one", _observed(inventory=None), _floor())

    with pytest.raises(ValueError, match=r"uses 3\.0 induced labels"):
        FlooredScore("many-to-one", _observed(inventory=3.0), _floor())


def test_two_rows_are_equal_on_the_observation_and_the_floor() -> None:
    observed = _observed()
    one = FlooredScore("many-to-one", observed, _floor())
    same = FlooredScore("many-to-one", observed, _floor())
    other = FlooredScore("many-to-one", observed, _floor(0.5))

    assert one == same
    assert hash(one) == hash(same)
    assert one != other
    assert one != "a row"
    assert repr(one).startswith("FlooredScore(")
    assert one.floor != "a floor"
    assert hash(one.floor) == hash(same.floor)
    assert repr(one.floor).startswith("Floor(")


def test_floor_values_are_frozen_copyable_and_pickleable() -> None:
    floor = _floor()
    row = FlooredScore("many-to-one", _observed(), floor)

    for value, field in ((floor, "_inventory"), (row, "_floor")):
        assert copy(value) is value
        assert deepcopy(value) is value
        assert loads(dumps(value)) == value

        with pytest.raises(AttributeError, match="immutable"):
            setattr(value, field, 0)

        with pytest.raises(AttributeError, match="immutable"):
            delattr(value, field)


def test_floor_values_preserve_an_interval_without_a_coverage_level() -> None:
    result = MetricResult(
        0.42,
        100,
        Resampling("token", 3),
        components={"induced_labels": 4.0},
        ci=Interval(0.40, 0.44),
    )
    floor = Floor("many-to-one", 4, result)
    row = FlooredScore("many-to-one", result, floor)

    restored_floor = loads(dumps(floor))
    restored_row = loads(dumps(row))

    assert restored_floor.ci == Interval(0.40, 0.44)
    assert restored_floor.ci.level is None
    assert restored_row.floor.ci.level is None
    assert restored_row == row


def test_floor_values_snapshot_nested_metric_values() -> None:
    result = MetricResult(
        0.42,
        100,
        Resampling("token", 3),
        components={"induced_labels": 4.0, "matched": 42.0},
        ci=Interval(0.40, 0.44, 0.95),
    )
    floor = Floor("many-to-one", 4, result)
    row = FlooredScore("many-to-one", result, floor)
    floor_hash = hash(floor)
    row_hash = hash(row)
    exposed_result = floor.result
    exposed_resampling = exposed_result.resampling
    exposed_interval = floor.ci
    exposed_observed = row.observed

    for value, field, replacement in (
        (result, "_value", 0.99),
        (result, "_components", {"matched": 99.0}),
        (exposed_result, "_value", 0.01),
        (exposed_resampling, "_unit", "sentence"),
        (exposed_interval, "_low", 0.0),
        (exposed_observed, "_value", 0.0),
    ):
        with suppress(AttributeError):
            setattr(value, field, replacement)

    assert floor.value == pytest.approx(0.42)
    assert dict(floor.result.components) == {
        "induced_labels": 4.0,
        "matched": 42.0,
    }
    assert floor.result.resampling.unit == "token"
    assert floor.ci is not None
    assert floor.ci.low == pytest.approx(0.40)
    assert row.observed.value == pytest.approx(0.42)
    assert hash(floor) == floor_hash
    assert hash(row) == row_hash
