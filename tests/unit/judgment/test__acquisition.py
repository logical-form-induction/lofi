"""Tests for constrained RC2.1 judgment acquisition."""

from types import MappingProxyType
from typing import cast
from uuid import UUID

import pytest
from bead.items.item import Item

from lofi.judgment._acquisition import (
    AcquisitionError,
    AcquisitionScore,
    PoolExhaustedError,
    RC21Acquisition,
)
from lofi.spec.acquisition import (
    BalanceSettings,
    BatchBalanceSettings,
    BatchConstraintSettings,
    BatchCoverageSettings,
    BatchDiversitySettings,
    BatchMinOccurrenceSettings,
    CategoricalBinningSettings,
    ConditionalUniquenessSettings,
    DiversitySettings,
    GridDimensionSettings,
    GridStratificationSettings,
    GroupedQuantileSettings,
    ListConstraintSettings,
    OrderingSettings,
    QuantileSettings,
    RC21AcquisitionSettings,
    UniquenessSettings,
)

TEMPLATE = UUID(int=0)


def _item(index: int, lemma: str, group: str) -> Item:
    return Item(
        id=UUID(int=index + 1),
        item_template_id=TEMPLATE,
        item_metadata={"lemma": lemma, "group": group},
    )


def _unique(expression: str) -> UniquenessSettings:
    return UniquenessSettings("uniqueness", expression)


def _rule(
    batch: int,
    *,
    constraints: tuple[ListConstraintSettings, ...] = (_unique("item['lemma']"),),
    batch_constraints: tuple[BatchConstraintSettings, ...] = (),
) -> RC21Acquisition:
    return RC21Acquisition(
        RC21AcquisitionSettings(
            "rc2.1", batch, "item['group']", constraints, batch_constraints, 0.0
        )
    )


def _attributed(index: int, **attributes: str | float | bool) -> Item:
    return Item(
        id=UUID(int=index + 1),
        item_template_id=TEMPLATE,
        item_metadata={"group": "g", **attributes},
    )


def _scores(
    items: tuple[Item, ...], values: tuple[float, ...]
) -> tuple[AcquisitionScore, ...]:
    return tuple(
        AcquisitionScore(str(item.id), value)
        for item, value in zip(items, values, strict=True)
    )


def test_rc21_selects_high_variance_without_a_repeated_level() -> None:
    candidates = tuple(
        _item(index, lemma, "g")
        for index, lemma in enumerate(("admire", "call", "help", "visit"))
    )
    rule = _rule(2)

    batch = rule.select(
        candidates, _scores(candidates, (0.0, 0.4, 0.6, 1.0)), {"g": 1.0}, previous=()
    )

    random_variance = ((0.4 - 0.5) ** 2 + (0.6 - 0.5) ** 2) / 2.0
    assert batch.items == (candidates[0], candidates[3])
    assert batch.variance is not None
    assert batch.variance > random_variance
    assert batch.shares == {"g": 1.0}
    assert batch.performance == {"g": 1.0}


def test_each_bead_uniqueness_constraint_is_its_own_constraint() -> None:
    shared = Item(
        id=UUID(int=9),
        item_template_id=TEMPLATE,
        item_metadata={"left": "a", "right": "b", "group": "g"},
    )
    crossed = Item(
        id=UUID(int=10),
        item_template_id=TEMPLATE,
        item_metadata={"left": "b", "right": "a", "group": "g"},
    )
    candidates = (shared, crossed)
    scores = _scores(candidates, (0.0, 1.0))

    separate = _rule(2, constraints=(_unique("item['left']"), _unique("item['right']")))
    pooled = _rule(2, constraints=(_unique("[item['left'], item['right']]"),))
    left = _rule(2, constraints=(_unique("item['group']"),))

    assert (
        separate.select(candidates, scores, {"g": 1.0}, previous=()).items == candidates
    )
    assert (
        pooled.select(candidates, scores, {"g": 1.0}, previous=()).items == candidates
    )
    with pytest.raises(PoolExhaustedError, match="only 1 of 2"):
        left.select(candidates, scores, {"g": 1.0}, previous=())


def test_a_conditional_uniqueness_constraint_applies_where_it_holds() -> None:
    candidates = tuple(
        _item(index, lemma, "g") for index, lemma in enumerate(("a", "a", "b"))
    )
    scores = _scores(candidates, (0.0, 1.0, 0.5))
    conditional = ConditionalUniquenessSettings(
        "conditional_uniqueness", "item['lemma']", "item['lemma'] == 'b'"
    )

    batch = _rule(2, constraints=(conditional,)).select(
        candidates, scores, {"g": 1.0}, previous=()
    )

    assert batch.items == (candidates[0], candidates[1])


def test_a_diversity_constraint_is_checked_on_the_selected_batch() -> None:
    candidates = tuple(
        _item(index, lemma, "g") for index, lemma in enumerate(("a", "a", "b"))
    )
    scores = _scores(candidates, (0.0, 1.0, 0.5))
    diverse = DiversitySettings("diversity", "item['lemma']", 2)

    with pytest.raises(AcquisitionError, match="fewer than the declared 2"):
        _rule(2, constraints=(diverse,)).select(
            candidates, scores, {"g": 1.0}, previous=()
        )


def test_rc21_enforces_proportional_quotas_within_tolerance() -> None:
    candidates = (
        _item(0, "p-a", "high"),
        _item(1, "p-b", "high"),
        _item(2, "p-c", "high"),
        _item(3, "p-d", "low"),
    )

    batch = _rule(4).select(
        candidates,
        _scores(candidates, (0.0, 1.0, 0.2, 0.8)),
        {"high": 3.0, "low": 1.0},
        previous=(),
    )

    assert batch.shares == pytest.approx({"high": 0.75, "low": 0.25})


def test_a_candidate_whose_stratum_has_no_performance_is_refused() -> None:
    candidates = (_item(0, "p-a", "high"), _item(1, "p-b", "other"))

    with pytest.raises(AcquisitionError, match="'other'"):
        _rule(1).select(
            candidates, _scores(candidates, (0.0, 1.0)), {"high": 1.0}, previous=()
        )


def test_rc21_reports_an_empty_pool_as_exhausted() -> None:
    with pytest.raises(PoolExhaustedError, match="the acquisition pool is empty"):
        _rule(2).select((), (), {"g": 1.0}, previous=())


def test_rc21_reports_a_pool_that_cannot_fill_a_batch_as_exhausted() -> None:
    candidates = (_item(0, "admire", "g"), _item(1, "call", "g"))

    with pytest.raises(PoolExhaustedError, match="only 2 of 3") as raised:
        _rule(3).select(
            candidates, _scores(candidates, (0.0, 1.0)), {"g": 1.0}, previous=()
        )

    assert isinstance(raised.value, AcquisitionError)


def test_rc21_floors_a_stratum_whose_performance_is_negative() -> None:
    candidates = (
        _item(0, "p-a", "high"),
        _item(1, "p-b", "high"),
        _item(2, "p-c", "low"),
    )

    batch = _rule(2).select(
        candidates,
        _scores(candidates, (0.0, 1.0, 0.5)),
        {"high": 0.5, "low": -0.25},
        previous=(),
    )

    assert batch.items == candidates[:2]
    assert batch.shares == {"high": 1.0, "low": 0.0}
    assert batch.floored == ("low",)
    assert batch.performance == {"high": 0.5, "low": -0.25}


def test_rc21_splits_the_batch_evenly_when_no_stratum_is_positive() -> None:
    candidates = (
        _item(0, "p-a", "high"),
        _item(1, "p-b", "high"),
        _item(2, "p-c", "low"),
        _item(3, "p-d", "low"),
    )

    batch = _rule(2).select(
        candidates,
        _scores(candidates, (0.0, 1.0, 0.2, 0.8)),
        {"high": 0.0, "low": -0.5},
        previous=(),
    )

    assert batch.shares == {"high": 0.5, "low": 0.5}
    assert batch.floored == ("low",)


def test_rc21_records_no_floored_stratum_when_every_performance_is_non_negative() -> (
    None
):
    candidates = (_item(0, "p-a", "high"), _item(1, "p-b", "low"))

    batch = _rule(2).select(
        candidates,
        _scores(candidates, (0.0, 1.0)),
        {"high": 0.5, "low": 0.5},
        previous=(),
    )

    assert batch.floored == ()


def _ranked(count: int) -> tuple[float, ...]:
    """Return scores ranking candidate ``i`` before candidate ``i + 1``."""
    return tuple(
        (count - index) * (1.0 if index % 2 == 0 else -1.0) for index in range(count)
    )


def test_a_balance_constraint_holds_each_level_to_its_count() -> None:
    candidates = tuple(
        _attributed(index, kind=kind) for index, kind in enumerate("aaaabbbb")
    )
    balanced = BalanceSettings("balance", "item['kind']", None, 0.0)
    targeted = BalanceSettings(
        "balance", "item['kind']", MappingProxyType({"a": 1, "b": 3}), 0.0
    )

    equal = _rule(4, constraints=(balanced,)).select(
        candidates, _scores(candidates, _ranked(8)), {"g": 1.0}, previous=()
    )
    weighted = _rule(4, constraints=(targeted,)).select(
        candidates, _scores(candidates, _ranked(8)), {"g": 1.0}, previous=()
    )

    assert sorted(str(item.item_metadata["kind"]) for item in equal.items) == list(
        "aabb"
    )
    assert sorted(str(item.item_metadata["kind"]) for item in weighted.items) == list(
        "abbb"
    )


def test_a_quantile_constraint_takes_at_most_its_count_from_each_bin() -> None:
    candidates = tuple(_attributed(index, x=float(index)) for index in range(8))
    quantile = QuantileSettings("quantile", "item['x']", 4, 1)

    batch = _rule(4, constraints=(quantile,)).select(
        candidates, _scores(candidates, _ranked(8)), {"g": 1.0}, previous=()
    )

    assert sorted(
        int(cast("float", item.item_metadata["x"])) // 2 for item in batch.items
    ) == [
        0,
        1,
        2,
        3,
    ]


def test_a_grouped_quantile_constraint_bins_within_each_group() -> None:
    candidates = tuple(
        _attributed(index, x=float(index % 4), side="lr"[index // 4])
        for index in range(8)
    )
    grouped = GroupedQuantileSettings(
        "grouped_quantile", "item['x']", "item['side']", 2, 1
    )

    batch = _rule(4, constraints=(grouped,)).select(
        candidates, _scores(candidates, _ranked(8)), {"g": 1.0}, previous=()
    )

    assert sorted(
        (
            str(item.item_metadata["side"]),
            int(cast("float", item.item_metadata["x"])) // 2,
        )
        for item in batch.items
    ) == [("l", 0), ("l", 1), ("r", 0), ("r", 1)]


def test_a_grid_constraint_takes_at_most_its_count_from_each_cell() -> None:
    candidates = tuple(_attributed(index, kind="abc"[index % 3]) for index in range(6))
    grid = GridStratificationSettings(
        "grid_stratification",
        (
            GridDimensionSettings(
                "item['kind']",
                CategoricalBinningSettings("categorical", None, include_other=False),
            ),
        ),
        None,
        1,
    )

    batch = _rule(3, constraints=(grid,)).select(
        candidates, _scores(candidates, _ranked(6)), {"g": 1.0}, previous=()
    )

    assert sorted(str(item.item_metadata["kind"]) for item in batch.items) == list(
        "abc"
    )


def test_an_ordering_constraint_fixes_the_batch_order() -> None:
    candidates = (
        _attributed(0, block="x", verb="v", practice=False),
        _attributed(1, block="x", verb="v", practice=False),
        _attributed(2, block="x", verb="w", practice=False),
        _attributed(3, block="y", verb="w", practice=True),
    )
    ordering = OrderingSettings(
        "ordering",
        ((str(candidates[2].id), str(candidates[0].id)),),
        "item['verb']",
        "item['block']",
        None,
        3,
        "item['practice']",
    )

    batch = _rule(4, constraints=(ordering,)).select(
        candidates, _scores(candidates, _ranked(4)), {"g": 1.0}, previous=()
    )

    assert batch.items == (candidates[3], candidates[1], candidates[2], candidates[0])


def test_an_order_no_arrangement_satisfies_is_refused() -> None:
    candidates = tuple(_attributed(index, verb="v") for index in range(2))
    ordering = OrderingSettings("ordering", (), "item['verb']", None, None, None, None)

    with pytest.raises(AcquisitionError, match="ordering"):
        _rule(2, constraints=(ordering,)).select(
            candidates, _scores(candidates, _ranked(2)), {"g": 1.0}, previous=()
        )


def test_a_batch_diversity_constraint_excludes_a_level_already_in_enough_batches() -> (
    None
):
    candidates = tuple(
        _attributed(index, kind=kind) for index, kind in enumerate("aabb")
    )
    diversity = BatchDiversitySettings("diversity", "item['kind']", 1)
    rule = _rule(1, constraints=(), batch_constraints=(diversity,))
    scores = _scores(candidates, _ranked(4))

    first = rule.select(candidates, scores, {"g": 1.0}, previous=())
    second = rule.select(candidates[1:], scores[1:], {"g": 1.0}, previous=(first,))

    assert first.items == (candidates[0],)
    assert str(second.items[0].item_metadata["kind"]) == "b"


def test_coverage_and_occurrence_constraints_take_needed_levels_first() -> None:
    candidates = tuple(
        _attributed(index, kind=kind) for index, kind in enumerate("aaab")
    )
    scores = _scores(candidates, _ranked(4))
    coverage = BatchCoverageSettings("coverage", "item['kind']", ("b",), 1.0)
    occurrence = BatchMinOccurrenceSettings("min_occurrence", "item['kind']", 1)

    covered = _rule(2, constraints=(), batch_constraints=(coverage,)).select(
        candidates, scores, {"g": 1.0}, previous=()
    )
    occurring = _rule(2, constraints=(), batch_constraints=(occurrence,)).select(
        candidates, scores, {"g": 1.0}, previous=()
    )

    assert candidates[3] in covered.items
    assert candidates[3] in occurring.items


def test_a_batch_balance_constraint_holds_shares_across_batches() -> None:
    candidates = tuple(
        _attributed(index, kind=kind) for index, kind in enumerate("aaaabb")
    )
    scores = _scores(candidates, _ranked(6))
    balance = BatchBalanceSettings(
        "balance", "item['kind']", MappingProxyType({"a": 0.5, "b": 0.5}), 0.0
    )
    rule = _rule(2, constraints=(), batch_constraints=(balance,))

    first = rule.select(candidates, scores, {"g": 1.0}, previous=())
    second = rule.select(candidates[2:], scores[2:], {"g": 1.0}, previous=(first,))

    kinds = [str(item.item_metadata["kind"]) for item in (*first.items, *second.items)]
    assert sorted(kinds) == list("aabb")
