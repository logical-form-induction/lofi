"""Tests for the bead list and batch constraints an acquisition declares."""

import didactic.api as dx
import pytest

from lofi.spec.acquisition import (
    BalanceSettings,
    BatchBalanceSettings,
    BatchConstraintSettings,
    BatchConstraintSpec,
    BatchCoverageSettings,
    BatchDiversitySettings,
    BatchMinOccurrenceSettings,
    CategoricalBinningSettings,
    EqualWidthBinningSettings,
    GridDimensionSettings,
    GridStratificationSettings,
    GroupedQuantileSettings,
    ListConstraintSettings,
    ListConstraintSpec,
    OrderingSettings,
    QuantileBinningSettings,
    QuantileSettings,
    StdDevBinningSettings,
    ThresholdBinningSettings,
    batch_settings,
    list_settings,
)


def _list(document: str) -> ListConstraintSettings:
    return list_settings(ListConstraintSpec.model_validate_json(document))


def _batch(document: str) -> BatchConstraintSettings:
    return batch_settings(BatchConstraintSpec.model_validate_json(document))


def test_a_balance_constraint_resolves_its_targets_and_tolerance() -> None:
    settings = _list(
        '{"constraint_type": "balance", "property_expression": "item[\'g\']",'
        ' "target_counts": {"a": 2, "b": 1}, "tolerance": 0.25}'
    )

    assert isinstance(settings, BalanceSettings)
    assert dict(settings.target_counts or {}) == {"a": 2, "b": 1}
    assert settings.tolerance == pytest.approx(0.25)


def test_quantile_constraints_resolve_their_bins_and_groups() -> None:
    plain = _list(
        '{"constraint_type": "quantile", "property_expression": "item[\'x\']",'
        ' "n_quantiles": 4, "items_per_quantile": 1}'
    )
    grouped = _list(
        '{"constraint_type": "grouped_quantile", "property_expression":'
        ' "item[\'x\']", "group_by_expression": "item[\'g\']", "n_quantiles": 2,'
        ' "items_per_quantile": 3}'
    )

    assert plain == QuantileSettings("quantile", "item['x']", 4, 1)
    assert grouped == GroupedQuantileSettings(
        "grouped_quantile", "item['x']", "item['g']", 2, 3
    )


def test_a_grid_constraint_resolves_every_binning_strategy() -> None:
    settings = _list(
        '{"constraint_type": "grid_stratification", "items_per_cell": 2,'
        ' "group_by_expression": null, "dimensions": ['
        '{"property_expression": "item[\'a\']", "binning": {"binning": "quantile",'
        ' "n_quantiles": 3}},'
        '{"property_expression": "item[\'b\']", "binning": {"binning":'
        ' "equal_width", "n_bins": 2, "range_min": 0.0, "range_max": 1.0}},'
        '{"property_expression": "item[\'c\']", "binning": {"binning":'
        ' "threshold", "edges": [0.5]}},'
        '{"property_expression": "item[\'d\']", "binning": {"binning": "stddev",'
        ' "k_values": [0.0]}},'
        '{"property_expression": "item[\'e\']", "binning": {"binning":'
        ' "categorical", "categories": ["x"], "include_other": true}}]}'
    )

    assert settings == GridStratificationSettings(
        "grid_stratification",
        (
            GridDimensionSettings("item['a']", QuantileBinningSettings("quantile", 3)),
            GridDimensionSettings(
                "item['b']", EqualWidthBinningSettings("equal_width", 2, 0.0, 1.0)
            ),
            GridDimensionSettings(
                "item['c']", ThresholdBinningSettings("threshold", (0.5,))
            ),
            GridDimensionSettings("item['d']", StdDevBinningSettings("stddev", (0.0,))),
            GridDimensionSettings(
                "item['e']",
                CategoricalBinningSettings("categorical", ("x",), include_other=True),
            ),
        ),
        None,
        2,
    )


def test_an_ordering_constraint_resolves_its_presentation_rules() -> None:
    settings = _list(
        '{"constraint_type": "ordering", "precedence_pairs": [{"before": "i1",'
        ' "after": "i2"}], "no_adjacent_property": "item[\'v\']",'
        ' "block_by_property": "item[\'g\']", "min_distance": 2,'
        ' "max_distance": 4, "practice_item_property": "item[\'practice\']"}'
    )

    assert settings == OrderingSettings(
        "ordering",
        (("i1", "i2"),),
        "item['v']",
        "item['g']",
        2,
        4,
        "item['practice']",
    )


@pytest.mark.parametrize(
    ("document", "reason"),
    [
        (
            '{"constraint_type": "ordering", "min_distance": 2}',
            "no_adjacent_property",
        ),
        (
            '{"constraint_type": "ordering", "max_distance": 2}',
            "block_by_property",
        ),
        (
            '{"constraint_type": "ordering", "randomize_within_blocks": true}',
            "randomize_within_blocks",
        ),
        (
            (
                '{"constraint_type": "quantile", "property_expression": '
                '"item[\'x\']", "n_quantiles": 1, "items_per_quantile": 1}'
            ),
            "quantile count",
        ),
        (
            (
                '{"constraint_type": "balance", "property_expression": '
                '"item[\'x\']", "target_counts": null, "tolerance": 1.5}'
            ),
            "balance tolerance",
        ),
    ],
)
def test_a_malformed_list_constraint_is_refused(document: str, reason: str) -> None:
    with pytest.raises(dx.ValidationError, match=reason):
        _list(document)


def test_every_bead_batch_constraint_resolves() -> None:
    coverage = _batch(
        '{"constraint_type": "coverage", "property_expression": "item[\'v\']",'
        ' "target_values": ["a", "b"], "min_coverage": 0.5}'
    )
    balance = _batch(
        '{"constraint_type": "balance", "property_expression": "item[\'v\']",'
        ' "target_distribution": {"a": 0.5, "b": 0.5}, "tolerance": 0.1}'
    )
    diversity = _batch(
        '{"constraint_type": "diversity", "property_expression": "item[\'v\']",'
        ' "max_lists_per_value": 2}'
    )
    occurrence = _batch(
        '{"constraint_type": "min_occurrence", "property_expression":'
        ' "item[\'v\']", "min_occurrences": 3}'
    )

    assert coverage == BatchCoverageSettings("coverage", "item['v']", ("a", "b"), 0.5)
    assert isinstance(balance, BatchBalanceSettings)
    assert dict(balance.target_distribution) == {"a": 0.5, "b": 0.5}
    assert diversity == BatchDiversitySettings("diversity", "item['v']", 2)
    assert occurrence == BatchMinOccurrenceSettings("min_occurrence", "item['v']", 3)


def test_a_batch_balance_distribution_sums_to_one() -> None:
    with pytest.raises(dx.ValidationError, match="sums to"):
        _batch(
            '{"constraint_type": "balance", "property_expression": "item[\'v\']",'
            ' "target_distribution": {"a": 0.5}, "tolerance": 0.1}'
        )
