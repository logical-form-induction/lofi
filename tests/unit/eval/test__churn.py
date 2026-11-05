"""Tests for paired rank churn."""

from typing import cast

import pytest

from lofi.eval._churn import RankedScore, rank_churn


def test_rank_churn_reports_zero_instead_of_dropping_an_unchanged_ranking() -> None:
    result = rank_churn(
        (
            RankedScore("a", "constrained", 0.9),
            RankedScore("b", "constrained", 0.7),
            RankedScore("a", "unconstrained", 0.8),
            RankedScore("b", "unconstrained", 0.6),
        ),
        left="constrained",
        right="unconstrained",
    )

    assert result.left == "constrained"
    assert result.right == "unconstrained"
    assert result.methods == ("a", "b")
    assert result.changed == 0
    assert result.mean_absolute_change == 0.0
    assert result.to_spec() == {
        "left": "constrained",
        "right": "unconstrained",
        "methods": 2,
        "changed": 0,
        "value": 0.0,
    }


def test_rank_churn_uses_paired_midranks() -> None:
    result = rank_churn(
        (
            RankedScore("a", "constrained", 0.9),
            RankedScore("b", "constrained", 0.8),
            RankedScore("c", "constrained", 0.7),
            RankedScore("a", "unconstrained", 0.7),
            RankedScore("b", "unconstrained", 0.8),
            RankedScore("c", "unconstrained", 0.9),
        ),
        left="constrained",
        right="unconstrained",
    )

    assert result.changed == 2
    assert result.mean_absolute_change == pytest.approx(4.0 / 3.0)


def test_rank_churn_refuses_incomplete_repeated_or_undeclared_scores() -> None:
    with pytest.raises(ValueError, match="same nonempty method set"):
        rank_churn(
            (
                RankedScore("a", "constrained", 0.9),
                RankedScore("b", "unconstrained", 0.8),
            ),
            left="constrained",
            right="unconstrained",
        )

    with pytest.raises(ValueError, match="scored twice"):
        rank_churn(
            (
                RankedScore("a", "constrained", 0.9),
                RankedScore("a", "constrained", 0.8),
            ),
            left="constrained",
            right="unconstrained",
        )

    with pytest.raises(ValueError, match="not 'third'"):
        rank_churn(
            (RankedScore("a", "third", 0.9),),
            left="constrained",
            right="unconstrained",
        )

    with pytest.raises(TypeError, match="non-score"):
        rank_churn(
            (cast("RankedScore", 0.9),),
            left="constrained",
            right="unconstrained",
        )


def test_rank_churn_requires_two_named_distinct_conditions() -> None:
    with pytest.raises(ValueError, match="distinct conditions"):
        rank_churn((), left="same", right="same")

    with pytest.raises(ValueError, match="left ranking condition is empty"):
        rank_churn((), left="", right="other")
