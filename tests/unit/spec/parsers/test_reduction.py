"""Tests for tensor log-weight reduction declarations."""

from typing import cast

import pytest

from lofi.spec.parsers.reduction import (
    LogSumExpReductionSettings,
    LogSumExpReductionSpec,
    MaximumReductionSettings,
    MaximumReductionSpec,
    ReductionDeclaration,
    ReductionSpec,
)


class Resolver:
    """Resolve each reduction to a distinct test value."""

    def log_sum_exp(self) -> str:
        return "summed"

    def maximum(self) -> str:
        return "largest"


@pytest.mark.parametrize(
    ("document", "expected", "resolved"),
    [
        (
            {"kind": "log-sum-exp"},
            LogSumExpReductionSpec(),
            LogSumExpReductionSettings(),
        ),
        (
            {"kind": "maximum"},
            MaximumReductionSpec(),
            MaximumReductionSettings(),
        ),
    ],
)
def test_each_tag_reaches_its_reduction_settings(
    document: dict[str, str],
    expected: LogSumExpReductionSpec | MaximumReductionSpec,
    resolved: LogSumExpReductionSettings | MaximumReductionSettings,
) -> None:
    declared = ReductionSpec.model_validate(document)

    assert declared == expected
    assert cast("ReductionDeclaration", declared).settings() == resolved


def test_each_setting_resolves_without_a_tag_or_identity_branch() -> None:
    resolver = Resolver()

    assert LogSumExpReductionSettings().resolve(resolver) == "summed"
    assert MaximumReductionSettings().resolve(resolver) == "largest"
