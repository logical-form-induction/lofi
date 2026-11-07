from typing import TYPE_CHECKING

import didactic.api as dx
import pytest

from lofi.spec.evaluators.interannotator import (
    InterannotatorSettings,
    InterannotatorSpec,
)

if TYPE_CHECKING:
    from lofi._types import JsonValue


def test_a_spearman_only_ceiling_at_2000_replicates_is_declared() -> None:
    declared = InterannotatorSpec.model_validate(
        {"coefficients": ["spearman"], "replicates": 2000, "level": 0.95}
    )

    assert declared.settings() == InterannotatorSettings(("spearman",), 2000, 0.95)


@pytest.mark.parametrize(
    "payload",
    [
        {"coefficients": [], "replicates": 10, "level": 0.95},
        {"coefficients": ["spearman", "spearman"], "replicates": 10, "level": 0.95},
        {"coefficients": ["kendall"], "replicates": 10, "level": 0.95},
        {"coefficients": ["pearson"], "replicates": 0, "level": 0.95},
        {"coefficients": ["pearson"], "replicates": 10, "level": 1.0},
    ],
)
def test_an_ill_declared_ceiling_is_refused(payload: dict[str, JsonValue]) -> None:
    with pytest.raises(dx.ValidationError):
        InterannotatorSpec.model_validate(payload)
