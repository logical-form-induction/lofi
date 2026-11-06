"""Declared represented-space relations execute through packaged run documents."""

import json
from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import leaf, train

from lofi.run import read_metrics
from lofi.spec import ConfigDocumentError

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("document", "relation"),
    [
        ("related-consistency.yaml", "consistency"),
        ("related-mapped.yaml", "mapped"),
        ("related-erase-then-decode.yaml", "erase-then-decode"),
    ],
)
def test_each_category_type_relation_trains_with_finite_coherence(
    document: str,
    relation: str,
    tmp_path: Path,
) -> None:
    point = train(document, tmp_path)
    recorded = read_metrics(point.directory)

    assert recorded is not None
    assert recorded.stage == "closed"
    assert leaf(point.resolved, "model.spaces.categories.relation.via") == relation
    assert "coherence" in recorded.residuals
    residual = recorded.residuals["coherence"]
    assert residual.steps > 0
    assert 0.0 <= residual.last <= 1.0


def test_exact_and_truncated_coherence_persist_their_mass_evidence(
    tmp_path: Path,
) -> None:
    exact = read_metrics(train("coherence-exact.yaml", tmp_path / "exact").directory)
    truncated = read_metrics(
        train("coherence-truncated.yaml", tmp_path / "truncated").directory
    )
    tightened = read_metrics(
        train(
            "coherence-truncated.yaml",
            tmp_path / "tightened",
            "approximation.residual_mass=0.0",
        ).directory
    )

    assert exact is not None
    assert truncated is not None
    assert tightened is not None
    exact_residual = exact.residuals["coherence"]
    truncated_residual = truncated.residuals["coherence"]
    tightened_residual = tightened.residuals["coherence"]
    assert exact_residual.steps > 0
    assert exact_residual.maximum == pytest.approx(0.0)
    assert truncated_residual.steps > 0
    assert truncated_residual.last > 0.0
    assert truncated_residual.last > exact_residual.last
    assert truncated_residual.maximum <= 0.9 + 1e-6
    assert tightened_residual.maximum <= 1e-6
    assert tightened_residual.last < truncated_residual.last
    default = truncated.contributed["compile"]["approximation-residual-mass"]
    assert default.kind == "measure"
    assert default.value == pytest.approx(0.9)
    overrides = truncated.contributed["compile"]["approximation-overrides"]
    assert overrides.kind == "text"
    assert isinstance(overrides.value, str)
    assert json.loads(overrides.value) == {"model.readouts.category-terms": 0.95}


def test_a_missing_residual_bound_is_refused_before_a_run_directory(
    tmp_path: Path,
) -> None:
    root = tmp_path / "refused"

    with pytest.raises(ConfigDocumentError, match=r"loss\.terms\.1.*residual_mass"):
        train(
            "coherence-missing-bound.yaml",
            root,
        )

    assert not root.exists()
