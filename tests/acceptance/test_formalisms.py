"""Every packaged formalism composes, synthesizes and trains through one route."""

from typing import TYPE_CHECKING

import pytest
from tests.acceptance.driver import composed, leaf, train

from lofi.formalism import ProfileUnsupportedError
from lofi.run import plan_training

if TYPE_CHECKING:
    from pathlib import Path


@pytest.mark.parametrize(
    ("document", "formalism"),
    [
        ("directed-pack.yaml", "declared"),
        ("ab-pack.yaml", "ab"),
        ("cfg-pack.yaml", "cfg"),
        ("nl-diamond-pack.yaml", "tlg"),
        ("undirected-ccg-pack.yaml", "undirected-ccg"),
    ],
)
def test_each_pack_compiles_draws_data_and_trains(
    tmp_path: Path,
    document: str,
    formalism: str,
) -> None:
    point = train(document, tmp_path / formalism)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.grammar.formalism")
        == formalism
    )
    assert point.rows


def test_nl_diamond_refuses_two_region_items() -> None:
    run = composed("nl-diamond-refused.yaml")

    with pytest.raises(ProfileUnsupportedError, match="fan-out"):
        plan_training(run)


def test_declared_capacity_hypotheses_reach_the_compile_result() -> None:
    run = composed("composition-point.yaml")
    report = plan_training(run).learner.grammar.grammar.capacity

    assert report.rule_family == "pure_ccg"
    assert report.theorem_applicable
    assert report.result == "proper_superclass_of_context_free"


def test_an_all_of_normal_form_composes_draws_and_trains(tmp_path: Path) -> None:
    point = train("all-of-normal-form.yaml", tmp_path)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.grammar.normal_form.kind")
        == "all-of"
    )
    assert point.rows


@pytest.mark.parametrize(
    ("document", "kind"),
    [
        ("product-matching.yaml", "product"),
        ("subsumption-matching.yaml", "subsumption"),
    ],
)
def test_composite_matching_composes_draws_and_trains(
    tmp_path: Path, document: str, kind: str
) -> None:
    point = train(document, tmp_path / kind)

    assert (
        leaf(point.resolved, "data.datasets.train.dataset.grammar.matching.kind")
        == kind
    )
    assert point.rows
