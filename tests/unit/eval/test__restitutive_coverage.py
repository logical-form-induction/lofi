"""Tests for restitutive-scope coverage over a logical-form lexicon."""

from typing import TYPE_CHECKING, cast

import pytest
from tests.acceptance.driver import composed

from lofi.eval import restitutive_coverage
from lofi.run import plan_training

if TYPE_CHECKING:
    from lofi.data.synthetic.spec.corpus import CheckedSyntheticCorpus


def _interpreted_dataset() -> CheckedSyntheticCorpus:
    planned = plan_training(composed("restitutive-coverage.yaml"))
    return cast("CheckedSyntheticCorpus", planned.checked_dataset("train"))


def test_coverage_is_derived_from_declared_classes_and_typed_scope() -> None:
    dataset = _interpreted_dataset()
    interpretation = dataset.source_interpretation
    assert interpretation is not None
    try:
        rows = restitutive_coverage(
            dataset.source_grammar,
            interpretation,
            change_class="change-state",
            modifier_class="repetition",
        )
    finally:
        interpretation.service.close()

    by_form = {row.form: row for row in rows}
    assert by_form["breaks"].covered
    assert by_form["breaks"].change == "become"
    assert by_form["breaks"].modifier == "again"
    assert not by_form["taps"].covered
    assert by_form["taps"].change is None


def test_coverage_refuses_an_undeclared_semantic_class() -> None:
    dataset = _interpreted_dataset()
    interpretation = dataset.source_interpretation
    assert interpretation is not None
    try:
        with pytest.raises(ValueError, match="no constant in change class"):
            restitutive_coverage(
                dataset.source_grammar,
                interpretation,
                change_class="undeclared-change",
                modifier_class="repetition",
            )
    finally:
        interpretation.service.close()
