import json
from typing import TYPE_CHECKING

import pytest
from tests.conformance.evaluator import packaged_held_out_pass

from lofi._types import ItemId
from lofi.data.datasets import artifact_dataset_plugin
from lofi.data.expert import ExpertRating, ExpertRatings
from lofi.eval._alignment import Alignment, Contingency
from lofi.eval._expert import (
    PLUGIN,
    CategoryOccurrence,
    ExpertAnnotationPackage,
    ExpertCategoryEvaluator,
    evaluate_experts,
    export_annotation_package,
)
from lofi.eval._values import ConventionVector
from lofi.spec.datasets.expert import ExpertRatingsDatasetSpec
from lofi.spec.evaluators.alignment import AlignmentSettings
from lofi.spec.evaluators.expert import ExpertCategoryEvaluatorSpec
from lofi.stats import Bootstrap

if TYPE_CHECKING:
    from pathlib import Path


def _alignment() -> Alignment:
    return Alignment(
        {"I-A": "N", "I-B": "S", "I-C": "NP", "I-D": "S\\NP"},
        Contingency(
            {
                ("I-A", "N"): 3,
                ("I-B", "S"): 2,
                ("I-C", "NP"): 2,
                ("I-D", "S\\NP"): 1,
            },
            split="dev",
        ),
    )


def _package() -> ExpertAnnotationPackage:
    return export_annotation_package(
        (
            CategoryOccurrence(
                ItemId("second"),
                ("the", "cat"),
                "I-A",
                "N",
                '("N" "NP")',
            ),
            CategoryOccurrence(
                ItemId("first"),
                ("a", "dog"),
                "I-A",
                "N",
                '("N" "N")',
            ),
            CategoryOccurrence(ItemId("third"), ("runs",), "I-B", "S", '"S"'),
            CategoryOccurrence(ItemId("fourth"), ("Kim",), "I-C", "NP", '"NP"'),
            CategoryOccurrence(
                ItemId("fifth"),
                ("sleeps",),
                "I-D",
                "S\\NP",
                '"S\\\\NP"',
            ),
        ),
        _alignment(),
        examples_per_category=1,
    )


def _ratings(package: ExpertAnnotationPackage) -> ExpertRatings:
    labels = {
        "I-A": ("correct", "correct"),
        "I-B": ("correct", "incorrect"),
        "I-C": ("incorrect", "incorrect"),
        "I-D": ("incorrect", "incorrect"),
    }
    errors = {
        "I-A": ((), ()),
        "I-B": ((), ("wrong-arity",)),
        "I-C": (("wrong-result",), ("wrong-result",)),
        "I-D": (("wrong-arity",), ("wrong-arity", "wrong-result")),
    }
    return ExpertRatings(
        package.digest,
        (
            ExpertRating(category, annotator, rating, error)
            for category in labels
            for annotator, rating, error in zip(
                ("rater-a", "rater-b"), labels[category], errors[category], strict=True
            )
        ),
    )


def test_annotation_export_aligns_categories_and_selects_examples_stably() -> None:
    package = _package()

    assert tuple(category.category_id for category in package.categories) == (
        "I-A",
        "I-B",
        "I-C",
        "I-D",
    )
    first = package.categories[0]
    assert first.aligned_gold == "N"
    assert first.occurrences == 2
    assert tuple(example.item_id for example in first.examples) == (ItemId("first"),)
    assert len(package.digest) == 64
    assert json.loads(package.serialize())["digest"] == package.digest
    assert _package().digest == package.digest


def test_two_raters_yield_nominal_alpha_with_an_interval_and_taxonomy_rows() -> None:
    package = _package()
    result = evaluate_experts(
        package,
        _ratings(package),
        ("wrong-arity", "wrong-result"),
        bootstrap=Bootstrap(
            unit="induced-category", replicates=199, seed=7, level=0.95
        ),
    )

    assert result.agreement.result.value == pytest.approx(8 / 15)
    assert result.agreement.annotators == 2
    assert result.agreement.categories == 4
    assert result.agreement.result.ci is not None
    assert result.agreement.to_spec()["metric"] == "expert-rating-alpha"
    assert tuple(row.error for row in result.taxonomy) == (
        "wrong-arity",
        "wrong-result",
    )
    assert tuple(row.result.value for row in result.taxonomy) == (3 / 8, 3 / 8)
    assert all(row.result.ci is not None for row in result.taxonomy)
    assert tuple(row.to_spec()["metric"] for row in result.taxonomy) == (
        "expert-error-wrong-arity",
        "expert-error-wrong-result",
    )


def test_the_ratings_dataset_validates_once_at_its_json_boundary(
    tmp_path: Path,
) -> None:
    package = _package()
    source = tmp_path / "ratings.json"
    source.write_text(
        json.dumps(
            {
                "kind": "expert-ratings",
                "package_digest": package.digest,
                "ratings": [
                    {
                        "category_id": "I-A",
                        "annotator": "rater-a",
                        "rating": "correct",
                        "errors": [],
                    },
                    {
                        "category_id": "I-A",
                        "annotator": "rater-b",
                        "rating": "correct",
                        "errors": [],
                    },
                ],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )

    declaration = ExpertRatingsDatasetSpec(source=str(source))
    loaded = artifact_dataset_plugin(declaration).load(declaration)

    assert isinstance(loaded, ExpertRatings)
    assert loaded.package_digest == package.digest
    assert loaded.ratings[0] == ExpertRating("I-A", "rater-a", "correct", ())


def test_ratings_are_bound_to_the_exact_package_and_declared_taxonomy() -> None:
    package = _package()
    ratings = _ratings(package)
    other = ExpertAnnotationPackage(package.categories[:-1])
    bootstrap = Bootstrap(unit="induced-category", replicates=19, seed=3, level=0.95)

    with pytest.raises(ValueError, match="not"):
        evaluate_experts(
            other,
            ratings,
            ("wrong-arity", "wrong-result"),
            bootstrap=bootstrap,
        )

    with pytest.raises(ValueError, match="undeclared error labels"):
        evaluate_experts(
            package,
            ratings,
            ("wrong-arity",),
            bootstrap=bootstrap,
        )

    unknown = ExpertRatings(
        package.digest,
        (
            ExpertRating("I-UNKNOWN", "rater-a", "correct", ()),
            ExpertRating("I-UNKNOWN", "rater-b", "correct", ()),
        ),
    )
    with pytest.raises(ValueError, match="absent from their package"):
        evaluate_experts(
            package,
            unknown,
            ("wrong-arity", "wrong-result"),
            bootstrap=bootstrap,
        )


def test_duplicate_responses_and_an_empty_package_are_refused() -> None:
    response = ExpertRating("I-A", "rater-a", "correct", ())

    with pytest.raises(ValueError, match=r"rates category.*twice"):
        ExpertRatings("0" * 64, (response, response))

    with pytest.raises(ValueError, match="no induced category"):
        export_annotation_package((), _alignment(), examples_per_category=1)


def _score_pass(evaluator: ExpertCategoryEvaluator) -> None:
    held_out = packaged_held_out_pass()
    predictions = {}

    for outputs in held_out.published:
        predictions.update(evaluator.predict(outputs))

    for item, derivation in held_out.gold:
        gold = evaluator.annotate(item, derivation)

        for metric in evaluator.metrics:
            metric.update(predictions[item.item_id], gold, key=item.item_id)


def test_a_declared_evaluator_exports_the_pass_and_ingests_its_bound_ratings(
    tmp_path: Path,
) -> None:
    held_out = packaged_held_out_pass()
    conventions = ConventionVector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage="zero",
        self_arcs="allow",
        gold_derivation="canonical",
    )
    alignment = AlignmentSettings(
        "many-to-one",
        "largest-gold-marginal",
        "dev",
        "test",
        19,
        0.95,
        19,
        0.95,
    )
    first = ExpertCategoryEvaluator(
        held_out.grammar,
        readout="category-terms",
        conventions=conventions,
        alignment=alignment,
        ratings=None,
        taxonomy=("wrong-result",),
    )
    _score_pass(first)
    package = first.annotation_package()
    source = tmp_path / "expert-ratings.json"
    source.write_text(
        json.dumps(
            {
                "kind": "expert-ratings",
                "package_digest": package.digest,
                "ratings": [
                    {
                        "category_id": category.category_id,
                        "annotator": annotator,
                        "rating": "correct",
                        "errors": [],
                    }
                    for category in package.categories
                    for annotator in ("rater-a", "rater-b")
                ],
            }
        ),
        encoding="utf-8",
        newline="\n",
    )
    declaration = ExpertCategoryEvaluatorSpec.model_validate(
        {
            "kind": "expert-categories",
            "interval": {"replicates": 19, "level": 0.95},
            "readout": "category-terms",
            "ratings": {"kind": "expert-ratings", "source": str(source)},
            "taxonomy": ["wrong-result"],
            "conventions": dict(conventions.factors),
            "alignment": {
                "kind": "many-to-one",
                "tie_break": "largest-gold-marginal",
                "fit_on": "dev",
                "score_on": "test",
                "random_floor": {"replicates": 19, "level": 0.95},
                "stability": {"replicates": 19, "level": 0.95},
            },
        }
    )
    built = PLUGIN.construct(declaration, held_out.context)
    _score_pass(built)

    assert built.annotation_package().digest == package.digest
    assert tuple(metric.name for metric in built.metrics) == declaration.metrics()
    assert built.metrics[0].compute().value == 1.0
    assert built.metrics[1].compute().value == 0.0
