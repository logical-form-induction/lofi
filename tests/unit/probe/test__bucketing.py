from collections import Counter
from math import inf, nan
from typing import cast

import pytest

from lofi.probe._bucketing import (
    AbstractionBucket,
    AbstractionCase,
    AbstractionFeatures,
    BucketingReport,
    bucket_abstraction,
)


def a_case(
    sentence_length: int,
    derivation_depth: int,
    *,
    matched: bool,
    ambiguity: int = 1,
    span_length: int = 1,
) -> AbstractionCase:
    return AbstractionCase(
        AbstractionFeatures(
            sentence_length=sentence_length,
            derivation_depth=derivation_depth,
            ambiguity=ambiguity,
            span_length=span_length,
        ),
        matched=matched,
    )


def planted_depth() -> tuple[AbstractionCase, ...]:
    return tuple(
        a_case(8 + repetition, depth, matched=depth >= 3, ambiguity=repetition)
        for depth in range(6)
        for repetition in range(2)
    )


def diagnose(
    cases: tuple[AbstractionCase, ...],
    *,
    well_cutoff: float = 0.75,
    min_bucket_size: int = 2,
    min_accuracy_gap: float = 0.5,
) -> BucketingReport:
    return bucket_abstraction(
        cases,
        well_cutoff=well_cutoff,
        min_bucket_size=min_bucket_size,
        min_accuracy_gap=min_accuracy_gap,
    )


def test_a_planted_derivation_depth_boundary_is_recovered() -> None:
    cases = planted_depth()
    report = diagnose(cases, min_bucket_size=3)

    assert [
        (bucket.conditions, bucket.accuracy, bucket.size, bucket.label)
        for bucket in report.buckets
    ] == [
        (("derivation_depth <= 2",), 0.0, 6, "under"),
        (("derivation_depth > 2",), 1.0, 6, "well"),
    ]
    assert Counter(
        case for bucket in report.buckets for case in bucket.cases
    ) == Counter(cases)
    assert report.size == len(cases)


def test_uniform_outcomes_form_one_bucket_instead_of_splitting_anyway() -> None:
    cases = tuple(a_case(length, length % 3, matched=True) for length in range(4, 12))
    report = diagnose(cases, min_accuracy_gap=0.0)

    assert len(report.buckets) == 1
    assert report.buckets[0].conditions == ()
    assert report.buckets[0].accuracy == 1.0
    assert report.buckets[0].size == len(cases)
    assert report.buckets[0].label == "well"


def test_input_permutation_changes_neither_the_partition_nor_leaf_membership() -> None:
    cases = planted_depth()
    expected = diagnose(cases, min_bucket_size=3)
    permuted = cases[::2] + cases[1::2]

    assert diagnose(tuple(reversed(cases)), min_bucket_size=3) == expected
    assert diagnose(permuted, min_bucket_size=3) == expected


def test_equal_gain_uses_feature_order_and_then_the_lowest_cut() -> None:
    outcomes = (True, True, False, False, True, True)
    cases = tuple(
        a_case(
            value,
            value,
            matched=outcome,
            ambiguity=value,
            span_length=value,
        )
        for value, outcome in enumerate(outcomes, start=1)
    )
    report = diagnose(cases, min_bucket_size=2, min_accuracy_gap=0.4)

    assert report.buckets[0].conditions == ("sentence_length <= 2",)
    assert all(
        bucket.conditions[0] == "sentence_length > 2" for bucket in report.buckets[1:]
    )


def test_an_accuracy_gap_must_exceed_its_declared_minimum() -> None:
    report = diagnose(planted_depth(), min_bucket_size=3, min_accuracy_gap=1.0)

    assert len(report.buckets) == 1


def test_an_exact_gap_strictly_above_its_float_threshold_is_admissible() -> None:
    cases = tuple(
        a_case(length, 0, matched=matched)
        for length, matched in zip(
            range(1, 7),
            (False, False, False, True, False, False),
            strict=True,
        )
    )
    report = diagnose(
        cases,
        well_cutoff=0.2,
        min_bucket_size=3,
        min_accuracy_gap=1 / 3,
    )

    assert [bucket.conditions for bucket in report.buckets] == [
        ("sentence_length <= 3",),
        ("sentence_length > 3",),
    ]


def test_the_well_cutoff_is_inclusive_and_recorded_with_the_split_policy() -> None:
    cases = (a_case(4, 1, matched=True), a_case(5, 1, matched=False))
    report = diagnose(
        cases,
        well_cutoff=0.5,
        min_bucket_size=2,
        min_accuracy_gap=0.0,
    )

    assert report.buckets[0].accuracy == 0.5
    assert report.buckets[0].label == "well"
    assert report.well_cutoff == 0.5
    assert report.min_bucket_size == 2
    assert report.min_accuracy_gap == 0.0


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("sentence_length", 0, "sentence length"),
        ("sentence_length", True, "sentence length"),
        ("derivation_depth", -1, "derivation depth"),
        ("ambiguity", -1, "ambiguity"),
        ("span_length", 0, "span length"),
    ],
)
def test_abstraction_features_refuse_invalid_integers(
    field: str, value: int, message: str
) -> None:
    fields = {
        "sentence_length": 4,
        "derivation_depth": 1,
        "ambiguity": 0,
        "span_length": 2,
    }
    fields[field] = value

    with pytest.raises(
        TypeError if isinstance(value, bool) else ValueError, match=message
    ):
        AbstractionFeatures(**fields)


def test_a_span_longer_than_its_sentence_is_refused() -> None:
    with pytest.raises(ValueError, match="span length"):
        AbstractionFeatures(
            sentence_length=3,
            derivation_depth=1,
            ambiguity=0,
            span_length=4,
        )


def test_an_abstraction_case_requires_declared_features_and_a_boolean_match() -> None:
    features = AbstractionFeatures(
        sentence_length=3,
        derivation_depth=1,
        ambiguity=0,
        span_length=2,
    )

    with pytest.raises(TypeError, match="AbstractionFeatures"):
        AbstractionCase(cast("AbstractionFeatures", None), matched=True)

    with pytest.raises(TypeError, match="boolean"):
        AbstractionCase(features, matched=cast("bool", 1))


@pytest.mark.parametrize(
    ("arguments", "error", "message"),
    [
        ({"well_cutoff": nan}, ValueError, "well cutoff"),
        ({"well_cutoff": inf}, ValueError, "well cutoff"),
        ({"well_cutoff": -0.1}, ValueError, "well cutoff"),
        ({"well_cutoff": 1.1}, ValueError, "well cutoff"),
        ({"min_bucket_size": 0}, ValueError, "bucket size"),
        ({"min_bucket_size": True}, TypeError, "bucket size"),
        ({"min_accuracy_gap": nan}, ValueError, "accuracy gap"),
        ({"min_accuracy_gap": -0.1}, ValueError, "accuracy gap"),
        ({"min_accuracy_gap": 1.1}, ValueError, "accuracy gap"),
    ],
)
def test_bucketing_refuses_an_invalid_policy(
    arguments: dict[str, float | int], error: type[Exception], message: str
) -> None:
    policy: dict[str, float | int] = {
        "well_cutoff": 0.8,
        "min_bucket_size": 1,
        "min_accuracy_gap": 0.0,
    }
    policy.update(arguments)

    with pytest.raises(error, match=message):
        bucket_abstraction(
            (a_case(4, 1, matched=True),),
            well_cutoff=cast("float", policy["well_cutoff"]),
            min_bucket_size=cast("int", policy["min_bucket_size"]),
            min_accuracy_gap=cast("float", policy["min_accuracy_gap"]),
        )


def test_bucketing_refuses_no_cases_or_an_undeclared_case() -> None:
    with pytest.raises(ValueError, match="at least one"):
        diagnose(())

    with pytest.raises(TypeError, match="AbstractionCase"):
        diagnose((cast("AbstractionCase", None),))


def test_the_report_values_are_immutable() -> None:
    report = diagnose(planted_depth(), min_bucket_size=3)
    bucket = report.buckets[0]
    features = bucket.cases[0].features
    feature_field = "_sentence_length"
    bucket_field = "_accuracy"
    report_field = "_buckets"

    with pytest.raises(AttributeError, match="immutable"):
        setattr(features, feature_field, 99)

    with pytest.raises(AttributeError, match="immutable"):
        setattr(bucket, bucket_field, 0.5)

    with pytest.raises(AttributeError, match="immutable"):
        setattr(report, report_field, ())


def test_report_construction_refuses_no_bucket() -> None:
    with pytest.raises(ValueError, match="at least one bucket"):
        BucketingReport(
            (),
            well_cutoff=0.5,
            min_bucket_size=1,
            min_accuracy_gap=0.0,
        )


def test_report_construction_refuses_a_bucket_labeled_under_another_cutoff() -> None:
    cases = (a_case(4, 1, matched=True), a_case(5, 1, matched=False))
    bucket = AbstractionBucket((), cases, well_cutoff=0.75)

    with pytest.raises(ValueError, match="well cutoff"):
        BucketingReport(
            (bucket,),
            well_cutoff=0.5,
            min_bucket_size=1,
            min_accuracy_gap=0.0,
        )


def test_bucket_construction_refuses_no_case() -> None:
    with pytest.raises(ValueError, match="at least one case"):
        AbstractionBucket((), (), well_cutoff=0.5)


@pytest.mark.parametrize(
    "condition",
    [
        "depth <= 1",
        "sentence_length < 1",
        "sentence_length <= 1.5",
        "sentence_length  <= 1",
    ],
)
def test_bucket_construction_refuses_an_undeclared_condition(condition: str) -> None:
    cases = (a_case(1, 0, matched=True),)

    with pytest.raises(ValueError, match="condition"):
        AbstractionBucket((condition,), cases, well_cutoff=0.5)


def test_bucket_construction_refuses_a_case_outside_its_conditions() -> None:
    cases = (a_case(4, 0, matched=True),)

    with pytest.raises(ValueError, match=r"does not satisfy.*sentence_length <= 3"):
        AbstractionBucket(("sentence_length <= 3",), cases, well_cutoff=0.5)


def test_report_construction_rechecks_a_bucket_condition() -> None:
    cases = (a_case(4, 0, matched=True),)
    bucket = AbstractionBucket(("sentence_length > 3",), cases, well_cutoff=0.5)
    object.__setattr__(bucket, "_conditions", ("undeclared_feature <= 9",))

    with pytest.raises(ValueError, match="condition"):
        BucketingReport(
            (bucket,),
            well_cutoff=0.5,
            min_bucket_size=1,
            min_accuracy_gap=0.0,
        )
