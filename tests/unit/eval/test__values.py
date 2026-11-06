import math
from math import prod
from typing import TYPE_CHECKING, TypedDict, Unpack, cast

import pytest

from lofi._types import ItemId
from lofi.eval._values import (
    CONVENTION_LEVELS,
    MINIMUM_SEEDS,
    Aggregation,
    Comparison,
    ConventionVector,
    CoveragePenalty,
    GoldDerivation,
    MetricResult,
    PunctuationPolicy,
    Resampling,
    Sample,
    SelfArcPolicy,
    TrivialSpanPolicy,
    UnaryChainPolicy,
    convention_crossing,
    seed_summary,
)
from lofi.spec.evaluators.conventions import ConventionSettings
from lofi.stats import Interval, PValue, p_value_against_replicates

if TYPE_CHECKING:
    from types import ModuleType


@pytest.fixture
def torch_module() -> ModuleType:
    return pytest.importorskip("torch")


class _ConventionChanges(TypedDict, total=False):
    punctuation: PunctuationPolicy
    unary_chains: UnaryChainPolicy
    trivial_spans: TrivialSpanPolicy
    minimum_length: int
    aggregation: Aggregation
    coverage: CoveragePenalty
    self_arcs: SelfArcPolicy
    gold_derivation: GoldDerivation


def test_an_interval_reports_its_bounds_its_width_and_what_it_covers() -> None:
    interval = Interval(0.41, 0.57, 0.95)

    assert (interval.low, interval.high, interval.level) == (0.41, 0.57, 0.95)
    assert interval.width == pytest.approx(0.16)
    assert interval.covers(0.41)
    assert interval.covers(0.5)
    assert interval.covers(0.57)
    assert not interval.covers(0.58)


def test_a_point_interval_covers_only_its_point() -> None:
    point = Interval(0.5, 0.5, 0.9)

    assert point.width == 0.0
    assert point.covers(0.5)
    assert not point.covers(0.500001)


@pytest.mark.parametrize(
    ("low", "high", "level"),
    [
        (0.6, 0.4, 0.95),
        (0.1, 0.2, 0.0),
        (0.1, 0.2, 1.0),
        (0.1, 0.2, 95.0),
        (-math.inf, 0.2, 0.95),
        (0.1, math.nan, 0.95),
    ],
)
def test_an_ill_formed_interval_is_refused(
    low: float, high: float, level: float
) -> None:
    with pytest.raises(ValueError, match="interval"):
        Interval(low, high, level)


def test_intervals_are_equal_on_their_bounds_and_their_level() -> None:
    interval = Interval(0.1, 0.2, 0.95)

    assert interval == Interval(0.1, 0.2, 0.95)
    assert hash(interval) == hash(Interval(0.1, 0.2, 0.95))
    assert interval != Interval(0.1, 0.2, 0.5)
    assert interval != (0.1, 0.2, 0.95)


def test_a_resampling_scheme_names_its_unit_and_the_seed_it_drew_under() -> None:
    scheme = Resampling("sentence", 17)

    assert scheme.unit == "sentence"
    assert scheme.seed == 17
    assert scheme == Resampling("sentence", 17)
    assert hash(scheme) == hash(Resampling("sentence", 17))
    assert scheme != Resampling("sentence")
    assert scheme != Resampling("token", 17)


def test_a_scheme_that_drew_nothing_carries_no_seed() -> None:
    assert Resampling("item").seed is None


def test_a_scheme_without_a_unit_or_with_a_backward_seed_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        Resampling("")

    with pytest.raises(ValueError, match="negative"):
        Resampling("item", -1)

    with pytest.raises(TypeError):
        Resampling("item", seed=True)


def test_a_sample_carries_the_item_it_scored() -> None:
    sample = Sample(ItemId("s-17"), 0.75)

    assert sample.item_id == "s-17"
    assert sample.score == 0.75
    assert sample.weight == 1.0
    assert sample.stratum is None


def test_a_sample_carries_its_weight_and_the_stratum_it_was_drawn_from() -> None:
    sample = Sample(ItemId("s-17"), 0.75, weight=3.0, stratum="length-10-19")

    assert sample.weight == 3.0
    assert sample.stratum == "length-10-19"


def test_a_whole_number_score_is_stored_as_a_float() -> None:
    sample = Sample(ItemId("s-17"), 1, weight=2)

    assert isinstance(sample.score, float)
    assert isinstance(sample.weight, float)
    assert (sample.score, sample.weight) == (1.0, 2.0)


@pytest.mark.parametrize("score", [math.nan, math.inf, -math.inf])
def test_a_score_that_is_not_finite_is_refused(score: float) -> None:
    with pytest.raises(ValueError, match="not finite"):
        Sample(ItemId("s-17"), score)


def test_a_boolean_is_not_a_score() -> None:
    with pytest.raises(TypeError):
        Sample(ItemId("s-17"), score=True)


def test_a_weight_may_be_zero_but_never_negative() -> None:
    assert Sample(ItemId("s-17"), 0.5, weight=0.0).weight == 0.0

    with pytest.raises(ValueError, match="weight is negative"):
        Sample(ItemId("s-17"), 0.5, weight=-1.0)


def test_an_empty_id_or_stratum_is_refused() -> None:
    with pytest.raises(ValueError, match="empty"):
        Sample(ItemId(""), 0.5)

    with pytest.raises(ValueError, match="empty"):
        Sample(ItemId("s-17"), 0.5, stratum="")


def test_samples_are_equal_on_all_four_fields() -> None:
    sample = Sample(ItemId("s-17"), 0.5, weight=2.0, stratum="short")
    same = Sample(ItemId("s-17"), 0.5, weight=2.0, stratum="short")

    assert sample == same
    assert hash(sample) == hash(same)
    assert len({sample, same}) == 1
    assert sample != Sample(ItemId("s-18"), 0.5, weight=2.0, stratum="short")
    assert sample != Sample(ItemId("s-17"), 0.6, weight=2.0, stratum="short")
    assert sample != Sample(ItemId("s-17"), 0.5, weight=1.0, stratum="short")
    assert sample != Sample(ItemId("s-17"), 0.5, weight=2.0)
    assert sample != ("s-17", 0.5)


def test_a_metric_over_no_data_reports_no_value() -> None:
    empty = MetricResult(None, 0, Resampling("sentence"))

    assert empty.value is None
    assert empty.n == 0
    assert empty.ci is None
    assert dict(empty.components) == {}


def test_a_value_over_no_data_is_refused() -> None:
    with pytest.raises(ValueError, match="no data"):
        MetricResult(0.0, 0, Resampling("sentence"))


def test_a_metric_may_be_undefined_over_data_it_did_see() -> None:
    result = MetricResult(
        None,
        30,
        Resampling("sentence"),
        components={"tp": 0.0, "fp": 0.0, "fn": 5.0},
    )

    assert result.value is None
    assert result.n == 30
    assert result.components["fn"] == 5.0


def test_a_result_carries_its_count_its_scheme_and_its_interval() -> None:
    result = MetricResult(
        0.727,
        120,
        Resampling("sentence", 4),
        components={"tp": 8.0, "fp": 2.0, "fn": 4.0, "precision": 0.8, "recall": 2 / 3},
        ci=Interval(0.69, 0.76, 0.95),
    )

    assert result.value == 0.727
    assert result.n == 120
    assert result.resampling.unit == "sentence"
    assert result.resampling.seed == 4
    assert result.components["tp"] == 8.0
    assert result.components["recall"] == pytest.approx(2 / 3)
    assert result.ci is not None
    assert result.ci.covers(0.727)


def test_the_components_a_result_hands_back_are_a_copy() -> None:
    counts = {"tp": 8.0, "fp": 2.0}
    result = MetricResult(0.8, 10, Resampling("item"), components=counts)
    counts["tp"] = 0.0

    assert result.components["tp"] == 8.0
    assert len(result.components) == 2


def test_results_are_equal_however_their_components_were_ordered() -> None:
    result = MetricResult(0.5, 4, Resampling("item"), components={"tp": 1.0, "fp": 1.0})
    same = MetricResult(0.5, 4, Resampling("item"), components={"fp": 1.0, "tp": 1.0})

    assert result == same
    assert hash(result) == hash(same)
    assert len({result: "a", same: "b"}) == 1
    assert result != MetricResult(0.5, 4, Resampling("sentence"))
    assert result != MetricResult(0.5, 5, Resampling("item"))


def test_an_interval_around_an_absent_value_is_refused() -> None:
    with pytest.raises(ValueError, match="no value"):
        MetricResult(None, 12, Resampling("item"), ci=Interval(0.1, 0.2, 0.95))


def test_a_component_that_is_not_a_finite_number_is_refused() -> None:
    with pytest.raises(ValueError, match="not finite"):
        MetricResult(0.5, 4, Resampling("item"), components={"tp": math.nan})

    with pytest.raises(ValueError, match="empty"):
        MetricResult(0.5, 4, Resampling("item"), components={"": 1.0})


def test_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="negative"):
        MetricResult(None, -1, Resampling("item"))


def test_no_field_takes_a_tensor(torch_module: ModuleType) -> None:
    tensor = torch_module.tensor(1.0)

    with pytest.raises(TypeError):
        Sample(tensor, 0.5)

    with pytest.raises(TypeError):
        Sample(ItemId("s-17"), tensor)

    with pytest.raises(TypeError):
        Sample(ItemId("s-17"), 0.5, weight=tensor)

    with pytest.raises(TypeError):
        Sample(ItemId("s-17"), 0.5, stratum=tensor)

    with pytest.raises(TypeError):
        Interval(tensor, 1.0, 0.95)

    with pytest.raises(TypeError):
        Resampling("item", tensor)

    with pytest.raises(TypeError):
        MetricResult(tensor, 4, Resampling("item"))

    with pytest.raises(TypeError):
        MetricResult(0.5, tensor, Resampling("item"))

    with pytest.raises(TypeError):
        MetricResult(0.5, 4, tensor)

    with pytest.raises(TypeError):
        MetricResult(0.5, 4, Resampling("item"), components={"tp": tensor})

    with pytest.raises(TypeError):
        MetricResult(0.5, 4, Resampling("item"), ci=tensor)


def _vector(
    **changed: Unpack[_ConventionChanges],
) -> ConventionVector:
    return ConventionVector(
        punctuation=changed.get("punctuation", "strip"),
        unary_chains=changed.get("unary_chains", "collapse"),
        trivial_spans=changed.get("trivial_spans", "drop"),
        minimum_length=changed.get("minimum_length", 3),
        aggregation=changed.get("aggregation", "sentence"),
        coverage=changed.get("coverage", "zero"),
        self_arcs=changed.get("self_arcs", "forbid"),
        gold_derivation=changed.get("gold_derivation", "canonical"),
    )


def test_a_convention_vector_reports_the_eight_choices_behind_a_number() -> None:
    conventions = _vector()

    assert conventions.punctuation == "strip"
    assert conventions.unary_chains == "collapse"
    assert conventions.trivial_spans == "drop"
    assert conventions.minimum_length == 3
    assert conventions.aggregation == "sentence"
    assert conventions.coverage == "zero"
    assert conventions.self_arcs == "forbid"
    assert conventions.gold_derivation == "canonical"


def test_the_factor_view_carries_every_convention_in_declared_order() -> None:
    factors = _vector().factors

    assert tuple(factors) == tuple(CONVENTION_LEVELS)
    assert factors["punctuation"] == "strip"
    assert factors["minimum_length"] == 3


def test_flipping_any_one_convention_makes_a_different_vector() -> None:
    baseline = _vector()
    flipped = (
        _vector(punctuation="keep"),
        _vector(unary_chains="keep"),
        _vector(trivial_spans="keep"),
        _vector(minimum_length=0),
        _vector(aggregation="corpus"),
        _vector(coverage="ignore"),
        _vector(self_arcs="allow"),
        _vector(gold_derivation="oracle"),
    )

    assert all(vector != baseline for vector in flipped)

    assert _vector() == baseline
    assert hash(_vector()) == hash(baseline)
    assert baseline != "strip"


def test_an_evaluator_extends_the_vector_with_named_metric_conventions() -> None:
    baseline = _vector()
    extended = baseline.extended(
        {
            "alignment": "one-to-one",
            "alignment_tie_break": "least-gold-label",
            "decoder": "viterbi",
            "labeled": False,
        }
    )

    assert tuple(extended.factors.items()) == (
        *baseline.factors.items(),
        ("alignment", "one-to-one"),
        ("alignment_tie_break", "least-gold-label"),
        ("decoder", "viterbi"),
        ("labeled", False),
    )
    assert extended != baseline

    with pytest.raises(ValueError, match="already declares"):
        baseline.extended({"punctuation": "keep"})


def test_no_convention_can_be_rebound() -> None:
    conventions = _vector()

    for field in CONVENTION_LEVELS:
        with pytest.raises(AttributeError):
            setattr(conventions, field, "keep")

    assert not hasattr(conventions, "__dict__")
    assert conventions.punctuation == "strip"


def test_a_vector_is_built_from_declared_settings_field_by_field() -> None:
    declared = ConventionSettings(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage="ignore",
        self_arcs="allow",
        gold_derivation="oracle",
    )

    assert ConventionVector.from_settings(declared) == _vector(
        punctuation="keep",
        unary_chains="keep",
        trivial_spans="keep",
        minimum_length=0,
        aggregation="corpus",
        coverage="ignore",
        self_arcs="allow",
        gold_derivation="oracle",
    )
    assert (
        ConventionVector.from_settings(
            ConventionSettings(
                punctuation="strip",
                unary_chains="collapse",
                trivial_spans="drop",
                minimum_length=3,
                aggregation="sentence",
                coverage="zero",
                self_arcs="forbid",
                gold_derivation="canonical",
            )
        )
        == _vector()
    )


def test_a_convention_vector_reprs_as_the_call_that_rebuilds_it() -> None:
    assert repr(_vector()).startswith("ConventionVector(punctuation='strip', ")
    assert repr(_vector()).endswith("gold_derivation='canonical')")


def test_the_crossing_has_one_cell_per_product_of_the_declared_levels() -> None:
    crossing = convention_crossing()
    declared = prod(len(levels) for levels in CONVENTION_LEVELS.values())

    assert len(crossing) == declared
    assert len(set(crossing)) == declared


def test_every_declared_level_is_reachable_from_the_crossing() -> None:
    crossing = convention_crossing()

    for field, levels in CONVENTION_LEVELS.items():
        reached = {vector.factors[field] for vector in crossing}

        assert reached == set(levels)


def test_the_crossing_varies_the_last_convention_fastest() -> None:
    crossing = convention_crossing()

    assert crossing[0].gold_derivation == "canonical"
    assert crossing[1].gold_derivation == "oracle"
    assert crossing[0].punctuation == crossing[1].punctuation
    assert crossing[0].self_arcs != crossing[2].self_arcs


def _p_value() -> PValue:
    return p_value_against_replicates(
        3.0,
        [0.1, 0.4, 0.2, 0.9],
        tail="greater",
        null="permuted-arms",
        family="permutation",
        exact=False,
        seed=11,
    )


_FALSIFIER = "the ab arm scores no higher than the cfg arm"


def _comparison(*, falsifier: str = _FALSIFIER, smoke_test: bool = False) -> Comparison:
    return Comparison(
        "ab",
        "cfg",
        statistic=3.0,
        p_value=_p_value(),
        effect=0.42,
        falsifier=falsifier,
        smoke_test=smoke_test,
        family="grammar-comparison",
        ci=Interval(0.1, 0.7, 0.95),
    )


def test_a_comparison_carries_its_arms_its_statistic_and_its_falsifier() -> None:
    comparison = _comparison()

    assert comparison.left == "ab"
    assert comparison.right == "cfg"
    assert comparison.statistic == 3.0
    assert comparison.p_value == _p_value()
    assert comparison.effect == 0.42
    assert comparison.ci is not None
    assert comparison.ci.covers(0.42)
    assert comparison.falsifier == _FALSIFIER
    assert not comparison.smoke_test
    assert comparison.family == "grammar-comparison"

    with pytest.raises(AttributeError):
        comparison.__setattr__("_effect", 0.0)


def test_a_comparison_with_no_falsifier_is_refused() -> None:
    with pytest.raises(ValueError, match="a comparison's falsifier is empty"):
        _comparison(falsifier="")


def test_a_smoke_test_still_has_to_state_its_falsifier() -> None:
    stated = _comparison(
        smoke_test=True,
        falsifier="the undirected arm derives a string the directed arm cannot",
    )

    assert stated.smoke_test

    with pytest.raises(ValueError, match="a comparison's falsifier is empty"):
        _comparison(smoke_test=True, falsifier="")


def test_a_comparison_refuses_a_bare_float_where_a_p_value_belongs() -> None:
    with pytest.raises(TypeError, match="only a null test produces one"):
        Comparison(
            "ab",
            "cfg",
            statistic=3.0,
            p_value=cast("PValue", 0.03),
            effect=0.42,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="grammar-comparison",
        )


def test_a_comparison_refuses_a_smoke_test_flag_that_is_not_a_bool() -> None:
    with pytest.raises(TypeError, match="smoke-test flag is not a bool"):
        Comparison(
            "ab",
            "cfg",
            statistic=3.0,
            p_value=_p_value(),
            effect=0.42,
            falsifier=_FALSIFIER,
            smoke_test=cast("bool", 1),
            family="grammar-comparison",
        )


def test_a_comparison_refuses_an_interval_that_is_not_one() -> None:
    with pytest.raises(TypeError, match="a comparison's interval is not one"):
        Comparison(
            "ab",
            "cfg",
            statistic=3.0,
            p_value=_p_value(),
            effect=0.42,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="grammar-comparison",
            ci=cast("Interval", (0.1, 0.7)),
        )


def test_a_comparison_of_one_arm_with_itself_compares_nothing() -> None:
    with pytest.raises(ValueError, match="names one arm on both sides"):
        Comparison(
            "ab",
            "ab",
            statistic=1.0,
            p_value=_p_value(),
            effect=0.1,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="grammar-comparison",
        )


def test_a_comparison_needs_a_family_and_two_named_arms() -> None:
    with pytest.raises(ValueError, match="a comparison's family is empty"):
        Comparison(
            "ab",
            "cfg",
            statistic=1.0,
            p_value=_p_value(),
            effect=0.1,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="",
        )

    with pytest.raises(ValueError, match="a comparison's left arm is empty"):
        Comparison(
            "",
            "cfg",
            statistic=1.0,
            p_value=_p_value(),
            effect=0.1,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="grammar-comparison",
        )


def test_a_comparison_refuses_a_statistic_that_is_not_finite() -> None:
    with pytest.raises(ValueError, match="a comparison's statistic is not finite"):
        Comparison(
            "ab",
            "cfg",
            statistic=math.inf,
            p_value=_p_value(),
            effect=0.1,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="grammar-comparison",
        )

    with pytest.raises(ValueError, match="a comparison's effect is not finite"):
        Comparison(
            "ab",
            "cfg",
            statistic=1.0,
            p_value=_p_value(),
            effect=math.nan,
            falsifier=_FALSIFIER,
            smoke_test=False,
            family="grammar-comparison",
        )


def test_no_field_of_a_comparison_can_be_rebound() -> None:
    comparison = _comparison()

    for field in ("left", "right", "statistic", "effect", "falsifier", "smoke_test"):
        with pytest.raises(AttributeError):
            setattr(comparison, field, None)

    assert not hasattr(comparison, "__dict__")
    assert comparison.statistic == 3.0


def test_comparisons_are_equal_on_the_nine_fields_they_report() -> None:
    assert _comparison() == _comparison()
    assert hash(_comparison()) == hash(_comparison())
    assert _comparison() != _comparison(smoke_test=True)
    assert _comparison() != _comparison(falsifier="something else entirely")
    assert _comparison() != "ab against cfg"


def test_a_comparison_reprs_as_the_call_that_rebuilds_it() -> None:
    rendered = repr(_comparison())

    assert rendered.startswith("Comparison('ab', 'cfg', statistic=3.0, ")
    assert "smoke_test=False" in rendered
    assert f"falsifier={_FALSIFIER!r}" in rendered


def _seeded(values: dict[int, float]) -> list[MetricResult]:
    return [
        MetricResult(value, 20, Resampling("sentence", seed))
        for seed, value in values.items()
    ]


def test_a_seed_summary_reports_the_mean_with_the_maximum_beside_it() -> None:
    summary = seed_summary(_seeded({3: 0.70, 1: 0.80, 2: 0.72, 0: 0.74}))

    assert summary.value == pytest.approx(0.74)
    assert summary.n == 4
    assert summary.resampling.unit == "seed"
    assert summary.components["maximum"] == pytest.approx(0.80)
    assert summary.components["minimum"] == pytest.approx(0.70)
    assert summary.components["mean"] == pytest.approx(0.74)
    assert summary.components["seeds"] == 4.0
    assert summary.components["sd"] > 0.0


def test_a_seed_summary_does_not_depend_on_the_order_it_was_handed() -> None:
    results = _seeded({0: 0.74, 1: 0.80, 2: 0.72, 3: 0.70})

    assert seed_summary(results) == seed_summary(list(reversed(results)))


def test_a_seed_summary_over_too_few_seeds_is_refused() -> None:
    short = _seeded(dict.fromkeys(range(MINIMUM_SEEDS - 1), 0.7))

    with pytest.raises(ValueError, match=f"at least {MINIMUM_SEEDS} seeds"):
        seed_summary(short)


def test_a_seed_summary_refuses_one_run_reported_four_times() -> None:
    once = MetricResult(0.7, 20, Resampling("sentence", 0))

    with pytest.raises(ValueError, match="handed the seed 0 twice"):
        seed_summary([once, once, once, once])


def test_a_seed_summary_refuses_a_result_that_carries_no_seed() -> None:
    unseeded = _seeded({0: 0.7, 1: 0.8, 2: 0.9})
    unseeded.append(MetricResult(0.75, 20, Resampling("sentence")))

    with pytest.raises(ValueError, match="a result with no seed"):
        seed_summary(unseeded)


def test_a_seed_summary_refuses_a_result_with_no_value_or_another_unit() -> None:
    empty = _seeded({0: 0.7, 1: 0.8, 2: 0.9})
    empty.append(MetricResult(None, 0, Resampling("sentence", 3)))

    with pytest.raises(ValueError, match="a result with no value"):
        seed_summary(empty)

    mixed = _seeded({0: 0.7, 1: 0.8, 2: 0.9})
    mixed.append(MetricResult(0.75, 20, Resampling("token", 3)))

    with pytest.raises(ValueError, match="counted in different units"):
        seed_summary(mixed)


def test_a_seed_summary_refuses_a_bare_float() -> None:
    bare = cast("list[MetricResult]", [0.7, 0.8, 0.9, 0.75])

    with pytest.raises(TypeError, match="which is not a result"):
        seed_summary(bare)
