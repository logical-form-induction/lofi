import math
from typing import cast

import pytest
import torch

from lofi.judgment._values import (
    AgreementCeiling,
    LinkingDomain,
    SentenceScore,
    SpecialCase,
)
from lofi.stats import Interval

COUNTS = {"items": 50, "responses": 250}


def test_a_ceiling_reports_the_figure_the_interval_and_the_counts() -> None:
    ceiling = AgreementCeiling(0.5, 0.4, 0.6, **COUNTS)

    assert ceiling.value == pytest.approx(0.5)
    assert ceiling.interval == pytest.approx((0.4, 0.6))
    assert ceiling.items == COUNTS["items"]
    assert ceiling.responses == COUNTS["responses"]


@pytest.mark.parametrize(
    ("value", "lower", "upper"),
    [(0.5, 0.6, 0.7), (0.5, 0.3, 0.4), (0.5, 0.6, 0.4)],
)
def test_an_interval_that_misses_the_value_is_refused(
    value: float, lower: float, upper: float
) -> None:
    with pytest.raises(ValueError, match="must bracket its value"):
        AgreementCeiling(value, lower, upper, **COUNTS)


def test_an_interval_may_touch_the_value_at_either_end() -> None:
    at_lower = AgreementCeiling(0.4, 0.4, 0.6, **COUNTS)
    at_upper = AgreementCeiling(0.6, 0.4, 0.6, **COUNTS)

    assert at_lower.interval[0] == pytest.approx(at_lower.value)
    assert at_upper.interval[1] == pytest.approx(at_upper.value)


def test_a_figure_that_is_not_a_number_is_refused() -> None:
    with pytest.raises(ValueError, match="is not finite"):
        AgreementCeiling(float("nan"), 0.4, 0.6, **COUNTS)


@pytest.mark.parametrize("count", [50.0, "50", None])
def test_a_count_that_is_not_a_whole_number_is_refused(
    count: float | str | None,
) -> None:
    with pytest.raises(TypeError, match="is not an integer"):
        AgreementCeiling(0.5, 0.4, 0.6, items=cast("int", count), responses=250)


def test_a_negative_count_is_refused() -> None:
    with pytest.raises(ValueError, match="is negative"):
        AgreementCeiling(0.5, 0.4, 0.6, items=-1, responses=250)


SCORE = Interval(-10.0, 10.0)
LENGTH = Interval(1.0, 40.0)
SLOPE = Interval(0.0, 4.0)


def test_a_domain_reads_back_its_argument_columns_and_parameters() -> None:
    domain = LinkingDomain(
        "score", {"score": SCORE, "length": LENGTH}, {"slope": SLOPE}
    )

    assert domain.argument == "score"
    assert domain.columns == {"score": SCORE, "length": LENGTH}
    assert domain.parameters == {"slope": SLOPE}
    assert domain.readouts == ("score", "length")
    assert domain.parameter_names == ("slope",)


def test_a_domain_refuses_an_interval_with_no_interior() -> None:
    with pytest.raises(ValueError, match="has no interior"):
        LinkingDomain("score", {"score": Interval(1.0, 1.0)})


def test_a_domain_refuses_a_confidence_interval() -> None:
    with pytest.raises(TypeError, match="not a domain interval"):
        LinkingDomain("score", {"score": Interval(0.0, 1.0, 0.95)})


def test_a_domain_defaults_to_no_fitted_parameters() -> None:
    domain = LinkingDomain("score", {"score": SCORE})

    assert domain.parameters == {}
    assert domain.parameter_names == ()


def test_a_domain_keeps_its_columns_and_parameters_in_declaration_order() -> None:
    domain = LinkingDomain("b", {"b": SCORE, "a": LENGTH}, {"z": SLOPE, "y": SLOPE})

    assert domain.readouts == ("b", "a")
    assert domain.parameter_names == ("z", "y")


def test_a_domain_hands_out_no_live_mapping() -> None:
    columns = {"score": SCORE}
    domain = LinkingDomain("score", columns)
    columns["length"] = LENGTH

    assert domain.readouts == ("score",)

    with pytest.raises(TypeError):
        cast("dict[str, Interval]", domain.columns)["length"] = LENGTH


def test_a_domain_whose_argument_is_not_a_column_is_refused() -> None:
    with pytest.raises(ValueError, match="not among its columns"):
        LinkingDomain("length", {"score": SCORE})


@pytest.mark.parametrize("name", ["", None])
def test_a_domain_with_a_blank_or_missing_name_is_refused(name: str | None) -> None:
    with pytest.raises((ValueError, TypeError)):
        LinkingDomain(cast("str", name), {"score": SCORE})

    with pytest.raises((ValueError, TypeError)):
        LinkingDomain("score", {"score": SCORE, cast("str", name): LENGTH})

    with pytest.raises((ValueError, TypeError)):
        LinkingDomain("score", {"score": SCORE}, {cast("str", name): SLOPE})


def test_domains_declaring_the_same_names_and_intervals_are_one_value() -> None:
    domain = LinkingDomain("score", {"score": SCORE}, {"slope": SLOPE})
    same = LinkingDomain("score", {"score": SCORE}, {"slope": SLOPE})
    wider = LinkingDomain("score", {"score": SCORE}, {"slope": Interval(0.0, 5.0)})

    assert domain == same
    assert hash(domain) == hash(same)
    assert domain != wider
    assert domain != "score"
    assert "LinkingDomain('score'" in repr(domain)


def test_a_special_case_reads_back_its_name_and_parameters() -> None:
    case = SpecialCase("identity", [1, 0.0])

    assert case.name == "identity"
    assert case.parameters == (1.0, 0.0)
    assert repr(case) == "SpecialCase('identity', (1.0, 0.0))"


def test_a_special_case_may_take_no_parameters() -> None:
    assert SpecialCase("raw", ()).parameters == ()


def test_a_special_case_without_a_name_is_refused() -> None:
    with pytest.raises(ValueError, match="is empty"):
        SpecialCase("", (1.0,))


@pytest.mark.parametrize("value", [math.nan, math.inf])
def test_a_special_case_with_a_parameter_that_is_not_finite_is_refused(
    value: float,
) -> None:
    with pytest.raises(ValueError, match="is not finite"):
        SpecialCase("identity", (value,))


def test_special_cases_with_the_same_name_and_parameters_are_one_value() -> None:
    assert SpecialCase("identity", (1.0,)) == SpecialCase("identity", (1.0,))
    assert SpecialCase("identity", (1.0,)) != SpecialCase("identity", (2.0,))
    assert SpecialCase("identity", (1.0,)) != SpecialCase("raw", (1.0,))
    assert SpecialCase("identity", (1.0,)) != ("identity", (1.0,))
    assert hash(SpecialCase("identity", (1.0,))) == hash(
        SpecialCase("identity", (1.0,))
    )


def test_a_sentence_score_reads_back_its_tokens_and_totals() -> None:
    score = SentenceScore(
        ("the", "dog", "barked"),
        torch.tensor([-1.0, -2.0, -3.0]),
        torch.tensor([-0.5, -4.0, -6.0]),
    )

    assert score.tokens == ("the", "dog", "barked")
    assert score.length == 3
    assert float(score.log_probability) == pytest.approx(-6.0)
    assert float(score.unigram_log_probability) == pytest.approx(-10.5)
    assert torch.equal(score.log_probabilities, torch.tensor([-1.0, -2.0, -3.0]))
    assert torch.equal(
        score.unigram_log_probabilities, torch.tensor([-0.5, -4.0, -6.0])
    )
    assert repr(score) == (
        "SentenceScore(tokens=('the', 'dog', 'barked'), log_probability=-6.0, "
        "unigram_log_probability=-10.5)"
    )


def test_a_sentence_score_over_no_tokens_is_refused() -> None:
    with pytest.raises(ValueError, match="covers no tokens"):
        SentenceScore((), torch.zeros(0), torch.zeros(0))


def test_a_sentence_score_with_an_empty_token_is_refused() -> None:
    with pytest.raises(ValueError, match="scored token is empty"):
        SentenceScore(("the", ""), torch.zeros(2), torch.zeros(2))


@pytest.mark.parametrize(
    ("log_probabilities", "unigram", "message"),
    [
        (torch.zeros(2), torch.zeros(3), "over 3 tokens"),
        (torch.zeros(3, 1), torch.zeros(3), "over 3 tokens"),
        (torch.tensor([0.0, 0.5, 0.0]), torch.zeros(3), "positive value"),
        (torch.zeros(3), torch.tensor([0.0, math.inf, 0.0]), "not finite"),
        (torch.zeros(3), torch.tensor([0.0, math.nan, 0.0]), "not finite"),
    ],
)
def test_a_sentence_score_whose_vectors_are_not_log_probabilities_is_refused(
    log_probabilities: torch.Tensor, unigram: torch.Tensor, message: str
) -> None:
    with pytest.raises(ValueError, match=message):
        SentenceScore(("a", "b", "c"), log_probabilities, unigram)


def test_a_sentence_score_with_an_integer_vector_is_refused() -> None:
    with pytest.raises(TypeError, match="not a floating-point kind"):
        SentenceScore(("a", "b"), torch.zeros(2, dtype=torch.int64), torch.zeros(2))
