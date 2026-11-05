import pytest

from lofi.deduce import Counting, InsideChart, SpanIndex, Viterbi


def test_a_category_sums_its_weight_over_its_tags() -> None:
    chart = InsideChart[str, int](SpanIndex(2), Counting(), normal_form="eisner")

    chart.add((0, 1), "S", "FC", 2)
    chart.add((0, 1), "NP", "OTHER", 1)
    chart.add((0, 1), "S", "OTHER", 3)
    chart.add((0, 1), "S", "FC", 4)

    assert chart.categories_at((0, 1)) == (("S", 9), ("NP", 1))
    assert chart.tagged_at((0, 1)) == (
        ("S", "FC", 6),
        ("S", "OTHER", 3),
        ("NP", "OTHER", 1),
    )
    assert chart.inside((0, 1), "S") == 9
    assert chart.tagged((0, 1), "S", "FC") == 6
    assert chart.num_categories == 2
    assert chart.normal_form == "eisner"


def test_an_absent_category_weighs_zero_and_an_empty_span_holds_nothing() -> None:
    chart = InsideChart[str, float](SpanIndex(2), Viterbi())

    chart.add((0, 2), "S", "OTHER", -1.5)

    assert chart.inside((0, 2), "NP") == Viterbi().zero
    assert chart.tagged((0, 2), "S", "FC") == Viterbi().zero
    assert chart.categories_at((0, 1)) == ()
    assert chart.tagged_at((1, 2)) == ()
    assert chart.normal_form == "none"


def test_a_span_outside_the_index_is_refused() -> None:
    chart = InsideChart[str, int](SpanIndex(2), Counting())

    with pytest.raises(ValueError, match="outside the chart's span index"):
        chart.add((1, 3), "S", "OTHER", 1)

    with pytest.raises(ValueError, match="outside the chart's span index"):
        chart.add((1, 1), "S", "OTHER", 1)


def test_truncation_is_recorded_and_the_chart_renders_its_size() -> None:
    semiring = Counting()
    chart = InsideChart[str, int](SpanIndex(3), semiring)
    chart.add((0, 1), "S", "OTHER", 1)

    assert not chart.truncated

    chart.mark_truncated()

    assert chart.truncated
    assert chart.index.n == 3
    assert chart.semiring is semiring
    assert repr(chart) == "InsideChart(n=3, categories=1)"
