"""Tests for the token-count distributions."""

from collections import Counter
from math import inf, log, nan, sqrt
from random import Random
from typing import TYPE_CHECKING, cast

import pytest

from lofi.data.synthetic import NegativeBinomialLength, TabularLength, UniformLength

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.data.synthetic import LengthDistribution, OutOfSupportPolicy


def _draws(distribution: LengthDistribution, count: int, seed: int) -> list[int]:
    random = Random(seed)

    return [distribution.draw(random) for _ in range(count)]


def _mean(values: list[int]) -> float:
    return sum(values) / len(values)


def _variance(values: list[int]) -> float:
    mean = _mean(values)

    return sum((value - mean) ** 2 for value in values) / len(values)


def test_a_uniform_length_holds_its_bounds_and_moments() -> None:
    distribution = UniformLength(2, 7)

    assert distribution.min_length == 2
    assert distribution.max_length == 7
    assert distribution.mean == 4.5
    assert distribution.variance == pytest.approx(35 / 12)
    assert UniformLength(3, 3).variance == 0.0
    assert UniformLength(3, 3).mean == 3.0
    assert repr(distribution) == "UniformLength(2, 7)"


def test_a_uniform_length_draws_every_value_in_its_bounds_equally() -> None:
    drawn = _draws(UniformLength(2, 7), 6000, seed=1)
    counts = Counter(drawn)
    expected = len(drawn) / 6

    assert set(counts) == {2, 3, 4, 5, 6, 7}

    for count in counts.values():
        assert abs(count - expected) <= 3 * sqrt(expected * 5 / 6)

    assert _draws(UniformLength(4, 4), 20, seed=2) == [4] * 20


def test_a_uniform_length_is_equal_on_its_bounds() -> None:
    assert UniformLength(2, 7) == UniformLength(2, 7)
    assert hash(UniformLength(2, 7)) == hash(UniformLength(2, 7))
    assert UniformLength(2, 7) != UniformLength(3, 7)
    assert UniformLength(2, 7) != UniformLength(2, 8)
    assert UniformLength(2, 7).__eq__((2, 7)) is NotImplemented


def test_a_negative_binomial_length_holds_its_shape_bounds_and_moments() -> None:
    distribution = NegativeBinomialLength(3, 0.5, min_length=2, max_length=16)

    assert distribution.r == 3.0
    assert distribution.p == 0.5
    assert distribution.min_length == 2
    assert distribution.max_length == 16
    assert distribution.mean == 5.0
    assert distribution.variance == 6.0
    assert NegativeBinomialLength(2.5, 0.4, max_length=9).min_length == 1
    assert NegativeBinomialLength(2.5, 0.4, max_length=9).mean == pytest.approx(4.75)
    assert NegativeBinomialLength(2.5, 0.4, max_length=9).variance == pytest.approx(
        9.375
    )
    assert repr(distribution) == (
        "NegativeBinomialLength(3.0, 0.5, min_length=2, max_length=16, "
        "out_of_support='resample')"
    )


@pytest.mark.parametrize(
    ("r", "p", "min_length"),
    [(3, 0.5, 2), (2.5, 0.4, 1), (1, 0.2, 4), (0.5, 0.6, 1)],
)
def test_a_negative_binomial_length_draws_with_its_mean_and_variance(
    r: float, p: float, min_length: int
) -> None:
    distribution = NegativeBinomialLength(r, p, min_length=min_length, max_length=99)
    drawn = _draws(distribution, 20000, seed=3)
    deviation = sqrt(distribution.variance / len(drawn))

    assert min(drawn) >= min_length
    assert abs(_mean(drawn) - distribution.mean) <= 4 * deviation
    assert abs(_variance(drawn) - distribution.variance) <= 0.1 * distribution.variance


def test_a_whole_r_draws_as_the_sum_of_r_geometric_failure_counts() -> None:
    distribution = NegativeBinomialLength(4, 0.3, max_length=99)
    drawn = _draws(distribution, 20000, seed=4)
    random = Random(5)
    geometric = [
        1 + sum(int(log(random.random()) / log(0.7)) for _ in range(4))
        for _ in range(20000)
    ]

    assert abs(_mean(drawn) - _mean(geometric)) <= 4 * sqrt(
        2 * distribution.variance / 20000
    )


def test_a_negative_binomial_length_draws_beyond_its_upper_bound() -> None:
    distribution = NegativeBinomialLength(3, 0.5, min_length=2, max_length=4)
    drawn = _draws(distribution, 400, seed=6)

    beyond = 1 - 0.5**3 * (1 + 3 * 0.5 + 6 * 0.5**2)

    assert max(drawn) > 4
    assert abs(sum(value > 4 for value in drawn) / len(drawn) - beyond) <= 0.1


def test_a_large_rate_is_drawn_in_pieces_without_underflow() -> None:
    distribution = NegativeBinomialLength(2000, 0.5, max_length=99)
    drawn = _draws(distribution, 200, seed=7)

    assert all(value > 1000 for value in drawn)
    assert abs(_mean(drawn) - distribution.mean) <= 4 * sqrt(
        distribution.variance / len(drawn)
    )


def test_the_same_seed_draws_the_same_counts() -> None:
    distribution = NegativeBinomialLength(3, 0.5, min_length=2, max_length=16)

    assert _draws(distribution, 50, seed=8) == _draws(distribution, 50, seed=8)
    assert _draws(distribution, 50, seed=8) != _draws(distribution, 50, seed=9)
    assert _draws(UniformLength(1, 9), 50, seed=8) == _draws(
        UniformLength(1, 9), 50, seed=8
    )


def test_a_negative_binomial_length_is_equal_on_shape_and_bounds() -> None:
    base = NegativeBinomialLength(3, 0.5, min_length=2, max_length=16)

    assert base == NegativeBinomialLength(3.0, 0.5, min_length=2, max_length=16)
    assert hash(base) == hash(
        NegativeBinomialLength(3.0, 0.5, min_length=2, max_length=16)
    )
    assert base != NegativeBinomialLength(4, 0.5, min_length=2, max_length=16)
    assert base != NegativeBinomialLength(3, 0.6, min_length=2, max_length=16)
    assert base != NegativeBinomialLength(3, 0.5, min_length=3, max_length=16)
    assert base != NegativeBinomialLength(3, 0.5, min_length=2, max_length=17)
    assert base != NegativeBinomialLength(
        3, 0.5, min_length=2, max_length=16, out_of_support="clip"
    )
    assert base != UniformLength(2, 16)
    assert base.__eq__((3, 0.5)) is NotImplemented


def test_a_tabular_length_normalizes_and_holds_its_moments() -> None:
    distribution = TabularLength({4: 1.0, 2: 2.0, 7: 1.0})

    assert distribution.min_length == 2
    assert distribution.max_length == 7
    assert distribution.mean == pytest.approx(3.75)
    assert distribution.variance == pytest.approx(4.1875)
    assert distribution.masses == ((2, 0.5), (4, 0.25), (7, 0.25))
    assert distribution.out_of_support == "error"
    assert repr(distribution) == "TabularLength({2: 0.5, 4: 0.25, 7: 0.25})"


def test_a_tabular_length_draws_in_proportion_to_declared_mass() -> None:
    distribution = TabularLength({2: 1.0, 4: 3.0, 7: 6.0})
    drawn = _draws(distribution, 10000, seed=10)
    counts = Counter(drawn)

    assert set(counts) == {2, 4, 7}
    assert counts[2] / len(drawn) == pytest.approx(0.1, abs=0.02)
    assert counts[4] / len(drawn) == pytest.approx(0.3, abs=0.03)
    assert counts[7] / len(drawn) == pytest.approx(0.6, abs=0.03)
    assert drawn == _draws(distribution, 10000, seed=10)


def test_a_tabular_length_is_equal_on_normalized_mass() -> None:
    base = TabularLength({2: 1.0, 4: 3.0})

    assert base == TabularLength({4: 6.0, 2: 2.0})
    assert hash(base) == hash(TabularLength({2: 2.0, 4: 6.0}))
    assert base != TabularLength({2: 1.0, 4: 2.0})
    assert base != UniformLength(2, 4)


@pytest.mark.parametrize(
    ("build", "error", "fault"),
    [
        (lambda: UniformLength(0, 3), ValueError, "min_length is not positive"),
        (lambda: UniformLength(4, 3), ValueError, "min_length 4 exceeds max_length 3"),
        (lambda: UniformLength(1, -1), ValueError, "max_length is negative"),
        (lambda: UniformLength(-1, 3), ValueError, "min_length is negative"),
        (
            lambda: UniformLength(cast("int", 1.5), 3),
            TypeError,
            "min_length is not an integer",
        ),
        (
            lambda: UniformLength(1, cast("int", 2.5)),
            TypeError,
            "max_length is not an integer",
        ),
        (
            lambda: NegativeBinomialLength(0, 0.5, max_length=9),
            ValueError,
            "r is not positive",
        ),
        (
            lambda: NegativeBinomialLength(-1, 0.5, max_length=9),
            ValueError,
            "r is not positive",
        ),
        (
            lambda: NegativeBinomialLength(inf, 0.5, max_length=9),
            ValueError,
            "r is not finite",
        ),
        (
            lambda: NegativeBinomialLength(nan, 0.5, max_length=9),
            ValueError,
            "r is not finite",
        ),
        (
            lambda: NegativeBinomialLength(cast("float", "3"), 0.5, max_length=9),
            TypeError,
            "r is not a number",
        ),
        (
            lambda: NegativeBinomialLength(3, 0, max_length=9),
            ValueError,
            "p is not strictly between zero and one",
        ),
        (
            lambda: NegativeBinomialLength(3, 1, max_length=9),
            ValueError,
            "p is not strictly between zero and one",
        ),
        (
            lambda: NegativeBinomialLength(3, cast("float", "0.5"), max_length=9),
            TypeError,
            "p is not a number",
        ),
        (
            lambda: NegativeBinomialLength(3, 0.5, min_length=0, max_length=9),
            ValueError,
            "min_length is not positive",
        ),
        (
            lambda: NegativeBinomialLength(3, 0.5, min_length=10, max_length=9),
            ValueError,
            "min_length 10 exceeds max_length 9",
        ),
        (
            lambda: NegativeBinomialLength(
                3,
                0.5,
                max_length=9,
                out_of_support=cast("OutOfSupportPolicy", "truncate"),
            ),
            ValueError,
            "unknown out-of-support policy",
        ),
        (lambda: TabularLength({}), ValueError, "no positive mass"),
        (lambda: TabularLength({0: 1.0}), ValueError, "not positive"),
        (lambda: TabularLength({2: -1.0}), ValueError, "is negative"),
        (lambda: TabularLength({2: inf}), ValueError, "not finite"),
        (lambda: TabularLength({2: nan}), ValueError, "not finite"),
        (lambda: TabularLength({2: 0.0}), ValueError, "no positive mass"),
    ],
)
def test_a_distribution_refuses_bad_shape_parameters_and_bounds(
    build: Callable[[], LengthDistribution], error: type[Exception], fault: str
) -> None:
    with pytest.raises(error, match=fault):
        build()
