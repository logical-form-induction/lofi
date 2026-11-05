import math
from typing import TYPE_CHECKING

import pytest

from lofi._native import PythonCarrier
from lofi.deduce._semiring import (
    Boolean,
    Counting,
    Expectation,
    Forest,
    KBest,
    LogInside,
    Viterbi,
)
from lofi.deduce._spans import Cell, SpanIndex

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.deduce._ports import Semiring

type Branch = tuple[Cell, Cell, Cell]
type NativeSemiring = Boolean | Counting | Expectation | KBest | LogInside | Viterbi
type ExpectedWeight = tuple[float, tuple[float, ...]]

TOKENS = 4
GOAL: Cell = (0, TOKENS)
BRACKETINGS = 5


def _score(parent: Cell, left: Cell, right: Cell) -> float:
    # score by negative left-span width.
    del parent, right

    return -float(left[1] - left[0])


def _inside[W](
    semiring: Semiring[W],
    index: SpanIndex,
    axiom: Callable[[Cell], W],
    edge: Callable[[Cell, Cell, Cell], W],
) -> dict[Cell, W]:
    inside: dict[Cell, W] = {(i, i + 1): axiom((i, i + 1)) for i in range(index.n)}

    for parent, left, right in index.binary_sweep():
        combined = semiring.times(
            semiring.times(inside[left], inside[right]),
            edge(parent, left, right),
        )
        inside[parent] = semiring.plus(inside.get(parent, semiring.zero), combined)

    return inside


def test_log_inside_is_named_and_carries_its_identities() -> None:
    semiring = LogInside()

    assert semiring.name == "log-inside"
    assert semiring.zero == -math.inf
    assert semiring.one == 0.0


@pytest.mark.parametrize(
    ("semiring", "log_probability"),
    [
        (LogInside(), True),
        (Viterbi(), False),
        (Boolean(), False),
        (Counting(), False),
        (KBest(2), False),
        (Expectation(1), False),
        (Forest[str](), False),
    ],
)
def test_each_algebra_declares_whether_its_weights_are_log_probabilities(
    semiring: Semiring[object], *, log_probability: bool
) -> None:
    assert semiring.log_probability is log_probability


def test_plus_is_a_log_sum_exp() -> None:
    semiring = LogInside()

    assert math.isclose(semiring.plus(math.log(0.5), math.log(0.5)), 0.0, abs_tol=1e-12)
    assert math.isclose(
        semiring.plus(math.log(0.25), math.log(0.75)), 0.0, abs_tol=1e-12
    )
    assert math.isclose(
        semiring.plus(math.log(0.1), math.log(0.4)), math.log(0.5), abs_tol=1e-12
    )


def test_plus_absorbs_zero_exactly() -> None:
    semiring = LogInside()

    assert semiring.plus(-3.25, semiring.zero) == -3.25
    assert semiring.plus(semiring.zero, -3.25) == -3.25
    assert semiring.plus(semiring.zero, semiring.zero) == semiring.zero


def test_plus_does_not_overflow_on_a_wide_gap() -> None:
    semiring = LogInside()

    assert semiring.plus(-1e6, 0.0) == 0.0
    assert semiring.plus(0.0, -1e6) == 0.0
    assert semiring.plus(1000.0, 1000.0) == 1000.0 + math.log(2.0)


@pytest.mark.parametrize(
    ("left", "right", "expected"),
    [
        (-math.inf, -3.25, "-0x1.a000000000000p+1"),
        (-3.25, -math.inf, "-0x1.a000000000000p+1"),
        (-2.0, -5.0, "-0x1.f38fc781fdd48p+0"),
        (0.0, -1e6, "0x0.0p+0"),
        (1000.0, 1000.0, "0x1.f458b90bfbe8ep+9"),
    ],
)
def test_log_inside_matches_the_recorded_python_bits(
    left: float, right: float, expected: str
) -> None:
    assert float(LogInside().plus(left, right)).hex() == expected


def test_times_adds_in_log_space() -> None:
    semiring = LogInside()

    assert semiring.times(math.log(0.5), math.log(0.5)) == math.log(0.5) * 2
    assert semiring.times(-2.0, semiring.one) == -2.0
    assert semiring.times(semiring.zero, -2.0) == semiring.zero


def test_certainty_is_a_weight_and_not_a_missing_one() -> None:
    semiring = LogInside()

    assert semiring.is_zero(semiring.zero) is True
    assert semiring.is_zero(semiring.one) is False
    assert semiring.is_zero(0.0) is False
    assert not bool(semiring.one)


def test_boolean_disjoins_alternatives_and_conjoins_antecedents() -> None:
    semiring = Boolean()
    yes = semiring.one
    no = semiring.zero

    assert semiring.name == "boolean"
    assert no is False
    assert yes is True
    assert semiring.plus(no, yes) is True
    assert semiring.plus(no, no) is False
    assert semiring.times(yes, yes) is True
    assert semiring.times(yes, no) is False
    assert semiring.is_zero(no) is True
    assert semiring.is_zero(yes) is False


def test_viterbi_maximizes_where_log_inside_sums() -> None:
    viterbi = Viterbi()

    assert viterbi.name == "viterbi"
    assert viterbi.zero == -math.inf
    assert viterbi.one == 0.0
    assert viterbi.plus(-2.0, -5.0) == -2.0
    assert viterbi.plus(viterbi.zero, -5.0) == -5.0
    assert viterbi.times(-2.0, -5.0) == -7.0
    assert viterbi.times(viterbi.zero, -5.0) == -math.inf
    assert viterbi.is_zero(-math.inf) is True
    assert viterbi.is_zero(0.0) is False

    half = math.log(0.5)

    assert viterbi.plus(half, half) == half
    assert math.isclose(LogInside().plus(half, half), 0.0, abs_tol=1e-12)


def test_counting_sums_and_multiplies_exactly() -> None:
    semiring = Counting()

    assert semiring.name == "counting"
    assert semiring.zero == 0
    assert semiring.one == 1
    assert semiring.plus(3, 4) == 7
    assert semiring.times(3, 4) == 12
    assert semiring.times(semiring.zero, 4) == 0
    assert semiring.is_zero(0) is True
    assert semiring.is_zero(1) is False


def test_counting_promotes_past_the_native_word_exactly() -> None:
    semiring = Counting()

    assert semiring.plus(2**64 - 1, 1) == 2**64
    assert semiring.times(2**63, 4) == 2**65


def test_k_best_keeps_the_best_few_and_no_more() -> None:
    semiring = KBest(2)

    assert semiring.name == "k-best"
    assert semiring.k == 2
    assert semiring.zero == ()
    assert semiring.one == (0.0,)
    assert semiring.plus((-1.0, -4.0), (-2.0, -3.0)) == (-1.0, -2.0)
    assert semiring.times((-1.0, -4.0), (-2.0, -3.0)) == (-3.0, -4.0)
    assert semiring.times(semiring.zero, (-2.0,)) == ()
    assert semiring.is_zero(()) is True
    assert semiring.is_zero((0.0,)) is False


def test_k_best_keeps_two_derivations_that_score_alike() -> None:
    assert KBest(3).plus((-1.0,), (-1.0,)) == (-1.0, -1.0)


def test_a_k_best_width_is_at_least_one() -> None:
    with pytest.raises(ValueError, match="at least one"):
        KBest(0)


def test_forest_keeps_every_derivation() -> None:
    semiring: Forest[str] = Forest()
    left = frozenset({("a",), ("b",)})
    right = frozenset({("c",)})

    assert semiring.name == "forest"
    assert semiring.zero == frozenset()
    assert semiring.one == frozenset({()})
    assert semiring.plus(left, right) == frozenset({("a",), ("b",), ("c",)})
    assert semiring.times(left, right) == frozenset({("a", "c"), ("b", "c")})
    assert semiring.times(left, semiring.one) == left
    assert semiring.times(semiring.zero, left) == frozenset()
    assert semiring.is_zero(frozenset()) is True
    assert semiring.is_zero(semiring.one) is False


def test_expectation_carries_a_vector_under_a_probability() -> None:
    semiring = Expectation(2)

    assert semiring.name == "expectation"
    assert semiring.dim == 2
    assert semiring.zero == (0.0, (0.0, 0.0))
    assert semiring.one == (1.0, (0.0, 0.0))
    assert semiring.plus((0.25, (1.0, 0.0)), (0.5, (0.0, 2.0))) == (0.75, (1.0, 2.0))
    assert semiring.times((0.5, (1.0, 0.0)), (0.4, (0.0, 2.0))) == (0.2, (0.4, 1.0))
    assert semiring.is_zero(semiring.zero) is True
    assert semiring.is_zero(semiring.one) is False

    assert semiring.is_zero((0.0, (1.0, 0.0))) is False


def test_an_expectation_is_the_carried_mass_over_the_probability() -> None:
    semiring = Expectation(1)

    combined = semiring.plus((0.5, (1.0,)), (0.25, (1.0,)))

    assert combined == (0.75, (2.0,))
    assert math.isclose(combined[1][0] / combined[0], 8 / 3)


def _expectation_bits(weight: ExpectedWeight) -> tuple[str, ...]:
    """Return the exact binary64 representation of an expectation weight."""
    probability, coordinates = weight

    return tuple(float(value).hex() for value in (probability, *coordinates))


@pytest.mark.parametrize(
    ("left", "right", "plus_bits", "times_bits"),
    [
        (
            (0.1, (0.2, -0.3)),
            (0.4, (-0.5, 0.6)),
            (
                "0x1.0000000000000p-1",
                "-0x1.3333333333333p-2",
                "0x1.3333333333333p-2",
            ),
            (
                "0x1.47ae147ae147cp-5",
                "0x1.eb851eb851ebcp-6",
                "-0x1.eb851eb851eb8p-5",
            ),
        ),
        (
            (1e100, (1e-100, -3.25)),
            (1e-100, (1e100, 2.5)),
            (
                "0x1.249ad2594c37dp+332",
                "0x1.249ad2594c37dp+332",
                "-0x1.8000000000000p-1",
            ),
            (
                "0x1.0000000000000p+0",
                "0x1.4e718d7d7625ap+664",
                "0x1.6dc186ef9f45cp+333",
            ),
        ),
        (
            (2**-52, (float(2**53), 0.1)),
            (1.0, (2**-53, -0.2)),
            (
                "0x1.0000000000001p+0",
                "0x1.0000000000000p+53",
                "-0x1.999999999999ap-4",
            ),
            (
                "0x1.0000000000000p-52",
                "0x1.0000000000000p+53",
                "0x1.9999999999997p-4",
            ),
        ),
    ],
)
def test_expectation_matches_the_recorded_python_bits(
    left: ExpectedWeight,
    right: ExpectedWeight,
    plus_bits: tuple[str, ...],
    times_bits: tuple[str, ...],
) -> None:
    semiring = Expectation(2)

    assert _expectation_bits(semiring.plus(left, right)) == plus_bits
    assert _expectation_bits(semiring.times(left, right)) == times_bits


def test_an_expectation_carries_at_least_one_coordinate() -> None:
    with pytest.raises(ValueError, match="at least one coordinate"):
        Expectation(0)


def test_a_parameterized_algebra_shows_its_parameter() -> None:
    assert repr(KBest(4)) == "KBest(4)"
    assert repr(Expectation(3)) == "Expectation(3)"


@pytest.mark.parametrize(
    "semiring",
    [Boolean(), Counting(), Expectation(2), KBest(3), LogInside(), Viterbi()],
)
def test_a_native_semiring_declares_itself_as_its_lowering(
    semiring: NativeSemiring,
) -> None:
    assert semiring.lowering == semiring


def test_the_forest_declares_python_carried_weights() -> None:
    assert Forest[str]().lowering == PythonCarrier()


@pytest.mark.parametrize(
    "semiring",
    [Boolean(), Counting(), Expectation(2), KBest(3), LogInside(), Viterbi()],
)
def test_a_native_semiring_is_frozen(semiring: NativeSemiring) -> None:
    attribute = "name"
    with pytest.raises(AttributeError):
        setattr(semiring, attribute, "changed")


def test_boolean_agrees_with_a_positive_derivation_count() -> None:
    index = SpanIndex(TOKENS)
    derivable = _inside(
        Boolean(), index, lambda _cell: True, lambda _parent, _left, _right: True
    )
    counted = _inside(
        Counting(), index, lambda _cell: 1, lambda _parent, _left, _right: 1
    )

    assert counted[GOAL] == BRACKETINGS
    assert all(
        derivable[cell] is (counted[cell] > 0) for cell in index.spans_by_length()
    )


def test_viterbi_agrees_with_the_best_derivation_in_the_forest() -> None:
    index = SpanIndex(TOKENS)
    best = _inside(Viterbi(), index, lambda _cell: 0.0, _score)
    forest = _inside(
        Forest[Branch](),
        index,
        lambda _cell: frozenset({()}),
        lambda parent, left, right: frozenset({((parent, left, right),)}),
    )
    ranked = _inside(
        KBest(3),
        index,
        lambda _cell: (0.0,),
        lambda parent, left, right: (_score(parent, left, right),),
    )
    scored = sorted(
        (sum(_score(*branch) for branch in derivation) for derivation in forest[GOAL]),
        reverse=True,
    )

    assert len(forest[GOAL]) == BRACKETINGS
    assert best[GOAL] == scored[0]
    assert ranked[GOAL] == tuple(scored[:3])


def test_an_absent_axiom_leaves_the_goal_underivable_in_every_algebra() -> None:
    index = SpanIndex(TOKENS)
    gap = 2

    derivable = _inside(
        Boolean(),
        index,
        lambda cell: cell[0] != gap,
        lambda _parent, _left, _right: True,
    )
    counted = _inside(
        Counting(),
        index,
        lambda cell: 0 if cell[0] == gap else 1,
        lambda _parent, _left, _right: 1,
    )
    best = _inside(
        Viterbi(), index, lambda cell: -math.inf if cell[0] == gap else 0.0, _score
    )
    forest = _inside(
        Forest[Branch](),
        index,
        lambda cell: frozenset() if cell[0] == gap else frozenset({()}),
        lambda parent, left, right: frozenset({((parent, left, right),)}),
    )

    assert derivable[GOAL] is False
    assert counted[GOAL] == 0
    assert best[GOAL] == -math.inf
    assert forest[GOAL] == frozenset()

    assert derivable[0, 2] is True
    assert counted[3, 4] == 1


def test_the_expectation_of_a_constant_over_the_forest_is_that_constant() -> None:
    index = SpanIndex(TOKENS)
    total = _inside(LogInside(), index, lambda _cell: 0.0, _score)
    expectations = _inside(
        Expectation(1),
        index,
        lambda cell: (1.0, (float(cell[0]),)),
        lambda parent, left, right: (math.exp(_score(parent, left, right)), (0.0,)),
    )
    probability = expectations[GOAL][0]
    accumulated = expectations[GOAL][1][0]

    assert math.isclose(probability, math.exp(total[GOAL]))

    assert math.isclose(accumulated / probability, sum(range(TOKENS)))
