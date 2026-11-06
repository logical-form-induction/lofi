from itertools import permutations
from types import MappingProxyType
from typing import TYPE_CHECKING, override

import pytest
from hypothesis import strategies as st

from lofi._checks import LawViolationError
from lofi.stats import (
    Decisions,
    Dependence,
    DependenceError,
    NullFamily,
    PValue,
    Statistic,
    Tail,
    p_value_against_replicates,
)
from lofi.stats.laws import (
    CORRECTION_LAWS,
    NULL_MODEL_LAWS,
    CorrectionFixture,
    NullModelFixture,
    a_test_leaves_the_observed_data_alone,
    adjusted_p_values_are_monotone_in_the_raw_ones,
    an_uncovered_dependence_regime_is_refused,
    check_correction_laws,
    check_null_model_laws,
    no_p_value_is_lowered_by_correcting_it,
    one_observation_tests_to_one_p_value,
    the_provenance_names_the_null_that_ran,
    the_raw_p_values_come_back_unchanged,
)

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

EXAMPLES = 20
SPACE = 20

NOTHING_MATCHED: Mapping[str, str | int] = MappingProxyType({})


def difference(values: Sequence[float]) -> float:
    half = len(values) // 2

    return sum(values[:half]) / half - sum(values[half:]) / (len(values) - half)


class _PermutationNull:
    def __init__(self, name: str = "permutation") -> None:
        self._name = name

    @property
    def name(self) -> str:
        return self._name

    @property
    def family(self) -> NullFamily:
        return "permutation"

    @property
    def replicates(self) -> int | None:
        return None

    @property
    def seed(self) -> int | None:
        return None

    @property
    def matching(self) -> Mapping[str, str | int]:
        return NOTHING_MATCHED

    def rearranged(self, observed: tuple[float, ...]) -> list[tuple[float, ...]]:
        return list(permutations(observed))

    def test(
        self,
        statistic: Statistic[tuple[float, ...]],
        observed: tuple[float, ...],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        return p_value_against_replicates(
            statistic(observed),
            [statistic(arrangement) for arrangement in self.rearranged(observed)],
            tail=tail,
            null=self._name,
            family=self.family,
            exact=True,
            seed=None,
        )


class _AccumulatingNull(_PermutationNull):
    def __init__(self) -> None:
        super().__init__("accumulating")
        self._seen: list[tuple[float, ...]] = []

    @override
    def rearranged(self, observed: tuple[float, ...]) -> list[tuple[float, ...]]:
        self._seen.append(observed)

        return [*permutations(observed), *self._seen]


class _MisreportingNull(_PermutationNull):
    @override
    def test(
        self,
        statistic: Statistic[tuple[float, ...]],
        observed: tuple[float, ...],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        return p_value_against_replicates(
            statistic(observed),
            [statistic(arrangement) for arrangement in permutations(observed)],
            tail=tail,
            null="something else",
            family=self.family,
            exact=True,
            seed=None,
        )


class _SortingNull:
    @property
    def name(self) -> str:
        return "sorting"

    @property
    def family(self) -> NullFamily:
        return "permutation"

    @property
    def replicates(self) -> int | None:
        return None

    @property
    def seed(self) -> int | None:
        return None

    @property
    def matching(self) -> Mapping[str, str | int]:
        return NOTHING_MATCHED

    def test(
        self,
        statistic: Statistic[list[float]],
        observed: list[float],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        observed.sort()

        return p_value_against_replicates(
            statistic(observed),
            [statistic(list(arrangement)) for arrangement in permutations(observed)],
            tail=tail,
            null=self.name,
            family=self.family,
            exact=True,
            seed=None,
        )


class _Inflating:
    def __init__(self, factor: float, valid_under: frozenset[Dependence]) -> None:
        self._factor = factor
        self._valid_under = valid_under

    @property
    def name(self) -> str:
        return f"inflating by {self._factor}"

    @property
    def valid_under(self) -> frozenset[Dependence]:
        return self._valid_under

    def adjust(self, pvalues: Sequence[PValue]) -> tuple[float, ...]:
        factor = self._factor if len(pvalues) > 1 else 1.0

        return tuple(min(1.0, factor * pvalue.value) for pvalue in pvalues)

    def apply(
        self, pvalues: Sequence[PValue], alpha: float, dependence: Dependence, /
    ) -> Decisions:
        if dependence not in self._valid_under:
            raise DependenceError(
                correction=self.name,
                dependence=dependence,
                valid_under=self._valid_under,
            )

        return Decisions(self.name, alpha, dependence, pvalues, self.adjust(pvalues))


class _Sorting(_Inflating):
    @override
    def apply(
        self, pvalues: Sequence[PValue], alpha: float, dependence: Dependence, /
    ) -> Decisions:
        ordered = tuple(sorted(pvalues, key=lambda pvalue: pvalue.value))

        return Decisions(self.name, alpha, dependence, ordered, self.adjust(ordered))


class _Reversing(_Inflating):
    @override
    def apply(
        self, pvalues: Sequence[PValue], alpha: float, dependence: Dependence, /
    ) -> Decisions:
        adjusted = tuple(reversed(self.adjust(pvalues)))

        return Decisions(self.name, alpha, dependence, pvalues, adjusted)


class _Permissive(_Inflating):
    @override
    def apply(
        self, pvalues: Sequence[PValue], alpha: float, dependence: Dependence, /
    ) -> Decisions:
        return Decisions(self.name, alpha, dependence, pvalues, self.adjust(pvalues))


def a_p_value(rank: int) -> PValue:
    return p_value_against_replicates(
        float(rank),
        [float(step) for step in range(SPACE)],
        tail="greater",
        null="ranks",
        family="permutation",
        exact=True,
        seed=None,
    )


VECTORS = st.lists(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    min_size=4,
    max_size=4,
).map(tuple)
MUTABLE_VECTORS = st.lists(
    st.floats(min_value=-50.0, max_value=50.0, allow_nan=False, allow_infinity=False),
    min_size=4,
    max_size=4,
)
FAMILIES = st.lists(
    st.integers(min_value=0, max_value=SPACE - 1).map(a_p_value),
    min_size=1,
    max_size=5,
).map(tuple)
LEVELS = st.floats(min_value=0.01, max_value=0.5)

PERMUTATION = NullModelFixture(_PermutationNull(), difference, VECTORS, EXAMPLES)
INFLATING = CorrectionFixture(
    _Inflating(3.0, frozenset({Dependence.INDEPENDENT, Dependence.PRDS})),
    Dependence.PRDS,
    FAMILIES,
    LEVELS,
    EXAMPLES,
)


@pytest.mark.parametrize("law", NULL_MODEL_LAWS, ids=lambda law: law.__name__)
def test_each_null_model_law_holds_of_the_permutation_null(
    law: Callable[[NullModelFixture[tuple[float, ...]]], None],
) -> None:
    law(PERMUTATION)


@pytest.mark.parametrize("law", CORRECTION_LAWS, ids=lambda law: law.__name__)
def test_each_correction_law_holds_of_the_inflating_correction(
    law: Callable[[CorrectionFixture], None],
) -> None:
    law(INFLATING)


def test_the_null_suite_runs_every_law() -> None:
    check_null_model_laws(PERMUTATION)


def test_the_correction_suite_runs_every_law() -> None:
    check_correction_laws(INFLATING)


def test_the_suite_catches_a_null_whose_reference_set_grows_between_calls() -> None:
    with pytest.raises(LawViolationError, match="determinism"):
        one_observation_tests_to_one_p_value(
            NullModelFixture(_AccumulatingNull(), difference, VECTORS, EXAMPLES)
        )


def test_the_suite_catches_a_null_that_reports_someone_elses_provenance() -> None:
    with pytest.raises(LawViolationError, match="provenance"):
        the_provenance_names_the_null_that_ran(
            NullModelFixture(_MisreportingNull(), difference, VECTORS, EXAMPLES)
        )


def test_the_suite_catches_a_null_that_sorts_the_data_it_was_handed() -> None:
    with pytest.raises(LawViolationError, match="non-mutation"):
        a_test_leaves_the_observed_data_alone(
            NullModelFixture(_SortingNull(), difference, MUTABLE_VECTORS, EXAMPLES)
        )


def _correction_fixture(correction: _Inflating) -> CorrectionFixture:
    return CorrectionFixture(correction, Dependence.PRDS, FAMILIES, LEVELS, EXAMPLES)


def test_the_suite_catches_a_correction_that_makes_a_result_stronger() -> None:
    with pytest.raises(LawViolationError, match="no p-value is lowered"):
        no_p_value_is_lowered_by_correcting_it(
            _correction_fixture(_Inflating(0.5, frozenset(Dependence)))
        )


def test_the_suite_catches_a_correction_that_returns_the_family_reordered() -> None:
    with pytest.raises(LawViolationError, match="the raw p-values"):
        the_raw_p_values_come_back_unchanged(
            _correction_fixture(_Sorting(3.0, frozenset(Dependence)))
        )


def test_the_suite_catches_a_correction_that_pairs_the_wrong_values() -> None:
    distinct = (a_p_value(SPACE - 1), a_p_value(SPACE - 2))
    fixture = CorrectionFixture(
        _Reversing(3.0, frozenset(Dependence)),
        Dependence.PRDS,
        st.just(distinct),
        LEVELS,
        EXAMPLES,
    )

    with pytest.raises(LawViolationError, match="monotonicity"):
        adjusted_p_values_are_monotone_in_the_raw_ones(fixture)


def test_the_suite_catches_a_correction_that_answers_outside_its_guarantee() -> None:
    with pytest.raises(LawViolationError, match="an uncovered regime"):
        an_uncovered_dependence_regime_is_refused(
            _correction_fixture(_Permissive(3.0, frozenset({Dependence.INDEPENDENT})))
        )
