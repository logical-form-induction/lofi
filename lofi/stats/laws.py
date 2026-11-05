"""Statistical conformance laws."""

from math import isclose
from typing import TYPE_CHECKING, Final, NamedTuple

from hypothesis import given, settings
from hypothesis import strategies as st

from lofi._checks import fail_law
from lofi.stats._errors import DependenceError
from lofi.stats._values import TAILS, Dependence, PValue, Tail

if TYPE_CHECKING:
    from hypothesis.strategies import SearchStrategy

    from lofi.stats._ports import Correction, NullModel, Statistic

__all__ = [
    "CORRECTION_LAWS",
    "NULL_MODEL_LAWS",
    "CorrectionFixture",
    "NullModelFixture",
    "a_family_of_one_is_left_alone",
    "a_smaller_level_rejects_no_more",
    "a_test_leaves_the_observed_data_alone",
    "adjusted_p_values_are_monotone_in_the_raw_ones",
    "an_uncovered_dependence_regime_is_refused",
    "check_correction_laws",
    "check_null_model_laws",
    "no_p_value_is_lowered_by_correcting_it",
    "one_observation_tests_to_one_p_value",
    "the_provenance_names_the_null_that_ran",
    "the_raw_p_values_come_back_unchanged",
]

_TESTS: Final = 5
"""The repetition count for the determinism law."""

_SLACK: Final = 1e-12
"""Rounding tolerance for multiplication by a family size."""


class NullModelFixture[Data](NamedTuple):
    """Stores inputs for a null-model law.

    Parameters
    ----------
    model : NullModel[Data]
        The null under test.
    statistic : Statistic[Data]
        The summary the laws test with.
    data : SearchStrategy[Data]
        A strategy over the observed data the null rearranges.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    model: NullModel[Data]
    statistic: Statistic[Data]
    data: SearchStrategy[Data]
    max_examples: int = 50


class CorrectionFixture(NamedTuple):
    """Stores inputs for a correction law.

    Parameters
    ----------
    correction : Correction
        The procedure under test.
    dependence : Dependence
        A regime the procedure is valid under.
    families : SearchStrategy[tuple[PValue, ...]]
        A strategy over the p-value families it corrects.
    levels : SearchStrategy[float]
        A strategy over the levels to correct at.
    max_examples : int
        The example budget each law gives Hypothesis.
    """

    correction: Correction
    dependence: Dependence
    families: SearchStrategy[tuple[PValue, ...]]
    levels: SearchStrategy[float]
    max_examples: int = 50


def _null_budget[Data](fixture: NullModelFixture[Data]) -> settings:
    """Return the Hypothesis settings every null model law runs under.

    Parameters
    ----------
    fixture : NullModelFixture[Data]
        The null under test and its strategies.
    """
    # allow slow reference-distribution draws
    return settings(max_examples=fixture.max_examples, deadline=None)


def _correction_budget(fixture: CorrectionFixture) -> settings:
    """Return the Hypothesis settings every correction law runs under.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.
    """
    return settings(max_examples=fixture.max_examples, deadline=None)


def _not_above(smaller: float, larger: float) -> bool:
    """Return whether one number is at or below another, up to rounding.

    Parameters
    ----------
    smaller : float
        The number expected not to exceed the other.
    larger : float
        The number expected not to fall below the first.
    """
    return smaller <= larger or isclose(smaller, larger, rel_tol=_SLACK, abs_tol=_SLACK)


def one_observation_tests_to_one_p_value[Data](fixture: NullModelFixture[Data]) -> None:
    """Check that one null tests one observation to one p-value, every time.

    Parameters
    ----------
    fixture : NullModelFixture[Data]
        The null under test and its strategies.

    Raises
    ------
    LawViolationError
        When repeated tests of one observation disagree.
    """
    model = fixture.model

    @given(fixture.data)
    @_null_budget(fixture)
    def law(observed: Data) -> None:
        seen = {
            model.test(fixture.statistic, observed, tail="greater")
            for _ in range(_TESTS)
        }

        if len(seen) != 1:
            detail = (
                f"{model.name} tested one observation {_TESTS} times and "
                f"produced {len(seen)} different p-values"
            )

            fail_law(detail, law="determinism")

    law()


def the_provenance_names_the_null_that_ran[Data](
    fixture: NullModelFixture[Data],
) -> None:
    """Check that a p-value records the provenance of the null that made it.

    Parameters
    ----------
    fixture : NullModelFixture[Data]
        The null under test and its strategies.

    Raises
    ------
    LawViolationError
        When the recorded null, family, seed, matching criteria or tail is not the one
        that ran.
    """
    model = fixture.model

    @given(fixture.data, st.sampled_from(TAILS))
    @_null_budget(fixture)
    def law(observed: Data, tail: Tail) -> None:
        pvalue = model.test(fixture.statistic, observed, tail=tail)
        recorded = pvalue.provenance
        declared = (model.name, model.family, model.seed, dict(model.matching))
        got = (
            recorded.null,
            recorded.family,
            recorded.seed,
            dict(recorded.matching),
        )

        if got != declared:
            detail = f"{model.name} declares {declared!r} and recorded {got!r}"

            fail_law(detail, law="provenance")

        if recorded.tail != tail:
            detail = (
                f"{model.name} was asked for the {tail!r} tail and recorded "
                f"the {recorded.tail!r} one"
            )

            fail_law(detail, law="provenance")

    law()


def a_test_leaves_the_observed_data_alone[Data](
    fixture: NullModelFixture[Data],
) -> None:
    """Check that testing leaves the observed data as it was.

    Parameters
    ----------
    fixture : NullModelFixture[Data]
        The null under test and its strategies.

    Raises
    ------
    LawViolationError
        When the observed data changed under a test.
    """
    model = fixture.model

    @given(fixture.data)
    @_null_budget(fixture)
    def law(observed: Data) -> None:
        before = repr(observed)
        model.test(fixture.statistic, observed, tail="greater")

        if repr(observed) != before:
            detail = f"{model.name} changed {before} into {observed!r}"

            fail_law(detail, law="non-mutation")

    law()


def the_raw_p_values_come_back_unchanged(fixture: CorrectionFixture) -> None:
    """Check that a correction returns the family it was given, in order.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        When the returned raw p-values differ from those handed in, in value or in
        order.
    """
    correction = fixture.correction

    @given(fixture.families, fixture.levels)
    @_correction_budget(fixture)
    def law(family: tuple[PValue, ...], alpha: float) -> None:
        decisions = correction.apply(family, alpha, fixture.dependence)

        if decisions.raw != family:
            detail = (
                f"{correction.name} was handed {family!r} and returned "
                f"{decisions.raw!r}"
            )

            fail_law(detail, law="the raw p-values")

    law()


def no_p_value_is_lowered_by_correcting_it(fixture: CorrectionFixture) -> None:
    """Check that correcting for multiplicity never makes a result stronger.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        When an adjusted p-value falls below the raw one it came from.
    """
    correction = fixture.correction

    @given(fixture.families, fixture.levels)
    @_correction_budget(fixture)
    def law(family: tuple[PValue, ...], alpha: float) -> None:
        decisions = correction.apply(family, alpha, fixture.dependence)

        for raw, adjusted in zip(decisions.raw, decisions.adjusted, strict=True):
            if not _not_above(raw.value, adjusted):
                detail = (
                    f"{correction.name} corrected {raw.value!r} to "
                    f"{adjusted!r} in a family of {len(family)}"
                )

                fail_law(detail, law="no p-value is lowered")

    law()


def adjusted_p_values_are_monotone_in_the_raw_ones(fixture: CorrectionFixture) -> None:
    """Check that correcting preserves the order of the family.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        When one raw p-value is below another and its adjusted value is above the
        other's.
    """
    correction = fixture.correction

    @given(fixture.families, fixture.levels)
    @_correction_budget(fixture)
    def law(family: tuple[PValue, ...], alpha: float) -> None:
        decisions = correction.apply(family, alpha, fixture.dependence)
        pairs = tuple(zip(decisions.raw, decisions.adjusted, strict=True))

        for first, second in ((a, b) for a in pairs for b in pairs):
            if first[0].value <= second[0].value and not _not_above(
                first[1], second[1]
            ):
                detail = (
                    f"{correction.name} corrected {first[0].value!r} to "
                    f"{first[1]!r} and {second[0].value!r} to {second[1]!r}"
                )

                fail_law(detail, law="monotonicity")

    law()


def a_family_of_one_is_left_alone(fixture: CorrectionFixture) -> None:
    """Check that a family of one test is not corrected.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        When a single p-value comes back changed.
    """
    correction = fixture.correction

    @given(fixture.families, fixture.levels)
    @_correction_budget(fixture)
    def law(family: tuple[PValue, ...], alpha: float) -> None:
        only = family[:1]

        if not only:
            return

        decisions = correction.apply(only, alpha, fixture.dependence)

        if not isclose(
            decisions.adjusted[0], only[0].value, rel_tol=_SLACK, abs_tol=_SLACK
        ):
            detail = (
                f"{correction.name} corrected the single p-value "
                f"{only[0].value!r} to {decisions.adjusted[0]!r}"
            )

            fail_law(detail, law="a family of one")

    law()


def a_smaller_level_rejects_no_more(fixture: CorrectionFixture) -> None:
    """Check that lowering the level never rejects a hypothesis it did not.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        When the family rejected at the lower level is not contained in the one rejected
        at the higher level.
    """
    correction = fixture.correction

    @given(fixture.families, fixture.levels, fixture.levels)
    @_correction_budget(fixture)
    def law(family: tuple[PValue, ...], one: float, other: float) -> None:
        lower, higher = min(one, other), max(one, other)
        strict = correction.apply(family, lower, fixture.dependence)
        lenient = correction.apply(family, higher, fixture.dependence)
        gained = {
            index
            for index, rejected in enumerate(strict.reject)
            if rejected and not lenient.reject[index]
        }

        if gained:
            detail = (
                f"{correction.name} rejects {sorted(gained)!r} at {lower!r} and "
                f"not at {higher!r}"
            )

            fail_law(detail, law="a smaller level")

    law()


def an_uncovered_dependence_regime_is_refused(fixture: CorrectionFixture) -> None:
    """Check that a procedure refuses a regime its guarantee does not cover.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        When applying the procedure under an uncovered regime returns rather than
        refusing.
    """
    correction = fixture.correction
    uncovered = tuple(
        regime for regime in Dependence if regime not in correction.valid_under
    )

    if not uncovered:
        return

    @given(fixture.families, fixture.levels, st.sampled_from(uncovered))
    @_correction_budget(fixture)
    def law(family: tuple[PValue, ...], alpha: float, regime: Dependence) -> None:
        try:
            decisions = correction.apply(family, alpha, regime)

        except DependenceError:
            return

        detail = (
            f"{correction.name} is valid under "
            f"{sorted(r.name for r in correction.valid_under)!r} and returned "
            f"{decisions.adjusted!r} under {regime.name}"
        )

        fail_law(detail, law="an uncovered regime")

    law()


NULL_MODEL_LAWS: Final = (
    one_observation_tests_to_one_p_value,
    the_provenance_names_the_null_that_ran,
    a_test_leaves_the_observed_data_alone,
)
"""Null-model laws in conformance order."""

CORRECTION_LAWS: Final = (
    the_raw_p_values_come_back_unchanged,
    no_p_value_is_lowered_by_correcting_it,
    adjusted_p_values_are_monotone_in_the_raw_ones,
    a_family_of_one_is_left_alone,
    a_smaller_level_rejects_no_more,
    an_uncovered_dependence_regime_is_refused,
)
"""Correction laws in conformance order."""


def check_null_model_laws[Data](fixture: NullModelFixture[Data]) -> None:
    """Run every null model law against one null.

    Parameters
    ----------
    fixture : NullModelFixture[Data]
        The null under test and its strategies.

    Raises
    ------
    LawViolationError
        On the first law the null fails.
    """
    for law in NULL_MODEL_LAWS:
        law(fixture)


def check_correction_laws(fixture: CorrectionFixture) -> None:
    """Run every correction law against one procedure.

    Parameters
    ----------
    fixture : CorrectionFixture
        The procedure under test and its strategies.

    Raises
    ------
    LawViolationError
        On the first law the procedure fails.
    """
    for law in CORRECTION_LAWS:
        law(fixture)
