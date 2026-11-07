from typing import TYPE_CHECKING, cast

import pytest
from hypothesis import strategies as st

from lofi.stats import (
    BenjaminiHochberg,
    BenjaminiYekutieli,
    Bonferroni,
    Correction,
    Dependence,
    DependenceError,
    Holm,
    PValue,
    false_discovery_correction,
    p_value_against_replicates,
)
from lofi.stats.laws import CORRECTION_LAWS, CorrectionFixture, check_correction_laws

if TYPE_CHECKING:
    from collections.abc import Callable, Sequence

EXAMPLES = 20

FINE = tuple(float(step) for step in range(10000))
COARSE = tuple(float(step) for step in range(20))

HARMONIC_OF_FIVE = 137 / 60


def a_p_value(probability: float, reference: tuple[float, ...] = FINE) -> PValue:
    observed = float(len(reference) - round(probability * len(reference)))

    return p_value_against_replicates(
        observed,
        reference,
        tail="greater",
        null="rank permutation",
        family="permutation",
        exact=True,
        seed=None,
    )


def a_family(*probabilities: float) -> tuple[PValue, ...]:
    return tuple(a_p_value(probability) for probability in probabilities)


def rejected(
    correction: Correction, family: tuple[PValue, ...], alpha: float
) -> set[int]:
    decisions = correction.apply(family, alpha, Dependence.PRDS)

    return {at for at, reject in enumerate(decisions.reject) if reject}


CORRECTIONS = (Bonferroni(), Holm(), BenjaminiHochberg(), BenjaminiYekutieli())
NAMES = [correction.name for correction in CORRECTIONS]
FALSE_DISCOVERY = (BenjaminiHochberg(), BenjaminiYekutieli())
FALSE_DISCOVERY_NAMES = [correction.name for correction in FALSE_DISCOVERY]

# 100 evenly spaced null p-values
UNIFORM = a_family(*((step + 1) / 100 for step in range(100)))

# ten small p-values followed by 90 null p-values
PLANTED = a_family(*(0.0001 * (step + 1) for step in range(10))) + UNIFORM[:90]

FAMILIES = st.lists(
    st.integers(min_value=1, max_value=len(COARSE)).map(
        lambda rank: a_p_value(rank / len(COARSE), COARSE)
    ),
    min_size=1,
    max_size=5,
).map(tuple)
LEVELS = st.floats(min_value=0.01, max_value=0.5)

FIXTURES = tuple(
    CorrectionFixture(correction, regime, FAMILIES, LEVELS, EXAMPLES)
    for correction in CORRECTIONS
    for regime in sorted(correction.valid_under, key=lambda regime: regime.name)
)
FIXTURE_NAMES = [
    f"{fixture.correction.name}-{fixture.dependence.name}" for fixture in FIXTURES
]


class _JustApplies:
    def apply(
        self, pvalues: Sequence[PValue], _alpha: float, _dependence: Dependence, /
    ) -> tuple[PValue, ...]:
        return tuple(pvalues)


@pytest.mark.parametrize("correction", CORRECTIONS, ids=NAMES)
def test_every_correction_answers_under_the_regimes_it_declares_and_no_others(
    correction: Correction,
) -> None:
    family = a_family(0.01, 0.4)

    assert isinstance(correction, Correction)

    for regime in Dependence:
        if regime in correction.valid_under:
            assert correction.apply(family, 0.05, regime).dependence is regime

        else:
            with pytest.raises(DependenceError, match=correction.name):
                correction.apply(family, 0.05, regime)


def test_a_type_with_only_an_apply_method_is_not_a_correction() -> None:
    assert not isinstance(_JustApplies(), Correction)


def test_bonferroni_multiplies_every_p_value_by_the_family_size() -> None:
    family = a_family(0.001, 0.02, 0.03, 0.5, 0.6)
    decisions = Bonferroni().apply(family, 0.05, Dependence.ARBITRARY)

    assert decisions.adjusted == pytest.approx((0.005, 0.1, 0.15, 1.0, 1.0))
    assert decisions.reject == (True, False, False, False, False)
    assert decisions.n_rejected == 1


def test_holm_raises_each_adjusted_value_to_the_largest_one_before_it() -> None:
    # raw products: 0.05, 0.08, 0.09, 0.08, 0.05
    family = a_family(0.01, 0.02, 0.03, 0.04, 0.05)
    decisions = Holm().apply(family, 0.05, Dependence.ARBITRARY)

    assert decisions.adjusted == pytest.approx((0.05, 0.08, 0.09, 0.09, 0.09))
    assert decisions.n_rejected == 1


def test_holm_rejects_everything_bonferroni_does_and_sometimes_more() -> None:
    family = a_family(0.001, 0.02, 0.03, 0.5, 0.6)

    assert rejected(Bonferroni(), family, 0.1) < rejected(Holm(), family, 0.1)


def test_the_step_up_procedure_lowers_each_value_to_the_smallest_one_after_it() -> None:
    # raw products: 0.005, 0.05, 0.05, 0.625, 0.6
    family = a_family(0.001, 0.02, 0.03, 0.5, 0.6)
    decisions = BenjaminiHochberg().apply(family, 0.05, Dependence.PRDS)

    assert decisions.adjusted == pytest.approx((0.005, 0.05, 0.05, 0.6, 0.6))
    assert decisions.n_rejected == 3


def test_the_conservative_procedure_penalizes_by_the_harmonic_sum() -> None:
    family = a_family(0.001, 0.02, 0.03, 0.5, 0.6)
    decisions = BenjaminiYekutieli().apply(family, 0.05, Dependence.ARBITRARY)

    assert decisions.adjusted == pytest.approx(
        (
            0.005 * HARMONIC_OF_FIVE,
            0.05 * HARMONIC_OF_FIVE,
            0.05 * HARMONIC_OF_FIVE,
            1.0,
            1.0,
        )
    )
    assert decisions.n_rejected == 1


def test_the_conservative_procedure_rejects_a_subset_of_the_other_one() -> None:
    lenient = rejected(BenjaminiHochberg(), PLANTED, 0.2)
    conservative = rejected(BenjaminiYekutieli(), PLANTED, 0.2)

    assert conservative < lenient


@pytest.mark.parametrize("correction", CORRECTIONS, ids=NAMES)
def test_a_family_of_one_is_returned_uncorrected(correction: Correction) -> None:
    family = a_family(0.04)
    decisions = correction.apply(family, 0.05, Dependence.PRDS)

    assert decisions.adjusted == pytest.approx((0.04,))
    assert decisions.reject == (True,)


@pytest.mark.parametrize("correction", CORRECTIONS, ids=NAMES)
def test_a_correction_hands_back_the_family_it_was_given(
    correction: Correction,
) -> None:
    family = a_family(0.5, 0.001, 0.2)
    decisions = correction.apply(family, 0.05, Dependence.PRDS)

    assert decisions.raw == family
    assert decisions.raw[0].value == 0.5
    assert decisions.raw[0].provenance.null == "rank permutation"
    assert decisions.correction == correction.name
    assert decisions.dependence is Dependence.PRDS
    assert len(decisions.adjusted) == len(family)


@pytest.mark.parametrize("correction", FALSE_DISCOVERY, ids=FALSE_DISCOVERY_NAMES)
def test_a_hundred_uniform_p_values_are_not_all_rejected(
    correction: Correction,
) -> None:
    decisions = correction.apply(UNIFORM, 0.05, Dependence.PRDS)

    assert len(decisions.reject) == len(UNIFORM)
    assert decisions.n_rejected == 0
    assert min(decisions.adjusted) > 0.05


@pytest.mark.parametrize(
    ("correction", "levels", "counts"),
    [
        (BenjaminiHochberg(), (0.005, 0.01, 0.2), (0, 10, 12)),
        (BenjaminiYekutieli(), (0.05, 0.2, 0.5), (0, 10, 11)),
    ],
    ids=FALSE_DISCOVERY_NAMES,
)
def test_the_level_moves_the_rejection_count_on_both_false_discovery_branches(
    correction: Correction, levels: tuple[float, ...], counts: tuple[int, ...]
) -> None:
    rejections = tuple(
        correction.apply(PLANTED, alpha, Dependence.PRDS).n_rejected for alpha in levels
    )

    assert rejections == counts


def test_the_step_up_procedure_refuses_arbitrary_dependence() -> None:
    with pytest.raises(DependenceError, match="benjamini-hochberg is valid under"):
        BenjaminiHochberg().apply(a_family(0.01, 0.5), 0.05, Dependence.ARBITRARY)


@pytest.mark.parametrize(
    ("regime", "expected"),
    [
        (Dependence.INDEPENDENT, BenjaminiHochberg),
        (Dependence.PRDS, BenjaminiHochberg),
        (Dependence.ARBITRARY, BenjaminiYekutieli),
    ],
    ids=[regime.name for regime in Dependence],
)
def test_a_regime_is_routed_to_a_procedure_whose_guarantee_covers_it(
    regime: Dependence,
    expected: type[BenjaminiHochberg | BenjaminiYekutieli],
) -> None:
    correction = false_discovery_correction(regime)
    decisions = correction.apply(a_family(0.001, 0.4), 0.05, regime)

    assert isinstance(correction, expected)
    assert regime in correction.valid_under
    assert decisions.dependence is regime


def test_a_regime_that_is_not_one_of_the_declared_regimes_is_refused() -> None:
    spelled = cast("Dependence", "prds")

    with pytest.raises(TypeError, match="dependence regime is not one"):
        false_discovery_correction(spelled)

    with pytest.raises(TypeError, match="dependence regime is not one"):
        Bonferroni().apply(a_family(0.01), 0.05, spelled)


@pytest.mark.parametrize("correction", CORRECTIONS, ids=NAMES)
def test_a_correction_refuses_scores_that_were_never_tested(
    correction: Correction,
) -> None:
    importances = cast("tuple[PValue, ...]", (0.81, 0.44, 0.02))

    with pytest.raises(TypeError, match="which is not a p-value"):
        correction.apply(importances, 0.05, Dependence.PRDS)


@pytest.mark.parametrize("correction", CORRECTIONS, ids=NAMES)
def test_a_correction_refuses_an_empty_family(correction: Correction) -> None:
    with pytest.raises(ValueError, match="no p-values to correct"):
        correction.apply((), 0.05, Dependence.PRDS)


@pytest.mark.conformance
@pytest.mark.parametrize("fixture", FIXTURES, ids=FIXTURE_NAMES)
def test_every_law_holds_of_every_correction_under_every_regime_it_covers(
    fixture: CorrectionFixture,
) -> None:
    check_correction_laws(fixture)


@pytest.mark.conformance
@pytest.mark.parametrize("law", CORRECTION_LAWS, ids=lambda law: law.__name__)
def test_each_law_runs_against_the_step_up_procedure_on_its_own(
    law: Callable[[CorrectionFixture], None],
) -> None:
    fixture = CorrectionFixture(
        BenjaminiHochberg(), Dependence.PRDS, FAMILIES, LEVELS, EXAMPLES
    )

    law(fixture)
