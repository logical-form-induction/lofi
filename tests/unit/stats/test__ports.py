from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING

import pytest

from lofi.stats import (
    Decisions,
    Dependence,
    DependenceError,
    NullFamily,
    PValue,
    Tail,
    p_value_against_replicates,
)
from lofi.stats._ports import Correction, NullModel, Statistic

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

NOTHING_MATCHED: Mapping[str, str | int] = MappingProxyType({})


class _SignFlipNull:
    def __init__(self, name: str) -> None:
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

    def test(
        self,
        statistic: Statistic[tuple[float, ...]],
        observed: tuple[float, ...],
        /,
        *,
        tail: Tail,
    ) -> PValue:
        flipped = [
            statistic(
                tuple(sign * value for sign, value in zip(signs, observed, strict=True))
            )
            for signs in product((1.0, -1.0), repeat=len(observed))
        ]

        return p_value_against_replicates(
            statistic(observed),
            flipped,
            tail=tail,
            null=self._name,
            family=self.family,
            exact=True,
            seed=None,
        )


class _Unadjusted:
    def __init__(self, valid_under: frozenset[Dependence]) -> None:
        self._valid_under = valid_under

    @property
    def name(self) -> str:
        return "unadjusted"

    @property
    def valid_under(self) -> frozenset[Dependence]:
        return self._valid_under

    def apply(
        self, pvalues: Sequence[PValue], alpha: float, dependence: Dependence, /
    ) -> Decisions:
        if dependence not in self._valid_under:
            raise DependenceError(
                correction=self.name,
                dependence=dependence,
                valid_under=self._valid_under,
            )

        return Decisions(
            self.name,
            alpha,
            dependence,
            pvalues,
            tuple(pvalue.value for pvalue in pvalues),
        )


def total(values: tuple[float, ...]) -> float:
    return sum(values)


def surviving(
    null: NullModel[tuple[float, ...]],
    correction: Correction,
    samples: Sequence[tuple[float, ...]],
    *,
    alpha: float,
    dependence: Dependence,
) -> Decisions:
    family = tuple(null.test(total, sample, tail="greater") for sample in samples)

    return correction.apply(family, alpha, dependence)


def test_an_implementation_offers_the_surface_each_protocol_declares() -> None:
    assert isinstance(_SignFlipNull("sign flips"), NullModel)
    assert isinstance(_Unadjusted(frozenset(Dependence)), Correction)


def test_a_consumer_reaches_a_null_and_a_correction_through_the_ports() -> None:
    decisions = surviving(
        _SignFlipNull("sign flips"),
        _Unadjusted(frozenset(Dependence)),
        [(3.0, 2.5, 4.0, 3.5, 2.0, 3.0, 4.5), (0.5, -0.4, 0.3, -0.2, 0.1, -0.6, 0.2)],
        alpha=0.05,
        dependence=Dependence.ARBITRARY,
    )

    assert len(decisions.raw) == 2
    assert decisions.adjusted == tuple(pvalue.value for pvalue in decisions.raw)
    # one of 128 sign assignments reaches the observation
    assert decisions.adjusted[0] == pytest.approx(1 / 2**7)
    assert decisions.reject == (True, False)


def test_a_null_records_what_it_declared_about_itself() -> None:
    null = _SignFlipNull("sign flips")
    result = null.test(total, (1.0, -2.0, 3.0, 0.5), tail="greater")

    assert result.provenance.null == "sign flips"
    assert result.provenance.family == "permutation"
    assert result.provenance.replicates == 2**4
    assert result.provenance.seed is None
    assert result.provenance.exact is True


def test_an_enumerated_null_tests_to_the_same_p_value_twice() -> None:
    null = _SignFlipNull("sign flips")
    sample = (1.0, -2.0, 3.0, 0.5)

    assert null.test(total, sample, tail="greater") == null.test(
        total, sample, tail="greater"
    )


def test_a_regime_a_procedure_does_not_cover_is_refused() -> None:
    correction = _Unadjusted(frozenset({Dependence.INDEPENDENT}))
    family = (_SignFlipNull("sign flips").test(total, (1.0, 2.0), tail="greater"),)

    with pytest.raises(DependenceError, match="ARBITRARY"):
        correction.apply(family, 0.05, Dependence.ARBITRARY)


def test_a_procedure_says_which_regimes_it_covers() -> None:
    correction = _Unadjusted(frozenset({Dependence.INDEPENDENT, Dependence.PRDS}))
    family = (_SignFlipNull("sign flips").test(total, (1.0, 2.0), tail="greater"),)

    assert correction.apply(family, 0.05, Dependence.PRDS).dependence is Dependence.PRDS
    assert Dependence.ARBITRARY not in correction.valid_under
