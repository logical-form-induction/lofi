"""Conformance tests for entailment oracles."""

from time import perf_counter
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest

from lofi.lf import Abstain, EntailmentOracle, Proved, Refuted

if TYPE_CHECKING:
    from collections.abc import Callable

    from lofi.terms import Term

__all__ = ["EntailmentOracleSuite", "OracleCase"]

# how much longer than its declared wall clock an abstaining call may take
SLACK: Final = 1.0

# how much faster an abstention on the wall clock must return than the decision
# it cut short
PROMPTNESS: Final = 10.0


class OracleCase(NamedTuple):
    """Inputs for the entailment-oracle conformance laws.

    Attributes
    ----------
    build : Callable[[], EntailmentOracle]
        Builds the oracle under test.
    proved : tuple[Term, Term]
        A premise and a hypothesis the oracle decides by proving, which do not
        entail each other in both directions.
    refuted : tuple[Term, Term]
        A premise and a hypothesis the oracle decides by countermodel.
    beyond : tuple[Term, Term]
        A pair using something outside what the oracle reads.
    costly : tuple[Term, Term]
        A pair the oracle decides, expensive enough to time against the wall
        clock.
    timeout : float
        A wall clock generous enough for every decided pair above.
    clock : float
        A wall clock the costly pair's decision outlasts many times over, and
        long enough that no single step of the decision spans it.
    out_of_bound : Callable[[], None]
        Builds an oracle whose profile lies outside its bound, which must raise.
    """

    build: Callable[[], EntailmentOracle]
    proved: tuple[Term, Term]
    refuted: tuple[Term, Term]
    beyond: tuple[Term, Term]
    costly: tuple[Term, Term]
    timeout: float
    clock: float
    out_of_bound: Callable[[], None]


class EntailmentOracleSuite:
    """Conformance laws for entailment-oracle implementations."""

    pytestmark = pytest.mark.conformance

    def test_the_oracle_answers_the_declared_port(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        oracle = entailment_oracle_conformance.build()

        assert isinstance(oracle, EntailmentOracle)

    def test_a_proved_pair_is_proved_with_a_certificate(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        case = entailment_oracle_conformance
        oracle = case.build()
        premise, hypothesis = case.proved
        verdict = oracle.entails(premise, hypothesis, timeout=case.timeout)

        assert isinstance(verdict, Proved)
        assert verdict.proof.derivation

    def test_a_refuted_pair_yields_a_countermodel(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        case = entailment_oracle_conformance
        oracle = case.build()
        premise, hypothesis = case.refuted
        verdict = oracle.entails(premise, hypothesis, timeout=case.timeout)

        assert isinstance(verdict, Refuted)
        assert verdict.evidence

    def test_a_pair_outside_the_fragment_abstains_with_a_reason(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        case = entailment_oracle_conformance
        oracle = case.build()
        premise, hypothesis = case.beyond
        verdict = oracle.entails(premise, hypothesis, timeout=case.timeout)

        assert isinstance(verdict, Abstain)
        assert verdict.reason

    def test_a_hard_pair_abstains_within_the_declared_wall_clock(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        case = entailment_oracle_conformance
        premise, hypothesis = case.costly
        started = perf_counter()
        decided = case.build().entails(premise, hypothesis, timeout=case.timeout)
        unclocked = perf_counter() - started
        started = perf_counter()
        verdict = case.build().entails(premise, hypothesis, timeout=case.clock)
        elapsed = perf_counter() - started

        assert not isinstance(decided, Abstain)
        assert isinstance(verdict, Abstain)
        assert case.clock <= elapsed < case.clock + SLACK
        assert elapsed * PROMPTNESS < unclocked

    def test_a_repeated_call_returns_the_same_verdict_class(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        case = entailment_oracle_conformance
        oracle = case.build()
        premise, hypothesis = case.costly
        first = oracle.entails(premise, hypothesis, timeout=case.timeout)
        second = oracle.entails(premise, hypothesis, timeout=case.timeout)

        assert type(first) is type(second)

    def test_a_profile_outside_the_bound_is_refused_at_construction(
        self, entailment_oracle_conformance: OracleCase
    ) -> None:
        with pytest.raises(ValueError, match=r"does not serve|cannot serve"):
            entailment_oracle_conformance.out_of_bound()
