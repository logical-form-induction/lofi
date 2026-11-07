"""One budgeted and memoized service for symbolic proof calls."""

from typing import TYPE_CHECKING, Final, override

import pytest
from tests.language import language
from tests.verdicts import proved, refuted

from lofi.lf import Abstain, Graded, LogicProfile, Proved, ProverBound, Refuted
from lofi.lf.provers import ProofBudget, ProofService, ProofSite, Questions

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.lf import Verdict
    from lofi.terms import Term


LANGUAGE: Final = language(("rain", "wet", "cold"))
RAIN: Final = LANGUAGE.constant("rain")
WET: Final = LANGUAGE.constant("wet")
COLD: Final = LANGUAGE.constant("cold")
SITE: Final = ProofSite("fixture")
OTHER_SITE: Final = ProofSite("other-fixture")


class _Oracle:
    """A recording oracle with deterministic typed answers."""

    def __init__(
        self,
        profile: LogicProfile,
        *,
        abstain: bool = False,
        abstention_cause: str = "gave-up",
        crash_on: Term | None = None,
    ) -> None:
        self._bound = ProverBound("fixture", profile.fragment)
        self._abstain = abstain
        self._abstention_cause = abstention_cause
        self._crash_on = crash_on
        self.calls: list[tuple[Term, Term, tuple[Term, ...], float]] = []

    @property
    def bound(self) -> ProverBound:
        return self._bound

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        self.calls.append((premise, hypothesis, tuple(theory), timeout))

        if premise is self._crash_on:
            msg = "fixture worker crashed"
            raise RuntimeError(msg)

        if self._abstain:
            return Abstain("fixture stopped", cause=self._abstention_cause)

        return proved("same term") if premise == hypothesis else refuted("different")


class _StartingOracle(_Oracle):
    """A long-lived process fixture that starts lazily on its first question."""

    def __init__(self, profile: LogicProfile) -> None:
        super().__init__(profile)
        self._started = False
        self._starts = 0

    @property
    def process_starts(self) -> int:
        return self._starts

    @override
    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        if not self._started:
            self._started = True
            self._starts += 1

        return super().entails(premise, hypothesis, theory=theory, timeout=timeout)


class _GradedOracle(_Oracle):
    """An oracle returning one declared degree for every question."""

    def __init__(self, profile: LogicProfile, degree: float) -> None:
        super().__init__(profile)
        self._degree = degree

    @override
    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        self.calls.append((premise, hypothesis, tuple(theory), timeout))
        return Graded("fixture", self._degree, "graded fixture decision")


def _service(
    *,
    abstain: bool = False,
    abstention_cause: str = "gave-up",
    crash_on: Term | None = None,
) -> tuple[ProofService, _Oracle]:
    oracle = _Oracle(
        LANGUAGE.profile,
        abstain=abstain,
        abstention_cause=abstention_cause,
        crash_on=crash_on,
    )
    return (
        ProofService(
            lambda: oracle,
            profile=LANGUAGE.profile,
            workers=1,
            batch_size=2,
            timeout=1.0,
        ),
        oracle,
    )


def test_the_service_constructs_one_long_lived_oracle_per_worker() -> None:
    oracles: list[_Oracle] = []

    def factory() -> _Oracle:
        oracle = _Oracle(LANGUAGE.profile)
        oracles.append(oracle)
        return oracle

    service = ProofService(
        factory,
        profile=LANGUAGE.profile,
        workers=3,
        batch_size=5,
        timeout=2.0,
    )

    try:
        assert service.workers == 3
        assert service.batch_size == 5
        assert service.timeout == 2.0
        assert len(oracles) == 3
        assert len({id(oracle) for oracle in oracles}) == 3
    finally:
        service.close()


def test_a_worker_factory_must_return_one_distinct_oracle_per_worker() -> None:
    oracle = _Oracle(LANGUAGE.profile)

    with pytest.raises(ValueError, match="distinct oracle"):
        ProofService(
            lambda: oracle,
            profile=LANGUAGE.profile,
            workers=2,
            batch_size=2,
            timeout=1.0,
        )


def test_process_starts_are_reported_for_the_current_step_and_lifetime() -> None:
    oracle = _StartingOracle(LANGUAGE.profile)
    service = ProofService(
        lambda: oracle,
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=2,
        timeout=1.0,
    )

    try:
        service.decide(
            Questions((RAIN, WET), (WET, COLD)), budget=ProofBudget(), site=SITE
        )
        first = service.step_report(SITE)
        service.open_step(SITE)
        service.decide(Questions((COLD,), (RAIN,)), budget=ProofBudget(), site=SITE)
        second = service.step_report(SITE)
        lifetime = service.report
    finally:
        service.close()

    assert first.process_starts == 1
    assert second.process_starts == 0
    assert lifetime.process_starts == 1


def test_questions_deduplicate_pairs_and_hold_one_common_theory() -> None:
    questions = Questions(
        (RAIN, RAIN, WET),
        (WET, WET, COLD),
        theory=(COLD,),
    )

    assert questions.premises == (RAIN, WET)
    assert questions.goals == (WET, COLD)
    assert questions.theory == (COLD,)


def test_the_service_publishes_its_single_prover_identity() -> None:
    service, _ = _service()

    try:
        assert service.prover == "fixture"
    finally:
        service.close()


def test_zero_calls_abstains_without_reaching_the_oracle() -> None:
    service, oracle = _service()

    try:
        verdicts = service.decide(
            Questions((RAIN,), (WET,)),
            budget=ProofBudget(calls_per_step=0),
            site=SITE,
        )
    finally:
        service.close()

    assert oracle.calls == []
    assert isinstance(verdicts[0], Abstain)
    assert verdicts[0].cause == "budget-spent"


def test_decisions_are_memoized_across_steps_but_abstentions_are_not() -> None:
    service, oracle = _service()
    questions = Questions((RAIN,), (WET,), theory=(COLD,))
    budget = ProofBudget(calls_per_step=1)

    try:
        first = service.decide(questions, budget=budget, site=SITE)
        service.open_step(SITE)
        second = service.decide(questions, budget=budget, site=SITE)
        report = service.report
        step_report = service.step_report(SITE)
    finally:
        service.close()

    assert first == second
    assert len(oracle.calls) == 1
    assert oracle.calls[0][2] == (COLD,)
    assert report.calls == 1
    assert report.memo_hits == 1
    assert report.questions == 2
    assert report.memo_hit_rate == 0.5
    assert report.wall_seconds > 0.0
    assert report.questions_per_second > 0.0
    assert report.process_starts is None
    assert step_report.calls == 0
    assert step_report.memo_hits == 1
    assert step_report.questions == 1
    assert step_report.memo_hit_rate == 1.0

    open_service, open_oracle = _service(abstain=True)

    try:
        open_service.decide(questions, budget=budget, site=SITE)
        open_service.open_step(SITE)
        open_service.decide(questions, budget=budget, site=SITE)
        open_report = open_service.report
    finally:
        open_service.close()

    assert len(open_oracle.calls) == 2
    assert open_report.abstentions["gave-up"] == 2


def test_a_settled_graded_decision_is_memoized() -> None:
    oracle = _GradedOracle(LANGUAGE.profile, 0.5)
    service = ProofService(
        lambda: oracle,
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=1,
        timeout=1.0,
    )
    questions = Questions((RAIN,), (WET,))

    try:
        first = service.decide(questions, budget=ProofBudget(), site=SITE)
        second = service.decide(questions, budget=ProofBudget(), site=SITE)
    finally:
        service.close()

    assert first == second == (Graded("fixture", 0.5, "graded fixture decision"),)
    assert len(oracle.calls) == 1


def test_mutual_unit_graded_evidence_settles_equivalence() -> None:
    oracle = _GradedOracle(LANGUAGE.profile, 1.0)
    service = ProofService(
        lambda: oracle,
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=1,
        timeout=1.0,
    )

    try:
        verdict = service.equivalent(RAIN, WET, budget=ProofBudget(), site=SITE)
    finally:
        service.close()

    assert isinstance(verdict, Graded)
    assert verdict.degree == 1.0
    assert len(oracle.calls) == 2


def test_a_bounded_cache_evicts_the_least_recently_used_decision() -> None:
    oracle = _Oracle(LANGUAGE.profile)
    service = ProofService(
        lambda: oracle,
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=1,
        timeout=1.0,
        cache_size=1,
    )
    budget = ProofBudget()

    try:
        service.decide(Questions((RAIN,), (WET,)), budget=budget, site=SITE)
        service.decide(Questions((WET,), (COLD,)), budget=budget, site=SITE)
        service.decide(Questions((RAIN,), (WET,)), budget=budget, site=SITE)
    finally:
        service.close()

    assert service.cache_size == 1
    assert len(oracle.calls) == 3


def test_a_custom_abstention_cause_survives_service_reporting() -> None:
    service, _ = _service(abstain=True, abstention_cause="license-expired")

    try:
        service.decide(Questions((RAIN,), (WET,)), budget=ProofBudget(), site=SITE)
        report = service.report
    finally:
        service.close()

    assert report.abstentions["license-expired"] == 1
    assert report.abstentions["gave-up"] == 0


def test_a_worker_crash_abstains_without_losing_other_questions() -> None:
    service, oracle = _service(crash_on=RAIN)

    try:
        verdicts = service.decide(
            Questions((RAIN, WET), (WET, COLD)),
            budget=ProofBudget(),
            site=SITE,
        )
        report = service.report
    finally:
        service.close()

    assert len(oracle.calls) == 2
    assert isinstance(verdicts[0], Abstain)
    assert verdicts[0].cause == "gave-up"
    assert "RuntimeError" in verdicts[0].reason
    assert isinstance(verdicts[1], Refuted)
    assert report.abstentions["gave-up"] == 1


def test_the_step_call_budget_is_shared_by_every_question() -> None:
    service, oracle = _service()

    try:
        verdicts = service.decide(
            Questions((RAIN, WET), (RAIN, COLD)),
            budget=ProofBudget(calls_per_step=1),
            site=SITE,
        )
        report = service.report
    finally:
        service.close()

    assert len(oracle.calls) == 1
    assert not isinstance(verdicts[0], Abstain)
    assert isinstance(verdicts[1], Abstain)
    assert verdicts[1].cause == "budget-spent"
    assert report.abstentions["budget-spent"] == 1


def test_open_step_resets_only_the_step_budget() -> None:
    service, oracle = _service()
    budget = ProofBudget(calls_per_step=1)

    try:
        service.decide(Questions((RAIN,), (WET,)), budget=budget, site=SITE)
        first = service.decide(Questions((WET,), (COLD,)), budget=budget, site=SITE)
        service.open_step(SITE)
        second = service.decide(Questions((WET,), (COLD,)), budget=budget, site=SITE)
    finally:
        service.close()

    assert isinstance(first[0], Abstain)
    assert not isinstance(second[0], Abstain)
    assert len(oracle.calls) == 2


def test_a_zero_total_cpu_budget_spends_no_call() -> None:
    service, oracle = _service()

    try:
        verdict = service.decide(
            Questions((RAIN,), (WET,)),
            budget=ProofBudget(cpu_seconds=0.0),
            site=SITE,
        )[0]
    finally:
        service.close()

    assert oracle.calls == []
    assert isinstance(verdict, Abstain)
    assert verdict.cause == "budget-spent"


def test_equivalence_uses_the_shared_memo_and_combines_two_proofs() -> None:
    service, oracle = _service()
    budget = ProofBudget()

    try:
        verdict = service.equivalent(
            RAIN, RAIN, budget=budget, site=SITE, theory=(COLD,)
        )
        service.open_step(SITE)
        repeated = service.equivalent(
            RAIN, RAIN, budget=budget, site=SITE, theory=(COLD,)
        )
        report = service.report
    finally:
        service.close()

    assert isinstance(verdict, Proved)
    assert verdict.proof.system == "mutual-entailment"
    assert repeated == verdict
    assert len(oracle.calls) == 1
    assert report.calls == 1
    assert report.memo_hits == 3


def test_equivalence_stops_after_the_forward_nonproof() -> None:
    service, oracle = _service()

    try:
        verdict = service.equivalent(
            RAIN,
            WET,
            budget=ProofBudget(calls_per_step=1),
            site=SITE,
        )
    finally:
        service.close()

    assert isinstance(verdict, Refuted)
    assert len(oracle.calls) == 1


def test_equivalence_spends_one_budget_across_both_directions() -> None:
    service, oracle = _service()

    try:
        verdict = service.equivalent(
            RAIN,
            RAIN,
            budget=ProofBudget(calls_per_step=0),
            site=SITE,
        )
    finally:
        service.close()

    assert isinstance(verdict, Abstain)
    assert verdict.cause == "budget-spent"
    assert oracle.calls == []


def test_simultaneous_use_site_budgets_are_isolated() -> None:
    service, oracle = _service()
    budget = ProofBudget(calls_per_step=1)

    try:
        first = service.decide(Questions((RAIN,), (WET,)), budget=budget, site=SITE)
        second = service.decide(
            Questions((WET,), (COLD,)), budget=budget, site=OTHER_SITE
        )
        spent = service.decide(Questions((COLD,), (RAIN,)), budget=budget, site=SITE)
        first_report = service.step_report(SITE)
        second_report = service.step_report(OTHER_SITE)
    finally:
        service.close()

    assert not isinstance(first[0], Abstain)
    assert not isinstance(second[0], Abstain)
    assert isinstance(spent[0], Abstain)
    assert len(oracle.calls) == 2
    assert first_report.calls == 1
    assert first_report.abstentions["budget-spent"] == 1
    assert second_report.calls == 1
    assert second_report.abstentions["budget-spent"] == 0
