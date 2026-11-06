from statistics import fmean
from time import perf_counter
from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.language import language
from tests.verdicts import proved

from lofi._types import ItemId
from lofi.eval import ConventionVector
from lofi.eval._equivalence import (
    ACCOUNTING_MEASURES,
    EQUIVALENCE_MEASURES,
    EquivalenceAccountingMetric,
    EquivalenceLedger,
    EquivalenceMetric,
    EquivalenceScores,
    equivalence_scores,
    equivalence_status,
)
from lofi.eval._ports import SZS_STATUSES, Metric
from lofi.lf import (
    ABSTENTION_CAUSES,
    Abstain,
    LogicProfile,
    ProverBound,
    SearchBound,
    TruthTableOracle,
)
from lofi.lf.provers import ProofBudget, ProofService

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.eval._equivalence import AccountingMeasure, EquivalenceMeasure
    from lofi.lf import AbstentionCause, EntailmentOracle, Verdict
    from lofi.terms import Term


LANGUAGE: Final = language(
    (
        "cold",
        "dog",
        "p0",
        "p1",
        "p10",
        "p11",
        "p12",
        "p2",
        "p3",
        "p4",
        "p5",
        "p6",
        "p7",
        "p8",
        "p9",
        "q0",
        "q1",
        "q10",
        "q11",
        "q12",
        "q13",
        "q14",
        "q15",
        "q2",
        "q3",
        "q4",
        "q5",
        "q6",
        "q7",
        "q8",
        "q9",
        "rain",
        "rex",
        "wet",
    ),
    types={"dog": "e -> t", "rex": "e"},
)

STORE: Final = LANGUAGE.store

# a generous wall clock for every decided pair, and one the costly pair outlasts
TIMEOUT: Final = 30.0
CLOCK: Final = 1e-3
BUDGET: Final = ProofBudget()
CONVENTIONS: Final = ConventionVector(
    punctuation="keep",
    unary_chains="keep",
    trivial_spans="keep",
    minimum_length=1,
    aggregation="corpus",
    coverage="zero",
    self_arcs="allow",
    gold_derivation="canonical",
)
_SERVICES: list[ProofService] = []


def form(text: str) -> Term:
    return LANGUAGE.read(text)


def _op(name: str, *arguments: Term) -> Term:
    return LANGUAGE.apply(LANGUAGE.constant(name), *arguments)


RAIN: Final = form("rain")
WET: Final = form("wet")
COLD: Final = form("cold")
RAIN_AND_WET: Final = form("rain ∧ wet")
RAIN_MEANS_WET: Final = form("rain => wet")
WET_MEANS_RAIN: Final = form("wet => rain")
QUANTIFIED: Final = form("∀x.dog(x)")

LETTERS: Final = tuple(LANGUAGE.constant(f"p{at}") for at in range(13))
LONG_LETTERS: Final = tuple(LANGUAGE.constant(f"q{at}") for at in range(16))


def _conjoin(forms: Sequence[Term]) -> Term:
    held = forms[-1]

    for each in reversed(forms[:-1]):
        held = _op("and", each, held)

    return held


BIG_CONJUNCTION: Final = _conjoin(LETTERS)
LONGER_TEXT: Final = " ∧ ".join(f"q{at}" for at in range(16))

FIRST: Final = LANGUAGE.profile.fragment
BOUND: Final = ProverBound("truth-table", FIRST)


def _profile(theory: tuple[Term, ...] = (RAIN_MEANS_WET,)) -> LogicProfile:
    return LogicProfile(
        STORE,
        formula_sort=LANGUAGE.term_sort,
        application=LANGUAGE.calculus.app,
        classes=LANGUAGE.constant_classes,
        fragment=FIRST,
        type_kinds=LANGUAGE.profile.type_kinds,
        variables=LANGUAGE.profile.variables,
        semantics_name=LANGUAGE.profile.semantics,
        negation=LANGUAGE.profile.negation,
        label_map=LANGUAGE.profile.label_map,
        theory=theory,
        domains={},
        roles=LANGUAGE.profile.logical_roles,
    )


def _entailment(*, theory: tuple[Term, ...] = (RAIN_MEANS_WET,)) -> TruthTableOracle:
    return TruthTableOracle(STORE, _profile(theory), bound=BOUND, atoms=20)


def _service(
    oracle: EntailmentOracle,
    /,
    *,
    profile: LogicProfile,
    timeout: float = TIMEOUT,
) -> ProofService:
    service = ProofService(
        lambda: oracle,
        profile=profile,
        workers=1,
        batch_size=8,
        timeout=timeout,
    )
    _SERVICES.append(service)
    return service


def _proof_service(*, timeout: float = TIMEOUT) -> ProofService:
    return _service(_entailment(), profile=_profile(), timeout=timeout)


def _metric(
    service: ProofService,
    measure: EquivalenceMeasure = "equivalence",
    *,
    budget: ProofBudget = BUDGET,
    theory: Sequence[Term] = (),
) -> EquivalenceMetric:
    return EquivalenceMetric(
        EquivalenceLedger(service, LANGUAGE, budget=budget, theory=theory),
        measure=measure,
        conventions=CONVENTIONS,
    )


class _Recording:
    """An entailment oracle proving everything, remembering what it was asked."""

    def __init__(self) -> None:
        self.asked: list[tuple[Term, Term]] = []

    @property
    def bound(self) -> ProverBound:
        return BOUND

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        del theory, timeout
        self.asked.append((premise, hypothesis))
        return proved("stub")


class _Abstaining:
    """An entailment oracle abstaining at once, always with one cause."""

    def __init__(self, cause: AbstentionCause) -> None:
        self._cause = cause

    @property
    def bound(self) -> ProverBound:
        return BOUND

    def entails(
        self,
        premise: Term,
        hypothesis: Term,
        /,
        *,
        theory: Sequence[Term] = (),
        timeout: float,
    ) -> Verdict:
        del premise, hypothesis, theory, timeout
        return Abstain(
            "stopped",
            cause=self._cause,
            bound=(
                SearchBound("questions", 1) if self._cause == "search-bound" else None
            ),
        )


def _assumed_service() -> ProofService:
    return _service(_entailment(theory=()), profile=_profile(()))


def teardown_module() -> None:
    for service in _SERVICES:
        service.close()


def test_the_ledger_reports_its_default_theory_and_service() -> None:
    service = _proof_service()
    metric = _metric(service)

    assert metric.ledger.theory == ()
    assert metric.ledger.service.prover == "truth-table"


def test_the_ledger_preserves_its_declared_theory() -> None:
    assumptions = (RAIN_MEANS_WET, WET_MEANS_RAIN)

    assert _metric(_assumed_service(), theory=assumptions).ledger.theory == assumptions


def test_alpha_variants_are_the_same_interned_terms_for_a_pure_oracle() -> None:
    recording = _Recording()
    service = _service(recording, profile=_profile(()))
    first = form("∀x.dog(x)")
    second = form("∀y.dog(y)")
    third = form("∀z.dog(z)")

    assert first is second
    assert (
        equivalence_status(service, first, third, budget=BUDGET, theory=()) == "Theorem"
    )
    assert (
        equivalence_status(service, second, first, budget=BUDGET, theory=())
        == "Theorem"
    )
    assert recording.asked == [(first, first)]


def test_the_pure_equivalence_view_decides_both_directions() -> None:
    assert (
        equivalence_status(
            _proof_service(), RAIN, RAIN_AND_WET, budget=BUDGET, theory=()
        )
        == "Theorem"
    )
    assert (
        equivalence_status(_proof_service(), RAIN, WET, budget=BUDGET, theory=())
        == "CounterSatisfiable"
    )
    assert (
        equivalence_status(_proof_service(), QUANTIFIED, WET, budget=BUDGET, theory=())
        == "GaveUp"
    )


def test_assumptions_can_make_two_atoms_equivalent() -> None:
    service = _assumed_service()

    assert (
        equivalence_status(
            service,
            RAIN,
            WET,
            budget=BUDGET,
            theory=(RAIN_MEANS_WET, WET_MEANS_RAIN),
        )
        == "Theorem"
    )


def _scores(predicted: str, gold: Term, service: ProofService) -> EquivalenceScores:
    return equivalence_scores(service, LANGUAGE, predicted, gold, budget=BUDGET)


def test_text_that_does_not_parse_is_ill_formed_and_never_reaches_the_prover() -> None:
    recording = _Recording()
    service = _service(recording, profile=_profile(()))
    scores = _scores("rain ∧", RAIN, service)

    assert scores == EquivalenceScores(
        parsed=False, typed=False, status=None, seconds=None
    )
    assert not scores.well_formed
    assert not scores.decided
    assert recording.asked == []


def test_text_with_a_free_variable_is_ill_formed() -> None:
    recording = _Recording()
    service = _service(recording, profile=_profile(()))

    assert _scores("0", RAIN, service).parsed is False
    assert _scores("unknown", RAIN, service).parsed is False
    assert recording.asked == []


@pytest.mark.parametrize("text", ["rain(rex)"])
def test_text_that_does_not_type_check_is_ill_formed_but_parsed(text: str) -> None:
    recording = _Recording()
    service = _service(recording, profile=_profile(()))
    scores = _scores(text, RAIN, service)

    assert scores.parsed
    assert not scores.typed
    assert scores.status is None
    assert scores.seconds is None
    assert recording.asked == []


def test_well_formed_text_is_checked_and_timed() -> None:
    recording = _Recording()
    service = _service(recording, profile=_profile(()))
    scores = _scores("rain ∧ wet", RAIN, service)

    assert scores.well_formed
    assert scores.status == "Theorem"
    assert scores.decided
    assert scores.seconds is not None
    assert scores.seconds > 0.0
    assert scores.components()["theorem"] == 1.0
    assert scores.components()["proof-time"] == scores.seconds
    assert recording.asked == [(RAIN_AND_WET, RAIN), (RAIN, RAIN_AND_WET)]


def test_a_prediction_that_is_not_text_is_refused() -> None:
    with pytest.raises(TypeError, match="text"):
        equivalence_scores(
            _proof_service(),
            LANGUAGE,
            cast("str", RAIN),
            RAIN,
            budget=BUDGET,
        )


def _four_items(
    metric: EquivalenceMetric,
) -> None:
    metric.update("rain ∧ wet", RAIN, key=ItemId("theorem"))
    metric.update("wet", RAIN, key=ItemId("countermodel"))
    metric.update("dog(", RAIN, key=ItemId("ill-formed"))
    metric.update(LONGER_TEXT, LONG_LETTERS[0], key=ItemId("timeout"))


def test_the_metric_is_driven_through_the_protocol_it_conforms_to() -> None:
    metric: Metric[str, Term] = _metric(_proof_service())

    assert isinstance(metric, Metric)
    assert metric.name == "equivalence-equivalence"
    assert metric.order_invariant
    assert metric.conventions == CONVENTIONS
    assert metric.compute().value is None
    assert metric.compute().n == 0


def test_the_four_rates_are_separate_metrics_over_one_set_of_items() -> None:
    service = _proof_service(timeout=0.02)
    metrics = {
        measure: _metric(service, measure, budget=ProofBudget())
        for measure in EQUIVALENCE_MEASURES
    }

    for metric in metrics.values():
        _four_items(metric)

    results = {measure: metric.compute() for measure, metric in metrics.items()}

    assert {metric.name for metric in metrics.values()} == {
        "equivalence-equivalence",
        "equivalence-well-formed",
        "equivalence-timeout",
        "equivalence-proof-time",
    }
    assert results["equivalence"].value == 0.25
    assert results["well-formed"].value == 0.75
    assert results["timeout"].value == 0.25
    assert results["proof-time"].value is not None
    assert results["proof-time"].value > 0.0
    assert all(result.n == 4 for result in results.values())


def test_every_rate_is_a_component_of_every_result() -> None:
    metric = _metric(_proof_service(timeout=0.02), budget=ProofBudget())
    _four_items(metric)
    result = metric.compute()

    assert result.value == 0.25
    assert result.components["theorem"] == 0.25
    assert result.components["counter-satisfiable"] == 0.25
    assert result.components["timeout"] == 0.25
    assert result.components["ill-formed"] == 0.25
    assert result.components["resource-out"] == 0.0
    assert result.components["gave-up"] == 0.0
    assert result.components["parsed"] == 0.75
    assert result.components["well-formed"] == 0.75


def test_the_mean_proof_time_averages_decided_checks_only() -> None:
    metric = _metric(_proof_service(timeout=0.02), "proof-time", budget=ProofBudget())
    _four_items(metric)
    per_item = metric.per_item()
    decided = [
        row.components["proof-time"]
        for row in per_item.values()
        if "proof-time" in row.components
    ]

    assert set(per_item) == {
        ItemId("theorem"),
        ItemId("countermodel"),
        ItemId("ill-formed"),
        ItemId("timeout"),
    }
    assert per_item[ItemId("timeout")].components["timeout"] == 1.0
    assert "proof-time" not in per_item[ItemId("timeout")].components
    assert "proof-time" not in per_item[ItemId("ill-formed")].components
    assert len(decided) == 2
    assert metric.compute().value == fmean(decided)
    assert per_item[ItemId("timeout")].value is None
    assert per_item[ItemId("theorem")].value == decided[0]


def test_the_mean_proof_time_is_absent_when_nothing_was_decided() -> None:
    metric = _metric(_proof_service(), "proof-time")
    metric.update("dog(", RAIN, key=ItemId("ill-formed"))
    result = metric.compute()

    assert result.value is None
    assert result.n == 1
    assert "proof-time" not in result.components


def test_results_do_not_mix_outcomes_with_oracle_service_accounting() -> None:
    metric = _metric(_proof_service())
    metric.update("rain ∧ wet", RAIN, key=ItemId("a"))
    metric.update("wet", RAIN, key=ItemId("b"))
    result = metric.compute()

    assert "budget" not in result.components
    assert "calls" not in result.components
    assert all(
        "budget" not in row.components and "calls" not in row.components
        for row in metric.per_item().values()
    )


def test_two_pure_decisions_are_both_reported() -> None:
    metric = _metric(_proof_service())
    metric.update("rain ∧ wet", RAIN, key=ItemId("a"))
    metric.update("cold", WET, key=ItemId("b"))
    result = metric.compute()

    assert result.value == 0.5
    assert result.components["counter-satisfiable"] == 0.5
    assert result.n == 2


def test_alpha_variant_predictions_are_each_decided_by_the_pure_oracle() -> None:
    recording = _Recording()
    metric = _metric(_service(recording, profile=_profile(())))
    metric.update("∀x.dog(x)", QUANTIFIED, key=ItemId("x"))
    metric.update("∀y.dog(y)", QUANTIFIED, key=ItemId("y"))

    assert metric.compute().value == 1.0
    assert recording.asked == [(QUANTIFIED, QUANTIFIED)]


def test_an_item_is_scored_once_and_a_conflicting_rescore_is_refused() -> None:
    metric = _metric(_proof_service())
    metric.update("rain ∧ wet", RAIN, key=ItemId("a"))
    metric.update("rain ∧ wet", RAIN, key=ItemId("a"))

    assert metric.compute().n == 1

    with pytest.raises(ValueError, match="different pair"):
        metric.update("wet", RAIN, key=ItemId("a"))


def test_a_reset_forgets_the_items() -> None:
    metric = _metric(_proof_service())
    metric.update("rain ∧ wet", RAIN, key=ItemId("a"))
    metric.reset()

    assert metric.compute().value is None


def test_an_unknown_measure_is_refused_at_construction() -> None:
    with pytest.raises(ValueError, match="measure"):
        EquivalenceMetric(
            EquivalenceLedger(_proof_service(), LANGUAGE, budget=BUDGET),
            measure=cast("EquivalenceMeasure", "soundness"),
            conventions=CONVENTIONS,
        )

    assert _metric(_proof_service(), budget=BUDGET).ledger.budget is BUDGET
    assert _metric(_proof_service(), "timeout").measure == "timeout"


@pytest.mark.parametrize("cause", ABSTENTION_CAUSES)
def test_every_abstention_cause_maps_to_an_undecided_status(
    cause: AbstentionCause,
) -> None:
    status = equivalence_status(
        _service(_Abstaining(cause), profile=_profile(())),
        RAIN,
        WET,
        budget=BUDGET,
        theory=(),
    )

    assert status in SZS_STATUSES
    assert status not in {"Theorem", "CounterSatisfiable"}


def test_the_status_is_read_from_the_cause_not_the_clock() -> None:
    clocked = _Abstaining("timeout")
    spent = _Abstaining("budget-spent")
    gave_up = _Abstaining("out-of-fragment")
    started = perf_counter()
    status = equivalence_status(
        _service(clocked, profile=_profile(())),
        RAIN,
        WET,
        budget=BUDGET,
        theory=(),
    )
    elapsed = perf_counter() - started

    assert status == "Timeout"
    assert elapsed < TIMEOUT
    assert (
        equivalence_status(
            _service(spent, profile=_profile(())),
            RAIN,
            WET,
            budget=BUDGET,
            theory=(),
        )
        == "ResourceOut"
    )
    assert (
        equivalence_status(
            _service(gave_up, profile=_profile(())),
            RAIN,
            WET,
            budget=BUDGET,
            theory=(),
        )
        == "GaveUp"
    )


def test_rows_sharing_a_ledger_decide_each_item_once() -> None:
    recording = _Recording()
    ledger = EquivalenceLedger(
        _service(recording, profile=_profile(())), LANGUAGE, budget=BUDGET
    )
    rows = (
        *(
            EquivalenceMetric(ledger, measure=measure, conventions=CONVENTIONS)
            for measure in EQUIVALENCE_MEASURES
        ),
        *(
            EquivalenceAccountingMetric(
                ledger, measure=measure, conventions=CONVENTIONS
            )
            for measure in ACCOUNTING_MEASURES
        ),
    )

    for row in rows:
        row.update("rain ∧ wet", RAIN, key=ItemId("a"))

    assert recording.asked == [(RAIN_AND_WET, RAIN), (RAIN, RAIN_AND_WET)]
    assert rows[0].compute().value == 1.0


def test_accounting_rows_count_calls_and_memo_hits_apart_from_the_rates() -> None:
    recording = _Recording()
    ledger = EquivalenceLedger(
        _service(recording, profile=_profile(())), LANGUAGE, budget=BUDGET
    )
    calls = EquivalenceAccountingMetric(
        ledger, measure="proof-calls", conventions=CONVENTIONS
    )
    hits = EquivalenceAccountingMetric(
        ledger, measure="memo-hits", conventions=CONVENTIONS
    )

    for row in (calls, hits):
        row.update("rain ∧ wet", RAIN, key=ItemId("a"))
        row.update("rain ∧ wet", RAIN, key=ItemId("b"))
        row.update("dog(", RAIN, key=ItemId("ill-formed"))

    assert calls.name == "equivalence-proof-calls"
    assert calls.compute().value == 2.0
    assert hits.compute().value == 2.0
    assert calls.per_item()[ItemId("b")].value == 0.0
    assert hits.per_item()[ItemId("b")].value == 2.0
    assert calls.per_item()[ItemId("ill-formed")].value == 0.0
    assert calls.compute().n == 3


def test_a_budget_of_zero_calls_makes_every_uncached_check_abstain() -> None:
    recording = _Recording()
    ledger = EquivalenceLedger(
        _service(recording, profile=_profile(())),
        LANGUAGE,
        budget=ProofBudget(calls_per_step=0),
    )
    rate = EquivalenceMetric(ledger, measure="equivalence", conventions=CONVENTIONS)
    calls = EquivalenceAccountingMetric(
        ledger, measure="proof-calls", conventions=CONVENTIONS
    )

    for row in (rate, calls):
        row.update("rain ∧ wet", RAIN, key=ItemId("a"))

    assert rate.compute().value == 0.0
    assert rate.compute().components["resource-out"] == 1.0
    assert calls.compute().value == 0.0
    assert recording.asked == []


def test_an_unknown_accounting_measure_is_refused() -> None:
    with pytest.raises(ValueError, match="accounting measure"):
        EquivalenceAccountingMetric(
            EquivalenceLedger(_proof_service(), LANGUAGE, budget=BUDGET),
            measure=cast("AccountingMeasure", "questions"),
            conventions=CONVENTIONS,
        )
