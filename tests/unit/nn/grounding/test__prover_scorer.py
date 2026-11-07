from typing import TYPE_CHECKING, Final, cast

import pytest
from tests.conformance.component_plugin import plugin_context
from tests.language import language
from tests.verdicts import proved, refuted

from lofi._types import ComponentId
from lofi.lf import ProverBound, Refuted, TruthTableOracle
from lofi.lf.models import CountermodelFilter, CountermodelSettings
from lofi.lf.provers import (
    ProofBudget,
    ProofService,
    ProofSite,
    Questions,
    ServiceReport,
)
from lofi.nn.build import BuildContext, build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.grounding import ProverScorer
from lofi.spec.provers import ProofBudgetSpec
from lofi.spec.scorers import ProverScorerSpec
from lofi.terms import NonTerminatingError, Normalization

if TYPE_CHECKING:
    from lofi.lf import Verdict
    from lofi.terms import Term

LANGUAGE: Final = language(("rain", "wet"))
RAIN: Final = LANGUAGE.constant("rain")
WET: Final = LANGUAGE.constant("wet")
NORMALIZATION: Final = Normalization(
    fuel=32,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)


def _redex(value: Term) -> Term:
    store = LANGUAGE.store
    abstraction = store.signature.by_name(LANGUAGE.calculus.lam)
    application = store.signature.by_name(LANGUAGE.calculus.app)
    variable = store.bvar(LANGUAGE.term_sort, 0)
    return store.op(application, store.op(abstraction, variable), value)


class Service:
    def __init__(self) -> None:
        self.steps = 0

    def open_step(self, site: ProofSite, /) -> None:
        del site
        self.steps += 1

    @property
    def prover(self) -> str:
        return "test-prover"

    def step_report(self, site: ProofSite, /) -> ServiceReport:
        del site
        return ServiceReport(2, 0, {}, 0.5, 2, 1.0, None)

    def decide(
        self,
        questions: Questions,
        /,
        *,
        budget: ProofBudget,
        site: ProofSite,
    ) -> tuple[Verdict, ...]:
        del budget, site
        return tuple(
            proved("p") if premise == RAIN and goal == WET else refuted("c")
            for premise, goal in zip(questions.premises, questions.goals, strict=True)
        )


class RecordingService:
    def __init__(self) -> None:
        self.received: Questions | None = None

    def open_step(self, site: ProofSite, /) -> None:
        del site

    @property
    def prover(self) -> str:
        return "recording-prover"

    def step_report(self, site: ProofSite, /) -> ServiceReport:
        del site
        questions = 0 if self.received is None else len(self.received.premises)
        return ServiceReport(questions, 0, {}, 0.0, questions, 0.0, None)

    def decide(
        self,
        questions: Questions,
        /,
        *,
        budget: ProofBudget,
        site: ProofSite,
    ) -> tuple[Verdict, ...]:
        del budget, site
        self.received = questions
        return tuple(
            refuted("c") if premise == RAIN and goal == WET else proved("p")
            for premise, goal in zip(
                questions.premises,
                questions.goals,
                strict=True,
            )
        )


def _scorer(
    service: Service | RecordingService,
    /,
    *,
    normalization: Normalization = NORMALIZATION,
    countermodels: CountermodelFilter | None = None,
) -> ProverScorer:
    """Return a scorer under one interpretation-owned normalization."""
    return ProverScorer(
        cast("ProofService", service),
        ProofBudget(),
        LANGUAGE,
        normalization,
        ProofSite("fixture.prover-scorer"),
        countermodels=countermodels,
    )


def _build_context(
    service: ProofService,
    normalization: Normalization,
    /,
) -> BuildContext:
    """Return a component context over the scorer's exact language."""
    base = plugin_context()

    return BuildContext(
        component=ComponentId("grounding.scorer"),
        policy=base.policy,
        language=LANGUAGE,
        codec=NeuralCodecView(LANGUAGE.codec.at(LANGUAGE.term_sort)),
        vocabulary=base.vocabulary,
        actions=base.actions,
        rules=base.rules,
        categories=(RAIN, WET),
        producers={"normalization": normalization, "service": service},
        targets={},
        loader=None,
    )


def test_the_prover_scorer_returns_certified_scores_without_resetting_the_step() -> (
    None
):
    service = Service()
    scorer = _scorer(service)
    scores = scorer.score(Questions((RAIN, WET), (WET, RAIN)))

    assert service.steps == 0
    assert scorer.scorer_kind == "prover"
    assert scorer.source_identity == "test-prover"
    assert scorer.normalization is NORMALIZATION
    assert scorer.certified
    assert not scorer.differentiable
    assert scores.values.tolist() == [1.0, 0.0]
    assert scores.decided.tolist() == [True, True]
    assert scores.ledger.calls == 2
    assert scores.ledger.cpu_seconds == 0.5


def test_countermodels_remove_refuted_pairs_before_the_proof_service() -> None:
    service = RecordingService()
    countermodels = CountermodelFilter(
        LANGUAGE,
        CountermodelSettings(
            "exhaustive",
            carriers={"e": 1, "v": 1},
            structures=4,
            seed=3,
            full_function_space_cap=16,
        ),
    )
    scorer = _scorer(service, countermodels=countermodels)
    plain_service = RecordingService()
    plain = _scorer(plain_service)
    questions = Questions((RAIN, RAIN), (WET, RAIN))

    scores = scorer.score(questions)
    plain_scores = plain.score(questions)

    assert service.received == Questions((RAIN,), (RAIN,))
    assert isinstance(scores.evidence[0], Refuted)
    assert tuple(type(item) for item in scores.evidence) == tuple(
        type(item) for item in plain_scores.evidence
    )
    assert scores.values.tolist() == [0.0, 1.0]
    assert scores.values.tolist() == plain_scores.values.tolist()
    assert scores.decided.tolist() == [True, True]
    assert scores.ledger.calls == 1
    assert scores.ledger.questions == 1
    assert plain_scores.ledger.calls == 2


def test_beta_redexes_are_normalized_before_proof_dispatch() -> None:
    service = RecordingService()
    scorer = _scorer(service)

    scores = scorer.score(Questions((_redex(RAIN),), (_redex(WET),)))

    assert service.received == Questions((RAIN,), (WET,))
    assert scores.decided.tolist() == [True]


def test_the_plugin_reads_the_exact_normalization_from_its_named_producer() -> None:
    bound = ProverBound("truth-table", LANGUAGE.profile.fragment)
    service = ProofService(
        lambda: TruthTableOracle(
            LANGUAGE.store,
            LANGUAGE.profile,
            bound=bound,
            atoms=8,
        ),
        profile=LANGUAGE.profile,
        workers=1,
        batch_size=8,
        timeout=30.0,
    )
    normalization = Normalization(
        fuel=7,
        order="weak-head",
        eta="contract",
        associative_commutative=(),
    )

    try:
        scorer = build_component(
            ProverScorerSpec(budget=ProofBudgetSpec(calls_per_step=4)),
            _build_context(service, normalization),
        )

        assert isinstance(scorer, ProverScorer)
        assert scorer.normalization is normalization
    finally:
        service.close()


def test_the_interpretation_order_controls_reduction_beneath_a_lambda() -> None:
    store = LANGUAGE.store
    abstraction = store.signature.by_name(LANGUAGE.calculus.lam)
    application = store.signature.by_name(LANGUAGE.calculus.app)
    variable = store.bvar(LANGUAGE.term_sort, 0)
    identity = store.op(abstraction, variable)
    term = store.op(abstraction, store.op(application, identity, variable))
    reduced = store.op(abstraction, variable)
    full_service = RecordingService()
    weak_service = RecordingService()
    full = _scorer(full_service)
    weak = _scorer(
        weak_service,
        normalization=Normalization(
            fuel=32,
            order="weak-head",
            eta="off",
            associative_commutative=(),
        ),
    )

    full.score(Questions((term,), (WET,)))
    weak.score(Questions((term,), (WET,)))

    assert full_service.received == Questions((reduced,), (WET,))
    assert weak_service.received == Questions((term,), (WET,))


def test_the_interpretation_eta_policy_controls_proof_dispatch() -> None:
    store = LANGUAGE.store
    abstraction = store.signature.by_name(LANGUAGE.calculus.lam)
    application = store.signature.by_name(LANGUAGE.calculus.app)
    variable = store.bvar(LANGUAGE.term_sort, 0)
    expanded = store.op(abstraction, store.op(application, RAIN, variable))
    service = RecordingService()
    scorer = _scorer(
        service,
        normalization=Normalization(
            fuel=32,
            order="leftmost-outermost",
            eta="contract",
            associative_commutative=(),
        ),
    )

    scorer.score(Questions((expanded,), (WET,)))

    assert service.received == Questions((RAIN,), (WET,))


def test_the_interpretation_fuel_bounds_proof_normalization() -> None:
    scorer = _scorer(
        RecordingService(),
        normalization=Normalization(
            fuel=1,
            order="leftmost-outermost",
            eta="off",
            associative_commutative=(),
        ),
    )

    with pytest.raises(NonTerminatingError) as caught:
        scorer.score(Questions((_redex(_redex(RAIN)),), (WET,)))

    assert (caught.value.operation, caught.value.budget) == ("beta normalization", 1)
