import math
from typing import TYPE_CHECKING, Final, NamedTuple, cast, override

import didactic.api as dx
import pytest
import torch
from tests.conformance.objective import ObjectiveLawSuite
from tests.language import language
from tests.verdicts import proved

from lofi._types import ReadoutId
from lofi.deduce import SpanIndex
from lofi.lf import (
    Abstain,
    LogicFragment,
    LogicProfile,
    ProverBound,
    Refuted,
    TruthTableOracle,
)
from lofi.lf.provers import ProofBudget, ProofService, ProofSite
from lofi.nn._values import LogDistribution
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._errors import (
    EveryItemAbstainedError,
    MissingOutputError,
)
from lofi.nn.loss._parser import SPAN_AXES
from lofi.nn.loss._proof import (
    PLUGIN,
    FormOutcome,
    ProofShaping,
    ShapedAbstention,
    ShapedCredit,
    ShapedProofObjective,
    ShapedProofReward,
)
from lofi.nn.loss._suite import ObjectiveSuite
from lofi.nn.loss._values import (
    CellChart,
    ObjectiveOutputs,
    RecordingOutputs,
    SpanTerms,
)
from lofi.nn.loss.laws import (
    ObjectiveFixture,
    the_value_is_monotone_where_the_fixture_says,
)
from lofi.spec.losses.shaped_proof import ProofShapingSpec, ShapedProofLossSpec
from lofi.spec.provers import ProofBudgetSpec
from lofi.terms import Term

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from lofi.lf import Verdict
    from lofi.nn.loss._ports import LossComponentBindings, LossTerm
    from lofi.spec.language import LanguageAssembly
    from lofi.typecheck import LexicalSignature


LANGUAGE: Final = language(
    ("cold", "dog", "rain", "wet"),
    types={"cold": "t", "dog": "e -> t", "rain": "t", "wet": "t"},
)
STORE: Final = LANGUAGE.store
RULES: Final = LANGUAGE.rules
SPANS: Final = SpanIndex(2)


def _op(name: str, *arguments: Term) -> Term:
    return LANGUAGE.apply(LANGUAGE.constant(name), *arguments)


RAIN: Final = LANGUAGE.constant("rain")
WET: Final = LANGUAGE.constant("wet")
COLD: Final = LANGUAGE.constant("cold")

# a truth applied to a truth: closed, and the arrow rule has no way to type it
MISTYPED: Final = LANGUAGE.apply(RAIN, WET)

# a bare de Bruijn index: one enclosing binder short of denoting anything
UNBOUND: Final = STORE.bvar(LANGUAGE.term_sort, 0)

# outside the propositional fragment, so the prover abstains rather than deciding
QUANTIFIED: Final = LANGUAGE.read("forall x. dog(x)")

SIGNATURE: Final = LANGUAGE.lexicon
PROFILE: Final = LogicProfile(
    STORE,
    formula_sort=LANGUAGE.term_sort,
    application=LANGUAGE.calculus.app,
    classes=LANGUAGE.constant_classes,
    fragment=LogicFragment(order="first"),
    type_kinds=LANGUAGE.profile.type_kinds,
    variables=LANGUAGE.profile.variables,
    semantics_name=LANGUAGE.profile.semantics,
    negation=LANGUAGE.profile.negation,
    label_map=LANGUAGE.profile.label_map,
    theory=(_op("if", RAIN, WET),),
    domains={"e": 2, "v": 1},
    roles=LANGUAGE.profile.logical_roles,
)
BOUND: Final = ProverBound(
    "truth-table",
    LogicFragment(order="first"),
)
SHAPING: Final = ProofShaping(
    ill_formed=0.0,
    well_formed=0.25,
    type_checks=0.5,
    proved=1.0,
    length=0.05,
)
PROOF_BUDGET: Final = ProofBudget()
_SERVICES: list[ProofService] = []

# the class each form falls into under each reference, for the by-hand value
OUTCOMES: Final[Mapping[tuple[Term, Term | None], FormOutcome]] = {
    (RAIN, RAIN): FormOutcome.PROVED,
    (WET, RAIN): FormOutcome.TYPE_CHECKS,
    (MISTYPED, RAIN): FormOutcome.WELL_FORMED,
    (UNBOUND, RAIN): FormOutcome.ILL_FORMED,
    (WET, WET): FormOutcome.PROVED,
    (COLD, WET): FormOutcome.TYPE_CHECKS,
    (MISTYPED, WET): FormOutcome.WELL_FORMED,
    (RAIN, None): FormOutcome.TYPE_CHECKS,
    (WET, None): FormOutcome.TYPE_CHECKS,
}

type Outputs = ObjectiveOutputs[Items]
type Weighted = tuple[tuple[Term, ...], tuple[float, ...]]


class Items(NamedTuple):
    words: tuple[tuple[str, ...], ...]
    gold_forms: tuple[Term | None, ...]


class Item(NamedTuple):
    beam: Weighted
    reference: Term | None


FIRST: Final = Item(((RAIN, WET, MISTYPED, UNBOUND), (0.4, 0.3, 0.2, 0.1)), RAIN)
SECOND: Final = Item(((WET, COLD, MISTYPED), (0.5, 0.3, 0.2)), WET)
UNREFERENCED: Final = Item(((RAIN, WET), (0.6, 0.4)), None)
UNDECIDABLE: Final = Item(((QUANTIFIED, RAIN), (0.7, 0.3)), RAIN)
FORMS: Final = ReadoutId("forms")


class TimingOut:
    """An oracle whose every decision runs out the clock."""

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
        del premise, hypothesis, theory

        return Abstain(f"no answer within {timeout} seconds", cause="timeout")


class UnexpectedOracle:
    """An oracle that fails a test if a proof request reaches it."""

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
        pytest.fail("the proof oracle was asked")


class RecordingOracle:
    """An oracle that records the two directions of a proved equivalence."""

    def __init__(self) -> None:
        self.asked: list[tuple[Term, Term, float]] = []

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
        del theory
        self.asked.append((premise, hypothesis, timeout))
        return proved("recorded")


def oracle() -> TruthTableOracle:
    return TruthTableOracle(STORE, PROFILE, bound=BOUND, atoms=12)


def reward(
    prover: TruthTableOracle
    | TimingOut
    | UnexpectedOracle
    | RecordingOracle
    | None = None,
) -> ShapedProofReward[LexicalSignature]:
    service = ProofService(
        lambda: oracle() if prover is None else prover,
        profile=PROFILE,
        workers=1,
        batch_size=8,
        timeout=30.0,
    )
    _SERVICES.append(service)

    return ShapedProofReward(
        LANGUAGE.checker,
        service,
        signature=SIGNATURE,
        rules=RULES,
        shaping=SHAPING,
        budget=PROOF_BUDGET,
        site=ProofSite("fixture.shaped-proof"),
    )


def teardown_module() -> None:
    for service in _SERVICES:
        service.close()


def objective(
    paid: ShapedProofReward[LexicalSignature] | None = None,
) -> ShapedProofObjective[LexicalSignature]:
    return ShapedProofObjective(
        reward() if paid is None else paid,
        readout=FORMS,
        name="shaped-proof",
    )


class ProofSource(NamedTuple):
    """Language capabilities supplied to the shaped-proof plugin."""

    language: LanguageAssembly
    service: ProofService


class ProofContext:
    """Narrow fixture context for the shaped-proof plugin."""

    def __init__(self, service: ProofService) -> None:
        self._service = service

    @property
    def name(self) -> str:
        """Return the configured objective name."""
        return "proof-control"

    def proof_reward(self) -> ProofSource:
        """Return the fixture language and shared proof service."""
        return ProofSource(LANGUAGE, self._service)


def weighted_forms(weighted: Weighted) -> WeightedTerms:
    terms, probabilities = weighted
    leaf = torch.tensor(probabilities, dtype=torch.float64).log().requires_grad_()

    return WeightedTerms(terms, leaf, torch.tensor(-torch.inf), Estimate.SAMPLED)


def chart(items: int, fill: float = -1.0) -> CellChart:
    computed = torch.zeros(items, SPANS.num_spans, dtype=torch.bool)
    computed[:, SPANS.span_id(0, 2)] = True
    held = torch.where(computed, -0.5, fill)

    return CellChart(LogDistribution(held, SPAN_AXES, computed))


def assemble(*items: Item, fill: float = -1.0) -> Outputs:
    cells: dict[tuple[int, int], WeightedTerms] = {}

    for position, item in enumerate(items):
        cells[position, SPANS.span_id(0, 2)] = weighted_forms(item.beam)

    return ObjectiveOutputs(
        spans=SPANS,
        chart=chart(len(items), fill),
        batch=Items(
            tuple(("one", "item") for _ in items),
            tuple(item.reference for item in items),
        ),
        terms={FORMS: SpanTerms(SPANS, cells)},
    )


def bundle() -> Outputs:
    return assemble(FIRST, SECOND)


def inputs_of(outputs: Outputs) -> tuple[Tensor, ...]:
    return tuple(held.log_weights for held in outputs.terms[FORMS].cells.values())


def published(outputs: Outputs, position: int) -> WeightedTerms:
    return outputs.terms[FORMS].at(position, SPANS.span_id(0, 2))


def expected_payment(item: Item) -> float:
    terms, probabilities = item.beam
    total = 0.0

    for form, probability in zip(terms, probabilities, strict=True):
        outcome = OUTCOMES[form, item.reference]
        total += probability * SHAPING.value(outcome, form.size)

    return total / sum(probabilities)


def by_hand(outputs: Outputs) -> float:
    batch = outputs.batch
    losses: list[float] = []

    for position, reference in enumerate(batch.gold_forms):
        held = published(outputs, position)
        losses.append(
            -expected_payment(
                Item(
                    (held.terms, tuple(held.log_weights.detach().exp().tolist())),
                    reference,
                )
            )
        )

    return sum(losses) / len(losses)


def permuted(outputs: Outputs) -> Outputs:
    items = tuple(
        Item(
            (
                published(outputs, position).terms,
                tuple(published(outputs, position).log_weights.detach().exp().tolist()),
            ),
            reference,
        )
        for position, reference in enumerate(outputs.batch.gold_forms)
    )

    return assemble(*items[::-1])


def refilled(outputs: Outputs) -> Outputs:
    return ObjectiveOutputs(
        spans=outputs.spans,
        chart=chart(len(outputs.batch.words), fill=-7.0),
        batch=outputs.batch,
        terms=outputs.terms,
    )


def models_built(call: Callable[[], LossTerm]) -> int:
    counted = 0
    original = dx.Model.__init__

    def counting(model: dx.Model, /, **fields: int) -> None:
        nonlocal counted
        counted += 1

        original(model, **fields)

    with pytest.MonkeyPatch.context() as patch:
        patch.setattr(dx.Model, "__init__", counting)
        call()

    return counted


def fixture() -> ObjectiveFixture[Items]:
    heavier = Item(((RAIN, WET, MISTYPED, UNBOUND), (0.1, 0.2, 0.3, 0.4)), RAIN)

    return ObjectiveFixture(
        objective=objective(),
        outputs=bundle,
        inputs=inputs_of,
        unweighted=by_hand,
        monotone=lambda: (assemble(heavier, SECOND), assemble(FIRST, SECOND)),
        models_built=models_built,
        permuted=permuted,
        masked=refilled,
        tolerance=1e-9,
    )


class TestShapedProofObjectiveLaws(ObjectiveLawSuite):
    @pytest.fixture
    def conformance(self) -> ObjectiveFixture[Items]:
        return fixture()

    @override
    def test_the_value_is_monotone_where_the_fixture_says[BatchT](
        self,
        conformance: ObjectiveFixture[BatchT],
    ) -> None:
        """Check that moving weight onto the ill-formed forms scores higher."""
        the_value_is_monotone_where_the_fixture_says(conformance)


@pytest.mark.parametrize(
    ("form", "reference", "outcome"),
    [
        (UNBOUND, RAIN, FormOutcome.ILL_FORMED),
        (MISTYPED, RAIN, FormOutcome.WELL_FORMED),
        (WET, RAIN, FormOutcome.TYPE_CHECKS),
        (RAIN, RAIN, FormOutcome.PROVED),
    ],
)
def test_each_outcome_class_earns_its_declared_credit_before_the_length_penalty(
    form: Term, reference: Term, outcome: FormOutcome
) -> None:
    paid = reward().score(form, reference=reference)

    assert isinstance(paid, ShapedCredit)
    assert paid.outcome is outcome
    assert paid.credit == SHAPING.credit(outcome)
    assert paid.length == form.size
    assert paid.value == pytest.approx(paid.credit - SHAPING.length * form.size)


def test_the_four_declared_credits_are_four_different_numbers() -> None:
    earned = {
        SHAPING.credit(outcome)
        for outcome in (
            FormOutcome.ILL_FORMED,
            FormOutcome.WELL_FORMED,
            FormOutcome.TYPE_CHECKS,
            FormOutcome.PROVED,
        )
    }

    assert len(earned) == 4


def test_a_countermodel_and_a_timeout_on_the_same_pair_come_back_differently() -> None:
    refuted = reward().score(WET, reference=RAIN)
    timed = reward(TimingOut()).score(WET, reference=RAIN)

    assert isinstance(refuted, ShapedCredit)
    assert isinstance(refuted.verdict, Refuted)
    assert refuted.value == pytest.approx(
        SHAPING.credit(FormOutcome.TYPE_CHECKS) - SHAPING.length * WET.size
    )
    assert isinstance(timed, ShapedAbstention)
    assert timed.verdict.cause == "timeout"
    assert not isinstance(timed, ShapedCredit)


def test_a_countermodel_keeps_the_typing_credit_and_zeroes_the_proof_term() -> None:
    refuted = reward().score(WET, reference=RAIN)
    proved = reward().score(RAIN, reference=RAIN)

    assert isinstance(refuted, ShapedCredit)
    assert isinstance(proved, ShapedCredit)
    assert proved.credit - refuted.credit == pytest.approx(
        SHAPING.credit(FormOutcome.PROVED) - SHAPING.credit(FormOutcome.TYPE_CHECKS)
    )


def test_a_form_with_no_reference_stops_at_the_type_checking_credit() -> None:
    paid = reward().score(RAIN, reference=None)

    assert isinstance(paid, ShapedCredit)
    assert paid.outcome is FormOutcome.TYPE_CHECKS
    assert paid.verdict is None


def test_an_ill_formed_form_never_reaches_the_type_checker_or_the_prover() -> None:
    paid = reward(UnexpectedOracle()).score(UNBOUND, reference=RAIN)

    assert isinstance(paid, ShapedCredit)
    assert paid.outcome is FormOutcome.ILL_FORMED


def test_the_length_penalty_is_charged_per_node_of_the_form() -> None:
    conjoined = _op("and", RAIN, WET)

    paid = reward().score(conjoined, reference=conjoined)

    assert isinstance(paid, ShapedCredit)
    assert paid.outcome is FormOutcome.PROVED
    assert paid.length == conjoined.size
    assert paid.credit - paid.value == pytest.approx(SHAPING.length * conjoined.size)


def test_a_shaping_whose_credits_do_not_ladder_upward_is_refused() -> None:
    with pytest.raises(ValueError, match="pays a form for reaching less"):
        ProofShaping(
            ill_formed=0.0,
            well_formed=0.5,
            type_checks=0.25,
            proved=1.0,
            length=0.05,
        )


def test_a_negative_length_weight_is_refused() -> None:
    with pytest.raises(ValueError, match="cannot be negative"):
        ProofShaping(
            ill_formed=0.0,
            well_formed=0.25,
            type_checks=0.5,
            proved=1.0,
            length=-0.05,
        )


def test_a_shaping_refuses_reassignment_and_compares_by_its_declared_numbers() -> None:
    same = ProofShaping(
        ill_formed=0.0,
        well_formed=0.25,
        type_checks=0.5,
        proved=1.0,
        length=0.05,
    )
    other = ProofShaping(
        ill_formed=0.0,
        well_formed=0.25,
        type_checks=0.5,
        proved=1.0,
        length=0.10,
    )

    with pytest.raises(AttributeError, match="immutable"):
        SHAPING.__setattr__("_length", 1.0)

    assert same == SHAPING
    assert hash(same) == hash(SHAPING)
    assert other != SHAPING


def test_two_payments_agree_only_when_every_field_does() -> None:
    certificate = proved("no countermodel")
    paid = ShapedCredit(FormOutcome.PROVED, 1.0, 1, 0.95, certificate)

    assert paid == ShapedCredit(FormOutcome.PROVED, 1.0, 1, 0.95, certificate)
    assert paid != ShapedCredit(FormOutcome.PROVED, 1.0, 1, 0.95, None)
    assert paid != ShapedAbstention(Abstain("gave up"))

    with pytest.raises(AttributeError, match="immutable"):
        paid.__setattr__("_value", 2.0)

    with pytest.raises(AttributeError, match="immutable"):
        paid.__delattr__("_outcome")


def test_a_proved_equivalence_asks_the_oracle_in_both_directions() -> None:
    prover = RecordingOracle()

    paid = reward(prover).score(RAIN, reference=WET)

    assert isinstance(paid, ShapedCredit)
    assert paid.outcome is FormOutcome.PROVED
    assert prover.asked == [(RAIN, WET, 30.0), (WET, RAIN, 30.0)]


def test_the_value_is_the_mean_negative_expected_payment_of_the_items() -> None:
    outputs = bundle()

    term = objective().evaluate(outputs)

    assert term.count == 2
    assert term.value.item() == pytest.approx(by_hand(outputs), rel=1e-12)
    assert term.value.item() == pytest.approx(
        -(expected_payment(FIRST) + expected_payment(SECOND)) / 2.0, rel=1e-9
    )


def test_the_plugin_builds_the_declared_shaping_and_shared_proof_service() -> None:
    service = ProofService(
        oracle,
        profile=PROFILE,
        workers=1,
        batch_size=8,
        timeout=30.0,
    )
    _SERVICES.append(service)
    declaration = ShapedProofLossSpec(
        readout=FORMS,
        shaping=ProofShapingSpec(
            ill_formed=0.0,
            well_formed=0.25,
            type_checks=0.5,
            proved=1.0,
            length=0.05,
        ),
        budget=ProofBudgetSpec(calls_per_step=4),
    )

    built = PLUGIN.construct(
        declaration,
        cast("LossComponentBindings", ProofContext(service)),
    )
    measured = built.evaluate(bundle())

    assert built.name == "proof-control"
    assert built.shaping == SHAPING
    assert measured.count == 2
    assert measured.value.item() == pytest.approx(by_hand(bundle()), rel=1e-12)


def test_moving_weight_onto_a_proved_form_lowers_the_value() -> None:
    lighter = objective().evaluate(assemble(FIRST))
    heavier = objective().evaluate(
        assemble(Item(((RAIN, WET, MISTYPED, UNBOUND), (0.7, 0.1, 0.1, 0.1)), RAIN))
    )

    assert heavier.value.item() < lighter.value.item()


def test_an_item_with_no_reference_is_scored_without_asking_the_prover() -> None:
    term = objective(reward(UnexpectedOracle())).evaluate(assemble(UNREFERENCED))

    assert term.count == 1
    assert term.value.item() == pytest.approx(-expected_payment(UNREFERENCED), rel=1e-9)


def test_an_abstaining_member_leaves_the_beam_and_the_rest_are_renormalized() -> None:
    term = objective().evaluate(assemble(UNDECIDABLE))
    decided = SHAPING.value(FormOutcome.PROVED, RAIN.size)

    assert term.count == 1
    assert term.diagnostics["abstention_rate"].item() == 0.0
    assert term.value.item() == pytest.approx(-decided, rel=1e-12)


def test_a_batch_of_nothing_but_abstentions_raises_rather_than_meaning_a_zero() -> None:
    subject = objective(reward(TimingOut()))
    provable = assemble(
        Item(((RAIN, WET), (0.6, 0.4)), RAIN),
        Item(((WET, COLD), (0.5, 0.5)), WET),
    )

    with pytest.raises(EveryItemAbstainedError, match="all 2 of this batch abstained"):
        subject.evaluate(provable)


def test_an_item_that_abstains_throughout_enters_no_mean_and_is_reported() -> None:
    stalling = Item(((QUANTIFIED,), (1.0,)), QUANTIFIED)

    term = objective().evaluate(assemble(FIRST, stalling))

    assert term.count == 1
    assert term.diagnostics["abstention_rate"].item() == pytest.approx(0.5)
    assert term.value.item() == pytest.approx(-expected_payment(FIRST), rel=1e-9)


def test_the_objective_reads_the_scored_forms_and_the_batch_and_nothing_else() -> None:
    subject = objective()
    view = RecordingOutputs(bundle(), objective=subject.name, requires=subject.requires)

    subject.evaluate(view)

    assert subject.requires == frozenset({"batch", "spans", "terms"})
    assert view.touched == frozenset({"batch", "spans", "terms"})


def test_a_graph_publishing_no_decoded_terms_is_refused_at_wiring() -> None:
    available = bundle().published

    with pytest.raises(MissingOutputError, match="terms"):
        ObjectiveSuite([(1.0, objective())], published=available - {"terms"})

    wired = ObjectiveSuite([(1.0, objective())], published=available)

    assert wired.requires == frozenset({"batch", "spans", "terms"})


def test_an_item_naming_an_unpublished_readout_is_refused() -> None:
    outputs = bundle()
    orphaned = ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=Items((("one", "item"),), (RAIN,)),
        terms={ReadoutId("another"): outputs.terms[FORMS]},
    )

    with pytest.raises(ValueError, match="forms"):
        objective().evaluate(orphaned)


def test_a_batch_naming_more_items_than_gold_forms_is_refused() -> None:
    outputs = bundle()
    lopsided = ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=Items(outputs.batch.words, (RAIN,)),
        terms=outputs.terms,
    )

    with pytest.raises(ValueError, match="2 items and 1 gold forms"):
        objective().evaluate(lopsided)


def test_a_batch_naming_no_item_at_all_is_refused() -> None:
    outputs = bundle()
    empty = ObjectiveOutputs(
        spans=outputs.spans,
        chart=outputs.chart,
        batch=Items((), ()),
        terms=outputs.terms,
    )

    with pytest.raises(ValueError, match="the batch names none"):
        objective().evaluate(empty)


def test_the_gradient_of_the_term_reaches_the_beam_weights() -> None:
    outputs = bundle()
    term = objective().evaluate(outputs)

    gradients = torch.autograd.grad(term.value, inputs_of(outputs))

    assert all(bool(torch.isfinite(gradient).all()) for gradient in gradients)
    assert all(float(gradient.abs().max()) > 0.0 for gradient in gradients)


def test_the_reward_reads_its_single_proof_budget() -> None:
    paid = reward()

    assert paid.budget is PROOF_BUDGET


def test_the_shaping_the_reward_was_built_with_is_readable_from_the_objective() -> None:
    subject = objective()

    assert subject.shaping is SHAPING
    assert subject.shaping.credit(FormOutcome.PROVED) == 1.0
    assert math.isclose(subject.shaping.length, 0.05)
