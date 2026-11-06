from types import MappingProxyType
from typing import TYPE_CHECKING, cast

import pytest
import torch
from tests.conformance.component_plugin import plugin_context
from tests.language import language
from tests.verdicts import proved

from lofi._types import ReadoutId
from lofi.lf import (
    CLASSICAL,
    RTE,
    THREE_WAY,
    Abstain,
    ArgumentRenaming,
    InferenceLabel,
    Proved,
    ProverBound,
    TruthTableOracle,
)
from lofi.lf.provers import (
    ProofBudget,
    ProofService,
    ProofSite,
    Questions,
    ServiceReport,
)
from lofi.nn.build import BuildContext, build_component
from lofi.nn.codec import NeuralCodecView
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import (
    GroundingInput,
    GroundingQuery,
    Marginal,
    NeuralProofScorer,
    PairwiseGrounding,
    Phase,
    ProofSearch,
)
from lofi.nn.sites import ComponentId
from lofi.spec.groundings.neural_proof import (
    NeuralProofScorerSpec,
    RuleTemplate,
    SimilarityKernelSettings,
    SimilarityKernelSpec,
)
from lofi.spec.language import ConstantSpec, LanguageIso, LogicSpec, registered_language
from lofi.spec.provers import ProofBudgetSpec
from lofi.terms import Normalization

if TYPE_CHECKING:
    from lofi.lf import Verdict
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Term


DECLARED = language(
    ("rain", "wet", "cold", "dog", "cat", "love", "meta", "scope", "rex"),
    types={
        "dog": "e -> t",
        "cat": "e -> t",
        "love": "e -> e -> t",
        "meta": "(e -> t) -> t",
        "scope": "t -> t",
        "rex": "e",
    },
)


def _second_order() -> LanguageAssembly:
    source = registered_language("neo-davidsonian")
    return LanguageIso().forward(
        source.with_(
            name="neural-proof-second-order",
            constants=(
                *source.constants,
                ConstantSpec(name="dog", type="e -> t"),
                ConstantSpec(name="rex", type="e"),
                ConstantSpec(name="fido", type="e"),
            ),
            logic=LogicSpec(order="second"),
        )
    )


SECOND_ORDER = _second_order()
NORMALIZATION = Normalization(
    fuel=32,
    order="leftmost-outermost",
    eta="contract",
    associative_commutative=(),
)
TEMPLATES: tuple[RuleTemplate, ...] = (
    "identity",
    "conjunction-elimination",
    "conjunction-introduction",
    "modus-ponens",
)
SIMILARITY = SimilarityKernelSettings("rbf", 1.0)


class Service:
    def __init__(self, *, proves: bool = True) -> None:
        self.received: Questions | None = None
        self.steps = 0
        self.proves = proves

    @property
    def prover(self) -> str:
        return "test-prover"

    def step_report(self, site: ProofSite, /) -> ServiceReport:
        del site
        questions = 0 if self.received is None else len(self.received.premises)
        return ServiceReport(questions, 0, {}, 0.25, questions, 0.5, None)

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
            proved("checked")
            if self.proves
            else Abstain("no exact proof", cause="gave-up")
            for _ in questions.premises
        )

    def open_step(self, site: ProofSite, /) -> None:
        del site
        self.received = None
        self.steps += 1


def scorer(
    service: Service | None = None,
    *,
    declared: LanguageAssembly = DECLARED,
    order: str = "first",
    templates: tuple[RuleTemplate, ...] = TEMPLATES,
    depth: int = 3,
    similarity: SimilarityKernelSettings = SIMILARITY,
) -> tuple[NeuralProofScorer, Service]:
    held = Service() if service is None else service
    return (
        NeuralProofScorer(
            cast("ProofService", held),
            ProofBudget(calls_per_step=8),
            declared,
            NORMALIZATION,
            declared.codec.choices,
            ComponentId("neural-proof-test"),
            order=order,
            instantiations=8,
            templates=templates,
            depth=depth,
            width=8,
            similarity=similarity,
        ),
        held,
    )


def build_context(service: ProofService, /) -> BuildContext:
    base = plugin_context()
    return BuildContext(
        component=ComponentId("grounding.scorer"),
        policy=base.policy,
        language=DECLARED,
        codec=NeuralCodecView(DECLARED.codec.at(DECLARED.term_sort)),
        vocabulary=base.vocabulary,
        actions=base.actions,
        rules=base.rules,
        categories=(DECLARED.constant("rain"),),
        producers={"normalization": NORMALIZATION, "service": service},
        targets={},
        loader=None,
    )


def test_backward_chaining_is_pathwise_and_rechecks_the_best_discrete_proof() -> None:
    checked, service = scorer()
    premise = DECLARED.read("rain & (rain => wet)")
    goal = DECLARED.constant("wet")

    scores = checked.score(Questions((premise,), (goal,)))
    scores.values.sum().backward()

    evidence = scores.evidence[0]
    assert isinstance(evidence, ProofSearch)
    assert evidence.found
    assert evidence.verified
    assert isinstance(evidence.verdict, Proved)
    assert service.received == Questions((premise,), (goal,))
    assert checked.differentiable
    assert not checked.certified
    assert scores.values.requires_grad
    assert scores.decided.tolist() == [True]
    assert scores.ledger.verified_share == 1.0
    assert any(parameter.grad is not None for parameter in checked.parameters())


def test_search_and_verification_receive_beta_normal_forms() -> None:
    checked, service = scorer()
    redex = DECLARED.read(r"(\x. dog(x))(rex)")
    normal = DECLARED.read("dog(rex)")

    scores = checked.score(Questions((redex,), (normal,)))

    assert scores.evidence[0].found
    assert service.received == Questions((normal,), (normal,))


def test_declared_depth_bounds_backward_chaining() -> None:
    premise = DECLARED.read("dog(rex) & (dog(rex) => rain) & (rain => love(rex)(rex))")
    goal = DECLARED.read("love(rex)(rex)")
    shallow, shallow_service = scorer(depth=1)
    deep, deep_service = scorer(depth=2)

    shallow_scores = shallow.score(Questions((premise,), (goal,)))
    deep_scores = deep.score(Questions((premise,), (goal,)))

    assert not shallow_scores.evidence[0].found
    assert isinstance(shallow_scores.evidence[0].verdict, Abstain)
    assert shallow_scores.evidence[0].verdict.cause == "search-bound"
    assert shallow_service.received is None
    assert deep_scores.evidence[0].found
    assert deep_service.received == Questions((premise,), (goal,))


def test_conjunction_templates_eliminate_and_reintroduce_formulae() -> None:
    checked, service = scorer()
    premise = DECLARED.read("rain & wet")
    goal = DECLARED.read("wet & rain")

    scores = checked.score(Questions((premise,), (goal,)))

    assert float(scores.values[0].detach()) == pytest.approx(1.0)
    assert scores.evidence[0].found
    assert scores.evidence[0].verified
    assert service.received == Questions((premise,), (goal,))


def test_soft_unification_compares_only_type_compatible_choice_rows() -> None:
    checked, service = scorer(Service(proves=False))
    dog = DECLARED.read("dog(rex)")
    cat = DECLARED.read("cat(rex)")

    scores = checked.score(Questions((dog,), (cat,)))
    scores.values.sum().backward()

    evidence = scores.evidence[0]
    assert isinstance(evidence, ProofSearch)
    assert evidence.found
    assert not evidence.verified
    assert isinstance(evidence.verdict, Abstain)
    assert service.received == Questions((dog,), (cat,))
    assert 0.0 <= float(scores.values[0].detach()) <= 1.0
    assert scores.ledger.verified_share == 0.0
    assert any(parameter.grad is not None for parameter in checked.parameters())

    structurally_apart = checked.score(
        Questions((dog,), (DECLARED.read("love(rex)(rex)"),))
    )
    assert float(structurally_apart.values[0].detach()) == 0.0


def test_declared_cosine_kernel_scores_a_type_compatible_unifier() -> None:
    checked, service = scorer(similarity=SimilarityKernelSettings("cosine", 0.5))
    dog = DECLARED.read("dog(rex)")
    cat = DECLARED.read("cat(rex)")

    scores = checked.score(Questions((dog,), (cat,)))

    assert 0.0 <= float(scores.values[0].detach()) <= 1.0
    assert scores.evidence[0].found
    assert service.received == Questions((dog,), (cat,))


@pytest.mark.parametrize(
    "text",
    ["forall x. dog(x)", "meta(dog)", "scope(rain)"],
)
def test_higher_order_questions_are_honestly_out_of_fragment(text: str) -> None:
    checked, service = scorer()
    dog = DECLARED.read("dog(rex)")
    outside = DECLARED.read(text)

    scores = checked.score(Questions((dog,), (outside,)))

    evidence = scores.evidence[0]
    assert isinstance(evidence, ProofSearch)
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert not evidence.found
    assert not evidence.verified
    assert scores.decided.tolist() == [False]
    assert scores.ledger.abstentions["out-of-fragment"] == 1
    assert service.received is None


def test_second_order_pattern_unification_instantiates_a_predicate() -> None:
    checked, service = scorer(declared=SECOND_ORDER, order="second")
    premise = SECOND_ORDER.read("(forall P:(e -> t). P(rex) => P(fido)) & dog(rex)")
    goal = SECOND_ORDER.read("dog(fido)")

    scores = checked.score(Questions((premise,), (goal,)))
    scores.values.sum().backward()

    assert scores.evidence[0].found
    assert scores.evidence[0].verified
    assert service.received == Questions((premise,), (goal,))
    assert scores.values.requires_grad


def test_first_order_neural_proof_refuses_the_same_predicate_binder() -> None:
    checked, service = scorer(declared=SECOND_ORDER, order="first")
    premise = SECOND_ORDER.read("forall P:(e -> t). P(rex) => P(fido)")

    scores = checked.score(Questions((premise,), (SECOND_ORDER.read("dog(fido)"),)))

    evidence = scores.evidence[0]
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert "first" in str(evidence.verdict)
    assert "second" in str(evidence.verdict)
    assert service.received is None


def test_second_order_neural_proof_refuses_a_nonpattern_application() -> None:
    checked, service = scorer(declared=SECOND_ORDER, order="second")
    premise = SECOND_ORDER.read("forall F:(t -> t). F(F(dog(rex)))")

    scores = checked.score(Questions((premise,), (SECOND_ORDER.read("dog(rex)"),)))

    evidence = scores.evidence[0]
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert not evidence.found
    assert service.received is None


def test_conditional_goals_are_outside_the_declared_backward_chaining_fragment() -> (
    None
):
    checked, service = scorer()
    premise = DECLARED.constant("rain")
    conditional = DECLARED.read("rain => wet")

    scores = checked.score(Questions((premise,), (conditional,)))

    evidence = scores.evidence[0]
    assert isinstance(evidence, ProofSearch)
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert service.received is None


def test_undeclared_rule_templates_remove_their_forms_from_the_fragment() -> None:
    checked, service = scorer(templates=("identity",))
    premise = DECLARED.read("rain & (rain => wet)")
    goal = DECLARED.constant("wet")

    scores = checked.score(Questions((premise,), (goal,)))

    evidence = scores.evidence[0]
    assert isinstance(evidence, ProofSearch)
    assert isinstance(evidence.verdict, Abstain)
    assert evidence.verdict.cause == "out-of-fragment"
    assert service.received is None


def test_neural_proof_has_only_the_choice_embedding_parameter_site() -> None:
    checked, _ = scorer()
    (site,) = tuple(checked.site_map())

    assert tuple(name for name, _ in checked.named_parameters()) == (
        "_embedding._table.weight",
    )
    assert site.component == ComponentId("neural-proof-test")
    assert site.key == ("choice-embedding",)


def test_registered_plugin_builds_under_the_run_assigned_component() -> None:
    bound = ProverBound("truth-table", DECLARED.profile.fragment)
    service = ProofService(
        lambda: TruthTableOracle(
            DECLARED.store,
            DECLARED.profile,
            bound=bound,
            atoms=8,
        ),
        profile=DECLARED.profile,
        workers=1,
        batch_size=8,
        timeout=30.0,
    )

    try:
        built = build_component(
            NeuralProofScorerSpec(
                order="first",
                instantiations=8,
                templates=TEMPLATES,
                depth=2,
                width=4,
                similarity=SimilarityKernelSpec(kind="rbf", temperature=1.0),
                verification=ProofBudgetSpec(calls_per_step=4),
            ),
            build_context(service),
        )

        assert isinstance(built, NeuralProofScorer)
        (site,) = tuple(built.site_map())
        assert site.component == ComponentId("grounding.scorer")
    finally:
        service.close()


def test_pairwise_composition_accepts_binary_labels_and_names_three_way_refusal() -> (
    None
):
    checked, _ = scorer()

    PairwiseGrounding(
        DECLARED.store,
        checked,
        negation=DECLARED.constant("not"),
        application=DECLARED.calculus.app,
        labels=RTE.labels,
        label_map=RTE.label_map(CLASSICAL),
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=None,
        reward=None,
    )

    with pytest.raises(ValueError, match=r"entails.*contradicts.*neutral"):
        PairwiseGrounding(
            DECLARED.store,
            checked,
            negation=DECLARED.constant("not"),
            application=DECLARED.calculus.app,
            labels=THREE_WAY.labels,
            label_map=CLASSICAL,
            contradiction="negated-hypothesis",
            composition=Marginal(),
            baseline=None,
            reward=None,
        )


def test_binary_pairwise_grounding_asks_only_the_positive_question() -> None:
    checked, service = scorer()
    grounding = PairwiseGrounding(
        DECLARED.store,
        checked,
        negation=DECLARED.constant("not"),
        application=DECLARED.calculus.app,
        labels=RTE.labels,
        label_map=RTE.label_map(CLASSICAL),
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=None,
        reward=None,
    )

    class Pairs:
        premises = (ReadoutId("premise"),)
        hypotheses = (ReadoutId("hypothesis"),)
        alignments = ((ArgumentRenaming.identity(),),)
        labels = (InferenceLabel.ENTAILS,)

    def forms(term: Term) -> WeightedTerms:
        return WeightedTerms(
            (term,),
            torch.zeros(1, requires_grad=True),
            torch.tensor(-torch.inf),
            Estimate.EXACT,
        )

    premise = DECLARED.read("rain & (rain => wet)")
    goal = DECLARED.constant("wet")
    result = grounding.ground(
        GroundingInput(
            MappingProxyType(
                {
                    ReadoutId("premise"): forms(premise),
                    ReadoutId("hypothesis"): forms(goal),
                }
            ),
            choice_rows=MappingProxyType({}),
            pairs=Pairs(),
            query=GroundingQuery.FULL,
            sentences=None,
            phase=Phase.EVALUATE,
        )
    )

    assert grounding.labels == (
        InferenceLabel.ENTAILS,
        InferenceLabel.NON_ENTAILS,
    )
    assert checked.positive_questions_only
    assert result.ledger.questions == 1
    assert service.received == Questions((premise,), (goal,))
