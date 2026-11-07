from types import MappingProxyType
from typing import TYPE_CHECKING

import torch
from tests.language import apply, language

from lofi._types import ReadoutId
from lofi.lf import CLASSICAL, THREE_WAY, ArgumentRenaming, InferenceLabel
from lofi.lf.models import (
    CRISP,
    EXTREME,
    PRODUCT,
    RESIDUUM,
    Frame,
    FunctionValue,
    Structure,
)
from lofi.lf.provers import Questions
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import (
    GroundingInput,
    GroundingQuery,
    Marginal,
    ModelCheckingScorer,
    PairwiseGrounding,
    Phase,
)

if TYPE_CHECKING:
    from lofi.spec.language import LanguageAssembly


class Source:
    def __init__(
        self,
        structures: tuple[Structure, ...],
        *,
        evidence_label: str = "conditional satisfaction",
    ) -> None:
        self._structures = structures
        self._evidence_label = evidence_label
        self.steps = 0

    @property
    def name(self) -> str:
        return "fixture"

    @property
    def evidence_label(self) -> str:
        return self._evidence_label

    @property
    def trainable(self) -> bool:
        return False

    def structures(self) -> tuple[Structure, ...]:
        return self._structures

    def open_step(self) -> None:
        self.steps += 1


def fixture() -> tuple[LanguageAssembly, Source]:
    declared = language(("p", "q", "r"))
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 1, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=2,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)

    def structure(name: str, *, p: bool, q: bool) -> Structure:
        return Structure(
            name,
            frame,
            {
                vocabulary.id("p"): p,
                vocabulary.id("q"): q,
                vocabulary.id("r"): False,
            },
        )

    return declared, Source(
        (
            structure("both", p=True, q=True),
            structure("premise", p=True, q=False),
            structure("goal", p=False, q=True),
        )
    )


def scorer(
    declared: LanguageAssembly,
    source: Source,
) -> ModelCheckingScorer:
    return ModelCheckingScorer(
        declared,
        source,
        PRODUCT.build(
            implication=RESIDUUM,
            quantifier=EXTREME,
            equality=CRISP,
            margin=0.0,
        ),
        truth_threshold=0.75,
        abstain_band=0.05,
        premise_floor=0.75,
        unsatisfied_premise="undecided",
    )


def test_model_checking_scores_conditional_satisfaction_without_a_verdict() -> None:
    declared, source = fixture()
    checked = scorer(declared, source)
    p = declared.constant("p")
    q = declared.constant("q")
    scores = checked.score(Questions((p,), (q,)))

    assert scores.values.tolist() == [0.5]
    assert scores.decided.tolist() == [True]
    assert scores.evidence[0].structures == 3
    assert scores.evidence[0].premise_holding == 2
    assert scores.evidence[0].both_holding == 1
    assert checked.scorer_kind == "model-checking"
    assert checked.source_identity == "fixture"
    assert not checked.certified
    assert not checked.differentiable
    assert checked.proof_metadata is None


def test_a_false_everywhere_premise_is_undecided_and_a_tautology_is_high() -> None:
    declared, source = fixture()
    checked = scorer(declared, source)
    p = declared.constant("p")
    r = declared.constant("r")
    scores = checked.score(Questions((r, p), (p, p)))

    assert scores.values.tolist() == [0.0, 1.0]
    assert scores.decided.tolist() == [False, True]


def test_a_zero_width_abstention_band_decides_a_threshold_score() -> None:
    declared, source = fixture()
    checked = ModelCheckingScorer(
        declared,
        source,
        PRODUCT.build(
            implication=RESIDUUM,
            quantifier=EXTREME,
            equality=CRISP,
            margin=0.0,
        ),
        truth_threshold=0.5,
        abstain_band=0.0,
        premise_floor=0.75,
        unsatisfied_premise="undecided",
    )
    scores = checked.score(
        Questions((declared.constant("p"),), (declared.constant("q"),))
    )

    assert scores.values.tolist() == [0.5]
    assert scores.decided.tolist() == [True]


def test_an_unknown_intended_model_fact_leaves_the_question_undecided() -> None:
    declared = language(
        ("alice", "bob", "human"),
        types={"alice": "e", "bob": "e", "human": "e -> t"},
    )
    frame = Frame(
        declared.store,
        function_type=declared.function_type,
        truth_type="t",
        carriers={"e": 2, "v": 1},
        full_function_space_cap=4,
        henkin_pool=2,
        seed=2,
    )
    vocabulary = declared.store.signature.vocabulary(declared.term_sort)
    entities = frame.domain(declared.type_of(declared.constant("alice"))).values
    structure = Structure(
        "partial",
        frame,
        {
            vocabulary.id("alice"): entities[0],
            vocabulary.id("bob"): entities[1],
            vocabulary.id("human"): FunctionValue(
                entities,
                (True, False),
                (True, False),
            ),
        },
    )
    checked = scorer(
        declared,
        Source((structure,), evidence_label="truth in an intended model"),
    )
    human = declared.constant("human")
    questions = Questions(
        (apply(declared, human, declared.constant("alice")),),
        (apply(declared, human, declared.constant("bob")),),
    )

    scores = checked.score(questions)

    assert scores.values.tolist() == [0.0]
    assert scores.decided.tolist() == [False]
    assert scores.ledger.evidence == {"truth in an intended model": 1}


def test_question_order_does_not_change_model_checking_scores() -> None:
    declared, source = fixture()
    checked = scorer(declared, source)
    p = declared.constant("p")
    q = declared.constant("q")
    forward = checked.score(Questions((p, q), (q, p)))
    reverse = checked.score(Questions((q, p), (p, q)))

    assert forward.values.tolist() == list(reversed(reverse.values.tolist()))

    checked.open_step()
    assert source.steps == 1


def test_the_evaluation_relaxation_reads_the_same_conditional_score() -> None:
    declared, source = fixture()
    checked = scorer(declared, source)
    p = declared.constant("p")
    q = declared.constant("q")

    assert checked.relaxation(p, q) == 0.5


def test_the_ledger_names_truth_in_an_intended_model_without_certifying_it() -> None:
    declared, sampled = fixture()
    source = Source(
        sampled.structures(),
        evidence_label="truth in an intended model",
    )
    checked = scorer(declared, source)
    p = declared.constant("p")
    q = declared.constant("q")
    scores = checked.score(Questions((p,), (q,)))

    assert scores.ledger.evidence == {"truth in an intended model": 1}
    assert not checked.certified


def test_pairwise_model_checking_backpropagates_only_to_form_weights() -> None:
    declared, source = fixture()
    checked = scorer(declared, source)
    p = declared.constant("p")
    q = declared.constant("q")
    premise_weights = torch.tensor([-0.2, -1.7], requires_grad=True)
    hypothesis_weights = torch.zeros(1, requires_grad=True)

    class Pairs:
        premises = (ReadoutId("premise"),)
        hypotheses = (ReadoutId("hypothesis"),)
        alignments = ((ArgumentRenaming.identity(),),)
        labels = (InferenceLabel.ENTAILS,)

    grounding = PairwiseGrounding(
        declared.store,
        checked,
        negation=declared.constant("not"),
        application=declared.calculus.app,
        labels=THREE_WAY.labels,
        label_map=CLASSICAL,
        contradiction="negated-hypothesis",
        composition=Marginal(),
        baseline=None,
        reward=None,
    )
    grounded = grounding.ground(
        GroundingInput(
            MappingProxyType(
                {
                    ReadoutId("premise"): WeightedTerms(
                        (p, q),
                        premise_weights,
                        torch.tensor(-torch.inf),
                        Estimate.EXACT,
                    ),
                    ReadoutId("hypothesis"): WeightedTerms(
                        (q,),
                        hypothesis_weights,
                        torch.tensor(-torch.inf),
                        Estimate.EXACT,
                    ),
                }
            ),
            choice_rows=MappingProxyType({}),
            pairs=Pairs(),
            query=GroundingQuery.FULL,
            sentences=None,
            phase=Phase.TRAIN,
        )
    )
    grounded.log_probabilities.tensor[0, 0].backward()

    assert premise_weights.grad is not None
    assert torch.count_nonzero(premise_weights.grad) > 0
    assert all(
        not isinstance(value, torch.Tensor)
        for structure in source.structures()
        for value in structure.denotations.values()
    )
