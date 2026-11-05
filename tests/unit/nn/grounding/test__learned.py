"""Tests for structures fitted independently of inference labels."""

from types import MappingProxyType

import pytest
import torch

from lofi._types import ReadoutId
from lofi.lf import CLASSICAL, THREE_WAY, ArgumentRenaming, InferenceLabel
from lofi.lf.models import (
    CRISP,
    EXTREME,
    GOEDEL,
    PRODUCT,
    RESIDUUM,
    TNORM_QUANTIFIER,
    Arithmetic,
    FuzzyAlgebra,
    TruthAlgebra,
)
from lofi.lf.provers import Questions
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.grounding import (
    GroundingInput,
    GroundingQuery,
    LearnedStructureSource,
    Marginal,
    ModelCheckingScorer,
    PairwiseGrounding,
    Phase,
)
from lofi.spec.language import (
    ConstantSpec,
    LanguageAssembly,
    LanguageIso,
    registered_language,
)


def language() -> LanguageAssembly:
    declaration = registered_language("propositional")
    return LanguageIso().forward(
        declaration.with_(
            constants=(
                *declaration.constants,
                ConstantSpec(name="p", type="t"),
                ConstantSpec(name="q", type="t"),
            ),
            logic=declaration.logic.with_(postulates=("p", "q")),
        )
    )


def algebra() -> TruthAlgebra:
    return PRODUCT.build(
        implication=RESIDUUM,
        quantifier=EXTREME,
        equality=CRISP,
        margin=0.0,
    )


def source() -> LearnedStructureSource:
    return LearnedStructureSource(
        language(),
        algebra(),
        structures=2,
        carriers={"e": 2, "v": 1},
        full_function_space_cap=16,
        henkin_pool=4,
        denotations="from-vocabulary",
        seed=11,
        initialization_scale=0.25,
        theory_floor=0.0,
    )


def test_only_the_satisfiability_term_reaches_learned_denotations() -> None:
    fitted = source()
    parameters = tuple(fitted.parameters())
    auxiliary = fitted.auxiliary()

    auxiliary.backward()

    assert fitted.trainable
    assert auxiliary.item() > 0.0
    assert all(parameter.grad is not None for parameter in parameters)
    assert any(
        bool((parameter.grad != 0.0).any())
        for parameter in parameters
        if parameter.grad is not None
    )

    for parameter in parameters:
        parameter.grad = None

    (0.0 * fitted.auxiliary()).backward()

    assert all(
        parameter.grad is not None and bool((parameter.grad == 0.0).all())
        for parameter in parameters
    )


def test_conditional_scores_read_detached_structures() -> None:
    declared = language()
    fitted = LearnedStructureSource(
        declared,
        algebra(),
        structures=2,
        carriers={"e": 2, "v": 1},
        full_function_space_cap=16,
        henkin_pool=4,
        denotations="from-vocabulary",
        seed=11,
        initialization_scale=0.25,
        theory_floor=0.0,
    )
    checked = ModelCheckingScorer(
        declared,
        fitted,
        algebra(),
        truth_threshold=0.5,
        abstain_band=0.0,
        premise_floor=0.5,
        unsatisfied_premise="undecided",
    )
    scores = checked.score(
        Questions((declared.constant("p"),), (declared.constant("q"),))
    )

    assert not scores.values.requires_grad
    assert checked.auxiliary() is not None
    assert tuple(map(id, checked.parameters())) == tuple(map(id, fitted.parameters()))


def test_pairwise_grounding_publishes_and_registers_the_source_auxiliary() -> None:
    declared = language()
    fitted = LearnedStructureSource(
        declared,
        algebra(),
        structures=2,
        carriers={"e": 2, "v": 1},
        full_function_space_cap=16,
        henkin_pool=4,
        denotations="from-vocabulary",
        seed=11,
        initialization_scale=0.25,
        theory_floor=0.0,
    )
    checked = ModelCheckingScorer(
        declared,
        fitted,
        algebra(),
        truth_threshold=0.5,
        abstain_band=0.0,
        premise_floor=0.5,
        unsatisfied_premise="undecided",
    )
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

    class Pairs:
        premises = (ReadoutId("premise"),)
        hypotheses = (ReadoutId("hypothesis"),)
        alignments = ((ArgumentRenaming.identity(),),)
        labels = (InferenceLabel.ENTAILS,)

    p = declared.constant("p")
    q = declared.constant("q")
    grounded = grounding.ground(
        GroundingInput(
            MappingProxyType(
                {
                    ReadoutId("premise"): WeightedTerms(
                        (p,),
                        torch.zeros(1, requires_grad=True),
                        torch.tensor(-torch.inf),
                        Estimate.EXACT,
                    ),
                    ReadoutId("hypothesis"): WeightedTerms(
                        (q,),
                        torch.zeros(1, requires_grad=True),
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

    assert grounded.auxiliary is not None
    assert grounded.auxiliary.requires_grad
    assert tuple(map(id, grounding.parameters())) == tuple(map(id, fitted.parameters()))


def test_theory_fitting_reduces_dissatisfaction_without_inference_labels() -> None:
    fitted = source()
    optimizer = torch.optim.SGD(fitted.parameters(), lr=0.25)
    before = float(fitted.auxiliary().detach())

    for _ in range(5):
        optimizer.zero_grad()
        loss = fitted.auxiliary()
        loss.backward()
        optimizer.step()

    assert float(fitted.auxiliary().detach()) < before


class _RenamedMinimum:
    """The minimum t-norm registered under a name no branch could know."""

    @property
    def name(self) -> str:
        return "renamed-minimum"

    def conjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.minimum(left, right)

    def disjunction[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.maximum(left, right)

    def residuum[T](self, arithmetic: Arithmetic[T], left: T, right: T, /) -> T:
        return arithmetic.where_at_most(
            left, right, arithmetic.constant(1.0, left), right
        )


def test_learned_connectives_read_the_declared_tnorm_operations() -> None:
    declaration = registered_language("propositional")
    declared = LanguageIso().forward(
        declaration.with_(
            constants=(
                *declaration.constants,
                ConstantSpec(name="p", type="t"),
                ConstantSpec(name="q", type="t"),
            ),
            logic=declaration.logic.with_(postulates=("p & q", "p => q")),
        )
    )
    renamed = FuzzyAlgebra(
        _RenamedMinimum(),
        implication=RESIDUUM,
        quantifier=TNORM_QUANTIFIER,
        equality=CRISP,
        margin=0.0,
    )
    named = GOEDEL.build(
        implication=RESIDUUM, quantifier=TNORM_QUANTIFIER, equality=CRISP, margin=0.0
    )

    def fitted(truths: TruthAlgebra) -> float:
        return (
            LearnedStructureSource(
                declared,
                truths,
                structures=2,
                carriers={"e": 2, "v": 1},
                full_function_space_cap=16,
                henkin_pool=4,
                denotations="from-vocabulary",
                seed=11,
                initialization_scale=0.25,
                theory_floor=0.0,
            )
            .auxiliary()
            .item()
        )

    assert fitted(renamed) == fitted(named)


def test_scoring_reads_only_structures_whose_theory_meets_the_floor() -> None:
    def fitted(floor: float) -> LearnedStructureSource:
        return LearnedStructureSource(
            language(),
            algebra(),
            structures=2,
            carriers={"e": 2, "v": 1},
            full_function_space_cap=16,
            henkin_pool=4,
            denotations="from-vocabulary",
            seed=11,
            initialization_scale=0.25,
            theory_floor=floor,
        )

    assert len(fitted(0.0).structures()) == 2
    assert fitted(1.0).structures() == ()


def test_a_theory_floor_outside_the_unit_interval_is_refused() -> None:
    with pytest.raises(ValueError, match="unit interval"):
        LearnedStructureSource(
            language(),
            algebra(),
            structures=1,
            carriers={"e": 2, "v": 1},
            full_function_space_cap=16,
            henkin_pool=4,
            denotations="from-vocabulary",
            seed=11,
            initialization_scale=0.25,
            theory_floor=1.5,
        )


def test_denotation_logits_start_at_the_declared_scale() -> None:
    def spread(scale: float) -> float:
        fitted = LearnedStructureSource(
            language(),
            algebra(),
            structures=4,
            carriers={"e": 2, "v": 1},
            full_function_space_cap=16,
            henkin_pool=4,
            denotations="free",
            seed=11,
            initialization_scale=scale,
            theory_floor=0.0,
        )
        return float(torch.cat(tuple(fitted.parameters())).detach().abs().max())

    assert spread(2.0) == pytest.approx(8.0 * spread(0.25))
