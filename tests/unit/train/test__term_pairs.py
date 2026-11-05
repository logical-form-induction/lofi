"""Tests for generated term-space pair adapters."""

from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast

import pytest
from tests.support.term_reference import CALCULUS, formulas

from lofi.data import (
    PairDistribution,
    PairSampleReport,
    SampledPair,
)
from lofi.data.synthetic import LambdaCalculus, RecursiveTypeSampler
from lofi.interpret import (
    MeaningActions,
    MeaningSelection,
    compile_meaning_actions,
)
from lofi.judge import (
    PROOF_LOGICS,
    Action,
    ActionSet,
    ProofSearch,
)
from lofi.spec import CompositionSpec
from lofi.spec._rule_laws import LawGrammar
from lofi.terms import (
    Node,
    Normalization,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)
from lofi.train._term_pairs import MeaningPairSampler, MorphismPairSampler
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from lofi.judge import Rule, SchematicSequent

STORE: Final = TermStore(formulas())
LANGUAGE_SIGNATURE: Final = Signature(
    "semantic-action-categories",
    ("Cat", "Type"),
    (
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("bwd", "Cat", arg_sorts=("Cat", "Cat")),
        Operator("->", "Type", arg_sorts=("Type", "Type")),
    ),
    vocabularies=(
        Vocabulary("Cat", ("x", "y")),
        Vocabulary("Type", ("tx", "ty")),
    ),
    open_sorts=("Type",),
)
LANGUAGE_STORE: Final = TermStore(LANGUAGE_SIGNATURE)
TYPE_X: Final = LANGUAGE_STORE.const("Type", "tx")
TYPE_Y: Final = LANGUAGE_STORE.const("Type", "ty")
ERASURE: Final = Erasure(
    LANGUAGE_STORE,
    {"x": TYPE_X, "y": TYPE_Y},
    {
        name: LANGUAGE_STORE.op(
            LANGUAGE_SIGNATURE.by_name("->"),
            LANGUAGE_STORE.fvar("Type", "a1"),
            LANGUAGE_STORE.fvar("Type", "a0"),
        )
        for name in ("fwd", "bwd")
    },
    category_sort="Cat",
    function_type="->",
)
PATTERN_X: Final = Node(
    "#const[Cat]", payload=LANGUAGE_SIGNATURE.vocabulary("Cat").id("x")
)
PATTERN_FUNCTION: Final = Node("fwd", (PATTERN_X, PATTERN_X))
SOURCE: Final = STORE.fvar("Lf", "source")
TARGET: Final = STORE.fvar("Lf", "target")
IDENTITY: Final = Action("identity")
NORMALIZATION: Final = Normalization(
    fuel=8,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)


class RuleFixture(NamedTuple):
    """The semantic sampler's rule-name and arity capability."""

    name: str
    arity: int


class ProofRuleFixture(NamedTuple):
    """A finite proof schema used to compile public semantic actions."""

    name: str
    arity: int
    proof_schemata: tuple[SchematicSequent, ...]


LAW_GRAMMAR: Final = LawGrammar("direct")


def compiled_actions(
    schemata: tuple[SchematicSequent, ...] = (),
    *,
    logic: Literal["linear", "intuitionistic"] = "linear",
    supplied: Term | None = None,
) -> MeaningActions:
    """Compile one fixture rule through the public action boundary."""
    rule = cast("Rule[Term]", ProofRuleFixture("combine", 2, schemata))
    calculus = LambdaCalculus(STORE, CALCULUS)
    return compile_meaning_actions(
        (rule,),
        None if supplied is None else {"combine": supplied},
        erasure=ERASURE,
        calculus=calculus,
        normalization=NORMALIZATION,
        search=ProofSearch(
            contraction=PROOF_LOGICS[logic].contraction,
            weakening=PROOF_LOGICS[logic].weakening,
            max_uses=1,
            max_inhabitants=64,
        ),
        selection=MeaningSelection("all"),
    )


class FixedPairSampler:
    """A one-pair source for exercising a term morphism."""

    def sample(
        self,
        count: int,
        distribution: PairDistribution,
        /,
        *,
        seed: int,
    ) -> tuple[tuple[SampledPair[Term, Action], ...], PairSampleReport]:
        del distribution, seed
        pair = SampledPair((SOURCE,), SOURCE, (IDENTITY,))

        return (pair,) * count, PairSampleReport(count, {1: count})


def test_a_morphism_maps_every_term_and_preserves_actions_and_counts() -> None:
    sampler = MorphismPairSampler(FixedPairSampler(), lambda _term: TARGET)

    pairs, report = sampler.sample(2, PairDistribution(max_depth=1), seed=7)

    assert pairs == (
        SampledPair((TARGET,), TARGET, (IDENTITY,)),
        SampledPair((TARGET,), TARGET, (IDENTITY,)),
    )
    assert report == PairSampleReport(2, {1: 2})


def test_a_meaning_sampler_with_no_composing_action_emits_an_empty_draw() -> None:
    types = RecursiveTypeSampler(STORE, max_size=2, sort="Lf")
    actions = ActionSet((IDENTITY,), IDENTITY)
    sampler = MeaningPairSampler(
        types,
        actions,
        compile_meaning_actions(
            (),
            erasure=ERASURE,
            calculus=LambdaCalculus(STORE, CALCULUS),
            normalization=NORMALIZATION,
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
            selection=MeaningSelection("unique"),
        ),
        (),
        normalization=NORMALIZATION,
    )

    pairs, report = sampler.sample(0, PairDistribution(max_depth=2), seed=13)

    assert pairs == ()
    assert report == PairSampleReport(0, {})


def _unbounded_composition() -> Rule[Term]:
    """Return the witness-dependent generalized-composition rule."""
    return LAW_GRAMMAR.rule(
        CompositionSpec(view="fwd", order=None, direction="unrestricted")
    )


def _dynamic_actions(rule: Rule[Term]) -> MeaningActions:
    """Compile one witness-dependent rule through the meaning boundary."""
    return compile_meaning_actions(
        (rule,),
        erasure=ERASURE,
        calculus=LambdaCalculus(STORE, CALCULUS),
        normalization=NORMALIZATION,
        search=ProofSearch(
            contraction=True, weakening=False, max_uses=1, max_inhabitants=64
        ),
        selection=MeaningSelection("all"),
    )


def test_a_meaning_sampler_refuses_category_conditioned_actions() -> None:
    types = RecursiveTypeSampler(STORE, max_size=1, sort="Lf")
    rule = _unbounded_composition()
    identity = Action("identity")
    compose = Action("compose", rule.name)
    actions = ActionSet((identity, compose), identity)
    meanings = _dynamic_actions(rule)
    with pytest.raises(
        ValueError,
        match="semantic pair supervision requires category-invariant actions",
    ) as raised:
        MeaningPairSampler(
            types,
            actions,
            meanings,
            (rule,),
            normalization=NORMALIZATION,
        )

    assert rule.name in str(raised.value)


def test_a_meaning_sampler_emits_pairs_from_every_selected_action_term() -> None:
    types = RecursiveTypeSampler(STORE, max_size=2, sort="Lf")
    identity = Action("identity")
    combine = Action("combine", "combine")
    actions = ActionSet((identity, combine), identity)
    meanings = compiled_actions(
        (((PATTERN_FUNCTION, PATTERN_FUNCTION), PATTERN_FUNCTION),)
    )
    sampler = MeaningPairSampler(
        types,
        actions,
        meanings,
        (RuleFixture("combine", 2),),
        normalization=NORMALIZATION,
    )

    pairs, _ = sampler.sample(64, PairDistribution(max_depth=2), seed=23)

    assert all(pair.actions == (combine,) for pair in pairs)
    assert any(pair.output is pair.inputs[0] for pair in pairs)
    assert any(pair.output is pair.inputs[1] for pair in pairs)


def test_convergent_selected_terms_do_not_duplicate_an_action_label() -> None:
    types = RecursiveTypeSampler(STORE, max_size=1, sort="Lf")
    identity = Action("identity")
    combine = Action("combine", "combine")
    actions = ActionSet((identity, combine), identity)
    meanings = compiled_actions(
        (((PATTERN_FUNCTION, PATTERN_FUNCTION), PATTERN_FUNCTION),)
    )
    sampler = MeaningPairSampler(
        types,
        actions,
        meanings,
        (RuleFixture("combine", 2),),
        normalization=NORMALIZATION,
    )

    pairs, _ = sampler.sample(64, PairDistribution(max_depth=4), seed=29)

    assert all(pair.actions == (combine,) for pair in pairs)
    assert any(pair.inputs[0] is pair.inputs[1] for pair in pairs)
