"""Proof-term pushforwards over chart and lexical-form distributions."""

import math
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, cast

import pytest
import torch

from lofi._frozen import FrozenValue
from lofi.deduce import SpanIndex
from lofi.interpret import (
    MeaningActions,
    MeaningSelection,
    compile_meaning_actions,
)
from lofi.judge import PROOF_LOGICS, ActionSet, ProofCalculus, ProofSearch
from lofi.nn._values import ChartActionScores, LogDistribution
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.nn.loss._approximation import ApproximationBound
from lofi.nn.loss._parser import (
    SPLIT_AXES,
    SplitPosterior,
)
from lofi.nn.readouts import ProofTermMeaning, ProofTermPushforward
from lofi.terms import (
    BindingSite,
    Calculus,
    Node,
    Normalization,
    Operator,
    Signature,
    Term,
    TermStore,
    Vocabulary,
)
from lofi.typecheck import DISTINCT_MEANING_WEIGHTING, Erasure, MeaningWeighting

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.judge import Rule, SchematicSequent

SIGNATURE: Final = Signature(
    "proof-term-readout",
    ("Meaning",),
    (
        Operator("app", "Meaning", arg_sorts=("Meaning", "Meaning")),
        Operator(
            "lam",
            "Meaning",
            arg_sorts=("Meaning",),
            binding=(BindingSite((0,), "Meaning"),),
        ),
    ),
    vocabularies=(Vocabulary("Meaning", ("a", "b", "c", "d")),),
)
STORE: Final = TermStore(SIGNATURE)
LANGUAGE_SIGNATURE: Final = Signature(
    "proof-term-readout-categories",
    ("Cat", "Type"),
    (
        Operator("fwd", "Cat", arg_sorts=("Cat", "Cat")),
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
        "fwd": LANGUAGE_STORE.op(
            LANGUAGE_SIGNATURE.by_name("->"),
            LANGUAGE_STORE.fvar("Type", "a1"),
            LANGUAGE_STORE.fvar("Type", "a0"),
        )
    },
    category_sort="Cat",
    function_type="->",
)
PATTERN_X: Final = Node(
    "#const[Cat]", payload=LANGUAGE_SIGNATURE.vocabulary("Cat").id("x")
)
PATTERN_Y: Final = Node(
    "#const[Cat]", payload=LANGUAGE_SIGNATURE.vocabulary("Cat").id("y")
)
PATTERN_FUNCTION: Final = Node("fwd", (PATTERN_X, PATTERN_X))
NORMALIZATION: Final = Normalization(
    fuel=40,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)


class LambdaCalculus:
    """The proof-calculus channels of the fixture store."""

    @property
    def store(self) -> TermStore:
        return STORE

    @property
    def operators(self) -> Calculus:
        return Calculus("lam", "app")

    def hypothesis(self, index: int, /) -> Term:
        return STORE.fvar("Meaning", f"p{index}")

    def free_variable(self, name: str, /) -> Term:
        return STORE.fvar("Meaning", name)

    def bound_variable(self, index: int, /) -> Term:
        return STORE.bvar("Meaning", index)

    def application(self, function: Term, argument: Term, /) -> Term:
        return STORE.op(SIGNATURE.by_name("app"), function, argument)

    def abstraction(self, variable: str, body: Term, /) -> Term:
        return STORE.bind("lam", variable, body)


CALCULUS: Final[ProofCalculus] = LambdaCalculus()
ACTIONS: Final = ActionSet.derive(("combine",), identity="identity", null="null")


class ProofRuleFixture(NamedTuple):
    """A finite proof schema used to compile public semantic actions."""

    name: str
    arity: int
    proof_schemata: tuple[SchematicSequent, ...]


def compiled_actions(
    schemata: tuple[SchematicSequent, ...] = (),
    *,
    logic: Literal["linear", "intuitionistic"] = "linear",
    supplied: Term | None = None,
) -> MeaningActions:
    """Compile one fixture rule through the public action boundary."""
    rule = cast("Rule[Term]", ProofRuleFixture("combine", 2, schemata))
    return compile_meaning_actions(
        (rule,),
        None if supplied is None else {"combine": supplied},
        erasure=ERASURE,
        calculus=CALCULUS,
        normalization=NORMALIZATION,
        search=ProofSearch(
            contraction=PROOF_LOGICS[logic].contraction,
            weakening=PROOF_LOGICS[logic].weakening,
            max_uses=1,
            max_inhabitants=64,
        ),
        selection=MeaningSelection("all"),
    )


class MeaningFixture:
    """The semantic-action capability consumed by a proof-term readout."""

    @property
    def actions(self) -> MeaningActions:
        action = STORE.op(
            SIGNATURE.by_name("app"),
            CALCULUS.hypothesis(0),
            CALCULUS.hypothesis(1),
        )
        return compiled_actions(supplied=action)

    @property
    def normalization(self) -> Normalization:
        return NORMALIZATION


MEANING: Final[ProofTermMeaning] = MeaningFixture()
LOOSE: Final = ApproximationBound(0.999)


class AlternativeMeaningFixture:
    """Two semantic actions selected for the same grammar rule."""

    @property
    def actions(self) -> MeaningActions:
        return compiled_actions(
            (((PATTERN_FUNCTION, PATTERN_FUNCTION), PATTERN_FUNCTION),)
        )

    @property
    def normalization(self) -> Normalization:
        return NORMALIZATION


class FinalMeaningWeighting(FrozenValue):
    """Give all semantic-alternative mass to the final distinct meaning."""

    __slots__ = ()

    _noun = "test meaning weighting"

    @property
    def name(self) -> str:
        return "final"

    def shares(self, reached: Sequence[Term], /) -> Mapping[Term, float]:
        distinct = tuple(dict.fromkeys(reached))
        return {meaning: float(meaning is distinct[-1]) for meaning in distinct}


ALTERNATIVE_MEANING: Final[ProofTermMeaning] = AlternativeMeaningFixture()
FINAL_WEIGHTING: Final[MeaningWeighting] = FinalMeaningWeighting()


def atom(name: str) -> Term:
    return STORE.fvar("Meaning", name)


class CategoryConditionedMeaningFixture:
    """Two semantic actions selected for distinct erased sequents."""

    @property
    def actions(self) -> MeaningActions:
        return compiled_actions(
            (
                ((PATTERN_X, PATTERN_Y), PATTERN_X),
                ((PATTERN_X, PATTERN_Y), PATTERN_Y),
            ),
            logic="intuitionistic",
        )

    @property
    def normalization(self) -> Normalization:
        return NORMALIZATION


CATEGORY_CONDITIONED_MEANING: Final[ProofTermMeaning] = (
    CategoryConditionedMeaningFixture()
)


def test_a_proof_term_meaning_has_no_second_calculus_authority() -> None:
    assert isinstance(MEANING, ProofTermMeaning)
    assert not hasattr(MEANING, "calculus")


def exact(*terms: Term, logits: torch.Tensor | None = None) -> WeightedTerms:
    weights = (
        torch.zeros(len(terms), dtype=torch.float64)
        if logits is None
        else logits.log_softmax(dim=0)
    )
    return WeightedTerms(
        terms, weights, weights.new_full((), -torch.inf), Estimate.EXACT
    )


def distributions(
    spans: SpanIndex,
    action_logits: torch.Tensor,
    split_logits: torch.Tensor,
) -> tuple[SplitPosterior, ChartActionScores]:
    valid = torch.tensor(spans.split_mask).unsqueeze(0)
    split_scores = split_logits.masked_fill(~valid, -torch.inf)
    split_scores = split_scores - torch.logsumexp(split_scores, dim=-1, keepdim=True)
    split_scores = split_scores.masked_fill(~valid, -torch.inf)
    splits = SplitPosterior(
        spans,
        LogDistribution(
            split_scores,
            SPLIT_AXES,
            valid,
            normalized_over="split",
        ),
    )
    actions = ChartActionScores(
        ACTIONS,
        LogDistribution(
            action_logits.log_softmax(dim=-1).masked_fill(
                ~valid.unsqueeze(-1), -torch.inf
            ),
            (*SPLIT_AXES, "action"),
            valid,
        ),
    )
    return splits, actions


@pytest.mark.parametrize("arity", [1, 2])
def test_a_pushforward_refuses_category_conditioned_actions_without_a_witness(
    arity: int,
) -> None:
    with pytest.raises(
        ValueError,
        match=(
            "requires category-invariant semantic actions; "
            r"category-conditioned rules: \['combine'\]"
        ),
    ):
        ProofTermPushforward(
            ACTIONS,
            CATEGORY_CONDITIONED_MEANING,
            cap=8,
            rule_arities={"combine": arity},
            weighting=DISTINCT_MEANING_WEIGHTING,
            bound=LOOSE,
        )


def test_span_forms_are_beta_normal_proof_terms() -> None:
    spans = SpanIndex(2)
    action_logits = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[-2.0, 4.0, -2.0]]]],
        dtype=torch.float64,
    )
    split_logits = torch.zeros(
        1, spans.num_spans, spans.max_splits, dtype=torch.float64
    )
    splits, actions = distributions(spans, action_logits, split_logits)
    identity = STORE.bind("lam", "x", STORE.fvar("Meaning", "x"))
    lexical = {
        (0, spans.span_id(0, 1)): exact(identity),
        (0, spans.span_id(1, 2)): exact(atom("b")),
    }

    derived = ProofTermPushforward(
        ACTIONS,
        MEANING,
        cap=8,
        rule_arities={"combine": 2},
        weighting=DISTINCT_MEANING_WEIGHTING,
        bound=LOOSE,
    ).derive(lexical, spans=spans, splits=splits, actions=actions.by_action)

    root = derived[0, spans.span_id(0, 2)]
    assert root.terms == (atom("b"),)
    assert root.log_weights.exp().item() == pytest.approx(1.0 / (1.0 + math.exp(-6.0)))


def test_zero_weight_semantic_alternatives_create_no_log_route() -> None:
    spans = SpanIndex(2)
    action_logits = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[-2.0, 4.0, -2.0]]]],
        dtype=torch.float64,
    )
    split_logits = torch.zeros(
        1, spans.num_spans, spans.max_splits, dtype=torch.float64
    )
    splits, actions = distributions(spans, action_logits, split_logits)
    lexical = {
        (0, spans.span_id(0, 1)): exact(atom("a")),
        (0, spans.span_id(1, 2)): exact(atom("b")),
    }

    derived = ProofTermPushforward(
        ACTIONS,
        ALTERNATIVE_MEANING,
        cap=8,
        rule_arities={"combine": 2},
        weighting=FINAL_WEIGHTING,
        bound=LOOSE,
    ).derive(lexical, spans=spans, splits=splits, actions=actions.by_action)

    root = derived[0, spans.span_id(0, 2)]
    expected = CALCULUS.abstraction(
        "x",
        CALCULUS.application(
            atom("b"),
            CALCULUS.application(atom("a"), CALCULUS.free_variable("x")),
        ),
    )
    assert root.terms == (expected,)
    assert bool(root.log_weights.isfinite().all())


def test_pushforward_keeps_gradients_to_lexical_split_and_action_weights() -> None:
    spans = SpanIndex(3)
    lexical_logits = torch.tensor([0.2, -0.4], dtype=torch.float64, requires_grad=True)
    action_logits = torch.zeros(
        1,
        spans.num_spans,
        spans.max_splits,
        len(ACTIONS.actions),
        dtype=torch.float64,
        requires_grad=True,
    )
    split_logits = torch.zeros(
        1,
        spans.num_spans,
        spans.max_splits,
        dtype=torch.float64,
        requires_grad=True,
    )
    splits, actions = distributions(spans, action_logits, split_logits)
    lexical = {
        (0, spans.span_id(0, 1)): exact(atom("a"), atom("d"), logits=lexical_logits),
        (0, spans.span_id(1, 2)): exact(atom("b")),
        (0, spans.span_id(2, 3)): exact(atom("c")),
    }
    pushforward = ProofTermPushforward(
        ACTIONS,
        MEANING,
        cap=8,
        rule_arities={"combine": 2},
        weighting=DISTINCT_MEANING_WEIGHTING,
        bound=LOOSE,
    )

    derived = pushforward.derive(
        lexical,
        spans=spans,
        splits=splits,
        actions=actions.by_action,
    )
    target = STORE.op(
        SIGNATURE.by_name("app"),
        STORE.op(SIGNATURE.by_name("app"), atom("a"), atom("b")),
        atom("c"),
    )
    root = derived[0, spans.span_id(0, 3)]
    loss = -root.log_weights[root.terms.index(target)]
    loss.backward()

    assert lexical_logits.grad is not None
    assert bool(lexical_logits.grad.isfinite().all())
    assert bool(lexical_logits.grad.abs().sum() > 0.0)
    assert action_logits.grad is not None
    assert bool(action_logits.grad.isfinite().all())
    assert bool(action_logits.grad.abs().sum() > 0.0)
    assert split_logits.grad is not None
    assert bool(split_logits.grad.isfinite().all())
    assert bool(split_logits.grad.abs().sum() > 0.0)


def test_the_declared_frontier_caps_internal_terms_and_preserves_residual_mass() -> (
    None
):
    spans = SpanIndex(2)
    action_logits = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[-2.0, 4.0, -2.0]]]],
        dtype=torch.float64,
    )
    split_logits = torch.zeros(
        1, spans.num_spans, spans.max_splits, dtype=torch.float64
    )
    splits, actions = distributions(spans, action_logits, split_logits)
    lexical = {
        (0, spans.span_id(0, 1)): exact(
            atom("a"),
            atom("d"),
            logits=torch.tensor([1.0, 0.0], dtype=torch.float64),
        ),
        (0, spans.span_id(1, 2)): exact(atom("b")),
    }

    derived = ProofTermPushforward(
        ACTIONS,
        MEANING,
        cap=1,
        rule_arities={"combine": 2},
        weighting=DISTINCT_MEANING_WEIGHTING,
        bound=LOOSE,
    ).derive(lexical, spans=spans, splits=splits, actions=actions.by_action)

    root = derived[0, spans.span_id(0, 2)]
    assert root.terms == (STORE.op(SIGNATURE.by_name("app"), atom("a"), atom("b")),)
    assert root.estimate is Estimate.TRUNCATED
    assert root.log_residual.isfinite()
    assert torch.allclose(
        root.log_weights.exp().sum() + root.log_residual.exp(),
        torch.ones((), dtype=torch.float64),
    )


def capped_root(bound: ApproximationBound) -> WeightedTerms:
    """Return the root of a two-word span whose frontier keeps one of two forms."""
    spans = SpanIndex(2)
    action_logits = torch.tensor(
        [[[[0.0, 0.0, 0.0]], [[0.0, 0.0, 0.0]], [[-2.0, 4.0, -2.0]]]],
        dtype=torch.float64,
    )
    split_logits = torch.zeros(
        1, spans.num_spans, spans.max_splits, dtype=torch.float64
    )
    splits, actions = distributions(spans, action_logits, split_logits)
    lexical = {
        (0, spans.span_id(0, 1)): exact(
            atom("a"),
            atom("d"),
            logits=torch.tensor([1.0, 0.0], dtype=torch.float64),
        ),
        (0, spans.span_id(1, 2)): exact(atom("b")),
    }

    derived = ProofTermPushforward(
        ACTIONS,
        MEANING,
        cap=1,
        rule_arities={"combine": 2},
        weighting=DISTINCT_MEANING_WEIGHTING,
        bound=bound,
    ).derive(lexical, spans=spans, splits=splits, actions=actions.by_action)

    return derived[0, spans.span_id(0, 2)]


def test_a_truncated_derived_cell_is_checked_against_the_derived_bound() -> None:
    residual = float(capped_root(LOOSE).log_residual.exp())

    assert 0.0 < residual < LOOSE.residual_mass
    with pytest.raises(
        ValueError, match=r"proof-term span \(0, 2\): measured residual mass"
    ):
        capped_root(ApproximationBound(residual / 2.0))


def test_the_derived_bound_does_not_recheck_lexical_leaves() -> None:
    spans = SpanIndex(2)
    zeros = torch.zeros(1, spans.num_spans, spans.max_splits, dtype=torch.float64)
    splits, actions = distributions(
        spans, torch.zeros(*zeros.shape, len(ACTIONS.actions)), zeros
    )
    weights = torch.tensor([math.log(0.5)], dtype=torch.float64)
    leaf = WeightedTerms(
        (atom("a"),), weights, weights.new_tensor(math.log(0.5)), Estimate.TRUNCATED
    )

    derived = ProofTermPushforward(
        ACTIONS,
        MEANING,
        cap=8,
        rule_arities={"combine": 2},
        weighting=DISTINCT_MEANING_WEIGHTING,
        bound=ApproximationBound(0.1),
    ).derive(
        {(0, spans.span_id(0, 1)): leaf},
        spans=spans,
        splits=splits,
        actions=actions.by_action,
    )

    assert derived == {(0, spans.span_id(0, 1)): leaf}
