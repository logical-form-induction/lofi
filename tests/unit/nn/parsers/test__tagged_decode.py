import re
from math import exp, inf
from typing import TYPE_CHECKING, Final, cast, override

import pytest
import torch
from tests.support.rules import ccg_signature, declared_rules

from lofi._native import SpanDependent, Term
from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    KBest,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    SpanIndex,
    parse,
)
from lofi.deduce import LogInside as SymbolicLogInside
from lofi.judge import (
    CompositionRole,
    EisnerNormalForm,
    HockenmaierBiskNormalForm,
    NoNormalForm,
    NormalFormRole,
    RuleSet,
)
from lofi.lexicon import Entry
from lofi.nn.parsers import (
    LogSumExpReduction,
    MaximumReduction,
    ScoreChannels,
    TaggedDecoder,
    TaggedInsideKernel,
)
from lofi.spec import (
    ConstantPatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    TermPatternSpec,
)
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import Cell, Derivation, Ranked
    from lofi.judge import (
        Direction,
        NormalForm,
        PatternRule,
        Rule,
        RuleProbe,
        RuleRestriction,
        SchematicSequent,
        Slot,
    )
    from lofi.lexicon import Clamp
    from lofi.nn.parsers import TaggedTree
    from lofi.terms import Functors, TermMatching, TermRuleKernel

FORWARD: Final = "/"
BACKWARD: Final = "\\"

#: Absolute tolerance for log weights.
TOLERANCE = 1e-9

#: Number of derivations compared against the symbolic chart.
KEPT = 3

#: How many unary steps a derivation may take within one cell.
DEPTH = 2

#: The greatest composition order the normal form admits.
ORDER = 1

#: Seed under which the heaviest derivation is not the one of least risk.
DIVERGENT_SEED = 12

CATEGORY_NAMES: Final = ("NP", "T", "U", "W", "PP")
LIFTS: Final = (("NP", "T"), ("T", "U"), ("U", "W"))
CATEGORY_RULES: Final = declared_rules(
    ccg_signature(*CATEGORY_NAMES, name="tagged-decoder-tests"),
    RuleSetSpec(
        name="tagged-decoder-tests",
        rules=(
            PatternRuleSpec(
                name="join",
                premises=(
                    TermPatternSpec(name="category", sort="Cat"),
                    TermPatternSpec(name="category", sort="Cat"),
                ),
                conclusion=TermPatternSpec(name="category", sort="Cat"),
            ),
            *(
                PatternRuleSpec(
                    name=f"lift-{source}-{output}",
                    premises=(ConstantPatternSpec(name=source),),
                    conclusion=ConstantPatternSpec(name=output),
                )
                for source, output in LIFTS
            ),
        ),
    ),
)
COMPILED: Final = {
    rule.name: cast("PatternRule", rule) for rule in CATEGORY_RULES.rule_set
}
CATEGORY_STORE: Final = CATEGORY_RULES.context.store
CATEGORY_BY_NAME: Final = {
    name: CATEGORY_RULES.context.syntax.parse(CATEGORY_STORE, name)
    for name in CATEGORY_NAMES
}
FUNCTORS: Final = CATEGORY_RULES.context.functors[FORWARD][0]
CATEGORY_NAME_OF: Final = {term: name for name, term in CATEGORY_BY_NAME.items()}
ARGUMENT: Final = CATEGORY_BY_NAME["NP"]
FUNCTOR: Final = CATEGORY_BY_NAME["T"]
CATEGORIES: Final = (ARGUMENT, FUNCTOR)

type _Shape = tuple[Cell, Term, str, str | None, tuple["_Shape", ...]]
"""A derivation's span, category, tag and rule name, nested as its premises are."""

type Columns = tuple[dict[str, int], dict[str, int]]
"""The binary and unary score columns of each rule, by name."""

type Edge = tuple["Rule[Term]", "Cell", int | None]
"""One symbolic rule application: its rule, its span and its cut point."""


class ToyRule:
    """A binary rule joining two copies of one category."""

    def __init__(self, name: str, /, *, slash: str, order: int) -> None:
        self._name = name
        self._slash = slash
        self._order = order
        self._functors = FUNCTORS
        self._lowering = COMPILED["join"].lowering
        self._reachability_schemata = COMPILED["join"].reachability_schemata

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return 2

    @property
    def directions(self) -> frozenset[Direction]:
        return frozenset({"forward", "check"})

    @property
    def spine_order(self) -> int:
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def forward_is_complete(self) -> bool:
        return True

    @property
    def matching(self) -> TermMatching:
        return MATCHING

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        return self._lowering

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return self._reachability_schemata

    @property
    def normal_form_role(self) -> NormalFormRole:
        direction = "forward" if self._slash == FORWARD else "backward"
        principal = 0 if direction == "forward" else 1
        return NormalFormRole(
            composition=CompositionRole(
                direction, principal_slot=principal, order=self._order
            )
        )

    @property
    def functors(self) -> Functors[Term]:
        return self._functors

    @property
    def order(self) -> int:
        return self._order

    @property
    def slash(self) -> str:
        return self._slash

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if len(inputs) == 2 and inputs[0] == inputs[1]:
            yield inputs[0]

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return len(inputs) == 2 and all(item == output for item in inputs)

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        del output, known
        return
        yield

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "composition", "direction": "unrestricted"}


class RaisingRule(ToyRule):
    """A unary rule lifting the argument category to the functor category."""

    @property
    @override
    def arity(self) -> int:
        return 1

    @property
    @override
    def residual_slots(self) -> frozenset[int]:
        return frozenset({0})

    @property
    @override
    def lowering(self) -> TermRuleKernel:
        return COMPILED["lift-NP-T"].lowering

    @property
    @override
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return COMPILED["lift-NP-T"].reachability_schemata

    @property
    @override
    def normal_form_role(self) -> NormalFormRole:
        direction = "forward" if self.slash == FORWARD else "backward"
        return NormalFormRole(raising=direction)

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if inputs == (ARGUMENT,):
            yield FUNCTOR

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return inputs == (ARGUMENT,) and output == FUNCTOR

    @override
    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "type-raising"}


class LiftingRule(ToyRule):
    """A unary rule lifting one named category to another."""

    def __init__(self, name: str, /, *, source: Term, output: Term) -> None:
        super().__init__(name, slash=FORWARD, order=0)
        self._source = source
        self._output = output
        compiled = COMPILED[
            f"lift-{CATEGORY_NAME_OF[source]}-{CATEGORY_NAME_OF[output]}"
        ]
        self._lowering = compiled.lowering
        self._reachability_schemata = compiled.reachability_schemata

    @property
    @override
    def arity(self) -> int:
        return 1

    @property
    @override
    def residual_slots(self) -> frozenset[int]:
        return frozenset({0})

    @property
    @override
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole(raising="forward")

    @override
    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        if inputs == (self._source,):
            yield self._output

    @override
    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return inputs == (self._source,) and output == self._output

    @override
    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": "type-raising"}


MATCHING: Final = cast("TermMatching", CATEGORY_RULES.rule_set.matching)
FORWARD_COMPOSITION: Final = ToyRule("forward-composition", slash=FORWARD, order=ORDER)
BACKWARD_COMPOSITION: Final = ToyRule(
    "backward-composition", slash=BACKWARD, order=ORDER
)
RAISING: Final = RaisingRule("type-raising", slash=FORWARD, order=0)
WITH_RAISING: Final = (RAISING, FORWARD_COMPOSITION, BACKWARD_COMPOSITION)
LIFTED: Final = CATEGORY_BY_NAME["U"]
CHAINED_CATEGORIES: Final = (ARGUMENT, FUNCTOR, LIFTED)
SECOND_LIFT: Final = LiftingRule("second-lift", source=FUNCTOR, output=LIFTED)
CHAINED: Final = (RAISING, SECOND_LIFT, FORWARD_COMPOSITION)


DEEPEST: Final = CATEGORY_BY_NAME["W"]
LADDER: Final = (ARGUMENT, FUNCTOR, LIFTED, DEEPEST)
LADDER_RULES: Final = (
    LiftingRule("lift-one", source=ARGUMENT, output=FUNCTOR),
    LiftingRule("lift-two", source=FUNCTOR, output=LIFTED),
    LiftingRule("lift-three", source=LIFTED, output=DEEPEST),
    FORWARD_COMPOSITION,
)


def ladder_kernel() -> TaggedInsideKernel[Term]:
    """Return a kernel whose unary chain runs deeper than the declared depth."""
    return TaggedInsideKernel(
        LogSumExpReduction(),
        RuleSet("toy", LADDER_RULES, MATCHING, NoNormalForm[Term]()),
        LADDER,
        unary_depth=DEPTH,
    )


def chained_kernel() -> TaggedInsideKernel[Term]:
    """Return a kernel over two unary rules that chain, and so over two premises."""
    return TaggedInsideKernel(
        LogSumExpReduction(),
        RuleSet("toy", CHAINED, MATCHING, NoNormalForm[Term]()),
        CHAINED_CATEGORIES,
        unary_depth=DEPTH,
    )


def ruleset(
    normal_form: NormalForm[Term], rules: tuple[ToyRule, ...] = WITH_RAISING
) -> RuleSet[Term]:
    return RuleSet("toy", rules, MATCHING, normal_form)


def test_the_toy_rule_serializes_its_composition_direction() -> None:
    assert FORWARD_COMPOSITION.to_spec() == {
        "rule": "composition",
        "direction": "unrestricted",
    }


def kernel_over(
    normal_form: NormalForm[Term], rules: tuple[ToyRule, ...] = WITH_RAISING
) -> TaggedInsideKernel[Term]:
    return TaggedInsideKernel(
        LogSumExpReduction(), ruleset(normal_form, rules), CATEGORIES, unary_depth=DEPTH
    )


def scored(
    kernel: TaggedInsideKernel[Term],
    index: SpanIndex,
    /,
    *,
    seed: int,
    batch: int = 1,
) -> ScoreChannels:
    transitions = kernel.transitions
    generator = torch.Generator().manual_seed(seed)
    leaf = torch.randn(
        batch,
        index.n,
        len(transitions.categories),
        generator=generator,
        dtype=torch.float64,
    )
    rules = torch.randn(
        batch,
        index.num_spans,
        index.max_splits,
        len(transitions.rules),
        generator=generator,
        dtype=torch.float64,
    )
    unary = (
        None
        if not transitions.unary_rules
        else torch.randn(
            batch,
            index.num_spans,
            len(transitions.unary_rules),
            generator=generator,
            dtype=torch.float64,
        )
    )

    return ScoreChannels(leaf, rules, unary)


def leaf_axioms(
    leaf: torch.Tensor, tokens: Sequence[str], categories: tuple[Term, ...]
) -> Iterator[tuple[int, Entry[Term], float]]:
    """Enumerate one axiom per token and category, weighted by the leaf channel."""
    for position, token in enumerate(tokens):
        for column, category in enumerate(categories):
            yield (position, Entry(token, category), leaf[0, position, column].item())


def edge_score(
    columns: Columns, channels: ScoreChannels, index: SpanIndex, edge: Edge
) -> float:
    """Return the score the tensor chart reads for one symbolic rule application."""
    rule, span, split = edge
    start, end = span
    held = index.span_id(start, end)
    unary = channels.unary

    if split is None:
        if unary is None:
            message = f"{rule.name!r} has no unary channel to read"
            raise AssertionError(message)

        return unary[0, held, columns[1][rule.name]].item()

    return channels.rule[0, held, split - start - 1, columns[0][rule.name]].item()


def rule_columns(kernel: TaggedInsideKernel[Term]) -> Columns:
    """Return the binary and unary score columns, by rule name."""
    transitions = kernel.transitions

    return (
        {rule.name: column for column, rule in enumerate(transitions.rules)},
        {rule.name: column for column, rule in enumerate(transitions.unary_rules)},
    )


class Source:
    """A lexical source admitting every category at every token."""

    def __init__(self, leaf: torch.Tensor, categories: tuple[Term, ...], /) -> None:
        self._leaf = leaf
        self._categories = categories

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    @property
    def forms(self) -> tuple[str, ...]:
        return ()

    def entries(self, form: str) -> tuple[Entry[Term], ...]:
        del form
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[Term], float]]:
        return leaf_axioms(self._leaf, tokens, self._categories)


class RankedSource:
    """The same source, weighting each axiom for the k-best reduction."""

    def __init__(self, leaf: torch.Tensor, categories: tuple[Term, ...], /) -> None:
        self._leaf = leaf
        self._categories = categories

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    @property
    def forms(self) -> tuple[str, ...]:
        return ()

    def entries(self, form: str) -> tuple[Entry[Term], ...]:
        del form
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterator[tuple[int, Entry[Term], Ranked]]:
        for position, entry, weight in leaf_axioms(
            self._leaf, tokens, self._categories
        ):
            yield (position, entry, (weight,))


class Scorer:
    """An edge scorer reading the same channels the tensor chart reads."""

    lowering = SpanDependent()

    def __init__(
        self,
        kernel: TaggedInsideKernel[Term],
        channels: ScoreChannels,
        index: SpanIndex,
        /,
    ) -> None:
        self._columns = rule_columns(kernel)
        self._channels = channels
        self._index = index

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position, entry
        return 0.0

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del inputs, output

        return edge_score(
            self._columns, self._channels, self._index, (rule, span, split)
        )


class RankedScorer:
    """The same scorer, weighting each edge for the k-best reduction."""

    lowering = SpanDependent()

    def __init__(
        self,
        kernel: TaggedInsideKernel[Term],
        channels: ScoreChannels,
        index: SpanIndex,
        /,
    ) -> None:
        self._columns = rule_columns(kernel)
        self._channels = channels
        self._index = index

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> Ranked:
        del position, entry
        return (0.0,)

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> Ranked:
        del inputs, output

        return (
            edge_score(self._columns, self._channels, self._index, (rule, span, split)),
        )


def tokens_of(index: SpanIndex) -> tuple[str, ...]:
    return tuple(f"w{position}" for position in range(index.n))


def symbolic_k_best(
    kernel: TaggedInsideKernel[Term],
    channels: ScoreChannels,
    index: SpanIndex,
    k: int,
) -> Ranked:
    """Return the best few root weights the symbolic k-best reduction finds."""
    categories = kernel.transitions.categories
    reduction = KBest(k)
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=kernel.transitions.rule_set,
        source=RankedSource(channels.leaf, categories),
        semiring=reduction,
        scorer=RankedScorer(kernel, channels, index),
        goal=RootGoal(TargetSet(categories, MATCHING)),
        unary_depth=DEPTH,
    )
    total = reduction.zero

    for key in forest.items_at((0, index.n)):
        total = reduction.plus(total, forest.inside(key))

    return total


def symbolic_derivations(
    kernel: TaggedInsideKernel[Term], channels: ScoreChannels, index: SpanIndex
) -> list[tuple[float, _Shape]]:
    """Return every symbolic derivation of the whole input, heaviest first."""
    categories = kernel.transitions.categories
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=kernel.transitions.rule_set,
        source=Source(channels.leaf, categories),
        semiring=SymbolicLogInside(),
        scorer=Scorer(kernel, channels, index),
        goal=RootGoal(TargetSet(categories, MATCHING)),
        unary_depth=DEPTH,
    )
    found = [
        (symbolic_total(derivation), symbolic_shape(derivation))
        for key in forest.items_at((0, index.n))
        for derivation in forest.derivations(key)
    ]

    return sorted(found, key=lambda held: -held[0])


def symbolic_shape(derivation: Derivation[Term, float]) -> _Shape:
    name = derivation.edge.rule_name

    return (
        derivation.key.span,
        derivation.key.reading,
        derivation.key.nf_tag,
        None if name.startswith(LEXICAL_TAG_PREFIX) else name,
        tuple(symbolic_shape(child) for child in derivation.children),
    )


def symbolic_total(derivation: Derivation[Term, float]) -> float:
    return derivation.edge.weight + sum(
        symbolic_total(child) for child in derivation.children
    )


def symbolic_brackets(shape: _Shape) -> frozenset[Cell]:
    span, _, _, _, children = shape
    held = frozenset({span}) if len(children) == 2 else frozenset()

    return held.union(*(symbolic_brackets(child) for child in children))


def shape_of(kernel: TaggedInsideKernel[Term], tree: TaggedTree) -> _Shape:
    """Return a decoded derivation in the terms the symbolic chart names."""
    transitions = kernel.transitions

    if tree.rule is None:
        name = None
    elif len(tree.children) == 1:
        name = transitions.unary_rules[tree.rule].name
    else:
        name = transitions.rules[tree.rule].name

    return (
        tree.span,
        transitions.categories[tree.category],
        transitions.tags[tree.tag],
        name,
        tuple(shape_of(kernel, child) for child in tree.children),
    )


@pytest.mark.parametrize("n", [2, 3])
def test_the_best_few_derivations_are_the_symbolic_chart_s_best_few(n: int) -> None:
    index = SpanIndex(n)
    kernel = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(kernel, index, seed=n)
    reference = symbolic_derivations(kernel, channels, index)[:KEPT]

    found = TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    assert [shape_of(kernel, tree) for tree in found[0]] == [
        shape for _, shape in reference
    ]
    assert [tree.total for tree in found[0]] == pytest.approx(
        [total for total, _ in reference], abs=TOLERANCE
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_best_few_weights_are_the_symbolic_k_best_reduction(n: int) -> None:
    index = SpanIndex(n)
    kernel = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(kernel, index, seed=n + 11)

    found = TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    assert [tree.total for tree in found[0]] == pytest.approx(
        list(symbolic_k_best(kernel, channels, index, KEPT)), abs=TOLERANCE
    )


@pytest.mark.parametrize("n", [1, 2, 3, 4])
def test_the_heaviest_derivation_weighs_what_viterbi_says_it_does(n: int) -> None:
    index = SpanIndex(n)
    kernel = kernel_over(EisnerNormalForm[Term]())
    channels = scored(kernel, index, seed=n + 21)
    best = TaggedInsideKernel(
        MaximumReduction(), kernel.transitions.rule_set, CATEGORIES, unary_depth=DEPTH
    )

    found = TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    assert found[0][0].total == pytest.approx(
        best.log_partition(channels, index).item(), abs=TOLERANCE
    )


@pytest.mark.parametrize("n", [2, 3, 4])
def test_the_derivations_come_back_heaviest_first_and_all_different(n: int) -> None:
    index = SpanIndex(n)
    kernel = kernel_over(NoNormalForm[Term]())
    channels = scored(kernel, index, seed=n + 31)

    found = TaggedDecoder(kernel, k=KEPT).k_best(channels, index)[0]

    assert [tree.total for tree in found] == sorted(
        (tree.total for tree in found), reverse=True
    )
    assert len(set(found)) == len(found)


def test_an_item_with_fewer_derivations_than_asked_for_yields_what_it_has() -> None:
    index = SpanIndex(2)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=41)

    found = TaggedDecoder(kernel, k=20).k_best(channels, index)

    assert len(found[0]) == len(symbolic_derivations(kernel, channels, index))


def test_a_one_token_item_yields_the_derivations_of_its_own_cell() -> None:
    index = SpanIndex(1)
    kernel = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(kernel, index, seed=43)

    found = TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    assert [shape_of(kernel, tree) for tree in found[0]] == [
        shape for _, shape in symbolic_derivations(kernel, channels, index)[:KEPT]
    ]


def test_a_one_token_item_yields_one_derivation_per_state() -> None:
    index = SpanIndex(1)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=105)

    found = TaggedDecoder(kernel, k=4).k_best(channels, index)

    assert len(found[0]) == len(CATEGORIES)
    assert len({shape_of(kernel, tree) for tree in found[0]}) == len(CATEGORIES)
    assert [tree.total for tree in found[0]] == sorted(
        channels.leaf[0, 0].tolist(), reverse=True
    )


def test_an_item_nothing_derives_yields_no_tree() -> None:
    index = SpanIndex(3)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = ScoreChannels(
        torch.zeros(1, index.n, len(CATEGORIES), dtype=torch.float64),
        torch.full(
            (1, index.num_spans, index.max_splits, 1), -inf, dtype=torch.float64
        ),
    )

    found = TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    assert found == ((),)


def test_a_ragged_batch_decodes_each_item_over_its_own_tokens() -> None:
    index = SpanIndex(4)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=47, batch=2)

    found = TaggedDecoder(kernel, k=1).k_best(
        channels, index, lengths=torch.tensor([2, 4])
    )

    assert found[0][0].span == (0, 2)
    assert found[1][0].span == (0, 4)


def test_the_decode_refuses_a_width_of_less_than_one() -> None:
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))

    with pytest.raises(ValueError, match=r"at least one derivation, not 0"):
        TaggedDecoder(kernel, k=0)


def test_only_the_goal_categories_end_a_derivation() -> None:
    index = SpanIndex(2)
    kernel = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(kernel, index, seed=53)
    decoder = TaggedDecoder(kernel, k=10)

    lifted = decoder.k_best(channels, index, goals=(FUNCTOR,))
    every = decoder.k_best(channels, index)

    assert {shape_of(kernel, tree)[1] for tree in lifted[0]} == {FUNCTOR}
    assert {shape_of(kernel, tree)[1] for tree in every[0]} == set(CATEGORIES)


def test_a_goal_outside_the_inventory_is_refused_by_name() -> None:
    index = SpanIndex(2)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=59)
    goal = CATEGORY_BY_NAME["PP"]

    with pytest.raises(ValueError, match=rf"goal category {re.escape(repr(goal))} is"):
        TaggedDecoder(kernel, k=1).k_best(channels, index, goals=(goal,))


def test_a_type_raised_derivation_is_decoded_with_the_step_that_raised_it() -> None:
    index = SpanIndex(2)
    kernel = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(kernel, index, seed=61)

    found = TaggedDecoder(kernel, k=30).k_best(channels, index)
    raised = [
        tree
        for tree in found[0]
        for shape in [shape_of(kernel, tree)]
        if shape[3] == RAISING.name
    ]

    assert raised
    assert all(len(tree.children) == 1 for tree in raised)
    assert all(shape_of(kernel, tree)[2] == "FTR" for tree in raised)


def test_a_normal_form_narrows_the_derivations_a_decode_can_reach() -> None:
    index = SpanIndex(3)
    unfiltered = kernel_over(NoNormalForm[Term]())
    filtered = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(unfiltered, index, seed=67)
    decoder = TaggedDecoder(unfiltered, k=200)

    every = decoder.k_best(channels, index)
    admitted = TaggedDecoder(filtered, k=200).k_best(channels, index)

    assert len(admitted[0]) < len(every[0])


def enumerated_marginals(
    kernel: TaggedInsideKernel[Term], channels: ScoreChannels, index: SpanIndex
) -> dict[Cell, float]:
    """Return each span's bracketing share of every symbolic derivation's weight."""
    found = [
        (exp(total), symbolic_brackets(shape))
        for total, shape in symbolic_derivations(kernel, channels, index)
    ]
    mass = sum(weight for weight, _ in found)

    return {
        span: sum(weight for weight, held in found if span in held) / mass
        for span in index.spans_by_length(min_length=2)
    }


def bracket_gain(marginals: Mapping[Cell, float], brackets: frozenset[Cell]) -> float:
    """Return the expected number of constituents one bracketing gets right."""
    return sum(marginals[span] for span in brackets)


@pytest.mark.parametrize("n", [2, 3])
def test_the_mbr_derivation_is_a_symbolic_tree_of_the_greatest_expected_gain(
    n: int,
) -> None:
    index = SpanIndex(n)
    kernel = kernel_over(HockenmaierBiskNormalForm[Term]())
    channels = scored(kernel, index, seed=n + 71)
    reference = enumerated_marginals(kernel, channels, index)
    derivations = symbolic_derivations(kernel, channels, index)
    best = max(
        bracket_gain(reference, symbolic_brackets(shape)) for _, shape in derivations
    )

    found = TaggedDecoder(kernel).mbr_trees(channels, index)[0]

    assert found is not None
    assert shape_of(kernel, found) in {shape for _, shape in derivations}
    assert found.total == pytest.approx(best, abs=TOLERANCE)
    assert bracket_gain(reference, frozenset(found.constituents())) == pytest.approx(
        best, abs=TOLERANCE
    )


def test_the_mbr_derivation_is_not_always_the_heaviest_one() -> None:
    index = SpanIndex(4)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=DIVERGENT_SEED)

    heaviest = TaggedDecoder(kernel, k=1).k_best(channels, index)[0][0]
    risk = TaggedDecoder(kernel).mbr_trees(channels, index)[0]

    assert risk is not None
    assert risk.constituents() != heaviest.constituents()


def test_a_chart_certain_of_one_derivation_decodes_it_either_way() -> None:
    index = SpanIndex(3)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    leaf = torch.zeros(1, index.n, len(CATEGORIES), dtype=torch.float64)
    leaf[0, :, 1] = -inf
    rules = torch.full(
        (1, index.num_spans, index.max_splits, 1), -inf, dtype=torch.float64
    )
    rules[0, index.span_id(0, 2), 0, 0] = 0.0
    rules[0, index.span_id(0, 3), 1, 0] = 0.0
    channels = ScoreChannels(leaf, rules)

    heaviest = TaggedDecoder(kernel, k=1).k_best(channels, index)[0][0]
    risk = TaggedDecoder(kernel).mbr_trees(channels, index)[0]

    assert risk is not None
    assert risk == heaviest
    assert risk.constituents() == ((0, 3), (0, 2))


def test_the_mbr_derivation_uses_no_step_the_chart_gives_no_mass() -> None:
    index = SpanIndex(3)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=83)
    counts = kernel.expected_counts(channels, index)

    risk = TaggedDecoder(kernel).mbr_trees(channels, index)[0]

    assert risk is not None

    for start, end in risk.constituents():
        assert counts.rule[0, index.span_id(start, end)].sum().item() > 0.0


def test_an_item_the_chart_cannot_derive_has_no_mbr_tree() -> None:
    index = SpanIndex(3)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = ScoreChannels(
        torch.zeros(1, index.n, len(CATEGORIES), dtype=torch.float64),
        torch.full(
            (1, index.num_spans, index.max_splits, 1), -inf, dtype=torch.float64
        ),
    )

    assert TaggedDecoder(kernel).mbr_trees(channels, index) == (None,)


def test_minimum_bayes_risk_needs_log_sum_exp_reduction() -> None:
    index = SpanIndex(2)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=89)
    best = TaggedInsideKernel(
        MaximumReduction(), kernel.transitions.rule_set, CATEGORIES, unary_depth=DEPTH
    )

    with pytest.raises(ValueError, match=r"expected count requires log-sum-exp"):
        TaggedDecoder(best).mbr_trees(channels, index)


def test_a_ragged_batch_brackets_each_item_over_its_own_tokens() -> None:
    index = SpanIndex(4)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=97, batch=2)

    found = TaggedDecoder(kernel).mbr_trees(
        channels, index, lengths=torch.tensor([2, 4])
    )

    assert found[0] is not None
    assert found[1] is not None
    assert found[0].constituents() == ((0, 2),)
    assert max(end for _, end in found[1].constituents()) == 4


def unary_steps(tree: TaggedTree) -> int:
    """Return how many unary steps a derivation takes above its widest premise."""
    if len(tree.children) != 1:
        return 0

    return 1 + unary_steps(tree.children[0])


@pytest.mark.parametrize("n", [1, 2])
def test_a_two_step_unary_chain_decodes_as_the_symbolic_chart_does(n: int) -> None:
    index = SpanIndex(n)
    kernel = chained_kernel()
    channels = scored(kernel, index, seed=n + 151)
    reference = symbolic_derivations(kernel, channels, index)

    found = TaggedDecoder(kernel, k=len(reference)).k_best(channels, index)

    assert len(kernel.transitions.closure.schedule) > 1
    assert max(unary_steps(tree) for tree in found[0]) == DEPTH
    assert {shape_of(kernel, tree) for tree in found[0]} == {
        shape for _, shape in reference
    }
    assert sorted(tree.total for tree in found[0]) == pytest.approx(
        sorted(total for total, _ in reference), abs=TOLERANCE
    )


def base_scored(
    kernel: TaggedInsideKernel[Term], index: SpanIndex, /, *, seed: int
) -> ScoreChannels:
    """Return random channels admitting the first category alone at every token."""
    channels = scored(kernel, index, seed=seed)
    leaf = channels.leaf.clone()
    leaf[:, :, 1:] = -inf

    return ScoreChannels(leaf, channels.rule, channels.unary)


def test_a_chain_deeper_than_the_depth_stops_where_the_symbolic_chart_stops() -> None:
    index = SpanIndex(1)
    kernel = ladder_kernel()
    channels = base_scored(kernel, index, seed=163)
    forest = parse(
        tokens_of(index),
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        rules=kernel.transitions.rule_set,
        source=Source(channels.leaf, (LADDER[0],)),
        semiring=SymbolicLogInside(),
        scorer=Scorer(kernel, channels, index),
        goal=RootGoal(TargetSet(LADDER, MATCHING)),
        unary_depth=DEPTH,
    )
    reference = [
        (symbolic_total(derivation), symbolic_shape(derivation))
        for key in forest.items_at((0, index.n))
        for derivation in forest.derivations(key)
    ]

    found = TaggedDecoder(kernel, k=len(reference)).k_best(channels, index)

    assert forest.truncated
    assert {shape_of(kernel, tree) for tree in found[0]} == {
        shape for _, shape in reference
    }
    assert sorted(tree.total for tree in found[0]) == pytest.approx(
        sorted(total for total, _ in reference), abs=TOLERANCE
    )


def test_the_decoder_keeps_one_derivation_when_no_width_is_asked_for() -> None:
    index = SpanIndex(2)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=107)

    found = TaggedDecoder(kernel).k_best(channels, index)

    assert len(symbolic_derivations(kernel, channels, index)) > 1
    assert len(found[0]) == 1


def test_the_decoder_reports_the_kernel_and_the_width_it_was_given() -> None:
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    decoder = TaggedDecoder(kernel, k=7)

    assert decoder.kernel is kernel
    assert decoder.k == 7


def test_the_decode_leaves_no_gradient_on_the_scores_it_read() -> None:
    index = SpanIndex(3)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = scored(kernel, index, seed=101)
    channels.rule.requires_grad_()

    TaggedDecoder(kernel, k=KEPT).k_best(channels, index)

    assert channels.rule.grad is None


def test_the_decode_refuses_channels_it_cannot_read() -> None:
    index = SpanIndex(3)
    kernel = kernel_over(NoNormalForm[Term](), (FORWARD_COMPOSITION,))
    channels = ScoreChannels(
        torch.zeros(1, index.n, len(CATEGORIES), dtype=torch.float64),
        torch.zeros(1, index.num_spans, index.max_splits, 2, dtype=torch.float64),
    )

    with pytest.raises(ValueError, match=r"rule scores have shape"):
        TaggedDecoder(kernel, k=1).k_best(channels, index)
