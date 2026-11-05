from __future__ import annotations

import ast
from itertools import permutations, product
from math import comb, exp, isclose, isfinite, log
from pathlib import Path
from string import ascii_letters
from typing import TYPE_CHECKING, ClassVar, Final, Literal, NamedTuple, cast, override

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.declarations import SMALL, Witnessed, category_language, witnessed_grammars
from tests.support.rules import ccg_signature, declared_rules

import lofi.deduce._engine as engine
from lofi._native import SpanFree, UnitEdges
from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    Counting,
    CoverGoal,
    InsideChart,
    LogInside,
    PackedForest,
    ParsePolicy,
    Reachability,
    Retain,
    RootGoal,
    TopKPruner,
    Viterbi,
    outside_from_edges,
    parse,
)
from lofi.judge import (
    COMPOSITION_RESTRICTIONS,
    EisnerNormalForm,
    GeneralizedComposition,
    HockenmaierBiskNormalForm,
    NoNormalForm,
    NormalFormRole,
    PatternRule,
    RuleIndex,
    RuleProbe,
    RuleSet,
)
from lofi.lexicon import Entry
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    InfixConstructorNotationSpec,
    NodePatternSpec,
    PatternRuleSpec,
    RuleSetSpec,
    SubstitutionSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec.components.run import ParseSpec
from lofi.spec.grammar import (
    CategoryMeasure,
    FeatureErasureMatchingSpec,
    FunctorSpec,
    GrammarIso,
    IdentityMatchingSpec,
    OperatorSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language
from lofi.terms import (
    FeatureErasure,
    MVar,
    Node,
    RepresentativeKernel,
    Spine,
    TargetSet,
    Term,
)

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.deduce import (
        Cell,
        CellPruner,
        Derivation,
        Edge,
        EdgeScorer,
        ItemKey,
        Semiring,
    )
    from lofi.judge import (
        Direction,
        HeadIndex,
        NormalForm,
        NormalFormDecisionTable,
        PatternRule,
        Rule,
        RuleRestriction,
        SchematicSequent,
        Slot,
    )
    from lofi.lexicon import Clamp, LexicalSource
    from lofi.spec import PatternSpec, RuleSpec
    from lofi.spec.grammar import MatchingSpec
    from lofi.terms import Functors, Matching, Pattern, TermMatching, TermRuleKernel


_NEVER: Final = "never"
"""A primitive no lexical row bears, so a functor over it never fires a pattern."""

_RELATION_NAMES: Final = (
    *ascii_letters,
    "NP",
    _NEVER,
    "S/NP",
    "S\\NP",
    "NP/N",
    "N/N",
    "(S\\NP)/NP",
)

_RELATION_SIGNATURE: Final = ccg_signature(
    *ascii_letters, "NP", _NEVER, name="engine-relations"
)
_RELATION_ANCHOR: Final = declared_rules(
    _RELATION_SIGNATURE, RuleSetSpec(name="engine-relations", rules=())
)
_RELATION_STORE: Final = _RELATION_ANCHOR.context.store
_RELATION_SYNTAX: Final = _RELATION_ANCHOR.context.syntax
_RELATION_TERMS: Final[dict[str, Term]] = {
    name: _RELATION_SYNTAX.parse(_RELATION_STORE, name) for name in _RELATION_NAMES
}
_RELATION_LABELS: Final[dict[Term, str]] = {
    term: name for name, term in _RELATION_TERMS.items()
}


def _category(name: str) -> Term:
    """Return the declared category with this test name."""
    return _RELATION_TERMS[name]


def _label(term: Term) -> str:
    """Return one test category's declared name."""
    return _RELATION_LABELS[term]


def _pattern(term: Term) -> PatternSpec:
    """Return the ground pattern declaration naming one exact category."""
    if not term.args:
        return ConstantPatternSpec(name=_RELATION_SYNTAX.render(_RELATION_STORE, term))

    return NodePatternSpec(
        operator=FixedOperatorSpec(name=term.op.name),
        arguments=tuple(_pattern(argument) for argument in term.args),
    )


def _relation_rule(
    arity: int, row: tuple[tuple[str, ...], tuple[str, ...]] | None
) -> Rule[Term]:
    """Compile the pattern rule of one finite-relation row, or one that never fires.

    A row's outputs are the finite targets of one open conclusion, so a row
    with several outputs is still one clause set over one premise tuple.
    """
    if row is None:
        premises: tuple[PatternSpec, ...] = tuple(
            NodePatternSpec(
                operator=FixedOperatorSpec(name="/"),
                arguments=(
                    ConstantPatternSpec(name=_NEVER),
                    TermPatternSpec(name=f"X{at}", sort="Cat"),
                ),
            )
            for at in range(arity)
        )
        outputs: tuple[str, ...] = ()
    else:
        inputs, outputs = row
        premises = tuple(_pattern(_category(name)) for name in inputs)
    (rule,) = declared_rules(
        _RELATION_SIGNATURE,
        RuleSetSpec(
            name="relation",
            rules=(
                PatternRuleSpec(
                    name="relation",
                    premises=premises,
                    conclusion=TermPatternSpec(name="X0", sort="Cat")
                    if not outputs
                    else _pattern(_category(outputs[0]))
                    if len(outputs) == 1
                    else TermPatternSpec(name="Z", sort="Cat"),
                    targets=outputs if len(outputs) > 1 else None,
                    head_policy="pass-through" if arity == 1 else "non-absorbing",
                ),
            ),
        ),
    ).rule_set.rules

    return rule


class Casefold:
    """Compares category terms without case distinctions."""

    @property
    def name(self) -> str:
        return "casefold"

    @property
    def has_key(self) -> bool:
        return True

    @property
    def representatives(self) -> RepresentativeKernel:
        return RepresentativeKernel.identity().refine(
            tuple(
                (term, _RELATION_TERMS.get(label.casefold(), term))
                for term, label in _RELATION_LABELS.items()
            )
        )

    def key(self, term: Term, /) -> Term:
        return _RELATION_TERMS.get(_label(term).casefold(), term)

    def equivalent(self, left: Term, right: Term, /) -> bool:
        return self.key(left) is self.key(right)


EXACT: Final[TermMatching] = cast("TermMatching", _RELATION_ANCHOR.rule_set.matching)
"""The identity congruence the relation grammar compiles its kernels under."""
CASEFOLD: Final[TermMatching] = Casefold()


class Relation:
    """Enumerates a finite relation over declared category terms."""

    def __init__(
        self,
        name: str,
        arity: int,
        table: Mapping[tuple[str, ...], tuple[str, ...]],
        matching: TermMatching = EXACT,
        *,
        directions: frozenset[Direction] = frozenset({"forward", "check"}),
        complete: bool = True,
        probe: RuleProbe | None = None,
        admitted: Mapping[tuple[str, ...], tuple[str, ...]] | None = None,
        order: int = 0,
        publish_schemata: bool = True,
    ) -> None:
        self._name = name
        self._arity = arity
        self._table = {
            tuple(_category(name) for name in inputs): tuple(
                _category(name) for name in outputs
            )
            for inputs, outputs in table.items()
        }
        self._matching = matching
        self._directions = directions
        self._complete = complete
        self._probe = probe
        self._admitted = (
            self._table
            if admitted is None
            else {
                tuple(_category(name) for name in inputs): tuple(
                    _category(name) for name in outputs
                )
                for inputs, outputs in admitted.items()
            }
        )
        self._order = order
        rows = tuple(table.items())

        if len(rows) > 1:
            msg = "a finite-relation stand-in holds one premise tuple"
            raise ValueError(msg)

        compiled = cast("PatternRule", _relation_rule(arity, rows[0] if rows else None))
        self._reachability_schemata = (
            compiled.reachability_schemata if publish_schemata and rows else ()
        )
        self._lowering = compiled.lowering

    @property
    def name(self) -> str:
        return self._name

    @property
    def arity(self) -> int:
        return self._arity

    @property
    def directions(self) -> frozenset[Direction]:
        return self._directions

    @property
    def residual_slots(self) -> frozenset[int]:
        return frozenset()

    @property
    def normal_form_role(self) -> NormalFormRole:
        return NormalFormRole()

    @property
    def spine_order(self) -> int:
        return self._order

    @property
    def forward_is_complete(self) -> bool:
        return self._complete

    @property
    def matching(self) -> TermMatching:
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        return None

    @property
    def probe(self) -> RuleProbe | None:
        return self._probe

    @property
    def lowering(self) -> TermRuleKernel:
        return self._lowering

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        return self._reachability_schemata

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        yield from self._table.get(inputs, ())

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return any(
            self._matching.equivalent(output, candidate)
            for candidate in self._admitted.get(inputs, ())
        )

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        del output, known
        return iter(())

    def to_spec(self) -> Mapping[str, JsonValue]:
        return {"rule": self._name, "arity": self._arity}


def relations(
    name: str, arity: int, table: Mapping[tuple[str, ...], tuple[str, ...]]
) -> tuple[Relation, ...]:
    """Return one finite-relation stand-in per premise tuple, each named apart."""
    return tuple(
        Relation(f"{name}-{at}", arity, {inputs: outputs})
        for at, (inputs, outputs) in enumerate(table.items())
    )


class Source:
    """Fixed lexical row source."""

    def __init__(self, rows: Sequence[tuple[int, Entry[str], float]]) -> None:
        self._rows = tuple(
            (
                position,
                Entry(
                    entry.form,
                    _category(entry.category),
                    (None if entry.meaning is None else _category(entry.meaning)),
                    entry.weight,
                ),
                weight,
            )
            for position, entry, weight in rows
        )
        self.calls = 0

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    @property
    def length(self) -> int:
        return len(self._rows)

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], float]]:
        del tokens
        self.calls += 1
        return self._rows


class Scorer:
    """Looks up local lexical and rule factors."""

    def __init__(self, rules: Mapping[str, float] | None = None) -> None:
        self._rules = {} if rules is None else dict(rules)
        self.trials: list[str] = []

    @property
    def lowering(self) -> SpanFree:
        return SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del position
        return entry.weight

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del inputs, output, span, split
        self.trials.append(rule.name)
        return self._rules.get(rule.name, 0.0)


class Beam:
    """Proposes the same candidates for every rule trial."""

    def __init__(self, candidates: Sequence[str]) -> None:
        self._candidates = tuple(_category(candidate) for candidate in candidates)
        self.calls = 0

    def propose(
        self, rule: Rule[Term], inputs: tuple[Term, ...], span: Cell, /
    ) -> Iterator[Term]:
        del rule, inputs, span
        self.calls += 1
        yield from self._candidates


class RuleTags:
    """Tags derived items with the rule name."""

    def __init__(self, names: Sequence[str]) -> None:
        self._alphabet = frozenset({"L", *names})

    @property
    def name(self) -> str:
        return "rule-tags"

    @property
    def alphabet(self) -> frozenset[str]:
        return self._alphabet

    def lexical_tag(self) -> str:
        return "L"

    def decisions(self, rules: Sequence[Rule[Term]], /) -> NormalFormDecisionTable:
        return tuple(
            dict.fromkeys(product(self.alphabet, repeat=rule.arity), "admit")
            for rule in rules
        )

    def tag(self, rule: Rule[Term], children: tuple[str, ...], /) -> str:
        del children
        return rule.name

    def admits(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        del output
        return len(children) == len(inputs) == rule.arity


class Lowercase:
    """Normalizes category strings to lowercase."""

    def __init__(self) -> None:
        self.seen: list[str] = []

    def normalize(self, term: Term, /) -> Term:
        label = _label(term)
        self.seen.append(label)
        return _category(label.lower())


class IdentityNormalizer:
    """Returns each exact term while exercising the normalization boundary."""

    def normalize(self, term: Term, /) -> Term:
        return term


class Heads:
    """Indexes terms by their first character."""

    _code: ClassVar = {"A": 0, "B": 1, "C": 2, "X": 3, "Y": 4}

    @property
    def codes(self) -> frozenset[int]:
        return frozenset(self._code.values())

    def term(self, term: Term, /) -> int:
        return self._code[_label(term)[0]]

    def named(self, name: str, /) -> int:
        return self._code[name]


def source(*categories: str) -> Source:
    return Source(
        tuple(
            (position, Entry(token, category, weight=0.0), 0.0)
            for position, (token, category) in enumerate(
                zip(("a", "b", "c", "d"), categories, strict=False)
            )
        )
    )


def inventory(
    rules: Sequence[Rule[Term]],
    matching: Matching[Term] = EXACT,
    normal_form: NormalForm[Term] | None = None,
    *,
    index: RuleIndex[Term] | None = None,
) -> RuleSet[Term]:
    return RuleSet(
        "test",
        rules,
        matching,
        NoNormalForm[Term]() if normal_form is None else normal_form,
        index=index,
    )


def run(
    rules: RuleSet[Term],
    lexical: Source,
    *,
    scorer: Scorer | None = None,
    proposer: Beam | None = None,
    normalizer: Lowercase | None = None,
    unary_depth: int = 3,
    threads: int = 1,
) -> PackedForest[Term, float]:
    return parse(
        tuple("word" for _ in range(lexical.length)),
        rules=rules,
        source=lexical,
        semiring=LogInside(),
        scorer=Scorer() if scorer is None else scorer,
        goal=RootGoal(TargetSet((_category("S"),), rules.matching)),
        proposer=proposer,
        normalizer=normalizer,
        unary_depth=unary_depth,
        threads=threads,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


class UnitSource:
    """Unit-weight lexical category source."""

    def __init__(self, categories: Sequence[str]) -> None:
        self._categories = tuple(_category(category) for category in categories)

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        del tokens
        return tuple(
            (position, Entry(str(position), category), 1)
            for position, category in enumerate(self._categories)
        )


class UnitScorer:
    """Assigns unit weight to every chart edge."""

    @property
    def lowering(self) -> UnitEdges:
        return UnitEdges()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> int:
        del position, entry
        return 1

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> int:
        del rule, inputs, output, span, split
        return 1


def _constructor(name: str, spelling: str) -> ConstructorSpec:
    """Declare one binary category constructor."""
    return ConstructorSpec(
        name=name,
        kind=("Cat", "Cat", "Cat"),
        notation=InfixConstructorNotationSpec(
            spelling=spelling, precedence=10, associativity="left"
        ),
    )


FEATURED_SIGNATURE: Final = SignatureSpec(
    name="featured",
    primitives=("x", "t"),
    operators=(
        OperatorSpec(name="x[b]", sort="Cat", erases_to="x"),
        OperatorSpec(name="x[ng]", sort="Cat", erases_to="x"),
    ),
    constructors=(_constructor("fwd", "/"), _constructor("bwd", "\\")),
    functors=(
        FunctorSpec(
            name="fwd", constructor="fwd", argument_side="right", opposite="bwd"
        ),
        FunctorSpec(
            name="bwd", constructor="bwd", argument_side="left", opposite="fwd"
        ),
    ),
)
"""The featured category signature every compiled-rule test declares."""


def featured_compiled(
    *rules: RuleSpec, matching: MatchingSpec | None = None
) -> tuple[Rule[Term], ...]:
    """Compile a rule inventory over the featured signature, erasing features."""
    return declared_rules(
        FEATURED_SIGNATURE,
        RuleSetSpec(name="featured", rules=rules),
        matching=FeatureErasureMatchingSpec() if matching is None else matching,
    ).rule_set.rules


def featured_matching(matching: MatchingSpec, /) -> Matching[Term]:
    """Return the premise congruence a featured grammar compiles under."""
    return declared_rules(
        FEATURED_SIGNATURE,
        RuleSetSpec(name="featured", rules=()),
        matching=matching,
    ).rule_set.matching


_FEATURED: Final = declared_rules(
    FEATURED_SIGNATURE, RuleSetSpec(name="featured", rules=())
).context
FEATURED_STORE: Final = _FEATURED.store
FEATURED_FUNCTORS: Final[Functors[Term]] = _FEATURED.functors["fwd"][0]


def featured(name: str, *arguments: Term) -> Term:
    """Return a featured atom by name, or the functor a slash builds."""
    if arguments:
        result, argument = arguments
        return FEATURED_FUNCTORS.make(name, result, argument)

    return _FEATURED.syntax.parse(FEATURED_STORE, name)


def exact_pattern(term: Term) -> PatternSpec:
    """Return the ground pattern declaration naming one featured category."""
    if not term.args:
        return ConstantPatternSpec(name=_FEATURED.syntax.render(FEATURED_STORE, term))

    return NodePatternSpec(
        operator=FixedOperatorSpec(name=term.op.name),
        arguments=tuple(exact_pattern(argument) for argument in term.args),
    )


def ground_rule(name: str, premises: tuple[Term, ...], conclusion: Term) -> RuleSpec:
    """Declare the ground pattern rule from exact premises to one conclusion."""
    return PatternRuleSpec(
        name=name,
        premises=tuple(exact_pattern(premise) for premise in premises),
        conclusion=exact_pattern(conclusion),
        head_policy="pass-through" if len(premises) == 1 else "non-absorbing",
    )


CATEGORY_X: Final = featured("x")
CATEGORY_T: Final = featured("t")
CATEGORY_FUNCTORS: Final = FEATURED_FUNCTORS


class TermSource:
    """Single-term lexical source with unit weight."""

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        del tokens
        return ((0, Entry("x", CATEGORY_X), 1),)


class TermScorer:
    """Assigns unit weight to category-term edges."""

    @property
    def lowering(self) -> UnitEdges:
        return UnitEdges()

    def axiom_weight(
        self,
        position: int,
        entry: Entry[Term],
        /,
    ) -> int:
        del position, entry
        return 1

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> int:
        del rule, inputs, output, span, split
        return 1


class ExactTermScorer(TermScorer):
    """Assigns unit counting weights without enabling product factoring."""

    @property
    @override
    def lowering(self) -> SpanFree:
        return SpanFree()


class ViterbiTermScorer:
    """Assigns the Viterbi multiplicative identity to every edge."""

    @property
    def lowering(self) -> UnitEdges:
        return UnitEdges()

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
        del rule, inputs, output, span, split
        return 0.0


class ExactViterbiTermScorer(ViterbiTermScorer):
    """Assigns unit Viterbi weights without enabling product factoring."""

    @property
    @override
    def lowering(self) -> SpanFree:
        return SpanFree()


class RaisingBeam:
    """Proposes one nested type-raised conclusion per input."""

    def __init__(self) -> None:
        self.spans: list[Cell] = []

    def propose(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        span: Cell,
        /,
    ) -> Iterator[Term]:
        del rule
        self.spans.append(span)
        inner = CATEGORY_FUNCTORS.make("bwd", CATEGORY_T, inputs[0])
        yield CATEGORY_FUNCTORS.make("fwd", CATEGORY_T, inner)


def run_counting(
    rules: RuleSet[Term],
    categories: Sequence[str],
    *,
    proposer: Beam | None = None,
    unary_depth: int = 3,
) -> PackedForest[Term, int]:
    return parse(
        tuple("word" for _ in categories),
        rules=rules,
        source=UnitSource(categories),
        semiring=Counting(),
        scorer=UnitScorer(),
        goal=RootGoal(TargetSet((_category("S"),), rules.matching)),
        proposer=proposer,
        unary_depth=unary_depth,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def test_counting_promotes_a_parse_result_past_u64_max() -> None:
    """Return an exact Python integer for a sufficiently ambiguous parse."""

    def binary_tree_count(leaves: int) -> int:
        branchings = leaves - 1
        return comb(2 * branchings, branchings) // leaves

    word_bits = 64
    word_max = 2**word_bits - 1
    token_count = next(
        leaves
        for leaves in range(2, word_bits + 2)
        if binary_tree_count(leaves) > word_max
    )
    atom = _category("A")
    rules = inventory((Relation("combine", 2, {("A", "A"): ("A",)}),))
    chart = parse(
        tuple("word" for _ in range(token_count)),
        rules=rules,
        source=UnitSource(("A",) * token_count),
        semiring=Counting(),
        scorer=UnitScorer(),
        goal=RootGoal(TargetSet((atom,), rules.matching)),
        unary_depth=0,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )
    count = dict(chart.categories_at((0, token_count)))[atom]

    assert binary_tree_count(token_count - 1) <= word_max
    assert isinstance(count, int)
    assert count == binary_tree_count(token_count)
    assert count > word_max


def run_raising(
    rule: Rule[Term],
    normal_form: NormalForm[Term],
    *,
    proposer: RaisingBeam | None = None,
    unary_depth: int,
) -> PackedForest[Term, int]:
    rules = RuleSet(
        "raising",
        (rule,),
        rule.matching,
        normal_form,
    )

    return parse(
        ("x",),
        rules=rules,
        source=TermSource(),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((CATEGORY_X,), rule.matching)),
        proposer=proposer,
        unary_depth=unary_depth,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def test_preflight_stops_before_chart_construction_or_lexical_reads(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    check_only = Relation(
        "check",
        2,
        {},
        directions=frozenset({"check"}),
        complete=False,
    )
    rules = inventory((check_only,))
    lexical = source("A", "B")

    def refuse_chart(*args: object, **kwargs: object) -> None:
        del args, kwargs
        msg = "chart construction ran before validation"
        raise AssertionError(msg)

    monkeypatch.setattr(engine, "PackedForest", refuse_chart)

    with pytest.raises(ValueError, match="requires a checked proposer"):
        run(rules, lexical)

    assert lexical.calls == 0


def test_lexical_source_and_scorer_weights_are_distinct_factors() -> None:
    lexical = Source(((0, Entry("one", "S", weight=log(0.25)), log(0.5)),))
    rules = inventory(())
    chart = run(rules, lexical)
    (root,) = tuple(RootGoal(TargetSet((_category("S"),), EXACT)).roots(chart))

    assert isclose(exp(chart.inside(root)), 0.125)
    assert chart.edges(root)[0].rule_name == f"{LEXICAL_TAG_PREFIX}one"


def test_parse_accepts_a_declared_parallel_cell_count() -> None:
    combine = Relation("combine", 2, {("A", "B"): ("S",)})
    rules = inventory((combine,))

    serial = run(rules, source("A", "B"), threads=1)
    parallel = run(rules, source("A", "B"), threads=2)

    assert parallel.categories_at((0, 2)) == serial.categories_at((0, 2))
    assert parallel.items_at((0, 2)) == serial.items_at((0, 2))
    assert tuple(parallel.edges(key) for key in parallel.items_at((0, 2))) == tuple(
        serial.edges(key) for key in serial.items_at((0, 2))
    )


def test_each_feature_variant_combines_and_keeps_its_own_inside_weight() -> None:
    t = featured("t")
    bare = featured("fwd", featured("bwd", featured("x[b]"), t), t)
    progressive = featured("fwd", featured("bwd", featured("x[ng]"), t), t)
    chart = featured_parse(
        ((0, t, 1), (1, bare, 2), (1, progressive, 3), (2, t, 1)),
        featured_rules(),
    )
    erasure = FeatureErasure()

    assert erasure.equivalent(bare, progressive)
    assert chart.categories_at((1, 3)) == (
        (featured("bwd", featured("x[b]"), t), 2),
        (featured("bwd", featured("x[ng]"), t), 3),
    )
    assert chart.categories_at((0, 3)) == (
        (featured("x[b]"), 2),
        (featured("x[ng]"), 3),
    )
    assert [
        key.reading
        for key in RootGoal(TargetSet((featured("x"),), erasure)).roots(chart)
    ] == [
        featured("x[b]"),
        featured("x[ng]"),
    ]


def test_one_exact_category_reached_through_two_variants_sums_both() -> None:
    t = featured("t")
    plain = featured("bwd", featured("x[b]"), t)
    marked = featured("bwd", featured("x[b]"), featured("x[ng]"))
    chart = featured_parse(
        ((0, t, 2), (0, featured("x[ng]"), 5), (1, plain, 1), (1, marked, 1)),
        featured_rules(),
    )

    assert chart.categories_at((0, 2)) == ((featured("x[b]"), 7),)
    assert len(chart.items_at((0, 2))) == 1


def featured_category(depth: int) -> st.SearchStrategy[Term]:
    atoms = st.sampled_from(("x", "x[b]", "x[ng]", "t")).map(featured)

    if depth == 0:
        return atoms

    smaller = featured_category(depth - 1)

    return st.one_of(
        atoms,
        st.builds(featured, st.sampled_from(("fwd", "bwd")), smaller, smaller),
    )


FEATURED_LEXICONS: Final = st.integers(2, 4).flatmap(
    lambda length: st.lists(
        st.tuples(
            st.integers(0, length - 1),
            featured_category(2),
            st.integers(1, 3),
        ),
        min_size=length,
        max_size=2 * length + 2,
        unique_by=lambda row: (row[0], row[1]),
    ).filter(lambda rows: {row[0] for row in rows} == set(range(length)))
)


def closure_misses(
    chart: PackedForest[Term, int], rules: RuleSet[Term]
) -> list[tuple[Cell, Term]]:
    """Return each exact conclusion of two adjacent items the chart lacks."""
    missing: list[tuple[Cell, Term]] = []
    n = chart.index.n

    for start in range(n):
        for split in range(start + 1, n):
            for end in range(split + 1, n + 1):
                held = {term for term, _ in chart.categories_at((start, end))}

                for left in chart.items_at((start, split)):
                    for right in chart.items_at((split, end)):
                        for rule in rules.of_arity(2):
                            missing.extend(
                                ((start, end), output)
                                for output in rule.apply((left.reading, right.reading))
                                if output not in held
                            )

    return missing


@settings(max_examples=60, deadline=None)
@given(FEATURED_LEXICONS)
def test_the_chart_is_closed_over_exact_categories_and_every_step_is_sound(
    rows: list[tuple[int, Term, int]],
) -> None:
    rules = featured_rules((0, 1))
    chart = featured_parse(rows, rules)

    assert closure_misses(chart, rules) == []

    for span in chart.index.spans_by_length():
        for key in chart.items_at(span):
            for edge in chart.edges(key):
                assert edge.output is key.reading

                if edge.children:
                    rule = next(rule for rule in rules if rule.name == edge.rule_name)
                    premises = tuple(child.reading for child in edge.children)

                    assert key.reading in tuple(rule.apply(premises))


class FeaturedSource:
    """Weighted lexical categories, any number per position."""

    def __init__(self, rows: Sequence[tuple[int, Term, int]]) -> None:
        self._rows = tuple(rows)

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    def axioms(
        self, tokens: Sequence[str], /
    ) -> Iterable[tuple[int, Entry[Term], int]]:
        del tokens
        return tuple(
            (position, Entry(str(position), category), weight)
            for position, category, weight in self._rows
        )


def featured_rules(
    orders: Sequence[int] = (0,),
    normal_form: NormalForm[Term] | None = None,
) -> RuleSet[Term]:
    rules = featured_compiled(
        *(
            ApplicationSpec(view=slash)
            if order == 0
            else CompositionSpec(view=slash, order=order, direction="unrestricted")
            for order in orders
            for slash in ("fwd", "bwd")
        )
    )

    return RuleSet(
        "featured",
        rules,
        rules[0].matching,
        NoNormalForm[Term]() if normal_form is None else normal_form,
    )


def featured_parse(
    rows: Sequence[tuple[int, Term, int]],
    rules: RuleSet[Term],
    *,
    unary_depth: int = 0,
) -> PackedForest[Term, int]:
    length = 1 + max(position for position, _, _ in rows)

    return parse(
        tuple("w" for _ in range(length)),
        rules=rules,
        source=FeaturedSource(rows),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), rules.matching)),
        unary_depth=unary_depth,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def test_a_premise_pair_concludes_from_its_own_features_wherever_it_recurs() -> None:
    (application,) = featured_compiled(ApplicationSpec(view="bwd"))
    erasure = application.matching
    rules = RuleSet("featured", (application,), erasure, NoNormalForm[Term]())
    t = featured("t")
    bare = featured("bwd", featured("x[b]"), t)
    progressive = featured("bwd", featured("x[ng]"), t)
    chart = parse(
        ("a", "b", "c", "d"),
        rules=rules,
        source=FeaturedSource(
            tuple(
                (position, category, 1)
                for position, category in enumerate((t, bare, t, progressive))
            )
        ),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), erasure)),
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    outputs = {
        span: [edge.output for key in chart.items_at(span) for edge in chart.edges(key)]
        for span in ((0, 2), (2, 4))
    }

    assert erasure.equivalent(bare, progressive)
    assert outputs == {(0, 2): [featured("x[b]")], (2, 4): [featured("x[ng]")]}


def _relation_outputs(
    rule: Relation,
    inputs: tuple[str, ...],
    span: Cell,
    proposer: Beam | None,
) -> tuple[str, ...]:
    terms = tuple(_category(name) for name in inputs)
    native = tuple(rule.lowering.apply(terms)) if "forward" in rule.directions else ()
    proposed = engine._proposed(rule, terms, span, proposer, {})
    return tuple(_label(term) for term in dict.fromkeys((*native, *proposed)))


def test_incomplete_forward_combines_native_and_admitted_proposals() -> None:
    rule = Relation(
        "partial",
        2,
        {("A", "B"): ("X", "x")},
        CASEFOLD,
        complete=False,
        admitted={("A", "B"): ("Y",)},
    )
    beam = Beam(("X", "Y", "y", "Z"))

    outputs = _relation_outputs(rule, ("A", "B"), (0, 2), beam)
    again = _relation_outputs(rule, ("A", "B"), (0, 2), beam)

    assert outputs == ("X", "x", "Y", "y")
    assert again == outputs
    assert beam.calls == 2


def test_check_only_outputs_consume_admitted_proposals_each_time() -> None:
    rule = Relation(
        "checked",
        2,
        {},
        CASEFOLD,
        directions=frozenset({"check"}),
        complete=False,
        admitted={("A", "B"): ("Y",)},
    )
    beam = Beam(("Y", "y", "Z"))

    assert _relation_outputs(rule, ("A", "B"), (0, 2), beam) == ("Y", "y")
    assert _relation_outputs(rule, ("A", "B"), (0, 2), beam) == ("Y", "y")
    assert beam.calls == 2


def test_a_shared_proposer_keeps_same_named_inventories_independent() -> None:
    left = Relation(
        "shared",
        2,
        {},
        directions=frozenset({"check"}),
        complete=False,
        admitted={("A", "B"): ("X",)},
    )
    right = Relation(
        "shared",
        2,
        {},
        directions=frozenset({"check"}),
        complete=False,
        admitted={("A", "B"): ("Y",)},
    )
    beam = Beam(("X", "Y"))
    left_forest = run(inventory((left,)), source("A", "B"), proposer=beam)
    right_forest = run(inventory((right,)), source("A", "B"), proposer=beam)

    assert tuple(_label(key.reading) for key in left_forest.items_at((0, 2))) == ("X",)
    assert tuple(_label(key.reading) for key in right_forest.items_at((0, 2))) == ("Y",)


@pytest.mark.parametrize("kind", ["complete", "partial", "checked"])
def test_each_direction_path_inserts_every_exact_conclusion(kind: str) -> None:
    beam: Beam | None = None

    if kind == "complete":
        rule = Relation("rule", 2, {("A", "B"): ("S", "s")}, CASEFOLD)
    elif kind == "partial":
        rule = Relation(
            "rule",
            2,
            {("A", "B"): ("S", "s")},
            CASEFOLD,
            complete=False,
            admitted={("A", "B"): ("T",)},
        )
        beam = Beam(("T", "t"))
    else:
        rule = Relation(
            "rule",
            2,
            {},
            CASEFOLD,
            directions=frozenset({"check"}),
            complete=False,
            admitted={("A", "B"): ("S",)},
        )
        beam = Beam(("S", "s"))

    chart = run(inventory((rule,), CASEFOLD), source("A", "B"), proposer=beam)
    readings = tuple(_label(key.reading) for key in chart.items_at((0, 2)))

    assert readings == (("S", "s", "T", "t") if kind == "partial" else ("S", "s"))
    assert all(len(chart.edges(key)) == 1 for key in chart.items_at((0, 2)))


def test_a_rule_matching_more_finely_than_the_grammar_keeps_readings_apart() -> None:
    combine = Relation("combine", 2, {("A", "B"): ("S", "s")}, CASEFOLD)
    exact = Relation("exact", 1, {("S",): ("T",)}, EXACT)

    fine = run(inventory((combine, exact), CASEFOLD), source("A", "B"))
    coarse = run(inventory((combine,), CASEFOLD), source("A", "B"))

    assert sorted(_label(key.reading) for key in fine.items_at((0, 2))) == [
        "S",
        "T",
        "s",
    ]
    assert tuple(_label(key.reading) for key in coarse.items_at((0, 2))) == (
        "S",
        "s",
    )


def test_indexing_preserves_rule_order_and_reduces_trials() -> None:
    wanted = RuleProbe(2, ((frozenset({"A"}), frozenset({"B"})),))
    missed = RuleProbe(2, ((frozenset({"C"}), frozenset({"C"})),))
    plain_rules = (
        Relation("first", 2, {("A", "B"): ("X",)}, probe=wanted),
        Relation("missed", 2, {}, probe=missed),
        Relation("second", 2, {("A", "B"): ("Y",)}, probe=wanted),
    )
    indexed_rules = (
        Relation("first", 2, {("A", "B"): ("X",)}, probe=wanted),
        Relation("missed", 2, {}, probe=missed),
        Relation("second", 2, {("A", "B"): ("Y",)}, probe=wanted),
    )
    plain_scorer = Scorer()
    indexed_scorer = Scorer()
    plain = run(inventory(plain_rules), source("A", "B"), scorer=plain_scorer)
    index = RuleIndex(indexed_rules, cast("HeadIndex[Term]", Heads()))
    indexed = run(
        inventory(indexed_rules, index=index),
        source("A", "B"),
        scorer=indexed_scorer,
    )

    assert tuple(_label(key.reading) for key in plain.items_at((0, 2))) == ("X", "Y")
    assert tuple(_label(key.reading) for key in indexed.items_at((0, 2))) == (
        "X",
        "Y",
    )
    assert plain_scorer.trials == ["first", "second"]
    assert indexed_scorer.trials == plain_scorer.trials


def test_unary_indexing_preserves_closure_and_reduces_trials() -> None:
    wanted_a = RuleProbe(1, ((frozenset({"A"}),),))
    wanted_b = RuleProbe(1, ((frozenset({"B"}),),))
    missed = RuleProbe(1, ((frozenset({"C"}),),))
    plain_rules = (
        Relation("first", 1, {("A",): ("B",)}, probe=wanted_a),
        Relation("missed", 1, {}, probe=missed),
        Relation("second", 1, {("B",): ("Y",)}, probe=wanted_b),
    )
    indexed_rules = (
        Relation("first", 1, {("A",): ("B",)}, probe=wanted_a),
        Relation("missed", 1, {}, probe=missed),
        Relation("second", 1, {("B",): ("Y",)}, probe=wanted_b),
    )
    plain = run(inventory(plain_rules), source("A"))
    index = RuleIndex(indexed_rules, cast("HeadIndex[Term]", Heads()))
    indexed = run(inventory(indexed_rules, index=index), source("A"))
    plain_items = plain.items_at((0, 1))
    indexed_items = indexed.items_at((0, 1))

    assert indexed_items == plain_items
    assert [indexed.term(key) for key in indexed_items] == [
        plain.term(key) for key in plain_items
    ]
    assert [indexed.inside(key) for key in indexed_items] == [
        plain.inside(key) for key in plain_items
    ]
    assert [indexed.edges(key) for key in indexed_items] == [
        plain.edges(key) for key in plain_items
    ]


def test_native_chart_does_not_request_python_unary_candidates(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Select indexed unary rules without returning to the Python index."""
    wanted = RuleProbe(1, ((frozenset({"A"}),),))
    missed = RuleProbe(1, ((frozenset({"C"}),),))
    rules = (
        Relation("wanted", 1, {("A",): ("B",)}, probe=wanted),
        Relation("missed", 1, {}, probe=missed),
    )
    index = RuleIndex(rules, cast("HeadIndex[Term]", Heads()))

    def refuse_python_selection(*values: object) -> None:
        del values
        msg = "the chart requested Python unary candidates"
        raise AssertionError(msg)

    monkeypatch.setattr(RuleIndex, "unary_candidates", refuse_python_selection)
    chart = run(inventory(rules, index=index), source("A"))

    assert tuple(_label(key.reading) for key in chart.items_at((0, 1))) == ("A", "B")


def test_an_index_over_unary_and_binary_rules_closes_unary_steps() -> None:
    unary = Relation("unary", 1, {("A",): ("B",)})
    binary = Relation("binary", 2, {})
    index = RuleIndex((unary, binary), cast("HeadIndex[Term]", Heads()))
    chart = run(inventory((unary, binary), index=index), source("A"))

    assert tuple(_label(key.reading) for key in chart.items_at((0, 1))) == ("A", "B")


def test_complete_forward_memo_is_shared_across_normal_form_tags() -> None:
    left_a = Relation("left-a", 2, {("A", "B"): ("X",)})
    left_b = Relation("left-b", 2, {("A", "B"): ("X",)})
    finish = Relation("finish", 2, {("X", "C"): ("S",)})
    tags = cast("NormalForm[Term]", RuleTags(("left-a", "left-b", "finish")))
    chart = run(
        inventory((left_a, left_b, finish), normal_form=tags),
        source("A", "B", "C"),
    )

    assert len(chart.items_at((0, 2))) == 2
    assert len(chart.items_at((0, 3))) == 1
    assert len(chart.edges(chart.items_at((0, 3))[0])) == 2


def test_normal_form_tags_refine_item_keys() -> None:
    first = Relation("first", 2, {("A", "B"): ("S",)})
    second = Relation("second", 2, {("A", "B"): ("S",)})
    tagged = cast("NormalForm[Term]", RuleTags(("first", "second")))
    with_tags = run(inventory((first, second), normal_form=tagged), source("A", "B"))

    untagged_first = Relation("first", 2, {("A", "B"): ("S",)})
    untagged_second = Relation("second", 2, {("A", "B"): ("S",)})
    without_tags = run(inventory((untagged_first, untagged_second)), source("A", "B"))

    assert tuple(key.nf_tag for key in with_tags.items_at((0, 2))) == (
        "first",
        "second",
    )
    assert len(without_tags.items_at((0, 2))) == 1
    assert len(without_tags.edges(without_tags.items_at((0, 2))[0])) == 2


def test_outputs_are_normalized_before_filtering_and_insertion() -> None:
    rule = Relation("lower", 2, {("A", "B"): ("S", "s")})
    normalizer = Lowercase()
    chart = run(inventory((rule,)), source("A", "B"), normalizer=normalizer)
    (root,) = chart.items_at((0, 2))

    assert normalizer.seen == ["S", "s"]
    assert _label(root.reading) == "s"
    assert _label(chart.term(root)) == "s"
    assert len(chart.edges(root)) == 1


def test_the_sweep_uses_decompositions_for_ternary_rules() -> None:
    rule = Relation("ternary", 3, {("A", "B", "C"): ("S",)})
    chart = run(inventory((rule,)), source("A", "B", "C"))
    (root,) = chart.items_at((0, 3))
    (edge,) = chart.edges(root)

    assert edge.children == tuple(
        chart.items_at((position, position + 1))[0] for position in range(3)
    )
    assert edge.split is None


def test_the_four_word_b1_inventory_has_one_weighted_root() -> None:
    forward = relations(
        "forward-application",
        2,
        {("NP/N", "N"): ("NP",), ("(S\\NP)/NP", "NP"): ("S\\NP",)},
    )
    backward = Relation("backward-application", 2, {("NP", "S\\NP"): ("S",)})
    b1 = Relation("forward-composition-b1", 2, {("NP/N", "N/N"): ("NP/N",)}, order=1)
    source_factors = (0.8, 0.7, 0.6, 0.5)
    scorer_factors = (0.9, 0.85, 0.75, 0.65)
    categories = ("NP/N", "N", "(S\\NP)/NP", "NP")
    lexical = Source(
        tuple(
            (
                position,
                Entry(token, category, weight=log(scorer_factor)),
                log(source_factor),
            )
            for position, (token, category, source_factor, scorer_factor) in enumerate(
                zip(
                    ("the", "dog", "likes", "Mary"),
                    categories,
                    source_factors,
                    scorer_factors,
                    strict=True,
                )
            )
        )
    )
    rule_factors = {
        "forward-application-0": 0.55,
        "forward-application-1": 0.55,
        "backward-application": 0.45,
        "forward-composition-b1": 0.35,
    }
    scorer = Scorer({name: log(weight) for name, weight in rule_factors.items()})
    chart = run(inventory((*forward, backward, b1)), lexical, scorer=scorer)
    roots = tuple(RootGoal(TargetSet((_category("S"),), EXACT)).roots(chart))
    expected = 0.8 * 0.7 * 0.6 * 0.5 * 0.9 * 0.85 * 0.75 * 0.65 * 0.55 * 0.55 * 0.45

    assert len(roots) == 1
    assert isclose(exp(chart.inside(roots[0])), expected)


def test_an_edge_records_the_spine_order_its_rule_declares() -> None:
    b2 = Relation("composition-b2", 2, {("A", "B"): ("S",)}, order=2)
    chart = run(inventory((b2,)), source("A", "B"))
    (root,) = RootGoal(TargetSet((_category("S"),), EXACT)).roots(chart)

    assert tuple(edge.order for edge in chart.edges(root)) == (2,)


def test_lexical_prefix_collisions_are_refused_before_source_reads() -> None:
    lexical = source("A", "B")
    rule = Relation(f"{LEXICAL_TAG_PREFIX}bad", 2, {("A", "B"): ("S",)})

    with pytest.raises(ValueError, match="reserved lexical prefix"):
        run(inventory((rule,)), lexical)

    assert lexical.calls == 0
    assert LEXICAL_TAG_PREFIX == "lex:"


def test_the_engine_imports_no_concrete_rules_or_corpora() -> None:
    tree = ast.parse(Path(engine.__file__).read_text(encoding="utf-8"))
    modules = {
        node.module
        for node in ast.walk(tree)
        if isinstance(node, ast.ImportFrom) and node.module is not None
    }

    assert not any("corpus" in module or "corpora" in module for module in modules)
    assert not any(module.startswith("lofi.judge._") for module in modules)


def test_unary_closure_recomputes_convergent_paths_in_every_rule_order() -> None:
    declarations = (
        ("a-b", ("A",), ("B",)),
        ("a-c", ("A",), ("C",)),
        ("c-b", ("C",), ("B",)),
        ("b-d", ("B",), ("D",)),
    )

    for ordering in permutations(declarations):
        rules = tuple(
            Relation(name, 1, {inputs: outputs}) for name, inputs, outputs in ordering
        )
        chart = run_counting(inventory(rules), ("A",), unary_depth=4)
        by_reading = {_label(key.reading): key for key in chart.items_at((0, 1))}

        inside = parse(
            ("A",),
            rules=inventory(rules),
            source=UnitSource(("A",)),
            semiring=Counting(),
            scorer=UnitScorer(),
            goal=RootGoal(TargetSet((_category("S"),), EXACT)),
            unary_depth=4,
            policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
        )

        assert chart.inside(by_reading["B"]) == 2
        assert chart.inside(by_reading["D"]) == 2
        assert len(chart.edges(by_reading["B"])) == 2
        assert not chart.truncated
        assert inside.categories_at((0, 1)) == chart.categories_at((0, 1))
        assert not inside.truncated


@pytest.mark.parametrize(
    "relations",
    [
        (("self", ("A",), ("A",)),),
        (
            ("a-b", ("A",), ("B",)),
            ("b-c", ("B",), ("C",)),
            ("c-a", ("C",), ("A",)),
        ),
    ],
    ids=("self-loop", "long-cycle"),
)
def test_unary_cycles_are_excluded_and_reported(
    relations: tuple[tuple[str, tuple[str, ...], tuple[str, ...]], ...],
) -> None:
    rules = tuple(
        Relation(name, 1, {inputs: outputs}) for name, inputs, outputs in relations
    )
    chart = run_counting(inventory(rules), ("A",), unary_depth=8)

    assert chart.truncated
    assert all(
        edge.output != chart.term(edge.children[0])
        for key in chart.items_at((0, 1))
        for edge in chart.edges(key)
        if edge.children
    )
    assert chart.num_edges == len(chart.items_at((0, 1)))


def test_check_only_unary_raising_stops_at_the_depth_cap_within_its_cell() -> None:
    rule = Relation(
        "type-raising:fwd:targets=*",
        1,
        {},
        directions=frozenset({"check"}),
        complete=False,
        admitted={
            ("A",): ("B",),
            ("B",): ("C",),
            ("C",): ("D",),
            ("D",): ("E",),
        },
    )
    chart = run_counting(
        inventory((rule,)),
        ("A", "X"),
        proposer=Beam(("B", "C", "D", "E")),
        unary_depth=3,
    )

    assert chart.truncated
    assert {_label(key.reading) for key in chart.items_at((0, 1))} == {
        "A",
        "B",
        "C",
        "D",
    }
    assert {_label(key.reading) for key in chart.items_at((1, 2))} == {"X"}
    assert chart.items_at((0, 2)) == ()


def test_a_check_only_unary_rule_over_a_ground_premise_checks_proposals() -> None:
    (rule,) = declared_rules(
        _RELATION_SIGNATURE,
        RuleSetSpec(
            name="ground-check",
            rules=(
                PatternRuleSpec(
                    name="ground-check",
                    premises=(ConstantPatternSpec(name="A"),),
                    conclusion=TermPatternSpec(name="X0", sort="Cat"),
                    head_policy="pass-through",
                ),
            ),
        ),
    ).rule_set.rules

    assert rule.directions == frozenset({"check", "residual"})

    chart = run(inventory((rule,)), source("A", "X"), proposer=Beam(("B", "C")))

    assert {_label(key.reading) for key in chart.items_at((0, 1))} == {"A", "B", "C"}
    assert {_label(key.reading) for key in chart.items_at((1, 2))} == {"X"}


def test_finite_targeted_unary_raising_exhausts_before_a_large_cap() -> None:
    rule = Relation(
        "type-raising:fwd:targets=B",
        1,
        {("A",): ("B",)},
    )
    chart = run_counting(inventory((rule,)), ("A",), unary_depth=8)

    assert {_label(key.reading) for key in chart.items_at((0, 1))} == {"A", "B"}
    assert not chart.truncated


def test_repeated_targeted_raising_stops_at_the_depth_under_any_normal_form() -> None:
    (rule,) = featured_compiled(
        TypeRaisingSpec(view="fwd", targets=("t",), arguments=None),
        matching=IdentityMatchingSpec(),
    )
    unrestricted = run_raising(
        rule,
        NoNormalForm(),
        unary_depth=3,
    )
    guarded = run_raising(
        rule,
        HockenmaierBiskNormalForm(),
        unary_depth=3,
    )

    assert unrestricted.truncated
    assert guarded.truncated
    assert len(unrestricted.items_at((0, 1))) == 4
    assert guarded.categories_at((0, 1)) == unrestricted.categories_at((0, 1))
    assert tuple(key.nf_tag for key in guarded.items_at((0, 1))) == (
        "OTHER",
        "FTR",
        "FTR",
        "FTR",
    )


def test_real_untargeted_raising_uses_the_proposer_until_the_cap() -> None:
    (rule,) = featured_compiled(
        TypeRaisingSpec(view="fwd", arguments=None), matching=IdentityMatchingSpec()
    )
    beam = RaisingBeam()
    chart = run_raising(
        rule,
        NoNormalForm(),
        proposer=beam,
        unary_depth=3,
    )

    assert chart.truncated
    assert len(chart.items_at((0, 1))) == 4
    assert chart.items_at((0, 2)) == ()
    assert beam.spans == [(0, 1)] * 4


@pytest.mark.parametrize("depth", [-1, True, 1.5, "3"])
def test_invalid_unary_depth_is_refused_before_source_reads(
    depth: float | str,
) -> None:
    lexical = source("A")

    with pytest.raises((TypeError, ValueError), match="unary_depth"):
        run(inventory(()), lexical, unary_depth=cast("int", depth))

    assert lexical.calls == 0


@pytest.mark.parametrize("threads", [0, -1, True, 1.5, "2"])
def test_invalid_thread_count_is_refused_before_source_reads(
    threads: float | str,
) -> None:
    lexical = source("A")

    with pytest.raises((TypeError, ValueError), match="threads"):
        run(inventory(()), lexical, threads=cast("int", threads))

    assert lexical.calls == 0


ALPHABET: Final = ("A", "B", "C")


class ByOutput:
    """Weights a step by the category it concluded."""

    @property
    def lowering(self) -> SpanFree:
        return SpanFree()

    def __init__(self, weights: Mapping[str, float]) -> None:
        self._weights = dict(weights)

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
        del rule, inputs, span, split
        return self._weights[_label(output)]


RANKED: Final = ByOutput({"A": -1.0, "B": -2.0, "C": -3.0})


class KeepAll:
    """Keeps every item of every cell."""

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[Term], float]], /
    ) -> tuple[ItemKey[Term], ...]:
        del span

        return tuple(key for key, _ in items)


class Watched:
    """Records the cells it was asked about and defers to another pruner."""

    def __init__(self, inner: CellPruner[Term, float]) -> None:
        self._inner = inner
        self.spans: list[Cell] = []
        self.dropped: dict[Cell, tuple[float, ...]] = {}

    def keep(
        self, span: Cell, items: Sequence[tuple[ItemKey[Term], float]], /
    ) -> Sequence[ItemKey[Term]]:
        self.spans.append(span)
        kept = self._inner.keep(span, items)
        dropped = tuple(weight for key, weight in items if key not in kept)

        if dropped:
            self.dropped[span] = dropped

        return kept


def spreading(
    tokens: int,
    *,
    categories: Sequence[str] = ("A",),
    pruner: CellPruner[Term, float] | None = None,
    prune_lexical: bool = False,
    semiring: Semiring[float] | None = None,
    measure_pruned_mass: bool = False,
) -> PackedForest[Term, float]:
    rules = inventory(
        relations("m", 2, {(x, y): ALPHABET for x in ALPHABET for y in ALPHABET})
    )
    rows = tuple(
        (position, Entry(str(position), category, weight=0.0), 0.0)
        for position in range(tokens)
        for category in categories
    )

    return parse(
        tuple(str(position) for position in range(tokens)),
        rules=rules,
        source=Source(rows),
        semiring=Viterbi() if semiring is None else semiring,
        scorer=RANKED,
        goal=spreading_goal(rules),
        unary_depth=0,
        pruner=pruner,
        prune_lexical=prune_lexical,
        measure_pruned_mass=measure_pruned_mass,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def spreading_goal(rules: RuleSet[Term]) -> RootGoal[Term]:
    return RootGoal(TargetSet((_category("A"),), rules.matching))


def census(
    forest: PackedForest[Term, float],
) -> tuple[tuple[Cell, tuple[object, ...]], ...]:
    return tuple(
        (
            span,
            tuple(
                (key, forest.inside(key), forest.edges(key))
                for key in forest.items_at(span)
            ),
        )
        for span in forest.index.regions_by_size()
    )


def readings_by_cell(
    forest: PackedForest[Term, float],
) -> dict[Cell, tuple[str, ...]]:
    return {
        span: tuple(_label(key.reading) for key in forest.items_at(span))
        for span in forest.index.regions_by_size()
    }


def rooted(forest: PackedForest[Term, float]) -> ItemKey[Term]:
    span = (0, forest.index.n)

    return next(key for key in forest.items_at(span) if key.reading is _category("A"))


def total(tree: Derivation[Term, float]) -> float:
    return tree.edge.weight + sum(total(child) for child in tree.children)


def spine(tree: Derivation[Term, float]) -> tuple[str, str, tuple[object, ...]]:
    return (
        _label(tree.key.reading),
        tree.edge.rule_name,
        tuple(spine(child) for child in tree.children),
    )


def best(forest: PackedForest[Term, float]) -> Derivation[Term, float]:
    return max(forest.derivations(rooted(forest)), key=total)


def test_a_pruner_that_keeps_everything_builds_the_forest_no_pruner_builds() -> None:
    plain = spreading(4)
    kept = spreading(4, pruner=KeepAll())

    assert census(kept) == census(plain)
    assert not plain.truncated
    assert not kept.truncated


def test_a_beam_bounds_every_cell_and_keeps_the_heaviest_readings() -> None:
    forest = spreading(4, pruner=TopKPruner[Term](2))
    cells = readings_by_cell(forest)

    assert max(len(readings) for readings in cells.values()) == 2
    assert cells[0, 4] == ("A", "B")
    assert cells[0, 1] == ("A",)
    assert forest.truncated


def test_a_beam_as_wide_as_the_cell_leaves_the_forest_untouched() -> None:
    plain = spreading(4)
    widest = max(len(plain.items_at(span)) for span in plain.index.regions_by_size())
    beamed = spreading(4, pruner=TopKPruner[Term](widest))

    assert widest == len(ALPHABET)
    assert census(beamed) == census(plain)
    assert spine(best(beamed)) == spine(best(plain))
    assert total(best(beamed)) == total(best(plain))
    assert not beamed.truncated


def test_a_beam_of_one_keeps_the_derivation_viterbi_would_have_chosen() -> None:
    plain = spreading(4)
    narrow = spreading(4, pruner=TopKPruner[Term](1))

    assert narrow.inside(rooted(narrow)) == plain.inside(rooted(plain))
    assert total(best(narrow)) == total(best(plain))
    assert all(
        len(narrow.items_at(span)) == 1 for span in narrow.index.regions_by_size()
    )


def test_a_parse_declaration_drives_the_engine_components_together() -> None:
    settings = ParseSpec.model_validate(
        {
            "retain": "forest",
            "reachability": "all",
            "strategy": {"kind": "cky"},
            "span_index": {"kind": "contiguous"},
            "chart_store": {"kind": "native"},
            "pruner": {"kind": "top-k", "width": 1},
            "semiring": {"kind": "viterbi"},
        }
    ).build()
    rules = inventory((Relation("combine", 2, {("A", "B"): ("S",)}),))
    chart = parse(
        ("a", "b"),
        rules=rules,
        source=source("A", "B"),
        semiring=cast("Semiring[float]", settings.semiring),
        scorer=Scorer(),
        goal=RootGoal(TargetSet((_category("S"),), rules.matching)),
        unary_depth=0,
        pruner=cast("CellPruner[Term, float]", settings.pruner),
        prune_lexical=settings.prune_lexical,
        strategy=settings.strategy,
        span_index=settings.span_index,
        chart_store=settings.chart_store,
        policy=cast("ParsePolicy[Literal[Retain.FOREST]]", settings.policy),
    )

    assert chart.semiring.name == "viterbi"
    assert tuple(_label(key.reading) for key in chart.items_at((0, 2))) == ("S",)


def test_a_pruned_forest_records_the_weight_each_cell_lost_in_its_semiring() -> None:
    watched = Watched(TopKPruner[Term](1))
    forest = spreading(4, pruner=watched)

    assert watched.dropped
    assert dict(forest.pruned) == {
        span: max(weights) for span, weights in watched.dropped.items()
    }
    assert forest.truncated


def test_a_log_inside_forest_sums_the_weight_each_cell_lost() -> None:
    watched = Watched(TopKPruner[Term](1))
    forest = spreading(4, pruner=watched, semiring=LogInside())

    assert set(forest.pruned) == set(watched.dropped)
    for span, weights in watched.dropped.items():
        assert isclose(forest.pruned[span], log(sum(exp(weight) for weight in weights)))


def test_an_unpruned_forest_records_no_pruned_weight_or_mass() -> None:
    plain = spreading(4)
    kept = spreading(4, pruner=KeepAll())

    assert dict(plain.pruned) == dict(kept.pruned) == {}
    assert plain.pruned_mass is None
    assert kept.pruned_mass is None


def test_a_measured_beam_reports_the_exact_probability_mass_it_lost() -> None:
    plain = spreading(4, semiring=LogInside())
    beamed = spreading(
        4,
        pruner=TopKPruner[Term](1),
        semiring=LogInside(),
        measure_pruned_mass=True,
    )
    unmeasured = spreading(4, pruner=TopKPruner[Term](1), semiring=LogInside())
    goal = spreading_goal(inventory(()))
    mass = beamed.pruned_mass

    assert mass is not None
    assert unmeasured.pruned_mass is None
    assert isclose(mass.exhaustive, outside_from_edges(plain, goal).partition)
    assert isclose(mass.retained, outside_from_edges(beamed, goal).partition)
    assert mass.retained < mass.exhaustive
    assert isclose(mass.lost, exp(mass.exhaustive) - exp(mass.retained))
    assert isclose(mass.lost_fraction, mass.lost / exp(mass.exhaustive))


def test_a_measured_beam_that_keeps_everything_lost_no_mass() -> None:
    forest = spreading(
        3, pruner=KeepAll(), semiring=LogInside(), measure_pruned_mass=True
    )
    mass = forest.pruned_mass

    assert mass is not None
    assert mass.lost == 0.0
    assert mass.exhaustive == mass.retained


def test_measuring_pruned_mass_needs_a_pruner_and_log_inside_weights() -> None:
    with pytest.raises(ValueError, match="measures the mass a pruner lost"):
        spreading(3, semiring=LogInside(), measure_pruned_mass=True)

    with pytest.raises(ValueError, match="log-probability weights"):
        spreading(3, pruner=TopKPruner[Term](1), measure_pruned_mass=True)


def test_the_pruner_reads_each_cell_once_and_leaves_the_diagonal_alone() -> None:
    watched = Watched(TopKPruner[Term](2))
    forest = spreading(4, pruner=watched)
    spans = tuple(watched.spans)

    assert len(spans) == len(set(spans)) == 6
    assert set(spans) == {
        span for span in forest.index.regions_by_size() if span[1] - span[0] > 1
    }


def test_the_pruner_reads_the_diagonal_once_it_is_asked_to() -> None:
    watched = Watched(TopKPruner[Term](2))
    forest = spreading(4, pruner=watched, prune_lexical=True)
    spans = tuple(watched.spans)

    assert len(spans) == len(set(spans)) == 10
    assert set(spans) == set(forest.index.regions_by_size())


def test_a_lexical_cell_keeps_every_reading_until_the_beam_is_asked_for_it() -> None:
    plain = spreading(2, categories=ALPHABET, pruner=TopKPruner[Term](1))
    asked = spreading(
        2, categories=ALPHABET, pruner=TopKPruner[Term](1), prune_lexical=True
    )

    assert readings_by_cell(plain)[0, 1] == ALPHABET
    assert readings_by_cell(asked)[0, 1] == ("A",)
    assert readings_by_cell(plain)[0, 2] == ("A",)


class _Unweighted:
    """Gives each axiom and each step the log-weight zero."""

    @property
    def lowering(self) -> UnitEdges:
        return UnitEdges()

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
        del rule, inputs, output, span, split
        return 0.0


class FeaturedHeads:
    """Indexes featured categories by their head operator or atomic name."""

    def __init__(self) -> None:
        signature = FEATURED_STORE.signature
        self._operators = tuple(operator.name for operator in signature.operators)
        self._atoms = signature.vocabulary("Cat").names

    @property
    def codes(self) -> frozenset[int]:
        return frozenset(range(len(self._operators) + len(self._atoms)))

    def term(self, term: Term, /) -> int:
        if term.args:
            return self._operators.index(term.op.name)

        name = _FEATURED.syntax.render(FEATURED_STORE, term)

        return len(self._operators) + self._atoms.index(name)

    def named(self, name: str, /) -> int | None:
        if name in self._operators:
            return self._operators.index(name)

        if name in self._atoms:
            return len(self._operators) + self._atoms.index(name)

        return None


@pytest.mark.parametrize("retain", [Retain.FOREST, Retain.INSIDE])
@pytest.mark.parametrize(
    ("max_arity", "max_size", "refused"),
    [
        (2, None, 0),
        (1, None, 1),
        (None, 5, 0),
        (None, 4, 1),
    ],
)
def test_category_bounds_keep_every_in_bound_derivation_and_report_refusals(
    retain: Retain,
    max_arity: int | None,
    max_size: int | None,
    refused: int,
) -> None:
    """Apply both owner-declared dimensions at lexical and rule insertion."""
    t = featured("t")
    x = featured("x")
    result = featured("fwd", featured("fwd", x, t), t)
    (rule,) = featured_compiled(
        ground_rule("grow", (x,), result), matching=IdentityMatchingSpec()
    )
    matching = rule.matching
    rules = RuleSet("bounded", (rule,), matching, NoNormalForm[Term]())
    chart = parse(
        ("x",),
        rules=rules,
        source=FeaturedSource(((0, x, 1),)),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((result,), matching)),
        category_measure=CategoryMeasure((("fwd", 0), ("bwd", 0))),
        max_category_arity=max_arity,
        max_category_size=max_size,
        unary_depth=1,
        policy=ParsePolicy(retain, Reachability.ALL),
    )

    assert dict(chart.categories_at((0, 1))).get(result) == (1 if not refused else None)
    assert chart.refused_categories == refused
    assert chart.truncated is bool(refused)


def test_category_bounds_need_the_grammar_measure_before_lexical_reads() -> None:
    """Refuse an unmeasured bound before requesting lexical axioms."""
    lexical = source("A")

    with pytest.raises(ValueError, match="declared category measure"):
        parse(
            ("a",),
            rules=inventory(()),
            source=lexical,
            semiring=LogInside(),
            scorer=Scorer(),
            goal=RootGoal(TargetSet((_category("S"),), EXACT)),
            max_category_size=2,
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )

    assert lexical.calls == 0


class Declared(NamedTuple):
    """A generated grammar, lexicon and closure depth."""

    rules: tuple[Rule[Term], ...]
    matching: Matching[Term]
    rows: tuple[tuple[int, Term, int], ...]
    unary_depth: int


def swapped(term: Term) -> Term:
    """Return the category with every bare feature exchanged for progressive."""
    names = {"x[b]": "x[ng]", "x[ng]": "x[b]"}

    if term.args:
        return featured(term.op.name, *(swapped(arg) for arg in term.args))

    name = _FEATURED.syntax.render(FEATURED_STORE, term)

    return featured(names.get(name, name))


@st.composite
def variant_lexicons(
    draw: st.DrawFn, *, max_length: int = 5
) -> tuple[tuple[int, Term, int], ...]:
    """Draw a lexicon over a small pool closed under exchanging features."""
    drawn = draw(st.lists(featured_category(2), min_size=2, max_size=3, unique=True))
    pool = tuple(dict.fromkeys((*drawn, *(swapped(term) for term in drawn))))
    length = draw(st.integers(2, max_length))

    return tuple(
        (position, category, 1)
        for position in range(length)
        for category in draw(
            st.lists(st.sampled_from(pool), min_size=1, max_size=3, unique=True)
        )
    )


@st.composite
def declared_grammars(draw: st.DrawFn) -> Declared:
    matching: MatchingSpec = draw(
        st.sampled_from((FeatureErasureMatchingSpec(), IdentityMatchingSpec()))
    )
    direction = draw(st.sampled_from(COMPOSITION_RESTRICTIONS))
    orders = draw(
        st.lists(st.sampled_from((0, 1, 2, None)), min_size=1, max_size=4, unique=True)
    )
    slashes = draw(
        st.lists(st.sampled_from(("fwd", "bwd")), min_size=1, max_size=2, unique=True)
    )
    crossings = draw(st.lists(st.booleans(), min_size=1, max_size=2, unique=True))
    rules: list[RuleSpec] = [
        ApplicationSpec(view=slash)
        if order == 0
        else CompositionSpec(
            view=slash, order=order, direction=direction, crossed=crossed
        )
        for order in orders
        for slash in slashes
        for crossed in crossings
        if not (crossed and (order == 0 or direction == "unrestricted"))
    ]

    if draw(st.booleans()):
        rules.extend(SubstitutionSpec(view=slash, order=1) for slash in slashes)

    rows = draw(variant_lexicons(max_length=4))
    pool = st.sampled_from(
        tuple(
            dict.fromkeys(
                (*(category for _, category, _ in rows), featured("t"), featured("x"))
            )
        )
    )
    coercions = draw(st.lists(st.tuples(pool, pool), max_size=5, unique=True))
    rules.extend(
        ground_rule(f"coercion-{at}", (premise,), conclusion)
        for at, (premise, conclusion) in enumerate(coercions)
        if premise is not conclusion
    )
    return Declared(
        featured_compiled(*rules, matching=matching),
        featured_matching(matching),
        rows,
        draw(st.integers(0, 2)),
    )


class NamedScorer:
    """Weighs each step by its rule's name, so Viterbi has a choice to make."""

    @property
    def lowering(self) -> SpanFree:
        return SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        del entry
        return -0.25 * position

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del inputs, output, span, split
        return -0.01 * (sum(map(ord, rule.name)) % 97)


@settings(max_examples=4 * SMALL.examples, deadline=None)
@given(witnessed_grammars(SMALL))
def test_a_generated_grammar_derives_its_goal_over_its_witness(
    witnessed: Witnessed,
) -> None:
    language = category_language(
        witnessed.grammar, registered_language("propositional")
    )
    assembly = GrammarIso().forward(witnessed.grammar, language)
    grammar = assembly.grammar

    forest = parse(
        witnessed.sentence,
        rules=grammar.rules,
        source=assembly.lexicon,
        semiring=LogInside(),
        scorer=_Unweighted(),
        goal=grammar.goal,
        unary_depth=grammar.budget.unary_rounds,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )
    roots = tuple(grammar.goal.roots(forest))

    assert roots
    assert all(isfinite(forest.inside(root)) for root in roots)


type Snapshot[W] = tuple[
    bool, tuple[tuple[ItemKey[Term], W, tuple[Edge[Term, W], ...]], ...]
]


def snapshot[W](forest: PackedForest[Term, W]) -> Snapshot[W]:
    return (
        forest.truncated,
        tuple(
            (key, forest.inside(key), forest.edges(key))
            for span in forest.index.regions_by_size()
            for key in forest.items_at(span)
        ),
    )


class UnitRows[W]:
    """A generated lexicon's categories, each at the semiring's unit weight."""

    def __init__(
        self, rows: Sequence[tuple[int, Term, int]], semiring: Semiring[W]
    ) -> None:
        self._rows = tuple(
            (position, Entry(str(position), category), semiring.one)
            for position, category, _ in rows
        )

    @property
    def is_re_callable(self) -> bool:
        return True

    @property
    def clamps(self) -> tuple[Clamp[Term], ...]:
        return ()

    def axioms(self, tokens: Sequence[str], /) -> Iterable[tuple[int, Entry[Term], W]]:
        del tokens
        return self._rows


def parse_declared[W](
    declared: Declared,
    semiring: Semiring[W],
    scorer: EdgeScorer[Term, W],
    *,
    indexed: bool,
) -> PackedForest[Term, W]:
    rules = declared.rules
    length = 1 + max(position for position, _, _ in declared.rows)
    return parse(
        tuple("w" for _ in range(length)),
        rules=RuleSet(
            "declared",
            rules,
            declared.matching,
            NoNormalForm[Term](),
            index=RuleIndex(rules, FeaturedHeads()) if indexed else None,
        ),
        source=UnitRows(declared.rows, semiring),
        semiring=semiring,
        scorer=scorer,
        goal=RootGoal(TargetSet((featured("x"),), declared.matching)),
        unary_depth=declared.unary_depth,
        policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
    )


def inconsistent[W](forest: PackedForest[Term, W]) -> list[ItemKey[Term]]:
    """Return each item whose inside weight is not the sum over its steps."""
    semiring = forest.semiring
    wrong: list[ItemKey[Term]] = []

    for span in forest.index.regions_by_size():
        for key in forest.items_at(span):
            total = semiring.zero

            for edge in forest.edges(key):
                product = edge.weight

                for child in edge.children:
                    product = semiring.times(product, forest.inside(child))

                total = semiring.plus(total, product)

            if total != forest.inside(key):
                wrong.append(key)

    return wrong


@settings(max_examples=120, deadline=None)
@given(declared_grammars())
def test_the_native_memo_and_index_leave_the_forest_unchanged(
    declared: Declared,
) -> None:
    counted = snapshot(parse_declared(declared, Counting(), TermScorer(), indexed=True))
    best = snapshot(parse_declared(declared, Viterbi(), NamedScorer(), indexed=True))
    slow_counted = parse_declared(declared, Counting(), TermScorer(), indexed=False)
    slow_best = parse_declared(declared, Viterbi(), NamedScorer(), indexed=False)

    assert snapshot(slow_counted) == counted
    assert snapshot(slow_best) == best
    assert inconsistent(slow_counted) == []
    assert inconsistent(slow_best) == []


def test_a_repeated_premise_tuple_keeps_the_native_relation_exact() -> None:
    combine = Relation("combine", 2, {("A", "B"): ("C",)})
    forest = run_counting(inventory((combine,)), ("A", "B", "A", "B"))

    assert [_label(forest.term(key)) for key in forest.items_at((0, 2))] == ["C"]
    assert [_label(forest.term(key)) for key in forest.items_at((2, 4))] == ["C"]


class RecordingBeam(Beam):
    """Records the premises of every proposal."""

    def __init__(self, candidates: Sequence[str]) -> None:
        super().__init__(candidates)
        self.inputs: list[tuple[str, ...]] = []

    @override
    def propose(
        self, rule: Rule[Term], inputs: tuple[Term, ...], span: Cell, /
    ) -> Iterator[Term]:
        self.inputs.append(tuple(_label(term) for term in inputs))
        return super().propose(rule, inputs, span)


def test_a_checked_rule_proposes_afresh_wherever_its_premises_recur() -> None:
    checked = Relation(
        "checked",
        2,
        {},
        directions=frozenset({"check"}),
        complete=False,
        admitted={("A", "B"): ("C",)},
    )
    beam = RecordingBeam(("C",))
    run_counting(inventory((checked,)), ("A", "B", "A", "B"), proposer=beam)

    assert beam.inputs.count(("A", "B")) == 2


def test_a_checked_rule_beside_a_complete_one_still_proposes_at_every_span() -> None:
    plain = Relation("plain", 2, {("A", "B"): ("D",)})
    checked = Relation(
        "checked",
        2,
        {},
        directions=frozenset({"check"}),
        complete=False,
        admitted={("A", "B"): ("C",)},
    )
    beam = RecordingBeam(("C",))
    forest = run_counting(
        inventory((plain, checked)), ("A", "B", "A", "B"), proposer=beam
    )

    assert beam.inputs.count(("A", "B")) == 2
    assert [_label(forest.term(key)) for key in forest.items_at((2, 4))] == [
        "C",
        "D",
    ]


class Normalized(NamedTuple):
    """A generated grammar for comparing normal forms, and its lexicon."""

    rules: tuple[Rule[Term], ...]
    matching: Matching[Term]
    rows: tuple[tuple[int, Term, int], ...]
    unary_depth: int


@st.composite
def normal_form_grammars(draw: st.DrawFn, *, unbounded: bool) -> Normalized:
    """Draw a grammar declaring every composition degree up to its largest."""
    matching: MatchingSpec = draw(
        st.sampled_from((FeatureErasureMatchingSpec(), IdentityMatchingSpec()))
    )
    direction = (
        "unrestricted" if unbounded else draw(st.sampled_from(COMPOSITION_RESTRICTIONS))
    )
    largest = draw(st.integers(0, 2))
    degrees: tuple[int | None, ...] = (
        (0, None) if unbounded else tuple(range(largest + 1))
    )
    crossings = draw(st.lists(st.booleans(), min_size=1, max_size=2, unique=True))
    rules: list[RuleSpec] = [
        ApplicationSpec(view=slash)
        if degree == 0
        else CompositionSpec(
            view=slash, order=degree, direction=direction, crossed=crossed
        )
        for degree in degrees
        for slash in ("fwd", "bwd")
        for crossed in crossings
        if not (crossed and (degree == 0 or direction == "unrestricted"))
    ]
    raised = draw(st.lists(st.sampled_from(("fwd", "bwd")), max_size=2, unique=True))
    unary_depth = draw(st.integers(0, 2))
    # Raising a raised category under composition multiplies the categories
    # each longer span holds, so the deepest closure with raising stays short.
    longest = 3 if raised and unary_depth == 2 else 4
    rows = draw(variant_lexicons(max_length=longest))
    categories = tuple(dict.fromkeys(category for _, category, _ in rows))
    rules.extend(
        TypeRaisingSpec(view=slash, targets=("t", "x"), arguments=None)
        for slash in raised
    )
    pool = st.sampled_from((*categories, featured("t")))
    coercions = draw(st.lists(st.tuples(pool, pool), max_size=4, unique=True))
    rules.extend(
        ground_rule(f"coercion-{at}", (premise,), conclusion)
        for at, (premise, conclusion) in enumerate(coercions)
        if premise is not conclusion
    )
    return Normalized(
        featured_compiled(*rules, matching=matching),
        featured_matching(matching),
        rows,
        unary_depth,
    )


def parse_normalized(
    grammar: Normalized, normal_form: NormalForm[Term]
) -> PackedForest[Term, int]:
    return featured_parse(
        grammar.rows,
        RuleSet("normalized", grammar.rules, grammar.matching, normal_form),
        unary_depth=grammar.unary_depth,
    )


def readings(forest: PackedForest[Term, int]) -> frozenset[tuple[Cell, Term]]:
    return frozenset(
        (span, category)
        for span in forest.index.spans_by_length()
        for category, _ in forest.categories_at(span)
    )


def suffixes(forest: PackedForest[Term, int]) -> frozenset[tuple[Cell, Term]]:
    end = forest.index.n

    return frozenset(
        (span, category) for span, category in readings(forest) if span[1] == end
    )


@settings(max_examples=100, deadline=None)
@given(normal_form_grammars(unbounded=False))
def test_hockenmaier_bisk_keeps_every_suffix_category_and_adds_none(
    grammar: Normalized,
) -> None:
    unrestricted = parse_normalized(grammar, NoNormalForm[Term]())
    restricted = parse_normalized(grammar, HockenmaierBiskNormalForm[Term]())

    assert readings(restricted) <= readings(unrestricted)
    assert suffixes(restricted) == suffixes(unrestricted)


@settings(max_examples=60, deadline=None)
@given(normal_form_grammars(unbounded=True))
def test_eisner_keeps_every_suffix_category_under_unbounded_composition(
    grammar: Normalized,
) -> None:
    unrestricted = parse_normalized(grammar, NoNormalForm[Term]())
    restricted = parse_normalized(grammar, EisnerNormalForm[Term]())

    assert readings(restricted) <= readings(unrestricted)
    assert suffixes(restricted) == suffixes(unrestricted)


def test_eisner_completeness_requires_unrestricted_composition() -> None:
    """Distinguish the normal form's closure premise from a restricted rule set."""
    x = featured("x")
    rows = (
        (0, featured("fwd", x, x), 1),
        (1, featured("fwd", featured("fwd", x, x), x), 1),
        (2, featured("fwd", x, x), 1),
        (2, featured("bwd", x, x), 1),
    )
    target = featured("bwd", featured("fwd", x, x), x)

    def grammar(direction: Literal["steedman", "unrestricted"]) -> Normalized:
        rules = featured_compiled(
            *(
                ApplicationSpec(view=slash)
                if degree == 0
                else CompositionSpec(
                    view=slash, order=degree, direction=direction, crossed=crossed
                )
                for degree in (0, None)
                for slash in ("fwd", "bwd")
                for crossed in (False, True)
                if not (crossed and (degree == 0 or direction == "unrestricted"))
            )
        )
        return Normalized(rules, rules[0].matching, rows, 0)

    steedman = grammar("steedman")
    unrestricted = grammar("unrestricted")

    assert ((0, 3), target) in suffixes(
        parse_normalized(steedman, NoNormalForm[Term]())
    )
    assert ((0, 3), target) not in suffixes(
        parse_normalized(steedman, EisnerNormalForm[Term]())
    )
    assert suffixes(
        parse_normalized(unrestricted, EisnerNormalForm[Term]())
    ) == suffixes(parse_normalized(unrestricted, NoNormalForm[Term]()))


@settings(max_examples=60, deadline=None)
@given(normal_form_grammars(unbounded=False))
def test_eisner_adds_no_reading_under_bounded_composition(
    grammar: Normalized,
) -> None:
    unrestricted = parse_normalized(grammar, NoNormalForm[Term]())
    restricted = parse_normalized(grammar, EisnerNormalForm[Term]())

    assert readings(restricted) <= readings(unrestricted)


def test_a_reading_only_a_left_branching_degree_two_chain_derives_survives() -> None:
    t = featured("t")
    x = featured("x")
    rows = (
        (0, featured("fwd", x, t), 1),
        (1, featured("fwd", featured("fwd", t, x), t), 1),
        (2, featured("fwd", featured("fwd", t, x), x), 1),
    )
    rules = featured_rules((0, 1, 2))
    chained = featured("fwd", featured("fwd", featured("fwd", x, x), x), x)

    def suffix_categories(normal_form: NormalForm[Term]) -> frozenset[Term]:
        forest = featured_parse(
            rows,
            RuleSet("chain", rules.rules, rules.matching, normal_form),
        )
        return frozenset(category for category, _ in forest.categories_at((0, 3)))

    assert chained in suffix_categories(NoNormalForm[Term]())
    assert chained in suffix_categories(HockenmaierBiskNormalForm[Term]())
    assert chained not in suffix_categories(EisnerNormalForm[Term]())


def test_a_raised_category_meeting_another_feature_keeps_its_own_reading() -> None:
    x = featured("x")
    rules = featured_compiled(
        ApplicationSpec(view="fwd"),
        ApplicationSpec(view="bwd"),
        TypeRaisingSpec(view="fwd", targets=("t", "x"), arguments=None),
        TypeRaisingSpec(view="bwd", targets=("t", "x"), arguments=None),
    )
    erasure = rules[0].matching
    grammar = Normalized(
        rules,
        erasure,
        ((0, x, 1), (0, featured("fwd", featured("x[b]"), x), 1), (1, x, 1)),
        1,
    )
    unrestricted = parse_normalized(grammar, NoNormalForm[Term]())
    restricted = parse_normalized(grammar, HockenmaierBiskNormalForm[Term]())

    assert ((0, 2), x) in suffixes(unrestricted)
    assert suffixes(restricted) == suffixes(unrestricted)


def test_the_forest_names_the_normal_form_its_steps_passed() -> None:
    rows = (
        (0, featured("t"), 1),
        (1, featured("bwd", featured("x"), featured("t")), 1),
    )
    rules = featured_rules()
    restricted = RuleSet(
        "restricted", rules.rules, rules.matching, HockenmaierBiskNormalForm[Term]()
    )

    assert featured_parse(rows, rules).normal_form == "none"
    assert featured_parse(rows, restricted).normal_form == "hockenmaier-bisk"


def inside_declared[W](
    declared: Declared, semiring: Semiring[W], scorer: EdgeScorer[Term, W]
) -> InsideChart[Term, W]:
    rules = declared.rules
    length = 1 + max(position for position, _, _ in declared.rows)
    return parse(
        tuple("w" for _ in range(length)),
        rules=RuleSet(
            "declared",
            rules,
            declared.matching,
            NoNormalForm[Term](),
            index=RuleIndex(rules, FeaturedHeads()),
        ),
        source=UnitRows(declared.rows, semiring),
        semiring=semiring,
        scorer=scorer,
        goal=RootGoal(TargetSet((featured("x"),), declared.matching)),
        unary_depth=declared.unary_depth,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )


def by_span[W](
    chart: PackedForest[Term, W] | InsideChart[Term, W],
) -> dict[Cell, dict[Term, W]]:
    return {
        span: dict(chart.categories_at(span)) for span in chart.index.regions_by_size()
    }


def assert_close_by_span(
    left: dict[Cell, dict[Term, float]],
    right: dict[Cell, dict[Term, float]],
) -> None:
    """Require equal category support and numerically equal floating weights."""
    assert left.keys() == right.keys()

    for span, left_weights in left.items():
        right_weights = right[span]

        assert left_weights.keys() == right_weights.keys()
        assert all(
            isclose(weight, right_weights[category], rel_tol=1e-12, abs_tol=1e-12)
            for category, weight in left_weights.items()
        )


@settings(max_examples=120, deadline=None)
@given(declared_grammars())
def test_the_inside_chart_weighs_each_exact_category_as_the_forest_does(
    declared: Declared,
) -> None:
    counted = parse_declared(declared, Counting(), TermScorer(), indexed=True)
    best = parse_declared(declared, Viterbi(), NamedScorer(), indexed=True)
    counted_inside = inside_declared(declared, Counting(), TermScorer())
    best_inside = inside_declared(declared, Viterbi(), NamedScorer())

    assert by_span(counted_inside) == by_span(counted)
    assert_close_by_span(by_span(best_inside), by_span(best))
    assert counted_inside.truncated == counted.truncated


def test_inside_and_forest_choose_the_same_edge_from_a_unary_cycle() -> None:
    """Keep unary-cycle truncation independent of factored binary insertion order."""
    x = featured("x")
    bare = featured("x[b]")
    functor = featured("fwd", bare, x)
    rules = featured_compiled(
        ApplicationSpec(view="fwd"),
        CompositionSpec(view="fwd", order=1, direction="steedman"),
        ground_rule("bare-to-functor", (bare,), functor),
        ground_rule("functor-to-bare", (functor,), bare),
    )
    matching = rules[0].matching
    declared = Declared(
        rules,
        matching,
        ((0, bare, 1), (1, bare, 1), (1, functor, 1)),
        1,
    )
    forest = parse_declared(declared, Counting(), TermScorer(), indexed=True)
    inside = inside_declared(declared, Counting(), TermScorer())

    assert dict(forest.categories_at((0, 2))) == {bare: 1, functor: 3}
    assert by_span(inside) == by_span(forest)
    assert inside.truncated == forest.truncated


def test_inside_and_forest_count_raised_crossed_derivations_identically() -> None:
    """Keep tag-refined unary closure independent of chart retention."""
    x = featured("x")
    functor = featured("fwd", x, featured("x[b]"))
    rules = featured_compiled(
        ApplicationSpec(view="fwd"),
        ApplicationSpec(view="bwd"),
        CompositionSpec(view="fwd", order=1, direction="steedman", crossed=True),
        TypeRaisingSpec(view="bwd", targets=("t", "x"), arguments=None),
    )
    matching = rules[0].matching
    rows = ((0, functor, 1), (1, functor, 1), (2, x, 1))
    rule_set = RuleSet(
        "raised-crossed",
        rules,
        matching,
        HockenmaierBiskNormalForm[Term](),
    )
    forest = featured_parse(rows, rule_set, unary_depth=1)
    inside = parse(
        ("w", "w", "w"),
        rules=rule_set,
        source=FeaturedSource(rows),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((x,), matching)),
        unary_depth=1,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )

    assert by_span(inside) == by_span(forest)


@settings(max_examples=60, deadline=None)
@given(normal_form_grammars(unbounded=False))
def test_the_inside_chart_counts_normal_form_derivations_as_the_forest_does(
    grammar: Normalized,
) -> None:
    rules = RuleSet(
        "normalized",
        grammar.rules,
        grammar.matching,
        HockenmaierBiskNormalForm[Term](),
    )
    length = 1 + max(position for position, _, _ in grammar.rows)
    forest = featured_parse(grammar.rows, rules, unary_depth=grammar.unary_depth)
    inside = parse(
        tuple("w" for _ in range(length)),
        rules=rules,
        source=FeaturedSource(grammar.rows),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), grammar.matching)),
        unary_depth=grammar.unary_depth,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )

    assert by_span(inside) == by_span(forest)
    assert inside.normal_form == forest.normal_form == "hockenmaier-bisk"


def test_an_inside_chart_cannot_be_pruned() -> None:
    lexical = source("A", "B")
    retain: Retain = Retain.INSIDE

    with pytest.raises(ValueError, match="cannot be pruned"):
        parse(
            ("a", "b"),
            rules=inventory(()),
            source=lexical,
            semiring=LogInside(),
            scorer=Scorer(),
            goal=RootGoal(TargetSet((_category("S"),), EXACT)),
            unary_depth=0,
            pruner=cast("CellPruner[Term, float]", KeepAll()),
            policy=ParsePolicy(retain, Reachability.ALL),
        )

    assert lexical.calls == 0


def reach_declared[W](
    declared: Declared,
    semiring: Semiring[W],
    scorer: EdgeScorer[Term, W],
    reachability: Reachability,
) -> PackedForest[Term, W]:
    rules = declared.rules
    length = 1 + max(position for position, _, _ in declared.rows)

    return parse(
        tuple("w" for _ in range(length)),
        rules=RuleSet(
            "declared",
            rules,
            declared.matching,
            NoNormalForm[Term](),
            index=RuleIndex(rules, FeaturedHeads()),
        ),
        source=UnitRows(declared.rows, semiring),
        semiring=semiring,
        scorer=scorer,
        goal=RootGoal(TargetSet((featured("x"),), declared.matching)),
        unary_depth=declared.unary_depth,
        policy=ParsePolicy(Retain.FOREST, reachability),
    )


def root_reachable[W](
    forest: PackedForest[Term, W], matching: Matching[Term]
) -> frozenset[ItemKey[Term]]:
    """Return every item some derivation of a root uses."""
    goal = RootGoal(TargetSet((featured("x"),), matching))
    pending = list(goal.roots(forest))
    seen: set[ItemKey[Term]] = set()

    while pending:
        key = pending.pop()

        if key in seen:
            continue

        seen.add(key)
        pending.extend(child for edge in forest.edges(key) for child in edge.children)

    return frozenset(seen)


def every_item[W](forest: PackedForest[Term, W]) -> frozenset[ItemKey[Term]]:
    return frozenset(
        key for span in forest.index.spans_by_length() for key in forest.items_at(span)
    )


def same_on[W](
    used: frozenset[ItemKey[Term]],
    every: PackedForest[Term, W],
    reaching: PackedForest[Term, W],
) -> bool:
    return all(
        reaching.inside(key) == every.inside(key)
        and reaching.edges(key) == every.edges(key)
        for key in used
    )


def inside_reached(declared: Declared) -> InsideChart[Term, int]:
    rules = declared.rules
    length = 1 + max(position for position, _, _ in declared.rows)

    return parse(
        tuple("w" for _ in range(length)),
        rules=RuleSet(
            "declared",
            rules,
            declared.matching,
            NoNormalForm[Term](),
            index=RuleIndex(rules, FeaturedHeads()),
        ),
        source=UnitRows(declared.rows, Counting()),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), declared.matching)),
        unary_depth=declared.unary_depth,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ROOTS),
    )


def test_reachability_keeps_a_unary_path_between_initial_arrivals() -> None:
    leaf = featured("x[b]")
    intermediate = featured("t")
    root = featured("x")
    rules = featured_compiled(
        ground_rule("leaf-to-intermediate", (leaf,), intermediate),
        ground_rule("intermediate-to-root", (intermediate,), root),
        matching=IdentityMatchingSpec(),
    )
    matching = rules[0].matching
    declared = Declared(rules, matching, ((0, leaf, 1), (0, intermediate, 1)), 1)
    every = reach_declared(declared, Counting(), TermScorer(), Reachability.ALL)
    reaching = reach_declared(declared, Counting(), TermScorer(), Reachability.ROOTS)
    used = root_reachable(every, matching)

    assert used == every_item(reaching)
    assert same_on(used, every, reaching)
    assert reaching.categories_at((0, 1)) == every.categories_at((0, 1))


@pytest.mark.parametrize(
    ("refinement_pairs", "kept"), [(0, ("t", "x")), (1 << 16, ("t",))]
)
def test_the_declared_refinement_cap_reaches_the_reachability_planner(
    refinement_pairs: int, kept: tuple[str, ...]
) -> None:
    root = featured("x")
    argument = featured("t")
    rules = featured_compiled(
        ApplicationSpec(view="fwd"), matching=IdentityMatchingSpec()
    )
    matching = rules[0].matching
    forest = parse(
        ("w", "w"),
        rules=RuleSet(
            "declared",
            rules,
            matching,
            NoNormalForm[Term](),
            index=RuleIndex(rules, FeaturedHeads()),
        ),
        source=UnitRows(
            ((0, featured("fwd", root, argument), 1), (1, argument, 1), (1, root, 1)),
            Counting(),
        ),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((root,), matching)),
        unary_depth=0,
        policy=ParsePolicy(
            Retain.FOREST, Reachability.ROOTS, refinement_pairs=refinement_pairs
        ),
    )

    def categories(span: Cell) -> frozenset[Term]:
        return frozenset(category for category, _ in forest.categories_at(span))

    assert categories((1, 2)) == {featured(name) for name in kept}
    assert categories((0, 2)) == {root}


def higher_order_coercions() -> Declared:
    """Return a grammar whose root derivations mix a complete rule with others."""
    leaf = featured("x")
    higher = featured("fwd", featured("fwd", featured("x[b]"), leaf), leaf)
    rules = featured_compiled(
        ApplicationSpec(view="fwd"),
        CompositionSpec(view="fwd", order=2, direction="steedman"),
        ground_rule("coercion-0", (leaf,), higher),
        ground_rule("coercion-1", (higher,), leaf),
    )
    erasure = rules[0].matching
    rows = ((0, leaf, 1), (1, leaf, 1), (2, leaf, 1), (2, higher, 1))

    return Declared(rules, erasure, rows, 1)


def substituted_features() -> Declared:
    """Return a grammar whose root cell orders feature variants by arrival."""
    leaf = featured("x")
    higher = featured("fwd", featured("fwd", leaf, leaf), leaf)
    bare = featured("fwd", leaf, featured("x[b]"))
    rules = featured_compiled(
        ApplicationSpec(view="fwd"), SubstitutionSpec(view="fwd", order=1)
    )
    erasure = rules[0].matching
    rows = (
        (0, higher, 1),
        (1, higher, 1),
        (1, bare, 1),
        (1, leaf, 1),
        (2, higher, 1),
        (2, leaf, 1),
    )

    return Declared(rules, erasure, rows, 0)


def assert_reachability_keeps_root_derivations(declared: Declared) -> None:
    every = reach_declared(declared, Counting(), TermScorer(), Reachability.ALL)
    reaching = reach_declared(declared, Counting(), TermScorer(), Reachability.ROOTS)
    best = reach_declared(declared, Viterbi(), NamedScorer(), Reachability.ALL)
    best_reaching = reach_declared(
        declared, Viterbi(), NamedScorer(), Reachability.ROOTS
    )
    used = root_reachable(every, declared.matching)

    assert used <= every_item(reaching) <= every_item(every)
    assert same_on(used, every, reaching)
    assert by_span(inside_reached(declared)) == by_span(reaching)
    assert same_on(root_reachable(best, declared.matching), best, best_reaching)
    assert not reaching.truncated or every.truncated


@settings(max_examples=100, deadline=None)
@given(declared_grammars())
def test_reachability_keeps_every_item_a_root_derivation_uses(
    declared: Declared,
) -> None:
    assert_reachability_keeps_root_derivations(declared)


def test_reachability_keeps_root_derivations_through_higher_order_coercions() -> None:
    assert_reachability_keeps_root_derivations(higher_order_coercions())


def test_reachability_keeps_root_derivations_through_substituted_features() -> None:
    assert_reachability_keeps_root_derivations(substituted_features())


def _undeclared_sort(pattern: Pattern) -> Pattern:
    """Return a pattern whose metavariables take a sort no signature declares."""
    match pattern:
        case MVar():
            return MVar(pattern.name, "Undeclared")
        case Node():
            return Node(
                pattern.op,
                tuple(_undeclared_sort(argument) for argument in pattern.args),
                pattern.payload,
            )
        case Spine():
            return Spine(
                pattern.name,
                _undeclared_sort(pattern.core),
                pattern.ops,
                slot=pattern.slot,
                min_order=pattern.min_order,
                max_order=pattern.max_order,
            )


class UndeclaredSortSchemata:
    """A compiled rule whose reachability schemata name an undeclared sort."""

    __slots__ = ("_rule",)

    def __init__(self, rule: GeneralizedComposition, /) -> None:
        self._rule = rule

    @property
    def name(self) -> str:
        return self._rule.name

    @property
    def arity(self) -> int:
        return self._rule.arity

    @property
    def directions(self) -> frozenset[Direction]:
        return self._rule.directions

    @property
    def residual_slots(self) -> frozenset[int]:
        return self._rule.residual_slots

    @property
    def normal_form_role(self) -> NormalFormRole:
        return self._rule.normal_form_role

    @property
    def spine_order(self) -> int:
        return self._rule.spine_order

    @property
    def forward_is_complete(self) -> bool:
        return self._rule.forward_is_complete

    @property
    def matching(self) -> TermMatching:
        return self._rule.matching

    @property
    def restriction(self) -> RuleRestriction | None:
        return self._rule.restriction

    @property
    def probe(self) -> RuleProbe:
        return self._rule.probe

    @property
    def lowering(self) -> TermRuleKernel:
        return self._rule.lowering

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the compiled schemata over the undeclared sort."""
        return tuple(
            (
                tuple(_undeclared_sort(premise) for premise in premises),
                _undeclared_sort(conclusion),
            )
            for premises, conclusion in self._rule.reachability_schemata
        )

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        return self._rule.apply(inputs)

    def admits(self, inputs: tuple[Term, ...], output: Term, /) -> bool:
        return self._rule.admits(inputs, output)

    def residuate(
        self, output: Term, known: tuple[Slot[Term], ...], /
    ) -> Iterator[Term]:
        return self._rule.residuate(output, known)

    def to_spec(self) -> Mapping[str, JsonValue]:
        return self._rule.to_spec()


def test_reachability_refuses_a_schema_sort_the_signature_lacks() -> None:
    declared = substituted_features()
    (application,) = featured_compiled(ApplicationSpec(view="fwd"))

    assert type(application) is GeneralizedComposition

    rule = UndeclaredSortSchemata(application)

    with pytest.raises(ValueError, match="Undeclared"):
        reach_declared(
            declared._replace(rules=(rule,)),
            Counting(),
            TermScorer(),
            Reachability.ROOTS,
        )


def test_factorized_root_products_match_flat_unit_and_weighted_forests() -> None:
    erasure = FeatureErasure()
    rules = featured_rules()
    left = (
        featured("fwd", featured("x[b]"), featured("x[b]")),
        featured("fwd", featured("x[ng]"), featured("x[ng]")),
    )
    right = (featured("x[b]"), featured("x[ng]"))
    rows = tuple(
        (position, category, 1)
        for position, categories in enumerate((left, right))
        for category in categories
    )

    def parsed[W](
        semiring: Semiring[W],
        scorer: EdgeScorer[Term, W],
        reachability: Reachability,
    ) -> PackedForest[Term, W]:
        policy: ParsePolicy[Literal[Retain.FOREST]] = ParsePolicy(
            Retain.FOREST,
            reachability,
        )
        return parse(
            ("left", "right"),
            rules=rules,
            source=cast("LexicalSource[Term, W]", FeaturedSource(rows)),
            semiring=semiring,
            scorer=scorer,
            goal=RootGoal(TargetSet((featured("x"),), erasure)),
            unary_depth=0,
            policy=policy,
        )

    every = parsed(Counting(), TermScorer(), Reachability.ALL)
    reaching = parsed(Counting(), TermScorer(), Reachability.ROOTS)
    used = root_reachable(every, erasure)
    best = parsed(Viterbi(), NamedScorer(), Reachability.ALL)
    best_reaching = parsed(Viterbi(), NamedScorer(), Reachability.ROOTS)

    assert same_on(used, every, reaching)
    assert same_on(root_reachable(best, erasure), best, best_reaching)
    assert reaching.categories_at((0, 2)) == every.categories_at((0, 2))


@pytest.mark.parametrize(
    ("degree", "normal_form"),
    [
        (1, NoNormalForm[Term]()),
        (1, HockenmaierBiskNormalForm[Term]()),
        (None, NoNormalForm[Term]()),
        (None, HockenmaierBiskNormalForm[Term]()),
    ],
    ids=("fixed-admit", "fixed-normal", "unbounded-admit", "unbounded-premises"),
)
def test_composition_factors_erased_feet_without_changing_derivations(
    degree: int | None,
    normal_form: NormalForm[Term],
) -> None:
    (rule,) = featured_compiled(
        CompositionSpec(view="fwd", order=degree, direction="unrestricted")
    )
    erasure = rule.matching
    leaf_variants = (featured("x[b]"), featured("x[ng]"))
    feet = tuple(
        featured("fwd", featured("fwd", first, second), third)
        for first, second, third in product(leaf_variants, repeat=3)
    )
    left = tuple(featured("fwd", featured("x[b]"), foot) for foot in feet)
    right = tuple(featured("fwd", foot, featured("x[ng]")) for foot in feet)
    rows = tuple(
        (position, category, 1)
        for position, categories in enumerate((left, right))
        for category in categories
    )
    rules = RuleSet("factored-composition", (rule,), erasure, normal_form)
    goal = RootGoal(
        TargetSet((featured("fwd", featured("x[b]"), featured("x[ng]")),), erasure)
    )

    def counted(scorer: EdgeScorer[Term, int]) -> PackedForest[Term, int]:
        return parse(
            ("left", "right"),
            rules=rules,
            source=UnitRows(rows, Counting()),
            semiring=Counting(),
            scorer=scorer,
            goal=goal,
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, Reachability.ROOTS),
        )

    def best(scorer: EdgeScorer[Term, float]) -> PackedForest[Term, float]:
        return parse(
            ("left", "right"),
            rules=rules,
            source=UnitRows(rows, Viterbi()),
            semiring=Viterbi(),
            scorer=scorer,
            goal=goal,
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, Reachability.ROOTS),
        )

    def expanded[W](forest: PackedForest[Term, W]) -> tuple[object, ...]:
        return tuple(
            (
                key,
                forest.inside(key),
                tuple(
                    sorted(
                        repr(edge)
                        for edge in (
                            *forest.edges(key),
                            *(
                                edge
                                for held in forest.products(key)
                                for edge in held.edges()
                            ),
                        )
                    )
                ),
            )
            for span in forest.index.regions_by_size()
            for key in forest.items_at(span)
        )

    factored_count = counted(TermScorer())
    exact_count = counted(ExactTermScorer())
    factored_best = best(ViterbiTermScorer())
    exact_best = best(ExactViterbiTermScorer())

    assert expanded(factored_count) == expanded(exact_count)
    assert expanded(factored_best) == expanded(exact_best)
    (root,) = tuple(goal.roots(factored_count))
    (packed,) = factored_count.products(root)
    assert tuple(map(len, packed.axes)) == (len(feet), len(feet))
    assert len(exact_count.products(root)) == len(feet) ** 2


def test_output_normalization_keeps_composition_on_the_exact_pair_route() -> None:
    (rule,) = featured_compiled(
        CompositionSpec(view="fwd", order=1, direction="unrestricted")
    )
    erasure = rule.matching
    feet = (featured("x[b]"), featured("x[ng]"))
    rows = tuple(
        (position, category, 1)
        for position, categories in enumerate(
            (
                tuple(featured("fwd", featured("x[b]"), foot) for foot in feet),
                tuple(featured("fwd", foot, featured("x[ng]")) for foot in feet),
            )
        )
        for category in categories
    )
    rules = RuleSet("normalized-composition", (rule,), erasure, NoNormalForm[Term]())
    goal = RootGoal(
        TargetSet((featured("fwd", featured("x[b]"), featured("x[ng]")),), erasure)
    )
    forest = parse(
        ("left", "right"),
        rules=rules,
        source=UnitRows(rows, Counting()),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=goal,
        normalizer=IdentityNormalizer(),
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ROOTS),
    )
    (root,) = tuple(goal.roots(forest))

    assert len(forest.products(root)) == len(feet) ** 2


def test_output_normalization_disables_order_zero_factor_products() -> None:
    (rule,) = featured_compiled(ApplicationSpec(view="fwd"))
    erasure = rule.matching
    arguments = (featured("x[b]"), featured("x[ng]"))
    result = featured("x[b]")
    rows = tuple(
        (position, category, 1)
        for position, categories in enumerate(
            (
                tuple(featured("fwd", result, argument) for argument in arguments),
                arguments,
            )
        )
        for category in categories
    )
    rules = RuleSet("normalized-application", (rule,), erasure, NoNormalForm[Term]())
    goal = RootGoal(TargetSet((result,), erasure))
    forest = parse(
        ("left", "right"),
        rules=rules,
        source=UnitRows(rows, Counting()),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=goal,
        normalizer=IdentityNormalizer(),
        unary_depth=0,
        policy=ParsePolicy(Retain.FOREST, Reachability.ROOTS),
    )
    (root,) = tuple(goal.roots(forest))

    assert len(forest.products(root)) == len(arguments) ** 2
    assert all(
        tuple(map(len, product.axes)) == (1, 1) for product in forest.products(root)
    )


def test_reachability_filters_outputs_before_registration_without_poisoning_cache() -> (
    None
):
    components = relations("component", 2, {("A", "B"): ("C", "E"), ("X", "Y"): ("D",)})
    root = Relation("root", 2, {("D", "C"): ("S",)})
    rules = RuleSet(
        "span-specific-output",
        (*components, root),
        EXACT,
        NoNormalForm[Term](),
    )
    rows = (
        (0, _category("A"), 1),
        (0, _category("X"), 1),
        (1, _category("B"), 1),
        (1, _category("Y"), 1),
        (2, _category("A"), 1),
        (3, _category("B"), 1),
    )

    def parsed(
        retain: Literal[Retain.FOREST, Retain.INSIDE],
        reachability: Reachability,
    ) -> PackedForest[Term, int] | InsideChart[Term, int]:
        return parse(
            ("a", "b", "c", "d"),
            rules=rules,
            source=UnitRows(rows, Counting()),
            semiring=Counting(),
            scorer=TermScorer(),
            goal=RootGoal(TargetSet((_category("S"),), EXACT)),
            unary_depth=0,
            policy=ParsePolicy(retain, reachability),
        )

    every = parsed(Retain.FOREST, Reachability.ALL)
    reaching = parsed(Retain.FOREST, Reachability.ROOTS)
    inside = parsed(Retain.INSIDE, Reachability.ROOTS)
    assert isinstance(every, PackedForest)
    assert isinstance(reaching, PackedForest)
    assert isinstance(inside, InsideChart)

    used = root_reachable(every, EXACT)
    assert same_on(used, every, reaching)
    assert by_span(inside) == by_span(reaching)
    assert reaching.categories_at((0, 2)) == ((_category("D"), 1),)
    assert reaching.categories_at((2, 4)) == ((_category("C"), 1),)
    assert reaching.categories_at((0, 4)) == ((_category("S"), 1),)
    assert reaching._categories.lookup(_category("E")) is None
    assert inside._categories.lookup(_category("E")) is None


def test_reachability_drops_a_whole_input_item_no_unary_chain_leads_to_a_root() -> None:
    t = featured("t")
    rows = (
        (0, t, 1),
        (1, featured("bwd", featured("x"), t), 1),
        (1, featured("bwd", t, t), 1),
    )
    rules = featured_rules()

    def parsed(reachability: Reachability) -> PackedForest[Term, int]:
        return parse(
            ("a", "b"),
            rules=rules,
            source=FeaturedSource(rows),
            semiring=Counting(),
            scorer=TermScorer(),
            goal=RootGoal(TargetSet((featured("x"),), rules.matching)),
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, reachability),
        )

    every = parsed(Reachability.ALL)
    reaching = parsed(Reachability.ROOTS)
    (lift,) = featured_compiled(ground_rule("lift", (t,), featured("x")))
    lifted_rules = RuleSet(
        "lifted", (*rules.rules, lift), rules.matching, NoNormalForm[Term]()
    )
    lifted = parse(
        ("a", "b"),
        rules=lifted_rules,
        source=FeaturedSource(rows),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), rules.matching)),
        unary_depth=1,
        policy=ParsePolicy(Retain.FOREST, Reachability.ROOTS),
    )
    inside = parse(
        ("a", "b"),
        rules=rules,
        source=FeaturedSource(rows),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), rules.matching)),
        unary_depth=0,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ROOTS),
    )
    lifted_inside = parse(
        ("a", "b"),
        rules=lifted_rules,
        source=FeaturedSource(rows),
        semiring=Counting(),
        scorer=TermScorer(),
        goal=RootGoal(TargetSet((featured("x"),), rules.matching)),
        unary_depth=1,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ROOTS),
    )

    assert every.categories_at((0, 2)) == ((featured("x"), 1), (t, 1))
    assert reaching.categories_at((0, 2)) == ((featured("x"), 1),)
    assert lifted.categories_at((0, 2)) == ((featured("x"), 2), (t, 1))
    assert lifted_inside.categories_at((0, 2)) == lifted.categories_at((0, 2))
    assert inside.categories_at((0, 2)) == ((featured("x"), 1),)
    assert reaching.categories_at((1, 2)) == ((featured("bwd", featured("x"), t), 1),)
    assert not reaching.truncated


def test_declared_consumption_prunes_without_changing_a_root_derivation() -> None:
    """Use finite rule effects when the relation has no finite schema."""
    consuming = tuple(
        Relation(f"consuming-{at}", 2, {inputs: ("S",)}, publish_schemata=False)
        for at, inputs in enumerate((("A", "B"), ("C", "B")))
    )
    rules = RuleSet(
        "declared-consumption",
        consuming,
        EXACT,
        NoNormalForm[Term](),
        introduces=((_category("S"),),) * len(consuming),
        consumes=((_category("B"),),) * len(consuming),
    )
    source = UnitRows(
        (
            (0, _category("A"), 1),
            (0, _category("C"), 1),
            (1, _category("B"), 1),
            (1, _category("D"), 1),
        ),
        Counting(),
    )

    def parsed(reachability: Reachability) -> PackedForest[Term, int]:
        return parse(
            ("left", "right"),
            rules=rules,
            source=source,
            semiring=Counting(),
            scorer=TermScorer(),
            goal=RootGoal(TargetSet((_category("S"),), EXACT)),
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, reachability),
        )

    every = parsed(Reachability.ALL)
    reaching = parsed(Reachability.ROOTS)
    used = root_reachable(every, EXACT)

    assert used <= every_item(reaching) < every_item(every)
    assert same_on(used, every, reaching)
    assert reaching.categories_at((0, 2)) == every.categories_at((0, 2))
    assert reaching.categories_at((1, 2)) == ((_category("B"), 1),)


def test_unpresented_rule_conclusions_keep_unrestricted_reachability() -> None:
    """Keep every derivation when a rule cannot present its conclusions."""
    rule = Relation(
        "unpresented",
        2,
        {("A", "B"): ("S",)},
        publish_schemata=False,
    )
    rules = RuleSet(
        "unpresented",
        (rule,),
        EXACT,
        NoNormalForm[Term](),
    )
    lexical = UnitRows(
        ((0, _category("A"), 1), (1, _category("B"), 1)),
        Counting(),
    )

    def parsed(reachability: Reachability) -> PackedForest[Term, int]:
        return parse(
            ("left", "right"),
            rules=rules,
            source=lexical,
            semiring=Counting(),
            scorer=TermScorer(),
            goal=RootGoal(TargetSet((_category("S"),), EXACT)),
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, reachability),
        )

    every = parsed(Reachability.ALL)
    reaching = parsed(Reachability.ROOTS)

    assert reaching.categories_at((0, 2)) == every.categories_at((0, 2))
    assert reaching.categories_at((0, 2)) == ((_category("S"), 1),)


def test_reachability_keeps_a_feature_variant_the_root_matching_covers() -> None:
    t = featured("t")
    rows = ((0, t, 1), (1, featured("bwd", featured("x[b]"), t), 1))
    rules = featured_rules()

    def parsed(retain: Retain) -> PackedForest[Term, int] | InsideChart[Term, int]:
        return parse(
            ("a", "b"),
            rules=rules,
            source=FeaturedSource(rows),
            semiring=Counting(),
            scorer=TermScorer(),
            goal=RootGoal(TargetSet((featured("x"),), rules.matching)),
            unary_depth=0,
            policy=ParsePolicy(retain, Reachability.ROOTS),
        )

    assert parsed(Retain.FOREST).categories_at((0, 2)) == ((featured("x[b]"), 1),)
    assert parsed(Retain.INSIDE).categories_at((0, 2)) == ((featured("x[b]"), 1),)


def test_reachability_needs_a_goal_that_names_its_roots() -> None:
    with pytest.raises(ValueError, match="names its root categories"):
        parse(
            ("a",),
            rules=inventory(()),
            source=source("A"),
            semiring=LogInside(),
            scorer=Scorer(),
            goal=CoverGoal(1, penalty=1.0),
            unary_depth=0,
            policy=ParsePolicy(Retain.FOREST, Reachability.ROOTS),
        )


class GatedByChild(RuleTags):
    """Tags items with the rule name and admits one rule only above one tag."""

    def __init__(self, names: Sequence[str], gated: str, below: str) -> None:
        super().__init__(names)
        self._gated = gated
        self._below = below

    @override
    def decisions(self, rules: Sequence[Rule[Term]], /) -> NormalFormDecisionTable:
        return tuple(
            {
                children: (
                    "refuse"
                    if rule.name == self._gated and children[0] != self._below
                    else "admit"
                )
                for children in product(self.alphabet, repeat=rule.arity)
            }
            for rule in rules
        )

    @override
    def admits(
        self,
        rule: Rule[Term],
        children: tuple[str, ...],
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        return super().admits(rule, children, inputs, output) and (
            rule.name != self._gated or children[0] == self._below
        )


@pytest.mark.parametrize("below", ["left-a", "left-b"])
def test_a_premise_tuple_is_judged_afresh_under_each_tag_tuple(below: str) -> None:
    left_a = Relation("left-a", 2, {("A", "B"): ("X",)})
    left_b = Relation("left-b", 2, {("A", "B"): ("X",)})
    finish = Relation("finish", 2, {("X", "C"): ("S",)})
    gate = GatedByChild(("left-a", "left-b", "finish"), "finish", below)
    chart = run(
        inventory((left_a, left_b, finish), normal_form=cast("NormalForm[Term]", gate)),
        source("A", "B", "C"),
    )
    (root,) = chart.items_at((0, 3))
    (edge,) = chart.edges(root)

    assert edge.children[0].nf_tag == below


class OutputScorer(Scorer):
    """Weighs each step by the conclusion it draws."""

    @override
    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, span, split
        return self._rules[_label(output)]


def test_the_inside_chart_weighs_each_conclusion_of_one_step_by_its_own_score() -> None:
    rule = Relation("split", 2, {("A", "B"): ("S", "T")})
    scorer = OutputScorer({"S": -1.0, "T": -2.0})
    chart = parse(
        ("a", "b"),
        rules=inventory((rule,)),
        source=source("A", "B"),
        semiring=LogInside(),
        scorer=scorer,
        goal=RootGoal(TargetSet((_category("S"),), EXACT)),
        unary_depth=0,
        policy=ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )
    forest = run(inventory((rule,)), source("A", "B"), scorer=scorer)

    assert {
        _label(category): weight for category, weight in chart.categories_at((0, 2))
    } == {"S": -1.0, "T": -2.0}
    assert chart.categories_at((0, 2)) == forest.categories_at((0, 2))


class JoinedRelation(Relation):
    """Joins premises on the table rows each can fill, offering two keys a row."""

    def join_keys(self, term: Term, slot: int, /) -> tuple[tuple[int, str], ...]:
        rows = tuple(self._table)

        return tuple(
            (at, copy)
            for at, row in enumerate(rows)
            if row[slot] == term
            for copy in ("first", "second")
        )


def test_a_join_tries_each_rule_once_beside_the_head_rows_of_the_pair() -> None:
    wanted = RuleProbe(2, ((frozenset({"A"}), frozenset({"B"})),))
    plain = Relation("plain", 2, {("A", "B"): ("X",)}, probe=wanted)
    joined = JoinedRelation("joined", 2, {("A", "B"): ("Y",)})
    unpaired = JoinedRelation("unpaired", 2, {("C", "C"): ("Y",)})
    rules = (plain, joined, unpaired)
    scorer = Scorer()
    index = RuleIndex(rules, cast("HeadIndex[Term]", Heads()))
    chart = run(inventory(rules, index=index), source("A", "B"), scorer=scorer)

    assert index.joined == (joined, unpaired)
    assert tuple(_label(key.reading) for key in chart.items_at((0, 2))) == ("X", "Y")
    assert scorer.trials == ["plain", "joined"]
