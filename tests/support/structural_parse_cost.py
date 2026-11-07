"""Current-interface bridge for structural-probes' read-only parse-cost fixture."""

from __future__ import annotations

import json
from typing import TYPE_CHECKING, Final, NamedTuple, cast

import didactic.api as dx

from lofi.deduce import RootGoal, UnitEdges
from lofi.spec import (
    CoercionSetSpec,
    CombinatorSetSpec,
    ConstructorSpec,
    CoordinationSpec,
    InfixConstructorNotationSpec,
    RuleSetSpec,
    RuleSpec,
)
from lofi.spec.datasets import ccgbank_grammar
from lofi.spec.grammar import (
    BudgetSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    FunctorSpec,
    GrammarDecl,
    GrammarIso,
    HockenmaierBiskNormalFormSpec,
    IdentityMatchingSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    NormalFormSpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import registered_language

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence
    from pathlib import Path

    from didactic.types._typing import FieldValue

    from lofi.deduce import Cell
    from lofi.judge import Rule, RuleSet
    from lofi.lexicon import Entry, LexicalSource, StaticLexicon
    from lofi.spec.grammar import CategoryMeasure
    from lofi.terms import Term

type _CategoryShape = str | tuple[str, _CategoryShape, _CategoryShape]

__all__ = [
    "ParseCostRuntime",
    "ParseCostSentence",
    "UnitWeights",
    "fixture_composition_orders",
    "fixture_declaration",
    "fixture_runtime",
    "fixture_sentences",
]

_GRAMMAR_FILE: Final = "grammar.json"
_LEXICON_FILE: Final = "lexicon.json"
_SENTENCES_FILE: Final = "sentences.json"
_MAX_CATEGORY_ARITY: Final = 4
_MAX_CATEGORY_SIZE: Final = 31


class _Calculus(dx.Model, extra="ignore"):
    """Rule inventory read from structural-probes."""

    coercions: tuple[tuple[str, ...], ...]
    composition_restriction: str
    coordination_feature: str
    coordination_markers: tuple[str, ...]
    coordination_pairs: tuple[tuple[str, ...], ...]
    crossed_orders: tuple[int, ...]
    crossed_substitution_orders: tuple[int, ...]
    exceptional_rules: tuple[str, ...]
    orders: tuple[int, ...]
    substitution_orders: tuple[int, ...]
    type_raise: bool

    __axioms__ = (
        dx.axiom(
            "all(map (\\p -> len(p) == 2) coercions)",
            message="each coercion needs a source and target",
        ),
        dx.axiom(
            "all(map (\\p -> len(p) == 2) coordination_pairs)",
            message="each conjunction pair needs an unmarked and marked category",
        ),
    )


class _Adequacy(dx.Model, extra="ignore"):
    """Root and closure settings read from structural-probes."""

    roots: tuple[str, ...]
    unary_depth: int


class _FixtureGrammar(dx.Model, extra="ignore"):
    """Boundary model for the fixture fields this bridge consumes."""

    primitives: tuple[str, ...]
    features: tuple[str, ...]
    calculus: _Calculus
    adequacy: _Adequacy


class ParseCostSentence(dx.Model, extra="forbid"):
    """One sentence in structural-probes' declared sample."""

    tokens: tuple[str, ...]
    length: int
    section: int
    line: int
    measured_seconds: float | None

    __axioms__ = (
        dx.axiom("len(tokens) == length", message="sentence length must match tokens"),
    )


class ParseCostRuntime(NamedTuple):
    """Runtime parse inputs built without the unrelated derivation theory."""

    rules: RuleSet[Term]
    lexicon: StaticLexicon[Term]
    goal: RootGoal[Term]
    category_measure: CategoryMeasure
    unary_depth: int
    threads: int
    max_category_arity: int | None
    max_category_size: int | None


class UnitWeights[W]:
    """Replace fixture lexicon and edge weights by one semiring unit."""

    __slots__ = ("_one", "_source")

    def __init__(self, source: LexicalSource[Term, float], *, one: W) -> None:
        self._source = source
        self._one = one

    @property
    def is_re_callable(self) -> bool:
        """Return whether the wrapped source can be enumerated again."""
        return self._source.is_re_callable

    @property
    def lowering(self) -> UnitEdges:
        """Declare that every rule edge has the semiring unit."""
        return UnitEdges()

    @property
    def clamps(self) -> tuple[()]:
        """Return no clamps."""
        return ()

    def axioms(self, tokens: Sequence[str], /) -> Iterable[tuple[int, Entry[Term], W]]:
        """Return every fixture axiom with the semiring unit."""
        return tuple(
            (position, entry, self._one)
            for position, entry, _ in self._source.axioms(tokens)
        )

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> W:
        """Return the semiring unit."""
        del position, entry

        return self._one

    def edge_weight(
        self,
        rule: Rule[Term],
        inputs: tuple[Term, ...],
        output: Term,
        span: Cell,
        split: int | None,
        /,
    ) -> W:
        """Return the semiring unit."""
        del rule, inputs, output, span, split

        return self._one


_LEXICON_ADAPTER: Final = dx.TypeAdapter(dict[str, tuple[str, ...]])
_SENTENCE_ADAPTER: Final = dx.TypeAdapter(tuple[ParseCostSentence, ...])


def _grammar(root: Path) -> _FixtureGrammar:
    """Read and validate the fixture grammar boundary."""
    return _FixtureGrammar.model_validate_json(
        (root / _GRAMMAR_FILE).read_text(encoding="utf-8")
    )


def _lexicon(root: Path) -> dict[str, tuple[str, ...]]:
    """Read and validate the fixture lexicon boundary."""
    raw = cast(
        "FieldValue", json.loads((root / _LEXICON_FILE).read_text(encoding="utf-8"))
    )

    return _LEXICON_ADAPTER.validate(raw)


def _ungroup(text: str, /) -> str:
    """Remove every pair of parentheses enclosing one whole category."""
    held = text

    while held.startswith("(") and held.endswith(")"):
        depth = 0

        for position, character in enumerate(held):
            if character == "(":
                depth += 1
            elif character == ")":
                depth -= 1

            if depth == 0:
                if position != len(held) - 1:
                    return held

                held = held[1:-1]
                break
        else:
            return held

    return held


def _category_shape(text: str, /) -> _CategoryShape:
    """Return one feature-erased structural category key."""
    held = _ungroup(text)
    parentheses = 0
    brackets = 0
    split = -1

    for position, character in enumerate(held):
        if character == "[":
            brackets += 1
        elif character == "]":
            brackets -= 1
        elif brackets == 0:
            if character == "(":
                parentheses += 1
            elif character == ")":
                parentheses -= 1
            elif parentheses == 0 and character in ("/", "\\"):
                split = position

    if split == -1:
        return held.split("[", 1)[0]

    return (
        held[split],
        _category_shape(held[:split]),
        _category_shape(held[split + 1 :]),
    )


def _render_category(shape: _CategoryShape, /) -> str:
    """Render one feature-erased category key in declared infix syntax."""
    if isinstance(shape, str):
        return shape

    constructor, result, argument = shape
    rendered_result = _render_category(result)
    rendered_argument = _render_category(argument)

    if not isinstance(result, str):
        rendered_result = f"({rendered_result})"

    if not isinstance(argument, str):
        rendered_argument = f"({rendered_argument})"

    return f"{rendered_result}{constructor}{rendered_argument}"


def _subcategories(shape: _CategoryShape, /) -> frozenset[_CategoryShape]:
    """Return the structural categories occurring in one category."""
    if isinstance(shape, str):
        return frozenset((shape,))

    _, result, argument = shape

    return frozenset((shape,)) | _subcategories(result) | _subcategories(argument)


def _category_depth(shape: _CategoryShape, /) -> int:
    """Return the number of complete-tree levels one category occupies."""
    if isinstance(shape, str):
        return 1

    _, result, argument = shape

    return 1 + max(_category_depth(result), _category_depth(argument))


def _inventory_categories(
    fixture: _FixtureGrammar,
    lexicon: dict[str, tuple[str, ...]],
    /,
) -> tuple[str, ...]:
    """Return every category owned by the reduced fixture inventory."""
    calculus = fixture.calculus

    return (
        *(category for categories in lexicon.values() for category in categories),
        *(category for pair in calculus.coercions for category in pair),
        *(category for pair in calculus.coordination_pairs for category in pair),
        *fixture.adequacy.roots,
    )


def _lexical_categories(
    lexicon: dict[str, tuple[str, ...]],
    /,
) -> tuple[str, ...]:
    """Return every category assigned by the reduced fixture lexicon."""
    return tuple(category for categories in lexicon.values() for category in categories)


def _roots(
    fixture: _FixtureGrammar,
    lexicon: dict[str, tuple[str, ...]],
    /,
) -> tuple[str, ...]:
    """Return one reachable representative of each root matching class."""
    available = frozenset(
        subcategory
        for category in _lexical_categories(lexicon)
        for subcategory in _subcategories(_category_shape(category))
    )
    held: dict[_CategoryShape, None] = {}

    for category in fixture.adequacy.roots:
        shape = _category_shape(category)

        if shape in available:
            held.setdefault(shape, None)

    return tuple(map(_render_category, held))


def fixture_sentences(root: Path) -> tuple[ParseCostSentence, ...]:
    """Read and validate the fixture sentences."""
    raw = cast(
        "FieldValue", json.loads((root / _SENTENCES_FILE).read_text(encoding="utf-8"))
    )

    return _SENTENCE_ADAPTER.validate(raw)


def fixture_composition_orders(
    root: Path, /
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Return the straight and crossed composition orders the fixture declares."""
    calculus = _grammar(root).calculus

    return calculus.orders, calculus.crossed_orders


def _exceptional(calculus: _Calculus) -> tuple[RuleSpec, ...]:
    """Select and finitely enumerate the declared exceptional relations."""
    by_name = {rule.name: rule for rule in ccgbank_grammar().rules.rules}
    missing = set(calculus.exceptional_rules) - set(by_name)

    if missing:
        msg = f"unknown exceptional rules: {sorted(missing)!r}"
        raise ValueError(msg)

    unlike_targets = tuple(pair[0] for pair in calculus.coordination_pairs)
    held: list[RuleSpec] = []

    for name in calculus.exceptional_rules:
        rule = by_name[name]

        if name.startswith("clausal-adjunct-"):
            continue

        if name == "n-to-np" and ("N", "NP") in calculus.coercions:
            continue

        if name == "coordination-unlike":
            coordination = cast("CoordinationSpec", rule)
            rule = coordination.with_(
                conjunction_feature=calculus.coordination_feature,
                targets=unlike_targets,
            )
        elif isinstance(rule, CoordinationSpec):
            rule = rule.with_(conjunction_feature=calculus.coordination_feature)

        held.append(rule)

    return tuple(held)


def _independent_crossed_orders(calculus: _Calculus, /) -> tuple[int, ...]:
    """Return crossed orders not already covered by unrestricted composition."""
    straight = frozenset(calculus.orders)

    return tuple(order for order in calculus.crossed_orders if order not in straight)


def _declaration(
    root: Path,
    normal_form: NormalFormSpec | None = None,
    /,
) -> tuple[GrammarDecl, int]:
    """Translate the fixture data and return its owned category depth."""
    fixture = _grammar(root)
    calculus = fixture.calculus
    base_primitives = tuple(
        dict.fromkeys(name.split("[", 1)[0] for name in fixture.primitives)
    )
    featured = tuple(name for name in fixture.primitives if "[" in name)
    coercions = tuple(f"{source} => {target}" for source, target in calculus.coercions)
    lexicon = _lexicon(root)

    if calculus.composition_restriction != "unrestricted":
        msg = (
            "the parse-cost fixture must declare unrestricted composition, not "
            f"{calculus.composition_restriction!r}"
        )
        raise ValueError(msg)

    combinators = CombinatorSetSpec(
        application=True,
        composition=calculus.orders,
        crossed_composition=_independent_crossed_orders(calculus),
        substitution=calculus.substitution_orders,
        crossed_substitution=calculus.crossed_substitution_orders,
        direction="unrestricted",
        type_raising=calculus.type_raise,
    )

    declaration = GrammarDecl(
        name="structural-probes-parse-cost",
        signature=SignatureSpec(
            name="structural-probes-parse-cost",
            primitives=base_primitives,
            operators=tuple(
                OperatorSpec(name=name, sort="Cat", erases_to=name.split("[", 1)[0])
                for name in featured
            ),
            constructors=tuple(
                ConstructorSpec(
                    name=name,
                    kind=("Cat", "Cat", "Cat"),
                    notation=InfixConstructorNotationSpec(
                        spelling=name,
                        precedence=10,
                        associativity="left",
                    ),
                )
                for name in ("/", "\\")
            ),
            functors=(
                FunctorSpec(
                    name="/",
                    constructor="/",
                    argument_side="right",
                    opposite="\\",
                ),
                FunctorSpec(
                    name="\\",
                    constructor="\\",
                    argument_side="left",
                    opposite="/",
                ),
            ),
            conjunctions=ConjunctionsSpec(
                markers=calculus.coordination_markers,
                pairs=tuple(
                    ConjunctionPairSpec(unmarked=plain, marked=marked)
                    for plain, marked in calculus.coordination_pairs
                ),
            ),
        ),
        rules=RuleSetSpec(
            name="structural-probes-parse-cost",
            rules=(
                *_exceptional(calculus),
                combinators,
                CoercionSetSpec(
                    label="coercion",
                    entries=coercions,
                    matching=IdentityMatchingSpec(),
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, categories in lexicon.items()
                for category in categories
            )
        ),
        goal=RootGoalSpec(categories=_roots(fixture, lexicon)),
        matching=FeatureErasureMatchingSpec(),
        normal_form=(
            HockenmaierBiskNormalFormSpec(
                composition_primary=False,
                unit_primary=False,
            )
            if normal_form is None
            else normal_form
        ),
        budget=BudgetSpec(
            unary_rounds=fixture.adequacy.unary_depth,
            threads=8,
            max_category_arity=_MAX_CATEGORY_ARITY,
            max_category_size=_MAX_CATEGORY_SIZE,
        ),
    )
    category_depth = max(
        _category_depth(_category_shape(category))
        for category in _inventory_categories(fixture, lexicon)
    )

    return declaration, category_depth


def fixture_declaration(
    root: Path,
    normal_form: NormalFormSpec | None = None,
    /,
) -> GrammarDecl:
    """Translate the fixture data to the one current grammar declaration."""
    declaration, _ = _declaration(root, normal_form)

    return declaration


def fixture_runtime(
    root: Path,
    normal_form: NormalFormSpec | None = None,
    /,
) -> ParseCostRuntime:
    """Build only the current runtime values the parse-cost call consumes."""
    declaration, category_depth = _declaration(root, normal_form)
    language = registered_language("propositional")
    bounds = dict(language.bounds)
    category_sort = language.category_sort
    bounds[category_sort] = bounds[category_sort].with_(depth=category_depth)
    assembly = GrammarIso().forward(
        declaration,
        language.with_(bounds=bounds),
    )
    grammar = assembly.grammar

    if not isinstance(grammar.goal, RootGoal):
        msg = "the parse-cost fixture needs a root goal"
        raise TypeError(msg)

    return ParseCostRuntime(
        grammar.rules,
        assembly.lexicon,
        grammar.goal,
        assembly.category_measure,
        grammar.budget.unary_rounds,
        grammar.budget.threads,
        grammar.budget.max_category_arity,
        grammar.budget.max_category_size,
    )
