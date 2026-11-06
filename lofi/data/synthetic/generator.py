"""Sampling of grammatical strings from compiled grammars.

The generator tabulates derivation mass by token length and category, a
derivation weighing the product of the masses of its lexical entries. It
draws each string's token length from a declared distribution, then a goal
derivation of that length in proportion to its mass, and reparses the token
sequence to record the parser's matching derivation. Under a declared
interpretation it reads the parser's derivation into a logical form and
labels drawn pairs of strings by what the proof system establishes.
"""

from collections import Counter
from hashlib import sha256
from itertools import count, product
from math import exp, prod
from random import Random
from typing import TYPE_CHECKING, Final, Literal, Protocol

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._stats import spearman
from lofi._types import ItemId
from lofi.data._derivation import Derivation, DerivationNode
from lofi.data._supervision import Example, GoldDerivation
from lofi.data._values import (
    Item,
    Licensing,
    Provenance,
    Reproducibility,
    TokenSpan,
)
from lofi.data.synthetic._complexity import (
    ComplexityMeter,
    ComplexityReport,
    ComplexityUnreachableError,
)
from lofi.data.synthetic._entailment import GoldForm
from lofi.data.synthetic._illtyped import IllTyped
from lofi.data.synthetic._inference import (
    ConfusionReport,
    FragmentEscapeError,
    InferencePair,
    InferenceReport,
    NoisedLabel,
    PairRelations,
    PairRelationUnreachableError,
    ProportionUnreachableError,
    RelationReport,
    SyntheticInference,
    relation_complexity,
    synthetic_inference_item,
)
from lofi.data.synthetic._lengths import LengthOutOfSupportError
from lofi.data.synthetic._pairing import INDEPENDENT_PAIRING
from lofi.data.synthetic._scope import Readings, readings
from lofi.data.synthetic._semantics import SemanticReport
from lofi.data.synthetic._values import (
    AmbiguityReport,
    AmbiguityUnreachableError,
    Corrupted,
    CorruptionExhaustedError,
    CorruptionReport,
    GrammarKnobs,
    Histogram,
    IllTypingReport,
    InventoryReport,
    LengthReport,
    LexiconReport,
    NoiseReport,
    NonDerivability,
    ObservedLabel,
    ReadingCount,
    ScopeReport,
    ScopeUnreachableError,
    ShapeMeasurement,
    ShapeReport,
    ShapeUnreachableError,
    SyntheticCorpus,
    UncertifiedDerivationError,
    UnproductiveGrammarError,
)
from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    LogInside,
    ParsePolicy,
    Retain,
    SpanFree,
    parse,
)
from lofi.deduce import Derivation as ChartDerivation
from lofi.interpret import is_axiom
from lofi.lf import RELATION_LABELS, Decided, InferenceLabel
from lofi.terms import Term

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Mapping, Sequence

    from lofi.data._derivation import NodeId
    from lofi.data.synthetic._complexity import (
        FormComplexity,
        PairComplexityTarget,
    )
    from lofi.data.synthetic._corrupt import CorruptionOperator
    from lofi.data.synthetic._entailment import EntailmentCertificate
    from lofi.data.synthetic._inference import (
        InferenceNoise,
        LabelProportion,
    )
    from lofi.data.synthetic._injection import IllTypingOperator
    from lofi.data.synthetic._interpretation import InterpretationFunction
    from lofi.data.synthetic._lengths import LengthDistribution
    from lofi.data.synthetic._pairing import PairingStrategy
    from lofi.data.synthetic._values import (
        AmbiguityTarget,
        Corruption,
        GrammaticalityLabel,
        LabelNoise,
        ScopeTarget,
        ShapeSchedule,
    )
    from lofi.deduce import Cell, Chart, ItemKey, PackedForest
    from lofi.formalism import Grammar
    from lofi.judge import NormalForm, Rule, RuleSet
    from lofi.lexicon import Entry, StaticLexicon
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Matching, TermStore, TermSyntax

__all__ = ["SOURCE_METHOD", "RuleWeighting", "SyntheticGenerator"]

SOURCE_METHOD: Final = "generated"
"""The provenance method every synthetic item is released under."""

CC0: Final = Licensing("CC0-1.0")
"""The terms synthetic items are released under unless a caller names others."""

type _State = tuple[Term, str]
"""A chart item without its span: the category representative and its tag."""

type _Premise = tuple[_State, int]
"""A premise item: its state and the number of tokens it covers."""

type _UnaryStep = tuple[_State, str, Term, float]
"""One unary application: its source, rule, conclusion and local mass factor."""


class RuleWeighting(Protocol):
    """Supplies the relative mass of each rule application."""

    def weight(self, rule: str, /) -> float:
        """Return the relative mass of one application of the named rule."""
        ...


class _UniformRuleWeighting:
    """Assigns neutral mass to every rule application."""

    __slots__ = ()

    def weight(self, rule: str, /) -> float:
        """Return the neutral factor for the rule."""
        del rule

        return 1.0


_UNIFORM_RULE_WEIGHTING: Final[RuleWeighting] = _UniformRuleWeighting()


class _Production:
    """One way an item is concluded without a unary step on top.

    Parameters
    ----------
    term : Term
        The conclusion drawn.
    rule : str | None
        The rule applied, or nothing for a lexical entry.
    form : str | None
        The surface form of a lexical entry, or nothing for a rule step.
    premises : tuple[_Premise, ...]
        The premise items in rule order, none for a lexical entry.
    mass : float
        The mass of the derivations concluding this way: the product of the
        premise masses, or the mass of the lexical entry.
    """

    __slots__ = ("form", "mass", "premises", "rule", "term")

    def __init__(
        self,
        term: Term,
        *,
        rule: str | None,
        form: str | None,
        premises: tuple[_Premise, ...],
        mass: float,
    ) -> None:
        self.term = term
        self.rule = rule
        self.form = form
        self.premises = premises
        self.mass = mass


class _Sampled:
    """One drawn derivation node.

    Parameters
    ----------
    term : Term
        The category the node concludes.
    rule : str | None
        The rule applied, or nothing at a leaf.
    form : str | None
        The surface form at a leaf, or nothing above one.
    children : tuple[_Sampled, ...]
        The premise nodes in rule order.
    """

    __slots__ = ("children", "form", "rule", "term")

    def __init__(
        self,
        term: Term,
        *,
        rule: str | None,
        form: str | None,
        children: tuple[_Sampled, ...],
    ) -> None:
        self.term = term
        self.rule = rule
        self.form = form
        self.children = children

    def forms(self) -> Iterator[str]:
        """Yield the surface forms under the node, left to right."""
        for _, form in self.entries():
            yield form

    def entries(self) -> Iterator[tuple[Term, str]]:
        """Yield the category and surface form of each leaf, left to right."""
        if self.form is not None:
            yield self.term, self.form

        for child in self.children:
            yield from child.entries()

    @property
    def edge_name(self) -> str:
        """Return the name the chart gives the edge concluding this node."""
        return f"{LEXICAL_TAG_PREFIX}{self.form}" if self.rule is None else self.rule


def _shape(node: _Sampled, /) -> ShapeMeasurement:
    """Measure the depth, branching, slash direction and arity of a derivation."""
    binary = 0
    right = 0
    forward = 0
    slash = 0
    arity = 0
    rules = 0

    def visit(current: _Sampled) -> int:
        nonlocal arity, binary, forward, right, rules, slash
        depths = tuple(visit(child) for child in current.children)

        if current.rule is not None:
            rules += 1
            arity += len(current.children)
            direction = current.rule.rpartition(":")[2]

            if direction in ("/", "\\"):
                slash += 1
                forward += direction == "/"

        if len(depths) == 2:
            binary += 1
            right += depths[1] > depths[0]

        return 1 + max(depths, default=0)

    depth = visit(node)

    return ShapeMeasurement(
        depth,
        0.0 if binary == 0 else right / binary,
        0.0 if slash == 0 else forward / slash,
        0.0 if rules == 0 else arity / rules,
    )


class _NeutralScorer:
    """Gives every axiom and rule application the neutral log weight."""

    __slots__ = ()

    @property
    def lowering(self) -> SpanFree:
        """Declare that every rule receives the same local weight."""
        return SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Term], /) -> float:
        """Return zero for a lexical axiom.

        Parameters
        ----------
        position : int
            The token position, which does not enter the weight.
        entry : Entry[Term]
            The entry admitted there, which does not enter the weight.
        """
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
        """Return zero for a rule application.

        Parameters
        ----------
        rule : Rule[Term]
            The rule that fired, which does not enter the weight.
        inputs : tuple[Term, ...]
            The premises, which do not enter the weight.
        output : Term
            The conclusion, which does not enter the weight.
        span : Cell
            The interval covered, which does not enter the weight.
        split : int | None
            The split point, which does not enter the weight.
        """
        del rule, inputs, output, span, split

        return 0.0


def _compositions(total: int, parts: int) -> Iterator[tuple[int, ...]]:
    """Yield the ways to write a total as an ordered sum of positive parts.

    Parameters
    ----------
    total : int
        The total to partition.
    parts : int
        The number of positive parts.
    """
    if parts == 1:
        if total > 0:
            yield (total,)

        return

    for first in range(1, total):
        for rest in _compositions(total - first, parts - 1):
            yield (first, *rest)


def _distinct[T](terms: Iterable[T], matching: Matching[T]) -> tuple[T, ...]:
    """Keep the first term in each matching class.

    Parameters
    ----------
    terms : Iterable[T]
        The terms to filter.
    matching : Matching[T]
        The equivalence relation over terms.
    """
    kept: list[T] = []

    for term in terms:
        if not any(matching.equivalent(term, prior) for prior in kept):
            kept.append(term)

    return tuple(kept)


def _inventory(assembly: GrammarAssembly) -> InventoryReport:
    """Return the category inventory of a compiled grammar's lexicon.

    Parameters
    ----------
    assembly : GrammarAssembly
        The compiled grammar with its lexicon and syntax.
    """
    store = assembly.grammar.store
    lexicon = assembly.lexicon
    categories = tuple(
        dict.fromkeys(
            entry.category for form in lexicon.forms for entry in lexicon.entries(form)
        )
    )
    occurring: set[str] = set()
    pending = list(categories)

    while pending:
        term = pending.pop()

        if term.op.is_constant:
            occurring.add(
                store.signature.constant_core(term.op.sort, store.payload_name(term))
            )
        elif not term.args:
            occurring.add(term.op.erases_to or term.op.name)

        pending.extend(term.args)

    rendered = tuple(
        (assembly.syntax.render(store, category), category) for category in categories
    )
    measure = assembly.category_measure
    return InventoryReport(
        {text: measure.depth(category) for text, category in rendered},
        {text: measure.arity(category) for text, category in rendered},
        {text: measure.order(category) for text, category in rendered},
        tuple(
            name
            for name in store.signature.vocabulary(
                assembly.language.category_sort
            ).names
            if name in occurring
        ),
        max_arguments=assembly.max_category_arguments,
        max_order=assembly.max_category_order,
    )


def _knobs(grammar: Grammar) -> GrammarKnobs:
    """Read the three grammar dimensions off a compiled grammar.

    Parameters
    ----------
    grammar : Grammar
        The compiled grammar.
    """
    sort = grammar.targets.categories[0].op.sort
    operators = tuple(op for op in grammar.signature.operators if op.sort == sort)
    vocabulary = grammar.signature.vocabulary(sort)
    primitives = sum(
        grammar.signature.constant_core(sort, name) == name for name in vocabulary.names
    )
    constructors = tuple(
        op.name
        for op in operators
        if op.arity > 0 and all(arg == sort for arg in op.arg_sorts)
    )
    return GrammarKnobs(
        primitives, constructors, tuple(rule.name for rule in grammar.rules)
    )


class _Table:
    """Derivation mass by category, tag and token count.

    Parameters
    ----------
    rules : RuleSet[Term]
        The rule inventory, with its matching and normal form.
    lexicon : StaticLexicon[Term]
        The static lexicon the leaves are drawn from.
    max_length : int
        The largest token count tabulated.
    unary_depth : int
        How many unary applications may stack over one item.
    """

    __slots__ = (
        "_base",
        "_into",
        "_masses",
        "_matching",
        "_memo",
        "_normal_form",
        "_rules",
        "_states",
        "_terms",
        "_unary_depth",
        "_weighting",
    )

    def __init__(
        self,
        rules: RuleSet[Term],
        lexicon: StaticLexicon[Term],
        max_length: int,
        unary_depth: int,
        weighting: RuleWeighting,
    ) -> None:
        self._rules = rules
        self._matching: Matching[Term] = rules.matching
        self._normal_form: NormalForm[Term] = rules.normal_form
        self._unary_depth = unary_depth
        self._weighting = weighting
        self._terms: dict[_State, Term] = {}
        self._memo: dict[tuple[int, tuple[Term, ...]], tuple[Term, ...]] = {}
        self._base: dict[_Premise, list[_Production]] = {}
        self._into: dict[_Premise, list[_UnaryStep]] = {}
        self._masses: dict[_Premise, float] = {}
        self._states: list[tuple[_State, ...]] = [()]

        for length in range(1, max_length + 1):
            base = self._lexical(lexicon) if length == 1 else self._stepped(length)
            self._close_unary(base, length)

    @property
    def states(self) -> list[tuple[_State, ...]]:
        """Return the states reached at each token count, indexed by count."""
        return self._states

    def mass(self, premise: _Premise) -> float:
        """Return the mass of the derivations concluding a state over a token count.

        Parameters
        ----------
        premise : _Premise
            The state and token count to read.
        """
        return self._masses.get(premise, 0.0)

    def _state(self, term: Term, tag: str) -> _State:
        """Return the state a conclusion packs on, keeping its first term.

        Parameters
        ----------
        term : Term
            The conclusion term.
        tag : str
            The conclusion's normal form tag.
        """
        matching = self._matching
        state = (matching.key(term) if matching.has_key else term, tag)
        self._terms.setdefault(state, term)

        return state

    def _apply(
        self, position: int, rule: Rule[Term], inputs: tuple[Term, ...]
    ) -> tuple[Term, ...]:
        """Return a rule's distinct conclusions over premises, memoized.

        Parameters
        ----------
        position : int
            The rule's position in the rule inventory.
        rule : Rule[Term]
            The rule to apply.
        inputs : tuple[Term, ...]
            The premise terms.
        """
        key = (position, inputs)
        held = self._memo.get(key)

        if held is None:
            held = _distinct(rule.apply(inputs), self._matching)
            self._memo[key] = held

        return held

    def _conclusions(
        self, position: int, rule: Rule[Term], premises: tuple[_State, ...]
    ) -> Iterator[tuple[_State, Term]]:
        """Yield the admitted conclusions of a rule over premise states.

        Parameters
        ----------
        position : int
            The rule's position in the rule inventory.
        rule : Rule[Term]
            The rule to apply.
        premises : tuple[_State, ...]
            The premise states.
        """
        inputs = tuple(self._terms[state] for state in premises)
        tags = tuple(tag for _, tag in premises)
        normal_form = self._normal_form

        for output in self._apply(position, rule, inputs):
            if normal_form.admits(rule, tags, inputs, output):
                yield self._state(output, normal_form.tag(rule, tags)), output

    def _lexical(self, lexicon: StaticLexicon[Term]) -> dict[_State, list[_Production]]:
        """Return the productions of every one-token item.

        Parameters
        ----------
        lexicon : StaticLexicon[Term]
            The source of lexical entries.
        """
        base: dict[_State, list[_Production]] = {}
        tag = self._normal_form.lexical_tag()

        for form in lexicon.forms:
            log_mass = lexicon.log_mass(form)

            for entry in lexicon.entries(form):
                state = self._state(entry.category, tag)
                base.setdefault(state, []).append(
                    _Production(
                        entry.category,
                        rule=None,
                        form=form,
                        premises=(),
                        mass=exp(log_mass + entry.weight),
                    )
                )

        return base

    def _stepped(self, length: int) -> dict[_State, list[_Production]]:
        """Return the non-unary productions of every item over a token count.

        Parameters
        ----------
        length : int
            The token count to tabulate.
        """
        base: dict[_State, list[_Production]] = {}

        for position, rule in enumerate(self._rules):
            if rule.arity < 2:
                continue

            for sizes in _compositions(length, rule.arity):
                pools = tuple(self._states[size] for size in sizes)

                if not all(pools):
                    continue

                for premises in product(*pools):
                    items = tuple(zip(premises, sizes, strict=True))
                    mass = prod(
                        self._masses[item] for item in items
                    ) * self._weighting.weight(rule.name)

                    for state, output in self._conclusions(position, rule, premises):
                        base.setdefault(state, []).append(
                            _Production(
                                output,
                                rule=rule.name,
                                form=None,
                                premises=items,
                                mass=mass,
                            )
                        )

        return base

    def _close_unary(self, base: dict[_State, list[_Production]], length: int) -> None:
        """Stack unary rules over the base items of one token count and weigh them.

        Unary rules are applied round by round, each round over the items the
        previous one concluded, for as many rounds as the unary depth allows,
        the way the chart parser closes one cell.

        Parameters
        ----------
        base : dict[_State, list[_Production]]
            The productions available before unary closure.
        length : int
            The token count of the items.
        """
        unary = tuple(
            (position, rule)
            for position, rule in enumerate(self._rules)
            if rule.arity == 1
        )
        scheduled: dict[_State, None] = dict.fromkeys(base)
        frontier = tuple(scheduled)

        for _ in range(self._unary_depth):
            following: list[_State] = []

            for child in frontier:
                for position, rule in unary:
                    for state, output in self._conclusions(position, rule, (child,)):
                        self._into.setdefault((state, length), []).append(
                            (
                                child,
                                rule.name,
                                output,
                                self._weighting.weight(rule.name),
                            )
                        )

                        if state not in scheduled:
                            scheduled[state] = None
                            following.append(state)

            frontier = tuple(following)

            if not frontier:
                break

        for state in scheduled:
            self._base[(state, length)] = base.get(state, [])

        for state in scheduled:
            self._masses[(state, length)] = sum(
                production.mass * prod(step[3] for step in path)
                for path in self._paths_into((state, length))
                for production in self._base[(path[0][0] if path else state, length)]
            )

        self._states.append(tuple(scheduled))

    def _paths_into(self, premise: _Premise) -> Iterator[tuple[_UnaryStep, ...]]:
        """Yield the unary chains ending at an item, the empty chain first.

        A chain lists its steps from the base item upward. Each step names the
        item it leaves, the rule applied and the conclusion drawn, and a chain
        visits no item twice.

        Parameters
        ----------
        premise : _Premise
            The destination state and token count.
        """
        state, length = premise

        def walk(
            at: _State, visited: frozenset[_State], remaining: int
        ) -> Iterator[tuple[_UnaryStep, ...]]:
            yield ()

            if remaining == 0:
                return

            for source, rule, output, factor in self._into.get((at, length), ()):
                if source in visited:
                    continue

                for tail in walk(source, visited | {source}, remaining - 1):
                    yield (*tail, (source, rule, output, factor))

        yield from walk(state, frozenset((state,)), self._unary_depth)

    def draw(self, random: Random, premise: _Premise) -> _Sampled:
        """Draw one derivation of an item in proportion to its mass.

        Parameters
        ----------
        random : Random
            The seeded generator the draw consumes.
        premise : _Premise
            The state and token count to derive.
        """
        state, length = premise
        remaining = random.random() * self._masses[premise]
        chosen = None

        for path in self._paths_into(premise):
            start = path[0][0] if path else state

            for production in self._base[(start, length)]:
                chosen = (production, path)
                remaining -= production.mass * prod(step[3] for step in path)

                if remaining < 0.0:
                    return self._realize(random, production, path)

        if chosen is None:
            msg = f"{premise!r} has no production to draw"
            raise RuntimeError(msg)

        # rounding left a sliver of mass past the last production
        return self._realize(random, *chosen)

    def _realize(
        self,
        random: Random,
        production: _Production,
        path: tuple[_UnaryStep, ...],
    ) -> _Sampled:
        """Build the drawn node from its base production and unary chain.

        Parameters
        ----------
        random : Random
            The generator used for recursive draws.
        production : _Production
            The base production to realize.
        path : tuple[_UnaryStep, ...]
            The unary chain above the base production.
        """
        node = _Sampled(
            production.term,
            rule=production.rule,
            form=production.form,
            children=tuple(
                self.draw(random, premise) for premise in production.premises
            ),
        )

        for _, rule, output, _ in path:
            node = _Sampled(output, rule=rule, form=None, children=(node,))

        return node


def _locate(
    chart: Chart[Term, float],
    matching: Matching[Term],
    key: ItemKey[Term],
    drawn: _Sampled,
) -> ChartDerivation[Term, float] | None:
    """Return the parser's derivation of an item with a drawn node's structure.

    A match uses the same rule, an equivalent category, and recursively
    matching premises in the same order. Lexical matches cover one token.

    Parameters
    ----------
    chart : Chart[Term, float]
        The parsed forest.
    matching : Matching[Term]
        The congruence categories are compared under.
    key : ItemKey[Term]
        The item to read.
    drawn : _Sampled
        The node whose rule, category and premises the derivation must share.

    Returns
    -------
    ChartDerivation[Term, float] | None
        The matching derivation, or nothing when the item has none.
    """
    for edge in chart.edges(key):
        if edge.rule_name != drawn.edge_name or not matching.equivalent(
            edge.output, drawn.term
        ):
            continue

        found: list[ChartDerivation[Term, float]] = []

        for child_key, child in zip(edge.children, drawn.children, strict=True):
            located = _locate(chart, matching, child_key, child)

            if located is None:
                break

            found.append(located)
        else:
            return ChartDerivation(key, edge, tuple(found))

    return None


def _derivation_count(chart: Chart[Term, float], roots: Iterable[ItemKey[Term]]) -> int:
    """Return how many derivations conclude the root items between them.

    An item's count is the sum over its edges of the product of its premises'
    counts, memoized by item. The parser closes each cell's unary rules into
    an acyclic graph, so the recursion bottoms out at the lexical edges.

    Parameters
    ----------
    chart : Chart[Term, float]
        The parsed forest.
    roots : Iterable[ItemKey[Term]]
        The items to count the derivations of.
    """
    memo: dict[ItemKey[Term], int] = {}

    def count(key: ItemKey[Term]) -> int:
        held = memo.get(key)

        if held is None:
            held = sum(
                prod(count(child) for child in edge.children)
                for edge in chart.edges(key)
            )
            memo[key] = held

        return held

    return sum(count(root) for root in roots)


class _Draws:
    """One sample's draws: the grammatical strings, their corruptions and forms.

    Parameters
    ----------
    generator : SyntheticGenerator
        The generator that parses and certifies the drawn strings.
    assembly : GrammarAssembly
        The compiled grammar the draws are from, with its lexicon and the
        bytes of its declaration.
    random : Random
        The seeded generator every choice consumes.
    table : _Table
        The derivation masses the draws are proportional to.
    roots : dict[int, tuple[tuple[_State, float], ...]]
        The root items by token count, each with its derivation mass.
    length : LengthDistribution
        The distribution each string's token count is drawn from. Its upper
        bound is the most tokens any drawn string, corrupted ones included,
        may have.
    ambiguity : AmbiguityTarget | None
        The bands a drawn string must fall in, or nothing to accept every
        draw.
    scope : ScopeTarget | None
        The band a drawn string's reading count must fall in, or nothing to
        leave the readings unmeasured.
    """

    __slots__ = (
        "_accepted",
        "_ambiguity",
        "_assembly",
        "_derivations",
        "_exceeded",
        "_forms",
        "_generator",
        "_inapplicable",
        "_injected",
        "_interpreted",
        "_leaves",
        "_length",
        "_lengths",
        "_lexical",
        "_measured",
        "_random",
        "_rejected",
        "_rejected_candidates",
        "_resampled",
        "_resisted",
        "_roots",
        "_scope",
        "_scope_counts",
        "_scope_rejected",
        "_shape",
        "_shape_measurements",
        "_shape_rejected",
        "_table",
        "_unadmitted",
        "_undecided",
    )

    def __init__(
        self,
        generator: SyntheticGenerator,
        assembly: GrammarAssembly,
        random: Random,
        table: _Table,
        roots: dict[int, tuple[tuple[_State, float], ...]],
        *,
        length: LengthDistribution,
        ambiguity: AmbiguityTarget | None,
        scope: ScopeTarget | None,
        shape: ShapeSchedule | None,
    ) -> None:
        self._generator = generator
        self._assembly = assembly
        self._random = random
        self._table = table
        self._roots = roots
        self._length = length
        self._ambiguity = ambiguity
        self._lengths: list[int] = []
        self._resampled = 0
        self._lexical: list[int] = []
        self._leaves: Counter[tuple[Term, str]] = Counter()
        self._derivations: list[int] = []
        self._rejected = 0
        self._accepted: Counter[str] = Counter()
        self._rejected_candidates: Counter[str] = Counter()
        self._inapplicable: Counter[str] = Counter()
        self._forms: list[GoldForm] = []
        self._interpreted: list[tuple[ItemId, Term]] = []
        self._injected: Counter[str] = Counter()
        self._resisted: Counter[str] = Counter()
        self._unadmitted: Counter[str] = Counter()
        self._scope = scope
        self._scope_counts: list[ReadingCount] = []
        self._scope_rejected = 0
        self._exceeded = 0
        self._undecided = 0
        self._measured: list[int] = []
        self._shape = shape
        self._shape_measurements: list[ShapeMeasurement] = []
        self._shape_rejected = 0

    def grammatical(
        self, index: int, seed: int, split: str | None = None
    ) -> Example[GoldDerivation[str]]:
        """Draw one grammatical string within the ambiguity bands and record it.

        Under an interpretation, the parser's derivation is read into the
        string's logical form, which is certified to type as a formula.

        Parameters
        ----------
        index : int
            The item's position in the sample.
        seed : int
            The seed recorded with the item.
        """
        drawn, chart, derivations, counted, measured_shape = self._admitted(split)
        forms = tuple(drawn.forms())
        assembly = self._assembly
        lexicon = assembly.lexicon
        self._lengths.append(len(forms))
        self._lexical.extend(len(lexicon.entries(form)) for form in forms)
        self._leaves.update(drawn.entries())
        self._derivations.append(derivations)
        if measured_shape is not None:
            self._shape_measurements.append(measured_shape)
        located = _located(assembly, drawn, chart)
        example = Example(
            _item(
                assembly.declaration_digest,
                self._generator.licensing,
                forms,
                index,
                seed,
                split=split,
            ),
            GoldDerivation(_gold(located, assembly.syntax, assembly.grammar.store)),
        )
        interpretation = self._generator.interpretation

        if interpretation is not None:
            meaning = interpretation.formula(
                interpretation.read(located, interpretation.primaries)[0]
            )
            item_id = example.item.item_id
            self._forms.append(GoldForm(item_id, interpretation.render(meaning)))
            self._interpreted.append((item_id, meaning))

        if counted is not None:
            self._scope_counts.append(
                ReadingCount(
                    example.item.item_id,
                    derivations=counted.derivations,
                    readings=counted.count,
                )
            )

        return example

    @property
    def forms(self) -> tuple[GoldForm, ...]:
        """Return the logical form of each grammatical item, in draw order."""
        return tuple(self._forms)

    def corrupted(
        self,
        request: Corruption[CorruptionOperator],
        index: int,
        seed: int,
        split: str | None = None,
    ) -> Corrupted:
        """Draw one corruption of a grammatical string the parser cannot derive.

        The source string is drawn within the same ambiguity bands as a
        grammatical item, and the operator at its declared relative weight.

        Parameters
        ----------
        request : Corruption[CorruptionOperator]
            The operators to draw among and the budget of candidates.
        index : int
            The item's position in the sample.
        seed : int
            The seed the sample is drawn with.

        Raises
        ------
        CorruptionExhaustedError
            If the budget of candidates runs out before one is non-derivable.
        """
        generator = self._generator
        random = self._random

        for _ in range(request.attempts):
            drawn, _, _, _, _ = self._admitted(split)
            source = tuple(drawn.forms())
            operator = request.choose(random)
            kind = operator.name
            candidate = operator.apply(random, source)

            if candidate is None:
                self._inapplicable[kind] += 1

                continue

            certificate = generator.certify(candidate)

            if certificate is None:
                self._rejected_candidates[kind] += 1

                continue

            self._accepted[kind] += 1

            return Corrupted(
                _item(
                    self._assembly.declaration_digest,
                    generator.licensing,
                    candidate,
                    index,
                    seed,
                    split=split,
                ),
                kind,
                source,
                certificate,
            )

        msg = (
            f"{request.attempts} candidates from {request.operator_names!r} were all "
            f"derivable or inapplicable"
        )
        raise CorruptionExhaustedError(msg)

    def ill_typed(
        self, request: Corruption[IllTypingOperator], count: int
    ) -> tuple[IllTyped, ...]:
        """Inject refused logical forms among the interpreted items.

        The items whose forms are corrupted are drawn without replacement and
        injected in draw order, and the operator is drawn uniformly among the
        declared weights. A candidate the checker accepts is discarded and
        drawn again.

        Parameters
        ----------
        request : Corruption[IllTypingOperator]
            The operators to draw among and the budget of candidates.
        count : int
            How many forms to inject.

        Raises
        ------
        ValueError
            If the generator has no interpretation, or more forms are asked
            for than the sample holds interpreted items.
        CorruptionExhaustedError
            If the budget of candidates for one form runs out before one is
            refused.
        """
        interpreted = self._interpreted

        if count > len(interpreted):
            msg = (
                f"{count} ill-typed forms are asked of a sample with "
                f"{len(interpreted)} interpreted items"
            )
            raise ValueError(msg)

        drawn = sorted(self._random.sample(range(len(interpreted)), count))

        return tuple(self._injected_form(request, *interpreted[at]) for at in drawn)

    def ill_typing_report(
        self, request: Corruption[IllTypingOperator], forms: tuple[IllTyped, ...]
    ) -> IllTypingReport:
        """Return what the injected forms achieved against what was requested.

        Parameters
        ----------
        request : Corruption[IllTypingOperator]
            The proportion, operators and budget the forms were injected
            with.
        forms : tuple[IllTyped, ...]
            The injected forms, whose certificates the report counts.
        """
        return IllTypingReport.of(
            CorruptionReport(
                request,
                accepted=self._injected,
                rejected=self._resisted,
                inapplicable=self._unadmitted,
                sample_size=len(self._interpreted),
            ),
            forms,
        )

    def _injected_form(
        self, request: Corruption[IllTypingOperator], item_id: ItemId, form: Term
    ) -> IllTyped:
        """Corrupt one well-typed form until the checker refuses the result.

        Parameters
        ----------
        request : Corruption[IllTypingOperator]
            The operators to draw among and the budget of candidates.
        item_id : ItemId
            The item the form belongs to.
        form : Term
            The well-typed form to corrupt.

        Raises
        ------
        ValueError
            If the generator has no interpretation.
        CorruptionExhaustedError
            If the budget of candidates runs out before one is refused.
        """
        interpretation = self._generator.interpretation

        if interpretation is None:
            msg = "forms are injected under an interpretation, and none is declared"
            raise ValueError(msg)

        random = self._random

        for _ in range(request.attempts):
            operator = request.choose(random)
            kind = operator.name
            candidate = operator.apply(random, form)

            if candidate is None:
                self._unadmitted[kind] += 1

                continue

            certificate = interpretation.failure(candidate)

            if certificate is None:
                self._resisted[kind] += 1

                continue

            self._injected[kind] += 1
            rendered = interpretation.render(candidate)
            policy = request.policy
            label_key = (
                f"{item_id}\0{rendered}\0{policy.regime}\0{policy.noise!r}"
            ).encode()
            label_draw = int.from_bytes(sha256(label_key).digest()[:8]) / 2**64

            return IllTyped(
                item_id,
                kind,
                interpretation.render(form),
                rendered,
                certificate,
                observed_label=(
                    "well-typed" if policy.flips(label_draw) else "ill-typed"
                ),
            )

        msg = (
            f"{request.attempts} candidates from {request.operator_names!r} all typed "
            f"or were inapplicable to the form of {item_id!r}"
        )
        raise CorruptionExhaustedError(msg)

    def ambiguity_report(self) -> AmbiguityReport:
        """Return the ambiguity measured over the grammatical draws."""
        return AmbiguityReport(
            Histogram.of(self._lexical),
            Histogram.of(self._derivations),
            rejected=self._rejected,
            target=self._ambiguity,
        )

    def scope_report(self) -> ScopeReport | None:
        """Return the readings measured over the grammatical draws, if targeted."""
        scope = self._scope

        if scope is None:
            return None

        return ScopeReport(
            tuple(self._scope_counts),
            rejected=self._scope_rejected,
            exceeded=self._exceeded,
            undecided=self._undecided,
            target=scope,
        )

    def shape_report(self) -> ShapeReport | None:
        """Return accepted derivation shapes, when a target was declared."""
        target = self._shape

        if target is None:
            return None

        return ShapeReport(
            tuple(self._shape_measurements),
            rejected=self._shape_rejected,
            target=target,
        )

    def length_report(self) -> LengthReport:
        """Return the token counts measured over the grammatical draws."""
        return LengthReport(
            Histogram.of(self._lengths),
            resampled=self._resampled,
            distribution=self._length,
        )

    def lexicon_report(self) -> LexiconReport:
        """Return the joint of category and frequency measured over the draws."""
        assembly = self._assembly
        lexicon = assembly.lexicon
        by_category: dict[Term, list[int]] = {}
        by_form: list[int] = []
        ambiguities: list[int] = []

        for form in lexicon.forms:
            drawn = 0
            ambiguities.append(len(lexicon.entries(form)))

            for entry in lexicon.entries(form):
                held = self._leaves.get((entry.category, form), 0)
                by_category.setdefault(entry.category, []).append(held)
                drawn += held

            by_form.append(drawn)

        return LexiconReport(
            {
                assembly.syntax.render(assembly.grammar.store, category): Histogram.of(
                    counts
                )
                for category, counts in by_category.items()
            },
            Histogram.of(by_form),
            Histogram.of(ambiguities),
            spearman(
                [float(value) for value in ambiguities],
                [float(value) for value in by_form],
            ),
        )

    def corruption_report(
        self, request: Corruption[CorruptionOperator], sample_size: int
    ) -> CorruptionReport[CorruptionOperator]:
        """Return what the corrupted draws achieved against what was requested.

        Parameters
        ----------
        request : Corruption[CorruptionOperator]
            The proportion, operators and budget the sample was drawn with.
        sample_size : int
            How many items the whole sample holds.
        """
        return CorruptionReport(
            request,
            accepted=self._accepted,
            rejected=self._rejected_candidates,
            inapplicable=self._inapplicable,
            sample_size=sample_size,
        )

    def _admitted(
        self,
        split: str | None,
    ) -> tuple[
        _Sampled,
        Chart[Term, float],
        int,
        Readings | None,
        ShapeMeasurement | None,
    ]:
        """Draw strings until one falls within the ambiguity and readings bands.

        A drawn string is held against the lexical band first, on the mean
        number of lexical categories its tokens' forms have, then against
        the derivation band, then against the readings band. A string outside
        either ambiguity band is counted as rejected and drawn again.

        Raises
        ------
        AmbiguityUnreachableError
            If the budget of draws runs out before one meets the lexical and
            derivation bands. The message names the per-string means and
            derivation counts the rejected draws reached.
        ScopeUnreachableError
            If the budget of draws runs out before one meets the readings
            band.
        """
        ambiguity = self._ambiguity
        scope = self._scope
        schedule = self._shape
        shape = None if schedule is None else schedule.target(split)
        lexical = None if ambiguity is None else ambiguity.lexical
        band = None if ambiguity is None else ambiguity.derivations
        attempts = max(
            1 if ambiguity is None else ambiguity.attempts,
            1 if scope is None else scope.attempts,
            1 if shape is None else shape.attempts,
        )
        interpretation = self._generator.interpretation
        lexicon = self._assembly.lexicon
        means: list[float] = []
        counts: list[int] = []
        refused = 0
        missed_shapes: list[ShapeMeasurement] = []

        for _ in range(attempts):
            drawn = self._draw()
            forms = tuple(drawn.forms())
            measured_shape = None if shape is None else _shape(drawn)

            if (
                shape is not None
                and measured_shape is not None
                and not shape.admits(measured_shape)
            ):
                self._shape_rejected += 1
                missed_shapes.append(measured_shape)

                continue

            if lexical is not None:
                mean = sum(len(lexicon.entries(form)) for form in forms) / len(forms)

                if not lexical.admits(mean):
                    self._rejected += 1
                    means.append(mean)

                    continue

            chart = self._generator.parse(forms)
            derivations = _derivation_count(
                chart, self._generator.grammar.goal.roots(chart)
            )

            if band is not None and not band.admits(derivations):
                self._rejected += 1
                counts.append(derivations)

                continue

            if scope is None or interpretation is None:
                return drawn, chart, derivations, None, measured_shape

            if derivations > scope.derivations:
                self._exceeded += 1
                refused += 1

                continue

            held = readings(
                interpretation,
                chart,
                self._generator.grammar.goal.roots(chart),
                undecided=scope.undecided,
            )

            if held.derivations > scope.derivations:
                self._exceeded += 1
                refused += 1

                continue

            self._undecided += held.undecided
            self._measured.append(held.count)
            wanted = scope.readings

            if wanted is None or wanted.admits(held.count):
                return drawn, chart, derivations, held, measured_shape

            self._scope_rejected += 1
            refused += 1

        if missed_shapes:
            msg = (
                f"{attempts} draws all fell outside the derivation-shape target "
                f"{shape!r}; rejected measurements were {missed_shapes!r}"
            )
            raise ShapeUnreachableError(msg)

        if scope is not None and refused:
            msg = (
                f"{attempts} draws all fell outside the readings target "
                f"{scope.readings!r}, which the draws reach at "
                f"{Histogram.of(self._measured)!r} with {self._exceeded} draws "
                f"beyond {scope.derivations} derivations"
            )
            raise ScopeUnreachableError(msg)

        missed: list[str] = []

        if means:
            spread = (
                ""
                if min(means) == max(means)
                else f", from {min(means):.4g} to {max(means):.4g} per string"
            )
            missed.append(
                f"{len(means)} outside the lexical target {lexical!r} at a mean "
                f"of {sum(means) / len(means):.4g} categories per token{spread}"
            )

        if counts:
            missed.append(
                f"{len(counts)} outside the derivation target {band!r} at "
                f"{Histogram.of(counts)!r} derivations per string"
            )

        msg = f"{attempts} draws all fell outside the ambiguity target: " + (
            ", and ".join(missed)
        )
        raise AmbiguityUnreachableError(msg)

    def _draw(self) -> _Sampled:
        """Draw one root derivation: a token count, a root item, then a derivation.

        A token count the grammar derives no goal at, or beyond the tabulated
        bound, is counted and drawn again.
        """
        random = self._random
        length = self._draw_length()

        while length not in self._roots:
            self._resampled += 1
            length = self._draw_length()

        found = self._roots[length]
        remaining = random.random() * sum(total for _, total in found)

        for state, total in found:
            remaining -= total

            if remaining < 0.0:
                return self._table.draw(random, (state, length))

        # rounding left a sliver of mass past the last root item
        return self._table.draw(random, (found[-1][0], length))

    def _draw_length(self) -> int:
        """Return one in-support length under the declared boundary policy."""
        distribution = self._length

        while True:
            length = distribution.draw(self._random)

            if distribution.min_length <= length <= distribution.max_length:
                return length

            match distribution.out_of_support:
                case "resample":
                    self._resampled += 1
                case "clip":
                    return min(
                        distribution.max_length,
                        max(distribution.min_length, length),
                    )
                case "error":
                    msg = (
                        f"length draw {length} lies outside declared support "
                        f"[{distribution.min_length}, {distribution.max_length}]"
                    )
                    raise LengthOutOfSupportError(msg)


class SyntheticGenerator:
    """Draws grammatical strings, with gold derivations, from a compiled grammar.

    Parameters
    ----------
    assembly : GrammarAssembly
        The compiled grammar with its lexicon and syntax, beside the canonical
        bytes of the declaration it was lowered from.
    licensing : Licensing
        The terms every drawn item is released under.
    interpretation : InterpretationFunction | None
        The interpretation every drawn string is read under, or nothing to
        draw strings without logical forms.
    policy : ParsePolicy[Literal[Retain.FOREST]]
        The run-owned packed forest parsing policy.
    """

    __slots__ = (
        "_assembly",
        "_interpretation",
        "_knobs",
        "_licensing",
        "_policy",
        "_weighting",
    )

    def __init__(
        self,
        assembly: GrammarAssembly,
        *,
        policy: ParsePolicy[Literal[Retain.FOREST]],
        licensing: Licensing = CC0,
        interpretation: InterpretationFunction | None = None,
        weighting: RuleWeighting = _UNIFORM_RULE_WEIGHTING,
    ) -> None:
        if policy.retain is not Retain.FOREST:
            msg = "synthetic generation requires packed-forest retention"
            raise ValueError(msg)
        self._assembly = assembly
        self._knobs = _knobs(assembly.grammar).with_inventory(
            None if interpretation is None else interpretation.inventory
        )
        self._licensing = licensing
        self._interpretation = interpretation
        self._policy = policy
        self._weighting = weighting

    @property
    def knobs(self) -> GrammarKnobs:
        """Return the dimensions of the grammar drawn from."""
        return self._knobs

    @property
    def interpretation(self) -> InterpretationFunction | None:
        """Return the interpretation strings are read under, if one is declared."""
        return self._interpretation

    @property
    def grammar(self) -> Grammar:
        """Return the compiled grammar drawn from."""
        return self._assembly.grammar

    @property
    def lexicon(self) -> StaticLexicon[Term]:
        """Return the static lexicon drawn from."""
        return self._assembly.lexicon

    @property
    def licensing(self) -> Licensing:
        """Return the terms every drawn item is released under."""
        return self._licensing

    def sample(
        self,
        count: int,
        /,
        *,
        seed: int,
        splits: Sequence[str] = (),
        length: LengthDistribution,
        ambiguity: AmbiguityTarget | None = None,
        shape: ShapeSchedule | None = None,
        scope: ScopeTarget | None = None,
        corruption: Corruption[CorruptionOperator] | None = None,
        noise: LabelNoise | None = None,
        ill_typing: Corruption[IllTypingOperator] | None = None,
    ) -> SyntheticCorpus:
        """Draw a corpus of grammatical strings, and corruptions of them if asked.

        Parameters
        ----------
        count : int
            How many items to draw, corrupted ones included.
        seed : int
            The seed of the local random generator. Equal seeds draw equal
            corpora.
        length : LengthDistribution
            The distribution a string's token count is drawn from, within
            its bounds. A count at which no goal category is derivable, or
            beyond the upper bound, is discarded and drawn again, and the
            corpus reports how often. The derivation is then drawn among the
            derivations of that count in proportion to the product of the
            masses of its lexical entries, an entry's mass being the weight
            its lexicon was built with. A corrupted string is one token
            shorter than its source under deletion, so it may fall one below
            the lower bound.
        ambiguity : AmbiguityTarget | None
            The lexical and derivational ambiguity to aim at. A drawn string
            whose mean number of lexical categories per token, or whose root
            derivation count, falls outside its band is discarded and
            redrawn, within the target's budget, and the corpus reports what
            the accepted strings measure and how many were discarded.
        shape : ShapeSchedule | None
            The split-conditioned derivation depth, right-branching share,
            forward-slash share and mean rule arity bands to aim at. A
            derivation outside any selected band is discarded within that
            target's budget.
        scope : ScopeTarget | None
            The semantic ambiguity to aim at. A drawn string's derivations
            are read into logical forms and grouped by mutual entailment
            under the interpretation's oracle, and a string whose count of
            groups falls outside the band, or whose derivations outnumber
            the bound the readings are read from, is discarded and redrawn
            within the target's budget. The corpus reports the derivation
            count against the reading count of every accepted string.
        corruption : Corruption[CorruptionOperator] | None
            The share of the items to emit as corrupted strings, the operators
            to corrupt with and the budget per item. The number of corrupted
            items is that share of the count rounded to the nearest integer,
            a half rounding up, and their positions are drawn uniformly
            among the item indices. Every emitted corruption is one the
            parser finds non-derivable, and its source string is drawn within
            the same ambiguity bands as a grammatical item.
        noise : LabelNoise | None
            The rates at which items' grammaticality labels are flipped
            before emission, one rate per true label. The flips are drawn
            after every item, one independent draw per item whatever its true
            label, so the items drawn at a seed are the same at every rate
            and with no noise at all. Every item's record keeps its true
            label beside the observed one, and the corpus reports the flips by
            true label.
        ill_typing : Corruption[IllTypingOperator] | None
            The share of the grammatical items whose logical form to corrupt
            into one the type checker refuses, the operators to corrupt with
            and the budget per form. The number of injected forms is that
            share of the grammatical items rounded to the nearest integer, a
            half rounding up, and the items are drawn without replacement.
            The forms are drawn after every item and label, so the items and
            labels drawn at a seed are the same with and without them. Every
            emitted form carries the constraints the checker left unsatisfied
            on it, and a candidate the checker accepts is discarded and drawn
            again.

        Returns
        -------
        SyntheticCorpus
            The items in draw order, beside the declaration and theory bytes
            of the grammar they were drawn from, one label record per item
            and the reports of what the draws achieved, among them the joint
            of category and token frequency measured over the grammatical
            items. Under an interpretation, every grammatical item's logical
            form.

        Raises
        ------
        TypeError
            If the count or the seed is not an integer.
        ValueError
            If the count is negative, or readings or injected forms are
            requested without an interpretation.
        UninterpretableDerivationError
            If the interpretation assigns a drawn derivation no form.
        IllTypedFormError
            If a drawn form does not type as a formula.
        UnproductiveGrammarError
            If no goal category is derivable within the length bounds.
        UncertifiedDerivationError
            If the parser does not rebuild a drawn derivation, which happens
            when the unary rules cycle over one category.
        AmbiguityUnreachableError
            If the draws run out of budget before one meets the lexical and
            derivation targets.
        ScopeUnreachableError
            If the draws run out of budget before one meets the readings
            target.
        CorruptionExhaustedError
            If the candidates for one corrupted item run out of budget before
            one is non-derivable, or the candidates for one injected form run
            out before one is refused.
        """
        held_count = ensure_non_negative("count", count)
        lower = length.min_length
        bound = length.max_length

        if isinstance(seed, bool) or not isinstance(seed, int):
            msg = f"seed is not an integer: {seed!r}"
            raise TypeError(msg)
        held_splits = tuple(splits)
        if held_splits and len(held_splits) != held_count:
            msg = (
                f"a sample of {held_count} items received {len(held_splits)} "
                "split assignments"
            )
            raise ValueError(msg)
        if shape is not None:
            named_splits: tuple[str | None, ...] = (
                (None,) if not held_splits else tuple(sorted(set(held_splits)))
            )
            missing = tuple(
                split for split in named_splits if shape.target(split) is None
            )
            if missing:
                msg = f"the shape schedule has no target for splits {missing!r}"
                raise ValueError(msg)

        interpretation = self._interpretation

        if interpretation is not None:
            interpretation.open_proof_step()

        if scope is not None and interpretation is None:
            msg = "readings are read under an interpretation, and none is declared"
            raise ValueError(msg)

        if ill_typing is not None and interpretation is None:
            msg = "forms are injected under an interpretation, and none is declared"
            raise ValueError(msg)

        grammar = self._assembly.grammar
        table = _Table(
            grammar.rules,
            self._assembly.lexicon,
            bound,
            grammar.budget.unary_rounds,
            self._weighting,
        )
        roots = self._roots(table, lower, bound)

        if not roots:
            goals = ", ".join(
                self._assembly.syntax.render(grammar.store, category)
                for category in grammar.targets.categories
            )
            msg = (
                f"grammar {grammar.name!r} derives no string of at least {lower} "
                f"and at most {bound} tokens with root category {goals}"
            )
            raise UnproductiveGrammarError(msg)

        random = Random(seed)
        draws = _Draws(
            self,
            self._assembly,
            random,
            table,
            roots,
            length=length,
            ambiguity=ambiguity,
            scope=scope,
            shape=shape,
        )
        examples: list[Example[GoldDerivation[str]]] = []
        corrupted: list[Corrupted] = []
        truths: list[tuple[ItemId, GrammaticalityLabel]] = []
        report = None

        if corruption is None:
            for index in range(held_count):
                split = None if not held_splits else held_splits[index]
                example = draws.grammatical(index, seed, split)
                examples.append(example)
                truths.append((example.item.item_id, "grammatical"))
        else:
            slots = frozenset(
                random.sample(
                    range(held_count),
                    corruption.count(held_count, random),
                )
            )

            for index in range(held_count):
                split = None if not held_splits else held_splits[index]
                if index in slots:
                    record = draws.corrupted(corruption, index, seed, split)
                    corrupted.append(record)
                    truths.append((record.item.item_id, "ungrammatical"))
                else:
                    example = draws.grammatical(index, seed, split)
                    examples.append(example)
                    truths.append((example.item.item_id, "grammatical"))

            report = draws.corruption_report(corruption, held_count)

        labels = _labels(truths, noise, random)
        ill_typed: tuple[IllTyped, ...] = ()
        injected = None

        if ill_typing is not None:
            ill_typed = draws.ill_typed(
                ill_typing, ill_typing.count(len(examples), random)
            )
            injected = draws.ill_typing_report(ill_typing, ill_typed)

        return SyntheticCorpus(
            self._assembly.declaration,
            grammar.theory_specification,
            seed=seed,
            length=draws.length_report(),
            knobs=self._knobs,
            examples=tuple(examples),
            ambiguity=draws.ambiguity_report(),
            lexicon=draws.lexicon_report(),
            inventory=_inventory(self._assembly),
            corrupted=tuple(corrupted),
            corruption=report,
            labels=labels,
            noise=None if noise is None else NoiseReport.of(noise, labels),
            forms=draws.forms,
            ill_typed=ill_typed,
            ill_typing=injected,
            scope=draws.scope_report(),
            shape=draws.shape_report(),
        )

    def infer(
        self,
        count: int,
        /,
        *,
        seed: int,
        splits: Sequence[str] = (),
        premise: LengthDistribution,
        hypothesis: LengthDistribution,
        candidates: int,
        attempts: int,
        scope: ScopeTarget | None = None,
        ill_typing: Corruption[IllTypingOperator] | None = None,
        complexity: PairComplexityTarget | None = None,
        proportion: LabelProportion | None = None,
        noise: InferenceNoise | None = None,
        relations: PairRelations,
        pairing: PairingStrategy = INDEPENDENT_PAIRING,
    ) -> SyntheticInference:
        """Draw and label premises and hypotheses under complexity controls.

        Each side is drawn as its own sample, so the two lengths are
        controlled apart, and a pair is a premise drawn from the first beside
        a hypothesis drawn from the second. A candidate whose two forms fall
        outside a complexity band is discarded and drawn again. The oracle
        then decides the pair twice under the interpretation's clock: once on
        the hypothesis and, if that reaches no proof, once on its negation,
        so a pair is labeled by the relation the premise bears to the
        hypothesis rather than by one verdict. Both forms are retained beside
        the pair.

        Parameters
        ----------
        count : int
            How many labeled pairs to draw.
        seed : int
            The seed of the draw. Equal seeds draw equal runs.
        splits : Sequence[str]
            One partition per pair. Each partition receives its own candidate
            pools, so a sentence item cannot occur in more than one partition.
        premise : LengthDistribution
            The distribution a premise string's token count is drawn from.
        hypothesis : LengthDistribution
            The distribution a hypothesis string's token count is drawn from.
        candidates : int
            How many strings to draw on each side and within each partition to
            pair among.
        attempts : int
            How many candidate pairs one accepted pair may take before the
            complexity target or the label proportion is declared unreachable.
        scope : ScopeTarget | None
            The readings each side's strings are drawn within, or nothing to
            leave the readings unmeasured. A string outside the band is
            discarded and drawn again before any pair is drawn among the
            strings.
        ill_typing : Corruption[IllTypingOperator] | None
            The share of each side's strings whose logical form to corrupt
            into one the type checker refuses, or nothing to corrupt none.
            The forms are drawn after the side's strings, so the pairs drawn
            at a seed are the same with and without them.
        complexity : PairComplexityTarget | None
            The bands each side's form and the pair's predicate overlap must
            fall in, or nothing to leave the complexity uncontrolled.
        proportion : LabelProportion | None
            The share of the decided pairs each relation is aimed at, or
            nothing to take the labels as they fall.
        noise : InferenceNoise | None
            The rate at which a pair's relation is flipped, and where the
            flips land. The flips are drawn after every pair, two
            independent draws per pair whatever the relation, so the pairs
            drawn at a seed are the same at every rate and with no noise at
            all. A pair the oracle left undecided bears no relation to flip.
            Every record keeps the oracle's relation beside the emitted one,
            and the run reports the confusion matrix the flips achieved.
        relations : PairRelations
            The share of the pairs that are trivial, whose two forms are one
            term, and how complex an entailing pair must be. A trivial
            candidate beyond the share, a candidate that is not trivial once
            only trivial pairs can still fill the share, and an entailing
            candidate below the least complexity are refused and drawn again.
        pairing : PairingStrategy
            How one candidate is selected from the two measured pools.

        Returns
        -------
        SyntheticInference
            The labeled pairs beside the sample each side was drawn as, the
            achieved complexity distributions, the achieved label counts and,
            when noise was requested, the achieved confusion matrix.

        Raises
        ------
        ValueError
            If the generator has no interpretation, the declared fragment has
            no negation to decide a contradiction with, the count is negative,
            or the candidates or the attempt budget is not positive.
        FragmentEscapeError
            If a drawn form quantifies outside the declared fragment.
        ScopeUnreachableError
            If a side's draws run out of budget before one meets the readings
            target.
        CorruptionExhaustedError
            If the candidates for one injected form run out before one is
            refused.
        ComplexityUnreachableError
            If no candidate within the attempt budget falls in every band. The
            message names what the candidates measured.
        ProportionUnreachableError
            If candidates within the bands were drawn but none bore the
            relation a declared proportion's quota still wants.
        PairRelationUnreachableError
            If every candidate within the attempt budget was refused by the
            relation policy.
        """
        interpretation = self._interpretation

        if interpretation is None:
            msg = "pairs are labeled under an interpretation, and none is declared"
            raise ValueError(msg)

        profile = interpretation.language.profile
        fragment = profile.fragment
        negation_name = profile.negation

        if negation_name is None:
            msg = "a contradiction needs a designated negation constant"
            raise ValueError(msg)

        held = ensure_non_negative("a pair count", count)
        held_splits = tuple(splits)
        if held_splits and len(held_splits) != held:
            msg = (
                f"a draw of {held} inference pairs has {len(held_splits)} "
                "split assignments"
            )
            raise ValueError(msg)
        width = ensure_positive("a candidate count", candidates)
        budget = ensure_positive("an attempt budget", attempts)
        random = Random(seed)
        partitions = tuple(dict.fromkeys(held_splits))
        sample_width = width * max(1, len(partitions))
        left = self.sample(
            sample_width,
            seed=random.randrange(2**32),
            length=premise,
            scope=scope,
            ill_typing=ill_typing,
        )
        right = self.sample(
            sample_width,
            seed=random.randrange(2**32),
            length=hypothesis,
            scope=scope,
            ill_typing=ill_typing,
        )
        store = interpretation.store
        meter = ComplexityMeter(
            store,
            interpretation.language.profile,
            abstraction=interpretation.calculus.operators.lam,
        )
        controls = meter.controls
        negation = interpretation.language.constant(negation_name)
        admitted = fragment.quantifiers
        premises = _measured(interpretation, meter, left, quantifiers=admitted)
        hypotheses = _measured(interpretation, meter, right, quantifiers=admitted)
        interpretation.open_proof_step()
        quotas = None if proportion is None else proportion.quotas(held)
        trivial_quota = relations.quota(held)
        trivial = 0
        screened = 0
        filled: Counter[InferenceLabel] = Counter()
        pairs: list[InferencePair] = []
        measured: list[tuple[FormComplexity, FormComplexity]] = []
        outside = 0
        unwanted = 0
        partition_offsets = {
            partition: position * width for position, partition in enumerate(partitions)
        }

        for _ in range(held):
            pair_split = None if not held_splits else held_splits[len(pairs)]
            start = 0 if pair_split is None else partition_offsets[pair_split]
            stop = start + width
            premise_pool = premises[start:stop]
            hypothesis_pool = hypotheses[start:stop]
            wanted = None if quotas is None else _wanted(quotas, filled)
            missed: list[tuple[FormComplexity, FormComplexity]] = []
            reached: Counter[InferenceLabel] = Counter()
            undecided_reached = 0
            refused = 0

            for _ in range(budget):
                left_at, right_at = pairing.choose(
                    random,
                    tuple(measure for _, _, measure in premise_pool),
                    tuple(measure for _, _, measure in hypothesis_pool),
                )
                assumed, assumed_form, assumed_measure = premise_pool[left_at]
                decided, decided_form, decided_measure = hypothesis_pool[right_at]

                if complexity is not None and not complexity.admits(
                    assumed_measure, decided_measure
                ):
                    outside += 1
                    missed.append((assumed_measure, decided_measure))
                    continue

                distance = relation_complexity(assumed_form, decided_form)
                remaining = held - len(pairs)

                if (distance == 0 and trivial >= trivial_quota) or (
                    distance != 0 and trivial_quota - trivial >= remaining
                ):
                    screened += 1
                    refused += 1
                    continue

                label, forward, negated = _relation(
                    interpretation,
                    interpretation.language.apply(negation, decided_form),
                    assumed_form,
                    decided_form,
                )

                if not relations.admits(label, distance):
                    screened += 1
                    refused += 1
                    continue

                if wanted is not None and label != wanted:
                    unwanted += 1

                    if label is None:
                        undecided_reached += 1
                    else:
                        reached[label] += 1

                    continue

                if distance == 0:
                    trivial += 1

                pairs.append(
                    InferencePair(
                        assumed,
                        decided,
                        label=label,
                        entailment=forward,
                        contradiction=negated,
                        split=pair_split,
                    )
                )
                measured.append((assumed_measure, decided_measure))
                if label is not None:
                    filled[label] += 1

                break
            else:
                if proportion is not None and (reached or undecided_reached):
                    raise ProportionUnreachableError(
                        proportion,
                        filled,
                        attempts=budget,
                        reached=reached,
                        undecided=undecided_reached,
                    )

                if refused and not reached and len(missed) < budget:
                    msg = (
                        f"{refused} of {budget} candidate pairs were refused by "
                        f"{relations!r} and no other was accepted"
                    )

                    raise PairRelationUnreachableError(msg)

                msg = (
                    f"no candidate pair among {budget} falls in every declared "
                    f"complexity band {complexity!r}; the candidates measured "
                    f"{_measurements(complexity, missed)}"
                )

                raise ComplexityUnreachableError(msg)

        held_pairs = tuple(pairs)
        labels = _noised(held_pairs, noise, random)
        semantic_lexicon = interpretation.semantic_lexicon

        return SyntheticInference(
            seed=seed,
            pairs=held_pairs,
            labels=labels,
            noise=None if noise is None else ConfusionReport.of(noise, labels),
            premises=left,
            hypotheses=right,
            complexity=ComplexityReport(
                {
                    control: Histogram.of(
                        assumed.value(control) for assumed, _ in measured
                    )
                    for control in controls
                },
                {
                    control: Histogram.of(
                        decided.value(control) for _, decided in measured
                    )
                    for control in controls
                },
                overlap=Histogram.of(
                    assumed.overlap(decided) for assumed, decided in measured
                ),
                rejected=outside,
                target=complexity,
            ),
            inference=InferenceReport(
                filled,
                undecided=len(held_pairs) - sum(filled.values()),
                requested=proportion,
                rejected=unwanted,
            ),
            relations=RelationReport(
                relations, pairs=len(held_pairs), trivial=trivial, screened=screened
            ),
            semantic=(
                None
                if semantic_lexicon is None
                else SemanticReport.of(semantic_lexicon, (left, right))
            ),
        )

    def parse(self, forms: tuple[str, ...]) -> PackedForest[Term, float]:
        """Parse a token sequence under the grammar with neutral weights.

        Parameters
        ----------
        forms : tuple[str, ...]
            The surface forms, in order.
        """
        assembly = self._assembly
        grammar = assembly.grammar

        return parse(
            forms,
            rules=grammar.rules,
            source=assembly.lexicon,
            semiring=LogInside(),
            scorer=_NeutralScorer(),
            goal=grammar.goal,
            category_measure=assembly.category_measure,
            max_category_arity=grammar.budget.max_category_arity,
            max_category_size=grammar.budget.max_category_size,
            unary_depth=grammar.budget.unary_rounds,
            threads=grammar.budget.threads,
            policy=self._policy,
        )

    def derivation_count(self, forms: tuple[str, ...]) -> int:
        """Return how many root derivations the parser finds for a token sequence.

        Parameters
        ----------
        forms : tuple[str, ...]
            The surface forms, in order.
        """
        chart = self.parse(forms)

        return _derivation_count(chart, self._assembly.grammar.goal.roots(chart))

    def readings(self, forms: tuple[str, ...]) -> Readings:
        """Return the logical forms of a token sequence, one per reading.

        Every derivation the parser finds is read into a logical form, and
        two forms that entail each other under the interpretation's oracle,
        within the interpretation's clock, are one reading.

        Parameters
        ----------
        forms : tuple[str, ...]
            The surface forms, in order.

        Raises
        ------
        ValueError
            If the generator has no interpretation.
        """
        interpretation = self._interpretation

        if interpretation is None:
            msg = "readings are read under an interpretation, and none is declared"
            raise ValueError(msg)

        interpretation.open_proof_step()
        chart = self.parse(forms)

        return readings(
            interpretation,
            chart,
            self._assembly.grammar.goal.roots(chart),
        )

    def certify(self, forms: tuple[str, ...]) -> NonDerivability | None:
        """Return the certificate of a sequence the parser cannot derive.

        Parameters
        ----------
        forms : tuple[str, ...]
            The surface forms, in order.

        Returns
        -------
        NonDerivability | None
            The grammar and budget the parse ran under, or nothing when the
            parser finds a root derivation.
        """
        if self.derivation_count(forms) > 0:
            return None

        grammar = self._assembly.grammar

        return NonDerivability(
            self._assembly.declaration_digest,
            grammar.theory_digest,
            unary_rounds=grammar.budget.unary_rounds,
        )

    def verify_inference(self, pair: InferencePair) -> bool:
        """Return whether relabeling a pair from its retained forms reproduces it.

        Both forms are read back from the text the pair retains and the
        relation is decided again under the certificates' clock. A pair whose
        label, or either of whose certificates, differs from what the fresh
        decision reaches is declined.

        Parameters
        ----------
        pair : InferencePair
            The pair to decide again.

        Returns
        -------
        bool
            Whether the fresh decision reaches the same label and the same
            certificates.

        Raises
        ------
        ValueError
            If the generator has no interpretation, or the declared fragment
            holds no negation to decide a contradiction with.
        """
        interpretation = self._interpretation

        if interpretation is None:
            msg = "pairs are verified under an interpretation, and none is declared"

            raise ValueError(msg)

        negation_name = interpretation.language.profile.negation

        if negation_name is None:
            msg = "a contradiction needs a designated negation constant"
            raise ValueError(msg)

        premise = interpretation.read_text(pair.premise.text)
        hypothesis = interpretation.read_text(pair.hypothesis.text)
        label, forward, against = _relation(
            interpretation,
            interpretation.language.apply(
                interpretation.language.constant(negation_name), hypothesis
            ),
            premise,
            hypothesis,
        )

        return (
            label == pair.label
            and forward == pair.entailment
            and against == pair.contradiction
        )

    def verify_typing(self, record: IllTyped) -> bool:
        """Return whether checking an injected form again reproduces its certificate.

        The form is read back from its surface text and checked under the
        declared signature. A certificate whose constraints or language
        differ from what the checker returns is declined, and so is one whose
        form the checker now accepts.

        Parameters
        ----------
        record : IllTyped
            The injected form to check again.

        Raises
        ------
        ValueError
            If the generator has no interpretation.
        """
        interpretation = self._interpretation

        if interpretation is None:
            msg = "forms are verified under an interpretation, and none is declared"
            raise ValueError(msg)

        return interpretation.verify_typing(
            interpretation.read_text(record.text), record.certificate
        )

    def _roots(
        self, table: _Table, lower: int, bound: int
    ) -> dict[int, tuple[tuple[_State, float], ...]]:
        """Return the root items by token count, each with its derivation mass.

        Parameters
        ----------
        table : _Table
            The table of derivation counts.
        lower : int
            The smallest token count to inspect.
        bound : int
            The largest token count to inspect.
        """
        targets = self._assembly.grammar.targets
        roots: dict[int, tuple[tuple[_State, float], ...]] = {}

        for length in range(lower, bound + 1):
            found = tuple(
                (state, table.mass((state, length)))
                for state in table.states[length]
                if any(
                    targets.matching.equivalent(state[0], category)
                    for category in targets.categories
                )
            )

            if found:
                roots[length] = found

        return roots


def _measured(
    interpretation: InterpretationFunction,
    meter: ComplexityMeter,
    corpus: SyntheticCorpus,
    /,
    *,
    quantifiers: frozenset[str],
) -> tuple[tuple[GoldForm, Term, FormComplexity], ...]:
    """Return each of a sample's gold forms, read back and measured.

    Parameters
    ----------
    interpretation : InterpretationFunction
        The interpretation whose store the surface renderings read back into.
    meter : ComplexityMeter
        The meter the read-back forms are measured by.
    corpus : SyntheticCorpus
        The sample whose gold forms to read.
    quantifiers : frozenset[str]
        The binder names the declared fragment admits, which every read-back
        form must quantify within.

    Returns
    -------
    tuple[tuple[GoldForm, Term, FormComplexity], ...]
        One entry per grammatical item, in draw order: the gold form, the term
        its text reads back to, and that term's complexity.

    Raises
    ------
    ValueError
        If the sample holds no grammatical item to pair among.
    FragmentEscapeError
        If a read-back form quantifies outside the declared fragment.
    """
    held = tuple(corpus.forms.values())

    if not held:
        msg = "pairs are drawn among grammatical items, and the sample holds none"

        raise ValueError(msg)

    read = tuple(interpretation.read_text(gold.text) for gold in held)
    measured = tuple(
        (gold, form, meter.measure(form)) for gold, form in zip(held, read, strict=True)
    )

    for gold, _, complexity in measured:
        escaped = complexity.quantifiers - quantifiers

        if escaped:
            raise FragmentEscapeError(gold.text, escaped, quantifiers)

    return measured


def _wanted(
    quotas: Mapping[InferenceLabel, int], filled: Mapping[InferenceLabel, int], /
) -> InferenceLabel | None:
    """Return the first label whose quota is not yet filled.

    Parameters
    ----------
    quotas : Mapping[InferenceLabel, int]
        How many pairs of each label the run is to hold.
    filled : Mapping[InferenceLabel, int]
        How many it holds so far.

    Returns
    -------
    InferenceLabel | None
        The label to search for, or nothing when every quota is filled.
    """
    return next(
        (label for label in RELATION_LABELS if filled.get(label, 0) < quotas[label]),
        None,
    )


def _relation(
    interpretation: InterpretationFunction,
    negated: Term,
    premise: Term,
    hypothesis: Term,
    /,
) -> tuple[
    InferenceLabel | None,
    EntailmentCertificate,
    EntailmentCertificate | None,
]:
    """Decide the relation a premise bears to a hypothesis.

    The hypothesis is decided first. A proof settles the pair as entailment
    and the negated hypothesis is never asked about, so a pair the premise
    entails costs one decision and every other pair costs two.

    Parameters
    ----------
    interpretation : InterpretationFunction
        The interpretation whose oracle decides.
    negated : Term
        The hypothesis under the negation of the declared fragment.
    premise : Term
        The formula assumed.
    hypothesis : Term
        The formula decided.

    Returns
    -------
    tuple[InferenceLabel | None, EntailmentCertificate, EntailmentCertificate | None]
        The relation, the decision on the hypothesis, and the decision on its
        negation when one was taken.
    """
    forward = interpretation.decide(premise, hypothesis)

    if forward.label == "entailment":
        return InferenceLabel.ENTAILS, forward, None

    against = interpretation.decide(premise, negated)
    decided = Decided.from_verdicts(forward.verdict, against.verdict)

    return interpretation.language.profile.label_map.label(decided), forward, against


def _measurements(
    complexity: PairComplexityTarget | None,
    missed: Iterable[tuple[FormComplexity, FormComplexity]],
) -> str:
    """Return what rejected candidates measured on each controlled dimension.

    Parameters
    ----------
    complexity : PairComplexityTarget | None
        The bands the candidates were held to.
    missed : Iterable[tuple[FormComplexity, FormComplexity]]
        The premise and hypothesis measurements of each rejected candidate.
    """
    held = tuple(missed)
    parts: list[str] = []

    if complexity is not None:
        for side, target, position in (
            ("premise", complexity.premise, 0),
            ("hypothesis", complexity.hypothesis, 1),
        ):
            if target is None:
                continue

            for control in target.controls:
                values = Histogram.of(pair[position].value(control) for pair in held)
                parts.append(f"{side} {control} {values!r}")

        if complexity.overlap is not None:
            parts.append(
                f"overlap {Histogram.of(left.overlap(right) for left, right in held)!r}"
            )

    return ", ".join(parts)


def _labels(
    truths: Iterable[tuple[ItemId, GrammaticalityLabel]],
    noise: LabelNoise | None,
    random: Random,
) -> tuple[ObservedLabel, ...]:
    """Return one label record per item, each flipped by its own draw.

    Parameters
    ----------
    truths : Iterable[tuple[ItemId, GrammaticalityLabel]]
        Each item with its true label, in draw order.
    noise : LabelNoise | None
        The rates to flip each true label at, or nothing to flip no label.
    random : Random
        The seeded generator the flips consume, one draw per item when a
        rate is set.
    """
    if noise is None:
        return tuple(
            ObservedLabel.of(item_id, truth, flipped=False) for item_id, truth in truths
        )

    return tuple(
        ObservedLabel.of(item_id, truth, flipped=random.random() < noise.rate(truth))
        for item_id, truth in truths
    )


def _noised(
    pairs: Iterable[InferencePair],
    noise: InferenceNoise | None,
    random: Random,
) -> tuple[NoisedLabel, ...]:
    """Return one label record per pair, each flipped by its own draw.

    Parameters
    ----------
    pairs : Iterable[InferencePair]
        The labeled pairs, in draw order.
    noise : InferenceNoise | None
        The rate to flip at and where the flips land, or nothing to flip no
        relation.
    random : Random
        The seeded generator the flips consume, two draws per pair when a
        rate is set: one for whether the relation is flipped and one for
        where the flip lands.

    Returns
    -------
    tuple[NoisedLabel, ...]
        The records, in draw order, each holding the oracle's relation beside
        the one emitted for it.
    """
    if noise is None:
        return tuple(NoisedLabel.of(pair.label) for pair in pairs)

    records = []

    for pair in pairs:
        flip = random.random()
        where = random.random()
        label = pair.label

        if (
            label is not None
            and noise.confusion.confuses(label)
            and flip < noise.rate(label)
        ):
            records.append(NoisedLabel(label, noise.confusion.lands(label, where)))
        else:
            records.append(NoisedLabel.of(label))

    return tuple(records)


def _located(
    assembly: GrammarAssembly,
    drawn: _Sampled,
    chart: Chart[Term, float],
) -> ChartDerivation[Term, float]:
    """Return the parser's derivation of a drawn string with the drawn structure.

    Parameters
    ----------
    assembly : GrammarAssembly
        The compiled grammar the string was drawn from.
    drawn : _Sampled
        The drawn derivation.
    chart : Chart[Term, float]
        The parse of the drawn string.

    Raises
    ------
    UncertifiedDerivationError
        If the parser's forest holds no derivation with the drawn structure.
    """
    grammar = assembly.grammar
    located = next(
        (
            found
            for root in grammar.goal.roots(chart)
            if (found := _locate(chart, grammar.matching, root, drawn)) is not None
        ),
        None,
    )

    if located is None:
        msg = (
            f"the parser rebuilds no derivation of {' '.join(drawn.forms())!r} "
            f"with the drawn structure"
        )
        raise UncertifiedDerivationError(msg)

    return located


def _item(
    digest: str,
    licensing: Licensing,
    forms: tuple[str, ...],
    index: int,
    seed: int,
    *,
    split: str | None = None,
) -> Item:
    """Build the item for a drawn string, with byte spans into its text.

    Parameters
    ----------
    digest : str
        The digest of the declaration the string was drawn from, whose first
        twelve characters prefix the item identifier.
    licensing : Licensing
        The terms the item is released under.
    forms : tuple[str, ...]
        The surface forms, in order.
    index : int
        The item's position in the sample.
    seed : int
        The seed the sample is drawn with.
    """
    return synthetic_inference_item(
        ItemId(f"{digest[:12]}:{seed}:{index}"),
        " ".join(forms),
        split=split,
        provenance=Provenance(
            SOURCE_METHOD,
            licensing,
            reproducibility=Reproducibility(random_seed=seed),
        ),
    )


def _gold(
    located: ChartDerivation[Term, float], syntax: TermSyntax, store: TermStore
) -> Derivation[str]:
    """Return a located parser derivation as a gold derivation over category text.

    Parameters
    ----------
    located : ChartDerivation[Term, float]
        The parser derivation, rooted at the whole string.
    syntax : TermSyntax
        The syntax every category is written in.
    store : TermStore
        The store the categories belong to.
    """
    nodes: dict[NodeId, DerivationNode[str]] = {}
    identifiers = count()

    def visit(node: ChartDerivation[Term, float]) -> NodeId:
        node_id = next(identifiers)
        children = tuple(visit(child) for child in node.children)
        nodes[node_id] = DerivationNode(
            TokenSpan(*node.key.span),
            syntax.render(store, node.edge.output),
            rule=None if is_axiom(node) else node.edge.rule_name,
            children=children,
        )

        return node_id

    root = visit(located)

    return Derivation(root, nodes)
