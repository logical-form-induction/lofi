"""Measures the eager and lazy normalization policies over packed forests.

Each case parses one sentence of a fixed length under the combinatory rule
set, reads every item's meanings off the forest bottom-up under one policy,
packing the meanings each item holds by interned identity, and then
normalizes every held meaning the way the logical-form quotient does. The
timed work is the reading and the quotient; the parse is timed separately
and the ratio of the two is recorded as the end-to-end multiple.
"""

from itertools import product
from random import Random
from statistics import median
from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from pytest_benchmark.utils import load_timer
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi._native import SpanFree
from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    ParsePolicy,
    Reachability,
    Retain,
    Viterbi,
    parse,
)
from lofi.interpret import (
    NORMALIZATION_POLICIES,
    MeaningReading,
    MeaningSelection,
    StepContext,
    compile_meaning_actions,
)
from lofi.judge import ProofSchema, ProofSearch
from lofi.lexicon import Entry
from lofi.spec import ApplicationSpec, CompositionSpec, RuleSetSpec, TypeRaisingSpec
from lofi.spec.grammar import (
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    NoNormalFormSpec,
    OperatorSpec,
    RootGoalSpec,
    SignatureSpec,
)
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language
from lofi.terms import Normalization, OpName, SExpSyntax, StoreFullError, beta_normal
from lofi.typecheck import Erasure

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping

    from pytest_benchmark.fixture import BenchmarkFixture

    from lofi.deduce import Chart, Goal, ItemKey, PackedForest
    from lofi.interpret import NormalizationPolicy
    from lofi.judge import Rule
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly
    from lofi.terms import Calculus, Term, TermStore

type Category = Term
type Formula = Term

LENGTHS: Final = (10, 20, 30, 40)
PARSE_ROUNDS: Final = 5
READ_ROUNDS: Final = 3
#: The node cap of the region one case interns its meanings in.
CASE_NODES: Final = 1_000_000
NORMALIZATION: Final = Normalization(
    fuel=40,
    order="leftmost-outermost",
    eta="off",
    associative_commutative=(),
)

NOUNS: Final = ("dog", "cat", "boy")
DETERMINERS: Final = ("the", "some", "every")
VERBS: Final = ("sees", "likes")
ADJECTIVES: Final = ("big", "old")
PREPOSITIONS: Final = ("near", "with")
NAMES: Final = ("Bo", "Mary")
#: How many adjectives one noun takes at most.
ADJECTIVES_PER_NOUN: Final = 2
#: How deep prepositional phrases nest at most.
PHRASE_DEPTH: Final = 3
#: How many clauses a sentence embeds at most under an attitude verb.
EMBEDDINGS: Final = 2

# form, category and meaning in the declared language's surface syntax
LEXICON: Final = (
    ("Bo", "NP", "bo"),
    ("Mary", "NP", "mary"),
    (
        "likes",
        "(S[dcl]\\NP)/NP",
        "\\o. \\s. ∃e. like(e) & (agent(s)(e) & theme(o)(e))",
    ),
    (
        "sees",
        "(S[dcl]\\NP)/NP",
        "\\o. \\s. ∃e. see(e) & (agent(s)(e) & theme(o)(e))",
    ),
    (
        "thinks",
        "(S[dcl]\\NP)/S[dcl]",
        "\\p. \\s. ∃e. think(e) & (agent(s)(e) & content(p)(e))",
    ),
    ("surely", "S/S", "\\p. surely(p)"),
    ("quickly", "(S\\NP)\\(S\\NP)", "\\p. \\x. quickly(p(x))"),
    ("often", "(S\\NP)/(S\\NP)", "\\p. \\x. often(p(x))"),
    ("every", "NP/N", "\\p. every(p)"),
    ("some", "NP/N", "\\p. some(p)"),
    ("the", "NP/N", "\\p. the(p)"),
    ("dog", "N", "dog"),
    ("cat", "N", "cat"),
    ("boy", "N", "boy"),
    ("big", "N/N", "\\p. \\x. big(x) & p(x)"),
    ("old", "N/N", "\\p. \\x. old(x) & p(x)"),
    ("near", "(NP\\NP)/NP", "\\o. \\s. near(s)(o)"),
    ("with", "(NP\\NP)/NP", "\\o. \\s. with(s)(o)"),
)


def _grammar() -> GrammarDecl:
    """Declare the combinatory grammar over the lexicon's categories."""
    return GrammarDecl(
        name="policy-ccg",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "N", "conj"),
            operators=(
                OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
                OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
            ),
            conjunctions=ConjunctionsSpec(
                markers=("conj",),
                pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
            ),
        ),
        rules=RuleSetSpec(
            name="combinatory",
            rules=(
                ApplicationSpec(view="/"),
                ApplicationSpec(view="\\"),
                CompositionSpec(direction="unrestricted", order=1, view="/"),
                CompositionSpec(direction="unrestricted", order=1, view="\\"),
                TypeRaisingSpec(view="/", targets=("S",), arguments=None),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category)
                for form, category, _ in LEXICON
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        normal_form=NoNormalFormSpec(),
    )


class _Unweighted:
    """Scores every axiom and edge at zero."""

    __slots__ = ()
    lowering = SpanFree()

    def axiom_weight(self, position: int, entry: Entry[Category], /) -> float:
        del position, entry

        return 0.0

    def edge_weight(
        self,
        rule: Rule[Category],
        inputs: tuple[Category, ...],
        output: Category,
        span: tuple[int, int],
        split: int | None,
        /,
    ) -> float:
        del rule, inputs, output, span, split

        return 0.0


class _LambdaCalculus:
    """The proof calculus over one region of the language's formula store."""

    __slots__ = ("_calculus", "_sort", "_store")

    def __init__(self, store: TermStore, calculus: Calculus) -> None:
        self._store = store
        self._calculus = calculus
        (site,) = store.signature.by_name(calculus.lam).binding
        self._sort = site.sort

    @property
    def store(self) -> TermStore:
        return self._store

    @property
    def operators(self) -> Calculus:
        return self._calculus

    def hypothesis(self, index: int, /) -> Formula:
        return self._store.fvar(self._sort, f"p{index}")

    def free_variable(self, name: str, /) -> Formula:
        return self._store.fvar(self._sort, name)

    def bound_variable(self, index: int, /) -> Formula:
        return self._store.bvar(self._sort, index)

    def application(self, function: Formula, argument: Formula, /) -> Formula:
        operator = self._store.signature.by_name(self._calculus.app)

        return self._store.op(operator, function, argument)

    def abstraction(self, variable: str, body: Formula, /) -> Formula:
        return self._store.bind(self._calculus.lam, variable, body)


class _LinkedErasure:
    """Erase grammar categories through the linked language signature."""

    __slots__ = ("_erasure", "_source", "_source_syntax", "_target_syntax")

    def __init__(self, source: TermStore, erasure: Erasure, /) -> None:
        self._source = source
        self._erasure = erasure
        self._source_syntax = SExpSyntax(source.signature)
        self._target_syntax = SExpSyntax(erasure.store.signature)

    @property
    def store(self) -> TermStore:
        """Return the semantic store used for proof search."""
        return self._erasure.store

    @property
    def category_sort(self) -> str:
        """Return the source category sort."""
        return self._erasure.category_sort

    @property
    def atoms(self) -> Mapping[OpName, Term]:
        """Return core-atom images."""
        return self._erasure.atoms

    @property
    def constructors(self) -> Mapping[OpName, Term]:
        """Return constructor type-term images."""
        return self._erasure.constructors

    @property
    def function_type(self) -> OpName:
        """Return the target function-type constructor."""
        return self._erasure.function_type

    @property
    def target_sort(self) -> str:
        """Return the target sort of every image."""
        return self._erasure.target_sort

    def image(self, constructor: OpName, arguments: tuple[Term, ...], /) -> Term:
        """Instantiate one constructor image over erased arguments."""
        return self._erasure.image(constructor, arguments)

    def __call__(self, category: Term, /) -> Term:
        """Project one linked category into the semantic store and erase it."""
        if self.store.contains(category):
            return self._erasure(category)

        text = self._source_syntax.render(self._source, category)
        projected = self._target_syntax.parse(self.store, text)
        return self._erasure(projected)


class Setting(NamedTuple):
    """The grammar, language and erasure every case reads under.

    Attributes
    ----------
    grammar : GrammarAssembly
        The lowered grammar the sentences are parsed under.
    language : LanguageAssembly
        The lowered language the meanings are interned in.
    rules : Mapping[str, Rule[Category]]
        The rules keyed by the names the forest's edges use.
    entries : Mapping[str, tuple[Entry[Category, Formula], ...]]
        The meaning-bearing entries keyed by form.
    erasure : _LinkedErasure
        The root-store erasure declaration copied into each case region.
    """

    grammar: GrammarAssembly
    language: LanguageAssembly
    rules: Mapping[str, Rule[Category]]
    entries: Mapping[str, tuple[Entry[Category, Formula], ...]]
    erasure: _LinkedErasure

    def reading(
        self, store: TermStore, policy: NormalizationPolicy
    ) -> MeaningReading[Category]:
        """Build the meaning reading over one store region under one policy."""
        calculus = _LambdaCalculus(store, self.language.calculus)
        erasure = _LinkedErasure(
            self.grammar.store,
            Erasure(
                store,
                self.erasure.atoms,
                self.erasure.constructors,
                category_sort=self.erasure.category_sort,
                function_type=self.erasure.function_type,
            ),
        )
        actions = compile_meaning_actions(
            self.grammar.grammar.rules,
            erasure=erasure,
            calculus=calculus,
            normalization=NORMALIZATION,
            search=ProofSearch(
                contraction=True, weakening=False, max_uses=1, max_inhabitants=64
            ),
            selection=MeaningSelection("unique"),
        )

        if actions.uncovered or actions.ambiguous:
            pytest.fail(
                f"the rule set has no unique action for every rule: {actions!r}"
            )

        return MeaningReading[Category](
            actions,
            normalization=NORMALIZATION,
            policy=policy,
        )

    def chart(self, tokens: tuple[str, ...]) -> PackedForest[Category, float]:
        """Parse one sentence under the grammar's rules and lexicon."""
        grammar = self.grammar.grammar

        return parse(
            tokens,
            rules=grammar.rules,
            source=self.grammar.lexicon,
            semiring=Viterbi(),
            scorer=_Unweighted(),
            goal=grammar.goal,
            unary_depth=grammar.budget.unary_rounds,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )


def build_setting() -> Setting:
    """Lower the grammar and language, and declare their erasure."""
    declaration = _grammar()
    standard = registered_language("neo-davidsonian")
    names = (
        "agent",
        "big",
        "bo",
        "boy",
        "cat",
        "content",
        "dog",
        "every",
        "like",
        "mary",
        "near",
        "often",
        "old",
        "quickly",
        "see",
        "some",
        "surely",
        "the",
        "theme",
        "think",
        "with",
    )
    standing = {constant.name for constant in standard.constants}
    language = LanguageIso().extend(
        standard.with_(
            constants=(
                *standard.constants,
                *(
                    ConstantSpec(name=name, type="t")
                    for name in names
                    if name not in standing
                ),
            )
        ),
        declaration.signature.extend,
    )
    grammar = GrammarIso().forward_language(declaration, language)
    types = language.store
    entity = language.read_type("e")
    truth = language.read_type("t")
    predicate = language.read_type("e -> t")
    erasure = _LinkedErasure(
        grammar.store,
        Erasure(
            types,
            {"S": truth, "NP": entity, "N": predicate, "conj": truth},
            {
                name: types.op(
                    types.signature.by_name(language.function_type),
                    types.fvar(language.type_sort, "a1"),
                    types.fvar(language.type_sort, "a0"),
                )
                for name in ("/", "\\")
            },
            category_sort=grammar.language.category_sort,
            function_type=language.function_type,
        ),
    )
    schemata = {
        rule.name: rule
        for rule in grammar.grammar.rules
        if isinstance(rule, ProofSchema)
    }

    if len(schemata) != len(grammar.grammar.rules):
        pytest.fail("a combinatory rule declares no proof schema")

    entries: dict[str, list[Entry[Category, Formula]]] = {}

    for form, category, meaning in LEXICON:
        entry = Entry(
            form,
            grammar.syntax.parse(grammar.store, category),
            language.syntax.desugar(language.syntax.parse(meaning)).term,
        )
        entries.setdefault(form, []).append(entry)

    return Setting(
        grammar,
        language,
        {rule.name: rule for rule in grammar.grammar.rules},
        {form: tuple(held) for form, held in entries.items()},
        erasure,
    )


class _Slot:
    """One token of a sentence under construction, with what modifies it."""

    __slots__ = ("adjectives", "depth", "phrase", "word")

    def __init__(self, word: str, depth: int) -> None:
        self.word = word
        self.depth = depth
        self.adjectives = 0
        self.phrase = False


@pytest.fixture(scope="module")
def setting() -> Setting:
    """Lower the grammar and the language once per module."""
    return build_setting()


def sentence(length: int, seed: int) -> tuple[str, ...]:
    """Draw a sentence of exactly the given length by extending a clause.

    A transitive clause is extended one modifier at a time until it has the
    length asked for. A noun takes a bounded number of adjectives and at most
    one prepositional phrase, prepositional phrases nest to a bounded depth,
    the clause is embedded under a subject and an attitude verb a bounded
    number of times, and the whole sentence takes at most one preverbal
    adverb, one postverbal adverb and one sentential adverb.

    Parameters
    ----------
    length : int
        How many tokens the sentence has. At least five.
    seed : int
        The seed of the draw. Equal seeds draw equal sentences.
    """
    random = Random(seed)
    slots = [
        _Slot(random.choice(DETERMINERS), 0),
        _Slot(random.choice(NOUNS), 0),
        _Slot(random.choice(VERBS), 0),
        _Slot(random.choice(DETERMINERS), 0),
        _Slot(random.choice(NOUNS), 0),
    ]

    while len(slots) < length:
        room = length - len(slots)
        used = {slot.word for slot in slots}
        nouns = [at for at, slot in enumerate(slots) if slot.word in NOUNS]
        verbs = [at for at, slot in enumerate(slots) if slot.word in VERBS]
        clauses = [
            0,
            *(at + 1 for at, slot in enumerate(slots) if slot.word == "thinks"),
        ]
        options = [
            ("adjective", at)
            for at in nouns
            if slots[at].adjectives < ADJECTIVES_PER_NOUN
        ]

        if "often" not in used:
            options.extend(("often", at) for at in verbs)

        if "quickly" not in used:
            options.extend(("quickly", at) for at in verbs)

        if "surely" not in used:
            options.extend(("surely", at) for at in clauses)

        if room >= 3:
            options.extend(
                ("phrase", at)
                for at in nouns
                for _ in range(2)
                if slots[at].depth < PHRASE_DEPTH and not slots[at].phrase
            )

        if room >= 2 and len(clauses) <= EMBEDDINGS:
            options.extend(("clause", at) for at in clauses)

        kind, at = random.choice(options)

        if kind == "adjective":
            slots[at].adjectives += 1
            slots.insert(at, _Slot(random.choice(ADJECTIVES), slots[at].depth))
        elif kind == "often":
            slots.insert(at, _Slot("often", 0))
        elif kind == "quickly":
            end = at + 1

            while end < len(slots) and slots[end].word not in VERBS:
                end += 1

            slots.insert(end, _Slot("quickly", 0))
        elif kind == "surely":
            slots.insert(at, _Slot("surely", 0))
        elif kind == "phrase":
            slots[at].phrase = True
            depth = slots[at].depth + 1
            slots[at + 1 : at + 1] = [
                _Slot(random.choice(PREPOSITIONS), depth),
                _Slot(random.choice(DETERMINERS), depth),
                _Slot(random.choice(NOUNS), depth),
            ]
        else:
            slots[at:at] = [_Slot(random.choice(NAMES), 0), _Slot("thinks", 0)]

    return tuple(slot.word for slot in slots)


class Fold(NamedTuple):
    """The meanings every item of one forest holds under one policy.

    Attributes
    ----------
    held : Mapping[ItemKey[Category], tuple[Formula, ...]]
        Each item's distinct meanings, in the order they were reached.
    edges : int
        How many forest edges were read.
    applications : int
        How many rule applications were run, one per edge and combination
        of child meanings.
    completed : bool
        Whether every item was read, or the store's node cap stopped the
        reading first.
    """

    held: Mapping[ItemKey[Category], tuple[Formula, ...]]
    edges: int
    applications: int
    completed: bool

    @property
    def materialized(self) -> int:
        """Return how many distinct meanings the whole forest holds."""
        return sum(len(meanings) for meanings in self.held.values())


def fold(
    chart: Chart[Category, float],
    reading: MeaningReading[Category],
    setting: Setting,
    tokens: tuple[str, ...],
) -> Fold:
    """Read every item's meanings off a forest, smallest spans first.

    An item holds the distinct meanings its edges reach, packed by interned
    identity, so a meaning two edges reach is held once and read once by
    every parent. The reading stops, and reports itself incomplete, when
    the reading's store refuses another node.
    """
    lexicon = tuple(setting.entries.get(token, ()) for token in tokens)
    held: dict[ItemKey[Category], tuple[Formula, ...]] = {}
    edges = 0
    applications = 0

    try:
        for span in chart.index.regions_by_size():
            for key in chart.items_at(span):
                reached: dict[Formula, None] = {}

                for edge in chart.edges(key):
                    edges += 1

                    if edge.rule_name.startswith(LEXICAL_TAG_PREFIX):
                        form = edge.rule_name.removeprefix(LEXICAL_TAG_PREFIX)

                        for entry in lexicon[span[0]]:
                            if entry.form == form and entry.category == edge.output:
                                meaning = reading.axiom(entry)

                                if meaning is not None:
                                    reached.setdefault(meaning)

                        continue

                    context = StepContext(
                        setting.rules[edge.rule_name],
                        edge.output,
                        edge.weight,
                        edge.order,
                        tuple(chart.term(child) for child in edge.children),
                    )
                    below = tuple(held.get(child, ()) for child in edge.children)

                    for children in product(*below):
                        applications += 1
                        meaning = reading.step(context, children)

                        if meaning is not None:
                            reached.setdefault(meaning)

                held[key] = tuple(reached)

    except StoreFullError:
        return Fold(held, edges, applications, completed=False)

    return Fold(held, edges, applications, completed=True)


class Quotient(NamedTuple):
    """What normalizing every held meaning produced.

    Attributes
    ----------
    normalized : int
        How many held meanings were normalized.
    already_normal : int
        How many of those were their own normal form.
    completed : bool
        Whether every held meaning was normalized, or the store's node cap
        stopped the normalization first.
    """

    normalized: int
    already_normal: int
    completed: bool


def normalize_all(store: TermStore, calculus: Calculus, held: Fold) -> Quotient:
    """Normalize every held meaning as the quotient does."""
    normalized = 0
    already_normal = 0

    try:
        for meanings in held.held.values():
            for meaning in meanings:
                already_normal += (
                    beta_normal(store, meaning, calculus, fuel=40) is meaning
                )
                normalized += 1

    except StoreFullError:
        return Quotient(normalized, already_normal, completed=False)

    return Quotient(normalized, already_normal, completed=True)


def _timed[R](timer: Callable[[], float], call: Callable[[], R], rounds: int) -> float:
    """Return the median seconds one call takes over the given rounds."""
    seconds: list[float] = []

    for _ in range(rounds):
        start = timer()
        call()
        seconds.append(timer() - start)

    return median(seconds)


def _goal_meanings(
    chart: PackedForest[Category, float],
    held: Fold,
    goal: Goal[Category],
    store: TermStore,
    calculus: Calculus,
) -> tuple[Formula, ...]:
    """Return the normal forms the goal items over the whole sentence hold."""
    return tuple(
        dict.fromkeys(
            beta_normal(store, meaning, calculus, fuel=40)
            for key in goal.roots(chart)
            for meaning in held.held[key]
        )
    )


class Outcome(NamedTuple):
    """What one timed reading of a forest produced.

    Attributes
    ----------
    held : Fold
        The meanings read, complete or not.
    quotient : Quotient
        What normalizing the held meanings produced.
    completed : bool
        Whether both the reading and the quotient finished under the node cap.
    nodes : int
        How many nodes the case's store region held at the end.
    fold_seconds : float
        How long the reading alone took.
    """

    held: Fold
    quotient: Quotient
    completed: bool
    nodes: int
    fold_seconds: float


def _read_case(
    setting: Setting,
    chart: PackedForest[Category, float],
    tokens: tuple[str, ...],
    policy: NormalizationPolicy,
    timer: Callable[[], float],
) -> Outcome:
    """Read and quotient one forest under one policy in a fresh store region."""
    root = setting.language.store

    with root.region("case", max_nodes=CASE_NODES) as store:
        reading = setting.reading(store, policy)
        start = timer()
        held = fold(chart, reading, setting, tokens)
        fold_seconds = timer() - start
        quotient = (
            normalize_all(store, setting.language.calculus, held)
            if held.completed
            else Quotient(0, 0, completed=False)
        )

        return Outcome(
            held,
            quotient,
            held.completed and quotient.completed,
            store.local_nodes,
            fold_seconds,
        )

    pytest.fail("the term-store region suppressed a reading failure")


def _collect_case(
    outcomes: list[Outcome],
    setting: Setting,
    chart: PackedForest[Category, float],
    tokens: tuple[str, ...],
    policy: NormalizationPolicy,
    timer: Callable[[], float],
) -> None:
    outcomes.append(_read_case(setting, chart, tokens, policy, timer))


@pytest.mark.parametrize("policy", NORMALIZATION_POLICIES)
@pytest.mark.parametrize(
    "length",
    [
        pytest.param(length, marks=() if length == LENGTHS[0] else pytest.mark.slow)
        for length in LENGTHS
    ],
)
def test_reading_meanings_reports_consistent_complete_or_bounded_outcomes(
    pytestconfig: pytest.Config,
    setting: Setting,
    length: int,
    policy: NormalizationPolicy,
) -> None:
    timer = load_timer(pytestconfig.getoption("benchmark_timer")).target
    tokens = sentence(length, length)
    chart = setting.chart(tokens)
    outcomes: list[Outcome] = []

    for _ in range(READ_ROUNDS):
        _collect_case(outcomes, setting, chart, tokens, policy, timer)

    outcome = outcomes[-1]
    held = outcome.held

    assert len({item.completed for item in outcomes}) == 1
    assert len({item.held.materialized for item in outcomes}) == 1

    if policy == "eager":
        assert outcome.completed

    if not outcome.completed:
        assert outcome.nodes == CASE_NODES
        return

    assert len(held.held) == chart.num_items
    assert held.materialized >= chart.num_items
    assert outcome.quotient.normalized == held.materialized

    if policy == "eager":
        assert outcome.quotient.already_normal == held.materialized
    else:
        assert outcome.quotient.already_normal < held.materialized


@pytest.mark.parametrize("policy", NORMALIZATION_POLICIES)
@pytest.mark.parametrize("length", LENGTHS)
def test_reading_meanings_off_a_forest_under_a_policy(
    benchmark: BenchmarkFixture,
    pytestconfig: pytest.Config,
    setting: Setting,
    length: int,
    policy: NormalizationPolicy,
) -> None:
    timer = load_timer(pytestconfig.getoption("benchmark_timer")).target
    tokens = sentence(length, length)
    chart = setting.chart(tokens)
    parse_seconds = _timed(timer, lambda: setting.chart(tokens), PARSE_ROUNDS)
    outcomes: list[Outcome] = []

    benchmark.pedantic(
        _collect_case,
        args=(outcomes, setting, chart, tokens, policy, timer),
        rounds=READ_ROUNDS,
        iterations=1,
    )
    measured = benchmark.stats

    if measured is None:
        pytest.fail("the reading ran but pytest-benchmark timed nothing")

    outcome = outcomes[-1]
    held = outcome.held
    read_seconds = measured.stats.median
    fold_seconds = median(outcome.fold_seconds for outcome in outcomes)
    benchmark.extra_info["policy"] = policy
    benchmark.extra_info["length"] = length
    benchmark.extra_info["sentence"] = " ".join(tokens)
    benchmark.extra_info["completed"] = outcome.completed
    benchmark.extra_info["items"] = chart.num_items
    benchmark.extra_info["items_read"] = len(held.held)
    benchmark.extra_info["edges"] = held.edges
    benchmark.extra_info["applications"] = held.applications
    benchmark.extra_info["materialized"] = held.materialized
    benchmark.extra_info["quotient_normalizations"] = outcome.quotient.normalized
    benchmark.extra_info["already_normal"] = outcome.quotient.already_normal
    benchmark.extra_info["nodes_interned"] = outcome.nodes
    benchmark.extra_info["node_cap"] = CASE_NODES
    benchmark.extra_info["parse_seconds"] = parse_seconds
    benchmark.extra_info["fold_seconds"] = fold_seconds
    benchmark.extra_info["read_seconds"] = read_seconds
    benchmark.extra_info["us_per_application"] = fold_seconds / held.applications * 1e6
    benchmark.extra_info["us_per_edge"] = read_seconds / held.edges * 1e6
    benchmark.extra_info["multiple"] = (parse_seconds + read_seconds) / parse_seconds


@pytest.mark.parametrize(
    "length",
    [
        pytest.param(length, marks=() if length == LENGTHS[0] else pytest.mark.slow)
        for length in LENGTHS
    ],
)
def test_the_two_policies_read_pointer_identical_goal_meanings(
    setting: Setting, length: int
) -> None:
    tokens = sentence(length, length)
    chart = setting.chart(tokens)
    calculus = setting.language.calculus

    with setting.language.store.region("both", max_nodes=CASE_NODES) as store:
        eager = fold(chart, setting.reading(store, "eager"), setting, tokens)
        lazy = fold(chart, setting.reading(store, "lazy"), setting, tokens)

        assert eager.completed

        if not lazy.completed:
            pytest.skip(
                f"the lazy reading at {length} tokens exceeds the {CASE_NODES} "
                f"node cap after {lazy.applications} applications"
            )

        goal = setting.grammar.grammar.goal
        eager_goal = _goal_meanings(chart, eager, goal, store, calculus)
        lazy_goal = _goal_meanings(chart, lazy, goal, store, calculus)

        assert eager_goal
        assert len(eager_goal) == len(lazy_goal)
        assert all(
            held is other for held, other in zip(eager_goal, lazy_goal, strict=True)
        )
        assert lazy.materialized > eager.materialized
