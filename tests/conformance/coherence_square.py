"""Conformance laws for category and meaning typing over shipped derivations."""

from typing import TYPE_CHECKING, Final, NamedTuple

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.conformance.lf._naive_check import (
    Apply,
    AtomicType,
    Bound,
    Constant,
    ConstantDeclaration,
    FunctionType,
    GroundType,
    IllTypedError,
    Lambda,
    SignatureDeclaration,
    TermDeclaration,
    type_of,
)

from lofi._native import SpanFree
from lofi.deduce import Derivation, ParsePolicy, Reachability, Retain, Viterbi, parse
from lofi.interpret import (
    CategoryReading,
    LexicalTable,
    MeaningReading,
    MeaningSelection,
    compile_meaning_actions,
    read,
)
from lofi.judge import ProofSearch
from lofi.spec.language import LanguageIso
from lofi.terms import Normalization, OpName, SExpSyntax, Term, substrate_reaches
from lofi.typecheck import (
    CoherenceFixture,
    CoherenceItem,
    Erasure,
    LexicalSignature,
    Scheme,
    coherence_square,
)

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping

    from lofi.deduce import Cell, Chart, Edge, ItemKey, PackedForest
    from lofi.judge import ProofSchema, Rule
    from lofi.lexicon import Entry
    from lofi.spec.grammar import GrammarAssembly
    from lofi.spec.language import LanguageAssembly, LanguageDecl
    from lofi.terms import Calculus, Path, TermStore
    from lofi.typecheck import CoherenceReport, TypingResult

__all__ = ["CoherenceSquareConformance", "CoherenceSquareSuite", "SentenceBatch"]

type Category = Term
type Formula = Term
type SemanticType = Term
type ChartDerivation = Derivation[Category, float]
type Report = CoherenceReport[Category]

_SEARCH_FUEL: Final = 40
_SEARCH_USES: Final = 1
_MAX_TOKENS: Final = 4
_EXAMPLES: Final = 80
_ENTITY: Final = AtomicType("e")
_TRUTH: Final = AtomicType("t")
_MODIFIER: Final = FunctionType(_TRUTH, _TRUTH)
_BINARY_TRUTH: Final = FunctionType(_TRUTH, FunctionType(_TRUTH, _TRUTH))
_QUANTIFIER: Final = FunctionType(FunctionType(_ENTITY, _TRUTH), _TRUTH)


class _ConformanceErasure:
    """Erase categories after crossing the fixture's isomorphic stores."""

    __slots__ = ("_erasure", "_source", "_source_syntax", "_target_syntax")

    def __init__(
        self,
        source: TermStore,
        erasure: Erasure,
        /,
    ) -> None:
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


class SentenceBatch(NamedTuple):
    """Sentences to parse, with the counts the square reports over them.

    Attributes
    ----------
    sentences : tuple[tuple[str, ...], ...]
        Token sequences in input order.
    items : int
        Total chart items.
    instances : int
        Total derivations checked.
    strict_items : int
        Total items with a strict instance.
    """

    sentences: tuple[tuple[str, ...], ...]
    items: int
    instances: int
    strict_items: int


class CoherenceSquareConformance(NamedTuple):
    """Inputs shared by every binding of the square to shipped derivations.

    Attributes
    ----------
    grammar : GrammarAssembly
        Lowered grammar used to build charts.
    language : LanguageAssembly
        Lowered language used to type meanings.
    entries : tuple[Entry[Category, Formula], ...]
        Meaning-bearing lexicon for the grammar.
    proof_schemata : Mapping[str, ProofSchema]
        Structural proof schemata by rule name.
    schemes : Mapping[str, SemanticType]
        Ground constant types by name.
    atoms : Mapping[str, SemanticType]
        Semantic types of primitive categories by name.
    constructors : Mapping[str, SemanticType]
        Type-term images by category constructor.
    corpus : SentenceBatch
        Corpus and expected aggregate counts.
    three_word : SentenceBatch
        Three-word case containing strict instances.
    malformed : tuple[tuple[LanguageDecl, type[Exception]], ...]
        Invalid declarations paired with their expected exceptions.
    """

    grammar: GrammarAssembly
    language: LanguageAssembly
    entries: tuple[Entry[Category, Formula], ...]
    proof_schemata: Mapping[str, ProofSchema]
    schemes: Mapping[str, SemanticType]
    atoms: Mapping[str, SemanticType]
    constructors: Mapping[str, SemanticType]
    corpus: SentenceBatch
    three_word: SentenceBatch
    malformed: tuple[tuple[LanguageDecl, type[Exception]], ...]


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
    """The proof calculus over a language's formula store."""

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


class _Square:
    """One binding of the square: the readings, the checker and the oracles."""

    def __init__(self, conformance: CoherenceSquareConformance) -> None:
        grammar = conformance.grammar
        language = conformance.language
        self.conformance = conformance
        self.rules = {rule.name: rule for rule in grammar.grammar.rules}
        self.by_form: dict[str, list[Entry[Category, Formula]]] = {}

        for entry in conformance.entries:
            self.by_form.setdefault(entry.form, []).append(entry)

        erasure = Erasure(
            language.store,
            conformance.atoms,
            conformance.constructors,
            category_sort=grammar.language.category_sort,
            function_type=language.function_type,
        )
        self.erasure = _ConformanceErasure(grammar.store, erasure)
        self.calculus = _LambdaCalculus(language.store, language.calculus)
        self.reading = MeaningReading[Category](
            compile_meaning_actions(
                grammar.grammar.rules,
                erasure=self.erasure,
                calculus=self.calculus,
                normalization=Normalization(
                    fuel=_SEARCH_FUEL,
                    order="leftmost-outermost",
                    eta="off",
                    associative_commutative=(),
                ),
                search=ProofSearch(
                    contraction=True,
                    weakening=False,
                    max_uses=_SEARCH_USES,
                    max_inhabitants=64,
                ),
                selection=MeaningSelection("unique"),
            ),
            normalization=Normalization(
                fuel=40,
                order="leftmost-outermost",
                eta="off",
                associative_commutative=(),
            ),
            policy="eager",
        )
        self.categories = CategoryReading[Category, Formula]()
        self.signature = language.lexicon
        self.naive = SignatureDeclaration(
            (
                *(
                    ConstantDeclaration(name, _naive_type(language.store, ground))
                    for name, ground in conformance.schemes.items()
                ),
                ConstantDeclaration("not", _MODIFIER),
                *(
                    ConstantDeclaration(name, _BINARY_TRUTH)
                    for name in ("and", "or", "if", "iff")
                ),
                *(
                    ConstantDeclaration(name, _QUANTIFIER)
                    for name in ("forall", "exists")
                ),
            )
        )
        self.ground = language.read_type("e")

    def chart(self, tokens: tuple[str, ...]) -> PackedForest[Category, float]:
        """Parse one sentence under the grammar's rules and lexicon.

        Parameters
        ----------
        tokens : tuple[str, ...]
        """
        grammar = self.conformance.grammar.grammar

        return parse(
            tokens,
            rules=grammar.rules,
            source=self.conformance.grammar.lexicon,
            semiring=Viterbi(),
            scorer=_Unweighted(),
            goal=grammar.goal,
            unary_depth=grammar.budget.unary_rounds,
            policy=ParsePolicy(Retain.FOREST, Reachability.ALL),
        )

    def derivations(
        self, chart: Chart[Category, float]
    ) -> Iterator[tuple[ItemKey[Category], ChartDerivation]]:
        """Yield every derivation of every item, smallest spans first.

        Parameters
        ----------
        chart : Chart[Category, float]
        """
        for span in chart.index.regions_by_size():
            for key in chart.items_at(span):
                for tree in chart.derivations(key):
                    yield key, tree

    def table(self, tokens: tuple[str, ...]) -> LexicalTable[Category, Formula]:
        """Return the grammar's rules and each token's entries.

        Parameters
        ----------
        tokens : tuple[str, ...]
        """
        return LexicalTable(
            self.rules,
            tuple(tuple(self.by_form.get(token, ())) for token in tokens),
        )

    def items(
        self, chart: Chart[Category, float], tokens: tuple[str, ...]
    ) -> tuple[CoherenceItem[Category], ...]:
        """Read every chart item's packed category and its derivations' meanings.

        Parameters
        ----------
        chart : Chart[Category, float]
        tokens : tuple[str, ...]
        """
        found: dict[ItemKey[Category], list[Formula]] = {}
        table = self.table(tokens)

        for key, derivation in self.derivations(chart):
            meanings = found.setdefault(key, [])
            meanings.extend(
                meaning
                for meaning in read(derivation, self.reading, table)
                if meaning is not None
            )

        return tuple(
            CoherenceItem(key.span, chart.term(key), tuple(meanings), key.nf_tag)
            for key, meanings in found.items()
        )

    def category(
        self, derivation: ChartDerivation, tokens: tuple[str, ...]
    ) -> Category:
        """Read a derivation's exact category.

        Parameters
        ----------
        derivation : ChartDerivation
        tokens : tuple[str, ...]
        """
        categories = set(read(derivation, self.categories, self.table(tokens)))

        (category,) = categories if len(categories) == 1 else (None,)

        if category is None:
            msg = f"the category reading rejected {derivation!r}"
            raise ValueError(msg)

        return category

    def report(self, chart: Chart[Category, float], tokens: tuple[str, ...]) -> Report:
        """Run the square over one sentence's chart.

        Parameters
        ----------
        chart : Chart[Category, float]
        tokens : tuple[str, ...]
        """
        language = self.conformance.language

        return coherence_square(
            CoherenceFixture(
                self.items(chart, tokens),
                language.checker,
                language.lexicon,
                language.rules,
                self.erasure,
            )
        )

    def native(self, formula: Formula) -> TypingResult:
        """Type a meaning natively and default its quantified atoms.

        Parameters
        ----------
        formula : Formula
        """
        language = self.conformance.language
        result = language.checker.check(formula, language.lexicon, language.rules)

        return result.defaulted(language.store, self.ground)

    def naive_type(self, formula: Formula, result: TypingResult) -> str:
        """Return the naive checker's rendering of a meaning's type.

        Parameters
        ----------
        formula : Formula
        result : TypingResult
        """
        return _render(type_of(self.naive, self.declaration(formula, result)))

    def declaration(
        self, formula: Formula, result: TypingResult, path: Path = ()
    ) -> TermDeclaration:
        """Return the naive declaration of a meaning under its native typing.

        Parameters
        ----------
        formula : Formula
        result : TypingResult
        path : Path
        """
        store = self.conformance.language.store
        name = formula.op.name

        if formula.op.payload_sort == "index":
            return Bound(_payload(formula))

        if formula.op.payload_sort == "constant":
            return Constant(store.payload_name(formula))

        children = tuple(
            self.declaration(child, result, (*path, at))
            for at, child in enumerate(formula.args)
        )

        if name == self.conformance.language.calculus.lam and len(children) == 1:
            return Lambda(
                _naive_type(
                    self.conformance.language.store,
                    self.binder(formula, result, path),
                ),
                children[0],
            )

        if name == self.conformance.language.calculus.app and len(children) == 2:
            return Apply(children[0], children[1])

        msg = f"the naive checker declares no operator {name!r}"
        raise ValueError(msg)

    def binder(
        self, formula: Formula, result: TypingResult, path: Path
    ) -> SemanticType:
        """Return the ground type of the variable a binder node binds.

        Parameters
        ----------
        formula : Formula
        result : TypingResult
        path : Path
        """
        if formula.op.name == self.conformance.language.calculus.lam:
            return self.ground_type(result.assignment[path].args[0])

        found = _bound_occurrence(formula.args[0], (*path, 0), 0)

        if found is None:
            return self.ground

        return self.ground_type(result.assignment[found])

    def ground_type(self, held: SemanticType) -> SemanticType:
        """Return a type with every remaining metavariable read as the ground type.

        Parameters
        ----------
        held : SemanticType
        """
        store = self.conformance.language.store

        if held.op.payload_sort == "free":
            return self.ground

        if not held.args:
            return held

        return store.op(held.op, *(self.ground_type(child) for child in held.args))


def _payload(formula: Formula) -> int:
    """Return a leaf's payload or refuse a leaf without one.

    Parameters
    ----------
    formula : Formula
    """
    payload = formula.payload

    if payload is None:
        msg = f"{formula!r} has no payload"
        raise ValueError(msg)

    return payload


def _bound_occurrence(formula: Formula, path: Path, depth: int) -> Path | None:
    """Return the path of an occurrence bound at the given depth, if any.

    Parameters
    ----------
    formula : Formula
    path : Path
    depth : int
    """
    if formula.op.payload_sort == "index":
        return path if formula.payload == depth else None

    for at, child in enumerate(formula.args):
        shift = formula.op.shift(at, formula.op.sort)
        found = _bound_occurrence(child, (*path, at), depth + shift)

        if found is not None:
            return found

    return None


def _naive_type(store: TermStore, ground: SemanticType) -> GroundType:
    """Return a ground native type as the naive checker declares it.

    Parameters
    ----------
    ground : SemanticType
    """
    if ground.op.name == "->":
        domain, codomain = ground.args

        return FunctionType(_naive_type(store, domain), _naive_type(store, codomain))

    name = (
        store.payload_name(ground)
        if ground.op.payload_sort == "constant"
        else ground.op.name
    )

    if name == "e":
        return AtomicType("e")

    if name == "t":
        return AtomicType("t")

    msg = f"the naive checker declares no type {name!r}"
    raise ValueError(msg)


def _render(ground: GroundType) -> str:
    """Return a naive ground type as prefix text.

    Parameters
    ----------
    ground : GroundType
    """
    if isinstance(ground, AtomicType):
        return ground.name

    return f"(-> {_render(ground.domain)} {_render(ground.codomain)})"


def _render_native(store: TermStore, held: SemanticType) -> str:
    """Return a ground native type as the same prefix text.

    Parameters
    ----------
    held : SemanticType
    """
    if held.op.payload_sort == "constant":
        return store.payload_name(held)

    if not held.args:
        return held.op.name

    return (
        f"({held.op.name} "
        f"{' '.join(_render_native(store, child) for child in held.args)})"
    )


def _item_count(chart: Chart[Category, float]) -> int:
    """Return how many items a chart holds over every span.

    Parameters
    ----------
    chart : Chart[Category, float]
    """
    return sum(len(chart.items_at(span)) for span in chart.index.regions_by_size())


def _keys(chart: Chart[Category, float]) -> frozenset[tuple[Cell, Category, str]]:
    """Return the span, packed category and tag of every item in a chart.

    Parameters
    ----------
    chart : Chart[Category, float]
    """
    return frozenset(
        (key.span, chart.term(key), key.nf_tag)
        for span in chart.index.regions_by_size()
        for key in chart.items_at(span)
    )


def _edges(chart: Chart[Category, float]) -> tuple[Edge[Category, float], ...]:
    """Return every edge of a chart, smallest spans first.

    Parameters
    ----------
    chart : Chart[Category, float]
    """
    return tuple(
        edge
        for span in chart.index.regions_by_size()
        for key in chart.items_at(span)
        for edge in chart.edges(key)
    )


class CoherenceSquareSuite:
    """Conformance laws for coherence-square bindings."""

    pytestmark = pytest.mark.conformance

    def test_the_meaning_lexicon_agrees_with_the_grammar_lexicon(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        lexicon = coherence_conformance.grammar.lexicon

        assert set(square.by_form) == set(lexicon.forms)

        for form in lexicon.forms:
            assert [entry.category for entry in square.by_form[form]] == [
                entry.category for entry in lexicon.entries(form)
            ]
            assert all(entry.meaning is not None for entry in square.by_form[form])

    def test_every_rule_has_a_derived_action(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        actions = _Square(coherence_conformance).reading.actions

        assert actions.uncovered == frozenset()
        assert actions.ambiguous == frozenset()
        assert set(actions.rules) == {
            rule.name for rule in coherence_conformance.grammar.grammar.rules
        }

    def test_every_derivation_concludes_the_exact_category_of_its_item(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)

        for tokens in coherence_conformance.corpus.sentences:
            chart = square.chart(tokens)

            for key, derivation in square.derivations(chart):
                assert square.category(derivation, tokens) is chart.term(key)

    def test_the_square_commutes_on_the_corpus_with_its_counts_fixed(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        batch = coherence_conformance.corpus
        checker = coherence_conformance.language.checker
        charts = [square.chart(tokens) for tokens in batch.sentences]
        reports = [
            square.report(chart, tokens)
            for chart, tokens in zip(charts, batch.sentences, strict=True)
        ]

        assert len(set(batch.sentences)) == len(batch.sentences)
        assert [report.item_count for report in reports] == [
            _item_count(chart) for chart in charts
        ]
        assert sum(report.item_count for report in reports) == batch.items
        assert sum(len(report.instances) for report in reports) == batch.instances
        assert sum(len(report.strict_items) for report in reports) == batch.strict_items

        for report in reports:
            assert len(report.strict_items) <= report.item_count

            for instance in report.instances:
                assert not instance.erased.free
                assert instance.strict == bool(instance.inferred.free)
                assert instance.strict == bool(instance.substitution)
                assert instance.strict == (
                    checker.subsumes(instance.erased, instance.inferred) is None
                )

    def test_the_three_word_chart_records_strict_instances(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        batch = coherence_conformance.three_word
        checker = coherence_conformance.language.checker
        (tokens,) = batch.sentences
        chart = square.chart(tokens)
        report = square.report(chart, tokens)

        assert len(tokens) == 3
        assert report.item_count == batch.items == _item_count(chart)
        assert len(report.instances) == batch.instances
        assert len(report.strict_items) == batch.strict_items
        assert report.strict_items < _keys(chart)

        for instance in report.strict:
            assert instance.inferred != instance.erased
            assert instance.substitution
            assert checker.subsumes(instance.erased, instance.inferred) is None

        for instance in report.instances:
            if not instance.strict:
                assert not instance.substitution

    def test_generated_charts_agree_with_the_independent_naive_checker(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        forms = coherence_conformance.grammar.lexicon.forms
        charts: set[tuple[str, ...]] = set()
        judged = 0

        @given(st.lists(st.sampled_from(forms), min_size=1, max_size=_MAX_TOKENS))
        @settings(max_examples=_EXAMPLES, deadline=None, database=None)
        def law(drawn: list[str]) -> None:
            nonlocal judged
            tokens = tuple(drawn)
            chart = square.chart(tokens)
            report = coherence_square(
                CoherenceFixture(
                    square.items(chart, tokens),
                    coherence_conformance.language.checker,
                    square.signature,
                    coherence_conformance.language.rules,
                    square.erasure,
                )
            )
            charts.add(tokens)

            assert report.item_count == _item_count(chart)

            for instance in report.instances:
                result = square.native(instance.logical_form)
                scheme = result.scheme

                assert result.is_well_typed
                assert scheme is not None
                assert square.naive_type(instance.logical_form, result) == (
                    _render_native(coherence_conformance.language.store, scheme.body)
                )
                judged += 1

        law()

        assert len(charts) >= 50
        assert judged >= len(charts)

    def test_a_mistyped_meaning_is_refused_by_both_checkers(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        language = coherence_conformance.language
        (name, *_) = (
            name
            for name, ground in coherence_conformance.schemes.items()
            if not ground.args
        )
        mistyped = language.apply(
            language.constant("and"),
            language.constant(name),
            language.constant(name),
        )
        result = language.checker.check(mistyped, language.lexicon, language.rules)

        assert not result.is_well_typed
        assert all(constraint.kind == "unification" for constraint in result.residual)
        with pytest.raises(IllTypedError):
            type_of(square.naive, square.declaration(mistyped, result))

    def test_the_oracles_type_a_constant_from_its_declaration_not_the_checker(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        language = coherence_conformance.language
        store = language.store
        truth = language.read_type("t")
        (name, *_) = (
            name
            for name, ground in coherence_conformance.schemes.items()
            if not ground.args
        )
        atom = store.signature.vocabulary(language.term_sort).id(name)
        misdeclared = LexicalSignature(
            {**language.lexicon.schemes, atom: Scheme(truth)}
        )
        constant = language.constant(name)
        result = language.checker.check(constant, misdeclared, language.rules)
        scheme = result.scheme

        assert result.is_well_typed
        assert scheme is not None
        assert scheme.body is truth
        assert square.naive_type(constant, result) == _render_native(
            coherence_conformance.language.store,
            coherence_conformance.schemes[name],
        )

    def test_no_substrate_object_is_reachable_from_a_chart(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        square = _Square(coherence_conformance)
        (tokens,) = coherence_conformance.three_word.sentences
        chart = square.chart(tokens)
        items = square.items(chart, tokens)

        assert substrate_reaches(chart) == ()
        assert substrate_reaches(items) == ()
        assert substrate_reaches(_edges(chart)) == ()

    def test_a_malformed_language_is_refused_before_any_chart_is_built(
        self, coherence_conformance: CoherenceSquareConformance
    ) -> None:
        assert coherence_conformance.malformed

        for declaration, error in coherence_conformance.malformed:
            with pytest.raises(error):
                LanguageIso().forward(declaration)
