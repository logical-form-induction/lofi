"""Tests for the synthetic string generator."""

from __future__ import annotations

from collections import Counter
from functools import cache
from gc import get_referents
from itertools import pairwise, product
from json import dumps, loads
from math import sqrt
from types import BuiltinFunctionType, FunctionType, MethodType, ModuleType
from typing import TYPE_CHECKING, Final, Literal, cast
from unittest.mock import patch

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st
from tests.declarations import SMALL, category_language, grammars
from tests.support.derivations import axiom, step
from tests.support.term_reference import infix_syntax
from tests.unit.data.featurizers.conftest import TorchCodeTensorizer
from tests.unit.data.synthetic.conftest import (
    APPLICATION,
    FLAT_INVENTORY,
    JOINED_ENTRIES,
    JOINED_INVENTORY,
    SCOPING,
    built,
    flat_generator,
    grammar,
    interpretation,
    scoped_generator,
    spurious_generator,
)
from tests.unit.spec.grammar.test__signature import slash_views, slashes

from lofi.data import (
    Derivation,
    Example,
    GoldDerivation,
    Inventory,
    Item,
    ItemHasher,
    Licensing,
    SamplingExhaustedError,
    Unlabeled,
)
from lofi.data.featurizers.unlabeled import UnlabeledFeaturizer
from lofi.data.synthetic import (
    GRAMMATICALITY_LABELS,
    SOURCE_METHOD,
    AmbiguityTarget,
    AmbiguityUnreachableError,
    Corruption,
    CorruptionExhaustedError,
    GrammarKnobs,
    Histogram,
    IllTyped,
    IllTypedFormError,
    InventoryReport,
    LabelNoise,
    LengthOutOfSupportError,
    LengthReport,
    LexiconReport,
    NegativeBinomialLength,
    NegativeDataPolicy,
    NoiseReport,
    NonDerivability,
    ObservedLabel,
    PairRelations,
    ScopeTarget,
    ScopeUnreachableError,
    ShapeSchedule,
    ShapeTarget,
    ShapeUnreachableError,
    SymbolInventory,
    SyntheticCorpus,
    SyntheticGenerator,
    TabularLength,
    Target,
    TypingConflict,
    TypingFailure,
    UncertifiedDerivationError,
    UniformLength,
    UnproductiveGrammarError,
    WeightedOperator,
)
from lofi.data.synthetic._corrupt import (
    DELETION_PLUGIN,
    INSERTION_PLUGIN,
    SUBSTITUTION_PLUGIN,
    SWAP_PLUGIN,
    CorruptionContext,
)
from lofi.data.synthetic._injection import (
    ARGUMENT_SWAP_PLUGIN,
    OVER_APPLICATION_PLUGIN,
    SORT_MISMATCH_PLUGIN,
    IllTypingContext,
)
from lofi.data.synthetic.spec.corpus import WeightedRulesDerivationSamplerSpec
from lofi.data.synthetic.spec.interpretation import (
    InterpretationSpec,
    LexicalMeaningSourceSpec,
    MeaningSpec,
    lower_interpretation,
)
from lofi.data.synthetic.spec.operators import (
    ArgumentSwapSpec,
    DeletionSpec,
    InsertionSpec,
    OverApplicationSpec,
    SortMismatchSpec,
    SubstitutionSpec,
    SwapSpec,
)
from lofi.deduce import (
    LEXICAL_TAG_PREFIX,
    LogInside,
    ParsePolicy,
    Reachability,
    Retain,
    SpanFree,
    parse,
)
from lofi.spec import (
    ApplicationSpec,
    CompositionSpec,
    ConstantPatternSpec,
    ConstructorSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PrefixConstructorNotationSpec,
    RuleSetSpec,
    TermPatternSpec,
    TypeRaisingSpec,
)
from lofi.spec.grammar import (
    BudgetSpec,
    CategoryComplexityBoundSpec,
    ClassSpec,
    ConjunctionPairSpec,
    ConjunctionsSpec,
    FeatureErasureMatchingSpec,
    GeneratedLexiconSpec,
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    HockenmaierBiskNormalFormSpec,
    InlineLexiconSpec,
    LexicalEntrySpec,
    LexiconSpec,
    LexiconStructure,
    NoNormalFormSpec,
    NormalFormSpec,
    OperatorSpec,
    ParametricAmbiguityLawSpec,
    ParametricJointSpec,
    PowerLawRankLawSpec,
    RankLawSpec,
    RootGoalSpec,
    SignatureMorphismErasureSpec,
    SignatureSpec,
    TabularJointSpec,
    UniformRankLawSpec,
    canonical_bytes,
)
from lofi.spec.language import ConstantSpec, LanguageIso, registered_language
from lofi.spec.provers import TruthTableProverSpec

if TYPE_CHECKING:
    from collections.abc import Callable, Iterator, Sequence

    from lofi.data.synthetic import (
        FormTypingLabel,
        LengthDistribution,
    )
    from lofi.data.synthetic._corrupt import CorruptionOperator
    from lofi.data.synthetic._injection import IllTypingOperator
    from lofi.deduce import Cell
    from lofi.deduce import Derivation as ChartDerivation
    from lofi.deduce import Derivation as MeaningDerivation
    from lofi.judge import Rule
    from lofi.lexicon import Entry
    from lofi.terms import Term, TermStore

type Shape = tuple[tuple[int, int], str, str | None, tuple["Shape", ...]]

PRIMITIVES: Final = ("S", "NP", "conj", "N", "PP")
NO_TRIVIAL: Final = PairRelations(trivial=0.0, entailment=0)
CATEGORIES: Final = ("NP", "S\\NP", "(S\\NP)/NP", "NP/NP", "(S\\NP)\\(S\\NP)")
FOREST_POLICY: Final = ParsePolicy(Retain.FOREST, Reachability.ALL)
CONJUNCTIONS: Final = ConjunctionsSpec(
    markers=("conj",),
    pairs=(ConjunctionPairSpec(unmarked="NP", marked="NP[conj]"),),
)


def _swap(context: CorruptionContext) -> CorruptionOperator:
    return SWAP_PLUGIN.compile(SwapSpec())(context)


def _deletion(context: CorruptionContext) -> CorruptionOperator:
    return DELETION_PLUGIN.compile(DeletionSpec())(context)


def _insertion(context: CorruptionContext) -> CorruptionOperator:
    return INSERTION_PLUGIN.compile(InsertionSpec())(context)


def _substitution(context: CorruptionContext) -> CorruptionOperator:
    return SUBSTITUTION_PLUGIN.compile(SubstitutionSpec())(context)


STANDARD_CORRUPTIONS: Final = (_swap, _deletion, _insertion, _substitution)


def _corruption(
    generator: SyntheticGenerator,
    max_length: int,
    proportion: float,
    *,
    operators: tuple[
        Callable[[CorruptionContext], CorruptionOperator], ...
    ] = STANDARD_CORRUPTIONS,
    attempts: int,
    policy: NegativeDataPolicy | None = None,
) -> Corruption[CorruptionOperator]:
    context = CorruptionContext(
        generator.lexicon, generator.grammar.matching, max_length
    )
    return Corruption(
        proportion,
        operators=tuple(WeightedOperator(build(context), 1.0) for build in operators),
        attempts=attempts,
        draw="exact",
        policy=policy,
    )


def _argument_swap(context: IllTypingContext) -> IllTypingOperator:
    return ARGUMENT_SWAP_PLUGIN.construct(ArgumentSwapSpec(), context)


def _over_application(context: IllTypingContext) -> IllTypingOperator:
    return OVER_APPLICATION_PLUGIN.construct(OverApplicationSpec(), context)


def _sort_mismatch(context: IllTypingContext) -> IllTypingOperator:
    return SORT_MISMATCH_PLUGIN.construct(SortMismatchSpec(), context)


STANDARD_ILL_TYPING: Final = (_argument_swap, _over_application, _sort_mismatch)


def _ill_typing(
    generator: SyntheticGenerator,
    proportion: float,
    *,
    operators: tuple[
        Callable[[IllTypingContext], IllTypingOperator], ...
    ] = STANDARD_ILL_TYPING,
    attempts: int,
    policy: NegativeDataPolicy | None = None,
) -> Corruption[IllTypingOperator]:
    interpretation = generator.interpretation
    if interpretation is None:
        msg = "the test fixture needs an interpretation"
        raise ValueError(msg)
    context = IllTypingContext(interpretation)
    return Corruption(
        proportion,
        operators=tuple(WeightedOperator(build(context), 1.0) for build in operators),
        attempts=attempts,
        draw="exact",
        policy=policy,
    )


def _rules(combinators: Sequence[str]) -> RuleSetSpec:
    rules = [
        ApplicationSpec(view="/"),
        ApplicationSpec(view="\\"),
    ]

    if "composition" in combinators:
        rules.append(CompositionSpec(direction="unrestricted", order=1, view="/"))
        rules.append(CompositionSpec(direction="unrestricted", order=1, view="\\"))

    if "type-raising" in combinators:
        rules.append(TypeRaisingSpec(view="/", targets=("S",), arguments=None))

    return RuleSetSpec(name="knobbed", rules=tuple(rules))


def _entries(
    forms: Sequence[str], categories_per_form: int
) -> tuple[LexicalEntrySpec, ...]:
    return tuple(
        LexicalEntrySpec(
            form=form, category=CATEGORIES[(index + offset) % len(CATEGORIES)]
        )
        for index, form in enumerate(forms)
        for offset in range(categories_per_form)
    )


def _declaration(
    *,
    vocabulary_size: int = 4,
    primitive_count: int = 3,
    combinators: Sequence[str] = ("application",),
    categories_per_form: int = 1,
    entries: Sequence[LexicalEntrySpec] | None = None,
    forms: Sequence[str] | None = None,
    unary_rounds: int = 1,
    closure_rounds: int = 2,
    threads: int = 1,
    normal_form: NormalFormSpec | None = None,
) -> GrammarDecl:
    held = (
        tuple(f"w{index}" for index in range(vocabulary_size))
        if forms is None
        else tuple(forms)
    )

    return GrammarDecl(
        name="knobbed",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=PRIMITIVES[:primitive_count],
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=CONJUNCTIONS,
        ),
        rules=_rules(combinators),
        lexicon=InlineLexiconSpec(
            entries=(
                _entries(held, categories_per_form)
                if entries is None
                else tuple(entries)
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        normal_form=(
            HockenmaierBiskNormalFormSpec() if normal_form is None else normal_form
        ),
        budget=BudgetSpec(
            unary_rounds=unary_rounds,
            closure_rounds=closure_rounds,
            threads=threads,
        ),
    )


def _generator(declaration: GrammarDecl) -> SyntheticGenerator:
    return SyntheticGenerator(_assembly(declaration), policy=FOREST_POLICY)


def _assembly(declaration: GrammarDecl) -> GrammarAssembly:
    """Lower one test grammar inside the explicit propositional language."""
    language = category_language(declaration, registered_language("propositional"))

    return GrammarIso().forward(declaration, language)


def test_a_generator_refuses_an_inside_only_parse_policy() -> None:
    policy = cast(
        "ParsePolicy[Literal[Retain.FOREST]]",
        ParsePolicy(Retain.INSIDE, Reachability.ALL),
    )

    with pytest.raises(ValueError, match="requires packed-forest retention"):
        SyntheticGenerator(_assembly(_declaration()), policy=policy)


def test_declared_threads_reach_the_synthetic_parser() -> None:
    document = _declaration(threads=2).model_dump_json()
    generator = _generator(GrammarDecl.model_validate_json(document))

    with patch("lofi.data.synthetic.generator.parse", wraps=parse) as parser:
        generator.parse(("w0", "w1"))

    assert parser.call_args is not None
    assert parser.call_args.kwargs["threads"] == 2


class _UnitScorer:
    lowering = SpanFree()

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


def _parsed_shapes(assembly: GrammarAssembly, tokens: Sequence[str]) -> set[Shape]:
    """Return every root derivation the chart parser finds, as a shape."""
    grammar = assembly.grammar
    chart = parse(
        tokens,
        rules=grammar.rules,
        source=assembly.lexicon,
        semiring=LogInside(),
        scorer=_UnitScorer(),
        goal=grammar.goal,
        unary_depth=grammar.budget.unary_rounds,
        policy=FOREST_POLICY,
    )

    def shape(derivation: ChartDerivation[Term, float]) -> Shape:
        edge = derivation.edge
        rule = None if edge.rule_name.startswith(LEXICAL_TAG_PREFIX) else edge.rule_name

        return (
            derivation.key.span,
            assembly.syntax.render(grammar.store, edge.output),
            rule,
            tuple(shape(child) for child in derivation.children),
        )

    return {
        shape(derivation)
        for root in grammar.goal.roots(chart)
        for derivation in chart.derivations(root)
    }


def _gold_shape(derivation: Derivation[str]) -> Shape:
    def shape(node_id: int) -> Shape:
        node = derivation.node(node_id)

        return (
            (node.span.start, node.span.end),
            node.label,
            node.rule,
            tuple(shape(child) for child in node.children),
        )

    return shape(derivation.root)


def _unary_depth(derivation: Derivation[str]) -> int:
    def chain(node_id: int) -> int:
        children = derivation.node(node_id).children

        return 1 + chain(children[0]) if len(children) == 1 else 0

    return max(chain(node_id) for node_id in derivation.nodes)


def _reachable_types(root: SyntheticCorpus) -> Iterator[type]:
    seen: set[int] = set()
    pending = [root]

    while pending:
        value = pending.pop()

        if id(value) in seen or isinstance(
            value, type | ModuleType | FunctionType | MethodType | BuiltinFunctionType
        ):
            continue

        seen.add(id(value))

        yield type(value)

        pending.extend(get_referents(value))


def _unary(source: str, target: str) -> PatternRuleSpec:
    """Declare a unary pattern rule rewriting one atom into another."""

    def node(name: str) -> ConstantPatternSpec:
        return ConstantPatternSpec(name=name)

    return PatternRuleSpec(
        name=f"{source}-to-{target}",
        premises=(node(source),),
        conclusion=node(target),
        head_policy="pass-through",
    )


def _named_rewrite(name: str, source: str, target: str, arity: int) -> PatternRuleSpec:
    """Declare a named pattern rule with repeated atomic premises."""

    def node(atom: str) -> ConstantPatternSpec:
        return ConstantPatternSpec(name=atom)

    return PatternRuleSpec(
        name=name,
        premises=tuple(node(source) for _ in range(arity)),
        conclusion=node(target),
    )


def _functor(constructor: str, result: str, argument: str) -> str:
    """Write a functor category as an s-expression."""
    return f"({dumps(constructor)} {result} {argument})"


def _atom_text(name: str) -> str:
    """Write a category constant in canonical s-expression syntax."""
    return f'("#const[Cat]" {dumps(name)})'


def _explicit(*, right: str, left: str) -> GrammarDecl:
    """Declare an application-only grammar over explicitly listed constructors.

    A rightward functor takes its argument on its right and a leftward one on
    its left. Naming the same constructor for both declares the undirected
    arrow grammar, and naming the two slashes the directed twin.
    """
    result = TermPatternSpec(name="X", sort="Cat")
    argument = TermPatternSpec(name="Y", sort="Cat")

    def functor(constructor: str) -> NodePatternSpec:
        return NodePatternSpec(
            operator=FixedOperatorSpec(name=constructor), arguments=(result, argument)
        )

    return GrammarDecl(
        name="explicit",
        signature=SignatureSpec(
            name="explicit",
            primitives=("S", "NP", "N"),
            constructors=tuple(
                ConstructorSpec(
                    name=constructor,
                    kind=("Cat", "Cat", "Cat"),
                    notation=PrefixConstructorNotationSpec(
                        spelling="->" if constructor == "arrow" else constructor
                    ),
                )
                for constructor in dict.fromkeys((right, left))
            ),
            syntax="s-expression",
        ),
        rules=RuleSetSpec(
            name="application",
            rules=(
                PatternRuleSpec(
                    name="application-right",
                    premises=(functor(right), argument),
                    conclusion=result,
                ),
                PatternRuleSpec(
                    name="application-left",
                    premises=(argument, functor(left)),
                    conclusion=result,
                ),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="w0", category=_atom_text("NP")),
                LexicalEntrySpec(
                    form="w1",
                    category=_functor(left, _atom_text("S"), _atom_text("NP")),
                ),
                LexicalEntrySpec(
                    form="w2",
                    category=_functor(
                        right,
                        _functor(left, _atom_text("S"), _atom_text("NP")),
                        _atom_text("NP"),
                    ),
                ),
                LexicalEntrySpec(
                    form="w3",
                    category=_functor(right, _atom_text("NP"), _atom_text("NP")),
                ),
            )
        ),
        goal=RootGoalSpec(categories=(_atom_text("S"),)),
        budget=BudgetSpec(unary_rounds=1, closure_rounds=2, closure_nodes=32),
    )


def test_the_declaration_round_trips_and_the_corpus_holds_its_bytes() -> None:
    declaration = _declaration()
    encoded = canonical_bytes(declaration)
    iso = GrammarIso()
    assembly = iso.forward(declaration, registered_language("propositional"))

    assert GrammarDecl.model_validate_json(encoded) == declaration
    assert (
        iso.forward(
            GrammarDecl.model_validate_json(encoded),
            registered_language("propositional"),
        )
        is assembly
    )

    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        5, seed=3, length=UniformLength(1, 4)
    )

    assert corpus.declaration == encoded
    assert corpus.declaration_digest == assembly.declaration_digest
    assert corpus.theory == assembly.grammar.theory_specification
    assert corpus.theory_digest == assembly.grammar.theory_digest
    assert corpus.seed == 3
    assert corpus.max_length == 4
    assert len(corpus) == 5


def test_the_knobs_are_read_off_the_compiled_grammar() -> None:
    generator = _generator(
        _declaration(
            vocabulary_size=5,
            primitive_count=4,
            combinators=("application", "composition", "type-raising"),
            categories_per_form=2,
        )
    )

    assert generator.knobs == GrammarKnobs(
        4,
        ("/", "\\"),
        (
            "application:/",
            "application:\\",
            "composition:/:1:harmonic:unrestricted",
            "composition:\\:1:harmonic:unrestricted",
            "type-raising:/:targets=S",
        ),
    )
    assert (
        generator.sample(3, seed=1, length=UniformLength(1, 4)).knobs == generator.knobs
    )


@pytest.mark.parametrize(
    ("varied", "section", "field"),
    [
        (_declaration(primitive_count=5), "signature", "primitive_count"),
        (
            _declaration(combinators=("application", "composition")),
            "rules",
            "combinators",
        ),
    ],
    ids=["primitives", "combinators"],
)
def test_changing_one_knob_changes_only_its_dimension(
    varied: GrammarDecl, section: str, field: str
) -> None:
    base = _declaration()
    base_document = loads(canonical_bytes(base))
    varied_document = loads(canonical_bytes(varied))

    assert {
        key for key in base_document if base_document[key] != varied_document[key]
    } == {section}

    base_knobs = _generator(base).knobs
    varied_knobs = _generator(varied).knobs
    differing = {
        name
        for name in ("primitive_count", "constructors", "combinators")
        if getattr(base_knobs, name) != getattr(varied_knobs, name)
    }

    assert differing == {field}


@pytest.mark.parametrize(
    "varied",
    [_declaration(vocabulary_size=6), _declaration(categories_per_form=2)],
    ids=["vocabulary", "ambiguity"],
)
def test_changing_the_lexicon_changes_its_measured_joint_and_no_knob(
    varied: GrammarDecl,
) -> None:
    base = _declaration()
    base_document = loads(canonical_bytes(base))
    varied_document = loads(canonical_bytes(varied))

    assert {
        key for key in base_document if base_document[key] != varied_document[key]
    } == {"lexicon"}

    base_corpus = _generator(base).sample(30, seed=1, length=UniformLength(1, 4))
    varied_corpus = _generator(varied).sample(30, seed=1, length=UniformLength(1, 4))

    assert base_corpus.knobs == varied_corpus.knobs
    assert base_corpus.lexicon.forms.total == 4
    assert varied_corpus.lexicon.forms.total == 6 or (
        varied_corpus.lexicon.class_sizes != base_corpus.lexicon.class_sizes
    )
    assert varied_corpus.lexicon.class_sizes != base_corpus.lexicon.class_sizes


def test_changing_the_constructor_set_changes_only_its_dimension() -> None:
    directed = _explicit(right="/", left="\\")
    undirected = _explicit(right="arrow", left="arrow")
    directed_document = loads(canonical_bytes(directed))
    undirected_document = loads(canonical_bytes(undirected))
    directed_signature = directed_document["signature"]
    undirected_signature = undirected_document["signature"]

    assert {
        key
        for key in directed_document
        if directed_document[key] != undirected_document[key]
    } == {"signature", "rules", "lexicon"}
    assert {
        key
        for key in directed_signature
        if directed_signature[key] != undirected_signature[key]
    } == {"constructors"}
    assert [
        constructor["name"] for constructor in directed_signature["constructors"]
    ] == ["/", "\\"]
    assert [
        constructor["name"] for constructor in undirected_signature["constructors"]
    ] == ["arrow"]

    directed_knobs = _generator(directed).knobs
    undirected_knobs = _generator(undirected).knobs

    assert directed_knobs.constructors == ("/", "\\")
    assert undirected_knobs.constructors == ("arrow",)
    assert directed_knobs == GrammarKnobs(
        3, ("/", "\\"), ("application-right", "application-left")
    )
    assert undirected_knobs == GrammarKnobs(
        3, ("arrow",), ("application-right", "application-left")
    )


def test_an_undirected_grammar_draws_functors_on_either_side() -> None:
    assembly = _assembly(_explicit(right="arrow", left="arrow"))
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        30, seed=2, length=UniformLength(1, 5)
    )
    rules = Counter(
        node.rule
        for example in corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if node.rule is not None
    )

    for example in corpus.examples:
        assert _gold_shape(example.supervision.derivation) in _parsed_shapes(
            assembly, tuple(token.surface for token in example.item.tokens)
        )

    assert {len(example.item.tokens) for example in corpus.examples} == {2, 3, 4, 5}
    assert set(rules) == {"application-right", "application-left"}
    assert {
        tuple(token.surface for token in example.item.tokens)
        for example in corpus.examples
        if len(example.item.tokens) == 2
    } == {("w0", "w1"), ("w1", "w0")}


def _sampled_rules(normal_form: NormalFormSpec) -> Counter[str]:
    """Count the rules the generator uses under a normal form."""
    declaration = _declaration(
        vocabulary_size=5,
        combinators=("application", "composition", "type-raising"),
        unary_rounds=2,
        normal_form=normal_form,
    )
    corpus = SyntheticGenerator(_assembly(declaration), policy=FOREST_POLICY).sample(
        120, seed=11, length=UniformLength(1, 6)
    )

    return Counter(
        node.rule
        for example in corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if node.rule is not None
    )


@pytest.mark.parametrize(
    "normal_form",
    [NoNormalFormSpec(), HockenmaierBiskNormalFormSpec()],
    ids=lambda spec: spec.kind,
)
def test_every_example_reparses_to_its_gold_derivation(
    normal_form: NormalFormSpec,
) -> None:
    declaration = _declaration(
        vocabulary_size=5,
        combinators=("application", "composition", "type-raising"),
        unary_rounds=2,
        normal_form=normal_form,
    )
    assembly = _assembly(declaration)
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        120, seed=11, length=UniformLength(1, 6)
    )
    rules = Counter(
        node.rule
        for example in corpus.examples
        for node in example.supervision.derivation.nodes.values()
        if node.rule is not None
    )

    for example in corpus.examples:
        tokens = tuple(token.surface for token in example.item.tokens)
        derivation = example.supervision.derivation

        assert derivation.span.length == len(tokens)
        assert _gold_shape(derivation) in _parsed_shapes(assembly, tokens)
        assert _unary_depth(derivation) <= 2

    assert {len(example.item.tokens) for example in corpus.examples} == {
        2,
        3,
        4,
        5,
        6,
    }

    assert rules["type-raising:/:targets=S"] > 0
    assert rules["composition:/:1:harmonic:unrestricted"] > 0

    if isinstance(normal_form, HockenmaierBiskNormalFormSpec):
        # The filter bans configurations, not rules, so it removes derivations
        # from the free grammar without reaching for a rule the free grammar
        # never uses.
        assert set(rules) <= set(_sampled_rules(NoNormalFormSpec()))


def _chained(unary_rounds: int, forms: tuple[str, ...] = ("D",)) -> GrammarDecl:
    """Declare a grammar whose root derivations stack unary steps."""
    return GrammarDecl(
        name="chained",
        signature=SignatureSpec(
            name="chained", primitives=("D", "E", "F"), syntax="s-expression"
        ),
        rules=RuleSetSpec(name="chain", rules=(_unary("D", "E"), _unary("E", "F"))),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form="x", category=_atom_text(form)) for form in forms
            )
        ),
        goal=RootGoalSpec(categories=(_atom_text("F"),)),
        budget=BudgetSpec(unary_rounds=unary_rounds, closure_rounds=2, closure_nodes=8),
    )


def test_unary_rounds_bound_the_chains_the_sample_draws() -> None:
    with pytest.raises(UnproductiveGrammarError):
        _generator(_chained(1)).sample(1, seed=0, length=UniformLength(1, 1))

    corpus = _generator(_chained(2)).sample(5, seed=0, length=UniformLength(1, 1))

    for example in corpus.examples:
        derivation = example.supervision.derivation

        assert _unary_depth(derivation) == 2
        assert _gold_shape(derivation) == (
            (0, 1),
            _atom_text("F"),
            "E-to-F",
            (
                (
                    (
                        (0, 1),
                        _atom_text("E"),
                        "D-to-E",
                        (((0, 1), _atom_text("D"), None, ()),),
                    )
                ),
            ),
        )


def test_a_chain_longer_than_the_rounds_is_not_drawn_even_when_parsable() -> None:
    assembly = _assembly(_chained(1, forms=("D", "E")))
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        30, seed=0, length=UniformLength(1, 1)
    )
    direct = (
        (0, 1),
        _atom_text("F"),
        "E-to-F",
        (((0, 1), _atom_text("E"), None, ()),),
    )
    stacked = (
        (0, 1),
        _atom_text("F"),
        "E-to-F",
        (
            (
                (
                    (0, 1),
                    _atom_text("E"),
                    "D-to-E",
                    (((0, 1), _atom_text("D"), None, ()),),
                )
            ),
        ),
    )

    assert _parsed_shapes(assembly, ("x",)) == {direct, stacked}
    assert {
        _gold_shape(example.supervision.derivation) for example in corpus.examples
    } == {direct}


def test_a_lexicon_with_two_derivations_of_one_string_draws_both() -> None:
    declaration = _declaration(
        vocabulary_size=3,
        combinators=("application", "composition", "type-raising"),
        normal_form=NoNormalFormSpec(),
    )
    assembly = _assembly(declaration)
    shapes = _parsed_shapes(assembly, ("w0", "w2", "w0"))

    assert len(shapes) > 1

    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        80, seed=9, length=UniformLength(1, 3)
    )
    drawn = {
        _gold_shape(example.supervision.derivation)
        for example in corpus.examples
        if tuple(token.surface for token in example.item.tokens) == ("w0", "w2", "w0")
    }

    assert drawn == shapes


def test_a_length_with_several_root_categories_draws_each_in_proportion() -> None:
    declaration = _declaration(vocabulary_size=6).with_(
        goal=RootGoalSpec(categories=("S", "NP"))
    )
    assembly = _assembly(declaration)
    forms = tuple(f"w{index}" for index in range(6))
    derivations = Counter(
        (length, shape[1])
        for length in range(1, 4)
        for tokens in product(forms, repeat=length)
        for shape in _parsed_shapes(assembly, tokens)
    )

    assert derivations == {
        (1, "NP"): 2,
        (2, "NP"): 2,
        (2, "S"): 2,
        (3, "NP"): 2,
        (3, "S"): 8,
    }

    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        300, seed=3, length=UniformLength(1, 3)
    )
    drawn = Counter(
        (len(example.item.tokens), _gold_shape(example.supervision.derivation)[1])
        for example in corpus.examples
    )
    per_length = Counter(len(example.item.tokens) for example in corpus.examples)

    assert set(drawn) == set(derivations)

    for (length, root), count in derivations.items():
        expected = count / sum(
            held for (other, _), held in derivations.items() if other == length
        )

        assert abs(drawn[(length, root)] / per_length[length] - expected) < 0.1


def test_items_have_exact_utf8_spans_and_deterministic_identifiers() -> None:
    assembly = _assembly(_declaration(forms=("niño", "über", "æ", "ß")))
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        12, seed=5, length=UniformLength(1, 4)
    )
    digest = assembly.declaration_digest[:12]

    assert [example.item.item_id for example in corpus.examples] == [
        f"{digest}:5:{index}" for index in range(12)
    ]

    for example in corpus.examples:
        item = example.item
        text = item.text

        assert text is not None
        assert text == " ".join(token.surface for token in item.tokens)

        for token in item.tokens:
            span = token.span

            assert span is not None
            assert span.substring(text) == token.surface
            assert span.length == len(token.surface.encode())

        for earlier, later in pairwise(token.span for token in item.tokens):
            assert earlier is not None
            assert later is not None
            assert later.start == earlier.end + 1


def test_the_same_seed_draws_the_same_bytes_and_another_seed_varies_them() -> None:
    generator = _generator(_declaration())
    first = generator.sample(25, seed=17, length=UniformLength(1, 5))
    again = generator.sample(25, seed=17, length=UniformLength(1, 5))
    other = generator.sample(25, seed=18, length=UniformLength(1, 5))

    assert first == again
    assert hash(first) == hash(again)
    assert [example.item.tokens for example in first.examples] != [
        example.item.tokens for example in other.examples
    ]
    assert other.seed == 18


def test_provenance_records_the_method_the_terms_and_the_seed() -> None:
    assembly = _assembly(_declaration())
    default = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        2, seed=1, length=UniformLength(1, 3)
    )
    named = SyntheticGenerator(
        assembly, policy=FOREST_POLICY, licensing=Licensing("MIT")
    ).sample(2, seed=1, length=UniformLength(1, 3))

    for example in default.examples:
        provenance = example.item.provenance

        assert provenance is not None
        assert provenance.source_method == SOURCE_METHOD
        assert provenance.licensing == Licensing("CC0-1.0")
        assert provenance.reproducibility is not None
        assert provenance.reproducibility.random_seed == 1

    for example in named.examples:
        provenance = example.item.provenance

        assert provenance is not None
        assert provenance.licensing == Licensing("MIT")


@pytest.mark.parametrize(
    ("declaration", "max_length"),
    [
        (
            _declaration(
                entries=tuple(
                    LexicalEntrySpec(form="w0", category=category)
                    for category in ("NP/conj", "conj", "(S\\NP)/NP")
                ),
                closure_rounds=3,
            ),
            4,
        ),
        (_declaration(), 1),
    ],
    ids=["requires-five-tokens", "too-short"],
)
def test_an_unproductive_grammar_raises_the_named_exhaustion_error(
    declaration: GrammarDecl, max_length: int
) -> None:
    generator = _generator(declaration)

    with pytest.raises(UnproductiveGrammarError, match="root category S") as caught:
        generator.sample(0, seed=0, length=UniformLength(1, max_length))

    assert isinstance(caught.value, SamplingExhaustedError)
    assert f"at most {max_length} tokens" in str(caught.value)


def test_a_count_of_zero_draws_nothing_from_a_productive_grammar() -> None:
    corpus = _generator(_declaration()).sample(0, seed=0, length=UniformLength(1, 2))

    assert corpus.examples == ()
    assert len(corpus) == 0


def test_the_sample_rejects_a_bad_count_or_seed() -> None:
    generator = _generator(_declaration())

    with pytest.raises(ValueError, match="count is negative"):
        generator.sample(-1, seed=0, length=UniformLength(1, 2))

    with pytest.raises(TypeError, match="seed is not an integer"):
        generator.sample(1, seed=cast("int", 1.5), length=UniformLength(1, 2))


@pytest.mark.parametrize("primitives", [2, 4, 8, 16])
def test_a_corpus_over_a_generated_lexicon_reports_the_inventory_it_reached(
    primitives: int,
) -> None:
    names = ("S", *(f"P{index}" for index in range(1, primitives)))
    declaration = GrammarDecl(
        name="generated",
        signature=SignatureSpec(
            constructors=slashes(), functors=slash_views(), primitives=names
        ),
        rules=_rules(("application",)),
        lexicon=GeneratedLexiconSpec(
            forms=12,
            categories=8,
            category_shares=UniformRankLawSpec(),
            coupling=0.0,
            ambiguity=ParametricAmbiguityLawSpec(
                mean=2.0, dispersion=0.0, maximum=3, placement=0.0
            ),
            rank_law=UniformRankLawSpec(),
            seed=3,
            max_depth=2,
        ),
        matching=FeatureErasureMatchingSpec(),
    )
    corpus = _generator(declaration).sample(20, seed=1, length=UniformLength(2, 5))
    inventory = corpus.inventory

    assert len(corpus.examples) == 20
    assert inventory.size == 8
    assert max(inventory.depths.counts) <= 2
    assert set(inventory.primitives) <= set(names)
    assert inventory.realized_primitive_count == len(set(inventory.primitives))
    assert corpus.lexicon.realized_vocabulary_size == len(
        {token.surface for example in corpus.examples for token in example.item.tokens}
    )
    assert set(inventory.categories) == set(corpus.lexicon.class_sizes)
    assert corpus.knobs.primitive_count == primitives

    if primitives == 16:
        assert inventory.realized_primitive_count < primitives


def _bounded_category_report(
    field: Literal["max_arguments", "max_order"], value: int
) -> SyntheticCorpus:
    """Draw one corpus under a generated-category complexity bound."""
    names = ("S", *(f"P{index}" for index in range(1, 8)))
    lexicon = GeneratedLexiconSpec(
        forms=12,
        categories=8,
        category_shares=UniformRankLawSpec(),
        coupling=0.0,
        ambiguity=ParametricAmbiguityLawSpec(
            mean=2.0, dispersion=0.0, maximum=3, placement=0.0
        ),
        rank_law=UniformRankLawSpec(),
        seed=11,
        max_depth=3,
    ).with_(**{field: CategoryComplexityBoundSpec(value=value)})
    declaration = GrammarDecl(
        name=f"generated-{field}-{value}",
        signature=SignatureSpec(
            constructors=slashes(), functors=slash_views(), primitives=names
        ),
        rules=_rules(("application",)),
        lexicon=lexicon,
        matching=FeatureErasureMatchingSpec(),
    )

    return _generator(declaration).sample(100, seed=1, length=UniformLength(2, 5))


@pytest.mark.parametrize(
    ("field", "achieved"),
    [("max_arguments", "arguments"), ("max_order", "orders")],
)
def test_category_complexity_bounds_move_only_the_named_achieved_factor(
    field: Literal["max_arguments", "max_order"],
    achieved: Literal["arguments", "orders"],
) -> None:
    """A wider category bound raises its report at fixed PI-04 declarations."""
    narrow = _bounded_category_report(field, 1)
    wide = _bounded_category_report(field, 2)
    narrow_histogram = (
        narrow.inventory.arguments
        if achieved == "arguments"
        else narrow.inventory.orders
    )
    wide_histogram = (
        wide.inventory.arguments if achieved == "arguments" else wide.inventory.orders
    )
    narrow_declared = (
        narrow.inventory.max_arguments
        if field == "max_arguments"
        else narrow.inventory.max_order
    )
    wide_declared = (
        wide.inventory.max_arguments
        if field == "max_arguments"
        else wide.inventory.max_order
    )

    assert narrow_histogram.mean < wide_histogram.mean
    assert narrow_declared == 1
    assert wide_declared == 2
    assert narrow.inventory.size == wide.inventory.size == 8
    assert narrow.lexicon.realized_size == wide.lexicon.realized_size == 12
    assert narrow.lexicon.per_type_ambiguity == wide.lexicon.per_type_ambiguity
    assert sorted(narrow.lexicon.class_sizes.values()) == sorted(
        wide.lexicon.class_sizes.values()
    )
    assert narrow.length.distribution == wide.length.distribution
    assert narrow.lexicon.coupling == wide.lexicon.coupling
    assert narrow.lexicon.concordance == wide.lexicon.concordance


def test_a_lexicon_with_uneven_category_counts_is_measured_not_refused() -> None:
    declaration = _declaration().with_(
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="w0", category="NP"),
                LexicalEntrySpec(form="w0", category="NP/NP"),
                LexicalEntrySpec(form="w1", category="S\\NP"),
            )
        )
    )
    generator = _generator(declaration)
    corpus = generator.sample(30, seed=1, length=UniformLength(1, 3))

    assert set(corpus.ambiguity.lexical.counts) == {1, 2}
    assert corpus.lexicon.class_sizes == {"NP": 1, "NP/NP": 1, "S\\NP": 1}
    assert corpus.lexicon.forms.total == 2
    assert corpus.inventory == InventoryReport(
        {"NP": 0, "NP/NP": 1, "S\\NP": 1},
        {"NP": 0, "NP/NP": 1, "S\\NP": 1},
        {"NP": 0, "NP/NP": 1, "S\\NP": 1},
        ("S", "NP"),
        max_arguments=None,
        max_order=None,
    )

    with pytest.raises(
        AmbiguityUnreachableError,
        match=r"4 outside the lexical target Target\(3\.0, 0\.5\) at a mean of",
    ):
        generator.sample(
            5,
            seed=1,
            length=UniformLength(1, 3),
            ambiguity=AmbiguityTarget(lexical=Target(3, 0.5), attempts=4),
        )


def test_a_lexical_band_keeps_only_strings_whose_mean_ambiguity_lies_in_it() -> None:
    declaration = _declaration().with_(
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="w0", category="NP"),
                LexicalEntrySpec(form="w0", category="NP/NP"),
                LexicalEntrySpec(form="w1", category="S\\NP"),
            )
        )
    )
    generator = _generator(declaration)
    free = generator.sample(40, seed=1, length=UniformLength(1, 3))
    two = generator.sample(
        40,
        seed=1,
        length=UniformLength(1, 3),
        ambiguity=AmbiguityTarget(lexical=Target(1.5, 0), attempts=50),
    )
    three = generator.sample(
        40,
        seed=1,
        length=UniformLength(1, 3),
        ambiguity=AmbiguityTarget(lexical=Target(5 / 3, 0.01), attempts=50),
    )

    assert {_surface(example.item) for example in free.examples} == {
        ("w0", "w1"),
        ("w0", "w0", "w1"),
    }
    assert free.ambiguity.rejected == 0
    assert {_surface(example.item) for example in two.examples} == {("w0", "w1")}
    assert two.ambiguity.lexical == Histogram({1: 40, 2: 40})
    assert two.ambiguity.rejected > 0
    assert {_surface(example.item) for example in three.examples} == {
        ("w0", "w0", "w1")
    }
    assert three.ambiguity.lexical == Histogram({1: 40, 2: 80})
    assert three.ambiguity.rejected > 0


@pytest.mark.parametrize("value", [1.2, 2.0, 3.0])
def test_a_generated_lexicons_lexical_band_is_met_or_names_the_mean(
    value: float,
) -> None:
    declaration = GrammarDecl(
        name="generated-ambiguity",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "P1", "P2", "P3"),
        ),
        rules=_rules(("application",)),
        lexicon=GeneratedLexiconSpec(
            forms=40,
            categories=8,
            category_shares=PowerLawRankLawSpec(exponent=1.0),
            coupling=0.0,
            ambiguity=ParametricAmbiguityLawSpec(
                mean=2.0, dispersion=0.5, maximum=4, placement=0.5
            ),
            rank_law=PowerLawRankLawSpec(exponent=1.1),
            seed=3,
            max_depth=2,
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )
    lexical = Target(value, 0.1)
    ambiguity = AmbiguityTarget(lexical=lexical, attempts=200)

    try:
        corpus = _generator(declaration).sample(
            400,
            seed=5,
            length=NegativeBinomialLength(3.0, 0.6, min_length=2, max_length=9),
            ambiguity=ambiguity,
        )
    except AmbiguityUnreachableError as refused:
        message = str(refused)

        assert f"outside the lexical target {lexical!r} at a mean of " in message
        assert "categories per token" in message
    else:
        assert lexical.admits(corpus.ambiguity.lexical.mean)
        assert corpus.ambiguity.target is ambiguity


def test_a_one_token_string_is_measured_by_its_own_category_count() -> None:
    declaration = _declaration().with_(
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="w0", category="S"),
                LexicalEntrySpec(form="w0", category="NP"),
                LexicalEntrySpec(form="w1", category="S\\NP"),
            )
        )
    )
    corpus = _generator(declaration).sample(
        20,
        seed=1,
        length=UniformLength(1, 2),
        ambiguity=AmbiguityTarget(lexical=Target(2, 0), attempts=50),
    )

    assert {_surface(example.item) for example in corpus.examples} == {("w0",)}
    assert corpus.ambiguity.lexical == Histogram({2: 20})


def test_no_serialized_derivation_or_substrate_object_is_emitted() -> None:
    assembly = _assembly(
        _declaration(combinators=("application", "composition", "type-raising"))
    )
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        20, seed=6, length=UniformLength(1, 5)
    )
    foreign = sorted(
        {
            f"{held.__module__}.{held.__qualname__}"
            for held in _reachable_types(corpus)
            if held.__module__.partition(".")[0] in {"didactic", "panproto"}
            or held.__module__.startswith(
                ("lofi.terms", "lofi.deduce", "lofi.judge", "lofi.formalism")
            )
        }
    )

    assert foreign == []

    rule_names = {rule.name for rule in assembly.grammar.rules}

    for example in corpus.examples:
        for node in example.supervision.derivation.nodes.values():
            assert assembly.syntax.parse(assembly.store, node.label) is not None
            assert node.rule is None or node.rule in rule_names
            assert node.rule is None or not node.rule.startswith(LEXICAL_TAG_PREFIX)
            assert node.is_lexical == (node.rule is None)
            assert not node.features


def _cyclic(
    rules: tuple[PatternRuleSpec, ...], forms: tuple[str, ...], goal: str
) -> GrammarDecl:
    """Declare a grammar whose unary rules cycle over three atoms."""
    return GrammarDecl(
        name="cyclic",
        signature=SignatureSpec(
            name="cyclic", primitives=("D", "E", "F"), syntax="s-expression"
        ),
        rules=RuleSetSpec(name="cycle", rules=rules),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form="x", category=_atom_text(form)) for form in forms
            )
        ),
        goal=RootGoalSpec(categories=(_atom_text(goal),)),
        budget=BudgetSpec(unary_rounds=3, closure_rounds=2, closure_nodes=8),
    )


def test_a_unary_cycle_is_never_walked_twice() -> None:
    generator = _generator(_cyclic((_unary("D", "E"), _unary("E", "D")), ("D",), "E"))
    corpus = generator.sample(40, seed=1, length=UniformLength(1, 1))

    assert {
        _gold_shape(example.supervision.derivation) for example in corpus.examples
    } == {
        (
            (0, 1),
            _atom_text("E"),
            "D-to-E",
            (((0, 1), _atom_text("D"), None, ()),),
        )
    }


@pytest.mark.parametrize("arity", [1, 2])
def test_rule_weights_govern_each_application_in_a_draw(arity: int) -> None:
    declaration = _cyclic(
        (
            _named_rewrite("heavy", "D", "E", arity),
            _named_rewrite("light", "F", "E", arity),
        ),
        ("D", "F"),
        "E",
    )
    assembly = _assembly(declaration)
    length = UniformLength(arity, arity)

    def counts(weights: dict[str, float]) -> Counter[str]:
        weighting = WeightedRulesDerivationSamplerSpec(weights=weights).build(
            assembly.grammar.rules
        )
        corpus = SyntheticGenerator(
            assembly, policy=FOREST_POLICY, weighting=weighting
        ).sample(400, seed=9, length=length)

        return Counter(
            cast(
                "str",
                example.supervision.derivation.node(
                    example.supervision.derivation.root
                ).rule,
            )
            for example in corpus.examples
        )

    heavy = counts({"heavy": 9.0, "light": 1.0})
    light = counts({"heavy": 1.0, "light": 9.0})

    assert heavy["heavy"] > heavy["light"]
    assert light["light"] > light["heavy"]


def test_a_derivation_the_parser_refuses_is_reported_not_emitted() -> None:
    generator = _generator(
        _cyclic((_unary("D", "E"), _unary("E", "F"), _unary("F", "D")), ("D", "E"), "D")
    )

    with pytest.raises(UncertifiedDerivationError, match="rebuilds no derivation"):
        generator.sample(50, seed=1, length=UniformLength(1, 1))


def test_a_featured_goal_is_reached_under_feature_erasure() -> None:
    declaration = _declaration().with_(
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=("S", "NP", "conj"),
            operators=(
                OperatorSpec(name="S[dcl]", sort="Cat", erases_to="S"),
                OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),
            ),
            conjunctions=CONJUNCTIONS,
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="w0", category="NP"),
                LexicalEntrySpec(form="w1", category="S[dcl]\\NP"),
            )
        ),
        goal=RootGoalSpec(categories=("S[dcl]",)),
    )
    assembly = _assembly(declaration)
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        3, seed=0, length=UniformLength(1, 2)
    )

    assert {
        _gold_shape(example.supervision.derivation) for example in corpus.examples
    } == {
        (
            (0, 2),
            "S[dcl]",
            "application:\\",
            (((0, 1), "NP", None, ()), ((1, 2), "S[dcl]\\NP", None, ())),
        )
    }
    assert corpus.knobs.primitive_count == 3


def test_a_rule_admitting_several_categories_keeps_the_drawn_one() -> None:
    declaration = _declaration(
        vocabulary_size=1,
        unary_rounds=2,
        normal_form=NoNormalFormSpec(),
    ).with_(
        rules=RuleSetSpec(
            name="raised",
            rules=(
                ApplicationSpec(view="/"),
                TypeRaisingSpec(view="/", targets=("S", "NP"), arguments=None),
                PatternRuleSpec(
                    name="any-to-s",
                    premises=(TermPatternSpec(name="X", sort="Cat"),),
                    conclusion=ConstantPatternSpec(name="S"),
                    head_policy="pass-through",
                ),
            ),
        )
    )
    corpus = _generator(declaration).sample(60, seed=1, length=UniformLength(1, 1))
    below_root = Counter(
        example.supervision.derivation.node(1).label for example in corpus.examples
    )

    assert set(below_root) == {"NP", "S/(S\\NP)", "NP/(NP\\NP)"}
    assert min(below_root.values()) >= 10


def test_the_generator_exposes_its_grammar_and_licensing() -> None:
    assembly = _assembly(_declaration())
    generator = SyntheticGenerator(assembly, policy=FOREST_POLICY)

    assert generator.grammar is assembly.grammar
    assert generator.licensing == Licensing("CC0-1.0")


def _ambiguous(categories_per_form: int = 1, vocabulary_size: int = 5) -> GrammarDecl:
    """Declare the grammar with every combinator and no normal form."""
    return _declaration(
        vocabulary_size=vocabulary_size,
        combinators=("application", "composition", "type-raising"),
        categories_per_form=categories_per_form,
        normal_form=NoNormalFormSpec(),
    )


def _surface(item: Item) -> tuple[str, ...]:
    return tuple(token.surface for token in item.tokens)


def _enumerated(assembly: GrammarAssembly, tokens: tuple[str, ...]) -> int:
    """Count root derivations the way the chart enumerates them."""
    grammar = assembly.grammar
    chart = parse(
        tokens,
        rules=grammar.rules,
        source=assembly.lexicon,
        semiring=LogInside(),
        scorer=_UnitScorer(),
        goal=grammar.goal,
        unary_depth=grammar.budget.unary_rounds,
        policy=FOREST_POLICY,
    )

    return sum(1 for root in grammar.goal.roots(chart) for _ in chart.derivations(root))


def test_the_length_bounds_select_the_lengths_drawn() -> None:
    generator = _generator(_declaration(vocabulary_size=6))
    corpus = generator.sample(40, seed=2, length=UniformLength(3, 4))

    assert corpus.min_length == 3
    assert corpus.max_length == 4
    assert {len(example.item.tokens) for example in corpus.examples} == {3, 4}

    pinned = generator.sample(10, seed=2, length=UniformLength(4, 4))

    assert {len(example.item.tokens) for example in pinned.examples} == {4}


def test_bounds_the_goal_derives_nothing_within_are_reported_as_unproductive() -> None:
    generator = _generator(_declaration(vocabulary_size=6))

    with pytest.raises(UnproductiveGrammarError, match="at least 1 and at most 1"):
        generator.sample(1, seed=0, length=UniformLength(1, 1))

    with pytest.raises(UnproductiveGrammarError, match="at least 1 and at most 1"):
        generator.sample(1, seed=0, length=NegativeBinomialLength(2, 0.5, max_length=1))


def test_the_length_report_measures_the_items_it_ships_with() -> None:
    distribution = UniformLength(2, 5)
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        60, seed=3, length=distribution
    )
    report = corpus.length

    assert report.distribution is distribution
    assert report.resampled == 0
    assert report.achieved.total == len(corpus.examples) == 60
    assert report.achieved == Histogram.of(
        len(example.item.tokens) for example in corpus.examples
    )
    assert report == LengthReport(
        report.achieved, resampled=0, distribution=distribution
    )


def test_a_uniform_length_is_flat_over_the_bounds() -> None:
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        1200, seed=4, length=UniformLength(2, 7)
    )
    achieved = corpus.length.achieved
    expected = achieved.total / 6

    assert set(achieved.counts) == {2, 3, 4, 5, 6, 7}
    assert corpus.length.resampled == 0

    for length in range(2, 8):
        assert abs(achieved[length] - expected) <= 0.25 * expected


def test_a_length_the_grammar_cannot_realize_is_resampled_and_counted() -> None:
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        600, seed=5, length=UniformLength(1, 4)
    )
    report = corpus.length
    drawn = report.achieved.total + report.resampled

    assert set(report.achieved.counts) == {2, 3, 4}
    assert report.resampled > 0
    assert abs(report.resampled / drawn - 0.25) <= 0.05

    for length in (2, 3, 4):
        assert abs(report.achieved[length] / drawn - 0.25) <= 0.05


@pytest.mark.parametrize("seed", [1, 2])
@pytest.mark.parametrize(
    "distribution",
    [
        NegativeBinomialLength(3, 0.5, min_length=2, max_length=16),
        NegativeBinomialLength(1.5, 0.35, min_length=2, max_length=20),
        NegativeBinomialLength(8, 0.7, min_length=3, max_length=16),
    ],
    ids=repr,
)
def test_a_negative_binomial_length_achieves_its_mean_and_variance(
    distribution: NegativeBinomialLength, seed: int
) -> None:
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        1000, seed=seed, length=distribution
    )
    achieved = corpus.length.achieved
    mean = Target(distribution.mean, 0.3)
    variance = Target(distribution.variance, 0.2 * distribution.variance)

    assert achieved.total == 1000
    assert min(achieved.counts) >= distribution.min_length
    assert max(achieved.counts) <= distribution.max_length
    assert mean.admits(achieved.mean), (achieved.mean, mean)
    assert variance.admits(achieved.variance), (achieved.variance, variance)
    assert corpus.length.resampled <= 0.01 * achieved.total


def test_a_negative_binomial_draw_beyond_the_bound_is_resampled_and_counted() -> None:
    distribution = NegativeBinomialLength(3, 0.5, min_length=2, max_length=4)
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        400, seed=6, length=distribution
    )
    report = corpus.length
    beyond = 1 - 0.5**3 * (1 + 3 * 0.5 + 6 * 0.5**2)

    assert set(report.achieved.counts) == {2, 3, 4}
    assert report.resampled > 0
    assert abs(report.resampled / (400 + report.resampled) - beyond) <= 0.05
    assert all(len(example.item.tokens) <= 4 for example in corpus.examples)


def test_a_negative_binomial_draw_beyond_the_bound_can_be_clipped() -> None:
    distribution = NegativeBinomialLength(
        3,
        0.5,
        min_length=2,
        max_length=4,
        out_of_support="clip",
    )
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        400, seed=6, length=distribution
    )

    assert set(corpus.length.achieved.counts) == {2, 3, 4}
    assert corpus.length.resampled == 0


def test_a_negative_binomial_draw_beyond_the_bound_can_be_an_error() -> None:
    distribution = NegativeBinomialLength(
        3,
        0.5,
        min_length=2,
        max_length=2,
        out_of_support="error",
    )

    with pytest.raises(LengthOutOfSupportError, match="outside declared support"):
        _generator(_declaration(vocabulary_size=6)).sample(
            400, seed=6, length=distribution
        )


def test_a_tabular_length_distribution_is_drawn_directly() -> None:
    corpus = _generator(_declaration(vocabulary_size=6)).sample(
        1000,
        seed=8,
        length=TabularLength({2: 1.0, 4: 3.0}),
    )
    achieved = corpus.length.achieved

    assert set(achieved.counts) == {2, 4}
    assert achieved[2] / achieved.total == pytest.approx(0.25, abs=0.05)
    assert achieved[4] / achieved.total == pytest.approx(0.75, abs=0.05)
    assert corpus.length.resampled == 0


@pytest.mark.parametrize("field", ["right_branching", "slash_balance"])
def test_shape_target_sweeps_move_the_achieved_share_monotonically(
    field: str,
) -> None:
    generator = _generator(_declaration(vocabulary_size=10))
    achieved: list[float] = []

    for value in (0.0, 1 / 3, 2 / 3):
        target = ShapeTarget(
            right_branching=Target(value, 1e-12)
            if field == "right_branching"
            else None,
            slash_balance=Target(value, 1e-12) if field == "slash_balance" else None,
            attempts=100,
        )
        corpus = generator.sample(
            40,
            seed=12,
            length=UniformLength(4, 4),
            shape=ShapeSchedule(target, {}),
        )
        report = corpus.shape

        assert report is not None
        achieved.append(float(getattr(report, field)))
        assert corpus.length.achieved == Histogram({4: 40})
        assert (
            corpus.inventory.size
            == generator.sample(
                0,
                seed=12,
                length=UniformLength(4, 4),
                shape=ShapeSchedule(target, {}),
            ).inventory.size
        )

    assert achieved == sorted(achieved)
    assert achieved == pytest.approx([0.0, 1 / 3, 2 / 3])


def test_a_shape_target_reports_depth_and_arity() -> None:
    target = ShapeTarget(
        depth=Target(4.0, 1.0),
        arity=Target(2.0, 0.0),
        attempts=10,
    )
    corpus = _generator(_declaration(vocabulary_size=8)).sample(
        30,
        seed=9,
        length=UniformLength(3, 4),
        shape=ShapeSchedule(target, {}),
    )

    assert corpus.shape is not None
    assert corpus.shape.target == ShapeSchedule(target, {})
    assert corpus.shape.depths.total == 30
    assert corpus.shape.arity == 2.0
    assert corpus.shape.rejected == 0


def test_a_shape_target_refuses_an_unreachable_band_with_its_budget() -> None:
    target = ShapeTarget(arity=Target(1.0, 0.0), attempts=3)

    with pytest.raises(ShapeUnreachableError, match="3 draws"):
        _generator(_declaration(vocabulary_size=8)).sample(
            1,
            seed=9,
            length=UniformLength(3, 4),
            shape=ShapeSchedule(target, {}),
        )


def test_a_shape_schedule_draws_training_shallower_than_validation() -> None:
    schedule = ShapeSchedule(
        None,
        {
            "train": ShapeTarget(depth=Target(2.0, 0.0), attempts=100),
            "validation": ShapeTarget(depth=Target(3.0, 0.0), attempts=100),
        },
    )
    splits = ("train",) * 6 + ("validation",) * 6
    corpus = _generator(_declaration(vocabulary_size=8)).sample(
        len(splits),
        seed=9,
        splits=splits,
        length=UniformLength(2, 3),
        shape=schedule,
    )

    assert corpus.shape is not None
    by_split: dict[str, set[int]] = {}
    for example, measurement in zip(
        corpus.examples, corpus.shape.measurements, strict=True
    ):
        assert example.item.split is not None
        by_split.setdefault(example.item.split, set()).add(measurement.depth)
    assert by_split == {"train": {2}, "validation": {3}}


def test_corruption_sources_draw_lengths_without_entering_the_histogram() -> None:
    generator = _generator(_declaration(vocabulary_size=6))
    corpus = generator.sample(
        200,
        seed=7,
        length=UniformLength(1, 4),
        corruption=_corruption(generator, 4, 0.5, operators=(_swap,), attempts=100),
    )
    report = corpus.length

    assert len(corpus.corrupted) == 100
    assert report.achieved.total == len(corpus.examples) == 100
    assert report.resampled > 0
    assert report.achieved == Histogram.of(
        len(example.item.tokens) for example in corpus.examples
    )


def test_the_same_seed_repeats_a_negative_binomial_sample() -> None:
    generator = _generator(_declaration(vocabulary_size=6))
    distribution = NegativeBinomialLength(2, 0.4, min_length=2, max_length=12)
    first = generator.sample(50, seed=8, length=distribution)
    again = generator.sample(50, seed=8, length=distribution)
    other = generator.sample(50, seed=9, length=distribution)

    assert first == again
    assert first.length == again.length
    assert first.length != other.length


def test_derivation_counts_agree_with_the_charts_own_enumeration() -> None:
    assembly = _assembly(
        _ambiguous(categories_per_form=2).with_(budget=BudgetSpec(unary_rounds=2))
    )
    generator = SyntheticGenerator(assembly, policy=FOREST_POLICY)
    corpus = generator.sample(150, seed=4, length=UniformLength(1, 5))
    counts = Counter(
        generator.derivation_count(_surface(example.item))
        for example in corpus.examples
    )

    assert len(counts) > 2

    for example in corpus.examples:
        tokens = _surface(example.item)

        assert generator.derivation_count(tokens) == _enumerated(assembly, tokens)

    assert generator.derivation_count(("w1", "w1")) == 0


def test_unary_chains_are_counted_the_way_the_chart_enumerates_them() -> None:
    chained = _assembly(_chained(2, forms=("D", "E")))
    cyclic = _assembly(_cyclic((_unary("D", "E"), _unary("E", "D")), ("D", "E"), "E"))

    for assembly, expected in ((chained, 2), (cyclic, 2)):
        generator = SyntheticGenerator(assembly, policy=FOREST_POLICY)

        assert generator.derivation_count(("x",)) == expected
        assert generator.derivation_count(("x",)) == _enumerated(assembly, ("x",))


def test_the_ambiguity_report_measures_the_sample_it_ships_with() -> None:
    generator = _generator(_ambiguous(categories_per_form=2))
    corpus = generator.sample(80, seed=5, length=UniformLength(1, 5))
    report = corpus.ambiguity

    assert report.target is None
    assert report.rejected == 0
    assert report.derivations.total == len(corpus.examples) == 80
    assert report.lexical.total == sum(
        len(example.item.tokens) for example in corpus.examples
    )
    assert report.lexical == Histogram({2: report.lexical.total})
    assert report.derivations == Histogram.of(
        generator.derivation_count(_surface(example.item))
        for example in corpus.examples
    )
    assert report.derivations.mean > 1


@pytest.mark.parametrize("seed", [11, 12, 13, 14, 15])
def test_a_derivation_target_is_hit_within_tolerance_on_every_seed(seed: int) -> None:
    generator = _generator(_ambiguous())
    exact = AmbiguityTarget(derivations=Target(3, 0), attempts=400)
    corpus = generator.sample(
        40, seed=seed, length=UniformLength(1, 5), ambiguity=exact
    )

    assert corpus.ambiguity.target is exact
    assert corpus.ambiguity.derivations == Histogram({3: 40})
    assert corpus.ambiguity.rejected > 0

    banded = AmbiguityTarget(derivations=Target(3, 1), attempts=400)
    corpus = generator.sample(
        40, seed=seed, length=UniformLength(1, 5), ambiguity=banded
    )
    achieved = corpus.ambiguity.derivations

    assert achieved.total == 40
    assert set(achieved.counts) <= {2, 3, 4}
    assert len(achieved.counts) > 1
    assert abs(achieved.mean - 3) <= 1

    for example in corpus.examples:
        assert 2 <= generator.derivation_count(_surface(example.item)) <= 4


def test_three_derivations_unconstrained_collapse_to_one_under_the_normal_form() -> (
    None
):
    declaration = _ambiguous()
    free = _generator(declaration)
    normalized = _generator(
        declaration.with_(normal_form=HockenmaierBiskNormalFormSpec())
    )
    corpus = free.sample(
        30,
        seed=8,
        length=UniformLength(1, 5),
        ambiguity=AmbiguityTarget(derivations=Target(3, 0), attempts=400),
    )
    strings = {_surface(example.item) for example in corpus.examples}

    assert len(strings) > 1

    for tokens in strings:
        assert free.derivation_count(tokens) == 3
        assert normalized.derivation_count(tokens) == 1


def test_a_lexical_target_the_lexicon_cannot_meet_is_refused_naming_the_mean() -> None:
    generator = _generator(_ambiguous(categories_per_form=2))
    met = AmbiguityTarget(lexical=Target(2, 0.5), attempts=1)
    corpus = generator.sample(20, seed=1, length=UniformLength(1, 4), ambiguity=met)

    assert corpus.ambiguity.lexical == Histogram({2: corpus.ambiguity.lexical.total})
    assert corpus.ambiguity.lexical.mean == 2

    with pytest.raises(
        AmbiguityUnreachableError,
        match=r"1 outside the lexical target Target\(1\.0, 0\.5\) at a mean of 2 categ",
    ) as caught:
        generator.sample(
            20,
            seed=1,
            length=UniformLength(1, 4),
            ambiguity=AmbiguityTarget(lexical=Target(1, 0.5), attempts=1),
        )

    assert isinstance(caught.value, SamplingExhaustedError)
    assert (
        generator.sample(
            20,
            seed=1,
            length=UniformLength(1, 4),
            ambiguity=AmbiguityTarget(lexical=Target(1.5, 0.5), attempts=1),
        ).ambiguity.lexical.mean
        == 2
    )


def test_an_unreachable_derivation_target_exhausts_its_budget() -> None:
    generator = _generator(_ambiguous())

    with pytest.raises(AmbiguityUnreachableError, match="5 draws") as caught:
        generator.sample(
            3,
            seed=0,
            length=UniformLength(1, 4),
            ambiguity=AmbiguityTarget(derivations=Target(100, 0), attempts=5),
        )

    assert "5 outside the derivation target Target(100.0, 0.0)" in str(caught.value)


def test_an_exhausted_budget_names_what_each_band_refused() -> None:
    generator = _generator(_ambiguous(categories_per_form=2))
    target = AmbiguityTarget(
        lexical=Target(2, 0), derivations=Target(100, 0), attempts=6
    )

    with pytest.raises(AmbiguityUnreachableError) as caught:
        generator.sample(3, seed=0, length=UniformLength(1, 4), ambiguity=target)

    message = str(caught.value)

    assert message.startswith("6 draws all fell outside the ambiguity target: ")
    assert "lexical" not in message
    assert "6 outside the derivation target Target(100.0, 0.0)" in message

    with pytest.raises(AmbiguityUnreachableError) as caught:
        generator.sample(
            3,
            seed=0,
            length=UniformLength(1, 4),
            ambiguity=AmbiguityTarget(
                lexical=Target(1, 0), derivations=Target(100, 0), attempts=6
            ),
        )

    assert "6 outside the lexical target" in str(caught.value)
    assert "derivation" not in str(caught.value)


def test_corruption_emits_the_requested_share_and_every_item_is_recheckable() -> None:
    assembly = _assembly(_ambiguous(categories_per_form=2))
    generator = SyntheticGenerator(assembly, policy=FOREST_POLICY)
    request = _corruption(generator, 5, 0.4, attempts=100)
    corpus = generator.sample(
        500, seed=21, length=UniformLength(1, 5), corruption=request
    )
    report = corpus.corruption
    digest = assembly.declaration_digest[:12]

    assert report is not None
    assert len(corpus) == 500
    assert len(corpus.corrupted) == 200
    assert len(corpus.examples) == 300
    assert report.requested is request
    assert report.sample_size == 500
    assert report.corrupted == 200
    assert report.achieved_proportion == 0.4
    assert all(count > 0 for count in report.accepted.values())
    assert corpus.ambiguity.derivations.total == 300
    assert {example.item.item_id for example in corpus.examples} | {
        record.item.item_id for record in corpus.corrupted
    } == {f"{digest}:21:{index}" for index in range(500)}

    for record in corpus.corrupted:
        tokens = _surface(record.item)

        assert 1 <= len(tokens) <= 5
        assert generator.derivation_count(tokens) == 0
        assert generator.certify(tokens) == record.certificate
        assert record.certificate == NonDerivability(
            assembly.declaration_digest,
            assembly.grammar.theory_digest,
            unary_rounds=1,
        )
        assert generator.derivation_count(record.source) > 0
        assert generator.certify(record.source) is None
        assert record.item.text == " ".join(tokens)
        assert record.kind in request.operator_names


def test_corrupted_slots_are_drawn_among_the_indices_by_the_seed() -> None:
    generator = _generator(_declaration())
    request = _corruption(generator, 4, 0.25, attempts=50)

    def slots(seed: int) -> set[int]:
        corpus = generator.sample(
            40, seed=seed, length=UniformLength(1, 4), corruption=request
        )

        return {
            int(record.item.item_id.rpartition(":")[2]) for record in corpus.corrupted
        }

    first = slots(4)

    assert len(first) == 10
    assert first < set(range(40))
    assert first != set(range(10))
    assert first != slots(5)


def test_corruption_sources_are_drawn_within_the_derivation_band() -> None:
    generator = _generator(_ambiguous())
    exact = AmbiguityTarget(derivations=Target(3, 0), attempts=400)
    corpus = generator.sample(
        20,
        seed=16,
        length=UniformLength(1, 5),
        ambiguity=exact,
        corruption=_corruption(generator, 5, 0.5, attempts=100),
    )

    assert len(corpus.corrupted) == 10
    assert corpus.ambiguity.derivations == Histogram({3: 10})
    assert corpus.ambiguity.rejected > 0

    for record in corpus.corrupted:
        assert generator.derivation_count(record.source) == 3


@pytest.mark.parametrize(
    ("proportion", "count", "corrupted"),
    [(0.5, 5, 3), (0.3, 9, 3), (0.3, 11, 3), (0.25, 6, 2), (0.29, 100, 29)],
)
def test_the_corrupted_count_is_the_share_rounded_with_halves_up(
    proportion: float, count: int, corrupted: int
) -> None:
    generator = _generator(_declaration())
    corpus = generator.sample(
        count,
        seed=2,
        length=UniformLength(1, 4),
        corruption=_corruption(generator, 4, proportion, attempts=50),
    )

    assert len(corpus.corrupted) == corrupted
    assert len(corpus.examples) == count - corrupted


def test_a_corrupted_string_differs_from_its_source_by_its_operator() -> None:
    generator = _generator(_ambiguous(categories_per_form=2))
    corpus = generator.sample(
        120,
        seed=3,
        length=UniformLength(1, 4),
        corruption=_corruption(generator, 4, 0.5, attempts=100),
    )

    for record in corpus.corrupted:
        tokens = _surface(record.item)
        source = record.source

        if record.kind == "swap":
            assert len(tokens) == len(source)
            assert sorted(tokens) == sorted(source)
            assert sum(a != b for a, b in zip(tokens, source, strict=True)) == 2
        elif record.kind == "deletion":
            assert len(tokens) == len(source) - 1
        elif record.kind == "insertion":
            assert len(tokens) == len(source) + 1
        else:
            assert len(tokens) == len(source)
            assert sum(a != b for a, b in zip(tokens, source, strict=True)) == 1


def test_derivable_candidates_are_rejected_and_the_rate_is_reported() -> None:
    generator = _generator(_ambiguous(categories_per_form=3, vocabulary_size=6))
    corpus = generator.sample(
        200,
        seed=9,
        length=UniformLength(1, 5),
        corruption=_corruption(generator, 5, 0.5, attempts=200),
    )
    report = corpus.corruption

    assert report is not None
    assert sum(report.rejected.values()) > 0
    assert report.rejected["swap"] > 0
    assert 0 < report.rejection_rate < 1
    assert report.rejection_rate == sum(report.rejected.values()) / (
        sum(report.rejected.values()) + 100
    )

    for record in corpus.corrupted:
        assert generator.derivation_count(_surface(record.item)) == 0


def test_an_operator_the_string_never_admits_is_counted_as_inapplicable() -> None:
    generator = _generator(_declaration(vocabulary_size=4))
    corpus = generator.sample(
        20,
        seed=0,
        length=UniformLength(1, 2),
        corruption=_corruption(
            generator,
            2,
            0.5,
            operators=(_insertion, _deletion),
            attempts=50,
        ),
    )
    report = corpus.corruption

    assert report is not None
    assert report.inapplicable["insertion"] > 0
    assert report.accepted["insertion"] == 0
    assert report.accepted["deletion"] == 10
    assert all(len(record.item.tokens) == 1 for record in corpus.corrupted)


def _merging() -> GrammarDecl:
    """Declare a grammar deriving every string over its two forms."""
    atom = ConstantPatternSpec(name="NP")

    return GrammarDecl(
        name="merging",
        signature=SignatureSpec(
            name="merging", primitives=("NP",), syntax="s-expression"
        ),
        rules=RuleSetSpec(
            name="merge",
            rules=(
                PatternRuleSpec(name="merge", premises=(atom, atom), conclusion=atom),
            ),
        ),
        lexicon=InlineLexiconSpec(
            entries=(
                LexicalEntrySpec(form="x", category=_atom_text("NP")),
                LexicalEntrySpec(form="y", category=_atom_text("NP")),
            )
        ),
        goal=RootGoalSpec(categories=(_atom_text("NP"),)),
        budget=BudgetSpec(unary_rounds=1, closure_rounds=2, closure_nodes=8),
    )


def test_corruption_exhausts_its_budget_when_no_candidate_is_non_derivable() -> None:
    generator = _generator(_merging())

    with pytest.raises(CorruptionExhaustedError, match="4 candidates") as caught:
        generator.sample(
            4,
            seed=0,
            length=UniformLength(2, 3),
            corruption=_corruption(
                generator,
                3,
                1,
                operators=(_swap, _deletion),
                attempts=4,
            ),
        )

    assert isinstance(caught.value, SamplingExhaustedError)
    assert "('swap', 'deletion')" in str(caught.value)

    with pytest.raises(CorruptionExhaustedError, match="inapplicable"):
        generator.sample(
            4,
            seed=0,
            length=UniformLength(1, 1),
            corruption=_corruption(
                generator, 1, 1, operators=(_substitution,), attempts=4
            ),
        )


def test_a_zero_proportion_corrupts_nothing_but_still_reports() -> None:
    generator = _generator(_declaration())
    corpus = generator.sample(
        6,
        seed=1,
        length=UniformLength(1, 3),
        corruption=_corruption(generator, 3, 0, attempts=1),
    )

    assert corpus.corrupted == ()
    assert len(corpus.examples) == 6
    assert corpus.corruption is not None
    assert corpus.corruption.corrupted == 0
    assert corpus.corruption.achieved_proportion == 0.0
    assert generator.sample(0, seed=1, length=UniformLength(1, 3)).corruption is None


def test_the_same_seed_repeats_a_corrupted_and_targeted_sample() -> None:
    generator = _generator(_ambiguous(categories_per_form=2))
    ambiguity = AmbiguityTarget(derivations=Target(2, 1), attempts=50)
    corruption = _corruption(generator, 5, 0.3, attempts=50)

    def sample(seed: int) -> SyntheticCorpus:
        return generator.sample(
            30,
            seed=seed,
            length=UniformLength(1, 5),
            ambiguity=ambiguity,
            corruption=corruption,
        )

    first = sample(6)
    again = sample(6)
    other = sample(7)

    assert first == again
    assert first.corrupted == again.corrupted
    assert first.corruption == again.corruption
    assert [_surface(record.item) for record in first.corrupted] != [
        _surface(record.item) for record in other.corrupted
    ]


NOISED_COUNT: Final = 400
NOISE_RATES: Final = (0.0, 0.05, 0.1, 0.2)
PROPORTIONS: Final = (0.0, 0.1, 0.25, 0.5)


def _noised(
    seed: int, *, proportion: float = 0.25, rate: float = 0.2, count: int = NOISED_COUNT
) -> SyntheticCorpus:
    """Draw a corpus with corrupted items and flipped labels from a small grammar."""
    generator = _generator(_declaration(vocabulary_size=6))
    return generator.sample(
        count,
        seed=seed,
        length=UniformLength(1, 4),
        corruption=_corruption(generator, 4, proportion, attempts=100),
        noise=LabelNoise(grammatical=rate, ungrammatical=rate),
    )


def _binomial_tolerance(rate: float, count: int) -> float:
    """Return four standard errors of a share of a count drawn at a rate."""
    return 4 * sqrt(rate * (1 - rate) / count)


def _truths(corpus: SyntheticCorpus) -> dict[str, str]:
    """Return each item's grammaticality by which tuple of the corpus holds it."""
    return {
        **{example.item.item_id: "grammatical" for example in corpus.examples},
        **{record.item.item_id: "ungrammatical" for record in corpus.corrupted},
    }


@pytest.mark.parametrize("seed", [31, 32, 33, 34, 35])
def test_label_noise_flips_at_the_requested_rate_and_every_record_agrees(
    seed: int,
) -> None:
    corpus = _noised(seed)
    report = corpus.noise
    truths = _truths(corpus)

    assert report is not None
    assert len(corpus.labels) == len(corpus) == NOISED_COUNT
    assert [label.item_id for label in corpus.labels] == [
        f"{corpus.declaration_digest[:12]}:{seed}:{index}"
        for index in range(NOISED_COUNT)
    ]

    for label in corpus.labels:
        assert label.true_label == truths[label.item_id]
        assert label.flipped == (label.observed_label != label.true_label)
        assert {label.true_label, label.observed_label} <= set(GRAMMATICALITY_LABELS)

    flips = sum(1 for label in corpus.labels if label.flipped)

    assert flips > 0
    assert abs(flips / NOISED_COUNT - 0.2) <= _binomial_tolerance(0.2, NOISED_COUNT)
    assert report.requested == LabelNoise(grammatical=0.2, ungrammatical=0.2)
    assert report.flip_count == flips
    assert report.achieved_rate == flips / NOISED_COUNT
    assert report == NoiseReport.of(
        LabelNoise(grammatical=0.2, ungrammatical=0.2), corpus.labels
    )
    assert report.flipped["grammatical"] > 0
    assert report.flipped["ungrammatical"] > 0
    assert report.items["ungrammatical"] == len(corpus.corrupted) == 100


def test_the_items_drawn_at_a_seed_are_the_same_at_every_noise_rate() -> None:
    generator = _generator(_declaration(vocabulary_size=6))

    def sample(noise: LabelNoise | None) -> SyntheticCorpus:
        return generator.sample(
            60,
            seed=3,
            length=UniformLength(1, 4),
            corruption=_corruption(generator, 4, 0.25, attempts=100),
            noise=noise,
        )

    clean = sample(None)
    zero = sample(LabelNoise(grammatical=0.0, ungrammatical=0.0))
    noisy = sample(LabelNoise(grammatical=0.3, ungrammatical=0.3))

    assert clean.examples == zero.examples == noisy.examples
    assert clean.corrupted == zero.corrupted == noisy.corrupted
    assert clean.corruption == zero.corruption == noisy.corruption
    assert clean.noise is None
    assert clean.labels == zero.labels
    assert all(not label.flipped for label in clean.labels)
    assert all(label.observed_label == label.true_label for label in clean.labels)
    assert zero.noise == NoiseReport(
        LabelNoise(grammatical=0.0, ungrammatical=0.0),
        items={"grammatical": 45, "ungrammatical": 15},
        flipped={},
    )
    assert clean != zero
    assert noisy.noise is not None
    assert noisy.noise.flip_count > 0
    assert [label.true_label for label in noisy.labels] == [
        label.true_label for label in clean.labels
    ]
    assert noisy.labels != clean.labels
    assert sample(LabelNoise(grammatical=0.3, ungrammatical=0.3)) == noisy
    assert (
        generator.sample(
            60,
            seed=4,
            length=UniformLength(1, 4),
            corruption=_corruption(generator, 4, 0.25, attempts=100),
            noise=LabelNoise(grammatical=0.3, ungrammatical=0.3),
        ).labels
        != noisy.labels
    )


@pytest.mark.parametrize("rate", [0.1, 0.3, 1.0])
def test_contamination_flips_only_corrupted_items_at_the_declared_rate(
    rate: float,
) -> None:
    generator = _generator(_declaration(vocabulary_size=6))
    corpus = generator.sample(
        NOISED_COUNT,
        seed=17,
        length=UniformLength(1, 4),
        corruption=_corruption(generator, 4, 0.5, attempts=100),
        noise=LabelNoise(grammatical=0.0, ungrammatical=rate),
    )
    report = corpus.noise
    negatives = len(corpus.corrupted)
    flipped = [label for label in corpus.labels if label.flipped]

    assert report is not None
    assert negatives == NOISED_COUNT // 2
    assert flipped
    assert {label.true_label for label in flipped} == {"ungrammatical"}
    assert report.flipped["grammatical"] == 0
    assert abs(len(flipped) / negatives - rate) <= _binomial_tolerance(rate, negatives)
    assert report.achieved_rates["ungrammatical"] == len(flipped) / negatives


def test_the_flip_rate_of_each_true_label_is_its_own() -> None:
    def sample(grammatical: float, ungrammatical: float) -> SyntheticCorpus:
        generator = _generator(_declaration(vocabulary_size=6))
        return generator.sample(
            NOISED_COUNT,
            seed=19,
            length=UniformLength(1, 4),
            corruption=_corruption(generator, 4, 0.5, attempts=100),
            noise=LabelNoise(grammatical=grammatical, ungrammatical=ungrammatical),
        )

    positives = sample(0.4, 0.0)
    both = sample(0.4, 0.1)

    assert positives.noise is not None
    assert both.noise is not None
    assert positives.noise.flipped["ungrammatical"] == 0
    assert positives.noise.flipped["grammatical"] > 0
    assert both.noise.flipped["grammatical"] == positives.noise.flipped["grammatical"]
    assert 0 < both.noise.flipped["ungrammatical"] < both.noise.flipped["grammatical"]
    assert both.examples == positives.examples
    assert both.corrupted == positives.corrupted


def test_a_full_label_flip_rate_inverts_every_label() -> None:
    corpus = _noised(2, rate=1.0, count=40)

    assert corpus.noise is not None
    assert corpus.noise.flip_count == 40
    assert corpus.noise.achieved_rate == 1.0
    assert all(label.flipped for label in corpus.labels)
    assert dict(corpus.noise.observed) == {"grammatical": 10, "ungrammatical": 30}


@pytest.mark.parametrize("proportion", PROPORTIONS)
@pytest.mark.parametrize("rate", NOISE_RATES)
def test_the_design_cell_is_recovered_from_the_emitted_labels_alone(
    proportion: float, rate: float
) -> None:
    corpus = _noised(41, proportion=proportion, rate=rate)
    labels = corpus.labels
    count = NOISED_COUNT
    ungrammatical = sum(1 for label in labels if label.true_label == "ungrammatical")
    flips = sum(1 for label in labels if label.flipped)
    observed = sum(1 for label in labels if label.observed_label == "ungrammatical")
    expected = proportion * (1 - rate) + (1 - proportion) * rate

    assert ungrammatical == int(proportion * count + 0.5) == len(corpus.corrupted)
    assert abs(flips / count - rate) <= _binomial_tolerance(rate, count)
    assert abs(observed / count - expected) <= _binomial_tolerance(expected, count)
    report = corpus.noise

    assert report is not None
    assert report == NoiseReport.of(
        LabelNoise(grammatical=rate, ungrammatical=rate), labels
    )
    assert report.observed["ungrammatical"] == observed
    assert report.items["ungrammatical"] == ungrammatical
    assert report.flip_count == flips
    assert (flips == 0) == (rate == 0.0)


def test_the_observed_projection_pairs_each_item_with_its_observed_label() -> None:
    corpus = _noised(8, count=40)
    observed = corpus.observed()
    items = {
        **{example.item.item_id: example.item for example in corpus.examples},
        **{record.item.item_id: record.item for record in corpus.corrupted},
    }

    assert len(observed) == 40
    assert [example.item for example in observed] == [
        items[label.item_id] for label in corpus.labels
    ]
    assert [example.supervision.label for example in observed] == [
        label.observed_label for label in corpus.labels
    ]
    assert {example.supervision.label_set for example in observed} == {
        frozenset(GRAMMATICALITY_LABELS)
    }
    assert any(
        example.supervision.label != label.true_label
        for example, label in zip(observed, corpus.labels, strict=True)
    )


def test_no_featurizer_reads_the_true_label(monkeypatch: pytest.MonkeyPatch) -> None:
    generator = _generator(_declaration(vocabulary_size=6))
    corpus = generator.sample(
        40,
        seed=8,
        length=UniformLength(1, 4),
        corruption=_corruption(generator, 4, 0.25, attempts=100),
        noise=LabelNoise(grammatical=0.2, ungrammatical=0.2),
    )
    reads = 0
    original = ObservedLabel.true_label

    def spied(label: ObservedLabel) -> str:
        nonlocal reads
        reads += 1

        return original.fget(label)

    monkeypatch.setattr(ObservedLabel, "true_label", property(spied))
    padding = "<pad>"
    featurize = UnlabeledFeaturizer(
        ItemHasher(
            Inventory("words", generator.lexicon.forms, (padding,)),
            Inventory("categories", ("S",)),
        ),
        padding,
        TorchCodeTensorizer(),
    )
    batch = featurize(
        tuple(Example(example.item, Unlabeled()) for example in corpus.observed())
    )

    assert len(batch) == 40
    assert reads == 0
    assert corpus.labels[0].true_label in GRAMMATICALITY_LABELS
    assert reads == 1


PAIR_CATEGORIES: Final = tuple("ABCDEFGH")
PAIR_COUNT: Final = 1500


def _pair(
    lexicon: LexiconSpec, categories: tuple[str, ...] = PAIR_CATEGORIES
) -> GrammarDecl:
    """Declare the grammar in which any two categories make a sentence."""
    return GrammarDecl(
        name="pair",
        signature=SignatureSpec(
            name="pair", primitives=("S", *categories), syntax="s-expression"
        ),
        rules=RuleSetSpec(
            name="pair",
            rules=(
                PatternRuleSpec(
                    name="pair",
                    premises=(
                        TermPatternSpec(name="X", sort="Cat"),
                        TermPatternSpec(name="Y", sort="Cat"),
                    ),
                    conclusion=ConstantPatternSpec(name="S"),
                ),
            ),
        ),
        lexicon=lexicon,
        goal=RootGoalSpec(categories=(_atom_text("S"),)),
        budget=BudgetSpec(unary_rounds=0, closure_rounds=1, closure_nodes=32),
    )


def _structure(
    placement: float = 0.0,
    *,
    categories: tuple[str, ...] = PAIR_CATEGORIES,
    forms: int = 200,
    mean: float = 2.0,
    dispersion: float = 0.5,
    rank_law: RankLawSpec | None = None,
    seed: int = 5,
) -> LexiconStructure:
    return LexiconStructure(
        forms=forms,
        joint=ParametricJointSpec(
            categories=tuple(_atom_text(category) for category in categories),
            category_shares=PowerLawRankLawSpec(exponent=1.0),
        ),
        ambiguity=ParametricAmbiguityLawSpec(
            mean=mean,
            dispersion=dispersion,
            maximum=min(4, len(categories)),
            placement=placement,
        ),
        rank_law=(PowerLawRankLawSpec(exponent=1.0) if rank_law is None else rank_law),
        seed=seed,
    )


@cache
def _structured(placement: float) -> SyntheticCorpus:
    """Draw the pair corpus at a placement, every marginal held fixed."""
    return _generator(_pair(_structure(placement))).sample(
        PAIR_COUNT, seed=1, length=UniformLength(2, 2)
    )


def test_the_structured_lexicon_reports_size_ambiguity_correlation_and_diversity() -> (
    None
):
    corpus = _structured(0.5)
    report = corpus.lexicon
    generator = _generator(_pair(_structure(0.5)))

    assert corpus.knobs == GrammarKnobs(9, (), ("pair",))
    assert report.realized_size == len(generator.lexicon.forms) == 200
    assert report.per_type_ambiguity.total == 200
    assert report.per_type_ambiguity.mean == pytest.approx(2.0)
    assert report.ambiguity_frequency_correlation is not None
    assert 0.0 < report.type_token_ratio <= 1.0
    assert sum(report.token_counts.values()) == 2 * PAIR_COUNT
    assert sum(report.token_shares.values()) == pytest.approx(1.0)
    assert corpus.ambiguity.lexical.total == 2 * PAIR_COUNT


def test_the_report_measures_ambiguity_placement_monotonically() -> None:
    achieved = [
        _structured(placement).lexicon.ambiguity_frequency_correlation
        for placement in (-0.5, 0.0, 0.5)
    ]

    negative, independent, positive = achieved

    assert negative is not None
    assert independent is not None
    assert positive is not None
    assert negative < independent < positive


@pytest.mark.parametrize("seed", [1, 2, 3])
def test_a_global_power_law_gives_the_corpus_that_rank_frequency_curve(
    seed: int,
) -> None:
    structure = _structure(
        categories=("A",),
        forms=40,
        mean=1.0,
        dispersion=0.0,
        rank_law=PowerLawRankLawSpec(exponent=1.0),
    )
    corpus = _generator(_pair(structure, ("A",))).sample(
        500, seed=seed, length=UniformLength(2, 2)
    )
    report = corpus.lexicon

    assert report.class_sizes == {_atom_text("A"): 40}
    assert report.rank_frequency_exponent is not None
    assert Target(1.0, 0.15).admits(report.rank_frequency_exponent)

    flat = _generator(
        _pair(structure.with_(rank_law=UniformRankLawSpec()), ("A",))
    ).sample(500, seed=seed, length=UniformLength(2, 2))

    assert flat.lexicon.rank_frequency_exponent is not None
    assert flat.lexicon.rank_frequency_exponent < report.rank_frequency_exponent


def test_changing_only_placement_changes_only_the_achieved_correlation() -> None:
    base = _pair(_structure(0.0))
    varied = _pair(_structure(1.0))
    base_document = loads(canonical_bytes(base))
    varied_document = loads(canonical_bytes(varied))
    base_ambiguity = base_document["lexicon"]["ambiguity"]
    varied_ambiguity = varied_document["lexicon"]["ambiguity"]

    assert {
        key for key in base_document if base_document[key] != varied_document[key]
    } == {"lexicon"}
    assert {
        key for key in base_ambiguity if base_ambiguity[key] != varied_ambiguity[key]
    } == {"placement"}

    base_corpus = _structured(0.0)
    varied_corpus = _structured(1.0)

    assert _generator(base).lexicon.forms == _generator(varied).lexicon.forms
    assert base_corpus.knobs == varied_corpus.knobs
    assert base_corpus.lexicon.per_type_ambiguity == (
        varied_corpus.lexicon.per_type_ambiguity
    )
    assert base_corpus.lexicon.realized_size == varied_corpus.lexicon.realized_size
    assert base_corpus.lexicon.ambiguity_frequency_correlation != (
        varied_corpus.lexicon.ambiguity_frequency_correlation
    )


def test_the_tabular_and_parametric_spellings_lower_to_one_lexicon() -> None:
    parametric = _structure(1.0)
    tabular = parametric.with_(
        joint=TabularJointSpec(
            rows=tuple(
                ClassSpec(category=category, share=share)
                for category, share in cast(
                    "ParametricJointSpec", parametric.joint
                ).marginal()
            )
        )
    )
    assemblies = (
        _assembly(_pair(parametric)),
        _assembly(_pair(tabular)),
    )
    from_parametric, from_tabular = (
        SyntheticGenerator(held, policy=FOREST_POLICY) for held in assemblies
    )

    def categories(assembly: GrammarAssembly, form: str) -> list[str]:
        return [
            assembly.syntax.render(assembly.grammar.store, entry.category)
            for entry in assembly.lexicon.entries(form)
        ]

    assert from_tabular.lexicon.forms == from_parametric.lexicon.forms

    for form in from_parametric.lexicon.forms:
        assert from_tabular.lexicon.log_mass(form) == pytest.approx(
            from_parametric.lexicon.log_mass(form)
        )
        assert categories(assemblies[1], form) == categories(assemblies[0], form)
        assert [
            entry.weight for entry in from_tabular.lexicon.entries(form)
        ] == pytest.approx(
            [entry.weight for entry in from_parametric.lexicon.entries(form)]
        )

    drawn = from_tabular.sample(40, seed=1, length=UniformLength(2, 2))
    expected = from_parametric.sample(40, seed=1, length=UniformLength(2, 2))

    assert [_surface(example.item) for example in drawn.examples] == [
        _surface(example.item) for example in expected.examples
    ]
    assert drawn.lexicon == expected.lexicon


def test_renaming_the_categories_changes_nothing_the_lexicon_draws_or_measures() -> (
    None
):
    renamed = tuple("PQRTUVWX")
    original = _structured(1.0)
    relabeled = _generator(_pair(_structure(1.0, categories=renamed), renamed)).sample(
        PAIR_COUNT, seed=1, length=UniformLength(2, 2)
    )
    names = dict(
        zip(
            (_atom_text(category) for category in PAIR_CATEGORIES),
            (_atom_text(category) for category in renamed),
            strict=True,
        )
    )

    assert [_surface(example.item) for example in relabeled.examples] == [
        _surface(example.item) for example in original.examples
    ]
    assert relabeled.lexicon == LexiconReport(
        {names[category]: held for category, held in original.lexicon.classes.items()},
        original.lexicon.forms,
        original.lexicon.per_type_ambiguity,
        original.lexicon.ambiguity_frequency_correlation,
    )
    assert relabeled.lexicon.concordance == original.lexicon.concordance


INTERPRETED_LEXICON: Final[tuple[tuple[str, str, str], ...]] = (
    ("john", "NP", "john"),
    ("mary", "NP", "mary"),
    ("sleeps", "S\\NP", "\\x. sleep(x)"),
    ("runs", "S\\NP", "\\x. run(x)"),
    ("loves", "(S\\NP)/NP", "\\y. \\x. love(x)(y)"),
    ("and", "(S\\S)/S", "\\q. \\p. p & q"),
    ("or", "(S\\S)/S", "\\q. \\p. p | q"),
    ("not", "S/S", "\\p. ~p"),
)
INTERPRETED_CONSTANTS: Final = (
    ConstantSpec(name="john", type="e", classes=("individual",)),
    ConstantSpec(name="mary", type="e", classes=("individual",)),
    ConstantSpec(name="sleep", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="run", type="e -> t", classes=("predicate",)),
    ConstantSpec(name="love", type="e -> e -> t", classes=("predicate",)),
)
INTERPRETED_SYMBOLS: Final = SymbolInventory(
    {
        "individual": {"john": 0, "mary": 0},
        "predicate": {"sleep": 1, "run": 1, "love": 2},
    }
)


def _interpreted_grammar() -> GrammarDecl:
    return GrammarDecl(
        name="interpreted",
        signature=SignatureSpec(
            constructors=slashes(),
            functors=slash_views(),
            primitives=PRIMITIVES[:3],
            operators=(OperatorSpec(name="NP[conj]", sort="Cat", erases_to="NP"),),
            conjunctions=CONJUNCTIONS,
        ),
        rules=_rules(("application", "composition", "type-raising")),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category, meaning=meaning)
                for form, category, meaning in INTERPRETED_LEXICON
            )
        ),
        goal=RootGoalSpec(categories=("S",)),
        matching=FeatureErasureMatchingSpec(),
        budget=BudgetSpec(unary_rounds=1),
    )


def _interpretation_spec(
    *, actions: dict[str, str] | None = None
) -> InterpretationSpec:
    return InterpretationSpec(
        erasure=SignatureMorphismErasureSpec(atoms={"S": "t", "NP": "e", "conj": "e"}),
        constants=INTERPRETED_CONSTANTS,
        meanings=MeaningSpec(source=LexicalMeaningSourceSpec()),
        actions={} if actions is None else actions,
    )


def _interpreted(
    *, actions: dict[str, str] | None = None, atoms: int = 12
) -> SyntheticGenerator:
    assembly, interpretation = lower_interpretation(
        _interpreted_grammar(),
        registered_language("propositional").with_(
            logic=registered_language("propositional").logic.with_(
                prover=TruthTableProverSpec(atoms=atoms, timeout=5.0)
            )
        ),
        _interpretation_spec(actions=actions),
        GrammarIso(),
        LanguageIso(),
    )

    return SyntheticGenerator(
        assembly, policy=FOREST_POLICY, interpretation=interpretation
    )


def _interpreted_sample(
    generator: SyntheticGenerator,
    *,
    count: int = 24,
    seed: int = 11,
    noise: LabelNoise | None = None,
) -> SyntheticCorpus:
    return generator.sample(count, seed=seed, length=UniformLength(2, 6), noise=noise)


def _read_gold(
    generator: SyntheticGenerator,
    example: Example[GoldDerivation[str]],
) -> Term:
    """Read the gold derivation's meaning again from its category text and rules."""
    interpretation = generator.interpretation
    assert interpretation is not None
    assembly_store = generator.grammar.store
    derivation = example.supervision.derivation
    tokens = [token.surface for token in example.item.tokens]

    def visit(
        node_id: int,
    ) -> MeaningDerivation[Term, float]:
        node = derivation.node(node_id)
        category = generator.grammar.matching.key(_category(assembly_store, node.label))

        if node.rule is None:
            return axiom(
                interpretation.primary(tokens[node.span.start], category),
                node.span.start,
            )

        return step(
            interpretation.rule(node.rule),
            tuple(visit(child) for child in node.children),
            category,
            0.0,
        )

    (meaning,) = interpretation.read(visit(derivation.root), interpretation.primaries)

    return meaning


def _category(store: TermStore, label: str) -> Term:
    syntax = infix_syntax(store.signature, ("/", "/"), ("\\", "\\"))

    return syntax.parse(store, label)


def test_every_grammatical_item_carries_the_form_its_gold_derivation_reads_to() -> None:
    generator = _interpreted()
    interpretation = generator.interpretation
    assert interpretation is not None
    corpus = _interpreted_sample(generator)

    assert len(corpus.forms) == len(corpus.examples) == 24
    assert tuple(corpus.forms) == tuple(
        example.item.item_id for example in corpus.examples
    )

    for example in corpus.examples:
        form = corpus.forms[example.item.item_id]
        term = interpretation.read_text(form.text)

        assert interpretation.formula(term) is term
        assert interpretation.render(term) == form.text
        assert _read_gold(generator, example) is term

    assert corpus.knobs.inventory == INTERPRETED_SYMBOLS
    assert generator.knobs.inventory == INTERPRETED_SYMBOLS


def test_a_grammar_without_an_interpretation_draws_no_forms() -> None:
    generator = _generator(_interpreted_grammar())
    corpus = generator.sample(6, seed=1, length=UniformLength(2, 4))

    assert generator.interpretation is None
    assert generator.knobs.inventory is None
    assert corpus.forms == {}

    with pytest.raises(ValueError, match="none is declared"):
        generator.infer(
            2,
            seed=1,
            premise=UniformLength(2, 4),
            hypothesis=UniformLength(2, 4),
            candidates=2,
            attempts=1,
            relations=NO_TRIVIAL,
        )


def test_a_supplied_action_that_breaks_typing_is_caught_on_the_drawn_form() -> None:
    generator = _interpreted(actions={"application:\\": "\\a. \\f. f(a) & a"})

    with pytest.raises(IllTypedFormError, match="leaves"):
        _interpreted_sample(generator)


def test_the_form_texts_are_reachable_without_any_term_or_store() -> None:
    corpus = _interpreted_sample(_interpreted())
    foreign = sorted(
        {
            f"{held.__module__}.{held.__qualname__}"
            for held in _reachable_types(corpus)
            if held.__module__.partition(".")[0] in {"didactic", "panproto"}
            or held.__module__.startswith(
                (
                    "lofi.terms",
                    "lofi.deduce",
                    "lofi.judge",
                    "lofi.formalism",
                    "lofi.typecheck",
                    "lofi.interpret",
                )
            )
        }
    )

    assert foreign == []


SCOPED_LENGTH: Final = UniformLength(3, 3)
FLAT_LENGTH: Final = UniformLength(2, 3)
SPURIOUS_LENGTH: Final = UniformLength(2, 5)


def _scope(
    *, readings: Target | None = None, derivations: int = 64, attempts: int = 8
) -> ScopeTarget:
    return ScopeTarget(readings=readings, derivations=derivations, attempts=attempts)


def test_a_readings_target_over_the_standard_pair_is_achieved_and_reported() -> None:
    corpus = scoped_generator().sample(
        8, seed=5, length=SCOPED_LENGTH, scope=_scope(readings=Target(2.0, 0.0))
    )
    report = corpus.scope

    assert report is not None
    assert report.rejected == 0
    assert report.exceeded == 0

    band = report.target.readings

    assert band is not None
    assert band.admits(report.readings.mean)
    assert all(band.admits(count.readings) for count in report.counts)
    assert report.readings == Histogram({2: 8})


def test_a_quantifier_free_fragment_gives_one_reading_per_string() -> None:
    corpus = flat_generator().sample(8, seed=2, length=FLAT_LENGTH, scope=_scope())
    report = corpus.scope

    assert report is not None
    assert report.readings == Histogram({1: 8})
    assert report.derivations == Histogram({1: 8})
    assert report.undecided == 0
    assert report.target.readings is None


def test_a_target_of_two_readings_a_fragment_cannot_reach_is_refused() -> None:
    with pytest.raises(ScopeUnreachableError, match=r"reach at Histogram\(\{1: 5\}\)"):
        flat_generator().sample(
            4,
            seed=2,
            length=FLAT_LENGTH,
            scope=_scope(readings=Target(2.0, 0.0), attempts=5),
        )


def test_the_report_counts_derivations_against_readings_per_string() -> None:
    corpus = spurious_generator().sample(
        6, seed=7, length=SPURIOUS_LENGTH, scope=_scope(derivations=256, attempts=1)
    )
    report = corpus.scope

    assert report is not None
    assert tuple(count.item_id for count in report.counts) == tuple(
        example.item.item_id for example in corpus.examples
    )
    assert report.derivations == corpus.ambiguity.derivations
    assert report.readings == Histogram({1: 6})
    assert any(count.derivations > count.readings for count in report.counts)


def test_scope_counts_each_lexical_meaning_choice_as_a_derivation() -> None:
    generator = built(
        "polysemous",
        (
            ("john", "NP", "john"),
            ("john", "NP", "mary"),
            ("sleeps", "S\\NP", "\\x. sleep(x)"),
        ),
        FLAT_INVENTORY,
        APPLICATION,
    )
    corpus = generator.sample(
        1,
        seed=1,
        length=UniformLength(2, 2),
        scope=_scope(readings=Target(2.0, 0.0), derivations=2, attempts=1),
    )
    report = corpus.scope

    assert report is not None
    assert corpus.ambiguity.derivations == Histogram({1: 1})
    assert report.derivations == Histogram({2: 1})
    assert report.readings == Histogram({2: 1})


def test_a_sample_reads_its_readings_under_the_interpretation_clock() -> None:
    declaration = grammar("joined", JOINED_ENTRIES, SCOPING)

    def sampled(timeout: float) -> SyntheticCorpus:
        assembly, built = lower_interpretation(
            declaration,
            registered_language("propositional").with_(
                logic=registered_language("propositional").logic.with_(
                    prover=TruthTableProverSpec(timeout=timeout)
                )
            ),
            interpretation(JOINED_INVENTORY),
            GrammarIso(),
            LanguageIso(),
        )
        generator = SyntheticGenerator(
            assembly, policy=FOREST_POLICY, interpretation=built
        )

        return generator.sample(
            6, seed=1, length=UniformLength(3, 3), scope=_scope(attempts=1)
        )

    patient = sampled(5.0).scope
    impatient = sampled(1e-9).scope

    assert patient is not None
    assert impatient is not None
    assert patient.derivations == impatient.derivations
    assert patient.undecided == 0
    assert impatient.undecided > 0
    assert impatient.readings.mean > patient.readings.mean


def test_a_string_beyond_the_derivation_bound_is_redrawn_and_counted() -> None:
    with pytest.raises(ScopeUnreachableError, match="beyond 1 derivations"):
        spurious_generator().sample(
            4,
            seed=3,
            length=UniformLength(3, 5),
            scope=_scope(readings=Target(1.0, 0.0), derivations=1, attempts=4),
        )


def test_a_sample_drawn_without_a_scope_target_reports_no_readings() -> None:
    corpus = scoped_generator().sample(4, seed=5, length=SCOPED_LENGTH)

    assert corpus.scope is None


def test_a_scope_target_needs_an_interpretation() -> None:
    generator = _generator(_interpreted_grammar())

    with pytest.raises(ValueError, match="none is declared"):
        generator.sample(2, seed=1, length=UniformLength(2, 4), scope=_scope())


INJECTED_LENGTH: Final = UniformLength(2, 6)
"""The token counts the injected samples are drawn at."""


def _injected(
    generator: SyntheticGenerator,
    *,
    count: int = 24,
    seed: int = 11,
    length: LengthDistribution = INJECTED_LENGTH,
    proportion: float = 0.5,
    operators: tuple[
        Callable[[IllTypingContext], IllTypingOperator], ...
    ] = STANDARD_ILL_TYPING,
    attempts: int = 60,
    policy: NegativeDataPolicy | None = None,
    corruption: Corruption[CorruptionOperator] | None = None,
    noise: LabelNoise | None = None,
) -> SyntheticCorpus:
    return generator.sample(
        count,
        seed=seed,
        length=length,
        corruption=corruption,
        noise=noise,
        ill_typing=_ill_typing(
            generator,
            proportion,
            operators=operators,
            attempts=attempts,
            policy=policy,
        ),
    )


def test_every_injected_form_is_refused_by_the_checker_it_was_certified_by() -> None:
    generator = _interpreted()
    interpretation = generator.interpretation
    assert interpretation is not None
    corpus = _injected(generator)
    report = corpus.ill_typing

    assert report is not None
    assert len(corpus.ill_typed) == 12

    for record in corpus.ill_typed:
        source = interpretation.read_text(record.source)
        refused = interpretation.read_text(record.text)

        assert record.source == corpus.forms[record.item_id].text
        assert interpretation.check(source).is_well_typed
        assert not interpretation.check(refused).is_well_typed
        assert record.certificate.conflicts != ()
        assert record.certificate.language_digest == (
            interpretation.language.declaration_digest
        )
        assert interpretation.failure(refused) == record.certificate
        assert generator.verify_typing(record)


@pytest.mark.parametrize(
    ("policy", "observed"),
    [
        (NegativeDataPolicy("labeled-negatives"), "ill-typed"),
        (NegativeDataPolicy("positive-only"), "well-typed"),
    ],
)
def test_fixed_negative_data_regimes_assign_every_injected_form_the_declared_label(
    policy: NegativeDataPolicy, observed: FormTypingLabel
) -> None:
    corpus = _injected(_interpreted(), count=24, policy=policy)
    report = corpus.ill_typing

    assert report is not None
    assert {record.true_label for record in corpus.ill_typed} == {"ill-typed"}
    assert {record.observed_label for record in corpus.ill_typed} == {observed}
    assert report.injection.requested.policy == policy
    assert report.flipped == (len(corpus.ill_typed) if observed == "well-typed" else 0)


def test_label_flip_noise_is_drawn_over_injected_forms_and_reported() -> None:
    policy = NegativeDataPolicy("label-flips", noise=0.4)
    corpus = _injected(_interpreted(), count=80, policy=policy)
    report = corpus.ill_typing

    assert report is not None
    assert report.flipped == sum(record.flipped for record in corpus.ill_typed)
    assert dict(report.observed) == {
        "well-typed": report.flipped,
        "ill-typed": len(corpus.ill_typed) - report.flipped,
    }
    assert abs(report.achieved_noise - 0.4) <= 4 * (
        (0.4 * 0.6 / len(corpus.ill_typed)) ** 0.5
    )


def test_changing_only_the_negative_data_regime_preserves_the_injected_forms() -> None:
    labeled = _injected(
        _interpreted(), count=24, policy=NegativeDataPolicy("labeled-negatives")
    )
    contaminated = _injected(
        _interpreted(), count=24, policy=NegativeDataPolicy("positive-only")
    )

    def content(corpus: SyntheticCorpus) -> tuple[tuple[object, ...], ...]:
        return tuple(
            (form.item_id, form.kind, form.source, form.text, form.certificate)
            for form in corpus.ill_typed
        )

    assert content(labeled) == content(contaminated)


def test_a_tampered_typing_certificate_is_declined() -> None:
    generator = _interpreted()
    corpus = _injected(generator)
    record = corpus.ill_typed[0]
    held = record.certificate
    conflict = held.conflicts[0]

    def tampered(certificate: TypingFailure) -> IllTyped:
        return IllTyped(
            record.item_id,
            record.kind,
            record.source,
            record.text,
            certificate,
            observed_label=record.observed_label,
        )

    assert generator.verify_typing(record)
    assert not generator.verify_typing(
        tampered(TypingFailure(held.conflicts, language_digest="f" * 64))
    )
    assert not generator.verify_typing(
        tampered(
            TypingFailure(
                (
                    TypingConflict(
                        conflict.kind,
                        conflict.left,
                        conflict.right,
                        (*conflict.path, 0),
                    ),
                ),
                language_digest=held.language_digest,
            )
        )
    )
    assert not generator.verify_typing(
        IllTyped(
            record.item_id,
            record.kind,
            record.text,
            record.source,
            held,
            observed_label=record.observed_label,
        )
    )


@pytest.mark.parametrize("proportion", [0.0, 0.25, 0.5, 1.0])
def test_the_achieved_injection_rate_is_the_declared_one_within_rounding(
    proportion: float,
) -> None:
    corpus = _injected(_interpreted(), proportion=proportion)
    report = corpus.ill_typing

    assert report is not None
    assert report.injection.requested.proportion == proportion
    assert report.injection.sample_size == len(corpus.examples)
    assert report.injection.corrupted == len(corpus.ill_typed)
    assert abs(report.injection.achieved_proportion - proportion) <= 0.5 / len(
        corpus.examples
    )


def test_the_report_counts_the_certificate_kinds_the_injection_produced() -> None:
    corpus = _injected(scoped_generator(), count=8, length=UniformLength(3, 3))
    report = corpus.ill_typing

    assert report is not None
    assert dict(report.certificates) == Counter(
        kind for record in corpus.ill_typed for kind in record.certificate.kinds
    )
    assert sum(report.certificates.values()) >= len(corpus.ill_typed)
    assert sum(report.injection.accepted.values()) == len(corpus.ill_typed)
    assert set(report.certificates) <= {"unification", "annotation", "infinite_type"}


def test_an_inapplicable_operator_is_counted_and_another_is_drawn() -> None:
    corpus = _injected(_interpreted(), operators=(_argument_swap, _sort_mismatch))
    report = corpus.ill_typing

    assert report is not None
    assert report.injection.inapplicable["sort-mismatch"] > 0
    assert report.injection.accepted["sort-mismatch"] == 0
    assert sum(report.injection.accepted.values()) == len(corpus.ill_typed)


def test_an_injection_of_one_inapplicable_operator_runs_out_of_candidates() -> None:
    with pytest.raises(CorruptionExhaustedError, match="all typed or were"):
        _injected(_interpreted(), operators=(_sort_mismatch,), attempts=3)


def test_the_syntactic_and_semantic_corruption_reports_share_one_shape() -> None:
    generator = _interpreted()
    corpus = _injected(
        generator,
        corruption=_corruption(generator, 6, 0.25, attempts=100),
    )
    syntactic = corpus.corruption
    semantic = None if corpus.ill_typing is None else corpus.ill_typing.injection

    assert syntactic is not None
    assert semantic is not None
    assert type(syntactic) is type(semantic)
    assert syntactic.requested.operator_names != semantic.requested.operator_names

    for report in (syntactic, semantic):
        counts = (report.accepted, report.rejected, report.inapplicable)

        assert all(tuple(count) == report.requested.operator_names for count in counts)
        assert report.corrupted == sum(report.accepted.values())
        assert report.achieved_proportion == report.corrupted / report.sample_size


def test_injected_forms_need_an_interpretation() -> None:
    generator = _generator(_interpreted_grammar())

    with pytest.raises(ValueError, match="none is declared"):
        generator.sample(
            2,
            seed=1,
            length=UniformLength(2, 4),
            ill_typing=_ill_typing(_interpreted(), 0.5, attempts=4),
        )


def test_the_items_drawn_at_a_seed_are_the_same_with_and_without_forms() -> None:
    generator = _interpreted()
    noise = LabelNoise(grammatical=0.5, ungrammatical=0.5)
    plain = generator.sample(24, seed=11, length=UniformLength(2, 6), noise=noise)
    injected = _injected(generator, noise=noise)

    assert injected.examples == plain.examples
    assert injected.labels == plain.labels
    assert injected.ill_typed != ()
    assert plain.ill_typed == ()
    assert plain.ill_typing is None


@settings(max_examples=2 * SMALL.examples, deadline=None)
@given(grammars(SMALL), st.integers(0, 2**31 - 1))
def test_every_example_of_a_generated_grammar_reparses_to_its_gold_derivation(
    declaration: GrammarDecl, seed: int
) -> None:
    assembly = _assembly(declaration)
    corpus = SyntheticGenerator(assembly, policy=FOREST_POLICY).sample(
        SMALL.size, seed=seed, length=UniformLength(2, SMALL.length)
    )
    forms = set(assembly.lexicon.forms)

    assert len(corpus.examples) == SMALL.size

    for example in corpus.examples:
        tokens = tuple(token.surface for token in example.item.tokens)

        assert set(tokens) <= forms
        assert 2 <= len(tokens) <= SMALL.length
        assert _gold_shape(example.supervision.derivation) in _parsed_shapes(
            assembly, tokens
        )
