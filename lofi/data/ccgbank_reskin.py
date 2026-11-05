"""Derivation-preserving CCGbank reskinning with empirical lexical forms."""

from __future__ import annotations

from collections import Counter
from importlib import import_module
from json import dumps, loads
from math import log
from pathlib import Path
from random import Random
from typing import TYPE_CHECKING, Final, Literal, Protocol, cast, final

from lofi._frozen import FrozenValue
from lofi.data import Derivation, DerivationNode, Example, GoldDerivation
from lofi.data.datasets import DatasetPlugin, DatasetVocabulary, RefusedTraining
from lofi.data.synthetic import (
    AmbiguityReport,
    Corrupted,
    Corruption,
    CorruptionExhaustedError,
    CorruptionReport,
    GrammarKnobs,
    Histogram,
    InventoryReport,
    LengthReport,
    LexiconReport,
    NoiseReport,
    ObservedLabel,
    SyntheticCorpus,
    SyntheticDataset,
    SyntheticGenerator,
    UniformLength,
)
from lofi.data.synthetic._corrupt import CorruptionContext, CorruptionOperator
from lofi.data.synthetic._inference import synthetic_inference_item
from lofi.data.synthetic.spec.ccgbank import CcgbankReskinDatasetSpec
from lofi.data.synthetic.training import (
    CorpusSentences,
    corpus_featurizer_resources,
    corpus_sentences,
)
from lofi.deduce import ParsePolicy, Retain
from lofi.spec import (
    ApplicationSpec,
    ConstantPatternSpec,
    FixedOperatorSpec,
    NodePatternSpec,
    PatternRuleSpec,
    PatternSpec,
    RuleSetSpec,
    TermPatternSpec,
    fixed_seed,
)
from lofi.spec._interpolation import lower_package_resource
from lofi.spec.grammar import (
    GrammarAssembly,
    GrammarDecl,
    GrammarIso,
    InlineLexiconSpec,
    LexicalEntrySpec,
    NoNormalFormSpec,
    RootGoalSpec,
)
from lofi.spec.language import LanguageDecl

if TYPE_CHECKING:
    from collections.abc import Mapping

    import lofi.data.corpora.ccgbank.reader
    from lofi.data import Item
    from lofi.data.datasets import ReservedForms
    from lofi.data.synthetic._values import LabelNoise
    from lofi.data.synthetic.spec.operators import WeightedCorruptionOperatorFactory
    from lofi.terms import Term

__all__ = ["PLUGIN", "SOURCE_VERSION", "TARGET_VERSION", "CheckedCcgbankReskin"]

SOURCE_VERSION: Final = "lofi-ccgbank-reskin-decl"
TARGET_VERSION: Final = "lofi-ccgbank-reskin-native"
type _CcgbankSplit = Literal["train", "validation", "development", "test"]


def _canonical(declaration: CcgbankReskinDatasetSpec, /) -> bytes:
    document = dumps(
        loads(declaration.model_dump_json()), separators=(",", ":"), ensure_ascii=True
    )
    return document.encode("utf-8")


@final
class _CheckedCorruption(FrozenValue):
    """A corruption request awaiting its empirical lexical context."""

    _noun = "checked CCGbank reskin corruption"
    __slots__ = ("_attempts", "_draw", "_factories", "_proportion")

    def __init__(
        self,
        proportion: float,
        factories: tuple[WeightedCorruptionOperatorFactory, ...],
        attempts: int,
        draw: Literal["exact", "bernoulli"],
    ) -> None:
        object.__setattr__(self, "_proportion", proportion)
        object.__setattr__(self, "_factories", factories)
        object.__setattr__(self, "_attempts", attempts)
        object.__setattr__(self, "_draw", draw)

    def build(self, context: CorruptionContext, /) -> Corruption[CorruptionOperator]:
        """Build the request after the empirical lexicon is available."""
        return Corruption(
            self._proportion,
            operators=tuple(factory.build(context) for factory in self._factories),
            attempts=self._attempts,
            draw=self._draw,
        )


@final
class CheckedCcgbankReskin(FrozenValue):
    """Allocation-free values needed to lower one declared reskin."""

    _noun = "checked CCGbank reskin"
    __slots__ = (
        "_corruption",
        "_declaration",
        "_grammar",
        "_minimum_frequency",
        "_noise",
        "_prefix",
        "_seed",
        "_source",
        "_splits",
    )

    def __init__(
        self,
        declaration: bytes,
        grammar: GrammarAssembly,
        source: str,
        splits: tuple[_CcgbankSplit, ...],
        prefix: str,
        minimum_frequency: int,
        seed: int,
        corruption: _CheckedCorruption | None,
        noise: LabelNoise | None,
    ) -> None:
        object.__setattr__(self, "_declaration", declaration)
        object.__setattr__(self, "_grammar", grammar)
        object.__setattr__(self, "_source", source)
        object.__setattr__(self, "_splits", splits)
        object.__setattr__(self, "_prefix", prefix)
        object.__setattr__(self, "_minimum_frequency", minimum_frequency)
        object.__setattr__(self, "_seed", seed)
        object.__setattr__(self, "_corruption", corruption)
        object.__setattr__(self, "_noise", noise)

    @property
    def declaration(self) -> bytes:
        """Return the canonical dataset declaration."""
        return self._declaration

    @property
    def source_grammar(self) -> GrammarAssembly:
        """Return the grammar used to read the source derivations."""
        return self._grammar

    @property
    def source_interpretation(self) -> None:
        """Return no logical-form interpretation for syntactic source data."""
        return None

    @property
    def source_interpretation_declaration(self) -> None:
        """Return no logical-form interpretation declaration."""
        return None

    @property
    def source(self) -> str:
        """Return the interpolated CCGbank source path."""
        return self._source

    @property
    def splits(self) -> tuple[_CcgbankSplit, ...]:
        """Return source splits in declared order."""
        return self._splits

    @property
    def prefix(self) -> str:
        """Return the synthetic-form prefix."""
        return self._prefix

    @property
    def minimum_frequency(self) -> int:
        """Return the empirical leaf-category cutoff."""
        return self._minimum_frequency

    @property
    def seed(self) -> int:
        """Return the fixed reskin seed."""
        return self._seed

    @property
    def corruption(self) -> _CheckedCorruption | None:
        """Return the compiled corruption request, if declared."""
        return self._corruption

    @property
    def noise(self) -> LabelNoise | None:
        """Return the interior label-noise policy, if declared."""
        return self._noise


def _core(grammar: GrammarAssembly, label: str, /) -> str:
    term = grammar.syntax.parse(grammar.store, label)
    return grammar.syntax.render(grammar.store, term.core)


def _leaf_labels(example: Example[GoldDerivation[str]], /) -> tuple[str, ...]:
    derivation = example.supervision.derivation
    positioned = {
        node.span.start: node.label
        for node in derivation.nodes.values()
        if node.is_lexical
    }
    return tuple(positioned[token.index] for token in example.item.tokens)


def _pattern(base: GrammarAssembly, term: Term, /) -> PatternSpec:
    """Declare one exact category pattern from a source category."""
    if term.op.is_constant:
        return ConstantPatternSpec(
            name=base.store.payload_name(term), sort=term.op.sort
        )
    return NodePatternSpec(
        operator=FixedOperatorSpec(name=term.op.name),
        arguments=tuple(_pattern(base, argument) for argument in term.args),
    )


def _empirical_rules(
    base: GrammarAssembly,
    examples: tuple[Example[GoldDerivation[str]], ...],
    /,
) -> RuleSetSpec:
    """Declare the unique local productions found in retained derivations."""
    productions: dict[tuple[tuple[str, ...], str], str] = {}
    for example in examples:
        derivation = example.supervision.derivation
        for node in derivation.nodes.values():
            if node.is_lexical:
                continue
            premises = tuple(
                _core(base, child.label) if child.is_lexical else child.label
                for child in (derivation.nodes[child_id] for child_id in node.children)
            )
            productions.setdefault((premises, node.label), node.rule or "production")
    rules = tuple(
        PatternRuleSpec(
            name=f"{name}:{index}",
            premises=tuple(
                _pattern(base, base.syntax.parse(base.store, category))
                for category in premises
            ),
            conclusion=TermPatternSpec(name=f"Result{index}", sort="Cat"),
            targets=(output,),
        )
        for index, ((premises, output), name) in enumerate(productions.items())
    )
    if not rules:
        msg = "a CCGbank reskin retains no nonlexical production"
        raise ValueError(msg)
    return RuleSetSpec(
        name="ccgbank-reskin",
        rules=(ApplicationSpec(view="/"), ApplicationSpec(view="\\"), *rules),
    )


def _reskinned_grammar(
    base: GrammarAssembly,
    rows: tuple[tuple[str, str, int], ...],
    examples: tuple[Example[GoldDerivation[str]], ...],
    /,
) -> GrammarAssembly:
    """Compile the empirical grammar licensed by retained derivations."""
    declared = GrammarDecl.model_validate_json(base.declaration)
    language = LanguageDecl.model_validate_json(base.language.declaration)
    roots = tuple(
        dict.fromkeys(
            _core(
                base,
                example.supervision.derivation.nodes[
                    example.supervision.derivation.root
                ].label,
            )
            for example in examples
        )
    )
    reskinned = declared.with_(
        rules=_empirical_rules(base, examples),
        goal=RootGoalSpec(categories=roots),
        normal_form=NoNormalFormSpec(),
        budget=declared.budget.with_(
            closure_rounds=max(
                sum(
                    not node.is_lexical
                    for node in example.supervision.derivation.nodes.values()
                )
                for example in examples
            )
        ),
        lexicon=InlineLexiconSpec(
            entries=tuple(
                LexicalEntrySpec(form=form, category=category, weight=log(count))
                for form, category, count in rows
            )
        ),
    )
    return GrammarIso().forward(reskinned, language)


def _transformed(
    example: Example[GoldDerivation[str]],
    base: GrammarAssembly,
    form_by_category: Mapping[str, str],
    /,
) -> Example[GoldDerivation[str]]:
    derivation = example.supervision.derivation
    nodes = {
        node_id: DerivationNode(
            node.span,
            _core(base, node.label) if node.is_lexical else node.label,
            rule=node.rule,
            children=node.children,
            features=node.features,
        )
        for node_id, node in derivation.nodes.items()
    }
    forms = tuple(
        form_by_category[_core(base, label)] for label in _leaf_labels(example)
    )
    item = synthetic_inference_item(
        example.item.item_id,
        " ".join(forms),
        split=example.item.split,
        provenance=example.item.provenance,
    )
    return Example(item, GoldDerivation(Derivation(derivation.root, nodes)))


def _inventory(
    grammar: GrammarAssembly, categories: tuple[str, ...]
) -> InventoryReport:
    terms = tuple(
        grammar.syntax.parse(grammar.store, category) for category in categories
    )
    occurring: set[str] = set()
    pending = list(terms)
    while pending:
        term = pending.pop()
        if term.op.is_constant:
            occurring.add(
                grammar.store.signature.constant_core(
                    term.op.sort, grammar.store.payload_name(term)
                )
            )
        elif not term.args:
            occurring.add(term.op.erases_to or term.op.name)
        pending.extend(term.args)
    vocabulary = grammar.store.signature.vocabulary(grammar.language.category_sort)
    measure = grammar.category_measure
    return InventoryReport(
        {
            category: measure.depth(term)
            for category, term in zip(categories, terms, strict=True)
        },
        {
            category: measure.arity(term)
            for category, term in zip(categories, terms, strict=True)
        },
        {
            category: measure.order(term)
            for category, term in zip(categories, terms, strict=True)
        },
        tuple(name for name in vocabulary.names if name in occurring),
        max_arguments=grammar.max_category_arguments,
        max_order=grammar.max_category_order,
    )


def _knobs(grammar: GrammarAssembly) -> GrammarKnobs:
    compiled = grammar.grammar
    sort = compiled.targets.categories[0].op.sort
    operators = tuple(op for op in compiled.signature.operators if op.sort == sort)
    vocabulary = compiled.signature.vocabulary(sort)
    primitives = sum(
        compiled.signature.constant_core(sort, name) == name
        for name in vocabulary.names
    )
    constructors = tuple(
        op.name
        for op in operators
        if op.arity > 0 and all(argument == sort for argument in op.arg_sorts)
    )
    return GrammarKnobs(
        primitives, constructors, tuple(rule.name for rule in compiled.rules)
    )


def _reports(
    examples: tuple[Example[GoldDerivation[str]], ...], grammar: GrammarAssembly, /
) -> tuple[AmbiguityReport, LexiconReport, InventoryReport, LengthReport]:
    if not examples:
        msg = "a CCGbank reskin must retain at least one grammatical example"
        raise ValueError(msg)
    forms: Counter[str] = Counter()
    category_by_form: dict[str, str] = {}
    for example in examples:
        labels = _leaf_labels(example)
        for token, category in zip(example.item.tokens, labels, strict=True):
            forms[token.surface] += 1
            category_by_form[token.surface] = category
    ordered_categories = tuple(dict.fromkeys(category_by_form.values()))
    lengths = tuple(len(example.item.tokens) for example in examples)
    lexical = LexiconReport(
        {
            category: Histogram.of(
                forms[form]
                for form, held in category_by_form.items()
                if held == category
            )
            for category in ordered_categories
        },
        Histogram.of(forms.values()),
        Histogram.of(1 for _ in forms),
        None,
    )
    return (
        AmbiguityReport(
            Histogram.of(1 for example in examples for _ in example.item.tokens),
            Histogram.of(1 for _ in examples),
            rejected=0,
            target=None,
        ),
        lexical,
        _inventory(grammar, ordered_categories),
        LengthReport(
            Histogram.of(lengths),
            resampled=0,
            distribution=UniformLength(min(lengths), max(lengths)),
        ),
    )


def _corrupt(
    examples: tuple[Example[GoldDerivation[str]], ...],
    grammar: GrammarAssembly,
    request: Corruption[CorruptionOperator] | None,
    policy: ParsePolicy[Retain],
    random: Random,
) -> tuple[
    tuple[Example[GoldDerivation[str]], ...],
    tuple[Corrupted, ...],
    CorruptionReport[CorruptionOperator] | None,
    tuple[tuple[Item, Literal["grammatical", "ungrammatical"]], ...],
]:
    if request is None:
        return (
            examples,
            (),
            None,
            tuple((example.item, "grammatical") for example in examples),
        )
    slots = frozenset(
        random.sample(range(len(examples)), request.count(len(examples), random))
    )
    if len(slots) == len(examples):
        msg = "a CCGbank reskin corruption must leave one grammatical example"
        raise ValueError(msg)
    generator = SyntheticGenerator(
        grammar,
        policy=cast("ParsePolicy[Literal[Retain.FOREST]]", policy),
    )
    accepted: Counter[str] = Counter()
    rejected: Counter[str] = Counter()
    inapplicable: Counter[str] = Counter()
    positives: list[Example[GoldDerivation[str]]] = []
    negatives: list[Corrupted] = []
    truths: list[tuple[Item, Literal["grammatical", "ungrammatical"]]] = []
    for index, example in enumerate(examples):
        if index not in slots:
            positives.append(example)
            truths.append((example.item, "grammatical"))
            continue
        source = tuple(token.surface for token in example.item.tokens)
        for _ in range(request.attempts):
            operator = request.choose(random)
            candidate = operator.apply(random, source)
            if candidate is None:
                inapplicable[operator.name] += 1
                continue
            certificate = generator.certify(candidate)
            if certificate is None:
                rejected[operator.name] += 1
                continue
            item = synthetic_inference_item(
                example.item.item_id,
                " ".join(candidate),
                split=example.item.split,
                provenance=example.item.provenance,
            )
            record = Corrupted(item, operator.name, source, certificate)
            accepted[operator.name] += 1
            negatives.append(record)
            truths.append((item, "ungrammatical"))
            break
        else:
            msg = (
                f"{request.attempts} candidates for {example.item.item_id!r} from "
                f"{request.operator_names!r} were derivable or inapplicable"
            )
            raise CorruptionExhaustedError(msg)
    report = CorruptionReport(
        request,
        accepted=accepted,
        rejected=rejected,
        inapplicable=inapplicable,
        sample_size=len(examples),
    )
    return tuple(positives), tuple(negatives), report, tuple(truths)


def _labels(
    truths: tuple[tuple[Item, Literal["grammatical", "ungrammatical"]], ...],
    noise: LabelNoise | None,
    random: Random,
) -> tuple[ObservedLabel, ...]:
    return tuple(
        ObservedLabel.of(
            item.item_id,
            truth,
            flipped=False if noise is None else random.random() < noise.rate(truth),
        )
        for item, truth in truths
    )


class _Reading(Protocol):
    """The CCGbank reader module, loaded only when a dataset reads its corpus."""

    CcgbankCorpusReader: type[lofi.data.corpora.ccgbank.reader.CcgbankCorpusReader]


def _reading() -> _Reading:
    """Return the CCGbank reader at the lowering site that reads the corpus.

    The reader decodes through the optional corpus codecs, so discovering the
    dataset plugin imports neither the reader nor its codecs.
    """
    return cast("_Reading", import_module("lofi.data.corpora.ccgbank.reader"))


def _check(source: CcgbankReskinDatasetSpec, /) -> CheckedCcgbankReskin:
    prefix, minimum_frequency = source.mapping.settings()
    corruption = source.corruption
    checked_corruption = (
        None
        if corruption is None
        else _CheckedCorruption(
            corruption.proportion.value,
            tuple(entry.compile() for entry in corruption.operators.entries),
            corruption.attempts,
            corruption.proportion.draw,
        )
    )
    return CheckedCcgbankReskin(
        _canonical(source),
        GrammarIso().forward(source.grammar, source.language),
        source.source,
        tuple(source.splits),
        prefix,
        minimum_frequency,
        fixed_seed("a CCGbank reskin's seed", source.seed),
        checked_corruption,
        None if source.noise is None else source.noise.build(),
    )


def _lower(
    checked: CheckedCcgbankReskin, policy: ParsePolicy[Retain], /
) -> SyntheticDataset:
    if policy.retain is not Retain.FOREST:
        msg = "CCGbank reskin corruption requires packed-forest retention"
        raise ValueError(msg)
    reader = _reading().CcgbankCorpusReader(
        Path(lower_package_resource(checked.source)),
        splits=checked.splits,
    )
    source = tuple(
        example for split in checked.splits for example in reader.examples(split)
    )
    counts: Counter[str] = Counter(
        _core(checked.source_grammar, label)
        for example in source
        for label in _leaf_labels(example)
    )
    retained = tuple(
        sorted(
            (
                (category, count)
                for category, count in counts.items()
                if count >= checked.minimum_frequency
            ),
            key=lambda pair: (-pair[1], pair[0]),
        )
    )
    if not retained:
        msg = "the CCGbank reskin frequency filter retains no leaf category"
        raise ValueError(msg)
    form_by_category = {
        category: f"{checked.prefix}{index}"
        for index, (category, _) in enumerate(retained)
    }
    admitted = tuple(
        example
        for example in source
        if all(
            _core(checked.source_grammar, label) in form_by_category
            for label in _leaf_labels(example)
        )
    )
    if not admitted:
        msg = "the CCGbank reskin frequency filter retains no derivation"
        raise ValueError(msg)
    rows = tuple(
        (form_by_category[category], category, count) for category, count in retained
    )
    grammar = _reskinned_grammar(checked.source_grammar, rows, admitted)
    transformed = tuple(
        _transformed(example, checked.source_grammar, form_by_category)
        for example in admitted
    )
    maximum = max(len(example.item.tokens) for example in transformed)
    context = CorruptionContext(grammar.lexicon, grammar.grammar.matching, maximum)
    request = None if checked.corruption is None else checked.corruption.build(context)
    random = Random(checked.seed)
    examples, corrupted, corruption, truths = _corrupt(
        transformed, grammar, request, policy, random
    )
    ambiguity, lexicon, inventory, length = _reports(examples, grammar)
    labels = _labels(truths, checked.noise, random)
    noise = None if checked.noise is None else NoiseReport.of(checked.noise, labels)
    corpus = SyntheticCorpus(
        grammar.declaration,
        grammar.grammar.theory_specification,
        seed=checked.seed,
        length=length,
        knobs=_knobs(grammar),
        examples=examples,
        ambiguity=ambiguity,
        lexicon=lexicon,
        inventory=inventory,
        corrupted=corrupted,
        corruption=corruption,
        labels=labels,
        noise=noise,
    )
    return SyntheticDataset(
        checked.declaration,
        grammar,
        corpus,
        source_version=SOURCE_VERSION,
        target_version=TARGET_VERSION,
    )


def _validate(dataset: SyntheticDataset, /) -> None:
    declaration = CcgbankReskinDatasetSpec.model_validate_json(dataset.declaration)
    fixed_seed("a CCGbank reskin's seed", declaration.seed)
    if dataset.corpus.declaration != dataset.source_grammar.declaration:
        msg = "a CCGbank reskin corpus was lowered under another grammar"
        raise ValueError(msg)


def _vocabulary(
    dataset: SyntheticDataset, reserved: ReservedForms, /
) -> DatasetVocabulary:
    return DatasetVocabulary(
        dataset.source_grammar.lexicon.forms,
        reserved=reserved.indices,
        unknown=reserved.unknown,
    )


def _sentences(dataset: SyntheticDataset, /) -> CorpusSentences:
    declaration = CcgbankReskinDatasetSpec.model_validate_json(dataset.declaration)
    return corpus_sentences(dataset, declaration.training_split)


def _streams(declaration: CcgbankReskinDatasetSpec, /) -> tuple[str, ...]:
    del declaration
    return ("ccgbank-reskin",)


PLUGIN: Final = DatasetPlugin(
    spec=CcgbankReskinDatasetSpec,
    checked=CheckedCcgbankReskin,
    dataset=SyntheticDataset,
    check=_check,
    lower=_lower,
    validate=_validate,
    vocabulary=_vocabulary,
    resources=corpus_featurizer_resources,
    sentences=_sentences,
    pairs=RefusedTraining(
        "a CCGbank reskin supplies sentence training, not inference pairs"
    ),
    streams=_streams,
)
