"""Gold sentence corpora read from treebanks under a declared grammar.

A treebank reader yields gold derivations whose steps carry the corpus's own
rule names. The builder here lowers each step to the compiled rule that
licenses it, measures the ambiguity, lexical and inventory reports of the read
examples, and returns them as the one sentence corpus a parser trains on.
"""

from collections import Counter
from collections.abc import Mapping as RuntimeMapping
from hashlib import sha256
from itertools import product
from json import dumps
from types import MappingProxyType
from typing import TYPE_CHECKING, cast, final

from lofi._frozen import FrozenValue
from lofi._stats import spearman
from lofi.data import (
    AnalysisCounts,
    Derivation,
    DerivationNode,
    Example,
    GoldDerivation,
    LoadReport,
    reachability_report,
)
from lofi.data.corpora import GrammarInventory
from lofi.data.synthetic import (
    AmbiguityReport,
    GrammarKnobs,
    Histogram,
    InventoryReport,
    LengthReport,
    LexiconReport,
    ObservedLabel,
    SyntheticCorpus,
    UniformLength,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import HeadSelection
    from lofi.judge import Rule
    from lofi.spec.grammar import GrammarAssembly
    from lofi.terms import Term

__all__ = ["GoldCorpus", "corpus_digest", "gold_corpus", "lexical_counts"]


@final
class GoldCorpus(FrozenValue):
    """A treebank's gold sentence corpus beside its load report.

    Parameters
    ----------
    corpus : SyntheticCorpus
        The canonical examples with their measured reports.
    report : LoadReport[str]
        The reader's counts with the corpus's reachability and normal-form
        analyses.
    digest : str
        The digest of the canonical source analyses.
    """

    __slots__ = ("_corpus", "_digest", "_report")
    _noun = "gold sentence corpus"

    def __init__(
        self, corpus: SyntheticCorpus, report: LoadReport[str], digest: str
    ) -> None:
        object.__setattr__(self, "_corpus", corpus)
        object.__setattr__(self, "_report", report)
        object.__setattr__(self, "_digest", digest)

    @property
    def corpus(self) -> SyntheticCorpus:
        """Return the canonical examples with their measured reports."""
        return self._corpus

    @property
    def report(self) -> LoadReport[str]:
        """Return the load counts with reachability and normal-form analyses."""
        return self._report

    @property
    def digest(self) -> str:
        """Return the digest of the canonical source analyses."""
        return self._digest


def _in_normal_form(
    derivation: GoldDerivation[str], grammar: GrammarAssembly, /
) -> bool:
    """Return whether some recovered reading of a gold tree is in normal form."""
    tree = derivation.derivation
    normal_form = grammar.grammar.normal_form
    rules = grammar.grammar.rules
    terms = {
        node_id: grammar.syntax.parse(grammar.store, node.label)
        for node_id, node in tree.nodes.items()
    }
    tags: dict[int, set[str]] = {}

    for node_id in reversed(tree.preorder):
        node = tree.node(node_id)
        if node.is_lexical:
            tags[node_id] = {normal_form.lexical_tag()}
            continue

        inputs = tuple(terms[child] for child in node.children)
        output = terms[node_id]
        held: set[str] = set()
        for rule in rules.recover(inputs, output):
            for child_tags in product(*(tags[child] for child in node.children)):
                if normal_form.admits(rule, child_tags, inputs, output):
                    held.add(normal_form.tag(rule, child_tags))
        tags[node_id] = held

    return bool(tags[tree.root])


def corpus_digest(examples: tuple[Example[GoldDerivation[str]], ...], /) -> str:
    """Return canonical source-analysis bytes as a SHA-256 digest."""
    records = tuple(
        (
            str(example.item.item_id),
            tuple(token.surface for token in example.item.tokens),
            tuple(
                (
                    node_id,
                    node.span.start,
                    node.span.end,
                    node.label,
                    node.rule,
                    node.children,
                )
                for node_id, node in sorted(
                    example.supervision.derivation.nodes.items()
                )
            ),
        )
        for example in examples
    )
    encoded = dumps(records, separators=(",", ":"), ensure_ascii=True).encode()
    return sha256(encoded).hexdigest()


def lexical_counts(
    examples: tuple[Example[GoldDerivation[str]], ...], /
) -> Counter[tuple[str, str]]:
    """Count observed form-category pairs."""
    leaves: Counter[tuple[str, str]] = Counter()
    for example in examples:
        derivation = example.supervision.derivation
        by_position = {
            node.span.start: node.label
            for node in derivation.nodes.values()
            if node.is_lexical
        }
        for token in example.item.tokens:
            leaves[token.surface, by_position[token.index]] += 1
    return leaves


def _inventory(
    grammar: GrammarAssembly, categories: tuple[str, ...]
) -> InventoryReport:
    """Measure the category inventory present in the loaded examples."""
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

    vocabulary = grammar.store.signature.vocabulary(
        grammar.language.category_sort
    ).names
    measured = tuple(zip(categories, terms, strict=True))
    measure = grammar.category_measure
    return InventoryReport(
        {category: measure.depth(term) for category, term in measured},
        {category: measure.arity(term) for category, term in measured},
        {category: measure.order(term) for category, term in measured},
        tuple(name for name in vocabulary if name in occurring),
        max_arguments=grammar.max_category_arguments,
        max_order=grammar.max_category_order,
    )


def _knobs(grammar: GrammarAssembly) -> GrammarKnobs:
    """Read the grammar dimensions from the compiled grammar."""
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
        primitives,
        constructors,
        tuple(rule.name for rule in compiled.rules),
    )


def _reports(
    examples: tuple[Example[GoldDerivation[str]], ...],
    grammar: GrammarAssembly,
) -> tuple[AmbiguityReport, LexiconReport, InventoryReport]:
    """Measure ambiguity, lexical frequency and category inventory."""
    leaves = lexical_counts(examples)
    forms: dict[str, None] = {}
    categories: dict[str, None] = {}
    by_form: dict[str, set[str]] = {}

    for example in examples:
        derivation = example.supervision.derivation
        by_position = {
            node.span.start: node.label
            for node in derivation.nodes.values()
            if node.is_lexical
        }
        for token in example.item.tokens:
            label = by_position[token.index]
            forms.setdefault(token.surface, None)
            categories.setdefault(label, None)
            by_form.setdefault(token.surface, set()).add(label)

    form_names = tuple(forms)
    category_names = tuple(categories)
    form_counts = tuple(
        sum(leaves[form, category] for category in by_form[form]) for form in form_names
    )
    ambiguities = tuple(len(by_form[form]) for form in form_names)
    classes = {
        category: Histogram.of(
            leaves[form, category] for form in form_names if category in by_form[form]
        )
        for category in category_names
    }
    lexical = tuple(
        len(by_form[token.surface])
        for example in examples
        for token in example.item.tokens
    )
    return (
        AmbiguityReport(
            Histogram.of(lexical),
            Histogram.of(1 for _ in examples),
            rejected=0,
            target=None,
        ),
        LexiconReport(
            classes,
            Histogram.of(form_counts),
            Histogram.of(ambiguities),
            spearman(
                tuple(float(value) for value in ambiguities),
                tuple(float(value) for value in form_counts),
            ),
        ),
        _inventory(grammar, category_names),
    )


def _canonical_rule(
    source_rule: str | None,
    recovered: tuple[Rule[Term], ...],
    heads: Mapping[str, int],
    selections: Mapping[str, HeadSelection],
    /,
) -> str | None:
    """Return the one compiled rule identified by a corpus-local step.

    A corpus classifies derivation steps with its own rule names. The lowered
    dataset instead records the identity of the compiled rule that licenses the
    step, choosing among several recovered rules by the head slot the corpus's
    rule name selects. An unrecoverable step has no compiled rule identity.
    """
    if not recovered:
        return None
    selection = selections.get(source_rule or "")
    candidates = tuple(
        rule
        for rule in recovered
        if selection is None or heads.get(rule.name) == selection.slot
    )
    if len(candidates) != 1:
        names = tuple(rule.name for rule in recovered)
        msg = (
            f"corpus rule {source_rule!r} identifies {len(candidates)} compiled "
            f"rules among {names!r}"
        )
        raise ValueError(msg)
    return candidates[0].name


def _rule_heads(grammar: GrammarAssembly, /) -> Mapping[str, int]:
    """Return the head slot the grammar's dependency interpretation gives each rule."""
    dependency = grammar.grammar.interpretations.get("dependency")
    if dependency is None:
        msg = (
            "a corpus that selects rules by head declares a grammar with a "
            "dependency interpretation"
        )
        raise ValueError(msg)
    raw_heads = dependency.settings.get("heads")
    if not isinstance(raw_heads, RuntimeMapping):
        msg = "the grammar's dependency interpretation declares no rule heads"
        raise TypeError(msg)
    return cast("Mapping[str, int]", raw_heads)


def _canonical_examples(
    examples: tuple[Example[GoldDerivation[str]], ...],
    grammar: GrammarAssembly,
    selections: Mapping[str, HeadSelection],
    /,
) -> tuple[Example[GoldDerivation[str]], ...]:
    """Lower corpus-local step classes to compiled grammar rule identities."""
    heads = _rule_heads(grammar) if selections else MappingProxyType({})
    lowered: list[Example[GoldDerivation[str]]] = []

    for example in examples:
        derivation = example.supervision.derivation
        terms = {
            node_id: grammar.syntax.parse(grammar.store, node.label)
            for node_id, node in derivation.nodes.items()
        }
        nodes: dict[int, DerivationNode[str]] = {}
        for node_id, node in derivation.nodes.items():
            rule = None
            if not node.is_lexical:
                rule = _canonical_rule(
                    node.rule,
                    grammar.grammar.rules.recover(
                        tuple(terms[child] for child in node.children),
                        terms[node_id],
                    ),
                    heads,
                    selections,
                )
            nodes[node_id] = DerivationNode(
                node.span,
                node.label,
                rule=rule,
                children=node.children,
                features=node.features,
            )
        lowered.append(
            Example(
                example.item,
                GoldDerivation(
                    Derivation(
                        derivation.root,
                        nodes,
                        confidences={
                            node_id: confidence
                            for node_id in derivation.nodes
                            if (confidence := derivation.confidence_of(node_id))
                            is not None
                        },
                    )
                ),
            )
        )

    return tuple(lowered)


def gold_corpus(
    examples: tuple[Example[GoldDerivation[str]], ...],
    grammar: GrammarAssembly,
    read: LoadReport,
    /,
    *,
    seed: int,
    selections: Mapping[str, HeadSelection],
) -> GoldCorpus:
    """Build the gold sentence corpus of a treebank's read examples.

    Parameters
    ----------
    examples : tuple[Example[GoldDerivation[str]], ...]
        The reader's gold derivations, with corpus-local step names.
    grammar : GrammarAssembly
        The compiled grammar the corpus is read under.
    read : LoadReport
        The reader's record, emission and skip counts.
    seed : int
        The seed recorded on the corpus.
    selections : Mapping[str, HeadSelection]
        The head slot each corpus rule name selects among recovered rules.

    Returns
    -------
    GoldCorpus
        The canonical corpus, its load report and its source digest.

    Raises
    ------
    ValueError
        If the reader emitted no example or a step names more or fewer than
        one compiled rule among several recovered.
    """
    canonical = _canonical_examples(examples, grammar, selections)
    if not canonical:
        msg = "a treebank emitted no examples"
        raise ValueError(msg)

    ambiguity, lexicon, inventory = _reports(canonical, grammar)
    lengths = tuple(len(example.item.tokens) for example in canonical)
    grammar_inventory = GrammarInventory(grammar)
    reachability = reachability_report(
        grammar_inventory,
        grammar_inventory,
        (example.supervision.derivation for example in canonical),
    )
    report = LoadReport(
        read.records_seen,
        read.emitted,
        read.skips_by_reason,
        AnalysisCounts(
            reachability.checked_derivations,
            reachability.reachable_derivations,
            sum(_in_normal_form(example.supervision, grammar) for example in canonical),
        ),
        reachability,
    )
    corpus = SyntheticCorpus(
        grammar.declaration,
        grammar.grammar.theory_specification,
        seed=seed,
        length=LengthReport(
            Histogram.of(lengths),
            resampled=0,
            distribution=UniformLength(min(lengths), max(lengths)),
        ),
        knobs=_knobs(grammar),
        examples=canonical,
        ambiguity=ambiguity,
        lexicon=lexicon,
        inventory=inventory,
        labels=tuple(
            ObservedLabel.of(example.item.item_id, "grammatical", flipped=False)
            for example in canonical
        ),
    )
    return GoldCorpus(corpus, report, corpus_digest(canonical))
