"""Generative packed-chart parsing."""

from typing import TYPE_CHECKING, Final, Literal, cast, overload

from lofi._checks import ensure_non_negative, ensure_positive
from lofi._native import ChartProgram, MissBatch, RuleProgram
from lofi.deduce._forest import PackedForest
from lofi.deduce._inside import InsideChart, _ValueTable
from lofi.deduce._outside import goal_partition
from lofi.deduce._policy import (
    CKY,
    CONTIGUOUS_SPANS,
    NATIVE_CHARTS,
    ChartStore,
    InsideSweep,
    ParsePolicy,
    ParseStrategy,
    Reachability,
    Retain,
    SpanIndexFactory,
)
from lofi.deduce._ports import NamedRoots, NativeTermRule
from lofi.deduce._values import Edge, ItemKey, PrunedMass
from lofi.judge import GenerativeParsing

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.deduce._ports import (
        CategoryMeasure,
        CellPruner,
        EdgeScorer,
        Goal,
        Normalizer,
        Proposer,
        Semiring,
    )
    from lofi.deduce._spans import Cell, SpanIndex
    from lofi.judge import NormalFormDecision, NormalFormDecisionTable, Rule, RuleSet
    from lofi.lexicon import Entry, LexicalSource
    from lofi.terms import Term, TermMatching

__all__ = ["LEXICAL_TAG_PREFIX", "ParsePolicy", "Reachability", "Retain", "parse"]

LEXICAL_TAG_PREFIX: Final = "lex:"
"""Prefix reserved for lexical edge names."""

_MIN_NONUNARY_ARITY: Final = 2
"""Smallest arity handled by the span-decomposition sweep."""


def _unique[T](terms: Iterable[T]) -> tuple[T, ...]:
    """Keep the first occurrence of each exact term, in order."""
    return tuple(dict.fromkeys(terms))


def _proposed[T](
    rule: Rule[T],
    inputs: tuple[T, ...],
    span: Cell,
    proposer: Proposer[T] | None,
    admissions: dict[tuple[str, tuple[T, ...], T], bool],
) -> tuple[T, ...]:
    """Return proposed conclusions admitted once per exact rule application."""
    if proposer is None or "check" not in rule.directions:
        return ()

    admitted: list[T] = []

    for candidate in _unique(proposer.propose(rule, inputs, span)):
        key = (rule.name, inputs, candidate)
        accepted = admissions.get(key)

        if accepted is None:
            accepted = rule.admits(inputs, candidate)
            admissions[key] = accepted

        if accepted:
            admitted.append(candidate)

    return tuple(admitted)


def _normalized[T](
    outputs: tuple[T, ...],
    normalizer: Normalizer[T] | None,
) -> tuple[T, ...]:
    """Normalize outputs and remove the duplicates normalization made."""
    if normalizer is None:
        return outputs

    return _unique(normalizer.normalize(output) for output in outputs)


type _UnaryEdge[T] = tuple[ItemKey[T], Rule[T], ItemKey[T]]
"""A unary step's premise item, its rule and its conclusion item."""

type _NativeItem = tuple[int, int]
type _NativeClosure = tuple[
    list[tuple[_NativeItem, int, _NativeItem]], list[_NativeItem], bool
]


class _TagTable:
    """Dense normal form tags derived from the normal form's alphabet."""

    __slots__ = ("_ids", "_tags")

    def __init__(self, tags: Iterable[str]) -> None:
        held = tuple(sorted(set(tags)))
        self._tags = held
        self._ids = {tag: at for at, tag in enumerate(held)}

    def identify(self, tag: str) -> int:
        """Return one declared tag's dense identifier."""
        try:
            return self._ids[tag]
        except KeyError as error:
            msg = f"normal-form tag {tag!r} is outside the declared alphabet"
            raise ValueError(msg) from error

    def lookup(self, tag: str) -> int | None:
        """Return one declared tag's identifier when it exists."""
        return self._ids.get(tag)

    def tag(self, identifier: int) -> str:
        """Return the declared tag at one dense identifier."""
        return self._tags[identifier]

    def value(self, identifier: int) -> str:
        """Return the declared tag at one dense identifier."""
        return self.tag(identifier)

    def __len__(self) -> int:
        """Return the declared tag count."""
        return len(self._tags)


class _CategoryTable[T]:
    """Expose the native program's dense exact-category inventory."""

    __slots__ = ("_program",)

    def __init__(self, program: ChartProgram) -> None:
        self._program = program

    def identify(self, term: T) -> int | None:
        """Return the category id, or refuse it under the declared bounds."""
        return self._program.identify(cast("Term", term))

    @property
    def refused_categories(self) -> int:
        """Return how many exact categories the declared bounds refused."""
        return self._program.refused_categories

    def term(self, identifier: int) -> T:
        """Return one registered exact category."""
        return cast("T", self._program.term(identifier))

    def lookup(self, term: T) -> int | None:
        """Return one registered category's identifier."""
        return self._program.lookup(cast("Term", term))

    def value(self, identifier: int) -> T:
        """Return one registered exact category."""
        return self.term(identifier)


class _Resolver[T, W]:
    """Resolve native memo misses through the declared Python rule protocols."""

    __slots__ = (
        "_admissions",
        "_categories",
        "_decisions",
        "_forest",
        "_normal_form",
        "_normalizer",
        "_program",
        "_proposer",
        "_prune_lexical",
        "_pruner",
        "_roots",
        "_rule_program",
        "_rules",
        "_scorer",
        "_tags",
        "_tokens",
    )

    def __init__(
        self,
        rules: RuleSet[T],
        program: ChartProgram,
        categories: _CategoryTable[T],
        tags: _TagTable,
        proposer: Proposer[T] | None,
        normalizer: Normalizer[T] | None,
        scorer: EdgeScorer[T, W],
        rule_program: RuleProgram,
        decisions: NormalFormDecisionTable,
    ) -> None:
        self._rules = rules.rules
        self._rule_program = rule_program
        self._program = program
        self._categories = categories
        self._tags = tags
        self._proposer = proposer
        self._admissions: dict[tuple[str, tuple[T, ...], T], bool] = {}
        self._normalizer = normalizer
        self._scorer = scorer
        self._normal_form = rules.normal_form
        self._decisions = decisions
        self._pruner: CellPruner[T, W] | None = None
        self._roots: NamedRoots[T] | None = None
        self._prune_lexical = False
        self._tokens = 0
        self._forest: PackedForest[T, W] | None = None

    def configure_forest(
        self,
        forest: PackedForest[T, W],
        pruner: CellPruner[T, W] | None,
        roots: NamedRoots[T] | None,
        *,
        prune_lexical: bool,
    ) -> None:
        """Declare the forest and per-cell restrictions of one native sweep."""
        self._forest = forest
        self._tokens = forest.index.n
        self._pruner = pruner
        self._roots = roots
        self._prune_lexical = prune_lexical

    def identify(self, term: T) -> int | None:
        """Register one conclusion or refuse it under the declared bounds."""
        return self._categories.identify(term)

    def admit_rows(
        self,
        rule_id: int,
        tag_ids: list[int],
        input_id_pairs: list[tuple[int, int]],
        output_id: int,
    ) -> tuple[bool, ...]:
        """Decide exact premise-sensitive rows sharing a rule, tags, and output."""
        rule = self._rules[rule_id]
        children = tuple(self._tags.tag(identifier) for identifier in tag_ids)
        output = self._categories.term(output_id)

        return tuple(
            self._normal_form.admits(
                rule,
                children,
                tuple(self._categories.term(identifier) for identifier in input_ids),
                output,
            )
            for input_ids in input_id_pairs
        )

    def resolve(self, batch: MissBatch) -> None:
        """Resolve and push every admitted conclusion of one native miss."""
        categories = self._categories
        tags = self._tags
        inputs = tuple(categories.term(identifier) for identifier in batch.inputs)
        child_tags = tuple(tags.tag(identifier) for identifier in batch.tags)
        span = batch.span
        input_ids = batch.inputs

        prepared: list[tuple[int, Rule[T], NormalFormDecision, list[int] | None]] = []
        native_candidates: list[int] = []

        for position in batch.candidates:
            rule = self._rules[position]
            decision = self._decisions[position][child_tags]

            if decision == "refuse":
                continue
            complete = "forward" in rule.directions and rule.forward_is_complete
            cached = (
                self._program.cached_outputs(position, input_ids) if complete else None
            )

            prepared.append((position, rule, decision, cached))
            if cached is None and "forward" in rule.directions:
                native_candidates.append(position)

        native_outputs: dict[int, tuple[T, ...]] = {
            position: cast("tuple[T, ...]", tuple(outputs))
            for position, outputs in self._rule_program.apply(native_candidates, inputs)
        }

        for position, rule, decision, cached in prepared:
            output_tag = self._normal_form.tag(rule, child_tags)
            tag_id = tags.identify(output_tag)
            complete = "forward" in rule.directions and rule.forward_is_complete

            if cached is None:
                outputs = _normalized(
                    _unique(
                        (
                            *native_outputs.get(position, ()),
                            *(
                                ()
                                if complete
                                else _proposed(
                                    rule,
                                    inputs,
                                    span,
                                    self._proposer,
                                    self._admissions,
                                )
                            ),
                        )
                    ),
                    self._normalizer,
                )
                output_ids = [
                    identifier
                    for output in outputs
                    if (identifier := categories.identify(output)) is not None
                ]

                if complete:
                    self._program.remember_outputs(position, input_ids, output_ids)
            else:
                output_ids = cached

            for output_id in output_ids:
                output = categories.term(output_id)

                if not self._program.reachable(span, output_id):
                    continue
                if decision != "admit" and not self._normal_form.admits(
                    rule, child_tags, inputs, output
                ):
                    continue

                batch.push(position, output_id, tag_id)

    def weight(
        self,
        rule: int,
        inputs: list[int],
        output: int,
        span: Cell,
        split: int | None,
    ) -> W:
        """Return one dense-id rule application's local weight."""
        categories = self._categories
        return self._scorer.edge_weight(
            self._rules[rule],
            tuple(categories.term(identifier) for identifier in inputs),
            categories.term(output),
            span,
            split,
        )

    def root_keep(
        self,
        span: Cell,
        items: list[tuple[int, int, W]],
        closure: _NativeClosure,
    ) -> list[_NativeItem] | None:
        """Return the root-reaching items when the root cell is restricted."""
        roots = self._roots

        if roots is None or span != (0, self._tokens):
            return None

        categories = self._categories
        tags = self._tags

        def key(item: _NativeItem) -> ItemKey[T]:
            category, tag = item
            return ItemKey(span, categories.term(category), tags.tag(tag))

        nodes = tuple(key((category, tag)) for category, tag, _ in items)
        edges = tuple(
            (key(child), self._rules[rule], key(output))
            for child, rule, output in closure[0]
        )
        reached = _reaching(nodes, edges, roots)
        return [
            (category, tag)
            for category, tag, _ in items
            if key((category, tag)) in reached
        ]

    def prune_keep(
        self, span: Cell, items: list[tuple[int, int, W]]
    ) -> list[_NativeItem] | None:
        """Return the dense items selected by the declared cell pruner.

        The forest records the summed weight of the items the pruner dropped.
        """
        pruner = self._pruner
        forest = self._forest

        if (
            pruner is None
            or forest is None
            or (not self._prune_lexical and span[1] - span[0] == 1)
        ):
            return None

        categories = self._categories
        tags = self._tags
        keyed = tuple(
            (
                ItemKey(span, categories.term(category), tags.tag(tag)),
                weight,
            )
            for category, tag, weight in items
        )
        identifiers = {
            weighted[0]: (category, tag)
            for weighted, (category, tag, _) in zip(keyed, items, strict=True)
        }
        kept = tuple(pruner.keep(span, keyed))
        unknown = tuple(key for key in kept if key not in identifiers)

        if unknown:
            names = ", ".join(repr(key) for key in unknown)
            msg = f"span {span!r} holds no item {names}"
            raise ValueError(msg)

        retained = frozenset(kept)
        dropped = tuple(weight for key, weight in keyed if key not in retained)

        if dropped:
            semiring = forest.semiring
            total = dropped[0]

            for weight in dropped[1:]:
                total = semiring.plus(total, weight)

            forest.record_pruned(span, total)

        return [identifiers[key] for key in kept]


class _Kernel[T, W]:
    """The native rule-drawing kernel one parse shares across its cells.

    The steps a premise tuple licenses depend only on the exact premises and
    their normal form tags whenever every candidate rule enumerates its
    conclusions completely, so they are drawn once per such tuple and reused at
    every span and split it recurs at. A tuple whose candidates include a
    checked rule is drawn afresh, since its proposals depend on the span.

    Parameters
    ----------
    rules : RuleSet[T]
        The grammar's rules, their matching and their normal form.
    proposer : Proposer[T] | None
        Candidate conclusions for check-only rules.
    normalizer : Normalizer[T] | None
        The normal form conclusions take before insertion.
    """

    __slots__ = (
        "_categories",
        "_labels",
        "_native_rules",
        "_program",
        "_resolver",
        "_rules",
        "_scoring",
        "_tags",
    )

    def __init__(
        self,
        rules: RuleSet[T],
        proposer: Proposer[T] | None,
        normalizer: Normalizer[T] | None,
        scorer: EdgeScorer[T, W],
        category_measure: CategoryMeasure[T] | None,
        max_category_arity: int | None,
        max_category_size: int | None,
    ) -> None:
        self._rules = rules
        native_rules = tuple(rule for rule in rules if isinstance(rule, NativeTermRule))
        if len(native_rules) != len(rules):
            missing = tuple(
                rule.name for rule in rules if not isinstance(rule, NativeTermRule)
            )
            msg = f"native chart rules lack compiled term capabilities: {missing!r}"
            raise TypeError(msg)
        self._native_rules = native_rules
        self._scoring = scorer.lowering
        normal_form = rules.normal_form
        lexical_tag = normal_form.lexical_tag()
        self._tags = _TagTable((*normal_form.alphabet, lexical_tag))
        self._labels = _ValueTable[str]()

        for rule in rules:
            self._labels.identify(rule.name)

        rule_program = RuleProgram([rule.lowering.native for rule in native_rules])
        decisions = rules.normal_form.decisions(rules.rules)
        rule_rows = []

        for position, rule in enumerate(rules):
            row = [
                (
                    tuple(self._tags.identify(tag) for tag in children),
                    decision,
                    self._tags.identify(rules.normal_form.tag(rule, children)),
                )
                for children, decision in decisions[position].items()
            ]
            rule_rows.append(
                (
                    rule.arity,
                    rule.spine_order,
                    "forward" in rule.directions and rule.forward_is_complete,
                    row,
                )
            )

        self._program = ChartProgram(
            rule_rows,
            len(self._tags),
            rule_program,
            normalizes=normalizer is not None,
            category_measure=(
                () if category_measure is None else category_measure.constructors
            ),
            category_bounds=(max_category_arity, max_category_size),
        )
        self._categories = _CategoryTable(self._program)
        self._resolver = _Resolver(
            rules,
            self._program,
            self._categories,
            self._tags,
            proposer,
            normalizer,
            scorer,
            rule_program,
            decisions,
        )

    def sweep_inside(
        self,
        chart: InsideChart[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
    ) -> InsideSweep[T]:
        """Fill one inside chart and decode its root unary closure."""
        native = chart.sweep(
            active_arities,
            unary_depth,
            threads,
            self._program,
            self._resolver,
            scoring=self._scoring,
        )
        return self._decode_closure((0, chart.index.n), native)

    def configure_reachability(
        self,
        tokens: int,
        roots: NamedRoots[Term],
        lexical: Mapping[Cell, tuple[Term, ...]],
        unary_depth: int,
        refinement_pairs: int,
    ) -> None:
        """Compile and install one parse's root-directed category demands."""
        rows = tuple(
            (rule.arity, native.reachability_schemata, introduces, consumes)
            for rule, native, introduces, consumes in zip(
                self._rules,
                self._native_rules,
                self._rules.introduces,
                self._rules.consumes,
                strict=True,
            )
        )
        if any(
            not schemata and introduces is None for _, schemata, introduces, _ in rows
        ):
            return
        self._program.configure_reachability(
            roots.categories[0],
            cast("TermMatching", roots.matching).representatives,
            (tokens, unary_depth, refinement_pairs),
            roots.categories,
            tuple(sorted(lexical.items())),
            rows,
        )

    def reachable(self, span: Cell, category: T, /) -> bool:
        """Return whether one category remains root-reachable at a span."""
        identifier = self._categories.identify(category)
        return identifier is not None and self._program.reachable(span, identifier)

    def inside_chart(
        self,
        index: SpanIndex,
        semiring: Semiring[W],
        *,
        normal_form: str,
    ) -> InsideChart[T, W]:
        """Construct an inside chart sharing this kernel's dense identifiers."""
        return InsideChart(
            index,
            semiring,
            normal_form=normal_form,
            _identifiers=(self._categories, self._tags),
        )

    def forest_chart(
        self,
        index: SpanIndex,
        semiring: Semiring[W],
        pruner: CellPruner[T, W] | None,
        roots: NamedRoots[T] | None,
        *,
        normal_form: str,
        prune_lexical: bool,
    ) -> PackedForest[T, W]:
        """Construct a forest sharing identifiers and cell restrictions."""
        forest = PackedForest(
            index,
            semiring,
            normal_form=normal_form,
            _identifiers=(self._categories, self._tags, self._labels),
        )
        self._resolver.configure_forest(
            forest, pruner, roots, prune_lexical=prune_lexical
        )
        return forest

    def sweep_forest(
        self,
        chart: PackedForest[T, W],
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
    ) -> None:
        """Fill one packed forest through the native kernel."""
        chart.sweep(
            active_arities,
            unary_depth,
            threads,
            self._program,
            self._resolver,
            scoring=self._scoring,
        )

    def _decode_closure(
        self,
        span: Cell,
        native: tuple[
            list[tuple[tuple[int, int], int, tuple[int, int]]],
            list[tuple[int, int]],
            bool,
        ],
    ) -> InsideSweep[T]:
        """Map one native dense-id closure back to chart values."""
        categories = self._categories
        tags = self._tags
        edges, concluded, truncated = native
        return InsideSweep(
            tuple(
                (
                    ItemKey(span, categories.term(child[0]), tags.tag(child[1])),
                    self._rules.rules[rule],
                    ItemKey(span, categories.term(output[0]), tags.tag(output[1])),
                )
                for child, rule, output in edges
            ),
            tuple(
                ItemKey(span, categories.term(category), tags.tag(tag))
                for category, tag in concluded
            ),
            truncated,
        )


def _reaching[T](
    nodes: Iterable[ItemKey[T]],
    edges: tuple[_UnaryEdge[T], ...],
    roots: NamedRoots[T],
) -> frozenset[ItemKey[T]]:
    """Return the nodes a root is, or a chain of unary steps leads from to one."""
    matching = roots.matching
    kept = {
        node
        for node in nodes
        if any(matching.equivalent(node.reading, root) for root in roots.categories)
    }
    grown = True

    while grown:
        grown = False

        for child, _, key in edges:
            if key in kept and child not in kept:
                kept.add(child)
                grown = True

    return frozenset(kept)


def _validate_rule_names[T](rules: RuleSet[T]) -> None:
    """Reject names reserved for lexical edges."""
    collisions = tuple(
        rule.name for rule in rules if rule.name.startswith(LEXICAL_TAG_PREFIX)
    )

    if collisions:
        names = ", ".join(repr(name) for name in collisions)
        msg = (
            f"rule names use the reserved lexical prefix "
            f"{LEXICAL_TAG_PREFIX!r}: {names}"
        )
        raise ValueError(msg)


@overload
def parse[T, W](
    sentence: Sequence[str],
    *,
    rules: RuleSet[T],
    source: LexicalSource[T, W],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    goal: Goal[T],
    category_measure: CategoryMeasure[T] | None = None,
    max_category_arity: int | None = None,
    max_category_size: int | None = None,
    proposer: Proposer[T] | None = None,
    normalizer: Normalizer[T] | None = None,
    unary_depth: int,
    threads: int = 1,
    pruner: CellPruner[T, W] | None = None,
    prune_lexical: bool = False,
    measure_pruned_mass: bool = False,
    strategy: ParseStrategy = CKY,
    span_index: SpanIndexFactory = CONTIGUOUS_SPANS,
    chart_store: ChartStore = NATIVE_CHARTS,
    policy: ParsePolicy[Literal[Retain.FOREST]],
) -> PackedForest[T, W]: ...


@overload
def parse[T, W](
    sentence: Sequence[str],
    *,
    rules: RuleSet[T],
    source: LexicalSource[T, W],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    goal: Goal[T],
    category_measure: CategoryMeasure[T] | None = None,
    max_category_arity: int | None = None,
    max_category_size: int | None = None,
    proposer: Proposer[T] | None = None,
    normalizer: Normalizer[T] | None = None,
    unary_depth: int,
    threads: int = 1,
    pruner: None = None,
    prune_lexical: bool = False,
    measure_pruned_mass: bool = False,
    strategy: ParseStrategy = CKY,
    span_index: SpanIndexFactory = CONTIGUOUS_SPANS,
    chart_store: ChartStore = NATIVE_CHARTS,
    policy: ParsePolicy[Literal[Retain.INSIDE]],
) -> InsideChart[T, W]: ...


@overload
def parse[T, W](
    sentence: Sequence[str],
    *,
    rules: RuleSet[T],
    source: LexicalSource[T, W],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    goal: Goal[T],
    category_measure: CategoryMeasure[T] | None = None,
    max_category_arity: int | None = None,
    max_category_size: int | None = None,
    proposer: Proposer[T] | None = None,
    normalizer: Normalizer[T] | None = None,
    unary_depth: int,
    threads: int = 1,
    pruner: CellPruner[T, W] | None = None,
    prune_lexical: bool = False,
    measure_pruned_mass: bool = False,
    strategy: ParseStrategy = CKY,
    span_index: SpanIndexFactory = CONTIGUOUS_SPANS,
    chart_store: ChartStore = NATIVE_CHARTS,
    policy: ParsePolicy[Retain],
) -> PackedForest[T, W] | InsideChart[T, W]: ...


def parse[T, W](
    sentence: Sequence[str],
    *,
    rules: RuleSet[T],
    source: LexicalSource[T, W],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    goal: Goal[T],
    category_measure: CategoryMeasure[T] | None = None,
    max_category_arity: int | None = None,
    max_category_size: int | None = None,
    proposer: Proposer[T] | None = None,
    normalizer: Normalizer[T] | None = None,
    unary_depth: int,
    threads: int = 1,
    pruner: CellPruner[T, W] | None = None,
    prune_lexical: bool = False,
    measure_pruned_mass: bool = False,
    strategy: ParseStrategy = CKY,
    span_index: SpanIndexFactory = CONTIGUOUS_SPANS,
    chart_store: ChartStore = NATIVE_CHARTS,
    policy: ParsePolicy[Retain],
) -> PackedForest[T, W] | InsideChart[T, W]:
    """Parse a token sequence into a packed forest or an inside chart.

    Parameters
    ----------
    sentence : Sequence[str]
        The tokens to parse.
    rules : RuleSet[T]
        The grammar's rules, their matching and their normal form.
    source : LexicalSource[T, W]
        The lexical axioms each position admits.
    semiring : Semiring[W]
        The algebra inside weights accumulate in.
    scorer : EdgeScorer[T, W]
        The local weight of each axiom and each rule application.
    goal : Goal[T]
        The goal the parse is read under. The sweep reads its root categories
        only to keep the items that can reach one.
    category_measure : CategoryMeasure[T] | None
        Owner-declared category constructors and their structural measures.
        A measure is required when either category bound is declared.
    max_category_arity : int | None
        Greatest admitted functor spine arity, or no arity bound.
    max_category_size : int | None
        Greatest admitted exact category node count, or no size bound.
    proposer : Proposer[T] | None
        Candidate conclusions for check-only rules, or None where the grammar
        needs none.
    normalizer : Normalizer[T] | None
        The normal form rule outputs take before insertion, or None.
    unary_depth : int
        How many rounds of unary closure a cell runs.
    threads : int
        Number of cells within one span width the native kernel may run
        concurrently.
    pruner : CellPruner[T, W] | None
        The items each cell keeps once it has closed, or None to keep them all,
        which leaves the forest exhaustive. Only a forest is pruned.
    prune_lexical : bool
        Whether one-token cells are pruned too. They are not by default, so a
        beam never removes a lexical reading the source admitted.
    measure_pruned_mass : bool
        Whether to reparse the sentence without the pruner and record, on the
        pruned forest, the goal probability mass pruning removed. Only a
        pruned forest under log probability weights is measured.
    strategy : ParseStrategy
        Traversal strategy that runs the chart kernel.
    span_index : SpanIndexFactory
        Factory for the parse's region index.
    chart_store : ChartStore
        Factory for the chart storage selected by retention.
    policy : ParsePolicy[Retain]
        The declared retention and reachability choices. A forest keeps every
        item and step; an inside chart keeps exact-category inside weights. Root
        reachability constructs only items that can participate in a named root
        derivation, planned under the policy's refinement bound.

    Returns
    -------
    PackedForest[T, W] | InsideChart[T, W]
        The forest or the inside chart, marked truncated if closure or pruning
        dropped anything. A pruned forest also records, per cell, the weight
        its pruner removed and, when measured, the goal mass it removed.

    Raises
    ------
    ValueError
        If a pruner is given for an inside chart, which has no items to keep,
        reachability is asked of a goal that names no root categories, or
        pruned mass is to be measured without a pruner or log probability weights.
    """
    rules.validate(GenerativeParsing(proposer_available=proposer is not None))
    _validate_rule_names(rules)

    unary_depth = ensure_non_negative("unary_depth", unary_depth)
    threads = ensure_positive("threads", threads)
    max_category_arity = (
        None
        if max_category_arity is None
        else ensure_positive("max_category_arity", max_category_arity)
    )
    max_category_size = (
        None
        if max_category_size is None
        else ensure_positive("max_category_size", max_category_size)
    )

    if (
        max_category_arity is not None or max_category_size is not None
    ) and category_measure is None:
        msg = "category bounds need the grammar's declared category measure"
        raise ValueError(msg)

    retain = policy.retain
    reachability = policy.reachability

    if retain is Retain.INSIDE and pruner is not None:
        msg = "an inside chart keeps no items, so it cannot be pruned"
        raise ValueError(msg)

    if measure_pruned_mass and pruner is None:
        msg = (
            "measuring pruned mass measures the mass a pruner lost, and the "
            "parse declares no pruner"
        )
        raise ValueError(msg)

    if measure_pruned_mass and not semiring.log_probability:
        msg = (
            "measuring pruned mass sums goal probabilities, which needs "
            f"log-probability weights, which {semiring.name!r} does not carry"
        )
        raise ValueError(msg)

    roots: NamedRoots[T] | None = None
    if reachability is Reachability.ROOTS:
        if not isinstance(goal, NamedRoots):
            msg = "reachability needs a goal that names its root categories"
            raise ValueError(msg)

        roots = cast("NamedRoots[T]", goal)

    tokens = tuple(sentence)
    index = span_index.build(len(tokens))
    lexical_tag = rules.normal_form.lexical_tag()
    drawing = _Kernel(
        rules,
        proposer,
        normalizer,
        scorer,
        category_measure,
        max_category_arity,
        max_category_size,
    )
    active_arities = tuple(
        dict.fromkeys(rule.arity for rule in rules if rule.arity >= _MIN_NONUNARY_ARITY)
    )
    axioms = tuple(
        (
            position,
            entry,
            semiring.times(source_weight, scorer.axiom_weight(position, entry)),
        )
        for position, entry, source_weight in source.axioms(tokens)
    )
    normal_form = rules.normal_form.name

    if roots is not None and index.n > 0:
        lexical: dict[Cell, list[Term]] = {}

        for position, entry, _ in axioms:
            lexical.setdefault((position, position + 1), []).append(
                cast("Term", roots.matching.key(entry.category))
            )

        drawing.configure_reachability(
            index.n,
            cast("NamedRoots[Term]", roots),
            {span: tuple(categories) for span, categories in lexical.items()},
            unary_depth,
            policy.refinement_pairs,
        )

    if retain is Retain.INSIDE:
        inside = chart_store.inside(
            drawing,
            index,
            semiring,
            normal_form=normal_form,
        )

        for position, entry, weight in axioms:
            span = (position, position + 1)
            if roots is None or drawing.reachable(span, entry.category):
                inside.add(span, entry.category, lexical_tag, weight)

        closure = strategy.inside(
            drawing,
            inside,
            active_arities,
            unary_depth,
            threads,
        )

        if roots is not None and index.n > 0:
            span = (0, index.n)
            nodes = tuple(
                ItemKey(span, category, tag)
                for category, tag, _ in inside.tagged_at(span)
            )
            inside.restrict(
                span,
                (
                    (node.reading, node.nf_tag)
                    for node in _reaching(nodes, closure.edges, roots)
                ),
            )

        return inside

    sweep = _ForestSweep(
        drawing,
        index,
        semiring,
        roots,
        axioms,
        strategy,
        chart_store,
        normal_form=normal_form,
        lexical_tag=lexical_tag,
        active_arities=active_arities,
        unary_depth=unary_depth,
        threads=threads,
    )
    forest = sweep.fill(pruner, prune_lexical=prune_lexical)

    if measure_pruned_mass:
        exhaustive = sweep.fill(None, prune_lexical=False)
        forest.record_pruned_mass(
            PrunedMass(
                goal_partition(cast("PackedForest[T, float]", exhaustive), goal),
                goal_partition(cast("PackedForest[T, float]", forest), goal),
            )
        )

    return forest


class _ForestSweep[T, W]:
    """Fill packed forests for one sentence through one shared kernel."""

    __slots__ = (
        "_active_arities",
        "_axioms",
        "_chart_store",
        "_drawing",
        "_index",
        "_lexical_tag",
        "_normal_form",
        "_roots",
        "_semiring",
        "_strategy",
        "_threads",
        "_unary_depth",
    )

    def __init__(
        self,
        drawing: _Kernel[T, W],
        index: SpanIndex,
        semiring: Semiring[W],
        roots: NamedRoots[T] | None,
        axioms: tuple[tuple[int, Entry[T], W], ...],
        strategy: ParseStrategy,
        chart_store: ChartStore,
        *,
        normal_form: str,
        lexical_tag: str,
        active_arities: tuple[int, ...],
        unary_depth: int,
        threads: int,
    ) -> None:
        self._drawing = drawing
        self._index = index
        self._semiring = semiring
        self._roots = roots
        self._axioms = axioms
        self._strategy = strategy
        self._chart_store = chart_store
        self._normal_form = normal_form
        self._lexical_tag = lexical_tag
        self._active_arities = active_arities
        self._unary_depth = unary_depth
        self._threads = threads

    def fill(
        self, pruner: CellPruner[T, W] | None, *, prune_lexical: bool
    ) -> PackedForest[T, W]:
        """Return one forest swept under one cell pruner."""
        drawing = self._drawing
        roots = self._roots
        forest = self._chart_store.forest(
            drawing,
            self._index,
            self._semiring,
            pruner,
            roots,
            normal_form=self._normal_form,
            prune_lexical=prune_lexical,
        )

        for position, entry, weight in self._axioms:
            span = (position, position + 1)
            if roots is None or drawing.reachable(span, entry.category):
                forest.add(
                    ItemKey(span, entry.category, self._lexical_tag),
                    Edge(
                        f"{LEXICAL_TAG_PREFIX}{entry.form}",
                        (),
                        entry.category,
                        weight,
                    ),
                )

        self._strategy.forest(
            drawing, forest, self._active_arities, self._unary_depth, self._threads
        )

        return forest
