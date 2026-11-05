"""Definitional derivation-tree evaluation for chart tests.

The evaluator deliberately enumerates trees. It has no memo, join index,
closure table, or dependency-ordered recomputation in common with the chart
engine it checks.
"""

from itertools import product
from typing import TYPE_CHECKING, NamedTuple

if TYPE_CHECKING:
    from collections.abc import Iterable, Iterator, Sequence

    from lofi.deduce import Cell, EdgeScorer, Normalizer, Proposer, Semiring
    from lofi.judge import Rule, RuleSet
    from lofi.lexicon import LexicalSource

__all__ = ["ReferenceChart", "ReferenceTree", "reference_chart"]


class ReferenceTree[T, W](NamedTuple):
    """One explicitly enumerated derivation tree."""

    span: Cell
    category: T
    nf_tag: str
    weight: W
    rule_name: str
    children: tuple[ReferenceTree[T, W], ...]
    unary_path: tuple[T, ...]


class ReferenceChart[T, W]:
    """Exact category weights obtained by enumerating derivation trees."""

    __slots__ = ("_cells", "_n", "_semiring")

    def __init__(
        self,
        n: int,
        semiring: Semiring[W],
        cells: Iterable[tuple[Cell, tuple[ReferenceTree[T, W], ...]]],
    ) -> None:
        object.__setattr__(self, "_n", n)
        object.__setattr__(self, "_semiring", semiring)
        object.__setattr__(self, "_cells", tuple(cells))

    @property
    def n(self) -> int:
        """Return the token count."""
        return self._n

    def spans(self) -> Iterator[Cell]:
        """Yield every nonempty span in increasing-width order."""
        for width in range(1, self._n + 1):
            for start in range(self._n - width + 1):
                yield (start, start + width)

    def trees_at(self, span: Cell, /) -> tuple[ReferenceTree[T, W], ...]:
        """Return every derivation tree over one span."""
        return next((trees for held, trees in self._cells if held == span), ())

    def tagged_at(self, span: Cell, /) -> tuple[tuple[T, str, W], ...]:
        """Return weights by exact category and normal-form tag."""
        semiring = self._semiring
        weights: dict[tuple[T, str], W] = {}

        for tree in self.trees_at(span):
            key = (tree.category, tree.nf_tag)
            weights[key] = (
                semiring.plus(weights[key], tree.weight)
                if key in weights
                else tree.weight
            )

        return tuple(
            (category, tag, weight) for (category, tag), weight in weights.items()
        )

    def categories_at(self, span: Cell, /) -> tuple[tuple[T, W], ...]:
        """Return weights by exact category, summed across tags."""
        semiring = self._semiring
        weights: dict[T, W] = {}

        for category, _, weight in self.tagged_at(span):
            weights[category] = (
                semiring.plus(weights[category], weight)
                if category in weights
                else weight
            )

        return tuple(weights.items())

    def inside(self, span: Cell, category: T, /) -> W:
        """Return one exact category's weight over a span."""
        return dict(self.categories_at(span)).get(category, self._semiring.zero)


class _UnaryStep[T, W](NamedTuple):
    """One licensed unary edge and its local weight."""

    child: tuple[T, str]
    output: tuple[T, str]
    rule_name: str
    weight: W


def _spans(n: int) -> Iterator[Cell]:
    """Yield every nonempty span in increasing-width order."""
    for width in range(1, n + 1):
        for start in range(n - width + 1):
            yield (start, start + width)


def _cuts(span: Cell, arity: int) -> Iterator[tuple[Cell, ...]]:
    """Yield every ordered decomposition into nonempty adjacent parts."""
    start, end = span

    if arity == 1:
        yield (span,)
        return

    for split in range(start + 1, end - arity + 2):
        for rest in _cuts((split, end), arity - 1):
            yield ((start, split), *rest)


def _unique[T](values: Iterable[T]) -> tuple[T, ...]:
    """Keep exact values once, in first-occurrence order."""
    held: list[T] = []

    for value in values:
        if value not in held:
            held.append(value)

    return tuple(held)


def _outputs[T](
    rule: Rule[T],
    inputs: tuple[T, ...],
    span: Cell,
    proposer: Proposer[T] | None,
    normalizer: Normalizer[T] | None,
) -> tuple[T, ...]:
    """Enumerate a rule's exact conclusions through its declared directions."""
    drawn: list[T] = []

    if "forward" in rule.directions:
        drawn.extend(rule.apply(inputs))

    if (
        not rule.forward_is_complete
        and proposer is not None
        and "check" in rule.directions
    ):
        drawn.extend(
            output
            for output in proposer.propose(rule, inputs, span)
            if rule.admits(inputs, output)
        )

    normalized = (
        drawn
        if normalizer is None
        else [normalizer.normalize(output) for output in drawn]
    )

    return _unique(normalized)


def _rule_trees[T, W](
    span: Cell,
    parts: tuple[Cell, ...],
    rule: Rule[T],
    cells: dict[Cell, list[ReferenceTree[T, W]]],
    rules: RuleSet[T],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    proposer: Proposer[T] | None,
    normalizer: Normalizer[T] | None,
) -> Iterator[ReferenceTree[T, W]]:
    """Yield trees rooted in one nonunary rule over one decomposition."""
    choices = tuple(cells.get(part, []) for part in parts)

    if any(not trees for trees in choices):
        return

    normal_form = rules.normal_form
    split = parts[0][1] if rule.arity == 2 else None

    for children in product(*choices):
        inputs = tuple(child.category for child in children)
        tags = tuple(child.nf_tag for child in children)

        for output in _outputs(rule, inputs, span, proposer, normalizer):
            if not normal_form.admits(rule, tags, inputs, output):
                continue

            weight = scorer.edge_weight(rule, inputs, output, span, split)

            for child in children:
                weight = semiring.times(weight, child.weight)

            yield ReferenceTree(
                span,
                output,
                normal_form.tag(rule, tags),
                weight,
                rule.name,
                children,
                (output,),
            )


def _close_unary[T, W](
    span: Cell,
    base: list[ReferenceTree[T, W]],
    rules: RuleSet[T],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    proposer: Proposer[T] | None,
    normalizer: Normalizer[T] | None,
    unary_depth: int,
) -> None:
    """Append every tree licensed by the bounded acyclic unary graph."""
    unary = tuple(rule for rule in rules if rule.arity == 1)
    normal_form = rules.normal_form
    by_item: dict[tuple[T, str], list[ReferenceTree[T, W]]] = {}
    order: list[tuple[T, str]] = []

    for tree in base:
        item = (tree.category, tree.nf_tag)

        if item not in by_item:
            by_item[item] = []
            order.append(item)

        by_item[item].append(tree)

    frontier = tuple(order)
    scheduled = set(order)
    visited: set[tuple[T, str]] = set()
    category_graph: dict[T, set[T]] = {}
    item_graph: dict[tuple[T, str], set[tuple[T, str]]] = {
        item: set() for item in order
    }
    steps: list[_UnaryStep[T, W]] = []

    def reaches(start: T, target: T) -> bool:
        pending = [start]
        seen: set[T] = set()

        while pending:
            current = pending.pop()

            if current == target:
                return True

            if current not in seen:
                seen.add(current)
                pending.extend(category_graph.get(current, ()))

        return False

    for _ in range(unary_depth):
        following: list[tuple[T, str]] = []

        for child in frontier:
            if child in visited:
                continue

            visited.add(child)
            category, child_tag = child
            inputs = (category,)
            tags = (child_tag,)

            for rule in unary:
                for output in _outputs(rule, inputs, span, proposer, normalizer):
                    if reaches(output, category) or not normal_form.admits(
                        rule, tags, inputs, output
                    ):
                        continue

                    concluded = (output, normal_form.tag(rule, tags))
                    item_graph.setdefault(child, set()).add(concluded)
                    item_graph.setdefault(concluded, set())
                    category_graph.setdefault(category, set()).add(output)
                    steps.append(
                        _UnaryStep(
                            child,
                            concluded,
                            rule.name,
                            scorer.edge_weight(rule, inputs, output, span, None),
                        )
                    )

                    if concluded not in scheduled:
                        scheduled.add(concluded)
                        by_item[concluded] = []
                        order.append(concluded)
                        following.append(concluded)

        frontier = tuple(following)

        if not frontier:
            break

    indegree = dict.fromkeys(item_graph, 0)

    for outputs in item_graph.values():
        for output in outputs:
            indegree[output] += 1

    ready = [item for item in order if indegree[item] == 0]
    topological: list[tuple[T, str]] = []

    while ready:
        child = ready.pop(0)
        topological.append(child)

        for output in item_graph[child]:
            indegree[output] -= 1

            if indegree[output] == 0:
                ready.append(output)

    outgoing: dict[tuple[T, str], list[_UnaryStep[T, W]]] = {}

    for step in steps:
        outgoing.setdefault(step.child, []).append(step)

    for child in topological:
        for step in outgoing.get(child, ()):
            output, output_tag = step.output

            for tree in tuple(by_item[child]):
                by_item[step.output].append(
                    ReferenceTree(
                        span,
                        output,
                        output_tag,
                        semiring.times(step.weight, tree.weight),
                        step.rule_name,
                        (tree,),
                        (*tree.unary_path, output),
                    )
                )

    base[:] = [tree for item in order for tree in by_item[item]]


def reference_chart[T, W](
    sentence: Sequence[str],
    *,
    rules: RuleSet[T],
    source: LexicalSource[T, W],
    semiring: Semiring[W],
    scorer: EdgeScorer[T, W],
    proposer: Proposer[T] | None = None,
    normalizer: Normalizer[T] | None = None,
    unary_depth: int = 3,
) -> ReferenceChart[T, W]:
    """Enumerate every licensed derivation tree of a small sentence."""
    tokens = tuple(sentence)
    cells: dict[Cell, list[ReferenceTree[T, W]]] = {
        span: [] for span in _spans(len(tokens))
    }
    lexical_tag = rules.normal_form.lexical_tag()

    for position, entry, source_weight in source.axioms(tokens):
        span = (position, position + 1)
        weight = semiring.times(source_weight, scorer.axiom_weight(position, entry))
        cells[span].append(
            ReferenceTree(
                span,
                entry.category,
                lexical_tag,
                weight,
                f"lex:{entry.form}",
                (),
                (entry.category,),
            )
        )

    arities = tuple(dict.fromkeys(rule.arity for rule in rules if rule.arity >= 2))

    for span in _spans(len(tokens)):
        held = cells[span]

        for arity in arities:
            for parts in _cuts(span, arity):
                for rule in rules:
                    if rule.arity == arity:
                        held.extend(
                            _rule_trees(
                                span,
                                parts,
                                rule,
                                cells,
                                rules,
                                semiring,
                                scorer,
                                proposer,
                                normalizer,
                            )
                        )

        _close_unary(
            span,
            held,
            rules,
            semiring,
            scorer,
            proposer,
            normalizer,
            unary_depth,
        )

    return ReferenceChart(
        len(tokens),
        semiring,
        ((span, tuple(cells[span])) for span in _spans(len(tokens))),
    )
