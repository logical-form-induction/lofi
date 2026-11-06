"""Bounded unary closure over category and normal form states."""

from typing import TYPE_CHECKING, Never, final, override

import torch

from lofi._checks import ensure_non_negative
from lofi.nn.parsers._states import checked_state, distinct_outputs

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judge import Rule, RuleSet
    from lofi.nn.parsers._ports import LogWeightReduction

__all__ = ["UnaryClosure", "UnaryTransition"]


@final
class UnaryTransition:
    """One unary rule's category and normal form state transition."""

    __slots__ = ("_admitted", "_output", "_rule", "_source")

    def __init__(
        self,
        rule: int,
        source: tuple[int, int],
        output: tuple[int, int],
        *,
        admitted: bool,
    ) -> None:
        if not isinstance(admitted, bool):
            msg = f"transition admission is not Boolean: {admitted!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_rule", ensure_non_negative("rule index", rule))
        object.__setattr__(self, "_source", checked_state("source state", source))
        object.__setattr__(self, "_output", checked_state("output state", output))
        object.__setattr__(self, "_admitted", admitted)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a unary transition is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a unary transition is immutable"
        raise AttributeError(msg)

    @property
    def rule(self) -> int:
        """Return the unary rule index."""
        return self._rule

    @property
    def source_category(self) -> int:
        """Return the premise category index."""
        return self._source[0]

    @property
    def source_tag(self) -> int:
        """Return the premise tag index."""
        return self._source[1]

    @property
    def output_category(self) -> int:
        """Return the conclusion category index."""
        return self._output[0]

    @property
    def output_tag(self) -> int:
        """Return the conclusion tag index."""
        return self._output[1]

    @property
    def admitted(self) -> bool:
        """Return whether the transition satisfies the normal form."""
        return self._admitted

    def _identity(self) -> tuple[int, tuple[int, int], tuple[int, int], bool]:
        """Return fields in comparison order."""
        return (self._rule, self._source, self._output, self._admitted)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare transition fields."""
        if type(other) is not UnaryTransition:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Hash transition fields."""
        return hash((UnaryTransition, self._identity()))

    @override
    def __repr__(self) -> str:
        """Render the constructor call."""
        return (
            f"UnaryTransition({self._rule!r}, {self._source!r}, {self._output!r}, "
            f"admitted={self._admitted!r})"
        )


def _compile_unary[T](
    rule_set: RuleSet[T],
    rules: tuple[Rule[T], ...],
    categories: tuple[T, ...],
    tags: tuple[str, ...],
    tag_indices: Mapping[str, int],
    /,
) -> tuple[UnaryTransition, ...]:
    """Compile every premise tag for every unary rule and category.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The rule inventory.
    rules : tuple[Rule[T], ...]
        The rules in tensor-axis order.
    categories : tuple[T, ...]
        The categories in tensor-axis order.
    tags : tuple[str, ...]
        The normal form tags in tensor-axis order.
    tag_indices : Mapping[str, int]
        The column of each normal form tag.

    Raises
    ------
    ValueError
        If the normal form assigns a tag outside its declared alphabet.
    """
    normal_form = rule_set.normal_form
    compiled: list[UnaryTransition] = []

    for rule_index, rule in enumerate(rules):
        for source_category, source in enumerate(categories):
            outputs = distinct_outputs(rule, (source,), categories, rule_set)

            for output_category, output in outputs:
                for source_tag, source_name in enumerate(tags):
                    children = (source_name,)
                    output_name = normal_form.tag(rule, children)

                    if output_name not in tag_indices:
                        msg = (
                            f"normal form {normal_form.name!r} assigns unknown tag "
                            f"{output_name!r}"
                        )
                        raise ValueError(msg)

                    compiled.append(
                        UnaryTransition(
                            rule_index,
                            (source_category, source_tag),
                            (output_category, tag_indices[output_name]),
                            admitted=normal_form.admits(
                                rule, children, (source,), output
                            ),
                        )
                    )

    return tuple(compiled)


def _source_state(transition: UnaryTransition, tags: int, /) -> int:
    """Return the flat column the transition's premise occupies.

    Parameters
    ----------
    transition : UnaryTransition
        The unary transition.
    tags : int
        The normal form tags in tensor-axis order.
    """
    return transition.source_category * tags + transition.source_tag


def _output_state(transition: UnaryTransition, tags: int, /) -> int:
    """Return the flat column the transition's conclusion occupies.

    Parameters
    ----------
    transition : UnaryTransition
        The unary transition.
    tags : int
        The normal form tags in tensor-axis order.
    """
    return transition.output_category * tags + transition.output_tag


def _reaches(graph: Mapping[int, set[int]], start: int, target: int, /) -> bool:
    """Decide whether a unary path reaches one state from another.

    Parameters
    ----------
    graph : Mapping[int, set[int]]
        The dependency graph.
    start : int
        The starting state.
    target : int
        The target state.
    """
    pending = [start]
    seen: set[int] = set()

    while pending:
        current = pending.pop()

        if current == target:
            return True

        if current in seen:
            continue

        seen.add(current)
        pending.extend(graph.get(current, ()))

    return False


def _dependency_order(graph: Mapping[int, set[int]], /) -> tuple[int, ...]:
    """Order an acyclic state graph from premises to conclusions.

    Parameters
    ----------
    graph : Mapping[int, set[int]]
        The dependency graph.
    """
    indegree = dict.fromkeys(graph, 0)

    for outputs in graph.values():
        for output in outputs:
            indegree[output] += 1

    ready = [state for state in graph if indegree[state] == 0]
    ordered: list[int] = []
    at = 0

    while at < len(ready):
        source = ready[at]
        at += 1
        ordered.append(source)

        for output in graph[source]:
            indegree[output] -= 1

            if indegree[output] == 0:
                ready.append(output)

    return tuple(ordered)


type _Schedule = tuple[tuple[int, tuple[UnaryTransition, ...]], ...]
"""Incoming transitions per state, in application order."""


class UnaryClosure[T]:
    """Applies a rule set's unary rules within one cell, to a bounded depth.

    Closure selects acyclic transitions whose premises are reachable within the depth
    bound, then sums all derivations admitted by those transitions. ``truncated``
    reports whether the emptiest cell could lose a reachable transition.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The inventory whose arity-one rules are closed over.
    rules : tuple[Rule[T], ...]
        Those rules, in the order the unary score channel indexes them.
    categories : tuple[T, ...]
        The category inventory in tensor-axis order.
    tags : tuple[str, ...]
        The normal form's tags in tensor-axis order.
    depth : int
        How many rounds of unary rule application one cell takes.

    Raises
    ------
    TypeError
        If the depth is not an integer.
    ValueError
        If the depth is negative, or the normal form assigns a tag outside its
        alphabet.
    """

    __slots__ = (
        "_categories",
        "_depth",
        "_retained",
        "_rules",
        "_schedule",
        "_tags",
        "_transitions",
        "_truncated",
    )

    def __init__(
        self,
        rule_set: RuleSet[T],
        rules: tuple[Rule[T], ...],
        categories: tuple[T, ...],
        tags: tuple[str, ...],
        /,
        *,
        depth: int,
    ) -> None:
        held_depth = ensure_non_negative("unary_depth", depth)
        tag_indices = {tag: index for index, tag in enumerate(tags)}
        transitions = _compile_unary(rule_set, rules, categories, tags, tag_indices)
        states = len(categories) * len(tags)
        acyclic, cut_cycle = _acyclic(transitions, states, len(tags))
        schedule, cut_depth = _bounded(acyclic, states, len(tags), held_depth)

        self._rules = rules
        self._categories = categories
        self._tags = tags
        self._depth = held_depth
        self._transitions = transitions
        self._retained = tuple(
            transition for _, edges in schedule for transition in edges
        )
        self._schedule = schedule
        self._truncated = cut_cycle or cut_depth

    @property
    def rules(self) -> tuple[Rule[T], ...]:
        """Return the unary rules in score-axis order."""
        return self._rules

    @property
    def transitions(self) -> tuple[UnaryTransition, ...]:
        """Return every compiled transition, admitted or not."""
        return self._transitions

    @property
    def retained(self) -> tuple[UnaryTransition, ...]:
        """Return the transitions the closure may apply, premises first."""
        return self._retained

    @property
    def depth(self) -> int:
        """Return how many rounds of unary rule application one cell takes."""
        return self._depth

    @property
    def truncated(self) -> bool:
        """Return whether a cycle was cut, or a cell can run out of rounds."""
        return self._truncated

    @property
    def schedule(self) -> _Schedule:
        """Return each closed state with its premises, in order."""
        return self._schedule

    def state(self, category: int, tag: int, /) -> int:
        """Return the flat column a category and tag address.

        Parameters
        ----------
        category : int
            The category column.
        tag : int
            The normal form tag column.
        """
        return category * len(self._tags) + tag

    def reachable(self, present: Tensor, /) -> Tensor:
        """Return which states a cell reaches in fewer rounds than the depth.

        Parameters
        ----------
        present : Tensor
            True where a cell already holds a state, over its leading axes and one
            flat state axis.

        Returns
        -------
        Tensor
            True of the same shape where a state may serve as a premise, lying fewer
            unary steps from a state the cell already holds than the depth.
        """
        if self._depth < 1:
            return torch.zeros_like(present)

        columns = list(present.unbind(dim=-1))

        for _ in range(self._depth - 1):
            grown = list(columns)

            for state, edges in self._schedule:
                for edge in edges:
                    grown[state] = (
                        grown[state]
                        | columns[self.state(edge.source_category, edge.source_tag)]
                    )

            columns = grown

        return torch.stack(columns, dim=-1)

    def close(
        self, values: Tensor, scores: Tensor, reduction: LogWeightReduction, /
    ) -> Tensor:
        """Return cell values with every unary derivation within the depth folded in.

        Parameters
        ----------
        values : Tensor
            Weights over item, span, category and tag for one slab of spans, before
            any unary rule applies. A state the slab does not hold is infinitely
            negative.
        scores : Tensor
            Weights over item, span and unary rule for the same slab.
        reduction : LogWeightReduction
            The operation that combines alternative derivations.

        Returns
        -------
        Tensor
            The slab, of the same shape, with each state holding the total weight of
            the unary derivations the depth admits.
        """
        if not self._schedule:
            return values

        leading = tuple(values.shape[:-2])
        flat = values.reshape(*leading, -1)
        premises = list(self.reachable(flat.isfinite()).unbind(dim=-1))
        columns = list(flat.unbind(dim=-1))

        for state, edges in self._schedule:
            sources = [
                self.state(edge.source_category, edge.source_tag) for edge in edges
            ]
            lifted = [
                columns[source] + scores[..., edge.rule]
                for source, edge in zip(sources, edges, strict=True)
            ]
            stacked = torch.stack([columns[state], *lifted], dim=-1)
            admitted = torch.stack(
                [
                    torch.ones_like(columns[state], dtype=torch.bool),
                    *(premises[source] for source in sources),
                ],
                dim=-1,
            )
            columns[state] = reduction.reduce(stacked, admitted, dim=-1)

        return torch.stack(columns, dim=-1).reshape(values.shape)


def _acyclic(
    transitions: tuple[UnaryTransition, ...], states: int, tags: int, /
) -> tuple[tuple[UnaryTransition, ...], bool]:
    """Return the admitted transitions that close no cycle, and whether one was cut.

    Parameters
    ----------
    transitions : tuple[UnaryTransition, ...]
        The unary transitions.
    states : int
        The number of states.
    tags : int
        The normal form tags in tensor-axis order.
    """
    graph: dict[int, set[int]] = {state: set() for state in range(states)}
    kept: list[UnaryTransition] = []
    cut = False

    for source in range(states):
        for transition in transitions:
            if not transition.admitted or _source_state(transition, tags) != source:
                continue

            output = _output_state(transition, tags)

            if _reaches(graph, output, source):
                cut = True
                continue

            kept.append(transition)
            graph[source].add(output)

    return (tuple(kept), cut)


def _levels(
    incoming: Mapping[int, list[UnaryTransition]],
    order: tuple[int, ...],
    tags: int,
    /,
) -> dict[int, int]:
    """Return each state's fewest unary steps from a state no unary rule produces.

    Parameters
    ----------
    incoming : Mapping[int, list[UnaryTransition]]
        The incoming transitions for each state.
    order : tuple[int, ...]
        The state order.
    tags : int
        The normal form tags in tensor-axis order.
    """
    level: dict[int, int] = {}

    for state in order:
        premises = incoming[state]

        if not premises:
            level[state] = 0
            continue

        level[state] = min(
            level[_source_state(transition, tags)] + 1 for transition in premises
        )

    return level


def _bounded(
    transitions: tuple[UnaryTransition, ...], states: int, tags: int, depth: int, /
) -> tuple[_Schedule, bool]:
    """Order acyclic transitions by dependency and say whether the depth can bind.

    Parameters
    ----------
    transitions : tuple[UnaryTransition, ...]
        The unary transitions.
    states : int
        The number of states.
    tags : int
        The normal form tags in tensor-axis order.
    depth : int
        The maximum number of rule applications.
    """
    graph: dict[int, set[int]] = {state: set() for state in range(states)}
    incoming: dict[int, list[UnaryTransition]] = {state: [] for state in range(states)}

    for transition in transitions:
        output = _output_state(transition, tags)
        graph[_source_state(transition, tags)].add(output)
        incoming[output].append(transition)

    order = _dependency_order(graph)
    level = _levels(incoming, order, tags)
    schedule = tuple(
        (state, tuple(incoming[state])) for state in order if incoming[state]
    )
    cut = any(
        level[_source_state(transition, tags)] >= depth for transition in transitions
    )

    return (schedule, cut)
