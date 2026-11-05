"""Normal-form-refined inside charts."""

from itertools import product
from math import inf
from typing import TYPE_CHECKING, Final, Never, final, override

import torch

from lofi._checks import ensure_non_negative
from lofi.nn.parsers._chart import TensorChart
from lofi.nn.parsers._closure import UnaryClosure
from lofi.nn.parsers._inside import _root_ids
from lofi.nn.parsers._states import (
    category_index,
    checked_categories,
    checked_state,
    distinct_outputs,
    rules_by_arity,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.judge import NormalForm, Rule, RuleSet
    from lofi.nn.parsers._chart import _TensorChartAccess
    from lofi.nn.parsers._ports import LogWeightReduction

__all__ = [
    "NormalFormTransitions",
    "ScoreChannels",
    "TagTransition",
    "TaggedInsideChart",
    "TaggedInsideKernel",
    "TaggedRuleScores",
    "span_marginals",
]

_LEXICAL_RANK: Final = 3
"""The rank of category-indexed lexical scores."""

_RULE_RANK: Final = 4
"""The rank of split- and rule-indexed scores."""

_UNARY_RANK: Final = 3
"""The rank of span- and rule-indexed unary scores."""

_TAGGED_AXES: Final = ("item", "span", "category", "tag")
"""Tagged inside-chart axes."""

_TRANSITION_AXES: Final = ("item", "span", "split", "transition")
"""Expanded rule-score axes."""


@final
class TagTransition:
    """One category and normal-form state transition."""

    __slots__ = ("_admitted", "_output", "_premises", "_rule")

    def __init__(
        self,
        rule: int,
        premises: tuple[tuple[int, int], ...],
        output: tuple[int, int],
        *,
        admitted: bool,
    ) -> None:
        if not isinstance(admitted, bool):
            msg = f"transition admission is not Boolean: {admitted!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_rule", ensure_non_negative("rule index", rule))
        if len(premises) < 2:
            msg = (
                f"a combining transition needs at least two premises, not {premises!r}"
            )
            raise ValueError(msg)

        object.__setattr__(
            self,
            "_premises",
            tuple(
                checked_state(f"premise {index} state", premise)
                for index, premise in enumerate(premises)
            ),
        )
        object.__setattr__(self, "_output", checked_state("output state", output))
        object.__setattr__(self, "_admitted", admitted)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a tag transition is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a tag transition is immutable"
        raise AttributeError(msg)

    @property
    def rule(self) -> int:
        """Return the rule index."""
        return self._rule

    @property
    def premises(self) -> tuple[tuple[int, int], ...]:
        """Return the premise category and tag states in rule order."""
        return self._premises

    @property
    def output(self) -> tuple[int, int]:
        """Return the output category and tag state."""
        return self._output

    @property
    def admitted(self) -> bool:
        """Return whether the transition satisfies the normal form."""
        return self._admitted

    def _identity(
        self,
    ) -> tuple[int, tuple[tuple[int, int], ...], tuple[int, int], bool]:
        """Return fields in comparison order."""
        return (self._rule, self._premises, self._output, self._admitted)

    @override
    def __eq__(self, other: object) -> bool:
        """Compare transition fields."""
        if type(other) is not TagTransition:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Hash transition fields."""
        return hash((TagTransition, self._identity()))

    @override
    def __repr__(self) -> str:
        """Render the constructor call."""
        return (
            f"TagTransition({self._rule!r}, {self._premises!r}, {self._output!r}, "
            f"admitted={self._admitted!r})"
        )


@final
class _ChartEdge:
    """One binary edge in the compiled implementation of a rule transition."""

    __slots__ = (
        "_admitted",
        "_left",
        "_output",
        "_right",
        "_rule",
        "_scored",
        "_transition",
    )

    def __init__(
        self,
        rule: int,
        left: int,
        right: int,
        output: int,
        *,
        transition: int | None,
        scored: bool,
        admitted: bool,
    ) -> None:
        object.__setattr__(self, "_rule", ensure_non_negative("rule index", rule))
        object.__setattr__(self, "_left", ensure_non_negative("left state", left))
        object.__setattr__(self, "_right", ensure_non_negative("right state", right))
        object.__setattr__(self, "_output", ensure_non_negative("output state", output))
        object.__setattr__(self, "_transition", transition)
        object.__setattr__(self, "_scored", scored)
        object.__setattr__(self, "_admitted", admitted)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        del name, value
        msg = "a compiled chart edge is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        del name
        msg = "a compiled chart edge is immutable"
        raise AttributeError(msg)

    @property
    def rule(self) -> int:
        return self._rule

    @property
    def left(self) -> int:
        return self._left

    @property
    def right(self) -> int:
        return self._right

    @property
    def output(self) -> int:
        return self._output

    @property
    def transition(self) -> int | None:
        return self._transition

    @property
    def scored(self) -> bool:
        return self._scored

    @property
    def admitted(self) -> bool:
        return self._admitted


type _IndexedApplication[T] = tuple[
    Rule[T], tuple[T, ...], T, tuple[int, tuple[int, ...], int]
]
"""Rule application and its rule/category indices."""


class ScoreChannels:
    """Stores the lexical, combining-rule, and unary log weights one chart reads.

    Parameters
    ----------
    leaf_scores : Tensor
        Rank-3 log weights over item, token, and category.
    rule_scores : Tensor
        Rank-4 log weights over item, span, split, and combining rule.
    unary_scores : Tensor | None
        Rank-3 log weights over item, span, and unary rule, or None where the rule set
        declares no unary rule.

    Raises
    ------
    TypeError
        If a channel is not of a floating point dtype, or if two of them use different
        ones.
    ValueError
        If a channel has the wrong rank, or if two of them occupy different devices.
    """

    __slots__ = ("_leaf", "_rule", "_unary")

    def __init__(
        self,
        leaf_scores: Tensor,
        rule_scores: Tensor,
        unary_scores: Tensor | None = None,
        /,
    ) -> None:
        _checked_channel("lexical scores", leaf_scores, _LEXICAL_RANK)
        _checked_channel("rule scores", rule_scores, _RULE_RANK)

        if unary_scores is not None:
            _checked_channel("unary scores", unary_scores, _UNARY_RANK)

        beside = [("rule", rule_scores)]

        if unary_scores is not None:
            beside.append(("unary", unary_scores))

        for name, channel in beside:
            if channel.dtype != leaf_scores.dtype:
                msg = (
                    f"lexical scores use {leaf_scores.dtype}, but {name} scores use "
                    f"{channel.dtype}"
                )
                raise TypeError(msg)

            if channel.device != leaf_scores.device:
                msg = f"lexical and {name} scores occupy different devices"
                raise ValueError(msg)

        self._leaf = leaf_scores
        self._rule = rule_scores
        self._unary = unary_scores

    @property
    def leaf(self) -> Tensor:
        """Return log weights over item, token, and category."""
        return self._leaf

    @property
    def rule(self) -> Tensor:
        """Return log weights over item, span, split, and combining rule."""
        return self._rule

    @property
    def unary(self) -> Tensor | None:
        """Return log weights over item, span, and unary rule, or None."""
        return self._unary

    @override
    def __repr__(self) -> str:
        """Return the shape of each channel."""
        unary = None if self._unary is None else tuple(self._unary.shape)

        return (
            f"ScoreChannels(leaf={tuple(self._leaf.shape)}, "
            f"rule={tuple(self._rule.shape)}, unary={unary})"
        )


def span_marginals(counts: ScoreChannels, index: SpanIndex, /) -> Tensor:
    """Return the probability that each span is a constituent, from expected counts.

    Parameters
    ----------
    counts : ScoreChannels
        Expected step counts by channel, as the gradient of the log partition gives
        them.
    index : SpanIndex
        The span addressing the channels are laid out under.

    Returns
    -------
    Tensor
        One probability per item and span. A derivation builds a span of two tokens
        or more with one scored combining step and uses each token once, so the
        expected step count at a span is the probability of bracketing it.
    """
    return torch.cat(
        (counts.leaf.sum(dim=-1), counts.rule.sum(dim=(2, 3))[:, index.n :]), dim=1
    )


def _checked_channel(name: str, channel: Tensor, rank: int, /) -> None:
    """Raise if a score channel is not of a floating point dtype of the right rank.

    Parameters
    ----------
    name : str
        The name used in validation or reporting.
    channel : Tensor
        The score channel to validate.
    rank : int
        The required tensor rank.

    Raises
    ------
    TypeError
        If the channel is not of a floating point dtype.
    ValueError
        If it does not have the rank its axes call for.
    """
    if not channel.is_floating_point():
        msg = f"{name} are log weights, not {channel.dtype}"
        raise TypeError(msg)

    if channel.ndim != rank:
        msg = f"{name} run over {rank} axes, not {channel.ndim}"
        raise ValueError(msg)


def _tracked(scores: Tensor, /) -> Tensor:
    """Return a score tensor the partition's gradient can be taken against.

    Parameters
    ----------
    scores : Tensor
        The scores to inspect.
    """
    return scores if scores.requires_grad else scores.detach().requires_grad_()


def _filled(scores: Tensor, counts: Tensor | None, /) -> Tensor:
    """Return expected counts, with zeros where the partition read no score.

    Parameters
    ----------
    scores : Tensor
        The scores to inspect.
    counts : Tensor | None
        The expected counts.
    """
    return torch.zeros_like(scores) if counts is None else counts


class TaggedRuleScores:
    """Stores rule scores expanded over normal-form transitions."""

    __slots__ = ("_log_scores", "_mask")

    def __init__(self, log_scores: Tensor, mask: Tensor, /) -> None:
        if not log_scores.is_floating_point():
            msg = f"transition scores are log weights, not {log_scores.dtype}"
            raise TypeError(msg)

        if log_scores.ndim != len(_TRANSITION_AXES):
            msg = (
                f"transition scores run over {_TRANSITION_AXES}, not rank "
                f"{log_scores.ndim}"
            )
            raise ValueError(msg)

        if mask.dtype is not torch.bool:
            msg = f"a transition mask holds booleans, not {mask.dtype}"
            raise TypeError(msg)

        if tuple(mask.shape) != tuple(log_scores.shape):
            msg = (
                f"a transition mask has shape {tuple(mask.shape)}, not "
                f"{tuple(log_scores.shape)}"
            )
            raise ValueError(msg)

        self._log_scores = log_scores
        self._mask = mask

    @property
    def log_scores(self) -> Tensor:
        """Return masked scores over item, span, split, and transition."""
        return self._log_scores

    @property
    def mask(self) -> Tensor:
        """Return the Boolean mask over the score tensor."""
        return self._mask

    @property
    def axes(self) -> tuple[str, str, str, str]:
        """Return the score axis names."""
        return _TRANSITION_AXES


def _tag_transitions[T](
    normal_form: NormalForm[T],
    application: _IndexedApplication[T],
    tags: tuple[str, ...],
    tag_indices: dict[str, int],
    /,
) -> tuple[TagTransition, ...]:
    """Compile every child-tag tuple for one grammatical rule application.

    Parameters
    ----------
    normal_form : NormalForm[T]
        The normal-form automaton.
    application : _IndexedApplication[T]
        The indexed application transition.
    tags : tuple[str, ...]
        The normal-form tags in tensor-axis order.
    tag_indices : dict[str, int]
        The column of each normal-form tag.
    """
    rule, inputs, output, positions = application
    rule_index, input_categories, output_category = positions
    compiled = []

    for input_tags in product(range(len(tags)), repeat=len(inputs)):
        children = tuple(tags[tag] for tag in input_tags)
        output_name = normal_form.tag(rule, children)

        if output_name not in tag_indices:
            msg = (
                f"normal form {normal_form.name!r} assigns unknown tag {output_name!r}"
            )

            raise ValueError(msg)

        compiled.append(
            TagTransition(
                rule_index,
                tuple(zip(input_categories, input_tags, strict=True)),
                (output_category, tag_indices[output_name]),
                admitted=normal_form.admits(rule, children, inputs, output),
            )
        )

    return tuple(compiled)


def _compile_transitions[T](
    rule_set: RuleSet[T],
    categories: tuple[T, ...],
    rules: tuple[Rule[T], ...],
    tags: tuple[str, ...],
    tag_indices: dict[str, int],
    /,
) -> tuple[TagTransition, ...]:
    """Compile transitions for every grammatical category tuple.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The rule inventory.
    categories : tuple[T, ...]
        The categories in tensor-axis order.
    rules : tuple[Rule[T], ...]
        The rules in tensor-axis order.
    tags : tuple[str, ...]
        The normal-form tags in tensor-axis order.
    tag_indices : dict[str, int]
        The column of each normal-form tag.
    """
    compiled = []

    indexed = tuple(enumerate(categories))

    for rule_index, rule in enumerate(rules):
        for chosen in product(indexed, repeat=rule.arity):
            input_categories = tuple(index for index, _ in chosen)
            inputs = tuple(category for _, category in chosen)

            for output_index, output in distinct_outputs(
                rule, inputs, categories, rule_set
            ):
                compiled.extend(
                    _tag_transitions(
                        rule_set.normal_form,
                        (
                            rule,
                            inputs,
                            output,
                            (rule_index, input_categories, output_index),
                        ),
                        tags,
                        tag_indices,
                    )
                )

    return tuple(compiled)


def _compile_edges(
    transitions: tuple[TagTransition, ...], tags: int, visible: int, /
) -> tuple[tuple[_ChartEdge, ...], int]:
    """Binarize grammatical transitions through shared hidden prefix states."""
    prefixes: dict[tuple[int, tuple[int, ...]], int] = {}
    edges: dict[tuple[int, int, int, int, int | None, bool, bool], _ChartEdge] = {}

    for transition_index, transition in enumerate(transitions):
        premises = tuple(category * tags + tag for category, tag in transition.premises)
        output_category, output_tag = transition.output
        output = output_category * tags + output_tag

        if len(premises) == 2:
            edge = _ChartEdge(
                transition.rule,
                premises[0],
                premises[1],
                output,
                transition=transition_index,
                scored=True,
                admitted=transition.admitted,
            )
            key = (
                edge.rule,
                edge.left,
                edge.right,
                edge.output,
                edge.transition,
                edge.scored,
                edge.admitted,
            )
            edges.setdefault(key, edge)
            continue

        left = premises[0]

        for stop, right in enumerate(premises[1:-1], start=2):
            prefix = (transition.rule, premises[:stop])
            hidden = prefixes.setdefault(prefix, visible + len(prefixes))
            edge = _ChartEdge(
                transition.rule,
                left,
                right,
                hidden,
                transition=None,
                scored=False,
                admitted=True,
            )
            key = (
                edge.rule,
                edge.left,
                edge.right,
                edge.output,
                edge.transition,
                edge.scored,
                edge.admitted,
            )
            edges.setdefault(key, edge)
            left = hidden

        edge = _ChartEdge(
            transition.rule,
            left,
            premises[-1],
            output,
            transition=transition_index,
            scored=True,
            admitted=transition.admitted,
        )
        key = (
            edge.rule,
            edge.left,
            edge.right,
            edge.output,
            edge.transition,
            edge.scored,
            edge.admitted,
        )
        edges.setdefault(key, edge)

    return (tuple(edges.values()), visible + len(prefixes))


class NormalFormTransitions[T]:
    """Compiles category and tag transitions for a rule set.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The inventory whose combining rules index the split score channel and whose
        unary rules index the unary score channel.
    categories : Sequence[T]
        The category inventory in tensor-axis order.
    unary_depth : int
        How many rounds of unary rule application one cell takes.

    Raises
    ------
    TypeError
        If the depth is not an integer.
    ValueError
        If the depth is negative, the category inventory is empty or ambiguous, the
        rule set declares a non-positive-arity rule or a rule whose forward direction
        cannot enumerate its conclusions, the tag alphabet is empty, or the normal
        form assigns a tag outside it.
    """

    __slots__ = (
        "_categories",
        "_chart_states",
        "_closure",
        "_edges",
        "_lexical_tag",
        "_rule_set",
        "_rules",
        "_tags",
        "_transitions",
    )

    def __init__(
        self,
        rule_set: RuleSet[T],
        categories: Sequence[T],
        /,
        *,
        unary_depth: int,
    ) -> None:
        held = checked_categories(categories, rule_set)
        normal_form = rule_set.normal_form
        tags = tuple(sorted(normal_form.alphabet))

        if not tags:
            msg = f"normal form {normal_form.name!r} has an empty tag alphabet"
            raise ValueError(msg)

        lexical = normal_form.lexical_tag()

        if lexical not in normal_form.alphabet:
            msg = (
                f"normal form {normal_form.name!r} assigns unknown lexical tag "
                f"{lexical!r}"
            )
            raise ValueError(msg)

        unary, rules = rules_by_arity(rule_set)
        tag_indices = {tag: index for index, tag in enumerate(tags)}
        transitions = _compile_transitions(rule_set, held, rules, tags, tag_indices)
        edges, chart_states = _compile_edges(
            transitions, len(tags), len(held) * len(tags)
        )

        self._rule_set = rule_set
        self._categories = held
        self._rules = rules
        self._tags = tags
        self._lexical_tag = tag_indices[lexical]
        self._transitions = transitions
        self._edges = edges
        self._chart_states = chart_states
        self._closure = UnaryClosure(rule_set, unary, held, tags, depth=unary_depth)

    @property
    def rule_set(self) -> RuleSet[T]:
        """Return the rule set that defines the transitions."""
        return self._rule_set

    @property
    def categories(self) -> tuple[T, ...]:
        """Return categories in tensor-axis order."""
        return self._categories

    @property
    def tags(self) -> tuple[str, ...]:
        """Return normal-form tags in tensor-axis order."""
        return self._tags

    @property
    def rules(self) -> tuple[Rule[T], ...]:
        """Return combining rules in score-axis order."""
        return self._rules

    @property
    def transitions(self) -> tuple[TagTransition, ...]:
        """Return compiled transitions in mask-axis order."""
        return self._transitions

    @property
    def edges(self) -> tuple[_ChartEdge, ...]:
        """Return binary implementation edges in mask-axis order."""
        return self._edges

    @property
    def lexical_tag(self) -> int:
        """Return the lexical tag's tensor index."""
        return self._lexical_tag

    @property
    def closure(self) -> UnaryClosure[T]:
        """Return the bounded unary closure applied within each cell."""
        return self._closure

    @property
    def unary_rules(self) -> tuple[Rule[T], ...]:
        """Return unary rules in unary-score-axis order."""
        return self._closure.rules

    @property
    def truncated(self) -> bool:
        """Return whether the compiled unary closure may drop a derivation."""
        return self._closure.truncated

    @property
    def states(self) -> int:
        """Return how many category and tag columns one cell holds."""
        return len(self._categories) * len(self._tags)

    @property
    def chart_states(self) -> int:
        """Return the visible and hidden prefix columns used by the sweep."""
        return self._chart_states

    def checked_channels(self, channels: ScoreChannels, index: SpanIndex, /) -> int:
        """Return the batch size the score channels and the index agree on.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights to check.
        index : SpanIndex
            The span addressing they are laid out under.

        Returns
        -------
        int
            How many items every channel covers.

        Raises
        ------
        TypeError
            If a channel is not of a floating point dtype.
        ValueError
            If the index is empty, if a channel has a shape the index and the compiled
            transitions do not call for, if the channels disagree about the batch
            size, or if the unary channel and the unary rules disagree about each
            other's presence.
        """
        if index.n < 1:
            msg = f"a tagged chart needs an input of length at least one, not {index.n}"
            raise ValueError(msg)

        batch = _checked_leaf_categories(channels.leaf, index, len(self._categories))
        rules_batch = _checked_rule_scores(channels.rule, index, len(self._rules))

        if rules_batch != batch:
            msg = (
                f"lexical scores for {batch} items came with rule scores for "
                f"{rules_batch}"
            )
            raise ValueError(msg)

        self.checked_unary_scores(channels.unary, index, batch)

        return batch

    def goal_categories(self, goals: Sequence[T] | None, /) -> tuple[int, ...]:
        """Return the columns a sequence of goal categories addresses.

        Parameters
        ----------
        goals : Sequence[T] | None
            The categories a derivation may conclude at, or None for every category
            in the inventory.

        Returns
        -------
        tuple[int, ...]
            The category columns, in the order the goals name them and without
            repetition.

        Raises
        ------
        ValueError
            If a goal is outside the inventory, or if an empty sequence is given.
        """
        if goals is None:
            return tuple(range(len(self._categories)))

        found: list[int] = []

        for goal in goals:
            column = category_index(self._categories, self._rule_set, goal)

            if column is None:
                msg = f"goal category {goal!r} is outside the chart inventory"
                raise ValueError(msg)

            if column not in found:
                found.append(column)

        if not found:
            msg = "a tagged partition needs at least one goal category"
            raise ValueError(msg)

        return tuple(found)

    def checked_unary_scores(
        self, unary_scores: Tensor | None, index: SpanIndex, batch: int, /
    ) -> Tensor | None:
        """Return a unary score channel matching the rules it weights.

        Parameters
        ----------
        unary_scores : Tensor | None
            The unary rule scores, or None when unary rules are absent.
        index : SpanIndex
            The addressing used by the computation.
        batch : int
            The number of items in the batch.

        Raises
        ------
        TypeError
            If a supplied channel is not of a floating point dtype.
        ValueError
            If the rule set declares unary rules and no channel accompanies them, if
            a channel accompanies a rule set declaring none, or if its shape is not
            the one the index and the unary rules call for.
        """
        rules = len(self._closure.rules)

        if unary_scores is None:
            if rules:
                msg = (
                    f"{rules} unary rules need a unary score channel, which was "
                    "not supplied"
                )
                raise ValueError(msg)

            return None

        if not rules:
            msg = "a rule set declaring no unary rule takes no unary score channel"
            raise ValueError(msg)

        if not unary_scores.is_floating_point():
            msg = f"unary scores are log weights, not {unary_scores.dtype}"
            raise TypeError(msg)

        wanted = (batch, index.num_spans, rules)

        if tuple(unary_scores.shape) != wanted:
            msg = f"unary scores have shape {tuple(unary_scores.shape)}, not {wanted}"
            raise ValueError(msg)

        return unary_scores

    @property
    def item_multiplier(self) -> int:
        """Return the item-count factor introduced by tag refinement."""
        return len(self._tags)

    def item_count(self, index: SpanIndex, /) -> int:
        """Return the category-state count for an input length.

        Parameters
        ----------
        index : SpanIndex
            The addressing used by the computation.
        """
        return index.num_spans * len(self._categories) * self.item_multiplier

    def mask_scores(self, rule_scores: Tensor, index: SpanIndex, /) -> TaggedRuleScores:
        """Apply split and tag-transition masks to rule scores.

        Parameters
        ----------
        rule_scores : Tensor
            The combining-rule scores.
        index : SpanIndex
            The addressing used by the computation.
        """
        if index.n < 1:
            msg = (
                "transition masking needs an input of length at least one, "
                f"not {index.n}"
            )
            raise ValueError(msg)

        batch = _checked_rule_scores(rule_scores, index, len(self._rules))
        edges = self._edges

        if edges:
            columns = rule_scores.new_tensor(
                [edge.rule for edge in edges], dtype=torch.long
            )
            selected = rule_scores.index_select(-1, columns)
            scored = rule_scores.new_tensor(
                [edge.scored for edge in edges], dtype=torch.bool
            )
            selected = torch.where(scored, selected, torch.zeros_like(selected))
            admitted = rule_scores.new_tensor(
                [edge.admitted for edge in edges], dtype=torch.bool
            )
        else:
            selected = rule_scores.new_empty((*rule_scores.shape[:-1], 0))
            admitted = rule_scores.new_empty((0,), dtype=torch.bool)

        splits = rule_scores.new_tensor(index.split_mask, dtype=torch.bool)
        mask = (
            splits.unsqueeze(0).unsqueeze(-1).expand(batch, -1, -1, len(edges))
            & admitted
        )

        return TaggedRuleScores(selected.masked_fill(~mask, -inf), mask)


class TaggedInsideChart[T]:
    """Stores inside scores over categories and normal-form tags."""

    __slots__ = ("_index", "_inside", "_transitions")

    def __init__(
        self,
        inside: Tensor,
        transitions: NormalFormTransitions[T],
        index: SpanIndex,
        /,
    ) -> None:
        if not inside.is_floating_point():
            msg = f"a tagged chart holds log weights, not {inside.dtype}"
            raise TypeError(msg)

        if inside.ndim != len(_TAGGED_AXES):
            msg = f"a tagged chart runs over {_TAGGED_AXES}, not rank {inside.ndim}"
            raise ValueError(msg)

        wanted = (
            index.num_spans,
            len(transitions.categories),
            len(transitions.tags),
        )

        if tuple(inside.shape[1:]) != wanted:
            msg = (
                f"a tagged chart has trailing shape {tuple(inside.shape[1:])}, "
                f"not {wanted}"
            )
            raise ValueError(msg)

        self._inside = inside
        self._transitions = transitions
        self._index = index

    @property
    def inside(self) -> Tensor:
        """Return scores over item, span, category, and tag."""
        return self._inside

    @property
    def transitions(self) -> NormalFormTransitions[T]:
        """Return the transition table defining the state axes."""
        return self._transitions

    @property
    def index(self) -> SpanIndex:
        """Return the chart's span index."""
        return self._index

    @property
    def truncated(self) -> bool:
        """Return whether the closure that built the chart may have dropped a step."""
        return self._transitions.truncated

    @property
    def axes(self) -> tuple[str, str, str, str]:
        """Return the chart axis names."""
        return _TAGGED_AXES


def _checked_leaf_categories(
    leaf_scores: Tensor, index: SpanIndex, categories: int, /
) -> int:
    """Validate category-indexed lexical scores and return their batch size.

    Parameters
    ----------
    leaf_scores : Tensor
        The lexical scores.
    index : SpanIndex
        The addressing used by the computation.
    categories : int
        The categories in tensor-axis order.
    """
    if not leaf_scores.is_floating_point():
        msg = f"lexical scores are log weights, not {leaf_scores.dtype}"
        raise TypeError(msg)

    if leaf_scores.ndim != _LEXICAL_RANK:
        msg = (
            "lexical scores run over item, token, and category, "
            f"not rank {leaf_scores.ndim}"
        )
        raise ValueError(msg)

    batch = int(leaf_scores.shape[0])
    wanted = (batch, index.n, categories)

    if tuple(leaf_scores.shape) != wanted:
        msg = f"lexical scores have shape {tuple(leaf_scores.shape)}, not {wanted}"
        raise ValueError(msg)

    return batch


def _checked_rule_scores(rule_scores: Tensor, index: SpanIndex, rules: int, /) -> int:
    """Validate split- and rule-indexed scores and return their batch size.

    Parameters
    ----------
    rule_scores : Tensor
        The combining-rule scores.
    index : SpanIndex
        The addressing used by the computation.
    rules : int
        The rules in tensor-axis order.
    """
    if not rule_scores.is_floating_point():
        msg = f"rule scores are log weights, not {rule_scores.dtype}"
        raise TypeError(msg)

    if rule_scores.ndim != _RULE_RANK:
        msg = (
            "rule scores run over item, span, split, and rule, "
            f"not rank {rule_scores.ndim}"
        )
        raise ValueError(msg)

    batch = int(rule_scores.shape[0])
    wanted = (batch, index.num_spans, index.max_splits, rules)

    if tuple(rule_scores.shape) != wanted:
        msg = f"rule scores have shape {tuple(rule_scores.shape)}, not {wanted}"
        raise ValueError(msg)

    return batch


class _Gathers:
    """Index vectors addressing children and states for one transition table."""

    __slots__ = (
        "incoming_edges",
        "left_states",
        "right_states",
    )

    def __init__[T](
        self,
        reference: Tensor,
        transitions: NormalFormTransitions[T],
        /,
    ) -> None:
        candidates = transitions.edges
        incoming: list[list[int]] = [[] for _ in range(transitions.chart_states)]

        for edge_index, transition in enumerate(candidates):
            incoming[transition.output].append(edge_index)

        self.incoming_edges = tuple(
            reference.new_tensor(indices, dtype=torch.long) for indices in incoming
        )
        self.left_states = reference.new_tensor(
            [transition.left for transition in candidates],
            dtype=torch.long,
        )
        self.right_states = reference.new_tensor(
            [transition.right for transition in candidates],
            dtype=torch.long,
        )


class TaggedInsideKernel[T]:
    """Compute inside values over category and normal-form states.

    Parameters
    ----------
    reduction : LogWeightReduction
        The operation that combines alternative derivations.
    rule_set : RuleSet[T]
        The inventory whose combining rules index the split score channel and whose
        unary rules index the unary score channel.
    categories : Sequence[T]
        The category inventory in tensor-axis order.
    unary_depth : int
        How many rounds of unary rule application one cell takes.
    """

    __slots__ = ("_reduction", "_transitions")

    def __init__(
        self,
        reduction: LogWeightReduction,
        rule_set: RuleSet[T],
        categories: Sequence[T],
        /,
        *,
        unary_depth: int,
    ) -> None:
        self._reduction = reduction
        self._transitions = NormalFormTransitions(
            rule_set, categories, unary_depth=unary_depth
        )

    @property
    def reduction(self) -> LogWeightReduction:
        """Return the operation that combines alternative derivations."""
        return self._reduction

    @property
    def transitions(self) -> NormalFormTransitions[T]:
        """Return the compiled category and tag transitions."""
        return self._transitions

    def _lexical_chart(self, leaf_scores: Tensor, /) -> Tensor:
        """Return one-token spans holding their category scores at the lexical tag.

        Parameters
        ----------
        leaf_scores : Tensor
            The lexical scores.
        """
        tags = len(self._transitions.tags)
        tag_ids = leaf_scores.new_tensor(range(tags), dtype=torch.long)
        lexical = tag_ids == self._transitions.lexical_tag

        visible = (
            leaf_scores.unsqueeze(-1)
            .expand(-1, -1, -1, tags)
            .masked_fill(~lexical, -inf)
            .reshape(*leaf_scores.shape[:2], self._transitions.states)
        )

        return torch.cat(
            (
                visible,
                leaf_scores.new_full(
                    (
                        *leaf_scores.shape[:2],
                        self._transitions.chart_states - self._transitions.states,
                    ),
                    -inf,
                ),
            ),
            dim=-1,
        )

    def _close(self, values: Tensor, scores: Tensor, /) -> Tensor:
        """Apply unary closure to visible states and retain hidden prefix states."""
        transitions = self._transitions
        visible = values[..., : transitions.states].reshape(
            *values.shape[:-1], len(transitions.categories), len(transitions.tags)
        )
        closed = transitions.closure.close(visible, scores, self._reduction).reshape(
            *values.shape[:-1], transitions.states
        )

        return torch.cat((closed, values[..., transitions.states :]), dim=-1)

    def _combine(
        self,
        chart: _TensorChartAccess,
        scores: TaggedRuleScores,
        gathers: _Gathers,
        batch: int,
        /,
        *,
        width: int,
    ) -> Tensor:
        """Return one slab of spans built from their children at every split.

        Parameters
        ----------
        chart : _TensorChartAccess
            The packed chart values.
        scores : TaggedRuleScores
            The scores to inspect.
        gathers : _Gathers
            The precomputed child gather indices.
        batch : int
            The number of items in the batch.
        width : int
            How many tokens each cell in the slab spans.
        """
        start, stop = chart.bounds(width)
        spans = stop - start
        left, right = chart.read_width_children(width)
        left = left.permute(2, 0, 1, 3)
        right = right.permute(2, 0, 1, 3)
        joined = (
            left.index_select(-1, gathers.left_states)
            + right.index_select(-1, gathers.right_states)
            + scores.log_scores[:, start:stop]
        )
        base_mask = scores.mask[:, start:stop]
        outputs = []

        for incoming in gathers.incoming_edges:
            if incoming.numel() == 0:
                outputs.append(joined.new_full((batch, spans), -inf))
                continue

            alternatives = joined.index_select(-1, incoming).reshape(batch, spans, -1)
            mask = base_mask.index_select(-1, incoming).reshape(batch, spans, -1)
            outputs.append(self._reduction.reduce(alternatives, mask, dim=-1))

        return torch.stack(outputs, dim=-1)

    def _sweep(self, channels: ScoreChannels, index: SpanIndex, /) -> Tensor:
        """Return every closed visible and hidden prefix state, span-major.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        """
        batch = self._transitions.checked_channels(channels, index)
        unary = channels.unary
        lexical = self._lexical_chart(channels.leaf)

        if unary is not None:
            lexical = self._close(lexical, unary[:, : index.n])

        chart = TensorChart(
            index,
            channels.leaf,
            batch=batch,
            width=self._transitions.chart_states,
        )

        expanded = self._transitions.mask_scores(channels.rule, index)
        gathers = _Gathers(channels.leaf, self._transitions)

        with chart as storage:
            storage.write_width(1, lexical.transpose(0, 1))

            for width in range(2, index.n + 1):
                start = index.span_id(0, width)
                stop = start + index.n - width + 1

                if self._transitions.edges:
                    slab = self._combine(storage, expanded, gathers, batch, width=width)
                else:
                    slab = channels.leaf.new_full(
                        (batch, stop - start, self._transitions.chart_states), -inf
                    )

                if unary is not None:
                    slab = self._close(slab, unary[:, start:stop])

                storage.write_width(width, slab.transpose(0, 1))

            return storage.values()

    def inside(
        self, channels: ScoreChannels, index: SpanIndex, /
    ) -> TaggedInsideChart[T]:
        """Return the normal-form-refined inside chart.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights, the last present
            exactly when the rule set declares a unary rule.
        index : SpanIndex
            The span addressing every channel is laid out under.

        Returns
        -------
        TaggedInsideChart[T]
            The chart over item, span, category, and tag, with every cell closed under
            the rule set's unary rules to the compiled depth.
        """
        transitions = self._transitions
        values = self._sweep(channels, index)
        visible = values.permute(1, 0, 2)[..., : transitions.states].reshape(
            values.shape[1],
            index.num_spans,
            len(transitions.categories),
            len(transitions.tags),
        )

        return TaggedInsideChart(visible, transitions, index)

    def _unclose(
        self,
        wanted: Tensor,
        premises: Tensor,
        scores: Tensor,
        /,
    ) -> Tensor:
        """Return one slab's demand carried back through its unary closure.

        Parameters
        ----------
        wanted : Tensor
            True over span, item and chart state where a goal derivation reads the
            closed state.
        premises : Tensor
            True over span, item and visible state where the closure may lift the
            state, as the closure's depth bound decides from the cell before closure.
        scores : Tensor
            Weights over item, span and unary rule for the same slab.
        """
        closure = self._transitions.closure
        states = self._transitions.states
        usable = scores.isfinite().transpose(0, 1)
        columns = list(wanted[..., :states].unbind(dim=-1))

        for state, edges in reversed(closure.schedule):
            for edge in edges:
                source = closure.state(edge.source_category, edge.source_tag)
                columns[source] = columns[source] | (
                    columns[state] & premises[..., source] & usable[..., edge.rule]
                )

        return torch.cat((torch.stack(columns, dim=-1), wanted[..., states:]), dim=-1)

    def goal_support(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return which states occur in a finite-weight derivation of a goal.

        A state at a span is supported when some derivation of the whole item,
        concluding in a goal category at the root span and reading only finite
        weights, uses that state there. The pass is Boolean and exact: it marks the
        closed states the root goals read, carries that demand back through each
        cell's unary closure under the closure's own depth bound, and then through
        every admitted binary edge, hidden prefix edges included, whose children
        and weight are finite.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        goals : Sequence[T] | None
            The root categories a derivation may conclude in, or None for all of
            them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        Tensor
            A Boolean tensor over item, span, category, and tag. An item with no
            finite-weight goal derivation supports nothing.
        """
        transitions = self._transitions
        tags = len(transitions.tags)
        states = transitions.states

        with torch.no_grad():
            values = self._sweep(channels, index)
            present = values.isfinite()
            batch = int(values.shape[1])
            demand = torch.zeros_like(present)
            roots = _root_ids(present.transpose(0, 1), index, lengths)
            goal_states = present.new_tensor(
                [
                    category * tags + tag
                    for category in transitions.goal_categories(goals)
                    for tag in range(tags)
                ],
                dtype=torch.long,
            )
            items = torch.arange(batch, device=present.device)
            seeded = (roots.unsqueeze(-1), items.unsqueeze(-1), goal_states)
            demand[seeded] = present[seeded]

            edges = transitions.edges
            unary = channels.unary
            expanded = transitions.mask_scores(channels.rule, index)
            gathers = _Gathers(channels.leaf, transitions)
            outputs = present.new_tensor(
                [edge.output for edge in edges], dtype=torch.long
            )
            left_ids = present.new_tensor(index.left_child_ids, dtype=torch.long)
            right_ids = present.new_tensor(index.right_child_ids, dtype=torch.long)
            lexical = self._lexical_chart(channels.leaf)[..., :states].isfinite()
            chart = TensorChart(
                index, channels.leaf, batch=batch, width=transitions.chart_states
            )

            with chart as storage:
                for width in range(1, index.n + 1):
                    start, stop = storage.bounds(width)
                    storage.write_width(width, values[start:stop])

                for width in range(index.n, 0, -1):
                    start, stop = storage.bounds(width)
                    joined = None

                    if width > 1 and edges:
                        left, right = storage.read_width_children(width)
                        joined = (
                            left.permute(2, 0, 1, 3).index_select(
                                -1, gathers.left_states
                            )
                            + right.permute(2, 0, 1, 3).index_select(
                                -1, gathers.right_states
                            )
                            + expanded.log_scores[:, start:stop]
                        ).isfinite()

                    if unary is not None:
                        if width == 1:
                            before = lexical
                        elif joined is None:
                            before = present.new_zeros((batch, stop - start, states))
                        else:
                            built = joined.any(dim=2).to(torch.int64)
                            before = (
                                built.new_zeros(
                                    (batch, stop - start, transitions.chart_states)
                                )
                                .index_add_(-1, outputs, built)
                                .gt(0)[..., :states]
                            )

                        demand[start:stop] = self._unclose(
                            demand[start:stop],
                            transitions.closure.reachable(before.transpose(0, 1)),
                            unary[:, start:stop],
                        )

                    if joined is None:
                        continue

                    active = joined & (
                        demand[start:stop]
                        .transpose(0, 1)
                        .index_select(-1, outputs)
                        .unsqueeze(2)
                    )
                    item, span, split, edge = active.nonzero(as_tuple=True)
                    span = span + start
                    demand[left_ids[span, split], item, gathers.left_states[edge]] = (
                        True
                    )
                    demand[right_ids[span, split], item, gathers.right_states[edge]] = (
                        True
                    )

        supported = (present & demand)[..., :states]

        return supported.transpose(0, 1).reshape(
            batch, index.num_spans, len(transitions.categories), tags
        )

    def requires_span(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        span: tuple[int, int],
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return whether every finite-weight goal derivation builds one span.

        A derivation builds a span when a scored combining step concludes there. The
        decision is exact: it removes every scored step at the span, which leaves
        hidden prefix states in place, and asks whether a goal remains derivable at
        the root.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        span : tuple[int, int]
            The start, inclusive, and the end, exclusive, of the span.
        goals : Sequence[T] | None
            The root categories a derivation may conclude in, or None for all of
            them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        Tensor
            One Boolean per item. An item with no finite-weight goal derivation at
            all has none that omits the span, so it reads True.

        Raises
        ------
        ValueError
            If the span is empty or runs past the index.
        """
        start, end = span

        if not 0 <= start < end <= index.n:
            msg = f"{span} is not a span of an input of length {index.n}"
            raise ValueError(msg)

        with torch.no_grad():
            blocked = channels.rule.clone()
            blocked[:, index.span_id(start, end)] = -inf
            partition = self.log_partition(
                ScoreChannels(channels.leaf, blocked, channels.unary),
                index,
                goals=goals,
                lengths=lengths,
            )

        return ~partition.isfinite()

    def log_partition(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return the total weight of the selected root categories.

        Parameters
        ----------
        channels : ScoreChannels
            The parser score channels.
        index : SpanIndex
            The addressing used by the computation.
        goals : Sequence[T] | None
            The goal categories, or None to use the defaults.
        lengths : Tensor | None
            The item lengths, or None when all items are full length.
        """
        chart = self.inside(channels, index).inside
        roots = _root_ids(chart[..., 0, 0], index, lengths)
        batch = int(chart.shape[0])
        root = chart.gather(
            1,
            roots.reshape(batch, 1, 1, 1).expand(
                -1, 1, len(self._transitions.categories), len(self._transitions.tags)
            ),
        ).squeeze(1)
        goal_indices = self._transitions.goal_categories(goals)
        category_ids = root.new_tensor(
            range(len(self._transitions.categories)), dtype=torch.long
        )
        goal_mask = torch.zeros_like(category_ids, dtype=torch.bool)

        for category in goal_indices:
            goal_mask |= category_ids == category

        mask = goal_mask.unsqueeze(-1).expand_as(root)
        reduced = self._reduction.reduce(root, mask, dim=-1)

        return self._reduction.reduce(
            reduced, goal_mask.unsqueeze(0).expand(batch, -1), dim=-1
        )

    def _expected_counts(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        goals: Sequence[T] | None,
        lengths: Tensor | None,
        /,
    ) -> tuple[Tensor, ScoreChannels, ScoreChannels]:
        """Return the partition, the channels it read, and their expected counts.

        Parameters
        ----------
        channels : ScoreChannels
            The parser score channels.
        index : SpanIndex
            The addressing used by the computation.
        goals : Sequence[T] | None
            The goal categories, or None to use the defaults.
        lengths : Tensor | None
            The item lengths, or None when all items are full length.

        Raises
        ------
        ValueError
            If the reduction does not sum probabilities, where alone the gradient of
            the partition at a score is that score's expected count.
        """
        self._reduction.validate_expected_counts()

        track = torch.is_grad_enabled()

        with torch.enable_grad():
            leaves = _tracked(channels.leaf)
            rules = _tracked(channels.rule)
            unary = None if channels.unary is None else _tracked(channels.unary)
            tracked = ScoreChannels(leaves, rules, unary)
            partition = self.log_partition(tracked, index, goals=goals, lengths=lengths)

            if unary is None:
                leaf_counts, rule_counts = torch.autograd.grad(
                    partition.sum(),
                    (leaves, rules),
                    create_graph=track,
                    allow_unused=True,
                )
                unary_counts = None
            else:
                leaf_counts, rule_counts, unary_counts = torch.autograd.grad(
                    partition.sum(),
                    (leaves, rules, unary),
                    create_graph=track,
                    allow_unused=True,
                )

        return (
            partition,
            tracked,
            ScoreChannels(
                _filled(leaves, leaf_counts),
                _filled(rules, rule_counts),
                None if unary is None else _filled(unary, unary_counts),
            ),
        )

    def expected_counts(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> ScoreChannels:
        """Return how often a derivation drawn from the chart reads each score.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        goals : Sequence[T] | None
            The root categories the partition sums over, or None for all of them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        ScoreChannels
            The gradient of the log partition at each channel, of that channel's
            shape, which is the expected count of the steps that channel weights.
            The counts are themselves differentiable where the caller had gradient
            tracking on.
        """
        _, _, counts = self._expected_counts(channels, index, goals, lengths)

        return counts

    def derivation_entropy(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return entropy over normal-form-admissible derivations, in nats.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        goals : Sequence[T] | None
            The root categories the partition sums over, or None for all of them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        Tensor
            One entropy per item, in nats. An item the chart derives no tree for has
            no distribution to be uncertain about, and reads zero.
        """
        partition, scores, counts = self._expected_counts(
            channels, index, goals, lengths
        )
        pairs = (
            (scores.leaf, counts.leaf),
            (scores.rule, counts.rule),
            (scores.unary, counts.unary),
        )
        expected = torch.zeros_like(partition)

        for score, count in pairs:
            if score is None or count is None:
                continue

            weighted = torch.where(score.isfinite(), score * count, score.new_zeros(()))
            expected = expected + weighted.flatten(start_dim=1).sum(dim=-1)

        return (partition - expected).clamp_min(0.0)

    def constituent_marginals(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        /,
        *,
        goals: Sequence[T] | None = None,
        lengths: Tensor | None = None,
    ) -> Tensor:
        """Return the probability that each span is a constituent.

        Parameters
        ----------
        channels : ScoreChannels
            The lexical, combining-rule, and unary log weights.
        index : SpanIndex
            The span addressing every channel is laid out under.
        goals : Sequence[T] | None
            The root categories the partition sums over, or None for all of them.
        lengths : Tensor | None
            Token counts by item, or None when every item fills the index.

        Returns
        -------
        Tensor
            One probability per item and span. A derivation builds a span of two
            tokens or more with one scored combining step and uses each token once,
            so the expected step count at a span is the probability of bracketing it.
        """
        _, _, counts = self._expected_counts(channels, index, goals, lengths)

        return span_marginals(counts, index)
