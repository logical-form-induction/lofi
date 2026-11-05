"""Gold-derivation scoring over tagged chart channels."""

from __future__ import annotations

from collections.abc import Hashable
from typing import TYPE_CHECKING, final

from lofi.nn._values import DerivationSetScore, NormalFormDerivationScores
from lofi.nn.parsers._states import category_index
from lofi.nn.parsers._tagged import ScoreChannels, TaggedInsideKernel

if TYPE_CHECKING:
    from collections.abc import Sequence

    from torch import Tensor

    from lofi.deduce import SpanIndex
    from lofi.judge import Rule, RuleSet
    from lofi.nn._derivation import DerivationTree

__all__ = ["TaggedDerivationScoreChart"]


def _item_value(values: Tensor, item: int, /) -> float:
    """Return one detached scalar from an item-indexed tensor."""
    return float(values[item].detach().cpu().item())


@final
class TaggedDerivationScoreChart[T: Hashable]:
    """Score gold derivations against constrained and unrestricted tagged charts."""

    __slots__ = (
        "_categories",
        "_channels",
        "_constrained",
        "_constrained_partition",
        "_goals",
        "_index",
        "_lengths",
        "_rules",
        "_unconstrained",
        "_unconstrained_partition",
    )

    def __init__(
        self,
        channels: ScoreChannels,
        index: SpanIndex,
        rules: RuleSet[T],
        categories: Sequence[T],
        goals: Sequence[T],
        constrained: TaggedInsideKernel[T],
        unconstrained: TaggedInsideKernel[T],
        constrained_partition: Tensor,
        /,
        *,
        lengths: Tensor | None,
    ) -> None:
        detached = ScoreChannels(
            channels.leaf.detach(),
            channels.rule.detach(),
            None if channels.unary is None else channels.unary.detach(),
        )
        constrained.transitions.checked_channels(detached, index)
        unconstrained.transitions.checked_channels(detached, index)
        self._channels = detached
        self._index = index
        self._rules = rules
        self._categories = tuple(categories)
        self._goals = tuple(goals)
        self._constrained = constrained
        self._unconstrained = unconstrained
        self._constrained_partition = constrained_partition.detach()
        self._unconstrained_partition: Tensor | None = None
        self._lengths = None if lengths is None else lengths.detach()

    @property
    def normal_form(self) -> str:
        """Return the parser's declared normal form."""
        return self._rules.normal_form.name

    def _length(self, item: int, /) -> int:
        """Return one item's declared word count."""
        batch = int(self._channels.leaf.shape[0])

        if not 0 <= item < batch:
            msg = f"there is no item {item} in a derivation chart over {batch} items"
            raise IndexError(msg)

        if self._lengths is None:
            return self._index.n

        return int(self._lengths[item].detach().cpu().item())

    def _rule(
        self,
        node_id: int,
        named: str | None,
        inputs: tuple[T, ...],
        output: T,
        /,
    ) -> Rule[T]:
        """Return the unique declared rule licensing one gold node."""
        recovered = self._rules.recover(inputs, output)

        if named is None:
            if len(recovered) != 1:
                msg = (
                    f"gold derivation node {node_id} names no rule and "
                    f"{len(recovered)} declared rules license it"
                )
                raise ValueError(msg)

            return recovered[0]

        selected = tuple(rule for rule in recovered if rule.name == named)

        if not selected:
            msg = (
                f"gold derivation node {node_id} names rule {named!r}, which does "
                "not license its child and parent categories"
            )
            raise ValueError(msg)

        return selected[0]

    def _gold_score(
        self, item: int, derivation: DerivationTree[T], /
    ) -> tuple[float, bool]:
        """Return the raw gold score and whether the declared filter admits it."""
        length = self._length(item)
        root = derivation.node(derivation.root)

        if (root.span.start, root.span.end) != (0, length):
            msg = (
                f"gold derivation root covers [{root.span.start}, {root.span.end}), "
                f"not item {item}'s [0, {length})"
            )
            raise ValueError(msg)

        if not any(
            self._rules.matching.equivalent(root.label, goal) for goal in self._goals
        ):
            msg = f"gold derivation root {root.label!r} is not a declared grammar goal"
            raise ValueError(msg)

        normal_form = self._rules.normal_form
        scores = self._channels
        tags: dict[int, str] = {}
        admitted = True
        total = scores.leaf.new_zeros(())
        unary_rules = self._constrained.transitions.unary_rules
        combining_rules = self._constrained.transitions.rules

        for node_id in reversed(derivation.preorder):
            node = derivation.node(node_id)
            category = category_index(self._categories, self._rules, node.label)

            if category is None:
                msg = (
                    f"gold derivation node {node_id} has category {node.label!r}, "
                    "which is outside the tagged inventory"
                )
                raise ValueError(msg)

            if node.is_lexical:
                if node.span.length != 1:
                    msg = (
                        f"lexical gold derivation node {node_id} covers "
                        f"{node.span.length} tokens"
                    )
                    raise ValueError(msg)

                if node.rule is not None:
                    msg = (
                        f"lexical gold derivation node {node_id} names rule "
                        f"{node.rule!r}"
                    )
                    raise ValueError(msg)

                total = total + scores.leaf[item, node.span.start, category]
                tags[node_id] = normal_form.lexical_tag()
                continue

            children = derivation.children_of(node_id)
            reached = node.span.start

            for child in children:
                if child.span.start != reached:
                    msg = (
                        f"gold derivation node {node_id}'s children are not an "
                        f"ordered tiling of [{node.span.start}, {node.span.end}): "
                        f"a child starts at {child.span.start}, not {reached}"
                    )
                    raise ValueError(msg)

                reached = child.span.end

            if reached != node.span.end:
                msg = (
                    f"gold derivation node {node_id}'s children reach {reached}, "
                    f"not the end {node.span.end}"
                )
                raise ValueError(msg)

            inputs = tuple(child.label for child in children)
            rule = self._rule(node_id, node.rule, inputs, node.label)
            child_tags = tuple(tags[child] for child in node.children)
            admitted = admitted and normal_form.admits(
                rule, child_tags, inputs, node.label
            )
            tags[node_id] = normal_form.tag(rule, child_tags)
            span = self._index.span_id(node.span.start, node.span.end)

            if rule.arity == 1:
                if scores.unary is None:
                    msg = f"gold derivation node {node_id} uses an absent unary channel"
                    raise ValueError(msg)

                column = unary_rules.index(rule)
                total = total + scores.unary[item, span, column]
                continue

            if rule.arity < 2:
                msg = f"gold derivation node {node_id} uses rule arity {rule.arity}"
                raise ValueError(msg)

            split = children[-1].span.start
            _, offset = self._index.split_id(node.span.start, node.span.end, split)
            column = combining_rules.index(rule)
            total = total + scores.rule[item, span, offset, column]

        return (float(total.detach().cpu().item()), admitted)

    def _unrestricted_partitions(self) -> Tensor:
        """Return the lazily computed unrestricted item partitions."""
        standing = self._unconstrained_partition

        if standing is None:
            standing = self._unconstrained.log_partition(
                self._channels,
                self._index,
                goals=self._goals,
                lengths=self._lengths,
            ).detach()
            self._unconstrained_partition = standing

        return standing

    def score(
        self, item: int, derivation: DerivationTree[T], /
    ) -> NormalFormDerivationScores:
        """Score one item's gold derivation under both normal-form filters."""
        raw, admitted = self._gold_score(item, derivation)

        return NormalFormDerivationScores(
            DerivationSetScore(
                self.normal_form,
                raw if admitted else None,
                _item_value(self._constrained_partition, item),
            ),
            DerivationSetScore(
                self._unconstrained.transitions.rule_set.normal_form.name,
                raw,
                _item_value(self._unrestricted_partitions(), item),
            ),
        )
