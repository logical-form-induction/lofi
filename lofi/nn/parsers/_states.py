"""Category and normal-form state addressing for tagged charts."""

from typing import TYPE_CHECKING, Final

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.judge import Rule, RuleSet

__all__ = [
    "BINARY",
    "UNARY",
    "category_index",
    "checked_categories",
    "checked_state",
    "distinct_outputs",
    "rules_by_arity",
]

UNARY: Final = 1
"""The arity a unary score channel weights."""

BINARY: Final = 2
"""The arity a split-indexed score channel weights."""

_STATE_SIZE: Final = 2
"""The number of indices in a category and tag state."""


def checked_state(name: str, state: tuple[int, int], /) -> tuple[int, int]:
    """Return a category and tag index pair.

    Raises
    ------
    TypeError
        If the state is not a pair.
    ValueError
        If either index is negative.
    """
    if not isinstance(state, tuple) or len(state) != _STATE_SIZE:
        msg = f"{name} is not a category and tag pair: {state!r}"
        raise TypeError(msg)

    category, tag = state

    for field, value in ((f"{name} category", category), (f"{name} tag", tag)):
        if isinstance(value, bool) or not isinstance(value, int):
            msg = f"{field} is not an integer: {value!r}"
            raise TypeError(msg)

        if value < 0:
            msg = f"{field} is negative: {value!r}"
            raise ValueError(msg)

    return (category, tag)


def category_index[T](
    categories: tuple[T, ...], rule_set: RuleSet[T], output: T, /
) -> int | None:
    """Return the column of the category a conclusion matches, or None."""
    for index, category in enumerate(categories):
        if rule_set.matching.equivalent(category, output):
            return index

    return None


def checked_categories[T](
    categories: Sequence[T], rule_set: RuleSet[T], /
) -> tuple[T, ...]:
    """Return the category inventory in tensor-axis order.

    Raises
    ------
    ValueError
        If the inventory is empty, or if two of its entries are equivalent under the
        rule set's matching and would address one column.
    """
    held = tuple(categories)

    if not held:
        msg = "a tagged chart needs at least one category"
        raise ValueError(msg)

    for right, category in enumerate(held):
        for left in range(right):
            if rule_set.matching.equivalent(held[left], category):
                msg = (
                    f"categories at positions {left} and {right} are equivalent "
                    f"under {rule_set.matching.name!r}"
                )
                raise ValueError(msg)

    return held


def distinct_outputs[T](
    rule: Rule[T],
    inputs: tuple[T, ...],
    categories: tuple[T, ...],
    rule_set: RuleSet[T],
    /,
) -> tuple[tuple[int, T], ...]:
    """Return one column and conclusion per distinct class the rule licenses."""
    if "forward" not in rule.directions or not rule.forward_is_complete:
        return tuple(
            (index, output)
            for index, output in enumerate(categories)
            if rule.admits(inputs, output)
        )

    found: list[tuple[int, T]] = []
    seen: set[int] = set()

    for output in rule.apply(inputs):
        index = category_index(categories, rule_set, output)

        if index is not None and index not in seen:
            seen.add(index)
            found.append((index, output))

    return tuple(found)


def _checked_compilable[T](rules: tuple[Rule[T], ...], /) -> tuple[Rule[T], ...]:
    """Return rules that decide every candidate conclusion.

    Raises
    ------
    ValueError
        If a rule can neither enumerate nor check the conclusions a tensor transition
        table needs.
    """
    for rule in rules:
        enumerable = "forward" in rule.directions and rule.forward_is_complete

        if not enumerable and "check" not in rule.directions:
            msg = (
                f"rule {rule.name!r} cannot decide every conclusion "
                "needed to compile tensor transitions"
            )
            raise ValueError(msg)

    return rules


def rules_by_arity[T](
    rule_set: RuleSet[T], /
) -> tuple[tuple[Rule[T], ...], tuple[Rule[T], ...]]:
    """Return the rule set's unary rules and then its combining rules.

    Raises
    ------
    ValueError
        If the set declares a nullary rule, or if a rule can neither enumerate nor
        check its conclusions.
    """
    unsupported = tuple(rule for rule in rule_set if rule.arity < UNARY)

    if unsupported:
        listed = ", ".join(
            f"{rule.name!r} (arity {rule.arity})" for rule in unsupported
        )
        msg = f"a tagged chart needs positive-arity rules, so it cannot place {listed}"
        raise ValueError(msg)

    combining = tuple(rule for rule in rule_set if rule.arity > UNARY)

    return (
        _checked_compilable(rule_set.of_arity(UNARY)),
        _checked_compilable(combining),
    )
