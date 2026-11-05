"""Controller supervision from grammar rules."""

from itertools import product
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, override

from lofi._checks import ensure_non_negative
from lofi.judge import recover

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.judge import ActionSet, RuleSet
    from lofi.terms import Matching

__all__ = ["ControllerExample", "action_columns", "negatives", "positives"]

_PAIR: Final[int] = 2
"""The cell count scored per action."""


def action_columns[T](
    action_set: ActionSet, rule_set: RuleSet[T]
) -> Mapping[str, tuple[int, ...]]:
    """Return every column each rule of the inventory is labeled with.

    Parameters
    ----------
    action_set : ActionSet
        The actions the controller weighs, in its column order.
    rule_set : RuleSet[T]
        The inventory the supervision is generated from.

    Returns
    -------
    Mapping[str, tuple[int, ...]]
        The columns of each rule's actions, keyed by the rule's name.

    Raises
    ------
    ValueError
        If rule columns are undeclared or missing for a combining rule.
    """
    gathered: dict[str, list[int]] = {}

    for column, action in enumerate(action_set.actions):
        if action.rule is None:
            continue

        gathered.setdefault(action.rule, []).append(column)

    columns = {name: tuple(found) for name, found in gathered.items()}

    declared = {rule.name for rule in rule_set}
    stray = sorted(name for name in columns if name not in declared)

    if stray:
        msg = (
            f"the actions apply {stray}, which the rule set "
            f"{rule_set.name!r} does not declare"
        )
        raise ValueError(msg)

    unlabeled = sorted(
        rule.name for rule in rule_set if rule.arity > 1 and rule.name not in columns
    )

    if unlabeled:
        msg = (
            f"the rule set {rule_set.name!r} combines premises by {unlabeled}, "
            f"which no action applies, so a step they admit has no column"
        )
        raise ValueError(msg)

    return MappingProxyType(columns)


class ControllerExample[T]:
    """Stores one controller training example.

    Attributes
    ----------
    left : T
        The left cell of the pair.
    right : T
        The right cell of the pair.
    output : T
        The conclusion the pair is labeled against.
    columns : tuple[int, ...]
        The columns of actions applying the step's rules, ascending and distinct, or
        empty when the inventory excludes the step.

    Parameters
    ----------
    left : T
        The left cell of the pair.
    right : T
        The right cell of the pair.
    output : T
        The conclusion the pair is labeled against.
    columns : tuple[int, ...]
        The accepting columns, ascending and distinct.

    Raises
    ------
    TypeError
        If a column is not an integer.
    ValueError
        If a column is negative, or if the columns repeat or do not ascend.
    """

    __slots__ = ("_columns", "_left", "_output", "_right")

    def __init__(
        self, left: T, *, right: T, output: T, columns: tuple[int, ...]
    ) -> None:
        checked = tuple(
            ensure_non_negative("an accepting column", column) for column in columns
        )

        if tuple(sorted(set(checked))) != checked:
            msg = f"the accepting columns ascend and do not repeat: {checked}"
            raise ValueError(msg)

        self._left = left
        self._right = right
        self._output = output
        self._columns = checked

    @property
    def left(self) -> T:
        """Return the left cell of the pair."""
        return self._left

    @property
    def right(self) -> T:
        """Return the right cell of the pair."""
        return self._right

    @property
    def output(self) -> T:
        """Return the conclusion the pair is labeled against."""
        return self._output

    @property
    def columns(self) -> tuple[int, ...]:
        """Return accepted columns in ascending order."""
        return self._columns

    @property
    def is_positive(self) -> bool:
        """Return whether any action of the set accepts this step."""
        return bool(self._columns)

    @override
    def __repr__(self) -> str:
        """Return the pair, the conclusion and the columns accepting it."""
        return (
            f"ControllerExample({self._left!r}, right={self._right!r}, "
            f"output={self._output!r}, columns={self._columns!r})"
        )


def _distinct[T](matching: Matching[T], terms: tuple[T, ...]) -> tuple[T, ...]:
    """Return pairwise-inequivalent terms in input order.

    Parameters
    ----------
    matching : Matching[T]
        The congruence the rule set reads its terms under.
    terms : tuple[T, ...]
        The terms to thin.

    Returns
    -------
    tuple[T, ...]
        One representative of each class the terms fall in, in first-seen order.
    """
    kept: list[T] = []

    for term in terms:
        if not any(matching.equivalent(term, seen) for seen in kept):
            kept.append(term)

    return tuple(kept)


def _accepting[T](
    rule_set: RuleSet[T],
    columns: Mapping[str, tuple[int, ...]],
    pair: tuple[T, T],
    output: T,
) -> tuple[int, ...]:
    """Return the columns of the rules admitting one step, ascending.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The inventory to search.
    columns : Mapping[str, tuple[int, ...]]
        Every column of the actions a rule applies.
    pair : tuple[T, T]
        The two premises, left first.
    output : T
        The conclusion.

    Returns
    -------
    tuple[int, ...]
        The accepting columns, ascending and distinct.
    """
    return tuple(
        sorted(
            {
                column
                for rule in recover(rule_set, pair, output)
                for column in columns[rule.name]
            }
        )
    )


def positives[T](
    rule_set: RuleSet[T], /, *, action_set: ActionSet, cells: Sequence[T]
) -> tuple[ControllerExample[T], ...]:
    """Return the labeled steps the inventory licenses over a pool of cells.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The inventory the supervision is generated from.
    action_set : ActionSet
        The actions the controller weighs, in its column order.
    cells : Sequence[T]
        The terms to draw the pairs from.

    Returns
    -------
    tuple[ControllerExample[T], ...]
        One example per pair and distinct conclusion, each with at least one column.

    Raises
    ------
    ValueError
        If the two declarations disagree, as ``action_columns`` describes, if the
        inventory relates a pair by a rule that does not compute the checking direction.
    """
    columns = action_columns(action_set, rule_set)
    binary = rule_set.of_arity(_PAIR)
    # check the supplied premise-conclusion pair
    unlabelable = sorted(rule.name for rule in binary if "check" not in rule.directions)

    if unlabelable:
        msg = (
            f"the rule set {rule_set.name!r} relates a pair by {unlabelable}, "
            f"which do not check, so no step they license can be labeled"
        )
        raise ValueError(msg)

    # enumerate conclusions from the premise tuple
    licensing = tuple(rule for rule in binary if "forward" in rule.directions)
    examples: list[ControllerExample[T]] = []

    for left, right in product(cells, repeat=_PAIR):
        pair = (left, right)
        conclusions = _distinct(
            rule_set.matching,
            tuple(output for rule in licensing for output in rule.apply(pair)),
        )

        for output in conclusions:
            accepting = _accepting(rule_set, columns, pair, output)

            if not accepting:
                msg = (
                    f"the rule set {rule_set.name!r} licenses {pair!r} to "
                    f"{output!r} and then admits it under no rule"
                )
                raise ValueError(msg)

            examples.append(
                ControllerExample(left, right=right, output=output, columns=accepting)
            )

    return tuple(examples)


def _distinct_anchors[T](
    matching: Matching[T], anchors: tuple[tuple[T, T], ...]
) -> tuple[tuple[T, T], ...]:
    """Return pairwise-inequivalent anchors in input order.

    Parameters
    ----------
    matching : Matching[T]
        The congruence the rule set reads its terms under.
    anchors : tuple[tuple[T, T], ...]
        One kept daughter and one conclusion per licensed step.

    Returns
    -------
    tuple[tuple[T, T], ...]
        One representative of each class the anchors fall in.
    """
    kept: list[tuple[T, T]] = []

    for daughter, output in anchors:
        if not any(
            matching.equivalent(daughter, seen) and matching.equivalent(output, drawn)
            for seen, drawn in kept
        ):
            kept.append((daughter, output))

    return tuple(kept)


def _anchored[T](
    matching: Matching[T], anchors: tuple[tuple[T, T], ...], daughter: T, output: T
) -> bool:
    """Return whether one daughter and one conclusion are among the anchors.

    Parameters
    ----------
    matching : Matching[T]
        The congruence the rule set reads its terms under.
    anchors : tuple[tuple[T, T], ...]
        The anchors to search.
    daughter : T
        The kept daughter.
    output : T
        The conclusion.

    Returns
    -------
    bool
        Whether the anchors hold an equivalent pair.
    """
    return any(
        matching.equivalent(daughter, seen) and matching.equivalent(output, drawn)
        for seen, drawn in anchors
    )


def negatives[T](
    rule_set: RuleSet[T], /, *, action_set: ActionSet, cells: Sequence[T]
) -> tuple[ControllerExample[T], ...]:
    """Return negative examples produced by replacing one daughter.

    Parameters
    ----------
    rule_set : RuleSet[T]
        The inventory the supervision is generated from.
    action_set : ActionSet
        The actions the controller weighs, in its column order.
    cells : Sequence[T]
        The terms to draw the pairs and the replacements from.

    Returns
    -------
    tuple[ControllerExample[T], ...]
        One example per distinct refused step, each with no column, the steps with a
        replaced left daughter first.

    Raises
    ------
    ValueError
        If the two declarations disagree, as ``action_columns`` describes.
    """
    matching = rule_set.matching
    accepted = positives(rule_set, action_set=action_set, cells=cells)
    keeping_right = _distinct_anchors(
        matching, tuple((example.right, example.output) for example in accepted)
    )
    keeping_left = _distinct_anchors(
        matching, tuple((example.left, example.output) for example in accepted)
    )
    refused: list[ControllerExample[T]] = []

    for kept, output in keeping_right:
        refused.extend(
            ControllerExample(cell, right=kept, output=output, columns=())
            for cell in cells
            if not recover(rule_set, (cell, kept), output)
        )

    for kept, output in keeping_left:
        for cell in cells:
            # skip right daughters already enumerated as anchors
            if _anchored(matching, keeping_right, cell, output):
                continue

            if not recover(rule_set, (kept, cell), output):
                refused.append(
                    ControllerExample(kept, right=cell, output=output, columns=())
                )

    return tuple(refused)
