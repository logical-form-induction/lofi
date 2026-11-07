"""Structural reachability reports for decoded derivations."""

from typing import TYPE_CHECKING, Literal, Self, final, override

from lofi._checks import ensure_non_negative
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    from lofi.data._derivation import Derivation, NodeId, RuleName
    from lofi.data._ports import LabelMembership, StepRecovery

__all__ = [
    "DerivationOutcome",
    "DerivationOutcomeKind",
    "LabelBlocker",
    "ReachabilityReport",
    "ReachableDerivation",
    "RuleBlocker",
    "UnrecoverableStep",
    "reachability_report",
]

type DerivationOutcomeKind = Literal[
    "reachable",
    "unreachable_by_label",
    "unreachable_by_rule",
]
"""A per-derivation reachability outcome tag."""


@final
class ReachableDerivation[T: Hashable](FrozenValue):
    """A derivation with admitted labels and recoverable rules."""

    _noun = "reachability value"

    __slots__ = ()

    @property
    def kind(self) -> Literal["reachable"]:
        """Return the reachable outcome kind."""
        return "reachable"

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[()]]:
        """Return the reconstruction call."""
        return type(self), ()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another outcome is reachable."""
        if type(other) is not ReachableDerivation:
            return NotImplemented
        return True

    @override
    def __hash__(self) -> int:
        """Return the reachable-outcome hash."""
        return hash(ReachableDerivation)

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return "ReachableDerivation()"


@final
class LabelBlocker[T: Hashable](FrozenValue):
    """The first derivation node whose label is absent.

    Parameters
    ----------
    node_id : NodeId
        Identifier of the blocked node.
    label : T
        Label missing from the inventory.
    """

    _noun = "reachability value"

    __slots__ = ("_label", "_node_id")

    def __init__(self, node_id: NodeId, label: T) -> None:
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_label", label)

    @property
    def kind(self) -> Literal["unreachable_by_label"]:
        """Return the label-blocked outcome kind."""
        return "unreachable_by_label"

    @property
    def node_id(self) -> NodeId:
        """Return the blocking node identifier."""
        return self._node_id

    @property
    def label(self) -> T:
        """Return the absent label."""
        return self._label

    def _fields(self) -> tuple[NodeId, T]:
        """Return fields in reconstruction order."""
        return self._node_id, self._label

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[NodeId, T]]:
        """Return the reconstruction call."""
        return type(self), self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another blocker identifies the same label."""
        if type(other) is not LabelBlocker:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the node and label."""
        return hash((LabelBlocker, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"LabelBlocker{self._fields()!r}"


@final
class RuleBlocker[T: Hashable](FrozenValue):
    """The first derivation node with no recovering rule.

    Parameters
    ----------
    node_id : NodeId
        Identifier of the blocked node.
    inputs : Sequence[T]
        Premise labels in child order.
    output : T
        Conclusion label.
    rule : RuleName | None
        Recorded rule name, if present.
    """

    _noun = "reachability value"

    __slots__ = ("_inputs", "_node_id", "_output", "_rule")

    def __init__(
        self,
        node_id: NodeId,
        inputs: Sequence[T],
        output: T,
        rule: RuleName | None = None,
    ) -> None:
        object.__setattr__(self, "_node_id", node_id)
        object.__setattr__(self, "_inputs", tuple(inputs))
        object.__setattr__(self, "_output", output)
        object.__setattr__(self, "_rule", rule)

    @property
    def kind(self) -> Literal["unreachable_by_rule"]:
        """Return the rule-blocked outcome kind."""
        return "unreachable_by_rule"

    @property
    def node_id(self) -> NodeId:
        """Return the blocking node identifier."""
        return self._node_id

    @property
    def inputs(self) -> tuple[T, ...]:
        """Return premise labels in child order."""
        return self._inputs

    @property
    def output(self) -> T:
        """Return the conclusion label."""
        return self._output

    @property
    def rule(self) -> RuleName | None:
        """Return the recorded rule name, if present."""
        return self._rule

    def _fields(self) -> tuple[NodeId, tuple[T, ...], T, RuleName | None]:
        """Return fields in reconstruction order."""
        return self._node_id, self._inputs, self._output, self._rule

    @override
    def __reduce__(
        self,
    ) -> tuple[type[Self], tuple[NodeId, tuple[T, ...], T, RuleName | None]]:
        """Return the reconstruction call."""
        return type(self), self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another blocker identifies the same failed step."""
        if type(other) is not RuleBlocker:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the failed step."""
        return hash((RuleBlocker, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"RuleBlocker{self._fields()!r}"


type DerivationOutcome[T: Hashable] = (
    ReachableDerivation[T] | LabelBlocker[T] | RuleBlocker[T]
)
"""The reachability outcome for one decoded derivation."""


@final
class UnrecoverableStep[T: Hashable](FrozenValue):
    """An unlicensed local derivation shape and its occurrence count."""

    _noun = "reachability value"

    __slots__ = ("_inputs", "_occurrences", "_output")

    def __init__(self, inputs: Sequence[T], output: T, occurrences: int) -> None:
        held_occurrences = ensure_non_negative(
            "an unrecoverable-step occurrence count",
            occurrences,
        )

        if held_occurrences == 0:
            msg = "an unrecoverable step has no occurrences"
            raise ValueError(msg)

        object.__setattr__(self, "_inputs", tuple(inputs))
        object.__setattr__(self, "_output", output)
        object.__setattr__(self, "_occurrences", held_occurrences)

    @property
    def inputs(self) -> tuple[T, ...]:
        """Return premises in declared child order."""
        return self._inputs

    @property
    def output(self) -> T:
        """Return the unrecoverable conclusion."""
        return self._output

    @property
    def occurrences(self) -> int:
        """Return how often the exact shape failed."""
        return self._occurrences

    def _fields(self) -> tuple[tuple[T, ...], T, int]:
        """Return fields in reconstruction order."""
        return self._inputs, self._output, self._occurrences

    @override
    def __reduce__(self) -> tuple[type[Self], tuple[tuple[T, ...], T, int]]:
        """Return the reconstruction call."""
        return type(self), self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another failure has the same shape and count."""
        if type(other) is not UnrecoverableStep:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the failed shape and count."""
        return hash((UnrecoverableStep, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"UnrecoverableStep{self._fields()!r}"


@final
class ReachabilityReport[T: Hashable](FrozenValue):
    """Reachability outcomes and aggregate local-step counts.

    Parameters
    ----------
    checked_steps : int
        Number of nonlexical steps checked.
    reachable_steps : int
        Number of steps recovered by the rule set.
    checked_derivations : int
        Number of derivations checked.
    reachable_derivations : int
        Number of wholly reachable derivations.
    unrecoverable_steps : Sequence[UnrecoverableStep[T]]
        Failed rule shapes and their occurrence counts.
    outcomes : Sequence[DerivationOutcome[T]]
        One ordered outcome per checked derivation.

    Raises
    ------
    TypeError
        If a count, outcome, or unrecoverable step has the wrong type.
    ValueError
        If the counts, failed shapes, or outcomes are inconsistent.
    """

    _noun = "reachability value"

    __slots__ = (
        "_checked_derivations",
        "_checked_steps",
        "_outcomes",
        "_reachable_derivations",
        "_reachable_steps",
        "_unrecoverable_steps",
    )

    def __init__(
        self,
        checked_steps: int,
        reachable_steps: int,
        checked_derivations: int,
        reachable_derivations: int,
        unrecoverable_steps: Sequence[UnrecoverableStep[T]] = (),
        outcomes: Sequence[DerivationOutcome[T]] = (),
    ) -> None:
        held_checked_steps = ensure_non_negative("a checked-step count", checked_steps)
        held_reachable_steps = ensure_non_negative(
            "a reachable-step count",
            reachable_steps,
        )
        held_checked_derivations = ensure_non_negative(
            "a checked-derivation count",
            checked_derivations,
        )
        held_reachable_derivations = ensure_non_negative(
            "a reachable-derivation count",
            reachable_derivations,
        )

        if held_reachable_steps > held_checked_steps:
            msg = "reachable steps exceed checked steps"
            raise ValueError(msg)
        if held_reachable_derivations > held_checked_derivations:
            msg = "reachable derivations exceed checked derivations"
            raise ValueError(msg)
        if held_checked_derivations == 0 and held_checked_steps:
            msg = "checked steps require at least one checked derivation"
            raise ValueError(msg)

        held_failures = tuple(unrecoverable_steps)
        for position, failure in enumerate(held_failures):
            if not isinstance(failure, UnrecoverableStep):
                msg = (
                    f"unrecoverable step {position} is not an "
                    f"UnrecoverableStep: {failure!r}"
                )
                raise TypeError(msg)

        shapes = tuple((failure.inputs, failure.output) for failure in held_failures)
        if len(set(shapes)) != len(shapes):
            msg = "a reachability report repeats an unrecoverable shape"
            raise ValueError(msg)

        failed_steps = sum(failure.occurrences for failure in held_failures)
        if failed_steps != held_checked_steps - held_reachable_steps:
            msg = "unrecoverable-step occurrences do not account for failed steps"
            raise ValueError(msg)

        held_outcomes = tuple(outcomes)
        if len(held_outcomes) != held_checked_derivations:
            msg = "reachability outcomes do not account for checked derivations"
            raise ValueError(msg)

        for position, outcome in enumerate(held_outcomes):
            if not isinstance(
                outcome,
                (ReachableDerivation, LabelBlocker, RuleBlocker),
            ):
                msg = f"derivation outcome {position} has an invalid type: {outcome!r}"
                raise TypeError(msg)

        counted_reachable = sum(
            isinstance(outcome, ReachableDerivation) for outcome in held_outcomes
        )
        if counted_reachable != held_reachable_derivations:
            msg = "reachable outcomes do not match the reachable-derivation count"
            raise ValueError(msg)

        failed_shape_set = set(shapes)
        if any(
            isinstance(outcome, RuleBlocker)
            and (outcome.inputs, outcome.output) not in failed_shape_set
            for outcome in held_outcomes
        ):
            msg = "a rule blocker is absent from the failed-step aggregation"
            raise ValueError(msg)

        unreachable_derivations = held_checked_derivations - counted_reachable
        if failed_steps and unreachable_derivations == 0:
            msg = "failed steps require at least one unreachable derivation"
            raise ValueError(msg)

        object.__setattr__(self, "_checked_steps", held_checked_steps)
        object.__setattr__(self, "_reachable_steps", held_reachable_steps)
        object.__setattr__(self, "_checked_derivations", held_checked_derivations)
        object.__setattr__(self, "_reachable_derivations", counted_reachable)
        object.__setattr__(self, "_unrecoverable_steps", held_failures)
        object.__setattr__(self, "_outcomes", held_outcomes)

    @property
    def checked_steps(self) -> int:
        """Return the number of nonlexical nodes checked."""
        return self._checked_steps

    @property
    def reachable_steps(self) -> int:
        """Return the number of locally reachable steps."""
        return self._reachable_steps

    @property
    def checked_derivations(self) -> int:
        """Return the number of decoded derivations checked."""
        return self._checked_derivations

    @property
    def reachable_derivations(self) -> int:
        """Return the number of wholly reachable derivations."""
        return self._reachable_derivations

    @property
    def label_blocked_derivations(self) -> int:
        """Return the number of derivations blocked by a label."""
        return sum(isinstance(outcome, LabelBlocker) for outcome in self._outcomes)

    @property
    def rule_blocked_derivations(self) -> int:
        """Return the number of derivations blocked by a rule."""
        return sum(isinstance(outcome, RuleBlocker) for outcome in self._outcomes)

    @property
    def outcomes(self) -> tuple[DerivationOutcome[T], ...]:
        """Return one outcome per input derivation, in input order."""
        return self._outcomes

    @property
    def unrecoverable_steps(self) -> tuple[UnrecoverableStep[T], ...]:
        """Return failed shapes in first-observed order."""
        return self._unrecoverable_steps

    @property
    def step_reachability(self) -> float | None:
        """Return the reachable-step share, if any steps were checked."""
        if self._checked_steps == 0:
            return None
        return self._reachable_steps / self._checked_steps

    @property
    def derivation_reachability(self) -> float | None:
        """Return the reachable-derivation share, if any were checked."""
        if self._checked_derivations == 0:
            return None
        return self._reachable_derivations / self._checked_derivations

    def _fields(
        self,
    ) -> tuple[
        int,
        int,
        int,
        int,
        tuple[UnrecoverableStep[T], ...],
        tuple[DerivationOutcome[T], ...],
    ]:
        """Return fields in reconstruction order."""
        return (
            self._checked_steps,
            self._reachable_steps,
            self._checked_derivations,
            self._reachable_derivations,
            self._unrecoverable_steps,
            self._outcomes,
        )

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[Self],
        tuple[
            int,
            int,
            int,
            int,
            tuple[UnrecoverableStep[T], ...],
            tuple[DerivationOutcome[T], ...],
        ],
    ]:
        """Return the reconstruction call."""
        return type(self), self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report has the same fields."""
        if type(other) is not ReachabilityReport:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the outcomes, counts, and failed shapes."""
        return hash((ReachabilityReport, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"ReachabilityReport{self._fields()!r}"


def reachability_report[T: Hashable, Recovered](
    labels: LabelMembership[T],
    ruleset: StepRecovery[T, Recovered],
    decoded_derivations: Iterable[Derivation[T]],
    /,
) -> ReachabilityReport[T]:
    """Report label and rule reachability for decoded derivations.

    Parameters
    ----------
    labels : LabelMembership[T]
        Label membership checked for every node.
    ruleset : StepRecovery[T, Recovered]
        Recovery relation checked for every nonlexical node.
    decoded_derivations : Iterable[Derivation[T]]
        Derivations inspected in input order.

    Returns
    -------
    ReachabilityReport[T]
        Aggregate counts and one ordered outcome per derivation.
    """
    checked_steps = 0
    reachable_steps = 0
    failures: dict[tuple[tuple[T, ...], T], int] = {}
    outcomes: list[DerivationOutcome[T]] = []

    for derivation in decoded_derivations:
        label_blocker: LabelBlocker[T] | None = None
        rule_blocker: RuleBlocker[T] | None = None

        for node_id in derivation.preorder:
            node = derivation.node(node_id)

            if not labels.contains(node.label) and label_blocker is None:
                label_blocker = LabelBlocker(node_id, node.label)

            if node.is_lexical:
                continue

            inputs = tuple(child.label for child in derivation.children_of(node_id))
            output = node.label
            checked_steps += 1

            if ruleset.recover(inputs, output):
                reachable_steps += 1
                continue

            shape = inputs, output
            failures[shape] = failures.get(shape, 0) + 1
            if rule_blocker is None:
                rule_blocker = RuleBlocker(node_id, inputs, output, node.rule)

        if label_blocker is not None:
            outcomes.append(label_blocker)
        elif rule_blocker is not None:
            outcomes.append(rule_blocker)
        else:
            outcomes.append(ReachableDerivation())

    reachable_derivations = sum(
        isinstance(outcome, ReachableDerivation) for outcome in outcomes
    )
    return ReachabilityReport(
        checked_steps,
        reachable_steps,
        len(outcomes),
        reachable_derivations,
        tuple(
            UnrecoverableStep(inputs, output, occurrences)
            for (inputs, output), occurrences in failures.items()
        ),
        outcomes,
    )
