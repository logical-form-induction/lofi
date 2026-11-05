"""Rule-inventory values and reports."""

from hashlib import sha256
from json import dumps
from typing import TYPE_CHECKING, Final, Literal, Never, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._types import CompositionRestriction

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable, Sequence

    from lofi._types import JsonValue

__all__ = [
    "CAPACITY_RESULTS",
    "CATEGORY_SETS",
    "COMPOSITION_RESTRICTIONS",
    "RESTRICTION_MODES",
    "SEVERITIES",
    "Action",
    "ActionSet",
    "CapacityResult",
    "CategorySet",
    "ClosureReport",
    "CompositionRestriction",
    "Diagnostic",
    "RestrictionMode",
    "RuleRestriction",
    "Severity",
]


type RestrictionMode = Literal["unrestricted", "grammar_specific"]
"""Classifies how a grammar controls rule instances."""

RESTRICTION_MODES: Final[tuple[RestrictionMode, ...]] = (
    "unrestricted",
    "grammar_specific",
)
"""Supported restriction modes."""


COMPOSITION_RESTRICTIONS: Final[tuple[CompositionRestriction, ...]] = (
    "steedman",
    "kuhlmann",
    "unrestricted",
)
"""Supported composition direction restrictions."""


type CapacityResult = Literal[
    "not_characterized",
    "proper_superclass_of_context_free",
    "tag_equivalent",
    "proper_subclass_of_tag",
]
"""States a proved language-family relationship."""

CAPACITY_RESULTS: Final[tuple[CapacityResult, ...]] = (
    "not_characterized",
    "proper_superclass_of_context_free",
    "tag_equivalent",
    "proper_subclass_of_tag",
)
"""Capacity results published by the judge boundary."""


type CategorySet = Literal["finite", "infinite", "unknown"]
"""Classifies a reachable category set."""

CATEGORY_SETS: Final[tuple[CategorySet, ...]] = ("finite", "infinite", "unknown")
"""Supported reachable-category verdicts."""


type Severity = Literal["error", "warning"]
"""Classifies a diagnostic's effect."""

SEVERITIES: Final[tuple[Severity, ...]] = ("error", "warning")
"""Supported diagnostic severities."""


type _ClosureFields[T] = tuple[
    tuple[T, ...],
    bool,
    int,
    int,
    tuple[str, ...],
    tuple[T, ...],
]
"""Fields needed to reconstruct a closure report."""

type _ClosureReduction[T] = tuple[
    Callable[[_ClosureFields[T]], ClosureReport[T]],
    tuple[_ClosureFields[T]],
]
"""Closure-report pickle reduction."""


def _names(field: str, given: Iterable[str]) -> tuple[str, ...]:
    """Return validated names as a tuple.

    Parameters
    ----------
    field : str
        Field name used in error messages.
    given : Iterable[str]
        Names to validate.

    Returns
    -------
    tuple[str, ...]
        Validated names in input order.

    Raises
    ------
    TypeError
        If a name is not a string.
    ValueError
        If a name is empty.
    """
    return tuple(ensure_name(field, name) for name in given)


type _RestrictionFields = tuple[tuple[str, ...], tuple[str, ...]]
"""Fields needed to reconstruct a rule restriction."""

type _RestrictionReduction = tuple[
    Callable[[_RestrictionFields], RuleRestriction],
    tuple[_RestrictionFields],
]
"""Rule-restriction pickle reduction."""


class RuleRestriction:
    """Rule-instance premise-category restriction.

    Parameters
    ----------
    primary_targets : Iterable[str]
        Permitted primary-category targets.
    secondary_patterns : Iterable[str]
        Permitted secondary categories.

    Raises
    ------
    TypeError
        If a restriction value is not a string.
    ValueError
        If a value is empty or both restriction sets are empty.
    """

    __slots__ = ("_primary_targets", "_secondary_patterns")

    def __init__(
        self,
        primary_targets: Iterable[str] = (),
        secondary_patterns: Iterable[str] = (),
    ) -> None:
        held_targets = frozenset(_names("a restricted primary target", primary_targets))
        held_patterns = frozenset(
            _names("a restricted secondary pattern", secondary_patterns)
        )

        if not held_targets and not held_patterns:
            msg = "a rule restriction has no primary target or pattern"
            raise ValueError(msg)

        object.__setattr__(self, "_primary_targets", held_targets)
        object.__setattr__(self, "_secondary_patterns", held_patterns)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a rule restriction is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a rule restriction is immutable"

        raise AttributeError(msg)

    @property
    def primary_targets(self) -> frozenset[str]:
        """Return the permitted primary-category targets."""
        return self._primary_targets

    @property
    def secondary_patterns(self) -> frozenset[str]:
        """Return the permitted secondary-category patterns."""
        return self._secondary_patterns

    def qualify(self, name: str, /) -> str:
        """Return a rule name extended with this restriction.

        Two rules that differ only in their restriction license different
        steps, so the restriction is part of the rule's identity. Each
        nonempty field is appended in sorted order.

        Parameters
        ----------
        name : str
            The unrestricted rule name.

        Returns
        -------
        str
            The name followed by ``:only(...)`` listing the nonempty fields.
        """
        fields = (
            ("targets", self._primary_targets),
            ("patterns", self._secondary_patterns),
        )
        parts = ";".join(
            f"{field}={','.join(sorted(values))}" for field, values in fields if values
        )

        return f"{name}:only({parts})"

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the canonical JSON declaration."""
        return {
            "primary_targets": sorted(self._primary_targets),
            "secondary_patterns": sorted(self._secondary_patterns),
        }

    @override
    def __reduce__(self) -> _RestrictionReduction:
        """Return the reconstruction call."""
        fields: _RestrictionFields = (
            tuple(sorted(self._primary_targets)),
            tuple(sorted(self._secondary_patterns)),
        )

        return (_restore_rule_restriction, (fields,))

    @override
    def __eq__(self, other: object) -> bool:
        """Compare every restriction field."""
        if type(other) is not RuleRestriction:
            return NotImplemented

        return (
            self._primary_targets == other._primary_targets
            and self._secondary_patterns == other._secondary_patterns
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over every restriction field."""
        return hash(
            (
                RuleRestriction,
                self._primary_targets,
                self._secondary_patterns,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return (
            "RuleRestriction("
            f"primary_targets={sorted(self._primary_targets)!r}, "
            f"secondary_patterns={sorted(self._secondary_patterns)!r})"
        )


def _restore_rule_restriction(fields: _RestrictionFields) -> RuleRestriction:
    """Reconstruct a rule restriction from frozen fields."""
    primary_targets, secondary_patterns = fields

    return RuleRestriction(primary_targets, secondary_patterns)


class Diagnostic:
    """Rule-inventory finding.

    Parameters
    ----------
    code : str
        Stable finding code the requirement assigns.
    requirement : str
        Name of the requirement the inventory was checked against.
    message : str
        Human-readable finding.
    severity : Severity
        Whether the finding stops a run or emits a warning.
    rules : Iterable[str]
        Affected rule names.

    Raises
    ------
    TypeError
        If a name or slot has the wrong type.
    ValueError
        If an enum value, name, or slot is invalid.
    """

    __slots__ = (
        "_code",
        "_message",
        "_requirement",
        "_rules",
        "_severity",
    )

    def __init__(
        self,
        code: str,
        requirement: str,
        message: str,
        *,
        severity: Severity = "error",
        rules: Iterable[str] = (),
    ) -> None:
        if severity not in SEVERITIES:
            declared = ", ".join(SEVERITIES)
            msg = f"a diagnostic's severity is {severity!r}, not one of {declared}"
            raise ValueError(msg)

        self._code = ensure_name("a diagnostic's code", code)
        self._requirement = ensure_name("a diagnostic's requirement", requirement)
        self._severity: Severity = severity
        self._message = ensure_name("a diagnostic's message", message)
        self._rules = _names("a diagnosed rule", rules)

    @property
    def code(self) -> str:
        """Return the stable diagnostic code."""
        return self._code

    @property
    def requirement(self) -> str:
        """Return the name of the requirement the finding concerns."""
        return self._requirement

    @property
    def severity(self) -> Severity:
        """Return the diagnostic severity."""
        return self._severity

    @property
    def message(self) -> str:
        """Return the diagnostic message."""
        return self._message

    @property
    def rules(self) -> tuple[str, ...]:
        """Return the names of the rules the finding concerns."""
        return self._rules

    @override
    def __eq__(self, other: object) -> bool:
        """Compare all diagnostic fields."""
        if type(other) is not Diagnostic:
            return NotImplemented

        return (
            self._code == other._code
            and self._requirement == other._requirement
            and self._severity == other._severity
            and self._message == other._message
            and self._rules == other._rules
        )

    @override
    def __hash__(self) -> int:
        """Hash all diagnostic fields."""
        return hash(
            (
                Diagnostic,
                self._code,
                self._requirement,
                self._severity,
                self._message,
                self._rules,
            )
        )

    @override
    def __repr__(self) -> str:
        """Render the rebuilding constructor call."""
        return (
            f"Diagnostic({self._code!r}, {self._requirement!r}, {self._message!r}, "
            f"severity={self._severity!r}, rules={list(self._rules)!r})"
        )


class Action:
    """Controller output column.

    Parameters
    ----------
    name : str
        Unique action name.
    rule : str | None
        Applied rule name, or ``None`` for the identity action.

    Raises
    ------
    TypeError
        If a name has the wrong type.
    ValueError
        If a name is empty.
    """

    __slots__ = ("_name", "_rule")

    def __init__(self, name: str, rule: str | None = None) -> None:
        self._name = ensure_name("an action's name", name)
        self._rule = None if rule is None else ensure_name("an action's rule", rule)

    @property
    def name(self) -> str:
        """Return the action name."""
        return self._name

    @property
    def rule(self) -> str | None:
        """Return the rule name, or None for the identity action."""
        return self._rule

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another action has the same name and rule."""
        if type(other) is not Action:
            return NotImplemented

        return self._name == other._name and self._rule == other._rule

    @override
    def __hash__(self) -> int:
        """Return a hash over the name and the rule."""
        return hash((Action, self._name, self._rule))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this action."""
        return f"Action({self._name!r}, {self._rule!r})"


class ActionSet:
    """Ordered controller action set.

    Two actions apply no rule. The identity leaves a cell's category as it is,
    and the optional null action is the column a controller weighs when no
    combinator applies to a pair of cells. Every other action composes under the
    rule it names.

    Parameters
    ----------
    actions : Sequence[Action]
        Actions in controller-column order.
    identity : Action
        Identity action included among ``actions``.
    null : Action | None
        Null action included among ``actions``, or None where the set declares
        none.

    Raises
    ------
    ValueError
        If the set is empty, repeats a name, omits ``identity`` or ``null``, or
        declares one action as both, or a null action that applies a rule.
    """

    __slots__ = ("_actions", "_columns", "_identity", "_null")

    def __init__(
        self,
        actions: Sequence[Action],
        identity: Action,
        *,
        null: Action | None = None,
    ) -> None:
        ordered = tuple(actions)

        if not ordered:
            msg = "an action set has no actions"
            raise ValueError(msg)

        columns: dict[str, int] = {}

        for column, action in enumerate(ordered):
            if action.name in columns:
                msg = (
                    f"the action {action.name!r} appears at columns "
                    f"{columns[action.name]} and {column}"
                )
                raise ValueError(msg)

            columns[action.name] = column

        if identity not in ordered:
            msg = f"the identity action {identity.name!r} is not one of the columns"
            raise ValueError(msg)

        if null is not None:
            if null not in ordered:
                msg = f"the null action {null.name!r} is not one of the columns"
                raise ValueError(msg)

            if null == identity:
                msg = f"the action {null.name!r} cannot be both identity and null"
                raise ValueError(msg)

            if null.rule is not None:
                msg = f"the null action {null.name!r} applies the rule {null.rule!r}"
                raise ValueError(msg)

        self._actions = ordered
        self._identity = identity
        self._null = null
        self._columns = columns

    @classmethod
    def derive(
        cls,
        rules: Iterable[str],
        /,
        *,
        identity: str,
        null: str | None,
    ) -> ActionSet:
        """Return the action set a rule inventory and its declared extras define.

        Parameters
        ----------
        rules : Iterable[str]
            The names of the grammar's rules, in declaration order.
        identity : str
            The name of the identity action.
        null : str | None
            The name of the null action, or None where the learner declares none.

        Returns
        -------
        ActionSet
            The identity in column zero, then one action per rule named after the
            rule it applies, then the null action where one is declared.

        Raises
        ------
        ValueError
            If a rule name repeats or collides with the identity or null name.
        """
        identity_action = Action(identity)
        null_action = None if null is None else Action(null)
        trailing = () if null_action is None else (null_action,)

        return cls(
            (identity_action, *(Action(rule, rule) for rule in rules), *trailing),
            identity_action,
            null=null_action,
        )

    @property
    def actions(self) -> tuple[Action, ...]:
        """Return actions in controller-column order."""
        return self._actions

    @property
    def identity(self) -> Action:
        """Return the identity action."""
        return self._identity

    @property
    def null(self) -> Action | None:
        """Return the null action, or None where the set declares none."""
        return self._null

    @property
    def composing(self) -> tuple[Action, ...]:
        """Return every action but the identity and the null, in column order."""
        return tuple(
            action
            for action in self._actions
            if action not in (self._identity, self._null)
        )

    @property
    def digest(self) -> str:
        """Return a checkpoint digest of the action order."""
        declared: list[JsonValue] = [
            [[action.name, action.rule] for action in self._actions],
            self._identity.name,
        ]

        if self._null is not None:
            declared.append(self._null.name)

        payload = dumps(declared, ensure_ascii=False, separators=(",", ":"))

        return sha256(payload.encode()).hexdigest()

    def index(self, name: str) -> int:
        """Return the column occupied by an action.

        Parameters
        ----------
        name : str
            Action name.

        Returns
        -------
        int
            Zero-based controller column.

        Raises
        ------
        KeyError
            If the set has no action named ``name``.
        """
        try:
            return self._columns[name]

        except KeyError:
            declared = ", ".join(action.name for action in self._actions)
            msg = f"no action is named {name!r}; the set declares {declared}"

            raise KeyError(msg) from None

    @override
    def __eq__(self, other: object) -> bool:
        """Compare the ordered actions, the identity and the null action."""
        if type(other) is not ActionSet:
            return NotImplemented

        return (
            self._actions == other._actions
            and self._identity == other._identity
            and self._null == other._null
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the ordering, the identity and the null action."""
        return hash((ActionSet, self._actions, self._identity, self._null))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this set."""
        declared = f"{list(self._actions)!r}, {self._identity!r}"

        if self._null is not None:
            declared = f"{declared}, null={self._null!r}"

        return f"ActionSet({declared})"


class ClosureReport[T]:
    """Forward-chaining result.

    Parameters
    ----------
    reached : Iterable[T]
        Derived categories, including seeds.
    saturated : bool
        Whether the closure reached a fixpoint.
    steps : int
        Number of chaining steps.
    budget : int
        Node cap used by the closure.
    never_applicable : Iterable[str]
        Rule names that never fired.
    escaping : Iterable[T]
        Categories excluded from further expansion.

    Raises
    ------
    TypeError
        If a field has the wrong type.
    ValueError
        If a count is negative or a rule name is empty.
    """

    __slots__ = (
        "_budget",
        "_escaping",
        "_never_applicable",
        "_reached",
        "_saturated",
        "_steps",
    )

    def __init__(
        self,
        reached: Iterable[T],
        *,
        saturated: bool,
        steps: int,
        budget: int,
        never_applicable: Iterable[str] = (),
        escaping: Iterable[T] = (),
    ) -> None:
        if not isinstance(saturated, bool):
            msg = f"a closure's saturation is not a boolean: {saturated!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_reached", tuple(reached))
        object.__setattr__(self, "_saturated", saturated)
        object.__setattr__(
            self,
            "_steps",
            ensure_non_negative("a closure's step count", steps),
        )
        object.__setattr__(
            self,
            "_budget",
            ensure_non_negative("a closure's budget", budget),
        )
        object.__setattr__(
            self,
            "_never_applicable",
            _names("an inapplicable rule", never_applicable),
        )
        object.__setattr__(self, "_escaping", tuple(escaping))

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a closure report is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a closure report is immutable"

        raise AttributeError(msg)

    @property
    def reached(self) -> tuple[T, ...]:
        """Return derived categories, including seeds."""
        return self._reached

    @property
    def saturated(self) -> bool:
        """Return whether the closure reached a fixpoint."""
        return self._saturated

    @property
    def steps(self) -> int:
        """Return the number of chaining steps."""
        return self._steps

    @property
    def budget(self) -> int:
        """Return the closure node cap."""
        return self._budget

    @property
    def never_applicable(self) -> tuple[str, ...]:
        """Return the names of the rules that never fired."""
        return self._never_applicable

    @property
    def escaping(self) -> tuple[T, ...]:
        """Return categories excluded from further expansion."""
        return self._escaping

    @property
    def category_set(self) -> CategorySet:
        """Return the verdict on the reachable category set."""
        if not self._saturated:
            return "unknown"

        return "infinite" if self._escaping else "finite"

    @override
    def __reduce__(self) -> _ClosureReduction[T]:
        """Return the closure-report reconstruction call."""
        fields: _ClosureFields[T] = (
            self._reached,
            self._saturated,
            self._steps,
            self._budget,
            self._never_applicable,
            self._escaping,
        )

        return (_restore_closure_report, (fields,))

    @override
    def __repr__(self) -> str:
        """Render summary counts without listing categories."""
        return (
            f"ClosureReport(reached={len(self._reached)}, "
            f"saturated={self._saturated!r}, steps={self._steps!r}, "
            f"budget={self._budget!r}, "
            f"never_applicable={list(self._never_applicable)!r}, "
            f"escaping={len(self._escaping)}, "
            f"category_set={self.category_set!r})"
        )


def _restore_closure_report[T](fields: _ClosureFields[T]) -> ClosureReport[T]:
    """Reconstruct a closure report from frozen fields."""
    reached, saturated, steps, budget, never_applicable, escaping = fields

    return ClosureReport(
        reached,
        saturated=saturated,
        steps=steps,
        budget=budget,
        never_applicable=never_applicable,
        escaping=escaping,
    )
