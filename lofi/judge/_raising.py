"""Type raising rules and their controller action product."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Final, Literal, Never, final, override

from lofi.judge._ports import HOLE, NormalFormRole
from lofi.judge._proof import _functor_pattern, _term_pattern
from lofi.judge._values import Action, ActionSet
from lofi.terms import MVar, TermRuleKernel

if TYPE_CHECKING:
    from collections.abc import Iterator, Mapping, Sequence

    from lofi._types import JsonValue
    from lofi.judge._index import RuleProbe
    from lofi.judge._ports import Direction, SchematicSequent, Slot
    from lofi.judge._ruleset import RuleSet
    from lofi.judge._values import RuleRestriction
    from lofi.terms import Functors, OpName, Term, TermMatching, _CompiledRule

__all__ = [
    "RaiseConfiguration",
    "TypeRaising",
    "composition_roles",
    "raising_configuration",
    "soft_action_set",
]

type RaiseConfiguration = Literal["none", "functor", "argument", "both"]

_CONFIGURATIONS: Final[tuple[RaiseConfiguration, ...]] = (
    "none",
    "functor",
    "argument",
    "both",
)
_ACTION_MARKER: Final = ":raise="


def composition_roles[T](
    rules: RuleSet[T], /
) -> Mapping[str, tuple[int, Literal["forward", "backward", "undirected"]]]:
    """Return binary composition roles keyed by rule name."""
    roles: dict[str, tuple[int, Literal["forward", "backward", "undirected"]]] = {}

    for rule in rules:
        if rule.arity != 2:
            continue

        role = rule.normal_form_role.composition
        if role is None:
            continue

        principal = role.principal_slot
        direction = role.direction
        if principal not in (0, 1):
            msg = f"composition rule {rule.name!r} does not name its two input roles"
            raise ValueError(msg)
        roles[rule.name] = (principal, direction)

    return MappingProxyType(roles)


def soft_action_set[T](actions: ActionSet, rules: RuleSet[T], /) -> ActionSet:
    """Return the combinator-by-raising product used by a soft controller."""
    roles = composition_roles(rules)
    expanded: list[Action] = []

    for action in actions.actions:
        if action.rule not in roles:
            expanded.append(action)
            continue

        expanded.extend(
            Action(f"{action.name}{_ACTION_MARKER}{configuration}", action.rule)
            for configuration in _CONFIGURATIONS
        )

    return ActionSet(expanded, actions.identity, null=actions.null)


def raising_configuration(name: str, /) -> tuple[str, RaiseConfiguration] | None:
    """Read one exact soft-action product name, or none for a base action."""
    if _ACTION_MARKER not in name:
        return None

    base, configuration = name.rsplit(_ACTION_MARKER, 1)
    if configuration not in _CONFIGURATIONS:
        msg = f"action {name!r} has unknown raising configuration {configuration!r}"
        raise ValueError(msg)

    return base, configuration


_ARITY: Final = 1
"""Unary-rule arity."""


def _term_spec(term: Term) -> JsonValue:
    """Encode a term as a recursive JSON array."""
    children: list[JsonValue] = [_term_spec(child) for child in term.args]

    return [term.op.name, term.payload, children]


@final
class TypeRaising:
    """Type raising rule with optional finite targets and arguments.

    The rule is adopted from one compiled type raising recipe. A rule with
    finite targets enumerates its conclusions forward; one without checks and
    residuates every raised shape but enumerates nothing. A rule with finite
    arguments raises only a premise its matching identifies with one of them.

    Parameters
    ----------
    compiled : _CompiledRule
        The compiled type raising recipe.
    functors : Functors[Term]
        The functor view the recipe names.
    slash : OpName
        The outer constructor of that view.
    matching : TermMatching
        The premise congruence.
    targets : Sequence[Term] | None
        The declared targets, or ``None`` for every target.
    arguments : Sequence[Term] | None
        The categories that may raise, or ``None`` for every category.
    restriction : RuleRestriction | None
        The rule instance restriction, if declared.
    """

    __slots__ = (
        "_arguments",
        "_directions",
        "_functors",
        "_lowering",
        "_matching",
        "_name",
        "_restriction",
        "_slash",
        "_targets",
    )

    def __init__(
        self,
        compiled: _CompiledRule,
        functors: Functors[Term],
        slash: OpName,
        matching: TermMatching,
        targets: Sequence[Term] | None,
        arguments: Sequence[Term] | None,
        /,
        *,
        restriction: RuleRestriction | None = None,
    ) -> None:
        held_targets = None if targets is None else tuple(targets)
        held_arguments = None if arguments is None else tuple(arguments)
        target_key = (
            "*"
            if held_targets is None
            else ",".join(str(target) for target in held_targets)
        )
        directions: frozenset[Direction] = (
            frozenset({"check", "residual"})
            if held_targets is None
            else frozenset({"forward", "check", "residual"})
        )
        name = f"type-raising:{slash}:targets={target_key}"
        if held_arguments is not None:
            name += ":arguments=" + ",".join(str(value) for value in held_arguments)
        object.__setattr__(self, "_functors", functors)
        object.__setattr__(self, "_matching", matching)
        object.__setattr__(self, "_slash", slash)
        object.__setattr__(self, "_targets", held_targets)
        object.__setattr__(self, "_arguments", held_arguments)
        object.__setattr__(self, "_directions", directions)
        object.__setattr__(self, "_restriction", restriction)
        object.__setattr__(
            self,
            "_lowering",
            TermRuleKernel(
                compiled.raising_kernel(
                    matching, functors, held_targets, held_arguments
                )
            ),
        )
        object.__setattr__(
            self,
            "_name",
            name if restriction is None else restriction.qualify(name),
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Refuse field reassignment."""
        del name, value
        msg = "a type-raising rule is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Refuse field deletion."""
        del name
        msg = "a type-raising rule is immutable"

        raise AttributeError(msg)

    @property
    def name(self) -> str:
        """Return the rule name."""
        return self._name

    @property
    def arity(self) -> int:
        """Return the premise count."""
        return _ARITY

    @property
    def directions(self) -> frozenset[Direction]:
        """Return the computable directions."""
        return self._directions

    @property
    def spine_order(self) -> int:
        """Return the spine order of every instance."""
        return 0

    @property
    def residual_slots(self) -> frozenset[int]:
        """Return the single residuable premise slot."""
        return frozenset({0})

    @property
    def forward_is_complete(self) -> bool:
        """Report whether forward enumeration is complete."""
        return self._targets is not None

    @property
    def matching(self) -> TermMatching:
        """Return the term congruence."""
        return self._matching

    @property
    def restriction(self) -> RuleRestriction | None:
        """Return the rule instance restriction, if declared."""
        return self._restriction

    @property
    def probe(self) -> RuleProbe | None:
        """Return no premise-head constraints."""
        return None

    @property
    def lowering(self) -> TermRuleKernel:
        """Return the native implementation of this relation."""
        return self._lowering

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the raising direction used by normal form filters."""
        forward = self._slash == self._functors.forward_slash
        backward = self._slash == self._functors.backward_slash
        direction = (
            "undirected"
            if forward and backward
            else "forward"
            if forward
            else "backward"
        )
        return NormalFormRole(raising=direction)

    @property
    def functors(self) -> Functors[Term]:
        """Return the functor view."""
        return self._functors

    @property
    def slash(self) -> OpName:
        """Return the outer slash."""
        return self._slash

    @property
    def targets(self) -> tuple[Term, ...] | None:
        """Return the copied finite targets, if declared."""
        return self._targets

    @property
    def arguments(self) -> tuple[Term, ...] | None:
        """Return the copied finite raisable categories, if declared."""
        return self._arguments

    @property
    def proof_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the type raising schemata."""
        sort = self._functors.category_sort
        inputs = (
            (MVar("X", sort),)
            if self._arguments is None
            else tuple(_term_pattern(argument) for argument in self._arguments)
        )
        targets = (
            (MVar("T", sort),)
            if self._targets is None
            else tuple(_term_pattern(target) for target in self._targets)
        )
        schemata: list[SchematicSequent] = []

        for target in targets:
            for input_pattern in inputs:
                inner = _functor_pattern(
                    self._functors,
                    self._functors.opposite(self._slash),
                    target,
                    input_pattern,
                )

                if inner is None:
                    continue

                conclusion = _functor_pattern(
                    self._functors,
                    self._slash,
                    target,
                    inner,
                )

                if conclusion is not None:
                    schemata.append(((input_pattern,), conclusion))

        return tuple(schemata)

    @property
    def reachability_schemata(self) -> tuple[SchematicSequent, ...]:
        """Return the consumed premises and introduced conclusions."""
        return self.proof_schemata

    def apply(self, inputs: tuple[Term, ...], /) -> Iterator[Term]:
        """Enumerate conclusions for one premise.

        A rule with no finite targets enumerates nothing forward.
        """
        yield from self._lowering.apply(inputs)

    def admits(
        self,
        inputs: tuple[Term, ...],
        output: Term,
        /,
    ) -> bool:
        """Decide whether the rule relates a premise and conclusion."""
        return self._lowering.admits(inputs, output)

    def residuate(
        self,
        output: Term,
        known: tuple[Slot[Term], ...],
        /,
    ) -> Iterator[Term]:
        """Enumerate the filler for the unary premise hole."""
        yield from self._lowering.residuate(
            output, tuple(None if value is HOLE else value for value in known)
        )

    @property
    def relation(self) -> Mapping[str, JsonValue]:
        """Return the parameters that fix the relation the rule denotes."""
        return self.to_spec()

    def to_spec(self) -> Mapping[str, JsonValue]:
        """Return the rule declaration."""
        targets: JsonValue = (
            None
            if self._targets is None
            else [_term_spec(target) for target in self._targets]
        )
        arguments: JsonValue = (
            None
            if self._arguments is None
            else [_term_spec(argument) for argument in self._arguments]
        )

        return {
            "rule": "type-raising",
            "matching": self._matching.name,
            "slash": self._slash,
            "targets": targets,
            "arguments": arguments,
            "restriction": (
                None if self._restriction is None else self._restriction.to_spec()
            ),
        }
