"""Symbolic type raising and differentiable synthetic-tree continuation."""

from __future__ import annotations

from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, NamedTuple, Protocol, cast, runtime_checkable

import torch
from torch import nn

from lofi.judge import Action, ActionSet
from lofi.judge._raising import (
    RaiseConfiguration,
    composition_roles,
    raising_configuration,
)
from lofi.nn._values import Encoded
from lofi.nn.ports import SyntheticTreeEncoder
from lofi.terms import BatchEncoding, EncodingOverflowError

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

    from lofi.judge import NormalFormRole, Rule, RuleSet
    from lofi.nn.build import BuildContext
    from lofi.nn.ports import TermEncoder
    from lofi.spec.controllers import ControllerInput, ControllerRaising
    from lofi.terms import Functors, OpName, Path, Term, TermCodec

__all__ = [
    "SoftControllerRaising",
    "UnaryReencoding",
    "build_soft_controller_raising",
    "mix_raised",
    "reencode_unary",
    "reencode_unary_domain",
]

type _NormalFormDirection = Literal["forward", "backward", "undirected"]


@runtime_checkable
class _RaisingRule(Protocol):
    """A rule exposing the symbolic type-raising construction."""

    @property
    def name(self) -> str:
        """Return the rule name."""

    @property
    def normal_form_role(self) -> NormalFormRole:
        """Return the raising direction."""

    @property
    def functors(self) -> Functors[Term]:
        """Return the rule's functor algebra."""

    @property
    def slash(self) -> OpName:
        """Return the outer slash of a raised category."""

    @property
    def targets(self) -> tuple[Term, ...] | None:
        """Return the declared targets, or none for unrestricted targets."""

    @property
    def arguments(self) -> tuple[Term, ...] | None:
        """Return the declared arguments, or none for unrestricted arguments."""


class _ActionPlan(NamedTuple):
    """One product-action column and the transformation it scores."""

    base: int
    configuration: RaiseConfiguration
    principal: int
    direction: _NormalFormDirection


class UnaryReencoding(NamedTuple):
    """Encoded unary conclusions and the source rows in the rule's domain."""

    encoded: Encoded
    source_rows: tuple[int, ...]


def _raising_rules[T](rules: RuleSet[T], /) -> tuple[_RaisingRule, ...]:
    """Return the rules publishing the complete raising construction."""
    found: list[_RaisingRule] = []

    for rule in rules:
        if rule.normal_form_role.raising is None:
            continue

        if not isinstance(rule, _RaisingRule):
            msg = (
                f"raising rule {rule.name!r} does not publish the symbolic "
                "type-raising capability"
            )
            raise TypeError(msg)

        if rule.arguments is not None:
            msg = (
                "soft controller raising cannot enforce restricted arguments "
                f"for rule {rule.name!r}; its inputs carry no symbolic categories"
            )
            raise ValueError(msg)

        found.append(rule)

    if not found:
        msg = "soft controller raising requires a rule with normal-form role 'raising'"
        raise ValueError(msg)

    return tuple(found)


def _primitive_targets(categories: Sequence[Term], /) -> tuple[Term, ...]:
    """Return represented primitive category types in category order."""
    targets = tuple(
        dict.fromkeys(category for category in categories if not category.args)
    )

    if not targets:
        msg = "soft controller raising requires a represented primitive category type"
        raise ValueError(msg)

    return targets


def _rules_by_direction(
    raising: Sequence[_RaisingRule], plans: Sequence[_ActionPlan], /
) -> Mapping[_NormalFormDirection, _RaisingRule]:
    """Bind every scored composition direction to one raising construction."""
    bound: dict[_NormalFormDirection, _RaisingRule] = {}
    directions = dict.fromkeys(
        plan.direction for plan in plans if plan.configuration != "none"
    )

    for direction in directions:
        candidates = tuple(
            rule
            for rule in raising
            if rule.normal_form_role.raising in (direction, "undirected")
        )

        if len(candidates) != 1:
            msg = (
                f"soft controller raising needs exactly one rule for {direction!r} "
                f"composition, not {len(candidates)}"
            )
            raise ValueError(msg)

        bound[direction] = candidates[0]

    return MappingProxyType(bound)


def _target_columns(
    rules: Mapping[_NormalFormDirection, _RaisingRule],
    categories: Sequence[Term],
    /,
) -> tuple[tuple[Term, ...], Mapping[_NormalFormDirection, tuple[int, ...]]]:
    """Index the union of declared targets and retain each direction's support."""
    targets: dict[Term, int] = {}
    columns: dict[_NormalFormDirection, tuple[int, ...]] = {}

    for direction, rule in rules.items():
        allowed = (
            _primitive_targets(categories) if rule.targets is None else rule.targets
        )

        if not allowed:
            msg = f"soft controller raising rule {rule.name!r} has no targets"
            raise ValueError(msg)

        columns[direction] = tuple(
            dict.fromkeys(
                targets.setdefault(target, len(targets)) for target in allowed
            )
        )

    return tuple(targets), MappingProxyType(columns)


def mix_raised(
    rule: _RaisingRule,
    targets: Sequence[Term],
    cells: Encoded,
    probabilities: Tensor,
    /,
    *,
    codec: TermCodec,
    encoder: SyntheticTreeEncoder,
) -> Encoded:
    """Mix target-specific synthetic continuations of live cell vectors."""
    held_targets = tuple(targets)
    expected = (cells.tensor.shape[0], len(held_targets))

    if tuple(probabilities.shape) != expected:
        msg = (
            f"raise-with probabilities have shape {tuple(probabilities.shape)}, "
            f"not {expected}"
        )
        raise ValueError(msg)

    wrappers: list[Term] = []
    leaf_slots: list[int] = []

    for target in held_targets:
        inner = rule.functors.make(rule.functors.opposite(rule.slash), target, target)
        wrapper = rule.functors.make(rule.slash, target, inner)
        source_path: Path = (
            *rule.functors.argument_path,
            *rule.functors.argument_path,
        )
        wrappers.append(wrapper)
        leaf_slots.append(codec.path_to_slot(wrapper)[source_path])

    rows = cells.tensor.shape[0]
    structure = codec.encode_batch(tuple(wrappers) * rows)
    repeated = Encoded(
        cells.tensor.unsqueeze(1)
        .expand(rows, len(held_targets), cells.size("width"))
        .reshape(rows * len(held_targets), cells.size("width")),
        ("term", "width"),
    )
    raised = encoder.continue_synthetic(
        structure,
        repeated,
        leaf_slots=tuple(leaf_slots) * rows,
    ).tensor.reshape(rows, len(held_targets), cells.size("width"))

    return Encoded(
        (probabilities.unsqueeze(-1) * raised).sum(dim=1), cells.axes, cells.mask
    )


def _plans[T](
    actions: ActionSet, rules: RuleSet[T], /
) -> tuple[ActionSet, tuple[_ActionPlan, ...]]:
    """Recover the base action axis and product plan from a soft action set."""
    roles = composition_roles(rules)
    base_actions: list[Action] = []
    base_columns: dict[tuple[str, str | None], int] = {}
    plans: list[_ActionPlan] = []

    for action in actions.actions:
        marked = raising_configuration(action.name)

        if marked is None:
            base_name = action.name
            configuration: RaiseConfiguration = "none"
        else:
            base_name, configuration = marked

        key = (base_name, action.rule)

        if key not in base_columns:
            base_columns[key] = len(base_actions)
            base_actions.append(Action(base_name, action.rule))

        role = roles.get(action.rule)

        if marked is not None and role is None:
            msg = f"action {action.name!r} raises inputs of a non-composition rule"
            raise ValueError(msg)

        principal, direction = (0, "undirected") if role is None else role
        plans.append(
            _ActionPlan(base_columns[key], configuration, principal, direction)
        )

    identity_key = (actions.identity.name, actions.identity.rule)
    null_key = None if actions.null is None else (actions.null.name, actions.null.rule)
    identity = base_actions[base_columns[identity_key]]
    null = None if null_key is None else base_actions[base_columns[null_key]]

    return ActionSet(base_actions, identity, null=null), tuple(plans)


class SoftControllerRaising(nn.Module):
    """Mix declared-target raised inputs and score the product action axis."""

    _encoder: SyntheticTreeEncoder

    def __init__(
        self,
        actions: ActionSet,
        rules: RuleSet[Term],
        categories: Sequence[Term],
        *,
        codec: TermCodec,
        encoder: TermEncoder,
    ) -> None:
        super().__init__()

        if not isinstance(encoder, SyntheticTreeEncoder):
            msg = (
                "soft controller raising requires a synthetic-tree continuation "
                "term encoder"
            )
            raise TypeError(msg)

        encoder.validate_synthetic_continuation()

        base, plans = _plans(actions, rules)
        raising = _raising_rules(rules)
        rules_by_direction = _rules_by_direction(raising, plans)
        targets, columns = _target_columns(rules_by_direction, categories)
        width = encoder.output_width
        self._functor_targets = nn.Linear(width, len(targets))
        self._argument_targets = nn.Linear(width, len(targets))
        self._actions = actions
        self._base_actions = base
        self._plans = plans
        self._targets = targets
        self._target_columns = columns
        self._rules_by_direction = rules_by_direction
        self._codec = codec
        object.__setattr__(self, "_encoder", encoder)

    @property
    def base_actions(self) -> ActionSet:
        """Return the non-product action axis scored by the controller head."""
        return self._base_actions

    @property
    def action_set(self) -> ActionSet:
        """Return the product action axis this module scores."""
        return self._actions

    def _rule(self, direction: _NormalFormDirection, /) -> _RaisingRule:
        """Return the raising construction serving a composition direction."""
        return self._rules_by_direction[direction]

    def _mixture(
        self,
        cells: Encoded,
        /,
        *,
        role: Literal["functor", "argument"],
        direction: _NormalFormDirection,
    ) -> Encoded:
        """Return a mixture over this direction's declared raising targets."""
        rule = self._rule(direction)
        head = self._functor_targets if role == "functor" else self._argument_targets
        columns = self._target_columns[direction]
        return mix_raised(
            rule,
            tuple(self._targets[column] for column in columns),
            cells,
            head(cells.tensor)[:, columns].softmax(dim=-1),
            codec=self._codec,
            encoder=self._encoder,
        )

    def scores(
        self,
        head: _PairScorer,
        left: Encoded,
        right: Encoded,
        /,
    ) -> Tensor:
        """Return product-action scores from a base controller head."""
        cache: dict[
            tuple[Literal["functor", "argument"], _NormalFormDirection, int], Encoded
        ] = {}
        output: list[Tensor] = []

        for plan in self._plans:
            pair = [left, right]

            for role in ("functor", "argument"):
                if plan.configuration not in (role, "both"):
                    continue

                slot = plan.principal if role == "functor" else 1 - plan.principal
                key = (role, plan.direction, slot)

                if key not in cache:
                    cache[key] = self._mixture(
                        pair[slot], role=role, direction=plan.direction
                    )

                pair[slot] = cache[key]

            output.append(head(pair[0].tensor, pair[1].tensor)[:, plan.base])

        return torch.stack(output, dim=-1)


class _PairScorer(Protocol):
    """Score a pair on a base action axis."""

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        """Return base-action scores."""


def build_soft_controller_raising(
    mode: ControllerRaising,
    input_source: ControllerInput,
    context: BuildContext,
    encoder: TermEncoder,
    /,
) -> SoftControllerRaising | None:
    """Build the one shared soft-raising implementation when declared."""
    if mode == "none":
        return None

    if input_source != "inside":
        msg = "soft controller raising requires input 'inside', not 'span'"
        raise ValueError(msg)

    return SoftControllerRaising(
        context.actions,
        context.rules,
        context.categories,
        codec=cast("TermCodec", context.codec.codec),
        encoder=encoder,
    )


def reencode_unary_domain(
    rule: Rule[Term],
    terms: Sequence[Term],
    /,
    *,
    codec: TermCodec,
    encoder: TermEncoder,
) -> UnaryReencoding:
    """Encode every conclusion of inputs in a unary rule's domain.

    A complete unary rule may still be partial: an input outside its domain has no
    conclusion. It may also be multivalued. The returned row indices retain the
    correspondence between every encoded conclusion and its source input. A
    conclusion beyond the finite codec's capacity is outside the encoded domain.

    Parameters
    ----------
    rule : Rule[Term]
        A unary rule with complete forward enumeration.
    terms : Sequence[Term]
        Premises in batch order.
    codec : TermCodec
        The grammar codec that linearizes the conclusions.
    encoder : TermEncoder
        The encoder that represents the conclusions.

    Returns
    -------
    UnaryReencoding
        One vector per conclusion beside its source row, which may repeat.

    Raises
    ------
    ValueError
        If the rule is not unary or does not enumerate forward completely.
    """
    if rule.arity != 1:
        msg = f"{rule.name!r} has arity {rule.arity}, not unary arity 1"
        raise ValueError(msg)

    if "forward" not in rule.directions or not rule.forward_is_complete:
        msg = f"{rule.name!r} does not completely enumerate its unary conclusions"
        raise ValueError(msg)

    conclusions = []
    source_rows: list[int] = []

    for row, term in enumerate(terms):
        for conclusion in rule.apply((term,)):
            try:
                encoded = codec.encode(conclusion)
            except EncodingOverflowError:
                continue

            conclusions.append(encoded)
            source_rows.append(row)

    return UnaryReencoding(
        encoder.encode(BatchEncoding(tuple(conclusions))), tuple(source_rows)
    )


def reencode_unary(
    rule: Rule[Term],
    terms: Sequence[Term],
    /,
    *,
    codec: TermCodec,
    encoder: TermEncoder,
) -> Encoded:
    """Apply a total single-valued unary rule and encode its conclusions.

    This strict route serves teacher-forced scoring, where every supplied premise
    must license the declared action. Use :func:`reencode_unary_domain` when a
    decoded frontier may also contain inputs outside a rule's domain.

    Raises
    ------
    ValueError
        If the rule is not unary, is not complete and single-valued, or is undefined
        for any supplied input.
    EncodingOverflowError
        If a conclusion lies beyond the finite codec's capacity.
    """
    if rule.arity != 1:
        msg = f"{rule.name!r} has arity {rule.arity}, not unary arity 1"
        raise ValueError(msg)

    if "forward" not in rule.directions or not rule.forward_is_complete:
        msg = f"{rule.name!r} does not completely enumerate its unary conclusions"
        raise ValueError(msg)

    conclusions: list[Term] = []

    for row, term in enumerate(terms):
        produced = tuple(rule.apply((term,)))

        if len(produced) != 1:
            msg = (
                f"{rule.name!r} produced {len(produced)} conclusions "
                f"for input row {row}, not one"
            )
            raise ValueError(msg)

        conclusions.append(produced[0])

    encoded = BatchEncoding(tuple(codec.encode(term) for term in conclusions))

    return encoder.encode(encoded)
