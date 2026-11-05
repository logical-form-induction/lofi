"""Soft typing objectives generated from declared typing rules."""

from typing import TYPE_CHECKING, NamedTuple, Self, cast, override

import torch

from lofi._checks import CHECKS, ensure_non_negative
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._ports import LossTerm
from lofi.terms import positions
from lofi.typecheck import compile_soft_typing

if TYPE_CHECKING:
    from collections.abc import Callable, Mapping, Sequence

    from torch import Tensor

    from lofi.deduce import Cell
    from lofi.nn.loss._ports import PublishedOutputs, SoftTypingItems
    from lofi.terms import OpName, Path, Term
    from lofi.typecheck import (
        SoftTypingAddress,
        SoftTypingNodeTest,
        SoftTypingPlan,
        SoftTypingProjection,
        Substitution,
        TypingRule,
    )

__all__ = [
    "SoftTypingObjective",
    "TensorNodeMarginals",
    "TensorSoftTypingDistributions",
    "TensorSoftTypingInstance",
]


type TensorNodeMarginals = Mapping[Path, Mapping[Term, Tensor]]
"""Tensor-valued type marginals keyed by paths within one rule occurrence."""

type _Categorical[K] = Mapping[K, Tensor]


class TensorSoftTypingDistributions(NamedTuple):
    """Tensor marginals for one application of a typing rule.

    Attributes
    ----------
    binds : tuple[TensorNodeMarginals | None, ...]
        Marginals for types introduced under the premises.
    premises : tuple[TensorNodeMarginals, ...]
        Marginals for premise types.
    result : TensorNodeMarginals
        Marginals for the result type.
    """

    binds: tuple[TensorNodeMarginals | None, ...]
    premises: tuple[TensorNodeMarginals, ...]
    result: TensorNodeMarginals


class TensorSoftTypingInstance(NamedTuple):
    """One tensor-valued application of a typing rule.

    Attributes
    ----------
    span : Cell
        The half-open span concluded by the rule application.
    split : int
        The split point used by the application.
    op : OpName
        The logical-form operator whose typing rule applies.
    distributions : TensorSoftTypingDistributions
        Marginals for the rule's bindings, premises, and result.
    constituent_marginal : Tensor
        Posterior probability that the span is a constituent.
    split_probability : Tensor
        Conditional probability of the split given the constituent.
    principal : Term | None
        Principal type of the logical form. May be ``None`` only for objectives
        without a subsumption matcher.
    erased : Term | None
        Erased category type. May be ``None`` only for objectives without a
        subsumption matcher.
    """

    span: Cell
    split: int
    op: OpName
    distributions: TensorSoftTypingDistributions
    constituent_marginal: Tensor
    split_probability: Tensor
    principal: Term | None = None
    erased: Term | None = None


class _NodeMask(NamedTuple):
    """A result-node mask derived from one subsumption match."""

    masked_roots: tuple[Path, ...]
    kept: int
    total: int

    def keeps(self, path: Path, /) -> bool:
        """Return whether a result path is outside every masked subtree."""
        return not any(path[: len(root)] == root for root in self.masked_roots)


def _metavariable_atom(term: Term) -> int | None:
    """Return a type metavariable's atom index, if present.

    Raises
    ------
    ValueError
        If the canonical free-variable operator is malformed.
    """
    if not term.op.is_free_variable:
        return None

    payload = term.payload

    if payload is None:
        msg = f"malformed type metavariable: {term!r}"
        raise ValueError(msg)

    return ensure_non_negative("metavariable atom", payload)


def _node_mask(
    principal: Term,
    erased: Term,
    subsumes: Callable[[Term, Term], Substitution | None],
) -> _NodeMask:
    """Build the node mask induced by a one-way subsumption match.

    Parameters
    ----------
    principal : Term
        General type whose metavariables may bind.
    erased : Term
        Fixed category erasure.
    subsumes : Callable[[Term, Term], Substitution | None]
        One-way matcher.

    Raises
    ------
    ValueError
        If the principal type does not subsume the erased type.
    """
    substitution = subsumes(principal, erased)

    if substitution is None:
        msg = f"principal type {principal!r} does not subsume erased type {erased!r}"
        raise ValueError(msg)

    def substituted(node: Term) -> bool:
        """Return whether the match substituted this principal node."""
        atom = _metavariable_atom(node)

        return atom is not None and atom in substitution

    masked_roots = positions(principal, substituted)
    erased_paths = positions(erased, lambda _node: True)
    mask = _NodeMask(masked_roots, 0, len(erased_paths))

    return mask._replace(kept=sum(mask.keeps(path) for path in erased_paths))


def _zero(like: Tensor) -> Tensor:
    """Return a graph-connected scalar zero.

    Parameters
    ----------
    like : Tensor
        Tensor whose graph, device, and dtype the zero follows.
    """
    return like.unsqueeze(0)[:0].sum()


def _sum(values: Sequence[Tensor], like: Tensor) -> Tensor:
    """Return the scalar sum of values, or a connected zero when empty.

    Parameters
    ----------
    values : Sequence[Tensor]
        Scalars to add.
    like : Tensor
        Tensor whose graph, device, and dtype an empty sum follows.
    """
    if not values:
        return _zero(like)

    return torch.stack(tuple(values)).sum()


def _checked_probability(name: str, value: Tensor) -> Tensor:
    """Return one scalar probability.

    Parameters
    ----------
    name : str
        Field name used in error messages.
    value : Tensor
        Probability to check.

    Raises
    ------
    TypeError
        If checks are enabled and the value is not floating point.
    ValueError
        If checks are enabled and the value is not scalar, finite, or in the unit
        interval.
    """
    if not CHECKS:
        return value

    if not value.is_floating_point():
        msg = f"{name} is not floating point: {value.dtype}"
        raise TypeError(msg)

    if value.ndim != 0:
        msg = f"{name} is not scalar: {tuple(value.shape)}"
        raise ValueError(msg)

    detached = value.detach()

    if not bool(torch.isfinite(detached)) or not bool(
        (detached >= 0.0) & (detached <= 1.0)
    ):
        msg = f"{name} is outside [0, 1]: {float(detached)}"
        raise ValueError(msg)

    return value


def _checked_marginal(
    name: str,
    marginal: Mapping[Term, Tensor],
) -> Mapping[Term, Tensor]:
    """Return one categorical marginal.

    Parameters
    ----------
    name : str
        Marginal name used in error messages.
    marginal : Mapping[Term, Tensor]
        Type probabilities to check.

    Raises
    ------
    TypeError
        If checks are enabled and a probability is not floating point.
    ValueError
        If checks are enabled and the marginal is empty, a probability is not scalar,
        finite, or in the unit interval, or the probabilities do not sum to one.
    """
    if not CHECKS:
        return marginal

    if not marginal:
        msg = f"{name} is empty"
        raise ValueError(msg)

    values = [
        _checked_probability(f"{name}[{term!r}]", probability)
        for term, probability in marginal.items()
    ]
    total = torch.stack(values).sum().detach()

    if not bool(torch.isclose(total, total.new_ones(()), rtol=1e-7, atol=1e-7)):
        msg = f"{name} sums to {float(total)}, not one"
        raise ValueError(msg)

    return marginal


def _occurrence(
    distributions: TensorSoftTypingDistributions,
    address: SoftTypingAddress,
) -> TensorNodeMarginals:
    """Return the marginals at one compiled rule occurrence.

    Parameters
    ----------
    distributions : TensorSoftTypingDistributions
        Marginals for one rule application.
    address : SoftTypingAddress
        Compiled occurrence and path to read.

    Raises
    ------
    ValueError
        If a compiled binding occurrence has no marginals.
    """
    if address.role == "result":
        return distributions.result

    if address.role == "premise":
        return distributions.premises[address.index]

    marginal = distributions.binds[address.index]

    if marginal is None:
        msg = f"binding {address.index} supplies no marginals"
        raise ValueError(msg)

    return marginal


def _check_distributions(
    plan: SoftTypingPlan,
    distributions: TensorSoftTypingDistributions,
) -> None:
    """Check distribution arity against a compiled rule plan.

    Parameters
    ----------
    plan : SoftTypingPlan
        Compiled arity and required binding positions.
    distributions : TensorSoftTypingDistributions
        Marginals supplied for one rule application.

    Raises
    ------
    ValueError
        If tuple lengths do not match the plan or a required binding has no
        marginals.
    """
    if len(distributions.binds) != plan.binding_count:
        msg = (
            f"rule {plan.op!r} has {plan.binding_count} bindings but received "
            f"{len(distributions.binds)} binding distributions"
        )
        raise ValueError(msg)

    if len(distributions.premises) != plan.premise_count:
        msg = (
            f"rule {plan.op!r} has {plan.premise_count} premises but received "
            f"{len(distributions.premises)} premise distributions"
        )
        raise ValueError(msg)

    for position in plan.required_bindings:
        if distributions.binds[position] is None:
            msg = (
                f"rule {plan.op!r} received no binding distributions at "
                f"position {position}"
            )
            raise ValueError(msg)


def _at(
    distributions: TensorSoftTypingDistributions,
    address: SoftTypingAddress,
) -> Mapping[Term, Tensor]:
    """Return the marginal at one compiled address.

    Parameters
    ----------
    distributions : TensorSoftTypingDistributions
        Marginals for one rule application.
    address : SoftTypingAddress
        Occurrence and path to read.

    Raises
    ------
    TypeError
        If checks are enabled and a probability is not floating point.
    ValueError
        If the occurrence supplies no marginal at the path or, with checks enabled,
        the marginal is invalid.
    """
    occurrence = _occurrence(distributions, address)

    try:
        marginal = occurrence[address.path]

    except KeyError:
        msg = (
            f"{address.role} {address.index} supplies no marginal "
            f"at path {address.path!r}"
        )
        raise ValueError(msg) from None

    return _checked_marginal(
        f"{address.role} {address.index} marginal at {address.path!r}",
        marginal,
    )


def _matches(test: SoftTypingNodeTest, term: Term) -> bool:
    """Return whether a term root passes a compiled node test.

    Parameters
    ----------
    test : SoftTypingNodeTest
        Compiled root constraints.
    term : Term
        Candidate type rooted at the addressed path.
    """
    if len(term.args) != test.arity:
        return False

    bindings: dict[str, OpName | int] = {}

    if test.operator_variable is None:
        if term.op.name != test.operator:
            return False
    else:
        bindings[test.operator_variable] = term.op.name

    if test.payload_variable is None:
        return term.payload == test.payload

    if term.payload is None:
        return False

    held = bindings.get(test.payload_variable)

    return held is None or held == term.payload


def _project(
    projection: SoftTypingProjection,
    distributions: TensorSoftTypingDistributions,
) -> _Categorical[Term | OpName | int]:
    """Return one compiled term, operator, or payload projection.

    Parameters
    ----------
    projection : SoftTypingProjection
        The addressed marginal and projection to apply.
    distributions : TensorSoftTypingDistributions
        Marginals for one rule application.

    Raises
    ------
    ValueError
        If a matched node lacks the projected field.
    """
    marginal = _at(distributions, projection.address)
    test = projection.node

    if test is None:
        return cast("_Categorical[Term | OpName | int]", marginal)

    matched = tuple(
        (term, probability)
        for term, probability in marginal.items()
        if _matches(test, term)
    )
    like = next(iter(marginal.values()))
    total = _sum([probability for _, probability in matched], like)

    if not bool(total.detach() > 0.0):
        return {}

    projected: dict[Term | OpName | int, Tensor] = {}

    for term, probability in matched:
        value = projection.field.value(term)
        held = projected.get(value)
        share = probability / total
        projected[value] = share if held is None else held + share

    return projected


def _cross_entropy[K](
    source: Mapping[K, Tensor],
    target: Mapping[K, Tensor],
    like: Tensor,
) -> Tensor:
    """Return categorical cross-entropy from a source into a target.

    Parameters
    ----------
    source : Mapping[K, Tensor]
        Distribution weighting the loss.
    target : Mapping[K, Tensor]
        Distribution being evaluated.
    like : Tensor
        Tensor whose graph, device, and dtype an empty sum follows.
    """
    pieces = []

    for value, probability in source.items():
        if not bool(probability.detach() > 0.0):
            continue

        received = target.get(value)

        if received is None:
            return probability.new_full((), torch.inf)

        pieces.append(-probability * received.log())

    return _sum(pieces, like)


def _address_is_scored(address: SoftTypingAddress, mask: _NodeMask | None) -> bool:
    """Return whether a compiled address contributes under a result-node mask."""
    return mask is None or address.role != "result" or mask.keeps(address.path)


def _projection_zero(
    projection: SoftTypingProjection,
    distributions: TensorSoftTypingDistributions,
    like: Tensor,
) -> Tensor:
    """Connect a masked projection to a scalar zero."""
    marginal = _at(distributions, projection.address)

    return _sum([_zero(probability) for probability in marginal.values()], like)


def _masked_result_zero(
    distributions: TensorSoftTypingDistributions,
    mask: _NodeMask | None,
    like: Tensor,
) -> Tensor:
    """Connect every masked result marginal to a scalar zero."""
    if mask is None:
        return _zero(like)

    zeros = []

    for path, marginal in distributions.result.items():
        if mask.keeps(path):
            continue

        checked = _checked_marginal(f"result marginal at {path!r}", marginal)
        zeros.extend(_zero(probability) for probability in checked.values())

    return _sum(zeros, like)


def _local_loss(
    plan: SoftTypingPlan,
    distributions: TensorSoftTypingDistributions,
    like: Tensor,
    mask: _NodeMask | None = None,
) -> Tensor:
    """Return one compiled rule application's local loss.

    Parameters
    ----------
    plan : SoftTypingPlan
        Compiled fixed targets and directed comparisons.
    distributions : TensorSoftTypingDistributions
        Marginals for one rule application.
    like : Tensor
        Tensor whose graph, device, and dtype an empty sum follows.
    mask : _NodeMask | None
        Result-node mask, or ``None`` to score every compiled address.
    """
    _check_distributions(plan, distributions)
    fixed = []

    for target in plan.targets:
        if not _address_is_scored(target.address, mask):
            continue

        marginal = _at(distributions, target.address)
        mass = _sum(
            [
                probability
                for term, probability in marginal.items()
                if _matches(target.node, term)
            ],
            like,
        )
        fixed.append(-mass.log())

    compared = []

    for comparison in plan.comparisons:
        if not (
            _address_is_scored(comparison.source.address, mask)
            and _address_is_scored(comparison.target.address, mask)
        ):
            compared.append(
                _projection_zero(comparison.source, distributions, like)
                + _projection_zero(comparison.target, distributions, like)
            )
            continue

        compared.append(
            _cross_entropy(
                _project(comparison.source, distributions),
                _project(comparison.target, distributions),
                like,
            )
        )

    return _sum(
        (*fixed, *compared, _masked_result_zero(distributions, mask, like)), like
    )


class SoftTypingObjective:
    """Scores one declared typing rule as a soft constraint.

    Parameters
    ----------
    rule : TypingRule
        The declaration compiled into fixed targets and directed comparisons.
    subsumes : Callable[[Term, Term], Substitution | None] | None
        One-way matcher. When provided, every instance must supply principal and
        erased types.

    Raises
    ------
    TypeError
        If the rule contains a spine pattern.
    """

    __slots__ = ("_plan", "_rule", "_subsumes")

    def __init__(
        self,
        rule: TypingRule,
        *,
        subsumes: Callable[[Term, Term], Substitution | None] | None = None,
    ) -> None:
        self._rule = rule
        self._plan = compile_soft_typing(rule)
        self._subsumes = subsumes

    @classmethod
    def from_rules(
        cls,
        rules: Mapping[OpName, TypingRule],
        /,
        *,
        subsumes: Callable[[Term, Term], Substitution | None] | None = None,
    ) -> tuple[Self, ...]:
        """Return one objective per declared typing rule.

        Parameters
        ----------
        rules : Mapping[OpName, TypingRule]
            Typing rules keyed by logical-form operator.
        subsumes : Callable[[Term, Term], Substitution | None] | None
            One-way matcher. When provided, every instance must supply principal and
            erased types.

        Raises
        ------
        TypeError
            If a rule contains a spine pattern.
        ValueError
            If a mapping key disagrees with its rule's operator.
        """
        objectives = []

        for op, rule in rules.items():
            if op != rule.op:
                msg = f"typing rule key {op!r} disagrees with operator {rule.op!r}"
                raise ValueError(msg)

            objectives.append(cls(rule, subsumes=subsumes))

        return tuple(objectives)

    @property
    def name(self) -> str:
        """Return the typing rule's operator name."""
        return self._rule.op

    @property
    def op(self) -> OpName:
        """Return the logical-form operator this objective scores."""
        return self._rule.op

    @property
    def rule(self) -> TypingRule:
        """Return the typing-rule declaration this objective follows."""
        return self._rule

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"batch", "cells"})

    def evaluate(self, outputs: PublishedOutputs[SoftTypingItems], /) -> LossTerm:
        """Return the marginal-weighted sum for this rule's applications.

        Parameters
        ----------
        outputs : PublishedOutputs[SoftTypingItems]
            What the graph published for the step.

        Raises
        ------
        MissingOutputError
            If the step publishes no cell tensor to anchor an inapplicable term.
        TypeError
            If a split is not an integer or, with checks enabled, a probability is not
            floating point.
        ValueError
            If a split is negative, distribution arity is wrong, a required
            occurrence or path lacks marginals, or checks are enabled and a
            probability or marginal is invalid. Also raised when coherence types are
            missing or incomplete for the configured matcher, types are supplied
            without a matcher, or a principal type does not subsume its erased type.
        """
        cells = outputs.cells

        if cells is None:
            raise MissingOutputError(
                objective=self.name,
                missing=frozenset({"cells"}),
                published=frozenset({"batch"}),
            )

        like = cells.tensor
        instances = tuple(
            instance
            for instance in outputs.batch.typing_instances
            if instance.op == self._rule.op
        )

        if not instances:
            return LossTerm.inapplicable(like)

        losses = []
        kept_nodes = 0
        total_nodes = 0

        for position, instance in enumerate(instances):
            ensure_non_negative("a soft-typing split", instance.split)
            constituent = _checked_probability(
                f"{self.name!r} instance {position} constituent marginal",
                instance.constituent_marginal,
            )
            split = _checked_probability(
                f"{self.name!r} instance {position} split probability",
                instance.split_probability,
            )
            weight = constituent * split
            principal = instance.principal
            erased = instance.erased

            if (principal is None) != (erased is None):
                msg = (
                    "a soft-typing instance must provide both principal and erased "
                    "types"
                )
                raise ValueError(msg)

            if principal is None and self._subsumes is not None:
                msg = (
                    "a soft-typing objective with a matcher requires coherence "
                    "types on every instance"
                )
                raise ValueError(msg)

            if principal is None:
                mask = None
            else:
                if self._subsumes is None:
                    msg = "a soft-typing instance has coherence types but no matcher"
                    raise ValueError(msg)

                held_erased = cast("Term", erased)
                mask = _node_mask(principal, held_erased, self._subsumes)
                kept_nodes += mask.kept
                total_nodes += mask.total

            loss = _local_loss(self._plan, instance.distributions, weight, mask)

            if not bool(weight.detach() > 0.0):
                losses.append(_zero(weight) + _zero(loss))
                continue

            losses.append(weight * loss)

        density = 1.0 if total_nodes == 0 else kept_nodes / total_nodes

        return LossTerm(
            _sum(losses, like),
            len(instances),
            diagnostics={"mask_density": like.new_tensor(density)},
        )

    @override
    def __repr__(self) -> str:
        """Return a representation naming the typing-rule operator."""
        return f"SoftTypingObjective(op={self._rule.op!r})"
