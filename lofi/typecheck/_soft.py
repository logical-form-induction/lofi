"""Symbolic reference for marginal-weighted soft typing."""

from itertools import combinations
from math import fsum, inf, isclose, log
from typing import TYPE_CHECKING, Final, Literal, NamedTuple, Protocol, final

from lofi._checks import ensure_fraction, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.terms import MVar, Node, PVar, Spine

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi.deduce import Cell
    from lofi.terms import OpName, Path, Pattern, Term
    from lofi.typecheck._values import TypingRule

__all__ = [
    "OPERATOR_FIELD",
    "PAYLOAD_FIELD",
    "WHOLE_TERM",
    "NodeMarginals",
    "SoftTypingAddress",
    "SoftTypingComparison",
    "SoftTypingDistributions",
    "SoftTypingField",
    "SoftTypingInstance",
    "SoftTypingNodeTest",
    "SoftTypingPlan",
    "SoftTypingProjection",
    "SoftTypingReport",
    "SoftTypingTarget",
    "SoftTypingTerm",
    "compile_soft_typing",
    "soft_typing_reference",
]


type NodeMarginals = Mapping[Path, Mapping[Term, float]]
"""Type marginals keyed by paths within one typing-rule occurrence."""

type _Role = Literal["bind", "premise", "result"]


class SoftTypingAddress(NamedTuple):
    """Location of a pattern node within one rule application.

    Attributes
    ----------
    role : _Role
        Rule occurrence containing the node.
    index : int
        Binding or premise position, or zero for the result.
    path : Path
        Path to the node within that occurrence.
    """

    role: _Role
    index: int
    path: Path


class SoftTypingNodeTest(NamedTuple):
    """Compiled root test for a node pattern.

    Attributes
    ----------
    arity : int
        Required child count.
    operator : OpName | None
        Required operator, or ``None`` for an operator variable.
    operator_variable : str | None
        Operator-variable name, if present.
    payload : int | None
        Required payload, or ``None`` when no payload is fixed.
    payload_variable : str | None
        Payload-variable name, if present.
    """

    arity: int
    operator: OpName | None
    operator_variable: str | None
    payload: int | None
    payload_variable: str | None


class SoftTypingField(Protocol):
    """Read the projected value of one matched node."""

    @property
    def name(self) -> str:
        """Return the field's name in refusals."""
        ...

    def value(self, term: Term, /) -> Term | OpName | int:
        """Return the field of one matched node.

        Raises
        ------
        ValueError
            If the node carries no such field.
        """
        ...


@final
class _WholeTerm(FrozenValue):
    """Read the matched node itself."""

    _noun = "whole-term field"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "term"

    def value(self, term: Term, /) -> Term | OpName | int:
        return term


@final
class _OperatorField(FrozenValue):
    """Read the matched node's operator name."""

    _noun = "operator field"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "operator"

    def value(self, term: Term, /) -> Term | OpName | int:
        return term.op.name


@final
class _PayloadField(FrozenValue):
    """Read the matched node's payload."""

    _noun = "payload field"
    __slots__ = ()

    @property
    def name(self) -> str:
        return "payload"

    def value(self, term: Term, /) -> Term | OpName | int:
        payload = term.payload

        if payload is None:
            msg = "a payload-variable match has no payload"
            raise ValueError(msg)

        return payload


WHOLE_TERM: Final[SoftTypingField] = _WholeTerm()
"""The field a term variable projects: the whole node."""

OPERATOR_FIELD: Final[SoftTypingField] = _OperatorField()
"""The field an operator variable projects."""

PAYLOAD_FIELD: Final[SoftTypingField] = _PayloadField()
"""The field a payload variable projects."""


class SoftTypingProjection(NamedTuple):
    """One variable occurrence projected from a node marginal.

    Attributes
    ----------
    address : SoftTypingAddress
        Location of the variable.
    field : SoftTypingField
        Field projected from the marginal.
    node : SoftTypingNodeTest | None
        Enclosing node test whose matches the field is read from, or None when
        the variable stands for the whole node marginal.
    """

    address: SoftTypingAddress
    field: SoftTypingField
    node: SoftTypingNodeTest | None


class SoftTypingTarget(NamedTuple):
    """A fixed node pattern scored at one rule occurrence.

    Attributes
    ----------
    address : SoftTypingAddress
        Location of the node.
    node : SoftTypingNodeTest
        Test determining the accepted root mass.
    """

    address: SoftTypingAddress
    node: SoftTypingNodeTest


class SoftTypingComparison(NamedTuple):
    """Directed cross-entropy between two variable projections.

    Attributes
    ----------
    source : SoftTypingProjection
        Projection weighting the cross-entropy.
    target : SoftTypingProjection
        Projection being evaluated.
    """

    source: SoftTypingProjection
    target: SoftTypingProjection


class SoftTypingPlan(NamedTuple):
    """Backend-neutral traversal plan for one soft-typing rule.

    Attributes
    ----------
    op : OpName
        Logical-form operator naming the rule.
    binding_count : int
        Number of binding-distribution positions.
    premise_count : int
        Number of premise-distribution positions.
    required_bindings : tuple[int, ...]
        Active binding positions that must supply marginals.
    targets : tuple[SoftTypingTarget, ...]
        Fixed node patterns in rule traversal order.
    comparisons : tuple[SoftTypingComparison, ...]
        Directed variable comparisons in traversal order.
    """

    op: OpName
    binding_count: int
    premise_count: int
    required_bindings: tuple[int, ...]
    targets: tuple[SoftTypingTarget, ...]
    comparisons: tuple[SoftTypingComparison, ...]


class SoftTypingDistributions(NamedTuple):
    """Node marginals for one application of a typing rule.

    Attributes
    ----------
    binds : tuple[NodeMarginals | None, ...]
        Marginals for the types introduced under the premises.
    premises : tuple[NodeMarginals, ...]
        Marginals for the premise types.
    result : NodeMarginals
        Marginals for the result type.
    """

    binds: tuple[NodeMarginals | None, ...]
    premises: tuple[NodeMarginals, ...]
    result: NodeMarginals


class SoftTypingInstance(NamedTuple):
    """One split-local application of a soft typing rule.

    Attributes
    ----------
    span : Cell
        The half-open span concluded by the rule application.
    split : int
        The split point used by the application.
    op : OpName
        The logical-form operator whose typing rule applies.
    distributions : SoftTypingDistributions
        Node marginals for the rule's bindings, premises, and result.
    constituent_marginal : float
        Posterior probability that the span is a constituent.
    split_probability : float
        Conditional probability of the split given the constituent.
    """

    span: Cell
    split: int
    op: OpName
    distributions: SoftTypingDistributions
    constituent_marginal: float
    split_probability: float


class SoftTypingTerm(NamedTuple):
    """Loss accumulated for one declared typing rule.

    Attributes
    ----------
    op : OpName
        The logical-form operator naming the rule.
    unweighted : float
        Sum of the rule's local losses before chart weighting.
    weighted : float
        Sum after multiplication by rule marginals.
    instance_count : int
        Number of rule applications accumulated.
    """

    op: OpName
    unweighted: float
    weighted: float
    instance_count: int


class SoftTypingReport(NamedTuple):
    """Per-rule soft-typing terms and their sentence-level total.

    Attributes
    ----------
    terms : tuple[SoftTypingTerm, ...]
        One term per declared rule, in rule declaration order.
    total : float
        Sum of the marginal-weighted terms.
    """

    terms: tuple[SoftTypingTerm, ...]
    total: float


def _patterns(rule: TypingRule) -> tuple[tuple[_Role, int, Pattern], ...]:
    """List the rule occurrences included by its enforcement policy.

    Parameters
    ----------
    rule : TypingRule
        Rule whose active occurrences are listed.
    """
    enforced = range(len(rule.premises)) if rule.enforced is None else rule.enforced
    active: list[tuple[_Role, int, Pattern]] = []

    for position in enforced:
        bind = rule.binds[position]

        if bind is not None:
            active.append(("bind", position, bind))

        active.append(("premise", position, rule.premises[position]))

    active.append(("result", 0, rule.result))

    return tuple(active)


def _node_test(pattern: Node) -> SoftTypingNodeTest:
    """Compile a node pattern into a root test.

    Parameters
    ----------
    pattern : Node
        Node pattern to compile.
    """
    operator = pattern.op
    payload = pattern.payload

    return SoftTypingNodeTest(
        len(pattern.args),
        None if isinstance(operator, PVar) else operator,
        operator.name if isinstance(operator, PVar) else None,
        None if isinstance(payload, PVar) else payload,
        payload.name if isinstance(payload, PVar) else None,
    )


def compile_soft_typing(rule: TypingRule, /) -> SoftTypingPlan:
    """Compile rule traversal and directed comparisons.

    Parameters
    ----------
    rule : TypingRule
        Rule to compile.

    Returns
    -------
    SoftTypingPlan
        Fixed targets and directed variable comparisons.

    Raises
    ------
    TypeError
        If the rule contains a spine pattern.
    """
    targets: list[SoftTypingTarget] = []
    variables: dict[
        tuple[Literal["term", "pattern"], str],
        list[SoftTypingProjection],
    ] = {}

    def visit(role: _Role, index: int, pattern: Pattern, path: Path) -> None:
        address = SoftTypingAddress(role, index, path)

        if isinstance(pattern, MVar):
            variables.setdefault(("term", pattern.name), []).append(
                SoftTypingProjection(address, WHOLE_TERM, None)
            )
            return

        if isinstance(pattern, Spine):
            msg = "soft typing does not admit spine patterns"
            raise TypeError(msg)

        node = _node_test(pattern)
        targets.append(SoftTypingTarget(address, node))

        if isinstance(pattern.op, PVar):
            variables.setdefault(("pattern", pattern.op.name), []).append(
                SoftTypingProjection(address, OPERATOR_FIELD, node)
            )

        if isinstance(pattern.payload, PVar):
            variables.setdefault(("pattern", pattern.payload.name), []).append(
                SoftTypingProjection(address, PAYLOAD_FIELD, node)
            )

        for position, child in enumerate(pattern.args):
            visit(role, index, child, (*path, position))

    for role, index, pattern in _patterns(rule):
        visit(role, index, pattern, ())

    comparisons: list[SoftTypingComparison] = []

    for projections in variables.values():
        for lower, upper in combinations(projections, 2):
            if rule.direction in ("up", "both"):
                comparisons.append(SoftTypingComparison(lower, upper))

            if rule.direction in ("down", "both"):
                comparisons.append(SoftTypingComparison(upper, lower))

    required_bindings = tuple(
        index for role, index, _ in _patterns(rule) if role == "bind"
    )

    return SoftTypingPlan(
        rule.op,
        len(rule.binds),
        len(rule.premises),
        required_bindings,
        tuple(targets),
        tuple(comparisons),
    )


def _marginal(
    name: str,
    marginal: Mapping[Term, float],
) -> Mapping[Term, float]:
    """Validate and return one categorical marginal.

    Parameters
    ----------
    name : str
        Marginal name used in error messages.
    marginal : Mapping[Term, float]
        Type probabilities to validate.

    Raises
    ------
    TypeError
        If a probability is not numeric or is a boolean.
    ValueError
        If the marginal is empty, a probability is invalid, or the probabilities
        do not sum to one.
    """
    if not marginal:
        msg = f"{name} is empty"
        raise ValueError(msg)

    total = fsum(
        ensure_fraction(f"{name}[{term!r}]", probability)
        for term, probability in marginal.items()
    )

    if not isclose(total, 1.0, rel_tol=1e-9, abs_tol=1e-9):
        msg = f"{name} sums to {total!r}, not one"
        raise ValueError(msg)

    return marginal


def _at(
    distributions: SoftTypingDistributions,
    address: SoftTypingAddress,
) -> Mapping[Term, float]:
    """Return the validated marginal at one pattern path.

    Parameters
    ----------
    distributions : SoftTypingDistributions
        Marginals supplied for one rule application.
    address : SoftTypingAddress
        Location within the rule application.

    Raises
    ------
    TypeError
        If a probability is not numeric or is a boolean.
    ValueError
        If the occurrence supplies no marginal at the path or the marginal is
        invalid.
    """
    match address.role:
        case "bind":
            occurrence = distributions.binds[address.index]

            if occurrence is None:
                msg = (
                    "rule received no binding distributions at position "
                    f"{address.index}"
                )
                raise ValueError(msg)
        case "premise":
            occurrence = distributions.premises[address.index]
        case "result":
            occurrence = distributions.result

    try:
        marginal = occurrence[address.path]

    except KeyError:
        msg = (
            f"{address.role} {address.index} supplies no marginal "
            f"at path {address.path!r}"
        )
        raise ValueError(msg) from None

    return _marginal(
        f"{address.role} {address.index} marginal at {address.path!r}",
        marginal,
    )


def _matches(node: SoftTypingNodeTest, term: Term) -> bool:
    """Return whether a term root passes a compiled node test.

    Parameters
    ----------
    node : SoftTypingNodeTest
        Compiled root test.
    term : Term
        Candidate type rooted at the tested path.
    """
    if len(term.args) != node.arity:
        return False

    bindings: dict[str, OpName | int] = {}

    if node.operator_variable is not None:
        bindings[node.operator_variable] = term.op.name
    elif term.op.name != node.operator:
        return False

    if node.payload_variable is not None:
        if term.payload is None:
            return False

        held = bindings.get(node.payload_variable)

        if held is not None and held != term.payload:
            return False
    elif term.payload != node.payload:
        return False

    return True


def _project_pattern_variable(
    node: SoftTypingNodeTest,
    marginal: Mapping[Term, float],
    field: SoftTypingField,
) -> Mapping[Term | OpName | int, float]:
    """Project matching node mass onto one field of the matched nodes.

    Parameters
    ----------
    node : SoftTypingNodeTest
        Compiled node test containing the variable.
    marginal : Mapping[Term, float]
        Candidate type probabilities at the node.
    field : SoftTypingField
        Node field to project.

    Raises
    ------
    ValueError
        If a matched node lacks the field.
    """
    matched = tuple(
        (term, probability)
        for term, probability in marginal.items()
        if _matches(node, term)
    )
    total = fsum(probability for _, probability in matched)

    if total == 0.0:
        return {}

    projected: dict[Term | OpName | int, float] = {}

    for term, probability in matched:
        value = field.value(term)
        projected[value] = projected.get(value, 0.0) + probability / total

    return projected


def _target_loss(
    target: SoftTypingTarget,
    distributions: SoftTypingDistributions,
) -> float:
    """Return the fixed-node loss for one compiled target.

    Parameters
    ----------
    target : SoftTypingTarget
        Compiled node pattern.
    distributions : SoftTypingDistributions
        Marginals supplied for one rule application.
    """
    marginal = _at(distributions, target.address)
    mass = fsum(
        probability
        for term, probability in marginal.items()
        if _matches(target.node, term)
    )

    return inf if mass == 0.0 else -log(mass)


def _projection(
    projection: SoftTypingProjection,
    distributions: SoftTypingDistributions,
) -> tuple[tuple[Term | OpName | int, float], ...]:
    """Project one compiled variable occurrence.

    Parameters
    ----------
    projection : SoftTypingProjection
        Variable occurrence to project.
    distributions : SoftTypingDistributions
        Marginals supplied for one rule application.

    Raises
    ------
    ValueError
        If an operator or payload projection has no node test.
    """
    marginal = _at(distributions, projection.address)
    node = projection.node

    if node is None:
        return tuple(marginal.items())

    return tuple(_project_pattern_variable(node, marginal, projection.field).items())


def _cross_entropy[K](
    source: Sequence[tuple[K, float]],
    target: Sequence[tuple[K, float]],
) -> float:
    """Return categorical cross-entropy from a source into a target.

    Parameters
    ----------
    source : Sequence[tuple[K, float]]
        Distribution weighting the loss.
    target : Sequence[tuple[K, float]]
        Distribution being evaluated.
    """
    pieces: list[float] = []

    for term, probability in source:
        if probability == 0.0:
            continue

        received = next(
            (held for candidate, held in target if candidate == term),
            0.0,
        )

        if received == 0.0:
            return inf

        pieces.append(-probability * log(received))

    return fsum(pieces)


def _check_distributions(
    plan: SoftTypingPlan,
    distributions: SoftTypingDistributions,
) -> None:
    """Validate distribution arity and required bindings against a compiled plan.

    Parameters
    ----------
    plan : SoftTypingPlan
        Compiled arity and required binding positions.
    distributions : SoftTypingDistributions
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


def _rule_loss(
    plan: SoftTypingPlan,
    distributions: SoftTypingDistributions,
) -> float:
    """Return one rule application's masked soft-typing loss.

    Parameters
    ----------
    plan : SoftTypingPlan
        Compiled arity, masking, traversal, and directed comparisons.
    distributions : SoftTypingDistributions
        Marginals predicted at each rule occurrence.
    """
    _check_distributions(plan, distributions)

    return fsum(
        (
            *(_target_loss(target, distributions) for target in plan.targets),
            *(
                _cross_entropy(
                    _projection(comparison.source, distributions),
                    _projection(comparison.target, distributions),
                )
                for comparison in plan.comparisons
            ),
        )
    )


def soft_typing_reference(
    rules: Mapping[OpName, TypingRule],
    instances: Sequence[SoftTypingInstance],
    /,
) -> SoftTypingReport:
    """Compute the symbolic masked soft-typing loss.

    Each local loss is weighted by the rule marginal, expressed as the
    constituent marginal times the conditional split probability. The report
    contains one term for every declared rule, including rules unused by the
    supplied instances. Metavariable occurrences follow each enforced binding
    and premise in premise order, then the result. ``up`` sends marginals
    forward in that order, ``down`` sends them backward, and ``both`` scores
    both directions.

    Parameters
    ----------
    rules : Mapping[OpName, TypingRule]
        Typing rules keyed by logical-form operator.
    instances : Sequence[SoftTypingInstance]
        Split-local rule applications to score.

    Returns
    -------
    SoftTypingReport
        Per-rule terms and their sentence-level total.

    Raises
    ------
    KeyError
        If an instance names no declared rule.
    TypeError
        If a rule contains a spine pattern, a probability is not numeric, or a
        split is not an integer.
    ValueError
        If a rule key disagrees with its operator, a split is negative, a
        probability is invalid, a distribution has the wrong arity, or a
        required node marginal is absent.
    """
    grouped: dict[OpName, list[SoftTypingInstance]] = {op: [] for op in rules}
    plans: dict[OpName, SoftTypingPlan] = {}

    for op, rule in rules.items():
        if rule.op != op:
            msg = f"typing rule key {op!r} disagrees with operator {rule.op!r}"
            raise ValueError(msg)

        plans[op] = compile_soft_typing(rule)

    for instance in instances:
        ensure_non_negative("a soft-typing split", instance.split)
        grouped[instance.op].append(instance)

    terms: list[SoftTypingTerm] = []

    for op in rules:
        local: list[float] = []
        weighted: list[float] = []

        for position, instance in enumerate(grouped[op]):
            constituent = ensure_fraction(
                f"{op!r} instance {position} constituent marginal",
                instance.constituent_marginal,
            )
            split = ensure_fraction(
                f"{op!r} instance {position} split probability",
                instance.split_probability,
            )
            loss = _rule_loss(plans[op], instance.distributions)
            rule_marginal = constituent * split
            local.append(loss)
            weighted.append(0.0 if rule_marginal == 0.0 else rule_marginal * loss)

        terms.append(SoftTypingTerm(op, fsum(local), fsum(weighted), len(grouped[op])))

    return SoftTypingReport(tuple(terms), fsum(term.weighted for term in terms))
