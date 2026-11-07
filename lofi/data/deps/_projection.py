"""Formalism-independent dependency projection."""

from typing import (
    TYPE_CHECKING,
    Never,
    Protocol,
    Self,
    final,
    override,
    runtime_checkable,
)

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Mapping, Sequence

    from lofi._types import HeadSelection
    from lofi.data._derivation import Derivation, DerivationNode, NodeId
    from lofi.data._ports import StepRecovery
    from lofi.data._values import Token

__all__ = [
    "AmbiguousProjectionStepError",
    "DependencyProjector",
    "HeadProjection",
    "UnrecoverableProjectionStepError",
    "project_dependencies",
]


class UnrecoverableProjectionStepError(ValueError):
    """Error raised when an unnamed step has no usable rule."""


class AmbiguousProjectionStepError(ValueError):
    """Error raised when an unnamed step has several usable rules."""


@final
class HeadProjection[D = Never, S = Never]:
    """Heads, dependencies, and formalism state for one subtree."""

    __slots__ = ("_dependencies", "_heads", "_state")

    def __init__(
        self,
        heads: Iterable[int],
        dependencies: Iterable[D] = (),
        state: S | None = None,
    ) -> None:
        held_heads = tuple(heads)
        held_dependencies = tuple(dependencies)

        for position, head in enumerate(held_heads):
            if isinstance(head, bool) or not isinstance(head, int):
                msg = f"head {position} is not an integer: {head!r}"
                raise TypeError(msg)

            if head < 0:
                msg = f"head {position} is negative: {head}"
                raise ValueError(msg)

        object.__setattr__(self, "_heads", held_heads)
        object.__setattr__(self, "_dependencies", held_dependencies)
        object.__setattr__(self, "_state", state)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a head projection is immutable"

        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a head projection is immutable"

        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable projection."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable projection."""
        del memo
        return self

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[HeadProjection],
        tuple[tuple[int, ...], tuple[D, ...], S | None],
    ]:
        """Return the projection reconstruction call."""
        return HeadProjection, self._fields()

    @property
    def heads(self) -> tuple[int, ...]:
        """Return the projected lexical-head positions."""
        return self._heads

    @property
    def dependencies(self) -> tuple[D, ...]:
        """Return the projected dependencies."""
        return self._dependencies

    @property
    def state(self) -> S | None:
        """Return the formalism-specific projection state, if present."""
        return self._state

    def _fields(self) -> tuple[tuple[int, ...], tuple[D, ...], S | None]:
        """Return fields in declaration order."""
        return self._heads, self._dependencies, self._state

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another projection has the same fields."""
        if type(other) is not HeadProjection:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the projection fields."""
        return hash((HeadProjection, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"HeadProjection{self._fields()!r}"


@runtime_checkable
class DependencyProjector[L: Hashable, D, S](Protocol):
    """A dependency convention applied by the derivation fold."""

    def lexical(
        self,
        node: DerivationNode[L],
        token: Token,
        /,
    ) -> HeadProjection[D, S]:
        """Project a lexical node."""
        ...

    def step(
        self,
        node: DerivationNode[L],
        rule_name: str,
        children: tuple[HeadProjection[D, S], ...],
        selection: HeadSelection,
        /,
    ) -> HeadProjection[D, S]:
        """Project a rule step over its projected children."""
        ...


def _head_child[L: Hashable](node: DerivationNode[L]) -> int:
    """Read an unnamed AUTO node's declared head child."""
    try:
        stated = node.features["head"]
    except KeyError:
        msg = "an unnamed nonlexical node states no AUTO head child"
        raise ValueError(msg) from None

    try:
        head = int(stated)
    except ValueError as error:
        msg = f"an unnamed nonlexical node states a noninteger AUTO head: {stated!r}"
        raise ValueError(msg) from error

    if head < 0 or head >= len(node.children):
        msg = (
            f"an unnamed node of arity {len(node.children)} states AUTO head child "
            f"{head}"
        )
        raise ValueError(msg)

    return head


def _recovered_name[L: Hashable](
    node: DerivationNode[L],
    children: Sequence[DerivationNode[L]],
    head_selections: Mapping[str, HeadSelection],
    recovery: StepRecovery[L, str] | None,
) -> str:
    """Recover one unnamed step and check its declared head child."""
    if recovery is None:
        msg = "an unnamed nonlexical node has no step recovery"
        raise UnrecoverableProjectionStepError(msg)

    head = _head_child(node)
    candidates = tuple(
        dict.fromkeys(
            recovery.recover(tuple(child.label for child in children), node.label)
        )
    )
    surviving = tuple(
        name
        for name in candidates
        if name in head_selections and head_selections[name].slot == head
    )

    if not surviving:
        msg = (
            "an unnamed nonlexical node has no recovered rule with AUTO head "
            f"child {head}"
        )
        raise UnrecoverableProjectionStepError(msg)

    if len(surviving) > 1:
        msg = (
            "an unnamed nonlexical node has ambiguous recovered rules with AUTO "
            f"head child {head}: {surviving!r}"
        )
        raise AmbiguousProjectionStepError(msg)

    return surviving[0]


def _selection_for(
    rule_name: str,
    child_count: int,
    head_selections: Mapping[str, HeadSelection],
) -> HeadSelection:
    """Return and validate the selection declared for one step."""
    try:
        selection = head_selections[rule_name]
    except KeyError:
        msg = f"derivation rule {rule_name!r} has no head selection"
        raise KeyError(msg) from None

    if child_count == 0:
        if selection.slot is not None:
            msg = f"nullary rule {rule_name!r} has head slot {selection.slot}"
            raise ValueError(msg)

        return selection

    if selection.slot is None:
        msg = f"positive-arity rule {rule_name!r} has no head slot"
        raise ValueError(msg)

    if selection.slot >= child_count:
        msg = (
            f"rule {rule_name!r} has head slot {selection.slot} "
            f"outside arity {child_count}"
        )
        raise ValueError(msg)

    return selection


def project_dependencies[L: Hashable, D, S](
    derivation: Derivation[L],
    tokens: Sequence[Token],
    head_selections: Mapping[str, HeadSelection],
    projector: DependencyProjector[L, D, S],
    /,
    *,
    recovery: StepRecovery[L, str] | None = None,
) -> HeadProjection[D, S]:
    """Project a derivation with tokens and declared head selections.

    Raises
    ------
    AmbiguousProjectionStepError
        If recovery finds multiple rules with the declared head child.
    UnrecoverableProjectionStepError
        If an unnamed step cannot be recovered uniquely.
    KeyError
        If a lexical node has no token or a rule has no head selection.
    ValueError
        If tokens repeat a position, a lexical node spans multiple tokens, or a head
        declaration is invalid.
    """
    by_position = {token.index: token for token in tokens}

    if len(by_position) != len(tokens):
        msg = "projection tokens repeat a position"
        raise ValueError(msg)

    def visit(node_id: NodeId) -> HeadProjection[D, S]:
        node = derivation.node(node_id)

        if not node.children and node.rule is None:
            if node.span.length != 1:
                msg = "a lexical derivation node must cover exactly one token"
                raise ValueError(msg)

            try:
                token = by_position[node.span.start]
            except KeyError:
                msg = f"a lexical node at position {node.span.start} has no token"
                raise KeyError(msg) from None

            projection = projector.lexical(node, token)
        else:
            child_nodes = tuple(derivation.node(child) for child in node.children)
            rule_name = node.rule

            if rule_name is None:
                rule_name = _recovered_name(
                    node,
                    child_nodes,
                    head_selections,
                    recovery,
                )

            below = tuple(visit(child) for child in node.children)
            selection = _selection_for(rule_name, len(below), head_selections)
            projection = projector.step(node, rule_name, below, selection)

        return projection

    return visit(derivation.root)
