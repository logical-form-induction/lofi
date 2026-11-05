"""Bisk-Hockenmaier dependency projection for induced CCGs."""

from typing import TYPE_CHECKING, Final, Never, Self, final, override

from lofi.data._derivation import Dependency
from lofi.data.deps._projection import HeadProjection, project_dependencies
from lofi.data.deps._templates import CCGDependencyConvention, CCGProjectionReport

if TYPE_CHECKING:
    from collections.abc import Hashable, Mapping, Sequence

    from lofi._types import HeadSelection
    from lofi.data._derivation import Derivation, DerivationNode
    from lofi.data._ports import StepRecovery
    from lofi.data._values import Token
    from lofi.data.deps._ccg import CCGCategoryAlgebra

__all__ = [
    "BISK_HOCKENMAIER_CONVENTION",
    "BiskHockenmaierDependencyProjector",
    "InducedDependencyState",
    "project_bisk_hockenmaier_dependencies",
]

BISK_HOCKENMAIER_CONVENTION: Final[CCGDependencyConvention[Never]] = (
    CCGDependencyConvention("bisk-hockenmaier-conll-2008-coverage-secondary")
)
"""The Bisk-Hockenmaier CoNLL-2008 dependency convention."""

_DEPENDENCY_LABEL = "dep"
_BINARY_ARITY = 2


@final
class InducedDependencyState[C: Hashable]:
    """Category and pending coordination state for an induced projection.

    Parameters
    ----------
    category : C
        Category produced at this node.
    coordinator_heads : tuple[int, ...]
        Zero-based coordination marker heads awaiting a merge step.
    report : CCGProjectionReport
        Convention report for the projection.

    Raises
    ------
    TypeError
        If a coordination marker head is not an integer.
    ValueError
        If a coordination marker head is negative.
    """

    __slots__ = ("_category", "_coordinator_heads", "_report")

    def __init__(
        self,
        category: C,
        coordinator_heads: tuple[int, ...],
        report: CCGProjectionReport,
    ) -> None:
        for position, head in enumerate(coordinator_heads):
            if isinstance(head, bool) or not isinstance(head, int):
                msg = f"coordination marker head {position} is not an integer: {head!r}"
                raise TypeError(msg)

            if head < 0:
                msg = f"coordination marker head {position} is negative: {head}"
                raise ValueError(msg)

        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_coordinator_heads", coordinator_heads)
        object.__setattr__(self, "_report", report)

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "an induced dependency state is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "an induced dependency state is immutable"
        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable state."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable state."""
        del memo
        return self

    @property
    def category(self) -> C:
        """Return the projected category."""
        return self._category

    @property
    def coordinator_heads(self) -> tuple[int, ...]:
        """Return coordination marker heads awaiting a merge step."""
        return self._coordinator_heads

    @property
    def report(self) -> CCGProjectionReport:
        """Return the projection's convention report."""
        return self._report

    def _fields(self) -> tuple[C, tuple[int, ...], CCGProjectionReport]:
        """Return fields in declaration order."""
        return self._category, self._coordinator_heads, self._report

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[InducedDependencyState[C]],
        tuple[C, tuple[int, ...], CCGProjectionReport],
    ]:
        """Return the reconstruction call."""
        return InducedDependencyState, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another state has the same fields."""
        if type(other) is not InducedDependencyState:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the state fields."""
        return hash((InducedDependencyState, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"InducedDependencyState{self._fields()!r}"


def _state[C: Hashable](
    projection: HeadProjection[Dependency, InducedDependencyState[C]],
) -> InducedDependencyState[C]:
    """Require an induced dependency state.

    Parameters
    ----------
    projection : HeadProjection[Dependency, InducedDependencyState[C]]
        Projection whose state is required.

    Raises
    ------
    ValueError
        If the projection has no state.
    """
    state = projection.state

    if state is None:
        msg = "an induced dependency projection has no state"
        raise ValueError(msg)

    return state


def _arc(head: int, dependent: int, label: str = _DEPENDENCY_LABEL) -> Dependency:
    """Build a one-based, nonreflexive dependency arc.

    Parameters
    ----------
    head : int
        Zero-based head position, or negative one for the artificial root.
    dependent : int
        Zero-based dependent position.
    label : str
        Dependency label.

    Raises
    ------
    ValueError
        If the converted arc is reflexive.
    """
    converted_head = head + 1
    converted_dependent = dependent + 1

    if converted_head == converted_dependent:
        msg = f"an induced dependency arc is reflexive at {converted_head}"
        raise ValueError(msg)

    return Dependency(converted_head, converted_dependent, label)


def _only_head(heads: tuple[int, ...], subject: str) -> int:
    """Return the sole head in a declared coordination position.

    Parameters
    ----------
    heads : tuple[int, ...]
        Heads found at the position.
    subject : str
        Position named in an error message.

    Raises
    ------
    ValueError
        If the position does not have exactly one head.
    """
    if len(heads) != 1:
        msg = f"{subject} has {len(heads)} heads rather than one"
        raise ValueError(msg)

    return heads[0]


@final
class BiskHockenmaierDependencyProjector[C: Hashable]:
    """A Bisk-Hockenmaier dependency projector for induced CCGs.

    It uses one-based token positions and artificial root position zero.

    Parameters
    ----------
    algebra : CCGCategoryAlgebra[C]
        Structural category operations used to identify head-sharing functors.
    coordination_marks : frozenset[str]
        Exact rule names for the coordination mark phase.

    Raises
    ------
    ValueError
        If a coordination mark name is empty.
    """

    __slots__ = ("_algebra", "_coordination_marks", "_report")

    def __init__(
        self,
        algebra: CCGCategoryAlgebra[C],
        coordination_marks: frozenset[str],
    ) -> None:
        if "" in coordination_marks:
            msg = "a coordination mark rule has no name"
            raise ValueError(msg)

        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_coordination_marks", coordination_marks)
        object.__setattr__(
            self,
            "_report",
            CCGProjectionReport(BISK_HOCKENMAIER_CONVENTION.name, 0, 0),
        )

    @override
    def __setattr__[V](self, name: str, value: V) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value
        msg = "a Bisk-Hockenmaier dependency projector is immutable"
        raise AttributeError(msg)

    @override
    def __delattr__(self, name: str) -> Never:
        """Raise AttributeError on field deletion."""
        del name
        msg = "a Bisk-Hockenmaier dependency projector is immutable"
        raise AttributeError(msg)

    def __copy__(self) -> Self:
        """Return this immutable projector."""
        return self

    def __deepcopy__(self, memo: dict[int, Self]) -> Self:
        """Return this immutable projector."""
        del memo
        return self

    @property
    def algebra(self) -> CCGCategoryAlgebra[C]:
        """Return the structural category algebra."""
        return self._algebra

    @property
    def convention(self) -> CCGDependencyConvention[Never]:
        """Return the Bisk-Hockenmaier dependency convention."""
        return BISK_HOCKENMAIER_CONVENTION

    @property
    def coordination_marks(self) -> frozenset[str]:
        """Return the exact coordination mark rule names."""
        return self._coordination_marks

    def report(
        self,
        projection: HeadProjection[Dependency, InducedDependencyState[C]],
        /,
    ) -> CCGProjectionReport:
        """Return the projection's convention report.

        Parameters
        ----------
        projection : HeadProjection[Dependency, InducedDependencyState[C]]
            Projection whose report is returned.

        Raises
        ------
        ValueError
            If the projection has no state.
        """
        return _state(projection).report

    def lexical(
        self,
        node: DerivationNode[C],
        token: Token,
        /,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Project one lexical node.

        Parameters
        ----------
        node : DerivationNode[C]
            Lexical derivation node.
        token : Token
            Token covered by the node.
        """
        return HeadProjection(
            (token.index,),
            (),
            InducedDependencyState(node.label, (), self._report),
        )

    def _merge(
        self,
        node: DerivationNode[C],
        children: tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...],
        states: tuple[InducedDependencyState[C], ...],
        selection: HeadSelection,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Project a declared coordination merge.

        Parameters
        ----------
        node : DerivationNode[C]
            Merge output node.
        children : tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...]
            Child projections in derivation order.
        states : tuple[InducedDependencyState[C], ...]
            State for each child.
        selection : HeadSelection
            Declared merge head selection.

        Raises
        ------
        ValueError
            If the step is not binary with one marked right child.
        """
        if len(children) != _BINARY_ARITY or selection.slot != 0:
            msg = "an induced coordination merge requires two children and slot 0"
            raise ValueError(msg)

        pending = tuple(
            position for position, state in enumerate(states) if state.coordinator_heads
        )

        if pending != (1,):
            msg = "an induced coordination merge requires one marked right child"
            raise ValueError(msg)

        left = _only_head(children[0].heads, "a left conjunct")
        marker = _only_head(
            states[1].coordinator_heads,
            "a marked coordination state",
        )
        right = _only_head(children[1].heads, "a right conjunct")
        dependencies = tuple(
            dependency for child in children for dependency in child.dependencies
        )
        return HeadProjection(
            children[0].heads,
            (*dependencies, _arc(left, marker), _arc(marker, right)),
            InducedDependencyState(node.label, (), self._report),
        )

    def _functor(
        self,
        node: DerivationNode[C],
        children: tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...],
        states: tuple[InducedDependencyState[C], ...],
        selection: HeadSelection,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Project a functor step.

        The dependency reverses when the functor shares its argument's head.

        Parameters
        ----------
        node : DerivationNode[C]
            Functor output node.
        children : tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...]
            Child projections in derivation order.
        states : tuple[InducedDependencyState[C], ...]
            State for each child.
        selection : HeadSelection
            Declared functor head selection.

        Raises
        ------
        ValueError
            If the step is not binary or contains an unmerged marker.
        """
        if len(children) != _BINARY_ARITY:
            msg = f"an induced functor step has arity {len(children)} rather than two"
            raise ValueError(msg)

        if any(state.coordinator_heads for state in states):
            msg = "a coordination marker must merge before a functor step"
            raise ValueError(msg)

        slot = selection.slot

        if slot is None:
            msg = "an induced functor step has no head slot"
            raise ValueError(msg)

        selected = children[slot]
        other = children[1 - slot]
        modifier = self._algebra.shares_head(states[slot].category)
        heads = other.heads if modifier else selected.heads
        governors = other.heads if modifier else selected.heads
        dependents = selected.heads if modifier else other.heads
        dependencies = tuple(
            dependency for child in children for dependency in child.dependencies
        )
        emitted = tuple(
            _arc(head, dependent) for head in governors for dependent in dependents
        )
        return HeadProjection(
            heads,
            (*dependencies, *emitted),
            InducedDependencyState(node.label, (), self._report),
        )

    def _mark(
        self,
        node: DerivationNode[C],
        children: tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...],
        states: tuple[InducedDependencyState[C], ...],
        selection: HeadSelection,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Retain a declared coordination marker until its merge.

        Parameters
        ----------
        node : DerivationNode[C]
            Mark output node.
        children : tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...]
            Child projections in derivation order.
        states : tuple[InducedDependencyState[C], ...]
            State for each child.
        selection : HeadSelection
            Declared mark head selection.

        Raises
        ------
        ValueError
            If the mark is not binary at slot one or contains another mark.
        """
        if len(children) != _BINARY_ARITY or selection.slot != 1:
            msg = "an induced coordination mark requires two children and slot 1"
            raise ValueError(msg)

        if any(state.coordinator_heads for state in states):
            msg = "a coordination mark contains an unmerged marker"
            raise ValueError(msg)

        marker = _only_head(children[0].heads, "a coordination marker")
        dependencies = tuple(
            dependency for child in children for dependency in child.dependencies
        )
        return HeadProjection(
            children[1].heads,
            dependencies,
            InducedDependencyState(node.label, (marker,), self._report),
        )

    def _pass_through(
        self,
        node: DerivationNode[C],
        children: tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...],
        states: tuple[InducedDependencyState[C], ...],
        selection: HeadSelection,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Project a pass-through or ordinary non-absorbing step.

        Parameters
        ----------
        node : DerivationNode[C]
            Output node.
        children : tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...]
            Child projections in derivation order.
        states : tuple[InducedDependencyState[C], ...]
            State for each child.
        selection : HeadSelection
            Declared head selection.

        Raises
        ------
        ValueError
            If the step would discard an unmerged marker.
        """
        slot = selection.slot

        if slot is None:
            msg = "a pass-through step has no head slot"
            raise ValueError(msg)

        pending = tuple(
            position for position, state in enumerate(states) if state.coordinator_heads
        )

        if pending and pending != (slot,):
            msg = "a pass-through step would discard a coordination marker"
            raise ValueError(msg)

        selected = children[slot]
        dependencies = tuple(
            dependency for child in children for dependency in child.dependencies
        )
        return HeadProjection(
            selected.heads,
            dependencies,
            InducedDependencyState(
                node.label,
                states[slot].coordinator_heads,
                self._report,
            ),
        )

    def step(
        self,
        node: DerivationNode[C],
        rule_name: str,
        children: tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...],
        selection: HeadSelection,
        /,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Project one rule step.

        Parameters
        ----------
        node : DerivationNode[C]
            Nonlexical derivation node.
        rule_name : str
            Rule licensing the node.
        children : tuple[HeadProjection[Dependency, InducedDependencyState[C]], ...]
            Child projections in derivation order.
        selection : HeadSelection
            Declared semantic policy and physical head slot.

        Raises
        ------
        ValueError
            If a positive-arity step has no head, conventions differ, or a
            coordination step does not follow mark with merge.
        """
        if not children:
            return HeadProjection(
                (),
                (),
                InducedDependencyState(node.label, (), self._report),
            )

        states = tuple(_state(child) for child in children)

        if any(state.report != self._report for state in states):
            msg = "an induced projection step combines different conventions"
            raise ValueError(msg)

        slot = selection.slot

        if slot is None:
            msg = f"positive-arity rule {rule_name!r} has no head slot"
            raise ValueError(msg)

        if slot >= len(children):
            msg = (
                f"rule {rule_name!r} has head slot {slot} outside arity {len(children)}"
            )
            raise ValueError(msg)

        declared_mark = rule_name in self._coordination_marks

        if declared_mark and selection.policy != "non-absorbing":
            msg = "a declared coordination mark requires non-absorbing head policy"
            raise ValueError(msg)

        if selection.policy == "left-conjunct":
            return self._merge(node, children, states, selection)

        if selection.policy == "functor":
            return self._functor(node, children, states, selection)

        if declared_mark:
            return self._mark(node, children, states, selection)

        return self._pass_through(node, children, states, selection)

    def complete(
        self,
        projection: HeadProjection[Dependency, InducedDependencyState[C]],
        /,
    ) -> HeadProjection[Dependency, InducedDependencyState[C]]:
        """Attach a completed sentence projection to root position zero.

        Parameters
        ----------
        projection : HeadProjection[Dependency, InducedDependencyState[C]]
            Complete derivation projection with one sentence head.

        Raises
        ------
        ValueError
            If the projection is incomplete, has other than one head, or already
            has a root arc.
        """
        state = _state(projection)

        if state.report != self._report:
            msg = "an induced projection was completed under another convention"
            raise ValueError(msg)

        if state.coordinator_heads:
            msg = "an induced projection ends with an unmerged coordination marker"
            raise ValueError(msg)

        root = _only_head(projection.heads, "a completed sentence")

        if any(dependency.head == 0 for dependency in projection.dependencies):
            msg = "an induced dependency projection already has a root arc"
            raise ValueError(msg)

        return HeadProjection(
            projection.heads,
            (*projection.dependencies, _arc(-1, root, "root")),
            state,
        )


def project_bisk_hockenmaier_dependencies[C: Hashable](
    derivation: Derivation[C],
    tokens: Sequence[Token],
    head_selections: Mapping[str, HeadSelection],
    projector: BiskHockenmaierDependencyProjector[C],
    /,
    *,
    recovery: StepRecovery[C, str] | None = None,
) -> HeadProjection[Dependency, InducedDependencyState[C]]:
    """Project an induced CCG derivation under the Bisk-Hockenmaier convention.

    Parameters
    ----------
    derivation : Derivation[C]
        Derivation to project.
    tokens : Sequence[Token]
        Tokens indexed from zero.
    head_selections : Mapping[str, HeadSelection]
        Exact head selection for every reached rule.
    projector : BiskHockenmaierDependencyProjector[C]
        Projector with the category algebra and exact coordination mark rules.
    recovery : StepRecovery[C, str] | None
        Optional recovery relation for unnamed steps.

    Returns
    -------
    HeadProjection[Dependency, InducedDependencyState[C]]
        Completed projection with one-based arcs and one root arc from position zero.

    Raises
    ------
    KeyError
        If a lexical node has no token or a rule has no head selection.
    ValueError
        If the derivation cannot form one complete dependency tree.
    """
    projection = project_dependencies(
        derivation,
        tokens,
        head_selections,
        projector,
        recovery=recovery,
    )
    return projector.complete(projection)
