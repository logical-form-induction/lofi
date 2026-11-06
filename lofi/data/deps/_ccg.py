"""Recursive CCG dependency projection.

A projected subtree is a :class:`DecoratedCategory`: its category tree with the
lexical heads at every node, an :class:`ArgumentSlot` on every argument node,
and a flag on every functor node saying whether its result takes its argument's
head. An argument slot names the lexical heads that govern the argument, each
with the slot number through which it governs, so one argument can be shared
by heads that number it differently. Argument nodes may also have a coindex;
nodes with the same coindex are filled by the same heads and gain the same
governors.

Application and composition align the functor's argument with the matching
part of the other daughter node by node. Each aligned pair is unified: both
nodes receive the union of their heads and governors, as does every node
coindexed with the functor's node. A dependency is emitted the first time a
node holds a governor and a head together, and never again.
"""

from typing import (
    TYPE_CHECKING,
    Protocol,
    final,
    override,
    runtime_checkable,
)

from lofi._frozen import FrozenValue
from lofi.data.deps._projection import HeadProjection
from lofi.data.deps._templates import CCGDependencyConvention, CCGProjectionReport

if TYPE_CHECKING:
    from collections.abc import Callable, Hashable, Iterable, Mapping

    from lofi._types import HeadSelection
    from lofi.data import DerivationNode, Token
    from lofi.data.deps._templates import CategoryDirection, CoindexedPath

__all__ = [
    "ArgumentSlot",
    "CCGCategoryAlgebra",
    "CCGDependencyProjector",
    "DecoratedCategory",
    "Governor",
    "LexicalHead",
    "ProjectedDependency",
]

_BINARY_ARITY = 2


@final
class LexicalHead[C: Hashable](FrozenValue):
    """A lexical head and its original category."""

    _noun = "lexical head"

    __slots__ = ("_category", "_form", "_position")

    def __init__(self, position: int, form: str, category: C) -> None:
        if isinstance(position, bool) or not isinstance(position, int):
            msg = f"a lexical-head position is not an integer: {position!r}"
            raise TypeError(msg)

        if position < 0:
            msg = f"a lexical-head position is negative: {position}"
            raise ValueError(msg)

        object.__setattr__(self, "_position", position)
        object.__setattr__(self, "_form", form)
        object.__setattr__(self, "_category", category)

    @property
    def position(self) -> int:
        """Return the token position."""
        return self._position

    @property
    def form(self) -> str:
        """Return the surface form."""
        return self._form

    @property
    def category(self) -> C:
        """Return the original lexical category."""
        return self._category

    def _fields(self) -> tuple[int, str, C]:
        """Return fields in declaration order."""
        return self._position, self._form, self._category

    @override
    def __reduce__(self) -> tuple[type[LexicalHead[C]], tuple[int, str, C]]:
        """Return the reconstruction call."""
        return LexicalHead, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another lexical head has the same fields."""
        if type(other) is not LexicalHead:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((LexicalHead, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"LexicalHead{self._fields()!r}"


type Governor[C: Hashable] = tuple[LexicalHead[C], int]
"""A lexical head and the argument slot through which it governs."""

type Coindex = tuple[int, int]
"""The token position that declared a coindex and the declared number."""


def _require_positive(value: int, label: str) -> None:
    """Raise unless a value is a positive integer."""
    if isinstance(value, bool) or not isinstance(value, int):
        msg = f"an argument {label} is not an integer: {value!r}"
        raise TypeError(msg)

    if value < 1:
        msg = f"an argument {label} is not positive: {value}"
        raise ValueError(msg)


@final
class ArgumentSlot[C: Hashable](FrozenValue):
    """An open argument, its governors, and an optional coindex.

    Parameters
    ----------
    index : int
        The argument's slot number in the category that introduced it.
    governors : Iterable[Governor[C]]
        Each lexical head governing the argument, with the slot through which
        it governs.
    coindex : Coindex | None
        The coindex shared with other argument nodes filled by the same heads.

    Raises
    ------
    TypeError
        If the index, a governing slot or a coindex part is not an integer.
    ValueError
        If the index or a governing slot is not positive, or a coindex has a
        negative position or a nonpositive number.
    """

    _noun = "argument slot"

    __slots__ = ("_coindex", "_governors", "_index")

    def __init__(
        self,
        index: int,
        governors: Iterable[Governor[C]] = (),
        coindex: Coindex | None = None,
    ) -> None:
        _require_positive(index, "slot")
        held_governors = tuple(governors)

        for _, governing_slot in held_governors:
            _require_positive(governing_slot, "governing slot")

        if coindex is not None:
            position, number = coindex

            if isinstance(position, bool) or not isinstance(position, int):
                msg = f"an argument coindex position is not an integer: {position!r}"
                raise TypeError(msg)

            if position < 0:
                msg = f"an argument coindex position is negative: {position}"
                raise ValueError(msg)

            _require_positive(number, "coindex")

        object.__setattr__(self, "_index", index)
        object.__setattr__(self, "_governors", held_governors)
        object.__setattr__(self, "_coindex", coindex)

    @property
    def index(self) -> int:
        """Return the positive argument-slot number."""
        return self._index

    @property
    def governors(self) -> tuple[Governor[C], ...]:
        """Return the governing heads, each with its governing slot."""
        return self._governors

    @property
    def coindex(self) -> Coindex | None:
        """Return the coindex, if declared."""
        return self._coindex

    def _fields(self) -> tuple[int, tuple[Governor[C], ...], Coindex | None]:
        """Return fields in declaration order."""
        return self._index, self._governors, self._coindex

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[ArgumentSlot[C]],
        tuple[int, tuple[Governor[C], ...], Coindex | None],
    ]:
        """Return the reconstruction call."""
        return ArgumentSlot, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another argument slot has the same fields."""
        if type(other) is not ArgumentSlot:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((ArgumentSlot, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"ArgumentSlot{self._fields()!r}"


type _DecoratedFields[C: Hashable] = tuple[
    C,
    tuple[LexicalHead[C], ...],
    DecoratedCategory[C] | None,
    DecoratedCategory[C] | None,
    ArgumentSlot[C] | None,
    bool,
]
"""Identity fields of a decorated category."""

type _DecoratedReconstruction[C: Hashable] = tuple[
    _DecoratedFields[C],
    CCGProjectionReport,
]
"""Serialized decoration fields and nonidentity report evidence."""


@final
class DecoratedCategory[C: Hashable](FrozenValue):
    """A category tree with lexical heads, argument slots and head passing.

    Projection reports do not affect equality, hashing, or representation.

    Parameters
    ----------
    category : C
        The category at this node.
    heads : Iterable[LexicalHead[C]]
        The lexical heads at this node.
    result : DecoratedCategory[C] | None
        The result node of a functor.
    argument : DecoratedCategory[C] | None
        The argument node of a functor.
    slot : ArgumentSlot[C] | None
        The argument binding of a node in argument position.
    shares_head : bool
        Whether a functor's result takes its argument's head.

    Raises
    ------
    ValueError
        If only one of the result and argument nodes is given, or an atomic
        node is declared to share its head.
    """

    _noun = "decorated category"

    __slots__ = (
        "_argument",
        "_category",
        "_heads",
        "_projection_report",
        "_result",
        "_shares_head",
        "_slot_binding",
    )

    def __init__(
        self,
        category: C,
        heads: Iterable[LexicalHead[C]],
        result: DecoratedCategory[C] | None = None,
        argument: DecoratedCategory[C] | None = None,
        slot: ArgumentSlot[C] | None = None,
        *,
        shares_head: bool = False,
    ) -> None:
        if (result is None) != (argument is None):
            msg = "a decorated functor must have both result and argument nodes"
            raise ValueError(msg)

        if shares_head and result is None:
            msg = "an atomic decorated category cannot share its argument's head"
            raise ValueError(msg)

        object.__setattr__(self, "_category", category)
        object.__setattr__(self, "_heads", tuple(heads))
        object.__setattr__(self, "_result", result)
        object.__setattr__(self, "_argument", argument)
        object.__setattr__(self, "_slot_binding", slot)
        object.__setattr__(self, "_shares_head", shares_head)
        object.__setattr__(self, "_projection_report", CCGProjectionReport(None, 0, 0))

    @property
    def category(self) -> C:
        """Return the category at this node."""
        return self._category

    @property
    def heads(self) -> tuple[LexicalHead[C], ...]:
        """Return the lexical heads at this node."""
        return self._heads

    @property
    def result(self) -> DecoratedCategory[C] | None:
        """Return the result node, if this is a functor."""
        return self._result

    @property
    def argument(self) -> DecoratedCategory[C] | None:
        """Return the argument node, if this is a functor."""
        return self._argument

    @property
    def shares_head(self) -> bool:
        """Return whether this functor's result takes its argument's head."""
        return self._shares_head

    @property
    def slot(self) -> int | None:
        """Return this node's argument-slot number, if it has one."""
        return None if self._slot_binding is None else self._slot_binding.index

    @property
    def governors(self) -> tuple[Governor[C], ...]:
        """Return the heads governing this open argument, with their slots."""
        return () if self._slot_binding is None else self._slot_binding.governors

    @property
    def coindex(self) -> Coindex | None:
        """Return this argument node's coindex, if declared."""
        return None if self._slot_binding is None else self._slot_binding.coindex

    @property
    def binding(self) -> ArgumentSlot[C] | None:
        """Return this node's argument binding, if present."""
        return self._slot_binding

    @property
    def matched_template_count(self) -> int:
        """Return the number of lexical templates matched below this node."""
        return self._projection_report.matched_template_count

    @property
    def projection_report(self) -> CCGProjectionReport:
        """Return the convention provenance and template counts."""
        return self._projection_report

    @property
    def open_slots(self) -> tuple[int, ...]:
        """Return unfilled argument-slot numbers in structural order."""
        found: list[int] = []
        pending: list[DecoratedCategory[C]] = [self]

        while pending:
            current = pending.pop()

            if current.slot is not None and not current.heads:
                found.append(current.slot)

            if current.argument is not None:
                pending.append(current.argument)

            if current.result is not None:
                pending.append(current.result)

        return tuple(sorted(found))

    def _fields(self) -> _DecoratedFields[C]:
        """Return identity fields in declaration order."""
        return (
            self._category,
            self._heads,
            self._result,
            self._argument,
            self._slot_binding,
            self._shares_head,
        )

    def _reconstruction(self) -> _DecoratedReconstruction[C]:
        """Return structural fields with nonidentity report evidence."""
        return self._fields(), self._projection_report

    @override
    def __reduce__(
        self,
    ) -> tuple[
        Callable[[_DecoratedReconstruction[C]], DecoratedCategory[C]],
        tuple[_DecoratedReconstruction[C]],
    ]:
        """Return the reconstruction call."""
        return _restore_decorated_category, (self._reconstruction(),)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another decorated category has the same identity fields."""
        if type(other) is not DecoratedCategory:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the category tree."""
        return hash((DecoratedCategory, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the identity fields as a constructor call."""
        return f"DecoratedCategory{self._fields()!r}"


def _restore_decorated_category[C: Hashable](
    reconstruction: _DecoratedReconstruction[C],
) -> DecoratedCategory[C]:
    """Restore a decorated category from serialized fields."""
    fields, report = reconstruction
    category, heads, result, argument, slot, shares_head = fields
    return _with_report(
        DecoratedCategory(
            category, heads, result, argument, slot, shares_head=shares_head
        ),
        report,
    )


@final
class ProjectedDependency[C: Hashable](FrozenValue):
    """A CCGbank predicate-argument dependency."""

    _noun = "projected dependency"

    __slots__ = ("_argument", "_predicate", "_rule", "_slot")

    def __init__(
        self,
        predicate: LexicalHead[C],
        argument: LexicalHead[C],
        slot: int,
        rule: str,
    ) -> None:
        if isinstance(slot, bool) or not isinstance(slot, int):
            msg = f"a projected dependency slot is not an integer: {slot!r}"
            raise TypeError(msg)

        if slot < 1:
            msg = f"a projected dependency slot is not positive: {slot}"
            raise ValueError(msg)

        if not rule:
            msg = "a projected dependency has no extraction rule"
            raise ValueError(msg)

        object.__setattr__(self, "_predicate", predicate)
        object.__setattr__(self, "_argument", argument)
        object.__setattr__(self, "_slot", slot)
        object.__setattr__(self, "_rule", rule)

    @property
    def predicate(self) -> LexicalHead[C]:
        """Return the predicate head."""
        return self._predicate

    @property
    def argument(self) -> LexicalHead[C]:
        """Return the argument head."""
        return self._argument

    @property
    def predicate_position(self) -> int:
        """Return the predicate token position."""
        return self._predicate.position

    @property
    def predicate_form(self) -> str:
        """Return the predicate surface form."""
        return self._predicate.form

    @property
    def argument_position(self) -> int:
        """Return the argument token position."""
        return self._argument.position

    @property
    def argument_form(self) -> str:
        """Return the argument surface form."""
        return self._argument.form

    @property
    def raw_category(self) -> C:
        """Return the predicate's original lexical category."""
        return self._predicate.category

    @property
    def slot(self) -> int:
        """Return the positive predicate-argument slot."""
        return self._slot

    @property
    def rule_id(self) -> str:
        """Return the extraction rule identifier."""
        return self._rule

    def _fields(self) -> tuple[LexicalHead[C], LexicalHead[C], int, str]:
        """Return fields in declaration order."""
        return self._predicate, self._argument, self._slot, self._rule

    @override
    def __reduce__(
        self,
    ) -> tuple[
        type[ProjectedDependency[C]],
        tuple[LexicalHead[C], LexicalHead[C], int, str],
    ]:
        """Return the reconstruction call."""
        return ProjectedDependency, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another dependency has the same fields."""
        if type(other) is not ProjectedDependency:
            return NotImplemented
        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the fields."""
        return hash((ProjectedDependency, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"ProjectedDependency{self._fields()!r}"


@runtime_checkable
class CCGCategoryAlgebra[C](Protocol):
    """Structural operations over CCG categories."""

    def is_functor(self, category: C, /) -> bool:
        """Return whether a category has result and argument nodes."""
        ...

    def result(self, category: C, /) -> C:
        """Return a functor's result category."""
        ...

    def argument(self, category: C, /) -> C:
        """Return a functor's argument category."""
        ...

    def equivalent(self, left: C, right: C, /) -> bool:
        """Return whether two categories have the same relevant structure."""
        ...

    def shares_head(self, category: C, /) -> bool:
        """Return whether a functor's result takes its argument's head."""
        ...

    def coindices(self, category: C, /) -> tuple[CoindexedPath, ...]:
        """Return the argument paths a category itself declares coindexed."""
        ...


def _slot_count[C: Hashable](state: DecoratedCategory[C]) -> int:
    """Count result-spine argument slots."""
    result = state.result
    return 0 if result is None else _slot_count(result) + 1


def _with_slot[C: Hashable](
    state: DecoratedCategory[C],
    slot: ArgumentSlot[C] | None,
) -> DecoratedCategory[C]:
    """Return a node with its argument binding replaced."""
    return DecoratedCategory(
        state.category,
        state.heads,
        state.result,
        state.argument,
        slot,
        shares_head=state.shares_head,
    )


def _decorate[C: Hashable](
    category: C,
    heads: tuple[LexicalHead[C], ...],
    governors: tuple[LexicalHead[C], ...],
    algebra: CCGCategoryAlgebra[C],
    coindices: Mapping[tuple[CategoryDirection, ...], Coindex] | None = None,
) -> DecoratedCategory[C]:
    """Decorate every functor spine with heads, governed slots and head passing.

    Every node on the result spine takes ``heads``, and every argument of that
    spine is governed by ``governors`` through its own slot, up to the first
    functor that passes its argument's head to its result: that functor's
    argument is governed, but the arguments of its result belong to the
    argument's head and are numbered but ungoverned. Arguments nested inside an
    argument are numbered but ungoverned.
    """
    declared = {} if coindices is None else coindices

    def build(
        current: C,
        path: tuple[CategoryDirection, ...],
        current_heads: tuple[LexicalHead[C], ...],
        current_governors: tuple[LexicalHead[C], ...],
    ) -> DecoratedCategory[C]:
        if not algebra.is_functor(current):
            return DecoratedCategory(current, current_heads)

        shares_head = algebra.shares_head(current)
        result = build(
            algebra.result(current),
            (*path, "result"),
            current_heads,
            () if shares_head else current_governors,
        )
        argument_path: tuple[CategoryDirection, ...] = (*path, "argument")
        argument = build(algebra.argument(current), argument_path, (), ())
        slot = _slot_count(result) + 1
        binding = ArgumentSlot(
            slot,
            ((governor, slot) for governor in current_governors),
            declared.get(argument_path),
        )
        return DecoratedCategory(
            current,
            current_heads,
            result,
            _with_slot(argument, binding),
            shares_head=shares_head,
        )

    return build(category, (), heads, governors)


def _type_raised[C: Hashable](
    category: C,
    source: DecoratedCategory[C],
    algebra: CCGCategoryAlgebra[C],
) -> DecoratedCategory[C] | None:
    """Return a type-raised decoration of a source, if the category raises it.

    A category ``T|(T|A)`` raises a source of category ``A``. The raised node
    passes its argument's head to its result, and the source becomes the
    argument's argument, so the functor it is applied to finds its argument
    already filled. A governed source keeps its binding on the raised node
    alone, and the argument's argument takes an ungoverned slot.
    """
    raised = algebra.argument(category)

    if not (
        algebra.is_functor(raised)
        and algebra.equivalent(algebra.argument(raised), source.category)
        and algebra.equivalent(algebra.result(raised), algebra.result(category))
    ):
        return None

    inner_result = _decorate(algebra.result(raised), (), (), algebra)
    inner = _relabel(algebra.argument(raised), source, algebra)
    inner_slot = ArgumentSlot(_slot_count(inner_result) + 1)
    argument = DecoratedCategory(
        raised,
        (),
        inner_result,
        _with_slot(inner, inner_slot),
        shares_head=algebra.shares_head(raised),
    )
    result = _decorate(algebra.result(category), (), (), algebra)
    return DecoratedCategory(
        category,
        source.heads,
        result,
        _with_slot(argument, ArgumentSlot(_slot_count(result) + 1)),
        source.binding,
        shares_head=True,
    )


def _restructured[C: Hashable](
    category: C,
    source: DecoratedCategory[C],
    algebra: CCGCategoryAlgebra[C],
) -> DecoratedCategory[C]:
    """Decorate a functor category for a source that is not a functor.

    A type raising category is built by :func:`_type_raised`. Otherwise the
    category is decorated afresh with the source's heads and no governors. If
    the source is a governed argument and one of the new result-spine
    arguments is equivalent to it, that argument takes the source's heads and
    binding and the new node keeps only an ungoverned slot; if none is, the new
    node keeps the source's binding.
    """
    raised = _type_raised(category, source, algebra)

    if raised is not None:
        return raised

    fresh = _decorate(category, source.heads, (), algebra)
    binding = source.binding

    if binding is None:
        return fresh

    spine: list[DecoratedCategory[C]] = []
    current = fresh

    while current.result is not None and current.argument is not None:
        spine.append(current.argument)
        current = current.result

    for depth in reversed(range(len(spine))):
        argument = spine[depth]

        if algebra.equivalent(argument.category, source.category):
            target = DecoratedCategory(
                argument.category,
                source.heads,
                argument.result,
                argument.argument,
                binding,
                shares_head=argument.shares_head,
            )
            rebuilt = _replaced(fresh, depth, target)
            return _with_slot(rebuilt, ArgumentSlot(binding.index))

    return _with_slot(fresh, binding)


def _replaced[C: Hashable](
    state: DecoratedCategory[C],
    depth: int,
    argument: DecoratedCategory[C],
) -> DecoratedCategory[C]:
    """Return a state with the argument at a result-spine depth replaced."""
    result = state.result

    if result is None:
        return state

    return DecoratedCategory(
        state.category,
        state.heads,
        _replaced(result, depth - 1, argument) if depth else result,
        state.argument if depth else argument,
        state.binding,
        shares_head=state.shares_head,
    )


def _relabel[C: Hashable](
    category: C,
    source: DecoratedCategory[C],
    algebra: CCGCategoryAlgebra[C],
) -> DecoratedCategory[C]:
    """Relabel a decorated category with a new category.

    Where the new category is equivalent to the source's, every node keeps
    its heads, binding and head passing. Where it differs, matching functor
    structure is relabeled part by part and its head passing is read from the
    new category; a functor replacing an atom is built by
    :func:`_restructured`; and an atom keeps the source's heads and binding.
    """
    same = algebra.equivalent(category, source.category)

    if algebra.is_functor(category):
        if source.result is None or source.argument is None:
            return _with_report(
                _restructured(category, source, algebra),
                source.projection_report,
            )

        result = _relabel(algebra.result(category), source.result, algebra)
        argument = _relabel(algebra.argument(category), source.argument, algebra)
        return _with_report(
            DecoratedCategory(
                category,
                source.heads,
                result,
                argument,
                source.binding,
                shares_head=(
                    source.shares_head if same else algebra.shares_head(category)
                ),
            ),
            source.projection_report,
        )

    return _with_report(
        DecoratedCategory(category, source.heads, slot=source.binding),
        source.projection_report,
    )


def _with_report[C: Hashable](
    state: DecoratedCategory[C],
    report: CCGProjectionReport,
) -> DecoratedCategory[C]:
    """Set convention provenance and template counts at a state root."""
    rebuilt = DecoratedCategory(
        state.category,
        state.heads,
        state.result,
        state.argument,
        state.binding,
        shares_head=state.shares_head,
    )
    object.__setattr__(rebuilt, "_projection_report", report)
    return rebuilt


def _state[C: Hashable](
    projection: HeadProjection[ProjectedDependency[C], DecoratedCategory[C]],
) -> DecoratedCategory[C]:
    """Require a projected CCG state."""
    state = projection.state

    if state is None:
        msg = "a CCG dependency projection has no decorated category"
        raise ValueError(msg)

    return state


def _held_dependencies[C: Hashable](
    children: tuple[HeadProjection[ProjectedDependency[C], DecoratedCategory[C]], ...],
) -> tuple[ProjectedDependency[C], ...]:
    """Collect child dependencies in derivation order."""
    return tuple(dependency for child in children for dependency in child.dependencies)


def _held_report[C: Hashable](
    children: tuple[HeadProjection[ProjectedDependency[C], DecoratedCategory[C]], ...],
) -> CCGProjectionReport:
    """Combine compatible child projection reports."""
    reports = tuple(_state(child).projection_report for child in children)

    if not reports:
        return CCGProjectionReport(None, 0, 0)

    first = reports[0]
    provenance = first.convention_name, first.declared_template_count

    if any(
        (report.convention_name, report.declared_template_count) != provenance
        for report in reports[1:]
    ):
        msg = "a CCG projection step combines different dependency conventions"
        raise ValueError(msg)

    return CCGProjectionReport(
        first.convention_name,
        first.declared_template_count,
        sum(report.matched_template_count for report in reports),
    )


def _unique[T: Hashable](*groups: Iterable[T]) -> tuple[T, ...]:
    """Combine groups in order without repeating an element."""
    return tuple(dict.fromkeys(item for group in groups for item in group))


def _composition_depth[C: Hashable](
    expected: DecoratedCategory[C],
    other: DecoratedCategory[C],
    algebra: CCGCategoryAlgebra[C],
) -> int | None:
    """Return how far down another functor's result spine a category matches."""
    current = other
    depth = 0

    while current.result is not None and current.argument is not None:
        current = current.result
        depth += 1

        if algebra.equivalent(expected.category, current.category):
            return depth

    return None


def _spine[C: Hashable](
    state: DecoratedCategory[C],
    depth: int,
) -> tuple[DecoratedCategory[C], tuple[DecoratedCategory[C], ...]]:
    """Return the result-spine node at a depth and the arguments above it."""
    outside: list[DecoratedCategory[C]] = []
    current = state

    for _ in range(depth):
        if current.result is None or current.argument is None:
            msg = "a composed projection output has too few argument nodes"
            raise ValueError(msg)

        outside.append(current.argument)
        current = current.result

    return current, tuple(reversed(outside))


type _Path = tuple[CategoryDirection, ...]
"""Result and argument steps from a state root."""

type _Update[C: Hashable] = tuple[tuple[LexicalHead[C], ...], tuple[Governor[C], ...]]
"""Heads and governors a node gains by unification."""


def _aligned[C: Hashable](
    expected: DecoratedCategory[C],
    actual: DecoratedCategory[C],
    algebra: CCGCategoryAlgebra[C],
) -> tuple[tuple[_Path, DecoratedCategory[C], DecoratedCategory[C]], ...]:
    """Pair the nodes of two equivalent category trees, root first.

    Pairing descends below a pair of functors only while their results and
    arguments are pairwise equivalent.
    """
    pairs: list[tuple[_Path, DecoratedCategory[C], DecoratedCategory[C]]] = []
    pending: list[tuple[_Path, DecoratedCategory[C], DecoratedCategory[C]]] = [
        ((), expected, actual)
    ]

    while pending:
        path, left, right = pending.pop()
        pairs.append((path, left, right))

        if (
            left.result is not None
            and left.argument is not None
            and right.result is not None
            and right.argument is not None
            and algebra.equivalent(left.result.category, right.result.category)
            and algebra.equivalent(left.argument.category, right.argument.category)
        ):
            pending.append(((*path, "argument"), left.argument, right.argument))
            pending.append(((*path, "result"), left.result, right.result))

    return tuple(pairs)


def _merged_update[K: Hashable, C: Hashable](
    updates: dict[K, _Update[C]],
    key: K,
    update: _Update[C],
) -> None:
    """Add heads and governors to the update recorded for a key."""
    heads, governors = updates.get(key, ((), ()))
    updates[key] = (_unique(heads, update[0]), _unique(governors, update[1]))


class _Unification[C: Hashable]:
    """Heads and governors gained by nodes, by path and by coindex."""

    def __init__(self) -> None:
        self.paths: dict[_Path, _Update[C]] = {}
        self.coindices: dict[Coindex, _Update[C]] = {}

    def apply(
        self,
        state: DecoratedCategory[C],
        rule_name: str,
        emitted: list[ProjectedDependency[C]],
    ) -> DecoratedCategory[C]:
        """Return a state with every gain applied, appending its dependencies.

        Every governor and head of a node that gains anything is appended,
        including pairs it already held; the caller drops repeats.
        """

        def visit(current: DecoratedCategory[C], path: _Path) -> DecoratedCategory[C]:
            gains: list[_Update[C]] = []

            if path in self.paths:
                gains.append(self.paths[path])

            if current.coindex is not None and current.coindex in self.coindices:
                gains.append(self.coindices[current.coindex])

            heads = current.heads
            binding = current.binding

            if gains:
                heads = _unique(heads, *(gained for gained, _ in gains))
                governors = _unique(
                    current.governors, *(governed for _, governed in gains)
                )
                emitted.extend(
                    ProjectedDependency(predicate, head, slot, rule_name)
                    for predicate, slot in governors
                    for head in heads
                )

                if binding is not None:
                    binding = ArgumentSlot(binding.index, governors, binding.coindex)

            result = (
                None
                if current.result is None
                else visit(current.result, (*path, "result"))
            )
            argument = (
                None
                if current.argument is None
                else visit(current.argument, (*path, "argument"))
            )
            return DecoratedCategory(
                current.category,
                heads,
                result,
                argument,
                binding,
                shares_head=current.shares_head,
            )

        return _with_report(visit(state, ()), state.projection_report)


def _arc[C: Hashable](
    dependency: ProjectedDependency[C],
) -> tuple[LexicalHead[C], LexicalHead[C], int]:
    """Return a dependency's predicate, argument and slot, without its rule."""
    return dependency.predicate, dependency.argument, dependency.slot


def _composed_state[C: Hashable](
    category: C,
    remaining: DecoratedCategory[C],
    outside: tuple[DecoratedCategory[C], ...],
    heads: tuple[LexicalHead[C], ...],
    algebra: CCGCategoryAlgebra[C],
) -> DecoratedCategory[C]:
    """Build an output spine from a result and composed arguments."""
    if not outside:
        return _relabel(category, remaining, algebra)

    if not algebra.is_functor(category):
        msg = "a composed projection output has too few argument nodes"
        raise ValueError(msg)

    result = _composed_state(
        algebra.result(category),
        remaining,
        outside[:-1],
        heads,
        algebra,
    )
    argument = _relabel(algebra.argument(category), outside[-1], algebra)
    return DecoratedCategory(category, heads, result, argument)


def _functor_projection[C: Hashable](
    node: DerivationNode[C],
    rule_name: str,
    children: tuple[HeadProjection[ProjectedDependency[C], DecoratedCategory[C]], ...],
    selection: HeadSelection,
    algebra: CCGCategoryAlgebra[C],
) -> HeadProjection[ProjectedDependency[C], DecoratedCategory[C]]:
    """Project an application or composition step.

    A functor that passes its argument's head yields the other daughter's
    heads, and the output's root keeps exactly those heads: heads already held
    by the functor's argument node reach the other daughter's arguments but not
    its root.
    """
    if len(children) != _BINARY_ARITY:
        msg = f"functor step {rule_name!r} has arity {len(children)} rather than two"
        raise ValueError(msg)

    slot = selection.slot

    if slot is None:
        msg = f"positive-arity rule {rule_name!r} has no head slot"
        raise ValueError(msg)

    functor = _state(children[slot])
    other = _state(children[1 - slot])
    expected = functor.argument
    remaining = functor.result

    if expected is None or remaining is None:
        msg = f"selected head of functor rule {rule_name!r} is not a functor"
        raise ValueError(msg)

    if algebra.equivalent(expected.category, other.category):
        depth = 0
    else:
        found = _composition_depth(expected, other, algebra)

        if found is None:
            msg = f"functor rule {rule_name!r} does not apply or compose structurally"
            raise ValueError(msg)

        depth = found

    if expected.slot is None:
        msg = f"functor rule {rule_name!r} reaches an unnumbered argument"
        raise ValueError(msg)

    actual, _ = _spine(other, depth)
    functor_side = _Unification[C]()
    other_side = _Unification[C]()
    spine: _Path = ("result",) * depth

    for path, left, right in _aligned(expected, actual, algebra):
        gained = (
            _unique(left.heads, right.heads),
            _unique(left.governors, right.governors),
        )

        if left.coindex is not None:
            _merged_update(functor_side.coindices, left.coindex, gained)

        if right.coindex is not None:
            _merged_update(other_side.coindices, right.coindex, gained)

        _merged_update(functor_side.paths, ("argument", *path), gained)
        _merged_update(other_side.paths, (*spine, *path), gained)

    emitted: list[ProjectedDependency[C]] = []
    unified_functor = functor_side.apply(functor, rule_name, emitted)
    unified_other = other_side.apply(other, rule_name, emitted)
    report = _held_report(children)

    if functor.shares_head:
        heads = other.heads
        passed = DecoratedCategory(
            unified_other.category,
            heads,
            unified_other.result,
            unified_other.argument,
            unified_other.binding,
            shares_head=unified_other.shares_head,
        )
        state = _relabel(node.label, passed, algebra)
    else:
        heads = functor.heads
        unified_remaining = unified_functor.result

        if unified_remaining is None:
            msg = f"selected head of functor rule {rule_name!r} is not a functor"
            raise ValueError(msg)

        _, outside = _spine(unified_other, depth)
        state = _composed_state(
            node.label,
            unified_remaining,
            outside,
            heads,
            algebra,
        )

    held = _held_dependencies(children)
    seen = frozenset(_arc(dependency) for dependency in held)
    return HeadProjection(
        tuple(head.position for head in heads),
        (
            *held,
            *(
                dependency
                for dependency in _unique(emitted)
                if _arc(dependency) not in seen
            ),
        ),
        _with_report(state, report),
    )


def _merged_conjuncts[C: Hashable](
    left: DecoratedCategory[C],
    right: DecoratedCategory[C],
    algebra: CCGCategoryAlgebra[C],
) -> DecoratedCategory[C]:
    """Merge two conjuncts node by node, uniting heads and governors.

    The merge descends into a pair of functors only while their categories
    are equivalent; below that the left conjunct's nodes are kept.
    """
    result = left.result
    argument = left.argument

    if (
        left.result is not None
        and left.argument is not None
        and right.result is not None
        and right.argument is not None
        and algebra.equivalent(left.category, right.category)
    ):
        result = _merged_conjuncts(left.result, right.result, algebra)
        argument = _merged_conjuncts(left.argument, right.argument, algebra)

    binding = left.binding

    if binding is not None:
        binding = ArgumentSlot(
            binding.index,
            _unique(left.governors, right.governors),
            binding.coindex,
        )

    return DecoratedCategory(
        left.category,
        _unique(left.heads, right.heads),
        result,
        argument,
        binding,
        shares_head=left.shares_head,
    )


def _template_path_target[C: Hashable](
    category: C,
    path: tuple[CategoryDirection, ...],
    template_name: str,
    algebra: CCGCategoryAlgebra[C],
) -> C:
    """Resolve one declared path to its argument category."""
    if path[-1] != "argument":
        msg = (
            f"dependency template {template_name!r} path {path!r} does not name "
            "an argument node"
        )
        raise ValueError(msg)

    current = category

    for direction in path:
        if not algebra.is_functor(current):
            msg = (
                f"dependency template {template_name!r} path {path!r} does not "
                "name an argument node"
            )
            raise ValueError(msg)

        current = (
            algebra.result(current)
            if direction == "result"
            else algebra.argument(current)
        )

    return current


def _validate_convention[C: Hashable](
    convention: CCGDependencyConvention[C],
    algebra: CCGCategoryAlgebra[C],
) -> None:
    """Validate one dependency convention against its category algebra."""
    templates = convention.templates

    for position, template in enumerate(templates):
        targets: dict[int, list[C]] = {}

        for binding in template.coindices:
            target = _template_path_target(
                template.category,
                binding.path,
                template.name,
                algebra,
            )
            targets.setdefault(binding.index, []).append(target)

        for coindex, grouped in targets.items():
            first = grouped[0]

            if any(not algebra.equivalent(first, target) for target in grouped[1:]):
                msg = (
                    f"dependency template {template.name!r} coindex {coindex} "
                    "joins nonequivalent argument categories"
                )
                raise ValueError(msg)

        for other in templates[position + 1 :]:
            if algebra.equivalent(template.category, other.category):
                msg = (
                    "dependency convention has overlapping templates: "
                    f"{template.name!r} and {other.name!r}"
                )
                raise ValueError(msg)


@final
class CCGDependencyProjector[C: Hashable](FrozenValue):
    """Structural CCG dependency projection.

    A lexical category's coindices are those its algebra reads from the
    category itself together with those of the first convention template
    equivalent to it.

    Parameters
    ----------
    algebra : CCGCategoryAlgebra[C]
        Structural category operations used for every match.
    convention : CCGDependencyConvention[C] | None
        Optional caller-owned nonlocal dependency declaration.

    Raises
    ------
    ValueError
        If a template path is invalid, coindexed arguments are incompatible, or
        templates overlap under the algebra.
    """

    _noun = "CCG dependency projector"

    __slots__ = ("_algebra", "_convention")

    def __init__(
        self,
        algebra: CCGCategoryAlgebra[C],
        convention: CCGDependencyConvention[C] | None = None,
    ) -> None:
        if convention is not None:
            _validate_convention(convention, algebra)

        object.__setattr__(self, "_algebra", algebra)
        object.__setattr__(self, "_convention", convention)

    @property
    def algebra(self) -> CCGCategoryAlgebra[C]:
        """Return the injected category algebra."""
        return self._algebra

    @property
    def convention(self) -> CCGDependencyConvention[C] | None:
        """Return the caller-supplied dependency convention, if present."""
        return self._convention

    def report(
        self,
        projection: HeadProjection[ProjectedDependency[C], DecoratedCategory[C]],
        /,
    ) -> CCGProjectionReport:
        """Return the convention provenance and counts bound to a projection.

        Parameters
        ----------
        projection : HeadProjection[ProjectedDependency[C], DecoratedCategory[C]]
            Completed projection whose bound report is returned.

        Raises
        ------
        ValueError
            If the projection has no completed state.
        """
        return _state(projection).projection_report

    def _projection_report(self, matched_templates: int) -> CCGProjectionReport:
        """Build a report for this projector's declared convention."""
        convention = self._convention

        if convention is None:
            return CCGProjectionReport(None, 0, matched_templates)

        return CCGProjectionReport(
            convention.name,
            len(convention.templates),
            matched_templates,
        )

    def lexical(
        self,
        node: DerivationNode[C],
        token: Token,
        /,
    ) -> HeadProjection[ProjectedDependency[C], DecoratedCategory[C]]:
        """Project a lexical category."""
        head = LexicalHead(token.index, token.surface, node.label)
        convention = self._convention
        template = (
            None
            if convention is None
            else next(
                (
                    candidate
                    for candidate in convention.templates
                    if self._algebra.equivalent(node.label, candidate.category)
                ),
                None,
            )
        )
        declared = (
            *self._algebra.coindices(node.label),
            *(() if template is None else template.coindices),
        )
        coindices = {binding.path: (token.index, binding.index) for binding in declared}
        matched_templates = 0 if template is None else 1

        state = _decorate(node.label, (head,), (head,), self._algebra, coindices)
        return HeadProjection(
            (token.index,),
            (),
            _with_report(state, self._projection_report(matched_templates)),
        )

    def step(
        self,
        node: DerivationNode[C],
        rule_name: str,
        children: tuple[
            HeadProjection[ProjectedDependency[C], DecoratedCategory[C]], ...
        ],
        selection: HeadSelection,
        /,
    ) -> HeadProjection[ProjectedDependency[C], DecoratedCategory[C]]:
        """Project one structurally classified CCG step."""
        dependencies = _held_dependencies(children)
        slot = selection.slot

        if not children:
            state = _decorate(node.label, (), (), self._algebra)
            return HeadProjection(
                (), dependencies, _with_report(state, self._projection_report(0))
            )

        if slot is None:
            msg = f"positive-arity rule {rule_name!r} has no head slot"
            raise ValueError(msg)

        selected = _state(children[slot])

        if len(children) == 1:
            state = _relabel(node.label, selected, self._algebra)
            return HeadProjection(
                tuple(head.position for head in state.heads), dependencies, state
            )

        if selection.policy == "left-conjunct":
            merged = _state(children[0])

            for child in children[1:]:
                merged = _merged_conjuncts(merged, _state(child), self._algebra)

            state = _relabel(node.label, merged, self._algebra)
            state = _with_report(state, _held_report(children))
            return HeadProjection(
                tuple(head.position for head in state.heads), dependencies, state
            )

        if selection.policy != "functor":
            state = _relabel(node.label, selected, self._algebra)
            state = _with_report(state, _held_report(children))
            return HeadProjection(
                tuple(head.position for head in state.heads),
                dependencies,
                state,
            )

        return _functor_projection(
            node,
            rule_name,
            children,
            selection,
            self._algebra,
        )
