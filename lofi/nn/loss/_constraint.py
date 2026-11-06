"""Batch localization for analyst constraints.

A constraint names dataset occurrences of a span and what the span is pinned to:
a permitted set of categories, or one logical form. Both kinds localize to a batch
through ``resolve_spans``, and the batch-local value they produce is the same type,
so an objective scores a pinned meaning exactly as it scores a pinned category.
"""

from typing import TYPE_CHECKING, final, override

from lofi._checks import ensure_finite, ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi._types import ItemId, ReadoutId
from lofi.lexicon import ClampMode, ClampScope
from lofi.terms import TargetSet

if TYPE_CHECKING:
    from collections.abc import Hashable, Iterable, Sequence

    from lofi.terms import Matching, Term

__all__ = [
    "BatchLocalConstraint",
    "BatchLocalNegativeBracket",
    "Constraint",
    "ConstraintOccurrence",
    "LogicalFormConstraint",
    "NegativeBracketConstraint",
    "resolve_spans",
]

type _Span = tuple[int, int]
"""A half-open token span."""


def _checked_span(span: _Span) -> _Span:
    """Return a validated nonempty token span.

    Parameters
    ----------
    span : _Span
        The half-open span to validate.

    Raises
    ------
    TypeError
        If the value is not iterable or an endpoint is not an integer.
    ValueError
        If there are not two endpoints or the span is negative, empty, or reversed.
    """
    start, end = span
    checked_start = ensure_non_negative("a constraint span's start", start)
    checked_end = ensure_non_negative("a constraint span's end", end)

    if checked_end <= checked_start:
        msg = f"a constraint span is empty or reversed: [{start}, {end})"
        raise ValueError(msg)

    return checked_start, checked_end


def _checked_penalty(mode: ClampMode, penalty: float | None, /) -> float | None:
    """Return a penalty coherent with its clamp mode."""
    if mode is ClampMode.HARD:
        if penalty is not None:
            msg = "a hard constraint has no penalty"
            raise ValueError(msg)

        return None

    if penalty is None:
        msg = "a soft constraint declares a positive finite penalty"
        raise ValueError(msg)

    checked = ensure_finite("a soft constraint penalty", penalty)

    if checked <= 0.0:
        msg = f"a soft constraint penalty is not positive: {penalty!r}"
        raise ValueError(msg)

    return checked


@final
class ConstraintOccurrence(FrozenValue):
    """One dataset item and constrained span.

    Parameters
    ----------
    item_id : ItemId
        Dataset item identifier.
    span : _Span
        Constrained half-open token span.

    Raises
    ------
    TypeError
        If an item identifier is not text or a span endpoint is not an integer.
    ValueError
        If the item identifier is empty or the span is empty or reversed.
    """

    _noun = "constraint occurrence"

    __slots__ = ("_item_id", "_span")

    def __init__(self, item_id: ItemId, span: _Span) -> None:
        object.__setattr__(
            self,
            "_item_id",
            ItemId(ensure_name("a constraint occurrence's item id", item_id)),
        )
        object.__setattr__(self, "_span", _checked_span(span))

    @property
    def item_id(self) -> ItemId:
        """Return the dataset item identifier."""
        return self._item_id

    @property
    def span(self) -> _Span:
        """Return the constrained token span."""
        return self._span

    def _fields(self) -> tuple[ItemId, _Span]:
        """Return fields in constructor order."""
        return self._item_id, self._span

    @override
    def __reduce__(self) -> tuple[type[ConstraintOccurrence], tuple[ItemId, _Span]]:
        """Return the reconstruction call."""
        return ConstraintOccurrence, self._fields()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another occurrence has the same fields."""
        if type(other) is not ConstraintOccurrence:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the item identifier and span."""
        return hash((ConstraintOccurrence, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the rebuilding constructor call."""
        return f"ConstraintOccurrence{self._fields()!r}"


def _positions_by_spans(
    occurrences: Iterable[ConstraintOccurrence],
    batch_item_ids: Sequence[ItemId],
    /,
) -> dict[tuple[_Span, ...], list[int]]:
    """Group batch positions by the spans declared for their item."""
    spans_by_item: dict[ItemId, list[_Span]] = {}

    for occurrence in occurrences:
        spans_by_item.setdefault(occurrence.item_id, []).append(occurrence.span)

    positions_by_spans: dict[tuple[_Span, ...], list[int]] = {}

    for position, item_id in enumerate(batch_item_ids):
        spans = spans_by_item.get(item_id)

        if spans is not None:
            positions_by_spans.setdefault(tuple(spans), []).append(position)

    return positions_by_spans


@final
class BatchLocalNegativeBracket[ScopeT: Hashable](FrozenValue):
    """A requirement that named spans not be constituents in one batch.

    A negative bracket has no category or meaning target. Its probability is the
    complement of the parser's constituent marginal, so it carries neither a target
    set nor a readout identifier.
    """

    _noun = "batch-local negative bracket"

    __slots__ = ("_item_positions", "_scope", "_spans")

    def __init__(
        self,
        spans: Iterable[_Span],
        item_positions: Iterable[int],
        scope: ScopeT,
    ) -> None:
        held_spans = tuple(_checked_span(span) for span in spans)
        held_positions = tuple(
            ensure_non_negative("a batch-local item position", position)
            for position in item_positions
        )

        if not held_spans:
            msg = "a batch-local negative bracket has no span occurrence"
            raise ValueError(msg)

        if not held_positions:
            msg = "a batch-local negative bracket has no item position"
            raise ValueError(msg)

        object.__setattr__(self, "_spans", held_spans)
        object.__setattr__(self, "_item_positions", held_positions)
        object.__setattr__(self, "_scope", scope)

    @property
    def spans(self) -> tuple[_Span, ...]:
        """Return the excluded constituent spans in source order."""
        return self._spans

    @property
    def item_positions(self) -> tuple[int, ...]:
        """Return the contributing positions in batch order."""
        return self._item_positions

    @property
    def scope(self) -> ScopeT:
        """Return the originating scope."""
        return self._scope

    def _fields(self) -> tuple[tuple[_Span, ...], tuple[int, ...], ScopeT]:
        """Return fields in constructor order."""
        return self._spans, self._item_positions, self._scope

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another negative bracket has the same fields."""
        if type(other) is not BatchLocalNegativeBracket:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the batch-local fields."""
        return hash((BatchLocalNegativeBracket, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the field values."""
        return (
            f"BatchLocalNegativeBracket(spans={self._spans!r}, "
            f"item_positions={self._item_positions!r}, scope={self._scope!r})"
        )


@final
class BatchLocalConstraint[T, ScopeT: Hashable](FrozenValue):
    """A constraint localized to positions in one batch.

    Parameters
    ----------
    targets : TargetSet[T]
        Permitted categories and their matching relation.
    spans : Iterable[_Span]
        Occurrence spans shared by the selected batch items.
    item_positions : Iterable[int]
        Positions of the selected items in the batch.
    scope : ScopeT
        Original constraint scope retained as provenance.
    clamp_scope : ClampScope
        Structural unit constrained by the declaration.
    readout : ReadoutId
        Published mass read by the localized constraint.
    mode : ClampMode
        Whether the constraint is hard or carries a finite penalty.
    penalty : float | None
        Positive multiplier for a soft constraint; absent for a hard constraint.
    negate : bool
        Whether the constraint excludes rather than permits the targets.

    Raises
    ------
    TypeError
        If a span endpoint or item position is not an integer.
    ValueError
        If the target, span, or item-position sequence is empty, or a value is
        outside its valid range.
    """

    _noun = "batch-local constraint"

    __slots__ = (
        "_clamp_scope",
        "_item_positions",
        "_mode",
        "_negate",
        "_penalty",
        "_readout",
        "_scope",
        "_spans",
        "_targets",
    )

    def __init__(
        self,
        targets: TargetSet[T],
        spans: Iterable[_Span],
        item_positions: Iterable[int],
        scope: ScopeT,
        *,
        readout: ReadoutId,
        clamp_scope: ClampScope = ClampScope.CONSTITUENT,
        mode: ClampMode = ClampMode.HARD,
        penalty: float | None = None,
        negate: bool = False,
    ) -> None:
        held_spans = tuple(_checked_span(span) for span in spans)
        held_positions = tuple(
            ensure_non_negative("a batch-local item position", position)
            for position in item_positions
        )

        if not targets.categories:
            msg = "a batch-local constraint has no target"
            raise ValueError(msg)

        if not held_spans:
            msg = "a batch-local constraint has no span occurrence"
            raise ValueError(msg)

        if not held_positions:
            msg = "a batch-local constraint has no item position"
            raise ValueError(msg)

        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "_spans", held_spans)
        object.__setattr__(self, "_item_positions", held_positions)
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_clamp_scope", clamp_scope)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_penalty", _checked_penalty(mode, penalty))
        object.__setattr__(
            self,
            "_readout",
            ReadoutId(ensure_name("a constraint readout id", readout)),
        )
        object.__setattr__(self, "_negate", negate)

    @property
    def targets(self) -> TargetSet[T]:
        """Return the target categories and their matching relation."""
        return self._targets

    @property
    def spans(self) -> tuple[_Span, ...]:
        """Return the occurrence spans in source order."""
        return self._spans

    @property
    def item_positions(self) -> tuple[int, ...]:
        """Return the contributing positions in batch order."""
        return self._item_positions

    @property
    def scope(self) -> ScopeT:
        """Return the originating scope."""
        return self._scope

    @property
    def clamp_scope(self) -> ClampScope:
        """Return the structural unit constrained by the declaration."""
        return self._clamp_scope

    @property
    def readout(self) -> ReadoutId:
        """Return the published mass read by this constraint."""
        return self._readout

    @property
    def mode(self) -> ClampMode:
        """Return whether the constraint is hard or soft."""
        return self._mode

    @property
    def penalty(self) -> float | None:
        """Return the soft penalty, or None for a hard constraint."""
        return self._penalty

    @property
    def negate(self) -> bool:
        """Return whether the targets are excluded."""
        return self._negate

    def _fields(
        self,
    ) -> tuple[
        TargetSet[T],
        tuple[_Span, ...],
        tuple[int, ...],
        ScopeT,
        ReadoutId,
        ClampScope,
        ClampMode,
        float | None,
        bool,
    ]:
        """Return fields in constructor order."""
        return (
            self._targets,
            self._spans,
            self._item_positions,
            self._scope,
            self._readout,
            self._clamp_scope,
            self._mode,
            self._penalty,
            self._negate,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another batch-local constraint has the same fields."""
        if type(other) is not BatchLocalConstraint:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the batch-local fields."""
        return hash((BatchLocalConstraint, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the field values."""
        return (
            f"BatchLocalConstraint(targets={self._targets!r}, "
            f"spans={self._spans!r}, item_positions={self._item_positions!r}, "
            f"scope={self._scope!r}, readout={self._readout!r}, "
            f"clamp_scope={self._clamp_scope.value!r}, "
            f"mode={self._mode.value!r}, penalty={self._penalty!r}, "
            f"negate={self._negate!r})"
        )


def resolve_spans[T, ScopeT: Hashable](
    targets: TargetSet[T],
    occurrences: Iterable[ConstraintOccurrence],
    scope: ScopeT,
    batch_item_ids: Sequence[ItemId],
    /,
    *,
    readout: ReadoutId,
    clamp_scope: ClampScope,
    mode: ClampMode,
    penalty: float | None,
    negate: bool,
) -> tuple[BatchLocalConstraint[T, ScopeT], ...]:
    """Return dataset occurrences localized to one batch.

    Parameters
    ----------
    targets : TargetSet[T]
        Permitted categories and their matching relation.
    occurrences : Iterable[ConstraintOccurrence]
        Dataset item and span occurrences in source order.
    scope : ScopeT
        Original constraint scope retained as provenance.
    batch_item_ids : Sequence[ItemId]
        Dataset item identifiers in batch order.
    readout : ReadoutId
        Published target mass assigned by the wiring layer.
    clamp_scope : ClampScope
        Structural unit constrained by the declaration.
    mode : ClampMode
        Whether the constraint is hard or soft.
    penalty : float | None
        Positive multiplier for a soft constraint; absent for a hard constraint.
    negate : bool
        Whether the constraint excludes rather than permits the targets.

    Returns
    -------
    tuple[BatchLocalConstraint[T, ScopeT], ...]
        Local constraints grouped by identical span sequences, each listing every
        batch position whose item has that sequence, in batch order.

    Raises
    ------
    TypeError
        If the readout id is not text.
    ValueError
        If the readout id is empty.
    """
    checked_readout = ReadoutId(ensure_name("a constraint readout id", readout))
    positions_by_spans = _positions_by_spans(occurrences, batch_item_ids)

    return tuple(
        BatchLocalConstraint(
            targets,
            spans,
            positions,
            scope,
            readout=checked_readout,
            clamp_scope=clamp_scope,
            mode=mode,
            penalty=penalty,
            negate=negate,
        )
        for spans, positions in positions_by_spans.items()
    )


@final
class NegativeBracketConstraint[ScopeT: Hashable](FrozenValue):
    """A dataset-global requirement that named spans not be constituents."""

    _noun = "negative-bracket constraint"

    __slots__ = ("_occurrences", "_scope")

    def __init__(
        self,
        occurrences: Iterable[ConstraintOccurrence],
        scope: ScopeT,
    ) -> None:
        held_occurrences = tuple(occurrences)

        if not held_occurrences:
            msg = "a negative-bracket constraint has no occurrence"
            raise ValueError(msg)

        object.__setattr__(self, "_occurrences", held_occurrences)
        object.__setattr__(self, "_scope", scope)

    @property
    def occurrences(self) -> tuple[ConstraintOccurrence, ...]:
        """Return the dataset occurrences in source order."""
        return self._occurrences

    @property
    def scope(self) -> ScopeT:
        """Return the originating scope."""
        return self._scope

    def to_spans(
        self,
        batch_item_ids: Sequence[ItemId],
        /,
    ) -> tuple[BatchLocalNegativeBracket[ScopeT], ...]:
        """Return this negative bracket localized to one batch."""
        return tuple(
            BatchLocalNegativeBracket(spans, positions, self._scope)
            for spans, positions in _positions_by_spans(
                self._occurrences, batch_item_ids
            ).items()
        )

    def _fields(self) -> tuple[tuple[ConstraintOccurrence, ...], ScopeT]:
        """Return fields in constructor order."""
        return self._occurrences, self._scope

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another negative bracket has the same fields."""
        if type(other) is not NegativeBracketConstraint:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the dataset-global fields."""
        return hash((NegativeBracketConstraint, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the field values."""
        return (
            f"NegativeBracketConstraint(occurrences={self._occurrences!r}, "
            f"scope={self._scope!r})"
        )


@final
class Constraint[T, ScopeT: Hashable](FrozenValue):
    """A target constraint over dataset-global occurrences.

    Parameters
    ----------
    targets : TargetSet[T]
        Permitted categories and their matching relation.
    occurrences : Iterable[ConstraintOccurrence]
        Dataset item and span occurrences in source order.
    scope : ScopeT
        Original constraint scope retained as provenance.
    clamp_scope : ClampScope
        Structural unit constrained by the declaration.
    mode : ClampMode
        Whether the constraint is hard or carries a finite penalty.
    penalty : float | None
        Positive multiplier for a soft constraint; absent for a hard constraint.
    negate : bool
        Whether the constraint excludes rather than permits the targets.

    Raises
    ------
    ValueError
        If the target or occurrence sequence is empty.
    """

    _noun = "constraint"

    __slots__ = (
        "_clamp_scope",
        "_mode",
        "_negate",
        "_occurrences",
        "_penalty",
        "_scope",
        "_targets",
    )

    def __init__(
        self,
        targets: TargetSet[T],
        occurrences: Iterable[ConstraintOccurrence],
        scope: ScopeT,
        *,
        clamp_scope: ClampScope = ClampScope.CONSTITUENT,
        mode: ClampMode = ClampMode.HARD,
        penalty: float | None = None,
        negate: bool = False,
    ) -> None:
        held_occurrences = tuple(occurrences)

        if not targets.categories:
            msg = "a constraint has no target"
            raise ValueError(msg)

        if not held_occurrences:
            msg = "a constraint has no occurrence"
            raise ValueError(msg)

        object.__setattr__(self, "_targets", targets)
        object.__setattr__(self, "_occurrences", held_occurrences)
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_clamp_scope", clamp_scope)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_penalty", _checked_penalty(mode, penalty))
        object.__setattr__(self, "_negate", negate)

    @property
    def targets(self) -> TargetSet[T]:
        """Return the target categories and their matching relation."""
        return self._targets

    @property
    def occurrences(self) -> tuple[ConstraintOccurrence, ...]:
        """Return the dataset occurrences in source order."""
        return self._occurrences

    @property
    def scope(self) -> ScopeT:
        """Return the originating scope."""
        return self._scope

    @property
    def clamp_scope(self) -> ClampScope:
        """Return the structural unit constrained by the declaration."""
        return self._clamp_scope

    @property
    def mode(self) -> ClampMode:
        """Return whether the constraint is hard or soft."""
        return self._mode

    @property
    def penalty(self) -> float | None:
        """Return the soft penalty, or None for a hard constraint."""
        return self._penalty

    @property
    def negate(self) -> bool:
        """Return whether the targets are excluded."""
        return self._negate

    def to_spans(
        self,
        batch_item_ids: Sequence[ItemId],
        /,
        *,
        readout: ReadoutId,
    ) -> tuple[BatchLocalConstraint[T, ScopeT], ...]:
        """Return this constraint localized to one batch.

        Parameters
        ----------
        batch_item_ids : Sequence[ItemId]
            Dataset item identifiers in batch order.
        readout : ReadoutId
            Published target mass assigned by the wiring layer.

        Returns
        -------
        tuple[BatchLocalConstraint[T, ScopeT], ...]
            Local constraints grouped by identical span sequences.
        """
        return resolve_spans(
            self._targets,
            self._occurrences,
            self._scope,
            batch_item_ids,
            readout=readout,
            clamp_scope=self._clamp_scope,
            mode=self._mode,
            penalty=self._penalty,
            negate=self._negate,
        )

    def _fields(
        self,
    ) -> tuple[
        TargetSet[T],
        tuple[ConstraintOccurrence, ...],
        ScopeT,
        ClampScope,
        ClampMode,
        float | None,
        bool,
    ]:
        """Return fields in constructor order."""
        return (
            self._targets,
            self._occurrences,
            self._scope,
            self._clamp_scope,
            self._mode,
            self._penalty,
            self._negate,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another constraint has the same fields."""
        if type(other) is not Constraint:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the dataset-global fields."""
        return hash((Constraint, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the field values."""
        return (
            f"Constraint(targets={self._targets!r}, "
            f"occurrences={self._occurrences!r}, scope={self._scope!r}, "
            f"clamp_scope={self._clamp_scope.value!r}, "
            f"mode={self._mode.value!r}, penalty={self._penalty!r}, "
            f"negate={self._negate!r})"
        )


@final
class LogicalFormConstraint[ScopeT: Hashable](FrozenValue):
    """A constraint pinning dataset-global occurrences to one logical form.

    The form is held as declared, and the matching decides which readings count
    as that form: its key is applied where the form is linearized for scoring.
    Localizing to a batch produces the same batch-local value a category
    constraint produces, over the singleton target set of the form.

    Parameters
    ----------
    form : Term
        The required meaning of every occurrence.
    occurrences : Iterable[ConstraintOccurrence]
        Dataset item and span occurrences in source order.
    scope : ScopeT
        Original constraint scope retained as provenance.
    clamp_scope : ClampScope
        Structural unit constrained by the declaration.
    matching : Matching[Term]
        Congruence under which a reading counts as the form.
    mode : ClampMode
        Whether the constraint is hard or carries a finite penalty.
    penalty : float | None
        Positive multiplier for a soft constraint; absent for a hard constraint.
    negate : bool
        Whether the constraint forbids rather than requires the form.

    Raises
    ------
    ValueError
        If the occurrence sequence is empty.
    """

    _noun = "logical-form constraint"

    __slots__ = (
        "_clamp_scope",
        "_form",
        "_mode",
        "_negate",
        "_occurrences",
        "_penalty",
        "_scope",
        "_targets",
    )

    def __init__(
        self,
        form: Term,
        occurrences: Iterable[ConstraintOccurrence],
        scope: ScopeT,
        *,
        matching: Matching[Term],
        clamp_scope: ClampScope = ClampScope.CONSTITUENT,
        mode: ClampMode = ClampMode.HARD,
        penalty: float | None = None,
        negate: bool = False,
    ) -> None:
        held_occurrences = tuple(occurrences)

        if not held_occurrences:
            msg = "a logical-form constraint has no occurrence"
            raise ValueError(msg)

        object.__setattr__(self, "_form", form)
        object.__setattr__(self, "_targets", TargetSet((form,), matching))
        object.__setattr__(self, "_occurrences", held_occurrences)
        object.__setattr__(self, "_scope", scope)
        object.__setattr__(self, "_clamp_scope", clamp_scope)
        object.__setattr__(self, "_mode", mode)
        object.__setattr__(self, "_penalty", _checked_penalty(mode, penalty))
        object.__setattr__(self, "_negate", negate)

    @property
    def form(self) -> Term:
        """Return the required meaning."""
        return self._form

    @property
    def targets(self) -> TargetSet[Term]:
        """Return the form as a singleton target set under its matching."""
        return self._targets

    @property
    def occurrences(self) -> tuple[ConstraintOccurrence, ...]:
        """Return the dataset occurrences in source order."""
        return self._occurrences

    @property
    def scope(self) -> ScopeT:
        """Return the originating scope."""
        return self._scope

    @property
    def clamp_scope(self) -> ClampScope:
        """Return the structural unit constrained by the declaration."""
        return self._clamp_scope

    @property
    def mode(self) -> ClampMode:
        """Return whether the constraint is hard or soft."""
        return self._mode

    @property
    def penalty(self) -> float | None:
        """Return the soft penalty, or None for a hard constraint."""
        return self._penalty

    @property
    def negate(self) -> bool:
        """Return whether the form is forbidden rather than required."""
        return self._negate

    def to_spans(
        self,
        batch_item_ids: Sequence[ItemId],
        /,
        *,
        readout: ReadoutId,
    ) -> tuple[BatchLocalConstraint[Term, ScopeT], ...]:
        """Return this constraint localized to one batch.

        Parameters
        ----------
        batch_item_ids : Sequence[ItemId]
            Dataset item identifiers in batch order.
        readout : ReadoutId
            Published target mass assigned by the wiring layer.

        Returns
        -------
        tuple[BatchLocalConstraint[Term, ScopeT], ...]
            Local constraints grouped by identical span sequences.
        """
        return resolve_spans(
            self._targets,
            self._occurrences,
            self._scope,
            batch_item_ids,
            readout=readout,
            clamp_scope=self._clamp_scope,
            mode=self._mode,
            penalty=self._penalty,
            negate=self._negate,
        )

    def _fields(
        self,
    ) -> tuple[
        TargetSet[Term],
        tuple[ConstraintOccurrence, ...],
        ScopeT,
        ClampScope,
        ClampMode,
        float | None,
        bool,
    ]:
        """Return the identity-bearing fields in constructor order."""
        return (
            self._targets,
            self._occurrences,
            self._scope,
            self._clamp_scope,
            self._mode,
            self._penalty,
            self._negate,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another logical form constraint has the same fields."""
        if type(other) is not LogicalFormConstraint:
            return NotImplemented

        return self._fields() == other._fields()

    @override
    def __hash__(self) -> int:
        """Return a hash over the dataset-global fields."""
        return hash((LogicalFormConstraint, self._fields()))

    @override
    def __repr__(self) -> str:
        """Return the field values."""
        return (
            f"LogicalFormConstraint(form={self._form!r}, "
            f"occurrences={self._occurrences!r}, scope={self._scope!r}, "
            f"clamp_scope={self._clamp_scope.value!r}, "
            f"matching={self._targets.matching!r}, mode={self._mode.value!r}, "
            f"penalty={self._penalty!r}, negate={self._negate!r})"
        )
