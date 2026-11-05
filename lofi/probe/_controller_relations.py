"""Declared row relations for controller diagnostics."""

from typing import TYPE_CHECKING, ClassVar, final, override

import torch

from lofi._checks import ensure_name, ensure_non_negative, ensure_positive
from lofi._frozen import FrozenValue
from lofi.nn.sites import Role
from lofi.probe._diagnose import (
    DiagnosticResult,
    Measurement,
    _ControllerRelationValue,
)
from lofi.probe._scores import consistency, smoothness
from lofi.probe._values import Observation

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi.nn.sites import DTypeClass
    from lofi.probe._diagnose import DiagnosticContext
    from lofi.probe._values import Trace

__all__ = ["ControllerRelation", "ControllerRelations"]

_SEQUENCE_CONSISTENCY = "sequence-consistency"
_SEQUENCE_SMOOTHNESS = "sequence-smoothness"
_SIMILAR_CONSISTENCY = "similar-input-consistency"
_SEQUENCE_COUNT = "sequences"
_SIMILAR_COUNT = "similar-input-groups"
_RELATION_AXIS = "relation-row"


def _row(value: int, rows: int, field: str) -> int:
    """Return a row index within the relation's extent.

    Parameters
    ----------
    value : int
        The index to validate.
    rows : int
        The exclusive upper bound.
    field : str
        What membership the message names.

    Returns
    -------
    int
        The validated index.

    Raises
    ------
    TypeError
        If the index is not an integer or is a boolean.
    ValueError
        If the index is negative or lies beyond the relation's rows.
    """
    checked = ensure_non_negative(field, value)

    if checked >= rows:
        msg = f"{field} is {checked}, beyond the relation's {rows} rows"
        raise ValueError(msg)

    return checked


def _sequence(
    values: Iterable[int], rows: int, position: int, occupied: set[int]
) -> tuple[int, ...]:
    """Return one checked ordered sequence.

    Parameters
    ----------
    values : Iterable[int]
        The row indices in declared order.
    rows : int
        The relation's row extent.
    position : int
        Which sequence is being checked.
    occupied : set[int]
        Rows already used by earlier sequences.

    Returns
    -------
    tuple[int, ...]
        The checked indices in their declared order.

    Raises
    ------
    TypeError
        If an index is not an integer or is a boolean.
    ValueError
        If the sequence is empty, repeats a row, overlaps an earlier sequence, or
        names a row outside the relation.
    """
    held = tuple(
        _row(value, rows, f"row {offset} of sequence {position}")
        for offset, value in enumerate(values)
    )

    if not held:
        msg = f"sequence {position} is empty"
        raise ValueError(msg)

    repeated = len(held) - len(set(held))

    if repeated:
        msg = f"sequence {position} repeats {repeated} rows"
        raise ValueError(msg)

    overlap = occupied.intersection(held)

    if overlap:
        msg = (
            f"sequence {position} overlaps earlier sequences at rows {sorted(overlap)}"
        )
        raise ValueError(msg)

    occupied.update(held)

    return held


def _similar_group(values: Iterable[int], rows: int, position: int) -> frozenset[int]:
    """Return one checked unordered similar-input group.

    Parameters
    ----------
    values : Iterable[int]
        The rows declared similar.
    rows : int
        The relation's row extent.
    position : int
        Which group is being checked.

    Returns
    -------
    frozenset[int]
        The group's distinct row indices.

    Raises
    ------
    TypeError
        If an index is not an integer or is a boolean.
    ValueError
        If a row repeats, the group has fewer than two rows, or an index lies outside
        the relation.
    """
    held = tuple(
        _row(value, rows, f"row {offset} of similar-input group {position}")
        for offset, value in enumerate(values)
    )

    if len(held) != len(set(held)):
        msg = f"similar-input group {position} repeats a row"
        raise ValueError(msg)

    if len(held) < 2:
        msg = f"similar-input group {position} has fewer than two distinct rows"
        raise ValueError(msg)

    return frozenset(held)


@final
class ControllerRelation(FrozenValue, _ControllerRelationValue):
    """Declares ordered sequences and unordered similar-input row groups.

    Parameters
    ----------
    rows : int
        The positive flat-row extent the memberships index.
    sequences : Iterable[Iterable[int]]
        Ordered nonempty row sequences, pairwise disjoint from one another.
    similar_inputs : Iterable[Iterable[int]]
        Unordered groups of at least two distinct rows. Distinct groups may overlap.

    Raises
    ------
    TypeError
        If the extent or an index is not an integer, or is a boolean.
    ValueError
        If the extent is not positive; an index is negative or out of range; a
        sequence is empty, repeated or overlapping; or a similar-input group is too
        small, repeated or duplicates another group.
    """

    __slots__ = ("_rows", "_sequences", "_similar_inputs")

    _noun = "controller relation"

    def __init__(
        self,
        rows: int,
        sequences: Iterable[Iterable[int]] = (),
        similar_inputs: Iterable[Iterable[int]] = (),
    ) -> None:
        extent = ensure_positive("a controller relation's row extent", rows)
        occupied: set[int] = set()
        held_sequences = tuple(
            _sequence(values, extent, position, occupied)
            for position, values in enumerate(sequences)
        )
        held_similar = tuple(
            _similar_group(values, extent, position)
            for position, values in enumerate(similar_inputs)
        )
        seen: set[frozenset[int]] = set()

        for position, group in enumerate(held_similar):
            if group in seen:
                msg = (
                    f"similar-input group {position} repeats an earlier group with "
                    f"rows {sorted(group)}"
                )
                raise ValueError(msg)

            seen.add(group)

        object.__setattr__(self, "_rows", extent)
        object.__setattr__(self, "_sequences", held_sequences)
        object.__setattr__(self, "_similar_inputs", held_similar)

    @property
    def rows(self) -> int:
        """Return the flat-row extent the memberships index."""
        return self._rows

    @property
    def sequences(self) -> tuple[tuple[int, ...], ...]:
        """Return the ordered row sequences in declaration order."""
        return self._sequences

    @property
    def similar_inputs(self) -> tuple[frozenset[int], ...]:
        """Return the unordered similar-input groups in declaration order."""
        return self._similar_inputs

    @property
    def referenced(self) -> frozenset[int]:
        """Return every row named by either relation."""
        return frozenset(
            row for group in (*self._sequences, *self._similar_inputs) for row in group
        )

    def _identity(
        self,
    ) -> tuple[int, tuple[tuple[int, ...], ...], tuple[frozenset[int], ...]]:
        """Return the fields on which two declarations agree."""
        return (self._rows, self._sequences, self._similar_inputs)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another declaration has the same extent and relations."""
        if type(other) is not ControllerRelation:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the row extent and declared relations."""
        return hash((ControllerRelation, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        similar = tuple(tuple(sorted(group)) for group in self._similar_inputs)

        return f"ControllerRelation({self._rows!r}, {self._sequences!r}, {similar!r})"


def _ordered(observation: Observation, rows: tuple[int, ...], over: str) -> Observation:
    """Return selected rows over a fresh relation axis.

    Parameters
    ----------
    observation : Observation
        The controller reading.
    rows : tuple[int, ...]
        Row indices in the order the score must use.
    over : str
        The action-axis name.

    Returns
    -------
    Observation
        The selected rows over the relation and action axes.
    """
    action = observation.position(over)
    row = 1 - action
    index = torch.tensor(rows, dtype=torch.long, device=observation.tensor.device)
    values = observation.tensor.movedim((row, action), (0, 1)).index_select(0, index)
    covering = (
        observation.covering().movedim((row, action), (0, 1)).index_select(0, index)
    )

    return Observation(
        observation.site,
        values,
        axes=(_RELATION_AXIS, over),
        mask=covering,
    )


def _ensure_reading(
    observation: Observation, relation: ControllerRelation, over: str
) -> None:
    """Hold a reading to one relation's rank, extent and live memberships.

    Parameters
    ----------
    observation : Observation
        The controller reading.
    relation : ControllerRelation
        The memberships to apply.
    over : str
        The action-axis name.

    Raises
    ------
    KeyError
        If the action axis was not observed.
    ValueError
        If the reading is not rank two, has another row extent, has no actions, or
        masks a referenced row completely.
    """
    if observation.rank != 2:
        msg = (
            f"the controller relation at {observation.site!r} requires a rank-two "
            f"reading, not rank {observation.rank}"
        )
        raise ValueError(msg)

    action = observation.position(over)
    row = 1 - action
    extent = int(observation.tensor.shape[row])

    if extent != relation.rows:
        msg = (
            f"the controller relation at {observation.site!r} declares "
            f"{relation.rows} rows, but the reading holds {extent}"
        )
        raise ValueError(msg)

    actions = int(observation.tensor.shape[action])

    if actions == 0:
        msg = f"the controller reading at {observation.site!r} holds no actions"
        raise ValueError(msg)

    covering = observation.covering().movedim((row, action), (0, 1))
    live = covering.any(dim=1)
    masked = tuple(
        index for index in sorted(relation.referenced) if not bool(live[index])
    )

    if masked:
        msg = (
            f"the controller relation at {observation.site!r} references masked "
            f"rows {masked}"
        )
        raise ValueError(msg)


def _one_score(
    observation: Observation,
    rows: tuple[int, ...],
    declared: DTypeClass,
    over: str,
    *,
    moving: bool,
) -> float:
    """Return one consistency or smoothness score.

    Parameters
    ----------
    observation : Observation
        The controller reading.
    rows : tuple[int, ...]
        The related rows in score order.
    declared : DTypeClass
        What the published point declares its numbers to mean.
    over : str
        The action-axis name.
    moving : bool
        Whether to score consecutive changes instead of distributional consistency.

    Returns
    -------
    float
        The scalar score.
    """
    selected = _ordered(observation, rows, over)
    scored = (
        smoothness(selected, declared, across=_RELATION_AXIS, over=over)
        if moving
        else consistency(selected, declared, across=_RELATION_AXIS, over=over)
    )

    return float(scored.tensor)


def _mean(values: Iterable[float]) -> float:
    """Return the arithmetic mean, or one for an empty relation family.

    Parameters
    ----------
    values : Iterable[float]
        The per-group scores.

    Returns
    -------
    float
        Their mean, with the neutral perfect score for no declared groups.
    """
    held = tuple(values)

    return 1.0 if not held else sum(held) / len(held)


class ControllerRelations:
    """Measures declared sequence and similar-input agreement at controller sites.

    Parameters
    ----------
    over : str
        The name of the axis the actions run along.
    """

    __slots__ = ("_over",)

    tag: ClassVar[str] = "controller-relations"
    requires: ClassVar[frozenset[Role]] = frozenset({Role.COMBINATOR_LOGITS})
    scores: ClassVar[frozenset[str]] = frozenset(
        {_SEQUENCE_CONSISTENCY, _SEQUENCE_SMOOTHNESS, _SIMILAR_CONSISTENCY}
    )

    def __init__(self, over: str = "action") -> None:
        self._over = ensure_name("the axis controller relations run over", over)

    @property
    def over(self) -> str:
        """Return the name of the axis the actions run along."""
        return self._over

    def run(self, trace: Trace, /, *, context: DiagnosticContext) -> DiagnosticResult:
        """Return sequence and similar-input scores at every decision point read.

        Parameters
        ----------
        trace : Trace
            The readings the scope took.
        context : DiagnosticContext
            Published sites and their caller-declared row relations.

        Returns
        -------
        DiagnosticResult
            Three scalar score measurements and two group counts per point.

        Raises
        ------
        KeyError
            If a reading is unpublished, the action axis is absent, or a read
            controller point has no relation.
        ValueError
            If published controller points were not read or a reading disagrees with
            its relation, numeric declaration or masks.
        """
        taken = dict(context.readings(trace))
        published = context.sites_for(Role.COMBINATOR_LOGITS)
        read = tuple(site for site in published if site in taken)

        if published and not read:
            msg = (
                f"the published controller-relation points {published!r} were not "
                "read, so there are no relations to measure"
            )
            raise ValueError(msg)

        measured: list[Measurement] = []

        for site in read:
            observation = taken[site]
            relation = context.relation_at(site)
            declared = context.spec_at(site).dtype_class
            _ensure_reading(observation, relation, self._over)
            sequence_consistency = _mean(
                _one_score(
                    observation,
                    sequence,
                    declared,
                    self._over,
                    moving=False,
                )
                for sequence in relation.sequences
            )
            sequence_smoothness = _mean(
                _one_score(
                    observation,
                    sequence,
                    declared,
                    self._over,
                    moving=True,
                )
                for sequence in relation.sequences
            )
            similar_consistency = _mean(
                _one_score(
                    observation,
                    tuple(sorted(group)),
                    declared,
                    self._over,
                    moving=False,
                )
                for group in relation.similar_inputs
            )
            measured.extend(
                (
                    Measurement(site, _SEQUENCE_CONSISTENCY, sequence_consistency),
                    Measurement(site, _SEQUENCE_SMOOTHNESS, sequence_smoothness),
                    Measurement(site, _SIMILAR_CONSISTENCY, similar_consistency),
                    Measurement(site, _SEQUENCE_COUNT, float(len(relation.sequences))),
                    Measurement(
                        site, _SIMILAR_COUNT, float(len(relation.similar_inputs))
                    ),
                )
            )

        return DiagnosticResult(self.tag, measured)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this diagnostic."""
        return f"ControllerRelations({self._over!r})"
