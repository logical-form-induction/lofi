"""Relations asserted between paired expressions, read as design rows."""

from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name
from lofi.judgment._attributes import NO_FACTORS
from lofi.judgment._design import DesignError, JudgmentRow, RowColumns
from lofi.judgment._sources import (
    AnonymousParticipantPolicy as _AnonymousParticipantPolicy,
)
from lofi.judgment._sources import JudgmentSkip

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence

    from lofi.judgment._attributes import ItemFactors
    from lofi.judgment._ports import RelatedItem
    from lofi.judgment._sources import AnonymousParticipantPolicy

__all__ = ["RelationSource", "related_item_key"]

_JOIN: Final = " -> "
"""The separator between the item identities of a related-item key."""


def related_item_key(items: Sequence[str], /) -> str:
    """Return the row key of an ordered tuple of related items.

    Parameters
    ----------
    items : Sequence[str]
        Identities of the related items, in the order the relation runs.

    Raises
    ------
    ValueError
        If fewer than two identities are given or one is empty.
    """
    held = tuple(ensure_name("a related item identity", item) for item in items)
    if len(held) < 2:
        msg = f"a related-item key relates at least two items: {held!r}"
        raise ValueError(msg)
    return _JOIN.join(held)


@final
class RelationSource:
    """Relations asserted between paired expressions, read as design rows.

    The declared inventory determines response-level order. Annotated
    relations become rows keyed by the related-item key of the item and the
    item it is related to, with the declared factors read from the item
    record under that key; relations
    without an annotator are counted as skips unless the singleton policy is
    declared, in which case each occurrence receives a distinct participant.

    Parameters
    ----------
    relations : Iterable[RelatedItem]
        Related items, as a corpus reader emits them.
    inventory : Sequence[str]
        The relation names in response-level order, at least two.
    factors : ItemFactors
        The declared factors, read from the item record each related-item key
        names.
    participant_policy : AnonymousParticipantPolicy
        Policy for relations that name no annotator.

    Raises
    ------
    DesignError
        If the inventory has fewer than two relations or repeats one, a
        relation is outside the inventory, or a declared factor has no item
        record under a related-item key.
    TypeError
        If a relation name is not a string.
    ValueError
        If a relation name is empty.
    """

    __slots__ = (
        "_factors",
        "_inventory",
        "_participant_policy",
        "_rows",
        "_singleton_participants",
        "_skips",
    )

    def __init__(
        self,
        relations: Iterable[RelatedItem],
        *,
        inventory: Sequence[str],
        factors: ItemFactors = NO_FACTORS,
        participant_policy: AnonymousParticipantPolicy = (
            _AnonymousParticipantPolicy.SKIP
        ),
    ) -> None:
        if not isinstance(participant_policy, _AnonymousParticipantPolicy):
            msg = (
                "a relation source's participant policy is not an "
                f"AnonymousParticipantPolicy: {participant_policy!r}"
            )

            raise TypeError(msg)

        names = tuple(ensure_name("a relation name", name) for name in inventory)

        if len(names) < 2:
            msg = f"a relation inventory has at least two relations: {names!r}"

            raise DesignError(msg)

        if len(frozenset(names)) != len(names):
            msg = f"a relation inventory repeats a relation: {names!r}"

            raise DesignError(msg)

        levels = {name: level for level, name in enumerate(names)}
        rows: list[JudgmentRow] = []
        skips: Counter[str] = Counter()
        held = tuple(relations)
        occupied = {
            participant
            for related in held
            if (participant := related.supervision.annotator) is not None
        }
        singleton_participants = 0

        for related in held:
            relation = related.supervision
            level = levels.get(relation.relation)

            if level is None:
                msg = (
                    f"the relation {relation.relation!r} is outside the inventory "
                    f"{names!r}"
                )

                raise DesignError(msg)

            participant = relation.annotator

            if participant is None:
                if participant_policy is _AnonymousParticipantPolicy.SKIP:
                    skips[JudgmentSkip.MISSING_PARTICIPANT] += 1

                    continue

                base = f"anonymous-{singleton_participants}"
                participant = base
                collision = 0

                while participant in occupied:
                    collision += 1
                    participant = f"{base}-{collision}"

                occupied.add(participant)
                singleton_participants += 1

            key = related_item_key((related.item.item_id, relation.other))
            columns = RowColumns(factors.levels(key), {})
            rows.append(JudgmentRow(participant, key, level, columns))

        self._inventory = names
        self._factors = factors.names
        self._participant_policy = participant_policy
        self._rows = tuple(rows)
        self._singleton_participants = singleton_participants
        self._skips: Mapping[str, int] = MappingProxyType(dict(skips))

    @property
    def inventory(self) -> tuple[str, ...]:
        """Return the relation names in response-level order."""
        return self._inventory

    @property
    def response_levels(self) -> int:
        """Return the number of relations in the inventory."""
        return len(self._inventory)

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factors every row has a level of, in column order."""
        return self._factors

    @property
    def participant_policy(self) -> AnonymousParticipantPolicy:
        """Return the declared policy for relations without an annotator."""
        return self._participant_policy

    @property
    def singleton_participants(self) -> int:
        """Return the number of pseudo-participants created by the policy."""
        return self._singleton_participants

    @property
    def skips(self) -> Mapping[str, int]:
        """Return how many relations became no row, by reason."""
        return self._skips

    def rows(self) -> tuple[JudgmentRow, ...]:
        """Return one row per relation admitted under the participant policy."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the row count, declarations and disposition counts."""
        return (
            f"RelationSource(rows={len(self._rows)}, inventory={self._inventory!r}, "
            f"factors={self._factors!r}, participant_policy="
            f"{self._participant_policy.value!r}, singleton_participants="
            f"{self._singleton_participants}, skipped={sum(self._skips.values())})"
        )
