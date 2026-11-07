"""Complete participant-group views of one judgment design."""

from collections.abc import Sequence
from types import MappingProxyType
from typing import TYPE_CHECKING, Never, final, override

import torch

from lofi._checks import ensure_name, refuse_reassignment
from lofi.judgment._design import JudgmentDesign

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["GroupedJudgmentSet"]


def _active_groups(groups: Sequence[int], count: int) -> tuple[int, ...]:
    """Return unique in-range group indices in canonical order."""
    held = tuple(groups)

    for group in held:
        if isinstance(group, bool) or not isinstance(group, int):
            msg = f"a participant group index is not an integer: {group!r}"

            raise TypeError(msg)

        if not 0 <= group < count:
            msg = f"there is no participant group {group} among {count} groups"

            raise IndexError(msg)

    canonical = tuple(sorted(held))

    if not canonical:
        msg = "a grouped judgment batch needs at least one participant group"

        raise ValueError(msg)

    if len(frozenset(canonical)) != len(canonical):
        msg = f"a grouped judgment batch repeats a participant group: {held!r}"

        raise ValueError(msg)

    return canonical


@final
class _GroupedJudgmentBatch:
    """A view containing every response of selected participant groups."""

    __slots__ = (
        "_active_groups",
        "_design",
        "_item_ids",
        "_lengths",
        "_response_items",
        "_words",
    )

    def __init__(
        self,
        design: JudgmentDesign,
        item_ids: tuple[str, ...],
        words: tuple[tuple[str, ...], ...],
        response_items: Tensor,
        active_groups: tuple[int, ...],
    ) -> None:
        object.__setattr__(self, "_design", design)
        object.__setattr__(self, "_item_ids", item_ids)
        object.__setattr__(self, "_words", words)
        object.__setattr__(self, "_lengths", tuple(map(len, words)))
        object.__setattr__(self, "_response_items", response_items.clone())
        object.__setattr__(self, "_active_groups", active_groups)

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("grouped judgment batch")

    @property
    def design(self) -> JudgmentDesign:
        """Return the selected response rows with the dataset's global layout."""
        return self._design

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return the selected neural item identifiers in neural item order."""
        return self._item_ids

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return the selected neural items in neural item and word order."""
        return self._words

    @property
    def lengths(self) -> tuple[int, ...]:
        """Return the word count of every selected neural item."""
        return self._lengths

    @property
    def response_items(self) -> Tensor:
        """Return each response row's neural item index at integer precision."""
        return self._response_items.clone()

    @property
    def active_groups(self) -> tuple[int, ...]:
        """Return selected participant group indices in canonical order."""
        return self._active_groups

    def response_rows(self, values: Tensor, /) -> Tensor:
        """Gather one neural-item value for every selected response row.

        Parameters
        ----------
        values : Tensor
            Values whose leading axis follows ``item_ids``.

        Raises
        ------
        ValueError
            If the leading axis does not cover exactly the selected neural items.
        """
        if values.ndim == 0 or values.shape[0] != len(self._item_ids):
            msg = (
                f"response readouts require {len(self._item_ids)} neural item rows, "
                f"not shape {tuple(values.shape)!r}"
            )

            raise ValueError(msg)

        positions = self._response_items.to(device=values.device)

        return values.index_select(0, positions)

    @override
    def __repr__(self) -> str:
        """Return the selected groups, response count and neural item count."""
        return (
            f"_GroupedJudgmentBatch(active_groups={self._active_groups!r}, "
            f"responses={self._design.n_rows}, items={len(self._item_ids)})"
        )


@final
class GroupedJudgmentSet:
    """One complete judgment design joined to its unique neural items.

    Parameters
    ----------
    design : JudgmentDesign
        The complete validated ordinal-response design.
    item_ids : Sequence[str]
        Unique identifiers of the neural items.
    words : Sequence[Sequence[str]]
        Words of each neural item, in ``item_ids`` order.

    Raises
    ------
    TypeError
        If the design is not a judgment design, identifiers or words are not
        sequences, an identifier or word is not a string, or a word sequence is
        text rather than a sequence of words.
    ValueError
        If the design is empty or presents alternatives, identifiers repeat,
        identifiers and words differ in length, a neural item is empty, or the
        neural identifiers do not map exactly onto the design's item table.
    """

    __slots__ = ("_design", "_item_ids", "_item_positions", "_words")

    def __init__(
        self,
        design: JudgmentDesign,
        item_ids: Sequence[str],
        words: Sequence[Sequence[str]],
    ) -> None:
        if not isinstance(design, JudgmentDesign):
            msg = f"a grouped judgment design is not a judgment design: {design!r}"

            raise TypeError(msg)

        if isinstance(item_ids, str) or not isinstance(item_ids, Sequence):
            msg = f"neural item identifiers are not a sequence: {item_ids!r}"

            raise TypeError(msg)

        if isinstance(words, str) or not isinstance(words, Sequence):
            msg = f"neural item words are not a sequence of sequences: {words!r}"

            raise TypeError(msg)

        checked_ids = tuple(
            ensure_name("a neural item identifier", item_id) for item_id in item_ids
        )
        checked_words: list[tuple[str, ...]] = []

        for item in words:
            if isinstance(item, str) or not isinstance(item, Sequence):
                msg = f"a neural item's words are not a sequence: {item!r}"

                raise TypeError(msg)

            checked_words.append(tuple(item))

        held_words = tuple(checked_words)

        if design.n_rows == 0:
            msg = "a grouped judgment set needs at least one response row"

            raise ValueError(msg)

        if design.alternatives_per_row is not None:
            msg = "a grouped judgment set contains items judged on their own"

            raise ValueError(msg)

        if len(checked_ids) != len(held_words):
            msg = (
                f"{len(checked_ids)} neural item identifiers were paired with "
                f"{len(held_words)} word sequences"
            )

            raise ValueError(msg)

        if len(frozenset(checked_ids)) != len(checked_ids):
            msg = f"neural item identifiers must be unique: {checked_ids!r}"

            raise ValueError(msg)

        for item_id, item_words in zip(checked_ids, held_words, strict=True):
            if not item_words:
                msg = f"neural item {item_id!r} has no word"

                raise ValueError(msg)

            for word in item_words:
                ensure_name(f"a word of neural item {item_id!r}", word)

        given = frozenset(checked_ids)
        expected = frozenset(design.item_levels)

        if given != expected:
            missing = tuple(sorted(expected - given))
            extra = tuple(sorted(given - expected))
            msg = (
                "neural item identifiers must map exactly onto the design's item "
                f"table; missing={missing!r}, extra={extra!r}"
            )

            raise ValueError(msg)

        object.__setattr__(self, "_design", design)
        object.__setattr__(self, "_item_ids", checked_ids)
        object.__setattr__(self, "_words", held_words)
        object.__setattr__(
            self,
            "_item_positions",
            MappingProxyType(
                {item_id: position for position, item_id in enumerate(checked_ids)}
            ),
        )

    @override
    def __setattr__[ValueT](self, name: str, value: ValueT) -> Never:
        """Raise AttributeError on field reassignment."""
        del name, value

        refuse_reassignment("grouped judgment set")

    @property
    def design(self) -> JudgmentDesign:
        """Return the complete validated response design."""
        return self._design

    @property
    def item_ids(self) -> tuple[str, ...]:
        """Return all neural item identifiers in neural item order."""
        return self._item_ids

    @property
    def words(self) -> tuple[tuple[str, ...], ...]:
        """Return all neural items in neural item and word order."""
        return self._words

    @property
    def group_count(self) -> int:
        """Return the number of participant groups in the complete design."""
        return len(self._design.participant_levels)

    def select(self, active_groups: Sequence[int], /) -> _GroupedJudgmentBatch:
        """Return every response row belonging to explicit participant groups.

        Parameters
        ----------
        active_groups : Sequence[int]
            Participant group indices to select.
        """
        groups = _active_groups(active_groups, self.group_count)
        participants = self._design.participants
        selected = torch.zeros_like(participants, dtype=torch.bool)

        for group in groups:
            selected.logical_or_(participants == group)

        rows = selected.nonzero(as_tuple=False).flatten()
        observed = tuple(sorted(frozenset(int(value) for value in participants[rows])))

        if observed != groups:
            missing = tuple(group for group in groups if group not in observed)
            msg = f"participant groups have no response rows: {missing!r}"

            raise ValueError(msg)

        selected_design = self._design.select_rows(tuple(map(int, rows)))
        response_source_items = tuple(
            self._item_positions[self._design.item_levels[int(code)]]
            for code in selected_design.items
        )
        source_items = tuple(sorted(frozenset(response_source_items)))
        local = {source: position for position, source in enumerate(source_items)}
        response_items = torch.tensor(
            [local[source] for source in response_source_items], dtype=torch.int64
        )

        return _GroupedJudgmentBatch(
            selected_design,
            tuple(self._item_ids[position] for position in source_items),
            tuple(self._words[position] for position in source_items),
            response_items,
            groups,
        )

    @override
    def __repr__(self) -> str:
        """Return the response, participant-group and neural-item counts."""
        return (
            f"GroupedJudgmentSet(responses={self._design.n_rows}, "
            f"groups={self.group_count}, items={len(self._item_ids)})"
        )
