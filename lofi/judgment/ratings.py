"""Per-participant ordinal ratings of single expressions, read as design rows."""

from collections import Counter
from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

from lofi.judgment._attributes import NO_FACTORS
from lofi.judgment._design import DesignError, JudgmentRow, RowColumns
from lofi.judgment._sources import JudgmentSkip

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.judgment._attributes import ItemFactors
    from lofi.judgment._ports import RatedItem

__all__ = ["RatingSource"]


@final
class RatingSource:
    """Per-participant ordinal ratings of single expressions, read as design rows.

    Ratings with responses and participants become rows with the declared
    factors, read from the rated item. Missing responses or participants are
    counted as skips.

    Parameters
    ----------
    ratings : Iterable[RatedItem]
        Rated items, as a corpus reader emits them.
    levels : int
        Number of ordered levels every rating's scale has.
    factors : ItemFactors
        The declared factors, read from the item each rating names.

    Raises
    ------
    DesignError
        If a rating's scale is not ordinal, has another number of levels, or
        a declared factor has no item a rating names.
    """

    __slots__ = ("_factors", "_levels", "_rows", "_skips")

    def __init__(
        self,
        ratings: Iterable[RatedItem],
        *,
        levels: int,
        factors: ItemFactors = NO_FACTORS,
    ) -> None:
        rows: list[JudgmentRow] = []
        skips: Counter[str] = Counter()

        for rating in ratings:
            scale = rating.supervision.scale

            if scale.levels is None:
                msg = (
                    f"the item {rating.item.item_id!r} was rated on a scale "
                    f"without ordered levels, and a rating source reads ordinal "
                    f"ratings alone"
                )

                raise DesignError(msg)

            if scale.levels != levels:
                msg = (
                    f"the item {rating.item.item_id!r} was rated on a scale of "
                    f"{scale.levels} levels, and this source is over {levels}"
                )

                raise DesignError(msg)

            value = rating.supervision.value
            participant = rating.supervision.annotator

            if value is None:
                skips[JudgmentSkip.MISSING_RESPONSE] += 1

                continue

            if participant is None:
                skips[JudgmentSkip.MISSING_PARTICIPANT] += 1

                continue

            rows.append(
                JudgmentRow(
                    participant,
                    rating.item.item_id,
                    scale.level_of(value),
                    RowColumns(factors.levels(rating.item.item_id), {}),
                )
            )

        self._levels = levels
        self._factors = factors.names
        self._rows = tuple(rows)
        self._skips: Mapping[str, int] = MappingProxyType(dict(skips))

    @property
    def response_levels(self) -> int:
        """Return the number of ordered response levels every row is over."""
        return self._levels

    @property
    def factors(self) -> tuple[str, ...]:
        """Return the factors every row has a level of, in column order."""
        return self._factors

    @property
    def skips(self) -> Mapping[str, int]:
        """Return how many ratings became no row, by reason."""
        return self._skips

    def rows(self) -> tuple[JudgmentRow, ...]:
        """Return one row per rating that has a response and a participant."""
        return self._rows

    @override
    def __repr__(self) -> str:
        """Return the row count, the level count and the factors."""
        return (
            f"RatingSource(rows={len(self._rows)}, levels={self._levels}, "
            f"factors={self._factors!r}, skipped={sum(self._skips.values())})"
        )
