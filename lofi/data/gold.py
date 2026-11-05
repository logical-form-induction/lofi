"""Declared aggregation of repeated corpus responses."""

from collections import Counter
from typing import TYPE_CHECKING, Final, Protocol, final

from lofi._frozen import FrozenValue
from lofi.data._supervision import Example, Relational
from lofi.registry import ComponentPlugin
from lofi.spec.gold import MajorityGoldSettings, MajorityGoldSpec

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi._types import ItemId
    from lofi.data import Supervision

__all__ = ["PLUGIN", "GoldAggregator", "MajorityGold"]


class GoldAggregator(Protocol):
    """Reduce repeated responses without changing unrelated examples."""

    def __call__(
        self, examples: Sequence[Example[Supervision[str]]], /
    ) -> tuple[Example[Supervision[str]], ...]:
        """Return examples with repeated categorical relations reduced."""
        ...


@final
class MajorityGold(FrozenValue):
    """Reduce each item pair to its unique most frequent relation."""

    __slots__ = ()
    _noun = "majority gold aggregator"

    def __call__(
        self, examples: Sequence[Example[Supervision[str]]], /
    ) -> tuple[Example[Supervision[str]], ...]:
        """Return one anonymous majority relation per repeated item pair."""
        counts: dict[tuple[ItemId, ItemId], Counter[str]] = {}
        first: dict[tuple[ItemId, ItemId], Example[Supervision[str]]] = {}

        for example in examples:
            response = example.supervision
            if not isinstance(response, Relational):
                continue
            key = (example.item.item_id, response.other)
            counts.setdefault(key, Counter())[response.relation] += 1
            first.setdefault(key, example)

        selected: dict[tuple[ItemId, ItemId], Example[Supervision[str]]] = {}
        for key, frequencies in counts.items():
            greatest = max(frequencies.values())
            winners = tuple(
                label for label, count in frequencies.items() if count == greatest
            )
            if len(winners) != 1:
                msg = f"responses for item pair {key!r} have no unique majority"
                raise ValueError(msg)
            example = first[key]
            response = example.supervision
            if not isinstance(response, Relational):
                msg = "a majority relation group lost its relational response"
                raise TypeError(msg)
            selected[key] = Example(
                example.item,
                Relational(response.other, winners[0]),
            )

        emitted: set[tuple[ItemId, ItemId]] = set()
        result: list[Example[Supervision[str]]] = []
        for example in examples:
            response = example.supervision
            if not isinstance(response, Relational):
                result.append(example)
                continue
            key = (example.item.item_id, response.other)
            if key not in emitted:
                result.append(selected[key])
                emitted.add(key)
        return tuple(result)


def _build(settings: MajorityGoldSettings, context: None, /) -> MajorityGold:
    """Build parameter-free majority aggregation."""
    del settings, context
    return MajorityGold()


PLUGIN: Final = ComponentPlugin(spec=MajorityGoldSpec, build=_build)
"""The majority-gold plugin."""
