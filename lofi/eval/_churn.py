"""Rank churn between two declared scoring conditions."""

from statistics import fmean
from typing import TYPE_CHECKING, Final, Literal, final

from lofi._checks import ensure_finite, ensure_name
from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterable, Sequence

    from lofi._types import JsonValue

__all__ = [
    "ARMS",
    "Arm",
    "ArmMixingError",
    "RankChurnResult",
    "RankedScore",
    "rank_churn",
]

type Arm = Literal["re-scoring", "re-running"]
"""A baseline system's evidence arm."""

ARMS: Final[tuple[Arm, ...]] = ("re-scoring", "re-running")
"""Baseline evidence arms in report order."""


class ArmMixingError(ValueError):
    """Signals an attempt to pool incompatible baseline evidence arms."""


@final
class RankedScore(FrozenValue):
    """One method's score under the two conditions a ranking comparison names."""

    _noun = "ranked score"
    __slots__ = ("_condition", "_method", "_score")

    def __init__(self, method: str, condition: str, score: float) -> None:
        object.__setattr__(self, "_method", ensure_name("a ranked method", method))
        object.__setattr__(
            self, "_condition", ensure_name("a ranking condition", condition)
        )
        object.__setattr__(self, "_score", ensure_finite("a ranked score", score))

    @property
    def method(self) -> str:
        """Return the ranked method."""
        return self._method

    @property
    def condition(self) -> str:
        """Return the ranking condition."""
        return self._condition

    @property
    def score(self) -> float:
        """Return the score used to rank the method."""
        return self._score


@final
class RankChurnResult(FrozenValue):
    """Paired changes between two complete method rankings."""

    _noun = "rank churn"
    __slots__ = ("_changed", "_left", "_mean_absolute_change", "_methods", "_right")

    def __init__(
        self,
        left: str,
        right: str,
        methods: Sequence[str],
        *,
        changed: int,
        mean_absolute_change: float,
    ) -> None:
        object.__setattr__(self, "_left", left)
        object.__setattr__(self, "_right", right)
        object.__setattr__(self, "_methods", tuple(methods))
        object.__setattr__(self, "_changed", changed)
        object.__setattr__(self, "_mean_absolute_change", mean_absolute_change)

    @property
    def left(self) -> str:
        """Return the first ranking condition."""
        return self._left

    @property
    def right(self) -> str:
        """Return the second ranking condition."""
        return self._right

    @property
    def methods(self) -> tuple[str, ...]:
        """Return methods in lexical order."""
        return self._methods

    @property
    def changed(self) -> int:
        """Return how many methods changed rank."""
        return self._changed

    @property
    def mean_absolute_change(self) -> float:
        """Return mean absolute midrank displacement."""
        return self._mean_absolute_change

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the paired ranking result as report columns."""
        return {
            "left": self._left,
            "right": self._right,
            "methods": len(self._methods),
            "changed": self._changed,
            "value": self._mean_absolute_change,
        }


def _midranks(scores: dict[str, float]) -> dict[str, float]:
    """Return descending midranks keyed by method."""
    ordered = sorted(scores.items(), key=lambda item: (-item[1], item[0]))
    ranks: dict[str, float] = {}
    position = 0

    while position < len(ordered):
        end = position + 1

        while end < len(ordered) and ordered[end][1] == ordered[position][1]:
            end += 1

        rank = (position + 1 + end) / 2.0

        for method, _ in ordered[position:end]:
            ranks[method] = rank

        position = end

    return ranks


def rank_churn(
    scores: Iterable[RankedScore],
    *,
    left: str,
    right: str,
) -> RankChurnResult:
    """Return paired midrank displacement between two named conditions."""
    named_left = ensure_name("the left ranking condition", left)
    named_right = ensure_name("the right ranking condition", right)

    if named_left == named_right:
        msg = "rank churn compares two distinct conditions"
        raise ValueError(msg)

    by_condition: dict[str, dict[str, float]] = {named_left: {}, named_right: {}}

    for row in scores:
        if not isinstance(row, RankedScore):
            msg = f"rank churn was handed a non-score: {row!r}"
            raise TypeError(msg)

        if row.condition not in by_condition:
            msg = (
                f"rank churn compares {named_left!r} and {named_right!r}, "
                f"not {row.condition!r}"
            )
            raise ValueError(msg)

        held = by_condition[row.condition]

        if row.method in held:
            msg = f"{row.method!r} is scored twice under {row.condition!r}"
            raise ValueError(msg)

        held[row.method] = row.score

    left_scores = by_condition[named_left]
    right_scores = by_condition[named_right]

    if not left_scores or set(left_scores) != set(right_scores):
        msg = "rank churn requires the same nonempty method set under both conditions"
        raise ValueError(msg)

    methods = tuple(sorted(left_scores))
    left_ranks = _midranks(left_scores)
    right_ranks = _midranks(right_scores)
    changes = tuple(abs(left_ranks[method] - right_ranks[method]) for method in methods)
    return RankChurnResult(
        named_left,
        named_right,
        methods,
        changed=sum(change > 0.0 for change in changes),
        mean_absolute_change=fmean(changes),
    )
