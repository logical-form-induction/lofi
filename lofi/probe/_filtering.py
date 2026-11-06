"""Score filtering of a counterfactual population, in three arms.

Each candidate sentence carries one declared score, larger meaning more
acceptable to the source that scored it, and every pair generated from a
sentence carries that sentence's score. A filter reports three conditions
together over the same ordered pairs: a primary arm keeping the pairs whose
score reaches the primary inclusive minimum, one sweep arm per declared
threshold, and the unfiltered arm keeping every pair. A filter declares where
its scores came from, and its declaration refuses a primary arm whose scores
were fitted under a grammatical signature, since such a filter is not
independent of the signatures a probe compares.
"""

from __future__ import annotations

import json
from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, Literal, override

from lofi._frozen import FrozenValue

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping, Sequence
    from pathlib import Path

    from lofi.probe._counterfactuals import GeneratedPair, GeneratedPairId

__all__ = [
    "FilterArm",
    "FilterArmKind",
    "FilterReport",
    "filter_counterfactuals",
    "read_sentence_scores",
    "require_score_coverage",
]

type FilterArmKind = Literal["primary", "sweep", "unfiltered"]
"""The role one filter condition plays in a report."""


class FilterArm(FrozenValue):
    """One filter condition and the pairs it keeps.

    Parameters
    ----------
    kind : FilterArmKind
        The condition's role in the report.
    threshold : float | None
        The inclusive minimum score, or None for the unfiltered condition.
    pairs : tuple[GeneratedPair, ...]
        The kept pairs, in their population order.
    total : int
        How many pairs the population holds before filtering.
    """

    __slots__ = ("_kind", "_pairs", "_threshold", "_total")

    _noun = "filter arm"

    def __init__(
        self,
        kind: FilterArmKind,
        threshold: float | None,
        pairs: tuple[GeneratedPair, ...],
        *,
        total: int,
    ) -> None:
        object.__setattr__(self, "_kind", kind)
        object.__setattr__(self, "_threshold", threshold)
        object.__setattr__(self, "_pairs", pairs)
        object.__setattr__(self, "_total", total)

    @property
    def kind(self) -> FilterArmKind:
        """Return the condition's role in the report."""
        return self._kind

    @property
    def threshold(self) -> float | None:
        """Return the inclusive minimum score, if this arm filters."""
        return self._threshold

    @property
    def pairs(self) -> tuple[GeneratedPair, ...]:
        """Return the kept pairs in population order."""
        return self._pairs

    @property
    def ids(self) -> tuple[GeneratedPairId, ...]:
        """Return the kept pair identities in population order."""
        return tuple(pair.id for pair in self._pairs)

    @property
    def count(self) -> int:
        """Return how many pairs this arm keeps."""
        return len(self._pairs)

    @property
    def dropped(self) -> int:
        """Return how many pairs this arm filters out."""
        return self._total - len(self._pairs)

    @override
    def __repr__(self) -> str:
        """Return the kind, threshold and kept pair count."""
        return (
            f"FilterArm({self._kind!r}, threshold={self._threshold!r}, "
            f"count={len(self._pairs)!r}, total={self._total!r})"
        )


class FilterReport(FrozenValue):
    """The primary, threshold-sweep and unfiltered conditions of one filter.

    Parameters
    ----------
    primary : FilterArm
        The condition at the primary inclusive minimum.
    sweep : tuple[FilterArm, ...]
        One condition per declared threshold, in declared order.
    unfiltered : FilterArm
        The condition keeping every pair.
    """

    __slots__ = ("_primary", "_sweep", "_unfiltered")

    _noun = "filter report"

    def __init__(
        self, primary: FilterArm, sweep: tuple[FilterArm, ...], unfiltered: FilterArm
    ) -> None:
        object.__setattr__(self, "_primary", primary)
        object.__setattr__(self, "_sweep", sweep)
        object.__setattr__(self, "_unfiltered", unfiltered)

    @property
    def primary(self) -> FilterArm:
        """Return the condition at the primary inclusive minimum."""
        return self._primary

    @property
    def sweep(self) -> tuple[FilterArm, ...]:
        """Return the threshold conditions in declared order."""
        return self._sweep

    @property
    def unfiltered(self) -> FilterArm:
        """Return the condition keeping every pair."""
        return self._unfiltered

    @property
    def arms(self) -> tuple[FilterArm, ...]:
        """Return every condition in primary, sweep and unfiltered order."""
        return (self._primary, *self._sweep, self._unfiltered)

    @override
    def __repr__(self) -> str:
        """Return the primary threshold and the number of sweep conditions."""
        return (
            f"FilterReport(primary={self._primary.threshold!r}, "
            f"thresholds={len(self._sweep)!r})"
        )


def read_sentence_scores(path: Path, /) -> Mapping[tuple[str, ...], float]:
    """Return the declared score of each candidate sentence in a JSON Lines file.

    Each line is an object ``{"tokens": [...], "score": number}`` naming one
    sentence by its whitespace-separated tokens, as the candidate-sentence file
    spells it, and the finite score of that sentence.

    Parameters
    ----------
    path : Path
        The JSON Lines file of sentence scores.

    Raises
    ------
    ValueError
        If a line is not such an object, names no token, scores a sentence
        twice or gives a score that is not a finite real number.
    """
    scores: dict[tuple[str, ...], float] = {}

    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines()):
        where = f"score line {number + 1} of {path}"
        record = json.loads(line)

        if not isinstance(record, dict) or set(record) != {"tokens", "score"}:
            msg = f"{where} is not an object naming exactly tokens and score"
            raise ValueError(msg)

        tokens = record["tokens"]
        score = record["score"]

        if (
            not isinstance(tokens, list)
            or not tokens
            or not all(isinstance(token, str) and token for token in tokens)
        ):
            msg = f"{where} names no nonempty list of tokens"
            raise ValueError(msg)

        if (
            isinstance(score, bool)
            or not isinstance(score, int | float)
            or not isfinite(score)
        ):
            msg = f"{where} gives a score that is not a finite real number"
            raise ValueError(msg)

        sentence = tuple(tokens)

        if sentence in scores:
            msg = f"{where} scores the sentence {' '.join(sentence)!r} twice"
            raise ValueError(msg)

        scores[sentence] = float(score)

    return MappingProxyType(scores)


def require_score_coverage(
    scores: Mapping[tuple[str, ...], float], sentences: Iterable[Sequence[str]], /
) -> None:
    """Refuse scores that do not cover exactly the candidate sentences.

    Raises
    ------
    ValueError
        If a candidate sentence has no score or a score names no candidate.
    """
    candidates = frozenset(tuple(sentence) for sentence in sentences)
    missing = candidates - frozenset(scores)
    extra = frozenset(scores) - candidates

    if missing or extra:
        msg = (
            f"the declared scores miss {len(missing)} candidate sentences and "
            f"score {len(extra)} sentences that are not candidates"
        )
        raise ValueError(msg)


def filter_counterfactuals(
    pairs: Sequence[GeneratedPair],
    scores: Mapping[tuple[str, ...], float],
    /,
    *,
    primary: float,
    sweep: Sequence[float],
) -> FilterReport:
    """Return the primary, sweep and unfiltered conditions over ordered pairs.

    Parameters
    ----------
    pairs : Sequence[GeneratedPair]
        The generated pairs, in report order.
    scores : Mapping[tuple[str, ...], float]
        The score of every sentence a pair was generated from.
    primary : float
        The primary condition's inclusive minimum.
    sweep : Sequence[float]
        The sweep conditions' inclusive minima, in declared order.

    Raises
    ------
    ValueError
        If a pair's sentence has no score.
    """
    held = tuple(pairs)
    unscored = sum(pair.tokens not in scores for pair in held)

    if unscored:
        msg = f"{unscored} counterfactual pairs come from sentences with no score"
        raise ValueError(msg)

    total = len(held)

    def arm(kind: FilterArmKind, threshold: float) -> FilterArm:
        kept = tuple(pair for pair in held if scores[pair.tokens] >= threshold)

        return FilterArm(kind, threshold, kept, total=total)

    return FilterReport(
        arm("primary", primary),
        tuple(arm("sweep", threshold) for threshold in sweep),
        FilterArm("unfiltered", None, held, total=total),
    )
