"""Alignment stability across seeds."""

from itertools import combinations
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_non_negative
from lofi.eval._alignment import Alignment
from lofi.eval._clustering import _entropy as _marginal_entropy
from lofi.eval._values import MINIMUM_SEEDS, MetricResult, Resampling
from lofi.stats import Bootstrap, Interval

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from lofi._types import JsonValue

__all__ = ["AlignmentStability", "alignment_stability"]


type _Row = tuple[str, ...]
"""Defines one seed's recovered map, as the gold image of each shared induced label."""


class AlignmentStability:
    """Stores agreement among maps recovered across seeds.

    Attributes
    ----------
    agreement : MetricResult
        Mean pairwise agreement and interval.
    entropy : Mapping[str, float]
        Recovered-image entropy by shared label.
    labels : tuple[str, ...]
        Labels assigned by every seed.
    dropped : tuple[str, ...]
        Labels unassigned by at least one seed.
    seeds : int
        Number of recovered maps.

    Parameters
    ----------
    agreement : MetricResult
        Mean pairwise agreement and interval.
    entropy : Mapping[str, float]
        Recovered-image entropy by shared label.
    labels : Sequence[str]
        Labels assigned by every seed.
    dropped : Sequence[str]
        Labels unassigned by at least one seed.
    seeds : int
        Number of recovered maps.

    Raises
    ------
    TypeError
        If ``agreement`` or ``seeds`` has an invalid type.
    ValueError
        If counts or label inventories are inconsistent.
    """

    __slots__ = ("_agreement", "_dropped", "_entropy", "_labels", "_seeds")

    def __init__(
        self,
        agreement: MetricResult,
        entropy: Mapping[str, float],
        *,
        labels: Sequence[str],
        dropped: Sequence[str],
        seeds: int,
    ) -> None:
        if not isinstance(agreement, MetricResult):
            msg = f"a stability's agreement is not a result: {agreement!r}"
            raise TypeError(msg)

        self._agreement = agreement
        self._labels = tuple(labels)
        self._dropped = tuple(dropped)
        self._seeds = ensure_non_negative("a stability's seed count", seeds)
        self._entropy: Mapping[str, float] = MappingProxyType(dict(entropy))

        shared = set(self._labels)
        both = shared & set(self._dropped)

        if both:
            msg = f"a stability calls {sorted(both)!r} both shared and dropped"
            raise ValueError(msg)

        if set(self._entropy) != shared:
            msg = (
                f"a stability reports entropies for {sorted(self._entropy)!r} "
                f"over the shared labels {sorted(shared)!r}"
            )
            raise ValueError(msg)

    @property
    def agreement(self) -> MetricResult:
        """Return the mean pairwise agreement, with the interval around it."""
        return self._agreement

    @property
    def entropy(self) -> Mapping[str, float]:
        """Return the entropy of each shared label's recovered image, in nats."""
        return self._entropy

    @property
    def labels(self) -> tuple[str, ...]:
        """Return the induced labels every seed assigned."""
        return self._labels

    @property
    def dropped(self) -> tuple[str, ...]:
        """Return the induced labels some seed left unassigned."""
        return self._dropped

    @property
    def seeds(self) -> int:
        """Return the map count."""
        return self._seeds

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the stability as report columns.

        Returns
        -------
        dict[str, JsonValue]
            Serialized stability result.
        """
        interval = self._agreement.ci
        entropy: dict[str, JsonValue] = {
            label: self._entropy[label] for label in self._labels
        }

        return {
            "seeds": self._seeds,
            "agreement": self._agreement.value,
            "agreement_ci_low": None if interval is None else interval.low,
            "agreement_ci_high": None if interval is None else interval.high,
            "agreement_ci_level": None if interval is None else interval.level,
            "labels": list(self._labels),
            "dropped": list(self._dropped),
            "entropy": entropy,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a stability with the same five fields."""
        if type(other) is not AlignmentStability:
            return NotImplemented

        return (
            self._agreement == other._agreement
            and dict(self._entropy) == dict(other._entropy)
            and self._labels == other._labels
            and self._dropped == other._dropped
            and self._seeds == other._seeds
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the agreement, the entropies and the inventories."""
        return hash(
            (
                AlignmentStability,
                self._agreement,
                tuple(sorted(self._entropy.items())),
                self._labels,
                self._dropped,
                self._seeds,
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this stability."""
        return (
            f"AlignmentStability({self._agreement!r}, {dict(self._entropy)!r}, "
            f"labels={list(self._labels)!r}, dropped={list(self._dropped)!r}, "
            f"seeds={self._seeds!r})"
        )


def alignment_stability(
    alignments: Sequence[Alignment], *, bootstrap: Bootstrap
) -> AlignmentStability:
    """Measure agreement among maps recovered across seeds.

    Parameters
    ----------
    alignments : Sequence[Alignment]
        Fitted alignments from distinct seeds.
    bootstrap : Bootstrap
        Seed-level bootstrap configuration.

    Returns
    -------
    AlignmentStability
        Pairwise agreement and recovered-image entropies.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If there are too few alignments or no shared assigned label.
    """
    if not isinstance(bootstrap, Bootstrap):
        msg = f"a stability's bootstrap is not one: {bootstrap!r}"
        raise TypeError(msg)

    rows, labels, dropped = _rows(alignments)
    drawn = bootstrap.interval(rows, _agreement)

    return AlignmentStability(
        MetricResult(
            drawn.observed,
            len(rows),
            Resampling(bootstrap.unit, bootstrap.seed),
            components={
                "labels": float(len(labels)),
                "dropped": float(len(dropped)),
                "pairs": float(len(rows) * (len(rows) - 1) // 2),
            },
            ci=Interval(drawn.bca.low, drawn.bca.high, drawn.bca.level),
        ),
        {
            label: _entropy([row[position] for row in rows])
            for position, label in enumerate(labels)
        },
        labels=labels,
        dropped=dropped,
        seeds=len(rows),
    )


def _rows(
    alignments: Sequence[Alignment],
) -> tuple[tuple[_Row, ...], tuple[str, ...], tuple[str, ...]]:
    """Return seed maps over the shared inventory.

    Parameters
    ----------
    alignments : Sequence[Alignment]
        Fitted alignments from distinct seeds.

    Returns
    -------
    tuple[tuple[_Row, ...], tuple[str, ...], tuple[str, ...]]
        Seed maps, shared labels, and dropped labels.

    Raises
    ------
    TypeError
        If an input is not an alignment.
    ValueError
        If there are too few alignments or no shared assigned label.
    """
    for entry in alignments:
        if not isinstance(entry, Alignment):
            msg = f"a recovered alignment is not one: {entry!r}"
            raise TypeError(msg)

    if len(alignments) < MINIMUM_SEEDS:
        msg = (
            f"a stability over seeds needs at least {MINIMUM_SEEDS} of them and "
            f"was handed {len(alignments)}"
        )
        raise ValueError(msg)

    seen = {label for entry in alignments for label in entry.mapping}
    labels = tuple(
        sorted(
            label
            for label in seen
            if all(entry.apply(label) is not None for entry in alignments)
        )
    )
    dropped = tuple(sorted(seen - set(labels)))

    if not labels:
        msg = (
            f"no induced label was assigned by all {len(alignments)} seeds, so "
            f"there is nothing to compare their maps on"
        )
        raise ValueError(msg)

    rows = tuple(
        tuple(_assigned(entry, label) for label in labels) for entry in alignments
    )

    return (rows, labels, dropped)


def _assigned(alignment: Alignment, label: str) -> str:
    """Return the assigned gold label.

    Parameters
    ----------
    alignment : Alignment
        Fitted alignment.
    label : str
        Shared induced label.

    Returns
    -------
    str
        Assigned gold label.

    Raises
    ------
    ValueError
        If ``label`` is unassigned.
    """
    image = alignment.apply(label)

    if image is None:
        msg = f"an alignment left {label!r} unassigned after it was called shared"
        raise ValueError(msg)

    return image


def _agreement(rows: Sequence[_Row]) -> float:
    """Return mean pairwise agreement among recovered maps.

    Parameters
    ----------
    rows : Sequence[_Row]
        Recovered maps over a shared inventory.

    Returns
    -------
    float
        Mean pairwise agreement.
    """
    pairs = list(combinations(range(len(rows)), 2))

    if not pairs:
        return 1.0

    labels = len(rows[0])
    total = 0.0

    for left, right in pairs:
        matched = sum(
            1
            for position in range(labels)
            if rows[left][position] == rows[right][position]
        )
        total += matched / labels

    return total / len(pairs)


def _entropy(images: Sequence[str]) -> float:
    """Return recovered-image entropy for one induced label.

    Parameters
    ----------
    images : Sequence[str]
        Gold-label images across seeds.

    Returns
    -------
    float
        Entropy in nats.
    """
    counts: dict[str, int] = {}

    for image in images:
        counts[image] = counts.get(image, 0) + 1

    return _marginal_entropy(tuple(counts.values()), len(images))
