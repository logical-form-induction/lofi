"""Alignment of induced and reference label inventories."""

import json
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import ensure_name, ensure_non_negative
from lofi._frozen import FrozenValue
from lofi.eval._values import MetricResult, Resampling

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi._types import JsonValue

__all__ = [
    "Alignment",
    "Contingency",
    "Split",
    "SplitLeakError",
    "apply_alignment",
]


type Split = str
"""Defines a named portion of a dataset."""


class SplitLeakError(ValueError):
    """Signals an attempt to score an alignment on its fitting split."""


class Contingency(FrozenValue):
    """Co-occurrence counts of induced and gold labels.

    Parameters
    ----------
    counts : Mapping[tuple[str, str], int]
        Counts keyed by induced and gold labels.
    split : Split
        Dataset split on which counts were accumulated.

    Raises
    ------
    TypeError
        If a label is not text or a count is not an integer.
    ValueError
        If a label is empty, a count is negative, or a key is malformed.
    """

    _noun = "alignment value"

    __slots__ = ("_counts", "_gold_labels", "_induced_labels", "_split", "_total")

    def __init__(self, counts: Mapping[tuple[str, str], int], *, split: Split) -> None:
        checked: dict[tuple[str, str], int] = {}

        for pair, count in counts.items():
            checked[_pair(pair)] = ensure_non_negative(f"the count for {pair!r}", count)

        object.__setattr__(self, "_counts", MappingProxyType(checked))
        object.__setattr__(self, "_split", ensure_name("a dataset split", split))
        object.__setattr__(
            self,
            "_induced_labels",
            tuple(sorted({induced for induced, _ in checked})),
        )
        object.__setattr__(
            self,
            "_gold_labels",
            tuple(sorted({gold for _, gold in checked})),
        )
        object.__setattr__(self, "_total", sum(checked.values()))

    @property
    def counts(self) -> Mapping[tuple[str, str], int]:
        """Return the counts, as a read-only view keyed by induced and gold label."""
        return self._counts

    @property
    def split(self) -> Split:
        """Return the split the counts were accumulated on."""
        return self._split

    @property
    def induced_labels(self) -> tuple[str, ...]:
        """Return the induced labels that occur, in ascending order."""
        return self._induced_labels

    @property
    def gold_labels(self) -> tuple[str, ...]:
        """Return the gold labels that occur, in ascending order."""
        return self._gold_labels

    @property
    def total(self) -> int:
        """Return how many units the table counts in all."""
        return self._total

    def count(self, induced: str, gold: str) -> int:
        """Return how often two labels co-occurred.

        Parameters
        ----------
        induced : str
            Induced label.
        gold : str
            Gold label.

        Returns
        -------
        int
            Co-occurrence count.
        """
        return self._counts.get((induced, gold), 0)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the table in a fixed column order.

        Returns
        -------
        dict[str, JsonValue]
            Serialized contingency.
        """
        return {
            "split": self._split,
            "counts": [
                [induced, gold, self._counts[induced, gold]]
                for induced, gold in sorted(self._counts)
            ],
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value is a table with the same counts and split."""
        if type(other) is not Contingency:
            return NotImplemented

        return self._split == other._split and dict(self._counts) == dict(other._counts)

    @override
    def __hash__(self) -> int:
        """Return a hash over the split and the sorted counts."""
        return hash((Contingency, self._split, tuple(sorted(self._counts.items()))))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this table."""
        return f"Contingency({dict(self._counts)!r}, split={self._split!r})"


class Alignment(FrozenValue):
    """A frozen correspondence from induced labels to gold labels.

    Parameters
    ----------
    mapping : Mapping[str, str]
        Gold label for each aligned induced label.
    contingency : Contingency
        Counts used to fit the correspondence.

    Raises
    ------
    TypeError
        If ``contingency`` or a label has an invalid type.
    ValueError
        If a label is unknown or empty.
    """

    _noun = "alignment value"

    __slots__ = ("_contingency", "_is_injective", "_mapping")

    def __init__(self, mapping: Mapping[str, str], contingency: Contingency) -> None:
        if not isinstance(contingency, Contingency):
            msg = f"an alignment's contingency is not one: {contingency!r}"
            raise TypeError(msg)

        checked: dict[str, str] = {}

        for induced, gold in mapping.items():
            checked[_known("an induced label", induced, contingency.induced_labels)] = (
                _known("a gold label", gold, contingency.gold_labels)
            )

        object.__setattr__(self, "_mapping", MappingProxyType(checked))
        object.__setattr__(self, "_contingency", contingency)
        object.__setattr__(
            self,
            "_is_injective",
            len(set(checked.values())) == len(checked),
        )

    @property
    def mapping(self) -> Mapping[str, str]:
        """Return the correspondence, as a read-only view."""
        return self._mapping

    @property
    def contingency(self) -> Contingency:
        """Return the counts the correspondence was fitted on."""
        return self._contingency

    @property
    def split(self) -> Split:
        """Return the split the correspondence was fitted on."""
        return self._contingency.split

    @property
    def is_injective(self) -> bool:
        """Return whether no two induced labels share a gold label."""
        return self._is_injective

    def apply(self, induced: str) -> str | None:
        """Return the gold label for an induced label.

        Parameters
        ----------
        induced : str
            Induced label.

        Returns
        -------
        str | None
            Aligned gold label, or ``None`` if unaligned.
        """
        return self._mapping.get(induced)

    def to_spec(self) -> dict[str, JsonValue]:
        """Return the alignment in a fixed column order.

        Returns
        -------
        dict[str, JsonValue]
            Serialized alignment.
        """
        return {
            "pairs": [
                [induced, self._mapping[induced]] for induced in sorted(self._mapping)
            ],
            "contingency": self._contingency.to_spec(),
        }

    def serialize(self) -> str:
        """Return the canonical text of this alignment.

        Returns
        -------
        str
            JSON serialization of the alignment.
        """
        return json.dumps(self.to_spec(), sort_keys=False)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another value maps the same labels from the same counts."""
        if type(other) is not Alignment:
            return NotImplemented

        return (
            dict(self._mapping) == dict(other._mapping)
            and self._contingency == other._contingency
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the sorted pairs and the contingency."""
        return hash(
            (Alignment, tuple(sorted(self._mapping.items())), self._contingency)
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this alignment."""
        return f"Alignment({dict(self._mapping)!r}, {self._contingency!r})"


def apply_alignment(
    alignment: Alignment, contingency: Contingency, *, unit: str
) -> MetricResult:
    """Score counts with an alignment fitted on another split.

    Parameters
    ----------
    alignment : Alignment
        Fitted label correspondence.
    contingency : Contingency
        Counts to score.
    unit : str
        Resampling-unit name.

    Returns
    -------
    MetricResult
        Accuracy and alignment counts.

    Raises
    ------
    TypeError
        If an input has an invalid type.
    ValueError
        If ``unit`` is empty.
    """
    if not isinstance(alignment, Alignment):
        msg = f"an applied alignment is not one: {alignment!r}"
        raise TypeError(msg)

    if not isinstance(contingency, Contingency):
        msg = f"a scored contingency is not one: {contingency!r}"
        raise TypeError(msg)

    counted = ensure_name("a scored unit", unit)
    matched = 0
    unaligned = 0

    for (induced, gold), count in contingency.counts.items():
        image = alignment.apply(induced)

        if image is None:
            unaligned += count

        elif image == gold:
            matched += count

    total = contingency.total

    return MetricResult(
        None if total == 0 else matched / total,
        total,
        Resampling(counted),
        components={
            "matched": float(matched),
            "unaligned": float(unaligned),
            "induced_labels": float(len(contingency.induced_labels)),
            "gold_labels": float(len(contingency.gold_labels)),
            "fitted_induced_labels": float(len(alignment.contingency.induced_labels)),
        },
    )


def _pair(pair: tuple[str, str]) -> tuple[str, str]:
    """Return a validated contingency key.

    Parameters
    ----------
    pair : tuple[str, str]
        Induced and gold label pair.

    Returns
    -------
    tuple[str, str]
        Validated label pair.

    Raises
    ------
    TypeError
        If a label is not text.
    ValueError
        If a label is empty or ``pair`` does not contain two values.
    """
    induced, gold = pair

    return (
        ensure_name("an induced label", induced),
        ensure_name("a gold label", gold),
    )


def _known(field: str, label: str, inventory: tuple[str, ...]) -> str:
    """Return a label known to an inventory.

    Parameters
    ----------
    field : str
        Field name used in errors.
    label : str
        Candidate label.
    inventory : tuple[str, ...]
        Allowed labels.

    Returns
    -------
    str
        Validated label.

    Raises
    ------
    ValueError
        If ``label`` does not occur in ``inventory``.
    """
    if label not in inventory:
        msg = f"{field} does not occur in the contingency: {label!r}"
        raise ValueError(msg)

    return label
