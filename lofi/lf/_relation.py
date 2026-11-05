"""Inference labels and their two-question logical relation."""

from enum import StrEnum
from types import MappingProxyType
from typing import TYPE_CHECKING, Final, final, override

from lofi._checks import ensure_name
from lofi._frozen import FrozenValue
from lofi.lf._verdict import Abstain, Graded, Proved, Refuted

if TYPE_CHECKING:
    from collections.abc import Mapping

    from lofi.lf._verdict import Verdict

__all__ = [
    "CLASSICAL",
    "STRICT",
    "Decided",
    "InferenceCell",
    "InferenceLabel",
    "LabelMap",
    "LabelStatus",
    "Reward",
]


class InferenceLabel(StrEnum):
    """A label exposed by inference data and predictions."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    NON_ENTAILS = "non-entails"


class InferenceCell(StrEnum):
    """One joint outcome of the positive and negative entailment questions."""

    ENTAILS = "entails"
    CONTRADICTS = "contradicts"
    NEUTRAL = "neutral"
    DEGENERATE = "degenerate"


class LabelStatus(StrEnum):
    """Whether available evidence settles a declared inference label."""

    HOLDS = "holds"
    FAILS = "fails"
    OPEN = "open"


def _question_values(verdict: Verdict, /) -> frozenset[bool]:
    """Return the truth values consistent with one question's verdict."""
    if isinstance(verdict, Proved):
        return frozenset({True})

    if isinstance(verdict, Refuted):
        return frozenset({False})

    if isinstance(verdict, Abstain):
        return frozenset({False, True})

    if isinstance(verdict, Graded):
        if verdict.degree == 1.0:
            return frozenset({True})

        if verdict.degree == 0.0:
            return frozenset({False})

        return frozenset({False, True})

    msg = f"an inference verdict is a Verdict, not {verdict!r}"
    raise TypeError(msg)


_CELL: Final[Mapping[tuple[bool, bool], InferenceCell]] = MappingProxyType(
    {
        (True, False): InferenceCell.ENTAILS,
        (False, True): InferenceCell.CONTRADICTS,
        (False, False): InferenceCell.NEUTRAL,
        (True, True): InferenceCell.DEGENERATE,
    }
)


@final
class Decided(FrozenValue):
    """The relation cells still possible under the verdicts obtained."""

    _noun = "decided inference relation"
    __slots__ = ("_cells",)

    def __init__(self, cells: frozenset[InferenceCell]) -> None:
        if not cells:
            msg = "a decided inference relation retains at least one cell"
            raise ValueError(msg)

        if any(not isinstance(cell, InferenceCell) for cell in cells):
            msg = f"relation cells are InferenceCell values, not {cells!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_cells", cells)

    @classmethod
    def all(cls) -> Decided:
        """Return the relation with no question settled."""
        return cls(frozenset(InferenceCell))

    @classmethod
    def from_verdicts(cls, positive: Verdict, negative: Verdict, /) -> Decided:
        """Return cells consistent with entailment of a goal and its negation."""
        cells = frozenset(
            _CELL[(entailed, contradicted)]
            for entailed in _question_values(positive)
            for contradicted in _question_values(negative)
        )
        return cls(cells)

    @property
    def cells(self) -> frozenset[InferenceCell]:
        """Return the relation cells consistent with the evidence."""
        return self._cells

    @property
    def complete(self) -> bool:
        """Return whether the evidence selects exactly one relation cell."""
        return len(self._cells) == 1

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another relation retains the same cells."""
        return type(other) is Decided and self._cells == other._cells

    @override
    def __hash__(self) -> int:
        """Return a hash over the retained cells."""
        return hash((Decided, self._cells))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this relation."""
        return f"Decided({self._cells!r})"


@final
class LabelMap(FrozenValue):
    """A declared map from inference labels to compatible relation cells."""

    _noun = "inference label map"
    __slots__ = ("_cells", "_name")

    def __init__(
        self,
        name: str,
        cells: Mapping[InferenceLabel, frozenset[InferenceCell]],
    ) -> None:
        if not cells:
            msg = "an inference label map names at least one label"
            raise ValueError(msg)

        if any(not isinstance(label, InferenceLabel) for label in cells):
            msg = "an inference label map contains only InferenceLabel keys"
            raise TypeError(msg)

        if any(not held for held in cells.values()):
            msg = "an inference label maps to at least one relation cell"
            raise ValueError(msg)

        if any(
            not isinstance(cell, InferenceCell)
            for held in cells.values()
            for cell in held
        ):
            msg = "an inference label map contains only InferenceCell values"
            raise TypeError(msg)

        object.__setattr__(self, "_name", ensure_name("an inference label map", name))
        object.__setattr__(self, "_cells", MappingProxyType(dict(cells)))

    @property
    def name(self) -> str:
        """Return the declared map name."""
        return self._name

    @property
    def cells(self) -> Mapping[InferenceLabel, frozenset[InferenceCell]]:
        """Return each label's compatible relation cells."""
        return self._cells

    @property
    def labels(self) -> tuple[InferenceLabel, ...]:
        """Return the mapped labels in declared order."""
        return tuple(self._cells)

    def status(self, label: InferenceLabel, decided: Decided, /) -> LabelStatus:
        """Return whether a label holds, fails, or remains open."""
        if not isinstance(label, InferenceLabel):
            msg = f"an inference label is an InferenceLabel, not {label!r}"
            raise TypeError(msg)

        if label not in self._cells:
            msg = f"the inference label {label!r} is not mapped by {self._name!r}"
            raise ValueError(msg)

        if not isinstance(decided, Decided):
            msg = f"label status requires Decided evidence, not {decided!r}"
            raise TypeError(msg)

        compatible = self._cells[label]

        if decided.cells <= compatible:
            return LabelStatus.HOLDS

        if decided.cells.isdisjoint(compatible):
            return LabelStatus.FAILS

        return LabelStatus.OPEN

    def label(self, decided: Decided, /) -> InferenceLabel | None:
        """Return the first label the evidence settles, or None while none holds."""
        return next(
            (
                label
                for label in self._cells
                if self.status(label, decided) is LabelStatus.HOLDS
            ),
            None,
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another map has the same name and cell relation."""
        return (
            type(other) is LabelMap
            and self._name == other._name
            and self._cells == other._cells
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the declared mapping."""
        return hash(
            (
                LabelMap,
                self._name,
                tuple(self._cells.items()),
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this mapping."""
        return f"LabelMap({self._name!r}, {dict(self._cells)!r})"


CLASSICAL: Final = LabelMap(
    "classical",
    {
        InferenceLabel.ENTAILS: frozenset(
            {InferenceCell.ENTAILS, InferenceCell.DEGENERATE}
        ),
        InferenceLabel.CONTRADICTS: frozenset(
            {InferenceCell.CONTRADICTS, InferenceCell.DEGENERATE}
        ),
        InferenceLabel.NEUTRAL: frozenset({InferenceCell.NEUTRAL}),
    },
)
"""Classical consequence, under which inconsistency entails every goal."""

STRICT: Final = LabelMap(
    "strict",
    {
        InferenceLabel.ENTAILS: frozenset({InferenceCell.ENTAILS}),
        InferenceLabel.CONTRADICTS: frozenset({InferenceCell.CONTRADICTS}),
        InferenceLabel.NEUTRAL: frozenset({InferenceCell.NEUTRAL}),
    },
)
"""Cell-exact labels, under which degenerate evidence supports no label."""


@final
class Reward(FrozenValue):
    """A label reward derived only from the relation evidence available."""

    _noun = "inference reward"
    __slots__ = ("_decided", "_label", "_value")

    def __init__(
        self,
        label: InferenceLabel,
        decided: Decided,
        value: float | None,
    ) -> None:
        if not isinstance(label, InferenceLabel):
            msg = f"an inference reward label is an InferenceLabel, not {label!r}"
            raise TypeError(msg)

        if not isinstance(decided, Decided):
            msg = f"an inference reward requires Decided evidence, not {decided!r}"
            raise TypeError(msg)

        if value not in (None, 0.0, 1.0):
            msg = f"a logical inference reward is zero, one, or open, not {value!r}"
            raise ValueError(msg)

        object.__setattr__(self, "_label", label)
        object.__setattr__(self, "_decided", decided)
        object.__setattr__(self, "_value", value)

    @classmethod
    def from_decided(
        cls,
        label: InferenceLabel,
        decided: Decided,
        /,
        *,
        mapping: LabelMap,
    ) -> Reward:
        """Return one for holds, zero for failure, and none while open."""
        status = mapping.status(label, decided)
        value = (
            1.0
            if status is LabelStatus.HOLDS
            else 0.0
            if status is LabelStatus.FAILS
            else None
        )
        return cls(label, decided, value)

    @property
    def label(self) -> InferenceLabel:
        """Return the label whose status is rewarded."""
        return self._label

    @property
    def decided(self) -> Decided:
        """Return the evidence the reward was derived from."""
        return self._decided

    @property
    def value(self) -> float | None:
        """Return one, zero, or None where the label remains open."""
        return self._value

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another reward has the same label, evidence, and value."""
        return (
            type(other) is Reward
            and self._label is other._label
            and self._decided == other._decided
            and self._value == other._value
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the label, evidence, and value."""
        return hash((Reward, self._label, self._decided, self._value))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this reward."""
        return f"Reward({self._label!r}, {self._decided!r}, {self._value!r})"
