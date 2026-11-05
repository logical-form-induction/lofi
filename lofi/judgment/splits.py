"""K-fold splits that keep each declared unit in one fold.

Each fold reports the fraction of held-out participants absent from its
retained rows. A fold above the coverage threshold requires an explicit
unseen-participant policy.
"""

from __future__ import annotations

import math
import random
from typing import TYPE_CHECKING, NamedTuple, Protocol, final, override

from lofi.judgment._design import DesignError, JudgmentRow

if TYPE_CHECKING:
    from collections.abc import Iterable

    from lofi.spec import HeldOutParticipantPolicy

__all__ = [
    "CoverageError",
    "CoverageGate",
    "CoverageReport",
    "Fold",
    "FoldCoverage",
    "KFold",
    "Split",
    "Splitter",
    "SplitterFactory",
    "kfold",
    "leave_one_group_out",
]

_MINIMUM_FOLDS = 2
"""The fewest folds a split has."""


type FoldCount = int | None
"""A fixed fold count or one fold per level of the held-out unit."""


class Splitter(Protocol):
    """Generates folds that keep every level of one declared unit together."""

    @property
    def unit(self) -> str:
        """Return the response, participant, item or factor held out together."""
        ...

    def split(self, rows: Iterable[JudgmentRow], gate: CoverageGate, /) -> Split:
        """Return the admitted folds over the rows."""
        ...


class SplitterFactory(Protocol):
    """Builds one declared split generator."""

    def __call__(self, unit: str, /, *, folds: FoldCount, seed: int) -> Splitter:
        """Return the splitter selected for a comparison."""
        ...


class CoverageError(ValueError):
    """Reports a fold refused for the participants its retained rows never show.

    Parameters
    ----------
    coverage : FoldCoverage
        Coverage of the refused fold.
    threshold : float
        Largest new-participant fraction admitted without a policy.
    """

    def __init__(self, coverage: FoldCoverage, threshold: float) -> None:
        self.coverage = coverage
        self.threshold = threshold

        super().__init__(
            f"fold {coverage.fold} holds out {coverage.participants} participants "
            f"of whom {coverage.unseen} appear in no retained row, a fraction of "
            f"{coverage.fraction:.3f} above the threshold {threshold:.3f}, and no "
            f"policy for predicting an unseen participant was declared"
        )


@final
class FoldCoverage:
    """The participants one fold holds out, and how many of them are new.

    Parameters
    ----------
    fold : int
        Zero-based index of the fold.
    participants : int
        Number of distinct participants among the held-out rows.
    unseen : int
        Number of those participants that no retained row names.

    Raises
    ------
    ValueError
        If the counts are negative or more participants are unseen than
        held out.
    """

    __slots__ = ("_fold", "_participants", "_unseen")

    def __init__(self, fold: int, *, participants: int, unseen: int) -> None:
        if fold < 0 or participants < 0 or unseen < 0:
            msg = (
                f"a fold's coverage counts are nonnegative: fold {fold}, "
                f"{participants} participants, {unseen} unseen"
            )

            raise ValueError(msg)

        if unseen > participants:
            msg = (
                f"fold {fold} cannot hold out {participants} participants of whom "
                f"{unseen} are unseen"
            )

            raise ValueError(msg)

        self._fold = fold
        self._participants = participants
        self._unseen = unseen

    @property
    def fold(self) -> int:
        """Return the zero-based index of the fold."""
        return self._fold

    @property
    def participants(self) -> int:
        """Return the number of distinct participants among the held-out rows."""
        return self._participants

    @property
    def unseen(self) -> int:
        """Return how many held-out participants no retained row names."""
        return self._unseen

    @property
    def fraction(self) -> float:
        """Return the new-participant fraction, zero for a fold holding out none."""
        if self._participants == 0:
            return 0.0

        return self._unseen / self._participants

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another coverage has the same fold and counts."""
        if type(other) is not FoldCoverage:
            return NotImplemented

        return (self._fold, self._participants, self._unseen) == (
            other._fold,
            other._participants,
            other._unseen,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the fold and the counts."""
        return hash((FoldCoverage, self._fold, self._participants, self._unseen))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return (
            f"FoldCoverage({self._fold!r}, participants={self._participants!r}, "
            f"unseen={self._unseen!r})"
        )


@final
class CoverageReport:
    """The annotator coverage of every fold of one split.

    Parameters
    ----------
    unit : str
        Unit the split holds out together.
    participants : int
        Number of distinct participants across every row.
    folds : Iterable[FoldCoverage]
        Coverage of each fold, in fold order.

    Raises
    ------
    ValueError
        If the fold coverages are not numbered consecutively from zero.
    """

    __slots__ = ("_folds", "_participants", "_unit")

    def __init__(
        self,
        unit: str,
        *,
        participants: int,
        folds: Iterable[FoldCoverage],
    ) -> None:
        coverages = tuple(folds)

        if tuple(coverage.fold for coverage in coverages) != tuple(
            range(len(coverages))
        ):
            msg = (
                f"a report covers folds numbered from zero in order, and was given "
                f"{[coverage.fold for coverage in coverages]!r}"
            )

            raise ValueError(msg)

        self._unit = unit
        self._participants = participants
        self._folds = coverages

    @property
    def unit(self) -> str:
        """Return the unit the split holds out together."""
        return self._unit

    @property
    def participants(self) -> int:
        """Return the number of distinct participants across every row."""
        return self._participants

    @property
    def folds(self) -> tuple[FoldCoverage, ...]:
        """Return the coverage of each fold, in fold order."""
        return self._folds

    @property
    def largest_fraction(self) -> float:
        """Return the largest new-participant fraction of any fold, zero for none."""
        return max((coverage.fraction for coverage in self._folds), default=0.0)

    @override
    def __repr__(self) -> str:
        """Return the unit, the participant count and each fold's fraction."""
        fractions = ", ".join(f"{coverage.fraction:.3f}" for coverage in self._folds)

        return (
            f"CoverageReport(unit={self._unit!r}, participants={self._participants}, "
            f"fractions=[{fractions}])"
        )


@final
class CoverageGate:
    """The admission rule for a fold's annotator coverage.

    A declared policy for predicting a participant the fit never saw applies
    to every fold. The threshold only decides whether a policy is required: a
    fold whose new-participant fraction is at most the threshold is admitted
    without one, and a fold above it is refused unless one is declared.

    Parameters
    ----------
    threshold : float
        Largest new-participant fraction admitted without a policy, between
        zero and one.
    policy : HeldOutParticipantPolicy | None
        How a response from an unseen participant is predicted, or None
        when no such prediction is declared.

    Raises
    ------
    ValueError
        If the threshold is not a number between zero and one.
    """

    __slots__ = ("_policy", "_threshold")

    def __init__(
        self,
        threshold: float,
        *,
        policy: HeldOutParticipantPolicy | None = None,
    ) -> None:
        if not (math.isfinite(threshold) and 0.0 <= threshold <= 1.0):
            msg = f"a coverage threshold lies between zero and one: {threshold!r}"

            raise ValueError(msg)

        self._threshold = float(threshold)
        self._policy = policy

    @property
    def threshold(self) -> float:
        """Return the largest new-participant fraction admitted without a policy."""
        return self._threshold

    @property
    def policy(self) -> HeldOutParticipantPolicy | None:
        """Return the declared policy for an unseen participant, if any."""
        return self._policy

    def admit(self, coverage: FoldCoverage, /) -> HeldOutParticipantPolicy | None:
        """Return the declared policy a fold is admitted under, or None if none is.

        Parameters
        ----------
        coverage : FoldCoverage
            Coverage of the fold.

        Raises
        ------
        CoverageError
            If the fold's fraction exceeds the threshold and no policy is
            declared.
        """
        if self._policy is None and coverage.fraction > self._threshold:
            raise CoverageError(coverage, self._threshold)

        return self._policy

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"CoverageGate({self._threshold!r}, policy={self._policy!r})"


@final
class Fold:
    """One fold: the rows held out, the rows retained, and the coverage.

    Parameters
    ----------
    index : int
        Zero-based index of the fold.
    units : frozenset[str]
        Levels of the unit held out.
    held_out : tuple[JudgmentRow, ...]
        Rows of those levels, in source order.
    retained : tuple[JudgmentRow, ...]
        Every other row, in source order.
    coverage : FoldCoverage
        Participants held out, and how many of them are unseen.
    policy : HeldOutParticipantPolicy | None
        Policy the fold was admitted under, or None when none was declared
        and its coverage needed none.
    """

    __slots__ = ("_coverage", "_held_out", "_index", "_policy", "_retained", "_units")

    def __init__(
        self,
        index: int,
        *,
        units: frozenset[str],
        held_out: tuple[JudgmentRow, ...],
        retained: tuple[JudgmentRow, ...],
        coverage: FoldCoverage,
        policy: HeldOutParticipantPolicy | None,
    ) -> None:
        self._index = index
        self._units = units
        self._held_out = held_out
        self._retained = retained
        self._coverage = coverage
        self._policy = policy

    @property
    def index(self) -> int:
        """Return the zero-based index of the fold."""
        return self._index

    @property
    def units(self) -> frozenset[str]:
        """Return the levels of the unit held out."""
        return self._units

    @property
    def held_out(self) -> tuple[JudgmentRow, ...]:
        """Return the held-out rows, in source order."""
        return self._held_out

    @property
    def retained(self) -> tuple[JudgmentRow, ...]:
        """Return the retained rows, in source order."""
        return self._retained

    @property
    def coverage(self) -> FoldCoverage:
        """Return the participants held out, and how many of them are unseen."""
        return self._coverage

    @property
    def policy(self) -> HeldOutParticipantPolicy | None:
        """Return the policy the fold was admitted under, if one was declared."""
        return self._policy

    @override
    def __repr__(self) -> str:
        """Return the index, the unit count and the row counts."""
        return (
            f"Fold({self._index}, units={len(self._units)}, "
            f"held_out={len(self._held_out)}, retained={len(self._retained)})"
        )


@final
class Split:
    """The folds of one split and their coverage report.

    Parameters
    ----------
    folds : tuple[Fold, ...]
        Folds in index order.
    report : CoverageReport
        Coverage of every fold.
    """

    __slots__ = ("_folds", "_report")

    def __init__(self, folds: tuple[Fold, ...], report: CoverageReport) -> None:
        self._folds = folds
        self._report = report

    @property
    def folds(self) -> tuple[Fold, ...]:
        """Return the folds in index order."""
        return self._folds

    @property
    def report(self) -> CoverageReport:
        """Return the coverage of every fold."""
        return self._report

    @override
    def __repr__(self) -> str:
        """Return the fold count and the report."""
        return f"Split(folds={len(self._folds)}, report={self._report!r})"


class _Part(NamedTuple):
    """One fold's levels, its rows and its coverage, before admission."""

    units: frozenset[str]
    held_out: tuple[JudgmentRow, ...]
    retained: tuple[JudgmentRow, ...]
    coverage: FoldCoverage


def _unit_of(row: JudgmentRow, unit: str, occurrence: int) -> str:
    """Return a row's level of the unit.

    Parameters
    ----------
    row : JudgmentRow
        Row read.
    unit : str
        The response, participant, item, or a factor the row has a level of.
    occurrence : int
        Zero-based row occurrence, used only for response-grain splitting.

    Raises
    ------
    DesignError
        If the unit is a factor the row has no level of.
    """
    if unit == "response":
        return str(occurrence)

    if unit == "participant":
        return row.participant

    if unit == "item":
        return row.item

    level = row.factors.get(unit)

    if level is None:
        msg = (
            f"a split by {unit!r} needs a level of it on every row, and a row has "
            f"the factors {tuple(row.factors)!r}"
        )

        raise DesignError(msg)

    return level


@final
class KFold:
    """A k-fold split generator holding out every row of a unit together.

    Each level of the unit is assigned to exactly one fold, so no level
    straddles two folds. Levels are shuffled by the seed and then dealt,
    largest first, to whichever fold holds the fewest rows, so the folds
    are balanced in rows rather than in levels.

    Parameters
    ----------
    unit : str
        The response occurrence, participant, item, or a declared factor
        whose levels are held out together.
    folds : int
        Number of folds, at least two.
    seed : int
        Seed of the shuffle that orders the levels before they are dealt.

    Raises
    ------
    ValueError
        If the unit is empty or fewer than two folds are asked for.
    """

    __slots__ = ("_folds", "_seed", "_unit")

    def __init__(self, unit: str, *, folds: int, seed: int) -> None:
        if not unit:
            msg = "a split names the unit it holds out"

            raise ValueError(msg)

        if isinstance(folds, bool) or not isinstance(folds, int):
            msg = f"a split fold count is not an integer: {folds!r}"

            raise TypeError(msg)

        if folds < _MINIMUM_FOLDS:
            msg = f"a split has at least {_MINIMUM_FOLDS} folds: {folds}"

            raise ValueError(msg)

        self._unit = unit
        self._folds = folds
        self._seed = seed

    @property
    def unit(self) -> str:
        """Return the unit held out together."""
        return self._unit

    @property
    def folds(self) -> int:
        """Return the number of folds."""
        return self._folds

    @property
    def seed(self) -> int:
        """Return the seed of the shuffle."""
        return self._seed

    def _assign(self, rows: tuple[JudgmentRow, ...]) -> tuple[frozenset[str], ...]:
        """Return the levels of the unit held out by each fold.

        Parameters
        ----------
        rows : tuple[JudgmentRow, ...]
            Rows to split.

        Raises
        ------
        DesignError
            If a row has no level of the unit, or the unit has fewer levels
            than there are folds.
        """
        counts: dict[str, int] = {}

        for occurrence, row in enumerate(rows):
            level = _unit_of(row, self._unit, occurrence)
            counts[level] = counts.get(level, 0) + 1

        if len(counts) < self._folds:
            msg = (
                f"a split by {self._unit!r} into {self._folds} folds needs at least "
                f"{self._folds} levels, and the rows have {len(counts)}"
            )

            raise DesignError(msg)

        levels = list(counts)
        random.Random(self._seed).shuffle(levels)
        levels.sort(key=counts.__getitem__, reverse=True)
        assigned: list[set[str]] = [set() for _ in range(self._folds)]
        sizes = [0] * self._folds

        for level in levels:
            smallest = min(range(self._folds), key=sizes.__getitem__)
            assigned[smallest].add(level)
            sizes[smallest] += counts[level]

        return tuple(frozenset(members) for members in assigned)

    def _partition(self, records: tuple[JudgmentRow, ...]) -> tuple[_Part, ...]:
        """Return each fold's levels, rows and coverage.

        Parameters
        ----------
        records : tuple[JudgmentRow, ...]
            Rows to split.

        Raises
        ------
        DesignError
            If a row has no level of the unit, or the unit has fewer levels
            than there are folds.
        """
        parts = []

        indexed = tuple(enumerate(records))

        for index, units in enumerate(self._assign(records)):
            held_out = tuple(
                row
                for occurrence, row in indexed
                if _unit_of(row, self._unit, occurrence) in units
            )
            retained = tuple(
                row
                for occurrence, row in indexed
                if _unit_of(row, self._unit, occurrence) not in units
            )
            seen = {row.participant for row in retained}
            participants = {row.participant for row in held_out}
            coverage = FoldCoverage(
                index,
                participants=len(participants),
                unseen=len(participants - seen),
            )
            parts.append(_Part(units, held_out, retained, coverage))

        return tuple(parts)

    def _report(
        self, records: tuple[JudgmentRow, ...], parts: tuple[_Part, ...]
    ) -> CoverageReport:
        """Return the coverage report of a partition.

        Parameters
        ----------
        records : tuple[JudgmentRow, ...]
            Rows split.
        parts : tuple[_Part, ...]
            The partition of those rows.
        """
        return CoverageReport(
            self._unit,
            participants=len({row.participant for row in records}),
            folds=(part.coverage for part in parts),
        )

    def coverage(self, rows: Iterable[JudgmentRow], /) -> CoverageReport:
        """Return the annotator coverage of every fold, admitting nothing.

        Parameters
        ----------
        rows : Iterable[JudgmentRow]
            Rows to split.

        Raises
        ------
        DesignError
            If a row has no level of the unit, or the unit has fewer levels
            than there are folds.
        """
        records = tuple(rows)

        return self._report(records, self._partition(records))

    def split(self, rows: Iterable[JudgmentRow], gate: CoverageGate, /) -> Split:
        """Return the folds, each admitted through the coverage gate.

        Parameters
        ----------
        rows : Iterable[JudgmentRow]
            Rows to split.
        gate : CoverageGate
            Admission rule each fold's coverage passes through.

        Raises
        ------
        CoverageError
            If a fold's new-participant fraction exceeds the gate's threshold
            and the gate declares no policy.
        DesignError
            If a row has no level of the unit, or the unit has fewer levels
            than there are folds.
        """
        records = tuple(rows)
        parts = self._partition(records)
        folds = tuple(
            Fold(
                index,
                units=part.units,
                held_out=part.held_out,
                retained=part.retained,
                coverage=part.coverage,
                policy=gate.admit(part.coverage),
            )
            for index, part in enumerate(parts)
        )

        return Split(folds, self._report(records, parts))

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"KFold({self._unit!r}, folds={self._folds!r}, seed={self._seed!r})"


@final
class LeaveOneGroupOut:
    """One fold for every level of a response, participant, item or factor.

    The seed orders levels that have the same row count, so fold numbering is
    reproducible without changing which rows are held out together.
    """

    __slots__ = ("_seed", "_unit")

    def __init__(self, unit: str, *, seed: int) -> None:
        if not unit:
            msg = "a split names the unit it holds out"
            raise ValueError(msg)

        self._unit = unit
        self._seed = seed

    @property
    def unit(self) -> str:
        """Return the unit held out one level at a time."""
        return self._unit

    @property
    def seed(self) -> int:
        """Return the seed ordering the folds."""
        return self._seed

    def _generator(self, rows: tuple[JudgmentRow, ...]) -> KFold:
        """Return a k-fold generator with one fold per observed unit level."""
        levels = {
            _unit_of(row, self._unit, occurrence) for occurrence, row in enumerate(rows)
        }

        return KFold(self._unit, folds=len(levels), seed=self._seed)

    def coverage(self, rows: Iterable[JudgmentRow], /) -> CoverageReport:
        """Return participant coverage with one fold per unit level."""
        records = tuple(rows)
        return self._generator(records).coverage(records)

    def split(self, rows: Iterable[JudgmentRow], gate: CoverageGate, /) -> Split:
        """Return admitted folds with one fold per unit level."""
        records = tuple(rows)
        return self._generator(records).split(records, gate)

    @override
    def __repr__(self) -> str:
        """Return a reconstructing representation."""
        return f"LeaveOneGroupOut({self._unit!r}, seed={self._seed!r})"


def kfold(unit: str, /, *, folds: FoldCount, seed: int) -> Splitter:
    """Build a fixed-count k-fold splitter."""
    if folds is None:
        msg = "the kfold splitter needs an integer fold count, not null"
        raise ValueError(msg)

    return KFold(unit, folds=folds, seed=seed)


def leave_one_group_out(unit: str, /, *, folds: FoldCount, seed: int) -> Splitter:
    """Build a splitter holding out one level of the declared unit per fold."""
    if folds is not None:
        msg = "the leave-one-group-out splitter declares its fold count as null"
        raise ValueError(msg)

    return LeaveOneGroupOut(unit, seed=seed)
