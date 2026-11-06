"""Sensitivity of ablation conclusions to intervention and metric semantics."""

from enum import StrEnum
from typing import TYPE_CHECKING, Final, Literal, override

from lofi._checks import ensure_finite
from lofi._frozen import FrozenValue
from lofi.probe._interventions import MeanFrom, Resample, Zero

if TYPE_CHECKING:
    from collections.abc import Callable, Iterable

    from lofi._types import JsonValue
    from lofi.probe._interventions import Intervention
    from lofi.probe._values import Trace

__all__ = [
    "AblationMetric",
    "AblationSemantics",
    "AblationSensitivityCell",
    "AblationSensitivityReport",
    "ablation_sensitivity",
]


class AblationSemantics(StrEnum):
    """Names what replaces an activation in a sensitivity arm."""

    RESAMPLE = "resample"
    MEAN = "mean"
    ZERO = "zero"


class AblationMetric(StrEnum):
    """Names the two behavioral scales every core arm reports."""

    LOGIT_DIFFERENCE = "logit-difference"
    PROBABILITY = "probability"


type _CellKey = tuple[AblationSemantics, AblationMetric]

_CORE_GRID: Final[tuple[_CellKey, ...]] = tuple(
    (semantics, metric)
    for semantics in (AblationSemantics.RESAMPLE, AblationSemantics.MEAN)
    for metric in AblationMetric
)
_ZERO_GRID: Final[tuple[_CellKey, ...]] = tuple(
    (AblationSemantics.ZERO, metric) for metric in AblationMetric
)
_ORDER: Final[tuple[_CellKey, ...]] = (*_CORE_GRID, *_ZERO_GRID)


class AblationSensitivityCell(FrozenValue):
    """One conclusion under a declared ablation semantics and behavioral metric.

    Parameters
    ----------
    semantics : AblationSemantics
        What replaced the ablated activation.
    metric : AblationMetric
        The behavioral scale on which the value was measured.
    value : float
        The finite measured value.
    conclusion : bool
        The conclusion drawn from that value under the caller's declared rule.

    Raises
    ------
    TypeError
        If a declaration or the conclusion has the wrong type, or the value is not
        numeric.
    ValueError
        If the value is not finite.
    """

    __slots__ = ("_conclusion", "_metric", "_semantics", "_value")

    _noun = "ablation sensitivity cell"

    def __init__(
        self,
        semantics: AblationSemantics,
        metric: AblationMetric,
        value: float,
        *,
        conclusion: bool,
    ) -> None:
        if not isinstance(semantics, AblationSemantics):
            msg = f"an ablation semantics is not one: {semantics!r}"
            raise TypeError(msg)

        if not isinstance(metric, AblationMetric):
            msg = f"an ablation metric is not one: {metric!r}"
            raise TypeError(msg)

        if not isinstance(conclusion, bool):
            msg = f"an ablation conclusion is not a bool: {conclusion!r}"
            raise TypeError(msg)

        object.__setattr__(self, "_semantics", semantics)
        object.__setattr__(self, "_metric", metric)
        object.__setattr__(
            self,
            "_value",
            ensure_finite("an ablation sensitivity value", value),
        )
        object.__setattr__(self, "_conclusion", conclusion)

    @property
    def semantics(self) -> AblationSemantics:
        """Return what replaced the ablated activation."""
        return self._semantics

    @property
    def metric(self) -> AblationMetric:
        """Return the behavioral scale on which the value was measured."""
        return self._metric

    @property
    def value(self) -> float:
        """Return the measured value."""
        return self._value

    @property
    def conclusion(self) -> bool:
        """Return the conclusion drawn from the measured value."""
        return self._conclusion

    @property
    def key(self) -> _CellKey:
        """Return the cell's coordinate in the sensitivity grid."""
        return (self._semantics, self._metric)

    def to_json(self) -> dict[str, JsonValue]:
        """Return the cell as JSON-ready report fields."""
        return {
            "semantics": self._semantics.value,
            "metric": self._metric.value,
            "value": self._value,
            "conclusion": self._conclusion,
        }

    def _identity(self) -> tuple[AblationSemantics, AblationMetric, float, bool]:
        """Return the fields this cell is equal on."""
        return (self._semantics, self._metric, self._value, self._conclusion)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another cell records the same finding."""
        if type(other) is not AblationSensitivityCell:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the cell's finding."""
        return hash((AblationSensitivityCell, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this cell."""
        return (
            f"AblationSensitivityCell(AblationSemantics.{self._semantics.name}, "
            f"AblationMetric.{self._metric.name}, {self._value!r}, "
            f"conclusion={self._conclusion!r})"
        )


def _coordinates(keys: Iterable[_CellKey]) -> str:
    """Return sensitivity coordinates in canonical order for an error message."""
    held = frozenset(keys)

    return repr(
        tuple(
            (semantics.value, metric.value)
            for semantics, metric in _ORDER
            if (semantics, metric) in held
        )
    )


class AblationSensitivityReport(FrozenValue):
    """The complete core ablation-sensitivity grid and its data-driven conclusion.

    The core grid always crosses resample and mean ablation with logit-difference
    and probability metrics. Zero ablation can appear only as an explicit additional
    two-metric arm. The four core cell conclusions determine reporting: agreement
    yields that conclusion, while disagreement sets ``flip`` and leaves
    ``conclusion`` unavailable with status ``"unreported"``.

    Parameters
    ----------
    cells : Iterable[AblationSensitivityCell]
        Exactly the four core cells, optionally followed by both zero-ablation cells.

    Raises
    ------
    TypeError
        If a member is not an ablation sensitivity cell.
    ValueError
        If a coordinate repeats, the core grid is incomplete, or a zero-ablation arm
        does not carry both metrics.
    """

    __slots__ = ("_cells", "_conclusion", "_flip")

    _noun = "ablation sensitivity report"

    def __init__(self, cells: Iterable[AblationSensitivityCell]) -> None:
        indexed: dict[_CellKey, AblationSensitivityCell] = {}

        for cell in tuple(cells):
            if type(cell) is not AblationSensitivityCell:
                msg = f"an ablation sensitivity report contains cells, not {cell!r}"
                raise TypeError(msg)

            if cell.key in indexed:
                msg = (
                    "an ablation sensitivity report repeats the cell "
                    f"{(cell.semantics.value, cell.metric.value)!r}"
                )
                raise ValueError(msg)

            indexed[cell.key] = cell

        keys = frozenset(indexed)
        expected = frozenset(_CORE_GRID)

        if any(semantics is AblationSemantics.ZERO for semantics, _metric in keys):
            expected |= frozenset(_ZERO_GRID)

        if keys != expected:
            missing = expected - keys
            unexpected = keys - expected
            msg = (
                "an ablation sensitivity report requires the complete core two-by-two "
                f"grid and any complete zero arm; missing {_coordinates(missing)}, "
                f"unexpected {_coordinates(unexpected)}"
            )
            raise ValueError(msg)

        ordered = tuple(indexed[key] for key in _ORDER if key in indexed)
        conclusions = frozenset(indexed[key].conclusion for key in _CORE_GRID)
        flip = len(conclusions) != 1

        object.__setattr__(self, "_cells", ordered)
        object.__setattr__(self, "_flip", flip)
        object.__setattr__(
            self,
            "_conclusion",
            None if flip else next(iter(conclusions)),
        )

    @property
    def cells(self) -> tuple[AblationSensitivityCell, ...]:
        """Return the cells in canonical semantics-then-metric order."""
        return self._cells

    @property
    def core(self) -> tuple[AblationSensitivityCell, ...]:
        """Return the four cells that determine the reported conclusion."""
        return self._cells[: len(_CORE_GRID)]

    @property
    def zero(self) -> tuple[AblationSensitivityCell, ...]:
        """Return the explicit off-distribution arm, empty when it was not requested."""
        return self._cells[len(_CORE_GRID) :]

    @property
    def default_intervention(self) -> AblationSemantics:
        """Return resampling, the default core intervention."""
        return AblationSemantics.RESAMPLE

    @property
    def flip(self) -> bool:
        """Return whether the four core cells disagree on their conclusion."""
        return self._flip

    @property
    def conclusion(self) -> bool | None:
        """Return the agreed conclusion, or None when the core cells disagree."""
        return self._conclusion

    @property
    def status(self) -> Literal["reported", "unreported"]:
        """Return whether the core grid licenses reporting its conclusion."""
        return "unreported" if self._flip else "reported"

    def to_json(self) -> dict[str, JsonValue]:
        """Return the complete grid and reporting convention as JSON-ready fields."""
        cells: list[JsonValue] = [cell.to_json() for cell in self._cells]

        return {
            "default_intervention": self.default_intervention.value,
            "cells": cells,
            "flip": self._flip,
            "status": self.status,
            "conclusion": self._conclusion,
        }

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another report carries the same grid."""
        if type(other) is not AblationSensitivityReport:
            return NotImplemented

        return self._cells == other._cells

    @override
    def __hash__(self) -> int:
        """Return a hash over the complete grid."""
        return hash((AblationSensitivityReport, self._cells))

    @override
    def __repr__(self) -> str:
        """Return the grid size, flip flag and report status."""
        return (
            f"AblationSensitivityReport(cells={len(self._cells)!r}, "
            f"flip={self._flip!r}, status={self.status!r})"
        )


def ablation_sensitivity(
    matched: Trace,
    measure: Callable[[Intervention, AblationMetric], float],
    conclude: Callable[[AblationMetric, float], bool],
    /,
    *,
    seed: int,
    include_zero: bool = False,
) -> AblationSensitivityReport:
    """Measure the complete sensitivity grid from one matched counterfactual trace.

    Resampling from ``matched`` is the default and first arm. Mean ablation reads the
    same trace. Zero ablation runs only when explicitly requested and never
    contributes to the flip decision.

    Parameters
    ----------
    matched : Trace
        Closed observations from the caller's matched counterfactual distribution.
    measure : Callable[[Intervention, AblationMetric], float]
        Runs one intervention-metric cell and returns its finite value.
    conclude : Callable[[AblationMetric, float], bool]
        Applies the caller's declared conclusion rule to a measured value.
    seed : int
        The seed of the resampling intervention.
    include_zero : bool
        Whether to append the explicit off-distribution zero-ablation arm.

    Returns
    -------
    AblationSensitivityReport
        The complete core grid, optional complete zero arm and reporting decision.

    Raises
    ------
    TypeError
        If a callback is not callable, the zero-arm flag is not Boolean, the matched
        value is not a trace, the seed is not an integer, a measurement is not
        numeric, or a conclusion is not Boolean.
    ValueError
        If the matched trace is open, the seed is negative, or a measurement is not
        finite.
    """
    if not callable(measure):
        msg = f"an ablation sensitivity measure is not callable: {measure!r}"
        raise TypeError(msg)

    if not callable(conclude):
        msg = f"an ablation sensitivity conclusion rule is not callable: {conclude!r}"
        raise TypeError(msg)

    if not isinstance(include_zero, bool):
        msg = f"whether to include zero ablation is not a bool: {include_zero!r}"
        raise TypeError(msg)

    arms: tuple[tuple[AblationSemantics, Intervention], ...] = (
        (AblationSemantics.RESAMPLE, Resample(matched, seed)),
        (AblationSemantics.MEAN, MeanFrom(matched)),
    )

    if include_zero:
        arms = (*arms, (AblationSemantics.ZERO, Zero()))

    cells: list[AblationSensitivityCell] = []

    for semantics, intervention in arms:
        for metric in AblationMetric:
            value = ensure_finite(
                f"the {semantics.value} {metric.value} sensitivity value",
                measure(intervention, metric),
            )
            cells.append(
                AblationSensitivityCell(
                    semantics,
                    metric,
                    value,
                    conclusion=conclude(metric, value),
                )
            )

    return AblationSensitivityReport(cells)
