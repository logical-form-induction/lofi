"""Training-step inputs, outputs, and reports."""

from math import isfinite
from types import MappingProxyType
from typing import TYPE_CHECKING, override

from lofi._checks import NumericalFailureError, ensure_name, ensure_non_negative

if TYPE_CHECKING:
    from collections.abc import Iterable, Mapping

    from lofi.train._ports import StepDeclaration

__all__ = [
    "EvalResult",
    "StepEdge",
    "StepGraphError",
    "StepOutcome",
    "TrainResult",
    "edges",
]


class StepGraphError(Exception):
    """Reports an invalid step graph."""


def _finite(name: str, value: float) -> float:
    """Return one metric value as a float, refusing a value that is not finite.

    Raises
    ------
    TypeError
        If the value is not a number.
    NumericalFailureError
        If the value is infinite or not a number.
    """
    if isinstance(value, bool) or not isinstance(value, int | float):
        msg = f"metrics[{name!r}] is not a number: {value!r}"
        raise TypeError(msg)

    number = float(value)

    if not isfinite(number):
        quantity = f"metrics[{name!r}]"
        raise NumericalFailureError(quantity, number)

    return number


def _checked_metrics(metrics: Mapping[str, float]) -> Mapping[str, float]:
    """Return a validated read-only metric mapping.

    Parameters
    ----------
    metrics : Mapping[str, float]
        What the step reported, keyed by the name the component publishes it under.

    Returns
    -------
    Mapping[str, float]
        A read-only copy.

    Raises
    ------
    TypeError
        If a key is not a string or a value is not a number.
    ValueError
        If a key is empty.
    NumericalFailureError
        If a value is not finite.
    """
    checked = {
        ensure_name("a metric name", name): _finite(name, value)
        for name, value in metrics.items()
    }

    return MappingProxyType(checked)


def _ordered(metrics: Mapping[str, float]) -> tuple[tuple[str, float], ...]:
    """Return a metric mapping as a tuple of pairs, in name order.

    Parameters
    ----------
    metrics : Mapping[str, float]
        The numbers a result reports.

    Returns
    -------
    tuple[tuple[str, float], ...]
        One entry per metric, ordered by name.
    """
    return tuple(sorted(metrics.items()))


class TrainResult:
    """Records a training-step result.

    Parameters
    ----------
    step : int
        The number of optimizer steps the run had taken when this was reported.
    items : int
        How many items the step consumed.
    dropped : int
        How many items the step declined to consume.
    metrics : Mapping[str, float]
        The numbers the step reported, keyed by the name the component that computed
        each one publishes it under.

    Raises
    ------
    TypeError
        If a count is not an integer or a metric is not a number.
    ValueError
        If a count is negative.
    NumericalFailureError
        If a metric is not finite.
    """

    __slots__ = ("_dropped", "_items", "_metrics", "_step")

    def __init__(
        self,
        step: int,
        items: int,
        dropped: int = 0,
        metrics: Mapping[str, float] = MappingProxyType({}),
    ) -> None:
        self._step = ensure_non_negative("step", step)
        self._items = ensure_non_negative("items", items)
        self._dropped = ensure_non_negative("dropped", dropped)
        self._metrics = _checked_metrics(metrics)

    @property
    def step(self) -> int:
        """Return the optimizer-step count."""
        return self._step

    @property
    def items(self) -> int:
        """Return the consumed-item count."""
        return self._items

    @property
    def dropped(self) -> int:
        """Return the dropped-item count."""
        return self._dropped

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return metrics by name."""
        return self._metrics

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another result reports the same numbers at the same step."""
        if type(other) is not TrainResult:
            return NotImplemented

        return (
            self._step == other._step
            and self._items == other._items
            and self._dropped == other._dropped
            and dict(self._metrics) == dict(other._metrics)
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the step, the counts and the metrics."""
        return hash(
            (
                TrainResult,
                self._step,
                self._items,
                self._dropped,
                _ordered(self._metrics),
            )
        )

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"TrainResult(step={self._step}, items={self._items}, "
            f"dropped={self._dropped}, metrics={dict(self._metrics)!r})"
        )


class EvalResult:
    """Records an evaluation-step result.

    Parameters
    ----------
    items : int
        How many items were scored.
    dropped : int
        How many items were not scored.
    metrics : Mapping[str, float]
        The numbers the evaluation reported, keyed by the name the metric that computed
        each one publishes it under.

    Raises
    ------
    TypeError
        If a count is not an integer or a metric is not a number.
    ValueError
        If a count is negative.
    NumericalFailureError
        If a metric is not finite.
    """

    __slots__ = ("_dropped", "_items", "_metrics")

    def __init__(
        self,
        items: int,
        dropped: int = 0,
        metrics: Mapping[str, float] = MappingProxyType({}),
    ) -> None:
        self._items = ensure_non_negative("items", items)
        self._dropped = ensure_non_negative("dropped", dropped)
        self._metrics = _checked_metrics(metrics)

    @property
    def items(self) -> int:
        """Return the scored-item count."""
        return self._items

    @property
    def dropped(self) -> int:
        """Return the dropped-item count."""
        return self._dropped

    @property
    def metrics(self) -> Mapping[str, float]:
        """Return metrics by name."""
        return self._metrics

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another result reports the same numbers over as many items."""
        if type(other) is not EvalResult:
            return NotImplemented

        return (
            self._items == other._items
            and self._dropped == other._dropped
            and dict(self._metrics) == dict(other._metrics)
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the counts and the metrics."""
        return hash((EvalResult, self._items, self._dropped, _ordered(self._metrics)))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this result."""
        return (
            f"EvalResult(items={self._items}, dropped={self._dropped}, "
            f"metrics={dict(self._metrics)!r})"
        )


class StepOutcome[OutputsT, ResultT]:
    """Records a step outcome.

    Parameters
    ----------
    outputs : OutputsT
        What the step published, in the type the step's family declares.
    result : ResultT
        What the step reported, as a ``TrainResult`` or an ``EvalResult``.
    """

    __slots__ = ("_outputs", "_result")

    def __init__(self, outputs: OutputsT, result: ResultT) -> None:
        self._outputs = outputs
        self._result = result

    @property
    def outputs(self) -> OutputsT:
        """Return step outputs."""
        return self._outputs

    @property
    def result(self) -> ResultT:
        """Return the step result."""
        return self._result

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this outcome."""
        return f"StepOutcome({self._outputs!r}, {self._result!r})"


class StepEdge:
    """A step dependency.

    Parameters
    ----------
    producer : str
        The step that publishes the value.
    value : str
        The name the value is published and read under.
    consumer : str
        The step that reads it.

    Raises
    ------
    TypeError
        If a field is not a string.
    ValueError
        If a field is empty, or if a step is named as its own producer.
    """

    __slots__ = ("_consumer", "_producer", "_value")

    def __init__(self, producer: str, value: str, consumer: str) -> None:
        self._producer = ensure_name("producer", producer)
        self._value = ensure_name("value", value)
        self._consumer = ensure_name("consumer", consumer)

        if self._producer == self._consumer:
            msg = f"the step {producer!r} reads {value!r} from itself"

            raise ValueError(msg)

    @property
    def producer(self) -> str:
        """Return the producer step."""
        return self._producer

    @property
    def value(self) -> str:
        """Return the value name."""
        return self._value

    @property
    def consumer(self) -> str:
        """Return the consumer step."""
        return self._consumer

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another edge joins the same two steps over the same value."""
        if type(other) is not StepEdge:
            return NotImplemented

        return (
            self._producer == other._producer
            and self._value == other._value
            and self._consumer == other._consumer
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the two steps and the value."""
        return hash((StepEdge, self._producer, self._value, self._consumer))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this edge."""
        return f"StepEdge({self._producer!r}, {self._value!r}, {self._consumer!r})"


def _publishers(steps: Iterable[StepDeclaration]) -> dict[str, str]:
    """Return the step that publishes each value.

    Parameters
    ----------
    steps : Iterable[StepDeclaration]
        The steps of the graph.

    Returns
    -------
    dict[str, str]
        The publishing step, keyed by the value it publishes.

    Raises
    ------
    StepGraphError
        If two steps share a name or publish the same value.
    """
    published: dict[str, str] = {}
    seen: set[str] = set()

    for step in steps:
        if step.name in seen:
            msg = f"two steps are named {step.name!r}"

            raise StepGraphError(msg)

        seen.add(step.name)

        for value in sorted(step.produces):
            if value in published:
                msg = f"{published[value]!r} and {step.name!r} both publish {value!r}"

                raise StepGraphError(msg)

            published[value] = step.name

    return published


def edges(
    steps: Iterable[StepDeclaration],
    available: frozenset[str] = frozenset(),
) -> tuple[StepEdge, ...]:
    """Return step-graph dependencies.

    Parameters
    ----------
    steps : Iterable[StepDeclaration]
        The steps of the graph.
    available : frozenset[str]
        The values that enter the graph from outside.

    Returns
    -------
    tuple[StepEdge, ...]
        The edges, ordered by consumer and then by value.

    Raises
    ------
    StepGraphError
        If two steps share a name, if two steps publish one value, or if a step requires
        a value nothing publishes and nothing supplies.
    """
    declared = tuple(steps)
    published = _publishers(declared)
    built: list[StepEdge] = []

    for step in sorted(declared, key=lambda declaration: declaration.name):
        for value in sorted(step.requires):
            if value in available:
                continue

            if value not in published:
                msg = f"{step.name!r} requires {value!r} and nothing publishes it"

                raise StepGraphError(msg)

            built.append(StepEdge(published[value], value, step.name))

    return tuple(built)
