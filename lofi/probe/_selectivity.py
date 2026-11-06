"""Classifier probes over spans, scored against a control task of the same spans.

A probe that labels spans well may be reading the representation or may be
fitting the labels on its own, and raw accuracy does not tell the two apart. A
control task assigns every span key a label drawn at random, so nothing about
the representation predicts it, and the same probe is fitted to it at the same
capacity. Selectivity is the gap between the two held-out accuracies, and it is
what a classifier probe reports.
"""

from math import sqrt
from random import Random
from types import MappingProxyType
from typing import TYPE_CHECKING, Self, override

import torch
from torch.nn.functional import cross_entropy, relu

from lofi._checks import (
    ensure_finite,
    ensure_name,
    ensure_non_negative,
    ensure_positive,
)

if TYPE_CHECKING:
    from collections.abc import Mapping, Sequence

    from torch import Tensor

__all__ = [
    "ClassifierProbe",
    "ControlTask",
    "ProbeCapacity",
    "ProbeResult",
    "SpanInventory",
]


class SpanInventory:
    """The spans a classifier probe reads, with their keys and their labels.

    Parameters
    ----------
    vectors : Tensor
        One vector per span, spans along the outer axis and coordinates along the
        inner one.
    keys : Sequence[str]
        The identity of each span, in the order the vectors run. A control task
        draws one label per distinct key, so two spans sharing a key are labeled
        alike under it.
    labels : Sequence[int]
        The real label of each span, in the same order.
    classes : int
        How many labels the task chooses between. Every label lies below it, and a
        control task draws over the same range.

    Raises
    ------
    TypeError
        If the vectors are not a floating point tensor, a key is not a string, or a
        label or the class count is not an integer.
    ValueError
        If the vectors are not one per span, no span or no coordinate is given, the
        keys or the labels are not one per span, the class count is below two, or a
        label lies outside it.
    """

    __slots__ = ("_classes", "_keys", "_labels", "_vectors")

    def __init__(
        self,
        vectors: Tensor,
        keys: Sequence[str],
        labels: Sequence[int],
        *,
        classes: int,
    ) -> None:
        if not isinstance(vectors, torch.Tensor) or not vectors.is_floating_point():
            msg = f"a span inventory holds floating point vectors, not {vectors!r}"
            raise TypeError(msg)

        if vectors.ndim != 2:
            msg = (
                "a span inventory holds one vector per span, not a tensor of shape "
                f"{tuple(vectors.shape)}"
            )
            raise ValueError(msg)

        self._vectors = vectors.detach().to(torch.float64)
        self._keys = tuple(ensure_name("a span key", key) for key in keys)
        self._labels = tuple(
            ensure_non_negative("a span label", label) for label in labels
        )
        self._classes = ensure_positive("a span inventory's class count", classes)

        spans = int(vectors.shape[0])

        if len(self._keys) != spans or len(self._labels) != spans:
            msg = (
                f"{spans} span vectors carry {len(self._keys)} keys and "
                f"{len(self._labels)} labels"
            )
            raise ValueError(msg)

        if spans == 0:
            msg = "a span inventory holds no span"
            raise ValueError(msg)

        if vectors.shape[1] == 0:
            msg = "a span vector of no coordinates leaves a probe nothing to read"
            raise ValueError(msg)

        if self._classes < 2:
            msg = f"a task of one class classifies nothing: {classes!r}"
            raise ValueError(msg)

        for label in self._labels:
            if label >= self._classes:
                msg = f"a span is labeled {label!r} among {self._classes} classes"
                raise ValueError(msg)

    @property
    def vectors(self) -> Tensor:
        """Return one vector per span, spans outermost."""
        return self._vectors

    @property
    def keys(self) -> tuple[str, ...]:
        """Return the identity of each span, in the order the vectors run."""
        return self._keys

    @property
    def labels(self) -> tuple[int, ...]:
        """Return the real label of each span, in the order the vectors run."""
        return self._labels

    @property
    def classes(self) -> int:
        """Return how many labels the task chooses between."""
        return self._classes

    @property
    def features(self) -> int:
        """Return how many coordinates a span vector has."""
        return int(self._vectors.shape[1])

    def relabeled(self, labels: Sequence[int]) -> SpanInventory:
        """Return the same spans under other labels.

        Parameters
        ----------
        labels : Sequence[int]
            The label of each span, one per span, in the order the vectors run.

        Returns
        -------
        SpanInventory
            The same vectors, keys and class count, labeled as given.

        Raises
        ------
        ValueError
            If the labels are not one per span, or one lies outside the class count.
        """
        return SpanInventory(self._vectors, self._keys, labels, classes=self._classes)

    def __len__(self) -> int:
        """Return how many spans the inventory holds."""
        return len(self._keys)

    @override
    def __repr__(self) -> str:
        """Return how many spans, coordinates and classes the inventory holds."""
        return (
            f"SpanInventory({len(self._keys)} spans, features={self.features!r}, "
            f"classes={self._classes!r})"
        )


class ControlTask:
    """Labels drawn at random for each span key, under a declared seed.

    The label of a key is drawn from its key and the seed alone, so the same key
    is labeled alike in every inventory the task is applied to, and a probe that
    scores well on it is fitting the keys rather than reading the representation.

    Parameters
    ----------
    seed : int
        The seed every label is drawn under.

    Raises
    ------
    TypeError
        If the seed is not an integer.
    ValueError
        If the seed is negative.
    """

    __slots__ = ("_seed",)

    def __init__(self, *, seed: int) -> None:
        self._seed = ensure_non_negative("a control-task seed", seed)

    @property
    def seed(self) -> int:
        """Return the seed every label is drawn under."""
        return self._seed

    def label(self, key: str, classes: int) -> int:
        """Return the label this task draws for one span key.

        Parameters
        ----------
        key : str
            The identity of the span.
        classes : int
            How many labels the task draws between.

        Returns
        -------
        int
            The label, below the class count, drawn from the key and the seed.

        Raises
        ------
        TypeError
            If the key is not a string or the class count is not an integer.
        ValueError
            If the key is empty or the class count is not positive.
        """
        drawn = ensure_name("a span key", key)
        over = ensure_positive("a control task's class count", classes)

        return Random(f"{self._seed}\n{drawn}").randrange(over)

    def assignment(self, inventory: SpanInventory) -> Mapping[str, int]:
        """Return the label this task draws for each key of an inventory.

        Parameters
        ----------
        inventory : SpanInventory
            The spans whose keys are labeled.

        Returns
        -------
        Mapping[str, int]
            One label per distinct key, in ascending key order, as a read-only view.
        """
        return MappingProxyType(
            {
                key: self.label(key, inventory.classes)
                for key in sorted(set(inventory.keys))
            }
        )

    def over(self, inventory: SpanInventory) -> SpanInventory:
        """Return the same spans labeled by this control task.

        Parameters
        ----------
        inventory : SpanInventory
            The spans to relabel, whose vectors, keys and class count are kept.

        Returns
        -------
        SpanInventory
            The same inventory under the drawn labels.
        """
        return inventory.relabeled(
            [self.label(key, inventory.classes) for key in inventory.keys]
        )

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another control task draws under the same seed.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not ControlTask:
            return NotImplemented

        return self._seed == other._seed

    @override
    def __hash__(self) -> int:
        """Return a hash over the seed the task draws under."""
        return hash((ControlTask, self._seed))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this control task."""
        return f"ControlTask(seed={self._seed!r})"


class ProbeCapacity:
    """What a classifier probe may fit with, and how it is fitted.

    Parameters
    ----------
    hidden : int
        The width of the probe's hidden layer, or zero for a probe that scores its
        classes straight off the span vector.
    epochs : int
        How many full-batch steps the fit takes.
    learning_rate : float
        The optimizer's step size.
    seed : int
        The seed the starting weights are drawn under.

    Raises
    ------
    TypeError
        If a count or the seed is not an integer, or the rate is not a number.
    ValueError
        If a count or the seed is negative, the epoch count is zero, or the rate is
        not positive and finite.
    """

    __slots__ = ("_epochs", "_hidden", "_learning_rate", "_seed")

    def __init__(
        self, *, hidden: int, epochs: int, learning_rate: float, seed: int
    ) -> None:
        self._hidden = ensure_non_negative("a probe's hidden width", hidden)
        self._epochs = ensure_positive("a probe's epoch count", epochs)
        self._learning_rate = ensure_finite("a learning rate", learning_rate)
        self._seed = ensure_non_negative("a probe seed", seed)

        if self._learning_rate <= 0.0:
            msg = f"a learning rate is not positive: {learning_rate!r}"
            raise ValueError(msg)

    @property
    def hidden(self) -> int:
        """Return the width of the probe's hidden layer, zero where it has none."""
        return self._hidden

    @property
    def epochs(self) -> int:
        """Return how many full-batch steps the fit takes."""
        return self._epochs

    @property
    def learning_rate(self) -> float:
        """Return the optimizer's step size."""
        return self._learning_rate

    @property
    def seed(self) -> int:
        """Return the seed the starting weights are drawn under."""
        return self._seed

    def parameters(self, features: int, classes: int) -> int:
        """Return how many weights a probe of this capacity fits.

        Parameters
        ----------
        features : int
            How many coordinates a span vector has.
        classes : int
            How many labels the probe chooses between.

        Returns
        -------
        int
            The weights and biases of every layer, counted together.

        Raises
        ------
        TypeError
            If either count is not an integer.
        ValueError
            If either count is not positive.
        """
        wide = ensure_positive("a span vector's coordinate count", features)
        many = ensure_positive("a probe's class count", classes)

        if self._hidden == 0:
            return wide * many + many

        return wide * self._hidden + self._hidden + self._hidden * many + many

    def _identity(self) -> tuple[int, int, float, int]:
        """Return the four fields a capacity is equal on."""
        return (self._hidden, self._epochs, self._learning_rate, self._seed)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another capacity states the same four fields.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not ProbeCapacity:
            return NotImplemented

        return self._identity() == other._identity()

    @override
    def __hash__(self) -> int:
        """Return a hash over the four fields the capacity is equal on."""
        return hash((ProbeCapacity, self._identity()))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this capacity."""
        return (
            f"ProbeCapacity(hidden={self._hidden!r}, epochs={self._epochs!r}, "
            f"learning_rate={self._learning_rate!r}, seed={self._seed!r})"
        )


class ProbeResult:
    """A classifier probe's held-out accuracy beside its control-task accuracy.

    Parameters
    ----------
    accuracy : float
        The share of held-out spans the probe labeled correctly on the real task.
    control_accuracy : float
        The share it labeled correctly on the control task, fitted at the same
        capacity over the same spans.
    control_seed : int
        The seed the control task drew its labels under.
    capacity : ProbeCapacity
        What both fits were run at.
    parameters : int
        How many weights each fit moved.

    Raises
    ------
    TypeError
        If an accuracy is not a number, the seed or the weight count is not an
        integer, or the capacity is not one.
    ValueError
        If an accuracy is not finite or lies outside zero to one, the seed is
        negative, or the weight count is not positive.
    """

    __slots__ = (
        "_accuracy",
        "_capacity",
        "_control_accuracy",
        "_control_seed",
        "_parameters",
    )

    def __init__(
        self,
        accuracy: float,
        control_accuracy: float,
        *,
        control_seed: int,
        capacity: ProbeCapacity,
        parameters: int,
    ) -> None:
        if type(capacity) is not ProbeCapacity:
            msg = f"a probe result reports a ProbeCapacity, not {capacity!r}"
            raise TypeError(msg)

        self._accuracy = ensure_finite("a probe accuracy", accuracy)
        self._control_accuracy = ensure_finite(
            "a control-task accuracy", control_accuracy
        )
        self._control_seed = ensure_non_negative("a control-task seed", control_seed)
        self._capacity = capacity
        self._parameters = ensure_positive("a probe's weight count", parameters)

        for name, value in (
            ("a probe accuracy", self._accuracy),
            ("a control-task accuracy", self._control_accuracy),
        ):
            if not 0.0 <= value <= 1.0:
                msg = f"{name} is not a proportion: {value!r}"
                raise ValueError(msg)

    @property
    def accuracy(self) -> float:
        """Return the share of held-out spans labeled correctly on the real task."""
        return self._accuracy

    @property
    def control_accuracy(self) -> float:
        """Return the share labeled correctly on the control task."""
        return self._control_accuracy

    @property
    def selectivity(self) -> float:
        """Return how far the real task ran ahead of the control task."""
        return self._accuracy - self._control_accuracy

    @property
    def control_seed(self) -> int:
        """Return the seed the control task drew its labels under."""
        return self._control_seed

    @property
    def capacity(self) -> ProbeCapacity:
        """Return what both fits were run at."""
        return self._capacity

    @property
    def parameters(self) -> int:
        """Return how many weights each fit moved."""
        return self._parameters

    @override
    def __repr__(self) -> str:
        """Return the two accuracies, the gap between them, and the seed."""
        return (
            f"ProbeResult(accuracy={self._accuracy!r}, "
            f"control_accuracy={self._control_accuracy!r}, "
            f"selectivity={self.selectivity!r}, "
            f"control_seed={self._control_seed!r})"
        )


def _layer(inputs: int, outputs: int, draws: torch.Generator) -> tuple[Tensor, Tensor]:
    """Return one layer's starting weight and bias, both fitted.

    Parameters
    ----------
    inputs : int
        How many coordinates the layer reads.
    outputs : int
        How many it writes.
    draws : torch.Generator
        The generator the weight is drawn from.

    Returns
    -------
    tuple[Tensor, Tensor]
        A weight drawn at a scale the input count sets, and a bias at zero.
    """
    weight = torch.randn(inputs, outputs, generator=draws, dtype=torch.float64) / sqrt(
        inputs
    )
    bias = torch.zeros(outputs, dtype=torch.float64)

    return weight.requires_grad_(), bias.requires_grad_()


class _Probe:
    """The weights one probe fits, and the class scores they give a span vector.

    Parameters
    ----------
    capacity : ProbeCapacity
        The hidden width the weights take and the seed they are drawn under.
    features : int
        How many coordinates a span vector has.
    classes : int
        How many labels the probe chooses between.
    """

    __slots__ = ("_hidden", "_output", "_weights")

    def __init__(self, capacity: ProbeCapacity, features: int, classes: int) -> None:
        draws = torch.Generator().manual_seed(capacity.seed)

        if capacity.hidden == 0:
            self._hidden: tuple[Tensor, Tensor] | None = None
            self._output = _layer(features, classes, draws)
        else:
            self._hidden = _layer(features, capacity.hidden, draws)
            self._output = _layer(capacity.hidden, classes, draws)

        held = [] if self._hidden is None else list(self._hidden)
        self._weights = [*held, *self._output]

    @property
    def weights(self) -> list[Tensor]:
        """Return every weight and bias the fit moves."""
        return self._weights

    def scores(self, vectors: Tensor) -> Tensor:
        """Return the score this probe gives each class of each span.

        Parameters
        ----------
        vectors : Tensor
            One span vector per row.

        Returns
        -------
        Tensor
            One row per span over the classes, higher scoring the more likely.
        """
        weight, bias = self._output

        if self._hidden is None:
            return vectors @ weight + bias

        inner, offset = self._hidden

        return relu(vectors @ inner + offset) @ weight + bias


def _fitted(capacity: ProbeCapacity, fitting: SpanInventory) -> _Probe:
    """Return a probe of the given capacity fitted on the given spans.

    Parameters
    ----------
    capacity : ProbeCapacity
        What the probe fits with, and how.
    fitting : SpanInventory
        The spans the probe fits on.

    Returns
    -------
    _Probe
        The weights after the capacity's full-batch steps, drawn under its seed.
    """
    probe = _Probe(capacity, fitting.features, fitting.classes)
    optimizer = torch.optim.Adam(probe.weights, lr=capacity.learning_rate)
    targets = torch.tensor(fitting.labels)

    for _ in range(capacity.epochs):
        optimizer.zero_grad()
        cross_entropy(probe.scores(fitting.vectors), targets).backward()
        optimizer.step()

    return probe


def _held_out_accuracy(
    capacity: ProbeCapacity, fitting: SpanInventory, held_out: SpanInventory
) -> float:
    """Return the share of held-out spans a probe fitted on the other spans labels.

    Parameters
    ----------
    capacity : ProbeCapacity
        What the probe fits with, and how.
    fitting : SpanInventory
        The spans the probe fits on.
    held_out : SpanInventory
        The spans it is scored on.

    Returns
    -------
    float
        The proportion of held-out spans whose highest-scoring class is the label.
    """
    probe = _fitted(capacity, fitting)

    with torch.no_grad():
        predicted = probe.scores(held_out.vectors).argmax(dim=-1)

    matched = int((predicted == torch.tensor(held_out.labels)).sum())

    return matched / len(held_out)


class ClassifierProbe:
    """A span classifier reported against a control task of the same spans.

    The real task and the control task are fitted by one procedure at one
    capacity, from the same starting weights, over the same span vectors and
    keys. The two accuracies are therefore comparable, and their gap is the
    selectivity the result reports.

    Parameters
    ----------
    capacity : ProbeCapacity
        What both fits run at.
    control : ControlTask
        The labels the real task's accuracy is measured against.

    Raises
    ------
    TypeError
        If the capacity is not one, or the control task is not one.
    """

    __slots__ = ("_capacity", "_control")

    def __init__(self, capacity: ProbeCapacity, control: ControlTask) -> None:
        if type(capacity) is not ProbeCapacity:
            msg = f"a probe fits at a ProbeCapacity, not {capacity!r}"
            raise TypeError(msg)

        if type(control) is not ControlTask:
            msg = f"a probe is scored against a ControlTask, not {control!r}"
            raise TypeError(msg)

        self._capacity = capacity
        self._control = control

    @property
    def capacity(self) -> ProbeCapacity:
        """Return what both fits run at."""
        return self._capacity

    @property
    def control(self) -> ControlTask:
        """Return the labels the real task's accuracy is measured against."""
        return self._control

    def under(self, control: ControlTask) -> Self:
        """Return the same probe scored against another control task.

        Parameters
        ----------
        control : ControlTask
            The control task the new probe draws its labels from.

        Returns
        -------
        Self
            A probe of the same capacity under the given control task.
        """
        return type(self)(self._capacity, control)

    def run(self, fitting: SpanInventory, held_out: SpanInventory) -> ProbeResult:
        """Return what this probe reached on both tasks and the gap between them.

        Parameters
        ----------
        fitting : SpanInventory
            The spans both fits run on.
        held_out : SpanInventory
            The spans both fits are scored on, held out from the ones fitting saw.

        Returns
        -------
        ProbeResult
            The real-task accuracy, the control-task accuracy, the seed the control
            labels were drawn under, the capacity both fits ran at, and how many
            weights each moved.

        Raises
        ------
        TypeError
            If either inventory is not one.
        ValueError
            If the two inventories disagree about how many coordinates a span vector
            has or how many classes the task has, since a probe cannot then be fitted
            on one and scored on the other.
        """
        for name, inventory in (("fitted", fitting), ("scored", held_out)):
            if type(inventory) is not SpanInventory:
                msg = f"a probe is {name} on a SpanInventory, not {inventory!r}"
                raise TypeError(msg)

        if fitting.features != held_out.features:
            msg = (
                f"a probe fitted on span vectors of {fitting.features} coordinates "
                f"cannot be scored on ones of {held_out.features}"
            )
            raise ValueError(msg)

        if fitting.classes != held_out.classes:
            msg = (
                f"a probe fitted over {fitting.classes} classes cannot be scored "
                f"over {held_out.classes}"
            )
            raise ValueError(msg)

        accuracy = _held_out_accuracy(self._capacity, fitting, held_out)
        control_accuracy = _held_out_accuracy(
            self._capacity, self._control.over(fitting), self._control.over(held_out)
        )

        return ProbeResult(
            accuracy,
            control_accuracy,
            control_seed=self._control.seed,
            capacity=self._capacity,
            parameters=self._capacity.parameters(fitting.features, fitting.classes),
        )

    @override
    def __repr__(self) -> str:
        """Return the capacity the probe fits at and the control task it runs."""
        return f"ClassifierProbe({self._capacity!r}, {self._control!r})"
