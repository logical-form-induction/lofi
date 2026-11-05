"""The description length of a probing task's labels under an online code.

Accuracy at one capacity says how well a probe of that size reads a
representation, and nothing about how much of the reading was the probe's own
work. The online code prices the labels instead of the probe: the labels are
transmitted in blocks, each block charged under a probe fitted on the labels
already sent, so a probe's weights are never transmitted and never charged.
What comes out is a number of bits that a probe of any size can be compared on,
reported beside the accuracy and the selectivity rather than in place of them.
"""

from itertools import pairwise
from math import log, log2
from typing import TYPE_CHECKING, Final, override

import torch
from torch.nn.functional import cross_entropy

from lofi._checks import ensure_finite, ensure_non_negative, ensure_positive
from lofi.probe._selectivity import (
    ProbeCapacity,
    SpanInventory,
    _fitted,
    _Probe,
)

if TYPE_CHECKING:
    from collections.abc import Sequence

    from lofi.probe._ports import Classifier, ProbeReport

__all__ = [
    "VOITA_BLOCKS",
    "BlockSchedule",
    "CodeBlock",
    "Codelength",
    "ProbeComplexity",
    "measure_complexity",
    "online_codelength",
    "uniform_codelength",
]


class BlockSchedule:
    """Where an online code closes one block of labels and opens the next.

    A schedule is stated as prefix fractions of the transmitted labels rather
    than as counts, so one schedule applies to sets of any size. The fractions
    ascend and the last is one, since every label is transmitted exactly once.

    Parameters
    ----------
    fractions : Sequence[float]
        The share of the labels transmitted by the end of each block, ascending,
        the last of them one.

    Raises
    ------
    TypeError
        If a fraction is not a number.
    ValueError
        If no fraction is given, a fraction is not finite, the fractions do not
        strictly ascend, the first is not positive, or the last is not one.
    """

    __slots__ = ("_fractions",)

    def __init__(self, fractions: Sequence[float]) -> None:
        self._fractions = tuple(
            ensure_finite("a block boundary", fraction) for fraction in fractions
        )

        if not self._fractions:
            msg = "a block schedule of no boundary transmits nothing"
            raise ValueError(msg)

        if self._fractions[0] <= 0.0:
            msg = (
                "a block schedule's first boundary is not positive: "
                f"{self._fractions[0]!r}"
            )
            raise ValueError(msg)

        if self._fractions[-1] != 1.0:
            msg = (
                "a block schedule's last boundary transmits every label, so it is "
                f"one, not {self._fractions[-1]!r}"
            )
            raise ValueError(msg)

        for earlier, later in pairwise(self._fractions):
            if earlier >= later:
                msg = (
                    f"a block boundary at {later!r} does not follow one at "
                    f"{earlier!r}, so the blocks have no order"
                )
                raise ValueError(msg)

    @property
    def fractions(self) -> tuple[float, ...]:
        """Return the share transmitted by the end of each block, ascending."""
        return self._fractions

    def boundaries(self, count: int) -> tuple[int, ...]:
        """Return where the blocks end when this many labels are transmitted.

        A fraction that falls below one label is rounded up to one, and two
        fractions that land on the same label give one boundary, so a schedule
        finer than the set is not a run of empty blocks.

        Parameters
        ----------
        count : int
            How many labels are transmitted.

        Returns
        -------
        tuple[int, ...]
            One past the last label of each block, ascending, the last of them
            the count.

        Raises
        ------
        TypeError
            If the count is not an integer.
        ValueError
            If the count is not positive.
        """
        many = ensure_positive("a transmitted label count", count)

        return tuple(sorted({max(1, round(f * many)) for f in self._fractions}))

    def __len__(self) -> int:
        """Return how many boundaries the schedule states."""
        return len(self._fractions)

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another schedule states the same boundaries.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not BlockSchedule:
            return NotImplemented

        return self._fractions == other._fractions

    @override
    def __hash__(self) -> int:
        """Return a hash over the boundaries the schedule states."""
        return hash((BlockSchedule, self._fractions))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this schedule."""
        return f"BlockSchedule({self._fractions!r})"


VOITA_BLOCKS: Final = BlockSchedule(
    (
        0.001,
        0.002,
        0.004,
        0.008,
        0.016,
        0.032,
        0.0625,
        0.125,
        0.25,
        0.5,
        1.0,
    )
)
"""Specifies the eleven doubling prefixes an online code is usually read over."""


class CodeBlock:
    """One block of labels and what transmitting it cost.

    Parameters
    ----------
    start : int
        The first label of the block, counting from the first transmitted.
    end : int
        One past its last label.
    bits : float
        What the block cost under the code that was current when it was sent.

    Raises
    ------
    TypeError
        If an endpoint is not an integer or the cost is not a number.
    ValueError
        If the start is negative, the block holds no label, or the cost is not
        finite and non-negative.
    """

    __slots__ = ("_bits", "_end", "_start")

    def __init__(self, start: int, end: int, bits: float) -> None:
        self._start = ensure_non_negative("a block's first label", start)
        self._end = ensure_positive("a block's last label", end)
        self._bits = ensure_finite("a block's cost", bits)

        if self._end <= self._start:
            msg = f"a block from {start!r} to {end!r} transmits no label"
            raise ValueError(msg)

        if self._bits < 0.0:
            msg = f"a block cost no bits to transmit: {bits!r}"
            raise ValueError(msg)

    @property
    def start(self) -> int:
        """Return the first label of the block."""
        return self._start

    @property
    def end(self) -> int:
        """Return one past the last label of the block."""
        return self._end

    @property
    def bits(self) -> float:
        """Return what the block cost."""
        return self._bits

    def __len__(self) -> int:
        """Return how many labels the block transmits."""
        return self._end - self._start

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether another block spans the same labels at the same cost.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not CodeBlock:
            return NotImplemented

        return (self._start, self._end, self._bits) == (
            other._start,
            other._end,
            other._bits,
        )

    @override
    def __hash__(self) -> int:
        """Return a hash over the block's endpoints and cost."""
        return hash((CodeBlock, self._start, self._end, self._bits))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this block."""
        return f"CodeBlock({self._start!r}, {self._end!r}, {self._bits!r})"


class Codelength:
    """What a set of labels cost under an online code, and what it would cost raw.

    Parameters
    ----------
    blocks : Sequence[CodeBlock]
        The blocks in the order they were transmitted, the first starting at
        zero and each starting where the one before it ended.
    uniform : float
        What the same labels cost under the uniform code over the classes, which
        is the code a reader who knows nothing about them has.
    schedule : BlockSchedule
        The schedule the blocks were cut at.

    Raises
    ------
    TypeError
        If a block is not one, the uniform cost is not a number, or the schedule
        is not one.
    ValueError
        If no block is given, the blocks do not run from zero without gap or
        overlap, or the uniform cost is not finite and positive.
    """

    __slots__ = ("_bits", "_blocks", "_schedule", "_uniform")

    def __init__(
        self, blocks: Sequence[CodeBlock], uniform: float, schedule: BlockSchedule
    ) -> None:
        if type(schedule) is not BlockSchedule:
            msg = f"a codelength records a BlockSchedule, not {schedule!r}"
            raise TypeError(msg)

        for block in blocks:
            if type(block) is not CodeBlock:
                msg = f"a codelength is made of CodeBlocks, not {block!r}"
                raise TypeError(msg)

        self._blocks = tuple(blocks)
        self._uniform = ensure_finite("a uniform codelength", uniform)
        self._schedule = schedule

        if not self._blocks:
            msg = "a codelength of no block transmits nothing"
            raise ValueError(msg)

        at = 0

        for block in self._blocks:
            if block.start != at:
                msg = (
                    f"a block starting at {block.start!r} follows one ending at "
                    f"{at!r}, so the labels between are transmitted twice or not "
                    "at all"
                )
                raise ValueError(msg)

            at = block.end

        if self._uniform <= 0.0:
            msg = f"a uniform codelength is not positive: {uniform!r}"
            raise ValueError(msg)

        self._bits = sum(block.bits for block in self._blocks)

    @property
    def bits(self) -> float:
        """Return what the labels cost under the online code, blocks together."""
        return self._bits

    @property
    def uniform(self) -> float:
        """Return what the same labels cost under the uniform code."""
        return self._uniform

    @property
    def blocks(self) -> tuple[CodeBlock, ...]:
        """Return the blocks in the order they were transmitted."""
        return self._blocks

    @property
    def schedule(self) -> BlockSchedule:
        """Return the schedule the blocks were cut at."""
        return self._schedule

    @property
    def labels(self) -> int:
        """Return how many labels were transmitted."""
        return self._blocks[-1].end

    @property
    def information(self) -> float:
        """Return the bits the representation saved against knowing nothing.

        This is negative where the probe transmitted the labels worse than the
        uniform code did, which is a representation the probe read nothing from
        and then paid to consult.
        """
        return self._uniform - self._bits

    @property
    def compression(self) -> float:
        """Return the uniform cost over the online cost."""
        return self._uniform / self._bits

    @override
    def __repr__(self) -> str:
        """Return the two costs, the block count and the schedule."""
        return (
            f"Codelength(bits={self._bits!r}, uniform={self._uniform!r}, "
            f"blocks={len(self._blocks)!r}, schedule={self._schedule!r})"
        )


def uniform_codelength(inventory: SpanInventory) -> float:
    """Return what an inventory's labels cost under the uniform code.

    Parameters
    ----------
    inventory : SpanInventory
        The spans whose labels are transmitted.

    Returns
    -------
    float
        One label's cost, the logarithm of the class count in base two, once per
        span.

    Raises
    ------
    TypeError
        If the inventory is not one.
    """
    if type(inventory) is not SpanInventory:
        msg = f"labels are transmitted out of a SpanInventory, not {inventory!r}"
        raise TypeError(msg)

    return len(inventory) * log2(inventory.classes)


def _bits_under(probe: _Probe, inventory: SpanInventory) -> float:
    """Return what an inventory's labels cost under a fitted probe's predictions.

    Parameters
    ----------
    probe : _Probe
        The fitted probe, which scores the classes of a span vector.
    inventory : SpanInventory
        The spans whose labels are transmitted.

    Returns
    -------
    float
        The summed negative log-probability of the labels, in bits.
    """
    with torch.no_grad():
        nats = cross_entropy(
            probe.scores(inventory.vectors),
            torch.tensor(inventory.labels),
            reduction="sum",
        ).item()

    return nats / log(2)


def _portion(inventory: SpanInventory, start: int, end: int) -> SpanInventory:
    """Return the spans an inventory holds between two positions.

    Parameters
    ----------
    inventory : SpanInventory
        The spans to cut.
    start : int
        The first span kept.
    end : int
        One past the last span kept.

    Returns
    -------
    SpanInventory
        The same vectors, keys, labels and class count over that range alone.
    """
    return SpanInventory(
        inventory.vectors[start:end],
        inventory.keys[start:end],
        inventory.labels[start:end],
        classes=inventory.classes,
    )


def online_codelength(
    inventory: SpanInventory,
    capacity: ProbeCapacity,
    *,
    schedule: BlockSchedule = VOITA_BLOCKS,
) -> Codelength:
    """Return what an inventory's labels cost under an online code.

    The sender and the receiver agree in advance on the span vectors, the order
    the spans are transmitted in, the class count, the block schedule and the
    fitting procedure, which is the capacity's width, seed, step size and epoch
    count. None of that is charged. The labels are then sent block by block.
    The first block is charged under the uniform code over the classes, since
    the receiver has seen no label yet. Every later block is charged the
    negative log-probability, in bits, that a probe fitted on exactly the labels
    already transmitted gives its labels.

    The probe's weights are therefore never transmitted. The receiver reruns
    the agreed fit on the labels already in hand and arrives at the same
    weights, so no term in the total counts a weight and no width is charged
    for. A family with more freedom pays for the freedom where it does damage,
    in the early blocks, where it fits a short prefix worse and so charges more
    for the labels that follow. That is the sense in which the total is free of
    probe complexity, and it is why two probes of different width are
    comparable on it while their accuracies at one capacity are not.

    Freedom that the labels reward is still rewarded, so the total is a
    statement about a representation and a probe family together and never
    about a representation alone. A family that cannot express the mapping pays
    for that in every block.

    Parameters
    ----------
    inventory : SpanInventory
        The spans whose labels are transmitted, in the order they are sent.
    capacity : ProbeCapacity
        The fit the sender and the receiver both run.
    schedule : BlockSchedule
        Where the blocks end.

    Returns
    -------
    Codelength
        The blocks, their total, what the same labels cost under the uniform
        code, and the schedule the blocks were cut at.

    Raises
    ------
    TypeError
        If the inventory, the capacity or the schedule is not one.
    """
    if type(capacity) is not ProbeCapacity:
        msg = f"an online code fits at a ProbeCapacity, not {capacity!r}"
        raise TypeError(msg)

    if type(schedule) is not BlockSchedule:
        msg = f"an online code cuts blocks at a BlockSchedule, not {schedule!r}"
        raise TypeError(msg)

    uniform = uniform_codelength(inventory)
    boundaries = schedule.boundaries(len(inventory))
    per_label = log2(inventory.classes)
    blocks = [CodeBlock(0, boundaries[0], boundaries[0] * per_label)]

    for start, end in pairwise(boundaries):
        probe = _fitted(capacity, _portion(inventory, 0, start))
        blocks.append(
            CodeBlock(start, end, _bits_under(probe, _portion(inventory, start, end)))
        )

    return Codelength(blocks, uniform, schedule)


class ProbeComplexity:
    """A probe's codelength beside the accuracies it was reported with.

    A codelength says how much of a task's labels a representation accounts for
    and says nothing about how often a probe of a stated size gets a held-out
    span right, which is the number a reader of a probing result also wants.
    The two travel together, so that neither is read as the other.

    Parameters
    ----------
    report : ProbeReport
        What the probe reached on the real task and on its control task.
    codelength : Codelength
        What the fitting spans' labels cost under the online code.

    Raises
    ------
    TypeError
        If the codelength is not one, or the report carries an accuracy, a
        control accuracy or a selectivity that is not a number.
    ValueError
        If either accuracy is not a proportion, or the selectivity is not the
        real-task accuracy less the control-task one.
    """

    __slots__ = ("_codelength", "_report")

    def __init__(self, report: ProbeReport, codelength: Codelength) -> None:
        if type(codelength) is not Codelength:
            msg = f"a probe's complexity reports a Codelength, not {codelength!r}"
            raise TypeError(msg)

        accuracy = ensure_finite("a probe report's accuracy", report.accuracy)
        control = ensure_finite(
            "a probe report's control accuracy", report.control_accuracy
        )
        gap = ensure_finite("a probe report's selectivity", report.selectivity)

        for name, value in (("accuracy", accuracy), ("control accuracy", control)):
            if not 0.0 <= value <= 1.0:
                msg = f"a probe report's {name} is not a proportion: {value!r}"
                raise ValueError(msg)

        if abs(gap - (accuracy - control)) > 1e-9:
            msg = (
                f"a probe report's selectivity of {gap!r} is not its accuracy of "
                f"{accuracy!r} less its control accuracy of {control!r}"
            )
            raise ValueError(msg)

        self._report = report
        self._codelength = codelength

    @property
    def report(self) -> ProbeReport:
        """Return what the probe reached on the real task and on the control."""
        return self._report

    @property
    def codelength(self) -> Codelength:
        """Return what the fitting spans' labels cost under the online code."""
        return self._codelength

    @property
    def accuracy(self) -> float:
        """Return the share of held-out spans labeled correctly on the real task."""
        return self._report.accuracy

    @property
    def control_accuracy(self) -> float:
        """Return the share labeled correctly on the control task."""
        return self._report.control_accuracy

    @property
    def selectivity(self) -> float:
        """Return how far the real task ran ahead of the control task."""
        return self._report.selectivity

    @property
    def description_length(self) -> float:
        """Return what the fitting spans' labels cost, in bits."""
        return self._codelength.bits

    @override
    def __repr__(self) -> str:
        """Return the accuracy, the selectivity and the description length."""
        return (
            f"ProbeComplexity(accuracy={self.accuracy!r}, "
            f"selectivity={self.selectivity!r}, "
            f"description_length={self.description_length!r})"
        )


def measure_complexity(
    probe: Classifier,
    fitting: SpanInventory,
    held_out: SpanInventory,
    *,
    schedule: BlockSchedule = VOITA_BLOCKS,
) -> ProbeComplexity:
    """Return a probe's accuracies and the codelength of the labels it fits on.

    Parameters
    ----------
    probe : Classifier
        The probe, which supplies the capacity the online code fits at.
    fitting : SpanInventory
        The spans the probe fits on, whose labels the online code transmits.
    held_out : SpanInventory
        The spans the probe is scored on.
    schedule : BlockSchedule
        Where the online code's blocks end.

    Returns
    -------
    ProbeComplexity
        The probe's report and the codelength, together.
    """
    return ProbeComplexity(
        probe.run(fitting, held_out),
        online_codelength(fitting, probe.capacity, schedule=schedule),
    )
