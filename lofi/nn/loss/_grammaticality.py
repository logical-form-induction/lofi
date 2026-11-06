"""Binary cross-entropy of grammaticality labels against the published root mass."""

import math
from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_finite, ensure_name
from lofi._types import ReadoutId
from lofi.nn.loss._errors import ImproperMassError
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import MASS_AXES, log_complement, log_mass
from lofi.registry import ComponentPlugin
from lofi.spec.losses.grammaticality import (
    GrammaticalityLossSettings,
    GrammaticalityLossSpec,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.nn.loss._ports import (
        LabeledItems,
        LossComponentBindings,
        PublishedOutputs,
    )
    from lofi.nn.loss._values import SpanReadout

__all__ = ["GrammaticalityObjective"]


def _checked_noise(noise: float) -> float:
    """Return a label-noise rate under which a label still says something.

    Parameters
    ----------
    noise : float
        The probability that an observed label was flipped.

    Raises
    ------
    TypeError
        If the rate is not a number.
    ValueError
        If the rate is not finite, is negative, or is at least one half.
    """
    checked = ensure_finite("the label-noise rate", noise)

    if not 0.0 <= checked < 0.5:
        msg = (
            f"the label-noise rate is {noise}, and a label flipped at least half "
            "the time says nothing about the sentence"
        )
        raise ValueError(msg)

    return checked


class GrammaticalityObjective:
    """Score binary grammaticality labels against the published root target mass.

    The grammaticality score of an item is the mass the grammar's target set holds
    at the item's sentence-spanning cell, read from the mass pair published under
    the declared id. The value is the mean over the labeled items of the negative
    log probability of the observed label, where a positive label's probability is
    the score and a negative label's the complement. Under a nonzero label-noise
    rate the observed label is taken to have been flipped with that probability, so
    the probability of the observed label mixes the score and its complement at the
    declared rate. The range of the score over every item of the batch, labeled or
    not, is reported as a diagnostic.

    Parameters
    ----------
    readout : ReadoutId
        The identifier the root target mass pair is published under.
    noise : float
        The probability that an observed label was flipped.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the id or the name is not a string, or the noise rate is not a number.
    ValueError
        If the id or the name is empty, or the noise rate is not finite, is negative,
        or is at least one half.
    """

    __slots__ = ("_name", "_noise", "_readout")

    def __init__(
        self,
        *,
        readout: ReadoutId,
        noise: float,
        name: str,
    ) -> None:
        self._readout = ReadoutId(ensure_name("a readout id", readout))
        self._noise = _checked_noise(noise)
        self._name = ensure_name("the objective's name", name)

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads."""
        return frozenset({"batch", "readouts"})

    @property
    def readout(self) -> ReadoutId:
        """Return the identifier the root target mass is read under."""
        return self._readout

    @property
    def noise(self) -> float:
        """Return the probability that an observed label was flipped."""
        return self._noise

    def _published(self, readouts: Mapping[ReadoutId, SpanReadout]) -> SpanReadout:
        """Return the mass pair published under the declared id.

        Parameters
        ----------
        readouts : Mapping[ReadoutId, SpanReadout]
            What every readout published, keyed by id.

        Raises
        ------
        ValueError
            If nothing was published under the declared id, or if what was published
            there is not a mass pair.
        """
        pair = readouts.get(self._readout)

        if pair is None:
            offered = ", ".join(sorted(readouts)) or "nothing"
            msg = (
                f"{self._name} reads the root mass {self._readout!r}, and the step "
                f"published {offered}"
            )
            raise ValueError(msg)

        if pair.axes != MASS_AXES:
            msg = (
                f"{self._name} reads a mass pair over {MASS_AXES}, and "
                f"{self._readout!r} is over {pair.axes}"
            )
            raise ValueError(msg)

        return pair

    def _roots(self, pair: SpanReadout, lengths: tuple[int, ...]) -> Tensor:
        """Return the mass pair at every item's sentence-spanning cell, in one gather.

        Parameters
        ----------
        pair : SpanReadout
            The published root target mass pair.
        lengths : tuple[int, ...]
            The number of words of each item, in item order.

        Returns
        -------
        Tensor
            One log mass and log complement per item, attached to the published
            tensor's graph.

        Raises
        ------
        ValueError
            If an item is empty or longer than the chart.
        UncomputedCellError
            If the readout never computed an item's root cell.
        ImproperMassError
            If a published log probability at a root exceeds zero or is not a
            number.
        """
        cells = pair.gather([(item, 0, length) for item, length in enumerate(lengths)])
        held = torch.stack((log_mass(cells), log_complement(cells)), dim=-1)
        proper = held <= 0.0

        if not bool(proper.all()):
            item, outcome = (int(index) for index in proper.logical_not().nonzero()[0])
            raise ImproperMassError(
                readout=self._readout,
                item=item,
                span=(0, lengths[item]),
                value=float(held[item, outcome]),
            )

        return held

    def _observed(self, roots: Tensor, labels: tuple[bool, ...]) -> Tensor:
        """Return the log probability of each observed label.

        Parameters
        ----------
        roots : Tensor
            The log mass and log complement at each labeled item's root, one row
            per label.
        labels : tuple[bool, ...]
            Whether each item is labeled grammatical, in row order.

        Returns
        -------
        Tensor
            One log probability per label, under the declared label-noise rate.
        """
        positive = torch.tensor(labels, device=roots.device)
        agreeing = torch.where(positive, roots[:, 0], roots[:, 1])

        if self._noise == 0.0:
            return agreeing

        flipped = torch.where(positive, roots[:, 1], roots[:, 0])

        return torch.logaddexp(
            agreeing + math.log1p(-self._noise), flipped + math.log(self._noise)
        )

    def evaluate(self, outputs: PublishedOutputs[LabeledItems], /) -> LossTerm:
        """Return the mean negative log probability of the batch's labels.

        Parameters
        ----------
        outputs : PublishedOutputs[LabeledItems]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the labeled items, counted in them, reporting the lowest
            and highest score over every item of the batch and the difference
            between them as ``lowest``, ``highest`` and ``range``. A batch with no
            labeled item is the zero term, still reporting the range where the batch
            has an item.

        Raises
        ------
        ValueError
            If the declared id was not published or is not a mass pair, if the batch
            labels another number of items than it has lengths for, or if an item is
            empty or longer than the chart.
        UncomputedCellError
            If the readout never computed an item's root cell.
        ImproperMassError
            If a published log probability at a root exceeds zero or is not a number.
        """
        pair = self._published(outputs.readouts)
        batch = outputs.batch
        lengths = batch.lengths
        labels = batch.labels
        like = pair.distribution.tensor

        if len(labels) != len(lengths):
            msg = (
                f"the batch labels {len(labels)} items and gives lengths for "
                f"{len(lengths)}"
            )
            raise ValueError(msg)

        if not lengths:
            return LossTerm.inapplicable(like)

        roots = self._roots(pair, lengths)
        scores = roots[:, 0].detach().exp()
        lowest = scores.min()
        highest = scores.max()
        diagnostics = {"lowest": lowest, "highest": highest, "range": highest - lowest}
        labeled = [item for item, label in enumerate(labels) if label is not None]

        if not labeled:
            return LossTerm(
                LossTerm.inapplicable(like).value, 0, diagnostics=diagnostics
            )

        observed = self._observed(
            roots[torch.tensor(labeled, device=roots.device)],
            tuple(bool(labels[item]) for item in labeled),
        )

        return LossTerm(-observed.mean(), len(labeled), diagnostics=diagnostics)

    @override
    def __repr__(self) -> str:
        """Return the name, the declared id and the noise rate."""
        return (
            f"GrammaticalityObjective(name={self._name!r}, "
            f"readout={self._readout!r}, noise={self._noise!r})"
        )


def _build(
    settings: GrammaticalityLossSettings, context: LossComponentBindings, /
) -> GrammaticalityObjective:
    """Build grammaticality scoring against the declared root mass publication.

    Parameters
    ----------
    settings : GrammaticalityLossSettings
        The publication's identifier and the label-noise rate.
    context : LossComponentBindings
        The declared name; the objective reads none of the built components.
    """
    return GrammaticalityObjective(
        readout=ReadoutId(settings.readout),
        noise=settings.noise,
        name=context.name,
    )


PLUGIN: Final = ComponentPlugin(spec=GrammaticalityLossSpec, build=_build)
"""The grammaticality objective's plugin."""
