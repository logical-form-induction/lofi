"""Likelihood of graded judgments against the published root mass."""

from types import MappingProxyType
from typing import TYPE_CHECKING, final, override

import torch
from torch import nn
from torch.nn import functional

from lofi._checks import ensure_finite, ensure_name
from lofi._types import ReadoutId
from lofi.judgment.linking import latent_contribution
from lofi.nn.loss._errors import ImproperMassError
from lofi.nn.loss._ports import LossTerm
from lofi.nn.loss._readout import (
    MASS_AXES,
    log_complement,
    log_mass,
)

if TYPE_CHECKING:
    from collections.abc import Mapping

    from torch import Tensor

    from lofi.judgment._ports import Linking
    from lofi.judgment.grouped import _GroupedJudgmentBatch
    from lofi.judgment.ordinal import OrdinalHead
    from lofi.nn.loss._ports import PublishedOutputs
    from lofi.nn.loss._values import SpanReadout

__all__ = ["JudgmentLikelihoodObjective", "level_residuals"]


def level_residuals(log_probabilities: Tensor, responses: Tensor) -> Tensor:
    """Return each level's observed frequency less its mean predicted probability.

    Parameters
    ----------
    log_probabilities : Tensor
        The log probability of every level, one row per judgment.
    responses : Tensor
        The recorded level of each judgment, in row order.

    Returns
    -------
    Tensor
        One residual per level, summing to zero.
    """
    levels = log_probabilities.shape[1]
    predicted = log_probabilities.exp().mean(dim=0)
    observed = functional.one_hot(responses, levels).to(predicted.dtype).mean(dim=0)

    return observed - predicted


def _initial_parameters(
    linking: Linking, parameters: tuple[float, ...] | None
) -> tuple[float, ...]:
    """Return the values the linking's fitted parameters start at.

    Parameters
    ----------
    linking : Linking
        The linking whose fitted parameters are being started.
    parameters : tuple[float, ...] | None
        The starting values in domain order, or None to start at the closed form
        the linking declares over every fitted parameter.

    Raises
    ------
    TypeError
        If a starting value is not a number.
    ValueError
        If a starting value is not finite, if the vector is not as long as the
        domain's parameter list, or if no vector is given and the linking declares
        no closed form over its fitted parameters.
    """
    fitted = linking.domain.parameter_names

    if parameters is None and not fitted:
        return ()

    if parameters is None:
        for case in linking.special_cases:
            if len(case.parameters) == len(fitted):
                return case.parameters

        msg = (
            f"the linking fits {fitted!r} and declares no closed form over them, "
            "so their starting values must be given as parameters"
        )
        raise ValueError(msg)

    checked = tuple(
        ensure_finite(f"linking parameter {index}", value)
        for index, value in enumerate(parameters)
    )

    if len(checked) != len(fitted):
        msg = (
            f"the linking fits the parameters {fitted!r}, and {len(checked)} "
            "starting values were given"
        )
        raise ValueError(msg)

    return checked


@final
class _RootColumns:
    """The published mass pair at every item's root, as the columns a linking reads."""

    __slots__ = ("_readouts", "_rows")

    def __init__(self, readouts: Mapping[str, Tensor], rows: int) -> None:
        self._readouts: Mapping[str, Tensor] = MappingProxyType(dict(readouts))
        self._rows = rows

    @property
    def n_rows(self) -> int:
        """Return the number of items each column has a value for."""
        return self._rows

    @property
    def readouts(self) -> Mapping[str, Tensor]:
        """Return the log mass and, where the linking reads it, the log complement."""
        return self._readouts


@final
class JudgmentLikelihoodObjective(nn.Module):
    """Score graded judgments against the published root target mass.

    Each item's score is the mass the grammar's target set holds
    at its sentence-spanning cell, read from the mass pair published under the
    declared id. The linking maps that pair to a latent score, and the head
    gives the probability of each response level at that score. The value is
    the mean over the judgments of the negative log probability of the recorded
    level. The head's cutpoints and the linking's fitted parameters are this
    module's parameters, trained with the graph that publishes the mass.

    Parameters
    ----------
    readout : ReadoutId
        The identifier the root target mass pair is published under.
    linking : Linking
        The map from the mass pair to the latent score.
    head : OrdinalHead
        The cutpoints between adjacent response levels.
    parameters : tuple[float, ...] | None
        The initial values of the linking's fitted parameters, or None to start
        at the closed form the linking declares.
    name : str
        The name the term is logged and namespaced under.

    Raises
    ------
    TypeError
        If the id or the name is not a string, or a parameter is not a number.
    ValueError
        If the id or the name is empty, the parameter vector is not as long as
        the linking fits, no vector is given and the linking declares no closed
        form, or the linking reads a column beyond the mass pair.
    """

    def __init__(
        self,
        *,
        readout: ReadoutId,
        linking: Linking,
        head: OrdinalHead,
        parameters: tuple[float, ...] | None = None,
        name: str = "judgment",
    ) -> None:
        super().__init__()
        self._readout = ReadoutId(ensure_name("a readout id", readout))
        self._name = ensure_name("the objective's name", name)
        domain = linking.domain
        further = tuple(
            column for column in domain.readouts if column != domain.argument
        )

        if len(further) > 1:
            msg = (
                f"{self._name} hands its linking the mass pair, and this one reads "
                f"{domain.readouts!r}"
            )
            raise ValueError(msg)

        self._linking = linking
        self._argument = domain.argument
        self._complement = further[0] if further else None
        self.head = head
        self.parameters_of_linking = nn.Parameter(
            torch.tensor(_initial_parameters(linking, parameters), dtype=torch.float64)
        )

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
    def levels(self) -> int:
        """Return the number of ordered response levels."""
        return self.head.levels

    @property
    def linking(self) -> Linking:
        """Return the map from the mass pair to the latent score."""
        return self._linking

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

    def _latent(self, roots: Tensor) -> Tensor:
        """Return the latent score of every item through the linking.

        Parameters
        ----------
        roots : Tensor
            The log mass and log complement at each item's root, one row per item.
        """
        columns = {self._argument: roots[:, 0]}

        if self._complement is not None:
            columns[self._complement] = roots[:, 1]

        return latent_contribution(
            self._linking,
            _RootColumns(columns, roots.shape[0]),
            self.parameters_of_linking,
        )

    def evaluate(self, outputs: PublishedOutputs[_GroupedJudgmentBatch], /) -> LossTerm:
        """Return the mean negative log probability of the batch's judgments.

        Parameters
        ----------
        outputs : PublishedOutputs[_GroupedJudgmentBatch]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean over the responses, counted in them, reporting the mean log
            probability of the recorded levels as ``log_density``, each level's
            observed frequency less its mean predicted probability as
            ``level_residual_<k>``, and the lowest and highest score over every
            item of the batch and the difference between them as ``lowest``,
            ``highest`` and ``range``.

        Raises
        ------
        ValueError
            If the declared id was not published or is not a mass pair, if an item
            is empty or longer than the chart, or if a judgment records a level the
            head does not have.
        IndexError
            If a judgment names an item outside the batch.
        UncomputedCellError
            If the readout never computed an item's root cell.
        ImproperMassError
            If a published log probability at a root exceeds zero or is not a number.
        """
        pair = self._published(outputs.readouts)
        batch = outputs.batch
        lengths = batch.lengths
        design = batch.design
        like = pair.distribution.tensor

        roots = self._roots(pair, lengths)
        scores = roots[:, 0].detach().exp()
        lowest = scores.min()
        highest = scores.max()
        diagnostics = {"lowest": lowest, "highest": highest, "range": highest - lowest}

        if design.response_levels != self.head.levels:
            msg = (
                f"a head over {self.head.levels} levels was given a design over "
                f"{design.response_levels}"
            )

            raise ValueError(msg)

        responses = design.responses.to(device=like.device)
        log_probabilities = self.head.log_probabilities(
            batch.response_rows(self._latent(roots))
        )
        observed = log_probabilities.gather(1, responses.unsqueeze(1)).squeeze(1)
        residuals = level_residuals(log_probabilities.detach(), responses)
        diagnostics["log_density"] = observed.detach().mean()

        for level, residual in enumerate(residuals.unbind()):
            diagnostics[f"level_residual_{level}"] = residual

        return LossTerm(-observed.mean(), design.n_rows, diagnostics=diagnostics)

    @override
    def __repr__(self) -> str:
        """Return the name, the declared id, the head and the linking."""
        return (
            f"JudgmentLikelihoodObjective(name={self._name!r}, "
            f"readout={self._readout!r}, head={self.head!r}, "
            f"linking={self._linking!r})"
        )
