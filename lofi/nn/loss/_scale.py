"""Chart-cell vector-norm regularization."""

from typing import TYPE_CHECKING, Final, override

import torch

from lofi._checks import ensure_finite, ensure_name
from lofi.nn.loss._errors import MissingOutputError
from lofi.nn.loss._ports import LossTerm
from lofi.registry import ComponentPlugin
from lofi.spec.losses.chart_scale import (
    ChartScaleLossSettings,
    ChartScaleLossSpec,
    ScalePenaltyName,
)

if TYPE_CHECKING:
    from torch import Tensor

    from lofi.nn.loss._ports import LossComponentBindings, PublishedOutputs

__all__ = ["ChartScaleObjective", "ScalePenaltyName"]


def _checked_target(target: float) -> float:
    """Return the target norm.

    Parameters
    ----------
    target : float
        The Euclidean norm the regularizer pulls a cell toward.

    Returns
    -------
    float
        The target, widened to float.

    Raises
    ------
    TypeError
        If the target is not a number.
    ValueError
        If the target is not finite, or is not positive.
    """
    checked = ensure_finite("the target norm", target)

    if checked <= 0.0:
        msg = f"the target norm is a positive length, not {target}"
        raise ValueError(msg)

    return checked


def _live(mask: Tensor | None, tensor: Tensor) -> Tensor | None:
    """Return which positions of a per-cell quantity hold data.

    Parameters
    ----------
    mask : Tensor | None
        The cells' mask over their outermost axes, or None where every
        position holds data.
    tensor : Tensor
        The per-cell quantity the mask is being read against.

    Returns
    -------
    Tensor | None
        A boolean tensor of the quantity's shape, or None where there was no mask.
    """
    if mask is None:
        return None

    widened = mask.reshape(tuple(mask.shape) + (1,) * (tensor.ndim - mask.ndim))

    return widened.expand_as(tensor)


class ChartScaleObjective[BatchT]:
    """Penalizes deviations from a target chart-cell norm.

    Parameters
    ----------
    target : float
        The Euclidean norm each cell is pulled toward.
    over : str
        The axis a cell's vector runs along.
    name : str
        The name the term is logged and namespaced under.
    penalty : ScalePenaltyName
        Form of the departure penalty.

    Raises
    ------
    TypeError
        If the target is not a number, or a name is not a string.
    ValueError
        If the target is not finite or not positive, or if a name is empty.
    """

    __slots__ = ("_name", "_over", "_penalty", "_squared", "_target")

    def __init__(
        self,
        *,
        target: float,
        over: str,
        name: str,
        penalty: ScalePenaltyName,
    ) -> None:
        self._target = _checked_target(target)
        self._over = ensure_name("the cell vector's axis", over)
        self._name = ensure_name("the objective's name", name)
        self._penalty = penalty
        self._squared = self._target * self._target

    @property
    def name(self) -> str:
        """Return the name this objective is logged and namespaced under."""
        return self._name

    @property
    def requires(self) -> frozenset[str]:
        """Return the published outputs this objective reads, which are the cells."""
        return frozenset({"cells"})

    @property
    def target(self) -> float:
        """Return the Euclidean norm each cell is pulled toward."""
        return self._target

    @property
    def over(self) -> str:
        """Return the axis a cell's vector runs along."""
        return self._over

    @property
    def penalty(self) -> ScalePenaltyName:
        """Return the form of the departure penalty."""
        return self._penalty

    def evaluate(self, outputs: PublishedOutputs[BatchT], /) -> LossTerm:
        """Return how far the step's cells sit from the target norm.

        Parameters
        ----------
        outputs : PublishedOutputs[BatchT]
            What the graph published for the step.

        Returns
        -------
        LossTerm
            The mean squared departure over the live cells, counted in cells, reporting
            their root mean square norm as ``norm``.

        Raises
        ------
        MissingOutputError
            If the step published no cells.
        ValueError
            If the configured axis is absent or covered by the mask.
        """
        cells = outputs.cells

        if cells is None:
            raise MissingOutputError(
                objective=self._name,
                missing=self.requires,
                published=frozenset(),
            )

        dim = cells.axis(self._over)
        mask = cells.mask

        if mask is not None and dim < mask.ndim:
            msg = (
                f"the axis {self._over!r} is covered by a mask over "
                f"{cells.mask_axes}, so it addresses the batch rather than a "
                "cell's features"
            )
            raise ValueError(msg)

        squares = cells.tensor.pow(2).sum(dim=dim)
        norms = torch.linalg.vector_norm(cells.tensor, dim=dim)

        if self._penalty is ScalePenaltyName.SQUARED_RATIO:
            penalty = (squares / self._squared - 1.0).pow(2)
        elif self._penalty is ScalePenaltyName.SQUARED_DISTANCE:
            penalty = (norms - self._target).pow(2)
        else:
            penalty = (norms - self._target).abs()
        live = _live(mask, penalty)

        if live is None:
            count = penalty.numel()
        else:
            count = int(live.sum())
            penalty = penalty.where(live, penalty.new_zeros(()))
            squares = squares.where(live, squares.new_zeros(()))

        if count == 0:
            return LossTerm.inapplicable(cells.tensor)

        norm = (squares.detach().sum() / count).sqrt()

        return LossTerm(penalty.sum() / count, count, diagnostics={"norm": norm})

    @override
    def __repr__(self) -> str:
        """Return the name, the target norm and the axis a cell runs along."""
        return (
            f"ChartScaleObjective(name={self._name!r}, target={self._target!r}, "
            f"over={self._over!r}, penalty={self._penalty!r})"
        )


def _build[BatchT](
    settings: ChartScaleLossSettings, context: LossComponentBindings, /
) -> ChartScaleObjective[BatchT]:
    """Build the regularizer pulling each chart cell toward the declared norm.

    Parameters
    ----------
    settings : ChartScaleLossSettings
        The target norm.
    context : LossComponentBindings
        The declared name; the regularizer reads none of the built components.
    """
    return ChartScaleObjective(
        target=settings.target,
        over="width",
        name=context.name,
        penalty=settings.penalty,
    )


PLUGIN: Final = ComponentPlugin(spec=ChartScaleLossSpec, build=_build)
"""The chart-scale regularizer's plugin."""
