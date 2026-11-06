"""Tensor-reduction interfaces for chart sweeps."""

from typing import TYPE_CHECKING, Protocol, runtime_checkable

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["LogWeightReduction"]


@runtime_checkable
class LogWeightReduction(Protocol):
    """Reduce alternative tensor-valued log weights."""

    def reduce(self, weights: Tensor, mask: Tensor, /, *, dim: int) -> Tensor:
        """Reduce the real alternatives along one axis.

        Parameters
        ----------
        weights : Tensor
            Log weights whose axis ``dim`` runs over the alternatives.
        mask : Tensor
            True where a position holds a real alternative, broadcastable against the
            weights.
        dim : int
            The axis to reduce.

        Returns
        -------
        Tensor
            The combined log weights, of the weights' shape with axis ``dim`` removed
            and floating point dtype.
        """

    @property
    def probabilistic(self) -> bool:
        """Return whether the reduction defines posterior marginals."""

    def mixing_weights(self, weights: Tensor, mask: Tensor, /, *, dim: int) -> Tensor:
        """Return log weights for mixing the retained alternatives.

        The result has the same shape as ``weights``. Log-sum-exp normalizes every
        real alternative, while maximum selects one largest alternative. A row with
        no finite real alternative remains entirely at negative infinity.
        """

    def validate_expected_counts(self) -> None:
        """Refuse a reduction whose partition gradient is not an expected count."""
