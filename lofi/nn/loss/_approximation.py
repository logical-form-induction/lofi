"""Runtime checks for one resolved residual-mass bound."""

from math import sqrt
from typing import final

import torch
from torch import Tensor

from lofi._frozen import FrozenValue
from lofi.nn.forms import Estimate, WeightedTerms
from lofi.spec.approximation import residual_mass_bound

__all__ = ["ApproximationBound"]


@final
class ApproximationBound(FrozenValue):
    """A resolved bound that checks each measured truncation residual."""

    _noun = "approximation bound"
    __slots__ = ("_residual_mass",)

    def __init__(self, residual_mass: float, /) -> None:
        object.__setattr__(self, "_residual_mass", residual_mass_bound(residual_mass))

    @property
    def residual_mass(self) -> float:
        """Return the largest permitted residual probability mass."""
        return self._residual_mass

    def paired_member(self) -> ApproximationBound:
        """Return the per-frontier budget whose two-way union meets this bound."""
        return ApproximationBound(1.0 - sqrt(1.0 - self._residual_mass))

    def check_terms(
        self,
        weighted: WeightedTerms,
        /,
        *,
        component: str,
    ) -> WeightedTerms:
        """Return a term frontier after checking its truncation residual."""
        if weighted.estimate is Estimate.TRUNCATED:
            self.check_residual(weighted.log_residual.exp(), component=component)

        return weighted

    def check_residual(
        self,
        residual: Tensor,
        /,
        *,
        component: str,
    ) -> Tensor:
        """Return a scalar residual after checking it against this bound."""
        if residual.numel() != 1:
            msg = f"{component}: a truncation residual is scalar, not {residual.shape}"
            raise ValueError(msg)

        if not residual.is_floating_point():
            msg = f"{component}: a truncation residual must be floating point"
            raise TypeError(msg)

        actual = float(residual.detach().item())
        tolerance = 8.0 * torch.finfo(residual.dtype).eps

        if not torch.isfinite(residual).item() or not 0.0 <= actual <= 1.0:
            msg = f"{component}: a truncation residual is a probability, not {actual!r}"
            raise ValueError(msg)

        if actual > self._residual_mass + tolerance:
            msg = (
                f"{component}: measured residual mass {actual:g} exceeds the "
                f"declared bound {self._residual_mass:g}"
            )
            raise ValueError(msg)

        return residual
