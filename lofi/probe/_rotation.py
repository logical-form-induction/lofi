"""A shared differentiable parameterization of an orthogonal rotation."""

from typing import TYPE_CHECKING

import torch

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["OrthogonalOptimizer"]


class OrthogonalOptimizer:
    """A seeded orthogonal matrix optimized through a Cayley transform."""

    __slots__ = ("_basis", "_generator", "_optimizer")

    def __init__(
        self, like: Tensor, learning_rate: float, draws: torch.Generator
    ) -> None:
        if not isinstance(like, torch.Tensor) or not like.is_floating_point():
            msg = "a differentiable site reading is a floating point tensor"
            raise TypeError(msg)
        if like.ndim == 0:
            msg = "a differentiable site reading has a width axis"
            raise ValueError(msg)
        if isinstance(learning_rate, bool) or not isinstance(
            learning_rate, int | float
        ):
            msg = (
                "an orthogonal optimizer's learning rate is numeric, not "
                f"{learning_rate!r}"
            )
            raise TypeError(msg)
        if not 0.0 < float(learning_rate) < float("inf"):
            msg = (
                "an orthogonal optimizer's learning rate is positive: "
                f"{learning_rate!r}"
            )
            raise ValueError(msg)
        if not isinstance(draws, torch.Generator):
            msg = f"an orthogonal optimizer draws from a torch Generator, not {draws!r}"
            raise TypeError(msg)

        side = int(like.shape[-1])
        basis, _ = torch.linalg.qr(
            torch.randn(side, side, generator=draws, dtype=torch.float64)
        )
        self._basis = basis.to(device=like.device, dtype=like.dtype)
        self._generator = torch.zeros(
            side, side, device=like.device, dtype=like.dtype, requires_grad=True
        )
        self._optimizer = torch.optim.Adam([self._generator], lr=float(learning_rate))

    @property
    def parameter(self) -> Tensor:
        """Return the unconstrained generator optimized by this primitive."""
        return self._generator

    def matrix(self) -> Tensor:
        """Return the current orthogonal matrix with its gradient connection."""
        skew = self._generator - self._generator.T
        identity = torch.eye(int(skew.shape[0]), device=skew.device, dtype=skew.dtype)
        return self._basis @ torch.linalg.solve(identity - skew, identity + skew)

    def gradients_of(
        self, loss: Tensor, additional: tuple[Tensor, ...] = ()
    ) -> tuple[Tensor, ...]:
        """Return finite gradients for the rotation and additional parameters."""
        if loss.ndim != 0 or not bool(torch.isfinite(loss)):
            msg = f"a search loss is not a finite scalar: {loss.detach()!r}"
            raise ValueError(msg)
        if not loss.requires_grad:
            msg = "no gradient reaches the search loss"
            raise ValueError(msg)

        gradients = torch.autograd.grad(
            loss, (self._generator, *additional), allow_unused=True
        )
        names = ("rotation", *("additional parameter" for _ in additional))
        held: list[Tensor] = []

        for name, gradient in zip(names, gradients, strict=True):
            if gradient is None:
                msg = (
                    f"the scores do not depend on the {name}; no gradient reaches "
                    f"the {name}"
                )
                raise ValueError(msg)
            if not bool(torch.isfinite(gradient).all()):
                msg = f"the {name} gradient is not finite"
                raise ValueError(msg)
            held.append(gradient)

        return tuple(held)

    def step(self, gradient: Tensor) -> None:
        """Apply one supplied generator gradient."""
        if tuple(gradient.shape) != tuple(self._generator.shape):
            msg = (
                f"a rotation gradient has shape {tuple(gradient.shape)}, not "
                f"{tuple(self._generator.shape)}"
            )
            raise ValueError(msg)
        self._optimizer.zero_grad(set_to_none=True)
        self._generator.grad = gradient
        self._optimizer.step()

    def finish(self) -> Tensor:
        """Return the detached optimized orthogonal matrix."""
        return self.matrix().detach()
