"""Registered similarity kernels between learned choice rows."""

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

import torch
from torch.nn import functional

from lofi._frozen import FrozenValue
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from torch import Tensor

__all__ = ["COSINE", "RBF", "SimilarityKernel", "similarity_kernel"]


@runtime_checkable
class SimilarityKernel(Protocol):
    """Score how far two choice rows unify, as a degree in the unit interval."""

    def similarity(
        self, left: Tensor, right: Tensor, /, *, temperature: float
    ) -> Tensor:
        """Return a scalar unification degree at a positive temperature."""
        ...


@final
class _Cosine(FrozenValue):
    """Rescale cosine agreement onto the unit interval and sharpen it."""

    _noun = "cosine similarity kernel"
    __slots__ = ()

    def similarity(
        self, left: Tensor, right: Tensor, /, *, temperature: float
    ) -> Tensor:
        cosine = functional.cosine_similarity(left, right, dim=0)
        return ((cosine + 1.0) / 2.0).clamp(0.0, 1.0).pow(1.0 / temperature)


@final
class _Rbf(FrozenValue):
    """Decay exponentially in the mean squared distance between rows."""

    _noun = "RBF similarity kernel"
    __slots__ = ()

    def similarity(
        self, left: Tensor, right: Tensor, /, *, temperature: float
    ) -> Tensor:
        distance = (left - right).square().mean()
        return torch.exp(-distance / temperature)


COSINE: Final[SimilarityKernel] = _Cosine()
"""Cosine agreement rescaled to the unit interval."""

RBF: Final[SimilarityKernel] = _Rbf()
"""A radial basis function of mean squared distance."""


def similarity_kernel(name: str, /) -> SimilarityKernel:
    """Return the similarity kernel installed under ``name``.

    Raises
    ------
    RegistryError
        If no installed ``lofi.similarity_kernel`` entry is named ``name``, or
        the entry names something other than a similarity kernel.
    """
    member = Registry[object]("similarity_kernel").get(name)

    if not isinstance(member, SimilarityKernel):
        msg = (
            f"the similarity_kernel entry {name!r} names {member!r}, "
            "not a similarity kernel"
        )
        raise RegistryError(msg, family="similarity_kernel")

    return member
