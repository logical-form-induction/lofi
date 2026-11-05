"""The installed features of an ordered pair of child representations."""

from typing import TYPE_CHECKING, Final, Protocol, final, runtime_checkable

from lofi._frozen import FrozenValue
from lofi.registry import Registry, RegistryError

if TYPE_CHECKING:
    from torch import Tensor

__all__ = [
    "DIFFERENCE",
    "LEFT",
    "PRODUCT",
    "RIGHT",
    "PairFeature",
    "pair_feature",
]


@runtime_checkable
class PairFeature(Protocol):
    """Derive one width-preserving feature from an ordered child pair."""

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        """Return the feature, of the children's own shape."""
        ...


@final
class _Left(FrozenValue):
    """The left child itself."""

    _noun = "left-child feature"
    __slots__ = ()

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        del right

        return left


@final
class _Right(FrozenValue):
    """The right child itself."""

    _noun = "right-child feature"
    __slots__ = ()

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        del left

        return right


@final
class _Difference(FrozenValue):
    """The left child less the right."""

    _noun = "child-difference feature"
    __slots__ = ()

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        return left - right


@final
class _Product(FrozenValue):
    """The elementwise product of the children."""

    _noun = "child-product feature"
    __slots__ = ()

    def __call__(self, left: Tensor, right: Tensor, /) -> Tensor:
        return left * right


LEFT: Final[PairFeature] = _Left()
"""The installed left-child feature."""

RIGHT: Final[PairFeature] = _Right()
"""The installed right-child feature."""

DIFFERENCE: Final[PairFeature] = _Difference()
"""The installed child-difference feature."""

PRODUCT: Final[PairFeature] = _Product()
"""The installed child-product feature."""


def pair_feature(name: str, /) -> PairFeature:
    """Return the child-pair feature installed under one name.

    Raises
    ------
    RegistryError
        If no ``lofi.pair_feature`` entry holds the name, or the entry names
        something other than a pair feature.
    """
    member = Registry[object]("pair_feature").get(name)

    if not isinstance(member, PairFeature):
        msg = f"the pair_feature entry {name!r} names {member!r}, not a pair feature"
        raise RegistryError(msg, family="pair_feature")

    return member
