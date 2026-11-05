"""Component-resolution errors."""

from typing import Final

from lofi._checks import ensure_name

__all__ = ["RegistryError", "TagCollisionError", "UnknownTagError"]

_COLLIDING: Final = 2
"""The minimum distribution count for a tag collision."""


class RegistryError(ValueError):
    """A refused component resolution.

    Parameters
    ----------
    message : str
        What was refused.
    family : str
        The family whose registry refused it.

    Raises
    ------
    TypeError
        If the family is not a string.
    ValueError
        If the family is empty.
    """

    __slots__ = ("family",)

    def __init__(self, message: str, *, family: str) -> None:
        self.family = ensure_name("family", family)

        super().__init__(message)


class UnknownTagError(RegistryError):
    """Reports an unknown registry tag.

    Parameters
    ----------
    family : str
        The family that was asked.
    tag : str
        The tag that was asked for.
    available : tuple[str, ...]
        Every tag the family does hold, in the order a message lists them.
    """

    __slots__ = ("available", "tag")

    def __init__(self, *, family: str, tag: str, available: tuple[str, ...]) -> None:
        self.tag = tag
        self.available: tuple[str, ...] = available

        offered = ", ".join(available) or "nothing"
        msg = f"no {family} is registered as {tag!r}; the family holds {offered}"

        super().__init__(msg, family=family)


class TagCollisionError(RegistryError):
    """Reports duplicate tag declarations within a family.

    Parameters
    ----------
    family : str
        The family both declared in.
    tag : str
        The tag both claimed.
    distributions : tuple[str, ...]
        What claimed it, each a distribution or, where an entry point has no
        distribution, its value, in the order a message names them.

    Raises
    ------
    ValueError
        If fewer than two claimants are named.
    """

    __slots__ = ("distributions", "tag")

    def __init__(
        self, *, family: str, tag: str, distributions: tuple[str, ...]
    ) -> None:
        if len(distributions) < _COLLIDING:
            msg = f"a collision needs two claimants, and {distributions} names fewer"

            raise ValueError(msg)

        self.tag = tag
        self.distributions: tuple[str, ...] = distributions

        claimants = " and ".join(distributions)
        message = (
            f"{claimants} each declare {tag!r} as a {family}, so which one a "
            f"configuration names is decided by installation order"
        )

        super().__init__(message, family=family)
