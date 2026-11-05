"""Divergence specifications for coherence objectives."""

from typing import Final, override

from lofi._checks import ensure_finite

__all__ = [
    "KL",
    "CrossEntropy",
    "Divergence",
    "FenchelYoung",
    "KullbackLeibler",
    "resolved",
]


class KullbackLeibler:
    """Computes forward KL divergence from parent to children."""

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other value is this same divergence.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not KullbackLeibler:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the hash of this divergence."""
        return hash(KullbackLeibler)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this divergence."""
        return "KullbackLeibler()"


class CrossEntropy:
    """Computes child-distribution cross-entropy under the parent."""

    __slots__ = ()

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other value is this same divergence.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not CrossEntropy:
            return NotImplemented

        return True

    @override
    def __hash__(self) -> int:
        """Return the hash of this divergence."""
        return hash(CrossEntropy)

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this divergence."""
        return "CrossEntropy()"


class FenchelYoung:
    """Computes the alpha-indexed Fenchel-Young loss.

    Parameters
    ----------
    alpha : float
        Which entropy the loss is built on.

    Raises
    ------
    TypeError
        If alpha is not a number.
    ValueError
        If alpha is not finite or is below one, which leaves the family the loss is
        defined over.
    """

    __slots__ = ("_alpha",)

    def __init__(self, alpha: float = 1.5) -> None:
        checked = ensure_finite("alpha", alpha)

        if checked < 1.0:
            msg = f"alpha indexes the entropy family from one upward, not {alpha}"
            raise ValueError(msg)

        self._alpha = checked

    @property
    def alpha(self) -> float:
        """Return which entropy the loss is built on."""
        return self._alpha

    @override
    def __eq__(self, other: object) -> bool:
        """Return whether the other value is this loss at the same alpha.

        Parameters
        ----------
        other : object
            The value to compare against.
        """
        if type(other) is not FenchelYoung:
            return NotImplemented

        return self._alpha == other._alpha

    @override
    def __hash__(self) -> int:
        """Return the hash of the loss and its alpha."""
        return hash((FenchelYoung, self._alpha))

    @override
    def __repr__(self) -> str:
        """Return the constructor call that rebuilds this divergence."""
        return f"FenchelYoung({self._alpha!r})"


type Divergence = KullbackLeibler | CrossEntropy | FenchelYoung
"""Defines a parent-child divergence specification."""

KL: Final[KullbackLeibler] = KullbackLeibler()
"""Forward KL divergence as the default."""


def resolved(divergence: Divergence, /) -> Divergence:
    """Return the comparison a divergence specification actually makes.

    The Fenchel-Young loss built on the Shannon entropy is the KL divergence, so a
    loss written at alpha one resolves to KL and is infinite where KL is. Every other
    specification resolves to itself.

    Parameters
    ----------
    divergence : Divergence
        The comparison as it was written.

    Returns
    -------
    Divergence
        The comparison it computes, which is the same value except where a
        Fenchel-Young loss at alpha one resolves to KL.
    """
    if isinstance(divergence, FenchelYoung) and divergence.alpha == 1.0:
        return KL

    return divergence
